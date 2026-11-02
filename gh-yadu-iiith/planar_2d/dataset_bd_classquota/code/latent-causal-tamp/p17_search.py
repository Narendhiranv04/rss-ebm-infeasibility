"""P17 corrected finite rearrangement search (replaces p16_common.search for P17; P16 code untouched for historical replay).

SearchState = (object cells, robot q, last moved object if the no-consecutive rule is active, per-object move counts, depth). Between
actions the gripper is always open and free (attachment is part of a single relocation record, never a search state), so it is not a
key field. Equivalence key = (sorted occupancy, EXACT q bytes, last); dominance inside one key: an earlier entry with travel <= J and
counts <= counts elementwise (hence no fewer remaining moves per object and no smaller remaining depth) permits a superset of the
future actions at no greater cost. P16 omitted `last` from the key: a cheaper path ending with object X could prune a dearer path ending
with Y although only the latter may move X next (test_p17 last-mover counterexample)."""
from __future__ import annotations

import itertools
import time

import numpy as np

import p15_search as ps

V, CI, SI, U_PLAN, U_BUD = ps.V, ps.CI, ps.SI, ps.U_PLAN, ps.U_BUD
DECIDED = ps.DECIDED


def key(cells, q, last, rule=True):
    return (tuple(sorted(cells.items())), np.asarray(q, float).tobytes(), last if rule else None)


def dominated(entries, J, counts) -> bool:
    return any(J2 <= J and all(c2.get(o, 0) <= counts.get(o, 0) for o in set(c2) | set(counts)) for J2, c2 in entries)


def search(provider, order, movers, slots, start_cells, start_q, n_max=4, per_obj=2, node_cap=600, no_consecutive=True) -> dict:
    """Iterative deepening on relocations; DFS in guidance order (reorder only), travel branch-and-bound (admissible: costs >= 0) and
    sound dominance. Claims are restricted to the declared finite graph: MIN_MOVES requires every shallower pass fully enumerated with
    every applicable edge / goal decided; MIN_TRAVEL additionally the same at the found depth."""
    log = {"edges": {}, "goals": {}, "pruned_bnb": 0, "pruned_dominated": 0, "nodes": 0, "terminal_q_alternatives": 0, "node_cap_hit": False}
    t0, best, undecided = time.process_time(), {"plan": None}, set()

    def rec(kind, depth, st):
        log[kind][st] = log[kind].get(st, 0) + 1
        if st not in DECIDED:
            undecided.add(depth)

    def dfs(cells, q, counts, last, steps, recs, J, left, total, seen):
        if log["nodes"] >= node_cap:
            log["node_cap_hit"] = True
            return
        log["nodes"] += 1
        if best["plan"] is not None and best["plan"]["moves"] == total and J >= best["plan"]["travel"]:
            log["pruned_bnb"] += 1
            return
        k = key(cells, q, last, no_consecutive)
        if dominated(seen.get(k, []), J, counts):
            log["pruned_dominated"] += 1
            return
        seen.setdefault(k, []).append((J, dict(counts)))
        if left == 0:
            st, r, tj = provider.goal(cells, q)
            rec("goals", total, st)
            if st == V and (best["plan"] is None or J + tj < best["plan"]["travel"]):
                best["plan"] = {"moves": total, "travel": J + tj, "steps": steps, "records": recs + [r]}
                log.setdefault("anytime", []).append({"cpu_s": time.process_time() - t0, "nodes": log["nodes"], "moves": total, "travel": J + tj})
            return
        cands = [(o, d) for o in movers if counts.get(o, 0) < per_obj and not (no_consecutive and o == last) for d in slots(cells, o)]
        for o, d in order(cells, q, frozenset(x for x in counts if counts[x]), cands):
            if provider.truncated() or log["node_cap_hit"]:
                return
            outs = provider.edges(cells, q, o, d)
            for st, *_ in outs:
                rec("edges", total, st)
            ver = [x for x in outs if x[0] == V]
            log["terminal_q_alternatives"] += max(0, len(ver) - 1)
            for st, rs, q2, tj, meta in sorted(ver, key=lambda x: x[3]):
                dfs({**cells, o: d}, q2, {**counts, o: counts.get(o, 0) + 1}, o, steps + [{"object": o, "dest": d, **meta}], recs + rs, J + tj,
                    left - 1, total, seen)

    flags = {}
    for n in range(n_max + 1):
        tr0, nc0 = provider.truncated(), log["node_cap_hit"]
        dfs(dict(start_cells), np.asarray(start_q, float), {}, None, [], [], 0.0, n, n, {})
        flags[n] = {"search_enumeration_complete": not (provider.truncated() or tr0 or log["node_cap_hit"] or nc0), "all_decided": n not in undecided}
        if best["plan"] is not None or provider.truncated() or log["node_cap_hit"]:
            break
    found = best["plan"]
    nf = found["moves"] if found else None
    shorter_ok = found is not None and all(flags[m]["search_enumeration_complete"] and flags[m]["all_decided"] for m in range(nf))
    depth_ok = found is not None and flags[nf]["search_enumeration_complete"] and flags[nf]["all_decided"]
    complete_no_plan = found is None and all(f["search_enumeration_complete"] and f["all_decided"] for f in flags.values()) and len(flags) == n_max + 1
    claim = ("MIN_MOVES_AND_MIN_TRAVEL_WITHIN_FULLY_ADJUDICATED_FINITE_GRAPH" if shorter_ok and depth_ok else
             "MIN_MOVES_WITHIN_FULLY_ADJUDICATED_FINITE_GRAPH" if shorter_ok else "BEST_VERIFIED_WITHIN_BUDGET" if found else
             "NO_PLAN_IN_FULLY_ADJUDICATED_FINITE_GRAPH" if complete_no_plan else "NO_PLAN_FOUND")
    return {"found": found is not None, "plan": found, "flags": {str(k): v for k, v in flags.items()}, "claim": claim,
            "min_moves_proven": shorter_ok, "min_travel_proven": shorter_ok and depth_ok, "log": log}


def exhaustive(provider, movers, slots, start_cells, start_q, n_max=4, per_obj=2, no_consecutive=True) -> dict:
    """Reference for tests: enumerate EVERY legal action sequence and EVERY verified motion mode (no pruning); lexicographic optimum over
    verified plans, and whether any undecided outcome occurred at a depth <= the optimum (then no proof is available)."""
    best, undecided = None, set()

    def go(cells, q, counts, last, J, depth, steps):
        nonlocal best
        st, _, tj = provider.goal(cells, q)
        if st not in DECIDED:
            undecided.add(depth)
        if st == V and (best is None or (depth, J + tj) < (best[0], best[1])):
            best = (depth, J + tj, list(steps))
        if depth == n_max:
            return
        for o in movers:
            if counts.get(o, 0) >= per_obj or (no_consecutive and o == last):
                continue
            for d in slots(cells, o):
                for s, rs, q2, t, meta in provider.edges(cells, q, o, d):
                    if s not in DECIDED:
                        undecided.add(depth + 1)
                    if s == V:
                        go({**cells, o: d}, q2, {**counts, o: counts.get(o, 0) + 1}, o, J + t, depth + 1, steps + [(o, d)])
    go(dict(start_cells), np.asarray(start_q, float), {}, None, 0.0, 0, [])
    return {"best": best, "undecided_depths": sorted(undecided)}


def random_toy(seed, n_obj=3, n_slots=3, modes=2, p_v=0.6, p_goal=0.25, p_unknown=0.0, nq=4):
    """Deterministic random finite graph over cell occupancies and discrete q values (tests only)."""
    import zlib
    R = lambda k: np.random.default_rng([seed, zlib.crc32(repr(k).encode())])  # noqa: E731  (outcome = pure function of the query)
    objs = [chr(65 + i) for i in range(n_obj)]
    cells0 = {o: i + 1 for i, o in enumerate(objs)} | {"t0": 99}
    free = list(range(10, 10 + n_slots))
    cache_e, cache_g = {}, {}

    class T:
        def truncated(self):
            return False

        def goal(self, cells, q):
            k = (tuple(sorted(cells.items())), float(q[0]))
            if k not in cache_g:
                rng = R(("g",) + k)
                r = rng.random()
                cache_g[k] = (V, round(float(rng.uniform(0.1, 2)), 3)) if r < p_goal else (U_PLAN if r < p_goal + p_unknown else CI, 0.0)
            st, t = cache_g[k]
            return st, {"q": np.zeros((2, 3)), "gap": np.zeros(2)}, t

        def edges(self, cells, q, o, d):
            k = (tuple(sorted(cells.items())), float(q[0]), o, d)
            if k not in cache_e:
                rng, out = R(("e",) + k), []
                for _ in range(modes):
                    r = rng.random()
                    if r < p_v:
                        out.append((V, [{"q": np.zeros((2, 3)), "gap": np.zeros(2)}], np.array([float(rng.integers(0, nq))]),
                                    round(float(rng.uniform(0.1, 2)), 3), {"via": "toy"}))
                    elif r < p_v + p_unknown:
                        out.append((U_PLAN, None, None, 0.0, {}))
                cache_e[k] = out or [(CI, None, None, 0.0, {})]
            return cache_e[k]
    slots = lambda cells, o: [s for s in free + list(range(1, n_obj + 1)) if s not in cells.values()]  # noqa: E731
    return T(), objs, slots, cells0, np.array([0.0])


def compare_with_reference(seeds=range(200), **kw) -> dict:
    """Corrected search vs exhaustive reference on random toy graphs (no unknowns): equal (moves, travel) optimum or both no plan."""
    bad = []
    for s in seeds:
        T, objs, sl, c0, q0 = random_toy(s, **kw)
        ref = exhaustive(T, objs, sl, c0, q0, 3, 2)
        r = search(T, lambda c, q, m, cand: cand, objs, sl, c0, q0, 3, 2, node_cap=10 ** 7)
        got = (r["plan"]["moves"], round(r["plan"]["travel"], 9)) if r["found"] else None
        want = (ref["best"][0], round(ref["best"][1], 9)) if ref["best"] else None
        if got != want:
            bad.append((s, got, want))
    return {"cases": len(list(seeds)), "mismatches": bad}


__all__ = ["search", "exhaustive", "key", "dominated", "random_toy", "compare_with_reference", "itertools"]
