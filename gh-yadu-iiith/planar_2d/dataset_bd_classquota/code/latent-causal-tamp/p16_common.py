"""P16 shared, scene-only task interface and the common non-learned rearrangement search.

Task = {scene, goal_cell, fixed_ids, target='t0', key}: built from the physical scene and declared static metadata only (fixed structures
are an object property). No teacher sequence, catalogue, clearing set or label is read here. propose_movers / propose_slots derive
candidates from the CURRENT hypothetical state; verify_sequence certifies a proposed ordered repair step by step (used by B);
search() is the expanded-domain tri-state lexicographic search (repeat moves allowed up to per_obj, dominance-pruned) used by D and the
guidance baselines."""
from __future__ import annotations

import time

import numpy as np

import p15_search as ps

V, CI, SI, U_BUD = ps.V, ps.CI, ps.SI, ps.U_BUD
DECIDED = ps.DECIDED
MOVER_CAP, SLOT_K = 6, 6


def task_from_root(root, key=None) -> dict:
    """Scene-only task from a constructed root: scene, goal and the fixed-structure object ids (static metadata). Nothing else."""
    from scene import scene_from_dict
    s = scene_from_dict(root["scene"])
    fixed = {o.id for o in s.objects if o.kind != "target" and o.cell in set(root.get("fixed", []))}
    return {"scene": root["scene"], "goal_cell": root["goal_cell"], "fixed_ids": sorted(fixed), "target": "t0", "key": key or root.get("scene_hash", "")}


def propose_movers(task, cells) -> tuple[list, list]:
    """All movable non-target objects (not fixed structures), nearest-to-target first; (kept, omitted beyond MOVER_CAP)."""
    from scene import scene_from_dict
    s = scene_from_dict(task["scene"])
    T = np.asarray(s.lattice.center(cells["t0"]))
    mov = [o.id for o in s.objects if o.kind != "target" and o.id not in set(task["fixed_ids"])]
    mov.sort(key=lambda o: (float(np.linalg.norm(np.asarray(s.lattice.center(cells[o])) - T)), o))
    return sorted(mov[:MOVER_CAP]), mov[MOVER_CAP:]


def slot_fn(phys, idx=None, k=4):
    """Destinations on the current state: k nearest free annulus-safe cells >= 2 cells from target and goal (freed source cells included),
    plus library-witnessed destinations from the object's current cell (other roots only); at most SLOT_K."""
    import p6_scenes as p6
    T = phys.s0.get("t0").cell

    def f(cells, o):
        s = phys.scene(cells, phys.s0.robot.q_rad)
        ds_ = p6._dests(phys.scr, s, o, [T, phys.goal_cell], [], k)
        occ = set(cells.values())
        if idx is not None:
            ds_ += sorted(d for (src, d) in idx.by_key if src == cells[o] and d not in ds_ and d in phys.scr.valid and d not in occ
                          and p6._cheb(d, T) >= 2 and p6._cheb(d, phys.goal_cell) >= 2)
        return ds_[:SLOT_K]
    return f


def best_edge(outs):
    ver = [x for x in outs if x[0] == V]
    return min(ver, key=lambda x: x[3]) if ver else None


def verify_sequence(phys, seq) -> dict:
    """Certify a proposed ordered repair [(object, dest), ...] then the target, step by step on the actual sequential state; per step the
    cheapest certified motion mode is used. Returns VERIFIED with the plan, or the first non-verified status (UNKNOWN never = infeasible)."""
    cells, q = {o.id: o.cell for o in phys.s0.objects}, np.asarray(phys.s0.robot.q_rad, float)
    steps, recs, J = [], [], 0.0
    for j, (o, d) in enumerate(seq):
        outs = phys.edges(cells, q, o, d)
        e = best_edge(outs)
        if e is None:
            return {"status": outs[0][0] if outs else ps.U_PLAN, "failed_step": j, "plan": None}
        st, rs, q2, tj, meta = e
        steps.append({"object": o, "dest": d, **meta})
        recs += rs
        J += tj
        cells, q = {**cells, o: d}, q2
    st, rec, tj = phys.goal(cells, q)
    if st != V:
        return {"status": st, "failed_step": "target", "plan": None}
    return {"status": V, "failed_step": None, "plan": {"moves": len(seq), "travel": J + tj, "steps": steps, "records": recs + [rec]}}


def search(provider, order, movers, slots, start_cells, start_q, n_max=4, per_obj=1, node_cap=600) -> dict:
    """Iterative deepening on relocations; DFS in guidance order with travel branch-and-bound and dominance on (occupancy, q) against
    entries with no more moves per object and no more travel. An object may move up to per_obj times (never twice in a row). Claims as
    P15 (finite declared graph only); a node-cap or budget stop marks the pass incomplete."""
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
        key = (tuple(sorted(cells.items())), tuple(np.round(np.asarray(q, float), 12)))
        if any(J2 <= J and all(c2.get(o, 0) <= counts.get(o, 0) for o in c2) for J2, c2 in seen.get(key, [])):
            log["pruned_dominated"] += 1
            return
        seen.setdefault(key, []).append((J, dict(counts)))
        if left == 0:
            st, r, tj = provider.goal(cells, q)
            rec("goals", total, st)
            if st == V and (best["plan"] is None or J + tj < best["plan"]["travel"]):
                best["plan"] = {"moves": total, "travel": J + tj, "steps": steps, "records": recs + [r]}
                log.setdefault("anytime", []).append({"cpu_s": time.process_time() - t0, "nodes": log["nodes"], "moves": total, "travel": J + tj})
            return
        cands = [(o, d) for o in movers if counts.get(o, 0) < per_obj and o != last for d in slots(cells, o)]
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
    claim = ("MIN_MOVES_AND_MIN_TRAVEL_WITHIN_FULLY_ADJUDICATED_FINITE_GRAPH" if shorter_ok and depth_ok else
             "MIN_MOVES_WITHIN_FULLY_ADJUDICATED_FINITE_GRAPH" if shorter_ok else "BEST_VERIFIED_WITHIN_BUDGET" if found else "NO_PLAN_FOUND")
    return {"found": found is not None, "plan": found, "flags": {str(k): v for k, v in flags.items()}, "claim": claim,
            "min_moves_proven": shorter_ok, "min_travel_proven": shorter_ok and depth_ok, "log": log}


# ------------------------------------------------------------------ DENSE clutter family (physics-first, no prescreen) for families 6-8
def register():
    """P13 grammars + DENSE: four movers drawn from the target neighbourhood (Chebyshev <= 2) and the link-0 sweep window, plus 0-2 fixed
    grasp-side structures; no intent predicate (unconditioned). Mechanism tags (slot conflict, freed-slot reuse, repeat move, plan length)
    are assigned AFTERWARDS from physics only (search on DEV), never from model scores."""
    import p13_pipeline as pp
    import p13_scenes as s13
    pd = pp._register()
    if "DENSE" in s13.FAMS:
        return pd
    orig = s13.layout

    def layout(rng, fam, scr, a0):
        if fam != "DENSE":
            return orig(rng, fam, scr, a0)
        valid = sorted(scr.valid)
        T = int(rng.choice([c for c in valid if 1 <= c // 8 <= 6 and c % 8 >= 3]))
        goal = int(rng.choice([c for c in valid if s13._cheb(c, T) >= 3]))
        sides = [str(x) for x in rng.permutation(["W", "N", "S"])]
        fixed = [s13._nb(T, x) for x in sides[:int(rng.integers(0, 3))]]
        near = sorted({x for x in valid if 1 <= s13._cheb(x, T) <= 2} | set(s13._arc(scr, T, a0, pad=0.0)))
        near = [x for x in near if x not in fixed and x != goal and s13._cheb(x, goal) >= 2]
        if len(near) < 4:
            return None
        role = {"T": T, **dict(zip("ABCD", (int(x) for x in rng.choice(near, 4, replace=False))))}
        fixed = [x for x in fixed if x is not None]
        cells = list(role.values()) + fixed
        if len(set(cells)) != len(cells) or any(s13._cheb(goal, x) < 2 for x in cells):
            return None
        return role, fixed, goal
    s13.layout = layout
    s13.FAMS = s13.FAMS + ("DENSE",)
    s13.POOL["DENSE"], s13.TOK["DENSE"] = "AB", s13.TOK["NAT"]
    orig_pred = s13._predicates
    s13._predicates = lambda cx, s, fam, *a: (True, []) if fam == "DENSE" else orig_pred(cx, s, fam, *a)
    return pd


# ------------------------------------------------------------------ scene-only catalogue and deterministic physical labels (B and D)
POOL_CAP = 3


def catalogue(task, phys, idx=None) -> dict:
    """pool = up to POOL_CAP movable objects nearest the target; parking = distinct first scene-only slot per pool object (original
    state); edits = every nonempty pool subset parked (D lattice / B scene queries); sequences = empty, singles and ordered pairs (B
    candidates; 3-move repairs are left to the search). Pure function of the task + physics screens."""
    import itertools
    cells0 = {o.id: o.cell for o in phys.s0.objects}
    movers, _ = propose_movers(task, cells0)
    T = np.asarray(phys.s0.lattice.center(cells0["t0"]))
    pool = sorted(movers, key=lambda o: (float(np.linalg.norm(np.asarray(phys.s0.lattice.center(cells0[o])) - T)), o))[:POOL_CAP]
    sl, park = slot_fn(phys, idx), {}
    for o in pool:
        d = next((x for x in sl(cells0, o) if x not in park.values()), None)
        if d is not None:
            park[o] = d
    pool = [o for o in pool if o in park]
    edits = [{"kind": "+".join(u), "edit": {o: park[o] for o in u}, "eligible": True} for n in range(1, len(pool) + 1) for u in itertools.combinations(pool, n)]
    seqs = [[]] + [[o] for o in pool] + [list(p) for p in itertools.permutations(pool, 2)]
    return {"pool": pool, "parking": park, "edits": edits, "sequences": [[(o, park[o]) for o in s] for s in seqs], "movers": movers}


def label(task, phys, cat, sequences=True) -> dict:
    """Statuses for the original task, every lattice edit (target-only query on the edited scene, robot at q0) and every sequence
    (verify_sequence). y = 1 VERIFIED, y = 0 only from a sound certificate (target CERTIFIED_INFEASIBLE / a step CERTIFIED or
    STATIC_INVALID), y = None otherwise (UNKNOWN: masked, never negative)."""
    cells0, q0 = {o.id: o.cell for o in phys.s0.objects}, np.asarray(phys.s0.robot.q_rad, float)
    yof = lambda st: 1 if st == V else 0 if st in (CI, SI) else None  # noqa: E731
    st0 = phys.goal(cells0, q0)[0]
    edits = {e["kind"]: phys.goal({**cells0, **e["edit"]}, q0)[0] for e in cat["edits"]}
    seqs = {}
    for s in cat["sequences"] if sequences else []:
        if not s:
            continue
        r = verify_sequence(phys, s)
        seqs[".".join(f"{o}>{d}" for o, d in s)] = {"status": r["status"], "failed_step": r["failed_step"], "y": yof(r["status"]),
                                                     "travel": r["plan"]["travel"] if r["plan"] else None}
    return {"original": {"status": st0, "y": yof(st0)}, "edits": {k: {"status": v, "y": yof(v)} for k, v in edits.items()}, "sequences": seqs,
            "used": dict(phys.used), "stats": dict(phys.stats)}
