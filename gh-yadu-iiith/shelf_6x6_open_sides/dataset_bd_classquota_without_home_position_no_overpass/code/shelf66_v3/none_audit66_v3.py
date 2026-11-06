# SHELF66_V3 COPY (2026-11-05) of ../none_audit66.py (V2.1): the same search, run on the SHELF66V3 planner of this folder (fingers
# checked pre-shaped).  New: audit_root(record) for in-process use -- run_shelf66_full.py (V3) counts a teacher NONE only when this
# audit is CONFIRMED -- and A.RRT_ITERS is restored afterwards (a pool worker teaches its next root with the teacher's 5,000).
"""NONE audit for the 2,500-scene 6 x 6 shelf run (shelf66_v3): tries much harder than the teacher to SOLVE every NONE scene, so
that a NONE that is in fact solvable is found.  Read-only on the run; writes only --out (one JSON line per root, resumable).

    python shelf66_v3/none_audit66_v3.py --run <run dir> --out <audit.jsonl> [--ids a,b] [--workers 4] [--iters 20000] [--seeds 3] [--park 6]

The teacher's NONE: T -> goal fails from the ready pose (first plannable of the 32 grasp poses, RRT-Connect 5,000 iterations,
seed 0), and no repair was found: sets of <= 4 objects of its pool (intended blockers + distractors near T / goal / path), each
object moved to ONE designated parking cell, relocations planned from the ready pose, the sequence then executed chained.
This audit widens each of those limits.  A scene stays NONE ("CONFIRMED") only if all of these fail:
  A  direct: T -> goal from the ready pose, seeds 0..S-1, ITERS RRT iterations (strong planner);
  B  one move first, executed chained: every obstacle to each of its first 2 parking candidates, then T (strong) from where the
     arm stopped -- the start-pose effect behind the INCONSISTENT label;
  C  repairs: every set U of obstacles (ALL of them, not only the teacher's pool), |U| <= 4, whose REMOVAL lets T reach the goal
     (strong planner); for each U, every move order, each object to the first of its P parking candidates (designated first,
     then free cells away from T, the goal and the straight path, nearest first) whose relocation plans in the scene it faces;
     then T (strong).  A candidate counts only after chained execution in MuJoCo with the teacher's checks (end cell, height,
     no other object displaced), every move planned from where the arm is (strong planner).
     V2: per object up to KEEP = 3 plannable cells among the first P candidates, every combination (a parked object can block T).
Statuses: SOLVABLE_DIRECT / SOLVABLE_ONE_MOVE / SOLVABLE_REPAIR (the NONE label is contradicted), CONFIRMED (nothing found in
the whole search above), INCONCLUSIVE (the per-root time budget ran out first), ERROR."""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent                                # shelf66_v3/
sys.path.insert(0, str(HERE.parent / "dataset-bd"))                 # bd_core (shared)
sys.path.insert(0, str(HERE))                                       # this folder's bd_shelf / bd_3dt (SHELF66V3)


AUDIT_VERSION = "NONE_AUDIT66_V3"     # = V2.1's search on the SHELF66V3 planner (fingers pre-shaped in every check)
# V2: repairs try every combination of up to KEEP plannable parking cells per object (V1: the first).
# V2.1: a state reached by a sequence of moves is set up in a conflict-free order (put()).  V2 used A.settle, which teleports in
# dict order and raised "cell ... is occupied" when an object was parked in a cell another one had left earlier in the sequence
# (6 of 45 roots ended as ERROR).  Wherever V2's order worked, put() teleports in the same order, so V2's other results stand.
KEEP = 3


class Budget(Exception):
    pass


def audit(job):
    """one root of --run, by id (the command-line audit)."""
    run, rid, iters, nseeds, P, budget = job
    return audit_root(json.load(open(Path(run) / "roots" / f"{rid}.json")), iters, nseeds, P, budget)


def audit_root(r, iters=20000, nseeds=3, P=6, budget=3600.0):
    """audit one root record (a stored root file's content, or the record a runner is about to store)."""
    import bd_shelf  # noqa: F401  (paths + the 6 x 6 shelf env for bd_3dt)
    import bd_3dt as A
    iters0 = A.RRT_ITERS
    try:
        return _audit(A, r, iters, nseeds, P, budget)
    finally:
        A.RRT_ITERS = iters0                 # the teacher's ceiling again, for whatever this process plans next


def _audit(A, r, iters, nseeds, P, budget):
    TS = A.env()["TS"]
    t0 = time.time()
    rid = r["ids"]["root_id"]
    objs = r["objects"]
    cells0 = {k: tuple(v) for k, v in r["initial_cells"].items()}
    src, goal = tuple(r["target_src"]), tuple(r["goal"])
    H = {o["id"]: o["height"] for o in objs}
    obst = [o["id"] for o in objs if o["id"] != "t0"]
    seeds = list(range(nseeds))
    st = {"t_queries": 0, "reloc_queries": 0, "executions": 0, "removal_sets_checked": 0, "orders_tried": 0}
    res = {"root_id": rid, "audit_version": AUDIT_VERSION, "n_obstacles": len(obst), "teacher_scope": r["labels"]["blocker"]["scope"],
           "search": {"iters": iters, "seeds": nseeds, "park": P, "keep": KEEP, "budget_s": budget}}
    e = A.build(objs, cells0)

    def tick():
        if time.time() - t0 > budget:
            raise Budget

    def q(env, o, d, strong):
        """plan o -> d from the arm's current pose (A.query: grasp poses in order, first plannable wins)."""
        A.RRT_ITERS = iters if strong else 5000
        for s in (seeds if strong else [0]):
            tick()
            st["t_queries" if o == "t0" else "reloc_queries"] += 1
            p, _ = A.query(env, o, tuple(d), seed=s)
            if p is not None:
                return p, s
        return None, None

    def run_ok(env, o, d, p):
        """execute p in MuJoCo; None when the teacher's checks pass (end cell, height, nobody else displaced)."""
        st["executions"] += 1
        before = {n: env.cell_of(n) for n in env.obj_index if n != o}
        try:
            env.run_plan(p)
        except TS.GraspFailed as err:
            return f"GRASP_FAILED:{str(err)[:60]}"
        z = env.obj_pos(o)[2] - env.table_z - H[o] / 2
        moved = sorted(n for n in before if env.cell_of(n) != before[n])
        if env.cell_of(o) != tuple(d) or abs(z) > 0.01 or moved:
            return f"NOT_VERIFIED:end={env.cell_of(o)} z={z:.3f} displaced={moved}"
        return None

    def chained(steps):
        """the dataset's execution: start at the ready pose, each move planned from where the arm is, executed, verified; then T."""
        A.settle(e, cells0)
        log = []
        for o, d in list(steps) + [("t0", goal)]:
            p, s = q(e, o, d, strong=True)
            if p is None:
                return False, log + [f"{o}->{list(d)}: no plan"]
            err = run_ok(e, o, d, p)
            log.append(f"{o}->{list(d)} (seed {s}, grasp {p['grasp_id']})" + (f" FAILED {err}" if err else ""))
            if err:
                return False, log
        return True, log

    def put(env, state):
        """A.settle for a state reached by a sequence of moves: reset (home cells, ready pose), then teleport -- in dict order, as
        A.settle does, except that an object whose cell is still occupied (by one that leaves it later in the sequence) waits."""
        env.reset()
        todo = [o for o, c in state.items() if env.cell_of(o) != tuple(c)]
        while todo:
            occ = {env.cell_of(n) for n in env.obj_index}
            o = next((x for x in todo if tuple(state[x]) not in occ), None)
            if o is None:
                raise ValueError(f"no conflict-free order to set up {state}")
            env.teleport(o, tuple(state[o]))
            todo.remove(o)

    def done(status, steps=None, log=None, phase=None):
        res.update(status=status, phase=phase, solution=[[o, list(d)] for o, d in steps] if steps is not None else None,
                   execution_log=log, stats=st, wall_s=round(time.time() - t0, 1))
        return res

    line = []
    for t in np.linspace(0, 1, 29)[1:-1]:
        x = tuple(int(round(v)) for v in (1 - t) * np.array(src) + t * np.array(goal))
        if x not in line and x not in (src, goal):
            line.append(x)
    cheb = lambda a, b: max(abs(a[0] - b[0]), abs(a[1] - b[1]))  # noqa: E731
    allc = [(i, j) for i in range(A.GRID["R"]) for j in range(A.GRID["C"])]

    def cands(o, cells):
        occ = set(cells.values())
        free = [x for x in allc if x not in occ and x not in (src, goal)]
        key = lambda x: (cheb(x, cells0[o]), abs(x[0] - cells0[o][0]), x)  # noqa: E731
        des = [tuple(x) for x in r["parking"].get(o, []) if tuple(x) in free]
        good = sorted((x for x in free if x not in des and cheb(x, src) >= 2 and cheb(x, goal) >= 2 and x not in line), key=key)
        rest = sorted((x for x in free if x not in des and x not in good), key=key)
        return (des + good + rest)[:P]

    try:
        # A: direct, strong planner
        A.settle(e, cells0)
        if q(e, "t0", goal, strong=True)[0] is not None:
            ok, log = chained([])
            res["A_direct"] = {"plan_found": True, "executed": ok, "log": log}
            if ok:
                return done("SOLVABLE_DIRECT", [], log, "A")
        else:
            res["A_direct"] = {"plan_found": False}
        # B: one move first (chained), then T from where the arm stopped
        tried_b = []
        for o in obst:
            for d in cands(o, cells0)[:2]:
                A.settle(e, cells0)
                p, _ = q(e, o, d, strong=False)
                if p is None or run_ok(e, o, d, p):
                    tried_b.append([o, list(d), "relocation fails"])
                    continue
                pt, _ = q(e, "t0", goal, strong=True)
                if pt is None:
                    tried_b.append([o, list(d), "T: no plan"])
                    continue
                err = run_ok(e, "t0", goal, pt)
                if err is None:
                    ok, log = chained([(o, d)])             # re-run from scratch for the record (same plans: deterministic)
                    res["B_one_move"] = tried_b + [[o, list(d), "SOLVED" if ok else "solved once, not on re-run"]]
                    if ok:
                        return done("SOLVABLE_ONE_MOVE", [(o, d)], log, "B")
                tried_b.append([o, list(d), f"T: {err}"])
        res["B_one_move"] = tried_b
        # C: repairs over every obstacle set whose removal frees T
        clear, tq_memo = [], {}
        for k in range(1, min(4, len(obst)) + 1):
            for U in itertools.combinations(obst, k):
                if any(set(c) <= set(U) for c in clear):
                    clear.append(U)                          # removing more never hurts: no query needed
                    continue
                keep = [x for x in objs if x["id"] not in U]
                cl = {x["id"]: cells0[x["id"]] for x in keep}
                e2 = A.build(keep, cl)
                A.settle(e2, cl)
                st["removal_sets_checked"] += 1
                if q(e2, "t0", goal, strong=True)[0] is not None:
                    clear.append(U)
        res["C_clearing_sets_by_removal"] = [list(U) for U in clear]
        reloc_memo, tried_c = {}, []

        def note(steps, why):
            if len(tried_c) < 80:
                tried_c.append([[o, list(d)] for o, d in steps] + [why])

        def parks(state, o):
            """up to KEEP parking cells (of the first P candidates) whose relocation plans in this state (from the ready pose)."""
            out = []
            for d in cands(o, state):
                key = (tuple(sorted(state.items())), o, d)
                if key not in reloc_memo:
                    put(e, state)
                    reloc_memo[key] = q(e, o, d, strong=False)[0] is not None
                if reloc_memo[key]:
                    out.append(d)
                    if len(out) >= KEEP:
                        break
            return out

        def dfs(state, order, steps):
            """V2: every combination of the plannable parking cells (not just the first): a parked object can still block T."""
            if len(steps) == len(order):
                sk = tuple(sorted(state.items()))
                if sk not in tq_memo:
                    put(e, state)
                    tq_memo[sk] = q(e, "t0", goal, strong=True)[0] is not None
                if not tq_memo[sk]:
                    note(steps, "T: no plan after the moves")
                    return None
                ok, log = chained(steps)
                if ok:
                    return steps, log
                note(steps, "chained execution: " + log[-1])
                return None
            o = order[len(steps)]
            ps = parks(state, o)
            if not ps:
                note(steps, f"{o}: no relocation plans")
            for d in ps:
                got = dfs({**state, o: d}, order, steps + [(o, d)])
                if got:
                    return got
            return None
        for U in clear:
            for order in itertools.permutations(U):
                st["orders_tried"] += 1
                got = dfs(dict(cells0), order, [])
                if got:
                    res["C_tried"] = tried_c
                    return done("SOLVABLE_REPAIR", got[0], got[1], "C")
        res["C_tried"] = tried_c
        res["C_distinct_final_states_T_checked"] = len(tq_memo)
        return done("CONFIRMED")
    except Budget:
        return done("INCONCLUSIVE")
    except Exception as err:  # noqa: BLE001
        res["trace"] = traceback.format_exc()[-1500:]
        return done(f"ERROR:{type(err).__name__}:{str(err)[:120]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ids", default="")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--park", type=int, default=6)
    ap.add_argument("--budget-s", type=float, default=5400)
    a = ap.parse_args()
    run, out = Path(a.run), Path(a.out)
    if a.ids:
        ids = a.ids.split(",")
    else:
        ids = [e["root_id"] for e in map(json.loads, open(run / "ledger.jsonl")) if e.get("accepted") and e.get("observed") == "NONE"]
    done = {r["root_id"] for r in map(json.loads, open(out)) if not r["status"].startswith("ERROR")} if out.exists() else set()   # ERRORs re-run
    todo = [i for i in ids if i not in done]
    print(f"{len(ids)} NONE roots, {len(done)} already audited, {len(todo)} to do", flush=True)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool, open(out, "a") as f:
        for res in pool.imap_unordered(audit, [(str(run), i, a.iters, a.seeds, a.park, a.budget_s) for i in todo]):
            f.write(json.dumps(res) + "\n")
            f.flush()
            print(f"{res['root_id']}: {res['status']} ({res['wall_s']:.0f} s) {res.get('solution')}", flush=True)


if __name__ == "__main__":
    main()
