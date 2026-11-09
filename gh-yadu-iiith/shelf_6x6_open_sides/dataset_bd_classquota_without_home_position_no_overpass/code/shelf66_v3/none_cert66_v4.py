"""NONE certificate + audit V4 for the 6 x 6 shelf (SHELF66V3 planner), 2026-11-07.

Why: the V3 audit (none_audit66_v3) tried only the first 6 parking cells per object (nearest first, away from T's path).  All 3 scenes
it had CONFIRMED became feasible after ONE relocation to a cell it never tried (re-check with every free cell): they were K1.

V4 counts a NONE only when it is PROVEN by a blocking certificate, and checks it empirically on top:
  CERT  F0 = the target's feasible grasp directions (src -> goal, pick + place, from the ready pose) with T ALONE on the shelf under the
        scene's own arm ceiling.  For every direction a in F0, blockers[a] = the obstacles o such that with ONLY T and o on the shelf
        (same ceiling) direction a no longer plans.  PROVEN when F0 is not empty and every direction has >= KMIN (5) blockers.
        Argument: an object never makes a direction feasible by being present (collisions only add), so after ANY <= 4 moves (each
        object moved at most once: <= 4 objects leave their cells, whatever they block at their new cells only adds) every direction of
        F0 still has >= 1 of its own blockers in place, and the directions outside F0 are infeasible even on the empty shelf.  The arm
        ceiling stays the same (heights do not change).
  A     direct: T -> goal in the real scene, strong planner (ITERS RRT iterations, seeds 0..S-1): must fail;
  B     one move first, chained (the arm then starts where it stopped): every obstacle to every free cell (B_CELLS=None) or to its
        first B_CELLS cells, then T (teacher planner): must fail -- covers the start-pose effect, which the certificate argues about
        only through collisions.
Statuses: CONFIRMED (cert proven, A and B fail), NOT_PROVEN (cert short: a direction with < KMIN blockers -> not counted),
SOLVABLE_DIRECT / SOLVABLE_ONE_MOVE (the label is wrong), INCONCLUSIVE (budget), ERROR."""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dataset-bd"))
sys.path.insert(0, str(HERE))

AUDIT_VERSION = "NONE_CERT66_V4"
KMIN = 5
B_CELLS = None            # None: every free cell for every obstacle


class Budget(Exception):
    pass


def ceiling_build(A, objs, cells, ceil_h):
    """A.build with the arm ceiling of a scene whose tallest object is ceil_h (shelf_scene reads CEILING_MARGIN when it builds)."""
    TS = A.env()["TS"]
    m0 = TS.CEILING_MARGIN
    TS.CEILING_MARGIN = m0 + max(ceil_h - max(o["height"] for o in objs), 0.0)
    try:
        return A.build(objs, cells)
    finally:
        TS.CEILING_MARGIN = m0


def directions(A, e, name, dst, among=None):
    """distinct azimuths whose grasp plans name -> dst (pick + place) in env e from the ready pose (among: only these)."""
    TS = A.env()["TS"]
    i = e.obj_id(name)
    e.objects[i]["pos"] = e.obj_pos(i)
    q0, ok = e.q_now(), []
    for g in TS.grasp_poses(e.objects[i], e.table_z):
        a = g["azimuth_deg"]
        if (among is None or a in among) and a not in ok and e.plan_pick_place(name, tuple(dst), g, q_start=q0)["feasible"]:
            ok.append(a)
    return sorted(ok)


def certificate(A, objs, cells, src, goal, kmin=KMIN, tick=lambda: None):
    """the blocking certificate of a scene (see the module doc)."""
    t0 = time.time()
    ceil_h = max(o["height"] for o in objs)
    T = next(o for o in objs if o["id"] == "t0")
    e0 = ceiling_build(A, [T], {"t0": tuple(src)}, ceil_h)
    A.settle(e0, {"t0": tuple(src)})
    F0 = directions(A, e0, "t0", goal)
    blockers = {a: [] for a in F0}
    for o in objs:
        if o["id"] == "t0" or not F0:
            continue
        tick()
        cl = {"t0": tuple(src), o["id"]: tuple(cells[o["id"]])}
        e = ceiling_build(A, [T, o], cl, ceil_h)
        A.settle(e, cl)
        left = directions(A, e, "t0", goal, F0)
        for a in F0:
            if a not in left:
                blockers[a].append(o["id"])
    count = {str(a): len(v) for a, v in blockers.items()}
    return {"F0": F0, "blockers": {str(a): v for a, v in blockers.items()}, "count": count, "kmin": kmin,
            "min_count": min(count.values()) if count else None,
            "proven": bool(F0) and all(len(v) >= kmin for v in blockers.values()), "wall_s": round(time.time() - t0, 1)}


def audit_root(r, iters=20000, nseeds=3, budget=3600.0, b_cells=B_CELLS, kmin=KMIN):
    import bd_shelf  # noqa: F401
    import bd_3dt as A
    iters0 = A.RRT_ITERS
    try:
        return _audit(A, r, iters, nseeds, budget, b_cells, kmin)
    finally:
        A.RRT_ITERS = iters0


def _audit(A, r, iters, nseeds, budget, b_cells, kmin):
    TS = A.env()["TS"]
    t0 = time.time()
    objs = r["objects"]
    cells0 = {k: tuple(v) for k, v in r["initial_cells"].items()}
    src, goal = tuple(r["target_src"]), tuple(r["goal"])
    H = {o["id"]: o["height"] for o in objs}
    obst = [o["id"] for o in objs if o["id"] != "t0"]
    st = {"t_queries": 0, "reloc_queries": 0, "executions": 0}
    res = {"root_id": r["ids"]["root_id"], "audit_version": AUDIT_VERSION, "n_obstacles": len(obst),
           "search": {"iters": iters, "seeds": nseeds, "budget_s": budget, "b_cells": b_cells, "kmin": kmin}}

    def tick():
        if time.time() - t0 > budget:
            raise Budget

    def done(status, steps=None, phase=None):
        res.update(status=status, phase=phase, solution=[[o, list(d)] for o, d in steps] if steps is not None else None, stats=st,
                   wall_s=round(time.time() - t0, 1))
        return res

    def q(env, o, d, strong):
        A.RRT_ITERS = iters if strong else 5000
        for s in (range(nseeds) if strong else [0]):
            tick()
            st["t_queries" if o == "t0" else "reloc_queries"] += 1
            p, _ = A.query(env, o, tuple(d), seed=s)
            if p is not None:
                return p
        return None

    def run_ok(env, o, d, p):
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

    try:
        cert = certificate(A, objs, cells0, src, goal, kmin, tick)
        res["certificate"] = cert
        e = A.build(objs, cells0)
        A.settle(e, cells0)
        p = q(e, "t0", goal, strong=True)
        if p is not None and run_ok(e, "t0", goal, p) is None:
            return done("SOLVABLE_DIRECT", [], "A")
        occ = set(cells0.values())
        free = [(i, j) for i in range(A.GRID["R"]) for j in range(A.GRID["C"]) if (i, j) not in occ and (i, j) != goal]
        b = {"pairs": 0, "relocated": 0}
        for o in obst:
            near = sorted(free, key=lambda x: (A.cheb(x, cells0[o]), x))
            for d in (near if b_cells is None else near[:b_cells]):
                A.settle(e, cells0)
                b["pairs"] += 1
                p = q(e, o, d, strong=False)
                if p is None or run_ok(e, o, d, p):
                    continue
                b["relocated"] += 1
                pt = q(e, "t0", goal, strong=False)
                if pt is not None and run_ok(e, "t0", goal, pt) is None:
                    res["B_one_move"] = b
                    return done("SOLVABLE_ONE_MOVE", [(o, d)], "B")
        res["B_one_move"] = b
        return done("CONFIRMED" if cert["proven"] else "NOT_PROVEN")
    except Budget:
        return done("INCONCLUSIVE")
    except Exception as err:  # noqa: BLE001
        res["trace"] = traceback.format_exc()[-1500:]
        return done(f"ERROR:{type(err).__name__}:{str(err)[:120]}")


if __name__ == "__main__":
    # python none_cert66_v4.py <root.json> [b_cells|all]: audit one stored root, print the result
    rec = json.load(open(sys.argv[1]))
    bcl = None if len(sys.argv) < 3 or sys.argv[2] == "all" else int(sys.argv[2])
    out = audit_root(rec, b_cells=bcl)
    print(json.dumps({k: out.get(k) for k in ("root_id", "status", "phase", "solution", "wall_s", "B_one_move")}))
    print(json.dumps(out.get("certificate", {}).get("count")), "min", out.get("certificate", {}).get("min_count"))
