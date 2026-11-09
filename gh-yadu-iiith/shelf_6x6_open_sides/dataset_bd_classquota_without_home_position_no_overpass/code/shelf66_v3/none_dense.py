"""DENSE NONE construction for the 6 x 6 shelf (SHELF66V3 planner / teacher; user 2026-11-06: "make sure that none is definitely
infeasible so place as many obstacles as possible").

A scene is built so that NO set of <= 4 obstacles, even removed from the shelf outright (more than any 4 moves can do), lets the target
reach its goal:
  1. start / goal as bd_3dt.construct (spec source / displacement); the obstacles' dimensions are drawn first (heights 12-18 cm), so
     the scene's arm ceiling (tallest object + CEILING_MARGIN) is known;
  2. F0 = the target's feasible grasp directions on the EMPTY shelf UNDER THAT CEILING (shelf_scene builds the ceiling from the
     objects present; it is patched to the scene's own here, as obj_pickable3 does);
  3. every candidate cell (rows 0-5, within 2 cells of start or goal, or on / next to the straight path) is PROBED: which directions
     of F0 still plan with one standard blocker there (bd_3dt_plus.still_feasible's test, same ceiling);
  4. cells are chosen greedily until every direction of F0 is cut by >= KMIN (5) chosen cells, each of which cuts it ON ITS OWN
     -> removing any 4 obstacles leaves every direction with at least one blocker; then more cells are filled ("as many obstacles as
     possible": every remaining candidate cell, up to N_MAX objects);
  5. every obstacle must be pickable ALONE on the empty shelf from its cell (the scene's ceiling): no fixed objects;
  6. the real scene is checked: the target's move does not plan, and no single removal of one of the KMIN-cover cells frees it.
The scene is then labelled like every root (V3 teacher, consistency checks, NONE audit gate: counted only if CONFIRMED)."""
from __future__ import annotations

import itertools
import time

import numpy as np

KMIN = 5                    # chosen cells cutting each feasible direction (any 4 removed -> >= 1 left)
N_MAX = 14                  # obstacles per scene (fill cells after the cover)
ATTEMPTS = 6                # start / goal pairs per root
METHOD = "dense_none_v1"


def ceiling_build(A, objs, cells, ceil_h):
    """A.build with the arm ceiling of a scene whose tallest object is ceil_h (shelf_scene reads CEILING_MARGIN in __init__)."""
    TS = A.env()["TS"]
    m0 = TS.CEILING_MARGIN
    TS.CEILING_MARGIN = m0 + max(ceil_h - max(o["height"] for o in objs), 0.0)
    try:
        return A.build(objs, cells)
    finally:
        TS.CEILING_MARGIN = m0


def directions(A, e, name, dst, among=None):
    """distinct azimuths whose grasp plans name -> dst in env e from the ready pose (among: only these)."""
    TS = A.env()["TS"]
    i = e.obj_id(name)
    e.objects[i]["pos"] = e.obj_pos(i)
    q0, ok = e.q_now(), []
    for g in TS.grasp_poses(e.objects[i], e.table_z):
        a = g["azimuth_deg"]
        if (among is None or a in among) and a not in ok and e.plan_pick_place(name, tuple(dst), g, q_start=q0)["feasible"]:
            ok.append(a)
    return sorted(ok)


def pickable_alone(A, obj, cell, ceil_h):
    """obj alone on the shelf at cell (scene ceiling): some grasp plans a pick-and-place to a free mid cell."""
    for dst in ((2, 2), (3, 4), (1, 3)):
        if tuple(dst) == tuple(cell):
            continue
        e = ceiling_build(A, [obj], {obj["id"]: tuple(cell)}, ceil_h)
        if directions(A, e, obj["id"], dst):
            return True
    return False


def construct_dense_none(spec, seed, root_id, A, P, bc, trace=None):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    R, C = A.GRID["R"], A.GRID["C"]
    lo, hi = A.DISP[spec["displacement"]]
    T = {"id": "t0", "shape": "cylinder", "size": float(rng.uniform(0.016, 0.020)), "height": float(rng.uniform(0.12, 0.16)), "color": "green"}
    dims = []
    for j in range(N_MAX):
        cyl = bool(rng.integers(2))
        dims.append({"id": f"o{j:02d}", "shape": "cylinder" if cyl else "box", "size": float(rng.uniform(0.013, 0.018) if cyl else rng.uniform(0.012, 0.016)),
                     "height": float(rng.uniform(0.12, 0.18)), "color": A.COLORS[j % len(A.COLORS)]})
    ceil_h = max([T["height"]] + [d["height"] for d in dims])
    probe = {**P.PROBE, "id": "p", "height": max(P.PROBE["height"], 0.12)}
    stats = {"pairs": 0, "probed_cells": 0, "short_cover": 0, "unpickable": 0, "t_feasible": 0, "single_removal_frees": 0}
    for attempt in range(ATTEMPTS):
        src = (int(rng.choice(A.ROWS[spec["source"]])), int(rng.integers(1, C - 1)))
        goals = [(r, c) for r in range(1, R) for c in range(C) if lo <= A.cheb((r, c), src) <= hi]
        if not goals:
            continue
        goal = goals[int(rng.integers(len(goals)))]
        stats["pairs"] += 1
        e0 = ceiling_build(A, [T], {"t0": src}, ceil_h)
        F0 = directions(A, e0, "t0", goal)
        if trace is not None:
            trace.append(("F0", src, goal, F0, round(time.time() - t0)))
        if not F0:
            continue
        route = set(P.line(src, goal))
        near_route = {(r + a, c + b) for r, c in route for a in (-1, 0, 1) for b in (-1, 0, 1)}
        cand = sorted(x for x in itertools.product(range(R), range(C))
                      if x not in (src, goal) and (A.cheb(x, src) <= 2 or A.cheb(x, goal) <= 2 or x in near_route))
        cuts = {}
        for x in cand:
            e = ceiling_build(A, [T, probe], {"t0": src, "p": x}, ceil_h)
            cuts[x] = set(F0) - set(directions(A, e, "t0", goal, F0))
            stats["probed_cells"] += 1
        if trace is not None:
            trace.append(("cuts", {f"{x[0]},{x[1]}": sorted(v) for x, v in cuts.items() if v}, round(time.time() - t0)))
        # 4. greedy KMIN-cover (row 0 only when it cuts something; rows 1-5 preferred on ties), then fill
        need, pick = {a: KMIN for a in F0}, []
        while any(v > 0 for v in need.values()) and len(pick) < N_MAX:
            score = lambda x: (sum(need[a] > 0 for a in cuts[x]), x[0] > 0, -A.cheb(x, src), rng.random())  # noqa: E731
            opts = [x for x in cand if x not in pick and any(need[a] > 0 for a in cuts[x])]
            if not opts:
                break
            x = max(opts, key=score)
            pick.append(x)
            for a in cuts[x]:
                need[a] -= 1
        if any(v > 0 for v in need.values()):
            stats["short_cover"] += 1
            if trace is not None:
                trace.append(("short_cover", dict(need), pick))
            continue
        cover = list(pick)
        fill = [x for x in cand if x not in pick and x[0] >= 1]
        rng.shuffle(fill)
        fill.sort(key=lambda x: -len(cuts[x]))                     # cells that cut something first
        pick += fill[:N_MAX - len(pick)]
        # 5. no fixed objects: every obstacle pickable alone (its dims re-drawn up to 3 times, else the cell is dropped / pair fails)
        objs, cells, ok = [T], {"t0": list(src)}, True
        for j, x in enumerate(pick):
            d = dims[j]
            for k in range(4):
                if pickable_alone(A, d, x, ceil_h):
                    break
                stats["unpickable"] += 1
                cyl = bool(rng.integers(2))
                d = {**d, "shape": "cylinder" if cyl else "box", "size": float(rng.uniform(0.013, 0.018) if cyl else rng.uniform(0.012, 0.016)),
                     "height": float(rng.uniform(0.12, ceil_h))}
            else:
                if x in cover:
                    ok = False
                    break
                continue
            dims[j] = d
            objs.append(d)
            cells[d["id"]] = [int(x[0]), int(x[1])]
        if not ok:
            if trace is not None:
                trace.append(("cover_cell_unpickable", x))
            continue
        # 6. the real scene: T blocked, and with any ONE cover object removed still blocked
        e = A.build(objs, cells)
        A.settle(e, cells)
        if A.query(e, "t0", list(goal))[0] is not None:
            stats["t_feasible"] += 1
            continue
        id_at = {tuple(v): k for k, v in cells.items()}
        frees = None
        for x in cover:
            keep = [o for o in objs if o["id"] != id_at[x]]
            cl = {o["id"]: cells[o["id"]] for o in keep}
            e1 = A.build(keep, cl)
            A.settle(e1, cl)
            if A.query(e1, "t0", list(goal))[0] is not None:
                frees = x
                break
        if frees is not None:
            stats["single_removal_frees"] += 1
            if trace is not None:
                trace.append(("single_removal_frees", frees))
            continue
        break
    else:
        raise A.Reject("DENSE_NONE_NOT_FOUND:" + ",".join(f"{k}={v}" for k, v in stats.items()))
    # construction dict (as bd_3dt_plus.construct_route's)
    occ = {tuple(v) for v in cells.values()}
    avoid = {src, goal} | P.hand_zone(src, F0) | P.hand_zone(goal, F0) | route | {(0, c) for c in range(C)}
    loose = {src, goal} | {(0, c) for c in range(C)}
    id_at = {tuple(v): k for k, v in cells.items()}
    cover_ids = [id_at[x] for x in cover if x in id_at]
    roles, park, alt, taken = {}, {}, {}, set()

    def area(x):
        if x in near_route:
            return "near_path"
        if A.cheb(x, src) <= 2:
            return "near_T"
        if A.cheb(x, goal) <= 2:
            return "near_goal"
        return "far"
    for o in objs[1:]:
        x = tuple(cells[o["id"]])
        roles[o["id"]] = "intended_blocker" if o["id"] in cover_ids else f"distractor:{area(x)}"
        cc = P.parking_cells(x, avoid, occ | taken, P.N_ALT_PARK) or P.parking_cells(x, loose, occ | taken, P.N_ALT_PARK)
        alt[o["id"]] = [list(y) for y in cc]
        park[o["id"]] = alt[o["id"]][:bc.PARKING_N[spec["parking"]]]
        if cc:
            taken.add(cc[0])
    least = min(F0, key=lambda a: sum(a in cuts[x] for x in cover))
    intended = [id_at[x] for x in cover if least in cuts[x]]
    return {"objs": objs, "cells": cells, "src": list(src), "goal": list(goal), "roles": roles, "parking": park, "alt_parking": alt,
            "intended": intended, "wrong_dest": [list(x) for x in P.line(src, goal) if x not in occ][:2],
            "pools": {"route": [list(x) for x in cand], "grasp": [], "place": [], "transfer": []},
            "empty_solve": {"feasible_azimuths": F0, "arm_ceiling_object_h": ceil_h}, "greedy": [],
            "construction": {"method": METHOD, "kmin": KMIN, "n_max": N_MAX, "attempts": attempt + 1, "cover_cells": [list(x) for x in cover],
                             "cut_count": {str(a): sum(a in cuts[x] for x in pick if tuple(x) in occ) for a in F0},
                             "blocking_cells": {f"{x[0]},{x[1]}": sorted(v) for x, v in cuts.items() if v}, "stats": stats,
                             "construct_s": round(time.time() - t0), "K_by_construction": None,
                             "distractors": {"requested": "dense", "placed": len(objs) - 1 - len(cover_ids)}}}
