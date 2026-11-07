# SHELF66_V3 COPY (2026-11-05) of shelf66_full/: identical except shelf_scene_6x6/shelf_scene.py (the fingers are checked PRE-SHAPED -- GRIP_JOINTS --
# and pre-shaped before every pick) and the version strings (generator SHELF66V3, teacher BD3DS66_TEACHER_V2).  shelf66_full/ = V2, unchanged.
"""SHELF66_FULL COPY (2,500-scene run, generator SHELF66V2) of shelf66/bd_3dt_plus.py.  Changes against the 10-scene sample's SHELF66:
  * distractors: when the requested area (near_T / near_goal / near_path / far) has too few free cells on the 6 x 6 grid, the rest
    come from free cells anywhere on the shelf; if there are still too few, as many as fit are placed (possibly none) -- instead of
    rejecting a finished layout (DISTRACTOR_ROLE_POOL_TOO_SMALL).  The same safety check as before: the minimal repair must still
    relocate independently with the distractors in place (fewer distractors are tried when a draw breaks it).  Every distractor's
    role is the area it really stands in; construction["distractors"] records requested / placed / fallback.
  * K = 3 constructions get 45 min (CONSTRUCT_S_K) instead of 30: one start / goal pair alone takes ~17 min to search.
  * still_feasible plans each direction until one grip height works (the same blocking map, fewer plans).
SHELF66 COPY of bd_samples/bd_3dt_plus.py (grid size from the scene; F0 limits in 45-degree units for 16 directions).
Class-quota extension of bd_3dt (dataset-bd, open tabletop).  dataset-bd's own files are imported, never modified.

Construction for requested K >= 1 ("route" construction): bd_3dt's greedy gives every blocker its own grasp to cut, so one move
reopens a grasp and high K verifies as 1 (or NONE).  Here the cells that block each of the target's feasible side grasps are
PROBED with the scene's planner (one standard blocker at a time), and blockers are added until every feasible grasp is blocked by
at least K of them -- so no set of fewer than K moves can free the target.  K <= 2 adds blockers greedily; K >= 3 searches the
layouts as a minimal repair of K uncrowded blockers plus blockers away from it (covers(); crowded blockers cannot be picked),
with up to 8 blockers for K = 4 (MAX_OBJECTS_K4, teacher pool POOL_CAP_K4).
Every class uses the scene's 8 side-grasp azimuths (every 45 deg).  Parking cells avoid the hand zones of the target
(both sides of every azimuth at its start and goal cell), its carry path and row 0.  K = 0 uses bd_3dt.construct unchanged.
No causal dependence between objects (a limitation of this work): every blocker of the minimal set must be pickable and parkable
from the original scene with everything else in place, so each object moves once and in any order (independent()).
Labels: bd_3dt.teach / record unchanged (verified K, D, B, blocker masks), plus `construction` metadata."""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

import bd_core as bc
import bd_3dt as A

GENERATOR_VERSION = "BD_GEN_V1+ROUTE_V15+SHELF66V3"   # SHELF66: grid size from the scene, azimuth-density-scaled F0 limits; V2: see the header; V3: shelf_scene checks the fingers pre-shaped (GRIP_JOINTS)
MAX_OBJECTS, MAX_OBJECTS_K4 = 7, 9          # obstacles per scene (dataset-bd: 6); K = 4 needs extra blockers away from the repair
POOL_CAP, POOL_CAP_K4 = 6, 8                # objects the teacher's repair search covers (dataset-bd: 5): set per root in construct()
A.POOL_CAP = POOL_CAP
ATTEMPTS = 10                     # start / goal pairs tried per root
PROBE = {"shape": "cylinder", "size": 0.013, "height": 0.12, "color": "red"}     # the smallest intended blocker
N_ALT_PARK = 5                                                                   # parking cells kept per object (solutions)
MAX_F0 = {4: 4}                   # K -> most feasible target azimuths accepted (5+: an uncrowded K=4 repair is rare)
TRACE = None                      # debugging: a list collects per-attempt construction steps
ATTEMPTS_K4 = 80                  # start / goal pairs tried for K = 4 (those with > MAX_F0 feasible azimuths cost ~3 s)
DISTRACTOR_TRIES = 4              # distractor draws tried before a root is rejected (each re-checks the minimal repair)
COVER_CHECKS = 20                 # K >= 3: layouts checked for independent relocation per start / goal pair
MAX_CROWD = 1                     # K >= 3: objects next to the minimal repair's blockers allowed (more: they cannot be picked)
N_PARK_TRY = 3                    # parking cells tried per blocker before it counts as not relocatable
CONSTRUCT_S = 1800.0              # construction wall cap (the teacher needs ~1.5-3 min of the runner's 40 min)
CONSTRUCT_S_K = {3: 2700.0}       # SHELF66V2: K = 3 gets 45 min (one start / goal pair: ~6 min probing + ~10-20 min of candidate checks)


def steps8(a):
    """Unit grid step toward the hand for azimuth a (row = x, col = y)."""
    return int(round(math.cos(math.radians(a)))), int(round(math.sin(math.radians(a))))


def inside(x):
    return 0 <= x[0] < A.GRID["R"] and 0 <= x[1] < A.GRID["C"]


def hand_zone(rc, azimuths=range(0, 360, 45), reach=2):
    return {(rc[0] + k * s[0], rc[1] + k * s[1]) for a in azimuths for s in [steps8(a)] for k in range(1, reach + 1)}


def line(src, goal):
    out = []
    for t in np.linspace(0, 1, 29)[1:-1]:
        x = tuple(int(round(v)) for v in (1 - t) * np.array(src) + t * np.array(goal))
        if x not in out and x not in (src, goal):
            out.append(x)
    return out


def still_feasible(T, src, goal, probe_cell, azimuths):
    """Azimuths (of `azimuths`) whose grasp still plans the target's move with one standard blocker at probe_cell."""
    TS = A.env()["TS"]
    e = A.build([T, {**PROBE, "id": "p"}], {"t0": src, "p": probe_cell})
    i = e.obj_id("t0")
    e.objects[i]["pos"] = e.obj_pos(i)
    q0 = e.q_now()
    ok = []                                           # SHELF66V2: a direction already feasible is not planned again (same set)
    for g in TS.grasp_poses(e.objects[i], e.table_z):
        a = g["azimuth_deg"]
        if a in azimuths and a not in ok and e.plan_pick_place("t0", tuple(goal), g, q_start=q0)["feasible"]:
            ok.append(a)
    return ok


def parking_cells(c0, avoid, occ, n):
    """Free cells for an obstacle from c0: outside `avoid`, rows 1..7; the nearest first, then spread in distance (3, 5, ... cells)."""
    cands = sorted((x for x in itertools.product(range(1, A.GRID["R"]), range(A.GRID["C"])) if x not in avoid and x not in occ),
                   key=lambda x: (A.cheb(x, c0), abs(x[0] - c0[0]), x))
    if not cands:
        return []
    out = [cands[0]]
    for want in (3, 5, 2, 4, 6):                      # meaningful displacements: near, mid, far
        x = next((x for x in cands if A.cheb(x, c0) >= want and x not in out), None)
        if x is not None and len(out) < n:
            out.append(x)
    return out


def independent(objs, cells, park, S, goal):
    """No causal dependence: every object of S can be picked and parked both from the ORIGINAL scene and with all the other objects
    of S already parked (so in any order, each object moved once), and with all of S parked the target's own move plans."""
    e = A.build(objs, cells)
    for o in S:
        for k, state in enumerate((cells, {**cells, **{x: park[x] for x in S if x != o}})):
            A.settle(e, state)
            p, qi = A.query(e, o, park[o])
            if p is None:
                if TRACE is not None:
                    TRACE.append(("dependent", o, cells[o], park[o], ("from_original", "others_parked")[k], qi["grasp_reasons"]))
                return False
    A.settle(e, {**cells, **{o: park[o] for o in S}})
    p, qi = A.query(e, "t0", goal)
    if p is None and TRACE is not None:
        TRACE.append(("target_still_blocked", [cells[o] for o in S], [park[o] for o in S], qi["grasp_reasons"]))
    return p is not None


_REACH = {}


def reach_azimuths():
    """Per cell, the side-grasp azimuths whose pre-grasp and grasp poses the arm reaches (table_scene.reach_map's test, empty table,
    a reference object of mid height); computed once per process."""
    if not _REACH:
        TS = A.env()["TS"]
        e = A.build([{"id": "r", "shape": "cylinder", "size": 0.015, "height": 0.15, "color": "red"}], {"r": (A.GRID["R"] // 2, A.GRID["C"] // 2)})
        ref = dict(e.objects[0])
        for rc in itertools.product(range(A.GRID["R"]), range(A.GRID["C"])):
            ref["pos"] = np.array([*e.cell_xy(rc), e.table_z + ref["height"] / 2])
            _REACH[rc] = set()
            for g in TS.grasp_poses(ref, e.table_z):
                pre = g["position"] - g["approach"] * TS.PRE_DIST
                qa = e.ik_pose(pre, g["R"], e.q_ready, first=True)
                if qa is not None:
                    try:
                        e.chain(qa, pre, [g["position"]], g["R"], step=0.01)
                        _REACH[rc].add(g["azimuth_deg"])
                    except RuntimeError:
                        pass
    return _REACH


def free_side(q, occupied, reach):
    """Cheap pre-check before planning: the object at q has a reachable side grasp whose hand-zone cells (1 and 2 steps out) hold no
    other object."""
    return any(all((q[0] + k * s[0], q[1] + k * s[1]) not in occupied for k in (1, 2)) for a in reach[q] for s in [steps8(a)])


def covers(blocks, F0, K, budget, rng, src, reach=None):
    """Blocker layouts (<= budget cells) that make a K-move repair necessary and possible: Q = K cells cutting one feasible grasp a
    -- the minimal repair, with at most MAX_CROWD objects next to its blockers (counting the target) -- plus R = cells that do not cut
    a and are not next to Q, so that every other feasible grasp is cut at least K times.  Layouts whose other grasps are cut K+1
    times come first (Q is then the only K-move repair), then the least crowded (objects next to Q, then extra blockers two cells
    from Q: those still block Q's pick), then the fewest objects; per Q the 3 smallest R with the most clearance.
    reach (reach_azimuths()): layouts in which a blocker of Q has no free reachable side (free_side()) are dropped before planning."""
    cut = {x: blocks[x] & set(F0) for x in blocks if blocks[x] & set(F0)}
    found = []
    for a in F0:
        for Q in itertools.combinations([x for x in cut if a in cut[x]], K):
            crowd = sum(A.cheb(q, y) == 1 for q in Q for y in Q + (src,) if y != q)
            if crowd > MAX_CROWD:
                continue
            if reach and not all(free_side(q, set(Q) - {q} | {src}, reach) for q in Q):
                continue
            n = {b: sum(b in cut[x] for x in Q) for b in F0 if b != a}
            free = [x for x in cut if a not in cut[x] and all(A.cheb(x, q) >= 2 for q in Q)]
            hit = []
            for extra in (1, 0):                      # other grasps cut K+1 times: Q is the only K-move repair
                for r in range(budget - K + 1):
                    hit = [R for R in itertools.combinations(free, r) if all(n[b] + sum(b in cut[x] for x in R) >= K + extra for b in n)
                           and (not reach or all(free_side(q, set(Q + R) - {q} | {src}, reach) for q in Q))]
                    if hit:
                        near2 = {R: sum(A.cheb(q, y) == 2 for q in Q for y in R) for R in hit}
                        hit.sort(key=near2.get)
                        found += [(1 - extra, crowd, near2[R], K + r, rng.random(), list(Q) + list(R)) for R in hit[:3]]
                        break
                if hit:
                    break
    return [S for *_, S in sorted(found)]


def construct_route(spec, seed, root_id) -> dict:
    t_start = time.time()
    rng = np.random.default_rng(seed)
    K = spec["K"]
    cap = CONSTRUCT_S_K.get(K, CONSTRUCT_S)                 # SHELF66V2: construction wall cap per K
    lo, hi = A.DISP[spec["displacement"]]
    T = {"id": "t0", "shape": "cylinder", "size": float(rng.uniform(0.016, 0.020)), "height": float(rng.uniform(0.12, 0.16)), "color": "green"}
    lo_d, hi_d = bc.N_DISTRACT[spec["distractors"]]
    n_obj = MAX_OBJECTS_K4 if K >= 4 else MAX_OBJECTS
    budget = min(n_obj - lo_d, A.POOL_CAP)                  # the teacher searches over at most POOL_CAP objects
    dims = []
    for j in range(n_obj):
        cyl = bool(rng.integers(2))
        dims.append({"id": f"o{j:02d}", "shape": "cylinder" if cyl else "box", "size": float(rng.uniform(0.013, 0.018) if cyl else rng.uniform(0.012, 0.016)),
                     "height": float(rng.uniform(0.12, 0.18)), "color": A.COLORS[j % len(A.COLORS)]})
    stats = {"probed": 0, "covers": 0, "checked": 0, "unpickable_cells": 0}
    for attempt in range(ATTEMPTS_K4 if K >= 4 else ATTEMPTS):
        if time.time() - t_start > cap:
            raise A.Reject("ROUTE_TIME_CAP:" + ",".join(f"{k}={v}" for k, v in stats.items()))
        src = (int(rng.choice(A.ROWS[spec["source"]])), int(rng.integers(1, A.GRID["C"] - 1)))
        goals = [(r, c) for r in range(1, A.GRID["R"]) for c in range(A.GRID["C"]) if lo <= A.cheb((r, c), src) <= hi]
        if not goals:
            continue
        goal = goals[int(rng.integers(len(goals)))]
        F0 = A.feasible_grasps([T], {"t0": src}, "t0", goal)
        if TRACE is not None:
            TRACE.append(("F0", src, goal, F0))
        eff = len(F0) * 8 / len(A.env()["scene"]["azimuths"])     # SHELF66: feasible directions in 45-degree units (16 directions: 22.5 deg apart)
        if not F0 or eff * K > 3 * budget or eff > MAX_F0.get(K, 8):   # far too many grasps to block with <= budget objects
            continue
        zone = set()
        for a in F0:
            zone |= hand_zone(src, [a]) | hand_zone(goal, [a])
        probe = sorted(x for x in (zone | hand_zone(src, reach=1) | hand_zone(goal, reach=1) | set(line(src, goal)))
                       if inside(x) and x not in (src, goal))
        blocks = {x: set(F0) - set(still_feasible(T, src, goal, x, F0)) for x in probe}
        stats["probed"] += 1
        route = set(line(src, goal)) | {(r + a, c + b) for r, c in line(src, goal) for a in (-1, 0, 1) for b in (-1, 0, 1)}
        base_avoid = {src, goal} | hand_zone(src) | hand_zone(goal) | route | {(0, c) for c in range(A.GRID["C"])}
        memo, chosen, bad = {}, {}, set()

        def placements(cells_, movers=None):
            """((blocker ids -> cells, parking cell per mover, parking cells taken, avoid), None) when every mover's pick-and-place
            plans from this scene (target and all blockers in place; N_PARK_TRY parking cells tried, never next to a blocker),
            else (None, the first mover's cell that cannot be relocated).  movers: blocker ids (default: all of them)."""
            key = (tuple(cells_), movers)
            if key in memo:
                return memo[key]
            avoid_ = base_avoid | set().union(*(hand_zone(x, reach=1) for x in cells_))
            bl_ = {dims[j]["id"]: x for j, x in enumerate(cells_)}
            sc = {"t0": list(src), **{o: list(x) for o, x in bl_.items()}}
            e, first_, taken_ = A.build([T] + dims[:len(cells_)], sc), {}, set()
            for o, x in bl_.items():
                if movers is not None and o not in movers:
                    continue
                spaced = set(cells_) | taken_ | {y for t in taken_ for y in hand_zone(t, reach=1)}   # parked movers never side by side
                first_[o], why = None, {}
                for c in parking_cells(x, avoid_, spaced, N_PARK_TRY):
                    A.settle(e, sc)
                    p, qi = A.query(e, o, list(c))
                    if p is not None:
                        first_[o] = list(c)
                        break
                    why = qi["grasp_reasons"]
                if first_[o] is None:
                    if TRACE is not None:
                        TRACE.append(("unpickable", x, len(cells_), why))
                    memo[key] = None, x
                    return memo[key]
                taken_.add(tuple(first_[o]))
            memo[key] = (bl_, first_, taken_, avoid_), None
            return memo[key]

        def pickable(cells_):
            return placements(cells_)[0] is not None

        def minimal_independent(cells_):
            """A tight grasp's K blockers -- a minimal repair; the other blockers stay -- each relocate once, independently (from the
            original scene and with the other K-1 parked), and then the target's move plans."""
            bl_ = {dims[j]["id"]: x for j, x in enumerate(cells_)}
            n = {a: sum(a in blocks[x] for x in cells_) for a in F0}
            for S in dict.fromkeys(tuple(o for o, x in bl_.items() if a in blocks[x]) for a in F0 if n[a] == min(n.values())):
                res, x = placements(cells_, S)
                if res is None:
                    if placements([x])[0] is None:    # not even with only the target present: no cover with this cell
                        bad.add(x)
                        stats["unpickable_cells"] += 1
                    continue
                if independent([T] + dims[:len(cells_)], {"t0": list(src), **{o: list(c) for o, c in bl_.items()}}, res[1], S, goal):
                    chosen[tuple(cells_)] = S
                    return True
            return False
        if K >= 3:                                    # exact: the covers with the least crowded minimal repair first
            cands = covers(blocks, F0, K, budget, rng, src, reach_azimuths())
            stats["covers"] += len(cands)
            if TRACE is not None:
                TRACE.append(("covers", F0, len(cands), cands[:COVER_CHECKS]))
            pick, checked = None, 0
            for S in cands:
                if checked >= COVER_CHECKS or time.time() - t_start > cap:
                    break
                if bad & set(S):
                    continue
                checked += 1
                stats["checked"] += 1
                if minimal_independent(S):
                    pick = S
                    break
            if pick is not None:
                search = "exact_cover"
                break
            if K >= 4:                                # the greedy below never gave an independent K = 4; K = 3 falls back to it
                continue
        need, pick, refused = {a: K for a in F0}, [], set()
        while any(v > 0 for v in need.values()) and len(pick) < budget:
            score = lambda x: (sum(need[a] > 0 for a in blocks[x]), -A.cheb(x, src))  # noqa: E731
            opts = sorted((x for x in probe if x not in pick and x not in refused and score(x)[0] > 0), key=score, reverse=True)
            x = next((x for x in opts[:6 if K < 4 else 12] if pickable(pick + [x]) or refused.add(x)), None)   # independence while building
            if x is None:
                break
            pick.append(x)
            for a in blocks[x]:
                need[a] -= 1
        if TRACE is not None:
            TRACE.append(("greedy", {a: [x for x in probe if a in blocks[x]] for a in F0}, dict(need), pick, sorted(refused)))
        if any(v > 0 for v in need.values()) or not minimal_independent(pick):
            continue
        search = "greedy"
        break
    else:
        raise A.Reject("ROUTE_BLOCKING_NOT_FOUND:" + ",".join(f"{k}={v}" for k, v in stats.items()))
    bl, first, taken, avoid = placements(pick, chosen[tuple(pick)])[0]      # parking checked for the minimal repair's blockers
    for j in range(len(pick), n_obj):
        dims[j]["height"] = float(rng.uniform(0.06, 0.18))
    allc = [(r, c) for r in range(A.GRID["R"]) for c in range(A.GRID["C"])]
    role_cells = {"near_T": [x for x in allc if 1 <= A.cheb(x, src) <= 2], "near_goal": [x for x in allc if 1 <= A.cheb(x, goal) <= 2],
                  "near_path": [x for x in allc if x in route], "far": [x for x in allc if A.cheb(x, src) >= 3 and A.cheb(x, goal) >= 3 and x not in route]}
    used = set(pick) | {src, goal} | taken
    movers = chosen[tuple(pick)]                      # the minimal repair's blockers, and the grasps it frees
    tight = {a for a in F0 if {o for o, x in bl.items() if a in blocks[x]} == set(movers)}
    near_movers = set().union(*(hand_zone(bl[o], reach=1) | hand_zone(tuple(first[o]), reach=1) for o in movers))
    base = [x for x in set(role_cells[spec["distractor_role"]]) if x not in used and x not in zone and x not in near_movers]
    strict = sorted(x for x in base if not blocks.get(x))          # cuts no feasible grasp
    loose = sorted(x for x in base if not blocks.get(x, set()) & tight)   # cuts no grasp the repair frees (else K would rise)
    want = min(int(rng.integers(lo_d, hi_d + 1)), n_obj - len(pick))
    dpool = strict if len(strict) >= want else loose
    fallback = None
    if len(dpool) < want:                             # SHELF66V2: requested area too small on the 6 x 6 grid -> free cells anywhere
        rest = [x for x in allc if x not in base and x not in used and x not in zone and x not in near_movers]
        r_strict = sorted(x for x in rest if not blocks.get(x))
        r_loose = sorted(x for x in rest if not blocks.get(x, set()) & tight)
        extra = r_strict if len(dpool) + len(r_strict) >= want else r_loose
        dpool = dpool + [x for x in extra if x not in dpool]
        fallback = "other_area" if extra else None
    nd = min(want, len(dpool))
    if nd < want:                                     # SHELF66V2: as many as fit (possibly none) -- was DISTRACTOR_ROLE_POOL_TOO_SMALL
        fallback = "fewer"
    home = [x for x in dpool if x in base]            # cells of the requested area are drawn first
    away = [x for x in dpool if x not in base]
    tries = 0
    while True:                                       # the minimal repair must still relocate independently with the distractors in place
        k1 = min(nd, len(home))
        dis = ([home[i] for i in rng.choice(len(home), k1, replace=False)] if k1 else []) + \
              ([away[i] for i in rng.choice(len(away), nd - k1, replace=False)] if nd - k1 else [])
        objs, cells = [T] + dims[:len(pick) + nd], {"t0": list(src)}
        for o, rc in zip(objs[1:], pick + dis):
            cells[o["id"]] = [int(rc[0]), int(rc[1])]
        if not nd or independent(objs, cells, {o: first[o] for o in movers}, movers, goal):
            break
        tries += 1
        if tries >= (DISTRACTOR_TRIES if fallback != "fewer" else 2):   # SHELF66V2: one distractor fewer -- was DISTRACTORS_BREAK_REPAIR
            nd, tries, fallback = nd - 1, 0, "fewer"

    def area(x):                                      # SHELF66V2: the area a distractor really stands in (the requested one if it is there)
        if x in role_cells[spec["distractor_role"]]:
            return spec["distractor_role"]
        return next((k for k in ("near_path", "near_T", "near_goal", "far") if x in role_cells[k]), "far")
    dinfo = {"requested": want, "placed": nd, "requested_area": spec["distractor_role"], "areas": [area(x) for x in dis], "fallback": fallback}
    roles = {o["id"]: ("intended_blocker" if j < len(pick) else f"distractor:{area(dis[j - len(pick)])}") for j, o in enumerate(objs[1:])}
    occ = {tuple(v) for v in cells.values()}
    park, alt, taken = {}, {}, {tuple(v) for v in first.values()}
    for o in objs[1:]:
        cands = parking_cells(tuple(cells[o["id"]]), avoid, occ | (taken - {tuple(first.get(o["id"], (-1, -1)))}), N_ALT_PARK)
        if o["id"] in first:                          # the blockers keep the parking cell their relocation was checked with
            cands = [tuple(first[o["id"]])] + [x for x in cands if x != tuple(first[o["id"]])][:N_ALT_PARK - 1]
        alt[o["id"]] = [list(x) for x in cands]
        park[o["id"]] = alt[o["id"]][:bc.PARKING_N[spec["parking"]]]
        if cands:
            taken.add(cands[0])
    wrong = [list(x) for x in line(src, goal) if x not in occ][:2]
    pools = {"route": [list(x) for x in probe], "grasp": [list(x) for x in pick if A.cheb(x, src) <= 2],
             "place": [list(x) for x in pick if A.cheb(x, goal) <= 2], "transfer": [list(x) for x in pick if x in route]}
    return {"objs": objs, "cells": cells, "src": list(src), "goal": list(goal), "roles": roles, "parking": park, "alt_parking": alt,
            "intended": [o["id"] for o in objs[1:len(pick) + 1]], "wrong_dest": wrong, "pools": pools, "empty_solve": {"feasible_azimuths": F0},
            "greedy": [{"cell": list(x), "blocks_azimuths": sorted(blocks[x]), "necessary_by_construction": True} for x in pick],
            "construction": {"method": "route", "search": search, "attempts": attempt + 1, "blocking_cells": {f"{x[0]},{x[1]}": sorted(v) for x, v in blocks.items() if v},
                             "K_by_construction": min(sum(a in blocks[x] for x in pick) for a in F0), "distractors": dinfo}}


def construct(spec, seed, root_id) -> dict:
    A.POOL_CAP = POOL_CAP_K4 if spec["K"] >= 4 else POOL_CAP     # kept for this root's teacher and record
    if spec["K"] >= 1:
        return construct_route(spec, seed, root_id)
    c = A.construct(spec, seed, root_id)
    occ = {tuple(v) for v in c["cells"].values()}
    src, goal = tuple(c["src"]), tuple(c["goal"])
    avoid = {src, goal} | hand_zone(src) | hand_zone(goal) | set(line(src, goal)) | {(0, k) for k in range(A.GRID["C"])}
    c["alt_parking"] = {o: [list(x) for x in parking_cells(tuple(c["cells"][o]), avoid, occ, N_ALT_PARK)] for o in c["roles"]}
    c["construction"] = {"method": "bd_3dt"}
    return c


def record(spec, ids, c, t, store) -> dict:
    rec = A.record(spec, ids, c, t, store)
    rec.update(alt_parking=c["alt_parking"], construction=c["construction"], generator_version_plus=GENERATOR_VERSION,
               grasp_azimuths=A.env()["TS"].N_AZIMUTH)
    return rec
