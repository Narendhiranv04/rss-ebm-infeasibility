"""Class-quota extension of bd_2d (dataset-bd, planar 3R arm).  dataset-bd and latent-causal-tamp are imported, never modified.

Construction for requested K >= 1 ("route" construction): bd_2d chases the target's witness with one blocker per route, so one move
reopens a route and high K verifies as 0/1.  Here blockers are added on the sweep of the target's current witness until no set of
fewer than K of them, moved to their parking cells, lets the target's move verify; parking cells avoid every target sweep seen and
the cells next to blockers.  K = 0 uses bd_2d.construct unchanged.

V2: requested K >= 2 with mechanism grasp / place / mixed (and K=4 specs with intent "k4") come from the exact layout search of
bd_2d_exact (static grasp-screen cut maps, exact cover, up to 9-11 movable objects; V3: at most 3 immovable structures, movable
blockers preferred; V4: no immovable structures at all, route construction included; V5: NONE from movable blockers only
(bd_2d_exact.construct_none: >= 5 relocations needed); V6: an extra blocker chosen by both ends is one shared object instead of a
rejected layout, K=4 movers need 1 spare grasp, up to 14 / 20 objects for K=3 / K=4); K=4 specs with intent
"none" (or none given) keep the NONE construction, K = 1 and transfer K = 2 keep the route construction.

No causal dependence between objects (a limitation of this work): a verified minimal repair is kept only if every one of its
relocations is certified from the ORIGINAL scene (robot at q0, every other object in place) -- each object moves once, in any
order (independent()).  Teacher and record: bd_2d's (repair search over POOL_CAP objects, raised from 4)."""
from __future__ import annotations

import itertools
from dataclasses import replace

import numpy as np

import bd_core as bc
import bd_2d as A

GENERATOR_VERSION = "BD_GEN_V1+ROUTE2D_V6"
MAX_OBJECTS = 7                   # obstacles per scene (dataset-bd: 6)
POOL_CAP, ADMIN_CPU_S = 5, 1200.0  # teacher's repair pool (dataset-bd: 4) and per-root CPU cap (dataset-bd: 900 s); read by
A.POOL_CAP, A.ADMIN_CPU_S = POOL_CAP, ADMIN_CPU_S   # bd_2d.teach at call time
MAX_OBJECTS_EXACT = {2: 10, 3: 14, 4: 20}  # movable objects of an exact layout (must-move + extra blockers + distractors); V4:
                                           # no immovable structures, so movable blockers close every other grasp
ADMIN_CPU_EXACT = 1500.0          # teacher CPU cap for exact layouts
MAX_OBJECTS_NONE = 20             # V5 NONE: ~16-17 movable blockers make every grasp pair need >= 5 relocations (+ distractors)
ATTEMPTS = 6
FIXED_MIN, FIXED_MAX = 0, 0         # immovable context structures next to the start / goal (bd_2d: 1-3; V4: none -- user, 1 Nov)


def independent(c, root_id, moves):
    """Every (object, cell) of `moves` has a certified relocation edge from the original scene."""
    import p16_common as pc
    from scene import scene_to_dict
    s = c["scene"]
    task = {"scene": scene_to_dict(s), "goal_cell": c["goal"], "fixed_ids": c["fixed_ids"], "target": "t0", "key": root_id}
    phys, cells0, q0 = A._phys(task, {}), {o.id: o.cell for o in s.objects}, np.asarray(c["q0"], float)
    return all(pc.best_edge(phys.edges(cells0, q0, o, int(d))) is not None for o, d in moves)


def construct_route(spec, seed, root_id) -> dict:
    import p14_sweep_index as sx
    import p6_scenes as p6
    from scene import place_scene, scene_to_dict
    G = A.env()
    cfg, robot = G["cfg"], G["robot"]
    rng = np.random.default_rng(seed)
    K = spec["K"]
    lo, hi = A.DISP[spec["displacement"]]
    stratum = "S0" if spec["start_pose"] == "home" else str(rng.choice(["S1", "S2", "S3"]))
    q0 = tuple(p6.STRATA[stratum])
    qa = np.asarray(q0, float)
    lo_d, hi_d = bc.N_DISTRACT[spec["distractors"]]
    budget = min(MAX_OBJECTS - lo_d, A.POOL_CAP)

    def task_of(src, goal, cells, tag):
        s = place_scene(cfg, root_id, tag, [src], cells)
        s = replace(s, robot=replace(s.robot, q_rad=q0))
        return s, {"scene": scene_to_dict(s), "goal_cell": goal, "fixed_ids": [], "target": "t0", "key": f"{root_id}:{tag}"}

    queries = 0
    for attempt in range(ATTEMPTS):
        srcs = [x for x in G["region"][spec["source"]] if any(lo <= A.cheb(c, x) <= hi for c in G["valid"])]
        src = int(rng.choice(srcs))
        goal = int(rng.choice([c for c in G["valid"] if lo <= A.cheb(c, src) <= hi]))
        _, t_empty = task_of(src, goal, [], f"empty{attempt}")
        st, a_empty, _ = A.t_goal(t_empty, {}, {"t0": src}, qa)
        queries += 1
        if st != "VERIFIED":
            continue
        sw_empty = sx.swept_cells(robot, cfg, a_empty)
        seen = set().union(*map(set, sw_empty.values()))
        ok = lambda c: c in G["valid"] and c not in (src, goal)  # noqa: E731
        # immovable context structures next to the start / goal, off the empty-scene sweep (bd_2d's convention): they close some
        # access directions so that the movable blockers decide; never movers, never in the repair search
        anchors = {"grasp": [src], "place": [goal], "transfer": [src, goal], "mixed": [src, goal]}[spec["mechanism"]]
        nb = sorted({c for a_ in anchors for c in range(64) if A.cheb(c, a_) == 1 and c not in (src, goal)})
        off = [c for c in nb if c not in seen] or nb
        fixed = [int(x) for x in rng.choice(off, min(int(rng.integers(FIXED_MIN, FIXED_MAX + 1)), len(off)), replace=False)]
        s_fixed, t_fixed = task_of(src, goal, fixed, f"fixed{attempt}")
        st, a, _ = A.t_goal(t_fixed, {}, {o.id: o.cell for o in s_fixed.objects}, qa)
        queries += 1
        if st != "VERIFIED":
            continue
        if K >= 4:                                    # NONE: immovable structures on the target's sweep next to its start / goal until
            for it in range(8):                       # no move of the target verifies; movable objects then cannot repair it
                sw = sx.swept_cells(robot, cfg, a)
                seen |= set().union(*map(set, sw.values()))
                near = sorted(c for c in set().union(*map(set, sw.values())) if ok(c) and c not in fixed and min(A.cheb(c, x) for x in anchors) <= 1)
                if not near:
                    break
                fixed.append(int(rng.choice(near)))
                s_fixed, t_fixed = task_of(src, goal, fixed, f"none{attempt}.{it}")
                st, a, _ = A.t_goal(t_fixed, {}, {o.id: o.cell for o in s_fixed.objects}, qa)
                queries += 1
                if st != "VERIFIED":
                    break
            if st == "VERIFIED":
                continue
            cand = sorted(c for c in G["valid"] if ok(c) and c not in fixed and 1 <= min(A.cheb(c, x) for x in anchors) <= 2)
            if len(cand) < 2:
                continue
            pick = [int(x) for x in rng.choice(cand, min(len(cand), int(rng.integers(2, 4))), replace=False)]
            s, _ = task_of(src, goal, pick + fixed, f"nonepick{attempt}")
            ids = [o.id for o in s.objects if o.id != "t0"][:len(pick)]
            park = parking(G, src, goal, seen, pick + fixed, {o.id: o.cell for o in s.objects}, ids)
            break
        pick, done = [], False
        for it in range(3 * K + 4):
            if pick:
                s, task = task_of(src, goal, pick + fixed, f"r{attempt}.{it}")
                ids = [o.id for o in s.objects if o.id != "t0"][:len(pick)]          # the movable blockers (fixed ones come last)
                cells0 = {o.id: o.cell for o in s.objects}
                park = parking(G, src, goal, seen, pick + fixed, cells0, ids)
                cache, a = {}, None
                st, a0, _ = A.t_goal(task, cache, cells0, qa)
                queries += 1
                if st == "VERIFIED":
                    a = a0
                else:                                         # blocked: does a set of fewer than K moves free the target?
                    for k in range(1, K):
                        for R in itertools.combinations(ids, k):
                            st2, a2, _ = A.t_goal(task, cache, {**cells0, **{o: park[o] for o in R}}, qa)
                            queries += 1
                            if st2 == "VERIFIED":
                                a = a2
                                break
                        if a is not None:
                            break
                    if a is None:
                        done = True
                        break
            if len(pick) >= budget:
                break
            sw = sx.swept_cells(robot, cfg, a)
            seen |= set().union(*map(set, sw.values()))
            pools = {"grasp": {c for ph in ("pregrasp", "approach", "close_attach") for c in sw[ph] if A.cheb(c, src) <= 2 and ok(c)},
                     "place": {c for ph in ("carry", "release", "retreat") for c in sw[ph] if A.cheb(c, goal) <= 2 and ok(c)},
                     "transfer": {c for ph in ("pregrasp", "carry") for c in sw[ph] if A.cheb(c, src) >= 2 and A.cheb(c, goal) >= 2 and ok(c)}}
            m = spec["mechanism"] if spec["mechanism"] != "mixed" else ("grasp", "place", "transfer")[len(pick) % 3]
            opts = sorted(pools[m] - set(pick) - set(fixed)) or sorted(set().union(*pools.values()) - set(pick) - set(fixed))
            if not opts:
                break
            pick.append(int(rng.choice(opts)))
        if done and len(pick) <= budget:
            break
    else:
        raise A.Reject("ROUTE_BLOCKING_NOT_FOUND")
    near_sw = {c for c in G["valid"] for x in seen if A.cheb(c, x) == 1}
    first = set(park.values())
    for r in (2, 3):                                  # near rings crowded by the fixed structures: widen to distance 3
        role_cells = {"near_T": [c for c in G["valid"] if 1 <= A.cheb(c, src) <= r and c not in seen],
                      "near_goal": [c for c in G["valid"] if 1 <= A.cheb(c, goal) <= r and c not in seen],
                      "near_path": [c for c in near_sw if c not in seen],
                      "far": [c for c in G["valid"] if A.cheb(c, src) >= 3 and A.cheb(c, goal) >= 3 and c not in seen]}
        dpool = sorted({c for c in role_cells[spec["distractor_role"]] if ok(c) and c not in pick and c not in first and c not in fixed})
        if len(dpool) >= lo_d:
            break
    nd = min(int(rng.integers(lo_d, hi_d + 1)), MAX_OBJECTS - len(pick), len(dpool))
    if nd < lo_d:
        raise A.Reject("DISTRACTOR_ROLE_POOL_TOO_SMALL")
    dis = [int(x) for x in rng.choice(dpool, nd, replace=False)] if nd else []
    s = place_scene(cfg, root_id, "orig", [src], pick + dis + fixed)
    s = replace(s, robot=replace(s.robot, q_rad=q0))
    allids = [o.id for o in s.objects if o.id != "t0"]
    ids, fixed_ids = allids[:len(pick) + nd], allids[len(pick) + nd:]
    cells0 = {o.id: o.cell for o in s.objects}
    roles = {i: ("intended_blocker" if j < len(pick) else f"distractor:{spec['distractor_role']}") for j, i in enumerate(ids)}
    npark = bc.PARKING_N[spec["parking"]]
    park_all = parking(G, src, goal, seen, pick + dis + fixed, cells0, ids, n=max(npark, 5))
    park = {o: [{"cell": int(c), "in_target_sweep": c in seen} for c in v[:npark]] for o, v in park_all.items()}
    wrong = sorted((c for c in seen if c in G["valid"] and c not in set(pick + dis + fixed + [src]) and c != goal), key=lambda c: (A.cheb(c, src), c))
    return {"scene": s, "goal": goal, "q0": q0, "stratum": stratum, "roles": roles, "fixed_ids": fixed_ids, "fixed_cells": fixed, "parking": park,
            "alt_parking": {o: [int(c) for c in v] for o, v in park_all.items()}, "wrong_dest": wrong[:2], "empty_witness": a_empty,
            "swept": {k: list(v) for k, v in sw_empty.items()}, "intended": ids[:len(pick)], "pools": {"route": sorted(seen)},
            "chase_queries": queries, "blockers_only_scene_blocks_T": K < 4,
            "construction": {"method": "route_none" if K >= 4 else "route", "attempts": attempt + 1, "role_radius": r}}


def parking(G, src, goal, seen, occupied, cells0, ids, n=1):
    """Parking cells per object: free valid cells >= 2 from the target's start and goal, off every target sweep seen and not next to
    another object; nearest first, then spread in distance (meaningful displacements).  n = 1: {id: cell}; else {id: [cells]}."""
    occ, taken, out = set(occupied) | {src, goal}, set(), {}
    near = {c for x in occupied for c in G["valid"] if A.cheb(c, x) == 1}
    for o in ids:
        c0 = cells0[o]
        cands = sorted((c for c in G["valid"] if c not in occ and c not in taken and A.cheb(c, src) >= 2 and A.cheb(c, goal) >= 2
                        and c not in seen and c not in near), key=lambda c: (A.cheb(c, c0), c))
        if not cands:                                  # crowded board: allow sweep / neighbour cells
            cands = sorted((c for c in G["valid"] if c not in occ and c not in taken and A.cheb(c, src) >= 2 and A.cheb(c, goal) >= 2),
                           key=lambda c: (A.cheb(c, c0), c))
        if not cands:
            raise A.Reject("NO_PARKING_CELL")
        pick = [cands[0]] + [next((c for c in cands if A.cheb(c, c0) >= w and c != cands[0]), None) for w in (3, 5, 2, 4)]
        pick = list(dict.fromkeys(c for c in pick if c is not None))[:n]
        taken.add(pick[0])
        out[o] = pick[0] if n == 1 else pick
    return out


def construct(spec, seed, root_id) -> dict:
    import bd_2d_exact as E
    A.POOL_CAP, A.ADMIN_CPU_S = POOL_CAP, ADMIN_CPU_S
    exact = spec["K"] >= 2 and (spec["K"] < 4 or spec.get("intent") == "k4") and spec["mechanism"] in E.SPLITS
    if spec.get("intent") == "k4" and not exact:
        raise A.Reject("K4_NEEDS_GRASP_PLACE_OR_MIXED")
    if spec["K"] >= 4 and spec.get("intent") != "k4":   # V5: NONE from movable blockers only (>= 5 relocations needed)
        c = E.construct_none(spec, seed, root_id, MAX_OBJECTS_NONE)
        A.POOL_CAP = 4                                   # repairs of <= 4 relocations (the dataset's K <= 4); none exists
        return c
    if exact:
        c = E.construct_exact(spec, seed, root_id, MAX_OBJECTS_EXACT[spec["K"]])
        # teacher pool: the K must-move objects first, then extra blockers up to K + 3 (no smaller repair exists among ANY objects:
        # static cut maps), so the verified K is exact while the subset lattice stays small
        A.POOL_CAP, A.ADMIN_CPU_S = min(len(c["intended"]), max(POOL_CAP, spec["K"] + 3)), ADMIN_CPU_EXACT
        return c
    if spec["K"] >= 1:
        return construct_route(spec, seed, root_id)
    c = A.construct(spec, seed, root_id)
    c["construction"] = {"method": "bd_2d"}
    return c


def record(spec, ids, c, t, store) -> dict:
    rec = A.record(spec, ids, c, t, store)
    rec.update(construction={**rec["construction"], **c["construction"]}, alt_parking=c.get("alt_parking"), generator_version_plus=GENERATOR_VERSION)
    return rec
