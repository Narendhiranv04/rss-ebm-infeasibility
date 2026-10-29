"""2D domain adapter for the B/D dataset (bd_core): the project's planar 3R arm, 8x8 grid (0-based cell = row*8 + col, row 0 = y low),
side-grasp skill (declared grasp set of skill.py), disc objects. Offline teacher = p16_motion.DetPhysics (library lookup + certified
planner + scoped certificates), FRESH fixed budget per query, one exact shared cache per root (identical sub-queries are reused, never
re-searched with a larger budget).

construct(spec, seed, root_id) builds one root from requested factors; teach(c) labels it. Requested K is a construction INTENT only: the
observed stratum is the verified K found by the declared finite repair domain (pool <= 4 objects, first parking cell, <= 6 orders)."""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import replace

import numpy as np

import bd_core as bc
import config as cf

DOMAIN = "2d"
TEACHER_VERSION = "BD2D_TEACHER_V1:DetPhysics(P16_FAST,no-library)"
QB = {"motion_checks": 150, "online_moves": 8, "goal_plans": 3, "transit_plans": 6}       # per-query budget (declared, fixed)
ADMIN_CPU_S = 900.0                                                                      # per-root administrative cap (never a label)
MAX_SEQ_TRIES, MAX_ORDERS, POOL_CAP = 14, 6, 4
_G = {}


class Reject(Exception):
    pass


class AdminCap(Exception):
    pass


def env():
    if not _G:
        import geometry as ge
        import p6_scenes as p6
        from scene import robot_profile_hash
        cfg = cf.load_config("config.toml")
        robot = ge.make_robot(cfg)
        scr = p6.Screens(cfg, robot)
        valid = sorted(scr.valid)
        d = {c: math.dist(((c % 8 + 0.5) * 0.08, (c // 8 + 0.5) * 0.08), robot.base) for c in valid}
        srt = sorted(valid, key=lambda c: (d[c], c))
        n = len(srt)
        _G.update(cfg=cfg, robot=robot, scr=scr, valid=valid, region={"near": srt[:n // 3], "mid": srt[n // 3:2 * n // 3], "far": srt[2 * n // 3:]},
                  protocol=f"PLANAR3R_SIDEGRASP_V1:{robot_profile_hash(cfg)}", config_hash=bc.sha({"cfg": cfg, "QB": QB, "v": TEACHER_VERSION})[:16])
    return _G


def cheb(a, b):
    return max(abs(a // 8 - b // 8), abs(a % 8 - b % 8))


DISP = {"short": (1, 2), "medium": (3, 4), "long": (5, 7)}


def supports(spec) -> bool:
    """Spec realisable on the 2D board: some source in the region has a valid goal in the displacement band."""
    G = env()
    lo, hi = DISP[spec["displacement"]]
    return any(lo <= cheb(c, s) <= hi for s in G["region"][spec["source"]] for c in G["valid"])


def _phys(task, cache):
    import p16_motion as pm
    return pm.DetPhysics(task, None, budget=dict(QB), cache=cache)


def t_goal(task, cache, cells, q):
    """Original-target query. With no object besides T the historical sector certificate cannot run (empty-array bug in
    sector_cert._sector), and no blocker certificate applies, so the certified skill planner itself is called (same FAST budget)."""
    ph = _phys(task, cache)
    if len(cells) > 1:
        return ph.goal(cells, q)
    import skill as sk
    from scene import scene_from_dict
    r = sk.plan_skill(ph.robot, scene_from_dict(task["scene"]), "t0", task["goal_cell"], ph.cfg, f"bd2d/{task['key']}", q_start=q, q_goal=None)
    if r.status == sk.VERIFIED and r.traj is not None:
        return "VERIFIED", r.traj.arrays(), 0.0
    return ("CERTIFIED_INFEASIBLE" if r.status == sk.INFEASIBLE else "UNKNOWN_PLANNER"), {"planner": r.status, "reason": r.reason}, 0.0


def _neg(status):
    if status in ("CERTIFIED_INFEASIBLE", "STATIC_INVALID"):
        return "CERTIFIED"
    return "TIMEOUT" if status == "UNKNOWN_BUDGET" else "NO_PLAN_FOUND"


def label(status, replay=True, **meta):
    ok = status == "VERIFIED" and bool(replay)
    return bc.binary(ok, status, None if ok else _neg(status), **meta)


def construct(spec, seed, root_id) -> dict:
    """Factor-controlled root. Raises Reject(reason) for a recorded construction rejection."""
    import p14_sweep_index as sx
    import p6_scenes as p6
    from scene import place_scene, scene_to_dict
    G = env()
    cfg, robot = G["cfg"], G["robot"]
    rng = np.random.default_rng(seed)
    lo, hi = DISP[spec["displacement"]]
    srcs = [x for x in G["region"][spec["source"]] if any(lo <= cheb(c, x) <= hi for c in G["valid"])]
    src = int(rng.choice(srcs))
    goals = [c for c in G["valid"] if lo <= cheb(c, src) <= hi]
    if not goals:
        raise Reject("NO_GOAL_IN_DISPLACEMENT_BAND")
    goal = int(rng.choice(goals))
    stratum = "S0" if spec["start_pose"] == "home" else str(rng.choice(["S1", "S2", "S3"]))
    q0 = tuple(p6.STRATA[stratum])
    empty = place_scene(cfg, root_id, "empty", [src], [])
    empty = replace(empty, robot=replace(empty.robot, q_rad=q0))
    task0 = {"scene": scene_to_dict(empty), "goal_cell": goal, "fixed_ids": [], "target": "t0", "key": root_id + ":empty"}
    st, a, _ = t_goal(task0, {}, {"t0": src}, np.asarray(q0, float))
    if st != "VERIFIED":
        raise Reject(f"T_NOT_VERIFIED_IN_EMPTY_SCENE:{st}")
    sw = sx.swept_cells(robot, cfg, a)
    allsw = set().union(*map(set, sw.values()))
    ok = lambda c: c in G["valid"] and c not in (src, goal)  # noqa: E731

    def mech_pools(swc):
        return {"grasp": {c for ph in ("pregrasp", "approach", "close_attach") for c in swc[ph] if cheb(c, src) <= 2 and ok(c)},
                "place": {c for ph in ("carry", "release", "retreat") for c in swc[ph] if cheb(c, goal) <= 2 and ok(c)},
                "transfer": {c for ph in ("pregrasp", "carry") for c in swc[ph] if cheb(c, src) >= 2 and cheb(c, goal) >= 2 and ok(c)}}
    K = spec["K"]
    fixed = []                       # immovable context structures (historical P13-P16 convention): close some access directions so
    if K:                            # that the movable blockers decide; never movers, never in the blocker pool, not counted as movable
        anchors = {"grasp": [src], "place": [goal], "transfer": [src, goal], "mixed": [src, goal]}[spec["mechanism"]]
        nb = sorted({c for a_ in anchors for c in range(64) if cheb(c, a_) == 1 and c not in (src, goal)})
        off = [c for c in nb if c not in allsw] or nb
        fixed = [int(x) for x in rng.choice(off, min(int(rng.integers(1, 4)), len(off)), replace=False)]
        sf = place_scene(cfg, root_id, "fixed", [src], fixed)
        sf = replace(sf, robot=replace(sf.robot, q_rad=q0))
        stf, af, _ = t_goal({"scene": scene_to_dict(sf), "goal_cell": goal, "fixed_ids": [], "target": "t0", "key": root_id + ":fixed"}, {},
                            {o.id: o.cell for o in sf.objects}, np.asarray(q0, float))
        if stf != "VERIFIED":
            raise Reject(f"FIXED_ALONE_BLOCKS:{stf}")
        sw = sx.swept_cells(robot, cfg, af)
        allsw = set().union(*map(set, sw.values()))
    pools = mech_pools(sw)
    # witness chasing: blocker j goes on the requested-mechanism sweep of the T witness that still exists with blockers 1..j-1 placed;
    # once T is no longer verified, the remaining intended blockers come from the last witness's pools (verified K decides later)
    chased, blocked = 0, None

    def t_with(cells_list, tag):
        sj = place_scene(cfg, root_id, tag, [src], cells_list + fixed)
        sj = replace(sj, robot=replace(sj.robot, q_rad=q0))
        tj = {"scene": scene_to_dict(sj), "goal_cell": goal, "fixed_ids": [], "target": "t0", "key": f"{root_id}:{tag}"}
        return t_goal(tj, {}, {o.id: o.cell for o in sj.objects}, np.asarray(q0, float))
    for attempt in range(3 if K else 1):                      # <= 3 chases; then keep the last (verified K decides)
        pick, cur = [], sw
        for j in range(K):
            m = spec["mechanism"] if spec["mechanism"] != "mixed" else ("grasp", "place", "transfer")[j % 3]
            P = mech_pools(cur)
            opts = sorted(P[m] - set(pick) - set(fixed)) or sorted(set().union(*P.values()) - set(pick) - set(fixed))
            if not opts:
                raise Reject("MECHANISM_POOL_TOO_SMALL")
            pick.append(int(rng.choice(opts)))
            stj, aj, _ = t_with(pick, f"chase{attempt}.{j}")
            chased += 1
            if stj != "VERIFIED":
                break                                         # T already blocked by fewer than K: fill the rest below
            cur = sx.swept_cells(robot, cfg, aj)
        blocked = K == 0 or stj != "VERIFIED"
        if blocked:
            break
    while len(pick) < K:                                      # requested count kept; these extra blockers may be unnecessary
        P = mech_pools(cur)
        opts = sorted(set().union(*P.values()) - set(pick) - set(fixed)) or sorted(set(G["valid"]) - set(pick) - set(fixed) - {src, goal})
        pick.append(int(rng.choice(opts)))
    lo_d, hi_d = bc.N_DISTRACT[spec["distractors"]]
    nd = min(int(rng.integers(lo_d, hi_d + 1)), bc.MAX_MOVABLE - K)
    near_sw = {c for c in G["valid"] for s in allsw if cheb(c, s) == 1}
    role_cells = {"near_T": [c for c in G["valid"] if 1 <= cheb(c, src) <= 2 and c not in allsw],
                  "near_goal": [c for c in G["valid"] if 1 <= cheb(c, goal) <= 2 and c not in allsw],
                  "near_path": [c for c in near_sw if c not in allsw],
                  "far": [c for c in G["valid"] if cheb(c, src) >= 3 and cheb(c, goal) >= 3 and c not in allsw]}
    dpool = sorted({c for c in role_cells[spec["distractor_role"]] if ok(c) and c not in pick and c not in fixed})
    if len(dpool) < lo_d:
        raise Reject("DISTRACTOR_ROLE_POOL_TOO_SMALL")
    nd = min(nd, len(dpool))                                  # stays inside the requested level (>= its minimum)
    dis = [int(x) for x in rng.choice(dpool, nd, replace=False)] if nd else []
    s = place_scene(cfg, root_id, "orig", [src], pick + dis + fixed)
    s = replace(s, robot=replace(s.robot, q_rad=q0))
    allids = [o.id for o in s.objects if o.id != "t0"]
    ids = allids[:K + nd]                                     # movable objects only
    fixed_ids = allids[K + nd:]
    roles = {i: ("intended_blocker" if j < K else f"distractor:{spec['distractor_role']}") for j, i in enumerate(ids)}
    occ = set(pick + dis + fixed + [src])
    npark = bc.PARKING_N[spec["parking"]]
    park, used = {}, set()
    for o in ids:
        c0 = s.get(o).cell
        cands = sorted((c for c in G["valid"] if c not in occ and c not in used and cheb(c, src) >= 2 and cheb(c, goal) >= 2),
                       key=lambda c: (math.dist(s.lattice.center(c), s.lattice.center(c0)), c))
        park[o] = [{"cell": int(c), "in_target_sweep": c in allsw} for c in cands[:npark]]
        if park[o]:
            used.add(park[o][0]["cell"])
    wrong = sorted((c for c in allsw if c in G["valid"] and c not in occ and c != goal), key=lambda c: (cheb(c, src), c))
    return {"scene": s, "goal": goal, "q0": q0, "stratum": stratum, "roles": roles, "fixed_ids": fixed_ids, "fixed_cells": fixed, "parking": park, "wrong_dest": wrong[:2],
            "empty_witness": a, "swept": {k: list(v) for k, v in sw.items()}, "intended": ids[:K], "pools": {k: sorted(v) for k, v in pools.items()}, "chase_queries": chased, "blockers_only_scene_blocks_T": blocked}


def teach(c, store, root_id) -> dict:
    """Binary labels for the original task, D edits (target-only), B sequences; verified K; blocker mask scope and validity."""
    from scene import scene_to_dict
    import p16_common as pc
    s, goal, q0 = c["scene"], c["goal"], np.asarray(c["q0"], float)
    task = {"scene": scene_to_dict(s), "goal_cell": goal, "fixed_ids": c["fixed_ids"], "target": "t0", "key": root_id}
    cache, t0, calls = {}, time.process_time(), {"goal": 0, "sequence": 0}
    cells0 = {o.id: o.cell for o in s.objects}

    def cap():
        if time.process_time() - t0 > ADMIN_CPU_S:
            raise AdminCap(f"root CPU > {ADMIN_CPU_S}s")

    def goal_q(edit):
        cap()
        calls["goal"] += 1
        return t_goal(task, cache, {**cells0, **edit}, q0)

    park = {o: v[0]["cell"] for o, v in c["parking"].items() if v}
    st0, a0, _ = goal_q({})
    original = label(st0)
    D, edits = [], {}

    def edit(sub):
        key = tuple(sorted(sub))
        if key not in edits:
            st, _, _ = goal_q({o: park[o] for o in sub})
            edits[key] = label(st, edit={o: park[o] for o in sub}, query_kind="SCENE_TARGET_ONLY")
        return edits[key]
    movable = [o for o in c["roles"] if o in park]
    for o in movable:
        D.append(edit([o]))
    if len(c["intended"]) > 1 and all(o in park for o in c["intended"]):
        D.append(edit(c["intended"]))
    pool = ([o for o in c["intended"] if o in park] + [o for o in movable if c["roles"][o] == "distractor:near_path"])[:POOL_CAP]
    audit = int(bc.sha(["audit", root_id])[:8], 16) % 4 == 0

    def seq(steps):
        cap()
        bc.check_sequence(steps)
        calls["sequence"] += 1
        r = pc.verify_sequence(_phys(task, cache), [(o, d) for o, d in steps])
        wids = [store.witness({k: np.asarray(v) for k, v in rec.items()}) for rec in r["plan"]["records"]] if r["plan"] else []
        return label(r["status"], failed_step=r["failed_step"], sequence=[[o, int(d)] for o, d in steps], n_actions=len(steps) + 1,
                     travel=r["plan"]["travel"] if r["plan"] else None, witnesses=wids)
    K, minimal, tries = (0 if original["y"] else None), None, 0
    clearing = {}
    if not original["y"]:
        for k in range(1, len(pool) + 1):
            subs = [list(x) for x in itertools.combinations(pool, k)]
            clr = [u for u in subs if edit(u)["y"]]
            clearing[k] = clr
            for u in clr:
                T = s.get("t0").xy_m
                base = sorted(u, key=lambda o: (math.dist(s.get(o).xy_m, T), o))
                for perm in list(dict.fromkeys(itertools.permutations(base)))[:MAX_ORDERS]:
                    if tries >= MAX_SEQ_TRIES:
                        break
                    tries += 1
                    r = seq([(o, park[o]) for o in perm])
                    if r["y"]:
                        K, minimal = k, r
                        break
                if minimal:
                    break
            if minimal:
                break
    if audit:
        for k in range(1, len(pool) + 1):
            for u in itertools.combinations(pool, k):
                edit(list(u))
    # blocker supervision over the pool scope only
    full = all(tuple(sorted(u)) in edits for k in range(1, len(pool) + 1) for u in itertools.combinations(pool, k))
    pos = [set(u) for u, v in edits.items() if v["y"] and set(u) <= set(pool)]
    minimal_sets = [u for u in pos if not any(v < u for v in pos)]
    monotone = all(edits[tuple(sorted(w))]["y"] for u in minimal_sets for k in range(len(u), len(pool) + 1)
                   for w in itertools.combinations(pool, k) if set(w) >= u and tuple(sorted(w)) in edits)
    valid = bool(full and not original["y"] and minimal_sets and monotone)
    union = set().union(*minimal_sets) if minimal_sets else set()
    blocker = {"scope": pool, "full_lattice": full, "audit_subset": audit, "minimal_sets": [sorted(u) for u in minimal_sets], "monotone": monotone,
               "blocker_supervision_valid": valid, "mask": {o: int(o in union) for o in pool} if valid else None,
               "why_invalid": None if valid else ("FEASIBLE_ORIGINAL" if original["y"] else "LATTICE_INCOMPLETE" if not full else
                                                  "NO_CLEARING_SET_IN_POOL" if not minimal_sets else "NON_MONOTONE")}
    # B: 8-12 strategic sequences, each <= 7 actions incl. T
    B = []
    if minimal:
        B.append({**minimal, "role": "minimal"})
    ms = [o for o, _ in (minimal["sequence"] if minimal else [])] or ([o for o in c["intended"] if o in park])
    dist = [o for o in movable if o not in ms]
    plans = []
    for k2, clr in sorted(clearing.items()):
        for u in clr:
            if not minimal or set(u) != set(ms):
                plans.append(("alternative", [(o, park[o]) for o in u]))
                break
    if len(ms) >= 2:
        plans.append(("wrong_order", [(o, park[o]) for o in reversed(ms)]))
        plans.append(("partial", [(o, park[o]) for o in ms[:-1]]))
    if dist:
        plans.append(("redundant+1", [(dist[0], park[dist[0]])] + [(o, park[o]) for o in ms]))
        plans.append(("noop_irrelevant", [(dist[0], park[dist[0]])]))
    if len(dist) >= 2:
        plans.append(("redundant+2", [(dist[0], park[dist[0]]), (dist[1], park[dist[1]])] + [(o, park[o]) for o in ms]))
    if ms and c["wrong_dest"]:
        plans.append(("wrong_destination", [(ms[0], c["wrong_dest"][0])] + [(o, park[o]) for o in ms[1:]]))
    if not minimal and ms:
        plans.append(("intended_union", [(o, park[o]) for o in ms]))
    for o in ms[:1]:
        if len(c["parking"][o]) > 1:
            plans.append(("alternative_destination", [(o, c["parking"][o][1]["cell"])] + [(x, park[x]) for x in ms[1:]]))
    seen = set()
    for role, steps in plans[:11]:
        key = tuple(steps)
        if key in seen or not steps:
            continue
        seen.add(key)
        r = seq(steps)
        B.append({**r, "role": role, "excess_actions": (len(steps) - K) if (K is not None and r["y"]) else None})
    return {"labels": {"original": original, "D": D, "B": B, "blocker": blocker, "edits_evaluated": len(edits)}, "verified_K": K,
            "teacher_calls": calls, "teacher_cpu_s": time.process_time() - t0,
            "original_witness": store.witness({k: np.asarray(v) for k, v in a0.items()}) if original["y"] else None}


def record(spec, ids, c, t, store) -> dict:
    s = c["scene"]
    from scene import scene_to_dict
    return {"ids": ids, "domain": DOMAIN, "spec": spec, "scene": scene_to_dict(s), "goal_cell": c["goal"], "start_stratum": c["stratum"], "q0": list(c["q0"]),
            "roles": c["roles"], "fixed_ids": c["fixed_ids"], "parking": c["parking"], "intended_blockers": c["intended"], "mechanism_pools": c["pools"], "construction": {"chase_queries": c["chase_queries"], "blockers_only_scene_blocks_T": c["blockers_only_scene_blocks_T"]},
            "empty_scene_T_witness": store.witness({k: np.asarray(v) for k, v in c["empty_witness"].items()}), "empty_scene_T_swept": c["swept"],
            "n_movable": len(c["roles"]), "verified_K": t["verified_K"], "observed": "NONE" if t["verified_K"] is None else t["verified_K"],
            "labels": t["labels"], "teacher_calls": t["teacher_calls"], "teacher_cpu_s": t["teacher_cpu_s"], "budget": {"per_query": QB, "admin_cpu_s": ADMIN_CPU_S}}
