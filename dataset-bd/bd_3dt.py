"""3D OPEN-TABLETOP domain adapter for the B/D dataset (bd_core): xArm7 (MuJoCo Menagerie) + the open table of the `yadu/dataset-3d`
branch (gh-yadu-iiith/table_scene (1).py, XArm7TableEnv: 8x8 grid, 7 cm pitch, cells (row, col) with row 0 nearest the robot; every
object is a free body; side grasps only = the scene's own N_AZIMUTH x TILTS_DEG(90) poses). The scene file is imported UNMODIFIED (its
name has a space, so via importlib) next to the 3D repository's random/ helpers it borrows from.

One env per root holds T and every obstacle. A planning query = the scene's own plan_pick_place over its grasp poses in its preference
order (first feasible wins; nothing moves). A hypothetical D edit teleports the edited obstacles to their parking cells and re-queries
ONLY the original target T from the ready pose. A B sequence is EXECUTED in MuJoCo (run_plan), step by step from the initial scene; a
step counts only if the mover ends in its destination cell upright and no other object changed cell. The scene's RRT-Connect wall-clock
limits are replaced by its own iteration ceiling (max_iters) so labels do not depend on server load; a per-root administrative wall
cap lives in bd_run."""
from __future__ import annotations

import importlib.util
import itertools
import os
import sys
import time

import numpy as np

import bd_core as bc

DOMAIN = "3dt"
RSS3D = os.environ.get("BD_RSS3D_ROOT", "/data/user1/rss_ebm_dataset/src/rss-3d/gh-yadu-iiith")   # yadu/dataset-3d checkout (b3b5b34+)
SCENE_FILE = "table_scene (1).py"
RRT_ITERS = 5000
TEACHER_VERSION = f"BD3DT_TEACHER_V1:table_scene.plan_pick_place(first feasible side grasp, RRT max_iters={RRT_ITERS}, no wall-clock)+MuJoCo run_plan"
POOL_CAP, MAX_SEQ_TRIES = 5, 8
ROWS = {"near": [1, 2], "mid": [3, 4], "far": [5, 6]}
DISP = {"short": (1, 2), "medium": (3, 4), "long": (5, 7)}
COLORS = ["red", "blue", "orange", "purple", "yellow", "#30b0c0", "#c06090"]   # the scene's colour names; green = T
GREEDY_CANDS = 10
_G = {}


class Reject(Exception):
    pass


class AdminCap(Exception):
    pass


def env():
    if not _G:
        sys.path.insert(0, os.path.join(RSS3D, "random"))
        spec = importlib.util.spec_from_file_location("table_scene", os.path.join(RSS3D, SCENE_FILE))
        TS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(TS)
        base = TS.RRTConnect

        class RRT(base):                       # the scene's planner with its iteration ceiling only (no wall-clock cut-off)
            def __init__(self, *a, **kw):
                kw["max_time"], kw["max_iters"] = 1e9, RRT_ITERS
                super().__init__(*a, **kw)
        TS.RRTConnect = RRT
        src = {f: bc.sha(open(os.path.join(RSS3D, f), "rb").read().hex())[:12] for f in (SCENE_FILE, "random/rrt_connect.py", "random/rrt_star.py")}
        geo = {"pitch": TS.PITCH, "x0": TS.GRID_X0, "table_z": TS.TABLE_Z, "azimuths": TS.N_AZIMUTH, "tilts": TS.TILTS_DEG, "lift": TS.CARRY_LIFT}
        _G.update(TS=TS, protocol="XARM7_TABLE_SIDEGRASP_V1:" + bc.sha(geo)[:12], geometry=geo, source_files=src,
                  config_hash=bc.sha({"v": TEACHER_VERSION, "geo": geo, "rrt": RRT_ITERS, "src": src})[:16])
    return _G


def supports(spec) -> bool:
    return spec["start_pose"] == "home"           # one validated start configuration (the scene's ready pose)


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def build(objs, cells):
    TS = env()["TS"]
    return TS.XArm7TableEnv([{**{k: o[k] for k in ("shape", "size", "height", "color")}, "name": o["id"], "cell": tuple(cells[o["id"]])}
                             for o in objs], verbose=False)


def query(e, name, dst, seed=0):
    """Plan name -> dst from the arm's current pose: the scene's grasp poses in its order, first feasible wins. Nothing moves."""
    TS = env()["TS"]
    i = e.obj_id(name)
    e.objects[i]["pos"] = e.obj_pos(i)
    q0, why = e.q_now(), {}
    for g in TS.grasp_poses(e.objects[i], e.table_z):
        p = e.plan_pick_place(name, tuple(dst), g, q_start=q0, seed=seed)
        if p["feasible"]:
            return p, {"grasp_id": g["id"], "azimuth": g["azimuth_deg"], "tried": len(why) + 1}
        why[g["id"]] = p["reason"][:90]
    return None, {"grasp_reasons": why, "tried": len(why)}


def feasible_grasps(objs, cells, name, dst):
    """Azimuths of every grasp pose that plans name -> dst (all poses tried; nothing moves)."""
    TS = env()["TS"]
    e = build(objs, cells)
    i = e.obj_id(name)
    e.objects[i]["pos"] = e.obj_pos(i)
    q0 = e.q_now()
    return [g["azimuth_deg"] for g in TS.grasp_poses(e.objects[i], e.table_z) if e.plan_pick_place(name, tuple(dst), g, q_start=q0)["feasible"]]


def kind(why):
    r = list(why.values())
    if r and all("overlap a neighbour" in x or "would hit" in x or "occupied" in x for x in r):
        return "CERTIFIED"                    # every grasp fails a deterministic geometric check of the protocol (no search involved)
    return "NO_PLAN_FOUND"


def label(plan, info, **meta):
    if plan is not None:
        return bc.binary(True, f"grasp {info['grasp_id']}", **info, **meta)
    hits = sorted({x.split("would hit ")[1].split(" near")[0] for x in info["grasp_reasons"].values() if "would hit " in x})
    return bc.binary(False, "no feasible side grasp", kind(info["grasp_reasons"]), **info, hit_objects=hits, **meta)


def witness(store, plan):
    q = [np.asarray(s["q"], float) for s in plan["segments"] if s["kind"] != "grip"]
    ph = [k for k, s in enumerate(plan["segments"]) if s["kind"] != "grip" for _ in s["q"]]
    return store.witness({"q": np.vstack(q), "segment": np.asarray(ph), "carry_xy": np.asarray(plan["carry_xy"], float),
                          "phases": np.asarray([s["phase"] for s in plan["segments"]])})


def settle(e, cells):
    """Scene at `cells`: reset (home cells, ready pose), then teleport whoever is elsewhere (destinations are free cells)."""
    e.reset()
    for o, c in cells.items():
        if e.cell_of(o) != tuple(c):
            e.teleport(o, tuple(c))


def construct(spec, seed, root_id) -> dict:
    if spec["start_pose"] != "home":
        raise Reject("NONHOME_START_NOT_VALIDATED_IN_3DT")
    rng = np.random.default_rng(seed)
    lo, hi = DISP[spec["displacement"]]
    for _ in range(20):
        src = (int(rng.choice(ROWS[spec["source"]])), int(rng.integers(1, 7)))
        goals = [(r, c) for r in range(1, 8) for c in range(8) if lo <= cheb((r, c), src) <= hi]
        if goals:
            break
    else:
        raise Reject("NO_GOAL_IN_DISPLACEMENT_BAND")
    goal = goals[int(rng.integers(len(goals)))]
    T = {"id": "t0", "shape": "cylinder", "size": float(rng.uniform(0.016, 0.020)), "height": float(rng.uniform(0.12, 0.16)), "color": "green"}
    F0 = feasible_grasps([T], {"t0": src}, "t0", goal)
    if not F0:
        raise Reject("T_NOT_FEASIBLE_ON_EMPTY_TABLE")
    inside = lambda x: 0 <= x[0] < 8 and 0 <= x[1] < 8  # noqa: E731
    off = lambda a, n=1: (int(round(n * np.cos(np.radians(a)))), int(round(n * np.sin(np.radians(a)))))  # noqa: E731  hand side of azimuth a
    side = lambda rc: [(rc[0] + off(a, n)[0], rc[1] + off(a, n)[1]) for n in (1, 2) for a in F0]  # noqa: E731  feasible grasps' hand cells first
    ring = lambda rc: [(rc[0] + a, rc[1] + b) for a in (-1, 0, 1) for b in (-1, 0, 1) if a or b]  # noqa: E731
    line = []
    for t in np.linspace(0, 1, 29)[1:-1]:
        x = tuple(int(round(v)) for v in (1 - t) * np.array(src) + t * np.array(goal))
        if x not in line and x not in (src, goal):
            line.append(x)
    bad = {src, goal}
    pools = {"grasp": side(src) + ring(src), "place": side(goal) + ring(goal),
             "transfer": line + [(r + a, c + b) for r, c in line for a, b in ((0, 1), (0, -1), (1, 0), (-1, 0))]}
    pools = {k: list(dict.fromkeys(x for x in v if inside(x) and x not in bad)) for k, v in pools.items()}
    K = spec["K"]
    if spec["mechanism"] == "mixed":
        pool = list(dict.fromkeys(x for trio in itertools.zip_longest(pools["grasp"], pools["place"], pools["transfer"]) for x in trio if x is not None))
    else:
        pool = pools[spec["mechanism"]]
    if len(pool) < K:
        raise Reject("MECHANISM_POOL_TOO_SMALL")
    dims = []
    for j in range(bc.MAX_MOVABLE):
        cyl = bool(rng.integers(2))
        dims.append({"id": f"o{j:02d}", "shape": "cylinder" if cyl else "box", "size": float(rng.uniform(0.013, 0.018) if cyl else rng.uniform(0.012, 0.016)),
                     "height": float(rng.uniform(0.12, 0.18) if j < K else rng.uniform(0.06, 0.18)), "color": COLORS[j % len(COLORS)]})
    pick, left, greedy = [], len(F0), []
    for j in range(K):        # constrained greedy: blocker j removes >= 1 of T's remaining feasible grasps and leaves >= K-1-j for the later ones
        cands = [x for x in pool if x not in pick][:GREEDY_CANDS]
        need, scored = K - 1 - j, []
        for x in cands:
            cl = {"t0": src, **{dims[i]["id"]: p for i, p in enumerate(pick)}, dims[j]["id"]: x}
            n = len(feasible_grasps([T] + dims[:j + 1], cl, "t0", goal)) if left else 0
            key = (0, n) if need <= n < left else (1, -n) if n < need else (2, 0)    # valid (most removed) < removes too many < removes none
            scored.append((key, cands.index(x), x, n))
        key, _, x, left = min(scored)
        pick.append(x)
        greedy.append({"cell": list(x), "feasible_grasps_left": left, "necessary_by_construction": key[0] == 0})
    used = set(pick) | bad
    path = set().union(*map(set, pools.values()))
    near_path = {(r + a, c + b) for r, c in line for a in (-1, 0, 1) for b in (-1, 0, 1)}
    allc = [(r, c) for r in range(8) for c in range(8)]
    role_cells = {"near_T": [x for x in allc if 1 <= cheb(x, src) <= 2], "near_goal": [x for x in allc if 1 <= cheb(x, goal) <= 2],
                  "near_path": [x for x in allc if x in near_path], "far": [x for x in allc if cheb(x, src) >= 3 and cheb(x, goal) >= 3 and x not in near_path]}
    lo_d, hi_d = bc.N_DISTRACT[spec["distractors"]]
    nd = min(int(rng.integers(lo_d, hi_d + 1)), bc.MAX_MOVABLE - K)
    dpool = sorted(x for x in set(role_cells[spec["distractor_role"]]) if x not in used and x not in set(pools["grasp"][:5] + pools["place"][:5]))
    if len(dpool) < lo_d:
        raise Reject("DISTRACTOR_ROLE_POOL_TOO_SMALL")
    nd = min(nd, len(dpool))
    dis = [dpool[i] for i in rng.choice(len(dpool), nd, replace=False)] if nd else []
    objs, cells = [T] + dims[:len(pick + dis)], {"t0": list(src)}
    for o, rc in zip(objs[1:], pick + dis):
        cells[o["id"]] = [int(rc[0]), int(rc[1])]
    roles = {o["id"]: ("intended_blocker" if j < K else f"distractor:{spec['distractor_role']}") for j, o in enumerate(objs[1:])}
    occ = {tuple(v) for v in cells.values()}
    npark = bc.PARKING_N[spec["parking"]]
    park, taken = {}, set()
    for o in objs[1:]:
        c0 = tuple(cells[o["id"]])
        cands = sorted((x for x in allc if x not in occ and x not in taken and cheb(x, src) >= 2 and cheb(x, goal) >= 2 and x not in path
                        and x not in near_path and x[0] >= 1), key=lambda x: (cheb(x, c0), abs(x[0] - c0[0]), x))
        park[o["id"]] = [list(x) for x in cands[:npark]]
        if cands:
            taken.add(cands[0])
    wrong = [list(x) for x in line if x not in occ][:2]
    return {"objs": objs, "cells": cells, "src": list(src), "goal": list(goal), "roles": roles, "parking": park, "intended": [o["id"] for o in objs[1:K + 1]],
            "wrong_dest": wrong, "pools": {k: [list(x) for x in v] for k, v in pools.items()}, "empty_solve": {"feasible_azimuths": F0}, "greedy": greedy}


def teach(c, store, root_id) -> dict:
    t0, calls = time.time(), {"plan": 0, "execute": 0}
    objs, cells0, goal = c["objs"], c["cells"], c["goal"]
    park = {o: v[0] for o, v in c["parking"].items() if v}
    e = build(objs, cells0)

    def q_t(cells):
        settle(e, cells)
        calls["plan"] += 1
        p, info = query(e, "t0", goal)
        return label(p, info, witness=witness(store, p) if p else None)
    original = q_t(cells0)
    edits = {}

    def edit(sub):
        key = tuple(sorted(sub))
        if key not in edits:
            edits[key] = {**q_t({**cells0, **{o: park[o] for o in sub}}), "edit": {o: park[o] for o in sub}, "query_kind": "SCENE_TARGET_ONLY"}
        return edits[key]
    movable = [o for o in c["roles"] if o in park]
    D = [edit([o]) for o in movable]
    if len(c["intended"]) > 1 and all(o in park for o in c["intended"]):
        D.append(edit(c["intended"]))
    rank = {"distractor:near_path": 0, "distractor:near_T": 1, "distractor:near_goal": 2}
    pool = ([o for o in c["intended"] if o in park] + sorted((o for o in movable if c["roles"][o] in rank), key=lambda o: rank[c["roles"][o]]))[:POOL_CAP]

    def seq(steps):
        bc.check_sequence(steps)
        settle(e, cells0)
        info, wids = [], []
        for j, (o, d) in enumerate(list(steps) + [("t0", goal)]):
            calls["execute"] += 1
            before = {n: e.cell_of(n) for n in e.obj_index if n != o}
            p, qi = query(e, o, d)
            step = {"object": o, "dest": list(d), **qi}
            fail = None
            if p is None:
                fail = label(None, qi)
            else:
                wids.append(witness(store, p))
                try:
                    e.run_plan(p)
                except env()["TS"].GraspFailed as err:
                    fail = bc.binary(False, f"GRASP_FAILED:{str(err)[:80]}", "OTHER_FAILURE")
                if fail is None:
                    h = next(x["height"] for x in objs if x["id"] == o)
                    z = e.obj_pos(o)[2] - e.table_z - h / 2
                    moved = sorted(n for n in before if e.cell_of(n) != before[n])
                    step.update(pushed=p.get("pushed", []), end_cell=e.cell_of(o), z_err=float(z))
                    if e.cell_of(o) != tuple(d) or abs(z) > 0.01 or moved:
                        fail = bc.binary(False, f"EXECUTION_NOT_VERIFIED:end={e.cell_of(o)} z_err={z:.3f} displaced={moved}", "OTHER_FAILURE")
            info.append({**step, "ok": fail is None})
            if fail is not None:
                return {**fail, "failed_step": j if o != "t0" else "target", "sequence": [[x, list(y)] for x, y in steps], "n_actions": len(steps) + 1,
                        "steps": info, "witnesses": wids}
        final = {n: list(e.cell_of(n)) for n in e.obj_index}
        return bc.binary(True, "executed", failed_step=None, sequence=[[x, list(y)] for x, y in steps], n_actions=len(steps) + 1, steps=info,
                         witnesses=wids, final_cells=final)
    K, minimal, tries, clearing, reloc = None, None, 0, {}, {"queries": 0, "feasible": 0}
    if original["y"]:
        r = seq([])
        K, minimal = (0, r) if r["y"] else (None, None)
    else:                    # BFS over sets of relocated scope objects, reached only through relocations that plan in the scene they face
        level = {(): ()}
        for k in range(1, min(len(pool), bc.MAX_NECESSARY) + 1):
            nxt = {}
            for S, order in level.items():
                for o in pool:
                    S2 = tuple(sorted(set(S) | {o}))
                    if o in S or S2 in nxt:
                        continue
                    settle(e, {**cells0, **{x: park[x] for x in S}})
                    reloc["queries"] += 1
                    calls["plan"] += 1
                    if query(e, o, park[o])[0] is not None:
                        reloc["feasible"] += 1
                        nxt[S2] = order + (o,)
            clearing[k] = [list(S) for S in nxt if edit(list(S))["y"]]
            for S in clearing[k]:
                if tries >= MAX_SEQ_TRIES:
                    break
                tries += 1
                r = seq([(o, park[o]) for o in nxt[tuple(S)]])
                if r["y"]:
                    K, minimal = k, r
                    break
            if minimal or not nxt:
                break
            level = nxt
    if not original["y"]:
        for k in range(1, len(pool) + 1):              # full T-only lattice over the scope (cheap here): blocker masks for every root
            for u in itertools.combinations(pool, k):
                edit(list(u))
    full = all(tuple(sorted(u)) in edits for k in range(1, len(pool) + 1) for u in itertools.combinations(pool, k))
    pos = [set(u) for u, v in edits.items() if v["y"] and set(u) <= set(pool)]
    mins = [u for u in pos if not any(v < u for v in pos)]
    valid = bool(full and not original["y"] and mins)
    union = set().union(*mins) if mins else set()
    blocker = {"scope": pool, "full_lattice": full, "minimal_sets": [sorted(u) for u in mins], "blocker_supervision_valid": valid,
               "mask": {o: int(o in union) for o in pool} if valid else None, "clearing_sets_reachable": clearing, "sequence_tries": tries, "relocation_queries": reloc,
               "why_invalid": None if valid else ("FEASIBLE_ORIGINAL" if original["y"] else "LATTICE_INCOMPLETE" if not full else "NO_CLEARING_SET_IN_POOL")}
    B = [{**minimal, "role": "minimal"}] if minimal else []
    ms = [o for o, _ in (minimal["sequence"] if minimal else [])] or [o for o in c["intended"] if o in park]
    dist = [o for o in movable if o not in ms]
    plans = []
    if dist and ms:
        plans.append(("redundant+1", [(dist[0], park[dist[0]])] + [(o, park[o]) for o in ms]))
    if dist:
        plans.append(("noop_irrelevant", [(dist[0], park[dist[0]])]))
    if ms and c["wrong_dest"]:
        plans.append(("wrong_destination", [(ms[0], c["wrong_dest"][0])] + [(o, park[o]) for o in ms[1:]]))
    if len(ms) >= 2:
        plans.append(("wrong_order", [(o, park[o]) for o in reversed(ms)]))
    if not minimal and ms:
        plans.append(("intended_union", [(o, park[o]) for o in ms]))
    for role, steps in plans[:4]:
        r = seq(steps)
        B.append({**r, "role": role, "excess_actions": (len(steps) - K) if (K is not None and r["y"]) else None})
    return {"labels": {"original": original, "D": D, "B": B, "blocker": blocker, "edits_evaluated": len(edits)}, "verified_K": K, "teacher_calls": calls,
            "teacher_wall_s": time.time() - t0}


def record(spec, ids, c, t, store) -> dict:
    return {"ids": ids, "domain": DOMAIN, "spec": spec, "objects": c["objs"], "initial_cells": c["cells"], "target_src": c["src"], "goal": c["goal"],
            "cell_convention": "table_scene (row, col), 0-based, row 0 nearest the robot; centre = (x0 + (r-3.5)*pitch, (c-3.5)*pitch)",
            "geometry": env()["geometry"], "source_files": env()["source_files"], "roles": c["roles"], "parking": c["parking"], "intended_blockers": c["intended"], "mechanism_pools": c["pools"],
            "empty_table_T_solve": c["empty_solve"], "greedy_blockers": c["greedy"], "n_movable": len(c["roles"]), "verified_K": t["verified_K"],
            "observed": "NONE" if t["verified_K"] is None else t["verified_K"], "labels": t["labels"], "teacher_calls": t["teacher_calls"],
            "teacher_wall_s": t["teacher_wall_s"], "budget": {"rrt_max_iters": RRT_ITERS, "seq_tries": MAX_SEQ_TRIES, "pool_cap": POOL_CAP}}
