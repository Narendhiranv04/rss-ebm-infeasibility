"""3D domain adapter for the B/D dataset (bd_core): xArm7 (MuJoCo Menagerie) + the existing cabinet shelf 8x8 grid of the
`yadu/dataset-3d` branch (grid_dataset.py GRID8: cells 3 cm deep x 5 cm wide; internal cells are 0-based (row, col), row 0 = front,
nearest the robot; the repository's USER ids are 1-based: id = row*8 + col + 1). The 3D repository is imported UNMODIFIED from
BD_RSS3D (default: the server checkout).

grid_dataset.GridPlanner moves only objects[0] and treats every obstacle as static geometry. A physical intervention therefore rebuilds
the scene with the object to move in the movable slot and every other object (including T) static at its CURRENT cell, plans it
(IK + RRT-Connect, side grasps only: the planner's own pitch x offset x height candidates), EXECUTES it in MuJoCo, and carries the
executed final pose (cell + in-cell offset) forward. A hypothetical D edit rebuilds the scene with the obstacle at its parking cell
and re-queries ONLY the original target T. Wall-clock limits of the planner are disabled (rrt_time, time_limit = 1e9): iteration and
attempt ceilings decide; a per-root administrative wall cap lives in bd_run."""
from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

import bd_core as bc

DOMAIN = "3d"
RSS3D = os.environ.get("BD_RSS3D", "/data/user1/rss_ebm_dataset/src/rss-3d/gh-yadu-iiith/random")
PLANNER = {"attempts": 3, "ik_steps": 10000, "rrt_iters": 4000, "rrt_time": 1e9, "time_limit": 1e9, "seed": 0}
TEACHER_VERSION = "BD3D_TEACHER_V1:GridPlanner(attempts=3,ik_steps=10000,rrt_iters=4000,no wall-clock)+MuJoCo execute"
POOL_CAP, MAX_SEQ_TRIES = 3, 6
ROWS = {"near": [1, 2], "mid": [3, 4], "far": [5, 6]}
DISP = {"short": (1, 2), "medium": (3, 4), "long": (5, 7)}
COLORS = ["red", "blue", "orange", "purple", "cyan", "pink", "yellow", "brown"]
_G = {}


class Reject(Exception):
    pass


class AdminCap(Exception):
    pass


def env():
    if not _G:
        sys.path.insert(0, RSS3D)
        import grid_dataset as GD
        import xarm7_cabinet_env as E
        _G.update(GD=GD, E=E, protocol="XARM7_SIDEGRASP_V1:" + bc.sha({"pitches": E.PITCHES_DEG, "offsets": E.GRASP_OFFSETS, "grid": GD.GRID8,
                                                                         "cab_w": GD.CABINET_W})[:12],
                  config_hash=bc.sha({"planner": PLANNER, "v": TEACHER_VERSION, "grid": GD.GRID8})[:16])
    return _G


def supports(spec) -> bool:
    """Only the validated 3D start configuration; at most 5 movable objects (cost and cabinet capacity)."""
    return spec["start_pose"] == "home" and spec["K"] + bc.N_DISTRACT[spec["distractors"]][0] <= 5


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def uid(rc):
    return rc[0] * 8 + rc[1] + 1                     # the 3D repository's 1-based user id


def _obstacle(GD, k, o, rc, off=(0.0, 0.0)):
    size = o["size"] if o["shape"] == "cylinder" else [o["size"], o["size"], o["size"]]
    return GD.make_obstacle(k, o["shape"], rc, size, o["color"], dx=off[0], dy=off[1], height=o["height"], name=o["id"],
                            allow_overflow=bool(off[0] or off[1]))      # an executed (physical) offset may exceed the cell-fit rule


def build(objs, state, mover):
    """Environment with `mover` movable at its current cell and every other object static at its current cell (+ executed offset)."""
    G = env()
    GD = G["GD"]
    obst = [_obstacle(GD, k, o, tuple(state[o["id"]]["rc"]), state[o["id"]].get("off", (0.0, 0.0)))
            for k, o in enumerate(x for x in objs if x["id"] != mover)]
    m = next(o for o in objs if o["id"] == mover)
    e = GD.build_env(tuple(state[mover]["rc"]), obst, dict(shape=m["shape"], size=m["size"], height=m["height"], color=m["color"]), seed=0)
    return e


def solve(objs, state, mover, dst, execute=False):
    """Plan (and optionally execute) mover -> dst. -> (ok, result dict, new state)."""
    G = env()
    GD = G["GD"]
    e = build(objs, state, mover)
    pl = GD.GridPlanner(e, attempts=PLANNER["attempts"], ik_steps=PLANNER["ik_steps"], rrt_iters=PLANNER["rrt_iters"], rrt_time=PLANNER["rrt_time"],
                        seed=PLANNER["seed"], time_limit=PLANNER["time_limit"], verbose=False)
    pl.place_target(tuple(state[mover]["rc"]))
    res = pl.solve(tuple(state[mover]["rc"]), tuple(dst))
    out = {"feasible": bool(res["feasible"]), "reason": res["reason"], "attempts": res["attempts"], "ik_runs": res["ik_runs"],
           "candidates": res["candidates"], "rrt_runs": res["rrt_runs"], "blocked_by": res["blocked_by"], "obstacle_caused": res.get("obstacle_caused")}
    new = state
    if res["feasible"] and execute:
        pl.execute(res)
        out.update(executed=True, verified=bool(res["verified"]), exec=res["exec"])
        p = e.obj_pos(0)
        rc = e.cell_of(0)
        if rc is not None:
            cx, cy = GD.cell_xy(rc)
            new = {**state, mover: {"rc": [int(rc[0]), int(rc[1])], "off": (float(p[0] - cx), float(p[1] - cy))}}
        out["traj"] = {f"traj_{s['name']}": s["traj"] for s in res["plan"]["order"]}
        out["record"] = res.get("record")
    return out["feasible"] and (out.get("verified", True)), out, new, e


def construct(spec, seed, root_id) -> dict:
    if spec["start_pose"] != "home":
        raise Reject("NONHOME_START_NOT_VALIDATED_IN_3D")       # the 3D protocol has one validated start configuration
    rng = np.random.default_rng(seed)
    src = (int(rng.choice(ROWS[spec["source"]])), int(rng.integers(1, 7)))
    lo, hi = DISP[spec["displacement"]]
    goals = [(r, c) for r in range(1, 8) for c in range(8) if lo <= cheb((r, c), src) <= hi]
    if not goals:
        raise Reject("NO_GOAL_IN_DISPLACEMENT_BAND")
    goal = goals[int(rng.integers(len(goals)))]
    T = {"id": "t0", "shape": "cylinder", "size": 0.012, "height": 0.08, "color": "green"}
    ok, r0, _, _ = solve([T], {"t0": {"rc": list(src)}}, "t0", goal)
    if not ok:
        raise Reject(f"T_NOT_FEASIBLE_IN_EMPTY_CABINET:{r0['reason'][:60]}")
    fr = lambda rc, dc: [(rc[0] - 1, rc[1] + d) for d in dc if 0 <= rc[0] - 1 and 0 <= rc[1] + d < 8]  # noqa: E731   (in front: robot side)
    lo_c, hi_c = sorted((src[1], goal[1]))
    pools = {"grasp": set(fr(src, (0, -1, 1))), "place": set(fr(goal, (0, -1, 1))),
             "transfer": {(r, c) for r in range(0, min(src[0], goal[0])) for c in range(lo_c, hi_c + 1)}}
    bad = {src, goal}
    pools = {k: sorted(v - bad) for k, v in pools.items()}
    K = spec["K"]
    if spec["mechanism"] == "mixed":
        names = [m for m in pools if pools[m]]
        pick = []
        for i in range(K):
            opts = [x for x in pools[names[i % len(names)]] if x not in pick] if names else []
            opts = opts or [x for m in pools for x in pools[m] if x not in pick]
            if not opts:
                raise Reject("MECHANISM_POOL_TOO_SMALL")
            pick.append(opts[int(rng.integers(len(opts)))])
    else:
        pool = pools[spec["mechanism"]]
        if len(pool) < K:
            raise Reject("MECHANISM_POOL_TOO_SMALL")
        pick = [pool[i] for i in rng.choice(len(pool), K, replace=False)] if K else []
    used = set(pick) | bad
    path = set().union(*map(set, pools.values()))
    near_path = {(r + a, c + b) for r, c in path for a in (-1, 0, 1) for b in (-1, 0, 1)}
    allc = [(r, c) for r in range(8) for c in range(8)]
    role_cells = {"near_T": [x for x in allc if 1 <= cheb(x, src) <= 2], "near_goal": [x for x in allc if 1 <= cheb(x, goal) <= 2],
                  "near_path": [x for x in near_path if x in set(allc)], "far": [x for x in allc if cheb(x, src) >= 3 and cheb(x, goal) >= 3 and x not in near_path]}
    lo_d, hi_d = bc.N_DISTRACT[spec["distractors"]]
    nd = min(int(rng.integers(lo_d, hi_d + 1)), 5 - K)
    dpool = sorted(x for x in set(role_cells[spec["distractor_role"]]) if x not in used and x not in path)
    if len(dpool) < lo_d:
        raise Reject("DISTRACTOR_ROLE_POOL_TOO_SMALL")
    nd = min(nd, len(dpool))
    dis = [dpool[i] for i in rng.choice(len(dpool), nd, replace=False)] if nd else []
    objs, state = [T], {"t0": {"rc": list(src)}}
    for j, rc in enumerate(pick + dis):
        cyl = bool(rng.integers(2))
        o = {"id": f"o{j:02d}", "shape": "cylinder" if cyl else "box", "size": float(rng.uniform(0.010, 0.012) if cyl else rng.uniform(0.008, 0.0095)),
             "height": float(rng.uniform(0.08, 0.12) if j < K else rng.uniform(0.05, 0.12)), "color": COLORS[j % len(COLORS)]}
        objs.append(o)
        state[o["id"]] = {"rc": list(rc)}
    roles = {o["id"]: ("intended_blocker" if j < K else f"distractor:{spec['distractor_role']}") for j, o in enumerate(objs[1:])}
    occ = {tuple(v["rc"]) for v in state.values()}
    npark = bc.PARKING_N[spec["parking"]]
    park, taken = {}, set()
    for o in objs[1:]:
        c0 = tuple(state[o["id"]]["rc"])
        cands = sorted((x for x in allc if x not in occ and x not in taken and cheb(x, src) >= 2 and cheb(x, goal) >= 2 and x not in path
                        and x[0] >= 1), key=lambda x: (cheb(x, c0), abs(x[0] - c0[0]), x))
        park[o["id"]] = [{"rc": list(x), "user_id": uid(x)} for x in cands[:npark]]
        if cands:
            taken.add(cands[0])
    wrong = [list(x) for x in sorted(path - occ - {goal})][:2]
    return {"objs": objs, "state": state, "src": list(src), "goal": list(goal), "roles": roles, "parking": park, "intended": [o["id"] for o in objs[1:K + 1]],
            "wrong_dest": wrong, "pools": {k: [list(x) for x in v] for k, v in pools.items()}, "empty_solve": r0}


def label(ok, res, **meta):
    if ok:
        return bc.binary(True, res["reason"][:120], **meta)
    why = res.get("reason", "")
    kind = "TIMEOUT" if "time limit" in why else ("CERTIFIED" if "occupied" in why else "NO_PLAN_FOUND")
    if res.get("executed") and not res.get("verified"):
        kind, why = "OTHER_FAILURE", f"EXECUTION_NOT_VERIFIED:{res.get('exec')}"
    return bc.binary(False, why[:200], kind, **meta)


def teach(c, store, root_id) -> dict:
    t0, calls = time.time(), {"plan": 0, "execute": 0}
    objs, s0, goal = c["objs"], c["state"], c["goal"]
    park = {o: v[0]["rc"] for o, v in c["parking"].items() if v}

    def q_t(state):
        calls["plan"] += 1
        ok, r, _, _ = solve(objs, state, "t0", goal)
        return label(ok, r, planner=dict((k, r[k]) for k in ("attempts", "ik_runs", "candidates", "rrt_runs", "blocked_by")))
    original = q_t(s0)
    edits = {}

    def edit(sub):
        key = tuple(sorted(sub))
        if key not in edits:
            st = {**s0, **{o: {"rc": park[o]} for o in sub}}
            edits[key] = {**q_t(st), "edit": {o: park[o] for o in sub}, "query_kind": "SCENE_TARGET_ONLY"}
        return edits[key]
    movable = [o for o in c["roles"] if o in park]
    D = [edit([o]) for o in movable]
    if len(c["intended"]) > 1 and all(o in park for o in c["intended"]):
        D.append(edit(c["intended"]))
    pool = ([o for o in c["intended"] if o in park] + [o for o in movable if c["roles"][o] == "distractor:near_path"])[:POOL_CAP]
    audit = int(bc.sha(["audit", root_id])[:8], 16) % 4 == 0

    def seq(steps):
        bc.check_sequence(steps)
        state, wids, info = dict(s0), [], []
        for j, (o, d) in enumerate(list(steps) + [("t0", goal)]):
            calls["execute"] += 1
            ok, r, state, _ = solve(objs, state, o, d, execute=True)
            info.append({"object": o, "dest": list(d), "ok": ok, "reason": r["reason"][:120], "exec": r.get("exec")})
            if r.get("traj"):
                wids.append(store.witness({k: np.asarray(v) for k, v in r["traj"].items()}))
            if not ok:
                return label(False, r, failed_step=j if o != "t0" else "target", sequence=[[x, list(y)] for x, y in steps], n_actions=len(steps) + 1,
                             steps=info, witnesses=wids)
        return label(True, r, failed_step=None, sequence=[[x, list(y)] for x, y in steps], n_actions=len(steps) + 1, steps=info, witnesses=wids,
                     final_state=state)
    K, minimal, tries, clearing = (None, None, 0, {})
    if original["y"]:
        r = seq([])
        K, minimal = (0, r) if r["y"] else (None, None)
    else:
        for k in range(1, len(pool) + 1):
            clr = [list(u) for u in itertools.combinations(pool, k) if edit(list(u))["y"]]
            clearing[k] = clr
            for u in clr:
                for perm in list(itertools.permutations(u))[:2]:
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
    full = all(tuple(sorted(u)) in edits for k in range(1, len(pool) + 1) for u in itertools.combinations(pool, k))
    pos = [set(u) for u, v in edits.items() if v["y"] and set(u) <= set(pool)]
    mins = [u for u in pos if not any(v < u for v in pos)]
    valid = bool(full and not original["y"] and mins)
    union = set().union(*mins) if mins else set()
    blocker = {"scope": pool, "full_lattice": full, "audit_subset": audit, "minimal_sets": [sorted(u) for u in mins], "blocker_supervision_valid": valid,
               "mask": {o: int(o in union) for o in pool} if valid else None,
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
    return {"ids": ids, "domain": DOMAIN, "spec": spec, "objects": c["objs"], "initial_state": c["state"], "target_src": c["src"], "goal": c["goal"],
            "goal_user_id": uid(c["goal"]), "cell_convention": "internal 0-based [row, col], row 0 = front; repository user id = row*8+col+1",
            "roles": c["roles"], "parking": c["parking"], "intended_blockers": c["intended"], "mechanism_pools": c["pools"],
            "empty_cabinet_T_solve": {k: c["empty_solve"][k] for k in ("reason", "attempts", "candidates")}, "n_movable": len(c["roles"]),
            "verified_K": t["verified_K"], "observed": "NONE" if t["verified_K"] is None else t["verified_K"], "labels": t["labels"],
            "teacher_calls": t["teacher_calls"], "teacher_wall_s": t["teacher_wall_s"], "budget": {"planner": PLANNER}}
