"""INITIAL_SCENE_ONE_SHOT_V1 endpoint geometry and proposals (Methodology v1.5 §2-3, §6;
Guide v1.7 §4-6). Moved from solve.py at G4A, which keeps the frozen A0 control.

Everything here reads only the initial scene, robot state and action/goal: no frozen
path, no planner outcome, no label. `generate_candidates` is the G4A automatic
geometric generator (B1-style control); it is not a learned method.
"""
from __future__ import annotations

import itertools
import math
import zlib

import numpy as np

import motion as mo
from config import stable_hash


def endpoint_xy(scene, dest):
    """A destination is a physical cell ID or, for declared off-grid controls, an [x, y] pose."""
    return scene.lattice.center(dest) if isinstance(dest, int) else (float(dest[0]), float(dest[1]))


def json_key(dest):
    return dest if isinstance(dest, int) else tuple(round(float(v), 9) for v in dest)


def signature(cells: dict) -> tuple:
    """Canonical physical assignment (stays dropped by the caller)."""
    return tuple(sorted((o, json_key(d)) for o, d in cells.items()))


def static_endpoints(scene, assignment: dict, target_id: str, cfg: dict) -> dict:
    """Y_static: support of every final pose, the target at rest, and pairwise disc
    compatibility (clearance delta, numerical gray band eps). No robot and no sweep:
    a VALID result says nothing about whether any motion exists."""
    delta, eps = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"]
    lat = scene.lattice
    size = (lat.cols * lat.pitch_m, lat.rows * lat.pitch_m)
    ids = {o.id: o for o in scene.objects}
    errors = [f"{o}: not a movable obstacle" for o in assignment if o not in ids or ids[o].kind != "obstacle"]
    errors += [f"{o}: cell {d} not a legal spot" for o, d in assignment.items()
               if isinstance(d, int) and d not in scene.legal_cells]
    if errors:
        return {"status": mo.INVALID, "errors": errors, "failed": [], "factors_checked": 0}
    pose = {o.id: (endpoint_xy(scene, assignment[o.id]) if o.id in assignment else o.xy_m, o.radius_m)
            for o in scene.objects}
    factors = [("support", (i,), min(x - r, size[0] - r - x, y - r, size[1] - r - y), 0.0)
               for i, ((x, y), r) in pose.items()]
    factors += [("target_at_rest" if target_id in (a, b) else "pair", (a, b),
                 math.dist(pose[a][0], pose[b][0]) - pose[a][1] - pose[b][1], delta)
                for a, b in itertools.combinations(sorted(pose), 2)]
    graded = [(kind, ids_, mo._band(v, need, eps), v) for kind, ids_, v, need in factors]
    return {"status": mo._conjoin(s for _, _, s, _ in graded), "errors": [], "factors_checked": len(graded),
            "failed": [{"factor": k, "objects": list(i), "status": s, "value_m": round(v, 6)}
                       for k, i, s, v in graded if s != mo.VALID]}


def goal_poses(action: dict) -> list:
    """Every allowed final target centre: the declared goal plus any declared alternatives."""
    return [tuple(action["goal_xy_m"])] + [tuple(p) for p in action.get("goal_alternatives_xy", [])]


def goal_obstruction(scene, assignment: dict, action: dict, cfg: dict) -> list:
    """Per allowed goal pose, the obstacles whose final pose provably overlaps it: even at
    the most favourable target placement (goal tolerance) and obstacle pose (release
    tolerance if moved, declared state uncertainty if it stays) clearance is < delta."""
    delta, eps = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"]
    rel, u = cfg["gripper2d"]["release_xy_tolerance_m"], cfg["evidence"]["state_pose_uncertainty_m"]
    t = scene.get(action["target_id"])
    out = []
    for g in goal_poses(action):
        hit = []
        for o in (o for o in scene.objects if o.kind == "obstacle"):
            xy = endpoint_xy(scene, assignment[o.id]) if o.id in assignment else o.xy_m
            slack = action["goal_tolerance_m"] + (rel if o.id in assignment else u)
            if math.dist(g, xy) - t.radius_m - o.radius_m + slack < delta - eps:
                hit.append(o.id)
        out.append(hit)
    return out


def goal_blockers(scene, assignment: dict, action: dict, cfg: dict) -> list:
    """Obstacles that make every allowed goal pose unusable; [] if any pose may be free."""
    per = goal_obstruction(scene, assignment, action, cfg)
    return sorted(set().union(*per)) if per and all(per) else []


def action_lower_bound(scene, action: dict, cfg: dict, unpickable: set) -> dict:
    """Sound scoped lower bound on moved objects for the new objective; 0 by default.
    Unique goal only: each obstacle provably on it must move (+1 each). If every one of
    them has a certified pick failure in S0 (pose-only screen, declared grasp family),
    none can be moved first, so some other object must move before them (+1)."""
    if action.get("goal_alternatives_xy"):
        return {"lower_bound": 0, "premises": ["goal region with alternatives: no occupancy bound"]}
    occ = goal_blockers(scene, {}, action, cfg)
    lb, premises = len(occ), [f"goal occupied by {o} (goal-overlap certificate)" for o in occ]
    if occ and all(o in unpickable for o in occ):
        lb += 1
        premises.append(f"{', '.join(occ)}: pick certified infeasible in S0, so another object must move first")
    return {"lower_bound": lb, "premises": premises or ["no sound bound beyond 0"]}


def rank_proposals(scene, action: dict, authored: list, cfg: dict) -> list:
    """Frozen primary ranking over learner fields only: all-stay plus the candidates, a
    fixed geometric task score (0 if the endpoints leave every goal pose provably
    occupied, else 1) filtered by the threshold, ranked by moved count then given
    order, deduplicated and capped. It never sees an outcome."""
    q = cfg["proposal"]
    pool = ([{"label": "all-stay", "cells": {}}] if q["include_all_stay"] else []) + list(authored)
    seen, ranked = set(), []
    for k, c in enumerate(pool):
        moves = {o: d for o, d in c["cells"].items() if not (isinstance(d, int) and d == scene.get(o).cell)}
        key = signature(moves)
        if key in seen or len(moves) > q["budget_cap"]:
            continue
        seen.add(key)
        score = 0.0 if goal_blockers(scene, moves, action, cfg) else 1.0
        ranked.append({"label": c["label"], "cells": moves, "moved": len(moves), "score": score, "declared_rank": k})
    passed = [c for c in ranked if c["score"] >= q["task_score_threshold"]]
    passed.sort(key=lambda c: (c["moved"], c["declared_rank"]))
    return passed[: q["max_candidates_to_verify"]]


# ------------------------------------------------------------------ G4A automatic generator
def geometry_seed(scene, action: dict, seed: int) -> int:
    """Diversity seed from geometry only (sorted poses, goal, robot), never from object IDs."""
    g = sorted((round(o.xy_m[0], 9), round(o.xy_m[1], 9), o.radius_m, o.kind) for o in scene.objects)
    return zlib.crc32(stable_hash([g, action["goal_xy_m"], list(scene.robot.q_rad), seed]).encode()) & 0x7FFFFFFF


def _seg_dist(p, a, b) -> float:
    a, b, p = np.asarray(a), np.asarray(b), np.asarray(p)
    t = 0.0 if np.allclose(a, b) else float(np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * (b - a))))


def generate_candidates(scene, action: dict, cfg: dict) -> dict:
    """B1_GEOMETRIC_V1. Rule (stated in report 0008):
    1. Obstacles ordered by (distance to the target-goal segment or goal, x, y); those within
       `corridor_m` are task-proximal. Free spots = legal cells not occupied initially, not
       the target's or the goal's cell; each obstacle prefers spots outside the corridor,
       then near itself, ties by (x, y).
    2. Raw pool: singles (every obstacle x first `destinations_per_object` spots), pairs and
       one occupied-destination chain among the first three proximal obstacles, a proximal
       triple, and seeded broad single/triple moves of any obstacle (geometry seed).
    3. Whole-assignment static screen, physical dedup, then a fixed round-robin over the
       categories [single, pair/chain, triple, broad] up to `max_static_valid` - 1 (+ all-stay).
    It reads no planner output, label, pick certificate, ID meaning or construction history."""
    g = cfg["generator"]
    t, goal = scene.get(action["target_id"]), tuple(action["goal_xy_m"])
    obs = [o for o in scene.objects if o.kind == "obstacle"]
    near = lambda o: (min(_seg_dist(o.xy_m, t.xy_m, goal), math.dist(o.xy_m, goal)), o.xy_m[0], o.xy_m[1])  # noqa: E731
    obs.sort(key=near)
    prox = [o for o in obs if near(o)[0] < g["corridor_m"]] or obs[:1]
    occupied = {o.cell for o in scene.objects}
    goal_cell = action["goal_cell"]
    free = [c for c in scene.legal_cells if c not in occupied and c != goal_cell]

    def prefer(o):
        def key(c):
            xy = scene.lattice.center(c)
            return (_seg_dist(xy, t.xy_m, goal) < g["corridor_m"], math.dist(xy, o.xy_m), xy[0], xy[1])
        return sorted(free, key=key)
    rng = np.random.default_rng(geometry_seed(scene, action, g["seed"]))
    cats = {"single": [], "pair": [], "triple": [], "broad": []}
    for o in obs:
        cats["single"] += [{o.id: c} for c in prefer(o)[: g["destinations_per_object"]]]
    top = prox[:3]
    for a, b in itertools.combinations(top, 2):
        pa, pb = prefer(a), [c for c in prefer(b) if c != prefer(a)[0]]
        cats["pair"].append({a.id: pa[0], b.id: pb[0]})
        cats["pair"].append({a.id: b.cell, b.id: pb[0]})             # chain: a takes b's spot, b moves on
    if len(top) == 3:
        used, tri = set(), {}
        for o in top:
            c = next(c for c in prefer(o) if c not in used)
            used.add(c)
            tri[o.id] = c
        cats["triple"].append(tri)
    for _ in range(g["broad_draws"]):
        o = obs[int(rng.integers(len(obs)))]
        cats["broad"].append({o.id: int(free[int(rng.integers(len(free)))])})
    if len(obs) >= 3:
        pick = sorted(rng.choice(len(obs), 3, replace=False).tolist())
        dests = rng.choice(len(free), 3, replace=False).tolist()
        cats["triple"].append({obs[i].id: int(free[d]) for i, d in zip(pick, dests)})
    raw = [("all-stay", {})] + [(k, c) for k in cats for c in cats[k]]
    raw = raw[: g["max_raw"]]
    seen, valid, rejected = set(), {k: [] for k in cats}, {"static_invalid": 0, "static_unresolved": 0,
                                                            "duplicate": 0, "too_many_moves": 0}
    for kind, cells in raw[1:]:
        cells = {o: d for o, d in cells.items() if d != scene.get(o).cell}
        sig = signature(cells)
        if not cells or sig in seen:
            rejected["duplicate"] += 1
            continue
        seen.add(sig)
        if len(cells) > g["max_moved"]:
            rejected["too_many_moves"] += 1
            continue
        st = static_endpoints(scene, cells, t.id, cfg)["status"]
        if st != mo.VALID:
            rejected["static_invalid" if st == mo.INVALID else "static_unresolved"] += 1
            continue
        valid[kind].append(cells)
    chosen, order = [], ["single", "pair", "triple", "broad"]
    while len(chosen) < g["max_static_valid"] - 1 and any(valid.values()):
        for k in order:
            if valid[k] and len(chosen) < g["max_static_valid"] - 1:
                chosen.append({"label": f"{k}:{' '.join(f'{o}>{d}' for o, d in sorted(valid[k][0].items()))}",
                               "cells": valid[k].pop(0), "category": k})
    return {"generator": g["name"], "candidates": chosen, "raw": len(raw), "rejected": rejected,
            "proximal": [o.id for o in prox], "geometry_seed": geometry_seed(scene, action, g["seed"])}
