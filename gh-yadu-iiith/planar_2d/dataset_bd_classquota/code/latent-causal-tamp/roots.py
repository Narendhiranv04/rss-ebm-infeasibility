"""G4A development roots and the frozen audit plan (G4A instruction §6, §8).

Roots are sampled from geometry only, in five construction strata (categories, not
feasibility labels). Every construction attempt is kept in the ledger with its seed
and outcome. The plan freezes, before any target outcome exists: roots and actions,
the automatic candidates and primary ranking per query, the seed/order panel
selections (stable-hash rules), budgets, seeds and the coverage targets. Its hash is
checked by the audit before collection.
"""
from __future__ import annotations

import math

import numpy as np

import data
import geometry as ge
import proposals as pr
from config import config_hash, stable_hash
from scene import place_scene, scene_to_dict

STRATA = ("UNOCCUPIED_GOAL", "OCCUPIED_GOAL", "ACCESS_DEPENDENCY", "COMPETING_DESTINATIONS", "NUISANCE_CONTROL")
NUISANCE_CELLS = 2.0                 # nuisance clutter keeps this many cells from the transport segment
TARGETS = {"roots_complete": 20, "zero_move_roots": 6, "nonempty_witness_roots": 8, "alternative_roots": 4,
           "goal_negative_roots": 3, "access_negative_roots": 3, "min_signatures_per_multi_alternative_query": 2,
           "max_first_pass_unknown_fraction": 0.5, "semantic_failures": 0}


def _neighbours(cell: int, cols: int, rows: int) -> list:
    r, c = divmod(cell, cols)
    return [rr * cols + cc for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if 0 <= rr < rows and 0 <= cc < cols]


def _layout(rng, stratum: str, n: int, cols: int, rows: int):
    """(target cell, goal cell, obstacle cells) drawn for one stratum, or a rejection reason."""
    cells = list(range(cols * rows))
    t, g = (int(x) for x in rng.choice(cells, 2, replace=False))
    obs = []
    if stratum in ("OCCUPIED_GOAL", "ACCESS_DEPENDENCY", "COMPETING_DESTINATIONS"):
        obs.append(g)
    if stratum == "ACCESS_DEPENDENCY":
        nb = [c for c in _neighbours(g, cols, rows) if c != t]
        obs += [int(c) for c in rng.choice(nb, min(2, len(nb)), replace=False)]
    if stratum == "COMPETING_DESTINATIONS":
        obs += [c for c in _neighbours(g, cols, rows) if c != t][:3]
    rest = [c for c in cells if c not in obs and c not in (t, g)]
    if stratum == "NUISANCE_CONTROL":                          # clutter only away from the transport segment
        xy = lambda c: ((c % cols + 0.5), (c // cols + 0.5))  # noqa: E731  (cell units)
        rest = [c for c in rest if pr._seg_dist(xy(c), xy(t), xy(g)) >= NUISANCE_CELLS]
        if len(rest) < n:
            return t, g, None
    obs += [int(c) for c in rng.choice(rest, max(0, n - len(obs)), replace=False)]
    return t, g, obs[:n]


def _predicate(stratum: str, t: int, g: int, obs: list, pitch: float, cols: int, rows: int):
    """Geometric stratum check (no planning); returns a rejection reason or None."""
    xy = lambda c: ((c % cols + 0.5) * pitch, (c // cols + 0.5) * pitch)  # noqa: E731
    if math.dist(xy(t), xy(g)) < 2 * pitch:
        return "target and goal closer than 2 cells"
    occupied = g in obs
    if stratum in ("UNOCCUPIED_GOAL", "NUISANCE_CONTROL") and occupied:
        return "goal occupied"
    if stratum != "UNOCCUPIED_GOAL" and stratum != "NUISANCE_CONTROL" and not occupied:
        return "goal not occupied"
    if stratum == "ACCESS_DEPENDENCY" and sum(c in obs for c in _neighbours(g, cols, rows)) < 2:
        return "goal occupant has fewer than two occupied neighbours"
    if stratum == "COMPETING_DESTINATIONS":
        near = [c for c in range(cols * rows) if c not in obs and c != t and 0 < math.dist(xy(c), xy(g)) <= 1.5 * pitch]
        if len(near) > 2:
            return "more than two free spots near the goal occupant"
    if stratum == "NUISANCE_CONTROL":
        seg = lambda c: pr._seg_dist(xy(c), xy(t), xy(g))  # noqa: E731
        if any(seg(c) < NUISANCE_CELLS * pitch - 1e-12 for c in obs):
            return f"an obstacle lies within {NUISANCE_CELLS} cells of the transport segment"
    return None


def build_roots(cfg: dict, robot) -> tuple:
    """(accepted roots, ledger) - strata interleaved so the first four are one each of S1-S4."""
    a, s = cfg["audit4a"], cfg["suite"]
    want = {k: a["roots"] // len(STRATA) for k in STRATA}
    roots, ledger, seen = [], [], set()
    for attempt in range(a["max_construction_attempts"]):
        if len(roots) == a["roots"]:
            break
        stratum = STRATA[len(roots) % len(STRATA)]
        if want[stratum] == 0:
            stratum = next(k for k in STRATA if want[k])
        seed = a["root_seed"] + attempt
        rng = np.random.default_rng(seed)
        n = int(rng.integers(a["min_obstacles"], a["max_obstacles"] + 1))
        t, g, obs = _layout(rng, stratum, n, s["cols"], s["rows"])
        why = "too few free spots away from the transport segment" if obs is None else \
            _predicate(stratum, t, g, obs, s["pitch_m"], s["cols"], s["rows"])
        obs = obs or []
        scene = None if why else place_scene(cfg, f"g4a_{seed}", "initial", [t], obs)
        if scene is not None:
            sig = stable_hash(sorted(o.xy_m for o in scene.objects) + [g])
            home = ge.Checker(robot, ge.make_world(scene), cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"])
            if sig in seen:
                why = "duplicate geometry"
            elif home.status_at(np.asarray(robot.q_home), robot.open_gap).status != ge.CLEAR:
                why = "home configuration not clear"
            seen.add(sig)
        ledger.append({"attempt": attempt, "seed": seed, "stratum": stratum, "obstacles": n, "target_cell": t,
                       "goal_cell": g, "obstacle_cells": obs, "accepted": why is None, "rejection": why})
        if why is None:
            roots.append({"root_id": f"g4a_r{len(roots):02d}", "stratum": stratum, "seed": seed, "scene": scene,
                          "goal_cell": g})
            want[stratum] -= 1
    return roots, ledger


def make_plan(cfg: dict, robot) -> dict:
    roots, ledger = build_roots(cfg, robot)
    queries = []
    for r in roots:
        q = data.make_query(cfg, r["root_id"], r["scene"], "t0", r["goal_cell"],
                            {"root_id": r["root_id"], "stratum": r["stratum"], "construction_seed": r["seed"]}, "SAFE_RETREAT")
        s0 = data.initial_scene(q)                               # generator sees learner fields only
        gen = pr.generate_candidates(s0, q["learner"]["action"], cfg)
        ranked = pr.rank_proposals(s0, q["learner"]["action"], gen["candidates"], cfg)
        queries.append({"query_id": r["root_id"], "root_id": r["root_id"], "stratum": r["stratum"], "seed": r["seed"],
                        "scene": scene_to_dict(r["scene"]), "goal_cell": r["goal_cell"], "hashes": q["hashes"],
                        "action": q["learner"]["action"], "generator": gen, "primary_ranking": ranked,
                        "audit_pool": [{"label": "all-stay", "cells": {}, "category": "all-stay"}] + gen["candidates"]})
    plan = {"stage": "G4A", "protocol": "INITIAL_SCENE_ONE_SHOT_V1", "config_hash": config_hash(cfg),
            "caps": cfg["audit4a"], "generator": cfg["generator"], "budgets": {
                "repair2d": cfg["repair2d"], "planner2d": cfg["planner2d"], "skill2d": cfg["skill2d"]},
            "end_policy": cfg["target_evaluation"]["end_policy"], "targets": TARGETS, "strata": STRATA,
            "ledger": ledger, "queries": queries, "pilot": [q["query_id"] for q in queries[: cfg["audit4a"]["pilot_roots"]]],
            "tie_policy": "geometry keys (distance, x, y); object IDs never order candidates",
            "seed_panel": seed_panel(cfg, queries), "order_panel": order_panel(cfg, queries)}
    plan["plan_hash"] = stable_hash(plan)
    return plan


def _h(*parts) -> str:
    return stable_hash(list(parts))


def seed_panel(cfg: dict, queries: list) -> list:
    """Stable-hash rule over plan inputs: per stratum the all-stay case of the smallest-hash
    root and its smallest-hash nonempty candidate, then nonempty fill to the cap."""
    a, cases = cfg["audit4a"], []
    for st in STRATA:
        qs = sorted((q for q in queries if q["stratum"] == st), key=lambda q: _h(q["query_id"]))
        if not qs:
            continue
        q = qs[0]
        cases.append((q["query_id"], {}))
        ne = sorted((c for c in q["audit_pool"] if c["cells"]), key=lambda c: _h(q["query_id"], pr.signature(c["cells"])))
        if ne:
            cases.append((q["query_id"], ne[0]["cells"]))
    rest = sorted(((q["query_id"], c["cells"]) for q in queries for c in q["audit_pool"] if c["cells"]),
                  key=lambda x: _h(x[0], pr.signature(x[1])))
    for x in rest:
        if len(cases) >= a["seed_panel_cases"]:
            break
        if x not in cases:
            cases.append(x)
    return [{"query_id": q, "cells": c, "keys": [f"{q}/seedpanel/{pr.signature(c)}/s{k}" for k in range(a["seed_panel_seeds"])]}
            for q, c in cases[: a["seed_panel_cases"]]]


def order_panel(cfg: dict, queries: list) -> list:
    """Multi-object candidates by smallest stable hash, at most two per stratum."""
    a = cfg["audit4a"]
    pool = sorted(((q, c) for q in queries for c in q["audit_pool"] if len(c["cells"]) >= 2),
                  key=lambda x: _h(x[0]["query_id"], pr.signature(x[1]["cells"])))
    out, per = [], {}
    for q, c in pool:
        if len(out) < a["order_panel_cases"] and per.get(q["stratum"], 0) < 2:
            per[q["stratum"]] = per.get(q["stratum"], 0) + 1
            out.append({"query_id": q["query_id"], "cells": c["cells"], "orders": a["order_panel_orders"],
                        "keys": [f"{q['query_id']}/orderpanel/{pr.signature(c['cells'])}/s{k}" for k in range(a["order_panel_seeds"])]})
    return out
