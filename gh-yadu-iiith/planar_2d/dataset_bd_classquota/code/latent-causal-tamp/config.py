"""Configuration registry (Guide v1.6 §2): TOML schema, physical consistency checks,
canonical hashes and the fixture reference-construction policy.

Split from scene.py (size, announced in reports 0003/0004): scene.py keeps the
scene records; this module owns everything about validated settings. Unknown,
missing or mistyped keys fail immediately.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import tomllib
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

VEC2, VEC3, LIMITS3, PAIRS = "vec2", "vec3", "limits3", "pairs"
SCHEMA = {
    "suite": {"name": str, "rows": int, "cols": int, "pitch_m": float,
              "obstacle_radius_m": float, "margin_m": float, "num_obstacles": int,
              "num_targets": int, "coordinate_frame": str, "domain_policy": str,
              "margin_policy": str},
    "arm2d": {"base_xy_m": VEC2, "link_lengths_m": VEC3, "link_full_widths_m": VEC3,
              "joint_limits_rad": LIMITS3, "q_home_rad": VEC3, "base_radius_m": float,
              "joint_housing_radius_m": float, "self_collision_exclusions": PAIRS},
    "gripper2d": {"model": str, "palm_length_m": float, "palm_full_width_m": float,
                  "finger_length_m": float, "finger_thickness_m": float,
                  "open_gap_m": float, "closed_gap_m": float, "tcp_at_palm_tip": bool,
                  "attachment_pose_policy": str, "contact_pos_tolerance_m": float,
                  "release_xy_tolerance_m": float, "rail_depth_m": float,
                  "grasp_center_x_m": float},
    "geometry": {"numeric_margin_band_m": float, "interval_max_step_rad": float,
                 "interval_certifier": str, "certify_max_evals": int},
    "planner2d": {"algorithm": str, "seed": int, "max_nodes": int, "max_wall_seconds": float,
                  "extend_step_rad": float, "shortcut_rounds": int, "pose_step_m": float,
                  "pose_phi_weight_m": float, "edge_max_evals": int},
    "skill2d": {"approach_dirs": int, "approach_dist_m": float, "retreat_dist_m": float,
                "line_step_m": float, "max_ik_jump_rad": float, "max_plan_failures": int,
                "max_direct_pairs": int},
    "repair2d": {"k_best": int, "max_orders": int, "max_relocations": int, "skill_plan_failures": int},
    "protocol": {"name": str, "proposal_observation": str, "target_feedback_to_selection": bool,
                 "target_calls_between_relocations": int, "target_reference_required": bool,
                 "execution_observation": str, "move_once": bool},
    "proposal": {"source": str, "budget_cap": int, "max_candidates_to_verify": int, "include_all_stay": bool,
                 "task_score_threshold": float},
    "target_evaluation": {"when": str, "start_state": str, "scope": str, "final_connector": str,
                          "max_queries_per_primary_episode": int, "end_policy": str, "home_return_diagnostic": bool},
    "evidence": {"state_pose_uncertainty_m": float, "reobserve_pose_tolerance_m": float},
    "generator": {"name": str, "corridor_m": float, "destinations_per_object": int, "seed": int, "broad_draws": int,
                  "max_raw": int, "max_static_valid": int, "max_moved": int},
    "audit4a": {"roots": int, "pilot_roots": int, "min_obstacles": int, "max_obstacles": int, "queries_per_root": int,
                "max_construction_attempts": int, "max_diagnostic_target_calls": int, "main_cpu_hours": float,
                "diagnostic_cpu_hours": float, "workers": int, "root_seed": int, "seed_panel_cases": int,
                "seed_panel_seeds": int, "order_panel_cases": int, "order_panel_orders": int, "order_panel_seeds": int},
    "render": {"image_px": int, "overview_xlim_m": VEC2, "overview_ylim_m": VEC2},
}


class ConfigError(ValueError):
    """Raised for unknown, missing, mistyped or physically inconsistent settings."""


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _vec(v, n):
    return isinstance(v, list) and len(v) == n and all(_num(x) for x in v)


def _normalize(value, spec):
    """Return the value with floats normalised, or None if it has the wrong type."""
    if spec is float:
        return float(value) if _num(value) else None
    if spec is int:
        return value if isinstance(value, int) and not isinstance(value, bool) else None
    if spec in (str, bool):
        return value if isinstance(value, spec) else None
    if spec in (VEC2, VEC3):
        return [float(x) for x in value] if _vec(value, 2 if spec == VEC2 else 3) else None
    if spec == LIMITS3:
        ok = isinstance(value, list) and len(value) == 3 and all(_vec(p, 2) for p in value)
        return [[float(x) for x in p] for p in value] if ok else None
    ok = isinstance(value, list) and all(
        isinstance(p, list) and len(p) == 2 and all(type(i) is int for i in p) for p in value)
    return value if ok else None


def validate_config(raw: dict) -> dict:
    """Check keys and types against SCHEMA, then physical consistency."""
    if set(raw) != set(SCHEMA):
        raise ConfigError(f"unknown sections {sorted(set(raw) - set(SCHEMA))}, "
                          f"missing sections {sorted(set(SCHEMA) - set(raw))}")
    cfg = {}
    for sec, spec in SCHEMA.items():
        got = raw[sec]
        if set(got) != set(spec):
            raise ConfigError(f"[{sec}] unknown keys {sorted(set(got) - set(spec))}, "
                              f"missing keys {sorted(set(spec) - set(got))}")
        cfg[sec] = {}
        for key, kind in spec.items():
            value = _normalize(got[key], kind)
            if value is None:
                raise ConfigError(f"[{sec}] {key} has invalid type/shape: {got[key]!r}")
            cfg[sec][key] = value
    _check_physics(cfg)
    return cfg


def _check_physics(cfg: dict) -> None:
    s, a, g, r = cfg["suite"], cfg["arm2d"], cfg["gripper2d"], cfg["render"]
    rad, delta, n_cells = s["obstacle_radius_m"], s["margin_m"], s["rows"] * s["cols"]
    rules = [
        (s["coordinate_frame"] == "world_xy_y_back", "coordinate_frame must be world_xy_y_back"),
        (s["margin_policy"] == "UNINFLATED_ORACLE_DELTA_ONCE", "margin_policy must apply delta once"),
        (s["domain_policy"] == "ALL_SUPPORTED",
         "RESTRICTED needs an explicit published spot list (introduced in G3)"),
        (s["rows"] > 0 and s["cols"] > 0 and s["pitch_m"] > 0, "lattice must be non-empty"),
        (rad > 0 and delta >= 0, "radius must be > 0 and margin >= 0"),
        (s["pitch_m"] > 2 * rad + delta, "distinct cells must be pair-compatible: pitch > 2r + delta"),
        (1 <= s["num_targets"] <= 3, "num_targets must be 1..3"),
        (0 <= s["num_obstacles"] and s["num_obstacles"] + s["num_targets"] <= n_cells,
         "objects must fit in distinct cells"),
        (all(x > 0 for x in a["link_lengths_m"] + a["link_full_widths_m"]), "links must be > 0"),
        (a["base_radius_m"] > 0 and a["joint_housing_radius_m"] > 0, "base/housing radii must be > 0"),
        (all(lo < hi for lo, hi in a["joint_limits_rad"]), "joint limits need lo < hi"),
        (all(lo <= q <= hi for q, (lo, hi) in zip(a["q_home_rad"], a["joint_limits_rad"])),
         "q_home outside joint limits"),
        (all(0 <= i <= 2 for p in a["self_collision_exclusions"] for i in p), "link index not in 0..2"),
        (g["open_gap_m"] > 2 * rad, "open_gap must exceed the object diameter 2r"),
        (0 < g["closed_gap_m"] < g["open_gap_m"], "need 0 < closed_gap < open_gap"),
        (min(g["palm_length_m"], g["palm_full_width_m"], g["finger_length_m"],
             g["finger_thickness_m"]) > 0, "gripper dimensions must be > 0"),
        (g["rail_depth_m"] > 0, "rail_depth must be > 0"),
        (g["grasp_center_x_m"] - rad >= delta, "held object must clear the rail/palm by delta"),
        (0 <= g["grasp_center_x_m"] <= g["finger_length_m"], "grasp contact must lie on the fingers"),
        (cfg["skill2d"]["approach_dist_m"] >= g["finger_length_m"] - g["grasp_center_x_m"] + rad + delta,
         "pregrasp finger tips must clear the object by delta"),
        (cfg["geometry"]["interval_certifier"] == "analytic_or_conservative_subdivision",
         "unknown interval_certifier"),
        (min(cfg["geometry"]["certify_max_evals"], cfg["planner2d"]["max_nodes"],
             cfg["skill2d"]["approach_dirs"], cfg["skill2d"]["max_plan_failures"]) > 0
         and cfg["skill2d"]["line_step_m"] > 0, "budgets and steps must be > 0"),
        (min(cfg["repair2d"].values()) > 0, "repair budgets must be > 0"),
        *_protocol_rules(cfg),
        (r["image_px"] > 0, "image_px must be > 0"),
        (r["overview_xlim_m"][0] < r["overview_xlim_m"][1]
         and r["overview_ylim_m"][0] < r["overview_ylim_m"][1], "overview limits need lo < hi"),
    ]
    errors = [msg for ok, msg in rules if not ok]
    if errors:
        raise ConfigError("; ".join(errors))


PROTOCOLS = ("FROZEN_CONTROL", "INITIAL_SCENE_ONE_SHOT_V1")


def _protocol_rules(cfg: dict) -> list:
    """Methodology v1.5 §9 / Guide v1.7 §2: contradictory protocol settings are rejected.
    [protocol] describes the primary case-analysis protocol; FROZEN_CONTROL runs (solve,
    replay) keep their archived semantics and never read these sections."""
    p, q, t = cfg["protocol"], cfg["proposal"], cfg["target_evaluation"]
    new = p["name"] == "INITIAL_SCENE_ONE_SHOT_V1"
    return [
        (p["name"] in PROTOCOLS, f"protocol name must be one of {PROTOCOLS}"),
        (not new or p["proposal_observation"] == "INITIAL_ONLY", "one-shot proposals use the initial observation only"),
        (not new or not p["target_feedback_to_selection"], "one-shot selection cannot receive target feedback"),
        (not new or p["target_calls_between_relocations"] == 0, "one-shot allows no target calls between relocations"),
        (not new or not p["target_reference_required"], "one-shot queries cannot require a frozen target path"),
        (not new or p["execution_observation"] == "VALIDATE_OR_ABORT", "reobservation may only validate or abort"),
        (not new or p["move_once"], "one-shot episodes are move-once"),
        (new or p["target_reference_required"], "FROZEN_CONTROL requires its frozen target path"),
        (q["source"] in ("MANUAL_CASE_CANDIDATES", "DETERMINISTIC_CASE_CANDIDATES"),
         "G3.5 proposals are manual or deterministic (no learned source before G5)"),
        (q["budget_cap"] > 0 and q["max_candidates_to_verify"] > 0, "proposal budgets must be > 0"),
        (0.0 <= q["task_score_threshold"] <= 1.0, "task_score_threshold must lie in [0, 1]"),
        (t["when"] == "AFTER_REPAIR_COMMIT_AND_COMPLETION" and t["start_state"] == "ACHIEVED",
         "target evaluation runs once, after the committed repair, from the achieved state"),
        (t["scope"] == "COMPLETE_TARGET_SKILL" and t["final_connector"] in ("HOME", "NONE"),
         "target evaluation plans the complete skill; final connector HOME or NONE"),
        (t["max_queries_per_primary_episode"] == 1, "exactly one target query per primary episode"),
        (t["end_policy"] in ("SAFE_RETREAT", "HOME_REQUIRED"), "end_policy must be SAFE_RETREAT or HOME_REQUIRED"),
        (cfg["evidence"]["state_pose_uncertainty_m"] >= 0 and cfg["evidence"]["reobserve_pose_tolerance_m"] >= 0,
         "evidence tolerances must be >= 0"),
        (cfg["evidence"]["reobserve_pose_tolerance_m"] <= cfg["gripper2d"]["release_xy_tolerance_m"],
         "reobservation tolerance cannot exceed the declared placement (release) tolerance"),
        (cfg["generator"]["name"] == "B1_GEOMETRIC_V1" and 1 <= cfg["generator"]["max_moved"] <= 3
         and cfg["generator"]["max_static_valid"] >= 1 and cfg["generator"]["max_raw"] <= 120,
         "G4A generator: B1_GEOMETRIC_V1, at most 3 moved, at most 120 raw proposals"),
        (cfg["audit4a"]["workers"] <= 8 and cfg["audit4a"]["max_diagnostic_target_calls"] <= 120
         and cfg["audit4a"]["main_cpu_hours"] <= 4.0 and cfg["audit4a"]["diagnostic_cpu_hours"] <= 2.0
         and 4 <= cfg["audit4a"]["min_obstacles"] <= cfg["audit4a"]["max_obstacles"] <= 8
         and cfg["audit4a"]["queries_per_root"] <= 2 and cfg["audit4a"]["max_construction_attempts"] <= 100,
         "G4A audit caps exceed the instruction's limits"),
    ]


def load_config(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return validate_config(tomllib.load(f))


# ------------------------------------------------------------------ V2 overlay (G4D; Guide v1.8 §4)
STR_LIST = "str_list"
SCHEMA_V2 = {
    "protocol": {"name": str, "schema_version": int, "proposal_observation": str, "order_source": str,
                 "executor_may_reorder": bool, "target_feedback_to_selection": bool,
                 "max_primary_target_evaluations": int, "target_end_policy": str, "home_return": str, "move_once": bool},
    "proposal": {"source": str, "move_cap": int, "max_candidates_to_verify": int, "task_score_threshold": float,
                 "rank": STR_LIST},
    "cost": {"policy": str, "relocation_backend": str, "waypoints_within_joint_limits": bool},
    "explanation": {"counterfactual_appearance": str, "max_omission_subset": int, "max_tested_subsets": int},
}
V2_RANK = ["move_count", "predicted_joint_travel", "endpoint_distance", "feasibility_energy", "signature"]


def validate_v2(raw: dict) -> dict:
    """Strict keys/types for the V2 overlay, then the protocol contract (spec v1.6 §12)."""
    if set(raw) != set(SCHEMA_V2):
        raise ConfigError(f"V2 sections must be exactly {sorted(SCHEMA_V2)}, got {sorted(raw)}")
    out = {}
    for sec, spec in SCHEMA_V2.items():
        if set(raw[sec]) != set(spec):
            raise ConfigError(f"[{sec}] V2 keys differ: {sorted(set(raw[sec]) ^ set(spec))}")
        out[sec] = {}
        for k, kind in spec.items():
            v = raw[sec][k]
            if kind == STR_LIST:
                norm = list(v) if isinstance(v, list) and all(isinstance(x, str) for x in v) else None
            else:
                norm = _normalize(v, kind)
            if norm is None:
                raise ConfigError(f"[{sec}] {k} has invalid type: {v!r}")
            out[sec][k] = norm
    p, q = out["protocol"], out["proposal"]
    rules = [(p["name"] == "INITIAL_SCENE_ORDERED_ONE_SHOT_V2", "V2 overlay must name INITIAL_SCENE_ORDERED_ONE_SHOT_V2"),
             (p["proposal_observation"] == "INITIAL_ONLY", "proposals use the initial observation only"),
             (p["order_source"] == "PROPOSED_SEQUENCE" and not p["executor_may_reorder"], "the executor may not reorder"),
             (not p["target_feedback_to_selection"], "no target feedback to selection"),
             (p["max_primary_target_evaluations"] == 1, "at most one primary target evaluation"),
             (p["target_end_policy"] == "GOAL_AND_SAFE_RETREAT" and p["home_return"] == "SEPARATE_DIAGNOSTIC",
              "target success = goal + safe retreat; HOME is a separate diagnostic"),
             (p["move_once"], "V2 repairs are move-once"),
             (q["source"] == "MANUAL_OR_DETERMINISTIC_SEQUENCES", "no learned proposal source before G5"),
             (1 <= q["move_cap"] <= 6 and q["max_candidates_to_verify"] > 0, "move cap in 1..6, positive verify budget"),
             (q["rank"] == V2_RANK, f"selection rank must be {V2_RANK}"),
             (out["cost"]["waypoints_within_joint_limits"], "joint travel needs waypoints inside finite limits"),
             (out["explanation"]["counterfactual_appearance"] == "INITIAL_ANCHOR", "counterfactuals keep the initial anchor"),
             (out["explanation"]["max_omission_subset"] >= 1 and out["explanation"]["max_tested_subsets"] >= 1,
              "explanation budgets must be >= 1")]
    errors = [m for ok, m in rules if not ok]
    if errors:
        raise ConfigError("; ".join(errors))
    return out


def load_config_v2(path: str | Path) -> dict:
    with open(path, "rb") as f:
        return validate_v2(tomllib.load(f))


def stable_hash(obj) -> str:
    """sha256 of canonical JSON (sorted keys, exact float repr)."""
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def config_hash(cfg: dict) -> str:
    return stable_hash(cfg)


def robot_profile_hash(cfg: dict) -> str:
    """Hash of the robot body/gripper profile (provisional until G1 freezes it)."""
    return stable_hash({"arm2d": cfg["arm2d"], "gripper2d": cfg["gripper2d"]})


# Reference-construction policy for fixtures from G2 on (reports 0004, 0005). Every
# policy plans and certifies the nominal action in the fixture's final scene except
# the sparse-seeded bootstrap. Dense primary roots hold 30-40 obstacles (Methodology
# §1); development scenes (D1, D2, ...) are explicitly sparse and never primary.
REFERENCE_POLICIES = {"FINAL_DENSE_PLANNED": "PRIMARY",
                      "FINAL_DEV_SCENE_PLANNED": "DEVELOPMENT_ONLY",
                      "SPARSE_SEEDED_REFERENCE": "BOOTSTRAP_ONLY"}
DENSE_OBSTACLES = (30, 40)


def validate_fixtures(fixtures: dict) -> dict:
    """Reject G2+ fixtures without an allowed policy/role pair, explicit dense roots
    outside 30-40 obstacles, and development scenes that are not explicitly sparse."""
    for name, fx in fixtures.items():
        if float(fx["stage"].lstrip("G").replace("_", ".")) < 2:
            continue
        if fx.get("protocol", "FROZEN_CONTROL") not in PROTOCOLS:
            raise ConfigError(f"{name}: unknown protocol {fx.get('protocol')!r}")
        if fx["stage"] == "G3_5":
            if fx.get("protocol") != "INITIAL_SCENE_ONE_SHOT_V1" or "reference_policy" in fx:
                raise ConfigError(f"{name}: G3.5 queries declare protocol INITIAL_SCENE_ONE_SHOT_V1 "
                                  "and no frozen reference policy")
            continue
        policy, role = fx.get("reference_policy"), fx.get("benchmark_role")
        if policy not in REFERENCE_POLICIES or REFERENCE_POLICIES[policy] != role:
            raise ConfigError(f"{name}: needs reference_policy in {sorted(REFERENCE_POLICIES)} "
                              f"with the matching benchmark_role, got {policy!r}/{role!r}")
        n = len(fx["obstacle_cells"]) if "obstacle_cells" in fx else None
        if policy == "FINAL_DENSE_PLANNED" and n is not None and not DENSE_OBSTACLES[0] <= n <= DENSE_OBSTACLES[1]:
            raise ConfigError(f"{name}: a dense primary root needs {DENSE_OBSTACLES} obstacles, has {n}")
        if policy == "FINAL_DEV_SCENE_PLANNED" and (n is None or n >= DENSE_OBSTACLES[0]):
            raise ConfigError(f"{name}: a development scene must list fewer than {DENSE_OBSTACLES[0]} obstacles")
    return fixtures



# ------------------------------------------------------------------ run records (moved from run.py at G4A)
def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True))


def write_manifest(out: Path, seeds: dict, chash: str, extra: dict) -> None:
    """Command, time, seeds, hashes, library versions and a sha256 per output file."""
    files = sorted(p.name for p in out.iterdir() if p.name != "manifest.json")
    write_json(out / "manifest.json", {
        "command": sys.argv, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seeds": seeds, "config_hash": chash, **extra,
        "versions": {"python": platform.python_version(), "numpy": version("numpy"),
                     "matplotlib": version("matplotlib"), "pillow": version("pillow"),
                     "platform": platform.platform()},
        "files": {n: "sha256:" + hashlib.sha256((out / n).read_bytes()).hexdigest() for n in files}})
