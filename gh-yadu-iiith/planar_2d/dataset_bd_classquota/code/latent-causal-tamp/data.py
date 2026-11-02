"""Primary query records, the learner allowlist and identity hashes (Methodology v1.5
§1, §10; Guide v1.7 §3).

A query has four namespaces. `learner` holds only allowlisted initial-scene, robot
and action/goal fields; `labels` decided evidence; `oracle_only` reference paths,
construction ancestry, root IDs and seeds; `execution` the committed repair and its
outcomes. Learner tensors and proposal inputs are built from `learner` by explicit
keys, never by flattening a record, so nothing privileged can reach them.
"""
from __future__ import annotations

import hashlib

import numpy as np

import geometry as ge
from config import robot_profile_hash, stable_hash
from scene import Scene, scene_from_dict, scene_to_dict
import viz

PROTOCOL = "INITIAL_SCENE_ONE_SHOT_V1"
PROTOCOL_V2 = "INITIAL_SCENE_ORDERED_ONE_SHOT_V2"
SCHEMA_V2 = 1
SCENE_KEYS = ("lattice", "objects", "robot", "legal_cells", "domain_policy")   # learner-visible scene fields
OBJECT_KEYS = ("id", "kind", "radius_m", "xy_m", "cell", "support")


def action_context(cfg: dict, scene: Scene, target_id: str, goal_cell: int, end_policy: str | None = None) -> dict:
    """Everything that defines the intended action, identical across candidate repairs:
    active target, unique goal pose with its tolerance, grasp family, contact mode, end
    policy and robot profile. A trajectory hash is never part of it. end_policy None is
    the G3.5 HOME-constrained scope (hash-identical to report 0007); "SAFE_RETREAT" is the
    G4A contract of addendum A1 (success through a checked retreat; HOME optional)."""
    sk = cfg["skill2d"]
    ctx = {"action": "TRANSPORT_TARGET_TO_GOAL", "target_id": target_id, "goal_cell": goal_cell,
            "goal_xy_m": list(scene.lattice.center(goal_cell)),
            "goal_tolerance_m": cfg["gripper2d"]["release_xy_tolerance_m"],
            "goal_orientation": "ANY (rotationally symmetric disc)",
            "grasp_family": {"approach_dirs": sk["approach_dirs"], "ik_branches": 2,
                             "approach_dist_m": sk["approach_dist_m"], "retreat_dist_m": sk["retreat_dist_m"]},
            "contact_mode": "PLANAR_SUPPORT_CONTACT_SIDE_GRASP", "no_lift": True, "pushing_allowed": False,
            "final_connector": cfg["target_evaluation"]["final_connector"],
            "robot_profile_hash": robot_profile_hash(cfg)}
    if end_policy is None:
        return ctx
    if end_policy != "SAFE_RETREAT":
        raise ValueError(f"unknown end policy {end_policy!r}")
    del ctx["final_connector"]
    return {**ctx, "end_policy": "SAFE_RETREAT", "home_return": "OPTIONAL_DIAGNOSTIC", "goal_alternatives_xy": [],
            "state_pose_uncertainty_m": cfg["evidence"]["state_pose_uncertainty_m"],
            "contract_addendum": "docs/addenda/A1_target_end_policy_v1.md"}


def learner_scene(scene: Scene) -> dict:
    d = scene_to_dict(scene)
    out = {k: d[k] for k in SCENE_KEYS}
    out["objects"] = [{k: o[k] for k in OBJECT_KEYS} for o in d["objects"]]
    return out


def make_query(cfg: dict, name: str, scene: Scene, target_id: str, goal_cell: int, oracle_only: dict,
               end_policy: str | None = None, protocol: str = PROTOCOL) -> dict:
    """V2 queries (protocol=PROTOCOL_V2) require the SAFE_RETREAT action context, so their
    action-context hash equals the G4A one for the same scene, target and goal."""
    if protocol == PROTOCOL_V2 and end_policy != "SAFE_RETREAT":
        raise ValueError("V2 queries use the goal + safe-retreat end policy")
    learner = {"scene": learner_scene(scene), "action": action_context(cfg, scene, target_id, goal_cell, end_policy)}
    q = {"protocol": protocol, "query_id": name, "learner": learner, "labels": {}, "oracle_only": oracle_only,
         "hashes": hashes(learner)}
    return {**q, "schema_version": SCHEMA_V2} if protocol == PROTOCOL_V2 else q


def hashes(learner: dict) -> dict:
    return {"observation": stable_hash(learner["scene"]), "action_context": stable_hash(learner["action"])}


def initial_scene(query: dict) -> Scene:
    """The immutable initial scene S0 rebuilt from learner fields only (labels neutral)."""
    d = query["learner"]["scene"]
    return scene_from_dict({**d, "root_id": "query", "variant_id": "initial", "action_hash": None,
                            "robot_profile_hash": query["learner"]["action"]["robot_profile_hash"]})


def learner_tensors(query: dict, image_px: int) -> dict:
    """float32 tensors (uint8 image) from allowlisted learner keys. Object slots follow
    storage order; IDs are not encoded. Clean image: no paths, arrows or status."""
    L = query["learner"]
    sc, act = L["scene"], L["action"]
    objs = sc["objects"]
    lat = sc["lattice"]
    centres = [((c % lat["cols"] + 0.5) * lat["pitch_m"], (c // lat["cols"] + 0.5) * lat["pitch_m"])
               for c in sc["legal_cells"]]
    target = [o["id"] == act["target_id"] for o in objs]
    rgb = viz.render_observation(initial_scene(query), image_px)[0]
    return {"object_xy": np.array([o["xy_m"] for o in objs], np.float32),
            "object_radius": np.array([o["radius_m"] for o in objs], np.float32),
            "is_active_target": np.array(target, np.float32),
            "is_obstacle": np.array([o["kind"] == "obstacle" for o in objs], np.float32),
            "robot_q": np.array(sc["robot"]["q_rad"], np.float32),
            "robot_gripper_gap": np.array([sc["robot"]["gripper_gap_m"]], np.float32),
            "robot_attached": np.array([sc["robot"]["attachment"] != "free"], np.float32),
            "goal_xy": np.array(act["goal_xy_m"], np.float32),
            "goal_tolerance": np.array([act["goal_tolerance_m"]], np.float32),
            "candidate_xy": np.array(centres, np.float32),
            "support_extent": np.array([lat["cols"] * lat["pitch_m"], lat["rows"] * lat["pitch_m"]], np.float32),
            "rgb": rgb}


# ------------------------------------------------------------------ labels (G4A; no model)
POSITIVE, NEGATIVE, UNKNOWN, CONTRADICTION = "VERIFIED_FEASIBLE", "CERTIFIED_INFEASIBLE", "UNRESOLVED", "CONTRADICTION"


def aggregate(evidence: list) -> dict:
    """Assignment-level label from all attempts: a complete witness is a positive, a
    validated assignment-wide certificate a negative, both together a contradiction that
    halts labelling; anything else (failed orders, budgets, unchecked proofs) is unknown."""
    pos = [e for e in evidence if e["kind"] == "complete_witness"]
    neg = [e for e in evidence if e["kind"] == "certificate" and e.get("validated")]
    label = CONTRADICTION if pos and neg else POSITIVE if pos else NEGATIVE if neg else UNKNOWN
    return {"label": label, "positive_sources": [e["source"] for e in pos],
            "negative_sources": sorted({e["certificate"] for e in neg}), "attempts": len(evidence)}


def label_masks(labels: list) -> dict:
    """Binary targets and masks: decided labels only; unknowns masked out of the binary
    loss and of the joint-pool denominator (which needs both a positive and a negative)."""
    if CONTRADICTION in labels:
        raise ValueError("contradictory evidence: labelling halted")
    y = np.array([1.0 if l == POSITIVE else 0.0 for l in labels], np.float32)
    mask = np.array([l in (POSITIVE, NEGATIVE) for l in labels])
    pool = mask if (y[mask] == 1).any() and (y[mask] == 0).any() else np.zeros_like(mask)
    return {"y": y, "binary_mask": mask, "joint_pool_mask": pool}


# ------------------------------------------------------------------ V2 learner inputs (G4D; Guide v1.8 §5)
def appearance_anchor(query: dict, image_px: int) -> dict:
    """The initial observation's image and instance masks: the only appearance source for
    every hypothetical (prefix, scene-intervention, omission) context of this query."""
    rgb, inst, ids = viz.render_observation(initial_scene(query), image_px)
    digest = hashlib.sha256(rgb.tobytes() + inst.tobytes() + "|".join(ids).encode()).hexdigest()
    return {"rgb": rgb, "instance": inst, "ids": ids, "hash": "sha256:" + digest}


def learner_tensors_v2(query: dict, cfg: dict, anchor: dict) -> dict:
    """V1 tensors plus the numeric robot profile (limits, link/gripper dimensions), FK joint
    positions of the initial configuration, normalized joints and the anchored instance
    masks. Built from allowlisted learner fields and the pinned config only."""
    base = learner_tensors(query, anchor["rgb"].shape[0])
    if not np.array_equal(base["rgb"], anchor["rgb"]):
        raise ValueError("appearance anchor does not belong to this query's initial observation")
    robot, q = ge.make_robot(cfg), query["learner"]["scene"]["robot"]["q_rad"]
    lim = np.array(robot.limits, np.float64)
    joints, theta = ge.fk(robot, q)
    prof = [*robot.lengths, *robot.widths, robot.base_r, robot.housing_r, robot.palm_len, robot.palm_w, robot.rail_depth,
            robot.finger_len, robot.finger_t, robot.open_gap, robot.closed_gap, robot.grasp_x]
    return {**base, "instance": anchor["instance"].astype(np.int16), "appearance_anchor": anchor["hash"],
            "robot_q_normalized": ((np.asarray(q) - lim[:, 0]) / (lim[:, 1] - lim[:, 0])).astype(np.float32),
            "robot_joint_limits": lim.astype(np.float32), "robot_profile": np.array(prof, np.float32),
            "robot_fk_joints": joints.astype(np.float32), "robot_fk_angles": theta.astype(np.float32)}


def counterfactual_tensors(tensors: dict, query: dict, poses: dict) -> dict:
    """Scene-intervention inputs: structured object poses replaced, everything else - robot
    state, task, appearance anchor (image and masks) - identical. No counterfactual image."""
    ids = [o["id"] for o in query["learner"]["scene"]["objects"]]
    xy = tensors["object_xy"].copy()
    for oid, p in poses.items():
        xy[ids.index(oid)] = p
    return {**tensors, "object_xy": xy}


def prefix_tensors(tensors: dict, query: dict, scene: Scene, seq) -> list:
    """One record per action of a canonical sequence: the ideal prefix object poses (all
    objects, stay objects included), the selected slot, its destination, step and length.
    Appearance and robot fields are the initial ones; no robot state is guessed."""
    ids, out = [o["id"] for o in query["learner"]["scene"]["objects"]], []
    xy = tensors["object_xy"].copy()
    for t, (oid, cell) in enumerate(seq):
        dest = np.array(scene.lattice.center(cell), np.float32)
        out.append({"prefix_object_xy": xy.copy(), "selected_slot": ids.index(oid), "destination_xy": dest,
                    "step": t, "length": len(seq)})
        xy[ids.index(oid)] = dest
    return out + [{"prefix_object_xy": xy.copy(), "selected_slot": -1, "destination_xy": None, "step": len(seq),
                   "length": len(seq)}]


# ------------------------------------------------------------------ V2 scoped labels and effect masks
ORDER, ASSIGNMENT, TRAJECTORY = "ORDER", "ASSIGNMENT", "TRAJECTORY"


def ordered_aggregate(evidence: list) -> dict:
    """Label of one ordered proposal: a complete repair-plus-target witness for exactly
    this order is positive; an accepted certificate is negative only if its scope covers
    the order (an order-specific proof, or an assignment-wide proof valid for every
    order). Trajectory-scope failures (one path, one connector, a timeout) never decide."""
    keep = [e for e in evidence if (e["kind"] == "complete_witness" and e.get("scope") == ORDER)
            or (e["kind"] == "certificate" and e.get("scope") in (ORDER, ASSIGNMENT))]
    return {**aggregate(keep), "ignored_out_of_scope": len(evidence) - len(keep)}


def effect_label(reference: str, alternative: str) -> dict:
    """Delta = Y(alternative) - Y(reference) in {-1, 0, 1}, definite only when both sides
    are decided; otherwise masked (delta None). Scene effect: reference = original scene,
    alternative = intervened scene (empty repair on both). Omission effect: reference =
    r without the omitted actions, alternative = r."""
    if CONTRADICTION in (reference, alternative):
        raise ValueError("contradictory evidence: effect labelling halted")
    y = {POSITIVE: 1, NEGATIVE: 0}
    if reference in y and alternative in y:
        return {"delta": y[alternative] - y[reference], "mask": True}
    return {"delta": None, "mask": False}
