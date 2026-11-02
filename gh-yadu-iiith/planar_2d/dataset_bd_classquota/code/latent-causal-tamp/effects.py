"""V2 explanation records: scene interventions at fixed robot state, robot-start
substitutions, repair omissions and their effect labels (Methodology v1.6 §6, §10;
Guide v1.8 §3, §5, §7). G4D migration.

Three different operations, never confused:
  SCENE_POSE_SUBSTITUTION   listed obstacles take other supported poses (ghosts: the
                            same objects, old occupancy removed); robot joints, gap,
                            attachment, task and every other field fixed.
  REPAIR_OMISSION           actions removed from an ordered repair; the omitted objects
                            stay at their initial poses; remaining order preserved.
  OBJECT_DELETION_DIAGNOSTIC  objects removed from a cloned scene; a separately scoped
                            diagnostic, never a repair and never a scene-effect label.
A ROBOT_START_SUBSTITUTION keeps every object fixed and changes only the initial joints.

Learner inputs for both sides of a comparison share the initial appearance anchor
(data.counterfactual_tensors). An effect is definite only when both sides are decided;
an ineligible comparison is reported, never turned into a negative example.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

import data
import geometry as ge
import motion as mo
import proposals as pr
import sequences as sq
import skill as sk
from config import stable_hash
from scene import POSE_EQ_TOL_M, scene_hash, scene_to_dict, substitute_pose

SCENE, OMISSION, DELETION, ROBOT_START = ("SCENE_POSE_SUBSTITUTION", "REPAIR_OMISSION", "OBJECT_DELETION_DIAGNOSTIC",
                                         "ROBOT_START_SUBSTITUTION")


def changed_fields(a, b) -> dict:
    """Field-level audit between two scenes: which objects changed which fields, whether the
    robot state or any non-object field changed."""
    da, db = scene_to_dict(a), scene_to_dict(b)
    oa, ob = {o["id"]: o for o in da["objects"]}, {o["id"]: o for o in db["objects"]}
    objs = {i: sorted(k for k in oa[i] if oa[i][k] != ob[i][k]) for i in oa if i in ob and oa[i] != ob[i]}
    other = sorted(k for k in da if k not in ("objects", "root_id", "variant_id") and da[k] != db[k])
    return {"changed_objects": objs, "added": sorted(set(ob) - set(oa)), "removed": sorted(set(oa) - set(ob)),
            "robot_unchanged": da["robot"] == db["robot"], "other_fields_changed": other}


def _robot_clear(robot, cfg, scene, target_id: str, q, gap: float) -> tuple[bool, str]:
    p = sk.params(cfg)
    if not ge.within_limits(robot, q):
        return False, "joints outside limits"
    v = mo.checkers(robot, scene, target_id, None, None, p["delta"], p["eps"])[("start", False)].status_at(np.asarray(q), gap)
    return v.status == ge.CLEAR, f"robot at the fixed state is {v.status} ({v.pair})"


def pose_substitution(robot, cfg: dict, scene, action: dict, poses: dict) -> dict:
    """Diagnostic edit: obstacles in `poses` take the given legal cells at fixed robot state.
    Eligible only if every pose is supported, the edited endpoints are pairwise clear
    (static factors, target at rest included) and the unchanged initial robot is clear."""
    why, child = [], scene
    for oid, cell in sorted(poses.items()):
        why += sq.validate_sequence(scene, action, ((oid, cell),), 1)
        if not why:
            child = substitute_pose(child, oid, cell)
    if not why:
        st = pr.static_endpoints(scene, poses, action["target_id"], cfg)
        why += [f"edited scene not well-defined: {f['factor']} {f['objects']} {f['status']}" for f in st["failed"]]
        ok, msg = _robot_clear(robot, cfg, child, action["target_id"], scene.robot.q_rad, scene.robot.gripper_gap_m)
        why += [] if ok else [msg]
    audit = changed_fields(scene, child)
    fixed = audit["robot_unchanged"] and not audit["other_fields_changed"] and set(audit["changed_objects"]) <= set(poses)
    return {"kind": SCENE, "objects": sorted(poses), "changes": {o: [scene.get(o).cell, c] for o, c in sorted(poses.items())},
            "eligible": not why and fixed, "reasons": why, "scene": child, "changed_fields": audit,
            "fixed_fields": ["robot.q_rad", "robot.gripper_gap_m", "robot.attachment", "action", "goal",
                             "other objects", "appearance anchor"],
            "fixed_robot_state_hash": stable_hash(scene_to_dict(scene)["robot"]),
            "baseline_scene_hash": scene_hash(scene), "intervened_scene_hash": scene_hash(child)}


def robot_start_substitution(robot, cfg: dict, scene, action: dict, q) -> dict:
    """Same object geometry, different initial joints (open, free gripper). Eligible only if
    within limits and clear of every object in the unchanged scene."""
    child = replace(scene, robot=replace(scene.robot, q_rad=tuple(float(v) for v in q)))
    ok, msg = _robot_clear(robot, cfg, child, action["target_id"], child.robot.q_rad, child.robot.gripper_gap_m)
    audit = changed_fields(scene, child)
    same_objects = not audit["changed_objects"] and not audit["added"] and not audit["removed"]
    return {"kind": ROBOT_START, "q_from": list(scene.robot.q_rad), "q_to": list(child.robot.q_rad),
            "eligible": ok and same_objects, "reasons": [] if ok else [msg], "scene": child, "changed_fields": audit,
            "baseline_scene_hash": scene_hash(scene), "intervened_scene_hash": scene_hash(child)}


def deletion_diagnostic(scene, ids) -> dict:
    """Removal from a clone. Scope DIAGNOSTIC_ONLY: not a legal repair, not a pose
    intervention, never a scene-effect or repair label."""
    child = replace(scene, objects=tuple(o for o in scene.objects if o.id not in set(ids)))
    return {"kind": DELETION, "objects": sorted(ids), "scope": "DIAGNOSTIC_ONLY", "label_use": "NONE", "scene": child,
            "changed_fields": changed_fields(scene, child)}


def effect_record(kind: str, objects: list, changes, reference: dict, alternative: dict, eligible: bool,
                  anchor_hash: str | None, fixed: list, conditioning=None, predicted=None) -> dict:
    """One explanation entry. reference/alternative: {"context_hash", "status", "evidence"}.
    Delta = Y(alternative) - Y(reference), masked unless both sides are decided and the
    comparison is eligible. Necessity is never asserted here."""
    lab = data.effect_label(reference["status"], alternative["status"]) if eligible else {"delta": None, "mask": False}
    return {"kind": kind, "objects": sorted(objects), "changes": changes, "conditioning": conditioning,
            "reference": reference, "alternative": alternative, "eligible": eligible, "label": lab,
            "appearance_anchor": anchor_hash, "fixed_fields": fixed, "predicted": predicted,
            "necessary_cause": "UNESTABLISHED", "minimum_repair": "UNESTABLISHED"}


def omission_plan(seq, v2: dict) -> list:
    """Planned omission comparisons for one sequence under the finite explanation budget:
    (omitted positions, omitted objects, remaining sequence)."""
    e = v2["explanation"]
    return [{"omitted_positions": list(s), "omitted_objects": [seq[i][0] for i in s],
             "remaining": [list(a) for a in sq.omit(seq, s)]}
            for s in sq.omission_subsets(seq, e["max_omission_subset"], e["max_tested_subsets"])]


def context_identity(query: dict, sequence=None, intervention=None, anchor_hash: str | None = None,
                     evidence=None) -> str:
    """Complete identity of one side of a comparison: the query's initial observation (all
    scene and robot fields), action context (task, goal, policy, robot profile), protocol
    and schema, the ordered proposal or intervention, the appearance anchor and the
    evidence identifiers. Identical sequences in different scenes or robot states differ."""
    return stable_hash({"observation": query["hashes"]["observation"], "action_context": query["hashes"]["action_context"],
                        "protocol": query["protocol"], "schema_version": query.get("schema_version"),
                        "sequence": None if sequence is None else [list(a) for a in sequence],
                        "intervention": intervention, "appearance_anchor": anchor_hash, "evidence": evidence})


def omission_records(query: dict, seq, status: dict, v2: dict, anchor_hash: str | None, evidence: dict | None = None) -> list:
    """Effect records for every planned omission of `seq` in `query`, from ordered labels
    keyed by the canonical sequence tuple (missing entries count as UNRESOLVED, never
    negative). `evidence` maps a sequence tuple to its evidence identifiers."""
    key = lambda s: tuple(tuple(a) for a in s)  # noqa: E731
    ev_ = evidence or {}
    side = lambda s: {"context_hash": context_identity(query, s, None, anchor_hash, ev_.get(key(s))),  # noqa: E731
                      "status": status.get(key(s), data.UNKNOWN), "sequence": [list(a) for a in s],
                      "evidence": ev_.get(key(s))}
    out = []
    for o in omission_plan(seq, v2):
        out.append(effect_record(OMISSION, o["omitted_objects"], o["omitted_positions"], side(o["remaining"]), side(seq),
                                 True, anchor_hash, ["initial scene", "robot state", "remaining action order",
                                                     "omitted objects stay"], conditioning=[list(a) for a in seq]))
    return out


def stays_at_initial(scene, seq, omitted: list) -> bool:
    """Every omitted object is at its initial pose in every recomputed prefix of r \\ S."""
    rest = [a for a in seq if a[0] not in set(omitted)]
    return all(np.hypot(*np.subtract(s.get(o).xy_m, scene.get(o).xy_m)) <= POSE_EQ_TOL_M
               for s in sq.prefix_scenes(scene, rest) for o in omitted)
