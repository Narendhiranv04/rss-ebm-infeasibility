"""G3.5 case analysis R0-R6 (Guide v1.7 §4-5): build each query's initial scene, run
the primary one-shot episode, the separate OFFLINE_CASE_AUDIT of every declared
candidate, privileged frozen-path diagnostics and the R6 controls, then derive
evidence and action-level bounds. Records keep the four namespaces apart.
"""
from __future__ import annotations

import copy

import numpy as np

import data
import evaluate as ev
import geometry as ge
import interventions as iv
import motion as mo
import skill as sk
import proposals as pr
import verify as vf
from config import robot_profile_hash, validate_config
from scene import place_scene, scene_hash, scene_to_dict


def build_scene(cfg: dict, fx: dict):
    """(initial scene, oracle_only provenance). Archived G2 scenes are loaded read-only."""
    src = fx["scene"]
    if "source_run" in src:
        frozen, snaps = iv.load_archived(src["source_run"], src["family"])
        return snaps[src["variant"]], {"construction": f"archived {src['family']}/{src['variant']}",
                                       "reference_action_hash": frozen.action_hash, "root_id": snaps["ref"].root_id}
    s = place_scene(cfg, src["root_id"], "initial", src["target_cells"], src["obstacle_cells"])
    return s, {"construction": src.get("lineage", "placed directly"), "root_id": src["root_id"]}


def poison_check(query: dict, image_px: int) -> dict:
    """Learner tensors are identical after poisoning or deleting every oracle_only and
    label field; changing the goal changes the action-context hash."""
    base = data.learner_tensors(query, image_px)
    poisoned = {**query, "oracle_only": {k: "POISON" for k in query["oracle_only"]} | {"blockers": ["o99"]},
                "labels": {"y": 1}}
    deleted = {k: v for k, v in query.items() if k not in ("oracle_only", "labels")}
    same = all(all(np.array_equal(base[k], data.learner_tensors(q, image_px)[k]) for k in base)
               for q in (poisoned, deleted))
    return {"tensors_unchanged_after_poison_and_delete": bool(same), "tensor_keys": sorted(base)}


def run_query(job) -> dict:
    """Pool worker: everything for one query; returns raw results for records and figures."""
    cfg, name, fx = job
    robot, rhash = ge.make_robot(cfg), robot_profile_hash(cfg)
    scene, oracle = build_scene(cfg, fx)
    query = data.make_query(cfg, name, scene, fx["action"]["target"], fx["action"]["goal_cell"], oracle)
    act = query["learner"]["action"]

    def target(state, key, c=cfg):
        return ev.evaluate_target(robot, c, state, act, key)

    ranked = pr.rank_proposals(data.initial_scene(query), act, fx["candidates"], cfg)
    occ = pr.goal_blockers(scene, {}, act, cfg)
    certs = {o: sk.pick_certificate(robot, scene, o, cfg) for o in occ}
    bound = pr.action_lower_bound(scene, act, cfg, {o for o, c in certs.items() if c and c.status == sk.INFEASIBLE})
    primary = ev.primary_episode(robot, cfg, query, scene, ranked, target, rhash)
    pool = [{"label": "all-stay", "cells": {}}] + fx["candidates"]
    audit = [ev.audit_candidate(robot, cfg, query, scene, c, target, f"{name}/audit/{i}") for i, c in enumerate(pool)]
    out = {"name": name, "fx": fx, "query": query, "scene": scene, "ranked": ranked, "bound": bound,
           "pick_certificates": {o: c and c.reason for o, c in certs.items()}, "primary": primary, "audit": audit,
           "poison": poison_check(query, cfg["render"]["image_px"]), "frozen": None, "control": None}
    if "frozen_family" in fx.get("privileged", {}):
        frozen, _ = iv.load_archived(fx["scene"]["source_run"], fx["privileged"]["frozen_family"])
        out["frozen"] = (frozen, iv.evaluate(robot, cfg, frozen, iv.Snapshot("initial", None, None, scene)))
    if "planner_exhaustion" in fx.get("controls", {}):
        c2 = copy.deepcopy(cfg)
        for sec, kv in fx["controls"]["planner_exhaustion"].items():
            c2[sec].update(kv)
        out["control"] = {"overrides": fx["controls"]["planner_exhaustion"],
                          "target": target(ev.start(scene, query), f"{name}/control/exhaustion", validate_config(c2))}
    return out


def bounds(r: dict) -> dict:
    """New-objective bounds: LB from sound scoped premises (default 0), UB from the
    fewest moves among complete repair-plus-target witnesses (primary or audit)."""
    wins = [len(a["cells"]) for a in r["audit"] if a["status"] == sk.VERIFIED]
    if r["primary"]["outcome"] == "SUCCESS":
        wins.append(len(r["primary"]["commit"]["cells"]))
    lb, ub = r["bound"]["lower_bound"], min(wins) if wins else None
    return {"lower_bound": lb, "upper_bound": ub, "premises": r["bound"]["premises"],
            "minimality": "MINIMUM_CERTIFIED" if ub is not None and ub == lb else "MINIMUM_UNRESOLVED"}


def _att(a: vf.Attempt) -> dict:
    return {"order": [list(m) for m in a.order], "relocation_status": a.status, "failed_relocation": a.stop,
            "reason": a.reason, "moved_history": list(vf.final_state(None, a).moved) if a.states else [],
            "wall_clock_dependent": any(p.wall_clock_cut for s in a.skills for _, p in s.plans)}


def _tgt(t: dict | None) -> dict | None:
    return t and {k: t.get(k) for k in ("status", "reason", "goal_reached", "place_error_m", "start", "seconds",
                                         "wall_clock_dependent")}


def record(r: dict) -> dict:
    p, q = r["primary"], r["query"]
    commit = p["commit"] and {k: v for k, v in p["commit"].items() if k != "attempt"}
    rec = {"protocol": q["protocol"], "cases": r["fx"]["cases"], "description": r["fx"].get("description"),
           "learner": q["learner"], "hashes": q["hashes"], "oracle_only": q["oracle_only"],
           "ranked_proposals": r["ranked"], "bounds": bounds(r), "pick_certificates": r["pick_certificates"],
           "leakage_audit": r["poison"],
           "execution": {"outcome": p["outcome"], "commit": commit, "events": p["trace"].events,
                         "achieved_moved_history": list(p["achieved"].moved) if p["achieved"] else None,
                         "target": _tgt(p["target"]),
                         "candidates_tried": [{"rank": t["rank"], "label": t["label"], "static_status": t["static"]["status"],
                                               "orders": [_att(a) for a in t["plan"]["attempts"]] if t["plan"] else [],
                                               "stop": t["plan"] and t["plan"]["stop"]} for t in p["tried"]]},
           "offline_case_audit": [{"label": a["label"], "cells": a["cells"], "moved": a["moved"],
                                   "static_status": a["static"]["status"], "static_failed": a["static"]["failed"] or a["static"]["errors"],
                                   "goal_blockers": a["goal_blockers"],
                                   "orders": [{**_att(o["attempt"]), "target": _tgt(o["target"])} for o in a["orders"]],
                                   "assignment_feasibility_status": a["status"], "certificate": a["certificate"],
                                   "unknown_reason": a["unknown_reason"]} for a in r["audit"]]}
    if r["frozen"]:
        fr, snap = r["frozen"]
        rec["oracle_only"] = {**rec["oracle_only"], "frozen_diagnostic": {
            "protocol": "FROZEN_CONTROL (privileged diagnostic only)", "action_hash": fr.action_hash, "F_fixed": snap.F,
            "blockers": [i for i, c in snap.checks.items() if c.status == mo.INVALID]}}
    if r["control"]:
        rec["controls"] = {"planner_exhaustion": {"overrides": r["control"]["overrides"],
                                                  "target": _tgt(r["control"]["target"]),
                                                  "assignment_feasibility_status": sk.UNRESOLVED,
                                                  "note": "finite budget: unknown, never a negative or a bound"}}
    return rec


def archive(res: dict, robot_hash: str):
    """Replay-compatible tasks/scenes/arrays: each recorded relocation in its pre-move
    scene and each target trajectory in its achieved scene."""
    tasks, scenes, arrays = {}, {}, {}

    def put(tid, scene, target, res_):
        arr = res_.traj.arrays()
        scenes[tid] = scene_to_dict(scene)
        tasks[tid] = {"target": target, "action_hash": mo.trajectory_hash(arr, target, scene_hash(scene), robot_hash)}
        arrays.update({f"{tid}__{k}": v for k, v in arr.items()})

    for name, r in sorted(res.items()):
        eps = [("primary", r["primary"]["commit"] and r["primary"]["commit"]["attempt"], r["primary"]["target"])]
        eps += [(f"audit{i}", o["attempt"], o["target"]) for i, a in enumerate(r["audit"]) for o in a["orders"]
                if o["target"]]
        for tag, att, tgt in eps:
            if att is None:
                continue
            before = [r["scene"]] + [s.scene for s in att.states[:-1]]
            for k, ((oid, _), sres, s) in enumerate(zip(att.order, att.skills, before)):
                put(f"{name}.{tag}.m{k}", s, oid, sres)
            if tgt and tgt["result"] and tgt["result"].traj is not None:
                put(f"{name}.{tag}.target", att.states[-1].scene if att.states else r["scene"], r["query"]["learner"]
                    ["action"]["target_id"], tgt["result"])
    return tasks, scenes, arrays


def readiness(res: dict) -> dict:
    """G3.5 instruction §12 matrix, computed from the recorded evidence (PASS / BLOCKED)."""
    by = {c: [r for r in res.values() if c in r["fx"]["cases"]] for c in ("R0", "R1", "R2", "R3", "R4", "R5", "R6")}
    rec = {n: record(r) for n, r in res.items()}
    events = [e for r in rec.values() for e in r["execution"]["events"]]

    def ok(flag):
        return "PASS" if flag else "BLOCKED"
    calls_ok = all(_boundary_ok(r["execution"]["events"]) for r in rec.values())
    r0 = any(bounds(r)["upper_bound"] == 0 and bounds(r)["minimality"] == "MINIMUM_CERTIFIED" for r in by["R0"])
    r1 = any(r["frozen"] and r["frozen"][1].F == mo.INVALID and r["primary"]["outcome"] == "SUCCESS"
             and not r["primary"]["commit"]["cells"] for r in by["R1"])
    r2 = any(bounds(r)["lower_bound"] >= 1 and bounds(r)["upper_bound"] == bounds(r)["lower_bound"] for r in by["R2"])
    r3 = any(len({json_cells(a["cells"]) for a in r["audit"] if a["status"] == sk.VERIFIED}) >= 2 for r in by["R3"])
    r4 = any(any(a["status"] == sk.VERIFIED and len(a["orders"]) >= 2 and a["orders"][0]["attempt"].status != vf.RELOCATED
                 for a in r["audit"]) for r in by["R4"])
    r5 = any(any(a["static"]["status"] == mo.VALID and a["status"] != sk.VERIFIED for a in r["audit"]) for r in by["R5"])
    return {"goal_conditioned_query_without_trajectory_fields": ok(all(r["poison"]["tensors_unchanged_after_poison_and_delete"]
                                                                       for r in res.values())),
            "primary_selection_has_no_target_feedback": ok(calls_ok and bool(events)),
            "R0_zero_move_witness_and_R1_objective_distinction": ok(r0 and r1),
            "R2_occupancy_certificate_and_witness": ok(r2),
            "R3_alternative_positives_and_R4_order_dependence": ok(r3 and r4),
            "R5_static_valid_not_action_valid": ok(r5)}


def _boundary_ok(events: list) -> bool:
    """No TARGET_EVALUATE before COMMIT or between RELOCATEs; at most one; nothing after."""
    kinds = [e["kind"] for e in events]
    if "TARGET_EVALUATE" not in kinds:
        return "COMMIT" not in kinds or "ABORT" in kinds
    t = kinds.index("TARGET_EVALUATE")
    return (kinds.count("TARGET_EVALUATE") == 1 and "COMMIT" in kinds[:t] and "RELOCATE" not in kinds[t:]
            and kinds[t + 1:] == ["FINISH"])


def json_cells(cells: dict) -> tuple:
    return tuple(sorted((o, pr.json_key(d)) for o, d in cells.items()))


def summary_rows(records: dict) -> list:
    """One metrics row per query (primary episode outcome, bounds, timing)."""
    rows = []
    for n, r in records.items():
        e = r["execution"]
        rows.append({"query": n, "cases": " ".join(r["cases"]), "outcome": e["outcome"],
                     "committed": e["commit"] and e["commit"]["label"], "moved": e["commit"] and len(e["commit"]["cells"]),
                     **{k: r["bounds"][k] for k in ("lower_bound", "upper_bound", "minimality")},
                     "target_seconds": e["target"] and e["target"]["seconds"], "episode_seconds": e["events"][-1]["t_s"]})
    return rows
