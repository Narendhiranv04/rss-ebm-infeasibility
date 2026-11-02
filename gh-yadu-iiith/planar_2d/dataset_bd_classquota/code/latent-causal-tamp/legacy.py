"""Reuse of archived G4A evidence under V2 scope rules (Guide v1.8 §3 "Legacy reuse";
G4D corrections §2.4 and §4). Moved here from g4d.py (report 0011), which re-exports it.

Positive: an archived first-pass witness labels only its exact recorded order, after
replaying every record (identity, policy, continuity, achieved state, target start,
safe retreat). Negative: each certificate used for transfer is RECONSTRUCTED from the
archived scene, assignment and action context, stamped with the archived identity
hashes and re-run through the independent checker; the archived `checker_accepted`
boolean is never a substitute. Unknowns stay unknown. A positive/negative clash on one
assignment is a CONTRADICTION that blocks downstream collection. Everything accepted
here is DERIVED_LEGACY evidence (V1 backend), never newly collected V2 evidence, and
legacy joint travel is diagnostic only.
"""
from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import certificates as cert
import config as cf
import data
import effects as ef
import geometry as ge
import motion as mo
import ordered as od
import proposals as pr
import sequences as sq
import skill as sk
import teacher
import verify as vf
from scene import scene_from_dict, scene_hash

LEGACY_POLICY = "LEGACY_V1_AUDIT_ORDER_SEARCH_FIRST_RELOCATION_WITNESS"
DERIVED = "DERIVED_LEGACY"


def _rd(path: Path):
    return json.loads(path.read_text())


def _certified(robot, scene, oid, a, d, e, n) -> bool:
    try:
        return all(v.status == ge.CLEAR for v in mo.certify_trajectory(robot, scene, oid, a, d, e, n))
    except ValueError:                                  # malformed record (e.g. attachment inconsistent): not a witness
        return False


def scope_checks(cfg: dict, q: dict) -> dict:
    """Archived query vs the current V2 contract: identities, target, robot profile, grasp
    family, contact rules, tolerances, end policy and move-once."""
    s, act, sk2 = scene_from_dict(q["scene"]), q["action"], cfg["skill2d"]
    v2q = data.make_query(cfg, q["query_id"], s, act["target_id"], q["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
    ids = {o.id: o for o in s.objects}
    return {"observation_hash": v2q["hashes"]["observation"] == q["hashes"]["observation"],
            "action_context_hash": v2q["hashes"]["action_context"] == q["hashes"]["action_context"]
            and v2q["learner"]["action"] == act,
            "target_identity": act.get("target_id") in ids and ids[act["target_id"]].kind == "target",
            "robot_profile": act.get("robot_profile_hash") == cf.robot_profile_hash(cfg),
            "grasp_family": act.get("grasp_family") == {"approach_dirs": sk2["approach_dirs"], "ik_branches": 2,
                                                        "approach_dist_m": sk2["approach_dist_m"],
                                                        "retreat_dist_m": sk2["retreat_dist_m"]},
            "contact_rules": all(act.get(k) == v for k, v in cert.SCOPE_RULES.items()),
            "tolerances": act.get("goal_tolerance_m") == cfg["gripper2d"]["release_xy_tolerance_m"]
            and act.get("state_pose_uncertainty_m") == cfg["evidence"]["state_pose_uncertainty_m"],
            "end_policy": act.get("end_policy") == "SAFE_RETREAT" and act.get("home_return") == "OPTIONAL_DIAGNOSTIC",
            "move_once": bool(cfg["protocol"]["move_once"])}


# ------------------------------------------------------------------ positives
def legacy_positive(cfg, robot, q: dict, tag: str, cells: dict, tasks: dict, scenes: dict, npz) -> dict:
    """Re-validate one archived first-pass witness for its exact recorded order: scope, every
    relocation certified in its own pre-move scene from the previous achieved state
    (hash-bound), placement at the destination, move-once, and the target record starting
    at the last achieved state and ending in a certified safe retreat."""
    s, act = scene_from_dict(q["scene"]), q["action"]
    rhash, d, e, n = cf.robot_profile_hash(cfg), cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"], \
        cfg["geometry"]["certify_max_evals"]
    arr = lambda t: {k.split("__")[1]: npz[k] for k in npz.files if k.startswith(t + "__")}  # noqa: E731
    ms = sorted((k for k in tasks if k.startswith(tag + ".m")), key=lambda k: int(k.rsplit(".m", 1)[1]))
    if tag + ".target" not in tasks:
        return {"v2_status": data.UNKNOWN, "migration": "WITNESS_NOT_ARCHIVED"}
    chk = {"scope": scope_checks(cfg, q)}
    scene, qs, gap, order, recs = s, np.asarray(s.robot.q_rad), s.robot.gripper_gap_m, [], []
    for m in ms:
        a, oid = arr(m), tasks[m]["target"]
        order.append([oid, cells.get(oid)])
        pre = scene_from_dict(scenes[m])
        c = {"pre_scene": scene_hash(pre) == scene_hash(scene), "certified": _certified(robot, pre, oid, a, d, e, n),
             "hash": mo.trajectory_hash(a, oid, scene_hash(pre), rhash) == tasks[m]["action_hash"],
             "starts_at_achieved": bool(np.array_equal(a["q"][0], qs) and a["gap"][0] == gap),
             "placed": oid in cells and math.dist(a["place_xy"], s.lattice.center(cells[oid])) <= 1e-9,
             "ends_in_retreat": str(a["phase"][-1]) == "retreat" and a["gap"][-1] == robot.open_gap}
        chk[m.rsplit(".", 1)[1]] = {k: bool(v) for k, v in c.items()}
        recs.append(a)
        if oid in cells:
            scene = replace(scene, objects=tuple(replace(o, xy_m=tuple(a["place_xy"]), cell=cells[oid]) if o.id == oid else o
                                                 for o in scene.objects))
        qs, gap = a["q"][-1], float(a["gap"][-1])
    t, pre = arr(tag + ".target"), scene_from_dict(scenes[tag + ".target"])
    w = teacher.check_witness(robot, cfg, pre, vf.RepairState(pre, tuple(qs), gap), act, t)
    chk["target"] = {k: bool(v) for k, v in {**w, "pre_scene": scene_hash(pre) == scene_hash(scene), "hash": mo.trajectory_hash(
        t, act["target_id"], scene_hash(pre), rhash) == tasks[tag + ".target"]["action_hash"]}.items()}
    chk["move_once_and_assignment"] = bool(len({o for o, _ in order}) == len(order) and sq.assignment(order) == cells)
    ok = all(all(v.values()) if isinstance(v, dict) else v for v in chk.values())
    jt = od.joint_travel(robot, recs)
    return {"recorded_sequence": order, "v2_status": data.POSITIVE if ok else data.UNKNOWN, "evidence_class": DERIVED,
            "migration": "VALIDATED_EXACT_RECORDED_ORDER" if ok else "VALIDATION_FAILED", "validation": chk,
            "legacy_cost": {"policy": LEGACY_POLICY, "J": 0.0 if not recs else jt["J"], "per_record": jt["per_record"],
                            "usable_as_v2_cost_label": False,
                            "why": "V1 seed keys and order search differ from the frozen V2 first-witness backend"}}


# ------------------------------------------------------------------ negatives (reconstruct + independent checker)
def reconstruct_certificate(robot, cfg: dict, scene, action: dict, cells: dict, kind: str) -> dict:
    """The claim body a producer makes for `kind` on this scene/assignment (the archive kept
    only kind and verdict). The checker decides; an unproducible claim is left empty."""
    if kind == "GOAL_OVERLAP":
        return {"obstacles": pr.goal_blockers(scene, cells, action, cfg)}
    if kind == "TARGET_PICK":
        kept = replace(scene, objects=tuple(o for o in scene.objects if o.id not in cells))
        pick = sk.pick_certificate(robot, kept, action["target_id"], cfg)
        return {"removed": sorted(cells), "candidates": len(pick.picks) if pick else len(sk.directions(sk.params(cfg)))}
    if kind == "FIRST_MOVER":
        scr = {m: od.first_screen(robot, scene, m, d, cfg) for m, d in cells.items()}
        return {"movers": {m: {"stage": v[0], "candidates": v[1]} if v else {"stage": None, "candidates": 0}
                           for m, v in scr.items()}}
    return {}


def revalidate_negative(robot, cfg: dict, q: dict, cells: dict, kinds: list, stamp: dict | None = None) -> dict:
    """New verdict per archived certificate kind. `stamp` = the identity the record claims
    (default: the archived query's hashes and assignment); the checker recomputes it."""
    s, act = scene_from_dict(q["scene"]), q["action"]
    learner = {"scene": data.learner_scene(s), "action": act}
    st = stamp or {"observation_hash": q["hashes"]["observation"], "action_context_hash": q["hashes"]["action_context"],
                   "assignment": [[o, d] for o, d in sorted(cells.items())]}
    sc, verdicts = scope_checks(cfg, q), []
    for k in kinds:
        rec = {**st, "kind": k, **reconstruct_certificate(robot, cfg, s, act, cells, k)}
        ok, why = cert.check(robot, s, learner, cells, rec, cfg)
        verdicts.append({"kind": k, "scope": cert.CERTIFICATE_SCOPE.get(k), "new_checker_accepted": ok,
                         "new_checker_reasons": why, "record": rec})
    wide = [v for v in verdicts if v["new_checker_accepted"] and v["scope"] == "ASSIGNMENT"]
    ok = bool(wide) and all(sc.values())
    return {"scope_checks": sc, "certificates": verdicts, "v2_status": data.NEGATIVE if ok else data.UNKNOWN,
            "evidence_class": DERIVED, "migration": "ASSIGNMENT_WIDE_TRANSFER_REVALIDATED" if ok else "TRANSFER_REJECTED"}


def revalidate(cfg, robot, g4a: Path) -> dict:
    """Every G4A assignment row under the corrected rules, plus a contradiction screen."""
    plan, rows = _rd(g4a / "g4a_plan.json"), _rd(g4a / "candidate_labels.json")
    tasks, scenes, npz = _rd(g4a / "result.json")["tasks"], _rd(g4a / "scenes.json"), np.load(g4a / "arrays.npz")
    qs, idx, out = {q["query_id"]: q for q in plan["queries"]}, {}, []
    for r in rows:
        i = idx.get(r["query_id"], 0)
        idx[r["query_id"]] = i + 1
        cells = {o: dd for o, dd in r["signature"]}
        row = {"query_id": r["query_id"], "audit_index": i, "assignment": cells, "moved": r["moved"],
               "v1_first_pass": r["first_pass"], "v1_diagnostic_enriched": r["diagnostic_enriched"],
               "archived_certificates": r["certificates"]}
        if r["first_pass"] == data.POSITIVE:
            row.update(legacy_positive(cfg, robot, qs[r["query_id"]], f"{r['query_id']}.audit{i}", cells, tasks, scenes, npz))
        elif r["first_pass"] == data.NEGATIVE:
            kinds = sorted({c["kind"] for c in r["certificates"]})
            row.update(revalidate_negative(robot, cfg, qs[r["query_id"]], cells, kinds),
                       orders_covered=math.factorial(r["moved"]))
        else:
            row.update(v2_status=data.UNKNOWN, migration="UNKNOWN_STAYS_UNKNOWN", unknown_reason=r["unknown_reason"])
        out.append(row)
    clash = [(r["query_id"], r["audit_index"]) for r in out if r["v2_status"] == data.NEGATIVE and any(
        p["query_id"] == r["query_id"] and p["v2_status"] == data.POSITIVE and p["assignment"] == r["assignment"] for p in out)]
    return {"rows": out, "contradictions": clash, "queries": qs}


# ------------------------------------------------------------------ derived omission pairs and two-move accounting
def omission_pairs(cfg: dict, v2: dict, rv: dict, file_hashes: dict) -> dict:
    """Deterministic omission comparisons from revalidated records: for each validated
    positive r and each omitted subset S (sizes 1..max, in omission_plan order), the
    remaining r \\ S keeps its order. Its side is POSITIVE only if a validated witness has
    exactly that order, NEGATIVE if a revalidated assignment-wide negative covers its
    assignment, else UNRESOLVED (masked). Delta = Y(r) - Y(r \\ S)."""
    by = {}
    for r in rv["rows"]:
        by.setdefault(r["query_id"], []).append(r)
    pairs = []
    for qid, rows in sorted(by.items()):
        q = rv["queries"][qid]
        s = scene_from_dict(q["scene"])
        query = data.make_query(cfg, qid, s, q["action"]["target_id"], q["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
        anchor = data.appearance_anchor(query, 64)["hash"]
        status, evid = {}, {}
        for r in rows:
            if r["v2_status"] == data.POSITIVE:
                k = tuple(tuple(a) for a in r["recorded_sequence"])
                status[k], evid[k] = data.POSITIVE, [f"{qid}.audit{r['audit_index']}", "witness_replay"]
        for r in rows:
            if r["v2_status"] == data.NEGATIVE:
                evid[("A",) + tuple(sorted(r["assignment"].items()))] = [f"{qid}.audit{r['audit_index']}"] + [
                    v["kind"] for v in r["certificates"] if v["new_checker_accepted"]]
        for r in rows:
            if r["v2_status"] != data.POSITIVE or not r["moved"]:
                continue
            seq = tuple(tuple(a) for a in r["recorded_sequence"])
            st = dict(status)
            ev_ = dict(evid)
            for o in ef.omission_plan(seq, v2):
                rem = tuple(tuple(a) for a in o["remaining"])
                if rem not in st:
                    neg = ("A",) + tuple(sorted(sq.assignment(rem).items()))
                    if neg in evid:
                        st[rem], ev_[rem] = data.NEGATIVE, evid[neg]
            for rec in ef.omission_records(query, seq, st, v2, anchor, ev_):
                rec.update(query_id=qid, evidence_class=DERIVED, source_files=file_hashes)
                pairs.append(rec)
    dec = [p for p in pairs if p["label"]["mask"]]
    return {"pairs": pairs, "summary": {
        "pairs_enumerated": len(pairs), "decided": len(dec), "masked_unknown": len(pairs) - len(dec),
        "decided_delta_plus_one": sum(p["label"]["delta"] == 1 for p in dec),
        "decided_delta_zero": sum(p["label"]["delta"] == 0 for p in dec),
        "decided_delta_minus_one": sum(p["label"]["delta"] == -1 for p in dec),
        "unique_roots_with_decided_pair": sorted({p["query_id"] for p in dec}),
        "unique_roots_with_decided_plus_one": sorted({p["query_id"] for p in dec if p["label"]["delta"] == 1}),
        "evidence_class": DERIVED, "note": "derived from revalidated V1 records; not independent new V2 examples"}}


def two_move_accounting(rv: dict) -> list:
    """Validated exact two-move positives and the checks the general two-move-witness
    requirement needs (continuity, achieved state, target start, safe retreat)."""
    out = []
    for r in rv["rows"]:
        if r["v2_status"] == data.POSITIVE and r["moved"] == 2:
            v = r["validation"]
            out.append({"query_id": r["query_id"], "audit_index": r["audit_index"], "sequence": r["recorded_sequence"],
                        "joint_continuity": all(v[m]["starts_at_achieved"] for m in ("m0", "m1")),
                        "achieved_state_consistency": all(v[m]["pre_scene"] for m in ("m0", "m1")) and v["target"]["pre_scene"],
                        "target_start_equality": v["target"]["starts_at_archived_state"],
                        "safe_retreat": v["target"]["ends_in_safe_retreat"], "scope": all(v["scope"].values()),
                        "evidence_class": DERIVED, "order_scope": "EXACT_RECORDED_ORDER_ONLY",
                        "same_endpoint_contrast": "NOT PROVIDED (other order unresolved)"})
    return out
