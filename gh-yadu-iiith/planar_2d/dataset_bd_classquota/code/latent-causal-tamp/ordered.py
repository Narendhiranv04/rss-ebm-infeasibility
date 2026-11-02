"""INITIAL_SCENE_ORDERED_ONE_SHOT_V2 evidence, cost and one-shot execution checks
(Methodology v1.6 §4, §8, §10, §12; Guide v1.8 §3, §7, §9). G4D migration.

Teacher path (privileged, offline): `teach_sequence` restarts from the immutable
initial state, verifies exactly the proposed order (verify.plan_fixed_order), stores
the FIRST complete relocation witness and its normalized joint travel before looking
at the target, then evaluates the target once from that witness's achieved state.
Evidence keeps separate status fields and explicit scopes: a failed trajectory is
never an order-wide negative, a witness labels only its own order.

Primary path: `primary_v2` runs evaluate.primary_episode with the fixed-order planner
(no reordering, frozen pool, at most one target call after the repair completes) and
`episode_audit` re-checks the boundary, order, joint continuity and target start.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

import certificates as cert
import data
import evaluate as ev
import geometry as ge
import motion as mo
import proposals as pr
import sequences as sq
import skill as sk
import verify as vf

NOT_RUN, NOT_AUTHORIZED = "NOT_RUN", "NOT_AUTHORIZED"


# ------------------------------------------------------------------ cost (first relocation witness)
def joint_travel(robot, records: list) -> dict:
    """J(w) = sum over consecutive waypoints of sum_j |dq_j| / (q_max_j - q_min_j) over every
    relocation record in order: pregrasp, approach, closure, carry, release, retreat and
    the transitions between relocations. Gripper aperture is excluded. The backend's
    segments are joint-linear between waypoints inside the finite limits [-pi, pi], so
    the differences are the actual unwrapped motion (no wrap shortcut can exist; a large
    certified step is real travel). A waypoint outside the limits or a discontinuous
    transition (next record not starting at the previous end in q and gap) makes the
    label unavailable."""
    rng = np.array([hi - lo for lo, hi in robot.limits], float)
    problems, per_record, per_phase = [], [], {}
    if not np.all(np.isfinite(rng)) or np.any(rng <= 0):
        problems.append("joint range not finite and nonzero")
    for k, arr in enumerate(records):
        q = np.asarray(arr["q"], float)
        d = np.abs(np.diff(q, axis=0))
        if not all(ge.within_limits(robot, x) for x in q):
            problems.append(f"record {k}: waypoint outside the joint limits (unwrapped path not established)")
        if k and not (np.array_equal(np.asarray(records[k - 1]["q"])[-1], q[0]) and records[k - 1]["gap"][-1] == arr["gap"][0]):
            problems.append(f"record {k}: does not start where record {k - 1} ended")
        seg = (d / rng).sum(axis=1)
        per_record.append(float(seg.sum()))
        for ph, v in zip(arr["phase"], seg):
            per_phase[str(ph)] = per_phase.get(str(ph), 0.0) + float(v)
    return {"J": None if problems else float(sum(per_record)), "per_record": per_record, "per_phase": per_phase,
            "problems": problems, "available": not problems, "joint_ranges_rad": rng.tolist()}


def cost_record(robot, v2: dict, s0: vf.RepairState, att: vf.Attempt | None, robot_hash: str) -> dict:
    """Cost label of the declared first-witness policy. Empty sequence: exactly 0. No
    complete relocation witness: no label (never a large or zero placeholder)."""
    base = {"policy": v2["cost"]["policy"], "backend": v2["cost"]["relocation_backend"],
            "start_q": list(s0.q), "start_gap": s0.gap}
    if att is None or att.status != vf.RELOCATED:
        return {**base, "label_available": False, "J": None, "reason": "NO_COMPLETE_RELOCATION_WITNESS"}
    arrs = [r.traj.arrays() for r in att.skills]
    jt = joint_travel(robot, arrs)
    starts = not arrs or (np.array_equal(arrs[0]["q"][0], np.asarray(s0.q)) and arrs[0]["gap"][0] == s0.gap)
    ok = jt["available"] and starts
    end = vf.final_state(s0, att)
    return {**base, "label_available": ok, "J": (0.0 if not arrs else jt["J"]) if ok else None,
            "reason": "OK" if ok else "; ".join(jt["problems"] + ([] if starts else ["first record not at S0"])),
            "witness_hash": ev.witness_hash(s0, att, robot_hash), "end_q": list(end.q), "end_gap": end.gap,
            "waypoints": int(sum(len(a["q"]) for a in arrs)), "per_record": jt["per_record"],
            "per_phase": jt["per_phase"], "joint_ranges_rad": jt["joint_ranges_rad"]}


# ------------------------------------------------------------------ certificate producers (checked in certificates.py)
def first_screen(robot, scene, oid: str, cell: int, cfg: dict):
    """(stage, family size) if `oid` moved first from `scene` to `cell` is excluded by a
    definite pose-only screen over the declared grasp family, else None."""
    p = sk.params(cfg)
    o, n = scene.get(oid), len(sk.directions(p))
    picks = sk.screen_pick(robot, mo.checkers(robot, scene, oid, None, None, p["delta"], p["eps"]), o.xy_m, o.radius_m, p)
    if not any(c.stage == "OK" for c in picks):
        return ("PICK", n) if all(c.certified for c in picks) else None
    dest, local = scene.lattice.center(cell), (robot.grasp_x, 0.0)
    ck = mo.checkers(robot, scene, oid, dest, local, p["delta"], p["eps"])
    places = sk.screen_place(robot, ck, dest, local, 2 * o.radius_m, p)
    ok = not any(c.stage == "OK" for c in places) and all(c.certified for c in places)
    return ("PLACE", n) if ok else None


def certificates_for(robot, cfg: dict, scene, learner: dict, seq) -> list:
    """Candidate negative records for one ordered proposal, each with its scope and the
    independent checker's verdict. Producers propose; only accepted records count."""
    cells, act = sq.assignment(seq), learner["action"]
    scope = cert.certificate_scope(scene, learner, cells)
    recs = []
    if pr.static_endpoints(scene, cells, act["target_id"], cfg)["status"] == mo.INVALID:
        recs.append({**scope, "kind": "STATIC_ENDPOINT_INVALID"})
    else:
        occ = pr.goal_blockers(scene, cells, act, cfg)
        if occ:
            recs.append({**scope, "kind": "GOAL_OVERLAP", "obstacles": occ})
        kept = replace(scene, objects=tuple(o for o in scene.objects if o.id not in cells))   # staying objects only
        pick = sk.pick_certificate(robot, kept, act["target_id"], cfg)
        if pick is not None and pick.status == sk.INFEASIBLE:
            recs.append({**scope, "kind": "TARGET_PICK", "removed": sorted(cells), "candidates": len(pick.picks)})
        screens = {o: first_screen(robot, scene, o, d, cfg) for o, d in seq}
        if seq and all(screens.values()):
            recs.append({**scope, "kind": "FIRST_MOVER", "movers": {o: {"stage": s, "candidates": n}
                                                                   for o, (s, n) in screens.items()}})
        if seq and screens[seq[0][0]]:
            s, n = screens[seq[0][0]]
            recs.append({**scope, "kind": "ORDER_FIRST_ACTION", "sequence": [list(a) for a in seq], "stage": s,
                         "candidates": n})
    out = []
    for r in recs:
        ok, why = cert.check(robot, scene, learner, cells, r, cfg)
        out.append({**r, "scope": cert.CERTIFICATE_SCOPE[r["kind"]], "checker_accepted": ok, "checker_reasons": why})
    return out


# ------------------------------------------------------------------ explicit termination -> normalized V2 reasons
def normalized_reason(status: str, termination: str, wall_cut: bool = False, unresolved_connectors: int = 0) -> str:
    """V2 reason from the explicit end event; the historical `reason` string is kept beside
    it. Only an independent, scope-matching proof turns any of these into a negative."""
    base = {"WITNESS_RECHECKED": "WITNESS", "WITNESS_REPLAY_REJECTED": "WITNESS_REPLAY_REJECTED",
            "SEARCH_LIMIT_PLAN_FAILURES": "SEARCH_LIMIT_PLAN_FAILURES",
            "SCHEDULE_COMPLETED_ALL_CERTIFIED_BLOCKED": "SCHEDULE_COMPLETED_ALL_PAIRS_CERTIFIED_BLOCKED",
            "SCHEDULE_COMPLETED_NO_WITNESS": "SCHEDULE_COMPLETED_NO_WITNESS"
            + ("_WITH_UNRESOLVED_CONNECTORS" if unresolved_connectors else "_UNCERTIFIED_REJECTIONS"),
            "START_STATE_NOT_CLEAR": "START_STATE_NOT_CLEAR"}.get(termination)
    if base is None and termination.endswith("_SCREEN_EXHAUSTED"):
        base = termination + ("_CERTIFIED" if status == sk.INFEASIBLE else "_UNCERTIFIED")
    return (base or f"UNCLASSIFIED:{termination or 'NONE'}") + ("+WALL_LIMIT_INVOLVED" if wall_cut else "")


def _skill_reason(res) -> str:
    return normalized_reason(res.status, res.termination, any(p.wall_clock_cut for _, p in res.plans),
                             sum(p.status == "UNRESOLVED" for _, p in res.plans))


def state_record(st: vf.RepairState) -> dict:
    return {"q": [float(v) for v in st.q], "gap": float(st.gap), "moved": list(st.moved), "attachment": "free",
            "objects": {o.id: [o.kind, o.radius_m, list(o.xy_m), o.cell] for o in st.scene.objects}}


# ------------------------------------------------------------------ offline teacher for one ordered proposal
def teach_sequence(robot, cfg: dict, v2: dict, query: dict, scene, seq, target_eval, key: str, robot_hash: str,
                   checkpoint=None) -> dict:
    """Evidence for one ordered proposal from the immutable initial state. The raw proposal
    is validated before any conversion (sequences.parse_sequence). Stages, each handed to
    `checkpoint(stage, record, arrays)` before the next starts: accepted/rejected
    certificates; the fixed-order relocation result with its first-witness cost (partial
    records only as diagnostics); then one target evaluation from that witness's achieved
    state, skipped (NOT_RUN) when an accepted certificate already covers this order."""
    seq, why = sq.parse_sequence(scene, query["learner"]["action"], seq, v2["proposal"]["move_cap"])
    if why:
        raise ValueError(f"not a canonical V2 sequence: {why}")
    put = checkpoint or (lambda *a: None)
    learner, s0 = query["learner"], ev.start(scene, query)
    certs = certificates_for(robot, cfg, scene, learner, seq)
    put("certificates", {"certificates": certs}, None)
    evidence = [{"kind": "certificate", "validated": c["checker_accepted"], "certificate": c["kind"], "scope": c["scope"],
                 "source": "teacher"} for c in certs]
    static = pr.static_endpoints(scene, sq.assignment(seq), learner["action"]["target_id"], cfg)["status"]
    rec = {"sequence": [list(a) for a in seq], "assignment": sq.assignment(seq), "moved": len(seq), "static_status": static,
           "certificates": certs, "relocation_attempt_status": NOT_RUN, "target_attempt_status": NOT_RUN,
           "simulation_status": NOT_RUN, "hardware_status": NOT_AUTHORIZED, "cost": None, "relocation": None,
           "target": None, "target_start_matches_achieved": None, "calls": {"relocation_planner": 0, "target_evaluations": 0}}
    first_blocked = any(c["kind"] in ("ORDER_FIRST_ACTION", "STATIC_ENDPOINT_INVALID") and c["checker_accepted"]
                        for c in certs)
    if static == mo.VALID and not first_blocked:
        plan = vf.plan_fixed_order(robot, cfg, s0, seq, f"{key}/reloc", {"plans": 0})
        att = plan["attempts"][0]
        rec["calls"]["relocation_planner"] = plan["relocation_plans"]
        rec["relocation_attempt_status"] = vf.RELOCATED if plan["witness"] else data.UNKNOWN
        rec["relocation"] = {"stop": plan["stop"], "policy": plan["policy"], "relocation_plans": plan["relocation_plans"],
                             "failed_action": att.stop, "reason": att.reason,
                             "normalized_reasons": [_skill_reason(r) for r in att.skills],
                             "complete_witness": plan["witness"] is not None,
                             "achieved_states": [state_record(s0)] + [state_record(x) for x in att.states]}
        rec["cost"] = cost_record(robot, v2, s0, plan["witness"], robot_hash)
        rec["_attempt"] = att
        put("relocation", {"relocation": rec["relocation"], "cost": rec["cost"]},
            [r.traj.arrays() for r in att.skills if r.traj is not None and r.status == sk.VERIFIED])
        if not plan["witness"]:
            evidence.append({"kind": "order_attempt", "scope": data.TRAJECTORY, "source": "teacher", "status": att.status})
        elif not any(c["checker_accepted"] for c in certs):
            achieved = vf.final_state(s0, plan["witness"])
            rec["calls"]["target_evaluations"] = 1
            t = target_eval(achieved, f"{key}/target")
            rec["target"], rec["target_attempt_status"] = t, t["status"]
            if t.get("result") is not None:
                t["normalized_reason"] = _skill_reason(t["result"])
            arr = t.get("result") and t["result"].traj is not None and t["result"].traj.arrays()
            inv = t.get("start") or {}
            rec["target_start_matches_achieved"] = bool(
                inv.get("q") == [float(v) for v in achieved.q] and inv.get("gripper_gap_m") == achieved.gap
                and (not arr or (np.array_equal(arr["q"][0], np.asarray(achieved.q)) and arr["gap"][0] == achieved.gap)))
            if t["status"] == ev.VALID and arr and rec["target_start_matches_achieved"]:
                evidence.append({"kind": "complete_witness", "scope": data.ORDER, "source": "teacher"})
    rec["aggregate"] = data.ordered_aggregate(evidence)
    rec["ordered_feasibility_status"] = rec["aggregate"]["label"]
    rec["evidence"] = evidence
    return rec


# ------------------------------------------------------------------ V2 primary episode
def fixed_order_planner(robot, cfg, s0, cand, key, used):
    return vf.plan_fixed_order(robot, cfg, s0, cand["sequence"], key, used)


def primary_v2(robot, cfg, query: dict, scene, ranked: list, target_eval, robot_hash: str, trace=None, executor=None):
    """One-shot V2 episode over a frozen ranked pool (sequences.rank_sequences). Every
    target invocation's input state is recorded for the audit."""
    calls = []

    def recorded(state, key):
        calls.append(state_record(state))
        return target_eval(state, key)
    cands = [{**r, "cells": dict(r["assignment"])} for r in ranked]
    out = ev.primary_episode(robot, cfg, query, scene, cands, recorded, robot_hash, trace, executor,
                             planner=fixed_order_planner)
    out["target_invocations"] = calls
    out["audit"] = episode_audit(out, ev.start(scene, query), ranked, cfg)
    return out


def episode_audit(out: dict, s0: vf.RepairState, ranked: list, cfg: dict) -> dict:
    """Recomputed one-shot contract. Each check is True / False / None (applicable but
    evidence missing); `decision` is PASS only if every applicable check is True, FAIL if
    any is False, else UNKNOWN. Protocol compliance (the event contract) is separate from
    the episode outcome: a no-witness or aborted episode can be compliant and unsuccessful."""
    ev_ = out["trace"].events
    kinds = [e["kind"] for e in ev_]
    t = [i for i, k in enumerate(kinds) if k == "TARGET_EVALUATE"]
    commits = [i for i, k in enumerate(kinds) if k == "COMMIT"]
    aborts = [i for i, k in enumerate(kinds) if k == "ABORT"]
    ends = [i for i, k in enumerate(kinds) if k in ("FINISH", "ABORT")]
    w = out["commit"] and out["commit"].get("attempt")
    n = len(w.order) if w else 0
    pre = kinds[:t[0]] if t else []
    reloc_ok = [e.get("certified") for e in ev_[:t[0]] if e["kind"] == "RELOCATE"] if t else []
    reob_ok = [e.get("matches_commitment") for e in ev_[:t[0]] if e["kind"] == "REOBSERVE"] if t else []
    c = {"single_terminal_event_last": len(ends) == 1 and ends[0] == len(kinds) - 1,
         "commits_at_most_one": len(commits) <= 1, "target_calls_at_most_one": len(t) <= 1,
         "no_target_before_commit": all(commits and i > commits[0] for i in t),
         "no_target_after_abort": all(not aborts or i < aborts[0] for i in t) and not (aborts and t),
         "target_after_all_relocations_and_reobservations": (not t) or (
             w is not None and pre.count("RELOCATE") == n and pre.count("REOBSERVE") == n
             and all(reloc_ok) and all(reob_ok) and "ABORT" not in pre),
         "no_target_without_completed_repair": bool(t) == (out.get("achieved") is not None)}
    applicable = out.get("achieved") is not None and bool(t)
    if w is not None:
        c["committed_order_equals_proposed"] = [list(m) for m in w.order] == [list(a) for a in
                                                                              ranked[out["commit"]["rank"]]["sequence"]]
        cont, prev = True, (np.asarray(s0.q), s0.gap)
        for res in w.skills:
            a = res.traj.arrays()
            cont &= bool(np.array_equal(a["q"][0], prev[0]) and a["gap"][0] == prev[1])
            prev = (a["q"][-1], a["gap"][-1])
        c["joint_continuity"] = cont
    if applicable:
        ach, final = out["achieved"], vf.final_state(s0, w)
        c["achieved_state_matches_commitment"] = not ev.validate_observation(cfg, final, {"state": ach})
        inv = out.get("target_invocations")
        c["target_invocation_equals_achieved"] = None if inv is None or len(inv) != 1 else (
            inv[0] == state_record(ach) and inv[0]["q"] == [float(v) for v in prev[0]] and inv[0]["gap"] == float(prev[1]))
        tgt = out["target"] or {}
        arr = tgt.get("result") is not None and tgt["result"].traj is not None and tgt["result"].traj.arrays()
        if arr:
            c["target_trajectory_starts_at_invocation"] = bool(np.array_equal(arr["q"][0], np.asarray(ach.q))
                                                               and arr["gap"][0] == ach.gap)
    protocol = [c[k] for k in ("single_terminal_event_last", "commits_at_most_one", "target_calls_at_most_one",
                               "no_target_before_commit", "no_target_after_abort",
                               "target_after_all_relocations_and_reobservations", "no_target_without_completed_repair")]
    vals = list(c.values())
    return {"checks": c, "protocol_compliant": all(protocol), "outcome": out.get("outcome"),
            "decision": "FAIL" if any(v is False for v in vals) else "UNKNOWN" if any(v is None for v in vals) else "PASS",
            "target_calls": len(t), "commits": len(commits),
            "reobserve": [{k: e.get(k) for k in ("object", "commanded_xy", "achieved_xy", "q", "gap", "attachment", "moved",
                                                 "matches_commitment")} for e in ev_ if e["kind"] == "REOBSERVE"]}
