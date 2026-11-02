"""INITIAL_SCENE_ONE_SHOT_V1: post-commit target evaluation, the primary episode with
its commitment boundary, and the separate offline case audit (Methodology v1.5 §2,
§9; Guide v1.7 §3-4).

The target evaluator is injected only into `primary_episode` after COMMIT and into
the privileged `audit_candidate` teacher path. Proposal ranking (proposals.py) and
relocation planning (verify.py) never import this module.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace

import numpy as np

import certificates as cert
import data
import geometry as ge
import motion as mo
import planner as pn
import proposals as pr
import skill as sk
import verify as vf
from config import stable_hash
from scene import scene_hash

VALID, INVALID, UNRESOLVED = sk.VERIFIED, sk.INFEASIBLE, sk.UNRESOLVED      # evidence words (spec §3)


@dataclass
class Trace:
    """Ordered events: PROPOSE, STATIC_CHECK, REPAIR_PLAN, COMMIT, RELOCATE, REOBSERVE,
    TARGET_EVALUATE, FINISH or ABORT, each with its component and state hash."""
    episode: str
    events: list = field(default_factory=list)
    t0: float = field(default_factory=time.perf_counter)

    def emit(self, kind: str, component: str, state, **info):
        self.events.append({"i": len(self.events), "kind": kind, "component": component,
                            "state_hash": state if isinstance(state, str) else scene_hash(state),
                            "t_s": round(time.perf_counter() - self.t0, 3), **info})


def evaluate_target(robot, cfg, state: vf.RepairState, action: dict, key: str, on_witness=None) -> dict:
    """Plan the complete target skill afresh from the final achieved state: its joints,
    gripper gap, attachment (free after every verified retreat) and object poses. One
    call is one skill query with the declared internal schedule (skill2d budgets,
    planner caps). Goal achieved = placement within the goal tolerance. Actions with
    end_policy SAFE_RETREAT (addendum A1) use `evaluate_safe_retreat`; the others keep
    the G3.5 HOME-constrained scope."""
    t = time.perf_counter()
    if state.gap != robot.open_gap:
        return {"status": UNRESOLVED, "reason": "START_GRIPPER_OUT_OF_SKILL_SCOPE", "result": None, "seconds": 0.0}
    if action.get("end_policy") == "SAFE_RETREAT":
        return evaluate_safe_retreat(robot, cfg, state, action, key, on_witness)
    final = "home" if cfg["target_evaluation"]["final_connector"] == "HOME" else None
    res = sk.plan_skill(robot, state.scene, action["target_id"], action["goal_cell"], cfg, key, q_start=state.q,
                        q_goal=final)
    reached = res.status == VALID and res.place_error_m <= action["goal_tolerance_m"]
    return {"status": res.status if reached or res.status != VALID else UNRESOLVED, "reason": res.reason,
            "result": res, "goal_reached": bool(reached), "place_error_m": res.place_error_m,
            "start": {"q": list(state.q), "gripper_gap_m": state.gap, "attachment": "free"},
            "wall_clock_dependent": any(p.wall_clock_cut for _, p in res.plans), "seconds": round(time.perf_counter() - t, 2)}


def evaluate_safe_retreat(robot, cfg, state: vf.RepairState, action: dict, key: str, on_witness=None) -> dict:
    """Addendum A1. Complete target success = approach, grasp, transport, release within
    the goal tolerance AND a certified retreat ending open-handed and clear; the record
    is re-certified independently from its first waypoint (the achieved state) to its
    last. HOME return is a separately planned, separately certified diagnostic that can
    never erase a through-retreat witness. `on_witness(record, arrays)` (G4D checkpoint hook)
    is called with the checked goal + safe-retreat witness before the HOME diagnostic."""
    t = time.perf_counter()
    res = sk.plan_skill(robot, state.scene, action["target_id"], action["goal_cell"], cfg, key, q_start=state.q,
                        q_goal=None)
    out = {"status": res.status, "reason": res.reason, "result": res, "goal_reached": False,
           "goal_achieved": False, "safe_retreat_status": "NOT_REACHED", "home_return_status": "NOT_RUN",
           "complete_target_status": res.status, "home": None, "place_error_m": res.place_error_m,
           "start": {"q": list(state.q), "gripper_gap_m": state.gap, "attachment": "free"},
           "wall_clock_dependent": any(p.wall_clock_cut for _, p in res.plans), "termination": res.termination,
           "unresolved_connectors": sum(p.status == ge.UNRESOLVED for _, p in res.plans)}
    if res.status == VALID:
        arr, delta, eps, ev = res.traj.arrays(), cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"], \
            cfg["geometry"]["certify_max_evals"]
        clear = all(v.status == ge.CLEAR for v in mo.certify_trajectory(robot, state.scene, action["target_id"], arr,
                                                                             delta, eps, ev))
        starts = np.array_equal(arr["q"][0], np.asarray(state.q)) and arr["gap"][0] == state.gap
        retreat = clear and starts and str(arr["phase"][-1]) == "retreat" and arr["gap"][-1] == robot.open_gap
        goal = math.dist(arr["place_xy"], action["goal_xy_m"]) <= action["goal_tolerance_m"]
        out.update(goal_achieved=bool(goal and clear), goal_reached=bool(goal and clear),
                   safe_retreat_status=VALID if retreat else UNRESOLVED,
                   complete_target_status=VALID if goal and retreat else UNRESOLVED)
        out["status"] = out["complete_target_status"]
        if not (goal and retreat):
            out["reason"] = "THROUGH_RETREAT_RECHECK_FAILED"
        elif on_witness is not None:
            on_witness({k: v for k, v in out.items() if k != "result"}, arr)
        if goal and retreat and cfg["target_evaluation"]["home_return_diagnostic"]:
            out["home"] = home_diagnostic(robot, cfg, state.scene, action["target_id"], arr, f"{key}/home")
            out["home_return_status"] = out["home"]["status"]
    out["seconds"] = round(time.perf_counter() - t, 2)
    return out


def home_diagnostic(robot, cfg, scene, target_id: str, arr: dict, key: str) -> dict:
    """Optional: certified connector from the retreat end to HOME with the target placed."""
    p, pc = sk.params(cfg), cfg["planner2d"]
    ck = mo.checkers(robot, scene, target_id, tuple(arr["place_xy"]), None, p["delta"], p["eps"])[("placed", False)]
    plan = None
    for rrt in (False, True):
        plan = pn.plan(ck, arr["q"][-1], np.asarray(robot.q_home), robot.open_gap, pc, p["evals"], key, rrt)
        if plan.status == ge.CLEAR:
            break
    ok = plan.status == ge.CLEAR
    return {"status": VALID if ok else UNRESOLVED, "reason": plan.reason, "nodes": plan.nodes,
            "path": [list(q) for q in plan.path] if ok else None}


def validate_observation(cfg, committed: vf.RepairState, obs: dict) -> list:
    """REOBSERVE: the achieved state must match the committed witness - same identities,
    free gripper at the committed gap, same joints and history; a moved object within
    the declared reobservation tolerance, an unmoved one within the state uncertainty."""
    st, why = obs["state"], []
    if obs.get("attachment", "free") != "free" or obs.get("attached_id") is not None:
        why.append("gripper still attached after release")
    if st.gap != committed.gap:
        why.append(f"gripper gap {st.gap:.4f} != committed {committed.gap:.4f}")
    if not np.array_equal(np.asarray(st.q), np.asarray(committed.q)):
        why.append("joint state differs from the committed retreat end")
    if st.moved != committed.moved:
        why.append("moved history differs")
    a, b = {o.id: o for o in committed.scene.objects}, {o.id: o for o in st.scene.objects}
    if set(a) != set(b) or any(a[i].kind != b[i].kind or a[i].radius_m != b[i].radius_m for i in a if i in b):
        why.append("object identity mismatch")
        return why
    for i, o in a.items():
        tol = cfg["evidence"]["reobserve_pose_tolerance_m"] if i in committed.moved else cfg["evidence"]["state_pose_uncertainty_m"]
        d = math.dist(o.xy_m, b[i].xy_m)
        if d > tol:
            why.append(f"{i}: achieved pose deviates {d * 1000:.2f} mm > {tol * 1000:.2f} mm")
    return why


def advance(state: vf.RepairState, committed: vf.RepairState, oid: str) -> vf.RepairState:
    """Expected state after a committed relocation from the actual (observed) state: the
    committed robot state and history, `oid` at its committed pose, every other object
    where it actually is (earlier admitted deviations are carried forward, not reset)."""
    now = {o.id: o for o in state.scene.objects}
    objs = tuple(o if o.id == oid else now.get(o.id, o) for o in committed.scene.objects)
    return replace(committed, scene=replace(committed.scene, objects=objs))


def start(scene, query: dict) -> vf.RepairState:
    return vf.RepairState(scene, tuple(scene.robot.q_rad), scene.robot.gripper_gap_m, (), query["query_id"],
                          query["hashes"]["action_context"])


def witness_hash(s0: vf.RepairState, att: vf.Attempt, robot_hash: str) -> str:
    before = [s0] + att.states[:-1]
    return stable_hash([mo.trajectory_hash(r.traj.arrays(), o, scene_hash(b.scene), robot_hash)
                        for (o, _), r, b in zip(att.order, att.skills, before)])


def primary_episode(robot, cfg, query: dict, scene, ranked: list, target_eval, robot_hash: str,
                    trace: Trace | None = None, executor=None, planner=None) -> dict:
    """PROPOSE -> per candidate STATIC_CHECK, REPAIR_PLAN (clones) -> COMMIT the first
    relocation witness -> per move RELOCATE (the committed record must start at the
    observed state and certify in the observed scene) and REOBSERVE (validate, else
    ABORT) -> one TARGET_EVALUATE from the observed achieved state -> FINISH. No
    candidate is tried after the commitment, whatever happens. `executor(k, committed
    next state)` returns the observation (default: exactly the committed state); tests
    inject deviations through it (fault injection, not physical robustness).
    `planner(robot, cfg, s0, cand, key, used)` replaces V1 order search; V2 passes the
    fixed-order verifier, and a witness whose order differs from cand["sequence"] is
    never committed (silent reordering aborts the episode before COMMIT)."""
    tr, s0, act = trace or Trace(query["query_id"]), start(scene, query), query["learner"]["action"]
    obs = query["hashes"]["observation"]
    phash = stable_hash({"observation": obs, "action": query["hashes"]["action_context"],
                         "ranked": [[c["label"], sorted((o, pr.json_key(d)) for o, d in c["cells"].items())]
                                    + ([[list(a) for a in c["sequence"]]] if "sequence" in c else []) for c in ranked]})
    tr.emit("PROPOSE", "proposal", obs, candidates=len(ranked), proposal_hash=phash)
    used, tried, commit = {"plans": 0}, [], None
    for rank, cand in enumerate(ranked):
        st = pr.static_endpoints(scene, cand["cells"], act["target_id"], cfg)
        tr.emit("STATIC_CHECK", "static_screen", scene, rank=rank, status=st["status"])
        row = {"rank": rank, "label": cand["label"], "cells": cand["cells"], "static": st, "plan": None}
        tried.append(row)
        if st["status"] != mo.VALID:
            continue
        key = f"{query['query_id']}/primary/c{rank}"
        plan = planner(robot, cfg, s0, cand, key, used) if planner else vf.plan_repair(robot, cfg, s0, cand["cells"], key, used)
        row["plan"] = plan
        tr.emit("REPAIR_PLAN", "relocation_planner", scene, rank=rank, orders=len(plan["attempts"]),
                relocation_plans=plan["relocation_plans"], stop=plan["stop"])
        if plan["witness"]:
            w = plan["witness"]
            if "sequence" in cand and [list(m) for m in w.order] != [list(a) for a in cand["sequence"]]:
                tr.emit("ABORT", "episode", scene, reason="WITNESS_ORDER_DIFFERS_FROM_PROPOSED_SEQUENCE")
                return {"trace": tr, "tried": tried, "commit": None, "target": None, "achieved": None,
                        "outcome": "ABORTED", "abort_reasons": ["silent reordering rejected"]}
            commit = {"rank": rank, "label": cand["label"], "cells": cand["cells"], "order": [list(m) for m in w.order],
                      "witness_hash": witness_hash(s0, w, robot_hash), "proposal_hash": phash,
                      "source_observation_hash": obs, "attempt": w}
            tr.emit("COMMIT", "commitment", scene, rank=rank, witness_hash=commit["witness_hash"])
            break
        if plan["stop"] == "RELOCATION_BUDGET":
            break
    out = {"trace": tr, "tried": tried, "commit": commit, "target": None, "achieved": None}
    if commit is None:
        tr.emit("FINISH", "episode", scene, outcome="NO_REPAIR_WITNESS_WITHIN_BUDGET")
        return {**out, "outcome": "NO_REPAIR_WITNESS_WITHIN_BUDGET"}
    state, w = s0, commit["attempt"]
    for k, ((oid, cell), res, nxt) in enumerate(zip(w.order, w.skills, w.states)):   # execute on a clone of S0
        ok = vf.motion_valid(robot, cfg, state, oid, res)
        tr.emit("RELOCATE", "executor", state.scene, object=oid, cell=cell, certified=ok)
        if not ok:
            tr.emit("ABORT", "episode", state.scene, reason="COMMITTED_MOTION_INVALID_IN_OBSERVED_STATE")
            return {**out, "outcome": "ABORTED", "abort_reasons": ["committed motion invalid in observed state"]}
        expected = advance(state, nxt, oid)
        obs = executor(k, expected) if executor else {"state": expected, "attachment": "free", "attached_id": None}
        why = validate_observation(cfg, expected, obs)
        got = obs["state"]
        tr.emit("REOBSERVE", "executor", got.scene, matches_commitment=not why, reasons=why, object=oid,
                commanded_xy=list(expected.scene.get(oid).xy_m), achieved_xy=list(got.scene.get(oid).xy_m)
                if oid in {o.id for o in got.scene.objects} else None, q=list(got.q), gap=got.gap,
                attachment=obs.get("attachment", "free"), moved=list(got.moved))
        if why:
            tr.emit("ABORT", "episode", obs["state"].scene, reason="ACHIEVED_STATE_DEVIATES_FROM_COMMITMENT")
            return {**out, "outcome": "ABORTED", "abort_reasons": why}
        state = obs["state"]
    out["achieved"] = state
    tr.emit("TARGET_EVALUATE", "target_evaluator", state.scene, start_q=list(state.q), start_gap=state.gap)
    out["target"] = target_eval(state, f"{query['query_id']}/primary/target")
    outcome = "SUCCESS" if out["target"]["status"] == VALID else f"TARGET_{out['target']['status']}"
    tr.emit("FINISH", "episode", state.scene, outcome=outcome, target_reason=out["target"]["reason"])
    return {**out, "outcome": outcome}


def audit_candidate(robot, cfg, query: dict, scene, cand: dict, target_eval, key: str) -> dict:
    """OFFLINE_CASE_AUDIT (privileged teacher path; never feeds a primary episode). Static
    check, then non-dominated orders with their own memo and budget until the first
    complete relocation sequence, then one target evaluation from that order's own
    achieved state. Negative evidence is a structured certificate record that the
    independent checker (certificates.py) must accept; the assignment label is the
    aggregate of all evidence (a witness plus an accepted certificate = CONTRADICTION)."""
    act, cells, learner = query["learner"]["action"], cand["cells"], query["learner"]
    st = pr.static_endpoints(scene, cells, act["target_id"], cfg)
    row = {"label": cand["label"], "cells": cells, "moved": len(cells), "static": st, "orders": [], "goal_blockers": [],
           "certificate": None, "certificates": [], "status": UNRESOLVED, "unknown_reason": None, "evidence": []}
    scope = cert.certificate_scope(scene, learner, cells)
    if st["status"] == mo.INVALID:
        _certify(robot, cfg, scene, learner, cells, row, {**scope, "kind": "STATIC_ENDPOINT_INVALID", "failed": st["failed"]})
        return _close(row, "STATIC_INVALID_NO_MOTION")
    if st["status"] != mo.VALID:
        return _close(row, "STATIC_GRAY_BAND")
    row["goal_blockers"] = pr.goal_blockers(scene, cells, act, cfg)
    if row["goal_blockers"]:
        _certify(robot, cfg, scene, learner, cells, row, {**scope, "kind": "GOAL_OVERLAP", "obstacles": row["goal_blockers"]})
    kept = replace(scene, objects=tuple(o for o in scene.objects if o.id not in cells))
    pick = sk.pick_certificate(robot, kept, act["target_id"], cfg)       # target grasp among staying objects only
    if pick is not None and pick.status == INVALID:
        _certify(robot, cfg, scene, learner, cells, row, {**scope, "kind": "TARGET_PICK", "removed": sorted(cells),
                                                            "candidates": len(pick.picks)})
    s0 = start(scene, query)
    plan = vf.plan_repair(robot, cfg, s0, cells, key, {"plans": 0})
    for att in plan["attempts"]:                        # only the last can be a complete relocation
        o = {"attempt": att, "target": None}
        row["orders"].append(o)
        row["evidence"].append({"kind": "order_attempt", "source": "first_pass", "status": att.status})
        if att.status == vf.RELOCATED:
            o["target"] = target_eval(vf.final_state(s0, att), f"{key}/target")
            if o["target"]["status"] == VALID:
                row["evidence"].append({"kind": "complete_witness", "source": "first_pass"})
    row["pruned_orders"], row["plan_stop"] = plan["pruned_orders"], plan["stop"]
    first = {a.order[0][0]: a for a in plan["attempts"] if a.stop == 0}
    if cells and all(m in first and first[m].skills[0].status == INVALID
                     and first[m].skills[0].reason in ("NO_PICK_CANDIDATE", "NO_PLACE_CANDIDATE") for m in cells):
        movers = {m: {"stage": first[m].skills[0].reason.split("_")[1],
                      "candidates": len(first[m].skills[0].places if "PLACE" in first[m].skills[0].reason
                                        else first[m].skills[0].picks)} for m in cells}
        _certify(robot, cfg, scene, learner, cells, row, {**scope, "kind": "FIRST_MOVER", "movers": movers})
    last = row["orders"][-1] if row["orders"] else None
    return _close(row, f"TARGET_{last['target']['status']}:{last['target']['reason']}" if last and last["target"]
                  else f"RELOCATION_ORDERS:{plan['stop']}")


def _certify(robot, cfg, scene, learner, cells, row, record) -> None:
    ok, why = cert.check(robot, scene, learner, cells, record, cfg)
    row["certificates"].append({**record, "checker_accepted": ok, "checker_reasons": why})
    row["evidence"].append({"kind": "certificate", "validated": ok, "certificate": record["kind"], "source": "first_pass"})


def _close(row: dict, unknown: str) -> dict:
    agg = data.aggregate(row["evidence"])
    row["aggregate"] = agg
    row["status"] = {data.POSITIVE: VALID, data.NEGATIVE: INVALID}.get(agg["label"], agg["label"] if
                                                                        agg["label"] == data.CONTRADICTION else UNRESOLVED)
    accepted = [c for c in row["certificates"] if c["checker_accepted"]]
    if row["status"] == INVALID:
        c = accepted[0]
        row["certificate"] = {"GOAL_OVERLAP": f"GOAL_OVERLAP:{','.join(c.get('obstacles', []))}",
                              "FIRST_MOVER": "FIRST_MOVER_POSE_ONLY_SCREEN_IN_S0",
                              "TARGET_PICK": "TARGET_PICK_BLOCKED_BY_STAYING_OBJECTS"}.get(c["kind"], c["kind"])
    elif row["status"] == UNRESOLVED:
        row["unknown_reason"] = unknown
    return row
