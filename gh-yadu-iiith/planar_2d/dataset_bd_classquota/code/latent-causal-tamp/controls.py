"""G4A controlled checks (G4A instruction §3, §4.2): the adversarial certificate suite
(each sound proof is accepted; removing or corrupting one premise is rejected) with a
re-audit of the G3.5 certificates and R4 bound, and the execution fault-injection
episodes. FAULT-INJECTION TESTS: they exercise the abort logic, not physical robustness.
"""
from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import certificates as cert
import data
import evaluate as ev
import proposals as pr
import skill as sk
from config import robot_profile_hash, validate_config
from scene import place_scene, scene_from_dict


def _query(cfg, scene, goal, end_policy="SAFE_RETREAT"):
    return data.make_query(cfg, "control", scene, "t0", goal, {}, end_policy)


def _row(name, kind, expected, result, note=""):
    ok, why = result
    return {"case": name, "kind": kind, "expected_accept": expected, "accepted": ok, "reasons": why,
            "as_expected": ok == expected, "note": note}


def certificate_suite(robot, cfg: dict) -> list:
    """Accepted and rejected premises on small full-3R scenes (R2: goal occupant o03 at 45;
    R4: goal occupant o00 at 28, pick-blocked by o01)."""
    out, delta, r = [], cfg["suite"]["margin_m"], cfg["suite"]["obstacle_radius_m"]
    s2 = place_scene(cfg, "c_r2", "initial", [18], [3, 13, 50, 45])
    q2 = _query(cfg, s2, 45)
    L2, scope = q2["learner"], cert.certificate_scope(s2, q2["learner"], {})
    goal = {**scope, "kind": "GOAL_OVERLAP", "obstacles": ["o03"]}
    out.append(_row("goal occupied exactly (R2 all-stay)", "GOAL_OVERLAP", True, cert.check(robot, s2, L2, {}, goal, cfg)))
    out.append(_row("premise removed: no blocker named", "GOAL_OVERLAP", False,
                    cert.check(robot, s2, L2, {}, {**goal, "obstacles": []}, cfg)))
    out.append(_row("stale object ID", "GOAL_OVERLAP", False, cert.check(robot, s2, L2, {}, {**goal, "obstacles": ["o99"]}, cfg)))
    out.append(_row("stale goal (action changed after production)", "GOAL_OVERLAP", False,
                    cert.check(robot, s2, _query(cfg, s2, 46)["learner"], {}, goal, cfg)))
    push = copy.deepcopy(L2)
    push["action"]["pushing_allowed"] = True
    out.append(_row("contact rule altered: pushing allowed", "GOAL_OVERLAP", False, cert.check(robot, s2, push, {}, goal, cfg)))
    esc = copy.deepcopy(L2)
    esc["action"]["goal_alternatives_xy"] = [list(s2.lattice.center(47))]
    out.append(_row("goal region with a free escape pose", "GOAL_OVERLAP", False,
                    cert.check(robot, s2, esc, {}, {**cert.certificate_scope(s2, esc, {}), "kind": "GOAL_OVERLAP",
                                                    "obstacles": ["o03"]}, cfg)))
    gxy = s2.lattice.center(45)
    for name, clearance, tol, expect in (("close to margin, clearance 4.0 mm < delta - eps", 0.004, None, True),
                                         ("close to margin, clearance 4.9 mm in the gray band", 0.0049, None, False),
                                         ("goal tolerance 10 mm reaches a clear placement", 0.004, 0.010, False)):
        d = clearance + 2 * r - cfg["gripper2d"]["release_xy_tolerance_m"]
        sc = _offgrid(cfg, s2, "o03", (gxy[0] + d, gxy[1]))
        qq = _query(cfg, sc, 45)
        if tol is not None:
            qq["learner"]["action"]["goal_tolerance_m"] = tol
        c = {**cert.certificate_scope(sc, qq["learner"], {}), "kind": "GOAL_OVERLAP", "obstacles": ["o03"]}
        out.append(_row(name, "GOAL_OVERLAP", expect, cert.check(robot, sc, qq["learner"], {}, c, cfg),
                        f"centre distance {d * 1000:.2f} mm"))
    big = replace(s2, objects=tuple(replace(o, radius_m=0.02) if o.id == "o03" else o for o in s2.objects))
    out.append(_row("altered radius after production", "GOAL_OVERLAP", False, cert.check(robot, big, L2, {}, goal, cfg)))
    s4 = place_scene(cfg, "c_r4", "initial", [20], [28, 27, 31, 58, 63])
    L4 = _query(cfg, s4, 28)["learner"]
    n = len(sk.directions(sk.params(cfg)))
    fm = lambda cells, movers: {**cert.certificate_scope(s4, L4, cells), "kind": "FIRST_MOVER", "movers": movers}  # noqa: E731
    one = {"o00": 29}
    out.append(_row("o00 cannot be picked first (R4)", "FIRST_MOVER", True,
                    cert.check(robot, s4, L4, one, fm(one, {"o00": {"stage": "PICK", "candidates": n}}), cfg)))
    two = {"o00": 29, "o01": 18}
    out.append(_row("a selected object (o01) can go first", "FIRST_MOVER", False, cert.check(
        robot, s4, L4, two, fm(two, {m: {"stage": "PICK", "candidates": n} for m in two}), cfg)))
    out.append(_row("one declared grasp omitted from the record", "FIRST_MOVER", False,
                    cert.check(robot, s4, L4, one, fm(one, {"o00": {"stage": "PICK", "candidates": n - 1}}), cfg)))
    out.append(_row("start-dependent connector failure offered", "FIRST_MOVER", False,
                    cert.check(robot, s4, L4, one, fm(one, {"o00": {"stage": "CONNECTOR", "candidates": n}}), cfg)))
    out.append(_row("planner timeout offered as a certificate", "PLANNER_BUDGET", False,
                    cert.check(robot, s4, L4, one, {**fm(one, {}), "kind": "PLANNER_BUDGET"}, cfg)))
    out.append(_row("assignment changed after production", "FIRST_MOVER", False, cert.check(
        robot, s4, L4, {"o00": 30}, fm(one, {"o00": {"stage": "PICK", "candidates": n}}), cfg)))
    few = copy.deepcopy(cfg)
    few["skill2d"]["approach_dirs"] = 12
    few = validate_config(few)
    out.append(_row("grasp family narrowed (12 directions) under the same record", "FIRST_MOVER", False,
                    cert.check(robot, s4, L4, one, fm(one, {"o00": {"stage": "PICK", "candidates": n}}), few)))
    out += target_pick_cases(robot, cfg, s2, L2, n)
    for name, scene, learner, claim, expect in (
            ("R4 bound 2 (occupant o00 + pick certificate)", s4, L4, {"occupants": ["o00"], "first_mover_term": True,
                                                                     "lower_bound": 2}, True),
            ("R4 bound overstated as 3", s4, L4, {"occupants": ["o00"], "first_mover_term": True, "lower_bound": 3}, False),
            ("R2 bound 1 (occupant o03)", s2, L2, {"occupants": ["o03"], "first_mover_term": False, "lower_bound": 1}, True),
            ("R2 bound with an unjustified +1 term", s2, L2, {"occupants": ["o03"], "first_mover_term": True,
                                                              "lower_bound": 2}, False),
            ("R2 bound under a goal region with an escape pose", s2, esc, {"occupants": ["o03"], "first_mover_term": False,
                                                                           "lower_bound": 1}, False)):
        out.append(_row(name, "ACTION_LOWER_BOUND", expect, cert.check_lower_bound(robot, scene, learner, claim, cfg)))
    return out + g35_reaudit(robot, cfg)


def target_pick_cases(robot, cfg, s2, L2, n) -> list:
    """TARGET_PICK (added after the pilot): plan root g4a_r00, whose target has no definite
    grasp in S0, versus the R2 scene, whose target is graspable."""
    plan = json.loads(Path("runs/g4a_evidence_audit/g4a_plan.json").read_text())
    q = next(x for x in plan["queries"] if x["query_id"] == "g4a_r00")
    s0 = scene_from_dict(q["scene"])
    L0 = _query(cfg, s0, q["goal_cell"])["learner"]
    tp = lambda sc, L, cells, removed, k=n: {**cert.certificate_scope(sc, L, cells), "kind": "TARGET_PICK",  # noqa: E731
                                             "removed": removed, "candidates": k}
    obs = [o for o in s0.objects if o.kind == "obstacle"]
    free = [c for c in s0.legal_cells if c not in {o.cell for o in s0.objects} and c != q["goal_cell"]]
    every = {o.id: free[i] for i, o in enumerate(obs)}
    one = {obs[0].id: free[0]}
    return [_row("g4a_r00 all-stay: target grasp blocked by staying objects", "TARGET_PICK", True,
                 cert.check(robot, s0, L0, {}, tp(s0, L0, {}, []), cfg)),
            _row("removed set differs from the moved objects", "TARGET_PICK", False,
                 cert.check(robot, s0, L0, one, tp(s0, L0, one, []), cfg)),
            _row("every obstacle moved away: target graspable", "TARGET_PICK", False,
                 cert.check(robot, s0, L0, every, tp(s0, L0, every, sorted(every)), cfg)),
            _row("one declared target grasp omitted", "TARGET_PICK", False,
                 cert.check(robot, s0, L0, {}, tp(s0, L0, {}, [], n - 1), cfg)),
            _row("R2 target is graspable", "TARGET_PICK", False, cert.check(robot, s2, L2, {}, tp(s2, L2, {}, []), cfg))]


def _offgrid(cfg, scene, oid, xy):
    return replace(scene, objects=tuple(replace(o, xy_m=tuple(xy), cell=None) if o.id == oid else o for o in scene.objects))


def g35_reaudit(robot, cfg: dict) -> list:
    """Re-check every G3.5 certificate from the archived records (read-only), with the G3.5
    HOME-constrained action contexts, recomputing the scenes from fixtures.json."""
    import cases                                              # G3.5 scene construction (no cycle at runtime)
    fx = json.loads(Path("fixtures.json").read_text())["fixtures"]
    recs = json.loads(Path("runs/g3_5_initial_scene/cases.json").read_text())
    out = []
    for name, rec in recs.items():
        scene, _ = cases.build_scene(cfg, fx[name])
        learner = rec["learner"]
        for a in rec["offline_case_audit"]:
            c, cells = a["certificate"] or "", {k: v if isinstance(v, int) else list(v) for k, v in a["cells"].items()}
            scope = cert.certificate_scope(scene, learner, cells)
            if c.startswith("GOAL_OVERLAP"):
                record = {**scope, "kind": "GOAL_OVERLAP", "obstacles": c.split(":")[1].split(",")}
            elif c.startswith("FIRST_MOVER"):
                n = len(sk.directions(sk.params(cfg)))
                record = {**scope, "kind": "FIRST_MOVER", "movers": {m: {"stage": "PICK", "candidates": n} for m in cells}}
            else:
                continue
            out.append(_row(f"G3.5 {name}: {a['label']}", record["kind"], True, cert.check(robot, scene, learner, cells, record, cfg),
                            "archived certificate re-audited"))
        b = rec["bounds"]
        if b["lower_bound"] > 0:
            occ = pr.goal_blockers(scene, {}, learner["action"], cfg)
            out.append(_row(f"G3.5 {name}: LB {b['lower_bound']}", "ACTION_LOWER_BOUND", True, cert.check_lower_bound(
                robot, scene, learner, {"occupants": occ, "first_mover_term": b["lower_bound"] > len(occ),
                                        "lower_bound": b["lower_bound"]}, cfg), "archived bound re-audited"))
    return out


# ------------------------------------------------------------------ fault injection (§4.2)
def _shift(state, oid, dxy):
    objs = tuple(replace(o, xy_m=(o.xy_m[0] + dxy[0], o.xy_m[1] + dxy[1]), cell=None) if o.id == oid else o
                 for o in state.scene.objects)
    return replace(state, scene=replace(state.scene, objects=objs))


def _rename(state, old, new):
    return replace(state, scene=replace(state.scene, objects=tuple(replace(o, id=new) if o.id == old else o
                                                                   for o in state.scene.objects)))


def fault_injection(robot, cfg: dict) -> list:
    """R4 layout, committed two-move repair (o01 then o00). A deviation is injected in the
    observation after relocation 1; the episode must ABORT before relocation 2, with no
    target call, no new proposal or commitment. A within-tolerance deviation proceeds and
    the target is planned from the observed (perturbed) state."""
    tol = cfg["evidence"]["reobserve_pose_tolerance_m"]
    s4 = place_scene(cfg, "fault_r4", "initial", [20], [28, 27, 31, 58, 63])
    q = data.make_query(cfg, "fault_r4", s4, "t0", 28, {}, "SAFE_RETREAT")
    ranked = [{"label": "o00->29, o01->18", "cells": {"o00": 29, "o01": 18}}]
    variants = {"pose beyond tolerance (3x)": lambda st: {"state": _shift(st, "o01", (3 * tol, 0.0))},
                "pose within tolerance (0.5x)": lambda st: {"state": _shift(st, "o01", (0.5 * tol, 0.0))},
                "gripper gap inconsistent": lambda st: {"state": replace(st, gap=st.gap * 0.5)},
                "attachment inconsistent": lambda st: {"state": st, "attachment": "attached", "attached_id": "o01"},
                "object identity changed": lambda st: {"state": _rename(st, "o01", "o99")}}
    out = []
    for name, inject in variants.items():
        calls, trace = [], ev.Trace(f"fault/{name}")

        def target(state, key, calls=calls):
            calls.append(state)
            return ev.evaluate_target(robot, cfg, state, q["learner"]["action"], key)

        def executor(k, committed, inject=inject):
            obs = {"state": committed, "attachment": "free", "attached_id": None}
            return {**obs, **inject(committed)} if k == 0 else obs
        res = ev.primary_episode(robot, cfg, q, s4, ranked, target, robot_profile_hash(cfg), trace, executor)
        kinds = [e["kind"] for e in trace.events]
        w = res["commit"]["attempt"]
        obs0 = executor(0, w.states[0])["state"]
        out.append({"case": name, "outcome": res["outcome"], "abort_reasons": res.get("abort_reasons", []),
                    "events": kinds, "relocations_executed": kinds.count("RELOCATE"), "target_calls": len(calls),
                    "proposals": kinds.count("PROPOSE"), "commits": kinds.count("COMMIT"),
                    "commanded_o01_xy": list(w.states[0].scene.get("o01").xy_m),
                    "observed_o01_xy": list(obs0.scene.get("o01").xy_m) if "o01" in {o.id for o in obs0.scene.objects} else None,
                    "commanded_gap": w.states[0].gap, "observed_gap": obs0.gap,
                    "target_status": res["target"] and res["target"]["status"],
                    "target_start_carries_observed_o01": bool(calls and "o01" in {o.id for o in obs0.scene.objects}
                                                             and calls[0].scene.get("o01").xy_m == obs0.scene.get("o01").xy_m),
                    "proposal_hash": trace.events[0]["proposal_hash"], "observation_hash": q["hashes"]["observation"]})
    return out


# ------------------------------------------------------------------ invalid-endpoint control pool (§6)
def invalid_pool(robot, cfg: dict, plan: dict, n_roots: int = 5) -> list:
    """Fixed static controls per root, kept out of the main static-valid pool: duplicate
    destination, onto the resting target, off the support, clearance exactly delta."""
    out = []
    for q in plan["queries"][:n_roots]:
        s0 = scene_from_dict(q["scene"])
        L = _query(cfg, s0, q["goal_cell"])["learner"]
        obs = sorted((o for o in s0.objects if o.kind == "obstacle"), key=lambda o: o.xy_m)
        a, b, t = obs[0], obs[1], s0.get("t0")
        r = cfg["suite"]["obstacle_radius_m"]
        bx, by = b.xy_m
        controls = {"duplicate destination": {a.id: b.cell, b.id: b.cell},
                    "onto the resting target": {a.id: t.cell}, "off the support": {a.id: [-0.01, 0.30]},
                    "clearance exactly delta": {a.id: [bx + 2 * r + cfg["suite"]["margin_m"], by]}}
        for name, cells in controls.items():
            cells = {k: v if isinstance(v, int) else tuple(v) for k, v in cells.items()}
            st = pr.static_endpoints(s0, cells, "t0", cfg)
            rec = {**cert.certificate_scope(s0, L, cells), "kind": "STATIC_ENDPOINT_INVALID"}
            ok, why = cert.check(robot, s0, L, cells, rec, cfg)
            out.append({"query_id": q["query_id"], "control": name, "cells": {k: list(v) if isinstance(v, tuple) else v
                                                                              for k, v in cells.items()},
                        "static_status": st["status"], "failed": st["failed"] or st["errors"], "checker_accepted": ok,
                        "checker_reasons": why})
    return out
