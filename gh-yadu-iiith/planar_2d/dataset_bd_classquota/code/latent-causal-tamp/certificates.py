"""Independent checker for G3.5/G4A/G4D impossibility certificates (G4A instruction §3).

A certificate is a record produced by the teacher path (evaluate.audit_candidate). The
checker never trusts its boolean: it recomputes the scope (observation and action-
context hashes, target and object identities, the evaluated assignment), the declared
rules (unique goal or listed alternatives, tolerances, no pushing, move-once, target
not a repair object) and the geometry or skill screen itself. It shares only trusted
primitives: disc distances, and the pose-only grasp screens of skill.py over the full
declared (direction, branch) family. Returns (accepted, reasons).
"""
from __future__ import annotations

import math
from dataclasses import replace

import data
import motion as mo
import proposals as pr
import skill as sk
from config import stable_hash

SCOPE_RULES = {"contact_mode": "PLANAR_SUPPORT_CONTACT_SIDE_GRASP", "pushing_allowed": False, "no_lift": True,
               "goal_orientation": "ANY (rotationally symmetric disc)"}


def certificate_scope(scene, learner: dict, cells: dict) -> dict:
    """Identity fields every certificate must carry (the producer stamps them)."""
    return {"observation_hash": stable_hash(data.learner_scene(scene)), "action_context_hash": stable_hash(learner["action"]),
            "assignment": [[o, d if isinstance(d, int) else list(d)] for o, d in sorted(cells.items())]}


def _scope(scene, learner: dict, cells: dict, cert: dict, cfg: dict) -> list:
    act, why = learner["action"], []
    expect = certificate_scope(scene, learner, cells)
    for k in ("observation_hash", "action_context_hash", "assignment"):
        if cert.get(k) != expect[k]:
            why.append(f"stale or mismatched {k}")
    ids = {o.id: o for o in scene.objects}
    if act["target_id"] not in ids or ids[act["target_id"]].kind != "target":
        why.append("action target is not a target object of this scene")
    if act["target_id"] in cells:
        why.append("target is a repair object (not allowed)")
    if [round(v, 12) for v in act["goal_xy_m"]] != [round(v, 12) for v in scene.lattice.center(act["goal_cell"])]:
        why.append("goal pose does not match the declared goal cell")
    why += [f"rule {k} != {v!r}" for k, v in SCOPE_RULES.items() if act.get(k) != v]
    if not cfg["protocol"]["move_once"] or cfg["protocol"]["target_calls_between_relocations"] != 0:
        why.append("protocol allows repeated moves or target action before repair completion")
    return why


def check_goal_overlap(scene, learner: dict, cells: dict, cert: dict, cfg: dict):
    """Every allowed final target placement (each declared goal pose, anywhere within the
    goal tolerance) overlaps some obstacle's final pose (anywhere within its tolerance:
    release tolerance if moved, declared state uncertainty if it stays) by more than the
    clearance delta minus the numerical band."""
    why = _scope(scene, learner, cells, cert, cfg)
    act = learner["action"]
    delta, eps = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"]
    rel, u = cfg["gripper2d"]["release_xy_tolerance_m"], cfg["evidence"]["state_pose_uncertainty_m"]
    ids = {o.id: o for o in scene.objects}
    claimed = cert.get("obstacles", [])
    if not claimed or any(o not in ids or ids[o].kind != "obstacle" for o in claimed):
        why.append("claimed blocker is missing or not an obstacle")
        return False, why
    t = ids[act["target_id"]]
    goals = [tuple(act["goal_xy_m"])] + [tuple(p) for p in act.get("goal_alternatives_xy", [])]
    for g in goals:
        def clearance(oid):
            o = ids[oid]
            d = cells.get(oid)
            xy = o.xy_m if d is None else (scene.lattice.center(d) if isinstance(d, int) else tuple(d))
            return math.dist(g, xy) - t.radius_m - o.radius_m + act["goal_tolerance_m"] + (rel if d is not None else u)
        if not any(clearance(o) < delta - eps for o in claimed):
            why.append(f"goal pose {g} admits a placement clear of every claimed blocker")
    return not why, why


def check_first_mover(robot, scene, learner: dict, cells: dict, cert: dict, cfg: dict):
    """Every selected object, moved first from S0, is excluded by a definite pose-only
    screen covering the whole declared grasp family: PICK (grasp pose, approach line,
    closure) or PLACE at its assigned destination. Connector or budget failures, a
    partial candidate family or an uncovered mover reject the proof."""
    why = _scope(scene, learner, cells, cert, cfg)
    movers = cert.get("movers", {})
    if set(movers) != set(cells) or not cells:
        why.append(f"movers {sorted(movers)} do not cover every selected object {sorted(cells)}")
    p = sk.params(cfg)
    family = sk.directions(p)
    for m, claim in movers.items():
        if m not in cells:
            continue
        if claim.get("stage") not in ("PICK", "PLACE"):
            why.append(f"{m}: {claim.get('stage')} is not a pose-only screen")
            continue
        if claim.get("candidates") != len(family):
            why.append(f"{m}: record covers {claim.get('candidates')} of {len(family)} declared grasps")
        o = scene.get(m)
        cands = sk.screen_pick(robot, mo.checkers(robot, scene, m, None, None, p["delta"], p["eps"]), o.xy_m,
                               o.radius_m, p)
        if claim["stage"] == "PLACE":
            if not any(c.stage == "OK" for c in cands):
                why.append(f"{m}: PLACE claimed but the pick screen already fails (wrong premise)")
            dest = scene.lattice.center(cells[m]) if isinstance(cells[m], int) else tuple(cells[m])
            local = (robot.grasp_x, 0.0)
            ck = mo.checkers(robot, scene, m, dest, local, p["delta"], p["eps"])
            cands = sk.screen_place(robot, ck, dest, local, 2 * o.radius_m, p)
        if sorted((c.phi, c.branch) for c in cands) != sorted(family):
            why.append(f"{m}: screen did not enumerate the declared family")
        if any(c.stage == "OK" for c in cands):
            why.append(f"{m}: some declared grasp passes the {claim['stage']} screen in S0 (can go first)")
        elif not all(c.certified for c in cands):
            why.append(f"{m}: some rejection is not definite (gray band / IK continuation)")
    return not why, why


def check_target_pick(robot, scene, learner: dict, cells: dict, cert: dict, cfg: dict):
    """The active target cannot be grasped even with every moved object deleted: the pick
    screen (grasp pose, approach, closure; pose-only, start-independent) rejects all
    declared grasps definitely among the objects that stay. Adding obstacles only adds
    collisions, so wherever the moved objects end, every realization fails to pick."""
    why = _scope(scene, learner, cells, cert, cfg)
    if sorted(cert.get("removed", [])) != sorted(cells):
        why.append("removed objects must be exactly the assignment's moved objects")
    p, family = sk.params(cfg), sk.directions(sk.params(cfg))
    if cert.get("candidates") != len(family):
        why.append(f"record covers {cert.get('candidates')} of {len(family)} declared grasps")
    kept = replace(scene, objects=tuple(o for o in scene.objects if o.id not in cells))
    t = kept.get(learner["action"]["target_id"])
    cands = sk.screen_pick(robot, mo.checkers(robot, kept, t.id, None, None, p["delta"], p["eps"]), t.xy_m, t.radius_m, p)
    if sorted((c.phi, c.branch) for c in cands) != sorted(family):
        why.append("screen did not enumerate the declared family")
    if any(c.stage == "OK" for c in cands):
        why.append("some declared grasp of the target passes among the staying objects")
    elif not all(c.certified for c in cands):
        why.append("some target-grasp rejection is not definite")
    return not why, why


def check_static(scene, learner: dict, cells: dict, cert: dict, cfg: dict):
    """Recompute the static endpoint factors (support, target at rest, pairs)."""
    why = [w for w in _scope(scene, learner, cells, cert, cfg) if "target is a repair" not in w]
    if pr.static_endpoints(scene, cells, learner["action"]["target_id"], cfg)["status"] != mo.INVALID:
        why.append("endpoints are not statically invalid")
    return not why, why


def check_order_first_action(robot, scene, learner: dict, cells: dict, cert: dict, cfg: dict):
    """V2 order-specific proof: the FIRST action of this canonical sequence is excluded in
    S0 by a definite pose-only PICK or PLACE screen over the whole declared grasp family.
    Nothing has moved before the first action, so it covers every realization of this
    order; it says nothing about other orders of the same endpoints."""
    why = _scope(scene, learner, cells, cert, cfg)
    seq = [tuple(a) for a in cert.get("sequence", [])]
    if not seq or len({o for o, _ in seq}) != len(seq) or {o: d for o, d in seq} != cells:
        return False, why + ["sequence missing, not canonical, or inconsistent with the assignment"]
    o, d = seq[0]
    probe = {o: d}
    ok, w = check_first_mover(robot, scene, learner, probe, {**certificate_scope(scene, learner, probe), "movers": {
        o: {"stage": cert.get("stage"), "candidates": cert.get("candidates")}}}, cfg)
    return ok and not why, why + [f"first action: {x}" for x in w]


# Which ordered proposals a certificate kind covers (Methodology v1.6 §4, Guide v1.8 §3).
# Endpoint-only or start-independent premises hold for every order of the assignment.
CERTIFICATE_SCOPE = {"GOAL_OVERLAP": "ASSIGNMENT",            # final endpoints only
                     "STATIC_ENDPOINT_INVALID": "ASSIGNMENT", # final endpoints only
                     "TARGET_PICK": "ASSIGNMENT",             # staying objects only; monotone in added obstacles
                     "FIRST_MOVER": "ASSIGNMENT",             # every selected object is excluded as first mover
                     "ORDER_FIRST_ACTION": "ORDER"}           # this sequence's first action only


def check(robot, scene, learner: dict, cells: dict, cert: dict, cfg: dict):
    kind = cert.get("kind")
    if kind == "ORDER_FIRST_ACTION":
        return check_order_first_action(robot, scene, learner, cells, cert, cfg)
    if kind == "GOAL_OVERLAP":
        return check_goal_overlap(scene, learner, cells, cert, cfg)
    if kind == "FIRST_MOVER":
        return check_first_mover(robot, scene, learner, cells, cert, cfg)
    if kind == "TARGET_PICK":
        return check_target_pick(robot, scene, learner, cells, cert, cfg)
    if kind == "STATIC_ENDPOINT_INVALID":
        return check_static(scene, learner, cells, cert, cfg)
    return False, [f"{kind!r} is not a sound certificate kind (planner exhaustion and failed orders never are)"]


def check_lower_bound(robot, scene, learner: dict, bound: dict, cfg: dict):
    """Re-derive the scoped action-level bound: each claimed goal occupant must move
    (goal overlap with everything else staying), and the +1 term needs each occupant's
    pick to fail definitely in S0 over the whole declared family."""
    occ = bound.get("occupants", [])
    lb, why = 0, []
    for o in occ:
        ok, w = check_goal_overlap(scene, learner, {}, {**certificate_scope(scene, learner, {}), "obstacles": [o]}, cfg)
        lb += ok
        why += w
    if occ and bound.get("first_mover_term"):
        n = len(sk.directions(sk.params(cfg)))
        for o in occ:                                          # pick screen: independent of any destination
            probe = {o: scene.get(o).cell}
            ok, w = check_first_mover(robot, scene, learner, probe, {**certificate_scope(scene, learner, probe),
                                      "movers": {o: {"stage": "PICK", "candidates": n}}}, cfg)
            why += [] if ok else [f"+1 term: {x}" for x in w]
        lb += not any(x.startswith("+1 term") for x in why)
    if lb != bound.get("lower_bound"):
        why.append(f"recomputed bound {lb} != claimed {bound.get('lower_bound')}")
    return not why, why
