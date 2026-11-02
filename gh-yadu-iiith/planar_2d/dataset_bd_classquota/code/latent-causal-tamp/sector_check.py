"""Independent checker for LINK0_SECTOR_PICK_V1 proof records (G4C-D2-P3a).

Kept apart from the producer (sector_cert.py) on purpose: it re-derives the annulus
radius, barriers, free sector and the closed-form IK with its own code, rebuilds the
world from the source records itself, re-runs the trusted pose-only pick screen
(skill.screen_pick, the primitive certificates.py also trusts) and never reads the
producer's verdict except to refuse a record whose claim disagrees with the proof.

Theorem (report 0017 §6, pick side only). Start q_s (open, free gripper), resting world
W. Every declared-skill execution reaches the exact pick configuration of some declared
(phi, branch) through configurations that are not COLLISION in W; link 0 cannot point at
any disc with rho < R0 = L1 + max(w0/2, r_h) + r + delta - eps, and joint 1 cannot wrap,
so q[0] stays in the open sector I(W, q_s). If each of the 48 candidates has no IK, a
definite screen rejection, or q[0] certifiably outside I, the moved object cannot be
picked from q_s under the declared skill. Place-side exclusions are not used (A7).
"""
from __future__ import annotations

import math

import numpy as np

import config as cf
import geometry as ge
import motion as mo
import skill as sk
from scene import Obj, scene_hash

VERSION = "LINK0_SECTOR_PICK_V1"
TAU = 1e-9                   # rad / m: fail-closed margin at every sector and annulus boundary
LABEL_SCOPES = {"ORIGINAL_TARGET", "SCENE_TARGET_ONLY", "FIRST_ACTION_ORDER"}
SCOPES = LABEL_SCOPES | {"CONTROL_FIXED_START"}


def constants(robot, cfg) -> dict:
    return {"L1": robot.lengths[0], "w0_half": robot.widths[0] / 2, "housing_r": robot.housing_r,
            "delta": cfg["suite"]["margin_m"], "eps": cfg["geometry"]["numeric_margin_band_m"],
            "joint0_limits": list(robot.limits[0]), "grasp_x": robot.grasp_x, "dirs": cfg["skill2d"]["approach_dirs"]}


def world_rows(scene) -> list:
    return [[o.id, o.kind, o.radius_m, float(o.xy_m[0]), float(o.xy_m[1]), o.cell] for o in scene.objects]


def rebuild(root_scene, objects=None, edit=None, q=None):
    """Scene from source records: root scene, optional saved object rows
    {id: [kind, r, [x, y], cell]}, optional cell edit {id: cell}, optional start q."""
    objs = []
    for o in root_scene.objects:
        if objects is not None:
            k, r, xy, c = objects[o.id]
            o = Obj(o.id, k, r, (float(xy[0]), float(xy[1])), c, o.support)
        if edit and o.id in edit:
            o = Obj(o.id, o.kind, o.radius_m, root_scene.lattice.center(edit[o.id]), edit[o.id], o.support)
        objs.append(o)
    if objects is not None and set(objects) != {o.id for o in root_scene.objects}:
        raise ValueError("saved objects differ from the root's object identities")
    robot = root_scene.robot if q is None else type(root_scene.robot)(tuple(float(v) for v in q),
                                                                       root_scene.robot.gripper_gap_m)
    return type(root_scene)(root_scene.root_id, root_scene.variant_id, root_scene.lattice, tuple(objs), robot,
                            root_scene.legal_cells, root_scene.domain_policy, root_scene.robot_profile_hash)


def _ik(robot, anchor, phi, branch):
    """Own closed form (mirrors geometry.ik's semantics, written separately)."""
    l1, l2, l3 = robot.lengths
    u = np.array([math.cos(phi), math.sin(phi)])
    w = np.asarray(anchor, float) - robot.grasp_x * u - l3 * u - np.asarray(robot.base, float)
    c2 = (w @ w - l1 * l1 - l2 * l2) / (2 * l1 * l2)
    if abs(c2) > 1.0 + 1e-12:
        return None
    sols = []
    for b in (1, -1):
        q2 = b * math.acos(min(1.0, max(-1.0, c2)))
        q1 = math.atan2(w[1], w[0]) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        q = (np.array([q1, q2, phi - q1 - q2]) + math.pi) % (2 * math.pi) - math.pi
        ok = all(lo <= v <= hi for v, (lo, hi) in zip(q, robot.limits))
        if ok and not any(np.allclose(q, s) for _, s in sols):
            sols.append((b, q))
    return dict(sols).get(branch)


def sector(robot, cfg, scene, q0: float):
    """(barriers [(id, rho, alpha)], (lo, hi)) or raises ValueError (fail closed)."""
    c = constants(robot, cfg)
    lo_lim, hi_lim = c["joint0_limits"]
    if not -math.pi < lo_lim <= hi_lim < math.pi:
        raise ValueError("joint 1 limits reach +-pi: wrap not excluded")
    bars = []
    for o in scene.objects:
        dx, dy = o.xy_m[0] - robot.base[0], o.xy_m[1] - robot.base[1]
        rho, a = math.hypot(dx, dy), math.atan2(dy, dx)
        if rho < c["L1"] + max(c["w0_half"], c["housing_r"]) + o.radius_m + c["delta"] - c["eps"] - TAU:
            bars.append((o.id, rho, a))
    if any(abs(a - q0) <= TAU for _, _, a in bars):
        raise ValueError("start link 0 on a barrier angle")
    lo = max([a for _, _, a in bars if a < q0], default=lo_lim)
    hi = min([a for _, _, a in bars if a > q0], default=hi_lim)
    return bars, (max(lo, lo_lim), min(hi, hi_lim))


def check(robot, cfg, rec: dict, src: dict) -> dict:
    """src: {"scene": rebuilt start scene, "scope", "moved", "sequence", "kind"}. Returns
    {"accepted", "verdict", "reasons", "excluded": [bool]*48, "unexcluded": [...]}."""
    why = []
    out = lambda: {"accepted": not why and verdict == "CERTIFIED_INFEASIBLE", "verdict": verdict,  # noqa: E731
                   "reasons": why, "excluded": ex, "unexcluded": [i for i, e in enumerate(ex) if not e]}
    verdict, ex = "REFUSED", [False] * 48
    s, q = src["scene"], [float(v) for v in src["scene"].robot.q_rad]
    if rec.get("version") != VERSION or src["scope"] not in SCOPES or rec["query"].get("scope") != src["scope"]:
        why.append("version or scope mismatch")
        return out()
    seq, kind = src["sequence"], src["kind"]
    if src["scope"] == "ORIGINAL_TARGET" and (kind != "ORDERED_TEACHER" or seq or src["moved"] != "t0"):
        why.append("ORIGINAL_TARGET needs the empty repair of the target")
    if src["scope"] == "SCENE_TARGET_ONLY" and (kind != "SCENE_TARGET_ONLY" or src["moved"] != "t0"):
        why.append("SCENE_TARGET_ONLY needs an eligible fixed-robot target-only edit")
    if src["scope"] == "FIRST_ACTION_ORDER" and (kind != "ORDERED_TEACHER" or not seq or src["moved"] != seq[0][0]):
        why.append("FIRST_ACTION_ORDER covers only the first action from S0")
    if src["scope"] in LABEL_SCOPES and scene_hash(s) != src.get("expected_hash"):
        why.append("rebuilt scene does not match the frozen plan's scene / intervention hash")
    if rec["query"].get("moved") != src["moved"] or rec["query"].get("sequence") != [list(a) for a in seq]:
        why.append("moved object or sequence differs from the source")
    st = rec["start"]
    if st.get("q") != q or st.get("gap") != robot.open_gap or s.robot.gripper_gap_m != robot.open_gap or \
            st.get("attachment") != "free" or s.robot.attachment != "free":
        why.append("start q / gap / attachment differs or is not open and free")
    if rec.get("world") != world_rows(s) or rec.get("scene_hash") != scene_hash(s):
        why.append("world or scene hash differs from the rebuilt source")
    if rec.get("profile") != {"robot_profile_hash": cf.robot_profile_hash(cfg), "config_hash": cf.config_hash(cfg)} \
            or rec.get("constants") != constants(robot, cfg):
        why.append("profile, config or geometric constants differ")
    try:
        bars, (lo, hi) = sector(robot, cfg, s, q[0])
    except ValueError as e:
        why.append(str(e))
        return out()
    if sorted(b[0] for b in rec.get("barriers", [])) != sorted(b[0] for b in bars) or \
            not np.allclose(rec.get("sector", [0, 0]), [lo, hi], rtol=0, atol=1e-12):
        why.append("barriers or sector differ from the recomputation")
    p = sk.params(cfg)
    o = s.get(src["moved"])
    if ge.Checker(robot, ge.make_world(s), p["delta"], p["eps"]).status_at(np.asarray(q), robot.open_gap).status != ge.CLEAR:
        why.append("start configuration is not certified clear")
    fam = sk.directions(p)
    cands = rec.get("candidates", [])
    if len(fam) != 48 or [[c.get("phi"), c.get("branch")] for c in cands] != [list(f) for f in fam]:
        why.append("candidate family is not exactly the 48 declared (phi, branch) in canonical order")
        return out()
    fresh = sk.screen_pick(robot, mo.checkers(robot, s, o.id, None, None, p["delta"], p["eps"]), o.xy_m, o.radius_m, p)
    for i, (c, f) in enumerate(zip(cands, fresh)):
        qi = _ik(robot, o.xy_m, c["phi"], c["branch"])
        if (qi is None) != (c["q"] is None) or (qi is not None and not np.allclose(qi, c["q"], rtol=0, atol=1e-9)):
            why.append(f"{i}: IK differs from the independent closed form")
        if c["screen"] != {"stage": f.stage, "reason": f.reason, "certified": bool(f.certified)}:
            why.append(f"{i}: recorded screen differs from the fresh screen")
        for r in c["excluded_by"]:
            if r == "NO_IK":
                ok = qi is None
            elif r == "SECTOR":
                ok = qi is not None and (qi[0] < lo - TAU or qi[0] > hi + TAU)
            elif r.startswith("DEFINITE_SCREEN:"):
                ok = f.stage != "OK" and f.stage == r.split(":", 1)[1] and f.certified
            else:
                ok = False
                why.append(f"{i}: {r!r} is not a sound pick-side exclusion")
            if not ok:
                why.append(f"{i}: claimed {r} does not hold")
            ex[i] = bool(ex[i] or ok)
    verdict = "CERTIFIED_INFEASIBLE" if all(ex) else "UNKNOWN"
    if rec.get("claim") != verdict:
        why.append(f"producer claim {rec.get('claim')} != checked {verdict}")
    return out()
