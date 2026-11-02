"""P10a constructors: 7 mechanisms x 2 waves = 14 independent reference roots, <= 16 queries each, frozen before any teacher call.

Static and planner-free. H1-H5, H7 use the pick-side grasp mechanism (target / mover graspability). Their INTENDED NEGATIVES are
required to be accepted by the existing SOUND certificate checkers (TARGET_PICK, ORDER_FIRST_ACTION via ordered.certificates_for)
at construction time: this is CERTIFICATE-TARGETED (adaptive) construction, disclosed and excluded from any blind claim.
Intended positives use only the pose screens as intent; the teacher decides them. H8 is constructor-disjoint from G+M: it is
selected by link-0 ANGLES only (a barrier in the link-0 annulus between the target's and the goal's angle, on the far side from
the start), never by pose screens. Intent is never a label and never a learner input.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import replace

import numpy as np

import data
import effects as ef
import g4c
import geometry as ge
import ordered as od
import p6_scenes as p6
import zlayout_r4 as z4
from p6_scenes import STRATA, _cell, _cheb, _xy
from scene import place_scene

VERSION = "P10A_MECHANISMS_V1"
MECH = ["H1", "H2", "H3", "H4", "H5", "H7", "H8"]
SEED0, MAX_ATTEMPTS, CPU_CAP, ROOT_CPU, QUERY_CAP = 2026102801, 150, 350.0, 60.0, 16
ROOTS = [{"index": i, "wave": w, "mechanism": h, "seed": SEED0 + i, "variant": ("kappa-1" if w == 1 else "kappa0") if h == "H2" else ""}
         for i, (w, h) in enumerate((w, h) for w in (1, 2) for h in MECH)]
NB = {"W": (0, -1), "N": (1, 0), "S": (-1, 0), "NW": (1, -1), "SW": (-1, -1), "WW": (0, -2), "NE": (1, 1), "SE": (-1, 1), "E": (0, 1)}


class Ctx:
    def __init__(self, cfg, robot):
        self.cfg, self.robot, self.scr = cfg, robot, p6.Screens(cfg, robot)

    def learner(self, s, goal):
        return data.make_query(self.cfg, "c", s, "t0", goal, {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]

    def certified(self, s, goal, seq=(), edit=None) -> list:
        """Accepted sound certificate kinds for a sequence query or a fixed-robot scene edit (empty list = none)."""
        import sector_cert
        import sector_check
        from scene import scene_hash
        ln, base, h = self.learner(s, goal), s, scene_hash(s)
        if edit:
            sub = ef.pose_substitution(self.robot, self.cfg, s, ln["action"], edit)
            if not sub["eligible"]:
                return ["INELIGIBLE"]
            s, seq, h = sub["scene"], (), sub["intervened_scene_hash"]
        acc = [c["kind"] for c in od.certificates_for(self.robot, self.cfg, s, ln, [list(x) for x in seq]) if c["checker_accepted"]]
        if not acc:                                          # same LINK0_SECTOR_PICK fallback as teacher_pilot.static_admission
            scope = "SCENE_TARGET_ONLY" if edit else "FIRST_ACTION_ORDER" if seq else "ORIGINAL_TARGET"
            moved, dest = (seq[0][0], seq[0][1]) if seq else ("t0", goal)
            rec = sector_cert.produce(self.robot, self.cfg, s, moved, scope, [list(x) for x in seq], dest)
            src = {"scene": sector_check.rebuild(base, edit=edit), "scope": scope, "moved": moved, "sequence": [list(x) for x in seq],
                   "kind": "SCENE_TARGET_ONLY" if edit else "ORDERED_TEACHER", "expected_hash": h}
            if sector_check.check(self.robot, self.cfg, rec, src)["accepted"]:
                acc.append("LINK0_SECTOR_PICK")
        return acc

    def open_(self, s, goal, moves: dict) -> bool:
        """Pose-screen intent: the target is graspable and placeable at the goal after the static substitution."""
        st = p6.moved(s, moves)
        return len(self.scr.pick(st, "t0")) > 0 and self.scr.place(st, "t0", goal) > 0

    def pickable(self, s, oid, moves=None) -> bool:
        return len(self.scr.pick(p6.moved(s, moves or {}), oid)) > 0


def _around(T, keys):
    r, c = divmod(T, 8)
    return {k: _cell(r + NB[k][0], c + NB[k][1]) for k in keys}


def _barrier(rng, scr, T, a0):
    """An annulus cell strictly between the start link-0 angle a0 and the target's angle (P7 MOT geometry)."""
    base = scr.robot.base
    ang = lambda x: math.atan2(_xy(x)[1] - base[1], _xy(x)[0] - base[0])  # noqa: E731
    lo, hi = sorted((ang(T), a0))
    inner = [x for x in range(64) if x not in scr.valid and math.dist(_xy(x), base) < scr.r0 - 0.02 and lo + 0.15 < ang(x) < hi - 0.15]
    return int(rng.choice(sorted(inner, key=lambda x: abs(ang(x) - a0))[:3])) if inner else None


def _layout(rng, mech, variant, scr, stratum):
    """(roles {name: cell}, fixed cells, goal) - template on the target's grasp-side enclosure (W, N, S; the arm approaches
    from the base side); geometry proposal only, the predicates and certificate checkers decide."""
    T = _cell(int(rng.integers(1, 7)), int(rng.integers(5, 8)))
    goal = _cell(int(rng.integers(0, 8)), int(rng.integers(3, 8)))
    nb = {k: v for k, v in _around(T, list(NB)).items() if v is not None}
    opt = lambda keys: [nb[k] for k in keys if k in nb and rng.random() < 0.5]  # noqa: E731
    role, fixed = {"T": T}, []
    if mech == "H1":                                         # A occludes the grasp (W), B blocks the link-0 sweep
        bar = _barrier(rng, scr, T, STRATA[stratum][0])
        if bar is None:
            return None
        role["A"], role["B"], fixed = nb.get("W"), bar, [nb.get("N"), nb.get("S")]
    elif mech == "H7":                                       # M = W occluder, itself enclosed (N, S, W of M fixed): the edit frees T,
        r, c = divmod(T, 8)                                  # no executable order can relocate M; D = far distractor
        far = [x for x in sorted(scr.valid) if _cheb(x, T) >= 2 and _cheb(x, goal) >= 2]
        role["M"], role["D"] = nb.get("W"), (int(rng.choice(far)) if far else None)
        fixed = [nb.get("N"), nb.get("S"), _cell(r, c - 2)] + opt(["NW", "SW"])
    elif mech == "H2":
        sides = [str(x) for x in rng.permutation(["W", "N", "S"])]
        role["A"] = nb.get(sides[0])
        if variant == "kappa-1":
            role["B"], fixed = nb.get(sides[1]), [nb.get(sides[2])]
        else:
            fixed = [nb.get(sides[1]), nb.get(sides[2])]
            far = [x for x in sorted(scr.valid) if 2 <= _cheb(x, T) <= 3]
            role["B"] = int(rng.choice(far)) if far else None
        fixed += opt(["NW", "SW"])
    elif mech == "H3":                                       # A = N (or S) of T, itself enclosed by T, its outer cell and the W diagonal
        side = str(rng.choice(["N", "S"]))
        r, c = divmod(T, 8)
        outer = _cell(r + (2 if side == "N" else -2), c)
        diag = nb.get("NW" if side == "N" else "SW")
        role["A"] = nb.get(side)
        bk = int(rng.integers(0, 2))
        role["B"] = [outer, diag][bk]
        fixed = [nb.get("W"), nb.get("S" if side == "N" else "N"), [outer, diag][1 - bk]]
    elif mech == "H4":
        role["A"] = nb.get("W")
        fixed = [nb.get("N"), nb.get("S")] + opt(["NW", "SW"])
        far = [x for x in sorted(scr.valid) if 2 <= _cheb(x, T) <= 3]
        role["C"] = int(rng.choice(far)) if far else None
    elif mech == "H5":
        sides = [str(x) for x in rng.permutation(["W", "N", "S"])]
        fixed = [nb.get(sides[0]), nb.get(sides[1])]
        far = [x for x in sorted(scr.valid) if _cheb(x, T) >= 3]
        if len(far) < 2 or nb.get(sides[2]) is None:
            return None
        role["X"], role["Y"] = [int(x) for x in rng.choice(far, 2, replace=False)]
        role["x"] = nb[sides[2]]                             # the harmful EDIT destination of X: closes the enclosure
    elif mech == "H8":                                       # angles only: barrier M inside the link-0 annulus
        base = scr.robot.base
        ang = lambda x: math.atan2(_xy(x)[1] - base[1], _xy(x)[0] - base[0])  # noqa: E731
        T = int(rng.choice(sorted(scr.valid)))
        goal = int(rng.choice(sorted(scr.valid)))
        lo, hi = sorted((ang(T), ang(goal)))
        if hi - lo < 0.5:
            return None
        inner = [x for x in range(64) if x not in scr.valid and math.dist(_xy(x), base) < scr.r0 - 0.02 and lo + 0.12 < ang(x) < hi - 0.12]
        far = [x for x in sorted(scr.valid) if _cheb(x, T) >= 2 and _cheb(x, goal) >= 2]
        if not inner or not far:
            return None
        role = {"T": T, "M": int(rng.choice(inner)), "D": int(rng.choice(far))}
    cells = list(role.values()) + fixed
    if mech == "H5":
        cells = [v for k, v in role.items() if k != "x"] + fixed
    if None in cells or len(set(cells)) != len(cells) or goal in cells or _cheb(goal, T) < 3:
        return None
    inside = {role.get("B")} if mech == "H1" else {role.get("M")} if mech == "H8" else set()
    if any(x not in scr.valid and x not in inside for x in cells + [goal]):
        return None
    return role, fixed, goal


MOVERS = {"H1": "AB", "H2": "AB", "H3": "AB", "H7": "MD", "H4": "AC", "H5": "XY", "H8": "MD"}
EDIT_ROLES = {"H1": ("A", "B"), "H2": ("A", "B"), "H3": ("A", "B"), "H7": ("M", "D"), "H4": ("A", "C"), "H5": ("X", "Y"), "H8": ("M", "D")}
# catalogue in outcome-blind priority order; first entry = ANCHOR (intended positive); tokens: roles, "edit:" = fixed-robot scene edit
CATALOGUE = {
    "H1": ["edit:A+B", "", "edit:A", "edit:B", "A B", "B A", "A", "B"],
    "H2": ["edit:A+B", "edit:A", "edit:B", "", "A B", "A", "B"],
    "H3": ["B A", "A B", "A", "B", "", "edit:A", "edit:B", "edit:A+B"],
    "H7": ["edit:A", "M", "", "edit:B", "edit:A+B", "D", "D M", "M D"],
    "H4": ["C A", "A", "C", "", "A C", "edit:A", "edit:B", "edit:A+B", "A*"],
    "H5": ["", "edit:A", "edit:B", "edit:A+B", "Y", "X", "Y X"],
    "H8": ["M", "", "edit:A", "edit:B", "edit:A+B", "D", "D M", "M D"]}


def _predicates(cx, s, mech, variant, ids, dest, goal, stratum):
    pred = []

    def chk(name, v, want=True):
        pred.append({"check": name, "value": v if isinstance(v, (int, bool)) else str(v), "want": want, "ok": bool(v) == want})
        return pred[-1]["ok"]
    mv = lambda *ks: {ids[k]: dest[k][0] for k in ks}  # noqa: E731
    tp = lambda xs: "TARGET_PICK" in xs  # noqa: E731
    if mech == "H1":
        ok = chk("z0: certificate", cx.certified(s, goal)) and chk("A+B edit: open (intent)", cx.open_(s, goal, mv("A", "B"))) and \
            chk("A edit (grasp freed, sweep barrier stays): sector certificate", "LINK0_SECTOR_PICK" in cx.certified(s, goal, edit=mv("A"))) and \
            chk("B edit (barrier gone, grasp occluded): TARGET_PICK certificate", tp(cx.certified(s, goal, edit=mv("B"))))
    elif mech == "H7":
        # executable-failure side NOT targeted: no tested enclosure gave a certified M relocation failure (probes 0/38); teacher decides
        ok = chk("z0: certificate", cx.certified(s, goal)) and chk("M edit: open (intent)", cx.open_(s, goal, mv("M")))
    elif mech == "H2":
        ok = chk("z0: TARGET_PICK certificate", tp(cx.certified(s, goal))) and chk("A+B edit: open (intent)", cx.open_(s, goal, mv("A", "B")))
        if variant == "kappa-1":
            ok = ok and chk("A edit: open", cx.open_(s, goal, mv("A"))) and chk("B edit: open", cx.open_(s, goal, mv("B")))
        else:
            ok = ok and chk("A edit: open", cx.open_(s, goal, mv("A"))) and chk("B edit: TARGET_PICK certificate", tp(cx.certified(s, goal, edit=mv("B"))))
        ok = ok and chk("a physical order is startable", cx.pickable(s, ids["A"]) or cx.pickable(s, ids["B"]))
    elif mech == "H3":
        ok = chk("z0: TARGET_PICK certificate", tp(cx.certified(s, goal))) and \
            chk("A first: ORDER_FIRST_ACTION certificate", "ORDER_FIRST_ACTION" in cx.certified(s, goal, [(ids["A"], dest["A"][0]), (ids["B"], dest["B"][0])])) and \
            chk("B pickable at S0", cx.pickable(s, ids["B"])) and chk("A pickable after B", cx.pickable(s, ids["A"], mv("B"))) and \
            chk("target open after B, A", cx.open_(s, goal, mv("A", "B")))
    elif mech == "H4":
        ok = chk("z0: TARGET_PICK certificate", tp(cx.certified(s, goal))) and chk("A edit: open", cx.open_(s, goal, mv("A"))) and \
            chk("C alone: TARGET_PICK certificate", tp(cx.certified(s, goal, [(ids["C"], dest["C"][0])]))) and \
            chk("C pickable", cx.pickable(s, ids["C"])) and chk("A pickable", cx.pickable(s, ids["A"])) and chk("open after C, A", cx.open_(s, goal, mv("A", "C")))
    elif mech == "H5":
        ok = chk("original open (feasible intent)", cx.open_(s, goal, {})) and chk("X edit -> adjacent cell: TARGET_PICK certificate",
                                                                                    tp(cx.certified(s, goal, edit={ids["X"]: dest["X"][0]}))) and \
            chk("Y edit: still open", cx.open_(s, goal, mv("Y"))) and chk("Y pickable", cx.pickable(s, ids["Y"]))
    else:                                                    # H8: angles only, no screens
        base, a0 = cx.robot.base, STRATA[stratum][0]
        ang = lambda c: math.atan2(_xy(c)[1] - base[1], _xy(c)[0] - base[0])  # noqa: E731
        T, M = s.get("t0").cell, s.get(ids["M"]).cell
        between = lambda x, a, b: min(a, b) < x < max(a, b)  # noqa: E731
        ok = chk("M between target and goal angle", between(ang(M), ang(T), ang(goal))) and \
            chk("M not between start link-0 angle and target", not between(ang(M), a0, ang(T)))
    return ok and all(p["ok"] for p in pred), pred


def _try(cx, spec, attempt, known):
    rng = np.random.default_rng([spec["seed"], attempt])
    stratum = str(rng.choice(["S0", "S1", "S3"])) if spec["mechanism"] == "H1" else f"S{int(rng.integers(0, 4))}"
    lay = _layout(rng, spec["mechanism"], spec["variant"], cx.scr, stratum)
    if lay is None:
        return None, "layout"
    role, fixed, goal = lay
    mech = spec["mechanism"]
    movers = list(MOVERS[mech])
    key = f"{spec.get('prefix', 'p10a')}_{spec['seed']}"
    s = place_scene(cx.cfg, key, "obstructed", [role["T"]], [role[k] for k in movers] + fixed)
    s = replace(s, robot=replace(s.robot, q_rad=tuple(STRATA[stratum])))
    ids = {k: f"o{i:02d}" for i, k in enumerate(movers)}
    d, e = cx.cfg["suite"]["margin_m"], cx.cfg["geometry"]["numeric_margin_band_m"]
    if ge.Checker(cx.robot, ge.make_world(s), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status != ge.CLEAR:
        return None, "start not clear"
    if g4c.stable_geometry(s, goal) in known:
        return None, "duplicate geometry"
    dest = {}
    for k in movers:
        dd = p6._dests(cx.scr, s, ids[k], [role["T"], goal], [x for v in dest.values() for x in v], 2 if (mech, k) == ("H4", "A") else 1)
        if len(dd) < (2 if (mech, k) == ("H4", "A") else 1):
            return None, f"no destination for {k}"
        dest[k] = dd
    if mech == "H5":
        dest["X"] = [role["x"]]
    ok, pred = _predicates(cx, s, mech, spec["variant"], ids, dest, goal, stratum)
    if not ok:
        return None, f"predicate: {pred[-1]['check']}"
    return {**spec, "attempt": attempt, "root": key, "scene": s, "goal_cell": goal, "stratum": stratum, "ids": ids, "dest": dest,
            "role_cells": role, "fixed": fixed, "predicates": pred,
            "construction": "ANGLE_ONLY (no pose screens)" if mech == "H8" else "CERTIFICATE_TARGETED (adaptive; sound checkers, no planner)"
            + ("; H7 relocation side untargeted" if mech == "H7" else "")}, None


def construct(cfg, robot, known: set) -> tuple:
    cx, roots, ledger, c0, unavailable = Ctx(cfg, robot), [], [], time.process_time(), []
    for spec in ROOTS:
        r0 = time.process_time()
        for attempt in range(MAX_ATTEMPTS):
            if time.process_time() - c0 > CPU_CAP or time.process_time() - r0 > ROOT_CPU:
                break                                        # recorded CONSTRUCTION_UNAVAILABLE below; never replaced
            rec, why = _try(cx, spec, attempt, known)
            ledger.append({"index": spec["index"], "seed": spec["seed"], "attempt": attempt, "accepted": rec is not None, "reason": why})
            if rec:
                known.add(g4c.stable_geometry(rec["scene"], rec["goal_cell"]))
                roots.append(rec)
                break
        else:
            unavailable.append({**spec, "status": "CONSTRUCTION_UNAVAILABLE", "attempts": sum(x["seed"] == spec["seed"] for x in ledger),
                                "root_cpu_s": round(time.process_time() - r0, 1)})
    return roots, ledger, {"cpu_s": round(time.process_time() - c0, 2), "screen_calls": cx.scr.calls, "unavailable": unavailable}


def catalogue(cfg, robot, r) -> dict:
    """Frozen catalogue: scene edits (kinds A / B / A+B over the mechanism's edit roles) and sequences, in priority order."""
    s, k, mech, ids, dest = r["scene"], r["root"], r["mechanism"], r["ids"], r["dest"]
    act = data.make_query(cfg, k, s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
    ea, eb = EDIT_ROLES[mech]
    edits = {"A": {ids[ea]: dest[ea][0]}, "B": {ids[eb]: dest[eb][0]}}
    edits["A+B"] = edits["A"] | edits["B"]
    scene_q = []
    for kind, ed in edits.items():
        sub = ef.pose_substitution(robot, cfg, s, act, ed)
        scene_q.append({"kind": kind, "edit": ed, "eligible": bool(sub["eligible"]), "intervened_scene_hash": sub["intervened_scene_hash"]})
    items = []
    for prio, tok in enumerate(CATALOGUE[mech][:QUERY_CAP]):
        if tok.startswith("edit:"):
            q = next(x for x in scene_q if x["kind"] == tok[5:])
            items.append({"priority": prio, "kind": "SCENE_TARGET_ONLY", "edit_kind": q["kind"], "edit": q["edit"], "eligible": q["eligible"],
                          "intervened_scene_hash": q["intervened_scene_hash"], "anchor": prio == 0})
        else:
            seq = [(ids[t.rstrip("*")], dest[t.rstrip("*")][1 if t.endswith("*") else 0]) for t in tok.split()]
            items.append({"priority": prio, "kind": "ORDERED_TEACHER", "sequence": [list(x) for x in seq], "anchor": prio == 0,
                          "z_layout": z4.build(k, seq, scene_q)})
    return {"scene_queries": scene_q, "items": items}
