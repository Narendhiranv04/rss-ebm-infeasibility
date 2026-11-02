"""P13 mechanism grammars (P13_MECH_V1): physics-first 2D layouts with SOUND-certificate intent predicates (mechanism-conditioned
case-control, disclosed) plus one unconditioned natural-mix stream. No learned output and no teacher label is consulted.

Primitives (robot base left of the support; link 0 cannot wrap, so the free link-0 sector between the start angle a0 and the target is
the only route): SWEEP barrier = inner-annulus cell strictly between a0 and the target bearing; IMPOSTOR = inner-annulus cell just
OUTSIDE that arc; GRASP occluder = a W / N / S neighbour of the target (the arm approaches from the base side), other sides FIXED.
  PC_SERIAL   A, B both sweep barriers               intent kappa +1 (each single edit sector-certified, A+B open)
  PC_SG       A sweep barrier, B sole-side occluder  intent kappa +1 (A edit TARGET_PICK, B edit sector)
  PR_SIDES    A, B occlude two open grasp sides      intent kappa -1 (each single edit open)
  PN_SIMP     A sweep barrier, B sweep impostor      intent kappa 0  (B edit sector-certified, A edit open)
  PN_GIMP     A sole-side occluder, B far-side decoy intent kappa 0  (B edit TARGET_PICK, A edit open)
  ORD_CHAIN   T <- A <- B (<- C) chain toward base    order: outer first; reversed first action ORDER_FIRST_ACTION
  OMIT_OPT    2-chain + far object D; repair "B A D"  omissions: B / A necessary (certified), D optional (teacher)
  TRI_SERIAL  three sweep barriers                   lattice Delta_ABC = +1 intent
  TRI_CTRL    two barriers + impostor                Delta_ABC = 0 intent (pairwise suffices)
  TRI_OR      barrier + two side occluders           Delta_ABC = -1 intent (A and (B or C))
  NAT         three movers near the target / sweep window, no prescreen (realistic mix; feasible originals allowed)
Edits use the D parking destinations (dest[x][0] := parking[x]), so every scene edit IS a D lattice query (exact attribution)."""
from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np

import effects as ef
import g4c
import geometry as ge
import p10b_scenes as pb
import zlayout_r4 as z4
from p6_scenes import STRATA, _cell, _cheb, _xy
from scene import place_scene

VERSION = "P13_MECH_V1_1"
PAIR = ("PC_SERIAL", "PC_SG", "PR_SIDES", "PN_SIMP", "PN_GIMP")
TRI = ("TRI_SERIAL", "TRI_CTRL", "TRI_OR")
FAMS = PAIR + ("ORD_CHAIN", "OMIT_OPT") + TRI + ("NAT",)
SUBSETS = ["A", "B", "C", "A+B", "A+C", "B+C", "A+B+C"]
# Outcome-blind tokens; a single relocation whose failure would rest only on a REMAINING sweep barrier has no static certificate
# (it would be UNKNOWN / BUDGET at 330 CPU-s), so it is left as an unlabelled omission slot instead of a catalogue query.
E3 = ["edit:A", "edit:B", "edit:A+B"]
# V1.1 (DEV, after pilot 2): PC_SERIAL is prediction-only (barrier relocations were UNKNOWN / BUDGET at 275-287 CPU-s on 3 / 3 dev roots)
TOK = {"PC_SERIAL": [""] + E3, "PN_SIMP": ["A B", "B A", ""] + E3, "PC_SG": ["A B", "B A", "A", ""] + E3,
       "PR_SIDES": ["A", "B", "A B", "B A", ""] + E3, "PN_GIMP": ["A", "B", "A B", "B A", ""] + E3,
       "ORD_CHAIN": ["B A", "A B", "A", "B", ""] + E3, "ORD_CHAIN3": ["C B A", "A B C", "B C A", "C B", "B A", "A", ""] + E3,
       "OMIT_OPT": ["B A D", "B A", "A D", "B D", "A B D", ""] + E3, "NAT": ["A", "B", "A B", "B A", ""] + E3}
TOK.update({f: [""] + [f"edit:{s}" for s in SUBSETS] for f in TRI})
POOL = {**{f: "AB" for f in PAIR + ("ORD_CHAIN", "OMIT_OPT", "NAT")}, **{f: "ABC" for f in TRI}}
NB = {"W": (0, -1), "N": (1, 0), "S": (-1, 0), "E": (0, 1), "NE": (1, 1), "SE": (-1, 1), "NW": (1, -1), "SW": (-1, -1)}


def _nb(T, k):
    r, c = divmod(T, 8)
    return _cell(r + NB[k][0], c + NB[k][1])


def _arc(scr, T, a0, inside=True, pad=0.15, out=0.6, near=None):
    """Inner-annulus cells strictly between a0 and the target bearing (inside; the `near` closest to a0 when given, since a barrier
    must separate a0 from every target pick angle, not only from the bearing) or within `out` rad outside that arc (impostors)."""
    base = scr.robot.base
    ang = lambda x: math.atan2(_xy(x)[1] - base[1], _xy(x)[0] - base[0])  # noqa: E731
    lo, hi = sorted((ang(T), a0))
    inner = [x for x in range(64) if x not in scr.valid and math.dist(_xy(x), base) < scr.r0 - 0.02 and abs(ang(x) - a0) > 0.12]
    if inside:
        arc = [x for x in inner if lo + pad < ang(x) < hi - pad]
        return sorted(arc, key=lambda x: abs(ang(x) - a0))[:near] if near else arc
    return [x for x in inner if (hi + pad < ang(x) < hi + out) or (lo - out < ang(x) < lo - pad)]


def layout(rng, fam, scr, a0):
    """(roles {name: cell}, fixed cells, goal) or None. Pure geometry; certificates / screens are applied afterwards."""
    valid = sorted(scr.valid)
    T = int(rng.choice([c for c in valid if 1 <= c // 8 <= 6 and c % 8 >= 4]))
    goal = int(rng.choice([c for c in valid if _cheb(c, T) >= 3]))
    sides = [str(x) for x in rng.permutation(["W", "N", "S"])]
    role, fixed = {"T": T}, []
    pick = lambda xs, n=1: [int(x) for x in rng.choice(xs, n, replace=False)] if len(xs) >= n else None  # noqa: E731
    if fam in ("PC_SERIAL", "TRI_SERIAL", "TRI_CTRL", "PN_SIMP"):
        n_in = {"PC_SERIAL": 2, "TRI_SERIAL": 3, "TRI_CTRL": 2, "PN_SIMP": 1}[fam]
        ins = pick(_arc(scr, T, a0, near=n_in + 2), n_in)
        if ins is None:
            return None
        role.update(zip("ABC", ins))
        if fam in ("TRI_CTRL", "PN_SIMP"):
            imp = pick(_arc(scr, T, a0, inside=False))
            if imp is None:
                return None
            role["C" if fam == "TRI_CTRL" else "B"] = imp[0]
    elif fam in ("PC_SG", "TRI_OR"):
        bar = pick(_arc(scr, T, a0, near=3))
        if bar is None:
            return None
        role["A"] = bar[0]
        if fam == "PC_SG":
            role["B"], fixed = _nb(T, sides[0]), [_nb(T, sides[1]), _nb(T, sides[2])]
        else:
            role["B"], role["C"], fixed = _nb(T, sides[0]), _nb(T, sides[1]), [_nb(T, sides[2])]
    elif fam == "PR_SIDES":
        role["A"], role["B"], fixed = _nb(T, sides[0]), _nb(T, sides[1]), [_nb(T, sides[2])]
    elif fam == "PN_GIMP":
        role["A"], fixed = _nb(T, sides[0]), [_nb(T, sides[1]), _nb(T, sides[2])]
        role["B"] = _nb(T, str(rng.choice(["E", "NE", "SE"])))
    elif fam in ("ORD_CHAIN", "OMIT_OPT"):
        r, c = divmod(T, 8)
        n = 3 if fam == "ORD_CHAIN" and rng.random() < 0.5 else 2
        role.update({k: _cell(r, c - 1 - i) for i, k in enumerate("ABC"[:n])})
        fixed = [_cell(r + d, c - i) for i in range(n) for d in (1, -1)]
        if fam == "OMIT_OPT":
            far = [x for x in valid if _cheb(x, T) >= 3 and _cheb(x, goal) >= 2]
            role["D"] = pick(far)[0] if far else None
    else:                                                               # NAT: no mechanism intent
        near = [x for x in valid if 1 <= _cheb(x, T) <= 2] + _arc(scr, T, a0, pad=0.0)
        three = pick(sorted(set(near)), 3)
        if three is None:
            return None
        role.update(zip("ABC", three))
    fixed = [x for x in fixed if x is not None]
    cells = list(role.values()) + fixed
    inner = set(_arc(scr, T, a0)) | set(_arc(scr, T, a0, inside=False)) | set(_arc(scr, T, a0, pad=0.0))
    if None in cells or len(set(cells)) != len(cells) or goal in cells or any(_cheb(goal, x) < 2 for x in cells) or \
            any(x not in scr.valid and x not in inner for x in cells):
        return None
    return role, fixed, goal


def _predicates(cx, s, fam, ids, dest, goal, role):
    """Sound-certificate / pose-screen INTENT (never a label). Returns (ok, predicate list)."""
    pred = []

    def chk(name, v, want=True):
        pred.append({"check": name, "value": v if isinstance(v, (int, bool)) else str(v), "want": want, "ok": bool(v) == want})
        return pred[-1]["ok"]
    ed = lambda ks: {ids[k]: dest[k][0] for k in ks}  # noqa: E731
    cert = lambda ks: cx.certified(s, goal, edit=ed(ks)) if ks else cx.certified(s, goal)  # noqa: E731
    sq = lambda toks: [(ids[k], dest[k][0]) for k in toks]  # noqa: E731
    tp = lambda x: "TARGET_PICK" in x  # noqa: E731
    if fam == "NAT":
        return True, pred
    ok = chk("z0: sound certificate", bool(cert("")))
    if fam in ("PC_SERIAL", "TRI_SERIAL", "TRI_CTRL", "PC_SG", "TRI_OR", "PN_SIMP", "PR_SIDES", "PN_GIMP"):
        pool = POOL[fam]
        want = {"PC_SERIAL": lambda u: u >= {"A", "B"}, "PC_SG": lambda u: u >= {"A", "B"}, "PR_SIDES": lambda u: bool(u & {"A", "B"}),
                "PN_SIMP": lambda u: "A" in u, "PN_GIMP": lambda u: "A" in u, "TRI_SERIAL": lambda u: u >= {"A", "B", "C"},
                "TRI_CTRL": lambda u: u >= {"A", "B"}, "TRI_OR": lambda u: "A" in u and bool(u & {"B", "C"})}[fam]
        for n in range(1, len(pool) + 1):
            for u in itertools.combinations(pool, n):
                if not ok:
                    break
                c = cert(u)
                ok = chk(f"edit {'+'.join(u)}: {'open (intent)' if want(set(u)) else 'certified'}",
                         (cx.open_(s, goal, ed(u)) and not c) if want(set(u)) else bool(c) and c != ["INELIGIBLE"])
        if ok and fam in PAIR:
            ok = chk("a physical start exists", cx.pickable(s, ids["A"]) or cx.pickable(s, ids["B"]))
    else:
        chain = [k for k in "CBA" if k in role]                            # outer -> inner
        ok = ok and chk("z0: TARGET_PICK", tp(cert(""))) and chk("inner first: ORDER_FIRST_ACTION", "ORDER_FIRST_ACTION" in
                                                                    cx.certified(s, goal, sq(list(reversed(chain)))))
        moved = {}
        for k in chain:
            ok = ok and chk(f"{k} pickable after {sorted(moved) or 'none'}", cx.pickable(s, ids[k], moved))
            moved[ids[k]] = dest[k][0]
        ok = ok and chk("target open after the chain", cx.open_(s, goal, moved))
        if fam == "OMIT_OPT":
            ok = ok and chk("D pickable after the chain", cx.pickable(s, ids["D"], moved))
    return ok and all(p["ok"] for p in pred), pred


def try_root(cx, spec, attempt, known):
    fam, key = spec["family"], f"p13_{spec['seed']}"
    if fam not in FAMS:
        return _orig_try(cx, spec, attempt, known)
    rng = np.random.default_rng([spec["seed"], attempt])
    stratum = str(rng.choice(["S0", "S1", "S2", "S3"]))
    lay = layout(rng, fam, cx.scr, STRATA[stratum][0])
    if lay is None:
        return None, "layout"
    role, fixed, goal = lay
    movers = [k for k in "ABCD" if k in role]
    s = place_scene(cx.cfg, key, "obstructed", [role["T"]], [role[k] for k in movers] + fixed)
    s = replace(s, robot=replace(s.robot, q_rad=tuple(STRATA[stratum])))
    d, e = cx.cfg["suite"]["margin_m"], cx.cfg["geometry"]["numeric_margin_band_m"]
    if ge.Checker(cx.robot, ge.make_world(s), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status != ge.CLEAR:
        return None, "start not clear"
    if g4c.stable_geometry(s, goal) in known:
        return None, "duplicate geometry"
    ids = {k: f"o{i:02d}" for i, k in enumerate(movers)}
    pool = [ids[k] for k in POOL[fam]]
    park = pb.parking(cx, s, goal, pool + [ids[k] for k in movers if k not in POOL[fam]])
    if len(park) != len(movers):
        return None, "no parking / destination"
    dest = {k: [park[ids[k]]] for k in movers}
    if fam == "ORD_CHAIN" and "C" in role:
        fam_tok = "ORD_CHAIN3"
    else:
        fam_tok = fam
    ok, pred = _predicates(cx, s, fam, ids, dest, goal, role)
    if not ok:
        return None, f"predicate: {pred[-1]['check']}"
    rec = {**spec, "attempt": attempt, "root": key, "scene": s, "goal_cell": goal, "stratum": stratum, "ids": ids, "dest": dest,
           "role_cells": role, "fixed": fixed, "predicates": pred, "mechanism": fam, "variant": fam_tok,
           "parking": {o: park[o] for o in pool},
           "construction": "NATURAL (no prescreen, no intent)" if fam == "NAT" else
           "P13 GRAMMAR + SOUND-CERTIFICATE / POSE-SCREEN INTENT (mechanism-conditioned case-control; no learned / teacher input)"}
    rec["family"], rec["fingerprint"] = fam, [role["T"], goal, sorted(o.cell for o in s.objects)]
    return rec, None


def catalogue(cfg, robot, r) -> list:
    """Frozen items: family tokens (edits = parking subsets of the pool) then any missing pool subset, plus the empty repair."""
    if r["family"] not in FAMS:
        return _orig_cat(cfg, robot, r)
    import data
    s, k, ids, dest = r["scene"], r["root"], r["ids"], r["dest"]
    act = data.make_query(cfg, k, s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
    pool = POOL[r["family"]]
    kinds = ["A", "B", "A+B"] if len(pool) == 2 else SUBSETS
    edits = {}
    for kind in kinds:
        ed = {ids[x]: dest[x][0] for x in kind.split("+")}
        sb = ef.pose_substitution(robot, cfg, s, act, ed)
        edits[kind] = {"kind": kind, "edit": ed, "eligible": bool(sb["eligible"]), "intervened_scene_hash": sb["intervened_scene_hash"],
                       "d_subset": sorted(kind.split("+"))}
    cat = [edits[x] for x in kinds]
    items = []
    for prio, tok in enumerate(TOK[r["variant"]]):
        if tok.startswith("edit:"):
            e = edits[tok[5:]]
            it = {"kind": "SCENE_TARGET_ONLY", "edit_kind": e["kind"], "edit": e["edit"], "eligible": e["eligible"],
                  "intervened_scene_hash": e["intervened_scene_hash"], "d_subset": e["d_subset"]}
            qid = pb._query_id(k, it["kind"], e["edit"])
        else:
            seq = [(ids[t], dest[t][0]) for t in tok.split()]
            it = {"kind": "ORDERED_TEACHER", "sequence": [list(x) for x in seq], "z_layout": z4.build(k, seq, cat), "d_subset": [] if not seq else None}
            qid = pb._query_id(k, it["kind"], seq)
        items.append({**it, "query": qid, "token": tok, "priority": prio, "anchor": prio == 0})
    if not any(it["token"] == "" for it in items):
        items.append({"kind": "ORDERED_TEACHER", "sequence": [], "z_layout": z4.build(k, [], cat), "d_subset": [],
                      "query": pb._query_id(k, "ORDERED_TEACHER", []), "token": "", "priority": len(items), "anchor": False})
    return items


_orig_try, _orig_cat = pb.try_root, pb.catalogue
pb.try_root, pb.catalogue = try_root, catalogue
for _f in FAMS:
    pb.MOVERS[_f] = "ABCD"
