"""P10b constructors (P10B_FAMILIES_V1): independent roots, frozen catalogues and the D parking map, all before any teacher call.

Reused unchanged from P10A_MECHANISMS_V1: H1, H2a (kappa-1), H2b (kappa0), H3, H4, H5 (same templates and sound-checker predicates).
New (static; sound checkers + pose-screen intent; never a planner):
  H7     edit vs executable: M occludes T's grasp; M's catalogue destination dM is enclosed on N / S / W, so the edit (teleport M to
         dM) frees T while the PHYSICAL placement at dM is expected to fail (no place-side certificate exists: any such failure is
         OPERATIONAL). M* = an open alternative destination.
  H8     constructor-disjoint obstruction: a link-0 sweep barrier only, accepted by the independent LINK0 sector checker on the
         ORIGINAL task (sound, no pose screens); the barrier-removal edit is the mechanism-specific pair.
  H10_3  grasp-side chain C -> B -> A -> T (A blocks T, B blocks A, C blocks B), trap destination A* inside the sweep sector.
  H10_4  H10_3 plus a sweep barrier E: four distinct relocations (C, B, A, E) in the main repair.
D parking map: for every movable pool object, in role order, the first cell of the frozen p6._dests rule (free, annulus-safe,
>= 2 cells from target and goal, nearest to the object), excluding earlier parkings. Frozen with the plan; never learned or executed.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import replace

import numpy as np

import data
import effects as ef
import g4c
import geometry as ge
import p10_scenes as ps
import p6_scenes as p6
import zlayout_r4 as z4
from p6_scenes import STRATA, _cell, _cheb, _xy
from scene import place_scene

VERSION = "P10B_FAMILIES_V1_1"   # V1.1 (after P10b-1, development evidence only): H3 placement intent
REUSE = {"H1": ("H1", ""), "H2a": ("H2", "kappa-1"), "H2b": ("H2", "kappa0"), "H3": ("H3", ""), "H4": ("H4", ""), "H5": ("H5", "")}
FAMILIES = list(REUSE) + ["H7", "H8", "H10_3", "H10_4"]
MOVERS = {"H7": "MD", "H8": "MD", "H10_3": "ABC", "H10_4": "ABCE"}
EDITS = {"H7": {"A": "M", "B": "D", "A+B": "MD"}, "H8": {"A": "M", "B": "D", "A+B": "MD"},
         "H10_3": {"A": "A", "B": "B", "A+B": "AB"}, "H10_4": {"A": "A", "B": "E", "A+B": "AE"}}
CATALOGUE = {   # outcome-blind priority; first = ANCHOR; "X*" = X to its second destination; omissions of the main repair included
    "H7": ["edit:A", "M", "M*", "", "edit:B", "edit:A+B", "D", "D M", "M D"],
    "H8": ["edit:A", "", "M", "D", "edit:B", "edit:A+B", "D M", "M D"],
    "H10_3": ["C B A", "B C A", "C A B", "A B C", "C B", "B A", "C A", "", "edit:A", "edit:B", "edit:A+B", "C B A*", "A"],
    "H10_4": ["C B A E", "E C B A", "C B E A", "B C A E", "C B A", "C B E", "B A E", "C A E", "", "edit:A", "edit:B", "edit:A+B", "E"]}
POOL_CAP = 4


def _layout(rng, fam, scr, stratum):
    """(role cells, fixed cells, goal, extra destinations) for a new family, or None."""
    a0, extra = STRATA[stratum][0], {}
    if fam in ("H10_3", "H10_4"):
        r = int(rng.integers(1, 7))
        role = {"T": _cell(r, 7), "A": _cell(r, 6), "B": _cell(r, 5), "C": _cell(r, 4)}
        fixed = [_cell(r + d, c) for c in (7, 6, 5) for d in (1, -1)]
        extra["A*"] = ps._barrier(rng, scr, role["T"], a0)          # trap: A parked inside the link-0 sweep sector
        if fam == "H10_4":
            role["E"] = ps._barrier(rng, scr, role["T"], a0)
            if role["E"] is not None and role["E"] == extra["A*"]:
                return None
    else:
        T = _cell(int(rng.integers(1, 7)), int(rng.integers(5, 8)) if fam == "H7" else int(rng.integers(4, 8)))
        role, fixed = {"T": T}, []
        if fam == "H7":
            r, c = divmod(T, 8)
            role["M"], fixed = _cell(r, c - 1), [_cell(r + 1, c), _cell(r - 1, c)]
            enc = [x for x in sorted(scr.valid) if _cheb(x, T) >= 3 and all(y in scr.valid for y in (x + 8, x - 8, x - 1)) and x % 8 > 0]
            if not enc:
                return None
            dm = int(rng.choice(enc))
            extra["M"], fixed = dm, fixed + [dm + 8, dm - 8, dm - 1]
        else:                                                         # H8: sweep barrier only (no grasp occluder)
            role["M"] = ps._barrier(rng, scr, T, a0)
    goal = _cell(int(rng.integers(0, 8)), int(rng.integers(3, 8)))
    cells = [v for v in role.values()] + fixed + list(extra.values())
    far = [x for x in sorted(scr.valid) if _cheb(x, role["T"]) >= 2 and _cheb(x, goal) >= 2 and x not in cells]
    if fam in ("H7", "H8"):
        role["D"] = int(rng.choice(far)) if far else None
    cells = [v for v in role.values()] + fixed + [v for v in extra.values()]
    if None in cells or len(set(cells)) != len(cells) or goal in cells or _cheb(goal, role["T"]) < 3:
        return None
    inside = {role.get("E"), role.get("M") if fam == "H8" else None, extra.get("A*")} - {None}
    if any(x not in scr.valid and x not in inside for x in cells + [goal]) or any(_cheb(goal, x) < 2 for x in cells):
        return None
    return role, fixed, goal, extra


def _predicates(cx, s, fam, ids, dest, goal):
    pred = []

    def chk(name, v, want=True):
        pred.append({"check": name, "value": v if isinstance(v, (int, bool)) else str(v), "want": want, "ok": bool(v) == want})
        return pred[-1]["ok"]
    mv = lambda *ks: {ids[k]: dest[k][0] for k in ks}  # noqa: E731
    sq = lambda *ks: [(ids[k], dest[k][0]) for k in ks]  # noqa: E731
    if fam == "H7":
        ok = chk("z0: certificate", cx.certified(s, goal)) and chk("M edit: open (intent)", cx.open_(s, goal, mv("M"))) and \
            chk("M pickable at S0", cx.pickable(s, ids["M"])) and chk("M place at enclosed dM: no screen place", cx.scr.place(s, ids["M"], dest["M"][0]), False) and \
            chk("M place at M*: screen place", cx.scr.place(s, ids["M"], dest["M"][1]))
    elif fam == "H8":
        e = cx.certified(s, goal, edit=mv("M"))
        ok = chk("z0: LINK0_SECTOR_PICK (sector checker; no pose screen)", "LINK0_SECTOR_PICK" in cx.certified(s, goal)) and \
            chk("barrier edit eligible and not certified", e != ["INELIGIBLE"] and not e)
    else:
        ok = chk("z0: TARGET_PICK", "TARGET_PICK" in cx.certified(s, goal)) and \
            chk("B first: ORDER_FIRST_ACTION", "ORDER_FIRST_ACTION" in cx.certified(s, goal, sq("B", "C", "A"))) and \
            chk("A first: ORDER_FIRST_ACTION", "ORDER_FIRST_ACTION" in cx.certified(s, goal, sq("A", "B", "C"))) and \
            chk("C B (A stays): TARGET_PICK", "TARGET_PICK" in cx.certified(s, goal, sq("C", "B"))) and \
            chk("C pickable at S0", cx.pickable(s, ids["C"])) and chk("B pickable after C", cx.pickable(s, ids["B"], mv("C"))) and \
            chk("A pickable after C, B", cx.pickable(s, ids["A"], mv("C", "B")))
        if fam == "H10_3":
            ok = ok and chk("target open after C, B, A", cx.open_(s, goal, mv("C", "B", "A")))
        else:
            ok = ok and chk("edit A (E stays): LINK0_SECTOR_PICK", "LINK0_SECTOR_PICK" in cx.certified(s, goal, edit=mv("A"))) and \
                chk("edit E (A stays): TARGET_PICK", "TARGET_PICK" in cx.certified(s, goal, edit=mv("E"))) and \
                chk("E pickable after C, B, A", cx.pickable(s, ids["E"], mv("C", "B", "A"))) and \
                chk("target open after C, B, A, E", cx.open_(s, goal, mv("C", "B", "A", "E")))
    return ok and all(p["ok"] for p in pred), pred


def parking(cx, s, goal, pool_ids) -> dict:
    """Frozen D parking map {object id: cell} (rule in the module docstring); outcome-blind and deterministic."""
    T, out = s.get("t0").cell, {}
    for oid in pool_ids:
        d = p6._dests(cx.scr, s, oid, [T, goal], list(out.values()), 1)
        if not d:
            return {}
        out[oid] = d[0]
    return out


def try_root(cx, spec, attempt, known):
    """One seeded attempt for any P10b family -> (record, None) or (None, rejection reason)."""
    fam, key = spec["family"], f"p10b_{spec['seed']}"
    if fam in REUSE:
        m, v = REUSE[fam]
        rec, why = ps._try(cx, {**spec, "mechanism": m, "variant": v, "prefix": "p10b"}, attempt, known)
        if rec is None:
            return None, why
        rec["construction"] = rec["construction"] + " [reused P10A_MECHANISMS_V1 template]"
        if fam == "H3" and spec.get("h3_version", "v1.1") == "v1.1":   # V1.1: the intended order must also PLACE (pose-screen intent)
            s, ids, dest = rec["scene"], rec["ids"], rec["dest"]
            pb_ = cx.scr.place(s, ids["B"], dest["B"][0])
            pa_ = cx.scr.place(p6.moved(s, {ids["B"]: dest["B"][0]}), ids["A"], dest["A"][0]) if pb_ else 0
            rec["predicates"] += [{"check": "B place at dB (intent, V1.1)", "value": pb_, "want": True, "ok": pb_ > 0},
                                  {"check": "A place at dA after B (intent, V1.1)", "value": pa_, "want": True, "ok": pa_ > 0}]
            if not (pb_ and pa_):
                return None, "predicate: H3 V1.1 placement intent"
            rec["construction"] += " + H3 V1.1 placement intent"
    else:
        rng = np.random.default_rng([spec["seed"], attempt])
        stratum = str(rng.choice(["S0", "S1", "S3"])) if fam in ("H8", "H10_3", "H10_4") else f"S{int(rng.integers(0, 4))}"
        lay = _layout(rng, fam, cx.scr, stratum)
        if lay is None:
            return None, "layout"
        role, fixed, goal, extra = lay
        movers = list(MOVERS[fam])
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
            dd = p6._dests(cx.scr, s, ids[k], [role["T"], goal], [x for v in dest.values() for x in v] + list(extra.values()), 2 if k == "M" else 1)
            if len(dd) < (2 if k == "M" else 1):
                return None, f"no destination for {k}"
            dest[k] = dd
        if fam == "H7":
            dest["M"] = [extra["M"], dest["M"][0]]
        if "A*" in extra:
            dest["A"] = [dest["A"][0], extra["A*"]]
        ok, pred = _predicates(cx, s, fam, ids, dest, goal)
        if not ok:
            return None, f"predicate: {pred[-1]['check']}"
        rec = {**spec, "attempt": attempt, "root": key, "scene": s, "goal_cell": goal, "stratum": stratum, "ids": ids, "dest": dest,
               "role_cells": role, "fixed": fixed, "predicates": pred, "mechanism": fam, "variant": "",
               "construction": "SECTOR_CERTIFICATE_ONLY (no pose screens)" if fam == "H8" else "CERTIFICATE_TARGETED (adaptive; sound checkers, no planner)"}
    pool = [rec["ids"][k] for k in (ps.MOVERS[REUSE[fam][0]] if fam in REUSE else MOVERS[fam])]
    rec["parking"] = parking(cx, rec["scene"], rec["goal_cell"], pool)
    if len(pool) > POOL_CAP or len(rec["parking"]) != len(pool):
        return None, "D parking map unavailable"
    rec["family"], rec["fingerprint"] = fam, [rec["scene"].get("t0").cell, rec["goal_cell"], sorted(o.cell for o in rec["scene"].objects)]
    return rec, None


def _query_id(k, kind, x):
    return f"scene:{k}:{z4.sig(sorted((o, int(c)) for o, c in x.items()))}" if kind == "SCENE_TARGET_ONLY" else f"seq:{k}:{z4.sig(x)}"


def catalogue(cfg, robot, r) -> list:
    """Frozen query list: the family catalogue (A/B/C edits A, B, A+B; sequences with z layouts) then every nonempty D parking
    subset in size order (deduplicated by physical identity). Each item: id, kind, priority, anchor, roles, layout / edit."""
    s, k, fam, ids, dest = r["scene"], r["root"], r["family"], r["ids"], r["dest"]
    act = data.make_query(cfg, k, s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
    if fam in REUSE:
        ea, eb = ps.EDIT_ROLES[REUSE[fam][0]]
        emap, toks = {"A": ea, "B": eb, "A+B": ea + eb}, ps.CATALOGUE[REUSE[fam][0]]
    else:
        emap, toks = EDITS[fam], CATALOGUE[fam]
    sub = lambda ed: ef.pose_substitution(robot, cfg, s, act, ed)  # noqa: E731
    edits = {}
    for kind, roles in emap.items():
        ed = {ids[x]: dest[x][0] for x in roles}
        sb = sub(ed)
        edits[kind] = {"kind": kind, "edit": ed, "eligible": bool(sb["eligible"]), "intervened_scene_hash": sb["intervened_scene_hash"]}
    cat = [edits[x] for x in ("A", "B", "A+B")]
    items, seen = [], {}
    for prio, tok in enumerate(toks):
        if tok.startswith("edit:"):
            e = edits[tok[5:]]
            it = {"kind": "SCENE_TARGET_ONLY", "edit_kind": e["kind"], "edit": e["edit"], "eligible": e["eligible"],
                  "intervened_scene_hash": e["intervened_scene_hash"]}
            qid = _query_id(k, it["kind"], e["edit"])
        else:
            seq = [(ids[t.rstrip("*")], dest[t.rstrip("*")][1 if t.endswith("*") else 0]) for t in tok.split()]
            it = {"kind": "ORDERED_TEACHER", "sequence": [list(x) for x in seq], "z_layout": z4.build(k, seq, cat)}
            qid = _query_id(k, it["kind"], seq)
        items.append({**it, "query": qid, "token": tok, "priority": prio, "anchor": prio == 0, "d_subset": None})
        seen[qid] = items[-1]
    inv = {v: kk for kk, v in ids.items()}
    pool = list(r["parking"])
    empty = _query_id(k, "ORDERED_TEACHER", [])
    if empty in seen:
        seen[empty]["d_subset"] = []
    for n in range(1, len(pool) + 1):
        for sub_ in itertools.combinations(pool, n):
            ed = {o: r["parking"][o] for o in sub_}
            qid = _query_id(k, "SCENE_TARGET_ONLY", ed)
            roles = sorted(inv[o] for o in sub_)
            if qid in seen:
                seen[qid]["d_subset"] = roles
                continue
            sb = sub(ed)
            items.append({"kind": "SCENE_TARGET_ONLY", "edit_kind": None, "edit": ed, "eligible": bool(sb["eligible"]),
                          "intervened_scene_hash": sb["intervened_scene_hash"], "query": qid, "token": "D:" + "+".join(roles),
                          "priority": len(items), "anchor": False, "d_subset": roles})
            seen[qid] = items[-1]
    if empty not in seen:                       # D needs y0: the empty repair is always queried
        items.append({"kind": "ORDERED_TEACHER", "sequence": [], "z_layout": z4.build(k, [], cat), "query": empty, "token": "",
                      "priority": len(items), "anchor": False, "d_subset": []})
    return items
