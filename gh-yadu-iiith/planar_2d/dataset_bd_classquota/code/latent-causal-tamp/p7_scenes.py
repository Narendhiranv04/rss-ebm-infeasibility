"""P7b: frozen construction of <= 48 independent roots (6 families x 8, two waves of 24) and <= 32-query catalogues.

Static and outcome-blind (no teacher / planner / skill search). Intent predicates use the pose-only screens
(p6_scenes.Screens) and the LINK0_SECTOR_PICK producer claim at the fixed start; they are construction
INTENT only - every label is decided later by the unchanged OPNEG-v1 teacher and certificate checkers.
  KP  complementary: A closes the target's grasp side, B the goal's place side         (kappa intent +1)
  KM  redundant: A and B are two of the target's three grasp-side neighbours           (kappa intent -1)
  K0  additive / irrelevant: A is a grasp-side neighbour, B a nearby irrelevant object  (kappa intent 0)
  ORD access chains of 2-3 moves (+ optional place-side blocker D; up to 4 moves)
  MOT a link-0 SWEEP barrier M inside the annulus between the start and the target / place blocker Y
      (grasp screens pass; the obstruction is motion-level, decided by the sector certificate or the teacher)
  ALT three interchangeable grasp-side neighbours: alternative 1-4 move repairs, 2 destinations each
A family that cannot be constructed within MAX_ATTEMPTS is recorded CONSTRUCTION_UNAVAILABLE (never replaced).
Independence: unique translation-invariant motif signature (family, role / fixed offsets, goal offset, stratum,
clutter count) and unique stable geometry against every historical root.
"""
from __future__ import annotations

import itertools
import math
import time
from dataclasses import replace

import numpy as np

import config as cf
import data
import effects as ef
import g4c
import geometry as ge
import p6_scenes as p6
import sector_cert
import sequences as sq
import zlayout_r4 as z4
from p6_scenes import STRATA, _cell, _cheb, _xy, moved
from scene import place_scene

VERSION = "P7_FAMILIES_V1"
FAMS = ["KP", "KM", "K0", "ORD", "MOT", "ALT"]
SEED0, MAX_ATTEMPTS, CONSTRUCTION_CPU_S, QUERY_CAP = 2026102401, 600, 900.0, 32
SIDES = {"N": 8, "S": -8, "W": -1}
PREFIX = "p7_"                        # root-key prefix (P8 reuses this constructor with "p8_")
MOT_STRATA = ["S0", "S1", "S3"]       # static preview: S2 (link 0 at 45 deg) admits no annulus barrier (0 / 200 attempts)
ROOTS = [{"index": i, "wave": w, "batch": 2 * (w - 1) + 1 + j // 2, "family": f, "split": sp, "seed": SEED0 + i,
          "stratum": MOT_STRATA[(j + 2 * w) % 3] if f == "MOT" else f"S{(fi + j + 2 * w) % 4}"}
         for i, (w, fi, f, j, sp) in enumerate((w, fi, f, j, sp) for w in (1, 2) for fi, f in enumerate(FAMS)
                                                 for j, sp in enumerate(("train", "validation", "train", "test")))]


def _nb(c, side):
    r, k = divmod(c, 8)
    rr, kk = r + {"N": 1, "S": -1, "W": 0}[side], k - (side == "W")
    return _cell(rr, kk)


def _blocks(rng, fam, scr, stratum):
    """(roles {name: cell}, fixed cells, goal, main order, edits {A, B}) or None."""
    r, c = int(rng.integers(1, 7)), int(rng.integers(5, 8))
    rg, cg = int(rng.integers(1, 7)), int(rng.integers(5, 8))
    T, goal = _cell(r, c), _cell(rg, cg)
    sides = list(rng.permutation(["N", "S", "W"]))
    encl = {s: _nb(T, s) for s in sides}
    role, fixed, place_side = {"T": T}, [], []
    if fam in ("KP", "KM", "K0", "ALT"):
        k = {"KP": 1, "K0": 1, "KM": 2, "ALT": 3}[fam]
        for name, s in zip("ABC", sides[:k]):
            role[name] = encl[s]
        fixed += [encl[s] for s in sides[k:]]
        main, edits = {"KP": ["A", "B"], "KM": ["A"], "K0": ["A"], "ALT": ["A"]}[fam], {"A": ["A"], "B": ["B"]}
    if fam == "KP":
        gs = list(rng.permutation(["N", "S", "W"]))
        role["B"] = _nb(goal, gs[0])
        place_side = [_nb(goal, s) for s in gs[1:]]
    if fam == "K0":
        near = [x for x in range(64) if 1 <= _cheb(x, T) <= 2 and x not in encl.values() and x != goal]
        role["B"] = int(rng.choice(near)) if near else None
    if fam == "ORD":
        L, withd = int(rng.integers(2, 4)), bool(rng.integers(0, 2))
        c = 7 if L == 3 else c
        T = _cell(r, c)
        role = {"T": T, **{k_: _cell(r, c - 1 - i) for i, k_ in enumerate("ABC"[:L])}}
        fixed = [_cell(r + d, c - i) for i in range(L) for d in (1, -1)]
        main = list("CBA"[3 - L:]) + (["D"] if withd else [])
        edits = {"A": list("ABC"[:L]), "B": ["D"]} if withd else {"A": ["A"], "B": ["B"]}
        if withd:
            role["D"] = _nb(goal, "W")
            place_side = [_nb(goal, "N"), _nb(goal, "S")]
    if fam == "MOT":                                      # barrier strictly between the start's link-0 angle and T / Y
        ang = lambda x: math.atan2(_xy(x)[1] - scr.robot.base[1], _xy(x)[0] - scr.robot.base[0])  # noqa: E731
        q0, yc = STRATA[stratum][0], _nb(goal, "W")
        if yc is None:
            return None
        lo, hi = sorted((max(ang(T), ang(yc)), q0)) if q0 > max(ang(T), ang(yc)) else sorted((min(ang(T), ang(yc)), q0))
        inner = [x for x in range(64) if x not in scr.valid and math.dist(_xy(x), scr.robot.base) < scr.r0 - 0.02
                 and lo + 0.15 < ang(x) < hi - 0.15]
        if not inner:
            return None
        inner = sorted(inner, key=lambda x: abs(ang(x) - q0))[:3]       # nearest the start angle: widest blocked range
        role["M"] = int(rng.choice(inner))
        role["Y"] = yc
        place_side = [_nb(goal, "N"), _nb(goal, "S")]
        main, edits = ["M", "Y"], {"A": ["M"], "B": ["Y"]}
    fixed += place_side
    cells = [v for k_, v in role.items()] + fixed
    if None in cells or len(set(cells)) != len(cells) or goal in cells or math.dist(_xy(goal), _xy(T)) < 3 * 0.08:
        return None
    place_roles = {"KP": ["B"], "ORD": ["D"], "MOT": ["Y"]}.get(fam, [])
    pick = [role["T"]] + [v for k_, v in role.items() if k_ in "ABC" and k_ not in place_roles] + [x for x in fixed if x not in place_side]
    gside = [goal] + place_side + [role[k_] for k_ in place_roles if k_ in role]
    if any(_cheb(a, b) < 2 for a in pick for b in gside):
        return None
    if any(x not in scr.valid for x in cells + [goal] if x != role.get("M")):
        return None
    return role, fixed, goal, main, edits


def _sector_blocked(robot, cfg, s, oid):
    return sector_cert.produce(robot, cfg, s, oid, "ORIGINAL_TARGET", [], 0)["claim"] == "CERTIFIED_INFEASIBLE"


def _predicates(scr, robot, cfg, s, fam, ids, dest, goal, main):
    pred, mv = [], {}

    def chk(name, v, want):
        pred.append({"check": name, "value": int(v), "want": want, "ok": (v > 0) == want})
        return pred[-1]["ok"]
    if fam == "MOT":
        ok = chk("target grasp screen at S0 (open)", len(scr.pick(s, "t0")), True) and \
            chk("target link-0 sector-blocked at S0", _sector_blocked(robot, cfg, s, "t0"), True) and \
            chk("Y link-0 sector-blocked at S0", _sector_blocked(robot, cfg, s, ids["Y"]), True)
    else:
        ok = chk("target grasp at S0", len(scr.pick(s, "t0")), False)
    for k in main:
        if not ok:
            return False, pred
        st = moved(s, mv)
        ok = chk(f"{k} grasp after {list(mv)}", len(scr.pick(st, ids[k])), True)
        if ok and fam == "MOT":
            ok = chk(f"{k} not sector-blocked after {list(mv)}", not _sector_blocked(robot, cfg, st, ids[k]), True)
        mv[ids[k]] = dest[k][0]
    end = moved(s, mv)
    ok = ok and chk("target grasp after main", len(scr.pick(end, "t0")), True) and chk("target place after main", scr.place(end, "t0", goal), True)
    one = lambda k: moved(s, {ids[k]: dest[k][0]})  # noqa: E731
    if ok and fam == "KP":
        ok = chk("A only: grasp opens", len(scr.pick(one("A"), "t0")), True) and chk("A only: place closed", scr.place(one("A"), "t0", goal), False) \
            and chk("B only: grasp closed", len(scr.pick(one("B"), "t0")), False)
    if ok and fam in ("KM", "ALT"):
        for k in [x for x in "ABC" if x in ids]:
            ok = ok and chk(f"{k} only: grasp + place open", min(len(scr.pick(one(k), "t0")), scr.place(one(k), "t0", goal)), True)
    if ok and fam == "K0":
        ok = chk("B only: grasp still closed", len(scr.pick(one("B"), "t0")), False) and \
            chk("A only: grasp + place open", min(len(scr.pick(one("A"), "t0")), scr.place(one("A"), "t0", goal)), True)
    if ok and fam == "ORD":
        ok = chk(f"{main[1]} grasp at S0 (order intent)", len(scr.pick(s, ids[main[1]])), False)
    if ok and fam == "MOT":
        ok = chk("M only: target not sector-blocked", not _sector_blocked(robot, cfg, one("M"), "t0"), True) and \
            chk("M only: place closed by Y", scr.place(one("M"), "t0", goal), False)
    return ok and all(p["ok"] for p in pred), pred


def _signature(fam, role, fixed, goal, stratum, nclutter):
    T = role["T"]
    off = lambda x: (x // 8 - T // 8, x % 8 - T % 8)  # noqa: E731
    return (fam, tuple(sorted((k, off(v)) for k, v in role.items())), tuple(sorted(off(x) for x in fixed)), off(goal), stratum, nclutter)


def _try(cfg, robot, scr, spec, attempt, known):
    rng = np.random.default_rng([spec["seed"], attempt])
    blk = _blocks(rng, spec["family"], scr, spec["stratum"])
    if blk is None:
        return None, "family block geometry not met"
    role, fixed, goal, main, edits = blk
    movers = [k for k in role if k != "T"]
    used = set(role.values()) | set(fixed) | {goal}
    far = [x for x in sorted(scr.valid) if all(_cheb(x, u) >= 2 for u in used)]
    ncl = min(int(rng.integers(0, 3)), max(0, len(far) - 1))
    clutter = [int(x) for x in rng.choice(far, ncl, replace=False)] if ncl else []
    key = f"{PREFIX}{spec['seed']}"
    s = place_scene(cfg, key, "obstructed", [role["T"]], [role[k] for k in movers] + fixed + clutter)
    s = replace(s, robot=replace(s.robot, q_rad=tuple(STRATA[spec["stratum"]])))
    ids = {k: f"o{i:02d}" for i, k in enumerate(movers)}
    d, e = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"]
    if ge.Checker(robot, ge.make_world(s), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status != ge.CLEAR:
        return None, f"start stratum {spec['stratum']} not clear"
    sig = _signature(spec["family"], role, fixed, goal, spec["stratum"], ncl)
    if g4c.stable_geometry(s, goal) in known or sig in known:
        return None, "duplicate geometry or motif signature"
    dest = {}
    for k in movers:
        dd = p6._dests(scr, s, ids[k], [role["T"], goal], [x for v in dest.values() for x in v], 2)
        if not dd:
            return None, f"destinations for {k} unavailable"
        dest[k] = dd if len(dd) == 2 else dd * 2
    used |= set(clutter) | {x for v in dest.values() for x in v}
    nu = [x for x in sorted(scr.valid) if all(_cheb(x, u) >= 2 for u in used)]
    if nu:
        ncell = nu[int(rng.integers(len(nu)))]
        nd = [x for x in sorted(scr.valid) if x not in used | {ncell} and _cheb(x, role["T"]) >= 2 and _cheb(x, goal) >= 2]
        if nd:
            s = place_scene(cfg, key, "obstructed", [role["T"]], [role[k] for k in movers] + fixed + clutter + [ncell])
            s = replace(s, robot=replace(s.robot, q_rad=tuple(STRATA[spec["stratum"]])))
            ids["N"] = f"o{len(movers) + len(fixed) + len(clutter):02d}"
            dest["N"] = sorted(nd, key=lambda x: (math.dist(_xy(x), _xy(ncell)), _xy(x)))[:1]
    ok, pred = _predicates(scr, robot, cfg, s, spec["family"], ids, dest, goal, main)
    if not ok:
        return None, f"static intent predicate failed: {pred[-1]['check']}"
    ref = place_scene(cfg, f"{key}_ref", "reference", [role["T"]], fixed + clutter + ([s.get(ids['N']).cell] if "N" in ids else []))
    ref = replace(ref, robot=s.robot)
    return {**spec, "attempt": attempt, "root": key, "scene": s, "ref_scene": ref, "target_cell": role["T"], "goal_cell": goal, "ids": ids,
            "dest": dest, "role_cells": role, "fixed": fixed, "clutter": clutter, "main": main, "edit_roles": edits, "predicates": pred,
            "signature": repr(sig), "_sig": sig}, None


def construct(cfg, robot, known: set) -> tuple:
    scr, roots, ledger, c0, unavailable = p6.Screens(cfg, robot), [], [], time.process_time(), []
    for spec in ROOTS:
        for attempt in range(MAX_ATTEMPTS):
            if time.process_time() - c0 > CONSTRUCTION_CPU_S:
                raise SystemExit(f"construction exceeded {CONSTRUCTION_CPU_S} CPU-s (STOP before any teacher call)")
            rec, why = _try(cfg, robot, scr, spec, attempt, known)
            ledger.append({"index": spec["index"], "seed": spec["seed"], "attempt": attempt, "accepted": rec is not None, "reason": why})
            if rec:
                known |= {g4c.stable_geometry(rec["scene"], rec["goal_cell"]), rec.pop("_sig")}
                roots.append(rec)
                break
        else:
            unavailable.append({**spec, "status": "CONSTRUCTION_UNAVAILABLE", "attempts": MAX_ATTEMPTS})
    return roots, ledger, {"cpu_s": round(time.process_time() - c0, 2), "screen_calls": scr.calls, "unavailable": unavailable}


def _seq(r, toks):
    return [(r["ids"][t.rstrip("*")], r["dest"][t.rstrip("*")][1 if t.endswith("*") else 0]) for t in toks]


def catalogue(cfg, robot, r) -> dict:
    """Frozen <= 32-query catalogue (reference root + empty + edits + sequences), priority order, deduplicated."""
    s, k, main = r["scene"], r["root"], r["main"]
    core = [x for x in r["ids"] if x != "N"]
    act = data.make_query(cfg, k, s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
    eds = [("A", r["edit_roles"]["A"]), ("B", r["edit_roles"]["B"]), ("A+B", r["edit_roles"]["A"] + r["edit_roles"]["B"])]
    eds = [(kind, {r["ids"][t]: r["dest"][t][0] for t in toks}) for kind, toks in eds] + \
        ([("nuisance", {r["ids"]["N"]: r["dest"]["N"][0]})] if "N" in r["ids"] else [])
    scene_q = []
    for kind, ed in eds:
        sub = ef.pose_substitution(robot, cfg, s, act, ed)
        scene_q.append({"kind": kind, "edit": ed, "eligible": bool(sub["eligible"]), "intervened_scene_hash": sub["intervened_scene_hash"]})
    long_ = (main + [x for x in core if x not in main])[:4]
    if len(long_) < 4 and "N" in r["ids"]:
        long_ = long_ + ["N"]
    rows = [("original", []), ("main", main), ("long", long_)]
    rows += [("omission", [a for i, a in enumerate(long_) if i != t]) for t in range(len(long_))] if len(long_) >= 2 else []
    rows += [("single", [x]) for x in core] + [("alternative", [main[0] + "*"] + main[1:])]
    rows += [("pair", list(p)) for p in itertools.permutations(core, 2)]
    rows += [("permutation", list(p)) for p in itertools.islice(itertools.permutations(main), 1, 3)] if len(main) >= 3 else []
    rows += [("alternative", [x + "*"]) for x in core] + ([("single", ["N"])] if "N" in r["ids"] else [])
    rows = [(rl, _seq(r, t)) for rl, t in rows]
    for _, seq in list(rows):
        rows += [("closure", [a for i, a in enumerate(seq) if i != t]) for t in range(len(seq))] if len(seq) >= 2 else []
    room, pool, seen = QUERY_CAP - 1 - len(scene_q), [], set()
    for rl, seq in rows:
        if z4.sig(seq) in seen or len(pool) >= room:
            continue
        seen.add(z4.sig(seq))
        _, why = sq.parse_sequence(s, act, [list(x) for x in seq], z4.MOVE_CAP)
        pool.append({"sequence": [list(x) for x in seq], "role": rl, "canonical": not why, "parse": why,
                     "z_layout": z4.build(k, seq, [q for q in scene_q if q["kind"] != "nuisance"])})
    return {"scene_queries": scene_q, "pool": pool}


def plan(cfg, robot, pol, roots) -> dict:
    meta = {"stage": "P7B", "version": VERSION, "root_schedule": ROOTS, "strata": STRATA}
    out = p6.plan(cfg, robot, pol, roots, catalogue, meta)
    for k, r in out["roots"].items():
        src = next(x for x in roots if x["root"] == r["lineage"])
        r.update({"wave": src["wave"], "clutter": src["clutter"], "main": src["main"], "edit_roles": src["edit_roles"], "signature": src["signature"]})
    out["plan_hash"] = cf.stable_hash({k: v for k, v in out.items() if k != "plan_hash"})
    return out
