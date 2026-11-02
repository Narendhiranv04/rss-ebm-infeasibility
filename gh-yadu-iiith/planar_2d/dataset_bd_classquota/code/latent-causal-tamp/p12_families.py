"""P12 new 2D scene grammars (P12_FAMILIES_V1). Physics-first, no P10a/P10b template, no learned prediction, no teacher label:
  X_SWEEP_GATE   two blockers A, B sampled INDEPENDENTLY in the link-0 inner annulus within a +-0.5 rad window around the arc between the
                 start link-0 angle and the target bearing (either side of it), plus an extra object C within Chebyshev 2 of the target.
                 Prescreen (sound, analytic): the ORIGINAL task carries an accepted LINK0_SECTOR_PICK certificate.
  Y_BOTTLENECK   two blockers A, B sampled from annulus-safe cells at Chebyshev 1-2 from the target inside its approach cone (bearing to the
                 base within +-75 deg), asymmetric by construction; each non-cone Chebyshev-1 cell is a FIXED body with probability 0.7
                 (V1.1, DEV revision after V1 gave 0/60 prescreen passes); C sampled outside the cone within Chebyshev 2.
                 Prescreen (sound, analytic): the ORIGINAL task carries an accepted TARGET_PICK certificate.
The interaction class (kappa) of A, B is NOT controlled: it follows from geometry and is decided only by the teacher. Generic outcome-blind
catalogue: empty, singles, ordered pairs, ordered triples of A, B, C; edits A / B / A+B; every D parking subset of {A, B, C}."""
from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import g4c
import geometry as ge
import p10b_scenes as pb
import p6_scenes as p6
from p6_scenes import STRATA, _cheb, _xy
from scene import place_scene

VERSION = "P12_FAMILIES_V1_1"
FAMS = ("X_SWEEP_GATE", "Y_BOTTLENECK")
OUTD = Path("runs/p12")
TOKENS = ["", "A", "B", "C", "A B", "B A", "A C", "C A", "B C", "C B", "A B C", "A C B", "B A C", "B C A", "C A B", "C B A",
          "edit:A", "edit:B", "edit:A+B"]
for _f in FAMS:
    pb.EDITS[_f] = {"A": "A", "B": "B", "A+B": "AB"}
    pb.CATALOGUE[_f] = TOKENS
    pb.MOVERS[_f] = "ABC"


def _ang(base, c):
    x, y = _xy(c)
    return math.atan2(y - base[1], x - base[0])


def layout(rng, fam, scr, stratum):
    """(T, goal, A, B, C, fixed cells) or None. Pure geometry; no certificate, screen or label."""
    base, valid = scr.robot.base, sorted(scr.valid)
    T = int(rng.choice(valid))
    goal = int(rng.choice([c for c in valid if _cheb(c, T) >= 3]))
    if fam == "X_SWEEP_GATE":
        a0, at = STRATA[stratum][0], _ang(base, T)
        lo, hi = min(a0, at) - 0.5, max(a0, at) + 0.5
        inner = [c for c in range(64) if c not in scr.valid and math.dist(_xy(c), base) < scr.r0 - 0.02 and lo < _ang(base, c) < hi]
        if len(inner) < 2:
            return None
        A, B = (int(x) for x in rng.choice(inner, 2, replace=False))
        near = [c for c in valid if 1 <= _cheb(c, T) <= 2 and c != goal]
    else:
        bt = math.atan2(base[1] - _xy(T)[1], base[0] - _xy(T)[0])
        cone = lambda c: abs((math.atan2(_xy(c)[1] - _xy(T)[1], _xy(c)[0] - _xy(T)[0]) - bt + math.pi) % (2 * math.pi) - math.pi) <= math.radians(75)  # noqa: E731
        ring = [c for c in valid if 1 <= _cheb(c, T) <= 2 and c != goal]
        inc = [c for c in ring if cone(c)]
        if len(inc) < 2:
            return None
        A, B = (int(x) for x in rng.choice(inc, 2, replace=False))
        side = [c for c in ring if not cone(c) and _cheb(c, T) == 1]
        fixed = [c for c in side if rng.random() < 0.7]           # V1.1 (DEV revision): fixed bodies form the asymmetric bottleneck
        near = [c for c in ring if not cone(c) and c not in fixed]
        near = [c for c in near if c not in (A, B)]
        if not near:
            return None
        return T, goal, A, B, int(rng.choice(near)), fixed
    near = [c for c in near if c not in (A, B)]
    if not near:
        return None
    return T, goal, A, B, int(rng.choice(near)), []


def try_root(cx, spec, attempt, known):
    fam, key = spec["family"], f"p12_{spec['seed']}"
    if fam not in FAMS:
        return _orig_try(cx, spec, attempt, known)
    rng = np.random.default_rng([spec["seed"], attempt])
    stratum = f"S{int(rng.integers(0, 4))}"
    lay = layout(rng, fam, cx.scr, stratum)
    if lay is None:
        return None, "layout"
    T, goal, A, B, C, fixed = lay
    s = place_scene(cx.cfg, key, "obstructed", [T], [A, B, C] + fixed)
    s = replace(s, robot=replace(s.robot, q_rad=tuple(STRATA[stratum])))
    d, e = cx.cfg["suite"]["margin_m"], cx.cfg["geometry"]["numeric_margin_band_m"]
    if ge.Checker(cx.robot, ge.make_world(s), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status != ge.CLEAR:
        return None, "start not clear"
    if g4c.stable_geometry(s, goal) in known:
        return None, "duplicate geometry"
    ids = {"A": "o00", "B": "o01", "C": "o02"}
    cert = cx.certified(s, goal)
    need = "LINK0_SECTOR_PICK" if fam == "X_SWEEP_GATE" else "TARGET_PICK"
    if need not in cert:
        return None, f"prescreen: original lacks {need}"
    dest = {}
    for r in "ABC":
        dd = p6._dests(cx.scr, s, ids[r], [T, goal], [x for v in dest.values() for x in v], 1)
        if not dd:
            return None, f"no destination for {r}"
        dest[r] = dd
    rec = {**spec, "attempt": attempt, "root": key, "scene": s, "goal_cell": goal, "stratum": stratum, "ids": ids, "dest": dest,
           "role_cells": {"T": T, "A": A, "B": B, "C": C}, "fixed": fixed, "mechanism": fam, "variant": "",
           "predicates": [{"check": f"original {need} (sound prescreen)", "value": True, "want": True, "ok": True}],
           "construction": f"P12 GEOMETRY GRAMMAR + sound {need} prescreen (conditional hard stratum; no learned / teacher input)"}
    rec["parking"] = pb.parking(cx, s, goal, [ids[r] for r in "ABC"])
    if len(rec["parking"]) != 3:
        return None, "D parking map unavailable"
    rec["family"], rec["fingerprint"] = fam, [T, goal, sorted(o.cell for o in s.objects)]
    return rec, None


_orig_try = pb.try_root
pb.try_root = try_root


def probe(n=40, tag="") -> None:
    """DEV-only static constructibility: acceptance and rejection reasons per family (development seeds; never roster seeds)."""
    import collections
    import json
    import config as cf
    import p10_scenes as ps
    cfg = cf.load_config("config.toml")
    cx, out = ps.Ctx(cfg, ge.make_robot(cfg)), {}
    for i, fam in enumerate(FAMS):
        why = collections.Counter()
        for a in range(n):
            rec, w = try_root(cx, {"family": fam, "seed": 2026139000 + 101 * i, "index": -1, "wave": 0}, a, set())
            why["ACCEPTED" if rec else w] += 1
        out[fam] = {"attempts": n, "accept_rate": why["ACCEPTED"] / n, "reasons": dict(why)}
    (OUTD / "p12_1").mkdir(parents=True, exist_ok=True)
    (OUTD / "p12_1" / f"probe{tag}.json").write_text(json.dumps(out, indent=1))


def dev_specs(n_per=8):
    return [{"family": FAMS[i % 2], "seed": 2026138000 + i, "index": i, "wave": 0, "batch": 0, "split": "dev"} for i in range(2 * n_per)]


def dev_pilot(n_per=8) -> None:
    import p10b_dataset as pd
    d = OUTD / "dev"
    if not (d / "plan.json").exists():
        pd.build_plan(d, dev_specs(n_per), max_attempts=400, root_cpu=90.0)
    if not (d / "static_admission.json").exists():
        pd.admit(d)
    pd.collect(d, chunk_cpu=2600.0)


def atlas() -> None:
    """DEV atlas: accepted dev roots (statuses from the dev pilot) and rejected attempts (layout + rejection reason), static panels only."""
    import json
    import config as cf
    import p6_viz as pv
    import p10_scenes as ps
    from scene import scene_from_dict
    from viz import plt
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    cx, plan = ps.Ctx(cfg, robot), json.loads((OUTD / "dev" / "plan.json").read_text())
    f, ax = plt.subplots(2, 4, figsize=(22, 11))
    for j, fam in enumerate(FAMS):
        ks = [k for k, r in plan["roots"].items() if r["family"] == fam][:2]
        for i, k in enumerate(ks):
            r = plan["roots"][k]
            s = scene_from_dict(r["scene"])
            a = ax[j, i]
            pv._panel(a, cfg, robot, s, r["goal_cell"], f"DEV ACCEPTED {fam} {k[-3:]} ({r['stratum']})")
            for role, c in r["role_cells"].items():
                if role != "T":
                    x, y = _xy(c)
                    a.text(x, y + 0.03, role, ha="center", fontsize=9, weight="bold", color="#c0392b", zorder=12)
            pv._tag(a, r["predicates"][0]["check"], "#34495e", size=6)
        rej = [x for x in plan["attempts"] if not x["accepted"] and (x["reason"] or "").startswith("prescreen") and
               next(s["family"] for s in dev_specs() if s["seed"] == x["seed"]) == fam][:2]
        for i, x in enumerate(rej):
            rng = np.random.default_rng([x["seed"], x["attempt"]])
            st = f"S{int(rng.integers(0, 4))}"
            lay = layout(rng, fam, cx.scr, st)
            T, goal, A, B, C, fixed = lay
            s = place_scene(cfg, "rej", "obstructed", [T], [A, B, C] + fixed)
            s = replace(s, robot=replace(s.robot, q_rad=tuple(STRATA[st])))
            a = ax[j, 2 + i]
            pv._panel(a, cfg, robot, s, goal, f"DEV REJECTED {fam} seed {x['seed']} att {x['attempt']}")
            pv._tag(a, x["reason"], "#7f7f7f", size=6)
    (Path("visualizations/p12")).mkdir(parents=True, exist_ok=True)
    pv.png(f, Path("visualizations/p12") / "P12_1_dev_atlas.png", "P12-1 DEV: accepted roots and prescreen-rejected attempts per family (static panels)")
