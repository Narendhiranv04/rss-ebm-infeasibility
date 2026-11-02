"""P11-4: constructor-disjoint 2D stress test (P11_NEW_FAMILIES_V1). NON-adaptive generators: no certificate targeting, no pose screens,
no template of P10a/P10b; the teacher decides every outcome.
  NF1_CLUTTER  4-7 obstacles at uniformly random annulus-safe cells within Chebyshev 3 of a random target; random goal / start stratum.
  NF2_RING     each of the target's 8 neighbours occupied with probability 0.6, plus 1-3 random cells within Chebyshev 3.
Movers = the 3 obstacles nearest the target (ties by cell). Generic outcome-blind catalogue: empty, every single, ordered pair and ordered
triple of the movers (first destinations by the frozen p6._dests rule), edits A / B / A+B on the two nearest, every D parking subset.
Plans reuse p10b_dataset.build_plan / admit / collect with this family dispatch; blind labels stay under <batch>/sealed/ and are read only
after runs/p11/new_holdout/UNSEALED exists (written after the frozen-prediction commit)."""
from __future__ import annotations

import itertools
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import g4c
import geometry as ge
import p10b_scenes as pb
import p6_scenes as p6
from p6_scenes import STRATA, _cell, _cheb, _xy
from scene import place_scene

VERSION = "P11_NEW_FAMILIES_V1"
FAMS = ("NF1_CLUTTER", "NF2_RING")
OUTD = Path("runs/p11/new_holdout")
SEED0 = 2026120001
TOKENS = ["", "A", "B", "C", "A B", "B A", "A C", "C A", "B C", "C B", "A B C", "A C B", "B A C", "B C A", "C A B", "C B A",
          "edit:A", "edit:B", "edit:A+B"]
for _f in FAMS:                                    # register the generic catalogue with p10b_scenes (read-only extension)
    pb.EDITS[_f] = {"A": "A", "B": "B", "A+B": "AB"}
    pb.CATALOGUE[_f] = TOKENS
    pb.MOVERS[_f] = "ABC"


def roster() -> list:
    """40 roots in two batches of 20 (10 per family each), all 'blind'; frozen before any label."""
    return [{"index": i, "family": FAMS[i % 2], "split": "blind", "seed": SEED0 + i, "wave": 0, "batch": 1 + i // 20} for i in range(40)]


def _layout(rng, fam, scr):
    valid = sorted(scr.valid)
    T = int(rng.choice(valid))
    goal = int(rng.choice([c for c in valid if _cheb(c, T) >= 3]))
    near = [c for c in valid if 1 <= _cheb(c, T) <= 3 and c != goal]
    if fam == "NF1_CLUTTER":
        k = int(rng.integers(4, 8))
        if len(near) < k:
            return None
        obs = [int(x) for x in rng.choice(near, k, replace=False)]
    else:
        ring = [c for c in near if _cheb(c, T) == 1]
        obs = [c for c in ring if rng.random() < 0.6]
        rest = [c for c in near if c not in obs]
        extra = int(rng.integers(1, 4))
        if len(rest) < extra:
            return None
        obs += [int(x) for x in rng.choice(rest, extra, replace=False)]
    return T, goal, obs


def try_root(cx, spec, attempt, known):
    fam, key = spec["family"], f"p11_{spec['seed']}"
    if fam not in FAMS:
        return _orig_try(cx, spec, attempt, known)
    rng = np.random.default_rng([spec["seed"], attempt])
    stratum = f"S{int(rng.integers(0, 4))}"
    lay = _layout(rng, fam, cx.scr)
    if lay is None:
        return None, "layout"
    T, goal, obs = lay
    obs = sorted(obs, key=lambda c: (math.dist(_xy(c), _xy(T)), c))
    s = place_scene(cx.cfg, key, "obstructed", [T], obs)
    s = replace(s, robot=replace(s.robot, q_rad=tuple(STRATA[stratum])))
    d, e = cx.cfg["suite"]["margin_m"], cx.cfg["geometry"]["numeric_margin_band_m"]
    if ge.Checker(cx.robot, ge.make_world(s), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status != ge.CLEAR:
        return None, "start not clear"
    if g4c.stable_geometry(s, goal) in known:
        return None, "duplicate geometry"
    roles = "ABC"[:min(3, len(obs))]
    if len(roles) < 2:
        return None, "fewer than two movers"
    ids = {r: f"o{i:02d}" for i, r in enumerate(roles)}
    dest = {}
    for r in roles:
        dd = p6._dests(cx.scr, s, ids[r], [T, goal], [x for v in dest.values() for x in v], 1)
        if not dd:
            return None, f"no destination for {r}"
        dest[r] = dd
    rec = {**spec, "attempt": attempt, "root": key, "scene": s, "goal_cell": goal, "stratum": stratum, "ids": ids, "dest": dest,
           "role_cells": {"T": T, **{r: obs[i] for i, r in enumerate(roles)}}, "fixed": obs[len(roles):], "predicates": [],
           "mechanism": fam, "variant": "", "construction": "RANDOM (non-adaptive; no certificate targeting, no pose screens)"}
    rec["parking"] = pb.parking(cx, s, goal, [ids[r] for r in roles])
    if len(rec["parking"]) != len(roles):
        return None, "D parking map unavailable"
    rec["family"], rec["fingerprint"] = fam, [T, goal, sorted(o.cell for o in s.objects)]
    return rec, None


_orig_try = pb.try_root
pb.try_root = try_root
_orig_cat = pb.catalogue


def catalogue(cfg, robot, r):
    """Generic catalogue; tokens naming an absent mover (two-mover roots) are dropped before freezing."""
    if r["family"] in FAMS and "C" not in r["ids"]:
        pb.CATALOGUE[r["family"]] = [t for t in TOKENS if "C" not in t.split() and t not in ("edit:A+B",)] + ["edit:A+B"]
        try:
            return _orig_cat(cfg, robot, r)
        finally:
            pb.CATALOGUE[r["family"]] = TOKENS
    return _orig_cat(cfg, robot, r)


pb.catalogue = catalogue


def build(batch, specs) -> None:
    import p10b_dataset as pd
    pd.build_plan(OUTD / batch, specs, max_attempts=300, root_cpu=60.0)


def dev_specs():
    return [{"family": f, "seed": 2026129000 + i, "index": i, "wave": 0, "batch": 0, "split": "dev"} for i, f in enumerate(FAMS * 2)]


def sealed_records(batch) -> dict:
    """Adjudicated records of a P11 batch; refuses until runs/p11/new_holdout/UNSEALED exists (written after the prediction commit)."""
    import json
    import p6_run as p6r
    import operational_teacher as ot
    out = OUTD / batch
    plan = json.loads((out / "plan.json").read_text())
    if any(r["split"] == "blind" for r in plan["roots"].values()) and not (OUTD / "UNSEALED").exists():
        raise SystemExit("P11 new-holdout labels are sealed")
    static, pol = json.loads((out / "static_admission.json").read_text())["certificates"], ot.load_policy("teacher_opneg.toml")
    res = {}
    for k, r in plan["roots"].items():
        pf = out / ("sealed" if r["split"] == "blind" else "progress") / f"{k}.json"
        done = json.loads(pf.read_text())["done"] if pf.exists() else {}
        for q in plan["queries"]:
            if q["root"] == k:
                res[q["query"]] = p6r.judge(static, pol, q, done.get(q["query"]))
    return res


def _unused():
    return itertools, _cell


def dev_pilot() -> None:
    import p10b_dataset as pd
    d = OUTD / "dev"
    if not (d / "plan.json").exists():
        build("dev", dev_specs())
    if not (d / "static_admission.json").exists():
        pd.admit(d)
    pd.collect(d, chunk_cpu=1000.0)


def static_obstruction_rate(n=40) -> None:
    """Zero-teacher diagnostic: fraction of generated roots whose ORIGINAL task has an accepted sound static certificate (obstructed)."""
    import json
    import collections
    import config as cf
    import p10_scenes as ps
    cfg = cf.load_config("config.toml")
    cx, out = ps.Ctx(cfg, ge.make_robot(cfg)), {}
    for fam in FAMS:
        c = collections.Counter()
        for i in range(n):
            rec, _ = try_root(cx, {"family": fam, "seed": 2026127000 + i, "index": -1, "wave": 0}, 0, set())
            if rec is None:
                c["not_built"] += 1
                continue
            cert = cx.certified(rec["scene"], rec["goal_cell"])
            c["certified_obstructed" if cert else "no_static_certificate"] += 1
        out[fam] = dict(c)
    (OUTD / "dev").mkdir(parents=True, exist_ok=True)
    (OUTD / "dev" / "STATIC_OBSTRUCTION_RATE.json").write_text(json.dumps(out, indent=1))
