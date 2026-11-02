"""P5.1 D3-lite: frozen construction of 8 fresh independent roots and their 12-query catalogues.

Static and outcome-blind (no teacher call). Each root is drawn with the existing G4C family
rule (g4c._layout) from a predeclared seed: J2 = goal occupant A + adjacent access blocker B
(the r19 access/order pattern), J1 = goal occupant A + corridor blocker C (dual-blocker
interaction). The target must lie >= 0.45 m from the base (recorded selection bias against
the g4d_R1 near-base RRT tails). Every attempt and rejection is logged; no root is replaced
after any outcome. The reference scene (participants removed) is a separate root and never
shares a z vector with the obstructed root.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import config as cf
import data
import effects as ef
import g4c
import geometry as ge
import sequences as sq
import zlayout as zl
from scene import place_scene, scene_hash, scene_to_dict

VERSION = "D3_LITE_V1"
SEEDS = [2026102201, 2026102202, 2026102203, 2026102204, 2026102205, 2026102206, 2026102207, 2026102208]
FAMILY = ["J2", "J1", "J2", "J1", "J1", "J2", "J1", "J2"]
SPLIT = ["train", "train", "validation", "holdout", "train", "train", "validation", "holdout"]
BATCH = [1, 1, 1, 1, 2, 2, 2, 2]
MIN_TARGET_RHO, MAX_ATTEMPTS = 0.45, 400
HISTORICAL = ("runs/g4c_preflight/plan.json", "runs/g4c_pilot/pilot_plan.json", "runs/g4d_evidence/final_plan.json",
              "runs/g4c_d2_poc_s/final_plan.json")


def _known(root: Path) -> set:
    from scene import scene_from_dict
    out = set()
    for f in HISTORICAL:
        for r in json.loads((root / f).read_text())["roots"].values():
            if "scene" in r:
                out.add(g4c.stable_geometry(scene_from_dict(r["scene"]), r["goal_cell"]))
    return out


def _try(cfg, robot, i, attempt, known):
    """One deterministic attempt for root i: (record, None) or (None, rejection reason)."""
    rng = np.random.default_rng([SEEDS[i], attempt])
    lay = g4c._layout(rng, FAMILY[i])
    if lay is None:
        return None, "family intent predicate not met"
    t, g, role, obs = lay
    tx, ty = g4c._xy(t)
    if math.hypot(tx - robot.base[0], ty - robot.base[1]) < MIN_TARGET_RHO:
        return None, "target closer than 0.45 m to the base"
    k = f"d3_{SEEDS[i]}"
    s = place_scene(cfg, k, "obstructed", [t], obs)
    d, e = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"]
    if ge.Checker(robot, ge.make_world(s), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status != ge.CLEAR:
        return None, "initial robot not clear"
    if g4c.stable_geometry(s, g) in known:
        return None, "duplicate of a historical development geometry"
    A, B = "o00", "o01"                                          # role order: A (goal occupant), then B (J2) / C (J1)
    a = g4c._dests(s, A, t, g, k=2)
    b = g4c._dests(s, B, t, g, k=2, exclude=a)
    extras = [o for o in s.objects if o.kind == "obstacle" and o.id not in (A, B)]
    if len(a) < 2 or len(b) < 2 or not extras:
        return None, "destinations or nuisance object unavailable"
    nu = max(extras, key=lambda o: (g4c._corr(o.cell, t, g), o.xy_m))
    n = g4c._dests(s, nu.id, t, g, k=1, exclude=a + b)
    if not n:
        return None, "nuisance destination unavailable"
    ref = place_scene(cfg, f"{k}_ref", "reference", [t], [c for c in obs if c not in (role["A"], obs[1])])
    return {"index": i, "seed": SEEDS[i], "attempt": attempt, "family": FAMILY[i], "split": SPLIT[i], "batch": BATCH[i],
            "root": k, "scene": s, "ref_scene": ref, "target_cell": t, "goal_cell": g, "A": A, "B": B, "a": a, "b": b,
            "N": nu.id, "n": n[0]}, None


def construct(cfg, robot, root=Path(".")) -> tuple:
    known, roots, ledger = _known(root), [], []
    for i in range(len(SEEDS)):
        for attempt in range(MAX_ATTEMPTS):
            rec, why = _try(cfg, robot, i, attempt, known)
            ledger.append({"index": i, "seed": SEEDS[i], "attempt": attempt, "accepted": rec is not None, "reason": why})
            if rec:
                known.add(g4c.stable_geometry(rec["scene"], rec["goal_cell"]))
                roots.append(rec)
                break
        else:
            raise SystemExit(f"root {i}: no admissible layout in {MAX_ATTEMPTS} attempts (STOP before any teacher call)")
    return roots, ledger


def catalogue(cfg, robot, r) -> dict:
    """The frozen 12-query catalogue (1 reference + 11 obstructed-scene queries)."""
    k, s, A, B, (a, a2), (b, b2) = r["root"], r["scene"], r["A"], r["B"], r["a"], r["b"]
    act = data.make_query(cfg, k, s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
    edits = [("singleton:A", {A: a}), ("singleton:B", {B: b}), ("joint:A+B", {A: a, B: b}), ("nuisance", {r["N"]: r["n"]})]
    scene_q = []
    for kind, ed in edits:
        sub = ef.pose_substitution(robot, cfg, s, act, ed)
        scene_q.append({"kind": kind, "edit": ed, "eligible": bool(sub["eligible"]), "intervened_scene_hash": sub["intervened_scene_hash"]})
    seqs = [[], [(A, a)], [(B, b)], [(A, a), (B, b)], [(B, b), (A, a)], [(A, a2)], [(B, b2)]]
    role = ["original", "single", "single", "order_AB", "order_BA", "alternative", "alternative"]
    pool = []
    for seq, rl in zip(seqs, role):
        canon, why = sq.parse_sequence(s, act, [list(x) for x in seq], 3)
        lay = zl.build(k, seq, scene_q)
        pool.append({"sequence": [list(x) for x in seq], "role": rl, "canonical": not why, "parse": why, "z_layout": lay})
    return {"scene_queries": scene_q, "pool": pool}


def plan(cfg, robot, pol, roots) -> dict:
    out = {"stage": "P5_D3_LITE", "version": VERSION, "policy_id": pol["policy"]["policy_id"], "policy_hash": pol["policy_hash"],
           "seeds_policy": pol["policy"]["seeds"], "root_seeds": SEEDS, "families": FAMILY, "splits": SPLIT, "batches": BATCH,
           "min_target_rho_m": MIN_TARGET_RHO, "roots": {}, "queries": [], "units": [], "ineligible": [],
           "z_layout_version": zl.VERSION, "config_hash": cf.config_hash(cfg), "robot_profile_hash": cf.robot_profile_hash(cfg)}
    for r in roots:
        cat = catalogue(cfg, robot, r)
        for key, s, variant in ((r["root"] + "_ref", r["ref_scene"], "reference"), (r["root"], r["scene"], "obstructed")):
            q = data.make_query(cfg, key, s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
            out["roots"][key] = {"scene": scene_to_dict(s), "scene_hash": scene_hash(s), "hashes": q["hashes"], "goal_cell": r["goal_cell"],
                                 "lineage": r["root"], "variant": variant, "family": r["family"], "split": r["split"], "batch": r["batch"],
                                 "seed": r["seed"], "attempt": r["attempt"], "target_cell": r["target_cell"],
                                 "roles": {"A": r["A"], "B": r["B"], "N": r["N"]}, "queries": cat if variant == "obstructed" else None}
        todo = [(r["root"] + "_ref", "ORDERED_TEACHER", [], None, "reference")]
        todo += [(r["root"], "ORDERED_TEACHER", p["sequence"], None, p["role"]) for p in cat["pool"]]
        todo += [(r["root"], "SCENE_TARGET_ONLY", [], q, q["kind"]) for q in cat["scene_queries"]]
        for key, kind, seq, sqr, role in todo:
            if sqr is not None and not sqr["eligible"]:
                out["ineligible"].append({"root": key, "kind": role, "edit": sqr["edit"]})
                continue
            sig = zl.sig(sorted(sqr["edit"].items())) if sqr else zl.sig(seq)
            qid = f"{'scene' if sqr else 'seq'}:{key}:{sig}"
            out["queries"].append({"query": qid, "root": key, "lineage": r["root"], "kind": kind, "sequence": seq,
                                   "edit": sqr and sqr["edit"], "role": role, "split": r["split"], "batch": r["batch"]})
            for sd in pol["policy"]["seeds"]:
                key_ = pol["policy"]["seed_key"].format(root=key, sig=("scene/" if sqr else "") + sig, seed=sd)
                u = {"unit": f"{qid}#s{sd}", "query": qid, "seed": sd, "kind": kind, "root": key, "target_key": key_ + "/target",
                     "relocation_key": key_ + "/reloc"}
                u.update({"edit": sqr["edit"], "eligible": True, "intervened_scene_hash": sqr["intervened_scene_hash"]} if sqr
                         else {"sequence": seq})
                out["units"].append(u)
    out["plan_hash"] = cf.stable_hash({k: v for k, v in out.items() if k != "plan_hash"})
    return out
