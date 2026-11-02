"""P15-1: one validated scene -> D-input builder for arbitrary (state, target, goal, robot q) queries, and matched physical probes.

The builder re-runs the TRAINING pipeline (data.make_query -> appearance_anchor -> learner_tensors_v2 -> DINOv2 mask pooling) on the
hypothetical scene: object poses are the state's, the queried object takes the target role (kind 'target', is_active_target = 1, the former
target becomes an ordinary obstacle), goal_xy is the declared placement cell, and the scene's robot configuration is the current q (so
robot_q, robot_fk_joints, base and context are recomputed). The rendered image (identical cylinders, no target / goal / robot drawn) is
re-rendered for the moved objects, so visual tokens match the state. Nothing is patched in place. The physical probe asks the SAME task:
pick the object from q, carry and place it at the goal cell (skill.plan_skill), adjudicated VERIFIED (witness) / CERTIFIED (checker-accepted
sound certificate for exactly this task) / UNKNOWN (anything else, never a negative)."""
from __future__ import annotations

import collections
import math
from dataclasses import replace

import numpy as np

import config as cf
from p15_bench import OUT, rd

_DINO = None


def retarget_scene(scene, positions, target_id, q):
    """Hypothetical scene: per-object cells, target role moved to target_id, robot at q (open, free)."""
    objs = []
    for o in scene.objects:
        c = positions.get(o.id, o.cell)
        kind = "target" if o.id == target_id else ("obstacle" if o.kind == "target" else o.kind)
        objs.append(replace(o, cell=c, xy_m=scene.lattice.center(c), kind=kind))
    return replace(scene, objects=tuple(objs), robot=replace(scene.robot, q_rad=tuple(float(x) for x in q)))


def build_record(cfg, scene, target_id, goal_cell, key="p15"):
    """Learner record inputs for the query (scene, target_id, goal) through the exact training pipeline."""
    import data
    import learn_data as ld
    global _DINO
    if _DINO is None:
        _DINO = ld.dino()
    q = data.make_query(cfg, key, scene, target_id, goal_cell, {}, "SAFE_RETREAT", data.PROTOCOL_V2)
    anchor = data.appearance_anchor(q, ld.IMAGE_PX)
    t = data.learner_tensors_v2(q, cfg, anchor)
    ids = [o["id"] for o in q["learner"]["scene"]["objects"]]
    inputs = {k: np.asarray(t[k], np.float32) for k in ld.INPUT_KEYS if k != "visual"} | {"visual": ld.visual_features(*_DINO, anchor)}
    return {"inputs": inputs, "audit": {"ids": ids, "target": target_id, "goal": goal_cell, "anchor_hash": anchor["hash"]}}


def d_predict(models, rec, target_id, pool_ids):
    """Mean over seeds of p_feasible and blocker marginals for a rebuilt record; the queried object is never in its own pool."""
    import abc_models as am
    import d_blockers as db
    R = am.Root(rec)
    ids = rec["audit"]["ids"]
    if R.tid != ids.index(target_id):
        raise ValueError("target flag does not point at the queried object")
    pool = [ids.index(o) for o in pool_ids if o != target_id]
    ps = [db.predict(m, R, pool) for m in models]
    return {"p_feasible": float(np.mean([p["p_feasible"] for p in ps])), "marg": [float(np.mean([p["marg"][i] for p in ps])) for i in range(len(pool))],
            "pool": [ids[i] for i in pool]}


def certified(cfg, robot, scene, target_id, goal_cell):
    """Checker-accepted sound certificate kinds for picking target_id from the scene's robot q and placing it at goal_cell."""
    import data
    import ordered as od
    import sector_cert
    import sector_check
    from scene import scene_hash
    ln = data.make_query(cfg, "c", scene, target_id, goal_cell, {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]
    acc = [c["kind"] for c in od.certificates_for(robot, cfg, scene, ln, []) if c["checker_accepted"]]
    if not acc:
        rec = sector_cert.produce(robot, cfg, scene, target_id, "ORIGINAL_TARGET", [], goal_cell)
        src = {"scene": sector_check.rebuild(scene, edit=None), "scope": "ORIGINAL_TARGET", "moved": target_id, "sequence": [],
               "kind": "ORDERED_TEACHER", "expected_hash": scene_hash(scene)}
        if sector_check.check(robot, cfg, rec, src)["accepted"]:
            acc.append("LINK0_SECTOR_PICK")
    return acc


def probe(cfg, robot, scene, target_id, goal_cell, q, attempts=3) -> dict:
    """Matched physical query: VERIFIED (planner witness within `attempts` state-keyed tries), CERTIFIED (sound certificate), else UNKNOWN."""
    import skill as sk
    c = certified(cfg, robot, scene, target_id, goal_cell)
    if c:
        return {"status": "CERTIFIED_INFEASIBLE", "certificates": c}
    last = None
    for a in range(attempts):
        key = "p15probe/" + cf.stable_hash([str(scene.objects), target_id, goal_cell, list(map(float, q)), a])[7:23]
        r = sk.plan_skill(robot, scene, target_id, goal_cell, cfg, key, q_start=np.asarray(q, float), q_goal=None)
        last = r.status
        if r.status == sk.VERIFIED:
            return {"status": "VERIFIED_FEASIBLE", "attempt": a}
    return {"status": "UNKNOWN", "planner": last}


def run_probes(n_roots=30, goals_per_object=2) -> dict:
    """DEV matched probes (P13 train / validation roots, original scenes + one-relocation intermediate states from DEV teacher witnesses):
    every movable obstacle as temporary target, `goals_per_object` declared destination slots; old P14 interface vs corrected builder."""
    import os
    import torch
    os.environ["P10C_OUT"] = "runs/p13/models"
    import geometry as ge
    import p6_scenes as p6
    import p13_pipeline as pp
    import p14_d_search as ds
    from scene import scene_from_dict
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    scr = p6.Screens(cfg, robot)
    models = ds.d_models()
    rs = {r["key"]: r for r in pp._prepared("trainval") if str(r["audit"].get("source", "")).startswith("P13")}
    P = {k: r for d in sorted((pp.OUT / "data").glob("*_b*")) for k, r in pp.rd(d / "plan.json")["roots"].items()}
    keys = sorted(k for k in rs if P[k]["family"] not in ("TRI_SERIAL", "TRI_CTRL"))[:n_roots]
    rows = []
    for k in keys:
        root, rr = P[k], rs[k]
        s0 = scene_from_dict(root["scene"])
        q0 = np.asarray(s0.robot.q_rad, float)
        movers = sorted({o for it in root["items"] if it["kind"] == "ORDERED_TEACHER" for o, _ in it["sequence"]})
        for o in movers:
            T, G = s0.get("t0").cell, root["goal_cell"]
            for g in p6._dests(scr, s0, o, [T, G], [], goals_per_object):
                sk_ = retarget_scene(s0, {}, o, q0)
                new = d_predict(models, build_record(cfg, sk_, o, g, key=f"{k}/{o}/{g}"), o, movers + ["t0"])
                xy = torch.as_tensor(np.array([s0.get(x).xy_m for x in rr["ids"]]), dtype=torch.float64)
                R2 = ds.root_variant(rr["R"], robot, xy, rr["ids"].index(o), q0)
                old = float(np.mean([__import__("d_blockers").predict(m, R2, [rr["ids"].index(x) for x in movers if x != o])["p_feasible"] for m in models]))
                ph = probe(cfg, robot, sk_, o, g, q0)
                rows.append({"root": k, "family": root["family"], "object": o, "goal": g, "p_new": new["p_feasible"], "p_old": old, **ph})
    out = {"rows": rows, "summary": score(rows)}
    (OUT / "retarget").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "retarget" / "PROBES.json", out)
    return out


def score(rows) -> dict:
    """Decided probes only (VERIFIED / CERTIFIED); UNKNOWN reported, never scored. Root-cluster bootstrap CIs."""
    dec = [r for r in rows if r["status"] in ("VERIFIED_FEASIBLE", "CERTIFIED_INFEASIBLE")]
    res = {"probes": len(rows), "decided": len(dec), "status_counts": dict(collections.Counter(r["status"] for r in rows)),
           "roots": len({r["root"] for r in rows}), "decided_roots": len({r["root"] for r in dec})}
    roots = sorted({r["root"] for r in dec})
    rng = np.random.default_rng(0)
    for name in ("p_old", "p_new"):
        def m(sub):
            y = np.array([r["status"] == "VERIFIED_FEASIBLE" for r in sub], float)
            p = np.array([r[name] for r in sub])
            pos, neg = p[y == 1], p[y == 0]
            fr = float((pos >= 0.5).mean()) if pos.size else math.nan
            ir = float((neg < 0.5).mean()) if neg.size else math.nan
            auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if pos.size and neg.size else math.nan
            return {"balanced_accuracy": (fr + ir) / 2, "feasible_recall": fr, "infeasible_recall": ir, "auroc": auc,
                    "brier": float(np.mean((p - y) ** 2)), "nll": float(np.mean(-np.log(np.clip(np.where(y == 1, p, 1 - p), 1e-9, 1))))}
        base = m(dec)
        boots = []
        for _ in range(1000):
            pick = set(rng.choice(roots, len(roots)))
            sub = [r for r in dec if r["root"] in pick]
            if sub:
                boots.append(m(sub)["balanced_accuracy"])
        res[name] = base | {"balanced_accuracy_root_ci95": [float(np.nanpercentile(boots, 2.5)), float(np.nanpercentile(boots, 97.5))]}
    res["gate"] = {"threshold": "balanced_accuracy >= 0.70 and feasible_recall >= 0.70 (engineering-use, corrected interface)",
                   "pass": bool(res["p_new"]["balanced_accuracy"] >= 0.70 and res["p_new"]["feasible_recall"] >= 0.70)}
    return res


def run_intermediate(max_probes=60) -> dict:
    """Intermediate-state probes (DEV): state after the FIRST relocation of a VERIFIED >= 2-move DEV teacher sequence (object poses + robot q
    from the recorded witness end), temporary target = the NEXT mover, goal = its actual destination. Old interface (stale visual tokens,
    original goal) vs rebuilt record, against the matched physical probe from that q."""
    import os
    import torch
    os.environ["P10C_OUT"] = "runs/p13/models"
    import g4d_run as gr
    import geometry as ge
    import p13_pipeline as pp
    import p14_d_search as ds
    import d_blockers as db
    from pathlib import Path
    from scene import scene_from_dict
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    models = ds.d_models()
    rs = {r["key"]: r for r in pp._prepared("trainval") if str(r["audit"].get("source", "")).startswith("P13")}
    rows = []
    for d in sorted((pp.OUT / "data").glob("*_b*")):
        plan = pp.rd(d / "plan.json")
        for f in sorted(Path(d / "q").glob("seq__*/seed*/units/*/relocation.npz")):
            if len(rows) >= max_probes:
                break
            job = pp.rd(f.parent / "job.json")["unit"]
            k, seq = job["root"], job["sequence"]
            if k not in rs or len(seq) < 2 or not (f.parent / "final.json").exists():
                continue
            arrs = gr._arrays(f)
            root = plan["roots"][k]
            s0 = scene_from_dict(root["scene"])
            (o1, d1), (o2, d2) = seq[0], seq[1]
            q1 = np.asarray(arrs[0]["q"][-1], float)
            st = retarget_scene(s0, {o1: int(d1)}, o2, q1)
            movers = sorted({o for it in root["items"] if it["kind"] == "ORDERED_TEACHER" for o, _ in it["sequence"]})
            new = d_predict(models, build_record(cfg, st, o2, int(d2), key=f"{k}/mid/{o2}/{d2}"), o2, movers + ["t0"])
            rr = rs[k]
            xy = torch.as_tensor(np.array([st.get(x).xy_m for x in rr["ids"]]), dtype=torch.float64)
            R2 = ds.root_variant(rr["R"], robot, xy, rr["ids"].index(o2), q1)
            old = float(np.mean([db.predict(m, R2, [rr["ids"].index(x) for x in movers if x != o2])["p_feasible"] for m in models]))
            rows.append({"root": k, "family": root["family"], "object": o2, "goal": int(d2), "after": [o1, int(d1)], "p_new": new["p_feasible"], "p_old": old,
                         **probe(cfg, robot, st, o2, int(d2), q1)})
    out = {"rows": rows, "summary": score(rows)}
    cf.write_json(OUT / "retarget" / "PROBES_INTERMEDIATE.json", out)
    return out


def recalibration() -> dict:
    """DEV-only scalar recalibration of D's logit (p' = sigmoid((logit p - b) / T)), root-split 2-fold cross-validation (fit on one half of
    the probe roots, score on the other); reports cross-validated metrics for the corrected interface. Never uses evaluation roots."""
    from scipy.optimize import minimize
    rows = [r for f in ("PROBES.json", "PROBES_INTERMEDIATE.json") if (OUT / "retarget" / f).exists() for r in rd(OUT / "retarget" / f)["rows"]]
    dec = [r for r in rows if r["status"] in ("VERIFIED_FEASIBLE", "CERTIFIED_INFEASIBLE")]
    roots = sorted({r["root"] for r in dec})
    lg = lambda p: math.log(min(max(p, 1e-9), 1 - 1e-9) / (1 - min(max(p, 1e-9), 1 - 1e-9)))  # noqa: E731

    def fit(sub):
        x = np.array([lg(r["p_new"]) for r in sub])
        y = np.array([r["status"] == "VERIFIED_FEASIBLE" for r in sub], float)
        f = lambda th: float(np.mean(np.logaddexp(0, -(2 * y - 1) * (x - th[0]) / math.exp(th[1]))))  # noqa: E731
        return minimize(f, [0.0, 0.0], method="Nelder-Mead").x
    out_rows = []
    for fold in (0, 1):
        tr = {r for i, r in enumerate(roots) if i % 2 == fold}
        b, lt = fit([r for r in dec if r["root"] in tr])
        for r in dec:
            if r["root"] not in tr:
                out_rows.append({**r, "p_new": 1 / (1 + math.exp(-(lg(r["p_new"]) - b) / math.exp(lt))), "fold": fold, "bias": float(b), "temp": float(math.exp(lt))})
    b, lt = fit(dec)
    res = {"cross_validated": score(out_rows), "fit_all_dev": {"bias": float(b), "temperature": float(math.exp(lt))}, "decided": len(dec), "roots": len(roots)}
    res["gate"] = {"raw_pass": score(dec)["gate"]["pass"], "recalibrated_cv_pass": res["cross_validated"]["gate"]["pass"]}
    cf.write_json(OUT / "retarget" / "RECALIBRATION.json", res)
    return res
