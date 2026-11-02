"""P10c freeze + P10d: validation-only calibration, the frozen blind prediction snapshot (all arms), unsealing, strict TOP1_COMMIT
executed episodes (A / B / C / SF / geometry-only / uniform) and the planner-assisted G+M reference.

Freeze: per checkpoint, temperature (p7_eval.fit_temperature) and admission threshold (P8 rule: fewest false-feasible empty repairs,
then most successes, then fewest moves over {0.5, 0.7, 0.85, 0.95}) on P10b VALIDATION roots; root-prior marginals for every modelled
query and the TOP1 choice on validation, blind and H9 roots; D: p_feasible, conditional blocker marginals, MAP; the D threshold and
lambda_B chosen on validation. No teacher, IK or planner call in any learned path (counted). Blind labels unseal only after the
snapshot is committed. Episodes: ordered.primary_v2 with a ONE-element ranking (no fallback), seed namespace keyed by the physical
choice so identical (root, repair) choices are one deterministic episode.
"""
from __future__ import annotations

import contextlib
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch

import config as cf
import d_blockers as db
import p10_eval as pe
import p10_protocol as pp
import p10c_train_eval as pc
import p9_l3 as l3
import unique_factor_keyed as ufk

OUT = Path(__import__("os").environ.get("P10D_OUT", "runs/p10d"))
THRESHOLDS = (0.5, 0.7, 0.85, 0.95)


def _rd(p):
    return json.loads(Path(p).read_text())


def plans():
    return {k: r for b in pc.batches() for k, r in _rd(b / "plan.json")["roots"].items()}


def choose_threshold(evals, tau):
    """P8 rule on validation evals (rows carry y): (false-feasible empty, -success, moves on success, thr)."""
    best = None
    for thr in THRESHOLDS:
        ffe = succ = 0
        mv = []
        for e in evals:
            p = {c["query"]: 1 / (1 + math.exp(-c["u"] / tau)) for c in e["cands"]}
            adm = sorted([c for c in e["cands"] if p[c["query"]] >= thr], key=lambda c: (c["m"], c["J_pred"], c["C_d"], c["energy"], c["query"]))
            ch = adm[0] if adm else None
            y0 = next((c["y"] for c in e["cands"] if c["m"] == 0), None)
            ffe += bool(ch and ch["m"] == 0 and y0 == 0)
            if ch and ch["y"] == 1:
                succ += 1
                mv.append(ch["m"])
        key = (ffe, -succ, float(np.mean(mv)) if mv else 9.0, thr)
        best = min(best, key) if best else key
    return best[3]


def _mode(x):
    return "unique" if x["arm"] not in ("A", "SF") else "factorized"


def geometry_choice(r, rr, cfg) -> str:
    """Analytic geometry-only TOP1 (no IK / planner): shortest catalogue repair whose moved set covers the heuristic blockers
    (grasp-adjacent objects and link-0 sweep occluders, d_blockers.heuristic over the catalogue's movable objects), not statically vetoed; ties by
    analytic displacement then id. Abstains if none covers."""
    from scene import scene_from_dict
    s = scene_from_dict(r["scene"])
    movable = sorted({o for c in r["items"] if c["kind"] == "ORDERED_TEACHER" for o, _ in c["sequence"]})
    obs = [rr["ids"].index(o) for o in movable]                 # fixed bodies cannot be moved by any catalogue repair
    need = {rr["ids"][i] for i, h in zip(obs, db.heuristic(rr["R"], obs)) if h}
    rows = []
    for c in r["items"]:
        if c["kind"] != "ORDERED_TEACHER":
            continue
        seq = [tuple(a) for a in c["sequence"]]
        if need <= {o for o, _ in seq} and not pp.static_veto(s, seq, r["goal_cell"], cfg):
            disp = sum(math.dist(s.get(o).xy_m, s.lattice.center(d)) for o, d in seq)
            rows.append((len(seq), disp, c["query"]))
    return min(rows)[2] if rows else "ABSTAIN"


def freeze() -> None:
    if (OUT / "predictions.json").exists():
        raise SystemExit("P10d predictions already frozen")
    if Path("runs/p10b/UNSEALED").exists():
        raise SystemExit("blind labels already unsealed: no blind freeze possible")
    cfg, P = cf.load_config("config.toml"), plans()
    sets = {"validation": [r for r in pc.prepared("trainval") if r["split"] == "validation"], "blind": pc.prepared("blind"), "h9": pc.prepared("h9")}
    from scene import scene_from_dict
    veto = {}
    for sp, rs in sets.items():
        for r in rs:
            k = r["audit"].get("variant_of", r["key"])
            veto[(sp, r["key"])] = frozenset(c["query"] for c in r["cands"] if c["seq"] and pp.static_veto(
                scene_from_dict(P[k]["scene"]), [(r["ids"][i], l3.cell(d)) for i, d in c["seq"]], P[k]["goal_cell"], cfg))
    calls, snap = pp.instrument(), {"arms": {}, "d": {}, "baselines": {}}
    for f in sorted((pc.OUT / "runs").glob("*.json")):
        x = _rd(f)
        if x["arm"] in ("D", "D-pairless"):
            continue
        m, keyed = pc.load(x["name"])[0], pc.ARMS[x["arm"]][2]
        ctx = (lambda: ufk.installed()) if keyed else contextlib.nullcontext
        v0 = (lambda rs: pc._v0(copy.deepcopy(rs))) if x["arm"] == "B-V0" else (lambda rs: rs)
        with ctx():
            vev = [l3.root_eval9(m, r, _mode(x))["root"] for r in v0(sets["validation"])]
        tau = __import__("p7_eval").fit_temperature(vev)
        thr = choose_threshold(vev, tau)
        per = {}
        for split, rs in sets.items():
            for r in v0(rs):
                with ctx():
                    pm = pe.marginals(m, r, x["arm"] not in ("A", "SF"))
                sel = pp.top1(m, r, tau, thr, veto[(split, r["key"])], keyed)
                per.setdefault(split, {})[r["key"]] = {"marginals": pm, "top1": sel["choice"], "admitted": sel["admitted"]}
        snap["arms"][x["name"]] = {"arm": x["arm"], "seed": x["seed"], "regime": x["regime"], "tau": tau, "threshold": thr, "roots": per,
                                   "checkpoint_sha256": x["checkpoint_sha256"]}
    dsel = {}
    for f in sorted((pc.OUT / "runs").glob("*.json")):
        x = _rd(f)
        if x["arm"] not in ("D", "D-pairless"):
            continue
        m = db.ModelD(pair=x["pair"]).double()
        m.load_state_dict(torch.load(pc.OUT / "ckpt" / f"{x['name']}.pt", weights_only=True))
        m.eval()
        snap["d"][x["name"]] = {k: v for k, v in x.items() if k in ("arm", "seed", "lambda_B", "regime", "val_nll", "checkpoint_sha256")}
        snap["d"][x["name"]]["roots"] = {sp: {r["key"]: db.predict(m, r["R"], (r.get("d") or {}).get("pool", []))
                                              for r in rs if (r.get("d") or {}).get("pool") is not None} for sp, rs in sets.items()}
        dsel.setdefault((x["arm"], x["lambda_B"]), []).append(x["val_nll"])
    for arm in ("D", "D-pairless"):                       # lambda_B by mean validation NLL; threshold by validation macro F1
        lam = min((v, k[1]) for k, v in ((k, np.mean(v)) for k, v in dsel.items()) if k[0] == arm)[1] if any(k[0] == arm for k in dsel) else None
        snap["baselines"][f"{arm}_lambda_B"] = lam
        snap["baselines"][f"{arm}_threshold"] = d_threshold(snap, sets["validation"], arm, lam) if lam is not None else None
    for split, rs in sets.items():
        for r in rs:
            k = r["audit"].get("variant_of", r["key"])
            snap["baselines"].setdefault("heuristic", {}).setdefault(split, {})[r["key"]] = db.heuristic(r["R"], (r.get("d") or {}).get("pool", []))
            snap["baselines"].setdefault("geometry_top1", {}).setdefault(split, {})[r["key"]] = geometry_choice(P[k], r, cfg)
    phys = dict(calls)
    if sum(phys.values()):
        raise RuntimeError(f"physical call during learned prediction / analytic baselines: {phys}")
    snap |= {"version": "P10D_PREDICTIONS_V1", "prereg": _rd("runs/p10b/p10b0/P10B_PREREG.json")["hash"], "physical_calls": phys,
             "datasets": {n: l3.sha(pc.OUT / f"dataset_{n}.pt") for n in ("trainval", "blind", "h9")}}
    snap["hash"] = cf.stable_hash(snap)
    OUT.mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "predictions.json", snap)


def d_threshold(snap, val, arm, lam):
    best = None
    for thr in (0.3, 0.4, 0.5, 0.6, 0.7):
        f1 = []
        for name, d in snap["d"].items():
            if d["arm"] != arm or d["lambda_B"] != lam:
                continue
            for r in val:
                gt = (r.get("d") or {}).get("mask")
                if gt is None:
                    continue
                p = d["roots"]["validation"][r["key"]]
                pred = [int(p["p_feasible"] < 0.5 and v >= thr) for v in p["marg"]]
                f1.append(f1_score(gt, pred))
        key = (-float(np.mean(f1)) if f1 else 0.0, thr)
        best = min(best, key) if best else key
    return best[1]


def f1_score(gt, pred):
    tp = sum(a and b for a, b in zip(gt, pred))
    if not any(gt) and not any(pred):
        return 1.0
    return 2 * tp / max(1, sum(gt) + sum(pred))


def unseal() -> None:
    """Only after predictions.json is committed (its hash must be in HEAD)."""
    import subprocess
    h = _rd(OUT / "predictions.json")["hash"]
    tracked = subprocess.run(["git", "show", "HEAD:runs/p10d/predictions.json"], capture_output=True, text=True).stdout
    if not tracked or json.loads(tracked)["hash"] != h:
        raise SystemExit("frozen predictions are not committed in HEAD: refusing to unseal")
    Path("runs/p10b/UNSEALED").write_text(f"{h}\n")


# ------------------------------------------------------------------ P10d executed episodes (blind roots only; charged to 'execution')
def gm_rank(cfg, robot, r, k=5) -> list:
    """G+M (planner-assisted reference): p8_select.gm_rank's rule on the P10b catalogue (pose screens + LINK0 sector claim)."""
    import p8_select as psel
    return psel.gm_rank(cfg, robot, {"roots": {"x": {**r, "queries": {"pool": [c for c in r["items"] if c["kind"] == "ORDERED_TEACHER"]}}}}, "x", k)


def jobs(snap, P, cfg, robot) -> list:
    """Strict TOP1 jobs keyed by the physical choice (identical choices -> one episode) + G+M (ranked <= 5) + uniform."""
    seqs = {c["query"]: c["sequence"] for r in P.values() for c in r["items"] if c["kind"] == "ORDERED_TEACHER"}
    out, arms = {}, {}
    blind = sorted(k for k, r in P.items() if r["split"] == "blind")
    for k in blind:
        choices = {f"{n}": a["roots"]["blind"][k]["top1"] for n, a in snap["arms"].items() if a["regime"] == "full" and a["arm"] in ("A", "SF", "K-B", "K-C")}
        choices["geometry"] = snap["baselines"]["geometry_top1"]["blind"][k]
        cand = sorted(c["query"] for c in P[k]["items"] if c["kind"] == "ORDERED_TEACHER")
        choices["uniform"] = cand[int(np.random.default_rng(int(cf.stable_hash(k)[7:15], 16)).integers(len(cand)))]
        for arm, ch in choices.items():
            key = f"{k}::{ch}"
            arms.setdefault(key, []).append(arm)
            if ch not in ("ABSTAIN", "NOT_RUN"):
                out.setdefault(key, {"arm": "c" + cf.stable_hash(key)[7:19], "root": k, "ranked": [seqs[ch]], "choice": ch})
        out[f"{k}::GM"] = {"arm": "gm" + cf.stable_hash(k)[7:17], "root": k, "ranked": gm_rank(cfg, robot, P[k]), "choice": "GM"}
        arms.setdefault(f"{k}::GM", []).append("G+M")
    return list(out.values()), arms


def episodes(cap_cpu=14000.0) -> None:
    import geometry as ge
    import p7_oneshot as po
    snap, P = _rd(OUT / "predictions.json"), plans()
    cfg = cf.load_config("config.toml")
    js, arms = jobs(snap, P, cfg, ge.make_robot(cfg))
    cf.write_json(OUT / "episode_jobs.json", {"jobs": js, "arms": arms, "hash": cf.stable_hash(js), "predictions": snap["hash"]})
    po.CAPS.update({"cpu_s": cap_cpu, "wall_s": 60000.0, "episode_cpu_s": 180.0, "episode_wall_s": 600.0, "max_episodes": 10 ** 6, "max_rank": 5})
    for b in pc.batches():
        bj = [j for j in js if j["root"] in _rd(b / "plan.json")["roots"]]
        po.run_episodes(OUT / "episodes" / b.name, b / "plan.json", bj, lambda m: open(OUT / "episodes.log", "a").write(m + "\n"))


def episodes_all() -> None:
    episodes()
    (OUT / "EPISODES_DONE").write_text("done\n")


def episodes_batch(name, cap_cpu=4000.0) -> None:
    """One batch's episodes (separate ledger per batch dir, so batches can run in parallel without a shared file)."""
    import p7_oneshot as po
    js = _rd(OUT / "episode_jobs.json")["jobs"]
    b = Path("runs/p10b") / name
    po.CAPS.update({"cpu_s": cap_cpu, "wall_s": 60000.0, "episode_cpu_s": 180.0, "episode_wall_s": 600.0, "max_episodes": 10 ** 6, "max_rank": 5})
    bj = [j for j in js if j["root"] in _rd(b / "plan.json")["roots"]]
    po.run_episodes(OUT / "episodes" / name, b / "plan.json", bj, lambda m: open(OUT / f"episodes_{name}.log", "a").write(m + "\n"))
    (OUT / f"EPISODES_DONE_{name}").write_text("done\n")
