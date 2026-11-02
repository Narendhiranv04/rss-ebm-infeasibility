"""P12: ledger (runs/p12/guards, contract caps, one process per job), read-only audit + prereg draft + power (P12-0), prospective
F3 / F5 statistics (P12-4), photometric controls and D attribution (P12-6). P10 / P11 artifacts are read-only."""
from __future__ import annotations

import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import config as cf

OUT = Path("runs/p12")
CAPS = {"audit": 4000.0, "dev": 5000.0, "teacher": 40000.0, "execution": 12000.0, "secondary": 5000.0, "regression": 2000.0}
JOB = {"cpu_s": 3000.0, "wall_s": 9000.0}
FROZEN_ARMS = ("K-B", "K-C", "A", "SF", "B-F3", "B-F5s", "B-F5q", "B-F2", "B-F4", "B-V0", "B-U")


def _rd(p):
    return json.loads(Path(p).read_text())


def spent(cat=None) -> tuple:
    gs = [_rd(p) for p in (OUT / "guards").glob("*.json")]
    gs = [g for g in gs if cat is None or g["category"] == cat]
    return sum(g["cpu_s"] for g in gs), sum(g["wall_s"] for g in gs)


def job(name, call, cpu, module="p12_closure", category="audit", wall=None) -> dict:
    """Guarded job (process-tree CPU of THIS child only), charged to one P12 category; refuses past the category cap."""
    import teacher
    (OUT / "guards").mkdir(parents=True, exist_ok=True)
    left = CAPS[category] - spent(category)[0]
    cpu = min(cpu, JOB["cpu_s"], left)
    if cpu < 5:
        raise SystemExit(f"{category} budget exhausted")
    g = teacher.guard([sys.executable, "-c", f"import torch; torch.set_num_threads(1); import {module}; {module}.{call}"], cpu,
                      wall or JOB["wall_s"], OUT / f"heartbeat_{name}.txt", 15.0, 0.25, Path("."))
    n = 1
    while (OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json").exists():
        n += 1
    rec = {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | \
          {"cpu_cap_s": cpu, "call": f"{module}.{call}", "category": category, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cf.write_json(OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json", rec)
    (OUT / f"heartbeat_{name}.txt").unlink(missing_ok=True)
    print(json.dumps({k: rec[k] for k in ("stop", "returncode", "cpu_s", "wall_s")} | {category: round(spent(category)[0], 1)}))
    return rec


def sha(p) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ------------------------------------------------------------------ P12-0: audit + power
def audit() -> None:
    import subprocess
    sh = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    snap = _rd("runs/p10d/predictions.json")
    A = {"head": sh("rev-parse", "HEAD"), "origin": sh("rev-parse", "origin/main"), "tracked_changes": sh("status", "--short", "--untracked-files=no").splitlines(),
         "snapshot_hash_ok": cf.stable_hash({k: v for k, v in snap.items() if k != "hash"}) == snap["hash"], "checkpoints": {}}
    for arm in FROZEN_ARMS:
        for s in (0, 1, 2):
            n = f"F-{arm}-s{s}"
            x = _rd(f"runs/p10c/runs/{n}.json")
            a = snap["arms"][n]
            A["checkpoints"][n] = {"sha_ok": sha(f"runs/p10c/ckpt/{n}.pt").removeprefix("sha256:") == x["checkpoint_sha256"].removeprefix("sha256:"),
                                   "snapshot_sha_match": a["checkpoint_sha256"] == x["checkpoint_sha256"], "tau": a["tau"], "threshold": a["threshold"],
                                   "params": x["params"], "regime": x["regime"], "arm": x["arm"]}
    A["d_frozen"] = {k: snap["baselines"][k] for k in ("D_lambda_B", "D_threshold", "D-pairless_lambda_B", "D-pairless_threshold")}
    A["p11_gates"] = {"final": _rd("runs/p11/P11_FINAL_MANIFEST.json")["gates"]}
    A["all_checkpoints_ok"] = all(v["sha_ok"] and v["snapshot_sha_match"] for v in A["checkpoints"].values())
    (OUT / "p12_0").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "p12_0" / "AUDIT.json", A)


def power(sims=4000, seed=0) -> None:
    """Sensitivity of the two-sided exact sign test (Holm over 2: first test at 0.025) from the DISCLOSED P11 nine-root H1 deltas
    (POST-HOC source, used only to plan N): bootstrap per-root deltas, plus a pessimistic shrunken-effect scenario."""
    from scipy.stats import binomtest
    be = _rd("runs/p10d/BLIND_EVAL.json")["per_root"]["H1"]
    rng, out = np.random.default_rng(seed), {}
    for arm in ("B-F3", "B-F5s"):
        d = np.array([be["K-B"][k] - be[arm][k] for k in sorted(set(be["K-B"]) & set(be[arm]))])   # positive = full B better
        for label, pool in (("observed", d), ("shrunk_50pct_with_25pct_reversals", np.concatenate([d * 0.5, -d[: max(1, len(d) // 4)] * 0.5]))):
            for n in (16, 24, 32, 40, 48):
                hits = 0
                for _ in range(sims):
                    x = rng.choice(pool, n)
                    k, m = int((x > 0).sum()), int((x != 0).sum())
                    hits += m > 0 and binomtest(k, m).pvalue < 0.025
                out[f"{arm}|{label}|N={n}"] = hits / sims
        out[f"{arm}|observed_deltas"] = [float(v) for v in d]
    cf.write_json(OUT / "p12_0" / "POWER.json", out)


# ------------------------------------------------------------------ P12-2: roster, plans, records, frozen predictions (before ANY confirmation label)
HOLD = OUT / "holdout"
ROSTER_SEED0 = 2026140001


def roster(n=80) -> list:
    import p12_families as pf
    return [{"index": i, "family": pf.FAMS[i % 2], "split": "blind", "seed": ROSTER_SEED0 + i, "wave": 0, "batch": 1 + i // 20} for i in range(n)]


def build_batches(n=80) -> None:
    import p10b_dataset as pd
    import p12_families  # noqa: F401  (registers the family dispatch)
    for b in sorted({s["batch"] for s in roster(n)}):
        d = HOLD / f"batch{b}"
        if not (d / "plan.json").exists():
            pd.build_plan(d, [s for s in roster(n) if s["batch"] == b], max_attempts=400, root_cpu=120.0)


def plans():
    return {k: r for b in sorted(HOLD.glob("batch*/plan.json")) for k, r in _rd(b)["roots"].items()}


def holdout_records() -> None:
    import torch
    import learn_data as ld
    import p10c_train_eval as pc
    cfg, dino, recs = cf.load_config("config.toml"), ld.dino(), []
    for k, r in plans().items():
        ed = [it for it in r["items"] if it.get("edit_kind") in ("A", "B", "A+B")]
        sq = [it for it in r["items"] if it["kind"] == "ORDERED_TEACHER"]
        rec, _ = pc._record(cfg, dino, k, r, {}, "blind", "P12", ed, sq)
        ids = rec["audit"]["ids"]
        rec["audit"]["family"] = r["family"]
        rec["d"] = {"pool": [ids.index(o) for o in r["parking"]], "roles": [{v: kk for kk, v in r["ids"].items()}[o] for o in r["parking"]]}
        recs.append(rec)
    torch.save({"version": "P12_RECORDS_V1", "roots": recs}, HOLD / "records.pt")


def freeze_predictions() -> None:
    """Frozen per-root marginals, kappa_hat, TOP1, ranked rows for all 33 frozen checkpoints; D / heuristic / geometry. 0 physical calls."""
    import contextlib
    import copy
    import torch
    import abc7_train as t7
    import d_blockers as db
    import p10_eval as pe
    import p10_protocol as pp
    import p10c_train_eval as pc
    import p10d_execute as px
    import p9_l3 as l3
    import unique_factor_keyed as ufk
    from scene import scene_from_dict
    if (HOLD / "PREDICTIONS_FROZEN.json").exists() or (HOLD / "UNSEALED").exists():
        raise SystemExit("predictions already frozen or labels unsealed")
    if any((b.parent / "static_admission.json").exists() for b in HOLD.glob("batch*/plan.json")):
        raise SystemExit("teacher outcomes exist: no blind freeze possible")
    cfg, P, snap = cf.load_config("config.toml"), plans(), _rd("runs/p10d/predictions.json")
    ds = torch.load(HOLD / "records.pt", weights_only=False)
    rs = t7.prepare(ds)
    for r, rec in zip(rs, ds["roots"]):
        r["ids"], r["audit"], r["d"] = rec["audit"]["ids"], rec["audit"], rec["d"]
    veto = {r["key"]: frozenset(c["query"] for c in r["cands"] if c["seq"] and pp.static_veto(
        scene_from_dict(P[r["key"]]["scene"]), [(r["ids"][i], l3.cell(d)) for i, d in c["seq"]], P[r["key"]]["goal_cell"], cfg)) for r in rs}
    calls, out = pp.instrument(), {"arms": {}, "d": {}, "baselines": {}}
    for arm in FROZEN_ARMS:
        for s in (0, 1, 2):
            n = f"F-{arm}-s{s}"
            m, a = pc.load(n)[0], snap["arms"][n]
            keyed = pc.ARMS[arm][2]
            ctx = ufk.installed if keyed else contextlib.nullcontext
            per = {}
            for r in rs:
                rr = pc._v0([copy.deepcopy(r)])[0] if arm == "B-V0" else r
                try:
                    with ctx():
                        pm = pe.marginals(m, rr, arm not in ("A", "SF"))
                        rows = l3.root_eval9(m, rr, "unique" if arm not in ("A", "SF") else "factorized")["root"]["cands"]
                    sel = pp.top1(m, rr, a["tau"], a["threshold"], veto[r["key"]], keyed)
                    per[r["key"]] = {"marginals": pm, "top1": sel["choice"], "admitted": sel["admitted"],
                                     "rows": [{k: x[k] for k in ("query", "m", "u", "J_pred", "C_d", "energy")} for x in rows]}
                except Exception as e:                                   # scope / identity failure = coverage failure, never a dropped root
                    per[r["key"]] = {"coverage_failure": f"{type(e).__name__}: {e}"[:300]}
            out["arms"][n] = {"arm": arm, "seed": s, "tau": a["tau"], "threshold": a["threshold"], "checkpoint_sha256": a["checkpoint_sha256"], "roots": per}
    lam = {x: snap["baselines"][f"{x}_lambda_B"] for x in ("D", "D-pairless")}
    for n, x in snap["d"].items():
        if x["lambda_B"] != lam[x["arm"]]:
            continue
        m = db.ModelD(pair=x["arm"] == "D").double()
        m.load_state_dict(torch.load(f"runs/p10c/ckpt/{n}.pt", weights_only=True))
        m.eval()
        out["d"][n] = {"arm": x["arm"], "seed": x["seed"], "lambda_B": x["lambda_B"], "roots": {r["key"]: db.predict(m, r["R"], r["d"]["pool"]) for r in rs}}
    out["baselines"] = {k: snap["baselines"][k] for k in ("D_lambda_B", "D_threshold", "D-pairless_lambda_B", "D-pairless_threshold")} | \
        {"heuristic": {r["key"]: db.heuristic(r["R"], r["d"]["pool"]) for r in rs}, "geometry_top1": {r["key"]: px.geometry_choice(P[r["key"]], r, cfg) for r in rs},
         "static_veto": {k: sorted(v) for k, v in veto.items()}}
    if sum(calls.values()):
        raise RuntimeError(f"physical calls during frozen prediction: {dict(calls)}")
    out |= {"version": "P12_PREDICTIONS_V1", "physical_calls": dict(calls), "records_sha256": sha(HOLD / "records.pt"),
            "plans": {str(b): _rd(b)["plan_hash"] for b in sorted(HOLD.glob("batch*/plan.json"))}}
    out["hash"] = cf.stable_hash(out)
    cf.write_json(HOLD / "PREDICTIONS_FROZEN.json", out)


def scene_manifest() -> None:
    P = plans()
    man = {"roots": {k: {"family": r["family"], "seed": r["seed"], "batch": r["batch"], "scene_hash": r["scene_hash"], "fingerprint": r["fingerprint"],
                         "construction": r["construction"], "role_cells": r["role_cells"], "fixed": r["fixed"], "stratum": r["stratum"],
                         "queries": sorted(it["query"] for it in r["items"])} for k, r in P.items()},
           "unavailable": [u for b in sorted(HOLD.glob("batch*/plan.json")) for u in _rd(b)["unavailable"]],
           "attempts": {str(b): len(_rd(b)["attempts"]) for b in sorted(HOLD.glob("batch*/plan.json"))}}
    man["hash"] = cf.stable_hash(man)
    cf.write_json(HOLD / "SCENE_MANIFEST.json", man)


def sealed_records(batch) -> dict:
    """Adjudicated P12 holdout records; refuses until runs/p12/holdout/UNSEALED exists (written only after the frozen-prediction commit)."""
    import p6_run as p6r
    import operational_teacher as ot
    if not (HOLD / "UNSEALED").exists():
        raise SystemExit("P12 holdout labels are sealed")
    out = HOLD / batch
    plan, static, pol = _rd(out / "plan.json"), _rd(out / "static_admission.json")["certificates"], ot.load_policy("teacher_opneg.toml")
    res = {}
    for k in plan["roots"]:
        pf = out / "sealed" / f"{k}.json"
        done = _rd(pf)["done"] if pf.exists() else {}
        for q in plan["queries"]:
            if q["root"] == k:
                res[q["query"]] = p6r.judge(static, pol, q, done.get(q["query"]))
    return res


def unseal() -> None:
    import subprocess
    h = _rd(HOLD / "PREDICTIONS_FROZEN.json")["hash"]
    remote = subprocess.run(["git", "show", "origin/main:runs/p12/holdout/PREDICTIONS_FROZEN.json"], capture_output=True, text=True).stdout
    if not remote or json.loads(remote)["hash"] != h:
        raise SystemExit("frozen predictions are not on origin/main: refusing to unseal")
    (HOLD / "UNSEALED").write_text(h + "\n")


# ------------------------------------------------------------------ P12-4: prospective F3 / F5 tests and secondary endpoints
def _boot(d, n=2000, seed=0):
    if len(d) < 2:
        return [None, None]
    rng, a = np.random.default_rng(seed), np.asarray(d, float)
    ms = [a[rng.integers(0, len(a), len(a))].mean() for _ in range(n)]
    return [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]


def _sign(d):
    from scipy.stats import binomtest
    nz = [x for x in d if x != 0]
    k = sum(x > 0 for x in nz)
    return float(binomtest(k, len(nz)).pvalue) if nz else 1.0, k, len(nz) - k, len(d) - len(nz)


def _holm(ps):
    out, run = {}, 0.0
    for i, k in enumerate(sorted(ps, key=ps.get)):
        run = max(run, min(1.0, (len(ps) - i) * ps[k]))
        out[k] = run
    return out


def factor_tests() -> None:
    snap = _rd(HOLD / "PREDICTIONS_FROZEN.json")
    P, rec = plans(), {}
    for b in sorted(HOLD.glob("batch*/plan.json")):
        rec |= sealed_records(b.parent.name)
    V, C = "VERIFIED_FEASIBLE", "CERTIFIED_INFEASIBLE"
    four = {}
    for k, r in P.items():
        t = {it["token"]: it["query"] for it in r["items"]}
        qs = [t[""], t["edit:A"], t["edit:B"], t["edit:A+B"]]
        st = [rec.get(q, {}).get("teacher_status") for q in qs]
        if all(s in (V, C) for s in st):
            y = [int(s == V) for s in st]
            four[k] = {"q": qs, "kappa": y[3] - y[1] - y[2] + y[0], "y": y}
    def khat(arm, k, s=None):
        vals = []
        for n, a in snap["arms"].items():
            if a["arm"] == arm and (s is None or a["seed"] == s):
                m = a["roots"][k].get("marginals")
                if m is None or any(q not in m for q in four[k]["q"]):
                    return None
                q0, qa, qb, qab = four[k]["q"]
                vals.append(m[qab] - m[qa] - m[qb] + m[q0])
        return vals
    err = {arm: {k: float(np.mean([abs(v - four[k]["kappa"]) for v in khat(arm, k)])) for k in four if khat(arm, k)} for arm in FROZEN_ARMS}
    def contrast(abl):
        ks = sorted(set(err["K-B"]) & set(err[abl]))
        d = [err[abl][k] - err["K-B"][k] for k in ks]
        p, pos, neg, ties = _sign(d)
        seeds = {s: int(sum((float(np.mean([abs(v - four[k]["kappa"]) for v in khat(abl, k, s)])) - float(np.mean([abs(v - four[k]["kappa"]) for v in khat("K-B", k, s)]))) > 0 for k in ks)) for s in (0, 1, 2)}
        fam = {f: float(np.mean([x for k, x in zip(ks, d) if P[k]["family"] == f])) for f in sorted({P[k]["family"] for k in ks}) if any(P[k]["family"] == f for k in ks)}
        return {"roots": len(d), "mean_delta": float(np.mean(d)) if d else None, "ci95": _boot(d), "p_raw": p, "full_B_better": pos, "ablation_better": neg,
                "ties": ties, "effect_size_d": float(np.mean(d) / np.std(d, ddof=1)) if len(d) > 2 and np.std(d) > 0 else None,
                "full_B_better_roots_per_seed": seeds, "per_family_mean_delta": fam}
    prim = {"F3": contrast("B-F3"), "F5s": contrast("B-F5s")}
    h = _holm({k: v["p_raw"] for k, v in prim.items()})
    for k in prim:
        prim[k]["p_holm"] = h[k]
        prim[k]["status"] = ("UNDERPOWERED/NOT_EVALUABLE (< 24 proof-grounded roots)" if prim[k]["roots"] < 24 else
                             "CONFIRMED" if h[k] < 0.05 and prim[k]["mean_delta"] > 0 else "NOT_CONFIRMED")
    sec = {a: contrast(a) for a in ("B-F5q", "B-F2", "B-F4", "B-V0", "B-U", "K-C", "A", "SF")}
    hs = _holm({k: v["p_raw"] for k, v in sec.items()})
    for k in sec:
        sec[k]["p_holm_secondary_family"] = hs[k]
    nll = {}
    for arm in FROZEN_ARMS:
        v = []
        for k in four:
            for n, a in snap["arms"].items():
                if a["arm"] == arm and a["roots"][k].get("marginals"):
                    for q, y in zip(four[k]["q"], four[k]["y"]):
                        p = min(max(a["roots"][k]["marginals"][q], 1e-9), 1 - 1e-9)
                        v.append(-math.log(p if y else 1 - p))
        nll[arm] = float(np.mean(v)) if v else None
    res = {"cohort": {"proof_grounded_four_tuples": len(four), "kappa_classes": dict(collections.Counter(x["kappa"] for x in four.values())),
                      "by_family": dict(collections.Counter(P[k]["family"] for k in four)), "roots_total": len(P)},
           "primary": prim, "secondary": sec, "absolute_kappa_error": {a: float(np.mean(list(v.values()))) if v else None for a, v in err.items()},
           "four_bit_scene_nll": nll, "per_root_kappa": {k: v["kappa"] for k, v in four.items()}}
    (OUT / "p12_4").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "p12_4" / "FACTOR_TESTS.json", res)


# ------------------------------------------------------------------ P12-6: realistic photometric controls (POST-HOC on P10 roots; masks / geometry fixed)
PHOTO = {"G1_brightness_contrast": {"gain": 1.12, "contrast": 0.88, "wb": (1.0, 1.0, 1.0), "desat": 0.0},
         "G2_whitebalance_desat": {"gain": 1.0, "contrast": 1.0, "wb": (1.06, 1.0, 0.92), "desat": 0.30},
         "G3_mild_combined": {"gain": 0.92, "contrast": 1.08, "wb": (0.96, 1.0, 1.05), "desat": 0.15}}


def photometric(rgb, p):
    """Fixed per-variant recipe on the WHOLE image (texture-preserving): gain, contrast about mid-grey, white balance, desaturation."""
    x = rgb.astype(np.float32) / 255.0
    x = (x * p["gain"] - 0.5) * p["contrast"] + 0.5
    x = x * np.asarray(p["wb"], np.float32)
    g = x.mean(axis=2, keepdims=True)
    x = x * (1 - p["desat"]) + g * p["desat"]
    return (np.clip(x, 0, 1) * 255).round().astype(np.uint8)


def photometric_records() -> None:
    import copy
    import torch
    import data
    import learn_data as ld
    import p10d_execute as px
    from scene import scene_from_dict
    cfg, dino, P, out = cf.load_config("config.toml"), ld.dino(), px.plans(), []
    for n in ("trainval", "blind"):
        for rec in torch.load(f"runs/p10c/dataset_{n}.pt", weights_only=False)["roots"]:
            if rec["audit"]["split"] not in ("validation", "blind"):
                continue
            k = rec["audit"]["root"]
            q = data.make_query(cfg, k, scene_from_dict(P[k]["scene"]), "t0", P[k]["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
            a = data.appearance_anchor(q, ld.IMAGE_PX)
            for name, pr in PHOTO.items():
                v = copy.deepcopy(rec)
                for c in v["candidates"]:
                    c["privileged"] |= {"y": None, "source": None, "z": [None] * len(c["privileged"]["z"]), "cost": None}
                for e in v["edits"]:
                    e["privileged"] |= {"y": None, "source": None}
                v["inputs"]["visual"] = ld.visual_features(dino[0], dino[1], dino[2], {**a, "rgb": photometric(a["rgb"], pr)})
                v["audit"] |= {"photometric": name, "variant_of": k, "split": rec["audit"]["split"]}
                out.append(v)
    (OUT / "p12_6").mkdir(parents=True, exist_ok=True)
    torch.save({"version": "P12_PHOTOMETRIC_V1", "recipes": PHOTO, "roots": out}, OUT / "p12_6" / "photometric_records.pt")


def photometric_eval() -> None:
    """Same metrics as P11 H9 (per-root |dp|, crossings, TOP1 change, kappa shift) for every full-regime frozen checkpoint; 0 physical calls."""
    import contextlib
    import copy
    import torch
    import abc7_train as t7
    import p10_eval as pe
    import p10_protocol as pp
    import p10c_train_eval as pc
    import p10d_execute as px
    import unique_factor_keyed as ufk
    snap, P = _rd("runs/p10d/predictions.json"), px.plans()
    ds = torch.load(OUT / "p12_6" / "photometric_records.pt", weights_only=False)
    rs = t7.prepare(ds)
    for r, rec in zip(rs, ds["roots"]):
        r["ids"], r["audit"] = rec["audit"]["ids"], rec["audit"]
    tok = {k: {c["token"]: c["query"] for c in P[k]["items"]} for k in P}
    calls, res = pp.instrument(), collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
    for n, a in snap["arms"].items():
        if a["regime"] != "full":
            continue
        m, keyed = pc.load(n)[0], pc.ARMS[a["arm"]][2]
        for r in rs:
            rr = pc._v0([copy.deepcopy(r)])[0] if a["arm"] == "B-V0" else r
            with ufk.installed() if keyed else contextlib.nullcontext():
                pm = pe.marginals(m, rr, a["arm"] not in ("A", "SF"))
            ch = pp.top1(m, rr, a["tau"], a["threshold"], frozenset(), keyed)["choice"]
            b, v, o = r["audit"]["variant_of"], r["audit"]["photometric"], a["roots"][r["audit"]["split"]][r["audit"]["variant_of"]]
            om = o["marginals"]
            d = [abs(pm[q] - p) for q, p in om.items() if q in pm]
            cell = res[a["arm"]][v]
            cell["dp"].append((b, float(np.mean(d))))
            cell["cross"].append((b, float(np.mean([(pm[q] >= 0.5) != (p >= 0.5) for q, p in om.items() if q in pm]))))
            cell["top1"].append((b, float(ch != o["top1"])))
            t = tok[b]
            if P[b]["family"] == "H1":
                kk = lambda mm: mm[t["edit:A+B"]] - mm[t["edit:A"]] - mm[t["edit:B"]] + mm[t[""]]  # noqa: E731
                cell["kappa"].append((b, abs(kk(pm) - kk(om))))
    if sum(calls.values()):
        raise RuntimeError(f"physical calls: {dict(calls)}")
    out = {}
    for arm, vv in res.items():
        for v, cell in vv.items():
            for metric, lst in cell.items():
                per = collections.defaultdict(list)
                for b, x in lst:
                    per[b].append(x)
                vals = [float(np.mean(x)) for x in per.values()]
                out[f"{arm}|{v}|{metric}"] = {"roots": len(vals), "mean": float(np.mean(vals)), "ci95": _boot(vals)}
    severe = _rd("runs/p11/h9/H9_RESULTS.json")["by_arm"]
    out["_severe_P11_H9"] = {a: {m: severe[a][f"{m}|both"]["mean"] for m in ("mean_abs_dp", "top1_changed", "kappa_abs_change")} for a in severe}
    out["_note"] = "robustness CONDITIONAL on the recorded instance masks (segmentation / detection not tested); POST-HOC on disclosed P10 roots"
    cf.write_json(OUT / "p12_6" / "PHOTOMETRIC.json", out)


# ------------------------------------------------------------------ P12-6: D semantics (feasibility, infeasible-only blockers, no-clearing-set, matched A/B/C/SF)
def _cohort(cohort):
    import p10b_dataset as pd
    import p10c_train_eval as pc
    P, rec, dl = {}, {}, {}
    if cohort == "p10b":
        snap = _rd("runs/p10d/predictions.json")
        for b in pc.batches():
            plan = _rd(b / "plan.json")
            r = pd.records(b, ("train", "validation", "blind"))
            P |= plan["roots"]
            rec |= r
            dl |= pd.d_labels(plan, r)
        roots = [k for k in P if P[k]["split"] == "blind"]
        get = lambda part, n, k: snap[part][n]["roots"]["blind"].get(k)  # noqa: E731
        heur = snap["baselines"]["heuristic"]["blind"]
    else:
        snap = _rd(HOLD / "PREDICTIONS_FROZEN.json")
        for b in sorted(HOLD.glob("batch*/plan.json")):
            plan = _rd(b)
            r = sealed_records(b.parent.name)
            P |= plan["roots"]
            rec |= r
            dl |= pd.d_labels(plan, r)
        roots = list(P)
        get = lambda part, n, k: snap[part][n]["roots"].get(k)  # noqa: E731
        heur = snap["baselines"]["heuristic"]
    return snap, P, rec, dl, roots, get, heur


def d_semantics(cohort="p10b") -> None:
    import p11_closure as q
    snap, P, rec, dl, roots, get, heur = _cohort(cohort)
    res = {"cohort": cohort, "roots": len(roots), "categories": dict(collections.Counter(dl[k]["type"] for k in roots))}
    y0 = {k: dl[k]["y0"] for k in roots if dl[k]["y0"] is not None}
    arms = collections.defaultdict(lambda: collections.defaultdict(dict))
    for n, a in snap["arms"].items():
        if a["arm"] in ("A", "SF", "K-B", "K-C") and a.get("regime", "full") == "full":
            for k in y0:
                x = get("arms", n, k) or {}
                p = (x.get("marginals") or {}).get(f"seq:{k}:empty")
                if p is not None:
                    arms[a["arm"]][k][a["seed"]] = p
    for n, x in snap["d"].items():
        arm = x["arm"]
        if x["lambda_B"] != snap["baselines"][f"{arm}_lambda_B"]:
            continue
        for k in y0:
            v = get("d", n, k)
            if v is not None:
                arms[arm][k][x["seed"]] = v["p_feasible"]
    common = sorted(set.intersection(*[set(v) for v in arms.values()])) if arms else []
    res["feasibility"] = {"cohort_N": len(common), "prevalence_feasible": float(np.mean([y0[k] for k in common])) if common else None,
                          "arms": {a: q._feas_metrics([(float(np.mean(list(d[k].values()))), y0[k]) for k in common]) for a, d in arms.items()}}
    inf = {k: dl[k] for k in roots if dl[k]["mask"] is not None and dl[k]["y0"] == 0 and dl[k]["n"] > 0}
    preds = collections.defaultdict(lambda: collections.defaultdict(list))
    for n, x in snap["d"].items():
        arm = x["arm"]
        if x["lambda_B"] != snap["baselines"][f"{arm}_lambda_B"]:
            continue
        thr = snap["baselines"][f"{arm}_threshold"]
        for k in inf:
            v = get("d", n, k)
            preds[arm][k].append([int(v["p_feasible"] < 0.5 and m >= thr) for m in v["marg"]])
    for k in inf:
        preds["geometry_heuristic"][k].append(heur[k])
    def score(arm, ks):
        return {k: {m: float(np.mean([q._set_metrics(inf[k]["mask"], pm)[m] for pm in preds[arm][k]]))
                    for m in ("precision", "recall", "f1", "exact", "jaccard", "card_err", "tp", "fp", "fn")} | {"fp_rate_irrelevant": None} for k in ks}
    strata = {"infeasible_all": list(inf), "proof_grounded": [k for k in inf if inf[k]["provenance"] == "proof_grounded"],
              "operational": [k for k in inf if inf[k]["provenance"] != "proof_grounded"],
              "no_clearing_set_in_pool": [k for k in inf if inf[k]["type"] == "NO_CLEARING_SET_WITHIN_POOL"]}
    strata |= {f"blockers={c}": [k for k in inf if inf[k]["type"] == "MINIMAL_SETS" and sum(inf[k]["mask"]) == c] for c in range(1, 5)}
    res["blockers_infeasible_only"] = {s: {a: q._agg(score(a, ks)) for a in ("D", "D-pairless", "geometry_heuristic")} for s, ks in strata.items() if ks}
    f1 = {a: {k: v["f1"] for k, v in score(a, list(inf)).items()} for a in ("D", "D-pairless", "geometry_heuristic")}
    res["paired_infeasible_only"] = {"D-D-pairless": q.paired(f1, "D", "D-pairless"), "D-heuristic": q.paired(f1, "D", "geometry_heuristic")}
    res["union_vs_necessary"] = {"alternative_set_roots": sum(len(inf[k]["sets"]) > 1 for k in inf),
                                 "union_members_not_necessary": int(sum(sum(a and not b for a, b in zip(inf[k]["mask"], inf[k]["necessary"])) for k in inf))}
    sup, nopred = [], collections.Counter()
    for k in inf:
        r = P[k]
        it = {x["token"]: x for x in r["items"]}
        e = [it.get(t) for t in ("edit:A", "edit:B", "edit:A+B")]
        if dl[k]["n"] == 2 and all(e) and all(x.get("d_subset") for x in e):
            sup.append(k)
    der = collections.defaultdict(lambda: collections.defaultdict(list))
    for n, a in snap["arms"].items():
        if a["arm"] not in ("A", "SF", "K-B", "K-C") or a.get("regime", "full") != "full":
            continue
        for k in sup:
            m, roles, it = (get("arms", n, k) or {}).get("marginals", {}), dl[k]["roles"], {x["token"]: x for x in P[k]["items"]}
            st = {frozenset(): m[f"seq:{k}:empty"]} | {frozenset(roles.index(x) for x in it[t]["d_subset"]): m[it[t]["query"]] for t in ("edit:A", "edit:B", "edit:A+B")}
            if not any(p >= 0.5 for p in st.values()):
                nopred[a["arm"]] += 1
                continue
            der[a["arm"]][k].append(q._derived_mask(st, 2))
    res["matched"] = {"matched_infeasible_N": len(sup), "unsupported_N": len(inf) - len(sup), "NO_PREDICTED_CLEARING_SET_choices": dict(nopred), "arms": {}}
    for arm in ("D", "D-pairless", "geometry_heuristic"):
        res["matched"]["arms"][arm] = q._agg({k: v for k, v in score(arm, sup).items()}) if sup else None
    for arm, d in der.items():
        res["matched"]["arms"][arm] = q._agg({k: {m: float(np.mean([q._set_metrics(inf[k]["mask"], pm)[m] for pm in d[k]]))
                                                  for m in ("precision", "recall", "f1", "exact", "jaccard", "card_err", "tp", "fp", "fn")} | {"fp_rate_irrelevant": None}
                                              for k in d}) | {"scored_roots": len(d)}
    (OUT / "p12_6").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "p12_6" / f"D_SEMANTICS_{cohort}.json", res)


# ------------------------------------------------------------------ P12-5: strict TOP1 execution on the new holdout (frozen choices) + separate G+M
def exec_jobs() -> None:
    import geometry as ge
    import p10d_execute as px
    snap, P = _rd(HOLD / "PREDICTIONS_FROZEN.json"), plans()
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    seqs = {c["query"]: c["sequence"] for r in P.values() for c in r["items"] if c["kind"] == "ORDERED_TEACHER"}
    jobs, arms = {}, collections.defaultdict(list)
    for k in sorted(P):
        ch = {n: a["roots"][k]["top1"] for n, a in snap["arms"].items() if a["arm"] in ("A", "SF", "K-B", "K-C")}
        ch["geometry"] = snap["baselines"]["geometry_top1"][k]
        for arm, c in ch.items():
            key = f"{k}::{c}"
            arms[key].append(arm)
            if c not in ("ABSTAIN", "NOT_RUN"):
                jobs.setdefault(key, {"arm": "c" + cf.stable_hash(key)[7:19], "root": k, "ranked": [seqs[c]], "choice": c})
        jobs[f"{k}::GM"] = {"arm": "gm" + cf.stable_hash(k)[7:17], "root": k, "ranked": px.gm_rank(cfg, robot, P[k]), "choice": "GM"}
        arms[f"{k}::GM"].append("G+M")
    man = {"jobs": list(jobs.values()), "arms": arms, "predictions": snap["hash"], "protocol": "strict TOP1_COMMIT (one episode per (root, repair), "
           "180 CPU-s / 600 s cap, no fallback, SAFE_RETREAT); G+M = planner-assisted, <= 5 ranked candidates"}
    man["hash"] = cf.stable_hash(man)
    (OUT / "p12_5").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "p12_5" / "EXEC_JOBS.json", man)


def exec_batch(name) -> None:
    import p7_oneshot as po
    man = _rd(OUT / "p12_5" / "EXEC_JOBS.json")
    b = HOLD / name
    po.CAPS.update({"cpu_s": 2250.0, "wall_s": 60000.0, "episode_cpu_s": 180.0, "episode_wall_s": 600.0, "max_episodes": 10 ** 6, "max_rank": 5})
    js = [j for j in man["jobs"] if j["root"] in _rd(b / "plan.json")["roots"]]     # 4 parallel batches x 2,250 <= remaining execution cap
    po.run_episodes(OUT / "p12_5" / "episodes" / name, b / "plan.json", js, lambda m: open(OUT / "p12_5" / f"episodes_{name}.log", "a").write(m + "\n"))
    (OUT / "p12_5" / f"DONE_{name}").write_text("done\n")


def exec_continue(cap=1550.0) -> None:
    """Continue NOT_RUN_BUDGET episodes (never physically attempted) in the frozen root order with the remaining execution budget, one
    sequential process; removed placeholder keys are logged. Root coverage stays outcome-independent (fixed order)."""
    import p7_oneshot as po
    man = _rd(OUT / "p12_5" / "EXEC_JOBS.json")
    removed = []
    for b in sorted(HOLD.glob("batch*/plan.json")):
        name = b.parent.name
        lp = OUT / "p12_5" / "episodes" / name / "ledger.json"
        led = _rd(lp)
        keep = {k: v for k, v in led["episodes"].items() if v.get("outcome") != "NOT_RUN_BUDGET"}
        removed += [k for k in led["episodes"] if k not in keep]
        led["episodes"] = keep
        lp.write_text(json.dumps(led, indent=1, default=float))
    cf.write_json(OUT / "p12_5" / "continuation_removed_placeholders.json", {"removed": removed})
    done_roots = collections.defaultdict(list)
    for j in man["jobs"]:
        done_roots[j["root"]].append(f"{j['arm']}__{j['root']}" not in removed)
    partial = {k for k, v in done_roots.items() if any(v) and not all(v)}
    for only in (partial, None):                                       # pass 1: complete partial roots; pass 2: frozen root order
      for b in sorted(HOLD.glob("batch*/plan.json")):
        name = b.parent.name
        led = _rd(OUT / "p12_5" / "episodes" / name / "ledger.json")
        po.CAPS.update({"cpu_s": led["cpu_s"] + max(0.0, cap - _cont_used()), "wall_s": 60000.0, "episode_cpu_s": 180.0, "episode_wall_s": 600.0,
                        "max_episodes": 10 ** 6, "max_rank": 5})
        js = sorted([j for j in man["jobs"] if j["root"] in _rd(b)["roots"] and (only is None or j["root"] in only)], key=lambda j: (j["root"], j["arm"]))
        if only is None:                                               # unfinished roots only after pass 1 (no new partial roots beyond budget)
            js = [j for j in js if f"{j['arm']}__{j['root']}" not in _rd(OUT / "p12_5" / "episodes" / name / "ledger.json")["episodes"]]
        po.run_episodes(OUT / "p12_5" / "episodes" / name, b, js, lambda m: open(OUT / "p12_5" / "episodes_continue.log", "a").write(m + "\n"))
    (OUT / "p12_5" / "CONTINUE_DONE").write_text("done\n")


def _cont_used():
    """CPU of continuation episodes so far (episodes logged in episodes_continue.log)."""
    f = OUT / "p12_5" / "episodes_continue.log"
    if not f.exists():
        return 0.0
    return sum(float(l.rsplit("(", 1)[1].split()[0]) for l in f.read_text().splitlines() if "CPU-s)" in l)


def exec_eval() -> None:
    """Strict TOP1 executed success on roots whose every frozen choice was executed (coverage set fixed by job order, outcome-independent)."""
    man, P = _rd(OUT / "p12_5" / "EXEC_JOBS.json"), plans()
    led = {}
    for f in (OUT / "p12_5" / "episodes").glob("*/ledger.json"):
        led |= {k: (v, f.parent / k) for k, v in _rd(f)["episodes"].items()}
    jm = {f"{j['root']}::{j['choice']}": j for j in man["jobs"]}
    rec = {}
    for b in sorted(HOLD.glob("batch*/plan.json")):
        rec |= sealed_records(b.parent.name)
    ok_root = {}
    for k in P:
        keys = [key for key in man["arms"] if key.startswith(k + "::") and key in jm]
        ok_root[k] = all((led.get(f"{jm[key]['arm']}__{k}") or ({},))[0].get("outcome") not in (None, "NOT_RUN_BUDGET") for key in keys)
    cov = [k for k, v in ok_root.items() if v]
    wit = [k for k in cov if any(rec.get(c["query"], {}).get("teacher_status") == "VERIFIED_FEASIBLE" for c in P[k]["items"] if c["kind"] == "ORDERED_TEACHER" and c["sequence"])]
    per, cls, cost, moves, plans_ = (collections.defaultdict(lambda: collections.defaultdict(list)) for _ in range(5))
    for key, arms in man["arms"].items():
        k, ch = key.split("::", 1)
        if k not in cov:
            continue
        e, d = (led.get(f"{jm[key]['arm']}__{k}") if key in jm else None) or (None, None)
        res = _rd(d / "result.json") if d is not None and (d / "result.json").exists() else {}
        for a in arms:
            arm = a if a in ("geometry", "G+M") else a.split("-s")[0].removeprefix("F-")
            succ = float(bool(e) and e.get("outcome") == "SUCCESS")
            per[arm][k].append(succ)
            cls[arm]["ABSTAIN" if ch in ("ABSTAIN", "NOT_RUN") else (e or {}).get("outcome")].append(1)
            cost[arm][k].append((e or {}).get("cpu_s", 0.0))
            plans_[arm][k].append(res.get("relocation_plans", 0) or 0)
            if succ:
                moves[arm][k].append(res.get("moves") or 0)
    succ = {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in per.items()}
    def contrast(a, b, ks):
        d = [succ[a][k] - succ[b][k] for k in ks if k in succ[a] and k in succ[b]]
        p, pos, neg, ties = _sign(d)
        return {"roots": len(d), "mean": float(np.mean(d)) if d else None, "ci95": _boot(d), "sign_p": p, "a_better": pos, "b_better": neg}
    out = {"coverage": {"roots_total": len(P), "roots_fully_executed": len(cov), "witness_exists_among_covered": len(wit),
                        "episodes_executed": sum(1 for v in led.values() if v[0].get("outcome") not in (None, "NOT_RUN_BUDGET")), "episodes_planned": len(man["jobs"])},
           "success_all_covered": {a: float(np.mean(list(d.values()))) for a, d in succ.items()},
           "success_witness_exists": {a: float(np.mean([d[k] for k in wit if k in d])) for a, d in succ.items()},
           "by_family": {f: {a: float(np.mean([d[k] for k in cov if P[k]["family"] == f and k in d])) for a, d in succ.items()} for f in sorted({P[k]["family"] for k in cov})},
           "contrasts_all_covered": {f"{a}-{b}": contrast(a, b, cov) for a, b in (("K-B", "A"), ("K-B", "SF"), ("K-C", "K-B"), ("K-B", "geometry"), ("K-B", "G+M"))},
           "contrasts_witness_exists": {f"{a}-{b}": contrast(a, b, wit) for a, b in (("K-B", "A"), ("K-B", "SF"), ("K-C", "K-B"), ("K-B", "geometry"))},
           "outcome_classes": {a: {o: len(v) for o, v in d.items()} for a, d in cls.items()},
           "cpu_per_root_incl_abstain": {a: float(np.mean([np.mean(v) for v in d.values()])) for a, d in cost.items()},
           "relocation_plans_per_root": {a: float(np.mean([np.mean(v) for v in d.values()])) for a, d in plans_.items()},
           "moves_on_success": {a: float(np.mean([x for v in d.values() for x in v])) if d else None for a, d in moves.items()},
           "note": "coverage set = roots whose every frozen choice was executed before the execution cap; determined by the frozen root order, not by outcomes"}
    cf.write_json(OUT / "p12_5" / "EXEC_RESULTS.json", out)


def regression() -> None:
    import subprocess
    d = OUT / "regression"
    d.mkdir(parents=True, exist_ok=True)
    (d / "countcalls.py").write_text(Path("runs/p8/audit/regression/countcalls.py").read_text())
    subprocess.run([sys.executable, "run.py", "g4c-lf-p4-regression", "--out", str(d), "--cpu", "1900", "--wall", "3400"], capture_output=True, text=True)


def yield_summary() -> None:
    P, rec = plans(), {}
    for b in sorted(HOLD.glob("batch*/plan.json")):
        rec |= sealed_records(b.parent.name)
    st = collections.Counter(r["teacher_status"] for r in rec.values())
    fam = {f: dict(collections.Counter(rec[q]["teacher_status"] for k, r in P.items() if r["family"] == f for q in (it["query"] for it in r["items"]) if q in rec))
           for f in sorted({r["family"] for r in P.values()})}
    cert = collections.Counter((r.get("static_certificate") or {}).get("kind") for r in rec.values() if r["teacher_status"] == "CERTIFIED_INFEASIBLE")
    wl = collections.Counter(min((len(it["sequence"]) for it in r["items"] if it["kind"] == "ORDERED_TEACHER" and it["sequence"] and
                                  rec.get(it["query"], {}).get("teacher_status") == "VERIFIED_FEASIBLE"), default=None) for r in P.values())
    cost = []
    for b in sorted(HOLD.glob("batch*/sealed/*.json")):
        cost.append(sum(d["unit_cpu_s"] + d["coord_cpu_s"] for d in _rd(b)["done"].values()))
    out = {"roots": len(P), "queries": len(rec), "status_counts": dict(st), "by_family": fam, "certificate_kinds": {str(k): v for k, v in cert.items()},
           "min_verified_repair_length_per_root": {str(k): v for k, v in wl.items()}, "teacher_cpu_per_root": {"mean": float(np.mean(cost)), "median": float(np.median(cost)),
           "p90": float(np.percentile(cost, 90)), "n": len(cost)}, "attempts_ledger": _rd(HOLD / "SCENE_MANIFEST.json")["attempts"]}
    cf.write_json(OUT / "p12_3" / "YIELD.json", out) if (OUT / "p12_3").mkdir(parents=True, exist_ok=True) is None else None


def order_omission() -> None:
    """Secondary: proof-grounded same-endpoint order choice + margin; omission necessity (decided omissions of VERIFIED >=2-move repairs)."""
    snap, P, rec = _rd(HOLD / "PREDICTIONS_FROZEN.json"), plans(), {}
    for b in sorted(HOLD.glob("batch*/plan.json")):
        rec |= sealed_records(b.parent.name)
    y = lambda q: rec.get(q, {}).get("binary_training_label")  # noqa: E731
    pg = lambda q: rec.get(q, {}).get("teacher_status") in ("VERIFIED_FEASIBLE", "CERTIFIED_INFEASIBLE")  # noqa: E731
    acc, mar, om = (collections.defaultdict(lambda: collections.defaultdict(list)) for _ in range(3))
    for k, r in P.items():
        seq = {tuple(map(tuple, it["sequence"])): it["query"] for it in r["items"] if it["kind"] == "ORDERED_TEACHER"}
        pairs = [(q, q2) for s, q in seq.items() for s2, q2 in seq.items() if s < s2 and sorted(s) == sorted(s2) and len(s) >= 2
                 and {y(q), y(q2)} == {0, 1} and pg(q) and pg(q2)]
        omis = [(qo, y(qo)) for s, q in seq.items() if len(s) >= 2 and y(q) == 1 for t in range(len(s))
                for qo in [seq.get(tuple(a for i, a in enumerate(s) if i != t))] if qo and y(qo) is not None]
        for n, a in snap["arms"].items():
            m = a["roots"][k].get("marginals") or {}
            for q, q2 in pairs:
                g, bd = (q, q2) if y(q) else (q2, q)
                if g in m and bd in m:
                    acc[a["arm"]][k].append(float(m[g] > m[bd]))
                    mar[a["arm"]][k].append(m[g] - m[bd])
            for qo, yo in omis:
                if qo in m:
                    om[a["arm"]][k].append(float((m[qo] < 0.5) == (yo == 0)))
    red = lambda d: {arm: {k: float(np.mean(v)) for k, v in dd.items()} for arm, dd in d.items()}  # noqa: E731
    A, M, O = red(acc), red(mar), red(om)
    def con(D, a, b):
        ks = sorted(set(D.get(a, {})) & set(D.get(b, {})))
        d = [D[a][k] - D[b][k] for k in ks]
        p, pos, neg, ties = _sign(d)
        return {"roots": len(d), "mean": float(np.mean(d)) if d else None, "ci95": _boot(d), "sign_p": p, "a_better": pos, "b_better": neg}
    out = {"order": {"roots": len(A.get("K-B", {})), "accuracy": {a: float(np.mean(list(v.values()))) for a, v in A.items()},
                     "margin": {a: float(np.mean(list(v.values()))) for a, v in M.items()},
                     "C_vs_B_accuracy": con(A, "K-C", "K-B"), "C_vs_B_margin": con(M, "K-C", "K-B"), "B_vs_F2_accuracy": con(A, "K-B", "B-F2"),
                     "B_vs_A_accuracy": con(A, "K-B", "A")},
           "omission": {"roots": len(O.get("K-B", {})), "accuracy": {a: float(np.mean(list(v.values()))) for a, v in O.items()},
                        "B_vs_SF": con(O, "K-B", "SF"), "B_vs_A": con(O, "K-B", "A"), "B_vs_F4": con(O, "K-B", "B-F4")}}
    cf.write_json(OUT / "p12_4" / "ORDER_OMISSION.json", out)


def manifest() -> None:
    """P12-7: final manifest (sources, frozen evidence, figures, regression, ledger) with a self-hash."""
    import hashlib
    import subprocess
    reg = _rd(OUT / "regression" / "regression_result.json")
    gates = {g.stem: _rd(g).get("engineering") for g in sorted(OUT.glob("P12_*_GATE.json"))}
    src = ["p12_closure.py", "p12_families.py", "p12_viz.py", "tests/test_p12.py", "p10b_scenes.py", "p10b_dataset.py", "p10c_train_eval.py",
           "p10d_execute.py", "d_blockers.py", "p7_oneshot.py"]
    files = ["runs/p12/PREREG.json", "runs/p12/holdout/PREDICTIONS_FROZEN.json", "runs/p12/holdout/SCENE_MANIFEST.json",
             "runs/p12/holdout/LABELS_MANIFEST.json", "runs/p12/holdout/UNSEALED", "runs/p12/p12_3/YIELD.json", "runs/p12/p12_4/FACTOR_TESTS.json",
             "runs/p12/p12_4/ORDER_OMISSION.json", "runs/p12/p12_5/EXEC_JOBS.json", "runs/p12/p12_5/EXEC_RESULTS.json",
             "runs/p12/p12_6/PHOTOMETRIC.json", "runs/p12/p12_6/D_SEMANTICS_p10b.json", "runs/p12/p12_6/D_SEMANTICS_holdout.json",
             "runs/p12/regression/regression_result.json"] + sorted(map(str, OUT.glob("P12_*_GATE.json")))
    figs = sorted(map(str, Path("visualizations/p12").glob("*")))
    ck = [f"runs/p10c/ckpt/{k}.pt" for k in sorted(_rd(OUT / "p12_0" / "AUDIT.json")["checkpoints"])]
    out = {"stage": "P12", "start": "b5d3f8504876585b758d6d9b02b785e9c7e1d990",
           "lock_commits": {"prereg_predictions": "caa49df6afba5497b670d3417ce4172ddcf38cb1", "labels_hash": "a2d78339"},
           "status": "PARTIAL (engineering PASS; execution 49/80 roots under cap)", "gates": gates,
           "regression": {"tests": reg.get("tests"), "replays": reg.get("replays"), "p12_tests": "6/6 (tests/test_p12.py)"},
           "spend_cpu_s": {c: round(spent(c)[0], 1) for c in CAPS}, "caps": CAPS,
           "checkpoints": {p: sha(p) for p in ck}, "sources": {p: sha(p) for p in src if Path(p).exists()},
           "files": {p: sha(p) for p in files if Path(p).exists()}, "figures": {p: sha(p) for p in figs},
           "head_at_manifest": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    out["hash"] = "sha256:" + hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    cf.write_json(OUT / "P12_FINAL_MANIFEST.json", out)
