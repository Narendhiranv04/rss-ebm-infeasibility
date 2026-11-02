"""P10a learner side: unlabelled learner records, the frozen prediction snapshot (before any P10a teacher outcome, including
static certificates), and offline scoring of the frozen predictions against the adjudicated outcomes. No teacher, IK or
planner call: records use learner-visible geometry + DINOv2 only; the TOP1 static veto is pure disc geometry."""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path

import numpy as np
import torch

import abc7_train as t7
import config as cf
import joint_likelihood as jl
import learn_data as ld
import p10_protocol as pp
import p9_l3 as l3
import unique_factor as uf
import unique_factor_keyed as ufk
import zlayout_r4 as z4
from scene import scene_from_dict

OUT = Path("runs/p10/p10a")
REC = OUT / "learner_records.pt"


def _rd(p):
    return json.loads(Path(p).read_text())


def records() -> None:
    """Learner records of the 14 frozen P10a roots with NO labels (privileged fields empty)."""
    cfg, plan = cf.load_config("config.toml"), _rd(OUT / "plan.json")
    model, mean, std = ld.dino()
    recs = []
    for k, r in plan["roots"].items():
        s = scene_from_dict(r["scene"])
        cands = [{"sequence": p["sequence"], "query": p["z_layout"]["query"].get("zR", f"seq:{k}:empty"), "z_layout": p["z_layout"]} for p in r["queries"]["pool"]]
        planned = {q["query"] for q in plan["queries"] if q["root"] == k}
        if not {c["query"] for c in cands} <= planned:
            raise ValueError(f"{k}: candidate ids differ from the frozen plan query ids")
        edits = [{**e, "query": f"scene:{k}:{z4.sig(sorted(e['edit'].items()))}"} for e in r["queries"]["scene_queries"]]
        rec = ld.root_record(cfg, model, mean, std, k, s, r["goal_cell"], cands, edits, {}, "P10A_PILOT", "P10A")
        rec["audit"] |= {"layout": z4.VERSION, "family": r["mechanism"], "stratum": r["stratum"], "lineage": k, "wave": r["wave"]}
        recs.append(rec)
    torch.save({"version": "P10A_LEARNER_RECORDS_V1", "plan_hash": plan["plan_hash"], "dino": list(ld.DINO), "roots": recs}, REC)


def checkpoints() -> list:
    """Every A / B / C / SF checkpoint available at the freeze: P10-0 runs plus the frozen P9 K-B-full#s1."""
    out = []
    for f in sorted((pp.P0 / "runs").glob("*.json")):
        x = _rd(f)
        out.append({"name": x["name"], "arm": x["arm"], "base": pp.ARMS[x["arm"]][0], "kind": x["kind"], "path": str(pp.P0 / "ckpt" / f"ckpt_{x['name']}.pt"),
                    "sha256": x["checkpoint_sha256"], "tau": x["eval"]["root"]["tau"], "threshold": x["eval"]["root"]["threshold"]})
    fz = _rd(pp.FROZEN)["checkpoint"]
    out.append({"name": "FROZEN:" + fz["name"], "arm": "K-B-full", "base": "B-full", "kind": "keyed", "path": fz["path"], "sha256": fz["sha256"],
                "tau": fz["tau"], "threshold": fz["threshold"]})
    for c in out:
        if l3.sha(c["path"]) != c["sha256"]:
            raise ValueError(f"checkpoint hash mismatch: {c['name']}")
    return out


@torch.no_grad()
def marginals(m, r, keyed) -> dict:
    """Root-prior P(z_q = 1) for every modelled query of the root (no labels)."""
    if keyed:
        return {q: float(p) for q, p in uf.prior_marginals(uf.instances(m, r)).items()}
    lg = jl.factorized_logits(m, r, t7.forward(m, r))
    return {q: float(torch.sigmoid(x)) for q, x in lg.items()}


def freeze() -> None:
    """Frozen snapshot: per checkpoint and root, all root-prior marginals, per-candidate J_pred and the TOP1 choice."""
    if (OUT / "predictions.json").exists():
        raise SystemExit("P10a predictions already frozen; refusing to overwrite")
    if (OUT / "static_admission.json").exists() or (OUT / "progress.json").exists():
        raise SystemExit("teacher outcomes already exist: predictions can no longer be frozen blind")
    cfg, plan, ds = cf.load_config("config.toml"), _rd(OUT / "plan.json"), torch.load(REC, weights_only=False)
    if ds["plan_hash"] != plan["plan_hash"]:
        raise ValueError("learner records do not belong to the frozen plan")
    rs = t7.prepare(ds)
    for r, rec in zip(rs, ds["roots"]):
        r["ids"] = rec["audit"]["ids"]
    vetoed = {r["key"]: frozenset(c["query"] for c in r["cands"] if c["seq"] and pp.static_veto(
        scene_from_dict(plan["roots"][r["key"]]["scene"]), [(r["ids"][i], l3.cell(d)) for i, d in c["seq"]], plan["roots"][r["key"]]["goal_cell"], cfg))
        for r in rs}
    calls, out = pp.instrument(), {}
    for ck in checkpoints():
        m, keyed = l3.load(ck["base"], ck["path"]), ck["kind"] == "keyed"
        per = {}
        for r in rs:
            try:                                             # a catalogue the model cannot score is NOT_RUN with its exact blocker
                with ufk.installed() if keyed else contextlib.nullcontext():
                    pm = marginals(m, r, keyed)
                sel = pp.top1(m, r, ck["tau"], ck["threshold"], vetoed[r["key"]], keyed)
                with ufk.installed() if keyed else contextlib.nullcontext():
                    rows = l3.root_eval9(m, r, "unique" if keyed else "factorized")["root"]["cands"]
                if not all(math.isfinite(v) for v in pm.values()):
                    raise FloatingPointError(f"{ck['name']} {r['key']}: nonfinite marginal")
            except (jl.IdentityError, FloatingPointError, jl.CombinatoricsError) as e:
                per[r["key"]] = {"status": "NOT_RUN", "blocker": f"{type(e).__name__}: {e}", "marginals": {}, "top1": {"choice": "NOT_RUN"}, "rows": []}
                continue
            per[r["key"]] = {"marginals": pm, "top1": sel, "rows": [{k: x[k] for k in ("query", "m", "J_pred", "C_d", "p", "u")} for x in rows]}
        out[ck["name"]] = {"checkpoint": ck, "roots": per}
    snap = {"version": "P10A_PREDICTIONS_V1", "plan_hash": plan["plan_hash"], "records_sha256": l3.sha(REC),
            "prereg_hash": _rd(pp.P0 / "P10_0_PREREG.json")["hash"], "static_veto": {k: sorted(v) for k, v in vetoed.items()},
            "physical_calls_during_prediction": dict(calls), "arms": out}
    if sum(calls.values()):
        raise RuntimeError(f"physical call during learned prediction: {dict(calls)}")
    snap["hash"] = cf.stable_hash(snap)
    cf.write_json(OUT / "predictions.json", snap)


# ------------------------------------------------------------------ offline scoring (after adjudication; predictions read-only)
def p_of(pred, ck, root, q):
    return pred["arms"][ck]["roots"][root]["marginals"].get(q)


def kappa_rows(plan, rec, pred) -> list:
    """Per root with scene kinds A / B / A+B and the empty repair: observed kappa (decided only) and each arm's kappa_hat."""
    out = []
    for k, r in plan["roots"].items():
        qs = {q["edit_kind"] or "": q["query"] for q in plan["queries"] if q["root"] == k and q["kind"] == "SCENE_TARGET_ONLY"}
        e0 = next((q["query"] for q in plan["queries"] if q["root"] == k and q["kind"] == "ORDERED_TEACHER" and not q["sequence"]), None)
        if not {"A", "B", "A+B"} <= set(qs) or e0 is None:
            continue
        ids = {"0": e0, "A": qs["A"], "B": qs["B"], "AB": qs["A+B"]}
        st = {x: rec[q]["teacher_status"] for x, q in ids.items()}
        y = {x: rec[q]["binary_training_label"] for x, q in ids.items()}
        dec = all(v is not None for v in y.values())
        kap = y["AB"] - y["A"] - y["B"] + y["0"] if dec else None
        hat = {ck: (lambda p: None if None in p.values() else p["AB"] - p["A"] - p["B"] + p["0"])(
            {x: p_of(pred, ck, k, q) for x, q in ids.items()}) for ck in pred["arms"]}
        out.append({"root": k, "mechanism": r["mechanism"], "status": st, "decided": dec, "kappa": kap,
                    "stratum": None if not dec else "proof_grounded" if all(s in ("VERIFIED_FEASIBLE", "CERTIFIED_INFEASIBLE") for s in st.values())
                    else "operational", "kappa_hat": hat})
    return out


def binary_rows(plan, rec, pred) -> list:
    """Every decided query x checkpoint: root-prior p vs observed y (common per-event metrics)."""
    out = []
    for q in plan["queries"]:
        r = rec[q["query"]]
        if r["binary_training_label"] is None:
            continue
        for ck in pred["arms"]:
            p = p_of(pred, ck, q["root"], q["query"])
            if p is not None:
                out.append({"query": q["query"], "root": q["root"], "mechanism": q["mechanism"], "ck": ck, "p": p, "y": r["binary_training_label"],
                            "source": r["label_source"]})
    return out


def metrics(rows) -> dict:
    if not rows:
        return {"n": 0}
    p, y = np.clip([x["p"] for x in rows], 1e-9, 1 - 1e-9), np.array([x["y"] for x in rows])
    pos, neg = p[y == 1], p[y == 0]
    auc = None if not len(pos) or not len(neg) else float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg]))
    bins = np.minimum((p * 10).astype(int), 9)
    ece = float(sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).sum() for b in range(10) if (bins == b).any()) / len(p))
    return {"n": len(rows), "roots": len({x["root"] for x in rows}), "nll": float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
            "accuracy": float(np.mean((p >= 0.5) == y)), "brier": float(np.mean((p - y) ** 2)), "auroc": auc, "ece": ece,
            "false_pos": int(((p >= 0.5) & (y == 0)).sum()), "false_neg": int(((p < 0.5) & (y == 1)).sum())}


def boot(diffs, n=2000, seed=0) -> list:
    """Root-cluster bootstrap 95 % CI of the mean per-root difference (one value per independent root)."""
    if len(diffs) < 2:
        return [None, None]
    rng, d = np.random.default_rng(seed), np.asarray(diffs, float)
    ms = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return [float(np.percentile(ms, 2.5)), float(np.percentile(ms, 97.5))]


def _arm(ck):
    return "FROZEN" if ck.startswith("FROZEN") else ck.split("#")[0]


def _seedmean(pred, root, q) -> dict:
    out = {}
    for ck in pred["arms"]:
        p = p_of(pred, ck, root, q)
        if p is not None:
            out.setdefault(_arm(ck), []).append(p)
    return {a: float(np.mean(v)) for a, v in out.items()}


def mechanism_rows(plan, rec, pred) -> dict:
    """Mechanism-specific decided contrasts (proof-grounded vs operational kept apart), each with every arm's seed-mean prediction."""
    import p10_collect as pc
    proof = lambda *xs: all(rec[x]["teacher_status"] in ("VERIFIED_FEASIBLE", "CERTIFIED_INFEASIBLE") for x in xs)  # noqa: E731
    y = lambda x: rec[x]["binary_training_label"]  # noqa: E731
    out = {"H3_order": [], "H4_omission": [], "H5_empty": [], "H7_edit_vs_exec": [], "H8_anchor": []}
    for k, r in plan["roots"].items():
        tok = {pc._tok(plan, q): q["query"] for q in plan["queries"] if q["root"] == k}
        m = r["mechanism"]
        if m == "H3":
            a, b = tok["B A"], tok["A B"]
            if y(a) is not None and y(b) is not None and y(a) != y(b):
                good, bad = (a, b) if y(a) else (b, a)
                pg, pb = _seedmean(pred, k, good), _seedmean(pred, k, bad)
                out["H3_order"].append({"root": k, "stratum": "proof_grounded" if proof(a, b) else "operational", "feasible": good, "infeasible": bad,
                                        "correct": {x: pg[x] > pb[x] for x in pg if x in pb}, "margin": {x: pg[x] - pb[x] for x in pg if x in pb}})
        if m == "H4" and y(tok["C A"]) == 1:
            for o in ("A", "C"):
                if y(tok[o]) is not None:
                    p = _seedmean(pred, k, tok[o])
                    out["H4_omission"].append({"root": k, "omitted_repair": o, "y": y(tok[o]), "stratum": "proof_grounded" if proof(tok["C A"], tok[o]) else "operational",
                                               "correct": {x: (v < 0.5) == (y(tok[o]) == 0) for x, v in p.items()}, "p": p})
        if m == "H5" and y(tok[""]) is not None:
            out["H5_empty"].append({"root": k, "y_empty": y(tok[""]), "edits": {t: rec[tok[t]]["teacher_status"] for t in ("edit:A", "edit:B", "edit:A+B") if t in tok},
                                    "top1_empty": {ck: pred["arms"][ck]["roots"][k]["top1"]["choice"] == tok[""] for ck in pred["arms"]}})
        if m == "H7" and y(tok["edit:A"]) is not None and y(tok["M"]) is not None:
            pe_, px = _seedmean(pred, k, tok["edit:A"]), _seedmean(pred, k, tok["M"])
            obs = y(tok["edit:A"]) - y(tok["M"])
            out["H7_edit_vs_exec"].append({"root": k, "observed": obs, "stratum": "proof_grounded" if proof(tok["edit:A"], tok["M"]) else "operational",
                                           "pred": {x: pe_[x] - px[x] for x in pe_ if x in px},
                                           "sign_match": {x: bool(np.sign(round(pe_[x] - px[x], 1)) == np.sign(obs)) for x in pe_ if x in px}})
        if m == "H8":
            a = next(q["query"] for q in plan["queries"] if q["root"] == k and q["anchor"])
            out["H8_anchor"].append({"root": k, "status": rec[a]["teacher_status"], "p": _seedmean(pred, k, a)})
    return out


def top1_rows(plan, rec, pred) -> list:
    out = []
    for k in plan["roots"]:
        for ck, a in pred["arms"].items():
            c = a["roots"][k]["top1"]["choice"]
            st = c if c in ("ABSTAIN", "NOT_RUN") else rec[c]["teacher_status"] if c in rec else "NOT_IN_PLAN"
            ver = any(rec[q["query"]]["teacher_status"] == "VERIFIED_FEASIBLE" for q in plan["queries"] if q["root"] == k and q["kind"] == "ORDERED_TEACHER")
            out.append({"root": k, "mechanism": plan["roots"][k]["mechanism"], "ck": ck, "arm": _arm(ck), "choice": c, "status": st, "verified_exists": ver})
    return out


def evaluate() -> dict:
    """Offline scoring of the FROZEN predictions against adjudicated P10a outcomes; engineering gate."""
    import operational_teacher as ot
    import p10_collect as pc
    import p6_run as p6r
    plan, pred, prog = _rd(OUT / "plan.json"), _rd(OUT / "predictions.json"), _rd(OUT / "progress.json")
    adm, pol = _rd(OUT / "static_admission.json"), ot.load_policy("teacher_opneg.toml")
    rec = {q["query"]: p6r.judge(adm["certificates"], pol, q, prog["done"].get(q["query"])) for q in plan["queries"]}
    cf.write_json(OUT / "decisions.json", rec)
    kap, rows, mech, t1 = kappa_rows(plan, rec, pred), binary_rows(plan, rec, pred), mechanism_rows(plan, rec, pred), top1_rows(plan, rec, pred)
    arms = sorted({_arm(ck) for ck in pred["arms"]})
    common = {a: {s: metrics([x for x in rows if _arm(x["ck"]) == a and (s == "all" or (x["source"] == "PLANNER_BUDGET") == (s == "operational"))])
                  for s in ("all", "proof_grounded", "operational")} for a in arms}
    kerr = {a: [float(np.mean([abs(h - x["kappa"]) for ck, h in x["kappa_hat"].items() if _arm(ck) == a])) for x in kap if x["decided"]] for a in arms}
    pk = [x for x in kap if x["decided"] and x["stratum"] == "proof_grounded" and x["mechanism"] == "H1"]
    h1 = [float(np.mean([abs(x["kappa_hat"][c] - x["kappa"]) for c in x["kappa_hat"] if _arm(c) == "K-B-full"]) -
                np.mean([abs(x["kappa_hat"][c] - x["kappa"]) for c in x["kappa_hat"] if _arm(c) == "A"])) for x in pk] if "A" in arms else []
    st = [r["teacher_status"] for r in rec.values()]
    att = [a for d in prog["done"].values() for a in d["attempts"]]
    qcpu = [d["unit_cpu_s"] + d["coord_cpu_s"] for d in prog["done"].values()]
    import p10_protocol as pp
    tq = sum(bool(d["attempts"]) for d in prog["done"].values())
    checks = {"plan_frozen_before_predictions_and_teacher": (OUT / "plan.json").stat().st_mtime < (OUT / "predictions.json").stat().st_mtime
              < (OUT / "static_admission.json").stat().st_mtime, "predictions_bound_to_plan": pred["plan_hash"] == plan["plan_hash"],
              "zero_physical_calls_in_prediction": not sum(pred["physical_calls_during_prediction"].values()),
              "roots_le_14": len(plan["roots"]) <= pc.CAPS["roots"], "queries_le_224": len(plan["queries"]) <= pc.CAPS["queries"],
              "teacher_queries_le_224": tq <= pc.CAPS["queries"], "teacher_cpu_le_5000": pp.spent("teacher")[0] <= pc.CAPS["teacher_cpu_s"],
              "per_query_cpu_le_120_plus_coord": all(d["unit_cpu_s"] <= pc.CAPS["query_cpu_s"] for d in prog["done"].values()),
              "per_seed_le_100": all((a["cpu_s"] or 0) <= pc.CAPS["seed_cpu_s"] for a in att),
              "no_fatal": "FATAL_CONTRADICTION" not in st, "statuses_preserved": set(st) <= {ot.VERIFIED, ot.CERTIFIED, ot.BUDGET, ot.UNKNOWN, "NOT_RUN"},
              "no_budget_from_censored": all(not (r["teacher_status"] == ot.BUDGET and any(not a["clean"] for a in r["attempts"])) for r in rec.values()),
              "wave1_gate_recorded": bool(prog.get("gates")), "aggregate_wall": prog["wall_s"] <= pc.CAPS["aggregate_wall_s"]}
    res = {"stage": "P10a", "plan_hash": plan["plan_hash"], "predictions_hash": pred["hash"],
           "status_counts": {s: st.count(s) for s in sorted(set(st))}, "status_by_mechanism": {m: {s: sum(1 for r in rec.values() if r["query"]["mechanism"] == m
                                                                                                   and r["teacher_status"] == s) for s in sorted(set(st))} for m in pc.ps.MECH},
           "gates": prog.get("gates"), "kappa": kap, "kappa_abs_error_by_arm": kerr,
           "H1_primary_B_minus_A": {"per_root": h1, "n_roots": len(h1), "mean": float(np.mean(h1)) if h1 else None, "ci95": boot(h1),
                                    "status": "UNDERPOWERED (< 5 decided proof-grounded roots)" if len(h1) < 5 else "OK"},
           "mechanisms": mech, "top1": t1, "common_metrics": common, "teacher": {"queries_with_teacher_calls": tq, "seed_attempts": len(att),
           "seed_stops": {s: sum(a["stop"] == s for a in att) for s in {a["stop"] for a in att}}, "query_cpu_max": max(qcpu) if qcpu else 0.0,
           "static_admission_cpu_s": adm["cpu_s"], "teacher_cpu_s": round(pp.spent("teacher")[0], 1), "wall_s": round(prog["wall_s"], 1)},
           "checks": checks, "engineering": "PASS" if all(checks.values()) else "FAIL"}
    res["paired_root_nll"] = paired(rows, arms)
    res["top1_by_arm"] = {a: {s: sum(x["status"] == s for x in t1 if x["arm"] == a) for s in sorted({x["status"] for x in t1})} for a in arms}
    cf.write_json(OUT / "P10A_RESULTS.json", res)
    cf.write_json(Path("runs/p10/P10A_GATE.json"), {k: res[k] for k in ("stage", "plan_hash", "predictions_hash", "checks", "engineering", "status_counts", "gates")})
    return res


def paired(rows, arms) -> dict:
    """Exploratory: per independent root, mean NLL over its decided queries of the SEED-MEAN root-prior p; paired arm contrasts with
    root-cluster bootstrap CI and leave-one-root-out range (proof-grounded only = BUDGET labels excluded)."""
    out = {}
    for stratum in ("all", "proof_grounded"):
        per = {}
        for a in arms:
            byq = {}
            for x in rows:
                if _arm(x["ck"]) == a and (stratum == "all" or x["source"] != "PLANNER_BUDGET"):
                    byq.setdefault((x["root"], x["query"]), [x["y"], []])[1].append(x["p"])
            roots = {}
            for (k, _), (y, ps) in byq.items():
                p = min(max(float(np.mean(ps)), 1e-9), 1 - 1e-9)
                roots.setdefault(k, []).append(-math.log(p if y else 1 - p))
            per[a] = {k: float(np.mean(v)) for k, v in roots.items()}
        con = {}
        for a, b in (("K-B-full", "A"), ("K-C-full", "K-B-full"), ("K-B-full", "SF"), ("K-C-full", "A"), ("SF", "A")):
            if a in per and b in per:
                ks = sorted(set(per[a]) & set(per[b]))
                d = [per[a][k] - per[b][k] for k in ks]
                loro = [float(np.mean(d[:i] + d[i + 1:])) for i in range(len(d))]
                con[f"{a} - {b}"] = {"roots": len(d), "mean": float(np.mean(d)), "ci95": boot(d), "loro": [min(loro), max(loro)],
                                     "better_roots": sum(x < 0 for x in d), "worse_roots": sum(x > 0 for x in d)}
        out[stratum] = {"per_root_mean_nll": per, "contrasts": con}
    return out
