"""P14 ledger (runs/p14/guards, contract caps, one guarded OS process per job) and P14-A0 errata of P13 computed from frozen P13
artifacts only (zero physical labels). P10-P13 files are read-only; every P14 output is a new versioned file under runs/p14."""
from __future__ import annotations

import collections
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

import config as cf

OUT = Path("runs/p14")
P13 = Path("runs/p13")
CAPS = {"a_correction": 2500.0, "a_diagnostics": 3000.0, "d_index": 4000.0, "d_search": 5000.0, "d_pilot": 8000.0, "regression": 2000.0}
JOB = {"cpu_s": 2500.0, "wall_s": 20000.0}


def rd(p):
    return json.loads(Path(p).read_text())


def sha(p) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spent(cat=None) -> float:
    return sum(g["cpu_s"] for g in (rd(p) for p in (OUT / "guards").glob("*.json")) if cat is None or g["category"] == cat)


def job(name, call, cpu, module="p14_audit", category="a_correction", wall=None) -> dict:
    """One guarded child per OS process (never threads in one parent: P13 lesson), charged to one P14 category."""
    import teacher
    (OUT / "guards").mkdir(parents=True, exist_ok=True)
    cpu = min(cpu, JOB["cpu_s"], CAPS[category] - spent(category))
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
    print(json.dumps({k: rec[k] for k in ("stop", "returncode", "cpu_s", "wall_s")} | {category: round(spent(category), 1)}))
    return rec


def gate(stage, **kw) -> None:
    cf.write_json(OUT / f"P14_{stage}_GATE.json", {"stage": f"P14-{stage}", **kw, "spend": {c: round(spent(c), 1) for c in CAPS}})


def paired(a, b):
    """Root-paired a[k] - b[k] (positive = a larger): mean, bootstrap 95 % CI (2,000, seed 0), exact two-sided sign test, counts."""
    import p12_closure as p12
    ks = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in ks]
    p, pos, neg, ties = p12._sign(d)
    return {"roots": len(d), "mean_a_minus_b": float(np.mean(d)) if d else None, "ci95": p12._boot(d), "sign_p": p,
            "a_larger": pos, "b_larger": neg, "ties": ties}


# ------------------------------------------------------------------ P14-A0 errata (frozen P13 artifacts only)
def _exec_recompute():
    """Independent recomputation from the eight final ledgers + frozen jobs: outcome histogram by unique episode, per-arm root success
    (seed mean, abstain = failure), per-arm admission, attempt-conditional success and per-stream split."""
    man = rd(P13 / "p13_5" / "EXEC_JOBS.json")
    led = {}
    for f in sorted((P13 / "p13_5" / "episodes").glob("w*/ledger.json")):
        for k, v in rd(f)["episodes"].items():
            if k in led:
                raise ValueError(f"duplicate episode identity {k}")
            led[k] = v
    jm = {f"{j['root']}::{j['choice']}": f"{j['arm']}__{j['root']}" for j in man["jobs"]}
    hist = collections.Counter(v["outcome"] for v in led.values())
    plans = {k: r for d in sorted((P13 / "holdout").glob("*_b*")) for k, r in rd(d / "plan.json")["roots"].items()}
    per, att = (collections.defaultdict(lambda: collections.defaultdict(list)) for _ in range(2))
    for key, arms in man["arms"].items():
        k, ch = key.split("::", 1)
        e = led.get(jm.get(key, ""), None)
        for a in arms:
            arm = a if a in ("geometry", "G+M") else a.split("-s")[0].removeprefix("P13-")
            ok = float(bool(e) and e["outcome"] == "SUCCESS")
            per[arm][k].append(ok)
            att[arm][k].append(None if ch in ("ABSTAIN", "NOT_RUN") else ok)
    succ = {a: {k: float(np.mean(v)) for k, v in d.items()} for a, d in per.items()}
    out = {"episodes": len(led), "jobs": len(man["jobs"]), "histogram_final_ledgers": dict(hist), "roots": len(man["permutation"]),
           "success": {}, "admission": {}, "success_given_attempt": {}}
    for s in ("all", "blind", "natural"):
        ks = [k for k in man["permutation"] if s == "all" or plans[k].get("split") == "blind" and (s == "natural") == plans[k]["family"].startswith("NAT")]
        out["success"][s] = {a: float(np.mean([d[k] for k in ks])) for a, d in succ.items()}
        out["admission"][s] = {a: float(np.mean([x is not None for k in ks for x in att[a][k]])) for a in att}
        out["success_given_attempt"][s] = {a: float(np.mean([x for k in ks for x in att[a][k] if x is not None])) for a in att}
    return out, succ, led


def errata() -> dict:
    out = {"historical_hashes": {p: sha(p) for p in ("runs/p13/p13_4/F1_J.json", "runs/p13/p13_4/TARGETED.json", "runs/p13/p13_5/EXEC_RESULTS.json",
                                                     "runs/p13/p13_6/D13.json", "runs/p13/holdout/PREDICTIONS_FROZEN.json", "runs/p13/PREREG.json")}}
    fj = rd(P13 / "p13_4" / "F1_J.json")["J"]
    old = fj["B_success_J_minus_noJ"]
    out["J_sign"] = {"stored_field": "B_success_J_minus_noJ", "stored_mean_delta": old["mean_delta"],
                     "meaning_of_stored_value": "noJ - J (computed on negated success); the field name says J - noJ",
                     "corrected_J_minus_noJ": -old["mean_delta"], "corrected_ci95": [-old["ci95"][1], -old["ci95"][0]],
                     "J_better_roots": old.get("a_better"), "noJ_better_roots": old.get("b_better"), "sign_p": old["p_raw"],
                     "rates": {"J": fj["success_rate"]["K-B|J"], "noJ": fj["success_rate"]["K-B|noJ"]},
                     "endpoint": "TEACHER-LABELLED CHOICE CORRECTNESS (chosen query VERIFIED), not physically executed success"}
    out["J_regret"] = {"status": "ORACLE-FILTERED DIAGNOSTIC (alternatives restricted to teacher-VERIFIED candidates before choosing by J); "
                                 "not deployed travel regret", "historical": {k: v for k, v in fj["relative_travel_regret_mean"].items() if k.startswith("K-B")},
                       "comparable_selected": _j_selected()}
    ex, succ, led = _exec_recompute()
    rep = rd(P13 / "p13_5" / "EXEC_RESULTS.json")
    out["execution"] = ex | {"reported_success": rep["success"],
                             "max_abs_difference_vs_reported": max(abs(ex["success"]["all"][a] - rep["success"][a]) for a in rep["success"]),
                             "narrative_histogram_0064": {"SUCCESS": 295, "NO_REPAIR_WITNESS_WITHIN_BUDGET": 51, "NOT_RUN_BUDGET": 7},
                             "reconciliation": "0064's histogram was printed at 21:27 UTC before the worker-2 continuation (guard 21:28 UTC) ran the 7 "
                                               "never-attempted jobs (6 SUCCESS, 1 NO_REPAIR_WITNESS); EXEC_RESULTS.json was written at 21:36 UTC from the "
                                               "final ledgers, so per-arm success already used all 428 episodes; only the narrative histogram was stale",
                             "continuation_removed": rd(P13 / "p13_5" / "continuation_w2_removed.json")["removed"]}
    tg = rd(P13 / "p13_4" / "TARGETED.json")["primary"]
    out["F3"] = {"mean_delta_nats": tg["H_F3"]["mean_delta"], "bootstrap_ci95": tg["H_F3"]["ci95"], "sign_test": {"p_raw": tg["H_F3"]["p_raw"],
                 "p_holm": tg["H_F3"]["p_holm"], "B_better_roots": tg["H_F3"]["K-B_better"], "B-F3pm_better_roots": tg["H_F3"]["B-F3pm_better"]},
                 "reading": "root-level sign test favours removing F3 (more roots better without it); the mean-effect CI crosses 0 and the magnitude "
                            "(|0.022| nats) is below the 0.05 practical threshold: mean harm is NOT established; no demonstrated benefit"}
    out["F5"] = {"renamed": "COMBINED generic pairwise + higher-order block (B-F5spm removes cpl AND ho_*)", "challenge": tg["H_F5"]["mean_delta"],
                 "natural": rd(P13 / "p13_4" / "TARGETED.json")["safety"]["H_F5"]["natural_joint4"]["mean_delta"],
                 "generic_only_tested": False}
    out["F4"] = {"admission": {s: {a: ex["admission"][s][a] for a in ("K-B", "B-F4")} for s in ex["admission"]},
                 "success_given_attempt": {s: {a: ex["success_given_attempt"][s][a] for a in ("K-B", "B-F4")} for s in ex["success_given_attempt"]},
                 "root_success": {s: {a: ex["success"][s][a] for a in ("K-B", "B-F4")} for s in ex["success"]}}
    out["D"] = _d_gated()
    out["compute_P13_training"] = _training_bounds()
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "audit" / "P13_ERRATA.json", out)
    return out


def _j_selected():
    """Comparable selected candidates: roots where BOTH the J-choice and the no-J choice of a K-B seed are teacher-VERIFIED with witnessed
    travel (J_first); relative travel of the J-choice vs the no-J choice. Coverage reported; nothing imputed for other roots."""
    import p13_factors as f13
    snap, P, y, st, cost = f13._p13_blind()
    rows_j, rows_n, cov = {}, {}, collections.Counter()
    for s, a in f13._by_arm(snap)["K-B"].items():
        for k in P:
            rows = [r for r in a["roots"][k]["rows"] if 1 / (1 + math.exp(-r["u"] / a["tau"])) >= a["threshold"]]
            if not rows:
                cov["abstain"] += 1
                continue
            mm = min(r["m"] for r in rows)
            tied = [r for r in rows if r["m"] == mm]
            qj = min(tied, key=lambda r: (r["J_pred"], r["C_d"], r["energy"], r["query"]))["query"]
            qn = min(tied, key=lambda r: (r["C_d"], r["energy"], r["query"]))["query"]
            if qj == qn:
                cov["same_choice"] += 1
                continue
            if y.get(qj) == 1 and y.get(qn) == 1 and cost.get(qj) is not None and cost.get(qn) is not None:
                cov["both_verified"] += 1
                rows_j.setdefault(k, []).append(cost[qj])
                rows_n.setdefault(k, []).append(cost[qn])
            else:
                cov["different_choice_not_both_verified"] += 1
    a, b = {k: float(np.mean(v)) for k, v in rows_j.items()}, {k: float(np.mean(v)) for k, v in rows_n.items()}
    return {"coverage_choices": dict(cov), "witnessed_travel_noJ_minus_J": paired(b, a), "unit": "teacher first-witness travel (J_first)"}


def _d_gated() -> dict:
    """Blocker inference with the feasibility gate (every decided original): predicted-feasible => empty predicted mask; feasible originals
    have an empty true mask; versus P13's conditional truth-known-infeasible F1 (119 roots)."""
    import p10b_dataset as pd
    import p11_closure as q
    import p13_factors as f13
    import p13_pipeline as pp
    snap, P, y, st, cost = f13._p13_blind()
    rec = {}
    for d in sorted(pp.HOLD.glob("*_b*")):
        rec |= pp._sealed(d)
    plans = {k: r for d in sorted(pp.HOLD.glob("*_b*")) for k, r in pp.rd(d / "plan.json")["roots"].items() if k in P}
    dl = pd.d_labels({"roots": plans}, rec)
    ks = [k for k in P if dl[k]["y0"] is not None and (dl[k]["y0"] == 1 or dl[k]["mask"] is not None)]
    gt = {k: ([0] * dl[k]["n"] if dl[k]["y0"] == 1 else dl[k]["mask"]) for k in ks}
    res = {}
    for arm in ("D", "D-pairless"):
        lam, thr = snap["baselines"][f"{arm}_lambda_B"], snap["baselines"][f"{arm}_threshold"]
        ds = [x for x in snap["d"].values() if x["arm"] == arm and x["lambda_B"] == lam]
        per = {}
        for k in ks:
            ms = [q._set_metrics(gt[k], [int(v["roots"][k]["p_feasible"] < 0.5 and mm >= thr) for mm in v["roots"][k]["marg"]]) for v in ds]
            per[k] = {m: float(np.mean([x[m] for x in ms])) for m in ("f1", "exact")}
        res[arm] = {"gated_all": {m: float(np.mean([v[m] for v in per.values()])) for m in ("f1", "exact")},
                    "gated_feasible_originals_exact": float(np.mean([per[k]["exact"] for k in ks if dl[k]["y0"] == 1])),
                    "gated_infeasible_f1": float(np.mean([per[k]["f1"] for k in ks if dl[k]["y0"] == 0]))}
    geo = {k: q._set_metrics(gt[k], snap["baselines"]["heuristic"][k]) for k in ks}
    res["geometry_heuristic"] = {"gated_all": {m: float(np.mean([v[m] for v in geo.values()])) for m in ("f1", "exact")},
                                 "gated_feasible_originals_exact": float(np.mean([geo[k]["exact"] for k in ks if dl[k]["y0"] == 1]))}
    res |= {"roots": len(ks), "feasible_originals": sum(dl[k]["y0"] == 1 for k in ks),
            "conditional_truth_known_infeasible_P13": rd(P13 / "p13_6" / "D13.json")["all"]["all_infeasible"]["arms"],
            "note": "empty-vs-empty counts as F1 1 (p11 _set_metrics convention); union of minimal sets is not universal necessity"}
    return res


def _training_bounds() -> dict:
    """P13 training CPU: raw guard readings (over-counted by the threaded parent), min(guard, wall) upper bound, and the child-recorded
    process_time of completed runs (exact for those runs) + wall of terminated runs (upper bound)."""
    raw = ub = child = 0.0
    for f in sorted((P13 / "guards").glob("p13_3_train_*.json")):
        g = rd(f)
        raw, ub = raw + g["cpu_s"], ub + min(g["cpu_s"], g["wall_s"])
    for f in sorted((P13 / "models" / "runs").glob("P13-*.json")):
        child += rd(f).get("train_cpu_s", 0.0)
    killed = sum(min(rd(f)["cpu_s"], rd(f)["wall_s"]) for f in (P13 / "guards").glob("p13_3_train_*.json") if rd(f)["returncode"] != 0)
    return {"raw_guard_sum": round(raw, 1), "upper_bound_min_guard_wall": round(ub, 1), "child_process_time_completed_runs": round(child, 1),
            "terminated_runs_wall_upper_bound": round(killed, 1), "best_estimate": round(child + killed, 1),
            "status": "BOUNDED ESTIMATE, not an exact measurement (thread-parent RUSAGE bug); raw readings preserved in runs/p13/guards"}


def keying_audit(n_roots=40) -> dict:
    """Catalogue-invariance of keyed vs unkeyed instantiation with frozen P13 checkpoints (3 seeds each): (a) exact duplicate re-listing of a
    multi-move candidate, (b) reversed candidate order, (c) an omission query re-listed as its own candidate. Max |delta P| over the
    ORIGINAL root's modelled queries. Prediction accuracy (unkeyed NLL) is reported separately in 0063; this is invariance only."""
    import copy
    import os
    import torch
    os.environ["P10C_OUT"] = str(P13 / "models")
    import p13_pipeline as pp
    import unique_factor as uf
    import unique_factor_keyed as ufk
    pc = pp._arms()
    rs = [r for r in pp._prepared("test") if any(len(c["seq"]) >= 2 for c in r["cands"])][:n_roots]
    out = {}
    for arm, keyed in (("K-B", True), ("B-U", False)):
        res = collections.defaultdict(list)
        for s in (0, 1, 2):
            m = pc.load(f"P13-{arm}-s{s}")[0]
            for r in rs:
                with torch.no_grad(), (ufk.installed() if keyed else __import__("contextlib").nullcontext()):
                    base = uf.prior_marginals(uf.instances(m, r))
                    multi = [c for c in r["cands"] if len(c["seq"]) >= 2]
                    var = {"duplicate_relisting": dict(r, cands=r["cands"] + [copy.deepcopy(multi[0])]),
                           "reversed_order": dict(r, cands=list(reversed(r["cands"])))}
                    oq = {q for c in multi for sl, q in zip(c["slots"], c["z_q"]) if sl.startswith("zO")}
                    om = [c for c in r["cands"] if c["seq"] and c["query"] in oq]
                    if om:                                # base: the omission query exists ONLY as an omission slot; variant: also listed
                        r_base = dict(r, cands=[c for c in r["cands"] if c["query"] != om[0]["query"]])
                        b0 = uf.prior_marginals(uf.instances(m, r_base))
                        b1 = uf.prior_marginals(uf.instances(m, r))
                        res["omission_slot_vs_listed_candidate"].append(max(abs(b1[q] - b0[q]) for q in b0))
                    for name, rv in var.items():
                        try:
                            mv = uf.prior_marginals(uf.instances(m, rv))
                            res[name].append(max(abs(mv[q] - base[q]) for q in base))
                        except Exception as ex:                      # an identity guard refusing the variant is reported, not hidden
                            res[name + "_refused"].append(type(ex).__name__)
        out[arm] = {k: ({"n": len(v), "max": float(max(v)), "mean": float(np.mean(v)), "nonzero": int(sum(x > 1e-12 for x in v))}
                        if v and not isinstance(v[0], str) else dict(collections.Counter(v))) for k, v in res.items()}
    out["roots"] = len(rs)
    cf.write_json(OUT / "audit" / "KEYING_INVARIANCE.json", out)
    return out


def regression() -> None:
    """Historical suite + P12 / P13 / P14 tests + six replays, outputs under runs/p14/regression (fixture planner calls itemized)."""
    import subprocess
    d = OUT / "regression"
    d.mkdir(parents=True, exist_ok=True)
    (d / "countcalls.py").write_text(Path("runs/p8/audit/regression/countcalls.py").read_text())
    subprocess.run([sys.executable, "run.py", "g4c-lf-p4-regression", "--out", str(d), "--cpu", "1900", "--wall", "5000"], capture_output=True, text=True)


def manifest() -> None:
    """P14 final manifest: sources, prereg, P14-A and P14-D artifacts (separate sections), figures, regression, ledger."""
    import hashlib
    import subprocess
    reg = rd(OUT / "regression" / "regression_result.json")
    A = sorted(map(str, list((OUT / "audit").glob("*.json")) + list((OUT / "calibration").glob("*.json"))))
    D = sorted(map(str, list((OUT / "lookup").glob("*.json")) + list((OUT / "search").glob("*.json")) + list((OUT / "eval").glob("*.json"))))
    out = {"stage": "P14", "base": "99bccf7cbc57feb6150d0d50062548af4d3eebe1", "status": "PARTIAL",
           "gates": {g.stem: rd(g).get("engineering", rd(g).get("decision")) for g in sorted(OUT.glob("P14_*_GATE.json"))},
           "regression": {"tests": reg.get("tests"), "replays": reg.get("replays")}, "spend_cpu_s": {c: round(spent(c), 1) for c in CAPS}, "caps": CAPS,
           "prereg": sha(OUT / "P14_PREREG.json"), "eval_roster": sha(OUT / "eval_roster.json"),
           "sources": {p: sha(p) for p in ("p14_audit.py", "p14_eval.py", "p14_sweep_index.py", "p14_d_search.py", "tests/test_p14.py")},
           "P14_A": {p: sha(p) for p in A if "VALIDATION_ROWS" not in p} | {"runs/p14/calibration/VALIDATION_ROWS.json": sha(OUT / "calibration" / "VALIDATION_ROWS.json")},
           "P14_D": {p: sha(p) for p in D}, "figures": {str(p): sha(p) for p in sorted(Path("visualizations/p14").glob("*")) if p.suffix != ".sha256"},
           "historical_unchanged": rd(OUT / "audit" / "P13_ERRATA.json")["historical_hashes"],
           "head_at_manifest": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    out["hash"] = "sha256:" + hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    cf.write_json(OUT / "P14_FINAL_MANIFEST.json", out)
