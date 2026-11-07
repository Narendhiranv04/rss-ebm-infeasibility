"""Mock analysis: determinism, virtual budget comparison (Wilson 95%), coverage, diversity, worker sweep, lineage-bootstrap forecast.

  analyze MOCK_DIR OUT.json      reads MOCK_DIR/runs/* (reference, sweep_*, serial, pilot_A, pilot_B, interrupt*) + configs + rosters
"""
import gzip, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy3d as Q  # noqa: E402


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load_run(d):
    d, recs, units = Path(d), {}, {}
    for f in sorted((d / "units").glob("*.json")):
        doc = json.loads(f.read_text())
        units[f.stem] = {"cpu_s": doc["cpu_s"], "wall_s": doc["wall_s"], "n": len(doc["records"]),
                         "json_bytes": f.stat().st_size, "npz_bytes": (d / "units" / f"{f.stem}.npz").stat().st_size if (d / "units" / f"{f.stem}.npz").exists() else 0}
        for r in doc["records"]:
            recs[r["qid"]] = {**r, "unit": f.stem}
    led = [json.loads(x) for x in (d / "ledger.jsonl").read_text().splitlines()] if (d / "ledger.jsonl").exists() else []
    mon = [json.loads(x) for x in (d / "monitor.jsonl").read_text().splitlines()] if (d / "monitor.jsonl").exists() else []
    return {"recs": recs, "units": units, "ledger": led, "monitor": mon, "manifest": json.loads((d / "run_manifest.json").read_text())}


def compare(a, b, drop=("unit",)):
    """Canonical equality on shared queries.  Same unit composition: everything but CPU / wall.  Different composition (unit=query vs
    unit=root, or a subset): also drop the screening work counters, which depend on the shared IK cache's history (outcomes do not)."""
    shared = sorted(set(a) & set(b))
    bad = [q for q in shared if Q.canonical(a[q], drop) != Q.canonical(b[q], drop)]
    return {"shared": len(shared), "mismatched": len(bad), "examples": bad[:5], "ignored_fields": list(drop)}


XCOMP = ("unit", "work", "work_screen", "work_pre_route")
NEWF = ("unit", "work_pre_route")                                # AMENDMENT_002 field, absent from runs made before it


def negative_audit(recs):
    bad = []
    for q, r in recs.items():
        a, L, s = r.get("accounting", {}), r["label"], r["status"]
        ok = L in (0, 1, None) and (L is not None or s.startswith(("UNLABELED", "INTERRUPTED")))
        if a:
            ok &= a["nominal"] == a["screened"] + a["unscreened"]
            ok &= a["screened"] == a["ik_invalid"] + a["held_invalid"] + a["held_invalid_physical"] + a["geometry_invalid"] + a["valid"]
            ok &= a["valid"] == a["untested"] + a["unresolved"] + a["plan_failed"] + a["exec_failed"] + a["verified"]
            ok &= a["motion_tested"] == a["unresolved"] + a["plan_failed"] + a["exec_failed"] + a["verified"]
        if L == 1:
            ok &= s == "VERIFIED" and a["verified"] >= 1 and bool(r.get("witness_id"))
        if L == 0:
            ok &= a["verified"] == 0 and not r.get("witness_id") and r.get("negative_kind") in ("CERTIFIED_GEOMETRY", "BOUNDED_SEARCH", "EXECUTION", "DETERMINISTIC_PLAN")
            ok &= (s == "NEG_CERTIFIED_GEOMETRY" and a["valid"] == 0 and a["geometry_invalid"] >= 1) or (s == "NEG_UNDER_POLICY" and a["motion_tested"] >= 1)
            if r.get("negative_kind") == "BOUNDED_SEARCH":
                ok &= a["unresolved"] >= 1
        if not ok:
            bad.append(q)
    neg = [r for r in recs.values() if r["label"] == 0]
    return {"records": len(recs), "inconsistent": bad, "negatives": len(neg), "by_kind": Counter(r["negative_kind"] for r in neg),
            "negatives_with_untested_valid_units": sum(r["accounting"]["untested"] > 0 for r in neg),
            "negatives_fully_screened": sum(r["accounting"]["unscreened"] == 0 for r in neg),
            "unlabeled_never_zero": all(r["label"] is None for r in recs.values() if r["status"].startswith(("UNLABELED", "INTERRUPTED")))}


def strat(r):
    c = (r.get("clutter") or {}).get("between", 0)
    return {"K": str(r["K"]), "depth": str(r["depth"]), "band": r["band"], "shape": r["shape"], "between_band": "0" if c == 0 else "1-2" if c <= 2 else "3+"}


def policy_table(ref, real_adaptive):
    rows, strata = {}, ("K", "depth", "band", "shape")
    paired = {q: r for q, r in ref.items() if r.get("label") in (0, 1)}
    pos = [q for q, r in paired.items() if r["label"] == 1]
    for name, stages in {"reference_v1": Q.POLICIES["reference_v1"], "adaptive_v1": Q.POLICIES["adaptive_v1"], **Q.FIXED}.items():
        v = {q: Q.virtual(r, stages) for q, r in paired.items()}
        miss = [q for q in pos if v[q]["label"] != 1]
        cpu = [sum(v[q]["cpu"].values()) for q in paired]
        row = {"stages": stages, "n_paired": len(paired), "ref_pos": len(pos), "missed": len(miss), "rel_miss": round(len(miss) / max(1, len(pos)), 4),
               "rel_miss_ci95": wilson(len(miss), len(pos)), "virtual_cpu_mean_s": round(float(np.mean(cpu)), 3), "virtual_cpu_total_s": round(float(np.sum(cpu)), 1),
               "route_iters_total": int(sum(v[q]["work"].get("rrt_iters", 0) for q in paired)), "missed_qids": miss, "by_stratum": {}}
        for s in strata:
            for val in sorted({strat(paired[q])[s] for q in pos}):
                p = [q for q in pos if strat(paired[q])[s] == val]
                m = [q for q in p if q in miss]
                row["by_stratum"][f"{s}={val}"] = {"ref_pos": len(p), "missed": len(m), "rel_miss": round(len(m) / len(p), 4), "ci95": wilson(len(m), len(p))}
        rows[name] = row
    va = {q: Q.virtual(r, Q.POLICIES["adaptive_v1"]) for q, r in paired.items()}
    chk = {"compared": 0, "mismatch": []}
    for q, r in real_adaptive.items():
        if q not in va:
            continue
        chk["compared"] += 1
        st = {str(m["rank"]): (m["state"] if m["state"] != "UNRESOLVED" else f"UNRESOLVED@{m['tests'][-1]['iters']}") for m in r["modes"] if m["tests"]}
        w = Counter()
        for m in r["modes"]:
            for t in m["tests"]:
                w.update(t["work_plan"]); w.update(t.get("work_exec", {}))
        ok = va[q]["label"] == r["label"] and va[q]["verified_rank"] == r["verified_rank"] and va[q]["states"] == st and dict(w) == va[q]["work"]
        if r.get("witness_id") and ref[q].get("witness_id") and r["verified_rank"] == ref[q]["verified_rank"]:
            ok = ok and r["witness_id"] == ref[q]["witness_id"]
        if not ok:
            chk["mismatch"].append(q)
    return rows, chk


def derive_fixed(r, M, B):
    """Label of fixed (M, B) from an adaptive_v1 record (stages (4,500)->(8,1500)->(16,5000)): 1 iff one of the first M valid units
    was VERIFIED by a test with iters <= B.  Exact for M <= 4, B <= 1500 (those units are always tested at 1500 before stage 3)."""
    if r["label"] is None:
        return None
    valid = [m for m in r["modes"] if m["screen"] == "VALID"]
    if not valid:
        return r["label"]
    return int(any(m["state"] == "VERIFIED" and m["tests"][-1]["iters"] <= B for m in valid[:M]))


def choose_policy(rows, rule):
    ps = rule["policy_selection"]
    ok = []
    for name, r in rows.items():
        strat_ok = all(v["rel_miss"] <= ps["rel_missed_positive_max_stratum"] for v in r["by_stratum"].values() if v["ref_pos"] >= ps["stratum_min_ref_positives"])
        if r["rel_miss"] <= ps["rel_missed_positive_max_overall"] and strat_ok:
            ok.append(name)
    best = min(ok, key=lambda n: (rows[n]["virtual_cpu_mean_s"], rows[n]["route_iters_total"]))
    chosen = "adaptive_v1" if "adaptive_v1" in ok and rows["adaptive_v1"]["virtual_cpu_mean_s"] <= 1.15 * rows[best]["virtual_cpu_mean_s"] else best
    return {"passing": ok, "cheapest_passing": best, "chosen": chosen}


def summary(recs, roster):
    el = {q["qid"]: q for q in roster if not q["static"]}
    out = {"n": len(recs), "status": Counter(r["status"] for r in recs.values()), "label": Counter(str(r["label"]) for r in recs.values()),
           "negative_kind": Counter(r.get("negative_kind") for r in recs.values() if r["label"] == 0), "by_stratum": {}, "accounting": Counter(),
           "verified_rank": Counter(r["verified_rank"] for r in recs.values() if r["label"] == 1),
           "pos_lineages": len({el[q]["root_id"] for q, r in recs.items() if r["label"] == 1}),
           "neg_lineages": len({el[q]["root_id"] for q, r in recs.items() if r["label"] == 0}), "cpu_parts": Counter()}
    for q, r in recs.items():
        out["accounting"].update(r.get("accounting", {}))
        out["cpu_parts"].update({k: v for k, v in r.get("cpu_parts", {}).items()})
        for s, v in strat(el[q]).items():
            d = out["by_stratum"].setdefault(f"{s}={v}", {"n": 0, "1": 0, "0": 0, "unlabeled": 0, "cpu": []})
            d["n"] += 1; d["1" if r["label"] == 1 else "0" if r["label"] == 0 else "unlabeled"] += 1; d["cpu"].append(r["cpu_query_s"])
    for d in out["by_stratum"].values():
        c = d.pop("cpu"); d["cpu_mean_s"] = round(float(np.mean(c)), 2); d["cpu_p90_s"] = round(float(np.percentile(c, 90)), 2)
    out["cpu_query_s"] = {"mean": round(float(np.mean([r["cpu_query_s"] for r in recs.values()])), 3),
                          "median": round(float(np.median([r["cpu_query_s"] for r in recs.values()])), 3),
                          "p95": round(float(np.percentile([r["cpu_query_s"] for r in recs.values()], 95)), 3),
                          "max": round(float(max(r["cpu_query_s"] for r in recs.values())), 3)}
    return out


def coverage(recs, roster, rule):
    el = {q["qid"]: q for q in roster if not q["static"]}
    blocked = []
    for s, vals in rule["gates"]["coverage"]["strata"].items():
        for v in vals:
            n = sum(1 for q, r in recs.items() if r["label"] in (0, 1) and strat(el[q])[s] == str(v))
            if n < rule["gates"]["coverage"]["min_labeled_per_stratum"]:
                blocked.append(f"{s}={v}")
    return blocked


def sweep(runs, serial_recs):
    out = {}
    for name, r in runs.items():
        led = r["ledger"]
        st = [x for x in led if x["event"] == "start"][-1]; en = [x for x in led if x["event"] == "end"][-1]
        n = len(r["recs"]); cpu = [x["cpu_query_s"] for x in r["recs"].values()]
        mon = r["monitor"]
        out[name] = {"workers": st["workers"], "cpus": st["cpus"], "queries": n, "makespan_s": round(en["wall_s"], 1), "q_per_s": round(n / en["wall_s"], 4),
                     "cpu_per_query_mean_s": round(float(np.mean(cpu)), 3), "safety_stop": en.get("safety_stop"),
                     "peak_rss_gb": round(max([m["rss_max_gb"] for m in mon] or [0]), 3), "max_cpu_pct_sum": round(max([m["cpu_pct_sum"] for m in mon] or [0]), 1),
                     "max_temp_c": max([m["pkg_temp_c"] or 0 for m in mon] or [0]), "min_mem_avail_gb": round(min([m["mem_avail_gb"] for m in mon] or [0]), 1),
                     "max_load1": round(max([m["load1"] for m in mon] or [0]), 2), "violations": sorted({v for m in mon for v in m["violations"]}),
                     "max_affinity_union": max([len(m["affinity_union"]) for m in mon] or [0])}
        if serial_recs:
            sh = sorted(set(serial_recs) & set(r["recs"]))
            out[name]["cpu_inflation_vs_serial"] = round(float(np.mean([r["recs"][q]["cpu_query_s"] for q in sh]) /
                                                         np.mean([serial_recs[q]["cpu_query_s"] for q in sh])), 3) if sh else None
    return out


def forecast(pilot, prod_rows, workers, buffer=1.5, reps=2000, seed=7):
    """Production CPU = sum over production queries of the per-stratum mean pilot CPU (back-off: K x depth x band -> depth x band -> band
    -> all), plus the per-root overhead; root-lineage bootstrap over pilot roots."""
    recs = pilot["recs"]
    by_root = defaultdict(list)
    for q, r in recs.items():
        by_root[q.split(":")[0]].append(r)
    overhead = [u["cpu_s"] - sum(x["cpu_query_s"] for x in by_root[k]) for k, u in pilot["units"].items()]
    keys = [lambda x: (str(x["K"]), str(x["depth"]), x["band"]), lambda x: (str(x["depth"]), x["band"]), lambda x: (x["band"],), lambda x: ()]
    comp = Counter(tuple(k(x) for k in keys) for x in prod_rows)

    def total(roots):
        rs = [r for k in roots for r in by_root[k]]
        means = [defaultdict(list) for _ in keys]
        for r in rs:
            for j, k in enumerate(keys):
                means[j][k(r)].append(r["cpu_query_s"])
        tot, used = 0.0, Counter()
        for ck, n in comp.items():
            for j in range(len(keys)):
                if means[j].get(ck[j]):
                    tot += n * float(np.mean(means[j][ck[j]])); used[j] += n; break
        return tot, used
    roots = sorted(by_root)
    base, used = total(roots)
    n_roots = len({x["root_id"] for x in prod_rows})
    ov = float(np.mean(overhead)) if overhead else 0.0
    rng = np.random.default_rng(seed)
    boots = [total(list(rng.choice(roots, len(roots), replace=True)))[0] for _ in range(reps)]
    eff = 0.95
    days = lambda cpu_s: (cpu_s + n_roots * ov) / workers / eff / 86400  # noqa: E731
    pos_rate = sum(r["label"] == 1 for r in recs.values()) / len(recs)
    wb = sum(u["npz_bytes"] for u in pilot["units"].values()) / max(1, sum(r["label"] == 1 for r in recs.values()))
    jb = sum(u["json_bytes"] for u in pilot["units"].values()) / len(recs)
    n = len(prod_rows)
    return {"production_queries": n, "production_roots": n_roots, "backoff_level_counts": {["K*depth*band", "depth*band", "band", "all"][j]: c for j, c in used.items()},
            "cpu_hours": round((base + n_roots * ov) / 3600, 2), "cpu_hours_p05_p95": [round((np.percentile(boots, 5) + n_roots * ov) / 3600, 2),
                                                                                    round((np.percentile(boots, 95) + n_roots * ov) / 3600, 2)],
            "workers": workers, "efficiency_assumed": eff, "wall_days": round(days(base), 3), "wall_days_p95": round(days(np.percentile(boots, 95)), 3),
            "wall_days_p95_x_buffer": round(days(np.percentile(boots, 95)) * buffer, 3), "per_root_overhead_cpu_s": round(ov, 2),
            "positive_rate_pilot": round(pos_rate, 4), "bytes_per_witness": int(wb), "json_bytes_per_query": int(jb),
            "disk_gb": round((n * pos_rate * wb + n * jb) / 1e9, 3), "disk_gb_x_buffer": round((n * pos_rate * wb + n * jb) * buffer / 1e9, 3)}


def analyze(mock, out):
    mock = Path(mock)
    rule = json.loads((mock / "configs/selection_rule.json").read_text())
    roster = [json.loads(x) for x in gzip.open(mock / "roster.jsonl.gz", "rt")]
    prod = [json.loads(x) for x in gzip.open(mock / "production_roster_design.jsonl.gz", "rt")]
    prod = [x for x in prod if not x["static"]]
    runs = {p.name: load_run(p) for p in sorted((mock / "runs").iterdir()) if (p / "run_manifest.json").exists()}
    res = {"runs": {k: {"queries": len(v["recs"]), "units": len(v["units"]), "policy": v["manifest"]["policy"], "subset": v["manifest"]["subset"]} for k, v in runs.items()}}
    A, B = runs["pilot_A"]["recs"], runs["pilot_B"]["recs"]
    det = {"pilot_A_vs_B": compare(A, B), "serial_vs_pilot_A": compare(runs["serial"]["recs"], A, XCOMP),
           "reference_witnesses_vs_pilot_A": {"shared_positive_same_rank": 0, "witness_id_equal": 0}}
    for k in runs:
        if k.startswith("sweep_"):
            det[f"{k}_vs_pilot_A"] = compare(runs[k]["recs"], A, XCOMP)
            det[f"{k}_vs_sweep_8"] = compare(runs[k]["recs"], runs["sweep_8"]["recs"])
        if k.startswith("interrupt"):
            det[f"{k}_vs_pilot_A"] = compare(runs[k]["recs"], A)
    for q, r in runs["reference"]["recs"].items():
        if q in A and r.get("witness_id") and A[q].get("witness_id") and r["verified_rank"] == A[q]["verified_rank"]:
            det["reference_witnesses_vs_pilot_A"]["shared_positive_same_rank"] += 1
            det["reference_witnesses_vs_pilot_A"]["witness_id_equal"] += r["witness_id"] == A[q]["witness_id"]
    sup = mock / "runs_superseded"
    for p in sorted(sup.iterdir()) if sup.exists() else []:
        if (p / "run_manifest.json").exists() and p.name.startswith("reference"):
            det[f"reference_vs_superseded_{p.name}"] = compare(runs["reference"]["recs"], load_run(p)["recs"], NEWF)
    if "pilot_C1" in runs:
        C1, C2 = runs["pilot_C1"]["recs"], runs["pilot_C2"]["recs"]
        det["pilot_C1_vs_C2"] = compare(C1, C2)
        M, B = Q.FIXED["fixed_4x1500"][0]
        der = {q: derive_fixed(r, M, B) for q, r in A.items()}              # exact nested-prefix derivation from the adaptive records
        det["pilot_C1_vs_derived_from_adaptive_pilot_A"] = {"compared": len(der), "label_mismatch": [q for q in der if der[q] != C1[q]["label"]]}
        ref = runs["reference"]["recs"]
        vc = {q: Q.virtual(r, Q.FIXED["fixed_4x1500"]) for q, r in ref.items() if r.get("label") in (0, 1)}
        bad = []
        for q, v in vc.items():
            c = C1[q]
            st = {str(m["rank"]): (m["state"] if m["state"] != "UNRESOLVED" else f"UNRESOLVED@{m['tests'][-1]['iters']}") for m in c["modes"] if m["tests"]}
            w = Counter()
            for m in c["modes"]:
                for t in m["tests"]:
                    w.update(t["work_plan"]); w.update(t.get("work_exec", {}))
            ok = v["label"] == c["label"] and v["verified_rank"] == c["verified_rank"] and v["states"] == st and v["work"] == dict(w)
            ok = ok and (c["label"] != 1 or c["witness_id"] == ref[q]["witness_id"] or c["verified_rank"] != ref[q]["verified_rank"])
            if not ok:
                bad.append(q)
        det["virtual_fixed_4x1500_vs_real_pilot_C1"] = {"compared": len(vc), "mismatch": bad}
    res["negative_audit"] = {k: negative_audit(runs[k]["recs"]) for k in runs if k.startswith(("pilot", "reference"))}
    rows, vchk = policy_table(runs["reference"]["recs"], {q: A[q] for q in runs["reference"]["recs"] if q in A})
    det["virtual_adaptive_vs_real"] = vchk
    res.update(determinism=det, policies=rows, policy_choice=choose_policy(rows, rule), pilot=summary(A, roster),
               reference=summary(runs["reference"]["recs"], roster), coverage_blocked=coverage(A, roster, rule))
    sw = {k: v for k, v in runs.items() if k.startswith("sweep_") or k == "serial"}
    res["sweep"] = sweep(sw, runs["serial"]["recs"])
    ws = {k: v for k, v in res["sweep"].items() if k.startswith("sweep_")}
    best = max(ws, key=lambda k: ws[k]["q_per_s"])
    pick = min((k for k in ws if ws[k]["q_per_s"] >= 0.95 * ws[best]["q_per_s"]), key=lambda k: ws[k]["workers"])
    res["worker_choice"] = {"fastest": best, "chosen": pick, "workers": ws[pick]["workers"], "rule": rule["worker_choice"]}
    if "supplement_reference_on_adaptive_negatives" in runs and "pilot_C1" in runs:  # post-hoc diagnostic supplement (disclosed)
        Sx = runs["supplement_reference_on_adaptive_negatives"]["recs"]
        ref_lab = {q: (Sx[q]["label"] if q in Sx else A[q]["label"]) for q in A}
        el = {q["qid"]: q for q in roster if not q["static"]}
        pos = [q for q in A if ref_lab[q] == 1]
        bp = {"note": "reference_v1 run on every adaptive_v1 NEG_UNDER_POLICY pilot query (27); certified-geometry negatives have no valid unit, "
                      "adaptive positives are reference positives => reference label known for all 205 pilot queries", "ref_positives": len(pos),
              "supplement_new_positives": sum(1 for q in Sx if Sx[q]["label"] == 1)}
        for name, L in (("adaptive_v1", A), ("fixed_4x1500", runs["pilot_C1"]["recs"])):
            m = [q for q in pos if L[q]["label"] != 1]
            row = {"missed": len(m), "rel_miss": round(len(m) / len(pos), 4), "ci95": wilson(len(m), len(pos)), "missed_qids": m, "by_stratum": {}}
            for sname in ("band", "between_band", "depth", "K", "shape"):
                for v in sorted({strat(el[q])[sname] for q in pos}):
                    pp = [q for q in pos if strat(el[q])[sname] == v]; mm = [q for q in pp if q in m]
                    row["by_stratum"][f"{sname}={v}"] = {"ref_pos": len(pp), "missed": len(mm), "rel_miss": round(len(mm) / len(pp), 4), "ci95": wilson(len(mm), len(pp))}
            bp[name] = row
        res["broad_pilot_vs_reference"] = bp
        res["measured_cpu_ratio_adaptive_over_fixed_4x1500"] = round(res["pilot"]["cpu_query_s"]["mean"] / summary(runs["pilot_C1"]["recs"], roster)["cpu_query_s"]["mean"], 3)
    rec_pol = res["policy_choice"]["chosen"]
    base = "pilot_C1" if rec_pol == "fixed_4x1500" and "pilot_C1" in runs else "pilot_A"
    res["forecast"] = {"policy": rec_pol, "from_run": base, **forecast(runs[base], prod, ws[pick]["workers"])}
    res["forecast_adaptive_v1"] = {"policy": "adaptive_v1", "from_run": "pilot_A", **forecast(runs["pilot_A"], prod, ws[pick]["workers"])}
    res["forecast_full_quota_2500"] = {"note": "scaled by roots: the live quota is 2,500 roots incl. K4 510 / NONE 408 that have NO samples (COVERAGE_BLOCKED"
                                       " for K4 / NONE); per-query cost assumed = snapshot mean", "scale": round(2500 / res["forecast"]["production_roots"], 4),
                                       "cpu_hours": round(res["forecast"]["cpu_hours"] * 2500 / res["forecast"]["production_roots"], 2),
                                       "wall_days_p95_x_buffer": round(res["forecast"]["wall_days_p95_x_buffer"] * 2500 / res["forecast"]["production_roots"], 3)}
    if "pilot_C1" in runs:
        res["pilot_recommended"] = summary(runs["pilot_C1"]["recs"], roster)
        res["coverage_blocked_recommended"] = coverage(runs["pilot_C1"]["recs"], roster, rule)
    res["forecast_composition"] = {"by_K": Counter(str(x["K"]) for x in prod), "by_depth": Counter(str(x["depth"]) for x in prod),
                                   "by_band": Counter(x["band"] for x in prod)}
    res["safety"] = sweep({k: v for k, v in runs.items() if k.startswith(("pilot", "reference", "interrupt", "supplement"))}, None)
    sw_all = {**res["sweep"], **res["safety"]}
    res["safety_overall"] = {"runs": len(sw_all), "violations": sorted({x for v in sw_all.values() for x in v["violations"]}),
                             "safety_stops": [k for k, v in sw_all.items() if v["safety_stop"]], "max_affinity_union": max(v["max_affinity_union"] for v in sw_all.values()),
                             "max_cpu_pct_sum": max(v["max_cpu_pct_sum"] for v in sw_all.values()), "max_temp_c": max(v["max_temp_c"] for v in sw_all.values()),
                             "peak_rss_gb": max(v["peak_rss_gb"] for v in sw_all.values()), "min_mem_avail_gb": min(v["min_mem_avail_gb"] for v in sw_all.values())}
    for k, v in res["sweep"].items():                                    # steady-state estimate: N workers / mean per-query CPU at N
        v["steady_state_q_per_s"] = round(v["workers"] / v["cpu_per_query_mean_s"], 4)
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    print(json.dumps({k: res[k] for k in ("determinism", "policy_choice", "coverage_blocked", "worker_choice")}, default=str)[:3000])


if __name__ == "__main__":
    analyze(sys.argv[2], sys.argv[3])
