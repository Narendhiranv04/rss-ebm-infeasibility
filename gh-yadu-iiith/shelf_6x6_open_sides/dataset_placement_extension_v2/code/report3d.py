"""V2 mock analysis: run validity, determinism, real vs virtual policies, per-stratum misses (Wilson 95%), the frozen V2 choice rule,
V1 -> V2 regression, branch accounting, negative audit, storage, sweep, resources and the live-snapshot runtime / storage forecast.

  python report3d.py analyze MOCK_V2_DIR LIVE_ROSTER OUT.json
"""
import gzip, json, math, random, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import policy3d as Q  # noqa: E402
import run3d  # noqa: E402

XCOMP = ("work_screen", "work")                                # IK-cache-history dependent (differs between unit=root and unit=query)
CANDS = ("adaptive_v1", "fixed_4x1500", "fixed_4x5000", "fixed_8x5000")


def wilson(k, n, z=1.96):
    if n == 0:
        return [None, None]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def load_run(d):
    d = Path(d)
    recs, units = {}, {}
    for f in sorted((d / "units").glob("*.json")):
        doc = json.loads(f.read_text())
        units[f.stem] = doc
        for r in doc["records"]:
            recs[r["qid"]] = r
    led = [json.loads(x) for x in (d / "ledger.jsonl").read_text().splitlines() if x.strip()]
    mon = [json.loads(x) for x in (d / "monitor.jsonl").read_text().splitlines() if x.strip()] if (d / "monitor.jsonl").exists() else []
    man = json.loads((d / "run_manifest.json").read_text())
    return {"recs": recs, "units": units, "ledger": led, "monitor": mon, "manifest": man, "dir": d}


def validity(run):
    man, led = run["manifest"], run["ledger"]
    ends = [e for e in led if e.get("event") == "end"]
    done = {e["unit"]: e for e in led if e.get("event") == "unit_done"}
    bad = [u for u in man["units"] if u not in done or not run3d.validate_unit(run["dir"], u, man["units"][u], done[u])[0]]
    viol = sorted({v for m in run["monitor"] for v in m.get("violations", [])})
    starts = [e for e in led if e.get("event") == "start"]
    return {"complete": bool(ends) and ends[-1]["complete"] and not ends[-1]["safety_stop"], "units": len(man["units"]),
            "units_failing_validation": bad, "monitor_violations": viol, "starts": len(starts),
            "resume_audits": [s.get("resume_audit") for s in starts[1:]],
            "peak": {"cpu_pct_sum": max([m["cpu_pct_sum"] for m in run["monitor"]] or [0]), "rss_max_gb": max([m["rss_max_gb"] for m in run["monitor"]] or [0]),
                     "pkg_temp_c": max([m["pkg_temp_c"] or 0 for m in run["monitor"]] or [0]), "procs": max([m["procs"] for m in run["monitor"]] or [0]),
                     "affinity_union": sorted({c for m in run["monitor"] for c in m["affinity_union"]}),
                     "min_mem_avail_gb": min([m["mem_avail_gb"] for m in run["monitor"]] or [0]), "paused_ticks": sum(m["paused"] for m in run["monitor"]),
                     "swap_growth_gb": max([m["swap_growth_gb"] for m in run["monitor"]] or [0])},
            "wall_s": sum(e["wall_s"] for e in ends), "cpu_s": sum(r.get("cpu_query_s", 0) for r in run["recs"].values())}


def compare(a, b, drop=("work_screen",), qids=None):
    qs = sorted(set(a["recs"]) & set(b["recs"])) if qids is None else qids
    mis = [q for q in qs if Q.canonical(a["recs"][q], drop) != Q.canonical(b["recs"][q], drop)]
    return {"shared": len(qs), "mismatches": len(mis), "examples": mis[:5]}


def real_states(r):
    return {str(m["rank"]): (m["state"] if m["state"] != "UNRESOLVED" else f"UNRESOLVED@{m['tests'][-1]['iters']}") for m in r.get("modes", []) if m["tests"]}


def virtual_vs_real(ref, real, stages):
    mis = []
    for q, r in real["recs"].items():
        if q not in ref["recs"] or ref["recs"][q].get("label") is None and r.get("label") is None:
            continue
        v = Q.virtual(ref["recs"][q], stages)
        ok = v["label"] == r["label"] and v["states"] == real_states(r) and v["work"] == Q.motion_work(r)
        if r["label"] == 1:
            ok = ok and v["verified"] == [r["verified_rank"], r["verified_branch"], r["witness_id"]]
        if not ok:
            mis.append(q)
    return {"compared": len(real["recs"]), "mismatches": len(mis), "examples": mis[:5]}


def bclass(x):
    b = (x.get("clutter") or {}).get("between", 0)
    return "0" if b == 0 else "1-2" if b <= 2 else "3+"


STRATA = {"overall": lambda x: "all", "band": lambda x: x["band"], "depth": lambda x: str(x["depth"]), "between": bclass, "cohort": lambda x: x["cohort"],
          "shape": lambda x: x["shape"], "K": lambda x: x["K"], "dest_is_t_goal": lambda x: str(bool(x["dest_is_t_goal"]))}
DIFFICULT = {"band": ["far"], "depth": ["2"], "between": ["3+"], "cohort": ["holdout"], "shape": ["box", "cylinder"], "K": ["0", "1", "2", "3"],
             "dest_is_t_goal": ["True"]}


def policy_table(ref, cands, roster):
    out = {}
    pos = [q for q, r in ref["recs"].items() if r.get("label") == 1]
    for name, run in cands.items():
        rs = run["recs"]
        miss = [q for q in pos if rs.get(q, {}).get("label") != 1]
        extra = [q for q, r in rs.items() if r.get("label") == 1 and ref["recs"].get(q, {}).get("label") != 1]
        strata = {}
        for s, f in STRATA.items():
            groups = defaultdict(lambda: [0, 0])
            for q in pos:
                g = groups[f(roster[q])]; g[0] += 1; g[1] += q in miss
            strata[s] = {k: {"ref_pos": n, "missed": m, "rate": round(m / n, 4), "wilson95": wilson(m, n)} for k, (n, m) in sorted(groups.items())}
        fails = [f"{s}={k}" for s, ks in DIFFICULT.items() for k in ks if k in strata[s] and strata[s][k]["ref_pos"] >= 10 and strata[s][k]["rate"] > 0.10]
        overall = len(miss) / len(pos)
        cpu = [r["cpu_query_s"] for r in rs.values()]
        out[name] = {"ref_positives": len(pos), "missed": len(miss), "miss_rate": round(overall, 4), "wilson95": wilson(len(miss), len(pos)),
                     "candidate_positive_ref_not": extra, "labels": Counter(f"{r.get('label')}:{r.get('negative_kind') or r.get('status')}" for r in rs.values()),
                     "mean_cpu_s_per_query": round(statistics.mean(cpu), 3), "total_cpu_h": round(sum(cpu) / 3600, 3),
                     "pass_overall": overall <= 0.05, "failing_difficult_strata": fails, "pass": overall <= 0.05 and not fails,
                     "strata": strata, "missed_qids": miss}
    passing = {k: v for k, v in out.items() if v["pass"]}
    choice = None
    if passing:
        cheapest = min(passing, key=lambda k: passing[k]["mean_cpu_s_per_query"])
        choice = "adaptive_v1" if "adaptive_v1" in passing and passing["adaptive_v1"]["mean_cpu_s_per_query"] <= 1.15 * passing[cheapest]["mean_cpu_s_per_query"] else cheapest
    return out, {"passing": sorted(passing), "choice": choice, "rule": "cheapest passing (mean real CPU-s/query); adaptive_v1 if within 1.15x"}


def accounting_audit(runs):
    out = {}
    for name, run in runs.items():
        rs = [r for r in run["recs"].values() if "accounting" in r]
        a = defaultdict(int)
        bad = []
        for r in rs:
            acc = r["accounting"]
            for k, v in acc.items():
                if isinstance(v, int):
                    a[k] += v
            ok = (acc["nominal"] == acc["screened"] + acc["unscreened"] and acc["branch_unaccounted"] == 0
                  and acc["screened"] == sum(acc[s.lower()] for s in Q.NOT_UNIT) + acc["valid_screen"])
            if r.get("label") == 1:
                ok = ok and r.get("witness_id") and r["witness_id"] in {w["witness_id"] for w in r["witnesses"]}
            if r.get("label") == 0 and r.get("status") not in ("NEG_CERTIFIED_GEOMETRY", "NEG_UNDER_POLICY"):
                ok = False
            if r.get("status", "").startswith("UNLABELED") and r.get("label") is not None:
                ok = False
            if not ok:
                bad.append(r["qid"])
        out[name] = {"queries": len(rs), "violations": bad, "sums": dict(a)}
    return out


def negative_audit(run):
    rs = list(run["recs"].values())
    cert = [r for r in rs if r.get("status") == "NEG_CERTIFIED_GEOMETRY"]
    neg = [r for r in rs if r.get("status") == "NEG_UNDER_POLICY"]
    return {"certified": len(cert), "certificates_complete": sum(r["certificate"]["poses_screened"] == 32 and r["certificate"]["geometry_reasons_footprint_only"]
                                                                and r["certificate"]["held_invalid_physical_all_branches_tested"] for r in cert),
            "under_policy": Counter(r["negative_kind"] for r in neg), "unlabeled": Counter(r["status"] for r in rs if r.get("label") is None),
            "exec_first_failed": Counter(res["exec"]["first_failed"] for r in rs for m in r.get("modes", []) for br in m["branches"] for res in br["results"]
                                         if res["state"] == "EXEC_FAILED" and "exec" in res)}


def storage(run):
    npz = sorted(run["dir"].glob("units/*.npz"))
    js = sorted(run["dir"].glob("units/*.json"))
    nw = sum(len(u.get("witness_index", {})) for u in run["units"].values())
    nb = sum(p.stat().st_size for p in npz)
    return {"witnesses": nw, "npz_bytes": nb, "json_bytes": sum(p.stat().st_size for p in js), "bytes_per_witness": round(nb / nw) if nw else None,
            "queries": len(run["recs"]), "json_bytes_per_query": round(sum(p.stat().st_size for p in js) / max(1, len(run["recs"])))}


def v1_v2(v1_dir, v2run, roster):
    v1 = {}
    for f in sorted(Path(v1_dir).glob("units/*.json")):
        for r in json.loads(f.read_text())["records"]:
            v1[r["qid"]] = r
    tr = Counter()
    ex = defaultdict(list)
    for q, r in v1.items():
        r2 = v2run["recs"].get(q)
        if r2 is None:
            continue
        k = f"{r.get('label')}:{r.get('status')} -> {r2.get('label')}:{r2.get('status')}"
        tr[k] += 1
        if r.get("label") != r2.get("label"):
            ex[k].append(q)
    return {"compared": sum(tr.values()), "transitions": dict(tr), "changed_examples": {k: v[:6] for k, v in ex.items()}}


def forecast(live_roster, cost_run, workers, eff, n_boot=2000, seed=0, bytes_per_witness=None, json_per_query=None, pos_rate=None):
    rows = [json.loads(x) for x in gzip.open(live_roster, "rt")]
    el = [x for x in rows if not x["static"]]
    cost = defaultdict(list)
    for r in cost_run["recs"].values():
        cost[("k", r["K"], r["depth"], r["band"])].append(r["cpu_query_s"])
        cost[("kd", r["K"], r["depth"])].append(r["cpu_query_s"]); cost[("K", r["K"])].append(r["cpu_query_s"]); cost[("all",)].append(r["cpu_query_s"])
    by_root = defaultdict(list)
    for r in cost_run["recs"].values():
        by_root[r["qid"].split(":")[0]].append(r)

    def est(c):
        tot, src = 0.0, Counter()
        for x in el:
            for key in (("k", x["K"], x["depth"], x["band"]), ("kd", x["K"], x["depth"]), ("K", x["K"]), ("all",)):
                if len(c.get(key, [])) >= 3:
                    tot += statistics.mean(c[key]); src[key[0]] += 1; break
        return tot, src
    point, src = est(cost)
    rng, boots, roots = random.Random(seed), [], sorted(by_root)
    for _ in range(n_boot):
        c = defaultdict(list)
        for rt in (rng.choice(roots) for _ in roots):
            for r in by_root[rt]:
                c[("k", r["K"], r["depth"], r["band"])].append(r["cpu_query_s"]); c[("kd", r["K"], r["depth"])].append(r["cpu_query_s"])
                c[("K", r["K"])].append(r["cpu_query_s"]); c[("all",)].append(r["cpu_query_s"])
        boots.append(est(c)[0])
    boots.sort()
    p05, p95 = boots[int(0.05 * n_boot)], boots[int(0.95 * n_boot)]
    wall = lambda cpu_s: cpu_s / (workers * eff) / 86400  # noqa: E731
    disk = (len(el) * (pos_rate or 0.7) * (bytes_per_witness or 0) + len(rows) * (json_per_query or 0)) / 1e9
    return {"production_queries": len(el), "static_unlabeled": len(rows) - len(el), "roots": len({x["root_id"] for x in rows}),
            "by_K": Counter(x["K"] for x in el), "by_split": Counter(x["split"] for x in el), "cost_source_levels": dict(src),
            "cpu_hours": round(point / 3600, 2), "cpu_hours_p05_p95": [round(p05 / 3600, 2), round(p95 / 3600, 2)],
            "workers": workers, "parallel_efficiency": round(eff, 3), "wall_days_expected": round(wall(point), 3), "wall_days_p95": round(wall(p95), 3),
            "wall_days_p95_x1_5": round(1.5 * wall(p95), 3), "disk_gb": round(disk, 3), "disk_gb_x1_5": round(1.5 * disk, 3),
            "assumptions": ["per-query CPU from the real 12-worker adaptive_v1 run A, grouped K x depth x band (>=3 samples), backed off to K x depth,"
                            " K, overall", "root-lineage bootstrap over the mock's roots", "NONE (selected) costed at the backed-off overall mean; K4 has 0"
                            " selected roots and is not in the forecast", "disk = queries x positive rate x measured bytes / witness + JSON bytes / query"]}


def sweep(mock):
    out = {}
    for n in (8, 12, 16, 18):
        d = Path(mock) / f"runs/sweep_{n}"
        if not (d / "ledger.jsonl").exists():
            continue
        run = load_run(d)
        v = validity(run)
        q = len(run["recs"])
        out[str(n)] = {"queries": q, "wall_s": round(v["wall_s"], 1), "q_per_wall_s": round(q / v["wall_s"], 4), "cpu_s": round(v["cpu_s"], 1),
                       "cpu_per_query": round(v["cpu_s"] / q, 3), "violations": v["monitor_violations"], "peak": v["peak"]}
    return out


def analyze(mock, live_roster, out):
    mock = Path(mock).resolve()
    roster = {x["qid"]: x for x in (json.loads(l) for l in gzip.open(mock / "roster_v2.jsonl.gz", "rt"))}
    names = ["reference_v2", "adaptive_v1_A", "adaptive_v1_B", "fixed_4x1500", "fixed_4x5000", "fixed_8x5000", "serial_adaptive", "adaptive_v1_interrupt"]
    runs = {n: load_run(mock / "runs" / n) for n in names if (mock / "runs" / n / "ledger.jsonl").exists()}
    res = {"validity": {n: validity(r) for n, r in runs.items()}}
    A, ref = runs["adaptive_v1_A"], runs["reference_v2"]
    det = {}
    if "adaptive_v1_B" in runs:
        det["A_vs_B"] = compare(A, runs["adaptive_v1_B"])
    if "serial_adaptive" in runs:
        det["serial_vs_A"] = compare(A, runs["serial_adaptive"], XCOMP)
    if "adaptive_v1_interrupt" in runs:
        det["interrupt_vs_A"] = compare(A, runs["adaptive_v1_interrupt"])
    for n in (8, 12, 16, 18):
        d = mock / f"runs/sweep_{n}"
        if (d / "ledger.jsonl").exists():
            det[f"sweep_{n}_vs_A"] = compare(A, load_run(d), XCOMP)
    real = {"adaptive_v1": A, **{k: runs[k] for k in ("fixed_4x1500", "fixed_4x5000", "fixed_8x5000") if k in runs}}
    det["virtual_vs_real"] = {k: virtual_vs_real(ref, v, (Q.POLICIES | Q.FIXED)[k]) for k, v in real.items()}
    res["determinism"] = det
    res["policy_table"], res["policy_choice"] = policy_table(ref, real, roster)
    res["accounting"] = accounting_audit(runs)
    res["negative_audit"] = {n: negative_audit(runs[n]) for n in ("reference_v2", "adaptive_v1_A")}
    res["storage"] = {n: storage(r) for n, r in runs.items()}
    res["v1_to_v2"] = {"pilot_A_vs_adaptive_v1_A": v1_v2(mock.parent.parent / "dataset_placement_extension_v1/mock_v1/runs/pilot_A", A, roster)}
    res["branch_effects"] = {n: {k: res["accounting"][n]["sums"].get(k, 0) for k in ("units_verified_on_later_branch", "units_rescued_after_hold_failure",
                                 "units_rescued_after_exec_failure", "units_rescued_after_plan_failure", "units_first_eligible_hold_failed_later_held",
                                 "held_invalid_physical", "branch_candidates", "branch_eligible", "branch_hold_tested", "branch_held", "branch_hold_failed",
                                 "branch_attempts", "branch_exec_attempts", "branch_verified", "verified_witnesses")} for n in runs}
    res["coverage"] = {s: dict(Counter(f"{STRATA[s](roster[q])}:{r.get('label')}" for q, r in A["recs"].items())) for s in ("K", "depth", "band", "shape", "between", "cohort")}
    res["sweep"] = sweep(mock)
    sw = res["sweep"]
    ser = res["validity"].get("serial_adaptive")
    eff = 1.0
    if "12" in sw and ser:
        eff = min(1.0, (sw["12"]["q_per_wall_s"] / 12) / (len(runs["serial_adaptive"]["recs"]) / ser["wall_s"]))
    st = res["storage"]["adaptive_v1_A"]
    pos_rate = sum(r.get("label") == 1 for r in A["recs"].values()) / len(A["recs"])
    res["forecast_live"] = {k: forecast(live_roster, runs[k] if k != "adaptive_v1" else A, 12, eff, bytes_per_witness=st["bytes_per_witness"],
                                        json_per_query=st["json_bytes_per_query"], pos_rate=pos_rate)
                            for k in ("adaptive_v1", res["policy_choice"]["choice"]) if k and (k == "adaptive_v1" or k in runs)}
    res["forecast_live"]["note"] = {"parallel_efficiency_from": "sweep_12 q/s per worker vs serial q/s", "positive_rate_adaptive_A": round(pos_rate, 4)}
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    print(json.dumps({"validity": {n: (v["complete"], len(v["units_failing_validation"]), v["monitor_violations"]) for n, v in res["validity"].items()},
                      "determinism": {k: (v if "mismatches" not in v else v["mismatches"]) for k, v in det.items()}, "choice": res["policy_choice"],
                      "miss": {k: (v["missed"], v["ref_positives"], v["failing_difficult_strata"], v["mean_cpu_s_per_query"]) for k, v in res["policy_table"].items()}},
                     default=str)[:3000])


def v1v2_only(mock, an_path):
    """Recompute only the V1 -> V2 regression section of an existing analysis (path fix; everything else unchanged)."""
    mock = Path(mock).resolve()
    roster = {x["qid"]: x for x in (json.loads(l) for l in gzip.open(mock / "roster_v2.jsonl.gz", "rt"))}
    an = json.loads(Path(an_path).read_text())
    v1 = mock.parent.parent / "dataset_placement_extension_v1/mock_v1/runs"
    an["v1_to_v2"] = {"pilot_A_vs_adaptive_v1_A": v1_v2(v1 / "pilot_A", load_run(mock / "runs/adaptive_v1_A"), roster),
                      "v1_reference_vs_reference_v2": v1_v2(v1 / "reference", load_run(mock / "runs/reference_v2"), roster)}
    Path(an_path).write_text(json.dumps(an, indent=1, sort_keys=True, default=lambda o: dict(o) if isinstance(o, Counter) else str(o)))
    print(json.dumps(an["v1_to_v2"])[:3000])


if __name__ == "__main__":
    if sys.argv[1] == "analyze":
        analyze(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "v1v2":
        v1v2_only(sys.argv[2], sys.argv[3])
