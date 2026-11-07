#!/usr/bin/env python3
"""Q2 production amendment (Q2_PRODUCTION_AMENDMENT_V1) of the 2D placement v2 launch decision.  Explicit, versioned, traceable.

History kept intact: the pre-registered reviewed gate (audit/package/reviewed/mock_v2_gate.py) decided on Q1 and returned
NOT_READY_TO_LAUNCH for ONE gate only, RUNTIME_FORECAST (Q1 357,240 queries = 7.62 d with the 1.5x buffer > 7 d).  That decision file
(MOCK_V2_DECISION.json) is not modified.  After that verdict the user chose the smaller predeclared quota Q2 (4 root objects x 8
destinations; 3 prefix blocks x 2 unmoved objects x 4 destinations) and authorized it explicitly.  Q2 was NOT pre-registered as the
production quota; it is one of the four quotas launch_plan.QUOTAS defined before the pilot.

This tool re-runs EVERY original evidence gate of the reviewed decide() on the stored pilot runs (no new pilot), using the reviewed
module's own setup / static / compare_adaptive (its file hash is checked against the reviewed CHECKSUMS), and changes only the quota:
the composition-weighted forecast is recomputed over Q2's actual (block, dest_bin) mix with the same estimator rule
max(pilot mean, composition weighted) x 1.5, the same measured 25-worker boost-off throughput and the same <= 7.0 d limit.
The production policy is then frozen with quota Q2 (fails closed if one already exists).

    python q2_amendment.py <repo> decide        (repo = /home/naren/Desktop/rss-placement-ready-v2)
"""
import hashlib, json, shutil, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
L0 = HERE.parent
REVIEWED = L0 / "audit/package/reviewed"
QUOTA = "Q2"
AMENDMENT = "Q2_PRODUCTION_AMENDMENT_V1"


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def reviewed_module():
    # The reviewed gate (delivered package + review fixes R1-R5) is registered in audit/checksums.sha256; reviewed/CHECKSUMS.sha256 is a
    # copy of the DELIVERED package's checksums and lists the pre-review file.
    reg = {ln.split()[1]: ln.split()[0] for ln in (L0 / "audit/checksums.sha256").read_text().splitlines() if ln.strip()}
    want, got = reg.get("./audit/package/reviewed/mock_v2_gate.py"), sha(REVIEWED / "mock_v2_gate.py")
    if want != got:
        raise SystemExit(f"FAIL_CLOSED: reviewed mock_v2_gate.py hash {got[:12]} != audit/checksums.sha256 {str(want)[:12]}")
    sys.path.insert(0, str(REVIEWED))
    import mock_v2_gate as G
    return G, got


def composition_forecast(K, LP, L, run, quota):
    """The reviewed composition_forecast with the quota as a parameter (identical estimator otherwise)."""
    import gzip
    from collections import Counter, defaultdict
    import numpy as np
    recs = [(h, q, att) for h, q, m, att in K.load(run) if q["eligible"] and q["completed"]]
    files = {d: sum(f.stat().st_size for f in Path(run).glob(f"{d}/*")) for d in ("records", "modes", "witnesses")}
    nwit = sum(len(q["witness_refs"]) for _, q, _ in recs)
    per_w, base = files["witnesses"] / max(nwit, 1), (files["records"] + files["modes"]) / len(recs)
    cpu, byt, cpu_d, byt_d = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for h, q, att in recs:
        k, b = (q["block"], q["dest_bin"]), base + per_w * len(q["witness_refs"])
        cpu[k].append(q["volatile"]["cpu_s"]); byt[k].append(b); cpu_d[q["dest_bin"]].append(q["volatile"]["cpu_s"]); byt_d[q["dest_bin"]].append(b)
    plan = [json.loads(x) for x in gzip.decompress((L / "query_plan.jsonl.gz").read_bytes()).decode().splitlines()]
    comp = Counter((r[1], r[8]) for p in plan for r in p["queries"] if LP.in_quota(r, LP.QUOTAS[quota]) and not r[12])
    pick = lambda d, dd, k: float(np.mean(d[k])) if len(d[k]) >= 5 else float(np.mean(dd[k[1]]))  # noqa: E731
    strata = {f"{b}/{d}": {f"{quota.lower()}_queries": n, "pilot_n": len(cpu[(b, d)]), "cpu_s": round(pick(cpu, cpu_d, (b, d)), 2),
                           "estimate": "stratum mean" if len(cpu[(b, d)]) >= 5 else "pooled dest_bin mean (conservative fallback, < 5 pilot queries)"}
              for (b, d), n in sorted(comp.items())}
    return {"cpu_s": sum(n * pick(cpu, cpu_d, k) for k, n in comp.items()), "bytes": sum(n * pick(byt, byt_d, k) for k, n in comp.items()),
            "strata": strata, "teacher_queries": sum(comp.values())}


def decide(repo):
    G, gate_sha = reviewed_module()
    root, code, L, LP, R, K, Gd, T = G.setup(repo)
    assert L.resolve() == L0.resolve(), (L, L0)
    G.assert_no_full(root)
    pre = L / "mock_static_report.json"
    if pre.exists():                                              # the reviewed static() rewrites this file: keep the earlier one
        shutil.copyfile(pre, HERE / "pre_amendment_mock_static_report.json")
    G.static(repo, strict_runtime=True)
    shutil.copyfile(pre, HERE / "amendment_static_report.json")
    hist = L / "MOCK_V2_DECISION.json"
    old = json.loads(hist.read_text())
    pilots = [L / "pilot" / n for n in G.NAMES]
    gate, failures = {}, []
    for p in pilots + [L / "pilot" / "v2_adaptive_interrupted"]:
        audit = G.djson(p / "verification.json")
        stops = [G.djson(f)["stop"] for f in p.glob("run_[0-9]*.json")]
        gate[p.name + "_audit"] = audit["failed"] == 0
        gate[p.name + "_safety"] = all(x is None for x in stops) if p.name != "v2_adaptive_interrupted" else True
        failures += [p.name + ":AUDIT"] * (not gate[p.name + "_audit"]) + [p.name + ":SAFETY"] * (not gate[p.name + "_safety"])
    a, b = pilots[1:3]
    af, bf = G.djson(a / "fingerprint.json")["queries"], G.djson(b / "fingerprint.json")["queries"]
    gate["independent_25_worker_repro"] = af == bf
    serial = G.djson(pilots[3] / "fingerprint.json")["queries"]
    gate["serial_matches_parallel"] = all(t in af and af[t] == x for t, x in serial.items())
    gate["interrupt_matches"] = G.djson(L / "pilot/v2_adaptive_interrupted/fingerprint.json")["queries"] == af
    nested = G.compare_adaptive(K, pilots[0], a)
    gate["adaptive_nested_ref64"] = not nested.get("bad")
    refq = {q["trial_id"]: q for h, q, m, att in K.load(pilots[0]) if q.get("eligible") and q.get("completed")}
    adq = {q["trial_id"]: q for h, q, m, att in K.load(a) if q.get("eligible") and q.get("completed")}
    pos64 = [tid for tid, q in refq.items() if q["label"] == 1]
    missed = [tid for tid in pos64 if adq.get(tid, {}).get("label") != 1]

    def strata_stats(field):
        out = {}
        for k in {q[field] for q in refq.values()}:
            ids = [tid for tid in pos64 if refq[tid][field] == k]
            nbad = sum(tid in set(missed) for tid in ids)
            out[str(k)] = {"pos64": len(ids), "missed": nbad, "miss_frac": nbad / len(ids) if ids else None}
        return out
    miss_frac = len(missed) / len(pos64) if pos64 else 1.0
    strats = {"block": strata_stats("block"), "dest_bin": strata_stats("dest_bin")}
    gate["adaptive_recall_vs_64"] = miss_frac <= 0.02 and all(v["pos64"] < 20 or v["miss_frac"] <= 0.05 for d in strats.values() for v in d.values())
    total = G.djson(a / "summary.json")
    neg = [(h, q) for h, q, m, att in K.load(a) if q.get("completed") and q.get("eligible") and q.get("label") == 0]
    gate["negative_diversity"] = len(neg) >= 10 and len({h["root_group_id"] for h, q in neg}) >= 5 and \
        len({h["privileged_teacher"]["verified_class"] for h, q in neg}) >= 3
    miss_w = [q["trial_id"] for h, q, mode, att in K.load(a) if q.get("eligible") and q.get("completed") and
              {str(w) for v, w in zip(att["a_verified"], att["a_witness"]) if bool(v)} != set(map(str, q["witness_refs"]))]
    gate["all_verified_witnesses_stored"] = not miss_w
    for k, f in (("independent_25_worker_repro", "ADAPTIVE_25_WORKER_FINGERPRINTS_DIFFER"), ("serial_matches_parallel", "SERIAL_PARALLEL_DIFFER"),
                 ("interrupt_matches", "INTERRUPT_RESUME_DIFFER"), ("adaptive_nested_ref64", "ADAPTIVE_REF64_MISMATCH"),
                 ("adaptive_recall_vs_64", "ADAPTIVE_MISSED_TOO_MANY_64_REFERENCE_POSITIVES"), ("negative_diversity", "NEGATIVE_DIVERSITY"),
                 ("all_verified_witnesses_stored", "WITNESS_LOSS")):
        if not gate[k]:
            failures.append(f)
    nq = G.djson(L / "plan_summary.json")["quota_teacher_queries"][QUOTA]
    pilot_q = total["completed"]
    cpu_mean = float(total["query_cpu_s_sum"]) / pilot_q * nq * 1.5
    disk_mean = sum(total["bytes"].values()) / pilot_q * nq * 1.5
    comp = composition_forecast(K, LP, L, a, QUOTA)
    assert comp["teacher_queries"] == nq, (comp["teacher_queries"], nq)
    cpu_sec, disk_bytes = max(cpu_mean, comp["cpu_s"] * 1.5), max(disk_mean, comp["bytes"] * 1.5)
    disk_mb = disk_bytes / 1024 ** 2 + 500
    mon = [json.loads(x) for x in (a / "monitor.jsonl").read_text().splitlines() if x.strip()]
    busy = [x for x in mon if x.get("queued", 0) > 0 and x.get("workers_cpu_s") is not None]
    speed = (busy[-1]["workers_cpu_s"] - busy[0]["workers_cpu_s"]) / (busy[-1]["t"] - busy[0]["t"])
    days = cpu_sec / max(.1, speed) / 86400
    free_mb = shutil.disk_usage(L).free / 1024 ** 2
    gate["disk_safety"] = free_mb >= max(10240, disk_mb + 1536)
    gate["runtime_forecast"] = days <= 7.0
    failures += [k.upper() for k in ("disk_safety", "runtime_forecast") if not gate[k]]
    same = {k: gate[k] == old["gates"][k] for k in old["gates"] if k != "runtime_forecast"}
    report = {"amendment": AMENDMENT, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "verdict": "READY_TO_LAUNCH_Q2" if not failures else "NOT_READY_TO_LAUNCH", "quota": QUOTA, "quota_spec": LP.QUOTAS[QUOTA],
              "gates": gate, "failures": failures, "evidence_gates_identical_to_historical_q1_decision": same,
              "historical_q1_decision": {"file": "MOCK_V2_DECISION.json", "sha256": sha(hist), "verdict": old["verdict"], "failures": old["failures"],
                                         "q1_wall_days_with_1p5_buffer": old["forecast"]["wall_days_with_1p5_buffer"], "kept_unmodified": True},
              "reason": ("The pre-registered Q1 decision failed only RUNTIME_FORECAST (7.62 d > 7 d with the 1.5x buffer). The user chose the "
                         "smaller predeclared quota Q2 and authorized it explicitly; every other evidence gate is re-run unchanged on the stored pilots."),
              "user_authorization": ("I authorise starting the 2D Q2 placement dataset collection on my laptop: 238,160 physical placement queries, "
                                     "ADAPTIVE_FULL 8→32 valid mode pairs, 25 workers, the existing verified boost-OFF thermal/safety configuration."),
              "reviewed_gate_sha256": gate_sha, "nested": nested,
              "relative_to_reference_64": {"positive_reference_count": len(pos64), "adaptive_missed": len(missed), "miss_fraction": miss_frac,
                                           "wilson_95": K.wilson(len(missed), len(pos64)), "strata": strats},
              "pilot": {"completed": pilot_q, "positives": total["label_1"], "negatives": total["label_0"], "stored_witnesses": total["stored_witnesses"]},
              "forecast": {f"{QUOTA.lower()}_teacher_queries": nq, "cpu_hours_with_1p5_buffer": round(cpu_sec / 3600, 1),
                           "wall_days_with_1p5_buffer": round(days, 2), "wall_days_expected_no_buffer": round(comp["cpu_s"] / speed / 86400, 2),
                           "disk_mb_with_1p5_buffer_and_500mb_overhead": round(disk_mb, 1), "free_mb": round(free_mb, 1),
                           "pilot_throughput_cpu_s_per_s": round(speed, 2),
                           "estimators": {"pilot_mean_cpu_hours_x1p5": round(cpu_mean / 3600, 1),
                                          "composition_weighted_cpu_hours_x1p5": round(comp["cpu_s"] * 1.5 / 3600, 1),
                                          "strata": comp["strata"], "rule": f"max(pilot mean, {QUOTA}-composition weighted) x 1.5 (reviewed R5 rule)"}}}
    (HERE / "Q2_AMENDMENT_DECISION.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    if failures:
        print(json.dumps(report, indent=1)); raise SystemExit("NOT_READY_TO_LAUNCH under the Q2 amendment; thresholds unchanged")
    pol = {"pairs_per_query": 8, "rescue_pairs": 32, "coverage": "ADAPTIVE_FULL", "witness_store_max": 32, "reuse_archived": False}
    full = {"version": "PRODUCTION_POLICY_2D_V2", "policy": pol, "quota": QUOTA, "quota_spec": LP.QUOTAS[QUOTA], "budget_hash": T.budget_hash(pol),
            "forecast": {"teacher_queries": nq, "disk_mb": disk_mb, "wall_days": days},
            "decision_sha256": sha(HERE / "Q2_AMENDMENT_DECISION.json"), "amendment": f"{AMENDMENT}: amendment_q2/Q2_AMENDMENT_DECISION.json"}
    pf = L / "configs/production_policy.json"
    if pf.exists():
        raise SystemExit("FAIL_CLOSED: production policy already exists")
    pf.write_text(json.dumps(full, indent=1, sort_keys=True))
    fz = G.djson(L / "configs/frozen_hashes.json")
    fz["production_policy_sha256"] = sha(pf)
    fz["production_policy_amendment"] = {"id": AMENDMENT, "decision": "amendment_q2/Q2_AMENDMENT_DECISION.json",
                                         "decision_sha256": full["decision_sha256"], "quota": QUOTA, "historical_q1_decision_sha256": sha(hist)}
    (L / "configs/frozen_hashes.json").write_text(json.dumps(fz, indent=1, sort_keys=True))
    print(json.dumps({k: report[k] for k in ("verdict", "quota", "failures", "forecast")}, indent=1, default=str)[:3000])
    print(f"FROZEN production policy (quota {QUOTA}); the full job is NOT started by this tool.")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[2] != "decide":
        raise SystemExit("usage: q2_amendment.py <repo> decide")
    decide(Path(sys.argv[1]))
