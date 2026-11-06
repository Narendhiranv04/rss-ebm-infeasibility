"""Apply the PRE-REGISTERED pilot selection rule (configs/pilot_policy_selection.json, frozen before any pilot call) to the pilot
evidence; forecast the full run; freeze configs/production_policy.json only when every gate passes.

    python launch_decide.py gates <launch_dir> <run_a> <run_b>     # gates.json; run_a, run_b = two audited 25-worker 64-pair runs
    python launch_decide.py decide <launch_dir> <budget.json> <run64_25w> <soak_run_25w> <gates.json>
"""
import gzip, io, json, sys, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_check as K  # noqa: E402
import launch_plan as LP  # noqa: E402
import placement_teacher as T  # noqa: E402
P = T.P


def label_rule(b, rule):
    lr, out = rule["label_rule"], {}
    for B in rule["candidates"]:
        t = b["budgets"][str(B)]
        bad = [f"overall miss {t['miss_frac']}"] if (t["miss_frac"] or 0) > lr["max_miss_overall"] else []
        for dim, keys in lr["strata"].items():
            for k in keys:
                s = t["strata"][dim].get(k)
                if s and s["pos64"] >= lr["min_positives_for_stratum_test"] and s["miss_frac"] > lr["max_miss_per_stratum"]:
                    bad.append(f"{dim}={k} miss {s['miss_frac']} of {s['pos64']}")
        out[B] = {"passes": not bad and b["monotone_no_1_to_0"], "violations": bad}
    sel = next((B for B in rule["candidates"] if out[B]["passes"]), None)
    return sel, out


def throughput(soak):
    """Sustained worker CPU-s per wall-s while the queue was non-empty (thermal pauses included)."""
    m = [json.loads(x) for x in open(Path(soak) / "monitor.jsonl")]
    w = [x for x in m if x.get("queued", 0) > 0 and x.get("workers_cpu_s") is not None]
    if len(w) >= 2:
        return (w[-1]["workers_cpu_s"] - w[0]["workers_cpu_s"]) / (w[-1]["t"] - w[0]["t"]), "monitor slope while queue non-empty"
    r = json.loads(sorted(Path(soak).glob("run_*.json"))[-1].read_text())
    return r["children_cpu_s"] / r["wall_s"], "children CPU / wall (includes ramp and tail)"


def bytes_model(run64, B, cov):
    """Bytes per plan row when the pilot roots are re-packed exactly as production packs them under (B, cov, witness_store_max=1):
    one records JSON, one modes NPZ and one witness NPZ per root (+ half a 4 KiB block per file). -> (bytes per row, bytes per witness)."""
    by = defaultdict(list)
    for h, q, m, a in K.load(run64):
        by[h["root_id"]].append((q, m, a))
    tot = rows = nw = wbytes = 0
    for rid, rs in by.items():
        recs, arrs, ws = [], defaultdict(list), []
        for q, m, a in rs:
            rows += 1
            if not (q["eligible"] and q["completed"]):
                recs.append(q)
                continue
            v = K.virtual(q, a, B, cov)
            keep = (a["a_rank"] < 0) | ((a["a_rank"] >= 0) & (a["a_rank"] < v["planned"]))
            for k in K.MODE_KEYS:
                arrs[k].append(m[k])
            for k in T.ATT:
                arrs["c_" + k].append(a[k][keep])
            recs.append({**q, "volatile": {**q["volatile"], "attempt_cpu_s": q["volatile"]["attempt_cpu_s"][:v["planned"]]}, "witness_refs": q["witness_refs"][:1]})
            if v["label"] and q["witness_refs"]:
                ws.append(T.witness(run64, rid, q["witness_refs"][0]))
        mb, wb = io.BytesIO(), io.BytesIO()
        np.savez_compressed(mb, **{k: np.concatenate(v) if k.startswith("c_") else np.stack(v) for k, v in arrs.items()})
        if ws:
            np.savez_compressed(wb, q=np.concatenate([w["q"] for w in ws]), gap=np.concatenate([w["gap"] for w in ws]), attached=np.concatenate([w["attached"] for w in ws]),
                                phase_code=np.concatenate([[T.PHASES.index(str(x)) for x in w["phase"]] for w in ws]).astype(np.int8),
                                held_local=np.stack([w["held_local"] for w in ws]), place_xy=np.stack([w["place_xy"] for w in ws]))
            nw, wbytes = nw + len(ws), wbytes + len(wb.getvalue())
        tot += len(json.dumps({"queries": recs}, separators=(",", ":"))) + len(mb.getvalue()) + len(wb.getvalue()) + 3 * 2048
    return tot / rows, wbytes / max(nw, 1)


def gates(L, a, b):
    """determinism (a = b = serial reference on shared queries), nested, interrupt + resume, 25-worker safety of a and b (no safety
    stop in any of their sessions), and zero audit failures. Evidence: audit/pilot/gates_evidence.json."""
    import glob
    L = Path(L)
    A, PL = L / "audit" / "pilot", L / "pilot"
    runs = [Path(a), Path(b)]
    c3 = K.compare(A / f"compare_{runs[0].name}_{runs[1].name}_serial.json", [*runs, PL / "pilot_b64_serial_first"])
    c2 = K.compare(A / f"compare_{runs[0].name}_{runs[1].name}.json", runs)
    ci, n = json.loads((A / "compare_b8_vs_interrupt.json").read_text()), json.loads((A / "nested_b64r1_vs_b8.json").read_text())
    it = json.loads((A / "interrupt_check.json").read_text())
    names = {*runs, PL / "pilot_b64_serial_first", PL / "pilot_b8_w25", PL / "pilot_b8_w25_interrupt"}
    aud = {r.name: json.loads((r / "verification.json").read_text())["failed"] for r in names}
    stops = {r.name: [json.loads(Path(f).read_text())["stop"] for f in sorted(glob.glob(str(r / "run_1*.json")))] for r in runs}
    g = {"determinism": c3["mismatches"] == 0 and c3["shared_queries"] == 25 and c2["mismatches"] == 0 and c2["shared_queries"] == 347,
         "nested": n["compared"] == 300 and n.get("different", 0) == 0,
         "interrupt_resume": ci["mismatches"] == 0 and ci["shared_queries"] == 347 and it.get("packed_changed", 0) == 0 and it.get("committed_part_changed", 0) == 0,
         "safety_25": all(s is None for v in stops.values() for s in v), "audits": all(v == 0 for v in aud.values())}
    (A / "gates.json").write_text(json.dumps(g, indent=1))
    (A / "gates_evidence.json").write_text(json.dumps({"runs_judged": [str(r) for r in runs], "audit_failures": aud, "safety_stops_per_session": stops,
        "compare_with_serial": {k: c3[k] for k in ("shared_queries", "mismatches")}, "compare_pair": {k: c2[k] for k in ("shared_queries", "mismatches")},
        "compare_b8_interrupt": {k: ci[k] for k in ("shared_queries", "mismatches")}, "nested": {k: v for k, v in n.items() if k != "different_ids"}, "interrupt": it}, indent=1))
    return g


def decide(L, bj, run64, soak, gates_f):
    L = Path(L)
    rule = json.loads((L / "configs" / "pilot_policy_selection.json").read_text())
    b, gates = json.loads(Path(bj).read_text()), json.loads(Path(gates_f).read_text())
    sel, per_b = label_rule(b, rule)
    neg = b["budgets"][str(sel)]["negatives"] if sel else None
    g = rule["negative_diversity_gate"]
    neg_ok = bool(neg) and neg["queries"] >= g["min_label0"] and neg["lineage_groups"] >= g["min_lineage_groups"] and len(neg["classes"]) >= g["min_classes"]
    rate, how = throughput(soak)
    meta = b["row_meta"]
    plan = [json.loads(x) for x in gzip.decompress((L / "query_plan.jsonl.gz").read_bytes()).decode().splitlines()]
    free = psutil.disk_usage(str(L)).free / 2 ** 20
    budget_mb = free - 1536
    ladder = []
    for cov, qn in rule["workload_rule"]["ladder"]:
        if sel is None:
            break
        cpu = defaultdict(list)
        for (rid, blk, dbin, sbin, cls), v in zip(meta, b["virtual_rows"][f"{sel}/{cov}"]):
            cpu[dbin].append(v[1])
        mean = {k: float(np.mean(v)) for k, v in cpu.items()}
        comp = Counter(r[8] for p in plan for r in p["queries"] if LP.in_quota(r, LP.QUOTAS[qn]) and not r[12])
        n_static = sum(1 for p in plan for r in p["queries"] if LP.in_quota(r, LP.QUOTAS[qn]) and r[12])
        cpu_h = sum(n * mean.get(k, max(mean.values())) for k, n in comp.items()) / 3600
        per_q, w = bytes_model(run64, sel, cov)
        roots = sum(1 for p in plan if any(LP.in_quota(r, LP.QUOTAS[qn]) for r in p["queries"]))
        days = cpu_h * 3600 / rate / 86400
        disk = per_q * (sum(comp.values()) + n_static) / 2 ** 20 + days * 86400 / 2 * 300 / 2 ** 20     # + monitor log (300 B / 2 s)
        ladder.append({"coverage": cov, "quota": qn, "teacher_queries": sum(comp.values()), "static_rows": n_static, "roots": roots,
                       "cpu_h": round(cpu_h, 1), "wall_days": round(days, 2), "disk_mb": round(disk, 1), "bytes_per_row_packed": round(per_q),
                       "bytes_per_witness": round(w), "fits": days <= rule["workload_rule"]["max_projected_wall_days"] and disk <= budget_mb})
    pick = next((x for x in ladder if x["fits"]), None)
    verdict_gates = {"label_rule_selects_budget": sel is not None, "negative_diversity": neg_ok, "workload_fits": pick is not None, **gates}
    out = {"version": "PILOT_DECISION_V1", "decided": time.strftime("%Y-%m-%dT%H:%M:%S"), "rule_sha256": P.sha_file(L / "configs" / "pilot_policy_selection.json"),
           "selected_pairs": sel, "per_budget": per_b, "negatives_at_selected": neg, "throughput_worker_cpu_s_per_s": round(rate, 2), "throughput_basis": how,
           "disk_free_mb": round(free), "disk_budget_mb": round(budget_mb), "ladder": ladder, "chosen": pick, "gates": verdict_gates,
           "ready": all(verdict_gates.values())}
    (L / "audit" / "pilot_decision.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    if out["ready"]:
        pol = {"pairs_per_query": sel, "coverage": pick["coverage"], "witness_store_max": rule["workload_rule"]["production_witness_store_max"], "reuse_archived": True}
        prod = {"version": "PRODUCTION_POLICY_2D_V1", "frozen": out["decided"], "policy": pol, "budget_hash": T.budget_hash(pol), "quota": pick["quota"],
                "quota_spec": LP.QUOTAS[pick["quota"]], "forecast": {k: pick[k] for k in ("teacher_queries", "static_rows", "roots", "cpu_h", "wall_days", "disk_mb")},
                "decision_file_sha256": P.sha_file(L / "audit" / "pilot_decision.json"), "note": "B is a finite operational policy; zeros are policy-limited, not physical impossibility"}
        f = L / "configs" / "production_policy.json"
        f.write_text(json.dumps(prod, indent=1, sort_keys=True))
        fz = json.loads((L / "configs" / "frozen_hashes.json").read_text())
        fz["production_policy_sha256"] = P.sha_file(f)
        (L / "configs" / "frozen_hashes.json").write_text(json.dumps(fz, indent=1, sort_keys=True))
    return out


if __name__ == "__main__":
    f = gates if sys.argv[1] == "gates" else decide
    print(json.dumps(f(*sys.argv[2:]), indent=1, sort_keys=True))
