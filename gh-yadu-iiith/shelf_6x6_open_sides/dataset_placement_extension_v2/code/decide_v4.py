"""Derive production_v4/V4_READINESS_DECISION.json mechanically from saved evidence (missing or failing evidence = failed gate).
  SHELF_SRC=... python decide_v4.py OUT.json
Verdicts: 2D_Q2_CURRENT_STATE, 3D_V3_CURRENT_STATE, 3D_Q2_PARITY_ROSTER, SOURCE_QC, SPLIT_LEAKAGE, 3D_PHYSICAL_INTEGRATION, WITNESS_REPLAY,
PRODUCTION_GUARDS, STATUS_SERVICE, OFFHOST_BACKUP, and 3D_V4 = 3D_V4_READY_PENDING_NEW_USER_AUTHORIZATION only if every required gate passes.
"""
import gzip, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import prod4  # noqa: E402,F401
import prod3d  # noqa: E402

V3, V4 = ROOT / "production_v3", ROOT / "production_v4"


def load(p):
    return json.loads(p.read_text()) if p.exists() else None


def passed(p):
    t = p.read_text() if p.exists() else ""
    return " passed" in t and " failed" not in t and " error" not in t, (t.strip().splitlines() or ["missing"])[-1]


def main(out):
    g, ev = {}, {}
    cov, man = load(V4 / "COVERAGE_REPORT_V4.json") or {}, load(V4 / "frozen/roster_manifest_v4.json") or {}
    pre = load(V4 / "integration/production_metadata_preflight_v4.json") or {}
    it, fc = load(V4 / "integration/integration_results_v4.json") or {}, load(V4 / "frozen/forecast_v4.json") or {}
    sums = {ln.split("  ", 1)[1]: ln.split("  ", 1)[0] for ln in (V3 / "SHA256SUMS").read_text().splitlines()}
    rost = [json.loads(x) for x in gzip.open(V4 / "frozen/q2_parity_roster_v4.jsonl.gz", "rt")]
    phys = [x for x in rost if not x["static"]]
    qc = load(V3 / "QC_ELIGIBLE_ROOTS.json")["roots"]
    # source QC: the v3 QC / pin / split files are byte-identical to their frozen v3 hashes and every v4 roster root is eligible
    frozen_same = {f: prod3d.sha_file(V3 / f) == sums.get(f"./{f}") for f in ("QC_ELIGIBLE_ROOTS.json", "frozen/source_pin_first_cohort.json",
                                                                            "frozen/split_manifest_first_cohort.json")}
    ev["source_qc"] = {"v3_files_unchanged": frozen_same, "roster_roots_not_eligible": sorted({x["root_id"] for x in rost if not qc[x["root_id"]]["eligible"]})[:5],
                       "manifest_hashes_match": all(man.get("sha256", {}).get(k) == prod3d.sha_file(ROOT / v) for k, v in man.get("files", {}).items())}
    g["source_qc"] = all(frozen_same.values()) and not ev["source_qc"]["roster_roots_not_eligible"] and ev["source_qc"]["manifest_hashes_match"]
    gates = cov.get("gates", {})
    ev["roster"] = {"physical_queries": len(phys), "theoretical": cov.get("theoretical_max_before_destination_availability"), "static": len(rost) - len(phys),
                    "gates": gates, "regenerated_equal_in_preflight": pre.get("roster_regenerated_equal"), "by_block": cov.get("by_block"),
                    "by_band": cov.get("by_band"), "by_split": cov.get("by_split"), "by_K": cov.get("by_K"), "by_shape": cov.get("by_shape")}
    g["roster"] = bool(gates) and all(gates.values()) and len(phys) == cov.get("physical_queries") and pre.get("roster_regenerated_equal") is True \
        and pre.get("semantic_keys_unique") is True
    g["split_leakage"] = all(gates.get(k) for k in ("split_inherited_unchanged", "semantic_keys_single_split", "scene_layouts_single_split"))
    sb, sr = it.get("smoke_B_interrupt") or {}, it.get("smoke_B_resume") or {}
    rA, rB = it.get("replay_A") or {}, it.get("replay_B") or {}
    mA, mB = it.get("monitor_A") or {}, it.get("monitor_B") or {}
    ev["integration"] = {k: it.get(k) for k in ("smoke_A", "smoke_B_interrupt", "smoke_B_resume", "B_equals_A", "labels_by_block_A", "labels_by_band_A",
                                                "semantics_problems_A", "cpu_s_per_query_A", "marked_A", "monitor_A", "monitor_B")}
    g["physical_integration"] = ((it.get("audit_A") or {}).get("complete") is True and (it.get("audit_B") or {}).get("complete") is True
                                 and (it.get("B_equals_A") or {}).get("equal") is True and sb.get("descendants_alive_3s_after_kill") == []
                                 and "lock_held_after_kill" in sb and sb["lock_held_after_kill"] is None and sr.get("damaged_unit_quarantined_and_recomputed") is True
                                 and it.get("semantics_problems_A") == 0 and not mA.get("violations") and not mB.get("violations")
                                 and set(mA.get("affinity_union", [99])) <= set(range(4)) and (it.get("marked_A") or {}).get("manifest_mode") == "INTEGRATION_TEST"
                                 and {x["block"] for x in [json.loads(l) for l in gzip.open(V4 / "integration/smoke_roster_v4.jsonl.gz", "rt")]} ==
                                 {"root", "prefix1", "prefix2", "prefix3"})
    ev["replay"] = {"A": {k: rA.get(k) for k in ("replayed", "verified", "failures", "max_q_exec_diff")}, "B": {k: rB.get(k) for k in ("replayed", "verified", "failures")}}
    g["witness_replay"] = all(r.get("replayed", 0) > 0 and r.get("replayed") == r.get("verified") and not r.get("failures") for r in (rA, rB))
    t4, t2 = passed(V4 / "integration/pytest_v4.txt"), passed(V4 / "integration/pytest_v2.txt")
    tpl = load(V4 / "production_policy_v4.TEMPLATE.json") or {}
    okt, why_t, auth_t, cur = prod3d.check(V4 / "production_policy_v4.TEMPLATE.json", None, dry=True)
    ev["guards"] = {"pytest_v4": t4[1], "pytest_v2": t2[1], "template_version": tpl.get("version"),
                    "template_unauthorized": tpl.get("authorization") == {"authorized_by": None, "authorized_utc": None, "statement": None},
                    "template_refusals_other_than_authorization_now": why_t, "preflight": {k: pre.get(k) for k in (
                        "physical_queries", "planner_calls", "teacher_entry_calls", "run_dir_created", "refusals_other_than_authorization")},
                    "teacher_protocol": (tpl.get("frozen") or {}).get("teacher_protocol")}
    g["production_guards"] = (t4[0] and t2[0] and ev["guards"]["template_unauthorized"] and not why_t and tpl.get("version") == prod4.VERSION
                              and pre.get("planner_calls") == 0 and pre.get("teacher_entry_calls") == 0 and pre.get("run_dir_created") is False
                              and pre.get("refusals_other_than_authorization") == [] and ev["guards"]["teacher_protocol"] == "89cbabed5d7e4847")
    import shutil
    free = shutil.disk_usage(str(ROOT)).free / 2 ** 30
    ev["forecast"] = {k: fc.get(k) for k in ("production_queries", "cpu_hours", "cpu_hours_p05_p95", "scenarios_wall_hours", "wall_hours_expected",
                                             "wall_hours_buffered", "buffer_rule", "held_out_timing_checks", "disk_gb", "disk_gb_buffered")}
    ev["forecast"]["disk_free_gb_now"] = round(free, 1)
    g["forecast_storage"] = fc.get("production_queries") == len(phys) and free >= float(fc.get("disk_gb_buffered", 1e9)) + 5.0
    st = load(V4 / "status_service/STATUS_SERVICE_STATE.json") or {}
    ev["status_service"], ev["offhost_backup"] = st.get("service"), st.get("backup")
    pv = load(V4 / "push_verification/fresh_clone_verification_v4.json")
    ev["github"] = pv
    g["github"] = bool(pv) and pv.get("pass") is True
    started4 = any(V4.glob("run_*/run_manifest.json"))
    started3 = (V3 / "run_first_cohort/run_manifest.json").exists()
    req = ("source_qc", "roster", "split_leakage", "physical_integration", "witness_replay", "production_guards", "forecast_storage", "github")
    blockers = [k for k in req if not g.get(k)]
    P = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    d = {"2D_Q2_CURRENT_STATE": (st.get("observed_2d") or {}).get("state", "UNKNOWN"),
         "3D_V3_CURRENT_STATE": "not_started" if not started3 else "started", "3D_V3_BASE_REUSE": "none: no v3 label was collected",
         "3D_Q2_PARITY_ROSTER": P(g["roster"]), "SOURCE_QC": P(g["source_qc"]), "SPLIT_LEAKAGE": P(g["split_leakage"]),
         "3D_PHYSICAL_INTEGRATION": P(g["physical_integration"]), "WITNESS_REPLAY": P(g["witness_replay"]), "PRODUCTION_GUARDS": P(g["production_guards"]),
         "STATUS_SERVICE": (st.get("service") or {}).get("state", "UNKNOWN"), "OFFHOST_BACKUP": (st.get("backup") or {}).get("state", "UNKNOWN"),
         "3D_V4": "3D_V4_READY_PENDING_NEW_USER_AUTHORIZATION" if not blockers and not started4 else "NOT_READY",
         "3D_V4_COLLECTION": "NOT_STARTED" if not started4 else "STARTED"}
    doc = {"decision": d, "gates": g, "blockers": blockers, "evidence": ev,
           "workload": {"physical_queries": len(phys), "static_contrasts": len(rost) - len(phys), "roots": len({x["root_id"] for x in rost}),
                        "policy": tpl.get("policy"), "stages": tpl.get("stages"), "workers": tpl.get("workers"), "cpus": tpl.get("cpus"),
                        "expected_wall_h": fc.get("wall_hours_expected"), "buffered_wall_h": fc.get("wall_hours_buffered"), "disk_gb_buffered": fc.get("disk_gb_buffered")},
           "authorization": "a v3 authorization or token cannot start v4 (version, roster, code and hashes differ); see production_v4/V4_RUNBOOK.md section 4"}
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"decision": d, "gates": g, "blockers": blockers}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
