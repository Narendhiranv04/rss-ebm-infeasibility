"""Derive production_v3/PRODUCTION_READINESS_DECISION.json mechanically from saved evidence (no hand-entered verdict; a missing or failing
piece of evidence is a failed gate, never a pass).
  python decide_v3.py OUT.json            (run from a checkout whose SHELF_SRC is the pinned source; SHELF_SRC exported)

Verdicts: MOCK_V2 (MOCK_V2_PASS | FAIL), SNAPSHOT_INTEGRITY (SOURCE_SNAPSHOT_INTEGRITY_PASS | FAIL), COHORT_QC (FIRST_COHORT_QC_PASS | FAIL),
FULL_2500_SOURCE (FULL_2500_SOURCE_INCOMPLETE | COMPLETE; informational), PRODUCTION_SOFTWARE (PASS | FAIL), FIRST_COHORT
(FIRST_COHORT_READY_PENDING_USER_AUTHORIZATION only if every required gate passes, else NOT_READY + blockers), FULL_COLLECTION
(FULL_COLLECTION_NOT_STARTED | STARTED).
"""
import gzip, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import prod3d  # noqa: E402

V3 = ROOT / "production_v3"


def load(p):
    p = V3 / p if not str(p).startswith("/") else Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def passed(txt):
    return txt is not None and " passed" in txt and " failed" not in txt and " error" not in txt


def main(out):
    g, ev = {}, {}
    # 1 MOCK_V2: frozen decision + every V2 sidecar file byte-identical to its manifest
    d2 = json.loads((ROOT / "3D_MOCK_V2_DECISION.json").read_text())
    man = [ln.split("  ", 1) for ln in (ROOT / "ARTIFACT_MANIFEST.sha256").read_text().splitlines() if ln.strip()]
    bad = [p for h, p in man if not (ROOT / p).exists() or prod3d.sha_file(ROOT / p) != h]
    ev["mock_v2"] = {"decision": d2["decision"]["MOCK_V2"], "v2_manifest_files": len(man), "v2_files_changed_or_missing": bad[:10]}
    g["mock_v2"] = d2["decision"]["MOCK_V2"] == "MOCK_V2_PASS" and not bad
    # 2 SNAPSHOT_INTEGRITY: published snapshot verifies; final live state audited; selection reproduces
    import source_sync3d as SS
    src = Path(os.environ.get("SHELF_SRC", str(ROOT.parent / "dataset_bd_classquota_without_home_position_no_overpass"))).resolve()
    sm = json.loads(gzip.open(ROOT / "source_sync/SOURCE_SNAPSHOT_MANIFEST.json.gz", "rt").read())["files"]
    sbad = [p for p, e in sm.items() if not (src / p).exists() or prod3d.sha_file(src / p) != e["sha256"]]
    fs = load("source/SOURCE_FINAL_STATE_AUDIT.json") or {}
    ok_sel, _ = SS.select_check(src)
    ev["snapshot"] = {"published_files": len(sm), "published_files_differing": sbad[:10], "selection_reproduces": ok_sel,
                      "final_state_audit_pass": fs.get("pass"), "k_quota_run_stopped": fs.get("k_quota_run_stopped"),
                      "host_rehash": fs.get("published_files_rehashed_on_host"), "ledger_append_only": (fs.get("ledger") or {}).get("append_only"),
                      "final_k0_k3_identical": (fs.get("final_selection") or {}).get("k0_k3_identical_to_published")}
    g["snapshot"] = not sbad and ok_sel and fs.get("pass") is True
    # 3 COHORT_QC
    qc = load("QC_ELIGIBLE_ROOTS.json") or {"roots": {}, "equivalence": {}, "tool": {}}
    coh = load("frozen/cohort_manifest.json") or {"cohort_roots": []}
    eq = qc["equivalence"]
    rost = [json.loads(x) for x in gzip.open(V3 / "frozen/first_cohort_roster.jsonl.gz", "rt")] if (V3 / "frozen/first_cohort_roster.jsonl.gz").exists() else []
    ev["cohort_qc"] = {"tool_pinned": qc["tool"].get("sha256") == qc["tool"].get("pinned"), "author_jobs": eq.get("author_jobs"),
                       "rerun": eq.get("rerun"), "verdict_equal": eq.get("verdict_equal"), "first_grasp_equal": eq.get("first_grasp_equal"),
                       "exclusions_equal": eq.get("exclusions_equal"), "summary": qc.get("summary"),
                       "cohort_roots": len(coh["cohort_roots"]), "cohort_roots_not_eligible": [r for r in coh["cohort_roots"] if not qc["roots"].get(r, {}).get("eligible")][:5],
                       "roster_roots_not_eligible": sorted({x["root_id"] for x in rost if not qc["roots"].get(x["root_id"], {}).get("eligible")})[:5]}
    c = ev["cohort_qc"]
    g["cohort_qc"] = (c["tool_pinned"] and eq.get("rerun") == eq.get("author_jobs") == eq.get("verdict_equal") and eq.get("exclusions_equal") is True
                      and c["cohort_roots"] > 0 and not c["cohort_roots_not_eligible"] and not c["roster_roots_not_eligible"] and bool(rost))
    # 4 split / roster
    sa = load("SPLIT_AND_ROSTER_AUDIT.json") or {"split": {}, "leakage": {}, "roster": {}}
    ev["split_roster"] = {"old_assignments_changed": sa["split"].get("old_assignments_changed"), "components_spanning_two_splits": sa["split"].get("components_spanning_two_splits"),
                          "cross_split_layouts": sa["leakage"].get("cross_split_layouts"), "roster_equal_v2_restricted": sa["roster"].get("equal_to_v2_live_roster_restricted_to_cohort"),
                          "every_row_qc_eligible_and_pinned": sa["roster"].get("every_row_qc_eligible_and_pinned"), "coverage_missing": sa["roster"].get("coverage_missing"),
                          "roster_sha_matches_file": sa["roster"].get("sha256") == (prod3d.sha_file(V3 / "frozen/first_cohort_roster.jsonl.gz") if rost else None)}
    s = ev["split_roster"]
    g["split_roster"] = (s["old_assignments_changed"] == 0 and s["components_spanning_two_splits"] == 0 and s["cross_split_layouts"] == 0
                         and s["roster_equal_v2_restricted"] is True and s["every_row_qc_eligible_and_pinned"] is True and s["roster_sha_matches_file"]
                         and not any((s["coverage_missing"] or {"x": [1]}).values()))
    # 5 PRODUCTION_SOFTWARE: tests, template, preflight, integration smoke, replay, interrupt / corruption recovery, lock, refusals
    t3, t2 = [(V3 / f"integration/{n}").read_text() if (V3 / f"integration/{n}").exists() else None for n in ("pytest_v3.txt", "pytest_v2.txt")]
    it = load("integration/integration_results.json") or {}
    pre = load("integration/production_metadata_preflight.json") or {}
    tpl = load("production_policy.TEMPLATE.json") or {}
    okt, why_t, auth_t, _ = prod3d.check(V3 / "production_policy.TEMPLATE.json", None, dry=True) if tpl else (False, ["no template"], [], None)
    rA, rB = it.get("replay_A") or {}, it.get("replay_B") or {}
    ev["software"] = {"pytest_v3": (t3 or "missing").strip().splitlines()[-1:], "pytest_v2": (t2 or "missing").strip().splitlines()[-1:],
                      "template_unauthorized": tpl.get("authorization") == {"authorized_by": None, "authorized_utc": None, "statement": None},
                      "template_refusals_other_than_authorization_now": why_t,
                      "preflight": {k: pre.get(k) for k in ("rc", "planner_calls", "teacher_entry_calls", "run_dir_created", "refusals_other_than_authorization",
                                                            "roster_regenerated_equal", "physical_queries", "by_split")},
                      "smoke_A_complete": (it.get("audit_A") or {}).get("complete"), "smoke_B_complete": (it.get("audit_B") or {}).get("complete"),
                      "B_equals_A": (it.get("B_equals_A") or {}).get("equal"),
                      "workers_died_with_coordinator": (it.get("smoke_B_interrupt") or {}).get("descendants_alive_3s_after_kill") == [],
                      "lock_released_after_kill": (it.get("smoke_B_interrupt") or {}).get("lock_held_after_kill") is None and "smoke_B_interrupt" in it,
                      "damaged_npz_quarantined_and_recomputed": (it.get("smoke_B_resume") or {}).get("damaged_unit_quarantined_and_recomputed"),
                      "replay_A": {k: rA.get(k) for k in ("replayed", "verified", "n_failures")}, "replay_B": {k: rB.get(k) for k in ("replayed", "verified", "n_failures")},
                      "semantics_problems_A": sum(len(v["semantics_problems"]) for v in (it.get("provenance_A") or {}).values()),
                      "test_runs_marked": all((it.get(f"marked_{k}") or {}).get("marker_file") and (it.get(f"marked_{k}") or {}).get("manifest_mode") == "INTEGRATION_TEST" for k in "AB")}
    w = ev["software"]
    g["software"] = (passed(t3) and passed(t2) and w["template_unauthorized"] and not why_t and pre.get("rc") == 0 and pre.get("planner_calls") == 0
                     and pre.get("teacher_entry_calls") == 0 and pre.get("run_dir_created") is False and pre.get("refusals_other_than_authorization") == []
                     and pre.get("roster_regenerated_equal") is True and w["smoke_A_complete"] is True and w["smoke_B_complete"] is True and w["B_equals_A"] is True
                     and w["workers_died_with_coordinator"] and w["lock_released_after_kill"] and w["damaged_npz_quarantined_and_recomputed"] is True
                     and rA.get("replayed") == rA.get("verified") and rA.get("replayed", 0) > 0 and not rA.get("n_failures")
                     and rB.get("replayed") == rB.get("verified") and rB.get("replayed", 0) > 0 and not rB.get("n_failures")
                     and w["semantics_problems_A"] == 0 and w["test_runs_marked"])
    # 6 SAFETY: smoke + QC monitors fail-closed, no violation, inside the CPU ceiling
    qm = [json.loads(x) for x in (V3 / "qc/rerun/monitor.jsonl").read_text().splitlines()] if (V3 / "qc/rerun/monitor.jsonl").exists() else []
    mA, mB = it.get("monitor_A") or {}, it.get("monitor_B") or {}
    ev["safety"] = {"smoke_A": mA, "smoke_B": mB, "qc_ticks": len(qm), "qc_violations": sorted({v for m in qm for v in m["violations"]}),
                    "qc_affinity_union": sorted({c for m in qm for c in m["affinity_union"]}), "qc_max_temp_c": max((m["temp_c"] or 0 for m in qm), default=None)}
    g["safety"] = (bool(mA) and bool(mB) and not mA.get("violations") and not mB.get("violations") and set(mA.get("affinity_union", [99])) <= set(range(4))
                   and set(mB.get("affinity_union", [99])) <= set(range(4)) and not mA.get("unexpected_descendants") and bool(qm) and not ev["safety"]["qc_violations"]
                   and set(ev["safety"]["qc_affinity_union"]) <= set(range(12)))
    # 7 STORAGE / FORECAST (cohort-specific)
    fc = load("frozen/forecast_first_cohort.json") or {}
    import psutil
    free = psutil.disk_usage(str(ROOT)).free / 2 ** 30
    nphys = sum(1 for x in rost if not x["static"])
    ev["forecast"] = {k: fc.get(k) for k in ("production_queries", "cpu_hours", "cpu_hours_p05_p95", "wall_hours_expected", "wall_hours_p95_x1_5", "disk_gb",
                                             "disk_gb_x1_5", "workers", "parallel_efficiency", "strata_backed_off", "by_split", "by_K")}
    ev["forecast"]["disk_free_gb_now"] = round(free, 1)
    g["storage_forecast"] = bool(fc) and fc.get("production_queries") == nphys and free >= float(fc.get("disk_gb_buffered", 1e9)) + 5.0
    # 8 FINGER-WIDTH sensitivity (informational; never changes the frozen protocol)
    fw = load("finger_width/finger_width_summary.json")
    ev["finger_width"] = fw
    g["finger_width_analysis_done"] = bool(fw) and fw.get("complete") is True
    # 9 GITHUB push verification (independent fresh checkout)
    pv = load("push_verification/fresh_clone_verification.json")
    ev["github"] = pv
    g["github"] = bool(pv) and pv.get("pass") is True
    started = (ROOT / "production_v3/run_first_cohort").exists() or any((ROOT / "production_v3").glob("run_*/run_manifest.json"))
    req = ("mock_v2", "snapshot", "cohort_qc", "split_roster", "software", "safety", "storage_forecast", "finger_width_analysis_done", "github")
    blockers = [k for k in req if not g.get(k)]
    verdict = {"MOCK_V2": "MOCK_V2_PASS" if g["mock_v2"] else "FAIL",
               "SNAPSHOT_INTEGRITY": "SOURCE_SNAPSHOT_INTEGRITY_PASS" if g["snapshot"] else "FAIL",
               "COHORT_QC": "FIRST_COHORT_QC_PASS" if g["cohort_qc"] else "FAIL",
               "FULL_2500_SOURCE": "FULL_2500_SOURCE_INCOMPLETE",
               "PRODUCTION_SOFTWARE": "PASS" if g["software"] and g["safety"] else "FAIL",
               "FIRST_COHORT": "FIRST_COHORT_READY_PENDING_USER_AUTHORIZATION" if not blockers and not started else "NOT_READY",
               "FULL_COLLECTION": "FULL_COLLECTION_NOT_STARTED" if not started else "STARTED"}
    tri = (ROOT.parent / "dataset_bd_classquota_without_home_position_no_overpass/selected.json")
    sel = json.loads(tri.read_text()) if tri.exists() else {}
    doc = {"decision": verdict, "gates": g, "blockers": blockers, "evidence": ev,
           "first_cohort": {"qc_roots": len(coh["cohort_roots"]), "roots_with_queries": coh.get("roots_with_queries"), "by_class": coh.get("by_class"),
                            "by_qc_status": coh.get("by_qc_status"), "physical_queries": nphys, "static_contrasts": len(rost) - nphys,
                            "queries_by_split": (sa["roster"] or {}).get("by_split"), "policy": tpl.get("policy"), "stages": tpl.get("stages"),
                            "workers": tpl.get("workers"), "cpus": tpl.get("cpus"), "teacher_protocol": (tpl.get("frozen") or {}).get("teacher_protocol")},
           "full_2500": {"published_counts": sel.get("counts"), "quota": sel.get("quota"), "k4_selected": (sel.get("counts") or {}).get("4"),
                         "none_selected_final_run": (fs.get("final_selection") or {}).get("none_final"),
                         "note": "K4 (0 / 510) and NONE are generated separately by the source author and are not part of the first cohort"},
           "meaning": "FIRST_COHORT_READY_PENDING_USER_AUTHORIZATION: every gate above passed on saved evidence; the collector refuses to start until the "
                      "user records an authorization in a copy of production_policy.TEMPLATE.json and passes the token it then requires.",
           "next_user_action": "follow FIRST_COHORT_RUNBOOK.md section 4: copy the template, write authorized_by / authorized_utc / statement, compute the "
                               "token with prod3d.py token, and start launch_first_cohort.sh --execute in tmux"}
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"decision": verdict, "gates": g, "blockers": blockers}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
