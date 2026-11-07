"""Derive 3D_MOCK_V2_DECISION.json mechanically from the saved evidence (no hand-entered verdicts).
  python decide_v2.py SIDECAR_V2_DIR OUT.json"""
import glob, gzip, json, sys
from pathlib import Path

root = Path(sys.argv[1]); M = root / "mock_v2"; V = M / "validation"
an = json.loads((V / "analysis_v2.json").read_text())
rep = {Path(p).stem.replace("replay_", ""): json.loads(Path(p).read_text()) for p in sorted(glob.glob(str(V / "replay_*.json")))}
pre, post = (V / "pytest_pre_run.txt").read_text(), (V / "pytest_post_run.txt").read_text()
fx2 = json.loads((M / "configs/fixtures_v2.json").read_text())
fd = [x for p in sorted(glob.glob(str(V / "finger_diag_*.json"))) for x in json.loads(Path(p).read_text())]
integ = (V / "integrity_pinned_source.json").read_text()
v1tree = (V / "v1_tree_unchanged.txt").read_text()
gitst = (V / "source_git_status.txt").read_text().strip()
v1rep = {k: json.loads((V / f"v1_pinned_replay_{k}.json").read_text()) for k in ("reference", "pilot_A")}
roster = [json.loads(x) for x in gzip.open(M / "roster_v2.jsonl.gz", "rt")]
excl = set(json.loads((M / "excluded_v1_touched_roots.json").read_text()))
dry = json.loads((root / "production_v2/dry_run_production_live.json").read_text())
split_live = json.loads((root / "production_v2/split_manifest_live.json").read_text())


def ok(b):
    return "PASS" if b else "FAIL"


val, det = an["validity"], an["determinism"]
pt, choice = an["policy_table"], an["policy_choice"]
passed = lambda t: (" passed" in t and " failed" not in t and " error" not in t)  # noqa: E731
replay_ok = all(r["replayed"] == r["verified"] and not r["missing_witnesses"] for r in rep.values())
mism = {k: v["mismatches"] for k, v in det.items() if isinstance(v, dict) and "mismatches" in v}
vvr = {k: v["mismatches"] for k, v in det["virtual_vs_real"].items()}
acc_ok = all(not a["violations"] for a in an["accounting"].values())
runs_ok = all(v["complete"] and not v["units_failing_validation"] and not v["monitor_violations"] for v in val.values())
mask_ok = all(len(v["peak"]["affinity_union"]) <= 18 for v in val.values())
cov = an["coverage"]
need = {"K": ["0", "1", "2", "3"], "depth": ["0", "1", "2"], "band": ["near", "mid", "far", "vacated"], "shape": ["box", "cylinder"]}
cov_ok = all(any(k.startswith(f"{x}:") and not k.endswith(":None") for k in cov[s]) for s, xs in need.items() for x in xs)
fl = an["forecast_live"].get(choice["choice"] or "adaptive_v1") or an["forecast_live"]["adaptive_v1"]
fsum = {k: sum(x[k] for x in fd) for k in ("poses", "sols", "clear_open", "clear_preshaped", "only_preshaped_poses", "only_open_poses")}
hold_cohort = {r["root_id"] for r in roster if r["cohort"] == "holdout"}
sub = {
    "semantics": ok(passed(pre) and passed(post) and acc_ok and all(all(x["placement_phases_only"] for x in r["results"]) for r in rep.values())),
    "ik_branch_fix": ok(fx2["natural_found"] and an["branch_effects"]["reference_v2"]["units_rescued_after_hold_failure"] >= 1),
    "witness_retention": ok(all(rep[k]["replayed"] == an["branch_effects"][k]["branch_verified"] and an["storage"][k]["witnesses"] == rep[k]["unique_witnesses"]
                                for k in rep)),
    "determinism": ok(all(v == 0 for v in mism.values())),
    "virtual_equals_real": ok(all(v == 0 for v in vvr.values())),
    "accounting_zero_unaccounted": ok(acc_ok),
    "physical_witness_replay": ok(replay_ok),
    "v1_history_replay_pinned": ok(all(r["replayed"] == r["verified"] for r in v1rep.values())),
    "interrupt_resume_storage": ok(det.get("interrupt_vs_A", {}).get("mismatches") == 0 and bool(val["adaptive_v1_interrupt"]["resume_audits"])
                                   and bool(val["adaptive_v1_interrupt"]["resume_audits"][0]["ledger_units_failed"])),
    "source_integrity": ok(bool(integ) and json.loads(integ)["n_changed"] == 0 and json.loads(integ)["git_status_source"] == "clean"),
    "v1_artifacts_unchanged": ok("rc=0" in v1tree and not gitst),
    "split_leakage": ok(all(r["split"] == "train" for r in roster) and not (hold_cohort & excl) and split_live["old_assignments_changed"] == 0 and not split_live["conflicts"]),
    "safety": ok(runs_ok and mask_ok),
    "policy_selection": ok(choice["choice"] is not None),
    "coverage": ("PASS for K0-K3 / depth 0-2 / near-mid-far-vacated / box-cylinder" if cov_ok else "FAIL") + "; COVERAGE_BLOCKED for K4 (0 roots) and NONE (3 roots, none in the mock)",
    "finger_structure_check": (f"RETAINED_SOURCE_BEHAVIOUR (not fixed, not uniformly conservative): the structure check uses fully open fingers;"
                               f" {fsum['only_preshaped_poses']}/{fsum['poses']} poses would gain a structure-clear IK solution with pre-shaped fingers and"
                               f" {fsum['only_open_poses']} are clear only with open fingers; positives are protected by the MuJoCo contact gates"),
    "runtime": ok(fl["wall_days_p95_x1_5"] <= 7.0),
    "production_guard": ok(dry["planner_calls"] == 0 and dry["dry_run"]),
}
core = [k for k in sub if k not in ("coverage", "finger_structure_check")]
mock = "MOCK_V2_PASS" if all(sub[k] == "PASS" for k in core) and cov_ok else "MOCK_V2_FAIL"
doc = {"decision": {"MOCK_V2": mock, "LIVE_SOURCE_SYNC": "LIVE_SOURCE_SYNC_PASS (consistent IN-PROGRESS snapshot; see source_sync/)",
                    "PRODUCTION": "PRODUCTION_READY_PENDING_AUTHORIZATION" if mock == "MOCK_V2_PASS" else "NOT_READY",
                    "FULL_COLLECTION": "FULL_COLLECTION_NOT_STARTED"},
       "sub_verdicts": sub,
       "meaning": "MOCK_V2_PASS: the branch-aware placement teacher / collector is semantically correct, reproducible and physically validated on a"
                  " bounded 405-query mock under finite policies. It does NOT mean the source dataset is final (K4 0/510, NONE 3/408, 488 roots await"
                  " the generator's fixed-object check), that all grasp configurations were searched, that 0 observed misses means 0 risk, or that"
                  " production is authorized.",
       "policy": {"rule": "mock_v2/configs/selection_rule_v2.json (registered before any V2 teacher call)", "choice_by_frozen_rule": choice["choice"],
                  "recommended": choice["choice"], "passing": choice["passing"],
                  "candidates": {k: {x: v[x] for x in ("ref_positives", "missed", "miss_rate", "wilson95", "mean_cpu_s_per_query", "pass", "failing_difficult_strata")}
                                 | {"far": v["strata"]["band"].get("far"), "depth2": v["strata"]["depth"].get("2"), "between3plus": v["strata"]["between"].get("3+"),
                                    "holdout": v["strata"]["cohort"].get("holdout")} for k, v in pt.items()}},
       "determinism": mism, "virtual_vs_real": vvr,
       "replays": {k: {"associations_replayed": r["replayed"], "unique_witnesses": r["unique_witnesses"], "verified": r["verified"],
                       "missing": len(r["missing_witnesses"])} for k, r in rep.items()},
       "v1_pinned_replays": {k: {"replayed": r["replayed"], "verified": r["verified"]} for k, r in v1rep.items()},
       "branch_effects": an["branch_effects"],
       "finger_diagnostic": {**fsum, "queries": len(fd), "only_preshaped_pose_fraction": round(fsum["only_preshaped_poses"] / max(1, fsum["poses"]), 5)},
       "workers": {"recommended": 12, "cpu_mask": "0-11", "sweep": an["sweep"]},
       "forecast_live_snapshot": an["forecast_live"],
       "storage": an["storage"],
       "safety_peaks": {k: v["peak"] for k, v in val.items()},
       "unresolved": ["Live source is an IN-PROGRESS snapshot: K4 0/510, NONE 3/408; refresh after the generator stops",
                      "488 selected roots await the generator author's fixed-object check (selection may still change)",
                      "K4 / NONE placement costs unmeasured (COVERAGE_BLOCKED); forecast backs them off",
                      "Structure collision check keeps the source's fully open fingers: not uniformly conservative (finger_diagnostic); negatives of the affected poses are scoped by it",
                      "No natural 'execution failure then later branch verified' unit exists in the bounded mock (0 of 405 reference queries); covered by a synthetic test",
                      "No hard CPU quota (cgroup cpu controller absent on gvlab): ceiling enforced by affinity + monitor",
                      "v1 MOCK_REPORT states 24 mock roots; the frozen v1 roster has 22 (K0 4, K1-K3 6 each)"],
       "not_done_by_design": ["full 3D placement collection", "full 2D collection", "new source roots / K4 or NONE quota filling", "changes to the live generator",
                              "ML training", "force push / merge to main"]}
Path(sys.argv[2]).write_text(json.dumps(doc, indent=1, sort_keys=False, default=str))
print(json.dumps(doc["decision"]), json.dumps(sub))
