"""P8a / P8b machine-readable gates (engineering PASS is separate from any scientific outcome)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

import config as cf
import learn_data as ld

TEACHER = re.compile(r"teach_sequence|run_collection|run_rounds|evaluate_target\(")
SOURCES_A = ("abc7_models.py", "abc7_train.py", "abc8.py", "p8_eval.py", "p8_train.py", "p8_audit.py")
SOURCES_B = ("p8_select.py", "p8b_run.py")


def _rd(p):
    return json.loads(Path(p).read_text())


def gate_a(args) -> int:
    res, guard, aud = _rd("runs/p8/a/results.json"), _rd("runs/p8/a/guard.json"), _rd("runs/p8/audit/replay_audit.json")
    reg, tests = _rd("runs/p8/audit/regression/regression_result.json"), _rd("runs/p8/a_tests/guard.json")
    runs = [v for v in res["runs"].values() if v.get("status") == "RUN"]
    have = {(v["arm"], v["regime"], v["seed"]) for v in runs}
    need = {(a, g, s) for a in ("B-full", "B-structured", "B-none", "SF") for g in ("FULL_DECIDED", "SPARSE_AUX") for s in (0, 1, 2)}
    checks = {
        "historical_evidence_intact": aud["all_intact"],
        "baseline_regression_and_6_replays": bool(reg["passed"]) and len(reg["replays"]) == 6 and all(reg["replays"].values()),
        "p8_math_and_scope_tests": tests["exit"] == 0,
        "mandatory_3_seed_runs_both_regimes": need <= have,
        "factorized_control_present": any(a == "SF" for a, _, _ in have),
        "full_structured_none_present": {"B-full", "B-structured", "B-none"} <= {a for a, _, _ in have},
        "unconditional_and_conditional_evaluated": all("intervention" in v["validation"] and "conditional" in v["validation"] for v in runs),
        "guard_completed": guard["stop"] == "COMPLETED" and guard["overshoot_s"] == 0,
        "no_teacher_calls_in_sources": not any(TEACHER.search(Path(f).read_text()) for f in SOURCES_A),
        "leakage_audit_clean": not ld.leakage_audit(torch.load("runs/p7/training/dataset.pt", weights_only=False)),
    }
    out = {"stage": "P8A", "checks": checks, "missing_runs": sorted(map(list, need - have)), "cpu_s": guard["cpu_s"], "cap_cpu_s": 7000,
           "engineering_pass": all(checks.values())}
    out["verdict"] = "P8A_ENGINEERING_PASS" if out["engineering_pass"] else "P8A_BLOCKED"
    cf.write_json(Path(args.out), out)
    print(json.dumps(out, indent=1))
    return 0 if out["engineering_pass"] else 1


def gate_b(args) -> int:
    res, guard = _rd("runs/p8/b/results.json"), _rd("runs/p8/b/guard.json")
    gm = res.get("gm_smoke", [])
    checks = {
        "decision_funnel_csv": Path("runs/p8/b/decision_funnel.csv").exists(),
        "threshold_family_with_abstention": all("threshold" in v for v in res["arms"].values() if isinstance(v, dict) and "tau" in v),
        "sampler_report_and_decision": "use" in res.get("sampler_decision", {}),
        "gm_baseline_runs_under_primary_v2": bool(gm) and all(x["audit"] == "PASS" for x in gm),
        "gm_no_target_execution": all(x["outcome"] in ("TARGET_NOT_RUN", "NO_REPAIR_WITNESS_WITHIN_BUDGET", "ABORTED") for x in gm),
        "guard_completed": guard["stop"] == "COMPLETED" and guard["overshoot_s"] == 0,
        "no_teacher_collection_in_sources": not any(re.search(r"teach_sequence|run_collection|run_rounds", Path(f).read_text()) for f in SOURCES_B),
    }
    out = {"stage": "P8B", "checks": checks, "cpu_s": guard["cpu_s"], "cap_cpu_s": 3000, "sampler_use": res["sampler_decision"]["use"],
           "engineering_pass": all(checks.values())}
    out["verdict"] = "P8B_ENGINEERING_PASS" if out["engineering_pass"] else "P8B_BLOCKED"
    cf.write_json(Path(args.out), out)
    print(json.dumps(out, indent=1))
    return 0 if out["engineering_pass"] else 1
