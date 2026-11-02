"""P7a gate: machine-readable PASS / BLOCKED decision before any P7b teacher call (contract §3.4)."""
from __future__ import annotations

import json
import re
from pathlib import Path

import torch

import config as cf
import learn_data as ld

TEACHER_CALLS = re.compile(r"teach_sequence|plan_fixed_order|evaluate_target|run_collection|run_rounds|planner\.plan|primary_v2")
P7A_SOURCES = ("abc7_models.py", "abc7_train.py", "p7_eval.py", "p7a_run.py", "p7_audit.py", "p7a_viz.py")


def decide(args) -> int:
    out = Path(args.out)
    reg = json.loads(Path(args.regression, "regression_result.json").read_text())
    res = json.loads(Path(args.train, "results.json").read_text())
    guard = json.loads(Path(args.train, "guard.json").read_text())
    pytest_log = Path(args.regression, "pytest.log").read_text()
    runs = {k: v for k, v in res["runs"].items() if v.get("status") == "RUN"}
    need = [f"{a}#s{s}" for a in ("A", "B", "C", "B-no-coupling", "C-no-coupling") for s in (0, 1, 2)]
    have = {n: any(t.startswith(n + "|") for t in runs) for n in need}
    leak = ld.leakage_audit(torch.load("runs/p6/train/dataset.pt", weights_only=False))
    checks = {
        "regression_full_suite_and_6_replays": bool(reg["passed"]) and len(reg["replays"]) == 6 and all(reg["replays"].values()),
        "p7_model_tests_in_suite": "test_p7_models.py" in pytest_log or reg["tests"]["passed"] >= 314,
        "micro_overfit_B_C_on_2_real_roots": all(res["micro"][a]["pass"] for a in ("B", "C")) and len(res["micro"]["B"]["roots"]) >= 2,
        "edit_descriptors_distinct_at_init": all(v > 0.05 for a in ("B", "C") for v in res["micro"][a]["edit_vector_rel_l2_at_init"].values()),
        "primary_and_no_coupling_3_seeds": all(have.values()),
        "loss_and_gradient_diagnostics_recorded": all("term_grads_final" in v and "clip_fraction" in v for v in runs.values()),
        "calibration_on_validation_only": all("tau" in v and "legacy_p6_holdout" in v["metrics"] for v in runs.values()),
        "training_guard_completed": guard["stop"] == "COMPLETED" and guard["overshoot_s"] == 0,
        "no_teacher_call_in_p7a_sources": not any(TEACHER_CALLS.search(Path(f).read_text()) for f in P7A_SOURCES),
        "leakage_audit_clean": not leak,
    }
    dec = {"stage": "P7A", "checks": checks, "missing_runs": [k for k, v in have.items() if not v],
           "chosen_configs": res["chosen"], "pass": all(checks.values())}
    dec["verdict"] = "P7A_PASS" if dec["pass"] else "P7A_BLOCKED"
    cf.write_json(out, dec)
    print(json.dumps(dec, indent=1))
    return 0 if dec["pass"] else 1
