"""G4C-P0 outputs: plan.json, audit.json, schema.json, coverage_plan.csv/.png, overview.png
and manifest.json in runs/g4c_preflight/. Every future-outcome panel is NOT_RUN; the one
real scene with outcomes is the archived G4D R4 root, marked CONTEXT ONLY. Loaded planner,
skill, target and teacher entry points are wrapped with call counters (expected: 0).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

import config as cf
import g4b_figures as g4bf
import g4c
import geometry as ge
import viz
from scene import scene_from_dict
from viz import COLORS, plt

SOURCES = ("g4c.py", "g4c_figures.py", "tests/test_g4c.py", "sequences.py", "effects.py", "proposals.py", "scene.py", "data.py",
           "geometry.py", "config.py", "config.toml", "config_v2.toml", "g4d_run.py", "sampler.py")
HISTORICAL = ("runs/g4a_evidence_audit/g4a_plan.json", "runs/g4a_evidence_audit/candidate_labels.json",
              "runs/g4a_evidence_audit/primary_episodes.json", "runs/g4d_evidence/final_plan.json",
              "runs/g4d_evidence/ledger.json", "runs/g4d_evidence/result.json", "runs/g4d_evidence/unit_evidence.json",
              "runs/g4d_evidence/legacy_revalidation.json", "runs/g4b_sampler_exact/result.json")
DOCS = ("latent_causal_tamp_2D_ABC_guides_v1_7_v1_9/latent_causal_tamp_methodology_v1_7.md",
        "latent_causal_tamp_2D_ABC_guides_v1_7_v1_9/latent_causal_tamp_implementation_guide_v1_9.md",
        "instruction-phase-md/G4C_2D_ABC_Data_Readiness_PLAN_ONLY.md")


def audit(root: Path) -> dict:
    """Historical evidence map, reuse eligibility by scope, and the data gap (no new outcomes)."""
    res = json.loads((root / "runs/g4d_evidence/result.json").read_text())
    leg = json.loads((root / "runs/g4d_evidence/legacy_revalidation.json").read_text())
    g4b = json.loads((root / "runs/g4b_sampler_exact/result.json").read_text())
    return {"g4d": {"gate": res["gate"], "units": res["units"],
                    "reuse": "DEVELOPMENT_CONTROL_ONLY: G4D sparse roots are not independent G4C roots and never test roots",
                    "kept_unresolved": {"R2_chain": "UNKNOWN", "R5_baseline": "UNKNOWN", "R1_robot_start_delta": 0}},
            "g4a_legacy": {"positives_validated": leg["positives"], "negative_transfer": leg["negative_transfer"],
                           "decision": "NOT_READY_TO_SCALE (historical, unchanged)",
                           "reuse": "DERIVED_LEGACY, V1 backend; not counted toward G4C 20-root thresholds"},
            "g4b": {"decision": g4b["decision"], "scope": g4b["evidence_scope"], "joint_rz_mcmc": "NOT IMPLEMENTED (not required "
                    "for G4C; bounded exact z-marginal <= 8 active slots exists)", "use_as_supervision": "NEVER"},
            "new_roots": {"labels": "NOT_RUN", "teacher_calls": 0, "status": "G4C DATA COLLECTION NOT RUN"},
            "data_gap": "every G4C threshold is NOT_RUN: no G4C evidence exists yet"}


def schema(plan: dict) -> dict:
    k = next(k for k, v in plan["roots"].items() if v.get("family") == "J4")
    p = next(p for p in plan["roots"][k]["queries"]["pool"] if p["moved"] == 3)
    return {"dataset_version": g4c.VERSION, "execution_protocol": "INITIAL_SCENE_ORDERED_ONE_SHOT_V2",
            "comparative_model_version": "2D-ABC-v1",
            "record_fields": {"root_id": "split metadata only (never a learner input)", "query_id": "unit id",
                              "ordered_signature": "canonical [[object_id, cell], ...]", "scene_edit_id": "scene:<root>:<edit>",
                              "omission_id": "omission_of:<sequence signature> (remaining order preserved)",
                              "candidate_id": "per-object physical cell", "static_status": "VALID|INVALID|UNRESOLVED",
                              "ordered_feasibility_status": "VERIFIED_FEASIBLE|CERTIFIED_INFEASIBLE|UNRESOLVED",
                              "target_status": "per target evaluation", "certificate_scope": "ORDER|ASSIGNMENT",
                              "witness_exists": "complete relocation + target + safe retreat", "J_first_label_mask":
                              "first complete relocation witness exists (independent of target result); empty repair J=0",
                              "outcome_active_mask": "deterministic layout (z0, zS[catalogue], zR m>=1, zO[t] m>=2)",
                              "outcome_decided_mask": "active slot whose unit is POSITIVE or NEGATIVE",
                              "scene_pair_eligible_mask": "geometric fixed-robot eligibility (frozen in plan)",
                              "effect_decided_mask": "both sides decided and eligible",
                              "failure_provenance": "stop status, normalized reason (addendum A2), budget"},
            "privileged_namespace": ["labels", "outcomes", "witnesses", "certificates", "trajectories", "seeds", "root_id"],
            "learner_allowlist": "data.learner_tensors_v2 fields (scene, robot q/g/attachment, profile, FK, masks, anchor)",
            "example_ordered_record": {"query_id": f"seq:{k}:...", "root_id": k, "ordered_signature": p["sequence"],
                                       "static_status": p["static_status"], "bc_layout": p["bc_layout"],
                                       "ordered_feasibility_status": "NOT_RUN", "J_first_label_mask": "NOT_RUN"},
            "full_bc_pool_rule": "a root's pool counts for the exact B/C joint loss only if every active slot of every pool "
                                 "sequence is decided; a partly masked pool trains single-outcome terms only"}


def _panel(ax, robot, cfg, plan, k, title):
    v = plan["roots"][k]
    s = scene_from_dict(v["scene"])
    viz.scene_panel(ax, s, cfg, title)
    viz.draw_robot(ax, robot, s.robot.q_rad, s.robot.gripper_gap_m, alpha=0.6, zorder=6)
    ax.set(xlim=viz.LIMITS["grid"][0], ylim=viz.LIMITS["grid"][1])
    for o in s.objects:
        role = next((r for r, i in v["roles"].items() if i == o.id), "")
        ax.text(*o.xy_m, f"{o.id}\n{role}" if role else o.id, fontsize=5.5, ha="center", va="center", zorder=10)
    ax.add_patch(Circle(s.lattice.center(v["goal_cell"]), 0.024, fc="none", ec=COLORS["target_edge"], ls="--", lw=1.4, zorder=9))
    seq = max(v["queries"]["pool"], key=lambda p: (p["moved"], p["origin"] == "declared"))["sequence"]
    for t, (o, c) in enumerate(seq):
        a, b = np.array(s.get(o).xy_m), np.array(s.lattice.center(c))
        ax.add_patch(Circle(b, 0.024, fc="none", ec=COLORS["hypothetical"], ls="--", lw=1.2, zorder=8))
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=COLORS["hypothetical"], ls="--", zorder=8))
        ax.text(*((a + b) / 2), str(t + 1), fontsize=7, color="white", ha="center", va="center", zorder=11,
                bbox=dict(boxstyle="circle,pad=0.15", fc=COLORS["hypothetical"], ec="none"))
    Q = v["queries"]
    ax.text(0.01, 0.99, "\n".join([f"{v['family']}: {len(Q['pool'])} sequences, {sum(q['eligible'] for q in Q['scene_queries'])} "
                                   f"eligible edits", f"catalogue {Q['catalogue']['queries']}",
                                   f"max active z {max(p['bc_layout']['active_count'] for p in Q['pool'])}"]),
            transform=ax.transAxes, va="top", fontsize=6.5, family="monospace", bbox=dict(fc="white", ec=COLORS["muted"]))
    ax.text(0.01, 0.01, "SCHEMATIC / NOT_RUN", transform=ax.transAxes, fontsize=8, color="white", weight="bold",
            bbox=dict(fc="#7a7975", ec="none", pad=3))


def run(args) -> int:
    calls = g4bf._instrument()
    t0, c0, root, out = time.perf_counter(), time.process_time(), Path("."), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise SystemExit("plan.json already frozen; refusing to overwrite")
    hist_before = {f: g4c._sha(root / f) for f in HISTORICAL}
    cfg, v2 = cf.load_config(args.config), cf.load_config_v2(args.config_v2)
    robot = ge.make_robot(cfg)
    plan = g4c.make_plan(cfg, v2, robot, root)
    bad = g4c.validate(plan)
    if bad:
        raise SystemExit(f"plan validation failed: {bad}")
    plan2 = g4c.make_plan(cfg, v2, robot, root)
    cov = g4c.coverage(plan)
    cf.write_json(out / "plan.json", plan)
    cf.write_json(out / "audit.json", audit(root))
    cf.write_json(out / "schema.json", schema(plan))
    with open(out / "coverage_plan.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["threshold", "required", "planned_count", "plan_capacity", "measured_status", "planned_roots"])
        for r in cov:
            w.writerow([r["threshold"], r["required"], r["planned_count"], r["plan_capacity"], r["status"], " ".join(r["planned_roots"])])
    f, ax = plt.subplots(figsize=(12, 5))
    y = np.arange(len(cov))
    ax.barh(y + 0.2, [r["required"] for r in cov], 0.4, color=COLORS["muted"], label="required (proposed threshold)")
    ax.barh(y - 0.2, [r["planned_count"] for r in cov], 0.4, color=COLORS["hypothetical"], label="planned supporting roots (PLAN)")
    for i, r in enumerate(cov):
        ax.text(max(r["required"], r["planned_count"]) + 0.2, i, "measured: NOT_RUN", va="center", fontsize=8)
    ax.set_yticks(y, [r["threshold"] for r in cov], fontsize=8)
    ax.set(xlabel="independent roots (planned capacity only if every unit decides)")
    ax.set_title("G4C coverage PLAN vs proposed thresholds - no evidence collected (all measured values NOT_RUN)", fontsize=10,
                 loc="left")
    ax.legend(fontsize=8, loc="lower right")
    f.tight_layout()
    f.savefig(out / "coverage_plan.png", dpi=80)
    plt.close(f)
    picks = [next(k for k, v in plan["roots"].items() if v.get("family") == fam) for fam in ("J1", "J2", "J3", "J4", "J9")]
    f, axs = plt.subplots(2, 3, figsize=(21, 13))
    g4bf.r4_context(axs[0, 0], root)
    for a, k in zip(axs.ravel()[1:], picks):
        _panel(a, robot, cfg, plan, k, f"planned root {k} ({plan['roots'][k]['family']}): longest declared sequence")
    f.suptitle(f"G4C-P0 plan {plan['plan_hash'][:19]}...: {len(plan['units'])} planned units; status {plan['status']}; "
               "G4C DATA COLLECTION NOT RUN", fontsize=11)
    f.tight_layout()
    f.savefig(out / "overview.png", dpi=75)
    plt.close(f)
    hist_after = {f: g4c._sha(root / f) for f in HISTORICAL}
    src = {p: g4c._sha(root / p) for p in SOURCES}
    docs = {p: g4c._sha(root / p) for p in DOCS}
    ok, why = g4c.admit(plan)
    summary = {"plan_hash": plan["plan_hash"], "deterministic_regeneration": plan2["plan_hash"] == plan["plan_hash"],
               "validator_errors": bad, "admission": why, "historical_inputs_unchanged": hist_before == hist_after,
               "planner_target_teacher_calls": calls, "cpu_s": round(time.process_time() - c0, 2),
               "wall_s": round(time.perf_counter() - t0, 2),
               "decision": "G4C-PREFLIGHT_READY_PENDING_BUDGET" if not bad and not ok and not any(calls.values())
               and hist_before == hist_after and plan2["plan_hash"] == plan["plan_hash"] else "G4C-PREFLIGHT_BLOCKED",
               "collection": "G4C DATA COLLECTION NOT RUN"}
    cf.write_json(out / "preflight_result.json", summary)
    cf.write_manifest(out, {"root_seed_start": g4c.SEED0}, cf.config_hash(cfg),
                      {"plan_hash": plan["plan_hash"], "sources": src, "documents": docs, "historical_inputs": hist_after})
    print(json.dumps(summary))
    return 0 if summary["decision"].endswith("PENDING_BUDGET") else 1
