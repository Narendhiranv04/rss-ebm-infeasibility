"""Sole command dispatcher (Guide v1.7 §1): argument parsing, dispatch and the
cross-stage replay. Never imported by other modules. Each stage's command body lives
with the module that owns that stage (moved here-from at G4A, behaviour unchanged):

    scene (G0)          viz.cmd_scene                 runs/g0_scene
    check (G1)          figures.cmd_check             runs/g1_robot
    intervene (G2)      intervention_figures.cmd_intervene    runs/g2_interventions
    solve (G3)          assignment_figures.cmd_solve  runs/g3_repairs        (FROZEN_CONTROL)
    case-analysis (G3.5) case_figures.cmd_case_analysis runs/g3_5_initial_scene (INITIAL_SCENE_ONE_SHOT_V1)
    g4a-plan, g4a-audit audit.cmd_g4a_plan / cmd_g4a_audit runs/g4a_evidence_audit
    teacher-plan/-run   teacher.cmd_teacher_plan / cmd_teacher_run runs/g4a_teacher_budget_diagnostic (G4A follow-up)
    g4d-plan, g4d-migrate g4d.cmd_g4d_plan / cmd_g4d_migrate runs/g4d_migration (V2 migration; no collection)
    g4d-legacy/-finalize/-run/-judge g4d_gate.*   runs/g4d_evidence (corrections, bounded collection, gate)
    g4b                 g4b.cmd_g4b                   runs/g4b_sampler_exact (exact sampler proof; no planner)
    g4c-plan            g4c.cmd_g4c_plan              runs/g4c_preflight (frozen 20-root plan; no collection)
    g4c-pilot-*         g4c_pilot.*                   runs/g4c_pilot (authorized 5-root pilot: finalize, run, judge)
    g4c-d2-precheck     d2_precheck.run               runs/g4c_d2_precheck (witness-first static precheck; no teacher)
    g4c-d2-poc-*        d2_poc.*                      runs/g4c_d2_poc_s (authorized POC_S: finalize, run, judge)
    g4c-d2-p3a-static   sector_cert.run               runs/g4c_d2_p3a_static (link-0 sector certificate; zero oracle calls)
    g4c-lf-p4-*         teacher_pilot.*               runs/g4c_lf_p4 (guarded regression gate; OPNEG-v1 calibration pilot)
    p5-d3lite           d3lite_run.run                runs/p5/d3lite (8 fresh roots, 4+4 gated, OPNEG-v1)
    p5-g4e              abc_run.run                   runs/p5/g4e (A/B/C proof of concept; guarded, CPU only)
    replay              run.cmd_replay                any run directory

    python run.py scene --config config.toml --case G0_ILLUSTRATIVE --out runs/g0_scene
    python run.py replay --run runs/g1_robot
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import assignment_figures as afig
import audit
import case_figures as cfig
import config as cf
import figures as fig
import g4d
import g4d_gate
import g4b
import g4c
import g4c_pilot
import d2_poc
import abc_run
import d2_precheck
import d3lite_run
import p6_run
import p6_train
import p6_viz
import p7a_run
import p7a_gate
import p7a_viz
import p7_collect
import p7b_run
import p7b_oneshot
import p7b_viz
import p8_audit
import p8_train
import p8_gates
import p8b_run
import p8_collect
import p8c_train
import p8d_run
import p8_viz
import sector_cert
import teacher_pilot
import geometry as ge
import intervention_figures as ifig
import motion as mo
import scene as sc
import teacher
import viz


def cmd_replay(args) -> int:
    """Re-certify every stored trajectory from the run's own files and hashes."""
    run = Path(args.run)
    archived = json.loads((run / "config.json").read_text())["config"]
    filled = sorted(set(cf.SCHEMA) - set(archived))      # sections introduced after this run; unused by replay
    current = cf.load_config(args.config)
    keys = {s: sorted(set(cf.SCHEMA[s]) - set(archived[s])) for s in archived if set(cf.SCHEMA[s]) - set(archived[s])}
    cfg = cf.validate_config({**{s: current[s] for s in filled},
                              **{s: {**{k: current[s][k] for k in keys.get(s, [])}, **v} for s, v in archived.items()}})
    scenes, result = json.loads((run / "scenes.json").read_text()), json.loads((run / "result.json").read_text())
    robot, rhash = ge.make_robot(cfg), cf.robot_profile_hash(cfg)
    delta, eps, evals = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"], cfg["geometry"]["certify_max_evals"]
    data, report = np.load(run / "arrays.npz", allow_pickle=False), {}
    for tid in sorted({k.split("__")[0] for k in data.files}):
        arr = {k.split("__")[1]: data[k] for k in data.files if k.startswith(tid + "__")}
        task = result["tasks"][tid]
        s, target = sc.scene_from_dict(scenes[tid]), task.get("target", "t0")
        verdicts = mo.certify_trajectory(robot, s, target, arr, delta, eps, evals)
        bound = sc.scene_from_dict(scenes[task.get("hash_scene", tid)])     # G3: gamma_0 keeps its reference identity
        h = mo.trajectory_hash(arr, target, sc.scene_hash(bound), rhash)
        report[tid] = {"segments": len(verdicts), "all_certified": all(v.status == ge.CLEAR for v in verdicts),
                       "hash_matches_record": h == result["tasks"][tid]["action_hash"]}
    ok = all(r["all_certified"] and r["hash_matches_record"] for r in report.values())
    out = Path(args.report) if args.report else run / "replay.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    cf.write_json(out, {"replay_passed": ok, "run": str(run), "config_sections_filled_from_current": filled,
                        **({"config_keys_filled_from_current": keys} if keys else {}),
                      "trajectories": report})
    print(json.dumps({"replay_passed": ok, "trajectories": len(report)}))
    return 0 if ok else 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Latent causal TAMP stage runner")
    sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("scene", help="G0: build, snapshot and render a fixture scene")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--fixtures", default="fixtures.json")
    q.add_argument("--case", required=True)
    q.add_argument("--out", required=True)
    q.set_defaults(func=viz.cmd_scene)
    q = sub.add_parser("check", help="G1: robot audit, corner/nominal/clutter skills, figures")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--fixtures", default="fixtures.json")
    q.add_argument("--out", required=True)
    q.add_argument("--workers", type=int, default=16)
    q.add_argument("--no-figures", action="store_true", help="re-certification run: skip the G1 figure set")
    q.set_defaults(func=fig.cmd_check)
    q = sub.add_parser("intervene", help="G2: frozen reference actions, diagnostic interventions, attribution")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--fixtures", default="fixtures.json")
    q.add_argument("--out", required=True)
    q.set_defaults(func=ifig.cmd_intervene)
    q = sub.add_parser("solve", help="G3: A0 assignment, executable move-once repair search, figures")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--fixtures", default="fixtures.json")
    q.add_argument("--out", required=True)
    q.set_defaults(func=afig.cmd_solve)
    q = sub.add_parser("case-analysis", help="G3.5: R0-R6 one-shot initial-scene protocol cases, figures")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--fixtures", default="fixtures.json")
    q.add_argument("--out", required=True)
    q.set_defaults(func=cfig.cmd_case_analysis)
    q = sub.add_parser("g4a-plan", help="G4A: freeze roots, automatic candidates, panels and budgets (no outcomes)")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--out", required=True)
    q.set_defaults(func=audit.cmd_g4a_plan)
    q = sub.add_parser("g4a-audit", help="G4A: primary episodes, first-pass teacher audit, panels, controls, figures")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--out", required=True, help="directory holding g4a_plan.json")
    q.add_argument("--limit", type=int, help="pilot: only the first N planned queries")
    q.set_defaults(func=audit.cmd_g4a_audit)
    for name, fn, hlp in (("teacher-plan", teacher.cmd_teacher_plan, "G4A follow-up: freeze the bounded teacher diagnostic"),
                          ("teacher-run", teacher.cmd_teacher_run, "G4A follow-up: guarded teacher-budget evaluations")):
        q = sub.add_parser(name, help=hlp)
        q.add_argument("--config", default="config.toml")
        q.add_argument("--g4a", default="runs/g4a_evidence_audit")
        q.add_argument("--out", required=True)
        q.set_defaults(func=fn)
    for name, fn, hlp in (("g4d-plan", g4d.cmd_g4d_plan, "G4D: freeze the sparse V2 evidence plan (no outcomes, no budget)"),
                          ("g4d-migrate", g4d.cmd_g4d_migrate, "G4D: V1->V2 migration manifest, legacy label scope audit, figures"),
                          ("g4d-legacy", g4d_gate.cmd_g4d_legacy, "G4D: revalidate archived evidence, derived omission pairs"),
                          ("g4d-finalize", g4d_gate.cmd_g4d_finalize, "G4D: final execution plan linked to the pending plan"),
                          ("g4d-run", g4d_gate.cmd_g4d_run, "G4D: guarded, resumable bounded collection"),
                          ("g4d-judge", g4d_gate.cmd_g4d_judge, "G4D: evidence from re-checked checkpoints, gate, figures")):
        q = sub.add_parser(name, help=hlp)
        q.add_argument("--config", default="config.toml")
        q.add_argument("--config-v2", default="config_v2.toml")
        q.add_argument("--out", required=True)
        q.set_defaults(func=fn)
    q = sub.add_parser("g4b", help="G4B: exact Boltzmann sampler certification on finite spaces (no planner calls)")
    q.add_argument("--out", required=True)
    q.set_defaults(func=g4b.cmd_g4b)
    q = sub.add_parser("g4c-plan", help="G4C-P0: freeze the 20-root A/B/C development plan (no teacher/planner calls)")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--config-v2", default="config_v2.toml")
    q.add_argument("--out", required=True)
    q.set_defaults(func=g4c.cmd_g4c_plan)
    for name, fn, hlp in (("g4c-pilot-finalize", g4c_pilot.cmd_g4c_pilot_finalize, "G4C-P1: authorization record + derived pilot plan"),
                          ("g4c-pilot-run", g4c_pilot.cmd_g4c_pilot_run, "G4C-P1: guarded collection of the 59 pilot units"),
                          ("g4c-pilot-judge", g4c_pilot.cmd_g4c_pilot_judge, "G4C-P1: revalidated evidence, judge, figures")):
        q = sub.add_parser(name, help=hlp)
        q.add_argument("--config", default="config.toml")
        q.add_argument("--config-v2", default="config_v2.toml")
        q.add_argument("--instruction", default="instruction-phase-md/G4C_P1_5Root_Pilot_Collection_Instructions.md")
        q.add_argument("--out", required=True)
        q.set_defaults(func=fn)
    q = sub.add_parser("g4c-d2-precheck", help="G4C-D2-P0: static witness-first construction precheck (no teacher calls)")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--out", required=True)
    q.set_defaults(func=d2_precheck.run)
    for name, fn, hlp in (("g4c-d2-poc-finalize", d2_poc.cmd_finalize, "G4C-D2-P1: authorization record + frozen POC_S plan"),
                          ("g4c-d2-poc-run", d2_poc.cmd_run, "G4C-D2-P1: guarded collection (2 fresh bases + 68 units)"),
                          ("g4c-d2-poc-judge", d2_poc.cmd_judge, "G4C-D2-P1: revalidated evidence, Q1-Q6, figures")):
        q = sub.add_parser(name, help=hlp)
        q.add_argument("--config", default="config.toml")
        q.add_argument("--config-v2", default="config_v2.toml")
        q.add_argument("--instruction", default="instruction-phase-md/G4C_D2_P1_Witness_First_POC_S_Instructions.md")
        q.add_argument("--out", required=True)
        q.set_defaults(func=fn)
    q = sub.add_parser("g4c-d2-p3a-static", help="G4C-D2-P3a: static link-0 sector certificate adjudication (no planner)")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--out", required=True)
    q.set_defaults(func=sector_cert.run)
    q = sub.add_parser("g4c-lf-p4-regression", help="G4C-LF-P4: full suite + six replays under one process-tree guard")
    q.add_argument("--out", required=True)
    q.add_argument("--cpu", type=float, default=650.0)
    q.add_argument("--wall", type=float, default=1500.0)
    q.set_defaults(func=teacher_pilot.regression)
    q = sub.add_parser("g4c-lf-p4-pilot", help="G4C-LF-P4: bounded OPNEG-v1 calibration pilot (12 queries x 3 seeds)")
    for a, d in (("--config", "config.toml"), ("--config-v2", "config_v2.toml"), ("--policy", "teacher_opneg.toml"),
                 ("--regression", "runs/g4c_lf_p4/regression")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=teacher_pilot.pilot)
    q = sub.add_parser("p5-d3lite", help="P5: D3-lite collection (8 fresh roots, 4+4 with checkpoint gate)")
    for a, d in (("--config", "config.toml"), ("--config-v2", "config_v2.toml"), ("--policy", "teacher_opneg.toml"),
                 ("--regression", "runs/p5/regression")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=d3lite_run.run)
    q = sub.add_parser("p5-g4e", help="P5: G4E-ABC proof-of-concept training (no teacher calls)")
    q.add_argument("--config", default="config.toml")
    q.add_argument("--out", required=True)
    q.set_defaults(func=abc_run.run)
    q = sub.add_parser("p6-collect", help="P6: four-batch collection of 16 fresh roots (<= 4 moves, OPNEG-v1, gated)")
    for a, d in (("--config", "config.toml"), ("--config-v2", "config_v2_r4.toml"), ("--policy", "teacher_opneg.toml"),
                 ("--regression", "runs/p6/regression"), ("--g4b", "runs/p6/g4b")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=p6_run.run)
    q = sub.add_parser("p6-train", help="P6: A/B/C retraining, ablations, baselines, sampler, sealed holdout (guarded)")
    for a, d in (("--config", "config.toml"), ("--collect", "runs/p6/collect")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=p6_train.run)
    q = sub.add_parser("p6-viz", help="P6: scene-based PNG/GIF visualizations from recorded data (guarded)")
    for a, d in (("--config", "config.toml"), ("--collect", "runs/p6/collect"), ("--train", "runs/p6/train")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=p6_viz.run)
    q = sub.add_parser("p7a-train", help="P7a: controlled ABC7 training on the frozen P6 corpus (0 teacher calls, guarded)")
    q.add_argument("--out", required=True)
    q.set_defaults(func=p7a_run.run)
    q = sub.add_parser("p7a-gate", help="P7a: machine-readable gate before any P7b teacher call")
    for a, d in (("--regression", "runs/p7/audit/regression"), ("--train", "runs/p7/a_training"), ("--out", "runs/p7/a_gate.json")):
        q.add_argument(a, default=d)
    q.set_defaults(func=p7a_gate.decide)
    q = sub.add_parser("p7a-viz", help="P7a: diagnostic GIFs (guarded)")
    for a, d in (("--train", "runs/p7/a_training"), ("--out", "visualizations/p7/a")):
        q.add_argument(a, default=d)
    q.set_defaults(func=p7a_viz.run)
    q = sub.add_parser("p7b-collect", help="P7b: two-wave collection (48 roots max, OPNEG-v1, wave-1 gate)")
    for a, d in (("--config", "config.toml"), ("--config-v2", "config_v2_r4.toml"), ("--policy", "teacher_opneg.toml"),
                 ("--regression", "runs/p7/audit/regression"), ("--p7a-gate", "runs/p7/a_gate.json")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=p7_collect.run)
    q = sub.add_parser("p7b-train", help="P7b: ABC7 retraining with frozen P7a configs, sealed test, one-shot rankings (guarded)")
    for a, d in (("--collect", "runs/p7/collection"), ("--p7a-gate", "runs/p7/a_gate.json")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=p7b_run.run)
    q = sub.add_parser("p7b-oneshot", help="P7b: independent one-shot 2D simulator episodes of the frozen rankings")
    for a, d in (("--collect", "runs/p7/collection"), ("--train", "runs/p7/training")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=p7b_oneshot.run)
    q = sub.add_parser("p7b-viz", help="P7b: GIFs from recorded data only (guarded)")
    for a, d in (("--collect", "runs/p7/collection"), ("--train", "runs/p7/training"), ("--oneshot", "runs/p7/execution")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=p7b_viz.run)
    q = sub.add_parser("p8-audit", help="P8: read-only evidence recount, manifests and factor-scope audit")
    q.set_defaults(func=p8_audit.run)
    q = sub.add_parser("p8a-train", help="P8a: matched coupling comparison on saved evidence (0 teacher calls, guarded)")
    q.add_argument("--out", required=True)
    q.set_defaults(func=p8_train.run)
    q = sub.add_parser("p8a-gate", help="P8a machine-readable gate")
    q.add_argument("--out", default="runs/p8/P8A_GATE.json")
    q.set_defaults(func=p8_gates.gate_a)
    q = sub.add_parser("p8b-run", help="P8b: selection funnel, thresholds, sampler mixing, G+M baseline (0 teacher calls, guarded)")
    q.add_argument("--out", required=True)
    q.set_defaults(func=p8b_run.run)
    q = sub.add_parser("p8b-gate", help="P8b machine-readable gate")
    q.add_argument("--out", default="runs/p8/P8B_GATE.json")
    q.set_defaults(func=p8_gates.gate_b)
    q = sub.add_parser("p8c-collect", help="P8c: two-wave kappa-balanced collection (36 roots max, OPNEG-v1, wave-1 gate)")
    for a, d in (("--config", "config.toml"), ("--config-v2", "config_v2_r4.toml"), ("--policy", "teacher_opneg.toml"),
                 ("--p8a-gate", "runs/p8/P8A_GATE.json"), ("--p8b-gate", "runs/p8/P8B_GATE.json")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=p8_collect.run)
    q = sub.add_parser("p8c-train", help="P8c: six arms x 3 seeds on fresh + historical data; frozen P8d rankings (guarded)")
    q.add_argument("--out", required=True)
    q.set_defaults(func=p8c_train.run)
    q = sub.add_parser("p8d-run", help="P8d: independent one-shot episodes of the frozen P8c rankings")
    for a, d in (("--collect", "runs/p8/collection"), ("--rankings", "runs/p8/c_training/rankings.json")):
        q.add_argument(a, default=d)
    q.add_argument("--out", required=True)
    q.set_defaults(func=p8d_run.run)
    q = sub.add_parser("p8-viz", help="P8 diagnostics from saved records (guarded)")
    q.add_argument("--stage", default="all", choices=["a", "b", "cd", "all"])
    q.add_argument("--cpu", default="600")
    q.add_argument("--out", default="visualizations/p8")
    q.set_defaults(func=p8_viz.run)
    q = sub.add_parser("replay", help="G1-G3.5: independently re-certify a run's stored trajectories")
    q.add_argument("--run", required=True)
    q.add_argument("--report", help="where to write the replay record (default: <run>/replay.json)")
    q.add_argument("--config", default="config.toml", help="only for config sections the run predates")
    q.set_defaults(func=cmd_replay)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
