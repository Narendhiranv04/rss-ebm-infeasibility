"""G4A follow-up: bounded offline teacher target-budget diagnostic (OFFLINE_TEACHER only).

Three frozen all-stay cases (g4a_r04, r14, r15) are re-evaluated under the named
schedule TEACHER_BUDGET_V1: B1 = 2x and B2 = 4x the archived RRT node allowance
(planner2d.max_nodes, with its per-query wall safety cap scaled alike) and the
target skill's failure-based search allowances (skill2d.max_plan_failures, the
number of UNRESOLVED RRT connector results tolerated while enumerating the finite
pick x place pairs; skill2d.max_direct_pairs, the direct-phase pair cap). Grasp
family, geometry, tolerances, end policy and seeds (the archived keys) are unchanged.

Each evaluation runs in its own single-threaded process under an in-flight CPU guard
(psutil process tree, cancelled by process group). Only a completed, independently
re-checked SAFE_RETREAT witness is positive; censoring and budget ends stay UNKNOWN.
Nothing here feeds a primary episode or the deployment configuration.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import resource
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil

import config as cf
import data
import evaluate as ev
import geometry as ge
import motion as mo
import skill as sk
from scene import scene_from_dict

SCHEDULE, ROOTS, TIERS = "TEACHER_BUDGET_V1", ("g4a_r04", "g4a_r14", "g4a_r15"), (("B1", 2), ("B2", 4))
LIMITS = {"max_evaluations": 6, "cpu_per_eval_s": 900.0, "cpu_aggregate_s": 7200.0, "wall_total_s": 3 * 3600.0,
          "guard_margin_s": 30.0, "poll_s": 0.25, "workers": 1}
THREADS = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}


def teacher_cfg(cfg: dict, factor: int) -> dict:
    """Deep-copied, validated teacher overrides; the deployment dict is never touched."""
    t = copy.deepcopy(cfg)
    t["planner2d"]["max_nodes"] *= factor
    t["planner2d"]["max_wall_seconds"] *= factor
    t["skill2d"]["max_plan_failures"] *= factor
    t["skill2d"]["max_direct_pairs"] *= factor
    return cf.validate_config(t)


def _source_hash(repo: Path) -> dict:
    run = lambda *a: subprocess.run(["git", *a], cwd=repo, capture_output=True, text=True).stdout  # noqa: E731
    untracked = sorted(run("ls-files", "--others", "--exclude-standard").split())
    blob = run("diff", "HEAD") + "".join(f + hashlib.sha256((repo / f).read_bytes()).hexdigest()
                                          for f in untracked if f.endswith(".py"))
    return {"head": run("rev-parse", "HEAD").strip(), "uncommitted_diff_sha256": hashlib.sha256(blob.encode()).hexdigest(),
            "untracked_py": [f for f in untracked if f.endswith(".py")]}


def make_plan(cfg: dict, g4a: Path, repo: Path) -> dict:
    """Freeze cases, budgets, order, limits and identities before any new outcome."""
    plan4a, rows = json.loads((g4a / "g4a_plan.json").read_text()), json.loads((g4a / "candidate_labels.json").read_text())
    prim = json.loads((g4a / "primary_episodes.json").read_text())
    cases = []
    for q in ROOTS:
        p = next(x for x in plan4a["queries"] if x["query_id"] == q)
        stay = next(r for r in rows if r["query_id"] == q and r["moved"] == 0)
        if not (stay["first_pass"] == data.UNKNOWN and stay["unknown_reason"] == "TARGET_UNRESOLVED:PLANNER_BUDGET"):
            raise ValueError(f"{q}: archived all-stay is not target-budget UNRESOLVED; collection stopped")
        scene = scene_from_dict(p["scene"])
        cases.append({"query_id": q, "assignment": {}, "scene": p["scene"], "goal_cell": p["goal_cell"], "hashes": p["hashes"],
                      "action": p["action"], "robot": {"q": list(scene.robot.q_rad), "gripper_gap_m": scene.robot.gripper_gap_m,
                                                       "attachment": scene.robot.attachment},
                      "key": f"{q}/audit/0/target", "seed_base": cfg["planner2d"]["seed"],
                      "baseline": {"first_pass": {"key": f"{q}/audit/0/target", "status": stay["first_pass"],
                                                  "reason": stay["unknown_reason"], "per_query_records_archived": False},
                                   "primary": {"key": f"{q}/primary/target", "status": prim[q]["target"]["status"],
                                               "reason": prim[q]["target"]["reason"], "seconds": prim[q]["target"]["seconds"]}}})
    tiers = {"B0": {"factor": 1, **_budget(cfg)}, **{n: {"factor": f, **_budget(teacher_cfg(cfg, f))} for n, f in TIERS}}
    plan = {"stage": "G4A_FOLLOWUP", "schedule": SCHEDULE, "evidence_namespace": "OFFLINE_TEACHER", "cases": cases,
            "tiers": tiers, "order": [[t, q] for t, _ in TIERS for q in ROOTS], "limits": LIMITS, "threads": THREADS,
            "stopping": "per case: skip later tiers once a checked witness or accepted assignment-wide negative exists; "
                        "aggregate CPU or wall guard ends collection; censored evaluations stay UNKNOWN",
            "unchanged": ["grasp family and order", "geometry/collision/margins", "tolerances", "SAFE_RETREAT policy",
                          "seeds (archived keys)", "candidate generation and ranking", "primary episodes"],
            "deployment_config_hash": cf.config_hash(cfg), "g4a_plan_hash": plan4a["plan_hash"],
            "robot_profile_hash": cf.robot_profile_hash(cfg), "source": _source_hash(repo)}
    plan["plan_hash"] = cf.stable_hash(plan)
    return plan


def _budget(c: dict) -> dict:
    return {"max_nodes": c["planner2d"]["max_nodes"], "max_wall_seconds_per_query": c["planner2d"]["max_wall_seconds"],
            "max_plan_failures": c["skill2d"]["max_plan_failures"], "max_direct_pairs": c["skill2d"]["max_direct_pairs"],
            "grasp_family": 2 * c["skill2d"]["approach_dirs"], "config_hash": cf.config_hash(c)}


# ------------------------------------------------------------------ guard
def guard(argv: list, cpu_limit: float, wall_limit: float, heartbeat: Path, margin: float, poll: float, cwd) -> dict:
    """Run argv in a new session; cancel the whole process group once tree CPU reaches
    cpu_limit - margin or wall reaches wall_limit. Final CPU = max(rusage of the reaped
    tree, last in-flight tree sample): descendants killed before their own parent reaped
    them are absent from RUSAGE_CHILDREN but present in the sample. Overshoot is measured."""
    before = resource.getrusage(resource.RUSAGE_CHILDREN)
    t0, stop, peak, phase = time.perf_counter(), "COMPLETED", 0.0, None
    proc = subprocess.Popen(argv, cwd=cwd, start_new_session=True, env={**os.environ, **THREADS},
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    ps = psutil.Process(proc.pid)
    while proc.poll() is None:
        try:
            tree = [ps] + ps.children(recursive=True)
            cpu = sum(p.cpu_times().user + p.cpu_times().system for p in tree if p.is_running())
            cpu += ps.cpu_times().children_user + ps.cpu_times().children_system
        except psutil.Error:
            cpu = peak
        peak = max(peak, cpu)
        wall = time.perf_counter() - t0
        if peak >= cpu_limit - margin or wall >= wall_limit:
            stop = "CPU_CENSORED" if peak >= cpu_limit - margin else "WALL_CENSORED"
            phase = heartbeat.read_text() if heartbeat.exists() else None
            os.killpg(proc.pid, signal.SIGKILL)
            break
        time.sleep(poll)
    err = proc.stderr.read().decode()[-2000:] if proc.stderr else ""
    proc.wait()
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    used = max((after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime), peak)
    if stop == "COMPLETED" and proc.returncode != 0:
        stop = "ERROR"
    return {"stop": stop, "pid": proc.pid, "returncode": proc.returncode, "cpu_s": round(used, 2),
            "wall_s": round(time.perf_counter() - t0, 2), "trigger_cpu_s": cpu_limit - margin, "cpu_limit_s": cpu_limit,
            "overshoot_s": round(max(0.0, used - cpu_limit), 2), "active_phase": phase, "stderr_tail": err if stop == "ERROR" else ""}


# ------------------------------------------------------------------ worker (separate process)
def check_witness(robot, cfg: dict, scene, state, action: dict, arr: dict) -> dict:
    """Independent re-check of a target record: certifies clear in the initial scene, starts
    bit-exactly at the archived state, ends in a released, open-handed retreat, and places
    the target within the goal tolerance."""
    delta, eps, ev_ = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"], cfg["geometry"]["certify_max_evals"]
    try:
        clear = all(v.status == ge.CLEAR for v in mo.certify_trajectory(robot, scene, action["target_id"], arr, delta, eps, ev_))
    except ValueError:                                      # malformed record (e.g. attachment inconsistent): not a witness
        clear = False
    start = bool(np.array_equal(arr["q"][0], np.asarray(state.q)) and arr["gap"][0] == state.gap)
    retreat = str(arr["phase"][-1]) == "retreat" and arr["gap"][-1] == robot.open_gap
    goal = float(np.linalg.norm(np.asarray(arr["place_xy"]) - np.asarray(action["goal_xy_m"]))) <= action["goal_tolerance_m"]
    return {"certified_clear": clear, "starts_at_archived_state": start, "ends_in_safe_retreat": retreat,
            "goal_within_tolerance": goal, "accepted": bool(clear and start and retreat and goal)}


def worker(job_path: str) -> None:
    import planner as pn
    job = json.loads(Path(job_path).read_text())
    case, out, hb = job["case"], Path(job["out"]), Path(job["heartbeat"])
    cfg = cf.load_config(job["config"])
    tcfg = teacher_cfg(cfg, job["factor"])
    scene = scene_from_dict(case["scene"])
    query = data.make_query(cfg, case["query_id"], scene, "t0", case["goal_cell"], {}, "SAFE_RETREAT")
    if query["hashes"] != case["hashes"] or query["learner"]["action"] != case["action"]:
        raise SystemExit("identity mismatch with the frozen plan")
    real = pn.plan

    def traced(checker, qa, qb, gap, pcfg, max_evals, key, *a, **k):
        hb.write_text(key.split("/")[-1])
        return real(checker, qa, qb, gap, pcfg, max_evals, key, *a, **k)
    pn.plan = traced
    robot, state = ge.make_robot(tcfg), ev.start(scene, query)
    t = ev.evaluate_target(robot, tcfg, state, query["learner"]["action"], case["key"])
    res = t["result"]
    rec = {k: t.get(k) for k in ("status", "reason", "goal_achieved", "safe_retreat_status", "home_return_status",
                                  "complete_target_status", "place_error_m", "wall_clock_dependent")}
    rec["picks_ok"] = sum(c.stage == "OK" for c in res.picks) if res else None
    rec["places_ok"] = sum(c.stage == "OK" for c in res.places) if res else None
    rec["queries"] = [{"label": lab, "status": p.status, "reason": p.reason, "nodes": p.nodes, "seconds": round(p.seconds, 2),
                       "wall_clock_cut": p.wall_clock_cut} for lab, p in (res.plans if res else [])]
    if res is not None and res.traj is not None and t["status"] == sk.VERIFIED:
        arr = res.traj.arrays()
        rec["witness_check"] = check_witness(robot, tcfg, scene, state, query["learner"]["action"], arr)
        np.savez_compressed(out.with_suffix(".tmp.npz"), **arr)
        os.replace(out.with_suffix(".tmp.npz"), out.with_suffix(".npz"))
    Path(str(out) + ".tmp").write_text(json.dumps(rec, default=str))
    os.replace(str(out) + ".tmp", out)


# ------------------------------------------------------------------ schedule, records, figures
def run_schedule(plan: dict, out: Path, config_path: str, repo: Path) -> tuple:
    """B1 for r04, r14, r15, then B2 for those still unknown; aggregate CPU and wall guards."""
    L, spent, t0, trace, done = plan["limits"], 0.0, time.perf_counter(), [], {}
    by = {c["query_id"]: c for c in plan["cases"]}
    for tier, q in plan["order"]:
        row = {"tier": tier, "query_id": q, "budget": plan["tiers"][tier]}
        remaining, wall_left = L["cpu_aggregate_s"] - spent, L["wall_total_s"] - (time.perf_counter() - t0)
        if done.get(q) == data.POSITIVE:
            trace.append({**row, "stop": "SKIPPED_RESOLVED_AT_LOWER_TIER", "status": "NOT_RUN"})
            continue
        if sum(e.get("pid") is not None for e in trace) >= L["max_evaluations"] or remaining <= L["guard_margin_s"] \
                or wall_left <= 0:
            trace.append({**row, "stop": "NOT_RUN_AGGREGATE_LIMIT", "status": "NOT_RUN"})
            continue
        stem, hb = out / f"eval_{q}_{tier}", out / f".heartbeat_{q}_{tier}"
        job = out / f".job_{q}_{tier}.json"
        job.write_text(json.dumps({"case": by[q], "factor": plan["tiers"][tier]["factor"], "config": config_path,
                                   "out": str(stem.with_suffix(".json")), "heartbeat": str(hb)}))
        start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        g = guard([sys.executable, "-c", "import sys, teacher; teacher.worker(sys.argv[1])", str(job)],
                  min(L["cpu_per_eval_s"], remaining), wall_left, hb, L["guard_margin_s"], L["poll_s"], repo)
        spent += g["cpu_s"]
        res = json.loads(stem.with_suffix(".json").read_text()) if g["stop"] == "COMPLETED" and stem.with_suffix(".json").exists() else None
        ok = bool(res and res["status"] == sk.VERIFIED and res.get("witness_check", {}).get("accepted"))
        done[q] = data.POSITIVE if ok else data.UNKNOWN
        trace.append({**row, **g, "start_utc": start, "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "status": done[q], "evaluator": res, "aggregate_cpu_s": round(spent, 2)})
        for f in (job, hb):
            f.unlink(missing_ok=True)
    return trace, spent, time.perf_counter() - t0


def cmd_teacher_plan(args) -> int:
    cfg, out, repo = cf.load_config(args.config), Path(args.out), Path.cwd()
    out.mkdir(parents=True, exist_ok=True)
    if (out / "plan.json").exists():
        raise SystemExit("plan.json already frozen; refusing to overwrite")
    plan = make_plan(cfg, Path(args.g4a), repo)
    cf.write_json(out / "plan.json", plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "cases": ROOTS, "source": plan["source"]}))
    return 0


def cmd_teacher_run(args) -> int:
    import coverage
    cfg, out, repo, g4a = cf.load_config(args.config), Path(args.out), Path.cwd(), Path(args.g4a)
    plan = json.loads((out / "plan.json").read_text())
    if cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"}) != plan["plan_hash"] \
            or plan["deployment_config_hash"] != cf.config_hash(cfg):
        raise SystemExit("plan or deployment config changed since freezing")
    if (out / "teacher_budget_results.json").exists():
        raise SystemExit("results already collected; refusing to re-run")
    trace, spent, wall = run_schedule(plan, out, args.config, repo)
    rows, prim = json.loads((g4a / "candidate_labels.json").read_text()), json.loads((g4a / "primary_episodes.json").read_text())
    recon = {"source_records": [str(g4a / f) for f in ("candidate_labels.json", "primary_episodes.json", "seed_panel.json",
                                                       "result.json")],
             "certificates": coverage.certificate_reconciliation(rows), "primary_target_calls": coverage.call_accounting(prim)}
    cf.write_json(out / "reporting_reconciliation.json", recon)
    cf.write_json(out / "event_trace.json", [{k: v for k, v in e.items() if k != "evaluator"} for e in trace])
    zero = {r["root_id"] for r in rows if r["moved"] == 0 and r["first_pass"] == data.POSITIVE}
    new = sorted({e["query_id"] for e in trace if e["status"] == data.POSITIVE})
    results = {"schedule": SCHEDULE, "plan_hash": plan["plan_hash"], "evaluations": trace, "cpu_used_s": round(spent, 2),
               "wall_s": round(wall, 1), "new_zero_move_witnesses": new,
               "zero_move_roots": {"original_first_pass": sorted(zero), "diagnostic_enriched": sorted(zero | set(new))},
               "cpu_per_new_witness_s": round(spent / len(new), 1) if new else None}
    cf.write_json(out / "teacher_budget_results.json", results)
    cf.write_json(out / "config.json", {"deployment_config_hash": cf.config_hash(cfg), "deployment_config": cfg,
                                        "teacher_tiers": {n: teacher_cfg(cfg, f) for n, f in TIERS}})
    import csv
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "query_id", "status", "stop", "cpu_s", "wall_s", "max_nodes", "max_plan_failures", "evaluator_status",
                    "evaluator_reason", "connector_queries"])
        for e in trace:
            r = e.get("evaluator") or {}
            w.writerow([e["tier"], e["query_id"], e["status"], e["stop"], e.get("cpu_s"), e.get("wall_s"), e["budget"]["max_nodes"],
                        e["budget"]["max_plan_failures"], r.get("status"), r.get("reason"), len(r.get("queries", [])) or None])
    import teacher_figures
    teacher_figures.figures(out, cfg, plan, trace, recon)
    cf.write_manifest(out, {"planner_base": cfg["planner2d"]["seed"], "keys": [c["key"] for c in plan["cases"]]},
                      cf.config_hash(cfg), {"plan_hash": plan["plan_hash"], "schedule": SCHEDULE})
    print(json.dumps({"new_zero_move_witnesses": new, "cpu_used_s": round(spent, 1), "stops": [e["stop"] for e in trace]}))
    return 0
