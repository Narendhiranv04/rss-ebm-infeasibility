"""G4D bounded collection (report 0011): final execution plan, guarded unit worker with
atomic evidence checkpoints, resumable scheduler and checkpoint revalidation.

The final plan retains the pending plan's 20 units verbatim (roots, sequences,
interventions, seed keys, order, action/search policies) and adds the authorized
resource envelope plus a source manifest. Each unit runs in its own single-threaded
process under teacher.guard (process-tree CPU, process-group cancellation). Stages are
checkpointed before the next starts: accepted certificates; the fixed-order relocation
result with its first-witness cost; the checked goal + safe-retreat target witness
(before the optional HOME diagnostic). Evidence status is rebuilt from revalidated
checkpoints, separately from the process stop status, so cancellation never erases
complete evidence and partial trajectories never count. A unit is never rerun.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import certificates as cert
import config as cf
import data
import effects as ef
import evaluate as ev
import g4d
import geometry as ge
import legacy
import ordered as od
import teacher
import verify as vf
from scene import scene_from_dict, scene_hash

LIMITS = {"cpu_per_unit_s": 900.0, "guard_margin_s": 30.0, "cpu_aggregate_s": 18000.0, "workers": 1,
          "wall_per_unit_s": 1800.0, "wall_total_s": 21600.0, "poll_s": 0.25, "orchestration_reserve_s": 60.0}
SOURCES = ("g4d_run.py", "g4d.py", "legacy.py", "teacher.py", "ordered.py", "effects.py", "sequences.py", "verify.py",
           "evaluate.py", "skill.py", "planner.py", "motion.py", "geometry.py", "certificates.py", "data.py", "proposals.py",
           "scene.py", "cases.py", "interventions.py", "viz.py", "config.py", "config.toml", "config_v2.toml", "fixtures.json",
           "runs/g2_interventions/scenes.json", "runs/g2_interventions/arrays.npz", "runs/g2_interventions/result.json",
           "runs/g4d_migration/g4d_plan.json")
STAGES = ("certificates", "relocation", "target_witness", "final")


def _sha(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _rd(path: Path):
    return json.loads(path.read_text())


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def _atomic_npz(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.name[:-4] + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


# ------------------------------------------------------------------ final execution plan
def make_final_plan(cfg, v2, robot, root: Path) -> dict:
    """Link to the pending plan: its file and hash are verified, the corrected code must
    regenerate identical units and roots, and the units are copied verbatim."""
    pend_path = root / "runs/g4d_migration/g4d_plan.json"
    pend = _rd(pend_path)
    if cf.stable_hash({k: v for k, v in pend.items() if k != "plan_hash"}) != pend["plan_hash"]:
        raise SystemExit("pending plan modified")
    regen = json.loads(json.dumps(g4d.make_plan(cfg, v2, robot, root)))        # compare as stored (tuples -> lists)
    same = regen["units"] == pend["units"] and {k: (r["scene_hash"], r["hashes"]) for k, r in regen["roots"].items()} == \
        {k: (r["scene_hash"], r["hashes"]) for k, r in pend["roots"].items()} and regen["cases"] == pend["cases"]
    if not same:
        raise SystemExit("corrected code no longer regenerates the pending units/roots; plan not finalized")
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    plan = {"stage": "G4D_COLLECTION", "protocol": data.PROTOCOL_V2, "status": "FINAL_AUTHORIZED",
            "pending_plan": {"path": str(pend_path), "file_sha256": _sha(pend_path), "plan_hash": pend["plan_hash"],
                             "relationship": "units, roots, cases, seeds, order and action/search policies copied verbatim "
                             "(regenerated identically by the corrected code); the pending file is unchanged",
                             "why_hashes_changed": "source files were corrected (report 0011: episode audit, per-call "
                             "relocation budget, raw-proposal validation, negative revalidation, termination events, "
                             "context identity, checkpoint hook, runner), and the authorized envelope was added"},
            "documents": pend["documents"], "roots": pend["roots"], "cases": pend["cases"], "units": pend["units"],
            "omission_plans": pend["omission_plans"], "seeds": pend["seeds"], "policies": {
                **pend["policies"], "home_diagnostic": "kept ON (config.toml unchanged); the checked goal+safe-retreat witness "
                "is checkpointed through evaluate.evaluate_safe_retreat(on_witness) BEFORE the HOME diagnostic starts",
                "search_schedule": "unchanged: planner2d.max_nodes 4000, relocation failures 2, target failures 4, "
                "48-grasp family, margins, tolerances, supported transport"},
            "limits": LIMITS, "threads": teacher.THREADS,
            "call_caps": {"relocation_skills": pend["budget"]["work_upper_bound"]["relocation_skills"],
                          "target_evaluations": pend["budget"]["work_upper_bound"]["target_evaluations"]},
            "admission": "a unit starts only if spent CPU (units + coordinator) + cpu_per_unit_s + orchestration_reserve_s "
                         "<= cpu_aggregate_s; its wall limit is min(wall_per_unit_s, remaining global wall)",
            "resume": "only never-started units run; persisted spent CPU and the original global deadline are reused; a "
                      "STARTED unit without an end record becomes INTERRUPTED_AMBIGUOUS and is not repeated",
            "config_hash": cf.config_hash(cfg), "config_v2_hash": cf.stable_hash(v2), "robot_profile_hash": cf.robot_profile_hash(cfg),
            "source": {"head": head, "files": {f: _sha(root / f) for f in SOURCES}}}
    plan["plan_hash"] = cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    return plan


def verify_sources(plan: dict, root: Path) -> list:
    bad = [f for f, h in plan["source"]["files"].items() if _sha(root / f) != h]
    if cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"}) != plan["plan_hash"]:
        bad.append("final plan hash")
    return bad


# ------------------------------------------------------------------ unit worker (separate process)
def unit_scene(cfg, robot, plan: dict, unit: dict):
    """(scene, sequence, query, base key) for one unit, with identity checks against the plan."""
    root = plan["roots"][unit["root"]]
    base = scene_from_dict(root["scene"])
    if scene_hash(base) != root["scene_hash"]:
        raise SystemExit(f"{unit['unit']}: root scene hash mismatch")
    act = data.make_query(cfg, "x", base, "t0", root["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
    scene, seq = base, unit.get("sequence", [])
    if unit["kind"] == "SCENE_TARGET_ONLY":
        sub = ef.pose_substitution(robot, cfg, base, act, unit["edit"])
        if not sub["eligible"] or sub["intervened_scene_hash"] != unit["intervened_scene_hash"]:
            raise SystemExit(f"{unit['unit']}: intervention ineligible or changed")
        scene = sub["scene"]
    elif unit["kind"] == "ROBOT_START_TARGET_ONLY":
        rs = ef.robot_start_substitution(robot, cfg, base, act, unit["q_start"])
        if not rs["eligible"] or rs["intervened_scene_hash"] != unit["intervened_scene_hash"]:
            raise SystemExit(f"{unit['unit']}: robot start ineligible or changed")
        scene = rs["scene"]
    query = data.make_query(cfg, f"g4d_{unit['unit']}", scene, "t0", root["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
    if unit["kind"] == "ORDERED_TEACHER" and query["hashes"] != root["hashes"]:
        raise SystemExit(f"{unit['unit']}: query identity mismatch")
    key = unit["target_key"].rsplit("/target", 1)[0]
    if "relocation_key" in unit and unit["relocation_key"] != key + "/reloc":
        raise SystemExit(f"{unit['unit']}: seed keys inconsistent")
    return scene, seq, query, key


def worker(job_path: str) -> None:
    import planner as pn
    job = _rd(Path(job_path))
    cfg, v2, plan, unit = cf.load_config(job["config"]), cf.load_config_v2(job["config_v2"]), _rd(Path(job["plan"])), job["unit"]
    udir, hb = Path(job["unit_dir"]), Path(job["heartbeat"])
    robot = ge.make_robot(cfg)
    real = pn.plan

    def traced(checker, qa, qb, gap, pcfg, max_evals, key, *a, **k):
        hb.write_text(key)
        return real(checker, qa, qb, gap, pcfg, max_evals, key, *a, **k)
    pn.plan = traced
    scene, seq, query, key = unit_scene(cfg, robot, plan, unit)

    def put(stage, rec, arrays):
        if arrays:
            _atomic_npz(udir / f"{stage}.npz", {f"r{i}__{k}": v for i, a in enumerate(arrays) for k, v in a.items()})
        _atomic_json(udir / f"{stage}.json", {"unit": unit["unit"], "stage": stage, **rec})
    act = query["learner"]["action"]
    tcalls = []

    def target_eval(st, k):
        tcalls.append(od.state_record(st))
        return ev.evaluate_target(robot, cfg, st, act, k, on_witness=lambda rec, arr: put(
            "target_witness", {**rec, "invocation": tcalls[-1]}, [arr]))
    rec = od.teach_sequence(robot, cfg, v2, query, scene, seq, target_eval, key, cf.robot_profile_hash(cfg), checkpoint=put)
    t = rec.get("target") or {}
    put("final", {"ordered_feasibility_status": rec["ordered_feasibility_status"], "calls": rec["calls"],
                  "target_invocations": tcalls, "target": {k: v for k, v in t.items() if k not in ("result",)},
                  "target_start_matches_achieved": rec["target_start_matches_achieved"],
                  "relocation_attempt_status": rec["relocation_attempt_status"], "static_status": rec["static_status"]}, None)


# ------------------------------------------------------------------ scheduler (resumable, guarded)
def _worker_argv(job: Path) -> list:
    return [sys.executable, "-c", "import sys, g4d_run; g4d_run.worker(sys.argv[1])", str(job)]


def run_collection(plan: dict, out: Path, paths: dict, root: Path, worker_argv=_worker_argv) -> dict:
    """Execute the plan's units in order under the envelope; ledger persisted atomically."""
    L, lp = plan["limits"], out / "ledger.json"
    out.mkdir(parents=True, exist_ok=True)
    led = _rd(lp) if lp.exists() else {"plan_hash": plan["plan_hash"], "deadline_epoch": time.time() + L["wall_total_s"],
                                        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "unit_cpu_s": 0.0, "coordinator_cpu_s": 0.0, "units": {}, "resumes": 0}
    if led["plan_hash"] != plan["plan_hash"]:
        raise SystemExit("ledger belongs to another plan")
    led["resumes"] += lp.exists()
    for uid, st in led["units"].items():
        if st["state"] == "STARTED":
            st.update(state="ENDED", stop="INTERRUPTED_AMBIGUOUS", note="started before an interruption; not repeated")
    _atomic_json(lp, led)
    c0 = time.process_time()
    for unit in plan["units"]:
        uid = unit["unit"]
        if uid in led["units"]:
            continue
        led["coordinator_cpu_s"] += time.process_time() - c0
        c0 = time.process_time()
        spent, wall_left = led["unit_cpu_s"] + led["coordinator_cpu_s"], led["deadline_epoch"] - time.time()
        if spent + L["cpu_per_unit_s"] + L["orchestration_reserve_s"] > L["cpu_aggregate_s"] or wall_left <= L["guard_margin_s"]:
            led["units"][uid] = {"state": "ENDED", "stop": "NOT_RUN_BUDGET" if wall_left > L["guard_margin_s"] else "NOT_RUN_WALL"}
            _atomic_json(lp, led)
            continue
        udir = out / "units" / uid.replace(":", "__").replace(">", "-")
        udir.mkdir(parents=True, exist_ok=True)
        job, hb = udir / "job.json", udir / "heartbeat.txt"
        _atomic_json(job, {"unit": unit, "plan": str(paths["plan"]), "config": str(paths["config"]),
                           "config_v2": str(paths["config_v2"]), "unit_dir": str(udir), "heartbeat": str(hb)})
        led["units"][uid] = {"state": "STARTED", "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "dir": str(udir.relative_to(out))}
        _atomic_json(lp, led)
        g = teacher.guard(worker_argv(job), L["cpu_per_unit_s"], min(L["wall_per_unit_s"], wall_left), hb,
                          L["guard_margin_s"], L["poll_s"], root)
        led["unit_cpu_s"] += g["cpu_s"]
        led["units"][uid] = {**led["units"][uid], "state": "ENDED", **{k: g[k] for k in (
            "stop", "returncode", "cpu_s", "wall_s", "trigger_cpu_s", "cpu_limit_s", "overshoot_s", "active_phase", "stderr_tail")},
            "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "checkpoints": sorted(p.stem for p in udir.glob("*.json") if p.stem in STAGES),
            "aggregate_cpu_s": round(led["unit_cpu_s"] + led["coordinator_cpu_s"], 2)}
        _atomic_json(lp, led)
        if g["overshoot_s"] > 0 or led["unit_cpu_s"] + led["coordinator_cpu_s"] > L["cpu_aggregate_s"]:
            led["enforcement_failure"] = f"{uid}: overshoot {g['overshoot_s']} s or aggregate exceeded"
            _atomic_json(lp, led)
            break
    led["coordinator_cpu_s"] += time.process_time() - c0
    led["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_json(lp, led)
    return led


# ------------------------------------------------------------------ checkpoint revalidation -> unit evidence
def _arrays(path: Path) -> list:
    if not path.exists():
        return []
    z = np.load(path)
    n = 1 + max(int(k.split("__")[0][1:]) for k in z.files)
    return [{k.split("__")[1]: z[k] for k in z.files if k.startswith(f"r{i}__")} for i in range(n)]


def _scene_at(base, st: dict):
    from dataclasses import replace as rp
    objs = tuple(rp(o, xy_m=tuple(st["objects"][o.id][2]), cell=st["objects"][o.id][3]) for o in base.objects)
    return rp(base, objects=objs)


def unit_evidence(cfg, v2, robot, plan: dict, unit: dict, udir: Path) -> dict:
    """Evidence rebuilt only from checkpoints that re-check now: accepted certificates (the
    independent checker is re-run on each stored record), a complete relocation witness
    (each record re-certified in its pre-move scene from the previous achieved state;
    cost recomputed), and a target witness that starts at the last achieved state and
    ends in a certified safe retreat."""
    scene, seq, query, _ = unit_scene(cfg, robot, plan, unit)
    learner, s0 = query["learner"], ev.start(scene, query)
    out = {"unit": unit["unit"], "kind": unit["kind"], "root": unit["root"], "sequence": seq, "certificates": [],
           "relocation": None, "cost": None, "target": None, "evidence": [], "checks": {}}
    cp = {s: _rd(udir / f"{s}.json") for s in STAGES if (udir / f"{s}.json").exists()}
    cells = {o: d for o, d in seq}
    for c in cp.get("certificates", {}).get("certificates", []):
        rec = {k: v for k, v in c.items() if k not in ("scope", "checker_accepted", "checker_reasons")}
        ok, why = cert.check(robot, scene, learner, cells, rec, cfg)
        out["certificates"].append({"kind": c["kind"], "scope": cert.CERTIFICATE_SCOPE.get(c["kind"]), "accepted": ok,
                                    "reasons": why, "worker_accepted": c["checker_accepted"]})
        out["evidence"].append({"kind": "certificate", "validated": ok, "certificate": c["kind"],
                                "scope": cert.CERTIFICATE_SCOPE.get(c["kind"]), "source": "g4d_collection"})
    final_state, complete = s0, not seq
    if "relocation" in cp:
        r, recs = cp["relocation"], _arrays(udir / "relocation.npz")
        states = r["relocation"]["achieved_states"]
        d, e, n = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"], cfg["geometry"]["certify_max_evals"]
        ok, prev = len(recs) == len(states) - 1, (np.asarray(s0.q), s0.gap)
        for i, a in enumerate(recs):
            pre = _scene_at(scene, states[i])
            ok &= legacy._certified(robot, pre, seq[i][0], a, d, e, n)
            ok &= bool(np.array_equal(a["q"][0], prev[0]) and a["gap"][0] == prev[1])
            ok &= bool(np.array_equal(a["q"][-1], np.asarray(states[i + 1]["q"])) and str(a["phase"][-1]) == "retreat")
            prev = (a["q"][-1], a["gap"][-1])
        complete = bool(r["relocation"]["complete_witness"] and ok and len(recs) == len(seq))
        jt = od.joint_travel(robot, recs) if recs else {"J": 0.0}
        out["relocation"] = {**{k: r["relocation"][k] for k in ("stop", "failed_action", "reason", "normalized_reasons",
                                                                "relocation_plans")}, "complete_witness": complete,
                             "records_recertified": bool(ok), "partial_records": len(recs) if not complete else 0}
        out["cost"] = {**r["cost"], "recomputed_J": jt["J"] if complete else None,
                       "label_available": bool(complete and r["cost"]["label_available"]
                                               and abs((jt["J"] or 0.0) - (r["cost"]["J"] or 0.0)) < 1e-12)}
        if complete:
            final_state = vf.RepairState(_scene_at(scene, states[-1]), tuple(states[-1]["q"]), states[-1]["gap"],
                                         tuple(states[-1]["moved"]))
        out["checks"]["joint_continuity"] = bool(ok)
    if "target_witness" in cp and complete:
        tw, arr = cp["target_witness"], _arrays(udir / "target_witness.npz")[0]
        w = {k: bool(v) for k, v in teacher.check_witness(robot, cfg, final_state.scene, final_state, learner["action"],
                                                          arr).items()}
        inv = tw.get("invocation", {})
        start = inv == od.state_record(final_state)
        out["target"] = {"witness_check": w, "invocation_equals_achieved": start, "home_return_status": (cp.get("final", {})
                         .get("target", {}) or {}).get("home_return_status")}
        out["checks"]["target_start_equality"] = bool(start and w["starts_at_archived_state"])
        out["checks"]["achieved_state_consistency"] = start
        if w["accepted"] and start:
            out["evidence"].append({"kind": "complete_witness", "scope": data.ORDER, "source": "g4d_collection"})
    elif "final" in cp:
        t = cp["final"].get("target") or {}
        out["target"] = {"status": t.get("status"), "reason": t.get("reason"), "termination": t.get("termination"),
                         "normalized_reason": t.get("normalized_reason")} if t else None
    out["label"] = data.ordered_aggregate(out["evidence"])["label"]
    out["calls"] = (cp.get("final") or {}).get("calls")
    out["checkpoints"] = sorted(cp)
    return out
