#!/usr/bin/env python3
"""Bounded v2 mock orchestrator; no full-collection entry point is exposed.

Subcommands:
  init          Freeze metadata-only v2 plan, pin hashes, preserve v1 splits.
  static        Validate task-independent goals, 2-edit contexts, manifests and safety.
  mock          Execute a BOUNDED 25-root v2 mock (requires --execute-mock).
  interrupt     SIGKILL/resume one bounded adaptive pilot (requires --execute-mock).
  decide        Audit completed pilot runs; freeze v2 production policy only if ALL gates pass.

REQUIRES a local checkout prepared by prepare_v2.py, the robot teacher environment,
25-worker thermal instrumentation and the existing 5,000 source roots.
Neither this script nor prepare_v2.py can run production collection.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

V2 = "launch_ready_2d_v2"
SOURCE_REL = Path("gh-yadu-iiith/planar_2d/dataset_relocations_v1")
NAMES = ("v2_ref64_25", "v2_adaptive25_a", "v2_adaptive25_b", "v2_adaptive_serial")


def djson(p: Path):
    return json.loads(p.read_text())


def stable_hash(data: bytes):
    return hashlib.sha256(data).hexdigest()


def file_hash(p: Path):
    return stable_hash(p.read_bytes())


def setup(repo: Path):
    root = repo.resolve() / SOURCE_REL
    code = root / "code_v2"
    L = root / V2
    if not (code / "placement_teacher.py").is_file() or not (L / "configs" / "launch_config.json").is_file():
        raise RuntimeError("FAIL_CLOSED: v2 has not been prepared; run prepare_v2.py first")
    sys.path.insert(0, str(code))
    import launch_plan as LP
    import launch_runner as R
    import launch_check as K
    import launch_guard as G
    import placement_teacher as T
    return root, code, L, LP, R, K, G, T


def assert_no_full(root: Path):
    if (root / "full_2d_v2").exists():
        raise RuntimeError("FAIL_CLOSED: full_2d_v2 exists; this tool must never touch or resume production")


def boost_state():
    p = Path("/sys/devices/system/cpu/cpufreq/boost")
    return p.read_text().strip() if p.exists() else None


def snapshot(root, L, *, require_runtime=False):
    import psutil
    import shutil
    cfg = djson(L / "configs" / "launch_config.json")
    disk_mb = shutil.disk_usage(L).free // (1024 * 1024)
    boost = boost_state()
    allowed = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else 0
    snap = {"free_disk_mb": disk_mb, "cpu_boost": boost, "allowed_logical_cpus": allowed,
            "min_disk_free_mb": cfg.get("v2_min_free_disk_mb", 10240),
            "mem_available_mb": psutil.virtual_memory().available // (1024 * 1024),
            "full_dir_exists": (root / "full_2d_v2").exists()}
    if require_runtime:
        if boost != "0": raise RuntimeError("STOP: CPU boost must be OFF for the 25-worker gate; change it from a host terminal with sudo")
        if disk_mb < cfg.get("v2_min_free_disk_mb", 10240): raise RuntimeError("STOP: less than 10 GiB free; no mock/production readiness")
        if allowed < 32: raise RuntimeError("STOP: 32 logical CPUs not available to the process")
    return snap


def init(repo):
    root, code, L, LP, R, K, G, T = setup(repo)
    assert_no_full(root)
    if (L / "query_plan.jsonl.gz").exists():
        raise RuntimeError("Plan already frozen; use static to verify, never silently re-freeze")
    subprocess.run([sys.executable, str(code / "launch_plan.py"), "freeze", str(L)], cwd=code, check=True)
    old = djson(root / "launch_ready_2d_v1" / "split_manifest.json")
    new = djson(L / "split_manifest.json")
    old_map = {r["root_id"]: r["split"] for r in old["roots"]}
    new_map = {r["root_id"]: r["split"] for r in new["roots"]}
    if old_map != new_map:
        raise RuntimeError("FAIL_CLOSED: splits drifted vs v1; investigate before creating production artifacts")
    cfg_p = L / "configs" / "launch_config.json"
    cfg = djson(cfg_p)
    cfg["pair_protocol_id"] = T.pair_protocol()[0]
    cfg_p.write_text(json.dumps(cfg, indent=1, sort_keys=True))
    fz = {"launch_config_sha256": file_hash(cfg_p),
          "pilot_policy_selection_sha256": file_hash(L / "configs" / "pilot_policy_selection.json"),
          "production_policy_sha256": None, "v2_source_adapter_sha256": file_hash(code / "placement_teacher.py"),
          "v1_splits_preserved": True}
    (L / "configs" / "frozen_hashes.json").write_text(json.dumps(fz, indent=1, sort_keys=True))
    result = {"metadata_frozen": True, "original_split_identity": True, "production_policy_frozen": False,
              "roots": len(new_map), "pair_protocol_id": cfg["pair_protocol_id"]}
    (L / "mock_init.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def static(repo, *, strict_runtime=False):
    root, code, L, LP, R, K, G, T = setup(repo)
    assert_no_full(root)
    if not (L / "query_plan.jsonl.gz").exists():
        raise RuntimeError("Run init first")
    b = LP.build()  # metadata only: original 5,000 selected roots, no teacher calls
    snap = snapshot(root, L, require_runtime=strict_runtime)
    old = djson(root / "launch_ready_2d_v1" / "split_manifest.json")
    new = djson(L / "split_manifest.json")
    om = {r["root_id"]: r["split"] for r in old["roots"]}
    nm = {r["root_id"]: r["split"] for r in new["roots"]}
    errors = []
    if om != nm: errors.append("SPLIT_DRIFT")
    if len(b["plan"]) != 5000: errors.append("ROOT_COUNT")
    if not b["pilot"] or any(b["split"][b["gof"][r]] != "train" for r in b["pilot"]): errors.append("PILOT_NOT_TRAIN")
    if djson(L / "configs" / "launch_config.json")["pair_protocol_id"] != T.pair_protocol()[0]: errors.append("PROTOCOL_DRIFT")
    plan_hashes = djson(L / "plan_hashes.json")
    if plan_hashes["query_plan_sha256"] != stable_hash(__import__("gzip").decompress((L / "query_plan.jsonl.gz").read_bytes())):
        errors.append("PLAN_DRIFT")
    # Target-goal independence and protected-field independence, including ALL three hypothetical blocks.
    examples = sorted(b["pilot"] + sorted(b["R"], key=lambda x: LP.H("audit", x))[:80])
    twoedit, nonempty_goal, blocked_meta = 0, 0, 0
    for rid in examples:
        r = b["R"][rid]
        ordinary = LP.root_plan(rid, r, T.P.A.env(), rid in b["pilot"])
        edit = copy.deepcopy(r)
        edit["goal_cell"] = next(iter(edit["scene"]["legal_cells"]))
        # Any other target-goal assignment must not change a placement-only query roster.
        if ordinary != LP.root_plan(rid, edit, T.P.A.env(), rid in b["pilot"]):
            nonempty_goal += 1
        edit["labels"] = {"B": [], "D": [], "original": {}, "blocker": {}}
        edit.update(roles={}, intended_blockers=[], parking={}, alt_parking={}, construction={})
        if ordinary != LP.root_plan(rid, edit, T.P.A.env(), rid in b["pilot"]):
            blocked_meta += 1
        twoedit += sum(row[1] == "prefix3" and len(row[2]) == 2 and row[3] == "DESIGN_TWO_EDIT" for row in ordinary)
    if nonempty_goal: errors.append(f"TARGET_GOAL_AFFECTS_QUERIES:{nonempty_goal}")
    if blocked_meta: errors.append(f"PROTECTED_METADATA_AFFECTS_QUERIES:{blocked_meta}")
    if twoedit == 0: errors.append("NO_TWO_EDIT_QUERIES")
    if (L / "configs" / "production_policy.json").exists() and not strict_runtime:
        errors.append("PRODUCTION_POLICY_ALREADY_EXISTS_WHILE_PREP")
    try:
        integ = K.integrity(L)
        for name, rec in integ.items():
            if name.endswith(".sha256") and (rec["changed"] or rec["added"]):
                errors.append("BASELINE_INTEGRITY_" + name)
        if not all(integ.get("smoke_code_unchanged", {}).values()):
            errors.append("SMOKE_CODE_CHANGED")
        if integ.get("git_status_original_dataset") != "clean":
            errors.append("ORIGINAL_DATASET_GIT_DIRTY")
    except Exception as e:
        errors.append("BASELINE_INTEGRITY_EXCEPTION:" + str(e))
    try:
        guard = G.preflight(L, full=False)  # no physical teacher planning
    except Exception as e:
        errors.append(f"METADATA_PREFLIGHT_EXCEPTION:{e}")
        guard = {}
    if guard and not guard["passed"]:
        errors += [f"PREFLIGHT_{k}" for k, ok in guard["checks"].items() if not ok]
    result = {"status": "STATIC_PASS" if not errors else "STATIC_FAIL", "errors": errors, "two_edit_queries_in_audit_subset": twoedit,
              "pilot_roots": len(b["pilot"]), "plan_root_count": len(b["plan"]), "snapshot": snap,
              "metadata_preflight_calls": guard.get("teacher_calls")}
    (L / "mock_static_report.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    if errors: raise RuntimeError("STATIC_FAIL; review report, do not start pilot")
    return result


def pilot_cmd(code, L, name, pairs, cov, workers, subset="all"):
    return [sys.executable, str(code / "launch_runner.py"), "pilot", str(L), name,
            str(pairs), cov, str(workers), subset]


def run_cmd(cmd, cwd):
    print("EXECUTING BOUNDED PILOT:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, check=True)


def mock(repo, execute):
    root, code, L, LP, R, K, G, T = setup(repo)
    assert_no_full(root)
    static(repo, strict_runtime=True)
    if not execute: raise RuntimeError("REFUSED: physically planning the pilot requires --execute-mock")
    # Regression tests only. This includes a single existing development fixture, never production roots.
    run_cmd([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(code / "test_v2_contract.py"),   # REVIEW FIX R4
             str(code / "test_launch_ready.py"), str(code / "test_placement_smoke.py")], code)
    # Readable evidence for a STRICT bounded pilot; never call full or guard --execute-full.
    sequence = [
        ("v2_ref64_25", 64, "FULL", 25, "all"),
        ("v2_adaptive25_a", 8, "ADAPTIVE_FULL", 25, "all"),
        ("v2_adaptive25_b", 8, "ADAPTIVE_FULL", 25, "all"),
        ("v2_adaptive_serial", 8, "ADAPTIVE_FULL", "serial", "first_per_root"),
    ]
    for name, B, cov, w, subset in sequence:
        target = L / "pilot" / name
        if target.exists():
            # A prior partial/cancelled run is resumed under the same pinned run manifest.
            print(f"Existing pilot {name}: runner will only resume if its manifest matches", flush=True)
        run_cmd(pilot_cmd(code, L, name, B, cov, w, subset), code)
        run_cmd([sys.executable, str(code / "launch_check.py"), "audit", str(target)], code)
        if any(djson(f)["stop"] is not None for f in target.glob("run_[0-9]*.json")):   # REVIEW FIX R3: not run_manifest.json
            raise RuntimeError(f"THERMAL_OR_SAFETY_STOP in {name}; no gate passing, do not relax thresholds")
        if (root / "full_2d_v2").exists():
            raise RuntimeError("Unexpected full production directory appeared")
    print("PILOT RUNS COMPLETED. Run interrupt and decide; this is NOT production readiness yet.")


def interrupt(repo, execute, min_roots=5):
    import psutil
    root, code, L, LP, R, K, G, T = setup(repo)
    assert_no_full(root)
    static(repo, strict_runtime=True)
    if not execute: raise RuntimeError("REFUSED: requires --execute-mock")
    name = "v2_adaptive_interrupted"
    target = L / "pilot" / name
    if target.exists(): raise RuntimeError("Interruption test already exists; don't overwrite it")
    cmd = pilot_cmd(code, L, name, 8, "ADAPTIVE_FULL", 25, "all")
    with open(L / "mock_interrupt_stdout.log", "w") as fh:
        proc = subprocess.Popen(cmd, cwd=code, stdout=fh, stderr=subprocess.STDOUT)
        children = []
        deadline = time.monotonic() + 1200  # safety: 20min limit for reaching committed roots
        try:
            while proc.poll() is None and time.monotonic() < deadline:
                done = list((target / "records").glob("*.json")) if (target / "records").exists() else []
                if len(done) >= min_roots:
                    children = [p.pid for p in psutil.Process(proc.pid).children(recursive=True)]
                    proc.kill()        # SIGKILL coordinator; workers must die by PdeathSIG
                    proc.wait(timeout=15)
                    break
                time.sleep(.5)
            else:
                if proc.poll() is None: proc.kill(); proc.wait(timeout=15)
                raise RuntimeError("FAIL_CLOSED: insufficient committed roots for the SIGKILL test")
        finally:
            if proc.poll() is None: proc.kill(); proc.wait(timeout=15)
    time.sleep(3)
    survivors = [pid for pid in children if psutil.pid_exists(pid) and psutil.Process(pid).is_running() and psutil.Process(pid).status() != psutil.STATUS_ZOMBIE]
    if survivors:
        raise RuntimeError(f"Workers survived coordinator SIGKILL: {survivors}. Investigate; no automatic kill of unrelated PIDs.")
    import glob
    before = {str(p.relative_to(target)): file_hash(p) for sub in ("records", "modes", "witnesses") for p in (target / sub).glob("*")}
    if not before: raise RuntimeError("No completed artifact files captured at SIGKILL")
    run_cmd(cmd, code)  # deterministic continuation
    run_cmd([sys.executable, str(code / "launch_check.py"), "audit", str(target)], code)
    mismatch = [x for x, h in before.items() if file_hash(target / x) != h]
    if mismatch: raise RuntimeError(f"Resume changed committed files: {mismatch[:10]}")
    if djson(target / "verification.json")["failed"] != 0: raise RuntimeError("Resume replay audit failed")
    print(json.dumps({"interrupt_resume": "PASS", "committed_unchanged": len(before), "surviving_workers": 0}, indent=1))


def composition_forecast(K, LP, L, run):
    """REVIEW FIX R5: Q1 CPU-s and bytes from the pilot's per-(block, dest_bin) means, weighted by Q1's query composition."""
    import gzip
    from collections import Counter, defaultdict
    import numpy as np
    recs = [(h, q, att) for h, q, m, att in K.load(run) if q["eligible"] and q["completed"]]
    files = {d: sum(f.stat().st_size for f in Path(run).glob(f"{d}/*")) for d in ("records", "modes", "witnesses")}
    nwit = sum(len(q["witness_refs"]) for _, q, _ in recs)
    per_w, base = files["witnesses"] / max(nwit, 1), (files["records"] + files["modes"]) / len(recs)
    cpu, byt, cpu_d, byt_d = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for h, q, att in recs:
        k, b = (q["block"], q["dest_bin"]), base + per_w * len(q["witness_refs"])
        cpu[k].append(q["volatile"]["cpu_s"]); byt[k].append(b); cpu_d[q["dest_bin"]].append(q["volatile"]["cpu_s"]); byt_d[q["dest_bin"]].append(b)
    plan = [json.loads(x) for x in gzip.decompress((L / "query_plan.jsonl.gz").read_bytes()).decode().splitlines()]
    comp = Counter((r[1], r[8]) for p in plan for r in p["queries"] if LP.in_quota(r, LP.QUOTAS["Q1"]) and not r[12])
    pick = lambda d, dd, k: float(np.mean(d[k])) if len(d[k]) >= 5 else float(np.mean(dd[k[1]]))  # noqa: E731
    strata = {f"{b}/{d}": {"q1_queries": n, "pilot_n": len(cpu[(b, d)]), "cpu_s": round(pick(cpu, cpu_d, (b, d)), 2)} for (b, d), n in sorted(comp.items())}
    return {"cpu_s": sum(n * pick(cpu, cpu_d, k) for k, n in comp.items()), "bytes": sum(n * pick(byt, byt_d, k) for k, n in comp.items()), "strata": strata}


def compare_adaptive(K, ref, target):
    a = {q["trial_id"]: (q, att) for h, q, m, att in K.load(ref) if q["eligible"] and q["completed"]}
    b = {q["trial_id"]: (q, att) for h, q, m, att in K.load(target) if q["eligible"] and q["completed"]}
    if set(a) != set(b): return {"shared": len(set(a) & set(b)), "bad": ["QUERY_SET_DIFF"]}
    bad = []
    for tid, (q, att) in a.items():
        z, aa = b[tid]
        v8 = K.virtual(q, att, 8, "FULL")
        expected = v8 if v8["label"] else K.virtual(q, att, 32, "FULL")
        if z["label"] != expected["label"] or int(z["work"]["pairs_planned"]) != expected["planned"]:
            bad.append(tid)
            continue
        mask = (att["a_rank"] >= 0) & (att["a_rank"] < expected["planned"])
        # No archived success is permitted in the canonical v2 pilot.
        if int((att["a_rank"] < 0).sum()) or int((aa["a_rank"] < 0).sum()):
            bad.append(tid + ":ARCHIVED_REUSE")
            continue
        if any(not __import__("numpy").array_equal(aa[k], att[k][mask]) for k in K.T.ATT if k != "a_stored"):
            bad.append(tid + ":ATTEMPTS")
    return {"shared": len(a), "bad": bad[:100], "different": len(bad)}


def decide(repo):
    root, code, L, LP, R, K, G, T = setup(repo)
    assert_no_full(root)
    static(repo, strict_runtime=True)
    pilots = [L / "pilot" / n for n in NAMES]
    if any(not (p / "verification.json").exists() for p in pilots): raise RuntimeError("Run mock first; missing pilot audit")
    if not (L / "pilot" / "v2_adaptive_interrupted" / "verification.json").exists(): raise RuntimeError("Run interrupt first")
    gate, failures = {}, []
    for p in pilots + [L / "pilot" / "v2_adaptive_interrupted"]:
        audit = djson(p / "verification.json")
        stops = [djson(f)["stop"] for f in p.glob("run_[0-9]*.json")]   # REVIEW FIX R3: run_manifest.json has no "stop"
        gate[p.name + "_audit"] = audit["failed"] == 0
        gate[p.name + "_safety"] = all(x is None for x in stops) if p.name != "v2_adaptive_interrupted" else True
        if not gate[p.name + "_audit"]: failures.append(p.name + ":AUDIT")
        if not gate[p.name + "_safety"]: failures.append(p.name + ":SAFETY")
    # Byte-level scientific fingerprints across the two independent 25-worker adaptive runs.
    a, b = pilots[1:3]
    af, bf = djson(a / "fingerprint.json")["queries"], djson(b / "fingerprint.json")["queries"]
    gate["independent_25_worker_repro"] = af == bf
    if not gate["independent_25_worker_repro"]: failures.append("ADAPTIVE_25_WORKER_FINGERPRINTS_DIFFER")
    serial = djson(pilots[3] / "fingerprint.json")["queries"]
    gate["serial_matches_parallel"] = all(t in af and af[t] == x for t, x in serial.items())
    if not gate["serial_matches_parallel"]: failures.append("SERIAL_PARALLEL_DIFFER")
    iv = djson(L / "pilot" / "v2_adaptive_interrupted" / "fingerprint.json")["queries"]
    gate["interrupt_matches"] = iv == af
    if not gate["interrupt_matches"]: failures.append("INTERRUPT_RESUME_DIFFER")
    nested = compare_adaptive(K, pilots[0], a)
    gate["adaptive_nested_ref64"] = not nested.get("bad")
    if not gate["adaptive_nested_ref64"]: failures.append("ADAPTIVE_REF64_MISMATCH")
    # Predeclared finite-policy quality test against the broader (still non-exhaustive) 64 reference.
    refq = {q["trial_id"]: q for h, q, m, att in K.load(pilots[0]) if q.get("eligible") and q.get("completed")}
    adq = {q["trial_id"]: q for h, q, m, att in K.load(a) if q.get("eligible") and q.get("completed")}
    pos64 = [tid for tid, q in refq.items() if q["label"] == 1]
    missed = [tid for tid in pos64 if adq.get(tid, {}).get("label") != 1]
    def strata_stats(field):
        keys = {q[field] for q in refq.values()}
        out = {}
        for k in keys:
            ids = [tid for tid in pos64 if refq[tid][field] == k]
            nbad = sum(tid in set(missed) for tid in ids)
            out[str(k)] = {"pos64": len(ids), "missed": nbad, "miss_frac": nbad / len(ids) if ids else None}
        return out
    miss_frac = len(missed) / len(pos64) if pos64 else 1.0
    strats = {"block": strata_stats("block"), "dest_bin": strata_stats("dest_bin")}
    acceptable = miss_frac <= 0.02 and all(
        v["pos64"] < 20 or v["miss_frac"] <= 0.05
        for dim in strats.values() for v in dim.values())
    gate["adaptive_recall_vs_64"] = acceptable
    if not acceptable: failures.append("ADAPTIVE_MISSED_TOO_MANY_64_REFERENCE_POSITIVES")
    # Quality: no untested mode silently becomes negative certificate, and confidence labels remain policy-relative.
    total = djson(a / "summary.json")
    neg_rows = [(h, q) for h, q, m, att in K.load(a) if q.get("completed") and q.get("eligible") and q.get("label") == 0]
    neg_groups = {h["root_group_id"] for h, q in neg_rows}
    neg_classes = {h["privileged_teacher"]["verified_class"] for h, q in neg_rows}
    gate["negative_diversity"] = len(neg_rows) >= 10 and len(neg_groups) >= 5 and len(neg_classes) >= 3
    if not gate["negative_diversity"]: failures.append("NEGATIVE_DIVERSITY")
    missing_witnesses = []
    for h, q, mode, att in K.load(a):
        if not (q.get("eligible") and q.get("completed")):
            continue
        verified_ids = {str(w) for v, w in zip(att["a_verified"], att["a_witness"]) if bool(v)}
        recorded_ids = set(map(str, q["witness_refs"]))
        if verified_ids != recorded_ids:
            missing_witnesses.append(q["trial_id"])
    # Counts may differ because identical verified mode paths are content deduplicated.
    gate["all_verified_witnesses_stored"] = not missing_witnesses
    if not gate["all_verified_witnesses_stored"]: failures.append("WITNESS_LOSS:" + str(missing_witnesses[:8]))
    pilot_q = total["completed"]
    q1 = djson(L / "plan_summary.json")["quota_teacher_queries"]["Q1"]
    # Deliberately conservative estimate: pilot mean x 1.5 heterogeneity buffer.
    cpu_mean = float(total["query_cpu_s_sum"]) / pilot_q * q1 * 1.5
    disk_mean = sum(total["bytes"].values()) / pilot_q * q1 * 1.5
    # REVIEW FIX R5: the pilot over-represents cheap near / vacated destinations (first destinations of each block), while Q1
    # is mostly mid / far root destinations. Re-weight the pilot by Q1's (block, dest_bin) composition (pooled dest_bin when a
    # stratum has < 5 pilot queries) and keep the LARGER of the two estimates; the 1.5x buffer applies to both.
    comp = composition_forecast(K, LP, L, a)
    cpu_sec = max(cpu_mean, comp["cpu_s"] * 1.5)
    disk_bytes = max(disk_mean, comp["bytes"] * 1.5)
    # Monitor logs, in-flight parts and atomic writes need additional space.
    disk_mb = disk_bytes / 1024**2 + 500
    speed = 24.83  # historical rate only; v2 run-time forecast updated below
    mon = [json.loads(line) for line in (a / "monitor.jsonl").read_text().splitlines() if line.strip()]
    busy = [x for x in mon if x.get("queued", 0) > 0 and x.get("workers_cpu_s") is not None]
    if len(busy) >= 2 and busy[-1]["t"] > busy[0]["t"]:
        speed = (busy[-1]["workers_cpu_s"]-busy[0]["workers_cpu_s"]) / (busy[-1]["t"]-busy[0]["t"])
    days = cpu_sec / max(.1, speed) / 86400
    import shutil
    free_mb = shutil.disk_usage(L).free / 1024**2
    gate["disk_safety"] = free_mb >= max(10240, disk_mb + 1536)
    gate["runtime_forecast"] = days <= 7.0
    for k in ("disk_safety", "runtime_forecast"):
        if not gate[k]: failures.append(k.upper())
    report = {"verdict": "READY_TO_LAUNCH" if not failures else "NOT_READY_TO_LAUNCH", "gates": gate, "failures": failures,
              "nested": nested, "relative_to_reference_64": {"positive_reference_count": len(pos64),
                                        "adaptive_missed": len(missed), "miss_fraction": miss_frac,
                                        "wilson_95": K.wilson(len(missed), len(pos64)), "strata": strats}, "pilot": {"completed": pilot_q, "positives": total["label_1"], "negatives": total["label_0"],
                                    "stored_witnesses": total["stored_witnesses"]},
              "forecast": {"q1_teacher_queries": q1, "cpu_hours_with_1p5_buffer": round(cpu_sec / 3600,1),
                           "wall_days_with_1p5_buffer": round(days,2), "disk_mb_with_1p5_buffer_and_500mb_overhead": round(disk_mb,1),
                           "free_mb": round(free_mb,1), "pilot_throughput_cpu_s_per_s": round(speed,2),
                           "estimators": {"pilot_mean_cpu_hours_x1p5": round(cpu_mean / 3600, 1), "composition_weighted_cpu_hours_x1p5": round(comp["cpu_s"] * 1.5 / 3600, 1),
                                          "composition_weighted_wall_days_no_buffer": round(comp["cpu_s"] / max(.1, speed) / 86400, 2),
                                          "pilot_mean_disk_mb_x1p5": round(disk_mean / 1024**2, 1), "composition_weighted_disk_mb_x1p5": round(comp["bytes"] * 1.5 / 1024**2, 1),
                                          "strata": comp["strata"], "rule": "max(pilot mean, Q1-composition weighted) x 1.5 (review fix R5)"}},
              "explicit_scope": "No production query executed; readiness of the configured finite-policy 2D placement sidecar only"}
    (L / "MOCK_V2_DECISION.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    if failures:
        print(json.dumps(report, indent=1))
        raise RuntimeError("NOT_READY_TO_LAUNCH: all gates required; do not modify thresholds post hoc")
    # Freeze the COMPLETE production policy. This does NOT start the full job.
    pol = {"pairs_per_query": 8, "rescue_pairs": 32, "coverage": "ADAPTIVE_FULL", "witness_store_max": 32, "reuse_archived": False}
    P = T.P
    full = {"version": "PRODUCTION_POLICY_2D_V2", "policy": pol, "quota": "Q1", "quota_spec": LP.QUOTAS["Q1"],
            "budget_hash": T.budget_hash(pol), "forecast": {"teacher_queries": q1, "disk_mb": disk_mb, "wall_days": days},
            "decision_sha256": file_hash(L / "MOCK_V2_DECISION.json")}
    pf = L / "configs" / "production_policy.json"
    if pf.exists(): raise RuntimeError("FAIL_CLOSED: production policy already exists")
    pf.write_text(json.dumps(full, indent=1, sort_keys=True))
    fz = djson(L / "configs" / "frozen_hashes.json")
    fz["production_policy_sha256"] = file_hash(pf)
    (L / "configs" / "frozen_hashes.json").write_text(json.dumps(fz, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1))
    print("FROZEN v2 production policy; full job NOT started. Explicit later authorization required.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("repo", type=Path, help="The local rss-ebm-infeasibility checkout, e.g. /home/naren/Desktop/rss-placement-smoke")
    ap.add_argument("command", choices=("init", "static", "mock", "interrupt", "decide"))
    ap.add_argument("--execute-mock", action="store_true")
    args = ap.parse_args()
    if args.command == "init": init(args.repo)
    elif args.command == "static": static(args.repo)
    elif args.command == "mock": mock(args.repo, args.execute_mock)
    elif args.command == "interrupt": interrupt(args.repo, args.execute_mock)
    elif args.command == "decide": decide(args.repo)
    else: raise RuntimeError("Unknown command")

if __name__ == "__main__":
    try: main()
    except (RuntimeError, subprocess.CalledProcessError) as e:
        print(f"FAIL_CLOSED: {e}", file=sys.stderr)
        sys.exit(2)
