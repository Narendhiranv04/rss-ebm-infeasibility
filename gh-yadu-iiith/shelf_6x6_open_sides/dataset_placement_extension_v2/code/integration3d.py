"""Bounded, physically executed end-to-end integration smoke of the production v3 path: 12 TRAIN queries on 4 QC-eligible roots, run as
INTEGRATION_TEST (separate roster, test fixture, run dirs under production_v3/integration/, marked everywhere; the user's authorization
fields stay empty).  Same collector, guard, lock, pinned workers, atomic storage, monitor, resume and replay code as production.

  smoke A   uninterrupted run
  smoke B   coordinator SIGKILLed after its first unit with stored witnesses (workers must die with it, the lock must be released); one
            finished NPZ truncated + a torn ledger line appended; the identical command resumes; B must equal A query by query
  then      fresh-process MuJoCo replay of EVERY stored witness association of A and B, the collector's post-run audit of both, and the
            metadata-only preflight of the production template (0 teacher calls)

  python integration3d.py OUT_DIR SHELF_SRC          (OUT_DIR = production_v3/integration)
"""
import json, os, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
sys.path.insert(0, str(HERE))
import prod3d  # noqa: E402

KEYS = ("label", "status", "negative_kind", "verified_rank", "verified_branch", "witness_id", "certified_by")


def sh(args, env, timeout=7200):
    r = subprocess.run([str(a) for a in args], cwd=ROOT, env=env, capture_output=True, text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def ledger(run):
    out = []
    for x in (Path(run) / "ledger.jsonl").read_text().splitlines():
        try:
            out.append(json.loads(x))
        except ValueError:
            out.append({"event": "torn_line", "raw": x[:60]})
    return out


def records(run):
    out = {}
    for f in sorted((Path(run) / "units").glob("*.json")):
        for r in json.loads(f.read_text())["records"]:
            out[r["qid"]] = r
    return out


def peaks(run):
    mon = [json.loads(x) for x in (Path(run) / "monitor.jsonl").read_text().splitlines() if x.strip()]
    return {"ticks": len(mon), "max_cpu_pct_sum": max(m["cpu_pct_sum"] for m in mon), "max_temp_c": max(m["pkg_temp_c"] or 0 for m in mon),
            "affinity_union": sorted({c for m in mon for c in m["affinity_union"]}), "violations": sorted({v for m in mon for v in m["violations"]}),
            "min_mem_avail_gb": round(min(m["mem_avail_gb"] for m in mon), 1), "unexpected_descendants": sum(len(m["unexpected_descendants"]) for m in mon),
            "sensor": mon[-1]["sensor"]}


def main(out, src):
    out = Path(out).resolve()
    env = {k: v for k, v in os.environ.items() if k not in prod3d.OVERRIDES} | {"SHELF_SRC": str(Path(src).resolve())}
    res, fz = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, ROOT / "production_v3/frozen"
    pols = {}
    for k in ("A", "B"):
        pols[k] = out / f"test_policy_smoke_{k}.json"
        rc, so, se = sh([PY, "code/prod3d.py", "template", "--mode", "INTEGRATION_TEST", "--policy", "adaptive_v1", "--workers", 4, "--cpus", "0-3",
                         "--run-dir", f"production_v3/integration/smoke_{k}", "--roster", out / "smoke_roster.jsonl.gz",
                         "--source-pin", fz / "source_pin_first_cohort.json", "--split", fz / "split_manifest_first_cohort.json",
                         "--qc", ROOT / "production_v3/QC_ELIGIBLE_ROOTS.json", "--cohort", fz / "cohort_manifest.json", "--shelf-src", src,
                         "--forecast", fz / "forecast_first_cohort.json", "--out", pols[k]], env)
        assert rc == 0, se
    tok = {k: sh([PY, "code/prod3d.py", "token", pols[k]], env)[1].strip() for k in pols}
    runA, runB = out / "smoke_A", out / "smoke_B"
    assert not runA.exists() and not runB.exists(), "smoke run directories must be fresh"
    t0 = time.time()
    rc, so, se = sh([PY, "code/collect3d.py", "--policy-file", pols["A"], "--confirm", tok["A"]], env)
    res["smoke_A"] = {"rc": rc, "end": so.strip().splitlines()[-1] if so.strip() else se[-500:], "wall_s": round(time.time() - t0, 1)}
    assert rc == 0, se[-2000:]
    # ---- smoke B: kill the coordinator only, mid-run ----
    import psutil
    nunits = len(json.loads((runA / "run_manifest.json").read_text())["units"])
    p = subprocess.Popen([PY, "code/collect3d.py", "--policy-file", str(pols["B"]), "--confirm", tok["B"]], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    done, kids, t1 = [], [], time.time()
    while time.time() - t1 < 3600:
        time.sleep(0.2)
        if p.poll() is not None:
            break
        if (runB / "ledger.jsonl").exists():
            done = [e for e in ledger(runB) if e.get("event") == "unit_done"]
            kids = [c.pid for c in psutil.Process(p.pid).children(recursive=True)] if p.poll() is None else kids
            if any(e.get("npz_sha256") for e in done) or p.poll() is not None:
                break
    assert p.poll() is None and len(done) < nunits, f"smoke B ended before it could be interrupted: rc={p.poll()} {p.stderr.read()[-1500:] if p.poll() is not None else ''}"
    os.kill(p.pid, signal.SIGKILL); p.wait()
    time.sleep(3.0)
    alive = [k for k in kids if psutil.pid_exists(k) and psutil.Process(k).status() != psutil.STATUS_ZOMBIE]
    lock_after = prod3d.probe(prod3d.lock_path("INTEGRATION_TEST"))
    victim = next(e["unit"] for e in done if e.get("npz_sha256"))
    z = runB / "units" / f"{victim}.npz"
    size = z.stat().st_size
    with open(z, "r+b") as f:
        f.truncate(size // 2)
    with open(runB / "ledger.jsonl", "a") as f:
        f.write('{"event": "unit_done", "unit": "torn-by-integration-test"')
    res["smoke_B_interrupt"] = {"units_total": nunits, "units_done_before_kill": len(done), "killed": "coordinator only (SIGKILL)",
                                "descendants_before_kill": len(kids), "descendants_alive_3s_after_kill": alive, "lock_held_after_kill": lock_after,
                                "damaged_npz_unit": victim, "npz_bytes_before_after": [size, size // 2], "torn_ledger_line_appended": True}
    t2 = time.time()
    rc, so, se = sh([PY, "code/collect3d.py", "--policy-file", pols["B"], "--confirm", tok["B"]], env)
    starts = [e for e in ledger(runB) if e.get("event") == "start"]
    assert len(starts) == 2, f"expected the killed run's and the resumed run's start records, got {len(starts)}"
    ra = starts[-1]["resume_audit"]
    res["smoke_B_resume"] = {"rc": rc, "wall_s": round(time.time() - t2, 1), "resumed_units": starts[-1]["resumed"], "recomputed_units": starts[-1]["todo"],
                             "quarantined_files": ra["quarantined_files"], "ledger_units_failed": ra["ledger_units_failed"],
                             "damaged_unit_quarantined_and_recomputed": f"{victim}.npz" in ra["quarantined_files"] and victim in ra["ledger_units_failed"],
                             "torn_line_terminated_and_ignored": any(e.get("event") == "torn_line" for e in ledger(runB)), "start_records": len(starts), "end": so.strip().splitlines()[-1] if so.strip() else se[-500:]}
    assert rc == 0, se[-2000:]
    A, B = records(runA), records(runB)
    diff = [q for q in A if [A[q].get(k) for k in KEYS] + [[w["witness_id"] for w in A[q]["witnesses"]]] !=
            [B.get(q, {}).get(k) for k in KEYS] + [[w["witness_id"] for w in B.get(q, {}).get("witnesses", [])]]]
    res["B_equals_A"] = {"queries": len(A), "mismatches": diff, "equal": not diff and set(A) == set(B)}
    for k, run in (("A", runA), ("B", runB)):
        rc, so, se = sh([PY, "code/replay3d.py", "ours", run, out / "smoke_roster.jsonl.gz", out / f"replay_smoke_{k}.json", "--workers", 4, "--cpus", "0-3"], env)
        res[f"replay_{k}"] = json.loads((out / f"replay_smoke_{k}.json").read_text()) if rc == 0 else {"rc": rc, "err": se[-800:]}
        res[f"replay_{k}"].pop("results", None)
        rc, so, se = sh([PY, "code/collect3d.py", "--policy-file", pols[k], "--audit"], env)
        res[f"audit_{k}"] = json.loads(so) if rc == 0 else {"rc": rc, "err": se[-800:]}
        res[f"monitor_{k}"] = peaks(run)
        res[f"marked_{k}"] = {"marker_file": (run / "TEST_RUN_NOT_PRODUCTION.txt").exists(),
                              "manifest_mode": json.loads((run / "run_manifest.json").read_text())["mode"],
                              "ledger_modes": sorted({e.get("mode") for e in ledger(run) if e.get("event") in ("start", "end")} - {None})}
    import collect3d as C
    prov = {}
    for q, r in A.items():
        prov[q] = {"label": r["label"], "status": r["status"], "negative_kind": r.get("negative_kind"), "witnesses": len(r["witnesses"]),
                   "semantics_problems": C.semantics(r), "certificate_scope": (r.get("certificate") or {}).get("scope"),
                   "policy_scope": r.get("scope"), "screen_counts": r.get("screen_counts"), "accounting_unaccounted": r["accounting"].get("branch_unaccounted")}
    res["provenance_A"] = prov
    rc, so, se = sh([PY, "code/collect3d.py", "--policy-file", ROOT / "production_v3/production_policy.TEMPLATE.json", "--dry-run"], env)
    pre = json.loads(so.strip().splitlines()[-1]) if so.strip() else {"err": se[-1500:]}
    pre["rc"] = rc
    (out / "production_metadata_preflight.json").write_text(json.dumps(pre, indent=1))
    res["production_preflight"] = {k: pre.get(k) for k in ("rc", "physical_queries", "static_contrasts", "units", "roots", "by_split", "planner_calls",
                                                          "teacher_entry_calls", "run_dir_created", "refusals_other_than_authorization",
                                                          "authorization_refusals", "roster_regenerated_equal", "lock_free")}
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "integration_results.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    print(json.dumps({k: v for k, v in res.items() if k != "provenance_A"}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main(*sys.argv[1:3])
