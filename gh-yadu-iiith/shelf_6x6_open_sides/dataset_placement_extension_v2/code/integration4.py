"""Bounded physical integration of the v4 production path on the NEW Q2_3D_PARITY_V1 roster (32 TRAIN queries, 4 roots, every block,
near / mid / far / vacated, both shapes), run as INTEGRATION_TEST through collect4 / prod4 (the v3 collector unchanged).  Same flow as
integration3d: smoke A; smoke B with a coordinator-only SIGKILL after its first unit with witnesses, one truncated NPZ and a torn ledger
line, then resume; B == A; fresh-process replay of every stored witness; post-run audits; the zero-call preflight of the v4 template.
  python integration4.py OUT_DIR SHELF_SRC          (OUT_DIR = production_v4/integration)
"""
import gzip, json, os, signal, subprocess, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
sys.path.insert(0, str(HERE))
import prod4  # noqa: E402,F401
import prod3d  # noqa: E402
from integration3d import KEYS, ledger, peaks, records, sh  # noqa: E402


def main(out, src):
    import psutil
    import collect3d as C
    out, V3 = Path(out).resolve(), ROOT / "production_v3"
    env = {k: v for k, v in os.environ.items() if k not in prod3d.OVERRIDES} | {"SHELF_SRC": str(Path(src).resolve())}
    res, pols = {"started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, {}
    for k in ("A", "B"):
        pols[k] = out / f"test_policy_smoke_{k}.json"
        rc, so, se = sh([PY, "code/prod4.py", "template", "--mode", "INTEGRATION_TEST", "--policy", "adaptive_v1", "--workers", 4, "--cpus", "0-3",
                         "--run-dir", f"production_v4/integration/smoke_{k}", "--roster", out / "smoke_roster_v4.jsonl.gz",
                         "--source-pin", V3 / "frozen/source_pin_first_cohort.json", "--split", V3 / "frozen/split_manifest_first_cohort.json",
                         "--qc", V3 / "QC_ELIGIBLE_ROOTS.json", "--cohort", ROOT / "production_v4/frozen/roster_manifest_v4.json", "--shelf-src", src,
                         "--forecast", ROOT / "production_v4/frozen/forecast_v4.json", "--out", pols[k]], env)
        assert rc == 0, se
    tok = {k: sh([PY, "code/prod4.py", "token", pols[k]], env)[1].strip() for k in pols}
    runA, runB = out / "smoke_A", out / "smoke_B"
    assert not runA.exists() and not runB.exists(), "smoke run directories must be fresh"
    t0 = time.time()
    rc, so, se = sh([PY, "code/collect4.py", "--policy-file", pols["A"], "--confirm", tok["A"]], env)
    res["smoke_A"] = {"rc": rc, "end": so.strip().splitlines()[-1] if so.strip() else se[-500:], "wall_s": round(time.time() - t0, 1)}
    assert rc == 0, se[-2000:]
    nunits = len(json.loads((runA / "run_manifest.json").read_text())["units"])
    p = subprocess.Popen([PY, "code/collect4.py", "--policy-file", str(pols["B"]), "--confirm", tok["B"]], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    done, kids, t1 = [], [], time.time()
    while time.time() - t1 < 3600:
        time.sleep(0.2)
        if p.poll() is not None:
            break
        if (runB / "ledger.jsonl").exists():
            done = [e for e in ledger(runB) if e.get("event") == "unit_done"]
            kids = [c.pid for c in psutil.Process(p.pid).children(recursive=True)]
            if any(e.get("npz_sha256") for e in done):
                break
    assert p.poll() is None and len(done) < nunits, f"smoke B ended before it could be interrupted: {p.poll()}"
    os.kill(p.pid, signal.SIGKILL); p.wait(); time.sleep(3.0)
    alive = [k for k in kids if psutil.pid_exists(k) and psutil.Process(k).status() != psutil.STATUS_ZOMBIE]
    lock_after = prod3d.probe(prod3d.lock_path("INTEGRATION_TEST"))
    victim = next(e["unit"] for e in done if e.get("npz_sha256"))
    z = runB / "units" / f"{victim}.npz"
    size = z.stat().st_size
    with open(z, "r+b") as f:
        f.truncate(size // 2)
    with open(runB / "ledger.jsonl", "a") as f:
        f.write('{"event": "unit_done", "unit": "torn-by-integration-test"')
    res["smoke_B_interrupt"] = {"units_total": nunits, "units_done_before_kill": len(done), "descendants_before_kill": len(kids),
                                "descendants_alive_3s_after_kill": alive, "lock_held_after_kill": lock_after, "damaged_npz_unit": victim,
                                "npz_bytes_before_after": [size, size // 2], "torn_ledger_line_appended": True}
    rc, so, se = sh([PY, "code/collect4.py", "--policy-file", pols["B"], "--confirm", tok["B"]], env)
    starts = [e for e in ledger(runB) if e.get("event") == "start"]
    assert len(starts) == 2, len(starts)
    ra = starts[-1]["resume_audit"]
    res["smoke_B_resume"] = {"rc": rc, "resumed_units": starts[-1]["resumed"], "recomputed_units": starts[-1]["todo"], "quarantined_files": ra["quarantined_files"],
                             "ledger_units_failed": ra["ledger_units_failed"],
                             "damaged_unit_quarantined_and_recomputed": f"{victim}.npz" in ra["quarantined_files"] and victim in ra["ledger_units_failed"],
                             "torn_line_terminated_and_ignored": any(e.get("event") == "torn_line" for e in ledger(runB))}
    assert rc == 0, se[-2000:]
    A, B = records(runA), records(runB)
    diff = [q for q in A if [A[q].get(k) for k in KEYS] + [[w["witness_id"] for w in A[q]["witnesses"]]] !=
            [B.get(q, {}).get(k) for k in KEYS] + [[w["witness_id"] for w in B.get(q, {}).get("witnesses", [])]]]
    res["B_equals_A"] = {"queries": len(A), "mismatches": diff, "equal": not diff and set(A) == set(B)}
    rost = {json.loads(x)["qid"]: json.loads(x) for x in gzip.open(out / "smoke_roster_v4.jsonl.gz", "rt")}
    for k, run in (("A", runA), ("B", runB)):
        rc, so, se = sh([PY, "code/replay3d.py", "ours", run, out / "smoke_roster_v4.jsonl.gz", out / f"replay_smoke_{k}.json", "--workers", 4, "--cpus", "0-3"], env)
        res[f"replay_{k}"] = json.loads((out / f"replay_smoke_{k}.json").read_text()) if rc == 0 else {"rc": rc, "err": se[-800:]}
        res[f"replay_{k}"].pop("results", None)
        rc, so, se = sh([PY, "code/collect4.py", "--policy-file", pols[k], "--audit"], env)
        res[f"audit_{k}"] = json.loads(so) if rc == 0 else {"rc": rc, "err": se[-800:]}
        res[f"monitor_{k}"] = peaks(run)
        res[f"marked_{k}"] = {"marker": (run / "TEST_RUN_NOT_PRODUCTION.txt").read_text().strip(),
                              "manifest_mode": json.loads((run / "run_manifest.json").read_text())["mode"]}
    res["labels_by_block_A"] = dict(sorted(Counter(f"{rost[q]['block']}:{r['label']}:{r.get('negative_kind') or r['status']}" for q, r in A.items()).items()))
    res["labels_by_band_A"] = dict(sorted(Counter(f"{rost[q]['band']}:{r['label']}" for q, r in A.items()).items()))
    res["semantics_problems_A"] = sum(len(C.semantics(r)) for r in A.values())
    res["cpu_s_per_query_A"] = round(sum(r["cpu_query_s"] for r in A.values()) / len(A), 2)
    rc, so, se = sh([PY, "code/collect4.py", "--policy-file", ROOT / "production_v4/production_policy_v4.TEMPLATE.json", "--dry-run"], env)
    pre = json.loads(so.strip().splitlines()[-1]) if so.strip() else {"err": se[-1500:]}
    pre["rc"] = rc
    (out / "production_metadata_preflight_v4.json").write_text(json.dumps(pre, indent=1))
    res["production_preflight"] = {k: pre.get(k) for k in ("rc", "physical_queries", "static_contrasts", "units", "by_split", "by_block", "planner_calls",
                                                          "teacher_entry_calls", "run_dir_created", "refusals_other_than_authorization",
                                                          "authorization_refusals", "roster_regenerated_equal", "semantic_keys_unique")}
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (out / "integration_results_v4.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str))
    print(json.dumps(res, indent=1, default=str)[:7000])


if __name__ == "__main__":
    main(*sys.argv[1:3])
