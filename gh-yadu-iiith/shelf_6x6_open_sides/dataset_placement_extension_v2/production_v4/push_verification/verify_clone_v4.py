"""Independent fresh-checkout verification of the pushed production v4 commit (run inside the clone's sidecar directory).
  SHELF_SRC=<clone native dataset dir> python production_v4/push_verification/verify_clone_v4.py EXPECTED_SHA OUT.json
Checks: HEAD; the v4, v3 and V2 checksum sets; the frozen roster / contexts / duplicate map / manifest / coverage report rebuilt byte for
byte from the clone; the v4 template regenerated in the clone (frozen block equal except SHELF_SRC); the full-roster preflight (0 teacher
calls); the v4 test suite; replay of every smoke-A witness.
"""
import filecmp, json, os, subprocess, sys, tempfile
from pathlib import Path

ROOT, PY, SRC = Path.cwd(), sys.executable, os.environ["SHELF_SRC"]


def sh(*a, **kw):
    r = subprocess.run([str(x) for x in a], cwd=ROOT, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout, r.stderr


def main(expect, out):
    res = {"head": sh("git", "rev-parse", "HEAD")[1].strip()}
    res["head_matches"] = res["head"] == expect
    res["v4_sha256sums_ok"] = subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=ROOT / "production_v4").returncode == 0
    res["v3_sha256sums_ok"] = subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=ROOT / "production_v3").returncode == 0
    res["v2_manifest_ok"] = subprocess.run(["sha256sum", "-c", "--quiet", "ARTIFACT_MANIFEST.sha256"], cwd=ROOT).returncode == 0
    t = Path(tempfile.mkdtemp())
    rc, so, se = sh(PY, "code/q2roster3d.py", "freeze", SRC, "production_v3", t / "v4")
    names = ["frozen/q2_parity_roster_v4.jsonl.gz", "frozen/q2_contexts_v4.jsonl.gz", "frozen/semantic_duplicates_v4.json", "frozen/roster_manifest_v4.json",
             "COVERAGE_REPORT_V4.json"]
    res["roster_rebuilt_identical"] = {"rc": rc, **{n: rc == 0 and filecmp.cmp(t / "v4" / n, ROOT / "production_v4" / n, shallow=False) for n in names}}
    rc, so, se = sh(PY, "code/prod4.py", "template", "--mode", "PRODUCTION_FIRST_COHORT", "--policy", "adaptive_v1", "--workers", 12, "--cpus", "0-11",
                    "--run-dir", "production_v4/run_q2_parity", "--roster", "production_v4/frozen/q2_parity_roster_v4.jsonl.gz",
                    "--source-pin", "production_v3/frozen/source_pin_first_cohort.json", "--split", "production_v3/frozen/split_manifest_first_cohort.json",
                    "--qc", "production_v3/QC_ELIGIBLE_ROOTS.json", "--cohort", "production_v4/frozen/roster_manifest_v4.json", "--shelf-src", SRC,
                    "--forecast", "production_v4/frozen/forecast_v4.json", "--out", t / "regen.json")
    a, b = (json.loads(p.read_text())["frozen"] for p in (ROOT / "production_v4/production_policy_v4.TEMPLATE.json", t / "regen.json"))
    a["source"].pop("SHELF_SRC"); b["source"].pop("SHELF_SRC")
    res["template_frozen_equal_except_SHELF_SRC"] = rc == 0 and a == b
    rc, so, se = sh(PY, "code/collect4.py", "--policy-file", t / "regen.json", "--dry-run")
    d = json.loads(so.strip().splitlines()[-1]) if so.strip() else {}
    res["preflight_from_clone"] = {"rc": rc, **{k: d.get(k) for k in ("physical_queries", "planner_calls", "teacher_entry_calls", "run_dir_created",
                                                                      "refusals_other_than_authorization", "roster_regenerated_equal", "semantic_keys_unique")}}
    rc, so, se = sh("taskset", "-c", "12-13", PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "code/test_prod4.py", timeout=3600)
    res["pytest_v4_from_clone"] = so.strip().splitlines()[-1] if so.strip() else se[-300:]
    rc, so, se = sh("taskset", "-c", "0-3", PY, "code/replay3d.py", "ours", "production_v4/integration/smoke_A",
                    "production_v4/integration/smoke_roster_v4.jsonl.gz", t / "replay_A.json", "--workers", 4, "--cpus", "0-3")
    r = json.loads((t / "replay_A.json").read_text()) if rc == 0 else {}
    res["smoke_A_replay_from_clone"] = {k: r.get(k) for k in ("replayed", "verified", "failures", "max_q_exec_diff")}
    p = res["preflight_from_clone"]
    res["pass"] = bool(res["head_matches"] and res["v4_sha256sums_ok"] and res["v3_sha256sums_ok"] and res["v2_manifest_ok"]
                       and all(v for k, v in res["roster_rebuilt_identical"].items() if k != "rc") and res["template_frozen_equal_except_SHELF_SRC"]
                       and p["rc"] == 0 and p["planner_calls"] == 0 and p["teacher_entry_calls"] == 0 and p["refusals_other_than_authorization"] == []
                       and p["roster_regenerated_equal"] is True and p["physical_queries"] == 59972
                       and " passed" in res["pytest_v4_from_clone"] and " failed" not in res["pytest_v4_from_clone"]
                       and r.get("replayed") == r.get("verified") == 21 and not r.get("failures"))
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
