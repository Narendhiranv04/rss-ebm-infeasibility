"""Independent fresh-checkout verification of the pushed production v3 commit (run inside the clone's sidecar directory).
  SHELF_SRC=<clone native dataset dir> python production_v3/push_verification/verify_clone_v3.py EXPECTED_SHA OUT.json
Checks: clone HEAD; production_v3/SHA256SUMS; the V2 manifest; the published source snapshot (20,577 files) and its selection;
QC verdicts rebuilt from the committed per-process results; the frozen cohort files rebuilt byte for byte; the template regenerated
in the clone (frozen block equal except SHELF_SRC); the metadata preflight (0 teacher calls); the v3 test suite; smoke witness replay.
"""
import filecmp, gzip, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path.cwd()
PY = sys.executable
SRC = os.environ["SHELF_SRC"]
sys.path.insert(0, str(ROOT / "code"))


def sh(*a, **kw):
    r = subprocess.run([str(x) for x in a], cwd=ROOT, capture_output=True, text=True, **kw)
    return r.returncode, r.stdout, r.stderr


def main(expect, out):
    res = {}
    res["head"] = sh("git", "rev-parse", "HEAD")[1].strip()
    res["head_matches"] = res["head"] == expect
    res["v3_sha256sums_ok"] = subprocess.run(["sha256sum", "-c", "--quiet", "SHA256SUMS"], cwd=ROOT / "production_v3").returncode == 0
    res["v3_files_listed"] = sum(1 for _ in open(ROOT / "production_v3/SHA256SUMS"))
    res["v2_manifest_ok"] = subprocess.run(["sha256sum", "-c", "--quiet", "ARTIFACT_MANIFEST.sha256"], cwd=ROOT).returncode == 0
    import prod3d, source_sync3d as SS
    sm = json.loads(gzip.open(ROOT / "source_sync/SOURCE_SNAPSHOT_MANIFEST.json.gz", "rt").read())["files"]
    bad = [p for p, e in sm.items() if not (Path(SRC) / p).exists() or prod3d.sha_file(Path(SRC) / p) != e["sha256"]]
    res["source_snapshot"] = {"files": len(sm), "differing": bad[:5], "selection_reproduces": SS.select_check(SRC)[0]}
    t = Path(tempfile.mkdtemp())
    (t / "qc").mkdir()
    for f in (ROOT / "production_v3/qc/rerun").glob("part*.jsonl"):
        shutil.copy(f, t / "qc")
    rc, so, se = sh(PY, "code/qc3d.py", "classify", SRC, "production_v3/qc/author_evidence", t / "qc")
    res["qc_rebuilt_identical"] = rc == 0 and filecmp.cmp(t / "qc/QC_ELIGIBLE_ROOTS.json", ROOT / "production_v3/QC_ELIGIBLE_ROOTS.json", shallow=False)
    rc, so, se = sh(PY, "code/cohort3d.py", "build", SRC, "production_v3/QC_ELIGIBLE_ROOTS.json", "production_v2/split_manifest_live.json",
                    "production_v2/production_roster_live.jsonl.gz", "source_sync/SOURCE_SNAPSHOT_MANIFEST.json.gz", t / "cohort")
    names = ["first_cohort_roster.jsonl.gz", "split_manifest_first_cohort.json", "forecast_first_cohort.json", "cohort_manifest.json"]
    same = {n: filecmp.cmp(t / "cohort/frozen" / n, ROOT / "production_v3/frozen" / n, shallow=False) for n in names}
    pa, pb = (json.loads(p.read_text()) for p in (t / "cohort/frozen/source_pin_first_cohort.json", ROOT / "production_v3/frozen/source_pin_first_cohort.json"))
    same["source_pin (roots + source files; source files are path-relative)"] = pa == pb
    same["SPLIT_AND_ROSTER_AUDIT.json"] = filecmp.cmp(t / "cohort/SPLIT_AND_ROSTER_AUDIT.json", ROOT / "production_v3/SPLIT_AND_ROSTER_AUDIT.json", shallow=False)
    res["cohort_rebuilt_identical"] = {"rc": rc, **same}
    rc, so, se = sh(PY, "code/prod3d.py", "template", "--mode", "PRODUCTION_FIRST_COHORT", "--policy", "adaptive_v1", "--workers", 12, "--cpus", "0-11",
                    "--run-dir", "production_v3/run_first_cohort", "--roster", "production_v3/frozen/first_cohort_roster.jsonl.gz",
                    "--source-pin", "production_v3/frozen/source_pin_first_cohort.json", "--split", "production_v3/frozen/split_manifest_first_cohort.json",
                    "--qc", "production_v3/QC_ELIGIBLE_ROOTS.json", "--cohort", "production_v3/frozen/cohort_manifest.json", "--shelf-src", SRC,
                    "--forecast", "production_v3/frozen/forecast_first_cohort.json", "--out", t / "regen.json")
    a, b = (json.loads(p.read_text())["frozen"] for p in (ROOT / "production_v3/production_policy.TEMPLATE.json", t / "regen.json"))
    a["source"].pop("SHELF_SRC"); b["source"].pop("SHELF_SRC")
    res["template_frozen_equal_except_SHELF_SRC"] = rc == 0 and a == b
    rc, so, se = sh(PY, "code/collect3d.py", "--policy-file", t / "regen.json", "--dry-run")
    d = json.loads(so.strip().splitlines()[-1]) if so.strip() else {}
    res["preflight_from_clone"] = {"rc": rc, **{k: d.get(k) for k in ("physical_queries", "planner_calls", "teacher_entry_calls", "run_dir_created",
                                                                      "refusals_other_than_authorization", "roster_regenerated_equal", "by_split")}}
    rc, so, se = sh("taskset", "-c", "12-13", PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", "code/test_prod3d_v3.py", timeout=3600)
    res["pytest_v3_from_clone"] = so.strip().splitlines()[-1] if so.strip() else se[-300:]
    rc, so, se = sh("taskset", "-c", "0-3", PY, "code/replay3d.py", "ours", "production_v3/integration/smoke_A", "production_v3/integration/smoke_roster.jsonl.gz",
                    t / "replay_A.json", "--workers", 4, "--cpus", "0-3")
    r = json.loads((t / "replay_A.json").read_text()) if rc == 0 else {}
    res["smoke_A_replay_from_clone"] = {k: r.get(k) for k in ("replayed", "verified", "n_failures", "max_q_exec_diff")}
    p = res["preflight_from_clone"]
    res["pass"] = bool(res["head_matches"] and res["v3_sha256sums_ok"] and res["v2_manifest_ok"] and not bad and res["source_snapshot"]["selection_reproduces"]
                       and res["qc_rebuilt_identical"] and all(v for k, v in same.items()) and res["template_frozen_equal_except_SHELF_SRC"]
                       and p["rc"] == 0 and p["planner_calls"] == 0 and p["teacher_entry_calls"] == 0 and p["refusals_other_than_authorization"] == []
                       and " passed" in res["pytest_v3_from_clone"] and " failed" not in res["pytest_v3_from_clone"]
                       and r.get("replayed") == r.get("verified") == 8)
    Path(out).write_text(json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
