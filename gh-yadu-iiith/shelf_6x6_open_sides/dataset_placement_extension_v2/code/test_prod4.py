"""Production v4 tests: the Q2_3D_PARITY_V1 roster rule and the v4 guard / collector / launcher on the real code paths.  No test can start a
physical collection (each carries a refusal condition) and no test computes a token for a production-mode file.
    pytest -q code/test_prod4.py
"""
import gzip, json, os, subprocess, sys, time
from collections import Counter
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
sys.path.insert(0, str(HERE))
import prod3d  # noqa: E402

V3, V4 = ROOT / "production_v3", ROOT / "production_v4"
NEVER_T, NEVER_P = "production_v4/integration/pytest_must_not_exist", "production_v4/run_pytest_must_not_exist"
SRC = str(Path(os.environ.get("SHELF_SRC", str(ROOT.parent / "dataset_bd_classquota_without_home_position_no_overpass"))).resolve())
ENV = {k: v for k, v in os.environ.items() if k not in prod3d.OVERRIDES} | {"SHELF_SRC": SRC}
ROSTER = V4 / "frozen/q2_parity_roster_v4.jsonl.gz"


def run(args, env=None, timeout=1800):
    return subprocess.run([str(a) for a in args], cwd=ROOT, env=env or ENV, capture_output=True, text=True, timeout=timeout)


def mk(tmp, mode, name, guard="prod4"):
    if guard == "prod4":
        a = dict(workers=12, cpus="0-11", run_dir=NEVER_P, roster=ROSTER, cohort=V4 / "frozen/roster_manifest_v4.json", fc=V4 / "frozen/forecast_v4.json")
        if mode == "INTEGRATION_TEST":
            a.update(workers=4, cpus="0-3", run_dir=NEVER_T, roster=V4 / "integration/smoke_roster_v4.jsonl.gz")
    else:                                                       # a V3 policy, for the cross-version refusal
        a = dict(workers=12, cpus="0-11", run_dir="production_v3/run_pytest_must_not_exist", roster=V3 / "frozen/first_cohort_roster.jsonl.gz",
                 cohort=V3 / "frozen/cohort_manifest.json", fc=V3 / "frozen/forecast_first_cohort.json")
    out = Path(tmp) / name
    r = run([PY, f"code/{guard}.py", "template", "--mode", mode, "--policy", "adaptive_v1", "--workers", a["workers"], "--cpus", a["cpus"],
             "--run-dir", a["run_dir"], "--roster", a["roster"], "--source-pin", V3 / "frozen/source_pin_first_cohort.json",
             "--split", V3 / "frozen/split_manifest_first_cohort.json", "--qc", V3 / "QC_ELIGIBLE_ROOTS.json", "--cohort", a["cohort"],
             "--shelf-src", SRC, "--forecast", a["fc"], "--out", out])
    assert r.returncode == 0, r.stderr[-2000:]
    return out


@pytest.fixture(scope="session")
def pols(tmp_path_factory):
    t = tmp_path_factory.mktemp("pols4")
    return t, mk(t, "PRODUCTION_FIRST_COHORT", "prod.json"), mk(t, "INTEGRATION_TEST", "test.json"), mk(t, "PRODUCTION_FIRST_COHORT", "v3.json", "prod3d")


def edit(src, dst, f):
    d = json.loads(Path(src).read_text()); f(d); Path(dst).write_text(json.dumps(d)); return dst


def token(p):
    assert json.loads(Path(p).read_text())["mode"] == "INTEGRATION_TEST", "tests never compute a production token"
    r = run([PY, "code/prod4.py", "token", p])
    assert r.returncode == 0, r.stderr[-1500:]
    return r.stdout.strip()


def refused(r, *needles):
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "REFUSED" in out, out[-2500:]
    for n in needles:
        assert n in out, (n, out[-2500:])
    assert not (ROOT / NEVER_T).exists() and not (ROOT / NEVER_P).exists()


def collect(p, tok=None, env=None, *extra, entry="collect4"):
    return run([PY, f"code/{entry}.py", "--policy-file", p] + (["--confirm", tok] if tok else []) + list(extra), env=env)


# ---- the roster rule ----
@pytest.fixture(scope="session")
def roster():
    return [json.loads(x) for x in gzip.open(ROSTER, "rt")]


def test_per_root_counts_follow_the_q2_formula(roster):
    nt = {r: v for r, v in json.loads((V3 / "QC_ELIGIBLE_ROOTS.json").read_text())["roots"].items()}
    per = Counter(x["root_id"] for x in roster if not x["static"])
    want = {0: 0, 1: 8, 2: 24, 3: 44}
    assert all(per.get(r, 0) == want.get(v["non_target_objects"], 56) for r, v in nt.items())
    assert sum(per.values()) == 59972


def test_prefix_contexts_are_independent_valid_edits(roster):
    ctx = [json.loads(x) for x in gzip.open(V4 / "frozen/q2_contexts_v4.jsonl.gz", "rt")]
    by = {(c["root_id"], c["block"]): c for c in ctx}
    n = 0
    for (r, b), c in by.items():
        if b == "prefix2" and c["status"] == "APPLICABLE":
            p1 = by[(r, "prefix1")]
            assert len(c["moves"]) == 1 and c["moves"][0][0] != p1["moves"][0][0] and c["parent_scene"] == by[(r, "root")]["scene_hash"]
            n += 1
        if b == "prefix3" and c["status"] == "APPLICABLE":
            assert len(c["moves"]) == 2 and c["moves"][0][0] != c["moves"][1][0] and not any(o in [m[0] for m in c["moves"]] for o in c["objects"])
    assert n > 1000


def test_rule_ignores_protected_metadata():
    import q2roster3d as Q
    src = Path(SRC)
    sel = json.loads((src / "selected.json").read_text())["selected"]
    split = json.loads((V3 / "frozen/split_manifest_first_cohort.json").read_text())["split"]
    hb = lambda h: "short" if h < 0.13 else "medium" if h < 0.155 else "tall"  # noqa: E731
    for r in sel["3"][:25] + sel["1"][:25]:
        raw = json.loads((src / "raw/roots" / f"{r}.json").read_text())
        a = Q.plan_root(Q.allowed(raw, "3"), split[r], hb, "snap")
        for k in ("roles", "labels", "intended_blockers", "greedy_blockers", "parking", "mechanism_pools", "spec", "verified_K", "independent_moves",
                  "alt_parking", "label_consistency", "observed", "teacher_calls", "construction"):
            raw[k] = {"scrambled": k} if not isinstance(raw.get(k), list) else ["scrambled"]
        assert Q.plan_root(Q.allowed(raw, "3"), split[r], hb, "snap") == a


def test_destinations_free_and_bands_by_chebyshev(roster):
    for x in roster:
        occ = {tuple(c) for c in x["cells"].values()}
        if x["static"] is None:
            assert tuple(x["dest"]) not in occ and 0 <= x["dest"][0] < 6 and 0 <= x["dest"][1] < 6
            assert x["band"] == "vacated" or x["band"] == ("near" if x["cheb"] == 1 else "mid" if x["cheb"] <= 3 else "far")
            assert x["incl_prob"] and 0 < x["incl_prob"] <= 1


def test_roster_regenerates_identically():
    import q2roster3d as Q
    spl = json.loads((V3 / "frozen/split_manifest_first_cohort.json").read_text())["split"]
    snap = json.loads((V3 / "frozen/source_pin_first_cohort.json").read_text())["manifest_sha256"]
    _, rows, _, dmap, _ = Q.build(SRC, spl, snap)
    assert [json.dumps(x, sort_keys=True, default=list) for x in rows] == [json.dumps(json.loads(x), sort_keys=True) for x in gzip.open(ROSTER, "rt")]


# ---- guard / collector / launcher (real CLIs) ----
def test_launcher_refuses_no_args_and_unauthorized(pols):
    _, prod, _, _ = pols
    refused(run(["bash", "production_v4/launch_v4.sh"]), "no arguments")
    refused(run(["bash", "production_v4/launch_v4.sh", "--execute", "--policy-file", prod, "--confirm", "0" * 32]),
            "no recorded user authorization", "confirmation token missing or !=")


def test_v3_policy_and_token_never_authorize_v4(pols):
    _, prod, _, v3 = pols
    refused(collect(v3, "0" * 32), "is not a SHELF66V3_PLACE_PRODUCTION_V4_Q2_PARITY file")
    refused(collect(prod, "0" * 32, entry="collect3d"), "is not a SHELF66V3_PLACE_PRODUCTION_V3 file")
    assert json.loads(Path(prod).read_text())["version"] != json.loads(Path(v3).read_text())["version"]


def test_refuses_altered_roster_schedule_cpu_and_bounds(pols, tmp_path):
    _, prod, test, _ = pols
    rows = [json.loads(x) for x in gzip.open(V4 / "integration/smoke_roster_v4.jsonl.gz", "rt")]
    rows[0]["dest"] = [rows[0]["dest"][0], (rows[0]["dest"][1] + 1) % 6]
    alt = tmp_path / "alt.jsonl.gz"
    with gzip.open(alt, "wt") as f:
        f.writelines(json.dumps(x, sort_keys=True) + "\n" for x in rows)
    p = edit(test, tmp_path / "t.json", lambda d: d["inputs"].update(roster=str(alt)))
    refused(collect(p, token(p)), "frozen inputs differs", "test rows are not rows of the first-cohort roster")
    refused(collect(edit(prod, tmp_path / "a.json", lambda d: d.update(stages=[[8, 5000]])), "0" * 32), "!= the implemented schedule")
    refused(collect(edit(prod, tmp_path / "c.json", lambda d: d.update(workers=13, cpus="0-12")), "0" * 32), "workers 13 != the frozen production value 12")
    full = [json.loads(x) for x in gzip.open(ROSTER, "rt")]
    more = rows[1:] + [x for x in full if not x["static"] and x["split"] == "train" and x["qid"] not in {r["qid"] for r in rows}][:2]
    big = tmp_path / "big.jsonl.gz"
    with gzip.open(big, "wt") as f:
        f.writelines(json.dumps(x, sort_keys=True) + "\n" for x in more)
    q = edit(test, tmp_path / "b.json", lambda d: d["inputs"].update(roster=str(big)))
    refused(collect(q, token(q)), "test roster exceeds 32 TRAIN queries / 4 roots")


def test_refuses_sensor_loss_and_held_lock(pols, tmp_path):
    _, _, test, _ = pols
    refused(collect(test, token(test), ENV | {"RUN3D_SENSOR_OVERRIDE": str(tmp_path / "none")}), "temperature sensor unreadable")
    lk = tmp_path / "l.lock"
    h = subprocess.Popen([PY, "-c", "import sys,time;sys.path.insert(0,'code');import prod3d;prod3d.acquire(sys.argv[1],{'who':'holder4'});"
                          "print('HELD',flush=True);time.sleep(600)", str(lk)], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    try:
        assert h.stdout.readline().strip() == "HELD"
        refused(collect(test, token(test), ENV | {"SHELF66V3_LOCK_FILE": str(lk)}), "exclusive collector lock is held", "holder4")
    finally:
        h.kill(); h.wait()
    assert prod3d.probe(lk) is None


def test_dry_run_regenerates_full_roster_with_zero_teacher_calls(pols):
    _, prod, _, _ = pols
    r = collect(prod, None, None, "--dry-run")
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    d = json.loads(r.stdout.strip().splitlines()[-1])
    assert d["planner_calls"] == 0 and d["teacher_entry_calls"] == 0 and not d["run_dir_created"] and d["refusals_other_than_authorization"] == []
    assert d["physical_queries"] == 59972 == d["queries_in_units"] and d["roster_regenerated_equal"] is True and d["semantic_keys_unique"]
    assert set(d["by_block"]) == {"root", "prefix1", "prefix2", "prefix3"} and set(d["by_split"]) == {"train", "val", "test"}


def test_collector_scan_sees_v4_entry_points():
    p = subprocess.Popen(["bash", "-c", "exec -a 'python code/collect4.py --policy-file x' sleep 30"])
    try:
        time.sleep(0.5)
        import prod4  # noqa: F401
        assert p.pid in prod3d.other_collectors()
    finally:
        p.kill(); p.wait()


def test_frozen_v2_v3_code_unchanged():
    v2 = json.loads((ROOT / "mock_v2/production_policy.TEMPLATE.json").read_text())["frozen"]["code_sha256"]
    v3 = json.loads((V3 / "production_policy.TEMPLATE.json").read_text())["frozen"]["code_sha256"]
    assert {n: prod3d.sha_file(HERE / n) for n in v2} == v2 and {n: prod3d.sha_file(HERE / n) for n in v3} == v3
