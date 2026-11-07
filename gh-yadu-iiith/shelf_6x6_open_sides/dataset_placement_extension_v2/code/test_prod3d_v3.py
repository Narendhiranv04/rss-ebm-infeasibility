"""Production v3 integration tests: refusal paths of the launcher / collector / guard, the exclusive lock, the metadata preflight, label
semantics, extended unit validation and protocol identity.  Every test drives the real code (CLI subprocesses or module functions); none
can start a physical collection (each carries a deliberate refusal condition), and no test ever computes a token for a PRODUCTION file.
Policy fixtures are generated per session in a temporary directory from the frozen production_v3 inputs.
    pytest -q code/test_prod3d_v3.py
"""
import gzip, json, os, subprocess, sys, time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PY = sys.executable
sys.path.insert(0, str(HERE))
import prod3d  # noqa: E402

V3, FZ = ROOT / "production_v3", ROOT / "production_v3" / "frozen"
NEVER_T, NEVER_P = "production_v3/integration/pytest_must_not_exist", "production_v3/run_pytest_must_not_exist"
SRC = str(Path(os.environ.get("SHELF_SRC", str(ROOT.parent / "dataset_bd_classquota_without_home_position_no_overpass"))).resolve())
ENV = {k: v for k, v in os.environ.items() if k not in prod3d.OVERRIDES} | {"SHELF_SRC": SRC}


def run(args, env=None, timeout=1200):
    return subprocess.run([str(a) for a in args], cwd=ROOT, env=env or ENV, capture_output=True, text=True, timeout=timeout)


def mk(tmp, mode, name, **kw):
    a = dict(policy="adaptive_v1", workers=12, cpus="0-11", run_dir=NEVER_P, roster=FZ / "first_cohort_roster.jsonl.gz")
    if mode == "INTEGRATION_TEST":
        a.update(workers=4, cpus="0-3", run_dir=NEVER_T, roster=V3 / "integration/smoke_roster.jsonl.gz")
    a.update(kw)
    out = Path(tmp) / name
    r = run([PY, "code/prod3d.py", "template", "--mode", mode, "--policy", a["policy"], "--workers", a["workers"], "--cpus", a["cpus"],
             "--run-dir", a["run_dir"], "--roster", a["roster"], "--source-pin", FZ / "source_pin_first_cohort.json",
             "--split", FZ / "split_manifest_first_cohort.json", "--qc", V3 / "QC_ELIGIBLE_ROOTS.json", "--cohort", FZ / "cohort_manifest.json",
             "--shelf-src", SRC, "--forecast", FZ / "forecast_first_cohort.json", "--out", out])
    assert r.returncode == 0, r.stderr[-2000:]
    return out


@pytest.fixture(scope="session")
def pols(tmp_path_factory):
    t = tmp_path_factory.mktemp("pols")
    return t, mk(t, "PRODUCTION_FIRST_COHORT", "prod.json"), mk(t, "INTEGRATION_TEST", "test.json")


def edit(src, dst, f):
    d = json.loads(Path(src).read_text()); f(d); Path(dst).write_text(json.dumps(d)); return dst


def token(p, env=None):
    assert json.loads(Path(p).read_text())["mode"] == "INTEGRATION_TEST", "tests never compute a production token"
    r = run([PY, "code/prod3d.py", "token", p], env=env)
    assert r.returncode == 0, r.stderr[-1500:]
    return r.stdout.strip()


def refused(r, *needles):
    out = r.stdout + r.stderr
    assert r.returncode != 0 and "REFUSED" in out, out[-2500:]
    for n in needles:
        assert n in out, (n, out[-2500:])
    assert not (ROOT / NEVER_T).exists() and not (ROOT / NEVER_P).exists(), "a refused run created its run directory"


def collect(p, tok=None, env=None, *extra):
    return run([PY, "code/collect3d.py", "--policy-file", p] + (["--confirm", tok] if tok else []) + list(extra), env=env)


# ---- launcher (bash, real argument handling) ----
def test_launcher_refuses_no_args():
    refused(run(["bash", "production_v3/launch_first_cohort.sh"]), "no arguments")


def test_launcher_refuses_incomplete_args(pols):
    _, prod, _ = pols
    refused(run(["bash", "production_v3/launch_first_cohort.sh", "--policy-file", prod, "--confirm", "x"]), "--execute missing")
    refused(run(["bash", "production_v3/launch_first_cohort.sh", "--execute", "--policy-file", prod]), "--confirm <token> missing")
    refused(run(["bash", "production_v3/launch_first_cohort.sh", "--execute", "--confirm", "x"]), "--policy-file missing")
    refused(run(["bash", "production_v3/launch_first_cohort.sh", "--execute", "--policy-file", prod, "--confirm", "x"],
                env={k: v for k, v in ENV.items() if k != "SHELF_SRC"}), "SHELF_SRC is not exported")


def test_launcher_refuses_unauthorized_template_and_wrong_token(pols):
    _, prod, _ = pols
    r = run(["bash", "production_v3/launch_first_cohort.sh", "--execute", "--policy-file", prod, "--confirm", "0" * 32])
    refused(r, "no recorded user authorization", "confirmation token missing or !=", "authorization / hash / lock / resource check failed")


def test_launcher_refuses_conflicting_lock(pols, tmp_path):
    _, prod, _ = pols
    h = subprocess.Popen([PY, "-c", "import sys,time;sys.path.insert(0,'code');import prod3d;prod3d.acquire(prod3d.LOCK_FILE,{'who':'pytest-holder'});"
                          "print('HELD',flush=True);time.sleep(600)"], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    try:
        assert h.stdout.readline().strip() == "HELD"
        r = run(["bash", "production_v3/launch_first_cohort.sh", "--execute", "--policy-file", prod, "--confirm", "0" * 32])
        refused(r, "exclusive collector lock is held", "pytest-holder")
    finally:
        h.kill(); h.wait()
    assert prod3d.probe(prod3d.LOCK_FILE) is None, "SIGKILL of the holder must release the lock"


# ---- collector / guard refusals (real collect3d.py) ----
def test_collector_refuses_unauthorized_or_missing_token(pols):
    _, prod, _ = pols
    refused(collect(prod, "0" * 32), "no recorded user authorization", "confirmation token missing or !=")
    refused(collect(prod), "confirmation token missing")


def test_refuses_altered_roster(pols, tmp_path):
    _, prod, test = pols
    rows = [json.loads(x) for x in gzip.open(V3 / "integration/smoke_roster.jsonl.gz", "rt")]
    rows[0]["dest"] = [rows[0]["dest"][0], (rows[0]["dest"][1] + 1) % 6]
    alt = tmp_path / "alt_roster.jsonl.gz"
    with gzip.open(alt, "wt") as f:
        f.writelines(json.dumps(x, sort_keys=True) + "\n" for x in rows)
    p = edit(test, tmp_path / "t.json", lambda d: d["inputs"].update(roster=str(alt)))
    refused(collect(p, token(p)), "frozen inputs differs", "test rows are not rows of the first-cohort roster")
    q = edit(prod, tmp_path / "p.json", lambda d: d["inputs"].update(roster=str(alt)))
    refused(collect(q, "0" * 32), "frozen inputs differs", "roster is not the one the cohort manifest froze")


def test_refuses_altered_pin(pols, tmp_path):
    _, _, test = pols
    pin = json.loads((FZ / "source_pin_first_cohort.json").read_text())
    r0 = json.loads(gzip.open(V3 / "integration/smoke_roster.jsonl.gz", "rt").readline())["root_id"]
    pin["roots"][r0] = "0" * 64
    (tmp_path / "pin.json").write_text(json.dumps(pin))
    p = edit(test, tmp_path / "t.json", lambda d: d["inputs"].update(source_pin=str(tmp_path / "pin.json")))
    refused(collect(p, token(p)), "frozen inputs differs", "roster roots differ from the source pin", "roster roots without a QC pass")


def test_refuses_root_drift_in_shelf_src(pols, tmp_path):
    _, _, test = pols
    src, alt = Path(SRC), tmp_path / "src"
    (alt / "raw/roots").mkdir(parents=True)
    (alt / "code").symlink_to(src / "code"); (alt / "raw/witnesses").symlink_to(src / "raw/witnesses")
    for f in ("selected.json", "raw/EXCLUDED_FIXED_OBJECT.json"):
        (alt / f).write_bytes((src / f).read_bytes())
    r0 = json.loads(gzip.open(V3 / "integration/smoke_roster.jsonl.gz", "rt").readline())["root_id"]
    for f in (src / "raw/roots").iterdir():
        (alt / "raw/roots" / f.name).symlink_to(f) if f.stem != r0 else (alt / "raw/roots" / f.name).write_bytes(f.read_bytes() + b" ")
    env = ENV | {"SHELF_SRC": str(alt)}
    refused(collect(test, token(test, env), env), "SHELF_SRC resolves to", "roster roots differ from the source pin")


def test_refuses_schedule_policy_and_cpu_mismatch(pols, tmp_path):
    _, prod, test = pols
    refused(collect(edit(prod, tmp_path / "a.json", lambda d: d.update(stages=[[4, 1500]])), "0" * 32), "!= the implemented schedule")
    refused(collect(edit(prod, tmp_path / "b.json", lambda d: d.update(policy="fixed_4x1500")), "0" * 32),
            "policy 'fixed_4x1500' != the frozen production value 'adaptive_v1'")
    refused(collect(edit(prod, tmp_path / "c.json", lambda d: d.update(workers=13, cpus="0-12")), "0" * 32),
            "workers 13 != the frozen production value 12", "cpus '0-12' != the frozen production value '0-11'")
    p = edit(test, tmp_path / "d.json", lambda d: d.update(workers=6, cpus="0-5"))
    refused(collect(p, token(p)), "<= 4 workers")


def test_refuses_sensor_loss_and_overrides(pols, tmp_path):
    _, prod, test = pols
    refused(collect(test, token(test), ENV | {"RUN3D_SENSOR_OVERRIDE": str(tmp_path / "no_such_sensor")}), "temperature sensor unreadable")
    refused(collect(prod, "0" * 32, ENV | {"RUN3D_SENSOR_OVERRIDE": str(tmp_path / "x")}), "test-only environment override RUN3D_SENSOR_OVERRIDE")
    refused(collect(prod, "0" * 32, ENV | {"SHELF66V3_LOCK_FILE": str(tmp_path / "l")}), "test-only environment override SHELF66V3_LOCK_FILE")


def test_test_fixture_never_authorizes_production(pols, tmp_path):
    _, _, test = pols
    p = edit(test, tmp_path / "x.json", lambda d: d.update(mode="PRODUCTION_FIRST_COHORT"))
    refused(collect(p, "0" * 32), "must not carry the integration-test fixture", "frozen mode differs", "no recorded user authorization")
    q = edit(test, tmp_path / "y.json", lambda d: d["authorization"].update(authorized_by="someone", authorized_utc="now", statement="go"))
    refused(collect(q, token(q)), "must leave the user-authorization fields empty")


def test_test_roster_bounds(pols, tmp_path):
    _, _, test = pols
    rows = [json.loads(x) for x in gzip.open(V3 / "integration/smoke_roster.jsonl.gz", "rt")]
    full = [json.loads(x) for x in gzip.open(FZ / "first_cohort_roster.jsonl.gz", "rt")]
    extra = next(x for x in full if not x["static"] and x["split"] == "train" and x["qid"] not in {r["qid"] for r in rows})
    val = next(x for x in full if not x["static"] and x["split"] == "val")
    for name, rr in (("13", rows + [extra]), ("val", rows[:11] + [val])):
        f = tmp_path / f"r{name}.jsonl.gz"
        with gzip.open(f, "wt") as g:
            g.writelines(json.dumps(x, sort_keys=True) + "\n" for x in rr)
        p = edit(test, tmp_path / f"p{name}.json", lambda d: d["inputs"].update(roster=str(f)))
        refused(collect(p, token(p)), "test roster exceeds 12 TRAIN queries / 4 roots")


def test_lock_held_refuses_and_sigkill_releases(pols, tmp_path):
    _, _, test = pols
    lk = tmp_path / "collector.lock"
    h = subprocess.Popen([PY, "-c", "import sys,time;sys.path.insert(0,'code');import prod3d;prod3d.acquire(sys.argv[1],{'who':'holder'});"
                          "print('HELD',flush=True);time.sleep(600)", str(lk)], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    try:
        assert h.stdout.readline().strip() == "HELD"
        refused(collect(test, token(test), ENV | {"SHELF66V3_LOCK_FILE": str(lk)}), "exclusive collector lock is held", "holder")
        with pytest.raises(prod3d.LockHeld):
            prod3d.acquire(lk, {"who": "second"})
    finally:
        h.kill(); h.wait()
    assert prod3d.probe(lk) is None
    fd = prod3d.acquire(lk, {"who": "after-crash"}); os.close(fd)


def test_lock_race_exactly_one_winner(tmp_path):
    lk, go = tmp_path / "race.lock", time.time() + 2.0
    code = ("import sys,time,os;sys.path.insert(0,'code');import prod3d\nwhile time.time()<float(sys.argv[2]):pass\n"
            "try:\n fd=prod3d.acquire(sys.argv[1],{'i':sys.argv[3]});print('WON',flush=True);time.sleep(3)\nexcept prod3d.LockHeld:\n print('LOST')")
    ps = [subprocess.Popen([PY, "-c", code, str(lk), str(go), str(i)], cwd=ROOT, stdout=subprocess.PIPE, text=True) for i in range(8)]
    outs = [p.communicate(timeout=60)[0].strip() for p in ps]
    assert outs.count("WON") == 1 and outs.count("LOST") == 7, outs


def test_stale_lock_text_is_not_a_lock(tmp_path):
    lk = tmp_path / "stale.lock"
    lk.write_text('{"pid": 999999, "note": "left by a crashed run"}')
    assert prod3d.probe(lk) is None
    fd = prod3d.acquire(lk, {"who": "new"}); os.close(fd)


# ---- metadata preflight (zero teacher calls, nothing created) ----
def test_dry_run_metadata_preflight_zero_teacher_calls(pols):
    _, prod, _ = pols
    r = collect(prod, None, None, "--dry-run")
    assert r.returncode == 0, (r.stdout + r.stderr)[-3000:]
    d = json.loads(r.stdout.strip().splitlines()[-1])
    aud = json.loads((V3 / "SPLIT_AND_ROSTER_AUDIT.json").read_text())["roster"]
    assert d["planner_calls"] == 0 and d["teacher_entry_calls"] == 0 and not d["run_dir_created"] and d["refusals_other_than_authorization"] == []
    assert any("no recorded user authorization" in x for x in d["authorization_refusals"])
    assert d["physical_queries"] == aud["physical_queries"] == d["queries_in_units"] and d["roster_regenerated_equal"] is True
    assert set(d["by_split"]) == {"train", "val", "test"} and d["static_rows_excluded_from_units"]


# ---- label semantics, extended unit validation, protocol identity, template ----
def test_label_semantics():
    import collect3d as C
    ok1 = {"label": 1, "status": "VERIFIED", "witness_id": "w", "witnesses": [{"witness_id": "w"}]}
    assert C.semantics(ok1) == []
    assert C.semantics({"label": 0, "status": "NEG_UNDER_POLICY", "negative_kind": "BOUNDED_SEARCH", "scope": {"a": 1}, "witnesses": []}) == []
    assert C.semantics({"label": None, "status": "INTERRUPTED", "witnesses": []}) == []
    assert C.semantics({"label": 0, "status": "INTERRUPTED", "witnesses": []})                 # an interruption is never a negative
    assert C.semantics({"label": None, "status": "NEG_UNDER_POLICY", "witnesses": []})          # a negative status is never unlabeled
    assert C.semantics({"label": 0, "status": "NEG_CERTIFIED_GEOMETRY", "negative_kind": "CERTIFIED_GEOMETRY", "witnesses": []})  # no certificate
    assert C.semantics({**ok1, "witnesses": []}) and C.semantics({"label": 2, "status": "VERIFIED"})
    for f in sorted((ROOT / "mock_v2/runs/adaptive_v1_A/units").glob("*.json")):  # every real V2 record is valid
        assert all(C.semantics(r) == [] for r in json.loads(f.read_text())["records"]), f.name


def test_validate_unit_v3_checks_protocol_pin_and_semantics(tmp_path):
    import collect3d as C
    src = ROOT / "mock_v2/runs/adaptive_v1_A/units"
    uid = next(f.stem for f in sorted(src.glob("*.npz")))
    (tmp_path / "units").mkdir()
    for ext in ("json", "npz"):
        (tmp_path / "units" / f"{uid}.{ext}").write_bytes((src / f"{uid}.{ext}").read_bytes())
    doc = json.loads((src / f"{uid}.json").read_text())
    qids, root = [r["qid"] for r in doc["records"]], doc["records"][0]["qid"].split(":")[0]
    C.CTX.clear(); C.CTX.update(protocol=doc["protocol"], roots={root: doc["root_sha256"]})
    assert C.validate_unit_v3(tmp_path, uid, qids) == (True, "ok")
    C.CTX["protocol"] = "0" * 16
    assert C.validate_unit_v3(tmp_path, uid, qids) == (False, "protocol_mismatch")
    C.CTX.update(protocol=doc["protocol"], roots={root: "0" * 64})
    assert C.validate_unit_v3(tmp_path, uid, qids) == (False, "root_pin_mismatch")
    C.CTX.clear()


def test_teacher_change_changes_protocol_and_is_refused(pols):
    _, _, test = pols
    code = ("import sys,json;sys.path.insert(0,'code');import placement3d as P,prod3d;h=P.protocol()[0];P.TOL['xy_m']=0.03;"
            "ok,why,auth,cur=prod3d.check(sys.argv[1],'x');print(json.dumps([h,P.protocol()[0],why]))")
    r = run([PY, "-c", code, test])
    h0, h1, why = json.loads(r.stdout.strip().splitlines()[-1])
    assert h0 != h1 and any("revalidate first" in w for w in why) and any("frozen teacher_protocol differs" in w for w in why)


def test_template_records_actual_inputs_and_committed_template_unauthorized(pols):
    _, prod, test = pols
    for p in (prod, test):
        d = json.loads(Path(p).read_text())
        assert "production_v2" not in json.dumps(d["inputs"]) and d["source_layout"]["SHELF_SRC"] == SRC
        for k, v in d["inputs"].items():
            assert d["frozen"]["inputs"][k] == {"path": v, "sha256": prod3d.sha_file(ROOT / v)}
        assert d["stages"] == [[4, 500], [8, 1500], [16, 5000]] and all(v is None for v in d["authorization"].values())
    c = json.loads((V3 / "production_policy.TEMPLATE.json").read_text())
    assert c["mode"] == "PRODUCTION_FIRST_COHORT" and all(v is None for v in c["authorization"].values()) and c["test_authorization"] is None
    assert c["inputs"]["roster"] == "production_v3/frozen/first_cohort_roster.jsonl.gz" and (c["workers"], c["cpus"]) == (12, "0-11")


def test_frozen_v2_teacher_and_collector_bytes_unchanged():
    v2 = json.loads((ROOT / "mock_v2/production_policy.TEMPLATE.json").read_text())["frozen"]["code_sha256"]
    assert {n: prod3d.sha_file(HERE / n) for n in v2} == v2


def test_collector_process_group_as_session_leader():
    """A collector exec'd as a session leader (tmux shell exec) must not crash on setpgid (EPERM) and must lead its process group."""
    r = subprocess.run([PY, "-c", "import sys,os;sys.path.insert(0,'code');import collect3d;print(collect3d.own_group(), os.getsid(0)==os.getpid())"],
                       cwd=ROOT, env=ENV, capture_output=True, text=True, start_new_session=True, timeout=300)
    assert r.returncode == 0 and r.stdout.split() == ["True", "True"], r.stdout + r.stderr[-1500:]
    r = subprocess.run([PY, "-c", "import sys,os;sys.path.insert(0,'code');import collect3d;print(collect3d.own_group())"],
                       cwd=ROOT, env=ENV, capture_output=True, text=True, timeout=300)
    assert r.returncode == 0 and r.stdout.split() == ["True"], r.stdout + r.stderr[-1500:]


def test_ledger_torn_last_line_is_terminated_before_append(tmp_path):
    import collect3d as C
    p = tmp_path / "ledger.jsonl"
    p.write_text('{"event": "unit_done", "unit": "a"}\n{"event": "unit_done", "unit": "tor')
    h = C.ledger_open(p); h.write(json.dumps({"event": "start"}) + "\n"); h.close()
    lines = p.read_text().splitlines()
    assert json.loads(lines[-1]) == {"event": "start"} and lines[-2].endswith('"tor') and json.loads(lines[0])["unit"] == "a"
    h = C.ledger_open(p); h.write("x\n"); h.close()                  # an intact ledger gets no extra newline
    assert p.read_text().splitlines()[-2:] == ['{"event": "start"}', "x"]
