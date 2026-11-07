"""Tests of the post-3D 2D spillover queue (spill2d.py): 3D gate, server profile, allowlists, canary verdict, single-winner tick, exact disjoint
partition with no overwrite, verified partial copies, merge refusals.  No test runs the teacher.   pytest -q q2_spillover/test_spill2d.py"""
import json, os, shutil, subprocess, sys, time, types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import spill2d as S  # noqa: E402

# Portable: on gvlab (where the plan-loading tests run, never on the laptop while its 2D production is live) point these at the staged copies.
T_LAUNCH = Path(os.environ.get("SPILL2D_TEST_LAUNCH", S.LAUNCH))          # needs amendment_q2/Q2_AMENDMENT_DECISION.json + plan + configs
T_CODE = Path(os.environ.get("SPILL2D_TEST_CODE", S.CODE))
T_PACKED = Path(os.environ.get("SPILL2D_TEST_PACKED", S.ORIG))            # a dir with records/ modes/ witnesses/ of real packed roots


def fake3d(tmp, done=1466, complete=True, audit=True, replay=True, replay_ok=True):
    side = tmp / "side"; run = side / "production_v4/run_q2_parity"; run.mkdir(parents=True)
    units = {f"u{i}": [f"u{i}:q{j}" for j in range(41 if i < 1332 else 40)] for i in range(1466)}
    (run / "run_manifest.json").write_text(json.dumps({"units": units}))
    led = [{"event": "start"}] + [{"event": "unit_done", "unit": f"u{i}"} for i in range(done)] + ([{"event": "end", "complete": complete}])
    (run / "ledger.jsonl").write_text("\n".join(map(json.dumps, led)) + "\n")
    if audit:
        (side / "production_v4/run_q2_parity_audit.json").write_text(json.dumps({"complete": True, "validated": 1466, "missing_or_invalid": [],
                                                                                "stray_files": [], "witness_associations": 100}))
    if replay:
        (side / "production_v4/run_q2_parity_replay.json").write_text(json.dumps({"replayed": 100, "verified": 100 if replay_ok else 99, "failures": []}))
    return side


NAMES3D = tuple(n for n in S.PROJECT_PROCS if n != "launch_runner.py")   # the laptop's own 2D production must not mask the 3D-side checks


def gate(side, tmp):
    return S.gate3d(True, side=side, lock=tmp / "nolock", host_checks=False, names=NAMES3D)


def test_gate_refuses_absent_incomplete_unaudited_or_unreplayed(tmp_path):
    assert not S.gate3d(True, side=tmp_path / "none", lock=tmp_path / "x", host_checks=False)["pass"]
    for kw, needle in (({"done": 1465}, "incomplete"), ({"complete": False}, "incomplete"), ({"audit": False}, "audit"),
                       ({"replay": False}, "replay"), ({"replay_ok": False}, "replay")):
        t = tmp_path / needle / str(len(kw)) / str(list(kw)[0]); t.mkdir(parents=True)
        g = gate(fake3d(t, **kw), t)
        assert not g["pass"] and any(needle in r for r in g["reasons"]), (kw, g["reasons"])
    t = tmp_path / "ok"; t.mkdir()
    assert gate(fake3d(t), t)["pass"]


def test_gate_refuses_active_3d_processes_and_held_lock(tmp_path):
    side = fake3d(tmp_path)
    for name in ("python code/collect4.py --policy-file x", "python code/replay3d.py ours run roster out"):
        p = subprocess.Popen(["bash", "-c", f"exec -a '{name}' sleep 30"])
        try:
            time.sleep(0.4)
            g = gate(side, tmp_path)
            assert not g["pass"] and any("project compute still active" in r for r in g["reasons"])
        finally:
            p.kill(); p.wait()
    # the launching chain itself is never "project compute": a grandparent shell whose command line names "spill2d.py server-run" (the real
    # 2026-11-07 canary refusal: tmux -> bash -c '... (chain)' -> subshell -> python server-run) is excluded, a sibling with the same name is not
    probe = tmp_path / "probe.py"
    probe.write_text(f"import sys, json; sys.path.insert(0, {str(HERE)!r}); import spill2d as S\n"
                     "print(json.dumps(S.project_processes(names=('spill2d.py server-run',))))\n")
    r = subprocess.run(["bash", "-c", f"bash -c '{sys.executable} {probe}; true'; true # spill2d.py server-run --scope CANARY"],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and json.loads(r.stdout.strip().splitlines()[-1]) == [], (r.stdout, r.stderr)
    sib = subprocess.Popen(["bash", "-c", "exec -a 'python spill2d.py server-run --scope SPILLOVER' sleep 30"])
    try:
        time.sleep(0.4)
        assert [p["pid"] for p in S.project_processes(names=("spill2d.py server-run",))] == [sib.pid]
    finally:
        sib.kill(); sib.wait()
    lk = tmp_path / "held.lock"
    h = subprocess.Popen([sys.executable, "-c", "import fcntl,sys,time;f=open(sys.argv[1],'w');fcntl.flock(f,fcntl.LOCK_EX);print('H',flush=True);time.sleep(30)",
                          str(lk)], stdout=subprocess.PIPE, text=True)
    try:
        assert h.stdout.readline().strip() == "H"
        g = S.gate3d(True, side=side, lock=lk, host_checks=False, names=NAMES3D)
        assert not g["pass"] and any("lock still held" in r for r in g["reasons"])
    finally:
        h.kill(); h.wait()


def test_server_profile_refusals():
    good = {"host_profile": S.SRV["host_profile"], "execution": {"workers": 12, "worker_cpus": list(range(12)), "reserved_cpus": [12, 13]},
            "safety": {"require_ac_power": False}}
    assert S.check_server_profile(good) == []
    for mut, key in ((lambda c: c["execution"].update(workers=13), "workers"), (lambda c: c["execution"].update(worker_cpus=list(range(19))), "worker_cpus"),
                     (lambda c: c["execution"].update(reserved_cpus=list(range(12, 20))), "reserved_cpus"), (lambda c: c.update(host_profile="LAPTOP"), "host_profile"),
                     (lambda c: c["safety"].update(require_ac_power=True), "no_ac_requirement_on_server")):
        c = json.loads(json.dumps(good)); mut(c)
        assert key in S.check_server_profile(c), key
    c = json.loads(json.dumps(good)); c["execution"]["worker_cpus"] = list(range(17)); c["execution"]["reserved_cpus"] = [17, 18]
    assert "cpu_ceiling" in S.check_server_profile(c)


def test_server_sensor_profile_fails_closed_without_coretemp():
    R = types.SimpleNamespace(sensors=lambda: {"Tctl": "laptop"})
    S.server_profile(R)
    has = any((h / "name").exists() and (h / "name").read_text().strip() == "coretemp" for h in Path("/sys/class/hwmon").glob("hwmon*"))
    assert (R.sensors() is not None) == has                         # never falls back to the laptop k10temp reading


def _tasks():
    return {f"r{i}": ({"split": "train" if i < 8 else "val"}, [{"trial_id": f"r{i}:t{j}"} for j in range(3)]) for i in range(10)}


def test_allowlist_refusals():
    tasks = _tasks()
    own = {"owner": {"r0": "GVLAB", "r1": "GVLAB", "r2": "LAPTOP", "r3": "ORIGINAL"}, "class": {"r0": "UNTOUCHED", "r1": "UNTOUCHED", "r2": "PARTIAL", "r3": "PACKED"}}
    ok = {"roots": ["r0", "r1"], "owner_manifest_sha256": "h", "trial_ids": {r: [t["trial_id"] for t in tasks[r][1]] for r in tasks}}
    assert S.validate_allowlist("SPILLOVER", ok, tasks, own, "h") == []
    assert any("hash" in x for x in S.validate_allowlist("SPILLOVER", ok, tasks, own, "other"))
    bad = {**ok, "roots": ["r0", "r1", "r2"]}
    own2 = {**own, "owner": {**own["owner"], "r2": "GVLAB"}}
    assert any("PACKED / PARTIAL" in x for x in S.validate_allowlist("SPILLOVER", bad, tasks, own2, "h"))
    assert any("twice" in x for x in S.validate_allowlist("SPILLOVER", {**ok, "roots": ["r0", "r0", "r1"]}, tasks, own, "h"))
    drift = json.loads(json.dumps(ok)); drift["trial_ids"]["r0"] = ["r0:t0", "r0:t9"]
    assert any("trial ids" in x for x in S.validate_allowlist("SPILLOVER", drift, tasks, own, "h"))
    assert any("TRAIN" in x for x in S.validate_allowlist("CANARY", {**ok, "roots": ["r0", "r9"]}, tasks))
    assert S.validate_allowlist("CANARY", {**ok, "roots": ["r0", "r1"]}, tasks) == []


def test_canary_verdict_is_exact_or_hard_block():
    lap, labs = {"a": "x", "b": "y"}, {"a": (1, None), "b": (0, "NO_PLAN_FOUND")}
    srv_labs = {"a": [1, None], "b": [0, "NO_PLAN_FOUND"]}
    assert S.compare_core(lap, labs, dict(lap), srv_labs, {"failed": 0})["verdict"] == "PASS_EXACT"
    for srv, sl, ver in (({"a": "x", "b": "z"}, srv_labs, {"failed": 0}), (dict(lap), {**srv_labs, "b": [1, None]}, {"failed": 0}),
                         ({"a": "x"}, srv_labs, {"failed": 0}), (dict(lap), srv_labs, {"failed": 3})):
        assert S.compare_core(lap, labs, srv, sl, ver)["verdict"] == "HARD_BLOCK"


def test_tick_lock_single_winner(tmp_path):
    code = ("import sys,time;sys.path.insert(0,sys.argv[2]);import spill2d as S\nwhile time.time()<float(sys.argv[3]):pass\n"
            "l=S.acquire_tick_lock(sys.argv[1]);print('WON' if l else 'LOST',flush=True);time.sleep(2)")
    go = time.time() + 2.0
    ps = [subprocess.Popen([sys.executable, "-c", code, str(tmp_path / "tick.lock"), str(HERE), str(go)], stdout=subprocess.PIPE, text=True) for _ in range(6)]
    outs = [p.communicate(timeout=60)[0].strip() for p in ps]
    assert outs.count("WON") == 1 and outs.count("LOST") == 5, outs


@pytest.fixture(scope="module")
def plan():
    LP, R, K, T = S.code2d(T_CODE)
    return S.q2_tasks(T_LAUNCH, LP, R), T


def test_partition_exact_disjoint_union_no_overwrite(tmp_path, plan):
    tasks, T = plan
    run = tmp_path / "run"
    packed = sorted(f.stem for f in (T_PACKED / "records").glob("*.json"))[:3]
    assert len(packed) == 3, "needs >= 3 packed roots in the original run"
    for d in ("records", "modes", "witnesses"):
        (run / d).mkdir(parents=True)
    for r in packed:
        for d, e in (("records", "json"), ("modes", "npz"), ("witnesses", "npz")):
            if (T_PACKED / d / f"{r}.{e}").exists():
                shutil.copy2(T_PACKED / d / f"{r}.{e}", run / d / f"{r}.{e}")
    partial = sorted(r for r in tasks if r not in packed)[:2]
    for r in partial:
        (run / "parts" / r).mkdir(parents=True)
        tid = tasks[r][1][0]["trial_id"]
        (run / "parts" / r / f"{tid}.json").write_text(json.dumps({"trial_id": tid, "completed": True, "eligible": True}))
    man = S.partition(0.4, run=run, L=T_LAUNCH, out=tmp_path / "handoff")
    own, cls = man["owner"], man["class"]
    assert len(own) == S.EXPECT["roots"] and man["invariants"]["tasks"] == S.EXPECT["tasks"] and man["invariants"]["teacher_queries"] == S.EXPECT["teacher_queries"]
    assert all(own[r] == "ORIGINAL" for r in packed) and all(own[r] == "LAPTOP" for r in partial)
    assert all(cls[r] == "UNTOUCHED" for r, o in own.items() if o == "GVLAB")
    g = json.loads((tmp_path / "handoff/allowlist_gvlab.json").read_text()); lpt = json.loads((tmp_path / "handoff/allowlist_laptop.json").read_text())
    assert not set(g["roots"]) & set(lpt["roots"]) and len(g["roots"]) + len(lpt["roots"]) + len(packed) == S.EXPECT["roots"]
    ids = [i for r in tasks for i in man["trial_ids"][r]]
    assert len(ids) == len(set(ids)) == S.EXPECT["tasks"]
    est = man["estimated_laptop_cpu_s_by_owner"]
    assert 0.3 < est["GVLAB"] / (est["GVLAB"] + est["LAPTOP"]) < 0.45
    with pytest.raises(SystemExit, match="already exists"):
        S.partition(0.4, run=run, L=T_LAUNCH, out=tmp_path / "handoff")
    bad = tmp_path / "bad"; shutil.copytree(run, bad)
    h = json.loads((bad / "records" / f"{packed[0]}.json").read_text()); h["queries"] = h["queries"][1:]
    (bad / "records" / f"{packed[0]}.json").write_text(json.dumps(h))
    with pytest.raises(SystemExit, match="classification failures"):
        S.partition(0.4, run=bad, L=T_LAUNCH, out=tmp_path / "handoff_bad")


def test_copy_partials_verifies_hashes_and_never_touches_the_original(tmp_path):
    orig, rem = tmp_path / "orig", tmp_path / "rem"
    (orig / "parts/rA").mkdir(parents=True)
    (orig / "parts/rA/t1.json").write_text('{"x": 1}'); (orig / "parts/rA/w_1.npz").write_bytes(b"w")
    snap = {"parts/rA/t1.json": S.sha(orig / "parts/rA/t1.json"), "parts/rA/w_1.npz": S.sha(orig / "parts/rA/w_1.npz")}
    own = {"class": {"rA": "PARTIAL", "rB": "UNTOUCHED"}, "snapshot_sha256": {"rA": snap}}
    before = {p.name: S.sha(p) for p in (orig / "parts/rA").iterdir()}
    assert S.copy_partials(own, {"roots": ["rA", "rB"]}, orig, rem) == 1
    assert {p.name: S.sha(p) for p in (rem / "parts/rA").iterdir()} == before == {p.name: S.sha(p) for p in (orig / "parts/rA").iterdir()}
    assert S.copy_partials(own, {"roots": ["rA"]}, orig, rem) == 0          # idempotent
    shutil.rmtree(rem); (orig / "parts/rA/t1.json").write_text('{"x": 2}')
    with pytest.raises(SystemExit, match="changed since the handoff snapshot"):
        S.copy_partials(own, {"roots": ["rA"]}, orig, rem)


def test_merge_never_overwrites(tmp_path, monkeypatch):
    monkeypatch.setattr(S, "MERGED", tmp_path / "exists"); (tmp_path / "exists").mkdir()
    (tmp_path / "handoff").mkdir(); (tmp_path / "handoff/HANDOFF_ROOT_OWNER_MANIFEST.json").write_text(json.dumps({"owner": {}, "trial_ids": {}}))
    with pytest.raises(SystemExit, match="never overwrite"):
        S.merge(tmp_path / "srv", handoff=tmp_path / "handoff")


def test_canary_prep_verifies_bytes_against_the_laptop_manifest_and_skips_mismatches(tmp_path, plan):
    tasks, T = plan
    back = tmp_path / "backup"
    cands = S.canary_candidates(tasks, sorted(f.stem for f in (T_PACKED / "records").glob("*.json")))
    assert len(cands) >= 3, "needs >= 3 packed TRAIN roots with >= 3 blocks"
    man = {}
    for r in cands[:3]:
        for d, e in (("records", "json"), ("modes", "npz"), ("witnesses", "npz")):
            if (T_PACKED / d / f"{r}.{e}").exists():
                (back / d).mkdir(parents=True, exist_ok=True); shutil.copy2(T_PACKED / d / f"{r}.{e}", back / d / f"{r}.{e}")
                man[f"{d}/{r}.{e}"] = S.sha(T_PACKED / d / f"{r}.{e}")
    (back / f"records/{cands[0]}.json").write_text((back / f"records/{cands[0]}.json").read_text() + " ")   # corrupt one backup copy
    mf = tmp_path / "man.json"; mf.write_text(json.dumps(man))
    with pytest.raises(SystemExit, match="only 2 verifiable"):
        S.canary_prep(mf, L=T_LAUNCH, backup=back, ref=tmp_path / "ref0", allow_out=tmp_path / "a0.json", n=3)
    doc = S.canary_prep(mf, L=T_LAUNCH, backup=back, ref=tmp_path / "ref", allow_out=tmp_path / "a.json", n=2)
    assert doc["roots"] == sorted(cands[1:3]) and doc["skipped"][0][0] == cands[0]
    mine = {rel: h for rel, h in man.items() if Path(rel).stem in cands[1:3]}
    assert mine and all(S.sha(tmp_path / "ref" / rel) == h for rel, h in mine.items()) and not (tmp_path / "ref/records" / f"{cands[0]}.json").exists()
    assert doc["trial_ids"] == {r: [t["trial_id"] for t in tasks[r][1]] for r in cands[1:3]}
    with pytest.raises(SystemExit, match="never overwrite"):
        S.canary_prep(mf, L=T_LAUNCH, backup=back, ref=tmp_path / "ref", allow_out=tmp_path / "a.json", n=2)


def _queue(tmp_path, monkeypatch, state, ssh_out="", backup_state="VERIFIED"):
    monkeypatch.setattr(S, "SPILL", tmp_path); monkeypatch.setattr(S, "STATE", tmp_path / "state/Q.json")
    monkeypatch.setattr(S, "LAP_BACKUP", tmp_path / "relay"); (tmp_path / "relay").mkdir()
    (tmp_path / "relay/backup2d_last.json").write_text(json.dumps({"state": backup_state}))
    (tmp_path / "relay/backup2d_source_manifest.json").write_text("{}")
    (tmp_path / "state").mkdir(); (tmp_path / "state/Q.json").write_text(json.dumps({"state": state, "history": []}))
    calls = []
    monkeypatch.setattr(S, "ssh", lambda cmd, timeout=600: calls.append(cmd) or types.SimpleNamespace(stdout=ssh_out, stderr="", returncode=0))
    monkeypatch.setattr(S, "inhibitor_ok", lambda: True)
    monkeypatch.setattr(S, "code2d", lambda *_: (_ for _ in ()).throw(AssertionError("laptop tick imported the 2D code before the drain")))
    real = subprocess.run
    monkeypatch.setattr(S.subprocess, "run", lambda a, **k: calls.append(a) or types.SimpleNamespace(returncode=0, stdout="", stderr="")
                        if a and a[0] == "rsync" else real(a, **k))
    return calls


def test_tick_waits_for_3d_and_never_loads_2d_code_before_the_drain(tmp_path, monkeypatch):
    gate = json.dumps({"pass": False, "reasons": ["3D collection incomplete (400/1466 units, last end complete=False)"]})
    calls = _queue(tmp_path, monkeypatch, "WAIT_3D", ssh_out=gate + "\n")
    S.tick(); q = json.loads((tmp_path / "state/Q.json").read_text())
    assert q["state"] == "WAIT_3D" and q["gate3d_collection"]["pass"] is False and "taskset -c 12,13" in calls[0] and not q.get("last_error")
    (tmp_path / "b").mkdir()
    calls = _queue(tmp_path / "b", monkeypatch, "VERIFY_2D_SERVER")
    S.tick(); q = json.loads((tmp_path / "b/state/Q.json").read_text())
    assert q["state"] == "VERIFY_2D_SERVER" and q.get("canary_started") and not q.get("last_error")
    assert any("tmux new-session -d -s q2_canary" in c and "canary-prep" in c and "server-run --scope CANARY" in c for c in calls if isinstance(c, str))


def test_canary_waits_while_the_laptop_backup_is_unverified(tmp_path, monkeypatch):
    _queue(tmp_path, monkeypatch, "VERIFY_2D_SERVER", backup_state="BACKUP_FAILED_OR_PARTIAL")
    S.tick(); q = json.loads((tmp_path / "state/Q.json").read_text())
    assert q["state"] == "VERIFY_2D_SERVER" and not q.get("canary_started") and "not verified" in q["last_error"]


def test_drain_only_after_a_passing_canary_report(tmp_path, monkeypatch):
    _queue(tmp_path, monkeypatch, "VERIFY_2D_SERVER")
    (tmp_path / "PORTABILITY_CANARY_REPORT.json").write_text(json.dumps({"pass": False, "verdict": "HARD_BLOCK"}))
    S.tick(); assert json.loads((tmp_path / "state/Q.json").read_text())["state"] == "BLOCKED_CANARY"
    S.tick(); assert json.loads((tmp_path / "state/Q.json").read_text())["state"] == "BLOCKED_CANARY"       # terminal: never drains


def test_a_safety_stopped_original_is_never_handed_off_without_acknowledgement(tmp_path, monkeypatch):
    _queue(tmp_path, monkeypatch, "DRAIN_2D")
    orig = tmp_path / "orig"; orig.mkdir(); (orig / "DRAIN").write_text("x")
    (orig / "status.json").write_text(json.dumps({"final": True, "stop": "SWAP"}))
    monkeypatch.setattr(S, "ORIG", orig); monkeypatch.setattr(S, "coordinator_pid", lambda run: None)
    S.tick(); q = json.loads((tmp_path / "state/Q.json").read_text())
    assert q["state"] == "DRAIN_2D" and "ORIGINAL_2D_SAFETY_STOP:SWAP" in q["attention"][0]
    (tmp_path / "state/ACK_ORIGINAL_STOP").write_text("ok")
    S.tick(); q = json.loads((tmp_path / "state/Q.json").read_text())
    assert q["state"] == "PARTITION" and q["original_end"] == "SAFETY_STOP_SWAP_ACKNOWLEDGED"
    (orig / "status.json").write_text(json.dumps({"final": True, "stop": None})); (tmp_path / "state/ACK_ORIGINAL_STOP").unlink()
    (tmp_path / "state/Q.json").write_text(json.dumps({"state": "DRAIN_2D", "history": []}))
    S.tick(); assert json.loads((tmp_path / "state/Q.json").read_text())["original_end"] == "DRAINED"


def test_cache_hygiene_reads_and_deletes_nothing(tmp_path):
    (tmp_path / "d/e").mkdir(parents=True); (tmp_path / "d/e/f.bin").write_bytes(b"x" * 4096)
    h = S.sha(tmp_path / "d/e/f.bin")
    assert S.cache_hygiene([tmp_path / "d", tmp_path / "missing"]) == 1 and S.sha(tmp_path / "d/e/f.bin") == h


def test_pause_file_holds_the_queue_and_retry_only_from_blocked(tmp_path, monkeypatch):
    calls = _queue(tmp_path, monkeypatch, "WAIT_3D")
    (tmp_path / "state/PAUSE_QUEUE").write_text("hold")
    S.tick(); q = json.loads((tmp_path / "state/Q.json").read_text())
    assert q["state"] == "WAIT_3D" and q["paused"] is True and not calls
    with pytest.raises(SystemExit, match="not a retryable"):
        S.retry()
    (tmp_path / "state/Q.json").write_text(json.dumps({"state": "BLOCKED_CANARY", "canary_started": "x", "history": []}))
    (tmp_path / "PORTABILITY_CANARY_REPORT.json").write_text('{"pass": false}')
    S.retry(); q = json.loads((tmp_path / "state/Q.json").read_text())
    assert q["state"] == "VERIFY_2D_SERVER" and "canary_started" not in q and list((tmp_path / "reports").glob("canary_attempt_*.json"))
    assert any("mv $d $d.attempt_" in c for c in calls if isinstance(c, str))
