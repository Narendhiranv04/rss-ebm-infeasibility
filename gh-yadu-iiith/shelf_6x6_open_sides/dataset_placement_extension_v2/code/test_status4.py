"""Tests of the status / backup service (status4.py): destination re-verification, source-mismatch refusal, failed or partial mirror and 2D
remote copies never reported as verified, explicit health flags.   pytest -q code/test_status4.py"""
import hashlib, json, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import status4 as S  # noqa: E402


def h(b):
    return hashlib.sha256(b).hexdigest()


def fake_run(tmp, n=3):
    run = tmp / "run"; (run / "units").mkdir(parents=True)
    led = []
    for i in range(n):
        j, z = f'{{"unit": "u{i}"}}'.encode(), bytes([i]) * 1000
        (run / "units" / f"u{i}.json").write_bytes(j); (run / "units" / f"u{i}.npz").write_bytes(z)
        led.append({"event": "unit_done", "unit": f"u{i}", "json_sha256": h(j), "npz_sha256": h(z), "queries": 1, "t": time.time()})
    (run / "ledger.jsonl").write_text("\n".join(map(json.dumps, led)) + "\n")
    (run / "run_manifest.json").write_text("{}")
    return run


def test_backup_recopies_deleted_and_corrupt_destination_files(tmp_path):
    run, dst = fake_run(tmp_path), tmp_path / "dst"
    s = S.backup_units(run, dst)
    assert s["units_verified_in_backup"] == 3 and s["copied_now"] == 6
    (dst / "units/u0.npz").unlink()                               # deleted on the backup disk
    (dst / "units/u1.json").write_text("corrupt")                  # corrupted on the backup disk; the index still lists the right hash
    s = S.backup_units(run, dst)
    assert s["recopied_missing_or_corrupt"] == 1 and s["copied_now"] == 2 and s["units_verified_in_backup"] == 3
    assert S.sha(dst / "units/u0.npz") == S.sha(run / "units/u0.npz") and S.sha(dst / "units/u1.json") == S.sha(run / "units/u1.json")


def test_backup_never_copies_a_unit_that_does_not_match_its_ledger(tmp_path):
    run, dst = fake_run(tmp_path), tmp_path / "dst"
    (run / "units/u2.npz").write_bytes(b"tampered")
    s = S.backup_units(run, dst)
    assert "u2.npz" in s["skipped_source_mismatch"] and s["units_verified_in_backup"] == 2 and not (dst / "units/u2.npz").exists()


def _patched(monkeypatch, tmp_path, codes):
    monkeypatch.setattr(S, "MIRROR3", tmp_path / "mirror" / "run_q2_parity")
    real = subprocess.run

    def run(args, *a, **k):
        tool = args[0]
        if tool in codes:
            return subprocess.CompletedProcess(args, codes[tool], "", "")
        return real(args, *a, **k)
    monkeypatch.setattr(S.subprocess, "run", run)


def test_failed_mirror_rsync_is_never_verified(monkeypatch, tmp_path):
    _patched(monkeypatch, tmp_path, {"ssh": 0, "rsync": 23})
    (tmp_path / "mirror/run_q2_parity").mkdir(parents=True)
    (tmp_path / "mirror/run_q2_parity/backup_index.json").write_text(json.dumps({"units": {}}))   # stale but internally consistent
    assert S.mirror_3d()["state"] == "MIRROR_FAILED"


def test_network_failure_is_flagged_not_nothing_to_mirror(monkeypatch, tmp_path):
    _patched(monkeypatch, tmp_path, {"ssh": 255})
    assert S.mirror_3d()["state"] == "MIRROR_FAILED"


def test_2d_backup_detects_remote_corruption_and_requests_recopy(monkeypatch, tmp_path):
    run = tmp_path / "full_2d_v2"; (run / "records").mkdir(parents=True); (run / "witnesses").mkdir()
    (run / "records/r1.json").write_text('{"queries": []}'); (run / "witnesses/r1.npz").write_bytes(b"w" * 100)
    monkeypatch.setattr(S, "RUN2D", run); monkeypatch.setattr(S, "RELAY", tmp_path / "relay")
    calls = []

    def run_(args, *a, **k):
        calls.append(args)
        if args[0] == "ssh" and "sha256sum" in args[-1]:
            good = S.sha(run / "records/r1.json")
            return subprocess.CompletedProcess(args, 0, f"{good}  records/r1.json\n{'0' * 64}  witnesses/r1.npz\n", "")
        return subprocess.CompletedProcess(args, 0, "", "")
    monkeypatch.setattr(S.subprocess, "run", run_)
    s = S.backup_2d()
    assert s["state"] == "BACKUP_FAILED_OR_PARTIAL" and s["files_failing"] == 1 and s["roots_verified_on_destination"] == 0
    assert any(c[0] == "ssh" and "rm -f" in c[-1] and "witnesses/r1.npz" in c[-1] for c in calls)


def test_health_flags_are_explicit():
    now = time.time()
    s3 = {"t": now - 3000, "3D_V4_Q2_PARITY": {"state": "STOPPED_INCOMPLETE", "errors": ["temp"], "units_done": 100, "quarantine_dirs": 1},
          "backup": {"t": now - 3000, "units_verified_in_backup": 10}}
    s2 = {"state": "RUNNING", "status_age_s": 5000, "roots_packed": 200}
    h_ = S.health(s3, s2, {"state": "BACKUP_FAILED_OR_PARTIAL", "roots_verified_on_destination": 0}, {"state": "MIRROR_FAILED"},
                  {"consecutive_push_failures": 2})
    f = " ".join(h_["flags"])
    for k in ("STALE_3D", "STOPPED_3D", "QUARANTINE_3D", "BACKUP_LAGGING_3D", "MIRROR_FAILED_3D", "STALE_2D", "BACKUP_LAGGING_2D", "PUSH_FAILED"):
        assert k in f, k
    assert h_["overall"] == "ATTENTION"
    ok = S.health({"t": now, "3D_V4_Q2_PARITY": {"state": "RUNNING", "units_done": 5, "monitor_age_s": 3},
                   "backup": {"t": now, "units_verified_in_backup": 5}}, {"state": "NOT_STARTED"}, {"state": "NOTHING_TO_BACK_UP"},
                  {"state": "VERIFIED", "t": now}, {})
    assert ok["overall"] == "HEALTHY" and not ok["flags"]
    lag = S.health({"t": now, "3D_V4_Q2_PARITY": {"state": "RUNNING", "units_done": 5, "monitor_age_s": 3},
                    "backup": {"t": now, "units_verified_in_backup": 5}}, {"state": "NOT_STARTED"}, {"state": "NOTHING_TO_BACK_UP"},
                   {"state": "VERIFIED", "t": now - 4000}, {})
    assert any(f.startswith("MIRROR_LAGGING_3D") for f in lag["flags"])
    run2 = S.health(None, {"state": "RUNNING", "status_age_s": 10, "roots_packed": 5}, {"state": "VERIFIED", "t": now - 4000,
                    "roots_verified_on_destination": 5}, {"state": "NOTHING_TO_MIRROR"}, {})
    assert any(f.startswith("STALE_3D") for f in run2["flags"]) and any(f.startswith("BACKUP_LAGGING_2D") for f in run2["flags"])


def test_2d_states_drained_for_handoff_and_local_spillover_backup(monkeypatch, tmp_path):
    run = tmp_path / "run"; (run / "records").mkdir(parents=True)
    (run / "run_manifest.json").write_text(json.dumps({"roots": 2, "queries": 4}))
    (run / "records/r1.json").write_text(json.dumps({"queries": [{"eligible": True, "label": 1, "witness_refs": ["w"]}, {"eligible": False}]}))
    (run / "status.json").write_text(json.dumps({"final": True, "stop": None}))
    monkeypatch.setattr(S, "RELAY", tmp_path / "relay")
    assert S.state_2d(run, "x", "J")["state"] == "STOPPED_INCOMPLETE"             # final, not all roots, no drain: an unexplained stop
    (run / "DRAIN").write_text("handoff")
    s = S.state_2d(run, "x", "J")
    assert s["state"] == "DRAINED_FOR_HANDOFF" and s["job"] == "J" and s["teacher_queries_packed"] == 1
    (run / "status.json").write_text(json.dumps({"final": True, "stop": "SWAP"}))
    assert S.state_2d(run, "x", "J")["state"] == "STOPPED_INCOMPLETE"             # a safety stop stays a stop even with DRAIN present
    monkeypatch.setattr(S, "HANDOFF2D", tmp_path / "owner_map.json")
    assert S.state_2d(run, "full_2d_v2", "J")["state"] == "STOPPED_INCOMPLETE"    # ... until the owner map is frozen (the stop was ACKed)
    S.HANDOFF2D.write_text("{}")
    assert S.state_2d(run, "full_2d_v2", "J")["state"] == "HANDED_OFF"
    assert not any(f.startswith("STOPPED_2D") for f in S.health(None, S.state_2d(run, "full_2d_v2", "J"), {}, {}, {})["flags"])
    # a scoped coordinator whose long path pushes "laptop-run" past the 160-char display cut is still RUNNING (2026-11-08 false STOPPED flags)
    long = "/x" * 90 + "/venv/bin/python /very/long/q2_spillover/spill2d.py laptop-run"
    p = subprocess.Popen(["bash", "-c", f"exec -a '{long}' sleep 30"])
    try:
        time.sleep(0.4)
        s = S.state_2d(run, "remainder_laptop", "J")
        assert s["state"] == "RUNNING" and [c["pid"] for c in s["coordinator_processes"]] == [p.pid], s["coordinator_processes"]
    finally:
        p.kill(); p.wait()
    dst = tmp_path / "dst"
    b = S.backup_packed_local(run, dst)
    assert b["state"] == "VERIFIED" and b["roots_verified"] == 1 and S.sha(dst / "records/r1.json") == S.sha(run / "records/r1.json")
    (dst / "records/r1.json").write_text("corrupt")
    assert S.backup_packed_local(run, dst)["state"] == "VERIFIED" and S.sha(dst / "records/r1.json") == S.sha(run / "records/r1.json")
