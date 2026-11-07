"""Focused regression tests for the SHELF66V3 placement sidecar V2 (run on gvlab, one CPU: pytest -q code/test_mock3d_v2.py).
Physical fixtures: the V1 frozen fixtures (training roots outside both mock rosters, mock_v1/configs/fixtures.json) and the V2 natural
branch fixtures (mock_v2/configs/fixtures_v2.json, frozen before the main V2 runs).  Synthetic (injected) tests say so in their name."""
import gzip, inspect, json, os, signal, subprocess, sys, time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement3d as P  # noqa: E402
import plan3d as L  # noqa: E402
import policy3d as Q  # noqa: E402
import replay3d as R  # noqa: E402
import run3d  # noqa: E402

ROOT = P.HERE.parent
V1 = ROOT.parent / "dataset_placement_extension_v1"
FX = json.loads((V1 / "mock_v1/configs/fixtures.json").read_text())["fixtures"]
FX2P = ROOT / "mock_v2/configs/fixtures_v2.json"
ROSTER = [json.loads(x) for x in gzip.open(ROOT / "mock_v2/roster_v2.jsonl.gz", "rt")]
PY = sys.executable
_cache = {}


def scene_of(f, cache=None):
    raw = json.loads((P.SRC / "raw/roots" / f"{f['root_id']}.json").read_text())
    return raw, P.Scene(raw["objects"], f["cells"], cache)


def fx(kind, policy="reference_v2"):
    if (kind, policy) not in _cache:
        f = FX[kind]
        raw, sc = scene_of(f)
        rec, ws = Q.evaluate(sc, f["object"], tuple(f["dest"]), (Q.POLICIES | Q.FIXED)[policy], early_stop=policy not in Q.REFERENCE)
        _cache[(kind, policy)] = (f, raw, sc, rec, ws)
    return _cache[(kind, policy)]


def test_protocol_identity():
    assert P.G["protocol"] == "XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022" and P.G["config_hash"] == "8d521b1f6d23db23"
    assert P.S.NO_OVERPASS and P.S.ARM_CEILING and not P.S.RETURN_TO_READY and P.S.GRASP_AT == "middle,top"
    assert P.TEACHER == "SHELF66V3_PLACE_V2" and "policy3d.py" in P.protocol()[1]["sources"] and "branch_semantics" in P.protocol()[1]
    f, raw, sc, rec, ws = fx("obstructed_carry_certified")
    ps = P.poses(sc, f["object"])
    assert len(ps) == 32 and len({g["azimuth_deg"] for g in ps}) == 16 and {g["grasp_level"] for g in ps} == {"middle", "top"}
    r = Q.ranked(ps)
    assert sorted(g["id"] for g in r) == list(range(32)) and [g["grasp_level"] for g in r[:4]] == ["middle", "top", "middle", "top"]


def test_finger_preshaped_in_every_object_check(monkeypatch):
    f, raw, sc, rec, ws = fx("target_goal_destination")
    seen, orig = [], P.S.XArm7TableEnv._object_hits

    def spy(self, q, *a, **kw):
        seen.append(self._check_width)
        out = orig(self, q, *a, **kw)
        if self._check_width is not None:
            assert np.allclose(self.ik_data.qpos[self.grip_qadr], self._finger_q(self._check_width))
        return out
    monkeypatch.setattr(P.S.XArm7TableEnv, "_object_hits", spy)
    g = Q.ranked(P.poses(sc, f["object"]))[0]
    st, cands, elig, _, _ = P.screen(sc, f["object"], tuple(f["dest"]), g)
    if elig:
        P.plan_branch(sc, f["object"], tuple(f["dest"]), g, cands[elig[0]], 500, P.route_state())
    assert seen and all(x == pytest.approx(g["open_width"]) for x in seen)
    assert g["open_width"] < float(sc.e._grip_cal[0][-1]) - 0.02


def test_static_contrasts_are_unlabeled_and_never_planned():
    st = [q for q in ROSTER if q["static"]]
    assert st and all(Q.static_reason({k: tuple(v) for k, v in q["cells"].items()}, q["object"], tuple(q["dest"])) == q["static"] for q in st)
    units, _ = run3d.load_units(ROOT / "mock_v2/roster_v2.jsonl.gz", "all", set())
    assert not any(q["static"] for v in units.values() for q in v)


def test_route_is_branch_independent():
    """The carry route planned from one branch's context equals the route planned from another's (fresh route state each time)."""
    f, raw, sc, rec, ws = fx("vacated_after_edit")
    m = next(m for m in rec["modes"] if m["screen"] == "VALID" and m["n_eligible"] >= 2)
    g = Q.ranked(P.poses(sc, f["object"]))[m["rank"]]
    _, cands, elig, _, _ = P.screen(sc, f["object"], tuple(f["dest"]), g)
    routes = []
    for b in (elig[1], elig[0]):                               # later branch first, then the first one
        rt = P.route_state()
        pl = P.plan_branch(sc, f["object"], tuple(f["dest"]), g, cands[b], 5000, rt)
        routes.append(None if rt["route"] is None else np.asarray(rt["route"]))
    assert routes[0] is not None and np.array_equal(routes[0], routes[1])


def test_nested_rrt_budget_is_an_exact_prefix():
    f, raw, sc, rec, ws = fx("budget_exhausted_negative")
    m = next(m for m in rec["modes"] if m["state"] == "UNRESOLVED")
    g = Q.ranked(P.poses(sc, f["object"]))[m["rank"]]
    b = m["tests"][-1]["route_branch"]
    _, cands, _, _, _ = P.screen(sc, f["object"], tuple(f["dest"]), g)
    big, small = P.route_state(), P.route_state()
    P.plan_branch(sc, f["object"], tuple(f["dest"]), g, cands[b], 5000, big)
    P.plan_branch(sc, f["object"], tuple(f["dest"]), g, cands[b], 500, small)
    assert not big["info"]["found"] and not small["info"]["found"] and small["info"]["work"] == big["info"]["milestones"]["500"]


def test_injected_branch0_hold_failure_is_rescued_SYNTHETIC(monkeypatch):
    """Synthetic: branch 0 of the verified pose is forced to fail its physical hold.  V1 would reject the whole pose (it certified branch 0
    only); V2 certifies the next eligible branch and still reaches a verified placement in that pose."""
    f, raw, sc, rec, ws = fx("vacated_after_edit")
    m = next(m for m in rec["modes"] if sum(bool(br["hold"] and br["hold"]["ok"]) for br in m["branches"]) >= 2
             and any(r["state"] == "VERIFIED" for br in m["branches"][1:] for r in br["results"]))
    first = next(br["b"] for br in m["branches"] if br["eligible"])
    target = Q.ranked(P.poses(sc, f["object"]))[m["rank"]]["id"]
    orig = P.hold

    def fake(sc_, name, g, q_g):
        ok, ev, before = orig(sc_, name, g, q_g)
        if g["id"] == target and np.allclose(q_g, m_q[first]):
            return False, {**ev, "injected": "branch0_hold_failure"}, before
        return ok, ev, before
    _, m_q, _, _, _ = P.screen(sc, f["object"], tuple(f["dest"]), Q.ranked(P.poses(sc, f["object"]))[m["rank"]])
    monkeypatch.setattr(P, "hold", fake)
    rec2, ws2 = Q.evaluate(sc, f["object"], tuple(f["dest"]), Q.POLICIES["reference_v2"], early_stop=False)
    m2 = next(x for x in rec2["modes"] if x["rank"] == m["rank"])
    assert m2["screen"] == "VALID" and not m2["branches"][first]["hold"]["ok"]
    assert any(r["state"] == "VERIFIED" for br in m2["branches"] if br["b"] != first for r in br["results"])
    assert rec2["accounting"]["units_first_eligible_hold_failed_later_held"] >= 1 and rec2["label"] == 1


def test_injected_exec_failure_is_rescued_SYNTHETIC(monkeypatch):
    """Synthetic: every execution of the first verified branch is forced to fail verification; V2 executes the next branch (V1 stopped)."""
    f, raw, sc, rec, ws = fx("vacated_after_edit", "adaptive_v1")
    vr, vb = rec["verified_rank"], rec["verified_branch"]
    g_id = Q.ranked(P.poses(sc, f["object"]))[vr]["id"]
    orig, state = P.verify, {"g": None}

    def fake(sc_, name, dest, rec_, ev):
        v, vi = orig(sc_, name, dest, rec_, ev)
        if state["g"] == (g_id, vb):
            return False, {**vi, "first_failed": "injected"}
        return v, vi
    orig_exec = P.execute

    def exec_spy(sc_, name, pl, before):
        state["g"] = (g_id, vb) if np.allclose(pl["q_g"], rec_q[vb]) else None
        return orig_exec(sc_, name, pl, before)
    _, rec_q, _, _, _ = P.screen(sc, f["object"], tuple(f["dest"]), Q.ranked(P.poses(sc, f["object"]))[vr])
    monkeypatch.setattr(P, "verify", fake); monkeypatch.setattr(P, "execute", exec_spy)
    rec2, ws2 = Q.evaluate(sc, f["object"], tuple(f["dest"]), Q.POLICIES["adaptive_v1"], early_stop=True)
    m2 = next(x for x in rec2["modes"] if x["rank"] == vr)
    states = m2["tests"][-1]["branch_states"]
    assert "EXEC_FAILED" in states and (rec2["label"] == 1)
    assert rec2["accounting"]["units_rescued_after_exec_failure"] >= 1 or rec2["verified_rank"] != vr


def test_natural_branch_fixtures():
    """Natural (V1 pilot data): poses V1 rejected because branch 0 failed its hold, or units V1 ended as held_state_lost, re-evaluated by V2."""
    if not FX2P.exists():
        pytest.skip("fixtures_v2.json not frozen yet")
    fx2 = json.loads(FX2P.read_text())
    assert fx2["natural_found"], "no natural fixture"
    for f in fx2["fixtures"]:
        raw, sc = scene_of(f)
        rec, _ = Q.evaluate(sc, f["object"], tuple(f["dest"]), Q.POLICIES["reference_v2"], early_stop=False)
        m = next(x for x in rec["modes"] if x["rank"] == f["rank"])
        assert f["v1"]["state"] in ("HELD_INVALID_PHYSICAL", "EXEC_FAILED")
        assert Q.canonical(m) == f["v2_mode_canonical"]
        assert any(br["hold"] and br["hold"]["ok"] for br in m["branches"] if br["b"] != f["v1"]["branch_tested"])


def test_reference_keeps_every_verified_branch_witness():
    f, raw, sc, rec, ws = fx("vacated_after_edit")
    ver = [r["witness_id"] for m in rec["modes"] for br in m["branches"] for r in br["results"] if r["state"] == "VERIFIED"]
    assert len(ver) == len(set(ver)) == len(ws) == len(rec["witnesses"]) > 1
    assert all({"witness_id", "rank", "grasp_id", "branch"} <= set(w) for w in rec["witnesses"]) and rec["accounting"]["branch_unaccounted"] == 0


def test_virtual_equals_real_on_a_fixture():
    f, raw, sc, ref, _ = fx("pick_inaccessible_held_placeable")
    for pol in ("adaptive_v1", "fixed_4x1500", "fixed_4x5000"):
        _, _, _, r, _ = fx("pick_inaccessible_held_placeable", pol)
        v = Q.virtual(ref, (Q.POLICIES | Q.FIXED)[pol])
        assert v["label"] == r["label"] and v["verified"] == [r["verified_rank"], r["verified_branch"], r["witness_id"]]
        assert v["work"] == Q.motion_work(r)


def test_reproducible_and_cache_independent():
    f, raw, sc, rec, ws = fx("pick_inaccessible_held_placeable")
    raw2, sc2 = scene_of(f, {})
    rec2, ws2 = Q.evaluate(sc2, f["object"], tuple(f["dest"]), Q.POLICIES["reference_v2"], early_stop=False)
    assert Q.canonical(rec, ("work_screen", "work")) == Q.canonical(rec2, ("work_screen", "work")) and rec["label"] == 1
    assert [w for w, _, _ in ws] == [w for w, _, _ in ws2] == [Q.wid(a) for _, a, _ in ws2]


def test_obstructed_carry_is_a_scoped_certified_negative():
    f, raw, sc, rec, ws = fx("obstructed_carry_certified")
    assert rec["label"] == 0 and rec["status"] == "NEG_CERTIFIED_GEOMETRY" and rec["accounting"]["valid"] == 0
    c = rec["certificate"]
    assert c["poses_screened"] == 32 and c["geometry_reasons_footprint_only"] and c["held_invalid_physical_all_branches_tested"]
    assert "not a global impossibility" in c["scope"]


def test_budget_exhausted_negative_has_scope():
    f, raw, sc, rec, ws = fx("budget_exhausted_negative", "adaptive_v1")
    assert rec["label"] == 0 and rec["negative_kind"] == "BOUNDED_SEARCH" and rec["accounting"]["unresolved"] >= 1
    assert rec["scope"]["final_iters"] == 5000 and rec["scope"]["units_tested"] >= 1


def test_target_goal_cell_and_vacated_cell():
    f, raw, sc, rec, ws = fx("target_goal_destination")
    assert tuple(f["dest"]) == tuple(raw["goal"]) and rec["label"] in (0, 1)
    f, raw, sc, rec, ws = fx("vacated_after_edit")
    s0 = {k: tuple(v) for k, v in raw["initial_cells"].items()}
    assert rec["label"] in (0, 1) and Q.static_reason(s0, f["object"], tuple(f["dest"])) == "OCCUPIED"


def test_release_and_settling_failures_are_caught_SYNTHETIC():
    f, raw, sc, rec, ws = fx("target_goal_destination", "adaptive_v1")
    wz = ws[0][1]
    pl = P.unpack(wz)
    g = next(x for x in P.poses(sc, f["object"]) if x["id"] == int(wz["grasp"][0]))
    early = {**pl, "segs": [pl["segs"][0], {"phase": "release", "kind": "grip", "grip": g["open_width"]}]}
    ok, hev, before = P.hold(sc, f["object"], g, pl["q_g"])
    r1, ev1 = P.execute(sc, f["object"], early, before)
    v1, vi1 = P.verify(sc, f["object"], tuple(f["dest"]), r1, ev1)
    assert not v1 and vi1["first_failed"] in ("cell", "xy")
    ok, hev, before = P.hold(sc, f["object"], g, pl["q_g"])
    r2, ev2 = P.execute(sc, f["object"], pl, before)
    e, i = sc.e, sc.e.obj_id(f["object"])
    a = e.obj_qadr[i]
    e.data.qpos[a + 3:a + 7] = [np.cos(0.3), np.sin(0.3), 0, 0]; e.data.qpos[a + 2] += 0.03; e.wait(5)
    v2, vi2 = P.verify(sc, f["object"], tuple(f["dest"]), r2, ev2)
    assert not v2 and not vi2["gates"]["upright"]


def test_witness_roundtrip_branch_provenance_and_fresh_replay(tmp_path):
    f, raw, sc, rec, ws = fx("vacated_after_edit")
    for w, a, prov in ws[:3]:
        np.savez_compressed(tmp_path / "w.npz", **a)
        z = np.load(tmp_path / "w.npz"); wz = {k: z[k] for k in z.files}
        assert Q.wid(wz) == w and int(wz["branch"][0]) == prov["branch"] and int(wz["grasp"][0]) == prov["grasp_id"]
        out = R.replay_one(raw, {"qid": f["qid"], "cells": f["cells"], "object": f["object"], "dest": f["dest"]}, wz)
        assert out["verified"] and out["grasp_pose_identical"] and out["hold_identical"] and out["q_exec_max_abs_diff"] == 0.0


def test_interruption_is_unlabeled_never_zero():
    f = FX["budget_exhausted_negative"]
    raw, sc = scene_of(f)

    def boom(*_):
        raise Q.Interrupted("test guard")
    old = signal.signal(signal.SIGPROF, boom)
    signal.setitimer(signal.ITIMER_PROF, 0.5)
    try:
        with pytest.raises(Q.Interrupted):
            Q.evaluate(sc, f["object"], tuple(f["dest"]), Q.POLICIES["reference_v2"], early_stop=False)
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0); signal.signal(signal.SIGPROF, old)
    assert '"label": None, "status": "INTERRUPTED"' in inspect.getsource(run3d.worker)


def _unit_files(tmp_path, ws, uid="u1", qid="x:q0"):
    u = tmp_path / "units"; u.mkdir(parents=True, exist_ok=True)
    arrays = {f"{w}.{k}": v for w, a, _ in ws for k, v in a.items()}
    np.savez_compressed(u / f"{uid}.npz", **arrays)
    zs = run3d.sha_file(u / f"{uid}.npz")
    doc = {"unit": uid, "records": [{"qid": qid, "witnesses": [{"witness_id": w} for w, _, _ in ws]}], "witness_index": {w: {} for w, _, _ in ws},
           "npz_sha256": zs}
    (u / f"{uid}.json").write_text(json.dumps(doc))
    return {"json_sha256": run3d.sha_file(u / f"{uid}.json"), "npz_sha256": zs}


def test_resume_rejects_missing_truncated_and_orphan_npz(tmp_path):
    f, raw, sc, rec, ws = fx("vacated_after_edit")
    led = _unit_files(tmp_path, ws[:2])
    assert run3d.validate_unit(tmp_path, "u1", ["x:q0"], led) == (True, "ok")
    z = tmp_path / "units/u1.npz"
    good = z.read_bytes()
    z.write_bytes(good[: len(good) // 2])                       # truncated archive
    assert not run3d.validate_unit(tmp_path, "u1", ["x:q0"], led)[0]
    z.unlink()                                                 # absent archive
    assert run3d.validate_unit(tmp_path, "u1", ["x:q0"], led) == (False, "npz_missing")
    z.write_bytes(good)
    (tmp_path / "units/orphan.npz").write_bytes(good)           # orphan without JSON / ledger line
    (tmp_path / "ledger.jsonl").write_text(json.dumps({"event": "unit_done", "unit": "u1", **led}) + "\n")
    ok, audit = run3d.resume_audit(tmp_path, {"u1": [{"qid": "x:q0"}]})
    assert "u1" in ok and audit["quarantined_files"] == ["orphan.npz"] and not (tmp_path / "units/orphan.npz").exists()
    (tmp_path / "units/u1.npz").write_bytes(good[:100])          # corrupt after a completed ledger line -> recomputed, quarantined
    ok, audit = run3d.resume_audit(tmp_path, {"u1": [{"qid": "x:q0"}]})
    assert "u1" not in ok and set(audit["quarantined_files"]) == {"u1.json", "u1.npz"}


def _tiny_run(tmp_path, extra_env, args=(), qids=None):
    q = qids or [next(r["qid"] for r in ROSTER if not r["static"])]
    (tmp_path / "subset.txt").write_text("\n".join(q))
    env = {**os.environ, **extra_env, "OMP_NUM_THREADS": "1"}
    cmd = [PY, str(ROOT / "code/run3d.py"), str(tmp_path / "run"), "--roster", str(ROOT / "mock_v2/roster_v2.jsonl.gz"), "--policy", "fixed_4x500",
           "--workers", "1", "--cpus", os.environ.get("RUN3D_TEST_CPU", "0"), "--unit", "query", "--subset", str(tmp_path / "subset.txt"), *args]
    return subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_temperature_sensor_loss_fails_closed(tmp_path):
    s = tmp_path / "temp1_input"
    p = _tiny_run(tmp_path, {"RUN3D_SENSOR_OVERRIDE": str(s)})   # sensor absent at start -> refuse
    out, err = p.communicate(timeout=120)
    assert p.returncode != 0 and "temperature sensor unreadable" in (out + err)
    s.write_text("45000")
    qids = [r["qid"] for r in ROSTER if not r["static"]][:12]
    p = _tiny_run(tmp_path, {"RUN3D_SENSOR_OVERRIDE": str(s)}, qids=qids)
    time.sleep(8); s.write_text("91000")                         # pause ...
    time.sleep(6); s.unlink()                                    # ... then the sensor disappears while paused: must stop, never resume
    out, err = p.communicate(timeout=120)
    end = json.loads(out.strip().splitlines()[-1])
    assert p.returncode == 3 and any("temp_sensor_lost" in x for x in end["safety_stop"])
    mon = [json.loads(x) for x in (tmp_path / "run/monitor.jsonl").read_text().splitlines()]
    assert any(m["paused"] for m in mon) and mon[-1]["sensor_error"]


def test_source_hash_drift_is_refused(tmp_path):
    pin = json.loads((ROOT / "mock_v2/configs/source_pin.json").read_text())
    r0 = next(r for r in ROSTER if not r["static"])["root_id"]
    pin["roots"][r0] = "0" * 64
    (tmp_path / "pin.json").write_text(json.dumps(pin))
    p = _tiny_run(tmp_path, {}, ["--source-pin", str(tmp_path / "pin.json"), "--dry-run"])
    out, err = p.communicate(timeout=120)
    assert p.returncode != 0 and "differ from" in (out + err)


def test_dry_run_makes_zero_planner_calls(tmp_path):
    p = _tiny_run(tmp_path, {}, ["--dry-run"])
    out, err = p.communicate(timeout=120)
    res = json.loads(out.strip().splitlines()[-1])
    assert p.returncode == 0 and res["dry_run"] and res["planner_calls"] == 0 and not (tmp_path / "run/units").exists()


def test_production_launch_is_refused_without_authorization(tmp_path):
    import guard3d
    tpl = ROOT / "mock_v2/production_policy.TEMPLATE.json"
    ok, why = guard3d.check(str(tpl), "0" * 32)
    assert not ok and any("authorization" in w for w in why) and any("token" in w for w in why)
    p = _tiny_run(tmp_path, {}, ["--production", "--authorization", str(tpl), "--confirm", "0" * 32])
    out, err = p.communicate(timeout=120)
    assert p.returncode != 0 and "REFUSED (production)" in (out + err)
    sh = ROOT / "mock_v2/launch_full_3d.sh"
    r = subprocess.run(["bash", str(sh)], capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "REFUSED" in r.stdout + r.stderr
    r = subprocess.run(["bash", str(sh), "--execute-full", "--confirm", "0" * 32, "--policy-file", str(tpl)], capture_output=True, text=True, timeout=300)
    assert r.returncode == 2 and "REFUSED" in r.stdout + r.stderr


def _fake_src(tmp_path, roots):
    (tmp_path / "raw/roots").mkdir(parents=True, exist_ok=True)
    sel = {"selected": {"0": [r for r, _ in roots]}}
    for r, (fam, cells) in roots:
        (tmp_path / f"raw/roots/{r}.json").write_text(json.dumps({"ids": {"parent_family_id": fam}, "objects": [{"id": "t0", "shape": "box",
                                                       "size": 0.04, "height": 0.1}], "initial_cells": {"t0": cells}, "goal": [0, 0]}))
    (tmp_path / "selected.json").write_text(json.dumps(sel))


def test_split_extension_inherits_and_detects_conflicts(tmp_path):
    old = {"split": {"a": "train", "b": "test"}, "groups": {"a": ["a"], "b": ["b"]}, "sha256": "x"}
    (tmp_path / "old.json").write_text(json.dumps(old))
    _fake_src(tmp_path / "s1", [("a", ("f1", [1, 1])), ("b", ("f2", [2, 2])), ("c", ("f1", [3, 3])), ("d", ("f9", [4, 4]))])
    L.extend(str(tmp_path / "s1"), str(tmp_path / "old.json"), str(tmp_path / "new.json"))
    new = json.loads((tmp_path / "new.json").read_text())
    assert new["split"]["a"] == "train" and new["split"]["b"] == "test" and new["split"]["c"] == "train" and "d" in new["split"]
    _fake_src(tmp_path / "s2", [("a", ("f1", [1, 1])), ("b", ("f2", [2, 2])), ("e", ("f1", [5, 5])), ("e2", ("f2", [5, 5]))])
    r = subprocess.run([PY, str(ROOT / "code/plan3d.py"), "extend", str(tmp_path / "s2"), str(tmp_path / "old.json"), str(tmp_path / "bad.json")],
                       capture_output=True, text=True)
    assert r.returncode == 1 and (tmp_path / "bad.SPLIT_CONFLICT.json").exists() and not (tmp_path / "bad.json").exists()


def test_two_edit_descendants_and_split_closure():
    by = {}
    for q in ROSTER:
        by.setdefault(q["root_id"], {})[q["depth"]] = q["scene_hash"]
        objs = json.loads((P.SRC / "raw/roots" / f"{q['root_id']}.json").read_text())["objects"]
        assert L.scene_hash(objs, {k: tuple(v) for k, v in q["cells"].items()}) == q["scene_hash"]
    for q in ROSTER:
        if q["depth"] == 2:
            assert q["parent_scene"] == by[q["root_id"]][1] and len(q["edits"]) == 2
    assert all(q["split"] == "train" for q in ROSTER)
    roots = {q["root_id"] for q in ROSTER}
    assert len(roots) <= 48 and sum(not q["static"] for q in ROSTER) <= 500
    excl = set(json.loads((ROOT / "mock_v2/excluded_v1_touched_roots.json").read_text()))
    assert not ({q["root_id"] for q in ROSTER if q["cohort"] == "holdout"} & excl)


def test_roster_uses_no_protected_metadata():
    src = "".join(inspect.getsource(fn) for fn in (L.roster_for, L._edit, L.pick_roots, L.load_selected, L.roster_holdout, L.pick_holdout, L.hardness))
    for bad in ("roles", "parking", "intended", "greedy", "labels", "\"spec\"", "verified_K", "blocker"):
        assert bad not in src, bad
