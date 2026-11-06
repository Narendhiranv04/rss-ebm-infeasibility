"""Placement smoke sidecar V2: deterministic work budgets, held-state semantics, binary labels, accounting, witness replay, resume.
Fixture FIXTURE_PICK_BLOCKED_HELD_VALID (unit test only, not smoke data): root 2d-plus2d-srv-w0408-s006 object o08 -> cell 55, an
object no declared mode can pick from the root scene (pick access 0) that is still placeable once held."""
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pytest
import placement_smoke as P
import placement_smoke_check as C

V2 = Path(__file__).resolve().parents[1] / "smoke_repro_v2"
FIX = {"root": "2d-plus2d-srv-w0408-s006", "cls": "3", "prefix": [], "object": "o08", "dest": 55, "dest_tag": "FIXTURE", "trial_id": "fixture-o08-55"}


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    out = tmp_path_factory.mktemp("q")
    ref = json.loads((V2 / "roster.json").read_text())[0]
    return out, P.trial(FIX, out), P.trial(ref, out)


class _Space:                                            # every extension fails: only the attempt counter can stop the loop
    rng = np.random.default_rng(0)
    sample, near, steer, edge = (lambda s: np.zeros(3)), (lambda s, n, z: 0), (lambda s, a, b: b), (lambda s, a, b: None)


def test_failed_extensions_consume_attempts_and_stop_deterministically(monkeypatch):
    monkeypatch.setitem(P.WORK, "rrt_attempts", 7)
    ck, w = type("Ck", (), {"n_evals": 0})(), {"attempts": 0, "nodes": 0, "search_evals": 0, "shortcut_evals": 0, "stop": None}
    assert P.det_rrt(_Space(), np.zeros(3), np.ones(3), ck, w) is None
    assert w["attempts"] == 7 and w["nodes"] == 2 and w["stop"] == "ATTEMPTS"


def test_budget_cutoff_is_not_a_certificate():
    m = {"n_mode_witnesses": 0, "hard_screened": 2000, "total": 2304, "valid_not_planned": 296}
    lab = P.decide([], {}, m, [{"work": {"stop": "ATTEMPTS"}}])
    assert lab["label"] == 0 and lab["negative_kind"] == "WORK_BUDGET_EXHAUSTED" and not lab["is_certificate"]


def _plan(_=None):
    G = P.A.env()
    robot, p = G["robot"], P.sk.params(G["cfg"])
    s = P.scene_from_dict(P.load_root(FIX["root"])["scene"])
    obj = s.get("o08")
    a = next(c for c in P.held_modes(robot, s, obj, p) if c.stage == "OK")
    dxy = np.asarray(s.lattice.center(55))
    ck = P.mo.checkers(robot, s, "o08", dxy, a.local, p["delta"], p["eps"])
    b = next(c for c in P.sk.screen_place(robot, ck, dxy, a.local, a.gap, p) if c.stage == "OK")
    st, path, method, work = P.plan_carry(ck[("held", False)], a.q_end, b.q_end, a.gap, G["cfg"], p, "unit/det")
    return st, method, work, [q.tolist() for q in path]


def test_planning_is_identical_across_processes_and_ignores_the_clock(monkeypatch):
    with mp.get_context("fork").Pool(2) as pool:
        par = pool.map(_plan, [0, 1])
    here = _plan()
    monkeypatch.setattr(P.pn.time, "time", lambda: 1e18)          # a clock past any deadline cannot cut search or shortcutting
    late = _plan()
    assert par[0] == par[1] == here == late


def test_pick_blocked_object_is_still_placeable_once_held(runs):
    _, fx, _ = runs
    assert fx["pick_access_modes_ok"] == 0 and fx["held_modes_valid"] > 0
    assert fx["completed"] and fx["label"] == 1 and fx["witness_refs"]


def test_witness_is_placement_only_and_replays_phase_by_phase(runs):
    out, fx, _ = runs
    f = np.load(out / "trajectories" / f"{fx['witness_refs'][0]}.npz")
    core = {k: f[k] for k in C.CORE}
    s = P.scene_from_dict(P.load_root(FIX["root"])["scene"])
    G = P.A.env()
    robot, p, dxy = G["robot"], P.sk.params(G["cfg"]), np.asarray(s.lattice.center(55))
    assert [str(x) for x in f["phase_run_name"]] == ["carry", "release", "retreat"] and f["gap"][0] < robot.open_gap
    assert P.verify_witness(robot, s, "o08", core, dxy, p)[0]
    assert not P.verify_witness(robot, s, "o08", {**core, "held_local": core["held_local"] + [0.004, 0.0]}, dxy, p)[0]
    k = [str(x) for x in core["phase"]].index("release")
    assert not P.verify_witness(robot, s, "o08", {**core, "phase": np.array(["carry"] * len(core["phase"]))}, dxy, p)[0] and k > 0


def test_emergency_interruption_gives_no_label(tmp_path, monkeypatch):
    monkeypatch.setitem(P.SAFETY, "query_emergency_s", 1)
    r = P.trial({**FIX, "trial_id": "fixture-interrupt"}, tmp_path)
    assert r["completed"] is False and r["label"] is None and r["safety_interruption"] == "QUERY_EMERGENCY_WALL"
    assert not P.final(tmp_path / "parts" / "fixture-interrupt.json")       # resume re-runs it


def test_accounting_reuse_and_idempotent_resume(runs):
    out, _, ref = runs
    m = ref["mode_summary"]
    assert m["total"] == 2304 == m["hard_screened"] + m["motion_attempted"] + m["reused_reverified"] + m["not_motion_attempted"]
    assert m["reused_reverified"] == 1 and ref["reuse_check"]["reverified"] and ref["attempts"][0]["evidence_source"] == "REUSED_REVERIFIED"
    before = {f: f.read_bytes() for f in out.rglob("*") if f.is_file()}
    assert P.trial(json.loads((V2 / "roster.json").read_text())[0], out) == ref
    assert before == {f: f.read_bytes() for f in out.rglob("*") if f.is_file()}


def test_prefix_lineage_and_occupied_then_vacated(tmp_path):
    t = [x for x in json.loads((V2 / "roster.json").read_text()) if x["dest_tag"] == "vacated_by_prefix"]
    assert len(t) == 2 and {bool(x["prefix"]) for x in t} == {True, False}
    root = P.scene_from_dict(P.load_root(t[0]["root"])["scene"])
    h0 = P.scene_hash(root)
    edited = P.substitute_pose(root, *next(x for x in t if x["prefix"])["prefix"][0])
    assert P.scene_hash(root) == h0 != P.scene_hash(edited)                 # the sibling root scene is never mutated
    r = P.trial(next(x for x in t if not x["prefix"]), tmp_path)
    assert not r["eligible"] and r["label"] is None and r["excluded"] == "DEST_OCCUPIED"


def test_fingerprint_ignores_volatile_fields(runs, tmp_path):
    out, fx, _ = runs
    a = C.digest(out, fx)[0]
    assert C.digest(out, {**fx, "volatile": {"wall_s": 1e9, "cpu_s": 0, "worker_peak_rss_mb": 0}})[0] == a
    assert C.digest(out, {**fx, "label": 0})[0] != a
