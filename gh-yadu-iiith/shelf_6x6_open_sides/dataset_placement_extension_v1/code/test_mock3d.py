"""Focused regression tests for the SHELF66V3 placement sidecar (run on gvlab, one CPU: pytest -q code/test_mock3d.py).
Physical fixtures come from TRAINING roots outside the mock roster (mock_v1/configs/fixtures.json, frozen before the pilot)."""
import gzip, inspect, json, signal, sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement3d as P  # noqa: E402
import plan3d as L  # noqa: E402
import policy3d as Q  # noqa: E402
import replay3d as R  # noqa: E402
import fixtures3d as F  # noqa: E402

ROOT = P.HERE.parent
FX = json.loads((ROOT / "mock_v1/configs/fixtures.json").read_text())["fixtures"]
ROSTER = [json.loads(x) for x in gzip.open(ROOT / "mock_v1/roster.jsonl.gz", "rt")]
_cache = {}


def fx(kind, policy="reference_v1"):
    if (kind, policy) not in _cache:
        f = FX[kind]
        raw = json.loads((P.SRC / "raw/roots" / f"{f['root_id']}.json").read_text())
        sc = P.Scene(raw["objects"], f["cells"])
        rec, w = Q.evaluate(sc, f["object"], tuple(f["dest"]), Q.POLICIES[policy], early_stop=policy != "reference_v1")
        _cache[(kind, policy)] = (f, raw, sc, rec, w)
    return _cache[(kind, policy)]


def test_protocol_identity():
    assert P.G["protocol"] == "XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022" and P.G["config_hash"] == "8d521b1f6d23db23"
    assert P.S.NO_OVERPASS and P.S.ARM_CEILING and not P.S.RETURN_TO_READY and P.S.GRASP_AT == "middle,top"
    f, raw, sc, rec, w = fx("obstructed_carry_certified")
    ps = P.poses(sc, f["object"])
    assert len(ps) == 32 and len({g["azimuth_deg"] for g in ps}) == 16 and {g["grasp_level"] for g in ps} == {"middle", "top"}
    r = Q.ranked(ps)
    assert sorted(g["id"] for g in r) == list(range(32)) and len({g["azimuth_deg"] for g in r[:16]}) == 16
    assert [g["grasp_level"] for g in r[:4]] == ["middle", "top", "middle", "top"]


def test_finger_preshaped_in_every_object_check(monkeypatch):
    """Regression for the pre-V3 bug (object checks with FULLY OPEN fingers): every object check of screen + plan sees the fingers at
    the grasp's open width; the structure check keeps the source's default finger qpos (fully open = conservative), documented."""
    f, raw, sc, rec, w = fx("target_goal_destination")
    seen, orig = [], P.S.XArm7TableEnv._object_hits

    def spy(self, q, *a, **kw):
        seen.append(self._check_width)
        out = orig(self, q, *a, **kw)
        if self._check_width is not None:
            assert np.allclose(self.ik_data.qpos[self.grip_qadr], self._finger_q(self._check_width))
        return out
    monkeypatch.setattr(P.S.XArm7TableEnv, "_object_hits", spy)
    g = Q.ranked(P.poses(sc, f["object"]))[0]
    st, br, _ = P.screen(sc, f["object"], tuple(f["dest"]), g)
    if br:
        P.plan(sc, f["object"], tuple(f["dest"]), g, br, 500)
    assert seen and all(x == pytest.approx(g["open_width"]) for x in seen)
    full = float(sc.e._grip_cal[0][-1])
    assert g["open_width"] < full - 0.02                  # pre-shaped is >= 2 cm narrower than fully open
    if rec["label"] == 1:
        m = next(m for m in rec["modes"] if m["rank"] == rec["verified_rank"])
        h = m["tests"][-1]["exec"]["hold"]
        assert h["both_fingers"] and sum(h["pre_contacts"]) == 0 and h["object_shift_m"] <= P.TOL["hold_shift_m"]


def test_static_contrasts_are_unlabeled_and_never_planned():
    st = [q for q in ROSTER if q["static"]]
    assert st and all(Q.static_reason({k: tuple(v) for k, v in q["cells"].items()}, q["object"], tuple(q["dest"])) == q["static"] for q in st)
    import run3d
    units = run3d.load_units(ROOT / "mock_v1/roster.jsonl.gz", "all", set())
    assert not any(q["static"] for v in units.values() for q in v)
    assert Q.static_reason({"o": (0, 0), "p": (1, 1)}, "o", (1, 1)) == "OCCUPIED" and Q.static_reason({"o": (0, 0)}, "o", (6, 0)) == "OUT_OF_BOUNDS"


def test_nested_rrt_budget_is_an_exact_prefix():
    f, raw, sc, rec, w = fx("budget_exhausted_negative")
    m = next(m for m in rec["modes"] if m["state"] == "UNRESOLVED")
    g = Q.ranked(P.poses(sc, f["object"]))[m["rank"]]
    br = P.screen(sc, f["object"], tuple(f["dest"]), g)[1]
    big = P.plan(sc, f["object"], tuple(f["dest"]), g, br, 5000)["route"]
    small = P.plan(sc, f["object"], tuple(f["dest"]), g, br, 500)["route"]
    d = lambda a, b: {k: a[k] - b[k] for k in a if k != "cpu"}  # noqa: E731
    assert not big["found"] and not small["found"] and d(small["end"], small["start"]) == d(big["milestones"][500], big["start"])


def test_reproducible_and_cache_independent():
    f, raw, sc, rec, w = fx("pick_inaccessible_held_placeable")
    sc2 = P.Scene(raw["objects"], f["cells"], {})                   # cold cache, fresh scene
    rec2, w2 = Q.evaluate(sc2, f["object"], tuple(f["dest"]), Q.POLICIES["reference_v1"], early_stop=False)
    assert Q.canonical(rec) == Q.canonical(rec2) and rec["label"] == 1 and Q.wid(w) == Q.wid(w2) == rec["witness_id"]
    e = FX["pick_inaccessible_held_placeable"]["expect"]
    assert (rec["label"], rec["status"], rec["verified_rank"], rec["witness_id"]) == (e["label"], e["status"], e["verified_rank"], e["witness_id"])


def test_pick_inaccessible_but_held_placeable():
    """Mode level (frozen): the verified pose's own pick approach fails on every IK branch, yet the held object is placed and verified.
    The query-level variant (all 32 poses pick-inaccessible) was searched (fixtures.json 'level' / f1_search logs) and not found."""
    f, raw, sc, rec, w = fx("pick_inaccessible_held_placeable")
    assert rec["label"] == 1 and F.pose_pick_access(sc, f["object"], rec["verified_rank"]) == []
    assert f["level"].startswith("mode") or F.pick_access(sc, f["object"])[0] == 0


def test_obstructed_carry_is_a_certified_negative():
    f, raw, sc, rec, w = fx("obstructed_carry_certified")
    assert rec["label"] == 0 and rec["status"] == "NEG_CERTIFIED_GEOMETRY" and rec["accounting"]["valid"] == 0
    assert rec["accounting"]["geometry_invalid"] >= 1 and rec["accounting"]["motion_tested"] == 0


def test_budget_exhausted_negative_has_scope():
    f, raw, sc, rec, w = fx("budget_exhausted_negative")
    assert rec["label"] == 0 and rec["negative_kind"] == "BOUNDED_SEARCH" and rec["accounting"]["unresolved"] >= 1


def test_target_goal_cell_is_a_legal_destination():
    f, raw, sc, rec, w = fx("target_goal_destination")
    assert tuple(f["dest"]) == tuple(raw["goal"]) and rec["label"] in (0, 1)


def test_vacated_after_edit_vs_occupied_before():
    f, raw, sc, rec, w = fx("vacated_after_edit")
    assert rec["label"] in (0, 1) and f["depth"] == 1
    rid = f["root_id"]
    s0 = {k: tuple(v) for k, v in raw["initial_cells"].items()}
    assert Q.static_reason(s0, f["object"], tuple(f["dest"])) == "OCCUPIED"


def test_release_and_settling_failures_are_caught():
    """Synthetic (not a statistical sample): release right after the lift -> wrong cell; tilt the placed object -> not upright."""
    f, raw, sc, rec, w = fx("target_goal_destination") if fx("target_goal_destination")[3]["label"] == 1 else fx("pick_inaccessible_held_placeable")
    pl = P.unpack(w)
    g = next(x for x in P.poses(sc, f["object"]) if x["id"] == int(w["grasp"][0]))
    early = {**pl, "segs": [pl["segs"][0], {"phase": "release", "kind": "grip", "grip": g["open_width"]}]}
    ok, hev, before = P.hold(sc, f["object"], g, pl["q_g"])
    r1, ev1 = P.execute(sc, f["object"], early, before)
    v1, vi1 = P.verify(sc, f["object"], tuple(f["dest"]), r1, ev1)
    assert not v1 and vi1["first_failed"] in ("cell", "xy")
    ok, hev, before = P.hold(sc, f["object"], g, pl["q_g"])
    r2, ev2 = P.execute(sc, f["object"], pl, before)
    e, i = sc.e, sc.e.obj_id(f["object"])
    a = e.obj_qadr[i]
    e.data.qpos[a + 3:a + 7] = [np.cos(0.3), np.sin(0.3), 0, 0]
    e.data.qpos[a + 2] += 0.03
    e.wait(5)
    v2, vi2 = P.verify(sc, f["object"], tuple(f["dest"]), r2, ev2)
    assert not v2 and not vi2["gates"]["upright"]


def test_witness_roundtrip_and_fresh_replay(tmp_path):
    f, raw, sc, rec, w = fx("pick_inaccessible_held_placeable")
    np.savez_compressed(tmp_path / "w.npz", **w)
    z = np.load(tmp_path / "w.npz"); wz = {k: z[k] for k in z.files}
    assert Q.wid(wz) == rec["witness_id"] and wz["q"].shape[1] == 7 and wz["q_exec"].shape[1] == 7
    q = {"qid": f["qid"], "cells": f["cells"], "object": f["object"], "dest": f["dest"]}
    out = R.replay_one(raw, q, wz)
    assert out["verified"] and out["grasp_pose_identical"] and out["q_exec_max_abs_diff"] == 0.0


def test_interruption_is_unlabeled_never_zero():
    f = FX["budget_exhausted_negative"]
    raw = json.loads((P.SRC / "raw/roots" / f"{f['root_id']}.json").read_text())
    sc = P.Scene(raw["objects"], f["cells"])

    def boom(*_):
        raise Q.Interrupted("test guard")
    old = signal.signal(signal.SIGPROF, boom)
    signal.setitimer(signal.ITIMER_PROF, 0.5)
    try:
        with pytest.raises(Q.Interrupted):
            Q.evaluate(sc, f["object"], tuple(f["dest"]), Q.POLICIES["reference_v1"], early_stop=False)
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0); signal.signal(signal.SIGPROF, old)
    src = inspect.getsource(__import__("run3d").worker)
    assert '"label": None, "status": "INTERRUPTED"' in src


def test_two_edit_descendant_hash_and_split_closure():
    by = {}
    for q in ROSTER:
        by.setdefault(q["root_id"], {})[q["depth"]] = q["scene_hash"]
        objs = json.loads((P.SRC / "raw/roots" / f"{q['root_id']}.json").read_text())["objects"]
        assert L.scene_hash(objs, {k: tuple(v) for k, v in q["cells"].items()}) == q["scene_hash"]
    for q in ROSTER:
        if q["depth"] == 2:
            assert q["parent_scene"] == by[q["root_id"]][1] and len(q["edits"]) == 2
    man = json.loads((ROOT / "mock_v1/split_manifest.json").read_text())
    assert all(q["split"] == "train" for q in ROSTER)
    for g, rs in man["groups"].items():
        assert len({man["split"][r] for r in rs}) == 1


def test_roster_uses_no_protected_metadata():
    src = inspect.getsource(L.roster_for) + inspect.getsource(L._edit) + inspect.getsource(L.pick_roots) + inspect.getsource(L.load_selected)
    for bad in ("roles", "parking", "intended", "greedy", "labels", "\"spec\"", "verified_K", "blocker"):
        assert bad not in src, bad
