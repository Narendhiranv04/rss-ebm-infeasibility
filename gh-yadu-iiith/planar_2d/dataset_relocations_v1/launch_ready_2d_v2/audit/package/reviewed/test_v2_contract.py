"""Regression tests for the Approach B v2 launch-readiness candidate.

These tests run ONLY against code_v2 and launch_ready_2d_v2 after metadata init.
They must not invoke the full 2D collection. Some may call the physical teacher
on existing development fixture roots, consistent with the bounded mock contract.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

import launch_plan as LP
import launch_runner as R
import launch_guard as G
import launch_check as K
import placement_teacher as T

P = T.P
L = Path(__file__).resolve().parents[1] / "launch_ready_2d_v2"


def train_roots():
    s = json.loads((L / "split_manifest.json").read_text())
    return [r["root_id"] for r in s["roots"] if r["split"] == "train"]


def test_goal_cell_is_not_an_excluded_placement_destination():
    """A change in target task goal cannot affect a placement-only roster."""
    G0 = P.A.env()
    checked = 0
    for rid in train_roots()[:30]:
        r = P.load_root(rid)
        orig = LP.root_plan(rid, r, G0, False)
        altered = copy.deepcopy(r)
        legal = altered["scene"]["legal_cells"]
        altered["goal_cell"] = legal[-1] if r["goal_cell"] != legal[-1] else legal[0]
        assert LP.root_plan(rid, altered, G0, False) == orig
        checked += 1
    assert checked == 30


def test_protected_repair_metadata_does_not_change_any_placement_query():
    G0 = P.A.env()
    for rid in train_roots()[:25]:
        r = P.load_root(rid)
        orig = LP.root_plan(rid, r, G0, False)
        altered = copy.deepcopy(r)
        altered.update(roles={}, intended_blockers=[], parking={}, alt_parking={}, construction={}, verified_K=None, observed=None)
        altered["labels"] = {"B": [], "D": [], "original": {}, "blocker": {}}
        assert LP.root_plan(rid, altered, G0, False) == orig


def test_two_edit_prefixes_are_distinct_and_only_hypothetical():
    G0 = P.A.env()
    examples = 0
    for rid in train_roots()[:100]:
        r = P.load_root(rid)
        plans = LP.root_plan(rid, r, G0, False)
        for row in plans:
            if row[1] != "prefix3":
                continue
            _, _, prefix, kind, obj, *_ = row
            assert len(prefix) == 2 and kind == "DESIGN_TWO_EDIT"
            assert len(set(x[0] for x in prefix)) == 2
            assert obj not in {x[0] for x in prefix}
            scene = T.scene_for({"root": rid, "prefix": prefix})
            cells = [o.cell for o in scene.objects]
            assert len(set(cells)) == len(cells)
            examples += 1
    assert examples > 0


def test_adaptive_schedule_and_all_witness_retention_on_dev_fixture(tmp_path):
    """Runs just one dev-touched fixture; no new randomized scene root."""
    fx = {"root": "2d-plus2d-srv-w0408-s006", "trial_id": "contract-v2-adaptive-008",
          "prefix": [], "object": "o08", "dest": 55, "block": "root", "prefix_kind": None,
          "source_cell": 0, "source_bin": "", "dest_bin": "", "stratum_k": 0,
          "stratum_n": 0, "obj_rank": 0, "dest_rank": 0}
    pol = {"pairs_per_query": 8, "rescue_pairs": 32, "coverage": "ADAPTIVE_FULL",
           "witness_store_max": 32, "reuse_archived": False}
    q = T.run_query(fx, tmp_path, pol, emergency_cpu_s=1200)
    assert q["completed"] and q["label"] in (0, 1)
    assert q["prefix_provenance"] == "ROOT"
    assert not q["context_selected_with_protected_metadata"]
    assert q["mode_counts"]["reused_reverified"] == 0
    assert len(q["witness_refs"]) <= q["n_verified_pairs"]
    assert len(q["witness_refs"]) == len(set(q["witness_refs"]))
    if q["label"]:
        robot, pars = P.A.env()["robot"], P.sk.params(P.A.env()["cfg"])
        scene = T.scene_for(fx)
        for wid in q["witness_refs"]:
            core = T.witness(tmp_path, fx["root"], wid)
            assert P.verify_witness(robot, scene, fx["object"], core,
                                    np.asarray(scene.lattice.center(fx["dest"])), pars)[0]
    z = np.load(tmp_path / "parts" / fx["root"] / f"{fx['trial_id']}.npz")
    n = sum(z["a_rank"] >= 0)
    assert n <= 32
    assert n >= min(8, q["mode_counts"]["valid_pairs"])
    success_in_base = any(bool(v) and 0 <= int(r) < 8 for r, v in zip(z["a_rank"], z["a_verified"]))
    expected = min(8 if success_in_base else 32, q["mode_counts"]["valid_pairs"])
    assert n == expected


def test_production_guard_always_refuses_without_policy(monkeypatch):
    target = Path(R.cfg_of(L)["full_run_dir"])
    assert not target.exists()
    monkeypatch.setattr(P, "plan_carry", lambda *a, **k: pytest.fail("teacher motion planner invoked"))
    monkeypatch.setattr(R, "run", lambda *a, **k: pytest.fail("production runner invoked"))
    with pytest.raises(SystemExit):
        G.full(L, ["--execute-full", "--confirm", G.token(L)], execute=False)
    assert not target.exists()


def test_scientific_identity_pins_the_placement_adapter():
    sem = T.pair_protocol()[1]
    assert sem["adapter_sha256"] == P.sha_file(T.__file__)
    assert sem["smoke_code_sha256"] == P.sha_file(P.__file__)
