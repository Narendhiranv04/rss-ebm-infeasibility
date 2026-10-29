"""B/D dataset bootstrap (bd_core / bd_2d / bd_3d / bd_run): label semantics, action cap, verified vs requested K, grouped splits,
reproducibility, side-grasp restriction, readers of the existing JSON / NPZ storage."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

import bd_core as bc
import bd_2d


def test_binary_labels_only_and_reasons_are_metadata():
    assert bc.binary(True, "VERIFIED")["y"] == 1 and bc.binary(True, "VERIFIED")["negative_kind"] is None
    r = bc.binary(False, "UNKNOWN_BUDGET", "TIMEOUT")
    assert r["y"] == 0 and r["negative_kind"] == "TIMEOUT" and r["reason"] == "UNKNOWN_BUDGET"
    with pytest.raises(ValueError):
        bc.binary(False, "x", "UNKNOWN")                       # no third class
    m = {s: bd_2d.label(s)["negative_kind"] for s in ("CERTIFIED_INFEASIBLE", "STATIC_INVALID", "UNKNOWN_BUDGET", "UNKNOWN_PLANNER",
                                                     "UNKNOWN_TRANSIT")}
    assert m == {"CERTIFIED_INFEASIBLE": "CERTIFIED", "STATIC_INVALID": "CERTIFIED", "UNKNOWN_BUDGET": "TIMEOUT", "UNKNOWN_PLANNER": "NO_PLAN_FOUND",
                 "UNKNOWN_TRANSIT": "NO_PLAN_FOUND"}
    assert bd_2d.label("VERIFIED")["y"] == 1 and bd_2d.label("VERIFIED", replay=False)["y"] == 0


def test_seven_action_cap_t_last_and_redundancy_bound():
    assert bc.check_sequence([("o00", 3)] * 6) == 7
    with pytest.raises(ValueError):
        bc.check_sequence([("o00", 3)] * 7)
    with pytest.raises(ValueError):
        bc.check_sequence([("t0", 3)])
    assert bc.check_sequence([("o00", 3)] * 4, K=2) == 5
    with pytest.raises(ValueError):
        bc.check_sequence([("o00", 3)] * 5, K=2)               # 3 redundant > 2


def test_coverage_counts_verified_K_not_requested_K():
    led = [{"accepted": True, "spec": {**bc.SPECS[0], "K": 4}, "verified_K": 0}] * 30
    picks = bc.plan_wave(led, 6, 1, "t")
    assert all(p["K"] != 0 for p in picks)                     # K0 is already over-covered by OBSERVED K, so new specs ask for K >= 1
    assert bc.observed_key({"verified_K": None}) == "NONE" and bc.observed_key({"verified_K": 3}) == 3


def test_wave_plan_and_ids_are_deterministic():
    a, b = bc.plan_wave([], 8, 0, "salt"), bc.plan_wave([], 8, 0, "salt")
    assert a == b and len({json.dumps(x, sort_keys=True) for x in a}) > 1
    i1 = bc.ids("2d", "n", 3, 4, a[0], "c", "t", "p")
    assert i1 == bc.ids("2d", "n", 3, 4, a[0], "c", "t", "p") and i1["construction_seed"] != bc.ids("2d", "n", 3, 5, a[0], "c", "t", "p")["construction_seed"]
    assert all(bc.valid_spec(s) and s["K"] + bc.N_DISTRACT[s["distractors"]][0] <= bc.MAX_MOVABLE for s in bc.SPECS)


def test_grouped_split_integrity_and_store_roundtrip(tmp_path):
    st = bc.Store(tmp_path)
    w = st.witness({"q": np.arange(6.0).reshape(2, 3)})
    assert st.witness({"q": np.arange(6.0).reshape(2, 3)}) == w and np.array_equal(st.load_witness(w)["q"], np.arange(6.0).reshape(2, 3))
    spec = bc.SPECS[5]
    for slot in range(6):
        ids = bc.ids("2d", "t", 0, slot, spec, "c", "t", "p")
        st.save_root({"ids": ids, "spec": spec, "observed": 1, "verified_K": 1, "n_movable": 2,
                      "labels": {"original": {"y": 0}, "blocker": {"blocker_supervision_valid": True}, "D": [], "B": []}})
    m = bc.manifest(st, "2d")
    sp = json.loads((tmp_path / "SPLITS.json").read_text())["splits"]["iid"]
    assert m["n_roots"] == 6 and len({sp[r["root_id"]] for r in m["rows"]}) == 1      # one family -> one split


def test_2d_construction_is_reproducible_and_uses_declared_side_grasps(tmp_path):
    import skill as sk
    import config as cf
    p = sk.params(cf.load_config("config.toml"))
    assert len(sk.directions(p)) == 2 * p["approach_dirs"] == 48         # the declared side-grasp set is untouched
    spec = {"K": 0, "distractors": "1-2", "source": "near", "displacement": "short", "mechanism": "grasp", "parking": "one",
            "distractor_role": "far", "start_pose": "home"}
    try:
        a = bd_2d.construct(spec, 12345, "test-root")
    except bd_2d.Reject:
        pytest.skip("this seed is a recorded construction rejection")
    b = bd_2d.construct(spec, 12345, "test-root")
    from scene import scene_to_dict
    assert scene_to_dict(a["scene"]) == scene_to_dict(b["scene"]) and a["goal"] == b["goal"] and a["parking"] == b["parking"]
    assert not a["fixed_ids"] and all(r.startswith("distractor") for r in a["roles"].values())


def test_existing_json_npz_readers_still_load():
    import g4d_run as gr
    plan = json.loads(Path("runs/p16/data/dev/plan.json").read_text())
    assert plan["roots"]
    lab = json.loads(Path("runs/p16/data/dev/labels_v2w2.json").read_text())
    assert all("original" in v for v in lab.values())
    e = json.loads(Path("runs/p14/lookup/INDEX.json").read_text())["entries"][0]
    a = gr._arrays(Path(e["file"]))[e["index"]]
    assert {"q", "gap", "phase"} <= set(a)


@pytest.mark.skipif(not Path(os.environ.get("BD_RSS3D", "/data/user1/rss_ebm_dataset/src/rss-3d/gh-yadu-iiith/random")).exists(),
                    reason="3D repository not available here")
def test_3d_side_grasp_set_and_cell_convention():
    import bd_3d
    G = bd_3d.env()
    assert max(G["E"].PITCHES_DEG) <= 25 and min(G["E"].PITCHES_DEG) >= 0       # near-horizontal side approaches only
    assert bd_3d.uid((0, 0)) == 1 and bd_3d.uid((7, 7)) == 64
    assert bd_3d.supports({**bc.SPECS[0], "start_pose": "home"}) and not bd_3d.supports({**bc.SPECS[0], "start_pose": "nonhome"})
