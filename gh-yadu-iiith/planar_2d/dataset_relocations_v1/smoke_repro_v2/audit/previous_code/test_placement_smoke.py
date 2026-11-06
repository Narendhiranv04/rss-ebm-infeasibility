"""Placement smoke sidecar: predeclared roster, operational binary labels, content-addressed witness round trip (no teacher run)."""
import json
from pathlib import Path

import numpy as np
import placement_smoke as P

SMOKE = Path(__file__).resolve().parents[1] / "smoke_20261106"


def test_roster_is_frozen_and_bounded():
    t = P.roster()
    assert t == json.loads((SMOKE / "roster_predeclared.json").read_text())
    assert len({x["root"] for x in t}) <= 4 and len(t) >= 28
    for rid in {x["root"] for x in t}:
        objs = {x["object"] for x in t if x["root"] == rid and not x["prefix"]}
        assert len(objs) <= 2 and all(sum(x["object"] == o and x["root"] == rid and not x["prefix"] for x in t) <= 5 for o in objs)
    assert 1 <= sum(bool(x["prefix"]) for x in t) <= 2


def _c(stage, certified=True):
    c = P.sk.Candidate(0.0, 1, stage)
    c.certified = certified
    return c


def test_labels_are_binary_and_certificates_only_from_definite_screens():
    counts = lambda geo, unc, val, ok, bad: {"total": geo + unc + val + ok + bad, "geometrically_rejected": geo,  # noqa: E731
                                             "screen_uncertain_not_tested": unc, "valid_not_motion_tested": val, "verified_success": ok,
                                             "budget_failed": bad}
    assert P.decide([_c("OK")], [_c("OK")], counts(0, 0, 3, 1, 0), [{}], False)["label"] == 1
    lab = P.decide([_c("GRASP_POSE")], [_c("OK")], counts(4, 0, 0, 0, 0), [], False)
    assert lab == {"label": 0, "negative_kind": "CERTIFIED_SCREEN", "failure_phase": "POST_GRASP_STATE", "is_certificate": True}
    lab = P.decide([_c("OK")], [_c("OK")], counts(0, 0, 5, 0, 8), [{"replay": "CERTIFIED"}], False)
    assert lab["label"] == 0 and lab["negative_kind"] == "BUDGET_EXHAUSTED" and not lab["is_certificate"]
    lab = P.decide([_c("GRASP_POSE", False)], [_c("OK")], counts(0, 4, 0, 0, 0), [], False)
    assert lab["label"] == 0 and not lab["is_certificate"] and "UNRESOLVED" not in json.dumps(lab)


def test_witness_round_trip_is_content_addressed(tmp_path):
    src = sorted((SMOKE / "trajectories").glob("*.npz"))[0]
    f = np.load(src)
    core = {k: f[k] for k in ("q", "gap", "phase", "attached", "held_local", "place_xy")}
    robot = P.A.env()["robot"]
    a = P.save_witness(tmp_path, robot, str(f["scene_hash"]), str(f["object_id"]), core, {"kind": "TEST"})
    b = P.save_witness(tmp_path, robot, str(f["scene_hash"]), str(f["object_id"]), core, {"kind": "TEST"})
    assert a == b == src.stem and len(list(tmp_path.glob("trajectories/*"))) == 1
    g = np.load(tmp_path / "trajectories" / f"{a}.npz")
    assert all(np.array_equal(g[k], core[k]) for k in core) and np.isfinite(g["object_xy"]).all()
