"""Teacher V2 (bd_3dt): within a relocation sequence the arm goes to the ready pose only after the last relocation.
Needs MuJoCo, the 3D repository (BD_RSS3D_ROOT) and the xArm7 model; skipped otherwise."""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[3] / "raw" / "roots" / "3dt-plus3dt-w0000-s012.json"     # a K = 2 scene


@pytest.fixture(scope="module")
def A():
    pytest.importorskip("mujoco")
    import bd_3dt
    try:
        bd_3dt.env()
    except (FileNotFoundError, ModuleNotFoundError) as e:
        pytest.skip(f"3D scene not available: {e}")
    return bd_3dt


def test_ready_pose_once_after_last_relocation(A):
    r = json.loads(ROOT.read_text())
    seq = next(b["sequence"] for b in r["labels"]["B"] if b["role"] == "minimal")
    steps = [(o, d) for o, d in seq] + [("t0", r["goal"])]
    e = A.build(r["objects"], r["initial_cells"])
    A.settle(e, r["initial_cells"])
    q_ready, ends = e.q_ready.copy(), []
    for j, (o, d) in enumerate(steps):
        home = j >= len(steps) - 2
        p, _ = A.query(e, o, d, home=home)
        assert p is not None
        ends.append(p["segments"][-1]["phase"])
        e.run_plan(p)
        assert (np.abs(e.q_now() - q_ready).max() < 0.05) == home        # away from the ready pose exactly between relocations
    assert ends == ["retreat"] * (len(seq) - 1) + ["to_ready", "to_ready"]
    assert "ready pose once" in A.TEACHER_VERSION and "_plan_retreat" not in e.__dict__
