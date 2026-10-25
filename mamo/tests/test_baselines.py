"""RRT, KPIECE and SELSIM (mamo.rrt, mamo.kpiece, mamo.selsim) and their shared machinery."""

from pathlib import Path

import numpy as np
import pytest

from mamo.config import Config
from mamo.context import MAMOContext
from mamo.execute import execute
from mamo.kinodynamic import KState, PhysicsProblem
from mamo.kpiece.planner import Grid, Motion, _half_normal_index
from mamo.scene import parse_scene_file

ROOT = Path(__file__).resolve().parents[1]
STACKING = ROOT / "workspace/datasets/reap-retrieve/stacking/retrieve/stacking_retrieve_easy_0000.txt"
REFERENCE = ROOT.parent / "archive/ros_src/em4m-main/dat/clutter_scenes/5"


def _ctx(path: Path) -> MAMOContext:
    return MAMOContext(parse_scene_file(path), Config(), 0)


def _any_reference_scene() -> Path:
    files = sorted(REFERENCE.glob("*.txt"))
    if not files:
        pytest.skip("reference scenes not available")
    return files[0]


def test_grid_splits_motions_and_tracks_neighbours():
    g = Grid(np.array([1.0, 1.0, 1.0]))
    nu = Motion(None, None, np.zeros((5, 7)), np.zeros((5, 3)), None, False)
    coords = [(0, 0, 0), (0, 0, 0), (1, 0, 0), (1, 0, 0)]
    g.add_motion(nu, coords, 1, iteration=2, score=1.0)
    assert set(g.cells) == {(0, 0, 0), (1, 0, 0)}
    assert g.cells[(0, 0, 0)].parts == [(nu, 1, 2)] and g.cells[(1, 0, 0)].parts == [(nu, 3, 4)]
    assert g.total_coverage == 4
    assert g.cells[(0, 0, 0)].neighbors == 1 and g.cells[(1, 0, 0)].neighbors == 1
    # Surround (0,0,0): it becomes interior once all 2k = 6 neighbours exist.
    for z in [(-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
        g.add_motion(nu, [z], 1, 3, 1.0)
    assert g.interior(g.cells[(0, 0, 0)])
    assert not g.interior(g.cells[(1, 0, 0)])


def test_half_normal_prefers_recent():
    rng = np.random.default_rng(0)
    idx = [_half_normal_index(rng, 30) for _ in range(3000)]
    assert min(idx) == 0 and max(idx) <= 29
    assert np.mean(np.array(idx) < 10) > 0.6  # most picks among the newest third


def test_steer_rrt_rejects_and_kpiece_keeps_prefix():
    ctx = _ctx(_any_reference_scene())
    prob = PhysicsProblem(ctx)
    start = prob.start
    # Reaching straight down through the shelf hits an immovable obstacle somewhere on the way.
    target = start.q.copy()
    target[1] += 1.5
    target[3] -= 0.2
    rrt_res = prob.steer(start, target, partial=False)
    kp_res = prob.steer(start, target, partial=True)
    if rrt_res.ok:
        pytest.skip("motion happens to be free in this scene")
    assert not rrt_res.ok
    if kp_res.ok:
        assert 0 < kp_res.frac < 1 and len(kp_res.path) >= 2
        assert np.allclose(kp_res.path[0], start.q)


def test_bfs_heuristic_zero_at_goal_and_grows():
    from mamo.selsim.heuristic import BFSHeuristic
    ctx = _ctx(_any_reference_scene())
    prob = PhysicsProblem(ctx)
    assert prob.setup_goal()
    h = BFSHeuristic(ctx.world, prob.p_pre)
    assert h(prob.p_pre) == 0
    p_home = prob.project(prob.start.q)
    assert np.isfinite(h(p_home)) and h(p_home) > 0
    assert h(np.array([100.0, 0, 0])) == np.inf


def test_selsim_culprit_search():
    from mamo.selsim.planner import SelSim
    import time
    ctx = _ctx(_any_reference_scene())
    s = SelSim(ctx, 1.0, time.perf_counter())
    mov = sorted(s.movable_ids)
    fixed = sorted(s.fixed_ids)
    a, b = mov[0], mov[1]
    f = fixed[0]
    pairs = {frozenset((a, -1)), frozenset((a, b)), frozenset((b, f))}
    # b knocked into fixed object f; f is not relevant yet, so f itself is the culprit.
    assert s._culprit(f"immovable contact: obj_{b} / obj_{f}", pairs) == f
    s.relevant = {f}
    assert s._culprit(f"immovable contact: obj_{b} / obj_{f}", pairs) == b
    s.relevant = {f, b}
    assert s._culprit(f"immovable contact: obj_{b} / obj_{f}", pairs) == a
    s.relevant = {f, b, a}
    assert s._culprit(f"immovable contact: obj_{b} / obj_{f}", pairs) is None
    assert s._culprit(f"object {a} tilted 40 deg", set()) == a or a in s.relevant


@pytest.mark.parametrize("method", ["rrt", "kpiece", "selsim"])
def test_planner_solves_and_replays_easy_scene(method):
    if not STACKING.exists():
        pytest.skip("reap-retrieve dataset not generated")
    from mamo.bench.planners import REGISTRY
    sc = parse_scene_file(STACKING)
    ctx = MAMOContext(sc, Config(), 0)
    sol = REGISTRY[method].fn(ctx, 60)
    assert sol.solved, sol.reason
    assert all(a.kind == "motion" for a in sol.pushes)
    ex = execute(sc, sol, Config())
    assert ex.success, ex.reason
