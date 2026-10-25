"""Unit tests for the I-M4M additions: graspability, grasp directions, parameter sampling, OPEN order."""

from __future__ import annotations

import math

import numpy as np

from mamo.config import Config
from mamo.im4m.planner import IM4M, LazyEdge
from mamo.im4m.pickplace import grasp_directions, graspable
from mamo.im4m.robust import sample_scene
from mamo.scene import BOX, CYLINDER, IMMOVABLE, MOVABLE, Scene, SceneObject, default_shelf_parts


def _obj(shape: int, xs: float, ys: float, yaw: float = 0.0) -> SceneObject:
    return SceneObject(7, shape, MOVABLE, 0.7, -0.5, 0.85, 0, 0, yaw, xs, ys, 0.08, 0.3, 1.0, True)


def test_graspable_follows_gripper_width() -> None:
    cfg = Config()
    lim = cfg.pick_place.max_half_width
    assert graspable(_obj(CYLINDER, lim - 0.005, lim - 0.005), cfg)
    assert not graspable(_obj(CYLINDER, lim + 0.005, lim + 0.005), cfg)
    # a box is graspable if either side fits
    assert graspable(_obj(BOX, 0.08, lim - 0.005), cfg)
    assert not graspable(_obj(BOX, 0.08, lim + 0.005), cfg)


def test_grasp_directions_match_reference_pregrasps() -> None:
    cfg = Config()
    # thin along y only: fingers close along y, so the approach is along +-x; only +x faces the robot
    assert [round(math.degrees(y)) for y in grasp_directions(_obj(BOX, 0.08, 0.02), cfg)] == [0]
    # cylinder: 8 directions, of which those with |yaw| <= 90 deg are kept
    assert len(grasp_directions(_obj(CYLINDER, 0.02, 0.02), cfg)) == 5


def test_parameter_samples_are_clipped() -> None:
    objs = default_shelf_parts() + [
        SceneObject(6, CYLINDER, IMMOVABLE, 0.75, -0.5, 0.85, 0, 0, 0, 0.03, 0.03, 0.15, 0.3, 1.0, False),
        _obj(BOX, 0.03, 0.03),
    ]
    sc = Scene(objs, 6, "t")
    rng = np.random.default_rng(0)
    for _ in range(200):
        s = sample_scene(sc, rng)
        o = s.by_id(7)
        assert 0.15 - 1e-9 <= o.mass <= 0.6 + 1e-9 and 0.5 - 1e-9 <= o.mu <= 2.0 + 1e-9
        assert s.by_id(6).mass == 0.3  # immovables are not perturbed


class _FakeCtx:
    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)


def test_open_order_is_depth_first() -> None:
    """(3, 0) < (3, 1) < (2, 4) < (1, 0) < (0, 2), as in the paper; evaluated before unevaluated."""
    im = IM4M.__new__(IM4M)
    im.open, im.ctx = [], _FakeCtx()
    import itertools
    im.counter = itertools.count()

    class V:
        def __init__(self, actions: int, re: int, prio: float = 0.0) -> None:
            self.actions, self.noops, self.priority, self.version, self.closed = actions, re, prio, 0, False

    order = [(3, 0), (3, 1), (2, 4), (1, 0), (0, 2)]
    vs = [V(a, r) for a, r in order]
    for v in reversed(vs):
        im._push_open(v)
    parent = V(2, 0)
    im._edge_priority = lambda e: 0.0
    edge = LazyEdge(parent, 0, [(0, 0), (1, 0)], "push")  # would create a vertex with 3 actions
    im._push_open(edge)
    import heapq
    popped = [heapq.heappop(im.open)[-1] for _ in range(len(im.open))]
    assert popped[:2] == [vs[0], edge] and popped[2:] == vs[1:]


def test_dogar_dijkstra_finds_cheapest_path_around_walls() -> None:
    from types import SimpleNamespace
    from mamo.dogar.planner import Dogar
    valid = np.ones((5, 5), dtype=bool)
    valid[2, :4] = False  # wall with a gap at (2, 4)
    cost = np.full((5, 5), 1000, dtype=np.int64)
    agent = SimpleNamespace(valid=valid, cost=cost, start=(0, 0))
    dist, prev = Dogar._dijkstra(agent)
    path = Dogar._path(prev, (4, 0))
    assert path[0] == (0, 0) and path[-1] == (4, 0) and (2, 4) in path
    assert all(abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1 for a, b in zip(path, path[1:]))
    assert dist[(4, 0)] == 1000 * (len(path) - 1)
    valid[2, 4] = False
    assert (4, 0) not in Dogar._dijkstra(agent)[0]
