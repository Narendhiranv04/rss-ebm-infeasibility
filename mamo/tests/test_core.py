"""Fast unit tests for the building blocks (no full planning runs)."""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from mamo.config import Config, MAPFConfig
from mamo.geometry import Circle, OBB, shape_distance, shapes_intersect
from mamo.mapf.agent import Agent
from mamo.mapf.cbs import cbs, find_conflict
from mamo.ngr import NGR
from mamo.em4m.priority import Features, log_priority
from mamo.robot.arm import Arm, ObstacleSet, tool_rotation
from mamo.scene import BOX, CYLINDER, IMMOVABLE, MOVABLE, Scene, SceneObject, default_shelf_parts, parse_scene_file
from mamo.world import World, tilt_deg

SCENES = Path(__file__).resolve().parents[2] / "archive/ros_src/em4m-main/dat/clutter_scenes"
needs_scenes = pytest.mark.skipif(not SCENES.exists(), reason="reference scenes not available")


def small_scene() -> Scene:
    objs = default_shelf_parts() + [
        SceneObject(6, CYLINDER, IMMOVABLE, 0.75, -0.5, 0.77 + 0.075, 0, 0, 0, 0.03, 0.03, 0.15, 0.3, 1.0, False),
        SceneObject(7, BOX, MOVABLE, 0.62, -0.5, 0.77 + 0.06, 0, 0, 0.3, 0.03, 0.04, 0.06, 0.3, 1.0, True),
    ]
    return Scene(objs, 6, "small")


# ------------------------------------------------------------------ scene
@needs_scenes
def test_parse_reference_scene() -> None:
    sc = parse_scene_file(SCENES / "5" / "plan_100001_SCENE.txt")
    assert sc.ooi_id == 8 and not sc.ooi.movable
    assert len(sc.shelf_parts) == 5 and len(sc.movables) == 5
    assert sc.shelf.floor_top == pytest.approx(0.77)
    cyl = sc.by_id(7)  # cylinder: z_size is the full height, resting on the floor
    assert cyl.shape == CYLINDER and cyl.bottom_z == pytest.approx(0.77, abs=1e-6)
    box = sc.by_id(6)  # box: z_size is a half extent
    assert box.bottom_z == pytest.approx(0.77, abs=1e-6)


def test_geometry() -> None:
    a, b = Circle(0, 0, 0.05), Circle(0.11, 0, 0.05)
    assert not shapes_intersect(a, b) and shape_distance(a, b) == pytest.approx(0.01)
    box = OBB(0.2, 0, 0.05, 0.02, math.pi / 4)
    assert shapes_intersect(box, Circle(0.2, 0.0, 0.01))
    assert not shapes_intersect(box, Circle(0.2, 0.1, 0.02))
    assert shape_distance(OBB(0, 0, 0.1, 0.1, 0), OBB(0.3, 0, 0.1, 0.1, 0)) == pytest.approx(0.1)


# ------------------------------------------------------------------ world
def test_world_is_static_without_contact() -> None:
    w = World(small_scene())
    before = w.object_states()
    for _ in range(500):
        mujoco.mj_step(w.model, w.data)
    after = w.object_states()
    for oid in before:
        assert np.linalg.norm(after[oid][:3] - before[oid][:3]) < 2e-3
        assert tilt_deg(after[oid][3:]) < 1.0
    assert np.allclose(w.arm_q(), w.cfg.robot.home_q, atol=1e-3)


def test_ik_round_trip_and_collisions() -> None:
    w = World(small_scene())
    arm = Arm(w)
    R = tool_rotation(np.array([1.0, 0.0]), math.radians(30))
    p = np.array([0.6, -0.4, 0.80])
    q = arm.ik(p, R, np.array(w.cfg.robot.home_q))
    assert q is not None
    pf, Rf = arm.fk(q)
    assert np.linalg.norm(pf - p) < 3e-3
    assert np.allclose(Rf, R, atol=0.05)
    assert arm.collision_free(np.array(w.cfg.robot.home_q), ObstacleSet.everything())
    # The tool inside the immovable cylinder must collide with it.
    q_in = arm.ik(np.array([0.75, -0.5, 0.85]), None, np.array(w.cfg.robot.home_q))
    assert q_in is not None and not arm.collision_free(q_in, ObstacleSet.static())


# -------------------------------------------------------------------- NGR
def test_ngr_overlap_map_matches_voxels() -> None:
    sc = small_scene()
    ngr = NGR(sc.shelf, 0.01)
    ngr.add_points(np.array([[0.62 + dx, -0.5, 0.80] for dx in np.linspace(-0.1, 0.1, 21)]))
    mov = sc.by_id(7)
    assert ngr.intersects(mov)
    om = ngr.overlap_map(mov)
    # Overlap map at the object's cell agrees with the direct voxel test.
    i = int(round(mov.x / 0.01)) - ngr.origin[0]
    j = int(round(mov.y / 0.01)) - ngr.origin[1]
    assert om[i, j]
    far = mov.copy()
    far.y = -0.3
    assert not ngr.intersects(far)
    assert not om[i, j + 20]


# ------------------------------------------------------------------- MAPF
def _toy_agent(oid: int, start: tuple[int, int], goal_cols: slice, shape: tuple[int, int] = (20, 20)) -> Agent:
    o = SceneObject(oid, CYLINDER, MOVABLE, 0, 0, 0.8, 0, 0, 0, 0.015, 0.015, 0.05, 0.1, 1.0, True)
    valid = np.ones(shape, dtype=bool)
    outside = np.zeros(shape, dtype=bool)
    outside[:, goal_cols] = True
    cost = np.full(shape, MAPFConfig().cost_mult, dtype=np.int64)
    return Agent(o, start, valid, outside, cost, Circle(0, 0, 0.015), 0.01, np.zeros(3, dtype=np.int64))


def test_cbs_resolves_crossing() -> None:
    # Two 3 cm discs swap sides of a corridor; their straight paths collide.
    a = _toy_agent(1, (10, 5), slice(15, 20))
    b = _toy_agent(2, (10, 15), slice(0, 5))
    agents = {1: a, 2: b}
    res = cbs(agents, time_limit=10.0)
    assert res is not None
    assert find_conflict(res.paths, agents)[1] is None
    assert a.outside[res.paths[1][-1]] and b.outside[res.paths[2][-1]]


def test_invalid_goal_is_excluded() -> None:
    a = _toy_agent(1, (10, 13), slice(15, 20))
    first = cbs({1: a}, time_limit=5.0).paths[1][-1]
    a.invalid_goals = {first}
    second = cbs({1: a}, time_limit=5.0).paths[1][-1]
    assert second != first and a.outside[second]


# --------------------------------------------------------------- priority
def test_priority_prefers_clearer_scenes() -> None:
    clear = log_priority(Features(0.0, 0, []))
    one = log_priority(Features(0.01, 1, [0.05]))
    three = log_priority(Features(0.05, 3, [0.3, 0.3, 0.3]))
    assert clear == math.inf and one > three
