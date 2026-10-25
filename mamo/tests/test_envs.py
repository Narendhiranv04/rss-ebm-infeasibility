"""Generated environments (mamo.envs): scene format, surfaces, task checks."""

from __future__ import annotations

import numpy as np
import pytest

from mamo.envs import DIFFICULTIES, ENVIRONMENTS, generate
from mamo.scene import footprint_overlap, parse_scene_file, write_scene_file


@pytest.mark.parametrize("env", list(ENVIRONMENTS))
def test_generate_round_trip(env, tmp_path):
    for task in ("retrieve", "relocate"):
        sc = generate(env, task, "hard", seed=3)
        f = tmp_path / f"{sc.name}.txt"
        write_scene_file(sc, f)
        back = parse_scene_file(f)
        assert back.meta == sc.meta and back.target_ids == sc.target_ids
        assert back.task["type"] == task
        objs = [o for o in back.objects if not o.is_shelf]
        # on the 1 cm MAPF lattice, and never touching (M4M rounds poses to that lattice)
        for o in objs:
            assert abs(o.x * 100 - round(o.x * 100)) < 1e-6 and abs(o.y * 100 - round(o.y * 100)) < 1e-6
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                if abs(a.bottom_z - b.bottom_z) < 0.01:
                    assert not footprint_overlap(a, b, margin=0.0009)


def test_generate_is_deterministic():
    a, b = generate("access19", "relocate", "medium", 7), generate("access19", "relocate", "medium", 7)
    assert [(o.x, o.y, o.mass) for o in a.objects] == [(o.x, o.y, o.mass) for o in b.objects]


def test_difficulty_counts_grow():
    for env in ENVIRONMENTS:
        m = [np.mean([generate(env, "retrieve", d, s).meta["counts"]["movables"] for s in range(4)])
             for d in DIFFICULTIES]
        assert m[0] < m[1] < m[2], (env, m)


def test_retrieve_problem_lives_on_the_ooi_surface():
    sc = generate("multi_level_shelf", "retrieve", "hard", 1)
    ooi = sc.ooi
    assert abs(ooi.bottom_z - sc.shelf.floor_top) < 1e-6
    for o in sc.movables:  # only objects on the OoI's deck are agents of the MAPF problem
        assert abs(o.bottom_z - sc.shelf.floor_top) < 0.02
    off = [o for o in sc.objects if not o.is_shelf and o.movable and not sc.is_active(o.id)]
    assert all(o in sc.immovables for o in off)


def test_reference_scene_unchanged():
    from mamo.scene import generate_scene
    sc = generate_scene(6, seed=2)
    assert sc.task == {"type": "retrieve", "targets": [sc.ooi_id]}
    assert len(sc.movables) == sum(o.movable for o in sc.objects if not o.is_shelf)
    assert sc.shelf.floor_top == pytest.approx(0.77) and sc.shelf.ceiling == pytest.approx(1.09)


def test_goal_errors():
    from mamo.config import Config
    from mamo.execute import goal_errors
    from mamo.world import World
    sc = generate("open_tabletop", "relocate", "easy", 0)
    w = World(sc)
    errs = goal_errors(w, sc, Config())
    assert len(errs) == len(sc.task["targets"])  # nothing moved yet
    for g in sc.task["targets"]:
        pos, quat = w.object_pose(g["id"])
        w.set_object_pose(g["id"], np.array(g["goal"][:3]) + [0.005, 0, 0], quat)
    assert goal_errors(w, sc, Config()) == []
