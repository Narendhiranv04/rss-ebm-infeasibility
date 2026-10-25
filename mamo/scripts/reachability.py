"""Choose the xArm7 base placement in front of the reference shelf.

The PR2 in the papers reaches the whole shelf; the xArm7 is smaller, so its
base position matters. For each candidate base, and for each cell of a grid
over the shelf interior, we test what the planners will ask for:

* push in each of 8 directions: a collision-free IK solution (empty shelf) at
  the push start (5 cm behind the cell) and at the push end (10 cm past it),
  with any of the push planner's tool orientations;
* grasp: a collision-free pre-grasp IK solution for some approach yaw in
  +-60 deg and pitch 0-30 deg, as in retrieval.py.

    python scripts/reachability.py --workers 6
"""

from __future__ import annotations

import argparse
import itertools
import math
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mamo.config import Config  # noqa: E402
from mamo.push import push_rotation  # noqa: E402
from mamo.robot.arm import Arm, ObstacleSet, tool_rotation  # noqa: E402
from mamo.scene import Scene, default_shelf_parts  # noqa: E402
from mamo.world import World  # noqa: E402

DIRS = {f"{a}": (math.cos(math.radians(a)), math.sin(math.radians(a))) for a in range(0, 360, 45)}


def empty_scene() -> Scene:
    parts = default_shelf_parts()
    # A dummy OoI far outside the workspace keeps Scene well-formed.
    from mamo.scene import IMMOVABLE, BOX, SceneObject
    dummy = SceneObject(99, BOX, IMMOVABLE, 3.0, 3.0, 0.05, 0, 0, 0, 0.03, 0.03, 0.05, 0.1, 0.8, False)
    return Scene(parts + [dummy], 99, "empty")


def coverage(job: tuple[tuple[float, float, float], tuple | None]) -> tuple[tuple, dict[str, float] | None]:
    base, alts = job
    rng = np.random.default_rng(0)
    cfg = Config()
    cfg.robot.base_pos = base
    if alts:
        cfg.push.tool_alternatives = alts
    world = World(empty_scene(), cfg)
    arm = Arm(world)
    obs = ObstacleSet.static()
    home = np.array(cfg.robot.home_q)
    if not arm.collision_free(home, obs):
        return base, None
    shelf = world.scene.shelf
    xs = np.linspace(shelf.x_min + 0.05, shelf.x_max - 0.06, 6)
    ys = np.linspace(shelf.y_min + 0.08, shelf.y_max - 0.08, 6)
    pc, rc = cfg.push, cfg.retrieval
    z_push = shelf.floor_top + pc.height_above_shelf
    seeds = [home]

    def push_ok(p: np.ndarray, d: np.ndarray) -> bool:
        for alt in pc.tool_alternatives:
            R = push_rotation(d, alt)
            q0 = arm.ik_multi(np.append(p - 0.05 * d, z_push), R, seeds, rng, restarts=3, obstacles=obs)
            if q0 is None:
                continue
            if arm.ik_multi(np.append(p + 0.10 * d, z_push), R, [q0], rng, restarts=2, obstacles=obs) is not None:
                return True
        return False

    def grasp_ok(p: np.ndarray) -> bool:
        for yaw, pitch in itertools.product((0, -30, 30, -60, 60), (15, 0, 30)):
            d = np.array([math.cos(math.radians(yaw)), math.sin(math.radians(yaw))])
            R = tool_rotation(d, math.radians(pitch))
            pg = np.append(p - 0.03 * d, shelf.floor_top + 0.06)
            if arm.ik_multi(pg - rc.pregrasp_backoff * R[:, 2], R, seeds, rng, restarts=2, obstacles=obs) is not None:
                return True
        return False

    n = len(xs) * len(ys)
    res = {k: 0.0 for k in DIRS} | {"grasp": 0.0}
    for x, y in itertools.product(xs, ys):
        p = np.array([x, y])
        for k, d in DIRS.items():
            res[k] += push_ok(p, np.array(d)) / n
        res["grasp"] += grasp_ok(p) / n
    return base, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--x", type=float, nargs="+", default=[0.30, 0.36, 0.42])
    ap.add_argument("--y", type=float, nargs="+", default=[-0.50])
    ap.add_argument("--z", type=float, nargs="+", default=[0.43, 0.53, 0.63, 0.73])
    ap.add_argument("--alts", nargs="+", help="push tool alternatives as pitch:flip:yaw, e.g. 30:0:0 30:0:90")
    args = ap.parse_args()
    alts = tuple((float(p), f == "1", float(y)) for p, f, y in (a.split(":") for a in args.alts)) if args.alts else None
    bases = [(b, alts) for b in itertools.product(args.x, args.y, args.z)]
    t = time.time()
    rows = []
    with Pool(args.workers) as pool:
        for base, r in pool.imap_unordered(coverage, bases):
            if r is None:
                print(f"base {base}: home pose in collision", flush=True)
                continue
            push = np.mean([r[k] for k in DIRS])
            rows.append((push, r["grasp"], base, r))
            print(f"base x={base[0]:.2f} y={base[1]:.2f} z={base[2]:.2f}  push={push:.2f} grasp={r['grasp']:.2f}  "
                  + " ".join(f"{k}:{r[k]:.2f}" for k in DIRS) + f"  ({time.time() - t:.0f}s)", flush=True)
    rows.sort(key=lambda r: -(r[0] + r[1]))
    print("best", rows[0][2], f"push={rows[0][0]:.2f} grasp={rows[0][1]:.2f}")


if __name__ == "__main__":
    main()
