"""Which cells of a generated environment can the xArm7 retrieve an object from?

For every cell, one object of the environment's first class stands alone on it and
PLANRETRIEVAL (grasp, lift, extract towards the robot) is run from the home pose. Used to
place each environment relative to the robot base (mamo/envs.py defaults).

    python scripts/env_reachability.py open_tabletop --front 0.15 0.2 0.25 --height 0.2 0.3
"""

from __future__ import annotations

import argparse
import itertools
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _cell_ok(args) -> tuple[str, tuple[int, int], bool]:
    env_name, kw, rname, cell, seed = args
    from mamo.envs import ENVIRONMENTS
    from mamo.retrieval import plan_retrieval
    from mamo.robot.arm import Arm, ObstacleSet
    from mamo.scene import IMMOVABLE, Scene, SceneObject, BOX, CYLINDER
    from mamo.world import World

    env = ENVIRONMENTS[env_name](**kw)
    r = env.region(rname)
    cls = env.classes[0]
    x, y = r.center(cell)
    sx, sy, sz = cls.size
    h = sz if cls.shape == CYLINDER else 2 * sz
    o = SceneObject(len(env.parts) + 1, cls.shape, IMMOVABLE, x, y, r.surface_z + h / 2, 0, 0, 0, sx, sy, sz,
                    0.2, 1.0, False)
    meta = {"env": env.name, "robot": {"base": list(env.base)}, "config": env.config}
    sc = Scene(list(env.parts) + [o], o.id, meta=meta)
    w = World(sc)
    arm = Arm(w)
    home = np.array(w.cfg.robot.home_q)
    if not arm.collision_free(home, ObstacleSet.everything()):
        return rname, cell, False
    plan = plan_retrieval(arm, w.cfg, sc.ooi, w.object_states(), ObstacleSet.static({o.id}), home,
                          np.random.default_rng(seed), sc.shelf)
    return rname, cell, plan is not None


def sweep(env_name: str, kw: dict, workers: int) -> dict:
    from mamo.envs import ENVIRONMENTS
    env = ENVIRONMENTS[env_name](**kw)
    regions = env.start_regions or [r.name for r in env.regions]
    jobs = [(env_name, kw, rn, c, 0) for rn in regions for c in env.region(rn).cells()]
    with ProcessPoolExecutor(workers) as ex:
        res = list(ex.map(_cell_ok, jobs))
    out = {}
    for rn in regions:
        cells = {c: ok for r, c, ok in res if r == rn}
        out[rn] = cells
    return out


def show(env_name: str, kw: dict, out: dict) -> float:
    from mamo.envs import ENVIRONMENTS
    env = ENVIRONMENTS[env_name](**kw)
    tot = sum(len(v) for v in out.values())
    ok = sum(sum(v.values()) for v in out.values())
    print(f"{env_name} {kw}: {ok}/{tot} cells reachable")
    for rn, cells in out.items():
        r = env.region(rn)
        print(f"  {rn} (rows top = back; columns left = +y)")
        for j in reversed(range(r.ny)):
            print("   " + " ".join("#" if (i, j) in r.blocked else ("o" if cells.get((i, j)) else ".")
                                   for i in range(r.nx)))
    return ok / max(tot, 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--front", type=float, nargs="+", default=[None])
    ap.add_argument("--height", type=float, nargs="+", default=[None])
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    for f, h in itertools.product(a.front, a.height):
        kw = {k: v for k, v in (("front", f), ("height", h)) if v is not None}
        show(a.env, kw, sweep(a.env, kw, a.workers))
