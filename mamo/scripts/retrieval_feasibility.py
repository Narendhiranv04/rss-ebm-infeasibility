"""How many reference scenes admit PLANRETRIEVAL(O_I) with the xArm7?

M4M/E-M4M assume the problem without movable objects is solvable (otherwise the
MAMO instance is unsolvable as specified). The reference scenes were generated
for the PR2, so this measures which of them remain valid for the xArm7.

    python scripts/retrieval_feasibility.py --level 5 --limit 30
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mamo.robot.arm import Arm, ObstacleSet  # noqa: E402
from mamo.retrieval import plan_retrieval  # noqa: E402
from mamo.scene import parse_scene_file  # noqa: E402
from mamo.world import World  # noqa: E402

SCENES = Path(__file__).resolve().parents[2] / "archive/ros_src/em4m-main/dat/clutter_scenes"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", default="5")
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()
    files = sorted((SCENES / args.level).glob("plan_*_SCENE.txt"))[: args.limit]
    ok = 0
    for f in files:
        sc = parse_scene_file(f)
        w = World(sc)
        arm = Arm(w)
        t = time.time()
        stats: dict[str, int] = {}
        p = plan_retrieval(arm, w.cfg, sc.ooi, w.object_states(), ObstacleSet.static({sc.ooi_id}),
                           np.array(w.cfg.robot.home_q), np.random.default_rng(0), sc.shelf, stats)
        ok += p is not None
        print(f"{f.stem}: {'ok ' if p else 'FAIL'} {time.time() - t:5.1f}s  ooi=({sc.ooi.x:.2f},{sc.ooi.y:.2f})  {stats or 'no grasp fits the gripper'}",
              flush=True)
    print(f"feasible {ok}/{len(files)}")


if __name__ == "__main__":
    main()
