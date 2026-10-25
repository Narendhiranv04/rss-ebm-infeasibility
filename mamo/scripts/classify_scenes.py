"""Label reference scenes Easy / Medium / Hard as in the E-M4M paper.

Difficulty = number of movable objects overlapping the initial NGR V(gamma_OoI):
1 -> Easy, 2 -> Medium, >2 -> Hard (0 = trivially solved, 'infeasible' = no
PLANRETRIEVAL(O_I) for the xArm7). Writes a CSV used by the benchmark runner.

    python scripts/classify_scenes.py --levels 5 10 15 --out results/scene_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mamo.ngr import compute_ngr  # noqa: E402
from mamo.retrieval import plan_retrieval  # noqa: E402
from mamo.robot.arm import Arm, ObstacleSet  # noqa: E402
from mamo.scene import parse_scene_file  # noqa: E402
from mamo.world import World  # noqa: E402

SCENES = ROOT.parent / "archive/ros_src/em4m-main/dat/clutter_scenes"


def label(n: int | None) -> str:
    if n is None:
        return "infeasible"
    return {0: "trivial", 1: "easy", 2: "medium"}.get(n, "hard")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="+", default=["5", "10", "15"])
    ap.add_argument("--limit", type=int, default=10_000)
    ap.add_argument("--out", default=str(ROOT / "results/scene_labels.csv"))
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for level in args.levels:
        for f in sorted((SCENES / level).glob("plan_*_SCENE.txt"))[: args.limit]:
            t = time.time()
            sc = parse_scene_file(f)
            w = World(sc)
            arm = Arm(w)
            plan = plan_retrieval(arm, w.cfg, sc.ooi, w.object_states(), ObstacleSet.static({sc.ooi_id}),
                                  np.array(w.cfg.robot.home_q), np.random.default_rng(0), sc.shelf)
            blockers: list[int] | None = None
            if plan is not None:
                ngr = compute_ngr(arm, plan, sc.shelf, sc.ooi_id, w.object_states())
                blockers = [o.id for o in sc.movables if ngr.intersects(o)]
            n = None if blockers is None else len(blockers)
            rows.append({"level": level, "scene": f.name, "movables": len(sc.movables),
                         "blockers": "" if blockers is None else " ".join(map(str, blockers)),
                         "difficulty": label(n)})
            print(f"{level:>2} {f.stem}: {label(n):10s} blockers={blockers}  ({time.time() - t:.1f}s)", flush=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    from collections import Counter
    print(Counter((r["level"], r["difficulty"]) for r in rows))


if __name__ == "__main__":
    main()
