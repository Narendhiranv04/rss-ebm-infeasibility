"""Plan a scene with M4M or E-M4M, replay the solution, and save a video and a 2D plan figure.

    python scripts/render_solution.py --scene 10/plan_200003_SCENE.txt --algo em4m --out results/videos/200003_em4m

Writes <out>.mp4 (iso and top views side by side, captioned by stage),
<out>_plan.png (one top-view panel per push) and <out>.pkl (the solution, so
--load can re-render it without planning again).
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import imageio
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mamo import viz  # noqa: E402
from mamo.config import Config  # noqa: E402
from mamo.context import MAMOContext  # noqa: E402
from mamo.em4m.planner import em4m  # noqa: E402
from mamo.execute import execute  # noqa: E402
from mamo.m4m.planner import m4m  # noqa: E402
from mamo.scene import parse_scene_file  # noqa: E402

SCENES = ROOT.parent / "archive/ros_src/em4m-main/dat/clutter_scenes"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="path relative to dat/clutter_scenes, or an absolute path")
    ap.add_argument("--algo", choices=["m4m", "em4m"], default="em4m")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, help="output prefix")
    ap.add_argument("--load", action="store_true", help="re-render <out>.pkl instead of planning")
    ap.add_argument("--fps", type=int, default=25)
    args = ap.parse_args()

    path = Path(args.scene)
    scene = parse_scene_file(path if path.is_absolute() else SCENES / path)
    cfg = Config()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.load:
        with open(f"{out}.pkl", "rb") as f:
            saved = pickle.load(f)
        sol, ngr = saved["solution"], saved["ngr"]
    else:
        ctx = MAMOContext(scene, cfg, seed=args.seed)
        sol = (em4m if args.algo == "em4m" else m4m)(ctx)
        ngr = ctx.ngr
        with open(f"{out}.pkl", "wb") as f:
            pickle.dump({"scene": str(path), "algo": args.algo, "seed": args.seed, "solution": sol, "ngr": ngr}, f)
    print(f"{args.algo}: solved={sol.solved} pushes={len(sol.pushes)} "
          f"planning={sol.stats['planning_time']:.1f}s {sol.reason}")
    if not sol.solved:
        return

    frames: list[np.ndarray] = []
    views = None
    label = ["start"]

    def record(world) -> None:
        nonlocal views
        if views is None:
            views = viz.Views(world, names=("iso", "top"))
        frames.append(viz.caption(views.frame(), f"{args.algo.upper()}  {label[0]}"))

    def on_stage(text: str) -> None:
        label[0] = text

    # 20 sim steps of 2 ms per frame -> real time at 25 fps.
    res = execute(scene, sol, cfg, recorder=record, on_stage=on_stage)
    print(f"execution: success={res.success} {res.reason}")
    if frames:
        end = viz.caption(frames[-1], f"{args.algo.upper()}  " + ("OoI retrieved" if res.success else res.reason))
        frames.extend([end] * args.fps)  # hold the last frame for a second
    imageio.mimsave(f"{out}.mp4", frames, fps=args.fps, macro_block_size=8)
    print(f"wrote {len(frames)} frames to {out}.mp4")

    import matplotlib.pyplot as plt
    fig = viz.solution_figure(scene, ngr, sol.pushes, res.states, cfg.robot.base_pos,
                              f"{path.stem} {args.algo.upper()}: {len(sol.pushes)} pushes, "
                              f"planning {sol.stats['planning_time']:.0f} s, execution "
                              + ("succeeded" if res.success else f"failed ({res.reason})"))
    fig.savefig(f"{out}_plan.png", dpi=130)
    plt.close(fig)
    print(f"wrote {out}_plan.png")


if __name__ == "__main__":
    main()
