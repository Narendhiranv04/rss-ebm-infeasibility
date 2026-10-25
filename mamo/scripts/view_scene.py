"""Show a scene in 2D (top view) and 3D (MuJoCo), optionally with its NGR.

    python scripts/view_scene.py --scene 10/plan_200003_SCENE.txt --out results/figures/200003
    python scripts/view_scene.py --scene 10/plan_200003_SCENE.txt --interactive   # MuJoCo viewer

Writes <out>_2d.png and <out>_3d.png. The NGR (red) is the volume swept by the
robot and the OoI along PLANRETRIEVAL with only immovable obstacles; movables
inside it are the ones that must be rearranged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mamo.config import Config  # noqa: E402
from mamo.context import MAMOContext  # noqa: E402
from mamo.scene import parse_scene_file  # noqa: E402

SCENES = ROOT.parent / "archive/ros_src/em4m-main/dat/clutter_scenes"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True, help="path relative to dat/clutter_scenes, or an absolute path")
    ap.add_argument("--out", help="output prefix (writes <out>_2d.png, <out>_3d.png)")
    ap.add_argument("--no-ngr", action="store_true", help="skip planning the retrieval for the NGR")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--interactive", action="store_true", help="open the MuJoCo viewer instead")
    args = ap.parse_args()

    path = Path(args.scene)
    scene = parse_scene_file(path if path.is_absolute() else SCENES / path)
    cfg = Config()
    ctx = MAMOContext(scene, cfg, seed=args.seed)
    ctx.world.reset()
    if args.interactive:
        import mujoco.viewer
        mujoco.viewer.launch(ctx.world.model, ctx.world.data)
        return

    ngr = None
    if not args.no_ngr:
        if ctx.setup_ngr():
            ngr = ctx.ngr
            blockers = ctx.blockers(ctx.world.object_states())
        else:
            print("no retrieval with only immovable obstacles: NGR undefined")
    ctx.world.reset()

    from mamo import viz
    import matplotlib.pyplot as plt
    import imageio

    out = Path(args.out or ROOT / "results/figures" / path.stem)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_mov = len(scene.movables)
    title = f"{path.stem}: {n_mov} movable, {len(scene.immovables) - 1} immovable, OoI = {scene.ooi_id}"
    if ngr is not None:
        title += f"; NGR blockers: {', '.join(map(str, blockers)) or 'none'}"
    fig, ax = plt.subplots(figsize=(7, 6.2))
    viz.draw_scene_2d(ax, scene, ngr=ngr, base=cfg.robot.base_pos, title=title)
    ax.legend(handles=viz.legend_handles()[:4], loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(f"{out}_2d.png", dpi=150)
    plt.close(fig)

    views = viz.Views(ctx.world, names=("iso", "front", "top"))
    rows = [views.frame()]
    if ngr is not None:
        views.set_ngr(ngr)
        rows.append(views.frame())
    imageio.imwrite(f"{out}_3d.png", np.concatenate(rows, axis=0))
    print(f"wrote {out}_2d.png and {out}_3d.png" + (" (bottom row: with NGR voxels)" if ngr is not None else ""))


if __name__ == "__main__":
    main()
