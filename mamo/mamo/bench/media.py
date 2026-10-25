"""Scene views, plan figures and execution videos for the app (cached on disk)."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np


def scene_views(scene_path: Path, out_prefix: Path, with_ngr: bool = True) -> tuple[Path, Path]:
    """2D top view and 3D views (iso / front / top) of a scene, with its NGR."""
    import imageio
    import matplotlib.pyplot as plt

    from mamo import viz
    from mamo.config import Config
    from mamo.context import MAMOContext
    from mamo.scene import parse_scene_file

    p2, p3 = Path(f"{out_prefix}_2d.png"), Path(f"{out_prefix}_3d.png")
    if p2.exists() and p3.exists():
        return p2, p3
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    scene = parse_scene_file(scene_path)
    ctx = MAMOContext(scene, Config(), seed=0)
    cfg = ctx.cfg  # with the scene's robot base
    ngr, blockers = None, []
    relocate = scene.task["type"] == "relocate"
    if with_ngr and not relocate and ctx.setup_ngr():
        ngr = ctx.ngr
        blockers = ctx.blockers(ctx.world.object_states())
    ctx.world.reset()
    if scene.meta.get("counts"):
        c = scene.meta["counts"]
        what = (f"{scene.env}, {scene.task['type']} ({scene.meta.get('difficulty', '')}): {c['movables']} movable, "
                f"{c['immovables']} immovable, target(s) {', '.join(map(str, scene.target_ids))}")
    else:
        what = f"{len(scene.movables)} movable, {len(scene.immovables) - 1} immovable, OoI {scene.ooi_id}"
    title = f"{Path(scene_path).stem}: {what}" + (
        "" if relocate else f"; NGR blockers: {', '.join(map(str, blockers)) or 'none'}" if ngr is not None
        else "; no retrieval without movables")
    if scene.meta.get("regions"):
        fig = viz.draw_env_2d(scene, ngr=ngr, base=cfg.robot.base_pos, title=title)
    else:
        fig, ax = plt.subplots(figsize=(7, 6.2))
        viz.draw_scene_2d(ax, scene, ngr=ngr, base=cfg.robot.base_pos, title=title)
        ax.legend(handles=viz.legend_handles()[:4], loc="upper left", fontsize=7, framealpha=0.9)
        fig.tight_layout()
    fig.savefig(p2, dpi=110)
    plt.close(fig)
    views = viz.Views(ctx.world, names=("iso", "front", "top"), size=(360, 480))
    rows = [views.frame()]
    if ngr is not None:
        views.set_ngr(ngr)
        rows.append(views.frame())
    views.renderer.close()
    imageio.imwrite(p3, np.concatenate(rows, axis=0))
    return p2, p3


def _load(sol_file: Path):
    with open(sol_file, "rb") as fh:
        return pickle.load(fh)


def plan_figure(sol_file: Path, out: Path) -> Path:
    """One top-view panel per rearrangement action, from a saved solution (replays it once)."""
    if out.exists():
        return out
    import matplotlib.pyplot as plt

    from mamo import viz
    from mamo.config import Config
    from mamo.execute import execute
    from mamo.scene import parse_scene_file
    from mamo.world import cfg_for_scene

    data = _load(sol_file)
    job, sol = data["job"], data["solution"]
    scene = parse_scene_file(job["scene_path"])
    cfg = cfg_for_scene(scene, Config())
    res = execute(scene, sol, cfg)
    fig = viz.solution_figure(scene, data.get("ngr"), sol.pushes, res.states, cfg.robot.base_pos,
                              f"{Path(job['scene']).stem} {job['method']}: {len(sol.pushes)} actions, planning "
                              f"{sol.stats.get('planning_time', 0):.0f} s, replay "
                              + ("succeeded" if res.success else f"failed ({res.reason})"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=100)
    plt.close(fig)
    return out


def save_gif(frames: list[np.ndarray], out: Path, fps: int = 12, max_frames: int = 360) -> Path:
    """Write RGB frames as a looping GIF (subsampled to at most ``max_frames``; the last frame kept)."""
    from PIL import Image
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]
    ims = [Image.fromarray(f).quantize(colors=128, method=Image.Quantize.FASTOCTREE) for f in frames]
    out.parent.mkdir(parents=True, exist_ok=True)
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=int(1000 / fps), loop=0)
    return out


def video(sol_file: Path, out: Path, fps: int = 25, size: tuple[int, int] = (360, 480), every: int = 1,
          info: str = "") -> Path:
    """Replay a saved solution and record iso + top views, captioned by stage.

    ``out`` ending in .gif writes a GIF (12 fps, frames subsampled), anything else a video.
    ``every``: keep every n-th recorded frame (the simulator records every 20 physics steps).
    """
    if out.exists():
        return out
    import imageio

    from mamo import viz
    from mamo.config import Config
    from mamo.execute import execute
    from mamo.scene import parse_scene_file

    data = _load(sol_file)
    job, sol = data["job"], data["solution"]
    scene = parse_scene_file(job["scene_path"])
    frames: list[np.ndarray] = []
    views = None
    label = ["start"]
    tag = job["method"] + (f"  {info}" if info else "")
    calls = [0]

    def record(world) -> None:
        nonlocal views
        calls[0] += 1
        if (calls[0] - 1) % every:
            return
        if views is None:
            views = viz.Views(world, names=("iso", "top"), size=size)
        frames.append(viz.caption(views.frame(), f"{tag}  {label[0]}"))

    res = execute(scene, sol, Config(), recorder=record, on_stage=lambda s: label.__setitem__(0, s))
    gif = out.suffix.lower() == ".gif"
    if frames:
        done = "goals reached" if scene.task["type"] == "relocate" else "OoI retrieved"
        frames.extend([viz.caption(frames[-1], f"{tag}  " + (done if res.success else res.reason))]
                      * (24 if gif else fps))
    if views is not None:
        views.renderer.close()
    out.parent.mkdir(parents=True, exist_ok=True)
    if gif:
        save_gif(frames, out, fps=12)
    else:
        imageio.mimsave(out, frames, fps=fps, macro_block_size=8)
    return out


def scene_gif(scene_path: Path, out: Path, size: tuple[int, int] = (420, 560), seed: int = 0,
              title: str = "") -> dict:
    """Animated view of a retrieve problem and its task.

    1. the camera circles the scene (OoI yellow, movables blue, immovables red);
    2. the OoI's retrieval planned with only the immovables as obstacles is played back
       kinematically, with its NGR (purple voxels) and the movables inside it (orange): the
       objects a planner has to move out of the way. Returns {"blockers": [...], "retrieval": bool}.
    """
    import mujoco

    from mamo import viz
    from mamo.config import Config
    from mamo.context import MAMOContext
    from mamo.scene import parse_scene_file

    scene = parse_scene_file(scene_path)
    ctx = MAMOContext(scene, Config(), seed=seed)
    w = ctx.world
    m, d = w.model, w.data
    ok = ctx.setup_ngr()
    blockers = ctx.blockers(w.object_states()) if ok else []
    w.reset()
    head = title or Path(scene_path).stem
    task = (f"Task: retrieve the yellow OoI ({scene.ooi_id}); {len(blockers)} movable(s) in its way: "
            f"{', '.join(map(str, blockers)) or '-'}") if ok else "Task: retrieve the yellow OoI (no retrieval found)"
    r = mujoco.Renderer(m, *size, max_geom=20000)
    m.light_castshadow[:] = 0
    m.vis.headlight.ambient[:] = 0.35
    m.vis.headlight.diffuse[:] = 0.5
    cam = viz.make_camera("iso", scene, ctx.cfg.robot.base_pos)
    az0 = cam.azimuth
    frames = []

    def shot(text: str, voxels=None) -> None:
        r.update_scene(d, cam)
        if voxels is not None:
            scn = r.scene
            for pnt in voxels:
                if scn.ngeom >= scn.maxgeom:
                    break
                mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.full(3, half), pnt,
                                    np.eye(3).flatten(), np.array([0.6, 0.2, 0.9, 0.12], dtype=np.float32))
                scn.ngeom += 1
        frames.append(_two_line(viz, r.render(), head, text))

    # 1. orbit
    for k in range(36):
        cam.azimuth = az0 - 60 + 120 * (0.5 - 0.5 * np.cos(2 * np.pi * k / 36))
        shot(task)
    cam.azimuth = az0
    if ok:
        # 2. retrieval playback with the NGR and the blockers highlighted
        idx = np.argwhere(ctx.ngr.grid[::2, ::2, ::2]) * 2
        vox = (idx + ctx.ngr.origin) * ctx.ngr.res
        half = ctx.ngr.res
        for oid in blockers:
            m.geom_rgba[w.obj_geom[oid]] = [1.0, 0.5, 0.0, 1.0]
        ooi_q = w.obj_qadr[scene.ooi_id]
        grip = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "robot/xarm_gripper_base_link")
        rel = None
        for seg in ctx.initial_retrieval.segments:
            path = seg.path
            step = max(1, len(path) // 25)
            for q in list(path[::step]) + [path[-1]]:
                w.set_arm_q(q)
                mujoco.mj_kinematics(m, d)
                if seg.carrying:
                    if rel is None:  # grasp: fix the OoI in the gripper frame
                        R1 = d.xmat[grip].reshape(3, 3)
                        rel = (R1.T @ (d.qpos[ooi_q:ooi_q + 3] - d.xpos[grip]), d.qpos[ooi_q + 3:ooi_q + 7].copy(),
                               d.xquat[grip].copy())
                    R1 = d.xmat[grip].reshape(3, 3)
                    d.qpos[ooi_q:ooi_q + 3] = d.xpos[grip] + R1 @ rel[0]
                    dq = np.zeros(4)
                    inv = np.zeros(4)
                    mujoco.mju_negQuat(inv, rel[2])
                    mujoco.mju_mulQuat(dq, d.xquat[grip], inv)
                    qq = np.zeros(4)
                    mujoco.mju_mulQuat(qq, dq, rel[1])
                    d.qpos[ooi_q + 3:ooi_q + 7] = qq
                mujoco.mj_forward(m, d)
                shot(f"Retrieval to make possible ({seg.name}); purple = swept volume (NGR), orange = in the way", vox)
        for _ in range(12):
            shot(f"Retrieval to make possible; orange = must be moved first ({len(blockers)})", vox)
    r.close()
    save_gif(frames, out, fps=10)
    return {"blockers": blockers, "retrieval": ok}


def _two_line(viz, img: np.ndarray, head: str, text: str) -> np.ndarray:
    """Title at the top, task text at the bottom."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=14)
    for y, t in ((6, head), (im.height - 24, text)):
        x0, y0, x1, y1 = dr.textbbox((8, y), t, font=font)
        dr.rectangle((0, y0 - 4, min(im.width, x1 + 8), y1 + 4), fill=(255, 255, 255))
        dr.text((8, y), t, fill=(0, 0, 0), font=font)
    return np.asarray(im)
