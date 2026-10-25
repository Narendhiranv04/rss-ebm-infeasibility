"""Scene and solution figures: 2D top views (matplotlib) and 3D views (MuJoCo offscreen)."""

from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patches  # noqa: E402

from mamo.geometry import OBB, Circle, footprint  # noqa: E402
from mamo.ngr import NGR  # noqa: E402
from mamo.scene import Scene, Shelf  # noqa: E402
from mamo.world import World, quat_to_rpy  # noqa: E402

COLORS = {"ooi": "#f2b600", "movable": "#2f5fd0", "immovable": "#c03030", "shelf": "#8a8a90"}


# ------------------------------------------------------------------------ 2D
def _patch(shape, **kw) -> patches.Patch:
    if isinstance(shape, Circle):
        return patches.Circle((shape.cx, shape.cy), shape.r, **kw)
    assert isinstance(shape, OBB)
    c, s = math.cos(shape.yaw), math.sin(shape.yaw)
    corner = np.array([shape.cx, shape.cy]) - np.array([[c, -s], [s, c]]) @ np.array([shape.hx, shape.hy])
    return patches.Rectangle(corner, 2 * shape.hx, 2 * shape.hy, angle=math.degrees(shape.yaw), **kw)


def _on(sh: Shelf, z_bottom: float) -> bool:
    return abs(z_bottom - sh.floor_top) < 0.03


def draw_scene_2d(ax, scene: Scene, states: dict[int, np.ndarray] | None = None, ngr: NGR | None = None,
                  base: tuple[float, float, float] | None = None, title: str = "", labels: bool = True,
                  ghost: dict[int, np.ndarray] | None = None, surface: Shelf | None = None) -> None:
    """Top view of one surface (default: the OoI's): walls, objects on it (at ``states`` if
    given), NGR footprint, robot base, and for generated scenes the grid cells and goals.

    ``ghost`` draws objects at other states as dashed outlines (e.g. before a push).
    """
    sh = surface or scene.shelf
    multi = len(scene.surfaces) > 1 and bool(scene.meta.get("regions"))
    ax.add_patch(patches.Rectangle((sh.x_min, sh.y_min), sh.x_max - sh.x_min, sh.y_max - sh.y_min,
                                   fc="#f4f1ea", ec="none", zorder=0))
    for o in scene.shelf_parts:
        # side and back walls only (floor and ceiling boards cover the whole interior)
        if not (o.top_z > sh.floor_top + 1e-6 and o.bottom_z < sh.ceiling - 1e-6):
            continue
        ax.add_patch(_patch(footprint(o), fc=COLORS["shelf"], ec="none", zorder=1))
    for r in scene.meta.get("regions", []):  # grid cells of generated environments
        if abs(r["z"] - sh.floor_top) > 1e-3 or not sh.contains_xy(r["x0"] + 1e-3, r["yc"]):
            continue
        pc, pr = r["pitch"]
        blocked = {tuple(c) for c in r["blocked"]}
        for i in range(r["nx"]):
            for j in range(r["ny"]):
                cx, cy = r["x0"] + (j + 0.5) * pr, r["yc"] + ((r["nx"] - 1) / 2 - i) * pc
                ax.add_patch(patches.Rectangle((cx - pr / 2, cy - pc / 2), pr, pc, fc="#ddd" if (i, j) in blocked
                                               else "none", ec="#c8c4b8", lw=0.5, zorder=0.5))
    if scene.task["type"] == "relocate":
        for g in scene.task["targets"]:
            if not _on(sh, g["goal"][2] - scene.by_id(g["id"]).half_height):
                continue
            o = scene.by_id(g["id"])
            ax.add_patch(_patch(footprint(o, g["goal"][0], g["goal"][1], g["goal"][3]), fc="none",
                                ec=COLORS["ooi"], ls="--", lw=1.6, zorder=3))
            if labels:
                ax.text(g["goal"][0], g["goal"][1], f"goal {g['id']}", ha="center", va="center", fontsize=6,
                        color="#8a6a00", zorder=3)
    if ngr is not None:
        cols = ngr.column_mask(sh.floor_top, sh.ceiling)
        x0, y0 = (ngr.origin[:2] - 0.5) * ngr.res
        img = np.ma.masked_where(~cols.T, ngr.grid.sum(axis=2).T.astype(float))
        ax.imshow(img, origin="lower", cmap="Reds", alpha=0.45, vmin=0, vmax=max(1, img.max()),
                  extent=(x0, x0 + cols.shape[0] * ngr.res, y0, y0 + cols.shape[1] * ngr.res), zorder=2)
    for o in scene.objects:
        if o.is_shelf:
            continue
        z = float(states[o.id][2]) if states is not None and o.id in states else o.z
        if multi and not (_on(sh, z - o.half_height) and sh.contains_xy(*(states[o.id][:2] if states is not None
                                                                          and o.id in states else (o.x, o.y)))):
            continue  # on another surface
        kind = "ooi" if o.id in scene.target_ids else "movable" if o.movable else "immovable"
        if ghost is not None and o.id in ghost:
            s = ghost[o.id]
            g = footprint(o, float(s[0]), float(s[1]), quat_to_rpy(s[3:7])[2])
            ax.add_patch(_patch(g, fc="none", ec=COLORS[kind], ls="--", lw=1.0, alpha=0.7, zorder=3))
        x, y, yaw = o.x, o.y, o.yaw
        if states is not None and o.id in states:
            s = states[o.id]
            x, y, yaw = float(s[0]), float(s[1]), quat_to_rpy(s[3:7])[2]
        ax.add_patch(_patch(footprint(o, x, y, yaw), fc=COLORS[kind], ec="black", lw=0.6, alpha=0.85, zorder=4))
        if labels:
            ax.text(x, y, str(o.id), ha="center", va="center", fontsize=7, zorder=5,
                    color="black" if kind == "ooi" else "white", fontweight="bold")
    if base is not None:
        ax.add_patch(patches.Circle(base[:2], 0.06, fc="#555", ec="black", zorder=4))
        ax.text(base[0], base[1] - 0.09, "xArm7", ha="center", va="top", fontsize=7)
    xmin = min(sh.x_min, base[0] - 0.08 if base else sh.x_min) - 0.05
    xmax = max(sh.x_max, base[0] + 0.08 if base else sh.x_max) + 0.05
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(sh.y_min - 0.05, sh.y_max + 0.05)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, into the shelf)")
    ax.set_ylabel("y (m)")
    if title:
        ax.set_title(title, fontsize=9)


def env_surfaces(scene: Scene) -> list[Shelf]:
    """Surfaces worth a panel: those with grid cells (generated scenes), else the OoI's."""
    regions = scene.meta.get("regions", [])
    out = [s for s in scene.surfaces if any(abs(r["z"] - s.floor_top) < 1e-3 and s.contains_xy(r["x0"] + 1e-3, r["yc"])
                                            for r in regions)]
    return sorted(out, key=lambda s: (-s.floor_top, -s.x_min)) or [scene.shelf]


def draw_env_2d(scene: Scene, states: dict[int, np.ndarray] | None = None, ngr: NGR | None = None,
                base=None, title: str = "") -> plt.Figure:
    """One top-view panel per surface (top deck first); the NGR on the OoI's surface."""
    surfs = env_surfaces(scene)
    names = {}
    for r in scene.meta.get("regions", []):
        for s in surfs:
            if abs(r["z"] - s.floor_top) < 1e-3 and s.contains_xy(r["x0"] + 1e-3, r["yc"]):
                names.setdefault(id(s), []).append(r["name"])
    n = len(surfs)
    ncol = min(n, 2)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 5.0 * nrow + 0.5), squeeze=False)
    for k, s in enumerate(surfs):
        ax = axes[k // ncol][k % ncol]
        label = " + ".join(names.get(id(s), [])) or "shelf"
        draw_scene_2d(ax, scene, states, ngr if s is scene.shelf else None, base,
                      f"{label} (z = {s.floor_top:.2f} m)", surface=s)
    for k in range(n, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.legend(handles=legend_handles()[:4] + ([patches.Patch(fc="none", ec=COLORS["ooi"], ls="--", label="goal")]
                                               if scene.task["type"] == "relocate" else []),
               loc="lower center", ncol=5, fontsize=8, frameon=False)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    return fig


def draw_push(ax, action, color: str = "#1a9850") -> None:
    """Object waypoints of a push (solid) and the tool path (dotted)."""
    w = np.asarray(action.waypoints)
    ax.plot(w[:, 0], w[:, 1], "-", color=color, lw=2, zorder=6)
    ax.annotate("", xy=w[-1], xytext=w[-2], zorder=6,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
    t = np.asarray(action.tool_points)
    ax.plot(t[:, 0], t[:, 1], ":", color="black", lw=1.2, zorder=6)
    ax.plot(t[0, 0], t[0, 1], "x", color="black", ms=5, zorder=6)


def legend_handles() -> list:
    return [patches.Patch(fc=COLORS["ooi"], ec="black", label="object of interest"),
            patches.Patch(fc=COLORS["movable"], ec="black", label="movable"),
            patches.Patch(fc=COLORS["immovable"], ec="black", label="immovable"),
            patches.Patch(fc="#e06060", alpha=0.45, label="NGR (retrieval sweep)"),
            matplotlib.lines.Line2D([], [], color="#1a9850", lw=2, label="object path of a push"),
            matplotlib.lines.Line2D([], [], color="black", ls=":", label="tool path (x = start)")]


def solution_figure(scene: Scene, ngr: NGR | None, pushes: list, states: list[dict[int, np.ndarray]],
                    base, title: str) -> plt.Figure:
    """One panel per push (object paths on the state before it) plus the final state."""
    if pushes and all(getattr(a, "kind", "push") == "motion" for a in pushes):
        return motion_figure(scene, ngr, pushes, states, base, title)
    n = len(pushes) + 1
    ncol = min(n, 3)
    nrow = math.ceil(n / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.4 * nrow + 0.6), squeeze=False)
    for k in range(nrow * ncol):
        ax = axes[k // ncol][k % ncol]
        if k < len(pushes):
            st = states[k] if k < len(states) else None
            a = pushes[k]
            if getattr(a, "kind", "push") == "pick_place":
                what = f"pick-and-place {k + 1}: object {a.obj_id}"
            else:
                what = f"push {k + 1}: object {a.obj_id}" + (" (side face)" if a.info.get("tool", (0, 0, 0))[2] else "")
            draw_scene_2d(ax, scene, st, ngr, base, what)
            draw_push(ax, pushes[k])
        elif k == len(pushes):
            st = states[-1] if states else None
            draw_scene_2d(ax, scene, st, ngr, base, "after all actions" if pushes else "initial scene",
                          ghost=states[0] if pushes and states else None)
        else:
            ax.axis("off")
    fig.legend(handles=legend_handles(), loc="lower center", ncol=6, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))
    return fig


def motion_figure(scene: Scene, ngr: NGR | None, motions: list, states: list[dict[int, np.ndarray]],
                  base, title: str) -> plt.Figure:
    """Arm-motion solutions (RRT, KPIECE, SELSIM): the TCP path over the initial scene, with the
    motions that touched movable objects in green, and the final state."""
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), squeeze=False)
    n_sim = sum(a.simulated for a in motions)
    draw_scene_2d(axes[0][0], scene, states[0] if states else None, ngr, base,
                  f"arm path: {len(motions)} motions, {n_sim} touching movables")
    for a in motions:
        t = np.asarray(a.tool_points)
        if len(t):
            axes[0][0].plot(t[:, 0], t[:, 1], "-" if a.simulated else ":", lw=2 if a.simulated else 1.2,
                            color="#1a9850" if a.simulated else "black", zorder=6)
    draw_scene_2d(axes[0][1], scene, states[-1] if states else None, ngr, base, "after all motions",
                  ghost=states[0] if states else None)
    fig.legend(handles=legend_handles()[:4] + [
        matplotlib.lines.Line2D([], [], color="black", ls=":", label="TCP path"),
        matplotlib.lines.Line2D([], [], color="#1a9850", lw=2, label="TCP path touching movables")],
        loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0.08, 1, 0.95))
    return fig


# ------------------------------------------------------------------------ 3D
CAMERAS = {
    # name: (lookat, distance, azimuth, elevation); azimuth 0 looks along +x (into the shelf)
    "front": ((0.80, -0.5, 0.86), 1.45, 0.0, -32.0),     # over the robot's shoulder
    "iso": ((0.78, -0.5, 0.86), 1.55, 40.0, -38.0),      # from the robot's right
    "top": ((0.70, -0.5, 0.85), 1.30, 90.0, -89.9),      # oriented like the 2D plots: x right, y up
}


def make_camera(name: str, scene: Scene | None = None, base=None) -> mujoco.MjvCamera:
    lookat, dist, az, el = CAMERAS[name]
    if scene is not None and scene.meta.get("regions"):
        # generated environments (mamo.envs): frame the static parts and the robot base
        lo = np.min([[o.x - o.x_size, o.y - o.y_size, o.z - o.z_size] for o in scene.shelf_parts], axis=0)
        hi = np.max([[o.x + o.x_size, o.y + o.y_size, o.z + o.z_size] for o in scene.shelf_parts], axis=0)
        if base is not None:
            lo[:2], hi[:2] = np.minimum(lo[:2], np.array(base[:2]) - 0.1), np.maximum(hi[:2], np.array(base[:2]) + 0.1)
        lo[2] = max(lo[2], scene.surfaces[0].floor_top - 0.1)
        lookat = 0.5 * (lo + hi)
        dist = 0.9 * float(np.max(hi - lo)) + 0.6
    cam = mujoco.MjvCamera()
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = dist, az, el
    return cam


class Views:
    """Offscreen renderer for several fixed cameras; hides the shelf ceiling in the top view."""

    def __init__(self, world: World, names=("iso", "top"), size=(480, 640)) -> None:
        self.world = world
        self.names = names
        self.renderer = mujoco.Renderer(world.model, *size, max_geom=20000)
        self.cams = {n: make_camera(n, world.scene, world.cfg.robot.base_pos) for n in names}
        m = world.model
        self.ceiling = [g for g in range(m.ngeom) if (m.geom(g).name or "").startswith("shelf_")
                        and m.geom_pos[g][2] - m.geom_size[g][2] >= world.scene.shelf.ceiling - 1e-6]
        self.voxels: np.ndarray | None = None
        # The ceiling board would shadow the whole shelf interior: light it evenly instead.
        m.light_castshadow[:] = 0
        m.vis.headlight.ambient[:] = 0.35
        m.vis.headlight.diffuse[:] = 0.5

    def set_ngr(self, ngr: NGR | None, stride: int = 2) -> None:
        """Draw NGR voxels (subsampled every ``stride`` cells) as translucent red cubes."""
        if ngr is None:
            self.voxels = None
            return
        idx = np.argwhere(ngr.grid[::stride, ::stride, ::stride]) * stride
        self.voxels = (idx + ngr.origin) * ngr.res
        self.voxel_half = 0.5 * stride * ngr.res

    def _add_voxels(self) -> None:
        scn = self.renderer.scene
        h = self.voxel_half
        for p in self.voxels:
            if scn.ngeom >= scn.maxgeom:
                break
            mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_BOX, np.array([h, h, h]), p,
                                np.eye(3).flatten(), np.array([0.9, 0.1, 0.1, 0.12], dtype=np.float32))
            scn.ngeom += 1

    def render(self, name: str) -> np.ndarray:
        m = self.world.model
        saved = m.geom_rgba[self.ceiling].copy()
        if name == "top":
            m.geom_rgba[self.ceiling, 3] = 0.0
        self.renderer.update_scene(self.world.data, self.cams[name])
        if self.voxels is not None:
            self._add_voxels()
        img = self.renderer.render()
        m.geom_rgba[self.ceiling] = saved
        return img

    def frame(self) -> np.ndarray:
        return np.concatenate([self.render(n) for n in self.names], axis=1)


def caption(img: np.ndarray, text: str) -> np.ndarray:
    """Burn a caption into the top-left corner of an RGB frame."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.fromarray(img)
    dr = ImageDraw.Draw(im)
    font = ImageFont.load_default(size=20)
    x0, y0, x1, y1 = dr.textbbox((10, 8), text, font=font)
    dr.rectangle((0, 0, x1 + 10, y1 + 8), fill=(255, 255, 255))
    dr.text((10, 8), text, fill=(0, 0, 0), font=font)
    return np.asarray(im)
