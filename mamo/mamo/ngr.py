"""Negative goal region (NGR): the voxelised volume swept by the retrieval trajectory.

M4M Sec. IV and E-M4M 'Main Algorithm': V(gamma_OoI) is the volume occupied by
the robot arm (and the carried OoI) while executing the retrieval trajectory.
Movable objects must end up outside it. The reference implementation voxelises
the arm's collision spheres at 1 cm along the trajectory (robot.cpp UpdateNGR);
here we voxelise points sampled on the xArm7's collision surfaces.

Voxel (i, j, k) of the grid is centred at global lattice coordinates
(ox + i, oy + j, oz + k) * res, so MAPF lattice cells and NGR columns align.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
from scipy import ndimage

from mamo.geometry import footprint, point_dist
from mamo.retrieval import RetrievalPlan
from mamo.robot.arm import Arm
from mamo.robot.motion import densify
from mamo.scene import SceneObject, Shelf


def _surface_samples(model: mujoco.MjModel, g: int, spacing: float) -> np.ndarray:
    """Points on the surface of geom g, in the geom frame."""
    t = model.geom_type[g]
    size = model.geom_size[g]
    if t == mujoco.mjtGeom.mjGEOM_MESH:
        mid = model.geom_dataid[g]
        va, vn = model.mesh_vertadr[mid], model.mesh_vertnum[mid]
        fa, fn = model.mesh_faceadr[mid], model.mesh_facenum[mid]
        verts = model.mesh_vert[va:va + vn]
        faces = model.mesh_face[fa:fa + fn]
        tri = verts[faces]
        area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        n = max(len(verts), int(area.sum() / spacing**2))
        rng = np.random.default_rng(g)
        idx = rng.choice(len(tri), size=n, p=area / area.sum())
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        t0 = tri[idx]
        pts = t0[:, 0] + u[:, None] * (t0[:, 1] - t0[:, 0]) + v[:, None] * (t0[:, 2] - t0[:, 0])
        return np.vstack([pts, verts])
    if t == mujoco.mjtGeom.mjGEOM_BOX:
        return _box_samples(size, spacing)
    if t == mujoco.mjtGeom.mjGEOM_CYLINDER:
        return _cylinder_samples(size[0], size[1], spacing)
    raise ValueError(f"unsupported geom type {t}")


def _box_samples(h: np.ndarray, spacing: float) -> np.ndarray:
    axes = [np.linspace(-hi, hi, max(2, int(np.ceil(2 * hi / spacing)) + 1)) for hi in h]
    g = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    on_face = np.any(np.isclose(np.abs(g), h[None, :]), axis=1)
    return g[on_face]


def _cylinder_samples(r: float, hh: float, spacing: float) -> np.ndarray:
    nt = max(8, int(np.ceil(2 * math.pi * r / spacing)))
    nz = max(2, int(np.ceil(2 * hh / spacing)) + 1)
    th = np.linspace(0, 2 * math.pi, nt, endpoint=False)
    z = np.linspace(-hh, hh, nz)
    side = np.stack([np.repeat(r * np.cos(th)[None], nz, 0), np.repeat(r * np.sin(th)[None], nz, 0),
                     np.repeat(z[:, None], nt, 1)], axis=-1).reshape(-1, 3)
    caps = []
    for rr in np.linspace(0, r, max(2, int(np.ceil(r / spacing)) + 1)):
        m = max(1, int(np.ceil(2 * math.pi * rr / spacing)))
        a = np.linspace(0, 2 * math.pi, m, endpoint=False)
        for zz in (-hh, hh):
            caps.append(np.stack([rr * np.cos(a), rr * np.sin(a), np.full(m, zz)], axis=1))
    return np.vstack([side] + caps)


def _thin(pts: np.ndarray, spacing: float) -> np.ndarray:
    """Keep one point per spacing-sized cell."""
    _, keep = np.unique(np.floor(pts / spacing).astype(np.int64), axis=0, return_index=True)
    return pts[np.sort(keep)]


_SAMPLE_CACHE: dict[float, list[np.ndarray]] = {}


def _robot_samples(arm: Arm, spacing: float) -> dict[int, np.ndarray]:
    """Per-geom surface samples for the robot.

    Robot geom ids shift with the number of scene objects, but their order
    does not, so samples are cached by position in ``robot_geoms``.
    """
    geoms = list(arm.world.robot_geoms)
    if spacing not in _SAMPLE_CACHE:
        _SAMPLE_CACHE[spacing] = [_thin(_surface_samples(arm.model, g, spacing), spacing) for g in geoms]
    return dict(zip(geoms, _SAMPLE_CACHE[spacing]))


class NGR:
    """Boolean voxel grid over the shelf interior."""

    def __init__(self, shelf: Shelf, res: float) -> None:
        self.res = res
        self.origin = np.array([math.floor(shelf.x_min / res), math.floor(shelf.y_min / res),
                                math.floor(shelf.floor_top / res)], dtype=np.int64)
        upper = np.array([math.ceil(shelf.x_max / res), math.ceil(shelf.y_max / res),
                          math.ceil(shelf.ceiling / res)], dtype=np.int64)
        self.shape = tuple(int(v) for v in upper - self.origin + 1)
        self.grid = np.zeros(self.shape, dtype=bool)

    # --------------------------------------------------------------- building
    def add_points(self, pts: np.ndarray) -> None:
        idx = np.round(pts / self.res).astype(np.int64) - self.origin
        ok = np.all((idx >= 0) & (idx < np.array(self.shape)), axis=1)
        idx = idx[ok]
        self.grid[idx[:, 0], idx[:, 1], idx[:, 2]] = True

    @property
    def volume(self) -> int:
        return int(self.grid.sum())

    # ---------------------------------------------------------------- queries
    def column_mask(self, z_bottom: float, z_top: float) -> np.ndarray:
        """(nx, ny) mask of columns holding NGR voxels between two heights."""
        k0 = max(0, int(math.floor(z_bottom / self.res)) - self.origin[2])
        k1 = min(self.shape[2], int(math.ceil(z_top / self.res)) - self.origin[2] + 1)
        if k1 <= k0:
            return np.zeros(self.shape[:2], dtype=bool)
        return self.grid[:, :, k0:k1].any(axis=2)

    def footprint_kernel(self, o: SceneObject, yaw: float | None = None) -> np.ndarray:
        """Offsets (dx, dy) in cells covered by the object's footprint centred at the origin."""
        s = footprint(o, 0.0, 0.0, yaw)
        r = int(math.ceil((o.radius if o.shape == 2 else math.hypot(o.x_size, o.y_size)) / self.res)) + 1
        dx, dy = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij")
        inside = point_dist(s, dx * self.res, dy * self.res) <= 0.5 * self.res
        return np.stack([dx[inside], dy[inside]], axis=1)

    def overlap_map(self, o: SceneObject, yaw: float | None = None) -> np.ndarray:
        """(nx, ny) bool: does the object, centred at each cell, intersect the NGR?"""
        cols = self.column_mask(o.bottom_z, o.top_z)
        k = self.footprint_kernel(o, yaw)
        r = int(np.abs(k).max())
        ker = np.zeros((2 * r + 1, 2 * r + 1), dtype=bool)
        ker[k[:, 0] + r, k[:, 1] + r] = True
        return ndimage.binary_dilation(cols, structure=ker[::-1, ::-1])

    def object_voxels(self, o: SceneObject) -> tuple[int, int]:
        """(object voxels inside the NGR, total object voxels) at the object's pose."""
        k = self.footprint_kernel(o)
        cx = int(round(o.x / self.res)) - self.origin[0]
        cy = int(round(o.y / self.res)) - self.origin[1]
        k0 = int(math.floor(o.bottom_z / self.res)) - self.origin[2]
        k1 = int(math.ceil(o.top_z / self.res)) - self.origin[2] + 1
        nz = max(1, k1 - k0)
        total = len(k) * nz
        xs, ys = k[:, 0] + cx, k[:, 1] + cy
        ok = (xs >= 0) & (xs < self.shape[0]) & (ys >= 0) & (ys < self.shape[1])
        kk0, kk1 = max(0, k0), min(self.shape[2], k1)
        if kk1 <= kk0 or not ok.any():
            return 0, total
        inside = int(self.grid[xs[ok], ys[ok], kk0:kk1].sum())
        return inside, total

    def intersects(self, o: SceneObject) -> bool:
        return self.object_voxels(o)[0] > 0


def compute_ngr(arm: Arm, plan: RetrievalPlan, shelf: Shelf, ooi_id: int,
                object_states: dict[int, np.ndarray], res: float = 0.01,
                traj_step: float = 0.01, spacing: float = 0.004, shrink: float = 0.01,
                include_ooi: bool = True) -> NGR:
    """Voxelise the robot (and, with ``include_ooi``, the carried OoI) along a retrieval plan.

    [cpp] robot.cpp voxeliseTrajectory shrinks every collision sphere by RES before voxelising
    it; here every surface sample moves ``shrink`` towards its geom's centre. The M4M paper's
    NGR is "the volume occupied by the robot arm"; E-M4M's is that of "the robot arm and OoI".
    """
    ngr = NGR(shelf, res)
    model = arm.model
    robot_geoms = list(arm.world.robot_geoms)
    local = _robot_samples(arm, spacing)
    if shrink > 0:
        local = {g: p * np.maximum(0.0, 1.0 - shrink / np.maximum(np.linalg.norm(p, axis=1), 1e-9))[:, None]
                 for g, p in local.items()}
    ooi_geom = arm.world.obj_geom[ooi_id]
    ooi_local = _thin(_surface_samples(model, ooi_geom, spacing), spacing)
    arm.set_object_states(object_states)
    lo = np.array([shelf.x_min, shelf.y_min, shelf.floor_top])
    hi = np.array([shelf.x_max, shelf.y_max, shelf.ceiling])
    # A joint step of traj_step / 0.7 m keeps every robot point within ~traj_step.
    max_step = traj_step / 0.7
    for seg in plan.segments:
        arm.set_gripper(closed=seg.gripper_closed)
        if seg.carrying:
            arm.attach(ooi_id, seg.path[0])
        for q in densify(seg.path, max_step):
            arm.fk(q)
            arm._place_attached()
            d = arm.data
            pts = []
            for g in robot_geoms:
                c, rb = d.geom_xpos[g], model.geom_rbound[g]
                if np.any(c + rb < lo) or np.any(c - rb > hi):
                    continue  # geom cannot touch the shelf interior
                pts.append(c + local[g] @ d.geom_xmat[g].reshape(3, 3).T)
            if seg.carrying and include_ooi:
                pts.append(d.geom_xpos[ooi_geom] + ooi_local @ d.geom_xmat[ooi_geom].reshape(3, 3).T)
            if pts:
                ngr.add_points(np.vstack(pts))
        arm.detach()
    arm.set_gripper(closed=True)
    return ngr
