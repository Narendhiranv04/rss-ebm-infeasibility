"""SELSIM's heuristic: "a 3-D Breadth-First search initialized at the (x, y, z) coordinates of
the goal considering only the immovable objects in the environment" (paper Sec. VI).

As in the Likhachev lab's SMPL BFS heuristic the reference release configures
(robot.cpp: bfs_cost_per_cell 100, bfs_inflation_radius 0.02; 1 cm occupancy grid): the
shelf, immovable obstacles and the OoI are voxelised at 1 cm and inflated by 2 cm, the BFS
is 26-connected with unit steps, and h = 100 x (BFS cells from the TCP's cell to the goal).
As in SMPL's syncGridAndBfs, a cell is a wall when its centre is within 2 cm (Euclidean) of
an occupied cell, and the grid's boundary cells are walls.
"""

from __future__ import annotations

import itertools

import mujoco
import numpy as np

from scipy import ndimage

from mamo.world import IMMOVABLE_GEOM, OOI_GEOM, SHELF_GEOM, World

COST_PER_CELL = 100      # [cpp] bfs_cost_per_cell
INFLATION = 0.02         # [cpp] bfs_inflation_radius (m)
RES = 0.01               # [cpp] occupancy_grid/res
REACH = 0.85             # [xarm] grid half-size around the robot base (m)


class BFSHeuristic:
    def __init__(self, world: World, goal: np.ndarray) -> None:
        bx, by, bz = world.cfg.robot.base_pos
        self.origin = np.array([bx - REACH, by - REACH, max(0.0, bz - 0.6)])
        top = np.array([bx + REACH, by + REACH, bz + 0.9])
        self.shape = tuple(int(v) for v in np.ceil((top - self.origin) / RES))
        occ = self._occupancy(world)
        walls = ndimage.distance_transform_edt(~occ) * RES <= INFLATION
        walls[0, :, :] = walls[-1, :, :] = walls[:, 0, :] = walls[:, -1, :] = True
        walls[:, :, 0] = walls[:, :, -1] = True
        self.dist = self._bfs(walls, self.cell(goal))

    def cell(self, p: np.ndarray) -> tuple[int, int, int] | None:
        c = np.floor((p - self.origin) / RES).astype(int)
        if np.any(c < 0) or np.any(c >= self.shape):
            return None
        return tuple(int(v) for v in c)

    def cells_to_goal(self, p: np.ndarray) -> float:
        """BFS distance in cells (inf for walls and outside the grid)."""
        c = self.cell(p)
        if c is None:
            return np.inf
        d = self.dist[c]
        return np.inf if d < 0 else float(d)

    def __call__(self, p: np.ndarray) -> float:
        c = self.cell(p)
        if c is None:
            return np.inf
        d = self.dist[c]
        return np.inf if d < 0 else float(COST_PER_CELL * d)

    # ------------------------------------------------------------------ build
    def _occupancy(self, world: World) -> np.ndarray:
        m, d = world.model, world.data
        mujoco.mj_forward(m, d)
        occ = np.zeros(self.shape, dtype=bool)
        occ[:, :, : max(0, int(np.ceil((0.0 - self.origin[2]) / RES)))] = True  # below the ground
        for g in np.flatnonzero(np.isin(world.geom_cat, (SHELF_GEOM, IMMOVABLE_GEOM, OOI_GEOM))):
            c, R, size = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), m.geom_size[g]
            box = m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX
            half = size if box else np.array([size[0], size[0], size[1]])
            ext = np.abs(R) @ half + RES
            lo = np.clip(np.floor((c - ext - self.origin) / RES).astype(int), 0, self.shape)
            hi = np.clip(np.ceil((c + ext - self.origin) / RES).astype(int) + 1, 0, self.shape)
            if np.any(hi <= lo):
                continue
            ax = [self.origin[i] + (np.arange(lo[i], hi[i]) + 0.5) * RES for i in range(3)]
            P = np.stack(np.meshgrid(*ax, indexing="ij"), axis=-1)
            L = (P - c) @ R  # geom-frame coordinates
            if box:
                inside = np.all(np.abs(L) <= size, axis=-1)
            else:
                inside = (np.hypot(L[..., 0], L[..., 1]) <= size[0]) & (np.abs(L[..., 2]) <= size[1])
            occ[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] |= inside
        return occ

    def _bfs(self, occ: np.ndarray, goal: tuple[int, int, int] | None) -> np.ndarray:
        dist = np.full(self.shape, -1, dtype=np.int32)
        if goal is None:
            return dist
        nx, ny, nz = self.shape
        free = ~occ.ravel()
        flat = dist.ravel()
        g = np.ravel_multi_index(goal, self.shape)
        free[g] = True  # the goal itself is always reachable
        flat[g] = 0
        offs = [o for o in itertools.product((-1, 0, 1), repeat=3) if o != (0, 0, 0)]
        front = np.array([g])
        k = 0
        while front.size:
            k += 1
            x, y, z = np.unravel_index(front, self.shape)
            nxt = []
            for dx, dy, dz in offs:
                X, Y, Z = x + dx, y + dy, z + dz
                ok = (X >= 0) & (X < nx) & (Y >= 0) & (Y < ny) & (Z >= 0) & (Z < nz)
                idx = np.ravel_multi_index((X[ok], Y[ok], Z[ok]), self.shape)
                idx = idx[free[idx] & (flat[idx] < 0)]
                flat[idx] = k
                nxt.append(idx)
            front = np.unique(np.concatenate(nxt))
        return dist
