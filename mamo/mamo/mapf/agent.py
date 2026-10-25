"""MAPF agents: movable objects on a 4-connected 1 cm grid (M4M Sec. IV-A).

Each movable object is an agent whose pose changes only in x/y by one cell per
unit time (or waits). Its goal set is every collision-free cell where the
object lies outside the NGR. Per-cell maps are precomputed with numpy:

* ``valid``   object footprint collides with no immovable obstacle, shelf wall
              or the OoI, and its centre is on the shelf floor
* ``outside`` object (full footprint over its height range) misses the NGR
* ``cost``    per-step cost: 1 + 2^(1 - d/0.15), d = distance to the nearest
              obstacle (agent_lattice.cpp); scaled by COST_MULT

Cells are indexed in NGR-local coordinates (i, j) = global lattice - origin.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mamo.config import MAPFConfig
from mamo.geometry import Circle, OBB, boundary_points, footprint, point_dist, shapes_intersect, translated
from mamo.ngr import NGR
from mamo.scene import Scene, SceneObject


def _obstacle_shapes(scene: Scene, exclude: set[int]) -> list:
    """Footprints of everything a movable may not overlap: immovables, the OoI, shelf walls."""
    shapes = []
    for o in scene.objects:
        if o.id in exclude or (o.movable and scene.is_active(o.id)):
            continue
        # Only what spans the objects' height band (walls, objects on the shelf) is a 2D obstacle.
        if not scene.spans_band(o):
            continue
        shapes.append(footprint(o))
    return shapes


@dataclass
class Agent:
    obj: SceneObject
    start: tuple[int, int]                  # local cell
    valid: np.ndarray
    outside: np.ndarray
    cost: np.ndarray                        # integer step cost per cell
    shape0: object                          # footprint centred at (0, 0)
    res: float
    origin: np.ndarray
    invalid_goals: set[tuple[int, int]] = field(default_factory=set)
    goal_penalty: np.ndarray | None = None  # integer pseudo-goal edge cost per cell
    # agents_collide results keyed by (other agent id, cell offset); see agents_collide
    collide_cache: dict = field(default_factory=dict, repr=False, compare=False)

    @property
    def id(self) -> int:
        return self.obj.id

    def xy(self, cell: tuple[int, int]) -> tuple[float, float]:
        return ((cell[0] + self.origin[0]) * self.res, (cell[1] + self.origin[1]) * self.res)

    def shape_at(self, cell: tuple[int, int]):
        x, y = self.xy(cell)
        return translated(self.shape0, x, y)

    def is_goal_cell(self, cell: tuple[int, int]) -> bool:
        return bool(self.outside[cell]) and cell not in self.invalid_goals


def build_agent(o: SceneObject, scene: Scene, ngr: NGR, cfg: MAPFConfig) -> Agent:
    res = ngr.res
    nx, ny = ngr.shape[:2]
    gi, gj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    cx = (gi + ngr.origin[0]) * res
    cy = (gj + ngr.origin[1]) * res
    shape0 = footprint(o, 0.0, 0.0)
    obstacles = _obstacle_shapes(scene, exclude={o.id})

    bpts = boundary_points(shape0, 32)
    dist = np.full((nx, ny), np.inf)
    for obs in obstacles:
        # Agent boundary vs obstacle, and obstacle boundary vs agent (handles containment).
        px = cx[..., None] + bpts[:, 0]
        py = cy[..., None] + bpts[:, 1]
        d1 = point_dist(obs, px, py).min(axis=-1)
        center = np.array([obs.cx, obs.cy])
        opts = boundary_points(obs, 32) + center
        d2 = point_dist(shape0, opts[:, 0][None, None, :] - cx[..., None],
                        opts[:, 1][None, None, :] - cy[..., None]).min(axis=-1)
        dist = np.minimum(dist, np.minimum(d1, d2))
    shelf = scene.shelf
    on_floor = (cx >= shelf.x_min) & (cx <= shelf.x_max) & (cy >= shelf.y_min) & (cy <= shelf.y_max)
    valid = on_floor & (dist > 0.0)

    start = (int(round(o.x / res)) - int(ngr.origin[0]), int(round(o.y / res)) - int(ngr.origin[1]))
    start = (min(max(start[0], 0), nx - 1), min(max(start[1], 0), ny - 1))
    # The start pose is where the object is; never reject it on discretisation grounds.
    valid[start] = True

    if cfg.step_cost == "unit":
        cost = np.full(valid.shape, cfg.cost_mult, dtype=np.int64)
    else:
        obs_cost = np.power(2.0, 1.0 - np.minimum(dist, 10.0) / cfg.obstacle_cost_scale)
        cost = np.round((1.0 + obs_cost) * cfg.cost_mult).astype(np.int64)
    outside = ~ngr.overlap_map(o) & valid
    return Agent(o, start, valid, outside, cost, shape0, res, ngr.origin.copy())


def agents_collide(a: Agent, ca: tuple[int, int], b: Agent, cb: tuple[int, int]) -> bool:
    """Do the footprints of a at ca and b at cb overlap?

    Footprints only translate on the grid, so the answer depends only on the cell offset
    cb - ca; it is computed once per (other agent, offset) and cached on the agent (a pure
    speed-up: CBS spends most of its time here otherwise).
    """
    dx, dy = cb[0] - ca[0], cb[1] - ca[1]
    key = (b.id, dx, dy)
    hit = a.collide_cache.get(key)
    if hit is None:
        hit = shapes_intersect(a.shape0, translated(b.shape0, dx * a.res, dy * a.res))
        a.collide_cache[key] = hit
    return hit
