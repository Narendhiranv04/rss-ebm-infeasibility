"""Exact 2D footprint geometry for upright objects (circles and oriented boxes).

Every MAMO object rests on the shelf floor, so two objects collide iff their
footprints intersect (their z-ranges always overlap). The MAPF abstraction uses
these exact tests in place of the reference implementation's FCL checks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from mamo.scene import BOX, CYLINDER, SceneObject


@dataclass(frozen=True)
class Circle:
    cx: float
    cy: float
    r: float


@dataclass(frozen=True)
class OBB:
    cx: float
    cy: float
    hx: float
    hy: float
    yaw: float

    def axes(self) -> tuple[np.ndarray, np.ndarray]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([c, s]), np.array([-s, c])

    def corners(self) -> np.ndarray:
        ux, uy = self.axes()
        ctr = np.array([self.cx, self.cy])
        return np.array([ctr + sx * self.hx * ux + sy * self.hy * uy
                         for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


Shape = Circle | OBB


def footprint(o: SceneObject, x: float | None = None, y: float | None = None,
              yaw: float | None = None, margin: float = 0.0) -> Shape:
    x = o.x if x is None else x
    y = o.y if y is None else y
    yaw = o.yaw if yaw is None else yaw
    if o.shape == CYLINDER:
        return Circle(x, y, o.radius + margin)
    return OBB(x, y, o.x_size + margin, o.y_size + margin, yaw)


def translated(s: Shape, x: float, y: float) -> Shape:
    if isinstance(s, Circle):
        return Circle(x, y, s.r)
    return OBB(x, y, s.hx, s.hy, s.yaw)


def _obb_point_dist(b: OBB, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Signed-free distance from points to an OBB (0 inside)."""
    ux, uy = b.axes()
    dx, dy = px - b.cx, py - b.cy
    lx = np.abs(dx * ux[0] + dy * ux[1]) - b.hx
    ly = np.abs(dx * uy[0] + dy * uy[1]) - b.hy
    return np.hypot(np.maximum(lx, 0.0), np.maximum(ly, 0.0))


def point_dist(s: Shape, px: np.ndarray, py: np.ndarray) -> np.ndarray:
    """Distance from points to a shape (0 inside), vectorised."""
    if isinstance(s, Circle):
        return np.maximum(np.hypot(px - s.cx, py - s.cy) - s.r, 0.0)
    return _obb_point_dist(s, px, py)


def boundary_points(s: Shape, n: int = 32) -> np.ndarray:
    """Points on the shape boundary, relative to its centre."""
    if isinstance(s, Circle):
        th = np.linspace(0.0, 2 * math.pi, n, endpoint=False)
        return np.stack([s.r * np.cos(th), s.r * np.sin(th)], axis=1)
    per_side = max(2, n // 4)
    ux, uy = s.axes()
    t = np.linspace(-1.0, 1.0, per_side, endpoint=False)
    pts = []
    for a, b, ha, hb in ((ux, uy, s.hx, s.hy), (uy, -ux, s.hy, s.hx),
                         (-ux, -uy, s.hx, s.hy), (-uy, ux, s.hy, s.hx)):
        pts.append(a[None, :] * ha + t[:, None] * hb * b[None, :])
    return np.concatenate(pts, axis=0)


def _sat_obb_obb(a: OBB, b: OBB) -> bool:
    ca, cb = a.corners(), b.corners()
    for axis in (*a.axes(), *b.axes()):
        pa, pb = ca @ axis, cb @ axis
        if pa.max() < pb.min() or pb.max() < pa.min():
            return False
    return True


def shapes_intersect(a: Shape, b: Shape) -> bool:
    if isinstance(a, Circle) and isinstance(b, Circle):
        return math.hypot(a.cx - b.cx, a.cy - b.cy) < a.r + b.r
    if isinstance(a, Circle):
        return float(_obb_point_dist(b, np.array([a.cx]), np.array([a.cy]))[0]) < a.r
    if isinstance(b, Circle):
        return float(_obb_point_dist(a, np.array([b.cx]), np.array([b.cy]))[0]) < b.r
    return _sat_obb_obb(a, b)


def shape_distance(a: Shape, b: Shape) -> float:
    """Distance between two convex footprints (0 if intersecting)."""
    if shapes_intersect(a, b):
        return 0.0
    if isinstance(a, Circle) and isinstance(b, Circle):
        return math.hypot(a.cx - b.cx, a.cy - b.cy) - a.r - b.r
    if isinstance(a, Circle):
        return float(_obb_point_dist(b, np.array([a.cx]), np.array([a.cy]))[0]) - a.r
    if isinstance(b, Circle):
        return float(_obb_point_dist(a, np.array([b.cx]), np.array([b.cy]))[0]) - b.r
    # OBB-OBB: closest features are a corner of one against the other box.
    d1 = _obb_point_dist(b, a.corners()[:, 0], a.corners()[:, 1]).min()
    d2 = _obb_point_dist(a, b.corners()[:, 0], b.corners()[:, 1]).min()
    # Edge-edge minima are attained at corners for separated convex polygons.
    return float(min(d1, d2))


def segment_aabb_entry(p0: np.ndarray, direction: np.ndarray, lo: np.ndarray,
                       hi: np.ndarray) -> np.ndarray | None:
    """First intersection of the ray p0 + t*direction (t >= 0) with an AABB."""
    tmin, tmax = 0.0, math.inf
    for k in range(2):
        if abs(direction[k]) < 1e-12:
            if p0[k] < lo[k] or p0[k] > hi[k]:
                return None
            continue
        t1 = (lo[k] - p0[k]) / direction[k]
        t2 = (hi[k] - p0[k]) / direction[k]
        t1, t2 = min(t1, t2), max(t1, t2)
        tmin, tmax = max(tmin, t1), min(tmax, t2)
        if tmin > tmax:
            return None
    return p0 + tmin * direction


def aabb(o: SceneObject, x: float | None = None, y: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned 2D bounding box of an object's footprint."""
    s = footprint(o, x, y)
    if isinstance(s, Circle):
        c = np.array([s.cx, s.cy])
        return c - s.r, c + s.r
    c = s.corners()
    return c.min(axis=0), c.max(axis=0)
