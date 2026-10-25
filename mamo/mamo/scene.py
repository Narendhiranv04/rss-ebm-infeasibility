"""MAMO scene description: objects, the shelf, and the object of interest (OoI).

The on-disk format is the one written by the reference C++ implementation
(``dat/clutter_scenes/*/plan_*_SCENE.txt``). Only the ``O`` (objects) and
``ooi`` sections are needed; the remaining sections are PR2-specific logs.
Scenes of the generated environments (mamo.envs) add an optional ``meta`` section
between them: one line of JSON with the environment, the robot base and the task.

Object row columns (``Planner::parse_scene`` in planner.cpp):
    id, shape, type, x, y, z, roll, pitch, yaw, x_size, y_size, z_size, mass, mu, movable

* shape: 0 = box, 2 = cylinder
* type: -1 = shelf part (static), 0 = immovable obstacle, 1 = movable
* box sizes are half extents; a cylinder's radius is max(x_size, y_size) and
  z_size is its *full* height (``addCylinder`` in scripts/sim.py)
* mass == 0 marks a static ("locked") body
"""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

BOX = 0
CYLINDER = 2

SHELF = -1
IMMOVABLE = 0
MOVABLE = 1


@dataclass
class SceneObject:
    id: int
    shape: int
    type: int
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float
    x_size: float
    y_size: float
    z_size: float
    mass: float
    mu: float
    movable: bool

    @property
    def is_shelf(self) -> bool:
        return self.type == SHELF

    @property
    def radius(self) -> float:
        return max(self.x_size, self.y_size)

    @property
    def half_height(self) -> float:
        return 0.5 * self.z_size if self.shape == CYLINDER else self.z_size

    @property
    def bottom_z(self) -> float:
        return self.z - self.half_height

    @property
    def top_z(self) -> float:
        return self.z + self.half_height

    def volume(self) -> float:
        if self.shape == CYLINDER:
            return math.pi * self.radius**2 * self.z_size
        return 8.0 * self.x_size * self.y_size * self.z_size

    def copy(self) -> "SceneObject":
        return copy.copy(self)


@dataclass
class Shelf:
    """An axis-aligned support surface (a horizontal type -1 part) and the free space above it.

    The planners work on one of these: ``Scene.shelf``, the surface under the OoI. In the
    reference scenes that is the shelf floor, and its ceiling the shelf's top board.
    """

    floor_top: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    ceiling: float
    part_id: int = -1

    def contains_xy(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


OPEN_HEIGHT = 0.40    # free height assumed above a surface with nothing over it (tables, top decks)
ON_SURFACE_TOL = 0.02


@dataclass
class Scene:
    objects: list[SceneObject]
    ooi_id: int
    name: str = ""
    meta: dict = field(default_factory=dict)
    shelf: Shelf = field(init=False)

    def __post_init__(self) -> None:
        self.surfaces = _infer_surfaces(self.objects)
        self.shelf = _support(self.surfaces, self.by_id(self.ooi_id)) or _infer_shelf(self.objects)
        # Objects that belong to the problem: those resting on an active surface. A retrieve
        # problem lives on the OoI's surface (objects on other decks are fixed obstacles to it);
        # a relocate problem spans every surface.
        surfs = [self.shelf] if self.task["type"] == "retrieve" else self.surfaces
        self.active_ids = frozenset(o.id for o in self.objects if not o.is_shelf
                                    and any(on_surface(o, s) for s in surfs))

    # ---------------------------------------------------------------- task
    @property
    def task(self) -> dict:
        """``{"type": "retrieve" | "relocate", "targets": [...]}``; by default retrieve the OoI.

        Relocate targets are ``{"id", "goal": [x, y, z, yaw], "tol": [xy, yaw_deg]}``: the goal
        pose of the object's centre, reached when within ``xy`` metres (and ``yaw_deg``
        degrees unless negative). The OoI is the first target.
        """
        return self.meta.get("task") or {"type": "retrieve", "targets": [self.ooi_id]}

    @property
    def target_ids(self) -> list[int]:
        return [t if isinstance(t, int) else int(t["id"]) for t in self.task["targets"]]

    @property
    def env(self) -> str:
        return self.meta.get("env", "reference_shelf")

    def is_active(self, oid: int) -> bool:
        return oid in self.active_ids

    def spans_band(self, o: SceneObject) -> bool:
        """Does ``o`` reach into the free space above the planning surface (walls, objects on it)?"""
        return o.top_z > self.shelf.floor_top + 1e-6 and o.bottom_z < self.shelf.ceiling - 1e-6

    def surface_of(self, x: float, y: float, z: float) -> Shelf | None:
        """The highest surface below (x, y, z) that contains (x, y)."""
        best = None
        for s in self.surfaces:
            if s.contains_xy(x, y) and s.floor_top <= z + 1e-6 and (best is None or s.floor_top > best.floor_top):
                best = s
        return best

    def by_id(self, oid: int) -> SceneObject:
        for o in self.objects:
            if o.id == oid:
                return o
        raise KeyError(oid)

    @property
    def ooi(self) -> SceneObject:
        return self.by_id(self.ooi_id)

    @property
    def shelf_parts(self) -> list[SceneObject]:
        return [o for o in self.objects if o.is_shelf]

    @property
    def movables(self) -> list[SceneObject]:
        """Movable objects of the problem (those on an active surface)."""
        return [o for o in self.objects if o.movable and not o.is_shelf and o.id in self.active_ids]

    @property
    def immovables(self) -> list[SceneObject]:
        """Immovable obstacles, excluding shelf parts and the OoI (includes movables off the active surfaces)."""
        return [o for o in self.objects if not o.is_shelf and o.id != self.ooi_id
                and (not o.movable or o.id not in self.active_ids)]

    def copy(self) -> "Scene":
        # Surfaces and the problem's objects are fixed by the initial scene: keep them even
        # when objects have moved (or are in the gripper).
        sc = Scene.__new__(Scene)
        sc.objects = [o.copy() for o in self.objects]
        sc.ooi_id, sc.name, sc.meta = self.ooi_id, self.name, self.meta
        sc.surfaces, sc.shelf, sc.active_ids = self.surfaces, self.shelf, self.active_ids
        return sc


def on_surface(o: SceneObject, s: Shelf) -> bool:
    return abs(o.bottom_z - s.floor_top) < ON_SURFACE_TOL and s.contains_xy(o.x, o.y)


def _infer_surfaces(objects: list[SceneObject]) -> list[Shelf]:
    """Every thin horizontal shelf part is a surface; its ceiling is the next part above it."""
    slabs = [o for o in objects if o.is_shelf and o.z_size < 0.5 * min(o.x_size, o.y_size)]
    out = []
    for s in slabs:
        top = s.z + s.z_size
        x0, x1, y0, y1 = s.x - s.x_size, s.x + s.x_size, s.y - s.y_size, s.y + s.y_size
        above = [o.z - o.z_size for o in slabs if o is not s and o.z - o.z_size > top + 1e-6
                 and o.x - o.x_size < x1 and o.x + o.x_size > x0 and o.y - o.y_size < y1 and o.y + o.y_size > y0]
        out.append(Shelf(top, x0, x1, y0, y1, min(above, default=top + OPEN_HEIGHT), s.id))
    return sorted(out, key=lambda s: s.floor_top)


def _support(surfaces: list[Shelf], o: SceneObject) -> Shelf | None:
    cands = [s for s in surfaces if s.contains_xy(o.x, o.y) and s.floor_top <= o.bottom_z + ON_SURFACE_TOL]
    return max(cands, key=lambda s: s.floor_top, default=None)


def _infer_shelf(objects: list[SceneObject]) -> Shelf:
    parts = [o for o in objects if o.is_shelf]
    if not parts:
        raise ValueError("scene has no shelf (type -1) parts")
    # The floor is the widest thin slab with the lowest top surface.
    floor = min(parts, key=lambda o: (o.z + o.z_size, -o.x_size * o.y_size))
    upper = [o for o in parts if o is not floor and o.z_size < 0.5 * min(o.x_size, o.y_size)]
    ceiling = min((o.z - o.z_size for o in upper), default=float("inf"))
    return Shelf(
        floor_top=floor.z + floor.z_size,
        x_min=floor.x - floor.x_size,
        x_max=floor.x + floor.x_size,
        y_min=floor.y - floor.y_size,
        y_max=floor.y + floor.y_size,
        ceiling=ceiling,
    )


def parse_scene_file(path: str | Path) -> Scene:
    """Read the ``O`` and ``ooi`` sections of a reference-format scene file."""
    path = Path(path)
    objects: list[SceneObject] = []
    ooi_id: int | None = None
    meta: dict = {}
    with open(path, encoding="utf-8") as fh:
        lines = iter(fh.read().splitlines())
    for line in lines:
        tag = line.strip()
        if tag == "O":
            n = int(next(lines))
            for _ in range(n):
                f = next(lines).strip().split(",")
                objects.append(SceneObject(
                    id=int(f[0]), shape=int(f[1]), type=int(f[2]),
                    x=float(f[3]), y=float(f[4]), z=float(f[5]),
                    roll=float(f[6]), pitch=float(f[7]), yaw=float(f[8]),
                    x_size=float(f[9]), y_size=float(f[10]), z_size=float(f[11]),
                    mass=float(f[12]), mu=float(f[13]),
                    movable=f[14].strip() == "True",
                ))
        elif tag == "meta":
            meta = json.loads(next(lines))
        elif tag == "ooi":
            ooi_id = int(next(lines))
            # Everything after ooi in reference files is solution logging.
            break
    if ooi_id is None:
        raise ValueError(f"{path}: no 'ooi' section")
    return Scene(objects, ooi_id, name=path.stem, meta=meta)


def write_scene_file(scene: Scene, path: str | Path) -> None:
    rows = []
    for o in scene.objects:
        rows.append(",".join(str(v) for v in (
            o.id, o.shape, o.type, o.x, o.y, o.z, o.roll, o.pitch, o.yaw,
            o.x_size, o.y_size, o.z_size, o.mass, o.mu, "True" if o.movable else "False")))
    meta = "meta\n{}\n".format(json.dumps(scene.meta, separators=(",", ":"))) if scene.meta else ""
    text = "O\n{}\n{}\n{}ooi\n{}\n".format(len(rows), "\n".join(rows), meta, scene.ooi_id)
    Path(path).write_text(text, encoding="utf-8")


def default_shelf_parts() -> list[SceneObject]:
    """The refrigerator-style shelf used by every reference scene (ids 1-5)."""
    def part(i, x, y, z, sx, sy, sz):
        return SceneObject(i, BOX, SHELF, x, y, z, 0, 0, 0, sx, sy, sz, 0.0, 0.8, False)
    return [
        part(1, 0.78, -0.5, 0.75, 0.3, 0.4, 0.02),    # floor
        part(2, 1.10, -0.5, 0.93, 0.02, 0.4, 0.2),    # back wall
        part(3, 0.78, -0.5, 1.11, 0.3, 0.4, 0.02),    # ceiling
        part(4, 0.80, -0.92, 0.93, 0.32, 0.02, 0.2),  # right wall
        part(5, 0.80, -0.08, 0.93, 0.32, 0.02, 0.2),  # left wall
    ]


def footprint_overlap(a: SceneObject, b: SceneObject, margin: float = 0.0) -> bool:
    """Exact 2D overlap test between two upright object footprints."""
    from mamo.geometry import footprint, shapes_intersect
    return shapes_intersect(footprint(a, margin=margin), footprint(b))


def generate_scene(num_movable: int, num_immovable: int = 4, seed: int = 0,
                   min_size: float = 0.03, max_xy: float = 0.08,
                   max_z: float = 0.2, mu_range: tuple[float, float] = (0.8, 1.2),
                   max_mass: float = 0.5) -> Scene:
    """Random MAMO problem in the reference shelf (paper: Sec. V-A / Experimental Analysis).

    Mirrors ``BulletSim::addObjects`` sampling: half extents in [min_size, max_xy],
    heights up to ``max_z``, random yaw, mass U(0, max_mass), friction U(mu_range).
    One immovable is designated the OoI, as in the reference scene files.
    """
    rng = np.random.default_rng(seed)
    parts = default_shelf_parts()
    floor = parts[0]
    floor_top = floor.z + floor.z_size
    objects = list(parts)
    next_id = len(parts) + 1

    def sample(obj_type: int, movable: bool) -> SceneObject | None:
        nonlocal next_id
        for _ in range(500):
            shape = BOX if rng.random() < 0.5 else CYLINDER
            if shape == CYLINDER:
                r = rng.uniform(min_size, max_xy * 0.75)
                sx = sy = r
                h = rng.uniform(0.15, max_z)
                sz = h
                z = floor_top + 0.5 * h
            else:
                sx, sy = rng.uniform(min_size, max_xy * 0.75, size=2)
                sz = rng.uniform(0.09, 0.5 * max_z)
                z = floor_top + sz
            margin = max(sx, sy) + 0.02
            x = rng.uniform(floor.x - floor.x_size + margin, floor.x + floor.x_size - 0.02 - margin)
            y = rng.uniform(floor.y - floor.y_size + 0.02 + margin, floor.y + floor.y_size - 0.02 - margin)
            o = SceneObject(next_id, shape, obj_type, x, y, z, 0.0, 0.0, rng.uniform(0, 2 * math.pi),
                            sx, sy, sz, rng.uniform(0.05, max_mass), rng.uniform(*mu_range), movable)
            if any(footprint_overlap(o, p, margin=0.005) for p in objects if not p.is_shelf):
                continue
            next_id += 1
            return o
        return None

    for _ in range(num_immovable + 1):
        o = sample(IMMOVABLE, False)
        if o is not None:
            objects.append(o)
    for _ in range(num_movable):
        o = sample(MOVABLE, True)
        if o is not None:
            objects.append(o)
    immov = [o for o in objects if o.type == IMMOVABLE]
    # The OoI is the immovable closest to the shelf front, matching the
    # reference scenes where the OoI must be reachable through the opening.
    ooi = min(immov, key=lambda o: o.x)
    return Scene(objects, ooi.id, name=f"gen_{num_movable}_{seed}")
