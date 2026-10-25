"""The five rearrangement environments of REAP (Swoboda & Geffner 2026, App. B) for the xArm7.

Each environment is a set of static parts (shelf boards, tables, walls), grids of cells on
its support surfaces (where objects start and where goals are), object classes and the
robot's base. Dimensions are the paper's (Figures 5, 7-10); the robot is the xArm7 of the
rest of this package, with its base placed so the cells can be reached
(scripts/env_reachability.py). The paper's frame has the robot at the origin and the depth
axis along +y; here the robot faces +x, so paper (x, y) becomes (-y, x) and every
environment opens towards the robot (-x), as the reference shelf does.

Tasks (Scene.task):

* ``retrieve``: grasp the OoI and take it out towards the robot (M4M's problem; the M4M
  family, I-M4M and DOGAR solve it on the OoI's surface)
* ``relocate``: move one or more targets to goal cells, possibly on another surface
  (REAP's multi-level shelf and Access-19 task)

Difficulty: easy / medium / hard set the ranges of movable and immovable objects, targets
and blockers (movables placed in a target's access corridor) per environment
(``Environment.presets``); generate() samples within them and records the counts. For
retrieve problems make_problem() also requires, as the E-M4M / I-M4M papers' labels do, that
1 (easy), 2 (medium) or 3-6 (hard) movables overlap the NGR of the OoI's retrieval
planned with only the immovables as obstacles (retrieval_blockers()).
"""


from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mamo.scene import BOX, CYLINDER, IMMOVABLE, MOVABLE, SHELF, Scene, SceneObject, footprint_overlap

DIFFICULTIES = ("easy", "medium", "hard")
# movables overlapping the NGR of the OoI's retrieval, per difficulty (E-M4M / I-M4M labels)
NGR_BLOCKERS = {"easy": (1, 1), "medium": (2, 2), "hard": (3, 6)}
TASKS = ("retrieve", "relocate")
BOARD = 0.01          # half thickness of boards and walls


@dataclass(frozen=True)
class Region:
    """A grid of cells on a support surface. Columns run across (y), rows away from the robot (x)."""
    name: str
    surface_z: float
    x0: float                      # x of the grid's front edge (row 0)
    yc: float                      # y of the grid's centre line
    nx: int                        # columns
    ny: int                        # rows
    pitch_col: float               # cell size across (m)
    pitch_row: float               # cell size in depth (m)
    blocked: frozenset = frozenset()  # (col, row) cells that are not part of the grid

    def cells(self) -> list[tuple[int, int]]:
        return [(i, j) for j in range(self.ny) for i in range(self.nx) if (i, j) not in self.blocked]

    def center(self, cell: tuple[int, int]) -> tuple[float, float]:
        """Column 0 is on the robot's left (+y), row 0 at the front."""
        i, j = cell
        return self.x0 + (j + 0.5) * self.pitch_row, self.yc + ((self.nx - 1) / 2 - i) * self.pitch_col

    def to_meta(self) -> dict:
        return {"name": self.name, "z": self.surface_z, "x0": self.x0, "yc": self.yc, "nx": self.nx,
                "ny": self.ny, "pitch": [self.pitch_col, self.pitch_row], "blocked": [list(c) for c in sorted(self.blocked)]}


@dataclass(frozen=True)
class ObjectClass:
    name: str
    shape: int
    size: tuple[float, float, float]   # half extents (box) or (r, r, full height) (cylinder)
    weight: float = 1.0                # sampling weight
    cells: tuple[int, int] = (1, 1)    # cells covered (columns, rows) when not turned


@dataclass(frozen=True)
class Preset:
    movables: tuple[int, int]
    immovables: tuple[int, int]
    targets: tuple[int, int]           # relocate tasks; retrieve tasks always have one target
    blockers: tuple[int, int]          # movables placed in the targets' access corridors


@dataclass
class Environment:
    name: str
    title: str
    description: str
    regions: list[Region]
    parts: list[SceneObject]
    classes: list[ObjectClass]
    presets: dict[str, Preset]
    base: tuple[float, float, float] = (0.0, 0.0, 0.40)
    start_regions: tuple[str, ...] = ()      # where targets start (default: every region)
    goal_regions: tuple[str, ...] = ()       # where relocate goals are (default: every region)
    target_cells: dict = field(default_factory=dict)  # region -> allowed target start cells
    fixed_goal: tuple[str, tuple[int, int]] | None = None  # relocate goal of the OoI (Access-19)
    blocker_pattern: dict = field(default_factory=dict)  # region -> the only cells objects may use
    config: dict = field(default_factory=dict)  # Config overrides ("section.field": value)
    reach_rows: int = 99                      # access corridor: rows in front of a target that block it
    corridor_cols: int = 1                    # ... and columns either side
    around: tuple[int, int] | None = None     # grasps from above: corridor = cells within (columns, rows)

    def region(self, name: str) -> Region:
        return next(r for r in self.regions if r.name == name)


# ---------------------------------------------------------------- builders
class _Parts:
    def __init__(self) -> None:
        self.parts: list[SceneObject] = []

    def box(self, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float) -> None:
        """Static box spanning [x0, x1] x [y0, y1] x [z0, z1]."""
        i = len(self.parts) + 1
        self.parts.append(SceneObject(i, BOX, SHELF, (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, 0, 0, 0,
                                      (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2, 0.0, 0.8, False))

    def slab(self, x0: float, x1: float, y0: float, y1: float, top: float) -> None:
        self.box(x0, x1, y0, y1, top - 2 * BOARD, top)


def open_tabletop(front: float = 0.20, height: float = 0.30, base_z: float = 0.40) -> Environment:
    """App. B.2, Fig. 5: a 40 x 28 cm table, 10 x 7 cells of 40 mm, cylinders of three sizes."""
    s = base_z + height
    p = _Parts()
    p.slab(front, front + 0.28, -0.20, 0.20, s)
    p.box(front + 0.04, front + 0.24, -0.16, 0.16, 0.0, s - 2 * BOARD)  # pedestal under the table
    grid = Region("table", s, front, 0.0, 10, 7, 0.04, 0.04)
    return Environment(
        "open_tabletop", "Open tabletop", "40x28 cm table, 10x7 grid of 40 mm cells, thin/medium/thick cylinders "
        "(r 12.5/17.5/20 mm, h 8/10/12 cm). Neighbouring cylinders block the side grasp.",
        [grid], p.parts,
        [ObjectClass("thin", CYLINDER, (0.0125, 0.0125, 0.08), 15), ObjectClass("medium", CYLINDER, (0.0175, 0.0175, 0.10), 10),
         ObjectClass("thick", CYLINDER, (0.02, 0.02, 0.12), 5)],
        {"easy": Preset((6, 10), (0, 1), (1, 1), (1, 1)), "medium": Preset((11, 16), (1, 3), (1, 2), (2, 2)),
         "hard": Preset((17, 23), (3, 5), (2, 3), (3, 4))},
        (0.0, 0.0, base_z), reach_rows=3)


def confined_shelf(front: float = 0.18, height: float = 0.34, base_z: float = 0.43) -> Environment:
    """App. B.3, Fig. 7: a cubicle (inside 79 x 44 x 34 cm) open at the front, 9 x 4 cells of
    70 mm (two back-right cells unreachable), cylinders r 30 mm, h 20 cm."""
    s = base_z + height
    w, dpt, h = 0.395, 0.44, 0.34
    p = _Parts()
    p.slab(front, front + dpt, -w, w, s)                                       # floor
    p.box(front, front + dpt, -w, w, s + h, s + h + 2 * BOARD)                 # ceiling
    p.box(front + dpt, front + dpt + 2 * BOARD, -w - 2 * BOARD, w + 2 * BOARD, s - 2 * BOARD, s + h + 2 * BOARD)  # back
    p.box(front, front + dpt, w, w + 2 * BOARD, s - 2 * BOARD, s + h + 2 * BOARD)    # left
    p.box(front, front + dpt, -w - 2 * BOARD, -w, s - 2 * BOARD, s + h + 2 * BOARD)  # right
    grid = Region("shelf_interior", s, front + 0.015, 0.0, 9, 4, 0.07, 0.07, frozenset({(8, 2), (8, 3)}))
    return Environment(
        "confined_shelf", "Confined shelf", "Cubicle open at the front (inside 79x44x34 cm), 9x4 grid of 70 mm "
        "cells, bottle-sized cylinders (r 30 mm, h 20 cm). Grasps come in past the cylinders in front.",
        [grid], p.parts, [ObjectClass("bottle", CYLINDER, (0.03, 0.03, 0.20))],
        {"easy": Preset((3, 6), (0, 1), (1, 1), (1, 1)), "medium": Preset((7, 10), (1, 2), (1, 2), (2, 2)),
         "hard": Preset((11, 15), (2, 3), (2, 3), (3, 4))},
        (0.0, 0.0, base_z))


def multi_level_shelf(front: float = 0.30, height: float = 0.20, base_z: float = 0.40) -> Environment:
    """App. B.4, Fig. 8: an open shelf with two 80 x 45 cm decks 24 cm apart over a floor
    compartment split by a separator; grids of 70 mm cells (floor 1x5 + 1x5-1, middle deck
    6x5, top deck 6x5-2); boxes 6 x 6 x 9 cm grasped from the front."""
    s0 = base_z + height           # floor compartment
    s1, s2 = s0 + 0.24, s0 + 0.48  # middle and top decks (paper: 0.34 / 0.58 m, 24 cm apart)
    w, dpt = 0.40, 0.45
    p = _Parts()
    for top in (s0, s1, s2):
        p.slab(front, front + dpt, -w, w, top)
    for x in (front, front + dpt - 0.02):                       # corner posts (no walls)
        for y in (w - 0.02, -w):
            p.box(x, x + 0.02, y, y + 0.02, s0, s2 - 2 * BOARD)
    # floor compartment split by a separator; one column of cells on either side of it
    p.box(front, front + dpt, -BOARD, BOARD, s0, s1 - 2 * BOARD)
    x0 = front + 0.015
    regions = [
        Region("floor_left", s0, x0, 0.10, 1, 5, 0.07, 0.07),
        Region("floor_right", s0, x0, -0.10, 1, 5, 0.07, 0.07, frozenset({(0, 4)})),
        Region("middle_deck", s1, x0, 0.005, 6, 5, 0.07, 0.07),
        Region("top_deck", s2, x0, 0.005, 6, 5, 0.07, 0.07, frozenset({(4, 4), (5, 4)})),
    ]
    return Environment(
        "multi_level_shelf", "Multi-level shelf", "Open two-deck shelf over a split floor compartment "
        "(decks 80x45 cm, 24 cm apart), 67 cells of 70 mm, boxes 6x6x9 cm. Relocate moves the OoI to a cell "
        "elsewhere in the shelf.",
        regions, p.parts, [ObjectClass("box", BOX, (0.03, 0.03, 0.045))],
        {"easy": Preset((2, 4), (0, 1), (1, 1), (1, 1)), "medium": Preset((5, 8), (1, 2), (1, 2), (2, 2)),
         "hard": Preset((9, 13), (2, 4), (2, 3), (3, 4))},
        (0.0, 0.0, base_z))


def access19(front: float = 0.30, height: float = 0.40, base_z: float = 0.40) -> Environment:
    """App. B.5, Fig. 9: a dense bookcase (compartment 39 x 27.8 x 24 cm) whose roof is a top deck;
    5 x 7 cells of 40 x 36 mm on each; boxes 3 x 3 x 10 cm grasped from the front. [lattice] Rows
    are 40 mm deep (and the compartment 30 cm) so cell centres lie on M4M's 1 cm MAPF grid. Blockers
    stand in the paper's fixed pattern (columns 0, 2, 4); the OoI starts deep inside and its
    relocate goal is the top deck's cell (2, 3)."""
    s = base_z + height
    w, dpt, h = 0.195, 0.30, 0.24   # [lattice] 30 cm deep (paper 27.8) for 40 mm rows
    roof = s + h + 0.012                                   # paper: floor 0.34, roof top 0.592
    p = _Parts()
    p.slab(front, front + dpt, -w, w, s)
    p.box(front, front + dpt, -w - 2 * BOARD, w + 2 * BOARD, s + h, roof)       # roof = top deck
    p.box(front + dpt, front + dpt + 2 * BOARD, -w - 2 * BOARD, w + 2 * BOARD, s - 2 * BOARD, roof)
    p.box(front, front + dpt, w, w + 2 * BOARD, s - 2 * BOARD, s + h)
    p.box(front, front + dpt, -w - 2 * BOARD, -w, s - 2 * BOARD, s + h)
    x0 = front + 0.01
    regions = [Region("shelf_interior", s, x0, 0.0, 5, 7, 0.04, 0.04),
               Region("shelf_top", roof, x0, 0.0, 5, 7, 0.04, 0.04)]
    deep = [(i, j) for j in (5, 6) for i in range(5)] + [(2, 4)]
    pattern = frozenset((i, j) for i in (0, 2, 4) for j in range(7))
    return Environment(
        "access19", "Access-19", "Bookcase (compartment 39x27.8x24 cm, roof as top deck), 5x7 cells of 40x36 mm "
        "per level, boxes 3x3x10 cm in the paper's blocker pattern; the OoI starts deep inside and goes to the "
        "top deck's cell (2, 3) in relocate problems.",
        regions, p.parts, [ObjectClass("box", BOX, (0.015, 0.015, 0.05))],
        {"easy": Preset((3, 6), (0, 1), (1, 1), (1, 1)), "medium": Preset((7, 12), (1, 2), (1, 1), (2, 2)),
         "hard": Preset((13, 18), (2, 3), (1, 1), (3, 4))},
        (0.0, 0.0, base_z), start_regions=("shelf_interior",), goal_regions=("shelf_top",),
        target_cells={"shelf_interior": deep}, fixed_goal=("shelf_top", (2, 3)),
        blocker_pattern={"shelf_interior": pattern}, corridor_cols=2)


def stacking(front: float = 0.25, height: float = 0.0, base_z: float = 0.40) -> Environment:
    """App. B.6, Fig. 10: a parts table in front of the arm and a stack table behind it (35 x 35 cm
    each), 10 x 10 cells of 30 mm; cubes, oblongs and longs (30 mm section) lying flat. Targets
    are relocated onto the stack table (single layer: stacking itself is not one of our tasks).
    Blocks are 3 cm tall, so they are grasped from above."""
    s = base_z + height
    p = _Parts()
    p.slab(front, front + 0.35, -0.175, 0.175, s)
    p.box(front + 0.05, front + 0.30, -0.125, 0.125, 0.0, s - 2 * BOARD)
    p.slab(-front - 0.35, -front, -0.175, 0.175, s)
    p.box(-front - 0.30, -front - 0.05, -0.125, 0.125, 0.0, s - 2 * BOARD)
    regions = [Region("parts_table", s, front + 0.025, 0.005, 10, 10, 0.03, 0.03),
               Region("stack_table", s, -front - 0.325, 0.005, 10, 10, 0.03, 0.03)]
    return Environment(
        "stacking", "3D stacking", "Parts table in front and stack table behind the arm (35x35 cm), 10x10 cells "
        "of 30 mm, cubes / oblongs / longs (30, 60, 90 x 30 x 30 mm) grasped from above. Relocate moves blocks "
        "to goal cells on the stack table.",
        regions, p.parts,
        [ObjectClass("cube", BOX, (0.015, 0.015, 0.015), 20),
         ObjectClass("oblong", BOX, (0.015, 0.03, 0.015), 12, (2, 1)),
         ObjectClass("long", BOX, (0.015, 0.045, 0.015), 6, (3, 1))],
        {"easy": Preset((3, 6), (0, 1), (1, 2), (1, 1)), "medium": Preset((7, 10), (1, 2), (2, 3), (2, 2)),
         "hard": Preset((11, 14), (2, 3), (3, 4), (3, 4))},
        (0.0, 0.0, base_z), start_regions=("parts_table",), goal_regions=("stack_table",), around=(2, 1),
        config={"retrieval.approach_pitch_range_deg": [85.0, 90.0], "retrieval.grasp_height_range": [0.02, 0.02],
                "retrieval.grasp_top_margin": 0.01,
                "pick_place.pitches_deg": [90.0]})


ENVIRONMENTS = {f.__name__: f for f in (open_tabletop, confined_shelf, multi_level_shelf, access19, stacking)}


# --------------------------------------------------------------- generation
def _footprint_cells(region: Region, cell: tuple[int, int], cls: ObjectClass, turned: bool) -> list[tuple[int, int]]:
    """Cells covered by an object of ``cls`` anchored at ``cell`` (multi-cell blocks extend right/back)."""
    nc, nr = cls.cells[::-1] if turned else cls.cells
    return [(cell[0] + a, cell[1] + b) for a in range(nc) for b in range(nr)]


def _corridor(env: Environment, region: Region, cell: tuple[int, int]) -> set[tuple[int, int]]:
    """Cells in front of ``cell`` that an object must not stand on for a front grasp of it (for
    grasps from above: the cells around it, where the open fingers and the hand come down)."""
    i, j = cell
    if env.around is not None:
        ac, ar = env.around
        out = {(i + di, j + dj) for di in range(-ac, ac + 1) for dj in range(-ar, ar + 1)} - {cell}
        return {c for c in out if c in set(region.cells())}
    out = {(i + di, jj) for di in range(-env.corridor_cols, env.corridor_cols + 1)
           for jj in range(max(0, j - env.reach_rows), j)}
    out |= {(i - 1, j), (i + 1, j)}  # the fingers straddle the object
    return {c for c in out if c in set(region.cells())}


def generate(env_name: str, task: str = "retrieve", difficulty: str = "medium", seed: int = 0,
             counts: dict | None = None, mass_range=(0.05, 0.5), mu_range=(0.8, 1.2), **env_kw) -> Scene:
    """A random problem in environment ``env_name``.

    ``counts`` overrides the sampled numbers: {"movables", "immovables", "targets", "blockers"}.
    The OoI (first target) and the blockers (movables in a target's access corridor) are placed
    first, immovables never in a corridor (so the target stays accessible in principle), the
    other movables anywhere. Relocate goals are free cells of the goal regions.
    """
    env = ENVIRONMENTS[env_name](**env_kw)
    rng = np.random.default_rng(seed)
    pre = env.presets[difficulty]
    n = {k: int(rng.integers(lo, hi + 1)) for k, (lo, hi) in
         (("movables", pre.movables), ("immovables", pre.immovables), ("targets", pre.targets),
          ("blockers", pre.blockers))}
    if task == "retrieve":
        n["targets"] = 1
    n.update(counts or {})
    weights = np.array([c.weight for c in env.classes], dtype=float)
    weights /= weights.sum()

    regions = {r.name: r for r in env.regions}
    free = {(r.name, c) for r in env.regions for c in r.cells()}
    for rname, cells in env.blocker_pattern.items():
        free -= {(rname, c) for c in regions[rname].cells() if c not in cells}
    placed: list[tuple[str, tuple[int, int], str]] = []   # region, cell, role
    objects = list(env.parts)
    ids: dict = {}
    reserved: set = set()   # corridor cells: no immovables there

    def make(rname: str, cell: tuple[int, int], cls: ObjectClass, turned: bool, role: str) -> SceneObject | None:
        """The object on ``cell``, its centre on the 1 cm MAPF lattice, or None if it would come
        within 1 mm of an object already placed (M4M rounds poses to that lattice: objects
        that touch would overlap there, and no MAPF solution could separate them)."""
        r = regions[rname]
        xs, ys = zip(*(r.center(c) for c in _footprint_cells(r, cell, cls, turned)))
        x, y = round(float(np.mean(xs)), 2), round(float(np.mean(ys)), 2)
        sx, sy, sz = cls.size
        if cls.shape == BOX and turned:
            sx, sy = sy, sx
        h = sz if cls.shape == CYLINDER else 2 * sz
        movable = role in ("movable", "blocker") or (role == "target" and task == "relocate")
        o = SceneObject(len(objects) + 1, cls.shape, MOVABLE if movable else IMMOVABLE, x, y,
                        r.surface_z + h / 2, 0.0, 0.0, 0.0, sx, sy, sz,
                        float(rng.uniform(*mass_range)), float(rng.uniform(*mu_range)), movable)
        if any(footprint_overlap(o, p, margin=0.001) for p in objects
               if not p.is_shelf and abs(p.bottom_z - o.bottom_z) < 0.01):
            return None
        return o

    def take(cands: list, cls: ObjectClass, role: str) -> tuple | None:
        order = rng.permutation(len(cands))
        for k in order:
            rname, cell = cands[k]
            for turned in ([False, True] if cls.cells != (1, 1) else [False]):
                cov = _footprint_cells(regions[rname], cell, cls, turned)
                if all((rname, c) in free for c in cov):
                    o = make(rname, cell, cls, turned, role)
                    if o is None:
                        continue
                    for c in cov:
                        free.discard((rname, c))
                    placed.append((rname, cell, role))
                    objects.append(o)
                    ids[(rname, cell)] = o.id
                    return rname, cell
        return None

    def pick_class() -> ObjectClass:
        return env.classes[int(rng.choice(len(env.classes), p=weights))]

    # 1. targets
    start_regions = env.start_regions or tuple(regions)
    targets = []
    for _ in range(n["targets"]):
        cands = [(rn, c) for rn in start_regions for c in env.target_cells.get(rn, regions[rn].cells())
                 if (rn, c) in free]
        # with blockers to place, start deep enough to have cells in front
        deep = [x for x in cands if env.around is not None or x[1][1] >= min(n["blockers"], env.reach_rows)]
        got = take(deep or cands, pick_class(), "target")
        if got is None:
            break
        targets.append(got)
        reserved |= {(got[0], c) for c in _corridor(env, regions[got[0]], got[1])}

    # 2. relocate goals (free cells, not in front of each other's targets)
    goals = []
    if task == "relocate":
        goal_regions = env.goal_regions or tuple(regions)
        used = set()
        for k, (rn, cell) in enumerate(targets):
            if k == 0 and env.fixed_goal is not None:
                g = env.fixed_goal
            else:
                cands = [(gr, c) for gr in goal_regions for c in regions[gr].cells()
                         if (gr, c) in free and (gr, c) not in used and gr != rn]
                if not cands:  # a single region (tabletop, confined shelf): another cell of it
                    cands = [(gr, c) for gr in goal_regions for c in regions[gr].cells()
                             if (gr, c) in free and (gr, c) not in used]
                if not cands:
                    break
                g = cands[int(rng.integers(len(cands)))]
            used.add(g)
            goals.append(g)
            free.discard(g)
            reserved |= {(g[0], c) for c in _corridor(env, regions[g[0]], g[1])}

    # 3. blockers: movables in the targets' (and goals') corridors, directly in front first
    corridor = [x for x in sorted(reserved) if x in free]
    ends = targets + goals
    if env.around is not None:  # the cells where the fingers come down
        direct = {(rn, (c[0] + d, c[1])) for rn, c in ends for d in (-env.around[0], env.around[0])}
    else:
        direct = {(rn, (c[0], jj)) for rn, c in ends for jj in range(max(0, c[1] - env.reach_rows), c[1])}
    for _ in range(min(n["blockers"], n["movables"])):
        cands = [x for x in corridor if x in free]
        first = [x for x in cands if x in direct]
        if not cands or take(first or cands, pick_class(), "blocker") is None:
            break
    # 4. immovables outside every corridor (objects start only in the start regions)
    for _ in range(n["immovables"]):
        cands = [x for x in sorted(free) if x not in reserved and x[0] in start_regions]
        if not cands or take(cands, pick_class(), "immovable") is None:
            break
    # 5. the remaining movables in the start regions, outside the corridors unless the problem
    #    is hard (so easy / medium problems keep their number of blockers)
    while sum(r[2] in ("movable", "blocker") for r in placed) < n["movables"]:
        cands = [x for x in sorted(free) if x[0] in start_regions and (difficulty == "hard" or x not in reserved)]
        if not cands or take(cands, pick_class(), "movable") is None:
            break

    target_ids = [ids[t] for t in targets]
    if not target_ids:
        raise RuntimeError(f"{env_name}: no room for a target")
    if task == "retrieve":
        task_meta = {"type": "retrieve", "targets": target_ids[:1]}
    else:
        tg = []
        for oid, (gr, gc) in zip(target_ids, goals):
            o = next(o for o in objects if o.id == oid)
            gx, gy = (round(v, 2) for v in regions[gr].center(gc))
            tol = 0.5 * min(regions[gr].pitch_col, regions[gr].pitch_row)
            tg.append({"id": oid, "goal": [gx, gy, regions[gr].surface_z + (o.top_z - o.bottom_z) / 2, 0.0],
                       "tol": [round(tol, 4), -1], "region": gr, "cell": list(gc)})
        task_meta = {"type": "relocate", "targets": tg}
    roles = {}
    for (rname, cell, role) in placed:
        roles.setdefault(role, []).append(ids[(rname, cell)])
    counts_out = {"movables": len(roles.get("movable", [])) + len(roles.get("blocker", []))
                  + (len(target_ids) if task == "relocate" else 0),
                  "immovables": len(roles.get("immovable", [])), "targets": len(target_ids),
                  "blockers": len(roles.get("blocker", []))}
    meta = {"env": env.name, "difficulty": difficulty, "seed": seed, "counts": counts_out,
            "robot": {"base": list(env.base), "yaw": 0.0},
            "regions": [r.to_meta() for r in env.regions],
            "cells": {str(ids[(rn, c)]): [rn, list(c)] for rn, c, *_ in placed},
            "blockers": roles.get("blocker", []), "task": task_meta}
    if env.config:
        meta["config"] = env.config
    name = f"{env.name}_{task}_{difficulty}_{seed:04d}"
    return Scene(objects, target_ids[0], name=name, meta=meta)


def target_feasible(scene: Scene, seed: int = 0) -> bool:
    """Can every target be grasped and taken out with the movables removed (only immovables and
    the environment as obstacles)? The E-M4M papers drop scenes where this fails ("infeasible").
    For relocate tasks the same must hold for the target standing at its goal (the reverse of
    placing it there), so every goal is reachable too."""
    from mamo.retrieval import plan_retrieval
    from mamo.robot.arm import Arm, ObstacleSet
    from mamo.world import World

    checks = [(oid, None) for oid in scene.target_ids]
    if scene.task["type"] == "relocate":
        checks += [(int(g["id"]), g["goal"]) for g in scene.task["targets"]]
    for k, (oid, goal) in enumerate(checks):
        objs = [o.copy() for o in scene.objects]
        if goal is not None:
            o = next(o for o in objs if o.id == oid)
            o.x, o.y, o.z = goal[0], goal[1], goal[2]
        sc = Scene(objs, oid, scene.name, {**scene.meta, "task": {"type": "retrieve", "targets": [oid]}})
        w = World(sc)
        arm = Arm(w)
        # removable: the movables of the problem (for retrieve, those on the target's surface)
        relocate = scene.task["type"] == "relocate"
        movable = {o.id for o in sc.objects if o.movable and o.id != oid and (relocate or sc.is_active(o.id))}
        plan = plan_retrieval(arm, w.cfg, sc.by_id(oid), w.object_states(),
                              ObstacleSet.static({oid} | movable), np.array(w.cfg.robot.home_q),
                              np.random.default_rng(seed + k), sc.shelf)
        if plan is None:
            return False
    return True


def retrieval_blockers(scene: Scene, seed: int = 0) -> list[int] | None:
    """Movables overlapping the NGR of the OoI's retrieval planned with only the immovables (and
    the environment) as obstacles, or None if there is no such retrieval: the quantity the
    E-M4M / I-M4M papers label difficulty by (datasets.classify_scene)."""
    from mamo.ngr import compute_ngr
    from mamo.retrieval import plan_retrieval
    from mamo.robot.arm import Arm, ObstacleSet
    from mamo.world import World

    w = World(scene)
    arm = Arm(w)
    states = w.object_states()
    plan = plan_retrieval(arm, w.cfg, scene.ooi, states, ObstacleSet.static({scene.ooi_id}),
                          np.array(w.cfg.robot.home_q), np.random.default_rng(seed), scene.shelf)
    if plan is None:
        return None
    ngr = compute_ngr(arm, plan, scene.shelf, scene.ooi_id, states, w.cfg.ngr.res, w.cfg.ngr.traj_step,
                      shrink=w.cfg.ngr.shrink)
    return [o.id for o in scene.movables if ngr.intersects(o)]


def make_problem(env_name: str, task: str, difficulty: str, index: int, check: bool = True,
                 tries: int = 40, **kw) -> Scene:
    """Problem number ``index`` of a dataset: generate() with seeds index, index + 10^5, ...
    until (when ``check``) target_feasible() holds and, for retrieve problems, the number of
    movables in the OoI's NGR matches the difficulty (NGR_BLOCKERS)."""
    lo, hi = NGR_BLOCKERS[difficulty]
    for k in range(tries):
        seed = index + 100_000 * k
        sc = generate(env_name, task, difficulty, seed, **kw)
        if check and task == "retrieve":
            blk = retrieval_blockers(sc, seed)
            if blk is None or not lo <= len(blk) <= hi:
                continue
            sc.meta["ngr_blockers"] = blk
        elif check and not target_feasible(sc, seed):
            continue
        sc.name = f"{env_name}_{task}_{difficulty}_{index:04d}"
        sc.meta["feasible"] = bool(check) or None
        return sc
    raise RuntimeError(f"no feasible {env_name} {task} {difficulty} problem in {tries} tries")
