"""KPIECE: Kinodynamic Planning by Interior-Exterior Cell Exploration (Sucan & Kavraki,
"A Sampling-Based Tree Planner for Systems With Complex Dynamics", IEEE T-RO 2012), as the
MAMO baseline of the M4M papers.

Paper, Algorithm 2:

    1  nu_0 <- motion of duration 0 holding q_start ; G <- empty Grid ; G.ADDMOTION(nu_0)
    4  for i = 1 .. N:
    5      c <- G.SELECT(0.75)                     exterior cell with prob. 0.75, max Importance
    6      nu <- motion of Motions(c), half-normal over recency (newest first, sigma = m / 3)
    7      s <- state along nu, uniformly
    8      sample a control u and a duration t
    9      forward-propagate (s, u, t') for t' in (0, t]
   10-11   if valid: nu' <- (s, u, t') with t' maximal
   12      if nu' reaches the goal: return the path to nu'
   13      G.ADDMOTION(nu')                         split so that every part lies in one cell
   14-15   P <- alpha + beta * (coverage increase / time spent); score(c) *= min(P, 1)

    Importance(z) = log(I) * score / (S * (1 + |Neighbors(z)|) * Coverage(z))       (Sec. III-D)

with alpha = 0.7, beta = 5 (Sec. III-G); a cell is interior when all 2k neighbours exist
(Sec. III-C); cell sizes are 10% of the bounding box of the projections of 1000 sampled
states (Sec. III-G's initial guess; the paper's statistics-driven re-tuning between runs is
not done); goal biasing (Sec. V-B): a new cell's score is the goal heuristic of the state
that created it, and with probability 0.05 the expansion continues from one of the 30
motions closest to the goal (each in a different cell), chosen as OMPL's control KPIECE1
does (CloseSamples: the closest, which is then re-inserted with an inflated distance).

MAMO setup (reference: em4m-main/test/run_ompl.cpp with OMPL KPIECE1): the state is the arm
plus every object pose, the projection is the TCP position (ProjectionEvaluatorFK) and,
as in the kinematic experiments of Sec. IV-F, "the sampling of controls was replaced with
sampling of random states and connecting to the random states using a local planner": a
state within 0.2 x the joint-space extent (OMPL range) or, with probability 0.05, an IK
solution of the goal (OMPL goal bias); the local planner is STEERACTION (mamo.kinodynamic),
which keeps the motion's valid prefix when it runs into an immovable obstacle (t' maximal)
and simulates motions that touch movable objects. States along a simulated motion are only
known at its end; expanding from an earlier one re-simulates the prefix ("Because
States(nu) is not stored it may be necessary to recompute s", Sec. III-E).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from mamo.context import MAMOContext
from mamo.kinodynamic import KState, MotionAction, PhysicsProblem, finish
from mamo.solution import Solution

ALPHA, BETA = 0.7, 5.0          # [paper] Sec. III-G
EXTERIOR_BIAS = 0.75            # [paper] Sec. III-E: exterior cells 75% of the time
GOOD_MOTIONS = 30               # [paper] Sec. V-B: at most 30 good motions
GOOD_MOTION_BIAS = 0.05         # [paper] Sec. V-B: expand from a good motion 5% of the time
CELL_FRACTION = 0.1             # [paper] Sec. III-G: cell size = 10% of the bounding box
EXTENT_SAMPLES = 1000           # [paper] Sec. III-G: states sampled for the bounding box
CLOSE_INFLATION = 0.55          # [ompl] control KPIECE1 CloseSamples: re-inserted distance factor


@dataclass(eq=False)
class Motion:
    start: KState
    parent: tuple["Motion", int] | None   # (motion, index of the state along it) it starts from
    qs: np.ndarray                         # states along the motion, qs[0] = start (1-deg steps)
    tcp: np.ndarray                        # their TCP positions (the projection)
    end: KState
    simulated: bool                        # touched movables: object poses known at the end only
    h: float = 0.0                         # goal heuristic of the last state (higher = closer)
    cached: dict = field(default_factory=dict)  # index -> object poses of re-simulated prefixes

    @property
    def n(self) -> int:
        return len(self.qs) - 1


@dataclass(eq=False)
class Cell:
    coord: tuple[int, ...]
    iteration: int
    score: float
    coverage: int = 0
    selections: int = 1
    neighbors: int = 0
    parts: list[tuple[Motion, int, int]] = field(default_factory=list)  # (motion, first, last) state indices

    def importance(self) -> float:
        return math.log(self.iteration) * self.score / (
            self.selections * (1 + self.neighbors) * self.coverage)


class Grid:
    """Coverage of the projection space (Sec. III-B, C): instantiated cells and their parts."""

    def __init__(self, sizes: np.ndarray) -> None:
        self.sizes = sizes
        self.cells: dict[tuple[int, ...], Cell] = {}
        self.k = len(sizes)
        self.total_coverage = 0

    def coord(self, p: np.ndarray) -> tuple[int, ...]:
        return tuple(int(v) for v in np.floor(p / self.sizes))

    def interior(self, c: Cell) -> bool:
        return c.neighbors >= 2 * self.k

    def _neighbours(self, z: tuple[int, ...]):
        for i in range(self.k):
            for dz in (-1, 1):
                yield z[:i] + (z[i] + dz,) + z[i + 1:]

    def add_motion(self, nu: Motion, coords: list[tuple[int, ...]], first: int, iteration: int,
                   score: float) -> None:
        """Split ``nu`` into parts lying in one cell each; coords[j] is the cell of state first + j."""
        j0 = 0
        for j in range(1, len(coords) + 1):
            if j == len(coords) or coords[j] != coords[j0]:
                z = coords[j0]
                c = self.cells.get(z)
                if c is None:
                    c = self.cells[z] = Cell(z, iteration, score)
                    for w in self._neighbours(z):
                        if w in self.cells:
                            c.neighbors += 1
                            self.cells[w].neighbors += 1
                c.parts.append((nu, first + j0, first + j - 1))
                c.coverage += j - j0
                self.total_coverage += j - j0
                j0 = j

    def select(self, rng: np.random.Generator) -> Cell:
        """Sec. III-E: exterior with probability 0.75, then the cell of highest importance."""
        ext = [c for c in self.cells.values() if not self.interior(c)]
        intr = [c for c in self.cells.values() if self.interior(c)]
        pool = ext if (rng.random() < EXTERIOR_BIAS and ext) or not intr else intr
        cell = max(pool, key=Cell.importance)
        if cell.score < np.finfo(float).eps:
            # [ompl] Discretization::selectMotion: scores that underflow are all raised again
            for c in self.cells.values():
                c.score += 1.0 + math.log(c.iteration)
            cell = max(pool, key=Cell.importance)
        return cell


def _half_normal_index(rng: np.random.Generator, m: int) -> int:
    """Sec. III-E: floor(h((m / 3)^2)) -th most recent of m items."""
    return min(m - 1, int(abs(rng.normal(0.0, m / 3.0))))


def kpiece(ctx: MAMOContext, budget: float, goal_bias: float = 0.05, range_frac: float = 0.2) -> Solution:
    t0 = time.perf_counter()
    prob = PhysicsProblem(ctx)
    if not prob.setup_goal():
        return finish(prob, ctx, Solution(False, reason="no grasp for the OoI"), t0)
    if not prob.state_valid(prob.start):
        return finish(prob, ctx, Solution(False, reason="start state invalid"), t0)
    rng = prob.rng
    step = range_frac * prob.max_extent
    # Cell sizes: 10% of the TCP's bounding box over random states (Sec. III-G initial guess).
    pts = np.array([prob.project(prob.sample_uniform()) for _ in range(EXTENT_SAMPLES)])
    grid = Grid(np.maximum(np.ptp(pts, axis=0) * CELL_FRACTION, 1e-3))

    def heuristic(q: np.ndarray) -> float:
        return 1.0 / (1.0 + prob.distance_goal(q))

    root = Motion(prob.start, None, prob.start.q[None, :], prob.project(prob.start.q)[None, :], prob.start,
                  False, heuristic(prob.start.q))
    grid.add_motion(root, [grid.coord(prob.project(prob.start.q))], 0, 1, root.h)
    good: list[list] = []  # Sec. V-B: [goal distance, motion, cell] closest first, one per cell
    stats = {"iterations": 0, "resims": 0, "failed_expansions": 0}
    i = 1
    while time.perf_counter() - t0 < budget:
        i += 1
        stats["iterations"] += 1
        c_before, t_before = grid.total_coverage, time.perf_counter()
        # Lines 5-7: cell, motion, state.
        if good and rng.random() < GOOD_MOTION_BIAS:
            nu, cell = _select_good(good)
            j = nu.n
        else:
            cell = grid.select(rng)
            cell.selections += 1
            nu, a, b = cell.parts[len(cell.parts) - 1 - _half_normal_index(rng, len(cell.parts))]
            j = int(rng.integers(a, b + 1))
        s = _state_along(prob, nu, j, stats)
        # Lines 8-11: control = random nearby state (or the goal), propagated by the local planner.
        res = None
        if s is not None:
            x = prob.sample_goal() if rng.random() < goal_bias else prob.sample_near(s.q, step)
            res = prob.steer(s, x, partial=True)
        if res is not None and res.ok:
            new = Motion(s, (nu, j), res.path, res.tcp, res.end, res.simulated, heuristic(res.end.q))
            if prob.is_satisfied(new.end):                                   # line 12
                return finish(prob, ctx, _solution(new, prob), t0, cells=len(grid.cells), **stats)
            coords = [grid.coord(p) for p in res.tcp[1:]]
            grid.add_motion(new, coords, 1, i, new.h)                         # line 13
            _consider(good, 1.0 / new.h - 1.0, new, grid.cells[coords[-1]])
        else:
            stats["failed_expansions"] += 1
        if cell is not None:                                                  # lines 14-15
            dt = max(time.perf_counter() - t_before, 1e-6)
            p = ALPHA + BETA * (grid.total_coverage - c_before) / dt
            cell.score *= min(p, 1.0)
    return finish(prob, ctx, Solution(False, reason="budget exhausted"), t0, cells=len(grid.cells), **stats)


def _state_along(prob: PhysicsProblem, nu: Motion, j: int, stats: dict) -> KState | None:
    """State number j along nu (j = 0 is its start). The object poses inside a simulated motion
    are recomputed by simulating its prefix; None if that prefix violates a constraint."""
    if j == nu.n:
        return nu.end
    if j == 0:
        return nu.start
    if not nu.simulated:
        return KState(nu.qs[j].copy(), nu.start.objects)
    if j not in nu.cached:
        stats["resims"] += 1
        r = prob.simulate_motion(nu.start, nu.qs[j])
        nu.cached[j] = r.object_states if r.valid else None
    objs = nu.cached[j]
    return None if objs is None else KState(nu.qs[j].copy(), objs)


def _consider(good: list[list], d: float, nu: Motion, cell: Cell) -> None:
    """Keep the GOOD_MOTIONS motions closest to the goal (distance d), at most one per cell
    (Sec. V-B); sorted closest first like OMPL's CloseSamples."""
    same = [e for e in good if e[2] is cell]
    if same:
        if same[0][0] <= d:
            return
        good.remove(same[0])
    elif len(good) >= GOOD_MOTIONS:
        if good[-1][0] <= d:
            return
        good.pop()
    good.append([d, nu, cell])
    good.sort(key=lambda e: e[0])


def _select_good(good: list[list]) -> tuple[Motion, Cell]:
    """[ompl] control KPIECE1 CloseSamples::selectMotion: take the closest motion and re-insert
    it with distance (closest + farthest) x 0.55, so the set rotates."""
    d, nu, cell = good.pop(0)
    _consider(good, (d + (good[-1][0] if good else d)) * CLOSE_INFLATION, nu, cell)
    return nu, cell


def _solution(last: Motion, prob: PhysicsProblem) -> Solution:
    """Edges root -> goal: every motion runs up to the state its child starts from."""
    chain = [(last, last.n)]
    while chain[-1][0].parent is not None:
        chain.append(chain[-1][0].parent)
    edges = [MotionAction([nu.qs[0].copy(), nu.qs[j].copy()], nu.simulated, list(nu.tcp[:j + 1:5]) + [nu.tcp[j]])
             for nu, j in reversed(chain) if j > 0]
    return Solution(True, edges, prob.retrieval)
