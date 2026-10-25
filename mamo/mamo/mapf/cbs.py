"""Conflict-Based Search (Sharon et al. 2015) over the object-agent abstraction.

Both papers use CBS for the MAPF abstraction. Constraints are geometric as in
the reference implementation (conflicts.hpp): "agent a may not overlap agent
b's pose q at time t". The low-level search is space-time A* towards a pseudo-goal
connected to every goal cell; the pseudo-goal edge carries E-M4M's invalid-goal
penalty (zero for M4M). The reference uses h = 0 (Dijkstra, agent_lattice.cpp); here
h is the exact cost-to-go on the static grid (backward Dijkstra from the goal cells,
penalties included), which is consistent, so paths stay optimal and only the search
is faster.
"""

from __future__ import annotations

import heapq
import itertools
import time
from dataclasses import dataclass, field

import numpy as np

from mamo.mapf.agent import Agent, agents_collide

Cell = tuple[int, int]
Path = list[Cell]
MOVES = ((1, 0), (-1, 0), (0, 1), (0, -1), (0, 0))


@dataclass(frozen=True)
class Constraint:
    agent: int          # constrained agent id
    other: int          # agent whose pose must be avoided
    cell: Cell          # other's cell
    t: int


@dataclass
class CTNode:
    constraints: list[Constraint]
    paths: dict[int, Path]
    costs: dict[int, int]
    cost: int = 0
    conflicts: int = 0
    first_conflict: tuple | None = None


@dataclass
class CBSResult:
    paths: dict[int, Path]
    cost: int
    expanded: int
    time: float
    ll_expanded: int = 0


def pose_at(path: Path, t: int) -> Cell:
    return path[t] if t < len(path) else path[-1]


def cost_to_go(agent: Agent) -> np.ndarray:
    """Exact static cost from every cell to the pseudo-goal (inf where unreachable)."""
    valid, cost = agent.valid, agent.cost
    nx, ny = valid.shape
    h = np.full((nx, ny), np.inf)
    heap = []
    for c in map(tuple, np.argwhere(agent.outside)):
        if c in agent.invalid_goals:
            continue
        v = 0 if agent.goal_penalty is None else int(agent.goal_penalty[c])
        if v < h[c]:
            h[c] = v
            heap.append((v, c))
    heapq.heapify(heap)
    while heap:
        d, c = heapq.heappop(heap)
        if d > h[c]:
            continue
        step = int(cost[c])  # moving from a neighbour into c costs cost[c]
        for dx, dy in MOVES[:4]:
            n = (c[0] + dx, c[1] + dy)
            if 0 <= n[0] < nx and 0 <= n[1] < ny and valid[n] and d + step < h[n]:
                h[n] = d + step
                heapq.heappush(heap, (d + step, n))
    return h


def low_level(agent: Agent, constraints: list[Constraint], agents: dict[int, Agent],
              max_t: int, deadline: float, stats: dict | None = None,
              h: np.ndarray | None = None) -> tuple[Path, int] | None:
    """Space-time A* to the pseudo-goal for one agent (``h`` from cost_to_go; computed if None)."""
    mine = [c for c in constraints if c.agent == agent.id]
    by_t: dict[int, list[Constraint]] = {}
    for c in mine:
        by_t.setdefault(c.t, []).append(c)
    last_c = max((c.t for c in mine), default=-1)
    t_cap = last_c + 1  # beyond the last constraint, time no longer matters

    def violates(cell: Cell, t: int) -> bool:
        for c in by_t.get(t, ()):
            if agents_collide(agent, cell, agents[c.other], c.cell):
                return True
        return False

    def can_stay(cell: Cell, t: int) -> bool:
        for c in mine:
            if c.t >= t and agents_collide(agent, cell, agents[c.other], c.cell):
                return False
        return True

    valid, cost = agent.valid, agent.cost
    nx, ny = valid.shape
    penalty = agent.goal_penalty
    start = agent.start
    if h is None:
        h = cost_to_go(agent)
    if violates(start, 0) or not np.isfinite(h[start]):
        return None
    counter = itertools.count()
    # Heap entries: (f, tie, g, cell, t, goal_key). goal_key is None for lattice
    # states and the reached goal's key for pseudo-goal entries (h = 0 there).
    open_: list = [(float(h[start]), next(counter), 0, start, 0, None)]
    best: dict[tuple[Cell, int], int] = {(start, 0): 0}
    parent: dict[tuple[Cell, int], tuple[Cell, int] | None] = {(start, 0): None}
    expanded = 0
    while open_:
        _, _, g, cell, t, goal_key = heapq.heappop(open_)
        if goal_key is not None:
            path = []
            k = goal_key
            while k is not None:
                path.append(k[0])
                k = parent[k]
            path.reverse()
            _count(stats, expanded)
            return path, g
        tk = min(t, t_cap)
        if best.get((cell, tk), 1 << 62) < g:
            continue
        expanded += 1
        if expanded % 2048 == 0 and time.perf_counter() > deadline:
            _count(stats, expanded)
            return None
        if agent.is_goal_cell(cell) and can_stay(cell, t):
            pen = 0 if penalty is None else int(penalty[cell])
            heapq.heappush(open_, (g + pen, next(counter), g + pen, cell, t, (cell, tk)))
        if t >= max_t:
            continue
        for dx, dy in MOVES:
            nc = (cell[0] + dx, cell[1] + dy)
            if not (0 <= nc[0] < nx and 0 <= nc[1] < ny) or not valid[nc]:
                continue
            nt = t + 1
            if violates(nc, nt):
                continue
            ng = g + int(cost[nc])
            ntk = min(nt, t_cap)
            if ng < best.get((nc, ntk), 1 << 62) and np.isfinite(h[nc]):
                best[(nc, ntk)] = ng
                parent[(nc, ntk)] = (cell, tk)
                heapq.heappush(open_, (ng + float(h[nc]), next(counter), ng, nc, nt, None))
    _count(stats, expanded)
    return None


def _count(stats: dict | None, n: int) -> None:
    if stats is not None:
        stats["ll_expanded"] = stats.get("ll_expanded", 0) + n


def find_conflict(paths: dict[int, Path], agents: dict[int, Agent]) -> tuple[int, tuple | None]:
    """Count conflicting agent pairs and return the earliest conflict."""
    ids = sorted(paths)
    n_conf = 0
    first = None
    for a, b in itertools.combinations(ids, 2):
        pa, pb = paths[a], paths[b]
        T = max(len(pa), len(pb))
        for t in range(T):
            ca, cb = pose_at(pa, t), pose_at(pb, t)
            if ca == pa[0] and cb == pb[0]:
                # [cpp] cbs.cpp: pairs still both at their starts are never conflicts (two objects
                # a few mm apart can overlap once their centres are rounded to the 1 cm grid)
                continue
            if agents_collide(agents[a], ca, agents[b], cb):
                n_conf += 1
                if first is None or t < first[0]:
                    first = (t, a, ca, b, cb)
                break
    return n_conf, first


def cbs(agents: dict[int, Agent], time_limit: float = 30.0, max_t: int = 400,
        stats: dict | None = None) -> CBSResult | None:
    t0 = time.perf_counter()
    deadline = t0 + time_limit
    hs: dict[int, np.ndarray] = {}

    def h_of(aid: int) -> np.ndarray:
        if aid not in hs:
            hs[aid] = cost_to_go(agents[aid])
        return hs[aid]

    root = CTNode([], {}, {})
    for aid, ag in agents.items():
        r = low_level(ag, [], agents, max_t, deadline, stats, h_of(aid))
        if r is None:
            return None
        root.paths[aid], root.costs[aid] = r
    root.cost = sum(root.costs.values())
    root.conflicts, root.first_conflict = find_conflict(root.paths, agents)
    counter = itertools.count()
    open_ = [(root.cost, root.conflicts, next(counter), root)]
    expanded = 0
    while open_:
        if time.perf_counter() > deadline:
            return None
        _, _, _, node = heapq.heappop(open_)
        if node.first_conflict is None:
            return CBSResult(node.paths, node.cost, expanded, time.perf_counter() - t0,
                             (stats or {}).get("ll_expanded", 0))
        expanded += 1
        t, a, ca, b, cb = node.first_conflict
        for me, other, ocell in ((a, b, cb), (b, a, ca)):
            cons = node.constraints + [Constraint(me, other, ocell, t)]
            r = low_level(agents[me], cons, agents, max(max_t, t + 1), deadline, stats, h_of(me))
            if r is None:
                continue
            child = CTNode(cons, dict(node.paths), dict(node.costs))
            child.paths[me], child.costs[me] = r
            child.cost = sum(child.costs.values())
            child.conflicts, child.first_conflict = find_conflict(child.paths, agents)
            heapq.heappush(open_, (child.cost, child.conflicts, next(counter), child))
    return None
