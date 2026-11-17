"""E-M4M (Saxena & Likhachev, ICAPS 2023) on the 2D pick-and-place dataset: Algorithm 1 as in our MuJoCo port
(m4m_em4m_project/mamo/mamo/em4m/planner.py), with the action changed from a simulated push to the dataset's pick-and-place.

Kept from the paper / port:
* NGR = swept footprint of gamma_OoI, the target task planned with every movable ignored; objects in the NGR must be relocated.
* Best-first search over rearrangements (vertex = object arrangement + the arm state its last action ended in).
* DONE(v): the task from v (target from q_ref + connector from the achieved state), tried once per vertex (Lines 5 / 7).
* EXPANDSTATE(v): MAPF over the NGR objects with the invalid goals kappa(v) (Lines 13-14); every object that moves in the solution is tried
  once (Lines 18-21); a failed action adds its goal to INVALIDGOALS(v) (Line 26). v stays in OPEN and is re-expanded with the new invalid
  goals; when nothing new can be excluded, the goals of the valid actions are excluded too ('hallucinated'); then v is closed.
* Negative cache: invalid goals are excluded and goals near them penalised (agent_lattice.cpp checkNNCost: weight / ceil(d), a failed goal
  weighs 5, a hallucinated one 1, cost x COST_MULT 1000).
* Priority: the paper's Naive Bayes model with the reference's fitted constants (priority.py of the port); OPEN ties prefer more actions,
  then more no-ops, then random.

Changed for this dataset (deviations):
* Every motion query runs under the per-call CPU timeout of common.py (the reference's planners time out too).
* PLANPUSH + simulated ISVALID -> one relocation by the dataset's planner (deterministic: one sample per goal instead of SAMPLES = 5;
  a failed goal therefore weighs 5, as a goal whose 5 samples all failed).
* MAPF: pick-and-place does not follow grid paths, so CBS is replaced by its optimal goal assignment (sum of 4-connected lattice distances
  around the other objects, unit step cost, + the negative-cache penalty), solved exactly with linear_sum_assignment.
  Goals: free legal cells outside the NGR. The penalty box is +-1 lattice cell (8 cm) for the reference's +-3 cells of 1 cm.
* phi3 (mass x friction per NGR object) is undefined in the kinematic 2D domain and is omitted from the priority.
* Positive cache: subsumed by the dataset's exact per-root motion cache (an identical relocation query is never re-planned).
* Dataset rules: each object moves at most once, at most 4 relocations."""
from __future__ import annotations

import collections
import heapq
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats
from scipy.optimize import linear_sum_assignment

from .common import MAX_RELOC, Result, free_cells, relaxed_target, start, target_blockers, task_ok

PRIOR, BETA_A, BETA_B, LAMBDA_NUM_OBJS = 0.3084358524, 0.3403101839, 45.0750087432, 1.5634743875   # mamo_search.cpp createDists()
COST_MULT, INVALID_W, BOX = 1000, 5, 1
BIG = 10 ** 12


@dataclass
class Vertex:
    cells: dict
    q: np.ndarray
    seq: list
    key: tuple
    parent: "Vertex | None" = None
    priority: float = 0.0
    noops: int = 0
    closed: bool = False
    checked: bool = False
    invalid: dict = field(default_factory=dict)        # INVALIDGOALS(v): object -> {cell: failures}
    valid: set = field(default_factory=set)            # (object, cell) of valid actions from v
    kappa: set | None = None                           # invalid goals of the last MAPF call
    version: int = 0


class EM4M2D:
    def __init__(self, m, rng):
        self.m, self.rng = m, rng
        self.open, self.cnt, self.vertices = [], itertools.count(), {}
        self.st = collections.Counter()

    # ------------------------------------------------------------------ NGR, features, priority
    def ngr_objects(self, cells) -> list[str]:
        return target_blockers(self.m, self.gamma, cells)

    def priority(self, cells) -> float:
        occ = [c for o, c in cells.items() if o != "t0" and c in self.ngr]
        phi1 = len(occ) / max(1, len(self.ngr))
        if phi1 == 0:
            return math.inf
        phi2 = len(self.ngr_objects(cells))
        x = min(max(phi1, 1e-12), 1 - 1e-12)
        return float(math.log(PRIOR) + stats.beta.logpdf(x, BETA_A, BETA_B) + stats.expon.logpdf(phi2, scale=1 / LAMBDA_NUM_OBJS))

    def push_open(self, v: Vertex) -> None:
        v.version += 1
        heapq.heappush(self.open, (-v.priority, -len(v.seq), -v.noops, float(self.rng.random()), next(self.cnt), v.version, v))

    def top(self) -> Vertex | None:
        while self.open:
            v, ver = self.open[0][-1], self.open[0][-2]
            if v.closed or ver != v.version:
                heapq.heappop(self.open)
                continue
            return v
        return None

    # ------------------------------------------------------------------ MAPF (goal assignment)
    def next_kappa(self, v: Vertex):
        if v.kappa is None:
            return set()
        invalid = {(o, g) for o, gs in v.invalid.items() for g in gs}
        if invalid - v.kappa:
            return v.kappa | invalid
        if v.valid - v.kappa:
            self.st["hallucinated"] += len(v.valid - v.kappa)
            return v.kappa | invalid | v.valid
        return None

    def lattice_dist(self, cells: dict, src: int, agents) -> dict:
        """4-connected BFS on the lattice; cells of non-agent objects are blocked (they are obstacles of the MAPF instance)."""
        lat = self.m.lat
        blocked = {c for o, c in cells.items() if o not in agents}
        legal = set(self.m.legal)
        d, dq = {src: 0}, collections.deque([src])
        while dq:
            c = dq.popleft()
            r, k = divmod(c, lat.cols)
            for rr, kk in ((r + 1, k), (r - 1, k), (r, k + 1), (r, k - 1)):
                n = rr * lat.cols + kk
                if 0 <= rr < lat.rows and 0 <= kk < lat.cols and n in legal and n not in blocked and n not in d:
                    d[n] = d[c] + 1
                    dq.append(n)
        return d

    def penalty(self, v: Vertex, o: str, kappa_o: set) -> dict:
        w = {c: INVALID_W * n for c, n in v.invalid.get(o, {}).items()}
        for c in kappa_o:
            w.setdefault(c, 1)
        cols = self.m.lat.cols
        pen = collections.Counter()
        for c, n in w.items():
            r0, k0 = divmod(c, cols)
            for dr in range(-BOX, BOX + 1):
                for dk in range(-BOX, BOX + 1):
                    d = math.hypot(dr, dk)
                    if d >= 1:
                        pen[(r0 + dr) * cols + (k0 + dk)] += n / math.ceil(d)
        return {c: math.ceil(p * COST_MULT) for c, p in pen.items()}

    def mapf(self, v: Vertex, agents: list[str], kappa: set):
        goals = [c for c in free_cells(self.m, v.cells) if c not in self.ngr]
        if len(goals) < len(agents):
            return None
        C = np.full((len(agents), len(goals)), BIG, dtype=np.int64)
        for i, o in enumerate(agents):
            ko = {g for a, g in kappa if a == o}
            d = self.lattice_dist(v.cells, v.cells[o], set(agents))
            pen = self.penalty(v, o, ko)
            for j, g in enumerate(goals):
                if g in d and g not in ko:
                    C[i, j] = COST_MULT * d[g] + pen.get(g, 0)
        ri, ci = linear_sum_assignment(C)
        if any(C[i, j] >= BIG for i, j in zip(ri, ci)):
            return None
        return {agents[i]: goals[j] for i, j in zip(ri, ci)}

    # ------------------------------------------------------------------ search
    def expand(self, v: Vertex) -> None:
        m = self.m
        moved = {o for o, _ in v.seq}
        agents = sorted(o for o in self.ngr_objects(v.cells) if o not in moved)
        if len(v.seq) >= MAX_RELOC or not agents:
            v.closed = True
            self.st["closed_no_moves"] += 1
            return
        kappa = self.next_kappa(v)
        if kappa is None:
            v.closed = True
            self.st["closed_no_new_goals"] += 1
            return
        sol = self.mapf(v, agents, kappa)
        v.kappa = kappa
        if sol is None:
            v.closed = True
            self.st["closed_no_mapf"] += 1
            return
        failed = False
        for o in agents:                                                   # Line 18
            g = sol[o]
            r = m.relocate(v.cells, v.q, o, g)                              # Lines 20-21: the dataset's planner (capped)
            if r is None:                                                   # Line 26
                v.invalid.setdefault(o, {})[g] = v.invalid.get(o, {}).get(g, 0) + 1
                failed = True
                self.st["invalid_actions"] += 1
                continue
            self.st["valid_actions"] += 1
            v.valid.add((o, g))
            cells = {**v.cells, o: g}
            key = tuple(sorted(cells.items()))
            prev = self.vertices.get(key)
            if prev is not None:
                if prev.closed or len(prev.seq) < len(v.seq) + 1 or prev.kappa is not None:
                    continue
                prev.parent, prev.seq, prev.q = v, v.seq + [(o, g)], r[0]
                continue
            u = Vertex(cells, r[0], v.seq + [(o, g)], key, v, self.priority(cells))
            self.vertices[key] = u
            self.push_open(u)
            self.st["generated"] += 1
        if failed:
            v.noops += 1
        self.push_open(v)

    def solve(self) -> Result:
        m = self.m
        self.gamma = relaxed_target(m)
        r = start(m, self.gamma)
        if r:
            return r
        self.ngr = m.swept(self.gamma, "t0", m.cells0["t0"], m.legal)
        root = Vertex(dict(m.cells0), m.qref, [], tuple(sorted(m.cells0.items())))
        root.priority = self.priority(root.cells)
        root.checked = True                                                # the start check above is DONE(root)
        self.vertices[root.key] = root
        self.push_open(root)
        while True:
            v = self.top()
            if v is None:
                return Result("ABSTAIN", [], "OPEN exhausted", dict(self.st))
            if not v.checked:                                              # DONE(v), once per vertex
                v.checked = True
                self.st["done_checks"] += 1
                if task_ok(m, self.gamma, v.cells, v.q):
                    return Result("REPAIR", list(v.seq), "done", dict(self.st, vertices=len(self.vertices)))
            heapq.heappop(self.open)
            self.st["expansions"] += 1
            self.expand(v)


def solve(m, rng) -> Result:
    return EM4M2D(m, rng).solve()
