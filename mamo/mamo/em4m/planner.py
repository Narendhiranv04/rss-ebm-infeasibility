"""E-M4M: best-first search over rearrangements (Saxena & Likhachev, ICAPS 2023).

Algorithm 1 of the paper:

* A vertex v is a rearrangement v.M (plus the robot state its last push ended in).
* DONE(v): v.M clears the initial NGR V(gamma_OoI) (Line 5), or
  PLANRETRIEVAL(v.M U I) succeeds (Line 7).
* EXPANDSTATE(v): CBS with the invalid goals kappa(v) (Lines 13-14); v is closed
  if CBS fails. For every object that moves in the MAPF solution, one
  PLANPUSH + ISVALID (Lines 18-21); a valid push creates a successor, an
  invalid one adds the final state of pi_m to INVALIDGOALS(v) (Line 26).
* v stays in OPEN and is re-expanded. kappa(v) for the next CBS call adds the
  new invalid goals; when every invalid goal was already in the previous call,
  it adds the final states of all valid pushes found from v instead. If neither
  adds anything, v is closed (paper, 'MAPF Abstraction').
* Negative cache: invalid goals are excluded from the goal set and, through a
  pseudo-goal edge, goals near them are penalised (paper, 'Caching Unsuccessful
  Push Actions'; the penalty shape is agent_lattice.cpp checkNNCost).
* Positive cache: a valid push is stored under (O_m, pi_m) with the objects it
  moved; it is reused at another vertex whose CBS solution has the same pi_m for
  O_m when those objects are at the same (discretised) poses and every other
  object is clear of the stored push (paper, 'Caching Successful Push Actions').
  The approach to the stored push is re-planned from that vertex's robot state.
* Priority: Naive Bayes model (priority.py); OPEN ties prefer more actions,
  then more no-ops, then random (mamo_search.hpp).

DONE's PLANRETRIEVAL(v.M U I) runs once per vertex: with the paper's
deterministic MHA* planner a repeat call on the same v.M returns the same result.

Ablations (paper Table 2): NEG-DB = no positive cache, POS-DB = no negative
cache (invalid goals are still excluded, but not penalised), NO-DB = neither,
TIEBREAK = lexicographic (phi1, phi2, sum phi3) priority.
"""

from __future__ import annotations

import heapq
import itertools
import math
import time
from dataclasses import dataclass, field

import numpy as np

from mamo.context import MAMOContext
from mamo.geometry import footprint, shapes_intersect
from mamo.solution import Solution, _finish
from mamo.mapf.agent import Agent
from mamo.mapf.cbs import CBSResult
from mamo.em4m.priority import Features, features, log_priority
from mamo.push import PushAction, PushFailure
from mamo.retrieval import RetrievalPlan, Segment, retrieval_free
from mamo.robot.arm import ObstacleSet
from mamo.robot.motion import plan as plan_motion
from mamo.scene import CYLINDER
from mamo.simulate import simulate_push
from mamo.world import MOVABLE_GEOM, Snapshot, quat_to_rpy

Cell = tuple[int, int]


@dataclass
class Vertex:
    sid: int
    states: dict[int, np.ndarray]          # object poses (x, y, z, quat)
    snapshot: Snapshot                     # full simulator state (robot included)
    arm_q: np.ndarray
    key: tuple
    parent: "Vertex | None" = None
    action: PushAction | None = None
    actions: int = 0
    noops: int = 0
    priority: float = 0.0
    feats: Features | None = None
    closed: bool = False
    try_finalise: bool = False
    mapf: CBSResult | None = None
    agents: dict[int, Agent] | None = None
    invalid: dict[int, dict[Cell, int]] = field(default_factory=dict)   # INVALIDGOALS(v): failures per goal
    valid: set[tuple[int, Cell]] = field(default_factory=set)           # final states of valid pushes from v
    kappa: set[tuple[int, Cell]] | None = None                          # invalid goals of the last CBS call
    version: int = 0


@dataclass
class DBEntry:
    relevant: dict[int, tuple]             # discretised pose of each moved object before the push
    result: dict[int, np.ndarray]          # their poses after the push
    action: PushAction


class EM4M:
    def __init__(self, ctx: MAMOContext, budget: float | None = None) -> None:
        self.ctx = ctx
        self.cfg = ctx.cfg
        # [cpp] em4m-main (and im4m-main) agent_lattice.cpp: unit MAPF step costs (M4M's
        # obstacle-proximity cost is not used); ctx.cfg is this run's own copy.
        self.cfg.mapf.step_cost = "unit"
        sc = self.cfg.search
        self.budget = sc.em4m_budget if budget is None else budget
        self.samples = self.cfg.push.samples
        self.vertices: dict[tuple, Vertex] = {}
        self.open: list = []
        self.counter = itertools.count()
        self.ids = itertools.count()
        self.push_db: dict[tuple, list[DBEntry]] = {}
        self.stats: dict[str, int] = {}

    # ------------------------------------------------------------- helpers
    def _bump(self, k: str, n: int = 1) -> None:
        self.stats[k] = self.stats.get(k, 0) + n

    def _disc(self, oid: int, s: np.ndarray) -> tuple:
        """Discretised pose used for hashing (mamo_node.cpp GetObjectsHash)."""
        sc = self.cfg.search
        o = self.ctx.scene.by_id(oid)
        r, p, y = quat_to_rpy(s[3:7])
        rb, pb = round(r / sc.hash_res_theta), round(p / sc.hash_res_theta)
        key = [oid, round(s[0] / sc.hash_res_xy), round(s[1] / sc.hash_res_xy)]
        if o.shape != CYLINDER or pb or rb:
            key.append(round(y / sc.hash_res_theta) % round(2 * math.pi / sc.hash_res_theta))
            if pb:
                key.append(pb)
            if rb:
                key.append(rb)
        return tuple(key)

    def _key(self, states: dict[int, np.ndarray]) -> tuple:
        return tuple(self._disc(o.id, states[o.id]) for o in self.ctx.scene.movables)

    def _set_priority(self, v: Vertex) -> None:
        v.feats = features(self.ctx.scene_at(v.states), self.ctx.ngr)
        if self.cfg.search.priority == "tiebreak":
            # [paper] lexicographic on (phi1, phi2, sum phi3), lower first; negated so that
            # larger means higher priority like the learned one.
            k = v.feats.tiebreak_key()
            v.priority = (math.inf,) * 3 if k[0] == 0 else tuple(-x for x in k)
        else:
            v.priority = log_priority(v.feats)

    def _push_open(self, v: Vertex) -> None:
        v.version += 1
        rnd = float(self.ctx.rng.random())
        p = v.priority
        pk = tuple(-x for x in p) if isinstance(p, tuple) else (-p,)
        heapq.heappush(self.open, (pk, -v.actions, -v.noops, rnd, next(self.counter), v.version, v))

    def _top(self) -> Vertex | None:
        while self.open:
            *_, version, v = self.open[0]
            if v.closed or version != v.version:
                heapq.heappop(self.open)
                continue
            return v
        return None

    def _close(self, v: Vertex) -> None:
        v.closed = True

    # ---------------------------------------------------------------- DONE
    def _done(self, v: Vertex) -> RetrievalPlan | None:
        ctx = self.ctx
        if not ctx.blockers(v.states):
            # Line 5: v.M clears V(gamma_OoI); return home and execute gamma_OoI.
            t = time.perf_counter()
            ctx.arm.set_gripper(closed=True)
            ctx.arm.set_object_states(v.states)
            home = plan_motion(ctx.arm, v.arm_q, [ctx.home], ObstacleSet.everything(), ctx.rng,
                               time_limit=self.cfg.retrieval.plan_time)
            # NGR voxels are 1 cm, and gamma_OoI was planned against O_I only: a
            # movable just outside the voxels can still graze it, so re-check it.
            if home is not None and not retrieval_free(ctx.arm, ctx.initial_retrieval, ctx.scene.ooi_id,
                                                       v.states, ObstacleSet.everything({ctx.scene.ooi_id})):
                home = None
                self._bump("reused_retrieval_rejected")
            ctx.timers.robot_planner += time.perf_counter() - t
            if home is not None:
                return RetrievalPlan([Segment("return_home", home, True, False)] + ctx.initial_retrieval.segments)
        if v.try_finalise:
            return None
        v.try_finalise = True
        return ctx.retrieve_all(v.states, v.arm_q)

    # ---------------------------------------------------------------- MAPF
    def _next_kappa(self, v: Vertex) -> set[tuple[int, Cell]] | None:
        """kappa(v) for the next CBS call, or None if nothing new can be added (close v)."""
        if v.kappa is None:
            return set()
        invalid = {(oid, g) for oid, goals in v.invalid.items() for g in goals}
        if invalid - v.kappa:
            return v.kappa | invalid
        # Every invalid push was in the last call: also exclude goals of valid pushes.
        if v.valid - v.kappa:
            self._bump("hallucinated", len(v.valid - v.kappa))
            return v.kappa | invalid | v.valid
        return None

    def _apply_constraints(self, agent: Agent, kappa: set[Cell], failures: dict[Cell, int]) -> None:
        agent.invalid_goals = set(kappa)
        if not self.cfg.search.use_negative_cache or not kappa:
            agent.goal_penalty = None
            return
        # Penalty near known invalid goals ([cpp] agent_lattice.cpp checkNNCost): sum of
        # weight / ceil(d) over cached goals in a +-3 cell box, on the pseudo-goal edge, in
        # units of one MAPF step. Weight = failed push samples at that goal (ResetInvalidPushes:
        # the reference tries SAMPLES = 5 pushes per goal, the paper one, so a failed goal
        # weighs 5) and 1 for a hallucinated goal of a valid push (AddHallucinatedConstraint).
        weights = {c: self.cfg.search.invalid_goal_weight * n for c, n in failures.items()}
        for c in kappa:
            weights.setdefault(c, 1)
        b = self.cfg.search.invalid_goal_box
        pen = np.zeros(agent.valid.shape)
        nx, ny = pen.shape
        for (ci, cj), n in weights.items():
            for di in range(-b, b + 1):
                for dj in range(-b, b + 1):
                    i, j = ci + di, cj + dj
                    if not (0 <= i < nx and 0 <= j < ny):
                        continue
                    d = math.hypot(di, dj)
                    if d >= 1:
                        pen[i, j] += n / math.ceil(d)
        agent.goal_penalty = np.ceil(pen * self.cfg.mapf.cost_mult).astype(np.int64)

    def _run_mapf(self, v: Vertex, kappa: set[tuple[int, Cell]]) -> bool:
        """Lines 13-14: RUNMAPF(v, kappa, V(gamma_OoI))."""
        ctx = self.ctx
        if v.agents is None:
            v.agents = ctx.build_agents(v.states)
        for aid, a in v.agents.items():
            self._apply_constraints(a, {g for oid, g in kappa if oid == aid}, v.invalid.get(aid, {}))
        res = ctx.run_mapf(v.agents)
        v.kappa = kappa
        v.mapf = res
        return res is not None

    # ---------------------------------------------------------- push DB
    def _db_lookup(self, v: Vertex, oid: int, path: list[Cell]) -> tuple[dict[int, np.ndarray], PushAction] | None:
        if not self.cfg.search.use_positive_cache:
            return None
        entries = self.push_db.get((oid, tuple(path)))
        if not entries:
            return None
        ctx = self.ctx
        arm = ctx.arm
        for e in entries:
            if any(self._disc(i, v.states[i]) != d for i, d in e.relevant.items()):
                continue
            irrelevant = {o.id for o in ctx.scene.movables if o.id not in e.relevant}
            # Irrelevant objects (at v's poses) must not touch the stored push.
            arm.set_object_states(v.states)
            arm.set_gripper(closed=True)
            obs = ObstacleSet(frozenset({MOVABLE_GEOM}), frozenset(o.id for o in ctx.scene.movables) - irrelevant)
            if not all(arm.collision_free(q, obs) for q in e.action.push + e.action.retract):
                continue
            # [paper] "all other 'irrelevant' objects in u.M are in configurations that will not be
            # affected by gamma_m": the pushed objects' resulting poses must not overlap them either.
            after = ctx.scene_at({**v.states, **e.result})
            if any(_overlap(after.by_id(i), after.by_id(j)) for i in e.result for j in irrelevant):
                self._bump("db_rejected_overlap")
                continue
            # Re-plan the approach from v's robot state (robot.cpp planToPoseGoal).
            t = time.perf_counter()
            approach = plan_motion(arm, v.arm_q, [e.action.push[0]], ObstacleSet.everything(), ctx.rng,
                                   time_limit=self.cfg.push.plan_time)
            ctx.timers.push_planner += time.perf_counter() - t
            if approach is None:
                continue
            new_states = {k: s.copy() for k, s in v.states.items()}
            for i, s in e.result.items():
                new_states[i] = s.copy()
            a = e.action
            action = PushAction(a.obj_id, a.cells, a.waypoints, a.tool_points, approach, a.push, a.retract,
                                a.info | {"from_db": True})
            return new_states, action
        return None

    def _db_store(self, before: dict[int, np.ndarray], after: dict[int, np.ndarray],
                  moved: list[int], action: PushAction, path: list[Cell]) -> None:
        if not self.cfg.search.use_positive_cache:
            return
        relevant = {i: self._disc(i, before[i]) for i in moved if i in before}
        relevant[action.obj_id] = self._disc(action.obj_id, before[action.obj_id])
        entry = DBEntry(relevant, {i: after[i].copy() for i in relevant}, action)
        # Keyed by (O_m, pi_m) as in the paper; the reference keeps one entry per key, we keep all.
        self.push_db.setdefault((action.obj_id, tuple(path)), []).append(entry)

    # ------------------------------------------------------------ expand
    def _expand(self, v: Vertex, deadline_ok) -> None:
        ctx = self.ctx
        kappa = self._next_kappa(v)
        if kappa is None:
            self._close(v)
            self._bump("closed_no_new_goals")
            return
        if not self._run_mapf(v, kappa):
            self._close(v)  # Lines 15-17
            self._bump("closed_no_mapf")
            return
        world = ctx.world
        succs: list[tuple[dict[int, np.ndarray], Snapshot, np.ndarray, PushAction]] = []
        failed = False
        for aid, path in v.mapf.paths.items():  # Line 18
            if path[0] == path[-1]:
                continue
            if not deadline_ok():
                break
            goal = path[-1]
            hit = self._db_lookup(v, aid, path)
            if hit is not None:
                new_states, action = hit
                world.restore(v.snapshot)
                world.set_object_states(new_states)
                world.set_arm_q(action.retract[-1] if action.retract else action.push[-1])
                world.data.ctrl[world.arm_act] = world.arm_q()
                succs.append((new_states, world.snapshot(), world.arm_q(), action))
                v.valid.add((aid, goal))
                self._bump("db_hits")
                continue
            valid = False
            for _ in range(self.samples):
                if not deadline_ok():
                    break
                world.restore(v.snapshot)
                action = ctx.plan_push(v.states, v.agents[aid], path, v.arm_q)  # Line 20
                if isinstance(action, PushFailure):
                    continue
                t = time.perf_counter()
                res = simulate_push(ctx.sim, ctx.arm, action, self.cfg)  # Line 21
                ctx.timers.sim += time.perf_counter() - t
                self._bump("push_sims")
                if res.valid:
                    succs.append((res.object_states, world.snapshot(), world.arm_q(), action))
                    self._db_store(v.states, res.object_states, res.moved, action, path)
                    v.valid.add((aid, goal))
                    self._bump("valid_pushes")
                    valid = True
                    break
                self._bump("invalid_pushes")
            if not valid:  # Line 26
                failures = v.invalid.setdefault(aid, {})
                failures[goal] = failures.get(goal, 0) + 1
                failed = True
        world.restore(v.snapshot)
        self._create_succs(v, succs, failed)

    def _create_succs(self, v: Vertex, succs, failed: bool = False) -> None:
        for states, snap, arm_q, action in succs:
            key = self._key(states)
            n_actions = v.actions + 1
            prev = self.vertices.get(key)
            if prev is not None:
                # [cpp] mamo_search.cpp re-parents a duplicate with no more actions. A vertex that has
                # already been expanded keeps its parent: its children's approaches start from its
                # current arm configuration (the reference's replay teleports the arm instead).
                if prev.closed or prev.actions < n_actions or prev.kappa is not None:
                    continue
                prev.parent, prev.action, prev.actions = v, action, n_actions
                prev.states, prev.snapshot, prev.arm_q = states, snap, arm_q
                continue
            u = Vertex(next(self.ids), states, snap, arm_q, key, v, action, n_actions, 0)
            self._set_priority(u)
            self.vertices[key] = u
            self._push_open(u)
            self._bump("generated")
        # v stays in OPEN for re-expansion with more invalid goals (closed only in _expand).
        # [cpp] mamo_node.cpp / mamo_search.cpp: noops grows when an expansion had a failed push.
        if failed:
            v.noops += 1
        self._push_open(v)

    # -------------------------------------------------------------- main
    def solve(self) -> Solution:
        ctx = self.ctx
        t_wall = time.perf_counter()
        if not ctx.setup_ngr():
            return _finish(ctx, Solution(False, reason="no retrieval trajectory without movables"), t_wall)
        world = ctx.world
        states = world.object_states()
        root = Vertex(next(self.ids), states, world.snapshot(), world.arm_q(), self._key(states))
        self._set_priority(root)
        self.vertices[root.key] = root
        self._push_open(root)

        def deadline_ok() -> bool:  # [cpp] mamo_search.cpp Solve: wall-clock timeout
            return time.perf_counter() - t_wall < self.budget

        while deadline_ok():
            v = self._top()
            if v is None:
                return self._result(Solution(False, reason="OPEN exhausted"), t_wall)
            plan = self._done(v)
            if plan is not None:
                pushes = []
                u = v
                while u.parent is not None:
                    pushes.append(u.action)
                    u = u.parent
                pushes.reverse()
                world.restore(v.snapshot)
                return self._result(Solution(True, pushes, plan), t_wall, v)
            heapq.heappop(self.open)  # re-pushed by _create_succs unless closed
            self._bump("expansions")
            self._expand(v, deadline_ok)
        return self._result(Solution(False, reason="budget exhausted"), t_wall)

    def _result(self, sol: Solution, t_wall: float, goal: Vertex | None = None) -> Solution:
        sol = _finish(self.ctx, sol, t_wall)
        sol.stats["planning_time"] = sol.stats["wall_time"]  # the budget is wall-clock time
        sol.stats.update({f"em4m_{k}": v for k, v in self.stats.items()})
        sol.stats["em4m_vertices"] = len(self.vertices)
        if goal is not None:
            sol.stats["solution_depth"] = goal.actions
        return sol


def _overlap(a, b) -> bool:
    """Do two objects' footprints intersect while their heights overlap?"""
    if a.top_z <= b.bottom_z or b.top_z <= a.bottom_z:
        return False
    return shapes_intersect(footprint(a), footprint(b))


def em4m(ctx: MAMOContext, budget: float | None = None) -> Solution:
    return EM4M(ctx, budget).solve()
