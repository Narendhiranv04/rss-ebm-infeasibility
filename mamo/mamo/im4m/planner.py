"""I-M4M: Improved M4M (Saxena & Likhachev, ICRA 2024).

I-M4M is E-M4M (mamo/em4m, unchanged) with the paper's improvements:

* Depth-first search (Sec. IV-B): OPEN is ordered lexicographically by
  (actions, reexpands) -- more valid rearrangement actions on the path first, then
  fewer re-expansions. Remaining ties use the E-M4M priority, then random
  (mamo_search.hpp OPENCompare).
* Prehensile rearrangements (Sec. IV-A, ``pick_place=True``, the -PnP variants): for
  each object that moves in the MAPF solution and fits in the gripper, a pick-and-place
  edge to the final state of its MAPF path is generated next to the push edge. As in
  mamo_node.cpp, only failed pushes become invalid goals.
* Robust I-M4M (Sec. IV-D, ``robust=RobustSim(...)``): pushes are validated in
  parallel simulations with sampled masses and friction coefficients (robust.py).
* Eagerly lazy evaluation (Sec. IV-C, ``lazy=True``): expanding a fully evaluated
  vertex runs MAPF and adds one *unevaluated* entry per candidate action to OPEN,
  without planning or simulating it. An unevaluated entry is evaluated when it reaches
  the top of OPEN; if valid, the resulting (fully evaluated) state is inserted.
  Unevaluated entries are not hashed (duplicates with different parents are kept);
  fully evaluated states are. Unevaluated entries are ordered as the vertex they
  would create (actions + 1, no re-expansions, priority of the MAPF-suggested
  rearrangement), behind fully evaluated vertices with the same (actions, reexpands).

Everything else -- MAPF with invalid goals, DONE, the push planner, push database,
interaction constraints -- is E-M4M's. The released I-M4M code also runs CBS with
suboptimality weight 1, i.e. the optimal CBS used here.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass

import numpy as np

from mamo.context import MAMOContext
from mamo.em4m.planner import EM4M, Cell, Vertex
from mamo.solution import Solution
from mamo.im4m.pickplace import PickPlaceAction, PickPlaceFailure, PickPlacePlanner, graspable, simulate_pick_place
from mamo.em4m.priority import features, log_priority
from mamo.push import PushFailure
from mamo.simulate import simulate_push


@dataclass
class LazyEdge:
    """An unevaluated rearrangement action (v, object, MAPF path, push | pick_place)."""
    parent: Vertex
    aid: int
    path: list[Cell]
    kind: str
    closed: bool = False   # interface shared with Vertex in OPEN
    version: int = 0


class IM4M(EM4M):
    def __init__(self, ctx: MAMOContext, budget: float | None = None, pick_place: bool = False,
                 lazy: bool = False, robust=None) -> None:
        super().__init__(ctx, ctx.cfg.search.im4m_budget if budget is None else budget)
        self.pick_place = pick_place
        self.lazy = lazy
        self.robust = robust
        self.pnp_planner = PickPlacePlanner(ctx.arm, ctx.cfg, ctx.scene.shelf)
        self.pnp_stats: dict = {}

    # ------------------------------------------------------------- OPEN
    def _push_open(self, v) -> None:
        """Depth-first order: (more actions, fewer re-expansions, evaluated first, priority, random)."""
        v.version += 1
        if isinstance(v, LazyEdge):
            key = (-(v.parent.actions + 1), 0, 1, -self._edge_priority(v))
        else:
            key = (-v.actions, v.noops, 0, -v.priority)
        heapq.heappush(self.open, (*key, float(self.ctx.rng.random()), next(self.counter), v.version, v))

    def _edge_priority(self, e: LazyEdge) -> float:
        """E-M4M priority of the rearrangement the MAPF solution suggests for this edge."""
        states = {k: s.copy() for k, s in e.parent.states.items()}
        x, y = e.parent.agents[e.aid].xy(e.path[-1])
        states[e.aid][0], states[e.aid][1] = x, y
        return log_priority(features(self.ctx.scene_at(states), self.ctx.ngr))

    # ------------------------------------------------------------ expand
    def _expand(self, v: Vertex, deadline_ok) -> None:
        reexpansion = v.kappa is not None
        kappa = self._next_kappa(v)
        if kappa is None:
            self._close(v)
            self._bump("closed_no_new_goals")
            return
        if not self._run_mapf(v, kappa):
            self._close(v)
            self._bump("closed_no_mapf")
            return
        if reexpansion:
            v.noops += 1  # [paper] reexpands: the number of times v has been re-expanded
        for aid, path in v.mapf.paths.items():
            if path[0] == path[-1]:
                continue
            kinds = ["push"]
            if self.pick_place and graspable(self.ctx.scene_at(v.states).by_id(aid), self.cfg):
                kinds.insert(0, "pick_place")  # mamo_node.cpp tries pick-and-place first
            for kind in kinds:
                if self.lazy:
                    self._push_open(LazyEdge(v, aid, path, kind))
                    self._bump("lazy_edges")
                elif deadline_ok():
                    self._evaluate(v, aid, path, kind, deadline_ok)
        self.ctx.world.restore(v.snapshot)
        self._push_open(v)  # stays in OPEN for re-expansion with more invalid goals

    def _evaluate(self, v: Vertex, aid: int, path: list[Cell], kind: str, deadline_ok) -> None:
        """Plan and simulate one rearrangement action from v; insert the successor if valid."""
        ctx = self.ctx
        world = ctx.world
        goal = path[-1]
        if kind == "push":
            hit = self._db_lookup(v, aid, path)
            if hit is not None:
                new_states, action = hit
                world.restore(v.snapshot)
                world.set_object_states(new_states)
                world.set_arm_q(action.retract[-1] if action.retract else action.push[-1])
                world.data.ctrl[world.arm_act] = world.arm_q()
                self._add_successor(v, new_states, world.snapshot(), world.arm_q(), action)
                v.valid.add((aid, goal))
                self._bump("db_hits")
                return
        valid = False
        for _ in range(self.samples if kind == "push" else 1):
            if not deadline_ok():
                break
            world.restore(v.snapshot)
            if kind == "push":
                action = ctx.plan_push(v.states, v.agents[aid], path, v.arm_q)
                failed = isinstance(action, PushFailure)
            else:
                t = time.perf_counter()
                ctx.timers.bump("pick_place_plans")
                obj = ctx.scene_at(v.states).by_id(aid)
                action = self.pnp_planner.plan(obj, np.array(v.agents[aid].xy(goal)), path, v.states, v.arm_q,
                                               ctx.rng, self.pnp_stats)
                ctx.timers.push_planner += time.perf_counter() - t
                failed = isinstance(action, PickPlaceFailure)
            if failed:
                continue
            t = time.perf_counter()
            if kind == "push" and self.robust is not None:
                res = self.robust.simulate_push(world.snapshot(), action)
                if res.valid:  # continue from the selected sample's outcome
                    world.restore(v.snapshot)
                    world.set_object_states(res.object_states)
                    world.set_arm_q(res.arm_q)
                    world.data.ctrl[world.arm_act] = res.arm_q
            elif kind == "push":
                res = simulate_push(ctx.sim, ctx.arm, action, self.cfg)
            else:
                res = simulate_pick_place(ctx.sim, ctx.arm, action, self.cfg)
            ctx.timers.sim += time.perf_counter() - t
            self._bump(f"{kind}_sims")
            if res.valid:
                self._add_successor(v, res.object_states, world.snapshot(), world.arm_q(), action)
                if kind == "push":
                    self._db_store(v.states, res.object_states, res.moved, action, path)
                v.valid.add((aid, goal))
                self._bump(f"valid_{kind}es" if kind == "push" else "valid_pick_places")
                valid = True
                break
            self._bump(f"invalid_{kind}es" if kind == "push" else "invalid_pick_places")
        if not valid and kind == "push":  # E-M4M Line 26; pick-and-place failures are not recorded
            failures = v.invalid.setdefault(aid, {})
            failures[goal] = failures.get(goal, 0) + 1
        world.restore(v.snapshot)

    def _add_successor(self, v: Vertex, states, snap, arm_q, action) -> None:
        key = self._key(states)
        n_actions = v.actions + 1
        prev = self.vertices.get(key)
        if prev is not None:
            if prev.closed or prev.actions < n_actions or prev.kappa is not None:
                return  # expanded vertices keep their parent (see EM4M._create_succs)
            prev.parent, prev.action, prev.actions = v, action, n_actions
            prev.states, prev.snapshot, prev.arm_q = states, snap, arm_q
            self._push_open(prev)
            return
        u = Vertex(next(self.ids), states, snap, arm_q, key, v, action, n_actions, 0)
        self._set_priority(u)
        self.vertices[key] = u
        self._push_open(u)
        self._bump("generated")

    # -------------------------------------------------------------- main
    def solve(self) -> Solution:
        ctx = self.ctx
        t_wall = time.perf_counter()
        if not ctx.setup_ngr():
            return self._result(Solution(False, reason="no retrieval trajectory without movables"), t_wall)
        world = ctx.world
        states = world.object_states()
        root = Vertex(next(self.ids), states, world.snapshot(), world.arm_q(), self._key(states))
        self._set_priority(root)
        self.vertices[root.key] = root
        self._push_open(root)

        def deadline_ok() -> bool:  # [cpp] mamo_search.cpp Solve: wall-clock timeout
            return time.perf_counter() - t_wall < self.budget

        while deadline_ok():
            item = self._top()
            if item is None:
                return self._result(Solution(False, reason="OPEN exhausted"), t_wall)
            heapq.heappop(self.open)
            if isinstance(item, LazyEdge):
                self._bump("lazy_evaluations")
                self._evaluate(item.parent, item.aid, item.path, item.kind, deadline_ok)
                continue
            v = item
            plan = self._done(v)
            if plan is not None:
                actions = []
                u = v
                while u.parent is not None:
                    actions.append(u.action)
                    u = u.parent
                actions.reverse()
                world.restore(v.snapshot)
                return self._result(Solution(True, actions, plan), t_wall, v)
            self._bump("expansions")
            self._expand(v, deadline_ok)
        return self._result(Solution(False, reason="budget exhausted"), t_wall)

    def _result(self, sol: Solution, t_wall: float, goal: Vertex | None = None) -> Solution:
        sol = super()._result(sol, t_wall, goal)
        sol.stats["num_pick_places"] = sum(isinstance(a, PickPlaceAction) for a in sol.pushes)
        sol.stats.update({f"pnp_{k}": n for k, n in self.pnp_stats.items()})
        return sol


def im4m(ctx: MAMOContext, budget: float | None = None, pick_place: bool = False, lazy: bool = False,
         robust=None) -> Solution:
    return IM4M(ctx, budget, pick_place, lazy, robust).solve()
