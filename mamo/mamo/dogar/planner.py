"""DOGAR: backward recursive rearrangement planning (Dogar & Srinivasa, RSS 2011, Algorithm 1).

    plan <- Reconfigure(o, G, NGR, move, avoid, r_next)
      for each action a in the action library:
        avoidVol <- volume of the objects in avoid
        (tau1, nu) <- a(o, G, avoidVol)                       # move o into G, never touching avoidVol
        tau2 <- GoTo(tau1[end], r_next, avoidVol + Volume(o, nu[1]))
        vol <- Volume(robot, tau1 + tau2) + Volume(o, nu)
        move_next <- move + FindPenetrated(vol, obj)
        if move_next is empty: return {tau}
        NGR_next <- NGR + vol ; avoid_next <- avoid + {o}
        for o_i in move_next:                                  # search over orderings
          plan <- Reconfigure(o_i, StablePoses(o_i) - NGR_next, NGR_next, move_next - {o_i}, avoid_next, tau[0])
          if plan: return plan + {tau}
      return empty

Planning is backwards in time: actions found deeper in the recursion are executed first.
Movable objects not in ``avoid`` may be penetrated by the robot and the moved object;
whatever is penetrated is added to ``move`` and must be moved (earlier) out of the way.
Plans are monotone: each object is moved at most once.

This is the DOGAR baseline as the M4M (ICRA 2023, Sec. V-A) and E-M4M (ICAPS 2023)
papers re-implement it for MAMO in a 3D shelf:

* the top-level action is the same OoI retrieval trajectory as M4M (PLANRETRIEVAL with
  only immovable obstacles); its volume is the first NGR
* the action library is the M4M push planner (push.py), validated in the physics
  simulator; the paper's push-grasp/sweep capture regions are replaced by simulation,
  and object poses are known exactly (no uncertainty regions)
* DOGAR knows which objects to move but not where: goals are sampled outside the latest
  NGR -- around the closest free cell (M4M paper, ``goal_sampling="closest"``) or uniformly
  (E-M4M paper, ``"uniform"``)
* a push is valid only if no interaction constraint is violated and the object ends up
  completely outside the NGR; objects in ``avoid`` are treated as immovable
* the action library is sampled (``samples`` push goals per object and call), so the
  search restarts with new samples until a plan is found or the budget is spent

While a push is planned and simulated, the penetrable movables (neither the pushed object
nor in ``avoid``) are removed from the scene: by construction they are moved out of the
way before the push is executed. Execution (execute.py) replays the plan forwards with
every object present.
"""

from __future__ import annotations

import heapq
import time
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from mamo.config import Config
from mamo.context import MAMOContext
from mamo.solution import Solution, _finish
from mamo.mapf.agent import Agent, build_agent
from mamo.ngr import NGR, _surface_samples, _thin, compute_ngr
from mamo.push import PushAction, PushFailure
from mamo.retrieval import RetrievalPlan, Segment
from mamo.robot.arm import ObstacleSet
from mamo.robot.motion import plan as plan_motion
from mamo.scene import IMMOVABLE, Scene
from mamo.simulate import ConstraintMonitor, cartesian_timing

Cell = tuple[int, int]


class _OutOfTime(Exception):
    pass


@dataclass
class Step:
    obj_id: int
    action: PushAction          # tau1 (its approach is filled in when the plan is assembled)
    goto: list[np.ndarray]      # tau2: from the end of tau1 to the start of the next executed action


class Dogar:
    def __init__(self, ctx: MAMOContext, budget: float | None = None, goal_sampling: str = "closest",
                 samples: int = 10) -> None:
        self.ctx = ctx
        self.cfg: Config = ctx.cfg
        self.budget = ctx.cfg.search.em4m_budget if budget is None else budget
        self.goal_sampling = goal_sampling
        self.samples = samples       # sampled push goals per object per call (the action library)
        self.stats: dict[str, int] = {}
        self._obj_local: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------ helpers
    def _bump(self, k: str, n: int = 1) -> None:
        self.stats[k] = self.stats.get(k, 0) + n

    def _check_time(self) -> None:
        if time.perf_counter() - self.t0 >= self.budget:  # wall clock, like the other baselines
            raise _OutOfTime

    def _far(self, states: dict[int, np.ndarray], removed: set[int]) -> dict[int, np.ndarray]:
        """``states`` with the ``removed`` objects parked on the floor far from the shelf."""
        out = {k: s.copy() for k, s in states.items()}
        for i, oid in enumerate(sorted(removed)):
            o = self.ctx.scene.by_id(oid)
            out[oid][:3] = (3.0 + 0.3 * i, 3.0, o.half_height + 0.001)
            out[oid][3:7] = (1.0, 0.0, 0.0, 0.0)
        return out

    def _penetrated(self, vol: NGR, candidates: set[int]) -> list[int]:
        """FindPenetrated(vol, obj): candidates (at their initial poses) intersecting vol."""
        sc = self.ctx.scene_at(self.s0)
        return [oid for oid in sorted(candidates) if vol.intersects(sc.by_id(oid))]

    def _union(self, a: NGR, b: NGR) -> NGR:
        out = NGR(self.ctx.scene.shelf, a.res)
        out.grid = a.grid | b.grid
        return out

    def _agent(self, oid: int, ngr: NGR, avoid: set[int], removed: set[int]) -> Agent:
        """Grid for one object: avoid objects are obstacles, penetrable ones absent, goals outside ngr."""
        objs = []
        for o in self.ctx.scene_at(self.s0).objects:
            if o.id in removed:
                continue
            if o.id in avoid and o.movable:
                o = replace(o, type=IMMOVABLE, movable=False)
            objs.append(o)
        sc = Scene(objs, self.ctx.scene.ooi_id, self.ctx.scene.name, self.ctx.scene.meta)
        return build_agent(sc.by_id(oid), sc, ngr, self.cfg.mapf)

    def _sample_goal(self, agent: Agent, dist: dict[Cell, int]) -> Cell | None:
        """A push goal outside the NGR among the cells the object can reach (``dist``)."""
        cands = np.array([c for c in dist if agent.outside[c]])
        if len(cands) == 0:
            return None
        rng = self.ctx.rng
        if self.goal_sampling == "uniform":  # E-M4M paper: uniformly outside the NGR
            return tuple(int(v) for v in cands[rng.integers(len(cands))])
        # M4M paper: "finds the closest cell outside the latest NGR for an object and samples
        # points around this location" (sigma = the push planner's 2.5 cm).
        start = np.array(agent.start)
        best = cands[np.argmin(np.linalg.norm(cands - start, axis=1))]
        target = best + rng.normal(0.0, self.cfg.push.sigma / agent.res, 2)
        return tuple(int(v) for v in cands[np.argmin(np.linalg.norm(cands - target, axis=1))])

    @staticmethod
    def _dijkstra(agent: Agent) -> tuple[dict[Cell, int], dict[Cell, Cell | None]]:
        """Cheapest paths from the object's start over its grid with the MAPF step cost: the
        single-agent MAPF solution the M4M push planner takes its paths from."""
        valid, cost = agent.valid, agent.cost
        nx, ny = valid.shape
        dist = {agent.start: 0}
        prev: dict[Cell, Cell | None] = {agent.start: None}
        heap = [(0, agent.start)]
        while heap:
            g, c = heapq.heappop(heap)
            if g > dist[c]:
                continue
            for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                n = (c[0] + d[0], c[1] + d[1])
                if 0 <= n[0] < nx and 0 <= n[1] < ny and valid[n]:
                    ng = g + int(cost[n])
                    if ng < dist.get(n, 1 << 62):
                        dist[n], prev[n] = ng, c
                        heapq.heappush(heap, (ng, n))
        return dist, prev

    @staticmethod
    def _path(prev: dict[Cell, Cell | None], goal: Cell) -> list[Cell]:
        path = [goal]
        while prev[path[-1]] is not None:
            path.append(prev[path[-1]])
        return path[::-1]

    def _object_points(self, oid: int) -> np.ndarray:
        if oid not in self._obj_local:
            m = self.ctx.arm.model
            g = self.ctx.world.obj_geom[oid]
            self._obj_local[oid] = _thin(_surface_samples(m, g, 0.004), 0.004)
        return self._obj_local[oid]

    # ---------------------------------------------------------- action
    def _try_push(self, oid: int, ngr: NGR, avoid: set[int]):
        """One push from the action library: (action, final pose, swept object poses) or None."""
        ctx, world = self.ctx, self.ctx.world
        movables = {o.id for o in ctx.scene.movables}
        removed = movables - avoid - {oid}
        states = self._far(self.s0, removed)
        t = time.perf_counter()
        agent = self._agent(oid, ngr, avoid, removed)
        dist, prev = self._dijkstra(agent)
        goal = self._sample_goal(agent, dist)
        path = self._path(prev, goal) if goal is not None else None
        ctx.timers.mapf += time.perf_counter() - t  # goal and path search
        if path is None or len(path) < 2:
            self._bump("no_goal")
            return None
        t = time.perf_counter()
        sc = ctx.scene_at(states)
        action = ctx.pusher.plan(sc.by_id(oid), agent, path, states, ctx.home, ctx.rng, ctx.push_stats,
                                 plan_approach=False)
        ctx.timers.push_planner += time.perf_counter() - t
        ctx.timers.bump("push_plans")
        if isinstance(action, PushFailure):
            return None
        # Simulate tau1 from the push start, with avoid objects immovable and penetrables removed.
        t = time.perf_counter()
        world.restore(self.snap0)
        world.set_object_states(states)
        world.set_arm_q(action.push[0])
        world.data.ctrl[world.arm_act] = action.push[0]
        # The push is planned and replayed with the gripper closed (the replay's approach closes it).
        world.data.qpos[ctx.arm._gripper_qadr] = ctx.arm.gripper_qpos(True)
        world.data.ctrl[world.gripper_act] = 255.0
        mujoco.mj_forward(world.model, world.data)
        monitor = ConstraintMonitor(world, self.cfg, immovable_ooi=True)
        monitor.movables = [o for o in monitor.movables if o.id not in removed]
        for a in avoid:
            monitor.is_immovable[world.obj_geom[a]] = True
        poses: list[np.ndarray] = []
        ctx.sim.recorder = lambda w: poses.append(w.object_states()[oid].copy())
        rc = self.cfg.robot
        segs = [cartesian_timing(ctx.arm, action.push, rc.push_speed, rc.max_joint_speed),
                cartesian_timing(ctx.arm, action.retract, rc.push_speed, rc.max_joint_speed)]
        try:
            res = ctx.sim.run(segs, [True, True], oid, monitor)
        finally:
            ctx.sim.recorder = None
        ctx.timers.sim += time.perf_counter() - t
        self._bump("push_sims")
        if not res.valid or res.target_displacement <= self.cfg.push.min_displacement:
            self._bump("invalid_pushes")
            return None
        final = res.object_states[oid]
        if ngr.intersects(ctx.scene_at({**self.s0, oid: final}).by_id(oid)):
            self._bump("ends_inside_ngr")  # E-M4M paper: the object must leave the NGR completely
            return None
        self._bump("valid_pushes")
        poses.append(final.copy())
        return action, final, poses, states

    def _volume(self, action: PushAction, goto: list[np.ndarray], oid: int, poses: list[np.ndarray],
                states: dict[int, np.ndarray]) -> NGR:
        """Volume(robot, tau1 + tau2) + Volume(o, nu)."""
        ctx = self.ctx
        segs = [Segment("push", action.push + action.retract[1:], True, False), Segment("goto", goto, True, False)]
        vol = compute_ngr(ctx.arm, RetrievalPlan(segs), ctx.scene.shelf, ctx.scene.ooi_id, states,
                          self.cfg.ngr.res, self.cfg.ngr.traj_step)
        local = self._object_points(oid)
        for s in poses:
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, s[3:7])
            vol.add_points(s[:3] + local @ R.reshape(3, 3).T)
        return vol

    # ------------------------------------------------------- recursion
    def _reconfigure(self, oid: int, ngr: NGR, move: list[int], avoid: set[int], r_next: np.ndarray,
                     depth: int) -> list[Step] | None:
        ctx = self.ctx
        self._bump("reconfigure_calls")
        self.stats["max_depth"] = max(self.stats.get("max_depth", 0), depth)
        for _ in range(self.samples):
            self._check_time()
            out = self._try_push(oid, ngr, avoid)
            if out is None:
                continue
            action, final, poses, states = out
            # tau2 = GoTo(tau1 end -> r_next), avoiding the avoid objects and o at its final pose
            t = time.perf_counter()
            ctx.arm.set_gripper(closed=True)
            ctx.arm.set_object_states({**states, oid: final})
            obs = ObstacleSet.everything()  # penetrables are parked far away, so only these remain
            goto = plan_motion(ctx.arm, action.retract[-1], [r_next], obs, ctx.rng,
                               time_limit=self.cfg.retrieval.plan_time)
            ctx.timers.robot_planner += time.perf_counter() - t
            if goto is None:
                self._bump("goto_failed")
                continue
            t = time.perf_counter()
            vol = self._volume(action, goto, oid, poses, states)
            ctx.timers.robot_planner += time.perf_counter() - t
            step = Step(oid, action, goto)
            penetrated = self._penetrated(vol, {o.id for o in ctx.scene.movables} - avoid - {oid})
            move_next = move + [p for p in penetrated if p not in move]
            if not move_next:
                return [step]
            ngr_next = self._union(ngr, vol)
            avoid_next = avoid | {oid}
            for oi in move_next:
                plan = self._reconfigure(oi, ngr_next, [m for m in move_next if m != oi], avoid_next,
                                         action.push[0], depth + 1)
                if plan is not None:
                    return plan + [step]
        return None

    # -------------------------------------------------------------- main
    def solve(self) -> Solution:
        ctx = self.ctx
        t_wall = self.t0 = time.perf_counter()
        if not ctx.setup_ngr():
            return self._result(Solution(False, reason="no retrieval trajectory without movables"), t_wall)
        world = ctx.world
        self.s0 = world.object_states()
        self.snap0 = world.snapshot()
        retrieval = ctx.initial_retrieval
        ooi = ctx.scene.ooi_id
        # Top level: the retrieval is the action for the OoI; its volume is the first NGR.
        ngr = ctx.ngr
        move = self._penetrated(ngr, {o.id for o in ctx.scene.movables})
        if not move:
            return self._result(Solution(True, [], retrieval), t_wall)
        steps = first = None
        try:
            # The action library is sampled, so one pass of the search is incomplete: restart
            # with new samples until a plan is found or the budget is spent (like M4M, which
            # keeps sampling pushes until its time runs out).
            while first is None:
                steps = None
                self._bump("restarts")
                for oi in move:
                    steps = self._reconfigure(oi, ngr, [m for m in move if m != oi], {ooi}, ctx.home, 1)
                    if steps is not None:
                        break
                if steps is None:
                    continue
                # GoTo from home to the first executed push (Alg. 1 starts from the robot's
                # configuration); if there is none, keep searching.
                t = time.perf_counter()
                world.restore(self.snap0)
                ctx.arm.set_gripper(closed=True)
                ctx.arm.set_object_states(self.s0)
                first = plan_motion(ctx.arm, ctx.home, [steps[0].action.push[0]], ObstacleSet.everything(),
                                    ctx.rng, time_limit=self.cfg.retrieval.plan_time)
                ctx.timers.robot_planner += time.perf_counter() - t
                if first is None:
                    self._bump("first_goto_failed")
                    self._check_time()
        except _OutOfTime:
            world.restore(self.snap0)
            return self._result(Solution(False, reason="budget exhausted"), t_wall)
        world.restore(self.snap0)
        # Assemble in execution order: each tau2 is the approach of the next push, and the last
        # tau2 leads home to the retrieval.
        actions = []
        prev = first
        for s in steps:
            a = s.action
            actions.append(PushAction(a.obj_id, a.cells, a.waypoints, a.tool_points, prev, a.push, a.retract,
                                      {**a.info, "dogar": True}))
            prev = s.goto
        plan = RetrievalPlan([Segment("return_home", prev, True, False)] + retrieval.segments)
        return self._result(Solution(True, actions, plan), t_wall)

    def _result(self, sol: Solution, t_wall: float) -> Solution:
        sol = _finish(self.ctx, sol, t_wall)
        sol.stats["planning_time"] = sol.stats["wall_time"]  # the budget is wall-clock time
        sol.stats.update({f"dogar_{k}": v for k, v in self.stats.items()})
        return sol


def dogar(ctx: MAMOContext, budget: float | None = None, goal_sampling: str = "closest",
          samples: int = 10) -> Solution:
    return Dogar(ctx, budget, goal_sampling, samples).solve()
