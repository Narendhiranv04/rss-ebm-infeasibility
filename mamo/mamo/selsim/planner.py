"""SELSIM: Planning with Selective Physics-based Simulation (Saleem & Likhachev, ICRA 2020),
as the MAMO baseline of the M4M papers.

Paper, Algorithm 2 (MAIN, TRACK, RELEVANTOBJECT) around Algorithm 1 (GETSUCCS):

    C_R <- {}                                         relevant objects
    loop:
        objects in C_R -> simulator and collision checker; path <- PLAN()
        if no path: return no solution
        if C_R = O: return path
        TRACK(path) with every object in the simulator:
            simulate the actions in order, collecting interacting pairs P
            on a violated interaction constraint at O_v:
                C_R <- C_R U RELEVANTOBJECT(P, O_v)   BFS from O_v over P to the first object not in C_R
                break (re-plan)
        if tracked without violation: return path

    GETSUCCS(s, a):  collision-check (s, a) against the environment and C_R
        no object touched                -> s'_c
        a fixed object (immovable, OoI) touched, or the robot hits the shelf  -> invalid
        relevant movables touched        -> simulate with only C_R; invalid on a violation, else s'_e

PLAN is the paper's Lazy Weighted A* (Sec. IV-D): successors that need simulation enter OPEN
unevaluated with their (known) cost, and are simulated only when popped for expansion.
The state is the lattice arm configuration plus the poses of the relevant movable objects.

Action space (E-M4M paper: "SELSIM uses simple motion primitives that change only one of
the q degrees-of-freedom of the robot") and search parameters are those of the lab's SMPL
arm planner as the reference release configures it (config/pr2.mprim, pr2_right_arm.yaml,
robot.cpp): 1 deg lattice; long primitives +-7 deg on joints 1-4, short primitives +-4 deg
on all 7 joints, short ones active within 10 cm (TCP) of the start or the goal and long ones
elsewhere (distances as SMPL measures them: BFS cells to the goal, Manhattan cells to the
start); an IK "snap" to the goal pose within 1 cm (BFS distance); epsilon 100; heuristic: 3-D
BFS from the goal position over the immovables (mamo.selsim.heuristic). The action cost is the
paper's path length (1000 per degree of joint-space distance; SMPL charges 1000 per action).
The goal is the pre-grasp pose of the OoI (as for RRT and KPIECE, mamo.kinodynamic); the
grasp, lift and extract that follow it are tracked with the path. Interaction constraints
are E-M4M's full set (mamo.simulate.ConstraintMonitor).
"""

from __future__ import annotations

import heapq
import itertools
import math
import re
import time
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np

from mamo.context import MAMOContext
from mamo.execute import simulate_retrieval
from mamo.kinodynamic import KState, MotionAction, PhysicsProblem, finish
from mamo.selsim.heuristic import RES as HEUR_RES
from mamo.selsim.heuristic import BFSHeuristic
from mamo.solution import Solution
from mamo.world import GROUND, IMMOVABLE_GEOM, MOVABLE_GEOM, OOI_GEOM, ROBOT, SHELF_GEOM, quat_to_rpy

LONG_PRIMS = [(j, 7.0) for j in range(4)]        # [cpp] pr2.mprim: 7 deg on joints 1-4
SHORT_PRIMS = [(j, 4.0) for j in range(7)]       # [cpp] pr2.mprim: 4 deg on all joints
SHORT_DIST_THRESH = 0.10                         # [cpp] short_dist_mprims_thresh
SNAP_THRESH = 0.01                               # [cpp] xyzrpy_snap_dist_thresh
COST_PER_DEG = 1000                              # [paper] path length; [cpp] SMPL cost unit 1000
EPSILON = 100.0                                  # [cpp] robot.cpp planning epsilon
LATTICE = math.radians(1.0)                      # [cpp] planning/discretization: 1 deg


@dataclass
class Node:
    q: np.ndarray
    objects: dict[int, np.ndarray]     # poses of the relevant objects (all others are absent)
    g: float
    parent: tuple | None
    path: list[np.ndarray] | None      # action from the parent
    simulated: bool = False


class SelSim:
    def __init__(self, ctx: MAMOContext, budget: float, t0: float) -> None:
        self.ctx, self.budget, self.t0 = ctx, budget, t0
        self.prob = PhysicsProblem(ctx)
        self.world, self.arm, self.cfg = ctx.world, ctx.arm, ctx.cfg
        self.scene = ctx.scene
        cat = self.world.geom_cat
        self.fixed_ids = {o for o in self.world.obj_geom if cat[self.world.obj_geom[o]] in (IMMOVABLE_GEOM, OOI_GEOM)}
        self.movable_ids = {o for o in self.world.obj_geom if cat[self.world.obj_geom[o]] == MOVABLE_GEOM}
        self.all_ids = self.fixed_ids | self.movable_ids
        self.is_env = np.isin(cat, (GROUND, SHELF_GEOM))
        self.is_robot = cat == ROBOT
        self.geom_obj = self.world.geom_obj
        # Objects outside the planning model are parked far away, resting on the ground plane.
        self.parked = {oid: np.array([30.0 + oid, 30.0, self._half_height(oid) + 1e-3, 1.0, 0.0, 0.0, 0.0])
                       for oid in self.prob.start.objects}
        self.relevant: set[int] = set()
        self.track_reason = ""
        self.stats = {"iterations": 0, "expansions": 0, "plan_sims": 0, "goal_checks": 0, "track_sims": 0,
                      "relevant": []}

    def _half_height(self, oid: int) -> float:
        m, g = self.world.model, self.world.obj_geom[oid]
        s = m.geom_size[g]
        return float(s[2] if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_BOX else s[1])

    def time_left(self) -> float:
        return self.budget - (time.perf_counter() - self.t0)

    # ------------------------------------------------------------- main loop
    def run(self) -> Solution:
        prob = self.prob
        if not prob.setup_goal():
            return Solution(False, reason="no grasp for the OoI")
        if not prob.state_valid(prob.start):
            return Solution(False, reason="start state invalid")
        self.h = BFSHeuristic(self.world, prob.p_pre)
        self.p_start = prob.project(prob.start.q)
        while self.time_left() > 0:
            self.stats["iterations"] += 1
            path = self.plan()
            if path is None:
                return Solution(False, reason="budget exhausted" if self.time_left() <= 0 else "no solution exists")
            ok, actions, retrieval, culprit = self.track(path)
            if ok:
                return Solution(True, actions, retrieval)
            if culprit is None:
                return Solution(False, reason=f"tracking failed, no irrelevant object to blame: {self.track_reason}")
            self.relevant.add(culprit)
            self.stats["relevant"].append(culprit)
            # Once C_R = O a plan is valid as found (Alg. 2 line 36); it is still tracked so the
            # solution carries the simulated object poses and the grasp.
        return Solution(False, reason="budget exhausted")

    # ------------------------------------------------------------------ PLAN
    def _full_objects(self, objs: dict[int, np.ndarray]) -> dict[int, np.ndarray]:
        out = {oid: self.parked[oid] for oid in self.prob.start.objects}
        out.update(objs)
        return out

    def _key(self, q: np.ndarray, objs: dict[int, np.ndarray]) -> tuple:
        """Lattice configuration + relevant movable poses at E-M4M's resolution (1 cm, 5 deg)."""
        sc = self.cfg.search
        lat = tuple(np.round((q - self.prob.start.q) / LATTICE).astype(int))
        poses = tuple((oid,) + tuple(round(float(v) / sc.hash_res_xy) for v in s[:3])
                      + tuple(round(a / sc.hash_res_theta) for a in quat_to_rpy(s[3:7]))
                      for oid, s in sorted(objs.items()) if oid in self.movable_ids)
        return lat, poses

    def _cost(self, q0: np.ndarray, q1: np.ndarray) -> float:
        return COST_PER_DEG * float(np.linalg.norm(q1 - q0)) / LATTICE

    def _actions(self, q: np.ndarray) -> list[np.ndarray]:
        # [smpl] ManipLatticeActionSpace: goal distance = BFS cells x res, start distance =
        # Manhattan cells x res; short primitives near either, the snap within its threshold.
        p = self.prob.project(q)
        d_goal = self.h.cells_to_goal(p) * HEUR_RES
        c, c0 = self.h.cell(p), self.h.cell(self.p_start)
        d_start = np.inf if c is None or c0 is None else HEUR_RES * sum(abs(a - b) for a, b in zip(c, c0))
        near = d_goal <= SHORT_DIST_THRESH or d_start <= SHORT_DIST_THRESH
        out = []
        for j, deg in (SHORT_PRIMS if near else LONG_PRIMS):
            for sgn in (1.0, -1.0):
                q2 = q.copy()
                q2[j] += sgn * math.radians(deg)
                if self.arm.lo[j] <= q2[j] <= self.arm.hi[j]:
                    out.append(q2)
        if d_goal <= SNAP_THRESH:  # [smpl] xyzrpy snap: IK to the goal pose seeded at q
            q2 = self.arm.ik(self.prob.p_pre, self.prob.R_pre, q)
            if q2 is not None:
                out.append(q2)
        return out

    def _check_edge(self, q0: np.ndarray, q1: np.ndarray) -> tuple[bool, set[int]]:
        """Collision check (s, a) at 1 deg against the environment and the relevant objects
        (Alg. 1 lines 2-6). Returns (valid, relevant movables touched)."""
        m, d, w = self.arm.model, self.arm.data, self.world
        n = self.prob.waypoints(q0, q1)  # [smpl] sphere-motion interpolation
        touched: set[int] = set()
        for a in np.linspace(0.0, 1.0, n)[1:]:
            d.qpos[w.arm_qadr] = q0 + a * (q1 - q0)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            k = d.ncon
            if not k:
                continue
            g1, g2, dist = d.contact.geom1[:k], d.contact.geom2[:k], d.contact.dist[:k]
            r1, r2 = self.is_robot[g1], self.is_robot[g2]
            if np.any(r1 & r2 & (dist < 0)):
                return False, touched
            other = np.where(r1 & ~r2, g2, np.where(r2 & ~r1, g1, -1))
            other = other[other >= 0]
            if np.any(self.is_env[other]):
                return False, touched
            for oid in set(self.geom_obj[other].tolist()) - {-1}:
                if oid not in self.relevant:
                    continue           # irrelevant objects are not in the collision checker
                if oid in self.fixed_ids:
                    return False, touched
                touched.add(oid)
        return True, touched

    def _sim(self, node: Node, q1: np.ndarray, objects_present: set[int], pairs: set | None = None):
        """External simulation of one action with only ``objects_present`` in the simulator."""
        prob = self.prob
        objs = node.objects
        recorder = None
        if pairs is not None:
            prob.set_sim_state(KState(node.q, self._full_objects(objs)))
            recorder = _PairRecorder(self.world, pairs)  # after the state is set: initial contacts
        self.ctx.sim.recorder, self.ctx.sim.recorder_every = recorder, 1
        try:
            res = prob.simulate_motion(KState(node.q, self._full_objects(objs)), q1, monitored=objects_present)
        finally:
            self.ctx.sim.recorder, self.ctx.sim.recorder_every = None, 20
        if recorder is not None:
            recorder(self.world)
        objs2 = {oid: s for oid, s in res.object_states.items() if oid in objects_present}
        return res, objs2

    def plan(self) -> list[Node] | None:
        """Lazy Weighted A* in S_sel (paper Sec. IV-D) with GETSUCCS (Alg. 1)."""
        prob = self.prob
        start_objs = {oid: s for oid, s in prob.start.objects.items() if oid in self.relevant}
        self.arm.detach()
        self.arm.set_gripper(closed=False)
        start = Node(prob.start.q.copy(), start_objs, 0.0, None, None)
        k0 = self._key(start.q, start.objects)
        nodes = {k0: start}
        best_g = {k0: 0.0}
        closed: set = set()
        tie = itertools.count()
        h0 = self.h(prob.project(start.q))
        open_ = [(EPSILON * h0 if np.isfinite(h0) else 0.0, next(tie), k0, None)]  # always expand the start
        while open_:
            if self.time_left() <= 0:
                return None
            f, _, key, lazy = heapq.heappop(open_)
            if lazy is not None:                       # unevaluated edge: simulate it now
                parent, q1 = lazy
                self.stats["plan_sims"] += 1
                res, objs = self._sim(parent, q1, set(parent.objects))
                if not res.valid:
                    continue
                key = self._key(q1, objs)
                g = parent.g + self._cost(parent.q, q1)
                if key in closed or g >= best_g.get(key, np.inf):
                    continue
                best_g[key] = g
                nodes[key] = Node(q1.copy(), objs, g, self._key(parent.q, parent.objects), [parent.q, q1], True)
            if key in closed:
                continue
            node = nodes[key]
            closed.add(key)
            self.stats["expansions"] += 1
            if prob.near_goal(node.q) and self._grasp_ok(node):
                return self._path(nodes, key)
            self.arm.set_object_states(self._full_objects(node.objects))
            for q1 in self._actions(node.q):
                ok, touched = self._check_edge(node.q, q1)
                if not ok:
                    continue
                g = node.g + self._cost(node.q, q1)
                h = self.h(prob.project(q1))
                if not np.isfinite(h):
                    continue
                if touched:                            # needs simulation: lazy (Sec. IV-D)
                    heapq.heappush(open_, (g + EPSILON * h, next(tie), None, (node, q1)))
                    continue
                k1 = self._key(q1, node.objects)
                if k1 in closed or g >= best_g.get(k1, np.inf):
                    continue
                best_g[k1] = g
                nodes[k1] = Node(q1, node.objects, g, key, [node.q, q1])
                heapq.heappush(open_, (g + EPSILON * h, next(tie), k1, None))
        return None

    def _grasp_ok(self, node: Node) -> bool:
        """Goal test: at the pre-grasp pose the grasp, lift and extract succeed in simulation with
        the relevant objects (as RRT and KPIECE check it with all of them, mamo.kinodynamic)."""
        prob = self.prob
        self.stats["goal_checks"] += 1
        plan = prob.grasp_plan(node.q)
        if plan is None:
            return False
        # The OoI is always in the simulator here: it is the object being grasped.
        ooi = self.scene.ooi_id
        prob.set_sim_state(KState(node.q, self._full_objects({ooi: prob.start.objects[ooi], **node.objects})))
        t = time.perf_counter()
        res = simulate_retrieval(self.world, self.arm, self.ctx.sim, self.scene, plan, self.cfg,
                                 monitored=set(node.objects))
        self.ctx.timers.sim += time.perf_counter() - t
        self.arm.set_object_states(self._full_objects(node.objects))
        return res.success

    @staticmethod
    def _path(nodes: dict, key) -> list[Node]:
        out = []
        while key is not None:
            out.append(nodes[key])
            key = nodes[key].parent
        return out[::-1]

    # ----------------------------------------------------------------- TRACK
    def track(self, path: list[Node]):
        """Alg. 2 TRACK: every action in the presence of all objects. Actions that touch no
        object are collision checked (Alg. 1 line 3: nothing to simulate), the others simulated."""
        prob = self.prob
        pairs: set[frozenset] = set()
        state = KState(prob.start.q.copy(), dict(prob.start.objects))
        actions: list[MotionAction] = []
        run: list[np.ndarray] = [state.q]
        everything = set(self.all_ids)
        for node in path[1:]:
            q1 = node.path[1]
            self.arm.set_object_states(state.objects)
            touched = self._touched_any(state.q, q1)
            if not touched:
                run.append(q1)
                state = KState(q1, state.objects)
                continue
            if len(run) > 1:
                actions.append(self._action(run, False))
            self.stats["track_sims"] += 1
            res, objs = self._sim(Node(state.q, state.objects, 0, None, None), q1, everything, pairs)
            if not res.valid:
                self.track_reason = res.reason
                return False, None, None, self._culprit(res.reason, pairs)
            actions.append(self._action([state.q, q1], True))
            state = KState(q1, objs)
            run = [q1]
        if len(run) > 1:
            actions.append(self._action(run, False))
        # The grasp that follows the pre-grasp goal, with every object present.
        plan = prob.grasp_plan(state.q)
        if plan is None:
            self.track_reason = "no grasp trajectory from the pre-grasp configuration"
            return False, None, None, None
        self.stats["track_sims"] += 1
        prob.set_sim_state(state)
        rec = _PairRecorder(self.world, pairs)
        self.ctx.sim.recorder, self.ctx.sim.recorder_every = rec, 1
        t = time.perf_counter()
        try:
            res = simulate_retrieval(self.world, self.arm, self.ctx.sim, self.scene, plan, self.cfg)
        finally:
            self.ctx.sim.recorder, self.ctx.sim.recorder_every = None, 20
            self.ctx.timers.sim += time.perf_counter() - t
        rec(self.world)
        if not res.success:
            self.track_reason = res.reason
            return False, None, None, self._culprit(res.reason, pairs, retrieval=True)
        return True, actions, plan, None

    def _action(self, qs: list[np.ndarray], simulated: bool) -> MotionAction:
        return MotionAction(qs, simulated, [self.prob.project(q) for q in qs])

    def _touched_any(self, q0: np.ndarray, q1: np.ndarray) -> bool:
        m, d, w = self.arm.model, self.arm.data, self.world
        n = self.prob.waypoints(q0, q1)
        for a in np.linspace(0.0, 1.0, n)[1:]:
            d.qpos[w.arm_qadr] = q0 + a * (q1 - q0)
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            k = d.ncon
            g1, g2 = d.contact.geom1[:k], d.contact.geom2[:k]
            r1, r2 = self.is_robot[g1], self.is_robot[g2]
            other = np.where(r1 & ~r2, g2, np.where(r2 & ~r1, g1, -1))
            if np.any(self.geom_obj[other[other >= 0]] >= 0):
                return True
        return False

    def _culprit(self, reason: str, pairs: set, retrieval: bool = False) -> int | None:
        """RELEVANTOBJECT(P, O_v): BFS from the object the violated constraint is about, over the
        interactions P, to the first object not in C_R. Extension: when the violation is about no
        object (the robot touching the shelf) or that BFS finds nothing, the BFS starts at the
        robot and may pass through it -- the objects that interacted with the robot caused it."""
        ov = self._violated_object(reason, retrieval)
        if ov is not None:
            found = self._bfs(ov, pairs, through_robot=False)
            if found is not None:
                return found
        self.stats["robot_rooted_bfs"] = self.stats.get("robot_rooted_bfs", 0) + 1
        return self._bfs(-1 if ov is None else ov, pairs, through_robot=True)

    def _bfs(self, root: int, pairs: set, through_robot: bool) -> int | None:
        queue, seen = deque([root]), {root}
        while queue:
            s = queue.popleft()
            if s >= 0 and s not in self.relevant:
                return s
            for p in pairs:
                if s in p:
                    for o in p - {s}:
                        if (o >= 0 or through_robot) and o not in seen:  # -1 is the robot
                            seen.add(o)
                            queue.append(o)
        return None

    def _violated_object(self, reason: str, retrieval: bool = False) -> int | None:
        m = re.match(r"object (\d+) ", reason) or re.search(r": object (\d+) ", reason)
        if m:
            return int(m.group(1))
        if "immovable contact" in reason:
            ids = [int(x) for x in re.findall(r"obj_(\d+)", reason)]
            # while the OoI is carried it is not an immovable obstacle (execute.simulate_retrieval)
            fixed = [o for o in ids if o in self.fixed_ids and not (retrieval and o == self.scene.ooi_id)]
            return fixed[0] if fixed else (ids[0] if ids else None)
        return None


class _PairRecorder:
    """Collects the interacting body pairs (robot = -1, objects) while a simulation runs; called
    every step. Contacts present before the action (objects resting together) are not
    interactions "that arise as a result of the robot's actions" (paper Sec. III-A)."""

    def __init__(self, world, pairs: set) -> None:
        self.w, self.pairs = world, pairs
        self.robot = world.geom_cat == ROBOT
        self.initial = self._current()

    def _current(self) -> set[frozenset]:
        d = self.w.data
        n = d.ncon
        out: set[frozenset] = set()
        if not n:
            return out
        g1, g2, dist = d.contact.geom1[:n], d.contact.geom2[:n], d.contact.dist[:n]
        o1, o2 = self.w.geom_obj[g1], self.w.geom_obj[g2]
        for a, b, r1, r2, dd in zip(o1, o2, self.robot[g1], self.robot[g2], dist):
            if dd >= 0:
                continue
            if a >= 0 and b >= 0 and a != b:
                out.add(frozenset((int(a), int(b))))
            elif a >= 0 and r2:
                out.add(frozenset((int(a), -1)))
            elif b >= 0 and r1:
                out.add(frozenset((int(b), -1)))
        return out

    def __call__(self, world) -> None:
        self.pairs |= self._current() - self.initial


def selsim(ctx: MAMOContext, budget: float) -> Solution:
    t0 = time.perf_counter()
    s = SelSim(ctx, budget, t0)
    sol = s.run()
    st = dict(s.stats)
    st["relevant"] = ",".join(map(str, st["relevant"]))
    return finish(s.prob, ctx, sol, t0, **st)
