"""Physics-based MAMO planning in the full state space (arm joints + object poses).

The machinery the sampling-based baselines RRT and KPIECE share, ported from the
reference release (em4m-main/src/sampling/ompl.cpp and robot.cpp), where both are
OMPL planners searching "the entire MAMO state space X by randomly sampling robot
motions" (M4M paper Sec. V, E-M4M paper Sec. 'Experimental Results'):

* state = arm configuration + pose of every object (PushplanStateSpace); distance,
  sampling and interpolation use the joints only, objects ride along
* a motion is checked by STEERACTION (robot.cpp): the straight joint-space line is
  interpolated so that no point of the robot moves more than 1.745 cm between waypoints
  (SMPL's sphere-motion bound); a self-colliding waypoint invalidates it; a waypoint touching an immovable obstacle (shelf, immovables,
  OoI) ends the motion -- RRT rejects it, KPIECE keeps the valid prefix; a motion that
  touches a movable object is forward simulated in MuJoCo and is valid if no
  interaction constraint is violated (mamo.simulate.ConstraintMonitor), its end state
  carrying the simulated object poses; a motion touching nothing is not simulated
* the goal (PushplanPoseGoal) is the OoI pre-grasp pose of one grasp sampled at the
  start (robot.cpp ComputeGrasps), within 1.5 cm per axis and 0.05 rad, and the grasp
  must succeed in simulation from there (CHECKGRASPTRAJ: reach, grasp, lift, extract)
* KPIECE projects states to the TCP position (ProjectionEvaluatorFK)

A solution is a list of MotionAction edges (replayed by mamo.execute) and the retrieval.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mamo.context import MAMOContext
from mamo.retrieval import RetrievalPlan, Segment
from mamo.robot.arm import ObstacleSet
from mamo.robot.motion import cartesian_path
from mamo.simulate import ConstraintMonitor, SimResult, time_parameterise
from mamo.solution import Solution
from mamo.world import GROUND, IMMOVABLE_GEOM, MOVABLE_GEOM, OOI_GEOM, ROBOT, SHELF_GEOM

STEER_RES = math.radians(1.0)        # [cpp] SteerAction: res = DEG5 / 5 = 0.01745, which SMPL's
                                     # fillMotionInterpolation uses as metres of collision-sphere motion
GOAL_POS_TOL = 0.015                 # [cpp] run_ompl.cpp: goal position tolerance per axis (m)
GOAL_ROT_TOL = 0.05                  # [cpp] run_ompl.cpp: goal orientation tolerance (rad)
STATIC_CATS = (GROUND, SHELF_GEOM, IMMOVABLE_GEOM, OOI_GEOM)


@dataclass
class KState:
    q: np.ndarray                       # arm joints
    objects: dict[int, np.ndarray]      # pose (x, y, z, qw, qx, qy, qz) of every object


@dataclass
class SteerResult:
    ok: bool
    end: KState | None = None
    path: np.ndarray | None = None      # 1-deg waypoints from the start to the end (rows)
    tcp: np.ndarray | None = None       # TCP position at every waypoint
    frac: float = 0.0                   # fraction of the requested motion that is valid
    simulated: bool = False             # the motion touched a movable object and was simulated
    reason: str = ""


@dataclass
class MotionAction:
    """One straight joint-space edge of a tree solution (gripper open)."""
    path: list[np.ndarray]
    simulated: bool                     # touched movables during planning: replay settles after it
    tool_points: list = field(default_factory=list)  # TCP positions along the edge (figures)
    obj_id: int = -1
    kind: str = "motion"


@dataclass
class Counters:
    steers: int = 0
    sims: int = 0
    goal_checks: int = 0
    extra: dict = field(default_factory=dict)


class PhysicsProblem:
    """The MAMO problem as the reference OMPL baselines see it (one per planning run)."""

    def __init__(self, ctx: MAMOContext) -> None:
        self.ctx = ctx
        self.world, self.arm, self.sim, self.cfg = ctx.world, ctx.arm, ctx.sim, ctx.cfg
        self.rng = ctx.rng
        self.lo, self.hi = self.arm.lo, self.arm.hi
        cat = self.world.geom_cat
        self.is_movable = cat == MOVABLE_GEOM
        self.is_static = np.isin(cat, STATIC_CATS)
        self.is_robot = cat == ROBOT
        self.start = KState(np.array(self.cfg.robot.home_q), self.world.object_states())
        self.count = Counters()
        self.joint_radius = self._joint_radii()
        self.goal_q: list[np.ndarray] = []           # fixed IK solutions of the pre-grasp pose
        self._last_ik: np.ndarray | None = None
        self.p_pre = self.R_pre = self.p_grasp = self.R_grasp = None
        self.lift_h = 0.0
        self.p_out = None
        self.retrieval: RetrievalPlan | None = None  # grasp trajectory of the last satisfied goal check

    # -------------------------------------------------------------- state space
    @property
    def max_extent(self) -> float:
        """OMPL getMaximumExtent: sum of the joint ranges (all joints are bounded here)."""
        return float(np.sum(self.hi - self.lo))

    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """[cpp] PushplanStateSpace::distance: Euclidean in joint space, objects ignored."""
        return float(np.linalg.norm(a - b))

    def sample_uniform(self) -> np.ndarray:
        return self.rng.uniform(self.lo, self.hi)

    def sample_near(self, q: np.ndarray, dist: float) -> np.ndarray:
        """[cpp] sampleUniformNear: every joint uniform within +-dist, clipped to its limits."""
        return self.rng.uniform(np.maximum(self.lo, q - dist), np.minimum(self.hi, q + dist))

    def project(self, q: np.ndarray) -> np.ndarray:
        """[cpp] ProjectionEvaluatorFK: TCP position."""
        return self.arm.fk(q)[0]

    def _joint_radii(self) -> np.ndarray:
        """[smpl] RobotMotionCollisionModel: per arm joint, the radius about its axis anchor of a
        sphere holding every collision geom it moves (its own link's, and the next joint's sphere)."""
        m, d, w = self.arm.model, self.arm.data, self.world
        d.qpos[w.arm_qadr] = self.cfg.robot.home_q
        mujoco.mj_kinematics(m, d)
        joints = list(w.arm_joint_ids)
        bodies = [m.jnt_bodyid[j] for j in joints]

        def in_subtree(b: int, root: int) -> bool:
            while b > 0:
                if b == root:
                    return True
                b = m.body_parentid[b]
            return False

        geoms = [g for g in range(m.ngeom) if self.is_robot[g] and (m.geom_contype[g] or m.geom_conaffinity[g])]
        radii = np.zeros(len(joints))
        for i in range(len(joints) - 1, -1, -1):
            a = d.xanchor[joints[i]]
            nxt = bodies[i + 1] if i + 1 < len(joints) else None
            r = 0.0
            for g in geoms:
                b = m.geom_bodyid[g]
                if in_subtree(b, bodies[i]) and (nxt is None or not in_subtree(b, nxt)):
                    r = max(r, float(np.linalg.norm(d.geom_xpos[g] - a)) + float(m.geom_rbound[g]))
            if nxt is not None:
                r = max(r, float(np.linalg.norm(d.xanchor[joints[i + 1]] - a)) + radii[i + 1])
            radii[i] = r
        return radii

    def waypoints(self, a: np.ndarray, b: np.ndarray) -> int:
        """[smpl] fillMotionInterpolation: ceil(sum_j R_j |dq_j| / res) + 1 waypoints."""
        motion = float(np.sum(self.joint_radius * np.abs(b - a)))
        return max(2, int(math.ceil(motion / STEER_RES)) + 1)

    # ------------------------------------------------------------- collisions
    def _touches(self, q: np.ndarray) -> tuple[bool, bool]:
        """(touches a movable, touches a static obstacle or itself) at q -- one collision query
        on the planning model, whose object poses were set by the caller."""
        m, d, w = self.arm.model, self.arm.data, self.world
        d.qpos[w.arm_qadr] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        self.arm.n_checks += 1
        n = d.ncon
        if n == 0:
            return False, False
        g1, g2, dist = d.contact.geom1[:n], d.contact.geom2[:n], d.contact.dist[:n]
        r1, r2 = self.is_robot[g1], self.is_robot[g2]
        other = np.where(r1 & ~r2, g2, np.where(r2 & ~r1, g1, -1))
        if np.any(r1 & r2 & (dist < 0)):
            # [cpp] both of the reference's collision spaces check self-collision, so a
            # self-colliding waypoint trips both flags and the whole motion fails
            return True, True
        valid = other >= 0
        mov = bool(np.any(self.is_movable[other[valid]]))
        static = bool(np.any(self.is_static[other[valid]]))
        return mov, static

    def state_valid(self, s: KState) -> bool:
        """[cpp] StateValidityChecker: no contact with immovable obstacles (movables may touch)."""
        self.arm.detach()
        self.arm.set_gripper(closed=False)
        self.arm.set_object_states(s.objects)
        return not self._touches(s.q)[1]

    # ---------------------------------------------------------------- steering
    def steer(self, frm: KState, to: np.ndarray, partial: bool) -> SteerResult:
        """[cpp] Robot::SteerAction. ``partial`` (KPIECE): a motion cut short by an immovable
        obstacle is valid up to the last free waypoint; otherwise (RRT) it is invalid."""
        self.count.steers += 1
        n = self.waypoints(frm.q, to)
        path = frm.q + (to - frm.q) * np.linspace(0.0, 1.0, n)[:, None]
        self.arm.detach()
        self.arm.set_gripper(closed=False)
        self.arm.set_object_states(frm.objects)
        tcp = np.empty((n, 3))
        mov_collision = False
        immov_idx = -1
        site = self.world.tcp_site
        for i in range(n):
            mov, static = self._touches(path[i])
            tcp[i] = self.arm.data.site_xpos[site]
            mov_collision = mov_collision or mov
            if static:
                immov_idx = i
                if not partial:
                    return SteerResult(False, reason="immovable collision")
                if mov_collision:
                    return SteerResult(False, reason="movable and immovable collision")
                break
        if immov_idx == 0:
            return SteerResult(False, reason="start collides")
        size = n if immov_idx < 0 else immov_idx
        if size < 2:  # a motion that does not move (the reference would add a copy of the start)
            return SteerResult(False, reason="no valid prefix")
        path, tcp = path[:size], tcp[:size]
        frac = size / n
        if not mov_collision:
            return SteerResult(True, KState(path[-1].copy(), frm.objects), path, tcp, frac)
        res = self.simulate_motion(frm, path[-1])
        if not res.valid:
            return SteerResult(False, reason=res.reason, simulated=True)
        return SteerResult(True, KState(path[-1].copy(), res.object_states), path, tcp, frac, True)

    def set_sim_state(self, s: KState) -> None:
        """Put the simulator at rest in state ``s`` (gripper open, no grasp)."""
        w, d = self.world, self.world.data
        w.reset()
        w.set_arm_q(s.q)
        d.ctrl[w.arm_act] = s.q
        d.qpos[self.arm._gripper_qadr] = self.arm.gripper_qpos(False)
        w.set_object_states(s.objects)
        mujoco.mj_forward(w.model, d)

    def simulate_motion(self, frm: KState, q_end: np.ndarray, monitored: set[int] | None = None,
                        settle: bool = True) -> SimResult:
        """[cpp] SimPushes of the steer action: track frm.q -> q_end, then settle (2 s).
        ``monitored``: the movable objects whose constraints are checked (default: all)."""
        t = time.perf_counter()
        self.set_sim_state(frm)
        monitor = ConstraintMonitor(self.world, self.cfg, immovable_ooi=True)  # after the state is set
        if monitored is not None:
            monitor.movables = [o for o in monitor.movables if o.id in monitored]
        timing = time_parameterise([frm.q, q_end], self.cfg.robot.max_joint_speed)
        res = self.sim.run([timing], [False], None, monitor, settle=settle)
        self.ctx.timers.sim += time.perf_counter() - t
        self.count.sims += 1
        return res

    # -------------------------------------------------------------------- goal
    def setup_goal(self) -> bool:
        """[cpp] Robot::ComputeGrasps: one OoI grasp (pre-grasp, grasp, lift and extract poses),
        here from PLANRETRIEVAL with only the immovable obstacles."""
        t = time.perf_counter()
        plan = self.ctx.retrieve(self.start.objects, ObstacleSet.static({self.ctx.scene.ooi_id}), self.start.q)
        self.ctx.timers.robot_planner += time.perf_counter() - t
        if plan is None:
            return False
        seg = {s.name: s for s in plan.segments}
        self.p_pre, self.R_pre = self.arm.fk(seg["reach"].path[0])
        self.p_grasp, self.R_grasp = self.arm.fk(seg["reach"].path[-1])
        self.lift_h = float(self.arm.fk(seg["lift"].path[-1])[0][2] - self.p_grasp[2])
        self.p_out = self.arm.fk(seg["extract"].path[-1])[0]
        # A fixed set of goal configurations for distanceGoal (the reference draws a new random-seed
        # IK solution on every call; a fixed set keeps distances comparable over the run).
        self.goal_q = [seg["reach"].path[0].copy()]
        for _ in range(20):
            q = self.arm.ik(self.p_pre, self.R_pre, self.sample_uniform())
            if q is not None and all(self.distance(q, g) > 1e-3 for g in self.goal_q):
                self.goal_q.append(q)
        self._last_ik = self.goal_q[0]
        return True

    def near_goal(self, q: np.ndarray) -> bool:
        p, R = self.arm.fk(q)
        if np.any(np.abs(p - self.p_pre) > GOAL_POS_TOL):
            return False
        qa, qb = np.zeros(4), np.zeros(4)
        mujoco.mju_mat2Quat(qa, R.flatten())
        mujoco.mju_mat2Quat(qb, self.R_pre.flatten())
        return 2.0 * math.acos(min(1.0, abs(float(qa @ qb)))) < GOAL_ROT_TOL

    def distance_goal(self, q: np.ndarray) -> float:
        """[cpp] PushplanPoseGoal::distanceGoal: joint distance to an IK solution of the goal pose
        (here the nearest of the fixed set from setup_goal)."""
        return min(self.distance(q, g) for g in self.goal_q)

    def sample_goal(self) -> np.ndarray:
        """[cpp] PushplanPoseGoal::sampleGoal / RunIK: IK of the pre-grasp pose from a random seed
        (up to 5 tries, else the last solution returned)."""
        for _ in range(5):
            q = self.arm.ik(self.p_pre, self.R_pre, self.sample_uniform())
            if q is not None:
                self._last_ik = q
                return q.copy()
        return self._last_ik.copy()

    def grasp_plan(self, q: np.ndarray) -> RetrievalPlan | None:
        """[cpp] Robot::ComputeGraspTraj: to the exact pre-grasp, grasp, lift, then out of the shelf,
        each by IK near the previous configuration; the IK states must be free of the immovable
        obstacles (getStateNearPose checks them against m_cc_i, the OoI removed)."""
        arm = self.arm
        obs = ObstacleSet.static({self.ctx.scene.ooi_id})
        arm.detach()
        arm.set_object_states(self.start.objects)
        arm.set_gripper(closed=False)
        q_pre = arm.ik(self.p_pre, self.R_pre, q)
        if q_pre is None or not arm.collision_free(q_pre, obs):
            return None
        reach = cartesian_path(arm, q_pre, self.p_grasp, self.R_grasp, None)
        if reach is None or not arm.collision_free(reach[-1], obs):
            return None
        lift = cartesian_path(arm, reach[-1], self.p_grasp + np.array([0, 0, self.lift_h]), self.R_grasp, None)
        if lift is None or not arm.collision_free(lift[-1], obs):
            return None
        extract = cartesian_path(arm, lift[-1], self.p_out, self.R_grasp, None)
        if extract is None or not arm.collision_free(extract[-1], obs):
            return None
        return RetrievalPlan([Segment("pregrasp", [q, q_pre], False, False), Segment("reach", reach, False, False),
                              Segment("lift", lift, True, True), Segment("extract", extract, True, True)])

    def is_satisfied(self, s: KState) -> bool:
        """[cpp] PushplanPoseGoal::isSatisfied: near the pre-grasp pose and CHECKGRASPTRAJ succeeds
        in simulation with the state's object poses."""
        if not self.near_goal(s.q):
            return False
        self.count.goal_checks += 1
        plan = self.grasp_plan(s.q)
        if plan is None:
            return False
        from mamo.execute import simulate_retrieval
        t = time.perf_counter()
        self.set_sim_state(s)
        res = simulate_retrieval(self.world, self.arm, self.sim, self.ctx.scene, plan, self.cfg)
        self.ctx.timers.sim += time.perf_counter() - t
        self.count.sims += 1
        if not res.success:
            return False
        self.retrieval = plan
        return True


def finish(problem: PhysicsProblem | None, ctx: MAMOContext, sol: Solution, t_wall: float, **extra) -> Solution:
    """Statistics of a sampling-based run: all of its wall time is planning time."""
    wall = time.perf_counter() - t_wall
    tm = ctx.timers
    sol.stats = {"planning_time": wall, "wall_time": wall, "sim_time": tm.sim,
                 "robot_planner_time": wall - tm.sim, "push_planner_time": 0.0, "mapf_time": 0.0,
                 "num_pushes": len(sol.pushes), **tm.counts, **extra}
    if problem is not None:
        c = problem.count
        sol.stats.update(steers=c.steers, sims=c.sims, **c.extra)
        sol.stats.setdefault("goal_checks", c.goal_checks)
    return sol


def simulate_motion_action(sim, arm, action: MotionAction, cfg) -> SimResult:
    """Replay one MotionAction from the world's current state (mamo.execute)."""
    w = sim.w
    monitor = ConstraintMonitor(w, cfg, immovable_ooi=True)
    timing = time_parameterise(action.path, cfg.robot.max_joint_speed)
    return sim.run([timing], [False], None, monitor, settle=action.simulated)
