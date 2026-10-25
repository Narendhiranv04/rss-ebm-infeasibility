"""Prehensile (pick-and-place) rearrangement for I-M4M.

I-M4M paper Sec. IV-A, robot.cpp PlanPickPlace, sim.py SimPickPlace:

* an object is graspable if its footprint fits inside the open gripper along some axis
* the placement pose is the final state of the object's MAPF path (the rest is discarded)
* known grasp poses: for boxes, approaches along each face normal whose width fits; for
  cylinders, 8 directions (Object::initPregrasps)
* the trajectory is: approach (motion planner) -> pre-grasp, grasp (hard-coded straight
  move), lift 2 cm, transport (motion planner) -> above the placement, lower 1 cm,
  release, retract 10 cm. Every part must be collision-free with all objects
  (movable and immovable); only the gripper fingers may touch the object being grasped.
* only the grasping and placement manoeuvres are simulated, since the rest is
  collision-free by construction; the robot is moved directly to the start of each
  simulated part. execute.py replays the whole trajectory.

xArm7 changes, as in retrieval.py: grasps may be pitched down up to 30 deg and are held
at least 3 cm above the shelf (the wrist hangs below the tool axis), and the fingertips
reach 3.5 cm past the near face of wide objects instead of the object centre.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mamo.config import Config
from mamo.execute import set_grasped_mass
from mamo.geometry import Circle, footprint
from mamo.robot.arm import Arm, ObstacleSet, tool_rotation
from mamo.robot.motion import cartesian_path, plan
from mamo.scene import CYLINDER, SceneObject, Shelf
from mamo.simulate import ConstraintMonitor, SimResult, Simulator, cartesian_timing, time_parameterise


@dataclass
class PickPlaceFailure:
    reason: str


@dataclass
class PickPlaceAction:
    obj_id: int
    cells: list[tuple[int, int]]            # MAPF path (only its last cell is used)
    waypoints: np.ndarray                   # object start and placement (x, y), for plots
    tool_points: np.ndarray                 # pre-grasp and placement tool positions, for plots
    approach: list[np.ndarray]              # start -> pre-grasp
    reach: list[np.ndarray]                 # pre-grasp -> grasp, gripper open
    lift: list[np.ndarray]                  # grasp -> +2 cm, object held
    transport: list[np.ndarray]             # -> above the placement, object held
    lower: list[np.ndarray]                 # -> 1 cm lower, object held
    retract: list[np.ndarray]               # release -> 10 cm back, gripper open
    info: dict = field(default_factory=dict)
    kind = "pick_place"

    @property
    def final_q(self) -> np.ndarray:
        return self.retract[-1]


def graspable(o: SceneObject, cfg: Config) -> bool:
    """Object fits inside the open gripper along some horizontal axis (Object::m_graspable)."""
    lim = cfg.pick_place.max_half_width
    if o.shape == CYLINDER:
        return o.radius <= lim
    return min(o.x_size, o.y_size) <= lim


def grasp_directions(o: SceneObject, cfg: Config) -> list[float]:
    """World yaws of the tool (pointing at the object) for the object's known grasps."""
    lim = cfg.pick_place.max_half_width
    if o.shape == CYLINDER:
        yaws = [o.yaw + k * math.pi / 4 for k in range(8)] if o.radius <= lim else []
    else:
        yaws = []
        if o.y_size <= lim:  # fingers close along the object's y axis: approach along +-x
            yaws += [o.yaw, o.yaw + math.pi]
        if o.x_size <= lim:
            yaws += [o.yaw + math.pi / 2, o.yaw - math.pi / 2]
    # [xarm] only approaches from the robot's side of the shelf are reachable (config).
    out = [math.atan2(math.sin(y), math.cos(y)) for y in yaws]
    return [y for y in out if abs(y) <= math.radians(cfg.pick_place.max_approach_yaw_deg) + 1e-6]


def _extent_along(o: SceneObject, direction: np.ndarray) -> float:
    s = footprint(o)
    if isinstance(s, Circle):
        return s.r
    return 0.5 * float(np.ptp(s.corners() @ direction))


def _rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class PickPlacePlanner:
    def __init__(self, arm: Arm, cfg: Config, shelf: Shelf) -> None:
        self.arm = arm
        self.cfg = cfg
        self.shelf = shelf

    def plan(self, obj: SceneObject, goal_xy: np.ndarray, cells: list[tuple[int, int]],
             object_states: dict[int, np.ndarray], q_start: np.ndarray, rng: np.random.Generator,
             stats: dict | None = None, goal_z: float | None = None,
             turns: list[float] | None = None,
             via: tuple[float, float] | None = None) -> PickPlaceAction | PickPlaceFailure:
        """Plan picking ``obj`` and placing its centre at ``goal_xy``.

        ``goal_z``: the object's centre height when placed (default: where it is now, i.e. on
        the same surface). ``turns``: rotations of the object about z (rad) to try at the
        placement (default: ``place_yaws`` evenly spaced ones).
        ``via`` = (x_out_pick, x_out_place) [ext]: for moves between compartments (decks,
        shelves), the held object slides straight out to tool x <= x_out_pick, the motion
        planner only connects the two compartment fronts, the object slides straight in to the
        placement and the hand retracts out to x_out_place (default None: I-M4M's transport).
        """
        pc, rc = self.cfg.pick_place, self.cfg.retrieval
        arm = self.arm
        arm.detach()
        arm.set_object_states(object_states)
        if not graspable(obj, self.cfg):
            return _fail(stats, "not graspable")
        everything = ObstacleSet.everything()
        # Only the fingers may touch the object being grasped (reach, release, retract).
        fingers_ok = ObstacleSet(everything.categories, frozenset(), frozenset({obj.id}))
        c = np.array([obj.x, obj.y])
        z = obj.z + 0.5 * obj.half_height  # [cpp] grasp at a quarter of the height above the centre
        on_shelf = abs(obj.bottom_z - self.shelf.floor_top) < 0.02 and self.shelf.contains_xy(obj.x, obj.y)
        floor = self.shelf.floor_top if on_shelf else obj.bottom_z  # the surface the object stands on
        z = min(max(z, floor + rc.grasp_height_range[0]), obj.top_z - rc.grasp_top_margin)
        dirs = grasp_directions(obj, self.cfg)
        if not dirs:
            return _fail(stats, "no grasp direction")

        # 1. approach: plan to any reachable pre-grasp (multi-goal, as planToPoseGoal)
        arm.set_gripper(closed=False)
        grasps = []
        for yaw in dirs:
            a = np.array([math.cos(yaw), math.sin(yaw)])
            depth = _extent_along(obj, a)
            p_grasp = np.append(c - max(0.0, depth - pc.finger_depth) * a, z)
            # p_grasp is min(depth, finger_depth) inside the near face; the pre-grasp 7.5 cm outside it.
            back = min(depth, pc.finger_depth) + pc.pregrasp_offset
            for pitch in pc.pitches_deg:
                R = tool_rotation(a, math.radians(pitch))
                p_pre = p_grasp - back * R[:, 2]
                q_pre = arm.ik_multi(p_pre, R, [q_start], rng, restarts=4, obstacles=everything)
                if q_pre is not None and (not pc.reach_first
                                          or cartesian_path(arm, q_pre, p_grasp, R, fingers_ok) is not None):
                    grasps.append((q_pre, p_grasp, R))
        if not grasps:
            return _fail(stats, "pregrasps unreachable")
        approach = plan(arm, q_start, [g[0] for g in grasps], everything, rng,
                        time_limit=pc.plan_time, max_iters=pc.max_rrt_iters)
        if approach is None:
            return _fail(stats, "approach")
        q_pre = approach[-1]
        _, p_grasp, R = min(grasps, key=lambda g: float(np.max(np.abs(g[0] - q_pre))))

        # 2. grasp: straight move to the grasp pose with the gripper open
        reach = cartesian_path(arm, q_pre, p_grasp, R, fingers_ok)
        if reach is None:
            return _fail(stats, "grasp IK")
        q_grasp = reach[-1]

        # 3. close and lift 2 cm with the object attached
        arm.set_gripper(closed=True)
        arm.attach(obj.id, q_grasp)
        try:
            lift = cartesian_path(arm, q_grasp, p_grasp + np.array([0.0, 0.0, pc.lift]), R, everything)
            if lift is None:
                return _fail(stats, "postgrasp IK")
            _, rel_p, _ = arm.attached

            # 4. transport to above the placement pose (yaw free: try several tool yaws)
            z_place = obj.z if goal_z is None else goal_z
            z_obj = z_place + pc.lift
            goals, targets = [], []
            if turns is None:
                turns = [2 * math.pi * k / pc.place_yaws for k in range(pc.place_yaws)]
            for turn in turns:
                Rp = _rot_z(turn) @ R
                p_place = np.array([goal_xy[0], goal_xy[1], z_obj]) - Rp @ rel_p
                q = arm.ik_multi(p_place, Rp, [lift[-1]], rng, restarts=3, obstacles=everything)
                if q is not None:
                    goals.append(q)
                    targets.append((p_place, Rp))
            if not goals:
                return _fail(stats, "placement unreachable")
            k_place = None
            if via is None:
                transport, k_place = plan_upright(arm, lift[-1], R, targets, everything, rng,
                                                  time_limit=pc.plan_time, max_iters=pc.max_rrt_iters)
            else:
                transport, k_place = self._transport_via(lift[-1], R, goals, targets, via, everything, rng)
            if transport is None:
                return _fail(stats, "transport")
            q_pp = transport[-1]
            if k_place is None:
                k_place = int(np.argmin([float(np.max(np.abs(g - q_pp))) for g in goals]))
            p_place, Rp = targets[k_place]

            # 5. lower 1 cm
            lower = cartesian_path(arm, q_pp, p_place - np.array([0.0, 0.0, pc.lower]), Rp, everything)
            if lower is None:
                return _fail(stats, "place IK")
        finally:
            arm.detach()

        # 6. release and retract 10 cm back along the tool direction
        arm.set_object_states({**object_states, obj.id: _placed_state(object_states[obj.id], goal_xy,
                                                                      z_place + pc.lift - pc.lower)})
        arm.set_gripper(closed=False)
        back = Rp[:, 2].copy()
        back[2] = 0.0
        back /= max(np.linalg.norm(back), 1e-9)
        p_ret = p_place - np.array([0.0, 0.0, pc.lower]) - pc.retract * back
        if via is not None and back[0] > 0.5:  # slide the hand out of the compartment
            p_ret = p_ret - max(0.0, (p_ret[0] - via[1]) / back[0]) * back
        retract = cartesian_path(arm, lower[-1], p_ret, Rp, fingers_ok)
        arm.set_gripper(closed=True)
        if retract is None:
            return _fail(stats, "retract IK")
        _fail(stats, "planned")
        start = np.array([obj.x, obj.y])
        return PickPlaceAction(obj.id, list(cells), np.array([start, np.asarray(goal_xy)]),
                               np.array([q_to_xyz(arm, q_pre), p_place]), approach, reach, lift, transport,
                               lower, retract, {"tool": (0.0, False, 0.0), "pick_place": True})


    def _transport_via(self, q_lift: np.ndarray, R: np.ndarray, goals: list, targets: list,
                       via: tuple[float, float], obstacles: ObstacleSet,
                       rng: np.random.Generator) -> tuple[list[np.ndarray] | None, int | None]:
        """Straight out, planned between the fronts, straight in (see ``plan``'s ``via``).
        Returns the path and the index of the placement it ends at."""
        arm, pc = self.arm, self.cfg.pick_place
        p0 = arm.fk(q_lift)[0]
        out = cartesian_path(arm, q_lift, np.array([min(p0[0], via[0]), p0[1], p0[2]]), R, obstacles)
        if out is None:
            return None, None
        ins_starts, ins_paths, ins_k = [], [], []
        for k, (q_goal, (p_place, Rp)) in enumerate(zip(goals, targets)):
            p_front = np.array([min(p_place[0], via[1]), p_place[1], p_place[2]])
            q_front = arm.ik_multi(p_front, Rp, [q_goal], rng, restarts=3, obstacles=obstacles)
            if q_front is None:
                continue
            ins = cartesian_path(arm, q_front, p_place, Rp, obstacles)
            if ins is not None:
                ins_starts.append(q_front)
                ins_paths.append(ins)
                ins_k.append(k)
        if not ins_starts:
            return None, None
        fronts = [(arm.fk(q)[0], targets[k][1]) for q, k in zip(ins_starts, ins_k)]
        mid, j = plan_upright(arm, out[-1], R, fronts, obstacles, rng, time_limit=pc.plan_time,
                              max_iters=pc.max_rrt_iters)
        if mid is None:
            return None, None
        # the IK branch at the front may differ from the one the slide in starts from
        ins = cartesian_path(arm, mid[-1], targets[ins_k[j]][0], targets[ins_k[j]][1], obstacles)
        if ins is None:
            return None, None
        return list(out) + list(mid[1:]) + list(ins[1:]), ins_k[j]


def plan_upright(arm: Arm, q_start: np.ndarray, R_grasp: np.ndarray, targets: list[tuple[np.ndarray, np.ndarray]],
                 obstacles: ObstacleSet, rng: np.random.Generator, time_limit: float = 1.0,
                 max_iters: int = 1500, step: float = 0.05,
                 goal_bias: float = 0.2) -> tuple[list[np.ndarray] | None, int | None]:
    """Transport the held object with the tool's roll and pitch fixed and its yaw free.

    [cpp] robot.cpp PlanPickPlace plans the place motion with addPathConstraint: end-effector
    roll and pitch within 5 deg (DEG5), yaw within 2 pi. Here an RRT grows in (x, y, z, yaw) of
    the tool, every orientation being R_z(yaw) R_grasp, and every edge is a straight tool motion
    solved by IK (``cartesian_path``: 1 cm steps, slerp between orientations that differ by a
    rotation about the vertical, collision-checked with the object attached), so the carried
    object never tilts. ``targets``: (tool position, orientation) placements; returns the joint
    path and the index of the placement reached.
    """
    import time as _time
    p0 = arm.fk(q_start)[0]
    yaws = [math.atan2((Rp @ R_grasp.T)[1, 0], (Rp @ R_grasp.T)[0, 0]) for _, Rp in targets]
    pts = np.vstack([p0] + [p for p, _ in targets])
    lo = pts.min(axis=0) - np.array([0.15, 0.15, 0.0])
    hi = pts.max(axis=0) + np.array([0.15, 0.15, 0.10])
    nodes = [(p0, 0.0, q_start, -1, [q_start])]   # (position, yaw, joints, parent, joint path from parent)

    def steer(i: int, p: np.ndarray, yaw: float, max_len: float | None):
        pn, yn, qn, _, _ = nodes[i]
        d = p - pn
        dy = math.atan2(math.sin(yaw - yn), math.cos(yaw - yn))
        n = float(np.linalg.norm(d)) + 0.1 * abs(dy)  # 10 cm per rad of yaw
        if max_len is not None and n > max_len:
            d, dy = d * (max_len / n), dy * (max_len / n)
        p1, y1 = pn + d, yn + dy
        path = cartesian_path(arm, qn, p1, _rot_z(y1) @ R_grasp, obstacles)
        return None if path is None else (p1, y1, path)

    t0 = _time.perf_counter()
    for _ in range(max_iters):
        if _time.perf_counter() - t0 > time_limit:
            break
        if rng.random() < goal_bias:
            k = int(rng.integers(len(targets)))
            p_s, y_s = targets[k][0], yaws[k]
        else:
            p_s, y_s = rng.uniform(lo, hi), float(rng.uniform(-math.pi, math.pi))
        dist = [float(np.linalg.norm(n[0] - p_s)) + 0.1 * abs(math.atan2(math.sin(y_s - n[1]), math.cos(y_s - n[1])))
                for n in nodes]
        i = int(np.argmin(dist))
        e = steer(i, p_s, y_s, step)
        if e is None:
            continue
        nodes.append((e[0], e[1], e[2][-1], i, e[2]))
        # try to finish: a straight tool motion to the nearest placement
        k = int(np.argmin([float(np.linalg.norm(p - e[0])) for p, _ in targets]))
        if float(np.linalg.norm(targets[k][0] - e[0])) <= 2 * step:
            last = steer(len(nodes) - 1, targets[k][0], yaws[k], None)
            if last is not None:
                out = list(last[2])
                j = len(nodes) - 1
                while j > 0:
                    out = list(nodes[j][4][:-1]) + out
                    j = nodes[j][3]
                return out, k
    return None, None


def _placed_state(s: np.ndarray, goal_xy: np.ndarray, z: float) -> np.ndarray:
    """Planning-time guess of the placed object's pose (only used to check the retract)."""
    out = s.copy()
    out[0], out[1], out[2] = goal_xy[0], goal_xy[1], z
    return out


def q_to_xyz(arm: Arm, q: np.ndarray) -> np.ndarray:
    return arm.fk(q)[0].copy()


def _fail(stats: dict | None, key: str) -> PickPlaceFailure:
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1
    return PickPlaceFailure(key)


# ------------------------------------------------------------------- simulation
def _weld_id(world) -> int:
    return mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")


def _relpose(world, eq: int) -> tuple[np.ndarray, np.ndarray]:
    m, d = world.model, world.data
    b1, b2 = m.eq_obj1id[eq], m.eq_obj2id[eq]
    R1 = d.xmat[b1].reshape(3, 3)
    q1inv = np.zeros(4)
    mujoco.mju_negQuat(q1inv, d.xquat[b1])
    rel_q = np.zeros(4)
    mujoco.mju_mulQuat(rel_q, q1inv, d.xquat[b2])
    return R1.T @ (d.xpos[b2] - d.xpos[b1]), rel_q


def _set_weld(world, eq: int, rel: tuple[np.ndarray, np.ndarray], active: bool) -> None:
    m, d = world.model, world.data
    m.eq_data[eq, 0:3] = 0.0
    m.eq_data[eq, 3:6] = rel[0]
    m.eq_data[eq, 6:10] = rel[1]
    m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = int(active)


def _teleport(world, q: np.ndarray, held: int | None, eq: int, rel) -> None:
    """Move the arm (and the held object, rigidly) to q at rest."""
    m, d = world.model, world.data
    world.set_arm_q(q)
    d.ctrl[world.arm_act] = q
    d.qvel[:] = 0.0
    mujoco.mj_kinematics(m, d)
    if held is not None:
        b1 = m.eq_obj1id[eq]
        R1 = d.xmat[b1].reshape(3, 3)
        a = world.obj_qadr[held]
        d.qpos[a:a + 3] = d.xpos[b1] + R1 @ rel[0]
        qq = np.zeros(4)
        mujoco.mju_mulQuat(qq, d.xquat[b1], rel[1])
        d.qpos[a + 3:a + 7] = qq
    mujoco.mj_forward(m, d)


def simulate_pick_place(sim: Simulator, arm: Arm, action: PickPlaceAction, cfg: Config,
                        full: bool = False) -> SimResult:
    """SIMULATE the pick-and-place from the world's current state.

    full=False (planning): simulate the grasp (reach, close, lift) and the placement (lower,
    release, retract, 1 s hold); approach and transport are skipped (collision-free by
    construction). full=True (execution): simulate everything.
    """
    w = sim.w
    m, d = w.model, w.data
    pc, rc = cfg.pick_place, cfg.robot
    start_states = w.object_states()
    monitor = ConstraintMonitor(w, cfg, immovable_ooi=True)
    eq = _weld_id(w)
    ooi_body = m.eq_obj2id[eq]
    m.eq_obj2id[eq] = w.obj_body[action.obj_id]
    oid = action.obj_id
    held_mass = None

    def run(path, timing, closed) -> SimResult:
        segs = [cartesian_timing(arm, path, rc.push_speed, rc.max_joint_speed) if timing == "cart"
                else time_parameterise(path, rc.max_joint_speed)]
        segs[0].append((segs[0][-1][0] + 0.3, segs[0][-1][1]))  # let the servos settle
        return sim.run(segs, [closed], None, monitor, settle=False)

    def hold(q, closed, seconds) -> SimResult:
        return sim.run([[(0.0, q), (seconds, q)]], [closed], None, monitor, settle=False)

    try:
        if full:
            res = run(action.approach, "joint", False)
        else:
            _teleport(w, action.reach[0], None, eq, None)
            res = hold(action.reach[0], False, 0.3)
        if res.valid:
            res = run(action.reach, "cart", False)
        if res.valid:
            # Weld where the planner attached the object, then close the fingers (see execute.py).
            rel = _relpose(w, eq)
            _set_weld(w, eq, rel, True)
            held_mass = set_grasped_mass(w, w.obj_body[oid], cfg.physics.grasped_mass)
            res = hold(action.reach[-1], True, 0.6)
        if res.valid:
            res = run(action.lift, "cart", True)
        if res.valid:
            if full:
                res = run(action.transport, "joint", True)
            else:
                _teleport(w, action.transport[-1], oid, eq, rel)
                res = hold(action.transport[-1], True, 0.2)
        if res.valid:
            res = run(action.lower, "cart", True)
        if res.valid:
            d.eq_active[eq] = 0
            m.body_mass[w.obj_body[oid]], m.body_inertia[w.obj_body[oid]] = held_mass
            held_mass = None
            res = hold(action.lower[-1], False, pc.release_time)
        if res.valid:
            res = run(action.retract, "cart", False)
        if res.valid:
            res = hold(action.retract[-1], False, pc.settle_time)
    finally:
        d.eq_active[eq] = 0
        m.eq_obj2id[eq] = ooi_body
        if held_mass is not None:
            m.body_mass[w.obj_body[oid]], m.body_inertia[w.obj_body[oid]] = held_mass
    if not res.valid:
        return res
    end = w.object_states()
    moved = [i for i in end if np.linalg.norm(end[i][:3] - start_states[i][:3]) > cfg.push.min_displacement]
    disp = float(np.linalg.norm(end[oid][:2] - start_states[oid][:2]))
    # [paper] / [cpp] SimPickPlace: valid if no interaction constraint is violated (the 'some
    # object moved' rule is E-M4M's, for pushes).
    return SimResult(True, "", end, moved, disp, res.sim_time, w.arm_q())
