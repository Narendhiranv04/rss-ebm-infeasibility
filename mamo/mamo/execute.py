"""Execute a full MAMO solution in simulation, from the initial scene.

Neither paper simulates the final retrieval during planning (it is collision-
free by construction). This replays every push in order and then performs the
retrieval with a real grasp, as an end-to-end check of the returned plan.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mamo.config import Config
from mamo.solution import Solution
from mamo.robot.arm import Arm
from mamo.scene import BOX, Scene
from mamo.simulate import ConstraintMonitor, Simulator, cartesian_timing, simulate_push, time_parameterise
from mamo.world import World, quat_to_rpy, relocate_goals, tilt_deg


@dataclass
class ExecutionResult:
    success: bool
    reason: str
    ooi_final: np.ndarray | None = None
    states: list[dict[int, np.ndarray]] = field(default_factory=list)  # object states before/after each push


def _weld_id(world: World) -> int:
    return mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")


def _relpose(world: World) -> tuple[np.ndarray, np.ndarray]:
    """Pose of the OoI (weld body2) in the gripper base frame (body1)."""
    m, d = world.model, world.data
    eq = _weld_id(world)
    b1, b2 = m.eq_obj1id[eq], m.eq_obj2id[eq]
    R1 = d.xmat[b1].reshape(3, 3)
    q1inv = np.zeros(4)
    mujoco.mju_negQuat(q1inv, d.xquat[b1])
    rel_q = np.zeros(4)
    mujoco.mju_mulQuat(rel_q, q1inv, d.xquat[b2])
    return R1.T @ (d.xpos[b2] - d.xpos[b1]), rel_q


def set_grasped_mass(world: World, body: int, mass: float | None) -> tuple[float, np.ndarray]:
    """Set a body's mass (inertia scaled to match) and return the old (mass, inertia).

    The reference simulator gives a grasped object a fixed 50 g mass while it is held
    (sim.py grasped_mass), so very light or heavy objects do not fight the grasp constraint.
    """
    m = world.model
    old = (float(m.body_mass[body]), m.body_inertia[body].copy())
    if mass is not None and old[0] > 0:
        m.body_inertia[body] = old[1] * (mass / old[0])
        m.body_mass[body] = mass
    return old


def _activate_grasp(world: World, rel: tuple[np.ndarray, np.ndarray]) -> None:
    m, d = world.model, world.data
    eq = _weld_id(world)
    m.eq_data[eq, 0:3] = 0.0
    m.eq_data[eq, 3:6] = rel[0]
    m.eq_data[eq, 6:10] = rel[1]
    m.eq_data[eq, 10] = 1.0
    d.eq_active[eq] = 1


def goal_errors(world: World, scene: Scene, cfg: Config) -> list[str]:
    """Relocate targets that are not at their goal: centre within ``tol[0]`` in x/y and 2 cm in
    z, upright, and (if ``tol[1]`` >= 0) the yaw within ``tol[1]`` degrees."""
    out = []
    for oid, g in relocate_goals(scene).items():
        pos, quat = world.object_pose(oid)
        gx, gy, gz, gyaw = g["goal"]
        tol_xy, tol_yaw = (list(g.get("tol", [0.02, -1])) + [-1])[:2]
        err = float(np.hypot(pos[0] - gx, pos[1] - gy))
        if err > tol_xy or abs(pos[2] - gz) > 0.02:
            out.append(f"object {oid} {100 * err:.1f} cm from its goal (dz {100 * (pos[2] - gz):.1f} cm)")
        elif tilt_deg(quat) > cfg.constraints.max_tilt_deg:
            out.append(f"object {oid} not upright at its goal")
        elif tol_yaw >= 0:
            o = scene.by_id(oid)
            period = math.pi if o.shape == BOX else 2 * math.pi  # a box looks the same turned by 180 deg
            d = (quat_to_rpy(quat)[2] - gyaw) % period
            if math.degrees(min(d, period - d)) > tol_yaw:
                out.append(f"object {oid} at its goal but turned {math.degrees(min(d, period - d)):.0f} deg")
    return out


def execute(scene: Scene, sol: Solution, cfg: Config | None = None, recorder=None,
            on_stage=None) -> ExecutionResult:
    """Replay ``sol``; ``recorder(world)`` is called every 20 steps, ``on_stage(label)`` at each stage."""
    world = World(scene, cfg or Config())
    cfg = world.cfg  # with the scene's robot base
    arm = Arm(world)
    sim = Simulator(world, cfg)
    sim.recorder = recorder
    stage = on_stage or (lambda label: None)
    states = [world.object_states()]
    for k, action in enumerate(sol.pushes):
        if getattr(action, "kind", "push") == "pick_place":  # I-M4M prehensile action
            from mamo.im4m.pickplace import simulate_pick_place
            stage(f"pick-and-place {k + 1}/{len(sol.pushes)}: object {action.obj_id}")
            res = simulate_pick_place(sim, arm, action, cfg, full=True)
        elif getattr(action, "kind", "push") == "motion":  # RRT / KPIECE / SELSIM edge
            from mamo.kinodynamic import simulate_motion_action
            stage(f"motion {k + 1}/{len(sol.pushes)}")
            res = simulate_motion_action(sim, arm, action, cfg)
        else:
            stage(f"push {k + 1}/{len(sol.pushes)}: object {action.obj_id}")
            res = simulate_push(sim, arm, action, cfg)
        states.append(world.object_states())
        if not res.valid:
            return ExecutionResult(False, f"push {k + 1}/{len(sol.pushes)} invalid on replay: {res.reason}",
                                   states=states)
    if scene.task["type"] == "relocate":
        if sol.retrieval is not None:  # optional: e.g. a final retreat of the arm
            stage("final arm motion")
            for seg in sol.retrieval.segments:
                timing = time_parameterise(seg.path, cfg.robot.max_joint_speed)
                res = sim.run([timing], [seg.gripper_closed], None, ConstraintMonitor(world, cfg), settle=False)
                if not res.valid:
                    return ExecutionResult(False, f"final segment '{seg.name}': {res.reason}", states=states)
        sim.run([[(0.0, world.arm_q()), (1.0, world.arm_q())]], [False], None, None, settle=False)
        missed = goal_errors(world, scene, cfg)
        pos = world.object_pose(scene.ooi_id)[0]
        return ExecutionResult(not missed, "; ".join(missed), pos, states)
    if sol.retrieval is None:
        return ExecutionResult(False, "no retrieval trajectory", states=states)
    return simulate_retrieval(world, arm, sim, scene, sol.retrieval, cfg, stage, states)


def simulate_retrieval(world: World, arm: Arm, sim: Simulator, scene: Scene, retrieval, cfg: Config,
                       stage=None, states: list | None = None, monitored: set[int] | None = None) -> ExecutionResult:
    """Grasp the OoI and carry it out of the shelf along ``retrieval`` from the world's current state.
    ``monitored``: the movable objects whose constraints are checked (default: all)."""
    stage = stage or (lambda label: None)
    states = states if states is not None else []
    rc = cfg.robot
    monitor = ConstraintMonitor(world, cfg, immovable_ooi=False)
    if monitored is not None:
        monitor.movables = [o for o in monitor.movables if o.id in monitored]
    ooi = scene.ooi_id
    weld = _weld_id(world)
    old_mass = None
    try:
        for seg in retrieval.segments:
            stage(f"retrieval: {seg.name}")
            # (The gripper's own finger linkages are equality constraints too, so check the weld itself.)
            if seg.carrying and not world.data.eq_active[weld]:
                # Hold the OoI rigidly where it is -- the grasp the planner assumed (arm.attach) --
                # and only then close the fingers around it: closing first lets the finger
                # linkage shove a light OoI centimetres across the shelf.
                _activate_grasp(world, _relpose(world))
                old_mass = set_grasped_mass(world, world.obj_body[ooi], cfg.physics.grasped_mass)
                hold = [(0.0, seg.path[0]), (0.6, seg.path[0])]
                sim.run([hold], [True], None, None, settle=False)
            timing = cartesian_timing(arm, seg.path, rc.push_speed, rc.max_joint_speed) if seg.name in (
                "reach", "lift", "extract") else time_parameterise(seg.path, rc.max_joint_speed)
            # Hold the last waypoint briefly so the position servos settle before the next segment.
            timing.append((timing[-1][0] + 0.4, timing[-1][1]))
            res = sim.run([timing], [seg.gripper_closed], None, monitor, settle=False)
            if not res.valid:
                return ExecutionResult(False, f"retrieval segment '{seg.name}': {res.reason}",
                                       world.object_pose(ooi)[0], states)
        pos, _ = world.object_pose(ooi)
        if pos[0] > scene.shelf.x_min:
            return ExecutionResult(False, "OoI still inside the shelf", pos, states)
        return ExecutionResult(True, "", pos, states)
    finally:
        if old_mass is not None:  # a planner reuses its world after checking a grasp
            m = world.model
            m.body_mass[world.obj_body[ooi]], m.body_inertia[world.obj_body[ooi]] = old_mass
