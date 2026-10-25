"""Forward-simulate robot trajectories and check interaction constraints.

M4M Alg. 1 SIMULATEPUSH / E-M4M Alg. 1 ISVALID. Constraints (paper Sec. III):
  * neither the robot nor any object may touch an immovable obstacle
    (the OoI counts as immovable while the scene is being rearranged)
  * movable objects may not fall off the shelf, tilt beyond 25 deg,
    or move faster than 1 m/s
  * the robot may not hit the shelf (reference: checkTableCollision)
A push is valid if no constraint is violated and at least one object moved
(E-M4M paper; here > 1 cm, the MAPF grid; the reference requires the pushed
object itself to move > 1 cm). After the motion the scene settles for
2 s before the resulting object poses are read.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from mamo.config import Config
from mamo.world import (GROUND, IMMOVABLE_GEOM, MOVABLE_GEOM, OOI_GEOM, ROBOT, SHELF_GEOM, World,
                        tilt_deg)


@dataclass
class SimResult:
    valid: bool
    reason: str = ""
    object_states: dict[int, np.ndarray] = field(default_factory=dict)
    moved: list[int] = field(default_factory=list)       # objects displaced > 1 cm ("relevant objects")
    target_displacement: float = 0.0
    sim_time: float = 0.0                                # simulated seconds
    arm_q: np.ndarray | None = None


class ConstraintMonitor:
    def __init__(self, world: World, cfg: Config, immovable_ooi: bool = True) -> None:
        self.w = world
        self.cfg = cfg.constraints
        cat = world.geom_cat
        self.is_robot = cat == ROBOT
        self.is_static = (cat == SHELF_GEOM) | (cat == GROUND)
        self.is_shelf = cat == SHELF_GEOM
        immov = cat == IMMOVABLE_GEOM
        if immovable_ooi:
            immov = immov | (cat == OOI_GEOM)
        self.is_immovable = immov
        self.movables = [o for o in world.scene.movables]
        self.shelf = world.scene.shelf
        # Retrieve problems keep every object on the OoI's surface; relocate problems may move
        # objects between surfaces, so an object only has to stay above some surface.
        self.single_surface = world.scene.task["type"] == "retrieve"
        self.initial_pairs = self._contact_pairs()

    def _contact_pairs(self) -> set[tuple[int, int]]:
        d = self.w.data
        n = d.ncon
        g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
        return {(min(a, b), max(a, b)) for a, b in zip(g1, g2)}

    def check(self) -> str:
        """Return a violation description, or '' if all constraints hold."""
        c = self.cfg
        w, d = self.w, self.w.data
        n = d.ncon
        if n:
            g1, g2 = d.contact.geom1[:n], d.contact.geom2[:n]
            dist = d.contact.dist[:n]
            pen = dist < -c.contact_depth_tol
            if c.immovable_contact_forbidden:
                bad = pen & ((self.is_immovable[g1] & ~self.is_static[g2]) |
                             (self.is_immovable[g2] & ~self.is_static[g1]))
                # [cpp] sim.py checkInteractions: any contact counts, including one present when the
                # simulation starts (no scene starts with one; a planned state that has one is invalid).
                for k in np.flatnonzero(bad):
                    return f"immovable contact: {self._name(g1[k])} / {self._name(g2[k])}"
            if c.robot_shelf_contact_forbidden:
                bad = pen & ((self.is_robot[g1] & self.is_static[g2]) | (self.is_robot[g2] & self.is_static[g1]))
                if np.any(bad):
                    k = int(np.flatnonzero(bad)[0])
                    return f"robot hit shelf: {self._name(g1[k])} / {self._name(g2[k])}"
        for o in self.movables:
            a = w.obj_qadr[o.id]
            pos, quat = d.qpos[a:a + 3], d.qpos[a + 3:a + 7]
            if self._held(o.id):
                continue  # [cpp] sim.py: the picked object is exempt while it is carried (grasp_at)
            if tilt_deg(quat) > c.max_tilt_deg:
                return f"object {o.id} tilted {tilt_deg(quat):.0f} deg"
            v = d.qvel[w.obj_dadr[o.id]:w.obj_dadr[o.id] + 3]
            if float(np.linalg.norm(v)) > c.max_speed:
                return f"object {o.id} moved at {np.linalg.norm(v):.2f} m/s"
            if self.single_surface:
                if pos[2] < self.shelf.floor_top - 0.02 or not self.shelf.contains_xy(pos[0], pos[1]):
                    return f"object {o.id} fell off the shelf"
            elif self.w.scene.surface_of(pos[0], pos[1], pos[2] + 0.02) is None:
                return f"object {o.id} fell off its surface"
        return ""

    def _held(self, oid: int) -> bool:
        """Is ``oid`` welded to the gripper (carried between surfaces)?"""
        m, d = self.w.model, self.w.data
        eq = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_EQUALITY, "grasp_weld")
        return bool(d.eq_active[eq]) and m.eq_obj2id[eq] == self.w.obj_body[oid]

    def _name(self, g: int) -> str:
        m = self.w.model
        return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or \
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])


def time_parameterise(path: list[np.ndarray], max_speed: float) -> list[tuple[float, np.ndarray]]:
    """Waypoints with cumulative times so no joint exceeds ``max_speed``."""
    out = [(0.0, path[0])]
    t = 0.0
    for a, b in zip(path[:-1], path[1:]):
        t += max(float(np.max(np.abs(b - a))) / max_speed, 1e-3)
        out.append((t, b))
    return out


def cartesian_timing(arm, path: list[np.ndarray], speed: float, max_joint_speed: float) -> list[tuple[float, np.ndarray]]:
    """Timing for a tool-space path: TCP speed bounded by ``speed``, joints by ``max_joint_speed``."""
    out = [(0.0, path[0])]
    t = 0.0
    p_prev = arm.fk(path[0])[0]
    for a, b in zip(path[:-1], path[1:]):
        p = arm.fk(b)[0]
        dt = max(float(np.linalg.norm(p - p_prev)) / speed, float(np.max(np.abs(b - a))) / max_joint_speed, 1e-3)
        t += dt
        out.append((t, b))
        p_prev = p
    return out


class Simulator:
    def __init__(self, world: World, cfg: Config) -> None:
        self.w = world
        self.cfg = cfg
        self.recorder = None  # optional callable(world) invoked every few steps
        self.recorder_every = 20

    def run(self, timed_segments: list[list[tuple[float, np.ndarray]]], gripper_closed: list[bool],
            target: int | None, monitor: ConstraintMonitor | None, settle: bool = True,
            weld: tuple[int, bool] | None = None) -> SimResult:
        """Track timed joint trajectories with the position servos, then settle.

        The servos are PD (force = kp (ctrl - q) - kv qdot, plus joint damping), which lag a
        moving reference by (kv + damping) / kp * velocity -- 0.085 rad at 0.8 rad/s. The
        reference drives the PR2 with velocity control; the equivalent here is velocity
        feedforward: ctrl = q_ref + (kv + damping) / kp * qdot_ref.
        """
        w, d, m = self.w, self.w.data, self.w.model
        start_states = w.object_states()
        dt = m.opt.timestep
        acts = np.asarray(w.arm_act)
        dofs = m.jnt_dofadr[m.actuator_trnid[acts, 0]]
        ff = (-m.actuator_biasprm[acts, 2] + m.dof_damping[dofs]) / m.actuator_gainprm[acts, 0]
        every = self.cfg.constraints.check_every
        steps = 0
        for seg, closed in zip(timed_segments, gripper_closed):
            d.ctrl[w.gripper_act] = 255.0 if closed else 0.0
            times = np.array([t for t, _ in seg])
            qs = np.array([q for _, q in seg])
            T = times[-1]
            t = 0.0
            while t <= T + 1e-9:
                k = int(np.searchsorted(times, t, side="right"))
                if k >= len(times):
                    q = qs[-1]
                    v = 0.0
                else:
                    span = max(times[k] - times[k - 1], 1e-9)
                    a = (t - times[k - 1]) / span
                    q = qs[k - 1] + a * (qs[k] - qs[k - 1])
                    v = (qs[k] - qs[k - 1]) / span
                d.ctrl[w.arm_act] = q + ff * v
                mujoco.mj_step(m, d)
                steps += 1
                t += dt
                if monitor is not None and steps % every == 0:
                    v = monitor.check()
                    if v:
                        return SimResult(False, v, w.object_states(), sim_time=steps * dt, arm_q=w.arm_q())
                if self.recorder is not None and steps % self.recorder_every == 0:
                    self.recorder(w)
        if settle:
            n_settle = int(self.cfg.physics.settle_time / dt)
            for i in range(n_settle):
                mujoco.mj_step(m, d)
                steps += 1
                if monitor is not None and steps % every == 0:
                    v = monitor.check()
                    if v:
                        return SimResult(False, v, w.object_states(), sim_time=steps * dt, arm_q=w.arm_q())
                if self.recorder is not None and steps % self.recorder_every == 0:
                    self.recorder(w)
        end = w.object_states()
        moved = [oid for oid in end
                 if np.linalg.norm(end[oid][:3] - start_states[oid][:3]) > self.cfg.push.min_displacement]
        disp = float(np.linalg.norm(end[target][:2] - start_states[target][:2])) if target is not None else 0.0
        return SimResult(True, "", end, moved, disp, steps * dt, w.arm_q())


def simulate_push(sim: Simulator, arm, action, cfg: Config) -> SimResult:
    """Execute approach + push + retract from the world's current state (M4M SIMULATEPUSH)."""
    w = sim.w
    monitor = ConstraintMonitor(w, cfg, immovable_ooi=True)
    rc = cfg.robot
    segs = [time_parameterise(action.approach, rc.max_joint_speed),
            cartesian_timing(arm, action.push, rc.push_speed, rc.max_joint_speed)]
    closed = [True, True]
    if action.retract:
        segs.append(cartesian_timing(arm, action.retract, rc.push_speed, rc.max_joint_speed))
        closed.append(True)
    res = sim.run(segs, closed, action.obj_id, monitor)
    # E-M4M: valid if no constraint is violated and at least one object is moved.
    if res.valid and not res.moved:
        res.valid = False
        res.reason = f"no object moved more than {cfg.push.min_displacement * 100:.0f} cm"
    return res
