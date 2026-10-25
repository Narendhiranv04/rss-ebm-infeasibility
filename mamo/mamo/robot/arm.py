"""Arm kinematics and collision checking on a dedicated planning copy of the model.

Planning never steps physics: it poses the arm (and, when retrieving, the
grasped OoI) kinematically and asks MuJoCo's collision detector for contacts
against a chosen obstacle set, mirroring how the reference planner gives SMPL
a collision space containing exactly the objects to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mamo.world import GROUND, IMMOVABLE_GEOM, MOVABLE_GEOM, OOI_GEOM, ROBOT, SHELF_GEOM, World


_GRIPPER_TABLE: np.ndarray | None = None


def rot_from_axes(z_axis: np.ndarray, y_hint: np.ndarray) -> np.ndarray:
    """Rotation whose z-axis is ``z_axis`` and whose y-axis is closest to ``y_hint``."""
    z = z_axis / np.linalg.norm(z_axis)
    y = y_hint - (y_hint @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=1)


def tool_rotation(direction_xy: np.ndarray, pitch: float, flip: bool = False) -> np.ndarray:
    """Tool frame pointing along a horizontal direction, pitched down by ``pitch`` (rad).

    The gripper's finger axis (tool y) stays horizontal, matching the reference
    push/grasp poses (yaw = direction, pitch fixed, roll 0).
    """
    d = np.array([direction_xy[0], direction_xy[1], 0.0])
    d /= np.linalg.norm(d)
    z = np.cos(pitch) * d + np.array([0.0, 0.0, -np.sin(pitch)])
    y = np.array([-d[1], d[0], 0.0])
    if flip:
        y = -y
    return rot_from_axes(z, y)


def _rot_error(R: np.ndarray, Rt: np.ndarray) -> np.ndarray:
    """Axis-angle vector rotating R onto Rt (world frame)."""
    Re = Rt @ R.T
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, Re.flatten())
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


@dataclass(frozen=True)
class ObstacleSet:
    """Which geoms the arm must avoid.

    categories: geom categories treated as obstacles
    ignore_objects: object ids never treated as obstacles
    finger_contact_ok: object ids the gripper fingers may touch (the grasp target)
    """
    categories: frozenset[int]
    ignore_objects: frozenset[int] = frozenset()
    finger_contact_ok: frozenset[int] = frozenset()

    @staticmethod
    def everything(ignore: set[int] | None = None) -> "ObstacleSet":
        return ObstacleSet(frozenset({GROUND, SHELF_GEOM, IMMOVABLE_GEOM, OOI_GEOM, MOVABLE_GEOM}),
                           frozenset(ignore or ()))

    @staticmethod
    def static(ignore: set[int] | None = None) -> "ObstacleSet":
        """Shelf, immovable obstacles and the OoI (PLANRETRIEVAL(O_I))."""
        return ObstacleSet(frozenset({GROUND, SHELF_GEOM, IMMOVABLE_GEOM, OOI_GEOM}), frozenset(ignore or ()))


class Arm:
    """Kinematics + collision queries for the xArm7 inside a World."""

    def __init__(self, world: World) -> None:
        self.world = world
        self.cfg = world.cfg
        # Planning copy with a clearance margin on robot geoms.
        self.model = world.spec.compile()
        margin = self.cfg.robot.planning_margin
        for g in world.robot_geoms:
            self.model.geom_margin[g] = margin
        self.data = mujoco.MjData(self.model)
        self.lo = world.arm_limits[:, 0].copy()
        self.hi = world.arm_limits[:, 1].copy()
        self._gripper_qadr = [self.model.jnt_qposadr[j] for j in range(self.model.njnt)
                              if self.model.jnt_bodyid[j] in world.gripper_body_ids]
        self._gripper_states = self._measure_gripper_states()
        self._grasp_q: np.ndarray | None = None
        self.set_gripper(closed=True)
        self._cache_obstacles: dict[ObstacleSet, np.ndarray] = {}
        self.attached: tuple[int, np.ndarray, np.ndarray] | None = None
        self.n_checks = 0
        names = [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, self.model.geom_bodyid[g]) or ""
                 for g in range(self.model.ngeom)]
        self._is_finger = np.array(["finger" in n for n in names])

    # ---------------------------------------------------------------- gripper
    def _measure_gripper_states(self) -> np.ndarray:
        """Steady-state finger joint values from open to closed (rows), from a short simulation.

        The gripper is the same in every scene, so the table is computed once per process.
        """
        global _GRIPPER_TABLE
        if _GRIPPER_TABLE is not None:
            return _GRIPPER_TABLE
        m = self.world.model
        rows = []
        for ctrl in np.linspace(0.0, 255.0, 35):
            d = mujoco.MjData(m)
            d.qpos[self.world.arm_qadr] = self.cfg.robot.home_q
            # Lift objects far away so the fingers close on nothing.
            for oid, a in self.world.obj_qadr.items():
                d.qpos[a:a + 3] = [5.0 + 0.5 * oid, 5.0, 0.5]
                d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
            d.ctrl[self.world.arm_act] = self.cfg.robot.home_q
            d.ctrl[self.world.gripper_act] = ctrl
            for _ in range(600):
                mujoco.mj_step(m, d)
            rows.append(d.qpos[self._gripper_qadr].copy())
        _GRIPPER_TABLE = np.array(rows)
        return _GRIPPER_TABLE

    def _grasp_width_state(self, oid: int) -> np.ndarray:
        """Most-closed finger state that does not penetrate object ``oid`` (fingers stop on it)."""
        m, d = self.model, self.data
        og = self.world.obj_geom[oid]
        best = self._gripper_states[0]
        for qg in self._gripper_states:
            d.qpos[self._gripper_qadr] = qg
            mujoco.mj_kinematics(m, d)
            mujoco.mj_collision(m, d)
            n = d.ncon
            g1, g2, dist = d.contact.geom1[:n], d.contact.geom2[:n], d.contact.dist[:n]
            other = np.where(g1 == og, g2, np.where(g2 == og, g1, -1))
            hit = (other >= 0) & (dist < 0)
            if np.any(np.isin(m.geom_bodyid[other[hit]], list(self.world.gripper_body_ids))):
                break
            best = qg
        return best.copy()

    def gripper_qpos(self, closed: bool) -> np.ndarray:
        return self._gripper_states[-1 if closed else 0]

    def set_gripper(self, closed: bool) -> None:
        """Open or closed; while an object is attached, 'closed' means closed on that object."""
        q = self._grasp_q if closed and self._grasp_q is not None else self.gripper_qpos(closed)
        self.data.qpos[self._gripper_qadr] = q
        self.gripper_closed = closed

    # ------------------------------------------------------------------ scene
    def set_object_states(self, states: dict[int, np.ndarray]) -> None:
        for oid, s in states.items():
            a = self.world.obj_qadr[oid]
            self.data.qpos[a:a + 7] = s

    def attach(self, oid: int, q: np.ndarray) -> None:
        """Rigidly attach object ``oid`` to the TCP at its current relative pose for configuration q."""
        p, R = self.fk(q)
        a = self.world.obj_qadr[oid]
        op = self.data.qpos[a:a + 3].copy()
        oq = self.data.qpos[a + 3:a + 7].copy()
        oR = np.zeros(9)
        mujoco.mju_quat2Mat(oR, oq)
        oR = oR.reshape(3, 3)
        self.attached = (oid, R.T @ (op - p), R.T @ oR)
        # Plan with the fingers where they really stop: on the object, not fully closed.
        self._grasp_q = self._grasp_width_state(oid)
        self.set_gripper(closed=True)
        # The carried object needs the same clearance as the robot itself.
        self.model.geom_margin[self.world.obj_geom[oid]] = self.cfg.robot.planning_margin

    def detach(self) -> None:
        if self.attached is not None:
            self.model.geom_margin[self.world.obj_geom[self.attached[0]]] = 0.0
        self.attached = None
        self._grasp_q = None

    def _place_attached(self) -> None:
        if self.attached is None:
            return
        oid, rel_p, rel_R = self.attached
        p = self.data.site_xpos[self.world.tcp_site]
        R = self.data.site_xmat[self.world.tcp_site].reshape(3, 3)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, (R @ rel_R).flatten())
        a = self.world.obj_qadr[oid]
        self.data.qpos[a:a + 3] = p + R @ rel_p
        self.data.qpos[a + 3:a + 7] = q
        mujoco.mj_kinematics(self.model, self.data)

    # ------------------------------------------------------------- kinematics
    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.world.arm_qadr] = q
        mujoco.mj_kinematics(self.model, self.data)
        s = self.world.tcp_site
        return self.data.site_xpos[s].copy(), self.data.site_xmat[s].reshape(3, 3).copy()

    def ik(self, p_target: np.ndarray, R_target: np.ndarray | None, q_seed: np.ndarray,
           max_iters: int = 200, pos_tol: float = 2e-3, rot_tol: float = 0.02,
           damping: float = 1e-3, rot_weight: float = 0.3) -> np.ndarray | None:
        """Damped least-squares IK on the TCP site; returns None on failure."""
        m, d = self.model, self.data
        s = self.world.tcp_site
        dadr = self.world.arm_dadr
        q = np.clip(q_seed.astype(float).copy(), self.lo, self.hi)
        jp = np.zeros((3, m.nv))
        jr = np.zeros((3, m.nv))
        for _ in range(max_iters):
            d.qpos[self.world.arm_qadr] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            ep = p_target - d.site_xpos[s]
            if R_target is not None:
                er = _rot_error(d.site_xmat[s].reshape(3, 3), R_target)
            else:
                er = np.zeros(3)
            if np.linalg.norm(ep) < pos_tol and np.linalg.norm(er) < rot_tol:
                return q
            mujoco.mj_jacSite(m, d, jp, jr, s)
            if R_target is not None:
                J = np.vstack([jp[:, dadr], rot_weight * jr[:, dadr]])
                e = np.concatenate([ep, rot_weight * er])
            else:
                J = jp[:, dadr]
                e = ep
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(J.shape[0]), e)
            step = np.max(np.abs(dq))
            if step > 0.2:
                dq *= 0.2 / step
            q = np.clip(q + dq, self.lo, self.hi)
        return None

    def ik_multi(self, p_target: np.ndarray, R_target: np.ndarray | None, seeds: list[np.ndarray],
                 rng: np.random.Generator, restarts: int = 6,
                 obstacles: ObstacleSet | None = None) -> np.ndarray | None:
        """IK from the given seeds and then random restarts; optionally require collision-free."""
        tries = list(seeds) + [rng.uniform(self.lo, self.hi) for _ in range(restarts)]
        for seed in tries:
            q = self.ik(p_target, R_target, seed)
            if q is None:
                continue
            if obstacles is None or self.collision_free(q, obstacles):
                return q
        return None

    # -------------------------------------------------------------- collision
    def _obstacle_mask(self, obs: ObstacleSet) -> np.ndarray:
        mask = self._cache_obstacles.get(obs)
        if mask is None:
            cat = self.world.geom_cat
            mask = np.isin(cat, list(obs.categories))
            for oid in obs.ignore_objects:
                mask[self.world.obj_geom[oid]] = False
            self._cache_obstacles[obs] = mask
        if self.attached is not None:
            mask = mask.copy()
            mask[self.world.obj_geom[self.attached[0]]] = False
        return mask

    def collision_free(self, q: np.ndarray, obs: ObstacleSet) -> bool:
        m, d = self.model, self.data
        d.qpos[self.world.arm_qadr] = q
        mujoco.mj_kinematics(m, d)
        self._place_attached()
        mujoco.mj_collision(m, d)
        self.n_checks += 1
        n = d.ncon
        if n == 0:
            return True
        g1 = d.contact.geom1[:n]
        g2 = d.contact.geom2[:n]
        dist = d.contact.dist[:n]
        cat = self.world.geom_cat
        mask = self._obstacle_mask(obs)
        r1 = cat[g1] == ROBOT
        r2 = cat[g2] == ROBOT
        hit = (r1 & mask[g2]) | (r2 & mask[g1])
        if obs.finger_contact_ok and np.any(hit):
            ok_geoms = np.zeros(m.ngeom, dtype=bool)
            ok_geoms[[self.world.obj_geom[o] for o in obs.finger_contact_ok]] = True
            allowed = (self._is_finger[g1] & ok_geoms[g2]) | (self._is_finger[g2] & ok_geoms[g1])
            hit &= ~allowed
        # Robot self-collision (MuJoCo already skips parent/child pairs).
        self_hit = r1 & r2 & (dist < 0)
        if np.any(hit | self_hit):
            return False
        if self.attached is not None:
            ag = self.world.obj_geom[self.attached[0]]
            a1 = g1 == ag
            a2 = g2 == ag
            # The carried object must keep the planning margin from obstacles (not the robot).
            if np.any((a1 & mask[g2]) | (a2 & mask[g1])):
                return False
        return True

    def contacts(self, q: np.ndarray, obs: ObstacleSet) -> list[tuple[str, str, float]]:
        """Debug helper: names of colliding robot/obstacle geom pairs."""
        m, d = self.model, self.data
        d.qpos[self.world.arm_qadr] = q
        mujoco.mj_kinematics(m, d)
        self._place_attached()
        mujoco.mj_collision(m, d)
        mask = self._obstacle_mask(obs)
        cat = self.world.geom_cat
        out = []
        for i in range(d.ncon):
            c = d.contact[i]
            if (cat[c.geom1] == ROBOT and mask[c.geom2]) or (cat[c.geom2] == ROBOT and mask[c.geom1]):
                name = lambda g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or \
                    mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g])
                out.append((name(c.geom1), name(c.geom2), float(c.dist)))
        return out

    def edge_free(self, q0: np.ndarray, q1: np.ndarray, obs: ObstacleSet) -> bool:
        step = self.cfg.robot.collision_check_step
        n = max(1, int(np.ceil(np.max(np.abs(q1 - q0)) / step)))
        for i in range(1, n + 1):
            if not self.collision_free(q0 + (q1 - q0) * (i / n), obs):
                return False
        return True
