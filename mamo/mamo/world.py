"""MuJoCo world: the shelf, the scene objects and a mounted xArm7.

Everything in the scene is simulated as in the reference Bullet setup
(scripts/sim.py): shelf parts are static, every other object (including
immovable obstacles and the OoI) is a free rigid body with the mass and
friction from the scene file. Interaction constraints, not the simulator,
are what forbid touching immovables.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from mamo.config import Config
from mamo.scene import CYLINDER, Scene, SceneObject

ASSET_DIR = Path(__file__).parent / "assets"
XARM_XML = ASSET_DIR / "xarm7" / "xarm7.xml"

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ROBOT_PREFIX = "robot/"

# Geom categories used by collision filtering and constraint monitoring.
GROUND, SHELF_GEOM, ROBOT, MOVABLE_GEOM, IMMOVABLE_GEOM, OOI_GEOM = range(6)


def _quat_from_rpy(r: float, p: float, y: float) -> np.ndarray:
    """Quaternion for R = Rz(yaw) Ry(pitch) Rx(roll), Bullet's getQuaternionFromEuler."""
    q = np.zeros(4)
    mujoco.mju_euler2Quat(q, np.array([r, p, y]), "XYZ")
    return q


def cfg_for_scene(scene: Scene, cfg: Config) -> Config:
    """``cfg`` with the robot base and home pose (meta "robot") and the parameter overrides
    (meta "config", ``{"section.field": value}``) of the scene's environment, if any."""
    r, over = scene.meta.get("robot"), scene.meta.get("config")
    if not r and not over:
        return cfg
    cfg = copy.deepcopy(cfg)
    if r:
        cfg.robot.base_pos = tuple(r["base"])
        cfg.robot.base_yaw = float(r.get("yaw", 0.0))
        if r.get("home_q"):
            cfg.robot.home_q = tuple(r["home_q"])
    for key, val in (over or {}).items():
        section, name = key.split(".")
        setattr(getattr(cfg, section), name, tuple(val) if isinstance(val, list) else val)
    return cfg


def relocate_goals(scene: Scene) -> dict[int, dict]:
    """Goal entries of a relocate task by object id (empty for retrieve tasks)."""
    t = scene.task
    return {int(g["id"]): g for g in t["targets"]} if t["type"] == "relocate" else {}


@dataclass
class Snapshot:
    qpos: np.ndarray
    qvel: np.ndarray
    act: np.ndarray
    ctrl: np.ndarray
    time: float
    eq_active: np.ndarray


class World:
    """A compiled MuJoCo model of one MAMO scene plus helpers."""

    def __init__(self, scene: Scene, cfg: Config | None = None) -> None:
        self.scene = scene
        self.cfg = cfg_for_scene(scene, cfg or Config())
        self.spec = self._build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self._index()
        self.reset()

    # ------------------------------------------------------------------ build
    def _build_spec(self) -> mujoco.MjSpec:
        pc = self.cfg.physics
        spec = mujoco.MjSpec()
        spec.option.timestep = pc.timestep
        spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        spec.option.impratio = 10.0
        spec.visual.global_.offwidth = 1280
        spec.visual.global_.offheight = 960

        wb = spec.worldbody
        wb.add_light(pos=[0.6, -0.5, 2.5], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
        wb.add_geom(name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1],
                    rgba=[0.9, 0.9, 0.92, 1])

        for o in self.scene.shelf_parts:
            wb.add_geom(name=f"shelf_{o.id}", type=mujoco.mjtGeom.mjGEOM_BOX,
                        size=[o.x_size, o.y_size, o.z_size], pos=[o.x, o.y, o.z],
                        quat=_quat_from_rpy(o.roll, o.pitch, o.yaw),
                        friction=[pc.shelf_mu, 0.005, 0.0001], rgba=[0.55, 0.55, 0.58, 0.35])

        for o in self.scene.objects:
            if o.is_shelf:
                continue
            self._add_object(spec, o)
        # Relocate goals: translucent, non-colliding copies of the targets at their goal poses.
        for oid, g in relocate_goals(self.scene).items():
            o = self.scene.by_id(oid)
            x, y, z, yaw = g["goal"]
            cyl = o.shape == CYLINDER
            wb.add_geom(name=f"goal_{oid}", type=mujoco.mjtGeom.mjGEOM_CYLINDER if cyl else mujoco.mjtGeom.mjGEOM_BOX,
                        size=[o.radius, 0.5 * o.z_size, 0] if cyl else [o.x_size, o.y_size, o.z_size],
                        pos=[x, y, z], quat=_quat_from_rpy(0, 0, yaw), contype=0, conaffinity=0, group=1,
                        rgba=[1.0, 0.84, 0.0, 0.25])

        # Robot on a (visual-only) pedestal in front of the shelf.
        rc = self.cfg.robot
        bx, by, bz = rc.base_pos
        wb.add_geom(name="pedestal", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.08, bz / 2, 0],
                    pos=[bx, by, bz / 2], contype=0, conaffinity=0, rgba=[0.3, 0.3, 0.32, 1])
        robot = mujoco.MjSpec.from_file(str(XARM_XML))
        robot.meshdir = str(XARM_XML.parent / "assets")
        if pc.gravity_compensation:
            for b in robot.bodies:
                b.gravcomp = 1.0
        frame = wb.add_frame(pos=[bx, by, bz - 0.12], quat=_quat_from_rpy(0, 0, rc.base_yaw))
        spec.attach(robot, prefix=ROBOT_PREFIX, frame=frame)
        # Fingers act as rigid pushing surfaces: give the finger meshes collision.
        for g in spec.geoms:
            if g.name == "" and g.parent.name in (ROBOT_PREFIX + "left_finger", ROBOT_PREFIX + "right_finger") \
                    and g.type == mujoco.mjtGeom.mjGEOM_MESH:
                g.contype, g.conaffinity, g.group = 1, 1, 2

        if pc.combine_friction_as_product:
            self._add_friction_pairs(spec)
        # Grasp constraint, activated once the gripper has closed on the OoI
        # (the reference simulator adds a fixed constraint the same way).
        # Stiff (time constant 2 timesteps): with MuJoCo's soft default a tall OoI
        # sags up to 3 cm off the planned grasp while it is carried.
        spec.add_equality(name="grasp_weld", type=mujoco.mjtEq.mjEQ_WELD,
                          name1=ROBOT_PREFIX + "xarm_gripper_base_link", name2=f"obj_{self.scene.ooi_id}",
                          objtype=mujoco.mjtObj.mjOBJ_BODY, active=False,
                          solref=[2 * pc.timestep, 1.0], solimp=[0.99, 0.999, 0.001, 0.5, 2.0])
        return spec

    def _add_object(self, spec: mujoco.MjSpec, o: SceneObject) -> None:
        if o.id in self.scene.target_ids:
            rgba = [1.0, 0.84, 0.0, 1]
        elif o.movable:
            rgba = [0.15, 0.35, 0.9, 1]
        else:
            rgba = [0.75, 0.15, 0.15, 1]
        body = spec.worldbody.add_body(name=f"obj_{o.id}", pos=[o.x, o.y, o.z],
                                       quat=_quat_from_rpy(o.roll, o.pitch, o.yaw))
        body.add_freejoint(name=f"obj_{o.id}")
        if o.shape == CYLINDER:
            size = [o.radius, 0.5 * o.z_size, 0]
            gtype = mujoco.mjtGeom.mjGEOM_CYLINDER
        else:
            size = [o.x_size, o.y_size, o.z_size]
            gtype = mujoco.mjtGeom.mjGEOM_BOX
        mass = o.mass if o.mass > 0 else 0.1
        body.add_geom(name=f"obj_{o.id}", type=gtype, size=size, mass=mass,
                      friction=[o.mu, 0.005, 0.0001], rgba=rgba)

    def _add_friction_pairs(self, spec: mujoco.MjSpec) -> None:
        """Bullet combines friction multiplicatively; MuJoCo takes the max.

        Explicit contact pairs reproduce the product for object-shelf and
        object-object contacts (robot contacts keep MuJoCo's default rule).
        """
        pc = self.cfg.physics
        objs = [o for o in self.scene.objects if not o.is_shelf]
        for o in objs:
            for s in self.scene.shelf_parts:
                mu = o.mu * pc.shelf_mu
                spec.add_pair(geomname1=f"obj_{o.id}", geomname2=f"shelf_{s.id}",
                              friction=[mu, mu, 0.005, 0.0001, 0.0001], condim=4)
            spec.add_pair(geomname1=f"obj_{o.id}", geomname2="ground",
                          friction=[o.mu, o.mu, 0.005, 0.0001, 0.0001], condim=4)
        for i, a in enumerate(objs):
            for b in objs[i + 1:]:
                mu = a.mu * b.mu
                spec.add_pair(geomname1=f"obj_{a.id}", geomname2=f"obj_{b.id}",
                              friction=[mu, mu, 0.005, 0.0001, 0.0001], condim=4)

    # ------------------------------------------------------------------ index
    def _index(self) -> None:
        m = self.model
        self.arm_joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, ROBOT_PREFIX + j)
                              for j in ARM_JOINTS]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_joint_ids])
        self.arm_dadr = np.array([m.jnt_dofadr[j] for j in self.arm_joint_ids])
        self.arm_limits = np.array([m.jnt_range[j] for j in self.arm_joint_ids])
        self.arm_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ROBOT_PREFIX + f"act{i}")
                                 for i in range(1, 8)])
        self.gripper_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, ROBOT_PREFIX + "gripper")
        self.tcp_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, ROBOT_PREFIX + "link_tcp")
        self.robot_body_ids = {b for b in range(m.nbody)
                               if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(ROBOT_PREFIX)}
        self.gripper_body_ids = {b for b in self.robot_body_ids
                                 if any(k in mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b)
                                        for k in ("gripper", "finger", "knuckle"))}

        self.obj_body: dict[int, int] = {}
        self.obj_geom: dict[int, int] = {}
        self.obj_qadr: dict[int, int] = {}
        self.obj_dadr: dict[int, int] = {}
        for o in self.scene.objects:
            if o.is_shelf:
                continue
            b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, f"obj_{o.id}")
            j = m.body_jntadr[b]
            self.obj_body[o.id] = b
            self.obj_geom[o.id] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"obj_{o.id}")
            self.obj_qadr[o.id] = m.jnt_qposadr[j]
            self.obj_dadr[o.id] = m.jnt_dofadr[j]

        cat = np.full(m.ngeom, -1, dtype=np.int64)
        gobj = np.full(m.ngeom, -1, dtype=np.int64)
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            body = m.geom_bodyid[g]
            if name == "ground":
                cat[g] = GROUND
            elif name.startswith("shelf_"):
                cat[g] = SHELF_GEOM
            elif body in self.robot_body_ids:
                cat[g] = ROBOT
        for o in self.scene.objects:
            if o.is_shelf:
                continue
            g = self.obj_geom[o.id]
            gobj[g] = o.id
            # Movables off the problem's surfaces are fixed obstacles to it (Scene.active_ids);
            # relocate targets are ordinary movables.
            if o.id == self.scene.ooi_id and self.scene.task["type"] == "retrieve":
                cat[g] = OOI_GEOM
            elif o.movable and self.scene.is_active(o.id):
                cat[g] = MOVABLE_GEOM
            else:
                cat[g] = IMMOVABLE_GEOM
        self.geom_cat = cat
        self.geom_obj = gobj
        self.robot_geoms = np.flatnonzero((cat == ROBOT) & (m.geom_contype > 0))

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.set_arm_q(np.array(self.cfg.robot.home_q))
        self.data.ctrl[self.arm_act] = self.cfg.robot.home_q
        self.data.ctrl[self.gripper_act] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def snapshot(self) -> Snapshot:
        d = self.data
        return Snapshot(d.qpos.copy(), d.qvel.copy(), d.act.copy(), d.ctrl.copy(), d.time,
                        d.eq_active.copy())

    def restore(self, s: Snapshot) -> None:
        d = self.data
        d.qpos[:] = s.qpos
        d.qvel[:] = s.qvel
        d.act[:] = s.act
        d.ctrl[:] = s.ctrl
        d.time = s.time
        d.eq_active[:] = s.eq_active
        mujoco.mj_forward(self.model, d)

    def arm_q(self, data: mujoco.MjData | None = None) -> np.ndarray:
        d = data or self.data
        return d.qpos[self.arm_qadr].copy()

    def set_arm_q(self, q: np.ndarray, data: mujoco.MjData | None = None) -> None:
        d = data or self.data
        d.qpos[self.arm_qadr] = q
        d.qvel[self.arm_dadr] = 0.0

    def object_pose(self, oid: int, data: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = data or self.data
        a = self.obj_qadr[oid]
        return d.qpos[a:a + 3].copy(), d.qpos[a + 3:a + 7].copy()

    def set_object_pose(self, oid: int, pos: np.ndarray, quat: np.ndarray,
                        data: mujoco.MjData | None = None) -> None:
        d = data or self.data
        a = self.obj_qadr[oid]
        d.qpos[a:a + 3] = pos
        d.qpos[a + 3:a + 7] = quat
        v = self.obj_dadr[oid]
        d.qvel[v:v + 6] = 0.0

    def object_states(self, data: mujoco.MjData | None = None) -> dict[int, np.ndarray]:
        """Pose (x, y, z, qw, qx, qy, qz) of every non-shelf object."""
        d = data or self.data
        return {oid: d.qpos[a:a + 7].copy() for oid, a in self.obj_qadr.items()}

    def set_object_states(self, states: dict[int, np.ndarray], data: mujoco.MjData | None = None) -> None:
        d = data or self.data
        for oid, s in states.items():
            a = self.obj_qadr[oid]
            d.qpos[a:a + 7] = s
            v = self.obj_dadr[oid]
            d.qvel[v:v + 6] = 0.0

    def tcp_pose(self, data: mujoco.MjData | None = None) -> tuple[np.ndarray, np.ndarray]:
        d = data or self.data
        return d.site_xpos[self.tcp_site].copy(), d.site_xmat[self.tcp_site].reshape(3, 3).copy()

    def scene_with_states(self, states: dict[int, np.ndarray]) -> Scene:
        """A Scene whose object poses are taken from simulator states."""
        sc = self.scene.copy()
        for o in sc.objects:
            if o.id in states:
                s = states[o.id]
                o.x, o.y, o.z = map(float, s[:3])
                o.roll, o.pitch, o.yaw = quat_to_rpy(s[3:7])
        return sc


def quat_to_rpy(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = map(float, q)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


def tilt_deg(q: np.ndarray) -> float:
    """Angle between the object's z-axis and the world z-axis."""
    w, x, y, z = map(float, q)
    r22 = 1 - 2 * (x * x + y * y)
    return math.degrees(math.acos(max(-1.0, min(1.0, r22))))
