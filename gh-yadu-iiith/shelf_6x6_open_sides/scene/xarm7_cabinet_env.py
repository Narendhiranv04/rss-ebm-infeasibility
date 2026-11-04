"""
MuJoCo environment: UFactory xArm7 (from MuJoCo Menagerie, with its stock gripper)
+ open-front cabinet divided into cells + pick-and-placeable boxes / cylinders + static obstacles.

Setup
-----
    pip install mujoco numpy pillow
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie
    # (or set XARM7_XML below / env var XARM7_XML to wherever xarm7.xml lives)
    # rrt_star.py must sit next to this file.

Run
---
    python xarm7_cabinet_env.py                    # viewer + scripted single-cell scene (see SCENE below)
    python xarm7_cabinet_env.py --headless         # no window
    python xarm7_cabinet_env.py --random 4         # 4 random objects of random sizes, random target
    python xarm7_cabinet_env.py --no-planner       # no RRT* for transits (straight Cartesian lines)
    python xarm7_cabinet_env.py --lift 0.01        # carry the object only 1 cm above the shelf

Configuring a scene
-------------------
    env = XArm7CabinetEnv(
        n_cells=1, cell_w=0.30,                    # number / width of cabinet cells
        objects=[dict(name="target", shape="cylinder", size=0.02, height=0.08,
                      color="green", cell=0, depth=0.18, dy=0.05), ...],
        obstacles=[dict(cell=0, depth=0.12, dy=-0.10, size=[0.02, 0.02, 0.04]),   # on the shelf
                   dict(pos=[0.45, -0.30, 0.25], size=[0.05, 0.05, 0.25])])       # anywhere (world)
    env.pick_and_place("target", cell=0, depth=0.08, dy=-0.05, lift=0.015)   # carried 1.5 cm high

    object keys : shape 'box'|'cylinder', size = half-width / radius (m), height (m), color (name or
                  "r g b a"), cell, depth (from cabinet front to object centre), dy (from cell centre),
                  optional name, mass, friction
    obstacle    : size = box half-sizes; either cell/depth/dy (it then rests on the shelf) or pos
                  (world centre); optional name, rgba

Motion pipeline
---------------
    transit outside the cabinet : joint-space RRT* (rrt_star.py), collision-checked with a 1.5 cm margin
    inside the cell             : straight Cartesian moves with a fixed hand orientation, every IK
                                  waypoint collision-checked with a tight margin before executing
    grasp selection             : searches hand pitch (nose-down tilt) and grasp height so that hand
                                  and wrist clear the shelf, the obstacles and the other objects at
                                  both the pick and the place location
"""
import os
import sys
import itertools
import numpy as np
import mujoco

from rrt_star import (MujocoCollisionChecker, RRTStar, shortcut, smooth,
                      time_parameterize, path_length, densify)

# ----------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
XARM7_XML = os.environ.get(
    "XARM7_XML", os.path.join(HERE, "mujoco_menagerie", "ufactory_xarm7", "xarm7.xml"))

# ----------------------------------------------------------------------------- scene parameters
CAB_X0 = 0.62                # cabinet front face (x)
CAB_DEPTH = 0.34
CELL_W = 0.30                # default cell width (y); override with XArm7CabinetEnv(cell_w=...)
SHELF_Z = 0.33               # shelf surface height
CELL_H = 0.32                # clear height inside a cell
WALL = 0.01
STANDOFF = 0.07              # how far in front of the cabinet face the gripper lines up before sliding in

BASE_Z = 0.12                # xArm7 base height (it sits on a 12 cm pedestal)
GRASP_SITE_Z = 0.142         # centre of the finger pads along the gripper's approach (z) axis
GRIPPER_OPEN, GRIPPER_CLOSED = 0.0, 255.0

# gripper / grasp limits (measured on the Menagerie model: pad gap 8.9 cm open, 0.4 cm closed)
GRIP_MIN_WIDTH = 0.010       # narrowest object the pads can hold (m)
GRIP_MAX_WIDTH = 0.075       # widest object that still leaves approach clearance (m)
GRIP_CLEARANCE = 0.012       # per-side clearance when pre-shaping the fingers around an object
APPROACH_DIST = 0.03         # fingers stop this far (+ object half-width) in front of the object
LIFT = 0.015                 # default lift above the shelf while carrying (1.5 cm)
MIN_LIFT, MAX_LIFT = 0.008, 0.10   # 8 mm keeps the object clear of the shelf with margin + slip
PAD_HALF_H = 0.012           # keep the finger pads this far inside the object's height
MIN_HEIGHT = 0.035           # fingertips need ~2.2 cm shelf clearance + the pads must fit on the object
PITCHES_DEG = (0, 5, 10, 15, 20, 25)   # candidate nose-down hand tilts for deep / low grasps
MAX_REACH_DEPTH = 0.25       # beyond this the wrist cannot clear the shelf (any pitch)

COLORS = {"red": "0.85 0.1 0.1 1", "blue": "0.1 0.2 0.85 1", "green": "0.1 0.7 0.2 1",
          "yellow": "0.95 0.8 0.1 1", "orange": "0.95 0.5 0.1 1", "purple": "0.55 0.2 0.75 1"}

# desired gripper orientation: approach axis = world +x (horizontal, into the cell),
# finger-opening axis = world y.  Columns are gripper x, y, z axes in the world frame.
R_DES = np.array([[0.0, 0.0, 1.0],
                  [0.0, 1.0, 0.0],
                  [-1.0, 0.0, 0.0]])

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
HAND_ROOT = "xarm_gripper_base_link"

# planner settings
PLAN_MARGIN = 0.015          # safety buffer for RRT* transits (m)
CELL_MARGIN = 0.003          # safety buffer for moves inside the (tight) cell (m)
PLAN_VMAX = 1.0              # joint speed limit for planned transits (rad/s)
PLAN_AMAX = 1.5              # joint acceleration limit (rad/s^2); joint 3 saturates its 30 Nm limit near 2
CELL_VMAX = 0.4              # slower joint limits for the in-cell Cartesian moves
CELL_AMAX = 0.8


def R_pitch(deg):
    """Hand orientation tilted nose-down by `deg` (rotation about world y applied to R_DES)."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]) @ R_DES


def cell_center_y(i, n_cells, cell_w=CELL_W):
    return (i - (n_cells - 1) / 2.0) * cell_w


# ----------------------------------------------------------------------------- scene description
def normalize_objects(objects):
    """Fill in defaults so every object dict has all keys."""
    out = []
    for i, o in enumerate(objects or []):
        o = dict(o)
        o.setdefault("name", f"obj{i}")
        o.setdefault("shape", "box")
        o.setdefault("size", 0.02)
        o.setdefault("height", 0.05)
        o.setdefault("color", "red")
        o.setdefault("cell", 0)
        o.setdefault("depth", 0.17)
        o.setdefault("dy", 0.0)
        o.setdefault("friction", 1.5)
        vol = (4 if o["shape"] == "box" else np.pi) * o["size"] ** 2 * o["height"]
        o.setdefault("mass", max(0.02, 400.0 * vol))           # ~ light plastic / wood
        if o["shape"] not in ("box", "cylinder"):
            raise ValueError(f"{o['name']}: shape must be 'box' or 'cylinder'")
        out.append(o)
    return out


def normalize_obstacles(obstacles, n_cells, cell_w):
    """Resolve cell/depth/dy obstacles into world positions (resting on the shelf)."""
    out = []
    for k, ob in enumerate(obstacles or []):
        ob = dict(ob)
        ob.setdefault("name", f"obstacle{k}")
        ob.setdefault("rgba", "0.9 0.6 0.1 1")
        size = np.asarray(ob["size"], float)
        if "pos" not in ob:
            ob["pos"] = [CAB_X0 + ob.get("depth", 0.17),
                         cell_center_y(ob.get("cell", 0), n_cells, cell_w) + ob.get("dy", 0.0),
                         SHELF_Z + size[2]]
        ob["pos"], ob["size"] = list(map(float, ob["pos"])), list(map(float, size))
        out.append(ob)
    return out


def footprint_radius(o):
    return o["size"] * (np.sqrt(2.0) if o["shape"] == "box" else 1.0)


def validate_scene(objects, obstacles, n_cells, cell_w):
    """Raise ValueError for objects the gripper can't hold or that don't fit / overlap."""
    half_in = cell_w / 2 - WALL / 2
    for o in objects:
        w = 2 * o["size"]
        if not GRIP_MIN_WIDTH <= w <= GRIP_MAX_WIDTH:
            raise ValueError(f"{o['name']}: width {w*100:.1f} cm outside the graspable range "
                             f"{GRIP_MIN_WIDTH*100:.0f}-{GRIP_MAX_WIDTH*100:.1f} cm")
        if o["height"] < MIN_HEIGHT or o["height"] > CELL_H - 0.10:
            raise ValueError(f"{o['name']}: height {o['height']*100:.1f} cm must be within "
                             f"{MIN_HEIGHT*100:.1f}-{(CELL_H-0.10)*100:.0f} cm")
        if not 0 <= o["cell"] < n_cells:
            raise ValueError(f"{o['name']}: cell {o['cell']} does not exist (n_cells={n_cells})")
        r = footprint_radius(o)
        if abs(o["dy"]) + r > half_in or not r < o["depth"] < CAB_DEPTH - r:
            raise ValueError(f"{o['name']}: does not fit inside cell {o['cell']} at depth "
                             f"{o['depth']}, dy {o['dy']}")
    for a, b in itertools.combinations(objects, 2):
        if a["cell"] == b["cell"] and np.hypot(a["depth"] - b["depth"], a["dy"] - b["dy"]) \
                < footprint_radius(a) + footprint_radius(b) + 0.005:
            raise ValueError(f"objects {a['name']} and {b['name']} overlap")
    for o in objects:
        x = CAB_X0 + o["depth"]
        y = cell_center_y(o["cell"], n_cells, cell_w) + o["dy"]
        r = footprint_radius(o)
        for ob in obstacles:
            (px, py, pz), (hx, hy, hz) = ob["pos"], ob["size"]
            if abs(x - px) < hx + r and abs(y - py) < hy + r and pz - hz < SHELF_Z + o["height"]:
                raise ValueError(f"object {o['name']} overlaps {ob['name']}")


def random_objects(n_objects, n_cells=1, cell_w=CELL_W, seed=0, size_range=(0.012, 0.03),
                   height_range=(0.04, 0.10), depth_range=(0.06, 0.22),
                   shapes=("box", "cylinder"), colors=("red", "blue", "green", "yellow"),
                   obstacles=(), max_tries=2000):
    """`n_objects` random non-overlapping boxes / cylinders spread over the cells."""
    rng = np.random.default_rng(seed)
    obstacles = normalize_obstacles(obstacles, n_cells, cell_w)
    objs = []
    for _ in range(max_tries):
        if len(objs) == n_objects:
            break
        size = float(rng.uniform(*size_range))
        cand = dict(shape=str(rng.choice(shapes)), color=str(rng.choice(colors)), size=round(size, 4),
                    height=round(float(rng.uniform(*height_range)), 3), cell=len(objs) % n_cells,
                    depth=round(float(rng.uniform(*depth_range)), 3),
                    dy=round(float(rng.uniform(-1, 1) * (cell_w / 2 - WALL / 2 - 1.42 * size - 0.005)), 3))
        try:
            validate_scene(normalize_objects(objs + [cand]), obstacles, n_cells, cell_w)
            objs.append(cand)
        except ValueError:
            continue
    if len(objs) < n_objects:
        raise RuntimeError(f"could only fit {len(objs)} of {n_objects} objects")
    return objs


# ----------------------------------------------------------------------------- scene XML (no robot)
def build_scene_xml(n_cells=1, objects=None, obstacles=None, cell_w=CELL_W):
    """objects / obstacles: already-normalised lists (see normalize_objects / normalize_obstacles)."""
    total_w = n_cells * cell_w
    cx = CAB_X0 + CAB_DEPTH / 2
    top_z = SHELF_Z + CELL_H

    cab = [
        f'<geom name="shelf" type="box" pos="{cx} 0 {SHELF_Z/2}" size="{CAB_DEPTH/2} {total_w/2+WALL} {SHELF_Z/2}" rgba="0.55 0.55 0.58 1"/>',
        f'<geom name="ceiling" type="box" pos="{cx} 0 {top_z+WALL}" size="{CAB_DEPTH/2} {total_w/2+WALL} {WALL}" rgba="0.45 0.45 0.48 1"/>',
        f'<geom name="back" type="box" pos="{CAB_X0+CAB_DEPTH+WALL} 0 {(top_z+SHELF_Z)/2}" size="{WALL} {total_w/2+WALL} {CELL_H/2+WALL}" rgba="0.4 0.4 0.43 1"/>',
    ]
    for k in range(n_cells + 1):
        y = (k - n_cells / 2.0) * cell_w
        cab.append(f'<geom name="div{k}" type="box" pos="{cx} {y} {SHELF_Z+CELL_H/2}" '
                   f'size="{CAB_DEPTH/2} {WALL/2} {CELL_H/2}" rgba="0.5 0.5 0.53 1"/>')
    for ob in obstacles or []:
        cab.append(f'<geom name="{ob["name"]}" type="box" pos="{" ".join(map(str, ob["pos"]))}" '
                   f'size="{" ".join(map(str, ob["size"]))}" rgba="{ob["rgba"]}"/>')

    bodies = []
    for o in objects or []:
        y = cell_center_y(o["cell"], n_cells, cell_w) + o["dy"]
        x = CAB_X0 + o["depth"]
        z = SHELF_Z + o["height"] / 2 + 0.001
        rgba = COLORS.get(o["color"], o["color"])
        h2 = o["height"] / 2
        size = f'{o["size"]} {o["size"]} {h2}' if o["shape"] == "box" else f'{o["size"]} {h2}'
        g = (f'<geom name="{o["name"]}_g" type="{o["shape"]}" size="{size}" rgba="{rgba}" '
             f'mass="{o["mass"]}" friction="{o["friction"]} 0.02 0.002"/>')
        bodies.append(f'<body name="{o["name"]}" pos="{x} {y} {z}"><freejoint name="{o["name"]}_free"/>{g}</body>')

    nl = "\n    "
    return f"""
<mujoco model="xarm7_cabinet_scene">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="5"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="4096"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.32 0.4" rgb2="0.32 0.4 0.5" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0.3 0 2.5" dir="0 0 -1" diffuse="0.9 0.9 0.9" directional="true"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="grid"/>
    <geom name="pedestal" type="cylinder" pos="0 0 {BASE_Z/2}" size="0.06 {BASE_Z/2}" rgba="1 1 1 1"/>
    {nl.join(cab)}
    {nl.join(bodies)}
  </worldbody>
</mujoco>
"""


# ----------------------------------------------------------------------------- environment
class XArm7CabinetEnv:
    def __init__(self, n_cells=1, objects=None, obstacles=None, n_objects=3, cell_w=CELL_W, seed=0,
                 xml_path=XARM7_XML, use_planner=True, verbose=True):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"xarm7.xml not found at {xml_path}\n"
                "Run:  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie\n"
                "or set the XARM7_XML environment variable.")
        self.n_cells, self.cell_w = n_cells, cell_w
        self.use_planner, self.verbose, self.seed = use_planner, verbose, seed
        self.obstacles = normalize_obstacles(obstacles, n_cells, cell_w)
        if objects is None:
            objects = random_objects(n_objects, n_cells, cell_w, seed, obstacles=self.obstacles)
        self.objects = normalize_objects(objects)
        validate_scene(self.objects, self.obstacles, n_cells, cell_w)
        self.obj_index = {o["name"]: i for i, o in enumerate(self.objects)}
        if len(self.obj_index) != len(self.objects):
            raise ValueError("object names must be unique")

        # --- scene + attach xArm7 (stock gripper included)
        scene = mujoco.MjSpec.from_string(
            build_scene_xml(n_cells, self.objects, self.obstacles, cell_w))
        arm = mujoco.MjSpec.from_file(xml_path)
        for b in arm.bodies:                                   # gravity-compensate the arm links
            b.gravcomp = 1.0
        arm.body(HAND_ROOT).add_site(                          # centre of the finger pads
            name="grasp", pos=[0, 0, GRASP_SITE_Z], size=[0.006, 0, 0], rgba=[0, 1, 0, 1])
        frame = scene.worldbody.add_frame(pos=[0, 0, 0])
        scene.attach(arm, frame=frame, prefix="")
        self.model = scene.compile()
        self.data = mujoco.MjData(self.model)
        self.plan_model = self._build_planning_model(scene)

        m = self.model
        self.qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.jrange = np.clip(np.array([m.jnt_range[m.joint(j).id] for j in ARM_JOINTS]), -3.1, 3.1)  # avoid multi-turn solutions
        self.act_arm = [m.actuator(f"act{i}").id for i in range(1, 8)]
        self.act_grip = m.actuator("gripper").id
        self.site_id = m.site("grasp").id
        self.obj_geom = [m.geom(o["name"] + "_g").id for o in self.objects]
        self.ik_data = mujoco.MjData(m)                        # scratch data for IK
        self.q_home = np.array([0.0, -0.25, 0.0, 0.9, 0.0, 1.15, 0.0])  # Menagerie "home"
        self.viewer = None
        self.held = None                                       # index of the object in the gripper
        self.grasp = None                                      # grasp used for the held object

        # --- collision checker on the planning model
        pm = self.plan_model
        root = pm.body("link_base").id
        robot = [b for b in range(pm.nbody) if pm.body_rootid[b] == root and pm.body_weldid[b] != 0]
        hand = self._subtree(pm, pm.body(HAND_ROOT).id)
        self.checker = MujocoCollisionChecker(pm, self.qadr, robot, hand, pm.site("grasp").id,
                                              margin=PLAN_MARGIN)
        self.last_plan = None                                  # info about the most recent plan
        kp = m.actuator_gainprm[self.act_arm, 0]                # velocity feed-forward gain (see execute)
        kv = -m.actuator_biasprm[self.act_arm, 2]
        self.servo_lead = (kv + m.dof_damping[self.dadr]) / kp
        self.max_track_err = 0.03                              # rad; warn if execution deviates more
        self._calibrate_gripper()
        self.reset()

    # -- planning model: same scene, but with always-touching pairs excluded and safety margins
    @staticmethod
    def _subtree(m, root):
        out = {root}
        for b in range(m.nbody):
            a = b
            while a > 0:
                if a in out:
                    out.add(b); break
                a = m.body_parentid[a]
        return sorted(out)

    def _build_planning_model(self, spec):
        m = self.model
        hand = [m.body(b).name for b in self._subtree(m, m.body(HAND_ROOT).id)]
        for a, b in itertools.combinations(hand, 2):          # gripper parts touch each other by design
            e = spec.add_exclude(); e.bodyname1, e.bodyname2 = a, b
        e = spec.add_exclude(); e.bodyname1, e.bodyname2 = "link_base", "link1"   # base is welded to world
        pm = spec.compile()
        # MuJoCo's mid-phase (per-body BVH) culls geom pairs whose bounding boxes don't overlap
        # *without* inflating them by geom_margin, so near-misses inside the safety margin can be
        # silently dropped.  The planning model needs those, so the mid-phase is disabled here
        # (only for planning; the simulation model keeps it).
        pm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        root = pm.body("link_base").id
        obj_bodies = {pm.body(o["name"]).id for o in self.objects}
        for g in range(pm.ngeom):                              # safety margin on everything that moves
            b = pm.geom_bodyid[g]
            if (pm.body_rootid[b] == root and pm.body_weldid[b] != 0) or b in obj_bodies:
                pm.geom_margin[g] = PLAN_MARGIN
        return pm

    def _calibrate_gripper(self):
        """Map gripper command -> inner pad gap by simulating the fingers once (scratch data)."""
        m, d = self.model, mujoco.MjData(self.model)
        l, r = m.geom("left_finger_pad_1").id, m.geom("right_finger_pad_1").id
        pad_t = 2 * m.geom_size[l][1]
        ctrls, gaps = np.linspace(GRIPPER_OPEN, GRIPPER_CLOSED, 7), []
        for c in ctrls:
            d.ctrl[self.act_grip] = c
            for _ in range(600):
                mujoco.mj_step(m, d)
            gaps.append(np.linalg.norm(d.geom_xpos[l] - d.geom_xpos[r]) - pad_t)
        self._grip_cal = (np.array(gaps)[::-1], ctrls[::-1])   # increasing gap for np.interp

    # -- basics
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # start in a "ready" pose: gripper horizontal, in front of the cabinet
        ready = np.array([CAB_X0 - STANDOFF, 0.0, SHELF_Z + 0.20])
        self.q_cmd = self.ik(ready, q0=self.q_home, iters=300)
        self.data.qpos[self.qadr] = self.q_cmd
        self.data.ctrl[self.act_arm] = self.q_cmd
        self.data.ctrl[self.act_grip] = GRIPPER_OPEN
        self.held = self.grasp = None
        mujoco.mj_forward(self.model, self.data)
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()

    def grasp_pos(self):
        return self.data.site_xpos[self.site_id].copy()

    def obj_id(self, target):
        """Object index from an index or a name."""
        if isinstance(target, str):
            if target not in self.obj_index:
                raise KeyError(f"no object named {target!r}; have {list(self.obj_index)}")
            return self.obj_index[target]
        return int(target)

    def obj_pos(self, target):
        return self.data.body(self.objects[self.obj_id(target)]["name"]).xpos.copy()

    def cell_y(self, cell):
        return cell_center_y(cell, self.n_cells, self.cell_w)

    def _names(self, pairs):
        pm = self.plan_model
        nm = lambda g: pm.geom(g).name or pm.body(pm.geom_bodyid[g]).name
        return ", ".join(sorted({f"{nm(a)}<->{nm(b)}" for a, b in pairs}))

    # -- inverse kinematics: damped least squares on position + orientation, null-space pull toward
    #    the home posture with the base turned toward the target (see _seed)
    def ik(self, p, R=R_DES, q0=None, iters=60, tol=1e-4):
        d = self.ik_data
        q = (self.q_cmd if q0 is None else q0).copy()
        q_rest = self._seed(p)
        for _ in range(iters):
            d.qpos[:] = self.data.qpos            # keep object poses etc. consistent
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            cur_p = d.site_xpos[self.site_id]
            cur_R = d.site_xmat[self.site_id].reshape(3, 3)
            e_p = p - cur_p
            e_r = 0.5 * sum(np.cross(cur_R[:, i], R[:, i]) for i in range(3))
            if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < tol:
                break
            jp = np.zeros((3, self.model.nv)); jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, d, jp, jr, self.site_id)
            J = np.vstack([jp, jr])[:, self.dadr]
            err = np.concatenate([e_p, 0.5 * e_r])
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, err)
            null = (np.eye(7) - J.T @ np.linalg.solve(JJt, J)) @ (0.1 * (q_rest - q))
            q = np.clip(q + dq + null, self.jrange[:, 0], self.jrange[:, 1])
        return q

    def _seed(self, p):
        """IK seed for a hand position p: the home posture with the base turned toward p.
        Seeding from plain home can land on a contorted branch for targets far to the side,
        from which straight Cartesian moves soon run into joint limits."""
        q = self.q_home.copy()
        q[0] = np.arctan2(p[1], p[0])
        return q

    def _pose_error(self, q, p, R):
        d = self.ik_data
        d.qpos[:] = self.data.qpos; d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        cur_R = d.site_xmat[self.site_id].reshape(3, 3)
        return (np.linalg.norm(d.site_xpos[self.site_id] - p),
                np.linalg.norm(0.5 * sum(np.cross(cur_R[:, i], R[:, i]) for i in range(3))))

    # -- joint-space planning with RRT*
    def ik_goal(self, p, R=R_DES, tol=2e-3, n_seeds=8):
        """IK for a planning goal.  Solves from several seeds (current command, home, random
        perturbations of home) and returns the collision-free solution closest to the current
        configuration -- a different IK branch (e.g. a flipped wrist) would make the planned
        motion needlessly long."""
        rng = np.random.default_rng(self.seed)
        q_cur = self.data.qpos[self.qadr].copy()
        seeds = [self.q_cmd, self._seed(p), self.q_home] + \
                [self.q_home + rng.normal(scale=0.4, size=7) for _ in range(n_seeds)]
        self._sync_checker(PLAN_MARGIN)
        best, best_d = None, np.inf
        for q0 in seeds:
            q = self.ik(p, R, q0=np.clip(q0, self.jrange[:, 0], self.jrange[:, 1]), iters=300, tol=1e-5)
            if self._pose_error(q, p, R)[0] < tol and self.checker.is_free(q):
                dist = np.linalg.norm(q - q_cur)
                if dist < best_d:
                    best, best_d = q, dist
        if best is None:
            raise RuntimeError(f"no collision-free IK solution for p={np.round(p, 3)}")
        return best

    def _sync_checker(self, margin):
        self.checker.margin = margin
        self.checker.sync(self.data.qpos)
        if self.held is not None:
            self.checker.attach([self.obj_geom[self.held]], self.model, self.data)
        else:
            self.checker.detach()

    def plan_joint(self, q_goal, max_time=20.0):
        """RRT* from the current arm configuration to q_goal; returns a dense time-parameterised
        trajectory (one configuration per sim step)."""
        self._sync_checker(PLAN_MARGIN)
        q_start = self.data.qpos[self.qadr].copy()
        ck = self.checker
        ck.n_checks = 0
        edge = lambda a, b: ck.edge_free(a, b, res=0.03)
        planner = RRTStar(ck.is_free, edge, self.jrange[:, 0], self.jrange[:, 1],
                          max_time=max_time, seed=self.seed)
        raw = planner.plan(q_start, q_goal)
        if raw is None:
            raise RuntimeError(f"RRT* found no path ({planner.stats})")
        short = shortcut(raw, edge, seed=self.seed)
        smoothed = smooth(short, ck)
        traj = time_parameterize(smoothed, PLAN_VMAX, PLAN_AMAX, self.model.opt.timestep)
        self.last_plan = dict(planner.stats, raw=raw, shortcut=short, traj=traj,
                              raw_len=path_length(raw), short_len=path_length(short),
                              n_checks=ck.n_checks, duration=len(traj) * self.model.opt.timestep)
        if self.verbose:
            s = self.last_plan
            how = "direct line" if s["direct"] else \
                f'{s["iters"]} iters, {s["nodes"]} nodes, first sol. @ iter {s["first_iter"]}'
            print(f'   RRT*: {how} | path {s["raw_len"]:.2f} -> {s["short_len"]:.2f} rad '
                  f'| {s["time"]:.2f}s plan, {s["duration"]:.2f}s motion')
        return traj

    def execute(self, traj, settle=60):
        """Track a dense joint trajectory with the position servos.  The servos are PD
        (torque = kp (u - q) - kv qdot) plus joint damping b, which lags behind a moving target by
        (kv + b)/kp * qdot; commanding u = q_des + lead * qdot_des cancels that lag so the arm
        actually follows the collision-checked path."""
        dt = self.model.opt.timestep
        qd = np.gradient(traj, dt, axis=0) if len(traj) > 1 else np.zeros_like(traj)
        max_err = 0.0
        for q, v in zip(traj, qd):
            self.q_cmd = q
            self.data.ctrl[self.act_arm] = q + self.servo_lead * v
            self.step()
            max_err = max(max_err, float(np.max(np.abs(self.data.qpos[self.qadr] - q))))
        self.data.ctrl[self.act_arm] = self.q_cmd
        self.wait(settle)
        if max_err > self.max_track_err:
            print(f"   WARNING: arm deviated {max_err:.3f} rad from the planned trajectory "
                  f"(the collision check assumed it would follow it); consider lowering PLAN_VMAX/AMAX")
        return max_err

    def move_joint(self, q_goal, **kw):
        self.execute(self.plan_joint(q_goal, **kw))

    def move_to(self, p, R=R_DES, **kw):
        """Collision-free move of the grasp site to pose (p, R): IK for the goal, then RRT*."""
        if not self.use_planner:
            return self.move_cartesian([p], R, check=False)
        self.move_joint(self.ik_goal(p, R), **kw)

    # -- Cartesian moves (fixed hand orientation), collision-checked before execution
    def _ik_chain(self, q0, p0, points, R, step=0.005):
        """Joint path for the polyline p0 -> points (orientation R) by sequential IK from q0."""
        pts = densify([np.asarray(p0, float)] + [np.asarray(p, float) for p in points], step)
        q, qs = np.asarray(q0, float).copy(), [np.asarray(q0, float).copy()]
        for p in pts[1:]:
            q = self.ik(p, R, q0=q, iters=60, tol=2e-4)
            ep, er = self._pose_error(q, p, R)
            if ep > 2e-3 or er > 2e-2:
                raise RuntimeError(f"Cartesian path leaves the reachable workspace at {np.round(p, 3)}")
            qs.append(q)
        return qs

    def cartesian_path(self, points, R=R_DES):
        """Joint path from the current configuration along current-grasp-point -> points."""
        return self._ik_chain(self.data.qpos[self.qadr], self.grasp_pos(), points, R)

    def _check_path(self, path, ignore=(), margin=CELL_MARGIN, what="move"):
        self._sync_checker(margin)
        with self.checker.ignoring(ignore):
            for q in path:
                hits = self.checker.collisions(q)
                if hits:
                    raise RuntimeError(f"{what} blocked: {self._names(hits)}")

    def approach(self, waypoints, R, ignores):
        """Enter the cabinet along the straight polyline waypoints[0] (stand-off) -> ... -> waypoints[-1].

        The joint path is built *backwards*: IK at the final in-cell pose (seeded from home, the
        same posture find_grasp verified), then sequential IK back out to the stand-off.  RRT* then
        plans to that stand-off configuration, so the arm always arrives in a posture from which the
        straight slide-in is reachable.  ignores[k] = geoms allowed to touch on segment k."""
        wps = [np.asarray(w, float) for w in waypoints]
        q = self.ik(wps[-1], R, q0=self._seed(wps[-1]), iters=300, tol=1e-6)
        segs = []
        for k in reversed(range(len(wps) - 1)):
            seg = self._ik_chain(q, wps[k + 1], [wps[k]], R)
            self._check_path(seg, ignores[k], what=f"approach to {np.round(wps[k + 1], 3)}")
            segs.append(seg[::-1]); q = seg[-1]
        path = [q for s in segs[::-1] for q in s]
        if self.use_planner:
            self.move_joint(path[0])                         # RRT* to the stand-off configuration
        else:
            q_now = self.data.qpos[self.qadr].copy()
            self.execute(time_parameterize([q_now, path[0]], PLAN_VMAX, PLAN_AMAX, self.model.opt.timestep))
        self.execute(time_parameterize(path, CELL_VMAX, CELL_AMAX, self.model.opt.timestep), settle=40)

    def move_cartesian(self, points, R=R_DES, ignore=(), check=True, margin=CELL_MARGIN):
        """Straight-line Cartesian move(s); every IK waypoint is collision-checked first
        (contacts with geoms in `ignore` are allowed, e.g. the object being grasped)."""
        path = self.cartesian_path(points, R)
        if check:
            self._check_path(path[1:], ignore, margin, what=f"Cartesian move to {np.round(points[-1], 3)}")
        traj = time_parameterize(path, CELL_VMAX, CELL_AMAX, self.model.opt.timestep)
        self.execute(traj, settle=40)

    def move_linear(self, p_target, steps=300, R=R_DES):
        """Original unchecked straight-line move of the grasp point (kept for compatibility)."""
        p0 = self.grasp_pos()
        for k in range(1, steps + 1):
            a = 0.5 - 0.5 * np.cos(np.pi * k / steps)
            self.q_cmd = self.ik(p0 + a * (np.asarray(p_target) - p0), R)
            self.data.ctrl[self.act_arm] = self.q_cmd
            self.step()

    def wait(self, steps=200):
        self.step(steps)

    # -- gripper
    def set_gripper(self, open_):
        self.data.ctrl[self.act_grip] = GRIPPER_OPEN if open_ else GRIPPER_CLOSED

    def set_gripper_width(self, width):
        """Open the fingers to an inner pad gap of `width` metres (keeps the hand narrow)."""
        gaps, ctrls = self._grip_cal
        self.data.ctrl[self.act_grip] = float(np.interp(width, gaps, ctrls))

    # -- grasp selection
    def _pose_free(self, p, R, ignore=(), blockers=None):
        """IK + collision check of one hand pose; tallies what blocked it into `blockers`."""
        q = self.ik(p, R, q0=self._seed(p), iters=300, tol=1e-6)
        ep, er = self._pose_error(q, p, R)
        if ep > 1e-3 or er > 1e-2:
            if blockers is not None:
                blockers["out of reach"] = blockers.get("out of reach", 0) + 1
            return None
        with self.checker.ignoring(ignore):
            hits = self.checker.collisions(q)
        if hits and blockers is not None:
            pm, role = self.plan_model, self.checker.role
            for a, b in hits:
                g = b if role[a] != 0 else a                  # the static / other-object side
                name = pm.geom(g).name or pm.body(pm.geom_bodyid[g]).name
                name = name[:-2] if name.endswith("_g") else name
                blockers[name] = blockers.get(name, 0) + 1
        return None if hits else q

    def _lines_free(self, q0, p0, points, R, ignore, blockers):
        """Is the straight polyline p0 -> points (from configuration q0) reachable and free?"""
        try:
            path = self._ik_chain(q0, p0, points, R, step=0.01)
        except RuntimeError:
            blockers["out of reach"] = blockers.get("out of reach", 0) + 1
            return False
        with self.checker.ignoring(ignore):
            for q in path[1:]:
                hits = self.checker.collisions(q)
                if hits:
                    pm, role = self.plan_model, self.checker.role
                    for a, b in hits:
                        gg = b if role[a] != 0 else a
                        name = pm.geom(gg).name or pm.body(pm.geom_bodyid[gg]).name
                        name = name[:-2] if name.endswith("_g") else name
                        blockers[name] = blockers.get(name, 0) + 1
                    return False
        return True

    def _carry_points(self, route, start, goal):
        """Constant-height carry polyline from `start` (lifted at the pick spot) to `goal` (above
        the place spot): 'direct' = straight across the cell, 'around' = back out, slide sideways
        in front of the cabinet, slide back in."""
        if route == "direct":
            return [goal]
        return [self._entrance(start), self._entrance(goal), goal]

    def _low_carry_route(self, i, R, q_g, gp, pp, up, same_cell, blockers):
        """Find a carry route that keeps the object `up` above the shelf.  Checked with the object
        virtually attached to the hand, so the carried object's own sweep is checked as well."""
        g = self.obj_geom[i]
        try:
            q_up = self._ik_chain(q_g, gp, [gp + up], R, step=0.01)[-1]
        except RuntimeError:
            return None
        R_obj = self.data.geom_xmat[g].reshape(3, 3)
        self.checker.attach_relative([(g, R.T @ (self.obj_pos(i) - gp), R.T @ R_obj)])
        try:
            for route in (("direct", "around") if same_cell else ("around",)):
                if self._lines_free(q_up, gp + up, self._carry_points(route, gp + up, pp + up),
                                    R, (), blockers):
                    return route
            return None
        finally:
            self.checker.detach()

    def find_grasp(self, target, place=None, lift=LIFT, low_carry=True):
        """Choose hand pitch and grasp height for `target` so that the grasp pose, the pre-grasp
        pose and (if given) the matching place pose are collision-free -- and so are the full
        straight in/out paths of both (the knuckles are the widest part of the hand and sweep past
        neighbours that the final pose doesn't touch).  Checked with fully open fingers, which is
        conservative (the real hand is pre-shaped / closed and narrower).  Prefers a level hand and
        a grasp at the object's mid-height.

        place = (x, y, cell) of the object centre at the place spot.  With a place spot the grasp
        also gets a low-carry route that keeps the object `lift` above the shelf all the way
        (low_carry=False instead validates the RRT* variant: back out, transit, slide in)."""
        i = self.obj_id(target)
        o, p = self.objects[i], self.obj_pos(i)
        g = self.obj_geom[i]
        z_c = p[2]
        zs = np.arange(SHELF_Z + PAD_HALF_H, SHELF_Z + o["height"] - PAD_HALF_H + 1e-9, 0.004)
        zs = sorted(zs, key=lambda z: abs(z - z_c)) or [z_c]
        back = np.array([o["size"] + APPROACH_DIST, 0, 0])
        self._sync_checker(CELL_MARGIN)
        at_pick, at_place, at_carry = {}, {}, {}             # what blocked the candidates, and how often
        stage = 0                                            # furthest check any candidate got past
        for pitch in PITCHES_DEG:
            R = R_pitch(pitch)
            carry_tries = 0                                  # the carry check is the expensive one
            for z in zs:
                gp = np.array([p[0], p[1], z])
                q_g = self._pose_free(gp, R, [g], at_pick)            # cheap end-pose tests first
                if q_g is None or self._pose_free(gp - back, R, (), at_pick) is None:
                    continue
                if place is not None:
                    pp = np.array([place[0], place[1], SHELF_Z + o["height"] / 2 + 0.004 + (z - z_c)])
                    q_p = self._pose_free(pp, R, [g], at_place)
                    if q_p is None or self._pose_free(pp - back, R, [g], at_place) is None:
                        continue
                # full paths, built outward from the inner pose exactly like approach() does
                up, ent = np.array([0, 0, lift]), self._entrance
                if not (self._lines_free(q_g, gp, [gp - back, ent(gp - back)], R, [g], at_pick) and
                        self._lines_free(q_g, gp, [gp + up], R, [g], at_pick)):
                    continue
                if place is None or not low_carry:           # backs out after lifting
                    if not self._lines_free(q_g, gp, [gp + up, ent(gp + up)], R, [g], at_pick):
                        continue
                    if place is None:
                        return dict(pitch=pitch, R=R, z_rel=z - z_c, pick=gp, route=None)
                stage = max(stage, 1)
                if not (self._lines_free(q_p, pp, [pp + up], R, [g], at_place) and
                        self._lines_free(q_p, pp, [pp - back, ent(pp - back)], R, [g], at_place)):
                    continue
                if not low_carry:                            # RRT* transit, then slide in above the spot
                    if self._lines_free(q_p, pp, [pp + up, ent(pp + up)], R, [g], at_place):
                        return dict(pitch=pitch, R=R, z_rel=z - z_c, pick=gp, route=None)
                    continue
                stage = max(stage, 2)
                if carry_tries == 3:                         # a blocked carry rarely depends on z
                    continue
                carry_tries += 1
                pp_level = np.array([pp[0], pp[1], gp[2]])      # carry stays level at gp + lift
                route = self._low_carry_route(i, R, q_g, gp, pp_level, up, place[2] == o["cell"], at_carry)
                if route is not None:
                    return dict(pitch=pitch, R=R, z_rel=z - z_c, pick=gp, route=route)
        top = lambda b: ", ".join(k for k, _ in sorted(b.items(), key=lambda kv: -kv[1])[:3])
        msg = f"no collision-free grasp for {o['name']}"
        if o["depth"] > MAX_REACH_DEPTH:
            msg += f": depth {o['depth']:.2f} m is beyond the ~{MAX_REACH_DEPTH} m reach"
        else:
            where, blk = [("at the pick location", at_pick), ("at the place location", at_place),
                          (f"while carrying the object {lift*100:.1f} cm above the shelf", at_carry)][stage]
            hit = top({k: v for k, v in blk.items() if k != "out of reach"})
            msg += (f": the hand would hit {hit} {where}" if hit else
                    f": the path leaves the arm's reachable workspace {where}")
            if stage == 2:
                msg += " -- use a different place spot, or allow_high_transit=True to carry it over"
        raise RuntimeError(msg + " (the open hand is ~15 cm wide and the wrist hangs below it)")

    def check_place_location(self, target, cell, depth, dy):
        """Raise if the target object would not fit at (cell, depth, dy)."""
        i = self.obj_id(target)
        o = self.objects[i]
        r = footprint_radius(o)
        x, y = CAB_X0 + depth, self.cell_y(cell) + dy
        if not 0 <= cell < self.n_cells:
            raise ValueError(f"cell {cell} does not exist")
        if abs(dy) + r > self.cell_w / 2 - WALL / 2 or not r < depth < CAB_DEPTH - r:
            raise ValueError(f"{o['name']} does not fit inside cell {cell} at depth {depth}, dy {dy}")
        for j, oj in enumerate(self.objects):
            if j != i:
                pj = self.obj_pos(j)
                if np.hypot(x - pj[0], y - pj[1]) < r + footprint_radius(oj) + 0.005:
                    raise ValueError(f"place location is occupied by {oj['name']}")
        for ob in self.obstacles:
            (px, py, pz), (hx, hy, hz) = ob["pos"], ob["size"]
            if abs(x - px) < hx + r and abs(y - py) < hy + r and pz - hz < SHELF_Z + o["height"]:
                raise ValueError(f"place location overlaps {ob['name']}")
        return x, y

    # -- high-level skills (approach along cabinet +x with fingers opening along y)
    @staticmethod
    def _entrance(p):
        """Stand-off point in front of the cabinet on the same lane / height as p."""
        return np.array([CAB_X0 - STANDOFF, p[1], p[2]])

    def pick(self, target, grasp=None, lift=LIFT, retreat=True):
        """Grasp `target` and lift it `lift` off the shelf; with retreat=True also back out of the cell."""
        i = self.obj_id(target)
        o, g = self.objects[i], self.obj_geom[i]
        grasp = grasp or self.find_grasp(i, lift=lift)
        R, gp = grasp["R"], grasp["pick"]
        pre = gp - np.array([o["size"] + APPROACH_DIST, 0, 0])
        if self.verbose:
            print(f"   grasp {o['name']}: pitch {grasp['pitch']} deg, "
                  f"{(gp[2] - SHELF_Z) * 100:.1f} cm above the shelf")
        self.set_gripper_width(2 * o["size"] + 2 * GRIP_CLEARANCE)   # pre-shape during the transit
        # RRT* transit to the stand-off, then slide in and put the fingers around the object
        self.approach([self._entrance(pre), pre, gp], R, ignores=[(), [g]])
        self.wait(100)
        self.set_gripper(False); self.wait(400)          # close
        self.held, self.grasp = i, grasp
        lifted = gp + np.array([0, 0, lift])
        self.move_cartesian([lifted], R, ignore=[g])     # lift off the shelf by `lift`
        if retreat:
            self.move_cartesian([self._entrance(lifted)], R)   # back out, carrying (held object checked)

    def place(self, cell=0, depth=0.17, dy=0.0, lift=LIFT, route=None):
        """Put the held object down at (cell, depth, dy).
        route=None        : RRT* transit to the cell (height not controlled), then slide in
        route='direct'    : carry straight across the cell, `lift` above the shelf
        route='around'    : back out / sideways / back in, `lift` above the shelf"""
        if self.held is None:
            raise RuntimeError("place() called with nothing in the gripper")
        i = self.held
        o, g, R = self.objects[i], self.obj_geom[i], self.grasp["R"]
        x, y = self.check_place_location(i, cell, depth, dy)
        rel = self.obj_pos(i) - self.grasp_pos()          # object offset in the hand (incl. any slip)
        put = np.array([x, y, SHELF_Z + o["height"] / 2 + 0.004]) - rel
        above = put + np.array([0, 0, lift])
        if route is None:
            # RRT* transit (carrying), slide in above the spot, lower onto the shelf
            self.approach([self._entrance(above), above, put], R, ignores=[(), [g]])
        else:
            # low carry, level at the current (lifted) height; every waypoint is collision-checked
            # with the held object attached
            start = self.grasp_pos()
            self.move_cartesian(self._carry_points(route, start, np.array([*above[:2], start[2]])), R)
            rel = self.obj_pos(i) - self.grasp_pos()      # re-measure: corrects slip during the carry
            put = np.array([x, y, SHELF_Z + o["height"] / 2 + 0.004]) - rel
            self.move_cartesian([np.array([*put[:2], self.grasp_pos()[2]]), put], R, ignore=[g])
        self.set_gripper_width(2 * o["size"] + 2 * GRIP_CLEARANCE); self.wait(300)   # release
        self.held = self.grasp = None
        back = put - np.array([o["size"] + APPROACH_DIST, 0, 0])
        self.move_cartesian([back], R, ignore=[g])       # pull the fingers off the object
        self.move_cartesian([self._entrance(back)], R)   # back out of the cell

    def pick_and_place(self, target, cell=0, depth=0.17, dy=0.0, lift=LIFT, allow_high_transit=False):
        """Pick `target` (index or name), lift it only `lift` off the shelf (default 1.5 cm), carry it
        at that height and put it down at (cell, depth, dy).  Within one cell the object is slid
        straight across; otherwise it is carried out, sideways in front of the cabinet and back in.
        Everything (grasp, lift, carry incl. the object's own sweep, release, retreat) is validated
        before the arm moves.  If no low route exists (e.g. an obstacle in front of the cabinet) it
        raises, unless allow_high_transit=True, which falls back to an RRT* transit (height free)."""
        if not MIN_LIFT <= lift <= MAX_LIFT:
            raise ValueError(f"lift must be within {MIN_LIFT*100:.1f}-{MAX_LIFT*100:.0f} cm")
        i = self.obj_id(target)
        x, y = self.check_place_location(i, cell, depth, dy)
        try:
            grasp = self.find_grasp(i, place=(x, y, cell), lift=lift)
        except RuntimeError as err:
            if not allow_high_transit or "while carrying" not in str(err):
                raise
            grasp = self.find_grasp(i, place=(x, y, cell), lift=lift, low_carry=False)
        if self.verbose:
            how = {"direct": "straight across the cell", "around": "out, sideways and back in",
                   None: "NOT kept low: no low route, using an RRT* transit"}[grasp["route"]]
            print(f"   carry {lift*100:.1f} cm above the shelf, {how}" if grasp["route"] else f"   {how}")
        self.pick(i, grasp, lift=lift, retreat=grasp["route"] is None)
        self.place(cell, depth, dy, lift=lift, route=grasp["route"])
        return self.obj_pos(i)

    # -- viewer / rendering
    def launch_viewer(self):
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.lookat[:] = [0.45, 0, 0.3]
        self.viewer.cam.distance = 2.0
        self.viewer.cam.azimuth = 30
        self.viewer.cam.elevation = -25

    def render(self, path="preview.png", w=960, h=720, lookat=(0.45, 0, 0.3), distance=2.0,
               azimuth=30, elevation=-25):
        r = mujoco.Renderer(self.model, h, w)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat; cam.distance = distance; cam.azimuth = azimuth; cam.elevation = elevation
        r.update_scene(self.data, cam)
        img = r.render()
        from PIL import Image
        Image.fromarray(img).save(path)


# ----------------------------------------------------------------------------- demo scene
# Edit these to change the cabinet, the objects, the obstacles and the task.
SCENE = dict(
    n_cells=1,
    cell_w=0.30,
    objects=[
        dict(name="target", shape="cylinder", color="green", size=0.020, height=0.08, depth=0.18, dy=0.055),
        dict(name="blue_box", shape="box", color="blue", size=0.020, height=0.05, depth=0.20, dy=-0.07),
        dict(name="red_box", shape="box", color="red", size=0.015, height=0.04, depth=0.27, dy=0.06),
    ],
    obstacles=[
        dict(name="shelf_block", cell=0, depth=0.15, dy=-0.115, size=[0.02, 0.02, 0.05]),  # on the shelf
        dict(name="post", pos=[0.42, -0.32, 0.30], size=[0.04, 0.04, 0.30]),              # on the floor
    ],
)
TASK = dict(target="target", cell=0, depth=0.08, dy=-0.05, lift=0.015)   # lift: 1-2 cm


if __name__ == "__main__":
    use_planner = "--no-planner" not in sys.argv
    lift = float(sys.argv[sys.argv.index("--lift") + 1]) if "--lift" in sys.argv else LIFT
    if "--random" in sys.argv:
        n = int(sys.argv[sys.argv.index("--random") + 1])
        env = XArm7CabinetEnv(n_cells=1, n_objects=n, seed=1, use_planner=use_planner)
        task = None
    else:
        env = XArm7CabinetEnv(**SCENE, use_planner=use_planner)
        task = dict(TASK, lift=lift) if "--lift" in sys.argv else TASK
    for o in env.objects:
        print(f"{o['name']:>9}: {o['color']} {o['shape']} {2*o['size']*100:.1f} cm wide, "
              f"{o['height']*100:.1f} cm tall, depth {o['depth']:.3f}, dy {o['dy']:+.3f}")
    if "--headless" not in sys.argv:
        env.launch_viewer()

    if task is None:                                    # random scene: try objects until one works
        for i in range(len(env.objects)):
            for depth, dy in itertools.product([0.08, 0.12, 0.16], [0.05, -0.05, 0.0]):
                try:
                    x, y = env.check_place_location(i, 0, depth, dy)
                    env.find_grasp(i, place=(x, y, 0), lift=lift)
                    task = dict(target=env.objects[i]["name"], cell=0, depth=depth, dy=dy, lift=lift)
                    break
                except (ValueError, RuntimeError):
                    continue
            if task:
                break
        if task is None:
            sys.exit("no object in this random scene can be moved to a free spot -- try another seed")

    print(f"pick {task['target']} -> cell {task['cell']}, depth {task['depth']}, dy {task['dy']}")
    final = env.pick_and_place(**task)
    goal = np.array([CAB_X0 + task["depth"], env.cell_y(task["cell"]) + task["dy"]])
    print("   final pos:", np.round(final, 3), f"| xy error {np.linalg.norm(final[:2] - goal)*100:.1f} cm")
    print("done")
    if env.viewer is not None:
        while env.viewer.is_running():
            env.step(10)
