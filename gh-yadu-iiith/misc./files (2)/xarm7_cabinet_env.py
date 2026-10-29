"""
MuJoCo environment: UFactory xArm7 (MuJoCo Menagerie, stock gripper) + open-front cabinet + pick-and-place.

Main task (python xarm7_cabinet_env.py): "clear the path, then move the target"
--------------------------------------------------------------------------------
  * one cabinet cell whose shelf carries a 6 x 6 grid (row 0 = front, column 0 = -y); a grid cell
    holds at most one object
  * 1 target (green cylinder) and 3 obstacles (blocks) standing in the target's path, the rest empty
  * the arm first picks each obstacle that is in the way and puts it into a free grid cell where it
    blocks nothing, then picks the target and puts it into its goal cell
  * arm motion outside the cabinet: RRT-Connect (rrt_connect.py; RRT* in rrt_star.py is selectable),
    then shortcut / smoothing / time scaling.  Inside the cabinet: straight, collision-checked
    Cartesian moves; objects are carried only ~1.5 cm above the shelf

Setup
-----
    pip install -U "mujoco>=3.8" numpy pillow       # older MuJoCo versions misbehave (see MIN_MUJOCO)
    git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie
    # or set the XARM7_XML environment variable to the path of xarm7.xml
    # rrt_connect.py and rrt_star.py must sit next to this file

Run
---
    python xarm7_cabinet_env.py                      # viewer + the scripted scene (GRID_SCENE / GRID_TASK)
    python xarm7_cabinet_env.py --headless           # no window
    python xarm7_cabinet_env.py --random-grid 3      # random layout (first solvable seed >= 3)
    python xarm7_cabinet_env.py --planner rrt_star   # rrt_connect (default) | rrt_star
    python xarm7_cabinet_env.py --lift 0.02          # carry height above the shelf (m)
    python xarm7_cabinet_env.py --no-planner         # straight lines instead of a sampling planner

Objects: dict(name, label, shape 'box'|'cylinder', size = half-width / radius, height, color,
grid=(row, col)); see GRID_SCENE.  Obstacles inside the cabinet are just more objects; static
obstacles can be added with obstacles=[dict(pos=[x, y, z], size=[hx, hy, hz])].

Limits (measured): objects 1-7.5 cm wide; the wrist hangs below the hand, so low objects cannot be
grasped deep in the cabinet -- 6 cm tall blocks can be reached in every grid row.
"""
import os
import sys
import itertools
from contextlib import contextmanager
import numpy as np
import mujoco

MIN_MUJOCO = (3, 8)   # older versions: 3.2 crashes, 3.3-3.6 silently drop the object, 3.7 fails IK
if tuple(int(v) for v in mujoco.__version__.split(".")[:2]) < MIN_MUJOCO \
        and not os.environ.get("XARM7_SKIP_VERSION_CHECK"):
    raise ImportError(f"MuJoCo {mujoco.__version__} is too old for this environment (tested on 3.8-3.16).\n"
                      f"Upgrade with:  pip install -U \"mujoco>=3.8\"")

_BLOCK_CACHES = {}     # measured neighbour-blocking patterns, shared by identical scenes (they take ~20 s)

from rrt_connect import RRTConnect
from rrt_star import (MujocoCollisionChecker, RRTStar, shortcut, smooth,
                      time_parameterize, path_length, densify)

# ----------------------------------------------------------------------------- paths
HERE = os.path.dirname(os.path.abspath(__file__))
XARM7_XML = os.environ.get(
    "XARM7_XML", os.path.join(HERE, "mujoco_menagerie", "ufactory_xarm7", "xarm7.xml"))

# ----------------------------------------------------------------------------- scene parameters
CAB_X0 = 0.62                # cabinet front face (x)
CAB_DEPTH = 0.27             # shallow enough that every grid row is within the arm's ~25 cm reach
CELL_W = 0.50                # default cell width (y); wide enough that the outer grid columns keep a
                             # hand-width (~15 cm) of clearance from the side walls

# 6 x 6 grid on the shelf of the (single) cell.  Row 0 is at the front, column 0 at -y.
# 6 cm columns x 4 cm rows hold one 3 cm object each; every cell was verified to be pickable.
GRID = dict(rows=6, cols=6, pitch_dy=0.06, pitch_depth=0.04, depth0=0.03)
FOOTPRINT_GAP = 0.004        # minimum clearance between object footprints
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
GRASP_OFFSETS = (0.0, 0.005, 0.010)   # grip this far behind the object centre if needed: closing the
                             # fingers slides the pads ~1.3 cm forward, into a neighbour right behind

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
PLAN_SLACK = 0.002           # extra margin when validating a task plan on a what-if scene, so
                             # objects landing 1-2 mm off their cell centre can't invalidate it
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


def grid_cell_center(cell, grid=GRID):
    """(depth, dy) of the centre of grid cell (row, col)."""
    r, c = cell
    return grid["depth0"] + r * grid["pitch_depth"], (c - (grid["cols"] - 1) / 2.0) * grid["pitch_dy"]


def grid_cell_at(depth, dy, grid=GRID):
    """Grid cell (row, col) containing the point (depth, dy), or None if outside the grid."""
    r = int(np.floor((depth - grid["depth0"]) / grid["pitch_depth"] + 0.5))
    c = int(np.floor(dy / grid["pitch_dy"] + grid["cols"] / 2.0))
    return (r, c) if 0 <= r < grid["rows"] and 0 <= c < grid["cols"] else None


def footprints_overlap(xa, ya, sa, xb, yb, sb, gap=FOOTPRINT_GAP):
    """Axis-aligned square footprints (half-size s) closer than `gap`?  Objects stand upright and
    cubes are spawned axis-aligned, so this is tighter than circumscribed circles and lets 3 cm
    cubes sit in adjacent 4 cm grid rows."""
    return abs(xa - xb) < sa + sb + gap and abs(ya - yb) < sa + sb + gap


class GraspError(RuntimeError):
    """No feasible grasp / carry / place.  `stage` = 0 pick, 1 place, 2 carry; `blockers` maps the
    names of what was in the way to how often it blocked a candidate grasp."""

    def __init__(self, msg, stage=0, blockers=None):
        super().__init__(msg)
        self.stage, self.blockers = stage, dict(blockers or {})


# ----------------------------------------------------------------------------- scene description
def normalize_objects(objects, grid=None):
    """Fill in defaults so every object dict has all keys.  With a grid, an object may be given as
    grid=(row, col) instead of depth / dy."""
    out = []
    for i, o in enumerate(objects or []):
        o = dict(o)
        o.setdefault("name", f"obj{i}")
        o.setdefault("label", o["name"][:1].upper())
        if "grid" in o:
            if grid is None:
                raise ValueError(f"{o['name']}: grid=... given but the environment has no grid")
            o["depth"], o["dy"] = grid_cell_center(o["grid"], grid)
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


def validate_scene(objects, obstacles, n_cells, cell_w, grid=None):
    """Raise ValueError for objects the gripper can't hold or that don't fit / overlap, and -- with
    a grid -- unless every object sits in its own grid cell."""
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
        if a["cell"] == b["cell"] and footprints_overlap(a["depth"], a["dy"], a["size"],
                                                         b["depth"], b["dy"], b["size"]):
            raise ValueError(f"objects {a['name']} and {b['name']} overlap")
    if grid is not None:
        if n_cells != 1:
            raise ValueError("the grid is defined for a single-cell cabinet (n_cells=1)")
        cell_room = min(grid["pitch_dy"], grid["pitch_depth"]) - FOOTPRINT_GAP
        seen = {}
        for o in objects:
            gc = grid_cell_at(o["depth"], o["dy"], grid)
            if gc is None:
                raise ValueError(f"{o['name']} is outside the {grid['rows']}x{grid['cols']} grid")
            if 2 * o["size"] > cell_room:
                raise ValueError(f"{o['name']} ({2*o['size']*100:.1f} cm) is too big for a grid cell "
                                 f"(max {cell_room*100:.1f} cm)")
            if gc in seen:
                raise ValueError(f"grid cell {gc} holds both {seen[gc]} and {o['name']} "
                                 "(one object per cell)")
            seen[gc] = o["name"]
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
def build_scene_xml(n_cells=1, objects=None, obstacles=None, cell_w=CELL_W, grid=None):
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

    if grid is not None:                       # grid lines: visual only (contype = conaffinity = 0)
        rows, cols, pd, pdy = grid["rows"], grid["cols"], grid["pitch_depth"], grid["pitch_dy"]
        d_lo, d_hi = grid["depth0"] - pd / 2, grid["depth0"] + (rows - 0.5) * pd
        y_half, zc = cols * pdy / 2, SHELF_Z + 0.0004
        line = 'type="box" rgba="0.12 0.12 0.12 1" contype="0" conaffinity="0"'
        for k in range(cols + 1):
            cab.append(f'<geom name="grid_col{k}" {line} pos="{CAB_X0 + (d_lo + d_hi) / 2} '
                       f'{(k - cols / 2) * pdy} {zc}" size="{(d_hi - d_lo) / 2} 0.0012 0.0004"/>')
        for k in range(rows + 1):
            cab.append(f'<geom name="grid_row{k}" {line} pos="{CAB_X0 + d_lo + k * pd} 0 {zc}" '
                       f'size="0.0012 {y_half} 0.0004"/>')

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
                 xml_path=XARM7_XML, use_planner=True, verbose=True, grid=None,
                 planner="rrt_connect", lift=None):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(
                f"xarm7.xml not found at {xml_path}\n"
                "Run:  git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie\n"
                "or set the XARM7_XML environment variable.")
        self.n_cells, self.cell_w = n_cells, cell_w
        self.use_planner, self.verbose, self.seed = use_planner, verbose, seed
        if planner not in ("rrt_connect", "rrt_star"):
            raise ValueError("planner must be 'rrt_connect' or 'rrt_star'")
        self.planner, self.grid = planner, grid
        self.lift = LIFT if lift is None else lift
        self._virtual = {}                                     # what-if object positions (planning)
        self._planning = False                                 # validating a task plan -> PLAN_SLACK
        self._block_cache = None                               # filled lazily (needs the objects)
        self.obstacles = normalize_obstacles(obstacles, n_cells, cell_w)
        if objects is None:
            objects = random_objects(n_objects, n_cells, cell_w, seed, obstacles=self.obstacles)
        self.objects = normalize_objects(objects, grid)
        validate_scene(self.objects, self.obstacles, n_cells, cell_w, grid)
        self.obj_index = {o["name"]: i for i, o in enumerate(self.objects)}
        if len(self.obj_index) != len(self.objects):
            raise ValueError("object names must be unique")

        # --- scene + attach xArm7 (stock gripper included)
        scene = mujoco.MjSpec.from_string(
            build_scene_xml(n_cells, self.objects, self.obstacles, cell_w, grid))
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
        self.obj_qadr = [m.joint(o["name"] + "_free").qposadr[0] for o in self.objects]
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
        """Map gripper command -> inner pad gap, and pad gap -> finger joint angles, by simulating
        the fingers once (scratch data).  The joint angles let the planner put the fingers of the
        collision model exactly where the real ones will be (pre-shaped, or closed on an object)."""
        m, d = self.model, mujoco.MjData(self.model)
        l, r = m.geom("left_finger_pad_1").id, m.geom("right_finger_pad_1").id
        pad_t = 2 * m.geom_size[l][1]
        hand = set(self._subtree(m, m.body(HAND_ROOT).id))
        self._finger_qadr = np.array([m.jnt_qposadr[j] for j in range(m.njnt) if m.jnt_bodyid[j] in hand])
        ctrls, gaps, qs = np.linspace(GRIPPER_OPEN, GRIPPER_CLOSED, 9), [], []
        for c in ctrls:
            d.ctrl[self.act_grip] = c
            for _ in range(600):
                mujoco.mj_step(m, d)
            gaps.append(np.linalg.norm(d.geom_xpos[l] - d.geom_xpos[r]) - pad_t)
            qs.append(d.qpos[self._finger_qadr].copy())
        order = np.argsort(gaps)                                # increasing gap for np.interp
        self._grip_cal = (np.array(gaps)[order], ctrls[order])
        self._grip_q = np.array(qs)[order]

    def _set_fingers(self, width):
        """Put the collision model's fingers at inner pad gap `width`."""
        gaps = self._grip_cal[0]
        self.checker.d.qpos[self._finger_qadr] = [np.interp(width, gaps, self._grip_q[:, k])
                                                  for k in range(self._grip_q.shape[1])]

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
        i = self.obj_id(target)
        if i in self._virtual:                                 # what-if position during planning
            return self._virtual[i].copy()
        return self.data.body(self.objects[i]["name"]).xpos.copy()

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
        for i, p in self._virtual.items():                    # what-if object positions
            a = self.obj_qadr[i]
            self.checker.d.qpos[a:a + 3] = p
            self.checker.d.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        if self.held is not None:
            self.checker.attach([self.obj_geom[self.held]], self.model, self.data)
        else:
            self.checker.detach()
        self._set_fingers(self._current_width())              # fingers as *commanded*, not as simulated

    def plan_joint(self, q_goal, max_time=20.0):
        """RRT-Connect (or RRT*, see `planner`) from the current arm configuration to q_goal; returns a dense time-parameterised
        trajectory (one configuration per sim step)."""
        self._sync_checker(PLAN_MARGIN)
        q_start = self.data.qpos[self.qadr].copy()
        ck = self.checker
        ck.n_checks = 0
        edge = lambda a, b: ck.edge_free(a, b, res=0.03)
        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        if self.planner == "rrt_connect":
            name, planner = "RRT-Connect", RRTConnect(ck.is_free, edge, lo, hi, max_time=max_time,
                                                       seed=self.seed)
        else:
            name, planner = "RRT*", RRTStar(ck.is_free, edge, lo, hi, max_time=max_time, seed=self.seed)
        raw = planner.plan(q_start, q_goal)
        if raw is None:
            raise RuntimeError(f"{name} found no path ({planner.stats})")
        short = shortcut(raw, edge, seed=self.seed)
        smoothed = smooth(short, ck)
        traj = time_parameterize(smoothed, PLAN_VMAX, PLAN_AMAX, self.model.opt.timestep)
        self.last_plan = dict(planner.stats, raw=raw, shortcut=short, traj=traj,
                              raw_len=path_length(raw), short_len=path_length(short),
                              n_checks=ck.n_checks, duration=len(traj) * self.model.opt.timestep)
        if self.verbose:
            s = self.last_plan
            how = "direct line" if s.get("direct") else f'{s["iters"]} iters, {s["nodes"]} nodes'
            print(f'   {name}: {how} | path {s["raw_len"]:.2f} -> {s["short_len"]:.2f} rad '
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
                    blk = {}
                    self._tally(hits, blk)
                    raise GraspError(f"{what} blocked: {self._names(hits)}", 0, blk)

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
    def _tally(self, hits, blockers):
        """Count what blocked a candidate (the static / other-object side of each contact)."""
        pm, role = self.plan_model, self.checker.role
        for a, b in hits:
            g = b if role[a] != 0 else a
            name = pm.geom(g).name or pm.body(pm.geom_bodyid[g]).name
            name = name[:-2] if name.endswith("_g") else name
            blockers[name] = blockers.get(name, 0) + 1

    def _pose_free(self, p, R, ignore=(), blockers=None, widths=(None,)):
        """IK + collision check of one hand pose, once per finger opening in `widths`
        (None = leave the fingers as they are)."""
        q = self.ik(p, R, q0=self._seed(p), iters=300, tol=1e-6)
        ep, er = self._pose_error(q, p, R)
        if ep > 1e-3 or er > 1e-2:
            if blockers is not None:
                blockers["out of reach"] = blockers.get("out of reach", 0) + 1
            return None
        for w in widths:
            if w is not None:
                self._set_fingers(w)
            with self.checker.ignoring(ignore):
                hits = self.checker.collisions(q)
            if hits:
                if blockers is not None:
                    self._tally(hits, blockers)
                return None
        return q

    def _lines_free(self, q0, p0, points, R, ignore, blockers, width=None):
        """Is the straight polyline p0 -> points (from configuration q0) reachable and free,
        with the fingers at pad gap `width`?"""
        try:
            path = self._ik_chain(q0, p0, points, R, step=0.01)
        except RuntimeError:
            blockers["out of reach"] = blockers.get("out of reach", 0) + 1
            return False
        if width is not None:
            self._set_fingers(width)
        with self.checker.ignoring(ignore):
            for q in path[1:]:
                hits = self.checker.collisions(q)
                if hits:
                    self._tally(hits, blockers)
                    return False
        return True

    def _carry_points(self, route, start, goal):
        """Constant-height carry polyline from `start` (lifted at the pick spot) to `goal` (above
        the place spot): 'direct' = straight across the cell, 'around' = back out, slide sideways
        in front of the cabinet, slide back in."""
        if route == "direct":
            return [goal]
        return [self._entrance(start), self._entrance(goal), goal]

    def _low_carry_route(self, i, R, q_g, gp, pp, up, same_cell, blockers, width):
        """Find a carry route that keeps the object `up` above the shelf.  Checked with the object
        virtually attached to the hand (fingers closed on it), so its own sweep is checked too."""
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
                                    R, (), blockers, width):
                    return route
            return None
        finally:
            self.checker.detach()

    def _pick_ok(self, i, R, gp, blockers, w_open, w_shut, back, up, backs_out):
        """All pick-side checks for one grasp candidate: arrive pre-shaped and close at gp, the
        pre-grasp pose, the slide in, the lift, and (backs_out) the retreat while holding it."""
        g, ent = self.obj_geom[i], self._entrance
        q_g = self._pose_free(gp, R, [g], blockers, (w_open, w_shut))
        if q_g is None or self._pose_free(gp - back, R, (), blockers, (w_open,)) is None:
            return None
        if not (self._lines_free(q_g, gp, [gp - back, ent(gp - back)], R, [g], blockers, w_open) and
                self._lines_free(q_g, gp, [gp + up], R, [g], blockers, w_shut)):
            return None
        if backs_out and not self._lines_free(q_g, gp, [gp + up, ent(gp + up)], R, [g], blockers, w_shut):
            return None
        return q_g

    def _block_key(self):
        return (tuple(sorted((o["shape"], o["size"], o["height"]) for o in self.objects)),
                tuple(sorted(self.grid.items())), self.cell_w, getattr(self, "lift", None), CAB_X0, CAB_DEPTH,
                CELL_MARGIN, PLAN_SLACK)

    def _current_width(self):
        """Inner pad gap the gripper has been commanded to (closed on the object while holding one).
        The collision model must use this, not the simulated finger positions: a width command
        takes ~0.3 s to take effect, and a path checked against the still-wide-open fingers would
        be rejected (or, the other way round, accepted) for the wrong hand."""
        if self.held is not None:
            return self._finger_widths(self.held)[1]
        gaps, ctrls = self._grip_cal
        return float(np.interp(self.data.ctrl[self.act_grip], ctrls[::-1], gaps[::-1]))

    def _finger_widths(self, i):
        o = self.objects[i]
        return (min(2 * o["size"] + 2 * GRIP_CLEARANCE, self._grip_cal[0][-1]),   # pre-shaped
                max(2 * o["size"] - 0.001, self._grip_cal[0][0]))                 # closed on it

    def find_grasp(self, target, place=None, lift=LIFT, low_carry=True):
        """Choose hand pitch, grasp height and grasp offset for `target` so that the whole
        pick -> lift -> carry -> place -> release -> retreat sequence is collision-free.

        Every check uses the finger opening the hand really has at that moment: pre-shaped
        (object width + clearance) while reaching in, closing and releasing; closed on the object
        while lifting, carrying and lowering it.  This matters: the gripper is a linkage and its
        pads slide ~1.3 cm forward as it closes.  The knuckles, the widest part of the hand, are
        checked along the full in / out paths, not just at the end poses.  Prefers a level hand,
        a grasp at the object centre and at mid-height.

        place = (x, y, cell) of the object centre at the place spot.  With a place spot the grasp
        also gets a low-carry route that keeps the object `lift` above the shelf all the way
        (low_carry=False instead validates the RRT transit variant: back out, transit, slide in)."""
        i = self.obj_id(target)
        o, p = self.objects[i], self.obj_pos(i)
        g = self.obj_geom[i]
        z_c = p[2]
        zs = np.arange(SHELF_Z + PAD_HALF_H, SHELF_Z + o["height"] - PAD_HALF_H + 1e-9, 0.004)
        zs = sorted(zs, key=lambda z: abs(z - z_c)) or [z_c]
        back = np.array([o["size"] + APPROACH_DIST, 0, 0])
        w_open, w_shut = self._finger_widths(i)
        up, ent = np.array([0, 0, lift]), self._entrance
        self._sync_checker(CELL_MARGIN + (PLAN_SLACK if self._planning else 0.0))
        at_pick, at_place, at_carry = {}, {}, {}             # what blocked the candidates, and how often
        stage = 0                                            # furthest check any candidate got past
        for pitch in PITCHES_DEG:
            R = R_pitch(pitch)
            carry_tries = 0                                  # the carry check is the expensive one
            for off, z in itertools.product(GRASP_OFFSETS, zs):
                gp = np.array([p[0] - off, p[1], z])
                # cheap end-pose tests first: arrive pre-shaped and close; pre-grasp pose
                # pick side (paths built outward from the inner pose exactly like approach() does)
                q_g = self._pick_ok(i, R, gp, at_pick, w_open, w_shut, back, up,
                                    backs_out=place is None or not low_carry)
                if q_g is None:
                    continue
                if place is None:
                    return dict(pitch=pitch, R=R, z_rel=z - z_c, offset=off, pick=gp, route=None)
                stage = max(stage, 1)
                pp = np.array([place[0] - off, place[1], SHELF_Z + o["height"] / 2 + 0.004 + (z - z_c)])
                q_p = self._pose_free(pp, R, [g], at_place, (w_shut, w_open))   # arrive closed, release
                if q_p is None or self._pose_free(pp - back, R, [g], at_place, (w_open,)) is None:
                    continue
                if not (self._lines_free(q_p, pp, [pp + up], R, [g], at_place, w_shut) and
                        self._lines_free(q_p, pp, [pp - back, ent(pp - back)], R, [g], at_place, w_open)):
                    continue
                if not low_carry:                            # RRT transit, then slide in above the spot
                    if self._lines_free(q_p, pp, [pp + up, ent(pp + up)], R, [g], at_place, w_shut):
                        return dict(pitch=pitch, R=R, z_rel=z - z_c, offset=off, pick=gp, route=None)
                    continue
                stage = max(stage, 2)
                if carry_tries == 3:                         # a blocked carry rarely depends on z
                    continue
                carry_tries += 1
                pp_level = np.array([pp[0], pp[1], gp[2]])   # carry stays level at gp + lift
                route = self._low_carry_route(i, R, q_g, gp, pp_level, up, place[2] == o["cell"],
                                              at_carry, w_shut)
                if route is not None:
                    return dict(pitch=pitch, R=R, z_rel=z - z_c, offset=off, pick=gp, route=route)
        top = lambda b: ", ".join(k for k, _ in sorted(b.items(), key=lambda kv: -kv[1])[:3])
        msg = f"no collision-free grasp for {o['name']}"
        blk = {}
        if p[0] - CAB_X0 > MAX_REACH_DEPTH:
            msg += f": depth {p[0] - CAB_X0:.2f} m is beyond the ~{MAX_REACH_DEPTH} m reach"
        else:
            where, blk = [("at the pick location", at_pick), ("at the place location", at_place),
                          (f"while carrying the object {lift*100:.1f} cm above the shelf", at_carry)][stage]
            hit = top({k: v for k, v in blk.items() if k != "out of reach"})
            msg += (f": the hand would hit {hit} {where}" if hit else
                    f": the path leaves the arm's reachable workspace {where}")
            if stage == 2:
                msg += " -- use a different place spot, or allow_high_transit=True to carry it over"
        raise GraspError(msg + " (the open hand is ~15 cm wide and the wrist hangs below it)", stage, blk)

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
                if footprints_overlap(x, y, o["size"], pj[0], pj[1], oj["size"]):
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

    # -- task planning on the grid: clear the target's path, then move the target
    @contextmanager
    def hypothetical(self, moves):
        """Plan on a what-if scene.  moves = {object index: grid cell, or None to take it out}.
        Only the planning side sees it (obj_pos, the collision checker); nothing is simulated."""
        saved = dict(self._virtual)
        for i, cell in moves.items():
            self._virtual[i] = self._cell_point(i, cell) if cell is not None else np.array([5.0 + i, 5.0, 0.5])
        try:
            yield
        finally:
            self._virtual = saved

    def _cell_point(self, i, cell):
        depth, dy = grid_cell_center(cell, self.grid)
        return np.array([CAB_X0 + depth, dy, SHELF_Z + self.objects[i]["height"] / 2])

    def cell_of(self, target):
        """Grid cell the object is in (None if it is off the grid)."""
        p = self.obj_pos(target)
        return grid_cell_at(p[0] - CAB_X0, p[1] - self.cell_y(0), self.grid)

    def occupancy(self):
        """{grid cell: object index} for every object on the grid."""
        occ = {}
        for i in range(len(self.objects)):
            c = self.cell_of(i)
            if c is not None:
                occ[c] = i
        return occ

    def _move_check(self, i, cell):
        """Raise ValueError / GraspError unless object i can be moved to grid `cell` right now
        (pick, low carry, place and retreat all collision-free)."""
        depth, dy = grid_cell_center(cell, self.grid)
        x, y = self.check_place_location(i, 0, depth, dy)
        return self.find_grasp(i, place=(x, y, 0), lift=self.lift)

    def _object_blockers(self, err, exclude=()):
        """Objects (not walls / shelf) named as being in the way in a GraspError."""
        return [self.obj_index[n] for n in err.blockers if n in self.obj_index and
                self.obj_index[n] not in exclude]

    def _blocking_offsets(self, row):
        """Grid offsets (dr, dc) at which a single neighbour stops the hand from picking an object
        in `row` -- measured with the real collision model (fingers pre-shaped / closed, all grasp
        offsets tried), not guessed.  Reference object: the largest one, in column cols//2 - 1.
        Memoized per row; neighbours 3+ columns away are always clear of the ~15 cm hand."""
        if self._block_cache is None:
            self._block_cache = _BLOCK_CACHES.setdefault(self._block_key(), {})
        if row in self._block_cache:
            return self._block_cache[row]
        blocked = set()
        order = [int(k) for k in np.argsort([-o["size"] for o in self.objects])]
        if len(order) >= 2:
            ref, nb = order[0], order[1]
            c0, rows, cols = self.grid["cols"] // 2 - 1, self.grid["rows"], self.grid["cols"]
            away = {j: None for j in range(len(self.objects)) if j not in (ref, nb)}
            planning, self._planning = self._planning, True
            try:
                with self.hypothetical({**away, ref: (row, c0), nb: None}):
                    g = self.find_grasp(ref)
                    w_open, w_shut = self._finger_widths(ref)
                    back = np.array([self.objects[ref]["size"] + APPROACH_DIST, 0, 0])
                    up, p = np.array([0, 0, self.lift]), self.obj_pos(ref)
                    variants = [np.array([p[0] - off, p[1], g["pick"][2]]) for off in GRASP_OFFSETS]
                    for dr, dc in itertools.product(range(-row, rows - row), range(-2, 3)):
                        if (dr, dc) == (0, 0) or not 0 <= c0 + dc < cols:
                            continue
                        self._virtual[nb] = self._cell_point(nb, (row + dr, c0 + dc))
                        self._sync_checker(CELL_MARGIN + PLAN_SLACK)
                        if all(self._pick_ok(ref, g["R"], gp, {}, w_open, w_shut, back, up, True) is None
                               for gp in variants):
                            blocked.add((dr, dc))
            except (ValueError, RuntimeError):
                pass                                          # row unreachable: nothing to learn
            finally:
                self._planning = planning
        self._block_cache[row] = blocked
        return blocked

    def _corridor(self, cell):
        """Grid cells where another object would stop the hand from reaching `cell`."""
        r, c = cell
        return {(r + dr, c + dc) for dr, dc in self._blocking_offsets(r)
                if 0 <= r + dr < self.grid["rows"] and 0 <= c + dc < self.grid["cols"]}

    def corridor_table(self):
        """Measured blocking pattern per row, as text (X = a neighbour there blocks, O = the object)."""
        out = []
        for r in range(self.grid["rows"]):
            offs = self._blocking_offsets(r)
            lines = [f"row {r}:"]
            for dr in range(self.grid["rows"] - 1 - r, -r - 1, -1):
                lines.append("   " + " ".join("O" if (dr, dc) == (0, 0) else "X" if (dr, dc) in offs else "."
                                              for dc in range(-2, 3)))
            out.append("\n".join(lines))
        return "\n".join(out)

    def _grid_search(self, t, goal, start, banned, keepout, max_moves=6):
        """Breadth-first search over grid states for the fewest cube moves that clear the target.

        Cheap grid-level model: an object can be moved from cell a to free cell b only if nothing
        else is in a's approach corridor (to pick it) or in b's corridor (to put it down) -- the low
        carry leaves through a's lane and enters through b's.  The target can go once nothing is in
        its own corridor, the goal's corridor or the goal.  Cubes are never parked inside that
        zone, so each one moves at most once.  `banned` holds (state, object, cell) moves that
        failed the full collision check; `keepout` holds extra cells found to block the target."""
        from collections import deque
        movable = [j for j in start if j != t]
        zone = self._corridor(start[t]) | self._corridor(goal) | {goal} | set(keepout)
        all_cells = [(r, c) for r in range(self.grid["rows"]) for c in range(self.grid["cols"])]
        lanes = [grid_cell_center(start[t], self.grid)[1], grid_cell_center(goal, self.grid)[1]]
        # preferred parking first (BFS keeps the first plan found among equally short ones):
        # far from the target's lanes, then deep in the cabinet
        all_cells.sort(key=lambda cell: (-round(min(abs(grid_cell_center(cell, self.grid)[1] - l)
                                                     for l in lanes), 3), -cell[0]))
        key = lambda s: tuple(s[j] for j in movable)
        queue, seen = deque([(start, [])]), {key(start)}
        while queue:
            s, path = queue.popleft()
            occ = set(s.values())
            if not (occ - {s[t]}) & zone:
                return path
            if len(path) >= max_moves:
                continue
            for j in movable:
                others = occ - {s[j]}
                if self._corridor(s[j]) & others:              # j itself is boxed in
                    continue
                for cell in all_cells:
                    if cell in occ or cell in zone or self._corridor(cell) & others \
                            or (key(s), j, cell) in banned:
                        continue
                    s2 = dict(s); s2[j] = cell
                    if key(s2) not in seen:
                        seen.add(key(s2))
                        queue.append((s2, path + [(j, cell)]))
        return None

    def plan_rearrangement(self, target, goal, max_replans=20):
        """Plan the moves that clear the target's path and then move the target to grid `goal`.

        Returns [(object index, grid cell), ...]: the cubes to move out of the way (each to a cell
        where it blocks nothing), then the target.  A grid-level breadth-first search proposes the
        shortest sequence; every move is then validated with find_grasp (pick, 1-2 cm carry, place,
        retreat) on a what-if copy of the scene as it will be at that point.  A move that fails is
        banned and the search runs again."""
        t, goal = self.obj_id(target), tuple(goal)
        self._planning = True
        try:
            return self._plan_rearrangement(t, goal, max_replans)
        finally:
            self._planning = False

    def _plan_rearrangement(self, t, goal, max_replans):
        start = {j: self.cell_of(j) for j in range(len(self.objects))}
        if any(c is None for c in start.values()):
            raise RuntimeError("every object must be on the grid to plan a rearrangement")
        with self.hypothetical({j: None for j in start if j != t}):
            try:
                self._move_check(t, goal)
            except (ValueError, RuntimeError) as err:
                raise RuntimeError(f"{self.objects[t]['name']} can't reach {goal} even on an empty "
                                   f"grid: {err}") from None
        banned, keepout = set(), set()
        for attempt in range(max_replans):
            plan = self._grid_search(t, goal, start, banned, keepout)
            if plan is None:
                raise RuntimeError("no sequence of moves clears the target's path (the open hand needs "
                                   "~3 grid columns of free space in front of whatever it reaches for)")
            state, saved = dict(start), dict(self._virtual)
            try:
                for j, cell in plan:
                    try:
                        self._move_check(j, cell)
                    except (ValueError, RuntimeError):
                        banned.add((tuple(state[m] for m in start if m != t), j, cell))
                        break
                    state[j] = cell
                    self._virtual[j] = self._cell_point(j, cell)
                else:
                    try:
                        self._move_check(t, goal)
                        if self.verbose and attempt:
                            print(f"   (plan found after {attempt} re-plan(s))")
                        return plan + [(t, goal)]
                    except GraspError as err:
                        culprits = {state[j] for j in self._object_blockers(err, exclude={t})}
                        if not culprits or culprits <= keepout:
                            raise RuntimeError(f"target move fails for a reason the grid can't fix: {err}")
                        keepout |= culprits
            finally:
                self._virtual = saved
        raise RuntimeError(f"no valid plan after {max_replans} re-plans")

    def rearrange(self, target, goal, max_replans=3):
        """Clear the target's path (pick & place every blocking object to a free cell where it
        blocks nothing), then pick & place the target into grid cell `goal`.

        Each move is re-validated on the real scene right before the arm moves; if objects landed
        far enough off their planned cells that a step no longer works, the remaining moves are
        re-planned from the actual scene.  Returns the executed moves."""
        t, goal = self.obj_id(target), tuple(goal)
        done = []
        for attempt in range(max_replans + 1):
            if self.verbose:
                print(("planning" if attempt == 0 else "re-planning") +
                      f": move {self.objects[t]['name']} {self.cell_of(t)} -> {goal}")
            moves = self.plan_rearrangement(t, goal)
            if self.verbose:
                for k, (i, cell) in enumerate(moves, 1):
                    role = "target" if i == t else "clear the path"
                    print(f"   plan {k}: {self.objects[i]['name']:<7} {self.cell_of(i)} -> {cell}   ({role})")
            for i, cell in moves:
                if self.verbose:
                    print(f"[{len(done) + 1}] {self.objects[i]['name']} {self.cell_of(i)} -> {cell}")
                depth, dy = grid_cell_center(cell, self.grid)
                try:
                    final = self.pick_and_place(i, 0, depth, dy, lift=self.lift)
                except GraspError as err:                     # raised before the arm moved
                    if self.held is not None:                 # too late to re-plan: it is mid-carry
                        raise RuntimeError(f"failed while carrying {self.objects[i]['name']}: {err}") from err
                    if self.verbose:
                        print(f"      step no longer valid on the real scene ({err}); re-planning")
                    break
                done.append((i, cell))
                if self.verbose:
                    print(f"      placed, {np.hypot(final[0] - CAB_X0 - depth, final[1] - dy)*100:.1f} cm "
                          "from the cell centre")
            else:
                return done
        raise RuntimeError(f"gave up after {max_replans} re-plans")

    def blockers_of(self, target, goal):
        """Names of the objects that, each on its own (all others taken away), stop `target` from
        being picked, carried low and placed at grid `goal` -- i.e. the objects in its path."""
        t, goal, names = self.obj_id(target), tuple(goal), []
        with self.hypothetical({k: None for k in range(len(self.objects)) if k != t}):
            try:
                self._move_check(t, goal)
            except (ValueError, GraspError) as err:
                raise RuntimeError(f"{self.objects[t]['name']} cannot be moved to {goal} even on an empty "
                                   f"grid: {err}") from None
        for j in range(len(self.objects)):
            if j == t:
                continue
            with self.hypothetical({k: None for k in range(len(self.objects)) if k not in (t, j)}):
                try:
                    self._move_check(t, goal)
                except GraspError:
                    names.append(self.objects[j]["name"])
        return names

    def grid_text(self, goal=None):
        """ASCII picture of the grid, back row on top (labels: object 'label', G = goal, . = empty)."""
        occ = self.occupancy()
        lines = ["        " + " ".join(f"c{c}" for c in range(self.grid["cols"]))]
        for r in reversed(range(self.grid["rows"])):
            cells = []
            for c in range(self.grid["cols"]):
                if (r, c) in occ:
                    cells.append(f" {self.objects[occ[(r, c)]]['label']}")
                else:
                    cells.append(" G" if goal is not None and (r, c) == tuple(goal) else " .")
            tag = " (back)" if r == self.grid["rows"] - 1 else " (front)" if r == 0 else ""
            lines.append(f"  r{r}   " + " ".join(cells) + tag)
        return "\n".join(lines)

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
# Single cell with a 6 x 6 grid.  The cylinder T must go from (4, 1) to (3, 4); cubes 1 and 2 are
# in the way of picking it (1 straight in front, 2 diagonally in front -- the hand is ~15 cm wide)
# and cube 3 is in front of the goal.  Edit cells here; one object per cell.
GRID_SCENE = dict(
    n_cells=1,
    cell_w=CELL_W,
    grid=GRID,
    objects=[
        dict(name="target", label="T", shape="cylinder", color="green", size=0.015, height=0.05, grid=(3, 4)),
        dict(name="block1", label="1", shape="box", color="red", size=0.015, height=0.06, grid=(2, 3)),
        dict(name="block2", label="2", shape="box", color="red", size=0.015, height=0.06, grid=(3, 2)),
        dict(name="block3", label="3", shape="box", color="red", size=0.015, height=0.06, grid=(2, 5)),
    ],
)
GRID_TASK = dict(target="target", goal=(3, 0))


def random_grid_scene(seed):
    """Target cylinder in a random cell, 3 blocks placed in its pick / place corridors."""
    rng = np.random.default_rng(seed)
    rows, cols = GRID["rows"], GRID["cols"]
    t = (int(rng.integers(2, rows)), int(rng.integers(0, cols)))
    goal = t
    while abs(goal[1] - t[1]) < 2:
        goal = (int(rng.integers(1, rows - 1)), int(rng.integers(0, cols)))
    path = lambda cell: [(r, c) for r in range(cell[0]) for c in (cell[1] - 1, cell[1], cell[1] + 1)
                         if 0 <= c < cols]
    pool = sorted(set(path(t)) | set(path(goal)) - {t, goal})
    blocks = [pool[k] for k in rng.choice(len(pool), 3, replace=False)]
    objs = [dict(name="target", label="T", shape="cylinder", color="green", size=0.015, height=0.05, grid=t)]
    for k, (cell, col) in enumerate(zip(blocks, ["red", "blue", "orange"]), 1):
        objs.append(dict(name=f"block{k}", label=str(k), shape="box", color=col, size=0.015, height=0.06,
                         grid=cell))
    return dict(GRID_SCENE, objects=objs), dict(target="target", goal=goal)


def solvable_grid_scene(seed, max_tries=15):
    """First random layout (seed, seed+1, ...) for which a collision-safe sequence of moves exists.
    Some random layouts have none -- e.g. a cluster in which every block is boxed in -- and are skipped."""
    skipped = []
    for s in range(seed, seed + max_tries):
        scene, task = random_grid_scene(s)
        try:
            XArm7CabinetEnv(**scene, verbose=False).plan_rearrangement(task["target"], task["goal"])
        except (RuntimeError, ValueError) as err:
            skipped.append((s, str(err).split("(")[0].strip()[:100]))
            continue
        return scene, task, s, skipped
    raise RuntimeError(f"no solvable random layout in seeds {seed}..{seed + max_tries - 1}")


if __name__ == "__main__":
    arg = lambda flag, default: sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default
    planner = arg("--planner", "rrt_connect")
    lift = float(arg("--lift", LIFT))
    if "--random-grid" in sys.argv:
        scene, task, used, skipped = solvable_grid_scene(int(arg("--random-grid", 0)))
        for s, why in skipped:
            print(f"(random layout {s} skipped: {why})")
        print(f"random layout {used}")
    else:
        scene, task = GRID_SCENE, GRID_TASK
    env = XArm7CabinetEnv(**scene, planner=planner, lift=lift, use_planner="--no-planner" not in sys.argv)
    for o in env.objects:
        print(f"{o['label']} = {o['name']:<7}: {o['color']} {o['shape']}, {2*o['size']*100:.1f} cm wide, "
              f"{o['height']*100:.0f} cm tall, grid cell {env.cell_of(o['name'])}")
    print("\ngrid before (G = goal of the target):\n" + env.grid_text(goal=task["goal"]) + "\n")
    in_path = env.blockers_of(task["target"], task["goal"])
    print("objects in the target's path (each one alone stops the move): " + (", ".join(in_path) or "none"))
    if "--headless" not in sys.argv:
        env.launch_viewer()

    env.rearrange(task["target"], task["goal"])

    print("\ngrid after:\n" + env.grid_text() + "\n")
    ok = env.cell_of(task["target"]) == tuple(task["goal"])
    print("done:", "target is in its goal cell" if ok else "TARGET NOT AT GOAL")
    if env.viewer is not None:
        while env.viewer.is_running():
            env.step(10)
