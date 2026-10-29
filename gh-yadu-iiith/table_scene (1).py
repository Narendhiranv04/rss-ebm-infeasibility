#!/usr/bin/env python3
"""
table_scene.py -- xArm7 + an OPEN table (no walls, no ceiling) with an 8 x 8 grid of cells.

Put it next to xarm7_cabinet_env.py, rrt_connect.py, rrt_star.py and mujoco_menagerie/ (it only
borrows constants and two helpers from them; nothing is changed).

    python table_scene.py                     # runs the USER SETTINGS below, with the viewer
    python table_scene.py --headless          # same, no window
    python table_scene.py --reach             # how many of the grasps reach each cell
    python table_scene.py --dataset target    # one trajectory per grasp pose -> dataset/
    python table_scene.py --once              # exit when the run is over (default: wait for 'reset' / 'quit')

COLLISION CHECKING.  Feasibility = inverse kinematics reaches the pose (and the whole path) within the joint
limits, the path does not jump between IK branches, AND the arm + open hand keep COLLISION_MARGIN clear of the
table and of every object and do not touch themselves (MuJoCo contact query on a planning copy of the scene, at
every waypoint of every free move and every straight hand move).  The one exception is the object being picked /
carried / placed (the fingers close on it); object-to-object clearance during the carry is the footprint rule's job.

STRUCTURE (methods of XArm7TableEnv; each one that moves the arm first runs IK + RRT-Connect, then moves):
    MoveToObject(name)     arm to the object's hard-coded cell, fingers open around it
    CloseGripper()         close, if the object can be grabbed (and check it is held)      } Pick(name)
    MoveToLocation(cell)   with the object held: lift, carry sideways, lower, at the hard-coded cell
    OpenGripper()          release                                                          } Place(cell)
    PickAndPlace(name, cell) = Pick + Place (next grasp if the fingers miss).  The locations are the
    hard-coded ones below: TARGET_CELL / OBSTACLES (where things are), GOAL_CELL / MOVE_COMMANDS (where they go).
    Free moves (home -> hover, retreat -> home) use RRT-Connect in joint space, checked against the table, the
    objects and the arm itself (see COLLISION CHECKING).
If a move is infeasible the program asks which object to move out of the way first and where to (type
'hint' to test which single objects matter), then tries again.  Type 'reset' at any time -- also while
the arm is moving -- to restore the simulation and start over; 'quit' to leave.

SIDEWAYS ONLY.  A carried object is lifted just CARRY_LIFT (2 cm) and then only moves in the table
plane -- never up and over another object.  The route is planned in the plane by RRT-Connect
(rrt_connect.py) with a grid-occupancy rule: the object's footprint keeps CARRY_GAP away from
every other object's footprint.  (This is a footprint test in 2-D, not an arm collision check.)

GRASP POSES per object = N_AZIMUTH approach azimuths (default 8, every 45 deg) x len(TILTS_DEG) tilts from vertical
(default only 90 deg: the hand LEVEL with the table, a front grasp -- "0 deg" above the horizontal -> 8 poses; no top
grasps.  Tilt 0 = from the top, 75 = nearly horizontal).  For a pose (azimuth a, tilt t):
    approach axis  z_g = (-sin t cos a, -sin t sin a, -cos t)    hand -> object (tilt 0: straight down)
    closing axis   y_g = (-sin a, cos a, 0)                      horizontal, perpendicular to the approach
    x_g = y_g x z_g
    the hand sits on the side of azimuth a, tilted toward the object.
Every pose is tried as one complete trajectory: hover -> approach -> close -> lift 2 cm ->
carry sideways -> lower -> open -> retreat -> home.
"""
import argparse
import csv
import itertools
import json
import math
import os
import re
import sys
import threading
import time

import numpy as np
import mujoco

from xarm7_cabinet_env import (XARM7_XML, ARM_JOINTS, HAND_ROOT, BASE_Z, GRASP_SITE_Z,
                               GRIPPER_OPEN, GRIPPER_CLOSED, COLORS)
from rrt_connect import RRTConnect
from rrt_star import shortcut, time_parameterize, densify, MujocoCollisionChecker

# =============================================================================== USER SETTINGS
# Cells are (row, col), both 0..7.  Row 0 is the row NEAREST the robot, col 0 is the -y edge.
TARGET_CELL = (0, 6)
GOAL_CELL = (7, 7)
TARGET = dict(shape="cylinder", size=0.020, height=0.16, color="green")   # size = radius / half-width (m)

OBSTACLES = [                       # any number; shape "box" | "cylinder"; colour name, "#rrggbb" or "r g b a"
    dict(cell=(0, 7), shape="box",      size=0.014, height=0.15, color="blue"),
    dict(cell=(1, 6), shape="box",      size=0.014, height=0.17, color="yellow"),
    dict(cell=(0, 5), shape="box",      size=0.016, height=0.16, color="red"),
   dict(cell=(3, 6), shape="box",      size=0.015, height=0.15, color="blue"),
    dict(cell=(4, 5), shape="box",      size=0.014, height=0.18, color="blue"),
    dict(cell=(1, 5), shape="box",      size=0.014, height=0.18, color="blue"),
    dict(cell=(2, 5), shape="box",      size=0.014, height=0.18, color="blue"),
    dict(cell=(3, 5), shape="box",      size=0.014, height=0.18, color="blue"),
    dict(cell=(0, 5), shape="box",      size=0.014, height=0.18, color="blue"),
    #dict(cell=(5, 5), shape="box",      size=0.014, height=0.18, color="blue"),
    #dict(cell=(6, 5), shape="box",      size=0.014, height=0.18, color="blue"),
]


# Commands run in this order BEFORE the target is moved to GOAL_CELL.  Each one is
# (object name, (row, col)): the arm picks that object and puts it in that cell.
# Names: "target", "obstacle1", "obstacle2", ... (the order of OBSTACLES above)
# e.g. MOVE_COMMANDS = [("obstacle1", (4, 4))]    (empty list = the arm only moves the target)
MOVE_COMMANDS = []

# =============================================================================== geometry
ROWS, COLS = 8, 8
PITCH = 0.07                        # cell pitch (m) -> the grid is 0.56 m x 0.56 m.  0.07 is the smallest pitch at which the
                                    # hand fits between neighbouring objects with the target; 0.05 / 0.06 do not
TABLE_Z = 0.08                      # table-top height (m)
GRID_X0 = 0.40                      # x of the grid centre (m); the robot base is at the origin

CARRY_LIFT = 0.025                   # sideways-only: the object is lifted just this much (m)
CARRY_GAP = 0.006                   # footprint clearance to other objects while carrying (m)
FOOTPRINT_GAP = 0.008               # minimum clearance between objects when placed (m)
FINGER_CLEARANCE = 0.006            # per-side opening beyond the object width when pre-shaping (m)
GRASP_TOP_MARGIN = 0.012            # pads grip the TOP of the object: pad centre = top - this (m).  Keeps the
                                    # wide gripper body above neighbours; min pad centre is 1.5 cm above the table
HAND_FOOTPRINT_RULE = True          # carry rule uses the measured hand footprint (False: the object only)
PRE_DIST = 0.13                     # hover this far back along the approach axis (m)
MIN_HEIGHT = 0.035
HOME_HOVER = 0.30                   # "ready" pose: this far above the table, hand pointing down

N_AZIMUTH = 8                       # approach directions, evenly spaced around the object (every 45 deg)
TILTS_DEG = (90.0,)                 # from vertical: 90 = hand level with the table (front grasp, "0 deg"); 0 = straight
                                    # down (top grasp), 75 = nearly horizontal.  The old set was (0, 15, 30, 45, 75) x 12
COLLISION_MARGIN = 0.005            # clearance of the arm / open hand to the table and the objects (m)
N_GRASPS = N_AZIMUTH * len(TILTS_DEG)   # grasp poses per object (any number: edit N_AZIMUTH / TILTS_DEG)

PLAN_VMAX, PLAN_AMAX = 1.0, 1.5     # joint limits for free moves
CART_VMAX, CART_AMAX = 0.5, 1.0     # slower limits for approach / carry / retreat


# Footprint of the xArm gripper (closed on an object), measured from the model with a top-down grasp.
# Bands of height above the pad centre (m) -> half-extent along the closing axis (m, depends on the pad
# gap) and across it.  The fingers are narrow; the linkage 3-5 cm above the pads is the widest part.
HAND_BANDS = [(-0.03, 0.00), (0.00, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.05), (0.05, 0.30)]
_HAND_EY = [lambda g: g / 2 + 0.016, lambda g: g / 2 + 0.026, lambda g: g / 2 + 0.036,
            lambda g: 0.072, lambda g: g / 2 + 0.050, lambda g: 0.052]
_HAND_EX = [0.017, 0.017, 0.017, 0.026, 0.029, 0.029]


def _handles_object(fn):
    """Decorator: while fn runs, its first argument is 'the object being handled' -- the one object the hand may
    touch (see XArm7TableEnv.arm_hits)."""
    def wrapped(self, name, *a, **kw):
        old = self._ck_obj
        self._ck_obj = self.obj_id(name)
        try:
            return fn(self, name, *a, **kw)
        finally:
            self._ck_obj = old
    wrapped.__name__, wrapped.__doc__ = fn.__name__, fn.__doc__
    return wrapped


class ReachError(RuntimeError):
    """No grasp (of all N_GRASPS) gives a kinematically feasible pick -> carry -> place."""


class GraspFailed(RuntimeError):
    """The object cannot be grabbed, or the gripper closed but the object did not come with it."""


class ObjectLost(ReachError):
    """The object did not stay where it was put (tipped over / fell off the table)."""


class ResetRequested(Exception):
    """The user typed 'reset'."""


class QuitRequested(Exception):
    """The user typed 'quit'."""


class NoInput(Exception):
    """Input ended (stdin closed): nobody to ask."""


def _color(c):
    if isinstance(c, str) and c.startswith("#") and len(c) == 7:
        return " ".join(f"{int(c[k:k + 2], 16) / 255:.3f}" for k in (1, 3, 5)) + " 1"
    if isinstance(c, (list, tuple)):
        return " ".join(str(float(v)) for v in (list(c) + [1.0])[:4])
    return COLORS.get(c, c)


def footprint_radius(o):
    return o["size"] * (math.sqrt(2) if o["shape"] == "box" else 1.0)


def rot_to_quat(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


# =============================================================================== the grasps
def grasp_frame(azimuth_deg, tilt_deg):
    """Rotation matrix (columns = gripper x, y, z in the world) for one grasp pose."""
    a, t = math.radians(azimuth_deg), math.radians(tilt_deg)
    z = np.array([-math.sin(t) * math.cos(a), -math.sin(t) * math.sin(a), -math.cos(t)])
    y = np.array([-math.sin(a), math.cos(a), 0.0])
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def grasp_poses(obj, table_z, robot_xy=(0.0, 0.0)):
    """All the grasp poses (N_GRASPS) for one object at its current position, preferred ones first.

    Each: id, azimuth_deg, tilt_deg, R, quat (w x y z), approach, closing, position (pad centre),
    width (object width along the closing axis), open_width / shut_width (pad gap)."""
    p = np.asarray(obj["pos"], float)
    h = obj["height"]
    z = table_z + max(0.015, h - GRASP_TOP_MARGIN)
    to_robot = math.degrees(math.atan2(robot_xy[1] - p[1], robot_xy[0] - p[0]))
    out = []
    for tilt in TILTS_DEG:
        for k in range(N_AZIMUTH):
            az = 360.0 / N_AZIMUTH * k
            R = grasp_frame(az, tilt)
            c = R[:, 1]
            width = 2 * obj["size"] if obj["shape"] == "cylinder" else \
                2 * obj["size"] * (abs(c[0]) + abs(c[1]))
            out.append(dict(azimuth_deg=az, tilt_deg=tilt, R=R, quat=rot_to_quat(R), approach=R[:, 2],
                            closing=c, position=np.array([p[0], p[1], z]), width=width,
                            open_width=width + 2 * FINGER_CLEARANCE, shut_width=max(width - 0.001, 0.0)))
    # preference: top-down first, then the hand coming from the robot's side
    wrap = lambda d: abs((d + 180.0) % 360.0 - 180.0)
    out.sort(key=lambda g: (g["tilt_deg"], wrap(g["azimuth_deg"] - to_robot)))
    for i, g in enumerate(out):
        g["id"] = i
    return out


# =============================================================================== scene XML
def build_scene_xml(objects, table_z, x0, pitch, rows, cols):
    hx, hy = rows * pitch / 2 + 0.08, cols * pitch / 2 + 0.08
    geoms = [f'<geom name="table_top" type="box" pos="{x0} 0 {table_z - 0.02}" size="{hx} {hy} 0.02" '
             f'rgba="0.72 0.58 0.42 1" friction="1.0 0.02 0.002"/>']
    for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        geoms.append(f'<geom name="leg{k}" type="box" pos="{x0 + sx * (hx - 0.03)} {sy * (hy - 0.03)} {(table_z - 0.04) / 2}" '
                     f'size="0.02 0.02 {(table_z - 0.04) / 2}" rgba="0.45 0.35 0.25 1"/>')
    line = 'type="box" rgba="0.15 0.12 0.1 1" contype="0" conaffinity="0"'
    zc = table_z + 0.0004
    for k in range(cols + 1):
        geoms.append(f'<geom name="grid_col{k}" {line} pos="{x0} {(k - cols / 2) * pitch} {zc}" '
                     f'size="{rows * pitch / 2} 0.0008 0.0004"/>')
    for k in range(rows + 1):
        geoms.append(f'<geom name="grid_row{k}" {line} pos="{x0 + (k - rows / 2) * pitch} 0 {zc}" '
                     f'size="0.0008 {cols * pitch / 2} 0.0004"/>')
    bodies = []
    for o in objects:
        x, y, _ = o["pos"]
        h2 = o["height"] / 2
        size = f'{o["size"]} {o["size"]} {h2}' if o["shape"] == "box" else f'{o["size"]} {h2}'
        bodies.append(
            f'<body name="{o["name"]}" pos="{x} {y} {table_z + h2 + 0.0005}"><freejoint name="{o["name"]}_free"/>'
            f'<geom name="{o["name"]}_g" type="{o["shape"]}" size="{size}" rgba="{_color(o["color"])}" '
            f'mass="{o["mass"]}" friction="{o["friction"]} 0.02 0.002"/></body>')
    nl = "\n    "
    return f"""
<mujoco model="xarm7_table_scene">
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
    <geom name="pedestal" type="cylinder" pos="0 0 {BASE_Z / 2}" size="0.06 {BASE_Z / 2}" rgba="1 1 1 1"/>
    {nl.join(geoms)}
    {nl.join(bodies)}
  </worldbody>
</mujoco>
"""


# =============================================================================== environment
class XArm7TableEnv:
    _collision_check = True                                # built in (collision_check.install() leaves it alone)

    def __init__(self, objects, table_z=TABLE_Z, x0=GRID_X0, pitch=PITCH, rows=ROWS, cols=COLS,
                 verbose=True, xml_path=XARM7_XML):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"xarm7.xml not found at {xml_path} (clone mujoco_menagerie next to this file "
                                    "or set XARM7_XML)")
        self.table_z, self.x0, self.pitch, self.rows, self.cols, self.verbose = table_z, x0, pitch, rows, cols, verbose
        self.objects = self._normalize(objects)
        self.obj_index = {o["name"]: i for i, o in enumerate(self.objects)}

        scene = mujoco.MjSpec.from_string(build_scene_xml(self.objects, table_z, x0, pitch, rows, cols))
        arm = mujoco.MjSpec.from_file(xml_path)
        for b in arm.bodies:
            b.gravcomp = 1.0
        arm.body(HAND_ROOT).add_site(name="grasp", pos=[0, 0, GRASP_SITE_Z], size=[0.006, 0, 0], rgba=[0, 1, 0, 1])
        scene.attach(arm, frame=scene.worldbody.add_frame(pos=[0, 0, 0]), prefix="")
        self.model = scene.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.jrange = np.clip(np.array([m.jnt_range[m.joint(j).id] for j in ARM_JOINTS]), -3.1, 3.1)
        self.act_arm = [m.actuator(f"act{i}").id for i in range(1, 8)]
        self.act_grip = m.actuator("gripper").id
        self.site_id = m.site("grasp").id
        self.obj_qadr = [m.joint(o["name"] + "_free").qposadr[0] for o in self.objects]
        self.ik_data = mujoco.MjData(m)
        self.q_home = np.array([0.0, -0.25, 0.0, 0.9, 0.0, 1.15, 0.0])
        kp = m.actuator_gainprm[self.act_arm, 0]
        kv = -m.actuator_biasprm[self.act_arm, 2]
        self.servo_lead = (kv + m.dof_damping[self.dadr]) / kp
        self.max_track_err = 0.04
        self._wrist_bodies = [m.body(n).id for n in ("link5", "link6", "link7", HAND_ROOT)]
        self.console = None                                    # typed commands (set by main)
        self.current = None                                    # the pick in progress (see MoveToObject)
        self.last_free = dict(direct=True, iters=0)
        self._snap, self.pushed = {}, []
        self.viewer = None
        self._ck_obj = None                                    # the object being handled (its contacts are allowed)
        self.ck = self._build_checker(xml_path)
        self.reset()
        self._calibrate_gripper()

    # -- collision checking: arm + open hand vs the table, every object and itself
    def _build_checker(self, xml_path):
        """Planning copy of the scene for MuJoCo's contact query: the gripper's own parts (and base / link1), which
        touch by design, are excluded; the robot geoms get a margin so near misses count."""
        spec = mujoco.MjSpec.from_string(build_scene_xml(self.objects, self.table_z, self.x0, self.pitch, self.rows, self.cols))
        arm = mujoco.MjSpec.from_file(xml_path)
        arm.body(HAND_ROOT).add_site(name="grasp", pos=[0, 0, GRASP_SITE_Z], size=[0.006, 0, 0], rgba=[0, 1, 0, 1])
        spec.attach(arm, frame=spec.worldbody.add_frame(pos=[0, 0, 0]), prefix="")
        m = self.model
        hand = [m.body(b).name for b in self._subtree(m, m.body(HAND_ROOT).id)]
        for a, b in list(itertools.combinations(hand, 2)) + [("link_base", "link1")]:
            e = spec.add_exclude(); e.bodyname1, e.bodyname2 = a, b
        pm = spec.compile()
        pm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)    # mid-phase culling ignores geom margins
        root = pm.body("link_base").id
        robot = [b for b in range(pm.nbody) if pm.body_rootid[b] == root and pm.body_weldid[b] != 0]
        for g in range(pm.ngeom):
            if pm.geom_bodyid[g] in robot:
                pm.geom_margin[g] = 2 * COLLISION_MARGIN
        self._obj_geoms = [[g for g in range(pm.ngeom) if pm.geom_bodyid[g] == pm.body(o["name"]).id] for o in self.objects]
        return MujocoCollisionChecker(pm, np.array([pm.joint(j).qposadr[0] for j in ARM_JOINTS]), robot,
                                      self._subtree(pm, pm.body(HAND_ROOT).id), pm.site("grasp").id, margin=COLLISION_MARGIN)

    def arm_hits(self, q):
        """What the arm / open hand would touch at configuration q: names of the table / objects, 'itself' for
        self-contact; [] if free.  The object being handled (picked, carried, placed) is allowed.  The object
        positions are copied from the simulation, so moved objects are seen where they are."""
        ck = self.ck
        ck.sync(self.data.qpos)
        handled = {i for i in (self._ck_obj, self.held, self.current["i"] if self.current else None) if i is not None}
        with ck.ignoring([g for i in handled for g in self._obj_geoms[i]]):
            pairs = ck.collisions(q)
        names = set()
        for a, b in pairs:
            st = [g for g in (a, b) if ck.role[g] == 0]
            if not st:
                names.add("itself")
            for g in st:
                body = ck.m.body(ck.m.geom_bodyid[g]).name
                names.add(body if body != "world" else (ck.m.geom(g).name or "table"))
        return sorted(names)

    # -- objects
    def cell_xy(self, cell):
        r, c = cell
        return np.array([self.x0 + (r - (self.rows - 1) / 2) * self.pitch, (c - (self.cols - 1) / 2) * self.pitch])

    def cell_at(self, xy):
        r = int(round((xy[0] - self.x0) / self.pitch + (self.rows - 1) / 2))
        c = int(round(xy[1] / self.pitch + (self.cols - 1) / 2))
        return (r, c) if 0 <= r < self.rows and 0 <= c < self.cols else None

    def _normalize(self, objects):
        out, names = [], set()
        for k, o in enumerate(objects):
            o = dict(o)
            o.setdefault("name", f"obstacle{k}" if k else "target")
            o.setdefault("shape", "box"); o.setdefault("size", 0.015); o.setdefault("height", 0.06)
            o.setdefault("color", "red"); o.setdefault("mass", 0.05); o.setdefault("friction", 1.2)
            cell = tuple(int(v) for v in o["cell"])
            if o["name"] in names:
                raise ValueError(f"duplicate object name {o['name']!r}")
            names.add(o["name"])
            if o["shape"] not in ("box", "cylinder"):
                raise ValueError(f"{o['name']}: shape must be 'box' or 'cylinder'")
            if not (0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols):
                raise ValueError(f"{o['name']}: cell {cell} is outside the {self.rows}x{self.cols} grid")
            if 2 * o["size"] > self.pitch - FOOTPRINT_GAP - 0.004:
                raise ValueError(f"{o['name']}: {2 * o['size'] * 100:.1f} cm is too wide for a {self.pitch * 100:.0f} cm cell "
                                 f"(max {(self.pitch - FOOTPRINT_GAP - 0.004) * 100:.1f} cm)")
            if o["height"] < MIN_HEIGHT:
                raise ValueError(f"{o['name']}: needs height >= {MIN_HEIGHT * 100:.1f} cm for the finger pads")
            o["cell"] = cell
            o["pos"] = np.array([*self.cell_xy(cell), self.table_z + o["height"] / 2])
            o["home_pos"] = o["pos"].copy()                 # where reset() puts it back
            out.append(o)
        seen = {}
        for o in out:
            if o["cell"] in seen:
                raise ValueError(f"{o['name']} and {seen[o['cell']]} are in the same cell {o['cell']}")
            seen[o["cell"]] = o["name"]
        return out

    def obj_id(self, name):
        if isinstance(name, str):
            if name not in self.obj_index:
                raise KeyError(f"no object named {name!r}; have {list(self.obj_index)}")
            return self.obj_index[name]
        return int(name)

    def obj_pos(self, name):
        return self.data.body(self.objects[self.obj_id(name)]["name"]).xpos.copy()

    def cell_of(self, name):
        return self.cell_at(self.obj_pos(name)[:2])

    def grid_text(self, goal=None):
        occ = {self.cell_of(o["name"]): o for o in self.objects}
        lines = ["        " + " ".join(f"c{c}" for c in range(self.cols))]
        for r in reversed(range(self.rows)):
            row = []
            for c in range(self.cols):
                o = occ.get((r, c))
                row.append(("T" if o["name"] == "target" else o["name"].replace("obstacle", "")) if o
                           else ("G" if goal == (r, c) else "."))
            lines.append(f"  r{r}    " + "  ".join(row) + (" (nearest the robot)" if r == 0 else " (far)" if r == self.rows - 1 else ""))
        return "\n".join(lines)

    # -- gripper
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

    def _calibrate_gripper(self):
        """Map gripper command -> pad gap, measured with the arm held in the 'ready' pose (hand
        pointing down), which is how it is used.  (With the arm straight up the same command gives
        a gap up to 2.7 cm wider, so the pre-shaped fingers would land on the object.)"""
        m, d = self.model, mujoco.MjData(self.model)
        l, r = m.geom("left_finger_pad_1").id, m.geom("right_finger_pad_1").id
        pad_t = 2 * m.geom_size[l][1]
        d.qpos[:] = self.data.qpos
        d.ctrl[self.act_arm] = self.q_ready
        ctrls, gaps = np.linspace(GRIPPER_OPEN, GRIPPER_CLOSED, 9), []
        for c in ctrls:
            d.ctrl[self.act_grip] = c
            for _ in range(500):
                mujoco.mj_step(m, d)
            gaps.append(np.linalg.norm(d.geom_xpos[l] - d.geom_xpos[r]) - pad_t)
        order = np.argsort(gaps)
        self._grip_cal = (np.array(gaps)[order], ctrls[order])

    def set_gripper_width(self, width):
        gaps, ctrls = self._grip_cal
        self.data.ctrl[self.act_grip] = float(np.interp(width, gaps, ctrls))

    def set_gripper(self, open_):
        self.data.ctrl[self.act_grip] = GRIPPER_OPEN if open_ else GRIPPER_CLOSED

    # -- basics
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for o, a in zip(self.objects, self.obj_qadr):
            o["pos"] = o["home_pos"].copy()                    # back to the starting cell
            self.data.qpos[a:a + 3] = o["pos"]; self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.ready_p = np.array([self.x0 - 0.12, 0.0, self.table_z + HOME_HOVER])
        self.ready_R = grasp_frame(0.0, 0.0)
        self.q_ready = self.ik(self.ready_p, self.ready_R, self.q_home, iters=300)
        self.q_cmd = self.q_ready.copy()
        self.data.qpos[self.qadr] = self.q_cmd
        self.data.ctrl[self.act_arm] = self.q_cmd
        self.data.ctrl[self.act_grip] = GRIPPER_OPEN
        self.held = None
        self.current = None
        mujoco.mj_forward(self.model, self.data)
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)

    def step(self, n=1):
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
        if self.viewer is not None:
            self.viewer.sync()
        if self.console is not None:
            self.console.check_reset()                         # raises ResetRequested if 'reset' was typed

    def reset_all(self):
        """Restore the simulation: objects in their starting cells, arm at the ready pose, gripper open."""
        self.reset()
        if self.viewer is not None:
            self.viewer.sync()

    def wait(self, steps=200):
        self.step(steps)

    def grasp_pos(self):
        return self.data.site_xpos[self.site_id].copy()

    def q_now(self):
        return self.data.qpos[self.qadr].copy()

    def teleport(self, name, cell):
        """Instantly put an object into a cell (scene set-up; the arm does not move)."""
        i = self.obj_id(name)
        o = self.objects[i]
        for j, oj in enumerate(self.objects):
            if j != i and self.cell_of(j) == tuple(cell):
                raise ValueError(f"cell {tuple(cell)} is occupied by {oj['name']}")
        a = self.obj_qadr[i]
        self.data.qpos[a:a + 3] = [*self.cell_xy(cell), self.table_z + o["height"] / 2 + 0.0005]
        self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.data.qvel[self.model.joint(o["name"] + "_free").dofadr[0]:][:6] = 0
        mujoco.mj_forward(self.model, self.data)
        self.wait(100)

    # -- inverse kinematics (damped least squares on position + orientation)
    def ik(self, p, R, q0, iters=100, tol=1e-4):
        d = self.ik_data
        q = np.clip(np.asarray(q0, float), self.jrange[:, 0], self.jrange[:, 1])
        q_rest = self.q_home.copy(); q_rest[0] = math.atan2(p[1], p[0])
        for _ in range(iters):
            d.qpos[:] = self.data.qpos
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, d)
            mujoco.mj_comPos(self.model, d)
            cur_p, cur_R = d.site_xpos[self.site_id], d.site_xmat[self.site_id].reshape(3, 3)
            e_p = p - cur_p
            e_r = 0.5 * sum(np.cross(cur_R[:, i], R[:, i]) for i in range(3))
            if np.linalg.norm(e_p) < tol and np.linalg.norm(e_r) < tol:
                break
            jp, jr = np.zeros((3, self.model.nv)), np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, d, jp, jr, self.site_id)
            J = np.vstack([jp, jr])[:, self.dadr]
            JJt = J @ J.T + 1e-4 * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, np.concatenate([e_p, 0.5 * e_r]))
            null = (np.eye(7) - J.T @ np.linalg.solve(JJt, J)) @ (0.1 * (q_rest - q))
            q = np.clip(q + dq + null, self.jrange[:, 0], self.jrange[:, 1])
        return q

    def pose_error(self, q, p, R):
        d = self.ik_data
        d.qpos[:] = self.data.qpos; d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        cur_R = d.site_xmat[self.site_id].reshape(3, 3)
        return (np.linalg.norm(d.site_xpos[self.site_id] - p),
                np.linalg.norm(0.5 * sum(np.cross(cur_R[:, i], R[:, i]) for i in range(3))))

    def ik_pose(self, p, R, q_prefer, first=False):
        """IK for one pose from several seeds; the solution closest to q_prefer, or None
        (first=True: the first one that works -- faster, used for the reachability map)."""
        seeds = [q_prefer, self.q_home, self.q_ready]
        base = math.atan2(p[1], p[0])
        for dq1 in (0.0, 0.5, -0.5):
            s = self.q_home.copy(); s[0] = base + dq1
            seeds.append(s)
        best, best_d = None, np.inf
        for s in seeds:
            q = self.ik(p, R, s, iters=250, tol=1e-5)
            ep, er = self.pose_error(q, p, R)
            if ep < 1e-3 and er < 1e-2:
                dist = np.linalg.norm(q - q_prefer)
                if dist < best_d:
                    best, best_d = q, dist
                if first:
                    break
        return best

    def chain(self, q0, p0, points, R, step=0.005, max_jump=0.12):
        """Joint path along the polyline p0 -> points at fixed orientation R, by sequential IK.
        Raises RuntimeError if a point is out of reach or the solution jumps between IK branches."""
        pts = densify([np.asarray(p0, float)] + [np.asarray(p, float) for p in points], step)
        q, qs = np.asarray(q0, float).copy(), [np.asarray(q0, float).copy()]
        for p in pts[1:]:
            q_new = self.ik(p, R, q, iters=60, tol=2e-4)
            ep, er = self.pose_error(q_new, p, R)
            if ep > 2e-3 or er > 2e-2:
                raise RuntimeError(f"out of reach at {np.round(p, 3)}")
            if np.max(np.abs(q_new - q)) > max_jump:
                raise RuntimeError(f"IK jumps branch near {np.round(p, 3)}")
            hit = self.arm_hits(q_new)
            if hit:
                raise RuntimeError(f"the arm / hand would hit {', '.join(hit)} near {np.round(p, 3)}")
            q = q_new
            qs.append(q)
        return qs

    # -- the sideways-only carry route: RRT-Connect in the table plane
    @_handles_object
    def plan_carry_xy(self, name, goal_cell, grasp, seed=0, max_time=5.0, start_xy=None):
        """Route for the object's centre from where it is to goal_cell, in the table plane.

        The carried footprint is the object plus the gripper, by height band (HAND_BANDS): only the
        bands that overlap another object's height count, so a short neighbour only meets the narrow
        fingers while a tall one also meets the wide linkage.  It must keep CARRY_GAP away from every
        other object.  (A 2-D footprint rule from measured sizes, not a collision query.)"""
        i = self.obj_id(name)
        o = self.objects[i]
        p0 = np.asarray(start_xy, float) if start_xy is not None else self.obj_pos(i)[:2]
        goal = self.cell_xy(goal_cell)
        others = [(self.obj_pos(j)[:2], footprint_radius(oj)) for j, oj in enumerate(self.objects) if j != i]
        for j, oj in enumerate(self.objects):
            if j != i and self.cell_of(j) == tuple(goal_cell):
                raise ValueError(f"goal cell {tuple(goal_cell)} is occupied by {oj['name']}")
        c = np.asarray(grasp["closing"][:2], float)
        c = c / np.linalg.norm(c)
        v = np.array([-c[1], c[0]])                              # across the closing axis
        gap = grasp["open_width"]
        z_rel = grasp["position"][2] - self.table_z              # pad centre above the table
        tilt = math.radians(grasp["tilt_deg"])
        side = np.array([math.cos(math.radians(grasp["azimuth_deg"])), math.sin(math.radians(grasp["azimuth_deg"]))])
        r_me = footprint_radius(o)
        # for each other object: the hand bands that overlap its height (pad centre = 0)
        parts = []
        for j, oj in enumerate(self.objects):
            if j == i:
                continue
            top = oj["height"] - z_rel
            bands = [k for k, (lo, hi) in enumerate(HAND_BANDS) if top > lo and -z_rel < hi] \
                if HAND_FOOTPRINT_RULE else []
            parts.append((self.obj_pos(j)[:2], footprint_radius(oj), bands))

        def is_free(xy):
            for q, r, bands in parts:
                if np.linalg.norm(q - xy) < r_me + r + CARRY_GAP:              # the object itself
                    return False
                for k in bands:
                    hc = min(HAND_BANDS[k][1], 0.12) * math.tan(tilt)           # tilted hand leans outward
                    ctr = xy + side * hc
                    a, b = abs(np.dot(q - ctr, c)), abs(np.dot(q - ctr, v))
                    if math.hypot(max(a - _HAND_EY[k](gap), 0.0), max(b - _HAND_EX[k], 0.0)) < r + CARRY_GAP:
                        return False
            return True

        def edge_free(a, b):
            n = max(2, int(np.linalg.norm(b - a) / 0.003))
            return all(is_free(a + (b - a) * k / n) for k in range(n + 1))

        span = np.array([self.rows, self.cols]) * self.pitch / 2 + 0.25 * self.pitch
        lo, hi = np.array([self.x0, 0.0]) - span, np.array([self.x0, 0.0]) + span
        if not is_free(p0):
            raise ReachError(f"the hand (closing axis {np.degrees(np.arctan2(c[1], c[0])):.0f} deg) would overlap a neighbour at the start")
        if not is_free(goal):
            raise ReachError(f"the hand (closing axis {np.degrees(np.arctan2(c[1], c[0])):.0f} deg) would overlap a neighbour at the goal")
        planner = RRTConnect(is_free, edge_free, lo, hi, step=0.02, max_time=max_time, try_direct=True, seed=seed)
        path = planner.plan(p0, goal)
        if path is None:
            raise ReachError(f"no sideways route for {o['name']} to cell {tuple(goal_cell)}: other objects block it")
        path = shortcut(path, edge_free, seed=seed)
        self.last_carry = dict(waypoints=len(path), direct=planner.stats.get("direct", False),
                               iters=planner.stats.get("iters"))
        return [np.asarray(q, float) for q in path]

    # -- free (empty-hand) joint-space moves: IK gives the end pose, RRT-Connect the route
    def _above_table(self, q):
        """The check a free move gets: the arm and open hand are clear of the table, the objects and themselves."""
        return not self.arm_hits(q)

    def plan_free_move(self, qa, qb, seed=0, max_time=3.0):
        """RRT-Connect in joint space from qa to qb.  Valid = inside the joint limits and clear of the table, the
        objects and the arm itself (see _above_table / arm_hits).  Tries the
        straight joint line first, so with nothing in the way it connects immediately."""
        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        is_free = lambda q: bool(np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9) and self._above_table(q))

        def edge_free(a, b):
            n = max(2, int(np.ceil(np.max(np.abs(b - a)) / 0.05)))
            return all(is_free(a + (b - a) * k / n) for k in range(1, n))

        planner = RRTConnect(is_free, edge_free, lo, hi, step=0.3, max_time=max_time, try_direct=True, seed=seed)
        try:
            path = planner.plan(qa, qb)
        except RuntimeError as err:
            raise RuntimeError(f"free move: {err}") from None
        if path is None:
            raise RuntimeError("free move: RRT-Connect found no joint-space route")
        path = shortcut(path, edge_free, seed=seed)
        self.last_free = dict(direct=bool(planner.stats.get("direct", False)), iters=planner.stats.get("iters", 0))
        return [np.asarray(q, float) for q in path]

    # -- the three planning steps of a pick-and-place (IK + RRT-Connect; nothing moves)
    def _plan_approach(self, g, q0, seed=0):
        """IK for the hover and grasp poses, RRT-Connect for the free move to the hover pose, then the
        straight approach.  -> (segments, joint configuration at the grasp pose)."""
        R, a, pg = g["R"], g["approach"], g["position"]
        pre = pg - a * PRE_DIST
        q_pre = self.ik_pose(pre, R, q0)
        if q_pre is None:
            raise RuntimeError(f"hover pose {np.round(pre, 3)} is out of reach")
        segs = [dict(phase="to_hover", kind="joint", q=self.plan_free_move(q0, q_pre, seed)),
                dict(phase="open", kind="grip", grip=g["open_width"])]
        ch = self.chain(q_pre, pre, [pg], R)
        segs.append(dict(phase="approach", kind="cart", q=ch))
        return segs, ch[-1]

    def _plan_carry(self, name, cell, g, q_g, seed=0):
        """Lift, then carry sideways along an RRT-Connect route in the table plane, then lower; IK along
        the whole way.  -> (segments, joint configuration, hand position at the put-down, route)."""
        R, pg = g["R"], g["position"]
        lift = pg + np.array([0, 0, CARRY_LIFT])
        ch = self.chain(q_g, pg, [lift], R)
        segs = [dict(phase="lift", kind="cart", q=ch)]
        # the route starts where the object was when the grasp was chosen (a box twists a few mm in the pads
        # when the fingers close; the footprint rule has only a few mm of margin at the start)
        xy = self.plan_carry_xy(name, cell, g, seed=seed, start_xy=g["position"][:2])
        pts = [np.array([x, y, lift[2]]) for x, y in xy[1:]]
        ch = self.chain(ch[-1], lift, pts, R)
        segs.append(dict(phase="carry", kind="cart", q=ch))
        gx, gy = xy[-1]
        put = np.array([gx, gy, pg[2]])
        ch = self.chain(ch[-1], pts[-1] if pts else lift, [put], R)
        segs.append(dict(phase="lower", kind="cart", q=ch))
        return segs, ch[-1], put, xy

    def _plan_retreat(self, g, q, p, seed=0):
        """Straight back along the approach axis, then an RRT-Connect free move to the ready pose.  Near the
        edge of the arm's reach the full PRE_DIST may end out of reach (a few mm of hand position decide it),
        so a shorter straight retreat is used then: the open fingers only have to clear the object."""
        for dist in (PRE_DIST, 0.11, 0.09, 0.07, 0.05):
            try:
                ch = self.chain(q, p, [p - g["approach"] * dist], g["R"])
                break
            except RuntimeError:
                if dist == 0.05:
                    raise
        return [dict(phase="retreat", kind="cart", q=ch),
                dict(phase="to_ready", kind="joint", q=self.plan_free_move(ch[-1], self.q_ready, seed))]

    @_handles_object
    def plan_pick_place(self, name, goal_cell, grasp, q_start=None, seed=0):
        """Kinematic plan of the whole trajectory for one grasp pose (a dry run: nothing moves).
        Returns dict(feasible, reason, segments, ...).
        segments: list of dict(phase, kind='joint'|'cart'|'grip', q=waypoints or grip=(cmd))."""
        i = self.obj_id(name)
        o = self.objects[i]
        q0 = self.q_now() if q_start is None else np.asarray(q_start, float)
        plan = dict(feasible=False, reason="", segments=[], name=o["name"], goal=tuple(goal_cell),
                    grasp_id=grasp["id"], carry_xy=None)
        segs = plan["segments"]
        try:
            s1, q_g = self._plan_approach(grasp, q0, seed)
            segs += s1
            segs.append(dict(phase="close", kind="grip", grip=None))
            s2, q_put, put, xy = self._plan_carry(name, goal_cell, grasp, q_g, seed)
            plan["carry_xy"] = xy
            segs += s2
            segs.append(dict(phase="release", kind="grip", grip=grasp["open_width"]))
            segs += self._plan_retreat(grasp, q_put, put, seed)
            plan["feasible"] = True
        except (RuntimeError, ValueError) as err:
            plan["reason"] = str(err)
        return plan

    # -- executing plans in the simulator
    def execute(self, traj, settle=40):
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
        if max_err > self.max_track_err and self.verbose:
            print(f"   (arm lagged {max_err:.3f} rad behind the plan)")
        return max_err

    def segment_traj(self, seg):
        vmax, amax = (PLAN_VMAX, PLAN_AMAX) if seg["kind"] == "joint" else (CART_VMAX, CART_AMAX)
        return time_parameterize(seg["q"], vmax, amax, self.model.opt.timestep)

    def run_plan(self, plan):
        i = self.obj_id(plan["name"])
        o = self.objects[i]
        others = {j: self.obj_pos(j) for j in range(len(self.objects)) if j != i}
        plan["pushed"] = []

        def report(phase):                                   # observation only: who got moved?
            for j, p0 in others.items():
                d = np.linalg.norm(self.obj_pos(j)[:2] - p0[:2])
                if d > 0.004 or abs(self.obj_pos(j)[2] - p0[2]) > 0.004:
                    plan["pushed"].append((self.objects[j]["name"], phase, round(d * 1000, 1)))
                    others[j] = self.obj_pos(j)
                    if self.verbose:
                        print(f"      ! {self.objects[j]['name']} was moved {d * 1000:.0f} mm during '{phase}'")

        for seg in plan["segments"]:
            if seg["kind"] == "grip":
                if seg["phase"] == "close":
                    self.set_gripper(False); self.wait(400)
                    self.held = i
                else:
                    self.set_gripper_width(seg["grip"]); self.wait(300 if seg["phase"] == "release" else 200)
                    if seg["phase"] == "release":
                        self.held = None
                report(seg["phase"])
                continue
            self.execute(self.segment_traj(seg), settle=40)
            report(seg["phase"])
            if seg["phase"] == "lift":
                z = self.obj_pos(i)[2] - o["height"] / 2 - self.table_z
                if z < 0.3 * CARRY_LIFT:
                    self.set_gripper_width(0.07); self.wait(200)
                    self.held = None
                    raise GraspFailed(f"{o['name']} did not come up with the gripper")

    def go_ready(self):
        try:
            path = self.plan_free_move(self.q_now(), self.q_ready)
        except RuntimeError:
            path = [self.q_now(), self.q_ready]
        self.execute(time_parameterize(path, PLAN_VMAX, PLAN_AMAX, self.model.opt.timestep))

    # =========================================================================== the structured pipeline
    #   Pick  = MoveToObject + CloseGripper
    #   Place = MoveToLocation + OpenGripper (+ backing away)
    # Each function that moves the arm first runs IK and RRT-Connect (planning only), then executes.
    # The locations are the hard-coded ones: where an object is (TARGET_CELL / OBSTACLES) and where it
    # goes (GOAL_CELL / MOVE_COMMANDS).

    @_handles_object
    def _select_grasp(self, name, destination=None, grasp_id=None, skip=()):
        """First of the grasps that is feasible.  With a destination: the WHOLE pick -> carry -> place
        trajectory (the hand's footprint during the carry depends on the grasp, so the grasp has to be
        chosen together with the destination).  Without: just hover + approach.  Nothing moves."""
        i = self.obj_id(name)
        o = self.objects[i]
        o["pos"] = self.obj_pos(i)                           # refresh: an earlier attempt may have nudged it
        q0 = self.q_now()
        reasons = {}
        for g in grasp_poses(o, self.table_z):
            if self.console is not None:
                self.console.check_reset()                   # planning makes no steps: look for 'reset' here
            if (grasp_id is not None and g["id"] != grasp_id) or g["id"] in skip:
                continue
            if destination is None:
                try:
                    self._plan_approach(g, q0)
                except (RuntimeError, ValueError) as err:
                    reasons[g["id"]] = str(err)
                    continue
                return g
            plan = self.plan_pick_place(name, destination, g, q_start=q0)
            if plan["feasible"]:
                return g
            reasons[g["id"]] = plan["reason"]
        top = {}
        for r in reasons.values():
            key = r.split(" near ")[0].split(" at ")[0]
            top[key] = top.get(key, 0) + 1
        what = f" -> {tuple(destination)}" if destination is not None else ""
        raise ReachError(f"no feasible grasp for {o['name']}{what}; " +
                         (", ".join(f"{k} x{v}" for k, v in sorted(top.items(), key=lambda kv: -kv[1])[:3])
                          or "every grasp was already tried"))

    def _watch(self, skip_name):
        self._snap = {j: self.obj_pos(j) for j, oj in enumerate(self.objects) if oj["name"] != skip_name}
        self.pushed = []

    def _report(self, phase):                                # observation only: did the hand push anything?
        for j, p0 in list(self._snap.items()):
            p = self.obj_pos(j)
            d = np.linalg.norm(p[:2] - p0[:2])
            if d > 0.004 or abs(p[2] - p0[2]) > 0.004:
                self.pushed.append((self.objects[j]["name"], phase, round(d * 1000, 1)))
                self._snap[j] = p
                if self.verbose:
                    print(f"      ! {self.objects[j]['name']} was moved {d * 1000:.0f} mm during '{phase}'")

    def _exec_segments(self, segs, i=None):
        for seg in segs:
            self.execute(self.segment_traj(seg), settle=40)
            self._report(seg["phase"])
            if seg["phase"] == "lift" and i is not None:
                o = self.objects[i]
                if self.obj_pos(i)[2] - o["height"] / 2 - self.table_z < 0.3 * CARRY_LIFT:
                    self.set_gripper_width(0.07); self.wait(200)
                    self.held = None
                    raise GraspFailed(f"{o['name']} did not come up with the gripper")

    def pad_gap(self):
        """Opening between the finger pads (m)."""
        m, d = self.model, self.data
        l, r = m.geom("left_finger_pad_1").id, m.geom("right_finger_pad_1").id
        return float(np.linalg.norm(d.geom_xpos[l] - d.geom_xpos[r]) - 2 * m.geom_size[l][1])

    @_handles_object
    def MoveToObject(self, name, destination=None, grasp_id=None, skip=()):
        """Move the arm to `name` (an object, obstacle or the target) at its hard-coded location, with the
        fingers open around it.  IK + RRT-Connect run first (a dry run picks the grasp -- with `destination`
        one that can also reach it); then: free move to the hover pose, open the fingers, straight approach.
        Returns the grasp used."""
        if self.held is not None:
            raise RuntimeError(f"MoveToObject: the gripper already holds {self.objects[self.held]['name']}")
        i = self.obj_id(name)
        o = self.objects[i]
        g = self._select_grasp(name, destination, grasp_id, skip)
        segs, _ = self._plan_approach(g, self.q_now())          # again, from where the arm is now
        self._watch(o["name"])
        self.current = dict(name=o["name"], i=i, grasp=g, destination=destination)
        if self.verbose:
            how = "straight" if self.last_free["direct"] else f"RRT-Connect, {self.last_free['iters']} iterations"
            print(f"   MoveToObject({o['name']} at {self.cell_of(i)}): grasp #{g['id']} (azimuth {g['azimuth_deg']:.0f} deg, "
                  f"tilt {g['tilt_deg']:.0f} deg); IK ok; free move: {how}")
        for seg in segs:
            if seg["kind"] == "grip":
                self.set_gripper_width(seg["grip"]); self.wait(200)
            else:
                self.execute(self.segment_traj(seg), settle=40)
            self._report(seg["phase"])
        return g

    def CloseGripper(self):
        """Close the fingers -- if the object can be grabbed: its axis must pass through the pad centre
        (within 8 mm) and the opening must be at least the object's width.  After closing, the pad gap must
        match the object's width; otherwise the fingers missed it, they are opened again and GraspFailed is
        raised."""
        cur = self.current
        if cur is None:
            raise GraspFailed("CloseGripper: no object selected (call MoveToObject first)")
        i, g, o = cur["i"], cur["grasp"], self.objects[cur["i"]]
        p, s = self.obj_pos(i), self.grasp_pos()
        off = float(np.linalg.norm(s[:2] - p[:2]))
        z_rel = s[2] - self.table_z
        gap0 = self.pad_gap()
        if off > 0.008 or not (0.005 < z_rel < o["height"]):
            raise GraspFailed(f"cannot grab {o['name']}: the pad centre is {off * 1000:.0f} mm off its axis "
                              f"and {z_rel * 100:.1f} cm above the table (need < 8 mm, inside its {o['height'] * 100:.0f} cm)")
        if gap0 < g["width"] - 0.002:
            raise GraspFailed(f"cannot grab {o['name']}: the fingers are open {gap0 * 1000:.0f} mm, "
                              f"the object is {g['width'] * 1000:.0f} mm wide")
        self.set_gripper(False)
        self.wait(400)
        gap = self.pad_gap()
        self.last_grab = dict(off_mm=off * 1000, gap_before_mm=gap0 * 1000, gap_after_mm=gap * 1000,
                              width_mm=g["width"] * 1000)
        narrowest = 2 * o["size"]                            # a box at a diagonal azimuth turns into the pads
        if gap < narrowest - 0.004:
            self.set_gripper_width(0.07); self.wait(200)
            raise GraspFailed(f"the fingers closed to {gap * 1000:.0f} mm but {o['name']} is at least "
                              f"{narrowest * 1000:.0f} mm wide: missed")
        self.held = i
        self._report("close")
        if self.verbose:
            print(f"   CloseGripper: {o['name']} grabbed (pad gap {gap * 1000:.0f} mm, object {o['size'] * 2000:.0f}-{g['width'] * 1000:.0f} mm)")

    def Pick(self, name, destination=None, grasp_id=None, skip=()):
        """MoveToObject + CloseGripper.  Returns the grasp used."""
        g = self.MoveToObject(name, destination, grasp_id, skip)
        self.CloseGripper()
        return g

    def MoveToLocation(self, cell):
        """With the object in the gripper: move it to `cell` (row, col).  IK + RRT-Connect run first (the
        sideways route in the table plane, then IK along it); then lift 2 cm, carry, lower."""
        cur = self.current
        if cur is None or self.held != cur["i"]:
            raise GraspFailed("MoveToLocation: nothing is in the gripper (call CloseGripper first)")
        cell = tuple(int(v) for v in cell)
        i, name, g = cur["i"], cur["name"], cur["grasp"]
        if not (0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols):
            raise ValueError(f"cell {cell} is outside the {self.rows}x{self.cols} grid")
        for j, oj in enumerate(self.objects):
            if j != i and self.cell_of(j) == cell:
                raise ValueError(f"cell {cell} is occupied by {oj['name']}")
        segs, q_put, put, xy = self._plan_carry(name, cell, g, self.q_now())
        if self.verbose:
            how = "straight" if self.last_carry["direct"] else "RRT-Connect detour"
            print(f"   MoveToLocation{cell}: IK ok; carry route {len(xy)} waypoint(s) [{how}]")
        self._exec_segments(segs[:-1], i)                # lift, carry: the object is still 2 cm above the table
        self._trim_to_centre(i, cell, g)                 # centre it there (nothing drags over the table) ...
        # ... lower it until it just touches the table, centring again while it is still above it.  The
        # distance is measured (the object slides a little in the pads); the arm can stall on the way down
        # (links touching near the base), so lower, centre, and repeat until it is down.
        for _ in range(8):
            low = self._lowest_point(i)
            if low < 0.0015:
                break
            gap = 0.0005 if low < 0.006 else 0.004
            s = self.grasp_pos()
            ch = self.chain(self.q_now(), s, [np.array([s[0], s[1], s[2] - (low - gap)])], g["R"])
            self._exec_segments([dict(phase="lower", kind="cart", q=ch)], i)
            if self._lowest_point(i) > 0.0015:
                self._trim_to_centre(i, cell, g)
        cur["cell"] = cell

    def _lowest_point(self, i):
        """Height of the object's lowest point above the table (m): it hangs a little tilted in the pads."""
        o = self.objects[i]
        ax_z = float(self.data.xmat[self.model.body(o["name"]).id].reshape(3, 3)[2, 2])
        return float(self.obj_pos(i)[2] - self.table_z - 0.5 * o["height"] * abs(ax_z)
                     - footprint_radius(o) * math.sqrt(max(1.0 - ax_z * ax_z, 0.0)))

    def _trim_to_centre(self, i, cell, g, tries=4, max_move=0.04):
        """Still holding the object just above its cell.  The hand holds it a little tilted and a little off
        the pad centre, and when the fingers open it settles upright onto the centre of its BASE.  So measure
        where its base centre is (object centre minus half its height along its own axis) and shift the hand,
        in small straight moves with the orientation unchanged, until that point is on the cell centre."""
        h = self.objects[i]["height"]
        for _ in range(tries):
            axis = self.data.xmat[self.model.body(self.objects[i]["name"]).id].reshape(3, 3)[:, 2]
            base_xy = (self.obj_pos(i) - 0.5 * h * axis)[:2]
            err = self.cell_xy(cell) - base_xy
            n = float(np.linalg.norm(err))
            if n < 0.0005 or n > max_move:
                return
            s = self.grasp_pos()
            step = err * min(1.0, 0.012 / n)                 # small straight moves (12 mm at most) track well
            target = np.array([s[0] + step[0], s[1] + step[1], s[2]])
            try:
                ch = self.chain(self.q_now(), s, [target], g["R"])
            except RuntimeError:
                return
            self._exec_segments([dict(phase="trim", kind="cart", q=ch)])

    def OpenGripper(self, retreat=True):
        """Open the fingers (release the object).  By default the hand then backs away along the approach
        axis and goes to the ready pose, so the open fingers cannot drag the object when the arm moves on;
        retreat=False only opens."""
        cur = self.current
        self.set_gripper_width(cur["grasp"]["open_width"] if cur else 0.07)
        self.wait(300)
        self.held = None
        self._report("release")
        if retreat:
            self._retreat()
            self.current = None

    def _retreat(self):
        cur = self.current
        if cur is None:
            self.go_ready()
            return
        self._exec_segments(self._plan_retreat(cur["grasp"], self.q_now(), self.grasp_pos()))

    def Place(self, cell):
        """MoveToLocation + OpenGripper (which backs away to the ready pose), then check the object stayed put."""
        self.MoveToLocation(cell)
        i, name = self.current["i"], self.current["name"]
        self.OpenGripper()
        err_mm = np.linalg.norm(self.obj_pos(i)[:2] - self.cell_xy(cell)) * 1000
        zax = self.data.xmat[self.model.body(name).id].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(float(np.clip(zax[2], -1, 1))))
        if self.verbose:
            print(f"      placed {err_mm:.1f} mm from the cell centre")
        if self.cell_of(i) is None or tilt > 25:
            raise ObjectLost(f"{name} did not stay in cell {cell}: it ended {err_mm:.0f} mm from the cell centre, "
                             f"tilted {tilt:.0f} deg" + (" and off the grid" if self.cell_of(i) is None else "") +
                             ".  A tall slim object gripped with a tilted hand tips over when released")
        if err_mm > 20:
            raise GraspFailed(f"{name} ended {err_mm:.0f} mm from the centre of cell {cell}")
        return cell

    def _recover(self):
        """After a failed grasp: open, back away along the approach axis, go to the ready pose."""
        self.set_gripper_width(0.07); self.wait(200)
        self.held = None
        cur = self.current
        try:
            if cur is not None:
                self._exec_segments(self._plan_retreat(cur["grasp"], self.q_now(), self.grasp_pos()))
            else:
                self.go_ready()
        except RuntimeError:
            self.go_ready()
        self.current = None

    @_handles_object
    def PickAndPlace(self, name, cell, grasp_id=None):
        """Pick(name) + Place(cell), trying the next grasp if the fingers miss.  Raises ReachError when no
        grasp is feasible (the move is infeasible), ValueError for a bad / occupied cell."""
        cell = tuple(int(v) for v in cell)
        i = self.obj_id(name)
        o = self.objects[i]
        if not (0 <= cell[0] < self.rows and 0 <= cell[1] < self.cols):
            raise ValueError(f"cell {cell} is outside the {self.rows}x{self.cols} grid")
        if self.cell_of(i) == cell:
            return cell
        for j, oj in enumerate(self.objects):
            if j != i and self.cell_of(j) == cell:
                raise ValueError(f"cell {cell} is occupied by {oj['name']}")
        tried = set()
        while True:
            if self.cell_of(i) is None:
                raise ObjectLost(f"{o['name']} is no longer on the grid")
            g = self.MoveToObject(name, cell, grasp_id, skip=tried)     # raises ReachError when none is left
            try:
                self.CloseGripper()
                self.Place(cell)
                return cell
            except GraspFailed as err:
                if self.verbose:
                    print(f"   grasp #{g['id']} failed: {err}")
                tried.add(g["id"])
                self._recover()
                if grasp_id is not None:
                    raise
            except ReachError:
                self._recover()
                raise

    def move_object(self, name, cell, grasp_id=None):
        """Kept for old scripts: the same as PickAndPlace."""
        return self.PickAndPlace(name, cell, grasp_id)

    # -- what-if (planning only): would the move become feasible if ONE other object were not there?
    @_handles_object
    def what_if(self, name, cell):
        """{other object: True if `name` -> cell has a feasible grasp with only that object taken away}.
        The hover / approach plan does not depend on the other objects, so it is computed once per grasp;
        only the carry (and the retreat after it) is re-planned for each object that is taken away."""
        i = self.obj_id(name)
        o = self.objects[i]
        o["pos"] = self.obj_pos(i)
        q0 = self.q_now()
        approach = []
        for g in grasp_poses(o, self.table_z):
            try:
                approach.append((g, self._plan_approach(g, q0)[1]))
            except (RuntimeError, ValueError):
                pass
        out = {}
        for j, oj in enumerate(self.objects):
            if j == i:
                continue
            if self.console is not None:
                self.console.check_reset()
            a = self.obj_qadr[j]
            saved = self.data.qpos[a:a + 3].copy()
            self.data.qpos[a:a + 3] = [5.0 + j, 5.0, self.table_z]          # park it far away
            mujoco.mj_kinematics(self.model, self.data)
            try:
                out[oj["name"]] = False
                for g, q_g in approach:
                    try:
                        _, q_put, put, _ = self._plan_carry(name, cell, g, q_g)
                        self._plan_retreat(g, q_put, put)
                        out[oj["name"]] = True
                        break
                    except (RuntimeError, ValueError):
                        continue
            finally:
                self.data.qpos[a:a + 3] = saved
                mujoco.mj_kinematics(self.model, self.data)
        return out

    # -- reachability of the grid
    def reach_map(self, tilt_subset=None, n_azimuth=None):
        """Number of grasp poses (of N_GRASPS, or a subset) whose hover + grasp pose is reachable, per cell."""
        out = np.zeros((self.rows, self.cols), int)
        ref = dict(self.objects[0]); ref["pos"] = np.zeros(3)
        for r in range(self.rows):
            for c in range(self.cols):
                ref["pos"] = np.array([*self.cell_xy((r, c)), self.table_z + ref["height"] / 2])
                n = 0
                for g in grasp_poses(ref, self.table_z):
                    if tilt_subset is not None and g["tilt_deg"] not in tilt_subset:
                        continue
                    if n_azimuth is not None and (g["azimuth_deg"] / (360.0 / N_AZIMUTH)) % (N_AZIMUTH // n_azimuth) != 0:
                        continue
                    pre = g["position"] - g["approach"] * PRE_DIST
                    qa = self.ik_pose(pre, g["R"], self.q_ready, first=True)
                    if qa is None:
                        continue
                    try:
                        self.chain(qa, pre, [g["position"]], g["R"], step=0.01)
                        n += 1
                    except RuntimeError:
                        pass
                out[r, c] = n
        return out

    # -- dataset: one trajectory per grasp pose
    def generate_dataset(self, name, goal_cell, out_dir="dataset", hz=50, seed=0):
        """For every one of the grasp poses, plan the complete pick -> carry -> place -> home
        trajectory (kinematically; nothing is simulated) and save it.  Files in out_dir:
            grasps.csv       the poses and whether each one is feasible (and why not)
            summary.json     scene + settings
            traj_XX.npz      t, q (N,7), ee_pos (N,3), ee_quat (N,4 wxyz), gripper_width (N),
                             phase (N, int), phase_names, object_xyz (N,3)  -- one per feasible grasp"""
        os.makedirs(out_dir, exist_ok=True)
        i = self.obj_id(name)
        o = dict(self.objects[i]); o["pos"] = self.obj_pos(i)
        goal_cell = tuple(goal_cell)
        grasps = grasp_poses(o, self.table_z)
        dt = self.model.opt.timestep
        stride = max(1, int(round(1.0 / (hz * dt))))
        rows, n_ok = [], 0
        for g in grasps:
            plan = self.plan_pick_place(name, goal_cell, g, q_start=self.q_ready, seed=seed)
            rows.append(dict(id=g["id"], azimuth_deg=g["azimuth_deg"], tilt_deg=g["tilt_deg"],
                             **{f"pos_{k}": round(float(v), 5) for k, v in zip("xyz", g["position"])},
                             **{f"quat_{k}": round(float(v), 5) for k, v in zip("wxyz", g["quat"])},
                             **{f"approach_{k}": round(float(v), 5) for k, v in zip("xyz", g["approach"])},
                             width_m=round(float(g["width"]), 4), feasible=plan["feasible"],
                             reason=plan["reason"]))
            if not plan["feasible"]:
                continue
            n_ok += 1
            Q, PH, GW, OB = [], [], [], []
            names = []
            width = self._grip_cal[0][-1]
            obj_xyz = o["pos"].copy()
            offs = None
            for seg in plan["segments"]:
                if seg["phase"] not in names:
                    names.append(seg["phase"])
                ph = names.index(seg["phase"])
                if seg["kind"] == "grip":
                    width = seg["grip"] if seg["grip"] is not None else g["shut_width"]
                    for _ in range(int(0.3 / (dt * stride))):             # 0.3 s hold at the grip event
                        Q.append(Q[-1] if Q else self.q_ready); PH.append(ph); GW.append(width)
                        OB.append(obj_xyz.copy())
                    continue
                tr = self.segment_traj(seg)[::stride]
                for q in tr:
                    Q.append(q); PH.append(ph); GW.append(width)
                    if seg["phase"] in ("lift", "carry", "lower"):
                        p, _ = self._fk(q)
                        if offs is None:
                            offs = o["pos"] - p
                        obj_xyz = p + offs
                    OB.append(obj_xyz.copy())
            Q = np.array(Q)
            ee = [self._fk(q) for q in Q]
            np.savez_compressed(os.path.join(out_dir, f"traj_{g['id']:02d}.npz"), t=np.arange(len(Q)) / hz, q=Q,
                                ee_pos=np.array([p for p, _ in ee]), ee_quat=np.array([r for _, r in ee]),
                                gripper_width=np.array(GW), phase=np.array(PH), phase_names=np.array(names),
                                object_xyz=np.array(OB), grasp_id=g["id"], azimuth_deg=g["azimuth_deg"],
                                tilt_deg=g["tilt_deg"], grasp_pos=g["position"], grasp_quat=g["quat"],
                                carry_xy=np.array(plan["carry_xy"]), hz=hz)
        with open(os.path.join(out_dir, "grasps.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        meta = dict(object=name, start_cell=self.cell_of(i), goal_cell=goal_cell, hz=hz, n_grasps=len(grasps),
                    n_feasible=n_ok, table_z=self.table_z, grid_x0=self.x0, pitch=self.pitch,
                    objects=[dict(name=ob["name"], shape=ob["shape"], size=ob["size"], height=ob["height"],
                                  color=str(ob["color"]), cell=self.cell_of(ob["name"])) for ob in self.objects])
        json.dump(meta, open(os.path.join(out_dir, "summary.json"), "w"), indent=2, default=list)
        if self.verbose:
            print(f"dataset: {n_ok}/{len(grasps)} grasp poses feasible -> {out_dir}/")
        return meta

    def _fk(self, q):
        d = self.ik_data
        d.qpos[:] = self.data.qpos; d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.site_id].copy(), rot_to_quat(d.site_xmat[self.site_id].reshape(3, 3))

    # -- viewing
    def launch_viewer(self):
        import mujoco.viewer
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        self.viewer.cam.lookat[:] = [self.x0, 0, self.table_z]
        self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = 1.3, 180, -50

    def render(self, path="preview.png", w=960, h=720, lookat=None, distance=1.15, azimuth=180, elevation=-55):
        r = mujoco.Renderer(self.model, h, w)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = lookat if lookat is not None else [self.x0, 0, self.table_z]
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r.update_scene(self.data, cam)
        from PIL import Image
        Image.fromarray(r.render()).save(path)


# =============================================================================== running the settings
def user_objects():
    objs = [dict(TARGET, name="target", label="T", cell=TARGET_CELL)]
    for k, o in enumerate(OBSTACLES, 1):
        objs.append(dict(o, name=o.get("name", f"obstacle{k}")))
    return objs


# =============================================================================== typed commands: ask / reset / quit
class Console:
    """Typed input, read by a background thread so it is accepted at any time.
        reset   restores the simulation -- even in the middle of a move (ResetRequested is raised from
                the next simulation step, or from the prompt you are at)
        quit    ends the program (also 'exit')
    Everything else is an answer to the next question."""

    def __init__(self, start=True):
        self.lines, self.eof, self.reset_flag = [], False, False
        self.cv = threading.Condition()
        if start:
            threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        try:
            while True:
                line = sys.stdin.readline()
                if not line:
                    break
                self.feed(line)
        except (OSError, ValueError):
            pass
        with self.cv:
            self.eof = True
            self.cv.notify_all()

    def feed(self, line):
        s = line.strip()
        with self.cv:
            if s.lower() == "reset":
                self.reset_flag = True
            else:
                self.lines.append(s)
            self.cv.notify_all()

    def check_reset(self):
        if self.reset_flag:
            self.reset_flag = False
            raise ResetRequested()

    def take(self, block=False):
        """One typed line: '' if there is none yet, None at end of input."""
        with self.cv:
            while block and not self.lines and not self.reset_flag and not self.eof:
                self.cv.wait(0.2)
            self.check_reset()
            if self.lines:
                s = self.lines.pop(0)
                if s.lower() in ("quit", "exit"):
                    raise QuitRequested()
                return s
            return None if self.eof else ""

    def ask(self, prompt):
        print(prompt, end="", flush=True)
        s = self.take(block=True)
        if s is None:
            print()
            raise NoInput()
        if not sys.stdin.isatty():
            print(s)
        return s


def parse_cell(text):
    n = re.findall(r"-?\d+", text)
    return (int(n[0]), int(n[1])) if len(n) == 2 else None


def resolve_name(env, text):
    t = text.strip().lower()
    if t in env.obj_index:
        return t
    if t in ("t", "target"):
        return "target"
    if t.isdigit() and f"obstacle{t}" in env.obj_index:
        return f"obstacle{t}"
    return None


def show_hint(env, name, cell):
    print(f"  testing, one object at a time, whether {name} -> {tuple(cell)} becomes feasible if ONLY that object "
          f"were gone (planning only, nothing moves; about half a minute) ...")
    res = env.what_if(name, cell)
    helps = [k for k, v in res.items() if v]
    if helps:
        print("  removing it ALONE makes the move feasible: " + ", ".join(helps))
    else:
        print("  no single object is enough on its own")
    print("  removing it alone changes nothing (redundant by itself, or only matters together with others): " +
          (", ".join(k for k, v in res.items() if not v) or "-"))


def clear_the_way(env, console, name, cell, depth, goal):
    """The move is infeasible: ask which object to move out of the way, and where.  True once that move was
    done (the caller then tries the original move again); False if the user gives up on `name`."""
    pad = "  " * depth
    while True:
        print("\n" + env.grid_text(goal=goal))
        print(pad + "objects: " + ", ".join(f"{o['name']} {env.cell_of(o['name'])}" for o in env.objects))
        ans = console.ask(f"{pad}Which object should be moved out of the way first?  "
                          f"[name or number | hint | skip | reset | quit] > ").lower()
        if ans in ("skip", "s"):
            return False
        if ans == "hint":
            show_hint(env, name, cell)
            continue
        obj = resolve_name(env, ans)
        if obj is None:
            print(f"{pad}  no object called {ans!r}")
            continue
        if obj == name:
            print(f"{pad}  {name} is the object being moved; pick another one")
            continue
        dest = None
        while dest is None:
            t = console.ask(f"{pad}Where should {obj} go?  [row,col | back | reset | quit] > ").lower()
            if t == "back":
                break
            c = parse_cell(t)
            if c is None:
                print(f"{pad}  type the cell as row,col  (e.g. 7,4)")
            elif not (0 <= c[0] < env.rows and 0 <= c[1] < env.cols):
                print(f"{pad}  {c} is outside the {env.rows}x{env.cols} grid")
            elif any(env.cell_of(o["name"]) == c for o in env.objects):
                print(f"{pad}  cell {c} is occupied by {[o['name'] for o in env.objects if env.cell_of(o['name']) == c][0]}")
            else:
                dest = c
        if dest is None:
            continue
        if do_move(env, console, obj, dest, depth + 1, goal):       # may itself ask again if THIS is infeasible
            return True
        print(f"{pad}({obj} was not moved)")


def do_move(env, console, name, cell, depth=0, goal=None):
    """PickAndPlace(name, cell).  If it is infeasible, ask which object to move out of the way and where
    (clear_the_way), then try again.  True when `name` ends up in `cell`."""
    pad = "  " * depth
    while True:
        try:
            env.PickAndPlace(name, cell)
            return True
        except ObjectLost as err:
            print(f"\n{pad}FAILED: {err}\n{pad}  moving other objects cannot fix this -- type 'reset' to restore the simulation")
            return False
        except (ReachError, GraspFailed) as err:
            print(f"\n{pad}INFEASIBLE: {name} -> {tuple(cell)}\n{pad}  {err}")
        except (ValueError, KeyError) as err:
            print(f"\n{pad}CANNOT MOVE {name} -> {tuple(cell)}: {err}")
            return False
        if console is None:
            return False
        try:
            if not clear_the_way(env, console, name, cell, depth, goal):
                return False
        except NoInput:
            print(f"{pad}(no input available to ask for help)")
            return False


def run_session(env, console):
    """MOVE_COMMANDS, then the target to GOAL_CELL.  True if the target ends in GOAL_CELL."""
    goal = tuple(GOAL_CELL)
    for name, cell in MOVE_COMMANDS:
        if not do_move(env, console, name, tuple(cell), goal=goal):
            print(f"({name} -> {tuple(cell)} was not done)")
    if do_move(env, console, "target", goal, goal=goal):
        env.go_ready()
    return env.cell_of("target") == goal


def idle(env, console):
    """After a run: wait for 'reset' (start over) or 'quit'.  The viewer keeps running meanwhile."""
    print("\ntype 'reset' to restore the simulation and run again, or 'quit' to leave")
    while True:
        if env.viewer is not None:
            if not env.viewer.is_running():
                return
            env.step(10)
            line = console.take()
        else:
            line = console.take(block=True)
        if line is None and env.viewer is None:
            return


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--once", action="store_true", help="exit when the run is over (do not wait for 'reset' / 'quit')")
    ap.add_argument("--reach", action="store_true", help="print how many of the grasps reach each cell")
    ap.add_argument("--dataset", metavar="OBJECT", help="export one trajectory per grasp pose for OBJECT -> GOAL_CELL")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--render", metavar="PREFIX")
    a = ap.parse_args()

    env = XArm7TableEnv(user_objects())
    for o in env.objects:
        print(f"{o['name']:<10} {o['shape']:<8} {2 * o['size'] * 100:4.1f} cm wide {o['height'] * 100:4.1f} cm tall "
              f"cell {env.cell_of(o['name'])}")
    print(f"\ngoal {GOAL_CELL}\n" + env.grid_text(goal=tuple(GOAL_CELL)) + "\n")
    if a.reach:
        m = env.reach_map()
        print(f"grasp poses (of {N_GRASPS}) that reach each cell (rows from far to near):")
        print(m[::-1])
        print(f"cells with no feasible grasp: {int((m == 0).sum())} of {m.size}")
        return
    if a.dataset:
        env.generate_dataset(a.dataset, GOAL_CELL, a.out, a.hz)
        return
    console = Console()
    env.console = console
    if not a.headless:
        env.launch_viewer()
    if a.render:
        env.render(f"{a.render}_before.png")
    ok = False
    while True:
        t0 = time.time()
        try:
            ok = run_session(env, console)
            print("\n" + env.grid_text(goal=tuple(GOAL_CELL)))
            print(f"\n{'DONE' if ok else 'NOT DONE'}: target in {env.cell_of('target')} (goal {tuple(GOAL_CELL)}), {time.time() - t0:.0f}s")
            if a.render:
                env.render(f"{a.render}_after.png")
            if a.once:
                break
            idle(env, console)
            break
        except ResetRequested:
            env.reset_all()
            print("\n*** simulation reset: every object is back in its starting cell, the arm is at the ready pose ***")
            print(env.grid_text(goal=tuple(GOAL_CELL)) + "\n")
        except QuitRequested:
            break
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
