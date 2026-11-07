#!/usr/bin/env python3
"""
shelf_scene.py -- xArm7 + shelves / a factory workbench with a grid of cells.  Same logic as table_scene.py
(MoveToObject, CloseGripper, MoveToLocation, OpenGripper, PickAndPlace; asks you what to move when a move is
infeasible; 'reset' / 'quit'), but the open table is replaced by one of these scenes:

    confined     a box-shaped shelf: floor, back wall, two side walls, ceiling -- open only at the front
    multilevel   three wooden boards on four posts (bottom / middle / top level), open on all sides
    factory      an electronics workbench: incoming / test / outgoing zones, camera gantry, conveyor, tool board
    table        the original open table (unchanged behaviour)

    python shelf_scene.py                          # SCENE below, with the viewer
    python shelf_scene.py --scene multilevel       # pick a scene
    python shelf_scene.py --scene factory --headless --once
    python shelf_scene.py --scene confined --reach # how many grasps reach each cell

Put it next to xarm7_cabinet_env.py, rrt_connect.py, rrt_star.py and mujoco_menagerie/.

CELLS.  (row, col) -- or (row, col, level) on the multi-level shelf.  Row 0 is the row nearest the robot, col 0 the
-y edge, level 0 the lowest board.  Every level has the same rows x cols grid.  An object stays on its own level
(a move to another level is refused); within a level it can go to any free cell.

WHAT IS CHECKED.  As in table_scene.py: inverse kinematics reaches the pose and the whole path (joint limits, no
jump between IK branches), and a 2-D footprint rule keeps the carried object and the hand away from the other
objects.  NEW for the shelf scenes: the arm and hand are also checked against the STRUCTURE (boards, walls,
ceiling, posts, gantry, bench) with MuJoCo's collision query, at every IK waypoint and every free move, with
3 mm clearance and the fingers fully open.  Without it a top-down grasp would be planned straight through a
ceiling.  The arm and open hand keep OBJECT_MARGIN from the objects other than the one being handled (shelf scenes:
on every IK waypoint and free move; open scenes: on free moves); the carried object is checked against the other
objects by the footprint rule, not against the structure (scene set-up checks that objects fit under the ceiling).

MOVES CHAIN.  The arm starts at the ready pose; after it puts an object down it backs away along the approach axis
and goes straight on to the next pick from there -- it does not return to the ready pose (RETURN_TO_READY = True
brings the old behaviour back).

GRASPS.  confined: the hand is level with the shelf floor (tilt 90 deg from vertical, "0 deg"), 8 azimuths every 45
deg, those facing the robot first.  multilevel: the hand comes from the front side -- azimuths 180 +- 45 deg (7), tilts
75 and 60 deg from vertical (nearly horizontal), the flatter one preferred.  factory / table: all 12 azimuths and tilts
0 / 15 / 30 / 45 / 75 deg, from above first.  See table_scene.py for how a pose is defined.

STRUCTURE (methods of XArm7TableEnv; each one that moves the arm first runs IK + RRT-Connect, then moves):
    MoveToObject(name)     arm to the object's cell, fingers open around it
    CloseGripper()         close, if the object can be grabbed (and check it is held)      } Pick(name)
    MoveToLocation(cell)   with the object held: lift, carry sideways, lower, at the given cell
    OpenGripper()          release, then back away (the next move starts from there)        } Place(cell)
    PickAndPlace(name, cell) = Pick + Place (next grasp if the fingers miss).
If a move is infeasible the program asks which object to move out of the way first and where to (type 'hint' to
test which single objects matter), then tries again.  Type 'reset' at any time -- also while the arm is
moving -- to restore the simulation and start over; 'quit' to leave.
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
SCENE = "confined"                  # "confined" | "multilevel" | "factory" | "table"     (or: --scene NAME)

# Per scene.  Cells are (row, col) -- or (row, col, level) on the multi-level shelf.  Row 0 is the row NEAREST the
# robot, col 0 is the -y edge, level 0 is the lowest board.  size = radius / half-width (m); shape "box" |
# "cylinder"; colour name, "#rrggbb" or "r g b a".  Objects are named "target", "obstacle1", "obstacle2", ... (the
# order of the list).  move_commands run BEFORE the target moves: (object name, cell) = the arm moves that object.
# An empty list = the arm only moves the target.
SETTINGS = {
    "confined": dict(
        target_cell=(2, 3), goal_cell=(0, 1),
        target=dict(shape="cylinder", size=0.018, height=0.16, color="green"),
        obstacles=[dict(cell=(2, 1), shape="cylinder", size=0.018, height=0.15, color="blue"),
                   dict(cell=(3, 4), shape="cylinder", size=0.018, height=0.17, color="yellow"),
                   dict(cell=(1, 5), shape="cylinder", size=0.018, height=0.16, color="red"),
                   dict(cell=(3, 2), shape="cylinder", size=0.018, height=0.16, color="red")],
        # add  dict(cell=(1, 3), shape="cylinder", size=0.018, height=0.16, color="red")  (right in front of the target)
        # to block it: the program then asks which object to move out of the way
        move_commands=[]),
    "multilevel": dict(
        target_cell=(2, 3, 1), goal_cell=(0, 2, 1),
        target=dict(shape="cylinder", size=0.027, height=0.075, color="green"),
        obstacles=[dict(cell=(1, 1, 1), shape="cylinder", size=0.027, height=0.075, color="yellow"),
                   dict(cell=(2, 5, 1), shape="cylinder", size=0.027, height=0.075, color="red"),
                   dict(cell=(1, 1, 0), shape="cylinder", size=0.027, height=0.075, color="red"),
                   dict(cell=(2, 4, 0), shape="cylinder", size=0.027, height=0.075, color="blue"),
                   dict(cell=(0, 2, 2), shape="cylinder", size=0.027, height=0.075, color="yellow"),
                   dict(cell=(1, 4, 2), shape="cylinder", size=0.027, height=0.075, color="red")],
        # add  dict(cell=(1, 3, 1), shape="cylinder", size=0.027, height=0.075, color="red")  (right in front of the target)
        # to block it.  Objects stay on their own level.
        move_commands=[]),
    "factory": dict(
        target_cell=(0, 1), goal_cell=(5, 6),
        target=dict(shape="box", size=0.012, height=0.06, color="green"),          # the module to be tested
        obstacles=[dict(cell=(1, 2), shape="box", size=0.012, height=0.05, color="#444444"),       # ICs
                   dict(cell=(2, 4), shape="box", size=0.012, height=0.05, color="#444444"),
                   dict(cell=(3, 3), shape="cylinder", size=0.011, height=0.05, color="blue"),     # capacitors
                   dict(cell=(4, 5), shape="cylinder", size=0.011, height=0.05, color="blue"),
                   dict(cell=(3, 6), shape="box", size=0.014, height=0.07, color="red"),           # relays
                   dict(cell=(2, 1), shape="box", size=0.014, height=0.045, color="yellow")],      # connectors
        move_commands=[]),
    "table": dict(
        target_cell=(0, 6), goal_cell=(7, 7),
        target=dict(shape="cylinder", size=0.020, height=0.16, color="green"),
        obstacles=[dict(cell=(0, 7), shape="box", size=0.014, height=0.15, color="blue"),
                   dict(cell=(1, 6), shape="box", size=0.014, height=0.17, color="yellow"),
                   dict(cell=(0, 5), shape="box", size=0.016, height=0.16, color="red")],
        move_commands=[]),
}
TARGET_CELL = GOAL_CELL = TARGET = OBSTACLES = MOVE_COMMANDS = None      # filled in by select_scene()


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
# SHELF66V3 (2026-11-05): the planner checks the fingers PRE-SHAPED (the grasp's open_width), and every pick pre-shapes them before
# its first motion.  Before, the checks used the fingers as they happened to be: FULLY open (~8.5 cm) right after reset(), so every
# plan made at the start of a scene (the teacher's original query, its edits, its relocation checks) saw a hand several cm too wide
# and called feasible moves infeasible (2,500-run audit: 16 of 30 NONE and 5 of 20 K1 were in fact K0).
GRIP_JOINTS = ("left_driver_joint", "left_finger_joint", "left_inner_knuckle_joint",
               "right_driver_joint", "right_finger_joint", "right_inner_knuckle_joint")
GRASP_AT = "middle,top"     # 6x6 shelf: pads at mid-height first; near the top (GRASP_TOP_MARGIN) where mid-height has no plan
GRASP_TOP_MARGIN = 0.012            # pads grip the TOP of the object: pad centre = top - this (m).  Keeps the
                                    # wide gripper body above neighbours; min pad centre is 1.5 cm above the table
HAND_FOOTPRINT_RULE = True          # carry rule uses the measured hand footprint (False: the object only)
PRE_DIST = 0.13                     # hover this far back along the approach axis (m)
MIN_HEIGHT = 0.035
HOME_HOVER = 0.30                   # "ready" pose: this far above the table, hand pointing down
NO_OVERPASS = True                  # the arm and gripper only go AROUND the obstacles, never over them: every object (except the one being
                                    # handled) stands in a vertical column from its shelf up to COLUMN_H; no part of the arm / hand may enter
                                    # it; the carry route counts the hand at every height
ARM_CEILING = True                  # NO_OVERPASS: the arm's ceiling is the tallest obstacle: over the shelf no part of the arm / hand may be
CEILING_MARGIN = 0.10               # higher than the tallest object's top + this (m: the hand's half thickness, the 2.5 cm carry lift)
DIRECT_MOVES = False                # (tried 2026-11-03, rejected by the user) short retreat + straight hand path to the next hover pose
RETREAT_DIST = 0.05                 # NO_OVERPASS scenes: after letting go the hand backs off only this far (m), just enough for the open fingers to
                                    # clear the object, and then goes straight to the next object
COLUMN_H = 1.2                      # column height above the surface the object stands on (m)
RETURN_TO_READY = False             # after putting an object down the hand backs away and the arm goes straight on to
                                    # the next pick (True: back to the ready pose after every move, the old behaviour)

N_AZIMUTH = 12                      # approach directions, evenly spaced around the object
TILTS_DEG = (0.0, 15.0, 30.0, 45.0, 75.0)   # 0 = straight down (top grasp); 75 = nearly horizontal (front grasp).  60 tips tall objects over
N_GRASPS = N_AZIMUTH * len(TILTS_DEG)   # grasp poses per object (any number: edit N_AZIMUTH / TILTS_DEG)
AZIMUTHS_DEG = [360.0 / N_AZIMUTH * k for k in range(N_AZIMUTH)]   # (replaced by the scene's list, see apply_scene_grasps)
PREFER_HIGH_TILT = False            # shelves: the flattest (front) grasps first;  table / factory: from above first

PLAN_VMAX, PLAN_AMAX = 1.0, 1.5     # joint limits for free moves
CART_VMAX, CART_AMAX = 0.5, 1.0     # slower limits for approach / carry / retreat


# Footprint of the xArm gripper (closed on an object), measured from the model with a top-down grasp.
# Bands of height above the pad centre (m) -> half-extent along the closing axis (m, depends on the pad
# gap) and across it.  The fingers are narrow; the linkage 3-5 cm above the pads is the widest part.
HAND_BANDS = [(-0.03, 0.00), (0.00, 0.02), (0.02, 0.03), (0.03, 0.04), (0.04, 0.05), (0.05, 0.30)]
_HAND_EY = [lambda g: g / 2 + 0.016, lambda g: g / 2 + 0.026, lambda g: g / 2 + 0.036,
            lambda g: 0.072, lambda g: g / 2 + 0.050, lambda g: 0.052]
_HAND_EX = [0.017, 0.017, 0.017, 0.026, 0.029, 0.029]
LEVEL_TILT_DEG = 80.0               # from this tilt on the hand counts as level: the bands lie flat behind the pads


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
    zs = {"middle": table_z + h / 2, "top": table_z + max(0.015, h - GRASP_TOP_MARGIN)}     # pad-centre heights
    modes = [x.strip() for x in GRASP_AT.split(",")]        # GRASP_AT: "top" (default), "middle", or a list tried in order, e.g. "middle,top"
    to_robot = math.degrees(math.atan2(robot_xy[1] - p[1], robot_xy[0] - p[0]))
    out = []
    for mi, mode in enumerate(modes):
        for tilt in TILTS_DEG:
            for az in AZIMUTHS_DEG:
                R = grasp_frame(az, tilt)
                c = R[:, 1]
                width = 2 * obj["size"] if obj["shape"] == "cylinder" else \
                    2 * obj["size"] * (abs(c[0]) + abs(c[1]))
                out.append(dict(azimuth_deg=az, tilt_deg=tilt, R=R, quat=rot_to_quat(R), approach=R[:, 2],
                                closing=c, position=np.array([p[0], p[1], zs[mode]]), width=width, grasp_level=mode, _mi=mi,
                                open_width=width + 2 * FINGER_CLEARANCE, shut_width=max(width - 0.001, 0.0)))
    # preference: the first grasp height first (GRASP_AT order), then top-down first, then the hand coming from the robot's side
    wrap = lambda d: abs((d + 180.0) % 360.0 - 180.0)
    out.sort(key=lambda g: (g["_mi"], -g["tilt_deg"] if PREFER_HIGH_TILT else g["tilt_deg"], wrap(g["azimuth_deg"] - to_robot)))
    for i, g in enumerate(out):
        g["id"] = i
    return out


# =============================================================================== scenes
# A scene = the static structure (shelf boards, walls, posts, bench, fixtures), one or more LEVELS (work
# surfaces, each with the same rows x cols grid), the ready pose, the grasp family and the camera.
STATIC_MARGIN = 0.003               # clearance the arm / hand keeps to the shelf structure (m)
OBJECT_MARGIN = 0.005               # ... and to the other objects (m)
FREE_MOVE_MARGIN = 0.02             # ... in the middle of a free (joint-space) move, away from its ends (m)
_FRIC = 'friction="1.0 0.02 0.002"'


def _slab(name, x0, x1, y0, y1, z0, z1, rgba, collide=True, friction=False, material=None):
    """Box given by its corner coordinates.  `material` (see build_scene_xml) only changes how it looks: the colour
    stays `rgba`, the material adds texture and gloss."""
    flags = ("" if collide else ' contype="0" conaffinity="0"') + (f' material="{material}"' if material else "")
    return (f'<geom name="{name}" type="box" pos="{(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {(z0 + z1) / 2:.4f}" '
            f'size="{(x1 - x0) / 2:.4f} {(y1 - y0) / 2:.4f} {(z1 - z0) / 2:.4f}" rgba="{rgba}"{flags}'
            f'{" " + _FRIC if friction else ""}/>')


def _cyl(name, x, y, z0, z1, r, rgba, collide=True, along_y=False):
    flags = "" if collide else ' contype="0" conaffinity="0"'
    if along_y:
        return (f'<geom name="{name}" type="cylinder" pos="{x:.4f} {(y[0] + y[1]) / 2:.4f} {z0:.4f}" euler="1.5708 0 0" '
                f'size="{r:.4f} {(y[1] - y[0]) / 2:.4f}" rgba="{rgba}"{flags}/>')
    return (f'<geom name="{name}" type="cylinder" pos="{x:.4f} {y:.4f} {(z0 + z1) / 2:.4f}" '
            f'size="{r:.4f} {(z1 - z0) / 2:.4f}" rgba="{rgba}"{flags}/>')


def norm_cell(cell, level=0):
    """(row, col) or (row, col, level) -> (row, col, level)."""
    c = tuple(int(v) for v in cell)
    return c + (level,) if len(c) == 2 else c[:3]


def scene_table():
    """The original open table (no structure): hand mostly from above, no structure check."""
    rows, cols, pitch, x0, z = ROWS, COLS, PITCH, GRID_X0, TABLE_Z
    hx, hy = rows * pitch / 2 + 0.08, cols * pitch / 2 + 0.08
    statics = [_slab("table_top", x0 - hx, x0 + hx, -hy, hy, z - 0.04, z, "0.72 0.58 0.42 1", friction=True)]
    for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        statics.append(_slab(f"leg{k}", x0 + sx * (hx - 0.03) - 0.02, x0 + sx * (hx - 0.03) + 0.02,
                             sy * (hy - 0.03) - 0.02, sy * (hy - 0.03) + 0.02, 0.0, z - 0.04, "0.45 0.35 0.25 1"))
    return dict(name="table", rows=rows, cols=cols, pitch=pitch, x0=x0, levels=[dict(z=z, clearance=None, name="table top")],
                statics=statics, decals=[], ready_p=[x0 - 0.12, 0.0, z + HOME_HOVER], ready_az=0.0, ready_tilt=0.0,
                azimuths=[360.0 / N_AZIMUTH * k for k in range(N_AZIMUTH)], tilts=TILTS_DEG, prefer_high_tilt=False,
                static_check=False, camera=dict(lookat=[x0, 0, z], distance=1.3, azimuth=180, elevation=-50))


SHELF66 = dict(rows=6, cols=6, pitch=0.075, x0=0.42, zb=0.55, side_walls=False, az_step=22.5)
# 2026-11-04 design (measured with the real planner, no-overpass columns + arm ceiling): the 8 x 8 / 6.5 cm shelf at x0 0.54, floor 0.50
# left rows 4-7 mostly unreachable; 6 x 6 at 7.5 cm, 12 cm closer, floor 5 cm higher, no side walls, 16 level grasp directions,
# mid-height grip with a near-the-top fallback: every cell picks with >= 6 directions and places with >= 4.


def scene_confined(clr=0.60):
    """6x6 VERSION (SHELF66): floor, back wall and ceiling, open at the front and at both sides (no side walls).
    Original text: a box-shaped shelf: floor, back wall, two side walls and a ceiling; open only at the front (as in the
    left picture).  The hand comes in level with the floor.  8 x 8 cells, sized so that every cell can be reached
    with the floor at 0.50 m: the rows span x = 0.31 .. 0.77 m (at this height the level hand reaches about 0.80 m,
    hence the 6.5 cm pitch), and the clear height of 0.60 m leaves room for the elbow above the middle rows.  The
    straight-in grasp (azimuth 180) reaches all 64 cells, so every cell can be carried to every other cell."""
    rows, cols, pitch, x0 = SHELF66["rows"], SHELF66["cols"], SHELF66["pitch"], SHELF66["x0"]
    zb = SHELF66["zb"]                                # shelf floor height
    xf, xb = x0 - rows * pitch / 2 - 0.04, x0 + rows * pitch / 2 + 0.04       # front edge, inner back wall
    hw, t = cols * pitch / 2 + 0.14, 0.02                         # inner half width (the arm fits at the outer columns)
    white, grey = "0.80 0.79 0.76 1", "0.33 0.35 0.39 1"          # warm white laminate; painted steel cabinet
    statics = [
        _slab("base", xf, xb + t, -hw - t, hw + t, 0.0, zb - 0.03, grey, material="steel"),
        _slab("floor_board", xf, xb + t, -hw - t, hw + t, zb - 0.03, zb, white, friction=True, material="laminate"),
        _slab("back_wall", xb, xb + t, -hw - t, hw + t, zb, zb + clr + 0.03, white, material="laminate"),
    ] + ([] if not SHELF66["side_walls"] else [      # 6x6 shelf: open at the left and right (no side walls)
        _slab("side_wall_L", xf, xb + t, hw, hw + t, zb, zb + clr, white, material="laminate"),
        _slab("side_wall_R", xf, xb + t, -hw - t, -hw, zb, zb + clr, white, material="laminate"),
    ]) + [
        _slab("ceiling", xf, xb + t, -hw - t, hw + t, zb + clr, zb + clr + 0.03, white, material="laminate"),
        _slab("front_step", max(xf - 0.14, 0.10), xf, -hw - t, hw + t, 0.0, 0.08, grey, material="steel"),   # kept low: the arm passes over it; clear of the robot base
    ]
    # looks only: an LED strip under the front edge of the ceiling, lighting the shelf floor
    lights = [f'<light name="shelf_led" type="spot" pos="{xf + 0.03:.3f} 0 {zb + clr - 0.01:.3f}" dir="0.45 0 -1" '
              f'cutoff="75" exponent="2" diffuse="0.06 0.06 0.055" specular="0 0 0" castshadow="false"/>']
    return dict(name="confined", rows=rows, cols=cols, pitch=pitch, x0=x0,
                levels=[dict(z=zb, clearance=clr, name="shelf")], statics=statics, decals=[], lights=lights,
                ready_p=[xf - 0.05, 0.0, zb + 0.20], ready_az=180.0, ready_tilt=0.0,      # hand down, just outside the shelf
                azimuths=[SHELF66["az_step"] * k for k in range(int(round(360 / SHELF66["az_step"])))], tilts=(90.0,), prefer_high_tilt=True,
                static_check=True, camera=dict(lookat=[x0, 0, zb + 0.12], distance=1.45, azimuth=35, elevation=-24))


def scene_multilevel(spacing=0.31, z0=0.09):
    """Three wooden shelf boards on four black posts, open on all sides (as in the right picture).  Objects stay on
    their own level; the hand comes in from the front between the boards."""
    rows, cols, pitch, x0 = 3, 6, 0.08, 0.58
    zs, board, top_gap = [z0, z0 + spacing, z0 + 2 * spacing], 0.03, spacing
    xf, xb = x0 - rows * pitch / 2 - 0.05, x0 + rows * pitch / 2 + 0.05
    hy = 0.35
    wood, post = "0.47 0.30 0.15 1", "0.10 0.10 0.10 1"
    statics = [_slab(f"board{k}", xf, xb, -hy, hy, z - board, z, wood, friction=True) for k, z in enumerate(zs)]
    statics.append(_slab("top_board", xf, xb, -hy, hy, zs[-1] + top_gap, zs[-1] + top_gap + board, wood))
    for k, (px, py) in enumerate([(xf + 0.03, 0.33), (xf + 0.03, -0.33), (xb - 0.03, 0.33), (xb - 0.03, -0.33)]):
        statics.append(_slab(f"post{k}", px - 0.015, px + 0.015, py - 0.015, py + 0.015, 0.0, zs[-1] + top_gap + board, post))
    return dict(name="multilevel", rows=rows, cols=cols, pitch=pitch, x0=x0,
                levels=[dict(z=z, clearance=top_gap - 0.03 if k == len(zs) - 1 else zs[k + 1] - board - z,
                             name=("bottom", "middle", "top")[k]) for k, z in enumerate(zs)],
                statics=statics, decals=[], ready_p=[xf - 0.20, 0.0, zs[1] + 0.15], ready_az=180.0, ready_tilt=90.0,
                azimuths=[135.0, 150.0, 165.0, 180.0, 195.0, 210.0, 225.0], tilts=(75.0, 60.0), prefer_high_tilt=True,
                static_check=True, camera=dict(lookat=[x0, 0, 0.36], distance=1.9, azimuth=60, elevation=-14))


def scene_factory():
    """An electronics assembly cell: workbench with incoming (green) / test (blue) / outgoing (orange) zones, a camera
    gantry over the test zone, a conveyor belt along one side, a tool board and a parts rack.  Open above the bench
    except for the gantry, so the hand can come from above or from the front."""
    rows, cols, pitch, x0, z = 6, 8, 0.07, 0.52, 0.10
    xa, xz, hy = x0 - rows * pitch / 2 - 0.10, x0 + rows * pitch / 2 + 0.10, cols * pitch / 2 + 0.22
    steel, dark, yellow = "0.55 0.58 0.62 1", "0.18 0.18 0.20 1", "0.90 0.75 0.10 1"
    statics = [_slab("bench_top", xa, xz, -hy, hy, z - 0.04, z, steel, friction=True),
               _slab("tool_board", xz, xz + 0.02, -hy, hy, z, z + 0.42, "0.25 0.30 0.38 1"),
               _slab("gantry_post_L", x0 - 0.02, x0 + 0.02, 0.44, 0.48, z, z + 0.62, yellow),
               _slab("gantry_post_R", x0 - 0.02, x0 + 0.02, -0.48, -0.44, z, z + 0.62, yellow),
               _slab("gantry_bar", x0 - 0.025, x0 + 0.025, -0.48, 0.48, z + 0.58, z + 0.62, yellow),
               _slab("camera_rod", x0 - 0.01, x0 + 0.01, -0.01, 0.01, z + 0.27, z + 0.58, dark),
               _slab("inspection_camera", x0 - 0.035, x0 + 0.035, -0.035, 0.035, z + 0.20, z + 0.27, dark),
               _slab("conveyor_belt", xa, xz, hy, hy + 0.20, 0.04, z - 0.01, dark),
               _slab("conveyor_rail_in", xa, xz, hy, hy + 0.012, z - 0.01, z + 0.03, "0.8 0.5 0.1 1"),
               _slab("conveyor_rail_out", xa, xz, hy + 0.188, hy + 0.20, z - 0.01, z + 0.03, "0.8 0.5 0.1 1")]
    for k, (sx, sy) in enumerate([(1, 1), (1, -1), (-1, 1), (-1, -1)]):
        lx, ly = (xz - 0.03 if sx > 0 else xa + 0.03), sy * (hy - 0.03)
        statics.append(_slab(f"bench_leg{k}", lx - 0.02, lx + 0.02, ly - 0.02, ly + 0.02, 0.0, z - 0.04, "0.35 0.37 0.40 1"))
    for k, ry in enumerate((hy + 0.05, hy + 0.10, hy + 0.15)):                     # conveyor rollers (visual)
        statics.append(_slab(f"conveyor_stripe{k}", xa + 0.05, xz - 0.05, ry - 0.004, ry + 0.004, z - 0.0095, z - 0.0085, "0.5 0.5 0.55 1", collide=False))
    zone = lambda name, r0, r1, c0, c1, rgba: _slab(
        name, x0 + (r0 - rows / 2) * pitch + 0.003, x0 + (r1 + 1 - rows / 2) * pitch - 0.003,
        (c0 - cols / 2) * pitch + 0.003, (c1 + 1 - cols / 2) * pitch - 0.003, z + 0.0002, z + 0.0006, rgba, collide=False)
    decals = [zone("zone_incoming", 0, 1, 0, 3, "0.25 0.65 0.35 1"), zone("zone_test", 2, 3, 3, 4, "0.25 0.45 0.85 1"),
              zone("zone_outgoing", 4, 5, 4, 7, "0.95 0.55 0.15 1")]
    for k, (py, col) in enumerate([(hy + 0.07, "0.2 0.6 0.9 1"), (hy + 0.13, "0.9 0.3 0.3 1")]):     # products on the belt
        decals.append(_slab(f"belt_item{k}", x0 - 0.10 + 0.14 * k, x0 - 0.06 + 0.14 * k, py - 0.025, py + 0.025, z - 0.01, z + 0.03, col, collide=False))
    for k in range(3):                                                                # parts rack, left of the bench (visual)
        decals.append(_slab(f"rack_shelf{k}", xa + 0.05, xz - 0.05, -hy - 0.20, -hy - 0.02, 0.10 + 0.13 * k, 0.12 + 0.13 * k, "0.35 0.37 0.40 1", collide=False))
        for j in range(4):
            decals.append(_slab(f"rack_bin{k}_{j}", xa + 0.08 + 0.12 * j, xa + 0.15 + 0.12 * j, -hy - 0.17, -hy - 0.05, 0.12 + 0.13 * k, 0.17 + 0.13 * k,
                                ("0.3 0.5 0.8 1", "0.8 0.3 0.3 1", "0.9 0.8 0.2 1", "0.3 0.7 0.4 1")[(j + k) % 4], collide=False))
    return dict(name="factory", rows=rows, cols=cols, pitch=pitch, x0=x0, levels=[dict(z=z, clearance=None, name="workbench")],
                statics=statics, decals=decals, ready_p=[x0 - 0.12, 0.0, z + HOME_HOVER], ready_az=0.0, ready_tilt=0.0,
                azimuths=[360.0 / N_AZIMUTH * k for k in range(N_AZIMUTH)], tilts=TILTS_DEG, prefer_high_tilt=False,
                static_check=True, camera=dict(lookat=[x0, 0.05, z + 0.05], distance=1.55, azimuth=38, elevation=-32))


SCENES = {"table": scene_table, "confined": scene_confined, "multilevel": scene_multilevel, "factory": scene_factory}


def make_scene(name):
    if name not in SCENES:
        raise ValueError(f"unknown scene {name!r}; choose from {list(SCENES)}")
    return SCENES[name]()


def apply_scene_grasps(scene):
    """The scene decides which grasp poses are generated (module level, read by grasp_poses)."""
    global AZIMUTHS_DEG, TILTS_DEG, PREFER_HIGH_TILT, N_GRASPS
    AZIMUTHS_DEG, TILTS_DEG, PREFER_HIGH_TILT = list(scene["azimuths"]), tuple(scene["tilts"]), bool(scene["prefer_high_tilt"])
    N_GRASPS = len(AZIMUTHS_DEG) * len(TILTS_DEG)


ROOM_BACK, ROOM_FRONT, ROOM_HALF_Y, ROOM_H = 1.0, -2.2, 2.2, 2.8   # the lab room (m): the setup stands against its back wall


def _room():
    """Looks only (nothing collides): the walls of the lab room around the setup, with skirting; the setup stands
    against the back wall.  No furniture or other objects."""
    xb, xf, hy, h = ROOM_BACK, ROOM_FRONT, ROOM_HALF_Y, ROOM_H
    wall, skirt = "0.90 0.89 0.86 1", "0.30 0.30 0.31 1"
    S = lambda name, box, rgba, mat="paint": _slab(name, *box, rgba, collide=False, material=mat)  # noqa: E731
    g = [S("room_wall_back", (xb, xb + 0.02, -hy, hy, 0, h), wall), S("room_wall_front", (xf - 0.02, xf, -hy, hy, 0, h), wall),
         S("room_wall_left", (xf, xb, hy, hy + 0.02, 0, h), wall), S("room_wall_right", (xf, xb, -hy - 0.02, -hy, 0, h), wall),
         S("skirt_back", (xb - 0.012, xb, -hy, hy, 0, 0.08), skirt), S("skirt_front", (xf, xf + 0.012, -hy, hy, 0, 0.08), skirt),
         S("skirt_left", (xf, xb, hy - 0.012, hy, 0, 0.08), skirt), S("skirt_right", (xf, xb, -hy, -hy + 0.012, 0, 0.08), skirt)]
    return g


def build_scene_xml(objects, scene, room=True, columns=False):
    """MuJoCo XML: the static structure, the grid lines of every level, and the objects as free bodies.  room=False
    leaves out the lab room's walls (looks only), for the planner's collision model."""
    geoms = list(scene["statics"]) + list(scene["decals"])
    line = 'type="box" rgba="0.15 0.12 0.1 1" contype="0" conaffinity="0"'
    x0, pitch, rows, cols = scene["x0"], scene["pitch"], scene["rows"], scene["cols"]
    for L, lv in enumerate(scene["levels"]):
        zc = lv["z"] + 0.0008
        for k in range(cols + 1):
            geoms.append(f'<geom name="grid{L}_col{k}" {line} pos="{x0} {(k - cols / 2) * pitch} {zc}" '
                         f'size="{rows * pitch / 2} 0.0008 0.0004"/>')
        for k in range(rows + 1):
            geoms.append(f'<geom name="grid{L}_row{k}" {line} pos="{x0 + (k - rows / 2) * pitch} 0 {zc}" '
                         f'size="0.0008 {cols * pitch / 2} 0.0004"/>')
    bodies = []
    for o in objects:
        x, y, _ = o["pos"]
        h2 = o["height"] / 2
        if columns:                         # looks-free, collision-free world geoms moved to the object's xy by _object_hits
            geoms.append(f'<geom name="{o["name"]}_col" type="cylinder" size="{footprint_radius(o)} {COLUMN_H / 2}" '
                         f'pos="{x} {y} {scene["levels"][o["level"]]["z"] + COLUMN_H / 2}" contype="0" conaffinity="0" rgba="1 0 0 0"/>')
        size = f'{o["size"]} {o["size"]} {h2}' if o["shape"] == "box" else f'{o["size"]} {h2}'
        bodies.append(
            f'<body name="{o["name"]}" pos="{x} {y} {scene["levels"][o["level"]]["z"] + h2 + 0.0005}"><freejoint name="{o["name"]}_free"/>'
            f'<geom name="{o["name"]}_g" type="{o["shape"]}" size="{size}" rgba="{_color(o["color"])}" material="plastic" '
            f'mass="{o["mass"]}" friction="{o["friction"]} 0.02 0.002"/></body>')
    # looks only (nothing here collides or changes the physics): a lab room (_room: dark porcelain tiles, the setup
    # against the back wall), a light sky -- lit like a room: a warm key light from the front above (it shines into open shelves and
    # is the only one casting shadows), a cool fill from the other side, four soft ceiling panels and a dim headlight;
    # the scene may add its own lights (scene["lights"])
    room = _room() if room else []
    nl = "\n    "
    return f"""
<mujoco model="xarm7_{scene['name']}_scene">
  <compiler angle="radian"/>
  <option timestep="0.002" integrator="implicitfast" cone="elliptic" impratio="5"/>
  <visual>
    <headlight diffuse="0.12 0.12 0.12" ambient="0.27 0.27 0.26" specular="0 0 0"/>
    <global offwidth="1280" offheight="960"/>
    <quality shadowsize="8192" offsamples="8"/>
    <map shadowclip="0.3" shadowscale="0.5"/>
  </visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.90 0.91 0.92" rgb2="0.62 0.64 0.67" width="512" height="3072"/>
    <texture name="floor" type="2d" builtin="checker" mark="edge" rgb1="0.30 0.30 0.31" rgb2="0.27 0.27 0.28" markrgb="0.18 0.18 0.18" width="512" height="512"/>
    <material name="floor" texture="floor" texrepeat="5 5" specular="0.35" shininess="0.6" reflectance="0.08"/>
    <texture name="laminate" type="2d" builtin="flat" mark="random" random="0.015" rgb1="1 1 1" markrgb="0.93 0.92 0.90" width="512" height="512"/>
    <material name="laminate" texture="laminate" texrepeat="2 2" specular="0.1" shininess="0.3"/>
    <material name="steel" specular="0.45" shininess="0.6" reflectance="0.03"/>
    <material name="paint" specular="0.05" shininess="0.1"/>
    <material name="plastic" specular="0.5" shininess="0.65"/>
    <!-- kept for older scripts that refer to it -->
    <texture name="grid" type="2d" builtin="checker" rgb1="0.25 0.32 0.4" rgb2="0.32 0.4 0.5" width="256" height="256"/>
    <material name="grid" texture="grid" texrepeat="6 6" reflectance="0"/>
  </asset>
  <worldbody>
    <light name="key" type="directional" pos="-1.0 0.8 2.6" dir="0.55 -0.3 -1" diffuse="0.58 0.55 0.49" specular="0.3 0.28 0.25" castshadow="true"/>
    <light name="fill" type="directional" pos="1.5 -1.5 2.0" dir="-0.5 0.6 -0.8" diffuse="0.24 0.26 0.30" specular="0 0 0" castshadow="false"/>
    <light name="ceiling0" type="spot" pos="-0.6 -1.0 2.9" dir="0 0 -1" cutoff="80" exponent="1" diffuse="0.20 0.20 0.19" specular="0.04 0.04 0.04" castshadow="false"/>
    <light name="ceiling1" type="spot" pos="-0.6 1.0 2.9" dir="0 0 -1" cutoff="80" exponent="1" diffuse="0.20 0.20 0.19" specular="0.04 0.04 0.04" castshadow="false"/>
    <light name="ceiling2" type="spot" pos="1.2 -1.0 2.9" dir="0 0 -1" cutoff="80" exponent="1" diffuse="0.20 0.20 0.19" specular="0.04 0.04 0.04" castshadow="false"/>
    <light name="ceiling3" type="spot" pos="1.2 1.0 2.9" dir="0 0 -1" cutoff="80" exponent="1" diffuse="0.20 0.20 0.19" specular="0.04 0.04 0.04" castshadow="false"/>
    {nl.join(scene.get("lights", []))}
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>
    <geom name="pedestal" type="cylinder" pos="0 0 {BASE_Z / 2}" size="0.06 {BASE_Z / 2}" rgba="0.22 0.23 0.25 1" material="steel"/>
    {nl.join(room)}
    {nl.join(geoms)}
    {nl.join(bodies)}
  </worldbody>
</mujoco>
"""


# =============================================================================== environment
class XArm7TableEnv:
    def __init__(self, objects, scene=None, verbose=True, xml_path=XARM7_XML):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"xarm7.xml not found at {xml_path} (clone mujoco_menagerie next to this file "
                                    "or set XARM7_XML)")
        self.scene = scene if scene is not None else make_scene("confined")
        apply_scene_grasps(self.scene)
        self.levels = self.scene["levels"]
        self.x0, self.pitch, self.rows, self.cols = (self.scene[k] for k in ("x0", "pitch", "rows", "cols"))
        self.table_z, self.verbose, self.xml_path = self.levels[0]["z"], verbose, xml_path
        self.objects = self._normalize(objects)
        self.obj_index = {o["name"]: i for i, o in enumerate(self.objects)}

        spec = mujoco.MjSpec.from_string(build_scene_xml(self.objects, self.scene, columns=NO_OVERPASS))
        arm = mujoco.MjSpec.from_file(xml_path)
        for b in arm.bodies:
            b.gravcomp = 1.0
        arm.body(HAND_ROOT).add_site(name="grasp", pos=[0, 0, GRASP_SITE_Z], size=[0.006, 0, 0], rgba=[0, 1, 0, 1])
        spec.attach(arm, frame=spec.worldbody.add_frame(pos=[0, 0, 0]), prefix="")
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.qadr = np.array([m.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.dadr = np.array([m.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.jrange = np.clip(np.array([m.jnt_range[m.joint(j).id] for j in ARM_JOINTS]), -3.1, 3.1)
        self.act_arm = [m.actuator(f"act{i}").id for i in range(1, 8)]
        self.act_grip = m.actuator("gripper").id
        self.grip_qadr = np.array([m.joint(j).qposadr[0] for j in GRIP_JOINTS])
        self._check_width = None                               # SHELF66V3: pad gap the checks put the fingers at (None: as they are)
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
        self.q_ready = self.q_home.copy()
        self.static_ck = self._build_static_ck() if self.scene["static_check"] else None
        self.n_branches = 4 if self.static_ck is not None else 1 if RETURN_TO_READY else 2   # IK branches tried per grasp
        self._ikc = {}                                                 # cached IK solutions per pose (shelves)
        # arm / hand versus the other objects (scenes with a structure check, and every scene when the arm does not
        # return to the ready pose: its free moves then start next to objects): the robot's collision geoms, and
        # each object's geoms; the object being worked on (_focus) is exempt
        self.object_check = self.static_ck is not None or not RETURN_TO_READY
        self._focus_i = None
        robot = set(self._subtree(m, m.body("link_base").id))
        self._robot_geoms = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in robot
                                      and (m.geom_contype[g] or m.geom_conaffinity[g])])
        self._obj_geoms = [np.flatnonzero(m.geom_bodyid == m.body(o["name"]).id) for o in self.objects]
        self._col_ids = [m.geom(o["name"] + "_col").id for o in self.objects] if NO_OVERPASS else []
        self._col_r = [footprint_radius(o) for o in self.objects]
        self._excuse = set()                                           # columns a free move starts inside (leaving is allowed)
        self.reset()
        self._calibrate_gripper()

    # -- objects, cells and levels.  A cell is (row, col) or (row, col, level); row 0 is the row nearest the robot,
    #    col 0 the -y edge, level 0 the lowest work surface.  Every level has the same rows x cols grid.
    def norm_cell(self, cell, level=0):
        return norm_cell(cell, level)

    def valid_cell(self, cell):
        r, c, L = norm_cell(cell)
        return 0 <= r < self.rows and 0 <= c < self.cols and 0 <= L < len(self.levels)

    def cell_xy(self, cell):
        r, c = cell[0], cell[1]
        return np.array([self.x0 + (r - (self.rows - 1) / 2) * self.pitch, (c - (self.cols - 1) / 2) * self.pitch])

    def cell_at(self, xy):
        r = int(round((xy[0] - self.x0) / self.pitch + (self.rows - 1) / 2))
        c = int(round(xy[1] / self.pitch + (self.cols - 1) / 2))
        return (r, c) if 0 <= r < self.rows and 0 <= c < self.cols else None

    def level_of(self, name):
        """The level an object stands on (the highest surface at or below it), None if it is on no surface."""
        i = self.obj_id(name)  # (no _focus here: _focus calls this)
        p, o = self.obj_pos(i), self.objects[i]
        best = None
        for k, lv in enumerate(self.levels):
            if lv["z"] <= p[2] + 0.02 and (best is None or lv["z"] > self.levels[best]["z"]):
                best = k
        if best is None or p[2] - self.levels[best]["z"] > o["height"] + 0.10:
            return None
        return best

    def _focus(self, i):
        """Work on object i: its level's surface is the 'table' (table_z) for the grasp / carry geometry, and the
        arm may touch it (not the others: see _object_hits)."""
        self._focus_i = self.obj_id(i)
        lv = self.level_of(i)
        self.table_z = self.levels[self.objects[i]["level"] if lv is None else lv]["z"]

    def _normalize(self, objects):
        out, names = [], set()
        for k, o in enumerate(objects):
            o = dict(o)
            o.setdefault("name", f"obstacle{k}" if k else "target")
            o.setdefault("shape", "box"); o.setdefault("size", 0.015); o.setdefault("height", 0.06)
            o.setdefault("color", "red"); o.setdefault("mass", 0.05); o.setdefault("friction", 1.2)
            cell = norm_cell(o["cell"])
            if o["name"] in names:
                raise ValueError(f"duplicate object name {o['name']!r}")
            names.add(o["name"])
            if o["shape"] not in ("box", "cylinder"):
                raise ValueError(f"{o['name']}: shape must be 'box' or 'cylinder'")
            if not self.valid_cell(cell):
                raise ValueError(f"{o['name']}: cell {cell} is outside the grid ({self.rows} rows x {self.cols} cols "
                                 f"x {len(self.levels)} level(s); cells are (row, col[, level]))")
            if 2 * o["size"] > self.pitch - FOOTPRINT_GAP - 0.004:
                raise ValueError(f"{o['name']}: {2 * o['size'] * 100:.1f} cm is too wide for a {self.pitch * 100:.0f} cm cell "
                                 f"(max {(self.pitch - FOOTPRINT_GAP - 0.004) * 100:.1f} cm)")
            if o["height"] < MIN_HEIGHT:
                raise ValueError(f"{o['name']}: needs height >= {MIN_HEIGHT * 100:.1f} cm for the finger pads")
            lv = self.levels[cell[2]]
            if lv.get("clearance") is not None and o["height"] + CARRY_LIFT + 0.06 > lv["clearance"]:
                raise ValueError(f"{o['name']}: {o['height'] * 100:.0f} cm is too tall for the {lv['clearance'] * 100:.0f} cm "
                                 f"clear height of level {cell[2]} (needs height + {(CARRY_LIFT + 0.06) * 100:.1f} cm for the lift and the hand)")
            o["cell"], o["level"] = cell, cell[2]
            o["pos"] = np.array([*self.cell_xy(cell), lv["z"] + o["height"] / 2])
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
        rc, lv = self.cell_at(self.obj_pos(name)[:2]), self.level_of(name)
        return None if rc is None or lv is None else (rc[0], rc[1], lv)

    def grid_text(self, goal=None):
        goal = None if goal is None else norm_cell(goal)
        occ = {self.cell_of(o["name"]): o for o in self.objects}
        lines = []
        for L, lv in enumerate(self.levels):
            if len(self.levels) > 1:
                lines.append(f"  level {L} ({lv.get('name', '')}, surface {lv['z'] * 100:.0f} cm)")
            lines.append("        " + " ".join(f"c{c}" for c in range(self.cols)))
            for r in reversed(range(self.rows)):
                row = []
                for c in range(self.cols):
                    o = occ.get((r, c, L))
                    row.append((("T" if o["name"] == "target" else o["name"].replace("obstacle", "")) if o
                                else ("G" if goal == (r, c, L) else ".")).ljust(2))
                lines.append(f"  r{r}    " + " ".join(row) + (" (nearest the robot)" if r == 0 else " (far)" if r == self.rows - 1 else ""))
        return "\n".join(lines)

    # -- the shelf structure: arm / hand versus the static geometry only (never versus the objects)
    def _build_static_ck(self):
        spec = mujoco.MjSpec.from_string(build_scene_xml([], self.scene, room=False))     # the room is looks only
        if NO_OVERPASS and ARM_CEILING and self.scene["name"] == "confined":
            # the arm's ceiling: a slab over the shelf from (tallest object's top + margin) up to the real ceiling
            zb = self.levels[0]["z"]
            zc = zb + max(o["height"] for o in self.objects) + CEILING_MARGIN
            xf, xb = self.x0 - self.rows * self.pitch / 2 - 0.04, self.x0 + self.rows * self.pitch / 2 + 0.04
            hw, ztop = self.cols * self.pitch / 2 + 0.14, zb + self.levels[0]["clearance"]
            spec.worldbody.add_geom(name="arm_ceiling", type=mujoco.mjtGeom.mjGEOM_BOX,
                                    size=[(xb - xf) / 2, hw, (ztop - zc) / 2], pos=[(xf + xb) / 2, 0.0, (zc + ztop) / 2])
        arm = mujoco.MjSpec.from_file(self.xml_path)
        arm.body(HAND_ROOT).add_site(name="grasp", pos=[0, 0, GRASP_SITE_Z], size=[0.006, 0, 0], rgba=[0, 1, 0, 1])
        spec.attach(arm, frame=spec.worldbody.add_frame(pos=[0, 0, 0]), prefix="")
        m = self.model
        hand = [m.body(b).name for b in self._subtree(m, m.body(HAND_ROOT).id)]
        for a, b in itertools.combinations(hand, 2):                     # the gripper's parts touch by design
            e = spec.add_exclude(); e.bodyname1, e.bodyname2 = a, b
        e = spec.add_exclude(); e.bodyname1, e.bodyname2 = "link_base", "link1"
        pm = spec.compile()
        # MuJoCo's mid-phase culls pairs without inflating by geom margin, so near-misses would be lost
        pm.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
        root = pm.body("link_base").id
        robot = [b for b in range(pm.nbody) if pm.body_rootid[b] == root and pm.body_weldid[b] != 0]
        for g in range(pm.ngeom):
            if pm.geom_bodyid[g] in robot:
                pm.geom_margin[g] = 0.01
        qadr = np.array([pm.joint(j).qposadr[0] for j in ARM_JOINTS])
        return MujocoCollisionChecker(pm, qadr, robot, self._subtree(pm, pm.body(HAND_ROOT).id), pm.site("grasp").id,
                                      margin=STATIC_MARGIN)

    def _static_hits(self, q):
        """Names of the static parts (and 'itself') the arm / open hand would touch at configuration q."""
        ck = self.static_ck
        ck.margin = STATIC_MARGIN
        hits = ck.collisions(q)
        names = set()
        for a, b in hits:
            st = [g for g in (a, b) if ck.role[g] == 0]
            if st:
                names.update(ck.m.geom(g).name or ck.m.body(ck.m.geom_bodyid[g]).name for g in st)
            else:
                names.add("itself")
        return sorted(names)

    def _object_hits(self, q, margin=OBJECT_MARGIN, every=False):
        """Names of the objects the arm / hand would come within `margin` of at configuration q, the objects
        where they are now (the one being worked on, _focus, is exempt: the hand holds it or reaches for it --
        except with every=True, for free moves: the hand is open and empty then and must touch nothing)."""
        if not self.object_check:
            return []
        m, d = self.model, self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = q
        if self._check_width is not None:                              # SHELF66V3: the fingers as the plan will have them
            d.qpos[self.grip_qadr] = self._finger_q(self._check_width)
        for j, c in enumerate(self._col_ids):                          # the columns stand where the objects are now
            m.geom_pos[c, :2] = self.obj_pos(j)[:2]
        mujoco.mj_kinematics(m, d)
        rg = self._robot_geoms
        names = []
        for j, c in enumerate(self._col_ids):
            if (j == self._focus_i and not every) or self.objects[j]["name"] in self._excuse:
                continue
            near = np.linalg.norm(d.geom_xpos[rg, :2] - d.geom_xpos[c, :2], axis=1) < m.geom_rbound[rg] + self._col_r[j] + margin
            if any(mujoco.mj_geomDistance(m, d, int(g), int(c), margin, None) < margin for g in rg[near]):
                names.append(self.objects[j]["name"])
        for j, hs in enumerate(self._obj_geoms):
            if j == self._focus_i and not every:
                continue
            for h in hs:
                near = np.linalg.norm(d.geom_xpos[rg] - d.geom_xpos[h], axis=1) \
                    < m.geom_rbound[rg] + m.geom_rbound[h] + margin               # bounding spheres first
                if any(mujoco.mj_geomDistance(m, d, int(g), int(h), margin, None) < margin for g in rg[near]):
                    names.append(self.objects[j]["name"])
                    break
        return names

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
        ctrls, gaps, fq = np.linspace(GRIPPER_OPEN, GRIPPER_CLOSED, 9), [], []
        for c in ctrls:
            d.ctrl[self.act_grip] = c
            for _ in range(500):
                mujoco.mj_step(m, d)
            gaps.append(np.linalg.norm(d.geom_xpos[l] - d.geom_xpos[r]) - pad_t)
            fq.append(d.qpos[self.grip_qadr].copy())               # SHELF66V3: the finger joints at this opening
        order = np.argsort(gaps)
        self._grip_cal = (np.array(gaps)[order], ctrls[order], np.array(fq)[order])

    def _finger_q(self, width):
        """SHELF66V3: the gripper's joint positions at pad gap `width` (from the calibration)."""
        gaps, _, fq = self._grip_cal
        return np.array([np.interp(width, gaps, fq[:, k]) for k in range(fq.shape[1])])

    def set_gripper_width(self, width):
        gaps, ctrls = self._grip_cal[:2]
        self.data.ctrl[self.act_grip] = float(np.interp(width, gaps, ctrls))

    def set_gripper(self, open_):
        self.data.ctrl[self.act_grip] = GRIPPER_OPEN if open_ else GRIPPER_CLOSED

    # -- basics
    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        for o, a in zip(self.objects, self.obj_qadr):
            o["pos"] = o["home_pos"].copy()                    # back to the starting cell
            self.data.qpos[a:a + 3] = o["pos"]; self.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        self.ready_p = np.array(self.scene["ready_p"], float)
        self.ready_R = grasp_frame(self.scene["ready_az"], self.scene["ready_tilt"])
        q = self.ik_pose(self.ready_p, self.ready_R, self.q_home)
        self.q_ready = q if q is not None else self.ik(self.ready_p, self.ready_R, self.q_home, iters=300)
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
        self._focus(i)
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

    def ik_candidates(self, p, R, q_prefer, limit=4):
        """IK solutions for one pose, nearest to q_prefer first.  With a structure check only the solutions that keep
        the arm clear of the structure count, and more seeds are tried: the arm has elbow-up and elbow-down kinds
        of solution, and under a ceiling only one of them may fit.  They depend only on the pose and the (static)
        structure, so they are cached.  The seeds are fixed (not the arm's current pose), so the result for a pose never
        depends on what the program did before."""
        key = (tuple(np.round(p, 4)), tuple(np.round(np.asarray(R).ravel(), 3)))
        if self.static_ck is not None and key in self._ikc:
            return sorted(self._ikc[key], key=lambda q: np.linalg.norm(q - q_prefer))[:limit]
        seeds = [self.q_home, self.q_ready]                    # fixed seeds only: the solutions depend on the pose alone
        base = math.atan2(p[1], p[0])
        for dq1 in (0.0, 0.5, -0.5):
            s = self.q_home.copy(); s[0] = base + dq1
            seeds.append(s)
        if self.static_ck is not None:
            rng = np.random.default_rng(7)
            for _ in range(12):
                s = self.q_home.copy(); s[0] = base
                seeds.append(s + rng.normal(scale=0.9, size=7))
        sols = []
        for s in seeds:
            q = self.ik(p, R, s, iters=250, tol=1e-5)
            ep, er = self.pose_error(q, p, R)
            if ep < 1e-3 and er < 1e-2 and (self.static_ck is None or not self._static_hits(q)) \
                    and all(np.linalg.norm(q - o) > 0.05 for o in sols):
                sols.append(q)
                if self.static_ck is not None and len(sols) >= limit:
                    break
        if self.static_ck is not None:
            self._ikc[key] = sols
        return sorted(sols, key=lambda q: np.linalg.norm(q - q_prefer))[:limit]

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
            if self.static_ck is not None:
                hit = self._static_hits(q_new)
                if hit:
                    raise RuntimeError(f"the arm / hand would hit the {', '.join(hit)} near {np.round(p, 3)}")
                hit = self._object_hits(q_new)
                if hit:
                    raise RuntimeError(f"the arm / hand would hit {', '.join(hit)} near {np.round(p, 3)}")
            if np.max(np.abs(q_new - q)) > max_jump:
                raise RuntimeError(f"IK jumps branch near {np.round(p, 3)}")
            q = q_new
            qs.append(q)
        return qs

    # -- the sideways-only carry route: RRT-Connect in the table plane
    def plan_carry_xy(self, name, goal_cell, grasp, seed=0, max_time=5.0, start_xy=None):
        """Route for the object's centre from where it is to goal_cell, in the table plane.

        The carried footprint is the object plus the gripper, by height band (HAND_BANDS): only the
        bands that overlap another object's height count, so a short neighbour only meets the narrow
        fingers while a tall one also meets the wide linkage.  It must keep CARRY_GAP away from every
        other object.  (A 2-D footprint rule from measured sizes, not a collision query.)
        A level hand (tilt >= LEVEL_TILT_DEG) lies flat instead: the same bands, measured along the hand, trail
        behind the pads in the approach direction at pad height (+- the band's thickness), so every neighbour
        tall enough to reach the hand meets a strip from the fingertips back to the wrist."""
        i = self.obj_id(name)
        self._focus(i)
        o = self.objects[i]
        p0 = np.asarray(start_xy, float) if start_xy is not None else self.obj_pos(i)[:2]
        goal_cell = norm_cell(goal_cell, o["level"] if self.level_of(i) is None else self.level_of(i))
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
        level = grasp["tilt_deg"] >= LEVEL_TILT_DEG
        # each hand band as a rectangle in the table plane: (centre offset from the object, half-extent along the
        # closing axis, half-extent across it, axis across it)
        if level:                                                        # flat hand: bands trail behind the pads
            rects = [(side * (lo + hi) / 2, _HAND_EY[k](gap), (hi - lo) / 2, side) for k, (lo, hi) in enumerate(HAND_BANDS)]
        else:                                                            # tilted hand leans outward
            rects = [(side * min(hi, 0.12) * math.tan(tilt), _HAND_EY[k](gap), _HAND_EX[k], v) for k, (lo, hi) in enumerate(HAND_BANDS)]
        # for each other object: the hand bands that overlap its height (pad centre = 0)
        parts = []
        for j, oj in enumerate(self.objects):
            if j == i or self.level_of(j) != self.level_of(i):         # other levels are not in the way
                continue
            top = oj["height"] - z_rel
            if not HAND_FOOTPRINT_RULE:
                bands = []
            elif level:                                                  # band k spans pad height +- _HAND_EX[k]
                bands = [k for k in range(len(HAND_BANDS)) if top > -_HAND_EX[k]]
            else:
                bands = [k for k, (lo, hi) in enumerate(HAND_BANDS) if top > lo and -z_rel < hi]
            if NO_OVERPASS:                                       # a neighbour is a column: the hand counts at every height
                bands = list(range(len(HAND_BANDS)))
            parts.append((self.obj_pos(j)[:2], footprint_radius(oj), bands))

        def is_free(xy):
            for q, r, bands in parts:
                if np.linalg.norm(q - xy) < r_me + r + CARRY_GAP:              # the object itself
                    return False
                for k in bands:
                    off, ey, ex, across = rects[k]
                    d = q - xy - off
                    a, b = abs(np.dot(d, c)), abs(np.dot(d, across))
                    if math.hypot(max(a - ey, 0.0), max(b - ex, 0.0)) < r + CARRY_GAP:
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
        """The check a free move gets.  Open scenes (forward kinematics, no contact query): the pad centre, the
        wrist and the hand stay above the table-top plane.  With a shelf / bench structure: the arm and the open
        hand must keep STATIC_MARGIN away from the structure instead.  Both: OBJECT_MARGIN from EVERY object
        (object_check), the one about to be picked too -- a free move can start next to the last one put down and
        sweep past the next one."""
        if self.static_ck is not None:
            return not self._static_hits(q) and not self._object_hits(q, every=True)
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return bool(d.site_xpos[self.site_id][2] >= self.table_z + 0.01
                    and np.all(d.xpos[self._wrist_bodies, 2] >= self.table_z + 0.05)) and not self._object_hits(q, every=True)

    def plan_free_move(self, qa, qb, seed=0, max_time=3.0):
        """RRT-Connect in joint space from qa to qb.  Valid = inside the joint limits and above the
        table plane (see _above_table); with a structure check also clear of the structure and the other
        objects.  Tries the straight joint line first, so with nothing in the way it connects immediately."""
        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        if NO_OVERPASS and not self._excuse and not getattr(self, "_in_free", False):    # leaving is allowed, entering is not: a move
            self._excuse, self._in_free = set(self._object_hits(qa, every=True)), True    # may start inside a column it is above
            try:
                return self.plan_free_move(qa, qb, seed, max_time)
            finally:
                self._excuse, self._in_free = set(), False
        is_free = lambda q: bool(np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9) and self._above_table(q))
        # edge check resolution (rad): with the arm stretched into a shelf 0.05 rad moves the hand ~4 cm between checks,
        # enough to pass through a 3 cm object; 0.02 rad (~1.6 cm) when the objects are checked
        res = 0.02 if self.object_check else 0.05

        def clear(q):
            """Inside a free move the arm sweeps fast and the servo lags the plan (up to ~0.04 rad): the clearance to the
            objects grows from OBJECT_MARGIN at the move's two ends (hover / retreat poses next to objects) to
            FREE_MOVE_MARGIN 0.2 rad of travel away from them, so that a path grazing an object mid-move is refused."""
            if not self.object_check:
                return True
            s = min(np.max(np.abs(q - qa)), np.max(np.abs(q - qb)))
            return not self._object_hits(q, OBJECT_MARGIN + (FREE_MOVE_MARGIN - OBJECT_MARGIN) * min(1.0, s / 0.2), every=True)

        def edge_free(a, b):
            n = max(2, int(np.ceil(np.max(np.abs(b - a)) / res)))
            return all(is_free(q) and clear(q) for q in (a + (b - a) * k / n for k in range(1, n)))

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
    def _plan_approach(self, g, q0, seed=0, branch=0):
        """IK for the grasp and hover poses, RRT-Connect for the free move to the hover pose, then the straight
        approach.  -> (segments, joint configuration at the grasp pose).
        Open table: the hover pose is solved first and the approach follows from it (branch 0: the IK solution
        nearest the arm's pose; 1: the one nearest the ready pose -- from elsewhere the nearest one can be an arm
        configuration whose straight approach / carry runs out of reach).  With a shelf / bench
        structure the GRASP pose is solved first (the k-th solution, `branch`, among those that clear the
        structure) and the approach is built backwards from it, so the hardest end -- the hand inside the
        shelf -- is the one that is guaranteed valid."""
        R, a, pg = g["R"], g["approach"], g["position"]
        pre = pg - a * PRE_DIST
        if self.static_ck is None:
            q_pre = self.ik_pose(pre, R, q0 if branch == 0 else self.q_ready)
            if q_pre is None:
                raise RuntimeError(f"hover pose {np.round(pre, 3)} is out of reach")
            ch = self.chain(q_pre, pre, [pg], R)
        else:
            cands = self.ik_candidates(pg, R, q0)
            if branch >= len(cands):
                raise RuntimeError(f"grasp pose {np.round(pg, 3)} is out of reach (or the arm would hit the structure there)")
            back = self.chain(cands[branch], pg, [pre], R)            # grasp -> hover
            q_pre, ch = back[-1], back[::-1]                          # hover -> grasp
        direct = self._direct_to_hover(q0, pre, R) if DIRECT_MOVES else None
        # SHELF66V3: the fingers are pre-shaped FIRST, where the arm is, and stay so until the grasp (before: only at the hover pose,
        # so the move there ran with whatever opening the fingers had -- fully open after a reset); the checks assume it (_check_width)
        segs = [dict(phase="open", kind="grip", grip=g["open_width"]),
                dict(phase="to_hover", kind="cart", q=direct) if direct is not None
                else dict(phase="to_hover", kind="joint", q=self.plan_free_move(q0, q_pre, seed))]
        segs.append(dict(phase="approach", kind="cart", q=ch))
        return segs, ch[-1]

    def _direct_to_hover(self, q0, pre, R):
        """NO_OVERPASS: the hand goes in a straight line from where it is to the next hover pose (same hand orientation, e.g. the
        next grasp comes from the same side), not along a joint-space swing.  None if the orientation differs or the straight line
        is blocked (the joint-space RRT-Connect then plans the move)."""
        p_now, _ = self._fk(q0)
        d = self.ik_data
        d.qpos[:] = self.data.qpos; d.qpos[self.qadr] = q0
        mujoco.mj_kinematics(self.model, d)
        Rn = d.site_xmat[self.site_id].reshape(3, 3)
        if np.arccos(np.clip((np.trace(Rn.T @ R) - 1) / 2, -1, 1)) > 0.12:
            return None
        if np.linalg.norm(np.asarray(pre) - p_now) < 1e-3:
            return None
        try:
            self._excuse = set(self._object_hits(np.asarray(q0, float), every=True))
            return self.chain(np.asarray(q0, float), p_now, [pre], R)
        except RuntimeError:
            return None
        finally:
            self._excuse = set()

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
        """Straight back along the approach axis; the next move starts from there (RETURN_TO_READY: then an
        RRT-Connect free move to the ready pose).  Near the edge of the arm's reach the full PRE_DIST may end
        out of reach (a few mm of hand position decide it), so a shorter straight retreat is used then: the open
        fingers only have to clear the object."""
        for dist in ((RETREAT_DIST, 0.07, 0.09, 0.11, PRE_DIST) if DIRECT_MOVES else (PRE_DIST, 0.11, 0.09, 0.07, 0.05)):
            try:
                ch = self.chain(q, p, [p - g["approach"] * dist], g["R"])
                break
            except RuntimeError:
                if dist == (PRE_DIST if DIRECT_MOVES else 0.05):
                    raise
        segs = [dict(phase="retreat", kind="cart", q=ch)]
        if RETURN_TO_READY:
            segs.append(dict(phase="to_ready", kind="joint", q=self.plan_free_move(ch[-1], self.q_ready, seed)))
        return segs

    def plan_pick_place(self, name, goal_cell, grasp, q_start=None, seed=0):
        """Kinematic plan of the whole trajectory for one grasp pose (a dry run: nothing moves).
        Returns dict(feasible, reason, segments, ...).
        segments: list of dict(phase, kind='joint'|'cart'|'grip', q=waypoints or grip=(cmd))."""
        i = self.obj_id(name)
        self._focus(i)
        o = self.objects[i]
        q0 = self.q_now() if q_start is None else np.asarray(q_start, float)
        plan = dict(feasible=False, reason="", segments=[], name=o["name"], goal=tuple(goal_cell),
                    grasp_id=grasp["id"], carry_xy=None)
        first_reason = ""
        at_ready = np.max(np.abs(q0 - self.q_ready)) < 1e-6
        self._check_width = grasp["open_width"]                # SHELF66V3: every check sees the fingers pre-shaped (see GRIP_JOINTS)
        try:
            for branch in range(self.n_branches):                  # the next IK branch if this one fails
                if self.static_ck is None and branch > 0 and at_ready:
                    break                                          # open scenes: from the ready pose both branches are one
                segs = plan["segments"] = []
                try:
                    s1, q_g = self._plan_approach(grasp, q0, seed, branch)
                    segs += s1
                    segs.append(dict(phase="close", kind="grip", grip=None))
                    s2, q_put, put, xy = self._plan_carry(name, goal_cell, grasp, q_g, seed)
                    plan["carry_xy"] = xy
                    segs += s2
                    segs.append(dict(phase="release", kind="grip", grip=grasp["open_width"]))
                    segs += self._plan_retreat(grasp, q_put, put, seed)
                    plan.update(feasible=True, branch=branch, reason="")
                    break
                except (RuntimeError, ValueError) as err:
                    first_reason = first_reason or str(err)
                    plan["reason"] = first_reason
                    if isinstance(err, ReachError) or ("out of reach" in str(err) and at_ready):
                        break                                          # another IK branch cannot fix this (from the ready
                        # pose; from elsewhere the branches come in another order, nearest the arm first: all are tried)
        finally:
            self._check_width = None
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

    def _select_grasp(self, name, destination=None, grasp_id=None, skip=()):
        """First of the grasps that is feasible.  With a destination: the WHOLE pick -> carry -> place
        trajectory (the hand's footprint during the carry depends on the grasp, so the grasp has to be
        chosen together with the destination).  Without: just hover + approach.  Nothing moves."""
        i = self.obj_id(name)
        self._focus(i)
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
                g["branch"] = plan["branch"]
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

    def MoveToObject(self, name, destination=None, grasp_id=None, skip=()):
        """Move the arm to `name` (an object, obstacle or the target) at its hard-coded location, with the
        fingers open around it.  IK + RRT-Connect run first (a dry run picks the grasp -- with `destination`
        one that can also reach it); then: free move to the hover pose, open the fingers, straight approach.
        Returns the grasp used."""
        if self.held is not None:
            raise RuntimeError(f"MoveToObject: the gripper already holds {self.objects[self.held]['name']}")
        i = self.obj_id(name)
        self._focus(i)
        o = self.objects[i]
        g = self._select_grasp(name, destination, grasp_id, skip)
        segs, _ = self._plan_approach(g, self.q_now(), branch=g.get("branch", 0))   # again, from where the arm is now
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
        i, name, g = cur["i"], cur["name"], cur["grasp"]
        lv = self.level_of(i)
        cell = norm_cell(cell, self.objects[i]["level"] if lv is None else lv)
        if not self.valid_cell(cell):
            raise ValueError(f"cell {cell} is outside the grid ({self.rows} rows x {self.cols} cols x {len(self.levels)} level(s))")
        if lv is not None and cell[2] != lv:
            raise ValueError(f"cell {cell} is on level {cell[2]} but {name} is on level {lv}: objects stay on their own level")
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
        axis, so the open fingers cannot drag the object when the arm moves on (and stays there: the next move
        starts from where the hand is; RETURN_TO_READY: to the ready pose); retreat=False only opens."""
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
        """MoveToLocation + OpenGripper (which backs away), then check the object stayed put."""
        self.MoveToLocation(cell)
        i, name = self.current["i"], self.current["name"]
        self.OpenGripper()
        err_mm = np.linalg.norm(self.obj_pos(i)[:2] - self.cell_xy(cell)) * 1000
        zax = self.data.xmat[self.model.body(name).id].reshape(3, 3)[:, 2]
        tilt = math.degrees(math.acos(float(np.clip(zax[2], -1, 1))))
        if self.verbose:
            print(f"      placed {err_mm:.1f} mm from the cell centre")
        if self.cell_of(i) is None or tilt > 25 or self.cell_of(i)[2] != cell[2]:
            raise ObjectLost(f"{name} did not stay in cell {cell}: it ended {err_mm:.0f} mm from the cell centre, "
                             f"tilted {tilt:.0f} deg" + (" and off the grid" if self.cell_of(i) is None else " and on another level" if self.cell_of(i)[2] != cell[2] else "") +
                             ".  A tall slim object gripped with a tilted hand tips over when released")
        if err_mm > 20:
            raise GraspFailed(f"{name} ended {err_mm:.0f} mm from the centre of cell {cell}")
        return cell

    def _recover(self):
        """After a failed grasp: open, back away along the approach axis (the ready pose only if that fails)."""
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

    def PickAndPlace(self, name, cell, grasp_id=None):
        """Pick(name) + Place(cell), trying the next grasp if the fingers miss.  Raises ReachError when no
        grasp is feasible (the move is infeasible), ValueError for a bad / occupied cell."""
        i = self.obj_id(name)
        self._focus(i)
        o = self.objects[i]
        lv = self.level_of(i)
        cell = norm_cell(cell, o["level"] if lv is None else lv)
        if not self.valid_cell(cell):
            raise ValueError(f"cell {cell} is outside the grid ({self.rows} rows x {self.cols} cols x {len(self.levels)} level(s))")
        if lv is not None and cell[2] != lv:
            raise ValueError(f"cell {cell} is on level {cell[2]} but {o['name']} is on level {lv}: objects stay on their own level")
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
    def what_if(self, name, cell):
        """{other object: True if `name` -> cell has a feasible grasp with only that object taken away}.
        The hover / approach plan does not depend on the other objects, so it is computed once per grasp;
        only the carry (and the retreat after it) is re-planned for each object that is taken away."""
        i = self.obj_id(name)
        self._focus(i)
        o = self.objects[i]
        o["pos"] = self.obj_pos(i)
        q0 = self.q_now()
        approach = []
        for g in grasp_poses(o, self.table_z):
            for branch in range(self.n_branches):
                try:
                    approach.append((g, self._plan_approach(g, q0, branch=branch)[1]))
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
    def reach_map(self, tilt_subset=None, n_azimuth=None, level=0):
        """Number of grasp poses (of N_GRASPS, or a subset) whose hover + grasp pose is reachable, per cell
        (of the empty grid: the objects are not checked)."""
        out = np.zeros((self.rows, self.cols), int)
        self.table_z = self.levels[level]["z"]
        object_check, self.object_check = self.object_check, False
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
                    if self.static_ck is not None:                    # shelves: the planner's own approach (all IK branches)
                        for br in range(self.n_branches):
                            try:
                                self._plan_approach(g, self.q_ready, branch=br)
                                n += 1
                                break
                            except (RuntimeError, ValueError):
                                continue
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
        self.object_check = object_check
        return out

    # -- dataset: one trajectory per grasp pose
    def generate_dataset(self, name, goal_cell, out_dir="dataset", hz=50, seed=0):
        """For every one of the grasp poses, plan the complete pick -> carry -> place -> retreat
        trajectory (-> home with RETURN_TO_READY) (kinematically; nothing is simulated) and save it.  Files in out_dir:
            grasps.csv       the poses and whether each one is feasible (and why not)
            summary.json     scene + settings
            traj_XX.npz      t, q (N,7), ee_pos (N,3), ee_quat (N,4 wxyz), gripper_width (N),
                             phase (N, int), phase_names, object_xyz (N,3)  -- one per feasible grasp"""
        os.makedirs(out_dir, exist_ok=True)
        i = self.obj_id(name)
        self._focus(i)
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
        cam = self.scene["camera"]
        self.viewer.cam.lookat[:] = cam["lookat"]
        self.viewer.cam.distance, self.viewer.cam.azimuth, self.viewer.cam.elevation = cam["distance"], cam["azimuth"], cam["elevation"]

    def render(self, path="preview.png", w=960, h=720, lookat=None, distance=None, azimuth=None, elevation=None):
        r = mujoco.Renderer(self.model, h, w)
        cam = mujoco.MjvCamera()
        sc = self.scene["camera"]
        cam.lookat[:] = lookat if lookat is not None else sc["lookat"]
        cam.distance = sc["distance"] if distance is None else distance
        cam.azimuth = sc["azimuth"] if azimuth is None else azimuth
        cam.elevation = sc["elevation"] if elevation is None else elevation
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
    return tuple(int(v) for v in n) if len(n) in (2, 3) else None


def select_scene(name):
    """Make `name` the active scene: its target / goal / obstacles / commands become the module settings."""
    st, g = SETTINGS[name], globals()
    g["SCENE"], g["TARGET_CELL"], g["GOAL_CELL"] = name, st["target_cell"], st["goal_cell"]
    g["TARGET"], g["OBSTACLES"], g["MOVE_COMMANDS"] = st["target"], st["obstacles"], st["move_commands"]
    return make_scene(name)


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
            fmt = "row,col" + (",level" if len(env.levels) > 1 else "")
            t = console.ask(f"{pad}Where should {obj} go?  [{fmt} | back | reset | quit] > ").lower()
            if t == "back":
                break
            c = parse_cell(t)
            lv = env.level_of(obj)
            if c is not None:
                c = env.norm_cell(c, 0 if lv is None else lv)
            if c is None:
                print(f"{pad}  type the cell as {fmt}  (e.g. 2,3)")
            elif not env.valid_cell(c):
                print(f"{pad}  {c} is outside the grid ({env.rows} rows x {env.cols} cols x {len(env.levels)} level(s))")
            elif lv is not None and c[2] != lv:
                print(f"{pad}  {obj} is on level {lv}; objects stay on their own level")
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
    goal = env.norm_cell(GOAL_CELL)
    for name, cell in MOVE_COMMANDS:
        if not do_move(env, console, name, tuple(cell), goal=goal):
            print(f"({name} -> {tuple(cell)} was not done)")
    if do_move(env, console, "target", goal, goal=goal) and RETURN_TO_READY:
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
    ap.add_argument("--scene", choices=list(SCENES), help="confined | multilevel | factory | table (default: SCENE in the settings)")
    ap.add_argument("--once", action="store_true", help="exit when the run is over (do not wait for 'reset' / 'quit')")
    ap.add_argument("--reach", action="store_true", help="print how many of the grasps reach each cell")
    ap.add_argument("--dataset", metavar="OBJECT", help="export one trajectory per grasp pose for OBJECT -> GOAL_CELL")
    ap.add_argument("--out", default="dataset")
    ap.add_argument("--hz", type=int, default=50)
    ap.add_argument("--render", metavar="PREFIX")
    a = ap.parse_args()

    scene = select_scene(a.scene or SCENE)
    env = XArm7TableEnv(user_objects(), scene)
    globals()["GOAL_CELL"] = env.norm_cell(GOAL_CELL)
    print(f"scene: {SCENE}")
    for o in env.objects:
        print(f"{o['name']:<10} {o['shape']:<8} {2 * o['size'] * 100:4.1f} cm wide {o['height'] * 100:4.1f} cm tall "
              f"cell {env.cell_of(o['name'])}")
    print(f"\ngoal {GOAL_CELL}\n" + env.grid_text(goal=tuple(GOAL_CELL)) + "\n")
    if a.reach:
        for L in range(len(env.levels)):
            m = env.reach_map(level=L)
            print(f"level {L}: grasp poses (of {N_GRASPS}) that reach each cell (rows from far to near):")
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
