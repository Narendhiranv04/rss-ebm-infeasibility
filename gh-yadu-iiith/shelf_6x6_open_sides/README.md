# 6 × 6 open-sided shelf (xArm7, MuJoCo)

This is the shelf scene used by the shelf datasets in this folder. It has one shelf level with 36 cells (6 × 6, 7.5 cm apart). The shelf is open at the front and at both sides, and it stands in front of a UFactory xArm7. The robot reaches in level with the shelf floor and picks objects from the side. The arm never passes over an obstacle.

## Contents

```
shelf_6x6_open_sides/
  README.md                           this file
  requirements.txt                    Python dependencies (mujoco, numpy)
  scene/
    shelf_scene.py                    the environment: scene, grasp set, planner, MuJoCo execution (XArm7TableEnv)
    xarm7_cabinet_env.py              robot constants, model path (XARM7_XML), IK helpers
    rrt_connect.py, rrt_star.py       RRT-Connect and path utilities used by the planner
    mujoco_menagerie/ufactory_xarm7/  the xArm7 model from MuJoCo Menagerie (BSD-3-Clause, UFACTORY; see its LICENSE)
  dataset_bd_classquota_without_home_position_no_overpass/
                                      the 2,500-scene class-quota dataset on this shelf, with its code (see its README)
```

The robot model is in the repository, so nothing else needs to be downloaded. To use a different copy of the model, set `XARM7_XML` to the path of its `xarm7.xml`. Requirements: Python 3.10, `mujoco==3.16.0`, `numpy==2.2.6` (`pip install -r requirements.txt`).

## Geometry

- **Frame:** the robot base is at the origin. `x` points away from the robot into the shelf, `y` to the robot's left, and `z` up.
- **Grid:** 6 rows × 6 columns with a 0.075 m pitch. Cells are `(row, col)`, 0-based. Row 0 is nearest the robot, and column 0 is the −y edge (the robot's right).
- **Cell centre:** `x = 0.42 + (row − 2.5) · 0.075` (0.2325 to 0.6075 m) and `y = (col − 2.5) · 0.075` (−0.1875 to 0.1875 m).
- **Shelf floor:** objects stand at `z = 0.55 m`. The clear height to the ceiling board is 0.60 m.
- **Shelf body:** the front edge is at `x = 0.155 m` and the back wall at `x = 0.685 m`. There are no side walls, so the shelf is open on the left and right. The floor board reaches 0.14 m past the outer columns. A low step (0.08 m) stands in front of the shelf, clear of the robot base.
- **Ready pose:** the hand points down at `(0.105, 0, 0.75)`, just outside the front edge.

## Motion rules

- **Grasps:** side grasps only, with the hand level (tilt 90°), from 16 directions every 22.5° around the object. The azimuth is where the hand is, measured from +x toward +y, so 180° is the hand on the robot's side of the object.
- **Grip height:** the pads close at the object's mid-height. If no plan exists there, they close near the top (top − 1.2 cm). That gives 32 grasp poses per object. The first one that plans is used: mid-height first, then the directions nearest the robot.
- **One move:** hover 13 cm back along the approach axis, move in, close, lift 2.5 cm, carry sideways, put down, open, and back off 5 cm.
- **No overpass:** every object except the one being moved stands in a vertical column (its footprint, 1.2 m high). No part of the arm, the hand or the held object may enter another object's column, so the arm goes around obstacles, never over them. A free move may start inside a column the arm is already in, and leave it.
- **Arm ceiling:** over the shelf, no part of the arm or hand may be higher than the tallest object's top + 10 cm. The held object is exempt.
- **No return to the ready pose:** after putting an object down, the arm backs off and goes straight to its next pick. Only the first move of a scene starts at the ready pose.
- **Fingers pre-shaped (SHELF66V3, 5 Nov 2026):** before every pick the fingers open to the object's width + 1.2 cm, and every collision check uses the fingers at that opening. Before this fix the checks used the fingers as they happened to be, fully open (about 8.5 cm) at the start of a scene, which made feasible grasps look infeasible.
- **Clearances:** 3 mm to the shelf, 5 mm to other objects, and 2 cm in the middle of free (joint-space) moves. Plans are checked against the shelf structure, all objects and the arm itself.

## Objects

- **Types:** cylinders and boxes, all of them free bodies, so every object is movable. `size` is the radius for a cylinder and the half-width for a box.
- **Target T (`t0`):** a green cylinder, radius 1.6–2.0 cm, height 12–16 cm.
- **Obstacles (`o00`, `o01`, …):** cylinders (radius 1.3–1.8 cm) or boxes (half-width 1.2–1.6 cm). They are 12–18 cm tall when built to block T, and 6–18 cm otherwise.

## Why this design

Under the no-overpass rule and the arm ceiling, the previous confined shelf left its back half mostly out of reach. That shelf had 8 × 8 cells at 6.5 cm, side walls, its floor at 0.50 m and its grid centre at 0.54 m. With a lone object, only 41 of 63 picks and 31 of 63 places planned. Neighbours 6.5 cm to the side also blocked most grasps.

This shelf fixes both problems:
- It is 12 cm closer to the robot, its floor is 5 cm higher, and it has no side walls.
- It has 16 grasp directions and the mid-height grip with the top fallback.

With the real planner, a lone object can be picked from every cell with at least 6 directions and placed with at least 4.

7.5 cm is the smallest pitch at which a side neighbour still leaves most cells usable. With one neighbour beside the object, 51 of 60 test cases stayed pickable at 7.5 cm, with 2.5 grasp directions left on average. At 7 cm the figures were 41 and 1.3.

## Use

The dataset code sets the scene up in `dataset_bd_classquota_without_home_position_no_overpass/code/shelf66_full/bd_shelf.py` (`env()`, `build()`), and `code/view_scene66.py` replays any scene of the dataset in the MuJoCo viewer. `shelf_scene.py` holds this shelf as its `"confined"` scene: the `SHELF66` settings near the top of the file.
