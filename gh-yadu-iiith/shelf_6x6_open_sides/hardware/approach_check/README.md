# Single-direction pick check (azimuth 180) — simulation, EXPLORATORY

Question: can the xArm7 pick objects on the 6 x 6 shelf using ONE approach direction only (azimuth 180: hand on the
robot's side, moving straight in along +x), without knocking neighbours over? Run 2026-11-22. No dataset split is
involved (synthetic single-pick scenes).

## Answer

- **Yes, azimuth 180 alone reaches the grid.** 412 of 414 standard-gripper cases planned, lifted and held the object.
  The two misses are cell (5,2) with a lone 12 cm object (see failures). No neighbour was ever moved (max 0.1 mm, 0.1°)
  and there was no physical contact.
- **The limit is clearance, not reach.** With side neighbours 7.5 cm away the open fingers pass about 9 mm from a
  r 1.8 cylinder and about 6.5 mm from a 1.6 box, for the r 2.0 target. A real robot wants 10 mm or more.
- The 16-direction reference (B) gives the same clearances, because with side neighbours present it also picks
  from 180. The single-direction rule costs nothing here; the tight gap comes from the 7.5 cm pitch, the gripper and
  the 6 mm-per-side finger pre-shaping.
- The G2 gripper gives the same result within 0.1 mm. Its fingers are the G1 shapes; only the housing is wider, and
  the housing never comes near a neighbour.

## Results (standard gripper unless marked; clearance = min distance robot -> any neighbour, approach .. retreat)

| approach | object | layout | nbr | pickable/tested | knocked | clr median mm | clr 5th pct mm | cases < 10 mm |
|---|---|---|---|---|---|---|---|---|
| A (180) | T r2.0 h12 | N0 | - | 35/36 | 0 | - | - | - |
| A | T | N1 / N2 | cyl r1.8 | 27/27, 27/27 | 0 | 8.9 | 8.6 | all |
| A | T | N1 / N2 | box 1.6 | 27/27, 27/27 | 0 | 6.6 | 6.3 | all |
| A | narrow r1.3 h12 | N0 | - | 26/27 | 0 | - | - | - |
| A | narrow | N1 / N2 | cyl / box | 27/27 each | 0 | 15.0 / 12.7 | 14.7 / 12.4 | 0 |
| A | wide box 1.6 h18 | N0 | - | 27/27 | 0 | - | - | - |
| A | wide | N1 / N2 | cyl / box | 27/27 each | 0 | 12.3 / 10.0 | 12.0 / 9.7 | 0 / 13 of 27 |
| B (16 dirs) | all three | N0 / N1 / N2 | cyl | 252/252 | 0 | same as A | same as A | same as A |
| A, G2 gripper | all three | N0 / N1 / N2 | cyl | 250/252 | 0 | same as A (±0.1) | same | same |

The full table, every case and the per-object maps are in `summary.md`, `results.csv` and `results.jsonl`. N1 and N2 give
identical clearances: the closest neighbour is always a side one (L/R), touched by the finger (fingertip or outer
knuckle) during the approach. The back neighbours are never closest.

## 6 x 6 maps, azimuth 180 only (standard gripper; cylinder and box neighbours combined)

`O` pickable and clearance >= 10 mm (or no neighbours) · `o` pickable but clearance < 10 mm · `x` no plan · `.` not tested.
Row 5 is at the back, row 0 is next to the robot. Column 0 is the robot's right (−y).

```
            N0 (alone)                 N1 (left+right)            N2 (L+R+behind+back diagonals)
       T      narrow   wide        T      narrow   wide        T      narrow   wide
r5  OOxOOO  OOxOOO  OOOOOO     oooooo  OOOOOO  OOoOoO      oooooo  OOOOOO  OOoOoO
r4  OOOOOO  O.O..O  O.O..O     o.o..o  O.O..O  o.o..O      o.o..o  O.O..O  o.o..O
r3  OOOOOO  O.O..O  O.O..O     o.o..o  O.O..O  O.o..O      o.o..o  O.O..O  O.o..O
r2  OOOOOO  OOOOOO  OOOOOO     oooooo  OOOOOO  OooooO      oooooo  OOOOOO  OooooO
r1  OOOOOO  O.O..O  O.O..O     o.o..o  O.O..O  O.o..O      o.o..o  O.O..O  O.o..O
r0  OOOOOO  OOOOOO  OOOOOO     oooooo  OOOOOO  OOOooo      oooooo  OOOOOO  OOOooo
```
The wide-object `o` cells are the box-neighbour runs at 9.7–10.0 mm. With cylinder neighbours they are all `O` (12 mm or more).

## Failures

- (5,2) with a lone 12 cm target or narrow object: the **arm ceiling** rule fails at mid-height (the ceiling is the
  tallest object + 10 cm, so it is low when the scene has only a 12 cm object). The top grasp has no IK solution clear
  of the structure. With 18 cm neighbours the ceiling rises and the same cell plans. This is a planning rule, not a
  physical limit.
- 66 of 412 standard (A) plans needed a second IK branch. The dataset teacher's `plan_pick_place` stops at the first
  "out of reach" from the ready pose, so it would not find these with that grasp (`teacher_rule=false` in the data).
  All of them are in rows 0 and 1.

## Method

- Shelf = the dataset's SHELF66 V3: `bd_shelf.env()` from `dataset_bd_classquota_without_home_position_no_overpass/code/shelf66_v3`.
  Its `shelf_scene.py` is byte-identical to `scene/shelf_scene.py`. Same no-overpass columns, arm ceiling, pre-shaped
  fingers (+6 mm per side), 3 mm / 5 mm clearances, RRT max_iters 5000. Only the scene's `azimuths` list is changed:
  [180] for A, 16 for B.
- Pick = the teacher's own pieces: `_plan_approach` (pre-shape, RRT free move to the hover pose 13 cm back, straight
  approach), then close, lift 2.5 cm (`chain`), and pull straight back 13 cm holding the object. That last step
  replaces the carry. Grasps are tried in the scene order (mid-height first, then near the top) over all IK branches.
- Execution = MuJoCo physics with the scene's controllers (`execute`, `segment_traj`). Every 5 steps the script
  records each neighbour's displacement and tilt, and every robot–neighbour contact. It also records the minimum
  distance (`mj_geomDistance`) from every robot geom to every neighbour, **visual finger meshes included**. (In the
  menagerie model the fingers only collide through their pads, so physics alone would miss a finger passing through
  a neighbour.) Meshes are measured by their convex hull, so the clearance is slightly conservative.
- Knocked = tilt > 10° or displacement > 5 mm. Lifted = object ≥ 0.75 cm up after the lift and still in the hand
  after the pull-back.
- Neighbours: N1 = (r, c±1), N2 = N1 + (r+1, c), (r+1, c±1). The front cell is always empty. Neighbours that would fall
  off the grid are left out, so edge cells have fewer.
- Cells: all 36 for target T alone. Otherwise rows 0, 2, 5 plus columns 0, 2, 5 (27 cells). Neighbour kind: cylinder
  and box in A; cylinder only in B and G2.
- Compute: one process at low priority. `nice` is not permitted in the sandbox, so `taskpolicy -c utility` was used.
  About 23 CPU-minutes in total (A+B 13.5, G2 10).

## Not tested

- A neighbour directly in front (row r−1): a straight-in pick needs that corridor empty by definition.
- Other heights and the real carry/place. Only pick, lift and pull-back were checked.
- Physical friction or compliance of real objects, sensor or positioning error, and real servo tracking error.
  The sim margin of about 9 mm (T) and 6.5 mm (box neighbours) leaves almost no room for a 2–5 mm placement error.
- Cells outside the cross pattern for N1/N2 and for the narrow/wide objects. Geometry is the same in every row (the
  clearance varied by about 1 mm across the cells tested), so similar values are expected.

Files: `approach_check.py` (runner; `--stage A|B|G2A`), `summarize.py`, `results.jsonl`, `results.csv`, `summary.md`, `run_*.log`.
