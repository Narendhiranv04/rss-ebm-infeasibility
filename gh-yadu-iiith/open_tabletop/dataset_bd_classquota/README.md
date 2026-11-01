# 3D open-tabletop B/D dataset (class-quota run)

A dataset of tabletop rearrangement problems for an xArm7 in MuJoCo, built with the B/D dataset scripts (`code/dataset-bd`). Each scene has a target object T that must reach a goal cell. Other objects may block it, and the label says how many objects must be moved out of the way first (K). Generated on 31 Oct – 1 Nov 2026 on a laptop and the lab server, which shared one ledger. The class quotas were met on 1 Nov 2026 at 11:54 IST.

This folder holds the **complete raw output** of the run: every scene file, every trajectory and the full attempt ledger. A combined one-file-per-scene export has not been made yet (see [Not done yet](#not-done-yet)).

## At a glance

**1,225 counted scenes.** `selected.json` lists them by class.

| Class | Meaning | Scenes | Obstacles per scene (min / median / max) |
|---|---|---|---|
| NONE | no verified repair found (see [Known issue: NONE](#known-issue-none)) | 200 | 1 / 5 / 7 |
| K0 | T reaches its goal directly | 250 | 0 / 2 / 5 |
| K1 | 1 object must be moved first | 175 | 1 / 3 / 6 |
| K2 | 2 objects | 175 | 2 / 4 / 7 |
| K3 | 3 objects | 175 | 3 / 6 / 9 |
| K4 | 4 objects | 250 | 4 / 6 / 9 |

`raw/` also keeps every other attempt: 1,590 accepted scene files in total (the 1,225 counted, scenes over a class quota, and 87 `DEPENDENT` scenes), and 3,800 ledger lines (the 2,210 rejected constructions are recorded there with their reasons).

## Folder layout

```
dataset_bd_classquota/
  README.md
  selected.json            counted scene ids per class, with the quotas
  raw/                     the run's dataset-bd store, unchanged
    ledger.jsonl           one line per attempt: spec, accepted, verified class, reason, wall time
    PROGRESS.json          final counts and quotas
    roots/<root_id>.json   one file per accepted scene: layout, labels, construction metadata
    witnesses/<ab>/<id>.npz   verified robot trajectories referenced by the labels
  code/                    the code that generated it
    dataset-bd/            B/D dataset library (bd_core, bd_3dt teacher, ...); see its README
    bd_3dt_plus.py         class-quota construction for K >= 1 (generator BD_GEN_V1+ROUTE_V14 / V15)
    run_3dt_plus.py, .sh   the runner (plans requests from the ledger until every quota is met)
    make_combined_3dt.py   combined one-file-per-scene export (not yet run on this data)
    view_3dt.py            interactive MuJoCo viewer for one scene
    viz_3dt.py             PNG / GIF renders + index.html (writes into <dir>/viz)
```

## The scene

- **Robot:** UFactory xArm7 from MuJoCo Menagerie, side grasps only: 8 azimuths, every 45°, with the hand level (tilt 90°).
- **Table:** the open table of `gh-yadu-iiith/table_scene (1).py` on this branch (b3b5b34): an 8 × 8 grid with a 7 cm pitch. Cells are `(row, col)`, 0-based, with row 0 nearest the robot. Cell centre = `(0.4 + (r − 3.5)·0.07, (c − 3.5)·0.07)`, table top at z = 0.08 m.
- **Objects:** cylinders and boxes of varying size, height and colour. The target T (`t0`) is always a green cylinder; the obstacles are `o00`, `o01`, …. Every object is movable (a free body).
- **Start:** the robot always starts at the scene's ready pose (`start_pose: home`).

## What a scene file contains (`raw/roots/<root_id>.json`)

| Field | Content |
|---|---|
| `ids` | root id, construction seed, generator / teacher / label-protocol versions, config hash |
| `spec` | the requested factors: `K`, `mechanism` (grasp / place / mixed / transfer), `source` (near / mid / far start rows), `displacement` (short / medium / long), `distractors`, `distractor_role`, `parking`, `start_pose` |
| `objects`, `initial_cells` | every object (id, shape, size, height, colour) and its starting cell |
| `target_src`, `goal` | T's start cell and goal cell |
| `roles`, `intended_blockers` | each obstacle's role (intended blocker or a distractor type) |
| `parking`, `alt_parking` | each obstacle's parking cells (the teacher uses the first one) and up to 5 alternatives |
| `verified_K`, `observed` | the verified class: 0–4, or `NONE` |
| `labels` | see below |
| `independent_moves` | true for every counted scene: each object of the minimal repair moves once, and each move works on its own |
| `construction`, `empty_table_T_solve`, `mechanism_pools`, `greedy_blockers` | how the layout was built; T's feasible grasp azimuths on an empty table |
| `budget`, `geometry`, `cell_convention`, `source_files`, `teacher_calls`, `teacher_wall_s` | teacher budget (RRT 5,000 iterations, no wall-clock limit), table geometry, hashes of the scene files used |

### Labels

All labels are binary: `y = 1` means verified within the teacher's budget, and `y = 0` comes with a `negative_kind` (`CERTIFIED`, `TIMEOUT`, `NO_PLAN_FOUND` or `OTHER_FAILURE`).

- **`original`:** can T reach its goal in the scene as it is?
- **`D`:** hypothetical edits. The listed obstacles are placed at their parking cells and only T is re-planned. There is one edit per obstacle, plus the intended blockers together.
- **`B`:** action sequences that are **executed** in MuJoCo, step by step, from the initial scene. A step counts only if the moved object ends upright in its destination cell and no other object changed cell. Roles:
  - `minimal`: the verified fewest-move repair, then T;
  - `redundant+1`: the repair plus one unnecessary move;
  - `noop_irrelevant`: moving a distractor only;
  - `wrong_destination`: one blocker moved to the wrong cell;
  - `wrong_order`: the repair in reverse order;
  - `intended_union`: the intended blockers, when no minimal repair was found.
  Each B entry has its `sequence`, `failed_step` and `witnesses` (the trajectories, one per executed step).
- **`blocker`:** the full lattice of edits over the teacher's object pool: the minimal clearing sets, a per-object blocker mask, and which clearing sets were reachable by actual relocations.

### Trajectories (`raw/witnesses/<first 2 chars>/<id>.npz`)

`q` (n × 7 joint path), `segment` (segment index per waypoint), `carry_xy` (start and end of the carry), `phases` (segment names).

```python
import json, numpy as np
sel = json.load(open("selected.json"))
rid = sel["selected"]["3"][0]                                   # first counted K3 scene
r = json.load(open(f"raw/roots/{rid}.json"))
b = next(b for b in r["labels"]["B"] if b["role"] == "minimal")
print(r["verified_K"], b["sequence"])                           # 3, and the three (object, parking cell) moves
w = b["witnesses"][0]
traj = np.load(f"raw/witnesses/{w[:2]}/{w}.npz")
print(traj["q"].shape)
```

## How the dataset was made

- **The rules for every scene:**
  - Each object moves at most once.
  - The objects in a repair don't depend on each other: every blocker can be picked and parked from the original scene with everything else in place, so the moves work in any order.
  - A task has at most 7 pick-and-place actions: up to 4 necessary moves, up to 2 redundant ones, then T.
- **Classes count the verified K, not the requested K.** The runner requests a class according to how often each request has produced each class so far, then keeps a scene under the class the teacher verified.
- **Selection:** the accepted scenes in ledger order, per verified class, up to the quota. That's what `selected.json` reproduces.
- **Generators:** K0 uses dataset-bd's `bd_3dt.construct` unchanged. K ≥ 1 uses `bd_3dt_plus.py`, the "route" construction: it probes which cells block each of T's feasible grasps and adds blockers until every grasp is blocked by at least K of them. Most counted K ≥ 1 scenes come from versions ROUTE_V14 and V15; a few come from earlier versions.
- **Teacher:** `BD3DT_TEACHER_V1` (the scene's own `plan_pick_place`, first feasible side grasp, RRT-Connect capped at 5,000 iterations, then MuJoCo `run_plan`).
- **Machines:** 919 counted scenes were made on the laptop and 306 on the lab server (90 of the 175 K3 scenes and 216 of the 250 K4 scenes).

### Spec balance

The mechanisms are balanced in every class. Start areas and move lengths are balanced for NONE through K2. For the hardest classes they are skewed toward what's feasible:
- **K3:** long moves 108, medium 52, short 15.
- **K4:** long moves 223, medium 27, no short; far starts 151, near 93, mid 6.

## Known issue: NONE

In this dataset, NONE means **the teacher's search found no repair it could execute**. It is not a proof that none exists:
- None of the 200 NONE scenes was built to be infeasible. They are requests for K0–K3 that the teacher could not verify, and every one is labelled `NO_PLAN_FOUND` (a planner result), not `CERTIFIED`.
- The teacher's repair search is narrow:
  - it uses only each object's first parking cell;
  - it checks each relocation from the ready pose in a teleported scene;
  - it keeps one order per set of objects.
- For 140 of the 200 NONE scenes, the `blocker` lattice shows a set of objects that frees T when moved away. At least one of these (`3dt-plus3dt-w0037-s008`) has been confirmed solvable with 3 moves executed in MuJoCo (o03 → (2,6), o00 → (3,7), o01 → (6,2), then T). It is really a K3 scene.
- The other 60 have no clearing set even with every pool object moved, so they are much more likely to be genuinely NONE.

A re-check of the NONE class (fuller search: all orders, alternative parking cells) is planned. Until then, treat NONE labels with care.

## Not done yet

- The combined one-file-per-scene export (`code/make_combined_3dt.py`, format of `3dt_20_combined`).
- Stage 2: three redundant-move variants and up to five verified solutions per base scene.
- The NONE re-check above.

No GIFs are stored in this dataset. Renders can be made locally with `code/viz_3dt.py`.

## Viewing a scene

The 3D code needs MuJoCo (3.x), NumPy, this branch's `gh-yadu-iiith` folder (`table_scene (1).py` and `random/`) and MuJoCo Menagerie's `ufactory_xarm7`. If the xArm7 model isn't next to the scene file, set `XARM7_XML` to its path.

```bash
cd gh-yadu-iiith/open_tabletop/dataset_bd_classquota
export BD_RSS3D_ROOT=$(cd ../.. && pwd)          # the gh-yadu-iiith folder
python code/view_3dt.py --dir raw                # counted scenes per class
python code/view_3dt.py --dir raw K3 0           # opens the first K3 scene; Enter replays its verified moves
```
