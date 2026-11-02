# 2D planar B/D dataset (class-quota run)

A dataset of 2D rearrangement problems for a planar 3-link arm, built with the B/D dataset scripts (`code/dataset-bd`). Each scene has a target disc T that must be picked and placed at a goal cell. Other discs may block it, and the label says how many of them must be moved out of the way first (K). Generated on 1–2 Nov 2026 on a laptop and the lab server, which shared one ledger. The class quotas were met on 2 Nov 2026 at 18:19 IST.

This folder holds the **complete raw output** of the run: every scene file, every trajectory and the full attempt ledger. A combined one-file-per-scene export has not been made yet (see [Not done yet](#not-done-yet)). The 3D open-tabletop dataset is in `../../open_tabletop/dataset_bd_classquota/`.

## At a glance

**5,000 counted scenes.** `selected.json` lists them by class.

| Class | Meaning | Scenes | Obstacles per scene (min / median / max) |
|---|---|---|---|
| NONE | no repair exists (see [NONE](#none)) | 1,000 | 2 / 18 / 19 |
| K0 | T reaches its goal directly | 1,250 | 0 / 2 / 12 |
| K1 | 1 object must be moved first | 800 | 1 / 7 / 12 |
| K2 | 2 objects | 800 | 4 / 8 / 12 |
| K3 | 3 objects | 1,150 | 5 / 12 / 14 |

There is no K4 class. K = 4 could not be built with movable objects only (0 of 10 attempts), so its 800 scenes were moved to K3 (+500), K1 (+150) and K2 (+150). A K = 4 scene that came up by chance would still have counted, but none did.

`raw/` also keeps every other attempt (as of this export): 7,218 scene files in total and 13,414 ledger lines. Besides the counted scenes, that includes scenes over a class quota, 131 `DEPENDENT` scenes, and the retired scenes described in [Older scenes](#older-scenes-and-conversions). Rejected constructions appear only in the ledger, with their reasons.

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
    dataset-bd/            B/D dataset library (bd_core, bd_2d teacher, ...); see its README
    bd_2d_plus.py          class-quota construction for K >= 1 (generator BD_GEN_V1+ROUTE2D_V6)
    bd_2d_exact.py         exact layout search for K >= 2 and the certified NONE construction
    run_2d_plus.py, .sh    the runner (plans requests from the ledger until every quota is met)
    bd_2d_convert.py, convert_2d.py, run_convert_2d.sh   conversion of the older scenes to movable-only
    fastgeom.py            faster drop-in for geometry.Checker.distances (same values to rounding)
    py310_shim/tomllib.py  tomllib for Python < 3.11
    latent-causal-tamp/    the planar-arm code the 2D teacher imports (config.toml, geometry, motion, scenes, ...), as used
```

## The scene

- **Robot:** a planar 3R arm. Its base is at (−0.14, 0.32) m, with links of 0.42, 0.34 and 0.20 m and a parallel gripper (opening 0.07 m). It grasps from the side.
- **Workspace:** an 8 × 8 grid of cells with a 0.08 m pitch. Cell `n` is row `n // 8`, column `n % 8`, with its centre at `x = 0.04 + 0.08·col`, `y = 0.04 + 0.08·row`.
- **Objects:** discs of radius 0.024 m, one per cell. The target T is `t0`; the obstacles are `o00`, `o01`, …. **Every object is movable.** No counted scene has immovable structures (`fixed_ids` is empty in all of them).
- **Start:** the arm starts at `q0` = (π/2, π/2, π/2) (`start_pose: home`, stratum S0).

## What a scene file contains (`raw/roots/<root_id>.json`)

| Field | Content |
|---|---|
| `ids` | root id, construction seed, generator / teacher / label-protocol / robot-protocol versions, config hash |
| `spec` | the requested factors: `K`, `mechanism` (grasp / place / mixed / transfer), `source` (near / mid / far), `displacement` (short / medium / long), `distractors`, `distractor_role`, `parking`, `start_pose` |
| `scene` | the grid, legal cells, every object (id, kind, cell, `xy_m`, `radius_m`) and the robot's start state |
| `goal_cell`, `q0`, `start_stratum` | T's goal cell and the arm's start configuration |
| `roles`, `intended_blockers` | each obstacle's role (intended blocker or a distractor type) |
| `parking`, `alt_parking` | each obstacle's parking cells (the teacher uses the first one) and alternatives |
| `verified_K`, `observed` | the verified class: 0–3, or `NONE` |
| `labels` | see below |
| `independent_moves` | true for every counted scene: each object of the minimal repair moves once, and each move works on its own |
| `construction`, `mechanism_pools`, `empty_scene_T_swept`, `empty_scene_T_witness` | how the layout was built (`K_static` is the exact static lower bound on K); the cells T's motion sweeps on an empty grid, and that trajectory |
| `budget`, `teacher_calls`, `teacher_cpu_s`, `geometry_impl` | teacher budget and cost; `fastgeom1` for scenes made after 1 Nov ~18:45 |

### Labels

All labels are binary: `y = 1` means verified within the teacher's budget, and `y = 0` comes with a `negative_kind` (`CERTIFIED`, `TIMEOUT`, `NO_PLAN_FOUND` or `OTHER_FAILURE`). `CERTIFIED` means infeasibility was proven, not just that the planner found nothing.

- **`original`:** can T reach its goal in the scene as it is?
- **`D`:** hypothetical edits. The listed obstacles are placed at their parking cells and only T is re-planned.
- **`B`:** action sequences executed step by step from the initial scene. Each step's trajectory is certified in the scene as it is at that step. Roles:
  - `minimal`: the verified fewest-move repair, then T;
  - `wrong_order`: the repair in reverse order;
  - `partial`: the repair without its last move;
  - `wrong_destination`: one blocker moved to the wrong cell;
  - `alternative`, `alternative_destination`: another clearing set, or the first blocker moved to its second parking cell;
  - `redundant+1`, `redundant+2`: the repair plus one or two unnecessary moves;
  - `noop_irrelevant`: moving a distractor only;
  - `intended_union`: the intended blockers, when no minimal repair exists.

  Each B entry has its `sequence` (`[object, cell]` pairs), `failed_step`, `n_actions` and `witnesses` (the trajectories, one per executed step).
- **`blocker`:** the lattice of edits over the teacher's object pool: the minimal clearing sets, a per-object blocker mask, and whether the supervision is valid.

### Trajectories (`raw/witnesses/<first 2 chars>/<id>.npz`)

`q` (n × 3 joint path), `gap` (gripper opening per waypoint), `phase` (segment name per step), `attached` (holding the object or not), `held_local` (grasp point in the gripper frame), `place_xy` (where the object is placed).

```python
import json, numpy as np
sel = json.load(open("selected.json"))
rid = sel["selected"]["3"][0]                                   # first counted K3 scene
r = json.load(open(f"raw/roots/{rid}.json"))
b = next(b for b in r["labels"]["B"] if b["role"] == "minimal")
print(r["verified_K"], b["sequence"])                           # 3, and the three [object, cell] moves
w = b["witnesses"][0]
traj = np.load(f"raw/witnesses/{w[:2]}/{w}.npz")
print(traj["q"].shape)
```

## How the dataset was made

- **The rules for every scene:**
  - Each object moves at most once.
  - The objects in a repair don't depend on each other: every blocker's relocation is certified from the original scene (arm at `q0`, everything else in place), so the moves work in any order. A verified repair that fails this check is recorded as `DEPENDENT` and never counted.
  - Every object is movable.
  - A task has at most 7 pick-and-place actions.
- **Classes count the verified K, not the requested K.** The runner requests a class according to how often each request has produced each class so far, then keeps a scene under the class the teacher verified.
- **Selection:** the accepted scenes in ledger order, per verified class, up to the quota, after replacing each converted scene by its conversion (`run_2d_plus.effective`). That's what `selected.json` reproduces.
- **Generators:**
  - K0 uses dataset-bd's `bd_2d.construct` unchanged.
  - K1 and transfer K2 use the "route" construction: blockers are added on T's current path until no smaller set of them, moved away, frees T.
  - K2 and K3 with grasp / place / mixed mechanisms use the exact layout search (`bd_2d_exact`): static grasp cut maps and an exact cover, so `K_static` is an exact lower bound.
  - Most counted scenes come from `BD_GEN_V1+ROUTE2D_V6`; earlier versions are listed per scene.
- **Teacher:** `BD2D_TEACHER_V1` (deterministic physics, repair search over a pool of 5 objects).
- **Machines:** 2,027 counted scenes were made on the laptop and 2,973 on the lab server (root ids with `srv`). Per class, laptop / server: NONE 249 / 751, K0 947 / 303, K1 444 / 356, K2 227 / 573, K3 160 / 990.

### NONE

- **974 of the 1,000 NONE scenes are certified infeasible.** They were built so that every (pick grasp, place grasp) pair of T is cut by at least 5 distinct objects (`construction.method = none_cover2d`, `K_static >= 5`). No repair of 4 or fewer moves exists among *any* objects. The target itself works on an empty grid.
- **26 NONE scenes are weaker:** K1 / K2 requests (24 from the route construction, 2 from the exact search) where the teacher found no repair. Like the 3D NONE class, these are "no repair found", not a proof. Their ids can be found with `construction.method != "none_cover2d"`.

### Older scenes and conversions

The first scenes of this run (generator versions before V4) had immovable "context structures" (`fixed_ids`) from dataset-bd's original generator. They were removed:
- 794 of those K0–K3 scenes were converted to all-movable scenes (`bd_2d_convert.py`, generator `…+CONV1`, root ids `2d-conv-…`, ledger lines with `converted_from`). A converted scene counts only where its class is certainly minimal.
- The 892 old NONE scenes were retired and replaced by certified movable-only NONE scenes.

The original files of these scenes are still in `raw/roots/`. They are never counted.

### Spec balance

K0 is balanced in mechanism, start area and move length. The other classes are skewed toward what can be built:
- **NONE:** mechanisms balanced; starts mostly far (far 608, mid 386, near 6).
- **K1:** transfer 305, grasp 203, place 151, mixed 141. Starts and move lengths are roughly even.
- **K2:** place 319, mixed 275, grasp 187, transfer 19. Long moves are fewer (187 of 800).
- **K3:** mixed 743, place 334, grasp 72, transfer 1 (K ≥ 3 is requested only for grasp / place / mixed). Starts: mid 513, far 398, near 239. Lengths: medium 457, short 416, long 277.

## Not done yet

- The combined one-file-per-scene export.
- Stage 2: three redundant-move variants and up to five verified solutions per base scene.

No GIFs are stored in this dataset.

## Running the code

The 2D teacher imports the planar-arm code in `code/latent-causal-tamp` (`config`, `geometry`, `motion`, `p6_scenes`, `p14_*`, `p16_*`, `scene`, …) and must run from that folder so it finds `config.toml`. It needs Python 3.10+ and NumPy. `code/run_2d_plus.sh` shows the run's environment with the laptop's absolute paths. From this folder, the same setup is:

```bash
C=$(pwd)/code
cd code/latent-causal-tamp
export PYTHONPATH=$C/py310_shim:$C/dataset-bd:$C:$C/latent-causal-tamp
python3 $C/run_2d_plus.py --out /tmp/my_2d_run --workers 4     # a new run with the same generator and teacher
```
