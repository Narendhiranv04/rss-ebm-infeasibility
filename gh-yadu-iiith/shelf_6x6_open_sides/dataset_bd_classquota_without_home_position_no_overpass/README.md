# 3D shelf B/D dataset: 6 × 6 open-sided shelf, without home position, no overpass (class-quota run)

This is a dataset of shelf rearrangement problems for an xArm7 in MuJoCo, built with the B/D dataset scripts (`code/`).

- **The task:** each scene has a target object T that must reach a goal cell on the shelf of [`../README.md`](../README.md). Other objects may block it, and the label says how many objects must be moved out of the way first (K).
- **Without home position:** the arm never returns to its ready pose between moves.
- **No overpass:** the arm and hand never pass over an obstacle, and over the shelf they stay below the tallest object's top + 10 cm (arm ceiling).

## Status: in progress (K3 completed 8 Nov 2026; NONE and K4 still open)

Generation is still running on the lab server. This commit holds everything generated so far. The data will be updated here when the class quotas are met.

| Class | Meaning | Counted now | Quota |
|---|---|---|---|
| NONE | no repair of ≤ 4 moves exists (audit-confirmed) | 3 | 408 |
| K0 | T reaches its goal directly | 511 | 511 ✔ |
| K1 | 1 object must be moved first | 357 | 357 ✔ |
| K2 | 2 objects | 357 | 357 ✔ |
| K3 | 3 objects | 357 | 357 ✔ |
| K4 | 4 objects | 0 | 510 |
| **Total** | | **1,585** | **2,500** |

The quotas are the class shares of the 1,225-scene open-tabletop dataset (`../../open_tabletop/dataset_bd_classquota/`), scaled by 2500 / 1225 and rounded by largest remainder. `selected.json` lists the counted scene ids per class.

Still open (see [Open points](#open-points)): NONE needs a new construction, and K4 has not been reached on this shelf.

## What changed since the code commit (V2 → V3)

The code commit (9e64069) held the SHELF66V2 generator. That version had a planner bug, and every scene was labelled again.

- **The bug:** the planner checked collisions with the gripper fingers as they happened to be. Right after a scene reset they are fully open (about 8.5 cm), while every executed pick pre-shapes them to the object's width + 1.2 cm. So every check at the start of a scene (the teacher's first query of T, its "what if this object were moved" edits, its relocation checks) saw a hand several cm too wide, and called feasible moves infeasible. An audit of the V2 NONE scenes found 16 of 30 to be K0, and 5 of 20 K1 scenes to be K0.
- **The fix (SHELF66V3):** `scene/shelf_scene.py` pre-shapes the fingers before every pick, and every check uses the fingers at that opening.
- **Re-teach:** all 1,088 accepted V2 scenes were labelled again by the V3 teacher. The layouts stayed the same, and the old label is kept in each ledger line (`retaught_from`). Of the 86 V2 NONE scenes, only 4 stayed NONE. 55 turned out to be K0–K3, 17 are solvable (`NONE_SOLVABLE`), 9 are `DEPENDENT`, and 1 could not be audited in time.
- **NONE audit gate (new):** the teacher's NONE only means "no repair found within its search". A NONE scene now counts only when an independent audit (`code/shelf66_v3/none_audit66_v3.py`, run in the worker, 1 h budget) confirms it. The audit tries much harder to solve the scene:
  - T directly, with a stronger planner (20,000 RRT iterations, 3 seeds);
  - every single relocation first;
  - every set of up to 4 objects whose removal frees T, in every order, with up to 3 parking cells each, executed in MuJoCo.

  Otherwise the scene is `NONE_SOLVABLE` (the audit found a solution) or `NONE_UNAUDITED` (out of time), and neither is counted.
- **Fixed objects (new rule):** a scene is not counted when one of its objects can never be picked, even alone on the empty shelf under the scene's own arm ceiling. 3 scenes have one (a short object in the back-right cell): `3ds66-srv-w0044-s020` (NONE), `3ds66-srv-w0004-s021` (K1) and `3ds66-srv-w0030-s007` (K0). They are listed in `raw/EXCLUDED_FIXED_OBJECT.json` and skipped in `selected.json`; the next scene of the same class takes each place. Scenes accepted after 5 Nov 09:20 have not been through this check yet; that is done before the final selection.

## Rules every counted scene satisfies

- Every object is movable, and each object is moved at most once.
- **Independent moves:** every object of the minimal repair can be picked and parked both from the original scene and with the other objects of the repair already parked, so the moves work in any order. Scenes where a move needs another one first are `DEPENDENT` and are not counted.
- **Executed labels:** labels come from moves executed in MuJoCo, under the scene's rules: no overpass, arm ceiling, no return to the ready pose.
- **Consistent labels:** a scene is not counted when an executed sequence that is not the minimal repair succeeds with fewer moves than its K, or with any number of moves for NONE (`INCONSISTENT`).
- **NONE is audit-confirmed** and **no object is fixed** (both above).

## The raw ledger so far

`raw/ledger.jsonl` has one line per attempt: 2,218 lines, all labelled by generator `BD_GEN_V1+ROUTE_V15+SHELF66V3`. Of these, 1,088 are re-taught V2 scenes and 1,130 are new. It holds the lines of the roots in the selection; the run's full ledger has more.

| Outcome | Lines |
|---|---|
| K0 / K1 / K2 / K3 | 540 / 405 / 396 / 357 |
| NONE (audit-confirmed) | 4 |
| NONE_SOLVABLE / NONE_UNAUDITED | 28 / 1 |
| DEPENDENT | 50 |
| INCONSISTENT | 22 |
| rejected during construction | 415 (374 hit the construction time cap, 35 found no blocking layout) |

The selection takes accepted scenes in ledger order up to each quota, so K0, K1 and K2 have more accepted scenes than are counted.

Root ids: `3ds66-lap-…` / `3ds66-srv-…` (V2 runs, re-taught) and `3ds66-lap3-…` / `3ds66-srv3-…` (V3 runs).

## How a scene is made

1. **Request.** The runner (`code/run_3dt_plus.py`, wrapped by `code/shelf66_v3/run_shelf66_full.py`) plans each request from the shared ledger: the class whose remaining quota and observed yield call for it. The other factors (start area, move length, distractors, mechanism, parking) are balanced by `bd_core.plan_wave`.
2. **Construction.**
   - K0 requests use `bd_3dt.construct`. NONE is requested as a K0-type construction with 3–4 distractors.
   - K ≥ 1 requests use the route construction of `bd_3dt_plus.py`. It first probes which cells block which of T's feasible grasps. Then it builds layouts in which every feasible grasp is blocked by at least K objects, and the K objects of the cheapest repair can each be picked and parked on their own.
3. **Teacher.** `bd_3dt.teach` labels the scene by planning and executing in MuJoCo. It searches repairs of up to 4 moves over a pool of at most 6 objects (8 for K4 requests). Each planner query uses RRT-Connect with 5,000 iterations and no wall-clock limit.
4. **Checks.** `DEPENDENT`, `INCONSISTENT`, then the NONE audit for a NONE label. A scene counts only when its verified class still has room in its quota.

Versions stored in every scene file:
- generator `BD_GEN_V1` + `BD_GEN_V1+ROUTE_V15+SHELF66V3` (`generator_version_plus`);
- teacher `BD3DS66_TEACHER_V2` (fingers pre-shaped in every check, moves chained, no return to the ready pose);
- robot protocol `XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022` (a hash of the scene geometry);
- label protocol `BD_BINARY_BUDGET_V1`;
- config hash `8d521b1f6d23db23`.

## Folder layout

```
dataset_bd_classquota_without_home_position_no_overpass/
  README.md
  selected.json            counted scene ids per class, quotas, counts, skipped fixed-object scenes
  raw/                     the run's dataset-bd store
    ledger.jsonl           one line per attempt: spec, accepted, verified class, reason, wall time, distractors placed
    PROGRESS.json          the runner's counts and quotas at the snapshot
    EXCLUDED_FIXED_OBJECT.json   scenes with an object that can never be picked (not counted)
    roots/<root_id>.json   one file per accepted scene: layout, labels, construction metadata, NONE audit
    witnesses/<ab>/<id>.npz   verified robot trajectories referenced by the labels
  code/
    shelf66_v3/            bd_shelf.py (shelf domain), bd_3dt.py (teacher), bd_3dt_plus.py (construction),
                           run_shelf66_full.py (runner), none_audit66_v3.py (NONE audit),
                           reteach_shelf66_v3.py (re-teach of the V2 scenes); shelf_scene_6x6 -> ../../../scene
    dataset-bd/bd_core.py  B/D dataset core (ids, store, spec factors, plan_wave)
    run_3dt_plus.py        the class-quota runner the wrapper builds on
    view_scene66.py        replays a scene in the MuJoCo viewer (or headless)
    run_3ds66_v3.sh, run_3ds66_v3_server.sh, run_reteach66_v3.sh, run_reteach66_v3_server.sh,
    merge_with_server_3ds66_v3.sh, merge_ledgers.py, append_lines.py
                           how it was run: laptop + lab server, one ledger merged every 5 min
    requirements.txt
```

Scene files and labels have the same format as in the open-tabletop dataset; its README describes every field and label. In addition, scene files here carry:

- `label_consistency`: the `INCONSISTENT` check;
- `none_audit`: the NONE audit's result for NONE-labelled scenes;
- `ledger_observed`: the class the runner recorded (after the checks);
- `reteach`: for re-taught V2 scenes, the old generator, teacher and label;
- `construction.distractors`: distractors requested, placed and where;
- `geometry`: rows, cols, `grasp_at`, `no_overpass`, `arm_ceiling`, `ceiling_margin` and `side_walls`.

## Dependencies and running it

Python 3.10 with `mujoco==3.16.0` and `numpy==2.2.6`; everything else is the standard library, and the xArm7 model is in `../scene/mujoco_menagerie/`.

```
pip install -r code/requirements.txt
cd code
python shelf66_v3/run_shelf66_full.py --out ../raw --name lap --workers 5      # resumable; stop: touch ../raw/STOP
python shelf66_v3/run_shelf66_full.py --out ../raw --select-only               # the runner's selection (no fixed-object skip)
python view_scene66.py ../raw/roots/<root_id>.json [--no-viewer]
```

Live controls in the run folder, with no restart needed:
- `MAX_RUNNING`: the number of workers;
- `K4_MAX_RUNNING`: how many K4 constructions may run at once;
- `QUOTA.json`: class quotas, e.g. `{"4": 0, "3": 600}`.

## Open points

- **NONE (3 of 408, to be replaced):** the 3 NONE scenes counted here were later found to be K1 (the V3 audit tried too few parking cells), so none of them will stay. A new certified NONE set is being built and will replace them.
- **NONE, earlier notes:** with the fixed planner, random cluttered K0-type scenes almost never give a confirmed NONE (about 1 %), and the few that do are fragile: in three of the four, removing one single object frees T, and they only count because that object cannot be relocated. A new construction is being built: every grasp direction of T is blocked by at least 5 objects that each block it on their own, so that no 4 moves (not even removing 4 objects outright) can free T. It will be added with its own version string.
- **K4 (0 of 510):** no K4 construction has succeeded on this shelf (V2 + V3: 0 of 36, all at the construction time cap). A 2-hour test that allowed ordered moves (front blocker first) found 0 as well: of 113 layouts checked, none could be cleared even front to back. Every feasible grasp direction has to be blocked 4 times, which needs 8–15 objects packed around the start and goal. With the arm reaching in only from the front, the inner blockers can then not be reached or parked. Whether K4 stays in the quota is still to be decided.
