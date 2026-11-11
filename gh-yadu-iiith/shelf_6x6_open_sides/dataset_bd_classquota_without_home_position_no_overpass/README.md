# 3D shelf B/D dataset: 6 × 6 open-sided shelf, without home position, no overpass (class-quota run)

This is a dataset of shelf rearrangement problems for an xArm7 in MuJoCo, built with the B/D dataset scripts (`code/`).

- **The task:** each scene has a target object T that must reach a goal cell on the shelf of [`../README.md`](../README.md). Other objects may block it, and the label says how many objects must be moved out of the way first (K).
- **Without home position:** the arm never returns to its ready pose between moves.
- **No overpass:** the arm and hand never pass over an obstacle, and over the shelf they stay below the tallest object's top + 10 cm (arm ceiling).

## Status: final (11 Nov 2026): 2,414 counted scenes; K3 stopped at 517 of 612

| Class | Meaning | Counted | Quota |
|---|---|---|---|
| NONE | no repair of ≤ 4 moves exists (certified, see [NONE](#none-certified-and-trimmed)) | 408 | 408 ✔ |
| K0 | T reaches its goal directly | 511 | 511 ✔ |
| K1 | 1 object must be moved first | 357 | 357 ✔ |
| K2 | 2 objects | 612 | 612 ✔ |
| K3 | 3 objects | 517 | 612 (stopped, 95 short) |
| K4 (ordered) | 4 objects, moves need an order | 9 | none (was 510; see [K4 ordered](#k4-ordered)) |
| **Total** | | **2,414** | **2,500** |

The original quotas were the class shares of the 1,225-scene open-tabletop dataset (`../../open_tabletop/dataset_bd_classquota/`), scaled by 2500 / 1225: NONE 408, K0 511, K1 357, K2 357, K3 357, K4 510. **K4 was not reached on this shelf** (see [Open points](#open-points)), so its 510 slots were moved to K2 and K3 (255 each, 9 Nov 2026), which keeps the total at 2,500. K2 reached 612. **K3 was stopped at 517 of 612** on 11 Nov 2026 by the user's decision (no more scenes were needed), so the total is 2,414 instead of 2,500. Both machines were stopped, and scenes that were still being constructed were discarded. The 539 new K2 and K3 scenes (1,902 objects) passed the fixed-object check on 11 Nov (every object can be picked alone). `raw/` also holds every other accepted scene of the run, beyond the counted ones: 566 more K0, 212 more K1 and 27 more K2 scenes, besides the DEPENDENT, NONE_SOLVABLE and INCONSISTENT ones listed below.

`selected.json` lists the counted scene ids per class and the file each class comes from (`raw/` for K0–K3, `raw_none/` for NONE, `raw_k4_ordered/` for K4).

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

## NONE: certified and trimmed

The first NONE scenes of this dataset (the 3 in the earlier commit, found by random clutter and a first audit) turned out to be K1: the audit tried only 6 parking cells per object, and one relocation to a cell it never tried freed T. They were dropped. The NONE class (`raw_none/`, 408 scenes) is built differently:

1. **Dense construction** (`code/shelf66_v3/none_dense.py`). For a start/goal pair, F0 = the grasp directions in which T alone (on the empty shelf, under the scene's arm ceiling) can be moved to its goal. Candidate cells around the start, the goal and the path are probed with one standard obstacle each, and cells are chosen until every direction of F0 is blocked by at least 5 chosen cells, each of which blocks it on its own. Every obstacle must be pickable alone on the empty shelf.
2. **Certificate** (`code/shelf66_v3/none_cert66_v4.py`, `NONE_CERT66_V4`). With the real obstacles: `blockers[a]` = the obstacles that, with only T and that obstacle on the shelf, make direction `a` fail. The scene is proven NONE when every direction of F0 has at least 5 blockers. An obstacle only adds collisions, so after any ≤ 4 moves (each object moved at most once) every direction still has one of its blockers in place. On top of that the audit tries T directly with a stronger planner (20,000 RRT iterations, 3 seeds) and every single obstacle moved first to its nearest 6 free cells, then T; both must fail. Status `CONFIRMED` is stored in each scene's `none_audit`.
3. **Trim** (`code/shelf66_v3/trim_none66_v3.py`, `NONE_TRIM66_V3`). The dense scenes held 13–14 obstacles, while K0–K3 scenes hold 0–7, which would give the class away by counting objects. So each scene keeps only the smallest set of obstacles that still meets the certificate (the tallest obstacle is always kept, so the arm ceiling stays the same), the obstacles are renumbered, and the scene is labelled again from scratch: V3 teacher, the consistency checks, and the certificate and audit of step 2 on the trimmed scene. At most 5 obstacles are marked `intended_blocker` (the teacher's checks need ≤ 7 actions). The record's `trim` field keeps the source scene (`from_root`), the id map and the removed obstacles.

`raw_none/` also holds 18 further certified NONE scenes that are not counted (the trim produced 426 in all; listed in `selected.json` under `none_extra`). The 408 counted scenes have 5 to 12 obstacles (median 7): 5: 4, 6: 112, 7: 96, 8: 103, 9: 60, 10: 26, 11: 5, 12: 2. Root ids are `3ds66-lapNt-…` / `3ds66-srvNt-…`.

**A limit of the proof.** The blocker tests use a randomised planner, so a direction that looks blocked could in rare cases be open. The trim found two such scenes: the teacher found a 4-move repair (K4, with moves that need an order). They are kept as ordered K4 instead of NONE. The other scenes passed the teacher's repair search and both audit checks, but the proof is not airtight.

## Rules every counted scene satisfies

- Every object is movable, and each object is moved at most once.
- **Independent moves:** every object of the minimal repair can be picked and parked both from the original scene and with the other objects of the repair already parked, so the moves work in any order. Scenes where a move needs another one first are `DEPENDENT` and are not counted.
- **Executed labels:** labels come from moves executed in MuJoCo, under the scene's rules: no overpass, arm ceiling, no return to the ready pose.
- **Consistent labels:** a scene is not counted when an executed sequence that is not the minimal repair succeeds with fewer moves than its K, or with any number of moves for NONE (`INCONSISTENT`).
- **NONE is certified** (above) and **no object is fixed**: every object of every counted scene can be picked alone on the empty shelf under the scene's own arm ceiling (all 10,761 objects of the K0–K3 scenes accepted after 5 Nov and of the dense NONE scenes were checked, 0 fixed).

## The raw ledger so far

`raw/ledger.jsonl` (K0–K3 run; `raw_none/ledger.jsonl` holds the 408 NONE scenes' lines)

`raw/ledger.jsonl` has one line per attempt: 4,071 lines (laptop and server runs merged), all labelled by generator `BD_GEN_V1+ROUTE_V15+SHELF66V3`. Of these, 1,088 are re-taught V2 scenes and 2,983 are new. It holds every line of the main run, so it also covers scenes that are not in the selection.

| Outcome | Lines |
|---|---|
| K0 / K1 / K2 / K3 | 1,077 / 569 / 639 / 517 |
| NONE lines of this run (not used, see NONE above) | 9 audit-confirmed, 73 NONE_SOLVABLE, 3 NONE_UNAUDITED |
| DEPENDENT | 108 |
| INCONSISTENT | 36 |
| rejected during construction | 1,040 (891 hit the construction time cap, 91 had too small a distractor role pool, 40 found no blocking layout, 18 had T infeasible on the empty table) |

The selection takes accepted scenes in ledger order up to each quota, so K0, K1 and K2 have more accepted scenes than are counted. K3 counts all 517 of its accepted scenes.

Root ids: `3ds66-lap-…` / `3ds66-srv-…` (V2 runs, re-taught) and `3ds66-lap3-…` / `3ds66-srv3-…` (V3 runs).

## How a scene is made

1. **Request.** The runner (`code/run_3dt_plus.py`, wrapped by `code/shelf66_v3/run_shelf66_full.py`) plans each request from the shared ledger: the class whose remaining quota and observed yield call for it. The other factors (start area, move length, distractors, mechanism, parking) are balanced by `bd_core.plan_wave`.
2. **Construction.**
   - K0 requests use `bd_3dt.construct`. (The NONE class is built separately, see [NONE](#none-certified-and-trimmed).)
   - K ≥ 1 requests use the route construction of `bd_3dt_plus.py`. It first probes which cells block which of T's feasible grasps. Then it builds layouts in which every feasible grasp is blocked by at least K objects, and the K objects of the cheapest repair can each be picked and parked on their own.
3. **Teacher.** `bd_3dt.teach` labels the scene by planning and executing in MuJoCo. It searches repairs of up to 4 moves over a pool of at most 6 objects (8 for K4 requests). Each planner query uses RRT-Connect with 5,000 iterations and no wall-clock limit.
4. **Checks.** `DEPENDENT`, `INCONSISTENT`, then the NONE audit for a NONE label. A scene counts only when its verified class still has room in its quota.

Versions stored in every scene file:
- generator `BD_GEN_V1` + `BD_GEN_V1+ROUTE_V15+SHELF66V3` (`generator_version_plus`);
- teacher `BD3DS66_TEACHER_V2` (fingers pre-shaped in every check, moves chained, no return to the ready pose);
- robot protocol `XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022` (a hash of the scene geometry);
- label protocol `BD_BINARY_BUDGET_V1`;
- config hash `8d521b1f6d23db23`.

## K4 ordered

No K4 scene with independent moves could be built (see [Open points](#open-points)). Nine scenes have a 4-move repair (2 from the NONE trim, 7 by-products of the main run) that the teacher executed in the simulator, but whose moves only work in the stored order. By the user's decision (10 Nov 2026) such scenes are counted as **K4, labelled ordered (DEPENDENT)**. They live in `raw_k4_ordered/` (same layout as `raw/`; ledger lines carry `"class": "K4_ORDERED"`). The independence rule of K0–K3 does not apply to them: do not use them where independent moves are required.

| Scene | Obstacles | T → goal | Repair (in this order) |
|---|---|---|---|
| `3ds66-srvNt-w0009-s012` | 7 | (1,1) → (2,5) | o01 → (3,3), o02 → (3,1), o00 → (3,0), o04 → (3,5) |
| `3ds66-srvNt-w0013-s010` | 6 | (3,1) → (5,5) | o01 → (1,4), o03 → (5,1), o00 → (4,0), o04 → (3,3) |
| `3ds66-srv-w0003-s004` | 4 | (5,1) → (3,5) | o00 → (2,0), o03 → (1,3), o01 → (2,2), o02 → (1,2) |
| `3ds66-srv-w0044-s014` | | | o02 → (3,4), o03 → (5,0), o01 → (3,0), o00 → (2,0) |
| `3ds66-lap3-w0051-s013` | | | o01 → (5,4), o02 → (3,5), o03 → (2,5), o00 → (1,3) |
| `3ds66-srv3-w0078-s012` | | | o01 → (4,0), o03 → (2,4), o04 → (5,1), o00 → (1,3) |
| `3ds66-srv3-w0082-s016` | | | o02 → (5,4), o04 → (5,5), o00 → (1,3), o01 → (3,4) |
| `3ds66-srv3-w0089-s022` | | | o01 → (1,2), o02 → (5,5), o04 → (1,1), o00 → (3,0) |
| `3ds66-srv3-w0101-s007` | 7 | (5,4) → (1,3) | o01 → (3,0), o05 → (2,0), o00 → (1,0), o02 → (5,1) |

## Folder layout

```
dataset_bd_classquota_without_home_position_no_overpass/
  README.md
  selected.json            counted scene ids per class, quotas, counts, skipped fixed-object scenes
  raw/                     the K0-K3 run's dataset-bd store
    ledger.jsonl           one line per attempt: spec, accepted, verified class, reason, wall time, distractors placed
    PROGRESS.json          the runner's counts and quotas at the snapshot
    EXCLUDED_FIXED_OBJECT.json   scenes with an object that can never be picked (not counted)
    roots/<root_id>.json   one file per accepted scene: layout, labels, construction metadata, NONE audit
    witnesses/<ab>/<id>.npz   verified robot trajectories referenced by the labels
  raw_k4_ordered/          ordered K4 scenes (roots, witnesses, ledger)
  raw_none/                the NONE scenes: ledger.jsonl (their lines), roots/, witnesses/ (408 counted + 18 further certified NONE scenes, see selected.json `none_extra`)
  code/
    shelf66_v3/            bd_shelf.py (shelf domain), bd_3dt.py (teacher), bd_3dt_plus.py (construction),
                           run_shelf66_full.py (runner), none_audit66_v3.py (NONE audit),
                           reteach_shelf66_v3.py (re-teach of the V2 scenes),
                           none_dense.py, none_cert66_v4.py, run_shelf66_none.py, trim_none66_v3.py (NONE: construction, certificate, runner, trim);
                           shelf_scene_6x6 -> ../../../scene
    dataset-bd/bd_core.py  B/D dataset core (ids, store, spec factors, plan_wave)
    run_3dt_plus.py        the class-quota runner the wrapper builds on
    view_scene66.py        replays a scene in the MuJoCo viewer (or headless)
    run_3ds66_none*.sh, run_trim66_none*_v3.sh, run_3ds66_v3_k23.sh (the NONE runs, the trim, the K2/K3 run with the moved K4 quota),
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

- **K4 (0 of the original 510, quota moved to K2 and K3):** no K4 construction succeeded on this shelf (V2 + V3: 0 of 36, all at the construction time cap). Tests that also allowed ordered moves, splitting the blockers, and building K4 from a K3 scene (0 of 18) found none either. Every feasible grasp direction has to be blocked 4 times, which needs 8–15 objects packed around the start and goal; with the arm reaching in only from the front, the inner blockers cannot then be reached or parked without hitting their neighbours. The trim step did turn up two small scenes (6–7 obstacles) with a 4-move repair, but their moves need an order, so they are not K4 under the independence rule. They are now counted as ordered K4 (see [K4 ordered](#k4-ordered)).
- **K3 (517 of 612):** generation was stopped on 11 Nov 2026 with 95 scenes of the quota still missing. K3 is the slow class (up to 5 hours per scene at the construction time cap), and the lab server could not be used for it: its watchdog stops our runs above 100 °C, and starting even 4–8 workers pushed it over that limit. K2 is complete (612).
