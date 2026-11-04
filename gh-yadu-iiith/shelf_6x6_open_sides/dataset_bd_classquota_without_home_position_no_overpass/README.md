# 3D shelf B/D dataset: 6 × 6 open-sided shelf, without home position, no overpass (class-quota run)

This is a dataset of shelf rearrangement problems for an xArm7 in MuJoCo, built with the B/D dataset scripts (`code/`).

- **The task:** each scene has a target object T that must reach a goal cell on the shelf of [`../README.md`](../README.md). Other objects may block it, and the label says how many objects must be moved out of the way first (K).
- **Without home position:** the arm never returns to its ready pose between moves.
- **No overpass:** the arm and hand never pass over an obstacle.

**Status:** generation started on 4 Nov 2026 at 14:37 IST. It runs on a laptop (5 workers) and the lab server (18 workers), which share one ledger. This commit holds the code. The data (`raw/`, `selected.json`) will be added here when the class quotas are met.

## Target

**2,500 counted scenes.** These are the class shares of the 1,225-scene open-tabletop dataset (`../../open_tabletop/dataset_bd_classquota/`), scaled by 2500 / 1225 and rounded by largest remainder.

| Class | Meaning | Scenes | Share |
|---|---|---|---|
| NONE | no verified repair found | 408 | 16.3 % |
| K0 | T reaches its goal directly | 511 | 20.4 % |
| K1 | 1 object must be moved first | 357 | 14.3 % |
| K2 | 2 objects | 357 | 14.3 % |
| K3 | 3 objects | 357 | 14.3 % |
| K4 | 4 objects | 510 | 20.4 % |

See [Open points](#open-points) about K4.

## Rules every counted scene satisfies

- Every object is movable, and each object is moved at most once.
- **Independent moves:** every object of the minimal repair can be picked and parked both from the original scene and with the other objects of the repair already parked, so the moves work in any order. Scenes where a move needs another one first are `DEPENDENT` and are not counted.
- **Executed labels:** labels come from moves executed in MuJoCo, under the scene's rules: no overpass, arm ceiling, no return to the ready pose.
- **Consistent labels** (new in this run): a scene is not counted when an executed sequence that is not the minimal repair succeeds with fewer moves than its K, or with any number of moves for NONE. Such scenes are recorded as `INCONSISTENT`. In the 10-scene sample, a NONE scene was solved by "move one unrelated object, then T", because the arm then started its approach from somewhere else.

## How a scene is made

1. **Request.** The runner (`code/run_3dt_plus.py`, wrapped by `code/shelf66_full/run_shelf66_full.py`) plans each request from the shared ledger: the class whose remaining quota and observed yield call for it. The other factors (start area, move length, distractors, mechanism, parking) are balanced by `bd_core.plan_wave`.
2. **Construction.**
   - K0 requests use `bd_3dt.construct`. NONE is requested as a K0-type construction with 3–4 distractors. In the sample, about a third of those came out NONE.
   - K ≥ 1 requests use the route construction of `bd_3dt_plus.py`. It first probes which cells block which of T's feasible grasps. Then it builds layouts in which every feasible grasp is blocked by at least K objects, and the K objects of the cheapest repair can each be picked and parked on their own.
3. **Teacher.** `bd_3dt.teach` labels the scene by planning and executing in MuJoCo. It searches repairs of up to 4 moves over a pool of at most 6 objects (8 for K4 requests). Each planner query uses RRT-Connect with 5,000 iterations and no wall-clock limit.
4. **Checks.** The runner then applies the `DEPENDENT` and `INCONSISTENT` checks above. A scene counts only when its verified class still has room in its quota.

Versions stored in every scene file:
- generator `BD_GEN_V1` + `BD_GEN_V1+ROUTE_V15+SHELF66V2` (`generator_version_plus`);
- teacher `BD3DS66_TEACHER_V1` (moves chained, no return to the ready pose);
- robot protocol `XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022` (a hash of the scene geometry);
- label protocol `BD_BINARY_BUDGET_V1`;
- config hash `2c61fe5f5032e98d`.

### Changes against the open-tabletop generator (ROUTE_V15)

- **SHELF66:** the grid size comes from the scene (6 × 6). Limits on the number of feasible grasps are counted in 45° units, because the shelf has 16 directions. A direction counts once even when both grip heights plan.
- **SHELF66V2:**
  - **Distractors:** if the requested area (near T, near the goal, near the path, far) has too few free cells on the 6 × 6 grid, the remaining distractors are placed anywhere on the shelf. If there is still too little room, as many are placed as fit. Before, the finished layout was thrown away. Each distractor's role is the area it really stands in, and `construction.distractors` records the requested and placed counts.
  - **Time:** K3 constructions get 45 min instead of 30.
  - **Probing:** a direction is not re-planned at the second grip height once it is feasible. The result is the same, with fewer plans.
  - **NONE through K0:** NONE is requested through K0-type constructions. The open-tabletop runner builds NONE through K4 constructions, which never finish on this shelf.
  - **K4 cap:** at most 2 K4 constructions run at once on the server, and none on the laptop.
  - **Checks and live control:** the `INCONSISTENT` check, and live quota changes through `QUOTA.json`.

## Folder layout

This is the layout once the data is added. `raw/` and `selected.json` are not in this commit yet.

```
dataset_bd_classquota_without_home_position_no_overpass/
  README.md
  selected.json            counted scene ids per class, with the quotas
  raw/                     the run's dataset-bd store, unchanged
    ledger.jsonl           one line per attempt: spec, accepted, verified class, reason, wall time, distractors placed
    PROGRESS.json          final counts and quotas
    roots/<root_id>.json   one file per accepted scene: layout, labels, construction metadata
    witnesses/<ab>/<id>.npz   verified robot trajectories referenced by the labels
  code/
    shelf66_full/          bd_shelf.py (shelf domain), bd_3dt.py (teacher), bd_3dt_plus.py (construction),
                           run_shelf66_full.py (runner); shelf_scene_6x6 -> ../../../scene
    dataset-bd/bd_core.py  B/D dataset core (ids, store, spec factors, plan_wave)
    run_3dt_plus.py        the class-quota runner the wrapper builds on
    view_scene66.py        replays a scene in the MuJoCo viewer (or headless)
    run_3ds66_full.sh, run_3ds66_full_server.sh, merge_with_server_3ds66.sh, merge_ledgers.py, append_lines.py
                           how it was run: laptop + server, one ledger merged every 5 min
    requirements.txt
```

Scene files and labels have the same format as in the open-tabletop dataset; its README describes every field and label. In addition, scene files here carry:

- `label_consistency`: the `INCONSISTENT` check above;
- `construction.distractors`: distractors requested, placed and where;
- `geometry`: rows, cols, `grasp_at`, `no_overpass`, `arm_ceiling`, `ceiling_margin` and `side_walls`.

Root ids are `3ds66-lap-…` (laptop) and `3ds66-srv-…` (server).

## Run it

```
pip install -r code/requirements.txt          # Python 3.10
cd code
python shelf66_full/run_shelf66_full.py --out ../raw --name lap --workers 5      # resumable; stop: touch ../raw/STOP
python shelf66_full/run_shelf66_full.py --out ../raw --select-only               # selected.json from the ledger
python view_scene66.py ../raw/roots/<root_id>.json [--no-viewer]
```

Live controls in the run folder, with no restart needed:
- `MAX_RUNNING`: the number of workers;
- `K4_MAX_RUNNING`: how many K4 constructions may run at once;
- `QUOTA.json`: class quotas, e.g. `{"4": 0, "3": 600}`.

## Open points

- **K4:** on this shelf no K4 construction has succeeded so far. In the 10-scene sample, all 47 tries hit the 30-min construction cap. A K4 layout needs about 8 blockers on the 36 cells, and the 4 that must move were never all pickable with the others in place. K4's 510 scenes may therefore not be reachable. The quota can be changed in place (`QUOTA.json`) once that is decided.
- **NONE:** as in the open-tabletop dataset, NONE means that no repair was found within the teacher's search (pool and depth above, T planned from the ready pose). It is not a proof that no repair exists. The `INCONSISTENT` check removes NONE scenes that a short executed sequence contradicts.
