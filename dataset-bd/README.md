# B/D dataset creation scripts

These scripts build the factor-controlled datasets for methods B and D, in 2D and in 3D on the open tabletop. Each root scene gives:

- **B:** ordered correction sequences;
- **D:** queries about the original target T, on the original scene and on hypothetical edited scenes.

## Label rules

- **Binary labels.** y = 1 means the move was verified within the declared teacher budget; y = 0 otherwise. A y = 0 label carries a `negative_kind` field: `CERTIFIED`, `TIMEOUT`, `NO_PLAN_FOUND` or `OTHER_FAILURE`.
- **At most 7 pick-and-place actions** per task: at most 4 necessary relocations, at most 2 redundant ones, and T is moved last.
- **Verified K, not requested K.** Coverage and splits count the K that was actually verified. A root with no verified repair is recorded as `NONE`.
- **Side grasps only.** IK and motion planning run offline, in the teacher only.

## Files

| File | Role |
|---|---|
| `bd_core.py` | Factor specification, the deterministic wave sampler, ids, the binary label protocol, the witness store (NPZ), atomic per-root JSON files, the ledger, manifests and grouped splits. |
| `bd_2d.py` | 2D domain adapter. It needs a `latent-causal-tamp` checkout on `PYTHONPATH` (`config`, `geometry`, `motion`, `p16_*`, `p6_scenes`, …). |
| `bd_3dt.py` | **3D open-tabletop** domain adapter. It wraps `gh-yadu-iiith/table_scene (1).py` from `yadu/dataset-3d` (b3b5b34) **without modifying it**: 8×8 grid, 7 cm pitch, side grasps from the scene's 8 directions. Each scene's source-file hashes are stored in the record. |
| `bd_3d.py` | The earlier 3D cabinet adapter (`grid_dataset.py`). It is superseded by `bd_3dt.py` and kept for reference. |
| `bd_run.py` | The resumable runner. It plans waves from the ledger and runs one spawned worker per root, each with an administrative wall-clock cap. It takes a `STOP` file for a clean stop and writes `ABORTED.json` to guard against runaway runs. |
| `bd_kfail.py` | Shows which requested K fail, per requested K: rejections by reason, the verified-K distribution, and the causes of `NONE`. |
| `bd_viz.py` | Dashboard and scene galleries. |
| `server/supervise.sh` | A supervisor that restarts the runner automatically; run it inside tmux. |
| `tests/test_bd_dataset.py` | Tests for label semantics, the action cap, verified vs requested K, grouped splits and reproducibility. |

## Run

```bash
export MUJOCO_GL=egl XARM7_XML=<mujoco_menagerie>/ufactory_xarm7/xarm7.xml
export BD_RSS3D_ROOT=<rss-ebm-infeasibility checkout>/gh-yadu-iiith      # for bd_3dt
python bd_run.py --domain 3dt --name smoke3dt --roots 16 --workers 8 --wave-size 16 --out <out_dir>
tmux new -d -s bd_mock3dt "bash server/supervise.sh 3dt mock3dt_1000 1000 8 32 <out_dir>"
touch <out_dir>/STOP                                                          # clean stop after the current wave
python bd_kfail.py <out_dir>                                                  # which K0-K4 fail
```

Each output directory contains:

- `roots/<root_id>.json`
- `witnesses/<aa>/<sha>.npz`
- `ledger.jsonl`
- `PROGRESS.json`
- `MANIFEST.json`
- `SPLITS.json`
- `KFAIL.json`

## How the 3D tabletop teacher works (`bd_3dt.py`)

- **Construction.** Intended blockers are placed greedily in the mechanism's cells (grasp, place, transfer or mixed). Each blocker must remove at least one of T's remaining feasible side grasps, while leaving enough grasps for the later blockers, so that each requested blocker is needed by construction.
- **D labels.** For a D label, the edited obstacles are teleported to their parking cells, and then only T is re-planned from the ready pose.
- **Verified K.** Verified K comes from a breadth-first search over sets of relocated objects. A set only counts if it can be reached through relocations that can each be planned in the scene they face. The sequence is then executed in MuJoCo (`run_plan`). A step counts only if the moved object ends in its destination cell, upright, and no other object changed cell.
- **Planner limits.** The scene's RRT-Connect wall-clock limits are replaced by its iteration ceiling, so the labels do not depend on server load.
