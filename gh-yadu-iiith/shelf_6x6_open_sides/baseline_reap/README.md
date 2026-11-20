# REAP on our 6x6 shelf

REAP (Swoboda & Geffner, CoRL 2026, https://github.com/elicited-preconditions/reap) as a baseline on the 6x6 open-sided
shelf (SHELF66V3: no overpass, arm ceiling, no return to ready). Feasibility comes from the shelf66 dataset teacher
itself, not from REAP's Franka grasp library, so REAP is judged by the rules that made the dataset labels.

## Results (V4 split, the 370 test scenes)

Feasibility oracle = the dataset teacher (xArm7, 6x6 shelf, no-overpass, ceiling); every check at test time uses the full teacher, and every plan REAP
finds is replayed in MuJoCo. Columns as in the paper's table (a): AUC = scene solvable (K0-K3) vs NONE from REAP's start-state value, Repair = % of
solved K0-K3 scenes whose plan has exactly K obstacle moves, Task = % of scenes with the verified outcome, Time = seconds per scene, Calls = distinct
feasibility queries per scene.

| REAP | AUC | Repair (%) | Task (%) | Time (s) | Calls |
|---|---|---|---|---|---|
| released confined weights, zero-shot (1 seed) | 0.984 | 55.0 | 73.0 | 168.1 | 46.0 |
| trained from scratch on 600 V4 train scenes (1 seed) | 0.789 | 82.6 | 63.2 | 208 | 39.7 |
| trained, batch 64, solvable scenes only, 6 h label (~14 h in total, 1 seed) | 0.707 | 70.1 | 77.0 | 166 | 63.4 |
| same run, 12 h label (~20 h in total, 1 seed) | 0.793 | 66.4 | 81.9 | 157 | 42.2 |
| same run, 24 h label (~26 h in total, 1 seed) | 0.574 | 63.6 | **82.2** | **150** | 51.1 |

Per-class results, caveats (the trained run uses a precomputed conflict library as its checker) and the failure analysis are in `RESULTS.md`.

## Layout
| path | what |
|---|---|
| `../reap/` | REAP clone (commit d2ba23d), venv `../reap/.venv` (Python 3.12, CPU torch, **mujoco 3.16.0** = the dataset's) |
| `roots66_to_pddl.py` | shelf66 roots (pushed dataset, `selected.json`) -> REAP problem `.pddl` + geometry sidecar `.json`, N per class |
| `shelf66_oracle.py` | pick / put feasibility with `bd_samples/shelf66_v3` (`bd_shelf` -> `bd_3dt.build/settle/query`) |
| `yadu_reap_checker.py` | REAP checker adapter, selected with `CONFINED_CHECKER=yadu` |
| `validate66.py` | the oracle against the labels |
| `replay66.py` | runs each REAP plan the way the teacher runs a sequence (chained, MuJoCo, end-cell / z / displacement checks) |
| `run_zeroshot.sh` | pretrained confined checkpoint -> our problems -> replay |
| `data/` | `domain.pddl` (REAP's confined-shelf domain, unchanged) + problem sets (`shelf66_s50`: 10 per class NONE/K0-K3, seed 0) |

REAP itself changes in two files only (`git -C ../reap diff`): `checker_info.py` accepts `CONFINED_CHECKER=yadu`;
`confined_shelf.py` dispatches to `yadu_reap_checker.py` and records each query in `_cache` (REAP's mp_calls count).

## How REAP's pick / put map onto our moves
- `put(o, d)` = the teacher's move check `bd_3dt.query(o -> d)`: approach, grasp, lift, in-shelf carry, lower, release,
  retreat, with o in its source cell and the others where the state says (planned from the ready pose, like the
  teacher's scene queries). REAP's state forgets the source once o is held, so the oracle remembers it when
  `pick(o, c)` is checked. If two picks reach the same symbolic state, the put is feasible when any remembered
  source works; the replay uses the real ones.
- `pick(o, c)` = a cheap filter: some grasp reaches o in c (free move to hover + straight approach, fingers pre-shaped).
- Goals fix only the target's cell, as in the dataset. REAP's own problems fix every object.

Validation (`validate66.py data/shelf66_s50`): the direct target move is feasible exactly for the K0 roots (50/50), and
every move of every verified minimal sequence passes pick + put (100/100).

## Commands
```bash
python3 roots66_to_pddl.py data/<set> --per-class N [--classes NONE,0,1,2,3] [--seed S]
../reap/.venv/bin/python validate66.py data/<set>
bash run_zeroshot.sh <set> <seed> <workers> <time_limit_s>          # laptop: workers <= 4
```

## Install notes (laptop, 2026-11-13)
- No conda: `pip install --user uv`, `uv python install 3.12`, `uv venv .venv`,
  `UV_TORCH_BACKEND=cpu uv pip install -e . -r requirements-tampanda.txt`, then `uv pip install mujoco==3.16.0`.
  REAP pins mujoco 3.4.0, but its confined checker does not use MuJoCo; ours needs the dataset's version.
- `mifrost` builds from source with Conan. Its nauty recipe downloads from users.cecs.anu.edu.au, which times out from
  here; the recipe was patched to try https://pallini.di.uniroma1.it/nauty2_8_8.tar.gz first
  (`~/.conan2/p/nauty*/e/conanfile.py` and uv's git checkout of mifrost).
- Cap build parallelism: `~/.conan2/global.conf` has `tools.build:jobs=8`; also export
  `CMAKE_BUILD_PARALLEL_LEVEL=8 MAKEFLAGS=-j8`.
- Sanity check: REAP's own confined ID set with its own checker gives 50/50, as in the paper (`outputs/repro`).

## Moving to the GPU server (retraining)
1. Copy `reap_shelf/`, `bd_samples/shelf66_v3/`, `bd_samples/dataset-bd/` (bd_core) and the REAP clone without `.venv`.
   Set `SHELF66_V3_DIR` and `REAP_SHELF_DIR` if the paths differ.
2. Rebuild the venv with the CUDA torch build (drop `UV_TORCH_BACKEND=cpu`), mujoco 3.16.0, and the same nauty / jobs
   fixes.
3. Training set: convert the dataset's train split (or fresh roots) with `roots66_to_pddl.py` into `data/train` and
   `data/validation` (REAP used 600 / 50 for confined), keeping a held-out eval set. Then
   `CONFINED_CHECKER=yadu REAP_DATA=<reap_shelf dir> python scripts/train.py --config configs/confined.yaml --data_path data --num_workers 16`
   (`--data_path` is relative to `$REAP_DATA` and needs `domain.pddl`, `train/`, `validation/`).
   Server limit: 18 cores in total, so 16 workers at most. Each check takes about 2-8 s, against microseconds for
   REAP's library, so training is far slower than their 24 h. Precomputing a grasp library from `shelf66_oracle.py`, as
   tampanda did for REAP, would remove most of that cost.
