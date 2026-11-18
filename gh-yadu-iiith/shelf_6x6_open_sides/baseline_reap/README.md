# REAP on our 6x6 shelf

REAP (Swoboda & Geffner, CoRL 2026, https://github.com/elicited-preconditions/reap) as a baseline on the 6x6 open-sided
shelf (SHELF66V3: no overpass, arm ceiling, no return to ready). Feasibility comes from the shelf66 dataset teacher
itself, not from REAP's Franka grasp library, so REAP is judged by the rules that made the dataset labels.

<!-- all-models-results:start -->
## Results of all models (V4 split, the same 370 test scenes)

Models and where they live (all in this repository, folder `gh-yadu-iiith/shelf_6x6_open_sides/baseline_<name>/`): REAP (`yadu/REAP`),
PIGINet (`yadu/PIGINet`), GRN (`yadu/GRN`), SceneCollisionNet (`yadu/SceneCollisionNet`). Split: `split_v4.json` (Narendhiran's V4 split from `naren/dataset-done`: 1,766 train / 364
val / 370 test scenes). Mean ± standard deviation over seeds.

### Feasibility predictors, one metric set

"T move" columns: can the target reach its goal in that state (AUROC over the stored test queries of that kind: the original scene, the
teacher's edited scenes with obstacles parked, and the rest of the teacher's T-only lattice, L). Scene-level columns enumerate
candidate states / plans (the original scene and every set of 1-3 obstacles sent to their first parking cell, 26 per scene on average);
PIGINet has two input forms (relocations as plan tokens, or the obstacles already parked in the state, which is what GRN sees).
`class acc`: NONE / K0-K3 (chance 0.28), `tau*` chosen on validation. `blocker set top-1`: the best-scoring candidate of the true size K is a
verified minimal set. `blocker-mask F1`: PIGINet from its best-scoring set at tau 0.5, GRN from its own obstruction head (tau_go tuned on
validation). The PIGINet "no pose / size values" rows are one seed.

| model | seeds | T move, original scene (AUROC) | T move, edited scene (AUROC) | T move, lattice L (AUROC) | solvable vs NONE (AUROC) | class acc, tau 0.5 | class acc, tau* | blocker set top-1 (given K) | blocker-mask F1 |
|---|---|---|---|---|---|---|---|---|---|
| GRN | 3 | 0.958 ± 0.001 | 0.960 ± 0.000 | 0.881 ± 0.003 | 0.975 ± 0.002 | 0.574 ± 0.013 | 0.601 ± 0.011 | 0.788 ± 0.009 | 0.834 ± 0.003 |
| PIGINet, full (image + pose + text), obstacles parked in the state | 3 | 0.967 ± 0.006 | 0.980 ± 0.005 | 0.955 ± 0.007 | 0.998 ± 0.000 | 0.741 ± 0.030 | 0.753 ± 0.024 | 0.928 ± 0.017 | 0.829 ± 0.018 |
| PIGINet, full (image + pose + text), relocations as plan tokens | 3 | 0.967 ± 0.006 | 0.980 ± 0.005 | 0.955 ± 0.007 | 0.977 ± 0.019 | 0.728 ± 0.013 | 0.728 ± 0.011 | 0.902 ± 0.016 | 0.825 ± 0.011 |
| PIGINet, no image tokens, obstacles parked in the state | 3 | 0.987 ± 0.001 | 0.993 ± 0.001 | 0.984 ± 0.001 | 1.000 ± 0.000 | 0.844 ± 0.024 | 0.855 ± 0.010 | 0.963 ± 0.009 | 0.864 ± 0.011 |
| PIGINet, no image tokens, relocations as plan tokens | 3 | 0.987 ± 0.001 | 0.993 ± 0.001 | 0.984 ± 0.001 | 0.998 ± 0.001 | 0.773 ± 0.021 | 0.791 ± 0.003 | 0.899 ± 0.016 | 0.808 ± 0.009 |
| PIGINet, no pose / size values, obstacles parked in the state | 1 | 0.934 | 0.927 | 0.876 | 0.947 | 0.473 | 0.511 | 0.763 | 0.620 |
| PIGINet, no pose / size values, relocations as plan tokens | 1 | 0.934 | 0.927 | 0.876 | 0.972 | 0.481 | 0.595 | 0.884 | 0.689 |
| SceneCollisionNet | 3 | 0.968 ± 0.003 | 0.979 ± 0.001 | 0.964 ± 0.004 | 0.996 ± 0.002 | 0.756 ± 0.003 | 0.764 ± 0.017 | 0.924 ± 0.017 | 0.250 ± 0.013 |

### The paper's table (a) columns

AUC = scene solvable (K0-K3) vs NONE from the method's own score, computed with `metrics66.py` (REAP's script) for every row. Repair / Task /
Time / Calls need a planner and are therefore empty for the predictors that have none.

| method | AUC | Repair (%) | Task (%) | Time (s) | Calls |
|---|---|---|---|---|---|
| REAP, released weights, zero-shot (1 seed) | 0.984 | 55.0 | 73.0 | 168.1 | 46.0 |
| REAP, trained from scratch on 600 V4 train scenes (1 seed) | 0.789 | 82.6 | 63.2 | 208 | 39.7 |
| PIGINet + TAMP, full (3 seeds) | 0.978 ± 0.021 | 68.9 ± 4.5 | 98.1 ± 0.2 | 26.9 ± 4.2 | 3.4 ± 0.4 |
| PIGINet + TAMP, no image tokens (3 seeds) | 0.995 ± 0.004 | 69.1 ± 0.7 | 99.1 ± 0.1 | 24.4 ± 2.3 | 3.3 ± 0.1 |
| GRN (feasibility predictor only, 3 seeds) | 0.975 ± 0.002 | – | – | – | – |
| SceneCollisionNet (feasibility predictor only, 3 seeds) | 0.996 ± 0.002 | – | – | – | – |

- REAP is run with the dataset teacher as feasibility oracle and every plan is replayed in MuJoCo. The trained row uses a precomputed
  conflict library as the checker (7-11 s per teacher check made online training impossible; the library agrees with the teacher on about
  77% of random moves), three restarted segments (about 3,600 updates), so it is handicapped; the object-count-only baseline reaches 0.976
  AUC on this split, so AUC does not separate the methods well.
- PIGINet + TAMP: candidate plans are scored, those below 0.5 dropped and the rest refined in score order by the teacher (300 s limit per
  scene). PIGINet is given the scene file's parking cells as destinations, so its Task / Time / Calls are optimistic; Time was measured on a
  laptop with 10 workers, REAP's on servers.
- GRN and PIGINet (as predictors) are never asked to choose placements; parking cells are given to the candidates.
- SceneCollisionNet: NVlabs' architecture on a scene point cloud of the layout and the xArm7 hand cloud at each of the 32 grasp poses (over the target, and over the goal); a grasp works with probability (1 - p_pick)(1 - p_place), the state score is the best grasp. It has no obstruction output, so its blocker-mask F1 uses the score gain when one obstacle is parked alone (delta tuned on validation).
<!-- all-models-results:end -->

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
