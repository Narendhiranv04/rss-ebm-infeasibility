# hybrid/: learned planners for shelf obstacle relocation (exploratory study)

**What this is:** a test of changes to the team's Approach B planner on the 6×6 shelf. The planner decides which objects to move out of the way so a target can be picked.
**What you'll do:** read RESULTS.md first, then use this map to find code, and the Setup section to re-run things.

- Every plan is checked with the dataset's own physics teacher (`BD3DS66_TEACHER_V2`, MuJoCo, chained execution).
- The team's code in `../approachb_3d_learning/src/` is **used unchanged**. Everything here only imports it.
- Approach B pipeline, in order: goal-blind cVAE → Ranker → Boltzmann X/Y model → plan search.

**Read first:** [RESULTS.md](RESULTS.md), the detailed write-up of the design changes, how they change the workflow, and every result.

## Words used here (each explained once)

| Term | Plain meaning |
|---|---|
| Physics teacher | The MuJoCo simulation that decides if a plan really works. |
| cVAE | A model that proposes candidate moves. |
| Ranker | A model that sorts the proposed moves. |
| Boltzmann X/Y model | A model that predicts two yes/no outcomes per plan: X = the moves work, Y = the target gets out. |
| Plan search | The step that chains moves into a full plan. |
| Goal-blind / goal-aware | Whether the model knows which object is the target. |
| Root | One starting shelf scene. |
| Cached | Saved results; re-running only does the missing work. |
| VAL / TEST | Two data splits. We only used VAL. |

## Layout

```
hybrid/
├── README.md               this file
├── RESULTS.md              design changes, workflow, results, caveats
├── docs/
│   ├── EXPERIMENT_LOG.md   every experiment: hypothesis → design → result (pre-registered entries marked)
│   ├── LIT_BRIEF_2026-11-20.md        literature brief (ARS quick mode, ~40 verified papers)
│   └── THREE_WAY_SCAN_2026-11-20.md   WHY/HOW/WHAT scan of the team's reading list (M4M family, recourse, EBM planner)
├── src/                    all code (one flat folder: the scripts import each other by name)
└── archive/
    └── phase3c_piginet.py  PIGINet plug-in, dropped from the comparison (model only partly trained)
```

### `src/` by role

| Role | File | What it does |
|---|---|---|
| Infrastructure | `cpu_run.py` | Runs the team's GPU-server training scripts unchanged on a laptop CPU, stubbing only the server guards. |
| | `physics.py` | `execute(raw, plan)`: the full physics teacher, call-for-call like `bd_3dt.teach.seq()`. |
| Models | `hazard.py` | **Per-step success model**: P(move k ok \| scene so far) × P(target ok \| final scene). |
| | `factor.py` | **Factor graph**: the hazard move factors × a noisy-AND target factor over per-object "blocks the target" variables. |
| | `gcvae.py` | **Goal-aware cVAE** (hybrid v2): adds a goal marker and is distilled from the Boltzmann model. |
| | `gcvae_abl.py` | E7: 7 teacher variants of the goal-aware cVAE. |
| Experiments | `v1_check.py` | E1: offline label check, Boltzmann vs cVAE vs their product. |
| | `phase2.py` | E2: placement in physics, 30 contexts. |
| | `phase3.py` | E3/E4: placement (90 contexts) and whole-plan search (160 roots), Boltzmann seeds, DIRECT-GEO, baselines. |
| | `phase3b.py` | E5: the team pipeline (cVAE → Ranker → Boltzmann), reproduced with the team's own sampling code. |
| | `phase4.py` | Plan-level planners with lazy verified retries (Boltzmann, cautious, per-step, factor, goal-aware pipeline, team pipeline). |
| | `e7_physics.py` | E7 physics: #1 cell of each goal-aware cVAE variant, Holm-corrected tests. |
| | `threshold.py` | Abstain threshold η chosen by cross-validation, and calibrated decision rule. |
| | `scaling.py` | Planning-time scaling vs number of objects and search depth. |

## Setup

Run everything from `../approachb_3d_learning/`.

- **venv:** `RSS_2027/.venv` (Python 3.12; torch CPU, mujoco 3.16.0, numpy, scikit-learn).
- **Data (gitignored, copied from the server):**
  - `snapshots/v4full_82926152fa9cc345_derived/` (features + scalers);
  - `snapshots/v4full_.../upstream/gh-yadu-iiith`, a symlink to this repo (root hashes verified);
  - `runs/XY3D_REAL/BD3DS66_XY_SIDECAR_V1.json` (gunzipped sidecar).
- **Trained models (gitignored, under `runs/`):**
  - `SAMPLER_V1/{cvae,ranker}_s{0,1,2}`;
  - `XY3D_REAL/smoke_cpu{,_s1,_s2}` (Boltzmann D / E / G / DIRECT-GEO, 3 seeds);
  - `HAZARD/`, `FACTOR/`, `GCVAE/`, `GCVAE_ABL/`.

## How to run things

Copy-paste these in order. Each block starts from `../approachb_3d_learning/`.

```bash
P=../../../.venv/bin/python
# retrain the team's models on CPU (unchanged code)
$P ../hybrid/src/cpu_run.py src/train.py fit --config configs/sampler_v1.json --kind cvae --seed 0
XY_SEED=1 $P ../hybrid/src/cpu_run.py src/xy3d_train.py smoke --sidecar runs/XY3D_REAL/BD3DS66_XY_SIDECAR_V1.json --out runs/XY3D_REAL/smoke_cpu_s1
# new models
$P ../hybrid/src/hazard.py train && $P ../hybrid/src/factor.py train && $P ../hybrid/src/gcvae.py run
# experiments (resumable; physics results are cached by (root, plan) and shared across phases)
$P ../hybrid/src/phase3.py run --workers 6
NO_RETRY=1 $P ../hybrid/src/phase3b.py run --workers 6
$P ../hybrid/src/phase4.py run --workers 6 --planners boltz_mean,boltz_min,hazard,gcvae_pipeline,factor
$P ../hybrid/src/phase4.py summary          # also phase3.py / e7_physics.py summary
$P ../hybrid/src/threshold.py; $P ../hybrid/src/scaling.py
# self-tests
for t in phase4 hazard factor gcvae gcvae_abl; do $P ../hybrid/src/$t.py selftest; done
```

What each block does:
1. Sets `P`, the Python to use.
2. Retrains the team's models on the CPU.
3. Trains the new models.
4. Runs the experiments. You can stop and re-run them.
5. Prints summaries and runs the threshold and scaling scripts.
6. Runs the quick self-tests.

### Practical notes (macOS laptop)
- **Physics runs in a process pool.** Run it outside sandboxed shells, which block the semaphores.
- **macOS can throttle background workers.** Run `taskpolicy -B -p <pid>` on the worker processes; it gave about 40% more CPU.
- **Results are cached by (root, plan) and shared across phases.** Re-running only does missing physics.

## Rules followed
- **Splits:** VAL only. The TEST split was never read. All results are **exploratory**; see the caveats in RESULTS.md.
- **The team's code was not modified.** The extra Boltzmann seeds come from the same trainer, with only the seed changed (`XY_SEED`).
- **Nothing here is committed yet.** The suggested branch is `nav/hybrid-xy`.
