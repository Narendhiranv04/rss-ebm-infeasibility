# Analytical baselines on the 2D dataset: publication release

E-M4M, MS-MCTS and MO-SeGMan on the 2D planar 3R dataset (rss-ebm-infeasibility, test split, 749 scenes, 3 seeds). They are scored
exactly like Approach B's Table I, so these rows can sit next to the learned model. This folder holds the code, the raw runs, the
tables and everything needed to re-run. It is a frozen copy of the working folder `D:\RRC\New folder\baselines2d`, taken on 2026-11-17.

**Results: see [RESULTS.md](RESULTS.md)** for the tables, metric definitions and notes for the paper. Headline at 300 CPU-s, Task %
(mean ± std over 3 seeds): E-M4M 61.7 ± 1.2, MO-SeGMan 30.7 ± 1.8, MS-MCTS 27.0 ± 0.3.

| Baseline | Paper | Code basis |
|---|---|---|
| E-M4M | Saxena & Likhachev, ICAPS 2023 ([arXiv 2303.13385](https://arxiv.org/abs/2303.13385)) | our MuJoCo port, `m4m_em4m_project/mamo/mamo/em4m` |
| MS-MCTS | Ren & Qureshi ([arXiv 2305.17175](https://arxiv.org/abs/2305.17175)) | written from the paper (no public code) |
| MO-SeGMan | Tuncer, Toussaint & Oguz ([arXiv 2511.01476](https://arxiv.org/abs/2511.01476)) | reference code [CankutBoraTuncer/MO-SeGMan](https://github.com/CankutBoraTuncer/MO-SeGMan) |

## Folder layout

```
README.md            this file: setup, protocol, deviations from the papers
RESULTS.md           final tables (300 s and 60 s), metric definitions, notes for the paper
results/             table.md / table.json, per_scene.csv, per_seed_summary.csv, flagged_scenes.csv
b2d/                 the code
  paths.py             data locations              dataset.py   selection, scene files, evaluation masks
  teacher.py           checker boot, X/Y verify, motion primitives with CPU accounting
  common.py            output contract, dataset rules, shared helpers
  em4m2d.py            E-M4M        msmcts.py   MS-MCTS        mosegman.py   MO-SeGMan
  run.py               runner (per-scene worker, verification, travel)      table.py   metrics
scripts/             export.py (writes results/), k0_uncapped_probe.py, parity.py, profiling / probe scripts from development
out/runs/main/       raw results.jsonl (6,741 runs) + table.md / table.json
out/runs/smoke/      5-scene smoke test
out/witnesses/       checker witness trajectories (travel)
out/cache/           per-scene exact motion caches
Dockerfile, dk.ps1   Linux container (the checker needs Unix CPU timers)
smoke_roots.txt      the 5 smoke scenes, one per class
```

## Data and checker (read-only, `D:\RRC\New folder\data2d\`)

| Folder | Source | Used for |
|---|---|---|
| `yadu-dataset-3d/gh-yadu-iiith/planar_2d/dataset_bd_classquota` | rss-ebm-infeasibility @ `yadu/dataset-3d` | original scenes, the dataset's motion code |
| `naren-dataset-done/final_dataset_generation` | rss-ebm-infeasibility @ `naren/dataset-done` | selection `k3_hard/2D_K3_STRENGTH_SELECTION_V3.json`, X/Y label manifest, newer scene stores |
| `lct-approach-b-xy-v1` | latent-causal-tamp @ `approach-b-xy-v1` | `bxy_teacher.py` (identical to `approach-b-universal-search`) |
| `lct-approach-b-final` | latent-causal-tamp @ `naren/approach-b-final` | Approach B's evaluation (`approach_b_final/evaluate.py`), copied here |

Test split: 749 scenes (K0 187, K1 120, K2 120, K3 172, NONE 150). Masks reproduce Approach B's exactly: 412 task, 405 repair, 187 no-op,
150 abstain roots (`b2d/dataset.py`).

## Re-running

Everything runs in a Linux container, because the checker's sector certificate uses a Unix CPU timer. `dk.ps1` mounts `D:\RRC\New folder`
at `/work` and runs from this folder. The data must stay at `D:\RRC\New folder\data2d` (see above). If you move this folder elsewhere
under `D:\RRC\New folder`, edit the `-w` path in `dk.ps1`. Outputs always go to `out/` next to the code.

```powershell
docker build -t baselines2d:py313 .                                    # Python 3.13, NumPy 2.4.0 (the dataset's requirements.lock)
.\dk.ps1 python scripts/parity.py 4                                    # re-run recorded X/Y labels (13/13 agreed)
.\dk.ps1 python -m b2d.run --tag main --seeds 0,1,2 --workers 6        # all three baselines on the test split (resumable)
.\dk.ps1 python -m b2d.table --tag main --cuts '300,60'                # metrics -> out/runs/main/table.md / .json
.\dk.ps1 python scripts/export.py --tag main                           # CSV exports -> results/
```

`b2d.run` skips runs already in `out/runs/<tag>/results.jsonl`, so the tables above can be rebuilt without re-running anything. A
full re-run from an empty `out/` takes about 30 h for 3 seeds on a 12-core laptop. With the saved caches, every motion query is
answered from cache, so it runs much faster, though the CPU charged per run stays the same.

## Protocol (same as Approach B)

- Each method outputs **NOOP**, **REPAIR** (ordered `[object, cell]` list) or **ABSTAIN** per scene.
- Every distinct REPAIR is checked by the finalized checker: X (relocations with carried state + connector to q_ref + replay) and
  Y (target from q_ref + replay). **Task %** = X = 1 and Y = 1 on K1–K3; **Repair %** = Task with exactly K moves; **No-op %** on K0;
  **Abstain %** on NONE; **Travel** = witnessed joint travel of successful repairs (`cost.jrec`).
- Dataset rules: only obstacles move, each at most once, to a legal free cell, at most 4 relocations.
- **Budget: 300 CPU-s per scene and run**, read also at 60 s. CPU is charged with the dataset's own virtual accounting (a cached motion
  result costs what its computation cost), so every run spends what it would spend alone. The 3 baselines × 3 seeds of a scene share
  one exact motion cache.
- Reported cost: motion-planner calls per run and CPU per run (Approach B's learned search makes 0 calls).

## Common adaptation: the motion layer

All three baselines plan every executable action with **the dataset's own planner** (`p16_motion.DetPhysics`, P16_FAST, per-query budget
`bd_2d.QB`), the one the X/Y checker uses, so a plan accepted by a baseline is the plan the checker re-verifies.

- **Per-call CPU timeout (45 s).** The dataset's planner has no fail-fast bound: on this laptop a failing target or relocation query
  runs ~300 CPU-s (two exhausted RRTs), the whole budget. The papers' planners all fail fast (MS-MCTS's linear planner, E-M4M's planning
  time limit, MO-SeGMan's RRT iteration caps). Successful queries take 3–33 s. A timed-out call is a planner failure, never a label.
  Cost of the cap: in 11 of 187 test K0 scenes the target plan needs longer (66–123 CPU-s measured uncapped on 3 of them), so all
  three baselines miss the no-op there (No-op 94.1%); in one of those they return a 1-move repair instead.
- **Geometric tests** use the papers' own constructions on this robot: the action planned with the other objects ignored and certified
  per object (`motion.object_check`), its swept footprint sampled along the path (equal to the certified footprint on the tested scene,
  0.1 s instead of 60 s), and the skill planner's static pick/place screens over its 48 grasps (grasp pose + certified straight
  approach/retreat line, no RRT).
- **Start check (all three):** the task is first tried in the scene as it is; success -> NOOP.

## Baseline-specific deviations

| Baseline | Deviation | Why |
|---|---|---|
| E-M4M | Simulated push -> one pick-and-place relocation (1 sample per goal; a failed goal weighs 5, as 5 failed push samples) | dataset is kinematic, pick-and-place only |
| E-M4M | CBS -> its optimal goal assignment (lattice distances around other objects + negative-cache penalty, `linear_sum_assignment`) | pick-and-place moves do not follow grid paths |
| E-M4M | Penalty box ±1 cell of 8 cm (reference ±3 cells of 1 cm); φ3 (mass × friction) omitted from the priority | lattice size; no mass in the 2D domain |
| E-M4M | Positive cache = the dataset's exact motion cache | identical queries are never re-planned |
| MS-MCTS | One stage (only T has a goal); discrete buffer regions (the paper's own variant) | single target; dataset rule |
| MS-MCTS | Linear picking/placing paths -> static pick/place screens; LMP sweep -> sampled footprint of the obstacle-free action | arm differs from the paper's gripper model |
| MS-MCTS | Constants not in the paper: 2 regions per expansion, 4 candidate cells per Algorithm 1 call, stuck depth 2, failure reward −4 m | unspecified |
| MO-SeGMan | Single goal object: sequence generation is trivial; no adaptive subgoals for T | T moves once in this dataset |
| MO-SeGMan | SGFS iteration limit 5 (reference: number of goal objects + 1 = 2) | 2 cannot reach K = 3 |
| MO-SeGMan | Critical-subset test is geometric (task trajectory clears the rest), not a re-run of RRT | each failing re-plan costs a 45 s timeout |
| MO-SeGMan | Images: 1 cm raster of the board; reachability flood fill starts at the TCP | 2D planar arm instead of a disc robot |
| all | Move-once, ≤ 4 relocations, legal grid cells | dataset rules (same as Approach B) |
