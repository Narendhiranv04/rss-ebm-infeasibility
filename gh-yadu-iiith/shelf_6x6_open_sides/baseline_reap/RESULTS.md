# REAP on the 6x6 open-sided shelf: results (V4 split, 370 test scenes)

Feasibility oracle = the dataset teacher (xArm7, 6x6 shelf, no-overpass, ceiling); test uses the full teacher for every
check, plus a replay of the found plan. Split: `split_v4.json` (Narendhiran's V4 split). Per-chunk CSVs in `results/`.

| Row | AUC | Repair (minimal) | Task | Time (s) | Calls |
|---|---|---|---|---|---|
| REAP zero-shot (released confined weights, no training on our data) | 0.984 | 55.0% | 73.0% | 168 | 46.0 |
| REAP trained from scratch on 600 V4 train scenes (this repo, `models/reap_shelf66_scratch_s1`) | 0.789 | 82.6% | 63.2% | 208 | 39.7 |

## Update 2026-11-19: batch 64, solvable scenes only, labelled training time

| Row (370 test scenes) | AUC | Repair (minimal) | Task | Time (s) | Calls |
|---|---|---|---|---|---|
| REAP zero-shot (released) | 0.984 | 55.0% | 73.0% | 168 | 46.0 |
| Scratch, 24 h, batch 8, with unsolvable scenes (first run, above) | 0.789 | 82.6% | 63.2% | 208 | 39.7 |
| 6 h batch 64, solvable scenes only (`results/scratch_v2_v4_s1`) | 0.679 | 78.1% | 66.2% | 196 | 47.0 |
| v4 line, **6 h label** (`models/reap_shelf66_scratch_v4_s1_6h`) | 0.707 | 70.1% | 77.0% | 166 | 63.4 |
| v4 line, **12 h label** (`models/reap_shelf66_scratch_v4_s1_12h`) | 0.793 | 66.4% | **81.9%** | 157 | 42.2 |

Solved % / Task % by class, 6 h label -> 12 h label (zero-shot solved % in brackets):
K0 (97.0): 100 / 100 -> 100 / 100. K1 (92.5): 94.3 / 92.5 -> 94.3 / 92.5. K2 (81.4): 79.1 / 75.6 -> 83.7 / 77.9.
K3 (42.2): 50.0 / 42.2 -> 59.8 / 56.9. NONE: 1.6 / 98.4 -> 0.0 / 100. Mean calls 63.4 -> 42.2; K3 time 239 s -> 216 s.
Full per-class tables: `results/scratch_v4_{6h,12h}_v4_s1/metrics.txt`.

What changed against the first scratch run (found by checking the setup against the authors' repo):
* `--search_batching 64` and `--worker_time_limit_s 60` (the paper's values; the first run used 8 / 300, left over from teacher-based runs).
* Training set = all 1,478 solvable V4 train scenes (the 288 unsolvable NONE scenes took 30% of worker time and are not in the authors' training
  sets); validation = the V4 split's own solvable validation scenes (`make_train_v4_all_solvable.py`).
* Supervised runner `server/drive_v4.sh`: 12 workers, auto-resume after memory-guard stops, labels by cumulative training time of the v4 line
  (which starts at 1.7 h; the model had already trained ~8 h before it in earlier restarts: the 6 h batch-64 run, a 1.4 h continuation, 0.6 h),
  so the "6 h label" is ~14 h of training in total and the "12 h label" ~20 h.
* The REAP validation score printed during training is a greedy rollout (no search); it stayed at 4/50 while the search-based test solved most easy
  scenes, so it was not used to choose checkpoints. Only the weights at each label are tested, once each.
* Same caveats as above: library checker with 2 cm tolerance (~77% agreement with the teacher), restarts with fresh optimizer/buffers,
  one run per label (no seeds). The 24 h label is still training.

Solved by class (zero-shot -> scratch): K0 97.0 -> 100, K1 92.5 -> 88.7, K2 81.4 -> 47.7, K3 42.2 -> 24.5, NONE 1.6 -> 0 (%).

## Caveats (must be stated with the trained row)
* Training used a precomputed conflict library (`lib66.py`, YADU_LIB_TOL=0.02, free move ignored) instead of the teacher
  (7-11 s per check made online training impossible: 208 updates in 6.5 h). The library agrees with the teacher on only
  ~77% of random moves (false "yes" 0.6-1.1%), so the trained rows are handicapped. See `validate_tol.py`.
* The scratch run is three restarted segments (5 h + 10.6 h + 8 h, ~3,600 updates), each restarted from the previous weights
  with fresh optimizer/buffers (the first killed by the memory guard, the second stalled when its workers vanished).
* Fine-tuning from the released weights was not run (on hold).
* AUC: the object-count-only baseline reaches 0.976 on this split, so AUC does not separate methods well.

## Why scratch is worse than zero-shot (from the test CSVs; causes not isolated)
Both models mostly end by the 300 s limit (timeouts: zero-shot 142/370, scratch 190/370). On the scenes each did not solve:

| | zero-shot | scratch |
|---|---|---|
| teacher-rejected moves proposed (infeasible steps) | 12.6 | 23.4 |
| nodes expanded before the time limit | 52 | 29 |
| filter calls | 79 | 59 |

Scratch proposes ~2x more infeasible moves, and each rejected check is slow, so it spends the time budget on dead ends:
its Q-function has not learned the preconditions. On solved scenes it needs fewer steps (4.5 vs 6.4), which explains the
higher "Repair (minimal)" (it only solves the easy scenes).
Likely reasons (not separated): ~3,600 updates only (REAP's own runs have microsecond checks and many more updates; the
validation score was still moving: 3 -> 19 -> 13 of 50), the ~77%-agreement library, three restarts with fresh
optimizer/buffers, search_batching 8 (paper: 64), 600 training scenes.
