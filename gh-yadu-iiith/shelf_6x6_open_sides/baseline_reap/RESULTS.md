# REAP on the 6x6 open-sided shelf: results (V4 split, 370 test scenes)

Feasibility oracle = the dataset teacher (xArm7, 6x6 shelf, no-overpass, ceiling); test uses the full teacher for every
check, plus a replay of the found plan. Split: `split_v4.json` (Narendhiran's V4 split). Per-chunk CSVs in `results/`.

| Row | AUC | Repair (minimal) | Task | Time (s) | Calls |
|---|---|---|---|---|---|
| REAP zero-shot (released confined weights, no training on our data) | 0.984 | 55.0% | 73.0% | 168 | 46.0 |
| REAP trained from scratch on 600 V4 train scenes (this repo, `models/reap_shelf66_scratch_s1`) | 0.789 | 82.6% | 63.2% | 208 | 39.7 |

Solved by class (zero-shot -> scratch): K0 97.0 -> 100, K1 92.5 -> 88.7, K2 81.4 -> 47.7, K3 42.2 -> 24.5, NONE 1.6 -> 0 (%).

## Caveats (must be stated with the trained row)
* Training used a precomputed conflict library (`lib66.py`, YADU_LIB_TOL=0.02, free move ignored) instead of the teacher
  (7-11 s per check made online training impossible: 208 updates in 6.5 h). The library agrees with the teacher on only
  ~77% of random moves (false "yes" 0.6-1.1%), so the trained rows are handicapped. See `validate_tol.py`.
* The scratch run is three restarted segments (5 h + 10.6 h + 8 h, ~3,600 updates), each restarted from the previous weights
  with fresh optimizer/buffers (the first killed by the memory guard, the second stalled when its workers vanished).
* Fine-tuning from the released weights was not run (on hold).
* AUC: the object-count-only baseline reaches 0.976 on this split, so AUC does not separate methods well.
