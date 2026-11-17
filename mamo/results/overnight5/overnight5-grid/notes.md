## Notes

**The run.** This is overnight run 5, with the same setup as `../overnight5-nogrid/notes.md`: the 9 paper methods, planners as in their papers (`grid.planners = False`), 300 s per run, 8 runs in parallel, friction 0.8, and one experiment interleaving both datasets (`workspace/experiments/20261028-230621_overnight5`). The dataset `overnight5-grid` holds the same 100 scheme-v2 problems as the no-grid copy. Here the grid rule is on and is judged at replay. Every push and pick-and-place, and the final state, must leave each object squarely in a cell of its own.

**Solved and replayed: grid vs the same problems without the grid, and overnight4 (old, harder dataset, grid):**

| Method | retrieval | pick-and-place | easy | medium | hard | total | no grid | overnight4 |
|---|---|---|---|---|---|---|---|---|
| DOGAR (2011) | 10 | 22 | 28 | 2 | 2 | **32** | 55 | 21 |
| I-M4M-PnP | 8 | 13 | 20 | 1 | 0 | **21** | 46 | 18 |
| E-M4M | 8 | 12 | 18 | 2 | 0 | **20** | 34 | 18 |
| Lazy I-M4M-PnP | 5 | 15 | 16 | 4 | 0 | **20** | 42 | 19 |
| I-M4M | 8 | 11 | 18 | 1 | 0 | **19** | 39 | 18 |
| M4M | 5 | 4 | 8 | 1 | 0 | **9** | 28 | 10 |
| RRT | 4 | 3 | 7 | 0 | 0 | **7** | 8 | 6 |
| KPIECE | 3 | 3 | 6 | 0 | 0 | **6** | 9 | 6 |
| SELSIM | 1 | 1 | 2 | 0 | 0 | **2** | 2 | 2 |

**Plans found vs plans that replay on the grid** (no grid in brackets):

| Method | planned | replayed |
|---|---|---|
| DOGAR | 54 (58) | 32 (55) |
| I-M4M-PnP | 36 (48) | 21 (46) |
| Lazy I-M4M-PnP | 37 (45) | 20 (42) |
| I-M4M | 33 (43) | 19 (39) |
| E-M4M | 32 (37) | 20 (34) |
| M4M | 20 (30) | 9 (28) |

* **The grid costs every push-based method 40-70 % of its solves.** The planners find almost as many plans as without the grid. They find fewer only in pick-and-place problems, where a stage that fails on the grid is re-planned and uses up budget. About 40 % of the plans then fail replay because a push leaves a box turned 20-45 deg or between cells. None of the papers' planners keeps objects square.
* **The easier dataset barely helps under the grid.** Totals are close to overnight4 (DOGAR 21 -> 32, the M4M family about the same). Without the grid, the same problems give 28-55.
* **Ranking:** DOGAR first, then the I-M4M variants and E-M4M close together (19-21), then M4M (9). The grid flattens the papers' order of the M4M family, and the no-grid copy restores it.
* **Medium and hard:** at most 4 of 30 medium and 2 of 20 hard problems are solved.
* **Coverage:** together the methods solve 49 problems, against 73 without the grid. Every problem solved here is also solved without the grid.

**Feasibility (`dataset/feasibility.csv`).** 49 feasible, 2 infeasible and 49 unknown. The witnesses are this run's replayed plans; no oracle was run.

**Repro.** `python scripts/overnight5_pipeline.py --workers 8 --budget 300`. Earlier overnight results are unchanged.
