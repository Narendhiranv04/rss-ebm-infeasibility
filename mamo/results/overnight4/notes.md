## Notes

**What changed since overnight3.** Same 100 problems (the `overnight4` dataset is a copy of `overnight3`), same budget (300 s for every method), load (8 runs in parallel) and friction (0.8). Every planner now runs as its paper describes it (decided 2026-10-28: "run everything as per their papers only"). The planners plan with continuous states and know nothing of the grid (`grid.planners = False`):

* CBS push goals are every free 1 cm lattice point outside the NGR, not just points near cell centres
* pick-and-places use any of the 8 placement yaws
* no grid check while planning, including RRT / KPIECE / SELSIM's goal test
* SELSIM's RELEVANTOBJECT is the paper's BFS from the violated object only; the robot-rooted fallback is gone

The grid is the problem's rule and is judged when the plan is replayed: every push and pick-and-place, and the final state, must leave each object in a cell of its own. DOGAR (2011) already ran this way in overnight3. Only the paper methods run; the [ext] DOGAR variants and the oracle do not.

**Solved and replayed, overnight3 (grid-aware planners) vs overnight4 (paper planners, grid judged at replay):**

| Method | overnight3 retrieval / pick-and-place / total | overnight4 retrieval / pick-and-place / total |
|---|---|---|
| DOGAR (2011) | 6 / 16 / 22 | 6 / 15 / **21** |
| Lazy I-M4M-PnP | 16 / 20 / 36 | 7 / 12 / **19** |
| E-M4M | 6 / 8 / 14 | 8 / 10 / **18** |
| I-M4M | 6 / 11 / 17 | 7 / 11 / **18** |
| I-M4M-PnP | 17 / 18 / 35 | 5 / 13 / **18** |
| M4M | 9 / 9 / 18 | 2 / 8 / **10** |
| RRT | 1 / 4 / 5 | 1 / 5 / **6** |
| KPIECE | 1 / 4 / 5 | 1 / 5 / **6** |
| SELSIM | 1 / 1 / 2 | 1 / 1 / **2** |

**OoI retrieval: plans found vs plans that replay on the grid (overnight3 -> overnight4):**

| Method | planned | replayed | replay failures, overnight4 |
|---|---|---|---|
| M4M | 10 -> 8 | 9 -> 2 | 3 box turned, 2 two objects in one cell, 1 other |
| E-M4M | 6 -> 16 | 6 -> 8 | 7 box turned, 1 out of its cell |
| I-M4M | 6 -> 17 | 6 -> 7 | 7 box turned, 2 out of its cell, 1 other |
| I-M4M-PnP | 19 -> 20 | 17 -> 5 | 9 box turned, 5 out of its cell, 1 other |
| Lazy I-M4M-PnP | 17 -> 20 | 16 -> 7 | 8 box turned, 5 out of its cell |
| DOGAR | 28 -> 28 | 6 -> 6 | 16 box turned, 2 out of its cell, 4 other |

* **Without the grid, E-M4M and I-M4M find almost three times as many retrieval plans (6 -> 16-17).** Their CBS goals are no longer limited to 9 lattice points per cell, so far fewer runs end with "OPEN exhausted" (38 -> 27 of 50 retrieval runs). This is the effect explained after overnight3: each failed push invalidates its goal, and with few goals per cell the search ran out.
* **Half of those plans fail the grid at replay.** The pushes leave a box turned 15-45 deg or between cells: nothing in the papers' planners keeps objects square. On balance E-M4M gains 4 problems and I-M4M 1.
* **I-M4M-PnP and Lazy I-M4M-PnP lose about half their solves (35-36 -> 18-19).** Their pick-and-places now aim at any free point with any of 8 yaws. 14 and 13 of their retrieval plans put a box down turned or between cells. In overnight3 they aimed at cell centres and turned boxes only by 90 deg.
* **M4M drops from 18 to 10.** 6 of its 8 retrieval plans fail at replay (a box turned, or two objects pushed into one cell).
* **DOGAR is unchanged** (22 -> 21; 21 problems in common). Its planner was already the paper's; one pick-and-place problem was not solved again.
* **Pick-and-place problems hold up better for every push method.** Each stage is replayed on the grid before the next. A stage that fails is planned again from the same state while the budget lasts (stage re-plans: E-M4M 34, I-M4M 35, I-M4M-PnP 70, Lazy 55, DOGAR 249). So a method keeps trying until a stage happens to land on the grid.
* **RRT, KPIECE and SELSIM barely change.** Their states were continuous in overnight3 too; only their goal test no longer checks the grid. RRT and KPIECE each gain one pick-and-place problem.
* **SELSIM without the robot-rooted BFS.** 7 of its runs ended with "tracking failed, no irrelevant object to blame". In those runs the robot touched the shelf, or the grasp failed, and the paper's RELEVANTOBJECT has no object to start from. It still solves the same 2 problems.
* **No method solves a hard problem**, as in overnight2 and overnight3.
* **Problem coverage.** Together the methods solve 36 problems, against 49 in overnight3; no problem is solved only in overnight4.

**Feasibility (`dataset/feasibility.csv`).** The labels are 58 feasible, 4 infeasible and 38 unknown, unchanged from overnight3. The witnesses are this run's replayed plans plus every earlier plan that replays under the grid: overnight3's runs, its oracle, and the overnight2 plans and oracle (`revalidation_grid.json`). overnight4 adds no new witness.

**Repro.** `python scripts/overnight4_pipeline.py --workers 8 --budget 300` (resumes from `pipeline_state.json`). The experiment is `workspace/experiments/20261028-145519_overnight4`. The overnight, overnight2 and overnight3 results are unchanged. Reproducing overnight3 needs `grid.planners = True`.
