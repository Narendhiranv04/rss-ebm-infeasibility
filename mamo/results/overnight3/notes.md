## Notes

**What changed since overnight2.** These are the same 100 problems (the `overnight3` dataset is a copy of `overnight2`), with the same methods, budget (300 s), load (8 runs in parallel) and friction (0.8). Only the grid rule is new (README, "Grid states"): every action must leave every object in a cell of its own, upright and square to the grid. **DOGAR** is the planner of its own paper (Dogar & Srinivasa 2011: Sweep / Push-grasp / PickUp / GoTo, 36 directions, exhaustive search; README "DOGAR", decided 2026-10-28). It plans without the grid and is judged on the grid when its plan is replayed. It was run on 2026-10-28 in the same experiment under the same load (8 runs at a time). The version the M4M papers re-implement, which ran in the night as "DOGAR", is kept as `dogar_m4m` [ext]: its rows and plans were renamed, with the old files in the experiment's `backup_before_dogar2011/`. Its grid-aware form is `dogar_grid` [ext].

**Solved and replayed, overnight2 (continuous states) vs overnight3 (grid states):**

| Method | overnight2 | overnight3 |
|---|---|---|
| Lazy I-M4M-PnP | 40 | 36 |
| I-M4M-PnP | 38 | 35 |
| DOGAR (Dogar & Srinivasa 2011) | - | 22 |
| DOGAR (M4M papers' version, grid-aware) [ext] | - | 21 |
| M4M | 21 | 18 |
| I-M4M | 33 | 17 |
| E-M4M | 32 | 14 |
| DOGAR (M4M papers' version) [ext] | 26 (as "DOGAR") | 12 |
| RRT | 8 | 5 |
| KPIECE | 11 | 5 |
| SELSIM | 2 | 2 |

* **Pick-and-place survives the grid best.** A placement aims at a cell centre and turns a box only by multiples of 90 deg, so I-M4M-PnP and Lazy I-M4M-PnP lose only 3-4 problems each. Pushes cannot stop an object exactly.
* **E-M4M and I-M4M lose half their solves.** 71 of their 100 runs end with "OPEN exhausted". Under the grid each object has far fewer push goals (lattice points near a cell centre, not every free 1 cm point). Pushes that leave an object between cells or turned are invalid, and every failed push invalidates its goal (E-M4M's negative cache). So the search runs out of successors. This matches the 120 s sanity study on the easy problems (E-M4M 24 -> 13).
* **M4M loses little (21 -> 18).** It has no OPEN list to exhaust. Its failed runs end on the budget (48), with no MAPF-suggested push left (18), or with CBS failing (14).
* **DOGAR (2011) plans 44 problems, 22 replay** (6 OoI retrievals, 16 pick-and-place problems; 21 easy, 1 medium).
  * Its replayed plans are almost all PickUps: only 2 contain a Sweep. A pick-up puts the object down on the surface's 1 cm lattice, and a carried box can be set down turned, so it is often off the grid.
  * Of its 28 planned retrievals, 22 fail replay. 18 of those leave an object off the grid (turned 19-45 deg, or between cells): 12 after pick-ups, 6 after sweeps. The other 4 differ in physics: while planning, the objects not yet dealt with are removed and only the grasp and placement are simulated; the replay has every object present (an unstable simulation, bodies forced together, an immovable touched, the hand on the shelf).
  * Pick-and-place problems fare better because each stage is replayed on the grid before the next. A stage that fails is planned again, as for every method, so DOGAR keeps trying until a stage happens to land on the grid.
  * The exhaustive search almost never completes: only 1 run ended with "action library exhausted"; the rest of its failed runs hit the 300 s limit. Over all runs it tried 33,839 actions (20,808 sweeps, 13,031 pick-ups), and only 1,090 were valid.
    * About 22,000 failed on the arm's kinematics: no hand pose at the start, or no straight reach or push.
    * 8,536 sweeps had no push distance (up to 60 cm) that leaves the object outside the NGR, because walls or fixed objects are in the way.
    * 771 failed in simulation.
* **DOGAR (M4M papers' version) [ext] plans 21 problems, but only 12 plans replay.** All 9 failures are OoI retrievals whose pushes leave an object turned 16-43 deg, or outside its cell. Its goals ignore the cells, so it has no reason to stop objects on them. Its pick-and-place relocations dropped from 14/50 to 9/50 for the stage-replay reason above. 77 runs end with "budget exhausted".
* **DOGAR (M4M papers' version, grid-aware) [ext] solves 21**: 23 planned, 2 replay failures (one box turned on replay, one finger touching an immovable). It is the best push-only method on the grid. The difference from its grid-agnostic form (12) is entirely the goal sampling and the off-cell check.
* The "Why runs failed" table shows each method's 6 most common reasons, so some DOGAR replay failures are hidden there; `results.csv` (`exec_reason`) has them all.
* **RRT, KPIECE, SELSIM** move objects only through incidental contact. Under the grid, their final state must also have every object on a cell, which cuts their already few solves (RRT 8 -> 5, KPIECE 11 -> 5).
* **No method solves a hard problem**, as in overnight2.

**Feasibility (`dataset/feasibility.csv`).** The labels are 58 feasible, 4 infeasible and 38 unknown (overnight2: 61 / 2 / 37). In the evidence column, "dogar (overnight2)" is the M4M papers' version that ran there. A witness counts only if it replays under the grid. The witnesses come from the overnight3 runs, from the overnight2 and oracle plans replayed under the grid (`revalidation_grid.json`), and from a new oracle run (900 s) on the 42 problems without one.

* The oracle found 3 new witnesses (2 easy, 1 medium). DOGAR (2011) found 1 more, `open_tabletop_retrieve_easy_0003`, which no other method or the oracle solved on the grid.
* 6 problems that were feasible in overnight2 are now unknown. Their only plans leave objects between cells, and no grid-valid plan was found.
* 2 unknown problems became feasible (new oracle witnesses), and 2 became infeasible. In those 2, the target's pick-and-place never succeeded in simulation even with every movable removed, over 20 seeds.
* The new free-cell check (enough free cells outside the NGR to hold the blockers) passes on all 100 problems.

**Repro.** `python scripts/overnight3_pipeline.py --workers 8 --budget 300 --oracle-budget 900` reproduces everything; it resumes from `pipeline_state.json`. The experiments are `workspace/experiments/20261027-232422_overnight3` (main and `dogar_grid`) and `workspace/experiments/20261028-064502_overnight3-oracle`. The overnight and overnight2 results are unchanged.
