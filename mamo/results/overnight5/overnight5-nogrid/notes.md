## Notes

**The run.** Overnight run 5 (2026-10-28 23:06 to 2026-10-29 12:28). Every planner runs as its paper describes it (`grid.planners = False`), with 300 s per run, 8 runs in parallel, and friction 0.8. Only the 9 paper methods run. The dataset `overnight5-nogrid` uses scheme v2, an easier ladder (README, "Dataset scheme v2 and the twin datasets"). It has 100 problems: 25 / 15 / 10 easy / medium / hard per task, in 4 categories, with Access-19 merged into the confined shelf.

Its scene files are the same as `overnight5-grid`'s, except that `meta["grid"] = false` switches the grid rule off. States are continuous, as in the papers, and a replay is judged only by the interaction constraints. Both datasets ran in one experiment (`workspace/experiments/20261028-230621_overnight5`), interleaved, so both had the same load. The grid copy has its own notes in `../overnight5-grid/notes.md`.

**Solved and replayed:**

| Method | retrieval | pick-and-place | easy | medium | hard | total | same problems, grid |
|---|---|---|---|---|---|---|---|
| DOGAR (2011) | 29 | 26 | 44 | 7 | 4 | **55** | 32 |
| I-M4M-PnP | 21 | 25 | 35 | 10 | 1 | **46** | 21 |
| Lazy I-M4M-PnP | 19 | 23 | 34 | 8 | 0 | **42** | 20 |
| I-M4M | 19 | 20 | 28 | 11 | 0 | **39** | 19 |
| E-M4M | 17 | 17 | 27 | 7 | 0 | **34** | 20 |
| M4M | 14 | 14 | 20 | 7 | 1 | **28** | 9 |
| KPIECE | 4 | 5 | 8 | 0 | 1 | **9** | 6 |
| RRT | 4 | 4 | 7 | 0 | 1 | **8** | 7 |
| SELSIM | 1 | 1 | 2 | 0 | 0 | **2** | 2 |

* **Without the grid, the M4M family ranks as in its papers.** The order is M4M < E-M4M < I-M4M < I-M4M-PnP (28 < 34 < 39 < 46). Lazy I-M4M-PnP is close to I-M4M-PnP. Only the grid reversed this order in overnight3 and overnight4.
* **Plans replay.** Of 58 / 48 / 45 / 43 / 37 / 30 plans found (DOGAR / PnP / Lazy / I-M4M / E-M4M / M4M), 2-4 per method fail the replay, for a grasp miss, a tilt or immovable contact. Planning and replay agree.
* **The easier ladder works at easy and medium.** Easy problems are solved by 7 of 9 methods, and 44 of 50 by DOGAR. Hard is still nearly unsolved, at most 4 of 20. Together the methods solve 73 problems: 49 easy, 19 medium and 5 hard.
* **DOGAR is ahead, as in overnight3 and overnight4.** It is the planner of Dogar & Srinivasa's own paper, with push-grasps, sweeps and PickUps. Its 55 solutions use 79 pick-and-places and 34 pushes. The M4M papers instead compare against a re-implementation that only has M4M's push planner and random goals outside the NGR (`dogar_m4m`, [ext], not run here). That is why DOGAR is weak in those papers (M4M: 40 / 20 / 6 %, E-M4M: 7 / 0 / 0 solved).
* **The M4M family is limited by its push planner on the xArm7.** In the runs that failed, per run:

| Method | push planner | CBS | simulation | push trajectories planned | pushes simulated | of which invalid |
|---|---|---|---|---|---|---|
| M4M | 206 s | 0 s | 18 s | 486 | 26.9 | 26.0 |
| E-M4M | 114 s | 68 s | 15 s | 246 | 17.5 | 15.8 |
| I-M4M | 112 s | 61 s | 13 s | 239 | 15.9 | 14.5 |
| I-M4M-PnP | 175 s | 52 s | 28 s | 201 | 8.7 (+23.4 pick-and-places) | 7.8 (+10.7) |

  The paper's push (3 cm above the surface, points drawn with sigma 2.5 cm, IK segment by segment) was tuned for the PR2. Only 1 in 14-18 planned push trajectories reaches simulation (the rest fail IK or the collision-free approach), and about 9 in 10 of those violate a constraint in simulation. The open tabletop shows this most clearly: the push-only methods solve 1 of 26 problems there, while the methods with pick-and-place (DOGAR, I-M4M-PnP) solve 8-9.
* **E-M4M and I-M4M often end with "OPEN exhausted" before 300 s.** Every failed push invalidates its CBS goal, and the search runs out of vertices.
* **RRT, KPIECE and SELSIM solve easy problems only**, as in the papers' medium and hard levels. SELSIM ends 8 runs with "tracking failed, no irrelevant object to blame" (the paper's RELEVANTOBJECT).

**Feasibility (`dataset/feasibility.csv`).** 73 feasible (each with a replayed plan from this run), 2 infeasible and 25 unknown. The reach and space checks are shared with the grid copy.

**Repro.** `python scripts/overnight5_pipeline.py --workers 8 --budget 300` (it resumes from `../pipeline_state.json`). Earlier overnight results are unchanged.
