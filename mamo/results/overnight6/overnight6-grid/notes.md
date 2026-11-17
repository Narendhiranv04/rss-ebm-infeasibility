## Notes

**The run.** This is overnight run 6, with the same setup as `../overnight6-nogrid/notes.md`: 7 methods (the M4M family, DOGAR and `dogar_m4m`), planners as in their papers (`grid.planners = False`), one 900 s run per problem read at 300 / 600 / 900 s, friction 0.8, and the new push defaults (5 cm back-off, 2 cm push height). There are three environments: confined shelf (28), multi-level shelf (24) and tabletop (48). The dataset `overnight6-grid` holds the same 100 problems as the no-grid copy, with the grid rule on and judged at replay. Every push and pick-and-place, and the final state, must leave each object squarely in a cell of its own.

**Solved and replayed: 300 / 600 / 900 s, overnight5 (300 s, same problems, paper push) and the no-grid copy (900 s):**

| Method | 300 s | 600 s | 900 s | overnight5 | no grid |
|---|---|---|---|---|---|
| DOGAR (2011) | 32 | 32 | **32** | 32 | 59 |
| Lazy I-M4M-PnP | 23 | 29 | **30** | 20 | 65 |
| I-M4M-PnP | 23 | 28 | **29** | 21 | 59 |
| I-M4M | 22 | 24 | **26** | 19 | 53 |
| E-M4M | 22 | 22 | **23** | 20 | 57 |
| DOGAR, M4M papers' version [ext] | 21 | 22 | **23** | - | 42 |
| M4M | 8 | 8 | **9** | 9 | 24 |

**Per environment at 900 s (confined / multi-level / tabletop):** DOGAR 7 / 8 / 17, Lazy I-M4M-PnP 9 / 5 / 16, I-M4M-PnP 8 / 6 / 15, I-M4M 10 / 7 / 9, E-M4M 8 / 6 / 9, `dogar_m4m` 9 / 6 / 8, M4M 4 / 3 / 2.

* **The grid, not the planners, sets the result here.** Many plans that the planners find fail at replay, and almost all of those failures are the grid rule:

| Method | plans found | failed replay | off the grid | of which a box turned | pushes / pick-and-places in the plans found |
|---|---|---|---|---|---|
| DOGAR | 54 | 22 | 19 | 17 | 32 / 73 |
| Lazy I-M4M-PnP | 58 | 28 | 24 | 17 | 50 / 140 |
| I-M4M-PnP | 59 | 30 | 24 | 19 | 68 / 156 |
| I-M4M | 47 | 21 | 21 | 19 | 99 / 20 |
| E-M4M | 47 | 24 | 24 | 20 | 90 / 18 |
| `dogar_m4m` | 41 | 18 | 16 | 14 | 56 / 19 |
| M4M | 22 | 13 | 11 | 11 | 40 / 6 |

  Most of them are a pushed box that ends **turned** in its cell (the pick-and-places include the final grasp of the OoI or target). The planners do not know the grid (as in their papers), and a push rarely moves a box without rotating it.
* **The push change helps only a little with the grid.** At 300 s the M4M family gains 2-3 solves over overnight5 (M4M loses 1), against 10-20 without the grid. The better pushes reach simulation and replay, then fail the grid rule. DOGAR solves exactly overnight5's 32 problems again.
* **DOGAR leads, ahead of Lazy I-M4M-PnP (30) and I-M4M-PnP (29).** Its plans have the fewest pushes (32 in 54 plans), and 41 % of them fail replay, against 44-59 % for the others. The other methods are within a few problems of each other, and `dogar_m4m` ties E-M4M at 23.
* **More time helps the pick-and-place methods.** From 300 s to 900 s, Lazy I-M4M-PnP gains 7 (6 of them pick-and-place problems) and I-M4M-PnP gains 6 (all pick-and-place). DOGAR finds nothing more after 300 s.
* **Retrieval with the grid is nearly unsolved.** At most 10 of 50 (DOGAR), and the M4M family solves 3-9.

**If clearing the way were enough (`grid_relaxed_replay.jsonl`, 2026-10-30).** The 139 plans that failed replay only on the grid (all of them OoI retrievals) were replayed again with the grid rule off. 118 succeed: the OoI comes out and no interaction constraint is violated. Counted as solved, the totals would be DOGAR 51, Lazy I-M4M-PnP 49, I-M4M-PnP 48, I-M4M 44, E-M4M 44, `dogar_m4m` 37, M4M 17. No pick-and-place plan fails on the grid at the final replay: `mamo.relocate` replays every stage with the grid rule and plans the stage again if it fails, so pick-and-place plans are already on the grid when they are returned.

**Feasibility (`dataset/feasibility.csv`).** 51 feasible, 1 infeasible and 48 unknown (overnight5: 49 / 2 / 49). The reach and space checks are shared with the no-grid copy. `open_tabletop_relocate_easy_0000`, labelled infeasible in overnight5, is solved by I-M4M-PnP here (see the no-grid notes).

**Repro.** `python scripts/overnight6_pipeline.py --workers 8 --budget 900` (it resumes from `../pipeline_state.json`). Earlier overnight results are unchanged.
