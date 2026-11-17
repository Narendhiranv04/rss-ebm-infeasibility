## Notes

**The run.** Overnight run 6 (2026-10-29 23:08 to 2026-10-30 17:32, planning). It runs 7 methods: the M4M family, DOGAR (Dogar & Srinivasa's own planner) and `dogar_m4m` (DOGAR as the M4M papers re-implement it, [ext]). The planners run as in their papers (`grid.planners = False`), with friction 0.8, 8 runs in parallel and **one 900 s run per problem**. Planners stop at their first solution, so a run's planning time also gives what a 300 s or 600 s run would have solved.

The 100 problems are overnight5's, regrouped into three environments: confined shelf (confined shelf + Access-19, 28 problems), multi-level shelf (24) and tabletop (open tabletop + 3D stacking, 48). This copy has continuous states, as in the papers. Both copies ran interleaved in one experiment (`workspace/experiments/20261029-230840_overnight6`). The grid copy has its own notes in `../overnight6-grid/notes.md`.

**What changed since overnight5 (decided 2026-10-29, `results/push_diagnosis/notes.md`).** The M4M family and `dogar_m4m` now start pushes 5 cm back from the object, as the authors' code does (`push.backoff = 0.05`), and push 2 cm above the surface (`push.height_above_shelf = 0.02`; the paper uses 3 cm). DOGAR's sweeps stay at 3 cm. Nothing else changed.

**Solved and replayed: 300 / 600 / 900 s, and overnight5 (300 s, same problems, paper push):**

| Method | 300 s | 600 s | 900 s | overnight5 |
|---|---|---|---|---|
| Lazy I-M4M-PnP | 55 | 62 | **65** | 42 |
| I-M4M-PnP | 56 | 59 | **59** | 46 |
| DOGAR (2011) | 55 | 57 | **59** | 55 |
| E-M4M | 54 | 57 | **57** | 34 |
| I-M4M | 51 | 52 | **53** | 39 |
| DOGAR, M4M papers' version [ext] | 39 | 40 | **42** | - |
| M4M | 23 | 23 | **24** | 28 |

**Per environment at 900 s, and overnight5 at 300 s (confined / multi-level / tabletop):**

| Method | confined shelf (28) | multi-level shelf (24) | tabletop (48) | overnight5 |
|---|---|---|---|---|
| Lazy I-M4M-PnP | 22 | 19 | 24 | 10 / 14 / 18 |
| I-M4M-PnP | 19 | 20 | 20 | 12 / 17 / 17 |
| DOGAR (2011) | 16 | 16 | **27** | 15 / 15 / 25 |
| E-M4M | 21 | 18 | 18 | 6 / 14 / 14 |
| I-M4M | 18 | 18 | 17 | 6 / 15 / 18 |
| DOGAR, M4M papers' version | 11 | 16 | 15 | - |
| M4M | 6 | 13 | 5 | 8 / 12 / 8 |

* **The push change is what the M4M family needed.** At the same 300 s, E-M4M goes from 34 to 54 solved, I-M4M from 39 to 51, I-M4M-PnP from 46 to 56 and Lazy I-M4M-PnP from 42 to 55. DOGAR, which did not change, solves exactly the same 55 problems as in overnight5. So the comparison is fair, and the gain comes from the push start and height. The largest gain is on the confined shelf (E-M4M 6 to 18 at 300 s), where pushes are the only way to clear the way.
* **The M4M family now matches or beats DOGAR.** At 900 s the order is Lazy I-M4M-PnP 65, then I-M4M-PnP and DOGAR 59, E-M4M 57, I-M4M 53. DOGAR still leads on the tabletop (27 of 48), where it can pick objects up and set them down anywhere. The M4M family leads on both shelves.
* **Against the DOGAR the papers tested, the papers' result holds.** The push-only `dogar_m4m` solves 42, against E-M4M's 57 and I-M4M's 53, and none of the 20 hard problems. Dogar's own planner solves 59 because of its PickUps.
* **M4M does not improve (28 to 24 at 900 s).** It loses 14 overnight5 solves and gains 10. M4M follows one CBS solution and stops when its suggested pushes run out ("no MAPF-suggested push left", 31 runs). The longer, backed-off pushes fail push IK more often, which ends those runs sooner. E-M4M, which replans CBS, does not have this problem.
* **More time helps a little.** From 300 s to 900 s the methods gain 1-3 solves, except Lazy I-M4M-PnP (+10), which keeps finding plans after 300 s on both tasks (+5 each).
* **Plans replay.** 3-10 plans per method fail the replay, mostly grasps where one finger misses the OoI and pushes where bodies are forced into each other.
* **Hard problems stay hard.** At most 6 of 20 (Lazy I-M4M-PnP).

**Feasibility (`dataset/feasibility.csv`).** 82 feasible (each with a replayed plan from this run), 1 infeasible and 17 unknown (overnight5: 73 / 2 / 25). One overnight5 "infeasible" problem, `open_tabletop_relocate_easy_0000`, is solved here by I-M4M-PnP on both copies. The infeasibility check is sampled (20 seeds of pick-and-place with every movable removed), so it can be wrong, and this label was.

**Repro.** `python scripts/overnight6_pipeline.py --workers 8 --budget 900` (it resumes from `../pipeline_state.json`). Earlier overnight results are unchanged.
