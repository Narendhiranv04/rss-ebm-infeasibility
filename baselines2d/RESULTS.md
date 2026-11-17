# Results: analytical baselines on the 2D test split

E-M4M, MS-MCTS and MO-SeGMan were run on all 749 test scenes (K0 187, K1 120, K2 120, K3 172, NONE 150) with 3 seeds each, 6,741 runs
in all. Every run has a 300 CPU-s budget, and the 60 s table is read from the same runs. All runs completed between 2026-11-14 and
2026-11-16 on one laptop (Docker, 6 workers). There were **0 errors and 0 censored runs**, and **all 1,501 returned repairs passed the
dataset's X/Y checker**.

Metrics are the same as Approach B's Table I(a) (`approach_b_final/evaluate.py`), reported as mean ± std over seeds 0, 1 and 2. The std
is the population std of the three per-seed values.

## Budget 300 CPU-s

| Method | Task % ↑ | Repair % ↑ | No-op % ↑ | Abstain % ↑ | Calls ↓ | Travel ↓ | Task K1 / K2 / K3 % | CPU/run (s) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| E-M4M | **61.7 ± 1.2** | **34.2 ± 1.5** | 94.1 ± 0.0 | 98.0 ± 0.0 | **17.7 ± 0.3** | 4.04 ± 0.11 | 83.6 / 78.3 / 34.9 | 141 ± 1 |
| MO-SeGMan | 30.7 ± 1.8 | 25.6 ± 1.1 | 94.1 ± 0.0 | **100.0 ± 0.0** | 19.3 ± 0.1 | 3.64 ± 0.06 | 56.1 / 37.2 / 8.5 | 159 ± 1 |
| MS-MCTS | 27.0 ± 0.3 | 23.3 ± 0.1 | 94.1 ± 0.0 | 98.7 ± 0.0 | 30.5 ± 0.1 | **2.90 ± 0.02** | 60.8 / 26.7 / 3.7 | 210 ± 0 |

## Budget 60 CPU-s (read from the same runs)

| Method | Task % ↑ | Repair % ↑ | No-op % ↑ | Abstain % ↑ | Calls ↓ | Travel ↓ | Task K1 / K2 / K3 % | CPU/run (s) |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| E-M4M | **15.5 ± 0.4** | **12.6 ± 0.7** | 93.6 ± 0.0 | 100.0 ± 0.0 | 5.9 ± 0.0 | 2.60 ± 0.06 | 25.8 / 19.2 / 5.8 | 44 ± 0 |
| MO-SeGMan | 2.0 ± 0.1 | 2.1 ± 0.1 | 93.6 ± 0.0 | 100.0 ± 0.0 | 8.0 ± 0.0 | 2.64 ± 0.02 | 5.3 / 1.7 / 0.0 | 47 ± 0 |
| MS-MCTS | 1.5 ± 0.0 | 1.5 ± 0.0 | 93.6 ± 0.0 | 100.0 ± 0.0 | 7.8 ± 0.0 | 1.92 ± 0.00 | 5.0 / 0.0 / 0.0 | 49 ± 0 |

At this cut, a plan found after 60 CPU-s counts as ABSTAIN. Calls and CPU count only work started within the cut.

## Metric definitions

| Metric | Definition | Scenes |
|---|---|---|
| Task % | REPAIR with X = 1 and Y = 1 from the finalized checker (`bxy_teacher`) | eval_task: 412 (K1–K3) |
| Repair % | Task success with exactly K relocations | eval_repair: 405 |
| No-op % | output NOOP | eval_noop: 187 (K0) |
| Abstain % | output ABSTAIN | eval_abstain: 150 (NONE) |
| Calls | motion-planner calls per run (all scenes) | 749 |
| Travel | witnessed joint travel Σ\|Δq\|/2π of the relocations, connector and target (`cost.jrec`), successful repairs only | Task successes |
| CPU/run | CPU-s per run, capped at the budget, with the dataset's virtual accounting (a cached motion result costs what computing it cost) | 749 |

Note that Repair % and Task % have different denominators (405 vs 412). This is why MO-SeGMan's Repair % can exceed its Task % at 60 s.

## Notes for the paper

1. **No-op 94.1% is a motion-layer effect and is the same for all three baselines.** Every motion call runs under a 45 CPU-s cap,
   because the dataset's planner has no fail-fast bound and a failing query would otherwise spend ~300 s. In 11 of the 187 K0 scenes the
   target plan needs longer than the cap. Measured with no cap (`scripts/k0_uncapped_probe.py`), it needs **66, 93 and 123 CPU-s** on
   three of them, and the checker confirms X = Y = 1 there with no move. The baselines abstain in 10 of those scenes. In 1
   (`2d-conv-plus2d-w0120-s010`) they return a verified 1-move repair instead of a no-op.
2. **Three NONE scenes have checker-verified repairs:** `2d-plus2d-srv-w0248-s004` (1 move, E-M4M and MS-MCTS),
   `2d-plus2d-srv-w0167-s011` (1 move, E-M4M and MS-MCTS) and `2d-plus2d-w0166-s021` (2 moves, E-M4M). The same repairs appear in all
   three seeds. These are dataset label errors, and they cap Abstain % below 100% for any method that finds them. Approach B is scored
   against the same labels, so these scenes should be checked with the dataset authors.
3. **E-M4M does not minimise the number of moves.** Over all seeds, 342 of its 763 verified K1–K3 successes use more than K moves
   (MO-SeGMan 63 of 380, MS-MCTS 45 of 334). This is why its Repair % is much lower than its Task %.
4. **Median CPU per successful K1–K3 run** is 108 s for E-M4M, 131 s for MO-SeGMan and 163 s for MS-MCTS. Single motion calls take
   3–33 s, so almost nothing finishes within 60 s.
5. **Seeds:** the motion layer is deterministic (DetPhysics). Seeds change only the planners' own random choices: E-M4M's tie
   breaking, MS-MCTS rollouts, and MO-SeGMan's relocation sampling.

## Files

| File | Content |
|---|---|
| `results/table.md`, `results/table.json` | the tables above; the JSON also holds per-seed values, per-K Task %, shorter-than-K counts and median CPU |
| `results/per_scene.csv` | 6,741 rows, one per (scene, method, seed): status, plan, X, Y, task success, travel, CPU, calls, reason |
| `results/per_seed_summary.csv` | every metric for each seed separately |
| `results/flagged_scenes.csv` | the K0 and NONE scenes from notes 1–2, with plans |
| `out/runs/main/results.jsonl` | raw run records, including the per-call motion trace (kind, start CPU-s, duration) |
| `out/runs/smoke/` | the 5-scene smoke test used during development |
| `out/witnesses/` | the checker's witness trajectories (travel is computed from these) |
| `out/cache/` | per-scene exact motion caches (re-running reuses them) |
