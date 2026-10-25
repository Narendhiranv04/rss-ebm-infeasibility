# Overnight benchmark (20261024-234342_overnight-900s)

424 runs (done, 424/424); budget 900 s; solved = planned and replayed successfully. Success within shorter budgets is read off the same runs.

## Success within each budget

| method | runs | <= 60 s | <= 120 s | <= 300 s | <= 600 s | <= 900 s |
|---|---|---|---|---|---|---|
| M4M | 53 | 16 | 20 | 20 | 20 | 20 |
| E-M4M | 53 | 21 | 24 | 24 | 24 | 26 |
| I-M4M | 53 | 23 | 24 | 27 | 28 | 28 |
| I-M4M-PnP | 53 | 21 | 25 | 26 | 28 | 28 |
| Lazy I-M4M | 53 | 23 | 25 | 26 | 26 | 26 |
| Lazy I-M4M-PnP | 53 | 22 | 25 | 29 | 30 | 31 |
| DOGAR | 53 | 7 | 7 | 9 | 9 | 11 |
| DOGAR (uniform) | 53 | 3 | 6 | 6 | 8 | 11 |

## Per environment (solved within 300 s -> within 900 s / runs)

| environment (300 s -> 900 s) | M4M | E-M4M | I-M4M | I-M4M-PnP | Lazy I-M4M | Lazy I-M4M-PnP | DOGAR | DOGAR (uniform) |
|---|---|---|---|---|---|---|---|---|
| Reference shelf (M4M) | 6 -> 6 / 9 | 6 -> 6 / 9 | 7 -> 7 / 9 | 6 -> 6 / 9 | 7 -> 7 / 9 | 7 -> 7 / 9 | 2 -> 2 / 9 | 2 -> 2 / 9 |
| Open tabletop | 0 -> 0 / 9 | 1 -> 1 / 9 | 1 -> 1 / 9 | 1 -> 2 / 9 | 1 -> 1 / 9 | 1 -> 2 / 9 | 0 -> 0 / 9 | 0 -> 0 / 9 |
| Confined shelf | 5 -> 5 / 9 | 6 -> 6 / 9 | 6 -> 6 / 9 | 5 -> 5 / 9 | 6 -> 6 / 9 | 6 -> 6 / 9 | 2 -> 2 / 9 | 1 -> 3 / 9 |
| Multi-level shelf | 4 -> 4 / 9 | 5 -> 6 / 9 | 5 -> 6 / 9 | 5 -> 6 / 9 | 5 -> 5 / 9 | 6 -> 6 / 9 | 2 -> 3 / 9 | 1 -> 3 / 9 |
| Access-19 | 0 -> 0 / 8 | 0 -> 0 / 8 | 0 -> 0 / 8 | 3 -> 3 / 8 | 0 -> 0 / 8 | 3 -> 3 / 8 | 0 -> 0 / 8 | 0 -> 0 / 8 |
| 3D stacking | 5 -> 5 / 9 | 6 -> 7 / 9 | 8 -> 8 / 9 | 6 -> 6 / 9 | 7 -> 7 / 9 | 6 -> 7 / 9 | 3 -> 4 / 9 | 2 -> 3 / 9 |

## Per difficulty

| difficulty (300 s -> 900 s) | M4M | E-M4M | I-M4M | I-M4M-PnP | Lazy I-M4M | Lazy I-M4M-PnP | DOGAR | DOGAR (uniform) |
|---|---|---|---|---|---|---|---|---|
| easy | 11 -> 11 / 18 | 13 -> 13 / 18 | 13 -> 13 / 18 | 15 -> 16 / 18 | 12 -> 12 / 18 | 14 -> 15 / 18 | 8 -> 9 / 18 | 6 -> 11 / 18 |
| medium | 7 -> 7 / 18 | 10 -> 11 / 18 | 10 -> 11 / 18 | 10 -> 11 / 18 | 11 -> 11 / 18 | 12 -> 12 / 18 | 1 -> 2 / 18 | 0 -> 0 / 18 |
| hard | 2 -> 2 / 17 | 1 -> 2 / 17 | 4 -> 4 / 17 | 1 -> 1 / 17 | 3 -> 3 / 17 | 3 -> 4 / 17 | 0 -> 0 / 17 | 0 -> 0 / 17 |

## Planning time

| method | solved | median planning time (s) | solved after 300 s |
|---|---|---|---|
| M4M | 20 | 19 | 0 |
| E-M4M | 26 | 25 | 2 |
| I-M4M | 28 | 26 | 1 |
| I-M4M-PnP | 28 | 18 | 2 |
| Lazy I-M4M | 26 | 17 | 0 |
| Lazy I-M4M-PnP | 31 | 19 | 2 |
| DOGAR | 11 | 20 | 2 |
| DOGAR (uniform) | 11 | 101 | 5 |

## Unsolved runs

| unsolved: reason | M4M | E-M4M | I-M4M | I-M4M-PnP | Lazy I-M4M | Lazy I-M4M-PnP | DOGAR | DOGAR (uniform) |
|---|---|---|---|---|---|---|---|---|
| CBS failed | 15 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CRASH | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| OPEN exhausted | 0 | 21 | 20 | 15 | 21 | 15 | 0 | 0 |
| budget exhausted | 15 | 4 | 4 | 5 | 5 | 5 | 37 | 41 |
| no MAPF-suggested push left | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| no retrieval trajectory without movables | 1 | 1 | 1 | 1 | 1 | 1 | 0 | 0 |
| plan failed on replay | 1 | 1 | 0 | 3 | 0 | 1 | 5 | 1 |

Plots: anytime_by_environment.png, anytime_by_difficulty.png, budget_300_vs_900.png (also .gif). Gallery: index.html (scene GIFs with their task, solution GIFs).