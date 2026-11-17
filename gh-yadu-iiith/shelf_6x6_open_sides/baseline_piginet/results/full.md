# PIGINet on shelf66, V4 test split: `full` (3 seeds: full_s0, full_s1, full_s2)

## A. Plan feasibility (every labelled test query)

| queries | n | feasible | AUROC | AP | acc@0.5 | bal. acc | F1 |
|---|---|---|---|---|---|---|---|
| original + D + B (labels stored in the scene files) | 3501 | 0.338 | 0.955 ± 0.008 | 0.914 ± 0.017 | 0.892 ± 0.015 | 0.877 ± 0.008 | 0.839 ± 0.016 |
| original (move T) | 370 | 0.181 | 0.967 ± 0.006 | 0.904 ± 0.019 | 0.939 ± 0.010 | 0.889 ± 0.008 | 0.828 ± 0.023 |
| D (edited scene, move T) | 1745 | 0.240 | 0.980 ± 0.005 | 0.949 ± 0.012 | 0.940 ± 0.010 | 0.920 ± 0.010 | 0.875 ± 0.019 |
| B (executed sequences) | 1386 | 0.504 | 0.904 ± 0.020 | 0.891 ± 0.026 | 0.819 ± 0.023 | 0.819 ± 0.024 | 0.817 ± 0.016 |
| L (rest of the T-only lattice) | 5264 | 0.038 | 0.955 ± 0.007 | 0.658 ± 0.042 | 0.966 ± 0.006 | 0.829 ± 0.014 | 0.603 ± 0.046 |
| P (lattice negatives in plan form) | 5960 | 0.000 | - | - | 0.990 ± 0.002 | - | - |
| all | 14725 | 0.094 | 0.976 ± 0.003 | 0.853 ± 0.020 | 0.958 ± 0.006 | 0.891 ± 0.001 | 0.785 ± 0.022 |

B sequences by role (accuracy at 0.5):

| role | n | feasible | acc | AUROC |
|---|---|---|---|---|
| intended_union | 42 | 0.000 | 1.000 ± 0.000 | - |
| minimal | 308 | 1.000 | 0.777 ± 0.004 | - |
| noop_irrelevant | 277 | 0.130 | 0.880 ± 0.011 | 0.939 ± 0.004 |
| redundant+1 | 232 | 0.461 | 0.784 ± 0.036 | 0.866 ± 0.022 |
| wrong_destination | 276 | 0.268 | 0.745 ± 0.093 | 0.811 ± 0.035 |
| wrong_order | 251 | 0.689 | 0.886 ± 0.015 | 0.909 ± 0.010 |

## B. Scene level (candidate plans enumerated and scored)

Candidates per scene: 26 on average (move T; every set of 1-3 obstacles to their first parking cells, then T).
`plan` = the relocations are plan tokens (PIGINet's interface); `edit` = the obstacles are already parked in the initial state.

### form: plan

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.728 ± 0.013 | 0.728 ± 0.011 |
| class macro-F1 | 0.725 ± 0.014 | 0.726 ± 0.012 |
| recall K0 | 0.811 ± 0.014 | 0.836 ± 0.037 |
| recall K1 | 0.541 ± 0.085 | 0.541 ± 0.085 |
| recall K2 | 0.651 ± 0.009 | 0.643 ± 0.005 |
| recall K3 | 0.712 ± 0.047 | 0.703 ± 0.040 |
| recall NONE | 0.930 ± 0.015 | 0.930 ± 0.008 |
| blocker-mask F1 | 0.825 ± 0.011 | 0.822 ± 0.010 |
| blocker-mask exact | 0.490 ± 0.009 | 0.479 ± 0.012 |
| tau* | | 0.3, 0.65, 0.35 |

- Solvable vs NONE (best candidate score): AUROC 0.977 ± 0.019
- Blocker set given the true K (K1-K3 scenes, n = 241): top-1 0.902 ± 0.016, top-3 0.994 ± 0.002

### form: edit

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.741 ± 0.030 | 0.753 ± 0.024 |
| class macro-F1 | 0.742 ± 0.033 | 0.754 ± 0.028 |
| recall K0 | 0.811 ± 0.014 | 0.801 ± 0.025 |
| recall K1 | 0.604 ± 0.086 | 0.642 ± 0.067 |
| recall K2 | 0.655 ± 0.074 | 0.659 ± 0.069 |
| recall K3 | 0.690 ± 0.024 | 0.716 ± 0.035 |
| recall NONE | 0.984 ± 0.000 | 0.989 ± 0.008 |
| blocker-mask F1 | 0.829 ± 0.018 | 0.835 ± 0.012 |
| blocker-mask exact | 0.483 ± 0.039 | 0.495 ± 0.024 |
| tau* | | 0.75, 0.95, 0.35 |

- Solvable vs NONE (best candidate score): AUROC 0.998 ± 0.000
- Blocker set given the true K (K1-K3 scenes, n = 241): top-1 0.928 ± 0.017, top-3 0.994 ± 0.002

- Direct feasibility (K0 vs rest, `original`): AUROC 0.967 ± 0.006, accuracy 0.939 ± 0.010

## C. Plan ordering (labelled plans of a scene, original + B)

- Scenes with both feasible and infeasible labelled plans: 264 (4.7 labelled plans per scene)
- Plans refined until the first feasible one: PIGINet order 1.135 ± 0.047, length order (the paper's baseline) 2.752 ± 0.000
- PIGINet's top plan is feasible: 0.889 ± 0.029
- Scenes whose labelled plans are all infeasible (62): plans PIGINet would still refine (p >= 0.5) 0.000 ± 0.000 of 4.7
