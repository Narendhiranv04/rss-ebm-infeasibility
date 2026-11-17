# PIGINet on shelf66, V4 test split: `noimg` (3 seeds: noimg_s0, noimg_s1, noimg_s2)

## A. Plan feasibility (every labelled test query)

| queries | n | feasible | AUROC | AP | acc@0.5 | bal. acc | F1 |
|---|---|---|---|---|---|---|---|
| original + D + B (labels stored in the scene files) | 3501 | 0.338 | 0.975 ± 0.002 | 0.950 ± 0.003 | 0.920 ± 0.003 | 0.920 ± 0.003 | 0.886 ± 0.002 |
| original (move T) | 370 | 0.181 | 0.987 ± 0.001 | 0.945 ± 0.010 | 0.954 ± 0.006 | 0.949 ± 0.002 | 0.881 ± 0.012 |
| D (edited scene, move T) | 1745 | 0.240 | 0.993 ± 0.001 | 0.981 ± 0.003 | 0.964 ± 0.005 | 0.960 ± 0.002 | 0.926 ± 0.010 |
| B (executed sequences) | 1386 | 0.504 | 0.932 ± 0.007 | 0.918 ± 0.011 | 0.856 ± 0.003 | 0.856 ± 0.003 | 0.863 ± 0.005 |
| L (rest of the T-only lattice) | 5264 | 0.038 | 0.984 ± 0.001 | 0.844 ± 0.002 | 0.976 ± 0.003 | 0.912 ± 0.009 | 0.730 ± 0.023 |
| P (lattice negatives in plan form) | 5960 | 0.000 | - | - | 0.981 ± 0.006 | - | - |
| all | 14725 | 0.094 | 0.989 ± 0.001 | 0.919 ± 0.001 | 0.965 ± 0.004 | 0.940 ± 0.006 | 0.830 ± 0.013 |

B sequences by role (accuracy at 0.5):

| role | n | feasible | acc | AUROC |
|---|---|---|---|---|
| intended_union | 42 | 0.000 | 1.000 ± 0.000 | - |
| minimal | 308 | 1.000 | 0.887 ± 0.023 | - |
| noop_irrelevant | 277 | 0.130 | 0.889 ± 0.009 | 0.949 ± 0.009 |
| redundant+1 | 232 | 0.461 | 0.819 ± 0.011 | 0.909 ± 0.013 |
| wrong_destination | 276 | 0.268 | 0.760 ± 0.035 | 0.891 ± 0.008 |
| wrong_order | 251 | 0.689 | 0.898 ± 0.010 | 0.904 ± 0.004 |

## B. Scene level (candidate plans enumerated and scored)

Candidates per scene: 26 on average (move T; every set of 1-3 obstacles to their first parking cells, then T).
`plan` = the relocations are plan tokens (PIGINet's interface); `edit` = the obstacles are already parked in the initial state.

### form: plan

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.773 ± 0.021 | 0.791 ± 0.003 |
| class macro-F1 | 0.775 ± 0.019 | 0.791 ± 0.003 |
| recall K0 | 0.940 ± 0.012 | 0.925 ± 0.000 |
| recall K1 | 0.660 ± 0.015 | 0.660 ± 0.015 |
| recall K2 | 0.709 ± 0.025 | 0.736 ± 0.020 |
| recall K3 | 0.654 ± 0.046 | 0.703 ± 0.012 |
| recall NONE | 0.973 ± 0.020 | 0.978 ± 0.015 |
| blocker-mask F1 | 0.808 ± 0.009 | 0.820 ± 0.007 |
| blocker-mask exact | 0.465 ± 0.043 | 0.501 ± 0.017 |
| tau* | | 0.55, 0.6, 0.9 |

- Solvable vs NONE (best candidate score): AUROC 0.998 ± 0.001
- Blocker set given the true K (K1-K3 scenes, n = 241): top-1 0.899 ± 0.016, top-3 0.992 ± 0.000

### form: edit

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.844 ± 0.024 | 0.855 ± 0.010 |
| class macro-F1 | 0.845 ± 0.025 | 0.852 ± 0.010 |
| recall K0 | 0.940 ± 0.012 | 0.876 ± 0.019 |
| recall K1 | 0.730 ± 0.054 | 0.730 ± 0.024 |
| recall K2 | 0.822 ± 0.052 | 0.798 ± 0.045 |
| recall K3 | 0.765 ± 0.029 | 0.866 ± 0.030 |
| recall NONE | 1.000 ± 0.000 | 1.000 ± 0.000 |
| blocker-mask F1 | 0.864 ± 0.011 | 0.881 ± 0.007 |
| blocker-mask exact | 0.573 ± 0.044 | 0.604 ± 0.008 |
| tau* | | 0.95, 0.65, 0.95 |

- Solvable vs NONE (best candidate score): AUROC 1.000 ± 0.000
- Blocker set given the true K (K1-K3 scenes, n = 241): top-1 0.963 ± 0.009, top-3 0.996 ± 0.000

- Direct feasibility (K0 vs rest, `original`): AUROC 0.987 ± 0.001, accuracy 0.954 ± 0.006

## C. Plan ordering (labelled plans of a scene, original + B)

- Scenes with both feasible and infeasible labelled plans: 264 (4.7 labelled plans per scene)
- Plans refined until the first feasible one: PIGINet order 1.141 ± 0.016, length order (the paper's baseline) 2.752 ± 0.000
- PIGINet's top plan is feasible: 0.876 ± 0.008
- Scenes whose labelled plans are all infeasible (62): plans PIGINet would still refine (p >= 0.5) 0.000 ± 0.000 of 4.7
