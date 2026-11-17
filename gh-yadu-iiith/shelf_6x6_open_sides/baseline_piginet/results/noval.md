# PIGINet on shelf66, V4 test split: `noval` (1 seed: noval_s0)

## A. Plan feasibility (every labelled test query)

| queries | n | feasible | AUROC | AP | acc@0.5 | bal. acc | F1 |
|---|---|---|---|---|---|---|---|
| original + D + B (labels stored in the scene files) | 3501 | 0.338 | 0.906 | 0.814 | 0.804 | 0.833 | 0.761 |
| original (move T) | 370 | 0.181 | 0.934 | 0.802 | 0.873 | 0.870 | 0.712 |
| D (edited scene, move T) | 1745 | 0.240 | 0.927 | 0.824 | 0.825 | 0.854 | 0.714 |
| B (executed sequences) | 1386 | 0.504 | 0.845 | 0.816 | 0.760 | 0.758 | 0.796 |
| L (rest of the T-only lattice) | 5264 | 0.038 | 0.876 | 0.232 | 0.893 | 0.780 | 0.318 |
| P (lattice negatives in plan form) | 5960 | 0.000 | - | - | 0.946 | - | - |
| all | 14725 | 0.094 | 0.951 | 0.721 | 0.893 | 0.888 | 0.608 |

B sequences by role (accuracy at 0.5):

| role | n | feasible | acc | AUROC |
|---|---|---|---|---|
| intended_union | 42 | 0.000 | 0.976 | - |
| minimal | 308 | 1.000 | 0.893 | - |
| noop_irrelevant | 277 | 0.130 | 0.848 | 0.913 |
| redundant+1 | 232 | 0.461 | 0.642 | 0.785 |
| wrong_destination | 276 | 0.268 | 0.449 | 0.730 |
| wrong_order | 251 | 0.689 | 0.912 | 0.901 |

## B. Scene level (candidate plans enumerated and scored)

Candidates per scene: 26 on average (move T; every set of 1-3 obstacles to their first parking cells, then T).
`plan` = the relocations are plan tokens (PIGINet's interface); `edit` = the obstacles are already parked in the initial state.

### form: plan

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.481 | 0.595 |
| class macro-F1 | 0.499 | 0.605 |
| recall K0 | 0.866 | 0.746 |
| recall K1 | 0.415 | 0.434 |
| recall K2 | 0.337 | 0.477 |
| recall K3 | 0.137 | 0.490 |
| recall NONE | 0.887 | 0.903 |
| blocker-mask F1 | 0.689 | 0.767 |
| blocker-mask exact | 0.183 | 0.340 |
| tau* | | 0.85 |

- Solvable vs NONE (best candidate score): AUROC 0.972
- Blocker set given the true K (K1-K3 scenes, n = 241): top-1 0.884, top-3 0.992

### form: edit

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.473 | 0.511 |
| class macro-F1 | 0.478 | 0.501 |
| recall K0 | 0.866 | 0.746 |
| recall K1 | 0.396 | 0.472 |
| recall K2 | 0.279 | 0.349 |
| recall K3 | 0.206 | 0.245 |
| recall NONE | 0.823 | 0.952 |
| blocker-mask F1 | 0.620 | 0.571 |
| blocker-mask exact | 0.187 | 0.199 |
| tau* | | 0.85 |

- Solvable vs NONE (best candidate score): AUROC 0.947
- Blocker set given the true K (K1-K3 scenes, n = 241): top-1 0.763, top-3 0.959

- Direct feasibility (K0 vs rest, `original`): AUROC 0.934, accuracy 0.873

## C. Plan ordering (labelled plans of a scene, original + B)

- Scenes with both feasible and infeasible labelled plans: 264 (4.7 labelled plans per scene)
- Plans refined until the first feasible one: PIGINet order 1.205, length order (the paper's baseline) 2.752
- PIGINet's top plan is feasible: 0.837
- Scenes whose labelled plans are all infeasible (62): plans PIGINet would still refine (p >= 0.5) 0.097 of 4.7
