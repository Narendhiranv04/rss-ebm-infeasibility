# SceneCollisionNet on shelf66, V4 test split: `scn_sgd` (1 seed: sgd_s0)

## A. State feasibility: can the target reach its goal in this layout? (score = max over the 32 grasps)

| queries | n | feasible | AUROC | AP | acc@0.5 | F1 |
|---|---|---|---|---|---|---|
| original + D (labels stored in the scene files) | 2115 | 0.229 | 0.976 | 0.934 | 0.933 | 0.853 |
| original (move T) | 370 | 0.181 | 0.966 | 0.883 | 0.916 | 0.770 |
| D (edited scene, obstacles parked) | 1745 | 0.240 | 0.977 | 0.941 | 0.936 | 0.866 |
| L (rest of the T-only lattice) | 5264 | 0.038 | 0.954 | 0.673 | 0.961 | 0.600 |
| all | 7379 | 0.093 | 0.970 | 0.846 | 0.953 | 0.764 |

## B. Per grasp (original + D test queries that carry labels)

| metric | value | n |
|---|---|---|
| grasp feasibility AUROC | 0.984 | 53962 |
| pick collision AUROC (hand at the grasp pose is unusable) | 0.974 | 52533 |
| place collision AUROC (hand over the goal collides) | 0.953 | 1736 |
| the teacher's winning grasp is among the 3 best-scored | 0.654 | |
| the teacher's winning grasp is the best-scored | 0.297 | |

## C. Scene level (layouts enumerated)

Candidates per scene: 26 on average.

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.751 | 0.751 |
| class macro-F1 | 0.750 | 0.746 |
| recall K0 | 0.776 | 0.716 |
| recall K1 | 0.623 | 0.585 |
| recall K2 | 0.721 | 0.709 |
| recall K3 | 0.706 | 0.775 |
| recall NONE | 0.952 | 0.952 |
| tau* | | 0.7 |

- Solvable vs NONE (best layout score): AUROC 0.992
- Blocker set given the true K (K1-K3 scenes, n = 241): top-1 0.921, top-3 0.979
- Blocker mask by counterfactual gain (delta tuned on validation: 0.01; n = 264 scenes): F1 0.364, precision 0.839, recall 0.233, exact 0.121
