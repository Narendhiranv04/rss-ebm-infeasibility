# SceneCollisionNet on shelf66, V4 test split: `scn` (3 seeds: adam_s0, adam_s1, adam_s2)

## A. State feasibility: can the target reach its goal in this layout? (score = max over the 32 grasps)

| queries | n | feasible | AUROC | AP | acc@0.5 | F1 |
|---|---|---|---|---|---|---|
| original + D (labels stored in the scene files) | 2115 | 0.229 | 0.978 ± 0.001 | 0.942 ± 0.002 | 0.935 ± 0.005 | 0.849 ± 0.015 |
| original (move T) | 370 | 0.181 | 0.968 ± 0.003 | 0.890 ± 0.015 | 0.922 ± 0.010 | 0.753 ± 0.040 |
| D (edited scene, obstacles parked) | 1745 | 0.240 | 0.979 ± 0.001 | 0.949 ± 0.001 | 0.938 ± 0.005 | 0.863 ± 0.012 |
| L (rest of the T-only lattice) | 5264 | 0.038 | 0.964 ± 0.004 | 0.743 ± 0.006 | 0.972 ± 0.002 | 0.676 ± 0.012 |
| all | 7379 | 0.093 | 0.974 ± 0.001 | 0.869 ± 0.003 | 0.961 ± 0.001 | 0.792 ± 0.008 |

## B. Per grasp (original + D test queries that carry labels)

| metric | value | n |
|---|---|---|
| grasp feasibility AUROC | 0.985 ± 0.001 | 53962 |
| pick collision AUROC (hand at the grasp pose is unusable) | 0.986 ± 0.001 | 52533 |
| place collision AUROC (hand over the goal collides) | 0.947 ± 0.002 | 1736 |
| the teacher's winning grasp is among the 3 best-scored | 0.669 ± 0.024 | |
| the teacher's winning grasp is the best-scored | 0.409 ± 0.030 | |

## C. Scene level (layouts enumerated)

Candidates per scene: 26 on average.

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.756 ± 0.003 | 0.764 ± 0.017 |
| class macro-F1 | 0.750 ± 0.006 | 0.759 ± 0.014 |
| recall K0 | 0.667 ± 0.051 | 0.687 ± 0.012 |
| recall K1 | 0.635 ± 0.032 | 0.648 ± 0.009 |
| recall K2 | 0.702 ± 0.024 | 0.705 ± 0.033 |
| recall K3 | 0.804 ± 0.021 | 0.807 ± 0.036 |
| recall NONE | 0.952 ± 0.035 | 0.957 ± 0.020 |
| tau* | | 0.7, 0.65, 0.2 |

- Solvable vs NONE (best layout score): AUROC 0.996 ± 0.002
- Blocker set given the true K (K1-K3 scenes, n = 241): top-1 0.924 ± 0.017, top-3 0.986 ± 0.005
- Blocker mask by counterfactual gain (delta tuned on validation: 0.01, 0.01, 0.01; n = 264 scenes): F1 0.250 ± 0.013, precision 0.824 ± 0.010, recall 0.147 ± 0.009, exact 0.119 ± 0.005
