# GRN on shelf66, V4 test split: `grn` (3 seeds: s0, s1, s2)

## A. Action feasibility: can the target reach its goal in this layout?

| queries | n | feasible | AUROC | AP | acc@0.5 | F1 |
|---|---|---|---|---|---|---|
| original + D (labels stored in the scene files) | 2115 | 0.229 | 0.959 ± 0.000 | 0.873 ± 0.002 | 0.901 ± 0.003 | 0.753 ± 0.011 |
| original (move T) | 370 | 0.181 | 0.958 ± 0.001 | 0.856 ± 0.004 | 0.919 ± 0.004 | 0.755 ± 0.013 |
| D (edited scene, obstacles parked) | 1745 | 0.240 | 0.960 ± 0.000 | 0.880 ± 0.003 | 0.897 ± 0.003 | 0.752 ± 0.011 |
| L (rest of the T-only lattice) | 5264 | 0.038 | 0.881 ± 0.003 | 0.276 ± 0.007 | 0.958 ± 0.002 | 0.282 ± 0.039 |
| all | 7379 | 0.093 | 0.940 ± 0.001 | 0.721 ± 0.007 | 0.941 ± 0.002 | 0.627 ± 0.016 |

Per-view heads (original + D test queries that carry labels):

- view feasibility AUROC 0.989 ± 0.000 (n = 26565 labelled views)
- IK reachability AUROC 1.000 ± 0.000, accuracy 0.999 ± 0.000 (n = 26565)
- obstruction (GO) mean absolute error on reachable views 0.116 ± 0.023 (1630 graphs)

## B. Scene level (layouts enumerated: original + every set of 1-3 obstacles parked at their first parking cell)

Candidates per scene: 26 on average.

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.574 ± 0.013 | 0.601 ± 0.011 |
| class macro-F1 | 0.574 ± 0.013 | 0.604 ± 0.013 |
| recall K0 | 0.692 ± 0.019 | 0.746 ± 0.032 |
| recall K1 | 0.547 ± 0.015 | 0.591 ± 0.032 |
| recall K2 | 0.376 ± 0.047 | 0.422 ± 0.052 |
| recall K3 | 0.422 ± 0.058 | 0.448 ± 0.026 |
| recall NONE | 0.995 ± 0.008 | 0.952 ± 0.013 |
| tau* | | 0.4, 0.45, 0.4 |

- Solvable vs NONE (best layout score): AUROC 0.975 ± 0.002
- Blocker set given the true K (K1-K3 scenes, n = 241): top-1 0.788 ± 0.009, top-3 0.970 ± 0.002
- Blocker mask from GRN's obstruction head (tau_go tuned on validation: 0.05, 0.1, 0.05; n = 264 scenes): F1 0.834 ± 0.003, precision 0.732 ± 0.002, recall 0.967 ± 0.007, exact 0.390 ± 0.005
- Same at tau_go = 0.5: F1 0.816 ± 0.010
