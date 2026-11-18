# GRN (Geometric Reasoning Network) on the 6x6 shelf dataset

Ait Bouhsain, Alami, Simeon, "Learning geometric reasoning networks for robot task and motion planning", ICLR 2025
(code: github.com/Smail8/geometric_reasoning_networks, MIT). **The paper itself could not be read** (OpenReview blocks automated
downloads, there is no arXiv version); everything here follows the authors' released code and released data. `base_gnn.py` is the
authors' `EdgeGATv2Conv` unchanged (MIT, `LICENSE_GRN_MIT`).

**What GRN's own task is** (checked on the authors' released dataset `panda_3d_10`, HuggingFace `Smile8/GRN_datasets`): a datapoint is one
movable object standing at a pose in a cluttered 3D scene (Panda, shelves / baskets / racks), and the label is whether the robot can
**pick** it: the recorded failing stages are `grasp`, `approach`, `lift` and `motion_planning`, there is no place stage, and the authors
plot "feasible pick positions". Outputs per object: an "action" (pick) feasibility, the feasibility of 5 grasp views (top / front / rear /
right / left), per-view IK reachability, and per neighbour the per-view grasp-obstruction ratio (GO, "who blocks which view"). **GRN does
not predict or propose placements**; every output is a feasibility probability for the pose it is given.

**What we ask of it here** (an adaptation, not GRN's native task): *can the target T go from where it stands to its goal in this layout?*
(our label is the whole move: grasp, carry and release at the goal) and *which objects obstruct it?* T is the movable node at its start
pose and the goal enters as an extra marker node.

Everything below is on the V4 split of the 6x6 shelf dataset (`split_v4.json`, from `naren/dataset-done`): 1,766 train / 364 val /
370 test scenes, the same test scenes as the other baselines. Mean ± standard deviation over seeds.

## Results

### 1. Every model, same split, same metrics

All rows are feasibility predictors scored on the same 370 test scenes. "T move" columns: can the target reach its goal in that state (AUROC
over the stored queries of that kind). Scene-level columns enumerate candidate states / plans (the original scene and every set of 1-3
obstacles sent to their first parking cell, 26 per scene on average); the PIGINet rows use its two input forms (relocations as plan
tokens, or the obstacles already parked in the state, which is what GRN sees). `class acc`: NONE / K0-K3 (chance 0.28); `tau*` chosen on
validation. `blocker set top-1`: the best-scoring candidate of the true size K is a verified minimal set. `blocker-mask F1`: PIGINet from its
best-scoring set at tau 0.5, GRN from its own obstruction head (tau_go tuned on validation); the PIGINet "no value" row is one seed.

| model | seeds | T move, original scene (AUROC) | T move, edited scene (AUROC) | T move, lattice L (AUROC) | solvable vs NONE (AUROC) | class acc, tau 0.5 | class acc, tau* | blocker set top-1 (given K) | blocker-mask F1 |
|---|---|---|---|---|---|---|---|---|---|
| **GRN** (this folder) | 3 | 0.958 ± 0.001 | 0.960 ± 0.000 | 0.881 ± 0.003 | 0.975 ± 0.002 | 0.574 ± 0.013 | 0.601 ± 0.011 | 0.788 ± 0.009 | 0.834 ± 0.003 |
| PIGINet, full (image + pose + text), obstacles parked in the state | 3 | 0.967 ± 0.006 | 0.980 ± 0.005 | 0.955 ± 0.007 | 0.998 ± 0.000 | 0.741 ± 0.030 | 0.753 ± 0.024 | 0.928 ± 0.017 | 0.829 ± 0.018 |
| PIGINet, full (image + pose + text), relocations as plan tokens | 3 | 0.967 ± 0.006 | 0.980 ± 0.005 | 0.955 ± 0.007 | 0.977 ± 0.019 | 0.728 ± 0.013 | 0.728 ± 0.011 | 0.902 ± 0.016 | 0.825 ± 0.011 |
| PIGINet, no image tokens, obstacles parked in the state | 3 | 0.987 ± 0.001 | 0.993 ± 0.001 | 0.984 ± 0.001 | 1.000 ± 0.000 | 0.844 ± 0.024 | 0.855 ± 0.010 | 0.963 ± 0.009 | 0.864 ± 0.011 |
| PIGINet, no image tokens, relocations as plan tokens | 3 | 0.987 ± 0.001 | 0.993 ± 0.001 | 0.984 ± 0.001 | 0.998 ± 0.001 | 0.773 ± 0.021 | 0.791 ± 0.003 | 0.899 ± 0.016 | 0.808 ± 0.009 |
| PIGINet, no pose / size values, obstacles parked in the state | 1 | 0.934 | 0.927 | 0.876 | 0.947 | 0.473 | 0.511 | 0.763 | 0.620 |
| PIGINet, no pose / size values, relocations as plan tokens | 1 | 0.934 | 0.927 | 0.876 | 0.972 | 0.481 | 0.595 | 0.884 | 0.689 |

### 2. The same models in the columns of the paper's table (a)

AUC = scene solvable (K0-K3) vs NONE from the method's own score, computed with REAP's `metrics66.py`; Repair / Task / Time / Calls need a
planner. REAP and "PIGINet + TAMP" are run with the dataset teacher as the motion planner (PIGINet is given the scene file's parking cells as
destinations, so its Task / Time / Calls are optimistic; the solver code is not in this folder). GRN is a predictor only, so those columns
are empty; its AUC is the solvable-vs-NONE AUROC above.

| method | AUC | Repair (%) | Task (%) | Time (s) | Calls |
|---|---|---|---|---|---|
| REAP, released weights, zero-shot (1 seed) | 0.984 | 55.0 | 73.0 | 168.1 | 46.0 |
| PIGINet + TAMP, full (3 seeds) | 0.978 ± 0.021 | 68.9 ± 4.5 | 98.1 ± 0.2 | 26.9 ± 4.2 | 3.4 ± 0.4 |
| PIGINet + TAMP, no image tokens (3 seeds) | 0.995 ± 0.004 | 69.1 ± 0.7 | 99.1 ± 0.1 | 24.4 ± 2.3 | 3.3 ± 0.1 |
| **GRN** (feasibility predictor only, 3 seeds) | 0.975 ± 0.002 | – | – | – | – |

### 3. GRN in detail (3 seeds)

**A. Can the target reach its goal in this layout?** (all labelled test queries)

| queries | n | feasible | AUROC | AP | acc @ 0.5 | F1 @ 0.5 |
|---|---|---|---|---|---|---|
| original + D (labels stored in the scene files) | 2115 | 0.229 | 0.959 ± 0.000 | 0.873 ± 0.002 | 0.901 ± 0.003 | 0.753 ± 0.011 |
| original (move T, scene as it is) | 370 | 0.181 | 0.958 ± 0.001 | 0.856 ± 0.004 | 0.919 ± 0.004 | 0.755 ± 0.013 |
| D (edited scene, obstacles parked) | 1745 | 0.240 | 0.960 ± 0.000 | 0.880 ± 0.003 | 0.897 ± 0.003 | 0.752 ± 0.011 |
| L (rest of the teacher's T-only lattice) | 5264 | 0.038 | 0.881 ± 0.003 | 0.276 ± 0.007 | 0.958 ± 0.002 | 0.282 ± 0.039 |
| all | 7379 | 0.093 | 0.940 ± 0.001 | 0.721 ± 0.007 | 0.941 ± 0.002 | 0.627 ± 0.016 |

Per-view heads, trained with the per-grasp reasons the teacher stored (IK is easy: the label depends only on the target's cell):

| per-view head (original + D test queries with labels) | value | n |
|---|---|---|
| view feasibility, AUROC | 0.989 ± 0.000 | 26565 |
| IK reachability, AUROC | 1.000 ± 0.000 | 26565 |
| IK reachability, accuracy | 0.999 ± 0.000 | 26565 |
| obstruction (GO), mean absolute error on reachable views (lower is better) | 0.116 ± 0.023 | 1630 graphs |

**B. Scene level**

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.574 ± 0.013 | 0.601 ± 0.011 |
| class macro-F1 | 0.574 ± 0.013 | 0.604 ± 0.013 |
| recall K0 | 0.692 ± 0.019 | 0.746 ± 0.032 |
| recall K1 | 0.547 ± 0.015 | 0.591 ± 0.032 |
| recall K2 | 0.376 ± 0.047 | 0.422 ± 0.052 |
| recall K3 | 0.422 ± 0.058 | 0.448 ± 0.026 |
| recall NONE | 0.995 ± 0.008 | 0.952 ± 0.013 |
| tau* (chosen on validation), per seed | – | 0.4, 0.45, 0.4 |

| scene-level metric (candidates: original + every set of 1-3 obstacles parked, 26 layouts per scene on average) | value |
|---|---|
| solvable (K0-K3) vs NONE, AUROC (best layout score) | 0.975 ± 0.002 |
| blocker set given the true K, top-1 (n = 241 K1-K3 scenes) | 0.788 ± 0.009 |
| blocker set given the true K, top-3 | 0.970 ± 0.002 |
| blocker mask from GRN's obstruction head, F1 (n = 264 scenes, tau_go tuned on validation) | 0.834 ± 0.003 |
|   precision | 0.732 ± 0.002 |
|   recall | 0.967 ± 0.007 |
|   exact match | 0.390 ± 0.005 |
|   tau_go chosen, per seed | 0.05, 0.1, 0.05 |
| blocker-mask F1 at tau_go = 0.5 | 0.816 ± 0.010 |

Confusion matrix of the class prediction at tau = 0.5, mean counts over the 3 seeds (370 scenes):

| true \ predicted | K0 | K1 | K2 | K3 | NONE |
|---|---|---|---|---|---|
| K0 | 46.3 | 15.3 | 2.3 | 0.7 | 2.3 |
| K1 | 8.0 | 29.0 | 11.0 | 0.7 | 4.3 |
| K2 | 0.0 | 17.0 | 32.3 | 16.7 | 20.0 |
| K3 | 1.3 | 2.7 | 19.0 | 43.0 | 36.0 |
| NONE | 0.0 | 0.3 | 0.0 | 0.0 | 61.7 |

**Per seed**

| seed | best epoch | val action AUROC | test AUROC, all | test AUROC, original + D | test AUROC, L | class acc, tau 0.5 | class acc, tau* | solvable vs NONE | blocker set top-1 | blocker-mask F1 |
|---|---|---|---|---|---|---|---|---|---|---|
| s0 | 97 | 0.9498 | 0.942 | 0.959 | 0.885 | 0.592 | 0.616 | 0.974 | 0.776 | 0.836 |
| s1 | 77 | 0.9471 | 0.939 | 0.959 | 0.880 | 0.570 | 0.589 | 0.973 | 0.793 | 0.835 |
| s2 | 99 | 0.9498 | 0.940 | 0.960 | 0.877 | 0.559 | 0.597 | 0.977 | 0.797 | 0.829 |

## Adaptation to the shelf (`build_graphs.py`)
- **Views.** The shelf has side grasps only, so a "view" is one of the 16 grasp azimuths (the authors' Panda uses 5 views); the output widths
  are 16 (IK, GO) and 1 + 16 (action + views). Nothing else in the architecture changes.
- **Graph.** `base` (the shelf, fixed) · `goal` (a fixed marker box of the target's size at the goal cell: GRN has no goal input) · the
  obstacles (fixed: the dataset labels only the target's move) · `t0` (the only movable node, carries every label).
- **Queries = graphs.** The original scene, the teacher's edited layouts (D), and the rest of the teacher's T-only lattice (L, reconstructed
  from `labels.blocker.minimal_sets`, action label only): 34,696 train / 6,548 val / 7,379 test graphs.
- **Per-view labels** come from the teacher's stored per-grasp failure reasons (32 grasps = 16 azimuths x 2 grasp levels, ids ordered as in
  `shelf_scene.grasp_poses`): "out of reach" -> IK failure; "would hit o03" -> obstruction by that object; "would hit the floor_board /
  arm_ceiling / back_wall" -> obstruction by the shelf (edge `base -> t0`); "overlap a neighbour at the goal", "no sideways route" -> the
  grasp fails, nobody is named. Only infeasible queries store all 32 reasons; a feasible query stores its winning grasp, so the rest of it
  is masked in the loss.
- **Training** (`train_grn.py`; the authors' hyper-parameters: 100 epochs, Adam, IK / GO lr 1e-3 batch 8192, GRN lr 1e-4 batch 512, loss
  BCE(F) + BCE(IK) + 10 x MSE(GO)): IK module -> GO module -> joint GRN. The authors' separate "AGF with ground-truth IK / GO" stage is
  skipped (most of our queries carry no ground-truth GO). No augmentation (the authors' dimension switching changes nothing for square
  footprints). 2.28 M parameters, from scratch, best validation action AUROC kept; about 35 minutes per seed on 8 CPU threads.
- **Scene level** (`evaluate_grn.py`): GRN scores one state, so a repair is a state; K = the smallest set size whose best state scores >= tau.

## Limits
- Parking cells are given to the candidates (as for PIGINet): GRN does not choose placements, and it was not designed to.
- The obstacles' own per-view labels do not exist, so only the target node is supervised.
- Blocker-mask F1 compares GRN's obstruction explanation (objects obstructing T's reachable views) with the dataset's blocker mask (union of
  the minimal clearing sets); the two are related, not identical, and the PIGINet column is computed differently (see section 1).
- Not run: the paper's planner-in-the-loop experiment, and the paper's own numbers and baselines (the paper was not readable).

## Files
| file | what |
|---|---|
| `grn_model.py` | the network (IK, GO and AGF modules, GRN) with the number of views as a parameter, and the batching |
| `base_gnn.py`, `LICENSE_GRN_MIT` | the authors' attention layer, unchanged (MIT) |
| `build_graphs.py` | dataset roots -> graphs and labels (`data/graphs.pkl`, `data/roots.pkl`) |
| `train_grn.py` | staged training (one run = one seed) |
| `evaluate_grn.py` | all the metrics above (`results/grn.md`, `results/grn.json`) |
| `split_v4.json` | the V4 split |
| `runs/s0, s1, s2` | trained weights (`best.pt`, 9 MB each), training logs, configs |
| `results/` | `grn.md` / `grn.json` |

## Run
```bash
export DATASET66=<folder with raw/, raw_none/, raw_k4_ordered/>   # default: ../dataset_bd_classquota_without_home_position_no_overpass (branch yadu/dataset-3d)
export V4_NEW_DIR=<naren/dataset-done: final_dataset_generation/FINAL_3D_DATASET_V4/roots_new>   # the 45 new K3 layouts
python build_graphs.py                                  # data/graphs.pkl, data/roots.pkl
python train_grn.py --name s0 --seed 0                  # ~35 min on 8 CPU threads (20 s / epoch)
python evaluate_grn.py s0 s1 s2 --tag grn               # results/grn.{md,json}  (needs DATASET66 as well)
```
Environment: Python 3.12, CPU torch, **torch_geometric 2.5.2** (the authors' version; the official `EdgeGATv2Conv` does not run on newer
releases), scikit-learn, numpy. Trained on 17-18 Nov 2026.
