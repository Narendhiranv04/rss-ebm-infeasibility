# SceneCollisionNet on the 6x6 shelf dataset

Danielczuk, Mousavian, Eppner, Fox, "Object rearrangement using learned implicit collision functions", ICRA 2021 (code: github.com/NVlabs/SceneCollisionNet).
The network is NVlabs' architecture as ported and layer-by-layer verified by Sajid Ansari (arXiv 2611.09130, `scn.py` on the lab server), made
batch-capable here; layers, sizes, initialisation (kaiming, no batch norm) and the PointNet++ semantics are unchanged (`scn_model.py`).

**What SceneCollisionNet does.** Given a scene point cloud and a rigid query body (a point cloud) at a pose, it predicts whether the body
collides with the scene: the scene is voxelised and encoded once (per-point MLP, scatter-max into a voxel grid, 3D conv pyramid), the body is encoded
with PointNet++, and a classifier reads the scene feature at the body's voxel, the body feature, the offset in the voxel and the rotation.

**What we ask of it here** (an adaptation): *can the target T reach its goal in this layout?* The scene cloud is built from the teacher's own geometry
(floor board, back wall, the virtual arm ceiling, every object as a cylinder / box). The query body is the xArm7 hand and fingers (1,024 points of the
teacher's hand geoms, fingers pre-shaped), placed at each of the target's 32 grasp poses (16 azimuths x 2 grasp levels) **over the target** ("pick")
and, with the same orientation, **over the goal** ("place"). A grasp works with probability (1 - p_pick)(1 - p_place); a layout's score is its best grasp.
Everything is on the V4 split of the dataset (`split_v4.json`, from `naren/dataset-done`): 1,766 train / 364 val / 370 test scenes, the 370 V4 test scenes. Mean ± standard deviation over 3 seeds.

## Results (V4 test, 370 scenes; mean ± std over 3 seeds)

**A. Can the target reach its goal in this layout?** (all labelled test queries, score = best grasp)

| queries | n | feasible | AUROC | AP | acc @ 0.5 | F1 @ 0.5 |
|---|---|---|---|---|---|---|
| original + D (labels stored in the scene files) | 2115 | 0.229 | 0.978 ± 0.001 | 0.942 ± 0.002 | 0.935 ± 0.005 | 0.849 ± 0.015 |
| original (move T, scene as it is) | 370 | 0.181 | 0.968 ± 0.003 | 0.890 ± 0.015 | 0.922 ± 0.010 | 0.753 ± 0.040 |
| D (edited scene, obstacles parked) | 1745 | 0.240 | 0.979 ± 0.001 | 0.949 ± 0.001 | 0.938 ± 0.005 | 0.863 ± 0.012 |
| L (rest of the teacher's T-only lattice) | 5264 | 0.038 | 0.964 ± 0.004 | 0.743 ± 0.006 | 0.972 ± 0.002 | 0.676 ± 0.012 |
| all | 7379 | 0.093 | 0.974 ± 0.001 | 0.869 ± 0.003 | 0.961 ± 0.001 | 0.792 ± 0.008 |

**B. Per grasp**, the network's own question (labels from the teacher's stored per-grasp reasons):

| per-grasp metric (original + D test queries with labels) | value | n labelled grasps |
|---|---|---|
| grasp feasibility, AUROC | 0.985 ± 0.001 | 53962 |
| pick collision, AUROC (the hand at the grasp pose is unusable) | 0.986 ± 0.001 | 52533 |
| place collision, AUROC (the hand over the goal collides) | 0.947 ± 0.002 | 1736 |
| the teacher's winning grasp is among the 3 best-scored | 0.669 ± 0.024 | – |
| the teacher's winning grasp is the best-scored | 0.409 ± 0.030 | – |

**C. Scene level** (layouts enumerated: the original scene and every set of 1-3 obstacles parked at their first parking cell)

| metric | tau = 0.5 | tau* (validation) |
|---|---|---|
| class accuracy (NONE, K0-K3) | 0.756 ± 0.003 | 0.764 ± 0.017 |
| class macro-F1 | 0.750 ± 0.006 | 0.759 ± 0.014 |
| recall K0 | 0.667 ± 0.051 | 0.687 ± 0.012 |
| recall K1 | 0.635 ± 0.032 | 0.648 ± 0.009 |
| recall K2 | 0.702 ± 0.024 | 0.705 ± 0.033 |
| recall K3 | 0.804 ± 0.021 | 0.807 ± 0.036 |
| recall NONE | 0.952 ± 0.035 | 0.957 ± 0.020 |
| tau* (chosen on validation), per seed | – | 0.7, 0.65, 0.2 |

| scene-level metric (26 layouts per scene on average) | value |
|---|---|
| solvable (K0-K3) vs NONE, AUROC (best layout score) | 0.996 ± 0.002 |
| blocker set given the true K, top-1 (n = 241 K1-K3 scenes) | 0.924 ± 0.017 |
| blocker set given the true K, top-3 | 0.986 ± 0.005 |
| blocker mask by counterfactual gain, F1 (n = 264 scenes, delta tuned on validation) | 0.250 ± 0.013 |
|   precision | 0.824 ± 0.010 |
|   recall | 0.147 ± 0.009 |
|   exact match | 0.119 ± 0.005 |
|   delta chosen, per seed | 0.01, 0.01, 0.01 |

Confusion matrix of the class prediction at tau = 0.5, mean counts over the 3 seeds (370 scenes):

| true \ predicted | K0 | K1 | K2 | K3 | NONE |
|---|---|---|---|---|---|
| K0 | 44.7 | 16.7 | 4.7 | 0.3 | 0.7 |
| K1 | 7.0 | 33.7 | 11.0 | 0.0 | 1.3 |
| K2 | 0.3 | 16.3 | 60.3 | 6.3 | 2.7 |
| K3 | 0.3 | 2.0 | 15.3 | 82.0 | 2.3 |
| NONE | 0.0 | 0.0 | 0.3 | 2.7 | 59.0 |

**Per seed**

| run | best epoch | val AUROC | test AUROC, all | test AUROC, original + D | test AUROC, L | class acc, tau 0.5 | solvable vs NONE | blocker set top-1 | blocker-mask F1 |
|---|---|---|---|---|---|---|---|---|---|
| adam_s0 | 23 | 0.9722 | 0.973 | 0.978 | 0.959 | 0.751 | 0.994 | 0.934 | 0.264 |
| adam_s1 | 23 | 0.9717 | 0.974 | 0.977 | 0.966 | 0.759 | 0.998 | 0.900 | 0.253 |
| adam_s2 | 27 | 0.9734 | 0.976 | 0.978 | 0.968 | 0.757 | 0.997 | 0.938 | 0.233 |

**Optimizer** (the authors train with SGD; Adam learned faster and reached a better validation AUROC):

| optimizer | AUROC, original + D | AUROC, L | class acc, tau 0.5 | solvable vs NONE | blocker set top-1 |
|---|---|---|---|---|---|
| Adam, lr 1e-4 (used; 3 seeds) | 0.978 ± 0.001 | 0.964 ± 0.004 | 0.756 ± 0.003 | 0.996 ± 0.002 | 0.924 ± 0.017 |
| SGD, lr 1e-3, momentum 0.9 (the authors' optimizer; 1 seed, best val AUROC 0.9685) | 0.976 | 0.954 | 0.751 | 0.992 | 0.921 |

## Adaptation to the shelf
- **Grid.** Bounds x 0.10-0.80, y -0.40-0.40, z 0.50-0.90 m, voxel grid [8, 16, 4] (8.75 x 5 x 10 cm); 8,192 scene points per layout (re-sampled every epoch).
- **Labels per grasp** (`build_scn_data.py`, from the teacher's stored per-grasp reasons; 32 grasps ordered as in `shelf_scene.grasp_poses`): "out of reach", "would hit ...",
  "overlap a neighbour at the start" -> the pick pose is unusable; "overlap a neighbour at the goal" -> the pick pose is fine, the place pose collides; a feasible query stores its
  winning grasp (works), earlier grasps failed for an unknown reason; an infeasible lattice query (L) means every grasp failed. Only 0.2% of the grasps are feasible.
- **Loss** (NVlabs' trainer: BCE, average of the top-10% hardest elements and a random 10%, 0.5 / 0.5): on the combined "grasp works" label, with the few feasible grasps averaged
  separately, plus the decomposed pick / place labels where the reason fixes them (weight 0.25 each), and -log max P for feasible L states whose winner is unknown.
- **Training.** 26.1 M parameters, from scratch, Adam lr 1e-4, batch 16 states, 30 epochs (about 38 s per epoch on one GPU, 35% of the infeasible lattice states used per epoch), best
  validation AUROC kept.
- **Geometry check** (`check_geometry.py`): at the grasps where the teacher says the hand would hit a named object, our hand cloud placed at the grasp pose penetrates that object in 73.6%
  of cases (the rest are arm links or the approach path, which the hand alone cannot reproduce); for grasps that fail for another reason it penetrates some obstacle in 11.0%.

## Limits
- The query body is the hand only: collisions of the arm links (and of the carried object along the path) are not seen by the network, only learned through the layout.
- Parking cells are given to the candidates; the network does not choose placements.
- It has no obstruction output, so its blocker mask comes from the score gain when one obstacle is parked alone; that rarely fixes scenes with several blockers (recall 0.15).
- Trained weights are not in this repository: each is 105 MB, above GitHub's 100 MB file limit (retraining takes about 20 minutes on one GPU).
- Not run: the original paper's own experiments and baselines.

## Files
| file | what |
|---|---|
| `scn_model.py` | the network (scene encoder, hand encoder, classifier), batching, the losses |
| `scn_geometry.py` | scene cloud, gripper cloud (from the teacher's hand geoms) and the 32 grasp poses |
| `build_scn_data.py` | dataset roots -> states and per-grasp labels (`data/states.pkl`, `data/roots_scn.pkl`) |
| `scn_data.py`, `train_scn.py` | data set and training (one run = one seed) |
| `evaluate_scn.py` | all the metrics above (`results/scn.md`, `results/scn.json`) |
| `check_geometry.py` | the frame / scale check of the hand cloud against the teacher's collision reasons |
| `split_v4.json`, `data/gripper_cloud_1024.npz` | the V4 split; the cached hand cloud |
| `results/` | `scn.md` / `scn.json` (3 seeds), `scn_sgd.md` / `scn_sgd.json` (optimizer ablation) |
| `runs/*/` | training logs and configs of the 4 runs |

## Run
```bash
export DATASET66=<folder with raw/, raw_none/, raw_k4_ordered/>   # default: ../dataset_bd_classquota_without_home_position_no_overpass (branch yadu/dataset-3d)
export V4_NEW_DIR=<naren/dataset-done: final_dataset_generation/FINAL_3D_DATASET_V4/roots_new>   # the 45 new K3 layouts
export SHELF66_V3_DIR=<.../code/shelf66_v3> XARM7_XML=<mujoco_menagerie>/ufactory_xarm7/xarm7.xml   # grasp poses (and the hand cloud, if not cached)
python build_scn_data.py                                       # data/states.pkl, data/roots_scn.pkl
python train_scn.py --name adam_s0 --seed 0 --opt adam --lr 1e-4 --epochs 30     # one GPU, ~20 min
python evaluate_scn.py adam_s0 adam_s1 adam_s2 --tag scn       # results/scn.{md,json}
```
Environment: Python 3.12, PyTorch (CUDA), numpy, scikit-learn, MuJoCo 3.16.0 (grasp poses / hand cloud only). Trained on 18-19 Nov 2026.
