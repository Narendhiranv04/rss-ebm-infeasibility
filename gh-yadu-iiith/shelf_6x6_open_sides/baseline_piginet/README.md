# PIGINet plan-feasibility predictor on the 6x6 shelf dataset

PIGINet (Yang, Garrett, Lozano-Pérez, Kaelbling, Fox, "Sequence-Based Plan Feasibility Prediction for Efficient Task and Motion
Planning", RSS 2023, arXiv 2211.01576) as a baseline for the shelf dataset of `../dataset_bd_classquota_without_home_position_no_overpass/`.

**What it does.** Given a scene, a goal and a candidate task plan (a list of "move object X to cell C" steps that ends with the
target's move), it outputs one number: the probability that the plan can be executed. It is trained only on that yes/no label. It does
not propose plans or placements, so this folder contains the predictor and its feasibility evaluation only. Train / validation / test =
Narendhiran's V4 split (`split_v4.json`, from `naren/dataset-done`): 1,766 / 364 / 370 scenes.

<!-- all-models-results:start -->
## Results of all models (V4 split, the same 370 test scenes)

Models and where they live (all in this repository, folder `gh-yadu-iiith/shelf_6x6_open_sides/baseline_<name>/`): REAP (`yadu/REAP`),
PIGINet (`yadu/PIGINet`), GRN (`yadu/GRN`). Split: `split_v4.json` (Narendhiran's V4 split from `naren/dataset-done`: 1,766 train / 364
val / 370 test scenes). Mean ± standard deviation over seeds.

### Feasibility predictors, one metric set

"T move" columns: can the target reach its goal in that state (AUROC over the stored test queries of that kind: the original scene, the
teacher's edited scenes with obstacles parked, and the rest of the teacher's T-only lattice, L). Scene-level columns enumerate
candidate states / plans (the original scene and every set of 1-3 obstacles sent to their first parking cell, 26 per scene on average);
PIGINet has two input forms (relocations as plan tokens, or the obstacles already parked in the state, which is what GRN sees).
`class acc`: NONE / K0-K3 (chance 0.28), `tau*` chosen on validation. `blocker set top-1`: the best-scoring candidate of the true size K is a
verified minimal set. `blocker-mask F1`: PIGINet from its best-scoring set at tau 0.5, GRN from its own obstruction head (tau_go tuned on
validation). The PIGINet "no pose / size values" rows are one seed.

| model | seeds | T move, original scene (AUROC) | T move, edited scene (AUROC) | T move, lattice L (AUROC) | solvable vs NONE (AUROC) | class acc, tau 0.5 | class acc, tau* | blocker set top-1 (given K) | blocker-mask F1 |
|---|---|---|---|---|---|---|---|---|---|
| GRN | 3 | 0.958 ± 0.001 | 0.960 ± 0.000 | 0.881 ± 0.003 | 0.975 ± 0.002 | 0.574 ± 0.013 | 0.601 ± 0.011 | 0.788 ± 0.009 | 0.834 ± 0.003 |
| PIGINet, full (image + pose + text), obstacles parked in the state | 3 | 0.967 ± 0.006 | 0.980 ± 0.005 | 0.955 ± 0.007 | 0.998 ± 0.000 | 0.741 ± 0.030 | 0.753 ± 0.024 | 0.928 ± 0.017 | 0.829 ± 0.018 |
| PIGINet, full (image + pose + text), relocations as plan tokens | 3 | 0.967 ± 0.006 | 0.980 ± 0.005 | 0.955 ± 0.007 | 0.977 ± 0.019 | 0.728 ± 0.013 | 0.728 ± 0.011 | 0.902 ± 0.016 | 0.825 ± 0.011 |
| PIGINet, no image tokens, obstacles parked in the state | 3 | 0.987 ± 0.001 | 0.993 ± 0.001 | 0.984 ± 0.001 | 1.000 ± 0.000 | 0.844 ± 0.024 | 0.855 ± 0.010 | 0.963 ± 0.009 | 0.864 ± 0.011 |
| PIGINet, no image tokens, relocations as plan tokens | 3 | 0.987 ± 0.001 | 0.993 ± 0.001 | 0.984 ± 0.001 | 0.998 ± 0.001 | 0.773 ± 0.021 | 0.791 ± 0.003 | 0.899 ± 0.016 | 0.808 ± 0.009 |
| PIGINet, no pose / size values, obstacles parked in the state | 1 | 0.934 | 0.927 | 0.876 | 0.947 | 0.473 | 0.511 | 0.763 | 0.620 |
| PIGINet, no pose / size values, relocations as plan tokens | 1 | 0.934 | 0.927 | 0.876 | 0.972 | 0.481 | 0.595 | 0.884 | 0.689 |

### The paper's table (a) columns

AUC = scene solvable (K0-K3) vs NONE from the method's own score, computed with `metrics66.py` (REAP's script) for every row. Repair / Task /
Time / Calls need a planner and are therefore empty for the predictors that have none.

| method | AUC | Repair (%) | Task (%) | Time (s) | Calls |
|---|---|---|---|---|---|
| REAP, released weights, zero-shot (1 seed) | 0.984 | 55.0 | 73.0 | 168.1 | 46.0 |
| REAP, trained from scratch on 600 V4 train scenes (1 seed) | 0.789 | 82.6 | 63.2 | 208 | 39.7 |
| PIGINet + TAMP, full (3 seeds) | 0.978 ± 0.021 | 68.9 ± 4.5 | 98.1 ± 0.2 | 26.9 ± 4.2 | 3.4 ± 0.4 |
| PIGINet + TAMP, no image tokens (3 seeds) | 0.995 ± 0.004 | 69.1 ± 0.7 | 99.1 ± 0.1 | 24.4 ± 2.3 | 3.3 ± 0.1 |
| GRN (feasibility predictor only, 3 seeds) | 0.975 ± 0.002 | – | – | – | – |

- REAP is run with the dataset teacher as feasibility oracle and every plan is replayed in MuJoCo. The trained row uses a precomputed
  conflict library as the checker (7-11 s per teacher check made online training impossible; the library agrees with the teacher on about
  77% of random moves), three restarted segments (about 3,600 updates), so it is handicapped; the object-count-only baseline reaches 0.976
  AUC on this split, so AUC does not separate the methods well.
- PIGINet + TAMP: candidate plans are scored, those below 0.5 dropped and the rest refined in score order by the teacher (300 s limit per
  scene). PIGINet is given the scene file's parking cells as destinations, so its Task / Time / Calls are optimistic; Time was measured on a
  laptop with 10 workers, REAP's on servers.
- GRN and PIGINet (as predictors) are never asked to choose placements; parking cells are given to the candidates.
<!-- all-models-results:end -->

## PIGINet in detail (V4 test, 370 scenes; mean ± std over 3 seeds)

| | full PIGINet (image + pose + text) | no image tokens | no pose / size values (1 seed) |
|---|---|---|---|
| Plan feasible or not, AUROC (3,501 stored queries: original + D + B) | 0.955 ± 0.008 | **0.975 ± 0.002** | 0.906 |
| – original, "move T" (K0 vs the rest), AUROC | 0.967 ± 0.006 | **0.987 ± 0.001** | 0.934 |
| – D, edited scene (obstacles parked), AUROC | 0.980 ± 0.005 | **0.993 ± 0.001** | 0.927 |
| – B, executed sequences, AUROC / accuracy | 0.904 / 0.819 | **0.932 / 0.856** | 0.845 / 0.760 |
| Scene solvable (K0–K3) vs NONE, AUROC (best candidate score) | 0.977 ± 0.019 | **0.998 ± 0.001** | 0.972 |
| Class NONE / K0–K3, accuracy | 0.728 ± 0.013 | **0.773 ± 0.021** | 0.481 |
| Blocker-mask F1 | **0.825 ± 0.011** | 0.808 ± 0.009 | 0.689 |
| Top-ranked labelled plan is feasible | 0.889 ± 0.029 | 0.876 | 0.837 |

Full tables (per label family, per B role, per class, confusion matrices): `results/{full,noimg,noval}.{md,json}`. The class
accuracy chance level is 0.28 (always K3).

- The "full" model is PIGINet as published. The no-image variant is the same network without the CLIP image tokens; it is better
  on almost every row here, because pose, size, height and shape are already exact in the value token and 1,766 scenes are few.
- Class and blocker numbers use PIGINet the way the paper does, as a scorer over candidate plans: "move T" and every set of 1–3
  obstacles sent to their first parking cell (the teacher's convention); predicted K = smallest set size with a score ≥ 0.5, else NONE.
  The destinations are the scene file's parking cells, so these rows measure how well PIGINet ranks given plans, not whether it can
  choose placements.
- Not run: the paper's planner run-time experiment (it needs a motion planner in the loop; ours takes 2–20 s per check).

## Files

| file | what |
|---|---|
| `piginet.py` | the network and the token / sequence encoding |
| `build_data.py` | dataset roots → training samples (`data/samples.pkl`) |
| `render_feats.py` | MuJoCo renders of every scene state → per-object CLIP image features |
| `train.py` | training (one run = one seed; `--no-image`, `--no-value` ablations) |
| `evaluate.py` | feasibility evaluation, scene-level classes and blocker sets, plan ordering |
| `split_v4.json` | the V4 split (train / val / test root ids per class) |

## Method

**Network.** The authors never released the model (it is in their private `fastamp` module; kitchen-worlds has only the planner and
the data generation). The architecture is the one Sajid Ansari recovered from the authors' released checkpoints and strict-load
verified (`exact_pigi.py`, arXiv 2611.09130): a frozen CLIP ViT-B/32 image feature → 3-layer MLP; a CLIP text feature → linear; a
15-d value → linear; the elements of a token are added; learned goal / initial-state encodings; 4 CLIP-style residual attention blocks
(width 256, 2 heads, MLP 1024); the first position → linear → sigmoid; positive-weighted binary cross-entropy; AdamW, lr 3e-4,
60 epochs, the epoch with the best validation AUROC is kept. 4.2 M trainable parameters.

**Sequence** (paper Sec. IV): `[plan actions] + [goal] + [one token per object]`.
- plan token = one pick-and-place: phrase + image of the moved object + value(pose, size, height, shape) + value(destination),
  with a sinusoidal step encoding; Sajid's port had a single action, ours has multi-step plans;
- goal token = phrase + image of the target + value(goal cell);
- object token = phrase + image + value(pose, size, height, shape). PIGINet's value holds the pose only; ours also holds the exact
  size, height and shape, which can only help the baseline.

**Images.** Every scene state is rendered in MuJoCo with the dataset's own scene (one camera through the open side), each object is
cropped with the segmentation render, and the crop is embedded with CLIP (81,085 crops).

**Training samples** (69,383 train / 13,058 val / 14,725 test; all from the scene files, no new teacher runs):

| kind | what | label |
|---|---|---|
| original | "move T" on the scene as it is | `labels.original` |
| D | "move T" on the teacher's edited scenes (one obstacle parked, or the intended blockers) | `labels.D` |
| B | executed sequences (minimal, wrong_destination, wrong_order, redundant+1, noop_irrelevant, intended_union) | `labels.B` |
| L | the rest of the teacher's T-only lattice: every subset S of its object pool parked | 1 if S is a minimal clearing set, 0 if S contains none (supersets are not stored, so skipped) |
| P | the lattice negatives as plans "move each o in S to its parking cell, then T" | 0, implied (T cannot move after S is parked); not executed by the teacher |

Without L and P the model never sees "2 of the 3 blockers moved is not enough" and never predicts K3 or NONE.

## Run

```bash
export DATASET66=<folder with raw/, raw_none/, raw_k4_ordered/>      # default: ../dataset_bd_classquota_without_home_position_no_overpass
export V4_NEW_DIR=<naren/dataset-done: final_dataset_generation/FINAL_3D_DATASET_V4/roots_new>   # the 45 new K3 layouts
python build_data.py                                                  # data/samples.pkl
export SHELF66_V3_DIR=<.../code/shelf66_v3> XARM7_XML=<mujoco_menagerie>/ufactory_xarm7/xarm7.xml MUJOCO_GL=egl
python render_feats.py --workers 5                                    # data/feats/, data/text.npz (about 35 min on 5 CPU workers)
python train.py --name full_s0 --seed 0                               # also: --no-image, --no-value; uses a GPU if there is one
python evaluate.py full_s0 full_s1 full_s2 --tag full                 # results/full.{md,json}
```
`render_feats.py` needs the dataset's scene code (`shelf66_v3`, MuJoCo 3.16.0) and the CLIP weights `openai/clip-vit-base-patch32`.
The reported runs were trained on one RTX PRO 5000 Blackwell (15 s per epoch), 17 Nov 2026. Checkpoints are not included.
