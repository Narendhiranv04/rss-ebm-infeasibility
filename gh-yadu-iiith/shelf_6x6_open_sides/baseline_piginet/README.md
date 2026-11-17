# PIGINet plan-feasibility predictor on the 6x6 shelf dataset

PIGINet (Yang, Garrett, Lozano-Pérez, Kaelbling, Fox, "Sequence-Based Plan Feasibility Prediction for Efficient Task and Motion
Planning", RSS 2023, arXiv 2211.01576) as a baseline for the shelf dataset of `../dataset_bd_classquota_without_home_position_no_overpass/`.

**What it does.** Given a scene, a goal and a candidate task plan (a list of "move object X to cell C" steps that ends with the
target's move), it outputs one number: the probability that the plan can be executed. It is trained only on that yes/no label. It does
not propose plans or placements, so this folder contains the predictor and its feasibility evaluation only. Train / validation / test =
Narendhiran's V4 split (`split_v4.json`, from `naren/dataset-done`): 1,766 / 364 / 370 scenes.

## Results (V4 test, 370 scenes; mean ± std over 3 seeds)

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
