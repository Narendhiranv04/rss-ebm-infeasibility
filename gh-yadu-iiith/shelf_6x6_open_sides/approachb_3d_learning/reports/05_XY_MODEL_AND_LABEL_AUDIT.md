# 05: Existing joint X/Y Boltzmann model, the 3D adapter, and the 3D supervision gap

**Verdict: `XY_BLOCKED_MISSING_SEQUENCE_LABELS`.**
- The existing model was found, and it is the finalized scene-first, root-consistent formulation.
- A 3D adapter is implemented. Its exact mathematics and gradients are verified, and on a SYNTHETIC GPU fixture it recovers planted coefficients.
- No genuine 3D X label exists. No real 3D X/Y training was run, and no labels were invented.

## 1. The existing model

The audit was read-only, using `git show`; nothing in that repository was modified.

**Location:**
- Repo `latent-causal-tamp`, branch `approach-b-xy-v1`, head `e113c6680` (phase P25).
- Also contained in `approach-b-universal-search` `45b0fb05f`.

**Exact math:** `bxy_core.py`
- `log_py` l.28, `log_px` l.32, `log_evidence` l.37, `marginals` l.51, `energy_root` / `brute_force` l.59/68; τ = 1.

**Production implementation:**
- `bxy_p11.py` holds the vectorized `log_evidence` (l.280–297), with a −inf mask for observed Y.
- `batch_loss` (l.312) is the root-mean NLL plus λ_J · Huber(log1p Ĵ, log1p J), δ = 1.
- The final arms are in `bxy_p17.py`: `ARMS` l.49, `Model` l.95, `sym` l.133, `batch_loss` l.154 adding κ·L_SYM.

**Coefficients:**
- `b_s` = scene MLP of the edited endpoint scene (**F3**).
- `a_r` = predecessor-aware ordered-prefix unary (**F2**).
- `w_r` = the single quadratic X–Y head (**F5**).
- Ĵ = travel readout.
- GEO2 adds geometry terms.

**Agreement with the methodology:**
- `final-fixed-methodology-1.pdf` §10.2–10.3 and §11.2–11.3 confirm the formulation, including `L_B = L_joint + λ_J L_J`.
- The PDF writes `a_r = h_X(g_r, v_s, c)`, which is scene-aware. The 2D code's `a_r` sees the final-prefix scene vector; the SA variant adds `xs(g, v_s, c)`.
- GEO2 and κ·L_SYM are recorded extensions (addendum 0129A, `METHODOLOGY_FINAL_IMPLEMENTED.md` §3).

**Inputs (all 2D):**
- 14-d per-object numerics plus a DINO feature;
- a 16-d context (3R robot `q`, goal);
- `edit_proj`;
- a GEO2 8×8 lattice with 48 modes;
- the F1 analytic screen.

**Training:** AdamW 1e-3, weight decay 1e-4, 12 roots per batch, ≤3000 updates, float64.

**Tests:** 744 passed at P25.

**Checkpoints:**
- 60 frozen D/E/DIRECT/G members at `/home/naren/bxy_checkpoint_backup/p23_1_work/stage/p21_runs/*/best.pt` (256 MB), also on Google Drive.
- 90 P21 runs, gitignored, in the worktree.
- Older arms in git under `runs/approach_b/bxy/ckpt/`.

**Status (P25): `APPROACH_B_2D_CLOSED_PRIMARY_NOT_ESTABLISHED_ORDER_SUPPORTED`** (P24, 237 accepted roots):

| Contrast | Effect | Result |
|---|---|---|
| D − DIRECT | +0.041 | not established |
| E − DIRECT | +0.052 | not established |
| D − E | −0.010 | F5 coupling unsupported |
| Order (O2) | +0.173 [0.058, 0.308] | supported |

## 2. 2D label semantics (the reference for what 3D needs)

**X (`bxy_teacher.py`):**
- an ordered prefix from S0 / q0, with the robot state carried forward;
- one terminal connector to q_ref;
- an independent replay.

**Y:**
- the original target from q_ref in the edited endpoint scene, run regardless of X.

**Handling:**
- Censored, ineligible and unattempted queries are absent, never 0.
- Same-endpoint orders share one Y node; distinct ordered prefixes have distinct X keys.
- AB/BA groups are evaluator-only.

## 3. What 3D supervision exists

Inventory over the 1,582 QC-eligible roots of the frozen v4 source pin: `reports/xy3d_label_inventory.json`.

| Label family | Train | Val | Test | What it is | Valid as X? | Valid as Y? |
|---|---|---|---|---|---|---|
| `original` | 1,120 | 246 | 216 | target-only attempt from the reset ready pose in the root scene | no | **yes** (root endpoint) |
| `D` | 3,849 | 801 | 780 | `SCENE_TARGET_ONLY`: target from the ready pose in a teleported edited scene. Edits move subsets to their single designated parking cell | no | **yes**, with a caveat: edit distribution limited to parking edits |
| `B` | 3,767 | 804 | 746 | chained no-reset sequences with the target **appended from the achieved state**. No connector to q_ref; roles minimal / noop / redundant / wrong_destination / wrong_order | **no** (no terminal connector; composite outcome) | **no** |
| `blocker` | 1,120 | 246 | 216 | blocker masks / minimal sets | privileged construction metadata, never a label | no |
| v4 `V_place` | 59,972 queries | | | conditional-on-grasp individual placement | **no** | **no** |

**3D X:**
- X = 1: **none**.
- X = 0: derivable only from integer `B.failed_step`, as in the old 2D V1 precedent, with no X = 1 counterpart. That one-sided view is not a valid X training set.

**3D Y:**
- 7,012 target-only attempts (`original` + `D`) are semantically Y.

**Privileged fields, never used as labels or features:**
- `roles`, `intended_blockers`, `construction`, `verified_K`, `blocker`.

## 4. The 3D adapter (`src/xy3d.py`)

**Math:** the probability model is ported with unchanged semantics: `log_py`, `log_px`, `log_evidence`, `marginals`, `energy_root`, `brute_force`. A vectorized `batched_log_evidence` is added for training.

**Neural coefficients:**
- `b_s = h_Y(v_s)` from the shared 3D `SceneEncoder` of the endpoint scene (**F3**).
- `a_r = h_X(g_r, v_s)`: scene-aware, as in the PDF (**F2**).
- `w_r = h_XY(g_r, v_s)` (**F5**).
- Ĵ = a travel readout. The Huber auxiliary applies on complete witnesses only.
- `g_r` is a GRU over ordered action tokens: `[moved object's 3D features, destination cell centre in metres, destination cell, step index]`.

**Arms:**

| Arm | What changes |
|---|---|
| `COUPLED` | w learned |
| `UNCOUPLED` | w ≡ 0 |
| `DIRECT` | a single success logit on complete (x, y) pairs; a labelled ablation |
| `ORDER_INVARIANT` | mean-pooled action **set**, step index dropped |

AB/BA same-endpoint contrasts come from the parent map.

**Selftest results (`reports/xy3d_selftest_cpu.json`):**
- Exact evidence vs brute-force enumeration through `E_root`, over 60 random roots with missing X and Y: max error **1.8e-15**, for both the scalar and the vectorized version.
- The joint over every (x, y) assignment sums to 1: log Z = **0.0**.
- `p_succ = σ(b)·σ(a + w)` to machine precision.
- There is exactly one Y node per endpoint scene. Reordered same-endpoint sequences get **different** `a_r` under the GRU arms, and **identical** `a_r` (< 1e-5) under `ORDER_INVARIANT`.
- `UNCOUPLED` has `w ≡ 0`.
- Every head receives a finite, non-zero gradient.

**Synthetic GPU proof (`reports/xy3d_synthetic_gpu.json`):** these are **SYNTHETIC FIXTURES with planted coefficients, not 3D labels**.
- Data: 3,000 training and 1,000 validation synthetic roots. Planted F3 depends on scene content, planted F2 on order, and planted w = 1.
- Early stopping on the synthetic validation split; the best checkpoint is restored. The run used the guard, its watchdog, CPU 7 and GPU 0, with a 207 MB peak.
- Planted-truth validation NLL: **1.947**.

| Arm | Validation NLL | Correlation with planted b | Correlation with planted a | Mean w |
|---|---|---|---|---|
| COUPLED | **2.008** | 0.94 | 0.95 | 0.85 (planted 1.0) |
| UNCOUPLED | 2.042 | 0.93 | 0.94 | 0 |
| ORDER_INVARIANT | 2.184 | 0.95 | **0.65** (cannot represent order) | 0 |

- The first attempt, without early stopping (400 roots, 1,500 full-batch steps), **overfit**: validation NLL rose to 54.7 and w diverged. It is kept as `xy3d_synthetic_gpu_attempt1_overfit.json` and was superseded by the protocol above.

## 5. What is missing before real 3D X/Y training

**A 3D X teacher query** (one per ordered correction `r` from the original root):
1. Relocate in order, from S0 / q0, with the robot state carried forward (pick, transfer, place for each step).
2. Run **one terminal connector to q_ref** (the 3D `q_ready`).
3. Replay independently.
4. Record CENSORED / INELIGIBLE as absent.
5. Store complete witnesses with joint travel.

**Y:** reuse the 3D target-only attempt from q_ref in the edited endpoint scene (`D` / `original` semantics). For new endpoint scenes reached by the new X sequences, run it with the same protocol, regardless of X.

**Design:**
- Every order of the same move set reaching the same endpoint must be queried, so AB/BA contrasts exist.
- Include no-op and redundant controls as in 2D.
- Use the inherited lineage split.

**Where:** this is CPU-heavy physics with IK and planning. It belongs on an authorized physics host, **not** on this training server. Nothing was generated here.

**When those labels exist:** run a tiny guarded GPU proof (`COUPLED` vs `UNCOUPLED` vs `DIRECT` vs `ORDER_INVARIANT`) with held-out roots, using `xy3d.XY3D` and `batched_log_evidence` unchanged.

**Optional, not done here:** pretraining an F3 Y-only scene head on the 7,012 authentic 3D Y attempts. It would **not** be an X/Y result, and its edit distribution is limited to parking edits.
