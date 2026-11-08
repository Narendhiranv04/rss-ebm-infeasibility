# 03: 3D placement sampler implementation (cVAE vs conditional flow matching)

**What this file covers:** the code in `src/`, the equations it implements, the data each component uses, and how the comparison is kept fair.

**Scope.** The sampler proposes destinations `d` for an object `o` that is already held, in a hypothetical scene `S_hyp`.
- Its label `V_place` is placement conditional on a grasp, scored by the pinned MuJoCo teacher.
- It says nothing about pick accessibility, action order, robot-state carry-over, the terminal connector, or whether the original target is restored. Those belong to X/Y; see `05`.

## Files

| File | Role |
|---|---|
| `src/data3d.py` | Builds the immutable snapshot (verified file by file against the gvlab index and the source pin) and the label-blind features. Joins are by `root_id` / `qid` / `witness_id`. Also writes the dataset audit and provides the whole-snapshot GPU tensor loader. |
| `src/models.py` | The shared scene encoder, the cVAE, flow matching, the uniform and band-stratified controls, and a separate ranker. |
| `src/train.py` | `smoke` / `fit` / `evaluate` / `freeze` commands, the fairness procedure and the metrics. Every GPU command runs under the guard (`--dry-run` hides CUDA before torch is imported). |
| `src/guard.py` | The resource guard and an independent watchdog process (see `01`). |
| `src/figs.py` | Figures, drawn only from real runs. |
| `src/xy3d.py` | The X/Y adapter (see `05`). |
| `tests/test_core.py` | 9 targeted invariants. |

## Inputs: what is used, and what is never used

**Per object (14 values, from the label-blind root fields in `q2roster3d.allowed`):**
- shape one-hot (box / cylinder);
- size and height;
- cell `(r/5, c/5)`;
- cell centre in metres, `x = 0.42 + (r-2.5)*0.075`, `y = (c-2.5)*0.075` (the xArm base is at the origin);
- distance and bearing from the base;
- an "is the held object" flag;
- offset and Chebyshev distance to the held object.

**Per scene:** the 6×6 occupancy grid.

**Normalization:** size, height, x, y and distance are standardized with TRAIN statistics only. The scalers are stored in `<snapshot>_derived/scalers.json`.

**Never inputs:**
- K. It is stored for reporting strata only.
- The goal cell, roles and target identity. Object ids are used only for ordering; the held object goes first and the rest are sorted.
- Blockers, repairs, construction spec, teacher outcomes and witnesses.

**Not available in the source, so not invented:**
- robot joint states per query;
- object orientation beyond the upright resting convention;
- any image or learned scene embedding.

The robot is the same pinned xArm7 protocol (`XARM7_SHELF66_LEVELGRASP_V1`) for every query, so it carries no per-query signal.

## Destination representation (shared by every sampler)

- A destination is a support-constrained **upright** placement at one of the 36 shelf cell centres, resting on the shelf floor.
- `legal(S_hyp, o)` = in-bounds cells that are free in the hypothetical scene. The held object's own cell is occupied, so it is never legal.
- Every recorded physical destination passes this check (asserted at featurization).
- No arbitrary SE(3) pose is generated. Grasp azimuth and level belong to the teacher.

## Shared encoder

- Each object gets an MLP embedding. The held object's token (always slot 0) keys a single-head attention over all objects.
- The encoder concatenates `[held token, attention context, masked mean, occupancy-grid MLP]` and projects it to `h = 128`.
- It is permutation-invariant over the non-held objects; `test_encoder_permutation_invariant_over_non_query_objects` checks this.
- **Each sampler owns its own encoder** with identical architecture and capacity.
- The cVAE's posterior encoder `q_psi` is a separate head and is distinct from this feature encoder.

## A. cVAE

- **Posterior:** `q_psi(z | S, o, d) = N(mu, diag sigma^2)` from `[h(S,o), onehot(d)]`, with `z ∈ R^8`.
- **Decoder:** `p_phi(d | z, S, o)` is a masked categorical over the legal cells. Illegal logits are set to −1e9.
- **Loss:** `L = CE(p_phi(d | z, S, o), d) + beta_KL · KL(q_psi || N(0, I))`. `beta_KL` warms up linearly from 0.1 to 1.0 over 10 epochs. Validation always uses `beta = 1`, so epochs are comparable.
- **Multiple valid destinations per context:** every VERIFIED destination of a context is its own training example, so the decoder learns a multimodal distribution.
- **Sampling:** draw `z ~ N(0, I)`, then a Gumbel-max categorical draw.
- **Held-out density:** an IWAE estimate (K = 64) of `log p(d | S, o)`.
- **Caveat:** this density is a proposal distribution, **not** a calibrated feasibility probability.

## B. Conditional flow matching

- **Coordinate:** `u(d) = 2·((r + 0.5 + j_r, c + 0.5 + j_c)/6) − 1 ∈ [−1, 1]^2`, with `j ~ U(−0.5, 0.5)^2`. The within-cell jitter dequantizes the 36 cells into a continuous density.
- **Interpolant:** `x_0 ~ N(0, I_2)`, `x_t = (1 − t) x_0 + t x_1`, `u_t = x_1 − x_0`.
- **Loss:** `L_FM = E || v_phi(x_t, t | S, o) − u_t ||^2`. Time uses a 32-dim sinusoidal embedding. The velocity network is a 3-layer MLP on `[x_t, emb(t), h]`.
- **Sampling:** 32 Euler steps, then a deterministic projection to the **nearest legal cell centre**.
- **Recorded per draw:** the out-of-domain rate (`|x| > 1`) and the illegal-cell rate before projection. Duplicates after projection count against the same raw budget.
- **Caveat:** this is a distributional interpolation in support coordinates, **not** a robot motion.

## C. Non-learned controls and a separate ranker

- **`uniform`:** exactly legality-matched uniform sampling over legal cells.
- **`stratified`:** round robin over Chebyshev bands from the source cell (near 1 / mid 2–3 / far 4–5, the roster's 3D bands), uniform inside each band.
- **Swept-volume sampler:** `NOT_AVAILABLE`. None is implemented or finalized in this repository, so none was invented.
- **`ranker`:** a discriminative feasibility classifier on declared labelled trials (VERIFIED vs NEG_*; UNLABELED excluded), proposing a deterministic top-M.
  - Its sigmoid scores are reported with Brier / ECE / AUROC on the evaluation split's labelled rows.
  - They are kept separate from the generators' densities.

## Fairness procedure (identical for cVAE, FM, uniform and stratified)

- All samplers use the same snapshot, the same legality mask and the same evaluation contexts: every context of the split with at least one labelled destination.
- Each draws `R = 8M` raw samples **with replacement** and keeps the **first M unique** cells.
- Reported: the unique count, the duplicate rate (`1 − unique / raw used`), and the illegal / out-of-domain rates before projection.
- `M ∈ {1, 2, 4, 8}`.
- Seeds: 3 training seeds × 3 evaluation seeds per learned family, and 3 evaluation seeds for each control.

## Metrics

All metrics are computed on known labels only. **An untested cell is UNKNOWN, never negative.**

| Metric | Definition |
|---|---|
| `hit_at_M` | Among contexts with ≥1 recorded VERIFIED destination: the fraction where some proposal is one of them. |
| `pos_recall` | Recorded positives recovered ÷ recorded positives. |
| `known_label_precision` | Proposals on recorded positives ÷ proposals on any labelled destination. This is the empirical placement success among proposals whose outcome is independently known. |
| `labeled_support` | The fraction of proposals that land on a labelled destination. |
| `known_negative_rate` | The fraction of proposals that land on a recorded negative. |
| Diversity | Mean pairwise cell distance among the unique proposals. |
| Held-out positive NLL | cVAE (IWAE) vs uniform (`log |legal|`). For FM this is `NOT_COMPUTED` (no exact likelihood without an ODE trace estimator). |
| Latency | GPU milliseconds per context. |
| Stratification | At M = 4, by shape, height bin, Q2 block and object count. |

## Training protocol (frozen in `configs/sampler_v1.json`)

- AdamW, lr 1e-3, weight decay 1e-4, batch 512, gradient-norm clip 5.
- At most 60 epochs, early stopping with patience 8 on the validation loss, a 20-minute walltime cap per run.
- Generators train on TRAIN VERIFIED rows only. The ranker trains on TRAIN labelled rows only.
- Checkpoint selection is by best val loss. `freeze` records every checkpoint's sha256 and the config hash before any TEST evaluation. The TEST command refuses to run if the selection is unfrozen or has changed.

## Tests (`tests/test_core.py`, 9/9 pass on the server)

1. The encoder is permutation-invariant over the non-held objects.
2. A scene edit updates features, occupancy and legality, and the held object's cell is never legal.
3. Every sampler (cVAE, FM, uniform, stratified) returns only legal cells.
4. The first-M-unique accounting is correct.
5. The FM coordinate round-trips exactly to the cell centres.
6. Split isolation: generators see only TRAIN VERIFIED rows, and the ranker only TRAIN labelled rows (UNLABELED excluded).
7. Guard queue detection works, and the guard never signals a foreign PID (argv must be our venv python plus our script).
8. The dry run hides CUDA before torch is imported.
9. The exact X/Y math passes (see `05`).
