# 09: BWL-1 sampler validation, reranking, mixture, categorical control and cVAE latent diagnostics

**Stage verdict: BWL-1 PASS (completed).**

| Verdict field | Value |
|---|---|
| `SAMPLER_OFFLINE` | **INCONCLUSIVE**, by the predeclared rule |
| `SAMPLER_NEW_PHYSICAL_PRECISION` | **NOT_RUN** (a physics-host manifest is prepared) |

**Status labels:**
- `RUN`: VAL (development) metrics; 3 new GPU fits.
- `ARCHIVED`: TEST, re-run byte-for-byte as evidence only.
- `PLANNING_ONLY`: the confirmation manifest.

**Pinned inputs:**
- Data: snapshot `v4full_82926152fa9cc345`, features `4a37b2ed…`.
- Code: the BWL commit on `naren/approachb-3d-learning` (`src/study.py` `48da982f…`, `src/models.py` `2f61fc89…`).
- Config: `configs/sampler_bwl1.json`, sha256 recorded in `results_bwl/study_val.json`.
- Raw outputs: `results_bwl/study_val.json`, `latent_val.json`, `REPRO_COMPARISON.json`.
- Figures: `figures/bwl1_*.png`.

## 1. Archived TEST, with the band-stratified control included

This is the exposed TEST set, reproduced from the frozen checkpoints only. TEST, M = 4, mean of 3 training seeds × 3 evaluation seeds; 1,385 contexts, 1,276 with a recorded positive. Every number here is identical in the reproduction.

| Sampler | hit@4 | Known-label precision @4 | Known-label fraction | Known-negative rate |
|---|---|---|---|---|
| **Band-stratified** | **0.688** | 0.758 | 0.294 | 0.071 |
| cVAE | 0.655 | 0.801 | 0.267 | 0.053 |
| Flow matching | 0.557 | 0.756 | 0.224 | 0.055 |
| Uniform | 0.451 | 0.713 | 0.183 | 0.053 |
| Ranker (top-4 of all legal cells) | 0.534 | 0.906 | **0.178** | 0.017 |

**Correction to the kickoff summary.** My earlier summary to you omitted the band-stratified row. The archived report `04` lists it.

**Band-stratified is ahead of the cVAE on hit@4** (0.688 vs 0.655). The cVAE has the higher known-label precision.

**Three limits on these numbers:**
- 70–82 % of every arm's proposals have **no recorded outcome** (UNKNOWN). Known-label precision is **not** a physically verified precision.
- The ranker's 0.906 is computed over only 17.8 % of its proposals. It is **not** an unconditional 90.6 % placement success rate.
- TEST was **not** used for anything below.

## 2. New VAL comparison (development roots only)

**Population.** VAL, 1,425 contexts with ≥ 1 recorded outcome (1,336 with a recorded VERIFIED cell), 225 independent roots.

**Arms and budgets** (identical legality mask, same object and scene inputs):
- Generators: 8M raw draws with replacement; the first M unique are kept.
- Rerank arms: the first P = f·M unique proposals out of 8P raw draws, re-ordered by the ranker, top M kept. f is 2 or 4, and is the same for the learned and analytical bases.
- `ranker_catalogue`: scores all legal cells. That catalogue (36 cells) is disclosed and charged as 36 "draws".
- Sobol: a scrambled Sobol set per evaluation seed, with a per-context Cranley–Patterson shift; an illegal draw snaps to the nearest legal cell.
- Mixture: each raw draw comes from the cVAE with probability λ, otherwise from the stratified control. λ ∈ {0.25, 0.5, 0.75}, predeclared.
- `cat`: a **conditional categorical** control. It has the cVAE's encoder and decoder width but no latent, and was trained the same way (3 seeds, `SAMPLER_BWL1/cat_s*`, about 29 s each on GPU 0).

**Seeds.**
- Trained arms: 3 training seeds × 3 evaluation seeds.
- Controls: 3 evaluation seeds.
- Training and evaluation seed pairs are matched, e.g. cvae_s*i* goes with ranker_s*i*.

**Uncertainty.** 95 % CIs come from a **root-cluster bootstrap**: 2,000 resamples of VAL roots, computed on per-context means over runs.

**Swept-volume sampler: `NOT_IMPLEMENTED / METHOD_UNSPECIFIED`.** Methodology §6.4 leaves it unspecified, and no audited implementation exists.

M = 4, VAL. Fractions are of all proposals: UNKNOWN = untested, cert = NEG_CERTIFIED_GEOMETRY, policy = NEG_UNDER_POLICY.

| Arm | hit@4 [95% CI] | Known-label precision @4 [95% CI] | Known fraction | Cert / policy / unknown | Raw draws / context | Raw illegal or snapped |
|---|---|---|---|---|---|---|
| cVAE | 0.673 [0.657, 0.690] | 0.819 [0.798, 0.839] | 0.275 | .030 / .020 / .725 | 4.4 | 0 |
| cat (no latent) | 0.652 [0.636, 0.666] | 0.813 [0.792, 0.833] | 0.268 | .032 / .019 / .732 | 4.4 | 0 |
| Flow matching | 0.563 [0.546, 0.578] | 0.784 [0.760, 0.807] | 0.223 | .031 / .017 / .777 | 4.3 | **0.242** |
| Uniform | 0.457 [0.441, 0.473] | 0.743 [0.719, 0.767] | 0.180 | .024 / .022 / .820 | 4.2 | 0 |
| **Band-stratified** | **0.700** [0.681, 0.718] | 0.780 [0.757, 0.803] | 0.299 | .041 / .025 / .701 | 4.3 | 0 |
| Sobol (separate from band-stratified) | 0.487 [0.472, 0.502] | 0.731 [0.707, 0.756] | 0.196 | .028 / .025 / .804 | 4.0 | 0.128 |
| Mixture λ = 0.25 / 0.5 / 0.75 | 0.687 / 0.686 / 0.674 | 0.788 / 0.797 / 0.811 | 0.29 / 0.29 / 0.28 | | 4.3–4.4 | 0 |
| **cVAE + ranker (P = 2M)** | 0.702 [0.685, 0.718] | **0.903** [0.883, 0.922] | 0.269 | .018 / .008 / .731 | 10.4 | 0 |
| **Stratified + ranker (P = 2M)** | **0.718** [0.700, 0.736] | 0.876 [0.856, 0.897] | 0.285 | .024 / .011 / .715 | 9.4 | 0 |
| cat + ranker (P = 2M) | 0.701 | 0.902 | 0.268 | | 10.2 | 0 |
| cVAE / stratified / cat + ranker (P = 4M) | 0.625 / 0.659 / 0.634 | 0.933 / 0.927 / 0.935 | 0.23 / 0.24 / 0.23 | | 26–30 | 0 |
| Ranker catalogue (all 36 scored) | 0.552 [0.526, 0.577] | **0.941** [0.921, 0.958] | 0.189 | .007 / .004 / .811 | 36 | 0 |

**Consistency check.** The VAL numbers for cvae, fm, uniform, stratified and the ranker equal the archived `eval_val.json` exactly (same seed schedule).

M = 1, 2 and 8 are in `study_val.json` and `figures/bwl1_budget_val.png`.

**Paired root-bootstrap deltas, M = 4, VAL** (`figures/bwl1_paired_deltas_val.png`):

| Contrast | Δ hit@4 [95% CI] | Δ precision@4 [95% CI] |
|---|---|---|
| **cVAE+rank P2 − stratified+rank P2 (predeclared primary)** | −0.016 [−0.032, 0.000] | **+0.027 [+0.019, +0.036]** |
| cVAE − stratified | **−0.026 [−0.044, −0.008]** | +0.039 [+0.026, +0.052] |
| cVAE − cat | +0.022 [+0.005, +0.039] | +0.006 [−0.006, +0.018] |
| cVAE+rank P2 − cVAE | +0.029 [+0.011, +0.046] | +0.085 [+0.073, +0.096] |
| stratified+rank P2 − stratified | +0.018 [−0.000, +0.038] | +0.097 [+0.083, +0.111] |
| cVAE+rank P4 − stratified+rank P4 | −0.034 [−0.047, −0.021] | +0.007 [+0.001, +0.012] |
| mix 0.5 − stratified | −0.014 [−0.031, +0.005] | +0.017 [+0.004, +0.031] |
| cVAE − flow matching | +0.111 [+0.092, +0.129] | +0.035 [+0.020, +0.051] |
| stratified − Sobol | +0.213 [+0.191, +0.234] | +0.049 [+0.029, +0.067] |

**Decision.** The predeclared rule (in the config, before any run) compares cVAE+rank P2 against stratified+rank P2 at M = 4. The learned arm is significantly better on known-label precision (+0.027) but not on hit@4: −0.016, whose CI lower bound −0.032 is below the −0.02 margin. **Verdict: `INCONCLUSIVE` (a trade-off), not `PROMISING`.** I am not claiming that the learned sampler is superior.

**Timestamp note.** The config's `declared_utc` string reads 02:55Z, which is my typo. The file's mtime is 02:38:13Z, before the 02:41:53Z launch, and its sha256 is bound into `study_val.json`.

## 3. Reading the results

1. **hit@M rewards copying the roster's design.** The Q2 roster sampled destinations by near / mid / far bands, and band-stratified copies those bands: its near / mid / far shares at M = 4 are 0.48 / 0.31 / 0.21, against uniform's 0.18 / 0.55 / 0.27. That is why it leads on hit@1 (0.400 vs 0.256) and hit@4. Known-label precision is less sensitive to this, and there the learned generators lead. Neither is a physical measurement.
2. **Reranking is the dominant effect.** It adds +0.085 (cVAE) and +0.097 (stratified) precision at a small hit cost or gain. After reranking, the learned-vs-analytical gap is +0.027 precision and −0.016 hit. With the larger pool (P = 4M) both converge to precision ≈ 0.93, and hit drops for both.
3. **Mixtures** trade along the same frontier: λ = 0.75 reaches precision 0.811 and hit 0.674. No λ dominates stratified on both metrics. This is exploratory.
4. **The ranker** discriminates well on VAL decided labels (3 seeds):
   - all decided: AUROC 0.921–0.935, AUPRC 0.972–0.977, Brier 0.091–0.099, ECE 0.020–0.031;
   - positives vs certified-geometry negatives only: AUROC 0.919–0.938;
   - positives vs policy-limited negatives only: AUROC 0.923–0.932.

   It was trained on pooled negatives (BCE over all decided labels). Its score is a separately trained discriminative estimate, not a generator density.
5. **Flow matching** pays for its continuous coordinate on a 36-cell support: 24.2 % of its raw draws are illegal and snapped, and its hit is 0.11 below the cVAE's.
6. **Strata, VAL, M = 4.** The learned precision advantage holds for K0–K3 and grows with clutter:
   - cVAE+rank vs stratified+rank precision: K3 0.935 vs 0.910; K0 0.826 vs 0.805.
   - Hit stays tied or favours stratified: K0 0.664 vs 0.699, K3 0.743 vs 0.744.
   - The breakdowns by height, shape, block and object count are in `study_val.json` under `strata_M4`.
7. **Runtime and memory.**
   - Batched GPU latency per context at M = 4 is 0.4 µs for cVAE and cat, 7.5 µs for flow matching, 0.9 µs for cVAE+rank and 4.4 µs for stratified+rank. Stratified's round-robin is a Python loop.
   - Peak VRAM ≤ 620 MB.
   - All negligible next to a teacher call.

## 4. cVAE latent diagnostic (VAL; `latent_val.json`, `figures/bwl1_cvae_latent_val.png`)

| | cvae_s0 | cvae_s1 | cvae_s2 |
|---|---|---|---|
| Aggregate KL (nats) | 0.100 | 0.016 | 0.151 |
| Active units (KL > 0.01) of 8 | 1 | **0** | 2 |
| I(z; d \| S) on VAL contexts (nats) | 0.099 | 0.017 | 0.160 |
| TV(p(d \| z = 0, S), p̄(d \| S)) | 0.055 | 0.012 | 0.063 |
| Held-out positive NLL (IWAE k = 64) | 3.180 | 3.185 | 3.189 |

| Control | Held-out positive NLL |
|---|---|
| cat (exact), s0 / s1 / s2 | 3.182 / 3.183 / 3.183 |
| Uniform | 3.450 |

**What this means.** The latent is **barely used**: one of eight units in s0, two in s2, none in s1, with at most 0.16 nats of information. The code's verdict label was "LATENT_USED", because its threshold (max I ≥ 0.05) was met by two seeds. **My reading is near-collapse.**

The cVAE's density is **indistinguishable from the latent-free categorical control**. Its proposals differ only slightly: +0.022 hit@4 (CI excludes 0), no precision difference. **The cVAE's advantage over uniform and flow matching therefore comes from its conditional categorical decoder over a masked 36-cell support, not from its latent variable.** No existing run was retrained.

## 5. Physics-host confirmation manifest (`PLANNING_ONLY`, nothing executed)

File: `results_bwl/PHYSICS_CONFIRMATION_MANIFEST.json.gz`, rows sha256 `9b4c751e6691d574…`.

**Selection.** TEST-split contexts (never trained on), 20 per object-count stratum, chosen by `sha256('BWL1_CONFIRM|'+qid)` **before** any proposal was generated: **138 contexts**.

**Tuples.** For each context, the M = 4 proposals of 9 arms (cvae, cvae+rank P2, cat, flow, uniform, stratified, stratified+rank P2, Sobol, mix 0.5) are unioned, keeping which arms proposed each tuple. That gives **2,666 tuples**:

| Archived status | Tuples |
|---|---|
| UNTESTED | 2,089 |
| VERIFIED | 432 |
| NEG_CERTIFIED_GEOMETRY | 92 |
| NEG_UNDER_POLICY | 53 |

**Protocol.** The pinned v4 placement teacher (policy `adaptive_v1`, teacher protocol `89cbabed5d7e4847`) must evaluate **every** tuple. Archived-label tuples double as reproducibility checks.

**Analysis.** Physically verified precision per arm over all of its tuples, with root-cluster CIs; no arm or budget may change after outcomes are seen.

**Caveat.** The TEST roots' recorded labels were already inspected. The new tuples' outcomes are unknown, so this is a fresh blind measurement of new proposals, but not a new-root test.

## 6. Commands (server, `P` as in `08`)

```bash
for s in 0 1 2; do $P src/train.py fit --config configs/sampler_bwl1.json --kind cat --seed $s; done
$P src/study.py compare  --config configs/sampler_bwl1.json      # VAL only, 28.9 s
$P src/study.py latent   --config configs/sampler_bwl1.json
$P src/study.py manifest --config configs/sampler_bwl1.json      # physics-host manifest only
```

**New checkpoints** (outside git): `cat_s0` `22e5416f…`, `cat_s1` `58b6d4cf…`, `cat_s2` `63a7bf3e…`. They are verified on both authorized mirrors (`/data/user1/approachb_3d_learning_ckpt_mirror/SAMPLER_BWL1/`, laptop `~/Desktop/approachb_3d_learning_ckpt_mirror/SAMPLER_BWL1/`).

**Resource telemetry:**
- BWL-1 GPU steps: 28 watchdog probes (cat fits 9, reproductions 10, compare 3, latent 3, manifest 3), **0 violations**, 0 stops.
- Maximums over all BWL GPU steps: GPU 41 °C, 76.5 W, our VRAM ≤ 594 MB; other CPUs ≤ 2.0 % mean busy; ≥ 120 GB RAM available.
- Never seen: a foreign GPU process, a pending queue line, or a yadu generator process.

## 7. Threats to validity

- **Known-label selection bias.** Most proposals are UNKNOWN; only the confirmation manifest removes this bias.
- **Roster-design coupling.** Hit@M partly measures agreement with the roster's band design, as shown above.
- **Single VAL population.** The best λ and pool factor were looked at on VAL (exploratory, not decisive). A post-selection claim needs the confirmation run.
- **Placement-only.** These are placement labels conditional on a grasp convention. They say nothing about X, Y or the repair.
