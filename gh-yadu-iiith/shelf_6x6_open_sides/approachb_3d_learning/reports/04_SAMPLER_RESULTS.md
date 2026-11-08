# 04: Sampler results (GPU-trained on the full verified v4 snapshot)

**State: `SAMPLER_GPU_TRAINED_ON_FULL_SNAPSHOT`.**
- Data: snapshot `v4full_82926152fa9cc345`.
- Hardware: GPU 0 (RTX PRO 5000 Blackwell), CPU 7 only.
- Selection frozen at `2026-11-08T01:12Z` (`results/FROZEN_FOR_TEST.json`, 15 checkpoint hashes). TEST was evaluated once, afterwards.
- Machine-readable version: `04_SAMPLER_RESULTS.json`. Raw evaluations: `results/eval_val.json` and `results/eval_test.json`.

## What ran

| Run family | Seeds | Epochs (early stop, patience 8) | Best val loss | Wall time per run | Peak VRAM |
|---|---|---|---|---|---|
| cVAE (CE + β·KL) | 0, 1, 2 | 33 / 28 / 36 | 3.177 / 3.184 / 3.185 | ~30 s | 114 MB |
| Flow matching (velocity MSE) | 0, 1, 2 | 19 / 19 / 29 | 1.646 / 1.653 / 1.628 | ~30 s | 113 MB |
| Ranker (BCE, labelled trials) | 0, 1, 2 | 25 / 37 / 37 | 0.307 / 0.290 / 0.285 | ~30 s | 140 MB |
| cVAE geometry-only | 0, 1, 2 | 33 / 30 / 36 | 3.176 / 3.181 / 3.181 | ~30 s | 114 MB |
| FM geometry-only | 0, 1, 2 | 19 / 19 / 29 | 1.645 / 1.654 / 1.627 | ~30 s | 113 MB |
| Ranker ablation (separate experiment `SAMPLER_V1_RANKER_ABLATION`, own freeze and test): full vs geometry-only | 0, 1, 2 each | | | ~30 s | |

About 27 s of each run is loading the snapshot to the GPU; training itself is about 3 s.

**Resource trace** over all GPU work:
- 75 watchdog probes, **0 violations**, 0 stops.
- Maximums: GPU 39 °C, 52 W, our VRAM 536 MB; CPU package 59 °C; the other 23 CPUs at most 2.3 % mean busy.

**Reproducibility:** retraining ranker seed 0 in the second experiment reproduced every epoch's loss exactly, and the same test AUROCs (0.9222 / 0.9312 / 0.9385).

## Main comparison

Same contexts, same legality mask, raw cap 8M with replacement, first M unique kept. Values are mean ± std over 3 training seeds × 3 evaluation seeds (controls: 3 evaluation seeds). Metrics count known labels only.

**TEST** (1,385 contexts with a labelled destination; 1,276 have a recorded positive):

| Sampler | hit@1 | hit@4 | hit@8 | Precision @1 | Precision @4 | Precision @8 | Recall @8 | Duplicates @8 | Diversity @4 |
|---|---|---|---|---|---|---|---|---|---|
| **cVAE** | 0.240 ± .004 | **0.655 ± .015** | **0.879 ± .005** | 0.795 ± .018 | **0.801 ± .010** | **0.798 ± .004** | **0.400** | 0.237 | 2.88 |
| Flow matching | 0.188 ± .006 | 0.557 ± .022 | 0.811 ± .009 | 0.770 ± .011 | 0.756 ± .010 | 0.768 ± .006 | 0.332 | 0.172 | 2.69 |
| Uniform (legal) | 0.145 ± .005 | 0.451 ± .003 | 0.710 ± .006 | 0.693 ± .040 | 0.713 ± .003 | 0.725 ± .009 | 0.261 | 0.119 | 3.19 |
| Band-stratified | 0.389 ± .005 | 0.688 ± .011 | 0.861 ± .007 | 0.793 ± .010 | 0.758 ± .010 | 0.741 ± .002 | 0.379 | 0.154 | 3.14 |
| Ranker (top-M, deterministic) | 0.182 ± .013 | 0.534 ± .012 | 0.798 ± .006 | **0.906 ± .009** | **0.906 ± .004** | **0.900 ± .003** | 0.328 | 0 | 1.67 |

**VAL** results are the same picture: cVAE hit@4 0.673 and precision@4 0.819; FM 0.563 / 0.784; uniform 0.457 / 0.743; stratified 0.700 / 0.780; ranker 0.552 / 0.941.

**Other measurements:**
- **Held-out positive NLL** (test, nats): cVAE **3.178** vs uniform 3.444 (`log |legal|`). FM: not computed (it has no exact likelihood).
- **Known-negative rate** (proposals on recorded negatives): about 0.05 for every generator and control, and **0.017** for the ranker.
- **FM before projection:** about **25 %** of raw draws land on an illegal cell, either occupied or off the shelf (test 0.251–0.254, val 0.236–0.248). About **11–12 %** land outside the shelf `[-1, 1]^2` entirely. The nearest-legal projection repairs all of them, so the final proposals are 100 % legal. This share is a real cost of the continuous parameterization on a discrete, occupancy-masked support. The cVAE's masked decoder has 0 illegal draws by construction.
- **GPU latency:** about 1.2 µs per context for cVAE and about 17 µs for FM (32 ODE steps), for the whole batch at M = 8.
- **Ranker calibration** (test labelled rows, n = 7,796, base rate 0.721): Brier 0.101, ECE 0.025, AUROC 0.922. These are discriminative scores, separate from the generators' densities.

## Ablations

**3D object context (shape / size / height) vs geometry-only (positions + occupancy):**

| Model | Full | Geometry-only | Conclusion |
|---|---|---|---|
| cVAE, test hit@4 / precision@4 | 0.655 / 0.801 | 0.650 / 0.801 | no measurable effect |
| FM, test hit@4 / precision@4 | 0.557 / 0.756 | 0.558 / 0.756 | no measurable effect |
| **Ranker, test AUROC** | **0.931 ± 0.007** | 0.907 ± 0.012 | **3D object context helps feasibility discrimination** |
| Ranker, test ECE / Brier | 0.023 / 0.095 | 0.047 / 0.112 | better calibrated with the 3D context |

- Positives-only proposal densities are driven by free space and distance; object shape and height mainly determine **which** tested cells fail.
- **Frozen vs jointly trained encoder:** not run. All encoders here are trained jointly. Capacity was not increased, because the baseline already works.

## Per stratum (TEST, M = 4, seed 0)

Known-label precision, cVAE vs uniform, by objects in the scene:

| Objects | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|
| cVAE | 1.000 | 0.929 | 0.841 | 0.776 | 0.769 | 0.743 | 0.681 |
| Uniform | 0.895 | 0.897 | 0.796 | 0.614 | 0.692 | 0.667 | 0.523 |

The gain is largest in cluttered scenes. Shape, height bin and block breakdowns are in `04_SAMPLER_RESULTS.json` and `figures/strata_*.png`.

## Reading the results

1. **cVAE > flow matching > uniform** on hit, recall and known-label precision, consistently on VAL and TEST and across seeds.
   - FM's continuous coordinate plus nearest-legal projection is a weaker fit for a 36-cell support than a masked categorical decoder. About 25 % of FM's raw draws need repair by the projection.
   - It is also 14× slower at 32 Euler steps.
2. **The band-stratified control** has the best hit@1 and matches cVAE on hit@4 and hit@8, but has lower precision.
   - Hit and recall of recorded positives reward reproducing the roster's own near / mid / far test design (`02`, Q2 selection limits).
   - Known-label precision is the cleaner learned-vs-control comparison. There, cVAE (0.80) beats stratified (0.76) and uniform (0.71) at M = 4–8.
3. **The discriminative ranker** has by far the highest precision (0.90–0.91) and the fewest known negatives, but low diversity (1.67).
   - A natural next stage is **cVAE proposals re-ranked or filtered by the ranker**. It is not evaluated here.
4. **Limits.**
   - All metrics use only destinations whose outcomes were independently recorded. Untested proposals are unknown, not failures.
   - No new teacher calls were made: no CPU physics or IK ran on the training server.
   - An unbiased success rate for unseen proposals needs a separate teacher evaluation on an authorized physics host.

## Figures (`figures/`)

- `learning_curves.png`: train and val per run.
- `budget_val.png` and `budget_test.png`: hit, precision, recall and duplicates vs M.
- `strata_val.png` and `strata_test.png`: known-label precision by shape, height, block and object count.
- `overlays_val.png`: six real VAL scenes, showing cVAE / FM / uniform proposal frequencies over 256 draws, with recorded VERIFIED cells (green), recorded negatives (red ×), occupied cells and the held object's source cell. Unmarked cells are untested.
