# 12: Blackwell readiness decision (BWL-4) — STOPPED

**Basis.**
- Scientific authority: `final-fixed-methodology-1.pdf` (5 Nov 2026, sha256 `b7a39e3c…`), read in full.
- Engineering contract: `BLACKWELL_APPROACH_B_EXECUTION_CONTRACT_v1.md`.
- Base commit: `3a820819` on `naren/approachb-3d-learning`.
- **BWL commit: `69a6eeb8cc4b2b725ce23839dee8a44f1f3b2e38`**, pushed; `git ls-remote` confirmed the remote head equals it. Source hashes are in `08`, `10` and `12_BLACKWELL_READINESS_DECISION.json`.

**Scope.** Only BWL-0 to BWL-4 were executed. **BWL-5 to BWL-8 were not started** and need your explicit authorization.

## Verdicts

```text
SAMPLER_OFFLINE:                 INCONCLUSIVE   (predeclared VAL rule: cVAE+rank vs stratified+rank = +0.027 precision, -0.016 hit; trade-off)
SAMPLER_NEW_PHYSICAL_PRECISION:  NOT_RUN        (138-context / 2,666-tuple physics manifest prepared)
XY_3D_MATH_AND_INTERFACE:        PASS           (10/10 test groups; real-format 3D records; all four arms)
XY_2D_PARITY:                    PARTIAL        (kernel + all modules/compositions identical to e113c6680, <= 6e-17; 3D GEO features are declared proxies)
Y_3D_REPLAY_VALIDITY:            PLANNING_ONLY  (0 replay-verified; 2,645 planned positives; 4,367 operational negatives)
X_3D_AUTHENTIC_DATA:             ABSENT         (0 records; 887-query pilot manifest prepared)
BLACKWELL_FULL_XY_TRAINING:      BLOCKED        (no authentic X; no replay-verified Y)
END_TO_END_3D:                   NOT_RUN
RESOURCES_AND_WATCHDOG:          PASS           (one ownership defect found and fixed; 39 probes, 0 violations)
```

| Stage | Verdict | Report |
|---|---|---|
| BWL-0 audit | PASS (one safety defect fixed) | `08` |
| BWL-1 sampler | PASS (stage complete; evidence INCONCLUSIVE) | `09` |
| BWL-2 X/Y fidelity | PASS (parity PARTIAL by design) | `10` |
| BWL-3 Y provenance and X specification | PASS (audit and manifests; nothing executed) | `11` |
| BWL-4 consolidation | PASS | this report |

## What changed in code vs what was only analyzed

| File | Change | Purpose |
|---|---|---|
| `src/guard.py` | changed | `OUR_SCRIPTS` lists every GPU entry point; the watchdog flags `STOP_REFUSED_NOT_OURS` |
| `src/train.py` | changed | preflight refuses an entry point the watchdog cannot stop; `evaluate --out` never overwrites; `--kind cat` |
| `src/models.py` | changed | added the `Categorical` control. Archived classes are unchanged, and the archived checkpoints load and reproduce |
| `src/study.py` | new | BWL-1 VAL comparison, latent diagnostics, confirmation manifest |
| `src/xy3d.py` | rewritten | network portion only: the kernel is unchanged; `XYNet` = the batched 2D source family (D, E, DIRECT-GEO, G) |
| `src/xy3d_data.py` | new | 3D learner-visible views, GEO3D, Y audit, Y replay and X pilot manifests |
| `src/xy3d_tests.py` | new | contract tests 1–10 and the SYNTHETIC recovery |
| `src/figs_bwl.py` | new | figures from saved outputs |
| `tests/test_core.py` | changed | adds watchdog-ownership and categorical tests (11 tests) |
| `configs/sampler_bwl1.json` | new | predeclared arms, pools, λ, bootstrap and decision rule |

**Only analyzed:** the snapshot, the 21 archived checkpoints, `eval_val.json` / `eval_test.json`, the 2D source, and the 3D generator source and records. None of them was modified.

**New checkpoints:** 3 (`cat_s0`–`cat_s2`), outside git, verified on both authorized mirrors.

**Figures** (`reports/figures/`):
- `bwl1_budget_val`, `bwl1_composition_val`, `bwl1_paired_deltas_val`, `bwl1_proposal_maps_val`, `bwl1_cvae_latent_val`;
- `bwl2_synthetic_recovery`, `bwl2_shared_y_ab_ba_schematic`;
- `bwl3_y_provenance`.

## Headline findings

1. **Band-stratified leads the cVAE on hit@4.** Archived TEST: 0.688 vs 0.655. VAL: 0.700 vs 0.673, CI of the difference [−0.044, −0.008]. The learned generators lead on known-label precision (VAL 0.819 vs 0.780).
2. **Reranking dominates.** It adds +0.085 to +0.097 precision. After reranking, learned vs analytical differ by +0.027 precision and −0.016 hit: **INCONCLUSIVE**.
3. **More than 70 % of every arm's proposals are untested (UNKNOWN).** No physical precision of new proposals exists until the confirmation manifest runs.
4. **The cVAE's latent is nearly collapsed.** At most 2 active units and I(z; d | S) ≤ 0.16 nats, with an NLL equal to the latent-free categorical control. The value comes from the masked conditional categorical decoder.
5. **The 3D X/Y adapter now *is* the source model family**, numerically verified. On SYNTHETIC data it recovers a planted F5 and separates it from a planted order effect. This is not physical evidence, and **F5 is still a hypothesis**: on synthetic data it barely improved success log loss, mirroring the 2D result.
6. **Every one of the 7,012 3D "Y" records is planning-based.** The 2,645 positives were never executed or replayed; the 511 K0 originals were executed once inside the teacher. The negatives are usable operational zeros. **There is no authentic 3D X.**

## What remains before complete X/Y training (all external to this server)

1. **2D Q2 merge.** The laptop and gvlab collections are still running; the merged audit is expected around 9 Nov ~22:30Z. Then a BWL-5 intake audit, followed by 2D sampler training under the same method. Q2 placement labels are never X.
2. **3D Y replay.** Run `PHYSICS_Y_REPLAY_MANIFEST` (120 records), then the full replay of the 2,645 positives.
3. **Authentic 3D X collection.** Run `PHYSICS_X_PILOT_MANIFEST` (40 roots, 887 queries, 191 same-endpoint order groups), apply its go / no-go rule, then scale up.
4. **Fresh physical sampler validation.** Run `PHYSICS_CONFIRMATION_MANIFEST` (2,666 tuples) on an authorized physics host.
5. **Only then** BWL-6: a real D / E / DIRECT-GEO / G training run, with paired multi-order X / Y evidence and the training loop ported (fold builder, inner rule selection).

## Optional next GPU work (each needs your authorization)

- `Y_PLANNING_SUPERVISION_PRELIMINARY`: an F3-only run on the 7,012 planning-based Y records, in a separate namespace. It would not be an X/Y result.
- A cleaner sampler-side study: draw a new locked blind root sample, or run the confirmation manifest first.

**STOPPED.** No further stage was started, and no job of mine is running on the server (GPU 0 at 2 MiB at the end).
