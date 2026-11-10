# 14: post-pilot recovery and training decision (contract v2, R0–R5)

Branch `naren/approachb-3d-learning`; no other branch was created or merged. Status JSON: `reports/14_STATUS.json`.

## Decision summary

| Stage | Outcome |
|---|---|
| **R0** reconcile + preserve | **DONE.** Canonical X/Y ledger (`physics_pilot/r0/`, 7,660 X / 10,419 Y keys). The 1,141 vs 866 negatives, 6,773 → 6,595 X and 2,645 → 2,527 Y are fully reconciled (`physics_pilot/R0_INTAKE_AND_BUDGET_LEDGER.md`). |
| **R1** historical 3D recovery | **DONE.** 52/52 chunks; **0 blocking discrepancies**; **19.56 / 60 CPU-h**. The work started on gvlab CPUs 16–17 and moved to Blackwell after a 34/34 parity PASS. 6,528 X1 verified · 862 X0 relocation re-confirmed · 67 X0 connector · 2,527 Y1 replay-verified · 4 MISSING (archived-record mismatches; two user-approved non-blocking categories). See `physics_pilot/R1_RECOVERY_REPORT.md`. |
| **R2** missing-Y pilot | **DONE** (≤ 6 CPU-h; 72 preregistered items). The strict bitwise witness-identity check failed 0/6 (same grasp and waypoints; drift ≤ 5.3e-5 rad); this is reported, not relaxed. See `physics_pilot/R2_MISSING_Y_PILOT_REPORT.md`. **The bulk missing-Y job (~2,929 queries) was NOT run:** no "c". |
| **R3** preparation | **DONE (preparation only).** Frozen 75-root challenge split; candidate-diverse 2D / 3D manifests (stratified+ranker, cVAE+ranker, categorical control). **The sampler verdict stays INCONCLUSIVE.** See `physics_pilot/R3_FUTURE_PILOT_PLAN.md`. |
| **R4** Blackwell intake + smoke | **DONE.** Sidecar V1: 7,656 X / 7,072 Y, 0 conflicts. The gate declared before any fit **PASSES** (train 1,035 roots / 5,471 X / 683 X0 / min cell 120; val 225 roots / 1,115 X; test roots used 0). One-seed **development-only** smoke of D / E / DIRECT-GEO / G: about 1 GPU-min, no watchdog stop; **no arm is ranked**. 2D: the 60 frozen models were reused and reproduce exactly. No P24 or test tuning. See `physics_pilot/R4_BLACKWELL_INTAKE_AND_SMOKE.md`. |
| **R5** | This report + status JSON. |

## Training decision

1. **3D: READY FOR THE MATCHED STUDY, NOT YET RUN.** Authentic labels clear the declared gate with large margins (5–27× each minimum). The next step needs explicit approval:
   - D / E / DIRECT-GEO / G × ≥ 5 seeds on sidecar V1;
   - validation-only selection;
   - **one** preregistered test evaluation;
   - on Blackwell GPU 0 + CPU 7, which is well under 1 GPU-h.

   No superiority claim is made from the smoke.
2. **2D Q2: intake only after the merge** (contract). The Q2 dataset status is in the next section.
3. **Sampler:** INCONCLUSIVE. Run the candidate-diverse pilot on the frozen challenge split when approved.
4. **Open data issues to carry forward:**
   - 4 archived 3D negatives disagree with the pinned teacher. One archived CERTIFIED negative is now plannable (`results_r1_diag/`).
   - R2 witness bitwise drift.
   - The bulk missing-Y job (needs "c").

## 2D Q2 dataset: status at hand-off (2026-11-09, 16:35 IST)

**Owner map V4 (final, frozen):** 4,620 roots / 249,965 tasks / 238,160 teacher queries, 0 duplicates. **No root is owned by Blackwell**, so the whole dataset will be `BITWISE_PORTABLE`. Map V4 replaced Blackwell at the user's request (15:50 IST). The Blackwell share never ran (attested), and its 2 partial roots are recomputed from scratch on gvlab.

| Owner | Roots | Teacher queries | Host | State at hand-off |
|---|---:|---:|---|---|
| ORIGINAL | 201 | 10,348 | laptop → gvlab copy | complete, sha-verified on gvlab |
| LAPTOP | 1,585 | 85,924 | laptop → gvlab copy | complete, sha-verified on gvlab |
| LAPTOP2 | 255 | 11,944 | laptop → gvlab copy | complete, sha-verified on gvlab |
| LAPTOP3 | 85 | 4,500 | laptop → gvlab copy | complete, sha-verified on gvlab |
| GVLAB | 1,520 | 74,160 | gvlab | complete |
| GVLAB2 | 281 | 13,724 | gvlab | complete (16:15 IST) |
| GVLAB3 | 520 | 28,076 | gvlab | **running** since 16:31 IST, 16 workers; ETA about 02:30 IST Nov 10 |
| GVLAB4 | 173 | 9,484 | gvlab | queued after GVLAB3 in the same chain; ETA about 06:00 IST Nov 10 |

**Done:** 3,927 / 4,620 roots (85 %).

**Remaining:** gvlab only. The chain is `chain_gvlab34.sh` in tmux `q2_chain34`. It retries starts refused only for busy CPUs, resumes after a safety stop at most 3 times, and stops on any other refusal.

**Merge:** after GVLAB4, one gvlab-only command (`q2_handoff_v2/RUNBOOK_Q2_V2.md`) writes `merged/q2_canonical_v4` plus `MERGED_Q2_V4_AUDIT.json`. 2D Q2 intake waits for `pass: true`.

**Laptop:** no 2D compute. The old V1 queue is frozen (`PAUSE_QUEUE`) and its sleep inhibitor has been stopped.

**gvlab ≥ 4 free cores:** sajid's 2 unpinned jobs started at 16:15. The start gate now accepts a momentarily busy worker CPU only when ≥ 4 CPUs stay idle after our 18 CPUs start (`MIGRATE_OK`). Observed: his jobs migrated to CPUs 21–22, and 6 CPUs are idle.

### Update 2026-11-10, 10:45 IST

- **Generation finished.** GVLAB3 finished at 03:23 IST (520/520 roots) and GVLAB4 at 07:07 IST (173/173 roots), with 0 pauses and 0 safety stops. **All 4,620 roots are generated.**
- **Merge finished** (10:12 IST, gvlab): `merged/q2_canonical_v4`, with every identity and snapshot-hash check passing. Every root is `BITWISE_PORTABLE`.
- **Counts** (from the merged records; full detail in `q2_handoff_v2/MERGED_Q2_V4_COUNTS.json`):

  | | Count |
  |---|---:|
  | Tasks | 249,965 |
  | Teacher queries | 238,160 |
  | Feasible | 176,038 (73.9 %) |
  | Infeasible | 62,122 |
  | Interrupted | 0 |
  | Verified grasp/place pairs | 1,272,077 |
  | Roots, train / val / test | 3,120 / 751 / 749 |

- **Certification pending.** The full `launch_check` audit re-certifies every stored witness at about 80 s per root on 16 processes. It started at 10:12 IST; the ETA is about 17:00–18:30 IST. Its output is `MERGED_Q2_V4_AUDIT.json`. 2D Q2 intake waits for `pass: true`.
- **Scope note:** Q2 scenes have no designated target, so Q2 gives single-relocation feasibility labels only, with no Y and no per-scene k\*. 2D X/Y lives in the BXY sidecars. A corrected per-scene k\* table (k\* = 0 means directly feasible; NONE means no plan found, treated as infeasible) for both 3D (recovering k = 0 from the raw roots) and 2D-BXY was proposed and not yet run.

## Resource and safety record

- **Caps:** CPU-h, GPU-h and wall-clock caps were respected (R1 19.56 / 60 CPU-h; R2 ≤ 6 CPU-h; R4 about 1 GPU-min of 6).
- **≥ 4 free cores:** kept on every server at all times. Blackwell used E-cores only after the P-core thermal finding.
- **Collaborators:**
  - The Blackwell guard yielded to sajid's queue once (R1).
  - The 2D Q2 Blackwell share runs alongside his jobs only on E-cores, with a pause watchdog (user-approved).
  - No collaborator process was ever touched.
- **Clean stops:** every stop (handover, thermal, yield, discrepancy) was clean and idempotent. Partial ledger records are kept under `handover_partial_records`; guard events are archived.
- **Never committed:** credentials, checkpoints or large NPZ files.
