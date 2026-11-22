# Pre-registration: per-step hazard scorer on V3.4 pools, TEST (2D + 3D)

Status: APPROVED by nav on 2026-11-22, with margin M = 10 kept as written. It was revised after Devil's Advocate reviews. It was
written after VAL and before any TEST access by us. It becomes binding when this file, `hybrid/src/{hz_eval,twod_hazard}.py` and
`docs/HZ_V34_FREEZE.sha256` are committed to git. That commit hash is then written below, and only after that does any TEST step run.

Freeze commit: `8e18429cb404cc846ca40899d2ed3a09da4489db` (this line is the only change after it).

## What is compared
- **Ours ("hazard"):** the per-step hazard scorer (`twod_hazard.py`, HZ_MODE=v34). It scores V3.4's IDENTICAL candidate pools, which are
  read from the sealed `K3ABL/runs/predictions/FULL_s0/test_{dom}.jsonl` with the pool hash checked per root. FULL s0, s1 and s2 share
  pools; the VAL and TEST hashes are verified equal. Ours decides with `exhaustive.score_root` and the stopping rule.
- **Baseline:** V3.4 FULL, seeds 0–2, under its frozen rule, from the team's stored TEST results
  (`code_v34/ablations_k3/reports/test/results_FULL_s{0,1,2}_{dom}.jsonl`). These files are hashed in the freeze file; nobody on our side
  reads them before the commit. They are first read by `confirm` in procedure step 4.
- **Models:** ours seeds 0–2, already trained and hashed, with no retraining. Each was trained on TRAIN rows of snapshot
  `abf_v32_fb0eef4f5ff433ee`, and its checkpoint is the best VAL-NLL epoch. FULL uses the same selection (xy.py `fit`; 2D s0 took epoch 48).
- **Split:** the historical V3.4 TEST indexes `K3ABL/runs/public/{public,private}_test_{dom}.jsonl` (2D 749 roots, 3D 370). The team
  consulted this split earlier, so it is not pristine for them; it is unseen by us. Inference is conditional on these 3 trained seeds
  per arm.

## Primary hypothesis, endpoints and decision rule (per domain; 2D and 3D decided separately)
- **H:** ours under the FROZEN rule (noop_gate 0.5, eta 0.3, alpha=beta=gamma=0) succeeds more often than FULL on eval_task roots,
  without being materially worse at no-op or abstain.
- **Endpoints**, from `benchmark.correct`:
  - task = SUCCESS on eval_task roots;
  - noop = NOOP on eval_noop roots;
  - abstain = ABSTAIN on eval_abstain roots.
- **Labels:** both arms' REPAIR picks are scored from the same merged label table (their cache plus ours); FULL's NOOP/ABSTAIN decisions
  come from its stored rows. On VAL this matched FULL's stored outcomes exactly (2D 1255/1255, 3D 698/698).
- **Per root:** each endpoint indicator is averaged over the 3 seeds. For task, a seed-run with outcome UNKNOWN is dropped for that root
  and arm. For noop and abstain, a REPAIR scores 0 whether its label is known or not (`benchmark.correct`).
  A root with no known run in either arm is excluded and counted; if exclusions exceed 2% of the mask, this is reported prominently.
- **Effect:** mean over roots of (ours − FULL).
- **Inference:** percentile bootstrap that resamples **worlds**. A world is a cluster of scene variants, defined by
  `hz_eval._cluster`: root id minus `-sNNN` and minus `conv-`. B = 10,000 and the rng seed is 20261122. "lb97.5" means
  `np.quantile(boot, 0.025)`, a one-sided 97.5% lower bound. The alpha is 0.025 per domain, Bonferroni over the 2 domains.
- **Decision:** H is supported iff:
  - task lb97.5 > 0, AND
  - noop lb97.5 > −M and abstain lb97.5 > −M, with non-inferiority margin **M = 10 points** (`hz_eval.MARGIN`).
- **Code:** `hz_eval.py confirm DOM test` computes everything above and prints `supported`.

## Mandatory sensitivity analyses (each reported next to the primary, with pre-stated consequences)
1. **Teacher.**
   - Ours' new picks are checked by the Mac copy of the teacher, while FULL's come from the gvlab cache.
   - `hz_eval.py crosscheck DOM test 4` re-runs on the Mac every REPAIR pick of FULL s0–2, plus any of ours that has only a gvlab label.
   - `confirm` then recomputes all endpoints with BOTH arms on Mac labels only (`sens_mac_teacher`).
   - **If the primary is supported but the Mac-only task lb97.5 is ≤ 0 (or a guard fails there), the result is reported as
     "teacher-sensitive, not confirmed".**
   - VAL agreement was 2D 100%, 3D about 96%, with symmetric flips.
2. **UNKNOWN as failure** (`sens_unknown_as_fail`). UNKNOWN counts per arm and seed are always reported. **If the task lb97.5 is ≤ 0
   under this sensitivity, the claim is downgraded to "not robust to unknowns".**
3. **Root-level bootstrap** (`lb97.5_root`): reported only, no consequence.

## Secondary (reported, no decision attached)
- Per-seed task, noop and abstain for both arms.
- Ours under the VAL-tuned rule (`chosen_rule_{dom}.json`, picked on seed 0 VAL; its criterion was fixed before VAL physics).
  FULL got no such tuning, so this is never the headline.
- **Exploratory:** abstain with and without the 3 flagged 2D NONE scenes (w0248-s004, w0167-s011, w0166-s021). This analysis was
  chosen after seeing them, so it is exploratory by construction.

## Precision (from VAL, exploratory, not a power guarantee)
The VAL 3-seed task effects were 2D +19.9 points and 3D +12.3 (root bootstrap 95% CI half-widths about 4 points). With similar TEST
sizes and world clustering (which widens the interval), the task lb97.5 stays above 0 for observed effects larger than about 5–6 points.

## Procedure (single run)
1. nav approves; commit this file, the code and the freeze file; write the commit hash above and commit that.
2. Pools: run `twod_hazard.py pool DOM test SEED i/4` for seeds 0–2 and both domains. It reads only the public index and the FULL_s0
   pools (no labels). `ours()` asserts that duplicate pool files agree.
3. Teacher: `hz_eval.py teach DOM test SEED 4`, then `crosscheck DOM test 4`. TEST uses only the frozen rule and the chosen rule
   (`rule_ids`). Known labels are reused, and new Mac labels are appended.
4. Run `metrics DOM test SEED`, then `confirm DOM test` **once**. The output files are committed as they are.
5. **Deviations:**
   - No re-runs. A crashed job is resumed (jobs are resumable and append-only).
   - A code change after the freeze is allowed only to fix a crash, and must not touch scoring, decisions, labels, endpoints or inference.
   - Every deviation is logged with its diff and reported next to the result.
   - Any change made after the first TEST metric is seen is disclosed as post-hoc, and the frozen-code result stays the headline.
6. Only the primary is CONFIRMATORY (TEST). Everything else is secondary or exploratory.

## Freeze contents (`docs/HZ_V34_FREEZE.sha256`)
- This file, `hz_eval.py` and `twod_hazard.py`.
- Model checkpoints, `train_*.json` and `chosen_rule_*.json`.
- Snapshot data, the public/private TEST indexes, and the FULL s0–2 TEST predictions (pools).
- The FULL s0–2 TEST result files, hashed but not read.
- Both verifier caches as of the freeze (theirs, read-only, may already hold TEST labels; and ours).
- V3.4 code (`code_v34/*.py`, `scripts/v32b_phys.py`), teacher code (`teacher2d_wt/*.py`, `fin3d/code/*.py`), and `pip freeze` of
  env2d and env3d.

## Known limits
- Only the scorer differs; the sampler and pools are V3.4's. The gain is from choosing among the same candidates.
- The 3D teacher is not bit-reproducible across machines (about 4% flips, symmetric); this is covered by sensitivity 1.
- No other baselines (E-M4M etc.) are part of this test.
- The endpoints are the team's; the rule grid and the tuning criterion were fixed before VAL physics. The margin M = 10 and the
  world-cluster bootstrap were added after VAL, on a reviewer's advice, before any TEST access.
