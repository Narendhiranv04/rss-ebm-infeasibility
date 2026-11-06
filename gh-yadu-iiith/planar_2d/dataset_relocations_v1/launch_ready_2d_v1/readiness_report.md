# Full 2D placement-sidecar extension: launch-readiness report

**Date:** 6 Nov 2026.
**Worktree:** `/home/naren/Desktop/rss-placement-smoke`, branch `naren/placement-smoke-2d`, HEAD `9e640690`. Everything here is uncommitted; nothing is pushed.

**Contract:** `Approach_B_Full_2D_Placement_Extension_Launch_Readiness_v1.md`. **Scientific authority:** `final-fixed-methodology-1.pdf` (5 Nov 2026).

## Verdict: `NOT_READY_TO_LAUNCH`

Every gate except one passed, and the pre-registered rule selected a policy (8 grasp pairs per query). The blockers are:

1. **25-worker thermal gate failed** (`NOT_READY_TO_LAUNCH_25_WORKERS`).
   - The rule, frozen before the pilot, needs both 25-worker 64-pair runs to finish with no safety stop. Run 1 was cancelled twice
     by the immediate Tctl ≥ 95 °C rule. Run 2 finished cleanly.
   - Both cancels happened at *low* parallelism, not under full load: once when every worker was frozen and something outside the
     sandbox heated the CPU, and once during a staggered start-up.
   - Under full 25-worker load the CPU holds at 86 °C with no pauses (§8).
2. **Disk headroom is too small for a full-size workload today.**
   - `/home` has 1,751 MB free. It had 3,893 MB at preflight; the roughly 2.1 GB drop came from other activity on `/home` (this work
     wrote about 52 MB).
   - The pre-registered disk budget is free space minus a 1,536 MB reserve, so only 215 MB now. Only the smallest workload fits
     (Q4, FIRST_SUCCESS, 53,752 queries).
   - The selected full-size workload (Q1, FULL) needs about 2.6 GB free.

The launch is locked: no production policy is frozen, so the guard refuses every `--execute-full`. The full run was not started.
The production planner call count is **0**, and `full_2d_v1/` does not exist.

| Gate | Result | Evidence |
|---|---|---|
| Source and baseline audit, smoke `SMOKE_PASS` reproduced from saved files | PASS | §1 |
| Lineage-closed 70/15/15 splits, no cross-split sharing | PASS | §2, `split_manifest.json` |
| Deterministic query plan from metadata only; candidates do not read protected fields | PASS | §3, test `test_candidates_ignore_protected_metadata` |
| Teacher equals the repaired smoke (attempts, 48×48 categories, witnesses) | PASS | test `test_teacher_reproduces_the_repaired_smoke` |
| Pre-registered budget rule selects a budget | PASS: 8 pairs | §6, `audit/pilot_decision.json` |
| Negative diversity at 8 pairs (≥ 10 label-0 queries, ≥ 5 lineage groups, ≥ 3 classes) | PASS: 37 / 12 / 5 | §6 |
| Determinism: serial vs two 25-worker runs | PASS: 347/347 and 25/25 identical | §7 |
| Nested budgets: independent 8-pair run vs first-8 accounting | PASS: 300/300 | §7 |
| SIGKILL + resume | PASS | §7 |
| Audits (re-certify every stored witness) | PASS: 0 failures in 5 runs | §7 |
| Metadata-only dry run of all 5,000 roots | PASS: 13/13 checks, 0 planner calls | §9 |
| 25-worker thermal safety | **FAIL** (run 1) | §8 |
| Workload fits time and disk limits | Only Q4 fits at 1.75 GB free | §9 |

## 1. Source audit and baseline protection

**Environment.**
- venv `/home/naren/Desktop/rss3d_env/venv` (Python 3.13.11, NumPy 2.4.6), with `PYTHONDONTWRITEBYTECODE=1` throughout.
- AMD Ryzen 9 8940HX: 16 cores / 32 logical CPUs, 2 core complexes (CCDs), all 32 online and allowed.
- 31 GB RAM, 16 GB swap. AC power.
- Sensor: hwmon `k10temp` `Tctl`. Tjmax is 100 °C.

**Original dataset.**
- 5,000 selected roots (NONE 1,000 / K0 1,250 / K1 800 / K2 800 / K3 1,150), all unique and present.
- `selected.json` sha256 `edb8dfca…`.
- 47,384 files hashed before any edit (`audit/baseline/original_dataset.sha256`).

**Repaired smoke.** All 2,460 `smoke_repro_v2` files match its manifest, and so do the code hashes. Re-auditing copies of the serial
and 28-worker runs in scratch reproduced:
- fingerprint `1ed5e19f…`;
- 1,530/1,530 checks;
- 30/4/1 labels;
- 222 witnesses.

**Integrity after all work** (`audit/integrity.json`):

| What | Files | Changed | Added |
|---|---|---|---|
| Original dataset | 47,384 | 0 | 0 (git status clean) |
| `smoke_20261106` | 1,129 | 0 | 0 |
| `smoke_repro_v2` | 2,461 | 0 | 0 |

The three smoke scripts are byte-identical. No bytecode was written.

## 2. Frozen splits (`split_manifest.json`, sha256 `f7e8cbba…`)

**Grouping.** Lineage groups are the union-find of three proven relations:
- `parent_family_id` (2,743 families; this is dataset-bd's own split key);
- identical object layouts (151 duplicate layouts, mostly NONE scenes);
- conversion origin (661 conversions).

This gives 2,431 groups; the largest has 137 roots (K0 obstacle-free scenes).

**Assignment.** Groups are assigned whole, by a deterministic class-stratified greedy, largest groups first. The 5 roots touched by
earlier development (the smoke roster and the unit fixture) are forced into train.

| Split | Roots | NONE / K0 / K1 / K2 / K3 | Groups |
|---|---|---|---|
| train | 3,500 | 700 / 875 / 560 / 560 / 805 | 1,269 |
| val | 751 | 150 / 188 / 120 / 120 / 173 | 582 |
| test | 749 | 150 / 187 / 120 / 120 / 172 | 580 |

- Families, layouts and conversion origins shared across splits: 0 each. Duplicate root ids: 0.
- Every pilot root is train. **No teacher query touched a val or test root.**

**Leakage note.** Groups cover proven sharing only. Near-duplicate layouts (differing by one object) are not merged.

## 3. Query plan (`query_plan.jsonl.gz`, sha256 `d078612b…`): metadata only, no teacher calls

T (`t0`) is never queried, because only the target action moves it.

**Candidates use scene geometry only.** No roles, intended blockers, parking cells, repair labels, Y labels or K are read. The test
scrambles all of those and checks that the root block and the design edit are unchanged.

**Root block.** Up to 4 obstacles, chosen round-robin over source bins (near / mid / far, by the dataset's base-distance tertiles),
each with 12 destinations.
- Destinations are legal, empty and not T's goal cell.
- They are stratified by Chebyshev distance from the source (near 1–2, mid 3–4, far 5–7), with hash (simple-random) order inside a
  stratum. Each row records its stratum's k and n, so the inclusion probability is k/n.
- The rows are interleaved, so any leading slice stays stratified.

**Two hypothetical object-only edits per root.**
- `prefix1` is a random design edit.
- `prefix2` is an archived executed first move. It is chosen uniformly over all archived B first moves, ignoring role and y, and
  flagged `context_selected_with_protected_metadata`. It falls back to a second design edit when no such move exists.
- Each edit has 2 objects (those nearest the edited one) × 6 destinations. The first object's first destination is the vacated
  cell, and the same object and cell in the root scene is stored as a static-ineligible contrast (no teacher call).
- `prefix_provenance` is always `HYPOTHETICAL`; no robot state is assumed.

**Totals.**
- 4,620 roots carry queries; 380 obstacle-free K0 roots carry none.
- 308,652 teacher queries plus 8,385 static rows (61.7 teacher queries per root, at most 72).
- Seeds: `planner.call_seed(41, "<trial_id>/carry<i>.<j>")`.

**Quota ladder.** Each quota is a leading slice that contains the pilot:

| Quota | Teacher queries |
|---|---|
| Q1 | 308,652 |
| Q2 | 205,768 |
| Q3 | 114,556 |
| Q4 | 53,752 |

## 4. Implementation (all in `dataset_relocations_v1/code/`; the three smoke scripts are unchanged)

| File | Lines | Role |
|---|---|---|
| `launch_plan.py` | 259 | lineage, splits, query plan, pilot roster, quotas; `freeze` / `verify` |
| `placement_teacher.py` | 293 | query under a budget policy, reusing smoke V2 held modes, `det_rrt`, witness build and certification; atomic parts; per-root pack; witness pack/unpack |
| `launch_runner.py` | 318 | topology-aware 25-worker runner, thermal and resource guards, resume, packing, status, CLI |
| `launch_guard.py` | 97 | metadata-only preflight (counts planner calls), confirmation token, guarded production entry |
| `launch_check.py` | 316 | audit (parallel per root), compare, nested, budget tables, interrupt check, integrity |
| `launch_decide.py` | 157 | gates and the pre-registered decision; freezes the production policy only if every gate passes |
| `launch_viz.py` | 210 | figures |
| `test_launch_ready.py` | 140 | 10 tests |

The full diff against the baseline copy is in `audit/code_changes.diff`. `launch_runner.py`, `launch_check.py` and
`placement_teacher.py` are 293–318 lines, above the 280-line guideline.

**Storage per root.**
- `records/<root>.json` holds the header (root, split, group, hashes, policy, `privileged_teacher.verified_class`) and the compact
  query records.
- `modes/<root>.npz` holds the 48×48 stage and category codes plus per-attempt arrays (rank, method, status, stop, counted work,
  verified, stored, witness id).
- `witnesses/<root>.npz` holds the packed carry → release → retreat arrays.
- Every verified pair's content id is recorded; stored witnesses are packed and re-certified by the audit. Unstored ones can be
  regenerated exactly (fixed seed and counted work).

## 5. Pilot design (pre-registered: `configs/pilot_policy_selection.json`, sha256 `5e731318…`, read-only before any call)

- **Roots:** 25 train roots, 5 per class, from distinct untouched lineage groups. Clutter caps 2/2/1 per class; 15 server-made and
  10 laptop-made roots, 7 of them conversions.
- **Queries per root:** 12 teacher queries (2 objects × 4 root destinations, plus 2 × 2 prefix queries), plus static contrasts.
  That is 300 teacher queries and 47 contrasts.
- **Reference:** 64 pairs with FULL coverage. Budgets 8/16/32 are read off as nested slices with exact accounting, and checked
  against an independently executed 8-pair run.

## 6. Pilot results (64-pair FULL reference, run 1; train roots only)

Of 300 completed queries, 265 are label 1 and 35 label 0. The 35 zeros break down as:
- 14 `CERTIFIED_SCREEN` (every pair definitely screened);
- 8 `NO_PLAN_FOUND` (no valid pair, screens inconclusive);
- 9 `SCHEDULE_EXHAUSTED_NO_WITNESS`;
- 4 `WORK_BUDGET_EXHAUSTED`.

The 47 static rows are all `DEST_OCCUPIED` and unlabeled. 13,829 witnesses are stored and re-certified.

| Pairs | Label 1 / 0 | 64-pair positives missed (Wilson 95%) | 1→0 flips | Pair coverage | CPU-s per query, FULL mean (p95) | CPU-s per query, FIRST_SUCCESS mean |
|---|---|---|---|---|---|---|
| **8** | 263 / 37 | 2/265 = 0.75% (0.21–2.71%) | 0 | 2.6% | 10.9 (44.3) | 4.5 |
| 16 | 263 / 37 | 2/265 = 0.75% | 0 | 5.1% | 20.8 (85.9) | 6.5 |
| 32 | 265 / 35 | 0 (0–1.43%) | 0 | 10.0% | 41.4 (155.7) | 9.4 |
| 64 | 265 / 35 | reference | — | 19.0% | 76.3 (261.3) | 12.4 |

**Strata at 8 pairs (missed / 64-pair positives).**
- By destination: near 2/135, mid 0/44, far 0/40, vacated 0/46.
- By block: root 0/172, prefix1 1/47, prefix2 1/46.
- All are within the 5% stratum limit. Both misses are near-destination prefix queries that 32 pairs rescues, so the rescue rate is
  2 of 37 zeros.

**Rule outcome:** 8 pairs is the smallest budget that passes.

**Label 0 at 8 pairs:**
- 37 queries, from 12 roots in 12 lineage groups and all 5 classes (NONE 11, K0 1, K1 12, K2 2, K3 11), covering 18 distinct
  objects.
- Kinds: 14 certified screens, 8 no-plan, 13 work-budget-exhausted, 2 schedule-exhausted.
- Only 15 of the 37 are planning-limited. Negatives are about 12% of queries.

**Labels by block at 8 pairs:** root 172/28, prefix1 46/4, prefix2 45/5. By destination: near 133/17, mid 44/9, far 40/7,
vacated 46/4.

**Mode level.**
- Attempted-pair success rate: 0.91 for ranks 0–7, falling to 0.87 for ranks 32–63.
- Failed attempts by cause: 1,182 hit the attempt limit, 543 the node limit, 137 had a certified link-0 sweep blocker.
- Held modes valid: 4,252. Pick-access modes valid: 3,768 (diagnostic only).
- 8 queries with zero pick access are still label 1, which confirms the held-state semantics.
- 4 queries reuse a re-verified archived suffix (`label_source` = BOTH). 46 pilot queries sit in an archived-move context.

**Interpretation.** A 64-pair result is a pilot reference, not physical truth: a 64-pair zero remains policy-limited while
surviving pairs stay untried (at 64 pairs, coverage is 19% of valid pairs). Mode-success fractions describe coverage over a
truncated catalogue; they are not calibrated probabilities.

## 7. Determinism, nesting, resume, audits

**Fingerprints.**

| Run | Fingerprint | Queries |
|---|---|---|
| `pilot_b64_w25_r1` | `6eb544df…` | 347 |
| `pilot_b64_w25_r2` | `6eb544df…` (identical) | 347 |
| `pilot_b64_serial_first` | 25/25 identical to both 25-worker runs | 25 |
| `pilot_b8_w25` | `85483e7c…` | 347 |
| `pilot_b8_w25_interrupt` | `85483e7c…` (identical) | 347 |

- **Nested budgets:** the independently executed 8-pair run equals the first-8 slice of the 64-pair run on 300/300 queries:
  per-attempt pick, place, method, status, stop, counted work, verified flag and witness id, plus label and negative kind.
- **SIGKILL + resume:**
  - The coordinator was killed after 8 roots were packed; 0 workers survived (they die with the coordinator).
  - 24/24 packed files were byte-identical after the resume.
  - 49/49 committed parts reappeared unchanged inside their packed roots.
  - 188 unfinished queries were re-run, and the final fingerprint equals the uninterrupted run.
- **Safety cancels:** run 1's two thermal cancels recorded 19 and then 4 in-flight queries as `completed=false`, `label=null`.
  They re-ran identically; r1 = r2.
- **Audits:** 0 failures across r1 and r2 (44,137 checks each), serial (4,253), 8-pair (8,565) and interrupt (8,565). Checks
  include:
  - witness content ids;
  - phase-by-phase re-certification;
  - the disjoint 48×48 partition matching the arrays;
  - ranks and coverage rule;
  - the work limits;
  - protected fields appearing only under `privileged_teacher`;
  - scene hashes.
- **Tests:**
  - `test_launch_ready.py` 10/10 and `test_placement_smoke.py` 9/9 pass.
  - The dataset's own tests give 6 passed, 1 skipped, 1 failed. That failure is the pre-existing missing `runs/p16` fixture, as in
    the smoke report; nothing new.

## 8. CPU, affinity and thermals (25 workers out of 32 logical CPUs)

**Affinity.**
- Workers: 0–15 (one thread per core), then 16, 24, 17, 25, 18, 26, 19, 27, 20 (SMT siblings alternating CCDs).
- Reserved: 21–23 and 28–31 (3 on CCD0, 4 on CCD1). The coordinator runs on the reserved set.
- Each worker is pinned to one CPU at nice 10, with `OMP`, `MKL`, `OpenBLAS`, `NumExpr` and `vecLib` threads all set to 1.
- `hard_cpu_quota=false`: the flatpak sandbox has no cgroup filesystem and no `systemd-run`. Affinity is not a quota.

**Policy.**
- Pause (SIGSTOP all workers) at ≥ 88 °C; resume at ≤ 80 °C; cancel immediately at ≥ 95 °C.
- Cancels also on sensor loss, cooling stall, more than 90% paused over 10 min, low memory, swap growth, low disk, worker RSS, a
  process leak, a dead worker, or AC loss.
- Per-query CPU-time guard: 1,200 s. Frozen workers do not advance it.

| Run | Sessions (stop) | Wall | Worker CPU | Pauses (time paused) | Peak / median Tctl | Min MemAvailable | Swap growth | Max worker RSS |
|---|---|---|---|---|---|---|---|---|
| 64 pairs, r1 | 3 (TEMPERATURE, TEMPERATURE, none) | 1,927 s | 25,428 s | 171 (686 s) | 95.4 / 85.5 °C | 10.1 GB | 0 | 127 MB |
| 64 pairs, r2 | 1 (none) | 1,607 s | 23,170 s | 133 (569 s) | 92.6 / 86.1 °C | 9.3 GB | 0 | 125 MB |
| 64 pairs, serial | 1 (none) | 1,169 s | 499 s | 232 (660 s) | 88.9 / 84.0 °C | 12.3 GB | 0 | 125 MB |
| 8 pairs | 1 (none) | 206 s | 3,202 s | 12 (58 s) | 90.5 / 80.3 °C | 12.0 GB | 0 | 129 MB |
| 8 pairs, interrupt resume | 1 (none) | 304 s | 1,880 s | 46 (210 s) | 91.6 / 83.3 °C | 12.1 GB | 0 | 129 MB |

**Steady full load is safe.** In r2, while the queue was non-empty and all 25 workers were busy (863 s, a 14.4-minute soak):
- peak 87.1 °C, median 86.25 °C;
- **0% of the time paused**;
- 24.83 worker CPU-s per second.

The CPU's power limit caps all-core frequency, so full load runs *cooler* than partial load (`audit/figures/thermal_soak_25_workers.png`).

**The hazard is partial load.** With few busy cores, each core boosts towards 5.39 GHz (`boost=1`, EPP `balance_performance`):
- A single core reached 90 °C during the single-threaded preflight.
- The serial reference spent 56% of its time paused.
- Run tails (queue empty, 1–3 workers left) spent 77% of their time paused, peaking at 92.6 °C.

**What caused run 1's two cancels.**
1. **Cancel at 917–933 s (tail).** At 917 s the coordinator froze all workers at 88.1 °C. CCD temperatures fell to about 70 °C, then
   CCD0 climbed to 91.6 °C and Tctl crossed 95 °C while every worker was stopped. The heat came from host processes outside the
   sandbox, which the runner cannot see (Codex was writing on `/home` during the session). My own session ran nothing but two
   polling loops then.
2. **Cancel at 2.3 s (resume start-up).** The staggered start (2 tasks per 0.25 s) gave a few boosting cores, which took Tctl from
   73 to above 95 °C in 2.3 s.

**Fix already applied.** Workers now start together (`dispatch_per_sample` 2 → 25, recorded in `configs/frozen_hashes.json`
`launch_config_history`). Run 2 started with Tctl *falling* from 80.5 to 73 °C. The remaining exposure is tails and rare pauses.

**Options for you to choose from** (no protection was relaxed):
- **(a) Recommended:** turn off CPU boost on the host for the run (`echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost`;
  reversible), or set EPP to `balance_power`. Then re-validate with two 25-worker runs (RUNBOOK §0).
- **(b)** Approve a supervised automatic resume after a thermal cancel (cool to ≤ 75 °C, then resume; each cancel costs only the
  in-flight work).
- **(c)** Amend the cancel rule, for example require ≥ 95 °C to be sustained for ≥ 1 s.

Fewer workers would not help: lower parallelism means more boost.

## 9. Selected policy, forecast and storage

**Proposed production policy** (not frozen while blockers stand):
- 8 pairs per query, FULL coverage, witness store maximum 1 (every verified pair's id is still recorded), archived-suffix reuse on.
- Per-pair work fixed at 1,000 attempts / 20,000 evaluations / 200 nodes / 150 shortcut rounds.
- Pair protocol `bb41a56b5deb2ff8`.

**Forecast.**
- CPU per query comes from the 25-worker run. Throughput is 24.83 worker CPU-s per second (r2 steady state).
- Disk is the pilot roots re-packed exactly as production packs them: 3.2 KB per row (2.4 KB per witness).
- Per-query cost cross-check: the independent 8-pair run averaged 10.7 CPU-s per teacher query.

| Coverage / quota | Teacher queries | Worker CPU-h | Wall days (steady) | Disk |
|---|---|---|---|---|
| **FULL / Q1 (preferred)** | 308,652 | 1,222 | 2.05 | 990 MB |
| FIRST_SUCCESS / Q1 | 308,652 | 515 | 0.86 | 925 MB |
| FULL / Q2 | 205,768 | 783 | 1.31 | 668 MB |
| FULL / Q3 | 114,556 | 441 | 0.74 | 383 MB |
| FIRST_SUCCESS / Q4 (only entry within 215 MB) | 53,752 | 67 | 0.11 | 181 MB |

- **Upper bound for FULL / Q1:** if the run were paused as much as r2 overall (35%), wall time is about 3.2 days. All entries fit the
  7-day limit.
- **Disk needed for FULL / Q1:** about 2.6 GB free at decision time (990 MB plus the 1,536 MB reserve). The runner cancels below
  1,024 MB free.
- **Dry run:** the metadata-only preflight over all 5,000 roots passed 13/13 checks in 10.5 s with 0 planner calls:
  - all selected roots and every root-file hash match;
  - the plan recomputes from the dataset; split and config hashes match;
  - the CPU sets match; sensor readable; AC on; memory headroom OK.

  The full task list builds in 2.7 s (317,037 rows, 399 MB in the coordinator). `gc.freeze()` before forking keeps it shared
  copy-on-write.

## 10. Exact future launch command (DO NOT EXECUTE NOW; it refuses until RUNBOOK §0 is done)

```bash
cd /home/naren/Desktop/rss-placement-smoke/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v1
./launch_full_2d.sh --execute-full --confirm <token printed by: python ../code/launch_runner.py token .>
```

- **Token.** It covers the query plan, split, launch config and production policy hashes, and changes if any of them changes. It is
  not shown here because the production policy is not frozen yet.
- **Commands.** Pause, resume, drain, cancel, status, resume-after-crash, audit and figures are in `RUNBOOK.md`.
- **Guard checks.** Its Python refusal paths are tested: no flags, a missing `--execute-full`, a wrong token, and the correct token
  without a frozen policy. The tests also assert no planner call and no `full_2d_v1/` folder.
- **Shell wrapper.** The permission classifier did not let me run `launch_full_2d.sh` itself. Run it once with no arguments to see
  it refuse.

## 11. Deviations and remaining risks

**Deviations during the pilot** (none touches labels):
- **Runner code changed between run 1 and the later runs:** monitor CPU fields, a deque task queue, `gc.freeze`, the guard split out
  into its own module, and cached witness-pack reads. Teacher code was unchanged, and r1 = r2 fingerprints.
- **Launch config `dispatch_per_sample` changed after run 1.** Both config hashes are recorded.
- **Pilot harness resumes:** run 1 needed two resumes after safety cancels, each logged in `pilot/sequence_r1_resume.log` and its
  `run_*.json` files.
- **Disk model in the decision:** switched from per-query compression to per-root packing, matching the rule's "measured packed
  bytes". The verdict is the same under both.

**Remaining risks:**
- Labels are policy-limited. At 8 pairs, 0.75% of 64-pair positives are missed (upper Wilson bound 2.7%).
- Planning-limited negatives are few (15 of 300 at 8 pairs). Calibrating a confidence head will lean on them and on screen-certified
  negatives.
- Archived-move contexts are chosen with protected metadata. They are flagged so training can drop or ablate them.
- T's goal cell is excluded as a destination: a grammar choice that you may want to revisit.
- Splits only merge proven sharing.
- There is no hard CPU quota.
- The coordinator packs roots synchronously (sub-second gaps in sampling).
- Results are for this machine and software stack only.

## 12. Frozen files and checksums

| File | sha256 |
|---|---|
| `split_manifest.json` | `f7e8cbba…` |
| `query_plan.jsonl.gz` (canonical content) | `d078612b…` |
| `pilot_roster.json` | `22850b9a…` |
| `configs/pilot_policy_selection.json` | `5e731318…` |
| `configs/launch_config.json` (current) | `0f2cd8ed…` |
| `configs/launch_config.json` (used by run 1) | `f90eb695…` |

Budget hashes are `07cf09e5…` for 64 pairs and `c794a164…` for 8 pairs.

Every file in this folder is listed in `audit/checksums.sha256`, apart from transient parts and the checksum file itself. The
original-data and smoke baselines are in `audit/baseline/`.

**Not done, by instruction:** the full collection, any val/test teacher call, new scenes, 3D, cVAE / flow / X / Y training, remote
machines, commit or push. **STOP:** waiting for your decision on the blockers and for your explicit authorization of production.
