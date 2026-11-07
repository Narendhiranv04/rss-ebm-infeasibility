# Final 2D placement mock, v2: readiness report

**Date:** 7 Nov 2026.
**Worktree:** `/home/naren/Desktop/rss-placement-ready-v2`, detached at `16d1b3b4d565ae89b833141e136c71157823e1dc` (branch
`naren/2d-dataset-relocation`), sparse checkout of `planar_2d`.
**Contract:** `INSTRUCTIONS.md`, `CLAUDE_KICKOFF.md`. **Scientific authority:** `final-fixed-methodology-1.pdf`.
**Commit status:** uncommitted, not pushed.

## Verdict: `NOT_READY_TO_LAUNCH`

**Exactly one gate fails: `RUNTIME_FORECAST`.**
- Q1 (357,240 teacher queries) is forecast at **7.62 days** with the pre-registered 1.5× buffer, against a 7-day limit.
- The forecast uses the measured boost-off throughput and is weighted by Q1's actual query mix.
- The expected (unbuffered) time is 5.08 days.

Every scientific, reproducibility, safety, storage and integrity gate passes (§9).

**The full collection was not started.**
- No production policy is frozen; `configs/production_policy.json` does not exist.
- The guard refuses every launch.
- The production planner was never called, and `full_2d_v2/` does not exist.

**A first `decide` run reported `READY_TO_LAUNCH`, and I did not accept it.**
- Its forecast scaled the pilot's *mean* CPU per query to Q1 (5.56 days with the buffer).
- The pilot is dominated by cheap near and vacated destinations (13 CPU-s per query for root/near), whereas Q1 is mostly root
  mid and far destinations (41.8 and 35.5 CPU-s).
- Review fix R5 takes the larger of the pilot-mean and Q1-composition-weighted estimates, and the gate then fails.
- That run's decision file and frozen policy are kept, unused, in `audit/superseded/`. The policy hash was cleared from
  `configs/frozen_hashes.json`.

**Your decision** (RUNBOOK_V2 §1):
- **(a) Production quota Q2:** 238,160 queries; 3.25 days expected, 4.87 days with the buffer. This passes.
- **(b) Keep Q1 under a raised time limit:** 7.62 days buffered; resampling the pilot roots gives a 95th percentile of 6.76 days
  unbuffered.

Either is an explicit amendment of the pre-registered rule. It needs no new physical runs: `decide` is re-run on the same frozen
evidence.

## 1. Source, versions and hashes

| Item | Value |
|---|---|
| Source commit | `16d1b3b4d565ae89b833141e136c71157823e1dc` (clean worktree; only the new `code_v2/`, `launch_ready_2d_v2/`) |
| Pair protocol id (pins teacher sources, smoke code, **adapter source**) | `938b09a23a581152` |
| `placement_teacher.py` (adapter) | `b39fd1b742727dfc…` |
| `placement_smoke.py` | `fef218feb5ea7a25…` (unchanged) |
| Teacher config / robot | `90d3eb68e8a42108` / `PLANAR3R_SIDEGRASP_V1:sha256:2d798676…` |
| `configs/launch_config.json` | `521b5e48efd2676a…` |
| `configs/pilot_policy_selection.json` (pre-registered before any v2 call) | `5fb0073fb11c09cb…` |
| `split_manifest.json` | `e253b6233c6fc752…`. Assignments are byte-identical to v1; only the version label differs |
| `query_plan.jsonl.gz` (canonical content) | `724268596b12889b…` |
| `pilot_roster.json` | `8b472cb680ca38c9…` |
| `selected.json` (5,000 roots) | `edb8dfca60cb7142…` (unchanged) |
| Budget hash, ADAPTIVE_FULL 8→32 | `0cd2597b0d9141fb` |
| Budget hash, 64-pair FULL reference | `3fa07b2b107d0598` |

Per-file hashes are in `audit/code_v2.sha256`. The `code` → `code_v2` diff is in `audit/code_v2_changes.diff` (630 lines).

## 2. What changed (patch candidate plus documented review fixes)

**Applied from the delivered `prepare_v2.py`:**
- **No target-goal veto.** T's goal cell is a destination whenever it is legal and empty.
- **No archived context.** `prefix2` is now a geometry-designed edit; no archived first move selects a context.
- **New two-edit context.** `prefix3` makes two sequential, distinct, legal object-only edits. The queried object has not moved;
  the edits are recorded `HYPOTHETICAL`, and the scene is root-consistent.
- **ADAPTIVE_FULL schedule.** Plan the first 8 valid pairs in the frozen order; only if none verifies, continue in the same order
  to 32.
- **Witnesses.** Every distinct verified witness within the attempted pairs is stored (up to 32), and archived reuse is off.
- **Pinned adapter.** The adapter source hash is part of the pair protocol.
- **Guard.** The full-mode guard adds checks for boost off, at least 10 GiB of disk, `reuse_archived=false`, and storage of all
  verified witnesses.

**Review fixes**, each found by running the package against the real source (`audit/package/review_fixes_*.diff`; the delivered
and reviewed packages are both saved):

| Fix | Problem | Correction |
|---|---|---|
| R1 | `prepare_v2.py` failed closed: its plan-prefix anchor also matched `QUOTAS["Q1"]` | Anchor on the `PLAN` line's own indentation |
| R2 | With up to 32 stored witnesses, two pairs with identical arrays would repeat an id in `witness_refs` (audit failure) | Content-deduplicated storage. It never triggered: 0 duplicates |
| R3 | The gate's safety-stop check globbed `run_*.json`, which includes `run_manifest.json` (no `stop` key), so the gate would crash with `KeyError` | Glob `run_[0-9]*.json` |
| R4 | The gate's `pytest` call wrote `.pytest_cache` into `code_v2` | `-p no:cacheprovider` |
| R5 | The runtime and disk forecast scaled the pilot mean, which is biased low because the pilot mix is near/vacated-heavy | Use the maximum of pilot-mean and Q1-composition-weighted, both ×1.5 |

**Added:**
- `code_v2/v2_report.py` reads saved files only and produces the evidence tables and figures. It is not part of the teacher.
- The pre-registration file now records the exact v2 decision rule and run list, frozen by `init` before any pilot call.

## 3. Splits, lineage and query plan

**Splits.** Identical to v1 for all 5,000 roots: train 3,500, val 751, test 749. 2,431 lineage groups (family, identical layout,
conversion origin), with 0 shared across splits.

**Q1 teacher queries:** 357,240 (v1: 308,652); per split, train 243,726, val 56,700, test 56,814. There are also 11,734
static-ineligible contrast rows, which are never labelled.

**Context blocks:**
- 103,128 one-edit design contexts and 48,588 **two-edit** contexts;
- 0 contexts chosen with protected metadata;
- mean 71.4 teacher queries per root (maximum 84); 380 obstacle-free roots have none.

**Quota ladder (Q1–Q4)**, all containing the pilot:

| Quota | Teacher queries |
|---|---|
| Q1 | 357,240 |
| Q2 | 238,160 |
| Q3 | 131,296 |
| Q4 | 62,122 |

**Static audit** (no teacher calls) passes:
- Q1 rows are unchanged when T's goal cell is altered, and unchanged when roles, blockers, parking, construction and every label
  are scrambled. Checked on 105 roots (pilot plus audit subset).
- 1,056 two-edit queries appear in that subset.
- Plan, split and protocol hashes match; baseline integrity passes; preflight made 0 planner calls.

**Pilot:** the same 25 train roots as v1, giving 346 teacher queries plus 71 static contrasts. **0 val/test teacher queries.**

## 4. Labels against the 64-pair reference (all 346 pilot queries)

| Policy | Label 1 / 0 | 64-pair positives missed | Wilson 95% | Positive here but 0 at 64 |
|---|---|---|---|---|
| Fixed 8 pairs (first 8 of the reference) | 299 / 47 | 2 / 301 (0.66%) | 0.18–2.39% | 0 |
| **ADAPTIVE 8→32 (canonical, measured)** | **301 / 45** | **0 / 301** | 0–1.26% | 0 |
| Fixed 32 pairs (first 32 of the reference) | 301 / 45 | 0 / 301 | 0–1.26% | 0 |
| 64 pairs (measured reference) | 301 / 45 | reference | | |

**Where 8 pairs falls short.**
- Both of its misses (`71fb2340…`, `df0fb602…`) are in prefix contexts at near or vacated destinations, in class NONE roots.
- The adaptive schedule escalated 13 queries to 32 pairs and rescued exactly those 2.
- 47 queries have 8 or fewer valid pairs.

**By stratum,** adaptive misses are 0 everywhere:

| Stratum | 64-pair positives |
|---|---|
| root | 172 |
| prefix1 | 46 |
| prefix2 | 42 |
| prefix3 | 41 |
| near | 152 |
| mid | 43 |
| far | 40 |
| vacated | 66 |

**Labels by destination (adaptive, label 1 / 0):** near 152/21, mid 43/10, far 40/7, vacated 66/7.

**Negatives under the adaptive policy (45):**
- 16 `CERTIFIED_SCREEN` (every pair fails definite static screens);
- 29 operational: 13 `NO_PLAN_FOUND` (no valid pair, screens inconclusive), 8 `WORK_BUDGET_EXHAUSTED`, 8 `SCHEDULE_EXHAUSTED_NO_WITNESS`;
- spread: 13 roots in 13 lineage groups; classes NONE 9, K0 2, K1 14, K2 3, K3 17; blocks root 28, prefix1 4, prefix2 8, prefix3 5.

This meets the minimum diversity gate; it is *not* a calibration claim.

**T's goal cell is now included:** 8 pilot queries place an object on it, and 5 are label 1.
**Unpickable but placeable:** 2 queries have zero pick-access modes yet a verified placement.

**Rank-wise success in the 64-pair reference** (fraction of planned pairs verified):

| Ranks | Rate | Verified / planned |
|---|---|---|
| 0–7 | 0.898 | 2,216 / 2,468 |
| 8–15 | 0.884 | |
| 16–31 | 0.851 | |
| 32–63 | 0.824 | |

These are coverage statistics over a truncated catalogue, not calibrated probabilities. A 64-pair result is not physical truth.

## 5. Witnesses and replay

| Run | Verified pairs | Distinct ids | Packed | Dedup removed | Per query | Bytes per query | Bytes per witness |
|---|---|---|---|---|---|---|---|
| Adaptive A and B, interrupt resume (each) | 2,223 | 2,223 | 2,223 | 0 | 6.42 | 9,138 | 1,059 |
| 64-pair reference | 14,976 | 14,976 | 14,976 | 0 | 43.3 | 44,139 | 914 |

Every stored witness is re-certified phase by phase (carry → release → retreat; attached exactly while carrying) by the audits.
`n_verified_pairs` equals the set of stored witness references in every query (gate `all_verified_witnesses_stored`).

**Figures** (`audit/figures/`):
- `v2_examples.png`: positive in the root scene; negative with its budget reason; destination on T's goal cell; two-edit context
  with lineage arrows; no pick-access mode but placeable.
- Real witness GIFs plus filmstrips for 4 of those cases.
- `v2_*_mode_pairs_escalated.png`: 48×48 matrix of a query escalated to 32 pairs.
- `v2_rank_success.png`.
- `v2_adaptive_vs_ref64.png`: recall by stratum, plus CPU-s distributions.
- `v2_thermal_25_workers.png`.

## 6. CPU, affinity and thermals (boost OFF, verified at mock start and at decision time)

**Configuration:**
- Workers on CPUs 0–15, 16, 24, 17, 25, 18, 26, 19, 27, 20 (one thread per core first, SMT siblings alternating CCDs).
- Reserved CPUs: 21–23 and 28–31.
- Each worker runs at nice 10 with one BLAS/OpenMP thread. `hard_cpu_quota=false`: affinity is not a cgroup.

| Run | Sessions / stops | Wall | Worker CPU | Pauses | Peak / median Tctl | Steady throughput |
|---|---|---|---|---|---|---|
| 64-pair reference, 25 workers | 1 / none | 1,811 s | 38,294 s | 0 | 81.0 / 69.1 °C | 24.85 CPU-s/s |
| Adaptive A, 25 workers | 1 / none | 470 s | 7,559 s | 0 | 66.4 / 63.3 °C | 24.36 CPU-s/s |
| Adaptive B, 25 workers | 1 / none | 502 s | 7,575 s | 0 | 64.4 / 61.7 °C | 24.35 CPU-s/s |
| Serial, first query per root | 1 / none | 80 s | 71 s | 0 | 59.4 °C | |
| Interrupt resume, 25 workers | 1 / none | 356 s | 6,061 s | 0 | 66.1 °C | 24.55 CPU-s/s |

- **Resources:** minimum MemAvailable 11.06 GB, 0 swap growth, worker RSS ≤ 140 MB, 25 descendant processes.
- **Disk:** at least 61 GB free throughout.
- **Thermals:** startup, steady and tail phases are all far below the 88 °C pause line. The 30-minute reference run is the longest
  soak.
- **Effect of boost off:** each CPU-second is about 2× slower than with boost on (tests took 208 s vs 101 s).
- **Not shown:** multi-day thermal stability is not demonstrated.

## 7. Reproducibility

| Check | Result |
|---|---|
| Adaptive A vs B (25 workers each) | identical per-query fingerprints, 417/417 rows, run fingerprint `5787aaa6…` |
| Serial first-per-root vs A | 25/25 identical |
| SIGKILL + resume vs A | 417/417 identical |
| Adaptive vs exact nested 8→32 slice of the 64-pair reference | 346/346 identical (labels, pairs planned, per-attempt pick/place/method/status/stop/counted work/verified/witness id) |
| SIGKILL test | coordinator killed after ≥ 5 packed roots; 0 surviving workers; 15 committed packed files byte-identical after resume; resume audit 0 failures |
| Audits | 0 failures: reference 48,052 checks, A / B / interrupt 9,793 each, serial 800 |
| Tests | 25/25 pass (6 v2 contract, 10 launch-ready, 9 smoke), run twice (after metadata init, and again inside the mock gate) |

## 8. Forecast (Q1, pre-registered 1.5× buffer, measured boost-off throughput 24.36 worker CPU-s/s)

| Estimator | Worker CPU-h ×1.5 | Wall days ×1.5 | Disk ×1.5 + 0.5 GB |
|---|---|---|---|
| Pilot mean (first decide; biased) | 3,251 | 5.56 | 5.17 GB |
| **Q1-composition weighted (R5; decisive)** | **4,454** | **7.62** | 4.65 GB + 0.5 |

- **Unbuffered composition-weighted:** 2,969 CPU-h, 5.08 days.
- **Root-resampled, unbuffered:** median 5.03, p95 6.76, p99 7.54 days.
- **Disk:** passes with a wide margin (61 GB free; the gate needs at least 10 GiB).
- **Extrapolation caveat:** the pilot has no prefix-block mid or far destinations. Those 86,438 Q1 queries use the root mid/far
  per-query cost (`MOCK_V2_DECISION.json` → `forecast.estimators.strata`).

**Other quotas,** same estimator (`audit/quota_forecasts_composition_weighted.json`):

| Quota | Teacher queries | Days expected | Days ×1.5 |
|---|---|---|---|
| Q2 | 238,160 | 3.25 | 4.87 |
| Q3 | 131,296 | 1.80 | 2.70 |
| Q4 | 62,122 | 0.65 | 0.97 |

## 9. Gates

| Gate | Result |
|---|---|
| Split identity with v1; complete lineage groups; 0 val/test teacher queries | PASS |
| Goal-independent and protected-metadata-independent candidates; two-edit contexts present | PASS |
| Audits of all 5 runs (48×48 accounting, phase-aware witness replay, schedule, hashes) | PASS |
| Zero safety stops (reference, A, B, serial, interrupt resume); boost off; thresholds unchanged | PASS |
| A = B fingerprints; serial = A; interrupt resume = A | PASS |
| Adaptive equals the exact nested slice of the new 64-pair reference | PASS |
| Recall vs 64: overall miss ≤ 2%, strata ≤ 5% | PASS (0/301) |
| Negative diversity (≥ 10 negatives, ≥ 5 groups, ≥ 3 classes) | PASS (45 / 13 / 5) |
| All verified witnesses stored | PASS |
| Disk ≥ max(10 GiB, forecast + 1.5 GiB) | PASS |
| **Q1 wall ≤ 7 days (×1.5, composition weighted)** | **FAIL (7.62)** |
| Launch guard: wrapper with no arguments refuses (exit 2); Python guard refusals tested; full metadata preflight = 0 planner calls | PASS |
| Original dataset, both smoke stores, v1 code and `launch_ready_2d_v1` unchanged | PASS (0 changed, 0 added; git clean) |

The full metadata preflight passes every infrastructure check, including boost off, disk and no conflicting run. It fails only on
the three production-policy checks, which is correct while no policy is frozen.

## 10. Remaining risks and caveats

- **Policy-limited labels.** Labels hold under ADAPTIVE 8→32 with fixed per-pair counted work. A 0 is not physical impossibility,
  and the 64-pair reference is not ground truth.
- **Rank-wise success is not calibration.** Any sampler confidence must be calibrated separately.
- **Forecast uncertainty.** The forecast extrapolates prefix-block mid/far cost from root mid/far. Throughput was measured for at
  most 30 minutes.
- **Multi-day operation is unproven.** Thermal behaviour over days, and host activity, are not demonstrated. With boost off the
  margin is large (peak 81 °C against the 88 °C pause line).
- **Power configuration.** Production must run with boost off, and the full-mode guard requires it. Turning boost back on is your
  host action; do it after any production run, or now if no run follows soon.
- **Code size.** `code_v2` modules are 106–324 lines; three exceed the 280-line guideline: `placement_teacher.py` 302,
  `launch_check.py` 320, `launch_runner.py` 324.
- **Nothing is committed or pushed.** v2 lives in the detached worktree above as untracked files (42 MB).

**STOP.** No full collection, training, 3D work, commit or push. Waiting for your decision on §1 (Q2, or a raised limit for Q1) and
your explicit authorization.
