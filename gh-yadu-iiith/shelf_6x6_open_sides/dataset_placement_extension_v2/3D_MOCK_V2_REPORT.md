# SHELF66V3 3D placement mock v2: closure report

| Verdict | Status |
|---|---|
| Mock | **`MOCK_V2_PASS`** |
| Live source | **`LIVE_SOURCE_SYNC_PASS`** (consistent IN-PROGRESS snapshot) |
| Production | **`PRODUCTION_READY_PENDING_AUTHORIZATION`** |
| Full collection | **`FULL_COLLECTION_NOT_STARTED`** |

Host `nav@gvlab` (Xeon Gold 5317, 12 cores / 24 logical CPUs), 2026-11-07. Every number below comes from files in this directory. `3D_MOCK_V2_DECISION.json` is generated mechanically from them by `code/decide_v2.py`.

**What the pass means.** The branch-aware placement teacher and the hardened collector are semantically correct, deterministic, physically re-verified and safe on a bounded mock of 42 roots and 405 physical queries, under finite policies.

**What it does not mean:**
- The source dataset is not final: K4 has 0 of 510 selected roots, NONE has 3 of 408, and 488 roots still await the generator author's fixed-object check.
- Not every possible grasp configuration was searched.
- Zero observed misses is not zero risk.
- Production is not authorized.

## 1. What was wrong in v1 and what changed

Auditing the v1 code (`../dataset_placement_extension_v1/code/policy3d.py`) confirmed three IK-branch defects:
1. **Screening:** the physical hold was certified for **branch 0 only**. If that failed, the whole grasp pose was discarded as `HELD_INVALID_PHYSICAL`.
2. **`run_mode`:** only the branch that **planned first** was held and executed. A hold failure (`held_state_lost`) or an execution failure ended the unit with no other branch tried.
3. **The geometric certificate** also certified only branch 0's hold, so a pose could become unlabeled instead of certified.

The v1 records show these defects actually fired. In pilot A, 3 multi-branch poses were rejected on branch 0 and 2 units ended `held_state_lost`.

**v2 (teacher `SHELF66V3_PLACE_V2`, versioned sidecar; v1 byte-unchanged):**
- **Branch inventory.** Each ranked pose lists its source IK candidates (at most 4, structure- and ceiling-clear, in stable order nearest `q_ready`). Each candidate gets an eligibility test: more than 5 mm from other objects with the fingers pre-shaped.
- **Lazy hold certificates.** Every eligible branch gets its own physical hold certificate, tested in candidate order and cached. A hold is a deterministic function of the restored scene, pose and branch, and a test confirms that.
- **Unit = pose.** A unit is a pose with at least one held branch. Branches are never counted as extra units, and every attempt is charged to its pose with branch-level accounting.
- **Unit test.** Each eligible branch is tried in order: hold certificate → `plan_branch` → re-instantiated hold (it must reproduce the certificate exactly) → MuJoCo execution → 13 gates. Early-stop policies stop at the first verified branch; the reference tries every branch.
- **Shared route.** The carry route is planned once per unit test and shared by its branches. `plan_carry_xy` has no branch argument and runs on the restored scene with a fixed seed; `test_route_is_branch_independent` checks that routes planned from different branches' contexts are identical. A route `ReachError` ends the unit, as in the source's `plan_pick_place`.
- **Scoped certificate.** `NEG_CERTIFIED_GEOMETRY` now requires:
  - all 32 poses screened;
  - only footprint-overlap geometry reasons;
  - every eligible branch of every `HELD_INVALID_PHYSICAL` pose hold-tested.

  It is scoped explicitly: "32 SHELF66V3 level grasp poses × the source's ≤ 4 fixed-seed IK candidates × the source 2-D hand-footprint carry model (not a global impossibility proof)".
- **Witness preservation.** Every distinct verified (unit, branch) trajectory met within the attempted policy is stored with its provenance. That means the rank, grasp id, azimuth, level and branch in the witness `branch` array, the unit `witness_index`, and each record's `witnesses` list.
  - Content-identical trajectories are stored once, and each query keeps its own link. Eight such pairs exist: the same object and destination in a 1-edit scene and its 2-edit child.
  - Early-stop policies stop at the first positive, as specified; no extra modes are run to inflate witness counts.
- **Collector `run3d.py` v2:**
  - per-unit write order NPZ → JSON (holding the NPZ SHA-256 and every witness id) → coordinator re-validation → ledger line;
  - a resume accepts a unit only if the ledger, JSON hash, NPZ hash and every witness re-hash agree; everything else is quarantined and recomputed;
  - a per-root source pin is frozen into the run manifest and checked again in each worker;
  - code, roster, protocol, worker policy and source pin are compared at every resume;
  - an unreadable temperature sensor means refuse or stop, even while paused, and the sensor identity is logged;
  - the monitor checks the affinity union of the whole own process tree (at most 18), descendant identity (workers plus multiprocessing's resource tracker only), CPU, RSS, memory, swap and disk;
  - production (all splits) runs only through `guard3d.py`.
- **Amendment.** `AMENDMENT_V2_001`, registered before any result, fixed a monitor defect found at the first reference launch. The CPU % always read 0.0 because process objects were re-created every tick; the affinity check was not affected. That launch was killed after 38 s with 0 units done and is kept in `mock_v2/runs_aborted/`. No label or rule changed.

## 2. Frozen design (before any v2 teacher call on mock queries)

| item | value |
|---|---|
| Selection rule | `mock_v2/configs/selection_rule_v2.json` (sha `e0228c0b…`), a **new** rule. The v1 rule and its literal choice `fixed_4x1500` stay on record unchanged |
| Run plan | `run_plan_v2.json` (`8051eb51…`) |
| Source pin | `source_pin.json` (`ca5692a5…`): 1,492 roots of `yadu/dataset-3d@8032ad0b`, equal to the v1 manifest |
| Roster | `roster_v2.jsonl.gz` (`d032659c…`): **42 train roots, 405 physical queries**, 42 static contrasts (unlabeled, never planned) |
| Regression cohort | the frozen v1 roster: 205 queries on **22** roots (K0 4, K1–K3 6 each). The v1 `MOCK_REPORT.md` says "24 train roots (6 per K0–K3)"; the frozen file has 22, so that statement is corrected here |
| Holdout cohort | 20 new train roots, 200 queries. Lineage groups are disjoint from all 218 roots v1 touched (roster, fixtures, fixture searches, baseline replays). Selection used geometry only, with roots at or above the class median object count and at least one far, high-clutter pair, in hash order |
| Holdout slots per root | far, far + max clutter, mid and T's goal cell (as an ordinary destination) in the root scene; far and vacated after 1 edit; far, far + max clutter, vacated and mid after 2 edits |
| Mock composition | far 173, mid 119, near 39, vacated 74; depth 0/1/2 = 167/104/134; box 199, cylinder 206; ≥ 3 objects between 82; T-goal destination 29; K0/K1/K2/K3 = 64/104/117/120 |
| Leakage | 0 val/test roots; 0 cross-split derived layouts |
| Candidate policies | `adaptive_v1` (4×500 → 8×1,500 → 16×5,000), `fixed_4x1500`, `fixed_4x5000`, `fixed_8x5000`. **Every candidate was run for real** on all 405 queries (12 workers, CPUs 0–11) |
| Reference | `reference_v2`: every valid unit × every eligible branch at 5,000 RRT iterations, no early stop. This is *finite reference recall*, not physical truth |
| Choice rule | A candidate passes with missed reference positives ≤ 5% overall **and** ≤ 10% in every predeclared difficult stratum that has ≥ 10 reference positives (far band, depth 2, ≥ 3 objects between, holdout, box, cylinder, each K, T-goal cell). Wilson 95% intervals are reported but are never a threshold. Choose the cheapest passing policy by measured mean CPU-s per query; `adaptive_v1` is preferred if within 1.15× |

## 3. Results

### 3.1 Labels (405 physical queries)

| run | label 1 | 0 certified geometry | 0 bounded search | 0 deterministic plan | unlabeled (no valid post-grasp state) | CPU-s / query | CPU-h | wall |
|---|---|---|---|---|---|---|---|---|
| `reference_v2` | 247 | 57 | 39 | 29 | 33 | 17.3 | 1.95 | 11.3 min |
| `adaptive_v1` (A) | 247 | 57 | 39 | 29 | 33 | 9.40 | 1.06 | 5.6 min |
| `fixed_4x1500` | 237 | 57 | 47 | 31 | 33 | 8.51 | 0.96 | 5.0 min |
| `fixed_4x5000` | 242 | 57 | 41 | 32 | 33 | 8.71 | 0.98 | 5.1 min |
| `fixed_8x5000` | 246 | 57 | 40 | 29 | 33 | 10.62 | 1.19 | 6.3 min |

- **Negative audit:** all 57 certificates are complete (32 poses screened, footprint-only reasons, every eligible branch of the physically invalid poses tested).
- **Execution failures:** across all 2,146 reference branch executions, 3 failed, 2 at the `attached` gate and 1 at `xy`.
- **Labelling rules held:** no unlabeled status was ever stored with label 0, and no candidate found a positive that the reference missed.

### 3.2 Policy comparison against the reference (247 reference positives)

| policy | missed (rate, Wilson 95%) | far (99) | depth 2 (71) | ≥ 3 between (33) | holdout (109) | box (121) | vacated (46) | result |
|---|---|---|---|---|---|---|---|---|
| `adaptive_v1` | **0** (0.0%, 0–1.5%) | 0 (0–3.7%) | 0 (0–5.1%) | 0 (0–10.4%) | 0 (0–3.4%) | 0 | 0 | pass |
| `fixed_4x1500` | 10 (4.0%, 2.2–7.3%) | 8 (8.1%, 4.2–15.1%) | 2 | **4 (12.1%, 4.8–27.3%)** | 4 | 7 (5.8%) | 2 | **fails** (≥ 3-between stratum > 10%) |
| `fixed_4x5000` | 5 (2.0%, 0.9–4.7%) | 3 (3.0%) | 2 | 1 | 3 | 3 | 2 (4.3%) | pass |
| `fixed_8x5000` | 1 (0.4%, 0.1–2.3%) | 1 | 0 | 0 | 0 | 0 | 0 | pass |

Per-K and per-shape tables and the missed query ids are in `mock_v2/validation/analysis_v2.json`; figure: `mock_v2/figures/policy_misses_by_stratum.png`.

**Decision by the frozen v2 rule:**
- `adaptive_v1`, `fixed_4x5000` and `fixed_8x5000` pass. The cheapest passing policy is `fixed_4x5000` at 8.71 CPU-s per query.
- `adaptive_v1` costs 9.40 CPU-s, 1.079× as much, which is within the 1.15× preference margin, so **the rule chooses `adaptive_v1`**. That is also the recommendation. It is the only candidate with no observed miss in any stratum: its upper bound is ≤ 3.7% on the far band and ≤ 10.4% on the 33 most cluttered positives.
- This replaces the v1 rule's literal choice of `fixed_4x1500`, which fails v2's clutter stratum and misses 8 of 99 far positives. The new rule is explicitly registered, and v1's preregistration is not rewritten.
- `fixed_4x5000`, which had never been broadly tested, is a defensible cheaper alternative at 2.0% misses. Its misses concentrate in vacated cells (2 of 46) and the far band (3 of 99).

### 3.3 v1 → v2 regression and branch effects

- **Same queries, same labels:**
  - on the same 205 queries, v1 pilot A versus v2 `adaptive_v1` gives **identical labels** (138 / 25 / 27 / 15);
  - the 40 v1 reference queries are also identical between `reference_v1` and `reference_v2`.
- **Branch effects in the reference** (summed over 405 queries):
  - 12 units had a first eligible branch fail its hold while a later branch held;
  - 3 units were **verified on a later branch after a hold failure**;
  - 0 units were rescued after an execution failure;
  - 219 units were verified after a plan failure, which v1's planning loop already handled;
  - 4 verified branch results come from poses v1 would have rejected. No positive query depended only on them, so **the v1 defect changed no query label in this mock**. v2 closes a latent false-negative path rather than correcting observed label errors.
- **Natural fixture (frozen, `fixtures_v2.json`):** in `3ds66-srv-w0033-s016:q0`, pose rank 1 was **rejected by v1** (branch 0 hold failed). v2 certifies branches 1 and 2, and both pass all 13 gates. GIF: `figures/witness_natural_fixture_rank1_laterbranch.gif`.
  - Four further mined candidates are recorded with their outcomes, including both v1 `held_state_lost` units. In those two, v2's other branch holds but fails to plan, so there was no rescue.
  - Two more hold-failure-then-verified units appeared in the holdout (`srv3-w0069-s007:h3`, `:h6`).
- **Not found:** a natural "execution failure, then a later branch verified" unit. The bounded mock has only 3 failed executions, none followed by a verified branch. This case is covered by the labelled synthetic test `test_injected_exec_failure_is_rescued_SYNTHETIC`.

**Reference mode and branch accounting** (12,960 nominal poses, all screened):

| item | count |
|---|---|
| IK_INVALID | 5,090 |
| HELD_INVALID (kinematic) | 3,761 |
| GEOMETRY_INVALID | 1,813 |
| HELD_INVALID_PHYSICAL (every eligible branch failed its hold) | 4 |
| valid held units | 2,292 |
| branch candidates | 17,664 |
| eligible branches | 9,443 |
| hold-tested (held 5,157 / failed 50) | 5,207 |
| branch attempts | 5,145 |
| plan failed | 2,766 |
| budget-unresolved | 188 |
| executions | 2,146 |
| verified | 2,143 |
| **unaccounted** | **0** |

Every query in every run has exactly one terminal class per candidate branch.

### 3.4 Determinism, virtual replay, storage and interruption

| check | result |
|---|---|
| `adaptive_v1` run A vs run B (full canonical records, 405) | **0 mismatches** |
| serial (1 worker, unit = query, 42) vs A | 0 |
| sweep 8 / 12 / 16 / 18 workers (unit = query, 167 each) vs A | 0 / 0 / 0 / 0 |
| SIGKILL of own process group after 10 units, then NPZ truncation + orphan JSON + stray temp file, then resume vs A (405) | **0**. Resume validated 9 units, flagged the damaged unit (`npz_sha_mismatch`), quarantined 4 files, recomputed 33 |
| branch-aware virtual replay from `reference_v2` vs the real run (label, unit states, verified rank / branch / witness id, counted motion work), each candidate × 405 | **0 / 0 / 0 / 0**. Prefix equivalence holds exactly under the new branch semantics; it was used only as a cross-check |
| unit JSON + NPZ re-validation (every unit of every run) | 0 failures |

**Fresh-process replay** of every stored (query, witness) association, each in its own query scene, gave 3,638 of 3,638 verified, with 0 difference in executed motion at stored precision and identical hold evidence:

| run | associations | unique witnesses |
|---|---|---|
| reference | 2,143 | 2,135 |
| A | 247 | 246 |
| B | 247 | 246 |
| interrupted run | 247 | 246 |
| `fixed_4x1500` | 237 | 236 |
| `fixed_4x5000` | 242 | 241 |
| `fixed_8x5000` | 246 | 245 |
| serial | 29 | 29 |

**v1 history under the pinned v1 code** against `8032ad0b`: 26 of 26 reference witnesses and 138 of 138 pilot-A witnesses verified, with 0 difference.

**Tests:** `pytest code/test_mock3d_v2.py` passes 25 of 25 both before the runs (9 min 16 s) and after them (7 min 13 s).

**Integrity:**
- the pinned source re-hashes with 12,424 entries and 0 changes, and the pinned clone has a clean git status;
- the v1 sidecar's 1,955 files are byte-identical to their pre-v2 hash list.

### 3.5 Safety and resources

| item | measured |
|---|---|
| CPU mask / own-tree affinity union | 0–11 for every 12-worker run (sweeps: exactly the requested 8, 12, 16 or 18; never more than 18) |
| peak own-tree CPU | 1,201% at 12 workers (1,802% at the 18-worker sweep) |
| processes | coordinator + workers + resource tracker only (0 unexpected descendants) |
| package temperature | peak 49 °C; 0 pauses |
| memory | worker RSS ≤ 0.81 GB; minimum available 51.9 GB; swap growth 0 |
| monitor violations | **0** in every run |
| fail-closed tests | sensor absent at start: refused; sensor removed while paused: stopped, never resumed (`test_temperature_sensor_loss_fails_closed`); source drift: refused; production without authorization: refused |

There is no hard CPU quota. gvlab's cgroup has only the `memory` and `pids` controllers, so the ceiling is enforced by affinity plus the fail-closed monitor; this is not claimed as a kernel quota.

**Worker sweep** (167 depth-0 queries):

| workers | queries / s | CPU-s / query |
|---|---|---|
| 8 | 0.586 | 13.1 |
| **12** | **0.846** | 13.3 |
| 16 | 0.832 | 17.7 |
| 18 | 0.821 | 19.8 |

Twelve workers on physical cores 0–11 is fastest. The SMT siblings add CPU cost without throughput, so 12 remains the production setting.

### 3.6 Structure-check finger width (instruction 3.3)

The source's object checks pre-shape the fingers, but its **structure** check keeps the default, fully open fingers. v2 keeps the source behaviour, because the source labels were built with it. It is now quantified on all 405 mock queries × 32 poses (12,960 poses, 94,790 IK seed solutions):
- 72 poses (0.56%) have no structure-clear solution with open fingers but would have one pre-shaped;
- **31 poses (0.24%) are clear only with open fingers.**

So the check is **not uniformly conservative**, which corrects v1's description. Positives are unaffected: every executed trajectory must also pass the physical robot–structure contact gate and the hold certificate. Negatives or unlabeled outcomes involving the 72 poses are scoped by this convention. Changing it would need a new protocol id and full revalidation; it was not changed.

### 3.7 Coverage

| stratum | coverage |
|---|---|
| K0–K3, depth 0–2, near / mid / far / vacated, box / cylinder, between 0 / 1–2 / 3+, both cohorts | labelled 1 and 0 in every cell |
| K4 | **COVERAGE_BLOCKED**: 0 roots exist |
| NONE | **COVERAGE_BLOCKED**: 3 selected roots, none in the mock |

## 4. Live source and runtime forecast

The live source is summarized in `source_sync/LIVE_3D_SOURCE_SYNC_REPORT.md`:
- the snapshot was cut at 2026-11-07 05:28:58 UTC;
- it has 1,585 selected roots (NONE 3, K0 511, K1 357, K2 357, K3 357, K4 0);
- 0 old bytes changed, and the extended split changed 0 old assignments, with 0 conflicts and 0 cross-split layouts;
- the live production roster design has **13,681 physical queries** on 1,469 roots that have eligible queries (train 9,674, val 2,066, test 1,941), plus 2,774 static contrasts.

**Forecast with `adaptive_v1` at 12 workers:**
- **Cost model:** per-query CPU from real run A, grouped K × depth × band (13,344 queries costed at that level, 307 backed off to K × depth, 30 NONE at the overall mean), with a 2,000-sample root-lineage bootstrap.
- **Parallel efficiency:** 0.881, measured as sweep-12 queries per second per worker against serial.

| quantity | value |
|---|---|
| CPU-hours | **35.2** (p05–p95 32.9–37.9) |
| wall, expected | 0.139 days (3.3 h) |
| wall, p95 | 0.149 days |
| **wall, p95 × 1.5** | **0.224 days (5.4 h)**, against the 7-day limit |
| disk (one witness per positive, 36 KB measured; 21 KB JSON per query) | 0.64 GB (0.97 GB with × 1.5) |
| full 2,500-root quota, scaled by root count | ~56 CPU-h, ~0.35 days at p95 × 1.5; K4 / NONE cost unmeasured, and K4 scenes are more cluttered, so treat this as a lower bound |

- **Why the earlier 4.4-hour estimate doesn't carry over unchanged:** the per-query cost rose from v1's 8.5 to 9.3 CPU-s. The causes are branch retries, which add held-branch hold, plan and execute attempts, and the larger, more cluttered K3 share in the live snapshot.
- **Evaluation-layer cost:** keeping every verified branch at production scale, as the reference does, would store about 8.7 witnesses per positive: 13,681 × 0.61 positives × 8.68 × 32.6 KB ≈ 2.4 GB.

## 5. Production preparation (nothing started)

- **Collector and guard:**
  - `run3d.py --production` needs `guard3d.check`, which requires all of the following:
    - a recorded user authorization in `production_policy.json`;
    - frozen roster, source-pin, split, code and protocol hashes equal to the current ones;
    - `--confirm` equal to `token(current hashes)`, so a token is invalid after any change;
    - the authorized policy, workers and mask (at most 18);
    - a readable sensor;
    - no other collector running.
  - `mock_v2/launch_full_3d.sh` refuses a no-argument call and re-runs the guard, the metadata dry run and the test suite before it could `exec`.
- **Exercised:** only refusals and metadata dry runs.
  - The all-splits production dry run on the live snapshot made **0 planner calls** and wrote nothing.
  - Refusals were tested for: no arguments; a wrong token on the unauthorized template; `--production` without authorization; and `guard3d.check` listing the missing authorization, the token mismatch and the running collector.
- `mock_v2/production_policy.TEMPLATE.json` is unauthorized. Exact future steps are in `3D_MOCK_V2_RUNBOOK.md`.
- **Readiness scope.** `PRODUCTION_READY_PENDING_AUTHORIZATION` means only your recorded authorization is missing for the audited live snapshot (K0–K3 plus 3 NONE). The recommended order is still:
  1. let the generator finish K4 / NONE;
  2. let its author run the fixed-object check;
  3. re-snapshot and re-pin (runbook §1);
  4. then authorize.

## 6. Unresolved and limitations

- The live source is IN PROGRESS (K4 0/510, NONE 3/408). 488 selected roots await the fixed-object check, so the selection may still change.
- K4 and NONE placement costs are unmeasured (`COVERAGE_BLOCKED`).
- The structure check keeps the source's fully open fingers, which is not uniformly conservative (§3.6).
- There is no natural execution-failure rescue in the mock; it is covered synthetically.
- The finite reference recall is not physical truth. `NEG_UNDER_POLICY` is not a proof of impossibility, and `NEG_CERTIFIED_GEOMETRY` holds only within its declared model.
- The CPU ceiling relies on affinity plus the monitor, not a kernel quota.

## 7. Where things are

On gvlab, the sidecar is at `~/approachb3d_mock/src/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/` and the live staging copy is at `~/approachb3d_mock/live_snapshot/20261107T052857Z/`. The laptop staging copy is `~/Desktop/shelf66v3_live_snapshots/20261107T052857Z/`.

On GitHub, `naren/3d-dataset-relocations` holds the same paths; commits are listed in `BRANCH_PROVENANCE.md` and `PUSH_AUDIT.md`. `ARTIFACT_MANIFEST.sha256` holds the SHA-256 of every file in this sidecar.
