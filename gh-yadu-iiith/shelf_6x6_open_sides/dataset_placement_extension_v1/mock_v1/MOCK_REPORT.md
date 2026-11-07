# MOCK_REPORT: SHELF66V3 3D placement-only extension, mock v1 (nav@gvlab, 2026-11-07)

**Verdict: `MOCK_PASS`.** Every predeclared gate was evaluated on real MuJoCo executions and replays; nothing was waived. Sub-verdicts are in §0.

Recommended production policy: **`adaptive_v1`** at **12 workers**, one per physical core (CPUs 0–11).
- This departs from the literal output of the frozen selection rule, which was `fixed_4x1500`. §6.3 explains why and asks for your confirmation.
- A projected collection over the snapshot roster takes 30 CPU-hours: about 2.6 h of wall time, or ≤ 4.4 h at p95 with the 1.5× buffer, and under 1 GB of disk.

**Nothing was launched beyond the bounded mock.** No full collection, no new source roots, no training, no change to the source generator or dataset, no push or merge.

Evidence lives in `mock_v1/` on gvlab:
- `runs/`, `validation/`, `figures/`, `configs/`;
- `validation/analysis.json` holds the machine-readable values behind every number below.

## 0. Gate summary

| gate | result | evidence |
|---|---|---|
| semantics | **PASS** | placement-only witnesses (phases ⊆ lift/carry/lower/release/retreat in all 296 replayed positives). Every positive is MuJoCo-verified. A mode-level pick-inaccessible / held-placeable fixture is labelled 1. Unlabeled queries are never 0 (§3, §8) |
| determinism | **PASS** | 0 canonical mismatches in every comparison (§7). Nested virtual budgets equal independent reruns: 36/36 for `adaptive_v1` and 36/36 for `fixed_4x1500` (after AMENDMENT_002) |
| physical witness | **PASS** | 296/296 stored positives re-verified in fresh processes with bit-identical executed samples. 24/24 source witnesses reproduce under the unchanged source `run_plan` (§8) |
| negative audit | **PASS** | 0 inconsistent records across 5 runs; every 0 carries kind, scope and accounting (§5) |
| interrupt / resume | **PASS** | SIGKILL of the run's own process group (14 PIDs, all `nav`) after 8 / 22 units; the resumed output is canonical-equal to the uninterrupted run, 205/205 (§7) |
| storage | **PASS** | content-addressed per-root NPZ round-trips and re-verifies. Compatibility converter verified on 24 source witnesses (§9) |
| source integrity | **PASS** | 12,424 manifest entries unchanged before and after; source git status clean (§10) |
| split / leakage | **PASS** | 1,025 lineage groups closed; all mock queries train; the roster code reads no protected metadata (tests) |
| finger regression | **PASS** (one inherited conservatism disclosed) | every object check uses pre-shaped fingers (instrumented test). The structure check keeps the source's fully open default, which affects 0.55% of poses and is conservative (§8.3) |
| safety (≤ 18 logical CPUs) | **PASS** | 12 runs, 0 monitor violations, 0 safety stops, affinity union ≤ 18, peak 50 °C, RSS ≤ 0.87 GB, MemAvailable ≥ 51 GB (§11) |
| coverage | **PASS** for K0–K3 | every predeclared stratum is labelled. **K4 and NONE are `COVERAGE_BLOCKED`**: K4 has no roots; NONE has 3 roots, 1 of them train (§12) |
| label quality | **PASS** | `adaptive_v1`: 0/26 reference positives missed on the reference subset (Wilson 0–12.9%) and 0/138 on the whole broad pilot (0–2.7%) (§6) |
| baseline replay | **PASS** | (a) 24/24 unchanged source replays match the recorded outcome, \|Δz\| ≤ 5.6e-7 m. (b) 24/24 converted placement-only replays are held and verified |
| visualisation | **PASS** | scene panel, verified witness GIF, negative-case GIF, budget, sweep, resource, label and accounting plots (§13) |
| runtime / forecast | **PASS** | 30.2 CPU-h (p05–p95: 26.8–33.5), 0.18 days at p95 × 1.5 with 12 workers, against a 7-day limit (§12) |

## 1. What ran

The host was nav@gvlab (Xeon Gold 5317; CPU k and k+12 share core k), in tmux session `approachb3d`. Every run used `run3d.py` with pinned single-thread workers, `nice 10`, a coordinator pinned to the same mask, and a monitor sampling every 2 s that fails closed.

| run | policy | subset (unit) | workers / CPUs | queries | wall |
|---|---|---|---|---|---|
| `reference` (v1b, after AMENDMENT_002) | `reference_v1` | 40 predeclared (root) | 12 / 0–11 | 40 | 90 s |
| `serial` | `adaptive_v1` | slot-1 queries (query) | 1 / 0 | 22 | 270 s |
| `sweep_8/12/16/18` | `adaptive_v1` | all 205 (query, cold cache) | 8/12/16/18, CPUs 0..N−1 | 205 each | 343 / 239 / 234 / 238 s |
| `pilot_A`, `pilot_B` | `adaptive_v1` | all 205 (root) | 12 / 0–11 | 205 each | 177 / 176 s |
| `interrupt` | `adaptive_v1` | all 205 (root) | 12 | 205 (SIGKILL after 8 units, then resume) | 114 s + 140 s |
| `pilot_C1`, `pilot_C2` | `fixed_4x1500` | all 205 (root) | 12 | 205 each | 160 / 157 s |
| `supplement_reference_on_adaptive_negatives` | `reference_v1` | 27 post-hoc (root) | 12 | 27 | 67 s |

Preserved but not used as evidence:
- `runs_aborted/reference_v1_monitor_crash`: the first launch. The monitor thread died on an `hwmon2` entry with no name file (AMENDMENT_001).
- `runs_superseded/reference_v1a_*`: two reference runs made before AMENDMENT_002. Both are canonical-equal to the final reference run.

## 2. Live source

See `../LIVE_SOURCE_AUDIT.md`.
- gvlab runs no generator and holds no shelf store. The live generator is `user1@10.4.25.63`, which is not accessible.
- The newest reachable data is GitHub `yadu/dataset-3d` @ `8032ad0b`: 1,492 selected roots (NONE 3, K0 511, K1 357, K2 357, K3 264, K4 0) and 10,927 witnesses, all present and loadable, with a single V3 protocol.
- There was therefore no CPU conflict; the 18-CPU ceiling belonged to the mock alone.

## 3. Placement teacher `SHELF66V3_PLACE_V1` (protocol `d00f1d83…`)

**Event:** `V_place(S_hyp, o, d; PI_PLACE_3D)` is conditional on a certified post-grasp state (README §Action semantics).

**Held-state mechanism:**
1. Restore the settled scene.
2. Set the arm to `q_g`, a structure-clear IK branch ≥ 5 mm from every other object.
3. Pre-shape the fingers to the grasp's open width; require 0 contacts.
4. Close in place (400 steps).
5. Certify: both finger sides on the object, object shift ≤ 10 mm, neighbours ≤ 4 mm, no other robot contact.

Physical held-invalid poses were rare: 4 of 663 valid units in the pilot. Two execution-stage hold failures occurred (`held_state_lost`): on the planned IK branch the box shifted 10.2 mm, against the 10 mm limit. Both queries were then verified by another unit.

**Executed and verified phases:** lift, carry, lower, release, retreat, settle, with the 13 gates in `placement3d.verify`. All gates are checked at every simulation step, including physics contacts, plus a margin-0 envelope check against the structure, the arm ceiling and other objects' no-overpass columns.

**Excluded from the label:** free move, hover, approach, pick, sequencing, ready return and the target goal.

Execution never contradicted a successful plan. The reference tested every valid unit with no early stop: 249 units planned and executed, 149 verified, 0 execution failures. The safety gates remain mandatory (§8).

## 4. Frozen design (all frozen before any roster query was labelled)

| item | value |
|---|---|
| lineage split (`split_manifest.json`, sha `768437e0…`) | union-find over parent family and identical layout: 1,025 groups (329 multi-root). Train 1,055 / val 231 / test 206 roots |
| mock roster (`roster.jsonl.gz`, sha `08e95c08…`) | 24 train roots (6 per K0–K3, distinct families), 205 eligible queries, 42 static contrasts (OCCUPIED, OUT_OF_BOUNDS; never planned) |
| strata (eligible) | K0 34 / K1 54 / K2 57 / K3 60; depth 0: 87, 1: 64, 2: 54; near 31 / mid 68 / far 72 / vacated 34; box 105 / cylinder 100. Mid and far appear in edited scenes (1:mid 22, 1:far 22, 2:mid 14, 2:far 20). 3 destinations equal T's goal |
| reference subset | 40 queries (2 slots per root by a fixed pattern) |
| production design (`production_roster_design.jsonl.gz`) | the same rule on all 1,492 selected roots: **12,751** eligible queries on 1,376 roots (116 K0 roots have no obstacle), plus 2,588 static contrasts |
| configs (read-only) | `selection_rule.json` `bb2d56da…`, `run_plan.json` `f125a65d…`, `fixtures.json` `941d10bc…` |
| amendments (read-only) | **001**: `run3d.py` monitor crash fix (fail closed on any monitor error). **002**: `policy3d.py` bookkeeping, recording pre-route work for exact virtual accounting of truncated units. Both are collector or bookkeeping changes only; no teacher semantics, thresholds or rules changed |

**Mode ranking (predeclared):** the k-th azimuth nearest the robot, grip heights alternating, then the other heights.

**Scheduling unit:** one valid, physically held pose. Its ≤ 4 IK branches are tried inside the unit and never counted as modes.

**Counted budgets:** RRT outer iterations, extensions, edge checks, shortcut checks, IK calls, object and structure checks, simulation steps. No wall clock decides an outcome. The CPU guard (1,200 CPU-s per query) only ever interrupts; it never fired.

## 5. Broad pilot results (`adaptive_v1`; pilot A = pilot B)

**Outcomes over 205 queries:**
- **138** labelled 1 (`VERIFIED`);
- **52** labelled 0:
  - 25 `NEG_CERTIFIED_GEOMETRY`: every held pose fails the deterministic footprint certificate at the start or goal;
  - 17 `BOUNDED_SEARCH`: some unit's carry route was unresolved at 5,000 iterations;
  - 10 `DETERMINISTIC_PLAN`: lift, carry, lower or retreat IK chains hit something or left reach;
- **15** unlabeled (`NO_VALID_POST_GRASP_STATE`).

Positives come from 22 lineages and negatives from 20. Every negative was fully screened (all 32 poses).

| stratum | n | 1 / 0 / unlabeled | mean / p90 CPU-s |
|---|---|---|---|
| K0 | 34 | 25 / 6 / 3 | 8.7 / 18.9 |
| K1 | 54 | 38 / 14 / 2 | 7.3 / 18.7 |
| K2 | 57 | 40 / 12 / 5 | 8.3 / 18.2 |
| K3 | 60 | 35 / 20 / 5 | 10.1 / 20.7 |
| depth 0 / 1 / 2 | 87 / 64 / 54 | 63/18/6, 43/16/5, 32/18/4 | 10.8 / 6.9 / 7.2 |
| near / mid / far / vacated | 31 / 68 / 72 / 34 | 22/6/3, 47/17/4, 45/21/6, 24/8/2 | 9.1 / 9.9 / 9.0 / 4.8 |
| box / cylinder | 105 / 100 | 71/26/8, 67/26/7 | 8.8 / 8.4 |
| objects between source and destination: 0 / 1–2 / 3+ | 84 / 97 / 24 | 71/12/1, 61/29/7, 6/11/7 | 6.2 / 10.2 / 10.6 |

**Mode accounting** over 6,560 nominal poses: IK-invalid 1,503; held-invalid 1,296 (kinematic) + 4 (physical); geometry-invalid 633; valid 663, of which untested 335, unresolved 70, plan-failed 118, exec-failed 2, verified 138; 2,461 unscreened. Unscreened poses were never needed because screening is lazy; that is the saving. 449 configurations were attempted. Verified nominal ranks span 0–23 (rank 0 for 55 queries).

**CPU per query:** mean 8.6 s, median 5.5 s, p95 20.5 s. By part, screening takes 84% of the CPU (the source teacher's 17-seed IK per pose), planning 9%, MuJoCo execution 5%, holds 1.2%, verification 0.3%.

## 6. Budget comparison and policy choice

### 6.1 Predeclared comparison: 40-query reference subset (36 paired; 26 reference positives)

Virtual replays come from `reference_v1` records. They are exact, verified by independent reruns: 36/36 for `adaptive_v1` and 36/36 for `fixed_4x1500`.

| policy | missed / 26 | relative miss (Wilson 95%) | virtual CPU-s/query | route iterations |
|---|---|---|---|---|
| `fixed_4x1500` | 1 | 3.8% (0.7–18.9%) | **6.34** | 22,469 |
| `fixed_4x5000` | 0 | 0% (0–12.9%) | 6.39 | 66,199 |
| `fixed_4x500` | 2 | 7.7% (2.1–24.1%) | 6.45 | 9,043 |
| `fixed_8x1500` | 1 | 3.8% | 6.90 | 28,861 |
| `fixed_8x500` | 2 | 7.7% | 7.00 | 11,435 |
| `fixed_8x5000` | 0 | 0% | 7.01 | 86,591 |
| `fixed_16x1500` = `fixed_32x1500` | 1 | 3.8% | 7.17 | 30,929 |
| `fixed_16x500` = `fixed_32x500` | 2 | 7.7% | 7.26 | 12,503 |
| `fixed_16x5000` = `fixed_32x5000` = `reference_v1` | 0 | 0% | 7.29 | 92,159 |
| **`adaptive_v1`** | 0 | 0% (0–12.9%) | 7.45 | 125,159 |

Every 500-iteration policy fails the predeclared 5% overall threshold; 1,500 and 5,000 both pass. The frozen rule picks the cheapest passing policy, preferring `adaptive_v1` only within 1.15× of it. On these virtual costs that gives `fixed_4x1500`, since 7.45 / 6.34 = 1.175.

The only strata with ≥ 10 reference positives are depth 0, box and cylinder, so the far band (9) and K3 (7) were never evaluated. `fixed_4x1500` misses 1/9 far and 1/7 K3 positives there.

### 6.2 Broad pilot against the reference (205 queries, 138 reference positives)

This is a **post-hoc diagnostic supplement, disclosed as such.**
- `reference_v1` was run on all 27 queries that `adaptive_v1` labelled `NEG_UNDER_POLICY` and found **no** further positive.
- Certified-geometry negatives have no valid unit, and adaptive positives are reference positives.
- The reference label is therefore known for all 205 pilot queries.

| | missed / 138 | overall (Wilson 95%) | far band | near / mid / vacated | 3+ objects between |
|---|---|---|---|---|---|
| `adaptive_v1` | **0** | 0% (0–2.7%) | 0/45 (0–7.9%) | 0 / 0 / 0 | 0/6 |
| `fixed_4x1500` | 6 | 4.3% (2.0–9.2%) | **6/45 = 13.3% (6.3–26.2%)** | 0 / 0 / 0 | 2/6 |

`fixed_4x1500` labels are identical to those derived exactly from the adaptive records (205/205), and C1 = C2. All six misses are far destinations. Measured CPU on the same 205 queries: `adaptive_v1` 8.62 s/query vs `fixed_4x1500` 7.84 s/query, a ratio of **1.10**.

### 6.3 Recommendation, and the deviation from the frozen rule's literal output

**Recommend `adaptive_v1`.**
1. `fixed_4x1500` systematically under-labels far placements: a 13.3% relative miss, above the rule's own 10% stratum limit. Its lower CI bound is 6.3%. The predeclared 40-query subset could not evaluate that stratum (9 < 10 positives).
2. The rule's adaptive-preference clause (≤ 1.15×) holds on measured costs (1.10×). It failed only on the subset's virtual estimate (1.175×).
3. `adaptive_v1` was the predeclared broad-pilot policy, so it carries the full evidence: two runs, serial, interrupt, sweeps and replays.
4. The extra cost is about 2.3 CPU-hours over the whole snapshot production.

No threshold was changed. `LAUNCH_READINESS_DECISION.json` records both `rule_literal_choice: fixed_4x1500` and `recommended: adaptive_v1`, so you decide.

`fixed_4x5000` would also have 0 observed misses at ≈ `fixed_4x1500` cost, but it was not run on the broad pilot. A 0 under `adaptive_v1` means 16 valid units were tested with up to 5,000 route iterations each, and not one placement was verified. That is a finite-policy negative, not proof of impossibility.

## 7. Reproducibility (canonical records exclude CPU and wall fields)

| comparison | shared | mismatched |
|---|---|---|
| pilot A vs pilot B (same everything) | 205 | **0** |
| interrupted + resumed vs pilot A | 205 | **0** |
| serial vs pilot A (different unit composition; cache-dependent screening counters excluded) | 22 | **0** |
| sweep 8 / 12 / 16 / 18 vs pilot A (single-query units, cold cache) | 205 each | **0** |
| sweep 12 / 16 / 18 vs sweep 8 (full records) | 205 each | **0** |
| pilot C1 vs C2 (`fixed_4x1500`) | 205 | **0** |
| C1 labels vs exact derivation from adaptive pilot A | 205 | **0** |
| final reference vs both superseded reference runs | 40 | **0** |
| virtual `adaptive_v1` vs real (label, unit states, rank, witness, counted plan/exec work) | 36 | **0** (3 before AMENDMENT_002, all in work counts only) |
| virtual `fixed_4x1500` vs real pilot C1 | 36 | **0** |
| reference witnesses vs pilot witnesses at the same rank | 26 | 26 equal IDs |

**Interrupt test:**
- `run_mock.sh` waited for 8 committed units, then sent `kill -9 -- -PGID`. The kill list `runs/interrupt_killed_processes.txt` holds 14 processes: coordinator, resource tracker and 12 workers, all `nav`, all in the run's own process group.
- The identical command resumed 14 units (8 reused by ledger SHA-256) and finished complete.
- No other user's process and no source process was touched.

## 8. Physical witnesses, replays, fixtures, tests

### 8.1 Fresh-process replay of every stored positive (`replay3d.py ours`)

Coverage: pilot A 138/138, reference 26/26, pilot C1 132/132. Each replay:
1. rebuilds the scene from the roster;
2. checks the stored grasp pose against the recomputed one (identical);
3. re-certifies the hold (identical evidence);
4. re-executes the stored waypoints;
5. re-runs all gates.

Executed arm samples and the final object pose are **bit-identical** (max |Δ| = 0 at stored float32 precision), and every witness is placement-phase only.

### 8.2 Source baseline (24 step-0 B witnesses; 6 roots per K0–K3)

- **(a) Unchanged source replay.** The source `run_plan` (with open, to_hover, approach and close) runs from the settled root scene. All 24 reproduce the recorded end cell and success; |Δz_err| ≤ 5.6e-7 m (bit-exact in 13 of them). This confirms the environment matches the generator hosts.
- **(b) Compatibility converter.** `convert` keeps lift … retreat, takes `q_g` = the first lift waypoint and the release width = open width. The result runs in the held-state harness. 24/24 are held and verified under our stricter gates (xy ≤ 9.4 mm, tilt ≤ 0.1°).

### 8.3 Finger regression

The pre-V3 bug was object checks run with fully open fingers.
- `test_finger_preshaped_in_every_object_check` instruments every object check in screen and plan. Each one runs at the grasp's open width, and the finger joint values equal the calibrated pre-shape.
- **Inherited and disclosed:** the source's structure check (`static_ck`) still uses the model's default, fully open fingers (8.9 cm).
  - `finger_diag.py`, over all 1,280 poses of the 40 reference queries: pre-shaped fingers would add 21 of 9,455 structure-clear IK solutions (0.2%) and 7 poses (0.55%).
  - The inherited check is therefore conservative and small. It was kept unchanged to preserve protocol identity.

### 8.4 Unit fixtures

All are train roots outside the mock roster, found by sha-ordered dev search and frozen in `fixtures.json` before the pilot.

| fixture | root / query | outcome |
|---|---|---|
| pick-inaccessible, held-placeable | `srv3-w0066-s021` o00 → (2,1) | **1**. Its only verified pose cannot be picked: the source `_plan_approach` (hover chain + 5,000-iteration free move) fails on every IK branch. **Mode level only:** a query where all 32 poses are pick-blocked yet placeable was searched for in ≈ 400 dev queries and **not found**. It is structurally rare: the source's carry-footprint strip, about 30 cm behind the pads, contains the 13 cm approach path |
| obstructed carry (certified) | `srv-w0042-s014` o01 → (2,1) | 0, `NEG_CERTIFIED_GEOMETRY`, 0 valid units, 0 motion tests |
| valid negative after exhausting the budget | `lap-w0015-s000` o00 → (5,3) | 0, `BOUNDED_SEARCH`, ≥ 1 unresolved |
| target goal cell as destination | `srv-w0042-s014` o00 → T goal (2,2) | 1 (legal destination) |
| vacated after edit vs occupied before | `lap-w0015-s000` o01 → (0,4) | depth 1 → 1. The same cell in the root scene is `OCCUPIED` (static, unlabeled) |
| release / settling failure (synthetic, not a sample) | — | release right after the lift → `cell` gate fails; tilted placed object → `upright` gate fails |
| admin interruption | — | the CPU-guard signal raises `Interrupted`; the record is `label: null`, never 0 |
| two-edit descendant hash | roster | every depth-2 scene's parent is its lineage's depth-1 hash, and hashes recompute from the cells |

`pytest code/test_mock3d.py`: **15/15 passed** before the freeze and again after all runs (`validation/pytest_post_run.txt`).

**A bug the tests caught, disclosed:**
- The first pick-access diagnostic inherited the last placement stage's RRT ceiling (500) for the source's free move instead of 5,000. That made one dev fixture look pick-blocked when it was not.
- The diagnostic now pins 5,000. Candidates were re-checked: 4 of 155 label-1 queries hold, and the fixture was re-frozen before the pilot.
- The teacher itself sets its ceiling explicitly on every call, so labels were never affected.

## 9. Storage

- **Per root:** `units/<root>.json` holds the records, accounting and evidence. `units/<root>.npz` holds keys `<witness_id>.<array>`, with ids from the sha256 content address (the source's `Store.witness` scheme).
- **Witness arrays:**
  - `q (N,7)` float64 planned waypoints with `segment` / `seg_off` / `seg_phase` / `seg_grip`, and `carry_xy`;
  - `q_exec (M,7)` with `gap_exec`, `obj_exec (M,7)`, `phase_exec`, `step_exec`, executed every 10 steps;
  - `q_g`, grasp id/azimuth/level/width, and the held-state evidence.
- **Measured size:** 34 KB per witness (compressed) plus 13 KB of JSON per query. The mock runs total 70 MB.
- **Integrity:** every pack loads and re-hashes to its id, and all positives re-verify (§8.1).

## 10. Source integrity and leakage

- **Source integrity:** `audit3d.py integrity` was run before the pilot, after all runs, and at the end. All 12,424 entries (selected roots, 10,927 referenced witnesses, `selected.json`, ledger, `PROGRESS.json`, `EXCLUDED_FIXED_OBJECT.json`, README) are unchanged, and source git status is clean.
- **Where the mock wrote:** only under `dataset_placement_extension_v1/`, on the local branch `naren/shelf66v3-placement-mock-v1`, which is not committed and not pushed.
- **Leakage:**
  - every mock query is train;
  - lineage groups are closed;
  - `plan3d` reads only object ids, shapes, sizes, heights, cells, family id and goal (goal for flagging only);
  - no roles, parking, blockers, labels or specs are read (test `test_roster_uses_no_protected_metadata`).

## 11. CPU policy and host safety

| config | queries / wall-s | CPU-s per query | inflation vs serial | steady-state estimate (N / mean CPU) | max busy CPUs | peak °C |
|---|---|---|---|---|---|---|
| serial | 0.081 | 12.2 | 1.00 | 0.082 | 1.0 | 37 |
| 8 workers | 0.598 | 12.8 | 1.00 | 0.626 | 8.0 | 47 |
| **12 workers (chosen)** | **0.857** | 12.9 | 1.02 | **0.933** | 12.0 | 47 |
| 16 workers | 0.877 (fastest) | 17.1 | 1.29 | 0.935 | 16.0 | 48 |
| 18 workers | 0.863 | 19.2 | 1.54 | 0.939 | 18.0 | 50 |

**Choosing the worker count:**
- SMT siblings (CPUs 12–17) add no throughput: steady state is flat within 1%. They inflate per-query CPU by 29% at 16 workers and 54% at 18.
- The predeclared rule (fastest; fewer workers within 5%) selects **12**, the physical cores 0–11. That leaves 12 logical CPUs free for other users.

**Safety record:**
- No hard CPU quota is available: cgroup delegation covers only memory and pids. Enforcement was per-worker affinity, a pinned coordinator, a 1-thread BLAS, and the monitor.
- Across all 12 runs: 0 violations, 0 safety stops. The affinity union never exceeded the worker count (max 18 in `sweep_18`).
- Peaks: package temperature 50 °C, worker RSS 0.87 GB; minimum MemAvailable 51 GB.
- Worker RSS plateaus at about 0.6 GB within a run.
- `max busy CPUs 18.04` in `sweep_18` is psutil sampling jitter; affinity bounds it at 18.

## 12. Production forecast

The forecast covers `adaptive_v1` at 12 workers on the snapshot production design: 12,751 queries on 1,376 roots.
- Per-query costs are measured by (K × depth × band). 96 queries back off to depth × band.
- Uncertainty is a root-lineage bootstrap (2,000 resamples of the 24 pilot roots).

| quantity | value |
|---|---|
| CPU-hours | **30.2** (p05–p95: 26.8–33.5) |
| wall time | 0.110 days (2.6 h) expected; p95 0.122 days; **p95 × 1.5 buffer = 0.184 days (4.4 h)**, against the 7-day limit |
| disk | 0.46 GB (0.69 GB with buffer); positive rate 0.67 |
| alternative `fixed_4x1500` | 27.95 CPU-h (23.8–31.5), 0.173 days at p95 × 1.5, 0.43 GB |
| full 2,500-root quota (scaled by roots) | ≈ 55 CPU-h, ≈ 0.33 days at p95 × 1.5. **K4 (510) and NONE (408) are unsampled (`COVERAGE_BLOCKED`), so their per-query cost is an assumption** |
| doubling queries per root (20 slots) | about 2× CPU and disk; still < 1 day |

The forecast is valid for the snapshot. The live store must be re-audited and the roster rebuilt from it (RUNBOOK §4).

## 13. Figures (`mock_v1/figures/`)

- `scenes_srv-w0002-s002.png`: root, 1-edit and 2-edit layouts, with edits and query arrows coloured by label, plus the MuJoCo renders.
- `witness_srv-w0012-s007_q9_depth2_far.gif`: a verified depth-2 far placement, replayed from the stored witness (re-verified, bit-identical).
- `negative_srv3-w0058-s016_q6_bounded_search.gif`: the held state of a `BOUNDED_SEARCH` negative (no carry route within 5,000 iterations).
- `budget_policies.png`: miss rate vs virtual CPU for all 14 policies, with Wilson CIs.
- `broad_pilot_miss_by_band.png`: §6.2.
- `worker_sweep.png`, `resources_pilot_A.png`: CPU, temperature, memory.
- `pilot_labels_by_stratum.png`, `pilot_mode_accounting.png`.

## 14. Limitations and unresolved issues

1. **The live store was not inspected.** It is on 10.4.25.63; this mock is based on the GitHub snapshot (1,492 / 2,500 roots). K3 is incomplete and K4 is absent. Before any collection: copy the store read-only, re-audit, rebuild the roster.
2. **Label quality is relative to `reference_v1`, a finite policy:** ≤ 32 valid units at 5,000 iterations. It is not a physical ground truth, and confidence limits come from 26 / 138 positives; no population-level error bound is claimed.
3. **The pick-inaccessible fixture is mode level**; no query-level instance was found (§8.4).
4. **The structure check keeps the source's fully open fingers** (conservative, 0.55% of poses).
5. **Two post-freeze collector/bookkeeping amendments** (001, 002) were made. Neither touches placement semantics.
6. **The policy recommendation departs from the frozen rule's literal output** (§6.3). It needs your confirmation.
7. **IK screening is 84% of the cost.** A cache shared across lineages would cut it, but was not attempted, to keep the source's per-pose IK semantics.
8. **No hard CPU quota exists on gvlab;** the ceiling relies on affinity plus the monitor.
