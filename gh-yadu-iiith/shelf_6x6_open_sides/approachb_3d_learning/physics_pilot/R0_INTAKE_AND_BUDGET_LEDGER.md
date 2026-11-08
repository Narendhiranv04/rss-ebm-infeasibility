# R0: Intake, reconciliation and budget ledger (post-pilot contract v2)

**Branch:** `naren/approachb-3d-learning`. At intake, HEAD = `c11886750b8b6eecf4355cb8aa85e3717e2576cc`, equal to `origin`, worktree clean.

**Authority:**
- `final-fixed-methodology-1.pdf` (sha256 `b7a39e3c…`);
- `APPROACH_B_POST_PILOT_RECOVERY_AND_TRAINING_CONTRACT_v2.md`.

**Verdict: R0 PASS.**

## 1. Preserved inputs (verified)

| Input | Where | Check |
|---|---|---|
| Reports 00–13, physics pilot sources and manifests | this branch at `c1188675` | present |
| Reuse audit report + `approach_b_reuse_audit/` | `reports/reuse_audit_20261108/` and `~/Desktop/approachb_reuse_audit_preserved_20261108/` | all files match the audit's own `SHA256SUMS` |
| 3D teacher sources and roots on gvlab | `~/approachb3d_v3/repo/…/shelf_6x6_open_sides` | 37 / 37 source-pin hashes and 1,582 / 1,582 root hashes (verified in S1). Per-root robot protocol, config hash and scene source hashes are re-checked on every item |
| Simulator | gvlab `~/approachb3d_mock/env` | MuJoCo 3.16.0, as pinned by the generator |
| S1 + chunk-0 outputs | `physics_pilot/results_s1/`, `results_s2/` | summaries are re-derived from the item JSONs by `summarize_s1.py` |
| 3D v4 snapshot and Blackwell checkpoints | Blackwell | unchanged since `08`. The 3 new `cat_s*` checkpoints are mirrored on the second disk and on the laptop |

## 2. Canonical query ledger

**Builder:** `physics_pilot/r0_ledger.py`. **Output:** `r0/R0_CANONICAL_QUERY_LEDGER.json.gz` (uncompressed sha256 `c6cef78d…`), **7,660 X keys** and **10,419 Y keys**.

**X key** = `(root, ordered moves)` under BD3DS66_X_V1:
- q0 = q_ref = `q_ready`;
- stop-point connector;
- pinned teacher, assets and budget, identical for every key.

**Y key** = `(root, full endpoint arrangement)` under BD3DS66_Y_REPLAY_V1 (records) or BD3DS66_Y_PLAN_V1 (new endpoints). AB and BA share the Y key.

Every X row carries its ancestor link (the prefix structure in the key), the stored roles that attempted it, its attempt count, its split, its K (for reporting only), its Y key and its verification status.

## 3. Reconciliation (raw → unique → eligible → done → remaining)

**X relocation negatives (the 1,141 vs 866 question)**

| Form | Count | Derivation |
|---|---|---|
| Raw relocation-failed sequences (stored) | **1,141** | B records whose relocations did not all execute |
| Collapsed duplicates | **254** | **Every one** is a pair `noop_irrelevant` [d] + `redundant+1` [d, …] of the **same root** failing at the **same first move d**. From the identical S0 / `q_ready` this is one deterministic physical query, so 254 keys have 2 sequences each |
| Unique failing-prefix X keys | **887** (train 635 / val 124 / test 128; CERTIFIED 562, NO_PLAN_FOUND 285, OTHER_FAILURE 40) | 1,141 − 254 |
| Ineligible or out of scope | **0** | |
| Status contradictions (a key both failing and executed) | **0** | |
| Already re-confirmed (S1 8 + chunk 0 13) | **21** | |
| **Remaining eligible** | **866** | 887 − 21 |
| Attempted but unstored BFS tries | 9 | no record, so no key: **MISSING**, never 0 |

**Other pools**

| Pool | Raw | Unique keys | Done | Remaining |
|---|---|---|---|---|
| X executed prefixes (connector pending) | 3,665 complete sequences + proper prefixes | **6,773** (4,839 / 992 / 942) | 178 X = 1 verified | **6,595** (train 4,665 / val 988 / test 942) |
| Y record positives (planning, with witness) | 2,645 | 2,645 (no duplicate endpoints within a root) | 118 replay-verified | **2,527** |
| Y record negatives (operational) | 4,367 | 4,367 | 30 re-confirmed (pilot); all are valid operational zeros with kind kept | n/a |
| X endpoints with **no** Y record | | **3,407** (train 2,460 / val 463 / test 484) | 0 | 3,407. Of these, **2,929** are reached by executed prefixes (the audit's number) and 478 only by failing prefixes |

**Pilot overlap with the BWL-3 887-query manifest** (from `S0_OVERLAP.json`; nothing re-collected):
- 151 of its X queries are already in this ledger (134 executed + 17 failing);
- 129 of its endpoints have Y records;
- its other 736 queries are fresh executions, deferred to the candidate-diverse stage.

**Split isolation:**
- All keys inherit their root's frozen split, and no key crosses splits.
- Test-split rows are recovered as **held-out labels only**: never used for tuning, never for a gate.
- The 75 DEPENDENT roots are not in this ledger; they form the separate challenge cohort.

## 4. Host safety preflight (07:45Z)

**gvlab:**
- Q2 spillover healthy: 14 processes, 12 workers on CPUs 0–11, 414 / 2,188 roots.
- **CPUs 16–17 idle** (kernel threads only).
- Package at 48 °C (Q2 pauses at 88 °C); 59 GB available; swap unchanged.
- Users present: sajid, abhishek, furqan; none of them running on CPUs 14–23.

**Laptop:** Q2 remainder at 401 / 2,231 roots.

**Guards (unchanged from S1):**
- workers pinned with taskset to CPU 16 or 17, nice 19, idle I/O;
- the worker guard stops on package ≥ 75 °C, MemAvailable < 16 GB, or swap growth > 256 MB;
- stops are by owned PID only;
- the executor's name matches none of the Q2 process-scan substrings.

**Blackwell:** guard check passes, self-stop test passed earlier, CPU 7 / GPU 0 free, script allowlist extended to the new entry points.

## 5. Budget ledger (separate caps; measured process CPU, retries included)

| Cap | Authorized | Ledger |
|---|---|---|
| R1 historical recovery, gvlab | **60 CPU-h**, ≤ 2 workers, CPUs 16–17, ≤ 42 h elapsed | `out_r1/R1_BUDGET_LEDGER.json`, written by `bwlsuper.py` after each chunk. Workers also stop on `STOP_BUDGET` between items |
| R2 missing-Y pilot, gvlab | **6 CPU-h** | `out_r2/` item `cpu_s` totals (`summarize_r2.py`) |
| R4 Blackwell | **6 GPU-h**, serial | per-run wall time × 1 GPU, logged in each run's JSON |

The work done before this contract (S1 1.23 + chunk 0 0.84 = **2.07 CPU-h**) is not charged to these caps.
