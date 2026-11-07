# Production v4 validation: job states, code changes, physical integration, guards, forecast, monitoring

## 1. Job states (observed, not assumed), 2026-11-07

| Job | State | Evidence |
|---|---|---|
| **2D Q2** (laptop) | **not_started** | The host process table, read outside the VS Code sandbox, has no `launch_runner.py`, `launch_guard.py` or `launch_full_2d.sh`. There is no tmux server or crontab. The configured `full_run_dir` (`…/dataset_relocations_v1/full_2d_v2`) does not exist, and `launch_ready_2d_v2/` holds only a superseded `production_policy`. Nothing 2D was touched. |
| **3D v3** (gvlab, 13,651 queries) | **not_started** | There is no `production_v3/run_first_cohort/`, no collector process and no lock holder; the lock file holds only the text of a finished pytest. No verified v3 label exists, so there is **no base to reuse**, and v4 is a fresh full roster. |
| **3D v4** (this) | prepared, **not started** | Authorization template only. |

## 2. Code: smallest versioned changes; frozen v2 / v3 bytes unchanged

| File | Role |
|---|---|
| `code/q2roster3d.py` (new) | The `Q2_3D_PARITY_V1` rule, coverage gates, smoke selection, stratified forecast, training-utility report, coverage figure |
| `code/prod4.py` (new, about 60 lines) | Re-parameterizes the **unchanged** v3 guard `prod3d.py`: version `SHELF66V3_PLACE_PRODUCTION_V4_Q2_PARITY`, `production_v4/` run prefixes, tests ≤ 32 TRAIN queries / 4 roots, the v4 code set (adds `q2roster3d.py`, `prod4.py`, `collect4.py`), and a collector scan that sees the v4 entry points |
| `code/collect4.py` (new, about 50 lines) | Runs the **unchanged** v3 collector `collect3d.py` with the v4 guard bound in and a v4 preflight that **regenerates the whole roster** from `SHELF_SRC` and compares it line by line |
| `code/integration4.py`, `code/test_prod4.py`, `code/decide_v4.py`, `code/status4.py` (new) | Physical integration, tests, mechanical decision, 10-minute status / backup service |
| `production_v4/launch_v4.sh` (new) | Inert launcher: preflight → guard → v4 and v2 tests → collector |

Byte-identical and verified by tests and manifests:
- the teacher `placement3d.py` and `policy3d.py` (protocol `89cbabed5d7e4847`);
- `plan3d.py`, `run3d.py` and `guard3d.py`;
- the v3 `prod3d.py` and `collect3d.py`;
- every `production_v3/` file (its `SHA256SUMS`) and every V2 file (`ARTIFACT_MANIFEST.sha256`).

The only file added to `production_v3/` is `SUPERSEDED_BY_V4.md`, a marker.

**Cross-version refusal (tested):**
- `collect4` refuses a V3 policy: "is not a SHELF66V3_PLACE_PRODUCTION_V4_Q2_PARITY file".
- `collect3d` refuses a V4 policy.
- The token binds the version, the v4 roster, manifest and forecast hashes, and the v4 code set, so **a v3 token can never authorize v4**.
- v3 and v4 share the exclusive collector lock, so they can never run together.

## 3. Bounded physical integration on the NEW roster (`integration/integration_results_v4.json`)

The run was **32 TRAIN queries on 4 roots**, one per K, label-blind and hash-ranked:
- 2 author-approved roots and 2 roots QC-certified by the re-run;
- the K3 root has 6 or more non-target objects (heavy clutter);
- per root: root near / mid / far plus a second object; prefix1 at the vacated cell; prefix2; prefix3 at the vacated cell plus a second object.

It ran as `INTEGRATION_TEST`, with the user-authorization fields empty, the test marked in the manifest and a marker file, and run directories under `integration/`.

| Check | Result |
|---|---|
| Smoke A (4 workers, CPUs 0–3) | complete: 32 queries in 76 s, 8.7 CPU-s per query |
| Labels (all three kinds, across blocks) | 21 × 1 `VERIFIED` (root 10, prefix1 2, prefix2 3, prefix3 6); 7 × 0: 2 `CERTIFIED_GEOMETRY`, 2 `BOUNDED_SEARCH`, 3 `DETERMINISTIC_PLAN`; 4 × None `NO_VALID_POST_GRASP_STATE`. Vacated cells gave 5 positives and 3 unlabeled. |
| Smoke B: SIGKILL of the coordinator alone after its first unit with witnesses | 0 of 5 descendants alive 3 s later; lock free |
| Then a truncated NPZ and a torn ledger line; resume with the same command | the damaged unit was quarantined (`npz_sha_mismatch`) and recomputed; a unit written but not yet logged was quarantined as an orphan and recomputed; the torn line was terminated and ignored; complete |
| B vs A | **32 / 32 identical**: label, status, kind, rank, branch, certificate, witness ids |
| Fresh-process MuJoCo replay of every stored witness | A **21 / 21**, B **21 / 21** verified; executed joints identical; hold identical |
| Post-run audits | complete: ledger ↔ JSON ↔ NPZ ↔ witness hashes, protocol, root pin, label semantics |
| Monitors | 0 violations; CPUs {0–3}; peak 39 °C |
| Training-utility report | `integration/training_utility_smoke_A.json` |

## 4. Production guards and preflight

`collect4 --dry-run` of `production_policy_v4.TEMPLATE.json`:
- regenerates all **59,972** queries and finds them **identical** to the frozen roster, with all semantic keys unique;
- makes **0** planner calls and **0** teacher-entry calls;
- creates no directory;
- has no refusal other than the missing user authorization.

`code/test_prod4.py` has **12 tests, all passed**, on the real CLIs:
- the per-root counts follow the Q2 formula (0 / 8 / 24 / 44 / 56);
- the prefix contexts are independent, valid edits;
- the rule ignores 14 scrambled protected fields;
- every destination is free in its own scene, and the bands follow Chebyshev distance;
- the roster regenerates byte-identically;
- the launcher refuses no arguments, and refuses an unauthorized template with a wrong token;
- **a v3 policy or token never authorizes v4**;
- an altered roster, schedule, CPU mask or test size (33 rows) is refused;
- sensor loss and a held lock are refused;
- the full dry run makes zero teacher calls;
- the collector scan sees the v4 entry points;
- the frozen v2 and v3 code is unchanged.

`code/test_mock3d_v2.py` (V2, unchanged) has **25 tests**. The run on the current tree is in `integration/pytest_v2.txt`.

## 5. Forecast (`frozen/forecast_v4.json`)

**Method.** The cost per physical query is the measured `adaptive_v1` run A of the V2 mock: 12 pinned workers, the same protocol, unit = root. It is stratified K × depth × band × shape, then backs off through K × depth × band → K × depth → K → all, with at least 3 samples per level. A root-lineage bootstrap (2,000 resamples) gives the spread, and the parallel efficiency is 0.881, from the V2 12-worker sweep.

**Held-out timing checks.** Measured CPU against the model's prediction, on queries the model never saw:

| Sample | Measured / predicted |
|---|---|
| v4 integration smoke: 32 queries, production mode (unit = root) | **1.005** |
| v3 integration smoke: 12 queries, unit = root | 1.289 |
| First-cohort TRAIN sample: 144 queries, unit = query (IK candidates not cached within a root, so slower) | 1.425 |

| Scenario | Wall hours (12 workers) |
|---|---|
| optimistic (bootstrap p05) | 14.5 |
| **base** (165.4 CPU-h) | **15.6** |
| production-mode held-out calibration (pooled 1.087) | 17.0 |
| conservative (bootstrap p95) | 17.2 |
| held-out calibrated (largest ratio, 1.425) | 22.3 |
| **buffered** (1.5 × max(p95, base × 1.425)) | **33.4** |

**Storage:** 2.57 GB expected (positive rate 0.61 × 36 KB per witness + 21 KB of JSON per query), **3.85 GB buffered**. gvlab has 61 GB free and `/ssd_scratch` has 77 GB.

**Thermal:** the 12-worker V2 runs peaked at 47–49 °C. The monitor pauses at 90 °C and stops at 95 °C.

## 6. 10-minute status, GitHub sync, backups

The service is implemented as `code/status4.py` (see `V4_RUNBOOK.md` §3). It was verified end to end with two manual cycles: gvlab snapshot → queue → laptop relay → telemetry commits `5ea10d9` and `197508b` on the new branch `naren/3d-dataset-relocations-telemetry`, which holds status files only. The gvlab queue items were removed only after each push succeeded.

| | State |
|---|---|
| `STATUS_SERVICE` | `IMPLEMENTED_AND_CYCLE_VERIFIED__SCHEDULER_NOT_INSTALLED`. Installing the two cron lines was refused by the assistant's permission policy (persistence); the user installs them (runbook §3) before launch. |
| `OFFHOST_BACKUP` | `OFFHOST_BACKUP_BLOCKED`. No durable off-host destination has been designated, and none was invented. Implemented interim copies: the hash-verified second-disk copy on gvlab `/ssd_scratch` (shared scratch), and the hash-verified laptop mirror (only while the laptop is on). |

The measured state is in `status_service/STATUS_SERVICE_STATE.json`.

## 7. Incident in this pass (no remote effect)

The first attempt to create the telemetry branch staged the whole repository tree. `git clone --no-checkout` still populates the index, so the orphan commit would have carried 97,632 files. I stopped it before its push. `git ls-remote` showed no telemetry branch on GitHub, and the local clone was deleted. The branch was recreated from an empty index, and its root commit `02e0272` contains only `README.md`.
