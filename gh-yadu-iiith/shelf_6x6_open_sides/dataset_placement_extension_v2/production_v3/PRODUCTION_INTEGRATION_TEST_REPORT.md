# Production v3 integration tests: the real launch path, validated without authorizing production

**Result:** every production-path check passed with the final code, on `nav@gvlab`:
- a bounded physical end-to-end smoke;
- kill and corruption recovery;
- 100% fresh-process witness replay;
- every refusal path;
- the exclusive lock;
- the zero-planner-call metadata preflight over all 13,651 production queries.

Evidence is in [`integration/`](integration). Machine-readable results are in [`integration/integration_results.json`](integration/integration_results.json) and [`integration/production_metadata_preflight.json`](integration/production_metadata_preflight.json). The test logs are [`integration/pytest_v3.txt`](integration/pytest_v3.txt) and [`integration/pytest_v2.txt`](integration/pytest_v2.txt).

## 1. What changed in the code, and why

The frozen V2 scientific teacher and collector are **byte-identical**: `placement3d.py`, `policy3d.py`, `plan3d.py`, `run3d.py` and `guard3d.py` all have the V2 hashes, and the teacher protocol is still `89cbabed5d7e4847`. All production repairs are new modules next to them:

| Module | Purpose |
|---|---|
| `code/prod3d.py` | Production guard v3, the exclusive lock and the template writer. **Every** run parameter comes from one frozen policy file: mode, policy, implemented stages, workers, CPU mask, unit, splits, run dir, roster, pin, split, QC roots, cohort manifest, forecast and `SHELF_SRC`. Nothing falls back to a default or `*_live` path. The guard checks frozen hashes of all inputs, of the collector, guard and teacher code, and of the source at `SHELF_SRC` (`selected.json`, `EXCLUDED_FIXED_OBJECT.json`, generator and scene code, robot model, every roster root). It also checks that `stages` equals the **implemented** schedule of the policy, that the policy equals the one the mock-V2 decision locked, and that the teacher protocol equals the protocol the V2 evidence was produced with (so a teacher change forces revalidation). Then the token (which binds the authorization text), the sensor, disk ≥ forecast × 1.5 + 5 GB, CPUs 0–11 idle, and no other collector or lock holder. |
| `code/collect3d.py` | Production collector v3. It reuses run3d's worker, atomic storage, monitor and resume **unchanged**, and adds: the guard; the exclusive flock taken after every refusal check and before any directory is created; workers that die with the coordinator (`PR_SET_PDEATHSIG`); unit validation extended to the frozen protocol, the pinned root bytes and the **label semantics**; fsynced ledger lines; termination of a torn last ledger line; a session-leader-safe process group; and integration-test marking. |
| `code/qc3d.py`, `code/cohort3d.py`, `code/source_final3d.py` | Source QC, cohort freeze and final-state audit (see `SOURCE_QC_AUDIT.md`) |
| `code/decide_v3.py` | Derives the readiness decision mechanically. It replaces the hard-coded verdict strings of `decide_v2.py`, which stays unchanged as a V2 artifact. |
| `code/integration3d.py`, `code/test_prod3d_v3.py`, `code/finger3d.py`, `code/figs_v3.py` | Integration smoke, tests, finger-width sensitivity and figures |
| `production_v3/launch_first_cohort.sh` | The inert launcher |

**Defects found and fixed in this pass.** Each now has a test.

1. **Check-then-start race.** `guard3d.other_collectors()` scanned `/proc`, so two launchers could both pass. It is replaced by a kernel `flock`, which is atomic and released when the holder dies.
2. **Template could disagree with the run.** `prod_template.py` hard-coded the `production_v2/*_live` paths, and the V2 guard never compared the policy file's `stages` with the schedule actually run. The template is replaced by `prod3d.py template`, which takes explicit arguments, and the guard now compares against the implemented schedule.
3. **Session-leader crash.** A collector exec'd as a **session leader** crashed on `os.setpgrp()` (EPERM) right after taking the lock. That happens, for example, when tmux's shell execs the launcher. Found in integration attempt 1 and fixed with `own_group()`.
4. **Lost ledger record.** A torn last ledger line made the next ledger record unparseable. Found in attempt 2 and fixed with `ledger_open()`. In that attempt the collector itself had correctly quarantined and recomputed the damaged unit.
5. **Unprotected source.** Source scene and generator code was not pinned by the production guard, only root bytes. It is now pinned.
6. **Unprotected label semantics.** A label-0 record carrying witnesses, or an `INTERRUPTED` record marked negative, would have passed V2 validation. They are now refused on resume and on every finished unit.

Attempts 1 and 2 are kept, each with a `NOTE.md`, in `integration/attempt1_aborted/` and `integration/attempt2_superseded/`.

## 2. Bounded physical end-to-end smoke (attempt 3, final code)

The smoke used the same collector, guard, lock, pinned workers, atomic JSON + NPZ, monitor, resume and replay as production. It ran as **`INTEGRATION_TEST`**, which means:

- a separate 12-row roster (`integration/smoke_roster.jsonl.gz`), every row byte-identical to a production TRAIN row;
- a test fixture (`test_authorization.fixture = INTEGRATION_TEST_ONLY_NOT_PRODUCTION`) whose **user-authorization fields stay empty**;
- run directories under `integration/`, a `TEST_RUN_NOT_PRODUCTION.txt` marker, and the mode recorded in the manifest and every ledger record.

The guard refuses a test fixture switched to production mode, and refuses a test roster with more than 12 TRAIN queries or more than 4 roots.

**Roster selection (label-blind):**
- 4 TRAIN roots, one per K, hash-ranked: two author-approved, two QC-certified here;
- per root, the depth-0 far, depth-1 vacated and depth-2 far query.

| Check | Result |
|---|---|
| Smoke A (4 workers, CPUs 0–3) | complete: 4 units, 12 queries in 50 s. Labels: 8 × 1 `VERIFIED`, 1 × 0 `NEG_CERTIFIED_GEOMETRY`, 1 × 0 `BOUNDED_SEARCH`, 1 × 0 `DETERMINISTIC_PLAN`, 1 × None `UNLABELED_NO_VALID_POST_GRASP_STATE` |
| Smoke B: **SIGKILL of the coordinator only**, right after its first unit with stored witnesses | 1 of 4 units done. **0 of 5 descendants alive 3 s later** (workers have PDEATHSIG; the resource tracker exits). **Lock free** immediately. |
| Deliberate damage before resume | that unit's NPZ truncated (87,114 → 43,557 bytes), and a torn line appended to the ledger |
| Resume (identical command) | the damaged unit failed validation (`npz_sha_mismatch`) and was quarantined with `REASON.json`; the torn line was terminated and ignored; the unfinished units were recomputed; complete |
| B vs A | **12 / 12 queries identical**: label, status, negative kind, verified rank and branch, certificate, and witness ids |
| Fresh-process MuJoCo replay of **every** stored witness association | A: **8 / 8 verified**. B: **8 / 8 verified**. Executed joints identical (max difference 0), hold identical, 0 failures. |
| Post-run audit (`collect3d.py --audit`) | A and B complete: ledger ↔ JSON ↔ NPZ ↔ witness hashes, protocol, root pin and label semantics all valid, with 0 stray files |
| Monitors | 0 violations; CPUs {0, 1, 2, 3} only; peak 41 °C; at least 59.5 GB of memory available; 0 unexpected processes |

**Negative and unlabeled provenance** (`provenance_A` in the results):
- the label-0 certified negative carries its certificate scope: 32 poses × ≤ 4 source IK candidates × the 2-D footprint model;
- the policy negatives carry their scope: stages, units tested;
- the unlabeled query carries its screen counts and no witness;
- no record has a label-semantics problem, and every record has 0 unaccounted branch attempts.

## 3. Metadata preflight of the exact production template (0 planner calls)

`collect3d.py --policy-file production_policy.TEMPLATE.json --dry-run` covers all **13,651** physical queries of all splits:

| Item | Value |
|---|---|
| Units / roots | 1,466 / 1,466 |
| Static rows excluded from units | 2,768 |
| Queries by split | train 9,664, val 2,066, test 1,921 |
| `planner_calls`, `teacher_entry_calls` | **0**, **0**. Every teacher entry point (Scene, screen, hold, plan_branch, execute, verify, evaluate, run_unit) is counted. |
| Directories created | none (`run_dir_created: false`) |
| Roster regenerated from the cohort roots by the frozen rule | identical |
| Refusals other than authorization | **none** |
| Authorization refusal | "no recorded user authorization" (the template is unauthorized) |

## 4. Test suites

**`code/test_prod3d_v3.py` (23 tests, all passed)** drives the real CLIs: `bash launch_first_cohort.sh`, `collect3d.py`, `prod3d.py`. Every test carries a deliberate refusal condition, so none can start a collection, and no test ever computes a token for a production-mode file.

**Launcher refusals:**
- no arguments;
- `--execute`, `--confirm` or `--policy-file` missing;
- `SHELF_SRC` not exported;
- unauthorized template with a wrong token;
- conflicting lock held by another process.

**Collector refusals:**
- unauthorized policy, or a missing or false token;
- altered roster: hash, rows outside the cohort roster, a roster other than the one the cohort manifest froze;
- altered source pin;
- root drift inside `SHELF_SRC`, and a different `SHELF_SRC`;
- stages not equal to the implemented schedule, a non-locked policy, 13 workers or CPU 12;
- a test fixture with more than 4 workers;
- an unreadable sensor;
- a test-only override set in production;
- a test fixture switched to production, or carrying filled authorization fields;
- a test roster that is too large or includes val rows.

**Lock tests:**
- a holder blocks the collector and a second `acquire`;
- SIGKILL of the holder frees it;
- 8 simultaneous launchers produce exactly 1 winner;
- leftover lock text is not a lock.

**Other tests:**
- the dry-run preflight;
- label semantics on crafted records and on all 405 real V2 records;
- extended unit validation (protocol and root-pin mismatch);
- a teacher change produces a new protocol and is refused with "revalidate first";
- the template records the actual inputs, and the committed template is unauthorized;
- the V2 teacher and collector bytes are unchanged;
- the session-leader process group;
- torn-ledger-line termination.

**`code/test_mock3d_v2.py` (25 tests):** the V2 suite, re-run unchanged. The result is in `integration/pytest_v2.txt`.

One environment-dependent behaviour is by design. The first v3 run (`integration/pytest_v3_first.txt`) failed 2 tests because a 12-worker finger-width job was occupying CPUs 0–11. The production preflight correctly refused with "CPUs 0-11 are 100% busy with other work". On an idle host, all pass. The launcher runs both suites immediately before starting, when the host must be idle anyway.

## 5. Resource discipline during this pass

| Job | CPUs |
|---|---|
| QC re-run, finger-width runs | 12 pinned workers on CPUs 0–11 |
| Smoke | 4 workers on CPUs 0–3 |
| Tests | CPUs 12–13 |

At most 14 logical CPUs were in use by this project at any time, under the 18-CPU ceiling. Peak package temperature was 45 °C. The affinity-plus-monitor scheme is **not** a hard cgroup quota (gvlab has no cgroup CPU controller); it is enforced by pinning and by the monitor, which stops the run on any process outside the mask.
