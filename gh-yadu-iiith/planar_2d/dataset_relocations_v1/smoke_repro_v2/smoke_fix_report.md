# 2D placement smoke test: repair and revalidation report (6 Nov 2026)

## 1. Executive verdict

**`SMOKE_PASS`**

The verdict covers the frozen 35-query smoke roster only. It does not license any larger run.

| Gate | Verdict | Evidence |
|---|---|---|
| `SEMANTICS` | **PASS** | Queries start from an already-held state; pick access is a diagnostic only. Witnesses are carry → release → retreat, replayed phase by phase. A fixture shows pick access 0 with a verified held placement (label 1). |
| `REPRODUCIBILITY_1V16` | **PASS** | Serial and two fresh 16-worker runs have identical canonical fingerprints, with 0 mismatches over all 35 queries. |
| `REPRODUCIBILITY_28_STRESS` | **PASS** | Two fresh 28-logical-CPU runs completed safely, with fingerprints identical to serial and 16-worker. |
| `RESOURCE_SAFETY` | **PASS, with caveats** | No swap, at least 10.3 GB free, 0 leftover processes. A preliminary run was correctly thermally cancelled at 97.5 °C. Short spikes reached 96.25 °C (§8). There is no hard cgroup quota; only affinity enforces the CPU cap. |
| `STORAGE_REPLAY` | **PASS** | Every run audits clean (1,530 checks each). All 222 witnesses re-verify from disk. Re-running a finished run is idempotent; a SIGKILL followed by resume leaves committed records byte-identical. |
| `DATA_INTEGRITY` | **PASS** | All 205 source files and all 1,129 files of the failed run match their pre-work hashes. git reports the original dataset clean, with no bytecode written. |

**What a completed operational 0 means.** The query was eligible, and the whole declared deterministic work policy ran to the end without producing a verified witness. That policy is at most 8 (held, place) pairs, each with a fixed work budget. A 0 is not a proof that no placement exists: it is a statement about this finite policy. The only exception is a 0 marked `is_certificate`, which requires every one of the 2,304 pairs to fail a definite static screen; that never happened here.

Queries that were not attempted, statically ineligible, or stopped by a safety guard carry `label: null`. They are never 0, and there is no third observed class.

## 2. Source and worktree audit

**Location and branch:**
- Worktree `/home/naren/Desktop/rss-placement-smoke`, branch `naren/placement-smoke-2d`.
- HEAD `9e64069038928dfae1e7f89391587e2f595374fc`. Nothing is committed, pushed or reset.
- `git status`: the only change is the untracked `gh-yadu-iiith/planar_2d/dataset_relocations_v1/`.

**Preserved evidence:**
- The failed run `smoke_20261106/` is untouched and write-protected; all 1,129 files still match their pre-work hashes.
- The previous scripts are copied to `audit/previous_code/`.
- The full diff is in [`audit/code_changes.diff`](audit/code_changes.diff) (1,415 lines).

**What was actually wrong** (from the local code, not the earlier summary):

1. **Wall clock decided outcomes.** The old code set `max_wall_seconds = 2.0` in `planner.plan`.
   - In `planner._rrt_connect` (unchanged here), search stops on `nodes ≥ max_nodes` or `time.time() > deadline`. A failed extension adds no node, and a `PoseSpace` edge can fail inside IK without any collision evaluation, so only the clock bounded failed work.
   - `planner._shortcut` also stops at the deadline.
   - The 12 s per-query cap stopped scheduling new pairs by clock.
2. **Pick-phase checks leaked into placement.**
   - The old `post_grasp()` re-implemented `skill.screen_pick` minus the approach line, but still required the open-finger grasp pose (pick context) and a certified closure motion.
   - Each witness began with a `close_attach` segment, i.e. the closure motion.
   - On this roster those two checks happened to reject no extra mode: 371 valid modes under both v1 and v2. So v1's labels were not changed by this, but its semantics and witnesses were wrong.
   - `skill.screen_pick` itself, with the approach line, would have kept only 305; v1 never called it.
3. **Nominal attachment on place screens.** Place modes were screened with the nominal attachment `(grasp_x, 0)`, as `skill.plan_skill` does, rather than each held mode's own `held_local` and gap. v2 screens per distinct attachment; on this roster all modes share one attachment within 1e-12.

| | v1 (failed run) | v2 (this run) |
|---|---|---|
| Protocol id / budget hash | `544a5c7b9b1bdcb6` / `3708dc9d6ab09535` | **`20d06d12096c2113` / `b31f9ccc06824db2`** |
| Event start | pre-closure (close_attach) | **already held (carry start)** |
| Stop rule | 2 s per pair, 12 s per query | **counted work only** |
| Code sha256 (`placement_smoke.py`) | `fa995712…` | `fef218feb5ea7a25…` |

**Teacher.** The teacher (`skill`, `motion`, `planner`, `geometry`, `config.toml`, `p16_motion`, `bd_2d`) is unmodified, and `planner.py` is not edited. The smoke code imports `planner.JointSpace`, `PoseSpace`, `_join`, `_shortcut`, `link0_sweep_blocker` and `call_seed`, and runs its own counted loop (`det_rrt`). Teacher config hash `90d3eb68e8a42108` and robot profile `sha256:2d7986…ae858` equal the roots'.

**Legacy test.** `test_existing_json_npz_readers_still_load` reads `runs/p16/…` and `runs/p14/…`, which do not exist in commit `9e640690` (`git ls-tree`: 0 matches). It is a pre-existing missing-fixture failure, not a regression.

## 3. Label semantics

**Event** (`V_place`, protocol `PLACEMENT_SMOKE_V2`): object *o* is already held at its source pose in a declared post-grasp mode. Can carry → release → straight local retreat to cell *d* be certified?

Excluded:
- motion from the start pose;
- pregrasp and approach;
- the open-finger grasp pose;
- the closure motion;
- any home or q_ref connector, and the target task.

**Held modes (48 = 24 tool axes × 2 elbows, `skill.directions`).** A mode is rejected only for its own held geometry:
- exact IK of the held tool–object transform (joint limits included);
- the contact model of `skill.screen_pick`: lateral offset ≤ `contact_tol`, centre inside the finger span, closed gap ≥ `closed_gap`;
- the attached configuration not clear in the carry context (robot bodies, held disc, other objects, support). This uses the same `Checker` that certifies carries.

`skill.screen_pick` still runs per query, but only as the diagnostic `pick_access_stage`; it never vetoes a label.

**Place modes (48).** `skill.screen_place` (IK, held pose, release, retreat) with each held mode's own `held_local` and closed gap.

**Pairs.** 48 × 48 = 2,304 per query. A pair is hard-screened when either mode fails a definite check. Valid pairs are planned in a fixed order (same elbow branch first, then joint distance, then indices), at most 8 per query.

**Witness validity** (`verify_witness`). Accepted only if all of these hold:
- phases exactly `carry+ release retreat+`;
- `attached` true exactly on carry segments, with a constant closed gap during the carry and an open gap after release;
- the object at its source at the start and within `release_tol` of the cell centre after release;
- every segment re-certified by `motion.certify_trajectory` in the query's own scene (that includes `check_record`: limits, gaps, attachment consistency).

The same check runs at creation, in the auditor (from the saved files), and for reuse.

**Reuse.** An archived dataset witness is reused only when:
- its first move is the same (root scene, object, destination);
- its `carry…retreat` suffix starts and releases at configurations equal to declared modes that pass v2's screens;
- it passes `verify_witness`.

All 5 candidates did; each is marked `evidence_source: REUSED_REVERIFIED` and counted in its own category.

**Hypothetical scenes.** Object-only `substitute_pose` edits, recorded with:
- `parent_reachability: HYPOTHETICAL`;
- `prefix_key`, `edited_objects` and `hypothetical_scene_hash`.

No achieved robot state is inferred, and the root scene is never mutated (tested).

## 4. Determinism implementation

**Work units** (per (held, place) pair, `WORK` is part of the protocol hash):
- **Extension attempt:** one call of extend (nearest + steer + edge). Failed ones count, so a failing loop always ends; at most **1,000**.
- **Collision evaluation:** one `Checker.distances` call (`n_evals`), including those inside rejected edges; at most **20,000** during search.
- **Accepted nodes:** at most **200**.
- **Shortcutting:** always the full **150** rounds, with no clock (`_shortcut(..., deadline=inf)`); its evaluations are recorded.
- **Pre-RRT stages:** direct, guide and link-0 certificate, bounded by `certify_max_evals` and recorded as `pre_evals`.

Limits are checked before each RRT-Connect step. Because a step's second extension may run before the next check, a counter can exceed its limit by at most one extension (seen as 1,001 attempts). The audit checks exactly this bound.

**How the limits were chosen** (frozen before any canonical run, from measurements, not labels):
- the failed run's work distribution: verified searches used up to 92–98 nodes;
- one calibration of the reference query: verified searches needed 28–72 attempts and ≤ 969 search evaluations; shortcutting about 2,300 evaluations; about 1,600 evaluations per second.

In the canonical runs, verified searches used at most 712 attempts, 136 nodes and 5,772 search evaluations. Budget stops were `NODES` 32 and `ATTEMPTS` 23; `EVALS` never bound.

**Seeds.** `planner.call_seed(seed=41, crc32("<trial_id>/carry<i>.<j>"))` per pair. No dependence on Python's `hash()`, PIDs, worker index, dispatch order or directory order. Ties are broken by mode indices, with one BLAS/OpenMP thread per worker.

**Emergency guards are separate.** None of these can produce a 0; each yields `completed: false`, `label: null` and a reason, and is re-run identically on resume:
- a per-query 1,800 s alarm;
- a per-run ceiling;
- memory, swap, disk and temperature guards.

Thermal duty-cycling freezes all workers (SIGSTOP) at ≥ 92 °C and resumes them at ≤ 85 °C. Since budgets count work, pausing cannot change an outcome; this was verified by identical fingerprints across runs with different suspension patterns.

## 5. Frozen roster

[`roster.json`](roster.json) is byte-identical to `smoke_20261106/roster_predeclared.json` (sha256 `02297bfd…`). It is regenerated by unchanged code and compared at every audit. No query was added, removed or replaced.

| Class | Root | Objects × destinations (archived parking, near, mid, far) | Extra |
|---|---|---|---|
| K2 | `2d-plus2d-srv-w0230-s011` | o01, o00 × 4 | hypothetical: after o01 → 4, o00 → 3 |
| K3 | `2d-plus2d-srv-w0408-s006` | o00, o02 × 4 | hypothetical: after o00 → 63, o02 → 45 (the vacated cell); at the root, o02 → 45 is ineligible (occupied) |
| K1 | `2d-conv-plus2d-srv-w0068-s006` | o00, o01 × 4 | — |
| K0 | `2d-plus2d-srv-w0111-s001` | o01, o00 × 4 | — |

There are 35 queries: 34 eligible and 1 statically ineligible. The 2D dataset defines no split, so every record has `split: null`.

## 6. Result counts (every canonical run is identical)

| | |
|---|---|
| Queries / eligible / completed / interrupted | 35 / 34 / 34 / 0 |
| Label 1 / label 0 / null | **30 / 4 / 1** (the occupied cell) |
| Zeros | all `WORK_BUDGET_EXHAUSTED`, failure phase `CARRY`, `is_certificate: false`: K0 o01 → 31, 13, 37, 53 (8 pairs each planned to budget, 0 witnesses) |
| Held modes valid / pick-access OK (diagnostic) | 371 / 305 over the 34 queries |
| Pairs (34 × 2,304 = 78,336) | hard-screened 70,358 + motion-attempted 272 + reused 5 + not motion-attempted 7,701 (screen-inconclusive 927, valid but beyond the 8-pair schedule 6,774) |
| Motion-attempted pairs | 217 verified, 55 failed (budget) |
| Planning methods | direct 92, guide line 17, RRT object-pose 108 solved, budget-exhausted 55 |
| Witnesses | 222 unique files = 217 newly planned + 5 reused (re-verified) = `n_mode_witnesses` |

**Recount.** The old counts (70,825 geometry-rejected, 233 motion-attempted, 7,278 untested) are not inherited: everything was recomputed under the v2 categories.

**Labels against v1.** Two v1 zeros became 1 under deterministic budgets: K1 o01 → 26 and → 0. All their v1 carry plans had been cut by the clock. The four K0 o01 zeros remain 0.

## 7. Repeat comparisons

| Run | Workers / CPUs | Canonical fingerprint |
|---|---|---|
| `serial` | 1 / {0} | `1ed5e19f5c061e5e…` |
| `workers16_repeat1` | 16 / physical cores 0–15 | `1ed5e19f5c061e5e…` |
| `workers16_repeat2` | 16 / 0–15 | `1ed5e19f5c061e5e…` |
| `workers28_repeat1` | 28 / 0–13 + 16–29 (both SMT threads) | `1ed5e19f5c061e5e…` |
| `workers28_repeat2` | 28 / same | `1ed5e19f5c061e5e…` |
| `interrupt_resume` | 16, SIGKILL then resume | `1ed5e19f5c061e5e…` |

[`comparisons.json`](comparisons.json): **0 mismatches**, 35 of 35 queries compared in all six runs.

**What a fingerprint covers.** sha256 over:
- each record without its `volatile` field (wall time, CPU, RSS);
- every mode array (dtype, shape, bytes; not the zip container);
- every witness's arrays.

The bitwise comparison includes the labels, mode categories and masks, per-attempt work counts and stop reasons, ordered witness identities, and the exact waypoint arrays. Timing, temperature, PIDs and scheduling are excluded.

**Interrupt test:**
- Killed with SIGKILL (parent and all 16 workers) once 10 records were committed. At that point there were 0 temp files and 0 surviving processes.
- The resume re-ran only the 25 unfinished queries, after deleting 35 orphan trajectories from queries killed in flight.
- All 19 committed files (10 records, 9 mode arrays) were byte-identical after the resume; the audit was clean and the fingerprint identical.
- Re-running finished `serial` changed none of its 291 files (0 queued, 0 CPU-s).

**Preliminary runs** (kept, not canonical):
- `prelim_serial_codeA` used the earlier driver, before duty-cycling; it is identical to `serial` on all 35 queries apart from `code_sha`.
- `prelim_workers16_thermal_cancel` was cancelled by the guard at 97.5 °C after 36.6 s. Its 2 in-flight queries were recorded with `completed: false`, `label: null`, `safety_interruption: TEMPERATURE`. Its 33 final records were identical to `prelim_serial_codeA`.

## 8. Resource report (local laptop only; no SSH or GPU)

**Machine:** AMD Ryzen 9 8940HX, 16 cores and 32 logical CPUs (siblings *i* and *i*+16), one NUMA node, 31 GB RAM with 16 GB swap. Preflight: 14.2 GB available, swap unused, load 1.3, 67 °C.

**Enforcement:**
- `sched_setaffinity` to the logged CPU set, a fixed process count, `maxtasksperchild=4`, `nice 10`, and `OMP`/`MKL`/`OPENBLAS`/`NUMEXPR` threads set to 1.
- **`hard_cpu_quota: false`.** `systemd-run` does not exist in the VS Code flatpak sandbox, so no cgroup quota was set; affinity is the only bound on total CPU.

| Run | Wall s | CPU s | Peak °C | Freezes (s) | Min MemAvailable MB | Swap | Workers' RSS / PSS MB |
|---|---|---|---|---|---|---|---|
| serial | 472.4 | 332.0 | 93.5 | 32 (95.5) | 11,230 | 0 | 78 / 64 |
| 16 #1 | 49.0 | 398.3 | 95.9 | 4 (11.4) | 10,861 | 0 | 1,177 / 699 |
| 16 #2 | 48.9 | 394.4 | 96.3 | 4 (11.4) | 10,769 | 0 | 1,173 / 700 |
| 28 #1 | 44.9 | 523.7 | 94.4 | 3 (8.5) | 10,501 | 0 | 2,043 / 1,197 |
| 28 #2 | 44.4 | 523.6 | 94.0 | 3 (8.0) | 10,339 | 0 | 2,038 / 1,197 |
| interrupt resume | 62.2 | 404.7 | 96.3 | 7 | — | 0 | — |

**Totals:**
- About 3,300 CPU-s across canonical and preliminary runs. The killed segment of the interrupt test was not measured, because SIGKILL leaves no summary.
- Plus tests, audits and rendering.
- No leftover processes after any run.

**Thermal caveat.** This laptop reaches about 90 °C on a single busy core and 97 °C within about 30 s of all-core load.
- **Policy:** warn at 85 °C, start no new query at ≥ 90 °C, freeze workers at ≥ 92 °C, cancel at ≥ 95 °C for 3 consecutive samples (1.5 s).
- **First 16-worker attempt:** cancelled correctly.
- **With freezing:** short spikes still reached 95.9–96.25 °C in three runs. That is above the 95 °C line, but never sustained long enough to trigger the cancel rule, and below AMD's 100 °C Tjmax. Firmware controls were never overridden.
- **Option:** for strictly ≤ 95 °C, lower the freeze threshold (for example to 88 °C). This changes only runtime, never results.

## 9. Data integrity and tests

**Integrity** ([`integrity.json`](integrity.json)): 0 of 205 source files changed, 0 of 1,129 failed-run files changed, git clean. The 205 source files are `selected.json`, README, ledger, PROGRESS, the 4 root files, every archived witness of those roots, and the teacher and dataset code. Every run sets `PYTHONDONTWRITEBYTECODE`, and the code sets `sys.dont_write_bytecode`.

**New tests** (`test_placement_smoke.py`): **9 passed**, run on the final code. They cover:
- failed extensions consume attempts and stop deterministically;
- a budget cut-off is not a certificate;
- identical plans in different processes, and with a clock set past every deadline;
- the pick-blocked fixture (pick access 0, held valid 2, label 1);
- witness phases and attachment, with tampered `held_local` and phase variants rejected;
- the emergency alarm gives `label: null`, re-run on resume;
- partition plus reuse accounting, and idempotent resume;
- prefix lineage, sibling immutability, and occupied versus vacated eligibility;
- fingerprints ignore `volatile` and change with the label.

The fixture `FIXTURE_PICK_BLOCKED_HELD_VALID` (root `2d-plus2d-srv-w0408-s006`, o08 → 55) uses an existing root and object. It is unit-test evidence only, not smoke data.

**Dataset tests** (`dataset-bd/tests/test_bd_dataset.py`): 6 passed, 1 skipped (no 3D repository), 1 failed. The failure is the pre-existing missing `runs/p16` fixture (§2); the test was not edited.

**Audits:** each canonical run passed 1,530 checks with 0 failures, reading only saved files.

## 10. Visuals and artifacts

**Figures** (in [`visuals/`](visuals/), all drawn from the reopened `serial` files with the teacher's own `geometry.bodies`):
- `<root>_placements.png` (4): scene-backed maps per root and context. Green = label 1; red = label 0 after planned pairs; red with crosses = label 0 from screens only (none occurred); grey = no label; hatched = occupied; orange dashes = target goal.
- `871fbd7d53ff6bbc_mode_pairs.png`: the 48 × 48 category matrix of the reference query, showing hard-screened, valid not planned, planned verified, and the reused archived pair.
- `61c28e576c65c064_witness.gif` and its `_filmstrip.png`: a real re-verified witness from the hypothetical-scene query (o00 held from cell 37 → cell 3 after o01 → 4). Full articulated arm, fingers, held disc, carry → release → retreat. Since witnesses are untimed waypoints, *s* is path length, not time.

**Files** ([`manifest.json`](manifest.json): sha256 and size of every other file in this folder (2,460 files), protocol semantics, code hashes, run fingerprints):
- `roster.json`, `comparisons.json`, `integrity.json`, `audit/`;
- one folder per run (`serial/`, `workers16_repeat{1,2}/`, `workers28_repeat{1,2}/`, `interrupt_resume/`), each containing:
  - `parts/`: per-query commits;
  - `records/` and `modes/`: per root;
  - `trajectories/`;
  - `summary.json`, `verification.json`, `fingerprint.json`;
  - `run_*.json`: resource logs;
- `*.cmd` (exact commands) and `*.log`;
- the preliminary runs.

**Commands** (from `gh-yadu-iiith/planar_2d/dataset_relocations_v1/code`, Python 3.13 venv with NumPy 2.4.6):

```
PYTHONDONTWRITEBYTECODE=1 python placement_smoke.py run ../smoke_repro_v2/<run> <1|16|28> <serial|physical16|logical28> <ceiling_s>
PYTHONDONTWRITEBYTECODE=1 python placement_smoke_check.py audit ../smoke_repro_v2/<run>
PYTHONDONTWRITEBYTECODE=1 python placement_smoke_check.py compare ../smoke_repro_v2 serial workers16_repeat1 workers16_repeat2 workers28_repeat1 workers28_repeat2 interrupt_resume
PYTHONDONTWRITEBYTECODE=1 python placement_smoke_check.py integrity ../smoke_repro_v2
PYTHONDONTWRITEBYTECODE=1 python placement_smoke_check.py visuals ../smoke_repro_v2/serial ../smoke_repro_v2/visuals
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider test_placement_smoke.py
```

**Code size.** No new files beyond the three existing scripts. They grew to 549, 337 and 116 lines, above the 120–280 guideline. The growth is the counted search loop, the held-state screen, phase-aware verification, thermal duty-cycling, fingerprints and comparisons.

## 11. Limitations and STOP

- **Zeros are policy-relative.** The 4 zeros (K0 o01) are outcomes of this finite policy (8 of up to 2,304 pairs, fixed per-pair work). Nothing is inferred about the 6,774 valid pairs left untested, or about physical impossibility.
- **The held-state convention is a declared choice.** A held mode is valid if the attached configuration is clear for carrying, under the gripper contact model. Pick access is recorded per mode but excluded from the label, per the contract.
- **Reproducibility scope.** It is shown for one machine and software stack (Python 3.13.11, NumPy 2.4.6). Bitwise equality across other CPUs or libraries is not claimed.
- **Thermal headroom is small** (§8), and no cgroup quota was available.
- **Hypothetical scenes are object-pose edits only;** no robot state was executed for them.
- **No further work done.** No 5,000-root labelling, 3D work, new scenes, sampler or X/Y training, X/Y changes, server use, commit or push. The next stage needs your authorization.

**STOP.**
