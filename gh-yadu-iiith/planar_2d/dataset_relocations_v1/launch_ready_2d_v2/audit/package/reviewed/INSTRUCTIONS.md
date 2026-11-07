# Approach B — Final 2D placement-sidecar mock and production launch-readiness v2

**Version:** V2 candidate, 7 November 2026  
**Repository:** `Narendhiranv04/rss-ebm-infeasibility`  
**Source branch:** `naren/2d-dataset-relocation`  
**Pinned source commit:** `16d1b3b4d565ae89b833141e136c71157823e1dc`  
**Scientific authority:** `final-fixed-methodology-1.pdf`, 5 November 2026  
**Scope:** PREPARE and EXECUTE ONE BOUNDED DEVELOPMENT MOCK; freeze a production policy only after all gates; **do not execute the full 2D dataset extension**.

## 0. Important honesty and use constraints

This package contains **candidate executable patch code**, not a claim that a 25-worker test has already run. It was authored against the available GitHub source at the pinned commit; the GitHub-connected audit covered `launch_plan.py`, `placement_teacher.py`, `placement_smoke.py`, `launch_runner.py`, `launch_guard.py`, `launch_decide.py`, `launch_check.py`, runbook, and pilot reports.

The code here can be syntax-checked without access to your laptop. Its integration and thermal results must be generated **locally**. **Do not print `READY_TO_LAUNCH` merely because source code exists or tests were previously green**. A source anchor mismatch, test failure, unaccounted label, changed science hash, insufficient disk, a safety cancellation, or differing pilot outputs is `NOT_READY_TO_LAUNCH`.

The scripts deliberately have **no full-production execution command**. `prepare_v2.py` creates new sibling `code_v2/` and `launch_ready_2d_v2/` directories; the existing `code/`, old failed smoke, passed smoke, `launch_ready_2d_v1/`, and both original 2D/3D datasets remain untouched. Avoid `git add -A` until auditing all generated outputs. Never push large pilot witness files or caches inadvertently.

## 1. Problems addressed by v2

| Priority | Prior issue | V2 disposition / required verification |
|---|---|---|
| P0 | One 25-worker run exceeded immediate 95°C cutoff | Keep the cutoff unchanged. Turn CPU boost OFF on the host, run two fresh clean 25-worker bounded pilot repeats; monitor startup/tails and ambient interference. No post-hoc relaxation. |
| P0 | Only ~1.75 GiB free at the report snapshot | Require **at least 10 GiB free** before v2 pilot and launch decision, plus forecast disk and an additional reserve. Do not delete original data or prior evidence. |
| P0 | Sampler query generator excludes target goal cell | Remove task-goal exclusion. Destination membership must depend on legal support/occupancy, not whether the placement is useful for target repair. Test goal invariance. |
| P0 | Archived solution metadata selects `prefix2` context | Replace with geometry-designed hypothetical edits. Keep privileged fields confined to analysis/provenance; no solution-oracle selection in primary train/val/test roster. |
| P1 | Both intermediate contexts are one-edit deep | Add a bounded `prefix3` with TWO sequential object-only hypothetical edits; the queried object has not already moved. No physical robot feasibility claim about that prefix. |
| P1 | Historical witness reuse can bypass B-pair selection | **Turn off archived witness reuse in the canonical v2 policy.** Reuse may remain a separate auxiliary baseline, not part of the operational B-pair label. |
| P1 | `witness_store_max=1` drops successful trial paths | Canonical adaptive policy retains every unique successful verified witness among the attempted pairs, at most 32 modes, deduplicated by content. |
| P1 | Fixed 8-pair negatives miss 2/265 positives vs the pilot's 64 reference | Predeclare `ADAPTIVE_FULL`: always evaluate first 8 valid mode pairs; if none succeeds, continue in identical order to 32. Do not confuse 32 with exhaustive physical truth. |
| P1 | New teacher adapter not pinned by source hash | Include `placement_teacher.py` hash in pair-protocol identity; freeze hashes after final edits, not before. |
| P1 | Shell launcher refusal not exercised | Verify the v2 wrapper refuses when called without arguments and before policy freeze. No physical planner call on refusal. |
| P2 | A few modules exceed 280 lines | Separate only if necessary. Auditability, determinism and regression coverage take precedence over arbitrary lines-per-file. |

### Why v2 is faithful to Approach B

The sampler is supervised by **`V_place(S,o,d;Π_place)`**: the object begins in a permitted, valid **already-grasped** configuration. The verified motion is `carry → release → local retreat`. It does **not** require pick access, target restoration, home/`q_ref` connection, or success of any preceding hypothetical edit. Binary `1` means at least one verified witness under the frozen finite trial policy; binary `0` means the **complete deterministic policy** found none. Interrupted and statically ineligible queries are **not committed negative training examples**.

X, Y, ordered causal attribution, Boltzmann energies, the existing 5,000 source roots, robot model and source dataset stay unchanged. A successful placement is **not** a successful corrective sequence. Every hypothetical state preserves original-root identity, split and prefix provenance.

## 2. Files delivered

- `prepare_v2.py`: strict-source, fail-closed patch candidate. It checks anchors and syntax and copies code into a NEW `code_v2/` without touching v1. Its default is no-write preview; `--apply` only creates the v2 candidate.
- `mock_v2_gate.py`: safe metadata freeze, static/source audit, bounded 25-root pilot runner, SIGKILL/resume test, label/witness comparison, and evidence-based readiness decision. **Contains no production-launch command**.
- `test_v2_contract.py`: test of goal independence, protected-field independence, two-edit contexts, adaptive scheduling, real held-state witness replay, source pinning, and guarded production refusal.
- `INSTRUCTIONS.md`: this contract.
- `CLAUDE_KICKOFF.md`: short ready-to-paste prompt for Opus 5.5 High to integrate, review and run v2.

The patcher stops if the pinned code differs. A competent local engineer/Claude instance must review any anchor incompatibility and correct the patch, not bypass the check blindly. `--allow-other-commit` is for a source-reviewed equivalent checkout only.

## 3. Create a clean v2 worktree

Use a NEW worktree to avoid mixing the prior local uncommitted branch with this source-pinned patch:

```bash
cd /home/naren/Desktop/rss-placement-smoke
git fetch origin naren/2d-dataset-relocation
git worktree add ../rss-placement-ready-v2 --detach 16d1b3b4d565ae89b833141e136c71157823e1dc
R=/home/naren/Desktop/rss-placement-ready-v2
PY=/home/naren/Desktop/rss3d_env/venv/bin/python
PACK=/absolute/path/to/approach_b_final_mock_v2
```

Do not copy files into the original dataset folder. Confirm `git status --short` is initially empty. Ensure Python 3.13, NumPy, psutil, pytest, and the existing 3R teacher imports are available in the selected environment.

## 4. Safety preflight BEFORE the mock

Close unrelated heavy CPU workloads, including IDE background indexing where practical. Check the hardware from a HOST terminal (not necessarily a Flatpak/VS Code sandbox):

```bash
lscpu | grep -E 'CPU\(s\)|Core\(s\)|Thread\(s\)|Model name'
free -h
df -h /home
cat /sys/devices/system/cpu/cpufreq/boost
```

Only if the CPU boost file exists and the host allows this reversible change:

```bash
# Host terminal, explicit user/admin action; do NOT put sudo in a worker script.
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
cat /sys/devices/system/cpu/cpufreq/boost   # must say 0
```

The original v1 pilot showed a clean 14.4-minute full-load soak at about 86°C, **but** an immediate 95°C safety cancel happened during startup/tail/external contention. Disabling boost is an intervention requiring a **new observed pilot**, not a guarantee. Keep pause at 88°C, resume at 80°C, cancel at 95°C, sensor-loss guards and resource safeguards unchanged. Do not rely on 25 affinity-bound workers as a hard cgroup quota. Ensure **≥10 GiB free disk** and AC power before running the bounded pilot.

Restore boost manually after the mock or later work as desired:

```bash
echo 1 | sudo tee /sys/devices/system/cpu/cpufreq/boost
```

**Do not restore boost before v2 readiness decision**, which verifies the observed host policy. Revalidate if your production power configuration will differ.

## 5. Prepare the candidate, metadata only

```bash
$PY "$PACK/prepare_v2.py" "$R"                 # preview: no writes
$PY "$PACK/prepare_v2.py" "$R" --apply         # copies into new v2 directories only
$PY "$PACK/mock_v2_gate.py" "$R" init          # freeze NEW metadata-only splits/query plan
$PY "$PACK/mock_v2_gate.py" "$R" static        # tests no target leakage / two-edit contexts / hashes
```

`init` asserts that **all 5,000 root train/validation/test assignments are identical to v1**. `static` checks the frozen plan, a subset of roots with altered target goals and intentionally scrambled protected repair metadata, one- and two-edit context legality, root/split source hashes, 32-logical-CPU topology and the original preflight. It does **not** call the motion planner.

The v2 plan will contain a **different number of queries** from v1 because it adds `prefix3` and allows the previously excluded target goal cell when unoccupied. Recompute its Q1/Q2/Q3/Q4 totals and checksums. **Do not reuse the v1 308,652-query forecast or the 1 GB disk estimate as if nothing changed.**

Note that both source and test may need a narrow code fix if they expose an unanticipated but legitimate scene shape. Document every such fix, rerun metadata freeze in a NEW v2 candidate if hashes change, and never silently overwrite a frozen plan.

## 6. Execute ONE bounded v2 mock on the development roots

Only after the static tests pass, boost is observed off, and disk has headroom:

```bash
$PY "$PACK/mock_v2_gate.py" "$R" mock --execute-mock
```

This runs project tests first, then **only** the 25 existing training pilot roots, with:

1. `v2_ref64_25`: 64-pair FULL, 25 workers — comparison reference, not physical truth.
2. `v2_adaptive25_a`: adaptive 8→32 FULL, 25 workers.
3. `v2_adaptive25_b`: independent repeat of adaptive 8→32 FULL, 25 workers.
4. `v2_adaptive_serial`: first query per pilot root, serial execution.

Every run is audited from saved files. All discovered verified trajectories are retained within the 32-pair adaptive budget; no archived source can bypass the first eight modes. A worker health/thermal stop causes `NOT_READY_TO_LAUNCH`; **do not relax or reinterpret any threshold** to force a pass.

If the mock fails before complete, preserve its output. Repair the cause, then resume only with matching policy, code and manifest IDs. If the teacher source changes, **start a new versioned v2 candidate**, because deterministic fingerprints must refer to one frozen teacher.

### Kill/resume mock (separate bounded pilot)

```bash
$PY "$PACK/mock_v2_gate.py" "$R" interrupt --execute-mock
```

The script starts an adaptive 25-worker pilot on the same 25 training roots, waits for five committed roots, SIGKILLs the coordinator, verifies no child-worker survivors, resumes, reaudits every stored trajectory and checks committed file hashes remain unchanged. This is NOT a full collection run.

## 7. Evaluate all gates and freeze readiness ONLY if proven

```bash
$PY "$PACK/mock_v2_gate.py" "$R" decide
```

`decide` checks all of the following, failing closed:

- Original root-group split identity, goal-independent candidate generation, no use of privileged repair metadata to select contexts.
- Correct 48×48 held-mode geometry; real carry/release/retreat trajectories with valid attachment and phase-aware certification.
- All 25-root pilot records accounted for; no interrupted eligible query labeled 0, no unexplained statically excluded query.
- Two clean 25-worker adaptive runs with **zero safety stops** and identical per-query scientific fingerprints.
- Serial first-per-root equals 25-worker results; independent adaptive run matches exact nested 8→32 trials from new 64 reference.
- SIGKILL/resume reproduces canonical records and packed witnesses; no orphaned files, modified original dataset or workers left alive.
- At least 10 negative labels across ≥5 lineage groups and ≥3 classes (minimum diversity; *not* a guarantee of calibration).
- `n_verified_pairs` equals stored distinct witness references modulo deduplication; the audit independently certifies each stored trajectory.
- Updated Q1 disk and CPU forecasts from **v2** pilot data with conservative 1.5× heterogeneity factor and additional disk/monitor overhead; max 7 days estimated and disk ≥ max(10 GiB, forecast + reserve).
- Host boost is OFF during the readiness decision, no post-hoc thermal threshold relaxation, and source/config hashes match the frozen policy.

If all pass, `decide` writes `MOCK_V2_DECISION.json` with `READY_TO_LAUNCH` and freezes `configs/production_policy.json`. This **does not** start the full run. The later `launch_full_2d.sh` still requires explicit `--execute-full --confirm <current token>` and a new passing preflight. Do NOT supply that token during this task.

If any fail, `MOCK_V2_DECISION.json` must record `NOT_READY_TO_LAUNCH`; no production policy should be frozen.

### Test launcher refusal without starting it

```bash
L="$R/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2"
set +e
"$L/launch_full_2d.sh"        # MUST refuse, exit nonzero, no planner call
code=$?
set -e
test "$code" -ne 0
```

When `READY_TO_LAUNCH`, do not confuse **software readiness** with proven long-run thermal stability across days, calibrated feasibility confidence, or validated end-to-end X/Y repair performance. Those remain separate downstream evaluations.

## 8. Visual and scientific audits Claude must additionally provide

Generate actual scene-backed 2D figures/GIFs from the v2 results, not illustrations: one positive initial placement, one negative with declared budget reasons, one target-goal-cell placement if eligible, one two-edit hypothetical scene with its prefix and lineage, and one formerly ungraspable-but-placable object. Plot a sample 48×48 mode mask and rank-wise successes. Report destination near/mid/far/vacated distributions and negatives across K classes. Audit geometric-ineligible cells separately; don't treat static occupancy as observed `0` labels.

**Recommended evidence figures:** 25-worker Tctl timeseries with pause/resume and startup/tail annotations; 64 vs adaptive positive/negative disagreement; CPU-s distribution; packed witness bytes/query; and a split/lineage summary. The model's confidence is **not trained now**; the fraction of successful tried grasp pairs is not a calibrated probability.

The ultimate scientific acceptance question is whether the sampler will be trained on sufficiently diverse, accurately defined individual-placement data. If adaptive 8→32 still misses too many 64-pair positives on the changed roster, or if all negatives collapse into one family, the pilot fails and the labeling protocol must be revised **before** production.

## 9. Deliverable report expected from the local executor

Write `launch_ready_2d_v2/FINAL_MOCK_V2_REPORT.md` with:

1. Exact source Git SHA and all v2 code/config/robot/teacher/plan/split hashes.
2. Patch diff and tests, explicitly identifying fixes and any deviations from this candidate patch.
3. New split counts, root-group closures and Q1–Q4 query totals, with zero val/test teacher queries during pilot.
4. Label-miss statistics (8 vs adaptive vs 32 vs 64), per-stratum breakdown, certified vs operational negatives, and calibration caveats.
5. Saved-witness counts, dedup accounting, replay evidence, actual GIFs and 48×48 mode matrices.
6. 25-worker CPU affinity/topology, peak Tctl, all safety pauses/stops, wall/CPU times, disk forecasts and available free space.
7. Reproducibility across repeated 25-worker runs, serial subset, and SIGKILL/resume.
8. Negative candidate/protected-metadata independence results; whether T's goal cell is now included when physically eligible.
9. Precise `READY_TO_LAUNCH`/`NOT_READY_TO_LAUNCH`, remaining risks, and **confirmation that no full dataset run began**.

### Hard restrictions

**Never** generate new independent scenes; mutate original 2D/3D corpora; train the placement cVAE, flow-matching model or X/Y energies; launch full Q1/Q2/Q3/Q4; use remote servers; relax the 95°C safety rule; silently remove old logs; or push/merge without explicit user approval.

The goal is **readiness**, not a favorable verdict. The correct result is `NOT_READY_TO_LAUNCH` if even one essential gate fails.
