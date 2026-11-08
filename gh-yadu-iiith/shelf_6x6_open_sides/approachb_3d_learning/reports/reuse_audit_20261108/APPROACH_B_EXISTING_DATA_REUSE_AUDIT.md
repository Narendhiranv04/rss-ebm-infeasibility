# Approach B: Existing 2D/3D Data Reuse Audit for Finalized X/Y Supervision

**Date:** 2026-11-08 · **Mode:** read-only evidence audit. No physics, planners, MuJoCo, GPU, training or data writes were run.
**Scientific authority:** `final-fixed-methodology-1.pdf` (5 Nov 2026, sha256 `b7a39e3c07db0464…`), §2.2, §7.5, §9.1–9.3, §11.1–11.5, §14, §16.4.
**Operating contract:** `instruction-phase-md/BLACKWELL_APPROACH_B_EXECUTION_CONTRACT_v1.md` §3.3, §6 (BWL-3 categories), §12.
**Evidence scripts and raw outputs:** [`approach_b_reuse_audit/`](approach_b_reuse_audit/) (`SHA256SUMS` inside).

---

## 0. Answer to the key question

> **How much authentic finalized X/Y supervision can we recover from the already-executed 2D/3D relocation experiments without generating everything again?**

| Domain | Already finalized-valid X/Y (use now) | Recoverable with limited verification (replay / connector / re-query, no new scene or sequence design) | Needs full re-execution | Missing entirely |
|---|---|---|---|---|
| **3D** (SHELF66V3, 1,582-root cohort) | **X: 0.** **Y: 0 replay-verified.** 4,367 Y=0 operational negatives are reusable as-is. | **X:** 6,773 unique ordered prefixes (3,665 complete stored sequences + 3,108 proper prefixes), connector + replay pending. 1,141 X=0 relocation failures become usable once the relocation policy is pinned. **Y:** 2,645 stored positive target witnesses need replay (511 already executed once inside the teacher). 2,929 unique X-candidate endpoints need a Y record. | 9 attempted but unstored sequences. Everything, if a different 3D X relocation policy is pinned. | Candidate-diverse (sampler-sourced) destinations; order-discordant AB/BA inside the cohort (35 potential); omission-of-first; multi-destination per object; K4/NONE. |
| **2D**, authentic X/Y sidecars (BXY teacher, 635 roots) | **X: 7,701** (3,952 X=1, all connector-witnessed and independently replayed; 3,749 X=0). **Y: 6,608** (2,555 Y=1, all replayed; 4,053 Y=0). Of these, **4,443 X / 3,831 Y on 398 development roots** may be used for training. P24 (3,258 X / 2,777 Y, 237 roots) stays held-out. | n/a | n/a | Candidate-diverse sequences. No shared roots with the Q2 sampler corpus. Wider destination coverage. |
| **2D**, 5,000-root class-quota corpus (legacy BD2D labels) | 28,730 Y=0 operational negatives (same `t_goal` teacher function). 6,326 X=0 `CERTIFIED` relocation failures (same relocation functions). | 6,555 Y positives: regenerate the witness, then replay. 19,026 unique prefixes with full step records: connector + replay. 3,411 X=0 `NO_PLAN_FOUND` failures: agreement sample. | 3,655 relocation-complete prefixes that have **no** stored records. | Same as above. |
| **2D Q2 placements** (238,160 planned trials) | **Never X/Y.** These are sampler supervision only. | n/a | n/a | n/a |

**Bottom line:**
- **3D:** no authentic finalized X exists today, and no 3D Y positive has been independently replayed. However, nearly all physical execution is already done and stored, with hash-verified trajectories. Converting it to authentic X/Y needs replays plus one connector per prefix, not new relocation planning. That is about 6.8k connector tests and 2.6k Y replays (estimated ≤ 36 h single-core, ≈ 3 h on 12 workers; see §8).
- **2D:** a sizeable authentic X/Y pool already exists: 4,443 X / 3,831 Y on development roots, with all four (X,Y) cells and 188 order-discordant same-endpoint groups. It is **construction-informed** (§3.1 of the Nov 5 PDF), however, and it sits on root corpora **disjoint** from the 5,000-root Q2 sampler corpus.
- **Shared gap (both domains):** candidate-diverse, sampler-sourced X/Y supervision on the same roots as the placement sidecars (§11.5). This exists nowhere and must be collected.

---

## 1. Definitions applied (Nov 5 §9; contract §3.3)

- **X_r = 1** requires all of the following under one pinned, finite-budget teacher:
  - every relocation of `r` is executed in order from `(S0, q0)`;
  - the achieved robot state is carried between moves, with no reset or home visit;
  - **then one terminal free-gripper connector to `q_ref`** succeeds in the edited scene.
- **X_r = 0:** an attempted query failed (operational, with `negative_kind` and phase).
- **Not attempted:** missing, never 0.
- **Y_s = 1:** the original target task is fully verified from the **fixed `q_ref`**, gripper free and nothing attached, in the ideal endpoint scene `s`, independently of X. Y is shared by all orders that reach `s`.
- **Not X:** a single placement, a teleported endpoint check, or a composite sequence whose target is appended from the achieved state.
- **Not Y:** a target planned from an achieved post-relocation state.

**Classification vocabulary used below:**
- `DIRECT` = satisfies the definition now.
- `LIMITED` = needs only replay, connector, re-query or an agreement sample, with no new sequence design.
- `REEXEC` = trace or state missing, so the sequence must be run again.
- `UNRESOLVED` = no split assignment or provenance decision yet.

**Y category mapping:** the BWL-3 categories (`Y_REPLAY_VERIFIED`, `Y_PLANNING_ONLY`, `Y_POLICY_FAILED`, `Y_UNATTEMPTED_OR_UNRESOLVED`, `Y_SEMANTICS_INCOMPATIBLE`) are used for 3D Y. One category was added: `Y_EXECUTED_IN_TEACHER`, meaning executed once by the teacher with checks but not independently replayed.

---

## 2. What was inspected (and what was not touched)

| Item | Path (local, read-only) | Identity |
|---|---|---|
| 3D generator source | `~/Desktop/rss-3d-relocations/…/shelf_6x6_open_sides/dataset_bd_classquota_without_home_position_no_overpass/code/shelf66_v3/bd_3dt.py` | sha256 `62dc853c…` (= source pin) |
| 3D scene/executor | `…/shelf_6x6_open_sides/scene/shelf_scene.py` | `a8453e24…`; **all 37 pinned source files match** `source_pin_first_cohort.json` |
| 3D store | `…/raw/roots/*.json` (2,072), `…/raw/witnesses/**/*.npz` (18,501) | data commit `32c3194fc` (live cut 2026-11-07T05:28:58Z); **1,582/1,582 cohort root sha256 = pin** |
| 3D cohort / split | `…/dataset_placement_extension_v2/production_v3/QC_ELIGIBLE_ROOTS.json`, `frozen/split_manifest_first_cohort.json`, `frozen/source_pin_first_cohort.json` | `52137ff1…`, `f9d37781…` (internal `c0f8fb5a…`), `422d3c1a…`: identical to the v4 snapshot `v4full_82926152fa9cc345` binding |
| 3D native source-witness replay | `…/dataset_placement_extension_v1/mock_v1/validation/source_baseline.json`; `…/v2/PUSH_AUDIT.md` | `dfa15d67…`; 24/24 + 6/6 |
| Prior 3D learning reports | `~/Desktop/rss-3d-learning/…/approachb_3d_learning/reports/02_*, 05_*` | read only (this is the BWL session's worktree: no git commands that take locks, no writes) |
| 2D X/Y teacher | `.claude/worktrees/approach-b-xy-v1/bxy_teacher.py` | `c18ac91c…`, worktree HEAD `45b0fb05f` |
| 2D X/Y sidecars | `…/runs/approach_b/bxy_{teacher_pilot,pmin,p15,p16,p20,p24}_v1/results/*/*.{x,y}.json` | P-min sidecar `0191f49b…`, design `8f164845…`; P24 design `25c94230…`, seal `4530a8d6…` |
| 2D 5,000-root corpus | `~/Desktop/rss-placement-smoke/gh-yadu-iiith/planar_2d/dataset_bd_classquota/` (idle worktree) | data commit `6199c3074`; `selected.json` `edb8dfca…`; `bd_2d.py` `eb97136c…`; `p16_common.py` `dec0b7ed…` |
| 2D Q2 split / plan summary | `~/Desktop/rss-2d-q2-commit/…/launch_ready_2d_v2/{split_manifest,plan_summary}.json`, `amendment_q2/Q2_AMENDMENT_DECISION.json` | commit `628795d`; split `e253b623…` (its `source_selected_sha256` = corpus `selected.json`) |

**Not touched:**
- the Blackwell server, its worktree on the server, GPU, checkpoints and processes (no SSH to any host);
- the live 2D Q2 run directory `rss-placement-ready-v2/…` and gvlab;
- the Q2 query plan (`query_plan.jsonl.gz` was deliberately **not** loaded while the laptop run is live).

All scans ran single-process at `nice 19 / ionice idle`, with a peak RSS of 58 MB. The laptop Q2 coordinator (pid 594063) was running throughout.

---

## 3. Task 1: the original 3D relocation dataset

### 3.1 What the teacher actually does (source evidence)

`bd_3dt.teach` (bd_3dt.py:239–358), with `shelf_scene.py`:

- **`settle(e, cells)`** (bd_3dt.py:145): `e.reset()` (shelf_scene.py:799) restores the reference state:
  - `mj_resetData`, all objects to their home cells;
  - the arm to `q_ready` (IK of the ready pose), gripper `GRIPPER_OPEN`, `held=None`, 500 settle steps;
  - then teleport of the edited objects (100 steps each).
- **Y-type queries:**
  - `q_t(cells)` (l.245) = `settle`, then **`query(e,"t0",goal)`**.
  - `query` (l.96) tries the 32 grasp poses (16 azimuths × {mid, top}) in preference order with `plan_pick_place` (RRT-Connect `max_iters=5000`, no wall-clock limit, `seed=0`, up to 4 IK branches); the first feasible one wins.
  - **Nothing is executed.** On success the planned joint path is stored as a witness (`witness()`, l.138).
  - `labels.original` = `q_t(S0)`. `labels.D` = `edit(sub)` for each single movable object plus the intended union, every edit at the object's **first parking cell**, `query_kind=SCENE_TARGET_ONLY`.
- **Ordered sequences:** `seq(steps)` (l.265) does `settle(S0)` once, then for each relocation and then for `t0`:
  1. `query` from **`e.q_now()`, the achieved state**;
  2. store the planned witness;
  3. **`e.run_plan(p)` executes it in MuJoCo**;
  4. verify that the mover ends in its cell, `|z_err| ≤ 1 cm`, and no other object changed cell.
  - There is **no reset between moves**, and the target is **appended from the achieved state**.
  - **No connector to `q_ready` exists anywhere:** `RETURN_TO_READY=False` (shelf_scene.py:155), and `_plan_retreat` (l.1175) ends each move at the retreat pose. `go_ready()` (l.1286) is never called by the teacher.
- **Stored B roles:**
  - `minimal`: the BFS-found repair, at parking cells;
  - `redundant+1`: one distractor first, then the minimal repair;
  - `noop_irrelevant`: the distractor alone;
  - `wrong_destination`: the first mover sent to `wrong_dest`, the rest to parking;
  - `wrong_order`: the minimal repair reversed;
  - `intended_union` (non-cohort only).
  - At most 4 plans are stored per root, plus the minimal one.
  - **Failed BFS sequence tries are not stored** (l.297–302).
- The BFS also computes the **full T-only lattice** over the ≤6-object pool for every non-K0 root (l.304–307). Only singletons and the intended union are stored in `D`. The rest survive only as `blocker.minimal_sets` / `clearing_sets_reachable`.

### 3.2 Data checks performed

- **Fixed reference configuration.** All 3,726 stored Y-positive witnesses start at the same joint vector, `q_ref = (-1.345029, -0.632166, 0.221281, 2.024222, 0.267756, 2.629261, 2.210774)` rad, with a max deviation of 7.0e-10 rad. Every B step-0 witness (4,990) starts at the same `q_ref`. **In 3D, q0 = q_ref.**
- **Carried state.** Over 7,944 consecutive-step boundaries, step j+1's first waypoint differs from step j's planned last waypoint by 0.69e-3 rad median (max 1.27e-3) and **never exactly 0**. So each plan started from the **achieved** (tracking-lagged) MuJoCo state, and that achieved arm state is recorded as the first row of the next witness.
- **Witness integrity.** All 15,447 referenced witness files exist and match their content hash (`bd_core.Store.witness` rule). 3,054 further NPZ files on disk are unreferenced. These are most likely unstored lattice-target plans and failed-BFS steps; they cannot be mapped to queries without recomputation.
- **Deterministic replay evidence (prior, not mine).** `replay3d.py source` re-executed 24 + 6 stored B **step-0** relocation witnesses with the unchanged `run_plan` from the settled root scene. 30/30 matched the recorded outcome, with |z_err| difference ≤ 5.6e-7 m. **Multi-step and target-witness replay has never been demonstrated.**

### 3.3 Answers (QC cohort: 1,582 roots; train 1,120 / val 246 / test 216)

| # | Question | Train | Val | Test | **Total** | Evidence status |
|---|---|---|---|---|---|---|
| 1 | Ordered relocation sequences attempted (m ≥ 1) | 3,417 | 719 | 679 | **4,815** | 4,806 stored + **9 attempted but not stored** (failed BFS tries: `sequence_tries` minus stored minimal) |
| 1a | Of the stored ones, unique within root | 3,409 | 718 | 679 | **4,806** | no duplicate ordered sequence within a root |
| 1b | Empty-sequence target executions (K0 roots; not relocations) | 358 | 86 | 67 | 511 | see §4 |
| 2 | Completed **all** corrective relocations | 2,592 | 556 | 517 | **3,665** | directly verified (MuJoCo execution + end-cell/z/displacement checks) |
| 2a | Failed at a relocation | 817 | 162 | 162 | **1,141** | `CERTIFIED` 782 · `NO_PLAN_FOUND` 317 · `EXECUTION_NOT_VERIFIED` 40 · `GRASP_FAILED` 2 |
| 3 | Saved, hash-verified physical witnesses for **every executed relocation step** | 3,409 | 718 | 679 | **4,806** | stored and replayable. **Independently replay-verified: 0 full sequences** (only 30 single step-0 samples) |
| 4 | Robot state **carried** across consecutive moves during execution | | | | **4,806** | verified from source + the 7,944 boundary gaps |
| 4a | Robot state **recorded** across moves | | | | **0 complete** | Arm joints are recoverable at every boundary that is followed by a stored witness. **Gripper opening, attachment and exact object poses after each move are not recorded** (only `end_cell`, `z_err`, `pushed`). |
| 5 | Terminal (post-last-relocation) arm configuration recorded | 1,975 | 436 | 399 | **2,810** of 3,665 | achieved q = first row of the target-step witness. For the other **855**, only the *planned* end of the last relocation is stored (the target plan failed, so there is no target witness); the achieved state is recoverable only by replay. |
| 6 | Already satisfy the finalized X, incl. connector to `q_ref` | 0 | 0 | 0 | **0** | no connector was ever planned or executed |
| 7 | Could become valid X by adding **only** the terminal connector (+ the replay needed to rebuild the achieved state) | 4,839 | 992 | 942 | **6,773 unique ordered prefixes** | = 3,665 complete stored sequences + 3,108 additional proper prefixes whose relocations all executed. All have complete hash-valid relocation witnesses. **Potential, not verified.** |
| 7a | Directly X=0 (a relocation failed), subject to pinning the relocation policy | 817 | 162 | 162 | **1,141** | no connector needed; the stored composite label is not reused |
| 8 | Require full re-execution | 8 | 1 | 0 | **9** | unstored BFS tries. **Conditional:** all 4,806 if the pinned 3D X teacher's relocation policy differs from `BD3DS66_TEACHER_V2`. |

**Important conditions on rows 7 and 7a:**

1. **Teacher identity.** Reuse is valid only if the 3D X teacher is pinned as: "`BD3DS66_TEACHER_V2` relocation executor (`bd_3dt.seq` loop: first-feasible grasp, RRT 5000, seed 0, `run_plan`, the same verification), plus a new terminal connector, plus independent replay". This mirrors the 2D `BXY_X_TEACHER_V1` (§5.1). If a different relocation policy is adopted, the historical relocations are not its X phases.
2. **State reconstruction.** A connector test from the stored joint vector against ideal cell-centred objects would *not* be authentic: object poses after real placement differ by mm and are not stored, and gripper width is not stored. The connector must start from the state rebuilt by **deterministic replay** of the stored relocation witnesses. This is demonstrated so far only for single step-0 moves.
3. **Gripper and attachment convention.** After `release`, the fingers sit at the last grasp's pre-shape width, not `GRIPPER_OPEN`. The connector specification must pin "free gripper" (open, nothing held) at `q_ref`, matching the Y reference state written by `reset()`.
4. **Do not use the stored composite B outcome** (`y`, the target from the achieved state): 2,810 successes and 1,996 failures in the cohort. It is neither X nor Y (Nov 5 §11.1; contract §6).

### 3.4 Y availability for the 6,773 X-candidate prefixes (avoids double counting shared endpoints)

| Endpoint Y source | Prefix count | **Unique endpoints** | Meaning |
|---|---|---|---|
| Stored record (`original`/`D`) | 3,520 (Y=1: 1,545; Y=0: 1,975) | 3,165 | a record exists; positives still planning-only (§4) |
| Lattice summary (`blocker`) | 1,707 (Y=1: 701; Y=0: 1,006) | 1,383 | The outcome is implied by `minimal_sets`/`clearing_sets` of a full lattice. No record and no witness pointer. Treat as `LIMITED` (re-query). |
| Unknown: parking-cell superset of a minimal set | 262 | 1,546 (both unknown rows combined) | not inferable (non-monotone possible) |
| Unknown: non-parking destination (`wrong_destination`) | 1,284 | (included above) | never queried |

**Full stored sequences only, as a (X status × endpoint Y) cross-tab** (`xtab3d.json`):

| | Y=1 (stored) | Y=1 (lattice) | Y=0 (stored) | Y unknown |
|---|---|---|---|---|
| X=0 (relocation failed) | 249 | 18 | 244 | 630 |
| X pending connector | 1,538 | 666 | 550 | 911 |

So X=0/Y=1 cases (267) already exist on the negative side. X=1/Y=0 and X=1/Y=1 are latent in the pending column, subject to the connector outcome.

### 3.5 Order contrasts (same endpoint, different order)

- **Cohort:**
  - **714** same-endpoint groups (`minimal` vs `wrong_order`), all with a shared Y=1 endpoint (355 stored, 324 lattice, and the 35 below).
  - **Only 35** groups (25 / 3 / 7 by split) differ in relocation completion: one order is X=0, the other is connector-pending.
  - The remaining 679 have both orders completing their relocations, so any X difference could only come from the connector.
  - Decided order contrasts today: **0**. The potential is ≤ 35, plus any connector-induced discordance.
- **Why so few:** the construction admits only `independent_moves` roots, so the cohort is order-independent by design. Roots whose minimal repair is order-dependent were labelled `DEPENDENT` and **excluded**.
- **Non-cohort roots (490, no split):**
  - The **75 `DEPENDENT` roots carry 72 of their 75 same-endpoint groups as order-discordant.** This is the order-contrast family the cohort lacks.
  - Lineage: 41 link (by `parent_family_id` or identical layout) to train roots, 10 to val, 5 to test, and 19 to none.
- **Omission:** `redundant+1` (825 in the cohort) vs `minimal` gives "omit the leading distractor" pairs, and prefixes give "omit the last move" pairs. **"B without A" (omit the first necessary mover) is never executed as X.**

### 3.6 Non-cohort 3D roots (unresolved)

- **490 roots outside the frozen split:**
  - over-quota K0 / K1 / K2 / K3: 169 / 95 / 76 / 5;
  - `DEPENDENT` 75, `NONE_SOLVABLE` 40, `INCONSISTENT` 25, `NONE` 4, `NONE_UNAUDITED` 1.
- **Their data:** 1,453 B sequences (1,042 complete, 411 relocation-failed), 1,806 connector-pending prefixes, 490 original + 1,940 D Y records.
- **Lineage:** 289 are lineage-linked to an existing split and 201 are unlinked.
- **Status: `UNRESOLVED`.** They cannot be used without an explicit split-extension decision. The obvious rule is inherit-by-lineage for linked roots and a new seeded assignment for unlinked ones. That decision is not made here.

---

## 4. Task 2: existing 3D Y evidence (the 7,012 target-only attempts)

### 4.1 Counts (cohort; one record per unique endpoint; no within-root duplicate endpoints)

| Family | Train | Val | Test | **Total** |
|---|---|---|---|---|
| `original` attempts (root endpoint) | 1,120 | 246 | 216 | **1,582** |
| `D` attempts (edited endpoint) | 3,849 | 801 | 780 | **5,430** |
| **All target-only attempts** | 4,969 | 1,047 | 996 | **7,012** |
| Planning positives | 1,851 | 408 | 386 | **2,645** (original 511 + D 2,134) |
| Planning negatives | 3,118 | 639 | 610 | **4,367** (`CERTIFIED` 4,004; `NO_PLAN_FOUND` 363) |
| Saved target witnesses (positives) | 1,851 | 408 | 386 | **2,645**, all present and hash-valid |
| Independently replay-verified positives | 0 | 0 | 0 | **0** |
| Lattice target attempts made but **not stored** as records | 6,829 | 1,214 | 1,360 | **9,403** |

### 4.2 BWL-3 categories

| Category | Count | Basis |
|---|---|---|
| `Y_REPLAY_VERIFIED` | **0** | no independent replay of any 3D target witness exists |
| `Y_EXECUTED_IN_TEACHER` (added) | **511** | K0 roots: `seq([])` re-planned the same target from `q_ref` (identical witness id in **511/511**) and **executed it in MuJoCo** with the end-cell/z/displacement checks. This is the strongest existing 3D Y=1, but it is not independent. |
| `Y_PLANNING_ONLY` (positives) | **2,134** | `D` positives: planned from `q_ref` in the teleported endpoint, never executed |
| `Y_POLICY_FAILED` | **4,367** | `CERTIFIED` 4,004: all 32 grasp poses are rejected by deterministic geometric checks of the declared protocol (not universal impossibility). `NO_PLAN_FOUND` 363: an operational budget or planner failure. Planning failure is a valid operational Y=0 under a plan-then-execute Y teacher. |
| `Y_UNATTEMPTED_OR_UNRESOLVED` | **0** within the 7,012 | Separately, 9,403 lattice attempts are unrecorded; their outcomes are partly recoverable via `blocker` (§3.4). |
| `Y_SEMANTICS_INCOMPATIBLE` | **0** found | Reasons, in order: (1) target identity: always `t0` to the root `goal`; (2) edit: always first parking cells, every `D` is `SCENE_TARGET_ONLY`; (3) `q_ref`: identical across all 3,726 witnesses; (4) reset: gripper open, nothing held; (5) endpoint: teleported ideal arrangement, which matches §2.2. |

**Checked semantics:**
- **Original target identity and goal:** `t0` to `rec["goal"]`. This is unchanged across `original`, `D` and B's target step.
- **Hypothetical endpoint:** object-to-cell arrangement (`ekey` in `audit3d_scan.py`).
- **Fixed reference robot configuration:** verified numerically, see §3.2.
- **Target-planner budget:** 32 grasp poses in fixed preference order, RRT-Connect `max_iters=5000`, wall clock disabled, `seed=0`, IK branches ≤ 4. Recorded per root as `budget: {rrt_max_iters: 5000, seq_tries: 8, pool_cap: 6}`.
- **Reset semantics:** `settle` → `reset()`, then teleports. This differs from post-execution scenes only by settling offsets of the executed objects. Y is defined on the ideal endpoint, so this is the correct convention.

### 4.3 Can the saved target witnesses be replayed without recomputing the dataset?

**Yes.**
- Every positive stores the full planned `q` path, the `segment` and `phases` arrays, `carry_xy`, and the `grasp_id`/`azimuth` in its record.
- **Replay procedure:**
  1. rebuild the root env and `settle` to the endpoint;
  2. re-expand the segments, recovering grip widths from `grasp_poses(...)[grasp_id]["open_width"]`;
  3. call `run_plan`;
  4. apply the teacher's checks.
- This is exactly the procedure `replay3d.py source` already used for B step-0 witnesses (30/30 agreement).
- No IK or planning is needed.
- **Cost:** see §8.
- **Recommendation:** run a small predeclared replay sample first (contract §6), then the full 2,645.

---

## 5. Task 3: 2D data

### 5.1 Authentic 2D X/Y sidecars (finalized-compatible)

- **The teacher matches the finalized definition.**
  - `BXY_X_TEACHER_V1` (bxy_teacher.py:1–14, `x_query` l.196): relocations from `(S0, q0)` via `DetPhysics` + `best_edge`, state carried, **one connector** `DetPhysics._transit` to `q_ref` with every body present, gripper open, nothing attached, plus **independent replay**.
  - `BXY_Y_TEACHER_V1`: `bd_2d.t_goal` from the same `q_ref` in the edited scene, independent of X, plus replay.
  - `q0 = q_ref` (`qref()` l.59). Censored, ineligible and unattempted queries are never 0.
- **Verified in data:**
  - **every X=1 row** (3,952) has a connector witness, `independent_replay=True` and `q0 == q_ref`;
  - **every Y=1 row** (2,555) has a target witness and a replay;
  - **0 censored or unattempted** rows across all 10 collections.

| Collection (roots) | Historical role / split | X (X=1 / X=0) | Y (Y=1 / Y=0) | (X,Y) cells 11 / 10 / 01 / 00 | Same-endpoint multi-order groups (X differs) |
|---|---|---|---|---|---|
| P9 pilot (20; ⊂ P-min train) | train | 57 (49 / 8) | 59 (29 / 30) | 27 / 22 / 8 / 0 | 16 (2) |
| P-min (197: train 176 / val 21) | train/val (mock2d_5000 lineage; its 34 test roots excluded as P7-exposed) | 2,165 (1,683 / 482) | 1,807 (1,492 / 315) | 1,425 / 258 / 407 / 75 | 535 (111) |
| P15 pilot (28) | dev | 268 (202 / 66) | 246 (202 / 44) | 173 / 29 / 53 / 13 | 50 (6) |
| P16 wA/wB/wC (87) | dev (P17 cross-fit) | 780 (378 / 402) | 718 (162 / 556) | 122 / 256 / 74 / 328 | 136 (23) |
| P20 k1/k2/k3 (86) | dev (P21) | 1,173 (427 / 746) | 1,001 (191 / 810) | 141 / 283 / 136 / 610 (+3 Y absent) | 145 (46) |
| **Development subtotal (398 roots)** | | **4,443** | **3,831** | 1,888 / 848 / 678 / 1,026 | **882 (188)** |
| P24 confirm (237) | **sealed confirmation, now exposed: test-only** | 3,258 (1,213 / 2,045) | 2,777 (479 / 2,298) (+1 row without key) | 360 / 847 / 316 / 1,728 (+7) | 408 (111) |
| **All (635 unique roots)** | | **7,701 (3,952 / 3,749)** | **6,608 (2,555 / 4,053)** | | **1,290 (299)** |

**Failure decomposition:**
- **X=0:** relocation `CERTIFIED` 707, relocation `NO_PLAN_FOUND` 2,907, **connector failure 135**. The connector-only failures are exactly the Nov 5 §9.1 "relocations succeed, stop fails" case.
- **Y=0:** `CERTIFIED` 2,393, `NO_PLAN_FOUND` 1,660.

**Candidate source:** every sequence comes from a **construction-informed catalogue**.
- The class tags are `repair`, `repair_reversed`, `omission_0`, `prefix_1`, `neutral_edit`, `harmful_edit`, `interference_*`, `repair_permutation`, `qref_corridor_destination`, and so on.
- The generators are the P16 `BXY_GEN_V21` and P20 `bxy_p20_k34.candidates` families (Nov 5 §3.1).
- **0 sequences were proposed by a placement sampler.**

**Root lineage:**
- The 635 roots use the namespaces `2d-mock2d_5000`, `2d-bxyg2pilot`, `2d-bxyg21dev`, `2d-bxyg34dev` and `2d-bxyg34confirm`.
- **Overlap with the 5,000-root class-quota corpus used by Q2: 0 roots.** The sampler and X/Y evidence therefore have no shared root provenance (Nov 5 §11.1).

### 5.2 The 5,000-root 2D class-quota corpus (legacy BD2D B/D labels)

**Scope and teacher:**
- `BD2D_TEACHER_V1` (`bd_2d.teach`, bd_2d.py:207).
- Splits come from the Q2 manifest: train 3,500 / val 751 / test 749.
- Classes: NONE 1,000, K0 1,250, K1 800, K2 800, K3 1,150.

**Y (target only, from `q0 = q_ref`):**
- 35,285 attempts (5,000 original + 30,285 D): positives 6,555; negatives 28,730 (`CERTIFIED` 18,301, `NO_PLAN_FOUND` 10,429).
- **No Y record stores a witness.** `original_witness` is returned by `teach` but not written by `record`, and D records never carry one.
- 25,565 further lattice attempts are unstored.
- **Teacher equivalence:** this is literally the same function (`bd_2d.t_goal`, same `QB` budget) that `BXY_Y_TEACHER_V1` calls. On the mock2d_5000 roots, P-min re-ran Y with replay and agreed with the legacy label in **661/661** comparable rows (528 Y=1, 133 Y=0).

**B sequences (23,804; executed with deterministic certified kinematics, `p16_common.verify_sequence`):**

| Outcome | Count | Step records stored? | Status for X |
|---|---|---|---|
| All relocations + target from the achieved state verified | 10,138 | **yes**: every step record and the target record, so the exact achieved q at each boundary is known | `LIMITED`: connector + replay of the records |
| All relocations verified, target failed | 3,929 | **no**: `plan=None` on failure (p16_common.py:76–78) | `REEXEC` |
| A relocation failed | 9,737 (`CERTIFIED` 6,326; `NO_PLAN_FOUND` 3,411) | no (not needed for X=0) | X=0 under the same relocation functions: `CERTIFIED` → `DIRECT`; `NO_PLAN_FOUND` → `LIMITED` (agreement sample, because budget and cache state can flip it) |

**Unique relocation-complete prefixes:**
- with step records: **19,026** (train 13,214 / val 2,872 / test 2,940);
- without records: **3,655** (2,551 / 574 / 530).

**Caveat on witness identity:**
- P-min compared X-teacher relocation witnesses with the legacy stored witnesses. They were byte-identical in only **250/972** comparable rows. Outcomes still agreed, since X=1 was found for 951 of them.
- The likely cause is per-root `DetPhysics` cache and mode-selection state.
- **Consequence:** reusing legacy 2D relocation records as X phases requires an explicit replay acceptance (`replay_x`) of the legacy records. You cannot assume the X teacher would regenerate them.

**Same-endpoint groups:** 2,950 (minimal vs wrong_order and others), of which 337 differ in relocation or composite outcome.

**Corpus cost:** 388,670 CPU-s over 85,171 calls, i.e. **4.56 CPU-s per call** (measured).

### 5.3 Q2 collection schema: individual placements, never X

- Q2 (`Q2_PRODUCTION_AMENDMENT_V1`) is **238,160 teacher queries** on the 5,000 class-quota roots, with the same split.
- Each trial is a **post-grasp individual placement** of one object to one destination cell, in one of three contexts:
  - the root scene;
  - a `prefix` context of one or two **recorded `HYPOTHETICAL` teleported edits** (`DESIGN_RANDOM_EDIT` / `DESIGN_TWO_EDIT`; "the queried object has not moved");
  - destination bands near / mid / far / vacated.
- Each trial records adaptive grasp-pair attempts, verified witnesses (up to 32), and certified, policy and unresolved statuses.
- **It contains no ordered multi-move execution, no carried robot state and no connector.** It is therefore **never X** and never Y (Nov 5 §11.1; contract §8).
- Its value for X/Y is indirect: the same roots and splits are where candidate-diverse X/Y should be collected, giving shared provenance.
- Partial run files were not inspected. The Q2 merge is the BWL-5 entry gate.

---

## 6. Task 4: reuse matrix

### 6.1 3D (SHELF66V3; cohort 1,582 roots; frozen splits 1,120 / 246 / 216)

| # | Item | Count (train / val / test) | Notes |
|---|---|---|---|
| 1 | **Existing records and verified scope** | 4,806 stored ordered sequences (3,409 / 718 / 679), all executed in MuJoCo with carried state and hash-valid step witnesses. 7,012 target-only Y records. 511 executed K0 targets. 59,972 v4 placement trials (sampler only). | No connector; no independent replay of any target or multi-step witness. |
| 2 | **Directly reusable X** | **0** | |
| 2 | **Directly reusable Y** | 4,367 Y=0 (3,118 / 639 / 610). Y=1: 0. | The negatives are operational; `CERTIFIED` 4,004 is within the declared grasp set. |
| 3a | Limited: **Y replay** | 2,645 positives (1,851 / 408 / 386), incl. 511 already executed in-teacher | stored witnesses, no planning |
| 3b | Limited: **X connector** (+ replay to rebuild state + independent replay) | **6,773 unique prefixes** (4,839 / 992 / 942), incl. 3,665 complete sequences | conditional on pinning the relocation policy (§3.3) |
| 3c | Limited: **X=0 from relocation failure** | 1,141 (817 / 162 / 162) | conditional on the same pin. `NO_PLAN_FOUND` 317 and `EXECUTION_NOT_VERIFIED` 40 deserve a replay/re-run sample. |
| 3d | Limited: **Y for X-candidate endpoints** | 2,929 unique endpoints (1,383 lattice-implied + 1,546 unknown) | re-query from `q_ref`, then replay the positives |
| 4 | **Complete re-execution** | 9 (8 / 1 / 0) unstored BFS tries; plus all, if the relocation policy is not pinned | |
| 5 | **Unresolved** | 490 non-cohort roots: 1,453 sequences, 1,806 prefixes, 2,430 Y records (incl. the **75 `DEPENDENT` order-discordant roots**). 3,054 unreferenced witness NPZ files. 9,403 unstored lattice Y outcomes. | needs a split decision or recomputation |
| 6 | **Missing contrast families** | see §7 | |
| 7 | **Additional physics** | see §8 | |

### 6.2 2D

| # | Item | Count | Notes |
|---|---|---|---|
| 1 | Existing records | **BXY sidecars:** 7,701 X / 6,608 Y on 635 roots (finalized-valid). **Class-quota corpus:** 23,804 legacy sequences + 35,285 legacy target-only labels on 5,000 roots. **Q2:** 238,160 placement trials (in progress; sampler only). | |
| 2 | **Directly reusable** | **Development:** 4,443 X / 3,831 Y (398 roots; train/val/dev lineages kept). **Held-out evaluation only:** P24 3,258 X / 2,777 Y. **Corpus:** 28,730 Y=0 and 6,326 `CERTIFIED` X=0. | P24 was used for the P24 confirmation and must not enter training. |
| 3 | **Limited verification** | **Corpus:** 6,555 Y=1 (regenerate + replay). 19,026 prefixes with records (connector + `replay_x` of legacy records). 3,411 X=0 `NO_PLAN_FOUND` (agreement sample). | the Y-teacher function is identical (661/661 prior agreement) |
| 4 | **Complete re-execution** | corpus: 3,655 relocation-complete prefixes without records | deterministic re-run, cheap in 2D |
| 5 | **Unresolved** | 2,218 class-quota non-selected root files (7,218 − 5,000; not scanned, no split) | |
| 6 | **Missing families** | see §7 | |
| 7 | **Additional physics** | see §8 | |

**Double-counting controls applied:**
- X is unique by `(root, ordered (object, destination) list)`.
- Y is unique by `(root, full object→cell arrangement)`, so AB/BA and shared prefixes share one Y.
- The P9 pilot rows reused inside the P-min sidecar are counted once (results-directory rows only).
- Prefix counts are unique trie nodes per root.
- All splits are the frozen historical ones: 3D `split_manifest_first_cohort.json`, 2D Q2 `split_manifest.json`, and the P-min design splits.

---

## 7. Missing causal / order contrast families

| Family (Nov 5 §9.5, §11.5, §15.5; contract §6) | 3D now | 2D now |
|---|---|---|
| **Candidate-diverse (sampler-proposed) destinations in X/Y** | **absent**: B uses first parking cells plus one `wrong_dest` per root | **absent**: all catalogue-constructed |
| X/Y on the **same roots** as the placement sidecar | partly: the B/D roots are the v4 placement roots, but the destinations are not | **absent**: 0 overlap between X/Y roots and Q2 roots |
| Same-endpoint AB/BA with **decided, differing X** | 0 decided; ≤ 35 potential in the cohort; 72 in the non-cohort `DEPENDENT` roots | present (188 in development, 111 in P24) |
| Omission "B without A" (necessary mover omitted) | absent as X (Y only, via singleton D) | present (`omission_0`, P-min `omission`) |
| X=1 / Y=0 | latent (pending connector) | present (848 in development) |
| X=0 / Y=1 | 267 potential | present (678 in development) |
| Connector-only failure (relocations OK, stop fails) | not measurable yet | 135 |
| Multiple destinations per object | absent | limited (`repair_park_variant`, `alternative_destination`) |
| Three-move prefix dependence with real carried state | executed (m=3: 1,334; m=4: 244) but X undecided | present (1,748 length-3 and 1,010 length-4 X rows across the sidecars, excluding P15) |
| K4 / audited NONE | absent (0 K4; NONE excluded from the cohort) | K4 absent in the corpus; present in P20 k-series |
| Travel `J` witnesses with connector + target | absent | present on X=1 & Y=1 pairs (`travel_parts`, `J_target`) |

---

## 8. Additional physics queries: measured counts vs estimates

**Measured unit costs** (from stored records; single-core, on the hosts that produced them):

| Code | Measurement | Value |
|---|---|---|
| U1 | 3D teacher wall per planner call (plan, or plan+execute) | **4.85 s** (188,577 s over 38,870 calls, 1,582 roots) |
| U2 | 3D lone-object K0 root: 2 target plans + 1 MuJoCo execution | **2.55 s median** (n=116). Used as an **upper bound** for replaying one stored pick-and-place. |
| U3 | 2D BXY X/Y teacher incl. independent replay | **8.1 CPU-s/query** (P-min: 32,157 CPU-s / 3,972 queries) |
| U4 | 2D BD2D corpus teacher | **4.56 CPU-s/call** |

**Measured counts → estimated cost.** Counts are measured. Costs are **estimates** of `count × unit cost`, single-core, with no parallel speed-up assumed.

| Work package | Measured count | Estimated single-core time |
|---|---|---|
| 3D-Y1: replay all stored Y positives | 2,645 replays | ≤ 2,645 × 2.55 s ≈ **≤ 1.9 h** |
| 3D-X1: connector pass over the prefix trie (replay each trie node once with state snapshot, then one connector plan + execute at each node) | 6,773 step-replays + 6,773 connector attempts | ≤ 6,773 × (2.55 + 4.85) s ≈ **≤ 13.9 h** |
| 3D-X2: independent from-scratch replay of the X=1 results | ≤ 11,555 step-replays (sum of prefix depths: 8,279 / 1,657 / 1,619) + ≤ 6,773 connector replays | ≤ 18,328 × 2.55 s ≈ **≤ 13.0 h** (upper bound: assumes all prefixes succeed) |
| 3D-Y2: Y records for X-candidate endpoints without one | 2,929 target plans (+ replays of the positives) | ≈ 2,929 × 4.85 s ≈ **3.9 h**, + ≤ 2.1 h of replays |
| 3D-X3: agreement sample of relocation failures (`NO_PLAN_FOUND` 317, `EXEC` 40) | ≤ 357 re-runs | **≤ 0.5–1 h** |
| **3D subtotal (recover existing evidence)** | | **≈ 36 h single-core upper bound** (1.9 + 13.9 + 13.0 + 6.0 + 1; ≈ 3 h on 12 workers; estimate) |
| 2D-Y: regenerate + replay corpus Y positives | 6,555 | 6,555 × 4.56–8.1 s ≈ **8–15 CPU-h** |
| 2D-X: connector + `replay_x` on corpus prefixes with records | 19,026 | ≤ 19,026 × 8.1 s ≈ **≤ 43 CPU-h** (upper bound: full X-teacher cost) |
| 2D-X: re-execute prefixes without records | 3,655 | ≈ 3,655 × 8.1 s ≈ **8 CPU-h** |
| 2D-X: `NO_PLAN_FOUND` agreement sample | ≤ 3,411 (sample ≈ 300) | **≈ 0.7 CPU-h** for the sample |
| **New candidate-diverse X/Y pilot, 3D** (e.g. 60 train/val roots × ~20 X + ~15 Y; avg m ≈ 2) | ≈ 1,200 X + 900 Y (**design estimate**) | ≈ 1,200 × ~20 s + 900 × ~7 s ≈ **8.5 h** |
| **New candidate-diverse X/Y pilot, 2D** (e.g. 100 Q2 train/val roots × ~20 queries) | ≈ 2,000 (**design estimate**) | ≈ 2,000 × 8.1 s ≈ **4.5 CPU-h** |
| Scale-up after the pilots (only if they establish decided contrasts) | design-dependent | 3D, all 1,366 train/val roots: ~190 h. 2D, 4,251 train/val roots: ~190 CPU-h (**rough estimates**, linear scaling) |

---

## 9. Prioritized minimal-additional-collection plan

All physics steps below belong on an **authorized physics host**, not Blackwell, and each needs **explicit authorization**. Nothing here was run.

### P0: desk decisions (no compute)
1. **Pin the 3D X teacher.** Recommended: "`BD3DS66_TEACHER_V2` relocation executor + terminal connector + independent replay".
   - Connector: `plan_free_move(q_now, q_ready)` + execute, then verify the arm at `q_ready` within tolerance, no object displaced, gripper opened to the reference state and nothing held.
   - This mirrors `bxy_teacher.x_query`.
   - Without this pin, the 6,773 / 1,141 reuse figures collapse to full re-execution.
2. **Decide the non-cohort 3D split extension**, at least for the 75 `DEPENDENT` roots, which are the only real 3D order-discordant family. Options: inherit by lineage (56 linked), seed the 19 unlinked, or exclude them and generate order-dependent cohort roots instead.
3. **Freeze reuse manifests** (`(root, x_key, y_key, source record, witness ids)`) so the physics host executes a predeclared list.

### P1: predeclared verification samples (≈ 1–2 h)
- 3D: replay ~100 `D` positives + ~50 K0 originals (Y); multi-step replay of ~100 stored sequences (determinism of the carried state); re-run ~100 relocation failures.
- 2D: ~300 corpus Y positives regenerated + replayed; ~300 `NO_PLAN_FOUND` relocation failures through `x_query`.
- **Gate:** agreement rates decide whether the bulk reuse in P2/P3 is accepted.

### P2: 3D Y bulk replay (≤ 2 h)
- Replay all 2,645 stored positives, which converts them to `Y_REPLAY_VERIFIED` or failed.

### P3: 3D X connector completion (≈ 14 h + ≤ 13 h replay)
- Train/val first: 5,831 prefixes. Test: 942 prefixes, for held-out evaluation only.
- Output: the first authentic 3D X labels, incl. connector-only failures.

### P4: 3D Y fill for X-candidate endpoints (≈ 6 h)
- 2,929 unique endpoints, giving decided (X,Y) pairs.

### P5: order and omission contrasts (pilot)
- AB/BA on order-dependent configurations (`DEPENDENT` roots, or newly constructed ones inside train/val lineages).
- "B without A" omissions.
- Report `EVIDENCE_BLOCKED` if decided discordant contrasts remain rare.

### P6: candidate-diverse X/Y pilots (3D ≈ 8.5 h, 2D ≈ 4.5 CPU-h)
- Sequences built from **predeclared** sampler-independent proposals (stratified/v4-verified destinations, cVAE, flow, random distractors), on cohort / Q2 **train/val** roots only, with splits fixed before generation.
- For 2D, this is the first X/Y on Q2 roots.
- Scale only after the pilot shows decided outcome coverage.

### 2D-specific
- Use the 4,443 X / 3,831 Y development pool now (BWL-6 entry, construction-informed caveat). Keep P24 as held-out evaluation only.
- Corpus reuse (P1 2D sample → regenerate Y positives → connector on 19,026 prefixes) is optional. It is **valuable mainly because it sits on the Q2 roots**, which gives shared provenance with the sampler.

---

## 10. Evidence limitations and threats to validity

1. **No independent replay was run in this audit.** "Replayable" means the stored data and the code path are sufficient. Only 30 single-step 3D replays have ever been checked.
2. **3D connector feasibility is unknown.** From the retreat pose deep in the shelf, a free move to `q_ready` may fail at a material rate. All 6,773 prefixes are *potential* X, not X=1.
3. **Cohort order-independence by design.** The 3D cohort and the 2D corpus admit only `independent_moves` scenes, so order effects are suppressed. 3D AB/BA discordance is concentrated in excluded roots.
4. **Destination bias.** 3D and 2D-corpus sequences use first parking cells (plus one wrong destination). 2D sidecars use construction-informed catalogues. Training on them and testing on sampler candidates is the §11.5 shift risk.
5. **Lattice-implied Y** (`blocker`) depends on the full-lattice assumption plus `minimal_sets` semantics, and has no record or witness pointer. Treat it as a pointer for re-query, not a label.
6. **Planning negatives are operational.** `CERTIFIED` is scoped to the declared grasp set and the deterministic geometry. `NO_PLAN_FOUND` is budget- and seed-scoped.
7. **2D legacy witness identity is low (250/972).** Outcome-level equivalence for corpus X is assumed only for `CERTIFIED` failures. Everything else needs the P1 sample.
8. **Snapshot scope.** The 3D store is the 2026-11-07 05:28Z live cut used by the v4 cohort. The generator kept running afterwards; later roots are outside this audit.
9. **Unit costs** are wall or CPU seconds on the original hosts. Estimates assume linear scaling and no failures; treat them as order-of-magnitude.
10. **Prior report 05 refined, not contradicted.** It counted 5,317 3D `B` entries; this audit counts 4,806 relocation sequences, and 4,806 + 511 K0 empty sequences = 5,317. It said "original/D valid as Y"; this audit refines that: negatives are reusable, D positives are planning-only, and K0 originals were executed in-teacher. It said "X=0 only one-sided"; this audit shows 1,141 individually valid X=0 under the pin, plus 6,773 connector-pending prefixes.

---

## 11. Reproduction

Run from `approach_b_reuse_audit/` (read-only inputs; ≈ 25 s, < 60 MB each):

```bash
P=/home/naren/miniconda3/bin/python; B=~/Desktop/rss-3d-relocations/gh-yadu-iiith/shelf_6x6_open_sides
nice -n 19 $P -I audit3d_scan.py $B/dataset_bd_classquota_without_home_position_no_overpass/raw $B/dataset_placement_extension_v2/production_v3 audit3d_scan.json
nice -n 19 $P -I xtab3d.py      $B/dataset_bd_classquota_without_home_position_no_overpass/raw $B/dataset_placement_extension_v2/production_v3
nice -n 19 $P -I inv2d_xy.py    ~/Desktop/boltzmann_lfd/.claude/worktrees/approach-b-xy-v1/runs/approach_b inv2d_xy.json
nice -n 19 $P -I scan2d_corpus.py ~/Desktop/rss-placement-smoke/gh-yadu-iiith/planar_2d/dataset_bd_classquota \
     ~/Desktop/rss-2d-q2-commit/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2/split_manifest.json scan2d_corpus.json
nice -n 19 $P -I final_measures.py  $B ~/Desktop/rss-placement-smoke/gh-yadu-iiith/planar_2d/dataset_bd_classquota ~/Desktop/boltzmann_lfd/.claude/worktrees/approach-b-xy-v1/runs/approach_b
nice -n 19 $P -I prefix_measures.py $B ~/Desktop/rss-placement-smoke/gh-yadu-iiith/planar_2d/dataset_bd_classquota \
     ~/Desktop/rss-2d-q2-commit/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2/split_manifest.json
```

| Output | sha256 |
|---|---|
| `audit3d_scan.json` | `92348bee9d208cf0bd28a82b8e5adda44d301c027dd95d98b9f07a874afc7306` |
| `xtab3d.json` | `1850825ff3becdc53e5d2b811f82aa4aab486da3d35181109fb2da01e9fa6abd` |
| `inv2d_xy.json` | `91e94956771703d3a954cdd9ca1d763d27f0e0d804a1b9e9d7972b3212622088` |
| `scan2d_corpus.json` | `bb9f7d9b888ad6fda50337b3008d06e1481397a9c4b657240798fb9bb0030fb3` |

**Example record identities:**
- 3D root `3ds66-lap3-w0047-s014` (K2): file sha256 `ed00e675…`.
  - `minimal` = `o00→(5,0)`, `o01→(1,5)`; its `wrong_order` reaches the same endpoint (Y=1 via `blocker.minimal_sets`).
  - Shared-intended-union `D` witness `87ee2872cfe3918454e32745`.
- 3D K0 root `3ds66-lap3-w0047-s000`: original and executed target share witness `967048492edcbf89321b79da`.

**Status: audit complete. STOP.** No methodology, code, configuration, dataset or run state was changed. The only new files are this report and `approach_b_reuse_audit/`, both uncommitted.
