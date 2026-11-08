# 13: Evidence recovery, physical pilot and training gate (single-session continuation, S0–S4)

**Work order.** `APPROACH_B_SINGLE_SESSION_NEXT_EXECUTION_v1.md` (2026-11-08). Scientific authority: `final-fixed-methodology-1.pdf` (sha256 `b7a39e3c…`).

**Scope of this session:**
- **Ran:** a read-only reconciliation; a bounded physics pilot and **one** bounded recovery chunk on gvlab; no physics on Blackwell.
- **Not touched:** the 2D Q2 runs; the method, sampler experiments and archived data.

**Verdicts:**

| Field | Verdict |
|---|---|
| S0 | PASS |
| S1 | **RECOVERY_GO** |
| S2 | chunk 0 run (see §5); the full recovery is **not** dispatched |
| S3 | readiness prepared; no new training run |

**STOPPED after S4.**

## 1. Complete vs proposed

| Done (with evidence) | Proposed / prepared, not executed |
|---|---|
| Reuse audit located, preserved in three places, counts reproduced | Full historical recovery: 52 remaining S2 chunks |
| Canonical X / Y keys; dedup overlap of history × the BWL-3 pilot × the 7,012 Y records | `BD3DS66_Y_PLAN_V1` for the 2,929 endpoints with no Y record (needs a new plan + execute + replay protocol) |
| Old 3D teacher pinned and validated on the physics host (policy parity, bitwise replay) | Candidate-diverse 3D / 2D X/Y pilots (after historical recovery) |
| **S1 pilot: 145 / 145 items decided, 0 mismatches** | Hard-order challenge cohort collection (membership proposed) |
| S2 chunk 0 (30 train roots): 202 / 202 decided; first authentic 3D X / Y sidecar (199 X, 342 Y) | D / E / DIRECT-GEO / G training configs on authentic 3D labels (gated) |
| Films of verified runs (rendered from the exact recorded simulator states) | 2D Q2 intake (after the merge) |

## 2. Reuse audit: preserved and reconciled

**Location.** It was found untracked on the laptop: `boltzmann_lfd/APPROACH_B_EXISTING_DATA_REUSE_AUDIT.md` (sha256 `0d3c79dd…`) and `boltzmann_lfd/approach_b_reuse_audit/` (10 files).

**Preserved in three places:**
1. the original location;
2. `~/Desktop/approachb_reuse_audit_preserved_20261108/`, with its own `SHA256SUMS` (manifest sha `359a884c…`);
3. this branch: `reports/reuse_audit_20261108/`, verbatim (all 10 files pass the audit's own `SHA256SUMS`).

**Scripts.** Analysis-only: the only writes are their own JSON outputs.

**Reproduced here (laptop, nice 19):**
- `audit3d_scan.py` and `xtab3d.py`: **every count identical**.
- The only difference is the illustrative `abba_examples` list, whose order varies with Python's set ordering.

**Provenance gap closed.** The audit's non-cohort figures (75 DEPENDENT roots, lineage 41 / 10 / 5 / 19, 72 discordant) were in the report but produced by no saved script. They are now recomputed by `physics_pilot/challenge_cohort.py`, which gives the **same numbers**.

| Quantity (3D cohort, 1,582 roots) | Audit | This session |
|---|---|---|
| Unique executed relocation prefixes (train / val / test) | 6,773 (4,839 / 992 / 942) | **6,773 (4,839 / 992 / 942)** (`s0_overlap.py`) |
| Endpoints of executed prefixes without any Y record | 2,929 | **2,929** |
| Y records / planning positives / operational negatives | 7,012 / 2,645 / 4,367 | same (BWL-3 `11` and here) |
| Relocation failures | 1,141 sequences | **887 unique failing-prefix keys**. These dedupe sequences that share a failing step, e.g. `noop_irrelevant` equals the first step of `redundant+1` |
| Complete sequences | 3,665 sequences | 3,094 unique keys (same deduplication) |
| Non-cohort DEPENDENT roots, discordant groups | 75; 72 / 75 | **75; 72 / 75 relocation-discordant** |

## 3. Canonical keys and deduplication (S0)

**X key** = `(root_id, X teacher BD3DS66_X_V1, q_ref = q_ready, ordered moves "obj>r,c.obj>r,c", stop-point connector)`.

**Y key** = `(root_id, t0 → root goal, q_ref = q_ready, gripper free, full endpoint arrangement "obj@r,c|…", Y teacher BD3DS66_Y_REPLAY_V1)`.

AB and BA share one Y key and have distinct X keys. Files: `physics_pilot/S0_OVERLAP.json` and `S0_HISTORICAL_PREFIXES.json.gz`.

**The BWL-3 887-query X pilot against history:**

| | Has a stored Y record (positive → replay only; negative → reusable) | No Y record |
|---|---|---|
| **Relocations already executed and stored** → replay + connector only | 93 | 41 |
| **Relocation failure already stored** → reconfirm only | 15 | 2 |
| **New full execution needed** | 55 | 681 |

**Duplicates eliminated:**
- **151** of the 887 BWL-3 X queries are historical executions, so their relocations are not re-run.
- **129** of its 642 endpoints already have Y records.
- The other 736 are fresh executions, deferred to the candidate-diverse stage after recovery.

## 4. S1: physics-host evidence (gvlab)

**Host eligibility**, verified before any run:
- **Authorized:** gvlab is the 3D v4 producer.
- **Identical sources:** all **37 / 37** pinned source files and all **1,582 / 1,582** cohort root hashes equal the source pin.
- **Same simulator:** MuJoCo **3.16.0**, as pinned in the generator's `requirements.txt` and in its own environment, so there is no simulator drift.
- **Room for the pilot:** the Q2 spillover uses CPUs 0–11 (12, 13 reserved), its project ceiling is 18 CPUs, its thermal pause is at 88 °C, and the host was at 47–50 °C.
- **Pilot footprint:** 2 workers on **CPUs 16–17**, nice 19, idle I/O.
- **Pilot guard:** stop on package ≥ 75 °C, MemAvailable < 16 GB, or swap growth > 256 MB.
- **No Q2 interference:** the executor's name (`bwlphys.py`) matches none of the spillover's process-scan substrings.

**Teacher pinned** (`physics_pilot/bwlphys.py`, the S1 version's sha256 prefix `2eb72234`):
- **X teacher (BD3DS66_X_V1).** It works from `settle(S0)` at `q_ready`:
  - it replays the **stored relocation witnesses** with the unchanged `run_plan`, in order, with the robot state carried (no reset) and `bd_3dt.seq`'s per-move checks;
  - each witness id is re-derived from its content on load;
  - it then adds **CONNECTOR_TO_QREADY_V1**: `plan_free_move(q_now, q_ready)` (the env's RRT-Connect, 5,000 iterations, seed 0), executed without `go_ready`'s straight-line fallback, then the gripper opened to `GRIPPER_OPEN`;
  - the connector is verified: |q − q_ready| ≤ 0.02 rad, nothing displaced by more than 4 mm, every object in its cell, nothing held, no robot–object contact;
  - X = 1 requires an **independent replay in a fresh process** that agrees.
- **Y teacher (BD3DS66_Y_REPLAY_V1).** Replay of the stored target witness from `q_ready` in the ideal endpoint, done in two fresh processes. Negatives get a deterministic re-query.

**Policy parity** (`results_s1/POLICY.json`, checked per item):

| Check | Result |
|---|---|
| Teacher version, robot protocol `XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022`, config hash `8d521b1f6d23db23`, scene source hashes | equal to the root's recorded values |
| `q_ready` vs the audit's q_ref | 4.0e-7 rad |
| Witness start vs `q_ready` | ≤ 4e-10 rad |

**Pilot.** Manifest `S1_PILOT_MANIFEST.json`, predeclared, items sha256 `b3804090…`. It ran 14 smoke items, then 131 pilot items.
- **Y items:** the BWL-3 replay manifest minus 18 planner-budget negatives.
- **X items:** **train / val roots only**, in strata.

| Stratum | n | Result |
|---|---|---|
| Original positives (K0, executed once in the generator) | 24 | **24 Y = 1 REPLAY_VERIFIED** |
| D positives, single-object edit | 24 | **24 Y = 1 REPLAY_VERIFIED** |
| D positives, multi-object edit | 24 | **24 Y = 1 REPLAY_VERIFIED** |
| Certified negatives (original / D) | 12 / 12 | 24 re-confirmed (same kind) |
| Planner-budget negatives (original / D) | 3 / 3 | 6 re-confirmed (same kind) |
| K1 complete single move | 6 | 6 X = 1 verified |
| K2 AB / BA, both orders executed | 10 (5 pairs) | 10 X = 1 (5 concordant pairs) |
| AB / BA, order-discordant relocation | 6 (3 pairs) | **3 X = 1 + 3 X = 0: 3 decided discordant order contrasts** |
| K3 complete three-move | 5 | 5 X = 1 |
| Proper prefix (intermediate stop) | 6 | 6 X = 1 |
| Relocation failures (certified 3, budget 1, execution 1) | 5 | 5 X = 0 re-confirmed |
| Complete sequences without a recorded terminal arm state | 5 | 5 X = 1. The replay rebuilt the missing state |
| **Total** | **145** | **145 decided; 0 mismatch, 0 environment errors, 0 timeouts** |

**Reproducibility and state reconstruction:**
- Teacher vs fresh-process replay: final state **max difference 0.0** (bitwise) on every item.
- Replayed move boundaries vs the recorded achieved arm state: 61 boundaries, median 3e-16 rad, **max 2.2e-11 rad**.
- Every replayed step agrees with the stored record (43 / 43 X items).
- The gripper and attachment state is reconstructed by the replay itself: release at the grasp's open width, `held = None`.

**Connector:**
- **35 / 35** planned and executed. Final |q − q_ready| ≤ 4.3e-4 rad, maximum tracking lag 1.3e-3 rad, 0 pushes, 0 contacts.
- **No connector failures were observed**, so connector-only X = 0 cases are not yet measured in 3D.

**Failed phases.** All 8 X = 0 failed at a **relocation**:
- certified grasp screen: 3 + 3 (discordant pairs);
- planner budget: 1;
- execution: 1.

**Cost.**
- Wall time: 4,420 s over 290 fresh processes.
- **CPU: 1.23 CPU-h** of the 6 CPU-h cap. Wall time was about 50 min on 2 workers.
- Unit costs:

| Item | Time |
|---|---|
| Y replay | 3.3 s |
| X item (replay + connector) | 10–20 s |
| Certified re-screen | about 70 s |
| Budget re-plan | about 100 s |

**Films** (`figures/films/`, decimated; full-length copies with hashes in `results_s1/FILMS_FULL_SHA256SUMS`):
- `s1_X-016` / `s1_X-017` (a discordant AB / BA pair) and `s1_Y-BWL3Y-0048` (edited-endpoint Y = 1). The K3 film `s1_X-022` is kept outside git (`~/Desktop/approachb_3d_learning_films_full/`).
- gvlab has no offscreen GL. The exact simulator states were recorded on gvlab (`trace`, which reproduces the same verdicts) and rendered on the laptop with `mj_forward` only, no simulation.

**S1 verdict: `RECOVERY_GO`.** The historical relocations and target witnesses replay deterministically under the pinned policy, the connector works, and every stored negative re-confirms.

**Limits:**
- Connector failures are not yet sampled, so the X = 1 rate for executed prefixes may be optimistic in harder shelf states.
- Y replay checks the stored witnesses only; the 2,929 endpoints without a record need `Y_PLAN_V1`.

## 5. S2: historical recovery work order and chunk 0

**Work order.** `physics_pilot/S2_RECOVERY_MANIFEST.json.gz` (items sha256 `d4de3f89…`) covers 1,582 roots in 53 chunks of 30 roots, ordered train → val → test and by hash within each split.

| | train | val | test |
|---|---|---|---|
| Executed X prefixes | 4,808 | 988 | 942 |
| Failing prefixes to re-confirm | 628 | 123 | 128 |
| Y positives to replay | 1,803 | 387 | 383 |

- Items done in S1 are skipped.
- Execution is idempotent: it resumes per item.
- Negative re-confirmations run once (their determinism was shown in S1).

**Chunk 0** (30 train roots; 202 items): the only chunk run, launched because S1 passed and it fit inside the remaining pilot cap.

| Result | Count |
|---|---|
| X = 1 verified | **143** |
| X = 0 relocation failures re-confirmed | **13** |
| Y = 1 replay-verified | **46** |
| Mismatches, environment errors, timeouts | 0 |

**Quality and cost:**
- Teacher vs replay max difference 0.0.
- 235 move boundaries, max 8.5e-11 rad.
- Connector **143 / 143**.
- Cost **0.84 CPU-h** (S1 + chunk 0 = **2.07 of the 6 CPU-h cap**), about 50 min wall on CPUs 16–17.
- 0 guard violations; package temperature ≤ 50 °C.
- Evidence: `physics_pilot/results_s2/`.

**Authentic 3D label sidecar so far:** `physics_pilot/BD3DS66_XY_SIDECAR_V0.json` (S1 + chunk 0).

| | Count |
|---|---|
| X labels | **199** (train 174 X = 1 / 20 X = 0; val 4 / 1) |
| Y labels | **342**: 118 `Y_REPLAY_VERIFIED` Y = 1, 224 archived operational Y = 0 (kind and scope kept) |
| Decided (X, Y) pairs | X1/Y1 **31**, X1/Y0 **42**, X0/Y1 **4**, X0/Y0 **6**: all four cells now exist in 3D |
| X with Y still missing | 116; their endpoints have no Y record at all, so they need `Y_PLAN_V1` |
| Decided same-endpoint order groups | 21, of which **3 discordant** |

**Rules applied to the sidecar:**
- A planning-only positive that has not been replayed stays MISSING, never 1.
- A connector-stop failure would apply only to that stopping prefix (none occurred).

## 6. Hard-order challenge cohort (75 DEPENDENT roots): proposal

File: `physics_pilot/CHALLENGE_COHORT_PROPOSAL.json`, rows sha256 `1cce1f60…`. The roots are K2 28, K3 40, K4 7, all with `independent_moves = False`.

**Membership rule:**
- A **lineage link** is a shared parent family or an identical layout. A linked root inherits the **strictest** linked split (test > val > train): 41 → train, 10 → val, 5 → test.
- The **19 unlinked** roots are split by a seeded hash: 40 % holdout, the rest split between dev-train and dev-val.

**Result: CHALLENGE_DEV_TRAIN 47, CHALLENGE_DEV_VAL 15, CHALLENGE_HOLDOUT 13.**

**Safeguards:**
- The frozen 1,582-root split and its test set are unchanged.
- No challenge root is appended to them.
- The challenge holdout is **aggregate-inspected, not pristine**: its stored composite outcomes were counted by the audit and by this script.
- None of these roots has authentic X / Y yet: they need BD3DS66_X_V1 collection, which is not run.
- No prevalence of order effects may be inferred from a cohort selected *for* order dependence.

## 7. 2D readiness and Q2 status (S3)

**Historical 2D X/Y (BXY teacher; finalized-valid).** From the audit's `inv2d_xy.json`:
- **Development:** 4,443 X / 3,831 Y on 398 roots.
- **P24 held-out:** 3,258 X / 2,777 Y on 237 roots. **Excluded from any development.**
- The 60 frozen P21 D / E / DIRECT-GEO / G members are present (`~/bxy_checkpoint_backup/p23_1_work/stage/p21_runs`, 830 MB backup).

This pool is usable now for an **intended-catalogue baseline**: its candidates come from construction-informed catalogues, not a sampler, and it is not candidate-diverse training. **No 2D training was run in this session.**

**2D Q2.** These are placement-only labels, never X / Y. Status at 04:26Z, untouched:

| Run | Roots done |
|---|---|
| Laptop remainder | 224 / 2,231 |
| gvlab spillover | 240 / 2,188 |
| Original (frozen) | 201 |

Both runs do about 51 roots per hour, putting the estimated finish around 9 Nov 18:00–20:00Z. That is an estimate, not a trigger. The intake audit (`BWL-5`) follows the merge and the split audit.

## 8. Resource telemetry

**gvlab:**
- Pilot workers on CPUs 16–17 only.
- Package temperature 47–50 °C throughout, MemAvailable ≥ 59 GB, swap unchanged (93 MB), **0 guard violations, 0 stops**.
- The Q2 workers kept running on CPUs 0–11 with no pause.
- **Incident:** one of my read-only searches (`find /`) kept running remotely after its local session was stopped. I killed it by verified PID; it touched nothing.

**Laptop:** only nice 19 analysis and rendering; the Q2 remainder was unaffected.

**Blackwell:** idle. No physics there, and no GPU job this session.

## 9. Remaining recovery and candidate-diverse deficits

Counts are measured. Times are CPU-h estimated from the unit costs **measured in this pilot**.

| Package | Remaining | Estimated CPU-h |
|---|---|---|
| Y positive replays (two fresh processes each) | 2,645 − 118 = **2,527** | ≈ 4.6 |
| X executed prefixes (replay + connector + independent replay; ≈ 12.7 CPU-s each) | 6,773 − 178 = **6,595** (train / val 5,653; test 942, held out) | ≈ 23 |
| Relocation-failure re-confirmations (≈ 70 s each) | 887 − 21 = **866** | ≈ 17 if all re-run. 21 / 21 agreed so far, so a certified-failure **sample** would cut this to ≈ 2 |
| `Y_PLAN_V1` for endpoints without a record | **2,929** | ≈ 10–45. Planning negatives are slow: about 70–100 s when every grasp is tried |
| **Total** | | **≈ 40–90 CPU-h** |

**Wall-time options:**

| Option | Wall time |
|---|---|
| Now, 2 workers on CPUs 16–17, no Q2 impact | ≈ 20–45 h |
| Now, 4 workers on CPUs 14–17 (inside the project ceiling) | ≈ 10–23 h |
| After Q2 finishes (≈ 9 Nov evening), most of gvlab's 24 CPUs | ≈ 2–5 h |

**Deficits that recovery cannot fill:**
- **Candidate-diverse (sampler-sourced) X / Y: 0 in 3D and 0 in 2D.** The BWL-3 pilot's 736 fresh-execution queries come after recovery.
- **3D order contrasts are scarce** in the order-independent cohort: ≤ 35 potential, 3 decided so far. Plus 72 in the challenge cohort, which is not collected.
- **Connector failures were never observed** (178 / 178), so connector-only X = 0 is not yet represented.
- "B without A" omissions and multiple destinations per object are absent from history.

## 10. Go / no-go

| Decision | Verdict | Evidence / condition |
|---|---|---|
| **S1 historical recovery protocol** | **GO** (`RECOVERY_GO`) | 347 / 347 items decided, 0 mismatches, bitwise replay, 178 / 178 connectors |
| **Larger host budget for the full S2 recovery** | **GO recommended; needs your budget approval** | Choose one: (a) 2–4 workers on CPUs 14–17 now, about 10–45 h, with Q2 untouched; or (b) wait for Q2 to finish, then most of gvlab, about 2–5 h. Also decide whether certified relocation failures are re-run in full or by sample. **Nothing beyond chunk 0 was dispatched.** |
| `Y_PLAN_V1` (2,929 endpoints) | **NOT YET**: protocol to implement and pilot first | needed for the X / Y pairing of 116 + about 3,700 prefixes |
| **BWL-5** (2D Q2 intake + 2D sampler training) | **NO-GO now** | waits for the merged dataset hashes and the split audit (Q2 at 224 + 240 of 4,419 roots) |
| **BWL-6, 3D** (real D / E / DIRECT-GEO / G) | **NO-GO now; GO after full S2 + `Y_PLAN_V1`** | Today: 199 X, 21 order groups, 3 discordant; training would mostly measure X / Y on order-independent scenes. Afterwards, about 6.6k X with paired Y supports a first matched run, with F5 still a hypothesis |
| **BWL-6, 2D** (intended-catalogue baseline) | **Ready; needs explicit authorization** | 4,443 X / 3,831 Y on 398 development roots, P24 excluded. The frozen P21 members already exist, so retraining is optional |
| **BWL-7** (branch-local search) | **NO-GO for physical claims** | Deterministic search tests that use no teacher data could start (not done in this session) |
| **BWL-8** (independent physical confirmation) | **NO-GO** | needs frozen models first |

**STOPPED.**
- No job of mine is running on any host: the gvlab workers have exited, the Blackwell GPU is idle, and the laptop is idle apart from Q2.
- The 2D Q2 runs continue untouched.
