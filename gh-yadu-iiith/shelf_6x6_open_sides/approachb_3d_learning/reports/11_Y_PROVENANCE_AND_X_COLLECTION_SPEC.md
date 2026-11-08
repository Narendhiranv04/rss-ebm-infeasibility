# 11: BWL-3 — 3D Y provenance audit and physics-host manifests for Y replay and authentic X

**Stage verdict: BWL-3 PASS (audit and manifests complete).**

| Verdict field | Value |
|---|---|
| `Y_3D_REPLAY_VALIDITY` | **PLANNING_ONLY** |
| `X_3D_AUTHENTIC_DATA` | **ABSENT** |

**Scope.**
- Nothing was relabelled.
- No physics or MuJoCo ran on the GPU server.
- No Y-only or X/Y training was run.

**Raw outputs.**
- `results_bwl/Y_PROVENANCE_AUDIT.json`, plus a row-level file `Y_PROVENANCE_ROWS.json.gz` (7,012 rows).
- `PHYSICS_Y_REPLAY_MANIFEST.json`.
- `PHYSICS_X_PILOT_MANIFEST.json.gz`.
- Figure `bwl3_y_provenance.png`.

**Command.** `taskset -c 7 nice -n 15 venv/bin/python src/xy3d_data.py yaudit | manifest-y`. `manifest-x` ran once under the GPU guard, for a tiny inference of cVAE destinations.

## 1. What the generator actually did

Source: `bd_3dt.py`, sha256 `62dc853c…`, repo commit `6f042be1`. Every root's `teacher_version` is `BD3DS66_TEACHER_V2 … fingers pre-shaped …` (1,582 / 1,582).

| Fact | Source lines |
|---|---|
| A target-only query `q_t` = `settle(cells)` followed by `query(e, 't0', goal)` | l.245–249 |
| `settle` resets the env (home cells, **READY pose**) and teleports the edited objects. So Y starts from the fixed **q_ref** in an **ideal** endpoint scene | l.145–150 |
| `query` = `plan_pick_place` over the scene's grasp poses; the first feasible one wins; "**Nothing moves**". **The target plan is never executed or replayed** | l.96–108 |
| `CERTIFIED` = every grasp fails a deterministic geometric check of the finite grasp set (no search); otherwise `NO_PLAN_FOUND` (RRT-Connect, 5,000 iterations) | l.124–136 |
| Only `seq()` runs plans in MuJoCo (`run_plan` l.280). It executes relocations from the **achieved** state and appends the target from there (the B records). There is no connector to q_ref, so B is composite: **neither X nor Y** | l.265–292 |
| K0 roots: `seq([])` executes the target **once** from the ready pose. This is an in-teacher execution, not an independent replay | l.298–300 |

## 2. Counts (all 7,012 records of the 1,582 QC roots)

| Category | original (train / val / test) | D (train / val / test) | Total |
|---|---|---|---|
| `Y_REPLAY_VERIFIED` | 0 | 0 | **0** |
| `Y_PLANNING_ONLY` (positive: a plan was found, never executed) | 358 / 86 / 67 | 1,493 / 322 / 319 | **2,645** |
| `Y_POLICY_FAILED:CERTIFIED_GRASP_SCREEN` (attempted; every grasp of the protocol's finite set geometrically blocked) | 724 / 147 / 140 | 2,146 / 424 / 423 | **4,004** |
| `Y_POLICY_FAILED:PLANNER_BUDGET` (attempted; RRT budget exhausted) | 38 / 13 / 9 | 210 / 55 / 38 | **363** |
| `Y_UNATTEMPTED_OR_UNRESOLVED` | 0 | 0 | 0 (unqueried endpoints are simply absent) |
| `Y_SEMANTICS_INCOMPATIBLE` | 0 | 0 | **0** |

**Semantic checks** (0 failures each): teacher version, label protocol `BD_BINARY_BUDGET_V1`, start pose `home`, edit legality (in bounds, free, distinct, not the target), witness present exactly when y = 1, no duplicate endpoint per root.

**Edit sizes of positives:** 0 (root): 511; 1: 1,328; 2: 317; 3: 302; 4: 130; 5: 44; 6: 13.

**The 511 original positives are exactly the K0 roots.** Each was also executed once in the teacher (`seq([])`, y = 1).

**How the categories can be used:**
- **Negatives** (4,367) are valid *operational* Y = 0 under the pinned planning policy, with their scope kept. A certified negative is relative to the protocol's finite grasp set; a budget negative is not an impossibility. Without a plan, the execute-and-verify Y protocol cannot succeed either, assuming the same planner and grasp set.
- **Positives** (2,645) are **not** final Y: the November 5 Y needs the target actually verified from q_ref. They are at most *preliminary planning supervision* (`Y_PLANNING_SUPERVISION_PRELIMINARY`), and a Y-only F3 run on them is **not authorized here**.
- **Edit distribution:** every D edit moves objects to their single designated parking cell. That is a narrow intervention family.

## 3. Y replay sample (physics host only): `PHYSICS_Y_REPLAY_MANIFEST.json`

**Selection.** 120 records, rows sha256 `791f5e95…`, chosen by `sha256('BWL3_YREPLAY|'+root+'|'+edit)` within strata, fixed before any replay:

| Stratum | Records |
|---|---|
| Original positives (K0, executed once) | 24 |
| D positives, single-object edit | 24 |
| D positives, multi-object edit | 24 |
| Certified negatives (original / D) | 12 / 12 |
| Planner-budget negatives (original / D) | 12 / 12 |

**Protocol.**
- **Positives:** (1) an independent MuJoCo replay of the **stored witness** from q_ref, passing only if T ends in the goal cell upright (|z_err| ≤ 0.01 m) and nothing else is displaced; (2) re-plan with seeds 0–2 and execute each.
- **Certified negatives:** re-run the deterministic screen.
- **Budget negatives:** re-plan with seeds 0–2 at the same budget and once at 4× (a sensitivity check that never relabels).
- **Statuses:** `REPLAY_PASS / REPLAY_FAIL / PLAN_IRREPRODUCIBLE / CENSORED / ENV_ERROR`. A censored run is never 0.

**Promotion.** `Y_REPLAY_VERIFIED` is granted per record only when its own witness passes. Promoting the whole pool needs all **2,645** positive replays.

## 4. Authentic ordered-prefix X pilot (physics host only): `PHYSICS_X_PILOT_MANIFEST.json.gz`

**Design (predeclared; rows sha256 `c541a464…`).**
- **Roots:** QC roots of the inherited **train and val** splits only, 10 per K0–K3 (8 train + 2 val), chosen by `sha256('BWL3_XPILOT|'+root)`. That is 40 roots (32 train / 8 val). **No test root is used.**
- **Queries:** 887 X queries plus 40 no-op rows (X ≡ 1 by convention; Y0 measured), over **682 unique Y endpoints** + 40 Y0.

| Query kind | Count |
|---|---|
| One-move stopping prefixes | 431 |
| AB | 164 |
| BA with the same endpoint | 164 |
| Three-move orders (4 of 6 permutations) | 108 |
| Nuisance move + historical repair | 18 |
| Distinct triple omissions (see the merged tags below) | 2 |

- **Same-endpoint order groups (m ≥ 2): 191.**
- **Merged provenance tags.** A physical query is listed once even when several design kinds produce it. Including merged tags, the design has 81 triple omissions, 30 historical minimal repairs, 20 historical reversed orders and 50 historical omissions.
- **Destination sources**, all fixed before outcomes:

| Source | Queries |
|---|---|
| Historical parking cell | 387 |
| **Frozen sampler `cvae_s0`** (seeded by the query tag) | 263 |
| Uniform random legal cell | 111 |
| Stratified band | 108 |
| Nuisance + historical | 18 |

- **Legality.** Every destination is free in the root scene, so both orders of every AB / BA pair are legal, and every endpoint passes the static legality rule.

**X protocol.**
- Start from S0 at the READY pose (= q_ref), gripper free.
- Relocate in order with the pinned BD3DS66_TEACHER_V2 planner, **carrying the achieved robot state** (no reset).
- Verify each move: the mover ends in its destination cell upright, and nothing else is displaced.
- At the stopping point, plan **and execute** a free-gripper **connector to q_ref**.
- X = 1 only if every phase and the connector succeed **and** an independent replay of the full witness reproduces it.
- X = 0 only for an attempted query, recording the failed phase (grasp / transfer / place / release / retreat / transition / connector), the negative kind, attempts and budget. Unattempted queries and environment errors are **missing**.

**Y protocol.** For every unique endpoint: settle the ideal endpoint at q_ref, plan the original target, **execute** it, verify it, and replay it. Y is measured even when X = 0.

**Travel J.** The joint-space path length of the executed X witness (relocations, transitions, connector) plus the executed target path. This is the November 5 convention, and J is recorded only for complete (1, 1) witnesses.

**Record schema.** The contract's §12 `XYTeacher` fields.

**X must never be inferred from:** placement labels, endpoint checks, B composites, one order's success applied to its permutations, or teleported edits.

**Go / no-go after the pilot.** Scale up only if the pilot yields decided AB / BA pairs with discordant X, all four (X, Y) cells, replay-reproducible witnesses and a measured teacher cost. Otherwise the architecture comparison is `EVIDENCE_BLOCKED`.

## 5. Gap matrix

| Supervision | Exists now | What is missing |
|---|---|---|
| X (ordered prefix + connector to q_ref) | **0** | the whole pilot above, then a scaled collection |
| Y (target executed and replayed from q_ref) | **0** replay-verified; 2,645 planning-only positives; 4,367 operational negatives | replay of the positives (the sample first); Y for every new X endpoint |
| Travel J (complete witnesses) | **0** | executed X and Y witnesses |
| Same-endpoint order contrasts | **0** decided | the pilot's 191 order groups |
| Omission contrasts | **0** decided | the pilot's omission and historical-omission queries |
