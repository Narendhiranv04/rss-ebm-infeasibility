# Hybrid experiment log

Each entry follows the same order: hypothesis → design (fixed before results) → result → interpretation → next.

**Status of evidence:**
- Every result before E7 used the same Approach-B VAL roots, about 20 comparisons in all, so all of it is **exploratory**.
- Confirmatory claims need a pre-registered run on TEST or on freshly generated scenes, agreed with the team first.

## Plain-language glossary

Each term is explained once, here.

| Term | Meaning |
|---|---|
| root / scene | One starting layout of objects on the shelf. A "root" is the saved record of a scene. |
| K (K1, K2, K3, K4) | How many moves the robot needs to solve the scene. K3 is hardest. "K ≥ 1" means at least one move. |
| NONE | A scene labelled unsolvable within the teacher's budget. |
| TRAIN / VAL / TEST | Three scene sets. TRAIN is for learning, VAL is for tuning and exploring, TEST must stay untouched until a final check. |
| exploratory vs confirmatory | Exploratory: we looked at results and chose what to test, so findings are only leads. Confirmatory: the test was fixed in advance, so findings count. |
| pre-registration | The test plan written down before seeing results, so we can't move the goalposts. |
| physics / teacher | The physics simulator that tells us whether a plan really works. It is the ground truth. |
| p11 | A model's predicted chance that a plan or placement succeeds. |
| cVAE | A neural network that proposes candidate placements. "Goal-aware" means it also sees the goal. |
| Ranker | A model that keeps the 4 most promising cells. |
| Boltzmann | A model that scores placements. Its "mean" averages 3 training seeds. |
| hazard (per-step hazard) | A model that scores each move of a plan for the chance of failing at that step. |
| factor graph (E8) | A success model that also tracks which objects block the target. |
| team pipeline | The team's planner: cVAE proposes, Ranker filters, Boltzmann scores. |
| beam search | Plan search that keeps only the best 16 partial plans at each step ("width 16"). |
| first try | Success of the first plan the planner commits to. |
| lazy retries / "within 5" | If a plan fails in physics, try the next-best plan, up to 5 plans per root. "Lazy" means only plans that are really needed are tried. |
| W–L | Paired wins and losses (for example 44–20): roots where arm A succeeded and B failed, versus the reverse. Roots where both agree are ignored. |
| discordant pairs | The roots where the two arms disagree. Only these carry information in a paired test. |
| sign test | A test on W–L only: if there were no real difference, wins and losses would split 50/50. |
| McNemar | The same test, under another name (the exact version), for paired yes/no outcomes. |
| p-value | How surprising the result would be if there were no real difference. Small means more surprising. |
| Holm correction | An adjustment that makes p-values stricter when many comparisons are run, so lucky hits are not over-claimed. |
| family-wise error | The chance of at least one false positive across a set of tests. |
| power | The chance a test detects a real effect. Low power means a null result tells us little. |
| Newcombe CI | A 95% confidence interval (plausible range) for a difference between two success rates. |
| α (length penalty) | A cost per move that stops the planner from adding needless moves. Bigger α means shorter plans. (In E9v2, α is also used for the 0.05 significance level; context makes clear which.) |
| AUROC | How well a score ranks successes above failures. 1.0 is perfect, 0.5 is chance. |
| Brier / ECE | Two measures of how good and how honest the probabilities are: whether a predicted 70% really succeeds about 70% of the time. Lower is better. |
| overconfident | Predicted success is higher than real success. |
| hit@4 / precision@4 | Whether, or how often, a good cell appears among the model's top 4. |
| isotonic calibration | A simple fit that maps raw scores to honest probabilities. |
| abstention | The planner declines to act when it expects to fail. |
| ablation | Removing or swapping one part to see what it was contributing. |
| winner's curse | A setting chosen because it won on some data tends to look better there than it really is. |

## Done (exploratory, VAL)

**What we asked / what we found:** Do the team's models pick good placements and plans? A simple Boltzmann scorer beat the cVAE and random picks. The team pipeline scored higher on plans (100 vs 88 of 160), but not significantly.

| # | Question | Result |
|---|---|---|
| E1 | Offline: does p11 rank labelled cells better than the cVAE? | Yes: hit@4 0.29 vs 0.08. Multiplying the two (hybrid v1) was worse than p11 alone. |
| E2 | Placement in physics, 30 contexts | Boltzmann #1 pick works 24/30, cVAE 17/30, random 15/30. |
| E3 | Placement, 90 contexts, 3 seeds, Ranker veto | Boltzmann (3-seed mean) 78/90; DIRECT-GEO 68; random 54; Ranker 51; cVAE 47. The Ranker veto never changed the #1 pick. |
| E4 | Whole-plan search, 160 roots | Boltzmann 88/160, DIRECT-GEO 73, Boltzmann objects + cVAE cells 48, random 40. Predicted p11 is the same for plans that work (0.972) and plans that fail (0.974). |
| E5 | Team pipeline (cVAE → Ranker → Boltzmann) | Plans 100/160 (vs Boltzmann over all cells 32 wins, 20 losses, p ≈ 0.13); placement 71/90. Failures move from move 1 to the target step. |
| E6 | Goal-aware cVAE (distilled from Boltzmann, β = 10) | Placement 73/90 vs old cVAE 47 (29 wins, 3 losses). Plan-level results pending (Phase 4). |

(Distilled: the cVAE is trained to copy the Boltzmann scorer, its "teacher". β sets how strongly it follows the teacher.)

## Phase 4: plan-level fixes (VAL, 160 roots with K ≥ 1; exploratory)

**What we asked / what we found:** Which planner solves the most scenes? The per-step hazard planner did best (124/160 on first try, 154/160 within 5). Its edge over the team pipeline was significant on first try (Holm p = 0.022), but there was no gain on the hardest scenes (K3).

Setup: physics; lazy retries, at most 5 plans per root; `runs/PHASE4/summary.json`.

| Planner | 1st try | Within 5 | Plans tried | Motion queries | Predicted vs real (1st) |
|---|---|---|---|---|---|
| Boltzmann mean | 88 (55%) | 131 (82%) | 2.11 | 5.1 | 0.97 / 0.55 |
| Cautious (min of 3 seeds) | 83 (52%) | 130 (81%) | 2.22 | 5.2 | 0.96 / 0.52 |
| Goal-aware cVAE pipeline | 103 (64%) | 137 (86%) | 1.97 | 5.8 | 0.96 / 0.64 |
| Factor graph (E8) | 110 (69%) | 150 (94%) | 1.67 | 4.7 | 0.99 / 0.69 |
| **Per-step hazard** | **124 (78%)** | **154 (96%)** | **1.48** | **4.1** | 0.96 / 0.78 |

- Reference rows (1st try only, no retries): team pipeline 100 (62.5%), no-structure 73, Boltzmann objects + cVAE cells 48, random 40.
- First-try paired sign tests, Holm-corrected over 7 comparisons:
  - per-step vs team pipeline: 44–20, Holm p = 0.022.
  - per-step vs Boltzmann: 49–13, p < 0.001.
  - per-step vs factor: 29–15, Holm p = 0.24.
  - All others not significant.
- By K (1st try), per-step vs team pipeline:
  - K1: 49/53 vs 38/53.
  - K2: 49/67 vs 35/67.
  - K3: 26/40 vs 27/40, so no gain on the hardest scenes.
- Timing per root (laptop CPU):
  - Planning: 0.4 s per-step, 0.8 s Boltzmann, 1.6 s goal-aware cVAE pipeline.
  - Physics: about 22–33 s per attempt.
- **Caveats:**
  - About half of the failed attempts happen after ≥ 1 move was executed. "Within 5" assumes a simulator reset, so it is optimistic for a real robot.
  - Every model is still overconfident under search.
  - The hazard model and the Boltzmann model were both early-stopped on these VAL roots.
- **Next:** a pre-registered confirmation of per-step vs team pipeline on fresh or TEST scenes (needs team agreement), and closed-loop replanning from the actual post-failure state.

## E9 (DRAFT → REJECTED by the ARS methodology review, 2026-11-21; superseded by E9v2): fresh-scene scale test

**What we asked / what we found:** Can per-step beat the team pipeline on brand-new scenes? This draft was rejected before it ran, because the review found it badly underpowered and confounded (reasons at the end). E9v2 replaces it.

- **Scenes:** never used by any earlier comparison. TEST stays untouched.
  - **F-solv:** 24 V4-VAL roots outside the Approach-B QC set, with labelled K2 or K3 and available locally. The 36 others exist only on the server.
  - **F-none:** 58 V4-VAL roots labelled NONE (unsolvable under the teacher's budget).
  - **F-K4:** 9 roots needing 4 moves (`raw_k4_ordered`, outside every split).
  - **Caveat:** F-solv and F-K4 were never QC-checked (quality-checked), so a root could contain an object that can never be picked.
- **Planners:** frozen, no retraining, no tuning.
  - team pipeline (cVAE → Ranker → Boltzmann);
  - Boltzmann mean;
  - per-step;
  - factor graph.
  - Search depth 3 on F-solv and F-none; depth 4 on F-K4, beyond the trained lengths.
- **H1 (primary):**
  - Hypothesis: per-step first-try success > team pipeline on F-solv ∪ F-K4 (n = 33).
  - Test: paired two-sided sign test, α = 0.05.
  - Power: VAL had about 40% discordant pairs, 69% of them in favour of per-step. With n = 33 that is about 13 discordant pairs, so **power is low (about 30–40%)**. Report the estimate and its CI; a null result is not evidence of no effect.
- **H2 (abstention):**
  - Each planner's score → isotonic calibration fitted on its 160 VAL first-try (score, success) pairs.
  - Decision rule: act iff p_cal ≥ λ/(1+λ), with λ = 1 primary and λ = 2 secondary. (p_cal is the calibrated success chance. λ is how many successes one failure costs.)
  - Metrics: abstain rate on F-none; act rate and success-when-acting on F-solv ∪ F-K4; utility (+1 success, −λ failure, 0 abstain).
  - Physics is run on F-none plans too, to check whether any "unsolvable" root is in fact solved.
- **Secondary:** lazy retries (≤ 5 plans), motion queries, planning time, and results by pool.
- **Review outcome:** both seats said Major Revision. Reasons E9 must not run as drafted:
  - **Power:** exact sign-test power is 12–19% at n = 33, not the stated 30–40%.
  - **Estimands mixed:** K4 roots at depth 4 are pooled with K2/K3.
  - **Unscreened roots:** they were not QC'd.
  - **Calibration:** fitted on solvable VAL roots, which mismatches the NONE base rate.
  - **Missing plan elements:** no stopping, failure or multiplicity plan; no length-penalised team arm; no label-source ablation.
  - **Unscoreable roots:** 32 of the 91 fresh roots exceed `MAXOBJ = 8`.

## E10: development arms requested by the ARS review (VAL, exploratory; fixed 2026-11-21 before any E10 physics)

**What we asked / what we found:** How much of per-step's lead is real, and how much is a handicap on the team arm? Most of the first-try lead came from the team pipeline lacking a length penalty. With the penalty (α* = 0.2), the arms look the same on first try. Per-step keeps an edge with retries, which comes from its move set.

- **Purpose:** remove the confounds the review found, and **freeze** each arm's settings for the confirmatory E9v2 on fresh scenes.
- **Common settings:** 160 VAL roots, the Phase-3 beam search (width 16, depth ≤ 3, cells ≠ goal), lazy retries ≤ 5. Primary metric: first-try success.

| Arm | What it is | What it isolates |
|---|---|---|
| `team_len_a{α}` | Team pipeline (same replayed search) with Algorithm 1's length penalty: choose argmax over all scored plans of log p11 − α·\|plan\| | Over-planning. **α grid fixed in advance: {0.05, 0.1, 0.2, 0.5, 1.0}.** α* = best first-try success on VAL (ties → smaller α); α* is frozen for E9v2 and only α* gets lazy retries. |
| `step_team_props` | Per-step scorer, but moves limited to the team's cVAE+Ranker 4 proposals (fresh generator, seed 0) | Scorer vs move set |
| `boltz_ranker_filter` | Boltzmann mean, but each move must have Ranker p ≥ 0.5 (3-seed mean; if no cell passes, keep the single best) | The cheap "move-feasibility filter in front of Boltzmann" baseline. Learned, not IK; an IK filter is not built. |
| `boltz_B` | Boltzmann D retrained on the same `labels.B` records as per-step (`hybrid/src/sidecar_from_b.py`); added when trained | Label source vs structure |

(IK = inverse kinematics, the geometric check of whether the robot arm can reach a pose.)

- **Amendment (2026-11-21, before any E10 physics):** `boltz_B` cannot be trained.
  - In chained records a failed move means the target is never tried, so the X0Y0 cell is 0 by construction (train cells: X1Y1 1973, X1Y0 615, X0Y1 25, X0Y0 0).
  - The team's trainer refuses it (`TRAINING_BLOCKED_INSUFFICIENT_PAIRED_EVIDENCE`). Its gate is not bypassed.
  - **Finding:** the X/Y model's identifiability depends on the teleport/replay labelling protocol.
  - **Replacement:** `outcome_B`, the same network and data as `hazard`, supervised only on whole-plan success (`hybrid/src/outcome_b.py`). `outcome_B` vs `hazard` isolates the value of step-level labels.
- **Result: length penalty** (first try, of 160; ties → smaller α).

  | α | Works | vs α = 0 (W–L) |
  |---|---|---|
  | 0 | 100 | – |
  | 0.05 | 110 | 10–0 |
  | 0.1 | 114 | 15–1 |
  | **0.2** | **119** | 21–2 |
  | 0.5 | 113 | 21–8 |
  | 1.0 | 79 | 21–42 (under-plans K3: 8/40) |

  - **α* = 0.2, FROZEN for E9v2.**
  - Implication: most of the earlier per-step "+15" over the team pipeline came from the missing length penalty. Per-step (124) vs team with α* (119) is about +5, and α* was tuned on these roots, so it is biased in the team arm's favour.
- **Result: the other arms** (160 roots; lazy retries ≤ 5; W–L = paired wins–losses, exact sign test, uncorrected).

  | Arm | First try | Within 5 | Motion queries / root |
  |---|---|---|---|
  | per-step `hazard` (Phase 4) | 124 | 154 | 4.1 |
  | `outcome_B` (same net and data, whole-plan labels only) | 120 | 151 | 4.6 |
  | `step_team_props` (per-step scorer, team's 4 proposals) | 122 | 143 | 5.3 |
  | `team_len_a0.2` (team pipeline, α*) | 119 | 141 | 5.5 |
  | `boltz_ranker_filter` | 107 | 145 | 5.0 |

  | Pair | First try W–L (p) | Within 5 W–L (p) |
  |---|---|---|
  | hazard vs team α* | 29–24 (0.58) | **18–5 (0.011)** |
  | hazard vs outcome_B | 19–15 (0.61) | 8–5 (0.58) |
  | outcome_B vs team α* | 26–25 (1.0) | 17–7 (0.064) |
  | step_team_props vs boltz_ranker_filter | 39–24 (0.077) | 13–15 (0.85) |

  - **Reading.**
    - On first try, once the team pipeline has α*, the arms can't be told apart (hazard vs team α* 29–24).
    - Of the 24-root gap between per-step (124) and the team pipeline without α (100), the length penalty accounts for 19. `outcome_B` adds 1 more and step-level labels 4 (hazard vs outcome_B 19–15, p = 0.61). `outcome_B` vs team also differs in network and move set, so it does not isolate the label source.
    - **The retry edge comes from the move set, not the scorer.** Same per-step scorer, all free cells vs the team's 4 proposals: hazard vs `step_team_props`, within 5: **16–5 (p = 0.027)**. Per-step scorer on the team's proposals vs team α*: 13–11 (p = 0.84). On first try neither matters (25–23 and 20–17).
    - With verified retries, the hazard *planner* (per-step scorer plus search over all free cells) keeps an edge over team α* (154 vs 141). This was the smallest p of the 8 E10 tests (×8 ≈ 0.09), so it is a lead, not a finding. It becomes E9v2's primary endpoint because it is the only difference large enough to test with power.
  - Files: `runs/PHASE4/summary_E10_three_way.json`, `summary_E10_outcomeB_teamlen02.json`, `summary_E10_steprops_rankerfilter.json`.
- **No significance claims from E10.** It is development data; E9v2 carries the confirmatory test.

## E11: Does averaging the three success models pick better plans? (development, VAL; written 2026-11-21 before any E11 physics)

- **What we ask:** averaging models often ranks better than any one model, because their mistakes partly cancel out. Does an equal-weight average of `hazard`, `factor` and `outcome_B` (as the search scorer) beat `hazard` alone?
- **Setup:** identical to the `hazard` arm (all free cells, beam search, lazy retries ≤ 5). Only the scorer changes ([src/e11_ensemble.py](../src/e11_ensemble.py)).
- **Read-out:** first try and within 5 tries vs `hazard` (paired W–L, exact sign test), plus motion queries per root.
- **Status:** VAL development only, one more comparison on reused roots, so the result is a lead at most. It does **not** change E9v2's frozen arms.
- **Result (160 roots):**

  | Planner | First try | Within 5 | Motion queries / root |
  |---|---|---|---|
  | **`ens3` (average of 3 models)** | **130** | **158** | 4.1 |
  | `hazard` | 124 | 154 | 4.1 |
  | `outcome_B` | 120 | 151 | 4.6 |
  | `team_len_a0.2` | 119 | 141 | 5.5 |
  | `factor` | 110 | 150 | 4.7 |

  | ens3 vs | First try W–L (p) | Within 5 W–L (p) |
  |---|---|---|
  | hazard | 21–15 (0.41) | 6–2 (0.29) |
  | team α* | 29–18 (0.14) | 17–0 (< 0.001) |
  | outcome_B | 26–16 (0.16) | 8–1 (0.039) |
  | factor | 28–8 (0.001) | 10–2 (0.039) |

  - **Reading:**
    - `ens3` has the highest point estimates, but there is **no detectable difference from `hazard`** (21–15, 6–2). It is clearly ahead only of the weaker arms (team α*, outcome_B, factor).
    - Most of its retry gain is on K2/K3 (within 5: K2 100%, K3 97.5%).
    - Motion queries are the same as `hazard` (4.1), so there is no efficiency gain. Mean plans tried fell slightly, 1.48 → 1.38.
    - Predicted vs real on first picks: 0.97 vs 0.81. Still overconfident, but the gap (0.16) is the smallest of the arms (others ≥ 0.19).
  - **Disclosure:** all three component models were early-stopped on these VAL roots, as every learned arm was. The ensemble was chosen after seeing E10, with equal weights fixed in advance and no weight search. The `hazard` row is the existing Phase-4 run (shared physics cache), not a re-run.
  - Uncorrected p-values on reused VAL roots, so this is a lead. To test it properly it would need its own pre-registered fresh-scene test (it is not an E9v2 arm). File: `runs/PHASE4/summary_E11_ens3.json`.

## E9v2: confirmatory test on fresh scenes (PRE-REGISTERED 2026-11-21, before any fresh scene was scored by any arm)

**What we asked / what we found:** On brand-new scenes, does the hazard planner beat the team pipeline (with length penalty) in success within 5 tries? No result is recorded in this log; this entry is the plan.

- **Question.** On scenes no model or setting has seen, does the **hazard planner** (per-step success model plus beam search over all free cells) beat the team pipeline with its length penalty (`team_len_a0.2`)? A win credits the planner as a whole. E10 suggests most of it comes from the move set.
- **Disclosed change of endpoint.** The rejected E9 draft used first-try success as primary. After seeing E10 (where first try showed no detectable difference), the primary moved to success within 5 tries. First try is the robot-relevant number, and it was demoted only because it has too little power. It stays as H2.
- **Scenes.**
  - New roots from the team's unchanged generator, run as `--name hx1` (seeds cannot collide with existing data; see `gen/README_homeserver.md` §0).
  - Quota K1/K2/K3 = 105/132/80, a VAL-like mix with about 5% spare.
  - Usable = `gen/qc_new_roots.py` passes: class 1–3, all fields present, ≤ 7 movable objects (the encoder's MAXOBJ = 8), team fixed-object QC (`obj_pickable3.py`), and no duplicate layout.
  - **Freeze before scoring.** When generation stops, the list of usable root ids is written, its sha256 recorded here and committed to git. Every usable root is used, including any generated after the quota was met. No root is dropped after any arm has been scored on it.
  - **Stopping rule.** The quota counts QC-usable roots, not merely accepted ones. Generation stops when the quota is met. There is no calendar deadline, because the server is power-limited and restarts need the user's go-ahead.
  - **Confirmatory floor:** 210 usable roots. Below that, E9v2 is reported as descriptive only.
  - The K mix matters: on VAL the retry edge sat in K2/K3 (W–L K1 1–1, K2 10–2, K3 7–2), so K3 must not be cut short.
- **Frozen arms** (models, code and settings hashed in [E9v2_FREEZE.sha256](E9v2_FREEZE.sha256)):
  - **A = `hazard`:** 3-seed mean, Phase-3 beam search (width 16, depth ≤ 3, cells ≠ goal).
  - **B = `team_len_a0.2`:** team cVAE → Ranker → Boltzmann replayed exactly as in phase3b, with argmax log p11 − 0.2·|plan|.
  - **Both arms:** lazy verified retries ≤ 5, pruning plans that share a failing prefix. Same teacher (BD3DS66_TEACHER_V2), same physics cache.
  - **E9v2 runner.** The current pipeline is wired to VAL: it replays the team generator over the 90 VAL contexts, reads VAL picks, and shares one generator stream across roots. Before the root list is frozen, a dedicated runner is written that:
    - reads hx1 roots into a fresh output folder;
    - seeds the team generator per root from sha256(root_id), so results don't depend on root order;
    - leaves the scoring and search code unchanged.

    It is smoke-tested on 5 VAL roots, then hashed into the freeze file. Any other code change after that is a deviation, logged here with its reason.
  - **Freeze file** also covers: encoder and model code (`data3d.py`, `models.py`, `xy3d*.py`), sampler config, the teacher and generator code, QC script, OUTCOME_B checkpoints, and package versions (`E9v2_pip_freeze.txt`).
- **Endpoints and tests.**
  - **H1 (primary): success within 5 verified tries.** Paired by root; exact two-sided sign test on discordant roots (= exact McNemar) at α = 0.05. Report the paired difference with a 95% Newcombe CI.
  - **H2 (secondary, fixed sequence): first-try success.** Same test. Tested only if H1 is significant, so the family-wise error stays ≤ 0.05 without correction.
  - **Descriptive only, no tests:**
    - results by K;
    - plans executed and motion-planner queries per root;
    - planning time;
    - predicted vs real success;
    - `step_team_props` (same scorer, team's 4 proposals): shows how much of any H1 win is the move set;
    - **no-reset within 5:** a retry counts only if the failed attempt executed no move. It needs no extra physics, and shows how much each arm relies on the simulator reset;
    - `outcome_B` and `factor`, run as extra arms if time allows. They are not part of the family.
- **Power (from VAL, an assumption).**
  - VAL discordance on H1 was 23/160 (14%), split 18–5. At n = 300 that means about 43 discordant roots.
  - The exact sign test's power is then 0.96 at a true 78/22 split, 0.71 at 70/30 and 0.44 at 65/35.
  - At n = 210 (about 30 discordant) these become 0.90, 0.59 and 0.36.
  - VAL's split is the optimistic case, since the winner's curse applies.
  - For H2 (VAL 29–24) power is low, and a null H2 is expected and uninformative.
- **Decision rule.**
  - The exact sign test decides; the Newcombe CI is reported alongside.
  - **Confirmed:** H1 p < 0.05 with hazard ahead → "the hazard planner improves verified-retry success over the team pipeline on fresh scenes."
  - **Reversed:** H1 p < 0.05 with the team ahead → reported as a confirmed win for the team pipeline.
  - **Not confirmed:** otherwise, reported as such. No re-analysis with other endpoints is presented as confirmatory.
- **Failures and exclusions.**
  - An arm that crashes, times out or returns no plan (`NO_PLAN`) on a root counts as a failure for that arm on that root.
  - A **teacher error** is the physics teacher raising an exception rather than returning a verdict. That attempt is re-run once. If it errors again, the root is excluded from **both** arms, even if only one arm's plan triggered it. Exclusions are listed with their reasons.
- **Known limitation, declared now.** Retries assume a simulator reset. On VAL, 90% of team failures and 51% of per-step failures happened after a move had been executed, so a real robot would not get these retries for free. H2 (first try) is the robot-relevant number.
- **Not in scope:** TEST, abstention (H3 is deferred until calibration is fitted on TRAIN search picks), REAP re-runs.

## E7: Does teacher error limit the goal-aware cVAE? (pre-registered 2026-11-20, before any E7 result)

**What we asked / what we found:** Does a student cVAE do better if its teacher makes fewer mistakes? Only partly. Real labels instead of teacher guesses gave the largest gain (81 vs 72 of 90), but it was not significant after Holm correction. Trusting the teacher less (β = 1) was clearly worse.

- **Hypothesis:** the student copies its teacher's mistakes, so teachers with fewer errors give students with higher physical success.
- **Variants:** all fixed now, all trained on TRAIN contexts only, 3 seeds each, same architecture and initialisation as E6 unless stated.

  | Variant | Teacher | β |
  |---|---|---|
  | base | Boltzmann 3-seed mean (= E6) | 10 |
  | beta1 | Boltzmann 3-seed mean | 1 (exact Bayes) |
  | pessimistic | min over the 3 Boltzmann seeds | 10 |
  | hazard | per-step hazard model, 3-seed mean | 10 |
  | agree | geometric mean of Boltzmann-mean and hazard | 10 |
  | anchored | Boltzmann mean, but cells with a real TRAIN label use that label (success → 0.95, failure → 0.05) | 10 |
  | nolatent | base teacher, goal-aware categorical model (no latent) | 10 |

- **Primary metric:** #1 pick works in physics on Phase 3's 90 placement contexts. Paired sign tests against `base`, with Holm correction across the 6 comparisons.
- **Secondary:** offline hit@4 and precision@4 on VAL labels.
- **Expected if the hypothesis holds:** pessimistic, agree and anchored ≥ base. If nolatent ≈ base, the latent is not needed.
- **Caveat:** VAL roots again, so the result stays exploratory.
- **Result** (physics, #1 cell works, of 90; `runs/GCVAE_ABL/physics_summary.json`):

  | Variant | Works | vs base (W–L) | Raw p | Holm p |
  |---|---|---|---|---|
  | base | 72 | – | – | – |
  | beta1 | 60 | 2–14 | 0.004 | 0.025 |
  | pessimistic | 74 | 7–5 | 0.77 | 1.0 |
  | hazard | 77 | 14–9 | 0.40 | 1.0 |
  | agree | 77 | 12–7 | 0.36 | 1.0 |
  | anchored | 81 | 12–3 | 0.035 | 0.18 |
  | nolatent | 73 | 1–0 | 1.0 | 1.0 |

  - Reference: the Boltzmann teacher's own #1 pick works 78/90.
- **Interpretation:**
  - The hypothesis is only partly supported. Real labels in place of teacher guesses gave the largest gain (81 vs 72), which exceeds the teacher itself, but it is not significant after Holm correction.
  - A better teacher (hazard / agree) gives +5, not significant. Pessimism changes nothing.
  - Trusting the teacher less (β = 1) is clearly worse. The student must follow the teacher strongly.
  - The latent variable is irrelevant: nolatent = base in 89/90 contexts, matching report 09's near-collapse finding.
- **Next (post-hoc, label it as such):**
  - anchored + hazard teacher, combined.
  - Any confirmation needs fresh scenes.

## E8: Factor-graph success model (started 2026-11-20)

**What we asked / what we found:** Does adding "which object blocks the target" to the success model help? Offline, yes: it ranked plans best (AUROC 0.937) and was best calibrated. It passed the gate and was added to Phase 4 as planner `factor`. There it reached 110/160 on first try, between the cVAE pipeline and per-step.

- **Hypothesis:** adding per-object "blocks the target" variables and per-step move factors to the X/Y model improves plan ranking, and gives "which object" guidance to the search.
- **Design:**
  - Model: per-step move factors (hazard-style) × a noisy-AND target factor over per-object "not blocking" variables, with the X/Y coupling kept. (Noisy-AND: the target is reachable only if every object is not blocking, each with its own chance.)
  - Trained on TRAIN executed-plan records (`labels.B`). Blocker variables are weakly supervised by TRAIN `minimal_sets`.
  - Offline: plan AUROC and calibration on VAL `labels.B`, blocker accuracy on VAL `minimal_sets`, compared with Boltzmann and hazard.
  - Physics: added as a Phase-4 planner (same search, lazy retries) only if it is not worse offline than hazard.
- **Offline result (VAL, exploratory):**

  | Model | Plan AUROC | Brier | ECE |
  |---|---|---|---|
  | factor | 0.937 | 0.097 | 0.017 |
  | hazard | 0.926 | 0.105 | 0.047 |
  | Boltzmann | 0.796 | 0.169 | 0.120 |

  - Blocker AUROC 0.898; the top-ranked blocker is in a minimal set in 92% of K≥1 roots.
  - Gate passed: added as Phase-4 planner `factor`.
