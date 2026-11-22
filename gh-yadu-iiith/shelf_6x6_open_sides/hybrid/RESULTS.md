# Results

This file covers two phases:
- **Phase 2 (current, 22 Nov 2026):** a per-step plan scorer inside the team's V3.4 pipeline, with a pre-registered TEST result. See section 0. The full report is [reports/plan_scorer_report_2026-11-22.pdf](reports/plan_scorer_report_2026-11-22.pdf).
- **Phase 1 (20–21 Nov 2026):** design changes to the older Approach B pipeline, on VAL only, all exploratory. Sections 1–9 are kept as written then. Phase 2 supersedes their headline.

---

## 0. Phase 2: per-step scorer in the V3.4 pipeline (22 Nov 2026)

**What changed.** V3.4 works like this:
1. The task-conditioned cVAE proposes 4 spots per blocker.
2. Every plan of 1 to 3 moves is listed.
3. A scorer rates each plan.
4. A rule picks one: do nothing if the empty plan scores at least 0.5; otherwise run the best plan if it scores at least eta = 0.3; otherwise give up.

Only the scorer in step 3 was replaced. XYNet, which scores the whole plan at once, became a per-step model, `src/twod_hazard.py`. That model scores each move, then "reach target" and "pick target", and multiplies the step scores. The candidate plans are V3.4's own, so every difference comes from re-scoring the same plans. Related work that also multiplies per-step success: STAP, Text2Motion. Related work that scores whole plans jointly: PIGINet.

**Confirmatory result** (pre-registration [docs/HZ_V34_PREREG.md](docs/HZ_V34_PREREG.md), frozen in commit 8e18429; outputs in [results_hz_v34_test/](results_hz_v34_test/)). Pass rule: task lower bound > 0, and do nothing / give up lower bounds > −10. Bounds are one-sided 97.5%, world-cluster bootstrap. Values are 3-seed means.

| TEST | Task | Do nothing | Give up | Verdict |
|---|---|---|---|---|
| 3D, ours vs XYNet | 90.5 vs 80.8 (lb +5.9) | 93.5 vs 86.1 (lb +0.5) | 90.3 vs 86.0 (lb −4.2) | **Supported** |
| 2D, ours vs XYNet | 74.4 vs 51.8 (lb +18.9) | 95.0 vs 91.1 (lb +0.6) | 84.2 vs 92.0 (lb −12.0) | **Not supported** (give up) |

The laptop and lab-server physics checks agree on 680 of 701 3D TEST runs (97%) and on all 1,184 in 2D.

**Where the gain comes from** (3D VAL, exploratory; 3-seed means unless marked):

| Version | Task | Do nothing | Give up |
|---|---|---|---|
| Ours (per-step, our encoder, step labels) | 90.8 | 91.4 | 94.9 |
| Same, without "which step failed" labels | 90.9 | 86.3 | 96.6 |
| Same, with XYNet's encoder | 87.0 | 88.2 | 89.7 |
| XYNet (team) | 78.5 | 83.9 | 78.7 |
| PIGINet-style whole-plan transformer (our reimplementation; seed 0, seeds 1–2 running) | 71.8 | 84.7 | 60.3 |

Reading:
- Scoring per step, instead of the whole plan, gives the task gain.
- About two-thirds of that gain remains with XYNet's encoder (87.0 vs 78.5); about a third needs our encoder.
- The step labels do not change task success. They add about 5 points on "do nothing", and they are ahead on all three seeds.
- XYNet's seeds differ by about 1 point on task.

**Exploratory follow-ups** (post-hoc; the TEST ones reuse the already-scored TEST set):
- **Equal give-up rate** (`explore/curves.py`). At equal give-up accuracy, ours solves about 10 more task points in 3D and about 25 in 2D.
- **E-M4M on 3D TEST** (team code, 300 CPU-s limit; our re-run matched the team's plans on 6/6 VAL check scenes; `explore/vs_classical.py ... test`):

  | | Task | Do nothing | Give up |
  |---|---|---|---|
  | Ours vs E-M4M | 90.5 vs 70.1 | 93.5 vs 97.0 | 90.3 vs 100 |
  | Difference (lb) | +20.3 (+13.6) | −3.5 (−11.1) | −9.7 (−16.1) |

  With VAL-chosen thresholds, ours gives 87.8 / 95.5 / 98.9. Median time per scene: E-M4M 109 s; ours about 0.1 s for scoring only, measured on a different machine.
- **Never-seen worlds.** 134 of 188 TEST worlds also appear in training. On the 55 task scenes from never-seen worlds, the lead is +5.5 (lb −2.5, not significant), against +10.9 on seen worlds. TEST shows new arrangements of mostly seen worlds.
- **Replays of the chosen TEST plans** (`../hardware/robust/`):

  | Setting | Ours | XYNet |
  |---|---|---|
  | G2 gripper | 91.5 | 83.1 |
  | Straight-in approach only | 9.7 | 10.4 |
  | Objects shifted 5 mm | 90.0 | 79.5 |
  | Objects shifted 10 mm | 83.0 | 75.5 |
  | Board 10 cm lower | 48.5 | 42.4 |

  About 90% of shift failures are "no feasible side grasp".
- **Calibration error** on 3D TEST: 6.4 vs 9.8 points (no confidence interval).
- **4 to 6 moves:** 1 of 15 scenes solved. The claim is limited to up to 3 moves.

**Caveats:**
- No real-robot result.
- Thresholds tuned after TEST was used are exploratory.
- Timings come from different machines.
- Several checks are single-seed or single-run.
- The full weak-points table is in the report.

**Where things are:**
- Analysis scripts: `explore/`, with outputs in `../approachb_3d_learning/runs/explore/`.
- Hardware checks: `../hardware/`.
- Running 22–23 Nov: noise-aware training labels (`explore/noise_labels/`), grasp check before acting (`explore/robust_grasp_check/`), a position-noise curve.

---

# Phase 1: design changes to the learned relocation planner, and what they do to the workflow

**Period:** 2026-11-20 to 21. **Machine:** laptop CPU (8 cores).

**Ground truth:** the dataset's own physics teacher (`BD3DS66_TEACHER_V2`: xArm7, RRT-Connect with 5,000 iterations, MuJoCo, moves chained).

**Scenes:**
- All results use Approach-B **VAL** roots: 90 placement contexts and 160 roots with verified K ≥ 1 (53 K1, 67 K2, 40 K3).
- **TEST was never read.**
- These VAL roots were reused for about 25 comparisons, so **every result here is exploratory.** See §7 and the per-experiment log in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

---

## 1. Headline

| | Team pipeline (baseline) | Best change: per-step success model | Difference |
|---|---|---|---|
| Plan works on the first try (160 roots) | 100/160 (62.5%) | **124/160 (77.5%)** | +15 points; 44 wins vs 20 losses, Holm-corrected p = 0.022 |
| Plan works within ≤ 5 verified tries (simulation) | 137/160 (86%) | **154/160 (96%)** | 22 wins vs 5 losses, p = 0.002 |
| Plans executed per root | 2.0 | **1.5** | |
| Motion-planner queries per root | 6.6 | **4.1** | |
| Physics time per root | 58 s | **43 s** | |
| Planning (model) time per root | ~1 s | **0.4 s** | |

**The three findings that drive everything else:**
1. **Placement has to be goal-aware.** The goal-blind cVAE is the weakest component in every test.
2. **Predicting *which step* fails is the most valuable missing information.** A per-step model beats the two-variable X/Y Boltzmann model as a plan scorer.
3. **Every learned scorer is overconfident on the plans the search picks** (it says 95–99%; reality is 55–78%). This is the optimizer's curse. Recalibration plus a cost-based threshold fixes the decision, but only for scorers that still rank the plans they pick (per-step and factor; not the Boltzmann model).

---

## 2. The baseline: the team's current design (what we started from)

```
OFFLINE
  placement queries (VERIFIED only) ──► cVAE   p(d | scene, object)          goal-blind by design (execution contract)
  labelled placements (VERIFIED/NEG) ─► Ranker p(feasible | scene, object, d) goal-blind
  X/Y sidecar (7,656 X / 7,072 Y) ───► Boltzmann X/Y:  P(Y)=σ(b),  P(X|Y)=σ(a + w·y),  p11 = σ(b)·σ(a+w)
ONLINE (Algorithm 1)
  scene ─► cVAE samples cells ─(Ranker keeps 4)─► extend partial repairs ─► score each with p11
        ─► cost C = −log p11 + α|r| + βD + γĴ ─► best plan with p11 ≥ η, else ABSTAIN ─► execute once ─► verify
```

**Properties of this design that matter for the results:**
- **The cVAE never sees the goal and never sees a failure.** It only learns "where do placements usually succeed".
- **The cVAE's latent is near-collapsed.** Report 09 found 0–2 of its 8 latent units active, so it behaves like a plain 36-cell classifier. E7 confirms this: removing the latent changed nothing.
- **The Boltzmann model sees two bits per plan.** Did the moves work (X)? Did the target get out (Y)? Its labels come from a slightly different protocol than deployment: X adds a return-to-ready connector, and Y teleports objects and starts the target from the ready pose.
- **There is no feedback.** Failures, the failing step, and the Boltzmann model's goal knowledge never reach the proposal model.
- **η, α, β and γ have no recorded selection procedure.** They were not found on any branch.

---

## 3. Design changes, one by one

Each change lists: **what changed → why (the evidence that motivated it) → effect on the workflow → result.**

### D1. Make placement goal-aware

- **Why:** wherever the goal-blind cVAE chooses cells, it fails on hard scenes.

  | Test | Goal-aware | Goal-blind |
  |---|---|---|
  | Placement, #1 cell works (90 contexts) | Boltzmann 78/90 | cVAE 47/90 (random 54/90) |
  | Whole plans, same objects and order, cells from... | Boltzmann 88/160 | cVAE 48/160 |
  | ...on K3 roots only | 23/40 | **1/40** |

  The cVAE parks objects where they block the target again.

- **What:** two ways to get goal awareness.
  - **(a) Score every free cell with a goal-aware success model.** Simple and the most accurate for single placements. But it lets the search wander into unfamiliar cells (see D5).
  - **(b) Goal-aware cVAE ("hybrid v2")**, the principled version of "cVAE × Boltzmann":
    - A goal-marker row is added to the scene set.
    - It is fitted to the Bayes target `w(cell) ∝ q_old(cell) · odds_success(cell)^β`, with β = 10, from the Boltzmann teacher. This follows CbAS (Brookes et al. 2019) and advantage-weighted regression.
    - Trained on TRAIN contexts only, in 162 s on CPU.
  - **(c) E7 mitigation of teacher error:** where a TRAIN cell has a real recorded outcome, use the label instead of the teacher's guess ("anchored").

- **Results** (placement, #1 cell works of 90):

  | Model | Works | vs old cVAE (W–L) |
  |---|---|---|
  | Old cVAE | 47 | |
  | Goal-aware cVAE | 72–73 | 29–3 |
  | Anchored | 81 | |
  | Teacher (Boltzmann) | 78 | |

  - The anchored student beats its own teacher, 12 wins vs 3 losses against base. That is p = 0.035 raw, not significant after Holm correction.
  - Trusting the teacher less (β = 1) is clearly worse: 60/90, Holm p = 0.025.
  - On whole plans, the goal-aware cVAE pipeline gives 103/160, the same as the team pipeline (100/160).

- **Workflow change:**
  - The proposal model now depends on a trained success model (a teacher) and on the goal.
  - A distillation step is added after the success model is trained.
  - This **breaks the team's "goal-blind sampler" contract**, so it must be agreed before adoption.
  - Placement-level gains do not carry over to whole plans, so on its own D1(b) is **not** the main lever.

### D2. Replace the two-bit X/Y outcome with a per-step success model (the main gain)

- **Why:**
  - **Failure analysis:** of the Boltzmann planner's first-try failures, 50 of 72 fail at **move 1** ("the arm has no feasible grasp or path"), not at the target.
  - **Data audit:** the dataset already records **which step failed and why** for every executed plan (`labels.B[*].failed_step`). The X/Y model throws this away.
  - **Literature:** process (step-level) verifiers beat outcome verifiers as plans get longer (Lightman et al., ICLR 2024). Discrete-time hazard models give the loss directly (Gensheimer & Narasimhan 2019).

- **What** ([src/hazard.py](src/hazard.py)):
  - `p(plan) = Π_k P(move k ok | scene before move k) · P(target ok | final scene)`.
  - **Encoder:** the team's SceneEncoder, applied to the scene before and after each step.
  - **Loss:** plain BCE, so the per-step probabilities stay calibrated enough to multiply.
  - **Data:** trained on TRAIN executed records only.

    | Step | Positives / negatives |
    |---|---|
    | move 1 | 2,314 / 510 |
    | move 2 | 1,646 / 69 |
    | move 3+ | 879 / 56 |
    | target | 2,333 / 617 |

  - **Training:** 3 seeds, about 40 s in total on CPU.

- **Offline** (VAL executed plans): AUROC **0.926** vs Boltzmann 0.796. Brier 0.105 vs 0.169.

- **Physics** (160 roots), against the team pipeline:

  | | Per-step | Team pipeline |
  |---|---|---|
  | First try | **124 (77.5%)** | 100 |
  | Within 5 tries | 154 | 137 |
  | K1, first try | 49/53 | 38/53 |
  | K2, first try | 49/67 | 35/67 |
  | K3, first try | 26/40 | 27/40 |

  - The gain is on K1 and K2. **No gain on K3.**
  - It also stops over-planning: on one-move scenes it usually plans one move, while the Boltzmann model planned 2–3 moves in 41 of 53.

- **Workflow change:**
  - **Training labels change** from the sidecar's X/Y protocol to **executed, chained records**, which match deployment semantics.
  - **Partial plans become scoreable** ("first two moves will work"), so the search can prune bad branches early.
  - **No sidecar is needed**, and retraining takes seconds.
  - **The model reports where a plan is expected to fail.** That gives an explanation, and a target for lazy checking (D4).

### D3. Factor graph: per-step chain + per-object blocker variables + X/Y-style coupling

- **Why:**
  - Choosing which obstacles to clear is a minimum-set-cover / minimum-constraint-removal problem (Hauser 2014; Kaznachey & Jagota 1997).
  - The team's notes ask for factor graphs.
  - Per-object "is it in the way?" variables make the target factor interpretable.

- **What** ([src/factor.py](src/factor.py)):
  - The hazard move factors.
  - A target factor `σ(c0) · Π_o σ(g_o)`: a noisy-AND over "object o does not block". It is exact, because the blockers are independent given the scene.
  - `c0` also takes the plan's hardest move. That keeps X and Y coupled, as in the Boltzmann model.
  - Blocker variables are weakly supervised by TRAIN `minimal_sets` (loss weight 0.3, fixed in advance).

- **Offline:**
  - Plan AUROC 0.937, Brier 0.097, ECE **0.017**: the best calibrated of all models.
  - The top-ranked blocker is in a true minimal set in 92% of roots.

- **Physics:**
  - First try 110/160. That is worse than per-step: 15 wins vs 29 losses, Holm p = 0.24, not significant.
  - Within 5 tries 150/160, equal to per-step (4 wins vs 8 losses, n.s.).
  - On K3 with retries: 90% vs per-step 92.5%.

- **Workflow change:**
  - Same data and training cost as D2.
  - Adds an explanation output ("o02 is blocking, p = 0.9") and a natural place to add more variables later.
  - Better offline scores did **not** turn into better first-try plans. Offline metrics alone are not enough to choose a planner.

### D4. Lazy verified retries (use the physics oracle sparingly)

- **Why:**
  - REAP (Ayush, 24 h) reaches 82.2% using about 51 teacher checks per scene.
  - E-M4M and Lazy Rearrangement verify every move.
  - Our learned planners use **zero** checks while planning. The fair comparison is success against the number of checks.
  - LazySP (Dellin & Srinivasa 2016; Haghtalab et al. 2018) is worst-case optimal in the number of expensive checks.

- **What** ([src/phase4.py](src/phase4.py) `lazy`):
  1. Execute the top-ranked plan.
  2. If it fails at move j, drop **every** plan that shares its first j+1 moves, since chained execution is deterministic and they would fail identically.
  3. If only the target fails, drop that exact plan.
  4. Execute the best remaining plan, at most 5 plans per root.

- **Results** (first try → within 5 tries; plans executed; motion queries):

  | Planner | First try | Within 5 | Plans | Queries |
  |---|---|---|---|---|
  | Boltzmann | 55% | 82% | 2.11 | 5.1 |
  | Team pipeline | 62.5% | 86% | 2.0 | 6.6 |
  | Goal-aware pipeline | 64% | 86% | 1.97 | 5.8 |
  | Factor graph | 69% | 94% | 1.67 | 4.7 |
  | **Per-step** | **77.5%** | **96%** | **1.48** | **4.1** |

- **Workflow change:**
  - Execution becomes a loop: plan, execute, read the failing step, prune, re-rank.
  - **Caveat:** about half of all failed attempts happen *after* at least one move was executed. In simulation we reset; on a real robot you cannot. The "within 5" numbers are therefore **simulation-optimistic**, and **first-try numbers are the honest real-robot figures.**
  - The real-robot version re-plans from the current state (a closed loop like Ren & Qureshi 2024, but with look-ahead). This is not done yet.

### D5. Calibration and a cost-based abstain threshold (instead of an arbitrary η)

- **Why:** every planner states 95–99% confidence in the plan it picks, but succeeds 55–78% of the time.
  - **Cause:** the search takes the maximum over thousands of noisy scores, so the winner's score includes a lucky upward error. This is the optimizer's curse (Smith & Winkler 2006).
  - **Simulation** with a fair model (error ±0.1): picking the best of 10 plans overstates success by 11 points; the best of 5,000 overstates it by 32 points.
  - **Unfamiliar plans** found by the search, and the label-protocol mismatch, make it worse.

- **Can the score separate good picks from bad?** AUROC among the chosen first plans:

  | Planner | AUROC |
  |---|---|
  | Boltzmann | **0.50** (no information) |
  | Team pipeline | 0.57 |
  | Goal-aware pipeline | 0.60 |
  | Per-step | 0.78 |
  | Factor graph | 0.79 |

- **What** ([src/threshold.py](src/threshold.py)):
  - Utility per root: +1 for a success, −λ for a failed attempt, 0 for abstaining. λ is the cost of a failed attempt.
  - Choose η by 5-fold cross-validation over roots.
  - **Or**, with no tuning: recalibrate the score with isotonic regression, then act iff `p_cal ≥ λ/(1+λ)`.

- **Results** (utility; acts on N of 160):

  | Planner | λ = 2, always act | λ = 2, CV η | λ = 3, always act | λ = 3, CV η |
  |---|---|---|---|---|
  | Boltzmann | −56 | 0 (acts on 0) | −128 | 0 |
  | Team pipeline | −20 | 3 (72) | −80 | −11 (9) |
  | Per-step | 52 | **61** (106) | 16 | **46** (98) |
  | Factor graph | 10 | **53** (86) | −40 | **39** (79) |

  - For the Boltzmann model, **no** η helps: its best policy is to abstain everywhere. So an arbitrary η is as good as any.
  - For per-step and factor, the calibrated rule matches the tuned η, and when it acts, 86–87% of attempts succeed.
  - The raw "optimal" η values (0.945–0.992) are model-specific and do not transfer between models.

- **Workflow change:**
  - **η is no longer a hand-set number.** The team decides the cost ratio λ, and η = λ/(1+λ) on calibrated probabilities.
  - **This adds a calibration step:** run the search on TRAIN scenes, physics-check the chosen plans, and fit the calibration map. That step has not been run yet; the results above use cross-validation on VAL.

### D6. Tried and not adopted

| Idea | Result | Reading |
|---|---|---|
| Hybrid v1: cVAE density × p11 | Offline worse than p11 alone (hit@4 0.10 vs 0.29) | A goal-blind prior drags down a goal-aware score. |
| Ranker veto before the Boltzmann pick | Never changed the #1 pick (90/90) | The veto threshold is too weak to matter. |
| Cautious scoring (min over 3 seeds) | 83 vs 88 first try | 3 seeds disagree too little to correct the winner's curse. |
| No X/Y structure (DIRECT-GEO) | Plans 73 vs 88; placement 68 vs 78 | The X/Y structure helps under search (placement p ≈ 0.03), but its log-loss is equal (0.200 vs 0.203). |
| cVAE latent | No effect (E7: 89/90 identical) | A goal-aware 36-cell classifier is enough. |
| PIGINet as a plan scorer | Paused (partly trained) | Code in `archive/`. |

---

## 4. The workflow, before and after

| Stage | Before (team pipeline) | After (proposed from these results) |
|---|---|---|
| Success-model labels | Sidecar X/Y (two bits per plan; replay/teleport protocol) | **Executed, chained records with the failing step** (`labels.B`): the same protocol as deployment |
| Success model | Boltzmann X/Y, p11 for the whole plan | **Per-step model** (or the factor graph, for blocker explanations) |
| Proposal model | Goal-blind cVAE, trained on successes only | Optional. Score all free cells, or use a **goal-aware cVAE distilled from the success model, anchored to real labels** |
| Search | Ranked expansion; cost −log p11 + α\|r\| + βD + γĴ | Same search. Partial plans are scored per step, so pruning is better, and the length penalty matters less because the per-step model already prefers short plans |
| Decision | p11 ≥ η, η chosen by hand | **Calibrated p ≥ λ/(1+λ)**; the team picks λ, the cost of a failed attempt |
| Execution | Once; verify | Once, plus **lazy retries** in simulation (prune plans that share the failing prefix); on a robot, **re-plan from the current state** (to build) |
| Feedback | None | **Loop on TRAIN scenes:** search → physics-check the chosen plans → add confident failures to training → recalibrate (expert iteration / conservative models). Not run yet. |
| Compute | cVAE + Ranker + Boltzmann; GPU server for training | Per-step model trains in ~40 s on CPU; planning 0.4 s per scene |

---

## 5. Time and scaling

### Planning time

Mean seconds per scene for the beam search, width 16, 1 CPU thread, with no physics running. Depths 4–5 go beyond the trained plan lengths, so they measure speed only.

SCALING_TABLE

### Physics time

The physics teacher dominates. One attempt takes about 18–33 s, and it grows with the number of steps (PHYSICS_BY_STEPS).

### Per scene, end to end

| Method | Decision time | Physics checks while planning | Executions | Total per scene |
|---|---|---|---|---|
| Per-step, first try | 0.4 s | 0 | 1 | ~22 s |
| Per-step, ≤ 5 tries | 0.4 s | 0 | 1.5 | 43 s |
| Team pipeline, ≤ 5 tries | ~1 s | 0 | 2.0 | 58 s |
| REAP 24 h (Ayush; TEST scenes) | ~150 s, including ~51 teacher checks | ~51 | 1 | ~150 s + execution |
| E-M4M (paper Table II) | ~182 s | ~27 | 1 | ~182 s + execution |

### Will it scale?

- **More objects (n) and longer plans (depth D):**
  - Beam search scores about `D × B × n × C` plans (B = 16 beam, C ≈ 30 free cells): **linear** in D and n, not exponential like exhaustive search.
  - The per-step model's cost per plan is linear in plan length, and shared prefixes are scored once.
  - The measured times above follow this.
  - Exact or analytical planners face an NP-hard problem (Wilfong 1991; Hauser 2014). REAP's time already grows with K (K3: 195 s).
- **Quality at larger D is unknown.** The training data has almost no plans longer than 3 moves (10 four-move plans in VAL). Per-step failure labels for move 2 and later are thin (125 negatives in TRAIN). Errors compound multiplicatively over steps, so long plans need more labelled data, not a different model.
- **A bigger grid or other shelves:** the scene features are normalised to the 6×6 grid (row/5, col/5) and the cell sets are fixed at 36. A different grid needs retraining and feature changes. The models themselves are set-based, so object count can change freely.
- **Physics dominates real time.** Fewer wasted attempts (1.5 vs 2.0 plans per scene) matters more than model speed.

---

## 6. Literature this builds on

Full lists, all verified, are in [docs/literature/related_work_brief.md](docs/literature/related_work_brief.md) and [docs/literature/three_way_scan.md](docs/literature/three_way_scan.md). The ones used here:

- **Per-step verification:** Lightman et al. 2024; Gensheimer & Narasimhan 2019; STAP (Agia et al. 2023); Wells et al. 2019.
- **Goal-aware generator from a success model:** CbAS (Brookes et al. 2019); AWR (Peng et al. 2019); Kim et al. 2022.
- **Overconfidence under search:** Smith & Winkler 2006; Gao et al. 2023; Coste et al. 2024; COMs (Trabucco et al. 2021); conformal selection (Jin & Candès 2023).
- **Lazy checking:** LazySP (Dellin & Srinivasa 2016; Haghtalab et al. 2018); Lazy Rearrangement (Wang et al. 2022).
- **Baselines and context:** REAP, PIGINet, GRN, SceneCollisionNet (team branches); Ren & Qureshi 2024 (OSNet/RPNet, greedy and closed-loop, 72% in their setting); M4M and E-M4M (Saxena & Likhachev 2023); algorithmic recourse (Karimi et al. 2021), which is the same "cheapest fix that works" shape as Algorithm 1, and fails the same way when the model is wrong.

---

## 7. Caveats (read before citing anything)

1. **Exploratory only.**
   - The same VAL roots carry about 25 comparisons.
   - Only per-step vs team pipeline (first try and ≤ 5 tries) and per-step vs Boltzmann survive multiple-comparison correction.
   - Confirmation needs a **pre-registered** run on fresh scenes or TEST, agreed with the team: TEST is shared and can be used once.
2. **Selection on VAL.** Both the per-step model and the Boltzmann model were early-stopped on these VAL roots, so neither is clean. The goal-aware cVAE's β was chosen on TRAIN.
3. **Retries assume a simulator reset.** About 50% of failed attempts happen after a move was executed. Use first-try numbers for real-robot claims.
4. **The REAP and E-M4M comparisons are indicative.** They are on different scenes (V4 TEST), and their "checks" are cheaper feasibility queries, while ours are full executions.
5. **Laptop CPU timings, one run each.** No variance is reported for times.
6. **Some models are one seed or partly converged.** The goal-aware cVAE was still improving at epoch 30. PIGINet was not completed.
7. **Algorithm 1 is not reproduced exactly.** Its α, β, γ and travel predictor were not available, so "team pipeline" here means the cVAE → Ranker → Boltzmann chain with p11 ranking.

---

## 8. Next steps (in priority order)

1. **A pre-registered confirmation** of per-step vs team pipeline, first try, on fresh or TEST scenes. Write a one-page plan with hypothesis, metric, sample size and analysis, and agree it with Naren and Ayush.
2. **A feedback loop on TRAIN scenes:** search → physics-check the chosen plans → retrain on confident failures → fit calibration. This targets the overconfidence at its source and makes η = λ/(1+λ) usable.
3. **Closed-loop re-planning** from the actual post-failure state, for honest real-robot retry numbers.
4. **The K3 gap:** neither the learned planners nor REAP do well on 3-move scenes. More long-plan training data is needed there.
5. **Ask Naren:** is there unpushed Algorithm 1 code, and what are α, β, γ and η? With them, the team's exact method can be reproduced alongside these results.

## 9. Where things are

- **Code:** [src/](src/). See [README.md](README.md) for the file map and commands.
- **Results (gitignored):** `../approachb_3d_learning/runs/`
  - `PHASE2/`, `PHASE3/`, `PHASE3B/`, `PHASE4/`: picks, `results.jsonl`, summaries, `threshold.json`, `scaling.json`;
  - `GCVAE_ABL/physics_summary.json`;
  - `HAZARD/`, `FACTOR/`, `GCVAE/`: model checkpoints and metrics.
- **Experiment log with pre-registrations:** [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).
