# Results: design changes to the learned relocation planner, and what they do to the workflow

**In one line:** we changed parts of the team's learned planner and tested each change in physics simulation. A per-step success model gave the biggest gain. Every result here is exploratory.

**Period:** 2026-11-20 to 21. **Machine:** laptop CPU (8 cores).

**Ground truth:** the dataset's own physics teacher (`BD3DS66_TEACHER_V2`: xArm7, RRT-Connect with 5,000 iterations, MuJoCo, moves chained).
- xArm7: the robot arm.
- RRT-Connect: the motion planner that finds arm paths.
- MuJoCo: the physics simulator.
- "Moves chained": each move starts from the scene the previous move left behind.

**Scenes:**
- All results use Approach-B **VAL** roots: 90 placement contexts and 160 roots with verified K ≥ 1 (53 K1, 67 K2, 40 K3).
- **TEST was never read.**
- These VAL roots were reused for about 25 comparisons, so **every result here is exploratory.** See §7 and the per-experiment log in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md).

### Plain-language key

- **Root:** one starting scene to solve (a shelf with objects, and a target to get out).
- **K:** how many objects must be moved to free the target. K1 = one, K2 = two, K3 = three.
- **VAL / TEST / TRAIN:** data splits. TRAIN is for fitting models. VAL is for choosing and checking. TEST is kept untouched for a final check.
- **Plan:** a list of moves (put object A in cell X, then ...) followed by taking out the target.
- **First try:** the first plan the planner picks is executed, and it works.
- **Within 5 tries:** if a plan fails, the planner may try another one, up to 5 plans. This assumes the simulator can be reset after a failure.
- **cVAE:** a model that suggests where to put an object. Here it is "goal-blind": it does not know where the target is.
- **Boltzmann X/Y model:** the team's success model. It predicts two yes/no outcomes per plan: X = the moves work, Y = the target gets out.
- **p11:** the Boltzmann model's probability that both X and Y happen, i.e. the whole plan works.
- **Ranker:** a model that keeps the 4 most promising cells for an object.
- **AUROC:** how well a score separates plans that work from plans that fail. 0.5 = coin flip, 1 = perfect.
- **Brier score:** average squared error of predicted probabilities. Lower is better.
- **ECE:** how far predicted probabilities are from real success rates, on average. Lower is better.
- **Calibration:** whether "90% sure" really means it works about 90% of the time.
- **Optimizer's curse:** if you pick the highest of many noisy scores, the winner's score is usually too high, because it includes a lucky error.
- **Lazy retries:** do not check plans in advance. Run the best plan, and only after it fails, throw away the plans that would fail the same way.
- **Sign test (wins vs losses):** on the roots where two methods differ, count how often each one wins. The p-value says how likely such a split is by chance. Small = unlikely to be chance.
- **Holm correction:** an adjustment of p-values when many comparisons are made, so lucky results do not look significant. "n.s." = not significant.
- **Abstain:** the planner declines to act when it is not confident enough.

---

## 1. Headline

**Takeaway:** a per-step success model beats the team pipeline in simulation. But after fixing the baseline (E10), most of the first-try gain disappears. The retry edge remains.

> **Update (E10, 2026-11-21): the table below compares against a team baseline WITHOUT Algorithm 1's length penalty.**
> - With the penalty at α* = 0.2 (tuned on these roots), the team pipeline works first try on **119/160**. Per-step gets 124 and the outcome-only model 120, and these can't be told apart (per-step vs team 29–24).
> - Within 5 verified tries, the per-step *planner* keeps an edge: **154 vs 141** (18–5, uncorrected p = 0.011). That edge comes mainly from searching all free cells: the same scorer on the team's 4 proposals gets 143.
> - So most of the first-try gain (19 of 24 roots) was the missing length penalty. Step-level labels add only about 4.
> - The retry edge is the claim being tested on fresh scenes (pre-registered E9v2 in [docs/EXPERIMENT_LOG.md](docs/EXPERIMENT_LOG.md)).

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
3. **Every learned scorer is overconfident on the plans the search picks** (it says 95–99%; reality is 55–78%). This is the optimizer's curse.
   - Recalibration plus a cost-based threshold fixes the decision.
   - It works only for scorers that still rank the plans they pick (per-step and factor; not the Boltzmann model).

---

## 2. The baseline: the team's current design (what we started from)

**Takeaway:** the team's design proposes placements without knowing the goal, and scores plans with only two yes/no outcomes. It gets no feedback from failures.

```
OFFLINE
  placement queries (VERIFIED only) ──► cVAE   p(d | scene, object)          goal-blind by design (execution contract)
  labelled placements (VERIFIED/NEG) ─► Ranker p(feasible | scene, object, d) goal-blind
  X/Y sidecar (7,656 X / 7,072 Y) ───► Boltzmann X/Y:  P(Y)=σ(b),  P(X|Y)=σ(a + w·y),  p11 = σ(b)·σ(a+w)
ONLINE (Algorithm 1)
  scene ─► cVAE samples cells ─(Ranker keeps 4)─► extend partial repairs ─► score each with p11
        ─► cost C = −log p11 + α|r| + βD + γĴ ─► best plan with p11 ≥ η, else ABSTAIN ─► execute once ─► verify
```

How to read it: the cVAE proposes cells, the Ranker keeps 4, the search builds plans, and p11 scores them. The cost adds penalties: α for plan length |r|, plus the β and γ terms. The plan runs only if p11 ≥ η (the confidence threshold).

**Properties of this design that matter for the results:**
- **The cVAE never sees the goal and never sees a failure.** It only learns "where do placements usually succeed".
- **The cVAE's latent is near-collapsed.** Report 09 found 0–2 of its 8 latent units active, so it behaves like a plain 36-cell classifier. E7 confirms this: removing the latent changed nothing.
- **The Boltzmann model sees two bits per plan.** Did the moves work (X)? Did the target get out (Y)?
  - Its labels come from a slightly different protocol than deployment.
  - X adds a return-to-ready connector.
  - Y teleports objects and starts the target from the ready pose.
- **There is no feedback.** Failures, the failing step, and the Boltzmann model's goal knowledge never reach the proposal model.
- **η, α, β and γ have no recorded selection procedure.** They were not found on any branch.

---

## 3. Design changes, one by one

Each change lists: **what changed → why (the evidence that motivated it) → effect on the workflow → result.**

### D1. Make placement goal-aware

**Takeaway:** goal-aware placement is far better for single placements. It does not by itself improve whole plans.

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
  - A distillation step is added after the success model is trained. (Distillation: training a model to copy a bigger or better one.)
  - This **breaks the team's "goal-blind sampler" contract**, so it must be agreed before adoption.
  - Placement-level gains do not carry over to whole plans, so on its own D1(b) is **not** the main lever.

### D2. Replace the two-bit X/Y outcome with a per-step success model (the main gain)

**Takeaway:** predicting success for each move separately works better than two yes/no bits. This is the main gain, on K1 and K2 roots.

- **Why:**
  - **Failure analysis:** of the Boltzmann planner's first-try failures, 50 of 72 fail at **move 1** ("the arm has no feasible grasp or path"), not at the target.
  - **Data audit:** the dataset already records **which step failed and why** for every executed plan (`labels.B[*].failed_step`). The X/Y model throws this away.
  - **Literature:** process (step-level) verifiers beat outcome verifiers as plans get longer (Lightman et al., ICLR 2024). Discrete-time hazard models give the loss directly (Gensheimer & Narasimhan 2019). (A hazard model predicts at which step something fails.)

- **What** ([src/hazard.py](src/hazard.py)):
  - `p(plan) = Π_k P(move k ok | scene before move k) · P(target ok | final scene)`.
  - **Encoder:** the team's SceneEncoder, applied to the scene before and after each step.
  - **Loss:** plain BCE, so the per-step probabilities stay calibrated enough to multiply. (BCE: the standard yes/no training loss.)
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

**Takeaway:** the factor graph scores best offline and explains which objects block. But it does not give better first-try plans than per-step.

- **Why:**
  - Choosing which obstacles to clear is a minimum-set-cover / minimum-constraint-removal problem (Hauser 2014; Kaznachey & Jagota 1997). (Find the smallest set of objects whose removal frees the target.)
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

**Takeaway:** retrying smartly after a failure lifts success within 5 tries. Per-step is best on every column. The retry numbers are simulation-optimistic.

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

**Takeaway:** the scores are overconfident. Per-step and factor scores can still be fixed by recalibrating. The Boltzmann score cannot.

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
  - Choose η by 5-fold cross-validation over roots. (Cross-validation: tune on part of the roots, test on the rest.)
  - **Or**, with no tuning: recalibrate the score with isotonic regression, then act iff `p_cal ≥ λ/(1+λ)`. (Isotonic regression: a fit that maps raw scores to real success rates while keeping the order.)

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

**Takeaway:** the proposed workflow trains on real executed records, scores each step, uses a calibrated threshold, and retries lazily.

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

**Takeaway:** every learned planner decides in under ~1 s, and planning time grows roughly linearly. Physics dominates real time. A hard limit of 8 objects blocks larger scenes.

### Planning time

Mean seconds per scene for the beam search, width 16, 1 CPU thread, with no physics running. (Beam search: keep only the best 16 partial plans at each depth.) Depths 4–5 go beyond the trained plan lengths, so they measure speed only.

**By search depth** (42 sampled roots, 5–6 per object count; [src/scaling.py](src/scaling.py) → `runs/PHASE4/scaling.json`):

| Depth (max moves) | Plans scored | Boltzmann | Per-step | Factor graph | Team pipeline (cVAE + Ranker proposals) |
|---|---|---|---|---|---|
| 1 | 115 | 0.016 s | 0.011 s | 0.017 s | 0.007 s (16 plans) |
| 2 | 1,470 | 0.24 s | 0.23 s | 0.24 s | 0.07 s (195 plans) |
| 3 (used in all experiments) | 2,421 | 0.46 s | **0.27 s** | 0.42 s | 0.12 s (330 plans) |
| 4 | 3,044 | 0.71 s | **0.39 s** | 0.54 s | n/a |
| 5 | 3,411 | 0.84 s | **0.38 s** | 0.54 s | n/a |

**By number of movable objects** (depth 3, seconds per scene):

| Objects | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| Boltzmann | 0.006 | 0.07 | 0.23 | 0.41 | 0.61 | 0.88 | 1.07 |
| **Per-step** | 0.004 | 0.07 | 0.16 | 0.27 | 0.37 | 0.48 | **0.58** |
| Factor graph | 0.005 | 0.09 | 0.25 | 0.39 | 0.57 | 0.81 | 0.90 |
| Team pipeline | 0.004 | 0.02 | 0.07 | 0.11 | 0.17 | 0.22 | 0.28 |

**Reading:**
- **Planning time grows roughly linearly** with the number of objects, and flattens with depth: the beam caps the work, and fewer objects remain to move.
- **The per-step model is the cheapest scorer**, about 2× faster than Boltzmann at 7 objects.
- **Every learned planner decides in under ~1 s**, even at 7 objects and depth 5.

**Hard limit found:** the team's scene encoding (`MAXOBJ = 8` in `approachb_3d_learning/src/data3d.py`) caps a scene at 8 objects including the target.
- **32 of 91 fresh V4 scenes cannot be scored at all.** These are NONE and K4 scenes with 8–11 movable objects.
- The cap applies to every model here, including the team's cVAE.
- **Scaling past 7 movable objects needs a new feature encoding and retraining**, not just more data.

### Physics time

The physics teacher dominates real time. Mean seconds per attempt, measured until success or the first failure:

| Steps in plan (moves + target) | 2 | 3 | 4 | 5 |
|---|---|---|---|---|
| Physics s per attempt | 11.6 (n = 759) | 22.5 (n = 1,053) | 35.6 (n = 1,757) | 28.4 (n = 78) |

The 5-step mean is lower because most of those attempts stop early on a failure.

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
- **Quality at larger D is unknown.**
  - The training data has almost no plans longer than 3 moves (10 four-move plans in VAL).
  - Per-step failure labels for move 2 and later are thin (125 negatives in TRAIN).
  - Errors compound multiplicatively over steps, so long plans need more labelled data, not a different model.
- **A bigger grid or other shelves:** the scene features are normalised to the 6×6 grid (row/5, col/5) and the cell sets are fixed at 36. A different grid needs retraining and feature changes. The models themselves are set-based, so object count can change freely.
- **Physics dominates real time.** Fewer wasted attempts (1.5 vs 2.0 plans per scene) matters more than model speed.

---

## 6. Literature this builds on

Full lists, all verified, are in [docs/LIT_BRIEF_2026-11-20.md](docs/LIT_BRIEF_2026-11-20.md) and [docs/THREE_WAY_SCAN_2026-11-20.md](docs/THREE_WAY_SCAN_2026-11-20.md). The ones used here:

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
   - Confirmation needs a **pre-registered** run on fresh scenes or TEST, agreed with the team: TEST is shared and can be used once. (Pre-registered: the test and the success rule are written down before the results are seen.)
2. **Selection on VAL.** Both the per-step model and the Boltzmann model were early-stopped on these VAL roots, so neither is clean. The goal-aware cVAE's β was chosen on TRAIN.
3. **Retries assume a simulator reset.** About 50% of failed attempts happen after a move was executed. Use first-try numbers for real-robot claims.
4. **The REAP and E-M4M comparisons are indicative.** They are on different scenes (V4 TEST), and their "checks" are cheaper feasibility queries, while ours are full executions.
5. **Laptop CPU timings, one run each.** No variance is reported for times.
6. **Some models are one seed or partly converged.** The goal-aware cVAE was still improving at epoch 30. PIGINet was not completed.
7. **Algorithm 1 is not reproduced exactly.** Its α, β, γ and travel predictor were not available, so "team pipeline" here means the cVAE → Ranker → Boltzmann chain with p11 ranking.

---

## 8. Next steps (in priority order)

1. **A pre-registered confirmation** on fresh scenes: E9v2 (written 2026-11-21). Primary: per-step vs team α*, success within 5 tries. Secondary: first try. Fresh roots are being generated (`hx1`). Agree the plan with Naren and Ayush before the results are read.
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
