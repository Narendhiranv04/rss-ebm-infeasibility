# Brief: can factor graphs or better search heuristics raise plan success?

Date 2026-11-21. Quick-mode ARS brief. Development only (VAL roots, exploratory). AI-assisted; see the verification note at the end.

## 1. Factor graphs in one paragraph

A factor graph splits one big probability (or cost) into many small pieces ("factors"), each looking at only a few variables, and multiplies them. "Does the arm reach move 2?" is one factor. "Does object o block the target?" is another. Because the pieces share variables, evidence on one piece updates the others. For us that could mean:
- per-object "is it in the way?" variables, so that a failed check on one plan teaches us about other plans;
- per-constraint factors for placements, for example "this cell must not block the target line";
- a solver that searches over placements instead of scoring finished plans.

Our E8 model already does the first of these (hazard moves times a noisy-AND over blocker latents). It had the best offline calibration (ECE 0.017) but a worse first try (110 vs 124). Likely reasons (my reading, not tested):
- Most first-try failures (50 of 72) happen at move 1, which the blocker factors do not touch.
- Multiplying many "o does not block" factors adds small errors per object, so the search, which picks the plan where errors happen to be favourable, is hurt more (optimizer's curse). The model says 0.99 for plans that really succeed 0.69 of the time.
- The 110 vs 124 gap is 15 wins vs 29 losses, Holm p = 0.24, so it may be noise.

## 2. Candidate ideas, ranked by payoff versus effort

**1. Calibrate on the plans the search picks, not on all plans (conformal / selection-aware recalibration).**
- What: fit a small recalibrator (isotonic or Platt, cross-fitted by root) on the scores of the plans the beam search actually selected, as a held-out set. Optionally add a conformal guarantee.
- Why here: every scorer says 95-99% on picked plans and gets 55-78%. Fixing this helps the stop/retry/abstain decision and the cost-based threshold, not first-try ranking.
- Cost: hours. It uses only logged plans and outcomes. No new physics.
- Cites: Angelopoulos & Bates 2021 (arXiv 2107.07511); Ren et al., KnowNo, CoRL 2023 (arXiv 2307.01928) uses conformal prediction to decide when a planner should ask for help, which is the same shape as "abstain or retry".

**2. Choose the next check by value of information (LazySP-style).**
- What: after a failure, re-rank the remaining plans by P(success) divided by expected check cost (12-36 s). Prefer a plan whose first move is already verified, and try the cheapest informative check first. We already prune plans that share a failed prefix.
- Why here: the retry edge (154 vs 141 within 5) is the real gain, and checks are the expensive resource.
- Cost: low to moderate, a re-ranking rule on the existing retry loop.
- Cites: Dellin & Srinivasa, ICAPS 2016 (arXiv 1603.03490); Haghtalab, Mackenzie, Procaccia 2017 (arXiv 1710.04101), on how few expensive checks a lazy search needs.

**3. Use failed retries as training data (active learning / dataset aggregation).**
- What: log every verified plan the planner runs, with its failing step, and retrain the hazard model on those. They come from the search-selected distribution, which is exactly where the model is overconfident.
- Why here: the training set holds mostly plans from the dataset's generator, not plans the optimizer prefers. Move-1 failures (2,314 positives vs 510 negatives) are the main error.
- Cost: moderate. It needs a new physics loop, and a split that keeps VAL clean.
- Cites: Kim, Wang, Kaelbling 2018 (arXiv 1807.09962) learn to guide task-and-motion planning from the planner's own experience. This is a loose analogue; the aggregation idea itself is my inference.

**4. Learned cost-to-go for best-first / A* instead of fixed-width beam.**
- What: score a partial plan as (cost so far) + (predicted moves still needed). The hazard model already scores partial plans, and the E8 blocker latents (sum of P(o blocks)) are a natural "remaining blockers" estimate.
- Why here: the beam is capped at depth 3 and width 16. An informed best-first search could use the width where it matters. K3 (40 roots) is where we gain nothing.
- Cost: moderate, since it is a new search loop. The heuristic is probably not admissible, so there are no optimality guarantees.
- Cites: Khodeir, Agro, Shkurti, 2021/RA-L (arXiv 2111.13144), learned search guidance in task-and-motion planning; the min-constraint-removal framing is Hauser 2014, already in RESULTS.md (not re-checked here).

**5. Monte Carlo tree search, or replanning from the post-failure state.**
- What: after a failed check, replan from the scene the failed prefix actually produced, rather than only discarding plans that share it.
- Why here: chained execution is deterministic, so the failure state is known and informative.
- Cost: higher. Each rollout would need the model, not physics, but the tree is big.
- Cites: Labbé et al. 2019 (arXiv 1904.10348); Song et al. 2019 (arXiv 1912.07024) on MCTS for rearrangement.

**6. Constraint-solver placement (Diffusion-CCSP, factored NLP / LGP).**
- What: generate placements by sampling from a product of constraint factors.
- Why here: probably low payoff. Searching all free cells already beats the 4 cVAE proposals, and the placement space is only 36 cells. It becomes interesting only if the arena grows.
- Cost: high.
- Cites: Yang, Mao, Du, CoRL 2023 (arXiv 2309.00966). I did not verify the Toussaint LGP/Factored-NLP papers (the arXiv search returned unrelated work), so cite them from the team's reading list only after checking.

## 3. Recommended next experiment (cheap, VAL only)

Selection-aware recalibration (idea 1), then replay with idea 2's ordering.
1. From the Phase 4 / E10 logs, collect every executed plan (score, first-failing step, outcome) for hazard, factor, and team arms.
2. Fit isotonic recalibration with 2-fold cross-fitting by root, so no root is scored by a recalibrator that saw it.
3. Re-rank the logged plans within each root by calibrated P (and, as a variant, by P/cost) and replay the retry order offline over the logged plans.

Success: on picked plans, ECE falls from about 0.25 (predicted 0.96 vs real 0.78) to under 0.08, Brier improves by at least 15%, and the replayed mean number of checks per root drops (hazard currently 1.48 plans, 4.1 queries) with no loss within 5 tries. If calibration improves but the ordering does not, idea 1 is only a decision fix, and we should move to idea 3. Caveat: replay can only use plans that were actually run, so it understates any gain from reordering.

## Verification note
Checked (title, authors, year found on arXiv listing): 2107.07511, 2307.01928, 1603.03490, 1710.04101, 1807.09962, 2111.13144, 1904.10348, 1912.07024, 2309.00966. Venues for LazySP (ICAPS), KnowNo and Diffusion-CCSP (CoRL) come from my memory, not the listing. Unverified: Toussaint LGP / Factored-NLP / Graph-NLP, M4M / E-M4M, GRN, Hauser 2014. Abstracts were not read in full.
