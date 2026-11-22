# Research Brief (plain-language version)

**Title**: Learned proposal + learned success model for obstacle relocation: prior work and next experiments
**Date**: 2026-11-20
**Mode**: Quick Research Brief (ARS deep-research, quick mode; four parallel search strands: robotics, machine learning, generative planning / oracle use, theory)
**AI Disclosure**: Produced with AI-assisted research tools. Every source below was confirmed to exist (title, authors, venue, year, URL) by a web search of the publisher, proceedings or arXiv page. Summaries come from abstracts and proceedings pages, not full-text reading (exception: Ren & Qureshi 2024).

---

## The main takeaways for our project

- **Our gap is normal.** Plans we pick look about 97% likely to work, but only about 55% work in the physics simulator. Other fields see the same thing. It is called the "optimizer's curse", and it has standard fixes.
- **Judge each move, not just the whole plan.** Models that score one step at a time do better as plans get longer. Most of our failures are "the arm cannot plan move 1", so a per-move score fits.
- **Don't multiply the generator by the score.** The better way to combine our placement generator (cVAE) and our success model is to retrain or steer the generator toward successful plans.
- **Use the learned model to rank, and the simulator to check.** Checking lazily (only the moves of the current best plan) has proven worst-case guarantees. A fair comparison with REAP is success versus number of simulator checks (REAP: ~80% with ~40 checks per scene).
- **Caveat.** This was a quick search from abstracts, not a full review. Results from language-model and molecule-design papers may not carry over to our small problem (36 cells, 3 moves or fewer).

---

## Plain-words glossary

| Term | Simple meaning |
|---|---|
| cVAE | A model that suggests where to put a moved object. Ours is "goal-blind": it ignores which target we want. |
| Boltzmann X/Y model | Our success model. X = can the moves be carried out? Y = does the target end up reachable? |
| Beam search | Keep only the few best partial plans at each step, then extend them. |
| Oracle | The slow, exact check. For us, the physics simulator and motion planner. |
| Optimizer's curse | If you pick the option with the best noisy score, that score is too optimistic. |
| Ensemble | Several models trained separately. Their disagreement shows how unsure they are. |
| Pessimistic scoring | Rank by a cautious score, such as the worst model's score or mean minus k times the spread (std). |
| Calibration | Whether "90% likely" really means right 90% of the time. |
| Process vs outcome verifier | Process: scores each step. Outcome: scores only the final result. |
| Hazard model | Predicts the chance of failing at step k, given steps before k worked. |
| Success-weighted refit | Retrain a generator, giving more weight to examples that succeeded. |
| GFlowNet | Samples complex objects with probability proportional to their reward, instead of always taking the top one. |
| Diffusion / energy model | Generate or score layouts by repeatedly nudging them toward a lower "cost". |
| LazySP | Plan first, then run expensive checks only on the path you are about to use. |
| NP-hard | No known fast method for the worst case. Small cases can still be solved exactly. |
| NAMO | Navigation among movable obstacles: pushing things aside to get through. |
| TAMP | Task and motion planning: deciding what to do and how the arm moves. |
| Conformal selection | A method that gives a provable cap on the share of wrong picks in a shortlist. |

---

## Background & Research Question

**Context.**
- Today's pipeline proposes placements with a goal-blind cVAE.
- It then scores plans with an exact two-variable Boltzmann model (X: moves executable, Y: target reaches goal).
- Our Phase 2/3 physics checks found the success model beats the cVAE.
- Open-loop plans from beam search are overconfident. "Open-loop" means we plan once and never re-check.
- REAP succeeds ~80% using ~40 oracle checks per scene.

**Research question.** Which existing methods help with these four things?
- (a) Combining a proposal model with a success model.
- (b) Overconfidence when search runs over a learned scorer.
- (c) Scoring success one step at a time.
- (d) Spending a slow physics oracle wisely.

**Scope.** In: TAMP and rearrangement learning, generator-plus-verifier methods from ML, energy and diffusion planning, theory of selection, search and hardness. Out: systematic coverage, full-text appraisal, 2D-domain baselines.

---

## Key Findings

### Finding 1: The closest robotics work predicts blockers and scores placements, but checks motion feasibility for real at every step

**Evidence strength**: Moderate (close task match; peer-reviewed; abstracts only except Ren & Qureshi).

| Paper | What it does | Why it matters for us |
|---|---|---|
| Ren & Qureshi (ICRA 2024; read in full via arXiv HTML) | Retrieves a target from a cabinet seen by an in-hand camera. One network (OSNet) picks which object to move. A second (RPNet) scores where to put it. | Closest task to ours. It checks reachability for real at every step. It has no generative placement model and no multi-step plan score. |
| Kim et al. (IJRR 2022) | Pairs a learned plan ranker with a learned placement sampler for moving objects among movable obstacles. | Like our setup, but its sampler is goal-conditioned. Our cVAE is goal-blind. |
| PLOI (Silver et al., AAAI 2021) | Predicts which objects matter, so search looks at fewer objects. | Supports a "which objects block" pre-filter. |

Details on Ren & Qureshi:
- OSNet is target-aware. Its labels count how often an object blocks RRTConnect (a standard path planner) paths to the target.
- RPNet scores placement regions from hand-made cost labels. Blocked, occupied or unseen regions get the maximum cost. Other regions cost their distance from the object's start.
- Planning is greedy and closed-loop. Each step checks the target is reachable, picks one move, checks that move's path, executes, and re-observes.
- Results: 72% vs 55% local and 45% random. OSNet alone reaches 71%, while RPNet alone reaches 47%.

### Finding 2: Overconfidence under search is the optimizer's curse, and has standard remedies

**Evidence strength**: Strong (theory plus repeated empirical results across fields).

| Paper | What it does | Why it matters for us |
|---|---|---|
| Smith & Winkler (Management Science 2006) | Shows that picking the best of several noisy but unbiased estimates is biased upward. Bayesian shrinkage corrects it. | Names our 97% vs 55% gap and gives a fix. |
| Gao et al. (ICML 2023) | Shows proxy scores drift from the truth in a predictable way as optimization pressure grows. | Our beam width and candidate count are that pressure. |
| Coste et al. (ICLR 2024); ensembles: Lakshminarayanan et al. (NeurIPS 2017) | Score with an ensemble's worst member, or mean minus k times std. | Largely removes the effect. We have three trained seeds to use. |
| COMs (Trabucco et al., ICML 2021) | Trains the scorer with the optimizer's own picks as penalised negatives. | Basis for hard-negative retraining. |
| Noseworthy et al. (RSS 2021) | Uses ensemble-disagreement active learning to fix plan-feasibility models with few executions. | Robotics evidence that this works with few real runs. |
| Jin & Candès (JMLR 2023) | Conformal selection: a provable false-discovery-rate guarantee on a shortlist. | A possible way to certify a shortlist of plans. |

### Finding 3: Per-step success models beat whole-outcome models as plans get longer

**Evidence strength**: Moderate (strong in language-model reasoning; robotics evidence is consistent but smaller).

| Paper | What it does | Why it matters for us |
|---|---|---|
| Lightman et al. (ICLR 2024) | Step-level (process) verifiers beat outcome verifiers for best-of-N on hard math. | Supports scoring each move. |
| Uesato et al. (2022 preprint) | Compares the two on easier problems. They were similar. | A caution: the gain may depend on difficulty. |
| Gensheimer & Narasimhan (PeerJ 2019) | Discrete-time hazard model: a ready-made loss for "fails at step k given steps before k succeeded". | Directly fits our move-by-move failures. |
| STAP (Agia et al., ICRA 2023) | Multiplies per-skill success probabilities and rejects out-of-distribution steps. | Robotics example of per-step scoring. |
| Wells et al. (RA-L 2019) | Learns per-action feasibility to guide TAMP. | Same idea in tabletop planning. |
| LevinTS (Orseau et al., NeurIPS 2018) | Tree search whose expansion bound tightens with the probability a per-step policy puts on the solution. | Better calibration pays off directly. |

### Finding 4: "Generator x success score" is better done by conditioning or refitting the generator

**Evidence strength**: Moderate (well-established in design and generation; untested on our task).

| Paper | What it does | Why it matters for us |
|---|---|---|
| CbAS (Brookes et al., ICML 2019) | Refits a generator on success-weighted samples, keeping it near the data. | Template for refitting the cVAE. |
| AWR (Peng et al., 2019 preprint) | Same idea: weight samples by success when refitting. | Same. |
| Model-inversion networks (Kumar & Levine, NeurIPS 2020) | Generate inputs conditioned on the desired score. | Alternative: condition the cVAE on success. |
| GFlowNets (Bengio et al., NeurIPS 2021) | Sample sequential discrete objects in proportion to a reward. | Fits plans over 36 cells. Sampling rather than taking the top pick hedges against scorer error. |
| Sequential Monte Carlo with a learned proposal and learned weights (Piché et al., ICLR 2019) | The inference-style alternative: a proposal plus learned weights. | A possible alternative design. |
| DIFUSCO (Sun & Yang, NeurIPS 2023) | Discrete diffusion. | Heavier option. |
| Diffusion-CCSP (Yang et al., CoRL 2023); Generative Skill Chaining (Mishra et al., CoRL 2023) | Composed constraint energies. | Heavier option. |

### Finding 5: Use the oracle lazily; exact reasoning is affordable at our size

**Evidence strength**: Strong for the theory; Moderate for transfer to our setting.

| Paper | What it does | Why it matters for us |
|---|---|---|
| LazySP (Dellin & Srinivasa, ICAPS 2016; Haghtalab et al., ICAPS 2018) | Checks only the edges (here: moves) of the current best candidate. Worst-case optimal in the number of checks. | A principled way to spend checks. |
| StrOLL (Bhardwaj et al., RSS 2019) | Learned check-ordering improves LazySP. | A later upgrade. |
| Driess et al. (RSS 2020) | Ranked "propose then verify" loops gave orders-of-magnitude speedups in TAMP. | Same loop shape as ours. |
| Wilfong (1991); Hauser (IJRR 2014; displacement variant, RSS 2013) | Moving obstacles is NP-hard in general. Choosing which obstacles to remove is the NP-hard Minimum Constraint Removal problem. | Shows the general problem is hard. |
| Han et al. (IJRR 2018) | Rearrangement cost-optimality is NP-hard but practically solvable by ILP/TSP reductions. | At ~7 objects, exact enumeration of blocker sets is cheap. |
| Lykouris & Vassilvitskii (ICML 2018 / JACM 2021) | Learning-augmented algorithms: follow a model while keeping a worst-case fallback. | Shows how to trust a model safely. |

---

## Analysis & Implications

**What this means.**
- Our observations match the literature closely.
- The Boltzmann model ranks plans well but is miscalibrated exactly where the search pushes it. That is expected (Finding 2) and fixable without a new model class.
- Failures concentrate at move 1. That points to step-level feasibility as the weak part (Findings 1, 3).
- The "hybrid" idea is supported in principled form: refit or condition the generator on success (Finding 4). Multiplying a goal-blind density by a score, which we found to fail, is not.
- The fair comparison with REAP is success against number of oracle checks (Finding 5), not one-shot success.

**Recommendations** (cheapest first):

| # | Do this | Plain explanation | Sources |
|---|---|---|---|
| 1 | **Pessimistic scoring** | Rank beam plans by min or mean - k*std over the three trained seeds, with k fixed before testing. Plot success against beam width. | Coste 2024; Gao 2023 |
| 2 | **Success against oracle checks** | Report success@1/2/4/8 plans and a LazySP variant that checks the riskiest move of the best plan first, caching results across plans. Compare with REAP's ~80% at ~40 checks. | Dellin 2016; Driess 2020 |
| 3 | **Per-step hazard model** | Replace the X head with per-move hazards plus a target term, trained on executed-plan records. Our data audit found 2,314/510 move-1 and 2,333/617 target examples in train, but only 125 later-move failures. | Lightman 2024; Gensheimer 2019 |
| 4 | **Hard-negative retraining** | Label the search's winners on TRAIN scenes with the oracle and add them to training. Never use val/test scenes for this. | COMs 2021; Noseworthy 2021 |
| 5 | **Goal-aware generator** | Refit the cVAE with weights exp(beta * logit p_success), or condition it on goal and success. Compare with the pipeline. | CbAS 2019; AWR 2019; Kim 2022 |
| 6 | **Blocker pre-filter** | A learned "which objects block" head, or exact enumeration of small blocker sets, to cut beam branching. | PLOI 2021; Ren & Qureshi 2024; Hauser 2014 |

---

## Limitations

- Quick mode, not a systematic search. Coverage is four targeted strands of 10-12 papers each, and relevant work may be missing (e.g. learned NAMO, Kang et al. shelf-retrieval papers whose PDFs could not be checked).
- Method and result lines come from abstracts and proceedings pages. Theorem statements were not checked against full proofs.
- Two sources are preprints (Uesato 2022, AWR 2019) and are labelled as such.
- Effect sizes from language-model and molecular-design papers may not transfer to a 36-cell, 3-move-or-fewer robotics problem.

---

## References

- Agia, C., et al. (2023). STAP: Sequencing task-agnostic policies. *ICRA*. https://arxiv.org/abs/2210.12250
- Anthony, T., Tian, Z., & Barber, D. (2017). Thinking fast and slow with deep learning and tree search. *NeurIPS*. https://arxiv.org/abs/1705.08439
- Audibert, J.-Y., Bubeck, S., & Munos, R. (2010). Best arm identification in multi-armed bandits. *COLT*. https://www.learningtheory.org/colt2010/papers/59Audibert.pdf
- Bengio, E., et al. (2021). Flow network based generative models for non-iterative diverse candidate generation. *NeurIPS*. https://neurips.cc/virtual/2021/poster/26729
- Bhardwaj, M., et al. (2019). Leveraging experience in lazy search. *RSS*. https://roboticsproceedings.org/rss15/p50.html
- Brookes, D., Park, H., & Listgarten, J. (2019). Conditioning by adaptive sampling for robust design. *ICML*. https://proceedings.mlr.press/v97/brookes19a.html
- Cobbe, K., et al. (2021). Training verifiers to solve math word problems. *arXiv preprint*. https://arxiv.org/abs/2110.14168
- Coste, T., et al. (2024). Reward model ensembles help mitigate overoptimization. *ICLR*. https://proceedings.iclr.cc/paper_files/paper/2024/hash/dda7f9378a210c25e470e19304cce85d-Abstract-Conference.html
- Dellin, C., & Srinivasa, S. (2016). A unifying formalism for shortest path problems with expensive edge evaluations via lazy best-first search over paths with edge selectors. *ICAPS*. https://ojs.aaai.org/index.php/ICAPS/article/view/13788
- Driess, D., Ha, J.-S., & Toussaint, M. (2020). Deep visual reasoning. *RSS*. https://arxiv.org/abs/2006.05398
- Du, Y., et al. (2022). Learning iterative reasoning through energy minimization. *ICML*. https://proceedings.mlr.press/v162/du22d.html
- Florence, P., et al. (2021). Implicit behavioral cloning. *CoRL*. https://proceedings.mlr.press/v164/florence22a.html
- Gao, H., Feng, S., & Yu, J. (2021). On minimizing the number of running buffers for tabletop rearrangement. *RSS*. https://www.roboticsproceedings.org/rss17/p033.html
- Gao, L., Schulman, J., & Hilton, J. (2023). Scaling laws for reward model overoptimization. *ICML*. https://proceedings.mlr.press/v202/gao23h.html
- Gensheimer, M., & Narasimhan, B. (2019). A scalable discrete-time survival model for neural networks. *PeerJ, 7*, e6257. https://peerj.com/articles/6257
- Haghtalab, N., et al. (2018). The provable virtue of laziness in motion planning. *ICAPS*. https://arxiv.org/abs/1710.04101
- Han, S. D., et al. (2018). Complexity results and fast methods for optimal tabletop rearrangement with overhand grasps. *IJRR, 37*. https://arxiv.org/abs/1711.07369
- Hauser, K. (2013). Minimum constraint displacement motion planning. *RSS*. https://roboticsproceedings.org/rss09/p17.html
- Hauser, K. (2014). The minimum constraint removal problem with three robotics applications. *IJRR, 33*(1). https://doi.org/10.1177/0278364913507795
- Janner, M., et al. (2022). Planning with diffusion for flexible behavior synthesis. *ICML*. https://proceedings.mlr.press/v162/janner22a.html
- Jin, Y., & Candès, E. (2023). Selection by prediction with conformal p-values. *JMLR, 24*(244). https://jmlr.org/papers/v24/22-1176.html
- Kim, B., et al. (2022). Representation, learning, and planning algorithms for geometric task and motion planning. *IJRR, 41*(2). https://arxiv.org/abs/2203.04605
- Kumar, A., & Levine, S. (2020). Model inversion networks for model-based optimization. *NeurIPS*. https://proceedings.neurips.cc/paper/2020/hash/373e4c5d8edfa8b74fd4b6791d0cf6dc-Abstract.html
- Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). Simple and scalable predictive uncertainty estimation using deep ensembles. *NeurIPS*. https://proceedings.neurips.cc/paper/2017/hash/9ef2ed4b7fd2c810847ffa5fa85bce38-Abstract.html
- Lightman, H., et al. (2024). Let's verify step by step. *ICLR*. https://proceedings.iclr.cc/paper_files/paper/2024/hash/aca97732e30bcf1303bc22ac3924fd16-Abstract-Conference.html
- Lykouris, T., & Vassilvitskii, S. (2021). Competitive caching with machine learned advice. *JACM, 68*(4). https://arxiv.org/abs/1802.05399
- Mishra, U., et al. (2023). Generative skill chaining. *CoRL*. https://arxiv.org/abs/2401.03360
- Noseworthy, M., et al. (2021). Active learning of abstract plan feasibility. *RSS*. https://arxiv.org/abs/2107.00683
- Orseau, L., et al. (2018). Single-agent policy tree search with guarantees. *NeurIPS*. https://arxiv.org/abs/1811.10928
- Ortiz-Haro, J., et al. (2023). Learning feasibility of factored nonlinear programs in robotic manipulation planning. *ICRA*. https://arxiv.org/abs/2210.12386
- Peng, X. B., et al. (2019). Advantage-weighted regression. *arXiv preprint*. https://arxiv.org/abs/1910.00177
- Piché, A., et al. (2019). Probabilistic planning with sequential Monte Carlo methods. *ICLR*. https://mlanthology.org/iclr/2019/piche2019iclr-probabilistic
- Ren, H., & Qureshi, A. H. (2024). Neural rearrangement planning for object retrieval from confined spaces perceivable by robot's in-hand RGB-D sensor. *ICRA*. https://arxiv.org/abs/2402.06976
- Silver, T., Chitnis, R., et al. (2021). Planning with learned object importance in large problem instances using graph neural networks. *AAAI*. https://arxiv.org/abs/2009.05613
- Smith, J. E., & Winkler, R. L. (2006). The optimizer's curse. *Management Science, 52*(3), 311–322. https://doi.org/10.1287/mnsc.1050.0451
- Sun, Z., & Yang, Y. (2023). DIFUSCO: Graph-based diffusion solvers for combinatorial optimization. *NeurIPS*. https://papers.neurips.cc/paper_files/paper/2023/hash/0ba520d93c3df592c83a611961314c98-Abstract-Conference.html
- Trabucco, B., et al. (2021). Conservative objective models for effective offline model-based optimization. *ICML*. https://proceedings.mlr.press/v139/trabucco21a.html
- Uesato, J., et al. (2022). Solving math word problems with process- and outcome-based feedback. *arXiv preprint*. https://arxiv.org/abs/2211.14275
- Wang, Z., et al. (2018). Active model learning and diverse action sampling for task and motion planning. *IROS*. https://arxiv.org/abs/1803.00967
- Wells, A. M., et al. (2019). Learning feasibility for task and motion planning in tabletop environments. *RA-L, 4*(2). https://kavrakilab.org/publications/wells2019learning-feasibility-for-tmp.html
- Wilfong, G. (1991). Motion planning in the presence of movable obstacles. *Annals of Mathematics and Artificial Intelligence, 3*, 131–150. https://doi.org/10.1007/BF01530890
- Yang, Z., et al. (2023). Compositional diffusion-based continuous constraint solvers. *CoRL*. https://arxiv.org/abs/2309.00966
