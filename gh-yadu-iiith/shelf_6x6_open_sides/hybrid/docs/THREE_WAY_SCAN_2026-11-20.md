# Three-way scan (WHY / HOW / WHAT): papers from the team's reading list (plain-language version)

**Date**: 2026-11-20
**Mode**: ARS deep-research `three-way-scan`

**AI Disclosure**: This scan was produced with AI-assisted research tools. Each paper was identified and read through its arXiv HTML or publisher page, as stated per entry. Full-text reads went through a summarising fetch tool, so check quotes and table numbers against the PDFs before citing them.

---

## The main takeaways for our project

- **Papers that verify moves beat papers that trust a score.** M4M, E-M4M and Lazy Rearrangement all check moves with simulation or motion planning. Our method commits to a plan on a learned score alone, and that is where our move-1 failures come from.
- **A learned score is safe when it only orders the search.** In E-M4M, the learned estimate decides what to try first, but every step is still verified. An overconfident score then costs time, not failed plans.
- **Cheap reachability checks may help.** Lazy Rearrangement uses fast arm-reachability (IK) checks. These are a low-cost stand-in for our main failure, "arm cannot plan move 1".
- **Trusting a wrong model is provably risky.** Karimi et al. prove that optimizing against a wrong model gives actions that fail. This matches our 97% predicted vs 55-62% real result.
- **Open gap.** In the sections read, no paper tests whether its success estimate stays calibrated under its own selection pressure. Our Phase 4 experiments aim at exactly that.

**Our method (the comparison frame):**
- The plan is chosen by a learned whole-plan success score: the Boltzmann X/Y model, with cVAE proposals.
- The plan is executed once, with no motion-planner calls while planning.
- Measured so far: predicted success ~97% vs real 55-62%. Failures split into "move not executable" and "target still blocked".

---

## Plain-words glossary

| Term | Simple meaning |
|---|---|
| WHY / HOW / WHAT | The paper's problem / its method / its results. |
| Boltzmann X/Y model | Our success model. X = moves executable. Y = target reaches goal. |
| cVAE | A model that proposes where to place objects. |
| CBS (Conflict-Based Search) | A multi-agent pathfinding method. It plans paths for several "agents" so they don't collide. |
| NGR (negative goal region) | The area the arm sweeps through. Objects there must be moved away. |
| IK (inverse kinematics) | Working out joint angles so the arm reaches a spot. Used as a quick reachability test. |
| Motion planning | Finding a collision-free arm path. Slow, so papers try to call it less. |
| PyBullet | A physics simulator. |
| RRT | A standard randomized path planner. Used here as a baseline. |
| PR2 | A real robot. "PR2: 12/16" means 12 of 16 real-robot trials worked. |
| Monotone plan | Each object is moved at most once. Non-monotone plans may move an object twice. |
| Best-first search | Always expand the most promising option next. |
| Naive Bayes | A very simple learned probability estimate. |
| Backjumping | When a branch fails, jump back to the real cause instead of undoing one step at a time. |
| Counterfactual explanation | "What would need to change to flip the decision?" It says where to get to, not how. |
| SCM (structural causal model) | A map of which variables cause which. |
| Recourse | The actions someone can take to get a better decision. |
| Energy-based model | Gives each layout a "cost". Low cost means good. |
| Langevin dynamics | Repeatedly nudging a layout toward lower cost, with a little noise. |
| Factor graph | A way to write a problem as many small constraints combined. |
| Minimum set cover | Pick the fewest items that cover everything. Here: the fewest blockers to clear. |
| Hopfield network | An older kind of neural net that settles to a low-energy state. |
| Ablation | Remove one part and see how much worse the results get. |

---

## Paper-by-paper

### M4M: Planning for Complex Non-prehensile Manipulation Among Movable Objects by Interleaving Multi-Agent Pathfinding and Physics-Based Simulation

Source: ICRA | Year: 2023 | Link: https://arxiv.org/abs/2303.13352

- **What it does:** Gets an object out of a cluttered shelf by pushing other objects aside, within physics limits.
- **Why it matters for us:** Like us, it decides which objects go where without physics calls. Unlike us, it verifies every move before committing. Our move-1 failures lack exactly that.

| Part | Detail |
|---|---|
| WHY | Retrieving an object from a cluttered shelf by pushing other objects aside, under physics limits. |
| HOW | CBS decides which objects leave the arm's swept NGR and where they go. It makes no simulator calls. Each push is then checked with IK, motion planning and PyBullet simulation. The first valid push is committed, and CBS replans. No learning. |
| WHAT | 92/73/62% success with 5/10/15 objects, vs RRT 55/40/26%. PR2: 12/16. |

**Method weaknesses** (empirical; read: full text via arXiv HTML)
- checked: found: Research hypothesis clarity; Variable operational definitions; Internal validity; External validity; Conclusion conservatism | checked: none found: Statistical reporting completeness | not checked: none
- Greedy commitment to the first valid push. This distorts the result when an early push blocks every later solution. [author-acknowledged, quote, Sec. VI: "This greedy behaviour makes M4M incomplete"]
- Planning times cover successful runs only. This favours methods that solve fewer, easier problems. [author-acknowledged for the successful-only timing, Sec. V; the distortion is reader-inferred]
- Validity depends on the simulator's friction model. [author-acknowledged, real-world section]

### E-M4M: Planning for Manipulation among Movable Objects: Deciding Which Objects Go Where, in What Order, and How

Source: ICAPS 33(1):668–676 | Year: 2023 | Link: https://arxiv.org/abs/2303.13385

- **What it does:** Searches over whole rearrangements, best option first. A small learned estimate guides which to try first. Every step is still checked in simulation.
- **Why it matters for us:** The closest analogue to us: a learned success probability guides search. But the learned score only orders the search and never replaces verification. An overconfident score costs time, not failed plans.

| Part | Detail |
|---|---|
| WHY | M4M is greedy and never reasons about order or alternative rearrangements. |
| HOW | Best-first search over rearrangements. Every edge is a simulated, accepted push. CBS proposes the goals. Caches remember invalid goals and successful pushes. A learned Naive Bayes estimate of P(this node leads to a solution) is used only as the search priority. |
| WHAT | Easy 97/98, Medium 45/63, Hard 15/39 vs M4M 78/25/7. Removing the learned priority gives 85/36/13. Removing the caches gives 82/25/7. PR2: 15/20. |

**Method weaknesses** (empirical; read: full text via arXiv HTML)
- checked: found: Research hypothesis clarity; Variable operational definitions; Internal validity; External validity; Conclusion conservatism | checked: none found: Statistical reporting completeness | not checked: none
- The priority is trained on scenes from the same generator as the test scenes. The benefit is overstated if they overlap or are more alike than real scenes. [reader-inferred]
- Monotone plans only. It misses problems that need an object moved twice. [author-acknowledged, "What E-M4M Can and Cannot Solve"]
- Fixed retrieval trajectory and NGR. This distorts the result when another approach path needs fewer moves. [author-acknowledged, "Main Algorithm"]

### Lazy Rearrangement Planning in Confined Spaces

Source: ICAPS 32(1):385–393 | Year: 2022 | Authors: Wang, Gao, Yu, Bekris | Link: https://ojs.aaai.org/index.php/ICAPS/article/view/19824 (arXiv 2203.10379)

- **What it does:** Plans object orders cheaply first, and runs the slow motion-planning check only when a full plan looks ready.
- **Why it matters for us:** Its IK reachability constraints are a cheap proxy for our main failure ("arm cannot plan move 1"). "Lazy verify, then backjump" sits between our zero calls and E-M4M's check-every-edge.

| Part | Detail |
|---|---|
| WHY | On shelves, objects can only be grasped from the side and block each other. Motion-planning checks dominate the runtime. |
| HOW | Cheap IK reachability constraints. A tree search over object orderings with forward checking. Motion-plan verification only when a branch reaches the goal, with backjumping if it fails. Buffer positions for non-monotone cases. No learning. |
| WHAT | 100% at 12–14 objects (monotone and non-monotone). 82–86% at 15–16 non-monotone. Verification still takes ~96% of the time. |

**Method weaknesses** (empirical; read: full text via arXiv HTML)
- checked: found: Research hypothesis clarity; Variable operational definitions; Internal validity; External validity; Conclusion conservatism | checked: none found: Statistical reporting completeness | not checked: none
- Uniform cylinders on a discrete grid. May not hold for mixed shapes or tight clearances. [reader-inferred; partly author-acknowledged via the clearance bound in Theorem 2]
- Every object has a fixed goal. Not comparable with tasks where placements are free. [reader-inferred]

### Improved-M4M (IEEE 10611234)

- **What it does:** Unknown. Not read.
- **Why it matters for us:** Cannot say yet.
- Not read: the IEEE page was empty and no open version was found. **Method weaknesses**: not assessed (read scope: none). Needs the exact title or DOI.

### Algorithmic Recourse: from Counterfactual Explanations to Interventions

Source: FAccT | Year: 2021 | Authors: Karimi, Schölkopf, Valera | Link: https://arxiv.org/abs/2002.06278

- **What it does:** Finds the cheapest set of real actions that would flip a model's decision, using a map of cause and effect.
- **Why it matters for us:** Same shape as Algorithm 1: the cheapest repair subject to success. Its central caveat, an optimiser trusting a wrong model, is our overconfidence result.

| Part | Detail |
|---|---|
| WHY | Counterfactual explanations say where to get to, not how. Acting on them ignores causal effects between variables. |
| HOW | Actions are structural interventions in a causal model (SCM). MINT finds the minimum-cost set of interventions that flips the decision. |
| WHAT | Proposition 3.1: counterfactual-based actions guarantee recourse only when the changed variables have no causal descendants. They can also cost 39–65% more. Recourse with a misspecified SCM is left open. |

**Method weaknesses** (theoretical/conceptual; read: arXiv HTML v4, nearly complete)
- checked: found: Conceptual definition precision; Testability | checked: none found: Argument logic structure | not checked: Counterargument handling; Theoretical novelty
- Guarantees assume the true SCM is known. They no longer hold when the graph or equations are learned. [author-acknowledged, quote: "the underlying causal model is rarely known"]
- ℓ1 range-normalised cost. The "% more costly" comparisons change if effort is not additive per feature. [author-acknowledged, cost design deferred to "cross-disciplinary" work]

### Energy-based Models are Zero-Shot Planners for Compositional Scene Rearrangement

Source: RSS | Year: 2023 | Authors: Gkanatsios, Jain, Xian, Zhang, Atkeson, Fragkiadaki | Link: https://arxiv.org/abs/2304.14391

- **What it does:** Builds a goal layout by adding up one learned "cost" per spatial relation (such as "left of"), then lowering the total cost. Local policies carry it out, with retries.
- **Why it matters for us:** A template for the team's factor-graph extension: one energy per constraint, summed. It also shows what happens without a feasibility term or calibration.

| Part | Detail |
|---|---|
| WHY | Planners fail on new compositions of spatial instructions. |
| HOW | One learned energy per spatial predicate. The energies are summed and minimised with Langevin dynamics to get a goal layout. Local policies execute with closed-loop retries. |
| WHAT | Composed tasks 77.2 vs 38.6 for an LLM planner. Removing goal generation drops this to 42.1. Ordering and physical feasibility are left open. |

**Method weaknesses** (empirical; read: arXiv HTML full text)
- checked: found: Variable operational definitions; Internal validity; External validity; Statistical reporting completeness | checked: none found: Research hypothesis clarity | not checked: Conclusion conservatism
- No collision, reachability or physics energy, so goals can be infeasible. [author-acknowledged, Sec. IV-E]
- Energies are never checked as probabilities. [reader-inferred]
- No seeds or CIs. About 10 real-robot tasks per setting. [reader-inferred, Tables I–VI]

### Approximating Minimum Set Cover in a Hopfield-Style Network

Source: Information Sciences 98:203–216 | Year: 1997 | Authors: Kaznachey, Jagota | Link: https://www.sciencedirect.com/science/article/pii/S0020025596002149

- **What it does:** Not known. Only the title was read. The title says it approximates minimum set cover with a Hopfield-style network.
- **Why it matters for us** (from the title only): It is the classic way to write "which blockers to clear" (minimum set cover) as a binary energy. One binary "clear" variable per blocker fits a factor-graph or Boltzmann extension.
- Title, authors and journal were confirmed through Crossref. This is the real title of the "Hopfield networks as minimal set covers" entry. The abstract and full text were unavailable (paywall).
- **Method weaknesses**: not assessed (read scope: metadata only).

---

## Cross-paper synthesis

| Question | Answer |
|---|---|
| Common WHY | Deciding which blocking objects to move, where, and in what order, when feasibility checks or model accuracy dominate. |
| Divergent HOW | Analytical planners (M4M, E-M4M, Lazy Rearrangement) verify moves with simulation or motion planning, and learn at most a search priority. Energy and recourse methods optimise a model-based objective. Our method commits to a plan on a learned score alone. |
| Strongest WHAT | E-M4M's ablations: a learned priority helps when every edge is still verified. Lazy Rearrangement: 82–100% at 12–16 objects, by verifying lazily. Karimi et al.: a formal proof that optimising against a wrong model yields actions that fail. |
| Unresolved global gap | In the sections read, no paper measures whether its learned or heuristic success estimate is calibrated under the optimiser's own selection pressure, or how many checks it takes to recover. That is the question our Phase 4 experiments (cautious scoring, lazy retries counted in motion queries, the per-step model) address directly. |
