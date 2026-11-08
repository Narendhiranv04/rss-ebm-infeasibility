# R3: Future scientific evidence: frozen challenge split and candidate-diverse pilot manifests (PREPARED, NOT RUN)

**Verdict: R3 PASS (preparation only).** No physics was run for anything in this report.

## 1. Hard-order challenge cohort: FROZEN

**File:** `r3/CHALLENGE_SPLIT_FROZEN.json` (rows sha256 `1cce1f60fe875904…`).

**Reproducibility:** `challenge_cohort.py` re-run on the pinned store gives identical rows.

**Cohort:** the 75 non-cohort DEPENDENT roots (`independent_moves = False`; K2 28, K3 40, K4 7).
- 72 / 75 of their same-endpoint groups are **relocation-discordant**.
- The cohort is selected *for* order dependence, so no order-effect prevalence may be inferred from it.

**Membership rule** (mechanical):
- **Lineage link** = shared parent family or identical layout with a cohort root.
- A linked root inherits the **strictest** linked split: 41 → train, 10 → val, 5 → test.
- The 19 unlinked roots are split by `sha256('BWL_CHALLENGE|'+root)`: 40 % holdout, the rest alternating dev-train / dev-val.

**Result: CHALLENGE_DEV_TRAIN 47 · CHALLENGE_DEV_VAL 15 · CHALLENGE_HOLDOUT 13.**

**Disclosure:**
- CHALLENGE_HOLDOUT is **aggregate-inspected**: its stored composite outcomes were counted by the reuse audit and by this script. It is **not a pristine blind test**.
- It is never merged with the frozen 1,582-root test split.
- No authentic X / Y exists for these roots, and collecting it is not authorized.

## 2. Candidate-diverse ordered X/Y manifests

These are separate from the historical, construction-informed catalogue.

| | 3D | 2D |
|---|---|---|
| File | `r3/CANDIDATE_DIVERSE_3D_MANIFEST.json.gz` (rows sha256 `26256240e766ca55…`) | `r3/CANDIDATE_DIVERSE_2D_MANIFEST.json.gz` (rows sha256 `d801ec445fdef14d…`) |
| Generator | `src/r3_manifest3d.py`; GPU inference on Blackwell under the guard | `physics_pilot/r3_manifest2d.py` (pure Python) |
| Destination sources | **samplers only, on the current branch-local edited scene**: cvae_s0 + ranker ('cvae+rank', P = 2M), band-stratified + ranker ('strat+rank'), latent-free categorical cat_s0 ('cat') | **declared non-learning baseline** (the Q2 sampler is not merged or trained): band-stratified + uniform, drawn in the branch scene |
| Construction-informed fields used | none (`construction_informed: false` on every row; no parking, intended, blocker or teacher outcome) | none |
| Roots | 40 cohort roots, 32 train / 8 val (8 + 2 per K0–K3), disjoint from the S1 / R2 pilot roots | 40 Q2 corpus roots, 32 train / 8 val (8 + 2 per verified class 0–3), not pilot or dev-touched; **no root shared with historical 2D X/Y** |
| X queries / unique Y endpoints | 1,123 / 958 | 1,199 / 1,006 |
| Designs | NO_OP 40 · ONE_MOVE · ALT_DEST · PAIR_AB / PAIR_BA_SAME_ENDPOINT · OMIT_B and OMIT_A ("B without A") · TRIPLE + reversed · NUISANCE_PLUS_PAIR | same set |
| Same-endpoint order groups | **205** | **233** |
| Moves per query (m) | 1: 696, 2: 356, 3: 71 | 1: 704, 2: 400, 3: 95 |
| Estimated cost | X ≈ 7 CPU-h (fresh execution: about 10 CPU-s per move of planning + execution, plus about 8 CPU-s connector + replay, from the S1 / R1 measurements); Y planning from R2 | ≈ 5 CPU-h (2D BXY teacher, 8.1 CPU-s per query including replay, measured in P-min) |

**Design rules applied to both:**
- **Every prefix is its own stopping query.** Low-Y prefixes stay as extendable branches; X has no monotone requirement.
- **AB / BA** pairs are kept only when both orders are legal and reach one shared endpoint (one Y node).
- **Order outcomes are not engineered.** AB and BA may both succeed, differ, or both fail, and the analysis must report all three.
- **Unnecessary moves stay visible.** One-move stopping queries sit next to the pairs, so the cost-aware selector can prefer a single sufficient move to an unnecessary second one.

**When Q2 is merged and a 2D sampler is trained,** a learned-proposal 2D manifest is generated as a **separate** file; the non-learning one stays as a control.

## 3. Placement confirmation vs sequence trials: kept separate

**Individual-placement confirmation:** `reports/results_bwl/PHYSICS_CONFIRMATION_MANIFEST.json.gz`.
- 138 TEST contexts, 2,666 tuples, 2,089 never tested.
- It validates **placement** proposals only.
- It is a **disclosed confirmation panel**: the TEST scenes' metrics were examined. It is not pristine blind evaluation.

**Ordered X / Y sequence trials:** the manifests above. They validate **high-level corrections**. The two are never substituted for each other.

**Sampler verdict:** offline remains **INCONCLUSIVE** (BWL-1). stratified+ranker, cVAE+ranker and the categorical control are co-equal proposal sources, and known-label offline precision is not physical success.

## 4. What remains unauthorized

None of these were run:
- the 3D or 2D candidate-diverse physics;
- the challenge-cohort collection;
- the placement confirmation panel;
- dynamic search.

Each needs an explicit budget. The estimates above are planning numbers, not measurements of these manifests.
