# 10: BWL-2 — 3D X/Y adapter brought to the November 5 / 2D source family; parity, invariants, SYNTHETIC recovery

**Stage verdict: BWL-2 PASS.**

| Verdict field | Value |
|---|---|
| `XY_3D_MATH_AND_INTERFACE` | **PASS** |
| `XY_2D_PARITY` | **PARTIAL** |
| Real 3D X/Y training | **not run** (no authentic X exists) |

**Parity detail.** The probability kernel and every neural module and composition are numerically identical to the 2D source. The 3D-specific *input* definitions cannot be checked against 2D: the GEO features are occupancy proxies, and the reference pose is a declared constant. Those inputs are therefore declared rather than verified.

**Code.**
- `src/xy3d.py` `fe5ab0c3…`: kernel, views, model and arms.
- `src/xy3d_data.py` `0084025d…`: 3D views, GEO3D, audit and manifests.
- `src/xy3d_tests.py` `f31cb253…`.

**Raw outputs.** `results_bwl/unit_tests.json`, `synthetic_recovery.json`; figures `bwl2_synthetic_recovery.png`, `bwl2_shared_y_ab_ba_schematic.png`.

## 1. What changed (and what did not)

**Kept unchanged.** The exact kernel: `log_py`, `log_px`, `log_evidence`, `marginals`, `energy_root`, `brute_force` and `batched_log_evidence`, τ = 1.

**Removed.** The kickoff neural adapter, which differed from the source in three ways:
- a GRU over action tokens instead of the source's ideal-prefix predecessor-pair mixing;
- an F3 that saw a sampler-style scene encoding with no target, goal, edit or context features;
- a DIRECT computed as `σ(a + b)`, the forbidden imitation of a direct head.

**Replaced by `XYNet`.** A batched, padded re-expression of the source modules. It adds no new mechanism.

**Parity matrix.** Source = `latent-causal-tamp` `e113c6680` (`45b0fb05` is identical for these files).

| Component | 2D source (file:lines) | 3D (`xy3d.py`) | Verification |
|---|---|---|---|
| Kernel: P(Y), P(X \| Y), exact observed-data evidence, marginals, p_succ | `bxy_core.py` 28–78; vectorized `bxy_p11.py` 280–297 | unchanged | **exact** vs `bxy_core` (max error 0.0) and vs brute force (≤ 1e-15) |
| Per-object numerics (positions, flags, base polar coordinates, relative to goal, relative to target) | `abc_models.Root.numeric` 44–51 | `XYNet.numeric` (+ height and shape; no DINO) | module parity on 2D-format inputs |
| Encoder, pooling, ideal-prefix action tokens | `abc_models.Shared` 54–89 | `encode` / `pooled` / `actions` (padded objects masked) | parity ≤ 6e-17 |
| **F3**: edit-aware scene vector (moved, d, \|d\|, x0 tokens; pooled + moved + **target** token) → `b = scene([v_s, c])` | `abc7_models.Shared7.scene_vec` 42–48; `bxy_core.ModelXY.coeffs` 97–98 | `scene_vec`, `coeffs` | parity |
| **F2** (D, E, DIRECT): predecessor pairs u < t, `mix`, mean, `order_s([g, scene_vec(final)])` | `abc7_models.ModelBC7._order` 140–152 (kind B, order pred) | `order` (pred) | parity |
| **G** (order-invariant F2): sorted actions, S0 tokens, pairs over u ≠ t, same modules, `phi_X0` | `bxy_p17.Model._order` 105–113 | `order` (invariant); a stable object key fixes the canonical order | parity against the verbatim source function |
| **F5**: `w = xy([g, v_s, c])` in D and G; w = 0 in E | `bxy_core` 89–90, 103; `bxy_p11.ARMS` 45–49 | `coeffs` | parity |
| **E**: scene-aware `xs([g, v_s, c])` added to a; capacity matched | `bxy_p11.Model` 244–259 | `coeffs` (arm E) | parity; parameter counts equal for all arms (469,895) |
| **DIRECT-GEO**: success logit `a_q + xs + geoX + geoY_parent`, no-op logit `b_0`; success-event BCE (an observed 0 on either side decides 0) | `bxy_p11.success_events` 156–167, `root_terms` 300–309; `bxy_p17.Model.coeffs` 115–124 | `coeffs`, `loss`, `labels_for` | parity; independent heads (no `xy`; has `xs`) |
| GEO2 heads `geoY(13)` + `geoX(8)` | `bxy_p17` 80–91, 102–124 | same heads; **3D features = `xy3d_data.GEO3D`** | head parity; **feature semantics UNVERIFIED**, because no audited 3D access map exists (see §3) |
| Travel `J = softplus(jt([v, c])) + softplus(cost(g))`, `J0 = softplus(jt([v_0, c]))`; Huber λ_J | `bxy_v2.ModelXY2.j_hat` 172–174; `bxy_p11.batch_loss` 312–319 | `coeffs`, `loss` | parity |
| SYM auxiliary (κ = 1) + P14 contrast | `bxy_p17.sym` 133–151, `batch_loss` 154–177 | `sym_batch` (vectorized) | covered by the t1 gradient test; formula transcribed line by line |
| Selection rule (no-op gate, η, α m + β C_d + γ Ĵ) | `bxy_p11.choose` 387–398 | `choose` | no-op test |
| F1 legality, F4 omission | F1 analytic screen; F4 = new queries | F1 in the sampler / search; omission = new ordered query from S0 | t7 |

**Context vector c (3D).** It contains:
- the declared **q_ref** ready pose: hand 5 cm outside the shelf front, 0.20 m above the floor, azimuth 180°, gripper free (`shelf_scene.py` l.338–339). q0 = q_ref for every root, because the spec's `start_pose` is "home". Joint angles are not stored in the roots;
- the goal (x, y);
- the target's size and height;
- the robot-capability flags (arm ceiling, no overpass, return to ready).

**Learner-visible root fields** (`xy3d_data.visible`): object ids, shape, size, height, initial cells, goal and shelf geometry. Every other field of a raw root is privileged and never read.

## 2. Tests

**Unit tests** (`xy3d_tests.py unit`): CPU 7, float64, forward and backward only, no optimizer step. **10 / 10 groups pass** (2.0 s). The existing pytest suite passes 11 / 11.

| # | Contract test | Result |
|---|---|---|
| 1 | Four-cell sum = 1 (error 4e-16); evidence = brute force with missing X / Y (≤ 1e-15); `gradcheck`; every active head gets a gradient | PASS. DIRECT's travel heads get **no** gradient by design: the source has λ_J = 0 for DIRECT |
| 2 | AB / BA share one Y node; X differs in D and E (Δa ≈ 9e-3 at initialization) and is identical in G (0); duplicate X / Y queries are deduplicated with no extra loss weight (loss difference 0); conflicting duplicates raise | PASS |
| 3 | Batch vs single-root coefficients, permuted query order, caching, and the Y marginal after adding a new-order X to an existing endpoint: all differences < 1e-10 | PASS |
| 4 | w ≡ 0 gives P(X) = σ(a) and the uncoupled evidence (error 0). The F5 gradient is > 0 on a jointly decided root and **exactly 0** for a Y-only root (also for `order_s`, `pairo`, `mix`): F5 cannot be learned from Y-only labels | PASS |
| 5 | Observed-data evidence for neither / Y-only / X-only / both / several X children sharing Y equals brute force (error 0); "neither" contributes exactly 0 | PASS |
| 6 | Swapping the order changes a (D, E, DIRECT) and leaves the shared b unchanged; changing only the goal changes b (Δ 0.027); permuting objects with stable ids leaves b, a, w, J unchanged (0) in all arms, G included | PASS. This needed the stable object key; with only a slot sort, G changed by 1e-6 |
| 7 | **Real-format 3D records**: 24 real roots (6 per K0–K3) round-trip. Y0 is the `original` label; each D label lands on its own endpoint; B sequences become **unlabelled design** X nodes; F4 omissions are recomputed from S0; negatives are preserved; **0 real X labels** attached. All four arms run forward and backward on these roots with SYNTHETIC X labels: finite loss and gradients | PASS |
| 9 | DIRECT-GEO and G have matched inputs and independent heads. Changing every privileged field (roles, intended blockers, parking, construction, greedy blockers, mechanism pools, grasp azimuths, verified_K, observed, spec, label y / negative kind / witness / role / failed step) changes **no** coefficient of any arm (max change 0.0) | PASS |
| — | No-op: a root with only Y0 gives p_succ(()) = σ(b_0) with no X query; the gate works; the no-op carries no action cost. Connector trap: X_A = 0 next to X_AB = 1 are distinct nodes with a finite loss | PASS |
| 10 | **Source parity** against the imported `e113c6680` modules (`parity_ref/`, hashes checked): kernel error 0.0; coefficients b, a, w, J, J0 for **all four arms** differ by ≤ 5.6e-17 from the 2D source composition (`ModelBC7` + heads, p17 invariant `_order` verbatim) on random 2D-format roots | **`SOURCE_PARITY_VERIFIED`** |

**Legacy 2D compatibility.** No 2D file was edited; the 2D repository is untouched, so its scores and replays are unchanged by construction. The parity test imports read-only copies of `abc_models.py`, `abc7_models.py` and `bxy_core.py`, recreated with `git show e113c6680:<file>`.

### Test 8 — planted-coefficient recovery (**SYNTHETIC**, GPU 0, CPU 7)

**Setup.**
- Random 6×6 scenes in the 3D format, with the real GEO3D features.
- Planted truth:
  - F3 = target and goal access;
  - F2 = access in the ideal predecessor scene, plus a **pure order term** (tall object moved first), so the order effect does not come through the GEO features alone;
  - F5: w = 1.5 (coupled truth) or 0 (uncoupled truth).
- Data: 800 training, 200 early-stopping and 400 held-out synthetic roots. Y is observed with probability 0.8, X with probability 0.85.
- Training: the source optimizer (AdamW 1e-3, weight decay 1e-4, clip 10), 32 roots per batch, ≤ 1,500 updates, early stopping. Float32, 2 seeds.
- 16 runs in 116 s, peak 99.5 MB VRAM, 11 watchdog probes, 0 violations.

| Truth | Arm | Held-out joint NLL (planted truth) | Learned mean w | Order concordance of a on AB / BA pairs | Success log loss |
|---|---|---|---|---|---|
| coupled (w = 1.5) | **D** | **6.31 / 6.26** (6.12) | **1.27 / 1.29** | 0.86 / 0.91 | 0.643 / 0.649 |
| | E | 6.41 / 6.43 | 0 | 0.90 / 0.92 | 0.643 / 0.642 |
| | G | 6.38 / 6.37 | 1.22 / 1.35 | **0.50 / 0.50** | 0.654 / 0.653 |
| | DIRECT-GEO | n/a | n/a | 0.49 / 0.85 (seed-unstable) | 0.651 / 0.645 |
| uncoupled (w = 0) | D | 7.16 / 7.07 (6.90) | **−0.02 / −0.04** | 0.92 / 0.94 | 0.619 / 0.616 |
| | E | 7.10 / 7.08 | 0 | 0.93 / 0.95 | 0.620 / 0.617 |
| | G | 7.35 / 7.38 | −0.06 / −0.03 | **0.50 / 0.50** | 0.639 / 0.638 |
| | DIRECT-GEO | n/a | n/a | 0.93 / 0.87 | 0.615 / 0.620 |

**Reading (SYNTHETIC only).**
- **A planted F5 is identifiable.** D recovers w ≈ 1.28 of 1.5 (shrunk) and beats E by about 0.13 nats of joint NLL. When the truth has w = 0, D learns w ≈ 0 and matches E.
- **A planted order effect is separable from F5.** G cannot represent order (concordance exactly 0.5) and has the worst NLL under both truths, while D and E rank the executable order higher about 90 % of the time.
- **Coupling barely moves success-event log loss** (D ≈ E). This mirrors the 2D finding that the F5 advantage was not established, and is **not** evidence that F5 helps on physical data.
- These numbers are math and interface checks, not efficacy.

## 3. Declared limits of the 3D adapter

- **GEO3D** keeps the source's 13 + 8 feature structure: clearances and access of the endpoint at q_ref, and per-action pick / place access in the ideal predecessor scene, or in S0 for G. It uses **observable occupancy proxies**: open compass directions with two free hand cells, where open sides count as free and the back wall does not. The 2D features were teacher-derived grasp-mode counts from access maps, and no audited 3D access map exists. Replacing the proxies with a 3D access map derived from the teacher would be an input change, not a model change.
- **No visual features in 3D.** This is the 2D `vision=False` form: zero input to the same encoder width.
- **q_ref is a declared constant pose descriptor**, not joint angles. Because every root shares it, it acts only as a bias.
- **The training loop is not ported** (fold builder, root-batch sampler, inner rule selection). It belongs to BWL-6 and needs authentic labels first. `XYNet.loss` already implements the exact root-mean objective + λ_J L_J + κ SYM.
