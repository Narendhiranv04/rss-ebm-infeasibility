"""P10b-P10e preregistration (P10B_PREREG_V1): roster, splits, endpoints, statistics, teacher policy, training / ablation plan,
D protocol, execution protocol and stage gates. Frozen (hashed, committed, pushed) before any roster teacher label."""
from __future__ import annotations

import config as cf

SEED0 = 2026111001
ALLOC = {"H1": (8, 4, 9), "H2a": (7, 3, 5), "H2b": (7, 3, 4), "H3": (8, 4, 9), "H4": (8, 3, 7), "H5": (7, 3, 5), "H7": (7, 3, 4),
         "H8": (8, 3, 4), "H10_3": (10, 5, 9), "H10_4": (10, 4, 9)}          # (train, validation, blind) independent roots


def roster() -> list:
    """180 roots. Batch 1 (P10b-1, 40 roots): per family 3 train + 1 validation. Batches 2-5 (P10b-2): the remaining roots,
    families interleaved round-robin, blind and non-blind mixed, in this fixed order. Seeds = SEED0 + index."""
    first, rest = [], []
    for f, (tr, va, bl) in ALLOC.items():
        lst = ["train"] * 3 + ["validation"]
        first += [(f, s) for s in lst]
        pools, rem = [["blind"] * bl, ["train"] * (tr - 3), ["validation"] * (va - 1)], []
        while any(pools):                                  # splits interleaved so a budget stop leaves a balanced set
            rem += [x.pop() for x in pools if x]
        rest.append([(f, s) for s in rem])
    inter = []
    while any(rest):
        for lst in rest:
            if lst:
                inter.append(lst.pop(0))
    out = []
    for i, (f, s) in enumerate(first + inter):
        out.append({"index": i, "family": f, "split": s, "seed": SEED0 + i, "wave": 0, "batch": 1 if i < 40 else 2 + (i - 40) // 35})
    return out


PREREG = {
    "version": "P10B_PREREG_V1",
    "frozen_models": {"A": "LOCAL-SEQUENCE GRU (abc8 'A'), factorized objective", "B": "ABC9K_KEYED_V1 B-full (prefix / predecessor)",
                      "C": "ABC9K_KEYED_V1 C-full (GRU order, replaces B's predecessor factor)", "SF": "factorized supervised control (abc8 'SF')",
                      "D": "D_BLOCKERS_V1 (d_blockers.py)", "inputs": "frozen mask-pooled DINOv2-small + object geometry + robot q0 + goal"},
    "unit_of_analysis": "independent reference root (constructor seed). All its orders, omissions, edits, D subsets and appearance "
                        "variants stay in one split; never resampled as independent replicates.",
    "families": "P10B_FAMILIES_V1 (p10b_scenes.py): H1, H2a (kappa-1), H2b (kappa0), H3, H4, H5 reused from P10A_MECHANISMS_V1; "
                "new H7 (enclosed destination), H8 (sweep barrier only, sector checker), H10_3 (grasp chain), H10_4 (chain + sweep barrier). "
                "Certificate-targeted construction is ADAPTIVE (sound static checkers, never teacher labels); H8 is accepted by the "
                "sector checker only (no pose screens). H9 = appearance variants (palette change, identical geometry) of validation and "
                "blind roots; labels reused by exact physical identity (not new observations).",
    "allocation": ALLOC, "total_roots": 180, "splits": {"train": 80, "validation": 35, "blind": 65},
    "training_data": {"full": "historical P8c learner view (178 train + 12 validation relabelled train; P8c test excluded) + P10a 14 "
                              "(disclosed development) + P10b train roots", "selection": "P10b validation roots only",
                      "short": "length-extrapolation copies: same sources minus every candidate with > 2 moves and every H10 root"},
    "statuses": "VERIFIED (1) / CERTIFIED (0, independently checked, scoped) / BUDGET (operational 0) / UNKNOWN, NOT_RUN, INELIGIBLE "
                "(missing). proof_grounded = all components VERIFIED or CERTIFIED; operational = >= 1 BUDGET.",
    "minimality": "verified_witness_length from the returned execution; minimum_required = k only relative to the frozen catalogue "
                  "when every shorter catalogue candidate is decided 0 (SOUND if all CERTIFIED, OPERATIONAL otherwise); else UNKNOWN",
    "teacher_policy": "P10B_S100_V1 (p10b_dataset.POLICY): OPNEG-v1 adjudication; seeds 100 CPU-s (kill 95), query 330 CPU-s / 900 s, "
                      "root 900 CPU-s, a censored seed ends the query (UNKNOWN), one worker; option B_FULL_SEEDS chosen before labels",
    "D": {"pool": "movable mechanism roles (<= 4); fixed bodies are context only", "parking": "p10b_scenes.parking (frozen rule)",
          "labels": "all 2^n parking subsets as SCENE_TARGET_ONLY queries (empty subset = the original task); d_blockers.minimal_sets; "
                    "union-participation mask, necessary = intersection; INCOMPLETE / NON_MONOTONE / INELIGIBLE -> mask UNKNOWN",
          "train_mask": "blocker loss only on COMPLETE PROOF-GROUNDED masks; operational masks evaluated separately",
          "grid": {"lambda_B": [0.5, 1.0, 2.0], "threshold": [0.3, 0.4, 0.5, 0.6, 0.7], "updates": 600, "lr": 1e-3},
          "thresholded": "conditional marginal P(b_i = 1 | y0 = 0, S) on roots predicted infeasible (p_feasible < 0.5); empty set otherwise",
          "controls": ["D-pairless (w = 0, retrained)", "geometry heuristic (d_blockers.heuristic)",
                       "A/B/C/SF counterfactual attribution only where the pool is the two edit roles AND the parking subsets equal zS[A], "
                       "zS[B], zS[A+B] (else UNSUPPORTED)", "single-edit score (weaker, labelled)"]},
    "endpoints": {
        "P1_TOP1": "blind obstructed roots: strict TOP1_COMMIT executed success (target + SAFE_RETREAT), paired by root; contrasts "
                   "B vs SF, C vs SF, A vs SF, B vs geometry-only; G+M = planner-assisted reference (not in the Holm family)",
        "P2_H1": "blind proof-grounded H1 four-tuples: |kappa_hat - kappa| per root, B vs A",
        "P3_ORDER": "blind proof-grounded same-endpoint order pairs (H3 + H10 permutations): order choice correct, C vs B",
        "P4_OMISSION": "blind decided leave-one-out omissions of witnessed repairs (H4, H10): necessity accuracy, B vs SF",
        "D_primary": "blind complete masks: per-object blocker F1 (macro by root), D vs D-pairless; secondary: vs heuristic; "
                     "D feasibility AUROC / NLL vs A/B/C/SF z0 on matched roots",
        "secondary": "common per-event NLL / accuracy / AUROC / Brier / ECE (root-prior marginals), kappa signs, H5 false-empty, H6 J, "
                     "H7 edit-vs-executable sign, H9 appearance invariance, length extrapolation (short vs full on 3-/4-move), TOP5 "
                     "(planner-assisted, never mixed with TOP1), complexity / coverage"},
    "statistics": {"ci": "paired root-cluster bootstrap 95 % (2,000 resamples, seed 0); seeds reported individually; arm value per "
                         "root = seed mean", "binary": "exact two-sided binomial on discordant roots", "multiplicity": "Holm over "
                         "P1-P4 primary contrasts at alpha 0.05 (each P has one primary contrast; P1 uses B vs SF as its primary)",
                   "power_rule": "< 5 decided independent roots = UNDERPOWERED (descriptive only); >= 10 recommended",
                   "power_note": "P10a per-root NLL SD ~ 0.96 -> ~29 roots for 0.5 nat at 80 %; per-mechanism blind counts 4-9 here "
                                 "will remain UNDERPOWERED for small effects", "equivalence": "never claimed from a CI crossing 0"},
    "calibration": "per checkpoint, temperature and admission threshold chosen on P10b validation roots (p7_eval.fit_temperature, "
                   "p8_select rule over {0.5, 0.7, 0.85, 0.95}); frozen with the predictions before unsealing",
    "selection_rule": "winner = arm with the highest blind TOP1 success that beats SF after Holm; else 'no demonstrated superiority'; "
                      "then H1, ORDER, OMISSION endpoints; then efficiency / coverage. Never P8c in-distribution NLL.",
    "training": {"updates": 400, "batch": 12, "optimizer": "AdamW 1e-3 / 1e-4, clip 10", "seeds": [0, 1, 2],
                 "selection": "validation own-likelihood NLL every 20 updates (P10b validation roots)",
                 "priority": ["full A/B/C/SF x3", "short A/B/C/SF x3", "D, D-pairless x3", "B-F2 order-invariant", "B-F4 no omission pair",
                              "B-F5 structured (no generic + HO)", "B-F5 quadratic (no HO)", "B-F3 no scene pair", "B visual-zero",
                              "B unique ontology", "J head / analytic static (selection-only, no retraining)"],
                 "rule": "ablations run in priority order while the training cap allows; the rest NOT_RUN"},
    "execution": {"TOP1_COMMIT": "one learned choice from the frozen catalogue (identity check, analytic static veto, calibrated admission, "
                                 "fewest moves, J, C_d, energy); one seeded episode of that repair + target + SAFE_RETREAT; no fallback; "
                                 "abstain = failure", "TOP5_PRECOMMIT": "secondary, planner-assisted, separate table",
                  "baselines": ["geometry-only TOP1 (analytic proposer, no planner ranking)", "G+M (planner-assisted)", "uniform"],
                  "cache": "identical (root, committed repair, episode seed) = one physical episode"},
    "gates": {"P10b-0": "audit PASS; D synthetic tests PASS; prereg hashed + pushed before roster labels; 0 physical calls in learned "
                        "inference; split / seed roster frozen",
              "P10b-1": "GO iff >= 70 % of required-contrast queries decided; >= 1 proof-grounded H1 and H3 root; >= 2 witnessed 3-move "
                        "and >= 2 witnessed 4-move roots (else H10 BLOCKED, others continue); >= 50 % complete D masks among roots with a "
                        "D pool; caps held", "P10b-2": "per 35-root batch: caps held, no FATAL; yield / power recomputed on non-blind only",
              "P10c": "all priority-1 runs finite; predictions + calibration frozen and committed before unsealing"},
    "caps": {"constructor_dev": 3000, "teacher": 24000, "training": 24000, "execution": 15000, "evaluation": 5000},
}


def freeze(path="runs/p10b/p10b0/P10B_PREREG.json") -> dict:
    from pathlib import Path
    import p10b_dataset as d
    p = {**PREREG, "roster": roster(), "teacher_execution_policy": d.POLICY,
         "contract_sha256": d._sha("instruction-phase-md/0038_P10b_P10e_2D_ABCD_Implementation_and_Scientific_Validation_Contract.md"),
         "sources": {f: d._sha(f) for f in ("p10b_scenes.py", "p10b_dataset.py", "p10b_prereg.py", "d_blockers.py", "p10_scenes.py",
                                            "tests/test_d_blockers.py", "teacher_opneg.toml", "config.toml", "config_v2_r4.toml")}}
    p["hash"] = cf.stable_hash(p)
    if Path(path).exists():
        raise SystemExit("P10b preregistration already frozen")
    cf.write_json(Path(path), p)
    return p
