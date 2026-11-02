"""P10 pre-registration (frozen data; hashed into runs/p10/p10_0/P10_0_PREREG.json before any P10a teacher query or training
on new labels). Changing anything here after that point invalidates the P10 confirmatory status of the affected endpoint."""

PREREG = {
    "version": "P10_PREREG_V1",
    "unit_of_analysis": "independent reference root (constructor seed). Candidates, orders, omissions, scene edits, appearance variants and "
                        "near-duplicates of one root are dependent and never resampled as independent replicates.",
    "status_values": {"VERIFIED_FEASIBLE": "complete re-checked witness incl. target and SAFE_RETREAT (1)",
                      "CERTIFIED_INFEASIBLE": "independently checked sound certificate within its stated scope (0); pick-side only, no place-side certificate",
                      "BUDGET_INFEASIBLE": "policy-relative operational 0 under pinned OPNEG-v1 (all declared seeds clean), never a proof",
                      "UNKNOWN": "missing", "NOT_RUN": "missing"},
    "strata": {"proof_grounded": "every component outcome VERIFIED or CERTIFIED", "operational": "at least one component BUDGET_INFEASIBLE, none missing",
               "undecided": "any component UNKNOWN / NOT_RUN: excluded from the endpoint, counted in yield"},
    "kappa": "kappa = y(A+B) - y(A) - y(B) + y(0) on decided outcomes; predicted kappa_hat = p_AB - p_A - p_B + p_0 from ROOT-PRIOR marginals "
             "(no labels); predicted sign = +1 if kappa_hat > 0.5, -1 if < -0.5, else 0",
    "hypotheses": {
        "H1": {"mechanism": "complementarity: neither edit alone, both together enable the task (kappa = +1)",
               "primary_endpoint": "per root |kappa_hat - kappa| on proof-grounded complete four-tuples",
               "primary_contrast": "B-keyed vs A (paired by root)", "secondary": ["sign accuracy", "four-bit NLL", "C, SF, frozen B"]},
        "H2": {"mechanism": "redundancy (kappa = -1) and neutral (kappa = 0) controls", "primary_endpoint": "exploratory: same as H1",
               "secondary": ["H1 gains must not degrade H2 calibration"]},
        "H3": {"mechanism": "same endpoints, opposite action order, same q0",
               "primary_endpoint": "paired order choice correct: p(feasible order) > p(infeasible order), proof-grounded pairs",
               "primary_contrast": "C-keyed (GRU) vs B-keyed (prefix / predecessor)", "secondary": ["probability margin", "A", "SF"]},
        "H4": {"mechanism": "omission necessity / redundancy (omitted sequences evaluated independently from S0)",
               "primary_endpoint": "necessity accuracy: p(omitted repair) < 0.5 iff the omitted repair is infeasible, decided omissions of witnessed repairs",
               "primary_contrast": "B-keyed vs SF", "secondary": ["TOP1 picks the strictly shorter successful repair"]},
        "H5": {"mechanism": "originally feasible task with harmful / neutral edits", "primary_endpoint": "exploratory: TOP1 chooses the empty repair; z0-zS calibration"},
        "H6": {"mechanism": "alternative destinations / travel", "primary_endpoint": "exploratory: J error on witnessed relocations, selection"},
        "H7": {"mechanism": "hypothetical edit succeeds, executable relocation fails", "primary_endpoint": "exploratory: sign of p(zS[A]) - p(zR(A)) matches observed"},
        "H8": {"mechanism": "constructor-disjoint carry-sweep obstruction (not selected with pose screens)", "primary_endpoint": "exploratory (P10a); execution in P10d"},
        "H9": {"mechanism": "nuisance appearance, fixed geometry", "primary_endpoint": "deferred (no appearance variants in P10a)"}},
    "metrics": {"common_all_arms": "per-observed-event NLL, accuracy, AUROC, Brier, ECE of ROOT-PRIOR marginals on the same decided outcomes",
                "own_joint_BC_only": "exact root NLL of the observed bits under P_K", "per_candidate": "reported separately (P8 convention)",
                "offline_oracle_conditionals": "diagnostic only, never deployable", "selection": "TOP1 on the same admissible catalogue: success, abstain, "
                "false-empty, moves, known-catalogue regret (only where a VERIFIED best exists), predicted J error, C_d"},
    "calibration": "temperature (p7_eval.fit_temperature) and admission threshold (p8_select rule over {0.5, 0.7, 0.85, 0.95}) chosen on the 12 P8c "
                   "validation roots per checkpoint and frozen before any P10a label; never refit on P10a or disclosed test",
    "statistics": {"ci": "root-cluster bootstrap 95 % CI of the mean per-root paired difference (2,000 resamples, seed 0)",
                   "binary_paired": "exact two-sided binomial test on discordant roots", "multiplicity": "one primary each for H1, H3, H4 "
                   "(and the TOP1 execution endpoint in P10d); everything else exploratory",
                   "power_rule": "an arm-endpoint with < 5 decided independent roots is UNDERPOWERED: descriptive only, no inferential claim",
                   "equivalence": "never claimed from a CI crossing 0"},
    "arms": {"A": "LOCAL-SEQUENCE GRU, factorized objective", "B": "ABC9K_KEYED_V1 B-full (prefix / predecessor order)",
             "C": "ABC9K_KEYED_V1 C-full (GRU order)", "SF": "factorized supervised control", "uniform": "seeded uniform",
             "geometry_only_TOP1": "analytic pose geometry proposer, no IK (P10d)", "G+M": "planner-assisted reference, reported with its planner calls (P10d)",
             "historical": "frozen K-B-full#s1 (200 updates), P9 unique / PoE checkpoints"},
    "splits": {"training": "P8c 178 train roots", "selection": "P8c 12 validation roots", "disclosed_test": "24 P8c test roots: descriptive only",
               "P10a": "14 constructibility-pilot roots: certificate-targeted construction = ADAPTIVE; never a blind holdout"},
    "execution_protocols_P10d": {"TOP1_COMMIT": "primary: one learned choice, zero planner calls on alternatives; plan / execute only the choice",
                                 "TOP5_PRECOMMIT": "secondary, planner-assisted: up to 5 ranked candidates relocation-planned before commit (P8d)"},
}
