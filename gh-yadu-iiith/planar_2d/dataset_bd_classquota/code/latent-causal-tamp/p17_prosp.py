"""P17 retrospective re-analysis of the 48 P16 D test roots and the prospective p17test cohort: pre-registration, sealed workers,
custody."""
from __future__ import annotations

import os
from pathlib import Path

import config as cf
from p17_eval import OUT, rd, sha, P17_LABEL_BUDGET

from p17_data import B_LEARNED, B_SEEDS, DATA, _plan, label_roots


# ------------------------------------------------------------------ P17_REANALYSIS_OF_P16_ROOTS (retrospective; all 48 P16 D test roots)
R_GUIDES = ("Dp17", "Dp17-norec", "Dp17-blk", "Dp16", "Dp16-norec", "Dp", "D", "geometry", "geometry-norec", "uniform", "oracle")


def retro_keys() -> list:
    """The 48 P16 D test roots (every non-NAT root of the P16 test plan, DENSE included) in P16 plan order."""
    return [k for k, r in rd(Path("runs/p16/data/test/plan.json"))["roots"].items() if r.get("family") != "NAT"]


def retro_d(part, guides=R_GUIDES) -> None:
    import p16_eval as p16e
    import p17_exec as ex
    orc = {k: v[0] for k, v in p16e.test_oracles().items() if v[0] is not None}
    ex.run_roots("test", retro_keys(), part, "d", guides=list(guides), oracles=orc, tag="retro", claims=DATA / "test" / "retro_claims")


# ------------------------------------------------------------------ prospective cohort: pre-registration, sealed execution, custody
P_GUIDES = ("Dp17", "Dp17-norec", "Dp17-blk", "Dp16", "Dp16-norec", "Dp", "D", "geometry", "geometry-norec", "uniform")
B_EXEC_ARMS = B_LEARNED + ("LB+G@LB", "LB@LB+G", "LB#top3", "LB+G#top3", "empty_always", "random", "geometry", "verify_first")
N_NAT_D_CONTROLS = 12
CODE = ("p16_motion.py", "p16_common.py", "p15_search.py", "p17_search.py", "p17_verify.py", "p17_domain.py", "p17_b.py", "p17_d.py", "p17_exec.py",
        "p17_eval.py", "p17_analysis.py", "p16_eval.py", "p16_b.py", "d_blockers.py", "unique_factor.py", "unique_factor_keyed.py", "abc7_models.py")


def p_rosters() -> dict:
    """Frozen prospective rosters (plan order): B on every p17test root; D on every non-NAT root + the first N_NAT_D_CONTROLS NAT roots."""
    roots = _plan("p17test")["roots"]
    nat = [k for k, r in roots.items() if r.get("family") == "NAT"]
    return {"b": list(roots), "d": [k for k, r in roots.items() if r.get("family") != "NAT"] + nat[:N_NAT_D_CONTROLS]}


def prereg() -> dict:
    """runs/p17/PREREG.json: everything that decides a prospective outcome, hashed before any p17test label or execution exists."""
    import glob
    import p17_analysis as pa
    if any((DATA / "p17test").glob("sealed_*.json")):
        raise SystemExit("p17test outputs already exist")
    ck = sorted(glob.glob("runs/p17/b/ckpt/*.pt") + glob.glob("runs/p17/d/ckpt/*.pt"))
    out = {"cohort": "p17test", "plan": sha(DATA / "p17test" / "plan.json"), "rosters": p_rosters(),
           "b": {"arms": list(B_EXEC_ARMS), "seeds": list(B_SEEDS), "policies": rd(OUT / "b" / "POLICIES.json"), "policies_sha": sha(OUT / "b" / "POLICIES.json"),
                 "primary": list(pa.B_PRIMARY), "secondary": [list(x) for x in pa.B_SECONDARY],
                 "endpoints": "strict TOP1 on decided originals (exact two-sided sign test, root bootstrap CI); no-op / repair / admission reported separately; "
                              "Holm over the strict-TOP1 secondary family; VERIFY_FIRST physical calls and virtual CPU reported"},
           "d": {"guides": list(P_GUIDES) + ["oracle (after unsealing; P17 parking-lattice union from sealed labels)"], "primary": ["Dp17", "geometry"],
                 "endpoints": "strict success (found + replay CLEAR) on common roots; claims; nodes / virtual CPU to first plan; extra moves vs proven optima",
                 "calibration": {k: sha(f"runs/p17/d/calibration/{k}.json") for k in ("D", "Dp", "Dp16", "Dp17")}},
           "budgets": {"seq": __import__("p17_b").SEQ_BUDGET, "d_search": __import__("p17_exec").D_BUDGET, "label": P17_LABEL_BUDGET, "node_cap": 500,
                       "n_max": 4, "per_obj": 2, "planner_profile": "P16_FAST"},
           "checkpoints": {c: sha(c) for c in ck}, "training_data": {f: sha(f) for f in ("runs/p17/b/TRAIN.pt", "runs/p17/d/D_ITEMS.pt")},
           "code": {c: sha(c) for c in CODE},
           "order": "freeze + push -> sealed labels (all p17test roots) + sealed B / D execution -> sealed-output manifest hash commit + push -> unseal -> oracle guide -> analysis",
           "mechanism_tags": "post hoc from physics labels only; natural (NAT) and constructed strata reported separately",
           "power": {"minimal_important_effect": "5 percentage points strict TOP1 (B); any non-zero bounded-budget success difference (D)",
                     "pilot_discordance": {"P16 LB vs LB+G": "16 / 112 decided roots, 0:16", "P16 LB vs ALWAYS_EMPTY": "65 / 112, 25:40",
                                           "P16 Dp16 vs geometry": "0 / 21"},
                     "expectation": "B: ~150 decided roots -> ~21 LB vs LB+G discordant pairs; an exact sign test at 0.05 needs >= 16:5, power ~0.5 at a "
                                    "true 3:1 split and ~1 at the P16 0:16 split. D: P16 showed 0 discordant pairs; a D vs geometry contrast with < 6 "
                                    "discordant pairs is reported UNDERPOWERED, never as equivalence.",
                     "status_rule": "paired results with fewer than 6 discordant roots are labelled UNDERPOWERED"}}
    cf.write_json(OUT / "PREREG.json", out)
    return out


def p_queue(part, mode) -> None:
    """Sealed prospective worker: mode 'label' (sealed labels: D roster first, then every other B root), 'b' or 'd' (sealed execution);
    atomic per-root claims."""
    import p17_exec as ex
    if not (OUT / "PREREG.json").exists():
        raise SystemExit("not pre-registered")
    ro = p_rosters()
    if mode == "label":
        q = ro["d"] + [k for k in ro["b"] if k not in ro["d"]]                # D roster first, then the remaining B roots
        claims = DATA / "p17test" / "label_claims"
        claims.mkdir(parents=True, exist_ok=True)
        for k in q:
            try:
                os.close(os.open(claims / f"{k}.claim", os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
            label_roots("p17test", [k], part, sealed=True)
    elif mode == "b":
        ex.run_roots("p17test", ro["b"], part, "b", arms=list(B_EXEC_ARMS), sealed=True, tag="exec_b", claims=DATA / "p17test" / "claims_b")
    elif mode == "d":
        ex.run_roots("p17test", ro["d"], part, "d", guides=list(P_GUIDES), sealed=True, tag="exec_d", claims=DATA / "p17test" / "claims_d")
