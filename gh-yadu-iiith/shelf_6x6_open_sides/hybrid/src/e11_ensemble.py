"""E11 (development, VAL): equal-weight ensemble of the three plan-success models as the beam-search scorer.

score(plan) = mean( hazard 3-seed mean, factor 3-seed mean, outcome_B seed mean )  -- probabilities, equal weight per model.
Same search, move set (all free cells), KEEP and lazy retries as phase4's `hazard` arm.  New file on purpose: phase4.py
is frozen for E9v2 (docs/E9v2_FREEZE.sha256), so this only registers an extra planner at run time.

    cd approachb_3d_learning && python ../hybrid/src/e11_ensemble.py run --workers 6
"""
import argparse, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase4 as P4  # noqa: E402

NAME = "ens3"


def picks():
    import torch, hazard, factor, outcome_b
    f = P4.OUT / f"plans_{NAME}.json"
    if f.exists():
        return
    Mm, models = P4.P3.Models(), [hazard.Hazard(), factor.FactorModel(), outcome_b.OutcomeB()]
    extra = lambda r, s: torch.stack([m(r, s).mean(0) for m in models])  # noqa: E731  (3 x n; plans_for takes .mean(0))
    torch.manual_seed(0)
    gen, out, t0 = torch.Generator().manual_seed(0), [], time.time()
    for p in json.loads((P4.P3.OUT / "picks_plan.json").read_text()):
        raw = Mm.load(p["root"])
        goal, cells0 = tuple(raw["goal"]), {k: tuple(v) for k, v in raw["initial_cells"].items()}
        sc = P4.plans_for("hazard", Mm, raw, cells0, sorted(o for o in cells0 if o != "t0"), goal, gen, extra)  # hazard branch = scorer-agnostic
        ranked = sorted(sc, key=sc.get, reverse=True)[:P4.KEEP]
        out.append({"root": p["root"], "K": p["K"], "plans": [[[o, list(c)] for o, c in s] for s in ranked],
                    "scores": [round(sc[s], 5) for s in ranked]})
    f.write_text(json.dumps(out))
    print(json.dumps({"planner": NAME, "roots": len(out), "s": round(time.time() - t0)}), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"]); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    P4.ALL = P4.ALL + (NAME,)
    if a.cmd == "run":
        picks()
        P4.ONLY = {NAME}
        a.planners = "hazard"  # already has plans: cmd_run's picks step is a no-op
        P4.cmd_run(a)
    P4.ONLY = {NAME, "hazard", "factor", "outcome_B", "team_len_a0.2"}
    P4.cmd_summary(a)
    import shutil; shutil.copy(P4.OUT / "summary.json", P4.OUT / "summary_E11_ens3.json")
