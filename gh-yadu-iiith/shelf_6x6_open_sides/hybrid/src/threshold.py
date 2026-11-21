"""Abstain threshold eta (Algorithm 1, line 28: act only if score >= eta) chosen by 5-fold cross-validation over roots, per planner.

Utility per root: +1 if the executed first plan works, -lam if it fails, 0 if the planner abstains.  Every root here is solvable
(VAL, verified_K >= 1), so abstaining is never 'right'; it only avoids a failed attempt.  Also: isotonic recalibration (CV) of the
score, then the decision-theoretic rule act iff p_cal >= lam / (1 + lam).  Exploratory (VAL first-try results from Phase 3/3b/4).

    python ../hybrid/src/threshold.py            (run from approachb_3d_learning/)
"""
import json, random, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase4 as P4  # noqa: E402

LAMS, FOLDS = (0.5, 1.0, 2.0, 3.0), 5


def data():
    res = P4.results()
    out = {}
    for k in ("boltz_mean", "team_pipeline", "gcvae_pipeline", "factor", "hazard"):
        E = json.loads((P4.OUT / f"plans_{k}.json").read_text())
        out[k] = [(e["root"], e["scores"][0], res[json.dumps([e["root"], e["plans"][0]])]["success"]) for e in E]
    return out


def util(rows, eta, lam):
    return sum((1 if y else -lam) for _, s, y in rows if s >= eta)


def best_eta(rows, lam):
    cands = sorted({s for _, s, _ in rows}) + [2.0]  # 2.0 = always abstain
    return max(cands, key=lambda e: (util(rows, e, lam), -e))  # ties -> lower eta (act more)


def isotonic(rows):
    """pool-adjacent-violators on (score, y) -> step function score -> calibrated p"""
    pts = sorted((s, y) for _, s, y in rows)
    blocks = [[s, s, y, 1] for s, y in pts]  # lo, hi, sum y, n
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] / blocks[i][3] > blocks[i + 1][2] / blocks[i + 1][3]:
            a, b = blocks[i], blocks.pop(i + 1)
            blocks[i] = [a[0], b[1], a[2] + b[2], a[3] + b[3]]; i = max(i - 1, 0)
        else:
            i += 1
    return lambda s: next((b[2] / b[3] for b in blocks if s <= b[1]), blocks[-1][2] / blocks[-1][3])


def folds(rows):
    roots = sorted({r for r, _, _ in rows}); random.Random(0).shuffle(roots)
    f = {r: i % FOLDS for i, r in enumerate(roots)}
    return [([x for x in rows if f[x[0]] != k], [x for x in rows if f[x[0]] == k]) for k in range(FOLDS)]


def main():
    out = {}
    for k, rows in data().items():
        n, o = len(rows), {"roots": len(rows), "first_try_success": sum(y for _, _, y in rows)}
        for lam in LAMS:
            cv_u = cv_act = cv_ok = cal_u = cal_act = 0
            for tr, te in folds(rows):
                e = best_eta(tr, lam)
                cv_u += util(te, e, lam); act = [x for x in te if x[1] >= e]; cv_act += len(act); cv_ok += sum(y for _, _, y in act)
                cal = isotonic(tr)
                act = [x for x in te if cal(x[1]) >= lam / (1 + lam)]
                cal_u += sum((1 if y else -lam) for _, _, y in act); cal_act += len(act)
            o[f"lam={lam}"] = {"always_act_utility": util(rows, -1, lam), "cv_eta_utility": round(cv_u, 1), "cv_acted": f"{cv_act}/{n}",
                               "cv_success_when_acting": round(cv_ok / max(1, cv_act), 3), "eta_full_data": round(best_eta(rows, lam), 4),
                               "isotonic_rule_utility": round(cal_u, 1), "isotonic_acted": f"{cal_act}/{n}"}
        out[k] = o
    print(json.dumps(out, indent=1))
    (P4.OUT / "threshold.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    f = isotonic([("a", 0.1, 0), ("b", 0.2, 1), ("c", 0.3, 0), ("d", 0.9, 1)])
    assert f(0.1) == 0.0 and abs(f(0.25) - 0.5) < 1e-9 and f(0.95) == 1.0  # PAV pools the violating pair
    main()
