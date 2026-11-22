"""EXPLORATORY calibration + "learned eta" check (no new compute; stored V3.4 decisions + teacher labels; VAL and post-hoc TEST).

For every root where an arm admits a repair under the frozen rule (p >= 0.3, a plan exists, teacher label known):
  * ECE (10 equal-width bins) and Brier of the plan's predicted success p vs the teacher's verdict;
  * temperature scaling: fit T on VAL (logit p / T, NLL), report TEST ECE before/after;
  * risk-targeted eta ("learned eta"): on a calibration split pick the lowest eta whose admitted plans fail <= target r,
    then report realised failure rate + coverage on TEST, both in-domain and across domains (2D VAL -> 3D TEST, and back).
Only plans with p >= 0.3 are visible (below that the stored rows keep no plan), so everything is conditional on p >= 0.3.
    python calib.py   -> runs/explore/calib/report.json
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402  (frozen; read-only)
import numpy as np  # noqa: E402

OUT = H.HZ.ABF.parent / "explore/calib"
SEEDS, RISKS = (0, 1, 2), (0.05, 0.10, 0.20)
import collections  # noqa: E402
UNK = collections.Counter()   # admitted plans dropped because the teacher label is UNKNOWN


def pairs(dom, split, arm):
    """-> arrays p, y (1 = teacher SUCCESS) over seeds x roots with an admitted, labelled repair."""
    L, P, Y = H.labels(dom), [], []
    for s in SEEDS:
        rows = H.ours(dom, split, s) if arm == "ours" else H._full_rows(dom, split, s)
        for r in rows:
            if arm == "ours":
                seq, p = r["decisions"][0], r["p11"][0]
            else:
                seq, p = (r["seq"], r["score_p11"]) if r["decision"] == "REPAIR" else (None, None)
            if not seq:
                continue
            o = H.V.outcome(seq, L, r["root_id"])
            if o == "UNKNOWN":
                UNK[(dom, split, arm)] += 1
                continue
            P.append(p); Y.append(o == "SUCCESS")
    return np.array(P, float), np.array(Y, float)


def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).sum() for b in range(bins) if (idx == b).any()) / len(p))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_T(p, y):
    z, best = logit(p), (1e9, 1.0)
    for T in np.exp(np.linspace(np.log(0.1), np.log(10), 400)):
        q = np.clip(1 / (1 + np.exp(-z / T)), 1e-9, 1 - 1e-9)
        nll = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        best = min(best, (nll, T))
    return best[1]


def scaled(p, T):
    return 1 / (1 + np.exp(-logit(p) / T))


def risk_eta(p, y, r, ucb=False):
    """lowest eta (>= 0.3, grid 0.01) whose admitted plans (p >= eta) fail at most r; None if no eta does.
    ucb=True: require the 95% Clopper-Pearson UPPER bound of the failure rate <= r, with n divided by 3 because the 3 seeds
    repeat the same roots (conservative clustering) -> the promise should then hold on new scenes."""
    from scipy.stats import beta
    for eta in np.round(np.arange(0.30, 1.0, 0.01), 2):
        m = p >= eta
        if m.sum() < 20:
            continue
        f = 1 - y[m].mean()
        if ucb:
            n = max(int(m.sum() / len(SEEDS)), 1); k = round(f * n)
            f = 1.0 if k >= n else float(beta.ppf(0.95, k + 1, n - k))
        if f <= r:
            return float(eta)
    return None


def at(p, y, eta, n_all):
    if eta is None:
        return None
    m = p >= eta
    return {"eta": eta, "fail_rate": round(100 * (1 - y[m].mean()), 2) if m.any() else None, "coverage": round(100 * m.sum() / n_all, 1)}


def main():
    D = {(d, s, a): pairs(d, s, a) for d in ("2d", "3d") for s in ("val", "test") for a in ("ours", "full")}
    rep = {"label": "EXPLORATORY (VAL fit, TEST post-hoc); conditional on p >= 0.3", "n": {"/".join(k): len(v[0]) for k, v in D.items()}}
    for d in ("2d", "3d"):
        for a in ("ours", "full"):
            (pv, yv), (pt, yt) = D[(d, "val", a)], D[(d, "test", a)]
            T = fit_T(pv, yv)
            rep[f"{d}_{a}"] = {"val_ece": round(ece(pv, yv), 4), "test_ece": round(ece(pt, yt), 4), "test_brier": round(float(np.mean((pt - yt) ** 2)), 4),
                               "T_fit_on_val": round(float(T), 3), "test_ece_after_T": round(ece(scaled(pt, T), yt), 4),
                               "test_fail_rate_admitted_at_0.3": round(100 * (1 - yt.mean()), 2)}
            for r in RISKS:
                for src in ("2d", "3d"):           # calibrate eta on src VAL, apply to d TEST (src != d = zero-shot transfer)
                    ps, ys = D[(src, "val", a)]
                    rep[f"{d}_{a}"][f"risk{int(r * 100)}_from_{src}_val"] = at(pt, yt, risk_eta(ps, ys, r), len(pt))
                    rep[f"{d}_{a}"][f"risk{int(r * 100)}_from_{src}_val_UCB95"] = at(pt, yt, risk_eta(ps, ys, r, ucb=True), len(pt))
    rep["unknown_dropped"] = {"/".join(k): v for k, v in UNK.items()}
    rep["note"] = "coverage = % of plans admitted at eta 0.3 (not of all roots); no data below p 0.3, so transfer is one-sided"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    p, y = np.array([0.1, 0.9, 0.9, 0.1]), np.array([0., 1., 1., 0.])
    assert abs(ece(p, y) - 0.1) < 1e-9 and risk_eta(np.full(30, 0.8), np.ones(30), 0.05) == 0.3   # self-check
    main()
