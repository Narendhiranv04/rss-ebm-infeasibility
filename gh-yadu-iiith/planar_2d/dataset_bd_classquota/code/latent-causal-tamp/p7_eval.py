"""P7 evaluation for ABC7 models: per-root predictions, kappa, calibration (validation-only temperature),
abstention coverage, effects, order pairs, restricted-catalogue selection and root-cluster bootstrap CIs."""
from __future__ import annotations

import math

import numpy as np
import torch

import abc7_train as t7
from abc_models import T

TAUS = np.exp(np.linspace(math.log(0.25), math.log(8.0), 61))
L_W = 0.64


def _cd(r, seq):
    return sum(float((torch.as_tensor(d, dtype=torch.float64) - r["R"].xy0[i]).norm()) for i, d in seq) / L_W


@torch.no_grad()
def evaluate(m, r) -> dict:
    outs, eff = t7.forward(m, r), []
    t7.losses(m, r, outs, eff)
    sp = {k: float(v) for k, v in t7.scene_probs(m, r, outs).items()}
    rows = []
    for c, o in zip(r["cands"], outs):
        row = {"query": c["query"], "m": len(c["seq"]), "y": c["y"], "source": c["src"], "u": float(o["u"]), "p": float(o["p"]),
               "energy": float(o["E"]), "J_pred": float(o["J"]), "J_obs": c["cost"], "C_d": _cd(r, c["seq"])}
        if "marg" in o:
            row |= {"slots": o["slots"], "marg": [float(x) for x in o["marg"]], "z_obs": c["z"]}
        rows.append(row)
    ys = {"z0": next((c["y"] for c in r["cands"] if not c["seq"]), None)} | r["scene_y"]
    kap = None
    if all(ys.get(k) is not None and k in sp for k in ("z0", "zS[A]", "zS[B]", "zS[A+B]")):
        f = lambda d: d["zS[A+B]"] - d["zS[A]"] - d["zS[B]"] + d["z0"]  # noqa: E731
        kap = {"pred": f(sp), "obs": f(ys), "scene_pred": {k: sp[k] for k in ("z0", "zS[A]", "zS[B]", "zS[A+B]")},
               "scene_obs": {k: ys[k] for k in ("z0", "zS[A]", "zS[B]", "zS[A+B]")}}
    keyed = {c["query"]: c for c in r["cands"]}
    order = [{"orders": [a["query"], b["query"]], "y": [a["y"], b["y"]], "u": [a["u"], b["u"]], "energy": [a["energy"], b["energy"]]}
             for a in rows for b in rows if a["query"] < b["query"] and a["m"] >= 2 and
             sorted(keyed[a["query"]]["seq"]) == sorted(keyed[b["query"]]["seq"])]
    dec = [x for x in rows if x["y"] is not None]
    pairs = [(a, b) for a in dec for b in dec if a["y"] == 1 and b["y"] == 0]
    return {"root": r["key"], "split": r["split"], "cands": rows, "effects": eff, "scene": sp, "kappa": kap, "order_pairs": order,
            "rank_acc": (sum(a["energy"] < b["energy"] for a, b in pairs) / len(pairs)) if pairs else None}


def fit_temperature(evals) -> float:
    u = np.array([c["u"] for e in evals for c in e["cands"] if c["y"] is not None])
    y = np.array([c["y"] for e in evals for c in e["cands"] if c["y"] is not None])
    if not len(u):
        return 1.0
    nll = [np.mean(np.logaddexp(0, -u / t) * y + np.logaddexp(0, u / t) * (1 - y)) for t in TAUS]
    return float(TAUS[int(np.argmin(nll))])


def _auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def select(e, tau, thr=0.5):
    """Restricted-catalogue choice: fewest moves among p_cal >= thr, then predicted J, displacement, energy."""
    el = sorted([c for c in e["cands"] if 1 / (1 + math.exp(-c["u"] / tau)) >= thr], key=lambda c: (c["m"], c["J_pred"], c["C_d"], c["energy"], c["query"]))
    return el[0] if el else None


def metrics(evals, tau=1.0, boot=200, seed=0) -> dict:
    rows = [(e["root"], c) for e in evals for c in e["cands"] if c["y"] is not None]
    if not rows:
        return {"decided": 0}
    y = np.array([c["y"] for _, c in rows])
    p = 1 / (1 + np.exp(-np.array([c["u"] for _, c in rows]) / tau))
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum(abs(p[bins == b].mean() - y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any())
    conf = (p >= 0.8) | (p <= 0.2)
    pairs = [o for e in evals for o in e["order_pairs"] if None not in o["y"] and o["y"][0] != o["y"][1]]
    eff = [x for e in evals for x in e["effects"]]
    kap = [e["kappa"] for e in evals if e["kappa"]]
    sel = []
    for e in evals:
        if any(c["y"] == 1 and c["m"] > 0 for c in e["cands"]):
            ch = select(e, tau)
            best = min((c["m"] for c in e["cands"] if c["y"] == 1), default=None)
            sel.append({"root": e["root"], "chosen": ch and ch["query"], "label": ch and ch["y"], "moves": ch and ch["m"], "best_verified_moves": best})
    out = {"decided": len(rows), "positives": int(y.sum()), "tau": tau, "accuracy": float(((p >= 0.5) == y).mean()), "auroc": _auc(y, p),
           "nll": float(-(y * np.log(pc) + (1 - y) * np.log(1 - pc)).mean()), "brier": float(((p - y) ** 2).mean()), "ece": float(ece),
           "coverage_0.2_0.8": float(conf.mean()), "accuracy_covered": float(((p >= 0.5) == y)[conf].mean()) if conf.any() else None,
           "order_pairs": len(pairs), "order_accuracy": (sum((o["u"][0] > o["u"][1]) == (o["y"][0] > o["y"][1]) for o in pairs) / len(pairs)) if pairs else None,
           "effect_mae": {k: (float(np.mean([abs(x["pred"] - x["observed"]) for x in eff if x["kind"] == k])) if any(x["kind"] == k for x in eff) else None)
                          for k in ("scene", "omission")},
           "effect_sign_acc": {k: (float(np.mean([np.sign(round(x["pred"])) == np.sign(x["observed"]) for x in eff if x["kind"] == k]))
                                   if any(x["kind"] == k for x in eff) else None) for k in ("scene", "omission")},
           "kappa_n": len(kap), "kappa_mae": float(np.mean([abs(k["pred"] - k["obs"]) for k in kap])) if kap else None,
           "kappa_sign_acc": float(np.mean([np.sign(round(k["pred"])) == np.sign(k["obs"]) for k in kap])) if kap else None,
           "selection": sel, "selection_success": (sum(s["label"] == 1 for s in sel) / len(sel)) if sel else None,
           "selection_minimal": (sum(s["label"] == 1 and s["moves"] == s["best_verified_moves"] for s in sel) / len(sel)) if sel else None,
           "rank_acc": float(np.mean([e["rank_acc"] for e in evals if e["rank_acc"] is not None])) if any(e["rank_acc"] is not None for e in evals) else None}
    roots = sorted({k for k, _ in rows})
    if boot and len(roots) > 1:                                   # root-cluster bootstrap of accuracy and AUROC
        rng, acc, auc = np.random.default_rng(seed), [], []
        idx = {k: [i for i, (kk, _) in enumerate(rows) if kk == k] for k in roots}
        for _ in range(boot):
            ii = [i for k in rng.choice(roots, len(roots)) for i in idx[k]]
            acc.append(((p[ii] >= 0.5) == y[ii]).mean())
            a = _auc(y[ii], p[ii])
            if a is not None:
                auc.append(a)
        out["ci95_accuracy"] = [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]
        out["ci95_auroc"] = [float(np.percentile(auc, 2.5)), float(np.percentile(auc, 97.5))] if auc else None
    return out


def energy_temperature_note():
    return f"Ebar_R = E_static + T softplus(-u) with T = {T}; calibration rescales u for probabilities only, never the energy"
