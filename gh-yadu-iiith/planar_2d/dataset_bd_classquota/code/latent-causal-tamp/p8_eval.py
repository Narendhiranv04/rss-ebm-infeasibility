"""P8a estimands, reported separately (no teacher call):
  (i)   unconditional intervention prediction: root-level P(z0), P(zS[I]); NLL / Brier, kappa by OBSERVED sign
  (ii)  OFFLINE CONDITIONAL-COMPLETION DIAGNOSTIC (oracle-revealed bits, never a robot input): exact conditionals from the
        enumerated joint for B / C; factorized arms (A, S, SF, B-none by construction) return their marginals. Masks:
          a  reveal z0, zS[A]            -> predict zS[B], zS[A+B]
          b  reveal z0, zS[A], zS[B]     -> predict zS[A+B]
          c  reveal zR, zO[0]            -> predict zO[t], t >= 1   (candidates with >= 3 moves)
          d  reveal every scene bit      -> predict zR              (nonempty candidates)
  (iii) repair prediction / selection from the initial input only (p7_eval.metrics).
Root-macro averages with root-cluster bootstrap intervals."""
from __future__ import annotations

import math

import numpy as np
import torch

import abc7_train as t7
import p7_eval as ev

SCENE = ["z0", "zS[A]", "zS[B]", "zS[A+B]"]
MASKS = {"a": (["z0", "zS[A]"], ["zS[B]", "zS[A+B]"]), "b": (["z0", "zS[A]", "zS[B]"], ["zS[A+B]"])}


def _cond(logP, Z, names, reveal: dict, hidden: list) -> dict:
    keep = torch.ones(len(Z), dtype=torch.bool)
    for n, v in reveal.items():
        keep &= Z[:, names.index(n)] == float(v)
    lp = logP[keep] - torch.logsumexp(logP[keep], 0)
    return {h: float(torch.exp(lp) @ Z[keep][:, names.index(h)]) for h in hidden}


def _nll(p, y):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -math.log(p if y else 1 - p)


@torch.no_grad()
def conditional(m, r) -> list:
    """[{task, hidden, p, y, nll}] for every applicable oracle-revealed mask of one root."""
    outs, rows = t7.forward(m, r), []
    ys = {"z0": next((c["y"] for c in r["cands"] if not c["seq"]), None)} | r["scene_y"]
    joint = m.kind in ("B", "C")
    sp = t7.scene_probs(m, r, outs)
    empty = next((o for c, o in zip(r["cands"], outs) if not c["seq"]), None)
    if empty is not None and all(ys.get(k) is not None for k in SCENE) and all(k in sp for k in SCENE):
        for task, (rev, hid) in MASKS.items():
            if joint:
                names = empty["order"][:empty["ks"]]
                pr = _cond(empty["logP_S"], empty["Z"][:, :empty["ks"]] if len(empty["Z"][0]) > empty["ks"] else empty["Z"], names,
                           {n: ys[n] for n in rev}, hid)
            else:
                pr = {h: float(sp[h]) for h in hid}
            rows += [{"task": task, "hidden": h, "p": pr[h], "y": ys[h], "nll": _nll(pr[h], ys[h])} for h in hid]
    for c, o in zip(r["cands"], outs):
        if not c["seq"] or c["y"] is None:
            continue
        z = dict(zip(c["slots"], c["z"]))
        if all(ys.get(k) is not None for k in SCENE if k in z) and all(z.get(k) is not None for k in z if k in SCENE):
            if joint:
                p = _cond(o["logP"], o["Z"], o["order"], {n: z[n] for n in o["order"][:o["ks"]]}, ["zR"])["zR"]
            else:
                p = float(o["p"])
            rows.append({"task": "d", "hidden": "zR", "p": p, "y": c["y"], "nll": _nll(p, c["y"])})
        om = [f"zO[{t}]" for t in range(len(c["seq"]))]
        if len(c["seq"]) >= 3 and all(z.get(n) is not None for n in om):
            for t, n in list(enumerate(om))[1:]:
                if joint:
                    p = _cond(o["logP"], o["Z"], o["order"], {"zR": c["y"], "zO[0]": z["zO[0]"]}, [n])[n]
                else:
                    p = float(torch.sigmoid(m.seq(r["R"], [a for k, a in enumerate(c["seq"]) if k != t])["u"]))
                rows.append({"task": "c", "hidden": n, "p": p, "y": z[n], "nll": _nll(p, z[n])})
    return rows


def unconditional(evals) -> dict:
    """Root-level scene beliefs vs decided labels and kappa by observed sign (from p7_eval.evaluate outputs)."""
    bits, kap = [], {}
    for e in evals:
        if e["kappa"]:
            for n in SCENE:
                p, y = e["kappa"]["scene_pred"][n], e["kappa"]["scene_obs"][n]
                bits.append((e["root"], _nll(p, y), (p - y) ** 2))
            s = e["kappa"]["obs"]
            kap.setdefault(s, []).append((round(e["kappa"]["pred"]), abs(e["kappa"]["pred"] - s)))
    per_sign = {str(s): {"n": len(v), "sign_acc": float(np.mean([int(np.sign(p) == np.sign(s)) for p, _ in v])),
                         "mae": float(np.mean([a for _, a in v]))} for s, v in sorted(kap.items())}
    return {"scene_bits": len(bits), "scene_nll": float(np.mean([b[1] for b in bits])) if bits else None,
            "scene_brier": float(np.mean([b[2] for b in bits])) if bits else None, "kappa_by_sign": per_sign,
            "kappa_balanced_sign_acc": float(np.mean([v["sign_acc"] for v in per_sign.values()])) if per_sign else None,
            "kappa_macro_mae": float(np.mean([v["mae"] for v in per_sign.values()])) if per_sign else None}


def summarize_conditional(rows_by_root: dict, boot=200, seed=0) -> dict:
    out = {}
    for task in ("a", "b", "c", "d"):
        per = {k: np.mean([x["nll"] for x in v if x["task"] == task]) for k, v in rows_by_root.items() if any(x["task"] == task for x in v)}
        if not per:
            out[task] = {"roots": 0, "nll": None}
            continue
        vals = np.array(list(per.values()))
        rng = np.random.default_rng(seed)
        bs = [rng.choice(vals, len(vals)).mean() for _ in range(boot)] if len(vals) > 1 else [vals.mean()]
        out[task] = {"roots": len(vals), "bits": sum(1 for v in rows_by_root.values() for x in v if x["task"] == task),
                     "nll": float(vals.mean()), "ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]}
    return out


def evaluate_all(m, roots, tau=1.0) -> dict:
    evals = [ev.evaluate(m, r) for r in roots]
    cond = {r["key"]: conditional(m, r) for r in roots}
    return {"repair": ev.metrics(evals, tau), "intervention": unconditional(evals), "conditional": summarize_conditional(cond),
            "_evals": evals, "_cond": cond}
