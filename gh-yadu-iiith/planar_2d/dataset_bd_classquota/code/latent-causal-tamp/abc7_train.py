"""P7 losses, diagnostics and training for ABC7 models (no teacher call).

Every term is first averaged over its independent observed units WITHIN a root, then over roots (equal
root weight), so a root with many candidates or 512 z-states cannot dominate:
  static  analytic endpoint BCE (catalogue endpoints + static checks)
  task    BCE(u(r), Y_r) per decided candidate; u = GRU / head logit (A, S) or the DERIVED logit of
          P(zR = 1 | r) (P(z0 = 1) for the empty repair) for B / C
  scene   BCE on each decided fixed-robot edit outcome zS[I], ONCE per root (B / C: root marginal of P_S;
          A / S: re-scored edited scene) - matched supervision across arms
  omit    B / C only: BCE on decided zO[r, t] marginals
  effect  squared error of scene effects (once per root) and omission effects
  cost    log-Huber on the first-witness travel J
  pool    A: CE(uniform over positives || softmax(-E/T)) over decided candidates.
          B / C: Methodology v1.7 L_joint over the fully decided pool, = ranking CE over Ebar_R plus
          -log P(z*(r) | r) per positive, the latter divided by its number of outcome bits.
UNKNOWN fields are masked (no gradient). Clipping max-norm is a declared hyperparameter; pre/post norms
and the clipped fraction are logged per epoch.
"""
from __future__ import annotations

import copy
import math
import time

import torch
from torch.nn import functional as F

import abc_train as at
from abc_models import T

TERMS = ("static", "task", "scene", "omit", "effect", "cost", "pool")


def prepare(ds, certified_only=False) -> list:
    roots = at.prepare(ds, certified_only)
    for r in roots:
        r["scene_y"] = {f"zS[{k}]": r["y"].get(q) for k, q in r["edit_q"].items()}
    return roots


def _bce(logit, y):
    return F.binary_cross_entropy_with_logits(logit, torch.tensor(float(y), dtype=torch.float64))


def _bce_p(p, y):
    p = p.clamp(1e-9, 1 - 1e-9)
    return -(torch.log(p) if y else torch.log1p(-p))


def _static(o, c):
    """Vectorized form of abc_train._static (same mean of the same per-label BCE terms; one autograd node)."""
    lg = [o["lu"][i] for i, y in enumerate(c["su"]) if y is not None] + [o["lp"][k] for k, y in enumerate(c["sp"]) if y is not None]
    if not lg:
        return None
    ys = [y for y in c["su"] if y is not None] + [y for y in c["sp"] if y is not None]
    return F.binary_cross_entropy_with_logits(torch.stack(lg), torch.tensor(ys, dtype=torch.float64))


def forward(model, r) -> list:
    model._memo = {}
    try:
        if model.kind in ("A", "S", "SF"):
            return [model.seq(r["R"], c["seq"]) for c in r["cands"]]
        return [model.joint(r["R"], c["seq"], c["slots"], r["edits"]) for c in r["cands"]]
    finally:
        model._memo = None


def scene_probs(model, r, outs) -> dict:
    """Root-level P(z0 = 1) and P(zS[I] = 1) for every edit kind of the root."""
    empty = next((i for i, c in enumerate(r["cands"]) if not c["seq"]), None)
    if empty is None:
        return {}
    if model.kind in ("A", "S", "SF"):                    # same edit set as the B / C layout (nuisance excluded)
        out = {"z0": outs[empty]["p"]}
        model._memo = {}
        try:
            out |= {f"zS[{k}]": torch.sigmoid(model.scene_logit(r["R"], poses)) for k, poses in r["edits"].items()
                    if f"zS[{k}]" in r["cands"][empty]["slots"]}
        finally:
            model._memo = None
        return out
    return dict(outs[empty]["scene_marg"])


def losses(model, r, outs=None, rec=None) -> dict:
    outs = outs or forward(model, r)
    L = {k: [] for k in TERMS}
    sp = scene_probs(model, r, outs)
    y0 = next((c["y"] for c in r["cands"] if not c["seq"]), None)
    for c, o in zip(r["cands"], outs):
        st = _static(o, c)
        if st is not None:
            L["static"].append(st)
        if c["y"] is not None:
            L["task"].append(_bce(o["u"], c["y"]))
        if c["cost"] is not None and c["seq"]:
            L["cost"].append(F.huber_loss(torch.log1p(o["J"]), torch.log1p(torch.tensor(c["cost"], dtype=torch.float64))))
        c["_p"] = o["p"]
    for n, y in r["scene_y"].items():
        if y is not None and n in sp:
            L["scene"].append(_bce_p(sp[n], y))
            if y0 is not None:
                pred = sp[n] - sp["z0"]
                L["effect"].append((pred - (y - y0)) ** 2)
                if rec is not None:
                    rec.append({"kind": "scene", "slot": n, "pred": float(pred), "observed": y - y0})
    byq = {c["query"]: c for c in r["cands"]}
    for c, o in zip(r["cands"], outs):
        for t in range(len(c["seq"]) if len(c["seq"]) >= 2 else 0):
            n = f"zO[{t}]"
            yo = c["z"][c["slots"].index(n)] if n in c["slots"] else None
            if yo is None:
                continue
            if model.kind not in ("A", "S", "SF"):
                po = o["marg"][c["slots"].index(n)]
                L["omit"].append(_bce_p(po, yo))
                pred = o["p"] - po
            elif model.kind == "SF":                      # P8 factorized control: its own independent omission prediction
                po = model.omit_p(r["R"], [a for k, a in enumerate(c["seq"]) if k != t])
                L["omit"].append(_bce_p(po, yo))
                pred = o["p"] - po
            else:
                qo = c["z_q"][c["slots"].index(n)]
                pred = o["p"] - byq[qo]["_p"] if qo in byq else None
            if pred is not None and c["y"] is not None:
                L["effect"].append((pred - (c["y"] - yo)) ** 2)
                if rec is not None:
                    rec.append({"kind": "omission", "query": c["query"], "omitted": t, "pred": float(pred), "observed": c["y"] - yo})
    for c in r.get("static_checks", []):
        xy = r["R"].xy0.clone()
        for i, d in c["seq"]:
            xy[i] = torch.as_tensor(d, dtype=torch.float64)
        _, lu, lp = model.static(r["R"], xy)
        if (st := _static({"lu": lu, "lp": lp}, c)) is not None:
            L["static"].append(st)
    if model.kind == "A":
        dec = [(c, o) for c, o in zip(r["cands"], outs) if c["y"] is not None]
        if {c["y"] for c, _ in dec} == {0, 1}:
            lp_ = torch.log_softmax(torch.stack([-o["E"] / T for _, o in dec]), 0)
            L["pool"].append(-torch.stack([lp_[i] for i, (c, _) in enumerate(dec) if c["y"] == 1]).mean())
    elif model.kind in ("B", "C"):
        full = [(c, o) for c, o in zip(r["cands"], outs) if c["y"] is not None and all(z is not None for z in c["z"])]
        if len(full) >= 2 and {c["y"] for c, _ in full} == {0, 1}:
            lp_ = torch.log_softmax(torch.stack([-o["E"] / T for _, o in full]), 0)
            terms = []
            for i, (c, o) in enumerate(full):
                if c["y"] == 1:
                    zs = [c["z"][c["slots"].index(n)] for n in o["order"]]
                    idx = int("".join(str(int(z)) for z in zs), 2)
                    terms.append(-lp_[i] - o["logP"][idx] / len(zs))
            L["pool"].append(torch.stack(terms).mean())
    return {k: (torch.stack(v).mean() if v else None) for k, v in L.items()}


def total(model, roots, lam=None, detail=False):
    lam = lam or {}
    per = [losses(model, r) for r in roots]
    tot, rec = 0.0, {}
    for k in TERMS:
        v = [p[k] for p in per if p[k] is not None]
        if v:
            m = torch.stack(v).mean()
            tot, rec[k] = tot + lam.get(k, 1.0) * m, float(m.detach())
            rec[f"n_{k}"] = len(v)
    return (tot, rec, per) if detail else (tot, rec)


@torch.no_grad()
def val_metric(model, roots) -> float:
    """Selection metric shared by every configuration of an arm: mean decided outcome NLL (task + scene [+ omit])."""
    _, rec = total(model, roots)
    return sum(rec.get(k, 0.0) for k in ("task", "scene", "omit"))


def term_grads(model, roots) -> dict:
    """Unclipped gradient norm of EACH unweighted term, per top-level module (one full batch)."""
    out = {}
    for k in TERMS:
        model.zero_grad()
        v = [p[k] for p in (losses(model, r) for r in roots) if p[k] is not None]
        if not v:
            continue
        torch.stack(v).mean().backward()
        g = {}
        for n, p in model.named_parameters():
            if p.grad is not None:
                m = n.split(".")[0]
                g[m] = math.hypot(g.get(m, 0.0), float(p.grad.norm()))
        out[k] = {"total": math.sqrt(sum(x * x for x in g.values())), "by_module": {m: round(x, 5) for m, x in sorted(g.items())}}
    model.zero_grad()
    return out


def train(model, roots, epochs, cpu_budget_s, log, lam=None, lr=1e-3, wd=1e-4, clip=None, val=None, every=5, patience=8, snap=None) -> list:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    hist, c0, best, wait = [], time.process_time(), (math.inf, None, -1), 0
    for ep in range(epochs):
        opt.zero_grad()
        tot, rec = total(model, roots, lam)
        if not torch.isfinite(tot):
            log(f"{model.kind}: non-finite loss at epoch {ep}; stopped")
            hist.append({"epoch": ep, "nonfinite": True})
            break
        tot.backward()
        pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip if clip is not None else float("inf")))
        rec |= {"grad_pre": pre, "clipped": bool(clip is not None and pre > clip)}
        opt.step()
        hist.append({"epoch": ep, "total": float(tot.detach()), **rec})
        if snap is not None and ep % every == 0:
            snap(ep, model)
        if val and ep % every == 0:
            vl = val_metric(model, val)
            hist[-1]["val"] = vl
            if vl < best[0]:
                best, wait = (vl, copy.deepcopy(model.state_dict()), ep), 0
            elif (wait := wait + 1) >= patience:
                log(f"{model.kind}: early stop at epoch {ep} (best val epoch {best[2]})")
                break
        if time.process_time() - c0 > cpu_budget_s:
            log(f"{model.kind}: CPU budget reached at epoch {ep}")
            break
    if val and best[1] is not None:
        model.load_state_dict(best[1])
        hist.append({"best_val_epoch": best[2], "best_val": best[0]})
    return hist
