"""G4E-ABC proof of concept: masked losses, training and evaluation for A, B, C (no teacher call).

Losses (all weights 1, Methodology v1.7): static BCE (unary/pair, analytic labels), task BCE
(A: GRU logit; B/C: logit of the DERIVED marginal p(z_R=1|r), or p(z_0=1) for the empty repair),
B/C outcome-field BCE on every decided active slot, effect MSE (scene and omission contrasts with
both sides decided), log-Huber first-witness cost, A's finite-pool feasibility loss (decided pools
with both classes), and B/C's exact joint pool likelihood over ALL (r, z) states of the fully
decided candidates. UNKNOWN / masked fields contribute exactly zero gradient.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

import abc_models as am


def prepare(ds, certified_only=False) -> list:
    roots = []
    drop = lambda y, src: None if (certified_only and src == "PLANNER_BUDGET") else y  # noqa: E731
    for rec in ds["roots"]:
        R = am.Root(rec)
        ids = rec["audit"]["ids"]
        edits = {e["kind"]: e["poses"] for e in rec["edits"]}
        cands = []
        for c in rec["candidates"]:
            pv = c["privileged"]
            pairs = [pv["static"]["pair"][f"{ids[i]}|{ids[j]}"] for i, j in R.pairs]
            cands.append({"seq": [(i, tuple(d)) for i, d in c["actions"]], "slots": c["z_slots"], "query": pv["query"],
                          "y": drop(pv["y"], pv["source"]), "w": pv["weight"], "src": pv["source"], "cost": pv["cost"],
                          "z": [drop(y, s) for y, s in zip(pv["z"], pv["z_source"])], "z_q": c["z_queries"],
                          "su": pv["static"]["unary"], "sp": pairs})
        ylab = {c["query"]: c["y"] for c in cands} | {e["privileged"]["query"]: drop(e["privileged"]["y"], e["privileged"]["source"])
                                                      for e in rec["edits"]}
        roots.append({"R": R, "edits": edits, "cands": cands, "y": ylab, "split": rec["audit"]["split"], "key": rec["audit"]["root"],
                      "edit_q": {e["kind"]: e["privileged"]["query"] for e in rec["edits"]},
                      "static_checks": [{"seq": [(i, tuple(d)) for i, d in c["actions"]], "su": c["unary"],
                                         "sp": [c["pair"][f"{ids[i]}|{ids[j]}"] for i, j in R.pairs]} for c in rec.get("static_checks", [])]})
    return roots


def _bce(logit, y, w=1.0):
    return w * F.binary_cross_entropy_with_logits(logit, torch.tensor(float(y), dtype=torch.float64))


def _logit(p):
    p = p.clamp(1e-9, 1 - 1e-9)
    return torch.log(p) - torch.log1p(-p)


def _static(o, c):
    terms = [_bce(o["lu"][i], y) for i, y in enumerate(c["su"]) if y is not None]
    terms += [_bce(o["lp"][k], y) for k, y in enumerate(c["sp"]) if y is not None]
    return torch.stack(terms).mean() if terms else None


def forward(model, r) -> list:
    if model.kind == "A":
        return [model.seq(r["R"], c["seq"]) for c in r["cands"]]
    model._memo = {}                                       # per-root, per-pass memo of order / scene slots (exact)
    try:
        return [model.joint(r["R"], c["seq"], c["slots"], r["edits"]) for c in r["cands"]]
    finally:
        model._memo = None


def losses(model, r, outs=None, rec=None) -> dict:
    """Per-term losses of one root (None where no decided supervision exists); `rec` (a list)
    receives every decided effect contrast as {kind, query, pred, observed} for evaluation."""
    outs = outs or forward(model, r)
    L = {k: [] for k in ("static", "task", "outcome", "effect", "cost", "pool")}
    byq = {c["query"]: (c, o) for c, o in zip(r["cands"], outs)}
    for c, o in zip(r["cands"], outs):
        if (s := _static(o, c)) is not None:
            L["static"].append(s)
        if model.kind == "A":
            p = o["p"]
            if c["y"] is not None:
                L["task"].append(_bce(o["u"], c["y"], c["w"]))
        else:
            k = c["slots"].index("zR" if c["seq"] else "z0")
            p = o["marg"][k]
            if c["y"] is not None:
                L["task"].append(_bce(_logit(p), c["y"], c["w"]))
            L["outcome"] += [_bce(_logit(o["marg"][i]), z) for i, z in enumerate(c["z"]) if z is not None]
        if c["cost"] is not None and c["seq"]:
            L["cost"].append(F.huber_loss(torch.log1p(o["J"]), torch.log1p(torch.tensor(c["cost"], dtype=torch.float64))))
        c["_p"] = p
    empty = byq.get(next((q for q in byq if q.endswith(":empty")), None))
    for kind, q in r["edit_q"].items():                                   # fixed-robot scene effects
        ye, y0 = r["y"].get(q), (empty[0]["y"] if empty else None)
        if empty is None or ye is None or y0 is None:
            continue
        if model.kind == "A":
            xy = r["R"].xy0.clone()
            for i, pz in r["edits"][kind].items():
                xy[i] = torch.as_tensor(pz, dtype=torch.float64)
            pred = model.scene_p(r["R"], xy) - empty[0]["_p"]
        else:
            o, sl = empty[1], empty[0]["slots"]
            if f"zS[{kind}]" not in sl:
                continue
            pred = o["marg"][sl.index(f"zS[{kind}]")] - o["marg"][sl.index("z0")]
        L["effect"].append((pred - (ye - y0)) ** 2)
        if rec is not None:
            rec.append({"kind": "scene", "query": q, "pred": float(pred), "observed": ye - y0})
    for c, o in zip(r["cands"], outs):                                    # repair omissions
        for t in range(len(c["seq"]) if len(c["seq"]) >= 2 else 0):
            qo = c["z_q"][c["slots"].index(f"zO[{t}]")]
            yo = r["y"].get(qo)
            if c["y"] is None or yo is None:
                continue
            if model.kind == "A":
                pred = c["_p"] - byq[qo][0]["_p"] if qo in byq else None
            else:
                pred = o["marg"][c["slots"].index("zR")] - o["marg"][c["slots"].index(f"zO[{t}]")]
            if pred is not None:
                L["effect"].append((pred - (c["y"] - yo)) ** 2)
                if rec is not None:
                    rec.append({"kind": "omission", "query": c["query"], "omitted": t, "pred": float(pred), "observed": c["y"] - yo})
    dec = [(c, o) for c, o in zip(r["cands"], outs) if c["y"] is not None]
    if model.kind == "A" and {c["y"] for c, _ in dec} == {0, 1}:
        lp = torch.log_softmax(torch.stack([-o["E"] / am.T for _, o in dec]), 0)
        L["pool"].append(-torch.stack([lp[i] for i, (c, _) in enumerate(dec) if c["y"] == 1]).mean())
    if model.kind != "A":
        full = [(c, o) for c, o in zip(r["cands"], outs) if all(z is not None for z in c["z"]) and c["y"] is not None]
        if len(full) >= 2 and {c["y"] for c, _ in full} == {0, 1}:
            lse = torch.logsumexp(torch.cat([-o["E_states"] / am.T for _, o in full]), 0)
            num = [-o["E_states"][int("".join(str(int(z)) for z in c["z"]), 2)] / am.T for c, o in full if c["y"] == 1]
            L["pool"].append(-(torch.stack(num) - lse).mean())
    for c in r.get("static_checks", []):                                  # analytic static checks incl. invalid endpoints
        xy = r["R"].xy0.clone()
        for i, d in c["seq"]:
            xy[i] = torch.as_tensor(d, dtype=torch.float64)
        _, lu, lp = model.static(r["R"], xy)
        if (st := _static({"lu": lu, "lp": lp}, c)) is not None:
            L["static"].append(st)
    return {k: (torch.stack(v).mean() if v else None) for k, v in L.items()}


def total(model, roots) -> tuple:
    terms = [losses(model, r) for r in roots]
    tot, rec = 0.0, {}
    for k in ("static", "task", "outcome", "effect", "cost", "pool"):
        v = [t[k] for t in terms if t[k] is not None]
        if v:
            m = torch.stack(v).mean()
            tot, rec[k] = tot + m, float(m.detach())
    return tot, rec


def train(model, roots, epochs, cpu_budget_s, log, lr=1e-3, wd=1e-4, clip=None, val=None, every=10, patience=None, snap=None) -> list:
    """P5 defaults (no clipping, no validation). P6: clip = max global grad norm (norms before/after
    recorded), NaN/Inf guard (stop, keep last finite state), validation-only early stopping on
    `val` roots every `every` epochs with the best state restored at the end; `snap(epoch, model)` is
    called at the same checkpoints (visualization snapshots only, never selection)."""
    import copy
    import time
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    hist, c0, best, wait = [], time.process_time(), (math.inf, None, -1), 0
    for ep in range(epochs):
        opt.zero_grad()
        tot, rec = total(model, roots)
        if not torch.isfinite(tot):
            log(f"{model.kind}: non-finite loss at epoch {ep}; stopped")
            hist.append({"epoch": ep, "nonfinite": True})
            break
        tot.backward()
        if clip is not None:
            pre = float(torch.nn.utils.clip_grad_norm_(model.parameters(), clip))
            post = math.sqrt(sum(float(p.grad.norm()) ** 2 for p in model.parameters() if p.grad is not None))
            rec |= {"grad_pre": pre, "grad_post": post}
        opt.step()
        hist.append({"epoch": ep, "total": float(tot.detach()), **rec})
        if snap is not None and ep % every == 0:
            snap(ep, model)
        if val and ep % every == 0:
            with torch.no_grad():
                vl = float(total(model, val)[0])
            hist[-1]["val"] = vl
            if vl < best[0]:
                best, wait = (vl, copy.deepcopy(model.state_dict()), ep), 0
            elif patience is not None and (wait := wait + 1) >= patience:
                log(f"{model.kind}: early stop at epoch {ep} (best val epoch {best[2]})")
                break
        if time.process_time() - c0 > cpu_budget_s:
            log(f"{model.kind}: CPU budget reached at epoch {ep}")
            break
    if val and best[1] is not None:
        model.load_state_dict(best[1])
        hist.append({"best_val_epoch": best[2], "best_val": best[0]})
    return hist


def grad_norms(model, roots) -> dict:
    model.zero_grad()
    terms = [losses(model, r) for r in roots]
    sum(t[k] for t in terms for k in t if t[k] is not None).backward()
    out = {}
    for n, p in model.named_parameters():
        g = 0.0 if p.grad is None else float(p.grad.norm())
        out[n.split(".")[0]] = math.hypot(out.get(n.split(".")[0], 0.0), g)
    return out
