"""P7a.1: read-only numerical audit of the P6 A/B/C implementation (no training, no teacher call).

For frozen P6 checkpoints on frozen P6 development roots it records, per scene edit, the learner-visible
edit evidence the P6 scene factor actually receives (pooled edited-scene vector), the scene logits and
the zS marginals under several different repairs r of the SAME root (root-consistency question), the
coupling / higher-order contributions, and, over the P6 training roots, every unweighted loss term with
its unclipped per-module gradient norm (optimization question). Output: runs/p7/audit/*.json.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

import abc_models as am
import abc_train as at
import config as cf

P6 = Path("runs/p6/train")
ROOTS = ["p6_2026102301", "p6_2026102303"]
CKPTS = {"B#s0": dict(name="B", measure="uniform", higher_order=True), "C#s0": dict(name="C", measure="uniform", higher_order=True),
         "B-no-coupling#s0": dict(name="B-no-coupling", measure="uniform")}


def _sum(t):
    return "sha256:" + hashlib.sha256(t.detach().numpy().tobytes()).hexdigest()[:16]


def load(tag):
    kw = dict(CKPTS[tag])
    m = am.build(kw.pop("name"), **kw)
    m.load_state_dict(torch.load(P6 / f"ckpt_{tag}.pt", weights_only=True), strict=True)
    return m.eval()


@torch.no_grad()
def trace(m, r) -> dict:
    """Edit evidence, scene logits and the r-dependence of zS marginals on one root."""
    R, c = r["R"], m.ctxn(r["R"].ctx)
    ed = {}
    for kind, poses in [("none", {})] + sorted(r["edits"].items()):
        xy = R.xy0.clone()
        for i, p in poses.items():
            xy[i] = torch.as_tensor(p, dtype=torch.float64)
        pv = m.pooled(m.encode(R, xy))
        ed[kind] = {"moved": sorted(poses), "pooled_checksum": _sum(pv), "logit": float(m.scene(torch.cat([pv, c]))[0]), "_v": pv}
    base = ed["none"]["_v"]
    for v in ed.values():
        v["rel_l2_to_unedited_pooled"] = float((v.pop("_v") - base).norm() / base.norm())
    rows = []
    for cand in sorted(r["cands"], key=lambda x: len(x["seq"]))[:: max(1, len(r["cands"]) // 6)]:
        o = m.joint(R, cand["seq"], cand["slots"], r["edits"])
        sl = cand["slots"]
        mk = lambda n: float(o["marg"][sl.index(n)]) if n in sl else None  # noqa: E731
        ho = float(o["E_states"].max() - o["E_states"].min())
        rows.append({"query": cand["query"], "moves": len(cand["seq"]), "K": len(sl), "p_z0": mk("z0"), "p_zS[A]": mk("zS[A]"),
                     "p_zS[B]": mk("zS[B]"), "p_zS[A+B]": mk("zS[A+B]"), "p_zR": mk("zR"), "W_abs_sum": float(o["W"].abs().sum()),
                     "energy_range": ho})
    spread = {k: max(x[k] for x in rows) - min(x[k] for x in rows) for k in ("p_z0", "p_zS[A]", "p_zS[B]", "p_zS[A+B]")}
    return {"root": r["key"], "edits": ed, "per_repair_marginals": rows, "zS_marginal_spread_across_r": spread,
            "teacher": {k: r["y"].get(q) for k, q in r["edit_q"].items()} | {"z0": next(c["y"] for c in r["cands"] if not c["seq"])}}


def loss_gradients(m, roots) -> dict:
    """Unweighted P6 loss terms and their UNCLIPPED gradient norms per module group (one full batch)."""
    out = {}
    for term in ("static", "task", "outcome", "effect", "cost", "pool"):
        m.zero_grad()
        vals = [at.losses(m, r)[term] for r in roots]
        vals = [v for v in vals if v is not None]
        if not vals:
            continue
        tot = torch.stack(vals).mean()
        tot.backward()
        g = {}
        for n, p in m.named_parameters():
            if p.grad is not None:
                k = n.split(".")[0]
                g[k] = (g.get(k, 0.0) ** 2 + float(p.grad.norm()) ** 2) ** 0.5
        out[term] = {"value": float(tot.detach()), "roots_with_term": len(vals), "grad_norm_total": sum(x ** 2 for x in g.values()) ** 0.5,
                     "grad_norm_by_module": {k: round(v, 4) for k, v in sorted(g.items())}}
    return out


def main(out="runs/p7/audit") -> None:
    torch.set_num_threads(1)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ds = torch.load(P6 / "dataset.pt", weights_only=False)
    roots = at.prepare(ds)
    by = {r["key"]: r for r in roots}
    res = {"traces": {t: [trace(load(t), by[k]) for k in ROOTS] for t in CKPTS}}
    train = [r for r in roots if r["split"] == "train"]
    res["loss_gradients_P6_trained"] = {t: loss_gradients(load(t), train) for t in CKPTS}
    torch.manual_seed(0)
    res["loss_gradients_P6_init"] = {n: loss_gradients(am.build(n, measure="uniform", higher_order=n != "A") if n != "A" else am.build("A"), train)
                                     for n in ("A", "B", "C")}
    res["sources"] = {f: cf.stable_hash(Path(f).read_text()) for f in ("abc_models.py", "abc_train.py", "p7_audit.py")}
    cf.write_json(out / "p6_numerical_audit.json", json.loads(json.dumps(res, default=float)))
    print(json.dumps({t: [x["zS_marginal_spread_across_r"] for x in v] for t, v in res["traces"].items()}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
