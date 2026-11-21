"""Outcome-only control for the hazard model: hazard.py's network, step featurization and encoder unchanged, but supervised ONLY by the
plan-level outcome: loss = BCE(prod_k p_step_k * p_target, y_plan), y_plan = B's y (1 iff moves AND target succeeded).  failed_step and
every step-level label are never read.  Plans: unique non-empty labels.B sequences per root, first label wins (as hazard.cmd_eval).
TRAIN fits, VAL early-stops (plan-level log-loss; hazard's rule: <= 80 epochs, patience 10, 90 s budget per seed); TEST is never read.

    MPLCONFIGDIR=$TMPDIR/mpl python -W ignore ../hybrid/src/outcome_b.py train|eval|selftest      (from approachb_3d_learning/)
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hazard as Hz  # noqa: E402  (puts approachb_3d_learning/src on sys.path)
from hazard import XD  # noqa: E402

OUT = Hz.A / "runs/OUTCOME_B"


def plans(split):
    """-> {rid: {plan: y}} unique non-empty plans of the split's QC roots (first label per plan)"""
    assert split in ("train", "val")  # never the test split
    out = {}
    for rid, s in XD.qc_roots().items():
        if s != split:
            continue
        seen = {}
        for b in XD.load(rid)["labels"]["B"]:
            seq = tuple((o, tuple(c)) for o, c in b["sequence"])
            if seq and len({o for o, _ in seq}) == len(seq):
                seen.setdefault(seq, int(b["y"]))
        if seen:
            out[rid] = seen
    return out


def dataset(split):
    """-> step batch (hazard.featurize of every unique (root, prefix) step), plan step indices idx [n, L] (-1 pad), y [n]"""
    feats, idx, ys, n = [], [], [], 0
    for rid, ps in plans(split).items():
        raw, uniq = XD.load(rid), {}
        for seq, y in ps.items():
            idx.append([n + uniq.setdefault(st[0], (len(uniq), st))[0] for st in Hz.plan_steps(seq, raw["goal"])]); ys.append(y)
        feats.append(Hz.featurize(raw, [st for _, st in uniq.values()])); n += len(uniq)
    L = max(map(len, idx))
    return (tuple(torch.cat(t) for t in zip(*feats)), torch.tensor([i + [-1] * (L - len(i)) for i in idx]),
            torch.tensor(ys, dtype=torch.float32))


def plan_logp(net, X, idx):
    """log p(plan) = sum of step log-sigmoids; only the steps these plans use are encoded"""
    m = idx >= 0
    u, inv = torch.unique(idx[m], return_inverse=True)
    lp = torch.zeros(idx.shape)
    lp[m] = torch.nn.functional.logsigmoid(net(*(x[u] for x in X)))[inv]
    return lp.sum(1)


def bce(logp, y):
    log1mp = torch.log(-torch.expm1(logp.clamp(max=-1e-7)))
    return -(y * logp + (1 - y) * log1mp).mean()


def train_seed(seed, tr, va, max_epochs=80, patience=10, budget_s=90):
    torch.manual_seed(seed)
    net = Hz.Net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    (Xtr, itr, ytr), (Xva, iva, yva) = tr, va
    best, bad, t0, log = (1e9, None, -1), 0, time.time(), []
    for ep in range(max_epochs):
        net.train()
        for i in torch.randperm(len(ytr)).split(256):
            opt.zero_grad(); bce(plan_logp(net, Xtr, itr[i]), ytr[i]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(bce(plan_logp(net, Xva, iva), yva))
        log.append(round(v, 4))
        if v < best[0]:
            best, bad = (v, {k: t.clone() for k, t in net.state_dict().items()}, ep), 0
        else:
            bad += 1
        if bad >= patience or time.time() - t0 > budget_s:
            break
    net.load_state_dict(best[1])
    torch.save({"state": best[1], "seed": seed, "epoch": best[2], "val_plan_logloss": best[0], "H": Hz.H}, OUT / f"s{seed}.pt")
    return net.eval(), {"best_epoch": best[2], "epochs_run": len(log), "val_plan_bce_curve": log, "train_s": round(time.time() - t0, 1)}


class OutcomeB(Hz.Hazard):
    """Same interface and scoring as hazard.Hazard (__call__(raw, seqs) -> [3, len(seqs)]), outcome-only checkpoints."""

    def __init__(self):
        self.nets = []
        for s in Hz.SEEDS:
            n = Hz.Net(); n.load_state_dict(torch.load(OUT / f"s{s}.pt", weights_only=False)["state"]); self.nets.append(n.eval())


def cmd_train():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, va = dataset("train"), dataset("val")
    out = {"plans": {"train": len(tr[2]), "val": len(va[2])}, "pos": {"train": int(tr[2].sum()), "val": int(va[2].sum())},
           "unique_steps": {"train": len(tr[0][3]), "val": len(va[0][3])}, "featurize_s": round(time.time() - t0, 1),
           "loss": "plan-level BCE of the step-probability product; no step labels", "threads": torch.get_num_threads(), "seeds": {}}
    print(json.dumps(out), flush=True)
    for s in Hz.SEEDS:
        _, info = train_seed(s, tr, va)
        out["seeds"][s] = info
        print(json.dumps({"seed": s, **{k: v for k, v in info.items() if k != "val_plan_bce_curve"}}), flush=True)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1))


def cmd_eval():
    from sklearn.metrics import roc_auc_score
    import phase3 as P3
    from sidecar_from_b import ece
    Mm = P3.Models()
    sc = {"a_boltzD_sidecar": lambda r, s: Mm.p11(r, s), "b_outcome_b": OutcomeB(), "c_hazard": Hz.Hazard()}
    ys, ps = [], {k: [] for k in sc}
    for rid, pl in plans("val").items():
        raw, seqs = XD.load(rid), list(pl)
        ys += list(pl.values())
        for k, f in sc.items():
            ps[k] += f(raw, seqs).mean(0).tolist()
    y = np.array(ys, float)
    res = {"n_plans": len(y), "pos": int(y.sum())}
    for k, p in ps.items():
        p = np.clip(np.array(p), 0, 1)
        res[k] = {"auroc": round(float(roc_auc_score(y, p)), 4), "brier": round(float(((p - y) ** 2).mean()), 4),
                  "ece10": round(ece(p, y), 4), "mean_p": round(float(p.mean()), 4)}
    out = json.loads((OUT / "metrics.json").read_text())
    out["plan_level_val"] = res
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(res, indent=1))


def cmd_selftest():
    m = OutcomeB()
    raw = XD.load(next(r for r, s in XD.qc_roots().items() if s == "val" and len(XD.load(r)["initial_cells"]) >= 4))
    pl = Hz.random_plans(raw, 100, seed=2)
    p = m(raw, pl)
    assert p.shape == (3, 100) and bool(((p > 0) & (p <= 1)).all())
    assert torch.allclose(m(raw, pl[5:6])[:, 0], p[:, 5], atol=1e-5)
    # training-path product == scoring-path product
    tr_plans = {raw["ids"]["root_id"]: {tuple(q): 0 for q in pl[:20]}}
    uniq, idx = {}, []
    for q in tr_plans[raw["ids"]["root_id"]]:
        idx.append([uniq.setdefault(st[0], (len(uniq), st))[0] for st in Hz.plan_steps(q, raw["goal"])])
    X = Hz.featurize(raw, [st for _, st in uniq.values()])
    L = max(map(len, idx))
    with torch.no_grad():
        lp = plan_logp(m.nets[0], X, torch.tensor([i + [-1] * (L - len(i)) for i in idx]))
    assert torch.allclose(lp.exp(), p[0, :20], atol=1e-5)
    assert abs(float(bce(torch.tensor([0.5, 0.5]).log(), torch.tensor([1.0, 0.0]))) - np.log(2)) < 1e-5
    print("selftest ok")


if __name__ == "__main__":
    torch.set_num_threads(2)
    {"train": cmd_train, "eval": cmd_eval, "selftest": cmd_selftest}[sys.argv[1]]()
