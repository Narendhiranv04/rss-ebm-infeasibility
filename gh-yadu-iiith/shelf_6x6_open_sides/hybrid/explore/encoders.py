"""EXPLORATORY encoder comparison (VAL only): swap ONLY the scene encoder of the per-step hazard model; data, labels, training loop,
pools (V3.4 FULL_s0 VAL pools), rules and teacher are unchanged.  Frozen code (twod_hazard / hz_eval) is imported read-only; every output goes
to runs/explore/enc/{kind}/ (twod_hazard.OUT and .Net are re-bound in this process only, so the frozen models are never touched).

Variants (same input: per-object features [n, MAXOBJ, 14] + mask + occupancy grid; same output width H):
  base  = placement.SceneEncoder (current: per-object MLP, ONE attention read-out from the query object, mean pool, MLP on the grid)
  tf    = transformer: per-object MLP -> 2 self-attention layers over objects (4 heads, padding masked) -> query token + mean + grid MLP
  gnn   = graph network on the complete object graph: 2 rounds of pairwise messages MLP([h_i, h_j]) averaged over neighbours j
  cnn   = base, but the occupancy grid goes through a small CNN (2 conv layers) instead of an MLP
  nophase = base encoder, but training ignores WHICH step failed (fail_step -> None): tests whether step-level supervision explains the
            gain over XYNet (2D: 3,728 of 10,191 failed TRAIN plans carry the step; 3D: 1,711 of 4,043)
(GRU over plan steps changes the training data layout; not in this file.)
    python encoders.py selftest
    python encoders.py train DOM KIND SEED          (venv python)
    python encoders.py pool DOM KIND SEED i/k       (venv python)
    python encoders.py eval DOM KIND SEED W         (env2d python3.14 for 2d, env3d python for 3d) -> runs/explore/enc/{kind}/metrics_{dom}_s{seed}.json
"""
import json, math, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import twod_hazard as HZ  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

PL, Hd = HZ.PL, HZ.H
EXP = HZ.RUNS / "explore/enc"


class TFEnc(nn.Module):
    def __init__(self, ncell, nfeat=14, h=Hd):
        super().__init__()
        self.phi = PL.mlp(nfeat, h, h)
        layer = nn.TransformerEncoderLayer(h, 4, 2 * h, dropout=0.0, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.grid, self.out = PL.mlp(ncell, h, h, 1), PL.mlp(3 * h, h, h, 1)

    def forward(self, obj, mask, occ):
        e = self.tf(self.phi(obj), src_key_padding_mask=~mask) * mask[..., None]
        mean = e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.out(torch.cat([e[:, 0], mean, self.grid(occ)], -1))


class GNNEnc(nn.Module):
    def __init__(self, ncell, nfeat=14, h=Hd, rounds=2):
        super().__init__()
        self.phi = PL.mlp(nfeat, h, h)
        self.msg = nn.ModuleList(PL.mlp(2 * h, h, h, 1) for _ in range(rounds))
        self.upd = nn.ModuleList(PL.mlp(2 * h, h, h, 1) for _ in range(rounds))
        self.grid, self.out = PL.mlp(ncell, h, h, 1), PL.mlp(3 * h, h, h, 1)

    def forward(self, obj, mask, occ):
        e = self.phi(obj) * mask[..., None]
        n = e.shape[1]
        pm = (mask[:, :, None] & mask[:, None, :]).float()                 # valid (i, j) pairs
        for msg, upd in zip(self.msg, self.upd):
            pair = torch.cat([e[:, :, None].expand(-1, -1, n, -1), e[:, None].expand(-1, n, -1, -1)], -1)
            m = (msg(pair) * pm[..., None]).sum(2) / pm.sum(2, keepdim=True).clamp(min=1)
            e = (e + upd(torch.cat([e, m], -1))) * mask[..., None]
        mean = e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.out(torch.cat([e[:, 0], mean, self.grid(occ)], -1))


class SetEnc(nn.Module):
    """XYNet-style set encoder (reviewer request: encoder-matched arm): per-object MLP, masked mean + max pooling, no attention,
    + the query-object token and the grid MLP so the step model keeps the same inputs."""
    def __init__(self, ncell, nfeat=14, h=Hd):
        super().__init__()
        self.phi = PL.mlp(nfeat, h, h)
        self.grid, self.out = PL.mlp(ncell, h, h, 1), PL.mlp(4 * h, h, h, 1)

    def forward(self, obj, mask, occ):
        e = self.phi(obj) * mask[..., None]
        mean = e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        mx = e.masked_fill(~mask[..., None], -1e9).max(1).values
        return self.out(torch.cat([e[:, 0], mean, mx, self.grid(occ)], -1))


class CNNEnc(PL.SceneEncoder):
    def __init__(self, ncell, nfeat=14, h=Hd, rc=None):
        super().__init__(ncell, nfeat, h)
        self.rc = rc
        self.grid = nn.Sequential(nn.Unflatten(1, (1, *rc)), nn.Conv2d(1, 16, 3, padding=1), nn.SiLU(), nn.Conv2d(16, 16, 3, padding=1), nn.SiLU(),
                                  nn.Flatten(), nn.Linear(16 * rc[0] * rc[1], h))


def net_class(kind, dom):
    class Net(HZ.Net):
        def __init__(self, mu, sd, ncell):
            super().__init__(mu, sd, ncell)
            if kind == "tf":
                self.enc = TFEnc(ncell)
            elif kind == "gnn":
                self.enc = GNNEnc(ncell)
            elif kind == "cnn":
                self.enc = CNNEnc(ncell, rc=PL.GRID[dom])
            elif kind == "setnet":
                self.enc = SetEnc(ncell)
            elif kind not in ("base", "nophase"):
                raise ValueError(kind)
    return Net


def bind(dom, kind):
    """re-bind the frozen module's model class and output dir in THIS process only."""
    HZ.Net, HZ.OUT = net_class(kind, dom), EXP / kind
    if kind == "nophase":      # why-it-works test: drop the failed-step label, every X failure trains as a whole-plan set-negative
        HZ.fail_step = lambda phase, m: None
    HZ.OUT.mkdir(parents=True, exist_ok=True)
    assert HZ.OUT != HZ.ABF / "runs/hz"


def selftest():
    for dom in ("2d", "3d"):
        R, C = PL.GRID[dom]
        n, MO = 5, PL.MAXOBJ[dom]
        Fe, M = torch.randn(n, 2, MO, 14), torch.zeros(n, 2, MO, dtype=torch.bool)
        M[..., :7] = True
        O, X = torch.rand(n, 2, R * C), torch.randn(n, 8)
        for kind in ("base", "tf", "gnn", "cnn", "setnet"):
            net = net_class(kind, dom)(torch.zeros(14), torch.ones(14), R * C)
            y = net(Fe, M, O, X)
            assert y.shape == (n,) and torch.isfinite(y).all(), (dom, kind)
            # padded objects must not change the output
            Fe2 = Fe.clone(); Fe2[..., 7:, :] = 99.0
            assert torch.allclose(net(Fe2, M, O, X), y, atol=1e-5), (dom, kind, "padding leak")
            print(dom, kind, "params", sum(p.numel() for p in net.parameters()))
    print("selftest OK")


def evaluate(dom, kind, seed, workers):
    """VAL metrics of this variant under the frozen rule; teacher queries only for picks without a label (theirs + ours + this dir)."""
    import os
    os.environ["V3OUT"] = str(EXP / kind / "teach_out")                   # keep teacher scratch out of the frozen runs/hz tree
    os.environ["FIN3D_STORE"] = str(EXP / kind / "teach_out/fin3d_store")
    import hz_eval as H
    bind(dom, kind)
    priv, rows = H.priv_of(dom, "val"), {}
    for p in sorted(HZ.OUT.glob(f"pool_{dom}_val_s{seed}*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            rows.setdefault(r["root_id"], r)
    assert set(priv) <= set(rows), f"{len(set(priv) - set(rows))} roots not scored"
    vf = HZ.OUT / f"verify_{dom}.jsonl"
    L = H.labels(dom)
    if vf.exists():
        for k, v in H.V.labels(dom, str(vf)).items():
            if None in L.get(k, (None, None)):
                L[k] = v
    jobs = []
    for rid, r in rows.items():
        d = r["decisions"][0]
        if d and None in L.get((rid, json.dumps(d)), (None, None)):
            e = priv[rid]
            jobs.append({"root_id": rid, "dom": dom.upper(), "path": H.local(e["path"]), "geom": e.get("geom", "original"), "ids": e["ids"],
                         "seqs": [[[e["ids"][o], c] for o, c in d]]})
    print(json.dumps({"kind": kind, "dom": dom, "seed": seed, "teacher_queries": len(jobs)}), flush=True)
    if jobs:
        H._run(dom, jobs, workers, vf)
        for k, v in H.V.labels(dom, str(vf)).items():
            if None in L.get(k, (None, None)):
                L[k] = v
    res = H.rates([{"masks": priv[rid]["masks"], "outcome": H.V.outcome(r["decisions"][0], L, rid)} for rid, r in rows.items()])
    tr = json.loads((HZ.OUT / f"train_{dom}_s{seed}.json").read_text()) if (HZ.OUT / f"train_{dom}_s{seed}.json").exists() else {}
    out = {"label": "EXPLORATORY VAL", "kind": kind, "dom": dom, "seed": seed, "frozen_rule": res, "best_val_nll": tr.get("best_val_nll")}
    (HZ.OUT / f"metrics_{dom}_s{seed}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "selftest":
        selftest()
    elif c == "train":
        bind(a[0], a[1]); HZ.train(a[0], int(a[2]))
    elif c == "pool":
        bind(a[0], a[1]); HZ.pool_runs(a[0], "val", int(a[2]), a[3])
    elif c == "eval":
        evaluate(a[0], a[1], int(a[2]), int(a[3]))
