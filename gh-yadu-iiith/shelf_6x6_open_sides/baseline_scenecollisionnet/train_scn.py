#!/usr/bin/env python3
"""Train SceneCollisionNet on the shelf states (build_scn_data.py), V4 train split; model selection on V4 val.

    python train_scn.py --name s0 [--seed 0] [--epochs 30] [--bs 16] [--l-frac 0.35] [--workers 8]

Per state: scene point cloud (8,192 surface points of the layout) -> voxel features once; 32 grasps x 2 queries (hand at the grasp pose over
the target = "pick", same orientation over the goal = "place") -> collision logits z_pick, z_place; P(grasp works) = (1 - p_pick)(1 - p_place).
Loss (NVlabs trainer: BCE, average of the top-10% hardest elements and a random 10%, 0.5 / 0.5):
  grasp feasibility  on every labelled grasp (negatives through the hard-negative average; the few feasible grasps, 0.2%, averaged separately)
  pick / place       the decomposed labels where the teacher's reason fixes them (weight 0.5)
  L states with y = 1 (winner unknown): -log max_g P_g
Score of a state = max over the 32 grasps of P(grasp works).  Writes runs/<name>/{best.pt, train.log, config.json}.
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

import scn_data as D
import scn_geometry as G
from scn_model import SCN, feasible_logprob, hard_neg_bce, log1mexp

HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to(b):
    return {k: v.to(DEV, non_blocking=True) for k, v in b.items()}


def loss_fn(model, b):
    zp, zl = model(b["pts"], b["pos"], b["R"], b["shift"])
    logP = feasible_logprob(zp, zl)
    fy = b["fy"]
    m = fy >= 0
    bce = -(fy.float() * logP + (1 - fy.float()) * log1mexp(logP))
    neg, pos = bce[m & (fy == 0)], bce[m & (fy == 1)]
    l_fy = hard_neg_bce(neg) if neg.numel() else zp.sum() * 0
    if pos.numel():
        l_fy = 0.5 * l_fy + 0.5 * pos.mean()
    tot = l_fy
    for z, key in ((zp, "pick"), (zl, "place")):
        lab = b[key]
        mk = lab >= 0
        if mk.any():
            tot = tot + 0.25 * hard_neg_bce(F.binary_cross_entropy_with_logits(z[mk], lab[mk].float(), reduction="none"))
    unk = (fy < 0).all(1) & (b["y"] > 0.5)                                  # L state, feasible, winner unknown
    if unk.any():
        tot = tot - logP[unk].max(1).values.mean()
    return tot, dict(fy=float(l_fy))


@torch.no_grad()
def score_states(model, ds, bs=32, workers=4):
    """-> state score [n] = max_g P(grasp g works),  per-grasp P [n, 32],  p_pick [n, 32],  p_place [n, 32]"""
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=False, num_workers=workers, collate_fn=D.collate)
    S, P, PK, PL = [], [], [], []
    for b in dl:
        b = to(b)
        zp, zl = model(b["pts"], b["pos"], b["R"], b["shift"])
        p = feasible_logprob(zp, zl).exp()
        S.append(p.max(1).values.cpu()); P.append(p.cpu()); PK.append(zp.sigmoid().cpu()); PL.append(zl.sigmoid().cpu())
    return [torch.cat(x).numpy() for x in (S, P, PK, PL)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--opt", default="sgd", choices=["sgd", "adam"])
    ap.add_argument("--l-frac", type=float, default=0.35, help="share of the (69%% of train) lattice states with y = 0 used per epoch")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: use only this many train states")
    a = ap.parse_args()
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = random.Random(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    states, roots = D.load()
    S = {p: [s for s in states if s["split"] == p] for p in ("train", "val")}
    model = SCN(G.gripper_cloud()).to(DEV)
    json.dump(dict(vars(a), n_train=len(S["train"]), n_val=len(S["val"]), params=sum(p.numel() for p in model.parameters())), open(out / "config.json", "w"), indent=1)
    log = open(out / "train.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    say(f"{a.name}: train {len(S['train'])} val {len(S['val'])} | params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | {DEV} | opt {a.opt} lr {a.lr} bs {a.bs}")
    opt = torch.optim.SGD(model.parameters(), lr=a.lr, momentum=0.9) if a.opt == "sgd" else torch.optim.Adam(model.parameters(), lr=a.lr)
    lat0 = [i for i, s in enumerate(S["train"]) if s["kind"] == "L" and s["y"] == 0]
    rest = [i for i, s in enumerate(S["train"]) if not (s["kind"] == "L" and s["y"] == 0)]
    val_ds = D.StateSet(S["val"], roots)
    yv = np.array([s["y"] for s in S["val"]])
    best, best_ep = -1.0, -1
    for ep in range(a.epochs):
        model.train()
        idx = rest + rng.sample(lat0, int(a.l_frac * len(lat0)))
        rng.shuffle(idx)
        if a.limit:
            idx = idx[:a.limit]
        dl = torch.utils.data.DataLoader(D.StateSet([S["train"][i] for i in idx], roots, train=True), batch_size=a.bs, shuffle=False,
                                         num_workers=a.workers, collate_fn=D.collate, drop_last=True, persistent_workers=False)
        t0, tot, n = time.time(), 0.0, 0
        for b in dl:
            loss, _ = loss_fn(model, to(b))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item(); n += 1
        sv = score_states(model, val_ds, workers=a.workers)[0]
        au, acc = roc_auc_score(yv, sv), float(np.mean((sv >= 0.5) == yv))
        mark = ""
        if au > best:
            best, best_ep, mark = au, ep, " *"
            torch.save(model.state_dict(), out / "best.pt")
        say(f"ep {ep:3d} loss {tot / max(1, n):.4f} | val state AUROC {au:.4f} acc@0.5 {acc:.4f} | {time.time() - t0:.0f}s ({n} steps){mark}")
    say(f"best epoch {best_ep}: val AUROC {best:.4f}")


if __name__ == "__main__":
    main()
