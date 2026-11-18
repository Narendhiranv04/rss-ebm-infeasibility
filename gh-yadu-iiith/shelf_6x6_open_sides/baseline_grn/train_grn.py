#!/usr/bin/env python3
"""Train GRN on the shelf graphs (build_graphs.py), V4 train split; model selection on V4 val.

    python train_grn.py --name s0 [--seed 0] [--epochs 100] [--threads 8]

Stages as in the authors' README (hyper-parameters unchanged: 100 epochs, Adam, IK / GO lr 1e-3 batch 8192, AGF / GRN lr 1e-4 batch 512):
  1  IK module   target node -> per-view reachability.  The label depends on the target's pose only, so the labels of all queries
                 of a scene are merged: one sample per scene.
  2  GO module   (target, neighbour) pairs -> per-view obstruction ratio, views with IK = 0 masked (as the authors do).
  3  GRN         IK and GO modules initialised from 1 and 2, joint training of everything with
                 loss = BCE(F) + BCE(IK) + 10 * MSE(GO)  (the authors' GRNLit loss).
  The authors' separate "AGF with ground-truth IK / GO inputs" stage is skipped: most of our queries carry no ground-truth GO
  (a feasible query stores only its winning grasp).
Labels are partial, so every term is masked (F: Fm, IK: IKm, GO: GOm x reachable view).  The epoch with the best validation
action AUROC is kept.  Writes runs/<name>/{best.pt, train.log, config.json}.
"""
import argparse
import json
import pickle
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from grn_model import GRN, collate

HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def masked_bce(logit, y, m):
    return (nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none") * m).sum() / m.sum().clamp(min=1)


def masked_mse(p, y, m):
    return (((p - y) ** 2) * m).sum() / m.sum().clamp(min=1)


def grn_loss(model, b):
    F, ik, go = model(b)
    Fm = F[b["mask"]]
    lF = masked_bce(Fm, b["F"], b["Fm"])
    lI = masked_bce(ik, b["IK"], b["IKm"])
    mov = b["mask"]
    nb = b["edge_index"][:, b["blocking"]]
    # a GO ratio only means something for a view the target can reach: mask with the IK label (1 = reachable)
    views = (b["IK"] > 0.5).float()
    owner = torch.cumsum(mov.long(), 0)[nb[1]] - 1            # row of the movable node each blocking edge points to
    lG = masked_mse(go, b["GO"], b["GOm"][:, None] * views[owner] * b["IKm"][owner])
    return lF + lI + 10 * lG, dict(F=lF.item(), IK=lI.item(), GO=lG.item())


@torch.no_grad()
def predict(model, graphs, bs=2048):
    model.eval()
    P = []
    for k in range(0, len(graphs), bs):
        b = collate(graphs[k:k + bs], DEV)
        F, ik, go = model(b)
        P.append(F[b["mask"]].sigmoid().cpu().numpy())
    return np.concatenate(P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--stage-epochs", type=int, default=100)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = random.Random(a.seed)
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    G = pickle.load(open(HERE / "data" / "graphs.pkl", "rb"))
    S = {p: [g for g in G if g["split"] == p] for p in ("train", "val", "test")}
    model = GRN(16).to(DEV)
    log = open(out / "train.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    json.dump(dict(vars(a), n_train=len(S["train"]), n_val=len(S["val"]), n_test=len(S["test"]),
                   params=sum(p.numel() for p in model.parameters())), open(out / "config.json", "w"), indent=1)
    say(f"{a.name}: train {len(S['train'])} val {len(S['val'])} test {len(S['test'])} graphs | params {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M | {DEV}")

    # ---- stage 1: IK module (one sample per scene: labels merged over the scene's queries)
    scenes = {}
    for g in S["train"]:
        if g["IKm"].any():
            s = scenes.setdefault(g["rid"], dict(x=g["x"][g["t"]], IK=np.zeros(16, np.float32), m=np.zeros(16, np.float32)))
            new = g["IKm"] > 0
            s["IK"][new] = g["IK"][new]; s["m"][new] = 1
    X = torch.tensor(np.stack([s["x"] for s in scenes.values()]), dtype=torch.float32, device=DEV)
    Y = torch.tensor(np.stack([s["IK"] for s in scenes.values()]), device=DEV)
    M = torch.tensor(np.stack([s["m"] for s in scenes.values()]), device=DEV)
    opt = torch.optim.Adam(model.IKModule.parameters(), lr=1e-3)
    for ep in range(a.stage_epochs):
        model.train()
        perm = torch.randperm(len(X))
        for k in range(0, len(X), 8192):
            i = perm[k:k + 8192]
            loss = masked_bce(model.IKModule(X[i]), Y[i], M[i])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 25 == 0 or ep == a.stage_epochs - 1:
            say(f"  [IK] ep {ep:3d} loss {loss.item():.4f} ({len(X)} scenes, {int(M.sum())} labelled views)")

    # ---- stage 2: GO module on (target, neighbour) pairs of the fully labelled graphs
    xs, ys, ms = [], [], []
    for g in S["train"]:
        if not g["gom"].any():
            continue
        t = g["t"]
        views = (g["IK"] > 0.5).astype(np.float32)
        for e in np.where(g["blocking"])[0]:
            xs.append(np.concatenate([g["x"][t], g["x"][g["src"][e]]])); ys.append(g["go"][e]); ms.append(views * g["IKm"])
    X = torch.tensor(np.stack(xs), dtype=torch.float32, device=DEV)
    Y = torch.tensor(np.stack(ys), device=DEV)
    M = torch.tensor(np.stack(ms), device=DEV)
    opt = torch.optim.Adam(model.GOModule.parameters(), lr=1e-3)
    for ep in range(a.stage_epochs):
        model.train()
        perm = torch.randperm(len(X))
        for k in range(0, len(X), 8192):
            i = perm[k:k + 8192]
            loss = masked_mse(model.GOModule(X[i]), Y[i], M[i])
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 25 == 0 or ep == a.stage_epochs - 1:
            say(f"  [GO] ep {ep:3d} loss {loss.item():.4f} ({len(X)} pairs)")

    # ---- stage 3: joint GRN
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    yv = np.array([g["y"] for g in S["val"]])
    best, best_ep = -1.0, -1
    for ep in range(a.epochs):
        model.train()
        t0 = time.time()
        idx = list(range(len(S["train"])))
        rng.shuffle(idx)
        tot, n, parts = 0.0, 0, np.zeros(3)
        for k in range(0, len(idx), a.bs):
            b = collate([S["train"][i] for i in idx[k:k + a.bs]], DEV)
            loss, p = grn_loss(model, b)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); n += 1; parts += [p["F"], p["IK"], p["GO"]]
        pv = predict(model, S["val"])[:, 0]
        au, acc = roc_auc_score(yv, pv), float(np.mean((pv >= 0.5) == yv))
        mark = ""
        if au > best:
            best, best_ep, mark = au, ep, " *"
            torch.save(model.state_dict(), out / "best.pt")
        say(f"ep {ep:3d} loss {tot / n:.4f} (F {parts[0] / n:.3f} IK {parts[1] / n:.3f} GO {parts[2] / n:.4f}) | val action AUROC {au:.4f} acc@0.5 {acc:.4f} | {time.time() - t0:.0f}s{mark}")
    model.load_state_dict(torch.load(out / "best.pt", map_location=DEV))
    yt = np.array([g["y"] for g in S["test"]])
    pt = predict(model, S["test"])[:, 0]
    say(f"best epoch {best_ep}: val AUROC {best:.4f} | test action AUROC {roc_auc_score(yt, pt):.4f} acc@0.5 {np.mean((pt >= 0.5) == yt):.4f}")


if __name__ == "__main__":
    main()
