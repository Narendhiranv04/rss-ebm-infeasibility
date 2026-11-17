#!/usr/bin/env python3
"""Train PIGINet on the shelf66 plan-feasibility samples (build_data.py), V4 train split; model selection on V4 val.

    python train.py --name full_s0 [--seed 0] [--no-image] [--no-value] [--epochs 60] [--threads 4]

Recipe: the released run's settings as far as they are known (Sajid's recovery of the wandb config: AdamW, lr 3e-4, 4 layers,
2 heads, d 256) + the paper's loss (BCE with positive weight = negatives / positives) and stopping rule (keep the epoch with the
best validation score).  Image features are the cached frozen-CLIP crops (render_feats.py), so the paper's image augmentation
does not apply; the initial-state tokens are shuffled every epoch instead.
Writes runs/<name>/{best.pt, train.log, config.json, pred_val.pkl, pred_test.pkl}.
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

from piginet import PIGINet, collate, tokens

HERE = Path(__file__).resolve().parent
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_FEATS = {}


def feats_of(sample):
    rid = sample["rid"]
    if rid not in _FEATS:
        f = HERE / "data" / "feats" / f"{rid}.pkl"
        _FEATS[rid] = pickle.load(open(f, "rb")) if f.exists() else None
    d = _FEATS[rid]
    if d is None:
        return None
    if sample.get("img_src"):                 # lattice edit: each object's crop from the render in which it stands at that cell
        return {o: d[st][o] for o, st in sample["img_src"].items() if st in d and o in d[st]}
    return d.get(f"D{sample['idx']}" if sample["kind"] == "D" else "orig")


def batches(samples, bs, use_image, rng=None):
    idx = list(range(len(samples)))
    if rng is not None:
        rng.shuffle(idx)
    for k in range(0, len(idx), bs):
        items = []
        for i in idx[k:k + bs]:
            s = samples[i]
            items.append((tokens(s, feats_of(s) if use_image else None, rng), s["y"]))
        yield {k: v.to(DEV) for k, v in collate(items).items()}


@torch.no_grad()
def predict(model, samples, use_image, bs=512):
    model.eval()
    out = []
    for b in batches(samples, bs, use_image):
        out.append(torch.sigmoid(model(b["img"], b["v1"], b["v2"], b["kind"], b["step"], b["mask"])).float().cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--no-image", action="store_true")
    ap.add_argument("--no-value", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    rng = random.Random(a.seed)
    out = HERE / "runs" / a.name
    out.mkdir(parents=True, exist_ok=True)
    d = pickle.load(open(HERE / "data" / "samples.pkl", "rb"))
    S = {p: [s for s in d["samples"] if s["split"] == p] for p in ("train", "val", "test")}
    use_image = not a.no_image
    if use_image:
        miss = sum(1 for p in S for s in S[p] if feats_of(s) is None)
        if miss:
            raise SystemExit(f"{miss} samples have no image features: run render_feats.py first (or pass --no-image)")
    model = PIGINet(use_image=use_image, use_value=not a.no_value).to(DEV)
    model.set_text(np.load(HERE / "data" / "text.npz")["feats"])
    pos = float(np.mean([s["y"] for s in S["train"]]))
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([(1 - pos) / pos], device=DEV))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)
    cfg = dict(vars(a), train=len(S["train"]), val=len(S["val"]), test=len(S["test"]), pos=pos, pos_weight=(1 - pos) / pos,
               params=sum(p.numel() for p in model.parameters()), split_sha=d.get("split_sha"))
    json.dump(cfg, open(out / "config.json", "w"), indent=1)
    log = open(out / "train.log", "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n"); log.flush()

    say(f"{a.name}: train {len(S['train'])} (feasible {pos:.3f}, pos_weight {(1 - pos) / pos:.2f}) val {len(S['val'])} test {len(S['test'])} "
        f"| params {cfg['params'] / 1e6:.2f}M | image {use_image} value {not a.no_value} | {DEV}")
    yv = np.array([s["y"] for s in S["val"]])
    best, best_ep = -1.0, -1
    for ep in range(a.epochs):
        model.train()
        t0, tot, n = time.time(), 0.0, 0
        for b in batches(S["train"], a.bs, use_image, rng):
            loss = lossf(model(b["img"], b["v1"], b["v2"], b["kind"], b["step"], b["mask"]), b["y"])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(b["y"]); n += len(b["y"])
        sched.step()
        pv = predict(model, S["val"], use_image)
        au, acc = roc_auc_score(yv, pv), float(np.mean((pv >= 0.5) == yv))
        mark = ""
        if au > best:
            best, best_ep, mark = au, ep, " *"
            torch.save(model.state_dict(), out / "best.pt")
        say(f"ep {ep:3d} loss {tot / n:.4f} | val AUROC {au:.4f} acc@0.5 {acc:.4f} | {time.time() - t0:.0f}s{mark}")
    model.load_state_dict(torch.load(out / "best.pt", map_location=DEV))
    for p in ("val", "test"):
        pr = predict(model, S[p], use_image)
        pickle.dump(dict(p=pr, y=np.array([s["y"] for s in S[p]])), open(out / f"pred_{p}.pkl", "wb"))
        say(f"best epoch {best_ep}: {p} AUROC {roc_auc_score([s['y'] for s in S[p]], pr):.4f} acc@0.5 {np.mean((pr >= 0.5) == np.array([s['y'] for s in S[p]])):.4f}")


if __name__ == "__main__":
    main()
