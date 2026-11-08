"""Figures from real runs only: learning curves, hit / known-label precision vs proposal budget M, per-stratum results, scene overlays.

    taskset -c 7 venv/bin/python src/figs.py --config configs/sampler_v1.json --split val     (overlays sample on the GPU under the guard)"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
COL = {"cvae": "#2a6fdb", "fm": "#d9534f", "uniform": "#7f7f7f", "stratified": "#b8860b", "ranker": "#2ca02c",
       "cvae_geo": "#8fb3ef", "fm_geo": "#eda9a7"}


def agg(rows, M, key):
    v = [r["by_M"][str(M)][key] for r in rows if key in r["by_M"][str(M)]]
    return (float(np.mean(v)), float(np.std(v))) if v else (np.nan, np.nan)


def curves(exp, out):
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for i, kind in enumerate(("cvae", "fm", "ranker")):
        for p in sorted(exp.glob(f"{kind}*_s*/curve.json")):
            c = json.loads(p.read_text())
            ls = "--" if "_geo" in p.parent.name else "-"
            ax[i].plot([e["epoch"] for e in c], [e["train_loss"] for e in c], ls, color=COL[kind], alpha=0.35)
            ax[i].plot([e["epoch"] for e in c], [e["val_loss"] for e in c], ls, color=COL[kind], label=p.parent.name)
        ax[i].set_title({"cvae": "cVAE: CE + beta KL (val at final beta)", "fm": "Flow matching: velocity MSE", "ranker": "Ranker: BCE (labelled trials)"}[kind])
        ax[i].set_xlabel("epoch"); ax[i].legend(fontsize=7)
    fig.suptitle("Learning curves (faint = train, solid = val; dashed = geometry-only ablation)")
    fig.tight_layout(); fig.savefig(out / "learning_curves.png", dpi=130); plt.close(fig)


def budget(res, out, split):
    keys = [("hit_at_M", "hit@M (contexts with >=1 known positive)"), ("known_label_precision", "known-label precision"),
            ("pos_recall", "recall of recorded positives"), ("duplicate_rate", "duplicate rate (raw budget)")]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4))
    for fam, d in res["samplers"].items():
        Ms = res["M"]
        for i, (k, t) in enumerate(keys):
            mu = [agg(d["rows"], M, k) for M in Ms]
            if all(np.isnan(m[0]) for m in mu):
                continue
            ax[i].errorbar(Ms, [m[0] for m in mu], yerr=[m[1] for m in mu], marker="o", capsize=3, label=fam, color=COL.get(fam, None),
                           ls="--" if fam.endswith("_geo") else "-")
            ax[i].set_title(t); ax[i].set_xscale("log", base=2); ax[i].set_xlabel("unique proposal budget M")
    ax[0].legend(fontsize=8)
    fig.suptitle(f"{split.upper()}: same contexts, same legality mask, raw cap 8M, first M unique (mean +- std over train x eval seeds)")
    fig.tight_layout(); fig.savefig(out / f"budget_{split}.png", dpi=130); plt.close(fig)


def strata(res, out, split, M=4):
    fams = [f for f in res["samplers"] if not f.startswith("ranker")]
    sk = list(res["samplers"][fams[0]]["rows"][0]["by_M"][str(M)]["by_stratum"].keys())
    fig, ax = plt.subplots(1, len(sk), figsize=(5 * len(sk), 4))
    for j, k in enumerate(sk):
        vals = list(res["samplers"][fams[0]]["rows"][0]["by_M"][str(M)]["by_stratum"][k].keys())
        x = np.arange(len(vals)); w = 0.8 / len(fams)
        for i, f in enumerate(fams):
            ys = [np.mean([r["by_M"][str(M)]["by_stratum"][k][v]["known_label_precision"] for r in res["samplers"][f]["rows"]]) for v in vals]
            ax[j].bar(x + i * w, ys, w, label=f, color=COL.get(f))
        ax[j].set_xticks(x + 0.4 - w / 2); ax[j].set_xticklabels(vals, rotation=30); ax[j].set_title(f"known-label precision by {k} (M={M})")
        ax[j].set_ylim(0.4, 1.0)
    ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out / f"strata_{split}.png", dpi=130); plt.close(fig)


def overlays(cfg, exp, out, split, n_scenes=6, nsamp=256):
    import torch, guard, train as TR
    from data3d import Data
    _, dev = TR.preflight(cfg, guard.load_cfg(), False)
    D = Data(ROOT / cfg["derived"], dev)
    C = D.contexts(split)
    C = C[(D.c_pos[C].sum(1) >= 2) & (D.c_neg[C].sum(1) >= 1)]
    g = torch.Generator().manual_seed(7)
    pick = C[torch.randperm(len(C), generator=g)[:n_scenes].to(C.device)]
    fams = {}
    for kind in ("cvae", "fm"):
        p = sorted(exp.glob(f"{kind}_s0/best.pt"))
        if p:
            fams[kind] = TR.load_model(p[0], dev)[0]
    from models import Uniform
    fams["uniform"] = Uniform()
    fig, axes = plt.subplots(len(pick), len(fams), figsize=(4 * len(fams), 4 * len(pick)))
    r = D.c_row[pick]
    obj_raw = D.obj[r].cpu().numpy(); mask = D.mask[r].cpu().numpy()
    for j, (name, m) in enumerate(fams.items()):
        gen = torch.Generator(device=dev).manual_seed(0)
        kw = {}
        d, _ = m.sample(D.obj[r], D.mask[r], D.occ[r], D.legal[r], nsamp, gen=gen, **kw)
        freq = torch.nn.functional.one_hot(d, 36).sum(1).float().cpu().numpy() / nsamp
        for i in range(len(pick)):
            ax = axes[i, j]
            ax.imshow(freq[i].reshape(6, 6), cmap="Blues", vmin=0, vmax=max(0.15, freq[i].max()), origin="lower")
            occ = D.occ[r[i]].cpu().numpy().reshape(6, 6)
            for rr, cc in zip(*np.nonzero(occ)):
                ax.add_patch(plt.Rectangle((cc - 0.45, rr - 0.45), 0.9, 0.9, fill=True, color="#555555", alpha=0.6))
            qr, qc = int(round(float(D.meta["src"][int(r[i])]) // 6)), int(D.meta["src"][int(r[i])] % 6)
            ax.add_patch(plt.Rectangle((qc - 0.45, qr - 0.45), 0.9, 0.9, fill=False, ec="#ff7f0e", lw=3))
            pos = D.c_pos[pick[i]].cpu().numpy().reshape(6, 6); neg = D.c_neg[pick[i]].cpu().numpy().reshape(6, 6)
            for rr, cc in zip(*np.nonzero(pos)):
                ax.add_patch(plt.Circle((cc, rr), 0.35, fill=False, ec="#2ca02c", lw=2.5))
            for rr, cc in zip(*np.nonzero(neg)):
                ax.plot(cc, rr, "x", color="#d62728", ms=14, mew=3)
            ax.set_xticks(range(6)); ax.set_yticks(range(6))
            if i == 0:
                ax.set_title(f"{name}: proposal frequency ({nsamp} draws)")
            if j == 0:
                ax.set_ylabel(f"{D.meta['qid'][int(r[i])]}\n{D.meta['shape'][int(r[i])]}, {D.meta['height_bin'][int(r[i])]}, {D.meta['block'][int(r[i])]}", fontsize=8)
    fig.suptitle("Real VAL scenes (row 0 = nearest the robot). grey = occupied, orange = held object's source cell, green ring = recorded VERIFIED,\n"
                 "red x = recorded negative (certified geometry / under policy); unmarked cells are UNTESTED, not negative", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.965]); fig.savefig(out / f"overlays_{split}.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True); ap.add_argument("--split", default="val"); ap.add_argument("--no-overlays", action="store_true")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    exp = ROOT / "runs" / cfg["experiment"]
    out = exp / "figures"; out.mkdir(exist_ok=True)
    curves(exp, out)
    res = json.loads((exp / f"eval_{a.split}.json").read_text())
    budget(res, out, a.split); strata(res, out, a.split)
    if not a.no_overlays:
        overlays(cfg, exp, out, a.split)
    print(json.dumps({"figures": sorted(p.name for p in out.glob("*.png"))}))
