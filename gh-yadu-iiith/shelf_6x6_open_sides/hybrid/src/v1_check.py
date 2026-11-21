"""Hybrid v1, critic-guided placement: offline check on VAL labels only (no physics, no test split).

For every VAL context (root, prefix of a labelled plan, object moved last) all legal cells d are scored by
    cvae    q(d | S, o)                      goal-blind placement density (3 seeds, averaged)
    critic  p11(prefix + (o, d))             frozen X/Y Boltzmann success probability
    hybrid  q(d) * p11(...)^lam              product of the two
and the top-M cells are compared with the labelled outcomes (success = X1 and Y1; X0 or Y0 = failure; missing Y = unknown).
Metrics follow reports/04: hit@M (a known-good cell is in the top M) and known-label precision@M (pooled over labelled cells in the top M).

    python ../hybrid/src/v1_check.py --arm D            (run from approachb_3d_learning/)
"""
import argparse, json, math, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

A = Path(__file__).resolve().parents[2] / "approachb_3d_learning"
sys.path.insert(0, str(A / "src"))
import data3d as D, models as Mo, xy3d as XY, xy3d_data as XD  # noqa: E402

LAMS = [0.0, 0.5, 1.0, 2.0, 4.0]  # 0 = cVAE only; reported side by side, none selected (VAL is the only split used)


def parse(key):
    return tuple((m.split(">")[0], tuple(map(int, m.split(">")[1].split(",")))) for m in key.split("."))


def val_contexts(sc):
    y = {(r["root_id"], r["y_key"]): r["y"] for r in sc["y_rows"]}
    split = XD.qc_roots()
    ctx = defaultdict(dict)  # (root, prefix, obj) -> {cell: success 0/1}
    for r in sc["x_rows"]:
        if split.get(r["root_id"]) != "val":
            continue
        yy = y.get((r["root_id"], r["y_key"]))
        s = 0 if r["x"] == 0 else (None if yy is None else int(yy == 1))
        if s is None:
            continue
        seq = parse(r["x_key"])
        ctx[(r["root_id"], seq[:-1], seq[-1][0])][seq[-1][1]] = s
    return ctx


def scorer(arm):
    """-> score(rid, prefix, o) = (legal cells, log q_cVAE per cell [3-seed mixture], p11 per cell), for the hypothetical scene after prefix."""
    cfg = json.loads((A / "configs/sampler_v1.json").read_text())
    scal = json.loads((A / cfg["derived"] / "scalers.json").read_text())
    mu, sd = np.array(scal["mean"], np.float32), np.array(scal["std"], np.float32)
    cvaes = []
    for s in (0, 1, 2):
        m = Mo.build("cvae", cfg); m.load_state_dict(torch.load(A / f"runs/SAMPLER_V1/cvae_s{s}/best.pt", weights_only=False)["state"]); cvaes.append(m.eval())
    ck = torch.load(A / f"runs/XY3D_REAL/smoke_cpu/{arm}_best.pt", weights_only=False)
    critic = XY.XYNet(ck["arm"], 5, 14).to(torch.float32); critic.load_state_dict(ck["state"]); critic.eval()
    raw_cache = {}

    def score(rid, prefix, o):
        raw = raw_cache.setdefault(rid, XD.load(rid))
        cells = {k: tuple(v) for k, v in raw["initial_cells"].items()}
        cells.update(dict(prefix))
        f, msk, occ, leg = D.scene_features(D.allowed(raw)["objects"], cells, o)
        f = (f - mu) / sd * msk[:, None]
        L = [int(i) for i in np.nonzero(leg)[0]]
        cand = [(i // D.COLS, i % D.COLS) for i in L]
        T = lambda x, dt=None: torch.as_tensor(x, dtype=dt)[None].expand(len(L), *np.shape(x))  # noqa: E731
        args = (T(f, torch.float32), T(msk), T(occ, torch.float32), T(leg))
        with torch.no_grad():
            lq = torch.stack([m.log_prob(*args, torch.tensor(L), k=64) for m in cvaes]).logsumexp(0) - math.log(len(cvaes))
            v, _, xm, _ = XD.view3d(raw, extra_seqs=[list(prefix) + [(o, c)] for c in cand], use_history=False)
            b, _ = XY.collate([v], None, "cpu", torch.float32)
            p11 = critic.predict(b)["psucc"][torch.tensor(xm)].clamp_min(1e-9)
        return cand, lq, p11
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="D"); ap.add_argument("--M", type=int, default=4); ap.add_argument("--out")
    a = ap.parse_args()
    torch.manual_seed(0)
    sc = json.loads((A / "runs/XY3D_REAL/BD3DS66_XY_SIDECAR_V1.json").read_text())
    score = scorer(a.arm)
    ctxs = val_contexts(sc)
    n_lab = sum(len(v) for v in ctxs.values()); n_pos_ctx = sum(any(v.values()) for v in ctxs.values())
    hits, prec, unif_hit = defaultdict(list), defaultdict(lambda: [0, 0]), []
    t0 = time.time()
    for (rid, prefix, o), lab in ctxs.items():
        cand, lq, p11 = score(rid, prefix, o)
        assert all(c in cand for c in lab), "a labelled destination is not free in its hypothetical scene"
        good = [c for c, s in lab.items() if s == 1]
        n, P = len(cand), len(good)
        if P:
            unif_hit.append(1 - math.comb(n - P, a.M) / math.comb(n, a.M) if n >= a.M else 1.0)
        scores = {f"hybrid_lam{l:g}" if l else "cvae_only": lq + l * p11.log() for l in LAMS}
        scores["critic_only"] = p11.log()
        for name, s in scores.items():
            top = {cand[i] for i in s.topk(min(a.M, n)).indices.tolist()}
            if P:
                hits[name].append(any(c in top for c in good))
            lab_top = [lab[c] for c in top if c in lab]
            prec[name][0] += sum(lab_top); prec[name][1] += len(lab_top)
    res = {"arm": a.arm, "M": a.M, "split": "val", "contexts": len(ctxs), "contexts_with_known_good": n_pos_ctx, "labelled_cells": n_lab,
           "base_rate_good": round(sum(sum(v.values()) for v in ctxs.values()) / n_lab, 3),
           "uniform_expected_hit": round(float(np.mean(unif_hit)), 3), "wall_s": round(time.time() - t0, 1),
           "methods": {k: {"hit@M": round(float(np.mean(hits[k])), 3), "known_label_precision@M": round(prec[k][0] / max(1, prec[k][1]), 3),
                           "labelled_in_topM": prec[k][1]} for k in hits}}
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
