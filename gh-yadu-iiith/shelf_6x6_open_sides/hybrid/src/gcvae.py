"""Hybrid v2: GOAL-AWARE placement cVAE distilled from the frozen X/Y Boltzmann success model.  No physics; TEST roots are never read.

Rules (fixed before any result was seen):
1. Goal input: one extra "goal marker" token appended to the scene set: shape flags 0, size 0, height 0 (raw units), position features of the
   goal cell (r/5, c/5, x, y, dist, angle), is_query 0, (dr, dc, cheb) w.r.t. the query; then the SAME TRAIN scaler as every object row.
   occ / legal unchanged.  MAXOBJ = 8 and scenes reach 8 objects, so the feature array grows to 9 rows (the SceneEncoder is a set encoder:
   no weight depends on the row count).  Initialised from runs/SAMPLER_V1/cvae_s{seed}/best.pt.
2. Teacher per TRAIN context (root, prefix, object o), scene = initial cells updated with the prefix: t(c) = seed-mean p11(prefix + [(o, c)])
   for every legal c;  w(c) ∝ q_old(c) exp(beta logit t(c)) over legal cells (q_old = 3-seed goal-blind cVAE mixture, IWAE k=64).
   Loss = E_{d~w} ELBO(d) computed exactly as sum_c w(c) ELBO(c) (one z sample per cell), beta_KL = config beta_kl_max.
3. beta in {1, 3, 10} chosen on a held-out 15% of TRAIN contexts (split by root, seed 0 only) by the mean teacher t of the model's top-1 cell
   (argmax IWAE log q); KL(w || q) is reported, not used (it is not comparable across betas).  Checkpoint = lowest held-out weighted loss.
4. Seeds 0, 1, 2 at the chosen beta on the same 85%; VAL (v1_check.val_contexts) is for reporting only.

    python ../hybrid/src/gcvae.py run          (from approachb_3d_learning/; prep + select + train + VAL report -> runs/GCVAE/)
    python ../hybrid/src/gcvae.py selftest
"""
import argparse, json, math, random, sys, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase3 as P3, v1_check as V  # noqa: E402

D, Mo, XD = V.D, V.Mo, V.XD
OUT = V.A / "runs/GCVAE"
CFG = json.loads((V.A / "configs/sampler_v1.json").read_text())
_SC = json.loads((V.A / CFG["derived"] / "scalers.json").read_text())
MU, SD = np.array(_SC["mean"], np.float32), np.array(_SC["std"], np.float32)
BETAS, HOLD, EPOCHS, BS, LR, M = (1.0, 3.0, 10.0), 0.15, 30, 128, 3e-4, 4
BKL = CFG["beta_kl_max"]


def feats(raw, cells, o):
    """scene_features + goal marker row -> normalised (F [MAXOBJ+1, 14], mask, occ, legal)"""
    f, msk, occ, leg = D.scene_features(D.allowed(raw)["objects"], cells, o)
    n, (gr, gc), (qr, qc) = int(msk.sum()), raw["goal"], cells[o]
    x, y = D.cell_xy(gr, gc)
    f, msk = np.concatenate([f, np.zeros((1, f.shape[1]), np.float32)]), np.concatenate([msk, [False]])
    f[n] = [0, 0, 0, 0, gr / 5, gc / 5, x, y, math.hypot(x, y), math.atan2(y, x) / math.pi, 0, (gr - qr) / 5, (gc - qc) / 5,
            D.cheb((gr, gc), (qr, qc)) / 5]
    msk[n] = True
    return (f - MU) / SD * msk[:, None], msk, occ, leg


def contexts(sc, split):
    """v1_check.val_contexts with the split as a parameter: (root, prefix, obj) -> {cell: success 0/1}"""
    assert split != "test"
    y = {(r["root_id"], r["y_key"]): r["y"] for r in sc["y_rows"]}
    sp = XD.qc_roots()
    ctx = defaultdict(dict)
    for r in sc["x_rows"]:
        if sp.get(r["root_id"]) != split:
            continue
        yy = y.get((r["root_id"], r["y_key"]))
        s = 0 if r["x"] == 0 else (None if yy is None else int(yy == 1))
        if s is None:
            continue
        seq = V.parse(r["x_key"])
        ctx[(r["root_id"], seq[:-1], seq[-1][0])][seq[-1][1]] = s
    return ctx


def load_cvae(path, key="state"):
    m = Mo.build("cvae", CFG); m.load_state_dict(torch.load(path, weights_only=False)[key]); return m.eval()


@torch.no_grad()
def iwae_all(m, f, msk, occ, leg, k=64):
    """batched CVAE.log_prob for every one of the 36 cells -> [B, 36] (illegal = -inf)"""
    c = m.enc(f, msk, occ); B = c.shape[0]
    mu, lv = m.post(torch.cat([c[:, None].expand(B, 36, -1), torch.eye(36).expand(B, 36, 36)], -1)).chunk(2, -1)
    z = mu[:, :, None] + torch.randn(B, 36, k, m.zdim) * (0.5 * lv).exp()[:, :, None]
    lp = Mo.masked_logits(m.dec(torch.cat([c[:, None, None].expand(B, 36, k, -1), z], -1)), leg[:, None, None]).log_softmax(-1)
    lpd = lp.diagonal(dim1=1, dim2=3).transpose(1, 2)                                  # lp[b, d, j, d]
    lpz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
    lq = -0.5 * (((z - mu[:, :, None]) / (0.5 * lv).exp()[:, :, None]).pow(2) + lv[:, :, None] + math.log(2 * math.pi)).sum(-1)
    return (torch.logsumexp(lpd + lpz - lq, 2) - math.log(k)).masked_fill(~leg, -math.inf)


def mix(models, *X, bs=64):
    return torch.cat([torch.stack([iwae_all(m, *(x[i:i + bs] for x in X)) for m in models]).logsumexp(0) - math.log(len(models))
                      for i in range(0, len(X[0]), bs)])


def wloss(m, f, msk, occ, leg, w):
    """sum_c w(c) [ -log p(c | z, S, o) + beta_KL KL(q(z | S, o, c) || N(0, I)) ] == E_{d~w} of CVAE.loss"""
    c = m.enc(f, msk, occ); B = c.shape[0]
    mu, lv = m.post(torch.cat([c[:, None].expand(B, 36, -1), torch.eye(36).expand(B, 36, 36)], -1)).chunk(2, -1)
    z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
    rec = -Mo.masked_logits(m.dec(torch.cat([c[:, None].expand(B, 36, -1), z], -1)), leg[:, None]).log_softmax(-1).diagonal(dim1=1, dim2=2)
    kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(-1)
    return (w * (rec + BKL * kl)).sum(1).mean()


def prep(Mm, ctxs):
    """-> tensors per context: goal-aware features, old-feature log q_old [36], teacher t [36], labels"""
    rows = defaultdict(list)
    for (rid, prefix, o), lab in ctxs:
        raw = Mm.load(rid)
        cells = {**{k: tuple(v) for k, v in raw["initial_cells"].items()}, **dict(prefix)}
        f, msk, occ, leg = feats(raw, cells, o)
        f0, m0, _, _ = D.scene_features(D.allowed(raw)["objects"], cells, o)
        L = [int(i) for i in np.nonzero(leg)[0]]
        t = np.zeros(36, np.float32)
        t[L] = Mm.p11(raw, [list(prefix) + [(o, (i // 6, i % 6))] for i in L]).mean(0).numpy()
        for k, v in (("f", f), ("msk", msk), ("occ", occ), ("leg", leg), ("f0", (f0 - MU) / SD * m0[:, None]), ("m0", m0), ("t", t)):
            rows[k].append(v)
    T = {k: torch.as_tensor(np.stack(v)) for k, v in rows.items()}
    T["occ"] = T["occ"].float()
    torch.manual_seed(0)
    T["lq_old"] = mix(Mm.cvaes, T["f0"], T["m0"], T["occ"], T["leg"])
    return T


def target(T, beta):
    lt = T["t"].clamp(1e-6, 1 - 1e-6).logit()
    return (T["lq_old"] + beta * lt).masked_fill(~T["leg"], -math.inf).softmax(1)


def evaluate(models, T, idx, beta=None):
    torch.manual_seed(0)
    lq = mix(models, T["f"][idx], T["msk"][idx], T["occ"][idx], T["leg"][idx])
    out = {"top1_teacher_t": float(T["t"][idx].gather(1, lq.argmax(1, keepdim=True)).mean())}
    if beta is not None:
        w = target(T, beta)[idx]
        out["kl_w_q"] = float((w * (w.clamp_min(1e-30).log() - lq.log_softmax(1).masked_fill(~T["leg"][idx], 0))).sum(1).mean())
    return out, lq


def train(T, fit, hold, beta, seed):
    torch.manual_seed(seed)
    m = load_cvae(V.A / f"runs/SAMPLER_V1/cvae_s{seed}/best.pt").train()
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=CFG["wd"])
    w, X = target(T, beta), [T[k] for k in ("f", "msk", "occ", "leg")]
    hl = lambda: float(wloss(m, *(x[hold] for x in X), w[hold]))  # noqa: E731
    with torch.no_grad():
        init = hl()
    best, curve = (init, 0, {k: v.clone() for k, v in m.state_dict().items()}), []
    g = torch.Generator().manual_seed(seed)
    for ep in range(1, EPOCHS + 1):
        perm = fit[torch.randperm(len(fit), generator=g)]
        for i in range(0, len(perm), BS):
            b = perm[i:i + BS]
            loss = wloss(m, *(x[b] for x in X), w[b]); opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():
            curve.append(round(hl(), 4))
        if curve[-1] < best[0]:
            best = (curve[-1], ep, {k: v.clone() for k, v in m.state_dict().items()})
    m.load_state_dict(best[2])
    return m.eval(), {"beta": beta, "seed": seed, "init_hold_loss": round(init, 4),
                      "best_epoch": best[1], "best_hold_loss": round(best[0], 4), "hold_loss_curve": curve}


def topk_metrics(name, score, cands_leg, labs, t, res):
    """v1_check main's hit@M / known-label precision@M, plus mean teacher t of the top-1"""
    for i, lab in enumerate(labs):
        s = score[i].masked_fill(~cands_leg[i], -math.inf)
        top = [(j // 6, j % 6) for j in s.topk(min(M, int(cands_leg[i].sum()))).indices.tolist()]
        good = [c for c, v in lab.items() if v == 1]
        if good:
            res[name]["hits"].append(any(c in top for c in good))
        lt = [lab[c] for c in top if c in lab]
        res[name]["p"][0] += sum(lt); res[name]["p"][1] += len(lt)
        res[name]["t1"].append(float(t[i, top[0][0] * 6 + top[0][1]]))


def cmd_run(a):
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sc = json.loads((V.A / "runs/XY3D_REAL/BD3DS66_XY_SIDECAR_V1.json").read_text())
    Mm = P3.Models()
    cache = OUT / "teacher_cache.pt"
    if cache.exists():
        ck = torch.load(cache, weights_only=False)
    else:
        ck = {}
        for split in ("train", "val"):
            cx = sorted(contexts(sc, split).items(), key=lambda x: (x[0][0], str(x[0][1]), x[0][2]))
            ck[split] = {"keys": [k for k, _ in cx], "labs": [v for _, v in cx], "T": prep(Mm, cx)}
            print(json.dumps({"prep": split, "contexts": len(cx), "s": round(time.time() - t0)}), flush=True)
        torch.save(ck, cache)
    Tr, Va = ck["train"]["T"], ck["val"]["T"]
    roots = sorted({k[0] for k in ck["train"]["keys"]})
    random.Random(0).shuffle(roots)
    held = set(roots[:round(HOLD * len(roots))])
    ish = torch.tensor([k[0] in held for k in ck["train"]["keys"]])
    fit, hold = torch.nonzero(~ish).squeeze(1), torch.nonzero(ish).squeeze(1)
    met = {"rules": __doc__.split("\n\n")[1], "train_contexts": len(ck["train"]["keys"]), "fit_contexts": len(fit), "heldout_contexts": len(hold),
           "heldout_roots": len(held), "val_contexts": len(ck["val"]["keys"]), "beta_kl": BKL, "epochs": EPOCHS, "lr": LR, "batch": BS,
           "selection_criterion": "max mean teacher t of top-1 (argmax IWAE log q, seed 0) on held-out TRAIN contexts", "selection": {}}
    met["selection"]["old_cvae"] = evaluate(Mm.cvaes, Tr, hold)[0]
    met["selection"]["teacher_top1_t"] = float(Tr["t"][hold].max(1).values.mean())
    runs = {}
    for beta in BETAS:
        m, info = train(Tr, fit, hold, beta, 0)
        runs[beta] = (m, info)
        met["selection"][f"beta{beta:g}"] = {**evaluate([m], Tr, hold, beta)[0], **{k: info[k] for k in ("best_epoch", "best_hold_loss")}}
        print(json.dumps({"beta": beta, **met["selection"][f"beta{beta:g}"], "s": round(time.time() - t0)}), flush=True)
    bstar = max(BETAS, key=lambda b: met["selection"][f"beta{b:g}"]["top1_teacher_t"])
    met["chosen_beta"] = bstar
    models, met["seeds"] = [runs[bstar][0]], [runs[bstar][1]]
    for s in (1, 2):
        m, info = train(Tr, fit, hold, bstar, s); models.append(m); met["seeds"].append(info)
        print(json.dumps({"seed": s, "best_epoch": info["best_epoch"], "s": round(time.time() - t0)}), flush=True)
    for s, m in enumerate(models):
        torch.save({"state": m.state_dict(), "beta": bstar, "seed": s, "cfg": CFG, "feat": "data3d.FEAT + goal marker row (gcvae.feats)"}, OUT / f"s{s}.pt")
    met["heldout_3seed"] = evaluate(models, Tr, hold, bstar)[0]
    # VAL report (v1_check main's metrics; teacher = 3-seed-mean Boltzmann p11)
    labs, leg = ck["val"]["labs"], Va["leg"]
    torch.manual_seed(0)
    lq_new = mix(models, Va["f"], Va["msk"], Va["occ"], leg)
    res = defaultdict(lambda: {"hits": [], "p": [0, 0], "t1": []})
    for name, s in (("old_cvae", Va["lq_old"]), ("goal_cvae", lq_new), ("teacher_boltzmann", Va["t"])):
        topk_metrics(name, s, leg, labs, Va["t"], res)
    met["val"] = {"M": M, "contexts_with_known_good": len(res["old_cvae"]["hits"]),
                  "methods": {k: {"hit@M": round(float(np.mean(v["hits"])), 3), "known_label_precision@M": round(v["p"][0] / max(1, v["p"][1]), 3),
                                  "labelled_in_topM": v["p"][1], "mean_teacher_t_top1": round(float(np.mean(v["t1"])), 3)} for k, v in res.items()}}
    met["wall_s"] = round(time.time() - t0)
    (OUT / "metrics.json").write_text(json.dumps(met, indent=1))
    print(json.dumps({k: met[k] for k in ("chosen_beta", "selection", "heldout_3seed", "val", "wall_s")}, indent=1))


class GCVAE:
    def __init__(self):
        self.models = [load_cvae(OUT / f"s{s}.pt") for s in (0, 1, 2)]

    def logq(self, raw, cells, o, exclude=()):
        """-> legal cells (free, not in exclude), log q (3-seed mixture, IWAE k=64 per cell as phase3.Models.place)"""
        f, msk, occ, leg = feats(raw, cells, o)
        L = [int(i) for i in np.nonzero(leg)[0] if (int(i) // 6, int(i) % 6) not in exclude]
        T = lambda x, dt=None: torch.as_tensor(x, dtype=dt)[None].expand(len(L), *np.shape(x))  # noqa: E731
        args = (T(f, torch.float32), T(msk), T(occ, torch.float32), T(leg))
        with torch.no_grad():
            lq = torch.stack([m.log_prob(*args, torch.tensor(L), k=64) for m in self.models]).logsumexp(0) - math.log(len(self.models))
        return [(i // 6, i % 6) for i in L], lq

    def propose(self, raw, cells, o, gen, n_raw=64, pool=8, exclude=()):
        """phase3b.propose without the Ranker: n_raw draws from the 3-seed mixture, first `pool` unique cells, up to M=4 of them by log q"""
        import train as TR
        f, msk, occ, leg = feats(raw, cells, o)
        for r, c in exclude:
            leg[r * 6 + c] = False
        X = (torch.as_tensor(f)[None], torch.as_tensor(msk)[None], torch.as_tensor(occ, dtype=torch.float32)[None], torch.as_tensor(leg)[None])
        with torch.no_grad():
            d = torch.cat([m.sample(*X, n_raw, gen=gen)[0] for m in self.models])        # [3, n_raw]
            d = d[torch.randint(len(self.models), (n_raw,), generator=gen), torch.arange(n_raw)][None]
            pl = TR.first_unique(d, pool)[0][0]
        cand, lq = self.logq(raw, cells, o, exclude)
        keep = [j for j, (r, c) in enumerate(cand) if pl[r * 6 + c]]
        return [cand[j] for j in sorted(keep, key=lambda j: -float(lq[j]))[:M]]


def selftest():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    sp = XD.qc_roots()
    rid = next(r for r, s in sp.items() if s == "train" and len(XD.load(r)["initial_cells"]) == D.MAXOBJ)  # full scene: marker needs row 9
    raw = XD.load(rid)
    cells = {k: tuple(v) for k, v in raw["initial_cells"].items()}
    o = sorted(k for k in cells if k != "t0")[0]
    f, msk, occ, leg = feats(raw, cells, o)
    assert f.shape == (D.MAXOBJ + 1, len(D.FEAT)) and msk.sum() == len(cells) + 1 and f[0, D.FEAT.index("is_query")] > 0
    assert np.allclose(f[-1, 4:6], np.array(raw["goal"]) / 5) and f[-1, D.FEAT.index("is_query")] == 0
    G = GCVAE()
    cand, lq = G.logq(raw, cells, o)
    assert len(cand) == int(leg.sum()) == len(lq) and torch.isfinite(lq).all() and abs(float(lq.logsumexp(0))) < 0.3, float(lq.logsumexp(0))
    X = [torch.as_tensor(x)[None] for x in (f, msk, occ.astype(np.float32), leg)]
    torch.manual_seed(1)
    ia = mix(G.models, *X)[0, [r * 6 + c for r, c in cand]]
    assert (ia - lq).abs().max() < 0.1, (ia - lq).abs().max()                             # batched IWAE == CVAE.log_prob
    raw2 = {**raw, "goal": list(cand[0])}
    assert (G.logq(raw2, cells, o)[1] - lq).abs().max() > 1e-3                            # goal-aware
    ex = {cand[int(lq.argmax())]}
    p = G.propose(raw, cells, o, torch.Generator().manual_seed(0), exclude=ex)
    c2, l2 = G.logq(raw, cells, o, ex)
    assert 1 <= len(p) <= M and len(set(p)) == len(p) and all(c in c2 and c not in ex for c in p)
    assert [float(l2[c2.index(c)]) for c in p] == sorted((float(l2[c2.index(c)]) for c in p), reverse=True)
    w = torch.zeros(1, 36); w[0, cand[0][0] * 6 + cand[0][1]] = 1
    assert torch.isfinite(wloss(G.models[0], *[x.float() if x.dtype == torch.float64 else x for x in X], w))
    print("selftest ok", rid, len(cand), p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "selftest"])
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else selftest()
