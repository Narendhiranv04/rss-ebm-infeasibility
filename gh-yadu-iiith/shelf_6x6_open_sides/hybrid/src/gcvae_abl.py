"""E7 (EXPERIMENT_LOG.md): does teacher error limit the goal-aware cVAE?  Pre-registered ablation of gcvae.py; no physics, TEST never read.

Everything is gcvae.py's procedure (goal-marker features, w(c) ∝ q_old(c) exp(beta logit t(c)), exact expected loss, 30 epochs, checkpoint =
lowest held-out weighted loss on the same 85/15 TRAIN root split, seeds 0/1/2, init from runs/SAMPLER_V1/cvae_s{seed}); only t, beta or the
model change.  beta is fixed per variant (no selection).
    base         t = Boltzmann 3-seed mean p11, beta 10  == E6: runs/GCVAE/s*.pt (chosen beta 10, same split/seeds/code) are copied, not retrained
    beta1        base teacher, beta 1
    pessimistic  t = min over the 3 Boltzmann seeds
    hazard       t = hazard.Hazard 3-seed mean
    agree        t = sqrt(Boltzmann mean * hazard mean)
    anchored     Boltzmann mean, cells with a TRAIN label set to 0.95 (success) / 0.05 (failure)
    nolatent     base teacher, models.Categorical + goal-marker row; loss sum_c w(c) (-log p(c)); init: encoder = cvae_s{seed} encoder,
                 decoder = cvae_s{seed} decoder with the z columns of its first layer dropped (i.e. the cVAE decoder at z = 0)

    python ../hybrid/src/gcvae_abl.py run         (from approachb_3d_learning/ -> runs/GCVAE_ABL/)
    python ../hybrid/src/gcvae_abl.py selftest
"""
import json, math, random, shutil, sys, time
from collections import defaultdict
from pathlib import Path
from unittest import mock

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gcvae as G, hazard as HZ, phase3 as P3  # noqa: E402

V, Mo = G.V, G.Mo
OUT = V.A / "runs/GCVAE_ABL"
VARIANTS = {"base": ("boltz", 10.0), "beta1": ("boltz", 1.0), "pessimistic": ("min", 10.0), "hazard": ("hazard", 10.0),
            "agree": ("agree", 10.0), "anchored": ("anchored", 10.0), "nolatent": ("boltz", 10.0)}


def teacher_rows(Mm, Hz, keys, leg):
    """-> per-seed Boltzmann p11 and hazard p, each [n, 3, 36] (illegal cells 0), for contexts keys[i] with legal mask leg[i]"""
    B, H = torch.zeros(len(keys), 3, 36), torch.zeros(len(keys), 3, 36)
    for i, (rid, prefix, o) in enumerate(keys):
        raw, L = Mm.load(rid), torch.nonzero(leg[i]).squeeze(1).tolist()
        seqs = [list(prefix) + [(o, (j // 6, j % 6))] for j in L]
        B[i][:, L], H[i][:, L] = Mm.p11(raw, seqs), Hz(raw, seqs)
    return B, H


def teacher(kind, Bs, Hs, labs):
    bm = Bs.mean(1)
    if kind == "anchored":
        bm = bm.clone()
        for i, lab in enumerate(labs):
            for (r, c), s in lab.items():
                bm[i, r * 6 + c] = 0.95 if s else 0.05
    return {"boltz": bm, "anchored": bm, "min": Bs.min(1).values, "hazard": Hs.mean(1), "agree": (bm * Hs.mean(1)).sqrt()}[kind]


def load_cat_from_cvae(path, key="state"):
    """Categorical initialised from a cVAE checkpoint: same encoder, decoder = cVAE decoder at z = 0"""
    sd, m = torch.load(path, weights_only=False)[key], Mo.build("cat", G.CFG)
    if "dec.0.weight" in sd and sd["dec.0.weight"].shape[1] != m.dec[0].in_features:
        sd = {k: v[:, :m.dec[0].in_features] if k == "dec.0.weight" else v for k, v in sd.items() if not k.startswith("post.")}
    m.load_state_dict(sd)
    return m.eval()


def wloss_cat(m, f, msk, occ, leg, w):
    return -(w * m.logits(f, msk, occ, leg).log_softmax(-1)).sum(1).mean()


def train(name, T, fit, hold, seed):
    if name != "nolatent":
        return G.train(T, fit, hold, VARIANTS[name][1], seed)
    # same loop as gcvae.train (optimiser, epochs, checkpoint rule); only the model and its loss are swapped in
    with mock.patch.object(G, "load_cvae", load_cat_from_cvae), mock.patch.object(G, "wloss", wloss_cat):
        return G.train(T, fit, hold, VARIANTS[name][1], seed)


def load(name, s):
    p = OUT / name / f"s{s}.pt"
    return load_cat_from_cvae(p) if name == "nolatent" else G.load_cvae(p)


@torch.no_grad()
def mixq(models, f, msk, occ, leg):
    """3-seed mixture log q over all 36 cells (illegal -inf): IWAE k=64 for cVAEs (gcvae.mix), exact for the categorical"""
    torch.manual_seed(0)
    if isinstance(models[0], Mo.Categorical):
        lq = torch.stack([m.logits(f, msk, occ, leg).log_softmax(-1) for m in models]).logsumexp(0) - math.log(len(models))
        return lq.masked_fill(~leg, -math.inf)
    return G.mix(models, f, msk, occ, leg)


def split(keys):
    roots = sorted({k[0] for k in keys})
    random.Random(0).shuffle(roots)
    held = set(roots[:round(G.HOLD * len(roots))])
    ish = torch.tensor([k[0] in held for k in keys])
    return torch.nonzero(~ish).squeeze(1), torch.nonzero(ish).squeeze(1)


def ctx_picks(Mm, models):
    out = []
    for p in json.loads((P3.OUT / "picks_ctx.json").read_text()):
        raw = Mm.load(p["root"])
        cells = {**{k: tuple(v) for k, v in raw["initial_cells"].items()}, **{o: tuple(c) for o, c in p["prefix"]}}
        X = [torch.as_tensor(x)[None] for x in G.feats(raw, cells, p["object"])]
        X[2] = X[2].float()
        lq = mixq(models, *X)[0]
        out.append([[j // 6, j % 6] for j in lq.topk(min(P3.M, int(X[3].sum()))).indices.tolist()])
    return out


def cmd_run():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ck = torch.load(G.OUT / "teacher_cache.pt", weights_only=False)
    Tr, Va, keys, labs = ck["train"]["T"], ck["val"]["T"], ck["train"]["keys"], ck["train"]["labs"]
    Mm = P3.Models()
    tf = OUT / "teachers_train.pt"
    if not tf.exists():
        Bs, Hs = teacher_rows(Mm, HZ.Hazard(), keys, Tr["leg"])
        assert (Bs.mean(1) - Tr["t"]).abs().max() < 1e-4  # recomputed Boltzmann mean == gcvae's cached teacher
        torch.save({"B": Bs, "H": Hs}, tf)
        print(json.dumps({"teachers": len(keys), "s": round(time.time() - t0)}), flush=True)
    tc = torch.load(tf, weights_only=False)
    fit, hold = split(keys)
    e6 = json.loads((G.OUT / "metrics.json").read_text())
    assert e6["chosen_beta"] == 10.0 and (e6["fit_contexts"], e6["heldout_contexts"]) == (len(fit), len(hold))
    met = {"rules": __doc__.split("\n\n")[1], "fit_contexts": len(fit), "heldout_contexts": len(hold), "val_contexts": len(ck["val"]["keys"]),
           "variants": {}}
    picks, labs_v = {}, ck["val"]["labs"]
    for name, (kind, beta) in VARIANTS.items():
        (OUT / name).mkdir(exist_ok=True)
        t1, info = time.time(), []
        T = {**Tr, "t": teacher(kind, tc["B"], tc["H"], labs)}
        for s in (0, 1, 2):
            if name == "base":
                shutil.copy(G.OUT / f"s{s}.pt", OUT / name / f"s{s}.pt"); info.append(e6["seeds"][s]); continue
            m, inf = train(name, T, fit, hold, s)
            torch.save({"state": m.state_dict(), "variant": name, "kind": m.kind, "teacher": kind, "beta": beta, "seed": s, "cfg": G.CFG,
                        "feat": "data3d.FEAT + goal marker row (gcvae.feats)"}, OUT / name / f"s{s}.pt")
            info.append(inf)
            print(json.dumps({"variant": name, "seed": s, "best_epoch": inf["best_epoch"], "s": round(time.time() - t0)}), flush=True)
        models = [load(name, s) for s in (0, 1, 2)]
        res = defaultdict(lambda: {"hits": [], "p": [0, 0], "t1": []})
        G.topk_metrics(name, mixq(models, Va["f"], Va["msk"], Va["occ"], Va["leg"]), Va["leg"], labs_v, Va["t"], res)
        r = res[name]
        met["variants"][name] = {"teacher": kind, "beta": beta, "val_hit@4": round(float(np.mean(r["hits"])), 3),
                                 "val_known_label_precision@4": round(r["p"][0] / max(1, r["p"][1]), 3), "val_labelled_in_top4": r["p"][1],
                                 "val_mean_boltzmann_p11_top1": round(float(np.mean(r["t1"])), 3),
                                 "train_s": None if name == "base" else round(time.time() - t1),
                                 "seeds": [{k: i[k] for k in ("seed", "best_epoch", "init_hold_loss", "best_hold_loss")} for i in info]}
        picks[name] = ctx_picks(Mm, models)
        print(json.dumps({name: {k: v for k, v in met["variants"][name].items() if k != "seeds"}}), flush=True)
    e6p = json.loads((V.A / "runs/PHASE4/picks_ctx_gcvae.json").read_text())
    agree = sum(a[0] == b["gcvae"][0] for a, b in zip(picks["base"], e6p))
    assert agree >= 0.8 * len(e6p), agree  # base reuses E6 models; only IWAE noise can differ
    met["base_vs_E6_top1_agreement"] = f"{agree}/{len(e6p)}"
    met["wall_s"] = round(time.time() - t0)
    (OUT / "picks_ctx.json").write_text(json.dumps({"variants": picks}))
    (OUT / "metrics.json").write_text(json.dumps(met, indent=1))
    print(json.dumps({"base_vs_E6_top1_agreement": met["base_vs_E6_top1_agreement"], "wall_s": met["wall_s"]}))


def selftest():
    torch.set_num_threads(1)
    ck = torch.load(G.OUT / "teacher_cache.pt", weights_only=False)
    Tr, keys, labs = ck["train"]["T"], ck["train"]["keys"], ck["train"]["labs"]
    assert all(V.XD.qc_roots()[k[0]] == "train" for k in keys)
    idx = [0, 1, 2]
    Bs, Hs = teacher_rows(P3.Models(), HZ.Hazard(), [keys[i] for i in idx], Tr["leg"][idx])
    assert (Bs.mean(1) - Tr["t"][idx]).abs().max() < 1e-4                                   # same teacher as gcvae's cache
    leg = Tr["leg"][idx]
    t = {k: teacher(k, Bs, Hs, [labs[i] for i in idx]) for k in ("boltz", "min", "hazard", "agree", "anchored")}
    assert (t["min"] <= t["boltz"] + 1e-7).all() and ((t["hazard"][leg] > 0) & (t["hazard"][leg] <= 1)).all()
    lo, hi = torch.minimum(t["boltz"], t["hazard"]), torch.maximum(t["boltz"], t["hazard"])
    assert ((t["agree"] >= lo - 1e-6) & (t["agree"] <= hi + 1e-6)).all() and (t["boltz"][~leg] == 0).all()
    for j, i in enumerate(idx):
        for (r, c), s in labs[i].items():
            assert t["anchored"][j, r * 6 + c] == (0.95 if s else 0.05)
        un = [k for k in range(36) if (k // 6, k % 6) not in labs[i]]
        assert torch.equal(t["anchored"][j, un], t["boltz"][j, un])
    # nolatent init == cVAE decoder at z = 0; expected CE with one-hot w == cross entropy; mixture normalised over legal cells
    p = V.A / "runs/SAMPLER_V1/cvae_s0/best.pt"
    cv, cat = G.load_cvae(p), load_cat_from_cvae(p)
    X = [Tr[k][idx] for k in ("f", "msk", "occ", "leg")]
    with torch.no_grad():
        c = cv.enc(*X[:3])
        assert torch.allclose(cv.dec(torch.cat([c, torch.zeros(len(idx), cv.zdim)], -1)), cat.dec(cat.enc(*X[:3])), atol=1e-5)
        d = leg.float().argmax(1)
        assert torch.allclose(wloss_cat(cat, *X, torch.nn.functional.one_hot(d, 36).float()), cat.loss(*X, d)[0], atol=1e-5)
        lq = mixq([cat, cat], *X)
        assert torch.allclose(lq.logsumexp(1), torch.zeros(len(idx)), atol=1e-4) and torch.isinf(lq[~leg]).all()
    w = G.target({**Tr, "t": t["agree"].new_zeros(len(Tr["t"]), 36).index_copy(0, torch.tensor(idx), t["agree"])}, 10.0)[idx]
    assert torch.allclose(w.sum(1), torch.ones(len(idx))) and (w[~leg] == 0).all()
    fit, hold = split(keys)
    assert not set(keys[i][0] for i in fit) & set(keys[i][0] for i in hold)
    print("selftest ok")


if __name__ == "__main__":
    {"run": cmd_run, "selftest": selftest}[sys.argv[1]]()
