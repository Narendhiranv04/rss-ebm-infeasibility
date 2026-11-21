"""E8 factor-graph plan-success model: P(plan) = prod_k P(move k ok | scene before k) * P(target ok | final scene, moves).

Factors (all exact, no sampling):
  move k     : hazard.Net (own SceneEncoder on the scene before/after move k + [dest, goal, is_target, k/3]) -> sigmoid.
  target     : NOISY-AND  P = sigma(c0) * prod_{o != t0} sigma(g_o).
               c0 = MLP(TargetEnc(final scene, t0 query), TargetEnc(t0 at goal), [dest, goal, 1, k/3], m)   ("everything else ok")
               g_o = MLP(TargetEnc context of the final scene, o's 14 scene features (incl. offsets to t0), 8 geometric features of o
                         relative to t0 / goal / the t0->goal segment, goal r/5 c/5)  = logit P(o does NOT block).
               Each "o blocks" is a binary latent; for independent blockers P(target ok) = sigma(c0) * prod_o P(o not blocking) exactly.
  X/Y coupling: c0 also takes m = clamp(min_k move-logit_k, -8, 8)/4 (8 when there are no moves), DETACHED so the target labels never
               train the move factors.  So a plan whose moves were "hard" can have a lower target probability; X and Y are not forced independent.

Training (TRAIN roots only; VAL for early stopping; TEST never read): step-level BCE exactly as hazard.py (same per-step labels, dedupe and
no pos_weight), PLUS 0.3 (fixed a priori) x BCE of P(o blocks | INITIAL scene) against labels.blocker.minimal_sets on roots with K >= 1 and
non-empty minimal_sets: o in the union -> 1, other non-target objects -> 0.  (K = 0 roots have blocker_supervision_valid = false: unused.)
Within an epoch m uses move logits computed at the start of the epoch (detached anyway); validation and inference are exact.

    python ../hybrid/src/factor.py train       (3 seeds -> runs/FACTOR/s{seed}.pt, metrics.json)
    python ../hybrid/src/factor.py eval        (VAL plan AUROC / Brier / reliability vs hazard & Boltzmann; blocker metrics -> eval.json)
    python ../hybrid/src/factor.py selftest
"""
import json, math, sys, time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as Fn

import hazard as HZ
from hazard import A, H, SEEDS, D, Mo, XD, plan_steps, stype, metrics, random_plans

OUT = A / "runs/FACTOR"
NG, AUX_W, BIG = 8, 0.3, 8.0


def geo(raw, cells):
    """-> [MAXOBJ, NG] per-object features in scene_features order (t0 first, row 0 zero): offsets to goal, Chebyshev to goal, on the
    t0->goal line, inside the t0/goal bounding box, distance to the segment, inside the 2-cell hand approach of t0 / of the goal."""
    t, g = cells["t0"], tuple(raw["goal"])
    line, G = set(XD.line_cells(t, g)), np.zeros((D.MAXOBJ, NG), np.float32)
    hand = lambda a: {(a[0] + k * dr, a[1] + k * dc) for dr, dc in XD.DIRS for k in (1, 2)}  # noqa: E731
    ht, hg = hand(t), hand(g)
    (ar, ac), (br, bc) = t, g
    for i, o in enumerate(sorted(o for o in cells if o != "t0"), 1):
        r, c = cells[o]
        L2 = (br - ar) ** 2 + (bc - ac) ** 2
        s = 0.0 if L2 == 0 else min(1.0, max(0.0, ((r - ar) * (br - ar) + (c - ac) * (bc - ac)) / L2))
        G[i] = [(r - g[0]) / 5, (c - g[1]) / 5, XD.cheb((r, c), g) / 5, (r, c) in line,
                min(ar, br) <= r <= max(ar, br) and min(ac, bc) <= c <= max(ac, bc),
                math.hypot(r - ar - s * (br - ar), c - ac - s * (bc - ac)) / 5, (r, c) in ht, (r, c) in hg]
    return G


def featurize(raw, steps):
    """hazard.featurize + G [n, MAXOBJ, NG] of the scene BEFORE each step (only used for target steps)"""
    cells0 = {k: tuple(v) for k, v in raw["initial_cells"].items()}
    G = torch.from_numpy(np.stack([geo(raw, {**cells0, **st[1]}) for st in steps]))
    return (*HZ.featurize(raw, steps), G)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.move, self.tenc = HZ.Net(), Mo.SceneEncoder(14, H)
        self.c0, self.g = Mo.mlp(2 * H + 6 + 1, H, 1), Mo.mlp(H + 14 + NG + 2, H, 1)
        nn.init.constant_(self.g[-1].bias, 3.0)  # start at P(o not blocking) ~ 0.95

    def move_logit(self, F, M, O, X, G=None):
        return self.move(F, M, O, X)

    def target_terms(self, F, M, O, X, G, m):
        """-> c0 logit [n], g logits [n, MAXOBJ-1] (logit P(o not blocking)), object mask [n, MAXOBJ-1]"""
        n = len(F)
        e = self.tenc(F.flatten(0, 1), M.flatten(0, 1), O.flatten(0, 1)).view(n, 2, H)
        c0 = self.c0(torch.cat([e.flatten(1), X, m[:, None]], -1))[:, 0]
        k = D.MAXOBJ - 1
        x = torch.cat([e[:, 0, None].expand(-1, k, -1), F[:, 0, 1:], G[:, 1:], X[:, None, 2:4].expand(-1, k, -1)], -1)
        return c0, self.g(x)[..., 0], M[:, 0, 1:]

    def target_logp(self, *a):
        c0, g, mask = self.target_terms(*a)
        return Fn.logsigmoid(c0) + (Fn.logsigmoid(g) * mask).sum(1)


def coupling(mlog, midx):
    """move logits [Nm], padded move indices [n, L] (-1 = pad) -> m [n]"""
    ml = torch.cat([mlog.detach(), torch.tensor([BIG])])
    return ml[midx].min(1).values.clamp(-BIG, BIG) / 4 if midx.shape[1] else torch.full((len(midx),), BIG / 4)


def bce_logp(logp, y):
    """BCE from log p (p = noisy-AND product), numerically safe for log(1-p)"""
    logp = logp.clamp(max=-1e-6)
    return -(y * logp + (1 - y) * torch.log(-torch.expm1(logp)))


def cat_batches(fs):
    return tuple(torch.cat(t) for t in zip(*fs))


class FactorModel:
    def __init__(self):
        self.nets = []
        for s in SEEDS:
            n = Net(); n.load_state_dict(torch.load(OUT / f"s{s}.pt", weights_only=False)["state"]); self.nets.append(n.eval())

    def __call__(self, raw, seqs):
        """-> [3, len(seqs)] plan success probability; empty plans are allowed (target step only)"""
        um, ut, rows = {}, {}, []
        for s in seqs:
            st = plan_steps(s, raw["goal"])
            rows.append(([um.setdefault(x[0], (len(um), x))[0] for x in st[:-1]], ut.setdefault(st[-1][0], (len(ut), st[-1]))[0]))
        L = max(len(r[0]) for r in rows)
        midx = torch.tensor([r[0] + [-1] * (L - len(r[0])) for r in rows], dtype=torch.long).view(len(rows), L)
        tid = torch.tensor([r[1] for r in rows])
        tidx = torch.zeros(len(ut), max(L, 1), dtype=torch.long) - 1  # move indices of each unique target step
        for (mi, ti) in rows:
            tidx[ti, :len(mi)] = torch.tensor(mi, dtype=torch.long)
        tidx = tidx[:, :L]
        bm = featurize(raw, [x for _, x in um.values()]) if um else None
        bt = featurize(raw, [x for _, x in ut.values()])
        out = []
        with torch.no_grad():
            for n in self.nets:
                ml = n.move_logit(*bm) if um else torch.zeros(0)
                lp =(Fn.logsigmoid(ml)[midx.clamp(min=0)] * (midx >= 0)).sum(1) if L else torch.zeros(len(rows))
                out.append((lp + n.target_logp(*bt, coupling(ml, tidx))[tid]).exp())
        return torch.stack(out)

    def blockers(self, raw, cells=None):
        """-> {obj: seed-mean P(obj blocks the target)} in `cells` (default: the initial scene)"""
        cells = {k: tuple(v) for k, v in (cells or raw["initial_cells"]).items()}
        st = [((), cells, "t0", tuple(raw["goal"]), 0, True)]
        b = featurize(raw, st)
        with torch.no_grad():
            g = torch.stack([n.target_terms(*b, torch.zeros(1))[1][0] for n in self.nets])
        p = (1 - g.sigmoid()).mean(0)
        return {o: float(p[i]) for i, o in enumerate(sorted(o for o in cells if o != "t0"))}


# ----------------------------------------------------------------------------------------------------------------- data / training
def examples(split):
    assert split in ("train", "val")  # never the test split
    mv, tg, aux = [], [], []  # (featurize batch, y, kind, [move idx]) per root
    roots = [r for r, s in XD.qc_roots().items() if s == split]
    nm = 0
    for rid in roots:
        raw, seen = XD.load(rid), {}
        msteps, my, tsteps, ty, tmi = [], [], [], [], []
        for b in raw["labels"]["B"]:
            f, seq = b["failed_step"], b["sequence"]
            assert (f is None) == (b["y"] == 1) and len({o for o, _ in seq}) == len(seq) and "t0" not in {o for o, _ in seq}
            n_ok = len(seq) if f in (None, "target") else f
            for st in plan_steps(seq, raw["goal"])[:n_ok + 1]:
                lab = int(b["y"]) if st[5] else int(st[4] < n_ok)
                if st[0] in seen:
                    assert seen[st[0]][1] == lab; continue
                if st[5]:
                    seen[st[0]] = (None, lab); tsteps.append(st); ty.append(lab)
                    tmi.append([seen[st[0][:j + 1]][0] for j in range(st[4])])
                else:
                    seen[st[0]] = (nm, lab); nm += 1; msteps.append(st); my.append((lab, stype(st[4], False)))
        if msteps:
            mv.append((featurize(raw, msteps), my))
        if tsteps:
            tg.append((featurize(raw, tsteps), ty, tmi))
        ms = raw["labels"]["blocker"]["minimal_sets"]
        if ms and raw["verified_K"] >= 1:
            u = set().union(*map(set, ms))
            objs = sorted(o for o in raw["initial_cells"] if o != "t0")
            lab = [float(o in u) for o in objs] + [0.0] * (D.MAXOBJ - 1 - len(objs))
            aux.append((featurize(raw, [((), {}, "t0", tuple(raw["goal"]), 0, True)]), lab))
    Xm = cat_batches([b for b, _ in mv]); ym = torch.tensor([l for _, y in mv for l, _ in y], dtype=torch.float32)
    km = np.array([k for _, y in mv for _, k in y])
    Xt = cat_batches([b for b, _, _ in tg]); yt = torch.tensor([l for _, y, _ in tg for l in y], dtype=torch.float32)
    L = max(len(i) for _, _, mi in tg for i in mi)
    tmi = torch.tensor([i + [-1] * (L - len(i)) for _, _, mi in tg for i in mi], dtype=torch.long)
    Xa = cat_batches([b for b, _ in aux]); ya = torch.tensor([l for _, l in aux])
    return {"Xm": Xm, "ym": ym, "km": km, "Xt": Xt, "yt": yt, "tmi": tmi, "Xa": Xa, "ya": ya, "roots": len(roots)}


def aux_loss(net, Xa, ya, i):
    c0, g, mask = net.target_terms(*(x[i] for x in Xa), torch.zeros(len(i)))
    return (Fn.binary_cross_entropy_with_logits(-g, ya[i], reduction="none") * mask).sum() / mask.sum()


def predict(net, d):
    """-> move probs [Nm], target probs [Nt], step-level BCE (moves + targets), aux BCE"""
    with torch.no_grad():
        ml = net.move_logit(*d["Xm"])
        lt = net.target_logp(*d["Xt"], coupling(ml, d["tmi"]))
        l = (Fn.binary_cross_entropy_with_logits(ml, d["ym"], reduction="sum") + bce_logp(lt, d["yt"]).sum()) / (len(ml) + len(lt))
        a = aux_loss(net, d["Xa"], d["ya"], torch.arange(len(d["ya"])))
    return ml.sigmoid().numpy(), lt.exp().numpy(), float(l), float(a)


def train_seed(seed, tr, va, max_epochs=80, patience=10, budget_s=150):
    torch.manual_seed(seed)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    best, bad, t0, log = (1e9, None, -1), 0, time.time(), []
    Nm, Nt, Na = len(tr["ym"]), len(tr["yt"]), len(tr["ya"])
    nb = math.ceil((Nm + Nt) / 256)
    for ep in range(max_epochs):
        net.eval()
        with torch.no_grad():
            mlog = net.move_logit(*tr["Xm"])
        mt = coupling(mlog, tr["tmi"])
        net.train()
        for im, it, ia in zip(torch.randperm(Nm).chunk(nb), torch.randperm(Nt).chunk(nb), torch.randperm(Na).chunk(nb)):
            opt.zero_grad()
            lm = Fn.binary_cross_entropy_with_logits(net.move_logit(*(x[im] for x in tr["Xm"])), tr["ym"][im], reduction="sum")
            lt = bce_logp(net.target_logp(*(x[it] for x in tr["Xt"]), mt[it]), tr["yt"][it]).sum()
            ((lm + lt) / (len(im) + len(it)) + AUX_W * aux_loss(net, tr["Xa"], tr["ya"], ia)).backward(); opt.step()
        net.eval()
        _, _, v, va_aux = predict(net, va)
        log.append((round(v, 4), round(va_aux, 4)))
        if v < best[0]:
            best, bad = (v, {k: t.clone() for k, t in net.state_dict().items()}, ep), 0
        else:
            bad += 1
        if bad >= patience or time.time() - t0 > budget_s:
            break
    net.load_state_dict(best[1])
    torch.save({"state": best[1], "seed": seed, "epoch": best[2], "val_logloss": best[0], "H": H, "aux_w": AUX_W}, OUT / f"s{seed}.pt")
    return net.eval(), {"best_epoch": best[2], "epochs_run": len(log), "val_step_bce_and_aux_curve": log, "train_s": round(time.time() - t0, 1)}


def step_metrics(pm, pt, d):
    p = np.concatenate([pm, pt]); y = np.concatenate([d["ym"].numpy(), d["yt"].numpy()])
    return metrics(p, y, np.concatenate([d["km"], np.array(["target"] * len(pt))]))


def cmd_train():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tr, va = examples("train"), examples("val")
    out = {"n": {s: {"move_steps": len(d["ym"]), "target_steps": len(d["yt"]), "aux_roots": len(d["ya"]), "aux_pos": int(d["ya"].sum()),
                     "roots": d["roots"]} for s, d in (("train", tr), ("val", va))},
           "featurize_s": round(time.time() - t0, 1), "aux_w": AUX_W, "H": H, "seeds": {}}
    print(json.dumps(out), flush=True)
    pms, pts = [], []
    for s in SEEDS:
        net, info = train_seed(s, tr, va)
        pm, pt, v, a = predict(net, va)
        pms.append(pm); pts.append(pt); info["val"] = step_metrics(pm, pt, va); info["val_aux_bce"] = round(a, 4)
        out["seeds"][s] = info
        print(json.dumps({"seed": s, **{k: v for k, v in info.items() if k not in ("val_step_bce_and_aux_curve", "val")}}), flush=True)
    out["val_seed_mean"] = step_metrics(np.mean(pms, 0), np.mean(pts, 0), va)
    out["total_train_s"] = round(time.time() - t0, 1)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["val_seed_mean"], indent=1))


def calib(y, p, bins=10):
    from sklearn.metrics import roc_auc_score
    y, p = np.asarray(y, float), np.asarray(p, float)
    b = np.minimum((p * bins).astype(int), bins - 1)
    rel = [{"bin": f"{i / bins:.1f}-{(i + 1) / bins:.1f}", "n": int((b == i).sum()), "pred": round(float(p[b == i].mean()), 3),
            "obs": round(float(y[b == i].mean()), 3)} for i in range(bins) if (b == i).any()]
    ece = sum(r["n"] * abs(r["pred"] - r["obs"]) for r in rel) / len(y)
    return {"auroc": round(float(roc_auc_score(y, p)), 4), "brier": round(float(((p - y) ** 2).mean()), 4), "ece": round(ece, 4),
            "mean_p": round(float(p.mean()), 4), "reliability": rel}


def cmd_eval():
    from sklearn.metrics import roc_auc_score
    import phase3 as P3
    Fm, Hz, Mm = FactorModel(), HZ.Hazard(), P3.Models()
    ys, pf, ph, pc, skipped = [], [], [], [], 0
    bl_y, bl_p, top1, bl_y_all, bl_p_all, top1_all = [], [], [], [], [], []
    for rid, s in XD.qc_roots().items():
        if s != "val":
            continue
        raw, seen = XD.load(rid), {}
        for b in raw["labels"]["B"]:
            seq = tuple((o, tuple(c)) for o, c in b["sequence"])
            if not seq or len({o for o, _ in seq}) < len(seq):
                skipped += 1; continue
            seen.setdefault(seq, b["y"])
        if seen:
            seqs = list(seen)
            ys += list(seen.values()); pf += Fm(raw, seqs).mean(0).tolist(); ph += Hz(raw, seqs).mean(0).tolist()
            pc += Mm.p11(raw, seqs).mean(0).tolist()
        ms = raw["labels"]["blocker"]["minimal_sets"]
        if ms:
            u, pb = set().union(*map(set, ms)), Fm.blockers(raw)
            yy, pp = [float(o in u) for o in pb], list(pb.values())
            hit = max(pb, key=pb.get) in u
            bl_y_all += yy; bl_p_all += pp; top1_all.append(hit)
            if raw["verified_K"] >= 1:
                bl_y += yy; bl_p += pp; top1.append(hit)
    plan = {"n": len(ys), "pos": int(sum(ys)), "skipped_empty_or_invalid": skipped,
            "factor": calib(ys, pf), "hazard": calib(ys, ph), "boltzmann_p11": calib(ys, pc)}
    blk = {"K>=1_valid": {"roots": len(top1), "objects": len(bl_y), "pos": int(sum(bl_y)), "auroc": round(float(roc_auc_score(bl_y, bl_p)), 4),
                          "top1_in_some_minimal_set": round(float(np.mean(top1)), 4)},
           "all_nonempty_minimal_sets": {"roots": len(top1_all), "objects": len(bl_y_all), "pos": int(sum(bl_y_all)),
                                         "auroc": round(float(roc_auc_score(bl_y_all, bl_p_all)), 4),
                                         "top1_in_some_minimal_set": round(float(np.mean(top1_all)), 4)}}
    raw = XD.load(next(r for r, s in XD.qc_roots().items() if s == "val" and len(XD.load(r)["initial_cells"]) >= 4))
    plans = random_plans(raw, 5000)
    Fm(raw, plans[:10])
    t = time.time(); Fm(raw, plans); dt = time.time() - t
    out = {"plan_level_val": plan, "blockers_val": blk, "step_level_val_seed_mean": json.loads((OUT / "metrics.json").read_text())["val_seed_mean"],
           "hazard_step_level_val_seed_mean": json.loads((HZ.OUT / "metrics.json").read_text())["val_seed_mean"],
           "timing": {"plans": len(plans), "call_s": round(dt, 3), "threads": torch.get_num_threads()}}
    (OUT / "eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


def cmd_selftest():
    Fm = FactorModel()
    raw = XD.load(next(r for r, s in XD.qc_roots().items() if s == "val" and len(XD.load(r)["initial_cells"]) >= 4))
    plans = random_plans(raw, 200, seed=1)
    p = Fm(raw, plans)
    assert p.shape == (3, 200) and bool(((p > 0) & (p <= 1)).all())
    assert torch.allclose(Fm(raw, plans[7:8])[:, 0], p[:, 7], atol=1e-5)  # batching / dedupe does not change scores
    assert Fm(raw, [[]]).shape == (3, 1)  # empty plan = target step only
    # exactness: recompute plan 3 by hand from the factors
    s, n = plans[3], Fm.nets[0]
    st = plan_steps(s, raw["goal"])
    with torch.no_grad():
        ml = n.move_logit(*featurize(raw, st[:-1]))
        c0, g, mask = n.target_terms(*featurize(raw, st[-1:]), (ml.min().clamp(-BIG, BIG) / 4)[None])
        ref = ml.sigmoid().prod() * c0.sigmoid()[0] * (g.sigmoid()[0] ** mask[0]).prod()
    assert torch.allclose(ref, p[0, 3], atol=1e-5), (ref, p[0, 3])
    pb = Fm.blockers(raw)
    assert set(pb) == set(raw["initial_cells"]) - {"t0"} and all(0 < v < 1 for v in pb.values())
    # moving every object far away cannot be required: blockers() on a hypothetical scene works and differs
    cells = dict(raw["initial_cells"]); cells.update(dict(plans[0]))
    assert set(Fm.blockers(raw, cells)) == set(pb)
    print("selftest ok")


if __name__ == "__main__":
    torch.set_num_threads(1)
    {"train": cmd_train, "eval": cmd_eval, "selftest": cmd_selftest}[sys.argv[1]]()
