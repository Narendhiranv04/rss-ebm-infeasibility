"""BWL-1 (VALIDATION ONLY): matched 3D placement-sampler comparison, cVAE latent diagnostics, physics-host confirmation manifest.

    P="env CUDA_VISIBLE_DEVICES=0 taskset -c 7 nice -n 10 venv/bin/python"
    $P src/study.py compare  --config configs/sampler_bwl1.json   every arm on VAL under one raw / unique / rerank-pool budget + root bootstrap
    $P src/study.py latent   --config configs/sampler_bwl1.json   cVAE KL per dimension, latent usage / collapse, vs the categorical control
    $P src/study.py manifest --config configs/sampler_bwl1.json   predeclared physics-host-only confirmation manifest (NOTHING is executed here)
Known-label precision counts only destinations whose outcome was independently recorded (VERIFIED / NEG_CERTIFIED_GEOMETRY / NEG_UNDER_POLICY);
an untested proposal is UNKNOWN, never negative, and nothing here is a physically verified precision of new proposals.  No TEST metric."""
from __future__ import annotations

import argparse, gzip, hashlib, json, os, sys, time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import guard, train as TR  # noqa: E402

BASE = ("cvae", "fm", "cat", "uniform", "stratified", "sobol")


def ckpt(cfg, member):
    exp = cfg["archived_experiment"] if member.split("_")[0] in cfg["archived_members"] else cfg["experiment"]
    return ROOT / "runs" / exp / member / "best.pt"


class Tables:
    """Evaluation contexts of one split (those with >= 1 recorded label, as in train.cmd_evaluate) and per-cell recorded status."""

    def __init__(self, D, split, derived):
        import torch
        st = torch.as_tensor(D.meta["status"].astype(np.int64), device=D.device)
        C = int(D.ctx.max()) + 1
        cert = torch.zeros(C, 36, dtype=torch.bool, device=D.device)
        pol = cert.clone()
        cert[D.ctx[st == 1], D.dest[st == 1]] = True
        pol[D.ctx[st == 2], D.dest[st == 2]] = True
        c = D.contexts(split)
        self.c = c[(D.c_pos[c] | D.c_neg[c]).any(1)]
        self.r = D.c_row[self.c]
        self.pos, self.neg, self.cert, self.pol = D.c_pos[self.c], D.c_neg[self.c], cert[self.c], pol[self.c]
        self.legal = D.legal[self.r]
        self.src = torch.as_tensor(D.meta["src"].astype(np.int64), device=D.device)[self.r]
        self.X = (D.obj[self.r], D.mask[self.r], D.occ[self.r])
        ri = self.r.cpu().numpy()
        z = np.load(Path(derived) / "features.npz", allow_pickle=False)
        self.root, self.qid = D.meta["root"][ri], D.meta["qid"][ri]
        self.strata = {k: z[k][ri].astype(str) for k in ("K", "n_obj", "shape", "height_bin", "block")}
        cell = torch.arange(36, device=D.device)
        self.cheb = torch.maximum((cell[None] // 6 - self.src[:, None] // 6).abs(), (cell[None] % 6 - self.src[:, None] % 6).abs())


def sobol(B, R, legal, gen, seed):
    """Randomized QMC control: one scrambled Sobol set per eval seed + a per-context Cranley-Patterson shift; illegal raw cells snap to the
    nearest legal cell centre (snap rate recorded as raw_illegal)."""
    import torch
    from models import CENTRES
    u = (torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(R).to(legal.device)[None] +
         torch.rand(B, 1, 2, device=legal.device, generator=gen)) % 1.0
    rc = (u * 6).floor().clamp(0, 5).long()
    ill = ~legal.gather(1, rc[..., 0] * 6 + rc[..., 1])
    d2 = ((2 * u - 1)[:, :, None] - CENTRES.to(legal.device)[None, None]).pow(2).sum(-1).masked_fill(~legal[:, None], float("inf"))
    return d2.argmin(-1), {"raw_illegal": ill, "raw_out_of_domain": torch.zeros_like(ill)}


def raw_draws(name, mods, T, R, gen, es):
    import torch
    obj, mask, occ = T.X
    if name in ("cvae", "fm", "cat", "uniform"):
        return mods[name].sample(obj, mask, occ, T.legal, R, gen=gen)
    if name == "stratified":
        return mods[name].sample(obj, mask, occ, T.legal, R, gen=gen, src=T.src)
    if name == "sobol":
        return sobol(len(T.r), R, T.legal, gen, 7919 + es)
    lam = float(name[3:])                                                     # mix<lambda>: per raw draw, cVAE w.p. lambda else stratified
    (d1, f1), (d2, f2) = raw_draws("cvae", mods, T, R, gen, es), raw_draws("stratified", mods, T, R, gen, es)
    pick = torch.rand(d1.shape, device=d1.device, generator=gen) < lam
    return torch.where(pick, d1, d2), {k: torch.where(pick, f1[k], f2[k]) for k in f1}


def propose(arm, M, mods, T, gen, es, cfg):
    """-> props [B, 36] bool, raw draws used [B], flags, keep.  Rerank arms: first P = f*M unique from 8P raw draws, ranker top-M of the pool."""
    import torch
    if arm == "ranker_catalogue":                                             # disclosed catalogue: all legal cells scored, deterministic top-M
        lg = mods["ranker"].logits_all(*T.X).masked_fill(~T.legal, -1e9)
        props = torch.zeros_like(T.legal).scatter_(1, lg.topk(M, 1).indices, True) & T.legal
        z = torch.zeros(len(T.r), M, dtype=torch.bool, device=lg.device)
        return props, torch.full((len(T.r),), 36, device=lg.device), {"raw_illegal": z, "raw_out_of_domain": z}, ~z
    if "+rank_P" in arm:
        base, f = arm.split("+rank_P")
        P = int(f) * M
        d, fl = raw_draws(base, mods, T, cfg["raw_cap_factor"] * P, gen, es)
        pool, used, keep = TR.first_unique(d, P)
        lg = mods["ranker"].logits_all(*T.X).masked_fill(~pool, -1e9)
        props = torch.zeros_like(pool).scatter_(1, lg.topk(M, 1).indices, True) & pool
        return props, used, fl, keep
    d, fl = raw_draws(arm, mods, T, cfg["raw_cap_factor"] * M, gen, es)
    props, used, keep = TR.first_unique(d, M)
    return props, used, fl, keep


def ctx_stats(props, used, fl, keep, T):
    import torch
    um = torch.arange(keep.shape[1], device=props.device)[None] < used[:, None].clamp(max=keep.shape[1])
    p, ch = props.float(), T.cheb.float()
    n = p.sum(1)
    ce = torch.stack([torch.arange(36, device=p.device) // 6, torch.arange(36, device=p.device) % 6], -1).float()
    pair = torch.einsum("bi,ij,bj->b", p, torch.cdist(ce, ce), p) / (n * (n - 1)).clamp(min=1)
    s = {"hit": (props & T.pos).any(1).float(), "n": n, "npos": (props & T.pos).sum(1).float(), "ncert": (props & T.cert).sum(1).float(),
         "npol": (props & T.pol).sum(1).float(), "nunk": (props & ~(T.pos | T.neg)).sum(1).float(), "raw": used.float(),
         "rawill": (fl["raw_illegal"] & um).sum(1).float(), "rawood": (fl["raw_out_of_domain"] & um).sum(1).float(),
         "near": (props & (T.cheb == 1)).sum(1).float(), "mid": (props & (T.cheb >= 2) & (T.cheb <= 3)).sum(1).float(),
         "far": (props & (T.cheb >= 4)).sum(1).float(), "disp": (p * ch).sum(1) / n.clamp(min=1), "pair": pair}
    return {k: v.cpu().numpy() for k, v in s.items()}


def agg(s, T):
    hp = T.pos.any(1).cpu().numpy()
    tot, known = s["n"].sum(), (s["npos"] + s["ncert"] + s["npol"]).sum()
    return {"hit": float(s["hit"][hp].mean()), "known_label_precision": float(s["npos"].sum() / max(known, 1e-9)), "known_label_fraction": float(known / tot),
            "unknown_fraction": float(s["nunk"].sum() / tot), "certified_negative_fraction": float(s["ncert"].sum() / tot),
            "policy_negative_fraction": float(s["npol"].sum() / tot), "positive_recall": float(s["npos"].sum() / T.pos.sum().item()),
            "unique_per_context": float(s["n"].mean()), "raw_draws_per_context": float(s["raw"].mean()), "duplicate_rate": float(1 - tot / s["raw"].sum()),
            "raw_snap_or_illegal_rate": float(s["rawill"].sum() / s["raw"].sum()), "raw_out_of_domain_rate": float(s["rawood"].sum() / s["raw"].sum()),
            "band_share": {b: float(s[b].sum() / tot) for b in ("near", "mid", "far")}, "mean_cheb_displacement": float((s["disp"] * s["n"]).sum() / tot),
            "mean_pairwise_cell_distance": float(s["pair"].mean()), "proposals_total": int(tot), "contexts": int(len(s["n"])), "contexts_with_positive": int(hp.sum())}


def boot_weights(roots, B, seed):
    u, inv = np.unique(roots, return_inverse=True)
    rng = np.random.default_rng(seed)
    return inv, rng.multinomial(len(u), np.full(len(u), 1 / len(u)), size=B).astype(float), len(u)


def root_sums(s, T, inv, nr):
    hp = T.pos.any(1).cpu().numpy()
    f = lambda v: np.bincount(inv, weights=v, minlength=nr)  # noqa: E731
    return {"hit": (f(s["hit"] * hp), f(hp.astype(float))), "prec": (f(s["npos"]), f(s["npos"] + s["ncert"] + s["npol"]))}


def ci(Wt, num, den):
    v = (Wt @ num) / np.maximum(Wt @ den, 1e-12)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], v


def cmd_compare(cfg, a):
    import torch
    from data3d import Data
    from models import build
    dev = setup(cfg, a)
    out_dir = ROOT / "runs" / cfg["experiment"] / "study_val"
    if (out_dir / "study_val.json").exists():
        raise SystemExit(f"REFUSED: {out_dir}/study_val.json exists")
    wd = TR.start_watchdog(out_dir, a.guard_config)
    try:
        t_start = time.time()
        D = Data(ROOT / cfg["derived"], dev)
        T = Tables(D, cfg["split"], ROOT / cfg["derived"])
        inv, Wt, nr = boot_weights(T.root, cfg["bootstrap"]["B"], cfg["bootstrap"]["seed"])
        seeds = range(len(cfg["train_seeds"]))
        load = lambda m: TR.load_model(ckpt(cfg, m), dev)[0]  # noqa: E731
        members = {k: [load(m) for m in v] for k, v in {**cfg["archived_members"], **cfg["new_members"]}.items()}
        ctrl = {k: build(k, cfg) for k in ("uniform", "stratified")}
        P = cfg["rerank_pool_factors"]
        arms = (["cvae", "fm", "cat", "uniform", "stratified", "sobol"] + [f"mix{l}" for l in cfg["mixture_lambdas"]] +
                [f"{b}+rank_P{f}" for b in ("cvae", "stratified", "cat") for f in P] + ["ranker_catalogue"])
        trained = lambda arm: any(x in arm for x in ("cvae", "fm", "cat", "mix", "rank"))  # noqa: E731
        res = {"split": cfg["split"], "contexts": int(len(T.r)), "contexts_with_positive": int(T.pos.any(1).sum()), "roots": nr,
               "config_sha256": TR.sha(a.config), "checkpoints": {m: TR.sha(ckpt(cfg, m)) for v in {**cfg["archived_members"], **cfg["new_members"]}.values() for m in v},
               "arms": {}, "label_scope": "known-label metrics over recorded VAL outcomes only; UNKNOWN proposals are never negatives"}
        percx = {}
        for arm in arms:
            runs, mean_s = [], {str(M): None for M in cfg["M"]}
            torch.cuda.reset_peak_memory_stats()
            lat = {str(M): [] for M in cfg["M"]}
            for s in (seeds if trained(arm) else [None]):
                mods = {**ctrl, **({k: v[s] for k, v in members.items()} if s is not None else {})}
                for es in cfg["eval_seeds"]:
                    gen = torch.Generator(device=dev).manual_seed(1000 + es)
                    row = {"train_seed": s, "eval_seed": es, "by_M": {}}
                    for M in cfg["M"]:
                        torch.cuda.synchronize(); t0 = time.time()
                        with torch.no_grad():
                            pr = propose(arm, M, mods, T, gen, es, cfg)
                        torch.cuda.synchronize()
                        lat[str(M)].append(1000 * (time.time() - t0) / len(T.r))
                        st = ctx_stats(*pr, T)
                        row["by_M"][str(M)] = agg(st, T)
                        mean_s[str(M)] = st if mean_s[str(M)] is None else {k: mean_s[str(M)][k] + st[k] for k in st}
                    runs.append(row)
                    if arm == "ranker_catalogue":
                        break                                                 # deterministic: one eval per trained ranker
            n_runs = len(runs)
            mean_s = {M: {k: v / n_runs for k, v in st.items()} for M, st in mean_s.items()}
            percx[arm] = mean_s
            pooled, cis = {}, {}
            for M, st in mean_s.items():
                pooled[M] = agg(st, T)
                rs = root_sums(st, T, inv, nr)
                cis[M] = {k: ci(Wt, *rs[k])[0] for k in rs}
            sd = {M: {k: float(np.std([r["by_M"][M][k] for r in runs])) for k in ("hit", "known_label_precision", "known_label_fraction")} for M in pooled}
            res["arms"][arm] = {"runs": n_runs, "pooled": pooled, "root_ci95": cis, "run_std": sd, "per_run": runs,
                                "latency_ms_per_context": {M: float(np.mean(v)) for M, v in lat.items()},
                                "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)}
        # paired root-cluster bootstrap deltas (same resamples for both arms)
        pairs = [("cvae+rank_P2", "stratified+rank_P2"), ("cvae", "stratified"), ("cvae", "cat"), ("cvae+rank_P2", "cvae"),
                 ("stratified+rank_P2", "stratified"), ("cvae+rank_P4", "stratified+rank_P4"), ("cat+rank_P2", "stratified+rank_P2"),
                 ("cvae", "fm"), ("cvae", "uniform"), ("stratified", "sobol")] + [(f"mix{l}", "stratified") for l in cfg["mixture_lambdas"]]
        res["paired_deltas"] = {}
        for x, y in pairs:
            res["paired_deltas"][f"{x} - {y}"] = {}
            for M in map(str, cfg["M"]):
                rx, ry = root_sums(percx[x][M], T, inv, nr), root_sums(percx[y][M], T, inv, nr)
                d = {}
                for k in ("hit", "prec"):
                    vx, vy = ci(Wt, *rx[k])[1], ci(Wt, *ry[k])[1]
                    pt = rx[k][0].sum() / rx[k][1].sum() - ry[k][0].sum() / ry[k][1].sum()
                    d[k] = {"point": float(pt), "ci95": [float(np.percentile(vx - vy, 2.5)), float(np.percentile(vx - vy, 97.5))]}
                res["paired_deltas"][f"{x} - {y}"][M] = d
        res["decision"] = decide(res, cfg)
        res["strata_M4"] = {arm: strata(percx[arm]["4"], T) for arm in ("cvae", "cat", "stratified", "cvae+rank_P2", "stratified+rank_P2", "ranker_catalogue", "uniform")}
        res["diversity_256"] = diversity(members, ctrl, T, dev, cfg)
        res["ranker_calibration_val"] = calibration(members["ranker"], D, cfg)
        res["examples"] = examples(members, ctrl, T, dev, cfg, out_dir)
        res["wall_s"] = round(time.time() - t_start, 1)
        res["device"] = torch.cuda.get_device_name(0)
        res["affinity"] = sorted(os.sched_getaffinity(0))
        res["watchdog_stop"] = (out_dir / "STOP_RESOURCE").exists()
        (out_dir / "study_val.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({"wrote": str(out_dir / "study_val.json"), "decision": res["decision"], "wall_s": res["wall_s"]}))
    finally:
        wd.terminate()


def decide(res, cfg):
    rule = cfg["predeclared_decision"]
    d = res["paired_deltas"][f"{rule['learned_arm']} - {rule['analytical_arm']}"][str(rule["primary_M"])]
    h, p = d["hit"], d["prec"]
    sig_pos = lambda x: x["ci95"][0] > 0  # noqa: E731
    sig_neg = lambda x: x["ci95"][1] < 0  # noqa: E731
    if sig_pos(p) and h["ci95"][0] > -0.02:
        v = "PROMISING"
    elif (sig_neg(h) and not sig_pos(p)) or (sig_neg(p) and not sig_pos(h)):
        v = "ANALYTICAL_AHEAD"
    elif not any(map(sig_pos, (h, p))) and not any(map(sig_neg, (h, p))) and abs(h["point"]) < 0.02 and abs(p["point"]) < 0.02:
        v = "TIED"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "rule": rule, "hit_delta": h, "precision_delta": p}


def strata(st, T):
    hp = T.pos.any(1).cpu().numpy()
    out = {}
    for k, vals in T.strata.items():
        out[k] = {}
        for v in sorted(set(vals)):
            m = vals == v
            kn = (st["npos"] + st["ncert"] + st["npol"])[m].sum()
            out[k][v] = {"contexts": int(m.sum()), "hit": float(st["hit"][m & hp].mean()) if (m & hp).any() else None,
                         "known_label_precision": float(st["npos"][m].sum() / kn) if kn else None, "known_label_fraction": float(kn / max(st["n"][m].sum(), 1e-9))}
    return out


def diversity(members, ctrl, T, dev, cfg, R=256):
    """Distribution diversity of each stochastic base sampler (train seed 0, eval seed 0): per-context entropy of 256 raw draws."""
    import torch
    out = {}
    for arm in ("cvae", "fm", "cat", "uniform", "stratified", "sobol", "mix0.5"):
        mods = {**ctrl, **{k: v[0] for k, v in members.items()}}
        gen = torch.Generator(device=dev).manual_seed(4242)
        with torch.no_grad():
            d, _ = raw_draws(arm, mods, T, R, gen, 0)
        f = torch.nn.functional.one_hot(d, 36).float().mean(1)
        H = -(f * f.clamp(min=1e-12).log()).sum(1)
        out[arm] = {"entropy_nats": float(H.mean()), "effective_cells": float(H.exp().mean()), "distinct_cells_in_256": float((f > 0).sum(1).float().mean()),
                    "legal_cells": float(T.legal.sum(1).float().mean())}
    return out


def calibration(rankers, D, cfg):
    import torch
    out = []
    lab = torch.nonzero((D.split == 1) & (D.label >= 0)).squeeze(1)
    st = torch.as_tensor(D.meta["status"].astype(np.int64), device=D.device)[lab]
    for s, m in enumerate(rankers):
        with torch.no_grad():
            pl = torch.cat([m.logits_all(D.obj[b], D.mask[b], D.occ[b]).gather(1, D.dest[b][:, None]).squeeze(1) for b in lab.split(4096)])
        pp, yy = torch.sigmoid(pl).cpu().numpy(), D.label[lab].float().cpu().numpy()
        row = {"train_seed": s}
        for name, sel in (("all_decided", np.ones(len(yy), bool)), ("pos_vs_certified_neg", (st != 2).cpu().numpy()), ("pos_vs_policy_neg", (st != 1).cpu().numpy())):
            row[name] = scores(pp[sel], yy[sel])
        out.append(row)
    return out


def scores(p, y):
    o = np.argsort(-p)
    ys = y[o]
    tp = np.cumsum(ys)
    ap = float((tp / np.arange(1, len(ys) + 1))[ys == 1].mean())
    rk = np.empty(len(p)); rk[np.argsort(p)] = np.arange(1, len(p) + 1)
    n1, n0 = ys.sum(), len(ys) - ys.sum()
    auc = float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / max(n1 * n0, 1))
    b = np.minimum((p * 10).astype(int), 9)
    ece = float(sum(abs(p[b == k].mean() - y[b == k].mean()) * (b == k).mean() for k in range(10) if (b == k).any()))
    return {"n": int(len(y)), "base_rate": float(y.mean()), "auroc": auc, "auprc": ap, "brier": float(((p - y) ** 2).mean()), "ece_10bin": ece}


def examples(members, ctrl, T, dev, cfg, out_dir, n=6, R=256):
    """Proposal frequency maps (train seed 0, eval seed 0) on n fixed VAL scenes for figures: raw-draw frequencies for samplers, M=4 sets for
    rerank / catalogue arms; recorded status per cell.  Scenes = first n by sha256(qid) among contexts with >= 1 positive and >= 1 negative."""
    import torch
    ok = (T.pos.any(1) & T.neg.any(1)).cpu().numpy()
    idx = sorted(np.nonzero(ok)[0], key=lambda i: hashlib.sha256(str(T.qid[i]).encode()).hexdigest())[:n]
    sub = Tables.__new__(Tables)
    I = torch.as_tensor(idx, device=dev)
    for k in ("r", "pos", "neg", "cert", "pol", "legal", "src", "cheb"):
        setattr(sub, k, getattr(T, k)[I])
    sub.X = tuple(x[I] for x in T.X)
    mods = {**ctrl, **{k: v[0] for k, v in members.items()}}
    freq = {}
    for arm in ("cvae", "fm", "cat", "uniform", "stratified", "sobol", "mix0.5"):
        gen = torch.Generator(device=dev).manual_seed(77)
        with torch.no_grad():
            d, _ = raw_draws(arm, mods, sub, R, gen, 0)
        freq[arm] = torch.nn.functional.one_hot(d, 36).float().mean(1).cpu().numpy()
    for arm in ("cvae+rank_P2", "stratified+rank_P2", "ranker_catalogue"):
        gen = torch.Generator(device=dev).manual_seed(77)
        with torch.no_grad():
            freq[arm] = propose(arm, 4, mods, sub, gen, 0, cfg)[0].float().cpu().numpy()
    occ = sub.X[2].cpu().numpy()
    np.savez_compressed(out_dir / "examples_val.npz", arms=np.array(list(freq)), freq=np.stack([freq[a] for a in freq], 1), occ=occ,
                        pos=sub.pos.cpu().numpy(), cert=sub.cert.cpu().numpy(), pol=sub.pol.cpu().numpy(), src=sub.src.cpu().numpy(),
                        qid=np.array([str(T.qid[i]) for i in idx]))
    return {"file": "examples_val.npz", "qids": [str(T.qid[i]) for i in idx]}


def cmd_latent(cfg, a):
    """cVAE latent usage on VAL: KL per dimension over VAL positives, active units, I(z; d | S) on VAL contexts, z-sensitivity of the argmax,
    and held-out positive NLL of cVAE (IWAE k=64) vs the categorical control (exact) vs legality-uniform."""
    import torch
    from data3d import Data
    dev = setup(cfg, a)
    out_dir = ROOT / "runs" / cfg["experiment"] / "latent_val"
    if (out_dir / "latent_val.json").exists():
        raise SystemExit("REFUSED: latent_val.json exists")
    wd = TR.start_watchdog(out_dir, a.guard_config)
    try:
        D = Data(ROOT / cfg["derived"], dev)
        T = Tables(D, "val", ROOT / cfg["derived"])
        pr = torch.nonzero((D.split == 1) & (D.label == 1)).squeeze(1)
        res = {"val_positive_rows": int(len(pr)), "val_contexts": int(len(T.r)), "cvae": [], "cat": []}
        torch.manual_seed(0)
        for s, m in enumerate(cfg["archived_members"]["cvae"]):
            net = TR.load_model(ckpt(cfg, m), dev)[0]
            with torch.no_grad():
                c = net.enc(D.obj[pr], D.mask[pr], D.occ[pr])
                mu, lv = net.post(torch.cat([c, torch.nn.functional.one_hot(D.dest[pr], 36).float()], -1)).chunk(2, -1)
                kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv)                   # [N, z]
                cc = net.enc(*T.X)
                B, K = len(T.r), 64
                z = torch.randn(B, K, net.zdim, device=dev)
                lp = net.dec(torch.cat([cc[:, None].expand(B, K, -1), z], -1)).masked_fill(~T.legal[:, None], -1e9).log_softmax(-1)
                pz, pbar = lp.exp(), lp.exp().mean(1)
                mi = (pz * (lp - pbar.clamp(min=1e-12).log()[:, None])).sum(-1).mean(1)
                p0 = net.dec(torch.cat([cc, torch.zeros(B, net.zdim, device=dev)], -1)).masked_fill(~T.legal, -1e9).softmax(-1)
                am = lp.argmax(-1)
                nll = -torch.cat([net.log_prob(D.obj[b], D.mask[b], D.occ[b], D.legal[b], D.dest[b]) for b in pr.split(4096)]).mean()
            kld = kl.mean(0).cpu().numpy()
            res["cvae"].append({"member": m, "kl_per_dim_nats": [float(x) for x in kld], "aggregate_kl_nats": float(kld.sum()),
                                "active_units_kl_gt_0.01": int((kld > 0.01).sum()), "zdim": net.zdim,
                                "mutual_information_z_d_given_S_nats": float(mi.mean()), "mi_p90": float(mi.quantile(0.9)),
                                "tv_p_z0_vs_marginal": float(0.5 * (p0 - pbar).abs().sum(1).mean()),
                                "fraction_contexts_argmax_changes_with_z": float((am != am[:, :1]).any(1).float().mean()),
                                "heldout_positive_nll_iwae64": float(nll)})
        for m in cfg["new_members"]["cat"]:
            net = TR.load_model(ckpt(cfg, m), dev)[0]
            with torch.no_grad():
                nll = -torch.cat([net.log_prob(D.obj[b], D.mask[b], D.occ[b], D.legal[b], D.dest[b]) for b in pr.split(4096)]).mean()
            res["cat"].append({"member": m, "heldout_positive_nll_exact": float(nll)})
        res["uniform_heldout_positive_nll"] = float(D.legal[pr].float().sum(1).log().mean())
        res["collapse_rule"] = "posterior collapse if active units == 0 and I(z;d|S) < 0.01 nats; partial if I < 0.05 nats"
        mis = [r["mutual_information_z_d_given_S_nats"] for r in res["cvae"]]
        res["verdict"] = ("POSTERIOR_COLLAPSE" if max(mis) < 0.01 and all(r["active_units_kl_gt_0.01"] == 0 for r in res["cvae"]) else
                          "NEAR_COLLAPSE_LATENT_WEAKLY_USED" if max(mis) < 0.05 else "LATENT_USED")
        res["device"], res["watchdog_stop"] = torch.cuda.get_device_name(0), (out_dir / "STOP_RESOURCE").exists()
        (out_dir / "latent_val.json").write_text(json.dumps(res, indent=1))
        print(json.dumps({"verdict": res["verdict"], "mi": mis, "nll": [r["heldout_positive_nll_iwae64"] for r in res["cvae"]],
                          "cat": [r["heldout_positive_nll_exact"] for r in res["cat"]]}))
    finally:
        wd.terminate()


def cmd_manifest(cfg, a):
    """Predeclared physics-host-only confirmation manifest (NOT executed on this server).  Contexts: TEST-split contexts (never trained on),
    selected by sha256(qid) order within n_obj strata BEFORE any proposal is generated; every arm's M=4 proposals (train seed 0, eval seed 0)
    are unioned per context with provenance.  The teacher must evaluate EVERY tuple, including ones with an archived label (reproducibility),
    so that precision of new proposals is measured without the known-label selection bias.  No metric is computed here."""
    import torch
    from data3d import Data
    from models import build
    dev = setup(cfg, a)
    out = ROOT / "runs" / cfg["experiment"] / "confirmation_manifest"
    if (out / "PHYSICS_CONFIRMATION_MANIFEST.json").exists():
        raise SystemExit("REFUSED: manifest exists")
    wd = TR.start_watchdog(out, a.guard_config)
    try:
        D = Data(ROOT / cfg["derived"], dev)
        T = Tables(D, "test", ROOT / cfg["derived"])
        per = 20
        chosen = []
        for v in sorted(set(T.strata["n_obj"])):
            ii = [i for i in range(len(T.r)) if T.strata["n_obj"][i] == v]
            chosen += sorted(ii, key=lambda i: hashlib.sha256(("BWL1_CONFIRM|" + str(T.qid[i])).encode()).hexdigest())[:per]
        I = torch.as_tensor(sorted(chosen), device=dev)
        sub = Tables.__new__(Tables)
        for k in ("r", "pos", "neg", "cert", "pol", "legal", "src", "cheb"):
            setattr(sub, k, getattr(T, k)[I])
        sub.X = tuple(x[I] for x in T.X)
        mods = {**{k: build(k, cfg) for k in ("uniform", "stratified")},
                **{k: TR.load_model(ckpt(cfg, v[0]), dev)[0] for k, v in {**cfg["archived_members"], **cfg["new_members"]}.items()}}
        arms = ["cvae", "cvae+rank_P2", "cat", "fm", "uniform", "stratified", "stratified+rank_P2", "sobol", "mix0.5"]
        props = {}
        for arm in arms:
            gen = torch.Generator(device=dev).manual_seed(2026)
            with torch.no_grad():
                props[arm] = propose(arm, 4, mods, sub, gen, 0, cfg)[0].cpu().numpy()
        roster = {}
        rp = ROOT / "snapshots/v4full_82926152fa9cc345/upstream/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/production_v4/frozen/q2_parity_roster_v4.jsonl.gz"
        with gzip.open(rp, "rt") as f:
            for line in f:
                q = json.loads(line)
                roster[q["qid"]] = q
        rows, ii = [], I.cpu().numpy()
        P, Ce, Po = sub.pos.cpu().numpy(), sub.cert.cpu().numpy(), sub.pol.cpu().numpy()
        for j, i in enumerate(ii):
            q = roster[str(T.qid[i])]
            for cell in range(36):
                src = [arm for arm in arms if props[arm][j, cell]]
                if src:
                    rows.append({"tuple_id": f"BWL1C-{len(rows):05d}", "context_ref_qid": str(T.qid[i]), "root_id": str(T.root[i]),
                                 "scene_hash": q["scene_hash"], "hypothetical_cells": q["cells"], "edits_from_root": q["edits"], "object_id": q["object"],
                                 "destination_cell": [cell // 6, cell % 6], "proposed_by": src,
                                 "archived_status": "VERIFIED" if P[j, cell] else "NEG_CERTIFIED_GEOMETRY" if Ce[j, cell] else "NEG_UNDER_POLICY" if Po[j, cell] else "UNTESTED",
                                 "stratum_n_obj": str(T.strata["n_obj"][i])})
        man = {"manifest": "BWL1_PHYSICS_CONFIRMATION_V1", "status": "PREDECLARED_NOT_EXECUTED", "host": "authorized physics host only (never this GPU server)",
               "teacher": "the pinned v4 placement teacher of the snapshot run manifest (Q2_3D_PARITY_V1 policy adaptive_v1, teacher protocol 89cbabed5d7e4847); every tuple evaluated",
               "selection": "TEST-split contexts, 20 per n_obj stratum by sha256('BWL1_CONFIRM|'+qid), chosen before proposals; arms' M=4 proposals unioned",
               "arms": arms, "members": {k: v[0] for k, v in {**cfg["archived_members"], **cfg["new_members"]}.items()}, "eval_seed": 0,
               "contexts": int(len(ii)), "tuples": len(rows), "tuples_untested": sum(r["archived_status"] == "UNTESTED" for r in rows),
               "analysis_plan": "per arm: physically verified precision over ALL its tuples (UNKNOWN eliminated), root-cluster bootstrap CIs, arms compared on the same contexts; "
                                "archived-label tuples double as teacher reproducibility checks; no arm / budget may be changed after outcomes are seen",
               "rows": rows}
        man["sha256_of_rows"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        (out / "PHYSICS_CONFIRMATION_MANIFEST.json").write_text(json.dumps(man, indent=1))
        print(json.dumps({k: v for k, v in man.items() if k != "rows"}))
    finally:
        wd.terminate()


def setup(cfg, a):
    return TR.preflight(cfg, guard.load_cfg(a.guard_config), False)[1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["compare", "latent", "manifest"])
    ap.add_argument("--config", required=True); ap.add_argument("--guard-config")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    {"compare": cmd_compare, "latent": cmd_latent, "manifest": cmd_manifest}[a.cmd](cfg, a)
