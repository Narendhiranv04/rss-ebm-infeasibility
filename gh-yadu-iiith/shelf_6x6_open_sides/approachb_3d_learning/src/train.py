"""GPU-only training / evaluation of the 3D placement samplers (cVAE, FM), the controls and the ranker, under the resource guard.

    taskset -c 7 nice -n 10 venv/bin/python src/train.py smoke    --config configs/sampler_v1.json --kind cvae|fm|ranker
    taskset -c 7 nice -n 10 venv/bin/python src/train.py fit      --config configs/sampler_v1.json --kind cvae --seed 0 [--geometry-only]
    taskset -c 7 nice -n 10 venv/bin/python src/train.py evaluate --config configs/sampler_v1.json --split val
    taskset -c 7 nice -n 10 venv/bin/python src/train.py freeze   --config configs/sampler_v1.json        (selection frozen BEFORE any test run)
    taskset -c 7 nice -n 10 venv/bin/python src/train.py evaluate --config configs/sampler_v1.json --split test
    any command with --dry-run: guard probe + data/config check only; CUDA is hidden before torch is imported (no GPU allocation, no step).

Every GPU command: guard.probe must pass first; an independent watchdog process (guard.py watch) runs for the whole command and stops only this
process on a violation (STOP_RESOURCE); 1 CPU (affinity must be the configured single CPU), 1 torch / BLAS thread, no DataLoader workers,
whole tensors on the GPU, a per-process VRAM fraction, a hard walltime per fit."""
from __future__ import annotations

import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
if "--dry-run" in sys.argv:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""                                  # dry run can never allocate a GPU

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import guard  # noqa: E402

STOP = {"flag": False}


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def preflight(cfg, gcfg, dry):
    aff = sorted(os.sched_getaffinity(0))
    p = guard.probe(gcfg)
    problems = list(p["violations"])
    if aff != [gcfg["cpu"]]:
        problems.append(f"CPU affinity {aff} != [{gcfg['cpu']}] (launch with taskset -c {gcfg['cpu']})")
    out = {"utc": utc(), "affinity": aff, "guard": p, "problems": problems}
    if problems:
        print(json.dumps(out, indent=1)); raise SystemExit("REFUSED: preflight failed (" + "; ".join(problems) + ")")
    if dry:
        return out, None
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if not torch.cuda.is_available():
        raise SystemExit("REFUSED: CUDA not available (no CPU-training fallback)")
    tot = torch.cuda.get_device_properties(0).total_memory / 2**20
    torch.cuda.set_per_process_memory_fraction(min(1.0, gcfg["gpu_vram_budget_mb"] / tot), 0)
    return out, torch.device("cuda:0")


def start_watchdog(run, gcfg_path):
    run.mkdir(parents=True, exist_ok=True)
    w = subprocess.Popen([sys.executable, str(HERE / "guard.py"), "watch", "--pid", str(os.getpid()), "--run", str(run)] +
                         (["--config", str(gcfg_path)] if gcfg_path else []), stdout=open(run / "watchdog.out", "a"), stderr=subprocess.STDOUT)
    signal.signal(signal.SIGTERM, lambda *a: STOP.__setitem__("flag", True))
    return w


def stopped(run):
    return STOP["flag"] or (run / "STOP_RESOURCE").exists()


def load(cfg, device, geometry_only=False):
    from data3d import Data
    return Data(ROOT / cfg["derived"], device, geometry_only)


def batches(idx, bs, gen):
    import torch
    perm = idx[torch.randperm(len(idx), generator=gen, device=idx.device)]
    return [perm[i:i + bs] for i in range(0, len(perm), bs)]


def fwd(model, D, rows):
    if model.kind == "ranker":
        return model.loss(D.obj[rows], D.mask[rows], D.occ[rows], D.legal[rows], D.dest[rows], D.label[rows])
    return model.loss(D.obj[rows], D.mask[rows], D.occ[rows], D.legal[rows], D.dest[rows], fwd.beta)


fwd.beta = 1.0


def train_rows(D, kind, split):
    import torch
    if kind == "ranker":                                                      # labelled trials only (UNLABELED never a negative)
        return torch.nonzero((D.split == ["train", "val", "test"].index(split)) & (D.label >= 0)).squeeze(1)
    return D.rows(split, positive_only=True)                                  # generators: VERIFIED destinations only


def run_dir(cfg, kind, seed, geo):
    return ROOT / "runs" / cfg["experiment"] / f"{kind}{'_geo' if geo else ''}_s{seed}"


def cmd_smoke(cfg, gcfg, a):
    import torch
    from models import build
    pre, dev = preflight(cfg, gcfg, a.dry_run)
    if a.dry_run:
        print(json.dumps({"dry_run": True, "preflight": pre["problems"], "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")})); return
    run = ROOT / "runs" / cfg["experiment"] / f"smoke_{a.kind}"
    wd = start_watchdog(run, a.guard_config)
    try:
        torch.manual_seed(0)
        D = load(cfg, dev)
        m = build(a.kind, cfg).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"])
        rows = train_rows(D, a.kind, "train")[:cfg["batch"]]
        before = [p.detach().clone() for p in m.parameters()]
        log = []
        for i in range(3):
            l, parts = fwd(m, D, rows)
            opt.zero_grad(); l.backward()
            gfin = all(torch.isfinite(p.grad).all().item() for p in m.parameters() if p.grad is not None)
            gnorm = float(torch.sqrt(sum((p.grad ** 2).sum() for p in m.parameters() if p.grad is not None)))
            opt.step(); log.append({"step": i, "loss": l.item(), **parts, "grad_finite": gfin, "grad_norm": gnorm})
        moved = sum(int(not torch.equal(b, p.detach())) for b, p in zip(before, m.parameters()))
        res = {"kind": a.kind, "device": torch.cuda.get_device_name(0), "steps": log, "params_updated": moved, "params": len(before),
               "split_of_rows": sorted(set(D.split[rows].tolist())), "labels_of_rows": sorted(set(D.label[rows].tolist())),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "affinity": sorted(os.sched_getaffinity(0)),
               "torch_threads": torch.get_num_threads(), "watchdog_stop": (run / "STOP_RESOURCE").exists()}
        res["pass"] = all(s["grad_finite"] and s["loss"] == s["loss"] for s in log) and moved > 0 and res["split_of_rows"] == [0]
        (run / "smoke.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))
    finally:
        wd.terminate()


def cmd_fit(cfg, gcfg, a):
    import torch
    from models import build
    pre, dev = preflight(cfg, gcfg, a.dry_run)
    run = run_dir(cfg, a.kind, a.seed, a.geometry_only)
    if a.dry_run:
        print(json.dumps({"dry_run": True, "would_write": str(run), "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES")})); return
    if (run / "best.pt").exists():
        raise SystemExit(f"REFUSED: {run}/best.pt exists (runs are never overwritten)")
    wd = start_watchdog(run, a.guard_config)
    t0, status = time.time(), "COMPLETED"
    try:
        torch.manual_seed(a.seed)
        gen = torch.Generator(device=dev).manual_seed(a.seed)
        D = load(cfg, dev, a.geometry_only)
        m = build(a.kind, cfg).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
        tr, va = train_rows(D, a.kind, "train"), train_rows(D, a.kind, "val")
        curve, best, bad = [], float("inf"), 0
        for ep in range(cfg["max_epochs"]):
            fwd.beta = cfg["beta_kl_max"] * min(1.0, (ep + 1) / cfg["beta_warmup_epochs"])
            m.train(); tl = []
            for b in batches(tr, cfg["batch"], gen):
                l, _ = fwd(m, D, b)
                opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step(); tl.append(l.item())
                if stopped(run):
                    break
            m.eval()
            with torch.no_grad():
                torch.manual_seed(10_000 + ep)                                # fixed validation noise per epoch
                fwd.beta = cfg["beta_kl_max"]                                 # validation always at the final beta (comparable across epochs)
                vl = [fwd(m, D, b)[0].item() * len(b) for b in va.split(4096)]
                vloss = sum(vl) / len(va)
            curve.append({"epoch": ep, "train_loss": sum(tl) / max(1, len(tl)), "val_loss": vloss, "beta": fwd.beta,
                          "wall_s": round(time.time() - t0, 1), "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)})
            (run / "curve.json").write_text(json.dumps(curve, indent=1))
            if vloss < best - 1e-4:
                best, bad = vloss, 0
                torch.save({"state": m.state_dict(), "epoch": ep, "val_loss": vloss, "cfg": cfg, "kind": a.kind, "seed": a.seed,
                            "geometry_only": a.geometry_only}, run / "best.pt")
            else:
                bad += 1
            if stopped(run):
                status = "STOPPED_FOR_RESOURCE_SAFETY"; break
            if bad >= cfg["patience"]:
                status = "EARLY_STOPPED"; break
            if time.time() - t0 > 60 * cfg["walltime_min_per_run"]:
                status = "WALLTIME_LIMIT"; break
        res = {"kind": a.kind, "seed": a.seed, "geometry_only": a.geometry_only, "status": status, "epochs": len(curve), "best_val_loss": best,
               "train_rows": int(len(tr)), "val_rows": int(len(va)), "wall_s": round(time.time() - t0, 1),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1), "best_sha256": sha(run / "best.pt") if (run / "best.pt").exists() else None,
               "device": torch.cuda.get_device_name(0), "affinity": sorted(os.sched_getaffinity(0)), "finished_utc": utc()}
        (run / "result.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res))
    finally:
        wd.terminate()


# ------------------------------------------------------------------------------------------------------------------------------ evaluation
def first_unique(d, M):
    """d [B, R] raw proposals -> (props bool [B, 36] holding the first M unique cells, raw_used [B])."""
    import torch
    B, R = d.shape
    oh = torch.nn.functional.one_hot(d, 36)                                   # [B, R, 36]
    seen_before = (oh.cumsum(1) - oh).gather(2, d[..., None]).squeeze(-1) > 0
    first = ~seen_before
    rank = first.long().cumsum(1)
    keep = first & (rank <= M)
    props = (oh * keep[..., None]).sum(1) > 0
    got = keep.sum(1)
    pos_m = torch.where(rank >= M, torch.arange(R, device=d.device)[None].expand(B, R), torch.full_like(d, R)).min(1).values
    raw_used = torch.where(got >= M, pos_m + 1, torch.full_like(got, R))
    return props, raw_used, keep


def metrics(props, raw_used, keep, flags, pos, neg, legal, M):
    import torch
    P = props
    haspos = pos.any(1)
    used = torch.arange(keep.shape[1], device=P.device)[None] < raw_used[:, None]
    out = {"unique_per_context": P.sum(1).float().mean().item(), "unique_legal_rate": (P & legal).sum().item() / max(1, P.sum().item()),
           "duplicate_rate": 1 - P.sum().item() / max(1, raw_used.sum().item()),
           "raw_illegal_rate": (flags["raw_illegal"] & used).sum().item() / max(1, used.sum().item()),
           "raw_out_of_domain_rate": (flags["raw_out_of_domain"] & used).sum().item() / max(1, used.sum().item()),
           "hit_at_M": (P & pos).any(1)[haspos].float().mean().item(),
           "pos_recall": (P & pos).sum().item() / max(1, pos.sum().item()),
           "known_label_precision": (P & pos).sum().item() / max(1, (P & (pos | neg)).sum().item()),
           "labeled_support": (P & (pos | neg)).sum().item() / max(1, P.sum().item()),
           "known_negative_rate": (P & neg).sum().item() / max(1, P.sum().item())}
    if M >= 2:
        ce = torch.stack([torch.arange(36, device=P.device) // 6, torch.arange(36, device=P.device) % 6], -1).float()
        dist = torch.cdist(ce, ce)
        pf = P.float()
        pair = torch.einsum("bi,ij,bj->b", pf, dist, pf)
        n = pf.sum(1)
        out["mean_pairwise_cell_distance"] = (pair / (n * (n - 1)).clamp(min=1)).mean().item()
    return out


def load_model(path, dev):
    import torch
    from models import build
    ck = torch.load(path, map_location=dev, weights_only=False)
    m = build(ck["kind"], ck["cfg"]).to(dev)
    m.load_state_dict(ck["state"]); m.eval()
    return m, ck


def cmd_evaluate(cfg, gcfg, a):
    import torch
    from models import build
    from data3d import Data
    pre, dev = preflight(cfg, gcfg, a.dry_run)
    exp = ROOT / "runs" / cfg["experiment"]
    if a.split == "test":
        fz = exp / "FROZEN_FOR_TEST.json"
        if not fz.exists():
            raise SystemExit("REFUSED: test evaluation before the selection is frozen (run `freeze` after the val report)")
        frozen = json.loads(fz.read_text())
        bad = [p for p, h in frozen["checkpoints"].items() if sha(ROOT / p) != h] + ([] if frozen["config_sha256"] == sha(a.config) else ["config"])
        if bad:
            raise SystemExit(f"REFUSED: frozen selection changed: {bad}")
    if a.dry_run:
        print(json.dumps({"dry_run": True, "split": a.split})); return
    wd = start_watchdog(exp / f"eval_{a.split}", a.guard_config)
    try:
        Dfull, Dgeo = Data(ROOT / cfg["derived"], dev), None
        results, lat = {"split": a.split, "utc": utc(), "M": cfg["M"], "raw_cap_factor": cfg["raw_cap_factor"], "samplers": {}}, {}
        srcs = torch.as_tensor(Dfull.meta["src"].astype("int64"), device=dev)
        runs = sorted(p for p in exp.glob("*_s*/best.pt"))
        families = {}
        for p in runs:
            m, ck = load_model(p, dev)
            fam = ck["kind"] + ("_geo" if ck["geometry_only"] else "")
            families.setdefault(fam, []).append((p, m, ck))
        families["uniform"] = [(None, build("uniform", cfg), {"seed": 0})]
        families["stratified"] = [(None, build("stratified", cfg), {"seed": 0})]
        C = Dfull.contexts(a.split)
        C = C[(Dfull.c_pos[C] | Dfull.c_neg[C]).any(1)]
        r = Dfull.c_row[C]
        pos, neg, legal = Dfull.c_pos[C], Dfull.c_neg[C], Dfull.legal[r]
        strata = {k: Dfull.meta[k][r.cpu().numpy()] for k in ("shape", "height_bin", "block", "n_obj")}
        for fam, members in families.items():
            if fam.startswith("ranker"):
                continue
            per_seed = []
            for p, m, ck in members:
                if ck.get("geometry_only"):
                    Dgeo = Dgeo or Data(ROOT / cfg["derived"], dev, geometry_only=True)
                    D = Dgeo
                else:
                    D = Dfull
                obj, mask, occ = D.obj[r], D.mask[r], D.occ[r]
                for es in cfg["eval_seeds"]:
                    gen = torch.Generator(device=dev).manual_seed(1000 + es)
                    row = {"train_seed": ck["seed"], "eval_seed": es, "by_M": {}}
                    for M in cfg["M"]:
                        R = cfg["raw_cap_factor"] * M
                        torch.cuda.synchronize(); t0 = time.time()
                        out = []
                        for i in range(0, len(r), 2048):
                            sl = slice(i, i + 2048)
                            kw = {"src": srcs[r[sl]]} if fam == "stratified" else {}
                            out.append(m.sample(obj[sl], mask[sl], occ[sl], legal[sl], R, gen=gen, **kw))
                        torch.cuda.synchronize()
                        d = torch.cat([o[0] for o in out]); fl = {k: torch.cat([o[1][k] for o in out]) for k in out[0][1]}
                        props, used, keep = first_unique(d, M)
                        mt = metrics(props, used, keep, fl, pos, neg, legal, M)
                        mt["latency_ms_per_context"] = 1000 * (time.time() - t0) / len(r)
                        if M == 4:
                            mt["by_stratum"] = {k: {str(v): {"hit_at_M": (props & pos).any(1)[(torch.as_tensor(strata[k] == v, device=dev)) & pos.any(1)].float().mean().item(),
                                                             "known_label_precision": (props & pos)[torch.as_tensor(strata[k] == v, device=dev)].sum().item() /
                                                             max(1, (props & (pos | neg))[torch.as_tensor(strata[k] == v, device=dev)].sum().item()),
                                                             "contexts": int((strata[k] == v).sum())} for v in sorted(set(strata[k].tolist()))}
                                                for k in strata}
                        row["by_M"][str(M)] = mt
                    per_seed.append(row)
                if fam.startswith("cvae"):                                    # proposal log-likelihood of held-out positives (IWAE)
                    pr = torch.nonzero((D.split == ["train", "val", "test"].index(a.split)) & (D.label == 1)).squeeze(1)
                    lp = torch.cat([m.log_prob(D.obj[b], D.mask[b], D.occ[b], D.legal[b], D.dest[b]) for b in pr.split(4096)])
                    per_seed[-1]["heldout_positive_nll"] = -lp.mean().item()
            if fam == "uniform":
                pr = torch.nonzero((Dfull.split == ["train", "val", "test"].index(a.split)) & (Dfull.label == 1)).squeeze(1)
                per_seed[-1]["heldout_positive_nll"] = Dfull.legal[pr].float().sum(1).log().mean().item()
            results["samplers"][fam] = {"members": [str(p.relative_to(ROOT)) if p else None for p, _, _ in members], "rows": per_seed}
        # separate ranker: deterministic top-M + calibration on labelled rows of this split
        for fam in [f for f in families if f.startswith("ranker")]:
            rows = []
            for p, m, ck in families[fam]:
                with torch.no_grad():
                    lg = torch.cat([m.logits_all(Dfull.obj[b], Dfull.mask[b], Dfull.occ[b]) for b in r.split(4096)])
                lg = lg.masked_fill(~legal, -1e9)
                row = {"train_seed": ck["seed"], "by_M": {}}
                for M in cfg["M"]:
                    top = lg.topk(M, 1).indices
                    props = torch.zeros_like(legal).scatter_(1, top, True) & legal
                    keep = torch.ones_like(top, dtype=torch.bool)
                    z = torch.zeros_like(top, dtype=torch.bool)
                    row["by_M"][str(M)] = metrics(props, torch.full((len(r),), M, device=dev), keep, {"raw_illegal": z, "raw_out_of_domain": z}, pos, neg, legal, M)
                lab = torch.nonzero((Dfull.split == ["train", "val", "test"].index(a.split)) & (Dfull.label >= 0)).squeeze(1)
                with torch.no_grad():
                    pl = torch.cat([m.logits_all(Dfull.obj[b], Dfull.mask[b], Dfull.occ[b]).gather(1, Dfull.dest[b][:, None]).squeeze(1) for b in lab.split(4096)])
                pp, yy = torch.sigmoid(pl), Dfull.label[lab].float()
                bins = torch.clamp((pp * 10).long(), max=9)
                ece = sum(((bins == k).float().mean() * (pp[bins == k].mean() - yy[bins == k].mean()).abs()).item() for k in range(10) if (bins == k).any())
                order = pp.argsort(descending=True)
                ys = yy[order]; tp = ys.cumsum(0); fp = (1 - ys).cumsum(0)
                auc = ((tp[-1] - tp)[ys == 0].sum() / max(1, tp[-1].item() * (len(ys) - tp[-1]).item())).item()
                row["calibration_labeled_rows"] = {"n": int(len(lab)), "brier": ((pp - yy) ** 2).mean().item(), "ece_10bin": ece, "auroc": 1 - auc,
                                                   "base_rate": yy.mean().item()}
                rows.append(row)
            results["samplers"][fam] = {"members": [str(p.relative_to(ROOT)) for p, _, _ in families[fam]], "rows": rows}
        results["contexts_evaluated"] = int(len(r)); results["contexts_with_positive"] = int(pos.any(1).sum())
        results["device"] = torch.cuda.get_device_name(0); results["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        out = exp / f"eval_{a.split}.json"
        out.write_text(json.dumps(results, indent=1)); print(json.dumps({"wrote": str(out), "contexts": len(r)}))
    finally:
        wd.terminate()


def cmd_freeze(cfg, gcfg, a):
    exp = ROOT / "runs" / cfg["experiment"]
    if not (exp / "eval_val.json").exists():
        raise SystemExit("REFUSED: freeze needs the val report first")
    if (exp / "FROZEN_FOR_TEST.json").exists():
        raise SystemExit("REFUSED: already frozen")
    ck = {str(p.relative_to(ROOT)): sha(p) for p in sorted(exp.glob("*_s*/best.pt"))}
    (exp / "FROZEN_FOR_TEST.json").write_text(json.dumps({"utc": utc(), "config_sha256": sha(a.config), "checkpoints": ck,
                                                          "val_report_sha256": sha(exp / "eval_val.json"),
                                                          "rule": "all trained members, best-val-loss checkpoint each; no hyperparameter or checkpoint change after this"}, indent=1))
    print(json.dumps({"frozen": len(ck)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["smoke", "fit", "evaluate", "freeze"])
    ap.add_argument("--config", required=True); ap.add_argument("--guard-config")
    ap.add_argument("--kind", choices=["cvae", "fm", "ranker"]); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--geometry-only", action="store_true"); ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg, gcfg = json.loads(Path(a.config).read_text()), guard.load_cfg(a.guard_config)
    {"smoke": cmd_smoke, "fit": cmd_fit, "evaluate": cmd_evaluate, "freeze": cmd_freeze}[a.cmd](cfg, gcfg, a)
