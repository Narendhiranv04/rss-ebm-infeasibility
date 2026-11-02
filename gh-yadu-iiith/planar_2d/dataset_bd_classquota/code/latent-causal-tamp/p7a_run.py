"""P7a.4: controlled ABC7 training on the frozen P6 corpus (0 teacher calls), one process-tree guard.

Order (predeclared; later items run only while the budget allows, never displacing earlier ones):
  1 micro-overfit on two frozen development roots with decided A / B / A+B (B, C, B-no-coupling, A)
  2 coefficient grid, seed 0, per arm A / B / C: clip in {1, 5} x lambda_pool in {0.5, 1}; choice = min
    validation outcome NLL (P6 validation roots only)
  3 seeds 1, 2 of A / B / C and seeds 0-2 of B / C-no-coupling at the parent arm's chosen config
  4 one seed each: S (non-energy), A-capacity, A / B / C-geometry, B / C-quadratic, B-structured
Temperature is fitted on validation only. The P6 holdout was already examined in P6: it is evaluated
as a LEGACY DIAGNOSTIC and is never used for any choice.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import abc7_models as a7
import abc7_train as t7
import config as cf
import p7_eval as ev
import teacher

CAPS = {"cpu_s": 6000.0, "wall_s": 12000.0, "reserve_s": 500.0}
MICRO = ["p6_2026102301", "p6_2026102303"]
GRID = [{"clip": c, "lam": {"pool": lp}} for c in (1.0, 5.0) for lp in (0.5, 1.0)]
EPOCHS, EVERY, PATIENCE, LR = 300, 5, 8, 1e-3
CAP_EXTRA = 1028                     # |A-capacity| within 1% of |C| (450,596 + 257 * 1028 + 128 = 714,920 vs 714,922)
SNAP_ROOT = "p6_2026102303"


def run(args) -> int:
    out = Path(args.out)
    if (out / "results.json").exists():
        raise SystemExit("P7a results exist; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", "import sys, p7a_run; p7a_run.worker(sys.argv[1])", str(out)],
                      CAPS["cpu_s"], CAPS["wall_s"], out / "heartbeat.txt", 30.0, 0.5, Path("."))
    cf.write_json(out / "guard.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"caps": CAPS})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s", "overshoot_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def micro(roots, log) -> dict:
    """Tiny-set capability: every decided field of two real roots, the root-level scene block and kappa."""
    res = {}
    for name in ("B", "C", "B-no-coupling", "A"):
        torch.manual_seed(0)
        m = a7.build(name)
        R = roots[0]["R"]
        v = {k: m.scene_vec(R, m.edited(R, p)) for k, p in [("none", {})] + sorted(roots[0]["edits"].items())}
        dist = {f"{a}|{b}": float((v[a] - v[b]).norm() / v["none"].norm()) for a in v for b in v if a < b}
        h = t7.train(m, roots, 300, 900.0, log, lam={"pool": 0.5}, lr=3e-3, clip=5.0)
        rows = []
        for r in roots:
            e = ev.evaluate(m, r)
            err = [abs(c["p"] - c["y"]) for c in e["cands"] if c["y"] is not None]
            err += [abs(e["scene"][k] - y) for k, y in (e["kappa"] or {}).get("scene_obs", {}).items()]
            for c in e["cands"]:
                err += [abs(p - z) for p, z in zip(c.get("marg", []), c.get("z_obs", [])) if z is not None]
            rows.append({"root": r["key"], "max_abs_field_error": max(err), "n_fields": len(err), "kappa": e["kappa"]})
        res[name] = {"edit_vector_rel_l2_at_init": dist, "final_loss": h[-1].get("total"), "roots": rows,
                     "pass": all(x["max_abs_field_error"] < 0.25 and x["kappa"] and round(x["kappa"]["pred"]) == x["kappa"]["obs"] for x in rows)}
        log(f"micro {name}: pass={res[name]['pass']} errors={[round(x['max_abs_field_error'], 3) for x in rows]}")
    return res


def fit(name, seed, cfg, train, val, budget, log, snap_root=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = a7.build(name.replace("-capacity", "").replace("-geometry", ""), vision="-geometry" not in name,
                 extra=CAP_EXTRA if name == "A-capacity" else 0)
    snaps = []
    snap = None
    if snap_root is not None:
        def snap(ep, mm):
            e = ev.evaluate(mm, snap_root)
            rec = {"epoch": ep, "scene": e["scene"], "kappa": e["kappa"], "cands": [{k: c[k] for k in ("query", "p", "y", "m")} for c in e["cands"]]}
            if ep % 40 == 0:
                rec["term_grads"] = {k: v["total"] for k, v in t7.term_grads(mm, train).items()}
            snaps.append(rec)
    c0 = time.process_time()
    hist = t7.train(m, train, EPOCHS, budget, log, lam=cfg["lam"], lr=LR, clip=cfg["clip"], val=val, every=EVERY, patience=PATIENCE, snap=snap)
    return m, {"history": hist, "snapshots": snaps, "cpu_s": round(time.process_time() - c0, 1), "params": sum(p.numel() for p in m.parameters()),
               "config": cfg, "seed": seed, "val_metric": t7.val_metric(m, val), "term_grads_final": t7.term_grads(m, train),
               "clip_fraction": float(np.mean([h["clipped"] for h in hist if "clipped" in h]))}


def worker(out) -> None:
    torch.set_num_threads(1)
    out, t0 = Path(out), time.process_time()
    log = lambda msg: open(out / "log.txt", "a").write(f"{time.process_time() - t0:8.1f} {msg}\n")  # noqa: E731
    ds = torch.load("runs/p6/train/dataset.pt", weights_only=False)
    full = t7.prepare(ds)
    legacy = t7.prepare(torch.load("runs/p6/train/dataset_sealed.pt", weights_only=False))
    train, val = [r for r in full if r["split"] == "train"], [r for r in full if r["split"] == "validation"]
    by = {r["key"]: r for r in full}
    res = {"micro": micro([by[k] for k in MICRO], log), "runs": {}, "grid": {}}
    left = lambda: CAPS["cpu_s"] - CAPS["reserve_s"] - (time.process_time() - t0)  # noqa: E731
    models, chosen = {}, {}

    def go(name, seed, cfg, n_left, snap=False):
        tag = f"{name}#s{seed}|clip{cfg['clip']}|pool{cfg['lam']['pool']}"
        if left() < 120:
            res["runs"][tag] = {"status": "NOT_RUN", "reason": f"remaining {left():.0f} CPU-s"}
            return None
        m, rec = fit(name, seed, cfg, train, val, left() / max(1, n_left), log, by[SNAP_ROOT] if snap else None)
        evals = {"validation": [ev.evaluate(m, r) for r in val]}
        tau = ev.fit_temperature(evals["validation"])
        evals["legacy_p6_holdout"] = [ev.evaluate(m, r) for r in legacy]
        rec |= {"status": "RUN", "tau": tau, "eval": evals,
                "metrics": {s: {"raw": ev.metrics(v, 1.0), "calibrated": ev.metrics(v, tau)} for s, v in evals.items()}}
        res["runs"][tag] = rec
        models[tag] = m
        torch.save(m.state_dict(), out / f"ckpt_{tag.replace('|', '_')}.pt")
        log(f"{tag}: {rec['cpu_s']} CPU-s, val {rec['val_metric']:.3f}, clip frac {rec['clip_fraction']:.2f}")
        return rec
    plan = [(a, 0, cfg) for a in ("A", "B", "C") for cfg in GRID]
    for k, (a, s, cfg) in enumerate(plan):
        r = go(a, s, cfg, len(plan) - k + 15, snap=True)
        if r:
            res["grid"].setdefault(a, []).append({"config": cfg, "val_metric": r["val_metric"]})
    for a in ("A", "B", "C"):
        chosen[a] = min(res["grid"][a], key=lambda x: x["val_metric"])["config"]
    res["chosen"] = chosen
    rest = [(a, s, chosen[a], s == 0) for s in (1, 2) for a in ("A", "B", "C")]
    rest += [(f"{a}-no-coupling", s, chosen[a], s == 0) for s in (0, 1, 2) for a in ("B", "C")]
    rest += [("S", 0, chosen["A"], False), ("A-capacity", 0, chosen["A"], False), ("B-geometry", 0, chosen["B"], False),
             ("C-geometry", 0, chosen["C"], False), ("A-geometry", 0, chosen["A"], False), ("B-quadratic", 0, chosen["B"], True),
             ("C-quadratic", 0, chosen["C"], False), ("B-structured", 0, chosen["B"], False)]
    for k, (a, s, cfg, snap) in enumerate(rest):
        go(a, s, cfg, len(rest) - k, snap=snap)
    res["data"] = {"train_roots": len(train), "validation_roots": len(val), "legacy_holdout_roots": len(legacy),
                   "train_candidates": sum(len(r["cands"]) for r in train)}
    res["config"] = {"epochs_max": EPOCHS, "every": EVERY, "patience": PATIENCE, "lr": LR, "wd": 1e-4, "grid": GRID, "cap_extra": CAP_EXTRA,
                     "model_version": a7.VERSION, "T": a7.T}
    res["cpu_s"] = round(time.process_time() - t0, 1)
    cf.write_json(out / "results.json", json.loads(json.dumps(res, default=float)))
    log("done")
