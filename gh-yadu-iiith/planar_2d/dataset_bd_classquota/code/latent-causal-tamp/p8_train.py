"""P8a: decisive coupling comparison on saved P4-P7 evidence (0 teacher calls, one process-tree guard, 7,000 CPU-s).

Predeclared protocol:
  data      train = runs/p7/training dataset 'train' roots (P7b train + P4-P6 history), validation = P7b validation roots,
            legacy diagnostic = P7b test roots (already disclosed in P7; never used for any choice)
  updates   UPDATES matched minibatch updates of BATCH roots from abc8.RootSampler (kappa-balanced, seeded; identical batch
            sequence for every arm with the same seed); validation outcome NLL every EVAL_EVERY updates, best restored
  grid      seed 0, FULL_DECIDED: clip {1, 5} x lambda_pool {0.25, 0.5} for B-full, B-structured, B-none; clip {1, 5} for SF
  primary   B-full, B-structured, B-none, SF x seeds 0-2 x {FULL_DECIDED, SPARSE_AUX (50 % of auxiliary labels hidden on
            training roots, abc8.sparse_aux)} at each arm's chosen config
  extra     B-quadratic (B-full config), C-full (B-full), C-structured (B-structured), A (B-full) x seeds 0-2 FULL_DECIDED,
            in this order, while the budget lasts (NOT_RUN otherwise)
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import abc7_train as t7
import abc8
import config as cf
import p7_eval as ev
import p8_eval as pe
import teacher

CAPS = {"cpu_s": 7000.0, "wall_s": 14000.0, "reserve_s": 300.0}
UPDATES, BATCH, EVAL_EVERY, LR, WD = 200, 12, 20, 1e-3, 1e-4
GRID = [{"clip": c, "pool": p} for c in (1.0, 5.0) for p in (0.25, 0.5)]
PRIMARY = ["B-full", "B-structured", "B-none", "SF"]
EXTRA = [("B-quadratic", "B-full"), ("C-full", "B-full"), ("C-structured", "B-structured"), ("A", "B-full")]


def fit(arm, seed, cfg, train, val, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = abc8.build(arm)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=WD)
    smp, lam = abc8.RootSampler(train, BATCH, seed), {"pool": cfg["pool"]}
    hist, best, c0 = [], (math.inf, None, -1), time.process_time()
    for u in range(UPDATES):
        batch = [train[i] for i in smp.next()]
        opt.zero_grad()
        tot, rec = t7.total(m, batch, lam)
        if not torch.isfinite(tot):
            hist.append({"update": u, "nonfinite": True})
            log(f"{arm}#s{seed}: non-finite loss at update {u}")
            break
        tot.backward()
        pre = float(torch.nn.utils.clip_grad_norm_(m.parameters(), cfg["clip"]))
        opt.step()
        rec |= {"update": u, "total": float(tot.detach()), "grad_pre": pre, "clipped": pre > cfg["clip"]}
        if (u + 1) % EVAL_EVERY == 0:
            rec["val"] = t7.val_metric(m, val)
            if rec["val"] < best[0]:
                best = (rec["val"], copy.deepcopy(m.state_dict()), u + 1)
        hist.append(rec)
    if best[1] is not None:
        m.load_state_dict(best[1])
    return m, {"history": hist, "best_update": best[2], "val_metric": best[0], "updates": UPDATES, "cpu_s": round(time.process_time() - c0, 1),
               "params": sum(p.numel() for p in m.parameters()), "clip_fraction": float(np.mean([h.get("clipped", False) for h in hist])),
               "sampler_groups": {str(k): int(v) for k, v in smp.groups.items()}}


def evaluate(m, val, legacy) -> dict:
    v = pe.evaluate_all(m, val)
    tau = ev.fit_temperature(v["_evals"])
    lg = pe.evaluate_all(m, legacy, tau)
    v["repair"] = ev.metrics(v["_evals"], tau)
    strip = lambda d: {k: x for k, x in d.items() if not k.startswith("_")}  # noqa: E731
    return {"tau": tau, "validation": strip(v), "legacy_p7_test": strip(lg),
            "evals": {"validation": v["_evals"], "legacy_p7_test": lg["_evals"]}, "cond_rows": {"validation": v["_cond"], "legacy_p7_test": lg["_cond"]}}


def run(args) -> int:
    out = Path(args.out)
    if (out / "results.json").exists():
        raise SystemExit("P8a results exist; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", "import sys, p8_train; p8_train.worker(sys.argv[1])", str(out)],
                      CAPS["cpu_s"], CAPS["wall_s"], out / "heartbeat.txt", 30.0, 0.5, Path("."))
    cf.write_json(out / "guard.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"caps": CAPS})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s", "overshoot_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def worker(out) -> None:
    torch.set_num_threads(1)
    out, t0 = Path(out), time.process_time()
    log = lambda msg: open(out / "log.txt", "a").write(f"{time.process_time() - t0:8.1f} {msg}\n")  # noqa: E731
    full = t7.prepare(torch.load("runs/p7/training/dataset.pt", weights_only=False))
    legacy = t7.prepare(torch.load("runs/p7/training/dataset_sealed.pt", weights_only=False))
    train, val = [r for r in full if r["split"] == "train"], [r for r in full if r["split"] == "validation"]
    sparse = copy.deepcopy(train)
    smask = abc8.sparse_aux(sparse)
    res = {"runs": {}, "grid": {}, "sparse_mask": smask, "data": {"train_roots": len(train), "validation_roots": len(val), "legacy_roots": len(legacy),
           "train_kappa_groups": {g: sum(abc8.kappa_group(r) == g for r in train) for g in ("1", "0", "-1", "none")}}}
    left = lambda: CAPS["cpu_s"] - CAPS["reserve_s"] - (time.process_time() - t0)  # noqa: E731

    def go(arm, seed, cfg, regime):
        tag = f"{arm}|{regime}|s{seed}|clip{cfg['clip']}|pool{cfg['pool']}"
        if left() < 200:
            res["runs"][tag] = {"status": "NOT_RUN", "reason": f"remaining {left():.0f} CPU-s"}
            log(f"{tag}: NOT_RUN")
            return None
        m, rec = fit(arm, seed, cfg, sparse if regime == "SPARSE_AUX" else train, val, log)
        rec |= {"status": "RUN", "arm": arm, "seed": seed, "regime": regime, "config": cfg, **evaluate(m, val, legacy)}
        torch.save(m.state_dict(), out / f"ckpt_{tag.replace('|', '_')}.pt")
        res["runs"][tag] = rec
        log(f"{tag}: {rec['cpu_s']} CPU-s, val {rec['val_metric']:.3f} @ {rec['best_update']}, clip {rec['clip_fraction']:.2f}")
        return rec
    for arm in PRIMARY:
        for cfg in (GRID if arm != "SF" else [g for g in GRID if g["pool"] == 0.5]):
            r = go(arm, 0, cfg, "FULL_DECIDED")
            if r:
                res["grid"].setdefault(arm, []).append({"config": cfg, "val_metric": r["val_metric"]})
    chosen = {a: min(res["grid"][a], key=lambda x: x["val_metric"])["config"] for a in PRIMARY if a in res["grid"]}
    res["chosen"] = chosen
    for regime in ("FULL_DECIDED", "SPARSE_AUX"):
        for seed in (0, 1, 2):
            for arm in PRIMARY:
                if regime == "FULL_DECIDED" and seed == 0:
                    continue                                  # the chosen grid run is the seed-0 FULL run
                go(arm, seed, chosen[arm], regime)
    for seed in (0, 1, 2):
        for arm, parent in EXTRA:
            go(arm, seed, chosen[parent], "FULL_DECIDED")
    res["config"] = {"updates": UPDATES, "batch": BATCH, "eval_every": EVAL_EVERY, "lr": LR, "wd": WD, "grid": GRID, "model_version": abc8.VERSION}
    res["cpu_s"] = round(time.process_time() - t0, 1)
    cf.write_json(out / "results.json", json.loads(json.dumps(res, default=float)))
    log("done")
