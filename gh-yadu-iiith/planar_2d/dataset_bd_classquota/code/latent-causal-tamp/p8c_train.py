"""P8c training + frozen P8d rankings (0 teacher calls, one guard: 11,000 CPU-s, 22,000 s wall).

Arms A, B-full, B-structured, B-none, C-full, SF x seeds 0-2 at the configurations frozen by P8a (A / C-full use B-full's,
as in P8a), UPDATES matched minibatch updates from the kappa-balanced sampler. Temperature and the selection threshold
(p8_select.choose_threshold over {0.5, 0.7, 0.85, 0.95}) come from P8c VALIDATION roots only. After every checkpoint,
threshold and the one-shot ranking set (A, B-full, B-structured, SF seed-0 models; U seeded uniform; G+M) are frozen and
hashed, the sealed test roots are loaded ONCE for offline metrics. Rankings: admitted candidates (calibrated p >= threshold)
in lexicographic order (fewest moves, predicted J, displacement, energy), at most 5; empty list = ABSTAIN.
"""
from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import abc7_train as t7
import config as cf
import geometry as ge
import p7_eval as ev
import p7_oneshot as po
import p8_eval as pe
import p8_learn as pl
import p8_select as ps
import p8_train as pt
import teacher

CAPS = {"cpu_s": 11000.0, "wall_s": 22000.0, "reserve_s": 1200.0}
ARMS = ["A", "B-full", "B-structured", "B-none", "C-full", "SF"]
OPTIONAL = ["B-quadratic"]            # P8a's best variant; offline only, after the mandatory 18 runs, never in P8d
EXEC = ["A", "B-full", "B-structured", "SF", "U", "G+M"]
C8 = Path("runs/p8/collection")


def run(args) -> int:
    out = Path(args.out)
    if (out / "results.json").exists():
        raise SystemExit("P8c results exist; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", "import sys, p8c_train; p8c_train.worker(sys.argv[1])", str(out)], CAPS["cpu_s"], CAPS["wall_s"],
                      out / "heartbeat.txt", 30.0, 0.5, Path("."))
    cf.write_json(out / "guard.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"caps": CAPS})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s", "overshoot_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def ranking(e, tau, thr, k=5) -> list:
    keep = [c for c in e["cands"] if 1 / (1 + np.exp(-c["u"] / tau)) >= thr]
    return [c["query"] for c in sorted(keep, key=lambda c: (c["m"], c["J_pred"], c["C_d"], c["energy"], c["query"]))][:k]


def worker(out) -> None:
    torch.set_num_threads(1)
    out, t0 = Path(out), time.process_time()
    log = lambda msg: open(out / "log.txt", "a").write(f"{time.process_time() - t0:8.1f} {msg}\n")  # noqa: E731
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    chosen = json.loads(Path("runs/p8/a/results.json").read_text())["chosen"]
    conf = {a: chosen.get(a, chosen["B-full"]) for a in ARMS + OPTIONAL}
    full = t7.prepare(pl.build(cfg, out, C8))
    train, val = [r for r in full if r["split"] == "train"], [r for r in full if r["split"] == "validation"]
    plan = json.loads((C8 / "plan.json").read_text())
    records = json.loads((C8 / "decisions.json").read_text())
    log(f"data: {len(train)} train roots ({sum(len(r['cands']) for r in train)} candidates), {len(val)} validation roots")
    res, models = {"runs": {}, "configs": conf}, {}
    left = lambda: CAPS["cpu_s"] - CAPS["reserve_s"] - (time.process_time() - t0)  # noqa: E731
    for seed, arm in [(s, a) for s in (0, 1, 2) for a in ARMS] + [(s, a) for s in (0, 1, 2) for a in OPTIONAL]:
        tag = f"{arm}#s{seed}"
        if left() < 200:
            res["runs"][tag] = {"status": "NOT_RUN", "reason": f"remaining {left():.0f} CPU-s"}
            continue
        m, rec = pt.fit(arm, seed, conf[arm], train, val, log)
        v = pe.evaluate_all(m, val)
        tau = ev.fit_temperature(v["_evals"])
        th = ps.choose_threshold(v["_evals"], records, C8, tau)
        rec |= {"status": "RUN", "arm": arm, "seed": seed, "tau": tau, "threshold": th,
                "validation": {k: x for k, x in v.items() if not k.startswith("_")} | {"repair": ev.metrics(v["_evals"], tau)}}
        res["runs"][tag], models[tag] = rec, m
        torch.save(m.state_dict(), out / f"ckpt_{tag}.pt")
        log(f"{tag}: {rec['cpu_s']} CPU-s, val {rec['val_metric']:.3f} @ {rec['best_update']}, tau {tau:.2f}, thr {th['chosen']}")
    test_keys = sorted(k for k, r in plan["roots"].items() if r["split"] == "test" and r["queries"])
    jobs, rng = [], np.random.default_rng(20261025)
    test = t7.prepare(pl.build(cfg, out, C8, sealed=True))
    blind = copy.deepcopy(test)                                            # rankings see a label-blinded copy only
    for r in blind:
        r["y"], r["scene_y"] = {k: None for k in r["y"]}, {k: None for k in r["scene_y"]}
        for c in r["cands"]:
            c["y"], c["cost"], c["z"] = None, None, [None] * len(c["z"])
    seqs = {p["z_layout"]["query"].get("zR", f"seq:{k}:empty"): p["sequence"] for k, r in plan["roots"].items() if r["queries"]
            for p in r["queries"]["pool"]}
    evs = {}
    for arm in ("A", "B-full", "B-structured", "SF"):
        m = models.get(f"{arm}#s0")
        if m is not None:
            evs[arm] = {e["root"]: e for e in (ev.evaluate(m, r) for r in blind if r["key"] in test_keys)}
    for i, k in enumerate(test_keys):
        order = list(rng.permutation(EXEC))
        for arm in order:
            if arm == "U":
                rk = po.uniform_ranking([p["sequence"] for p in plan["roots"][k]["queries"]["pool"]], 20261025 + i)[:5]
            elif arm == "G+M":
                rk = ps.gm_rank(cfg, robot, plan, k)
            elif arm in evs:
                r_ = res["runs"][f"{arm}#s0"]
                rk = [seqs[q] for q in ranking(evs[arm][k], r_["tau"], r_["threshold"]["chosen"])]
            else:
                continue
            jobs.append({"arm": arm, "root": k, "ranked": rk})
    res["oneshot_jobs_hash"] = cf.stable_hash(jobs)
    cf.write_json(out / "rankings.json", {"jobs": jobs, "hash": res["oneshot_jobs_hash"], "frozen_cpu_s": round(time.process_time() - t0, 1)})
    log("all checkpoints, thresholds and rankings frozen; reading sealed test labels once")
    for tag, m in models.items():
        e = pe.evaluate_all(m, test, res["runs"][tag]["tau"])
        res["runs"][tag]["test"] = {k: x for k, x in e.items() if not k.startswith("_")}
        res["runs"][tag]["test_evals"] = e["_evals"]
        res["runs"][tag]["test_cond"] = e["_cond"]
    res["cpu_s"] = round(time.process_time() - t0, 1)
    cf.write_json(out / "results.json", json.loads(json.dumps(res, default=float)))
    log("done")
