"""P7b.5: ABC7 retraining on P7b data with the configurations frozen at the P7a gate (0 teacher calls, one guard).

Runs (priority order): A / B / C x seeds 0-2; B / C-no-coupling x seeds 0-2; then one seed each of S, A-capacity,
A / B / C-geometry, B / C-quadratic, B-structured while the budget allows (NOT_RUN otherwise). Temperature and
the 0.5 feasibility threshold are frozen from P7b validation only; the sealed P7b test roots are loaded once, after
all checkpoints and the one-shot rankings are frozen. One-shot rankings (seed-0 models; fewest moves among
calibrated p >= 0.5, then predicted J, displacement, energy; at most 5) and a seeded uniform control are written
with their hashes before any episode.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import abc7_train as t7
import config as cf
import p7_eval as ev
import p7_learn as pl
import p7_oneshot as po
import p7a_run as pa
import teacher

CAPS = {"cpu_s": 12000.0, "wall_s": 24000.0, "reserve_s": 900.0}
PRIMARY = [(a, s) for s in (0, 1, 2) for a in ("A", "B", "C")] + [(f"{a}-no-coupling", s) for s in (0, 1, 2) for a in ("B", "C")]
SECONDARY = ["S", "A-capacity", "B-geometry", "C-geometry", "A-geometry", "B-quadratic", "C-quadratic", "B-structured"]


def run(args) -> int:
    out = Path(args.out)
    if (out / "results.json").exists():
        raise SystemExit("P7b training results exist; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", "import sys, p7b_run; p7b_run.worker(*sys.argv[1:])", str(out), args.collect, args.p7a_gate],
                      CAPS["cpu_s"], CAPS["wall_s"], out / "heartbeat.txt", 30.0, 0.5, Path("."))
    cf.write_json(out / "guard.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"caps": CAPS})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s", "overshoot_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def ranking(e, tau, thr=0.5, k=5) -> list:
    keep = [c for c in e["cands"] if 1 / (1 + np.exp(-c["u"] / tau)) >= thr]
    return [c["query"] for c in sorted(keep, key=lambda c: (c["m"], c["J_pred"], c["C_d"], c["energy"], c["query"]))][:k]


def worker(out, collect, gate_path) -> None:
    torch.set_num_threads(1)
    out, t0 = Path(out), time.process_time()
    log = lambda msg: open(out / "log.txt", "a").write(f"{time.process_time() - t0:8.1f} {msg}\n")  # noqa: E731
    gate = json.loads(Path(gate_path).read_text())
    chosen = gate["chosen_configs"]
    cfg = cf.load_config("config.toml")
    ds = pl.build(cfg, out, Path(collect))
    full = t7.prepare(ds)
    train, val = [r for r in full if r["split"] == "train"], [r for r in full if r["split"] == "validation"]
    log(f"data: {len(train)} train roots ({sum(len(r['cands']) for r in train)} candidates), {len(val)} validation roots")
    res, models = {"runs": {}, "chosen_configs": chosen}, {}
    left = lambda: CAPS["cpu_s"] - CAPS["reserve_s"] - (time.process_time() - t0)  # noqa: E731
    plan = [(a, s, True) for a, s in PRIMARY] + [(a, 0, False) for a in SECONDARY]
    for k, (a, s, prim) in enumerate(plan):
        cfgr = chosen[a.split("-")[0] if a != "S" else "A"]
        tag = f"{a}#s{s}"
        n_left = sum(1 for x in plan[k:] if x[2] == prim) + (len(SECONDARY) if prim else 0)
        if left() < 150:
            res["runs"][tag] = {"status": "NOT_RUN", "reason": f"remaining {left():.0f} CPU-s"}
            log(f"{tag}: NOT_RUN")
            continue
        snap = next((r for r in val if r["scene_y"] and all(y is not None for y in r["scene_y"].values())), None)
        m, rec = pa.fit(a, s, cfgr, train, val, left() / max(1, n_left), log, snap if (s == 0 and a in ("A", "B", "C", "B-no-coupling")) else None)
        evals = {"validation": [ev.evaluate(m, r) for r in val]}
        tau = ev.fit_temperature(evals["validation"])
        rec |= {"status": "RUN", "tau": tau, "eval": evals, "metrics": {"validation": {"raw": ev.metrics(evals["validation"], 1.0),
                                                                                         "calibrated": ev.metrics(evals["validation"], tau)}}}
        res["runs"][tag], models[tag] = rec, m
        torch.save(m.state_dict(), out / f"ckpt_{tag}.pt")
        log(f"{tag}: {rec['cpu_s']} CPU-s, val {rec['val_metric']:.3f}, tau {tau:.2f}, clip frac {rec['clip_fraction']:.2f}")
    log("all checkpoints frozen; unsealing P7b test roots once")
    dst = pl.build(cfg, out, Path(collect), sealed=True)
    test = t7.prepare(dst)
    for tag, m in models.items():
        e = [ev.evaluate(m, r) for r in test]
        res["runs"][tag]["eval"]["test"] = e
        res["runs"][tag]["metrics"]["test"] = {"raw": ev.metrics(e, 1.0), "calibrated": ev.metrics(e, res["runs"][tag]["tau"])}
    jobs, cplan = [], json.loads(Path(collect, "plan.json").read_text())
    seqs = {p["z_layout"]["query"].get("zR", f"seq:{k}:empty"): p["sequence"] for k, r in cplan["roots"].items() if r["queries"]
            for p in r["queries"]["pool"]}
    test_roots = [r["key"] for r in test if not r["key"].endswith("_ref")]
    for arm in ("A", "B", "C"):
        tag = f"{arm}#s0"
        if tag not in models:
            continue
        for e in res["runs"][tag]["eval"]["test"]:
            if e["root"] in test_roots:
                jobs.append({"arm": arm, "root": e["root"], "ranked": [seqs[q] for q in ranking(e, res["runs"][tag]["tau"])]})
    for i, k in enumerate(test_roots):
        cands = [p["sequence"] for p in cplan["roots"][k]["queries"]["pool"]]
        jobs.append({"arm": "U", "root": k, "ranked": po.uniform_ranking(cands, 20261024 + i)[:po.CAPS["max_rank"]]})
    res["oneshot_jobs"] = jobs
    res["oneshot_jobs_hash"] = cf.stable_hash(jobs)
    res["cpu_s"] = round(time.process_time() - t0, 1)
    cf.write_json(out / "rankings.json", {"jobs": jobs, "hash": res["oneshot_jobs_hash"]})
    cf.write_json(out / "results.json", json.loads(json.dumps(res, default=float)))
    log("done")
