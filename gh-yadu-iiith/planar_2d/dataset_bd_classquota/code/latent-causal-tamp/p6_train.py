"""P6.5: A/B/C retraining, ablations, baselines, K <= 9 sampler check and one sealed-holdout evaluation.

One process-tree guard covers the whole worker (9,000 CPU-s, 18,000 s wall, CPU only, 1 thread).
Runs execute in predeclared priority order; a run that no longer fits the remaining budget is
recorded as NOT_RUN. Every run: AdamW, gradient clipping (max norm 1.0, pre/post norms logged),
non-finite guard, validation-only early stopping (P6 validation roots), deterministic seed. B/C
primaries use the uniform reference measure and the higher-order coupling residual (P6); the
quadratic-only coupling is a named ablation. Holdout roots are loaded after every checkpoint is
frozen and evaluated exactly once. No teacher / planner call happens here.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import abc_models as am
import abc_run as ar
import abc_train as at
import config as cf
import g4b
import geometry as ge
import learn_data as ld
import p6_learn as pl
import p6_scenes as p6
import sampler as sp
import teacher
import zlayout_r4 as z4

CAPS = {"cpu_s": 9000.0, "wall_s": 18000.0, "reserve_s": 1300.0}
EPOCHS, EVERY, PATIENCE, CLIP, MIN_BUDGET_NEG = 300, 5, 8, 1.0, 8
SEEDS = [0, 1, 2]
PRIMARY = [(n, s) for s in SEEDS for n in ("A", "B", "C")]
ABLATIONS = ["B-no-coupling", "C-no-coupling", "A-capacity", "A-geometry", "B-geometry", "C-geometry", "A-no-order", "B-quadratic",
             "C-quadratic", "A+certified-only", "B+certified-only", "C+certified-only"]


def make(name):
    base = name.split("+")[0]
    bc = dict(measure="uniform", higher_order=not base.endswith("quadratic"))
    return {"A": lambda: am.build("A"), "A-capacity": lambda: am.build("A", extra=CAP_EXTRA), "A-no-order": lambda: am.build("A-no-order"),
            "A-geometry": lambda: am.build("A", vision=False), "B": lambda: am.build("B", **bc), "C": lambda: am.build("C", **bc),
            "B-quadratic": lambda: am.build("B", **bc), "C-quadratic": lambda: am.build("C", **bc),
            "B-no-coupling": lambda: am.build("B-no-coupling", measure="uniform"), "C-no-coupling": lambda: am.build("C-no-coupling", measure="uniform"),
            "B-geometry": lambda: am.build("B", vision=False, **bc), "C-geometry": lambda: am.build("C", vision=False, **bc)}[base]()


CAP_EXTRA = 515                       # A residual width: |A-capacity| within 1% of |C| (tests/test_p6_math.py)


def _auc(y, p):
    pos, neg = p[y == 1], p[y == 0]
    if not len(pos) or not len(neg):
        return None
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def metrics(evals) -> dict:
    rows = [c for e in evals for c in e["cands"] if c["y"] is not None]
    if not rows:
        return {"decided": 0}
    y, p = np.array([c["y"] for c in rows]), np.clip(np.array([c["p"] for c in rows]), 1e-6, 1 - 1e-6)
    pairs = [o for e in evals for o in e["order_pairs"] if None not in o["y"] and o["y"][0] != o["y"][1]]
    eff = [x for e in evals for x in e["effects"]]
    sel = [e["selected_label"] for e in evals if any(c["y"] == 1 and c["m"] > 0 for c in e["cands"])]
    return {"decided": len(rows), "positives": int(y.sum()), "accuracy": float(((p >= 0.5) == y).mean()), "brier": float(((p - y) ** 2).mean()),
            "nll": float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()), "auroc": _auc(y, p),
            "order_pairs": len(pairs), "order_accuracy": (sum((o["p"][0] > o["p"][1]) == (o["y"][0] > o["y"][1]) for o in pairs) / len(pairs)) if pairs else None,
            "effect_mae": {k: (float(np.mean([abs(x["pred"] - x["observed"]) for x in eff if x["kind"] == k])) if any(x["kind"] == k for x in eff) else None)
                           for k in ("scene", "omission")},
            "selection_roots": len(sel), "selection_success": (sum(s == 1 for s in sel) / len(sel)) if sel else None,
            "rank_acc": float(np.mean([e["rank_acc"] for e in evals if e["rank_acc"] is not None])) if any(e["rank_acc"] is not None for e in evals) else None}


def geometry_baseline(cfg, robot, plan, evals) -> list:
    """Analytic baseline: sequential pose-only grasp screens at the fixed start, then target grasp + place."""
    scr, out = p6.Screens(cfg, robot), []
    from scene import scene_from_dict
    for e in evals:
        r = plan["roots"][e["root"]]
        seqs = {x["z_layout"]["query"].get("zR", f"seq:{e['root']}:empty"): x["sequence"] for x in r["queries"]["pool"]} if r["queries"] else {}
        s0, rows = scene_from_dict(r["scene"]), []
        for c in e["cands"]:
            s, ok = s0, True
            for o, d in seqs.get(c["query"], []):
                if not scr.pick(s, o):
                    ok = False
                    break
                s = p6.moved(s, {o: d})
            ok = ok and bool(scr.pick(s, "t0")) and scr.place(s, "t0", r["goal_cell"]) > 0
            rows.append({**c, "p": float(ok)})
        elig = sorted([x for x in rows if x["p"] >= 0.5], key=lambda x: (x["m"], x["query"]))
        order = [{**o, "p": [next(x["p"] for x in rows if x["query"] == q) for q in o["orders"]]} for o in e["order_pairs"]]
        out.append({**e, "cands": rows, "effects": [], "order_pairs": order, "selected_label": elig[0]["y"] if elig else None, "rank_acc": None})
    return out


def uniform_baseline(evals) -> dict:
    sel = [np.mean([c["y"] for c in e["cands"] if c["m"] > 0 and c["y"] is not None]) for e in evals
           if any(c["y"] == 1 and c["m"] > 0 for c in e["cands"])]
    return {"p": 0.5, "expected_selection_success": float(np.mean(sel)) if sel else None, "roots": len(sel),
            "order_accuracy_expected": 0.5}


def sampler_check(m, r, rec) -> dict:
    """G4B kernels (cap 4) on the learned sampling energy of one root; exact detailed balance + chain TV."""
    ids = rec["audit"]["ids"]
    cell = lambda xy: int(round((xy[1] - 0.04) / 0.08)) * 8 + int(round((xy[0] - 0.04) / 0.08))  # noqa: E731
    main = max(r["cands"], key=lambda c: len(c["seq"]))
    keep = {ids[i] for i, _ in main["seq"]}
    dests = {}
    for c in r["cands"]:
        for i, d in c["seq"]:
            if ids[i] in keep:
                dests.setdefault(ids[i], set()).add(cell(d))
    dests = {o: sorted(v) for o, v in dests.items()}
    cat = [{"kind": e["kind"], "edit": {ids[i]: cell(p) for i, p in e["poses"].items()}, "eligible": e["eligible"]} for e in rec["edits"]]
    cache = {}

    @torch.no_grad()
    def energy(state):
        if state not in cache:
            seq = [(ids.index(o), ((d % 8 + 0.5) * 0.08, (d // 8 + 0.5) * 0.08)) for o, d in state]
            o = m.seq(r["R"], seq) if m.kind == "A" else m.joint(r["R"], seq, list(z4.build(r["key"], list(state), cat)["query"]), r["edits"])
            cache[state] = float(o["E"] if m.kind == "A" else o["E_bar"]) + ar.LAM_D * ar._cd(r, seq) + ar.LAM_Q * float(o["J"])
        return cache[state]
    dom, states = sp.Domain.make(dests, 4), g4b.reference_states(dests, 4)
    cert, pi, _ = g4b.certify(dom, energy, 1.0, states)
    chain, _ = sp.run_chain(dom, (), energy, 1.0, 4000, 0)
    freq = {s: 0 for s in states}
    for s in chain:
        freq[s] += 1
    top = sorted(zip(states, pi), key=lambda x: -x[1])[:6]
    return {"root": r["key"], "dests": dests, "states": len(states), "max_len": max(len(s) for s in states), "pass": cert["pass"],
            "max_error": cert["max_error"], "tv_chain_4000_vs_exact": 0.5 * sum(abs(freq[s] / len(chain) - p) for s, p in zip(states, pi)),
            "top": [[[list(a) for a in s], float(p), energy(s)] for s, p in top]}


def run(args) -> int:
    out = Path(args.out)
    if (out / "results.json").exists():
        raise SystemExit("P6 training results exist; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", "import sys, p6_train; p6_train.worker(*sys.argv[1:])", str(out), args.config, args.collect],
                      CAPS["cpu_s"], CAPS["wall_s"], out / "heartbeat.txt", 30.0, 0.5, Path("."))
    cf.write_json(out / "guard.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"caps": CAPS})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s", "overshoot_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def worker(out, config, collect) -> None:
    torch.set_num_threads(1)
    out, t0, w0 = Path(out), time.process_time(), time.time()
    log = lambda msg: open(out / "log.txt", "a").write(f"{time.process_time() - t0:8.1f} {msg}\n")  # noqa: E731
    cfg, plan = cf.load_config(config), json.loads(Path(collect, "plan.json").read_text())
    ds = pl.build(cfg, out, Path(collect))
    if (bad := ld.leakage_audit(ds)):
        raise SystemExit(f"leakage audit failed: {bad}")
    log(f"dataset {pl.summary(ds)}")
    full, cert = at.prepare(ds), at.prepare(ds, certified_only=True)
    train, val = [r for r in full if r["split"] == "train"], [r for r in full if r["split"] == "validation"]
    nbud = sum(c["src"] == "PLANNER_BUDGET" and c["y"] is not None for r in train if r["key"].startswith("p6_") for c in r["cands"])
    vroot = next((r for r in val if plan["roots"].get(r["key"], {}).get("family") == "F3" and not r["key"].endswith("_ref")), val[0] if val else None)
    runs = PRIMARY + [(n, 0) for n in ABLATIONS]
    weight = lambda n: 1.5 if "-" not in n and "+" not in n else 1.0  # noqa: E731
    res, models, snaps = {}, {}, {}
    for k, (name, seed) in enumerate(runs):
        tag = f"{name}#s{seed}"
        left = CAPS["cpu_s"] - CAPS["reserve_s"] - (time.process_time() - t0)
        budget = left * weight(name) / sum(weight(n) for n, _ in runs[k:])
        if name.endswith("certified-only") and nbud < MIN_BUDGET_NEG:
            res[tag] = {"status": "NOT_ESTABLISHED", "reason": f"{nbud} P6 budget negatives in train < {MIN_BUDGET_NEG}"}
            continue
        if budget < 60:
            res[tag] = {"status": "NOT_RUN", "reason": f"remaining training budget {left:.0f} CPU-s"}
            continue
        torch.manual_seed(seed)
        np.random.seed(seed)
        m, c1 = make(name), time.process_time()
        snap = None
        if seed == 0 and name in ("A", "B", "C") and vroot is not None:
            snaps[name] = []
            snap = lambda ep, mm, _n=name: snaps[_n].append({"epoch": ep, "eval": ar.evaluate(mm, vroot)})  # noqa: E731
        tr = [r for r in cert if r["split"] == "train"] if "certified-only" in name else train
        hist = at.train(m, tr, EPOCHS, budget, log, clip=CLIP, val=val, every=EVERY, patience=PATIENCE, snap=snap)
        torch.save(m.state_dict(), out / f"ckpt_{tag}.pt")
        res[tag] = {"status": "RUN", "history": hist, "params": sum(p.numel() for p in m.parameters()), "cpu_s": round(time.process_time() - c1, 1),
                    "budget_s": round(budget, 1), "grads": at.grad_norms(m, tr),
                    "eval": {"train": [ar.evaluate(m, r) for r in train if r["key"].startswith("p6_")], "validation": [ar.evaluate(m, r) for r in val]}}
        res[tag]["metrics"] = {s: metrics(v) for s, v in res[tag]["eval"].items()}
        models[tag] = m
        log(f"{tag}: {len(hist)} records, best val epoch {hist[-1].get('best_val_epoch')}, {res[tag]['cpu_s']} CPU-s")
    robot = ge.make_robot(cfg)
    ref = res["A#s0"]["eval"]["validation"]
    base = {"geometry": {"validation": metrics(geometry_baseline(cfg, robot, plan, ref))}, "uniform": {"validation": uniform_baseline(ref)}}
    rec = next(x for x in ds["roots"] if vroot is not None and x["audit"]["root"] == vroot["key"])
    samp = {t: sampler_check(models[f"{t}#s0"], vroot, rec) for t in ("A", "B", "C") if f"{t}#s0" in models}
    log("baselines and sampler checks done; unsealing holdout (checkpoints frozen)")
    dsh = pl.build(cfg, out, Path(collect), sealed=True)
    hold = at.prepare(dsh)
    for tag, m in models.items():
        res[tag]["eval"]["holdout"] = [ar.evaluate(m, r) for r in hold]
        res[tag]["metrics"]["holdout"] = metrics(res[tag]["eval"]["holdout"])
    href = res["A#s0"]["eval"]["holdout"]
    base["geometry"]["holdout"], base["uniform"]["holdout"] = metrics(geometry_baseline(cfg, robot, plan, href)), uniform_baseline(href)
    result = {"dataset": pl.summary(ds), "holdout_dataset": pl.summary(dsh), "splits": {r["key"]: r["split"] for r in full + hold},
              "runs": res, "baselines": base, "sampler": samp, "snapshots_root": vroot and vroot["key"], "snapshots": snaps,
              "budget_negatives_p6_train": nbud,
              "config": {"epochs_max": EPOCHS, "every": EVERY, "patience": PATIENCE, "clip": CLIP, "lr": 1e-3, "wd": 1e-4, "width": am.W,
                         "T": am.T, "measure_BC": "uniform", "cap_extra_A": CAP_EXTRA, "seeds": SEEDS, "dino": ld.DINO, "torch_threads": 1},
              "cpu_s": round(time.process_time() - t0, 1), "wall_s": round(time.time() - w0, 1)}
    cf.write_json(out / "results.json", result)
    log("done")
