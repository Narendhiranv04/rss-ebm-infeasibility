"""G4E-ABC proof-of-concept stage: dataset -> train A/B/C (+ ablations) -> evaluate -> sampler -> sealed holdout.

The whole worker runs inside one process-tree guard (teacher.guard: 3,600 CPU-s, 7,200 s wall).
No teacher / planner call happens here. Holdout roots are loaded only after every checkpoint is
frozen and are evaluated exactly once.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

import abc_models as am
import abc_train as at
import config as cf
import g4b
import learn_data as ld
import sampler as sp
import teacher
import zlayout as zl

CAPS = {"cpu_s": 3600.0, "wall_s": 7200.0}
MODELS = ["A", "A-no-order", "B", "B-no-coupling", "C", "C-no-coupling"]
CERT_ONLY = ["A", "B", "C"]
EPOCHS, LAM_D, LAM_Q, L_W = 400, 1.0, 1.0, 0.64


def run(args) -> int:
    out = Path(args.out)
    if (out / "results.json").exists():
        raise SystemExit("G4E results exist; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", "import sys, abc_run; abc_run.worker(sys.argv[1], sys.argv[2])", str(out), args.config],
                      CAPS["cpu_s"], CAPS["wall_s"], out / "heartbeat.txt", 30.0, 0.5, Path("."))
    cf.write_json(out / "guard.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"caps": CAPS})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s", "overshoot_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def _cd(r, seq):
    return sum(float((torch.as_tensor(d, dtype=torch.float64) - r["R"].xy0[i]).norm()) for i, d in seq) / L_W


@torch.no_grad()
def evaluate(m, r) -> dict:
    outs, eff = at.forward(m, r), []
    at.losses(m, r, outs, eff)
    rows = []
    for c, o in zip(r["cands"], outs):
        e = o["E"] if m.kind == "A" else o["E_bar"]
        row = {"query": c["query"], "m": len(c["seq"]), "y": c["y"], "source": c["src"], "p": float(c["_p"]), "energy": float(e),
               "E_static": float(o["E_static"]), "J_pred": float(o["J"]), "J_obs": c["cost"], "C_d": _cd(r, c["seq"])}
        if m.kind != "A":
            pz = torch.exp(-o["E_states"] / am.T - o["lse"])
            top = torch.topk(pz, min(4, len(pz)))
            row |= {"slots": o["slots"], "marg": [float(x) for x in o["marg"]], "z_obs": c["z"], "W_abs_sum": float(o["W"].abs().sum()),
                    "top_states": [[[int(v) for v in o["Z"][i]], float(p)] for p, i in zip(top.values, top.indices)]}
        rows.append(row)
    dec = [x for x in rows if x["y"] is not None]
    pairs = [(a, b) for a in dec for b in dec if a["y"] == 1 and b["y"] == 0]
    keyed = {c["query"]: c for c in r["cands"]}
    order = [{"orders": [a["query"], b["query"]], "y": [a["y"], b["y"]], "p": [a["p"], b["p"]], "energy": [a["energy"], b["energy"]]}
             for a in rows for b in rows if a["query"] < b["query"] and a["m"] >= 2 and
             sorted(keyed[a["query"]]["seq"]) == sorted(keyed[b["query"]]["seq"])]
    elig = sorted([x for x in rows if x["p"] >= 0.5], key=lambda x: (x["m"], x["J_pred"], x["C_d"], x["energy"], x["query"]))
    return {"root": r["key"], "split": r["split"], "cands": rows, "effects": eff, "order_pairs": order,
            "rank_acc": (sum(a["energy"] < b["energy"] for a, b in pairs) / len(pairs)) if pairs else None,
            "selected": elig[0]["query"] if elig else None, "selected_label": elig[0]["y"] if elig else None}


def sampler_check(m, r, rec) -> dict:
    """Validated finite sequence sampler (G4B kernels) on the learned sampling energy of one root."""
    ids = rec["audit"]["ids"]
    cell = lambda xy: int(round((xy[1] - 0.04) / 0.08)) * 8 + int(round((xy[0] - 0.04) / 0.08))  # noqa: E731
    dests = {}
    for c in r["cands"]:
        for i, d in c["seq"]:
            dests.setdefault(ids[i], set()).add(cell(d))
    dests = {o: sorted(v) for o, v in dests.items()}
    cat = [{"kind": e["kind"], "edit": {ids[i]: cell(p) for i, p in e["poses"].items()}, "eligible": e["eligible"]} for e in rec["edits"]]
    cache = {}

    @torch.no_grad()
    def energy(state):
        if state not in cache:
            seq = [(ids.index(o), ((d % 8 + 0.5) * 0.08, (d // 8 + 0.5) * 0.08)) for o, d in state]
            if m.kind == "A":
                o = m.seq(r["R"], seq)
                e = o["E"]
            else:
                o = m.joint(r["R"], seq, list(zl.build(r["key"], list(state), cat)["query"]), r["edits"])
                e = o["E_bar"]
            cache[state] = float(e) + LAM_D * _cd(r, seq) + LAM_Q * float(o["J"])
        return cache[state]
    dom, states = sp.Domain.make(dests, 2), g4b.reference_states(dests, 2)
    cert, pi, _ = g4b.certify(dom, energy, 1.0, states)
    chain, _ = sp.run_chain(dom, (), energy, 1.0, 3000, 0)
    freq = {s: 0 for s in states}
    for s in chain:
        freq[s] += 1
    tv = 0.5 * sum(abs(freq[s] / len(chain) - p) for s, p in zip(states, pi))
    top = sorted(zip(states, pi), key=lambda x: -x[1])[:5]
    return {"root": r["key"], "states": len(states), "pass": cert["pass"], "max_error": cert["max_error"], "tv_chain_3000_vs_exact": tv,
            "top": [[[list(a) for a in s], float(p), energy(s)] for s, p in top], "pi": [[[list(a) for a in s], float(p)] for s, p in zip(states, pi)],
            "chain_freq": [freq[s] / len(chain) for s in states]}


def worker(out, config) -> None:
    import abc_figures as fig
    torch.set_num_threads(1)
    out, t0, w0 = Path(out), time.process_time(), time.time()
    log = lambda msg: open(out / "log.txt", "a").write(f"{time.process_time() - t0:8.1f} {msg}\n")  # noqa: E731
    cfg = cf.load_config(config)
    ds = ld.build(cfg, out)
    if (bad := ld.leakage_audit(ds)):
        raise SystemExit(f"leakage audit failed: {bad}")
    log(f"dataset {ld.summary(ds)}")
    allr, cert = at.prepare(ds), at.prepare(ds, certified_only=True)
    train, val = [r for r in allr if r["split"] == "train"], [r for r in allr if r["split"] == "validation"]
    ctrain = [r for r in cert if r["split"] == "train"]
    runs = [(n, False) for n in MODELS] + [(n, True) for n in CERT_ONLY]
    res, models = {}, {}
    for k, (name, co) in enumerate(runs):
        tag = name + ("+certified-only" if co else "")
        budget = 0.8 * (CAPS["cpu_s"] - 400 - (time.process_time() - t0)) / (len(runs) - k)
        torch.manual_seed(0)
        m = am.build(name)
        hist = at.train(m, ctrain if co else train, EPOCHS, budget, log)
        torch.save(m.state_dict(), out / f"ckpt_{tag}.pt")
        res[tag] = {"history": hist, "params": sum(p.numel() for p in m.parameters()), "grads": at.grad_norms(m, ctrain if co else train),
                    "eval": {"train": [evaluate(m, r) for r in train], "validation": [evaluate(m, r) for r in val]}}
        models[tag] = m
        log(f"{tag}: {len(hist)} epochs, loss {hist[0]['total']:.3f} -> {hist[-1]['total']:.3f}")
    d3 = next(r for r in train if r["key"].startswith("d3_") and not r["key"].endswith("_ref"))
    rec = next(x for x in ds["roots"] if x["audit"]["root"] == d3["key"])
    samp = {t: sampler_check(models[t], d3, rec) for t in ("A", "B", "C")}
    log("sampler checks done; unsealing holdout (checkpoints frozen)")
    dsh = ld.build(cfg, out, sealed=True)
    hold = [r for r in at.prepare(dsh) if r["split"] == "holdout"]
    for tag, m in models.items():
        res[tag]["eval"]["holdout"] = [evaluate(m, r) for r in hold]
    result = {"dataset": ld.summary(ds), "holdout_dataset": ld.summary({"roots": [x for x in dsh["roots"] if x["audit"]["split"] == "holdout"]}),
              "splits": {r["key"]: r["split"] for r in allr} | {r["key"]: "holdout" for r in hold}, "runs": res, "sampler": samp,
              "config": {"epochs": EPOCHS, "lr": 1e-3, "wd": 1e-4, "width": am.W, "T": am.T, "lambda_d": LAM_D, "lambda_q": LAM_Q,
                         "dino": ld.DINO, "torch_threads": 1}, "cpu_s": round(time.process_time() - t0, 1), "wall_s": round(time.time() - w0, 1)}
    cf.write_json(out / "results.json", result)
    fig.figures(out, result, ds)
    log("done")

