"""R4: authentic 3D X/Y intake + bounded, DEVELOPMENT-ONLY one-seed smoke fits of the matched D / E / DIRECT-GEO / G arms (GPU 0, CPU 7).

    P="env CUDA_VISIBLE_DEVICES=0 taskset -c 7 nice -n 10 venv/bin/python"
    $P src/xy3d_train.py intake --sidecar SIDECAR.json --out runs/XY3D_REAL/intake.json          (CPU-light checks only; no fitting)
    $P src/xy3d_train.py smoke  --sidecar SIDECAR.json --out runs/XY3D_REAL/smoke_v1            (only if the PREDECLARED gate passes)

Data: BD3DS66_X_V1 / Y_REPLAY_V1 sidecar rows only (X1 verified, X0 reconfirmed; Y1 replay-verified, Y0 archived operational negatives);
planning-only positives and unattempted queries are MISSING (never labels).  Splits: inherited 3D cohort split; TRAIN fits, VAL early-stops /
reports; TEST roots are refused.  Model / loss: xy3d.XYNet + XYNet.loss (exact root-mean observed-data NLL, kappa SYM; DIRECT success BCE);
no travel labels yet (lambda_J term absent).  bxy_p11.TRAIN optimizer (AdamW 1e-3, wd 1e-4, clip 10, 12 roots/batch, eval every 20, patience
15, <= 3000 updates), float32 on GPU.  A smoke is NOT a D/E/DIRECT/G result; no superiority claim."""
from __future__ import annotations

import argparse, copy, json, os, sys, time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402

GATE = {"declared": "before any fit; written to <out>/GATE.json with the counts it was evaluated on",
        "train_x_labels_min": 300, "train_x0_min": 25, "train_x1_min": 100, "train_paired_cells_min_each": 5,
        "val_x_labels_min": 40, "val_roots_min": 10, "test_roots_used": 0}
TRAIN = {"lr": 1e-3, "wd": 1e-4, "clip": 10.0, "batch_roots": 12, "max_updates": 3000, "eval_every": 20, "patience": 15, "seed": 0}


def views(sidecar, splits):
    import xy3d as XY, xy3d_data as XD
    sc = json.loads(Path(sidecar).read_text())
    byroot = {}
    for r in sc["x_rows"]:
        byroot.setdefault(r["root_id"], {"x": [], "y": []})["x"].append(r)
    for r in sc["y_rows"]:
        if r["root_id"] in byroot:
            byroot[r["root_id"]]["y"].append(r)
    out, qc = {}, XD.qc_roots()
    for rid, d in sorted(byroot.items()):
        sp = qc.get(rid)
        if sp not in splits:
            continue
        raw = XD.load(rid)
        cells0 = {k: tuple(v) for k, v in raw["initial_cells"].items()}
        parse_y = lambda key: {p.split("@")[0]: tuple(map(int, p.split("@")[1].split(","))) for p in key.split("|")}  # noqa: E731
        y_edits = [{o: c for o, c in parse_y(y["y_key"]).items() if c != cells0[o]} for y in d["y"]]
        x_seqs = [[(m.split(">")[0], tuple(map(int, m.split(">")[1].split(",")))) for m in x["x_key"].split(".")] for x in d["x"]]
        v, ym, xm, ids = XD.view3d(raw, extra_edits=y_edits, extra_seqs=x_seqs, use_history=False)
        lab = XY.labels_for(v, ym, xm, obs_y={i: y["y"] for i, y in enumerate(d["y"])}, obs_x={i: x["x"] for i, x in enumerate(d["x"])})
        out[rid] = (sp, v, lab)
    return out


def counts(V):
    c = {"roots": {}, "x": {}, "x0": {}, "x1": {}, "y0": {}, "y1": {}, "cells": {}}
    for rid, (sp, v, lab) in V.items():
        c["roots"][sp] = c["roots"].get(sp, 0) + 1
        for q, x in lab["obs_x"].items():
            c["x"][sp] = c["x"].get(sp, 0) + 1
            c[f"x{x}"][sp] = c[f"x{x}"].get(sp, 0) + 1
            y = lab["obs_y"].get(v["parent"][q])
            if y is not None:
                k = f"{sp}|X{x}Y{y}"
                c["cells"][k] = c["cells"].get(k, 0) + 1
        for s, y in lab["obs_y"].items():
            c[f"y{y}"][sp] = c[f"y{y}"].get(sp, 0) + 1
    return c


def gate(c):
    g = GATE
    checks = {"train_x_labels": c["x"].get("train", 0) >= g["train_x_labels_min"], "train_x0": c["x0"].get("train", 0) >= g["train_x0_min"],
              "train_x1": c["x1"].get("train", 0) >= g["train_x1_min"],
              "train_paired_cells": all(c["cells"].get(f"train|X{a}Y{b}", 0) >= g["train_paired_cells_min_each"] for a in (0, 1) for b in (0, 1)),
              "val_x_labels": c["x"].get("val", 0) >= g["val_x_labels_min"], "val_roots": c["roots"].get("val", 0) >= g["val_roots_min"],
              "no_test_roots": c["roots"].get("test", 0) == 0}
    return {"checks": checks, "pass": all(checks.values())}


def cmd_intake(a):
    V = views(a.sidecar, ("train", "val"))
    c = counts(V)
    res = {"sidecar": a.sidecar, "counts": c, "gate": GATE, "gate_result": gate(c), "utc": time.strftime("%FT%TZ", time.gmtime())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=1))


def evaluate(m, B, L, XY):
    import torch
    with torch.no_grad():
        nll = float(m.loss(B, L, lam=0.0, kappa=0.0)[1]["nll"])
        P = m.predict(B)
        sq = L["succ_q"]
        dec = sq >= 0
        ps = P["psucc"].clamp(1e-6, 1 - 1e-6)
        succ = float(-(sq[dec] * ps[dec].log() + (1 - sq[dec]) * (1 - ps[dec]).log()).mean()) if dec.any() else None
        out = {"nll_objective": nll, "success_logloss": succ, "success_events": int(dec.sum())}
        if "px" in P:
            xo = L["x_obs"] >= 0
            px = P["px"].clamp(1e-6, 1 - 1e-6)
            out["x_marginal_logloss"] = float(-(L["x_obs"][xo] * px[xo].log() + (1 - L["x_obs"][xo]) * (1 - px[xo]).log()).mean()) if xo.any() else None
            yo = L["y_obs"] >= 0
            py = P["py"].clamp(1e-6, 1 - 1e-6)
            out["y_logloss"] = float(-(L["y_obs"][yo] * py[yo].log() + (1 - L["y_obs"][yo]) * (1 - py[yo]).log()).mean()) if yo.any() else None
    return out


def cmd_smoke(a):
    import torch
    import guard, train as TR, xy3d as XY
    out = Path(a.out)
    if (out / "SMOKE.json").exists():
        raise SystemExit("REFUSED: smoke output exists")
    out.mkdir(parents=True, exist_ok=True)
    V = views(a.sidecar, ("train", "val"))
    c = counts(V)
    g = gate(c)
    (out / "GATE.json").write_text(json.dumps({"gate": GATE, "counts": c, "result": g, "utc": time.strftime("%FT%TZ", time.gmtime())}, indent=1))
    if not g["pass"]:
        print(json.dumps({"verdict": "TRAINING_BLOCKED_INSUFFICIENT_PAIRED_EVIDENCE", **g})); return
    _, dev = TR.preflight({}, guard.load_cfg(a.guard_config), False)
    wd = TR.start_watchdog(out, a.guard_config)
    try:
        tr = [(v, l) for sp, v, l in V.values() if sp == "train"]
        va = [(v, l) for sp, v, l in V.values() if sp == "val"]
        Bv, Lv = XY.collate([v for v, _ in va], [l for _, l in va], dev, torch.float32)
        res = {"label": "DEVELOPMENT-ONLY ONE-SEED SMOKE on authentic BD3DS66 labels; not a D/E/DIRECT/G result", "train": TRAIN, "counts": c, "arms": {}}
        for arm in XY.ARMS:
            torch.manual_seed(TRAIN["seed"]); rng = np.random.default_rng(TRAIN["seed"])
            m = XY.XYNet(arm, 5, 14).to(dev, torch.float32)
            opt = torch.optim.AdamW(m.parameters(), lr=TRAIN["lr"], weight_decay=TRAIN["wd"])
            best, state, stale, hist, t0 = float("inf"), None, 0, [], time.time()
            for u in range(TRAIN["max_updates"]):
                idx = rng.choice(len(tr), size=min(TRAIN["batch_roots"], len(tr)), replace=False)
                B, L = XY.collate([tr[i][0] for i in idx], [tr[i][1] for i in idx], dev, torch.float32)
                loss, _ = m.loss(B, L, lam=0.0)
                opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), TRAIN["clip"]); opt.step()
                if (u + 1) % TRAIN["eval_every"] == 0:
                    e = evaluate(m, Bv, Lv, XY)["nll_objective"]
                    hist.append({"update": u + 1, "val_nll": e})
                    if e < best * (1 - 1e-3):
                        best, state, stale = e, copy.deepcopy(m.state_dict()), 0
                    else:
                        stale += 1
                    if stale >= TRAIN["patience"] or TR.stopped(out):
                        break
            m.load_state_dict(state)
            torch.save({"state": state, "arm": arm, "train": TRAIN}, out / f"{arm}_best.pt")
            res["arms"][arm] = {"val": evaluate(m, Bv, Lv, XY), "best_update": hist[[h["val_nll"] for h in hist].index(best)]["update"], "updates": hist[-1]["update"],
                                "wall_s": round(time.time() - t0, 1), "params": sum(p.numel() for p in m.parameters())}
            print(json.dumps({arm: res["arms"][arm]}), flush=True)
        res.update(device=torch.cuda.get_device_name(0), peak_vram_mb=round(torch.cuda.max_memory_allocated() / 2**20, 1), affinity=sorted(os.sched_getaffinity(0)),
                   watchdog_stop=(out / "STOP_RESOURCE").exists())
        (out / "SMOKE.json").write_text(json.dumps(res, indent=1))
    finally:
        wd.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["intake", "smoke"]); ap.add_argument("--sidecar", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--guard-config")
    a = ap.parse_args()
    {"intake": cmd_intake, "smoke": cmd_smoke}[a.cmd](a)
