"""P10c: learner records, matched A / B / C / SF training (full and <= 2-move regimes), retrained factor ablations, D training.

Records: historical P8c learner view (all 190 dev roots relabelled train; P8c test excluded) + P10a 14 (disclosed development, train)
+ P10b train / validation roots (labels from non-blind adjudication only). Blind P10b roots and H9 appearance variants (DINOv2 features
recomputed on a recoloured copy of the SAME observation; geometry identical) are built WITHOUT labels. Selection uses P10b validation
only. Every job runs under the process-tree guard in the 'training' ledger category (p10b_dataset.job).
"""
from __future__ import annotations

import contextlib
import copy
import os
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import abc7_models as a7
import abc7_train as t7
import abc8
import config as cf
import d_blockers as db
import p9_l3 as l3
import unique_factor_keyed as ufk

OUT = Path(os.environ.get("P10C_OUT", "runs/p10c"))            # runs/p10c_prelim = PRELIMINARY batch-1 development pass
UPDATES = 400
PALETTES = {"P1": ((200, 120, 60), (235, 235, 250)), "P2": ((60, 140, 200), (210, 230, 210))}   # (object RGB, support RGB)
ARMS = {"A": (lambda: abc8.build("A"), "poe", False), "SF": (lambda: abc8.build("SF"), "poe", False),
        "K-B": (lambda: abc8.build("B-full"), "unique", True), "K-C": (lambda: abc8.build("C-full"), "unique", True),
        "B-F2": (lambda: a7.ModelBC7("B", "full", order="none").double(), "unique", True),
        "B-F4": (lambda: a7.ModelBC7("B", "full", drop=("pair_omit",)).double(), "unique", True),
        "B-F5s": (lambda: abc8.build("B-structured"), "unique", True), "B-F5q": (lambda: abc8.build("B-quadratic"), "unique", True),
        "B-F3": (lambda: a7.ModelBC7("B", "full", drop=("pair_scene",)).double(), "unique", True),
        "B-V0": (lambda: abc8.build("B-full"), "unique", True), "B-U": (lambda: abc8.build("B-full"), "unique", False)}


def _rd(p):
    return json.loads(Path(p).read_text())


def batches():
    only = os.environ.get("P10C_BATCHES")                          # e.g. "batch1" for the preliminary pass
    return sorted(p.parent for p in Path("runs/p10b").glob("batch*/plan.json") if not only or p.parent.name in only.split(","))


# ------------------------------------------------------------------ learner records
def _record(cfg, dino, k, r, labels, split, source, edit_items, seq_items):
    import learn_data as ld
    import p6_learn as pl
    import zlayout_r4 as z4
    from scene import scene_from_dict
    s = scene_from_dict(r["scene"])
    cands = [{"sequence": it["sequence"], "query": it["z_layout"]["query"].get("zR", f"seq:{k}:empty"), "z_layout": it["z_layout"]} for it in seq_items]
    edits = [{"kind": e.get("edit_kind") or e["kind"], "edit": e["edit"], "eligible": e["eligible"],
              "query": f"scene:{k}:{z4.sig(sorted(e['edit'].items()))}"} for e in edit_items]
    rec = ld.root_record(cfg, *dino, k, s, r["goal_cell"], cands, edits, labels, split, source)
    rec["audit"] |= {"layout": z4.VERSION, "family": r.get("family", r.get("mechanism")), "stratum": r["stratum"], "lineage": k}
    rec["static_checks"] = pl.static_checks(cfg, s, rec["audit"]["ids"], r["seed"])
    return rec, s


def _labels(dec):
    import learn_data as ld
    import operational_teacher as ot
    dec = {q: x for q, x in dec.items() if x["teacher_status"] not in ("NOT_RUN", "FATAL_CONTRADICTION")}
    return ld._labels(dec, ot.loss_masks(dec))


def _recolor(cfg, dino, rec, s, goal, palette):
    import data
    import learn_data as ld
    q = data.make_query(cfg, rec["audit"]["root"], s, "t0", goal, {}, "SAFE_RETREAT", data.PROTOCOL_V2)
    a = data.appearance_anchor(q, ld.IMAGE_PX)
    rgb = a["rgb"].copy()
    obj, sup = palette
    rgb[a["instance"] > 0], rgb[a["instance"] == 0] = obj, sup
    v = copy.deepcopy(rec)
    v["inputs"]["visual"] = ld.visual_features(dino[0], dino[1], dino[2], {**a, "rgb": rgb})
    v["audit"] |= {"appearance": palette, "variant_of": rec["audit"]["root"]}
    return v


def build_records() -> None:
    """Writes dataset_trainval.pt (labelled), dataset_blind.pt and dataset_h9.pt (unlabelled; sealed labels attached at evaluation)."""
    import learn_data as ld
    import p10b_dataset as pd
    cfg, dino = cf.load_config("config.toml"), ld.dino()
    tv, blind, h9 = [], [], []
    for r in torch.load(l3.DEV, weights_only=False)["roots"]:
        r["audit"] |= {"split": "train", "source": "HIST_P8C_VIEW:" + r["audit"]["split"]}
        tv.append(r)
    pa, dec = _rd("runs/p10/p10a/plan.json"), _rd("runs/p10/p10a/decisions.json")
    lab = _labels(dec)
    for k, r in pa["roots"].items():
        rec, _ = _record(cfg, dino, k, r, lab, "train", "P10A_DISCLOSED", r["queries"]["scene_queries"], r["queries"]["pool"])
        tv.append(rec)
    for b in batches():
        plan = _rd(b / "plan.json")
        lab = _labels(pd.records(b))
        dl = pd.d_labels(plan, pd.records(b))
        for k, r in plan["roots"].items():
            items = r["items"]
            ed = [it for it in items if it.get("edit_kind") in ("A", "B", "A+B")]
            sq = [it for it in items if it["kind"] == "ORDERED_TEACHER"]
            sealed = r["split"] == "blind"
            rec, s = _record(cfg, dino, k, r, {} if sealed else lab, r["split"], "P10B", ed, sq)
            ids = rec["audit"]["ids"]
            rec["d"] = {"pool": [ids.index(o) for o in r["parking"]], "roles": [{v: kk for kk, v in r["ids"].items()}[o] for o in r["parking"]]}
            if not sealed:
                x = dl[k]
                rec["d"] |= {"y0": x["y0"], "mask": x["mask"], "type": x["type"], "provenance": x["provenance"], "sets": x["sets"]}
            (blind if sealed else tv).append(rec)
            if r["split"] in ("validation", "blind"):
                for name, pal in PALETTES.items():
                    v = _recolor(cfg, dino, {**rec, "inputs": rec["inputs"]} if sealed else _strip(rec), s, r["goal_cell"], pal)
                    v["audit"]["palette"] = name
                    h9.append(v)
    OUT.mkdir(parents=True, exist_ok=True)
    for name, rs in (("trainval", tv), ("blind", blind), ("h9", h9)):
        torch.save({"version": "P10C_RECORDS_V1", "roots": rs}, OUT / f"dataset_{name}.pt")


def _strip(rec):
    """Unlabelled copy (H9 variants never carry labels; they are scored against the original root's adjudication)."""
    v = copy.deepcopy(rec)
    for c in v["candidates"]:
        c["privileged"] |= {"y": None, "source": None, "z": [None] * len(c["privileged"]["z"]), "cost": None}
    for e in v["edits"]:
        e["privileged"] |= {"y": None, "source": None}
    v.pop("d", None)
    return v


# ------------------------------------------------------------------ prepared data, regimes
def prepared(name):
    ds = torch.load(OUT / f"dataset_{name}.pt", weights_only=False)
    rs = t7.prepare(ds)
    for r, rec in zip(rs, ds["roots"]):
        r["ids"], r["audit"], r["d"] = rec["audit"]["ids"], rec["audit"], rec.get("d")
    return rs


def short(rs):
    """<= 2-move regime: drop every H10 root, every root whose shortest VERIFIED repair needs > 2 moves, and every > 2-move candidate."""
    out = []
    for r in rs:
        ver = [len(c["seq"]) for c in r["cands"] if c["y"] == 1]
        if str(r["audit"].get("family", "")).startswith("H10") or (ver and min(ver) > 2):
            continue
        r = dict(r)
        r["cands"] = [c for c in r["cands"] if len(c["seq"]) <= 2]
        out.append(r)
    return out


def data(regime):
    rs = prepared("trainval")
    tr, va = [r for r in rs if r["split"] == "train"], [r for r in rs if r["split"] == "validation"]
    return (short(tr), short(va)) if regime == "short" else (tr, va)


def _v0(rs):
    for r in rs:
        r["R"].vis = torch.zeros_like(r["R"].vis)
    return rs


def build(arm):
    return ARMS[arm][0]()


def train_job(name, arm, seed, regime, updates=UPDATES) -> None:
    fac, objective, keyed = ARMS[arm]
    tr, va = data(regime)
    if arm == "B-V0":
        tr, va = _v0(tr), _v0(va)
    l3.build = lambda a: fac()
    c0 = time.process_time()
    with ufk.installed() if keyed else contextlib.nullcontext():
        m, rec = l3.fit(arm, seed, updates, tr, va, objective)
    ck = OUT / "ckpt" / f"{name}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ck)
    rec["history"] = [{k: h.get(k) for k in ("update", "total", "nll", "val", "grad_pre", "clipped")} for h in rec["history"]]
    rec |= {"name": name, "arm": arm, "seed": seed, "regime": regime, "keyed": keyed, "train_roots": len(tr), "val_roots": len(va),
            "train_cpu_s": round(time.process_time() - c0, 1), "checkpoint_sha256": l3.sha(ck), "prereg": _rd("runs/p10b/p10b0/P10B_PREREG.json")["hash"]}
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "runs" / f"{name}.json", rec)


def load(name):
    x = _rd(OUT / "runs" / f"{name}.json")
    m = build(x["arm"])
    m.load_state_dict(torch.load(OUT / "ckpt" / f"{name}.pt", weights_only=True))
    m.eval()
    return m, x


# ------------------------------------------------------------------ D training (feasibility on every decided y0; blocker term on complete proof-grounded masks)
def d_items(rs):
    out = []
    for r in rs:
        y0 = (r.get("d") or {}).get("y0")
        if y0 is None:
            y0 = next((c["y"] for c in r["cands"] if not c["seq"]), None)
        if y0 is None:
            continue
        d = r.get("d") or {}
        ok = d.get("mask") is not None and d.get("provenance") == "proof_grounded" and y0 == 0
        out.append((r["R"], d.get("pool", []) if ok else [], y0, d.get("mask") if ok else None))
    return out


def d_val_nll(m, items):
    with torch.no_grad():
        return float(np.mean([float(db.nll(*m(R, pool), y0, b, 1.0)) for R, pool, y0, b in items]))


def train_d(name, pair, seed, lam, regime="full", updates=600) -> None:
    tr, va = data(regime)
    tri, vai = d_items(tr), d_items(va)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = db.ModelD(pair=pair).double()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, hist, c0 = (math.inf, None, -1), [], time.process_time()
    for u in range(updates):
        batch = [tri[i] for i in rng.choice(len(tri), 12, replace=False)]
        opt.zero_grad()
        loss = torch.stack([db.nll(*m(R, pool), y0, b, lam) for R, pool, y0, b in batch]).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite D loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        if (u + 1) % 20 == 0:
            v = d_val_nll(m, vai)
            hist.append({"update": u + 1, "train": float(loss), "val": v})
            if v < best[0]:
                best = (v, copy.deepcopy(m.state_dict()), u + 1)
    m.load_state_dict(best[1])
    ck = OUT / "ckpt" / f"{name}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ck)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "runs" / f"{name}.json", {"name": name, "arm": "D" if pair else "D-pairless", "pair": pair, "seed": seed, "lambda_B": lam,
                                                  "regime": regime, "val_nll": best[0], "best_update": best[2], "history": hist,
                                                  "train_items": len(tri), "train_masks": sum(b is not None for *_, b in tri),
                                                  "val_items": len(vai), "val_masks": sum(b is not None for *_, b in vai),
                                                  "train_cpu_s": round(time.process_time() - c0, 1), "checkpoint_sha256": l3.sha(ck)})
