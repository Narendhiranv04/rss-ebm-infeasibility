"""P17 data: splits, existential labelling (B ordered repairs; D parking + multi-destination + temporary-target lattices), training
sets, per-family D temporary-target calibration and B policy calibration."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

import config as cf
from p17_eval import OUT, rd, sha, P17_LABEL_BUDGET

# ------------------------------------------------------------------ P17 labelling (B: existential ordered repairs; D: parking + multi-destination lattices)
DATA = OUT / "data"


def _plan(split):
    p = DATA / split / "plan.json"
    return rd(p) if p.exists() else rd(f"runs/p16/data/{split}/plan.json")


def label_roots(split, keys, part, sealed=False, temp=True) -> None:
    """Per root (checkpointed): B catalogue + existential labels (per-sequence budgets, shared exact cache), D lattices under both
    protocols, temporary-target lattices at the original state. Output under runs/p17/data/<split>/ (sealed_* for test splits)."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    import p17_b as pb
    import p17_d as pd
    import p17_domain as dm
    idx, plan = pl.index_v2()[0], _plan(split)
    f = DATA / split / (("sealed_labels" if sealed else "labels") + f"_{part}.json")
    res = rd(f) if f.exists() else {}
    for k in keys:
        if k in res or any(k in rd(g) for g in (DATA / split).glob(("sealed_labels" if sealed else "labels") + "_*.json") if g != f):
            continue
        c0 = time.process_time()
        task = pc.task_from_root(plan["roots"][k], k)
        cache = {}
        fac = lambda c: pm.DetPhysics(task, idx, budget=pb.SEQ_BUDGET, cache=c)  # noqa: E731
        phys = fac(cache)
        cat = pb.catalogue(task, phys, idx)
        lab = pb.label(task, fac, cat, idx)
        lphys = pm.DetPhysics(task, idx, budget=P17_LABEL_BUDGET, cache=cache)
        cells0 = {o.id: o.cell for o in lphys.s0.objects}
        sl = dm.slot_fn(lphys, idx, width=4, origin=cells0)
        md = pd.multi_dest_lattice(lphys, cat["pool"], sl, k=2)
        nat = plan["roots"][k].get("family") == "NAT"
        tl = pd.temp_lattice(lphys, cells0, np.asarray(lphys.s0.robot.q_rad, float), cat["pool"], cat["parking"]) if temp and not nat and len(cat["pool"]) >= 2 else []
        res[k] = {"family": plan["roots"][k].get("family"), "catalogue": {x: cat[x] for x in ("pool", "parking", "movers", "edits", "sequences", "generated", "kept", "omitted")},
                  **lab, "multi_dest": {"statuses": {"+".join(S) or "-": v for S, v in md["statuses"].items()}, "k": md["k"]},
                  "temp_lattices": tl, "protocol": {"labels": "P17_EXISTENTIAL_V1", "seq_budget": pb.SEQ_BUDGET, "domain": dm.DOMAIN_VERSION,
                                                     "layout": pb.LAYOUT, "planner_profile": "P16_FAST"}, "cpu_s": time.process_time() - c0}
        (DATA / split).mkdir(parents=True, exist_ok=True)
        cf.write_json(f, res)


def merged(split, kind="labels") -> dict:
    out = {}
    for g in sorted((DATA / split).glob(f"{kind}_*.json")):
        out |= rd(g)
    return out


# ------------------------------------------------------------------ P17 splits (new seeds; P13 grammars + DENSE via build_plan, REACH via p17_mech)
SEED0 = {"p17dev": 2026260000, "p17train": 2026270000, "p17calib": 2026280000, "p17test": 2026290000}


def build_p17_split(name, counts) -> None:
    """counts {family: n}; REACH uses p17_mech.reach_spec (geometry-first; first collision-free start per seed). Writes DATA/name/plan.json."""
    import p16_common as pc
    import p17_mech as m
    pdm = pc.register()
    fams = sorted(f for f in counts if f != "REACH")
    split = "blind" if name.startswith("p17test") else "dev"
    specs = [{"family": f, "seed": SEED0[name] + 1000 * i + j, "index": len(fams) * j + i, "wave": 0, "batch": 1, "split": split}
             for i, f in enumerate(fams) for j in range(counts[f])]
    sub = DATA / name / "grammar"
    if specs and not (sub / "plan.json").exists():
        pdm.build_plan(sub, specs, max_attempts=600, root_cpu=300.0)
    roots = dict(rd(sub / "plan.json")["roots"]) if specs else {}
    reach, att = {}, 0
    j = 0
    while len(reach) < counts.get("REACH", 0) and att < 50 * max(1, counts.get("REACH", 0)):
        att += 1
        spec = m.reach_spec(np.random.default_rng([SEED0[name] + 900000, j]))
        j += 1
        if spec is None:
            continue
        key = f"p17r_{SEED0[name] + 900000 + j}"
        root = m.build_root(spec, key)
        if root is None:
            continue
        reach[key] = {**root, "family": "REACH", "seed": SEED0[name] + 900000 + j, "split": split, "stratum": spec["stratum"]}
    plan = {"roots": {**roots, **reach}, "reach_attempts": att, "counts": counts, "name": name}
    plan["plan_hash"] = cf.stable_hash({k: v for k, v in plan.items()})
    cf.write_json(DATA / name / "plan.json", plan)


def leakage_p17() -> dict:
    import p14_sweep_index as sx
    import p15_lookup as pl
    fp = {}
    for d in list(Path("runs/p16/data").glob("*/plan.json")) + list(DATA.glob("*/plan.json")):
        for k, r in rd(d)["roots"].items():
            fp.setdefault(json.dumps(r["fingerprint"]), []).append((str(d.parent), k))
    idx = pl.index_v2()[0]
    P = sx._plans()
    idx_fp = {pl.json_fp(P[e["root"]]) for es in idx.by_key.values() for e in es}
    dup = {f: v for f, v in fp.items() if len(v) > 1}
    out = {"roots": sum(len(v) for v in fp.values()), "duplicates": dup, "index_overlap": [f for f in fp if f in idx_fp]}
    cf.write_json(DATA / "LEAKAGE.json", out)
    return out


def label_queue(part, sealed=False) -> None:
    """Claim (split, key) pairs atomically from runs/p17/data/label_queue.json; resumable; one provider set per root."""
    q = rd(DATA / ("sealed_label_queue.json" if sealed else "label_queue.json"))
    claims = DATA / ("sealed_claims" if sealed else "label_claims")
    claims.mkdir(parents=True, exist_ok=True)
    for split, k in q:
        try:
            os.close(os.open(claims / f"{split}__{k}.claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        label_roots(split, [k], part, sealed=sealed)


TRAIN_SPLITS, VAL_SPLITS = ("train", "dev", "calib", "p17train"), ("val",)
D_TRAIN_SPLITS = ("train", "dev", "p17train")                 # D items exclude P16 calib: it is the held-out D calibration evaluation set


def build_training_data() -> None:
    """runs/p17/b/TRAIN.pt: {train: P13 train (P13 protocol, tagged) + P17-labelled P16 train / dev / calib + p17train,
    val: P13 validation + P17-labelled P16 val}; runs/p17/d/D_ITEMS.pt: D items (P13: PARKING_LATTICE_V1 masked;
    P16 / P17 roots: MULTI_DEST_LATTICE_V1)."""
    import torch
    import learn_data as ld
    import p17_b as pb
    import p17_d as pd
    os.environ["P10C_OUT"] = "runs/p13/models"
    p13 = torch.load("runs/p13/models/dataset_trainval.pt", weights_only=False)["roots"]
    for r in p13:
        r["audit"]["protocol"] = "P13_TEACHER_CONSTRUCTOR_CATALOGUE"
    dino = ld.dino()
    recs, labs_all, counts = {}, {}, {}
    for s in TRAIN_SPLITS + VAL_SPLITS:
        labs = merged(s)
        plan = _plan(s)["roots"]
        recs[s] = pb.records(plan, labs, "validation" if s in VAL_SPLITS else "train", dino)
        labs_all[s] = labs
        counts[f"p17_{s}"] = len(labs)
    d = {"train": [r for r in p13 if r["audit"]["split"] == "train"] + [r for s in TRAIN_SPLITS for r in recs[s]],
         "val": [r for r in p13 if r["audit"]["split"] == "validation"] + [r for s in VAL_SPLITS for r in recs[s]],
         "counts": counts | {"p13_train": sum(r["audit"]["split"] == "train" for r in p13), "p13_val": sum(r["audit"]["split"] == "validation" for r in p13)}}
    (OUT / "b").mkdir(parents=True, exist_ok=True)
    torch.save(d, OUT / "b" / "TRAIN.pt")
    items = {"train": [], "val": [], "protocols": {"P13": "PARKING_LATTICE_V1 (NO_CLEARING masked)", "P16/P17 roots": "MULTI_DEST_LATTICE_V1"}}
    for part, src in (("train", [r for r in p13 if r["audit"]["split"] == "train"]), ("val", [r for r in p13 if r["audit"]["split"] == "validation"])):
        for r, rec in zip(pb.prepare(src), src):
            dd = dict(rec.get("d") or {})
            if dd.get("y0") is None:
                y0 = next((c["y"] for c in r["cands"] if not c["seq"]), None)
                dd = {"y0": y0, "pool": []} if y0 is not None else dd
            it = pd.item(r["R"], dd)
            if it:
                items[part].append(it)
    for s in D_TRAIN_SPLITS + VAL_SPLITS:
        part = "val" if s in VAL_SPLITS else "train"
        for r in pb.prepare(recs[s]):
            lab = labs_all[s][r["key"]]
            pool = lab["catalogue"]["pool"]
            st = {tuple(x.split("+")) if x != "-" else (): v for x, v in lab["multi_dest"]["statuses"].items()}
            dlat = pd.lattice_from({S: v for S, v in st.items()}, pool)
            dlat["pool"] = [r["ids"].index(o) for o in pool]
            it = pd.item(r["R"], dlat)
            if it:
                items[part].append(it)
    torch.save(items, OUT / "d" / "D_ITEMS.pt") if (OUT / "d").mkdir(parents=True, exist_ok=True) is None else None
    cf.write_json(OUT / "b" / "TRAIN_COUNTS.json", d["counts"] | {"sha": sha(OUT / "b" / "TRAIN.pt"), "d_items": {k: len(v) for k, v in items.items() if k != "protocols"},
                                                                  "d_masks": {k: sum(b is not None for *_, b in v) for k, v in items.items() if k != "protocols"}})


CAL_FIT, CAL_EVAL = ("p17calib",), ("calib",)          # no D family trains on either (D_TRAIN_SPLITS excludes calib); fit on P17 calib, held-out on P16 calib


def d_calibrate(kinds=("D", "Dp", "Dp16", "Dp17")) -> dict:
    """Per-family temporary-target feasibility calibration (fit on P17 calib roots, evaluated on held-out P16 calib roots; no D family trains on either);
    temporary-target blocker-marginal validation on every decided temporary lattice with a defined mask (both sides)."""
    import p17_d as pd
    plans = {s: _plan(s) for s in CAL_FIT + CAL_EVAL}
    probes = {s: pd.probes_from_labels(merged(s), plans[s]["roots"], s) + pd.probes_from_inter(merged(s, "temp_inter"), s) for s in CAL_FIT + CAL_EVAL}
    out = {}
    for kind in kinds:
        if kind == "Dp17" and not (Path("runs/p17/d/ckpt/P17-Dp-l2.0-s0.pt")).exists():
            continue
        fit = pd.score_probes(kind, [p for s in CAL_FIT for p in probes[s]], plans)
        ev = pd.score_probes(kind, [p for s in CAL_EVAL for p in probes[s]], plans)
        cal = pd.fit_calibration(kind, fit)
        a, b = cal["slope"], cal["bias"]
        held = [(1 / (1 + np.exp(-(a * p["logit_raw"] + b))), p["y"]) for p in ev]
        rec = lambda rows, c: float(np.mean([pp >= 0.5 for pp, y in rows if y == c])) if any(y == c for _, y in rows) else None  # noqa: E731
        cal["held_out"] = {"n": len(held), "positives": sum(y for _, y in held), "feasible_recall": rec(held, 1), "infeasible_recall": rec(held, 0),
                           "brier": float(np.mean([(pp - y) ** 2 for pp, y in held])) if held else None}
        masks = [p for p in fit + ev if p["y"] == 0 and p["type"] == "MINIMAL_SETS" and p["mask"] is not None]
        tp = sum(int(m >= 0.5) and b_ for p in masks for m, b_ in zip(p["marg"], p["mask"]))
        fp = sum(int(m >= 0.5) and not b_ for p in masks for m, b_ in zip(p["marg"], p["mask"]))
        fn = sum((m < 0.5) and b_ for p in masks for m, b_ in zip(p["marg"], p["mask"]))
        cal["temp_blocker_marginals"] = {"lattices_with_mask": len(masks), "f1": 2 * tp / max(1, 2 * tp + fp + fn), "tp": tp, "fp": fp, "fn": fn}
        (OUT / "d" / "calibration").mkdir(parents=True, exist_ok=True)
        cf.write_json(OUT / "d" / "calibration" / f"{kind}.json", cal)
        out[kind] = cal
    cf.write_json(OUT / "d" / "CALIBRATION_SUMMARY.json", out | {"probes": {s: len(v) for s, v in probes.items()}})
    return out


def temp_intermediate(split, keys, part) -> None:
    """Temporary-target lattices at INTERMEDIATE states: for a labelled VERIFIED >= 2-move sequence, apply its first relocation with the
    cheapest certified mode (actual terminal q) and build temporary lattices for the remaining pool objects there."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    import p17_d as pd
    idx, plan, labs = pl.index_v2()[0], _plan(split), merged(split)
    f = DATA / split / f"temp_inter_{part}.json"
    res = rd(f) if f.exists() else {}
    for k in keys:
        if k in res or k not in labs:
            continue
        seqs = [s for s, v in labs[k]["sequences"].items() if v["y"] == 1 and v["len"] >= 2]
        if not seqs:
            res[k] = {"skipped": "no verified >= 2-move sequence"}
            cf.write_json(f, res)
            continue
        task = pc.task_from_root(plan["roots"][k], k)
        phys = pm.DetPhysics(task, idx, budget=P17_LABEL_BUDGET)
        cells0, q0 = {o.id: o.cell for o in phys.s0.objects}, np.asarray(phys.s0.robot.q_rad, float)
        o1, d1 = seqs[0].split(".")[0].split(">")
        outs = [x for x in phys.edges(cells0, q0, o1, int(d1)) if x[0] == pm.V]
        if not outs:
            res[k] = {"skipped": "first move not re-verified"}
            cf.write_json(f, res)
            continue
        e = min(outs, key=lambda x: x[3])
        cells1, q1 = {**cells0, o1: int(d1)}, e[2]
        cat = labs[k]["catalogue"]
        pool = [o for o in cat["pool"] if o != o1]
        dest = {o: cat["parking"][o] for o in pool if cat["parking"][o] not in cells1.values()}
        pool = [o for o in pool if o in dest]
        res[k] = {"sequence": seqs[0], "state": {"cells": cells1, "q": [float(x) for x in q1]}, "dest": dest,
                  "lattices": pd.temp_lattice(phys, cells1, q1, pool, dest) if pool else [], "used": phys.used}
        cf.write_json(f, res)


# ------------------------------------------------------------------ P17 B calibration (p17calib only)
B_LEARNED = ("LB", "LB+G", "LB+G|nocpl", "LB-W", "LB-F2", "LC", "A", "SF", "FULLB", "P13:K-B", "P13:A", "P13:SF", "P13:K-C", "P13:B-F2")
B_SEEDS = (0, 1, 2)


def b_scores(arm, rs, seeds=B_SEEDS) -> dict:
    """Ensemble-mean prior marginals over seeds (no labels); 'X|nocpl' = inference-only coupling toggle."""
    import p16_b as p16b
    import p17_b as pb
    base = arm.split("|")[0]
    sc = []
    for s in seeds:
        m = p16b.load_frozen(base[4:], s) if base.startswith("P13:") else pb.load(base, s)
        if arm.endswith("|nocpl"):
            with pb.coupling_off(m):
                sc.append(pb.score(m, arm, rs))
        else:
            sc.append(pb.score(m, arm, rs))
    ens = {k: [dict(x, p=float(np.mean([s_[k][i]["p"] for s_ in sc]))) for i, x in enumerate(sc[0][k])] for k in sc[0]}
    for v in ens.values():
        for x in v:
            x["u"] = float(np.log(max(x["p"], 1e-9)) - np.log1p(-min(x["p"], 1 - 1e-9)))
    return ens


def b_calibrate_merge() -> dict:
    """Merge per-part policy files into runs/p17/b/POLICIES.json."""
    out = {}
    for f in sorted((OUT / "b").glob("POLICIES_*.json")):
        out |= rd(f)
    cf.write_json(OUT / "b" / "POLICIES.json", out)
    return out


def b_calibrate(arms=B_LEARNED, seeds=B_SEEDS, part=None) -> dict:
    """Per arm: (t0, t) of the no-op-first rule maximizing existential-label-scored correct decisions on p17calib (p16_b.calibrate; UNKNOWN
    choices count as not correct; roots with UNKNOWN y0 excluded). Nothing from test enters. Writes runs/p17/b/POLICIES.json."""
    import p16_b as p16b
    import p17_b as pb
    plan, labs = _plan("p17calib")["roots"], merged("p17calib")
    rs = pb.prepare(pb.records(plan, labs, "calib"))
    y = {r["key"]: dict(r["y"]) for r in rs}
    out = {"_seeds": len(seeds), "_split": "p17calib", "_roots": len(rs)}
    for a in arms:
        ens = b_scores(a, rs, seeds)
        out[a] = p16b.calibrate({k: v for k, v in ens.items() if y[k].get(next(x["query"] for x in v if x["m"] == 0)) is not None}, y)
        cf.write_json(OUT / "b" / (f"POLICIES_{part}.json" if part else "POLICIES.json"), out)
    return out
