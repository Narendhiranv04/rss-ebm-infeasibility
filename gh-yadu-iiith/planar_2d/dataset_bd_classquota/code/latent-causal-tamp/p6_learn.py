"""P6.4: learner-view dataset for A/B/C on the P6 roots (Z_LAYOUT_V3_R4) plus compatible historical data.

Same two namespaces as learn_data (INPUTS learner-visible, PRIVILEGED targets only). Historical
development data (P4 pilot, P5 D3-lite train/validation roots, V2 layouts) is used for TRAINING only
and never enters the P6 validation or holdout sets; P5 holdout roots are never loaded. Each P6
obstructed root also carries balanced analytic static checks (2 invalid + 2 valid placements that
are NOT teacher queries): their unary/pair labels come from proposals.static_endpoints only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

import learn_data as ld
import zlayout as zl
import zlayout_r4 as z4
from scene import scene_from_dict

VERSION = "P6_LEARN_VIEW_V1"


def static_checks(cfg, scene, ids, seed) -> list:
    """2 invalid (onto another object's cell) + 2 valid (onto a free cell) single placements, seeded."""
    rng = np.random.default_rng(seed)
    occ = {o.cell: o.id for o in scene.objects}
    movers = [o.id for o in scene.objects if o.kind == "obstacle"]
    free = [c for c in range(64) if c not in occ]
    out, lat = [], scene.lattice
    for want, pool in ((0, sorted(occ)), (1, free)):
        got = 0
        for _ in range(50):
            o, c = movers[int(rng.integers(len(movers)))], pool[int(rng.integers(len(pool)))]
            if occ.get(c) == o:
                continue
            st = ld._static(scene, {o: c}, ids, cfg)
            ok = int(all(v != 0 for v in st["unary"] if v is not None) and all(v != 0 for v in st["pair"].values() if v is not None))
            if ok == want:
                out.append({"actions": [(ids.index(o), list(lat.center(c)))], "unary": st["unary"], "pair": st["pair"], "valid": ok})
                got += 1
            if got == 2:
                break
    return out


def build(cfg, out: Path, p6=Path("runs/p6/collect"), sealed=False) -> dict:
    """Historical train data + P6 train/validation roots (+ P6 holdout only when sealed=True)."""
    rd = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    model, mean, std = ld.dino()
    hist = ld.build(cfg, out / "hist") if not sealed else {"roots": []}   # P4 + P5 train/validation, never P5 holdout
    for r in hist["roots"]:
        r["audit"] |= {"split": "train", "source": "HIST_" + r["audit"]["source"], "layout": zl.VERSION}
    plan = rd(p6 / "plan.json")
    dec = rd(p6 / "decisions.json") | (rd(p6 / "sealed_holdout.json") if sealed else {})
    lab = ld._labels({q: r for q, r in dec.items() if r["teacher_status"] not in ("NOT_RUN", "FATAL_CONTRADICTION")}, rd(p6 / "loss_masks.json"))
    roots = []
    for k, r in plan["roots"].items():
        if (r["split"] == "holdout") != sealed:
            continue
        s = scene_from_dict(r["scene"])
        if r["queries"] is None:
            cands, edits = [{"sequence": [], "query": f"seq:{k}:empty", "z_layout": z4.build(k, [], [])}], []
        else:
            cands = [{"sequence": p["sequence"], "query": p["z_layout"]["query"].get("zR", f"seq:{k}:empty"), "z_layout": p["z_layout"]}
                     for p in r["queries"]["pool"]]
            edits = [{**c, "query": f"scene:{k}:{z4.sig(sorted(c['edit'].items()))}"} for c in r["queries"]["scene_queries"] if c["kind"] != "nuisance"]
        rec = ld.root_record(cfg, model, mean, std, k, s, r["goal_cell"], cands, edits, lab, r["split"], "P6")
        rec["audit"] |= {"layout": z4.VERSION, "family": r["family"], "stratum": r["stratum"], "lineage": r["lineage"]}
        if r["queries"] is not None:
            rec["static_checks"] = static_checks(cfg, s, rec["audit"]["ids"], r["seed"])
        roots.append(rec)
    ds = {"version": VERSION, "z_layouts": [zl.VERSION, z4.VERSION], "dino": list(ld.DINO), "input_keys": list(ld.INPUT_KEYS),
          "roots": hist["roots"] + roots}
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ds, out / ("dataset_sealed.pt" if sealed else "dataset.pt"))
    return ds


def summary(ds) -> dict:
    base = ld.summary(ds)
    by = {}
    for r in ds["roots"]:
        a = r["audit"]
        x = by.setdefault(f"{a['source']}:{a['split']}", {"roots": 0, "candidates": 0, "decided": 0, "complete_z": 0})
        x["roots"] += 1
        x["candidates"] += len(r["candidates"])
        x["decided"] += sum(c["privileged"]["y"] is not None for c in r["candidates"])
        x["complete_z"] += sum(all(z is not None for z in c["privileged"]["z"]) for c in r["candidates"])
    return base | {"by_source_split": by, "static_checks": sum(len(r.get("static_checks", [])) for r in ds["roots"]),
                   "max_active_slots": max(len(c["z_slots"]) for r in ds["roots"] for c in r["candidates"])}
