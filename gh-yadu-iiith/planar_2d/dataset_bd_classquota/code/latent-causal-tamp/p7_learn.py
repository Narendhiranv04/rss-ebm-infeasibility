"""P7b learner-view dataset (no teacher call): P7b roots (Z_LAYOUT_V3_R4) plus historical development data.

Training = P7b train roots + every historical root with provenance (P4 pilot, P5 D3-lite train/validation, P6
train/validation AND the already-examined P6 holdout, all relabelled split 'train', source HIST_*). Selection =
P7b validation roots only. P7b test roots load only with sealed=True after every model choice is frozen.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

import learn_data as ld
import p6_learn as pl
import zlayout_r4 as z4
from scene import scene_from_dict

VERSION = "P7_LEARN_VIEW_V1"


def _p7_roots(cfg, model, mean, std, c, sealed):
    rd = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    plan = rd(c / "plan.json")
    dec = rd(c / "decisions.json") | (rd(c / "sealed_test.json") if sealed else {})
    lab = ld._labels({q: r for q, r in dec.items() if r["teacher_status"] not in ("NOT_RUN", "FATAL_CONTRADICTION")}, rd(c / "loss_masks.json"))
    out = []
    for k, r in plan["roots"].items():
        if (r["split"] == "test") != sealed:
            continue
        s = scene_from_dict(r["scene"])
        if r["queries"] is None:
            cands, edits = [{"sequence": [], "query": f"seq:{k}:empty", "z_layout": z4.build(k, [], [])}], []
        else:
            cands = [{"sequence": p["sequence"], "query": p["z_layout"]["query"].get("zR", f"seq:{k}:empty"), "z_layout": p["z_layout"]}
                     for p in r["queries"]["pool"]]
            edits = [{**e, "query": f"scene:{k}:{z4.sig(sorted(e['edit'].items()))}"} for e in r["queries"]["scene_queries"] if e["kind"] != "nuisance"]
        rec = ld.root_record(cfg, model, mean, std, k, s, r["goal_cell"], cands, edits, lab, r["split"], "P7B")
        rec["audit"] |= {"layout": z4.VERSION, "family": r["family"], "stratum": r["stratum"], "lineage": r["lineage"], "wave": r["wave"]}
        if r["queries"] is not None:
            rec["static_checks"] = pl.static_checks(cfg, s, rec["audit"]["ids"], r["seed"])
        out.append(rec)
    return out


def build(cfg, out: Path, collect=Path("runs/p7/collection"), p6_train=Path("runs/p6/train"), sealed=False) -> dict:
    model, mean, std = ld.dino()
    hist = []
    if not sealed:
        for f, src in ((p6_train / "dataset.pt", None), (p6_train / "dataset_sealed.pt", "HIST_P6_HOLDOUT")):
            for r in torch.load(f, weights_only=False)["roots"]:
                if src is None and r["audit"]["source"] == "P6":
                    src_r = "HIST_P6_" + r["audit"]["split"].upper()
                else:
                    src_r = src or r["audit"]["source"]
                r["audit"] |= {"split": "train", "source": src_r}
                hist.append(r)
    ds = {"version": VERSION, "z_layouts": ["Z_LAYOUT_V2", z4.VERSION], "dino": list(ld.DINO), "input_keys": list(ld.INPUT_KEYS),
          "roots": hist + _p7_roots(cfg, model, mean, std, Path(collect), sealed)}
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ds, out / ("dataset_sealed.pt" if sealed else "dataset.pt"))
    return ds
