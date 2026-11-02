"""P8c learner view: fresh P8 roots (train / validation; test only when sealed=True) plus ALL historical P4-P7 roots
relabelled split 'train' with source tags (P7 test roots were disclosed in P7 and are never called fresh)."""
from __future__ import annotations

from pathlib import Path

import torch

import p7_learn as p7l

VERSION = "P8_LEARN_VIEW_V1"


def build(cfg, out: Path, collect=Path("runs/p8/collection"), sealed=False) -> dict:
    model, mean, std = p7l.ld.dino()
    hist = []
    if not sealed:
        for f, tag in (("runs/p7/training/dataset.pt", None), ("runs/p7/training/dataset_sealed.pt", "HIST_P7_TEST_DISCLOSED")):
            for r in torch.load(f, weights_only=False)["roots"]:
                src = tag or (r["audit"]["source"] if r["audit"]["source"].startswith("HIST") else "HIST_P7B_" + r["audit"]["split"].upper())
                r["audit"] |= {"split": "train", "source": src}
                hist.append(r)
    roots = p7l._p7_roots(cfg, model, mean, std, Path(collect), sealed)
    for r in roots:
        r["audit"]["source"] = "P8C"
    ds = {"version": VERSION, "dino": list(p7l.ld.DINO), "input_keys": list(p7l.ld.INPUT_KEYS), "roots": hist + roots}
    out.mkdir(parents=True, exist_ok=True)
    torch.save(ds, out / ("dataset_sealed.pt" if sealed else "dataset.pt"))
    return ds
