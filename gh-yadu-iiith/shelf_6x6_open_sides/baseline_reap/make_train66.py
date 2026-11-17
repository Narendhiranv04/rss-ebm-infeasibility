#!/usr/bin/env python3
"""REAP training data from a split file (the test part is never touched).

    python make_train66.py <out dir> [--split split_v4.json] [--n-train 600] [--val-per-class 10] [--seed 0]

<out>/domain.pddl     REAP's confined-shelf domain (copied from data/domain.pddl)
<out>/validation/     --val-per-class roots per class: from the split's 'val' part when it has one (V4), else carved
                      out of 'train' (split66)
<out>/train/          the split's train roots, or --n-train of them, class-proportional (seeded)
Each problem is <root>.pddl + its geometry sidecar <root>.json (roots66_to_pddl.write_problem).
"""
import argparse
import json
import random
import shutil
from pathlib import Path

from roots66_to_pddl import find_root, write_problem

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--val-per-class", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default=str(HERE / "split_v4.json"))
    ap.add_argument("--n-train", type=int, default=0, help="keep only this many train roots, class-proportional (0 = all)")
    a = ap.parse_args()
    out = Path(a.out)
    split = json.loads(Path(a.split).read_text())
    test = {r for ids in split["test"].values() for r in ids}
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "validation").mkdir(parents=True, exist_ok=True)
    shutil.copy(HERE / "data" / "domain.pddl", out / "domain.pddl")
    has_val = "val" in split
    pools = {}
    for cls, ids in split["train"].items():
        ids = sorted(ids)
        assert not test & set(ids), "train and test overlap"
        src_val = sorted(split["val"].get(cls, [])) if has_val else ids
        val = set(random.Random(f"val:{a.seed}:{cls}").sample(src_val, min(a.val_per_class, len(src_val)))) if cls != "4" else set()
        assert not test & val, "validation and test overlap"
        pools[cls] = (sorted(set(ids) - val), sorted(val))
    n_all = sum(len(t) for t, _ in pools.values())
    counts = {}
    for cls, (rest, val) in pools.items():
        if a.n_train:
            k = max(1, round(a.n_train * len(rest) / n_all))
            rest = sorted(random.Random(f"sub:{a.seed}:{cls}").sample(rest, min(k, len(rest))))
        for rid, part in [(r, "train") for r in rest] + [(r, "validation") for r in val]:
            rec = json.loads(find_root(rid).read_text())
            rec["_root_id"] = rid
            write_problem(rec, cls, out / part)
        counts[cls] = (len(rest), len(val))
    for cls, (t, v) in counts.items():
        print(f"class {cls:>4}: train {t:4d}  validation {v:3d}")
    print(f"total: train {sum(t for t, _ in counts.values())}  validation {sum(v for _, v in counts.values())} -> {out}  "
          f"(split {Path(a.split).name} sha {split.get('sha256')})")


if __name__ == "__main__":
    main()
