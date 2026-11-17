#!/usr/bin/env python3
"""Seeded, class-stratified train / test split of the pushed 6x6 shelf dataset (selected.json, 2,414 roots).

    python make_split66.py [--test-frac 0.2] [--seed 0] [--out split66.json]

Every class (NONE, 0, 1, 2, 3, 4 = ordered K4) is shuffled with its own seeded RNG and its first round(frac * n) roots
go to test. The SAME file must be used for our approach and for every baseline (train on `train`, report on `test`).
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

from roots66_to_pddl import DATASET66


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "split66.json"))
    a = ap.parse_args()
    sel = json.loads((DATASET66 / "selected.json").read_text())
    split = {"dataset": str(DATASET66), "seed": a.seed, "test_frac": a.test_frac, "train": {}, "test": {}}
    for cls, ids in sel["selected"].items():
        ids = sorted(ids)
        random.Random(f"{a.seed}:{cls}").shuffle(ids)
        k = round(a.test_frac * len(ids))
        split["test"][cls], split["train"][cls] = sorted(ids[:k]), sorted(ids[k:])
    split["sha256"] = hashlib.sha256(json.dumps([split["train"], split["test"]], sort_keys=True).encode()).hexdigest()[:16]
    Path(a.out).write_text(json.dumps(split, indent=1))
    for cls in split["test"]:
        print(f"class {cls:>4}: train {len(split['train'][cls]):4d}  test {len(split['test'][cls]):4d}")
    print(f"total: train {sum(map(len, split['train'].values()))}  test {sum(map(len, split['test'].values()))}  "
          f"sha {split['sha256']} -> {a.out}")


if __name__ == "__main__":
    main()
