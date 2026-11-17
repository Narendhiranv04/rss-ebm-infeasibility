#!/usr/bin/env python3
"""Why is the library stricter than the teacher? For moves the teacher allows and the library refuses, classify:
    no_candidate   the library stored no motion at all for (class, ceiling, src, dst)
    arm            every stored motion is blocked by the arm clearance (approach / free move / lift / carry / retreat)
    carry          blocked only by the carry footprint clearance
    pick           the pick filter refused (approach into the source cell)
and, for no_candidate, whether the teacher finds the move with the scene's REAL object in an EMPTY shelf
(rounding / try limit) or only around obstacles.

    python diag_lib.py <problem dir> [--scenes 12] [--q 15]
"""
import argparse
import collections
import json
import random
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("--scenes", type=int, default=12)
    ap.add_argument("--q", type=int, default=15)
    a = ap.parse_args()
    from lib66 import LibOracle, R_PROBE, COLS
    from shelf66_oracle import Shelf66Oracle
    files = sorted(Path(a.problems).glob("*.json"))
    random.Random(1).shuffle(files)
    why = collections.Counter()
    for f in files[:a.scenes]:
        side = json.loads(f.read_text())
        lib, ref = LibOracle(side), Shelf66Oracle(side)
        rng = random.Random(side["problem"])
        lay = {o["id"]: tuple(o["cell"]) for o in side["objects"]}
        allc = [tuple(v) for v in side["cells"].values()]
        free = [c for c in allc if c not in lay.values()]
        for _ in range(a.q):
            o = rng.choice(list(lay)); d = rng.choice(free)
            others = frozenset((k, v) for k, v in lay.items() if k != o)
            if not (ref.pick(lay, o, lay[o]) and ref.move(others, o, lay[o], d)):
                continue
            ent = lib._lib.entries(lib.mc[o], lib.cc)
            obst = lib._obst(lay, o)
            idx = np.array([k for k, _ in obst]) if obst else np.array([], int)
            r = np.array([x for _, x in obst]) if obst else np.array([])
            pk = ent.get(("pick", lay[o]))
            if not lib._lib.feasible(pk, obst):
                why["pick" if pk else "pick_no_candidate"] += 1
                continue
            mv = ent.get(("move", lay[o], d))
            if not mv:
                why["no_candidate"] += 1
                continue
            if lib._lib.feasible(mv, obst):
                why["lib_yes"] += 1
                continue
            arm_block = all((idx.size and np.any(arm[idx] < r - R_PROBE)) for arm, _ in mv)
            carry_only = all((idx.size and np.any(carry[idx] < r)) and not np.any(arm[idx] < r - R_PROBE) for arm, carry in mv)
            why["arm" if arm_block else ("carry" if carry_only else "mixed")] += 1
        print(side["problem"], dict(why), flush=True)
    print("\nteacher-feasible moves the library refuses, by cause:", dict(why))


if __name__ == "__main__":
    main()
