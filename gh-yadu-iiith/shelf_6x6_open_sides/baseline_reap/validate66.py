#!/usr/bin/env python3
"""Agreement of Shelf66Oracle with the dataset labels.

  (a) every root with a verified minimal sequence: each move (obstacles, then the target), decomposed REAP-style,
      must give a feasible pick at its source and a feasible put at its destination (moves applied in order);
  (b) every root with K >= 1 or NONE: the target's direct move in the start scene must be INFEASIBLE (that is what
      makes it K >= 1 / NONE); K0: feasible.

    python validate66.py data/shelf66_s50
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

from shelf66_oracle import Shelf66Oracle


def main():
    stats = Counter()
    for js in sorted(Path(sys.argv[1]).glob("*.json")):
        side = json.loads(js.read_text())
        t = time.time()
        orc = Shelf66Oracle(side)
        lay = {o["id"]: tuple(o["cell"]) for o in side["objects"]}
        goal = tuple(side["goal_cell"])
        # (b) the direct target move in the start scene
        direct = orc.pick(lay, "t0", lay["t0"]) and orc.put({k: v for k, v in lay.items() if k != "t0"}, "t0", goal)
        expect = side["cls"] == "0"
        stats["direct_agree" if direct == expect else "direct_DISAGREE"] += 1
        msg = [f"direct {'ok' if direct else 'no'} (expect {'ok' if expect else 'no'})"]
        # (a) the verified minimal sequence
        seq = side["minimal_sequence"]
        if seq is not None:
            for o, d in list(seq) + [["t0", list(goal)]]:
                src, d = lay[o], tuple(d)
                p = orc.pick(lay, o, src)
                rest = {k: v for k, v in lay.items() if k != o}
                q = orc.put(rest, o, d)
                lay = {**rest, o: d}
                stats["seq_ok" if p and q else "seq_FAIL"] += 1
                if not (p and q):
                    msg.append(f"{o}{src}->{d}: pick {p} put {q}")
        print(f"{side['problem']} cls {side['cls']} {time.time() - t:.0f}s q={orc.queries}: " + "; ".join(msg), flush=True)
    print("\n", dict(stats))


if __name__ == "__main__":
    main()
