#!/usr/bin/env python3
"""Fill the feasibility table (shelf66_table) of every training problem before REAP trains on it.

Per problem, from its START layout: pick of every object at its cell, and the move of every object to every free cell
(the teacher's full pick-carry-place check). These are the queries a search makes first and keeps re-making in
neighbouring states; their stored trajectories then answer those states without the teacher. Resumable: a problem
whose table file has a 'warm' marker is skipped.

    python warm_table66.py <problem dir> [--workers 29] [--limit N]
Table files: $YADU_TABLE_DIR/<problem>.pkl (default ~/reap_table66).
"""
import argparse
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path


def warm(path):
    from shelf66_table import TableOracle
    side = json.loads(Path(path).read_text())
    t = time.time()
    tab = TableOracle(side)
    try:
        if pickle.loads(tab.path.read_bytes()).get("warm"):
            return side["problem"], "skip", 0, 0.0
    except Exception:
        pass
    lay = {o["id"]: tuple(o["cell"]) for o in side["objects"]}
    free = [tuple(v) for v in side["cells"].values() if tuple(v) not in lay.values()]
    n = 0
    for o in lay:
        others = frozenset((k, v) for k, v in lay.items() if k != o)
        if not tab.pick(lay, o, lay[o]):
            continue                                   # an object that cannot be picked has no moves to store
        for d in free:
            tab.move(others, o, lay[o], d)
            n += 1
    tab.save()
    d = pickle.loads(tab.path.read_bytes()); d["warm"] = True
    tab.path.write_bytes(pickle.dumps(d))
    return side["problem"], "warm", n, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("--workers", type=int, default=29)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    files = sorted(str(f) for f in Path(a.problems).glob("*.json"))
    if a.limit:
        files = files[:a.limit]
    t0, done = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=20) as pool:
        for name, what, n, dt in pool.imap_unordered(warm, files):
            done += 1
            print(f"{time.strftime('%H:%M:%S')} {done}/{len(files)} {name} {what} moves={n} {dt:.0f}s", flush=True)
    print(f"done {done} problems in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
