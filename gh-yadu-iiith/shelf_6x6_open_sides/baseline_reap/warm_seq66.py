#!/usr/bin/env python3
"""Second warm-up pass: the dataset's own relocation sequences, replayed through the feasibility table.

warm_table66.py fills each scene's table from its START layout. The states a successful repair passes through (one or
two obstacles already moved) are not covered there, and they are the ones REAP's searches must get through to learn.
The dataset records them: per root the B sequences (minimal repair + variants) and, for K3 in Narendhiran's V4
manifest, all 6 relocation orders (legacy_orders). Their recorded outcomes came from CHAINED execution (the arm starts
where the last move ended); REAP's checks plan every move from the ready pose. So the outcomes are not copied: each
sequence is replayed from the start layout and every step is asked of the table (teacher on a miss, REAP's rules),
then the target's move to its goal. A step the teacher refuses ends that sequence.

    python warm_seq66.py <problem dir> [--manifest FINAL_3D_SELECTION_MANIFEST.json] [--workers 29]
"""
import argparse
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path

from roots66_to_pddl import find_root


def sequences(rid, manifest_rows):
    rec = json.loads(find_root(rid).read_text())
    seqs = [b["sequence"] for b in rec["labels"].get("B", []) if b.get("sequence") is not None]
    for row in manifest_rows.get(rid, ()):
        seqs += [o["order"] for o in row.get("legacy_orders", ())]
    out, seen = [], set()
    for s in seqs:
        key = tuple((o, tuple(d)) for o, d in s)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def warm(job):
    path, seqs = job
    from shelf66_table import TableOracle
    side = json.loads(Path(path).read_text())
    t = time.time()
    tab = TableOracle(side)
    try:
        if pickle.loads(tab.path.read_bytes()).get("warm_seq"):
            return side["problem"], "skip", 0, 0, 0.0
    except Exception:
        pass
    start = {o["id"]: tuple(o["cell"]) for o in side["objects"]}
    goal = tuple(side["goal_cell"])
    n = ok = 0
    for seq in seqs:
        lay = dict(start)
        for o, d in list(seq) + [("t0", goal)]:
            if o not in lay or d in lay.values():
                break
            others = frozenset((k, v) for k, v in lay.items() if k != o)
            n += 1
            if not (tab.pick(lay, o, lay[o]) and tab.move(others, o, lay[o], d)):
                break
            lay = {**lay, o: d}
        else:
            ok += 1
    tab.save()
    d = pickle.loads(tab.path.read_bytes()); d["warm_seq"] = True
    tab.path.write_bytes(pickle.dumps(d))
    return side["problem"], "warm", n, ok, time.time() - t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("--manifest", default=None, help="V4 FINAL_3D_SELECTION_MANIFEST.json (adds K3 legacy_orders)")
    ap.add_argument("--workers", type=int, default=29)
    a = ap.parse_args()
    rows = {}
    if a.manifest:
        for r in json.loads(Path(a.manifest).read_text())["rows"]:
            rows.setdefault(r["layout_root_id"], []).append(r)
    jobs = []
    for f in sorted(Path(a.problems).glob("*.json")):
        rid = json.loads(f.read_text())["problem"]
        jobs.append((str(f), sequences(rid, rows)))
    print(f"{len(jobs)} problems, {sum(len(s) for _, s in jobs)} sequences", flush=True)
    t0, done = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=20) as pool:
        for name, what, n, ok, dt in pool.imap_unordered(warm, jobs):
            done += 1
            print(f"{time.strftime('%H:%M:%S')} {done}/{len(jobs)} {name} {what} steps={n} sequences_ok={ok} {dt:.0f}s", flush=True)
    print(f"done {done} problems in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
