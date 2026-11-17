#!/usr/bin/env python3
"""Build the conflict library (lib66) for every class + ceiling the given problems need. Resumable (one file per
class + ceiling + source cell); the most frequent classes first.

    python build_lib66.py <problem dir> [<problem dir> ...] [--workers 29]
Output: $YADU_LIB_DIR/<shape>_s<size>_h<height>_c<ceiling>/src_<r>_<c>.pkl
"""
import argparse
import collections
import json
import multiprocessing as mp
import time
from pathlib import Path

from lib66 import LIB_DIR, build_job, ceil_class, cells, combo_name, mover_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--workers", type=int, default=29)
    ap.add_argument("--shard", default=None, help="k/n: only combos with frequency rank %% n == k (another machine's share)")
    ap.add_argument("--exclude-shard", default=None, help="k/n: every combo EXCEPT that shard")
    ap.add_argument("--reverse", action="store_true", help="least frequent combos first (a helper machine working from the other end)")
    a = ap.parse_args()
    need = collections.Counter()
    for d in a.dirs:
        for f in Path(d).glob("*.json"):
            s = json.loads(f.read_text())
            cc = ceil_class(s["objects"])
            for o in s["objects"]:
                need[(mover_class(o), cc)] += 1
    ranked = [mcc for mcc, _ in need.most_common()]
    if a.shard or a.exclude_shard:
        k, n = map(int, (a.shard or a.exclude_shard).split("/"))
        keep = {mcc for i, mcc in enumerate(ranked) if (i % n == k) == bool(a.shard)}
        ranked = [mcc for mcc in ranked if mcc in keep]
    if a.reverse:
        ranked = ranked[::-1]
    jobs = [(mc, cc, src) for (mc, cc) in ranked for src in (cells()[::-1] if a.reverse else cells())]
    todo = [j for j in jobs if not (LIB_DIR / combo_name(j[0], j[1]) / f"src_{j[2][0]}_{j[2][1]}.pkl").exists()]
    print(f"{len(need)} class+ceiling combos, {len(jobs)} jobs, {len(todo)} to build -> {LIB_DIR}", flush=True)
    t0, done = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=10) as pool:
        for name, src, what, dt in pool.imap_unordered(build_job, todo):
            done += 1
            el = time.time() - t0
            eta = el / done * (len(todo) - done) / 3600
            print(f"{time.strftime('%H:%M:%S')} {done}/{len(todo)} {name} src{src} {what} {dt:.0f}s  eta {eta:.1f} h", flush=True)
    print(f"done {done} jobs in {(time.time() - t0) / 3600:.2f} h")


if __name__ == "__main__":
    main()
