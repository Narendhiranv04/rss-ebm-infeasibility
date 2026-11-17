#!/usr/bin/env python3
"""v3 extension of the library: non-frontal grasp motions (lib66.build_ext_job) for the TOP combos of the given problems.
    python build_ext66.py <problem dir> ... [--top 20] [--part k/n] [--workers 8]
--part k/n: every n-th job starting at k (jobs interleaved, so machines can share). Output: $YADU_LIB_DIR/<combo>/ext_src_r_c.pkl
"""
import argparse, collections, json, multiprocessing as mp, time
from pathlib import Path
from lib66 import LIB_DIR, build_ext_job, ceil_class, cells, combo_name, mover_class

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+"); ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--part", default=None); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    need = collections.Counter()
    for d in a.dirs:
        for f in Path(d).glob("*.json"):
            s = json.loads(f.read_text()); cc = ceil_class(s["objects"])
            for o in s["objects"]: need[(mover_class(o), cc)] += 1
    ranked = [mcc for mcc, _ in need.most_common()][:a.top]
    jobs = [(mc, cc, src) for (mc, cc) in ranked for src in cells()]
    if a.part:
        k, n = map(int, a.part.split("/")); jobs = jobs[k::n]
    todo = [j for j in jobs if not (LIB_DIR / combo_name(j[0], j[1]) / f"ext_src_{j[2][0]}_{j[2][1]}.pkl").exists()]
    print(f"top {a.top} combos, {len(jobs)} jobs in this part, {len(todo)} to build -> {LIB_DIR}", flush=True)
    t0, done = time.time(), 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=10) as pool:
        for name, src, what, dt in pool.imap_unordered(build_ext_job, todo):
            done += 1; el = time.time() - t0
            print(f"{time.strftime('%H:%M:%S')} {done}/{len(todo)} {name} src{src} {what} {dt:.0f}s  eta {el / done * (len(todo) - done) / 3600:.1f} h", flush=True)
    print(f"done {done} jobs in {(time.time() - t0) / 3600:.2f} h")

if __name__ == "__main__":
    main()
