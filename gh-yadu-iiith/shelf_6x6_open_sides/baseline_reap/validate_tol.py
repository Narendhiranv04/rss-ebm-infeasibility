#!/usr/bin/env python3
"""Library vs teacher under several clearance tolerances at once (teacher asked once per query).
    python validate_tol.py <problem dir> --seed 0 --tols 0,0.005,0.01,0.015,0.02,0.03 [--free-mode ignore]"""
import argparse, json, multiprocessing as mp, os, random
from collections import Counter
from pathlib import Path

def run_scene(job):
    path, walks, depth, nq, seed, tols = job
    import lib66
    from lib66 import LibOracle
    from shelf66_oracle import Shelf66Oracle
    side = json.loads(Path(path).read_text())
    lib, ref = LibOracle(side), Shelf66Oracle(side)
    rng = random.Random(f"{seed}:{side['problem']}")
    allc = [tuple(v) for v in side["cells"].values()]
    start = {o["id"]: tuple(o["cell"]) for o in side["objects"]}
    out = Counter()
    for _ in range(walks):
        lay = dict(start)
        for _d in range(depth):
            free = [c for c in allc if c not in lay.values()]
            for _k in range(nq):
                o = rng.choice(list(lay)); d = rng.choice(free)
                others = frozenset((k, v) for k, v in lay.items() if k != o)
                b = ref.pick(lay, o, lay[o]) and ref.move(others, o, lay[o], d)
                for t in tols:
                    lib66.R_TOL = t
                    a = lib.pick(lay, o, lay[o]) and lib.move(others, o, lay[o], d)
                    out[(t, "agree" if a == b else ("no" if b else "FALSE_YES"))] += 1
                out["n"] += 1; out["teacher_yes"] += bool(b)
            moves = [(o, d) for o in lay for d in free]; rng.shuffle(moves)
            for o, d in moves[:10]:
                others = frozenset((k, v) for k, v in lay.items() if k != o)
                if ref.pick(lay, o, lay[o]) and ref.move(others, o, lay[o], d):
                    lay = {**lay, o: d}; break
            else:
                break
    return side["problem"], out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("problems")
    ap.add_argument("--scenes", type=int, default=40); ap.add_argument("--walks", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3); ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--workers", type=int, default=36); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tols", default="0,0.005,0.01,0.015,0.02,0.03"); ap.add_argument("--free-mode", default="ignore")
    ap.add_argument("--out", default="validate_tol.json")
    a = ap.parse_args(); os.environ["YADU_LIB_FREE"] = a.free_mode
    tols = [float(x) for x in a.tols.split(",")]
    files = sorted(Path(a.problems).glob("*.json")); random.Random(a.seed).shuffle(files)
    tot = Counter()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, c in pool.imap_unordered(run_scene, [(str(f), a.walks, a.depth, a.q, a.seed, tols) for f in files[:a.scenes]]):
            tot.update(c); print(name, flush=True)
    n = tot["n"]; rep = dict(queries=n, teacher_yes=tot["teacher_yes"] / n, seed=a.seed, free_mode=a.free_mode, tol={})
    for t in tols:
        ag, fy = tot[(t, "agree")] / n, tot[(t, "FALSE_YES")] / n
        rep["tol"][str(t)] = dict(agree=ag, lib_no_teacher_yes=tot[(t, "no")] / n, false_yes=fy, gate=ag >= .8 and fy <= .03)
    Path(a.out).write_text(json.dumps(rep, indent=1)); print(json.dumps(rep, indent=1))
main() if __name__ == "__main__" else None
