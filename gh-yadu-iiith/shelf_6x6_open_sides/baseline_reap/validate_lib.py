#!/usr/bin/env python3
"""Library (lib66.LibOracle) against the plain teacher (Shelf66Oracle) on search-like query streams + dataset labels.

Per scene: random walks of teacher-feasible relocations from the start (depth <= DEPTH); at every state Q random
(object, free destination) moves are asked of both, as REAP asks them (pick at the source, then the move).
    agree / lib_no_teacher_yes (the library is stricter: rounded sizes, lower ceiling, no detours)
          / lib_yes_teacher_no (the library allows a move the teacher refuses -- the dangerous direction)
Also the dataset label check of validate66 (direct target move feasible exactly for K0; every minimal-sequence move).
Writes a JSON report; exit code 2 if the gate fails (lib_yes_teacher_no > --max-false-yes or agreement < --min-agree).

    python validate_lib.py <problem dir> [--scenes 40] [--walks 2] [--depth 3] [--q 6] [--workers 29] [--out report.json]
"""
import argparse
import json
import multiprocessing as mp
import random
import sys
import time
from collections import Counter
from pathlib import Path


def run_scene(job):
    path, walks, depth, nq, seed = job
    from lib66 import LibOracle
    from shelf66_oracle import Shelf66Oracle
    side = json.loads(Path(path).read_text())
    lib, ref = LibOracle(side), Shelf66Oracle(side)
    rng = random.Random(f"{seed}:{side['problem']}")
    allc = [tuple(v) for v in side["cells"].values()]
    start = {o["id"]: tuple(o["cell"]) for o in side["objects"]}
    out = Counter()
    t_lib = t_ref = 0.0

    def ask(o, lay, dst):
        nonlocal t_lib, t_ref
        others = frozenset((k, v) for k, v in lay.items() if k != o)
        t = time.time(); a = lib.pick(lay, o, lay[o]) and lib.move(others, o, lay[o], dst); t_lib += time.time() - t
        t = time.time(); b = ref.pick(lay, o, lay[o]) and ref.move(others, o, lay[o], dst); t_ref += time.time() - t
        return a, b

    for _ in range(walks):
        lay = dict(start)
        for _d in range(depth):
            free = [c for c in allc if c not in lay.values()]
            for _k in range(nq):
                o = rng.choice(list(lay)); d = rng.choice(free)
                a, b = ask(o, lay, d)
                out["agree" if a == b else ("lib_no_teacher_yes" if b else "lib_yes_teacher_no")] += 1
                out["teacher_yes"] += b
            moves = [(o, d) for o in lay for d in free]
            rng.shuffle(moves)
            for o, d in moves[:10]:
                others = frozenset((k, v) for k, v in lay.items() if k != o)
                if ref.pick(lay, o, lay[o]) and ref.move(others, o, lay[o], d):
                    lay = {**lay, o: d}
                    break
            else:
                break
    # dataset labels: the direct target move, and the minimal sequence
    lay = dict(start); goal = tuple(side["goal_cell"])
    a, _ = ask("t0", lay, goal) if goal not in lay.values() else (False, None)
    out["label_direct_ok" if a == (side["cls"] == "0") else "label_direct_bad"] += 1
    if side.get("minimal_sequence") is not None:
        for o, d in list(side["minimal_sequence"]) + [["t0", list(goal)]]:
            d = tuple(d)
            if o not in lay or d in lay.values():
                break
            others = frozenset((k, v) for k, v in lay.items() if k != o)
            ok = lib.pick(lay, o, lay[o]) and lib.move(others, o, lay[o], d)
            out["label_seq_ok" if ok else "label_seq_bad"] += 1
            lay = {**lay, o: d}
    out["t_lib"], out["t_ref"], out["lib_fallback"] = t_lib, t_ref, lib.lib_miss
    return side["problem"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--walks", type=int, default=2)
    ap.add_argument("--depth", type=int, default=3)
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--workers", type=int, default=29)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="validate_lib_report.json")
    ap.add_argument("--max-false-yes", type=float, default=0.03, help="gate: max fraction lib-yes / teacher-no")
    ap.add_argument("--min-agree", type=float, default=0.80, help="gate: min agreement")
    ap.add_argument("--free-mode", default=None, help="lib66 FREE_MODE for this validation: strict | tail | ignore")
    a = ap.parse_args()
    import os
    if a.free_mode:
        os.environ["YADU_LIB_FREE"] = a.free_mode            # inherited by the spawned workers before they import lib66
    files = sorted(Path(a.problems).glob("*.json"))
    random.Random(a.seed).shuffle(files)
    jobs = [(str(f), a.walks, a.depth, a.q, a.seed) for f in files[:a.scenes]]
    tot = Counter()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, c in pool.imap_unordered(run_scene, jobs):
            tot.update(c)
            print(name, dict(c), flush=True)
    n = tot["agree"] + tot["lib_no_teacher_yes"] + tot["lib_yes_teacher_no"]
    rep = dict(queries=n, agree=tot["agree"] / max(n, 1), lib_no_teacher_yes=tot["lib_no_teacher_yes"] / max(n, 1),
               lib_yes_teacher_no=tot["lib_yes_teacher_no"] / max(n, 1), teacher_yes=tot["teacher_yes"] / max(n, 1),
               label_direct_ok=tot["label_direct_ok"], label_direct_bad=tot["label_direct_bad"],
               label_seq_ok=tot["label_seq_ok"], label_seq_bad=tot["label_seq_bad"],
               s_per_query_lib=tot["t_lib"] / max(n, 1), s_per_query_teacher=tot["t_ref"] / max(n, 1),
               lib_fallback_queries=tot["lib_fallback"], free_mode=a.free_mode or "tail")
    rep["gate_pass"] = rep["lib_yes_teacher_no"] <= a.max_false_yes and rep["agree"] >= a.min_agree
    Path(a.out).write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    sys.exit(0 if rep["gate_pass"] else 2)


if __name__ == "__main__":
    main()
