#!/usr/bin/env python3
"""Feasibility table (shelf66_table.TableOracle) against the plain teacher (Shelf66Oracle), on search-like query streams.

For each scene: random walks of legal relocations from the start layout (depth <= DEPTH); at every visited state,
Q random moves (object, free destination) are asked of the table (warm: it keeps everything it learned in this scene)
and of a fresh teacher. Only table HITS can disagree (a miss asks the teacher). Reported per answer source:
    agree / table-no-teacher-yes (missed moves: the table's monotone 'infeasible' rule or a stale plan)
          / table-yes-teacher-no (a stored trajectory is still clear but the teacher's randomised search found nothing)
and the time per query of both.

    python validate_table.py <problem dir> [--scenes 20] [--walks 3] [--depth 4] [--q 6] [--workers 16] [--seed 0]
"""
import argparse
import json
import multiprocessing as mp
import os
import random
import tempfile
import time
from collections import Counter
from pathlib import Path


def run_scene(job):
    path, walks, depth, nq, seed = job
    os.environ["YADU_TABLE_DIR"] = tempfile.mkdtemp(prefix="vt_")       # fresh, private table per scene
    import shelf66_table
    shelf66_table.TABLE_DIR = Path(os.environ["YADU_TABLE_DIR"])
    from shelf66_oracle import Shelf66Oracle
    from shelf66_table import TableOracle
    side = json.loads(Path(path).read_text())
    tab, ref = TableOracle(side), Shelf66Oracle(side)
    rng = random.Random(f"{seed}:{side['problem']}")
    cells_all = [tuple(v) for v in side["cells"].values()]
    start = {o["id"]: tuple(o["cell"]) for o in side["objects"]}
    out = Counter()
    t_tab = t_ref = 0.0
    for _ in range(walks):
        lay = dict(start)
        for _d in range(depth):
            free = [c for c in cells_all if c not in lay.values()]
            for _k in range(nq):
                o = rng.choice(list(lay)); dst = rng.choice(free)
                others = frozenset((k, v) for k, v in lay.items() if k != o)
                before = dict(tab.stats)
                t = time.time(); a = tab.pick(lay, o, lay[o]) and tab.move(others, o, lay[o], dst); t_tab += time.time() - t
                t = time.time(); b = ref.pick(lay, o, lay[o]) and ref.move(others, o, lay[o], dst); t_ref += time.time() - t
                src = "hit" if tab.stats["teacher"] == before["teacher"] else "miss"
                out[f"{src}:{'agree' if a == b else ('table_no_teacher_yes' if b else 'table_yes_teacher_no')}"] += 1
            # step: a random move that the teacher says is feasible (search-like trajectory)
            moves = [(o, d) for o in lay for d in free]
            rng.shuffle(moves)
            for o, d in moves[:10]:
                others = frozenset((k, v) for k, v in lay.items() if k != o)
                if ref.pick(lay, o, lay[o]) and ref.move(others, o, lay[o], d):
                    lay = {**lay, o: d}
                    break
            else:
                break
    out["t_tab"], out["t_ref"] = t_tab, t_ref
    for k, v in tab.stats.items():
        out["stat_" + k] = v
    return side["problem"], out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("--scenes", type=int, default=20)
    ap.add_argument("--walks", type=int, default=3)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--q", type=int, default=6)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    files = sorted(Path(a.problems).glob("*.json"))
    random.Random(a.seed).shuffle(files)
    jobs = [(str(f), a.walks, a.depth, a.q, a.seed) for f in files[:a.scenes]]
    tot = Counter()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, c in pool.imap_unordered(run_scene, jobs):
            tot.update(c)
            print(name, {k: (round(v, 1) if isinstance(v, float) else v) for k, v in c.items() if not k.startswith("stat_")}, flush=True)
    n = sum(v for k, v in tot.items() if ":" in k)
    hits = sum(v for k, v in tot.items() if k.startswith("hit:"))
    print("\nqueries", n, "| table hits", hits, f"({100 * hits / max(n, 1):.1f}%)")
    for k in sorted(k for k in tot if ":" in k):
        print(f"  {k:32s} {tot[k]:6d}")
    print(f"time per query: table {tot['t_tab'] / max(n, 1):.3f} s, teacher {tot['t_ref'] / max(n, 1):.3f} s")
    print("table stats:", {k[5:]: round(v, 1) for k, v in tot.items() if k.startswith("stat_")})


if __name__ == "__main__":
    main()
