#!/usr/bin/env python3
"""Why does library v2 refuse moves the teacher allows?  (mode: free move ignored, as the better v2 validation mode)

For random moves in training scenes (the validation query stream): the teacher says yes AND the library says no =>
classify the cause from the stored data, and record WHICH grasp the teacher used in that layout (azimuth, how many
grasps it tried before one worked) for refused vs accepted moves.

    python diag_lib2.py <problem dir> [--scenes 36] [--q 14] [--workers 36] [--out diag_lib2.json]
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
import random
from pathlib import Path


def scene(job):
    path, nq, seed = job
    import numpy as np
    from lib66 import LibOracle, R_PROBE, COLS
    from shelf66_oracle import Shelf66Oracle
    side = json.loads(Path(path).read_text())
    lib, ref = LibOracle(side), Shelf66Oracle(side)
    rng = random.Random(f"d:{seed}:{side['problem']}")
    allc = [tuple(v) for v in side["cells"].values()]
    lay = {o["id"]: tuple(o["cell"]) for o in side["objects"]}
    free = [c for c in allc if c not in lay.values()]
    rows = []
    for _ in range(nq):
        o = rng.choice(list(lay)); d = rng.choice(free)
        others = frozenset((k, v) for k, v in lay.items() if k != o)
        if not (ref.pick(lay, o, lay[o]) and ref.move(others, o, lay[o], d)):
            continue                                              # teacher says no: not interesting here
        a_lib = lib.pick(lay, o, lay[o]) and lib.move(others, o, lay[o], d)
        # which grasp did the teacher use in this layout?
        ref._settle({**dict(others), o: lay[o]})
        plan, info = ref.A.query(ref.e, o, d)
        az = info.get("azimuth"); tried = info.get("tried")
        row = dict(lib_yes=bool(a_lib), az=az, tried=tried, cause=None, near=None, n_cand=None)
        if not a_lib:
            ent = lib._lib.entries(lib.mc[o], lib.cc)
            obst = lib._obst(lay, o)
            idx = np.array([k for k, _ in obst], int); r = np.array([x for _, x in obst])
            pk, mv = ent.get(("pick", lay[o])), ent.get(("move", lay[o], d))
            if not pk:
                row["cause"] = "pick: nothing stored (object cannot be picked in an EMPTY shelf)"
            elif not mv:
                row["cause"] = "move: nothing stored (no motion found in an EMPTY shelf)"
            else:
                row["n_cand"] = len(mv)
                fails = []
                for m in mv:
                    a = bool(idx.size and np.any(m[0, idx] < r - R_PROBE))
                    c = bool(idx.size and np.any(m[3, idx] < r))
                    fails.append((a, c))
                if all(a for a, c in fails) and not any(c for a, c in fails): row["cause"] = "arm clearance (approach/lift/lower/retreat) blocked in every stored motion"
                elif all(c for a, c in fails) and not any(a for a, c in fails): row["cause"] = "carry footprint blocked in every stored motion"
                else: row["cause"] = "mixed (some motions blocked by arm, some by carry)"
                # where are the blockers relative to the source / destination cell?
                near = set()
                for m in mv:
                    for j, (k, rad) in enumerate(obst):
                        if m[0, k] < rad - R_PROBE or m[3, k] < rad:
                            rr, cc = divmod(k, COLS)
                            for nm, (sr, sc) in (("src", lay[o]), ("dst", d)):
                                if max(abs(rr - sr), abs(cc - sc)) <= 1:
                                    near.add(nm)
                row["near"] = sorted(near) or ["elsewhere"]
        rows.append(row)
    return side["problem"], rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problems")
    ap.add_argument("--scenes", type=int, default=36)
    ap.add_argument("--q", type=int, default=14)
    ap.add_argument("--workers", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="diag_lib2.json")
    a = ap.parse_args()
    os.environ["YADU_LIB_FREE"] = "ignore"
    files = sorted(Path(a.problems).glob("*.json"))
    random.Random(a.seed).shuffle(files)
    allrows = []
    with mp.get_context("spawn").Pool(a.workers) as pool:
        for name, rows in pool.imap_unordered(scene, [(str(f), a.q, a.seed) for f in files[:a.scenes]]):
            allrows += rows
            print(name, len(rows), "teacher-feasible moves,", sum(not r["lib_yes"] for r in rows), "refused", flush=True)
    ref = [r for r in allrows if not r["lib_yes"]]; acc = [r for r in allrows if r["lib_yes"]]
    frontal = lambda r: r["az"] is not None and abs(((r["az"] - 180 + 180) % 360) - 180) <= 22.6
    out = dict(teacher_feasible=len(allrows), refused=len(ref), accepted=len(acc),
               causes=collections.Counter(r["cause"] for r in ref),
               blockers_near=collections.Counter(",".join(r["near"]) for r in ref if r["near"]),
               n_candidates_stored=collections.Counter(r["n_cand"] for r in ref if r["n_cand"] is not None),
               teacher_grasp_frontal_pct=dict(refused=100 * sum(map(frontal, ref)) / max(len(ref), 1), accepted=100 * sum(map(frontal, acc)) / max(len(acc), 1)),
               teacher_grasps_tried_mean=dict(refused=sum(r["tried"] or 0 for r in ref) / max(len(ref), 1), accepted=sum(r["tried"] or 0 for r in acc) / max(len(acc), 1)),
               teacher_azimuth_refused=collections.Counter(r["az"] for r in ref), teacher_azimuth_accepted=collections.Counter(r["az"] for r in acc))
    Path(a.out).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
