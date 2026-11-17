#!/usr/bin/env python3
"""shelf66 dataset roots (6x6 open-sided shelf, no overpass) -> REAP problems for the confined-shelf PDDL domain.

    python roots66_to_pddl.py <out dir> [--per-class N] [--classes NONE,0,1,2,3] [--seed S]

Reads selected.json of the pushed dataset (DATASET66, default the gh-yadu-iiith copy) and draws N roots per verified
class (stratified, seeded). For each root it writes <root_id>.pddl and the geometry sidecar <root_id>.json.

Cells: shelf_interior__<col>_<row>; (adjacent north a b): b one row deeper, (adjacent east a b): b one column to +y.
Goal: only the target's cell (t0 -> goal), as in the dataset; obstacles may end anywhere.
"""
import argparse
import json
import random
from pathlib import Path

import os

DATASET66 = Path(os.environ["DATASET66"]) if os.environ.get("DATASET66") else Path.home() / ("codebase/rss-ebm-infeasibility/gh-yadu-iiith/shelf_6x6_open_sides/"
                           "dataset_bd_classquota_without_home_position_no_overpass")


ROOT_DIRS = ("raw/roots", "raw_none/roots", "raw_k4_ordered/roots")
# V4's 45 new K3 layouts (naren/dataset-done, FINAL_3D_DATASET_V4/roots_new), kept outside the dataset checkout
V4_NEW = Path(os.environ.get("V4_NEW_DIR", Path(__file__).resolve().parent / "v4_new" / "roots"))


def find_root(rid, dataset=None):
    base = Path(dataset) if dataset else DATASET66
    for f in [base / d / f"{rid}.json" for d in ROOT_DIRS] + [V4_NEW / f"{rid}.json"]:
        if f.exists():
            return f
    raise FileNotFoundError(f"root {rid} not in {base} ({', '.join(ROOT_DIRS)}) or {V4_NEW}")


def cell_name(r, c):
    return f"shelf_interior__{c}_{r}"


def write_problem(rec: dict, cls: str, out: Path) -> dict:
    rows, cols = rec["geometry"]["rows"], rec["geometry"]["cols"]
    rid = rec["ids"]["root_id"] if isinstance(rec.get("ids"), dict) and "root_id" in rec["ids"] else rec["_root_id"]
    init = {k: tuple(v) for k, v in rec["initial_cells"].items()}
    ids = [o["id"] for o in rec["objects"]]
    pn = {i: f"cyl_{i}" for i in ids}
    cells = [(r, c) for r in range(rows) for c in range(cols)]
    occ = set(init.values())
    L = [f"(define (problem shelf66-{rid})", "  (:domain confined-shelf)", "  (:objects",
         "    " + " ".join(pn[i] for i in ids) + " - cylinder",
         "    " + " ".join(cell_name(r, c) for r, c in cells) + " - cell", "    c0 - color", "  )", "  (:init"]
    for r, c in cells:
        if r + 1 < rows:
            L.append(f"    (adjacent north {cell_name(r, c)} {cell_name(r + 1, c)})")
        if c + 1 < cols:
            L.append(f"    (adjacent east {cell_name(r, c)} {cell_name(r, c + 1)})")
    L += [f"    (color-of {pn[i]} c0)" for i in ids]
    L += [f"    (occupied {cell_name(*init[i])} {pn[i]})" for i in ids]
    L += [f"    (empty {cell_name(r, c)})" for r, c in cells if (r, c) not in occ]
    L += ["    (gripper-empty)", "  )", "  (:goal (and", f"    (occupied {cell_name(*rec['goal'])} {pn['t0']})", "  ))", ")"]
    (out / f"{rid}.pddl").write_text("\n".join(L) + "\n")
    minimal = next((b for b in rec["labels"]["B"] if b.get("role") == "minimal" and b.get("y")), None)
    side = dict(problem=rid, cls=cls, verified_K=rec["verified_K"], goal_cell=list(rec["goal"]), target="cyl_t0",
                minimal_sequence=minimal["sequence"] if minimal else None,
                objects=[dict(id=o["id"], pddl=pn[o["id"]], shape=o["shape"], size=o["size"], height=o["height"],
                              color=o["color"], cell=list(init[o["id"]])) for o in rec["objects"]],
                cells={cell_name(r, c): [r, c] for r, c in cells})
    (out / f"{rid}.json").write_text(json.dumps(side, indent=1))
    return side


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--classes", default="NONE,0,1,2,3,4")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dataset", default=str(DATASET66))
    ap.add_argument("--split", default=None, help="split66.json: convert ALL roots of --part (ignores --per-class)")
    ap.add_argument("--part", default="test", choices=("train", "val", "test"))
    a = ap.parse_args()
    ds, out = Path(a.dataset), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sel = json.loads((ds / "selected.json").read_text())
    rng = random.Random(a.seed)
    split = json.loads(Path(a.split).read_text()) if a.split else None
    for cls in a.classes.split(","):
        if split:
            if cls not in split[a.part]:
                continue
            pick, pool = split[a.part][cls], split[a.part][cls]
        else:
            pool = list(sel["selected"][cls])
            pick = rng.sample(pool, min(a.per_class, len(pool)))
        for rid in pick:
            rec = json.loads(find_root(rid, ds).read_text())
            rec["_root_id"] = rid
            write_problem(rec, cls, out)
        print(f"class {cls}: {len(pick)} of {len(pool)} -> {out}")


if __name__ == "__main__":
    main()
