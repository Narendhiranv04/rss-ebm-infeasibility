#!/usr/bin/env python3
"""shelf66 dataset roots -> SceneCollisionNet states and per-grasp labels, V4 split.

    python build_scn_data.py        ->  data/states.pkl, data/roots_scn.pkl

State = a layout of the scene ("can T reach its goal here?"): the original scene, the teacher's edited layouts (D) and the rest of the
teacher's T-only lattice (L), exactly the queries of the GRN baseline (branch yadu/GRN, build_graphs.py).  Each state has 32 grasps (16
azimuths x 2 grasp levels, shelf_scene.grasp_poses order) with labels from the teacher's stored per-grasp reasons:
    fy     1 = the move works with this grasp, 0 = it fails, -1 = unknown
    pick   1 = the hand at the grasp pose (start cell) is unusable (out of reach, hits something, overlaps a neighbour at the start),
           0 = fine, -1 = unknown        <- SceneCollisionNet's own question: does the gripper at this pose collide?
    place  1 = the hand at the same orientation over the GOAL collides ("overlaps a neighbour at the goal"), 0 = fine, -1 = unknown
  infeasible query with the 32 reasons : fy = 0 everywhere; pick / place from the reason of each grasp
  feasible query                       : the winning grasp fy = pick = place = 0 ... 1 for fy; earlier ids failed (fy = 0, reason unknown)
  L query, y = 0                       : every grasp failed (fy = 0); y = 1: only the state label is known (the winner is not)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import itertools  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import scn_geometry as G  # noqa: E402

HERE = Path(__file__).resolve().parent
SPLIT = HERE / "split_v4.json"                      # Narendhiran's V4 split (naren/dataset-done, FINAL_3D_SELECTION_MANIFEST.json)
# the pushed dataset (branch yadu/dataset-3d); override with DATASET66=<folder holding raw/, raw_none/, raw_k4_ordered/>
DATASET66 = Path(os.environ["DATASET66"]) if os.environ.get("DATASET66") else \
    HERE.parent / "dataset_bd_classquota_without_home_position_no_overpass"
ROOT_DIRS = ("raw/roots", "raw_none/roots", "raw_k4_ordered/roots")
# V4's 45 new K3 layouts: naren/dataset-done, final_dataset_generation/FINAL_3D_DATASET_V4/roots_new (V4_NEW_DIR=<that folder>)
V4_NEW = Path(os.environ["V4_NEW_DIR"]) if os.environ.get("V4_NEW_DIR") else HERE / "v4_new_roots"


def find_root(rid):
    for f in [DATASET66 / d / f"{rid}.json" for d in ROOT_DIRS] + [V4_NEW / f"{rid}.json"]:
        if f.exists():
            return f
    raise FileNotFoundError(f"root {rid} not found under {DATASET66} or {V4_NEW}")





def parse_reason(s):
    """-> ('ik',) | ('obj', [ids]) | ('struct',) | ('other',)"""
    if "out of reach" in s:
        return ("ik",)
    if "would hit" in s:
        head = s.split(" near ")[0]
        ids = re.findall(r"\bo\d\d\b|\bt0\b", head)
        return ("obj", ids) if ids else ("struct",)
    return ("other",)



def lattice_queries(rec):
    """The rest of the teacher's T-only lattice (labels.blocker): it queried every subset S of its object pool (<= 6 objects) parked at
    its first parking cell.  The file keeps the minimal clearing sets only, so y(S) is known for S = a minimal set (1) and for every S
    that contains none (0); supersets are skipped.  Subsets already stored as D queries are left out.  -> [(cells, y)]"""
    L = rec["labels"]
    init = {k: tuple(v) for k, v in rec["initial_cells"].items()}
    goal = tuple(rec["goal"])
    bl = L.get("blocker") or {}
    park = {o: tuple(v[0]) for o, v in (rec.get("parking") or {}).items() if v}
    out = []
    if not (bl.get("full_lattice") and L.get("original") and not L["original"].get("y") and all(o in park for o in bl.get("scope", []))):
        return out
    pool = list(bl["scope"])
    mins = [frozenset(m) for m in bl.get("minimal_sets") or []]
    in_D = {frozenset(d["edit"]) for d in L.get("D", []) if isinstance(d.get("edit"), dict)
            and all(park.get(o) == tuple(c) for o, c in d["edit"].items())}
    for k in range(1, len(pool) + 1):
        for S in itertools.combinations(sorted(pool), k):
            fs = frozenset(S)
            if fs in mins:
                y = 1
            elif any(m <= fs for m in mins):
                continue
            else:
                y = 0
            cells = {**init, **{o: park[o] for o in S}}
            if len(set(cells.values())) < len(cells) or goal in [park[o] for o in S]:
                continue
            if fs not in in_D:
                out.append((cells, y))
    return out




def grasp_labels(q, n=32):
    fy = -np.ones(n, np.int8); pick = -np.ones(n, np.int8); place = -np.ones(n, np.int8)
    if q["y"] == 1:
        w = int(q["grasp_id"])
        fy[:w] = 0
        if 0 <= w < n:
            fy[w] = 1; pick[w] = 0; place[w] = 0
        return fy, pick, place
    fy[:] = 0
    gr = q.get("grasp_reasons") or {}
    if len(gr) != n:
        return fy, pick, place
    for g in range(n):
        s = gr[str(g)]
        k = parse_reason(s)[0]
        if "neighbour at the goal" in s:
            pick[g], place[g] = 0, 1
        elif "no sideways route" in s:
            pass                                                # path-level failure: neither pose check is known to have failed
        elif k in ("ik", "obj", "struct") or "neighbour at the start" in s:
            pick[g] = 1
    return fy, pick, place


def main():
    split = json.loads(SPLIT.read_text())
    states, roots = [], {}
    for part in ("train", "val", "test"):
        for cls, rids in split[part].items():
            for rid in rids:
                rec = json.loads(find_root(rid).read_text())
                objs = {o["id"]: dict(id=o["id"], shape=o["shape"], size=float(o["size"]), height=float(o["height"])) for o in rec["objects"]}
                init = {k: tuple(v) for k, v in rec["initial_cells"].items()}
                goal = tuple(rec["goal"])
                pos, R, meta = G.grasps_of(objs["t0"], init["t0"])
                gx, gy = G.cell_xy(goal); sx, sy = G.cell_xy(init["t0"])
                bl = rec["labels"].get("blocker") or {}
                roots[rid] = dict(split=part, cls=cls, objects=list(objs.values()), init=init, goal=goal, pos=pos, R=R, meta=meta,
                                  shift=np.array([gx - sx, gy - sy, 0.0], np.float32),
                                  parking={o: tuple(v[0]) for o, v in (rec.get("parking") or {}).items() if v},
                                  minimal_sets=bl.get("minimal_sets") or [], mask=bl.get("mask"), mask_valid=bool(bl.get("blocker_supervision_valid")),
                                  minimal=next((b["sequence"] for b in rec["labels"].get("B", []) if b.get("role") == "minimal" and b.get("y")), None))
                L = rec["labels"]
                par = roots[rid]["parking"]
                items = []
                o = L.get("original")
                if o and o.get("y") is not None:
                    items.append(("original", None, o, init))
                for n, d in enumerate(L.get("D", [])):
                    if d.get("y") is None or not isinstance(d.get("edit"), dict) or any(par.get(a) != tuple(b) for a, b in d["edit"].items()):
                        continue
                    cells = dict(init); cells.update({a: tuple(b) for a, b in d["edit"].items()})
                    if len(set(cells.values())) == len(cells):
                        items.append(("D", n, d, cells))
                for kind, idx, q, cells in items:
                    fy, pk, pl = grasp_labels(q)
                    states.append(dict(rid=rid, split=part, cls=cls, kind=kind, idx=idx, y=int(q["y"]), cells=cells, fy=fy, pick=pk, place=pl))
                for cells, ly in lattice_queries(rec):
                    fy = np.zeros(32, np.int8) if ly == 0 else -np.ones(32, np.int8)
                    states.append(dict(rid=rid, split=part, cls=cls, kind="L", idx=None, y=int(ly), cells=cells, fy=fy,
                                       pick=-np.ones(32, np.int8), place=-np.ones(32, np.int8)))
    (HERE / "data").mkdir(exist_ok=True)
    pickle.dump(states, open(HERE / "data" / "states.pkl", "wb"))
    pickle.dump(roots, open(HERE / "data" / "roots_scn.pkl", "wb"))
    print(f"{len(states)} states, {len(roots)} scenes")
    for part in ("train", "val", "test"):
        S = [s for s in states if s["split"] == part]
        fy = np.stack([s["fy"] for s in S]); pk = np.stack([s["pick"] for s in S]); pl = np.stack([s["place"] for s in S])
        print(f"{part}: {len(S)} states (feasible {np.mean([s['y'] for s in S]):.3f}); labelled grasps: fy {int((fy >= 0).sum())} (feasible {int((fy == 1).sum())}), "
              f"pick {int((pk >= 0).sum())} (bad {int((pk == 1).sum())}), place {int((pl >= 0).sum())} (bad {int((pl == 1).sum())})")


if __name__ == "__main__":
    main()
