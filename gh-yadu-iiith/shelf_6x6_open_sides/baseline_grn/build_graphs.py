#!/usr/bin/env python3
"""shelf66 dataset roots -> GRN graphs (Ait Bouhsain, Alami, Simeon, ICLR 2025), V4 split.

    python build_graphs.py        ->  data/graphs.pkl, data/roots.pkl   (+ counts)

GRN's world (official code, `Dataset.py` / `utils.py`): a scene graph of boxes (l, w, h, x, y, z, yaw); every movable node carries
labels for 6 outputs "action + 5 grasp views" (F), per-view IK reachability (IK) and, on each incoming edge, the per-view
grasp-obstruction ratio (GO) = the share of the view's grasps blocked by that neighbour.  Our adaptation (the shelf has side grasps
only, so a "view" = one of the 16 grasp azimuths; V = 16):
    nodes   base  fixed   the shelf (frame of everybody)
            goal  fixed   a marker box (T's size) at the goal cell: GRN has no goal input, so the goal is a node
            oXX   fixed   the obstacles (their labels are not supervised: the dataset labels only the target's move)
            t0    movable the target; carries every label
    graph   one per QUERY "can T go to its goal in this layout?":  original scene, D (the teacher's edited layouts, obstacles
            parked), L (the rest of the T-only lattice; only the action label is known)
    labels  F[0] = y of the query ("action");  views F[1..16], IK[1..16], GO per incoming edge of t0 (base, goal, obstacles):
            from the teacher's stored per-grasp reasons (32 grasps = 16 azimuths x 2 grasp levels, ids ordered by
            shelf_scene.grasp_poses: level, then closeness of the azimuth to the direction of the object from the robot):
              "out of reach"                     -> IK failure
              "hit o00 ..."                      -> obstruction by that object
              "hit floor_board / arm_ceiling ..." -> obstruction by the shelf (edge base -> t0)
              "overlap a neighbour at the goal", "no sideways route": the grasp fails, nobody is named (counts in F only)
            only infeasible queries store all 32 reasons; a feasible query stores the winning grasp (its view is feasible = 1),
            everything else of it is unknown -> label mask 0.
"""
import itertools
import json
import math
import os
import pickle
import re
from collections import Counter
from pathlib import Path

import numpy as np

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


V = 16
AZ = [22.5 * i for i in range(16)]                      # shelf_scene AZIMUTHS_DEG
X0, PITCH, TABLE_Z, ROWS, COLS = 0.42, 0.075, 0.55, 6, 6


def cell_xy(cell):
    r, c = cell
    return X0 + (r - 2.5) * PITCH, (c - 2.5) * PITCH


def grasp_view(cell):
    """grasp id (0..31) -> azimuth index 0..15, for a target standing in `cell` (shelf_scene.grasp_poses ordering)."""
    x, y = cell_xy(cell)
    to_robot = math.degrees(math.atan2(0.0 - y, 0.0 - x))
    wrap = lambda d: abs((d + 180.0) % 360.0 - 180.0)  # noqa: E731
    order = sorted(range(16), key=lambda i: wrap(AZ[i] - to_robot))      # python sort is stable, like the teacher's
    return order + order                                                 # ids 0-15: first level, 16-31: second level


def box(o):
    return [2 * o["size"], 2 * o["size"], o["height"]]


def node_pose(o, cell):
    x, y = cell_xy(cell)
    return [x, y, TABLE_Z + o["height"] / 2, 0.0]


def parse_reason(s):
    """-> ('ik',) | ('obj', [ids]) | ('struct',) | ('other',)"""
    if "out of reach" in s:
        return ("ik",)
    if "would hit" in s:
        head = s.split(" near ")[0]
        ids = re.findall(r"\bo\d\d\b|\bt0\b", head)
        return ("obj", ids) if ids else ("struct",)
    return ("other",)


def view_labels(q, cell_t0):
    """per-view labels of one stored query q (original / D entry).  -> dict(F[16], Fm[16], IK[16], IKm[16], GO{node: [16]}, GOm)"""
    gv = grasp_view(cell_t0)
    F = np.zeros(V, np.float32); Fm = np.zeros(V, np.float32)
    IK = np.ones(V, np.float32); IKm = np.zeros(V, np.float32)
    GO = {}                                                              # node id -> ratios per view
    GOm = 0.0
    if q["y"] == 1:
        w = int(q["grasp_id"])
        v = gv[w] if 0 <= w < 32 else None
        if v is not None:
            F[v], Fm[v] = 1.0, 1.0
            IK[v], IKm[v] = 1.0, 1.0
        return dict(F=F, Fm=Fm, IK=IK, IKm=IKm, GO=GO, GOm=GOm)
    gr = q.get("grasp_reasons") or {}
    if len(gr) != 32:
        return dict(F=F, Fm=Fm, IK=IK, IKm=IKm, GO=GO, GOm=GOm)
    F[:] = 0.0; Fm[:] = 1.0                                              # every grasp failed: no view works
    for v in range(V):
        ids = [g for g in range(32) if gv[g] == v]                      # the view's two grasps
        kinds = [parse_reason(gr[str(g)]) for g in ids]
        IKm[v] = 1.0
        IK[v] = 0.0 if all(k[0] == "ik" for k in kinds) else 1.0
        for k in kinds:
            if k[0] == "obj":
                for o in set(k[1]):
                    GO.setdefault(o, np.zeros(V, np.float32))[v] += 0.5
            elif k[0] == "struct":
                GO.setdefault("base", np.zeros(V, np.float32))[v] += 0.5
    GOm = 1.0
    return dict(F=F, Fm=Fm, IK=IK, IKm=IKm, GO=GO, GOm=GOm)


def make_graph(rec, cells, goal, y, lab, rid, kind, idx, shape_of):
    """rec: root record, cells: {object id: (row, col)} of the layout, lab: view_labels(...) or None."""
    objs = {o["id"]: o for o in rec["objects"]}
    names = ["base", "goal"] + [o for o in objs if o != "t0"] + ["t0"]
    n = len(names)
    x = np.zeros((n, 7), np.float32)
    fixed = np.ones(n, bool)
    x[0] = [PITCH * COLS, PITCH * ROWS, 0.05, X0, 0.0, TABLE_Z - 0.025, 0.0]                 # the shelf board (frame of the scene)
    t = objs["t0"]
    x[1] = box(t) + node_pose(t, goal)[:3] + [0.0]
    for i, nm in enumerate(names):
        if nm in objs:
            o = objs[nm]
            x[i] = box(o) + node_pose(o, cells[nm])[:3] + [0.0]
    t_i = names.index("t0")
    fixed[t_i] = False
    # edges into the movable node: from base (frame), goal marker and every obstacle; plus its self loop (the code's layout)
    src, dst, eattr, blocking = [], [], [], []
    for j, nm in enumerate(names):
        if j == t_i:
            continue
        src.append(j); dst.append(t_i); eattr.append([1.0, 0.0]); blocking.append(True)
    src.append(t_i); dst.append(t_i); eattr.append([0.0, 1.0]); blocking.append(False)
    go = np.zeros((len(src), V), np.float32)
    gom = np.zeros(len(src), np.float32)
    if lab is not None and lab["GOm"]:
        for e, (s, b) in enumerate(zip(src, blocking)):
            if b:
                go[e] = lab["GO"].get(names[s], 0.0) if names[s] != "goal" else 0.0
                gom[e] = 1.0
    F = np.zeros(1 + V, np.float32); Fm = np.zeros(1 + V, np.float32)
    IK = np.ones(V, np.float32); IKm = np.zeros(V, np.float32)
    F[0], Fm[0] = float(y), 1.0
    if lab is not None:
        F[1:], Fm[1:] = lab["F"], lab["Fm"]
        IK, IKm = lab["IK"], lab["IKm"]
    return dict(rid=rid, kind=kind, idx=idx, y=int(y), names=names, x=x, fixed=fixed, t=t_i, src=np.array(src), dst=np.array(dst),
                eattr=np.array(eattr, np.float32), blocking=np.array(blocking), go=go, gom=gom, F=F, Fm=Fm, IK=IK, IKm=IKm)


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


def root_meta(rec, part, cls):
    """what evaluate_grn.py needs about a scene"""
    bl = (rec["labels"].get("blocker") or {})
    return dict(split=part, cls=cls, minimal_sets=bl.get("minimal_sets") or [], mask=bl.get("mask"), mask_valid=bool(bl.get("blocker_supervision_valid")),
                minimal=next((b["sequence"] for b in rec["labels"].get("B", []) if b.get("role") == "minimal" and b.get("y")), None))

def main():
    split = json.loads(SPLIT.read_text())
    G, metas = [], {}
    for part in ("train", "val", "test"):
        for cls, rids in split[part].items():
            for rid in rids:
                rec = json.loads(find_root(rid).read_text())
                metas[rid] = root_meta(rec, part, cls)
                init = {k: tuple(v) for k, v in rec["initial_cells"].items()}
                goal = tuple(rec["goal"])
                L = rec["labels"]
                par = {o: tuple(v[0]) for o, v in (rec.get("parking") or {}).items() if v}
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
                    g = make_graph(rec, cells, goal, q["y"], view_labels(q, cells["t0"]), rid, kind, idx, None)
                    g["split"], g["cls"] = part, cls
                    G.append(g)
                for cells, y in lattice_queries(rec):
                    g = make_graph(rec, cells, goal, y, None, rid, "L", None, None)
                    g["split"], g["cls"] = part, cls
                    G.append(g)
    Path(HERE / "data").mkdir(exist_ok=True)
    pickle.dump(G, open(HERE / "data" / "graphs.pkl", "wb"))
    pickle.dump(metas, open(HERE / "data" / "roots.pkl", "wb"))
    print(f"{len(G)} graphs")
    for part in ("train", "val", "test"):
        S = [g for g in G if g["split"] == part]
        c = Counter((g["kind"], g["y"]) for g in S)
        print(f"{part}: {len(S)} graphs, feasible {np.mean([g['y'] for g in S]):.3f}; " + ", ".join(f"{k[0]} y={k[1]}: {v}" for k, v in sorted(c.items())))
        lab = [g for g in S if g["gom"].any()]
        print(f"   with full per-view / obstruction labels: {len(lab)}; with a feasible-view label only: {sum(1 for g in S if g['Fm'][1:].any() and not g['gom'].any())}; "
              f"nodes/graph {np.mean([len(g['names']) for g in S]):.1f}")
    # label sanity: how many objects are named obstructors, how often is a view IK-infeasible
    tr = [g for g in G if g["split"] == "train" and g["gom"].any()]
    print("IK=0 share of labelled views:", float(np.mean([(1 - g["IK"][g["IKm"] > 0]).mean() for g in tr])),
          "| mean #views with F=1 in infeasible queries:", float(np.mean([g["F"][1:].sum() for g in tr])))
    print("GO nonzero edges per labelled graph:", float(np.mean([(g["go"].sum(1) > 0).sum() for g in tr])))


if __name__ == "__main__":
    main()
