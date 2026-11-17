#!/usr/bin/env python3
"""shelf66 dataset roots -> PIGINet samples (initial state + goal + task plan -> feasible?).

    python build_data.py            # writes data/samples.pkl and prints the counts

Every labelled query of a root becomes one sample, exactly the unit PIGINet is trained on (problem, one task plan, label):
    original   plan = [move T -> goal] on the scene as it is                         (labels.original)
    D          plan = [move T -> goal] on an EDITED scene (obstacles at parking)     (labels.D, the teacher's hypothetical edits)
    L          the rest of the teacher's T-only LATTICE (labels.blocker): it queried every subset S of its object pool (<= 6
               objects, parked at their first parking cell).  The file keeps the minimal clearing sets, so y(S) is known exactly
               for S = a minimal set (1) and for every S that contains no minimal set (0); supersets are skipped (not stored).
    P          the same negatives in PLAN form: [move each o in S -> parking, move T -> goal] with y = 0, implied by the lattice
               (T still cannot move after S is parked).  Not executed by the teacher; flagged kind "P".
    B          plan = [move o_i -> cell_i, ..., move T -> goal], executed in MuJoCo  (labels.B: minimal, wrong_destination,
               wrong_order, redundant+1, noop_irrelevant, intended_union)
Split: Narendhiran's V4 split (split_v4.json next to this file, keyed by root id) - the same test scenes as REAP and our model.
"""
import itertools
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

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


def root_samples(rid, split, cls):
    rec = json.loads(find_root(rid).read_text())
    objs = {o["id"]: dict(id=o["id"], shape=o["shape"], size=float(o["size"]), height=float(o["height"]), color=o["color"])
            for o in rec["objects"]}
    init = {k: tuple(v) for k, v in rec["initial_cells"].items()}
    goal = tuple(rec["goal"])
    L = rec["labels"]
    base = dict(rid=rid, split=split, cls=cls, goal=goal, objects=list(objs.values()))
    out = []

    def add(kind, role, y, cells, plan, neg=None, n=None, img_src=None):
        out.append(dict(base, kind=kind, role=role, y=int(y), cells=dict(cells), plan=plan, negative_kind=neg, idx=n, img_src=img_src))

    tmove = ("t0", init["t0"], goal)
    if L.get("original") and L["original"].get("y") is not None:
        add("original", "original", L["original"]["y"], init, [tmove], L["original"].get("negative_kind"))
    for n, d in enumerate(L.get("D", [])):
        if d.get("y") is None or not isinstance(d.get("edit"), dict):
            continue
        cells = dict(init)
        ok = True
        for o, c in d["edit"].items():
            if o not in cells:
                ok = False
            cells[o] = tuple(c)
        if ok and len(set(cells.values())) == len(cells):
            add("D", "edit%d" % len(d["edit"]), d["y"], cells, [("t0", cells["t0"], goal)], d.get("negative_kind"), n)
    for n, b in enumerate(L.get("B", [])):
        if b.get("y") is None or b.get("sequence") is None:
            continue
        plan, ok = [], True
        for o, c in b["sequence"]:
            if o not in init:
                ok = False
                break
            plan.append((o, init[o], tuple(c)))
        if ok:
            add("B", b.get("role") or "none", b["y"], init, plan + [tmove], b.get("negative_kind"), n)
    # ---- the rest of the T-only lattice (L) and its plan-form negatives (P)
    bl = L.get("blocker") or {}
    park = {o: tuple(v[0]) for o, v in (rec.get("parking") or {}).items() if v}
    single = {}                                            # object -> index of its single-edit D query (rendered state D<n>)
    for n, d in enumerate(L.get("D", [])):
        if isinstance(d.get("edit"), dict) and len(d["edit"]) == 1:
            (o, c), = d["edit"].items()
            if park.get(o) == tuple(c):
                single.setdefault(o, n)
    if bl.get("full_lattice") and L.get("original") and not L["original"].get("y") and all(o in park for o in bl.get("scope", [])):
        pool = list(bl["scope"])
        mins = [frozenset(m) for m in bl.get("minimal_sets") or []]
        in_D = {frozenset(d["edit"]) for d in L.get("D", []) if isinstance(d.get("edit"), dict)
                and all(park.get(o) == tuple(c) for o, c in d["edit"].items())}
        in_B = {frozenset((o, tuple(c)) for o, c in b["sequence"]) for b in L.get("B", []) if b.get("sequence") is not None}
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
                    add("L", "lattice%d" % k, y, cells, [("t0", cells["t0"], goal)],
                        img_src={o: ("D%d" % single[o] if o in S and o in single else "orig") for o in objs})
                if y == 0 and frozenset((o, park[o]) for o in S) not in in_B:
                    add("P", "implied%d" % k, 0, init, [(o, init[o], park[o]) for o in S] + [tmove])
    meta = dict(rid=rid, split=split, cls=cls, goal=goal, init=init, objects=list(objs.values()), single_edit=single,
                verified_K=rec.get("verified_K"), parking={k: [tuple(c) for c in v] for k, v in (rec.get("parking") or {}).items()},
                alt_parking={k: [tuple(c) for c in v] for k, v in (rec.get("alt_parking") or {}).items()},
                minimal_sets=(L.get("blocker") or {}).get("minimal_sets") or [],
                mask=(L.get("blocker") or {}).get("mask"), mask_valid=bool((L.get("blocker") or {}).get("blocker_supervision_valid")),
                minimal=next((b["sequence"] for b in L.get("B", []) if b.get("role") == "minimal" and b.get("y")), None),
                family=(rec.get("ids") or {}).get("parent_family_id"))
    return out, meta


def main():
    split = json.loads(SPLIT.read_text())
    samples, metas, missing = [], {}, []
    for part in ("train", "val", "test"):
        for cls, rids in split[part].items():
            for rid in rids:
                try:
                    s, m = root_samples(rid, part, cls)
                except FileNotFoundError:
                    missing.append(rid)
                    continue
                samples += s
                metas[rid] = m
    (HERE / "data").mkdir(exist_ok=True)
    pickle.dump(dict(samples=samples, roots=metas, split_sha=split.get("sha256")), open(HERE / "data" / "samples.pkl", "wb"))
    print(f"roots {len(metas)} (missing {len(missing)}), samples {len(samples)}")
    for part in ("train", "val", "test"):
        S = [s for s in samples if s["split"] == part]
        R = [m for m in metas.values() if m["split"] == part]
        print(f"\n{part}: {len(R)} roots, {len(S)} samples, feasible {sum(s['y'] for s in S) / max(1, len(S)):.3f}")
        print("  roots per class:", dict(sorted(Counter(m['cls'] for m in R).items())))
        c = Counter((s["kind"], s["role"], s["y"]) for s in S)
        for k in sorted(c):
            print(f"  {k[0]:9s} {k[1]:18s} y={k[2]}: {c[k]}")
        print("  plan length:", dict(sorted(Counter(len(s['plan']) for s in S).items())),
              "| objects:", dict(sorted(Counter(len(s['objects']) for s in S).items())))
    fam = {}
    for m in metas.values():
        fam.setdefault(m["family"], set()).add(m["split"])
    print("\nfamilies spanning more than one split:", sum(1 for f, s in fam.items() if f and len(s) > 1), "of", len(fam))
    print("shapes:", Counter(o["shape"] for m in metas.values() for o in m["objects"]),
          "| roots without parking for some obstacle:", sum(1 for m in metas.values() if any(o["id"] != "t0" and o["id"] not in m["parking"] for o in m["objects"])))


if __name__ == "__main__":
    main()
