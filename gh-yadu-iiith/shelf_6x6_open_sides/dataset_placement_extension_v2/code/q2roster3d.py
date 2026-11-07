"""Q2_3D_PARITY_V1: the 2D Q2 placement-roster procedure (planar_2d .../code_v2/launch_plan.py, quota Q2) applied to the SHELF66V3
6x6 shelf, for every QC-eligible first-cohort root (production v4).  Geometry only: object ids / shapes / sizes / heights, initial
cells and the shelf geometry; the goal cell only to flag `dest_is_t_goal` (never a veto, never a preference); K only as a reporting
field.  Never read: roles, blockers, parking, mechanism pools, B/D labels, repairs, construction spec, teacher outcomes.

Per root (Q2 = 4 x 8 root + 3 x (2 x 4) prefix = at most 56 physical queries):
  root     up to 4 non-target objects: base order = the 2D rule (round robin over source-distance bins near / mid / far, hash order
           inside a bin), then a greedy diversity pick over (source bin, shape, height tertile, source clutter); x up to 8 legal free
           destinations, stratified by Chebyshev distance from the source (near 1, mid 2-3, far 4-5), hash order inside a stratum,
           interleaved near / mid / far so every leading slice stays stratified; inclusion probability = k / n of the stratum
  prefix1  one-edit hypothetical scene: mover = 1st of the hash-ordered movers, to a hash-chosen free cell
  prefix2  an INDEPENDENT one-edit scene from the same root: mover = 2nd, to a hash-chosen free cell (preferring Chebyshev >= 2
           from prefix1's cell); not a continuation of prefix1
  prefix3  one two-edit scene: mover 1 then mover 2 (distinct), never back into mover 1's source
  each prefix: up to 2 UNMOVED objects nearest the edited objects' original cells x 4 destinations of the edited scene, the first
           object's first destination being the cell the (last) edit vacated; that (object, cell) in the root scene is a static
           OCCUPIED contrast; plus one OUT_OF_BOUNDS root contrast per root.  Static rows never reach the physical teacher.
  A context that cannot exist (too few movers / free cells, no unmoved object) is recorded NOT_APPLICABLE, never manufactured.
Exact semantic duplicates (same snapshot, full hypothetical object state, object, destination, protocol, policy) are labelled once;
every other context reference is kept in the duplicate map.

  python q2roster3d.py freeze SRC V3_DIR OUT_DIR      roster, contexts, duplicate map, roster manifest, coverage report + gates
  python q2roster3d.py smoke  OUT_DIR                 label-blind integration roster (4 TRAIN roots x 8 queries, every block)
  python q2roster3d.py forecast OUT_DIR               stratified runtime / storage forecast (v2 costs, held-out cohort timing check)
  python q2roster3d.py utility RUN_DIR OUT.json       training-utility report of a (partial) run joined with the roster
  python q2roster3d.py figure OUT_DIR                 coverage figure
"""
import gzip, hashlib, json, math, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan3d  # noqa: E402  (scene_hash, clutter: geometry helpers of the frozen V1 roster)

RULE = "Q2_3D_PARITY_V1"
Q2 = {"root_objects": 4, "root_dests": 8, "prefixes": 3, "prefix_objects": 2, "prefix_dests": 4}
DEST_BINS = {"near": (1, 1), "mid": (2, 3), "far": (4, 5)}
ROWS = COLS = 6
BLOCKS = ("root", "prefix1", "prefix2", "prefix3")
POLICY = {"name": "adaptive_v1", "stages": [[4, 500], [8, 1500], [16, 5000]]}
PROTOCOL = "89cbabed5d7e4847"


def H(*a):
    return hashlib.sha256(json.dumps([RULE, *a], default=list).encode()).hexdigest()


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def allowed(raw, k):
    """The only root fields the rule may see (everything else -- roles, blockers, labels, repairs, spec -- is dropped here)."""
    g = raw["geometry"]
    return {"root_id": raw["ids"]["root_id"], "K": k, "objects": [{x: o[x] for x in ("id", "shape", "size", "height")} for o in raw["objects"]],
            "cells": {o: tuple(c) for o, c in raw["initial_cells"].items()}, "goal": tuple(raw["goal"]),
            "geom": {x: g[x] for x in ("x0", "pitch", "rows", "cols")}}


def load(src):
    sel = json.loads((Path(src) / "selected.json").read_text())["selected"]
    return {r: allowed(json.loads((Path(src) / "raw/roots" / f"{r}.json").read_text()), k) for k in ("0", "1", "2", "3") for r in sel[k]}


def base_dist(geom, c):
    return math.hypot(geom["x0"] + (c[0] - 2.5) * geom["pitch"], (c[1] - 2.5) * geom["pitch"])


def source_edges(geom):
    d = sorted(base_dist(geom, (r, c)) for r in range(ROWS) for c in range(COLS))
    return d[11], d[23]                                         # tertiles of the 36 cells' horizontal distance from the arm base


def src_bin(geom, c):
    a, b = source_edges(geom)
    x = base_dist(geom, c)
    return "near" if x <= a + 1e-12 else "mid" if x <= b + 1e-12 else "far"


def nbr(cells, o):
    return sum(cheb(c, cells[o]) == 1 for k, c in cells.items() if k != o)


def clut_bin(n):
    return "0" if n == 0 else "1-2" if n <= 2 else "3+"


def free(cells, extra=()):
    occ = set(map(tuple, cells.values())) | set(extra)
    return [(r, c) for r in range(ROWS) for c in range(COLS) if (r, c) not in occ]


def allocate(n, sizes):
    k = {b: n // 3 + (i < n % 3) for i, b in enumerate(sizes)}
    k = {b: min(k[b], sizes[b]) for b in sizes}
    while sum(k.values()) < n and any(k[b] < sizes[b] for b in sizes):
        for b in sizes:
            if sum(k.values()) < n and k[b] < sizes[b]:
                k[b] += 1
    return k


def dests(key, src, pool, n):
    """2D rule: stratified destinations interleaved near / mid / far -> [(cell, bin, k, n_bin)]."""
    bins = {b: sorted((c for c in pool if lo <= cheb(c, src) <= hi), key=lambda c: H(key, c)) for b, (lo, hi) in DEST_BINS.items()}
    k = allocate(n, {b: len(v) for b, v in bins.items()})
    return [(bins[b][i], b, k[b], len(bins[b])) for i in range(max(k.values(), default=0)) for b in bins if i < k[b]]


def root_objects(v, hb):
    rid, cells, geom = v["root_id"], v["cells"], v["geom"]
    shape = {o["id"]: o["shape"] for o in v["objects"]}
    hgt = {o["id"]: o["height"] for o in v["objects"]}
    movers = sorted(o for o in cells if not o.startswith("t"))
    by = defaultdict(list)
    for o in sorted(movers, key=lambda o: H(rid, "obj", o)):
        by[src_bin(geom, cells[o])].append(o)
    rr = [o for i in range(len(movers)) for b in ("near", "mid", "far") if i < len(by[b]) for o in [by[b][i]]]
    att = {o: (src_bin(geom, cells[o]), shape[o], hb(hgt[o]), clut_bin(nbr(cells, o))) for o in movers}
    seen, out, rest = [set() for _ in range(4)], [], list(rr)
    while rest and len(out) < Q2["root_objects"]:
        o = max(rest, key=lambda o: (sum(att[o][i] not in seen[i] for i in range(4)), -rr.index(o)))
        out.append(o); rest.remove(o)
        for i in range(4):
            seen[i].add(att[o][i])
    return out, rr, att


def contexts(v):
    """[(block, kind, moves [(obj, cell)], status, reason)] -- the three hypothetical prefix scenes of the 2D rule."""
    rid, cells0 = v["root_id"], v["cells"]
    order = sorted((o for o in cells0 if not o.startswith("t")), key=lambda o: H(rid, "edit", o))
    pool0 = free(cells0)
    out = []
    if order and pool0:
        c1 = min(pool0, key=lambda c: H(rid, "edit1dest", c))
        out.append(("prefix1", "DESIGN_ONE_EDIT", [(order[0], c1)]))
        if len(order) > 1:
            pref = [c for c in pool0 if cheb(c, c1) >= 2] or pool0
            out.append(("prefix2", "DESIGN_ONE_EDIT_INDEPENDENT", [(order[1], min(pref, key=lambda c: H(rid, "edit2dest", c)))]))
    if len(order) >= 3 and len(pool0) >= 2:
        a, b = order[0], order[1]
        ca = min(pool0, key=lambda c: H(rid, "edit3first", c))
        after = {**cells0, a: ca}
        cand = [c for c in free(after) if c != cells0[a]]
        if cand:
            out.append(("prefix3", "DESIGN_TWO_EDIT", [(a, ca), (b, min(cand, key=lambda c: H(rid, "edit3second", c)))]))
    return out, order


def plan_root(v, split, hb, snapshot):
    """-> (physical + static rows, context records) for one root."""
    rid, cells0, geom = v["root_id"], v["cells"], v["geom"]
    objs = {o["id"]: o for o in v["objects"]}
    movers = [o for o in cells0 if not o.startswith("t")]
    rows, ctx, slot = [], [], 0

    def key(cells, o, d):
        state = sorted((x, objs[x]["shape"], round(objs[x]["size"], 9), round(objs[x]["height"], 9), list(cells[x])) for x in cells)
        return hashlib.sha256(json.dumps([snapshot, state, o, list(d), PROTOCOL, POLICY]).encode()).hexdigest()[:32]

    def row(block, kind, cells, parent, edits, o, d, band, k, n, oi, di, static=None, pairs=None):
        nonlocal slot
        x = {"qid": f"{rid}:v4:{block}:{oi}.{di}" if not static else f"{rid}:v4:static:{slot}", "root_id": rid, "K": v["K"], "slot": slot,
             "depth": len(edits), "block": block, "prefix_kind": kind, "scene_hash": plan3d.scene_hash(v["objects"], cells), "parent_scene": parent,
             "edits": {e: list(c) for e, c in edits}, "cells": {kk: list(c) for kk, c in cells.items()}, "object": o, "shape": objs[o]["shape"],
             "height": objs[o]["height"], "size": objs[o]["size"], "height_bin": hb(objs[o]["height"]), "src": list(cells[o]), "dest": list(d),
             "band": band, "cheb": cheb(d, cells[o]), "source_bin": src_bin(geom, cells[o]), "source_dist_m": round(base_dist(geom, cells[o]), 4),
             "src_clutter": clut_bin(nbr(cells, o)), "clutter": plan3d.clutter({kk: tuple(c) for kk, c in cells.items()}, o, tuple(d)) if not static else None,
             "dest_is_t_goal": tuple(d) == v["goal"], "n_obj": len(cells), "obj_rank": oi, "dest_rank": di, "stratum_k": k, "stratum_n": n,
             "incl_prob": round(k / n, 6) if n else None, "static": static, "pairs_with": pairs, "split": split, "reference": False, "cohort": RULE,
             "semantic_key": key(cells, o, d)}
        slot += 1
        rows.append(x)
        return x
    h0 = plan3d.scene_hash(v["objects"], cells0)
    chosen, rr, att = root_objects(v, hb)
    ctx.append({"root_id": rid, "block": "root", "status": "APPLICABLE" if chosen else "NOT_APPLICABLE", "reason": None if chosen else "no non-target object",
                "scene_hash": h0, "moves": [], "object_base_order": rr, "objects": chosen, "attributes": att, "destinations": {}})
    for oi, o in enumerate(chosen):
        ds = dests((rid, "root", o), cells0[o], free(cells0), Q2["root_dests"])
        ctx[-1]["destinations"][o] = [[list(d), b, k, n] for d, b, k, n in ds]
        for di, (d, b, k, n) in enumerate(ds):
            row("root", None, cells0, None, [], o, d, b, k, n, oi, di)
    eds, order = contexts(v)
    have = {e[0] for e in eds}
    for blk in BLOCKS[1:]:
        if blk not in have:
            need = {"prefix1": 1, "prefix2": 2, "prefix3": 3}[blk]
            ctx.append({"root_id": rid, "block": blk, "status": "NOT_APPLICABLE", "reason": f"{len(order)} non-target object(s) < {need} needed for the edit(s)"})
    for blk, kind, moves in eds:
        cells = dict(cells0)
        for a, c in moves:
            assert a in cells and not a.startswith("t") and tuple(c) in free(cells), (rid, blk, a, c)
            cells[a] = c
        moved = {a for a, _ in moves}
        vac = cells0[moves[-1][0]]
        qobj = sorted((o for o in movers if o not in moved), key=lambda o: (min(cheb(cells0[o], cells0[a]) for a in moved), H(rid, blk, o)))[:Q2["prefix_objects"]]
        rec = {"root_id": rid, "block": blk, "kind": kind, "status": "APPLICABLE" if qobj else "NOT_APPLICABLE",
               "reason": None if qobj else "no unmoved non-target object to query", "scene_hash": plan3d.scene_hash(v["objects"], cells), "parent_scene": h0,
               "moves": [[a, list(c)] for a, c in moves], "vacated": list(vac), "objects": qobj, "destinations": {}}
        ctx.append(rec)
        for oi, o in enumerate(qobj):
            use_vac = oi == 0 and tuple(vac) not in set(map(tuple, cells.values()))
            ds = ([(vac, "vacated", 1, 1)] if use_vac else []) + dests((rid, blk, o), cells[o], free(cells, [vac] if use_vac else []),
                                                                       Q2["prefix_dests"] - int(use_vac))
            rec["destinations"][o] = [[list(d), b, k, n] for d, b, k, n in ds]
            for di, (d, b, k, n) in enumerate(ds):
                q = row(blk, kind, cells, h0, moves, o, d, b, k, n, oi, di)
            if use_vac and not any(r["static"] == "OCCUPIED" and r["object"] == o and r["dest"] == list(vac) for r in rows):
                row("static", None, cells0, None, [], o, vac, "static", 1, 1, oi, 0, static="OCCUPIED",
                    pairs=f"{rid}:v4:{blk}:0.0")
    if chosen:
        o = chosen[0]
        row("static", None, cells0, None, [], o, (ROWS, cells0[o][1]), "static", 1, 1, 0, 0, static="OUT_OF_BOUNDS")
    return rows, ctx


def build(src, split, snapshot):
    roots = load(src)
    hs = sorted(o["height"] for v in roots.values() for o in v["objects"] if not o["id"].startswith("t"))
    e1, e2 = hs[len(hs) // 3], hs[2 * len(hs) // 3]
    hb = lambda h: "short" if h < e1 else "medium" if h < e2 else "tall"  # noqa: E731
    rows, ctx = [], []
    for r in sorted(roots):
        a, b = plan_root(roots[r], split[r], hb, snapshot)
        rows += a; ctx += b
    first, dup = {}, defaultdict(list)
    phys = []
    for x in rows:
        if x["static"]:
            continue
        k = x["semantic_key"]
        if k in first:
            dup[k].append(x["qid"])
        else:
            first[k] = x["qid"]; phys.append(x)
    keep = {x["qid"] for x in phys} | {x["qid"] for x in rows if x["static"]}
    out = [x for x in rows if x["qid"] in keep]
    dmap = {k: [first[k]] + v for k, v in sorted(dup.items())}
    return roots, out, ctx, dmap, (e1, e2)


def write_gz(rows, path):
    with gzip.GzipFile(path, "wb", mtime=0) as f:
        for x in rows:
            f.write((json.dumps(x, sort_keys=True, default=list) + "\n").encode())


def coverage(roots, rows, ctx, dmap, split, edges, v3):
    phys = [x for x in rows if not x["static"]]
    nt = Counter(sum(not o["id"].startswith("t") for o in v["objects"]) for v in roots.values())
    per_n = {1: 8, 2: 24, 3: 44}
    theo = sum(c * (0 if n == 0 else per_n.get(n, 56)) for n, c in nt.items())
    cnt = lambda f, rr=phys: dict(sorted(Counter(map(f, rr)).items()))  # noqa: E731
    na = Counter((c["block"], c["reason"]) for c in ctx if c["status"] == "NOT_APPLICABLE")
    short = []
    for c in ctx:
        if c["status"] != "APPLICABLE":
            continue
        want = Q2["root_dests"] if c["block"] == "root" else Q2["prefix_dests"]
        for o, ds in c["destinations"].items():
            if len(ds) < want:
                short.append((c["root_id"], c["block"], o, len(ds)))
    nobj_short = sum(1 for c in ctx if c["status"] == "APPLICABLE" and len(c["objects"]) < (Q2["root_objects"] if c["block"] == "root" else Q2["prefix_objects"])
                     and c["block"] != "root")
    per_root = Counter(x["root_id"] for x in phys)
    q_per_root = Counter(per_root.get(r, 0) for r in roots)
    # gates
    rep = defaultdict(set)
    for r, v in roots.items():
        n = sum(not o["id"].startswith("t") for o in v["objects"])
        for b, need in (("root", 1), ("prefix1", 2), ("prefix2", 2), ("prefix3", 3)):
            if n >= need:
                rep[(v["K"], split[r], b)].add(r)
    seen = {(x["K"], x["split"], x["block"]) for x in phys}
    g_blocks = sorted(f"K{k}:{s}:{b}" for (k, s, b) in rep if (k, s, b) not in seen)
    sk, sh = defaultdict(set), defaultdict(set)
    for x in rows:
        sk[x["semantic_key"]].add(x["split"]); sh[x["scene_hash"]].add(x["split"])
    for k, qs in dmap.items():
        for q in qs:
            sk[k].add(split[q.split(":")[0]])
    bad_edit, bad_dest = [], []
    for c in ctx:
        if c["status"] != "APPLICABLE" or c["block"] == "root":
            continue
        cells = dict(roots[c["root_id"]]["cells"])
        movers = [a for a, _ in c["moves"]]
        for a, d in c["moves"]:
            if a.startswith("t") or tuple(d) in set(map(tuple, cells.values())) or not (0 <= d[0] < ROWS and 0 <= d[1] < COLS):
                bad_edit.append((c["root_id"], c["block"]))
            cells[a] = tuple(d)
        if len(set(movers)) != len(movers) or len(set(map(tuple, cells.values()))) != len(cells) or any(o in movers for o in c["objects"]):
            bad_edit.append((c["root_id"], c["block"]))
    for x in rows:
        occ = {tuple(c) for k, c in x["cells"].items() if k != x["object"]} | {tuple(x["cells"][x["object"]])}
        inb = 0 <= x["dest"][0] < ROWS and 0 <= x["dest"][1] < COLS
        if (not x["static"] and (not inb or tuple(x["dest"]) in occ)) or (x["static"] == "OCCUPIED" and tuple(x["dest"]) not in occ) or \
                (x["static"] == "OUT_OF_BOUNDS" and inb):
            bad_dest.append(x["qid"])
    bands_by = {(s, d): {x["band"] for x in phys if x["split"] == s and x["depth"] == d} for s in ("train", "val", "test") for d in (0, 1, 2)}
    band_gap = sorted(f"{s}:d{d}:{b}" for (s, d), bs in bands_by.items() for b in (("near", "mid", "far") if d == 0 else ("near", "mid", "far", "vacated"))
                      if b not in bs)
    old = v3["split"]
    gates = {"blocks_present_where_representable": not g_blocks, "split_inherited_unchanged": all(split[r] == old[r] for r in roots),
             "semantic_keys_single_split": all(len(s) == 1 for s in sk.values()), "scene_layouts_single_split": all(len(s) == 1 for s in sh.values()),
             "edit_states_valid": not bad_edit, "destinations_legal_in_own_scene": not bad_dest,
             "bands_present_per_split_and_depth": not band_gap, "physical_semantic_keys_unique": len({x["semantic_key"] for x in phys}) == len(phys),
             "count_reconciles": True}
    return {"rule": RULE, "quota": Q2, "dest_bins_chebyshev": DEST_BINS, "source_bins": "tertiles of the 36 cells' horizontal distance from the arm base",
            "height_tertile_edges_m": [round(e, 5) for e in edges], "non_target_count_hist": dict(sorted(nt.items())),
            "theoretical_max_before_destination_availability": theo, "realized_before_dedupe": len(phys) + sum(len(v) - 1 for v in dmap.values()),
            "exact_semantic_duplicates_removed": sum(len(v) - 1 for v in dmap.values()), "physical_queries": len(phys),
            "static_contrasts": len(rows) - len(phys), "static_by_kind": cnt(lambda x: x["static"], [x for x in rows if x["static"]]),
            "reductions": {"contexts_not_applicable": {f"{b}: {r}": n for (b, r), n in sorted(na.items())},
                           "prefix_contexts_with_fewer_than_2_query_objects": nobj_short, "object_destination_lists_short": len(short),
                           "short_examples": short[:10]},
            "roots": len(roots), "roots_with_queries": len(per_root), "queries_per_root_hist": dict(sorted(q_per_root.items())),
            "by_block": cnt(lambda x: x["block"]), "by_split": cnt(lambda x: x["split"]), "by_K": cnt(lambda x: x["K"]),
            "by_depth": cnt(lambda x: x["depth"]), "by_band": cnt(lambda x: x["band"]), "by_shape": cnt(lambda x: x["shape"]),
            "by_source_bin": cnt(lambda x: x["source_bin"]), "by_height_bin": cnt(lambda x: x["height_bin"]), "by_src_clutter": cnt(lambda x: x["src_clutter"]),
            "by_block_band": cnt(lambda x: f"{x['block']}:{x['band']}"), "by_split_block": cnt(lambda x: f"{x['split']}:{x['block']}"),
            "by_K_split_block": cnt(lambda x: f"K{x['K']}:{x['split']}:{x['block']}"),
            "by_K_split_block_band": cnt(lambda x: f"K{x['K']}:{x['split']}:{x['block']}:{x['band']}"),
            "by_block_shape_source": cnt(lambda x: f"{x['block']}:{x['shape']}:{x['source_bin']}"),
            "dest_is_t_goal": sum(x["dest_is_t_goal"] for x in phys), "distinct_scenes": len({x["scene_hash"] for x in phys}),
            "distinct_root_objects_queried": len({(x["root_id"], x["object"]) for x in phys}),
            "distinct_dest_cells_per_root_mean": round(statistics.mean(len({tuple(x["dest"]) for x in phys if x["root_id"] == r}) for r in per_root), 2),
            "thin_strata_lt_20": {k: n for k, n in cnt(lambda x: f"K{x['K']}:{x['split']}:{x['block']}:{x['band']}").items() if n < 20},
            "gates": gates, "gate_details": {"missing_blocks": g_blocks, "bad_edits": bad_edit[:10], "bad_destinations": bad_dest[:10], "band_gaps": band_gap}}


def freeze(src, v3dir, out):
    import prod3d
    v3, out = Path(v3dir), Path(out)
    fz = out / "frozen"; fz.mkdir(parents=True, exist_ok=True)
    spl = json.loads((v3 / "frozen/split_manifest_first_cohort.json").read_text())
    coh = json.loads((v3 / "frozen/cohort_manifest.json").read_text())
    snapshot = json.loads((v3 / "frozen/source_pin_first_cohort.json").read_text())["manifest_sha256"]
    roots, rows, ctx, dmap, edges = build(src, spl["split"], snapshot)
    assert sorted(roots) == sorted(coh["cohort_roots"]), "v4 covers exactly the QC-eligible first cohort"
    write_gz(rows, fz / "q2_parity_roster_v4.jsonl.gz")
    write_gz(ctx, fz / "q2_contexts_v4.jsonl.gz")
    (fz / "semantic_duplicates_v4.json").write_text(json.dumps({"rule": "identical semantic key (snapshot, full object state, object, destination, "
                                                                "protocol, policy): one physical query, every context reference listed",
                                                                "duplicates": dmap}, indent=1, sort_keys=True))
    rep = coverage(roots, rows, ctx, dmap, spl["split"], edges, json.loads((v3.parent / "production_v2/split_manifest_live.json").read_text()))
    th, rb = rep["theoretical_max_before_destination_availability"], rep["realized_before_dedupe"]
    rep["gates"]["count_reconciles"] = rb <= th
    rep["count_reconciliation"] = {"theoretical": th, "realized_before_dedupe": rb, "difference": th - rb,
                                   "explained_by": "NOT_APPLICABLE contexts, prefix contexts with < 2 unmoved objects and short destination lists (see reductions)"}
    files = {"roster": "production_v4/frozen/q2_parity_roster_v4.jsonl.gz", "source_pin": "production_v3/frozen/source_pin_first_cohort.json",
             "split_manifest": "production_v3/frozen/split_manifest_first_cohort.json", "qc_roots": "production_v3/QC_ELIGIBLE_ROOTS.json",
             "contexts": "production_v4/frozen/q2_contexts_v4.jsonl.gz", "duplicates": "production_v4/frozen/semantic_duplicates_v4.json"}
    man = {"rule": RULE, "procedure": "2D Q2 (launch_plan.py, quota Q2) on the 6x6 shelf; see code/q2roster3d.py", "cohort_roots": sorted(roots),
           "n_cohort_roots": len(roots), "roots_with_queries": rep["roots_with_queries"], "physical_queries": rep["physical_queries"],
           "static_contrasts": rep["static_contrasts"], "teacher_protocol": PROTOCOL, "policy": POLICY, "files": files,
           "sha256": {k: prod3d.sha_file(prod3d.ROOT / v) for k, v in files.items()}}
    (fz / "roster_manifest_v4.json").write_text(json.dumps(man, indent=1, sort_keys=True))
    (out / "COVERAGE_REPORT_V4.json").write_text(json.dumps(rep, indent=1, sort_keys=True, default=list))
    print(json.dumps({k: rep[k] for k in ("theoretical_max_before_destination_availability", "realized_before_dedupe", "exact_semantic_duplicates_removed",
                                          "physical_queries", "static_contrasts", "by_block", "by_split", "by_band", "gates")}, default=list))


def smoke(out):
    """Label-blind integration roster: one TRAIN root per K with >= 4 non-target objects (K3: >= 6, heavy clutter), hash-ranked, at least
    one author-approved and one sidecar-certified root; 8 rows each covering root near / mid / far + a second object, prefix1 (vacated),
    prefix2, prefix3 (vacated + second object)."""
    out = Path(out)
    rows = [json.loads(x) for x in gzip.open(out / "frozen/q2_parity_roster_v4.jsonl.gz", "rt")]
    qc = json.loads((HERE.parent / "production_v3/QC_ELIGIBLE_ROOTS.json").read_text())["roots"]
    by = defaultdict(dict)
    for x in rows:
        if not x["static"] and x["split"] == "train":
            by[x["root_id"]][(x["block"], x["obj_rank"], x["dest_rank"])] = x
    want = [("root", 0, 0), ("root", 0, 1), ("root", 0, 2), ("root", 1, 0), ("prefix1", 0, 0), ("prefix2", 0, 1), ("prefix3", 0, 0), ("prefix3", 1, 0)]
    ok = lambda r, k: all(w in by[r] for w in want) and by[r][("root", 0, 0)]["K"] == k and \
        by[r][("root", 0, 0)]["n_obj"] - 1 >= (6 if k == "3" else 4)  # noqa: E731
    cand = {k: sorted((r for r in by if ok(r, k)), key=lambda r: H("smoke", r)) for k in ("0", "1", "2", "3")}
    for i in range(64):
        pick = [cand[k][(i >> (2 * j)) % min(4, len(cand[k]))] for j, k in enumerate(("0", "1", "2", "3"))]
        st = {qc[r]["status"] for r in pick}
        if "AUTHOR_QC_PASS" in st and "SIDECAR_REPRO_QC_PASS" in st:
            break
    sel = [by[r][w] for r in pick for w in want]
    (out / "integration").mkdir(parents=True, exist_ok=True)
    write_gz(sel, out / "integration/smoke_roster_v4.jsonl.gz")
    print(json.dumps({"roots": pick, "qc": [qc[r]["status"] for r in pick], "queries": len(sel), "bands": dict(Counter(x["band"] for x in sel)),
                      "blocks": dict(Counter(x["block"] for x in sel)), "shapes": dict(Counter(x["shape"] for x in sel))}))


def forecast(out):
    """Stratified CPU cost per physical query from the measured V2 adaptive_v1 run A (12 workers), hierarchy K x depth x band x shape ->
    K x depth x band -> K x depth -> K -> all (>= 3 samples), root-lineage bootstrap; checked on held-out first-cohort timing samples."""
    import random
    import report3d
    out = Path(out)
    root = HERE.parent
    rows = [json.loads(x) for x in gzip.open(out / "frozen/q2_parity_roster_v4.jsonl.gz", "rt")]
    phys = [x for x in rows if not x["static"]]
    A = report3d.load_run(root / "mock_v2/runs/adaptive_v1_A")
    an = json.loads((root / "mock_v2/validation/analysis_v2.json").read_text())
    eff, pos = an["forecast_live"]["adaptive_v1"]["parallel_efficiency"], an["forecast_live"]["note"]["positive_rate_adaptive_A"]
    st = an["storage"]["adaptive_v1_A"]
    keys = lambda x: (("kdbs", x["K"], x["depth"], x["band"], x["shape"]), ("kdb", x["K"], x["depth"], x["band"]), ("kd", x["K"], x["depth"]),  # noqa: E731
                      ("K", x["K"]), ("all",))

    def table(recs):
        c = defaultdict(list)
        for r in recs:
            for k in keys(r):
                c[k].append(r["cpu_query_s"])
        return c

    def est(c, qs):
        """qs: [(stratum keys, count)]; each stratum costs the mean of its finest level with >= 3 samples."""
        tot, lvl = 0.0, Counter()
        for ks, n in qs:
            for k in ks:
                if len(c.get(k, [])) >= 3:
                    tot += n * statistics.mean(c[k]); lvl[k[0]] += n; break
        return tot, lvl
    strata = lambda xs: list(Counter(keys(x) for x in xs).items())  # noqa: E731
    recs = list(A["recs"].values())
    c0 = table(recs)
    sp = strata(phys)
    point, lvl = est(c0, sp)
    by_root = defaultdict(list)
    for r in recs:
        by_root[r["qid"].split(":")[0]].append(r)
    rng, boots, rts = random.Random(0), [], sorted(by_root)
    for _ in range(2000):
        boots.append(est(table([r for rt in (rng.choice(rts) for _ in rts) for r in by_root[rt]]), sp)[0])
    boots.sort()
    p05, p95 = boots[100], boots[1900]
    held = {}
    for name, d in (("first_cohort_train_sample_144 (unit=query: no IK cache sharing)", root / "production_v3/finger_width/cohort_run_source"),
                    ("v3_integration_smoke_12 (unit=root, production mode)", root / "production_v3/integration/smoke_A"),
                    ("v4_integration_smoke_32 (unit=root, production mode)", out / "integration/smoke_A")):
        if (d / "units").exists():
            hr = [r for f in sorted((d / "units").glob("*.json")) for r in json.loads(f.read_text())["records"]]
            pred = est(c0, strata(hr))[0]
            meas = sum(r["cpu_query_s"] for r in hr)
            held[name] = {"queries": len(hr), "predicted_cpu_s": round(pred, 1), "measured_cpu_s": round(meas, 1), "measured_over_predicted": round(meas / pred, 3)}
    ratio = max([1.0] + [h["measured_over_predicted"] for h in held.values()])
    wall_h = lambda s: s / (12 * eff) / 3600  # noqa: E731
    disk = (len(phys) * pos * st["bytes_per_witness"] + len(phys) * st["json_bytes_per_query"]) / 1e9
    fc = {"production_queries": len(phys), "workers": 12, "parallel_efficiency": eff, "cost_source_levels": dict(lvl),
          "cpu_hours": round(point / 3600, 2), "cpu_hours_p05_p95": [round(p05 / 3600, 2), round(p95 / 3600, 2)],
          "held_out_timing_checks": held, "held_out_max_ratio": ratio,
          "scenarios_wall_hours": {"optimistic_p05": round(wall_h(p05), 2), "base": round(wall_h(point), 2), "conservative_p95": round(wall_h(p95), 2),
                                   "held_out_calibrated": round(wall_h(point * ratio), 2)},
          "wall_hours_expected": round(wall_h(point), 2),
          "wall_hours_buffered": round(1.5 * wall_h(max(p95, point * ratio)), 2),
          "buffer_rule": "1.5 x max(bootstrap p95, point x the largest held-out measured/predicted ratio)",
          "disk_gb": round(disk, 3), "disk_gb_x1_5": round(1.5 * disk, 3), "disk_gb_buffered": round(1.5 * disk, 3),
          "by_split": dict(Counter(x["split"] for x in phys)), "by_K": dict(Counter(x["K"] for x in phys)), "by_block": dict(Counter(x["block"] for x in phys)),
          "strata_backed_off_to_coarser_levels": {k: n for k, n in lvl.items() if k != "kdbs"},
          "assumptions": ["per-query CPU of the V2 adaptive_v1 run A (12 pinned workers, same teacher protocol 89cbabed5d7e4847)",
                          "parallel efficiency of the V2 12-worker sweep vs serial", "disk = queries x positive rate (run A) x bytes / witness + JSON / query"]}
    (out / "frozen/forecast_v4.json").write_text(json.dumps(fc, indent=1, sort_keys=True, default=dict))
    print(json.dumps({k: fc[k] for k in ("production_queries", "cpu_hours", "cpu_hours_p05_p95", "held_out_timing_checks", "scenarios_wall_hours",
                                         "wall_hours_buffered", "disk_gb_buffered")}, default=dict))


def utility(run, out_json):
    """Training-utility report of a (partial or finished) run: labels joined with the frozen roster by qid -> counts by block / band / shape /
    K / split / depth, effective distinct queries, scenes and context references (duplicate map), unlabeled and negative kinds kept apart."""
    run = Path(run)
    rost = {json.loads(x)["qid"]: json.loads(x) for x in gzip.open(HERE.parent / "production_v4/frozen/q2_parity_roster_v4.jsonl.gz", "rt")}
    dmap = json.loads((HERE.parent / "production_v4/frozen/semantic_duplicates_v4.json").read_text())["duplicates"]
    recs = [r for f in sorted((run / "units").glob("*.json")) for r in json.loads(f.read_text())["records"]]
    lab = lambda r: "1" if r["label"] == 1 else f"0:{r.get('negative_kind')}" if r["label"] == 0 else f"None:{r['status']}"  # noqa: E731
    by = lambda f: dict(sorted(Counter(f"{f(rost[r['qid']])}|{lab(r)}" for r in recs).items()))  # noqa: E731
    rep = {"run": str(run), "queries_labelled": len(recs), "labels": dict(Counter(lab(r) for r in recs)),
           "by_block": by(lambda x: x["block"]), "by_band": by(lambda x: x["band"]), "by_shape": by(lambda x: x["shape"]),
           "by_K": by(lambda x: f"K{x['K']}"), "by_split": by(lambda x: x["split"]), "by_depth": by(lambda x: x["depth"]),
           "effective_distinct_queries": len({rost[r["qid"]]["semantic_key"] for r in recs}),
           "distinct_scenes": len({rost[r["qid"]]["scene_hash"] for r in recs}), "witness_associations": sum(len(r["witnesses"]) for r in recs),
           "extra_context_references": sum(len(dmap.get(rost[r["qid"]]["semantic_key"], [None])) - 1 for r in recs),
           "loader_schema": "production_v4/Q2_PARITY_SPEC_AND_COVERAGE.md section 6"}
    Path(out_json).write_text(json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps({k: rep[k] for k in ("queries_labelled", "labels", "effective_distinct_queries", "distinct_scenes")}))


def figure(out):
    """Coverage figure: physical queries per block x destination band, and per K x split (V2 / V3 palette)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from figs3d import C, INK
    rep = json.loads((Path(out) / "COVERAGE_REPORT_V4.json").read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    bands, cols = ("near", "mid", "far", "vacated"), [C["s1"], C["s2"], C["s3"], C["s4"]]
    left = [0] * 4
    for b, col in zip(bands, cols):
        n = [rep["by_block_band"].get(f"{blk}:{b}", 0) for blk in BLOCKS]
        ax[0].barh(list(BLOCKS), n, left=left, color=col, height=0.6, edgecolor="white", linewidth=1.5)
        left = [a + c for a, c in zip(left, n)]
    for i, t in enumerate(left):
        ax[0].text(t + 300, i, f"{t:,}", va="center", fontsize=8, color=INK)
    ax[0].invert_yaxis(); ax[0].set_xlabel("physical queries"); ax[0].set_xlim(0, max(left) * 1.15)
    ax[0].set_title("Q2-parity roster: block x destination band", loc="left", color=INK, pad=24)
    ax[0].legend([Patch(color=c) for c in cols], bands, frameon=False, fontsize=7.5, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    sp, w = ("train", "val", "test"), 0.2
    for j, k in enumerate(("0", "1", "2", "3")):
        n = [sum(v for key, v in rep["by_K_split_block"].items() if key.startswith(f"K{k}:{s}:")) for s in sp]
        ax[1].bar([i + (j - 1.5) * w for i in range(3)], n, width=w * 0.92, color=cols[j], label=f"K{k}")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(sp); ax[1].set_ylabel("physical queries")
    ax[1].set_title("Physical queries by split and class", loc="left", color=INK, pad=24)
    ax[1].legend(frameon=False, fontsize=7.5, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    (Path(out) / "figures").mkdir(exist_ok=True)
    fig.savefig(Path(out) / "figures/q2_parity_coverage.png", dpi=150)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "freeze":
        freeze(*sys.argv[2:5])
    elif c == "smoke":
        smoke(sys.argv[2])
    elif c == "forecast":
        forecast(sys.argv[2])
    elif c == "utility":
        utility(sys.argv[2], sys.argv[3])
    elif c == "figure":
        figure(sys.argv[2])
