"""Launch planning for the 2D placement sidecar, from root metadata only (never calls the physical teacher).

  splits   lineage groups = union of parent_family_id, identical object layouts and conversion origin; whole groups go to
           train / val / test (70 / 15 / 15) by a deterministic class-stratified greedy; groups holding roots that earlier
           development touched (smoke roster, unit fixture) are forced to train.
  plan     per root, candidates chosen from scene geometry only (no roles, blockers, repairs, Y labels or K):
             root block     up to 4 non-target objects (source near / mid / far round robin) x 12 destinations
             prefix blocks  2 object-only edits (one random design edit; one archived executed first move, flagged as chosen
                            with protected metadata), each x the 2 objects nearest the edited one x 6 destinations, the first
                            object's first destination being the cell the edit vacated; plus that (object, cell) in the root
                            scene as a static-ineligible contrast (no teacher call)
           destinations: legal, empty, not T's goal; stratified by Chebyshev distance from the source (near 1-2, mid 3-4,
           far 5-7), simple random (hash) order inside a stratum, so inclusion probability = k / n of its stratum.
  pilot    25 train roots (5 per class, one per lineage group, >= 2 movable obstacles), a fixed prefix of each block.

    python launch_plan.py freeze <launch_dir>     # split_manifest.json, query_plan.jsonl.gz, pilot_roster.json, plan_summary.json
    python launch_plan.py verify <launch_dir>     # recompute from the dataset and compare (drift check)
"""
import gzip, hashlib, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path

import placement_smoke as P

PLAN = {"version": "PLACEMENT_PLAN_V1", "seed": "LAUNCH_READY_2D_V1", "split_targets": {"train": 0.70, "val": 0.15, "test": 0.15},
        "root_objects": 4, "root_dests": 12, "prefixes": 2, "prefix_objects": 2, "prefix_dests": 6,
        "dest_bins": {"near": [1, 2], "mid": [3, 4], "far": [5, 7]}, "source_bins": "bd_2d.env regions (base-distance tertiles)",
        "pilot": {"roots_per_class": 5, "root_objects": 2, "root_dests": 4, "prefix_dests": 2, "min_obstacles": 2},
        "dev_touched_roots": "smoke_repro_v2/roster.json roots + unit fixture 2d-plus2d-srv-w0408-s006"}
CLASSES = ("NONE", "0", "1", "2", "3")
COLS = ("trial_id", "block", "prefix", "prefix_kind", "object", "source_cell", "source_bin", "dest", "dest_bin", "stratum_k", "stratum_n",
        "pilot", "static_ineligible", "obj_rank", "dest_rank")
QUOTAS = {"Q1": {"root_objects": 4, "root_dests": 12, "prefixes": 2, "prefix_objects": 2, "prefix_dests": 6},   # = the design maximum
          "Q2": {"root_objects": 4, "root_dests": 8, "prefixes": 2, "prefix_objects": 2, "prefix_dests": 4},
          "Q3": {"root_objects": 3, "root_dests": 6, "prefixes": 2, "prefix_objects": 1, "prefix_dests": 4},
          "Q4": {"root_objects": 2, "root_dests": 4, "prefixes": 2, "prefix_objects": 1, "prefix_dests": 2}}   # = the pilot shape
TRIAL = lambda root, prefix, obj, dest: P.sha(f"PLACEMENT_LAUNCH_V1|{json.dumps([root, prefix, obj, dest])}".encode())[:16]  # noqa: E731
H = lambda *a: P.sha(json.dumps([PLAN["seed"], *a]).encode())  # noqa: E731
cheb = lambda a, b: max(abs(a // 8 - b // 8), abs(a % 8 - b % 8))  # noqa: E731


def selected():
    s = json.loads((P.DATA / "selected.json").read_text())["selected"]
    return {x: c for c in CLASSES for x in s[c]}


def lineage(R):
    """Union-find over the three proven sharing relations -> {group_id: sorted root ids}."""
    par = {x: x for x in R}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for key in (lambda r: r["ids"]["parent_family_id"], lambda r: json.dumps(sorted((o["id"], o["cell"]) for o in r["scene"]["objects"])),
                lambda r: r["construction"].get("converted_from") or r["ids"]["root_id"]):
        first = {}
        for x in sorted(R):
            k = key(R[x])
            par[find(x)] = find(first.setdefault(k, x))
    g = defaultdict(list)
    for x in sorted(R):
        g[find(x)].append(x)
    return {"grp-" + P.sha("|".join(v).encode())[:12]: v for v in g.values()}


def splits(groups, cls, forced):
    N, T = Counter(cls.values()), PLAN["split_targets"]
    cur, out = {s: Counter() for s in T}, {}
    for g in sorted(groups, key=lambda g: (g not in forced, -len(groups[g]), H("group", g))):
        n = Counter(cls[x] for x in groups[g])
        score = lambda s: (sum(k * (T[s] * N[c] - cur[s][c]) / N[c] for c, k in n.items()), -list(T).index(s))  # noqa: E731
        out[g] = "train" if g in forced else max(T, key=score)
        cur[out[g]].update(n)
    return out


def source_bin(cell, G):
    dist = lambda c: math.dist(((c % 8 + 0.5) * 0.08, (c // 8 + 0.5) * 0.08), G["robot"].base)  # noqa: E731  (bd_2d.env's own metric)
    edge = G.setdefault("_source_edges", {b: max(dist(c) for c in v) for b, v in G["region"].items()})
    return "near" if dist(cell) <= edge["near"] else "mid" if dist(cell) <= edge["mid"] else "far"


def allocate(n, sizes):
    k = {b: n // 3 + (i < n % 3) for i, b in enumerate(sizes)}
    k = {b: min(k[b], sizes[b]) for b in sizes}
    while sum(k.values()) < n and any(k[b] < sizes[b] for b in sizes):
        for b in sizes:
            if sum(k.values()) < n and k[b] < sizes[b]:
                k[b] += 1
    return k


def dests(key, src, pool, n):
    """Stratified destinations, interleaved near/mid/far so that any leading slice stays stratified: (cell, bin, k, n_bin)."""
    bins = {b: sorted((c for c in pool if lo <= cheb(c, src) <= hi), key=lambda c: H(key, c)) for b, (lo, hi) in PLAN["dest_bins"].items()}
    k = allocate(n, {b: len(v) for b, v in bins.items()})
    return [(bins[b][i], b, k[b], len(bins[b])) for i in range(max(k.values(), default=0)) for b in bins if i < k[b]]


def root_plan(rid, r, G, pilot):
    goal, legal = r["goal_cell"], set(r["scene"]["legal_cells"])
    cells0 = {o["id"]: o["cell"] for o in r["scene"]["objects"]}
    movers = sorted(o for o in cells0 if o != "t0")
    pool = lambda cells, extra=(): sorted(c for c in legal if c not in cells.values() and c != goal and c not in extra)  # noqa: E731
    rows = []

    def add(block, prefix, kind, o, cells, d, dbin, k, n, pil, static=False, ranks=(0, 0)):
        if static and TRIAL(rid, prefix, o, d) in {x[0] for x in rows}:
            return                                    # both edits vacated the same cell for the same object: one record
        rows.append([TRIAL(rid, prefix, o, d), block, prefix, kind, o, cells[o], source_bin(cells[o], G), d, dbin, k, n, pil, static, *ranks])
    bybin = defaultdict(list)
    for o in sorted(movers, key=lambda o: H(rid, "obj", o)):
        bybin[source_bin(cells0[o], G)].append(o)
    objs = [o for i in range(len(movers)) for b in ("near", "mid", "far") if i < len(bybin[b]) for o in [bybin[b][i]]][:PLAN["root_objects"]]
    for oi, o in enumerate(objs):
        for di, (d, b, k, n) in enumerate(dests((rid, "root", o), cells0[o], pool(cells0), PLAN["root_dests"])):
            add("root", [], None, o, cells0, d, b, k, n, pilot and oi < PLAN["pilot"]["root_objects"] and di < PLAN["pilot"]["root_dests"], ranks=(oi, di))
    edits = []
    order = sorted(movers, key=lambda o: H(rid, "edit", o))
    if order and pool(cells0):
        edits.append(("prefix1", "DESIGN_RANDOM_EDIT", order[0], min(pool(cells0), key=lambda c: H(rid, "edit1dest", c))))
        arch = sorted({tuple(b["sequence"][0]) for b in r["labels"]["B"] if b["witnesses"] and b["sequence"] and b["sequence"][0][0] in movers
                       and b["sequence"][0][1] in pool(cells0)} - {edits[0][2:]}, key=lambda m: H(rid, "edit2", *m))
        if arch:
            edits.append(("prefix2", "ARCHIVED_FIRST_MOVE", *arch[0]))
        elif len(order) > 1:
            edits.append(("prefix2", "DESIGN_RANDOM_EDIT", order[1], min(pool(cells0), key=lambda c: H(rid, "edit2dest", c))))
    for block, kind, a, c in edits:
        cells, prefix = {**cells0, a: c}, [[a, c]]
        near = sorted((o for o in movers if o != a), key=lambda o: (cheb(cells0[o], cells0[a]), H(rid, block, o)))[:PLAN["prefix_objects"]]
        for oi, o in enumerate(near):
            vac = oi == 0 and cells0[a] != goal
            ds = ([(cells0[a], "vacated", 1, 1)] if vac else []) + dests((rid, block, o), cells[o], pool(cells, [cells0[a]] if vac else []),
                                                                        PLAN["prefix_dests"] - vac)
            for di, (d, b, k, n) in enumerate(ds):
                add(block, prefix, kind, o, cells, d, b, k, n, pilot and oi == 0 and di < PLAN["pilot"]["prefix_dests"], ranks=(oi, di))
            if vac:                                   # the same object and cell before the edit: occupied, a legality record
                add("contrast_" + block, [], None, o, cells0, cells0[a], "vacated", 1, 1, pilot, True)
    assert len({x[0] for x in rows}) == len(rows), rid
    return rows


def in_quota(row, q):
    """Rows of a quota: a leading slice of every block (objects and destinations are stored in rank order)."""
    r = dict(zip(COLS, row))
    if r["block"] == "root":
        return r["obj_rank"] < q["root_objects"] and r["dest_rank"] < q["root_dests"]
    k = int(r["block"][-1])                              # prefix1 / prefix2 / contrast_prefix1 / contrast_prefix2
    if r["static_ineligible"]:
        return k <= q["prefixes"] and q["prefix_objects"] > 0
    return k <= q["prefixes"] and r["obj_rank"] < q["prefix_objects"] and r["dest_rank"] < q["prefix_dests"]


def build():
    cls = selected()
    R = {x: P.load_root(x) for x in cls}
    G = P.A.env()
    groups = lineage(R)
    gof = {x: g for g, v in groups.items() for x in v}
    v2 = P.HERE.parent / "smoke_repro_v2" / "roster.json"
    touched = {t["root"] for t in json.loads(v2.read_text())} | {"2d-plus2d-srv-w0408-s006"}
    forced = {gof[x] for x in touched}
    split = splits(groups, cls, forced)
    nobs = {x: sum(o["id"] != "t0" for o in R[x]["scene"]["objects"]) for x in R}
    edges = sorted(nobs.values())
    clut = lambda x: 0 if nobs[x] <= edges[len(edges) // 3] else 1 if nobs[x] <= edges[2 * len(edges) // 3] else 2  # noqa: E731
    pilot = []
    for c in CLASSES:                                  # 5 per class: train, untouched group, one root per group, clutter caps 2/2/1
        used, cap = set(), Counter()
        cand = sorted((x for x in R if cls[x] == c and split[gof[x]] == "train" and gof[x] not in forced and nobs[x] >= PLAN["pilot"]["min_obstacles"]),
                      key=lambda x: H("pilot", x))
        for strict in (True, False):
            for x in cand:
                if len([p for p in pilot if cls[p] == c]) < PLAN["pilot"]["roots_per_class"] and gof[x] not in used and (not strict or cap[clut(x)] < (2, 2, 1)[clut(x)]):
                    pilot.append(x)
                    used.add(gof[x])
                    cap[clut(x)] += 1
    rows, plan = [], []
    for x in sorted(R, key=lambda x: H("order", x)):    # root order: hash-shuffled, so a partial run spans every class and split
        s = P.scene_from_dict(R[x]["scene"])
        q = root_plan(x, R[x], G, x in pilot)
        plan.append({"root_id": x, "split": split[gof[x]], "root_group_id": gof[x], "verified_class": cls[x], "root_file_sha256": P.sha_file(P.DATA / "raw" / "roots" / f"{x}.json"),
                     "original_scene_hash": P.scene_hash(s), "n_obstacles": nobs[x], "clutter_bin": clut(x), "queries": q})
        rows.append({"root_id": x, "verified_class": cls[x], "root_group_id": gof[x], "split": split[gof[x]], "dev_touched": x in touched, "pilot": x in pilot})
    return {"cls": cls, "groups": groups, "split": split, "rows": rows, "plan": plan, "pilot": pilot, "forced": sorted(forced), "R": R, "gof": gof}


def write(out, b):
    out = Path(out)
    sel_hash = P.sha_file(P.DATA / "selected.json")
    sm = {"version": PLAN["version"], "seed": PLAN["seed"], "source_selected_sha256": sel_hash, "algorithm": "lineage union-find (parent_family_id, identical "
          "object layout, conversion origin) -> class-stratified greedy over groups, largest first; dev-touched groups forced to train",
          "targets": PLAN["split_targets"], "forced_train_groups": b["forced"], "roots": b["rows"]}
    blob = json.dumps(sm, indent=0, sort_keys=True).encode()
    lines = "".join(json.dumps(p, separators=(",", ":"), sort_keys=True) + "\n" for p in b["plan"]).encode()
    pil = [{**dict(zip(COLS, q)), "root": p["root_id"], "split": p["split"], "verified_class": p["verified_class"]} for p in b["plan"] for q in p["queries"] if q[11]]
    files = {"split_manifest.json": blob, "query_plan.jsonl.gz": gzip.compress(lines, mtime=0),
             "pilot_roster.json": json.dumps(pil, indent=0, sort_keys=True).encode()}
    hashes = {"split_manifest_sha256": P.sha(blob), "query_plan_sha256": P.sha(lines), "pilot_roster_sha256": P.sha(files["pilot_roster.json"]),
              "source_selected_sha256": sel_hash, "plan_spec_sha256": P.sha(json.dumps([PLAN, QUOTAS], sort_keys=True).encode()), "plan_spec": PLAN,
              "quotas": QUOTAS, "columns": COLS}
    files["plan_hashes.json"] = json.dumps(hashes, indent=1, sort_keys=True).encode()
    for name, data in files.items():                    # frozen: never overwrite a differing file
        f = out / name
        if f.exists() and f.read_bytes() != data:
            raise SystemExit(f"FROZEN FILE DIFFERS: {f}")
    out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        if not (out / name).exists():
            (out / name).write_bytes(data)
    (out / "plan_summary.json").write_text(json.dumps(summary(b), indent=1, sort_keys=True))
    return hashes


def summary(b):
    cls, sp, gof = b["cls"], b["split"], b["gof"]
    per = {s: {"roots": Counter(), "groups": len({g for g, v in sp.items() if v == s})} for s in ("train", "val", "test")}
    q = defaultdict(Counter)
    for p in b["plan"]:
        per[p["split"]]["roots"][p["verified_class"]] += 1
        for r in p["queries"]:
            q[p["split"]][("static" if r[12] else r[1].replace("prefix1", "prefix").replace("prefix2", "prefix")) + ("_pilot" if r[11] else "")] += 1
            q[p["split"]]["dest_" + r[8]] += not r[12]
            q[p["split"]]["teacher_queries"] += not r[12]
            q[p["split"]]["pilot_teacher_queries"] += r[11] and not r[12]
    kinds = Counter(r[3] for p in b["plan"] for r in p["queries"] if r[3])
    leak = {k: sum(len({sp[gof[x]] for x in v}) > 1 for v in idx.values()) for k, idx in (
        ("family", _index(b["R"], lambda r: r["ids"]["parent_family_id"])), ("layout", _index(b["R"], lambda r: json.dumps(sorted((o["id"], o["cell"]) for o in r["scene"]["objects"])))),
        ("conversion", _index(b["R"], lambda r: r["construction"].get("converted_from") or r["ids"]["root_id"])))}
    nq = [sum(not r[12] for r in p["queries"]) for p in b["plan"]]
    return {"roots": len(b["plan"]), "groups": len(b["groups"]), "largest_group": max(len(v) for v in b["groups"].values()),
            "per_split": {s: {"roots": sum(v["roots"].values()), "by_class": dict(v["roots"]), "groups": v["groups"], "queries": dict(q[s])} for s, v in per.items()},
            "cross_split_shared": leak, "prefix_kinds": dict(kinds), "teacher_queries_total": sum(nq), "roots_without_queries": sum(n == 0 for n in nq),
            "queries_per_root": {"mean": round(sum(nq) / len(nq), 2), "max": max(nq)}, "pilot_roots": b["pilot"],
            "pilot_by_class": dict(Counter(cls[x] for x in b["pilot"])), "pilot_teacher_queries": sum(q[s]["pilot_teacher_queries"] for s in q),
            "quota_teacher_queries": {k: sum(in_quota(r, v) and not r[12] for p in b["plan"] for r in p["queries"]) for k, v in QUOTAS.items()},
            "pilot_inside_every_quota": all(in_quota(r, v) for v in QUOTAS.values() for p in b["plan"] for r in p["queries"] if r[11])}


def _index(R, key):
    d = defaultdict(list)
    for x, r in R.items():
        d[key(r)].append(x)
    return d


if __name__ == "__main__":
    cmd, out = sys.argv[1], Path(sys.argv[2]).resolve()
    b = build()
    if cmd == "freeze":
        print(json.dumps(write(out, b), indent=1))
    elif cmd == "verify":
        old = json.loads((out / "plan_hashes.json").read_text())
        lines = "".join(json.dumps(p, separators=(",", ":"), sort_keys=True) + "\n" for p in b["plan"]).encode()
        same = old["query_plan_sha256"] == P.sha(lines) == P.sha(gzip.decompress((out / "query_plan.jsonl.gz").read_bytes()))
        print(json.dumps({"query_plan_reproduced": same, "selected_unchanged": old["source_selected_sha256"] == P.sha_file(P.DATA / "selected.json")}))
        sys.exit(0 if same else 1)
