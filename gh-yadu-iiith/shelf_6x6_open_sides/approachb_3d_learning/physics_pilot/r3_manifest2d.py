"""R3 (PREPARATION ONLY): predeclared candidate-diverse 2D ordered X/Y pilot manifest on Q2 corpus roots (first X/Y on the sampler's roots).

    python3 -I r3_manifest2d.py CORPUS_ROOTS_DIR Q2_SPLIT_MANIFEST.json OUT.json

The Q2 placement sampler is not trained yet (Q2 not merged), so destinations use the DECLARED NON-LEARNING proposal baseline: band-stratified
(Chebyshev near 1 / mid 2-3 / far >= 4 from the object's current cell, round robin, seeded) and uniform-legal (seeded), both drawn in the
CURRENT branch-local scene; never parking / intended / blocker fields.  When the merged Q2 sampler exists, the same design is regenerated with
learned proposals and kept as a SEPARATE manifest.  Roots: Q2 split train / val, not pilot, not dev_touched; 8 train + 2 val per verified class
K0..K3 by sha256('R3_CD2D|'+root).  Designs as in r3_manifest3d.py: NO_OP, ONE_MOVE, ALT_DEST, AB / BA (B re-sampled in A's scene, both orders
legal), OMIT (A alone, B alone), TRIPLE (2 orders), NUISANCE.  Teacher for the future run: the 2D BXY_X_TEACHER_V1 / BXY_Y_TEACHER_V1 (measured
8.1 CPU-s per query incl. replay, P-min)."""
import hashlib, json, sys, collections
from pathlib import Path

ROOTS, SPLIT, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
sp = json.loads(SPLIT.read_text())["roots"]
H = lambda t: int(hashlib.sha256(t.encode()).hexdigest()[:12], 16)  # noqa: E731
cheb = lambda a, b: max(abs(a // 8 - b // 8), abs(a % 8 - b % 8))  # noqa: E731
cand = collections.defaultdict(list)
for r in sp:
    if r["split"] in ("train", "val") and not r["pilot"] and not r["dev_touched"] and r["verified_class"] in ("0", "1", "2", "3"):
        cand[(r["verified_class"], r["split"])].append(r["root_id"])
chosen = [rid for k in ("0", "1", "2", "3") for s, n in (("train", 8), ("val", 2)) for rid in sorted(cand[(k, s)], key=lambda r: H(f"R3_CD2D|{r}"))[:n]]
split_of = {r["root_id"]: (r["split"], r["verified_class"], r["root_group_id"]) for r in sp}


def propose(cells, legal, o, source, tag, k=2):
    free = [c for c in legal if c not in set(cells.values())]
    if not free:
        return []
    if source == "uniform":
        return sorted(free, key=lambda c: H(f"{tag}|U|{c}"))[:k]
    src = cells[o]
    bands = [[c for c in free if cheb(c, src) == 1], [c for c in free if 2 <= cheb(c, src) <= 3], [c for c in free if cheb(c, src) >= 4]]
    out, j = [], H(tag) % 3
    while len(out) < k and any(bands):
        b = bands[j % 3]
        if b:
            c = min(b, key=lambda c: H(f"{tag}|S|{c}"))
            b.remove(c); out.append(c)
        j += 1
    return out


rows, ny = [], {}
for rid in chosen:
    d = json.loads((ROOTS / f"{rid}.json").read_text())
    objs = d["scene"]["objects"]
    legal = d["scene"]["legal_cells"]
    c0 = {o["id"]: o["cell"] for o in objs}
    tgt = next(o["id"] for o in objs if o["kind"] != "obstacle")
    movers = sorted((o for o in c0 if o != tgt), key=lambda o: H(f"R3M2|{rid}|{o}"))[:4]
    s, cls, grp = split_of[rid]

    def add(seq, kind, sources):
        cur = dict(c0)
        for o, dd in seq:
            if dd in set(cur.values()) or dd not in legal:
                return
            cur[o] = dd
        key = ".".join(f"{o}>{dd}" for o, dd in seq)
        yk = "|".join(f"{o}@{c}" for o, c in sorted(cur.items()))
        for r in rows:
            if r["root_id"] == rid and r["x_key"] == key:
                r["tags"] = sorted(set(r["tags"]) | {kind}); return
        ny.setdefault((rid, yk), len(ny))
        rows.append({"root_id": rid, "split": s, "root_group_id": grp, "verified_class_reporting_only": cls, "x_key": key, "m": len(seq), "tags": [kind],
                     "destination_sources": sources, "y_endpoint_key": yk, "construction_informed": False})
    rows.append({"root_id": rid, "split": s, "root_group_id": grp, "verified_class_reporting_only": cls, "x_key": "", "m": 0, "tags": ["NO_OP"], "destination_sources": [],
                 "y_endpoint_key": "|".join(f"{o}@{c}" for o, c in sorted(c0.items())), "construction_informed": False})
    first = {}
    for o in movers:
        for src in ("strat", "uniform"):
            ds = propose(c0, legal, o, src, f"{rid}|{o}|{src}|root")
            for j, dd in enumerate(ds):
                add([(o, dd)], "ONE_MOVE" if j == 0 else "ALT_DEST", [src])
            first.setdefault(o, {})[src] = ds[0] if ds else None
    for i in range(min(3, len(movers))):
        for j in range(i + 1, min(3, len(movers))):
            A, B = movers[i], movers[j]
            for src in ("strat", "uniform"):
                dA = first[A].get(src)
                if dA is None:
                    continue
                dB = next((x for x in propose({**c0, A: dA}, legal, B, src, f"{rid}|{B}|{src}|after|{A}", 4) if x != c0[A]), None)
                if dB is None:
                    continue
                add([(A, dA), (B, dB)], "PAIR_AB", [src, f"{src}@branch"]); add([(B, dB), (A, dA)], "PAIR_BA_SAME_ENDPOINT", [src, f"{src}@branch"])
                add([(A, dA)], "OMIT_B", [src]); add([(B, dB)], "OMIT_A_B_WITHOUT_A", [f"{src}@branch"])
    if len(movers) >= 3:
        A, B, C = movers[:3]
        dA = first[A].get("strat")
        dB = dA is not None and next((x for x in propose({**c0, A: dA}, legal, B, "strat", f"{rid}|T|B", 4) if x != c0[A]), None)
        dC = dB and next((x for x in propose({**c0, A: dA, B: dB}, legal, C, "strat", f"{rid}|T|C", 4) if x not in (c0[A], c0[B])), None)
        if dC is not None and dC is not False:
            add([(A, dA), (B, dB), (C, dC)], "TRIPLE", ["strat", "strat@branch", "strat@branch"])
            add([(C, dC), (B, dB), (A, dA)], "TRIPLE_REVERSED", ["strat@branch"] * 3)
    if len(movers) == 4:
        N, A, B = movers[3], movers[0], movers[1]
        dN, dA = first[N].get("uniform"), first[A].get("strat")
        dB = (dN is not None and dA is not None) and next((x for x in propose({**c0, N: dN, A: dA}, legal, B, "strat", f"{rid}|N|B", 4) if x not in (c0[A], c0[N])), None)
        if dB:
            add([(N, dN), (A, dA), (B, dB)], "NUISANCE_PLUS_PAIR", ["uniform", "strat", "strat@branch"])
x = [r for r in rows if r["m"] > 0]
man = {"manifest": "R3_CANDIDATE_DIVERSE_2D_V1", "status": "PREPARED_NOT_EXECUTED (no physics authorized)", "proposal_baseline": "NON-LEARNING (band-stratified + uniform)",
       "roots": len(chosen), "roots_by_split": dict(collections.Counter(split_of[r][0] for r in chosen)), "x_queries": len(x),
       "unique_y_endpoints": len({(r["root_id"], r["y_endpoint_key"]) for r in rows}), "by_tag": dict(collections.Counter(t for r in x for t in r["tags"])),
       "by_m": dict(collections.Counter(r["m"] for r in x)),
       "same_endpoint_order_groups": sum(1 for v in collections.Counter((r["root_id"], r["y_endpoint_key"]) for r in x if r["m"] >= 2).values() if v >= 2),
       "cost_estimate_cpu_h": round((len(x) + len({(r['root_id'], r['y_endpoint_key']) for r in rows})) * 8.1 / 3600, 1),
       "note": "Q2 roots share no root with historical 2D X/Y; these would be the first X/Y on the placement sampler's roots", "rows": rows}
man["sha256_of_rows"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
OUT.write_text(json.dumps(man, indent=1))
print(json.dumps({k: v for k, v in man.items() if k != "rows"}, indent=1))
