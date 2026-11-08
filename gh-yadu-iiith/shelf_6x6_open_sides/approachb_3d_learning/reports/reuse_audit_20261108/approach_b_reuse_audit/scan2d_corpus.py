"""Read-only scan of the committed 2D 5,000-root class-quota corpus (BD2D_TEACHER_V1) for X/Y evidence reuse."""
import json, os, sys, collections as C

CQ, SPLITF, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
sel = json.load(open(f"{CQ}/selected.json"))["selected"]
split = {r["root_id"]: r["split"] for r in json.load(open(SPLITF))["roots"]}
cls = {rid: k for k, v in sel.items() for rid in v}
cnt = C.Counter()


def ekey(c0, ed):
    return tuple(sorted({**{k: int(v) for k, v in c0.items()}, **{k: int(v) for k, v in ed.items()}}.items()))


for rid, k in cls.items():
    sp = split.get(rid, "NO_SPLIT")
    r = json.load(open(f"{CQ}/raw/roots/{rid}.json"))
    L = r["labels"]
    c0 = {o["id"]: o["cell"] for o in r["scene"]["objects"]}
    park = {o: v[0]["cell"] for o, v in r["parking"].items() if v}
    t = sp
    cnt[f"{t}|roots"] += 1
    cnt[f"{t}|class_{k}"] += 1
    yk = {}
    for fam, y, ed in [("original", L["original"], {})] + [("D", d, d.get("edit", {})) for d in L["D"]]:
        cnt[f"{t}|Y|{fam}|attempts"] += 1
        cnt[f"{t}|Y|{fam}|" + ("pos" if y["y"] else f"neg_{y.get('negative_kind')}")] += 1
        cnt[f"{t}|Y|{fam}|witness_field_{'yes' if y.get('witness') else 'no'}"] += 1
        kk = ekey(c0, ed)
        if kk in yk:
            cnt[f"{t}|Y|dup_endpoint"] += 1
        yk[kk] = y["y"]
    cnt[f"{t}|Y|lattice_not_stored"] += max(0, L.get("edits_evaluated", 0) - len(L["D"]))
    g = C.defaultdict(list)
    for b in L["B"]:
        m = len(b["sequence"])
        if m == 0:
            cnt[f"{t}|B|empty_seq"] += 1
            continue
        fs = b["failed_step"]
        cat = "complete_success" if fs is None else "target_failed_after_all_relocs" if fs == "target" else "reloc_failed"
        cnt[f"{t}|B|attempted"] += 1
        cnt[f"{t}|B|{cat}"] += 1
        cnt[f"{t}|B|{cat}|witnesses_{'yes' if b.get('witnesses') else 'none'}"] += 1
        if cat == "reloc_failed":
            cnt[f"{t}|B|reloc_failed|{b.get('negative_kind')}"] += 1
        endk = ekey(c0, {o: d for o, d in b["sequence"]})
        ys = "stored" if endk in yk else "unknown"
        cnt[f"{t}|B|{cat}|endpointY_{ys}{yk.get(endk, '')}"] += 1
        g[endk].append((tuple((o, int(d)) for o, d in b["sequence"]), cat))
    for e, v in g.items():
        if len({s for s, _ in v}) >= 2:
            cnt[f"{t}|ABBA|groups"] += 1
            if len({c for _, c in v}) > 1:
                cnt[f"{t}|ABBA|reloc_or_composite_outcome_differs"] += 1
json.dump(dict(sorted(cnt.items())), open(OUT, "w"), indent=1)
print("roots", len(cls))
