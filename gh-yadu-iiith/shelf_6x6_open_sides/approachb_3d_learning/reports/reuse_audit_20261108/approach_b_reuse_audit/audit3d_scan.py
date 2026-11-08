"""Read-only 3D SHELF66V3 X/Y evidence-reuse scan.  Streams one root JSON at a time; reads witness NPZs only for
existence, content-hash re-verification (bd_core.Store.witness rule) and first/last joint rows.  Writes one JSON."""
import hashlib, json, os, sys, collections as C
import numpy as np

RAW = sys.argv[1]            # .../dataset_bd_classquota_without_home_position_no_overpass/raw
PV3 = sys.argv[2]            # .../dataset_placement_extension_v2/production_v3
OUT = sys.argv[3]

pin = json.load(open(f"{PV3}/frozen/source_pin_first_cohort.json"))["roots"]
split = json.load(open(f"{PV3}/frozen/split_manifest_first_cohort.json"))["split"]
qc = json.load(open(f"{PV3}/QC_ELIGIBLE_ROOTS.json"))["roots"]
WCACHE = {}


def wpath(w):
    return f"{RAW}/witnesses/{w[:2]}/{w}.npz"


def wcheck(w):
    """exists, content-hash == id, first q row, last q row, phases."""
    if w in WCACHE:
        return WCACHE[w]
    p = wpath(w)
    if not os.path.exists(p):
        WCACHE[w] = r = {"exists": False}
        return r
    z = np.load(p)
    h = hashlib.sha256()
    for k in sorted(z.files):
        v = np.ascontiguousarray(np.asarray(z[k]))
        h.update(f"{k}|{v.dtype.str}|{v.shape}".encode())
        h.update(v.tobytes())
    q = z["q"]
    r = {"exists": True, "hash_ok": h.hexdigest()[:24] == w, "q0": q[0].copy(), "q1": q[-1].copy(),
         "phases": [str(x) for x in z["phases"]], "n": int(q.shape[0])}
    WCACHE[w] = r
    return r


def ekey(cells0, edit):
    """Y endpoint identity: full object->cell arrangement after the edit (target and fixed structure unchanged)."""
    c = {**{k: tuple(v) for k, v in cells0.items()}, **{k: tuple(v) for k, v in edit.items()}}
    return tuple(sorted(c.items()))


cnt = C.Counter()
byk = C.defaultdict(C.Counter)
qref_rows, qref_src = [], []
bound_gap = []            # |first q of step j+1 - last q of step j| (planned end vs achieved start)
step0_vs_qref = []
ab = []
rows = []
files = sorted(os.listdir(f"{RAW}/roots"))
for fn in files:
    rid = fn[:-5]
    path = f"{RAW}/roots/{fn}"
    raw = open(path, "rb").read()
    rec = json.loads(raw)
    coh = rid in pin
    sp = split.get(rid, "NOT_IN_COHORT")
    cls = str(rec.get("verified_K"))
    tag = sp
    cnt[f"roots_{'cohort' if coh else 'noncohort'}"] += 1
    if coh:
        cnt["pin_sha_match" if hashlib.sha256(raw).hexdigest() == pin[rid] else "pin_sha_MISMATCH"] += 1
    L, cells0, goal = rec["labels"], rec["initial_cells"], rec["goal"]
    park = {o: tuple(v[0]) for o, v in rec["parking"].items() if v}
    R = {"root": rid, "split": sp, "K": cls, "n_movable": rec.get("n_movable"), "edits_evaluated": L.get("edits_evaluated", 0)}
    # ---------------- Y: stored target-only attempts (original + D)
    yrecs = [("original", L["original"], {})] + [("D", d, d.get("edit", {})) for d in L["D"]]
    ykeys = {}
    for fam, y, ed in yrecs:
        k = ekey(cells0, ed)
        cnt[f"{tag}|Y|{fam}|attempts"] += 1
        lab = "pos" if y["y"] == 1 else f"neg_{y.get('negative_kind')}"
        cnt[f"{tag}|Y|{fam}|{lab}"] += 1
        if fam == "D" and y.get("query_kind") != "SCENE_TARGET_ONLY":
            cnt[f"{tag}|Y|D|query_kind_other"] += 1
        if fam == "D" and any(tuple(v) != park.get(o) for o, v in ed.items()):
            cnt[f"{tag}|Y|D|edit_not_first_parking_cell"] += 1
        if y["y"] == 1:
            w = y.get("witness")
            wc = wcheck(w) if w else {"exists": False}
            cnt[f"{tag}|Y|{fam}|pos_witness_{'present' if wc['exists'] else 'MISSING'}"] += 1
            if wc.get("exists"):
                cnt[f"{tag}|Y|{fam}|pos_witness_hash_{'ok' if wc['hash_ok'] else 'BAD'}"] += 1
                qref_rows.append(wc["q0"]); qref_src.append(fam)
                cnt[f"Y_witness_ends_{wc['phases'][-1]}"] += 1
        if k in ykeys:
            cnt[f"{tag}|Y|duplicate_endpoint_within_root"] += 1
            if ykeys[k] != y["y"]:
                cnt[f"{tag}|Y|duplicate_endpoint_CONFLICT"] += 1
        ykeys.setdefault(k, y["y"])
    cnt[f"{tag}|Y|unique_endpoints"] += len(ykeys)
    cnt[f"{tag}|Y|lattice_attempts_not_stored"] += max(0, L.get("edits_evaluated", 0) - len(L["D"]))
    # lattice-summary Y (blocker): sets with y=1 known (minimal + reachable clearing), sets w/o any minimal subset => y=0
    bl = L.get("blocker", {})
    lat_pos = set()
    if bl.get("full_lattice") and not L["original"]["y"]:
        for s in bl.get("minimal_sets", []):
            lat_pos.add(frozenset(s))
        for v in (bl.get("clearing_sets_reachable") or {}).values():
            for s in v:
                lat_pos.add(frozenset(s))
    mins = [frozenset(s) for s in bl.get("minimal_sets", [])] if bl.get("full_lattice") else None
    scope = set(bl.get("scope", []))

    def y_of_set(S):
        """Y for 'objects in S at their first parking cell': (value, source)."""
        if not S:
            return L["original"]["y"], "original"
        k = ekey(cells0, {o: park[o] for o in S})
        if k in ykeys:
            return ykeys[k], "stored_D"
        if L["original"]["y"]:
            return None, "unknown"
        if frozenset(S) in lat_pos:
            return 1, "lattice_summary"
        if mins is not None and set(S) <= scope and not any(m <= set(S) for m in mins):
            return 0, "lattice_summary"
        return None, "unknown"
    # ---------------- K0: executed target from q_ref (B minimal, empty sequence)
    for b in L["B"]:
        if b["role"] == "minimal" and not b["sequence"]:
            cnt[f"{tag}|K0exec|n"] += 1
            cnt[f"{tag}|K0exec|y{b['y']}"] += 1
            if b["witnesses"] and L["original"].get("witness") == b["witnesses"][0]:
                cnt[f"{tag}|K0exec|same_witness_as_original"] += 1
    # ---------------- B: ordered sequences
    seen_prefix = {}
    seqs = []
    for b in L["B"]:
        m = len(b["sequence"])
        if m == 0:
            continue
        fs = b["failed_step"]
        nrel_ok = m if fs in (None, "target") else fs
        cnt[f"{tag}|B|attempted"] += 1
        cnt[f"{tag}|B|m{m}"] += 1
        cnt[f"{tag}|B|role_{b['role']}"] += 1
        cat = "complete_all_relocs" if fs in (None, "target") else "reloc_failed"
        cnt[f"{tag}|B|{cat}"] += 1
        cnt[f"{tag}|B|composite_y{b['y']}"] += 1
        if fs not in (None, "target"):
            st = b["steps"][fs]
            why = b["reason"].split(":")[0]
            cnt[f"{tag}|B|reloc_fail_reason|{why}|{b.get('negative_kind')}"] += 1
        wits = b["witnesses"]
        wcs = [wcheck(w) for w in wits]
        # witnesses for every executed relocation step
        exec_steps = nrel_ok + (1 if fs not in (None, "target") and len(wits) > fs else 0)
        rel_w_ok = len(wits) >= nrel_ok and all(wc["exists"] and wc["hash_ok"] for wc in wcs[:nrel_ok])
        cnt[f"{tag}|B|reloc_witnesses_{'all_present_hash_ok' if rel_w_ok else 'INCOMPLETE'}"] += 1
        # robot state carried: step j+1 starts from achieved end of step j
        for j in range(min(len(wcs), m + 1) - 1):
            if wcs[j]["exists"] and wcs[j + 1]["exists"]:
                bound_gap.append(float(np.max(np.abs(wcs[j + 1]["q0"] - wcs[j]["q1"]))))
        if wcs and wcs[0]["exists"]:
            step0_vs_qref.append(wcs[0]["q0"])
        # terminal configuration after last relocation recorded? (= first q of the target-step witness)
        term = cat == "complete_all_relocs" and len(wits) == m + 1 and wcs[m]["exists"]
        if cat == "complete_all_relocs":
            cnt[f"{tag}|B|complete|terminal_q_recorded_via_target_witness" if term else f"{tag}|B|complete|terminal_q_only_planned_end_of_last_reloc"] += 1
            cnt[f"{tag}|B|complete|target_from_achieved_y{b['y']}"] += 1
        # endpoint Y availability for the full sequence
        S = [o for o, _ in b["sequence"]]
        dest_park = all(tuple(d) == park.get(o) for o, d in b["sequence"])
        endk = ekey(cells0, {o: d for o, d in b["sequence"]})
        if endk in ykeys:
            ysrc, yv = "stored", ykeys[endk]
        elif dest_park:
            yv, ysrc = y_of_set(S)
        else:
            yv, ysrc = None, "unknown_nonpark_dest"
        cnt[f"{tag}|B|endpointY_{ysrc}_{yv}"] += 1
        seqs.append({"role": b["role"], "seq": tuple((o, tuple(d)) for o, d in b["sequence"]), "end": endk, "cat": cat, "m": m,
                     "rel_w_ok": rel_w_ok, "yv": yv, "ysrc": ysrc, "nrel_ok": nrel_ok})
        # prefixes whose relocations all executed (connector-only candidates), dedup by ordered prefix
        for j in range(1, nrel_ok + 1):
            pf = tuple((o, tuple(d)) for o, d in b["sequence"][:j])
            if pf in seen_prefix:
                continue
            pk = ekey(cells0, dict(pf))
            pdp = all(tuple(d) == park.get(o) for o, d in pf)
            if pk in ykeys:
                pyv, pys = ykeys[pk], "stored"
            elif pdp:
                pyv, pys = y_of_set([o for o, _ in pf])
            else:
                pyv, pys = None, "unknown_nonpark_dest"
            seen_prefix[pf] = (rel_w_ok, pys, pyv, j == m)
        # X=0 direct: failed relocation -> full sequence X=0 (and every extension of the failed prefix)
    for pf, (wok, pys, pyv, full) in seen_prefix.items():
        cnt[f"{tag}|Xcand|unique_prefix"] += 1
        cnt[f"{tag}|Xcand|{'full_seq' if full else 'proper_prefix_only'}"] += 1
        cnt[f"{tag}|Xcand|witness_{'ok' if wok else 'INCOMPLETE'}"] += 1
        cnt[f"{tag}|Xcand|Y_{pys}_{pyv}"] += 1
    # unique ordered sequences that failed a relocation (direct X=0 under the historical relocation policy)
    x0 = {s["seq"] for s in seqs if s["cat"] == "reloc_failed"}
    cnt[f"{tag}|X0direct|unique_seq"] += len(x0)
    # AB/BA same-endpoint order contrasts among stored B sequences
    g = C.defaultdict(list)
    for s in seqs:
        g[s["end"]].append(s)
    for end, v in g.items():
        orders = {s["seq"] for s in v}
        if len(orders) >= 2:
            cnt[f"{tag}|ABBA|groups"] += 1
            outs = {s["seq"]: s["cat"] for s in v}
            if len(set(outs.values())) > 1:
                cnt[f"{tag}|ABBA|groups_reloc_outcome_differs"] += 1
            ab.append({"root": rid, "split": sp, "orders": [[list(map(list, x)) for x in o] for o in orders], "cats": list(outs.values()),
                       "Y": v[0]["yv"], "Ysrc": v[0]["ysrc"]})
    # attempted-but-unstored BFS sequences
    tries = bl.get("sequence_tries", 0) or 0
    bfs_min = 1 if (rec.get("verified_K") not in (None, 0) and any(b["role"] == "minimal" for b in L["B"])) else 0
    cnt[f"{tag}|B|bfs_seq_tries"] += tries
    cnt[f"{tag}|B|bfs_failed_tries_not_stored"] += max(0, tries - bfs_min)
    cnt[f"{tag}|teacher_calls_execute"] += rec.get("teacher_calls", {}).get("execute", 0)
    byk[f"{tag}|K{cls}"]["roots"] += 1
    byk[f"{tag}|K{cls}"]["B_seq"] += len(seqs)
    byk[f"{tag}|K{cls}"]["Xcand_prefix"] += len(seen_prefix)
    byk[f"{tag}|K{cls}"]["Y_stored"] += len(yrecs)
    byk[f"{tag}|K{cls}"]["ABBA"] += sum(1 for v in g.values() if len({s['seq'] for s in v}) >= 2)

qr = np.array(qref_rows)
s0 = np.array(step0_vs_qref)
out = {"counts": dict(sorted(cnt.items())), "by_class": {k: dict(v) for k, v in sorted(byk.items())},
       "qref": {"n_Y_witness_first_rows": int(len(qr)), "max_dev_from_first": float(np.max(np.abs(qr - qr[0]))) if len(qr) else None,
                "q_ref": qr[0].round(6).tolist() if len(qr) else None,
                "B_step0_max_dev_from_qref": float(np.max(np.abs(s0 - qr[0]))) if len(s0) else None, "n_B_step0": int(len(s0))},
       "boundary_gap_rad": {"n": len(bound_gap), "max": max(bound_gap) if bound_gap else None,
                            "p50": float(np.median(bound_gap)) if bound_gap else None, "p99": float(np.quantile(bound_gap, .99)) if bound_gap else None,
                            "n_exact_zero": int(sum(1 for x in bound_gap if x == 0.0))},
       "witness_files_checked": len(WCACHE), "witness_files_on_disk": sum(len(f) for _, _, f in os.walk(f"{RAW}/witnesses")),
       "abba_examples": ab[:40], "abba_total": len(ab)}
json.dump(out, open(OUT, "w"), indent=1, default=str)
print("done", len(files), "roots;", len(WCACHE), "witnesses checked")
