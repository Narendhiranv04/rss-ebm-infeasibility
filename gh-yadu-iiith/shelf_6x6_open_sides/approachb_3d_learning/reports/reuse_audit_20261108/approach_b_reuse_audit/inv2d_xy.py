"""Read-only inventory of committed/archived 2D Approach B X/Y teacher rows (BXY_X/Y_TEACHER_V1 family)."""
import glob, json, os, sys, collections as C

BASE = sys.argv[1]           # .../runs/approach_b
OUT = sys.argv[2]
COLS = {"P9_pilot": "bxy_teacher_pilot_v1/results", "PMIN": "bxy_pmin_v1/results", "P15_pilot": "bxy_p15_v1/results",
        "P16_wA": "bxy_p16_v1/wA/results", "P16_wB": "bxy_p16_v1/wB/results", "P16_wC": "bxy_p16_v1/wC/results",
        "P20_k1": "bxy_p20_v1/k1/results", "P20_k2": "bxy_p20_v1/k2/results", "P20_k3": "bxy_p20_v1/k3/results",
        "P24": "bxy_p24_v1/results"}
cnt = C.Counter()
X, Y = {}, {}               # key -> (collection, row-summary); first occurrence wins, duplicates counted
fields = C.defaultdict(C.Counter)
roots = C.defaultdict(set)
for col, rel in COLS.items():
    for f in sorted(glob.glob(f"{BASE}/{rel}/*/*.json")):
        kind = "x" if f.endswith(".x.json") else "y" if f.endswith(".y.json") else None
        if kind is None:
            cnt[f"{col}|other_file"] += 1
            continue
        r = json.load(open(f))
        if "row" in r and isinstance(r["row"], dict):
            r = r["row"]
        roots[col].add(r.get("root"))
        for k in ("side", "source", "teacher", "classes"):
            fields[f"{col}|{kind}|{k}"][json.dumps(r.get(k))[:60]] += 1
        key = r.get(f"{kind}_key")
        att, obs = r.get("attempted"), r.get("observed")
        st = "unattempted" if not att else ("censored" if not obs else f"y{r.get('y')}")
        cnt[f"{col}|{kind}|{st}"] += 1
        if st == "y0":
            cnt[f"{col}|{kind}|neg|{r.get('negative_kind')}|{str(r.get('failed_phase')).split(':')[0]}"] += 1
        if st == "y1":
            cnt[f"{col}|{kind}|pos_replay_{r.get('independent_replay')}"] += 1
            if kind == "x":
                cnt[f"{col}|x|pos_connector_witness_{'yes' if r.get('witness_connector') else 'NO'}"] += 1
                cnt[f"{col}|x|pos_qref_eq_q0_{r.get('q0') == r.get('q_ref')}"] += 1
            else:
                cnt[f"{col}|y|pos_target_witness_{'yes' if r.get('target_witness') else 'NO'}"] += 1
        if key is None:
            cnt[f"{col}|{kind}|NO_KEY"] += 1
            continue
        D = X if kind == "x" else Y
        summ = {"col": col, "root": r.get("root"), "st": st, "y_key": r.get("parent_y_key"),
                "acts": [tuple(a) for a in (r.get("ordered_actions") or [])], "side": r.get("side"), "source": r.get("source"),
                "nk": r.get("negative_kind"), "fp": r.get("failed_phase")}
        if key in D:
            cnt[f"{col}|{kind}|dup_key_of_{D[key]['col']}"] += 1
            if D[key]["st"] != st:
                cnt[f"{col}|{kind}|dup_key_STATUS_DIFFERS"] += 1
            continue
        D[key] = summ
# de-duplicated totals
tot = C.Counter()
for k, v in X.items():
    tot[f"X|{v['col']}|{v['st']}"] += 1
    tot[f"X|ALL|{v['st']}"] += 1
    tot[f"X|ALL|len{len(v['acts'])}|{v['st']}"] += 1
for k, v in Y.items():
    tot[f"Y|{v['col']}|{v['st']}"] += 1
    tot[f"Y|ALL|{v['st']}"] += 1
# four-cell on decided pairs; AB/BA groups (same Y key, >=2 decided X with the same action multiset, different order)
four = C.Counter()
grp = C.defaultdict(list)
for k, v in X.items():
    yk = v["y_key"]
    yv = Y.get(yk, {}).get("st") if yk else None
    four[(v["col"].split("_")[0], v["st"], yv)] += 1
    if yk:
        grp[yk].append(v)
ab = C.Counter()
for yk, v in grp.items():
    dec = [x for x in v if x["st"] in ("y0", "y1")]
    sets = C.defaultdict(list)
    for x in dec:
        if len(x["acts"]) >= 2:
            sets[tuple(sorted(x["acts"]))].append(x)
    for s, xs in sets.items():
        if len({tuple(x["acts"]) for x in xs}) >= 2:
            c = xs[0]["col"].split("_")[0]
            ab[(c, "same_endpoint_multi_order_groups")] += 1
            if len({x["st"] for x in xs}) > 1:
                ab[(c, "X_differs_across_orders")] += 1
            yst = Y.get(yk, {}).get("st")
            ab[(c, f"shared_Y_{yst}")] += 1
out = {"per_file_counts": dict(sorted(cnt.items())), "dedup_totals": dict(sorted(tot.items())),
       "four_cell_by_collection": {"|".join(map(str, k)): n for k, n in sorted(four.items(), key=str)},
       "abba": {"|".join(k): n for k, n in sorted(ab.items())},
       "roots_per_collection": {k: len(v) for k, v in roots.items()},
       "roots_total_unique": len(set().union(*roots.values())),
       "field_values": {k: dict(v.most_common(8)) for k, v in sorted(fields.items())},
       "unique_X": len(X), "unique_Y": len(Y)}
json.dump(out, open(OUT, "w"), indent=1)
print("unique X", len(X), "unique Y", len(Y))
