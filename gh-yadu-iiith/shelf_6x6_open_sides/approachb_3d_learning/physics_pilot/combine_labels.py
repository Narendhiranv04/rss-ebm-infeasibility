"""Join verified outcomes into the AUTHENTIC 3D X / Y label sidecar (BD3DS66_X_V1 / BD3DS66_Y_REPLAY_V1 / BD3DS66_Y_PLAN_V1).

    python3 -I combine_labels.py RAW_DIR OUT.json --pair M1 S1 [--pair M2 S2 ...] [--r2 R2_MANIFEST R2_SUMMARY]

Y per (root, endpoint key): 1 if REPLAY_VERIFIED (stored witness) or Y1_PLANNED_EXECUTED_REPLAY_VERIFIED (R2); 0 if R2 attempted failure or the
archived operational negative of a record (CERTIFIED / NO_PLAN_FOUND, scope kept).  Planning-only positives NOT yet replayed stay MISSING.
X per (root, ordered key): 1 only if X1_VERIFIED; 0 for re-confirmed relocation failure or stop-connector failure (that stopping prefix only);
everything else MISSING.  A key verified twice must agree (else the row is flagged CONFLICT and excluded)."""
import argparse, gzip, json, collections
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("raw"); ap.add_argument("out"); ap.add_argument("--pair", nargs=2, action="append", default=[]); ap.add_argument("--r2", nargs=2)
a = ap.parse_args()
RAW = Path(a.raw)
ck = lambda c: f"{int(c[0])},{int(c[1])}"  # noqa: E731
load = lambda p: json.loads((gzip.open(p, "rt") if p.endswith(".gz") else open(p)).read())  # noqa: E731
roots, Y, X, conflict = {}, {}, {}, []


def raw(rid):
    if rid not in roots:
        roots[rid] = json.loads((RAW / f"roots/{rid}.json").read_text())
    return roots[rid]


def ekey(rid, edit):
    e = {k: tuple(v) for k, v in raw(rid)["initial_cells"].items()}
    e.update({k: tuple(v) for k, v in edit.items()})
    return "|".join(f"{o}@{ck(c)}" for o, c in sorted(e.items()))


def put(d, k, v, field):
    if k in d and d[k][field] != v[field]:
        conflict.append({"key": list(k), "a": d[k], "b": v}); d[k]["CONFLICT"] = True
    else:
        d.setdefault(k, v)


for mf, sf in a.pair:
    it = {x["item"]: x for x in load(mf)["items"]}
    for r in load(sf)["rows"]:
        x = it.get(r["item"])
        if x is None:
            continue
        if x["kind"] == "Y" and r["final"] == "Y1_REPLAY_VERIFIED":
            put(Y, (x["root_id"], ekey(x["root_id"], x["edit"])), {"y": 1, "status": "Y_REPLAY_VERIFIED", "split": x["split"], "item": r["item"]}, "y")
        if x["kind"] == "X" and r["final"].startswith(("X1", "X0")):
            put(X, (x["root_id"], x["x_key"]), {"x": 1 if r["final"].startswith("X1") else 0, "status": r["final"], "split": x["split"], "m": x["m"],
                                               "y_key": x["y_endpoint_key"], "item": r["item"], "failed_phase": r.get("failed_phase")}, "x")
if a.r2:
    it = {x["item"]: x for x in load(a.r2[0])["items"]}
    for r in load(a.r2[1])["rows"]:
        x = it[r["item"]]
        if r["final"] == "Y1_PLANNED_EXECUTED_REPLAY_VERIFIED":
            put(Y, (x["root_id"], x["y_key"]), {"y": 1, "status": "Y_PLAN_V1_EXECUTED_REPLAY_VERIFIED", "split": x["split"], "item": r["item"]}, "y")
        elif r["final"].startswith("Y0_ATTEMPTED"):
            put(Y, (x["root_id"], x["y_key"]), {"y": 0, "status": "Y_PLAN_V1_" + r["final"], "split": x["split"], "item": r["item"]}, "y")
for rid in {k[0] for k in X}:                                               # archived operational Y = 0 on the X roots' endpoints
    for r in [raw(rid)["labels"]["original"]] + raw(rid)["labels"]["D"]:
        if r["y"] == 0:
            Y.setdefault((rid, ekey(rid, r.get("edit", {}))), {"y": 0, "status": f"Y_OPERATIONAL_NEG_{r['negative_kind']}", "split": None})
X = {k: v for k, v in X.items() if not v.get("CONFLICT")}
Y = {k: v for k, v in Y.items() if not v.get("CONFLICT")}
cells, by_split = collections.Counter(), collections.Counter()
for k, v in X.items():
    y = Y.get((k[0], v["y_key"]))
    cells[f"{v['split']}|X{v['x']}/Y{'?' if y is None else y['y']}"] += 1
    by_split[(v["split"], v["x"])] += 1
groups = collections.defaultdict(list)
for k, v in X.items():
    if v["m"] >= 2:
        groups[(k[0], v["y_key"])].append((k[1], v["x"]))
dec = [g for g in groups.values() if len(g) >= 2]
out = {"sidecar": "BD3DS66_XY_SIDECAR", "sources": {"pairs": a.pair, "r2": a.r2}, "x_labels": len(X), "y_labels": len(Y), "conflicts": conflict,
       "x_by_split": {f"{s}|X{x}": n for (s, x), n in sorted(by_split.items())}, "xy_cells_by_split": dict(sorted(cells.items())),
       "y_status": dict(collections.Counter(v["status"].split("_NEG_")[0] if "_NEG_" in v["status"] else v["status"] for v in Y.values())),
       "same_endpoint_order_groups_decided": len(dec), "order_discordant_groups": sum(1 for g in dec if len({x for _, x in g}) > 1),
       "order_discordant_examples": [g for g in dec if len({x for _, x in g}) > 1][:10],
       "x_rows": [{"root_id": k[0], "x_key": k[1], **v} for k, v in sorted(X.items())],
       "y_rows": [{"root_id": k[0], "y_key": k[1], **v} for k, v in sorted(Y.items())]}
Path(a.out).write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if not k.endswith("_rows") and k != "order_discordant_examples"}, indent=1))
