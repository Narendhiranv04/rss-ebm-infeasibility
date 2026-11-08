"""Join S1 / S2 verified outcomes into AUTHENTIC 3D X / Y label sidecar rows (BD3DS66_X_V1 / BD3DS66_Y_REPLAY_V1) and count (X, Y) cells.

    python3 -I combine_labels.py RAW_DIR S1_MANIFEST S1_SUMMARY S2_MANIFEST S2_SUMMARY OUT_SIDECAR.json

Y per (root, endpoint key):  1 only if REPLAY_VERIFIED here;  0 if the archived operational negative (CERTIFIED / NO_PLAN_FOUND, scope kept)
— reconfirmed or not, archived planning negatives are valid operational zeros (Nov 5 §11.2);  planning-only positives NOT yet replayed are
MISSING (never 1).  X per (root, ordered key): 1 only if X1_VERIFIED; 0 for reconfirmed relocation failure or stop-connector failure (scope:
this stopping prefix only, never its extensions); everything else MISSING."""
import json, sys, collections
from pathlib import Path

RAW = Path(sys.argv[1])
pairs = [(json.loads(Path(sys.argv[i]).read_text()), json.loads(Path(sys.argv[i + 1]).read_text())) for i in (2, 4) if Path(sys.argv[i + 1]).exists()]
ck = lambda c: f"{int(c[0])},{int(c[1])}"  # noqa: E731
Y, X, roots = {}, {}, {}
for man, summ in pairs:
    it = {x["item"]: x for x in man["items"]}
    for r in summ["rows"]:
        x = it[r["item"]]
        if x["kind"] == "Y" and r["final"] == "Y1_REPLAY_VERIFIED":
            d = roots.setdefault(x["root_id"], json.loads((RAW / f"roots/{x['root_id']}.json").read_text()))
            e = {k: tuple(v) for k, v in d["initial_cells"].items()}; e.update({k: tuple(v) for k, v in x["edit"].items()})
            Y[(x["root_id"], "|".join(f"{o}@{ck(c)}" for o, c in sorted(e.items())))] = {"y": 1, "status": "Y_REPLAY_VERIFIED", "split": x["split"], "item": r["item"]}
        if x["kind"] == "X" and r["final"].startswith(("X1", "X0")):
            X[(x["root_id"], x["x_key"])] = {"x": 1 if r["final"].startswith("X1") else 0, "status": r["final"], "split": x["split"], "m": x["m"],
                                             "y_key": x["y_endpoint_key"], "item": r["item"], "failed_phase": r.get("failed_phase")}
for rid in {k[0] for k in X}:                                               # archived operational Y = 0 on the X roots' endpoints
    d = roots.setdefault(rid, json.loads((RAW / f"roots/{rid}.json").read_text()))
    c0 = {k: tuple(v) for k, v in d["initial_cells"].items()}
    for r in [d["labels"]["original"]] + d["labels"]["D"]:
        if r["y"] == 0:
            e = dict(c0); e.update({k: tuple(v) for k, v in r.get("edit", {}).items()})
            Y.setdefault((rid, "|".join(f"{o}@{ck(c)}" for o, c in sorted(e.items()))), {"y": 0, "status": f"Y_OPERATIONAL_NEG_{r['negative_kind']}", "split": None})
cells, by_split = collections.Counter(), collections.Counter()
for k, v in X.items():
    y = Y.get((k[0], v["y_key"]))
    cells[f"X{v['x']}/Y{'?' if y is None else y['y']}"] += 1
    by_split[(v["split"], v["x"])] += 1
groups = collections.defaultdict(list)
for k, v in X.items():
    if v["m"] >= 2:
        groups[(k[0], v["y_key"])].append(v["x"])
dec = [g for g in groups.values() if len(g) >= 2]
out = {"sidecar": "BD3DS66_XY_SIDECAR_V0 (S1 + S2 chunks run so far)", "x_labels": len(X), "y_labels": len(Y),
       "x_by_split": {f"{a}|X{b}": n for (a, b), n in sorted(by_split.items())}, "xy_cells": dict(cells),
       "y_status": dict(collections.Counter(v["status"].split("_NEG_")[0] if "NEG" in v["status"] else v["status"] for v in Y.values())),
       "same_endpoint_order_groups_decided": len(dec), "order_discordant_groups": sum(1 for g in dec if len(set(g)) > 1),
       "x_rows": [{"root_id": k[0], "x_key": k[1], **v} for k, v in sorted(X.items())],
       "y_rows": [{"root_id": k[0], "y_key": k[1], **v} for k, v in sorted(Y.items())]}
Path(sys.argv[6]).write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if not k.endswith("_rows")}, indent=1))
