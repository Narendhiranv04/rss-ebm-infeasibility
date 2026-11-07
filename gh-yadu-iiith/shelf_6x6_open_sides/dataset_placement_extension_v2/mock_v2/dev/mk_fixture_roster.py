"""Dev only: a roster of the 5 V1 fixture queries (training roots outside both mock rosters) for collector end-to-end checks."""
import gzip, json, sys
sys.path.insert(0, "code")
import plan3d as L
fx = json.load(open("../dataset_placement_extension_v1/mock_v1/configs/fixtures.json"))["fixtures"]
rows = []
for kind, f in fx.items():
    rid = f["root_id"]
    raw = json.load(open(f"../dataset_bd_classquota_without_home_position_no_overpass/raw/roots/{rid}.json"))
    cells = {k: tuple(v) for k, v in f["cells"].items()}
    rows.append({"qid": f"{rid}:fx_{kind}", "root_id": rid, "K": "fx", "slot": len(rows), "depth": f.get("depth", 0),
                 "scene_hash": L.scene_hash(raw["objects"], cells), "cells": f["cells"], "object": f["object"], "dest": f["dest"], "band": "fx",
                 "shape": "fx", "cheb": L.cheb(f["dest"], cells[f["object"]]), "n_obj": len(cells), "static": None, "split": "train", "reference": True,
                 "dest_is_t_goal": tuple(f["dest"]) == tuple(raw["goal"]), "clutter": L.clutter(cells, f["object"], tuple(f["dest"])), "cohort": "fixture"})
with gzip.open("mock_v2/dev/fixture_roster.jsonl.gz", "wt") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(len(rows), [r["qid"] for r in rows])
