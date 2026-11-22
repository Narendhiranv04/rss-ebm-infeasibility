"""EXPLORATORY zero-shot geometry transfer (user 2026-11-22: "apply anywhere without further training"): the frozen-rule TEST 3D picks of
both arms (hazard s0, XYNet = FULL_s0) on task scenes, written for hardware/zeroshot0/height_check.py, which re-checks each plan with the
board moved to another height (0 cm = table level) using the unchanged teacher.  No model is retrained.
    python zeroshot_plans.py      -> hardware/zeroshot0/plans.jsonl  (refuses to overwrite)
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "hardware/zeroshot0/plans.jsonl"
dom = "3d"
priv, L = H.priv_of(dom, "test"), H.labels(dom)
ours = {r["root_id"]: r["decisions"][0] for r in H.ours(dom, "test", 0)}
full = {x["root_id"]: x for x in H._full_rows(dom, "test", 0)}
plans, idx = [], {}
for rid, e in priv.items():
    if not e["masks"].get("eval_task"):
        continue
    for arm, seq in (("hazard", ours[rid]), ("xynet", full[rid]["seq"] if full[rid]["decision"] == "REPAIR" else None)):
        if not seq:
            continue
        key = (rid, json.dumps(seq))
        if key not in idx:
            assert e.get("geom", "original") == "original"
            idx[key] = len(plans)
            plans.append({"plan": len(plans), "root_id": rid, "K": e.get("K"), "geom": "original", "path": H.local(e["path"]),
                          "ids": e["ids"], "seq": seq, "cached": list(L.get(key, (None, None))), "arms": []})
        plans[idx[key]]["arms"].append(arm)
with open(OUT, "x") as f:
    for p in plans:
        f.write(json.dumps(p) + "\n")
print("plans", len(plans), "task roots", sum(1 for e in priv.values() if e["masks"].get("eval_task")),
      "hazard", sum("hazard" in p["arms"] for p in plans), "xynet", sum("xynet" in p["arms"] for p in plans))
