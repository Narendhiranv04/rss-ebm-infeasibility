"""Summarize the R2 missing-endpoint Y pilot (BD3DS66_Y_PLAN_V1).  Pure analysis of bwlphys.py item JSONs.

    python3 -I summarize_r2.py R2_PILOT_MANIFEST.json OUT_DIR SUMMARY.json

Y = 1 only if the teacher's planned witness EXECUTED and passed AND a fresh-process replay passed with identical end state; Y = 0 if the planned
query failed (attempted, teacher kind kept); PLAN_EXEC_FAIL (plan found, execution failed) is recorded as such (operational 0, phase =
target execution); controls must reproduce the archived label (and, for positives, the identical witness id) or the protocol is MISMATCHED."""
import json, sys, collections
from pathlib import Path

import numpy as np

man, out, dst = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), Path(sys.argv[3])
rows, cost = [], collections.defaultdict(list)
for it in man["items"]:
    t, r = (json.loads((out / "items" / f"{it['item']}.{m}.json").read_text()) if (out / "items" / f"{it['item']}.{m}.json").exists() else None
            for m in ("teacher", "replay"))
    if t is None:
        rows.append({"item": it["item"], "stratum": it["stratum"], "final": "MISSING_NOT_RUN"}); continue
    tv, rv = t.get("verdict"), r and r.get("verdict")
    if tv == "PLAN_EXEC_PASS":
        same = r and "final" in r and "final" in t and max(float(np.max(np.abs(np.array(t["final"]["q"]) - np.array(r["final"]["q"])))),
                                                            max(float(np.max(np.abs(np.array(t["final"]["obj"][k]) - np.array(r["final"]["obj"][k])))) for k in t["final"]["obj"])) <= 1e-9
        final = "Y1_PLANNED_EXECUTED_REPLAY_VERIFIED" if rv == "REPLAY_PASS" and same else "Y_REPLAY_DISAGREES"
    elif tv and tv.startswith("Y0_ATTEMPTED"):
        final = tv
    elif tv == "PLAN_EXEC_FAIL":
        final = "Y0_PLAN_FOUND_EXECUTION_FAILED"
    else:
        final = f"MISSING_{tv}"
    c = t.get("control")
    ok_ctrl = None if not c else bool(c["label_agrees"] and (c["witness_identical"] in (True, None)))
    cost[it["stratum"].split("_")[0] + ("_pos" if final.startswith("Y1") else "_neg")].append((t.get("cpu_s") or 0) + ((r or {}).get("cpu_s") or 0))
    rows.append({"item": it["item"], "stratum": it["stratum"], "split": it["split"], "K": it["K"], "final": final, "control_ok": ok_ctrl, "control": c,
                 "kind": (t.get("plan") or {}).get("kind"), "cpu_s": round((t.get("cpu_s") or 0) + ((r or {}).get("cpu_s") or 0), 2),
                 "x_orders_reaching": it.get("x_orders_reaching"), "x_statuses": it.get("x_statuses"), "step": t.get("step")})
new = [r for r in rows if not r["stratum"].startswith("C_")]
ctrl = [r for r in rows if r["stratum"].startswith("C_")]
cpu = sum(r.get("cpu_s", 0) for r in rows)
pos, neg = [r["cpu_s"] for r in new if r["final"].startswith("Y1")], [r["cpu_s"] for r in new if r["final"].startswith("Y0")]
p = len(pos) / max(1, len(pos) + len(neg))
miss = man["population"]["missing_train_val"]
summ = {"items": len(rows), "new_endpoint_finals": dict(collections.Counter(r["final"] for r in new)),
        "new_endpoint_finals_by_stratum": dict(collections.Counter(f"{r['stratum']}|{r['final']}" for r in new)),
        "controls": {"n": len(ctrl), "agree": sum(1 for r in ctrl if r["control_ok"]), "finals": dict(collections.Counter(r["final"] for r in ctrl)),
                     "positive_witness_identical": sum(1 for r in ctrl if (r["control"] or {}).get("witness_identical") is True)},
        "cpu_h_total": round(cpu / 3600, 3),
        "cpu_s_per_query": {"positive_mean": float(np.mean(pos)) if pos else None, "positive_p90": float(np.percentile(pos, 90)) if pos else None,
                            "negative_mean": float(np.mean(neg)) if neg else None, "negative_p90": float(np.percentile(neg, 90)) if neg else None},
        "pilot_positive_rate_new_endpoints": p,
        "bulk_estimate": {"population_missing_all_splits": 3407, "population_missing_train_val": miss,
                          "cpu_h_all_3407": round(3407 * (p * (np.mean(pos) if pos else 0) + (1 - p) * (np.mean(neg) if neg else 0)) / 3600, 1)},
        "rows": rows}
dst.write_text(json.dumps(summ, indent=1, default=str))
print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1, default=str))
