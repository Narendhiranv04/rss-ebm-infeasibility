"""Summarize S1 pilot outputs (pure analysis of bwlphys.py item JSONs; no physics).

    python3 -I summarize_s1.py S1_PILOT_MANIFEST.json OUT_DIR SUMMARY.json

Final labels (fail closed):
  Y = 1  REPLAY_VERIFIED: teacher replay AND fresh-process replay both REPLAY_PASS with identical end state (object poses / q within 1e-9).
  Y = 0  stored operational negative RECONFIRMED by a deterministic re-query (kind unchanged); kind and scope kept.
  X = 1  teacher X_CANDIDATE_1 AND independent replay X_CANDIDATE_1 with final state agreement (<= 1e-6 m / rad).
  X = 0  relocation failure reconfirmed against the stored record, or connector failure in the teacher (phase recorded); never transferred to
         an extension of the prefix.  Anything else (mismatch, env error, timeout) = MISSING / needs investigation, never a label."""
import json, sys, collections
from pathlib import Path

import numpy as np

man, out, dst = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), Path(sys.argv[3])
it = {x["item"]: x for x in man["items"]}
res = {}
for p in (out / "items").glob("*.json"):
    name, mode, _ = p.name.rsplit(".", 2)
    res.setdefault(name, {})[mode] = json.loads(p.read_text())


def sdiff(a, b):
    if not a or not b or "final" not in a or "final" not in b:
        return None
    fa, fb = a["final"], b["final"]
    d = float(np.max(np.abs(np.array(fa["q"]) - np.array(fb["q"]))))
    return max(d, max(float(np.max(np.abs(np.array(fa["obj"][k]) - np.array(fb["obj"][k])))) for k in fa["obj"]))


rows, cost = [], collections.Counter()
for name, r in sorted(res.items()):
    x, t, rp = it[name], r.get("teacher"), r.get("replay")
    for m in (t, rp):
        if m:
            cost["wall_s"] += m.get("wall_s", 0) or 0
            cost["cpu_s"] += m.get("cpu_s", 0) or 0
            cost["processes"] += 1
    d = sdiff(t, rp)
    row = {"item": name, "kind": x["kind"], "phase": x["phase"], "stratum": x["stratum"], "split": x["split"], "root_id": x["root_id"],
           "teacher": t and t.get("verdict"), "replay": rp and rp.get("verdict"), "teacher_replay_state_max_abs_diff": d}
    if x["kind"] == "Y":
        if x["archived_y"] == 1:
            ok = t and rp and t["verdict"] == rp["verdict"] == "REPLAY_PASS" and d is not None and d <= 1e-9
            row["final"] = "Y1_REPLAY_VERIFIED" if ok else ("Y_REPLAY_FAILED" if t and t["verdict"] == "REPLAY_FAIL" else "MISSING_OR_MISMATCH")
            if t and t.get("step"):
                row.update(z_err=t["step"].get("z_err"), failure=t["step"].get("failure"), pushed=t["step"].get("pushed"),
                           witness_start_dq=t.get("witness_first_q_vs_qready"))
        else:
            row["final"] = f"Y0_RECONFIRMED_{x['negative_kind']}" if t and t["verdict"] == "NEGATIVE_RECONFIRMED" else f"NEGATIVE_{t and t['verdict']}"
    else:
        tv, rv = t and t.get("verdict"), rp and rp.get("verdict")
        if tv == "X_CANDIDATE_1":
            row["final"] = "X1_VERIFIED" if (rv == "X_CANDIDATE_1" and d is not None and d <= 1e-6) else "X_REPLAY_DISAGREES"
        elif tv in ("X0_CONNECTOR_NO_PLAN", "X0_CONNECTOR_FAILED"):
            row["final"] = tv.replace("X0_", "X0_STOP_")
        elif tv == "X0_RELOCATION_FAILED":
            row["final"] = "X0_RELOCATION_FAILED_RECONFIRMED"
        else:
            row["final"] = f"MISSING_{tv}"
        if t:
            st = t.get("steps", [])
            row.update(m=x["m"], steps_agree_with_record=all(s.get("agrees_with_record") for s in st) if st else None,
                       max_abs_z_err_vs_record=max([abs(s["z_err"] - s["recorded"]["z_err"]) for s in st if s.get("z_err") is not None and s.get("recorded", {}).get("z_err") is not None] or [None], key=lambda v: -1 if v is None else v),
                       boundary_q_vs_recorded=t.get("boundary_q_vs_recorded"), connector=t.get("connector"), failed_phase=t.get("failed_phase"),
                       policy_ok=all(v is True or (isinstance(v, float) and v < 1e-5) for v in t.get("policy", {}).values()) if t.get("policy") else None)
    rows.append(row)
# order contrasts: items sharing (root, Y endpoint) with m >= 2 and decided X on both
grp = collections.defaultdict(list)
for row in rows:
    x = it[row["item"]]
    if x["kind"] == "X" and x["m"] >= 2 and row["final"].startswith("X"):
        grp[(x["root_id"], x["y_endpoint_key"])].append(int(row["final"].startswith("X1")) if row["final"].startswith(("X1", "X0")) else None)
dec = [g for g in grp.values() if len(g) >= 2 and None not in g]
bq = [v for row in rows for v in (row.get("boundary_q_vs_recorded") or [])]
summ = {"items_done": len(rows), "items_in_manifest": len(it), "final_counts": dict(collections.Counter(r["final"] for r in rows)),
        "final_by_stratum": {f"{r['stratum']}|{r['final']}": n for r, n in []},
        "Y_positive_replays": sum(1 for r in rows if r["kind"] == "Y" and it[r["item"]]["archived_y"] == 1),
        "teacher_vs_independent_replay_max_state_diff": max([r["teacher_replay_state_max_abs_diff"] for r in rows if r["teacher_replay_state_max_abs_diff"] is not None] or [None], key=lambda v: -1 if v is None else v),
        "x_steps_agree_with_record": dict(collections.Counter(str(r.get("steps_agree_with_record")) for r in rows if r["kind"] == "X")),
        "boundary_state_reconstruction_rad": {"n": len(bq), "median": float(np.median(bq)) if bq else None, "max": float(np.max(bq)) if bq else None},
        "connector": {"attempted": sum(1 for r in rows if r.get("connector")), "planned": sum(1 for r in rows if (r.get("connector") or {}).get("planned")),
                      "succeeded": sum(1 for r in rows if (r.get("connector") or {}).get("ok")),
                      "failure_reasons": dict(collections.Counter(((r.get("connector") or {}).get("failure") or "") + ":" + str((r.get("connector") or {}).get("reason", ""))[:60]
                                                                  for r in rows if r.get("connector") and not r["connector"].get("ok")))},
        "order_groups_decided": len(dec), "order_groups_discordant": sum(1 for g in dec if len(set(g)) > 1),
        "cost": {**cost, "cpu_h": round(cost["cpu_s"] / 3600, 3)}, "rows": rows}
c = collections.Counter(f"{r['stratum']}|{r['final']}" for r in rows)
summ["final_by_stratum"] = dict(sorted(c.items()))
dst.write_text(json.dumps(summ, indent=1, default=str))
print(json.dumps({k: v for k, v in summ.items() if k != "rows"}, indent=1, default=str))
