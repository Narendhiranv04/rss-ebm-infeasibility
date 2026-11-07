"""V2 natural branch fixtures, mined from V1 records (no new roots): (A) poses V1 rejected as HELD_INVALID_PHYSICAL although they had > 1
eligible branch (V1 certified branch 0 only); (B) units V1 ended as EXEC_FAILED / held_state_lost (the branch that planned first failed its
hold and V1 executed no other branch).  Each candidate query is re-evaluated by V2 (reference_v2, fresh scene); a fixture is kept when V2
certifies a later branch of the same pose.  Run once, after the V2 freeze; the result is frozen in mock_v2/configs/fixtures_v2.json.
  python fixtures_v2.py OUT.json"""
import glob, gzip, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement3d as P, policy3d as Q  # noqa: E401

V1 = P.HERE.parents[1] / "dataset_placement_extension_v1"
idx = {q["qid"]: q for q in (json.loads(x) for x in gzip.open(V1 / "mock_v1/roster.jsonl.gz", "rt"))}
cands = {}
for run in ("pilot_A", "reference", "supplement_reference_on_adaptive_negatives"):
    for f in sorted(glob.glob(str(V1 / f"mock_v1/runs/{run}/units/*.json"))):
        for r in json.load(open(f))["records"]:
            for m in r.get("modes", []):
                if m["screen"] == "HELD_INVALID_PHYSICAL" and m["n_branches"] > 1:
                    cands.setdefault(("A_branch0_hold_rejected", r["qid"], m["rank"]), {"v1_state": "HELD_INVALID_PHYSICAL", "v1_elig_index": 0, "run": run})
                for t in m["tests"]:
                    if (t.get("exec") or {}).get("first_failed") == "held_state_lost":
                        cands.setdefault(("B_planned_branch_hold_lost", r["qid"], m["rank"]), {"v1_state": "EXEC_FAILED", "v1_elig_index": t["branch"], "run": run})
out, searched = [], []
for (kind, qid, rank), v1 in sorted(cands.items()):
    q = idx[qid]
    raw = json.loads((P.SRC / "raw/roots" / f"{q['root_id']}.json").read_text())
    sc = P.Scene(raw["objects"], q["cells"])
    rec, _ = Q.evaluate(sc, q["object"], tuple(q["dest"]), Q.POLICIES["reference_v2"], early_stop=False)
    m = next(x for x in rec["modes"] if x["rank"] == rank)
    elig = [br["b"] for br in m["branches"] if br["eligible"]]
    tested = elig[v1["v1_elig_index"]] if v1["v1_elig_index"] < len(elig) else None
    later_held = [br["b"] for br in m["branches"] if br["b"] != tested and br["hold"] and br["hold"]["ok"]]
    ver = [br["b"] for br in m["branches"] for r_ in br["results"] if r_["state"] == "VERIFIED"]
    s = {"kind": kind, "qid": qid, "rank": rank, "v1_run": v1["run"], "v2_screen": m["screen"], "v2_state": m["state"], "tested_branch": tested,
         "tested_branch_hold_ok": bool(m["branches"][tested]["hold"] and m["branches"][tested]["hold"]["ok"]) if tested is not None else None,
         "later_held": later_held, "verified_branches": ver, "query_label_v2": rec["label"], "query_status_v2": rec["status"]}
    searched.append(s); print(json.dumps(s), flush=True)
    if later_held and tested is not None and not s["tested_branch_hold_ok"]:
        out.append({**s, "root_id": q["root_id"], "cells": q["cells"], "object": q["object"], "dest": q["dest"],
                    "v1": {"state": v1["v1_state"], "branch_tested": tested, "source_run": v1["run"]}, "v2_mode_canonical": Q.canonical(m)})
doc = {"rule": __doc__.split("\n  python")[0], "teacher": P.protocol()[0], "candidates_searched": len(searched), "searched": searched,
       "natural_found": any(f["kind"].startswith("A") for f in out),
       "second_fixture_hold_or_exec_failure_rescued_by_verified_branch": any(f["kind"].startswith("B") and f["verified_branches"] for f in out),
       "fixtures": out}
Path(sys.argv[1]).write_text(json.dumps(doc, indent=1, sort_keys=True, default=str))
print(json.dumps({k: doc[k] for k in ("candidates_searched", "natural_found", "second_fixture_hold_or_exec_failure_rescued_by_verified_branch")}))
