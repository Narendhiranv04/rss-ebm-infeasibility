"""EXPLORATORY failure breakdown (no new compute): on task scenes, what happened to each arm's frozen-rule decision, and for physics
failures, which step and which teacher status.  Arms: ours = hazard (V3.4 rule), full = XYNet FULL.  Seeds 0-2 pooled (counts are runs).
    python failures.py DOM SPLIT   -> runs/explore/failures_{dom}_{split}.json
"""
import collections, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402  (frozen; read-only)

OUT = H.HZ.ABF.parent / "explore"


def records(dom):
    """(root, json(seq in object ids)) -> teacher result; later files win (our Mac cache over the shared cache)."""
    R = {}
    for p in (H.K3 / "runs/verifier_cache" / f"verify_{dom}.jsonl", H.OUT / f"verify_{dom}.jsonl"):
        if p.exists():
            for v in H.BM.jl(p):
                for x in v["results"]:
                    R[(v["root"], json.dumps(x["seq"]))] = x
    return R


def why(rec):
    x, y = rec.get("x") or {}, rec.get("y") or {}
    if x.get("x") == 0:
        return f"X fail @ {x.get('phase') or '?'} [{x.get('status')}]"
    if x.get("x") == 1 and y.get("y") == 0:
        return f"Y fail (target pick after the moves) [{y.get('status')}]"
    return f"unclear [{x.get('status')}/{y.get('status')}]"


def main(dom, split):
    priv, L, R = H.priv_of(dom, split), H.labels(dom), records(dom)
    task = {rid for rid, e in priv.items() if e["masks"].get("eval_task")}
    res = {}
    for arm in ("ours", "full"):
        c = collections.Counter()
        for s in (0, 1, 2):
            rows = {r["root_id"]: r for r in (H.ours(dom, split, s) if arm == "ours" else H._full_rows(dom, split, s))}
            for rid in task:
                r = rows[rid]
                seq = r["decisions"][0] if arm == "ours" else (r["seq"] if r["decision"] == "REPAIR" else ([] if r["decision"] == "NOOP" else None))
                o = H.V.outcome(seq, L, rid)
                if o in ("FAIL_X", "FAIL_Y"):
                    rec = R.get((rid, json.dumps(seq)))      # same key as V.labels (plan as stored)
                    c[why(rec) if rec else f"{o} (no teacher record found)"] += 1
                else:
                    c[o] += 1
        res[arm] = dict(c.most_common())
    out = {"label": f"EXPLORATORY {dom} {split}, task scenes, 3 seeds pooled (runs)", "n_runs": 3 * len(task), **res}
    (OUT / f"failures_{dom}_{split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
