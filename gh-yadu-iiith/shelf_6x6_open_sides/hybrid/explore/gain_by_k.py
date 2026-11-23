"""EXPLORATORY: task-success lead of ours over XYNet (FULL) by number of moves K, 3D, VAL and TEST (TEST post-hoc; prereg 8e18429 already
scored it once).  Same per-root logic as hz_eval.confirm (frozen rule, 3 seeds averaged per root, same label table, world-cluster bootstrap
B 10,000 rng 20261122).  K3 is also split: original roots vs hard-order / dependence children (root ids with D3H_ / D3_).
    python gain_by_k.py   -> runs/explore/gain_by_k_3d.json"""
import collections, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402
import numpy as np  # noqa: E402

DOM, L = "3d", H.labels("3d")


def eff(per):
    try:
        return H._effect(per, 10000, np.random.default_rng(20261122), H._cluster)
    except AssertionError:          # hard-order children: every root is its own world -> root-level bootstrap
        d = np.array([a - b for a, b in per.values()])
        bs = np.random.default_rng(20261122).choice(d, (10000, len(d))).mean(1)
        return {"n": len(d), "ours": round(100 * np.mean([a for a, _ in per.values()]), 2), "full": round(100 * np.mean([b for _, b in per.values()]), 2),
                "diff": round(100 * d.mean(), 2), "lb97.5_cluster": round(100 * np.quantile(bs, 0.025), 2), "bootstrap": "root-level"}
out = {"label": "EXPLORATORY; task success by K, ours frozen rule vs XYNet FULL, 3 seeds; TEST post-hoc"}
for split in ("val", "test"):
    priv = H.priv_of(DOM, split)
    ours = [{r["root_id"]: r["decisions"][0] for r in H.ours(DOM, split, s)} for s in (0, 1, 2)]
    full = [{x["root_id"]: x for x in H._full_rows(DOM, split, s)} for s in (0, 1, 2)]

    def ok(arm, s, rid, e):
        if arm == "ours":
            o = H.V.outcome(ours[s][rid], L, rid)
        else:
            x = full[s][rid]
            o = x["outcome"] if x["decision"] != "REPAIR" else H.V.outcome(x["seq"], L, rid)
        return H.BM.correct({"masks": e["masks"], "outcome": o})[0]

    groups = collections.defaultdict(dict)
    for rid, e in priv.items():
        if not e["masks"].get("eval_task"):
            continue
        a = [c for c in (ok("ours", s, rid, e) for s in range(3)) if c is not None]
        b = [c for c in (ok("full", s, rid, e) for s in range(3)) if c is not None]
        if a and b:
            k = str(e["K"])
            groups["K" + k][rid] = groups["all"][rid] = (float(np.mean(a)), float(np.mean(b)))
            if k == "3":
                groups["K3_" + ("variant" if ("D3H_" in rid or "D3_" in rid) else "original")][rid] = (float(np.mean(a)), float(np.mean(b)))
    out[split] = {g: eff(per) for g, per in sorted(groups.items())}
    for g, r in out[split].items():
        print(split, g, {x: r[x] for x in ("n", "ours", "full", "diff", "lb97.5_cluster")})
assert out["test"]["all"]["n"] == 241 and out["val"]["all"]["n"] == 220, "task-scene counts differ from the known 241 / 220"
(H.HZ.ABF.parent / "explore/gain_by_k_3d.json").write_text(json.dumps(out, indent=1).replace('"full"', '"xynet"'))
