"""EXPLORATORY overfitting check: TEST roots whose generated WORLD also appears in TRAIN (other scene variants of it) vs worlds never seen
in training.  Both arms, frozen rule, 3 seeds averaged per root; world-cluster bootstrap of each arm's rate and of ours - XYNet.
    python unseen_worlds.py DOM   -> runs/explore/unseen_worlds_{dom}.json"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402
import numpy as np  # noqa: E402
import eta_sweep as E  # noqa: E402

dom = sys.argv[1]
trainW = {H._cluster(json.loads(l)["root_id"]) for l in open(H.K3 / "runs/public" / f"private_train_{dom}.jsonl")}
priv, L, ours, full = E.load(dom, "test")
io, iff = E.indicators(priv, L, ours, "ours", 0.3), E.indicators(priv, L, full, "full", 0.3)
out = {"label": f"EXPLORATORY {dom} TEST, frozen rule", "test_worlds_seen_in_train": len({H._cluster(r) for r in priv} & trainW),
       "test_worlds_total": len({H._cluster(r) for r in priv})}
for grp, keep in (("seen_world", lambda r: H._cluster(r) in trainW), ("unseen_world", lambda r: H._cluster(r) not in trainW)):
    for m in E.MET:
        per = {r: (np.mean(io[m][r]), np.mean(iff[m][r])) for r in io[m] if r in iff[m] and keep(r)}
        if len({H._cluster(r) for r in per}) < 3:
            out[f"{grp}/{m}"] = f"too few worlds ({len(per)} roots)"; continue
        e = H._effect(per, 10000, np.random.default_rng(20261122), H._cluster) if len({H._cluster(r) for r in per}) < len(per) else \
            {"n": len(per), "ours": round(100 * np.mean([v[0] for v in per.values()]), 2), "full": round(100 * np.mean([v[1] for v in per.values()]), 2)}
        out[f"{grp}/{m}"] = {k: e[k] for k in ("n", "clusters", "ours", "full", "diff", "lb97.5_cluster") if k in e}
(H.HZ.ABF.parent / f"explore/unseen_worlds_{dom}.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
