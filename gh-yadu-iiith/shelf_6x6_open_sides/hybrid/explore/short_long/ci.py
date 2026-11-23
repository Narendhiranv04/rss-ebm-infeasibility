"""EXPLORATORY, 3D VAL: paired 3-seed differences with world-cluster bootstrap (B 10,000, rng 20261122, = hz_eval.confirm) for test 2.
Per root: mean over seeds 0-2 of BM.correct, labels = H.labels + every short_long / enc verify file (same merge as short_long.evaluate).
Comparisons (groups K1, K2, K3, all task; all noop; all abstain):
  <variant>: ours vs XYNet (same variant)            for noK3scenes, no3moveplans
  <variant>_vs_full: retrained ours vs frozen ours    for every variant (frozen = ABF3/runs/hz pools)
  no3moveplans_nostep_vs_no3moveplans, fulldata_nostep_vs_full
    env3d python ci.py   -> runs/explore/short_long/ci.json"""
import collections, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import hz_eval as H  # noqa: E402
import numpy as np  # noqa: E402

DOM = "3d"
R = H.HZ.RUNS / "explore/short_long"
P = H.priv_of(DOM, "val")
L = H.labels(DOM)
for f in sorted(R.glob(f"*/verify_{DOM}.jsonl")) + sorted((H.HZ.RUNS / "explore/enc").glob(f"*/verify_{DOM}.jsonl")):
    for k, v in H.V.labels(DOM, str(f)).items():
        if None in L.get(k, (None, None)):
            L[k] = v


def decisions(d, s):
    out = {}
    for p in sorted(d.glob(f"pool_{DOM}_val_s{s}*.jsonl")):
        for r in map(json.loads, open(p)):
            out.setdefault(r["root_id"], r["decisions"][0])
    assert set(out) == set(P), (d, s, len(out))
    return out


def arm(d):
    """{rid: {endpoint: mean correct over 3 seeds}}"""
    dec = [decisions(d, s) for s in range(3)]
    res = {}
    for rid, e in P.items():
        c = [H.BM.correct({"masks": e["masks"], "outcome": H.V.outcome(x[rid], L, rid)}) for x in dec]
        res[rid] = c
    return res


def eff(a, b, ep, keep):
    """paired effect a - b on endpoint index ep over roots in keep"""
    per = {}
    for rid in keep:
        x = [c[ep] for c in a[rid] if c[ep] is not None]
        y = [c[ep] for c in b[rid] if c[ep] is not None]
        if x and y:
            per[rid] = (float(np.mean(x)), float(np.mean(y)))
    try:
        return H._effect(per, 10000, np.random.default_rng(20261122), H._cluster)
    except AssertionError:      # hard-order K3 children: each root its own world -> root-level bootstrap
        d = np.array([u - v for u, v in per.values()])
        bs = np.random.default_rng(20261122).choice(d, (10000, len(d))).mean(1)
        return {"n": len(d), "diff": round(100 * d.mean(), 2), "lb97.5_cluster": round(100 * np.quantile(bs, 0.025), 2),
                "ub97.5_cluster": round(100 * np.quantile(bs, 0.975), 2), "bootstrap": "root-level"}


# endpoint index in BM.correct's tuple: 0 task, 1 noop, 2 abstain (asserted below on one root of each kind)
EP = {"task": 0, "noop": 1, "abstain": 2}
K = collections.defaultdict(list)
for rid, e in P.items():
    K["K" + str(e["K"])].append(rid)
for g, ep in (("K0", "noop"), ("KNONE", "abstain"), ("K1", "task")):
    c = H.BM.correct({"masks": P[K[g][0]]["masks"], "outcome": "SUCCESS"})
    assert [i for i, v in enumerate(c) if v is not None] == [EP[ep]], (g, c)

def arm_fullxy():
    """XYNet FULL s0-2 (frozen team predictions), REPAIR picks relabelled from the same table (= hz_eval.confirm)"""
    dec = [{x["root_id"]: (None if x["decision"] == "ABSTAIN" else [] if x["decision"] == "NOOP" else x["seq"]) for x in H._full_rows(DOM, "val", s)} for s in range(3)]
    assert all(set(d) == set(P) for d in dec)
    return {rid: [H.BM.correct({"masks": e["masks"], "outcome": H.V.outcome(d[rid], L, rid)}) for d in dec] for rid, e in P.items()}


arms = {"full_hz": arm(H.HZ.ABF / "runs/hz"), "full_xy": arm_fullxy()}
for v in ("noK3scenes", "no3moveplans", "no3moveplans_nostep", "fulldata_nostep"):
    arms[v + "_hz"] = arm(R / v / "hz")
for v in ("noK3scenes", "no3moveplans"):
    arms[v + "_xy"] = arm(R / v / "xy")
ALL = list(P)


def block(a, b):
    out = {g: eff(arms[a], arms[b], 0, K[g]) for g in ("K1", "K2", "K3")}
    out.update(task=eff(arms[a], arms[b], 0, ALL), noop=eff(arms[a], arms[b], 1, ALL), abstain=eff(arms[a], arms[b], 2, ALL))
    return out


out = {"label": "EXPLORATORY; 3D VAL; 3-seed paired, world-cluster bootstrap B 10,000 (root-level if clusters fail)"}
pairs = [("full_hz", "full_xy"), ("noK3scenes_hz", "noK3scenes_xy"), ("no3moveplans_hz", "no3moveplans_xy"),
         ("noK3scenes_hz", "full_hz"), ("no3moveplans_hz", "full_hz"), ("no3moveplans_nostep_hz", "full_hz"),
         ("no3moveplans_nostep_hz", "no3moveplans_hz"), ("fulldata_nostep_hz", "full_hz")]
for a, b in pairs:
    out[f"{a}__minus__{b}"] = r = block(a, b)
    print(a, "-", b, {g: (x["diff"], x["lb97.5_cluster"], x.get("ub97.5_cluster")) for g, x in r.items()})
(R / "ci.json").write_text(json.dumps(out, indent=1))
