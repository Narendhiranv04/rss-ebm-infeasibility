"""EXPLORATORY: tune the "do nothing" gate (noop_gate) together with eta, for BOTH arms (ours = hazard, full = XYNet), on VAL; apply to TEST.
Rule: NOOP if p_noop >= gate; else the argmax plan if its p >= eta; else ABSTAIN.  Only gates <= 0.5 (frozen value) are exact from stored
rows: a lower gate only turns stored decisions into NOOP (plans are not stored where the frozen rule already chose NOOP).
Selection exactly as eta_sweep: max over the grid of the 3-seed mean of mean(task, noop, abstain) on VAL; ties -> the setting closest to
frozen (higher gate, lower eta).  TEST: paired 3-seed effect, world-cluster bootstrap.  Reported as exploratory only.
    python gate_sweep.py val DOM | test DOM   -> runs/explore/gate/gate_{dom}[_test].json
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402
import numpy as np  # noqa: E402
import eta_sweep as E  # noqa: E402

OUT = H.HZ.ABF.parent / "explore/gate"
GATES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
GRID = [(g, e) for g in GATES for e in E.GRID]


def pnoop_full(dom, split):
    """XYNet p_noop per seed, from the team's raw prediction files (the results files may not carry it)."""
    return [{r["root_id"]: r["p_noop"] for r in map(json.loads, open(H.K3 / f"runs/predictions/FULL_s{s}" / f"{split}_{dom}.jsonl"))}
            for s in E.SEEDS]


def outcome(arm, row, rid, gate, eta, L, pn):
    p = row["p_noop"] if arm == "ours" else pn[rid]
    if p is not None and p >= gate:
        return H.V.outcome([], L, rid)
    return E.outcome(arm, row, rid, eta, L)


def indicators(priv, L, rows, arm, gate, eta, pns):
    out = {m: {} for m in E.MET}
    for rid, e in priv.items():
        for s in range(len(E.SEEDS)):
            c = H.BM.correct({"masks": e["masks"], "outcome": outcome(arm, rows[s][rid], rid, gate, eta, L, pns[s])})
            for i, m in enumerate(E.MET):
                if c[i] is not None:
                    out[m].setdefault(rid, []).append(c[i])
    return out


def val(dom):
    priv, L, ours, full = E.load(dom, "val")
    pns = {"ours": [None] * 3, "full": pnoop_full(dom, "val")}
    res = {}
    for arm, rows in (("ours", ours), ("full", full)):
        tab = {ge: E.rates(indicators(priv, L, rows, arm, *ge, pns[arm])) for ge in GRID}
        best = max(GRID, key=lambda ge: (round(np.mean(list(tab[ge].values())), 6), ge[0], -ge[1]))
        assert tab[(0.5, 0.3)] == E.rates(E.indicators(priv, L, rows, arm, 0.3)), "gate 0.5 must reproduce the frozen rule"  # self-check
        res[arm] = {"chosen": best, "at_chosen": tab[best], "at_frozen": tab[(0.5, 0.3)],
                    "table": {f"{g}/{e}": v for (g, e), v in tab.items()}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"gate_{dom}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({a: {k: r[k] for k in ("chosen", "at_frozen", "at_chosen")} for a, r in res.items()}))


def test(dom):
    ch = json.loads((OUT / f"gate_{dom}.json").read_text())
    priv, L, ours, full = E.load(dom, "test")
    io = indicators(priv, L, ours, "ours", *ch["ours"]["chosen"], [None] * 3)
    iff = indicators(priv, L, full, "full", *ch["full"]["chosen"], pnoop_full(dom, "test"))
    out = {"dom": dom, "split": "test", "label": "EXPLORATORY", "gate_eta": {a: ch[a]["chosen"] for a in ("ours", "full")}}
    for m in E.MET:
        per = {r: (np.mean(io[m][r]), np.mean(iff[m][r])) for r in io[m] if r in iff[m]}
        out[m] = H._effect(per, 10000, np.random.default_rng(20261122), H._cluster)
    (OUT / f"gate_{dom}_test.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: (v if not isinstance(v, dict) or "diff" not in v else {x: v[x] for x in ("ours", "full", "diff", "lb97.5_cluster")})
                      for k, v in out.items()}))


if __name__ == "__main__":
    {"val": val, "test": test}[sys.argv[1]](sys.argv[2])
