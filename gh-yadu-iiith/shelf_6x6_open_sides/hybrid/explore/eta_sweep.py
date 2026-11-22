"""EXPLORATORY: raise the admission threshold eta (noop_gate fixed 0.5, alpha=beta=gamma=0) for BOTH arms (ours = hazard, FULL = XYNet).

Pre-stated before any 2D TEST result existed (2026-11-22 ~13:30; 3D TEST had been seen):
  * grid eta in {0.30, 0.35, ..., 0.90}; criterion (same as hz_eval.tune) = max over the grid of the 3-seed mean of mean(task, noop, abstain)
    on VAL; ties -> lower eta.  Chosen separately per arm and domain.
  * TEST: apply each arm's VAL-chosen eta; paired 3-seed effect with hz_eval's world-cluster bootstrap. Reported as exploratory only.
Exact because with alpha=0 the repair is always the argmax plan: raising eta (>= 0.3) only turns REPAIR (p < eta) into ABSTAIN.
    python eta_sweep.py val DOM        -> runs/explore/eta/eta_{dom}.json
    python eta_sweep.py test DOM       -> runs/explore/eta/eta_{dom}_test.json   (requires eta_{dom}.json)
"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402  (frozen code, imported read-only)
import numpy as np  # noqa: E402

OUT = H.HZ.ABF.parent / "explore/eta"
GRID = [round(0.30 + 0.05 * i, 2) for i in range(13)]
SEEDS = (0, 1, 2)
MET = ("task", "noop", "abstain")


def load(dom, split):
    priv, L = H.priv_of(dom, split), H.labels(dom)
    ours = [{r["root_id"]: r for r in H.ours(dom, split, s)} for s in SEEDS]
    full = [{x["root_id"]: x for x in H._full_rows(dom, split, s)} for s in SEEDS]
    return priv, L, ours, full


def outcome(arm, row, rid, eta, L):
    if arm == "ours":
        d = row["decisions"][0]
        if d and row["p11"][0] < eta:
            d = None
        return H.V.outcome(d, L, rid)
    if row["decision"] == "REPAIR":
        if row["score_p11"] < eta:
            return "ABSTAIN"
        return H.V.outcome(row["seq"], L, rid)
    return row["outcome"]


def indicators(priv, L, rows, arm, eta):
    """-> {metric: {rid: [per-seed 0/1 (UNKNOWN dropped for task)]}}"""
    out = {m: {} for m in MET}
    for rid, e in priv.items():
        for s in range(len(SEEDS)):
            c = H.BM.correct({"masks": e["masks"], "outcome": outcome(arm, rows[s][rid], rid, eta, L)})
            for i, m in enumerate(MET):
                if c[i] is not None:
                    out[m].setdefault(rid, []).append(c[i])
    return out


def rates(ind):
    return {m: round(100 * float(np.mean([np.mean(v) for v in ind[m].values()])), 2) for m in MET}


def val(dom):
    priv, L, ours, full = load(dom, "val")
    res = {}
    for arm, rows in (("ours", ours), ("full", full)):
        tab = {eta: rates(indicators(priv, L, rows, arm, eta)) for eta in GRID}
        best = max(GRID, key=lambda e: (round(np.mean(list(tab[e].values())), 6), -e))
        res[arm] = {"chosen_eta": best, "table": {str(e): v for e, v in tab.items()}}
    # self-check: eta 0.3 must reproduce hz_eval confirm's 3-seed means on VAL
    c = json.loads((H.OUT / f"confirm_{dom}_val.json").read_text())["primary"]
    for m in MET:
        assert abs(res["ours"]["table"]["0.3"][m] - c[m]["ours"]) < 0.05, (m, res["ours"]["table"]["0.3"][m], c[m]["ours"])
        assert abs(res["full"]["table"]["0.3"][m] - c[m]["full"]) < 0.05, (m, res["full"]["table"]["0.3"][m], c[m]["full"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"eta_{dom}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({a: {"chosen_eta": r["chosen_eta"], "at_0.3": r["table"]["0.3"], "at_chosen": r["table"][str(r["chosen_eta"])]} for a, r in res.items()}))


def test(dom):
    ch = json.loads((OUT / f"eta_{dom}.json").read_text())
    priv, L, ours, full = load(dom, "test")
    io = indicators(priv, L, ours, "ours", ch["ours"]["chosen_eta"])
    iff = indicators(priv, L, full, "full", ch["full"]["chosen_eta"])
    out = {"dom": dom, "split": "test", "label": "EXPLORATORY", "eta": {a: ch[a]["chosen_eta"] for a in ("ours", "full")}}
    for m in MET:
        per = {r: (np.mean(io[m][r]), np.mean(iff[m][r])) for r in io[m] if r in iff[m]}
        out[m] = H._effect(per, 10000, np.random.default_rng(20261122), H._cluster)
    (OUT / f"eta_{dom}_test.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    {"val": val, "test": test}[sys.argv[1]](sys.argv[2])
