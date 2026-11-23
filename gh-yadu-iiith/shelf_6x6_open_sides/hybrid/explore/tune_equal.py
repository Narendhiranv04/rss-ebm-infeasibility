"""EXPLORATORY equal-tuning check (item #5, user 2026-11-22), 3D VAL only: give the PIGINet-style whole-plan arm the same threshold tuning
ours got (hz_eval.tune criterion: max mean(task, noop, abstain) over the 37 stored rules; ties -> higher task -> lower index), per seed.
Both arms are tuned AND scored on VAL, so both numbers are equally optimistic; compare them with each other, not with TEST.
    python tune_equal.py count            teacher queries still needed (no teacher)
    python tune_equal.py run WORKERS      label every rule's pick (env3d python), then -> runs/explore/tune_equal_3d_val.json
"""
import json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parents[1] / "approachb_3d_learning/runs/explore"
PIG = EXP / "enc/piginet"
os.environ["V3OUT"] = str(PIG / "teach_out")                       # teacher scratch stays in the PIGINet dir (as encoders.evaluate)
os.environ["FIN3D_STORE"] = str(PIG / "teach_out/fin3d_store")
sys.path.insert(0, str(HERE.parent / "src"))
import hz_eval as H  # noqa: E402  (frozen; read-only)

DOM, VF = "3d", PIG / "verify_3d.jsonl"


def rows(seed):
    out = {}
    for p in sorted(PIG.glob(f"pool_{DOM}_val_s{seed}*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            out.setdefault(r["root_id"], r)
    return out


def labels():
    L = H.labels(DOM)
    if VF.exists():
        for k, v in H.V.labels(DOM, str(VF)).items():
            if None in L.get(k, (None, None)):
                L[k] = v
    return L


def todo(priv, L):
    need = {}
    for s in (0, 1, 2):
        R = rows(s)
        assert set(priv) <= set(R), f"seed {s}: {len(set(priv) - set(R))} VAL roots not pooled"
        for rid, r in R.items():
            for d in r["decisions"]:
                if d and None in L.get((rid, json.dumps(d)), (None, None)):
                    need.setdefault(rid, set()).add(json.dumps(d))
    return need


def best(m):
    sc = [(sum(v[x] for x in ("task", "noop", "abstain")) / 3, v["task"], -k, k) for k, v in enumerate(m) if v["unknown_task"] == 0]
    return max(sc)[3]


def run(workers):
    priv = H.priv_of(DOM, "val")
    need = todo(priv, labels())
    jobs = [{"root_id": rid, "dom": DOM.upper(), "path": H.local(priv[rid]["path"]), "geom": priv[rid].get("geom", "original"),
             "ids": priv[rid]["ids"], "seqs": [[[priv[rid]["ids"][o], c] for o, c in json.loads(s)] for s in sorted(ss)]}
            for rid, ss in need.items()]
    if jobs:
        H._run(DOM, jobs, workers, VF)
    L = labels()
    out = {"label": "EXPLORATORY, 3D VAL, tuned and scored on VAL (both arms equally optimistic)", "seeds": {}}
    for s in (0, 1, 2):
        R = rows(s)
        pm = [H.rates([{"masks": priv[rid]["masks"], "outcome": H.V.outcome(r["decisions"][k], L, rid)} for rid, r in R.items()])
              for k in range(len(H.HZ.RULES))]
        om = list(json.loads((H.OUT / f"metrics_{DOM}_val_s{s}.json").read_text())["hazard_by_rule"].values())
        assert len(om) == len(pm) == len(H.HZ.RULES), "rule lists differ"
        assert all(o["n_task"] == p["n_task"] for o, p in zip(om, pm)), "different task roots"
        bp, bo = best(pm), best(om)
        out["seeds"][s] = {"piginet_frozen": pm[0], "piginet_tuned": {**pm[bp], "rule": H.HZ.RULES[bp]},
                           "ours_frozen": om[0], "ours_tuned": {**om[bo], "rule": H.HZ.RULES[bo]}}
        print(s, json.dumps(out["seeds"][s]))
    (EXP / "tune_equal_3d_val.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "count":
        n = todo(H.priv_of(DOM, "val"), labels())
        print(json.dumps({"roots": len(n), "queries": sum(map(len, n.values()))}))
    elif c == "run":
        run(int(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
