"""EXPLORATORY (reviewer requests, 3D, VAL + TEST): ranking quality without thresholds, and "learned ranking + verify the top k".

On the team's stored V3.4 pools (FULL_s0 predictions, pool hash checked, M = 4) both scorers (XYNet = bundle FULL scorer s0; hazard s0) rank
every K1-K3 repair (exhaustive.score_root keeps the top plans).  On task scenes the teacher checks each scorer's top 5.
  forced top-1  = success of the #1-ranked repair (no noop gate, no eta): pure ranking quality, immune to the probability scale.
  within top-k  = some repair among the top k succeeds: a "verify the top k in simulation, then execute" planner (k physics checks max).
TEST was already used for the frozen confirmatory analysis (commit d94d512); everything here is post-hoc and EXPLORATORY.
    python rank_check.py score SPLIT i/k      -> runs/explore/rank/top_3d_{split}.part{i}of{k}.jsonl
    python rank_check.py teach SPLIT W        -> runs/explore/rank/verify_3d.jsonl   (only plans without a label)
    python rank_check.py report               -> runs/explore/rank/report.json
"""
import hashlib, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import twod_hazard as HZ  # noqa: E402  (frozen; read-only)

OUT = HZ.RUNS / "explore/rank"
os.environ["V3OUT"] = str(OUT / "teach_out")
os.environ["FIN3D_STORE"] = str(OUT / "teach_out/fin3d_store")
import hz_eval as H  # noqa: E402
import benchmark as BM  # noqa: E402
import exhaustive as EX  # noqa: E402
import search as S  # noqa: E402

DOM, K = "3d", 5


def jl_safe(path):
    out = []
    for l in open(path):
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out


def score(split, part):
    import torch
    torch.set_num_threads(1)
    i, k = map(int, part.split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    fo = OUT / f"top_{DOM}_{split}.part{i}of{k}.jsonl"
    done = {r["root_id"] for r in jl_safe(fo)} if fo.exists() else set()
    rows = [json.loads(l) for l in open(H.K3 / "runs/public" / f"public_{split}_{DOM}.jsonl")]
    ref = {r["root_id"]: r for r in map(json.loads, open(H.K3 / "runs/predictions/FULL_s0" / f"{split}_{DOM}.jsonl"))}
    _, xy, _ = BM.load_bundle(H.K3 / "stage/bundle", DOM, "cpu", 0)
    scorers = {"xynet": xy, "hazard": HZ.HazardScorer(H.OUT / f"model_{DOM}_s0.pt")}
    with open(fo, "a") as f:
        for r in rows:
            if r["root_id"] in done or int(hashlib.sha256(r["root_id"].encode()).hexdigest(), 16) % k != i:
                continue
            root = S.Root(r, DOM)
            D = {int(o): [int(d) for d in v[:4]] for o, v in ref[r["root_id"]]["proposals"].items()}
            assert HZ.pool_hash(D) == ref[r["root_id"]]["pool_hash"], r["root_id"]
            rec = {"root_id": r["root_id"]}
            for name, sc in scorers.items():
                top = EX.score_root(root, D, sc, [HZ.FROZEN], 4, set())["per_M"]["4"]["top"][:K]
                rec[name] = [{"p": round(t[0], 6), "seq": json.loads(t[1])} for t in top]
            f.write(json.dumps(rec) + "\n"); f.flush()


def tops(split):
    out = {}
    for p in sorted(OUT.glob(f"top_{DOM}_{split}.part*.jsonl")):
        for r in jl_safe(p):
            out.setdefault(r["root_id"], r)
    return out


def labels():
    L = H.labels(DOM)
    vf = OUT / f"verify_{DOM}.jsonl"
    if vf.exists():
        for key, v in H.V.labels(DOM, str(vf)).items():
            old = L.get(key, (None, None))
            if None in old:
                L[key] = v
            elif None not in v and v != old:
                raise SystemExit(f"CONFLICT between label sources for {key}: {old} vs {v}")
    return L


def teach(split, workers):
    priv, L, jobs = H.priv_of(DOM, split), labels(), {}
    for rid, r in tops(split).items():
        if not priv[rid]["masks"].get("eval_task"):
            continue
        for name in ("xynet", "hazard"):
            for t in r[name]:
                if None in L.get((rid, json.dumps(t["seq"])), (None, None)):
                    jobs.setdefault(rid, {})[json.dumps(t["seq"])] = t["seq"]
    jl = []
    for rid, seqs in jobs.items():
        e = priv[rid]
        jl.append({"root_id": rid, "dom": DOM.upper(), "path": H.local(e["path"]), "geom": e.get("geom", "original"), "ids": e["ids"],
                   "seqs": [[[e["ids"][o], c] for o, c in s] for s in seqs.values()]})
    print(json.dumps({"split": split, "roots": len(jl), "queries": sum(len(j["seqs"]) for j in jl)}), flush=True)
    if jl:
        H._run(DOM, jl, workers, OUT / f"verify_{DOM}.jsonl")


def report():
    L, out = labels(), {"label": "EXPLORATORY 3D, stored V3.4 pools (M4), task scenes only; TEST post-hoc"}
    for split in ("val", "test"):
        priv, T = H.priv_of(DOM, split), tops(split)
        task = [rid for rid in priv if priv[rid]["masks"].get("eval_task")]
        assert set(task) <= set(T), f"{split}: {len(set(task) - set(T))} task roots not scored"
        for name in ("xynet", "hazard"):
            oc = {rid: [H.V.outcome(t["seq"], L, rid) for t in T[rid][name]] for rid in task}
            unk = sum(o.count("UNKNOWN") for o in oc.values())
            res = {"n_task": len(task), "unknown_plans": unk}
            for k in range(1, K + 1):
                res[f"within_top{k}"] = round(100 * sum("SUCCESS" in o[:k] for o in oc.values()) / len(task), 2)
            res["forced_top1"] = res["within_top1"]
            out[f"{split}_{name}"] = res
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    {"score": lambda: score(a[0], a[1]), "teach": lambda: teach(a[0], int(a[1])), "report": report}[c]()
