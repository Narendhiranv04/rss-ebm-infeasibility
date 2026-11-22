"""EXPLORATORY M-sweep (3D, VAL only): does a bigger candidate set (more cVAE samples per object) help, and which scorer gains more?

Same frozen pieces as V3.4: the bundle's task-conditioned cVAE (K3ABL/stage/bundle/3d), exhaustive.score_root (every ordered K1-K3 plan),
the frozen rule (noop_gate 0.5, eta 0.3, alpha=beta=gamma=0) and the unchanged teacher.  The cVAE proposals are ranked (top-M of its
marginal), so ONE call with M = 8 gives nested pools M = 4 / 6 / 8 (prefixes).  These pools are regenerated on CPU, so they are NOT the team's
stored CUDA pools; both scorers see the SAME regenerated pools.  Scorers: XYNet (the bundle's FULL scorer s0) and our hazard model s0.
    python msweep.py score i/k        -> runs/explore/msweep/score_3d_val.part{i}of{k}.jsonl   (resumable per root)
    python msweep.py teach W          -> runs/explore/msweep/verify_3d.jsonl                   (only picks without a label)
    python msweep.py report           -> runs/explore/msweep/report.json
"""
import json, os, sys, time


def jl_safe(path):
    """json lines; a crash can leave a truncated last line -> skipped."""
    out = []
    for l in open(path):
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:
            pass
    return out
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import twod_hazard as HZ  # noqa: E402  (frozen; read-only)

OUT = HZ.RUNS / "explore/msweep"
os.environ["V3OUT"] = str(OUT / "teach_out")                           # teacher scratch stays in this folder
os.environ["FIN3D_STORE"] = str(OUT / "teach_out/fin3d_store")
import hz_eval as H  # noqa: E402
import benchmark as BM  # noqa: E402
import exhaustive as EX  # noqa: E402
import search as S  # noqa: E402

DOM, MMAX, MS = "3d", 8, (4, 6, 8)
EX.MS = MS                                                              # score_root reports decisions for each nested pool size


def roots():
    return [S.Root(r, DOM) for r in map(json.loads, open(H.K3 / "runs/public" / f"public_val_{DOM}.jsonl"))]


def score(part):
    import torch
    torch.set_num_threads(1)
    i, k = map(int, part.split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    fo = OUT / f"score_{DOM}_val.part{i}of{k}.jsonl"
    done = {r["root_id"] for r in jl_safe(fo)} if fo.exists() else set()
    samp, xy, _ = BM.load_bundle(H.K3 / "stage/bundle", DOM, "cpu", 0)
    scorers = {"xynet": xy, "hazard": HZ.HazardScorer(H.OUT / f"model_{DOM}_s0.pt")}
    import hashlib
    with open(fo, "a") as f:
        for root in roots():
            if root.id in done or int(hashlib.sha256(root.id.encode()).hexdigest(), 16) % k != i:
                continue
            gen = torch.Generator(device="cpu").manual_seed(0)
            D = samp.propose_many(root, root.cells0, list(root.movable), MMAX, gen) if root.movable else {}
            D = {int(o): [int(d) for d in v] for o, v in D.items()}
            rec = {"root_id": root.id, "proposals": {str(o): v for o, v in D.items()}}
            for name, sc in scorers.items():
                t0 = time.time()
                r = EX.score_root(root, D, sc, [HZ.FROZEN], MMAX, set())
                rec[name] = {str(M): {"decision": r["per_M"][str(M)]["decisions"][0], "n_scored": r["per_M"][str(M)]["n_scored"]}
                             for M in MS if str(M) in r["per_M"]}
                rec[name]["p_noop"], rec[name]["s"] = r["p_noop"], round(time.time() - t0, 2)
            f.write(json.dumps(rec) + "\n"); f.flush()


def rows():
    out = {}
    for p in sorted(OUT.glob(f"score_{DOM}_val.part*.jsonl")):
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


def seq_of(d):
    return None if d is None else d["seq"]


def teach(workers):
    priv, L, jobs = H.priv_of(DOM, "val"), labels(), {}
    for rid, r in rows().items():
        for name in ("xynet", "hazard"):
            for M, v in r[name].items():
                if M in ("p_noop", "s"):
                    continue
                s = seq_of(v["decision"])
                if s and None in L.get((rid, json.dumps(s)), (None, None)):
                    jobs.setdefault(rid, {})[json.dumps(s)] = s
    jl = []
    for rid, seqs in jobs.items():
        e = priv[rid]
        jl.append({"root_id": rid, "dom": DOM.upper(), "path": H.local(e["path"]), "geom": e.get("geom", "original"), "ids": e["ids"],
                   "seqs": [[[e["ids"][o], c] for o, c in s] for s in seqs.values()]})
    print(json.dumps({"roots": len(jl), "queries": sum(len(j["seqs"]) for j in jl)}), flush=True)
    if jl:
        H._run(DOM, jl, workers, OUT / f"verify_{DOM}.jsonl")


def report():
    priv, L, R = H.priv_of(DOM, "val"), labels(), rows()
    assert set(priv) <= set(R), f"{len(set(priv) - set(R))} roots not scored"
    out = {"label": "EXPLORATORY 3D VAL M-sweep (regenerated CPU pools, frozen rule)", "n_roots": len(R)}
    for name in ("xynet", "hazard"):
        for M in map(str, MS):
            res = H.rates([{"masks": priv[rid]["masks"], "outcome": H.V.outcome(seq_of(r[name][M]["decision"]), L, rid)} for rid, r in R.items()])
            res["mean_plans"] = round(sum(r[name][M]["n_scored"] for r in R.values()) / len(R), 1)
            out[f"{name}_M{M}"] = res
        out[f"{name}_mean_s_per_root_all_M"] = round(sum(r[name]["s"] for r in R.values()) / len(R), 3)
    ref = {x["root_id"]: x for x in H._full_rows(DOM, "val", 0)}
    same = sum(1 for rid, r in R.items() if rid in ref and seq_of(r["xynet"]["4"]["decision"]) ==
               (None if ref[rid]["decision"] == "ABSTAIN" else [] if ref[rid]["decision"] == "NOOP" else ref[rid]["seq"]))
    out["xynet_M4_same_as_stored_V34_FULL_s0"] = f"{same}/{len(R)} (regenerated CPU pools: M4 here is NOT the V3.4 result)"
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if (OUT / "HOLD").exists() and not os.environ.get("MSWEEP_RUN"):   # user 2026-11-22: reviewer runs first; queue2 runs the sweep later
        print("msweep held (runs/explore/msweep/HOLD)"); sys.exit(0)
    {"score": lambda: score(a[0]), "teach": lambda: teach(int(a[0])), "report": report}[c]()
