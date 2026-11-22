"""EXPLORATORY scaling2d pilot: can the per-step hazard scorer + step-wise best-first search solve 2D scenes needing 4-6 relocations?

Data: team release naren/scaling2d-k12-test scaling2d_k12/G7/release_v2 (fetch_scaling2d.py), K_certified 4-6 only (15 roots; the 2D teacher
cannot judge > 6 relocations).  Sampler: the frozen V3.4 TaskCVAE bundle (K3ABL/stage/bundle/2d).  Scorer: our hazard models s0-s2
(trained on K<=3 data only; the step-index feature k/3 extrapolates for k > 3).  Search: search.search (V3.4 code) copied with the depth cap
3 replaced by KMAX and a larger expansion budget; nothing else changed.  Teacher: unchanged evaluate._verify_job via hz_eval._run.
No baseline (V3.4 / XYNet are K<=3 only).  Report: top-1 and any-of-top-5 success per K; plan length vs certified minimum K.
    python scaling_search.py search SEED [KMAX=6] [MAX_EXP=200]   -> runs/explore/scaling_out/search_s{SEED}.jsonl  (resumable)
    python scaling_search.py teach WORKERS                        -> runs/explore/scaling_out/verify_2d.jsonl     (resumable)
    python scaling_search.py report                               -> runs/explore/scaling_out/report.json
"""
import heapq, itertools, json, math, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402  (frozen; imported read-only: sets code paths / teacher env)
import benchmark as BM  # noqa: E402
import contracts as CT  # noqa: E402
import search as S  # noqa: E402
import torch  # noqa: E402

RUNS = H.HZ.ABF.parent
DATA, OUT = RUNS / "explore/scaling2d", RUNS / "explore/scaling_out"
TOPN = 5


def roots(kmin=4, kmax=6):
    lab = {r["root_id"]: r for r in map(json.loads, open(DATA / "private/LABELS.jsonl"))}
    out = []
    for p in sorted((DATA / "public/roots").glob("*.json")):
        rec = json.loads(p.read_text())
        rid = rec["ids"]["root_id"]
        if not kmin <= lab[rid]["K_certified"] <= kmax:
            continue
        v = CT.view2d(rec)
        assert v["cells0"][v["tid"]] == lab[rid]["target_cell"] and v["goal_cell"] == lab[rid]["goal_cell"], rid
        out.append((S.Root({"root_id": rid, "inputs": v}, "2d"), rec, p, lab[rid]))
    return out


def deep_search(root, sampler, scorer, M, kmax, max_exp, seed, fair=4):
    """search.search (iterative mode) with the depth cap 3 -> kmax.  Priority = -log p11 of the prefix."""
    gen = torch.Generator(device=sampler.dev).manual_seed(seed)
    nodes = {(): {"seq": [], **scorer.score(root, [[]])[0], "m": 0}}
    cost = lambda n: -math.log(max(n["p11"], 1e-12))  # noqa: E731
    heap, fifo, order, exp, pops = [(cost(nodes[()]), 0, ())], [()], itertools.count(1), set(), 0
    while (heap or fifo) and len(exp) < max_exp:
        pops += 1
        k = fifo.pop(0) if fair and pops % fair == 0 and fifo else (heapq.heappop(heap)[2] if heap else fifo.pop(0))
        if k in exp or len(k) >= kmax:
            continue
        exp.add(k)
        seq = [tuple(a) for a in k]
        moved = {i for i, _ in seq}
        todo = [o for o in root.movable if o not in moved]
        batch = sampler.propose_many(root, root.cells(seq), todo, M, gen)
        kids = [ck for o in todo for d in batch[o] for ck in [tuple(seq + [(o, int(d))])] if ck not in nodes]
        if not kids:
            continue
        for ck, s in zip(kids, scorer.score(root, [list(c) for c in kids])):
            nodes[ck] = {"seq": [list(a) for a in ck], **s, "m": len(ck)}
            heapq.heappush(heap, (cost(nodes[ck]), next(order), ck))
            fifo.append(ck)
    return nodes, len(exp)


def search(seed, kmax=6, max_exp=200):
    OUT.mkdir(parents=True, exist_ok=True)
    fo = OUT / f"search_s{seed}.jsonl"
    done = {json.loads(l)["root_id"] for l in open(fo)} if fo.exists() else set()
    samp, _, _ = BM.load_bundle(H.K3 / "stage/bundle", "2d", "cpu", 0)
    scor = H.HZ.HazardScorer(H.OUT / f"model_2d_s{seed}.pt")
    with open(fo, "a") as f:
        for root, rec, p, lab in roots():
            if root.id in done:
                continue
            t0 = time.time()
            nodes, nexp = deep_search(root, samp, scor, 4, kmax, max_exp, seed)
            plans = sorted((n for n in nodes.values() if n["m"]), key=lambda n: -n["p11"])[:TOPN]
            f.write(json.dumps({"root_id": root.id, "K": lab["K_certified"], "seed": seed, "kmax": kmax, "max_exp": max_exp,
                                "p_noop": nodes[()]["p11"], "n_candidates": len(nodes) - 1, "expansions": nexp,
                                "top": [{"seq": n["seq"], "p11": round(n["p11"], 6), "m": n["m"]} for n in plans],
                                "s": round(time.time() - t0, 2)}) + "\n")
            f.flush()
            print(f"{root.id} K{lab['K_certified']} s{seed}: {len(nodes) - 1} plans, best p {plans[0]['p11'] if plans else None:.3f} "
                  f"len {plans[0]['m'] if plans else None}, {time.time() - t0:.1f}s", flush=True)


def teach(workers):
    info = {r.id: (r, rec, p) for r, rec, p, _ in roots()}
    vf = OUT / "verify_2d.jsonl"
    tried = {(v["root"], json.dumps(x["seq"])) for v in BM.jl(vf) for x in v["results"]} if vf.exists() else set()
    want = {}
    for fo in sorted(OUT.glob("search_s*.jsonl")):
        for r in map(json.loads, open(fo)):
            for t in r["top"]:
                if (r["root_id"], json.dumps(t["seq"])) not in tried:
                    want.setdefault(r["root_id"], {})[json.dumps(t["seq"])] = t["seq"]
    jobs = []
    for rid, seqs in want.items():
        root, rec, p = info[rid]
        jobs.append({"root_id": rid, "dom": "2D", "path": str(p), "geom": rec.get("geometry_impl", "original"), "ids": root.ids,
                     "seqs": [[[root.ids[o], c] for o, c in s] for s in seqs.values()]})
    print(json.dumps({"roots": len(jobs), "queries": sum(len(j["seqs"]) for j in jobs)}), flush=True)
    if jobs:
        H._run("2d", jobs, workers, vf)


def report():
    L = H.V.labels("2d", str(OUT / "verify_2d.jsonl"))
    rows, agg = [], {}
    for fo in sorted(OUT.glob("search_s*.jsonl")):
        for r in map(json.loads, open(fo)):
            oc = [H.V.outcome(t["seq"], L, r["root_id"]) for t in r["top"]]
            ok1 = bool(oc) and oc[0] == "SUCCESS"
            first = next((i for i, o in enumerate(oc) if o == "SUCCESS"), None)
            rows.append({"root_id": r["root_id"], "K": r["K"], "seed": r["seed"], "top1": ok1, "within5": first is not None,
                         "len_top1": r["top"][0]["m"] if r["top"] else None, "outcomes": oc, "unknown": oc.count("UNKNOWN")})
            a = agg.setdefault(r["K"], {"runs": 0, "top1": 0, "within5": 0, "unknown_plans": 0})
            a["runs"] += 1; a["top1"] += ok1; a["within5"] += first is not None; a["unknown_plans"] += oc.count("UNKNOWN")
    out = {"label": "EXPLORATORY (scaling2d release_v2, K4-K6; no baseline)", "per_K": agg, "rows": rows}
    (OUT / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(agg, indent=1))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "search":
        search(int(a[0]), *(int(x) for x in a[1:]))
    elif c == "teach":
        teach(int(a[0]))
    elif c == "report":
        report()
