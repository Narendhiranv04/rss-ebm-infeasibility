"""Physical verification + metrics for the hazard arm on V3.4 pools (Mac-safe: no taskset / sched_setaffinity).

Teacher = V3.4's unchanged evaluate._verify_job behind construct._boot.  Known (root, seq) outcomes come from the ablation study's shared cache
(K3ABL/runs/verifier_cache, read-only) plus our own cache ABF3/runs/hz/verify_{dom}.jsonl (append-only).  Outcomes / masks / metric
definitions = benchmark.correct + scripts/v32b_phys.outcome.

    python hz_eval.py teach DOM SPLIT SEED WORKERS     teach every REPAIR picked under any rule that no cache knows yet
    python hz_eval.py metrics DOM SPLIT SEED           per-rule task / noop / abstain (+ FULL_s0 reference on the same roots)
"""
import collections, json, os, re, sys, time
from pathlib import Path

os.environ.setdefault("HZ_MODE", "v34")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import twod_hazard as HZ  # noqa: E402  (sets ABF_ROOT / K3 / sys.path to the V3.4 code)

K3, OUT = HZ.K3, HZ.OUT
os.environ.setdefault("V3OUT", str(OUT / "teach_out"))
_F3 = HZ.ABF / "fin3d"                                 # local copy of gvlab2 approachb_final_gen/final3d_v4 (= their env.sh FIN3D_CODE / FIN3D_SRC)
os.environ.setdefault("FIN3D_CODE", str(_F3 / "code"))
os.environ.setdefault("FIN3D_SRC", str(_F3 / "src/dataset_bd_classquota_without_home_position_no_overpass"))
os.environ.setdefault("FIN3D_STORE", str(OUT / "teach_out/fin3d_store"))
os.environ.setdefault("MUJOCO_GL", "disable")
MARGIN = 10.0   # non-inferiority margin (points) for noop / abstain in the primary decision rule
REMOTE = ("/home/projects/long-horizon/naren_workspace/k3abl/abf/", "/home/projects/long-horizon/abf_v2/root/")
import benchmark as BM  # noqa: E402
import importlib.util  # noqa: E402
_sp = importlib.util.spec_from_file_location("v32b_phys", HZ.CODE / "scripts/v32b_phys.py")
V = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(V)


def local(p):
    for pre in REMOTE:
        if p.startswith(pre):
            return str(HZ.ABF / p[len(pre):])
    return p


def labels(dom):
    """their cache (read-only) + ours; a decided label is never replaced by an undecided one; a decided disagreement raises."""
    L = V.labels(dom, str(K3 / "runs/verifier_cache" / f"verify_{dom}.jsonl"))
    for k, v in V.labels(dom, str(OUT / f"verify_{dom}.jsonl")).items():
        old = L.get(k)
        if old is None or None in old:
            L[k] = v if old is None or None not in v else old
        elif None not in v and v != old:
            raise SystemExit(f"CONFLICT between caches for {k}: {old} vs {v}")
    return L


def priv_of(dom, split):
    return {e["root_id"]: e for e in BM.jl(K3 / "runs/public" / f"private_{split}_{dom}.jsonl")}


def ours(dom, split, seed):
    """merge pool files (whole + parts), dedup by root; every root of the split must be present."""
    rows = {}
    for p in sorted(OUT.glob(f"pool_{dom}_{split}_s{seed}*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            old = rows.setdefault(r["root_id"], r)
            assert old["decisions"] == r["decisions"], f"{r['root_id']}: two pool files disagree ({p.name})"   # no silent re-run channel
    miss = set(priv_of(dom, split)) - set(rows)
    assert not miss, f"{len(miss)} roots not scored yet"
    return list(rows.values())


def rule_ids(split, dom):
    """VAL: every rule (tuning).  TEST: only the frozen rule and the VAL-chosen rule (chosen_rule_{dom}.json must exist)."""
    if split == "val":
        return list(range(len(HZ.RULES)))
    ch = OUT / f"chosen_rule_{dom}.json"
    assert ch.exists(), "run `tune` on VAL (and freeze it) before touching TEST"
    return sorted({0, json.loads(ch.read_text())["rule_index"]})


def teach(dom, split, seed, workers):
    priv, ks = priv_of(dom, split), rule_ids(split, dom)
    L, jobs = labels(dom), []
    tried = {(v["root"], json.dumps(x["seq"])) for v in BM.jl(OUT / f"verify_{dom}.jsonl") for x in v["results"]} if (OUT / f"verify_{dom}.jsonl").exists() else set()
    for r in ours(dom, split, seed):
        e = priv[r["root_id"]]
        seqs = {json.dumps(r["decisions"][k]) for k in ks if r["decisions"][k]}
        todo = [json.loads(s) for s in sorted(seqs) if None in L.get((r["root_id"], s), (None, None)) and (r["root_id"], s) not in tried]
        if todo:
            jobs.append({"root_id": r["root_id"], "dom": dom.upper(), "path": local(e["path"]), "geom": e.get("geom", "original"),
                         "ids": e["ids"], "seqs": [[[e["ids"][o], c] for o, c in s] for s in todo]})
    print(json.dumps({"dom": dom, "split": split, "roots": len(jobs), "queries": sum(len(j["seqs"]) for j in jobs)}), flush=True)
    _run(dom, jobs, workers, OUT / f"verify_{dom}.jsonl")


def _full_rows(dom, split, s):
    return list(BM.jl(HZ.ABF / "code_v34/ablations_k3/reports" / split / f"results_FULL_s{s}_{dom}.jsonl"))


def mac_labels(dom, split):
    """labels produced by the Mac teacher only: ours (verify_{dom}.jsonl) + crosscheck_{dom}_{split}.jsonl."""
    L = V.labels(dom, str(OUT / f"verify_{dom}.jsonl"))
    for k, v in V.labels(dom, str(OUT / f"crosscheck_{dom}_{split}.jsonl")).items():
        if L.get(k) is None or None in L[k]:
            L[k] = v
    return L


def crosscheck(dom, split, workers, seeds=(0, 1, 2)):
    """Teacher parity at scale: every REPAIR pick of FULL s0-2 and of ours s0-2 (rules used on this split) that has no Mac label yet is
    re-run on the Mac teacher (separate file, never merged into labels()).  Reports agreement with their gvlab-cached labels."""
    priv, fo = priv_of(dom, split), OUT / f"crosscheck_{dom}_{split}.jsonl"
    have = set(mac_labels(dom, split)) if fo.exists() or (OUT / f"verify_{dom}.jsonl").exists() else set()
    want = collections.defaultdict(set)
    for s in seeds:
        for x in _full_rows(dom, split, s):
            if x["decision"] == "REPAIR":
                want[x["root_id"]].add(json.dumps(x["seq"]))
        for r in ours(dom, split, s):
            for k in rule_ids(split, dom):
                if r["decisions"][k]:
                    want[r["root_id"]].add(json.dumps(r["decisions"][k]))
    jobs = []
    for rid, seqs in sorted(want.items()):
        todo = [json.loads(q) for q in sorted(seqs) if (rid, q) not in have]
        if todo:
            e = priv[rid]
            jobs.append({"root_id": rid, "dom": dom.upper(), "path": local(e["path"]), "geom": e.get("geom", "original"), "ids": e["ids"],
                         "seqs": [[[e["ids"][o], c] for o, c in q] for q in todo]})
    print(json.dumps({"dom": dom, "split": split, "picks": sum(map(len, want.values())), "todo": sum(len(j["seqs"]) for j in jobs)}), flush=True)
    _run(dom, jobs, workers, fo)
    their, mac = V.labels(dom, str(K3 / "runs/verifier_cache" / f"verify_{dom}.jsonl")), mac_labels(dom, split)
    keys = [(r, q) for r, qs in want.items() for q in qs]
    both = [k for k in keys if None not in their.get(k, (None,)) and None not in mac.get(k, (None,))]
    out = {"dom": dom, "split": split, "n_compared": len(both), "agree": sum(their[k] == mac[k] for k in both),
           "disagreements": [{"key": k, "theirs": their[k], "mac": mac[k]} for k in both if their[k] != mac[k]],
           "undecided_mac": sum(None in mac.get(k, (None,)) for k in keys)}
    (OUT / f"crosscheck_{dom}_{split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "disagreements"} | {"n_disagree": len(out["disagreements"])}))


def _run(dom, jobs, workers, outfile):
    import multiprocessing as mp
    import construct as CO
    import evaluate as EV
    Path(os.environ["V3OUT"]).mkdir(parents=True, exist_ok=True)
    with open(outfile, "a") as f:
        for g in sorted({j["geom"] for j in jobs}):
            sub = [j for j in jobs if j["geom"] == g]
            with mp.get_context("spawn").Pool(workers, initializer=CO._boot, initargs=(dom.upper(), g), maxtasksperchild=50) as pool:
                for i, r in enumerate(pool.imap_unordered(EV._verify_job, sub), 1):
                    ids = next(j["ids"] for j in sub if j["root_id"] == r["root"])
                    for x in r["results"]:
                        x["seq"] = [[ids.index(o), c] for o, c in x["seq"]]
                    r["teacher_provenance"] = "unchanged finalized teacher (evaluate._verify_job), V3.4 harness, on Mac"
                    f.write(json.dumps(r) + "\n"); f.flush()
                    if i % 25 == 0:
                        print(f"{time.strftime('%H:%M:%S')} {g} {i}/{len(sub)}", flush=True)


def _xy(dom, x):
    """(x, y) exactly as v32b_phys.labels reads them (3D: only resolution EVALUATED counts)."""
    out = []
    for k in ("x", "y"):
        r = x[k] or {}
        v = r.get(k) if dom == "2d" or r.get("resolution", "EVALUATED") == "EVALUATED" else None
        out.append(v if v in (0, 1) else None)
    return tuple(out)


def parity(dom, geom):
    """re-run one decided cached success + one decided failure of this geometry on this Mac; outcomes must match theirs.
    The teacher allows one geometry per process: run once per geometry (original, fastgeom1)."""
    import construct as CO
    import evaluate as EV
    priv, want = priv_of(dom, "val"), {}
    for v in BM.jl(K3 / "runs/verifier_cache" / f"verify_{dom}.jsonl"):
        e = priv.get(v["root"])
        for x in v["results"] if e and e["geom"] == geom else []:
            xy = _xy(dom, x)
            if x["seq"] and None not in xy:
                want.setdefault((e["geom"], xy == (1, 1)), (e, x["seq"], xy))
    bad = 0
    for g in sorted({k[0] for k in want}):
        CO._boot(dom.upper(), g)
        for (gg, _), (e, seq, xy) in sorted(want.items(), key=lambda t: str(t[0])):
            if gg != g:
                continue
            r = EV._verify_job({"dom": dom.upper(), "path": local(e["path"]), "seqs": [[[e["ids"][o], c] for o, c in seq]]})
            got = _xy(dom, r["results"][0])
            bad += got != xy
            print(json.dumps({"geom": g, "root": e["root_id"], "seq": seq, "theirs": xy, "mac": got, "cpu_s": r["cpu_s"]}), flush=True)
    raise SystemExit(f"PARITY FAILED on {bad}" if bad else 0)


def rates(rows):
    t, n, a = zip(*(BM.correct(x) for x in rows))
    m = lambda v: None if not [x for x in v if x is not None] else round(100 * sum(x for x in v if x is not None) / len([x for x in v if x is not None]), 1)  # noqa: E731
    return {"task": m(t), "noop": m(n), "abstain": m(a), "n_task": sum(x is not None for x in t),
            "unknown_task": sum(x["outcome"] == "UNKNOWN" and x["masks"].get("eval_task") for x in rows)}


def tune(dom, seed):
    """VAL only.  Criterion fixed 2026-11-21 before any VAL physics: max mean(task, noop, abstain); ties -> higher task -> lower rule index."""
    m = json.loads((OUT / f"metrics_{dom}_val_s{seed}.json").read_text())["hazard_by_rule"]
    sc = [(sum(v[x] for x in ("task", "noop", "abstain")) / 3, v["task"], -k, k) for k, v in enumerate(m.values()) if v["unknown_task"] == 0]
    best = max(sc)
    out = {"dom": dom, "seed_tuned_on": seed, "split": "val", "rule_index": best[3], "rule": HZ.RULES[best[3]], "val_objective": round(best[0], 2),
           "criterion": "max mean(task,noop,abstain) on VAL; ties: higher task, lower index"}
    (OUT / f"chosen_rule_{dom}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


def _cluster(rid):
    """world id: scene variants of one generated world share it (e.g. 2d-[conv-]plus2d-srv-w0052-s005 -> 2d-plus2d-srv-w0052)."""
    return re.sub(r"-s\d+$", "", rid).replace("-conv-", "-")


def _effect(per_root, B, rng, cl):
    """per_root {rid: (ours mean, full mean)} -> mean diff + cluster (world) bootstrap and root bootstrap percentile quantiles."""
    import numpy as np
    ks = sorted(per_root)
    d = np.array([per_root[k][0] - per_root[k][1] for k in ks])
    groups = collections.defaultdict(list)
    for i, k in enumerate(ks):
        groups[cl(k)].append(i)
    G = list(groups.values())
    assert len(G) < len(ks), "cluster key matched nothing: every root is its own world"
    cb = [np.concatenate([G[j] for j in rng.integers(0, len(G), len(G))]) for _ in range(B)]
    cboot = np.array([d[ix].mean() for ix in cb])
    rboot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(B)])
    q = lambda v, p: round(100 * float(np.quantile(v, p)), 2)  # noqa: E731
    return {"n": len(ks), "clusters": len(G), "ours": round(100 * float(np.mean([per_root[k][0] for k in ks])), 2),
            "full": round(100 * float(np.mean([per_root[k][1] for k in ks])), 2), "diff": round(100 * float(d.mean()), 2),
            "lb97.5_cluster": q(cboot, 0.025), "ub97.5_cluster": q(cboot, 0.975), "lb97.5_root": q(rboot, 0.025)}


def confirm(dom, split, seeds=(0, 1, 2), B=10000, margin=MARGIN):
    """Pre-registered analysis (docs/HZ_V34_PREREG.md).  Per root and metric (task / noop / abstain, BM.correct masks): indicator averaged over
    seeds (UNKNOWN seed-runs dropped for that root/arm); ours FROZEN rule vs FULL_s{seed}.  Effect = mean(ours - FULL) over roots where both arms
    have >= 1 known run.  Percentile bootstrap, resampling WORLDS (clusters), B, rng 20261122; lb97.5 = np.quantile(boot, 0.025).
    PRIMARY (supported): task lb97.5_cluster > 0  AND  noop, abstain lb97.5_cluster > -margin (non-inferiority, points).
    Sensitivities: UNKNOWN counted as failure; both arms on Mac-teacher labels only (mac_labels); root-level bootstrap (reported in each block)."""
    import numpy as np
    priv, Lm = priv_of(dom, split), labels(dom)
    Lmac = mac_labels(dom, split) if (OUT / f"crosscheck_{dom}_{split}.jsonl").exists() else None
    runs = {"ours": [{r["root_id"]: r["decisions"][0] for r in ours(dom, split, s)} for s in seeds],
            "full": [{x["root_id"]: x for x in _full_rows(dom, split, s)} for s in seeds]}

    def outcome(arm, s, rid, L):
        if arm == "ours":
            return V.outcome(runs["ours"][s][rid], Lm if L is None else L, rid)  # L None = primary: merged labels
        x = runs["full"][s][rid]
        # both arms' REPAIR picks scored from the same label table (VAL: identical to FULL's stored outcome, 1255/1255 2D, 698/698 3D)
        return x["outcome"] if x["decision"] != "REPAIR" else V.outcome(x["seq"], Lm if L is None else L, rid)

    def block(L, unk_fail, metric):
        mi = ("task", "noop", "abstain").index(metric)
        per, unk = {}, {"ours": 0, "full": 0}
        for rid, e in priv.items():
            v = {}
            for arm in ("ours", "full"):
                xs = []
                for s in range(len(seeds)):
                    o = outcome(arm, s, rid, L)
                    if o == "UNKNOWN" and e["masks"].get("eval_" + metric):
                        unk[arm] += 1
                        o = "FAIL_UNKNOWN" if unk_fail else o
                    c = BM.correct({"masks": e["masks"], "outcome": o})[mi]
                    if c is not None:
                        xs.append(c)
                v[arm] = xs
            if v["ours"] and v["full"]:
                per[rid] = (np.mean(v["ours"]), np.mean(v["full"]))
        n_mask = sum(1 for e in priv.values() if e["masks"].get("eval_" + metric))
        return {**_effect(per, B, np.random.default_rng(20261122), _cluster), "n_mask": n_mask, "excluded": n_mask - len(per),
                "excl_gt2pct": (n_mask - len(per)) > 0.02 * n_mask, "unknown_runs": unk}

    out = {"dom": dom, "split": split, "seeds": list(seeds), "margin": margin, "bootstrap": f"percentile, worlds, B={B}, rng 20261122"}
    out["primary"] = {m: block(None, False, m) for m in ("task", "noop", "abstain")}
    p = out["primary"]
    out["supported"] = bool(p["task"]["lb97.5_cluster"] > 0 and all(p[m]["lb97.5_cluster"] > -margin for m in ("noop", "abstain")))
    out["sens_unknown_as_fail"] = {m: block(None, True, m) for m in ("task", "noop", "abstain")}
    out["sens_mac_teacher"] = {m: block(Lmac, False, m) for m in ("task", "noop", "abstain")} if Lmac is not None else "run crosscheck first"
    (OUT / f"confirm_{dom}_{split}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"supported": out["supported"], **{m: {k: p[m][k] for k in ("n", "ours", "full", "diff", "lb97.5_cluster", "lb97.5_root", "excluded")} for m in p},
                      "mac_task": out["sens_mac_teacher"]["task"] if Lmac is not None else None, "unkfail_task_lb": out["sens_unknown_as_fail"]["task"]["lb97.5_cluster"]}))


def metrics(dom, split, seed):
    priv = priv_of(dom, split)
    L, res = labels(dom), {}
    pr = ours(dom, split, seed)
    for k in rule_ids(split, dom):
        rule = HZ.RULES[k]
        rows = [{"masks": priv[r["root_id"]]["masks"], "outcome": V.outcome(r["decisions"][k], L, r["root_id"])} for r in pr]
        res[json.dumps({"noop_gate": rule["noop_gate"], "eta": rule["eta"]}) + (" FROZEN" if k == 0 else "")] = rates(rows)
    ref = K3.parent / "ABF3/code_v34/ablations_k3/reports" / split / f"results_FULL_s0_{dom}.jsonl"
    roots = {r["root_id"] for r in pr}
    full = rates([x for x in BM.jl(ref) if x["root_id"] in roots]) if ref.exists() else None
    out = {"dom": dom, "split": split, "seed": seed, "n_roots": len(pr), "FULL_s0_same_roots": full, "hazard_by_rule": res}
    (OUT / f"metrics_{dom}_{split}_s{seed}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"FULL_s0": full, "hazard_frozen": res[next(iter(res))]}, indent=1))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "teach":
        teach(a[0], a[1], int(a[2]), int(a[3]))
    elif c == "metrics":
        metrics(a[0], a[1], int(a[2]))
    elif c == "confirm":
        confirm(a[0], a[1])
    elif c == "crosscheck":
        crosscheck(a[0], a[1], int(a[2]))
    elif c == "parity":
        parity(a[0], a[1])
    elif c == "tune":
        tune(a[0], int(a[1]))
    else:
        raise SystemExit(__doc__)
