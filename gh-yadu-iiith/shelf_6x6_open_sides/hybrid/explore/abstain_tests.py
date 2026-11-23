"""EXPLORATORY, 3D VAL only (TEST never read): can the PURE learned scorer give up better on impossible scenes?  (user 2026-11-23)
Ideas from the ARS literature scan (no run-time simulator check, no retraining):
  ens   seed ensemble: score every candidate with all 3 seeds; plan score = mean / min / mean - std over seeds (pessimistic ensemble,
        Lakshminarayanan et al. 2017); then the usual frozen decision logic (noop gate 0.5, best plan if score >= eta, else give up)
  conf  other confidence signals for the act/give-up choice on each seed's own pick: top score s1, gap s1 - s2, weakest step of the top plan
        (selective classification, Geifman & El-Yaniv 2017)
  cross pick with one seed, judge the pick with the other two (sample splitting against the winner's curse, Zrnic & Fithian 2024)
  N     is wrong acting on impossible scenes more frequent when there are more candidates (optimizer's curse, Smith & Winkler 2006)?
Thresholds are chosen by 5-fold cross-validation over WORLDS (hz_eval._cluster), so every reported number is out-of-fold.
    python abstain_tests.py score i/k        -> runs/explore/abstain/scores_3d_val.part{i}of{k}.jsonl  (every candidate, 3 seeds)
    python abstain_tests.py teach WORKERS    label every task-scene pick that no cache knows (teacher)
    python abstain_tests.py report           -> runs/explore/abstain/report.json
"""
import glob, hashlib, json, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import twod_hazard as HZ  # noqa: E402  (frozen; read-only)

OUT = HZ.RUNS / "explore/abstain"
os.environ["V3OUT"] = str(OUT / "teach_out")
os.environ["FIN3D_STORE"] = str(OUT / "teach_out/fin3d_store")
import hz_eval as H  # noqa: E402
import numpy as np  # noqa: E402

DOM, SPLIT, GATE, ETA = "3d", "val", 0.5, 0.3
ETAS = [round(x, 2) for x in np.arange(0.30, 0.951, 0.05)]
AGGS = ("seed0", "seed1", "seed2", "mean", "min", "mean_minus_std")


def score(part):
    import torch
    import exhaustive as EX
    import search as S
    torch.set_num_threads(1)
    i, k = map(int, part.split("/"))
    OUT.mkdir(parents=True, exist_ok=True)
    fo = OUT / f"scores_{DOM}_{SPLIT}.part{i}of{k}.jsonl"
    done = {json.loads(l)["root_id"] for l in open(fo)} if fo.exists() else set()
    rows = [json.loads(l) for l in open(H.K3 / "runs/public" / f"public_{SPLIT}_{DOM}.jsonl")]
    ref = {r["root_id"]: r for r in map(json.loads, open(H.K3 / "runs/predictions/FULL_s0" / f"{SPLIT}_{DOM}.jsonl"))}
    sc = [HZ.HazardScorer(H.OUT / f"model_{DOM}_s{s}.pt") for s in (0, 1, 2)]
    with open(fo, "a") as f:
        for r in rows:
            if r["root_id"] in done or int(hashlib.sha256(r["root_id"].encode()).hexdigest(), 16) % k != i:
                continue
            root = S.Root(r, DOM)
            D = {int(o): [int(d) for d in v[:4]] for o, v in ref[r["root_id"]]["proposals"].items()}
            assert HZ.pool_hash(D) == ref[r["root_id"]]["pool_hash"], r["root_id"]
            seqs = [list(map(list, s)) for s, rk in EX.stream(D, 4) if rk < 4]
            rec = {"root_id": r["root_id"], "seqs": seqs, "p": [], "minstep": [], "p_noop": []}
            for x in sc:
                rec["p_noop"].append(round(x.score(root, [[]])[0]["p11"], 6))
                rec["p"].append([round(c["p11"], 6) for c in x.score(root, seqs)])
                st = [HZ.steps_of(root, [tuple(a) for a in s]) for s in seqs]
                rec["minstep"].append([round(min(x.cache[y[0]] for y in s), 6) for s in st])
            f.write(json.dumps(rec) + "\n"); f.flush()


def load():
    out = {}
    for p in sorted(OUT.glob(f"scores_{DOM}_{SPLIT}.part*.jsonl")):
        for l in open(p):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            out.setdefault(r["root_id"], r)
    return out


def decide(p, p_noop, eta):
    """frozen V3.4 logic (exhaustive.Selectors.finish, alpha = beta = gamma = 0): -> (index | 'NOOP' | None, top score).
    shortcut: ties broken by lowest index, not V3.4's key(); report() checks agreement with the stored picks."""
    if p_noop >= GATE:
        return "NOOP", p_noop
    i = int(np.argmax(p))
    if p[i] < eta:
        return ("NOOP" if p_noop >= eta else None), float(p[i])
    return ("NOOP" if p_noop >= eta and p_noop > p[i] else i), float(p[i])


def agg(rec, how):
    P, N = np.array(rec["p"]), np.array(rec["p_noop"])
    if how.startswith("seed"):
        s = int(how[-1]); return P[s], N[s]
    if how == "mean":
        return P.mean(0), N.mean()
    if how == "min":
        return P.min(0), N.min()
    return np.clip(P.mean(0) - P.std(0), 0, 1), max(N.mean() - N.std(), 0.0)


def labels():
    L = H.labels(DOM)
    conflicts = 0
    for vf in sorted(set(glob.glob(str(HZ.RUNS / "explore/**/verify_3d.jsonl"), recursive=True))):
        for key, v in H.V.labels(DOM, vf).items():
            old = L.get(key, (None, None))
            if None in old:
                L[key] = v
            elif None not in v and v != old:
                conflicts += 1                               # keep the primary table's label
    return L, conflicts


def picks(R):
    """every (root, rule, eta) pick that needs a task label"""
    priv = H.priv_of(DOM, SPLIT)
    need = {}
    for rid, rec in R.items():
        if not priv[rid]["masks"].get("eval_task"):
            continue
        for how in AGGS:
            p, pn = agg(rec, how)
            for eta in ETAS:
                d, _ = decide(p, pn, eta)
                if isinstance(d, int):
                    need.setdefault(rid, set()).add(json.dumps(rec["seqs"][d]))
    return need


def teach(workers):
    R, priv = load(), H.priv_of(DOM, SPLIT)
    assert set(priv) <= set(R), f"{len(set(priv) - set(R))} VAL roots not scored yet"
    L, _ = labels()
    jobs = []
    for rid, ss in picks(R).items():
        todo = [json.loads(s) for s in ss if None in L.get((rid, s), (None, None))]
        if todo:
            e = priv[rid]
            jobs.append({"root_id": rid, "dom": DOM.upper(), "path": H.local(e["path"]), "geom": e.get("geom", "original"), "ids": e["ids"],
                         "seqs": [[[e["ids"][o], c] for o, c in s] for s in todo]})
    print(json.dumps({"roots": len(jobs), "queries": sum(len(j["seqs"]) for j in jobs)}), flush=True)
    if jobs:
        H._run(DOM, jobs, workers, OUT / f"verify_{DOM}.jsonl")


def outcome_rows(R, L, priv, choose):
    """choose(rid, rec) -> index | 'NOOP' | None  ->  rows for H.rates"""
    rows = []
    for rid, rec in R.items():
        d = choose(rid, rec)
        dec = [] if d == "NOOP" else None if d is None else rec["seqs"][d]
        rows.append({"masks": priv[rid]["masks"], "outcome": H.V.outcome(dec, L, rid), "rid": rid})
    return rows


def cv(R, L, priv, signal_of, grid, folds=5):
    """out-of-fold: per fold pick the threshold maximising mean(task, noop, abstain) on the other folds (ties -> higher task -> lower t)."""
    worlds = sorted({H._cluster(r) for r in R})
    fold = {w: int(hashlib.sha256(("cv" + w).encode()).hexdigest(), 16) % folds for w in worlds}
    chosen, out = [], []
    for f in range(folds):
        tr = {r: x for r, x in R.items() if fold[H._cluster(r)] != f}
        te = {r: x for r, x in R.items() if fold[H._cluster(r)] == f}
        best = None
        for t in grid:
            m = H.rates(outcome_rows(tr, L, priv, lambda rid, rec: signal_of(rec, t)))
            k = ((m["task"] + m["noop"] + m["abstain"]) / 3, m["task"], -t)
            best = max(best, (k, t)) if best else (k, t)
        chosen.append(best[1])
        out += outcome_rows(te, L, priv, lambda rid, rec: signal_of(rec, best[1]))
    return H.rates(out), chosen


def report():
    R, priv = load(), H.priv_of(DOM, SPLIT)
    assert set(priv) <= set(R), f"{len(set(priv) - set(R))} VAL roots not scored yet"
    R = {r: R[r] for r in priv}
    L, conflicts = labels()
    res = {"label": "EXPLORATORY, 3D VAL only; CV thresholds are out-of-fold (5 folds over worlds); TEST never read",
           "n_roots": len(R), "label_conflicts_ignored": conflicts}
    # 1. reproduction: our decide() vs the stored frozen-rule picks per seed
    rep = {}
    for s in (0, 1, 2):
        st = {r["root_id"]: r["decisions"][0] for r in H.ours(DOM, SPLIT, s)}
        same = 0
        for rid, rec in R.items():
            d, _ = decide(*agg(rec, f"seed{s}"), ETA)
            dec = [] if d == "NOOP" else None if d is None else rec["seqs"][d]
            same += (dec == st[rid])
        rep[s] = same
    res["reproduces_stored_frozen_picks"] = {**rep, "of": len(R)}
    assert min(rep.values()) >= 0.97 * len(R), f"re-scoring does not reproduce the stored picks: {rep}"
    # 2. ensemble rules at fixed eta (frozen 0.3) and with a CV-chosen eta
    res["ensemble"] = {}
    for how in AGGS:
        fixed = H.rates(outcome_rows(R, L, priv, lambda rid, rec: decide(*agg(rec, how), ETA)[0]))
        cvm, ch = cv(R, L, priv, lambda rec, t: decide(*agg(rec, how), t)[0], ETAS)
        res["ensemble"][how] = {"eta_0.3": fixed, "eta_cv": cvm, "cv_etas": ch}
    # 3. confidence signals on each seed's own pick (act at eta 0.3, then give up if signal < t)
    def sig_rule(kind, s):
        def f(rec, t):
            p, pn = agg(rec, f"seed{s}")
            d, s1 = decide(p, pn, ETA)
            if not isinstance(d, int):
                return d
            if kind == "s1":
                v = s1
            elif kind == "gap":
                v = s1 - float(np.partition(p, -2)[-2]) if len(p) > 1 else s1
            elif kind == "minstep":
                v = rec["minstep"][s][d]
            elif kind == "crossseed":                        # pick with seed s, judge the pick with the other two (sample splitting, Zrnic & Fithian 2024)
                v = float(np.mean([rec["p"][o][d] for o in (0, 1, 2) if o != s]))
            else:
                raise ValueError(kind)
            return d if v >= t else ("NOOP" if pn >= ETA else None)
        return f
    grids = {"s1": ETAS, "gap": [0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3], "minstep": [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95], "crossseed": [0.0] + ETAS}
    res["signals"] = {kind: {s: dict(zip(("rates", "cv_thresholds"), cv(R, L, priv, sig_rule(kind, s), g))) for s in (0, 1, 2)}
                      for kind, g in grids.items()}
    # 4. optimizer's curse: wrong acts on impossible scenes vs number of candidates
    none = [rid for rid in R if priv[rid]["masks"].get("eval_abstain")]
    by = []
    for rid in none:
        n = len(R[rid]["seqs"])
        acts = sum(isinstance(decide(*agg(R[rid], f"seed{s}"), ETA)[0], int) for s in (0, 1, 2))
        by.append((n, acts))
    by.sort()
    h = len(by) // 2
    res["wrong_acts_vs_candidates"] = {"n_impossible_scenes": len(by), "median_candidates": by[h][0] if by else None,
                                       "wrong_act_runs_fewer_half": sum(a for _, a in by[:h]), "wrong_act_runs_more_half": sum(a for _, a in by[h:]),
                                       "runs_per_half": [3 * h, 3 * (len(by) - h)]}
    (OUT / "report.json").write_text(json.dumps(res, indent=1))
    show = {how: {k: {m: v[k][m] for m in ("task", "noop", "abstain")} for k in ("eta_0.3", "eta_cv")} for how, v in res["ensemble"].items()}
    print(json.dumps({"repro": res["reproduces_stored_frozen_picks"], "ensemble": show, "vsN": res["wrong_acts_vs_candidates"]}, indent=1))
    for kind, v in res["signals"].items():
        print(kind, {s: {m: x["rates"][m] for m in ("task", "noop", "abstain")} for s, x in v.items()})


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    {"score": lambda: score(a[0]), "teach": lambda: teach(int(a[0])), "report": report}[c]()
