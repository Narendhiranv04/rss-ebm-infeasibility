"""P17 analyses (frozen before any prospective outcome is inspected).

B: root-level outcomes on DECIDED originals, separated into the no-op decision (feasible originals), the repair decision (infeasible
originals) and admission (how often a non-empty repair is chosen, and its physical precision); strict TOP1 = correct no-op OR strict repair
success (VERIFIED + replay CLEAR). Baselines ALWAYS_EMPTY, random, geometry and VERIFY_FIRST (its physical calls and CPU counted).
D: p16_eval.d_analysis on the common roots (primary Dp17 vs geometry) + effort (nodes / virtual CPU to the first verified plan)."""
from __future__ import annotations

import numpy as np

DECIDED = ("VERIFIED", "CERTIFIED_INFEASIBLE", "STATIC_INVALID")
B_PRIMARY = ("LB", "empty_always")
B_SECONDARY = (("LB", "verify_first"), ("LB+G", "LB"), ("LB+G", "LB+G|nocpl"), ("LB+G", "LB+G@LB"), ("LB+G", "LB-W"), ("LB", "FULLB"),
               ("LB", "LB-F2"), ("LB", "LC"), ("LB", "A"), ("LB", "SF"), ("LB", "P13:K-B"), ("LB+G", "empty_always"), ("LB", "geometry"),
               ("LB#top3", "LB"), ("LB#top3", "verify_first"))


def _ok_noop(x):
    return x["choice"].endswith(":empty") and x["status"] == "VERIFIED"


def _ok_rep(x):
    return x["choice"] not in ("ABSTAIN",) and not x["choice"].endswith(":empty") and x["status"] == "VERIFIED" and bool(x.get("replay"))


def _vf(v):
    """VERIFY_FIRST as an arm row: EMPTY if the original target certifies, else SEARCH (geometry-guided corrected search)."""
    x = v["verify_first"]
    return {"choice": "seq:x:empty" if x["choice"] == "EMPTY" else "SEARCH", "status": x["status"], "replay": x["status"] == "VERIFIED",
            "moves": x.get("moves"), "travel": x.get("travel"), "used": x.get("used"), "cpu_virtual": x.get("cpu_virtual")}


def b_analysis(rows) -> dict:
    """rows: {root: {'family', 'b': b_root output (+ 'verify_first')}}."""
    import p16_eval as p16e
    R = {k: v for k, v in rows.items() if v["b"]["y0_status"] in DECIDED}
    arm_rows = {k: dict(v["b"]["arms"], **({"verify_first": _vf(v["b"])} if "verify_first" in v["b"] else {})) for k, v in R.items()}
    feas = [k for k in R if R[k]["b"]["y0_status"] == "VERIFIED"]
    inf = [k for k in R if k not in feas]
    arms = sorted({a for v in arm_rows.values() for a in v})
    ok = lambda k, a: _ok_noop(arm_rows[k][a]) if k in feas else _ok_rep(arm_rows[k][a])  # noqa: E731
    per = {}
    for a in arms:
        ks = [k for k in R if a in arm_rows[k]]
        f, i = [k for k in feas if k in ks], [k for k in inf if k in ks]
        ch = lambda k: arm_rows[k][a]["choice"]  # noqa: E731
        rep = lambda k: ch(k) not in ("ABSTAIN",) and not ch(k).endswith(":empty")  # noqa: E731
        succ_rep = [k for k in i if _ok_rep(arm_rows[k][a])]
        used = [arm_rows[k][a].get("used") or {} for k in ks]
        per[a] = {"n": len(ks), "strict_top1": float(np.mean([ok(k, a) for k in ks])) if ks else None,
                  "noop": {"feasible_n": len(f), "correct": sum(_ok_noop(arm_rows[k][a]) for k in f), "unnecessary_relocation": sum(rep(k) for k in f),
                           "abstain": sum(ch(k) == "ABSTAIN" for k in f)},
                  "repair": {"infeasible_n": len(i), "success": len(succ_rep), "wrong_noop": sum(ch(k).endswith(":empty") for k in i),
                             "abstain": sum(ch(k) == "ABSTAIN" for k in i), "attempted": sum(rep(k) for k in i),
                             "moves_mean": float(np.mean([arm_rows[k][a]["moves"] for k in succ_rep])) if succ_rep else None,
                             "travel_mean": float(np.mean([arm_rows[k][a]["travel"] for k in succ_rep])) if succ_rep else None},
                  "admission": {"repairs_chosen": sum(rep(k) for k in ks), "rate": float(np.mean([rep(k) for k in ks])) if ks else None,
                                "precision": float(np.mean([arm_rows[k][a]["status"] == "VERIFIED" for k in ks if rep(k)])) if any(rep(k) for k in ks) else None,
                                "on_feasible": sum(rep(k) for k in f), "on_infeasible": sum(rep(k) for k in i)},
                  "physical_calls_mean": {c: float(np.mean([u.get(c, 0) for u in used])) for c in ("motion_checks", "online_moves", "goal_plans", "transit_plans")},
                  "cpu_virtual_mean": float(np.mean([arm_rows[k][a]["cpu_virtual"] for k in ks])) if a == "verify_first" and ks else None}
    a, b = B_PRIMARY
    common = lambda x, y: [k for k in R if x in arm_rows[k] and y in arm_rows[k]]  # noqa: E731
    prim = p16e.paired([ok(k, a) for k in common(a, b)], [ok(k, b) for k in common(a, b)]) if a in arms and b in arms else None
    sec = {}
    for x, y in B_SECONDARY:
        if x in arms and y in arms:
            ks = common(x, y)
            sec[f"{x}_vs_{y}|strict_top1"] = p16e.paired([ok(k, x) for k in ks], [ok(k, y) for k in ks])
            sec[f"{x}_vs_{y}|noop"] = p16e.paired([_ok_noop(arm_rows[k][x]) for k in ks if k in feas], [_ok_noop(arm_rows[k][y]) for k in ks if k in feas])
            sec[f"{x}_vs_{y}|repair"] = p16e.paired([_ok_rep(arm_rows[k][x]) for k in ks if k in inf], [_ok_rep(arm_rows[k][y]) for k in ks if k in inf])
    h = p16e.holm({k: v["p"] for k, v in sec.items() if k.endswith("|strict_top1")})
    for k in sec:
        sec[k]["p_holm_top1_family"] = h.get(k)
    fam = {}
    for k, v in R.items():
        fam.setdefault(v["family"], []).append(k)
    by_family = {f: {x: float(np.mean([ok(k, x) for k in ks if x in arm_rows[k]])) for x in arms} | {"n": len(ks)} for f, ks in fam.items()}
    return {"decided_roots": len(R), "undecided_originals": len(rows) - len(R), "feasible": len(feas), "infeasible": len(inf), "per_arm": per,
            "primary": {"pair": list(B_PRIMARY), "strict_top1": prim}, "secondary": sec, "by_family": by_family}


def d_analysis(rows, unions=None, primary=("Dp17", "geometry")) -> dict:
    """rows: {root: {'family', 'guides': {guide: d_root output}}} -> p16_eval.d_analysis + effort to the first verified plan + by family."""
    import p16_eval as p16e
    out = p16e.d_analysis(rows, primary=primary, unions=unions)
    guides = sorted({g for v in rows.values() for g in v["guides"]})
    ks = [k for k, v in rows.items() if all(g in v["guides"] for g in guides)]
    eff = {}
    for g in guides:
        first = [(rows[k]["guides"][g]["log"].get("anytime") or [{}])[0] for k in ks if rows[k]["guides"][g]["found"]]
        eff[g] = {"nodes_to_first_mean": float(np.mean([x.get("nodes", np.nan) for x in first])) if first else None,
                  "cpu_to_first_mean": float(np.mean([x.get("cpu_s", np.nan) for x in first])) if first else None}
    fam = {}
    for k in ks:
        fam.setdefault(rows[k]["family"], []).append(k)
    ok = lambda k, g: bool(rows[k]["guides"][g]["found"] and rows[k]["guides"][g]["replay"])  # noqa: E731
    out["effort_first_plan"] = eff
    out["by_family"] = {f: {g: sum(ok(k, g) for k in kk) for g in guides} | {"n": len(kk)} for f, kk in fam.items()}
    return out


def b_offline(exec_rows, labels) -> dict:
    """Label-scored metrics of every arm's frozen deployable marginals on decided queries (UNKNOWN excluded): AUROC, average precision,
    Brier, ECE (10 bins), NLL; no-op (z0) Brier / NLL and Cohen's kappa of the calibrated no-op call incl. the constant baselines;
    within-root pairwise ranking of decided repairs; equal-coverage admission curves; certified reversed-order pairs."""
    import p17_b as pb
    arms = sorted({a for v in exec_rows.values() for a in v["b"]["scores"]})
    lab = {k: {q: x["y"] for q, x in pb.labels_of(k, labels[k], labels[k]["catalogue"]).items() if x["y"] is not None} for k in exec_rows if k in labels}
    sg = lambda u: 1 / (1 + np.exp(-u))  # noqa: E731
    out = {}
    for a in arms:
        P, Y, rank, rev, z0 = [], [], [], [], []
        for k, v in exec_rows.items():
            if k not in lab or a not in v["b"]["scores"]:
                continue
            sc = dict(v["b"]["scores"][a])
            for q, u in sc.items():
                if q in lab[k]:
                    P.append(sg(u))
                    Y.append(lab[k][q])
                    if q.endswith(":empty"):
                        z0.append((sg(u), lab[k][q]))
            rp = [(sc[q], lab[k][q]) for q in sc if q in lab[k] and not q.endswith(":empty")]
            pos, neg = [u for u, y in rp if y == 1], [u for u, y in rp if y == 0]
            if pos and neg:
                rank.append(np.mean([(p_ > n_) + 0.5 * (p_ == n_) for p_ in pos for n_ in neg]))
            for q in sc:
                body = q.rsplit(":", 1)[1]
                parts = body.split(".")
                if len(parts) == 2 and q in lab[k]:
                    rq = q.rsplit(":", 1)[0] + ":" + ".".join(parts[::-1])
                    if rq in sc and rq in lab[k] and lab[k][q] == 1 and lab[k][rq] == 0:
                        rev.append((sc[q] > sc[rq]) + 0.5 * (sc[q] == sc[rq]))
        P, Y = np.asarray(P), np.asarray(Y)
        if not len(P):
            continue
        order = np.argsort(-P)
        tp = np.cumsum(Y[order])
        auroc = float(np.mean([(p_ > n_) + 0.5 * (p_ == n_) for p_ in P[Y == 1] for n_ in P[Y == 0]])) if Y.min() == 0 and Y.max() == 1 else None
        ap = float(np.sum((tp / np.arange(1, len(P) + 1))[Y[order] == 1]) / max(1, Y.sum()))
        bins = np.minimum((P * 10).astype(int), 9)
        ece = float(sum(abs(P[bins == b].mean() - Y[bins == b].mean()) * (bins == b).mean() for b in range(10) if (bins == b).any()))
        nll = float(-np.mean(np.log(np.clip(np.where(Y == 1, P, 1 - P), 1e-12, 1))))
        zp, zy = (np.array([x for x, _ in z0]), np.array([y for _, y in z0])) if z0 else (np.array([]), np.array([]))
        out[a] = {"n_queries": int(len(P)), "positives": int(Y.sum()), "auroc": auroc, "average_precision": ap, "brier": float(np.mean((P - Y) ** 2)),
                  "ece10": ece, "nll": nll, "noop": {"n": int(len(zp)), "brier": float(np.mean((zp - zy) ** 2)) if len(zp) else None,
                                                     "brier_constant_prevalence": float(np.mean((zy.mean() - zy) ** 2)) if len(zp) else None},
                  "within_root_rank_auc": float(np.mean(rank)) if rank else None, "within_root_roots": len(rank),
                  "reversed_order_pairs": {"n": len(rev), "verified_order_scored_higher": float(np.mean(rev)) if rev else None}}
    return out


def kappa(calls, truth) -> float | None:
    """Cohen's kappa of binary calls against binary truth (None if undefined)."""
    c, t = np.asarray(calls, float), np.asarray(truth, float)
    if not len(c):
        return None
    po = float(np.mean(c == t))
    pe = float(c.mean() * t.mean() + (1 - c.mean()) * (1 - t.mean()))
    return None if pe == 1 else (po - pe) / (1 - pe)


def noop_kappa(exec_rows) -> dict:
    """Kappa of each arm's calibrated no-op call (EMPTY chosen) against the decided original status, plus the constant baselines."""
    R = {k: v for k, v in exec_rows.items() if v["b"]["y0_status"] in DECIDED}
    truth = [int(R[k]["b"]["y0_status"] == "VERIFIED") for k in R]
    arms = sorted({a for v in R.values() for a in v["b"]["arms"]})
    out = {a: kappa([int(R[k]["b"]["arms"][a]["choice"].endswith(":empty")) for k in R], truth) for a in arms}
    return out | {"constant_empty": kappa([1] * len(truth), truth), "constant_repair": kappa([0] * len(truth), truth), "n": len(truth),
                  "feasible_prevalence": float(np.mean(truth)) if truth else None}


def admission_curves(exec_rows, labels, arms, cover=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)) -> dict:
    """Equal-coverage comparison on decided infeasible originals: each arm admits its top-scored non-empty repair on the fraction c of roots
    with the highest admitted score (coverage set by rank, not by a tuned threshold); precision = label-verified fraction among admitted."""
    import p17_b as pb
    out = {}
    inf = [k for k, v in exec_rows.items() if v["b"]["y0_status"] in ("CERTIFIED_INFEASIBLE", "STATIC_INVALID") and k in labels]
    for a in arms:
        best = []
        for k in inf:
            sc = [(u, q) for q, u in exec_rows[k]["b"]["scores"].get(a, []) if not q.endswith(":empty")]
            if not sc:
                continue
            lab = {q: x["y"] for q, x in pb.labels_of(k, labels[k], labels[k]["catalogue"]).items()}
            u, q = max(sc)
            best.append((u, lab.get(q)))
        best.sort(key=lambda x: -x[0])
        out[a] = {str(c): (lambda sel: {"admitted": len(sel), "verified": sum(y == 1 for _, y in sel), "unknown": sum(y is None for _, y in sel)})(
            best[:max(1, int(round(c * len(best))))]) for c in cover} | {"roots": len(best)}
    return out


def d_offline(kind, rs, labs) -> dict:
    """Original-target D diagnosis under both label protocols (PARKING_LATTICE_V1 and MULTI_DEST_LATTICE_V1): feasibility AUROC / Brier /
    recalls; on MINIMAL_SETS lattices the feasibility-gated predicted mask ((1 - p) * marginal >= 0.5) against the union: F1, exact set,
    per-object precision / recall, union != necessary, alternative minimal sets; lattice-type (incomplete / no-clearing) rates."""
    import collections
    import d_blockers as db
    import p17_d as pd
    import p17_mechjobs as mj
    ms = pd.models(kind)
    out = {}
    for proto in ("PARKING_LATTICE_V1", "MULTI_DEST_LATTICE_V1"):
        rows, types = [], collections.Counter()
        for r in rs:
            lab = labs[r["key"]]
            pool_ids = lab["catalogue"]["pool"]
            if proto == "PARKING_LATTICE_V1":
                d = mj._lat(lab)[0]
            else:
                st = {tuple(x.split("+")) if x != "-" else (): v for x, v in lab["multi_dest"]["statuses"].items()}
                d = pd.lattice_from(st, pool_ids)
            types[d["type"]] += 1
            if d["y0"] is None:
                continue
            pool = [r["ids"].index(o) for o in pool_ids]
            pr = [db.predict(m, r["R"], pool) for m in ms]
            p = float(np.mean([x["p_feasible"] for x in pr]))
            marg = np.mean([x["marg"] for x in pr], 0) if pool else np.zeros(0)
            rows.append({"y0": d["y0"], "p": p, "type": d["type"], "mask": d["mask"], "necessary": d["necessary"], "sets": d["sets"],
                         "pred": [int((1 - p) * m_ >= 0.5) for m_ in marg]})
        ys, ps = np.array([x["y0"] for x in rows]), np.array([x["p"] for x in rows])
        pos, neg = ps[ys == 1], ps[ys == 0]
        dec = [x for x in rows if x["y0"] == 0 and x["type"] == "MINIMAL_SETS"]
        tp = sum(a and b for x in dec for a, b in zip(x["pred"], x["mask"]))
        fp = sum(a and not b for x in dec for a, b in zip(x["pred"], x["mask"]))
        fn = sum(b and not a for x in dec for a, b in zip(x["pred"], x["mask"]))
        out[proto] = {"n_decided_originals": len(rows), "feasible": int(ys.sum()) if len(rows) else 0, "lattice_types": dict(types),
                      "auroc": float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else None,
                      "brier": float(np.mean((ps - ys) ** 2)) if len(rows) else None,
                      "feasible_recall": float((pos >= 0.5).mean()) if len(pos) else None, "infeasible_recall": float((neg < 0.5).mean()) if len(neg) else None,
                      "masks": len(dec), "blocker_f1": 2 * tp / max(1, 2 * tp + fp + fn), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                      "exact_union": float(np.mean([x["pred"] == x["mask"] for x in dec])) if dec else None,
                      "union_ne_necessary": sum(x["mask"] != x["necessary"] for x in dec), "alternative_minimal_sets": sum(len(x["sets"]) >= 2 for x in dec),
                      "all_positive_masks": sum(all(x["mask"]) for x in dec)}
    return out
