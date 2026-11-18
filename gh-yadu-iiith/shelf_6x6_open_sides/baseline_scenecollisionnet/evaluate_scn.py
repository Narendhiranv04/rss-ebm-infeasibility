#!/usr/bin/env python3
"""Evaluate trained SceneCollisionNet runs on the V4 test split, with the metrics of the GRN / PIGINet baselines.

    python evaluate_scn.py s0 s1 s2 [--tag scn]          ->  results/<tag>.{md,json}

A. State feasibility (score = max over the 32 grasps of P(grasp works)): every labelled test query = "can T reach its goal in this layout?"
   (original scene, D edits, L lattice): AUROC / AP / accuracy / F1 at 0.5.
B. The network's own question, per grasp, on the original + D test queries that carry labels: grasp feasibility AUROC, "pick" collision
   AUROC (hand at the grasp pose is unusable) and "place" collision AUROC (hand over the goal collides), winner among the top-3 grasps.
C. Scene level, by enumerating layouts (original + every set of 1-3 obstacles parked at their first parking cell, as for GRN / PIGINet):
   predicted K = smallest set size whose best layout scores >= tau (else NONE); solvable vs NONE AUROC; blocker set given the true K (top-1 / top-3);
   blocker mask: SceneCollisionNet has no obstruction output, so an obstacle is a predicted blocker when parking it ALONE raises the
   state's score by >= delta (counterfactual gain; delta tuned on validation) -- compared with labels.blocker.mask like the others.
"""
import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import scn_data as D
import scn_geometry as G
from scn_model import SCN
import train_scn as T

HERE = Path(__file__).resolve().parent
CLASSES = ["0", "1", "2", "3", "NONE"]


def load(name):
    m = SCN(G.gripper_cloud()).to(T.DEV)
    m.load_state_dict(torch.load(HERE / "runs" / name / "best.pt", map_location=T.DEV))
    return m.eval()


def binm(y, p, tau=0.5):
    y, p = np.asarray(y), np.asarray(p)
    out = dict(n=int(len(y)), feasible=float(y.mean()), acc=float(((p >= tau) == y).mean()))
    if 0 < y.sum() < len(y):
        out.update(auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)), f1=float(f1_score(y, p >= tau)))
    return out


def grasp_metrics(states, P, PK, PL):
    fy = np.stack([s["fy"] for s in states]); pk = np.stack([s["pick"] for s in states]); pl = np.stack([s["place"] for s in states])
    out = {}
    m = fy >= 0
    if 0 < fy[m].sum() < m.sum():
        out["grasp_feas_auroc"] = float(roc_auc_score(fy[m], P[m])); out["grasp_feas_n"] = int(m.sum())
    for name, lab, p in (("pick", pk, PK), ("place", pl, PL)):
        mk = lab >= 0
        if mk.any() and 0 < lab[mk].sum() < mk.sum():
            out[f"{name}_auroc"] = float(roc_auc_score(lab[mk], p[mk])); out[f"{name}_n"] = int(mk.sum())
    # the teacher's winning grasp is a feasible grasp: is one of the 3 best-scoring grasps feasible-labelled?
    win = [(i, int(np.argmax(f))) for i, f in enumerate(fy) if (f == 1).any()]
    if win:
        out["winner_in_top3"] = float(np.mean([w in np.argsort(-P[i])[:3] for i, w in win]))
        out["winner_top1"] = float(np.mean([w == int(np.argmax(P[i])) for i, w in win]))
    return out


def layouts(r):
    init, goal, par = r["init"], r["goal"], r["parking"]
    obst = sorted(o["id"] for o in r["objects"] if o["id"] != "t0" and o["id"] in par)
    out = [((), dict(init))]
    for k in (1, 2, 3):
        for S in itertools.combinations(obst, k):
            cells = dict(init); cells.update({o: par[o] for o in S})
            if len(set(cells.values())) < len(cells) or goal in [par[o] for o in S]:
                continue
            out.append((S, cells))
    return out


def scene_scores(model, roots, rids, workers):
    items, owner = [], []
    for rid in rids:
        for S, cells in layouts(roots[rid]):
            items.append(dict(rid=rid, cells=cells, y=0, fy=np.zeros(32, np.int8), pick=np.zeros(32, np.int8), place=np.zeros(32, np.int8)))
            owner.append((rid, S))
    s = T.score_states(model, D.StateSet(items, roots), workers=workers)[0]
    res = {}
    for (rid, S), p in zip(owner, s):
        r = res.setdefault(rid, dict(sets=[], p=[]))
        r["sets"].append(S); r["p"].append(float(p))
    return res


def predict_K(r, tau):
    best = {}
    for S, p in zip(r["sets"], r["p"]):
        best[len(S)] = max(best.get(len(S), 0.0), p)
    for k in (0, 1, 2, 3):
        if best.get(k, 0.0) >= tau:
            return str(k)
    return "NONE"


def class_metrics(roots, sc, tau):
    yt = [roots[r]["cls"] for r in sc]; yp = [predict_K(sc[r], tau) for r in sc]
    conf = Counter(zip(yt, yp))
    return dict(tau=tau, acc=float(np.mean([a == b for a, b in zip(yt, yp)])), macro_f1=float(f1_score(yt, yp, labels=CLASSES, average="macro")),
                recall={c: conf[(c, c)] / max(1, sum(1 for y in yt if y == c)) for c in CLASSES},
                confusion={f"{a}->{b}": conf[(a, b)] for a in CLASSES for b in CLASSES if conf[(a, b)]})


def truth_sets(m):
    out = {frozenset(s) for s in m["minimal_sets"]}
    if m["minimal"]:
        out.add(frozenset(o for o, _ in m["minimal"]))
    return out


def set_metrics(roots, sc):
    n = t1 = t3 = 0
    for rid, r in sc.items():
        m = roots[rid]
        if m["cls"] not in ("1", "2", "3"):
            continue
        k, truth = int(m["cls"]), truth_sets(m)
        cand = sorted(((p, S) for S, p in zip(r["sets"], r["p"]) if len(S) == k), reverse=True)
        if truth and cand:
            n += 1; ranked = [frozenset(S) for _, S in cand]
            t1 += ranked[0] in truth; t3 += any(x in truth for x in ranked[:3])
    return dict(n=n, top1=t1 / max(1, n), top3=t3 / max(1, n))


def mask_metrics(roots, sc, delta):
    tp = fp = fn = ex = n = 0
    for rid, r in sc.items():
        m = roots[rid]
        if not (m["mask_valid"] and m["mask"]):
            continue
        base = r["p"][r["sets"].index(())]
        gain = {S[0]: p - base for S, p in zip(r["sets"], r["p"]) if len(S) == 1}
        gold = {o for o, v in m["mask"].items() if v}
        pred = {o for o, g in gain.items() if g >= delta}
        tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred); ex += pred == gold; n += 1
    return dict(n=n, f1=2 * tp / max(1, 2 * tp + fp + fn), precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn), exact=ex / max(1, n), delta=delta)


def evaluate(name, states, roots, workers):
    model = load(name)
    test = [s for s in states if s["split"] == "test"]
    S, P, PK, PL = T.score_states(model, D.StateSet(test, roots), workers=workers)
    y = np.array([s["y"] for s in test])
    res = dict(run=name, queries=dict(all=binm(y, S)))
    for kind in ("original", "D", "L"):
        i = [n for n, s in enumerate(test) if s["kind"] == kind]
        res["queries"][kind] = binm(y[i], S[i])
    i = [n for n, s in enumerate(test) if s["kind"] in ("original", "D")]
    res["queries"]["stored"] = binm(y[i], S[i])
    res["grasps"] = grasp_metrics([test[j] for j in i], P[i], PK[i], PL[i])
    rv = [r for r, m in roots.items() if m["split"] == "val"]; rt = [r for r, m in roots.items() if m["split"] == "test"]
    sc = {"val": scene_scores(model, roots, rv, workers), "test": scene_scores(model, roots, rt, workers)}
    mv = {r: roots[r] for r in sc["val"]}; mt = {r: roots[r] for r in sc["test"]}
    grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    tau = max(grid, key=lambda t: class_metrics(mv, sc["val"], t)["macro_f1"])
    res["class_tau0.5"] = class_metrics(mt, sc["test"], 0.5)
    res["class_tau_star"] = class_metrics(mt, sc["test"], tau)
    best = {r: max(v["p"]) for r, v in sc["test"].items()}
    res["solvable_vs_none_auroc"] = float(roc_auc_score([mt[r]["cls"] != "NONE" for r in sc["test"]], [best[r] for r in sc["test"]]))
    res["sets"] = set_metrics(mt, sc["test"])
    dg = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5]
    delta = max(dg, key=lambda d: mask_metrics(mv, sc["val"], d)["f1"])
    res["mask"] = mask_metrics(mt, sc["test"], delta)
    res["candidates_per_scene"] = float(np.mean([len(v["sets"]) for v in sc["test"].values()]))
    json.dump(res, open(HERE / "runs" / name / "eval.json", "w"), indent=1)
    return res


def ms(vals):
    v = np.array(vals, float)
    return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    states, roots = D.load()
    R = [evaluate(n, states, roots, a.workers) for n in a.runs]
    tag = a.tag or a.runs[0]
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(R, open(HERE / "results" / f"{tag}.json", "w"), indent=1)
    g = lambda f: ms([f(r) for r in R])  # noqa: E731
    L = [f"# SceneCollisionNet on shelf66, V4 test split: `{tag}` ({len(R)} seed{'s' if len(R) > 1 else ''}: {', '.join(a.runs)})", "",
         "## A. State feasibility: can the target reach its goal in this layout? (score = max over the 32 grasps)", "",
         "| queries | n | feasible | AUROC | AP | acc@0.5 | F1 |", "|---|---|---|---|---|---|---|"]
    for k, lab in (("stored", "original + D (labels stored in the scene files)"), ("original", "original (move T)"), ("D", "D (edited scene, obstacles parked)"),
                   ("L", "L (rest of the T-only lattice)"), ("all", "all")):
        p = R[0]["queries"][k]
        L.append(f"| {lab} | {p['n']} | {p['feasible']:.3f} | " + " | ".join((g(lambda r, m=m: r["queries"][k][m]) if m in p else "-") for m in ("auroc", "ap", "acc", "f1")) + " |")
    gr = R[0]["grasps"]
    L += ["", "## B. Per grasp (original + D test queries that carry labels)", "", "| metric | value | n |", "|---|---|---|",
          f"| grasp feasibility AUROC | {g(lambda r: r['grasps']['grasp_feas_auroc'])} | {gr.get('grasp_feas_n')} |",
          f"| pick collision AUROC (hand at the grasp pose is unusable) | {g(lambda r: r['grasps']['pick_auroc'])} | {gr.get('pick_n')} |",
          f"| place collision AUROC (hand over the goal collides) | {g(lambda r: r['grasps']['place_auroc'])} | {gr.get('place_n')} |",
          f"| the teacher's winning grasp is among the 3 best-scored | {g(lambda r: r['grasps']['winner_in_top3'])} | |",
          f"| the teacher's winning grasp is the best-scored | {g(lambda r: r['grasps']['winner_top1'])} | |", "",
          "## C. Scene level (layouts enumerated)", "", f"Candidates per scene: {R[0]['candidates_per_scene']:.0f} on average.", "",
          "| metric | tau = 0.5 | tau* (validation) |", "|---|---|---|",
          f"| class accuracy (NONE, K0-K3) | {g(lambda r: r['class_tau0.5']['acc'])} | {g(lambda r: r['class_tau_star']['acc'])} |",
          f"| class macro-F1 | {g(lambda r: r['class_tau0.5']['macro_f1'])} | {g(lambda r: r['class_tau_star']['macro_f1'])} |"]
    for c in CLASSES:
        L.append(f"| recall {'K' + c if c != 'NONE' else c} | {g(lambda r: r['class_tau0.5']['recall'][c])} | {g(lambda r: r['class_tau_star']['recall'][c])} |")
    L += [f"| tau* | | {', '.join(str(r['class_tau_star']['tau']) for r in R)} |", "",
          f"- Solvable vs NONE (best layout score): AUROC {g(lambda r: r['solvable_vs_none_auroc'])}",
          f"- Blocker set given the true K (K1-K3 scenes, n = {R[0]['sets']['n']}): top-1 {g(lambda r: r['sets']['top1'])}, top-3 {g(lambda r: r['sets']['top3'])}",
          f"- Blocker mask by counterfactual gain (delta tuned on validation: {', '.join(str(r['mask']['delta']) for r in R)}; n = {R[0]['mask']['n']} scenes): "
          f"F1 {g(lambda r: r['mask']['f1'])}, precision {g(lambda r: r['mask']['precision'])}, recall {g(lambda r: r['mask']['recall'])}, exact {g(lambda r: r['mask']['exact'])}", ""]
    (HERE / "results" / f"{tag}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
