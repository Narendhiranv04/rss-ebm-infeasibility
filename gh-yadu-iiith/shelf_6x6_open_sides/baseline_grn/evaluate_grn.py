#!/usr/bin/env python3
"""Evaluate trained GRN runs on the V4 test split, with the same metrics as the PIGINet baseline (branch yadu/PIGINet, evaluate.py).

    python evaluate_grn.py s0 s1 s2 [--tag grn]          ->  results/<tag>.{md,json}

A. Action feasibility: every labelled test query = "can T reach its goal in this layout?" (original scene, D edits, L lattice):
   AUROC / AP / accuracy / F1 at 0.5.  Plus the per-view heads on the queries that carry labels: view feasibility and IK
   AUROC, GO mean absolute error.
B. Scene level, by enumerating layouts (GRN judges ONE state, so a repair is a state): the original scene and every set of 1-3
   obstacles parked at their first parking cell (the teacher's convention; the same candidates as PIGINet's "edit" form).
   - class NONE / K0..K3: predicted K = smallest set size whose best layout scores >= tau, else NONE (tau 0.5, and tau* =
     best validation macro-F1);
   - solvable vs NONE AUROC (best layout score);
   - blocker set given the true K: top-1 / top-3 among the layouts of size K;
   - blocker mask from GRN's own explanation head: an obstacle is a predicted blocker when its obstruction on T's reachable
     views, max_v GO_v * IK_v, is >= tau_go (tuned on validation); F1 / exact against labels.blocker.mask (valid scenes).
"""
import argparse
import itertools
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

import build_graphs as BG
import train_grn as T
from grn_model import GRN, collate

HERE = Path(__file__).resolve().parent
CLASSES = ["0", "1", "2", "3", "NONE"]
_RECS = {}


def rec_of(rid):
    if rid not in _RECS:
        _RECS[rid] = json.loads(BG.find_root(rid).read_text())
    return _RECS[rid]


def load(name):
    m = GRN(16).to(T.DEV)
    m.load_state_dict(torch.load(HERE / "runs" / name / "best.pt", map_location=T.DEV))
    return m.eval()


@torch.no_grad()
def forward_all(model, graphs, bs=2048):
    """-> action prob [n], per-view feasibility [n,16], IK prob [n,16], list of GO arrays (edges into T) [n]"""
    A, Vw, I, Go = [], [], [], []
    for k in range(0, len(graphs), bs):
        chunk = graphs[k:k + bs]
        b = collate(chunk, T.DEV)
        F, ik, go = model(b)
        F = F[b["mask"]].sigmoid().cpu().numpy()
        A.append(F[:, 0]); Vw.append(F[:, 1:]); I.append(ik.sigmoid().cpu().numpy())
        go = go.cpu().numpy(); e = 0
        for g in chunk:
            nb = int(g["blocking"].sum()); Go.append(go[e:e + nb]); e += nb
    return np.concatenate(A), np.concatenate(Vw), np.concatenate(I), Go


def binm(y, p, tau=0.5):
    y, p = np.asarray(y), np.asarray(p)
    out = dict(n=int(len(y)), feasible=float(y.mean()), acc=float(((p >= tau) == y).mean()))
    if 0 < y.sum() < len(y):
        out.update(auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)), f1=float(f1_score(y, p >= tau)))
    return out


def scene_graphs(rid):
    """the layouts of one scene: original + every set of 1-3 obstacles parked at their first parking cell"""
    rec = rec_of(rid)
    init = {k: tuple(v) for k, v in rec["initial_cells"].items()}
    goal = tuple(rec["goal"])
    par = {o: tuple(v[0]) for o, v in (rec.get("parking") or {}).items() if v}
    obst = sorted(o["id"] for o in rec["objects"] if o["id"] != "t0" and o["id"] in par)
    out = [((), BG.make_graph(rec, init, goal, 0, None, rid, "cand", None, None))]
    for k in (1, 2, 3):
        for S in itertools.combinations(obst, k):
            cells = dict(init); cells.update({o: par[o] for o in S})
            if len(set(cells.values())) < len(cells) or goal in [par[o] for o in S]:
                continue
            out.append((S, BG.make_graph(rec, cells, goal, 0, None, rid, "cand", None, None)))
    return rec, out


def per_scene(model, rids):
    res = {}
    for rid in rids:
        rec, C = scene_graphs(rid)
        A, _, I, Go = forward_all(model, [g for _, g in C])
        sets = [S for S, _ in C]
        res[rid] = dict(sets=sets, p=A, ik0=I[0], go0=Go[0], names0=[n for n in C[0][1]["names"]], src0=C[0][1]["src"][C[0][1]["blocking"]], graph0=C[0][1])
    return res


def predict_K(r, tau):
    best = {}
    for S, p in zip(r["sets"], r["p"]):
        best[len(S)] = max(best.get(len(S), 0.0), float(p))
    for k in (0, 1, 2, 3):
        if best.get(k, 0.0) >= tau:
            return str(k)
    return "NONE"


def class_metrics(metas, sc, tau):
    yt = [metas[r]["cls"] for r in sc]; yp = [predict_K(sc[r], tau) for r in sc]
    conf = Counter(zip(yt, yp))
    return dict(tau=tau, acc=float(np.mean([a == b for a, b in zip(yt, yp)])), macro_f1=float(f1_score(yt, yp, labels=CLASSES, average="macro")),
                recall={c: conf[(c, c)] / max(1, sum(1 for y in yt if y == c)) for c in CLASSES},
                confusion={f"{a}->{b}": conf[(a, b)] for a in CLASSES for b in CLASSES if conf[(a, b)]})


def truth_sets(m):
    out = {frozenset(s) for s in m["minimal_sets"]}
    if m["minimal"]:
        out.add(frozenset(o for o, _ in m["minimal"]))
    return out


def set_metrics(metas, sc):
    n = t1 = t3 = 0
    for rid, r in sc.items():
        m = metas[rid]
        if m["cls"] not in ("1", "2", "3"):
            continue
        k, truth = int(m["cls"]), truth_sets(m)
        cand = sorted(((float(p), S) for S, p in zip(r["sets"], r["p"]) if len(S) == k), reverse=True)
        if truth and cand:
            n += 1; ranked = [frozenset(S) for _, S in cand]
            t1 += ranked[0] in truth; t3 += any(x in truth for x in ranked[:3])
    return dict(n=n, top1=t1 / max(1, n), top3=t3 / max(1, n))


def blocker_scores(r):
    """obstacle -> max over views of GO x IK on the original scene's edges into T"""
    names, src = r["names0"], r["src0"]
    s = (r["go0"] * r["ik0"][None, :]).max(1)
    return {names[j]: float(v) for j, v in zip(src, s) if names[j] not in ("base", "goal")}


def mask_metrics(metas, sc, tau_go):
    tp = fp = fn = ex = n = 0
    for rid, r in sc.items():
        m = metas[rid]
        if not (m["mask_valid"] and m["mask"]):
            continue
        gold = {o for o, v in m["mask"].items() if v}
        pred = {o for o, v in blocker_scores(r).items() if v >= tau_go}
        tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred); ex += pred == gold; n += 1
    return dict(n=n, f1=2 * tp / max(1, 2 * tp + fp + fn), precision=tp / max(1, tp + fp), recall=tp / max(1, tp + fn), exact=ex / max(1, n), tau_go=tau_go)


def view_metrics(model, graphs):
    A, Vw, I, Go = forward_all(model, graphs)
    out = {}
    Fm = np.stack([g["Fm"][1:] for g in graphs]) > 0; Fy = np.stack([g["F"][1:] for g in graphs])
    if Fm.any() and 0 < Fy[Fm].sum() < Fm.sum():
        out["view_F_auroc"] = float(roc_auc_score(Fy[Fm], Vw[Fm])); out["view_F_n"] = int(Fm.sum())
    Im = np.stack([g["IKm"] for g in graphs]) > 0; Iy = np.stack([g["IK"] for g in graphs])
    if Im.any() and 0 < Iy[Im].sum() < Im.sum():
        out["ik_auroc"] = float(roc_auc_score(Iy[Im], I[Im])); out["ik_n"] = int(Im.sum()); out["ik_acc"] = float(((I[Im] >= .5) == Iy[Im]).mean())
    err = []
    for g, go in zip(graphs, Go):
        if g["gom"].any():
            b = g["blocking"]; views = (g["IK"] > 0.5)[None, :] & (g["IKm"] > 0)[None, :]
            if views.any():
                err.append(np.abs(go - g["go"][b])[np.broadcast_to(views, go.shape)].mean())
    if err:
        out["go_mae"] = float(np.mean(err)); out["go_n_graphs"] = len(err)
    return out


def evaluate(name, d):
    model = load(name)
    G = d["graphs"]
    test = [g for g in G if g["split"] == "test"]
    A, *_ = forward_all(model, test)
    y = np.array([g["y"] for g in test])
    res = dict(run=name, queries=dict(all=binm(y, A)))
    for kind in ("original", "D", "L"):
        i = [n for n, g in enumerate(test) if g["kind"] == kind]
        res["queries"][kind] = binm(y[i], A[i])
    i = [n for n, g in enumerate(test) if g["kind"] in ("original", "D")]
    res["queries"]["stored"] = binm(y[i], A[i])
    res["views"] = view_metrics(model, [g for g in test if g["kind"] in ("original", "D")])
    metas = d["roots"]
    sc = {p: per_scene(model, [r for r, m in metas.items() if m["split"] == p]) for p in ("val", "test")}
    mv = {r: metas[r] for r in sc["val"]}; mt = {r: metas[r] for r in sc["test"]}
    grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
    tau = max(grid, key=lambda t: class_metrics(mv, sc["val"], t)["macro_f1"])
    res["class_tau0.5"] = class_metrics(mt, sc["test"], 0.5)
    res["class_tau_star"] = class_metrics(mt, sc["test"], tau)
    best = {r: float(max(s["p"])) for r, s in sc["test"].items()}
    res["solvable_vs_none_auroc"] = float(roc_auc_score([mt[r]["cls"] != "NONE" for r in sc["test"]], [best[r] for r in sc["test"]]))
    res["sets"] = set_metrics(mt, sc["test"])
    tg = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    tau_go = max(tg, key=lambda t: mask_metrics(mv, sc["val"], t)["f1"])
    res["mask"] = mask_metrics(mt, sc["test"], tau_go)
    res["mask_tau_go0.5"] = mask_metrics(mt, sc["test"], 0.5)
    res["candidates_per_scene"] = float(np.mean([len(s["sets"]) for s in sc["test"].values()]))
    json.dump(res, open(HERE / "runs" / name / "eval.json", "w"), indent=1)
    return res


def ms(vals):
    v = np.array(vals, float)
    return f"{v.mean():.3f} ± {v.std():.3f}" if len(v) > 1 else f"{v.mean():.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()
    torch.set_num_threads(8)
    d = dict(graphs=pickle.load(open(HERE / "data" / "graphs.pkl", "rb")), roots=pickle.load(open(HERE / "data" / "roots.pkl", "rb")))
    R = [evaluate(n, d) for n in a.runs]
    tag = a.tag or a.runs[0]
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(R, open(HERE / "results" / f"{tag}.json", "w"), indent=1)
    g = lambda f: ms([f(r) for r in R])  # noqa: E731
    L = [f"# GRN on shelf66, V4 test split: `{tag}` ({len(R)} seed{'s' if len(R) > 1 else ''}: {', '.join(a.runs)})", "",
         "## A. Action feasibility: can the target reach its goal in this layout?", "", "| queries | n | feasible | AUROC | AP | acc@0.5 | F1 |", "|---|---|---|---|---|---|---|"]
    for k, lab in (("stored", "original + D (labels stored in the scene files)"), ("original", "original (move T)"), ("D", "D (edited scene, obstacles parked)"),
                   ("L", "L (rest of the T-only lattice)"), ("all", "all")):
        p = R[0]["queries"][k]
        L.append(f"| {lab} | {p['n']} | {p['feasible']:.3f} | " + " | ".join((g(lambda r, m=m: r["queries"][k][m]) if m in p else "-") for m in ("auroc", "ap", "acc", "f1")) + " |")
    v = R[0]["views"]
    L += ["", "Per-view heads (original + D test queries that carry labels):", "",
          f"- view feasibility AUROC {g(lambda r: r['views']['view_F_auroc'])} (n = {v.get('view_F_n')} labelled views)",
          f"- IK reachability AUROC {g(lambda r: r['views']['ik_auroc'])}, accuracy {g(lambda r: r['views']['ik_acc'])} (n = {v.get('ik_n')})",
          f"- obstruction (GO) mean absolute error on reachable views {g(lambda r: r['views']['go_mae'])} ({v.get('go_n_graphs')} graphs)", "",
          "## B. Scene level (layouts enumerated: original + every set of 1-3 obstacles parked at their first parking cell)", "",
          f"Candidates per scene: {R[0]['candidates_per_scene']:.0f} on average.", "",
          "| metric | tau = 0.5 | tau* (validation) |", "|---|---|---|",
          f"| class accuracy (NONE, K0-K3) | {g(lambda r: r['class_tau0.5']['acc'])} | {g(lambda r: r['class_tau_star']['acc'])} |",
          f"| class macro-F1 | {g(lambda r: r['class_tau0.5']['macro_f1'])} | {g(lambda r: r['class_tau_star']['macro_f1'])} |"]
    for c in CLASSES:
        L.append(f"| recall {'K' + c if c != 'NONE' else c} | {g(lambda r: r['class_tau0.5']['recall'][c])} | {g(lambda r: r['class_tau_star']['recall'][c])} |")
    L += [f"| tau* | | {', '.join(str(r['class_tau_star']['tau']) for r in R)} |", "",
          f"- Solvable vs NONE (best layout score): AUROC {g(lambda r: r['solvable_vs_none_auroc'])}",
          f"- Blocker set given the true K (K1-K3 scenes, n = {R[0]['sets']['n']}): top-1 {g(lambda r: r['sets']['top1'])}, top-3 {g(lambda r: r['sets']['top3'])}",
          f"- Blocker mask from GRN's obstruction head (tau_go tuned on validation: {', '.join(str(r['mask']['tau_go']) for r in R)}; n = {R[0]['mask']['n']} scenes): "
          f"F1 {g(lambda r: r['mask']['f1'])}, precision {g(lambda r: r['mask']['precision'])}, recall {g(lambda r: r['mask']['recall'])}, exact {g(lambda r: r['mask']['exact'])}",
          f"- Same at tau_go = 0.5: F1 {g(lambda r: r['mask_tau_go0.5']['f1'])}", ""]
    (HERE / "results" / f"{tag}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
