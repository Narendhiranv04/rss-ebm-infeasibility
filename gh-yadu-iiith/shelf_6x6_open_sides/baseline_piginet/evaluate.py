#!/usr/bin/env python3
"""Evaluate trained PIGINet runs on the V4 test split of the shelf66 dataset.

    python evaluate.py full_s0 full_s1 full_s2 [--tag full]      # -> runs/<name>/eval.json, results/<tag>.{json,md}

A. PLAN FEASIBILITY (PIGINet's own task): every labelled test query (original, D edits, B sequences): AUROC, AP, accuracy,
   balanced accuracy, F1; per query family and per B role.
B. SCENE LEVEL, by using PIGINet the way the paper does: enumerate candidate task plans, score them, take the best.
   Candidates of a scene: "move T" (k = 0) and, for every set of k = 1..3 obstacles, "move each to a parking cell, then T"
   (parking cells = the scene file's first two per obstacle, every combination; the teacher uses the first).
   - direct feasibility: p(move T) against K0 vs rest.
   - class NONE / K0..K3: predicted K = smallest k whose best plan scores >= tau; none -> NONE.  tau = 0.5 (the paper's
     filter) and tau* = best macro-F1 on the validation scenes.
   - solvable vs NONE: best score over all candidates.
   - blocker set: best-scoring set of the TRUE size K against the verified minimal repair (top-1 / top-3), and blocker-mask F1 of
     the predicted set (predicted K) against labels.blocker.mask.
C. PLAN ORDERING (the paper's use in the planner: refine plans in decreasing predicted feasibility): over each scene's labelled
   plans (original + B), the number of plans refined until the first feasible one, PIGINet order vs the paper's baseline order
   (ascending plan length, ties averaged).
"""
import argparse
import itertools
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

import train as T
from piginet import PIGINet

HERE = Path(__file__).resolve().parent
CLASSES = ["0", "1", "2", "3", "NONE"]


def load(name):
    cfg = json.load(open(HERE / "runs" / name / "config.json"))
    m = PIGINet(use_image=not cfg["no_image"], use_value=not cfg["no_value"]).to(T.DEV)
    m.load_state_dict(torch.load(HERE / "runs" / name / "best.pt", map_location=T.DEV))
    m.set_text(np.load(HERE / "data" / "text.npz")["feats"])
    return m.eval(), not cfg["no_image"]


def binm(y, p, tau=0.5):
    y, p = np.asarray(y), np.asarray(p)
    out = dict(n=int(len(y)), feasible=float(y.mean()), acc=float(((p >= tau) == y).mean()))
    if 0 < y.sum() < len(y):
        out.update(auroc=float(roc_auc_score(y, p)), ap=float(average_precision_score(y, p)),
                   bal_acc=float(balanced_accuracy_score(y, p >= tau)), f1=float(f1_score(y, p >= tau)))
    return out


def candidates(m):
    """m: root meta -> list of (k, set, plan, edited cells).  Every set of k <= 3 obstacles, each to its FIRST parking cell (the
    teacher's convention for its lattice and its minimal repair)."""
    init, goal = m["init"], m["goal"]
    tm = ("t0", init["t0"], goal)
    obst = sorted(o["id"] for o in m["objects"] if o["id"] != "t0" and m["parking"].get(o["id"]))
    out = [(0, (), [tm], dict(init))]
    for k in (1, 2, 3):
        for S in itertools.combinations(obst, k):
            dst = [m["parking"][o][0] for o in S]
            cells = {**init, **dict(zip(S, dst))}
            if len(set(cells.values())) < len(cells) or goal in dst:
                continue
            out.append((k, S, [(o, init[o], d) for o, d in zip(S, dst)] + [tm], cells))
    return out


def scene_scores(model, use_image, metas, form="plan"):
    """{rid: {k: [(score, set), ...] sorted by decreasing score}}.  form "plan": the relocations are plan tokens on the original
    scene (PIGINet's interface).  form "edit": the obstacles are already parked in the initial state and the plan is "move T"
    (the teacher's T-only lattice query)."""
    res = {}
    for rid, m in metas.items():
        C = candidates(m)
        if form == "plan":
            S = [dict(rid=rid, kind="B", idx=None, objects=m["objects"], cells=m["init"], plan=pl, goal=m["goal"], y=0) for _, _, pl, _ in C]
        else:
            se = m.get("single_edit", {})
            S = [dict(rid=rid, kind="L", idx=None, objects=m["objects"], cells=cells, plan=[("t0", cells["t0"], m["goal"])], goal=m["goal"], y=0,
                      img_src={o["id"]: (f"D{se[o['id']]}" if o["id"] in st and o["id"] in se else "orig") for o in m["objects"]})
                 for _, st, _, cells in C]
        p = T.predict(model, S, use_image, bs=1024)
        per = defaultdict(list)
        for (k, st, _, _), s in zip(C, p):
            per[k].append((float(s), st))
        res[rid] = {k: sorted(v, reverse=True) for k, v in per.items()}
    return res


def predict_K(sc, tau):
    for k in (0, 1, 2, 3):
        if sc.get(k) and sc[k][0][0] >= tau:
            return str(k)
    return "NONE"


def class_metrics(metas, scores, tau):
    yt = [m["cls"] for m in metas.values()]
    yp = [predict_K(scores[r], tau) for r in metas]
    conf = Counter(zip(yt, yp))
    rec = {c: conf[(c, c)] / max(1, sum(1 for y in yt if y == c)) for c in CLASSES}
    return dict(tau=tau, acc=float(np.mean([a == b for a, b in zip(yt, yp)])), macro_f1=float(f1_score(yt, yp, labels=CLASSES, average="macro")),
                recall=rec, confusion={f"{a}->{b}": conf[(a, b)] for a in CLASSES for b in CLASSES if conf[(a, b)]})


def truth_sets(m):
    out = [frozenset(s) for s in m["minimal_sets"]]
    if m["minimal"]:
        out.append(frozenset(o for o, _ in m["minimal"]))
    return set(out)


def blocker_metrics(metas, scores, tau):
    top1 = top3 = n = 0
    tp = fp = fn = 0
    exact = nm = 0
    for rid, m in metas.items():
        if m["cls"] not in ("1", "2", "3"):
            continue
        k, truth = int(m["cls"]), truth_sets(m)
        if truth and scores[rid].get(k):
            n += 1
            ranked = [frozenset(st) for _, st in scores[rid][k]]
            top1 += ranked[0] in truth
            top3 += any(r in truth for r in ranked[:3])
        if m["mask_valid"] and m["mask"]:
            kp = predict_K(scores[rid], tau)
            pred = set(scores[rid][int(kp)][0][1]) if kp in ("1", "2", "3") else set()
            gold = {o for o, v in m["mask"].items() if v}
            tp += len(pred & gold); fp += len(pred - gold); fn += len(gold - pred)
            exact += pred == gold; nm += 1
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    return dict(n_sets=n, set_top1_given_K=top1 / max(1, n), set_top3_given_K=top3 / max(1, n),
                mask_n=nm, mask_f1=f1, mask_precision=tp / max(1, tp + fp), mask_recall=tp / max(1, tp + fn), mask_exact=exact / max(1, nm))


def ordering(samples, p):
    by = defaultdict(list)
    for s, q in zip(samples, p):
        if s["kind"] in ("original", "B"):
            by[s["rid"]].append((len(s["plan"]), float(q), s["y"]))
    pig, base, top1, n, wasted_none = [], [], 0, 0, []
    for rid, L in by.items():
        ys = [y for _, _, y in L]
        if sum(ys) == 0:
            wasted_none.append(sum(1 for _, q, _ in L if q >= 0.5))       # NONE-like: plans PIGINet would still refine
            continue
        if sum(ys) == len(ys):
            continue
        n += 1
        o = sorted(L, key=lambda t: (-t[1], t[0]))
        r = next(i for i, t in enumerate(o) if t[2]) + 1
        pig.append(r); top1 += r == 1
        # baseline: ascending length, ties in random order -> expected rank of the first feasible plan
        lens = sorted(set(l for l, _, _ in L))
        seen, exp = 0, None
        for ln in lens:
            grp = [y for l, _, y in L if l == ln]
            g, f = len(grp), sum(grp)
            if f:
                exp = seen + (g + 1) / (f + 1)
                break
            seen += g
        base.append(exp)
    return dict(scenes=n, piginet_plans_to_first_feasible=float(np.mean(pig)), length_order_plans_to_first_feasible=float(np.mean(base)),
                piginet_top1_feasible=top1 / max(1, n), labelled_plans_per_scene=float(np.mean([len(L) for L in by.values()])),
                infeasible_only_scenes=len(wasted_none), plans_above_half_on_them=float(np.mean(wasted_none)) if wasted_none else 0.0,
                plans_on_them=float(np.mean([len(L) for L in by.values() if not any(y for _, _, y in L)])) if wasted_none else 0.0)


def evaluate(name, d):
    model, use_image = load(name)
    S = {p: [s for s in d["samples"] if s["split"] == p] for p in ("val", "test")}
    M = {p: {r: m for r, m in d["roots"].items() if m["split"] == p} for p in ("val", "test")}
    pt = T.predict(model, S["test"], use_image)
    yt = np.array([s["y"] for s in S["test"]])
    res = dict(run=name, queries=dict(all=binm(yt, pt)))
    i = [n for n, s in enumerate(S["test"]) if s["kind"] in ("original", "D", "B")]
    res["queries"]["stored"] = binm(yt[i], pt[i])
    for kind in ("original", "D", "B", "L", "P"):
        i = [n for n, s in enumerate(S["test"]) if s["kind"] == kind]
        res["queries"][kind] = binm(yt[i], pt[i])
    res["plan_B_roles"] = {}
    for role in sorted({s["role"] for s in S["test"] if s["kind"] == "B"}):
        i = [n for n, s in enumerate(S["test"]) if s["kind"] == "B" and s["role"] == role]
        res["plan_B_roles"][role] = binm(yt[i], pt[i])
    for form in ("plan", "edit"):
        sc = {p: scene_scores(model, use_image, M[p], form) for p in ("val", "test")}
        grid = [round(x, 2) for x in np.arange(0.05, 0.96, 0.05)]
        tau_star = max(grid, key=lambda t: class_metrics(M["val"], sc["val"], t)["macro_f1"])
        best = {r: max(v[0][0] for v in sc["test"][r].values() if v) for r in M["test"]}
        res[form] = {"class_tau0.5": class_metrics(M["test"], sc["test"], 0.5), "class_tau_star": class_metrics(M["test"], sc["test"], tau_star),
                     "solvable_vs_none_auroc": float(roc_auc_score([m["cls"] != "NONE" for m in M["test"].values()], [best[r] for r in M["test"]])),
                     "blocker_tau0.5": blocker_metrics(M["test"], sc["test"], 0.5), "blocker_tau_star": blocker_metrics(M["test"], sc["test"], tau_star),
                     "candidates_per_scene": float(np.mean([sum(len(v) for v in sc["test"][r].values()) for r in M["test"]]))}
    res["ordering"] = ordering(S["test"], pt)
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
    d = pickle.load(open(HERE / "data" / "samples.pkl", "rb"))
    R = [evaluate(n, d) for n in a.runs]
    tag = a.tag or a.runs[0]
    (HERE / "results").mkdir(exist_ok=True)
    json.dump(R, open(HERE / "results" / f"{tag}.json", "w"), indent=1)
    g = lambda f: ms([f(r) for r in R])  # noqa: E731
    L = [f"# PIGINet on shelf66, V4 test split: `{tag}` ({len(R)} seed{'s' if len(R) > 1 else ''}: {', '.join(a.runs)})", "",
         "## A. Plan feasibility (every labelled test query)", "", "| queries | n | feasible | AUROC | AP | acc@0.5 | bal. acc | F1 |", "|---|---|---|---|---|---|---|---|"]
    for k, lab in (("stored", "original + D + B (labels stored in the scene files)"), ("original", "original (move T)"), ("D", "D (edited scene, move T)"),
                   ("B", "B (executed sequences)"), ("L", "L (rest of the T-only lattice)"), ("P", "P (lattice negatives in plan form)"), ("all", "all")):
        p = R[0]["queries"][k]
        L.append(f"| {lab} | {p['n']} | {p['feasible']:.3f} | " + " | ".join(
            (g(lambda r, m=m: r["queries"][k][m]) if m in p else "-") for m in ("auroc", "ap", "acc", "bal_acc", "f1")) + " |")
    L += ["", "B sequences by role (accuracy at 0.5):", "", "| role | n | feasible | acc | AUROC |", "|---|---|---|---|---|"]
    for role, p in R[0]["plan_B_roles"].items():
        L.append(f"| {role} | {p['n']} | {p['feasible']:.3f} | {g(lambda r: r['plan_B_roles'][role]['acc'])} | "
                 + (g(lambda r: r['plan_B_roles'][role]['auroc']) if 'auroc' in p else "-") + " |")
    L += ["", "## B. Scene level (candidate plans enumerated and scored)", "",
          f"Candidates per scene: {R[0]['plan']['candidates_per_scene']:.0f} on average (move T; every set of 1-3 obstacles to their first parking cells, then T).",
          "`plan` = the relocations are plan tokens (PIGINet's interface); `edit` = the obstacles are already parked in the initial state.", ""]
    for form in ("plan", "edit"):
        f = lambda r: r[form]  # noqa: E731
        L += [f"### form: {form}", "", "| metric | tau = 0.5 | tau* (validation) |", "|---|---|---|",
              f"| class accuracy (NONE, K0-K3) | {g(lambda r: f(r)['class_tau0.5']['acc'])} | {g(lambda r: f(r)['class_tau_star']['acc'])} |",
              f"| class macro-F1 | {g(lambda r: f(r)['class_tau0.5']['macro_f1'])} | {g(lambda r: f(r)['class_tau_star']['macro_f1'])} |"]
        for c in CLASSES:
            L.append(f"| recall {'K' + c if c != 'NONE' else c} | {g(lambda r: f(r)['class_tau0.5']['recall'][c])} | {g(lambda r: f(r)['class_tau_star']['recall'][c])} |")
        L += [f"| blocker-mask F1 | {g(lambda r: f(r)['blocker_tau0.5']['mask_f1'])} | {g(lambda r: f(r)['blocker_tau_star']['mask_f1'])} |",
              f"| blocker-mask exact | {g(lambda r: f(r)['blocker_tau0.5']['mask_exact'])} | {g(lambda r: f(r)['blocker_tau_star']['mask_exact'])} |",
              f"| tau* | | {', '.join(str(f(r)['class_tau_star']['tau']) for r in R)} |", "",
              f"- Solvable vs NONE (best candidate score): AUROC {g(lambda r: f(r)['solvable_vs_none_auroc'])}",
              f"- Blocker set given the true K (K1-K3 scenes, n = {f(R[0])['blocker_tau0.5']['n_sets']}): top-1 {g(lambda r: f(r)['blocker_tau0.5']['set_top1_given_K'])}, "
              f"top-3 {g(lambda r: f(r)['blocker_tau0.5']['set_top3_given_K'])}", ""]
    L += [f"- Direct feasibility (K0 vs rest, `original`): AUROC {g(lambda r: r['queries']['original']['auroc'])}, accuracy {g(lambda r: r['queries']['original']['acc'])}", "",
          "## C. Plan ordering (labelled plans of a scene, original + B)", "",
          f"- Scenes with both feasible and infeasible labelled plans: {R[0]['ordering']['scenes']} ({R[0]['ordering']['labelled_plans_per_scene']:.1f} labelled plans per scene)",
          f"- Plans refined until the first feasible one: PIGINet order {g(lambda r: r['ordering']['piginet_plans_to_first_feasible'])}, "
          f"length order (the paper's baseline) {g(lambda r: r['ordering']['length_order_plans_to_first_feasible'])}",
          f"- PIGINet's top plan is feasible: {g(lambda r: r['ordering']['piginet_top1_feasible'])}",
          f"- Scenes whose labelled plans are all infeasible ({R[0]['ordering']['infeasible_only_scenes']}): plans PIGINet would still refine (p >= 0.5) "
          f"{g(lambda r: r['ordering']['plans_above_half_on_them'])} of {R[0]['ordering']['plans_on_them']:.1f}", ""]
    (HERE / "results" / f"{tag}.md").write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
