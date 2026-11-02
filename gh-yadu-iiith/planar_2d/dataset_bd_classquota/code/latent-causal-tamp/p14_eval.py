"""P14-A1: natural-mix diagnosis and matched-admission comparisons on FROZEN P13 predictions (exploratory, post-hoc; P13 blind roots stay
test data: no calibrator / threshold is fitted on them). Matched thresholds are derived from validation-only re-inference with the frozen
P13 checkpoints. Later P14-D3 comparison tables and figures are added here."""
from __future__ import annotations

import collections
import math

import numpy as np

import config as cf
from p14_audit import OUT, P13, paired, rd

ARMS = ("K-B", "K-C", "A", "SF", "B-F2", "B-F3pm", "B-F4", "B-F5spm", "B-F5qpm", "B-U", "B-V0", "B-F1off")
GRID = np.round(np.linspace(0.05, 0.99, 95), 3)


def _wilson(k, n):
    if not n:
        return [None, None]
    z, p = 1.96, k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def validation_rows() -> None:
    """Validation-only rows (u, m, J_pred, C_d, energy, y) per frozen P13 checkpoint (same scorer as the P13 freeze)."""
    import contextlib
    import copy
    import os
    import torch
    os.environ["P10C_OUT"] = str(P13 / "models")
    import p13_pipeline as pp
    import p9_l3 as l3
    import unique_factor_keyed as ufk
    pc = pp._arms()
    val = [r for r in pp._prepared("trainval") if r["split"] == "validation"]
    out = {}
    for f in sorted((P13 / "models" / "runs").glob("P13-*.json")):
        x = rd(f)
        if x["arm"] not in ARMS:
            continue
        m, keyed = pc.load(x["name"])[0], pc.ARMS[x["arm"]][2]
        rs = pc._v0(copy.deepcopy(val)) if x["arm"] == "B-V0" else val
        with (ufk.installed() if keyed else contextlib.nullcontext()), torch.no_grad():
            ev = [l3.root_eval9(m, r, "factorized" if x["arm"] in ("A", "SF") else "unique")["root"] for r in rs]
        out[x["name"]] = {r["key"]: [{k: c[k] for k in ("query", "u", "m", "J_pred", "C_d", "energy", "y")} for c in e["cands"]] for r, e in zip(rs, ev)}
    (OUT / "calibration").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "calibration" / "VALIDATION_ROWS.json", out)


def _choose(rows, tau, thr, key=("m", "J_pred", "C_d", "energy", "query")):
    adm = [r for r in rows if 1 / (1 + math.exp(-r["u"] / tau)) >= thr]
    return min(adm, key=lambda r: tuple(r[k] for k in key))["query"] if adm else "ABSTAIN"


def _val_curves(vrows, tau):
    """Validation coverage and false admission (chosen candidate labelled infeasible) per threshold."""
    cov, fa = {}, {}
    for t in GRID:
        ch = {k: _choose(rs, tau, t) for k, rs in vrows.items()}
        lab = {k: next((r["y"] for r in vrows[k] if r["query"] == q), None) for k, q in ch.items()}
        cov[float(t)] = float(np.mean([q != "ABSTAIN" for q in ch.values()]))
        fa[float(t)] = float(np.mean([lab[k] == 0 for k, q in ch.items()]))
    return cov, fa


def matched_admission() -> dict:
    """Per arm and seed: (i) historical threshold; (ii) validation-coverage-matched threshold (coverage of K-B at its own validation
    threshold, per seed); (iii) validation false-admission-constrained threshold (max coverage with false admission <= that of K-B). Test
    choices are scored by teacher label (proof-grounded) and by EXACT executed episode identity (root, repair) where it exists; otherwise
    UNKNOWN (never imputed). Also coverage-risk curves from frozen test scores (labels used only to score, not to choose)."""
    import p13_factors as f13
    snap, P, y, st, cost = f13._p13_blind()
    A = f13._by_arm(snap)
    vr = rd(OUT / "calibration" / "VALIDATION_ROWS.json")
    man = rd(P13 / "p13_5" / "EXEC_JOBS.json")
    led = {}
    for f in sorted((P13 / "p13_5" / "episodes").glob("w*/ledger.json")):
        led |= rd(f)["episodes"]
    ep = {f"{j['root']}::{j['choice']}": led.get(f"{j['arm']}__{j['root']}", {}).get("outcome") for j in man["jobs"]}
    if (OUT / "calibration" / "CF_JOBS.json").exists():                 # P14 counterfactual episodes: exact (root, repair) identity, same protocol
        cl = {}
        for f in sorted((OUT / "calibration" / "episodes").glob("w*/ledger.json")):
            cl |= rd(f)["episodes"]
        for j in rd(OUT / "calibration" / "CF_JOBS.json")["jobs"]:
            o = cl.get(f"{j['arm']}__{j['root']}", {}).get("outcome")
            if o and o != "NOT_RUN_BUDGET":
                ep[f"{j['root']}::{j['choice']}"] = o
    exe = set(man["permutation"])
    target = {s: None for s in (0, 1, 2)}
    res, curves = collections.defaultdict(dict), {}
    for arm in ARMS:
        for s, a in sorted(A[arm].items()):
            cov, fa = _val_curves(vr[f"P13-{arm}-s{s}"], a["tau"])
            if arm == "K-B":
                target[s] = {"coverage": cov[min(cov, key=lambda t: abs(t - a["threshold"]))], "false_admission": fa[min(fa, key=lambda t: abs(t - a["threshold"]))]}
            thr = {"historical": a["threshold"],
                   "coverage_matched": min(cov, key=lambda t: (abs(cov[t] - target[s]["coverage"]), t)),
                   "false_admission_matched": max([t for t in fa if fa[t] <= target[s]["false_admission"] + 1e-12] or [max(GRID)], key=lambda t: (cov[t], -t))}
            for name, t in thr.items():
                ch = {k: _choose(a["roots"][k]["rows"], a["tau"], t) for k in exe}
                out = {k: ("ABSTAIN" if q == "ABSTAIN" else ep.get(f"{k}::{q}") or "UNKNOWN_NOT_EXECUTED") for k, q in ch.items()}
                res[(arm, name)][s] = {"threshold": t, "val_coverage": cov[min(cov, key=lambda u: abs(u - t))], "choices": ch, "outcomes": out}
            if s == 0:
                curves[arm] = [{"t": float(t), "coverage": float(np.mean([_choose(a["roots"][k]["rows"], a["tau"], t) != "ABSTAIN" for k in exe])),
                                "label_correct_given_admitted": float(np.mean([y.get(q) == 1 for k in exe for q in [_choose(a["roots"][k]["rows"], a["tau"], t)] if q != "ABSTAIN"] or [np.nan]))}
                               for t in GRID[::5]]
    summ = {}
    for (arm, name), d in res.items():
        ks = sorted(exe)
        succ = {k: float(np.mean([d[s]["outcomes"][k] == "SUCCESS" for s in d])) for k in ks}
        unk = sum(d[s]["outcomes"][k] == "UNKNOWN_NOT_EXECUTED" for s in d for k in ks)
        lab = {k: float(np.mean([y.get(d[s]["choices"][k]) == 1 for s in d])) for k in ks}
        summ[f"{arm}|{name}"] = {"thresholds": [d[s]["threshold"] for s in sorted(d)], "val_coverage": [d[s]["val_coverage"] for s in sorted(d)],
                                 "test_admission": float(np.mean([d[s]["choices"][k] != "ABSTAIN" for s in d for k in ks])),
                                 "executed_success_lower_bound": float(np.mean(list(succ.values()))), "unknown_not_executed_choices": int(unk),
                                 "label_correct_root_rate": float(np.mean(list(lab.values()))), "_succ": succ, "_lab": lab}
    con = {}
    for name in ("historical", "coverage_matched", "false_admission_matched"):
        for arm in ARMS[1:]:
            a, b = summ[f"K-B|{name}"], summ[f"{arm}|{name}"]
            con[f"{name}|K-B-{arm}"] = {"label_correct": paired(a["_lab"], b["_lab"]), "executed_success_lower_bound": paired(a["_succ"], b["_succ"]),
                                         "unknown_choices_other": b["unknown_not_executed_choices"]}
    for v in summ.values():
        v.pop("_succ"), v.pop("_lab")
    out = {"kb_validation_targets": target, "summary": summ, "contrasts": con, "coverage_risk_test_s0": curves,
           "note": "EXPLORATORY post-hoc; thresholds from validation only; executed success counts only exact (root, repair) episodes, UNKNOWN otherwise"}
    cf.write_json(OUT / "calibration" / "MATCHED_ADMISSION.json", out)
    return out


def cf_jobs() -> None:
    """Frozen list of matched-threshold choices of the core arms never executed in P13 (identical strict TOP1 protocol and episode
    identity 'c' + stable_hash(root::choice)); deterministic sorted order, dealt round-robin to 4 workers."""
    import p13_factors as f13
    snap, P, y, st, cost = f13._p13_blind()
    A = f13._by_arm(snap)
    d = rd(OUT / "calibration" / "MATCHED_ADMISSION.json")
    man = rd(P13 / "p13_5" / "EXEC_JOBS.json")
    done = {f"{j['root']}::{j['choice']}" for j in man["jobs"]}
    plan = {j["root"]: j["plan"] for j in man["jobs"]}
    need = set()
    for name in ("coverage_matched", "false_admission_matched"):
        for arm in ("K-B", "K-C", "A", "SF", "B-F2", "B-F3pm", "B-F4", "B-F5spm"):
            for s, t in zip(sorted(A[arm]), d["summary"][f"{arm}|{name}"]["thresholds"]):
                for k in man["permutation"]:
                    q = _choose(A[arm][s]["roots"][k]["rows"], A[arm][s]["tau"], t)
                    if q != "ABSTAIN" and f"{k}::{q}" not in done:
                        need.add((k, q))
    seqs = {c["query"]: c["sequence"] for k in P for c in P[k]["items"] if c["kind"] == "ORDERED_TEACHER"}
    jobs = [{"arm": "c" + cf.stable_hash(f"{k}::{q}")[7:19], "root": k, "choice": q, "ranked": [seqs[q]], "plan": plan[k], "worker": i % 4}
            for i, (k, q) in enumerate(sorted(need))]
    cf.write_json(OUT / "calibration" / "CF_JOBS.json", {"jobs": jobs, "hash": cf.stable_hash(jobs)})


def cf_worker(w, cap) -> None:
    import p7_oneshot as po
    po.CAPS.update({"cpu_s": cap, "wall_s": 40000.0, "episode_cpu_s": 180.0, "episode_wall_s": 600.0, "max_episodes": 10 ** 6, "max_rank": 5})
    for j in (j for j in rd(OUT / "calibration" / "CF_JOBS.json")["jobs"] if j["worker"] == w):
        po.run_episodes(OUT / "calibration" / "episodes" / f"w{w}", j["plan"], [j], lambda m: open(OUT / "calibration" / f"cf_w{w}.log", "a").write(m + "\n"))


def natural_diag() -> dict:
    """Synthetic unconditioned (natural) stream and the challenge set from frozen P13 predictions: original-feasibility discrimination,
    feasible / infeasible recall at 0.5, Brier, NLL, 5-bin reliability with Wilson CIs, and joint four-bit NLL by outcome pattern.
    Admission on feasible originals = does the TOP1 rule pick the empty repair (correct) or a relocation / abstain."""
    import p10b_dataset as pd
    import p13_factors as f13
    import p13_pipeline as pp
    snap, P, y, st, cost = f13._p13_blind()
    A = f13._by_arm(snap)
    rec = {}
    for d in sorted(pp.HOLD.glob("*_b*")):
        rec |= pp._sealed(d)
    dl = pd.d_labels({"roots": {k: r for k, r in P.items()}}, rec)
    tg = rd(P13 / "p13_4" / "TARGETED.json")["per_root"]["four"]
    out = {}
    for stream in ("natural", "blind"):
        ks = [k for k in P if P[k]["stream"] == stream and dl[k]["y0"] is not None]
        y0 = {k: dl[k]["y0"] for k in ks}
        res = {"N": len(ks), "feasible": int(sum(y0.values()))}
        prob = {}
        for arm in ARMS:
            prob[arm] = {k: float(np.mean([A[arm][s]["roots"][k]["marginals"][f"seq:{k}:empty"] for s in A[arm]])) for k in ks}
        for arm in ("D", "D-pairless"):
            lam = snap["baselines"][f"{arm}_lambda_B"]
            ds = [x for x in snap["d"].values() if x["arm"] == arm and x["lambda_B"] == lam]
            prob[arm] = {k: float(np.mean([x["roots"][k]["p_feasible"] for x in ds])) for k in ks}
        for arm, pk in prob.items():
            ps, ys = np.array([pk[k] for k in ks]), np.array([y0[k] for k in ks])
            pos, neg = ps[ys == 1], ps[ys == 0]
            auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else None
            bins = []
            for lo in np.linspace(0, 0.8, 5):
                m = (ps >= lo) & (ps < lo + 0.2 + (1e-9 if lo >= 0.8 else 0))
                bins.append({"bin": [round(float(lo), 1), round(float(lo + 0.2), 1)], "n": int(m.sum()), "mean_p": float(ps[m].mean()) if m.any() else None,
                             "observed": float(ys[m].mean()) if m.any() else None, "wilson95": _wilson(int(ys[m].sum()), int(m.sum()))})
            empty = None
            if arm in ARMS and pos.size:
                empty = float(np.mean([a["roots"][k]["top1"] == f"seq:{k}:empty" for s, a in A[arm].items() for k in ks if y0[k] == 1]))
            res[arm] = {"auroc": auc, "feasible_recall": float((pos >= 0.5).mean()) if pos.size else None, "infeasible_recall": float((neg < 0.5).mean()) if neg.size else None,
                        "balanced_accuracy": float(((pos >= 0.5).mean() + (neg < 0.5).mean()) / 2) if pos.size and neg.size else None,
                        "brier": float(np.mean((ps - ys) ** 2)), "nll": float(np.mean(-np.log(np.clip(np.where(ys == 1, ps, 1 - ps), 1e-12, 1)))),
                        "mean_p_feasible_on_feasible": float(pos.mean()) if pos.size else None, "reliability": bins,
                        "TOP1_empty_on_feasible_originals": empty}
        pat = collections.defaultdict(lambda: collections.defaultdict(list))
        for k, f in tg.items():
            if P.get(k, {}).get("stream") != stream:
                continue
            for arm in ARMS:
                qs = [t["query"] for t in P[k]["items"] if t["token"] in ("", "edit:A", "edit:B", "edit:A+B")]
                order = {t["token"]: t["query"] for t in P[k]["items"]}
                qs = [order[x] for x in ("", "edit:A", "edit:B", "edit:A+B")]
                pat[arm][f["pattern"]].append(float(np.mean([f13._nll_joint(A[arm][s]["roots"][k], qs, f["y"]) for s in A[arm]])))
        res["joint4_nll_by_pattern"] = {arm: {p: {"n": len(v), "mean": float(np.mean(v))} for p, v in d.items()} for arm, d in pat.items()}
        out[stream] = res
    cf.write_json(OUT / "calibration" / "NATURAL_DIAG.json", out)
    return out


def cf_continue(extra=300.0) -> None:
    """Never-attempted NOT_RUN_BUDGET placeholders are logged and removed; the same frozen order resumes (per worker)."""
    import json as _j
    for f in sorted((OUT / "calibration" / "episodes").glob("w*/ledger.json")):
        led = rd(f)
        rm = [k for k, v in led["episodes"].items() if v.get("outcome") == "NOT_RUN_BUDGET"]
        if rm:
            led["episodes"] = {k: v for k, v in led["episodes"].items() if k not in rm}
            f.write_text(_j.dumps(led, indent=1, default=float))
            cf.write_json(OUT / "calibration" / f"cf_removed_{f.parent.name}.json", {"removed": rm})
            cf_worker(int(f.parent.name[1:]), led["cpu_s"] + extra)


# ------------------------------------------------------------------ P14-D2 / D3: method runs (identical scenes, slots, index, planner, budget)
METHODS = ("D", "D-pairless", "geometry", "uninformed", "oracle")
EVAL_BUDGET = {"motion_checks": 120, "online_moves": 6, "target_checks": 16}


def _roots(split):
    import os
    os.environ["P10C_OUT"] = str(P13 / "models")
    import p13_pipeline as pp
    import p10b_dataset as pd
    if split == "dev":
        rs = {r["key"]: r for r in pp._prepared("trainval") if str(r["audit"].get("source", "")).startswith("P13")}
        dirs = sorted((pp.OUT / "data").glob("*_b*"))
    else:
        rs = {r["key"]: r for r in pp._prepared("test")}
        dirs = sorted(pp.HOLD.glob("*_b*"))
    P, D = {}, {}
    for d in dirs:
        plan = pp.rd(d / "plan.json")
        P |= plan["roots"]
        D |= pd.d_labels(plan, pp._sealed(d))
    return rs, P, D


def run_method(method, split, roots, out_name, lookup="combined", modal_only=False, online=True, budget=None) -> None:
    """One process per (method, variant): runs the search on every listed root, replays each found plan, writes one JSON."""
    import p14_d_search as ds
    rs, P, D = _roots(split)
    ms = ds.d_models(method == "D-pairless") if method in ("D", "D-pairless") else []
    res = {}
    for k in roots:
        cx = ds.Ctx(lookup=lookup, modal_only=modal_only, online=online, budget=budget or EVAL_BUDGET)
        ids = rs[k]["ids"]
        dl = D[k]
        orc = {ids.index(o): m for o, m in zip(dl["pool"], dl["mask"])} if dl.get("mask") else ({ids.index(o): 0 for o in dl["pool"]} if dl.get("y0") == 1 else None)
        g = ds.Guide("geometry" if method == "lookup-only" else method, rs[k]["R"], cx.robot, ms, orc)
        out = ds.search(cx, P[k], rs[k], g, key=f"p14/{split}/{method}/{k}")
        if out["found"]:
            out["replay"] = ds.replay(cx, P[k], out["plan"])
            moved = [s["object"] for s in out["plan"]["steps"]]
            union = {o for o, m in zip(dl["pool"], dl["mask"] or [0] * len(dl["pool"])) if m} if dl.get("mask") or dl.get("y0") == 1 else None
            out["unnecessary_moves"] = None if union is None else sum(o not in union for o in moved)
            out["blockers_displaced"] = None if union is None else sum(o in union for o in moved)
            out["plan"] = {k2: v for k2, v in out["plan"].items() if k2 != "records"} | {"n_records": len(out["plan"]["records"])}
        out |= {"family": P[k]["family"], "y0": dl.get("y0"), "union_mask": dl.get("mask"), "pool": dl.get("pool")}
        res[k] = out
        (OUT / "search").mkdir(parents=True, exist_ok=True)
        cf.write_json(OUT / "search" / f"{out_name}.json", res)              # after every root: a capped job keeps what it finished


# ------------------------------------------------------------------ figures (recorded / certified motions only)
def _merge(tr, a):
    return {"q": np.concatenate([tr["q"][:1], a["q"]]), "gap": np.concatenate([tr["gap"][:1], a["gap"]]),
            "phase": np.concatenate([np.array(["pregrasp"]), a["phase"]]), "attached": np.concatenate([np.array([False]), a["attached"]]),
            "held_local": a["held_local"], "place_xy": a["place_xy"]}


def case_figure(split="dev", key=None, method="D", budget=None) -> None:
    """One search case: initial scene with D blocker scores, retrieved swept modes for the first move, executed certified trajectories,
    final scene and cost decomposition; plus a GIF sampled ONLY from the recorded / certified arrays of the found plan."""
    import torch
    import p6_scenes as p6
    import p6_viz as pv
    import p14_d_search as ds
    from scene import scene_from_dict
    from viz import COLORS, plt
    rs, P, D = _roots(split)
    cx = ds.Ctx(budget=budget or EVAL_BUDGET)
    rr, root = rs[key], P[key]
    g = ds.Guide(method, rr["R"], cx.robot, ds.d_models(method == "D-pairless") if method in ("D", "D-pairless") else [])
    out = ds.search(cx, root, rr, g)
    if not out["found"]:
        raise SystemExit("no plan found for the case figure")
    s0, goal, ids = scene_from_dict(root["scene"]), root["goal_cell"], rr["ids"]
    movers = sorted({o for it in root["items"] if it["kind"] == "ORDERED_TEACHER" for o, _ in it["sequence"]})
    xy = torch.as_tensor(np.array([s0.get(o).xy_m for o in ids]), dtype=torch.float64)
    sc = g.score(cx, rr["R"].tid, xy, np.asarray(s0.robot.q_rad, float), [ids.index(o) for o in movers])
    plan, recs = out["plan"], out["plan"]["records"]
    rel, i = [], 0
    for st in plan["steps"]:
        rel.append(_merge(recs[i], recs[i + 1]) if st["via"] == "lookup" else recs[i])
        i += 2 if st["via"] == "lookup" else 1
    f, ax = plt.subplots(1, 4, figsize=(24, 6))
    pv._panel(ax[0], cx.cfg, cx.robot, s0, goal, f"{key[-3:]} {root['family']}: initial (D blocker score)")
    for o, v in zip(movers, sc):
        x, y = s0.get(o).xy_m
        ax[0].text(x, y + 0.035, f"{o}:{v:.2f}", ha="center", fontsize=8, color="#c0392b", weight="bold")
    st0 = plan["steps"][0]
    lk = ds.sx.lookup_candidates(cx.idx, st0["object"], s0.get(st0["object"]).cell, st0["dest"], s0, goal, s0.get(st0["object"]).radius_m)
    pv._panel(ax[1], cx.cfg, cx.robot, s0, goal, f"lookup {st0['object']}->{st0['dest']}: {lk.get('modes', 0)} modes (swept cells)")
    for mi, c in enumerate([c for c in lk.get("candidates", []) if c["rank_in_mode"] == 0][:3]):
        for cell in c["entry"]["_all"]:
            ax[1].add_patch(plt.Rectangle(((cell % 8) * 0.08, (cell // 8) * 0.08), 0.08, 0.08, fc=["#2a6fdb", "#e07b00", "#2e9e5b"][mi], alpha=0.18, zorder=1))
        ax[1].text(0.01, 0.62 - 0.04 * mi, f"mode {mi}: support {c['support']} roots, travel {c['travel_J']:.2f}", fontsize=7, color=["#2a6fdb", "#e07b00", "#2e9e5b"][mi])
    sN = p6.moved(s0, {st["object"]: st["dest"] for st in plan["steps"]})
    pv._panel(ax[2], cx.cfg, cx.robot, sN, goal, "after relocations: executed tool paths (certified)")
    import geometry as ge
    for a, col in zip(rel + recs[-1:], ["#2a6fdb", "#e07b00", "#8e5bd0", "#c0392b"]):
        pts = np.array([ge.fk(cx.robot, q)[0][-1] for q, _, _ in __import__("motion").samples(a, 60)])
        ax[2].plot(pts[:, 0], pts[:, 1], color=col, lw=1.2)
    tv = [ds.travel(cx.robot, [a]) for a in rel] + [ds.travel(cx.robot, recs[-1:])]
    ax[3].bar(range(len(tv)), tv, color=["#2a6fdb"] * len(rel) + ["#c0392b"])
    ax[3].set(xticks=range(len(tv)), xticklabels=[f"{st['object']}->{st['dest']}\n({st['via']})" for st in plan["steps"]] + ["target"],
              ylabel="measured joint travel (normalized)", title=f"lexicographic cost: {plan['moves']} moves, travel {plan['travel']:.2f}")
    DSTV = __import__("pathlib").Path("visualizations/p14")
    DSTV.mkdir(parents=True, exist_ok=True)
    pv.png(f, DSTV / f"P14_search_case_{key[-3:]}.png", f"P14-D2 search case {key} ({method}-guided): scores, retrieved modes, executed plan, cost")
    frames = []
    for qq, gp, pos, mv in pv.trace_frames(cx.robot, s0, [(st["object"], st["dest"]) for st in plan["steps"]], rel, recs[-1:], n_move=10, n_target=12):
        fg, a = plt.subplots(figsize=(5, 5))
        pv._panel(a, cx.cfg, cx.robot, s0, goal, f"{key[-3:]}: certified plan ({plan['moves']} moves)\nRECORDED / CERTIFIED arrays", q=qq, gap=gp, pos=pos, held=mv)
        pv._tag(a, "replayed certified motion", COLORS["valid"])
        frames.append(pv._rgb(fg))
    pv.gif(DSTV / f"P14_search_case_{key[-3:]}.gif", frames, f"P14-D2 {key}: certified plan replay")


def d3_summary() -> dict:
    """D3 tables from the EVAL_* / ABL_* run files (identical roots, slots, index, planner, budget)."""
    S = OUT / "search"
    runs = {m: rd(S / f"EVAL_{m}.json") for m in METHODS if (S / f"EVAL_{m}.json").exists()}
    common = sorted(set.intersection(*[set(v) for v in runs.values()]))
    ok = lambda v: bool(v.get("found") and (v.get("replay") or {}).get("success"))  # noqa: E731

    def summ(d, ks):
        st = [d[k]["stats"] for k in ks]
        steps = [s for k in ks if ok(d[k]) for s in d[k]["plan"]["steps"]]
        return {"roots": len(ks), "success": float(np.mean([ok(d[k]) for k in ks])),
                "feasible_original_zero_move_success": float(np.mean([ok(d[k]) and d[k]["plan"]["moves"] == 0 for k in ks if d[k]["y0"] == 1] or [np.nan])),
                "mean_moves_solved": float(np.mean([d[k]["plan"]["moves"] for k in ks if ok(d[k])] or [np.nan])),
                "mean_travel_solved": float(np.mean([d[k]["plan"]["travel"] for k in ks if ok(d[k])] or [np.nan])),
                "unnecessary_moves_total": int(sum(d[k].get("unnecessary_moves") or 0 for k in ks if ok(d[k]))),
                "blockers_displaced_total": int(sum(d[k].get("blockers_displaced") or 0 for k in ks if ok(d[k]))),
                "depth_exhaustive_solved": int(sum(bool(d[k]["depth_exhaustive"]) for k in ks if ok(d[k]))),
                "steps_lookup": sum(s["via"] == "lookup" for s in steps), "steps_online": sum(s["via"] == "online" for s in steps),
                "online_fallback_fraction_of_verified_moves": float(sum(x["online_moves"] for x in st) / max(1, sum(x["online_moves"] + x["lookup_hits"] for x in st))),
                "lookup_hit_rate": float(sum(x["lookup_hits"] for x in st) / max(1, sum(x["lookup_hits"] + x["lookup_misses"] for x in st))),
                **{f"mean_{c}": float(np.mean([x[c] for x in st])) for c in ("nodes", "motion_checks", "online_moves", "target_checks", "D_calls")},
                "modal_rejections_rescued": int(sum(x["modal_rejections_rescued"] for x in st)), "lookup_unknown": int(sum(x["lookup_unknown"] for x in st)),
                "truncated": int(sum(bool(x["truncated"]) for x in st)), "cpu_s_total": float(sum(d[k]["cpu_s"] for k in ks)),
                "cpu_s_mean": float(np.mean([d[k]["cpu_s"] for k in ks]))}
    out = {"common_roots": len(common), "methods": {m: summ(d, common) for m, d in runs.items()}}
    suc = {m: {k: float(ok(d[k])) for k in common} for m, d in runs.items()}
    out["paired_success"] = {f"{a}-{b}": paired(suc[a], suc[b]) for a, b in (("D", "geometry"), ("D", "D-pairless"), ("D", "uninformed"), ("geometry", "uninformed"),
                                                                              ("oracle", "D"), ("oracle", "geometry")) if a in suc and b in suc}
    both = lambda a, b: [k for k in common if ok(runs[a][k]) and ok(runs[b][k])]  # noqa: E731
    out["paired_cost_on_both_solved"] = {f"{a}-{b}": {"moves": paired({k: runs[a][k]["plan"]["moves"] for k in both(a, b)}, {k: runs[b][k]["plan"]["moves"] for k in both(a, b)}),
                                                      "travel": paired({k: runs[a][k]["plan"]["travel"] for k in both(a, b)}, {k: runs[b][k]["plan"]["travel"] for k in both(a, b)}),
                                                      "cpu": paired({k: runs[a][k]["cpu_s"] for k in common}, {k: runs[b][k]["cpu_s"] for k in common})}
                                         for a, b in (("D", "geometry"), ("D", "uninformed"), ("D", "D-pairless")) if a in runs and b in runs}
    out["by_family"] = {f: {m: float(np.mean([ok(d[k]) for k in common if d[k]["family"] == f])) for m, d in runs.items()} for f in sorted({runs["D"][k]["family"] for k in common})}
    abl = {v: rd(S / f"ABL_{v}.json") for v in ("online", "modal", "local", "global") if (S / f"ABL_{v}.json").exists()}
    if abl and "D" in runs:
        ks = sorted(set.intersection(*[set(v) for v in abl.values()]) & set(runs["D"]))
        out["lookup_ablation"] = {"roots": len(ks), "D_combined_all_modes": summ(runs["D"], ks), **{v: summ(d, ks) for v, d in abl.items()}}
        out["modal_only_false_rejections"] = {"combined_rescued_by_rare_mode": int(sum(runs["D"][k]["stats"]["modal_rejections_rescued"] for k in ks)),
                                              "roots_solved_combined_not_modal": [k for k in ks if ok(runs["D"][k]) and not ok(abl["modal"][k])] if "modal" in abl else None}
    cf.write_json(OUT / "eval" / "D3_SUMMARY.json", out) if (OUT / "eval").mkdir(parents=True, exist_ok=True) is None else None
    return out
