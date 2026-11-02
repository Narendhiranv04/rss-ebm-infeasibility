"""P17 figures: mechanism panels (initial / after each certified relocation / final with the target at its goal) and GIFs drawn ONLY
from replay-certified recorded arrays; B and D result plots."""
from __future__ import annotations

from pathlib import Path

import numpy as np

DST = Path("visualizations/p17")


def mechanism(phys, root, plan, tag, rec) -> dict:
    import geometry as ge
    import motion as mo
    import p14_eval as p14
    import p6_scenes as p6
    import p6_viz as pv
    from viz import COLORS, plt
    steps, recs = plan["steps"], plan["records"]
    rel, i = [], 0
    for st in steps:
        rel.append(p14._merge(recs[i], recs[i + 1]) if st["via"] == "lookup" else recs[i])
        i += 2 if st["via"] == "lookup" else 1
    s0, goal = phys.s0, phys.goal_cell
    n = len(steps) + 2
    f, ax = plt.subplots(1, n, figsize=(4.6 * n, 4.8), squeeze=False)
    pv._panel(ax[0][0], phys.cfg, phys.robot, s0, goal, f"{tag}: initial ({rec['original_status'][:9]})")
    cells = {o.id: o.cell for o in s0.objects}
    for j, (st, a) in enumerate(zip(steps, rel)):
        cells[st["object"]] = st["dest"]
        sj = p6.moved(s0, {o: c for o, c in cells.items() if o != "t0"})
        pv._panel(ax[0][j + 1], phys.cfg, phys.robot, sj, goal, f"after {j + 1}: {st['object']}->{st['dest']} ({st['via']})")
        pts = np.array([ge.fk(phys.robot, q)[0][-1] for q, _, _ in mo.samples(a, 60)])
        ax[0][j + 1].plot(pts[:, 0], pts[:, 1], color="#2a6fdb", lw=1.4)
    sN = p6.moved(s0, {o: c for o, c in cells.items() if o != "t0"})
    pv._panel(ax[0][-1], phys.cfg, phys.robot, sN, goal, f"target action (travel {plan['travel']:.2f})")
    pts = np.array([ge.fk(phys.robot, q)[0][-1] for q, _, _ in mo.samples(recs[-1], 60)])
    ax[0][-1].plot(pts[:, 0], pts[:, 1], color=COLORS["target"], lw=1.4)
    (DST / "mechanisms").mkdir(parents=True, exist_ok=True)
    png = DST / "mechanisms" / f"{tag}.png"
    pv.png(f, png, f"P17 mechanism {tag}: certified ordered repair (tool paths from replayed arrays)")
    frames = []
    for qq, gp, pos, mv in pv.trace_frames(phys.robot, s0, [(s["object"], s["dest"]) for s in steps], rel, recs[-1:], n_move=10, n_target=12):
        fg, a = plt.subplots(figsize=(5, 5))
        pv._panel(a, phys.cfg, phys.robot, s0, goal, f"{tag}: {len(steps)} moves\nRECORDED / CERTIFIED arrays", q=qq, gap=gp, pos=pos, held=mv)
        pv._tag(a, f"replay CLEAR {rec['positive']['replay']['traj']}", COLORS["valid"])
        frames.append(pv._rgb(fg))
    gif = DST / "mechanisms" / f"{tag}.gif"
    pv.gif(gif, frames, f"P17 mechanism {tag}: certified replay")
    return {"png": str(png), "gif": str(gif)}


# ------------------------------------------------------------------ result figures (palette: 3 highlighted series + neutral controls; status colours)
KEY = {"LB": "#2a78d6", "LB+G": "#eb6834", "FULLB": "#1baf7a", "Dp17": "#2a78d6", "geometry": "#eb6834", "oracle": "#1baf7a"}
NEUTRAL, STATUS = "#a7a6a1", {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b", "none": "#d6d5d0"}


def learning_curves(arms, seeds=(0, 1, 2), out="b/P17_B_learning_curves") -> None:
    """Validation root NLL vs update for every B arm / seed (early-stopping point marked)."""
    import json
    import p6_viz as pv
    from viz import plt
    f, ax = plt.subplots(1, 1, figsize=(9, 5.2))
    for a in arms:
        for s in seeds:
            p = Path(f"runs/p17/b/runs/P17-{a}-s{s}.json")
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            h = [(x["update"] + 1, x["val"]) for x in r["history"] if x.get("val") is not None]
            ax.plot(*zip(*h), color=KEY.get(a, NEUTRAL), lw=1.6 if a in KEY else 0.8, alpha=0.9, label=a if s == seeds[0] else None)
            ax.plot(r["best_update"], r["val_root_nll"], "o", ms=4, color=KEY.get(a, NEUTRAL))
    ax.set(xlabel="update", ylabel="validation root NLL", title="P17 B training (dots: restored best checkpoint; early stopping)")
    ax.legend(fontsize=7, ncol=2, framealpha=0.95)
    (DST / out).parent.mkdir(parents=True, exist_ok=True)
    pv.png(f, DST / f"{out}.png", "P17 B learning curves (validation root NLL, every seed)")


def b_figures(an, out="b/P17_B_test") -> None:
    """Decision breakdown on decided feasible / infeasible originals per arm, and the prevalence curve of strict TOP1."""
    import p6_viz as pv
    from viz import plt
    P, arms = an["per_arm"], list(an["per_arm"])
    f, ax = plt.subplots(1, 3, figsize=(23, 5.8))
    x = np.arange(len(arms))
    for j, (side, parts) in enumerate((("noop", [("correct no-op", "correct", "good"), ("unnecessary relocation", "unnecessary_relocation", "critical"),
                                                 ("abstain", "abstain", "none")]),
                                       ("repair", [("repair verified", "success", "good"), ("wrong no-op", "wrong_noop", "critical"),
                                                   ("abstain", "abstain", "none")]))):
        n = max(1, P[arms[0]][side]["feasible_n" if side == "noop" else "infeasible_n"])
        bot = np.zeros(len(arms))
        for lab, k, c in parts:
            v = np.array([P[a][side][k] / n for a in arms])
            ax[j].bar(x, v, 0.65, bottom=bot, color=STATUS[c], label=lab, edgecolor="white", lw=1)
            bot += v
        rest = np.array([(P[a]["noop"]["feasible_n"] - P[a]["noop"]["correct"] - P[a]["noop"]["unnecessary_relocation"] - P[a]["noop"]["abstain"]) / n
                         if side == "noop" else (P[a]["repair"]["attempted"] - P[a]["repair"]["success"]) / n for a in arms])
        ax[j].bar(x, rest, 0.65, bottom=bot, color=STATUS["warning"], label="no-op chosen, not verified" if side == "noop" else "repair attempted, not verified",
                  edgecolor="white", lw=1)
        ax[j].set_xticks(x, arms, rotation=45, ha="right", fontsize=7)
        ax[j].set(ylim=(0, 1.05), ylabel=f"fraction of decided {'feasible' if side == 'noop' else 'infeasible'} originals (n = {n})",
                  title="feasible originals: no-op decision" if side == "noop" else "infeasible originals: repair decision")
        ax[j].legend(fontsize=7, loc="lower right", framealpha=0.95)
    pi, ends = np.linspace(0, 1, 21), []
    for a in arms:
        nf, ni = max(1, P[a]["noop"]["feasible_n"]), max(1, P[a]["repair"]["infeasible_n"])
        y = pi * P[a]["noop"]["correct"] / nf + (1 - pi) * P[a]["repair"]["success"] / ni
        ax[2].plot(pi, y, color=KEY.get(a, NEUTRAL), lw=2 if a in KEY else 0.8)
        if a in KEY or a in ("empty_always", "verify_first"):
            ends.append([y[-1], a])
    ends.sort()
    for i in range(1, len(ends)):                                     # keep direct labels >= 0.035 apart
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.035)
    for yy, a in ends:
        ax[2].text(1.01, yy, a, fontsize=7, va="center")
    ax[2].set(xlabel="feasible-original prevalence", ylabel="expected strict TOP1", title="prevalence sensitivity (per-class rates reweighted)", xlim=(0, 1.15))
    (DST / out).parent.mkdir(parents=True, exist_ok=True)
    pv.png(f, DST / f"{out}.png", "P17-B: no-op / repair breakdown (denominators = decided originals) and prevalence curve")


def d_figures(rows, an, out="d/P17_D") -> None:
    """Anytime success vs nodes, mean motion checks + virtual CPU, co-solved travel vs geometry."""
    import p6_viz as pv
    from viz import plt
    g = list(an["per_guide"])
    f, ax = plt.subplots(1, 3, figsize=(21, 5.6))
    ts, ends = [10, 25, 50, 100, 200, 500], []
    for k in g:
        y = [sum(1 for r in rows.values() if r["guides"][k]["found"] and r["guides"][k]["replay"] and
                 (r["guides"][k]["log"].get("anytime") or [{}])[0].get("nodes", 1e9) <= t) for t in ts]
        ax[0].plot(ts, y, marker="o", ms=3, lw=2 if k in KEY else 0.9, color=KEY.get(k, NEUTRAL))
        ends.append([y[-1], f"{k} ({an['per_guide'][k]['success']}/{an['common_roots']})"])
    ends.sort()
    lo, hi = ax[0].get_ylim()
    gap = 0.045 * (hi - lo)                                          # label spacing in axis units, not root counts
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for yy, lab in ends:
        ax[0].text(ts[-1] * 1.15, yy, lab, fontsize=7, va="center")
    ax[0].set(xscale="log", xlabel="search nodes to first verified plan", ylabel="roots solved (replay CLEAR)", title="anytime success", xlim=(8, 6000),
              ylim=(lo, max(hi, ends[-1][0] + gap)))
    x = np.arange(len(g))
    ax[1].bar(x, [an["per_guide"][k]["calls_mean"].get("motion_checks", 0) for k in g], 0.6, color=[KEY.get(k, NEUTRAL) for k in g])
    for i, k in enumerate(g):
        ax[1].text(i, an["per_guide"][k]["calls_mean"].get("motion_checks", 0) + 2, f"{an['per_guide'][k]['cpu_virtual_mean']:.0f}s", ha="center", fontsize=7)
    ax[1].set_xticks(x, g, rotation=40, ha="right", fontsize=7)
    ax[1].set(ylabel="mean motion checks per root (label: virtual CPU incl. D inference)", title="search effort")
    for k in g:
        if k == "geometry":
            continue
        both = [r for r in rows.values() if r["guides"][k]["found"] and r["guides"][k]["replay"] and r["guides"]["geometry"]["found"] and r["guides"]["geometry"]["replay"]]
        ax[2].scatter([r["guides"]["geometry"]["travel"] for r in both], [r["guides"][k]["travel"] for r in both], s=14,
                      color=KEY.get(k, NEUTRAL), alpha=0.8, label=k if k in KEY else None)
    lim = ax[2].get_xlim()[1]
    ax[2].plot([0, lim], [0, lim], color="#7a7975", ls="--", lw=1)
    ax[2].set(xlabel="geometry: plan travel", ylabel="guide: plan travel", title="co-solved roots (grey: other guides)")
    ax[2].legend(fontsize=7, framealpha=0.95)
    (DST / out).parent.mkdir(parents=True, exist_ok=True)
    pv.png(f, DST / f"{out}.png", "P17-D: anytime success, effort and co-solved travel (identical action domain, budgets and certifier)")


def calibration_figure(summary, out="d/P17_D_calibration") -> None:
    """Per-family temporary-target calibration: raw vs calibrated feasible / infeasible recall and CV log loss of each form."""
    import p6_viz as pv
    from viz import plt
    ks = [k for k in summary if k != "probes"]
    f, ax = plt.subplots(1, 2, figsize=(13, 4.8))
    x = np.arange(len(ks))
    for j, (lab, fn) in enumerate((("raw", lambda c: c["raw"]["feasible_recall"]), ("calibrated (CV)", lambda c: c["cv"]["feasible_recall"]),
                                   ("held-out", lambda c: c["held_out"]["feasible_recall"]))):
        ax[0].bar(x + (j - 1) * 0.25, [fn(summary[k]) or 0 for k in ks], 0.24, color=["#86b6ef", "#2a78d6", "#1c5cab"][j], label=lab)
    ax[0].set_xticks(x, ks)
    ax[0].set(ylim=(0, 1.18), ylabel="feasible recall at p' >= 0.5", title="temporary-target feasibility per family (own calibration)")
    ax[0].legend(fontsize=7, loc="upper center", ncol=3, framealpha=0.95)
    for j, form in enumerate(("identity", "bias", "full")):
        ax[1].bar(x + (j - 1) * 0.25, [summary[k]["cv_forms"][form]["cv_logloss"] or 0 for k in ks], 0.24, color=["#d6d5d0", "#2a78d6", "#eb6834"][j], label=form)
    ax[1].set_xticks(x, ks)
    ax[1].set(ylabel="root-split 2-fold CV log loss", title="calibration form selection (lowest chosen)")
    ax[1].legend(fontsize=7)
    (DST / out).parent.mkdir(parents=True, exist_ok=True)
    pv.png(f, DST / f"{out}.png", f"P17-D per-family calibration (probes: {summary.get('probes')})")
