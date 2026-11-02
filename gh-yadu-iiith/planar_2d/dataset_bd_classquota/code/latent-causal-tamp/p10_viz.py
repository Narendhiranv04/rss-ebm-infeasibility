"""P10 figures from recorded data only (P6.7 conventions via p6_viz): robot motion only from revalidated witness arrays;
certified negatives as static panels with their certificate kind; BUDGET / UNKNOWN / NOT_RUN as "NO VERIFIED TRACE";
fixed-robot edits tagged "HYPOTHETICAL SCENE EDIT"; catalogue arrows tagged "PROPOSED, NOT EXECUTED". SHA-256 beside every file."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config as cf
import effects as ef
import geometry as ge
import operational_teacher as ot
import p10_collect as pc
import p6_viz as pv
from scene import scene_from_dict
from viz import COLORS, plt

DST = Path("visualizations/p10")
ARM_COL = {"K-B-full": "#2a6fdb", "K-C-full": "#e07b00", "A": "#2e9e5b", "SF": "#8e5bd0", "FROZEN": "#5f6b7a"}


def _rd(p):
    return json.loads(Path(p).read_text())


def _arm(ck):
    return "FROZEN" if ck.startswith("FROZEN") else ck.split("#")[0]


def _arm_p(pred, root, q) -> dict:
    """Seed-mean root-prior p per arm (frozen snapshot), None when the arm does not model q."""
    out = {}
    for ck, a in pred["arms"].items():
        p = a["roots"][root]["marginals"].get(q)
        if p is not None:
            out.setdefault(_arm(ck), []).append(p)
    return {k: float(np.mean(v)) for k, v in out.items()}


def _roles(ax, s, r):
    inv = {v: k for k, v in r["ids"].items()}
    for o in s.objects:
        if o.id in inv:
            ax.text(o.xy_m[0], o.xy_m[1] + 0.03, inv[o.id], ha="center", fontsize=8, weight="bold", color="#c0392b", zorder=12)


def atlas(cfg, robot, plan) -> None:
    """All 14 constructed roots at q0 (pre-outcome): roles, goal (dashed), start stratum, anchor PROPOSED arrows."""
    ks = list(plan["roots"])
    f, ax = plt.subplots(2, 7, figsize=(28, 8.6))
    for a, k in zip(ax.flat, ks):
        r = plan["roots"][k]
        s = scene_from_dict(r["scene"])
        pv._panel(a, cfg, robot, s, r["goal_cell"], f"{k[-2:]} {r['mechanism']}{(' ' + r['variant']) if r['variant'] else ''} w{r['wave']} {r['stratum']}")
        _roles(a, s, r)
        anc = next(q for q in plan["queries"] if q["root"] == k and q["anchor"])
        if anc["sequence"]:
            pv._arrows(a, s, [tuple(x) for x in anc["sequence"]], COLORS["hypothetical"])
        pv._tag(a, f"anchor {pc._tok(plan, anc)} - PROPOSED, NOT EXECUTED", COLORS["hypothetical"], size=6)
    pv.png(f, DST / "P10a_scene_atlas.png", "P10a: 14 constructed roots at q0 (roles, goal, stratum, anchor proposal); constructor-targeted, ADAPTIVE")


def root_card(cfg, robot, plan, recs, pred, k) -> None:
    r = plan["roots"][k]
    s, mine = scene_from_dict(r["scene"]), sorted([q for q in plan["queries"] if q["root"] == k], key=lambda q: q["priority"])
    f, ax = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1, 1.45]})
    pv._panel(ax[0], cfg, robot, s, r["goal_cell"], f"{k} [{r['mechanism']} {r['variant']}] {r['stratum']}: catalogue destinations")
    _roles(ax[0], s, r)
    for q in mine:
        if q["sequence"]:
            pv._arrows(ax[0], s, [tuple(x) for x in q["sequence"]], COLORS["hypothetical"])
    pv._tag(ax[0], "PROPOSED, NOT EXECUTED", COLORS["hypothetical"])
    ax[1].axis("off")
    arms = [a for a in ARM_COL if any(_arm(ck) == a for ck in pred["arms"])]
    ax[1].text(0.0, 0.97, f"{'token':<10}{'status':<22}{'source':<14}" + "".join(f"{a[:8]:>9}" for a in arms), fontsize=7, family="monospace",
               transform=ax[1].transAxes, weight="bold")
    for i, q in enumerate(mine):
        x, y = recs[q["query"]], 0.97 - (i + 1) * 0.055
        st, p = x["teacher_status"], _arm_p(pred, k, q["query"])
        src = (x.get("static_certificate") or {}).get("kind") or x.get("label_source") or ""
        ax[1].add_patch(plt.Rectangle((-0.035, y), 0.025, 0.035, fc=pv.COL.get(st, "#999"), transform=ax[1].transAxes))
        ax[1].text(0.0, y, f"{pc._tok(plan, q) or '(empty)':<10}{st[:21]:<22}{src[:13]:<14}" + "".join(
            f"{p[a]:>9.2f}" if a in p else f"{'-':>9}" for a in arms) + ("  <- anchor" if q["anchor"] else ""), fontsize=7, family="monospace",
            transform=ax[1].transAxes)
    ax[1].text(0.0, 0.02, "p = frozen ROOT-PRIOR P(feasible), seed mean per arm, written before any P10a outcome", fontsize=7, transform=ax[1].transAxes)
    pv.png(f, DST / "roots" / f"{k}_card.png", f"{k}: catalogue, adjudicated statuses and frozen predictions")


def evidence_gif(cfg, robot, plan, recs, k) -> None:
    """Recorded witnesses of VERIFIED queries (sequences and scene edits; <= 4) then static panels of the decided negatives."""
    r = plan["roots"][k]
    s, goal = scene_from_dict(r["scene"]), r["goal_cell"]
    mine = sorted([recs[q["query"]] for q in plan["queries"] if q["root"] == k], key=lambda x: x["query"]["priority"])
    frames = []
    for x in [x for x in mine if x["teacher_status"] == ot.VERIFIED][:4]:
        q, w = x["query"], pv.witness(pc.OUT, x)
        se = ef.pose_substitution(robot, cfg, s, {"target_id": "t0"}, q["edit"])["scene"] if q["kind"] == "SCENE_TARGET_ONLY" else s
        seq = [tuple(a) for a in q["sequence"]]
        for qq, g, pos, mv in pv.trace_frames(robot, se, seq, w[1], w[2]):
            f, ax = plt.subplots(figsize=(5, 5))
            pv._panel(ax, cfg, robot, se, goal, f"{k[-2:]} {pc._tok(plan, q) or '(empty)'}: recorded witness", q=qq, gap=g, pos=pos, held=mv)
            if q["kind"] == "SCENE_TARGET_ONLY":
                pv._tag(ax, "HYPOTHETICAL SCENE EDIT - NOT A ROBOT RELOCATION", "#34495e", loc=(0.02, 0.93), size=6)
            pv._tag(ax, "VERIFIED - recorded trajectory", COLORS["valid"])
            frames.append(pv._rgb(f))
    for x in [x for x in mine if x["teacher_status"] != ot.VERIFIED]:
        q = x["query"]
        se = ef.pose_substitution(robot, cfg, s, {"target_id": "t0"}, q["edit"])["scene"] if q["kind"] == "SCENE_TARGET_ONLY" else s
        f, ax = plt.subplots(figsize=(5, 5))
        pv._panel(ax, cfg, robot, se, goal, f"{k[-2:]} {pc._tok(plan, q) or '(empty)'}")
        if q["sequence"]:
            pv._arrows(ax, s, [tuple(a) for a in q["sequence"]], pv.COL.get(x["teacher_status"], "#999"))
        if q["kind"] == "SCENE_TARGET_ONLY":
            pv._tag(ax, "HYPOTHETICAL SCENE EDIT - NOT A ROBOT RELOCATION", "#34495e", loc=(0.02, 0.93), size=6)
        pv._tag(ax, pv._status_line(x) if x["teacher_status"] in pv.NOTRACE else x["teacher_status"], pv.COL.get(x["teacher_status"], "#999"), size=6)
        frames += [pv._rgb(f)] * 6
    pv.gif(DST / "roots" / f"{k}_evidence.gif", frames, f"{k}: recorded witnesses, then static panels for non-VERIFIED outcomes")


def summary(plan, recs, pred, kap, comp) -> None:
    import p10_eval as pe
    f, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    arms = [a for a in ARM_COL if any(_arm(ck) == a for ck in pred["arms"])]
    for i, x in enumerate(kap):
        for j, a in enumerate(arms):
            v = [h for ck, h in x["kappa_hat"].items() if _arm(ck) == a and h is not None]
            if v:
                ax[0].scatter(i + (j - len(arms) / 2) * 0.12, np.mean(v), s=36, color=ARM_COL[a], label=a if i == 0 else None, zorder=3)
        if x["decided"]:
            ax[0].scatter(i, x["kappa"], marker="_", s=500, color="black", lw=2.5, label="observed kappa" if i == 0 else None, zorder=4)
    ax[0].set(xticks=range(len(kap)), xticklabels=[f"{x['root'][-2:]}\n{x['mechanism']}\n{(x['stratum'] or 'undecided')[:5]}" for x in kap],
              ylabel="kappa (observed bar; predicted dots, seed mean)", ylim=(-1.6, 1.6), title="Interaction contrast per root (H1 / H2)")
    ax[0].axhline(0, color="#bbb", lw=0.8)
    ax[0].legend(fontsize=7, loc="lower right")
    mechs, cats = ps_mechs(plan), [ot.VERIFIED, ot.CERTIFIED, ot.BUDGET, ot.UNKNOWN, "NOT_RUN"]
    bottom = np.zeros(len(mechs))
    for c in cats:
        v = np.array([sum(x["teacher_status"] == c for x in recs.values() if x["query"]["mechanism"] == m) for m in mechs])
        ax[1].bar(mechs, v, bottom=bottom, color=pv.COL[c], label=c, edgecolor="white", lw=1)
        bottom += v
    ax[1].set(title="P10a outcomes by mechanism (queries)", ylabel="queries")
    ax[1].legend(fontsize=7)
    rows = pe.binary_rows(plan, recs, pred)
    for a in arms:
        rr = [x for x in rows if _arm(x["ck"]) == a]
        if not rr:
            continue
        p, y = np.array([x["p"] for x in rr]), np.array([x["y"] for x in rr])
        b = np.minimum((p * 5).astype(int), 4)
        xs = [(p[b == i].mean(), y[b == i].mean()) for i in range(5) if (b == i).any()]
        ax[2].plot(*zip(*xs), "-o", color=ARM_COL[a], lw=2, ms=6, label=f"{a} (n={len(rr)})")
    ax[2].plot([0, 1], [0, 1], ":", color="#999")
    ax[2].set(xlabel="frozen root-prior p", ylabel="observed feasible fraction", title="Reliability on decided P10a outcomes", xlim=(0, 1), ylim=(0, 1))
    ax[2].legend(fontsize=7)
    pv.png(f, DST / "P10a_summary.png", "P10a: kappa per root (H1/H2), outcomes by mechanism, reliability per arm (frozen predictions)")
    f, ax = plt.subplots(figsize=(7, 3.6))
    ks = list(comp)
    ax.barh(ks, [comp[k]["used"] for k in ks], color="#2a6fdb", label="used")
    ax.barh(ks, [comp[k]["cap"] - comp[k]["used"] for k in ks], left=[comp[k]["used"] for k in ks], color="#dde3ea", label="remaining cap")
    for i, k in enumerate(ks):
        ax.text(comp[k]["cap"] + 30, i, f"{comp[k]['used']:.0f} / {comp[k]['cap']:.0f}", va="center", fontsize=8)
    ax.set(xlabel="CPU-s", title="P10 compute by ledger category")
    ax.legend(fontsize=7, loc="lower right")
    pv.png(f, DST / "P10_compute.png", "P10 compute ledger: used vs cap per category (guard records)")


def ps_mechs(plan):
    return sorted({r["mechanism"] for r in plan["roots"].values()})


def run() -> None:
    import p10_eval as pe
    import p10_protocol as pp
    import p6_run as p6r
    (DST / "roots").mkdir(parents=True, exist_ok=True)
    cfg = cf.load_config("config.toml")
    robot, plan, pred = ge.make_robot(cfg), _rd(pc.OUT / "plan.json"), _rd(pc.OUT / "predictions.json")
    pol, static = ot.load_policy("teacher_opneg.toml"), _rd(pc.OUT / "static_admission.json")["certificates"]
    prog = _rd(pc.OUT / "progress.json")
    recs = {q["query"]: p6r.judge(static, pol, q, prog["done"].get(q["query"])) for q in plan["queries"]}
    atlas(cfg, robot, plan)
    for k in plan["roots"]:
        root_card(cfg, robot, plan, recs, pred, k)
        evidence_gif(cfg, robot, plan, recs, k)
    comp = {c: {"used": pp.spent(c)[0], "cap": pp.CAPS[c]} for c in pp.CAPS}
    summary(plan, recs, pred, pe.kappa_rows(plan, recs, pred), comp)
    cf.write_json(DST / "INDEX.json", pv.INDEX)


def p10_0() -> None:
    """P10-0: training objective per run (smoothed; B / C own joint NLL vs A / SF factorized loss - different objectives, separate
    panels) and per-lineage validation NLL of the same observed bits (dots = seeds)."""
    import p10_protocol as pp
    DST.mkdir(parents=True, exist_ok=True)
    runs = [_rd(f) for f in sorted((pp.P0 / "runs").glob("*.json"))]
    f, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    for r in runs:
        y = np.convolve([h["nll"] for h in r["history"]], np.ones(20) / 20, mode="valid")
        ax[0 if r["kind"] == "keyed" else 1].plot(y, color=ARM_COL[r["arm"]], lw=1.5, alpha=0.85, label=f"{r['arm']} s{r['seed']}")
    ax[0].set(title="Keyed B / C: own exact joint NLL (train batches, 20-update mean)", xlabel="update", ylabel="NLL per batch", yscale="log")
    ax[1].set(title="A / SF: factorized Bernoulli loss (NOT a joint NLL)", xlabel="update", yscale="log")
    for a in ax[:2]:
        a.legend(fontsize=6, ncol=2)
    arms = [a for a in ARM_COL if any(r["arm"] == a for r in runs)]
    keys = sorted(pp._per_root(runs[0]))
    for j, a in enumerate(arms):
        for r in [r for r in runs if r["arm"] == a]:
            pr = pp._per_root(r)
            ax[2].scatter(np.arange(len(keys)) + (j - len(arms) / 2) * 0.15, [pr[k][0] for k in keys], s=22, color=ARM_COL[a],
                          label=a if r["seed"] == 0 else None, zorder=3)
    ax[2].set(xticks=range(len(keys)), xticklabels=[k[-2:] for k in keys], xlabel="validation lineage (obstructed + _ref)",
              ylabel="NLL of observed bits", title="Validation NLL per independent lineage (dots = seeds)")
    ax[2].legend(fontsize=7)
    pv.png(f, DST / "P10_0_training_and_validation.png", "P10-0: training curves and per-lineage validation NLL of A / B / C / SF")
