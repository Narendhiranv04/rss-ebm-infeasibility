"""BWL figures from saved real outputs only (no model is run): sampler budget / composition / paired deltas / proposal maps / cVAE latent (VAL),
SYNTHETIC X/Y recovery, a schematic same-endpoint AB / BA diagram on a real root, and the 3D Y provenance breakdown.

    taskset -c 7 nice -n 15 venv/bin/python src/figs_bwl.py --out reports/figures"""
from __future__ import annotations

import argparse, json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# fixed entity -> colour (reference categorical order); a rerank arm keeps its base sampler's hue and adds a dashed line + square marker
COL = {"cvae": "#2a78d6", "stratified": "#eb6834", "cat": "#1baf7a", "fm": "#eda100", "sobol": "#e87ba4", "mix0.5": "#6250d6",
       "ranker_catalogue": "#008300", "uniform": "#8a8a8a", "D": "#2a78d6", "E": "#eb6834", "DIRECT-GEO": "#008300", "G": "#6250d6"}
STAT = {"VERIFIED": "#008300", "NEG_CERTIFIED": "#e34948", "NEG_POLICY": "#eda100", "UNKNOWN": "#d4d4d4"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.6})


def col(arm):
    return COL.get(arm.split("+")[0], "#555555")


def style(arm):
    return {"color": col(arm), "ls": "--" if "+rank" in arm else "-", "marker": "s" if "+rank" in arm else "o", "lw": 1.8, "ms": 4}


def budget(S, out):
    arms = ["cvae", "cat", "fm", "uniform", "stratified", "sobol", "mix0.5", "cvae+rank_P2", "stratified+rank_P2", "ranker_catalogue"]
    Ms = [1, 2, 4, 8]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    for arm in arms:
        A = S["arms"][arm]
        for j, (k, ck) in enumerate((("hit", "hit"), ("known_label_precision", "prec"), ("known_label_fraction", None))):
            y = [A["pooled"][str(M)][k] for M in Ms]
            if ck:
                lo = [y[i] - A["root_ci95"][str(M)][ck][0] for i, M in enumerate(Ms)]
                hi = [A["root_ci95"][str(M)][ck][1] - y[i] for i, M in enumerate(Ms)]
                ax[j].errorbar(Ms, y, yerr=[lo, hi], capsize=2, label=arm, **style(arm))
            else:
                ax[j].plot(Ms, y, label=arm, **style(arm))
    for a, t in zip(ax, ("hit@M (contexts with a recorded VERIFIED cell)", "known-label precision@M (recorded labels only)", "known-label fraction of proposals")):
        a.set_xscale("log", base=2); a.set_xticks(Ms, [str(m) for m in Ms]); a.set_xlabel("unique proposals M"); a.set_title(t)
    ax[2].legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
    fig.suptitle(f"BWL-1 VAL only ({S['contexts']} contexts, {S['roots']} roots): bars = 95% root-cluster bootstrap CI; "
                 "untested proposals are UNKNOWN (never negative); not physically verified precision", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(out / "bwl1_budget_val.png", dpi=140); plt.close(fig)


def composition(S, out):
    arms = ["ranker_catalogue", "stratified+rank_P2", "cvae+rank_P2", "cat+rank_P2", "cvae", "cat", "fm", "mix0.5", "stratified", "sobol", "uniform"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [3, 1.2]})
    for i, arm in enumerate(arms):
        p = S["arms"][arm]["pooled"]["4"]
        parts = [p["known_label_fraction"] * p["known_label_precision"], p["certified_negative_fraction"], p["policy_negative_fraction"], p["unknown_fraction"]]
        left = 0.0
        for v, (nm, c) in zip(parts, STAT.items()):
            ax[0].barh(i, v, left=left, color=c, edgecolor="white", linewidth=1.5, label=nm if i == 0 else None)
            left += v
        ax[0].text(1.01, i, f"hit@4 {p['hit']:.3f} | prec {p['known_label_precision']:.3f}", va="center", fontsize=7.5)
    ax[0].set_yticks(range(len(arms)), arms); ax[0].set_xlim(0, 1.32); ax[0].set_xlabel("share of the M = 4 proposals")
    ax[0].legend(ncol=4, fontsize=7.5, frameon=False, loc="lower center", bbox_to_anchor=(0.4, 1.0)); ax[0].grid(axis="y", visible=False)
    D = S["diversity_256"]
    for arm, d in D.items():
        ax[1].scatter(d["effective_cells"], S["arms"][arm]["pooled"]["4"]["known_label_precision"], color=col(arm), s=40, zorder=3)
        ax[1].annotate(arm, (d["effective_cells"], S["arms"][arm]["pooled"]["4"]["known_label_precision"]), fontsize=7, xytext=(4, 2), textcoords="offset points")
    ax[1].set_xlabel("effective cells exp(H) of 256 raw draws"); ax[1].set_ylabel("known-label precision@4"); ax[1].set_title("diversity vs precision (seed 0)")
    fig.suptitle("BWL-1 VAL: what the M = 4 proposals land on (recorded status) - UNKNOWN dominates every arm", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out / "bwl1_composition_val.png", dpi=140); plt.close(fig)


def deltas(S, out):
    P = S["paired_deltas"]
    names = list(P)
    fig, ax = plt.subplots(1, 2, figsize=(13, 0.32 * len(names) + 1.6), sharey=True)
    for j, k in enumerate(("hit", "prec")):
        for i, n in enumerate(names):
            d = P[n]["4"][k]
            c = "#2a78d6" if d["ci95"][0] > 0 else "#eb6834" if d["ci95"][1] < 0 else "#8a8a8a"
            ax[j].errorbar(d["point"], i, xerr=[[d["point"] - d["ci95"][0]], [d["ci95"][1] - d["point"]]], fmt="o", color=c, capsize=2, ms=4)
        ax[j].axvline(0, color="#333333", lw=0.8)
        ax[j].set_title({"hit": "delta hit@4", "prec": "delta known-label precision@4"}[k])
    ax[0].set_yticks(range(len(names)), names); ax[0].invert_yaxis()
    fig.suptitle("BWL-1 VAL paired root-cluster bootstrap (2,000 resamples): blue = CI above 0, orange = below 0, grey = contains 0", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out / "bwl1_paired_deltas_val.png", dpi=140); plt.close(fig)


def heatmaps(z, out):
    arms = [str(a) for a in z["arms"]]
    n = len(z["qid"])
    fig, axs = plt.subplots(n, len(arms), figsize=(1.55 * len(arms), 1.6 * n + 0.6))
    for i in range(n):
        for j, arm in enumerate(arms):
            ax = axs[i, j]
            f = z["freq"][i, j].reshape(6, 6)
            ax.imshow(f, cmap="Blues", vmin=0, vmax=max(0.2, f.max()), origin="lower")
            for k in range(36):
                r, c = divmod(k, 6)
                if z["occ"][i, k] > 0:
                    ax.add_patch(plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, color="#555555", alpha=0.55))
                if z["pos"][i, k]:
                    ax.add_patch(plt.Circle((c, r), 0.33, fill=False, ec=STAT["VERIFIED"], lw=1.8))
                if z["cert"][i, k]:
                    ax.plot(c, r, "x", color=STAT["NEG_CERTIFIED"], ms=7, mew=2)
                if z["pol"][i, k]:
                    ax.plot(c, r, "^", color=STAT["NEG_POLICY"], ms=6, mfc="none", mew=1.6)
            sr, sc = divmod(int(z["src"][i]), 6)
            ax.add_patch(plt.Rectangle((sc - 0.45, sr - 0.45), 0.9, 0.9, fill=False, ec="#eb6834", lw=2))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title(arm, fontsize=7)
    fig.suptitle("BWL-1 six VAL scenes (rows): blue = proposal frequency (256 raw draws; M = 4 set for rerank / catalogue arms); grey = occupied; "
                 "orange box = held object's source; green circle = recorded VERIFIED, red x = NEG_CERTIFIED, yellow triangle = NEG_POLICY; unmarked = UNKNOWN",
                 fontsize=8)
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(out / "bwl1_proposal_maps_val.png", dpi=130); plt.close(fig)


def latent(Lt, out):
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    for s, r in enumerate(Lt["cvae"]):
        ax[0].bar(np.arange(r["zdim"]) + 0.27 * s - 0.27, r["kl_per_dim_nats"], 0.27, label=r["member"], color=["#2a78d6", "#86b6ef", "#1c5cab"][s])
    ax[0].axhline(0.01, color="#e34948", lw=0.8, ls="--"); ax[0].set_xlabel("latent dimension"); ax[0].set_ylabel("KL (nats, VAL positives)")
    ax[0].set_title("cVAE KL per latent dimension (red: 0.01 active-unit line)"); ax[0].legend(fontsize=7, frameon=False)
    mi = [r["mutual_information_z_d_given_S_nats"] for r in Lt["cvae"]]
    ch = [r["fraction_contexts_argmax_changes_with_z"] for r in Lt["cvae"]]
    ax[1].bar(range(3), mi, color="#2a78d6"); [ax[1].text(i, m, f"{m:.3f}\nargmax differs\nacross z: {c:.0%}", ha="center", va="bottom", fontsize=7) for i, (m, c) in enumerate(zip(mi, ch))]
    ax[1].set_xticks(range(3), [r["member"] for r in Lt["cvae"]]); ax[1].set_ylabel("I(z; d | S) nats"); ax[1].set_title("latent information on VAL contexts")
    ax[1].set_ylim(0, max(mi) * 1.6)
    vals = [("cVAE s%d" % i, r["heldout_positive_nll_iwae64"], "#2a78d6") for i, r in enumerate(Lt["cvae"])] + \
           [("cat s%d" % i, r["heldout_positive_nll_exact"], "#1baf7a") for i, r in enumerate(Lt["cat"])] + [("uniform", Lt["uniform_heldout_positive_nll"], "#8a8a8a")]
    ax[2].bar(range(len(vals)), [v[1] for v in vals], color=[v[2] for v in vals]); ax[2].set_ylim(3.0, 3.5)
    ax[2].set_xticks(range(len(vals)), [v[0] for v in vals], rotation=30, fontsize=7); ax[2].set_title("held-out VAL positive NLL (lower = denser)")
    [ax[2].text(i, v[1], f"{v[1]:.3f}", ha="center", va="bottom", fontsize=7) for i, v in enumerate(vals)]
    fig.suptitle("BWL-1 cVAE latent diagnostic vs the latent-free conditional categorical control (VAL)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(out / "bwl1_cvae_latent_val.png", dpi=140); plt.close(fig)


def synthetic(Y, out):
    R = Y["runs"]
    arms = ["D", "E", "DIRECT-GEO", "G"]
    fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
    for t, truth in enumerate(("coupled", "uncoupled")):
        for i, arm in enumerate(arms):
            rs = [r for r in R if r["truth"] == truth and r["arm"] == arm]
            x = t * 5 + i
            kw = {"color": COL[arm], "s": 30, "zorder": 3, "label": arm if t == 0 else None}
            if "heldout_joint_nll" in rs[0]:
                ax[0].scatter([x] * len(rs), [r["heldout_joint_nll"] for r in rs], **kw)
                ax[1].scatter([x] * len(rs), [r["mean_w"] for r in rs], **kw)
            ax[2].scatter([x] * len(rs), [r["order_concordance_a"] for r in rs], **kw)
            ax[3].scatter([x] * len(rs), [r["heldout_success_logloss"] for r in rs], **kw)
        ax[0].hlines(Y[f"planted_truth_heldout_joint_nll_{truth}"], t * 5 - 0.4, t * 5 + 3.4, color="#333333", ls="--", lw=1)
        ax[1].hlines(1.5 if truth == "coupled" else 0.0, t * 5 - 0.4, t * 5 + 3.4, color="#333333", ls="--", lw=1)
    for a, ttl in zip(ax, ("held-out joint NLL (dashed: planted truth)", "learned mean w (dashed: planted w)", "order concordance of a on AB/BA pairs",
                           "held-out success-event log loss")):
        a.set_xticks([1.5, 6.5], ["coupled truth (w=1.5)", "uncoupled truth (w=0)"]); a.set_title(ttl, fontsize=8.5)
    ax[2].axhline(0.5, color="#8a8a8a", lw=0.8)
    h, l = ax[2].get_legend_handles_labels()
    fig.legend(h, l, ncol=4, fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.93))
    fig.suptitle("BWL-2 test 8 - SYNTHETIC planted coefficients on random 6x6 3D-format scenes (2 seeds each); NOT a 3D X/Y training result", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.86]); fig.savefig(out / "bwl2_synthetic_recovery.png", dpi=140); plt.close(fig)


def shared_y(out):
    sys.path.insert(0, str(HERE))
    import xy3d_data as XD
    man = json.loads((XD.OUT / "PHYSICS_X_PILOT_MANIFEST.json").read_text())
    ab = next(r for r in man["rows"] if r["query_kind"] == "PAIR_AB" and r["destination_source"] == "historical_parking")
    ba = next(r for r in man["rows"] if r["query_kind"] == "PAIR_BA_SAME_ENDPOINT" and r["root_id"] == ab["root_id"] and r["y_endpoint"] == ab["y_endpoint"])
    raw = XD.load(ab["root_id"])
    cells, goal = {k: tuple(v) for k, v in raw["initial_cells"].items()}, tuple(raw["goal"])
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, row, ttl in ((axs[0], ab, "X query 1: order A then B"), (axs[1], ba, "X query 2: order B then A"), (axs[2], None, "shared endpoint: ONE Y node")):
        cur = dict(cells)
        if row is None:
            for o, d in ab["ordered_moves"]:
                cur[o] = tuple(d)
        ax.set_xlim(-0.6, 5.6); ax.set_ylim(-0.6, 5.6); ax.set_aspect("equal"); ax.set_xticks(range(6)); ax.set_yticks(range(6))
        ax.add_patch(plt.Rectangle((goal[1] - 0.45, goal[0] - 0.45), 0.9, 0.9, fill=False, ec="#008300", ls="--", lw=1.5))
        for o, (r, c) in cur.items():
            ax.add_patch(plt.Circle((c, r), 0.32, color="#008300" if o == "t0" else "#8a8a8a"))
            ax.text(c, r, o, ha="center", va="center", fontsize=6.5, color="white")
        if row is not None:
            for k, (o, d) in enumerate(row["ordered_moves"]):
                r0, c0 = cells[o]
                ax.annotate("", xy=(d[1], d[0]), xytext=(c0, r0), arrowprops={"arrowstyle": "->", "color": "#2a78d6" if k == 0 else "#eb6834", "lw": 2})
                ax.text((c0 + d[1]) / 2, (r0 + d[0]) / 2 + 0.25, f"{k + 1}", color="#2a78d6" if k == 0 else "#eb6834", fontsize=10, weight="bold")
        ax.set_title(ttl, fontsize=9); ax.set_xlabel("column"); ax.set_ylabel("row (0 = nearest the robot)")
    fig.suptitle(f"SCHEMATIC on real root {ab['root_id']} (predeclared X pilot queries; NO outcome exists yet)\n"
                 "two X variables (P(X | Y) may differ by order), one shared Y variable for the identical endpoint; "
                 "arrows = relocation source -> destination (not robot paths); green dashed = target goal", fontsize=8.5)
    fig.tight_layout(rect=[0, 0.02, 1, 0.88]); fig.savefig(out / "bwl2_shared_y_ab_ba_schematic.png", dpi=140); plt.close(fig)


def provenance(A, out):
    cats = ["Y_PLANNING_ONLY", "Y_POLICY_FAILED:CERTIFIED_GRASP_SCREEN", "Y_POLICY_FAILED:PLANNER_BUDGET"]
    cc = {"Y_PLANNING_ONLY": "#2a78d6", "Y_POLICY_FAILED:CERTIFIED_GRASP_SCREEN": "#e34948", "Y_POLICY_FAILED:PLANNER_BUDGET": "#eda100"}
    groups = [(f, s) for f in ("original", "D") for s in ("train", "val", "test")]
    fig, ax = plt.subplots(figsize=(10, 3.8))
    for i, (f, s) in enumerate(groups):
        left = 0
        for c in cats:
            v = A["by_category"].get(c, {}).get(f, {}).get(s, 0)
            ax.barh(i, v, left=left, color=cc[c], edgecolor="white", linewidth=1.5, label=c if i == 0 else None)
            if v > 60:
                ax.text(left + v / 2, i, str(v), ha="center", va="center", fontsize=7, color="white")
            left += v
    ax.set_yticks(range(len(groups)), [f"{f} / {s}" for f, s in groups]); ax.invert_yaxis(); ax.set_xlabel("records"); ax.grid(axis="y", visible=False)
    ax.legend(fontsize=7.5, frameon=False, loc="lower right")
    fig.suptitle(f"BWL-3: the {A['records']:,} 3D target-only records by provenance - Y_REPLAY_VERIFIED = {A['Y_REPLAY_VERIFIED']} "
                 f"\n(positives were planned, not executed or replayed; {A['original_positives_executed_once_in_teacher']} K0 originals were executed once inside the teacher)", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.88]); fig.savefig(out / "bwl3_y_provenance.png", dpi=140); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="reports/figures")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    S = json.loads((ROOT / "runs/SAMPLER_BWL1/study_val/study_val.json").read_text())
    budget(S, out); composition(S, out); deltas(S, out)
    heatmaps(np.load(ROOT / "runs/SAMPLER_BWL1/study_val/examples_val.npz"), out)
    latent(json.loads((ROOT / "runs/SAMPLER_BWL1/latent_val/latent_val.json").read_text()), out)
    synthetic(json.loads((ROOT / "runs/XY3D_BWL/synthetic_recovery.json").read_text()), out)
    shared_y(out)
    provenance(json.loads((ROOT / "runs/XY3D_BWL/Y_PROVENANCE_AUDIT.json").read_text()), out)
    print("figures:", sorted(p.name for p in out.glob("bwl*.png")))
