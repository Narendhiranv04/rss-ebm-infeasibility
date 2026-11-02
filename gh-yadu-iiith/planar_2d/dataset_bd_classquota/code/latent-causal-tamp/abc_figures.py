"""G4E-ABC diagnostic figures, drawn only from results.json (saved evaluations of real checkpoints)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from viz import COLORS, plt

ARCH = ("A", "B", "C")
CLR = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c", "A-no-order": "#9ecae1", "B-no-coupling": "#fc9272", "C-no-coupling": "#a1d99b"}


def _save(f, out, name, dpi=70):
    f.tight_layout()
    f.savefig(Path(out) / name, dpi=dpi)
    plt.close(f)


def figures(out, res, ds):
    runs = res["runs"]
    f, axs = plt.subplots(3, 3, figsize=(19, 13))
    for ax, tag in zip(axs.ravel(), runs):
        h = runs[tag]["history"]
        for k in ("total", "static", "task", "outcome", "effect", "cost", "pool"):
            ys = [x.get(k) for x in h]
            if any(y is not None for y in ys):
                ax.plot([y if y is not None else np.nan for y in ys], label=k, lw=1.2 if k != "total" else 2.2)
        ax.set(yscale="log", title=f"{tag} ({runs[tag]['params']:,} params)", xlabel="epoch (full batch)")
        ax.legend(fontsize=6)
    _save(f, out, "E1_loss_curves.png")
    f, axs = plt.subplots(len(ARCH), 1, figsize=(18, 4.2 * len(ARCH)))
    for ax, tag in zip(axs, ARCH):
        x, labels = 0, []
        for ev in runs[tag]["eval"]["train"]:
            for c in ev["cands"]:
                col = COLORS["valid"] if c["y"] == 1 else COLORS["invalid"] if c["y"] == 0 else "#cccccc"
                ax.bar(x, c["energy"], color=col, hatch="//" if c["source"] == "PLANNER_BUDGET" else None, edgecolor="#333")
                labels.append(c["query"].split(":", 1)[1][:26])
                x += 1
            x += 1
            labels.append("")
        ax.set_xticks(range(len(labels)), labels, rotation=90, fontsize=5)
        ax.set(ylabel="E_A (feas.)" if tag == "A" else f"E_bar_{tag} (exact z-marginal)")
        ax.set_title(f"{tag}: learned energy per decided training candidate (green = witness, red = negative, hatched = budget "
                     f"negative, grey = UNKNOWN masked); lower = preferred", fontsize=8, loc="left")
    _save(f, out, "E2_energies_by_candidate.png", 60)
    f, axs = plt.subplots(1, 3, figsize=(19, 5.5))
    for ax, split in zip(axs, ("train", "validation", "holdout")):
        k = 0
        for tag in ("A", "A-no-order", "B", "C"):
            for ev in runs[tag]["eval"].get(split, []):
                for pr in ev["order_pairs"]:
                    if None in pr["y"] or pr["y"][0] == pr["y"][1]:
                        continue
                    pos, neg = (0, 1) if pr["y"][0] == 1 else (1, 0)
                    ax.plot([k, k], [pr["p"][neg], pr["p"][pos]], color=CLR[tag], lw=2)
                    ax.scatter([k], [pr["p"][pos]], color=COLORS["valid"], zorder=3)
                    ax.scatter([k], [pr["p"][neg]], color=COLORS["invalid"], zorder=3)
                    ax.text(k, 1.02, tag, ha="center", fontsize=6, rotation=90)
                    k += 1
        ax.set(ylim=(0, 1.1), ylabel="predicted task probability", title=f"{split}: decided same-endpoint order pairs "
               "(green = witnessed order, red = negative order)")
    _save(f, out, "E3_order_pairs.png")
    f, axs = plt.subplots(2, 3, figsize=(17, 10))
    for ax, tag in zip(axs[0], ARCH):
        for split, mk in (("train", "o"), ("validation", "s"), ("holdout", "^")):
            pts = [(e["observed"], e["pred"]) for ev in runs[tag]["eval"].get(split, []) for e in ev["effects"]]
            if pts:
                o, p = np.array(pts).T
                ax.scatter(o + np.random.default_rng(0).uniform(-0.08, 0.08, len(o)), p, marker=mk, label=split, alpha=0.7)
        ax.plot([-1, 1], [-1, 1], "k--", lw=0.8)
        ax.set(xlabel="observed effect (decided pairs, jittered)", ylabel="predicted effect", title=f"{tag}: scene + omission effects")
        ax.legend(fontsize=7)
    for ax, tag in zip(axs[1], ARCH):
        pts = [(c["J_obs"], c["J_pred"]) for split in ("train", "validation", "holdout") for ev in runs[tag]["eval"].get(split, [])
               for c in ev["cands"] if c["J_obs"] is not None and c["m"] > 0]
        if pts:
            o, p = np.array(pts).T
            ax.scatter(o, p)
            ax.plot([0, max(o.max(), p.max())], [0, max(o.max(), p.max())], "k--", lw=0.8)
        ax.set(xlabel="measured first-witness J", ylabel="predicted J", title=f"{tag}: robot joint-travel cost")
    _save(f, out, "E4_effects_and_costs.png")
    f, axs = plt.subplots(2, 2, figsize=(17, 9))
    for row, tag in enumerate(("B", "C")):
        ev = next(e for e in runs[tag]["eval"]["train"] if any(c["m"] == 2 and c.get("slots") for c in e["cands"]))
        c = next(c for c in ev["cands"] if c["m"] == 2)
        ax = axs[row, 0]
        ax.bar(range(len(c["slots"])), c["marg"], color=CLR[tag])
        for i, z in enumerate(c["z_obs"]):
            ax.scatter([i], [z if z is not None else np.nan], color="k", marker="x", s=60, zorder=3)
        ax.set_xticks(range(len(c["slots"])), c["slots"], fontsize=7)
        ax.set(ylim=(0, 1.05), title=f"{tag}: exact marginals p(z_k=1 | r) for {c['query']} (x = observed label)")
        ax = axs[row, 1]
        ax.bar(range(len(c["top_states"])), [p for _, p in c["top_states"]], color=CLR[tag])
        ax.set_xticks(range(len(c["top_states"])), ["".join(map(str, z)) for z, _ in c["top_states"]], fontsize=7)
        ax.set(title=f"{tag}: top exact joint states P(z | r) (slot order as left)", ylabel="probability")
    _save(f, out, "E5_exact_bc_outcomes.png")
    f, ax = plt.subplots(figsize=(15, 5))
    mods = sorted({m for t in ARCH for m in runs[t]["grads"]})
    for i, tag in enumerate(ARCH + ("B-no-coupling", "C-no-coupling")):
        g = runs[tag]["grads"]
        ax.bar(np.arange(len(mods)) + 0.16 * i, [g.get(m, 0.0) + 1e-12 for m in mods], 0.16, label=tag, color=CLR[tag])
    ax.set_xticks(np.arange(len(mods)) + 0.4, mods, rotation=30, fontsize=8)
    ax.set(yscale="log", ylabel="gradient norm of total loss (after training)", title="gradient flow by module (cpl = B/C coupling)")
    ax.legend(fontsize=7)
    _save(f, out, "E6_gradient_flow.png")
    f, axs = plt.subplots(1, 3, figsize=(19, 5))
    for ax, tag in zip(axs, ARCH):
        s = res["sampler"][tag]
        pi = [p for _, p in s["pi"]]
        ax.bar(np.arange(len(pi)) - 0.2, pi, 0.4, label="exact Boltzmann pi")
        ax.bar(np.arange(len(pi)) + 0.2, s["chain_freq"], 0.4, label="3,000-step chain")
        ax.set(title=f"{tag}: sampler on {s['root']} (pass={s['pass']}, TV={s['tv_chain_3000_vs_exact']:.3f})", xlabel="canonical state")
        ax.legend(fontsize=7)
    _save(f, out, "E7_sampler.png")
