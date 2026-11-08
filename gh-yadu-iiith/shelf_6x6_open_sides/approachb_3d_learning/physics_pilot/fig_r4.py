"""R4 report figure from committed results_r4/{GATE,SMOKE}.json only: (a) predeclared gate, observed count / required minimum (log scale);
(b) one-seed development smoke, validation success log-loss per arm (NOT a result; no ranking claim).  python3 fig_r4.py"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = Path(__file__).resolve().parent
G, S = json.loads((H / "results_r4/GATE.json").read_text()), json.loads((H / "results_r4/SMOKE.json").read_text())
g, c = G["gate"], G["counts"]
rows = [("train X labels", c["x"]["train"], g["train_x_labels_min"]), ("train X0", c["x0"]["train"], g["train_x0_min"]),
        ("train X1", c["x1"]["train"], g["train_x1_min"]),
        ("train min X/Y cell", min(c["cells"][f"train|X{a}Y{b}"] for a in (0, 1) for b in (0, 1)), g["train_paired_cells_min_each"]),
        ("val X labels", c["x"]["val"], g["val_x_labels_min"]), ("val roots", c["roots"]["val"], g["val_roots_min"])]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.25, 1]}, facecolor="#fcfcfb")
y = range(len(rows))[::-1]
a.barh(list(y), [n / m for _, n, m in rows], color="#2a78d6", height=0.55)
a.axvline(1, color="#52514e", lw=1, ls="--"); a.set_ylim(-1.0, len(rows) - 0.5); a.text(1.06, -0.8, "required minimum", fontsize=8, color="#52514e")
for yy, (lab, n, m) in zip(y, rows):
    a.text(n / m * 1.08, yy, f"{n:,} (min {m})", va="center", fontsize=8, color="#0b0b0b")
a.set_yticks(list(y)); a.set_yticklabels([r[0] for r in rows]); a.set_xscale("log"); a.set_xlim(0.5, 200); a.set_xlabel("observed / required (log)")
a.set_facecolor("#fcfcfb"); a.grid(axis="x", color="#e6e5e0", lw=0.6); a.set_axisbelow(True)
a.set_title(f"(a) predeclared gate: PASS (all {len(G['result']['checks'])} checks; test roots used = {c['roots'].get('test', 0)})", loc="left", fontsize=10)
arms = list(S["arms"])
v = [S["arms"][k]["val"]["success_logloss"] for k in arms]
b.scatter(v, range(len(arms))[::-1], s=60, color="#2a78d6", zorder=3)
for yy, k, x in zip(range(len(arms))[::-1], arms, v):
    b.text(x, yy + 0.22, f"{x:.3f}", ha="center", fontsize=8, color="#0b0b0b")
b.set_yticks(list(range(len(arms))[::-1])); b.set_yticklabels(arms); b.set_ylim(-0.6, len(arms) - 0.3)
lo, hi = min(v), max(v); b.set_xlim(lo - 0.01, hi + 0.01); b.set_xlabel(f"validation success log-loss ({S['arms'][arms[0]]['val']['success_events']} events)")
b.set_facecolor("#fcfcfb"); b.grid(axis="x", color="#e6e5e0", lw=0.6); b.set_axisbelow(True)
b.set_title("(b) one-seed DEV smoke (pipeline check; not a result)", loc="left", fontsize=10)
fig.tight_layout(); out = H / "figures/R4_GATE_AND_SMOKE.png"; fig.savefig(out, dpi=160, facecolor="#fcfcfb"); print(out)
