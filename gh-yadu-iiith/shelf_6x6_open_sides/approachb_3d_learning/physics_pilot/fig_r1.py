"""R1 report figure from the committed ledger only (results_r1/R1_BUDGET_LEDGER.json): (a) per-chunk verified outcomes, (b) cumulative
measured CPU-h against the 60 CPU-h cap.  python3 fig_r1.py"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
H = Path(__file__).resolve().parent
L = json.loads((H / "results_r1/R1_BUDGET_LEDGER.json").read_text())
ch = sorted(L["chunks"], key=lambda c: c["chunk"])
order = [c["chunk"] for c in L["chunks"]]                     # execution order (for the CPU curve)
K = [("X1_VERIFIED", "X1 verified", "#2a78d6"), ("Y1_REPLAY_VERIFIED", "Y1 replay-verified", "#eb6834"),
     ("X0_RELOCATION_FAILED_RECONFIRMED", "X0 relocation (reconfirmed)", "#1baf7a"), ("X0_STOP_CONNECTOR_NO_PLAN", "X0 stop-connector", "#eda100"),
     ("MISSING_RELOCATION_MISMATCH_WITH_RECORD", "MISSING (record mismatch)", "#e87ba4")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
                     "xtick.color": "#52514e", "ytick.color": "#52514e"})
fig, (a, b) = plt.subplots(2, 1, figsize=(10, 6.4), gridspec_kw={"height_ratios": [1.6, 1]}, facecolor="#fcfcfb")
x = [int(c["chunk"][5:]) for c in ch]
bot = [0] * len(ch)
for k, lab, col in K:
    v = [c["finals"].get(k, 0) for c in ch]
    a.bar(x, v, bottom=bot, color=col, width=0.8, label=f"{lab}  ({sum(v):,})", edgecolor="#fcfcfb", linewidth=0.6)
    bot = [p + q for p, q in zip(bot, v)]
a.set_xlim(0.3, 52.7); a.set_ylabel("items per chunk"); a.set_facecolor("#fcfcfb"); a.grid(axis="y", color="#e6e5e0", lw=0.6); a.set_axisbelow(True)
a.set_title("(a) R1 verified outcomes per chunk (chunks 1-52; 9,988 items; 0 blocking discrepancies)", loc="left", fontsize=10, color="#0b0b0b")
a.legend(ncol=3, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.08))
a.set_ylim(0, max(bot) * 1.13)
a.text(4.5, max(bot) * 1.05, "gvlab (CPUs 16-17)", ha="center", fontsize=8, color="#52514e")
a.text(30, max(bot) * 1.05, "Blackwell (12 workers; E-cores 8-19 from CHUNK011)", ha="center", fontsize=8, color="#52514e")
a.axvline(8.5, color="#52514e", lw=0.8, ls=":")
cum = {c["chunk"]: c["cpu_h_cumulative"] for c in L["chunks"]}
y = [cum[c] for c in order]
b.plot(range(1, len(y) + 1), y, color="#2a78d6", lw=2, marker="o", ms=3)
b.axhline(L["cap_cpu_h"], color="#52514e", lw=1, ls="--"); b.text(1, L["cap_cpu_h"] - 4, f"authorized cap {L['cap_cpu_h']:.0f} CPU-h", fontsize=8, color="#52514e")
b.text(len(y), y[-1] + 2.5, f"{y[-1]:.2f} CPU-h", ha="right", fontsize=8, color="#0b0b0b")
b.set_xlim(0.3, 52.7); b.set_ylim(0, L["cap_cpu_h"] * 1.08); b.set_xlabel("chunks completed (execution order)"); b.set_ylabel("cumulative CPU-h")
b.set_facecolor("#fcfcfb"); b.grid(axis="y", color="#e6e5e0", lw=0.6); b.set_axisbelow(True)
b.set_title("(b) measured process CPU (all items incl. retries, both hosts) vs. cap", loc="left", fontsize=10, color="#0b0b0b")
fig.tight_layout(); out = H / "figures/R1_OUTCOMES_AND_BUDGET.png"; fig.savefig(out, dpi=160, facecolor="#fcfcfb"); print(out)
