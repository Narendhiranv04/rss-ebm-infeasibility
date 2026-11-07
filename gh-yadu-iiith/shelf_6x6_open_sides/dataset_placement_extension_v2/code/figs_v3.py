"""Production v3 figures (V2 palette and style).
  python figs_v3.py cohort V3_DIR OUT.png      QC provenance of the selected K0-K3 roots per class; physical queries per split x class
  python figs_v3.py finger SUMMARY.json OUT.png   finger-width sensitivity: targeted queries per source label class, unchanged vs changed
"""
import json, sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from figs3d import INK, INK2, C  # noqa: E402,F401  (rcParams + palette)

ST = [("AUTHOR_QC_PASS", "author-approved", C["s1"]), ("VACUOUS_PASS_NO_NON_TARGET", "no non-target object", C["s3"]),
      ("SIDECAR_REPRO_QC_PASS", "re-checked here (author tool)", C["s4"]), ("EXCLUDED", "excluded", C["s2"])]


def cohort(v3, out):
    v3 = Path(v3)
    qc = json.loads((v3 / "QC_ELIGIBLE_ROOTS.json").read_text())["roots"]
    aud = json.loads((v3 / "SPLIT_AND_ROSTER_AUDIT.json").read_text())["roster"]["by_split_K"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    ks = ["0", "1", "2", "3"]
    left = [0] * 4
    for key, lab, col in ST:
        n = [sum(1 for v in qc.values() if v["class"] == k and (v["status"] == key or (key == "EXCLUDED" and not v["eligible"]))) for k in ks]
        ax[0].barh([f"K{k}" for k in ks], n, left=left, color=col, height=0.6, edgecolor="white", linewidth=1.5)
        for i, x in enumerate(n):
            if x >= 25:
                ax[0].text(left[i] + x / 2, i, str(x), ha="center", va="center", color="white", fontsize=8)
        left = [a + b for a, b in zip(left, n)]
    ax[0].invert_yaxis(); ax[0].set_xlabel("selected roots"); ax[0].set_title("Fixed-object QC provenance (selected K0-K3)", loc="left", color=INK, pad=24)
    from matplotlib.patches import Patch
    shown = [(lab, col) for key, lab, col in ST if any(v["status"] == key or (key == "EXCLUDED" and not v["eligible"]) for v in qc.values())]
    ax[0].legend([Patch(color=c) for _, c in shown], [lab for lab, _ in shown], frameon=False, fontsize=7.5, ncol=3, loc="lower left",
                 bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    sp = ["train", "val", "test"]
    w = 0.2
    for j, k in enumerate(ks):
        n = [aud.get(f"{s}:K{k}", 0) for s in sp]
        ax[1].bar([i + (j - 1.5) * w for i in range(3)], n, width=w * 0.92, color=[C["s1"], C["s2"], C["s3"], C["s4"]][j], label=f"K{k}")
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(sp); ax[1].set_ylabel("physical queries")
    ax[1].set_title("First-cohort physical queries by split and class", loc="left", color=INK, pad=24)
    ax[1].legend(frameon=False, fontsize=7.5, ncol=4, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.2)
    fig.savefig(out, dpi=150)


def finger(summary, out):
    from matplotlib.patches import Patch
    s = json.loads(Path(summary).read_text())
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), constrained_layout=True, sharey=True)
    cls = sorted(set(s["targeted_comparison"]["by_source_class"]) | set(s["cohort_sample_comparison"]["by_source_class"]))
    for ax, key, ttl in ((axs[0], "targeted_comparison", "Targeted mock-V2 queries"), (axs[1], "cohort_sample_comparison", "First-cohort TRAIN sample")):
        by = s[key]["by_source_class"]
        n = [by.get(c, {"n": 0})["n"] for c in cls]
        chg = [by.get(c, {"label_changed": 0})["label_changed"] for c in cls]
        same = [a - b for a, b in zip(n, chg)]
        ax.barh(cls, same, color=C["s1"], height=0.6, edgecolor="white", linewidth=1.5)
        ax.barh(cls, chg, left=same, color=C["s2"], height=0.6, edgecolor="white", linewidth=1.5)
        for i, (a, b) in enumerate(zip(same, chg)):
            ax.text(a + b + 1, i, f"{b} changed / {a + b}", va="center", fontsize=8, color=INK2)
        ax.set_title(f"{ttl} ({s[key]['compared']} queries)", loc="left", color=INK)
        ax.set_xlabel("queries by source-exact label class"); ax.set_xlim(0, max(n) * 1.35)
    axs[0].invert_yaxis()
    fig.legend([Patch(color=C["s1"]), Patch(color=C["s2"])], ["label unchanged", "label changed"], frameon=False, fontsize=8, ncol=2,
               loc="outside upper right")
    fig.suptitle("Pre-shaped structure check (alternate identity) vs source-exact, adaptive_v1", x=0.01, ha="left", color=INK, fontsize=10)
    fig.savefig(out, dpi=150)


if __name__ == "__main__":
    {"cohort": cohort, "finger": finger}[sys.argv[1]](*sys.argv[2:4])
