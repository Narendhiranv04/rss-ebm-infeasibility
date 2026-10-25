"""Report of a long-budget experiment: success within 60 / 120 / 300 / 600 / 900 s, per environment,
difficulty and method, anytime curves, and an HTML gallery of the scene and solution GIFs.

    python scripts/overnight_report.py workspace/experiments/<id>_overnight-900s --out results/overnight

A planner's search does not depend on its time budget (the budget only stops it), so a run
given 900 s that succeeds after t seconds of planning is exactly what a run given any budget
>= t returns: one long run per problem yields the success rate for every shorter budget.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import mujoco  # noqa: F401  (before pandas: otherwise its DLL fails to load on Windows)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mamo.bench import report, runner  # noqa: E402

METHODS = ["m4m", "em4m", "im4m", "im4m_pnp", "lazy_im4m", "lazy_im4m_pnp", "dogar", "dogar_uniform"]
LABEL = {"m4m": "M4M", "em4m": "E-M4M", "im4m": "I-M4M", "im4m_pnp": "I-M4M-PnP", "lazy_im4m": "Lazy I-M4M",
         "lazy_im4m_pnp": "Lazy I-M4M-PnP", "dogar": "DOGAR", "dogar_uniform": "DOGAR (uniform)"}
# categorical slots of the reference palette (dataviz skill), fixed order
COLOR = dict(zip(METHODS, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]))
ENVS = ["reference_shelf", "open_tabletop", "confined_shelf", "multi_level_shelf", "access19", "stacking"]
ENV_LABEL = {"reference_shelf": "Reference shelf (M4M)", "open_tabletop": "Open tabletop",
             "confined_shelf": "Confined shelf", "multi_level_shelf": "Multi-level shelf", "access19": "Access-19",
             "stacking": "3D stacking"}
BUDGETS = [60, 120, 300, 600, 900]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e2dc"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        out.append("| " + " | ".join([str(idx)] + [str(v) for v in row.values]) + " |")
    return "\n".join(out)


def load(exp: Path) -> pd.DataFrame:
    df = report.frame(runner.read_results(exp)).drop_duplicates("id", keep="last")
    df["t"] = df["planning_time"].astype(float)
    df["ok"] = df["success"].astype(bool)
    return df


def within(df: pd.DataFrame, t: float) -> pd.Series:
    return df["ok"] & (df["t"] <= t)


def tables(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    methods = [m for m in METHODS if m in set(df.method)]
    envs = [e for e in ENVS if e in set(df.env)]
    n = df.groupby("method").size()
    t = {}
    # overall success at each budget
    t["overall"] = pd.DataFrame({f"<= {b} s": [int(within(df[df.method == m], b).sum()) for m in methods]
                                 for b in BUDGETS}, index=[LABEL[m] for m in methods])
    t["overall"].insert(0, "runs", [int(n[m]) for m in methods])
    t["overall"].index.name = "method"
    # per environment: 300 s -> 900 s
    rows = {}
    for e in envs:
        g = df[df.env == e]
        rows[ENV_LABEL[e]] = [f"{int(within(g[g.method == m], 300).sum())} -> {int(within(g[g.method == m], 900).sum())}"
                              f" / {int((g.method == m).sum())}" for m in methods]
    t["env"] = pd.DataFrame(rows, index=[LABEL[m] for m in methods]).T
    t["env"].index.name = "environment (300 s -> 900 s)"
    # per difficulty
    rows = {}
    for d in ["easy", "medium", "hard"]:
        g = df[df.difficulty == d]
        rows[d] = [f"{int(within(g[g.method == m], 300).sum())} -> {int(within(g[g.method == m], 900).sum())}"
                   f" / {int((g.method == m).sum())}" for m in methods]
    t["difficulty"] = pd.DataFrame(rows, index=[LABEL[m] for m in methods]).T
    t["difficulty"].index.name = "difficulty (300 s -> 900 s)"
    # times of solved runs
    s = df[df.ok]
    t["time"] = pd.DataFrame({"solved": [int((s.method == m).sum()) for m in methods],
                              "median planning time (s)": [f"{s[s.method == m].t.median():.0f}" if (s.method == m).any()
                                                           else "-" for m in methods],
                              "solved after 300 s": [int(((s.method == m) & (s.t > 300)).sum()) for m in methods]},
                             index=[LABEL[m] for m in methods])
    t["time"].index.name = "method"
    # why runs failed
    f = df[~df.ok].copy()
    f["why"] = np.where(f["solved"], "plan failed on replay",
                        f["reason"].fillna("").str.replace(r"\(.*", "", regex=True).str.strip().str.slice(0, 40))
    t["fail"] = f.groupby(["why", "method"]).size().unstack("method").fillna(0).astype(int)
    t["fail"] = t["fail"][[m for m in methods if m in t["fail"].columns]].rename(columns=LABEL)
    t["fail"].index.name = "unsolved: reason"
    return t


def _style(ax) -> None:
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, out: Path, name: str) -> None:
    from PIL import Image
    fig.savefig(out / f"{name}.png", dpi=120, facecolor="#fcfcfb")
    Image.open(out / f"{name}.png").convert("RGB").save(out / f"{name}.gif")
    plt.close(fig)


def plots(df: pd.DataFrame, out: Path) -> None:
    methods = [m for m in METHODS if m in set(df.method)]
    envs = [e for e in ENVS if e in set(df.env)]
    grid = np.linspace(0, 900, 181)
    # 1. anytime curves per environment (small multiples)
    ncol = 3
    nrow = int(np.ceil(len(envs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.6 * nrow + 0.8), sharey=True, squeeze=False)
    for k, e in enumerate(envs):
        ax = axes[k // ncol][k % ncol]
        g = df[df.env == e]
        for m in methods:
            gm = g[g.method == m]
            if gm.empty:
                continue
            y = [within(gm, x).sum() / len(gm) for x in grid]
            ax.step(grid, y, where="post", color=COLOR[m], lw=2, label=LABEL[m])
        ax.axvline(300, color=MUTED, ls="--", lw=1)
        ax.text(305, 1.02, "300 s (papers)", fontsize=7, color=MUTED, va="bottom")
        ax.set_title(ENV_LABEL[e], fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 900)
        ax.set_ylim(0, 1.08)
        ax.set_xticks([0, 300, 600, 900])
        _style(ax)
        if k % ncol == 0:
            ax.set_ylabel("problems solved (fraction)", fontsize=8, color=MUTED)
        if k // ncol == nrow - 1:
            ax.set_xlabel("planning time (s)", fontsize=8, color=MUTED)
    for k in range(len(envs), nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(methods), fontsize=8, frameon=False)
    fig.suptitle("Solved (and replayed) within t seconds of planning, per environment", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    save(fig, out, "anytime_by_environment")

    # 2. dumbbell: solved at 300 s vs 900 s per method
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ys = np.arange(len(methods))[::-1]
    for y, m in zip(ys, methods):
        g = df[df.method == m]
        a, b = int(within(g, 300).sum()), int(within(g, 900).sum())
        ax.plot([a, b], [y, y], color=GRID, lw=3, zorder=1)
        ax.scatter([a], [y], s=60, facecolor="#fcfcfb", edgecolor=COLOR[m], lw=2, zorder=2)
        ax.scatter([b], [y], s=60, color=COLOR[m], edgecolor="#fcfcfb", lw=2, zorder=3)
        ax.text(b + 0.4, y, f"{a} -> {b}", va="center", fontsize=8, color=INK)
    ax.set_yticks(ys, [LABEL[m] for m in methods])
    n_max = int(df.groupby("method").size().max())
    ax.set_xlim(0, n_max + 2)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_xlabel(f"problems solved and replayed (of {int(df.groupby('method').size().max())})", fontsize=8,
                  color=MUTED)
    ax.set_title("Budget 300 s (open circle) vs 900 s (filled)", fontsize=11, color=INK, loc="left")
    _style(ax)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    save(fig, out, "budget_300_vs_900")

    # 3. anytime per difficulty (all environments)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for ax, d in zip(axes, ["easy", "medium", "hard"]):
        g = df[df.difficulty == d]
        for m in methods:
            gm = g[g.method == m]
            if gm.empty:
                continue
            ax.step(grid, [within(gm, x).sum() / len(gm) for x in grid], where="post", color=COLOR[m], lw=2,
                    label=LABEL[m])
        ax.axvline(300, color=MUTED, ls="--", lw=1)
        ax.set_title(d, fontsize=10, color=INK, loc="left")
        ax.set_xlim(0, 900)
        ax.set_ylim(0, 1.05)
        ax.set_xticks([0, 300, 600, 900])
        ax.set_xlabel("planning time (s)", fontsize=8, color=MUTED)
        _style(ax)
    axes[0].set_ylabel("problems solved (fraction)", fontsize=8, color=MUTED)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(methods), fontsize=8, frameon=False)
    fig.suptitle("Solved within t seconds, per difficulty (all environments)", fontsize=12, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    save(fig, out, "anytime_by_difficulty")


ENV_TEXT = {
    "reference_shelf": "The M4M / E-M4M papers' shelf (reference scenes, 5-15 movables).",
    "open_tabletop": "40x28 cm table, 10x7 cells of 40 mm, thin / medium / thick cylinders.",
    "confined_shelf": "Cubicle 79x44x34 cm open at the front, 9x4 cells of 70 mm, bottle-sized cylinders.",
    "multi_level_shelf": "Open shelf: floor bays and two decks 24 cm apart, boxes 6x6x9 cm.",
    "access19": "Dense bookcase, 5x7 cells per level, boxes 3x3x10 cm in a fixed pattern; OoI deep inside.",
    "stacking": "Parts table in front of the arm (stack table behind), 10x10 cells of 30 mm, blocks grasped from above.",
}


def gallery(df: pd.DataFrame, t: dict, out: Path, exp: Path) -> None:
    sc_f, so_f = out / "scenes" / "index.json", out / "solutions" / "index.json"
    scenes = json.loads(sc_f.read_text(encoding="utf-8")) if sc_f.exists() else []
    sols = json.loads(so_f.read_text(encoding="utf-8")) if so_f.exists() else []
    order = {"easy": 0, "medium": 1, "hard": 2}

    def tbl(d: pd.DataFrame) -> str:
        return d.to_html(classes="t", border=0, escape=True)

    parts = [f"""<!doctype html><meta charset="utf-8"><title>Overnight benchmark</title>
<style>
body{{font:14px system-ui,sans-serif;margin:0 auto;max-width:1400px;padding:16px;color:#0b0b0b;background:#fcfcfb}}
h1,h2,h3{{font-weight:600}} .muted{{color:#52514e}}
table.t{{border-collapse:collapse;margin:8px 0 16px;font-size:13px}} .t th,.t td{{padding:4px 10px;border-bottom:1px solid #e4e2dc;text-align:right}}
.t th:first-child,.t td:first-child{{text-align:left}}
.grid{{display:flex;flex-wrap:wrap;gap:12px}} figure{{margin:0;max-width:100%}} figure img{{max-width:100%;border:1px solid #e4e2dc}}
figcaption{{font-size:12px;color:#52514e;max-width:560px}}
</style>
<h1>Overnight benchmark: OoI retrieval in six environments, 900 s budget</h1>
<p class="muted">Experiment <code>{html.escape(exp.name)}</code>. Every method on every problem with 900 s of planning
(3x the papers' 300 s); a solved plan counts only if it also retrieves the OoI when replayed from the initial
scene. Success within shorter budgets comes from the same runs (the search does not depend on the budget).</p>
<h2>Success within each budget</h2>{tbl(t["overall"])}
<img src="budget_300_vs_900.png" style="max-width:720px;width:100%">
<h2>Per environment (300 s -> 900 s)</h2>{tbl(t["env"])}
<img src="anytime_by_environment.png" style="width:100%">
<h2>Per difficulty (300 s -> 900 s)</h2>{tbl(t["difficulty"])}
<img src="anytime_by_difficulty.png" style="width:100%">
<h2>Planning time and failures</h2>{tbl(t["time"])}{tbl(t["fail"])}"""]
    for e in [x for x in ENVS if x in set(df.env)]:
        parts.append(f"<h2>{ENV_LABEL[e]}</h2><p class='muted'>{html.escape(ENV_TEXT.get(e, ''))}</p><h3>Problems "
                     f"(task: retrieve the yellow OoI)</h3><div class='grid'>")
        for s in sorted([s for s in scenes if s["env"] == e], key=lambda s: (order.get(s["difficulty"], 9), s["scene"])):
            g = df[(df.env == e) & (df.scene == s["scene"])]
            solved = (", ".join(f"{LABEL[r.method]} {r.t:.0f} s" for r in g[g.ok].sort_values("t").itertuples())
                      or "none") if len(g) else "not in the benchmark sample"
            blk = s.get("blockers")
            cap = (f"<b>{html.escape(Path(s['scene']).stem)}</b> ({s['difficulty']}): retrieve the OoI; "
                   f"{len(blk) if blk is not None else '?'} movable(s) in its retrieval path. Solved by: {html.escape(solved)}")
            parts.append(f"<figure><img src='{s['gif']}' loading='lazy'><figcaption>{cap}</figcaption></figure>")
            if s.get("png2d"):
                parts.append(f"<figure><img src='{s['png2d']}' loading='lazy' style='max-width:520px'>"
                             f"<figcaption>top view (purple/red = NGR)</figcaption></figure>")
        parts.append("</div><h3>Solutions (replays)</h3><div class='grid'>")
        for s in sorted([s for s in sols if s.get("env") == e and "error" not in s],
                        key=lambda s: (order.get(s["difficulty"], 9), s["scene"], METHODS.index(s["method"])
                                       if s["method"] in METHODS else 99)):
            ok = "replayed OK" if s.get("executed") else "replay FAILED"
            parts.append(f"<figure><img src='{s['gif']}' loading='lazy'><figcaption>{LABEL.get(s['method'], s['method'])}"
                         f" - {html.escape(Path(s['scene']).stem)} ({s['difficulty']}), planned in "
                         f"{(s.get('planning_time') or 0):.0f} s, {ok}</figcaption></figure>")
        parts.append("</div>")
    (out / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("experiment")
    ap.add_argument("--out", default="results/overnight")
    a = ap.parse_args()
    exp, out = Path(a.experiment), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = load(exp)
    df.drop(columns=["stats", "error"], errors="ignore").to_csv(out / "results.csv", index=False)
    t = tables(df)
    plots(df, out)
    status = runner.read_status(exp)
    md = [f"# Overnight benchmark ({exp.name})", "",
          f"{len(df)} runs ({status.get('state')}, {status.get('done')}/{status.get('total')}); budget 900 s; "
          "solved = planned and replayed successfully. Success within shorter budgets is read off the same runs.", "",
          "## Success within each budget", "", md_table(t["overall"]), "",
          "## Per environment (solved within 300 s -> within 900 s / runs)", "", md_table(t["env"]), "",
          "## Per difficulty", "", md_table(t["difficulty"]), "",
          "## Planning time", "", md_table(t["time"]), "", "## Unsolved runs", "", md_table(t["fail"]), "",
          "Plots: anytime_by_environment.png, anytime_by_difficulty.png, budget_300_vs_900.png (also .gif). "
          "Gallery: index.html (scene GIFs with their task, solution GIFs)."]
    (out / "report.md").write_text("\n".join(md), encoding="utf-8")
    gallery(df, t, out, exp)
    print((out / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
