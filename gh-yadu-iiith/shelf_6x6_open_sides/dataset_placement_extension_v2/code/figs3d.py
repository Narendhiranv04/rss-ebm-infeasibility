"""V2 figures (MUJOCO_GL=egl for the 3-D stills).

  scenes MOCK_V2 ROOT_ID OUT.png [RUN]     root / 1-edit / 2-edit layouts (schematic + MuJoCo render), queries coloured by a V2 run's label
  plots ANALYSIS.json MOCK_V2 OUT_DIR      policy misses by difficult stratum (Wilson 95%), cost vs miss, branch effects, worker sweep, resources
"""
import gzip, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"1": "#2a78d6", "0": "#eb6834", "None": "#8a8984", "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a", "s4": "#8a5fd1"}
PC = {"adaptive_v1": "#2a78d6", "fixed_4x1500": "#eb6834", "fixed_4x5000": "#1baf7a", "fixed_8x5000": "#8a5fd1"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def _roster(mock):
    return [json.loads(x) for x in gzip.open(Path(mock) / "roster_v2.jsonl.gz", "rt")]


def _labels(mock, run):
    out = {}
    for f in (Path(mock) / "runs" / run / "units").glob("*.json"):
        for r in json.loads(f.read_text())["records"]:
            out[r["qid"]] = r
    return out


def scenes(mock, rid, out, run="adaptive_v1_A"):
    import placement3d as P
    import replay3d as R
    rows = [q for q in _roster(mock) if q["root_id"] == rid]
    lab = _labels(mock, run)
    raw = json.loads((P.SRC / "raw/roots" / f"{rid}.json").read_text())
    shape = {o["id"]: (o["shape"], o["size"]) for o in raw["objects"]}
    by = {}
    for q in rows:
        by.setdefault(q["depth"], q)
    fig, ax = plt.subplots(2, 3, figsize=(13, 9.2), gridspec_kw={"height_ratios": [1, 0.8]}, constrained_layout=True)
    for d in range(3):
        a = ax[0, d]
        if d not in by:
            a.axis("off"); continue
        cells = by[d]["cells"]
        a.set_xlim(-0.5, 5.5); a.set_ylim(-0.5, 5.5); a.set_aspect("equal"); a.set_xticks(range(6)); a.set_yticks(range(6))
        a.set_xlabel("column"); a.set_ylabel("row (0 = nearest the robot)")
        gr, gc = raw["goal"]
        a.add_patch(plt.Rectangle((gc - 0.45, gr - 0.45), 0.9, 0.9, fill=False, ls="--", ec=INK2, lw=1))
        a.text(gc, gr - 0.38, "T goal", ha="center", fontsize=7, color=INK2)
        for o, (r, c) in cells.items():
            s, z = shape[o]
            rad = z / 0.075 * 1.2
            pat = plt.Rectangle((c - rad, r - rad), 2 * rad, 2 * rad) if s == "box" else plt.Circle((c, r), rad)
            pat.set(fc="#d6d4cf" if o != "t0" else "#bcd6f3", ec=INK, lw=0.8); a.add_patch(pat)
            a.text(c, r, "T" if o == "t0" else o[1:], ha="center", va="center", fontsize=7, color=INK)
        for e, new in by[d]["edits"].items():
            old = raw["initial_cells"][e] if d == 1 else next(q for q in rows if q["depth"] == 1)["cells"].get(e, raw["initial_cells"][e])
            if d == 2 and e in by[1]["edits"]:
                old = raw["initial_cells"][e]
            a.annotate("", xy=(new[1], new[0]), xytext=(old[1], old[0]), arrowprops=dict(arrowstyle="->", color=INK2, ls=":", lw=1))
        for q in rows:
            if q["depth"] != d or q["static"]:
                continue
            L = str(lab.get(q["qid"], {}).get("label"))
            a.annotate("", xy=(q["dest"][1], q["dest"][0]), xytext=(q["src"][1], q["src"][0]),
                       arrowprops=dict(arrowstyle="-|>", color=C[L], lw=1.6, shrinkA=6, shrinkB=2))
        a.set_title(["root scene", "1-edit hypothetical", "2-edit hypothetical"][d] + f"  ({by[d]['scene_hash'][:8]})", fontsize=9, color=INK)
    handles = [plt.Line2D([], [], color=C[k], lw=2, label=t) for k, t in (("1", "label 1 (verified)"), ("0", "label 0"), ("None", "unlabeled"))]
    handles.append(plt.Line2D([], [], color=INK2, lw=1, ls=":", label="hypothetical edit"))
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    for d in range(3):
        a = ax[1, d]; a.axis("off")
        if d in by:
            sc = P.Scene(raw["objects"], by[d]["cells"])
            a.imshow(R.render(sc, 640, 480))
            a.set_title("MuJoCo scene (settled)", fontsize=8, color=INK2)
    fig.suptitle(f"SHELF66V3 placement queries over one lineage, root {rid} (K{rows[0]['K']}); arrows = query object -> destination, coloured by the V2 label (" + run + ")",
                 color=INK, fontsize=11)
    fig.savefig(out, dpi=120); print(out)


def plots(an, mock, od):
    an, od = json.loads(Path(an).read_text()), Path(od)
    pt = an["policy_table"]
    names = [n for n in ("adaptive_v1", "fixed_4x1500", "fixed_4x5000", "fixed_8x5000") if n in pt]
    cells = [("overall", "all", "all"), ("band", "far", "far band"), ("depth", "2", "2-edit"), ("between", "3+", "3+ between"),
             ("cohort", "holdout", "holdout"), ("shape", "box", "box"), ("shape", "cylinder", "cylinder"), ("dest_is_t_goal", "True", "T-goal cell")]
    fig, a = plt.subplots(figsize=(10.5, 4.4))
    w = 0.8 / len(names)
    for j, n in enumerate(names):
        xs, ys, lo, hi = [], [], [], []
        for i, (s, k, _) in enumerate(cells):
            d = pt[n]["strata"][s].get(k)
            if not d:
                continue
            xs.append(i + (j - (len(names) - 1) / 2) * w); ys.append(d["rate"]); lo.append(d["rate"] - d["wilson95"][0]); hi.append(d["wilson95"][1] - d["rate"])
        a.bar(xs, ys, width=w * 0.9, color=PC[n], label=n, zorder=2)
        a.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=INK2, lw=0.8, capsize=2, zorder=3)
        for x, y in zip(xs, ys):
            if y == 0:
                a.text(x, 0.002, "0", ha="center", va="bottom", fontsize=6.5, color=PC[n], fontweight="bold")
    a.axhline(0.10, color=INK2, ls="--", lw=1); a.text(len(cells) - 0.5, 0.103, "stratum cap 10%", ha="right", fontsize=7, color=INK2)
    a.axhline(0.05, color=INK2, ls=":", lw=1); a.text(len(cells) - 0.5, 0.053, "overall cap 5%", ha="right", fontsize=7, color=INK2)
    a.set_xticks(range(len(cells))); a.set_xticklabels([f"{c[2]}\n(n={pt[names[0]]['strata'][c[0]].get(c[1], {}).get('ref_pos', 0)})" for c in cells], fontsize=8)
    a.set_ylabel("missed reference positives"); a.legend(frameon=False, ncol=4, fontsize=8, loc="upper left")
    a.set_title("Real 12-worker runs vs reference_v2 (all units x all eligible branches, 5000 it.): missed positives by stratum, Wilson 95%", fontsize=9)
    fig.tight_layout(); fig.savefig(od / "policy_misses_by_stratum.png", dpi=140); plt.close(fig)

    fig, a = plt.subplots(figsize=(6.4, 4.0))
    for n in names:
        r = pt[n]
        a.errorbar(r["mean_cpu_s_per_query"], r["miss_rate"], yerr=[[r["miss_rate"] - r["wilson95"][0]], [r["wilson95"][1] - r["miss_rate"]]], fmt="o",
                   color=PC[n], ms=6, lw=1, capsize=2)
        a.annotate(f"{n}  ({r['missed']}/{r['ref_positives']})", (r["mean_cpu_s_per_query"], r["miss_rate"]), textcoords="offset points", xytext=(6, 4), fontsize=7.5)
    a.axhline(0.05, color=INK2, ls=":", lw=1)
    a.set_xlabel("measured mean CPU-s per query (real run, 12 workers)"); a.set_ylabel("overall missed-positive rate")
    a.set_title(f"Cost vs recall; choice: {an['policy_choice']['choice']}", fontsize=9)
    fig.tight_layout(); fig.savefig(od / "policy_cost_vs_miss.png", dpi=140); plt.close(fig)

    be = an["branch_effects"]
    keys = [("units_rescued_after_hold_failure", "after hold failure"), ("units_rescued_after_exec_failure", "after execution failure"),
            ("units_rescued_after_plan_failure", "after plan failure (V1 too)"), ("units_first_eligible_hold_failed_later_held", "branch-0 hold failed,\nlater branch held")]
    runs = [r for r in ("reference_v2", "adaptive_v1_A") if r in be]
    fig, a = plt.subplots(figsize=(8.2, 3.6))
    w = 0.38
    for j, r in enumerate(runs):
        ys = [be[r][k] for k, _ in keys]
        xs = [i + (j - 0.5) * w for i in range(len(keys))]
        a.bar(xs, ys, width=w * 0.92, color=[C["s1"], C["s3"]][j], label=r, zorder=2)
        for x, y in zip(xs, ys):
            a.text(x, y, str(y), ha="center", va="bottom", fontsize=7)
    a.set_xticks(range(len(keys))); a.set_xticklabels([t for _, t in keys], fontsize=8); a.set_ylabel("units (summed over queries)")
    a.legend(frameon=False, fontsize=8); a.set_title("Units verified on a later IK branch, by the earlier branch's failure (V2 branch semantics)", fontsize=9)
    fig.tight_layout(); fig.savefig(od / "branch_effects.png", dpi=140); plt.close(fig)

    sw = an.get("sweep", {})
    if sw:
        ks = sorted(sw, key=int)
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        ax[0].plot([int(k) for k in ks], [sw[k]["q_per_wall_s"] for k in ks], "-o", color=C["s1"], lw=2, ms=6)
        for k in ks:
            ax[0].annotate(f"{sw[k]['q_per_wall_s']:.3f}", (int(k), sw[k]["q_per_wall_s"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax[0].set_xticks([int(k) for k in ks]); ax[0].set_xlabel("workers (pinned, 1 thread each)"); ax[0].set_ylabel("queries per wall-second")
        ax[0].set_title("Throughput (depth-0 sweep subset, unit = query)", fontsize=9)
        ax[1].bar(ks, [sw[k]["cpu_per_query"] for k in ks], color=C["s1"], width=0.55)
        ax[1].set_xlabel("workers"); ax[1].set_ylabel("CPU-s per query"); ax[1].set_title("Per-query CPU (SMT siblings above 12)", fontsize=9)
        fig.tight_layout(); fig.savefig(od / "worker_sweep.png", dpi=140); plt.close(fig)

    for run in ("reference_v2", "adaptive_v1_A"):
        p = Path(mock) / "runs" / run / "monitor.jsonl"
        if not p.exists():
            continue
        m = [json.loads(x) for x in p.read_text().splitlines()]
        t = [(x["t"] - m[0]["t"]) / 60 for x in m]
        fig, ax = plt.subplots(3, 1, figsize=(8, 5.6), sharex=True)
        ax[0].plot(t, [x["cpu_pct_sum"] / 100 for x in m], color=C["s1"], lw=1.2); ax[0].set_ylabel("busy CPUs (own tree)"); ax[0].axhline(18, color=INK2, ls="--", lw=1)
        ax[0].text(t[-1], 18.3, "ceiling 18", ha="right", fontsize=7, color=INK2)
        ax[1].plot(t, [x["pkg_temp_c"] for x in m], color=C["s2"], lw=1.2); ax[1].set_ylabel("package C"); ax[1].axhline(90, color=INK2, ls="--", lw=1)
        ax[2].plot(t, [x["rss_max_gb"] for x in m], color=C["s3"], lw=1.2); ax[2].set_ylabel("max RSS GB"); ax[2].set_xlabel("minutes")
        ax[0].set_title(f"{run}: own process tree (monitor, 2 s)", fontsize=9)
        fig.tight_layout(); fig.savefig(od / f"resources_{run}.png", dpi=130); plt.close(fig)
    print("plots ->", od)


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "scenes":
        scenes(*sys.argv[2:6])
    elif c == "plots":
        plots(*sys.argv[2:5])
