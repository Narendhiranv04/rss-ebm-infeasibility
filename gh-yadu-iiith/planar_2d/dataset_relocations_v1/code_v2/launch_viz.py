"""Readiness figures from saved files only: budget curves, cost, thermal soak trace, scene-backed placement maps (root scene and
hypothetical edits), a 48 x 48 pair matrix and real witness GIFs drawn from the packed trajectories.

    python launch_viz.py all <launch_dir> <run64> <budget.json> <thermal_run> <outdir>
"""
import json, sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_check as K  # noqa: E402
import placement_teacher as T  # noqa: E402
import matplotlib  # noqa: E402
import matplotlib.ticker  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402
import viz  # noqa: E402

P = T.P
SER = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]       # reference categorical order (slots 1-5), fixed
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#dddcd7", "#fcfcfb"
plt.rcParams.update({"figure.facecolor": SURF, "axes.facecolor": SURF, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "lines.linewidth": 2})


def budget_figure(bj, out):
    b = json.loads(Path(bj).read_text())
    Bs = [int(x) for x in sorted(b["budgets"], key=int)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
    series = [("all queries", None)] + [(k, k) for k in ("near", "mid", "far", "vacated")]
    for (name, key), col, mk in zip(series, SER, "osD^v"):
        y, lo, hi = [], [], []
        for B in Bs:
            t = b["budgets"][str(B)]
            s = t if key is None else t["strata"]["dest_bin"].get(key)
            n, m = (t["pos64"], t["missed_vs_64"]) if key is None else (s["pos64"], s["missed"])
            ci = K.wilson(n - m, n)
            y.append((n - m) / n), lo.append((n - m) / n - ci[0]), hi.append(ci[1] - (n - m) / n)
        a1.errorbar(Bs, y, yerr=[lo, hi] if key is None else None, color=col, marker=mk, ms=6, lw=2.4 if key is None else 1.4, capsize=3,
                    label=f"{name} (n={b['budgets']['64']['pos64'] if key is None else b['budgets']['64']['strata']['dest_bin'][key]['pos64']})")
    a1.axhline(0.98, color=INK2, lw=1, ls=":")
    a1.text(11, 0.9805, "selection line: miss <= 2% of 64-pair positives", fontsize=8, color=INK2, va="bottom")
    a1.set(xscale="log", xticks=Bs, xticklabels=Bs, xlabel="grasp-pair budget B (first B scheduled pairs)", ylabel="share of 64-pair positives also found at B",
           title="Label recall vs the B=64 pilot reference (not physical ground truth)")
    a1.legend(fontsize=8, frameon=False, loc="lower right")
    for cov, col, mk in (("FULL", SER[0], "o"), ("FIRST_SUCCESS", SER[1], "s")):
        y = [b["budgets"][str(B)][f"cpu_{cov}"]["mean"] for B in Bs]
        a2.plot(Bs, y, color=col, marker=mk, ms=6, label=f"{cov.lower().replace('_', ' ')} coverage")
        a2.annotate(f"{y[-1]:.1f} s", (Bs[-1], y[-1]), textcoords="offset points", xytext=(6, 0), fontsize=8, color=INK2, va="center")
    a2.set(xscale="log", xticks=Bs, xticklabels=Bs, xlabel="grasp-pair budget B", ylabel="mean worker CPU-s per query (25-worker run)",
           title="Teacher cost per query", ylim=(0, None))
    for ax in (a1, a2):
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    a2.legend(fontsize=8, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def thermal_figure(run, out, S):
    m = [json.loads(x) for x in open(Path(run) / "monitor.jsonl")]
    t = np.array([x["t"] for x in m]) / 60
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(t, [x["temp_c"] for x in m], color=SER[0], lw=1.4, label="Tctl (k10temp)")
    for v, txt in ((S["temp_cancel_c"], "cancel 95"), (S["temp_pause_c"], "pause 88"), (S["temp_resume_c"], "resume 80")):
        a1.axhline(v, color=INK2, lw=1, ls="--")
        a1.text(t[-1], v + 0.3, txt, ha="right", va="bottom", fontsize=8, color=INK2)
    fr = np.array([x["frozen"] for x in m])
    for i in np.nonzero(fr)[0]:
        a1.axvspan(t[i] - (t[1] - t[0]) / 2, t[i] + (t[1] - t[0]) / 2, color=GRID, lw=0)
    a1.set(ylabel="CPU temperature (C)", title=f"25-worker 64-pair pilot soak: {t[-1]:.1f} min, peak {max(x['temp_c'] for x in m):.1f} C, "
           f"workers frozen in {fr.mean():.1%} of samples (grey)")
    a2.plot(t, [x["done"] for x in m], color=SER[0], label="queries finished")
    a2.plot(t, [x["inflight"] for x in m], color=SER[1], ls="--", label="queries in flight")
    a2.set(xlabel="minutes since start", ylabel="queries")
    a2.legend(fontsize=8, frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def scene_maps(run, bj, out, roots):
    """Per root: one panel per (context, object): destinations coloured by the 64-pair label, a ring where 8 pairs disagree."""
    rows = K.load(run)
    b = json.loads(Path(bj).read_text())
    v8 = dict(zip(b["row_ids"], [x[0] for x in b["virtual_rows"]["8/FULL"]]))
    robot, made = P.A.env()["robot"], []
    for rid in roots:
        rs = [(h, q) for h, q, _, _ in rows if h["root_id"] == rid]
        panels = list(dict.fromkeys((json.dumps(q["prefix_key"]), q["object_id"]) for _, q in rs))
        fig, axs = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 5.4), squeeze=False)
        for ax, (pk, oid) in zip(axs[0], panels):
            sub = [q for _, q in rs if json.dumps(q["prefix_key"]) == pk and q["object_id"] == oid]
            s = T.scene_for({"root": rid, "prefix": json.loads(pk)})
            viz.draw_support(ax, s, cell_ids=True)
            viz.draw_objects(ax, s)
            o = s.get(oid)
            ax.add_patch(Circle(o.xy_m, o.radius_m + .009, fc="none", ec=INK, lw=2.2, zorder=7))
            for q in sub:
                c = q["destination_cell"]
                y = q["label"]
                fc = "#bbbbbb" if y is None else SER[0] if y == 1 else SER[1]
                ax.add_patch(Rectangle(((c % 8) * .08 + .005, (c // 8) * .08 + .005), .07, .07, fc=fc, alpha=.55, ec=fc, lw=1.2,
                                       hatch="////" if y is None else None, zorder=2))
                if y is not None and v8.get(q["trial_id"]) != y:
                    ax.add_patch(Rectangle(((c % 8) * .08 + .002, (c // 8) * .08 + .002), .076, .076, fc="none", ec=INK, lw=2, ls="--", zorder=3))
            for a, cell in json.loads(pk):
                src = P.scene_from_dict(P.load_root(rid)["scene"]).get(a).xy_m
                ax.add_patch(FancyArrowPatch(src, s.lattice.center(cell), arrowstyle="-|>", mutation_scale=12, color=INK2, ls="--", lw=1.4, zorder=8))
            viz.draw_robot(ax, robot, np.asarray(s.robot.q_rad), robot.open_gap, alpha=.25, zorder=4)
            ax.set(xlim=(-.42, .7), ylim=(-.08, .92), aspect="equal", xticks=[], yticks=[])
            ax.grid(False)
            ctx = "root scene" if pk == "[]" else f"HYPOTHETICAL edit {json.loads(pk)} (dashed arrow; no robot state assumed)"
            ax.set_title(f"{oid}: {ctx}", fontsize=8, loc="left")
        h = [Rectangle((0, 0), 1, 1, fc=SER[0], alpha=.55), Rectangle((0, 0), 1, 1, fc=SER[1], alpha=.55),
             Rectangle((0, 0), 1, 1, fc="#bbbbbb", alpha=.55, hatch="////"), Rectangle((0, 0), 1, 1, fc="none", ec=INK, ls="--", lw=2)]
        fig.legend(h, ["label 1 at B=64 (verified carry-release-retreat)", "label 0 at B=64 (policy-limited)", "statically ineligible (occupied; no teacher)",
                       "B=8 label differs from B=64"], loc="lower center", ncol=4, fontsize=8, frameon=False)
        fig.suptitle(f"{rid} (train pilot root): placement queries for the ringed, already-held object; arm faint at its start pose", fontsize=10)
        fig.tight_layout(rect=(0, .07, 1, .95))
        made.append(Path(out) / f"{rid}_placements.png")
        fig.savefig(made[-1], dpi=110)
        plt.close(fig)
    return made


def mode_matrix(run, tid, out):
    for h, q, m, a in K.load(run):
        if q["trial_id"] == tid:
            break
    cmap = ListedColormap(["#d9d9d9", "#f3e7c4", "#ffffff", SER[1], SER[0], SER[2]])
    fig, ax = plt.subplots(figsize=(7.6, 7.4))
    ax.imshow(m["pair_category"], cmap=cmap, vmin=-.5, vmax=5.5, interpolation="nearest")
    for i, j, r in zip(a["a_pick"], a["a_place"], a["a_rank"]):
        if 0 <= r < 64:
            ax.text(j, i, str(r + 1), ha="center", va="center", fontsize=4.5, color=INK)
    ax.grid(False)
    ax.set(xlabel="place mode (24 tool axes x 2 elbows)", ylabel="held mode (24 tool axes x 2 elbows)")
    ax.set_title(f"{h['root_id']} {q['object_id']} -> cell {q['destination_cell']} ({q['block']}): 2,304 pairs; numbers = schedule rank of planned pairs",
                 fontsize=8, loc="left")
    fig.legend([Rectangle((0, 0), 1, 1, fc=c, ec="#999") for c in cmap.colors], [P.CAT[k] for k in P.CAT], loc="lower center", ncol=3, fontsize=7, frameon=False)
    fig.tight_layout(rect=(0, .07, 1, 1))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def witness_gif(run, rid, q, wid, out, n=40):
    robot, core = P.A.env()["robot"], T.witness(run, rid, wid)
    s = T.scene_for({"root": rid, "prefix": q["prefix_key"]})
    oid, rad, frames, strip = q["object_id"], s.get(q["object_id"]).radius_m, [], []
    for k, (qq, gap, phase) in enumerate(P.mo.samples(core, n)):
        fig, ax = plt.subplots(figsize=(5.4, 5.0))
        viz.draw_support(ax, s, cell_ids=False)
        for o in s.objects:
            if o.id != oid:
                ax.add_patch(Circle(o.xy_m, o.radius_m, fc=viz.COLORS["target" if o.kind == "target" else "obstacle"], ec="k", lw=.5, zorder=3))
        held = (oid, rad, tuple(core["held_local"])) if phase == "carry" else None
        viz.draw_robot(ax, robot, qq, gap, held=held)
        ax.add_patch(Circle(P.ge.tool_point(robot, qq, core["held_local"]) if held else core["place_xy"], rad, fc=SER[0], ec="k", zorder=9))
        c = q["destination_cell"]
        ax.add_patch(Rectangle(((c % 8) * .08, (c // 8) * .08), .08, .08, fc="none", ec=SER[2], lw=2, zorder=2))
        ax.set(xlim=(-.42, .7), ylim=(-.08, .92), aspect="equal", xticks=[], yticks=[])
        ax.grid(False)
        ctx = "root scene" if not q["prefix_key"] else f"hypothetical edit {q['prefix_key']}"
        ax.set_title(f"{oid} held at cell {q['source_cell']} -> cell {c} ({ctx})   phase: {phase}\n"
                     f"s = {k / (n - 1):.2f} of the joint path (untimed waypoints); witness {wid[:10]}", fontsize=7.5, loc="left")
        fig.tight_layout()
        fig.canvas.draw()
        img = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3])
        frames.append(img)
        if k in np.linspace(0, n - 1, 6).astype(int):
            strip.append(img)
        plt.close(fig)
    frames[0].save(out, save_all=True, append_images=frames[1:] + [frames[-1]] * 8, duration=120, loop=0)
    sheet = Image.new("RGB", (strip[0].size[0] * len(strip), strip[0].size[1]), "white")
    for i, im in enumerate(strip):
        sheet.paste(im, (i * strip[0].size[0], 0))
    sheet.save(Path(out).with_name(Path(out).stem + "_filmstrip.png"))


def all_figures(L, run64, bj, therm, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    budget_figure(bj, out / "budget_recall_and_cost.png")
    thermal_figure(therm, out / "thermal_soak_25_workers.png", json.loads((Path(L) / "configs" / "launch_config.json").read_text())["safety"])
    rows = K.load(run64)
    b = json.loads(Path(bj).read_text())
    flips = [rid for rid, (y8, y64) in zip([m[0] for m in b["row_meta"]], zip([x[0] for x in b["virtual_rows"]["8/FULL"]], [x[0] for x in b["virtual_rows"]["64/FULL"]])) if y8 != y64]
    neg = [h["root_id"] for h, q, _, _ in rows if q["label"] == 0]
    cls = {}
    for h, _, _, _ in rows:
        cls.setdefault(h["privileged_teacher"]["verified_class"], h["root_id"])
    roots = list(dict.fromkeys(flips[:1] + neg[:1] + list(cls.values())[:2]))[:4]
    scene_maps(run64, bj, out, roots)
    q0 = max(((h, q) for h, q, _, _ in rows if q["eligible"] and q["completed"] and q["mode_counts"]["motion_attempted"] >= 32),
             key=lambda hq: (hq[1]["mode_counts"]["attempted_failure"] > 0 and hq[1]["label"] == 1, hq[1]["mode_counts"]["attempted_failure"]))
    mode_matrix(run64, q0[1]["trial_id"], out / f"{q0[1]['trial_id']}_mode_pairs.png")
    for blk in ("root", "prefix1"):
        h, q = next((h, q) for h, q, _, a in rows if q["label"] == 1 and q["block"] == blk and q["witness_refs"] and a["a_method"][0] > 1)
        witness_gif(run64, h["root_id"], q, q["witness_refs"][0], out / f"{q['trial_id']}_{blk}_witness.gif")
    return sorted(str(p) for p in out.iterdir())


if __name__ == "__main__":
    print("\n".join(all_figures(*sys.argv[2:7])))
