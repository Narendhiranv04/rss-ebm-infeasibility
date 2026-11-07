"""V2 mock evidence from saved files only (no planner call): adaptive-vs-64 tables, strata, negatives, rank-wise success,
witness dedup accounting, thermal phases, CPU / bytes, and scene-backed figures / GIFs.

    python v2_report.py <launch_dir_v2> <outdir>        # writes <outdir>/v2_evidence.json and figures
"""
import json, sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch_check as K  # noqa: E402
import launch_viz as V  # noqa: E402
import placement_teacher as T  # noqa: E402
from launch_viz import INK, INK2, SER, plt, Circle, Rectangle, FancyArrowPatch, viz  # noqa: E402
P = T.P
RUNS = ("v2_ref64_25", "v2_adaptive25_a", "v2_adaptive25_b", "v2_adaptive_serial", "v2_adaptive_interrupted")


def goal(rid, _c={}):
    return _c.setdefault(rid, P.load_root(rid)["goal_cell"])


def tables(L):
    ref = {q["trial_id"]: (h, q, a) for h, q, _, a in K.load(L / "pilot" / RUNS[0]) if q["eligible"] and q["completed"]}
    ada = {q["trial_id"]: (h, q, a) for h, q, _, a in K.load(L / "pilot" / RUNS[1]) if q["eligible"] and q["completed"]}
    rows = []
    for tid, (h, q, a) in sorted(ref.items()):
        y = {"8": K.virtual(q, a, 8, "FULL")["label"], "32": K.virtual(q, a, 32, "FULL")["label"], "64": q["label"], "adaptive": ada[tid][1]["label"]}
        rows.append({"tid": tid, "root": h["root_id"], "group": h["root_group_id"], "cls": h["privileged_teacher"]["verified_class"], "block": q["block"],
                     "dest_bin": q["dest_bin"], "source_bin": q["source_bin"], "goal_dest": q["destination_cell"] == goal(h["root_id"]), "y": y,
                     "kind": ada[tid][1]["negative_kind"], "pick0": ada[tid][1]["pick_access_modes_ok"] == 0, "valid": q["mode_counts"]["valid_pairs"],
                     "escalated": ada[tid][1]["work"]["pairs_planned"] > 8, "cpu": ada[tid][1]["volatile"]["cpu_s"], "cpu64": q["volatile"]["cpu_s"]})
    pos = [r for r in rows if r["y"]["64"]]
    miss = {}
    for pol in ("8", "adaptive", "32"):
        m = [r for r in pos if not r["y"][pol]]
        strata = {}
        for dim in ("block", "dest_bin", "cls"):
            d = defaultdict(lambda: [0, 0])
            for r in pos:
                d[r[dim]][0] += 1
                d[r[dim]][1] += not r["y"][pol]
            strata[dim] = {k: {"pos64": v[0], "missed": v[1], "miss_frac": round(v[1] / v[0], 4), "ci95": K.wilson(v[1], v[0])} for k, v in sorted(d.items())}
        flips = sum(r["y"][pol] == 1 and r["y"]["64"] == 0 for r in rows)
        miss[pol] = {"label1": sum(r["y"][pol] for r in rows), "label0": sum(not r["y"][pol] for r in rows), "missed": len(m), "pos64": len(pos),
                     "miss_frac": round(len(m) / len(pos), 4), "ci95": K.wilson(len(m), len(pos)), "pos_not_in_64": flips, "strata": strata,
                     "missed_ids": [r["tid"] for r in m]}
    neg = [r for r in rows if not r["y"]["adaptive"]]
    rank = defaultdict(lambda: [0, 0])
    for _, (h, q, a) in ref.items():
        for rk, v in zip(a["a_rank"], a["a_verified"]):
            if rk >= 0:
                b = "0-7" if rk < 8 else "8-15" if rk < 16 else "16-31" if rk < 32 else "32-63"
                rank[b][0] += 1
                rank[b][1] += bool(v)
    esc = [r for r in rows if r["escalated"]]
    return rows, {"completed_queries": len(rows), "miss_vs_ref64": miss,
                  "labels_by_block": {pol: dict(Counter(f"{r['block']}:{r['y'][pol]}" for r in rows)) for pol in ("adaptive", "64")},
                  "labels_by_dest_bin": {pol: dict(Counter(f"{r['dest_bin']}:{r['y'][pol]}" for r in rows)) for pol in ("adaptive", "64")},
                  "negatives_adaptive": {"queries": len(neg), "kinds": dict(Counter(r["kind"] for r in neg)), "classes": dict(Counter(r["cls"] for r in neg)),
                                         "roots": len({r["root"] for r in neg}), "lineage_groups": len({r["group"] for r in neg}), "blocks": dict(Counter(r["block"] for r in neg)),
                                         "certified_screen": sum(r["kind"] == "CERTIFIED_SCREEN" for r in neg), "operational": sum(r["kind"] != "CERTIFIED_SCREEN" for r in neg)},
                  "rank_success_ref64": {k: {"planned": v[0], "verified": v[1], "rate": round(v[1] / v[0], 4)} for k, v in sorted(rank.items())},
                  "escalated_to_32": {"queries": len(esc), "rescued": sum(r["y"]["adaptive"] for r in esc), "valid_le_8": sum(r["valid"] <= 8 for r in rows)},
                  "goal_cell_destinations": {"queries": sum(r["goal_dest"] for r in rows), "label1": sum(r["goal_dest"] and r["y"]["adaptive"] for r in rows)},
                  "pick_access_zero_but_label1": sum(r["pick0"] and r["y"]["adaptive"] for r in rows),
                  "cpu_s_adaptive": {"mean": round(float(np.mean([r["cpu"] for r in rows])), 2), "median": round(float(np.median([r["cpu"] for r in rows])), 2),
                                     "p95": round(float(np.percentile([r["cpu"] for r in rows], 95)), 2), "sum_h": round(sum(r["cpu"] for r in rows) / 3600, 3)},
                  "cpu_s_ref64": {"mean": round(float(np.mean([r["cpu64"] for r in rows])), 2), "sum_h": round(sum(r["cpu64"] for r in rows) / 3600, 3)}}


def witnesses(run):
    nver = dist = files = 0
    per = []
    for h, q, m, a in K.load(run):
        if q["eligible"] and q["completed"]:
            ids = [str(w) for v, w in zip(a["a_verified"], a["a_witness"]) if v]
            nver += len(ids)
            dist += len(set(ids))
            per.append(len(q["witness_refs"]))
    for f in Path(run).glob("witnesses/*.npz"):
        files += len(np.load(f)["ids"])
    b = {d: sum(f.stat().st_size for f in Path(run).glob(f"{d}/*")) for d in ("records", "modes", "witnesses")}
    n = len(per)
    return {"verified_pairs": nver, "distinct_ids": dist, "dedup_removed": nver - dist, "packed_witnesses": files, "queries": n,
            "witnesses_per_query_mean": round(float(np.mean(per)), 2), "bytes": b, "bytes_per_query": round(sum(b.values()) / max(n, 1)),
            "bytes_per_witness": round(b["witnesses"] / max(files, 1))}


def thermal(run):
    m = [json.loads(x) for x in open(Path(run) / "monitor.jsonl")]
    sess = [json.loads(f.read_text()) for f in sorted(Path(run).glob("run_[0-9]*.json"))]
    full = [x for x in m if x.get("queued", 0) > 0]
    tail = [x for x in m if x.get("queued", 0) == 0]
    t = [x["temp_c"] for x in m if x["temp_c"] is not None]
    ph = lambda s: {"samples": len(s), "peak_c": max((x["temp_c"] for x in s), default=None), "median_c": float(np.median([x["temp_c"] for x in s])) if s else None,  # noqa: E731
                    "frozen_frac": round(sum(x["frozen"] for x in s) / len(s), 3) if s else None}
    rate = None
    w = [x for x in full if x.get("workers_cpu_s") is not None]
    if len(w) >= 2 and w[-1]["t"] > w[0]["t"]:
        rate = round((w[-1]["workers_cpu_s"] - w[0]["workers_cpu_s"]) / (w[-1]["t"] - w[0]["t"]), 2)
    return {"sessions": len(sess), "stops": [s["stop"] for s in sess], "wall_s": round(sum(s["wall_s"] for s in sess), 1),
            "worker_cpu_s": round(sum(s["children_cpu_s"] for s in sess), 1), "pauses": sum(s["pauses"] for s in sess),
            "paused_s": round(sum(s["paused_s"] for s in sess), 1), "peak_c": max(t), "median_c": float(np.median(t)),
            "startup_first_60s": ph([x for x in m if x["t"] <= 60]), "steady_queue_nonempty": ph(full), "tail_queue_empty": ph(tail),
            "worker_cpu_s_per_s_steady": rate, "min_mem_available_mb": min(x["mem_available_mb"] for x in m),
            "max_swap_growth_mb": max(x["swap_growth_mb"] for x in m), "max_worker_rss_mb": max(x["max_worker_rss_mb"] for x in m),
            "min_disk_free_mb": min(x["disk_free_mb"] for x in m), "max_descendants": max(x["descendants"] for x in m)}


def fig_policy(ev, rows, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.9))
    cats = [("block", k) for k in ("root", "prefix1", "prefix2", "prefix3")] + [("dest_bin", k) for k in ("near", "mid", "far", "vacated")]
    x = np.arange(len(cats))
    for i, (pol, col, mk) in enumerate((("8", SER[1], "s"), ("adaptive", SER[0], "o"), ("32", SER[2], "D"))):
        st = ev["miss_vs_ref64"][pol]["strata"]
        y = [1 - st[d][k]["miss_frac"] if k in st[d] else np.nan for d, k in cats]
        a1.plot(x + (i - 1) * 0.18, y, ls="none", marker=mk, ms=7, color=col, label=f"{pol} pairs" if pol != "adaptive" else "adaptive 8 -> 32")
    a1.axhline(0.98, color=INK2, lw=1, ls=":")
    a1.set(xticks=x, xticklabels=[f"{k}\n(n={ev['miss_vs_ref64']['8']['strata'][d][k]['pos64']})" if k in ev['miss_vs_ref64']['8']['strata'][d] else k for d, k in cats],
           ylabel="share of 64-pair positives recovered", title="Recall vs the 64-pair reference, by context block and destination bin (v2 pilot)")
    a1.tick_params(axis="x", labelsize=8)
    a1.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3)
    bins = np.logspace(-1.3, 2.6, 40)
    a2.hist([r["cpu"] for r in rows], bins=bins, color=SER[0], alpha=.75, label="adaptive 8 -> 32 (measured)")
    a2.hist([r["cpu64"] for r in rows], bins=bins, color=SER[1], alpha=.45, label="64-pair FULL (measured)")
    a2.set(xscale="log", xlabel="worker CPU-s per query (25-worker runs)", ylabel="queries", title="Teacher cost per query")
    a2.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_rank(ev, out):
    r = ev["rank_success_ref64"]
    ks = ["0-7", "8-15", "16-31", "32-63"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(ks, [r[k]["rate"] for k in ks], color=SER[0], width=.6)
    for i, k in enumerate(ks):
        ax.text(i, r[k]["rate"] + .01, f"{r[k]['verified']}/{r[k]['planned']}", ha="center", fontsize=8, color=INK2)
    ax.set(ylim=(0, 1.05), xlabel="schedule rank of the planned (held, place) pair", ylabel="verified fraction",
           title="Rank-wise success, 64-pair reference (not a calibrated probability)")
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)


def fig_thermal(L, out, S):
    runs = [r for r in RUNS if (L / "pilot" / r / "monitor.jsonl").exists() and r != "v2_adaptive_serial"]
    fig, axs = plt.subplots(len(runs), 1, figsize=(11, 2.6 * len(runs)), sharex=False, squeeze=False)
    for ax, r in zip(axs[:, 0], runs):
        m = [json.loads(x) for x in open(L / "pilot" / r / "monitor.jsonl")]
        raw, t, off, cuts = [x["t"] for x in m], [], 0.0, []
        for i, x in enumerate(raw):                   # a resumed session restarts its clock: place it after the previous one
            if i and x < raw[i - 1]:
                off += raw[i - 1] + 2.0
                cuts.append(off / 60)
            t.append((x + off) / 60)
        t = np.array(t)
        seg = np.split(np.arange(len(t)), [int(np.searchsorted(t, c)) for c in cuts])
        for sg in seg:
            ax.plot(t[sg], [m[i]["temp_c"] for i in sg], color=SER[0], lw=1.2)
        for c in cuts:
            ax.axvline(c, color=INK, lw=1.2, ls=":")
            ax.text(c, 58, " SIGKILL -> resume", fontsize=8, color=INK2)
        for v in (S["temp_cancel_c"], S["temp_pause_c"], S["temp_resume_c"]):
            ax.axhline(v, color=INK2, lw=.9, ls="--")
        for i, x in enumerate(m):
            if x["frozen"]:
                ax.axvspan(t[i] - .017, t[i] + .017, color="#dddcd7", lw=0)
        qe = next((t[i] for i, x in enumerate(m) if x.get("queued", 1) == 0 and (not cuts or t[i] > cuts[-1])), None)
        if qe is not None:
            ax.axvline(qe, color=SER[1], lw=1.2)
            ax.text(qe, S["temp_cancel_c"] - .5, " tail (queue empty)", color=INK2, fontsize=8, va="top")
        ax.axvspan(0, 1, color=SER[2], alpha=.08, lw=0)
        ax.text(.02, S["temp_cancel_c"] - .5, "startup", color=INK2, fontsize=8, va="top")
        ax.set(ylabel="Tctl (C)", title=f"{r}: peak {max(x['temp_c'] for x in m):.1f} C, frozen {np.mean([x['frozen'] for x in m]):.1%} of samples "
               f"(dashed: resume 80 / pause 88 / cancel 95)", ylim=(55, 97))
        ax.title.set_fontsize(9)
    axs[-1, 0].set_xlabel("minutes since the run's first session started (a resumed session is placed after the killed one)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def panel(ax, run, h, q, caption):
    rid = h["root_id"]
    s = T.scene_for({"root": rid, "prefix": q["prefix_key"]})
    viz.draw_support(ax, s, cell_ids=True)
    viz.draw_objects(ax, s)
    o = s.get(q["object_id"])
    ax.add_patch(Circle(o.xy_m, o.radius_m + .009, fc="none", ec=INK, lw=2.2, zorder=7))
    g = goal(rid)
    ax.add_patch(Rectangle(((g % 8) * .08, (g // 8) * .08), .08, .08, fc="none", ec=SER[3], lw=2, ls="--", zorder=3))
    c, y = q["destination_cell"], q["label"]
    col = "#bbbbbb" if y is None else SER[0] if y == 1 else SER[1]
    ax.add_patch(Rectangle(((c % 8) * .08 + .005, (c // 8) * .08 + .005), .07, .07, fc=col, alpha=.6, ec=col, lw=1.5, zorder=2))
    root = P.scene_from_dict(P.load_root(rid)["scene"])
    for k, (a, cell) in enumerate(q["prefix_key"]):
        ax.add_patch(FancyArrowPatch(root.get(a).xy_m, s.lattice.center(cell), arrowstyle="-|>", mutation_scale=12, color=INK2, ls="--", lw=1.4, zorder=8))
        ax.text(*s.lattice.center(cell), f" edit {k + 1}", fontsize=7, color=INK2, zorder=9)
    viz.draw_robot(ax, P.A.env()["robot"], np.asarray(s.robot.q_rad), P.A.env()["robot"].open_gap, alpha=.22, zorder=4)
    ax.set(xlim=(-.42, .7), ylim=(-.08, .92), aspect="equal", xticks=[], yticks=[])
    ax.grid(False)
    ax.set_title(caption, fontsize=7.5, loc="left")


def examples(L, rows, out):
    run = L / "pilot" / RUNS[1]
    rec = {q["trial_id"]: (h, q, a) for h, q, _, a in K.load(run)}
    byid = {r["tid"]: r for r in rows}
    pick = lambda f: next((t for t, r in byid.items() if f(r)), None)  # noqa: E731
    chosen = [("positive, root scene", pick(lambda r: r["block"] == "root" and r["y"]["adaptive"] and not r["escalated"])),
              ("negative, root scene", pick(lambda r: r["block"] == "root" and not r["y"]["adaptive"] and r["kind"] != "CERTIFIED_SCREEN")),
              ("destination = T's goal cell", pick(lambda r: r["goal_dest"])),
              ("two-edit hypothetical context", pick(lambda r: r["block"] == "prefix3" and r["y"]["adaptive"])),
              ("no pick-access mode, still placeable", pick(lambda r: r["pick0"] and r["y"]["adaptive"]))]
    chosen = [(c, t) for c, t in chosen if t]
    import textwrap
    fig, axs = plt.subplots(1, len(chosen), figsize=(4.6 * len(chosen), 6.6), squeeze=False)
    used = []
    for ax, (cap, tid) in zip(axs[0], chosen):
        h, q, a = rec[tid]
        why = f"label {q['label']}" + (f" ({q['negative_kind']}, {q['negative_scope'].split(',')[0]}, phase {q['failure_phase']})" if q["label"] == 0 else "")
        pre = "root scene" if not q["prefix_key"] else f"HYPOTHETICAL edits {q['prefix_key']}"
        txt = (f"{cap.upper()}\n{h['root_id']} {q['object_id']} -> cell {q['destination_cell']}\n{pre}\n{why}\nplanned {q['work']['pairs_planned']} pairs, "
               f"{q['n_verified_pairs']} verified; pick-access modes {q['pick_access_modes_ok']}, held modes {q['held_modes_valid']}")
        panel(ax, run, h, q, "\n".join(textwrap.fill(line, 52) for line in txt.split("\n")))
        used.append({"case": cap, "trial_id": tid, "root": h["root_id"], "label": q["label"]})
    h_ = [Rectangle((0, 0), 1, 1, fc=SER[0], alpha=.6), Rectangle((0, 0), 1, 1, fc=SER[1], alpha=.6), Rectangle((0, 0), 1, 1, fc="none", ec=SER[3], ls="--", lw=2)]
    fig.legend(h_, ["destination, label 1 (adaptive)", "destination, label 0 (adaptive, policy-limited)", "T's goal cell (not excluded in v2)"], loc="lower center", ncol=3,
               fontsize=8, frameon=False)
    fig.suptitle("v2 pilot (train roots) - ringed object is already held; dashed arrows are hypothetical object-only edits, not robot motion", fontsize=10)
    fig.tight_layout(rect=(0, .05, 1, .9))
    fig.savefig(out / "v2_examples.png", dpi=110)
    plt.close(fig)
    gifs = []
    for cap, tid in chosen:
        h, q, a = rec[tid]
        if q["label"] == 1 and q["witness_refs"] and cap != "destination = T's goal cell" or (cap == "destination = T's goal cell" and q["label"] == 1):
            name = out / f"v2_{tid}_{cap.split(',')[0].split(' ')[0].replace('=', '').strip() or 'goal'}_witness.gif"
            V.witness_gif(run, h["root_id"], q, q["witness_refs"][0], name)
            gifs.append(str(name))
    esc = next((t for t, r in byid.items() if r["escalated"] and r["y"]["adaptive"]), None) or next((t for t, r in byid.items() if r["escalated"]), None)
    if esc:
        V.mode_matrix(run, esc, out / f"v2_{esc}_mode_pairs_escalated.png")
    return used, gifs, esc


def main(L, out):
    L, out = Path(L).resolve(), Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows, ev = tables(L)
    ev["witnesses"] = {r: witnesses(L / "pilot" / r) for r in RUNS if (L / "pilot" / r / "records").exists()}
    ev["thermal"] = {r: thermal(L / "pilot" / r) for r in RUNS if (L / "pilot" / r / "monitor.jsonl").exists()}
    S = json.loads((L / "configs" / "launch_config.json").read_text())["safety"]
    fig_policy(ev, rows, out / "v2_adaptive_vs_ref64.png")
    fig_rank(ev, out / "v2_rank_success.png")
    fig_thermal(L, out / "v2_thermal_25_workers.png", S)
    ev["figure_examples"], ev["gifs"], ev["mode_matrix_query"] = examples(L, rows, out)
    (out / "v2_evidence.json").write_text(json.dumps(ev, indent=1, sort_keys=True, default=str))
    return ev


if __name__ == "__main__":
    e = main(*sys.argv[1:3])
    print(json.dumps({k: v for k, v in e.items() if k not in ("thermal", "witnesses")}, indent=1, default=str)[:4000])
