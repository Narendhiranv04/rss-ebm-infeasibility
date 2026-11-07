"""Figures for the mock (MUJOCO_GL=egl for the 3-D stills / GIFs).

  scenes MOCK ROOT_ID OUT.png        root / 1-edit / 2-edit layouts (schematic + MuJoCo render), queries coloured by pilot label
  plots ANALYSIS.json MOCK OUT_DIR   policy comparison, worker sweep, resource traces, labels by stratum, mode accounting
  failgif MOCK QID OUT.gif           re-execute the first failed motion-tested unit of a negative query (deterministic), with frames
"""
import gzip, json, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
C = {"1": "#2a78d6", "0": "#eb6834", "None": "#8a8984", "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a"}
plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.spines.top": False, "axes.spines.right": False, "figure.facecolor": SURF, "axes.facecolor": SURF,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True})


def _roster(mock):
    return [json.loads(x) for x in gzip.open(Path(mock) / "roster.jsonl.gz", "rt")]


def _labels(mock, run="pilot_A"):
    out = {}
    for f in (Path(mock) / "runs" / run / "units").glob("*.json"):
        for r in json.loads(f.read_text())["records"]:
            out[r["qid"]] = r
    return out


def scenes(mock, rid, out):
    import placement3d as P
    import replay3d as R
    rows = [q for q in _roster(mock) if q["root_id"] == rid]
    lab = _labels(mock)
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
    fig.suptitle(f"SHELF66V3 placement queries over one lineage, root {rid} (K{rows[0]['K']}); arrows = query object -> destination, coloured by pilot label",
                 color=INK, fontsize=11)
    fig.savefig(out, dpi=120); print(out)


def plots(an, mock, od):
    an, od = json.loads(Path(an).read_text()), Path(od)
    pol = an["policies"]
    fig, a = plt.subplots(figsize=(7.2, 4.2))
    groups = {}
    for n, r in pol.items():                                    # identical (cost, miss) points share one label
        groups.setdefault((round(r["virtual_cpu_mean_s"], 2), r["missed"]), []).append(n)
    for k, (key, ns) in enumerate(sorted(groups.items())):
        r = pol[ns[0]]; lo, hi = r["rel_miss_ci95"]
        hl = any(n in ("adaptive_v1", "fixed_4x1500", "reference_v1") for n in ns)
        a.errorbar(r["virtual_cpu_mean_s"], r["rel_miss"], yerr=[[r["rel_miss"] - lo], [hi - r["rel_miss"]]], fmt="o", ms=5 if hl else 3.5,
                   color=C["s1"] if hl else INK2, ecolor=C["s1"] if hl else GRID, lw=1)
        lab = " / ".join(n.replace("fixed_", "") for n in sorted(ns))
        a.annotate(lab, (r["virtual_cpu_mean_s"], r["rel_miss"]), textcoords="offset points", xytext=(4, 5 if k % 2 else -11), fontsize=6.5, color=INK)
    a.axhline(0.05, color=C["s2"], lw=1, ls="--"); a.text(a.get_xlim()[1], 0.052, "predeclared max 5%", ha="right", fontsize=7, color=C["s2"])
    a.set_xlabel("virtual mean CPU-s per query (reference subset)"); a.set_ylabel("relative missed positives vs reference_v1")
    a.set_title(f"Budget comparison on {pol['reference_v1']['n_paired']} paired reference queries ({pol['reference_v1']['ref_pos']} reference positives), Wilson 95%",
                fontsize=9, color=INK)
    fig.tight_layout(); fig.savefig(od / "budget_policies.png", dpi=140); plt.close(fig)

    sw = an["sweep"]
    ks = sorted((k for k in sw if k.startswith("sweep_")), key=lambda k: sw[k]["workers"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot([sw[k]["workers"] for k in ks], [sw[k]["q_per_s"] for k in ks], "-o", color=C["s1"], lw=2, ms=6)
    for k in ks:
        ax[0].annotate(f"{sw[k]['q_per_s']:.3f}", (sw[k]["workers"], sw[k]["q_per_s"]), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
    ax[0].set_xlabel("workers (pinned, 1 thread each)"); ax[0].set_ylabel("queries per wall-second"); ax[0].set_xticks([sw[k]["workers"] for k in ks])
    ax[0].set_title("Throughput, matched sweep subset", fontsize=9)
    ax[1].bar([str(sw[k]["workers"]) for k in ks], [sw[k]["cpu_inflation_vs_serial"] for k in ks], color=C["s1"], width=0.55)
    ax[1].axhline(1.0, color=INK2, lw=1); ax[1].set_xlabel("workers"); ax[1].set_ylabel("CPU-s per query / serial")
    ax[1].set_title("Per-query CPU inflation (SMT / memory contention)", fontsize=9)
    fig.tight_layout(); fig.savefig(od / "worker_sweep.png", dpi=140); plt.close(fig)

    mon = [json.loads(x) for x in (Path(mock) / "runs/pilot_A/monitor.jsonl").read_text().splitlines()]
    t0 = mon[0]["t"]; t = [(m["t"] - t0) / 60 for m in mon]
    fig, ax = plt.subplots(3, 1, figsize=(7.5, 5.6), sharex=True)
    for a, (k, lab, f) in zip(ax, (("cpu_pct_sum", "busy CPUs (test-owned)", 0.01), ("pkg_temp_c", "package temp (C)", 1), ("mem_avail_gb", "MemAvailable (GB)", 1))):
        a.plot(t, [(m[k] or 0) * f for m in mon], color=C["s1"], lw=1.5); a.set_ylabel(lab, fontsize=8)
    ax[0].axhline(18, color=C["s2"], ls="--", lw=1); ax[0].text(t[-1], 18.3, "18-CPU ceiling", ha="right", fontsize=7, color=C["s2"])
    ax[-1].set_xlabel("minutes since start (pilot_A, 12 workers)"); fig.suptitle("Resource trace, broad pilot A (adaptive_v1, recommended)", fontsize=10)
    fig.tight_layout(); fig.savefig(od / "resources_pilot_A.png", dpi=140); plt.close(fig)

    st = an["pilot"]["by_stratum"]
    groups = [g for g in ("depth=0", "depth=1", "depth=2", "band=near", "band=mid", "band=far", "band=vacated", "shape=box", "shape=cylinder",
                          "K=0", "K=1", "K=2", "K=3") if g in st]
    fig, a = plt.subplots(figsize=(9, 3.8))
    bottom = np.zeros(len(groups))
    for k, lab in (("1", "label 1"), ("0", "label 0"), ("unlabeled", "unlabeled")):
        v = np.array([st[g][k] for g in groups]); a.bar(groups, v, bottom=bottom, color=C["None" if k == "unlabeled" else k], label=lab, width=0.6,
                                                      edgecolor=SURF, linewidth=1.5); bottom += v
    a.legend(frameon=False, ncol=3, fontsize=8); a.set_ylabel("queries"); a.tick_params(axis="x", rotation=35)
    a.set_title("Broad pilot A (adaptive_v1, recommended): outcomes by stratum (205 queries, 22 root units)", fontsize=9)
    fig.tight_layout(); fig.savefig(od / "pilot_labels_by_stratum.png", dpi=140); plt.close(fig)

    acc = an["pilot"]["accounting"]
    keys = ["ik_invalid", "held_invalid", "held_invalid_physical", "geometry_invalid", "untested", "unresolved", "plan_failed", "exec_failed", "verified", "unscreened"]
    fig, a = plt.subplots(figsize=(8, 3.4))
    a.barh(keys[::-1], [acc.get(k, 0) for k in keys[::-1]], color=C["s1"], height=0.6)
    for i, k in enumerate(keys[::-1]):
        a.text(acc.get(k, 0), i, f" {acc.get(k, 0)}", va="center", fontsize=7)
    a.set_xlabel(f"grasp poses over {an['pilot']['n']} pilot queries ({acc.get('nominal', 0)} nominal)")
    a.set_title("Mode accounting, pilot A (adaptive_v1; lazy screening: unscreened poses were never needed)", fontsize=9)
    fig.tight_layout(); fig.savefig(od / "pilot_mode_accounting.png", dpi=140); plt.close(fig)
    bp = an.get("broad_pilot_vs_reference")
    if bp:
        bands = ["near", "mid", "far", "vacated"]
        fig, a = plt.subplots(figsize=(7.2, 3.8))
        for j, (name, col) in enumerate((("adaptive_v1", C["s1"]), ("fixed_4x1500", C["s2"]))):
            st = bp[name]["by_stratum"]
            y = [st[f"band={b}"]["rel_miss"] for b in bands]
            lo = [y[i] - st[f"band={b}"]["ci95"][0] for i, b in enumerate(bands)]; hi = [st[f"band={b}"]["ci95"][1] - y[i] for i, b in enumerate(bands)]
            x = np.arange(len(bands)) + (j - 0.5) * 0.36
            a.bar(x, y, width=0.34, color=col, label=f"{name} ({bp[name]['missed']}/{bp['ref_positives']} missed)", edgecolor=SURF, linewidth=1.5)
            a.errorbar(x, y, yerr=[lo, hi], fmt="none", ecolor=INK2, lw=1, capsize=3)
            for xi, yi in zip(x, y):
                a.text(xi, yi + 0.004, f"{yi:.0%}" if yi else "0", ha="center", fontsize=7, color=INK)
        a.axhline(0.10, color=INK2, ls="--", lw=1); a.text(3.45, 0.105, "stratum max 10%", ha="right", fontsize=7, color=INK2)
        a.set_xticks(range(len(bands))); a.set_xticklabels([f"{b}\n({bp['adaptive_v1']['by_stratum'][f'band={b}']['ref_pos']} ref pos)" for b in bands])
        a.set_ylabel("relative missed positives vs reference_v1"); a.legend(frameon=False, fontsize=8)
        a.set_title("Broad pilot (205 queries), Wilson 95%: the far band separates the policies", fontsize=9)
        fig.tight_layout(); fig.savefig(od / "broad_pilot_miss_by_band.png", dpi=140); plt.close(fig)
    print("plots ->", od)


def failgif(mock, qid, out):
    import placement3d as P
    import policy3d as Q
    import replay3d as R
    from PIL import Image
    q = next(x for x in _roster(mock) if x["qid"] == qid)
    rec = _labels(mock)[qid]
    m = next(m for m in rec["modes"] if m["state"] in ("EXEC_FAILED", "PLAN_FAILED", "UNRESOLVED") and m["tests"])
    raw = json.loads((P.SRC / "raw/roots" / f"{q['root_id']}.json").read_text())
    sc = P.Scene(raw["objects"], q["cells"])
    g = Q.ranked(P.poses(sc, q["object"]))[m["rank"]]
    st, br, _ = P.screen(sc, q["object"], tuple(q["dest"]), g)
    pl = P.plan(sc, q["object"], tuple(q["dest"]), g, br, m["tests"][-1]["iters"])
    ok, hev, before = P.hold(sc, q["object"], g, pl.get("q_g", br[0]))
    frames = [R.render(sc)]
    info = {"qid": qid, "rank": m["rank"], "state": m["state"], "plan_ok": pl["ok"], "reason": pl.get("reason")}
    if pl["ok"]:
        P.REC_HOOK = (lambda: frames.append(R.render(sc)), 40)
        recd, ev = P.execute(sc, q["object"], pl, before)
        P.REC_HOOK = None
        v, vi = P.verify(sc, q["object"], tuple(q["dest"]), recd, ev)
        info.update(verified=v, first_failed=vi["first_failed"])
    else:
        frames += [frames[0]] * 10
    ims = [Image.fromarray(f) for f in frames]
    ims[0].save(out, save_all=True, append_images=ims[1:], duration=60, loop=0)
    print(json.dumps(info))


if __name__ == "__main__":
    c = sys.argv[1]
    {"scenes": lambda: scenes(*sys.argv[2:5]), "plots": lambda: plots(*sys.argv[2:5]), "failgif": lambda: failgif(*sys.argv[2:5])}[c]()
