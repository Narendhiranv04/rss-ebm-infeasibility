"""P6.7 (models): proposal, training, side-by-side and selected-execution GIFs for A / B / C.

Inputs are the recorded training snapshots and evaluations (runs/p6/train/results.json) and the
teacher records; a selected repair is animated only if the teacher holds a recorded VERIFIED witness
for exactly that sequence, otherwise the panel says NO VERIFIED TRACE with the teacher status.
"""
from __future__ import annotations

import numpy as np

import operational_teacher as ot
import p6_viz as pv
from scene import scene_from_dict
from viz import COLORS, plt

RANK = ["#1f6fd1", "#16a085", "#d35400"]


def _bars(ax, ev, title):
    rows = sorted(ev["cands"], key=lambda c: -c["p"])
    ax.bar(range(len(rows)), [c["p"] for c in rows], color=[pv.COL.get({1: ot.VERIFIED, 0: ot.CERTIFIED}.get(c["y"], ot.UNKNOWN)) for c in rows])
    ax.scatter(range(len(rows)), [c["y"] if c["y"] is not None else np.nan for c in rows], marker="_", s=120, color="black", zorder=5)
    ax.set_xticks(range(len(rows)), [c["query"].split(":", 2)[2][:22] for c in rows], rotation=80, fontsize=5)
    ax.set(ylim=(0, 1.05), title=title)
    ax.title.set_fontsize(8)
    ax.axhline(0.5, color="#888", lw=0.6, ls=":")
    ax.tick_params(labelsize=6)


def _slots(ax, c, title):
    if "marg" not in c:
        ax.axis("off")
        ax.text(0.05, 0.5, "A has no outcome field z\n(GRU task logit only)", fontsize=8, transform=ax.transAxes)
        return
    x = np.arange(len(c["slots"]))
    ax.bar(x - 0.2, c["marg"], 0.4, color=COLORS["hypothetical"], label="model P(z_k = 1 | r)")
    ax.bar(x + 0.2, [np.nan if z is None else z for z in c["z_obs"]], 0.4, color="#555", label="teacher z_k (masked if UNKNOWN)")
    ax.set_xticks(x, c["slots"], rotation=60, fontsize=6)
    ax.set(ylim=(0, 1.05), title=title)
    ax.title.set_fontsize(8)
    ax.legend(fontsize=5, loc="upper right")
    ax.tick_params(labelsize=6)


def _main(ev):
    return max(ev["cands"], key=lambda c: (c["m"], c["query"]))


def _scene_top(ax, cfg, robot, plan, ev, title):
    r = plan["roots"][ev["root"]]
    s = scene_from_dict(r["scene"])
    pv._panel(ax, cfg, robot, s, r["goal_cell"], title)
    seqs = {p["z_layout"]["query"].get("zR", f"seq:{ev['root']}:empty"): p["sequence"] for p in r["queries"]["pool"]}
    for n, c in enumerate(sorted(ev["cands"], key=lambda c: c["energy"])[:3]):
        pv._arrows(ax, s, [tuple(a) for a in seqs[c["query"]]], RANK[n])
        ax.text(0.02, 0.9 - 0.06 * n, f"#{n + 1} E={c['energy']:.2f} p={c['p']:.2f} {c['query'].split(':', 2)[2]}", color=RANK[n],
                fontsize=6, transform=ax.transAxes, weight="bold")
    pv._tag(ax, "MODEL PROPOSALS - NOT EXECUTED", COLORS["hypothetical"])


def model_visuals(dst, cfg, robot, plan, records, collect, res) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    runs = {m: res["runs"].get(f"{m}#s0", {}) for m in ("A", "B", "C")}
    evs = {m: [e for sp in ("validation", "holdout") for e in r.get("eval", {}).get(sp, []) if e["root"] in plan["roots"]
               and plan["roots"][e["root"]]["variant"] == "obstructed"] for m, r in runs.items()}
    for m, r in runs.items():
        if r.get("status") != "RUN":
            continue
        frames = []
        for ev in evs[m]:
            f, ax = plt.subplots(1, 3, figsize=(15, 5))
            sp = plan["roots"][ev["root"]]["split"]
            _scene_top(ax[0], cfg, robot, plan, ev, f"{m} on {ev['root']} ({sp}): lowest-energy proposals")
            _bars(ax[1], ev, "p(feasible | r) per candidate (colour = teacher label, bar - = y)")
            _slots(ax[2], _main(ev), f"longest candidate {_main(ev)['query'].split(':', 2)[2]}: z marginals")
            frames += [pv._rgb(f)] * pv.STATIC_FRAMES
        if frames:
            pv.gif(dst / f"{m}_proposals.gif", frames, f"{m} (seed 0): proposals, probabilities and z marginals on P6 validation + holdout roots")
        snaps, hist = res["snapshots"].get(m, []), [h for h in r["history"] if "total" in h]
        frames = []
        for sn in snaps:
            f, ax = plt.subplots(1, 3, figsize=(15, 5))
            hh = [h for h in hist if h["epoch"] <= sn["epoch"]]
            ax[0].plot([h["epoch"] for h in hh], [h["total"] for h in hh], color="#333", label="train total loss")
            vv = [h for h in hh if "val" in h]
            ax[0].plot([h["epoch"] for h in vv], [h["val"] for h in vv], "o-", color=COLORS["invalid"], ms=3, label="validation loss")
            ax[0].set(xlim=(0, max(5, hist[-1]["epoch"])), title=f"{m} training, epoch {sn['epoch']}", xlabel="epoch")
            ax[0].legend(fontsize=7)
            _bars(ax[1], sn["eval"], f"checkpoint snapshot on {res['snapshots_root']}")
            _slots(ax[2], _main(sn["eval"]), "longest candidate: z marginals")
            frames.append(pv._rgb(f))
        if frames:
            pv.gif(dst / f"{m}_training.gif", frames, f"{m} (seed 0): loss curves and recorded checkpoint predictions over training")
    roots = [e["root"] for e in evs["A"]] if runs["A"].get("status") == "RUN" else []
    frames, sel = [], []
    for k in roots:
        f, ax = plt.subplots(2, 3, figsize=(15, 9))
        for j, m in enumerate("ABC"):
            ev = next((e for e in evs[m] if e["root"] == k), None)
            if ev is None:
                ax[0, j].axis("off")
                ax[1, j].axis("off")
                continue
            met = res["runs"][f"{m}#s0"]["metrics"]
            _bars(ax[0, j], ev, f"{m}: {k} ({plan['roots'][k]['split']}); selected {ev['selected'] and ev['selected'].split(':', 2)[2]}")
            c = _main(ev)
            if "top_states" in c:
                ts = c["top_states"]
                ax[1, j].bar(range(len(ts)), [p for _, p in ts], color=COLORS["hypothetical"])
                ax[1, j].set_xticks(range(len(ts)), ["".join(map(str, z)) for z, _ in ts], fontsize=6, rotation=30)
                ax[1, j].set_title(f"{m}: top joint z states of {c['query'].split(':', 2)[2]}\nslots {','.join(c['slots'])}", fontsize=7)
            else:
                _slots(ax[1, j], c, "")
            ax[1, j].text(0.0, -0.32, f"val acc {met['validation'].get('accuracy')}  holdout acc {met.get('holdout', {}).get('accuracy')}",
                          transform=ax[1, j].transAxes, fontsize=6)
        frames += [pv._rgb(f)] * pv.STATIC_FRAMES
        sel.append(k)
    if frames:
        pv.gif(dst / "ABC_side_by_side.gif", frames, "A | B | C per P6 validation/holdout root: candidate probabilities and B/C joint z states")
    frames = []
    for k in sel:
        r = plan["roots"][k]
        s = scene_from_dict(r["scene"])
        side = []
        for m in "ABC":
            ev = next(e for e in evs[m] if e["root"] == k)
            rec = records.get(ev["selected"]) if ev["selected"] else None
            w = pv.witness(collect, rec) if rec and rec["teacher_status"] == ot.VERIFIED and rec["query"]["sequence"] else None
            tr = pv.trace_frames(robot, s, [tuple(a) for a in rec["query"]["sequence"]], w[1], w[2]) if w else None
            side.append((m, ev["selected"], rec, tr))
        n = max([len(t) for *_, t in side if t] + [pv.STATIC_FRAMES])
        for i in range(n):
            f, ax = plt.subplots(1, 3, figsize=(14, 5))
            for j, (m, q, rec, tr) in enumerate(side):
                title = f"{m} selects {q.split(':', 2)[2] if q else 'nothing (no p >= 0.5)'}"
                if tr:
                    qq, g, pos, mv = tr[min(i, len(tr) - 1)]
                    pv._panel(ax[j], cfg, robot, s, r["goal_cell"], title, q=qq, gap=g, pos=pos, held=mv)
                    pv._tag(ax[j], "teacher-RECORDED witness of the selected repair", COLORS["valid"], size=6)
                else:
                    pv._panel(ax[j], cfg, robot, s, r["goal_cell"], title)
                    st = rec["teacher_status"] if rec else "NOT_RUN"
                    pv._tag(ax[j], pv._status_line(rec) if rec else "NO SELECTION / NO VERIFIED TRACE", pv.COL.get(st, "#999"), size=6)
            frames.append(pv._rgb(f))
    if frames:
        pv.gif(dst / "ABC_selected_execution.gif", frames, "Repair selected by A | B | C per root, shown with the teacher's recorded witness if one exists")
