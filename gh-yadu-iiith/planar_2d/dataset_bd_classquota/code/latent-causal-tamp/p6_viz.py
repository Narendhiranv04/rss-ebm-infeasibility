"""P6.7: mandatory scene-based visualizations (PNG + GIF) from recorded data only.

Robot motion is drawn ONLY from recorded, revalidated witness arrays (relocation.npz /
target_witness.npz of the seed attempt that produced the VERIFIED record). Everything else is
labelled as what it is: certified negatives as static panels with their certificate, budget
negatives and UNKNOWN as "NO VERIFIED TRACE", fixed-robot edits as "HYPOTHETICAL SCENE EDIT - NOT A
ROBOT RELOCATION", proposals as "PROPOSED, NOT EXECUTED". Model GIFs show recorded checkpoint
snapshots and evaluations. 10 fps, final frame held >= 1.2 s, SHA-256 beside every file.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from PIL import Image

import config as cf
import effects as ef
import g4d_run as gr
import geometry as ge
import motion as mo
import operational_teacher as ot
import teacher
import viz
from scene import scene_from_dict
from viz import COLORS, plt

FRAME_MS, HOLD_MS, STATIC_FRAMES, DPI = 100, 1200, 8, 48
VIEW = ((-0.60, 0.72), (-0.22, 0.98))         # base, full grid and every start stratum
COL = {ot.VERIFIED: COLORS["valid"], ot.CERTIFIED: COLORS["invalid"], ot.BUDGET: "#8e44ad", ot.UNKNOWN: COLORS["unresolved"],
       "NOT_RUN": "#9a9a9a", "FATAL_CONTRADICTION": "#000000"}
NOTRACE = {ot.CERTIFIED: "CERTIFIED INFEASIBLE ({}) - no trajectory exists to show", ot.BUDGET: "BUDGET_INFEASIBLE (operational) - NO VERIFIED TRACE",
           ot.UNKNOWN: "UNKNOWN / NO VERIFIED TRACE", "NOT_RUN": "NOT RUN / NO VERIFIED TRACE", "FATAL_CONTRADICTION": "FATAL - see report"}
INDEX = []


def _sha(p):
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _register(path, what):
    (Path(str(path) + ".sha256")).write_text(_sha(path) + "\n")
    INDEX.append({"file": str(path), "what": what, "sha256": _sha(path)})


def _rgb(f):
    f.canvas.draw()
    a = np.asarray(f.canvas.buffer_rgba())[..., :3].copy()
    plt.close(f)
    return a


def gif(path, frames, what):
    ims = [Image.fromarray(a) for a in frames]
    dur = [FRAME_MS] * (len(ims) - 1) + [HOLD_MS]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=dur, loop=0, optimize=True)
    _register(path, what)


def png(f, path, what):
    f.savefig(path, dpi=70)
    plt.close(f)
    _register(path, what)


def _panel(ax, cfg, robot, s, goal, title, q=None, gap=None, pos=None, held=None):
    viz.scene_panel(ax, s, cfg, title, hide=tuple(pos or ()) + ((held,) if held else ()))
    ax.set(xlim=VIEW[0], ylim=VIEW[1])
    for oid, xy in (pos or {}).items():
        o = s.get(oid)
        fc, ec = (COLORS["target"], COLORS["target_edge"]) if o.kind == "target" else (COLORS["obstacle"], COLORS["obstacle_edge"])
        ax.add_patch(Circle(xy, o.radius_m, fc=fc, ec=ec, lw=0.8, zorder=8 if oid == held else 3))
    ax.add_patch(Circle(s.lattice.center(goal), 0.026, fc="none", ec=COLORS["target_edge"], ls="--", lw=1.4, zorder=4))
    viz.draw_robot(ax, robot, s.robot.q_rad if q is None else q, s.robot.gripper_gap_m if gap is None else gap, alpha=0.8, zorder=6)


def _tag(ax, text, color, loc=(0.02, 0.02), size=7):
    ax.text(*loc, text, transform=ax.transAxes, fontsize=size, color="white", weight="bold", va="bottom",
            bbox=dict(fc=color, ec="none", pad=2), zorder=30)


def _arrows(ax, s, seq, color):
    for n, (oid, c) in enumerate(seq, 1):
        a, b = s.get(oid).xy_m, s.lattice.center(c)
        ax.add_patch(Circle(b, 0.024, fc="none", ec=color, ls=":", lw=1.2, zorder=4))
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=color, lw=1.1, zorder=9))
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, str(n), fontsize=8, weight="bold", color=color, zorder=10)


def witness(collect: Path, rec) -> tuple:
    """(unit dir, relocation arrays, target arrays) of the seed attempt that verified `rec`, or None."""
    w = next((a for a in rec["attempts"] if a["terminal"] == "WITNESS"), None)
    if w is None:
        return None
    qd = collect / "q" / rec["query"]["query"].replace(":", "__").replace(">", "-") / f"seed{w['seed']}"
    st = json.loads((qd / "ledger.json").read_text())["units"][w["unit"]]
    d = qd / st["dir"]
    return d, gr._arrays(d / "relocation.npz"), gr._arrays(d / "target_witness.npz")


def trace_frames(robot, s, seq, rel, tw, n_move=14, n_target=18) -> list:
    """[(q, gap, positions {id: xy}, moving id)] sampled along the recorded arrays only."""
    pos, out = {o.id: tuple(o.xy_m) for o in s.objects}, []
    for k, a in enumerate(rel):
        oid = seq[k][0]
        for q, g, ph in mo.samples(a, n_move):
            pos[oid] = tuple(mo.target_xy(robot, a, ph, q, pos[oid]))
            out.append((q, g, dict(pos), oid))
        pos[oid] = tuple(s.lattice.center(seq[k][1]))
    for a in tw[:1]:
        t0 = pos["t0"]
        for q, g, ph in mo.samples(a, n_target):
            pos["t0"] = tuple(mo.target_xy(robot, a, ph, q, t0))
            out.append((q, g, dict(pos), "t0"))
    return out


def _status_line(rec):
    st = rec["teacher_status"]
    kind = (rec.get("static_certificate") or {}).get("kind") or rec.get("reason", "")
    return NOTRACE.get(st, "").format(kind) if st != ot.VERIFIED else "VERIFIED - recorded witness"


def root_visuals(dst, cfg, robot, plan, records, collect, k) -> None:
    r, ref = plan["roots"][k], plan["roots"][k + "_ref"]
    s, goal = scene_from_dict(r["scene"]), r["goal_cell"]
    mine = {q: x for q, x in records.items() if x["query"]["lineage"] == k}
    inv = {v: kk for kk, v in r["ids"].items()}
    f, ax = plt.subplots(1, 2, figsize=(10, 5))
    _panel(ax[0], cfg, robot, s, goal, f"{k} [{r['family']}, {r['split']}, start {r['stratum']}] at q0")
    for o in s.objects:
        if o.id in inv:
            ax[0].text(o.xy_m[0], o.xy_m[1] + 0.03, inv[o.id], ha="center", fontsize=8, weight="bold", color="#c0392b", zorder=12)
    _panel(ax[1], cfg, robot, scene_from_dict(ref["scene"]), goal, f"{k}_ref: reference scene (movers removed)")
    png(f, dst / "scene_start.png", f"{k}: initial scene, roles, goal (dashed), start stratum {r['stratum']}")
    pool = r["queries"]["pool"]
    f, ax = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1, 1.25]})
    _panel(ax[0], cfg, robot, s, goal, f"{k}: predeclared destinations (dotted rings, not executed)")
    for p in pool:
        _arrows(ax[0], s, [tuple(a) for a in p["sequence"]], COLORS["hypothetical"])
    ax[1].axis("off")
    lines = [(p["role"], p["z_layout"]["query"].get("zR", f"seq:{k}:empty")) for p in pool]
    lines += [(q["kind"], f"scene:{k}:" + ".".join(f"{o}>{c}" for o, c in sorted(q["edit"].items()))) for q in r["queries"]["scene_queries"]]
    lines = [("reference", f"seq:{k}_ref:empty")] + lines
    for i, (role, q) in enumerate(lines):
        st = records[q]["teacher_status"] if q in records else "NOT_RUN"
        ax[1].add_patch(plt.Rectangle((0, 1 - (i + 1) * 0.036), 0.03, 0.03, fc=COL[st], transform=ax[1].transAxes))
        ax[1].text(0.05, 1 - (i + 1) * 0.036 + 0.005, f"{role:<11} {q.split(':', 2)[2]:<34} {st}", fontsize=6.5, family="monospace",
                   transform=ax[1].transAxes)
    ax[1].legend(handles=[Patch(fc=c, label=s_) for s_, c in COL.items()], fontsize=6, loc="lower left", ncol=3, bbox_to_anchor=(0.0, -0.02))
    png(f, dst / "catalogue.png", f"{k}: frozen catalogue with teacher statuses")
    ver = [x for x in mine.values() if x["teacher_status"] == ot.VERIFIED and x["query"]["kind"] == "ORDERED_TEACHER" and x["query"]["sequence"]]
    ver = sorted(ver, key=lambda x: (x["query"]["role"] != "main", -len(x["query"]["sequence"])))[:4]
    frames = []
    for x in ver:
        w = witness(collect, x)
        seq = [tuple(a) for a in x["query"]["sequence"]]
        for q, g, pos, mv in trace_frames(robot, s, seq, w[1], w[2]):
            f, ax = plt.subplots(figsize=(5, 5))
            sd = next(a["seed"] for a in x["attempts"] if a["terminal"] == "WITNESS")
            _panel(ax, cfg, robot, s, goal, f"{x['query']['query'].split(':', 2)[2]}\nRECORDED witness, seed {sd}",
                   q=q, gap=g, pos=pos, held=mv)
            _tag(ax, "VERIFIED - recorded trajectory", COLORS["valid"])
            frames.append(_rgb(f))
    if not frames:
        f, ax = plt.subplots(figsize=(5, 5))
        _panel(ax, cfg, robot, s, goal, f"{k}: no VERIFIED nonempty repair")
        _tag(ax, "NO VERIFIED TRACE", COLORS["unresolved"])
        frames = [_rgb(f)] * STATIC_FRAMES
    gif(dst / "executed_repairs.gif", frames, f"{k}: recorded witness trajectories of VERIFIED repairs (main first, <= 4)")
    order_gif(dst, cfg, robot, s, goal, k, mine, collect)
    frames = []
    for q in r["queries"]["scene_queries"]:
        qid = f"scene:{k}:" + ".".join(f"{o}>{c}" for o, c in sorted(q["edit"].items()))
        rec = records.get(qid)
        se = ef.pose_substitution(robot, cfg, s, {"target_id": "t0"}, q["edit"])["scene"]
        tr = witness(collect, rec) if rec and rec["teacher_status"] == ot.VERIFIED else None
        steps = trace_frames(robot, se, [], [], tr[2]) if tr else [(None, None, None, None)] * STATIC_FRAMES
        for qq, g, pos, mv in steps:
            f, ax = plt.subplots(figsize=(5, 5))
            _panel(ax, cfg, robot, se, goal, f"edit {q['kind']}: {qid.split(':', 2)[2]}", q=qq, gap=g, pos=pos, held=mv)
            _tag(ax, "HYPOTHETICAL SCENE EDIT - NOT A ROBOT RELOCATION", "#34495e", loc=(0.02, 0.93), size=6)
            _tag(ax, _status_line(rec) if rec else "NOT IN PLAN", COL.get(rec and rec["teacher_status"], "#999"))
            frames.append(_rgb(f))
    gif(dst / "interventions.gif", frames, f"{k}: fixed-robot scene edits (target-only evaluation in the edited scene)")
    frames = []
    for p in pool:
        q = p["z_layout"]["query"].get("zR", f"seq:{k}:empty")
        f, ax = plt.subplots(figsize=(5, 5))
        _panel(ax, cfg, robot, s, goal, f"{p['role']}: {q.split(':', 2)[2]}  (K={p['z_layout']['active_count']})")
        _arrows(ax, s, [tuple(a) for a in p["sequence"]], COLORS["hypothetical"])
        _tag(ax, "PROPOSED, NOT EXECUTED", COLORS["hypothetical"], loc=(0.02, 0.93))
        st = records[q]["teacher_status"] if q in records else "NOT_RUN"
        _tag(ax, f"teacher: {st}", COL[st])
        frames += [_rgb(f)] * STATIC_FRAMES
    gif(dst / "candidate_orders.gif", frames, f"{k}: every predeclared candidate (arrows numbered by order) and its teacher status")


def order_gif(dst, cfg, robot, s, goal, k, mine, collect):
    seqs = {q: x for q, x in mine.items() if x["query"]["kind"] == "ORDERED_TEACHER" and len(x["query"]["sequence"]) >= 2}
    pairs = [(a, b) for a in sorted(seqs) for b in sorted(seqs) if a < b and
             sorted(map(tuple, seqs[a]["query"]["sequence"])) == sorted(map(tuple, seqs[b]["query"]["sequence"]))][:4]
    frames = []
    for a, b in pairs:
        side = []
        for q in (a, b):
            x = seqs[q]
            w = witness(collect, x) if x["teacher_status"] == ot.VERIFIED else None
            side.append(trace_frames(robot, s, [tuple(v) for v in x["query"]["sequence"]], w[1], w[2]) if w else None)
        n = max([len(v) for v in side if v] + [STATIC_FRAMES])
        for i in range(n):
            f, ax = plt.subplots(1, 2, figsize=(9, 4.8))
            for j, q in enumerate((a, b)):
                x, tr = seqs[q], side[j]
                if tr:
                    qq, g, pos, mv = tr[min(i, len(tr) - 1)]
                    _panel(ax[j], cfg, robot, s, goal, q.split(":", 2)[2], q=qq, gap=g, pos=pos, held=mv)
                    _tag(ax[j], "VERIFIED - recorded trajectory", COLORS["valid"])
                else:
                    _panel(ax[j], cfg, robot, s, goal, q.split(":", 2)[2])
                    _arrows(ax[j], s, [tuple(v) for v in x["query"]["sequence"]], COL[x["teacher_status"]])
                    _tag(ax[j], _status_line(x), COL[x["teacher_status"]], size=6)
            frames.append(_rgb(f))
    if not frames:
        f, ax = plt.subplots(figsize=(5, 5))
        _panel(ax, cfg, robot, s, goal, f"{k}: no same-endpoint order pair in the catalogue")
        frames = [_rgb(f)] * STATIC_FRAMES
    gif(dst / "order_comparison.gif", frames, f"{k}: same-endpoint order pairs side by side (recorded traces or static status panels)")


def run(args) -> int:
    out = Path(args.out)
    g = teacher.guard([sys.executable, "-c", "import sys, p6_viz; p6_viz.worker(*sys.argv[1:])", str(out), args.config, args.collect, args.train],
                      2000.0, 4000.0, out.parent / "viz_heartbeat.txt", 20.0, 0.5, Path("."))
    cf.write_json(out.parent / "viz_guard.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s", "overshoot_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def worker(out, config, collect, train) -> None:
    import p6_viz_models as vm
    out, collect, t0 = Path(out), Path(collect), time.process_time()
    cfg = cf.load_config(config)
    robot = ge.make_robot(cfg)
    plan = json.loads((collect / "plan.json").read_text())
    records = json.loads((collect / "decisions.json").read_text()) | json.loads((collect / "sealed_holdout.json").read_text())
    for k, r in plan["roots"].items():
        if r["variant"] == "obstructed":
            dst = out / "roots" / f"{k}_{r['family']}_{r['split']}"
            dst.mkdir(parents=True, exist_ok=True)
            root_visuals(dst, cfg, robot, plan, records, collect, k)
    if Path(train, "results.json").exists():
        vm.model_visuals(out / "models", cfg, robot, plan, records, collect, json.loads(Path(train, "results.json").read_text()))
    cf.write_json(out / "index.json", {"files": INDEX, "cpu_s": round(time.process_time() - t0, 1)})
    readme(out)


def readme(out) -> None:
    """visualizations/README.md: what every file shows, how to read the labels, SHA-256 of each file."""
    lines = ["# Visualizations", "", "Generated by `python run.py p6-viz` from recorded P6 data only (no new teacher calls).", "",
             "## How to read the labels", "",
             "- **VERIFIED - recorded trajectory**: the robot motion is replayed from the stored, independently re-checked witness arrays.",
             "- **CERTIFIED INFEASIBLE (kind)**: an accepted certificate; nothing was executed, so no motion is drawn.",
             "- **BUDGET_INFEASIBLE (operational) - NO VERIFIED TRACE**: all three OPNEG-v1 seeds ended cleanly without a witness; "
             "a policy-relative negative, not a proof.",
             "- **UNKNOWN / NO VERIFIED TRACE**: censored or unclean attempts; masked in training.",
             "- **HYPOTHETICAL SCENE EDIT - NOT A ROBOT RELOCATION**: a fixed-robot pose substitution evaluated target-only.",
             "- **PROPOSED, NOT EXECUTED** / **MODEL PROPOSALS - NOT EXECUTED**: candidate repairs (numbered arrows = order).",
             "- Dashed ring: goal cell. Dotted rings: predeclared destinations.", "", "## Files", "", "| file | what it shows | sha256 |",
             "|---|---|---|"]
    lines += [f"| `{Path(x['file']).relative_to(out.parent)}` | {x['what']} | `{x['sha256'][7:23]}...` |" for x in INDEX]
    (out.parent / "README.md").write_text("\n".join(lines) + "\n")
