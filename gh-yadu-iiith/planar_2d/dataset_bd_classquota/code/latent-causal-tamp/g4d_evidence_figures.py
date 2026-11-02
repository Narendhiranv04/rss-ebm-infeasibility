"""G4D evidence figures (report 0011), drawn only from saved, re-checked checkpoints and
their recorded trajectories. Every panel carries a written provenance and status tag:
NEW V2 (collected here), DERIVED LEGACY (revalidated G4A records), UNKNOWN or NOT RUN.
No path is interpolated; units without a recorded trajectory show none.

A  same endpoints under different orders (R4, R2)
B  baseline vs scene intervention at fixed robot state (R2, R5)
C  full repair vs omissions (R4) and the derived-legacy omission summary
D  actual joint continuity of the new multi-move witnesses
E  endpoint displacement vs measured joint travel (new V2; legacy diagnostic separately)
F  robot-start comparison (R1)
G  unit ledger: stop, label, CPU, wall, calls
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np

import data
import effects as ef
import g4d_figures as gf
import motion as mo
import repair_figures as rf
import viz
from scene import scene_from_dict
from viz import COLORS, plt

TAG = {data.POSITIVE: COLORS["valid"], data.NEGATIVE: COLORS["invalid"], data.UNKNOWN: COLORS["unresolved"],
       "NOT_RUN": "#7a7975"}
WORD = {data.POSITIVE: "POSITIVE", data.NEGATIVE: "NEGATIVE", data.UNKNOWN: "UNKNOWN", "NOT_RUN": "NOT RUN"}


def _arrays(path: Path) -> list:
    if not path.exists():
        return []
    z = np.load(path)
    n = 1 + max(int(k.split("__")[0][1:]) for k in z.files)
    return [{k.split("__")[1]: z[k] for k in z.files if k.startswith(f"r{i}__")} for i in range(n)]


def _udir(out, led, uid):
    st = led["units"].get(uid, {})
    return out / st["dir"] if "dir" in st else None


def _why(e) -> str:
    acc = [c["kind"] for c in e.get("certificates", []) if c["accepted"]]
    if acc:
        return "certificate: " + ", ".join(acc)
    r, t = e.get("relocation") or {}, e.get("target") or {}
    if r and not r.get("complete_witness"):
        return f"relocation {r.get('stop')}: {', '.join(r.get('normalized_reasons') or [])[:60]}"
    if t.get("witness_check"):
        return "target: goal + safe retreat re-checked" if t["witness_check"]["accepted"] else "target re-check failed"
    if t:
        return f"target {t.get('status')}: {t.get('normalized_reason') or t.get('reason')}"
    return f"stop {e['stop']}"


def unit_panel(ax, robot, cfg, plan, e, out, led, title, q=None):
    """Scene of a unit, its numbered proposal, recorded relocation and target paths."""
    root = plan["roots"][e["root"]]
    s = scene_from_dict(root["scene"])
    unit = next(u for u in plan["units"] if u["unit"] == e["unit"])
    if unit["kind"] == "SCENE_TARGET_ONLY":
        s = ef.pose_substitution(robot, cfg, s, {"target_id": "t0"}, unit["edit"])["scene"]
    qs = q if q is not None else unit.get("q_start", s.robot.q_rad)
    gf._panel(ax, robot, cfg, s, title, q=qs, alpha=0.6)
    gf._goal(ax, s, root["goal_cell"])
    seq = [tuple(a) for a in e["sequence"]]
    if seq:
        gf._seq(ax, s, seq, COLORS["hypothetical"])
    d = _udir(out, led, e["unit"])
    for k, a in enumerate(_arrays(d / "relocation.npz") if d else []):
        o = s.get(seq[k][0])
        p = np.array([mo.target_xy(robot, a, ph, qq, o.xy_m) for qq, _, ph in mo.samples(a, 120)])
        ax.plot(p[:, 0], p[:, 1], color=COLORS["hypothetical"], lw=1.1, zorder=7)
    tw = _arrays(d / "target_witness.npz") if d else []
    if tw:
        a = tw[0]
        p = np.array([mo.target_xy(robot, a, ph, qq, s.get("t0").xy_m) for qq, _, ph in mo.samples(a, 160)])
        ax.plot(p[:, 0], p[:, 1], color=COLORS["target"], lw=1.4, zorder=7)
        viz.draw_robot(ax, robot, a["q"][-1], float(a["gap"][-1]), alpha=0.3, zorder=5)
    lab = e["label"]
    ax.text(0.02, 0.02, f"NEW V2 | {WORD[lab]}", transform=ax.transAxes, fontsize=8, color="white", weight="bold",
            bbox=dict(fc=TAG[lab], ec="none", pad=3), zorder=20)
    rf._box(ax, textwrap.wrap(_why(e), 58) + [f"stop {e['stop']}  CPU {e.get('cpu_s')} s"], loc=(0.0, -0.08))


def _grid(n, cols=4, h=6.6):
    f, ax = plt.subplots((n + cols - 1) // cols, cols, figsize=(6 * cols, h * ((n + cols - 1) // cols)), squeeze=False)
    for a in ax.ravel()[n:]:
        a.axis("off")
    return f, ax.ravel()


def _save(f, path, bottom=0.12):
    f.tight_layout()
    f.subplots_adjust(bottom=bottom)
    f.savefig(path, dpi=75)
    plt.close(f)
    return path.name


def draw_all(out: Path, robot, cfg, plan, ev_, j, leg) -> list:
    led = __import__("json").loads((out / "ledger.json").read_text())
    names = []
    cont = j["blocking"]["B2_order_contrast"]["attempted"]
    f, ax = _grid(2 * len(cont), 4)
    for i, c in enumerate(cont):
        for k, s in enumerate(c["orders"]):
            uid = f"seq:{c['root']}:" + ".".join(f"{o}>{d}" for o, d in s)
            unit_panel(ax[2 * i + k], robot, cfg, plan, ev_[uid], out, led, f"{c['root']} order {k + 1}: {uid.split(':', 2)[2]}"
                       "\n(same endpoints, other order beside it)")
    names.append(_save(f, out / "A_order_contrast.png", 0.16))
    sc = [u for u in plan["units"] if u["kind"] == "SCENE_TARGET_ONLY"]
    refs = sorted({u["reference_unit"] for u in sc})
    f, ax = _grid(len(sc) + len(refs), 4)
    for i, uid in enumerate(refs + [u["unit"] for u in sc]):
        pair = next((x for x in j["blocking"]["B3_scene_effect_pair"]["pairs"] if x["unit"] == uid), None)
        t = (f"intervention {uid.split(':', 2)[2]} at the fixed initial robot state\nDelta = {pair['delta']} "
             f"({'decided' if pair['mask'] else 'masked'})") if pair else f"baseline {uid} (original scene, robot at q0)"
        unit_panel(ax[i], robot, cfg, plan, ev_[uid], out, led, t)
    names.append(_save(f, out / "B_scene_effects.png", 0.14))
    om = j["blocking"]["B4_omission_pair"]["new_v2"]
    f, ax = _grid(len(om) + 2, 3)
    full = om[0]["sequence"] if om else []
    unit_panel(ax[0], robot, cfg, plan, ev_["seq:R4:" + ".".join(f"{o}>{d}" for o, d in full)], out, led,
               f"full repair r = {full}")
    for i, o in enumerate(om, start=1):
        uid = "seq:R4:" + (".".join(f"{a}>{b}" for a, b in o["remaining"]) or "empty")
        unit_panel(ax[i], robot, cfg, plan, ev_[uid], out, led, f"omit {o['omitted']} -> {o['remaining']} (omitted stay)\n"
                   f"Delta = Y(r) - Y(r\\S) = {o['delta']} ({'decided' if o['mask'] else 'masked'})")
    s = leg["omission"]["summary"]
    ax[len(om) + 1].axis("off")
    ax[len(om) + 1].text(0, 1, "\n".join(["DERIVED LEGACY omission pairs (G4A, revalidated)", f"enumerated {s['pairs_enumerated']}",
                                          f"decided {s['decided']}: +1 x{s['decided_delta_plus_one']}, 0 x{s['decided_delta_zero']}",
                                          f"masked (unknown side) {s['masked_unknown']}",
                                          f"roots with a decided +1: {len(s['unique_roots_with_decided_plus_one'])}",
                                          "not independent examples; V1 backend"]), va="top", fontsize=10, family="monospace")
    names.append(_save(f, out / "C_omissions.png", 0.14))
    wit = [e for e in ev_.values() if len(e["sequence"]) >= 2 and e["relocation"] and e["relocation"]["complete_witness"]]
    f, ax = _grid(max(1, len(wit)), 2, 5.5)
    for a_, e in zip(ax, wit):
        d = _udir(out, led, e["unit"])
        recs, tw, x0 = _arrays(d / "relocation.npz"), _arrays(d / "target_witness.npz"), 0
        for i, r in enumerate(recs + tw):
            xs = np.arange(len(r["q"])) + x0
            for jn, col in enumerate(("#1b4f72", "#2a78d6", "#85c1e9")):
                a_.plot(xs, r["q"][:, jn], color=col, lw=1.4 if i < len(recs) else 0.9, ls="-" if i < len(recs) else ":",
                        label=f"q{jn}" if i == 0 else None)
            x0 += len(r["q"]) - 1
            if i < len(recs + tw) - 1:
                a_.axvline(x0, color=COLORS["muted"], ls="--", lw=0.8)
        a_.set(xlabel="waypoint index (relocations solid; target dotted, excluded from J)", ylabel="joint angle [rad]")
        a_.set_title(f"NEW V2 {e['unit']}: J = {e['cost']['recomputed_J']:.3f}; continuity {e['checks'].get('joint_continuity')}"
                     f", target start = achieved {e['checks'].get('target_start_equality')}\ndashed line = record boundary "
                     "(end q == next start q, bit-exact)", fontsize=9, loc="left")
        a_.legend(fontsize=8, loc="lower right")
    if not wit:
        ax[0].text(0.5, 0.5, "no new complete multi-move relocation witness", ha="center", transform=ax[0].transAxes)
    names.append(_save(f, out / "D_joint_continuity.png", 0.1))
    f, ax = plt.subplots(1, 2, figsize=(15, 5.5))
    for c in j["cases"]["D-V2-5"]["cost_labels"]:
        ax[0].plot(c["endpoint_distance"], c["J"], "o", ms=9, color=COLORS["hypothetical"])
        ax[0].annotate(f"{c['root']} {c['sequence']}", (c["endpoint_distance"], c["J"]), fontsize=8, xytext=(5, 4),
                       textcoords="offset points")
    ax[0].set(xlabel="endpoint displacement C_d (dimensionless)", ylabel="measured joint travel J (first V2 witness)",
              title="NEW V2 cost labels (FIRST_RELOCATION_WITNESS_NORMALIZED_JOINT_TRAVEL_V1)")
    lr = __import__("json").loads((out / "legacy_revalidation.json").read_text())["rows"]
    pts = [(r["recorded_sequence"], r["legacy_cost"]["J"], r["query_id"]) for r in lr if r.get("legacy_cost") and r["moved"]]
    import sequences as sq
    plan4a = {q["query_id"]: q for q in __import__("json").loads(Path("runs/g4a_evidence_audit/g4a_plan.json").read_text())["queries"]}
    for seq, J, q in pts:
        ax[1].plot(sq.endpoint_distance(scene_from_dict(plan4a[q]["scene"]), [tuple(a) for a in seq]), J, "x", color="#7a7975")
    ax[1].set(xlabel="endpoint displacement C_d", ylabel="legacy joint travel (V1 backend)",
              title="DERIVED LEGACY diagnostic only: not V2 cost labels")
    for a in ax:
        a.title.set_fontsize(9)
    names.append(_save(f, out / "E_endpoint_vs_joint_travel.png", 0.12))
    f, ax = _grid(2, 2)
    unit_panel(ax[0], robot, cfg, plan, ev_["seq:R1:empty"], out, led, "R1 empty repair, robot at home q0")
    st = next(u for u in plan["units"] if u["unit"] == "start:R1:0")
    unit_panel(ax[1], robot, cfg, plan, ev_["start:R1:0"], out, led,
               f"R1 empty repair, alternative start q = {[round(v, 3) for v in st['q_start']]}\n(identical object geometry)")
    names.append(_save(f, out / "F_robot_start.png", 0.16))
    f, a = plt.subplots(figsize=(17, 0.42 * len(ev_) + 1.2))
    a.axis("off")
    rows = [[k, e["stop"], WORD.get(e["label"], e["label"]), e.get("cpu_s"), e.get("wall_s"),
             (e.get("calls") or {}).get("relocation_planner"), (e.get("calls") or {}).get("target_evaluations"),
             _why(e)[:78]] for k, e in ev_.items()]
    t = a.table(cellText=rows, colLabels=["unit", "stop", "label", "CPU s", "wall s", "reloc calls", "target calls", "why"],
                loc="upper left", cellLoc="left")
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.auto_set_column_width(list(range(8)))
    for (r, c), cell in t.get_celld().items():
        if r and c == 2:
            cell.set_facecolor(TAG.get({v: k for k, v in WORD.items()}.get(cell.get_text().get_text()), "white"))
            cell.get_text().set_color("white")
    names.append(_save(f, out / "G_unit_ledger.png", 0.02))
    return names
