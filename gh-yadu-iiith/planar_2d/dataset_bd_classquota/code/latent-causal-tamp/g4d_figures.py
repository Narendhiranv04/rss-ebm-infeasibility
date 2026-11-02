"""G4D figures (from the frozen plan, the legacy audit and archived G4A trajectories).

A  canonical sequences, ghosts, ideal prefixes and the anchored learner input (R4).
B  explanation operations: scene substitution at fixed robot, omission, deletion (R2, R4, R5).
C  a validated legacy multi-move witness: recorded relocations, achieved states, target,
   joint continuity across records and legacy joint travel (G4A archive, kinematic only).
D  the frozen sparse G4D plan: every case on its real scene, with written NOT RUN status.
No figure draws an interpolated or invented path; planned items carry no outcome.
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle

import data
import effects as ef
import motion as mo
import repair_figures as rf
import sequences as sq
import viz
from scene import scene_from_dict
from viz import COLORS, plt

PLANNED = "#7a7975"


def _scene(plan, k):
    return scene_from_dict(plan["roots"][k]["scene"])


def _goal(ax, scene, cell):
    xy = scene.lattice.center(cell)
    ax.add_patch(Circle(xy, 0.024, fc="none", ec=COLORS["target_edge"], ls="--", lw=1.5, zorder=9))
    ax.text(xy[0], xy[1] + 0.031, "GOAL", fontsize=6, ha="center", color=COLORS["target_edge"], weight="bold", zorder=10)


def _seq(ax, scene, seq, color, dy=0.0):
    """Ghost (dashed circle at the proposed pose of the SAME object) and a numbered arrow."""
    for t, (o, c) in enumerate(seq):
        a, b = np.array(scene.get(o).xy_m), np.array(scene.lattice.center(c))
        ax.add_patch(Circle(b, 0.024, fc="none", ec=color, ls="--", lw=1.4, zorder=8))
        rf._arrow(ax, a, b, color, ls="--", lw=1.4)
        m = (a + b) / 2 + np.array([0.0, dy])
        ax.text(*m, str(t + 1), fontsize=7, color="white", weight="bold", ha="center", va="center", zorder=12,
                bbox=dict(boxstyle="circle,pad=0.15", fc=color, ec="none"))


def _panel(ax, robot, cfg, scene, title, q=None, alpha=0.85, board=False):
    viz.scene_panel(ax, scene, cfg, title)
    viz.draw_robot(ax, robot, q if q is not None else scene.robot.q_rad, scene.robot.gripper_gap_m, alpha=alpha, zorder=6)
    rf._ids(ax, scene)
    lim = ((-0.02, 0.66), (-0.02, 0.66)) if board else viz.LIMITS["grid"]     # grid view: support, base, full arm
    ax.set(xlim=lim[0], ylim=lim[1])


def _tag(ax, text, color=PLANNED, loc=(0.02, 0.02)):
    ax.text(*loc, text, transform=ax.transAxes, fontsize=7.5, color="white", weight="bold", va="bottom",
            bbox=dict(fc=color, ec="none", pad=3), zorder=20)


def fig_sequences(out, robot, cfg, v2, plan):
    s, act = _scene(plan, "R4"), {"target_id": "t0"}
    good, bad = (("o01", 18), ("o00", 29)), (("o00", 29), ("o01", 18))
    f, ax = plt.subplots(1, 4, figsize=(24, 7.4))
    _panel(ax[0], robot, cfg, s, "R4 initial scene S0 and two canonical sequences\n(same endpoints, different order)")
    _goal(ax[0], s, 28)
    _seq(ax[0], s, good, COLORS["hypothetical"], 0.012)
    _seq(ax[0], s, bad, "#8e44ad", -0.012)
    rf._box(ax[0], ["blue  r1 = (o01->18, o00->29)", "purple r2 = (o00->29, o01->18)",
                    f"x(r1) = x(r2) = {sq.assignment(good)}", f"C_d = {sq.endpoint_distance(s, good):.4f} for both",
                    "ordered signatures differ"], loc=(0.0, -0.08))
    pre = sq.prefix_scenes(s, good)
    for i, title in ((1, "ideal prefix before action 2 of r1\n(o01 at its proposed pose; nothing else moved)"),
                     (2, "final endpoint scene of r1\n(hypothetical geometry, no robot state)")):
        viz.scene_panel(ax[i], pre[i], cfg, title)
        rf._ids(ax[i], pre[i])
        ax[i].set(xlim=(-0.02, 0.66), ylim=(-0.02, 0.66))
        for o, _ in good[:i]:
            ax[i].add_patch(Circle(s.get(o).xy_m, 0.024, fc="none", ec=COLORS["muted"], ls=":", lw=1.2, zorder=4))
            ax[i].text(*s.get(o).xy_m, f"{o}\nvacated", fontsize=5.5, ha="center", va="center", color=COLORS["muted"])
        _goal(ax[i], s, 28)
        _tag(ax[i], "HYPOTHETICAL PREFIX (not an execution witness)")
    anchor = data.appearance_anchor({"learner": {"scene": data.learner_scene(s), "action": {
        "robot_profile_hash": s.robot_profile_hash}}}, 320)
    ax[3].imshow(anchor["rgb"])
    ax[3].contour(anchor["instance"] > 0, levels=[0.5], colors=[COLORS["hypothetical"]], linewidths=0.8)
    ax[3].set_title("learner appearance anchor: clean initial RGB + instance masks\n"
                    "(reused unchanged for every prefix, intervention and omission)", fontsize=9, loc="left")
    ax[3].set(xticks=[], yticks=[])
    rej = [("(o00->29, o00->30)", "duplicate id"), ("(o00->28)", "stay is not an action"),
           ("(t0->29)", "active target"), ("(o00->20)", "target's resting cell"), ("4 actions", "above move cap 3")]
    rf._box(ax[3], ["rejected before ranking:"] + [f" {a:20s} {b}" for a, b in rej], loc=(0.0, -0.04))
    f.tight_layout()
    f.subplots_adjust(bottom=0.25, top=0.86)
    f.savefig(out / "A_canonical_sequences.png", dpi=85)
    plt.close(f)


def fig_explanations(out, robot, cfg, v2, plan):
    s2, s4, s5 = _scene(plan, "R2"), _scene(plan, "R4"), _scene(plan, "R5")
    act2 = {"target_id": "t0", "goal_xy_m": list(s2.lattice.center(45))}
    sub = ef.pose_substitution(robot, cfg, s2, act2, {"o03": 60})
    f, ax = plt.subplots(1, 4, figsize=(24, 7.6))
    _panel(ax[0], robot, cfg, s2, "scene effect, baseline c0 (R2): o03 on the goal")
    _goal(ax[0], s2, 45)
    _tag(ax[0], "target-only outcome: NOT RUN")
    _panel(ax[1], robot, cfg, sub["scene"], "scene effect, intervention c0^I: o03 -> 60\nsame robot q, gap, attachment")
    ax[1].add_patch(Circle(s2.get("o03").xy_m, 0.024, fc="none", ec=COLORS["muted"], ls=":", lw=1.2, zorder=4))
    _goal(ax[1], s2, 45)
    rf._box(ax[1], [f"eligible: {sub['eligible']}", f"changed: {sub['changed_fields']['changed_objects']}",
                    f"robot unchanged: {sub['changed_fields']['robot_unchanged']}",
                    "Delta = Y(c0^I) - Y(c0): masked until both sides decided"], loc=(0.0, -0.08))
    _tag(ax[1], "target-only outcome: NOT RUN")
    good = (("o01", 18), ("o00", 29))
    _panel(ax[2], robot, cfg, s4, "omission within r = (o01->18, o00->29) (R4)\nr \\ {o01} = (o00->29); o01 stays")
    _goal(ax[2], s4, 28)
    _seq(ax[2], s4, sq.omit(good, [0]), COLORS["hypothetical"])
    ax[2].add_patch(Circle(s4.get("o01").xy_m, 0.03, fc="none", ec=COLORS["valid"], lw=1.6, zorder=8))
    rf._box(ax[2], ["planned omissions (budget: subsets <= 2):"] + [f" omit {o['omitted_objects']} -> {o['remaining']}"
                                                                    for o in ef.omission_plan(good, v2)]
            + ["omitted objects keep their initial pose", "Delta = Y(r) - Y(r \\ S); masked if either unknown"], loc=(0.0, -0.08))
    _tag(ax[2], "ordered outcomes: NOT RUN")
    u = [x for x in plan["units"] if x["unit"].startswith("scene:R5")]
    _panel(ax[3], robot, cfg, s5, "R5 joint vs singleton substitutions (frozen rule:\nrestore G2_D2 blockers singly and jointly)")
    for x in u:
        for o, c in x["edit"].items():
            ax[3].add_patch(Circle(s5.lattice.center(c), 0.024, fc="none", ec=COLORS["hypothetical"], ls="--", zorder=8))
            rf._arrow(ax[3], s5.get(o).xy_m, s5.lattice.center(c), COLORS["hypothetical"], ls="--")
    _goal(ax[3], s5, 53)
    rf._box(ax[3], [f"{x['unit']}: eligible {x['eligible']}" for x in u]
            + ["deletion (removing objects) is a separate", "DIAGNOSTIC_ONLY operation, never a label"], loc=(0.0, -0.08))
    _tag(ax[3], "target-only outcomes: NOT RUN")
    f.tight_layout()
    f.subplots_adjust(bottom=0.27)
    f.savefig(out / "B_explanation_operations.png", dpi=85)
    plt.close(f)


def fig_legacy(out, robot, cfg, legacy, root: Path):
    g4a = root / "runs/g4a_evidence_audit"
    rows = [r for r in legacy["rows"] if r["v2_status"] == data.POSITIVE and r["moved"] >= 2]
    r = max(rows, key=lambda x: (x["moved"], -x["legacy_cost"]["J"]))
    tag = f"{r['query_id']}.audit{r['audit_index']}"
    scenes, npz = json.loads((g4a / "scenes.json").read_text()), np.load(g4a / "arrays.npz")
    arr = lambda t: {k.split("__")[1]: npz[k] for k in npz.files if k.startswith(t + "__")}  # noqa: E731
    ms = [f"{tag}.m{k}" for k in range(r["moved"])]
    s0 = scene_from_dict(scenes[ms[0]])
    f, ax = plt.subplots(1, 3, figsize=(23, 7.6), gridspec_kw={"width_ratios": [1, 1, 1.4]})
    _panel(ax[0], robot, cfg, s0, f"{r['query_id']} S0 and the recorded order (validated legacy witness)", alpha=0.5)
    seq = [tuple(a) for a in r["recorded_sequence"]]
    _seq(ax[0], s0, seq, COLORS["hypothetical"])
    goal = scene_from_dict(scenes[tag + ".target"])
    for k, m in enumerate(ms):
        a = arr(m)
        obj = np.array([mo.target_xy(robot, a, p, q, s0.get(seq[k][0]).xy_m) for q, _, p in mo.samples(a, 120)])
        ax[0].plot(obj[:, 0], obj[:, 1], color=COLORS["hypothetical"], lw=1.0, alpha=0.7, zorder=7)
    rf._box(ax[0], [f"recorded sequence: {r['recorded_sequence']}", f"V2 status: {r['v2_status']} (this order only)",
                    "other orders of these endpoints: UNRESOLVED"], loc=(0.0, -0.08))
    t = arr(tag + ".target")
    _panel(ax[1], robot, cfg, goal, "achieved state after the last relocation, and the\nrecorded target action from it",
           q=t["q"][0], alpha=0.85)
    tp = np.array([mo.target_xy(robot, t, p, q, goal.get("t0").xy_m) for q, _, p in mo.samples(t, 160)])
    ax[1].plot(tp[:, 0], tp[:, 1], color=COLORS["target"], lw=1.4, zorder=7)
    viz.draw_robot(ax[1], robot, t["q"][-1], float(t["gap"][-1]), alpha=0.35, zorder=5)
    v = r["validation"]["target"]
    rf._box(ax[1], [f"target starts at achieved state: {v['starts_at_archived_state']}",
                    f"ends in safe retreat: {v['ends_in_safe_retreat']}  goal within tol: {v['goal_within_tolerance']}",
                    f"certified clear: {v['certified_clear']}  hash matches: {v['hash']}"], loc=(0.0, -0.08))
    viz.status_tag(ax[1], data.POSITIVE, "replay-validated (kinematic)")
    qs, b, x0 = [], [], 0
    for m in ms + [tag + ".target"]:
        q = arr(m)["q"]
        qs.append((np.arange(len(q)) + x0, q))
        x0 += len(q) - 1
        b.append(x0)
    for j, c in enumerate(("#1b4f72", "#2a78d6", "#85c1e9")):
        for i, (xs, q) in enumerate(qs):
            ax[2].plot(xs, q[:, j], color=c, lw=1.4 if i < len(ms) else 0.9, ls="-" if i < len(ms) else ":",
                       label=f"q{j}" if i == 0 else None)
    for i, xb in enumerate(b[:-1]):
        ax[2].axvline(xb, color=COLORS["muted"], lw=0.8, ls="--")
        ax[2].text(xb, ax[2].get_ylim()[1] if i else 3.3, " record boundary\n end q == next start q", fontsize=6.5, va="top")
    ax[2].set(xlabel="waypoint index (concatenated records; dotted = target, excluded from J)", ylabel="joint angle [rad]",
              ylim=(-3.3, 3.5))
    ax[2].legend(fontsize=8, loc="lower right")
    lc = r["legacy_cost"]
    ax[2].set_title(f"joint continuity across relocations; legacy joint travel J = {lc['J']:.3f} "
                    f"(per record {[round(x, 3) for x in lc['per_record']]})\npolicy {lc['policy']}: NOT a V2 cost label",
                    fontsize=9, loc="left")
    f.tight_layout()
    f.subplots_adjust(bottom=0.24)
    f.savefig(out / "C_legacy_witness_continuity_cost.png", dpi=85)
    plt.close(f)
    return {"query_id": r["query_id"], "audit_index": r["audit_index"], "sequence": r["recorded_sequence"], "J": lc["J"]}


def fig_plan(out, robot, cfg, plan, legacy, root: Path):
    f, axes = plt.subplots(2, 4, figsize=(26, 15.5))
    ax = axes.ravel()
    units = {u["unit"]: u for u in plan["units"]}
    scenes = json.loads((root / "runs/g4a_evidence_audit/scenes.json").read_text())
    r05 = next(r for r in legacy["rows"] if r["query_id"] == "g4a_r05" and r["moved"] == 0)
    s05 = scene_from_dict(scenes[f"g4a_r05.audit{r05['audit_index']}.target"])
    _panel(ax[0], robot, cfg, s05, "D-V2-0 empty repair: legacy g4a_r05 (validated)\n+ planned unit seq:R0:empty")
    g4a = json.loads((root / "runs/g4a_evidence_audit/g4a_plan.json").read_text())
    _goal(ax[0], s05, next(q["goal_cell"] for q in g4a["queries"] if q["query_id"] == "g4a_r05"))
    viz.status_tag(ax[0], data.POSITIVE if r05["v2_status"] == data.POSITIVE else data.UNKNOWN,
                   "legacy witness replay-validated; J = 0 exactly")
    for i, c in enumerate(plan["cases"][1:], start=1):
        ks = sorted({units[u]["root"] for u in c["units"]} | {u.split(":")[1] for u in units if c["id"] in units[u]["cases"]})
        k = "R2" if "R2" in ks else ks[0]
        s = _scene(plan, k)
        _panel(ax[i], robot, cfg, s, f"{c['id']} on {k} ({plan['roots'][k]['fixture']})\n" + textwrap.fill(c["requirement"], 58))
        _goal(ax[i], s, plan["roots"][k]["goal_cell"])
        cols = [COLORS["hypothetical"], "#8e44ad", "#16a085", "#c0392b"]
        mine = [units[u] for u in units if c["id"] in units[u]["cases"] and units[u]["root"] == k]
        lines = []
        for j, u in enumerate(x for x in mine if x["kind"] == "ORDERED_TEACHER"):
            if u["sequence"]:
                _seq(ax[i], s, [tuple(a) for a in u["sequence"]], cols[j % 4], 0.012 * (j - 1))
            lines.append(f"{u['unit']}")
        for u in (x for x in mine if x["kind"] != "ORDERED_TEACHER"):
            lines.append(f"{u['unit']} eligible={u.get('eligible', True)}")
            if u["kind"] == "ROBOT_START_TARGET_ONLY":
                viz.draw_robot(ax[i], robot, u["q_start"], s.robot.gripper_gap_m, alpha=0.4, zorder=5)
        others = sorted({units[u]["root"] for u in units if c["id"] in units[u]["cases"]} - {k})
        rf._box(ax[i], lines[:9] + ([f"+ units on {others}"] if others else []), loc=(0.0, -0.06))
        _tag(ax[i], "PLANNED - NOT RUN (budget not authorized)")
    b = plan["budget"]
    ax[7].axis("off")
    ax[7].text(0, 1, "\n".join(["G4D sparse plan (frozen, outcome-free)", f"plan status: {plan['status']}",
                                f"units {b['work_upper_bound']['units']}, relocation skills <= "
                                f"{b['work_upper_bound']['relocation_skills']}, target evaluations <= "
                                f"{b['work_upper_bound']['target_evaluations']}",
                                "per-unit CPU cap: UNSET", "aggregate CPU cap: UNSET", "",
                                "numbered arrow = action order; dashed circle = ghost",
                                "(proposed pose of the SAME object); faint arm = alternative start", "", "blocking acceptance:"]
                               + [" - " + textwrap.fill(x, 52, subsequent_indent="   ") for x in plan["acceptance"]["blocking"]]),
               va="top", fontsize=9, family="monospace")
    f.tight_layout(h_pad=6)
    f.subplots_adjust(bottom=0.08)
    f.savefig(out / "D_sparse_plan_cases.png", dpi=80)
    plt.close(f)


def draw_all(out: Path, robot, cfg, v2, plan, legacy) -> dict:
    root = Path(".")
    fig_sequences(out, robot, cfg, v2, plan)
    fig_explanations(out, robot, cfg, v2, plan)
    c = fig_legacy(out, robot, cfg, legacy, root)
    fig_plan(out, robot, cfg, plan, legacy, root)
    return {"legacy_example": c}
