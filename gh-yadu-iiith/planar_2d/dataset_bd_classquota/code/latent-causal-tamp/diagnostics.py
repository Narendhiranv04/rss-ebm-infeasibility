"""Figures that explain verification verdicts (Guide v1.6 §3; reports 0003, 0004).

Split from figures.py (size): rejected poses with their offending pair, segments
with clear endpoints but a certified interior collision, the link-0 sweep
certificate, a budget-exhausted search (UNRESOLVED, not a proof), the clutter
candidate verdicts, and the IK-tracking reclassification of report 0004.
"""
from __future__ import annotations

import math

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Wedge

import geometry as ge
import motion as mo
import planner as pn
import viz
from scene import Lattice
from viz import COLORS, plt, scene_panel, status_tag


def _pair_bodies(pair: str) -> tuple[str, ...]:
    return tuple(pair.split("|")) if pair else ()


def _obstacle_ring(ax, scene, pair: str):
    for o in scene.objects:
        if o.id in _pair_bodies(pair):
            ax.add_patch(Circle(o.xy_m, o.radius_m + 0.006, fc="none", ec=COLORS["invalid"], lw=2, zorder=9))


def diagnostics_figure(robot, cfg, ex: dict):
    """A: rejected candidate pose; B: segment with clear endpoints but a certified
    interior collision; C: link-0 sweep certificate; D: planner budget exhausted."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 14))
    for ax, key in zip(axes.flat, ("pose", "segment", "sweep", "budget")):
        if ex.get(key) is None:
            ax.text(0.5, 0.5, f"no '{key}' example produced in this run", ha="center", transform=ax.transAxes)
    if ex.get("pose") is None:
        return fig
    sc_, q, gap, v = ex["pose"]
    ax = axes[0, 0]
    scene_panel(ax, sc_, cfg, f"A. Rejected grasp pose: {v.pair}  d = {v.distance * 1000:.1f} mm < delta")
    viz.draw_robot(ax, robot, q, gap, highlight=_pair_bodies(v.pair), zorder=8)
    _obstacle_ring(ax, sc_, v.pair)
    status_tag(ax, "CERTIFIED_INFEASIBLE", "definite collision at this pose")
    if ex.get("segment") is None:
        return fig
    sc_, qa, qb, gap, v = ex["segment"]
    ax = axes[0, 1]
    scene_panel(ax, sc_, cfg, f"B. Joint-linear segment: both ends clear, collision at s = {v.s:.3f}")
    for qq, a in ((qa, 0.25), (qb, 0.25)):
        viz.draw_robot(ax, robot, qq, gap, alpha=a, zorder=7)
    viz.draw_robot(ax, robot, qa + v.s * (qb - qa), gap, highlight=_pair_bodies(v.pair), zorder=8)
    _obstacle_ring(ax, sc_, v.pair)
    status_tag(ax, "CERTIFIED_INFEASIBLE", f"{v.pair} (found by interval certifier, {v.evals} evals)")
    if ex.get("sweep") is None:
        return fig
    sc_, qa, qb, pl = ex["sweep"]
    ax = axes[1, 0]
    scene_panel(ax, sc_, cfg, f"C. Link-0 sweep certificate: {pl.reason.split(':', 1)[1]}")
    lo, hi = sorted((qa[0], qb[0]))
    ax.add_patch(Wedge(robot.base, robot.lengths[0], math.degrees(lo), math.degrees(hi),
                       fc=COLORS["invalid"], alpha=0.15, zorder=2))
    for qq in (qa, qb):
        viz.draw_robot(ax, robot, qq, robot.open_gap, alpha=0.45, zorder=7)
    _obstacle_ring(ax, sc_, pl.reason)
    status_tag(ax, "CERTIFIED_INFEASIBLE", "every path sweeps link 0 through this disc")
    if ex.get("budget") is None:
        return fig
    sc_, qa, qb, pl = ex["budget"]
    ax = axes[1, 1]
    scene_panel(ax, sc_, cfg, f"D. {pl.nodes}-node search ({pl.seconds:.0f} s): no path; dots = tree vertices")
    ax.plot(pl.tree_xy[:, 0], pl.tree_xy[:, 1], ".", ms=1.5, color=COLORS["hypothetical"], alpha=0.5, zorder=4)
    for qq in (qa, qb):
        viz.draw_robot(ax, robot, qq, robot.open_gap, alpha=0.45, zorder=7)
    status_tag(ax, "UNRESOLVED", "budget exhausted: NOT a proof of infeasibility")
    ax.text(0.02, 0.1, pl.reason, transform=ax.transAxes, fontsize=7, wrap=True, zorder=20,
            bbox=dict(fc="white", ec=COLORS["muted"], pad=2))
    return fig


def candidates_panel(ax, robot, scene, cfg, result, title: str):
    """Every declared pick candidate as a short ray at the target, coloured by its
    first failed check; the closest-to-home OK candidate's pose is drawn."""
    t = next(o for o in scene.objects if o.kind == "target")
    scene_panel(ax, scene, cfg, title)
    color = {"OK": COLORS["valid"]}
    for c in result.picks:
        u = np.array([math.cos(c.phi), math.sin(c.phi)])
        tone = color.get(c.stage, COLORS["invalid"] if c.certified else COLORS["unresolved"])
        a = np.asarray(t.xy_m) - (0.03 + 0.012 * (c.branch > 0)) * u
        ax.add_patch(FancyArrowPatch(a - 0.05 * u, a, arrowstyle="-|>", mutation_scale=6, lw=0.8,
                                     color=tone, zorder=6))
    viz.draw_robot(ax, robot, robot.q_home, robot.open_gap, alpha=0.6, zorder=7)
    ok = [c for c in result.picks if c.stage == "OK"]
    if ok:
        viz.draw_robot(ax, robot, ok[0].q_end, robot.open_gap, alpha=0.35, zorder=7)


def rejection_panel(ax, robot, scene, cfg, result):
    """Representative rejected pick poses (one per stage, offending pair in red) and
    the count of every stage / most frequent offending pair, for a failed query."""
    from collections import Counter
    scene_panel(ax, scene, cfg, "B. Why no pick survives: representative rejected poses")
    shown = set()
    for c in result.picks:
        if c.q_end is not None and c.stage not in shown and c.stage != "OK":
            shown.add(c.stage)
            viz.draw_robot(ax, robot, c.q_end, robot.open_gap, alpha=0.55, zorder=7,
                           highlight=_pair_bodies(c.reason))
            _obstacle_ring(ax, scene, c.reason)
    stages = Counter(c.stage for c in result.picks)
    pairs = Counter(c.reason for c in result.picks if c.reason)
    lines = [f"{s}: {n}" for s, n in stages.most_common()] + ["most frequent offending pairs:"]
    lines += [f"  {pr}: {n}" for pr, n in pairs.most_common(6)]
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7,
            family="monospace", bbox=dict(fc="white", ec=COLORS["muted"]), zorder=20)
    status_tag(ax, result.status, result.reason)


def diagnostic_examples(robot, cfg, done: dict) -> dict:
    """First real example of each rejection kind, preferring the clutter scene."""
    order = sorted(done, key=lambda k: (not k.startswith("G1_CLUTTER"), not k.startswith("G1_S2"), k))
    ex = {"pose": None, "segment": None, "sweep": None, "budget": None}
    delta, eps = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"]
    for tid in order:
        s, res, _ = done[tid]
        for c in res.picks:
            if ex["pose"] is None and c.stage == "GRASP_POSE" and c.certified:
                ck = mo.checkers(robot, s, "t0", None, None, delta, eps)[("start", False)]
                ex["pose"] = (s, c.q_end, robot.open_gap, ck.status_at(c.q_end, robot.open_gap))
        for label, pl in res.plans:
            d = pl.direct
            if ex["segment"] is None and label.startswith("pregrasp") and d and d.status == ge.COLLISION \
                    and 0 < d.s < 1:
                ex["segment"] = (s, *pl.ends, robot.open_gap, d)
            if ex["sweep"] is None and pl.reason.startswith("LINK0_SWEEP") and label.startswith("pregrasp"):
                ex["sweep"] = (s, *pl.ends, pl)
            if ex["budget"] is None and pl.reason.startswith("BUDGET_EXHAUSTED"):
                ex["budget"] = (s, *pl.ends, pl)
    if ex["budget"] is None:  # none occurred: rerun a solved RRT query with a deliberately tiny budget
        for tid in order:
            s, res, _ = done[tid]
            hit = next((pl for lb, pl in res.plans if lb.startswith("pregrasp") and pl.reason == "RRT_JOINT"), None)
            if hit:
                ck = mo.checkers(robot, s, "t0", None, None, delta, eps)[("start", False)]
                small = {**cfg["planner2d"], "max_nodes": 40}
                pl = pn.plan(ck, *hit.ends, robot.open_gap, small, cfg["geometry"]["certify_max_evals"], "demo")
                pl.reason += f" (node budget cut to 40 for illustration; full budget: {hit.reason}, VERIFIED)"
                ex["budget"] = (s, *hit.ends, pl)
                break
    return ex


def ik_examples(robot, cfg, done: dict) -> list:
    """First audit pick candidate of each kind whose approach classification depends
    on refinement: (cell, phi, branch, old fixed-step reason, refined reason)."""
    sk_ = cfg["skill2d"]
    out, seen = [], set()
    for tid in sorted(k for k in done if k.startswith("audit_")):
        s, res, _ = done[tid]
        xy = s.get("t0").xy_m
        for c in (c for c in res.picks if c.stage in ("APPROACH", "OK")):
            args = (robot, xy, (robot.grasp_x, 0.0), c.phi, c.branch, sk_["approach_dist_m"], 0.0,
                    sk_["line_step_m"], sk_["max_ik_jump_rad"])
            old = mo.ik_line(*args, depth=0)[1]           # the pre-0004 fixed-step tracker
            new = mo.ik_line(*args)[1]
            if old and old != (new or "OK") and (old, new) not in seen:     # reclassified only
                seen.add((old, new))
                out.append((int(tid[-2:]), c.phi, c.branch, old, new or "OK"))
    return out


def ik_tracking_figure(robot, cfg, examples: list):
    """Declared straight approaches whose label changed in report 0004. Top: joint
    angles of the elbow branch sampled finely (gaps = no IK) and at the 13 fixed
    5 mm samples the pre-0004 tracker used. Bottom: that tracker's largest joint
    step per sample gap against max_ik_jump_rad, the quantity its label rested on."""
    fig, axes = plt.subplots(2, len(examples), figsize=(6 * len(examples), 7.5), squeeze=False)
    sk_, lat = cfg["skill2d"], viz_lattice(cfg)
    for k, (cell, phi, branch, old, new) in enumerate(examples):
        u, xy = np.array([math.cos(phi), math.sin(phi)]), np.asarray(lat.center(cell))
        top, bottom = axes[0, k], axes[1, k]
        for n in (1200, round(sk_["approach_dist_m"] / sk_["line_step_m"])):
            d = np.linspace(sk_["approach_dist_m"], 0.0, n + 1)
            q = np.array([next((s for b, s in ge.ik(robot, xy - (robot.grasp_x + e) * u, phi) if b == branch),
                               np.full(3, np.nan)) for e in d])
            for j, color in enumerate((COLORS["link"], COLORS["hypothetical"], COLORS["target"])):
                top.plot(d, q[:, j], "-" if n > 100 else "o", color=color, lw=1.2, ms=4,
                         label=f"q{j + 1}" + (" fine" if n > 100 else " 5 mm samples"))
            if n < 100:
                step = np.nanmax(np.abs(np.diff(q, axis=0)), axis=1)
                bottom.bar((d[:-1] + d[1:]) / 2, np.nan_to_num(step, nan=0.0), width=0.004,
                           color=[COLORS["invalid"] if s > sk_["max_ik_jump_rad"] else COLORS["valid"] for s in step])
        bottom.axhline(sk_["max_ik_jump_rad"], color=COLORS["text"], lw=0.8, ls="--")
        bottom.set_yscale("log")
        for ax in (top, bottom):
            ax.set_xlim(sk_["approach_dist_m"] + 0.002, -0.002)
        top.axhline(math.pi, color=COLORS["muted"], lw=0.6, ls=":")
        top.axhline(-math.pi, color=COLORS["muted"], lw=0.6, ls=":")
        top.set(ylabel="joint angle [rad]")
        bottom.set(xlabel="approach offset behind the grasp pose [m]",
                   ylabel="largest joint step [rad] (dashed: max_ik_jump_rad)")
        top.set_title(f"audit cell {cell}, phi = {math.degrees(phi):.0f} deg, elbow {branch:+d}\n"
                      f"before 0004: {old} (counted as certified)\nnow: {new}", fontsize=9, loc="left")
        top.legend(fontsize=6, ncol=2, loc="lower left")
    fig.tight_layout()
    return fig


def viz_lattice(cfg):
    s = cfg["suite"]
    return Lattice(s["rows"], s["cols"], s["pitch_m"])
