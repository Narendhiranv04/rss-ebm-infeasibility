"""G3 figures: analytic assignment boards, executed move-once repairs, rejected
orders, restricted-domain control and the verification summary (Guide v1.6 G3).

Every robot pose drawn here is a configuration on a recorded, certified trajectory:
relocation skills planned from the achieved state, then the unchanged frozen
gamma_0. Rejected relocations have no trajectory and are drawn as the intended
move (red dashed) with the first certified offence written out.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

import figures as fig
import geometry as ge
import intervention_figures as ifig
import interventions as iv
import motion as mo
import verify as vf
from viz import COLORS, plt
import viz

TINT = {mo.VALID: "#d9efd9", mo.INVALID: "#f8d3d2", mo.UNRESOLVED: "#fbe7b5"}
TAG = {vf.EXECUTABLE: "VERIFIED_FEASIBLE", vf.UNRESOLVED: "UNRESOLVED", vf.TERMINAL_INFEASIBLE: "CERTIFIED_INFEASIBLE"}


def _ids(ax, scene, skip=()):
    for o in scene.objects:
        if o.id not in skip:
            ax.text(*o.xy_m, o.id, fontsize=6, ha="center", va="center", zorder=10)


def _view(ax):
    """Robot-view panels: base, full reach of every recorded connector and the support."""
    ax.set(xlim=(-0.72, 0.72), ylim=(-0.06, 1.34))


def _arrow(ax, a, b, color, ls="-", lw=1.6, alpha=1.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=11, lw=lw, ls=ls, color=color, alpha=alpha,
                                 shrinkA=4, shrinkB=4, zorder=9))


def _gamma0_trace(ax, robot, frozen, scene, alpha=0.7):
    frames = mo.samples(frozen.arrays, 200)
    t = scene.get(frozen.target_id)
    obj = np.array([mo.target_xy(robot, frozen.arrays, f[2], f[0], t.xy_m) for f in frames])
    ax.plot(obj[:, 0], obj[:, 1], color=COLORS["target"], lw=1.3, alpha=alpha, zorder=2)


def _box(ax, lines, loc=(0.02, 0.98)):
    """Text box: top-left inside a robot-view panel, or below a zoomed board (loc y < 0)."""
    ax.text(*loc, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7.5, family="monospace",
            bbox=dict(fc="white", ec=COLORS["muted"]), zorder=20)


def offence(res, scene, oid) -> str:
    """The first certified offence of a rejected relocation, with the blocking IDs."""
    if res.reason == "MOVE_ONCE_VIOLATION":
        return "object already moved: move-once rule"
    cands = res.places if res.reason == "NO_PLACE_CANDIDATE" else res.picks if res.reason == "NO_PICK_CANDIDATE" else []
    ids = {o.id for o in scene.objects}
    stages = Counter(c.stage for c in cands)
    who = Counter(p + " (itself)" * (p == oid) for c in cands for p in c.reason.split("|") if p in ids)
    text = f"{res.status}\n{res.reason}"
    if cands:
        text += f"\n{len(cands)} declared candidates: " + ", ".join(f"{k} x{n}" for k, n in stages.items())
        text += "\nfirst offending pair with: " + ", ".join(f"{k} x{n}" for k, n in sorted(who.items()))
    return text


# ------------------------------------------------------------------ episode panels
def motion_frame(ax, robot, cfg, scene, oid, cell, arr, frames, i, title, moved, last=False):
    """Frame i of one recorded relocation skill in its own pre-move scene."""
    o = scene.get(oid)
    viz.scene_panel(ax, scene, cfg, title, hide=[oid])
    _ids(ax, scene, skip=(oid,))
    ax.add_patch(Circle(scene.lattice.center(cell), o.radius_m, fc="none", ec=COLORS["hypothetical"], ls="--", lw=1.2))
    ax.add_patch(Circle(o.xy_m, o.radius_m, fc="none", ec=COLORS["muted"], ls=":", lw=1.0))
    _arrow(ax, o.xy_m, scene.lattice.center(cell), COLORS["hypothetical"], ls=":", lw=1.0, alpha=0.7)
    tcp = np.array([ge.fk(robot, f[0])[0][3] for f in frames[:i + 1]])
    path = np.array([mo.target_xy(robot, arr, f[2], f[0], o.xy_m) for f in frames[:i + 1]])
    ax.plot(tcp[:, 0], tcp[:, 1], color=COLORS["hypothetical"], lw=1.0, zorder=6)
    ax.plot(path[:, 0], path[:, 1], color=COLORS["obstacle_edge"], lw=1.3, zorder=6)
    q, g, ph = frames[i]
    viz.draw_robot(ax, robot, q, g, zorder=8)
    ax.add_patch(Circle(path[-1], o.radius_m, fc=COLORS["obstacle"], ec=COLORS["obstacle_edge"], lw=1.4, zorder=9))
    ax.text(*path[-1], oid, fontsize=6, ha="center", va="center", zorder=10)
    phase = "CONNECTOR -> gamma_0 start" if last and ph == "return_home" else ph.upper()
    ax.text(0.98, 0.98, phase, transform=ax.transAxes, ha="right", va="top", fontsize=9, weight="bold",
            bbox=dict(fc="white", ec=COLORS["muted"], pad=2), zorder=20)
    ax.text(0.98, 0.90, f"moved history H = [{', '.join(moved)}]", transform=ax.transAxes, ha="right", va="top",
            fontsize=8, zorder=20)
    _view(ax)


def relocation_panel(ax, robot, cfg, scene, oid, cell, res, title, moved, last=False):
    arr = res.traj.arrays()
    frames = mo.samples(arr, 200)
    motion_frame(ax, robot, cfg, scene, oid, cell, arr, frames, len(frames) - 1, title, moved, last)
    for name in ("close_attach", "release"):
        k = int(np.flatnonzero(arr["phase"] == name)[-1]) + 1
        viz.draw_robot(ax, robot, arr["q"][k], arr["gap"][k], alpha=0.3, zorder=7)
    viz.status_tag(ax, res.status, f"relocation {oid}: cell {scene.get(oid).cell} -> {cell}")


def failure_panel(ax, robot, cfg, state, oid, cell, res, title):
    scene = state.scene
    viz.scene_panel(ax, scene, cfg, title)
    _ids(ax, scene)
    viz.draw_robot(ax, robot, state.q, state.gap, alpha=0.8, zorder=8)
    o = scene.get(oid)
    ax.add_patch(Circle(o.xy_m, 0.032, fc="none", ec=COLORS["invalid"], lw=2, zorder=9))
    _arrow(ax, o.xy_m, scene.lattice.center(cell), COLORS["invalid"], ls="--")
    ax.plot(*scene.lattice.center(cell), marker="x", ms=10, mew=2.5, color=COLORS["invalid"], zorder=9)
    _box(ax, [f"REJECTED {oid} -> cell {cell}", offence(res, scene, oid)])
    viz.status_tag(ax, res.status, "relocation not executed")
    _view(ax)


def final_panel(ax, robot, cfg, frozen, att, title):
    moved = list(att.states[-1].moved)
    ifig.snapshot_panel(ax, robot, cfg, frozen, att.final, title, moved, q=att.states[-1].q)
    _view(ax)


def stepwise_figure(robot, cfg, frozen, s0, rows, title):
    """One row per attempted order: each relocation (executed or the rejected one),
    then the final achieved scene with the unchanged gamma_0 if every move verified."""
    cols = max(len(a.order) for _, a in rows) + 1
    f, axes = plt.subplots(len(rows), cols, figsize=(5.0 * cols, 5.4 * len(rows)), squeeze=False)
    for r, (label, att) in enumerate(rows):
        state = s0
        for k, ((oid, cell), res) in enumerate(zip(att.order, att.skills)):
            head = f"{label}\nstep {k + 1}/{len(att.order)}" if k == 0 else f"step {k + 1}/{len(att.order)}"
            if k < len(att.states):
                relocation_panel(axes[r, k], robot, cfg, state.scene, oid, cell, res, head, state.moved + (oid,),
                                 last=k == len(att.order) - 1)
                state = att.states[k]
            else:
                failure_panel(axes[r, k], robot, cfg, state, oid, cell, res, head)
        if att.final is not None:
            final_panel(axes[r, len(att.order)], robot, cfg, frozen, att,
                        f"final achieved scene: unchanged gamma_0\nattempt status {att.status}")
        for ax in axes[r]:
            if not ax.has_data() and not ax.patches:
                ax.axis("off")
    f.suptitle(title, fontsize=12, x=0.01, ha="left")
    f.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (5.3 * len(rows))))
    return f


def repair_gif(robot, cfg, frozen, s0, attempts, path, title, n=45):
    """Blocked state -> each relocation skill (from the achieved state) -> connector
    -> the unchanged gamma_0 replay -> final state; rejected orders hold on the offence."""
    blocked = iv.evaluate(robot, cfg, frozen, iv.Snapshot("blocked", None, None, s0.scene))
    blockers = [i for i, c in blocked.checks.items() if c.status == mo.INVALID]
    seq = [lambda ax: (ifig.snapshot_panel(ax, robot, cfg, frozen, blocked,
                                           f"{title}\nblocked state: frozen gamma_0 collides", blockers), _view(ax))] * 8
    for label, att in attempts:
        state = s0
        for k, ((oid, cell), res) in enumerate(zip(att.order, att.skills)):
            head = f"{label}\nrelocation {k + 1}/{len(att.order)}: {oid} -> cell {cell}"
            if k >= len(att.states):
                seq += [lambda ax, st=state, o=oid, c=cell, r=res, h=head:
                        failure_panel(ax, robot, cfg, st, o, c, r, h)] * 10
                break
            arr = res.traj.arrays()
            fr = mo.samples(arr, n)
            seq += [lambda ax, i=i, sc=state.scene, o=oid, c=cell, a=arr, fr=fr, h=head, m=state.moved + (oid,),
                    last=k == len(att.order) - 1: (motion_frame(ax, robot, cfg, sc, o, c, a, fr, i, h, m, last),
                                                   viz.status_tag(ax, "VERIFIED_FEASIBLE", "certified relocation"))
                    for i in range(n)]
            state = att.states[k]
        if att.final is not None:
            fr = mo.samples(frozen.arrays, n)
            seq += [lambda ax, i=i, sc=state.scene, fr=fr, lab=label: (fig.task_frame(ax, robot, sc, cfg, frozen.arrays, fr, i,
                    f"{lab}\nunchanged frozen gamma_0 replayed in the achieved scene\n{frozen.action_hash[:23]}",
                    "VERIFIED_FEASIBLE", np.asarray(frozen.arrays["place_xy"])), _ids(ax, sc, skip=("t0",)), _view(ax))
                    for i in range(n)]
            seq += [lambda ax, a=att, lab=label: final_panel(ax, robot, cfg, frozen, a, f"{lab}\nfinal: {a.status}")] * 10
    f, ax = plt.subplots(figsize=(7.0, 7.4))

    def draw(k):
        ax.clear()
        seq[k](ax)
        f.tight_layout()
    fig.write_gif(f, draw, len(seq), path, ms=80)
