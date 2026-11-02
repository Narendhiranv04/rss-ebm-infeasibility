"""G1 stage figures and animations, built only from recorded results (Guide v1.6 §3).

Split from viz.py (size): viz.py holds drawing primitives and the G0 overview; this
module composes task, audit and corner figures and GIFs; diagnostics.py explains
verification verdicts. Every robot pose drawn is either a screened
candidate pose or a configuration on a stored joint-linear segment (the executed
path itself); object poses follow from the phase context and the attachment.
"""
from __future__ import annotations

import csv
import json
from multiprocessing import Pool
from pathlib import Path
import config as cf
import scene as sc

import math

import numpy as np
from matplotlib.patches import Circle, Rectangle
from PIL import Image

import geometry as ge
import motion as mo
import skill as sk
import diagnostics as dg
import viz
from viz import COLORS, STATUS, plt


# ------------------------------------------------------------------ panels
def task_frame(ax, robot, scene, cfg, arrays, frames, k: int, title: str, status: str, dest_xy):
    """Robot, target and traces at frame k of a recorded task trajectory."""
    t = next(o for o in scene.objects if o.kind == "target")
    viz.scene_panel(ax, scene, cfg, title, hide=(t.id,))
    ax.add_patch(Circle(dest_xy, t.radius_m, fc="none", ec=COLORS["target_edge"], ls="--", lw=1.0))
    tcp = np.array([ge.fk(robot, f[0])[0][3] for f in frames[:k + 1]])
    obj = np.array([mo.target_xy(robot, arrays, f[2], f[0], t.xy_m) for f in frames[:k + 1]])
    ax.plot(tcp[:, 0], tcp[:, 1], color=COLORS["hypothetical"], lw=1.0, zorder=6)
    ax.plot(obj[:, 0], obj[:, 1], color=COLORS["target"], lw=1.4, zorder=6)
    q, gap, phase = frames[k]
    viz.draw_robot(ax, robot, q, gap, zorder=8)
    ax.add_patch(Circle(obj[-1], t.radius_m, fc=COLORS["target"], ec=COLORS["target_edge"], zorder=9))
    ax.text(0.98, 0.98, phase.upper(), transform=ax.transAxes, ha="right", va="top", fontsize=9,
            weight="bold", bbox=dict(fc="white", ec=COLORS["muted"], pad=2), zorder=20)
    viz.status_tag(ax, status)


def keyframe_panel(ax, robot, scene, cfg, arrays, title: str, status: str, dest_xy):
    """Whole task in one panel: TCP/object traces plus ghosts at phase ends."""
    frames = mo.samples(arrays, 240)
    task_frame(ax, robot, scene, cfg, arrays, frames, len(frames) - 1, title, status, dest_xy)
    phase = arrays["phase"]
    for name, alpha in (("approach", 0.35), ("carry", 0.55)):
        last = int(np.flatnonzero(phase == name)[-1]) + 1
        held = (None, 0.024, tuple(arrays["held_local"])) if name == "carry" else None
        viz.draw_robot(ax, robot, arrays["q"][last], arrays["gap"][last], held=held, alpha=alpha, zorder=7)


# ------------------------------------------------------------------ figures
def audit_figure(rows: list[dict], scene, robot, cfg, n_dirs: int):
    """64-cell reachability: complete-skill status and the candidate breakdown."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12.5))
    counts = {s: sum(r["status"] == s for r in rows) for s in STATUS}
    short = {"VERIFIED_FEASIBLE": "OK", "CERTIFIED_INFEASIBLE": "INF", "UNRESOLVED": "UNR"}
    panels = [("status", "A. Complete six-phase skill, pick c -> place 63-c (obstacle-free)\n   "
               + ", ".join(f"{STATUS[s][0]} {n}" for s, n in counts.items())),
              ("n_ik", f"B. Grasp candidates with exact IK inside joint limits (of {n_dirs})"),
              ("n_pose", f"C. ... whose open-finger grasp pose is collision-free (of {n_dirs})"),
              ("n_grasp", f"D. ... with certified approach and contact closure (of {n_dirs});\n"
                          "   ticks = certified tool-axis directions")]
    for ax, (key, title) in zip(axes.flat, panels):
        viz.scene_panel(ax, scene, cfg, title, hide=[o.id for o in scene.objects], limits="grid")
        for r in rows:
            x, y = scene.lattice.center(r["cell"])
            if key == "status":
                ax.add_patch(Rectangle((x - 0.04, y - 0.04), 0.08, 0.08, fc=STATUS[r["status"]][1], alpha=0.5, zorder=2))
                ax.text(x, y, short[r["status"]], ha="center", va="center", fontsize=7, weight="bold", zorder=4)
                continue
            ax.add_patch(Rectangle((x - 0.04, y - 0.04), 0.08, 0.08, fc=COLORS["hypothetical"],
                                   alpha=0.08 + 0.55 * r[key] / n_dirs, zorder=2))
            if key != "n_grasp":
                ax.text(x, y, str(r[key]), ha="center", va="center", fontsize=8, zorder=4)
                continue
            ax.text(x - 0.036, y + 0.036, str(r[key]), ha="left", va="top", fontsize=6, zorder=5)
            for phi in r["ok_phis"]:
                u = np.array([math.cos(phi), math.sin(phi)])
                ax.plot(*np.array([(x, y) - 0.03 * u, (x, y) - 0.016 * u]).T, color=COLORS["text"], lw=0.6, zorder=4)
        viz.draw_robot(ax, robot, robot.q_home, robot.open_gap, alpha=0.35, zorder=5)
        ax.plot(*robot.base, "k+", ms=10, zorder=6)
        ax.text(robot.base[0], robot.base[1] - 0.03, "base", ha="center", va="top", fontsize=7)
    fig.tight_layout()
    return fig


def corners_figure(tasks: list[dict], robot, cfg):
    """One keyframe panel per recorded corner task (rows: scene families)."""
    cols = 4
    fig, axes = plt.subplots(math.ceil(len(tasks) / cols), cols, figsize=(5.2 * cols, 5.4 * math.ceil(len(tasks) / cols)),
                             squeeze=False)
    for ax, t in zip(axes.flat, tasks):
        title = f"{t['label']}: cell {t['src']} -> {t['dst']}"
        if t["arrays"] is None:
            viz.scene_panel(ax, t["scene"], cfg, title)
            viz.status_tag(ax, t["status"], t["reason"])
        else:
            keyframe_panel(ax, robot, t["scene"], cfg, t["arrays"], title, t["status"],
                           t["scene"].lattice.center(t["dst"]))
    return fig


def write_gif(fig, draw, n: int, path, ms: int = 70) -> None:
    """Render draw(k) for k < n into a looping GIF with one shared palette."""
    imgs = []
    for k in range(n):
        draw(k)
        imgs.append(Image.fromarray(viz.figure_rgba(fig)[..., :3]))
    picks = [imgs[k] for k in sorted({0, n // 4, n // 2, 3 * n // 4, n - 1})]   # palette sees colours of every phase
    sheet = Image.new("RGB", (picks[0].width, picks[0].height * len(picks)))
    for k, im in enumerate(picks):
        sheet.paste(im, (0, k * im.height))
    pal = sheet.quantize(colors=128)
    frames = [im.quantize(palette=pal) for im in imgs]
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=ms, loop=0, optimize=True)
    plt.close(fig)


def animate_tasks(tasks: list[dict], robot, cfg, path, n: int = 90) -> None:
    """2x2 (or 1xN) synchronised animation of recorded tasks."""
    cols = 2 if len(tasks) == 4 else len(tasks)
    rows = math.ceil(len(tasks) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 5.0 * rows), squeeze=False)
    sampled = [mo.samples(t["arrays"], n) for t in tasks]

    def draw(k):
        for ax, t, fr in zip(axes.flat, tasks, sampled):
            ax.clear()
            task_frame(ax, robot, t["scene"], cfg, t["arrays"], fr, k,
                       f"{t['label']}: {t['src']} -> {t['dst']}", t["status"],
                       t["scene"].lattice.center(t["dst"]))
        fig.tight_layout()
    write_gif(fig, draw, n, path)


def skill_filmstrip(robot, scene, cfg, arrays, status: str, dest_xy):
    """One panel per phase, robot at the end of that phase, traces so far."""
    frames, phases = mo.samples(arrays, 400), list(mo.PHASES)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10.5))
    axes.flat[-1].axis("off")
    for ax, ph in zip(axes.flat, phases):
        k = max(i for i, f in enumerate(frames) if f[2] == ph)
        task_frame(ax, robot, scene, cfg, arrays, frames, k, ph, status, dest_xy)
        ax.set(xlabel="", ylabel="")
    return fig


# ------------------------------------------------------------------ G1 orchestration
def g1_figures(out, robot, cfg, fixtures: dict, done: dict) -> None:
    """Write every G1 figure and animation into the flat run directory."""
    d0 = done["audit_c00"][0]
    f = audit_figure(sk.audit_rows(done), d0, robot, cfg, 2 * cfg["skill2d"]["approach_dirs"])
    f.savefig(out / "overview.png", dpi=100)
    plt.close(f)
    task = lambda tid: {"label": tid.replace("G1_", ""), "scene": done[tid][0],  # noqa: E731
                        "status": done[tid][1].status, "reason": done[tid][1].reason,
                        "arrays": None if done[tid][1].traj is None else done[tid][1].traj.arrays(),
                        "src": fixtures[tid.rsplit("_t", 1)[0]]["tasks"][int(tid.rsplit("_t", 1)[1])][0],
                        "dst": fixtures[tid.rsplit("_t", 1)[0]]["tasks"][int(tid.rsplit("_t", 1)[1])][1]}
    rows = {case: [task(f"{case}_t{i}") for i in range(4)] for case in ("G1_D0", "G1_S1", "G1_S2")}
    f = corners_figure(rows["G1_D0"] + rows["G1_S1"] + rows["G1_S2"], robot, cfg)
    f.savefig(out / "corners.png", dpi=80)
    plt.close(f)
    for case, name in (("G1_D0", "corners_d0.gif"), ("G1_S1", "corners_s1.gif")):
        ok = [t for t in rows[case] if t["arrays"] is not None]
        if ok:
            animate_tasks(ok, robot, cfg, out / name)
    s1 = fixtures["G1_S1"]
    nominal = task("G1_S1_t%d" % s1["tasks"].index(s1["nominal_task"]))
    if nominal["arrays"] is not None:
        f = skill_filmstrip(robot, nominal["scene"], cfg, nominal["arrays"], nominal["status"],
                            nominal["scene"].lattice.center(nominal["dst"]))
        f.savefig(out / "skill.png", dpi=80)
        plt.close(f)
        animate_tasks([nominal], robot, cfg, out / "trace.gif", n=140)
    f = dg.diagnostics_figure(robot, cfg, dg.diagnostic_examples(robot, cfg, done))
    f.savefig(out / "diagnostics.png", dpi=90)
    plt.close(f)
    clutter = task("G1_CLUTTER_t0")
    f, axes = plt.subplots(1, 2, figsize=(16, 8))
    res = done["G1_CLUTTER_t0"][1]
    dg.candidates_panel(axes[0], robot, clutter["scene"], cfg, res,
                     f"A. 48 declared pick candidates for t0 (green OK, red certified, amber unresolved)")
    if clutter["arrays"] is not None:
        keyframe_panel(axes[1], robot, clutter["scene"], cfg, clutter["arrays"], "B. Verified skill",
                       clutter["status"], clutter["scene"].lattice.center(clutter["dst"]))
    else:
        dg.rejection_panel(axes[1], robot, clutter["scene"], cfg, res)
    f.savefig(out / "clutter.png", dpi=90)
    plt.close(f)


# ------------------------------------------------------------------ G1 command (moved from run.py at G4A)
def cmd_check(args) -> int:
    cfg = cf.load_config(args.config)
    fixtures = cf.validate_fixtures(json.loads(Path(args.fixtures).read_text())["fixtures"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    robot, chash, rhash = ge.make_robot(cfg), cf.config_hash(cfg), cf.robot_profile_hash(cfg)
    with Pool(args.workers) as pool:
        done = {tid: (s, res, secs) for tid, s, res, secs in pool.imap_unordered(sk.run_task, sk.g1_jobs(cfg, fixtures))}
    tasks, arrays, scenes = {}, {}, {}
    for tid in sorted(done):
        s, res, secs = done[tid]
        tasks[tid] = {**sk.summary(res, sc.scene_hash(s), rhash), "seconds": round(secs, 1),
                      "scene_hash": sc.scene_hash(s), "root_id": s.root_id}
        scenes[tid] = sc.scene_to_dict(s)
        if res.traj is not None:
            arrays.update({f"{tid}__{k}": v for k, v in res.traj.arrays().items()})
    if not args.no_figures:
        g1_figures(out, robot, cfg, fixtures, done)
    f = dg.ik_tracking_figure(robot, cfg, dg.ik_examples(robot, cfg, done))
    f.savefig(out / "ik_tracking.png", dpi=90)
    extent = traj_extent(robot, done)
    xl, yl = cfg["render"]["overview_xlim_m"], cfg["render"]["overview_ylim_m"]
    nominal = "G1_S1_t%d" % fixtures["G1_S1"]["tasks"].index(fixtures["G1_S1"]["nominal_task"])
    audit = [t for k, t in tasks.items() if k.startswith("audit_")]
    corners = {k: t["status"] for k, t in tasks.items() if k.startswith(("G1_D0", "G1_S1")) and k != nominal}
    gate = {"home_clear_of_all_64_cells": sk.home_check(cfg)["home"]["status"] == ge.CLEAR,
            "all_64_cells_verified_obstacle_free": all(t["status"] == sk.VERIFIED for t in audit),
            "corner_tasks_verified_D0_and_S1": all(v == sk.VERIFIED for v in corners.values()),
            "nominal_skill_verified": tasks[nominal]["status"] == sk.VERIFIED,
            "robot_frames_within_fixed_axes": bool(xl[0] <= extent[0] and extent[1] <= xl[1]
                                                   and yl[0] <= extent[2] and extent[3] <= yl[1])}
    sc.save_scene_json(sc.scene_from_dict(scenes[nominal]), out / "scene.json")
    cf.write_json(out / "scenes.json", scenes)
    np.savez_compressed(out / "arrays.npz", **arrays)
    cf.write_json(out / "config.json", {"config_hash": chash, "config": cfg})
    counts = {st: sum(t["status"] == st for t in audit) for st in (sk.VERIFIED, sk.INFEASIBLE, sk.UNRESOLVED)}
    cf.write_json(out / "result.json", {
        "stage": "G1", "gate_checks_passed": all(gate.values()), "gate": gate, "home_check": sk.home_check(cfg),
        "config_hash": chash, "robot_profile_hash": rhash, "robot_profile_status": "TESTED_G1",
        "nominal_task": nominal, "nominal_action_hash": tasks[nominal]["action_hash"],
        "audit_counts": counts, "robot_extent_m": extent, "tasks": tasks,
        "wall_clock_dependent_tasks": sorted(k for k, t in tasks.items() if t["wall_clock_dependent"]),
        "target_replay_status": "KINEMATIC_REPLAY_CERTIFIED (see replay)", "simulation_replay_status": "NOT_RUN",
        "physical_execution_status": "NOT_RUN"})
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["task", "status", "reason", "waypoints", "seconds"])
        w.writerows([[k, t["status"], t["reason"], t["waypoints"], t["seconds"]] for k, t in tasks.items()])
    cf.write_manifest(out, {"planner_base": cfg["planner2d"]["seed"], "per_query": "crc32(task/connector)"},
                    chash, {"nominal_action_hash": tasks[nominal]["action_hash"]})
    print(json.dumps({"out": str(out), "gate": gate, "audit_counts": counts}))
    return 0 if all(gate.values()) else 1


def traj_extent(robot, done) -> list[float]:
    """Bounding box of every body over every recorded waypoint and figure sample."""
    boxes = [viz.robot_extent(robot, q, g) for _, res, _ in done.values() if res.traj is not None
             for q, g, _ in mo.samples(res.traj.arrays(), 120)]
    boxes.append(viz.robot_extent(robot, robot.q_home, robot.open_gap))
    b = np.array(boxes)
    return [float(b[:, 0].min()), float(b[:, 1].max()), float(b[:, 2].min()), float(b[:, 3].max())]
