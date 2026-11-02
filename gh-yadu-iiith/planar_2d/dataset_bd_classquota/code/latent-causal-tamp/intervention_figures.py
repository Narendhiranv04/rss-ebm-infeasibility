"""G2 figures: frozen-action interventions and per-object attribution (Guide v1.6 §3).

Split from figures.py (size): figures.py holds G1 task figures, diagnostics.py the
verdict explanations. Here every robot pose is a configuration on the frozen,
recorded gamma_0 (never replanned); interventions are drawn as labelled snapshot
edits (dashed arrows), never as robot motion.
"""
from __future__ import annotations

import csv
import json
from multiprocessing import Pool
from pathlib import Path
from PIL import Image
import config as cf
import interventions as iv
import scene as sc

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

import figures as fig
import geometry as ge
import motion as mo
import viz
from viz import COLORS, plt

STATUS3 = {mo.VALID: COLORS["valid"], mo.INVALID: COLORS["invalid"], mo.UNRESOLVED: COLORS["unresolved"]}
TAG = {mo.VALID: "VERIFIED_FEASIBLE", mo.INVALID: "CERTIFIED_INFEASIBLE", mo.UNRESOLVED: "UNRESOLVED"}


def _config(frozen, segment: int, s: float):
    a = frozen.arrays
    return a["q"][segment] + s * (a["q"][segment + 1] - a["q"][segment]), \
        a["gap"][segment] + s * (a["gap"][segment + 1] - a["gap"][segment]), str(a["phase"][segment])


def _held(frozen, phase: str):
    return (None, 0.024, tuple(frozen.arrays["held_local"])) if mo.CONTEXT[phase][0] == "held" else None


def snapshot_panel(ax, robot, cfg, frozen, snap, title: str, ids=(), q=None):
    """Scene with IDs, the frozen action's TCP/object traces, the edit arrow, and
    either the robot at home or, if a moved object blocks, the certifier's first
    violating full-body configuration (segment ends ghosted, body outlined)."""
    s, t = snap.scene, snap.scene.get(frozen.target_id)
    viz.scene_panel(ax, s, cfg, title, hide=[o.id for o in s.objects])
    frames = mo.samples(frozen.arrays, 300)
    tcp = np.array([ge.fk(robot, f[0])[0][3] for f in frames])
    obj = np.array([mo.target_xy(robot, frozen.arrays, f[2], f[0], t.xy_m) for f in frames])
    ax.plot(tcp[:, 0], tcp[:, 1], color=COLORS["hypothetical"], lw=0.8, alpha=0.6, zorder=2)
    ax.plot(obj[:, 0], obj[:, 1], color=COLORS["target"], lw=1.2, alpha=0.8, zorder=2)
    viz.draw_objects(ax, s)
    first = next((snap.checks[i].first for i in ids if i in snap.checks and snap.checks[i].first), None)
    if first is not None and first["kind"] in ("ROBOT_BODY", "CARRIED_TARGET"):
        for end in (0.0, 1.0):
            qq, gg, ph = _config(frozen, first["segment"], end)
            viz.draw_robot(ax, robot, qq, gg, held=_held(frozen, ph), alpha=0.25, zorder=6)
        qq, gg, ph = _config(frozen, first["segment"], first["s"])
        viz.draw_robot(ax, robot, qq, gg, held=_held(frozen, ph), zorder=8, highlight=(first["body"],))
    else:
        viz.draw_robot(ax, robot, robot.q_home if q is None else q, robot.open_gap, alpha=0.8, zorder=8)
    e = snap.edit
    if e is not None and e.kind == "RELOCATE":
        a, b = s.lattice.center(e.src_cell), s.lattice.center(e.dst_cell)
        ax.add_patch(Circle(a, 0.024, fc="none", ec=COLORS["muted"], ls="--", lw=1.0, zorder=4))
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, lw=1.4, ls="--",
                                     color=COLORS["hypothetical"], zorder=9))
    for i in ids:
        if i in snap.checks and snap.checks[i].status == mo.INVALID:
            ax.add_patch(Circle(s.get(i).xy_m, 0.032, fc="none", ec=COLORS["invalid"], lw=2, zorder=9))
    lines = [f"F_fixed = {snap.F}"] + [f"c_{i} = {_c(snap.checks[i]) if i in snap.checks else '0 (removed)'}"
                                       for i in ids]
    if first is not None:
        deepest = min(min(snap.checks[i].min_by_phase.values()) for i in ids if i in snap.checks)
        lines.append(f"first certified violation: {first['body']} in {first['phase']}\n"
                     f"  ({first['kind'].lower()}), d = {first['distance'] * 1000:.1f} mm < delta")
        lines.append(f"deepest overlap at stored waypoints: {deepest * 1000:.1f} mm")
    ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=8, family="monospace",
            bbox=dict(fc="white", ec=COLORS["muted"]), zorder=20)
    viz.status_tag(ax, TAG[snap.F], "fixed action (unchanged gamma_0)")


def _c(check) -> str:
    return {mo.INVALID: "1 (blocks)", mo.VALID: "0", mo.UNRESOLVED: "? (unresolved)"}[check.status]


def lineage_figure(robot, cfg, frozen, snaps, variants, ids, labels):
    by = {s.variant: s for s in snaps}
    f, axes = plt.subplots(1, len(variants), figsize=(6.2 * len(variants), 7.0), squeeze=False)
    for ax, v, lab in zip(axes[0], variants, labels):
        e = by[v].edit
        edit = "" if e is None else f"\nedit: {e.obj_id} cell {e.src_cell} -> {e.dst_cell} (diagnostic, not robot-executed)"
        snapshot_panel(ax, robot, cfg, frozen, by[v], f"{lab}{edit}", ids)
    f.tight_layout()
    return f


def attribution_figure(panels):
    """Per obstacle x phase: status colour and min waypoint clearance [mm]."""
    f, axes = plt.subplots(1, len(panels), figsize=(6.5 * len(panels), 5.0), squeeze=False)
    phases = list(mo.PHASES)
    colour = {mo.VALID: 0, mo.UNRESOLVED: 1, mo.INVALID: 2}
    cmap = plt.matplotlib.colors.ListedColormap([COLORS["valid"], COLORS["unresolved"], COLORS["invalid"]])
    for ax, (title, snap) in zip(axes[0], panels):
        ids = sorted(snap.checks)
        grid = np.array([[colour[snap.checks[i].phase_status.get(p, mo.VALID)] for p in phases] for i in ids])
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=2, alpha=0.55, aspect="auto")
        for r, i in enumerate(ids):
            for c, p in enumerate(phases):
                d = snap.checks[i].min_by_phase.get(p)
                ax.text(c, r, "" if d is None else f"{d * 1000:.0f}", ha="center", va="center", fontsize=8)
            first = snap.checks[i].first
            if first:
                ax.text(phases.index(first["phase"]), r + 0.32, first["body"], ha="center", fontsize=7,
                        weight="bold", color=COLORS["text"])
        ax.set_xticks(range(len(phases)), phases, rotation=35, fontsize=8)
        ax.set_yticks(range(len(ids)), ids, fontsize=8)
        ax.set_title(f"{title}\nmin clearance at stored waypoints [mm]; colour = certified phase status\n"
                     f"(green VALID, amber UNRESOLVED, red INVALID; bold = first offending body)",
                     fontsize=9, loc="left")
    f.tight_layout()
    return f


def replay_gif(robot, cfg, panels, path, n: int = 90) -> None:
    """The same recorded gamma_0 replayed on several snapshots: sampled frames whose
    full body comes within delta of an object are outlined (the panel's certified
    F_fixed is printed); nothing is replanned."""
    delta = cfg["suite"]["margin_m"]
    cols = 3
    f, axes = plt.subplots((len(panels) + cols - 1) // cols, cols, figsize=(5.2 * cols, 5.4 * ((len(panels) + 2) // 3)),
                           squeeze=False)
    prepared = []
    for title, frozen, snap in panels:
        w = ge.make_world(snap.scene, exclude=(frozen.target_id,))
        held = (frozen.target_id, 0.024, tuple(frozen.arrays["held_local"]))
        cks = {False: ge.Checker(robot, w, delta, 0.0, self_pairs=False),
               True: ge.Checker(robot, w, delta, 0.0, held, self_pairs=False)}
        prepared.append((title, frozen, snap, cks, mo.samples(frozen.arrays, n)))

    def draw(k):
        for ax, (title, frozen, snap, cks, frames) in zip(axes.flat, prepared):
            ax.clear()
            q, g, ph = frames[k]
            carrying = mo.CONTEXT[ph][0] == "held"
            d = cks[carrying].distances(q, g)
            bad = [cks[carrying].labels[j].split("|") for j in np.flatnonzero(d < delta)]
            t = snap.scene.get(frozen.target_id)
            fig.task_frame(ax, robot, snap.scene, cfg, frozen.arrays, frames, k, title, TAG[snap.F],
                           np.asarray(frozen.arrays["place_xy"]))
            for o in snap.scene.objects:
                if o.id != t.id:
                    ax.text(*o.xy_m, o.id, fontsize=6, ha="center", va="center", zorder=10)
            if bad:
                viz.draw_robot(ax, robot, q, g, held=_held(frozen, ph), zorder=11, highlight=[b[0] for b in bad])
                for _, oid in bad:
                    ax.add_patch(Circle(snap.scene.get(oid).xy_m, 0.032, fc="none", ec=COLORS["invalid"], lw=2, zorder=12))
                ax.text(0.5, 0.14, "this frame: fixed-action COLLISION", transform=ax.transAxes, ha="center",
                        color="white", fontsize=9, weight="bold", bbox=dict(fc=COLORS["invalid"], ec="none"), zorder=20)
        f.tight_layout()
    fig.write_gif(f, draw, n, path, ms=90)


def snapshot_gif(robot, cfg, sequence, path) -> None:
    """Lineage snapshots one after another (held frames): privileged diagnostic
    edits are state changes between snapshots, not robot motion."""
    f, ax = plt.subplots(figsize=(7.2, 7.6))

    def draw(k):
        ax.clear()
        family, frozen, snap, ids = sequence[k // 3]
        e = snap.edit
        how = "reference (frozen gamma_0 planned here)" if e is None else \
            f"PRIVILEGED DIAGNOSTIC EDIT, not robot motion:\n{e.kind} {e.obj_id} cell {e.src_cell} -> {e.dst_cell}"
        snapshot_panel(ax, robot, cfg, frozen, snap, f"{family} / {snap.variant}\n{how}", ids)
        f.tight_layout()
    fig.write_gif(f, draw, 3 * len(sequence), path, ms=600)


def g2_figures(out, robot, cfg, fams: dict) -> None:
    """D1 triptych, D2 lineage, attribution heatmaps, fixed-action replay GIF and the
    intervention-snapshot GIF, from the recorded families."""
    ok = {n: f for n, f in fams.items() if f["status"] == "OK"}
    by = {n: {s.variant: s for s in f["snaps"]} for n, f in ok.items()}
    blk = {n: [s.edit.obj_id for s in f["snaps"] if s.variant.startswith("block")] for n, f in ok.items()}
    plan = {"G2_D1": (["ref", "block_a", "clear_a"], ["reference (gamma_0 planned here)", "one-object intrusion",
                                                     "intrusion corrected"]),
            "G2_D2": (["ref", "block_a", "block_ab", "clear_a", "clear_ab"],
                      ["reference", "blocker A", "blockers A + B", "A corrected, B still blocks", "both corrected"])}
    for n, (variants, labels) in plan.items():
        if n in ok:
            f = lineage_figure(robot, cfg, ok[n]["frozen"], ok[n]["snaps"], variants, blk[n], labels)
            f.savefig(out / f"{n.split('_')[1].lower()}_lineage.png", dpi=80)
            plt.close(f)
    panels = [(f"{n} {v}", by[n][v]) for n, v in (("G2_D1", "block_a"), ("G2_D2", "block_ab"), ("G2_D2", "clear_a"))
              if n in ok]
    f = attribution_figure(panels)
    f.savefig(out / "attribution.png", dpi=90)
    plt.close(f)
    replay = [(f"{n} / {v}", ok[n]["frozen"], by[n][v]) for n, vs in (("G2_D1", ("ref", "block_a", "clear_a")),
                                                                     ("G2_D2", ("block_ab", "clear_a", "clear_ab")))
              if n in ok for v in vs]
    replay_gif(robot, cfg, replay, out / "fixed_action_replay.gif")
    seq = [(n, ok[n]["frozen"], by[n][v], blk[n]) for n, (vs, _) in plan.items() if n in ok for v in vs]
    snapshot_gif(robot, cfg, seq, out / "interventions.gif")


# ------------------------------------------------------------------ G2 command (moved from run.py at G4A)
def run_family(job):
    """Pool worker: plan, freeze and intervene on one G2 fixture."""
    cfg, name, fx = job
    return iv.reference_family(ge.make_robot(cfg), cfg, name, fx)


def cmd_intervene(args) -> int:
    cfg = cf.load_config(args.config)
    fixtures = cf.validate_fixtures(json.loads(Path(args.fixtures).read_text())["fixtures"])
    g2 = {k: fx for k, fx in fixtures.items() if fx["stage"] == "G2"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    robot, chash = ge.make_robot(cfg), cf.config_hash(cfg)
    with Pool(len(g2)) as pool:
        fams = {f["name"]: f for f in pool.imap_unordered(run_family, [(cfg, k, fx) for k, fx in g2.items()])}
    families, gates, tasks, scenes, arrays = {}, {}, {}, {}, {}
    for name in sorted(fams):
        f = fams[name]
        families[name] = iv.family_summary(f, cfg)
        gates[name] = iv.family_gates(families[name], g2[name]["blockers"])
        if f["frozen"] is not None:              # replay-compatible: <name>__q etc. + scenes[name]
            tasks[name] = {"action_hash": f["frozen"].action_hash}
            scenes[name] = sc.scene_to_dict(f["scene"])
            arrays.update({f"{name}__{k}": v for k, v in f["frozen"].arrays.items()})
        for s in f["snaps"]:
            scenes[f"{name}/{s.variant}"] = sc.scene_to_dict(s.scene)
            Image.fromarray(viz.render_observation(s.scene, cfg["render"]["image_px"])[0]).save(
                out / f"obs_{name}_{s.variant}.png")             # clean learner observation, no overlays
    g2_figures(out, robot, cfg, fams)
    ok = all(g and all(g.values()) for g in gates.values())
    cf.write_json(out / "config.json", {"config_hash": chash, "config": cfg})
    cf.write_json(out / "scenes.json", scenes)
    np.savez_compressed(out / "arrays.npz", **arrays)
    cf.write_json(out / "result.json", {
        "stage": "G2", "gate_checks_passed": ok, "gates": gates, "families": families, "tasks": tasks,
        "config_hash": chash, "robot_profile_hash": cf.robot_profile_hash(cfg),
        "verification_scope": "FIXED_ACTION_KINEMATIC_REPLAY (privileged diagnostic edits; no corrective motion)",
        "simulation_replay_status": "NOT_RUN", "physical_execution_status": "NOT_RUN"})
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["family", "variant", "parent", "edit", "F_fixed", "whole_scene_replay", "Delta_I", "c", "oracle_evals"])
        w.writerows([[n, x["variant"], x["parent"], x["edit"] and f"{x['edit']['kind']} {x['edit']['obj_id']}",
                      x["F_fixed"], x["whole_scene_replay"], x.get("Delta_I"),
                      " ".join(f"{i}={c['c']}" for i, c in x["contributions"].items()), x["oracle_evals"]]
                     for n, fam in families.items() for x in fam["nodes"]])
    cf.write_manifest(out, {"planner_base": cfg["planner2d"]["seed"], "per_query": "crc32(task/connector)"}, chash,
                    {"families": {n: f["frozen"] for n, f in families.items()}})
    print(json.dumps({"out": str(out), "gates": gates}))
    return 0 if ok else 1
