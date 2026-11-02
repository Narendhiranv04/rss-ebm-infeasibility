"""Scene figures and the clean learner observation (Guide v1.6 §3).

Raster convention (Methodology §2 asks the renderer to state it): the
observation covers the support rectangle [0, cols*h] x [0, rows*h] seen from
above. Pixel column u increases with world +x. Pixel row v increases DOWNWARD,
i.e. with world -y, so v = 0 is the back edge y = rows*h and lattice row r = 0
lands at the bottom of the image. Integer (u, v) are pixel centres.
"""
from __future__ import annotations

import csv
import json
from itertools import combinations
from pathlib import Path
from PIL import Image
import config as cf
import geometry as ge
import scene as sc

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle  # noqa: E402

from geometry import bodies, make_robot  # noqa: E402
from scene import Scene, candidate_map, occupancy  # noqa: E402

# One small palette for the whole project. valid / invalid / unresolved are the
# status colours (verified or clear / certified collision / unresolved) and are
# always paired with a written label.
COLORS = {
    "support": "#fcfcfb", "cell_edge": "#d6d5d0", "text": "#0b0b0b", "muted": "#7a7975",
    "obstacle": "#a7b4c2", "obstacle_edge": "#3b4a5a",
    "target": "#eb6834", "target_edge": "#9a3a12",
    "link": "#3b4351", "housing": "#22262e", "gripper": "#596273",
    "hypothetical": "#2a78d6", "excluded": "#52514e",
    "valid": "#008300", "invalid": "#e34948", "unresolved": "#eda100",
}
SUPPORT_RGB, OBJECT_RGB, RIM_RGB = (236, 234, 228), (167, 180, 194), (59, 74, 90)


# ------------------------------------------------------------------ raster
def raster_geometry(scene: Scene, image_px: int) -> tuple[int, int, float]:
    """(height_px, width_px, metres per pixel); pixels are square."""
    res = scene.lattice.cols * scene.lattice.pitch_m / image_px
    return round(scene.lattice.rows * scene.lattice.pitch_m / res), image_px, res


def world_to_pixel(x, y, scene: Scene, image_px: int):
    h, _, res = raster_geometry(scene, image_px)
    return np.asarray(x) / res - 0.5, h - np.asarray(y) / res - 0.5


def pixel_to_world(u, v, scene: Scene, image_px: int):
    h, _, res = raster_geometry(scene, image_px)
    return (np.asarray(u) + 0.5) * res, (h - np.asarray(v) - 0.5) * res


def render_observation(scene: Scene, image_px: int):
    """Clean top-down learner RGB plus GT instance masks.

    Returns (rgb uint8 [H,W,3], instance int16 [H,W], ids). instance == k > 0
    marks ids[k-1]; 0 is support. All cylinders are identical, so the target
    looks like every obstacle: the query, not appearance, names the target.
    The robot is not rendered at G0 because its pose is not yet validated.
    """
    h, w, res = raster_geometry(scene, image_px)
    xs, ys = pixel_to_world(*np.meshgrid(np.arange(w), np.arange(h)), scene, image_px)
    rgb = np.empty((h, w, 3), np.uint8)
    rgb[:] = SUPPORT_RGB
    inst = np.zeros((h, w), np.int16)
    ids = [o.id for o in scene.objects]
    for k, o in enumerate(scene.objects, start=1):
        d = np.hypot(xs - o.xy_m[0], ys - o.xy_m[1])
        mask = d <= o.radius_m
        if inst[mask].any():
            raise ValueError(f"{o.id} overlaps another object; not a physical observation")
        inst[mask] = k
        rgb[mask] = OBJECT_RGB
        rgb[mask & (d > o.radius_m - 1.5 * res)] = RIM_RGB
    return rgb, inst, ids


# ------------------------------------------------------------------ robot drawing
ROLE = {"base": "housing", "housing1": "housing", "housing2": "housing", "link0": "link",
        "link1": "link", "link2": "link", "palm": "gripper", "rail": "gripper",
        "finger_l": "gripper", "finger_r": "gripper", "held": "target"}


def _outline(core: np.ndarray, r: float, n: int = 14) -> np.ndarray:
    """Drawing outline of a core dilated by r: polygon, disc or capsule."""
    if r == 0:
        return core
    a, b = core[0], core[-1]
    ang = math.atan2(b[1] - a[1], b[0] - a[0])
    arc = lambda c, t0: [c + r * np.array([math.cos(t), math.sin(t)])  # noqa: E731
                         for t in np.linspace(t0, t0 + math.pi, n)]
    return np.array(arc(b, ang - math.pi / 2) + arc(a, ang + math.pi / 2))


def robot_outlines(robot, q, gap: float, held=None) -> list[tuple[str, np.ndarray]]:
    """(body name, outline vertices) for every collision body, from geometry.bodies:
    the drawing shows exactly the geometry that the certifier checks."""
    return [(n, _outline(core, r)) for n, (core, r) in bodies(robot, q, gap, held).items()]


def robot_extent(robot, q, gap: float, held=None) -> tuple[float, float, float, float]:
    pts = np.vstack([v for _, v in robot_outlines(robot, q, gap, held)])
    return pts[:, 0].min(), pts[:, 0].max(), pts[:, 1].min(), pts[:, 1].max()


# ------------------------------------------------------------------ drawing
def draw_support(ax, scene: Scene, cell_ids: bool = True):
    lat = scene.lattice
    for cell in scene.legal_cells:
        r, c = lat.rc(cell)
        ax.add_patch(Rectangle((c * lat.pitch_m, r * lat.pitch_m), lat.pitch_m, lat.pitch_m,
                               fc=COLORS["support"], ec=COLORS["cell_edge"], lw=0.6, zorder=0))
        if cell_ids:
            ax.text(c * lat.pitch_m + 0.003, (r + 1) * lat.pitch_m - 0.003, str(cell), fontsize=5,
                    color=COLORS["muted"], ha="left", va="top", zorder=1)


def draw_objects(ax, scene: Scene, labels: bool = True, alpha: float = 1.0):
    for o in scene.objects:
        fc, ec = ((COLORS["target"], COLORS["target_edge"]) if o.kind == "target"
                  else (COLORS["obstacle"], COLORS["obstacle_edge"]))
        ax.add_patch(Circle(o.xy_m, o.radius_m, fc=fc, ec=ec, lw=0.9, alpha=alpha, zorder=3))
        if labels:
            ax.text(*o.xy_m, o.id, fontsize=6, ha="center", va="center", color=COLORS["text"], zorder=4)


def draw_robot(ax, robot, q, gap: float, held=None, alpha: float = 0.92, zorder: int = 5,
               highlight=()):
    """Full arm, housings, palm, rail, fingers (and held disc); bodies named in
    `highlight` get a red outline (offending collision pair)."""
    for name, verts in robot_outlines(robot, q, gap, held):
        ax.add_patch(Polygon(verts, closed=True, fc=COLORS[ROLE[name]], ec="none", alpha=alpha,
                             zorder=zorder))
        if name in highlight:
            ax.add_patch(Polygon(verts, closed=True, fc="none", ec=COLORS["invalid"], lw=2.2,
                                 zorder=zorder + 2))


def draw_candidate_domain(ax, scene: Scene, obj_id: str, ghost_k: int):
    """Per-object index k at every cell, target exclusion and one ghost pose."""
    cmap, lat = candidate_map(scene, obj_id), scene.lattice
    for k in range(1, cmap.size):
        ax.text(*lat.center(cmap.cell_of(k)), f"k={k}", fontsize=5.5, ha="center", va="center",
                color=COLORS["hypothetical"], zorder=6)
    o = scene.get(obj_id)
    ax.add_patch(Circle(o.xy_m, o.radius_m, fc="none", ec=COLORS["hypothetical"], lw=2, zorder=6))
    ax.text(o.xy_m[0], o.xy_m[1] - o.radius_m - 0.006, "k=0 stay", fontsize=6, ha="center",
            va="top", color=COLORS["hypothetical"], weight="bold", zorder=7)
    for t in (t for t in scene.objects if t.kind == "target"):
        r, c = lat.rc(t.cell)
        ax.add_patch(Rectangle((c * lat.pitch_m, r * lat.pitch_m), lat.pitch_m, lat.pitch_m, fc="none",
                               ec=COLORS["excluded"], hatch="///", lw=0.8, zorder=2))
        ax.text(t.xy_m[0], t.xy_m[1] - t.radius_m - 0.004, "excluded:\ntarget", fontsize=5,
                ha="center", va="top", color=COLORS["excluded"], zorder=7)
    gxy = lat.center(cmap.cell_of(ghost_k))
    ax.add_patch(Circle(gxy, o.radius_m, fc="none", ec=COLORS["hypothetical"], lw=1.6, ls="--", zorder=6))
    ax.add_patch(FancyArrowPatch(o.xy_m, gxy, arrowstyle="-|>", mutation_scale=10, lw=1.2,
                                 color=COLORS["hypothetical"], shrinkA=10, shrinkB=10, zorder=6))


def legend_handles():
    def dot(fc, ec, label):
        return Line2D([], [], ls="none", marker="o", ms=9, mfc=fc, mec=ec, label=label)
    return [dot(COLORS["obstacle"], COLORS["obstacle_edge"], "movable obstacle"),
            dot(COLORS["target"], COLORS["target_edge"], "active target"),
            Line2D([], [], color=COLORS["link"], lw=5, label="3R links (finite width)"),
            dot(COLORS["housing"], COLORS["housing"], "base / joint housings"),
            Line2D([], [], color=COLORS["gripper"], lw=5, label="palm + rail + 2 fingers")]


def world_axes(ax, scene: Scene, xlim, ylim, title):
    lat = scene.lattice
    ax.set(xlim=xlim, ylim=ylim, aspect="equal", xlabel="world x [m]  (right)",
           ylabel="world y [m]  (backward on support)")
    ax.set_title(title, fontsize=10, loc="left")
    top = ax.secondary_xaxis("top", functions=(lambda x: x / lat.pitch_m - 0.5,
                                               lambda c: (c + 0.5) * lat.pitch_m))
    top.set_xticks(range(lat.cols))
    top.set_xlabel("array column c  (increases with +x)", fontsize=8)
    right = ax.secondary_yaxis("right", functions=(lambda y: y / lat.pitch_m - 0.5,
                                                   lambda r: (r + 0.5) * lat.pitch_m))
    right.set_yticks(range(lat.rows))
    right.set_ylabel("array row r  (increases with +y)", fontsize=8)


def overview_figure(scene: Scene, cfg: dict, obs, notes: list[str]):
    """Four synchronized G0 panels: diagnostic world scene with the full arm,
    per-object candidate domain, clean learner RGB, GT instance masks."""
    rgb, inst, ids = obs
    lat, rcfg = scene.lattice, cfg["render"]
    side = (lat.cols * lat.pitch_m, lat.rows * lat.pitch_m)
    focus = next(o.id for o in scene.objects if o.kind == "obstacle")
    cmap, occ = candidate_map(scene, focus), occupancy(scene)
    free = [k for k in range(1, cmap.size) if cmap.cell_of(k) not in occ]
    ghost_k = min(free, key=lambda k: math.dist(lat.center(cmap.cell_of(k)), cmap.stay_xy_m))
    fig, axes = plt.subplots(2, 2, figsize=(16, 14.2))
    fig.suptitle(f"G0 scene record  |  root {scene.root_id} / variant {scene.variant_id}  |  "
                 "robot drawn at the stored home q (static pose; no motion evidence here)",
                 fontsize=12, x=0.02, ha="left")

    ax = axes[0, 0]
    world_axes(ax, scene, rcfg["overview_xlim_m"], rcfg["overview_ylim_m"],
                "A. Diagnostic world view: 64 cells (physical IDs), objects, full 3R arm + fingers")
    draw_support(ax, scene)
    draw_objects(ax, scene)
    draw_robot(ax, make_robot(cfg), scene.robot.q_rad, scene.robot.gripper_gap_m)
    ax.legend(handles=legend_handles(), loc="lower right", fontsize=7, framealpha=0.95)
    ax.text(rcfg["overview_xlim_m"][0] + 0.01, rcfg["overview_ylim_m"][1] - 0.01,
            f"q = {[round(v, 4) for v in scene.robot.q_rad]} rad (scene record)", fontsize=7,
            va="top", color=COLORS["text"])

    ax = axes[0, 1]
    pad = 0.02
    world_axes(ax, scene, (-pad, side[0] + pad), (-pad, side[1] + pad),
                f"B. Domain of {focus}: x_i in {{0..{cmap.size - 1}}}; k is per-object, not a cell ID\n"
                f"    ghost k={ghost_k} -> cell {cmap.cell_of(ghost_k)} (same ID, no clone); "
                "occupied cells stay candidates")
    draw_support(ax, scene, cell_ids=False)
    draw_objects(ax, scene, labels=False, alpha=0.35)
    draw_candidate_domain(ax, scene, focus, ghost_k)

    ax = axes[1, 0]
    ax.imshow(rgb, interpolation="nearest")
    ax.set_title(f"C. Clean learner RGB ({rgb.shape[1]}x{rgb.shape[0]} px, "
                 f"{raster_geometry(scene, rcfg['image_px'])[2] * 1000:.1f} mm/px): "
                 "pixel row v=0 is world y = back edge", fontsize=10, loc="left")
    ax.set(xlabel="pixel column u (with +x)", ylabel="pixel row v (with -y)")

    ax = axes[1, 1]
    labelled = np.ma.masked_equal(inst, 0)
    target_px = np.isin(inst, [k + 1 for k, i in enumerate(ids) if scene.get(i).kind == "target"])
    ax.imshow(labelled > 0, cmap=matplotlib.colors.ListedColormap([COLORS["obstacle"]]),
              extent=(0, side[0], 0, side[1]), origin="upper", interpolation="nearest")
    ax.imshow(np.ma.masked_where(~target_px, target_px), extent=(0, side[0], 0, side[1]),
              cmap=matplotlib.colors.ListedColormap([COLORS["target"]]), origin="upper",
              interpolation="nearest")
    for k, obj_id in enumerate(ids, start=1):
        v, u = np.nonzero(inst == k)
        cx, cy = pixel_to_world(u.mean(), v.mean(), scene, rcfg["image_px"])
        ax.text(cx, cy + 0.03, obj_id, fontsize=5.5, ha="center", color=COLORS["text"])
    xy = np.array([o.xy_m for o in scene.objects])
    ax.plot(xy[:, 0], xy[:, 1], "+", color=COLORS["text"], ms=6, mew=0.8)
    ax.set(xlim=(0, side[0]), ylim=(0, side[1]), aspect="equal", xlabel="world x [m]",
           ylabel="world y [m]")
    ax.set_title("D. GT instance masks mapped back to world (+ = scene.json centre)",
                 fontsize=10, loc="left")
    fig.text(0.02, 0.008, "\n".join(notes), fontsize=8, family="monospace", va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 0.97))
    return fig


def figure_rgba(fig) -> np.ndarray:
    """Rendered pixels of a figure (for pixel-exact comparisons in tests)."""
    fig.canvas.draw()
    return np.asarray(fig.canvas.buffer_rgba()).copy()


# ------------------------------------------------------------------ stage panels
STATUS = {"VERIFIED_FEASIBLE": ("VERIFIED", COLORS["valid"]),
          "CERTIFIED_INFEASIBLE": ("INFEASIBLE", COLORS["invalid"]),
          "UNRESOLVED": ("UNRESOLVED", COLORS["unresolved"])}
LIMITS = {"grid": ((-0.56, 0.68), (-0.06, 0.80))}  # audit view: support, base and home pose


def scene_panel(ax, scene, cfg, title: str, hide=(), limits="robot"):
    lim = (cfg["render"]["overview_xlim_m"], cfg["render"]["overview_ylim_m"]) if limits == "robot" \
        else LIMITS[limits]
    ax.set(xlim=lim[0], ylim=lim[1], aspect="equal", xlabel="x [m]", ylabel="y [m]")
    ax.set_title(title, fontsize=9, loc="left")
    ax.tick_params(labelsize=7)
    draw_support(ax, scene, cell_ids=False)
    for o in scene.objects:
        if o.id not in hide:
            fc, ec = ((COLORS["target"], COLORS["target_edge"]) if o.kind == "target"
                      else (COLORS["obstacle"], COLORS["obstacle_edge"]))
            ax.add_patch(Circle(o.xy_m, o.radius_m, fc=fc, ec=ec, lw=0.8, zorder=3))


def status_tag(ax, status: str, text: str = "", loc=(0.02, 0.02)):
    tag, color = STATUS[status]
    ax.text(*loc, f"{tag}  {text}".strip(), transform=ax.transAxes, fontsize=8, color="white",
            weight="bold", va="bottom", bbox=dict(fc=color, ec="none", pad=3), zorder=20)


# ------------------------------------------------------------------ G0 command (moved from run.py at G4A)
def load_fixture(path: str, case: str) -> dict:
    fixtures = cf.validate_fixtures(json.loads(Path(path).read_text())["fixtures"])
    if case not in fixtures:
        raise SystemExit(f"unknown case {case!r}; known: {sorted(fixtures)}")
    return fixtures[case]


def scene_checks(cfg: dict, fx: dict, s: sc.Scene, inst: np.ndarray, out: Path) -> dict:
    """Run-time evidence recorded in result.json (pytest holds the full gate)."""
    rebuilt = sc.make_scene(cfg, fx["root_id"], fx["variant_id"], fx["seed"])
    permuted = sc.Scene(**{**s.__dict__, "objects": tuple(reversed(s.objects))})
    gaps = [math.dist(a.xy_m, b.xy_m) - a.radius_m - b.radius_m for a, b in combinations(s.objects, 2)]
    res = raster_geometry(s, cfg["render"]["image_px"])[2]
    areas = np.bincount(inst.ravel(), minlength=len(s.objects) + 1)[1:] * res ** 2
    analytic = math.pi * cfg["suite"]["obstacle_radius_m"] ** 2
    x0, x1, y0, y1 = robot_extent(ge.make_robot(cfg), s.robot.q_rad, s.robot.gripper_gap_m)
    xl, yl = cfg["render"]["overview_xlim_m"], cfg["render"]["overview_ylim_m"]
    return {
        "rebuild_same_hash": sc.scene_hash(rebuilt) == sc.scene_hash(s),
        "json_roundtrip_same_hash": sc.scene_hash(sc.load_scene_json(out / "scene.json"))
        == sc.scene_hash(s),
        "permuted_input_same_hash": sc.scene_hash(permuted) == sc.scene_hash(s),
        "distinct_cells": len(sc.occupancy(s)) == len(s.objects),
        "min_surface_gap_m": float(min(gaps)),
        "min_surface_gap_meets_margin": bool(min(gaps) >= cfg["suite"]["margin_m"]),
        "mask_area_max_rel_error": float(np.max(np.abs(areas - analytic)) / analytic),
        "robot_drawing_within_axes": bool(xl[0] <= x0 and x1 <= xl[1] and yl[0] <= y0 and y1 <= yl[1]),
    }


def cmd_scene(args) -> int:
    cfg = cf.load_config(args.config)
    fx = load_fixture(args.fixtures, args.case)
    s = sc.make_scene(cfg, fx["root_id"], fx["variant_id"], fx["seed"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    chash, shash, rhash = cf.config_hash(cfg), sc.scene_hash(s), s.robot_profile_hash

    cf.write_json(out / "config.json", {"config_hash": chash, "config": cfg})
    sc.save_scene_json(s, out / "scene.json")
    rgb, inst, ids = render_observation(s, cfg["render"]["image_px"])
    Image.fromarray(rgb).save(out / "observation.png")
    np.savez_compressed(out / "arrays.npz", **sc.scene_arrays(s), observation_rgb=rgb,
                        instance_mask=inst, instance_ids=np.array(ids, dtype="U16"))

    checks = scene_checks(cfg, fx, s, inst, out)
    passed = all(v for k, v in checks.items() if isinstance(v, bool))
    notes = [f"scene_hash {shash}", f"config_hash {chash}",
             f"robot_profile_hash {rhash} (PROVISIONAL, untested)",
             f"gamma_0 hash: none (no target action before G1)  |  motion scope: NONE  |  "
             f"gripper: open g={s.robot.gripper_gap_m} m, m={s.robot.attachment}  |  moved history H=[]",
             f"{len(s.objects)} objects ({cfg['suite']['num_obstacles']} obstacles + "
             f"{cfg['suite']['num_targets']} target) on {len(s.legal_cells)} legal cells, "
             f"domain {s.domain_policy}, seed {fx['seed']}"]
    fig = overview_figure(s, cfg, (rgb, inst, ids), notes)
    fig.savefig(out / "overview.png", dpi=120)

    obstacles = [o.id for o in s.objects if o.kind == "obstacle"]
    sizes = [sc.candidate_map(s, i).size - 1 for i in obstacles]
    with open(out / "metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value", "unit"])
        w.writerows([["num_objects", len(s.objects), "count"], ["num_obstacles", len(obstacles), "count"],
                     ["occupied_cells", len(sc.occupancy(s)), "count"],
                     ["free_cells", len(s.legal_cells) - len(sc.occupancy(s)), "count"],
                     ["candidates_per_obstacle_min", min(sizes), "count"],
                     ["candidates_per_obstacle_max", max(sizes), "count"],
                     ["min_surface_gap", checks["min_surface_gap_m"], "m"],
                     ["required_pair_clearance_margin", cfg["suite"]["margin_m"], "m"],
                     ["mask_area_max_rel_error", checks["mask_area_max_rel_error"], "ratio"]])

    cf.write_json(out / "result.json", {
        "stage": "G0", "fixture": args.case, "gate_checks_passed": passed, "checks": checks,
        "root_family_id": s.root_id, "variant_id": s.variant_id, "scene_hash": shash,
        "config_hash": chash, "robot_profile_hash": rhash, "robot_profile_status": "PROVISIONAL",
        "domain_policy": s.domain_policy, "margin_policy": cfg["suite"]["margin_policy"],
        "query_status": None, "verification_scope": "NONE", "target_hash": None,
        "robot_pose_status": "SCHEMATIC_NOT_RUN", "moved_history": [],
        "target_replay_status": "NOT_ATTEMPTED", "simulation_replay_status": "NOT_RUN",
        "physical_execution_status": "NOT_RUN",
        "note": "No action query exists at G0; three-valued query status begins at G2."})

    cf.write_manifest(out, {"root": fx["seed"]}, chash, {"scene_hash": shash})
    print(json.dumps({"out": str(out), "gate_checks_passed": passed, "scene_hash": shash}))
    return 0 if passed else 1
