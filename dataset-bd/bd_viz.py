"""B/D dataset visuals: per-root 2D panels (scene, ids, T / goal, intended vs verified blockers, empty-scene T sweep, D edit before /
after, B success / failure strips drawn from stored witness arrays), real MuJoCo renders for 3D (overview + oblique, before / after,
keyframe GIF of a re-executed validated sequence) and a coverage dashboard over ledger + root records.

    python bd_viz.py dashboard <out_dir> <domain>
    python bd_viz.py gallery2d <out_dir> [n]
    python bd_viz.py gallery3d <out_dir> [n]"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle  # noqa: E402

import bd_core as bc  # noqa: E402

OK, BAD, BLUE, GREY = "#0ca30c", "#d03b3b", "#2a78d6", "#7a7975"


def _roots(out):
    return [json.loads(p.read_text()) for p in sorted((Path(out) / "roots").glob("*.json"))]


# ------------------------------------------------------------------ dashboard
def dashboard(out, domain):
    out = Path(out)
    st = bc.Store(out)
    L, R = st.ledger(), _roots(out)
    acc = [e for e in L if e.get("accepted")]
    fig, axs = plt.subplots(2, 3, figsize=(17, 9), constrained_layout=True)
    ks = [0, 1, 2, 3, 4, "NONE"]
    M = np.zeros((5, len(ks)), int)
    for e in acc:
        M[e["spec"]["K"], ks.index(e["observed"] if e["observed"] in ks else "NONE")] += 1
    ax = axs[0, 0]
    ax.imshow(M, cmap="Blues")
    for i in range(5):
        for j in range(len(ks)):
            ax.text(j, i, M[i, j], ha="center", va="center", fontsize=9)
    ax.set(xticks=range(len(ks)), xticklabels=[f"K{k}" if k != "NONE" else "no repair" for k in ks], yticks=range(5),
           yticklabels=[f"req K{k}" for k in range(5)], title=f"requested vs VERIFIED K ({len(acc)} accepted roots)")
    ax = axs[0, 1]
    rej = collections.Counter(e["reason"].split(":")[0] + ":" + e["reason"].split(":")[1][:28] for e in L if not e.get("accepted"))
    ax.barh(range(len(rej)), list(rej.values()), color=GREY)
    ax.set_yticks(range(len(rej)), list(rej.keys()), fontsize=7.5)
    ax.set_title(f"construction rejections / caps ({sum(rej.values())} of {len(L)} attempts)")
    ax = axs[0, 2]
    cnt = collections.Counter()
    for r in R:
        lab = r["labels"]
        cnt[("original", lab["original"]["y"])] += 1
        for d in lab["D"]:
            cnt[("D edit", d["y"])] += 1
        for b in lab["B"]:
            cnt[("B seq", b["y"])] += 1
    cats = ["original", "D edit", "B seq"]
    x = np.arange(3)
    ax.bar(x - 0.18, [cnt[(c, 1)] for c in cats], 0.36, color=OK, label="y = 1 (verified)")
    ax.bar(x + 0.18, [cnt[(c, 0)] for c in cats], 0.36, color=BAD, label="y = 0 (not verified within budget)")
    ax.set_xticks(x, cats)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("binary teacher labels (no third class)")
    ax = axs[1, 0]
    nk = collections.Counter(q["negative_kind"] for r in R for q in [r["labels"]["original"], *r["labels"]["D"], *r["labels"]["B"]] if not q["y"])
    ax.bar(range(len(nk)), list(nk.values()), color=BAD)
    ax.set_xticks(range(len(nk)), list(nk.keys()), fontsize=8)
    ax.set_title("why y = 0 (metadata, not labels)")
    ax = axs[1, 1]
    facs = ["source", "displacement", "mechanism", "distractors", "distractor_role", "parking"]
    yy, labels_ = 0, []
    for f in facs:
        c = collections.Counter(e["spec"][f] for e in acc)
        for lv in bc.FACTORS[f]:
            ax.barh(yy, c.get(lv, 0), color=BLUE)
            labels_.append(f"{f}={lv}")
            yy += 1
        yy += 0.5
    ax.set_title("factor coverage (accepted roots, requested levels)")
    pos = []
    yy = 0
    for f in facs:
        for lv in bc.FACTORS[f]:
            pos.append(yy)
            yy += 1
        yy += 0.5
    ax.set_yticks(pos, labels_, fontsize=7)
    ax.invert_yaxis()
    ax = axs[1, 2]
    if L:
        w = np.cumsum([e["wall_s"] for e in L])
        ax.plot(np.arange(1, len(acc) + 1), [e["wall_s"] for e in acc], ".", color=BLUE, alpha=0.6, label="wall s per accepted root")
        ax.set_xlabel("accepted root #")
        ax.set_ylabel("seconds")
        disk = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
        ax.set_title(f"cost: median {np.median([e['wall_s'] for e in acc]) if acc else 0:.0f} s/root (1 worker), "
                     f"{disk:.1f} MB on disk, {disk / max(1, len(acc)):.2f} MB/root", fontsize=9)
    fig.suptitle(f"B/D dataset coverage — {domain} — {out.name} (counts use VERIFIED K; requested K is construction intent only)", fontsize=12, x=0.01, ha="left")
    p = out / "viz" / "dashboard.png"
    p.parent.mkdir(exist_ok=True)
    fig.savefig(p, dpi=110)
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 2D gallery
def _scene2d(ax, s, goal, robot, roles, fixed_ids, verified, intended, title, q=None, sweep=()):
    import viz
    viz.draw_support(ax, s, cell_ids=True)
    for c in sweep:
        ax.add_patch(Rectangle(((c % 8) * 0.08, (c // 8) * 0.08), 0.08, 0.08, fc="#cfe0f5", ec="none", zorder=1))
    for o in s.objects:
        fc = viz.COLORS["target"] if o.id == "t0" else ("#52514e" if o.id in fixed_ids else viz.COLORS["obstacle"])
        ax.add_patch(Circle(o.xy_m, o.radius_m, fc=fc, ec="k", lw=0.6, zorder=3))
        if o.id != "t0":
            ax.text(*o.xy_m, o.id[1:], fontsize=6, ha="center", va="center", zorder=4, color="white" if o.id in fixed_ids else "k")
        if o.id in intended:
            ax.add_patch(Circle(o.xy_m, o.radius_m + 0.008, fill=False, ec=BAD, lw=1.6, ls="--", zorder=5))
        if o.id in verified:
            ax.add_patch(Circle(o.xy_m, o.radius_m + 0.013, fill=False, ec=OK, lw=1.8, zorder=5))
    ax.add_patch(Circle(s.lattice.center(goal), 0.028, fill=False, ec=viz.COLORS["target_edge"], ls=(0, (3, 2)), lw=1.6, zorder=4))
    if q is not None:
        viz.draw_robot(ax, robot, q, robot.open_gap, alpha=0.45, zorder=6)
    ax.set(xlim=(-0.6, 0.68), ylim=(-0.06, 0.95), aspect="equal", xticks=[], yticks=[])
    ax.set_title(title, fontsize=8.5, loc="left")


def _path(ax, robot, recs, color):
    import geometry as ge
    import motion as mo
    for a in recs:
        if "phase" not in a or len(a["q"]) < 2:
            continue
        pts = np.array([ge.fk(robot, qq)[0][-1] for qq, _, _ in mo.samples(a, 60)])
        ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.2, zorder=7)


def gallery2d(out, n=6):
    import config as cf
    import geometry as ge
    from scene import scene_from_dict
    out = Path(out)
    st = bc.Store(out)
    robot = ge.make_robot(cf.load_config("config.toml"))
    R = _roots(out)
    pick = sorted(R, key=lambda r: (-(r["verified_K"] or 0), r["ids"]["root_id"]))[:n]
    paths = []
    for r in pick:
        s = scene_from_dict(r["scene"])
        L = r["labels"]
        ver = set(L["blocker"]["minimal_sets"][0]) if L["blocker"]["minimal_sets"] else set()
        if not ver and L["B"] and L["B"][0]["role"] == "minimal":
            ver = {o for o, _ in L["B"][0]["sequence"]}
        fig, axs = plt.subplots(1, 4, figsize=(19, 5.2), constrained_layout=True)
        sw = sorted(set().union(*map(set, r["empty_scene_T_swept"].values())))
        _scene2d(axs[0], s, r["goal_cell"], robot, r["roles"], r["fixed_ids"], ver, r["intended_blockers"],
                 f"original  y = {L['original']['y']} ({L['original']['reason']})\nshaded: T sweep in the EMPTY scene; red dashed = intended, "
                 f"green = verified blocker", q=r["q0"], sweep=sw)
        _path(axs[0], robot, [st.load_witness(r["empty_scene_T_witness"])], BLUE)
        d = next((x for x in L["D"] if x["y"] and len(x["edit"]) >= 1), L["D"][0] if L["D"] else None)
        if d:
            cells = {o.id: o.cell for o in s.objects}
            from dataclasses import replace
            s2 = replace(s, objects=tuple(replace(o, cell=d["edit"].get(o.id, o.cell), xy_m=s.lattice.center(d["edit"].get(o.id, o.cell))) for o in s.objects))
            _scene2d(axs[1], s2, r["goal_cell"], robot, r["roles"], r["fixed_ids"], set(), set(),
                     f"D intervention (scene edit, re-query T only)\nedit {d['edit']} -> y = {d['y']} ({d['reason']})")
            for o, c2 in d["edit"].items():
                axs[1].add_patch(FancyArrowPatch(s.lattice.center(cells[o]), s.lattice.center(c2), arrowstyle="-|>", mutation_scale=14, color=BLUE, lw=1.6, zorder=8))
        for ax, b in zip(axs[2:], [next((x for x in L["B"] if x["y"]), None), next((x for x in L["B"] if not x["y"]), None)]):
            if b is None:
                ax.axis("off")
                continue
            cells = {o.id: o.cell for o in s.objects}
            _scene2d(ax, s, r["goal_cell"], robot, r["roles"], r["fixed_ids"], set(), set(),
                     f"B '{b['role']}': {b['sequence'] or 'no relocation'} -> T\n{'SUCCESS (verified + replayed)' if b['y'] else 'FAILURE at step ' + str(b['failed_step']) + ': ' + b['reason']}"
                     f" | {b['n_actions']} actions")
            for j, (o, c2) in enumerate(b["sequence"]):
                ax.add_patch(FancyArrowPatch(s.lattice.center(cells[o]), s.lattice.center(c2), arrowstyle="-|>", mutation_scale=14,
                                             color=OK if b["y"] else BAD, lw=1.6, zorder=8))
                ax.text(*s.lattice.center(c2), str(j + 1), fontsize=8, weight="bold", ha="center", va="center", zorder=9)
            if b.get("witnesses"):
                _path(ax, robot, [st.load_witness(w) for w in b["witnesses"]], OK if b["y"] else BAD)
        fig.suptitle(f"{r['ids']['root_id']} | requested K{r['spec']['K']} {r['spec']['mechanism']} / VERIFIED K = {r['observed']} | "
                     f"start {r['start_stratum']} | mask valid: {L['blocker']['blocker_supervision_valid']}", fontsize=10, x=0.01, ha="left")
        p = out / "viz" / f"{r['ids']['root_id']}.png"
        p.parent.mkdir(exist_ok=True)
        fig.savefig(p, dpi=100)
        plt.close(fig)
        paths.append(p)
    return paths


# ------------------------------------------------------------------ 3D gallery (real MuJoCo renders)
CAMS = {"overview": dict(lookat=(0.45, 0, 0.35), distance=1.5, azimuth=25, elevation=-22),
        "oblique": dict(lookat=(0.75, 0, 0.38), distance=0.7, azimuth=30, elevation=-35)}


def _render(e, path):
    e.render(path=str(path), w=640, h=480, **CAMS[path.stem.split("__")[-1]])


def gallery3d(out, n=3, gif=True):
    import bd_3d
    from PIL import Image
    out = Path(out)
    R = [r for r in _roots(out) if r["labels"]["B"] and r["labels"]["B"][0]["role"] == "minimal" and r["labels"]["B"][0]["y"]]
    R = sorted(R, key=lambda r: (-(r["verified_K"] or 0), r["ids"]["root_id"]))[:n]
    made = []
    for r in R:
        objs, b = r["objects"], r["labels"]["B"][0]
        d = out / "viz" / r["ids"]["root_id"]
        d.mkdir(parents=True, exist_ok=True)
        st0 = {k: {"rc": v["rc"]} for k, v in r["initial_state"].items()}
        fin = {k: {"rc": v["rc"], "off": tuple(v.get("off", (0.0, 0.0)))} for k, v in b["final_state"].items()}
        for tag, state in (("before", st0), ("after", fin)):
            e = bd_3d.build(objs, state, "t0")
            for cam in CAMS:
                _render(e, d / f"{tag}__{cam}.png")
        made.append(str(d))
        if gif and len(made) == 1:
            made.append(str(keyframes(r, d)))
    return made


def keyframes(r, d, every=60):
    """Re-execute the stored minimal sequence (same deterministic planner seeds) and capture frames from the oblique camera."""
    import mujoco
    import bd_3d
    from PIL import Image
    G = bd_3d.env()
    GD = G["GD"]
    objs, b = r["objects"], r["labels"]["B"][0]
    state = {k: {"rc": v["rc"]} for k, v in r["initial_state"].items()}
    frames = []
    for o, dst in [tuple(x) for x in b["sequence"]] + [("t0", r["goal"])]:
        e = bd_3d.build(objs, state, o)
        pl = GD.GridPlanner(e, **{k: v for k, v in bd_3d.PLANNER.items()}, verbose=False)
        pl.place_target(tuple(state[o]["rc"]))
        res = pl.solve(tuple(state[o]["rc"]), tuple(dst))
        if not res["feasible"]:
            break
        rnd = mujoco.Renderer(e.model, 360, 480)
        cam = mujoco.MjvCamera()
        cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = CAMS["oblique"]["lookat"], 0.75, 30, -32
        step0, k = e.step, [0]

        def step(nn=1):
            step0(nn)
            k[0] += 1
            if k[0] % every == 0:
                rnd.update_scene(e.data, cam)
                frames.append(Image.fromarray(rnd.render()))
        e.step = step
        pl.execute(res)
        e.step = step0
        p, rc = e.obj_pos(0), e.cell_of(0)
        cx, cy = GD.cell_xy(rc)
        state = {**state, o: {"rc": [int(rc[0]), int(rc[1])], "off": (float(p[0] - cx), float(p[1] - cy))}}
    path = d / "sequence.gif"
    if frames:
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=60, loop=0, optimize=True)
    return path


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "dashboard":
        print(dashboard(sys.argv[2], sys.argv[3]))
    elif cmd == "gallery2d":
        print(gallery2d(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 6))
    elif cmd == "gallery3d":
        print(gallery3d(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3))
