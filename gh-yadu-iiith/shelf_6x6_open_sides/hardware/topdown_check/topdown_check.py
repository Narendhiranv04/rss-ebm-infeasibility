"""Top-down grasp + horizontal slide check at TABLE LEVEL (xArm7 + G2), EXPLORATORY, kinematics only (no physics run).

    MUJOCO_GL=disable <env3d python> topdown_check.py cells [--part i/n]    -> results.jsonl  (one line per cell x neighbour kind)
    <env3d python> topdown_check.py val                                      -> val.json       (VAL front-column rule)
    <env3d python> topdown_check.py report                                   -> summary.md

Scene = the dataset's SHELF66 V3 module (bd_shelf.env()["TS"], as approach_check.py), patched in-process only:
  * board top zb = 0.12 = BASE_Z: the board is level with the robot's base plate (real robot bolted to the table, no pedestal);
    a solid table slab z 0..zb from x = -0.40 to the board's front edge (= hw_patch.py HW_TABLE=1 with HW_ZB=0.12);
  * shelf ceiling, back wall and the 8 cm front step removed (a flat table has none); ARM_CEILING off (a top-down hand is
    always above the objects); no-overpass columns off: only REAL neighbour geometry counts;
  * grid placement unchanged: x0 = 0.42, pitch 7.5 cm, centred on y = 0 -> board front edge x = 0.155 (grid front edge 0.195).
Grasp = tilt 0 (hand straight down), azimuth yaw or yaw+180, pad centre at mid-height, else near the top (shelf_scene GRASP_AT).
yaw 0: fingers close along y (toward the left/right neighbours); yaw 90: along x (toward front/back neighbours).
reach  = an IK branch at the grasp pose AND a straight 13 cm vertical approach (chain), clear of board / table / itself
         (the scene's own static checker, 3 mm).
clear  = along that approach the robot (all geoms, visual finger meshes included, fingers pre-shaped to width + 6 mm/side)
         stays >= 5 mm (OBJECT_MARGIN) from all 4 neighbours (mj_geomDistance; meshes as convex hulls).
slide  = from that grasp configuration, straight line one pitch (7.5 cm) in +x/-x/+y/-y at grasp height, same orientation
         (chain: IK each 5 mm, no branch jump, static checker) and visual geoms >= 3 mm from board / table.  Neighbours are
         NOT checked during the slide (the destination cell must be empty anyway)."""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
HERE = Path(__file__).resolve().parent
SH = HERE.parents[1]
sys.path.insert(0, str(SH / "dataset_bd_classquota_without_home_position_no_overpass/code/shelf66_v3"))
import bd_shelf  # noqa: E402

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

S = bd_shelf.env()["TS"]
G2_XML = SH / "hardware/g2_mjcf/xarm7_g2.xml"
ZB = S.BASE_Z                                                   # 0.12: board top = robot base-plate bottom
TARGET = dict(shape="cylinder", size=0.018, height=0.14)       # VAL 3D targets: all cylinders, median r 1.79 cm, h 14.1 cm
NBRS = {"tall_box": dict(shape="box", size=0.018, height=0.18),        # worst case in VAL (largest size, tallest)
        "median_cyl": dict(shape="cylinder", size=0.0147, height=0.147)}  # VAL non-target medians
DIRS = {"+x": (1, 0), "-x": (-1, 0), "+y": (0, 1), "-y": (0, -1)}
CLR_NBR, CLR_STATIC = S.OBJECT_MARGIN, S.STATIC_MARGIN
OUT = HERE / "results.jsonl"

S.SHELF66["zb"] = ZB
S.ARM_CEILING = False
_conf0 = S.SCENES["confined"]


def _confined(*a, **kw):
    sc = _conf0(*a, **kw)
    drop = ('name="front_step"', 'name="ceiling"', 'name="back_wall"')
    xf, hw = sc["x0"] - sc["rows"] * sc["pitch"] / 2 - 0.04, sc["cols"] * sc["pitch"] / 2 + 0.16
    sc["statics"] = [g for g in sc["statics"] if not any(k in g for k in drop)] + \
        [S._slab("hw_table", -0.40, xf, -hw, hw, 0.0, ZB, "0.55 0.45 0.35 1")]
    return sc


S.SCENES["confined"] = _confined


def make_env(cell, nbr):
    r, c = cell
    objs = [dict(name="target", color="green", cell=cell, **TARGET)]
    for k, (dr, dc) in enumerate(((1, 0), (-1, 0), (0, 1), (0, -1))):
        if 0 <= r + dr < 6 and 0 <= c + dc < 6:
            objs.append(dict(name=f"n{k}", color="red", cell=(r + dr, c + dc), **NBRS[nbr]))
    e = S.XArm7TableEnv(objs, scene=S.make_scene("confined"), verbose=False, xml_path=str(G2_XML))
    e.object_check = False                                     # neighbours are measured separately (visual geoms included)
    m = e.model
    robot = set(e._subtree(m, m.body("link_base").id))
    e._rg = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in robot and m.geom_bodyid[g] != m.body("link_base").id])
    e._nb = [g for o in e.objects[1:] for g in np.flatnonzero(m.geom_bodyid == m.body(o["name"]).id)]
    e._st = [m.geom(n).id for n in ("floor_board", "base", "hw_table")]
    return e


def min_dist(e, q, width, others):
    """Smallest distance (m, capped at 5 cm) from any robot geom (visual included) to `others`, fingers at pad gap `width`."""
    m, d = e.model, e.ik_data
    d.qpos[:] = e.data.qpos
    d.qpos[e.qadr] = q
    d.qpos[e.grip_qadr] = e._finger_q(width)
    mujoco.mj_kinematics(m, d)
    best, e._who = 0.05, ""
    for h in others:
        if m.geom_type[h] == mujoco.mjtGeom.mjGEOM_BOX and m.geom_size[h][0] > 0.1:   # big slabs: no bounding-sphere filter
            near = e._rg
        else:
            near = e._rg[np.linalg.norm(d.geom_xpos[e._rg] - d.geom_xpos[h], axis=1) < m.geom_rbound[e._rg] + m.geom_rbound[h] + 0.05]
        for g in near:
            dist = mujoco.mj_geomDistance(m, d, int(g), int(h), 0.05, None)
            if dist < best:
                best, e._who = dist, m.body(m.geom_bodyid[g]).name
    return best


def run_cell(cell, nbr):
    t0 = time.time()
    e = make_env(cell, nbr)
    o = e.objects[0]
    p0 = o["pos"]
    width = 2 * o["size"]
    open_w, shut_w = width + 2 * S.FINGER_CLEARANCE, width - 0.001
    zs = {"middle": ZB + o["height"] / 2, "top": ZB + max(0.015, o["height"] - S.GRASP_TOP_MARGIN)}
    grasps = []
    for yaw in (0, 90):
        for az in (yaw, yaw + 180):
            R = S.grasp_frame(az, 0.0)
            for lvl, z in zs.items():
                p = np.array([p0[0], p0[1], z])
                for q in e.ik_candidates(p, R, e.q_home, limit=4):
                    try:
                        path = e.chain(q, p, [p + [0, 0, S.PRE_DIST]], R)
                    except RuntimeError as err:
                        grasps.append(dict(yaw=yaw, az=az, level=lvl, reach=False, why=str(err)[:80]))
                        continue
                    clr, who = min(((min_dist(e, qq, open_w, e._nb), e._who) for qq in path), key=lambda t: t[0]) if e._nb else (0.05, "")
                    stc = min(min_dist(e, qq, open_w, e._st) for qq in path)
                    grasps.append(dict(yaw=yaw, az=az, level=lvl, reach=stc >= CLR_STATIC, why="" if stc >= CLR_STATIC else "visual mesh near board",
                                       clr_mm=round(clr * 1000, 1), clr_body=who, static_mm=round(stc * 1000, 1), q=q, p=p, R=R))
    ok = [g for g in grasps if g["reach"]]
    use = [g for g in ok if g["clr_mm"] >= CLR_NBR * 1000] or ok   # slide from the grasps one would use (clear ones if any)
    slide = {}
    for dn, (dx, dy) in DIRS.items():
        slide[dn] = False
        for g in use:
            try:
                path = e.chain(g["q"], g["p"], [g["p"] + [dx * e.pitch, dy * e.pitch, 0]], g["R"])
            except RuntimeError:
                continue
            if min(min_dist(e, qq, shut_w, e._st) for qq in path) >= CLR_STATIC:
                slide[dn] = True
                break
    best = max(ok, key=lambda g: g["clr_mm"]) if ok else None
    clear_yaws = sorted({g["yaw"] for g in ok if g["clr_mm"] >= CLR_NBR * 1000})
    return dict(cell=list(cell), nbr=nbr, n_nbrs=len(e._nb), reach=bool(ok), clear=bool(clear_yaws), clear_yaws=clear_yaws,
                reach_yaws=sorted({g["yaw"] for g in ok}), best_clr_mm=best and best["clr_mm"], best_level=best and best["level"],
                best_yaw=best and best["yaw"], best_clr_body=best and best["clr_body"], slide_from="clear" if use and use[0]["clr_mm"] >= CLR_NBR * 1000 else "reach-only",
                slide=slide, fails=collections.Counter(g["why"].split(" near")[0] for g in grasps if not g["reach"]),
                n_grasps_tried=len(grasps), sec=round(time.time() - t0, 1))


def cells(part):
    i, n = map(int, part.split("/"))
    done = {(tuple(r["cell"]), r["nbr"]) for r in map(json.loads, open(OUT))} if OUT.exists() else set()
    jobs = [((r, c), nb) for nb in NBRS for r in range(6) for c in range(6)][i::n]
    with open(OUT, "a") as f:
        for cell, nb in jobs:
            if (cell, nb) in done:
                continue
            rec = run_cell(cell, nb)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(cell, nb, rec["reach"], rec["clear_yaws"], rec["best_clr_mm"], rec["slide"], rec["sec"], "s", flush=True)


def val():
    """Front-column rule on the 3D VAL scenes: target blocked if any object stands in its column between it and row 0."""
    ABF = SH / "approachb_3d_learning/runs/ABF3"
    remote = ("/home/projects/long-horizon/naren_workspace/k3abl/abf/", "/home/projects/long-horizon/abf_v2/root/")
    local = lambda p: next((str(ABF / p[len(r):]) for r in remote if p.startswith(r)), p)  # noqa: E731  (= height_check.local)
    rows = [json.loads(x) for x in open(SH / "approachb_3d_learning/runs/K3ABL/runs/public/private_val_3d.jsonl")]
    byK = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        ic = json.load(open(local(r["path"])))["initial_cells"]
        tr, tc = ic["t0"]
        blocked = any(c == tc and rr < tr for k, (rr, c) in ic.items() if k != "t0")
        byK[r["K"]][0] += blocked; byK[r["K"]][1] += 1
    tot = [sum(v[0] for v in byK.values()), sum(v[1] for v in byK.values())]
    res = dict(split="VAL (K3ABL runs/public/private_val_3d.jsonl)", blocked=tot[0], n=tot[1], frac=round(tot[0] / tot[1], 3),
               by_K={k: dict(blocked=v[0], n=v[1]) for k, v in sorted(byK.items())})
    (HERE / "val.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))


def report():
    R = [json.loads(x) for x in open(OUT)]
    out = ["# topdown_check summary (generated, EXPLORATORY, sim kinematics only)\n",
           "Per cell: R reach / C neighbour clearance >= 5 mm / yaw that is clear (0, 9 = 90, B = both, - none) / "
           "slide directions that work (f = -x toward robot, b = +x, l = +y, r = -y). Row 5 back, row 0 next to the robot; "
           "col 0 = robot's right (-y).\n"]
    for nb in NBRS:
        rs = {tuple(r["cell"]): r for r in R if r["nbr"] == nb}
        out.append(f"\n## neighbours: {nb} {NBRS[nb]}  (target {TARGET})\n```")
        for row in reversed(range(6)):
            line = []
            for col in range(6):
                r = rs.get((row, col))
                if not r:
                    line.append("   ?    "); continue
                y = {(): "-", (0,): "0", (90,): "9", (0, 90): "B"}[tuple(r["clear_yaws"])]
                s = "".join(k if r["slide"][d] else "." for d, k in (("-x", "f"), ("+x", "b"), ("+y", "l"), ("-y", "r")))
                line.append(f"{'R' if r['reach'] else 'x'}{'C' if r['clear'] else 'x'}{y}{s} ")
            out.append(f"r{row}  " + " ".join(line))
        out.append("```")
        n = len(rs)
        clr = sorted(r["best_clr_mm"] for r in rs.values() if r["best_clr_mm"] is not None)
        out.append(f"reach {sum(r['reach'] for r in rs.values())}/{n}; clear {sum(r['clear'] for r in rs.values())}/{n}; "
                   f"best clearance mm min/median {clr[0] if clr else '-'} / {clr[len(clr) // 2] if clr else '-'}; slide per direction: "
                   + ", ".join(f"{d} {sum(r['slide'][d] for r in rs.values())}/{n}" for d in DIRS)
                   + f"; all 4 slides {sum(all(r['slide'].values()) for r in rs.values())}/{n}; "
                   f"grasp level used {dict(collections.Counter(r['best_level'] for r in rs.values()))}; "
                   f"slides from reach-only grasps {sum(r['slide_from'] == 'reach-only' for r in rs.values())}")
    if (HERE / "val.json").exists():
        v = json.loads((HERE / "val.json").read_text())
        out.append(f"\n## VAL front-column rule\nblocked {v['blocked']}/{v['n']} = {v['frac']}; by K: {v['by_K']}")
    (HERE / "summary.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cells", "val", "report", "one"])
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--cell", default="0,0")
    a = ap.parse_args()
    if a.cmd == "cells":
        cells(a.part)
    elif a.cmd == "val":
        val()
    elif a.cmd == "one":
        r = run_cell(tuple(map(int, a.cell.split(","))), "tall_box")
        print(json.dumps(r, indent=1))
    else:
        report()
