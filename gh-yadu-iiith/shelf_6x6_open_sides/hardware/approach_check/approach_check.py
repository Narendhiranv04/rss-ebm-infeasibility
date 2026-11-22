"""Single-approach-direction pick check on the SHELF66 V3 shelf (the one the dataset teacher uses).

For each case (approach set, object, neighbour layout, cell): plan a pick (pre-shape, free move to hover, straight approach,
close, lift 2.5 cm, pull straight back 13 cm holding the object) with the dataset's own planner pieces and checks
(no-overpass columns, arm ceiling, pre-shaped fingers, 3 mm / 5 mm clearances), then EXECUTE it in MuJoCo physics and
measure the neighbours (displacement, tilt) and the geometric clearance robot -> neighbour, visual finger meshes included.

    MUJOCO_GL=disable nice -n 10 <python> approach_check.py --stage A        # or B, G2A; --cells all|cross; --limit N
Writes results.jsonl (one line per case, resumable) next to this file.  Read-only use of the dataset / scene code."""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "disable")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]                                   # shelf_6x6_open_sides/
sys.path.insert(0, str(ROOT / "dataset_bd_classquota_without_home_position_no_overpass/code/shelf66_v3"))
import bd_shelf  # noqa: E402  (the teacher's env(): SHELF66 V3 scene module, RRT max_iters=5000, no wall-clock)

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

S = bd_shelf.env()["TS"]
G2_XML = ROOT / "hardware/g2_mjcf/xarm7_g2.xml"
OUT = HERE / "results.jsonl"

OBJECTS = {"T": dict(shape="cylinder", size=0.020, height=0.12),      # target cylinder (README extreme: r 2.0)
           "narrow": dict(shape="cylinder", size=0.013, height=0.12),
           "wide": dict(shape="box", size=0.016, height=0.18)}
NEIGHBOURS = {"cyl": dict(shape="cylinder", size=0.018, height=0.18), "box": dict(shape="box", size=0.016, height=0.18)}
LAYOUTS = {"N0": [], "N1": [("L", 0, 1), ("R", 0, -1)],                # (name, d_row, d_col); left = +y = col + 1
           "N2": [("L", 0, 1), ("R", 0, -1), ("B", 1, 0), ("BL", 1, 1), ("BR", 1, -1)]}
CROSS = sorted({(r, c) for r in range(6) for c in range(6) if r in (0, 2, 5) or c in (0, 2, 5)})
ALL = [(r, c) for r in range(6) for c in range(6)]
KNOCK_TILT, KNOCK_DISP = 10.0, 0.005
EVERY = 5                                                # measure every 5 physics steps (10 ms)


def classify(msg, real_hit):
    if msg.startswith("free move"):
        return "free-move RRT failed"
    if "would hit the" in msg:                           # static structure / self
        part = msg.split("would hit the ")[1].split(" near")[0]
        return "arm ceiling" if "arm_ceiling" in part else "self-collision" if part == "itself" else f"shelf ({part})"
    if "would hit" in msg:
        return "neighbour collision" if real_hit else "no-overpass column"
    if "out of reach" in msg or "IK jumps" in msg:
        return "IK / reach" if "structure" not in msg else "IK (no structure-free solution)"
    return "other: " + msg[:60]


def make_env(obj, layout, nbr, cell, azimuths, xml):
    r, c = cell
    objs = [dict(name="target", color="green", cell=cell, **OBJECTS[obj])]
    for nm, dr, dc in LAYOUTS[layout]:
        if 0 <= r + dr < 6 and 0 <= c + dc < 6:          # neighbours that would fall off the grid are left out
            objs.append(dict(name="n" + nm, color="red", cell=(r + dr, c + dc), **NEIGHBOURS[nbr]))
    scene = S.make_scene("confined")
    scene["azimuths"] = list(azimuths)
    e = S.XArm7TableEnv(objs, scene=scene, verbose=False, xml_path=str(xml))
    orig = e._object_hits

    def object_hits(q, margin=S.OBJECT_MARGIN, every=False):   # also note whether a REAL object (not only its column) was hit
        hits = orig(q, margin, every)
        if hits:
            cols, e._col_ids = e._col_ids, []
            try:
                e._real_hit = bool(orig(q, margin, every))
            finally:
                e._col_ids = cols
        return hits
    e._object_hits = object_hits
    return e


def plan_pick(e, g, seed=0):
    """plan_pick_place's approach + lift, then a straight pull-back with the object instead of the carry.  -> (segs, reason)."""
    e._focus(0)
    q0 = e.q_now()
    e._check_width = g["open_width"]
    reason, reach_fail = "", False
    try:
        for branch in range(e.n_branches):
            e._real_hit = False
            try:
                segs, q_g = e._plan_approach(g, q0, seed, branch)
                pg, R = g["position"], g["R"]
                lift = pg + np.array([0, 0, S.CARRY_LIFT])
                ch = e.chain(q_g, pg, [lift], R)
                back = e.chain(ch[-1], lift, [lift - g["approach"] * S.PRE_DIST], R)
                segs += [dict(phase="close", kind="grip", grip=None), dict(phase="lift", kind="cart", q=ch),
                         dict(phase="retreat", kind="cart", q=back)]
                return segs, dict(branch=branch, teacher_rule=not reach_fail)
            except (RuntimeError, ValueError) as err:
                reason = reason or classify(str(err), e._real_hit) + " | " + str(err)[:120]
                # plan_pick_place stops at the first "out of reach" from the ready pose; other IK branches often do reach
                # (row 0), so all are tried here and teacher_rule records whether the dataset teacher would have found it
                reach_fail = reach_fail or "out of reach" in str(err)
                if isinstance(err, S.ReachError):
                    break
    finally:
        e._check_width = None
    return None, reason


def execute(e, segs):
    """Run the plan in physics; track neighbours (max displacement / tilt) and min distance robot -> neighbour."""
    m, d = e.model, e.data
    robot = set(e._subtree(m, m.body("link_base").id))
    hand = set(e._subtree(m, m.body(S.HAND_ROOT).id))
    rg = np.array([g for g in range(m.ngeom) if m.geom_bodyid[g] in robot])       # all robot geoms, visual meshes too
    nb = [(o["name"], m.body(o["name"]).id, np.flatnonzero(m.geom_bodyid == m.body(o["name"]).id)) for o in e.objects[1:]]
    tid = m.body("target").id
    p0 = {n: d.xpos[b].copy() for n, b, _ in nb}
    st = dict(disp={n: 0.0 for n, _, _ in nb}, tilt={n: 0.0 for n, _, _ in nb}, clr=np.inf, clr_hand=np.inf, clr_body="",
              clr_phase="", clr_target=np.inf, contact=set(), knock_phase="", phase="")
    ncoll = set(np.flatnonzero([m.geom_bodyid[g] in robot for g in range(m.ngeom)]).tolist())
    nbgeoms = {g: n for n, _, gs in nb for g in gs}
    fromto = np.zeros(6)

    def measure():
        for n, b, gs in nb:
            dp = float(np.linalg.norm(d.xpos[b] - p0[n]))
            tl = math.degrees(math.acos(np.clip(d.xmat[b][8], -1, 1)))
            if (dp > KNOCK_DISP or tl > KNOCK_TILT) and not st["knock_phase"]:
                st["knock_phase"] = st["phase"]
            st["disp"][n] = max(st["disp"][n], dp)
            st["tilt"][n] = max(st["tilt"][n], tl)
            if st["phase"] in ("open", "to_hover"):
                continue                                 # clearance counted from the hover pose on (approach .. retreat)
            for h in gs:
                near = rg[np.linalg.norm(d.geom_xpos[rg] - d.geom_xpos[h], axis=1) < m.geom_rbound[rg] + m.geom_rbound[h] + 0.05]
                for g in near:
                    dist = mujoco.mj_geomDistance(m, d, int(g), int(h), 0.05, fromto)
                    if dist < st["clr"]:
                        st["clr"], st["clr_body"], st["clr_phase"] = dist, m.body(m.geom_bodyid[g]).name, st["phase"]
                    if m.geom_bodyid[g] in hand:
                        st["clr_hand"] = min(st["clr_hand"], dist)
                for g in m.body_geomadr[tid] + np.arange(m.body_geomnum[tid]):
                    if st["phase"] in ("lift", "retreat"):
                        st["clr_target"] = min(st["clr_target"], mujoco.mj_geomDistance(m, d, int(g), int(h), 0.05, fromto))
        for k in range(d.ncon):
            a, b = d.contact[k].geom1, d.contact[k].geom2
            if (a in ncoll and b in nbgeoms) or (b in ncoll and a in nbgeoms):
                st["contact"].add(nbgeoms.get(a, nbgeoms.get(b)))

    count = [0]

    def step(n=1):
        for _ in range(n):
            mujoco.mj_step(m, d)
            count[0] += 1
            if count[0] % EVERY == 0:
                measure()
    e.step = step

    lifted = held = False
    o = e.objects[0]
    for seg in segs:
        st["phase"] = seg["phase"]
        if seg["kind"] == "grip":
            if seg["phase"] == "close":
                e.set_gripper(False); e.wait(400)
            else:
                e.set_gripper_width(seg["grip"]); e.wait(200)
            continue
        e.execute(e.segment_traj(seg), settle=40)
        z = e.obj_pos(0)[2] - o["height"] / 2 - e.table_z
        if seg["phase"] == "lift":
            lifted = z > 0.3 * S.CARRY_LIFT
        if seg["phase"] == "retreat":
            st["phase"] = "hold"
            e.wait(200)
            z = e.obj_pos(0)[2] - o["height"] / 2 - e.table_z
            held = bool(z > 0.3 * S.CARRY_LIFT and np.linalg.norm(e.obj_pos(0)[:2] - e.grasp_pos()[:2]) < 0.03)
    dn = st["disp"]
    knocked = [n for n in dn if dn[n] > KNOCK_DISP or st["tilt"][n] > KNOCK_TILT]
    f = lambda v: None if not np.isfinite(v) else round(float(v) * 1000, 1)
    return dict(lifted=bool(lifted), held_after_retreat=held, knocked=knocked, knock_phase=st["knock_phase"],
                max_disp_mm={n: round(v * 1000, 1) for n, v in dn.items()}, max_tilt_deg={n: round(v, 1) for n, v in st["tilt"].items()},
                min_clear_mm=f(st["clr"]), min_clear_hand_mm=f(st["clr_hand"]), min_clear_body=st["clr_body"],
                min_clear_phase=st["clr_phase"], min_clear_target_mm=f(st["clr_target"]), contact=sorted(st["contact"]))


def run_case(gripper, azset, obj, layout, nbr, cell, xml):
    t0 = time.time()
    az = [180.0] if azset == "A" else [22.5 * k for k in range(16)]
    e = make_env(obj, layout, nbr, cell, az, xml)
    rec = dict(gripper=gripper, azset=azset, obj=obj, layout=layout, nbr=nbr if layout != "N0" else "-", cell=list(cell),
               n_neighbours=len(e.objects) - 1, planned=False, tried=[])
    for g in S.grasp_poses(e.objects[0], e.table_z):
        segs, reason = plan_pick(e, g)
        if segs is not None:
            rec.update(reason)
        else:
            rec["tried"].append([g["azimuth_deg"], g["grasp_level"], reason])
            continue
        rec.update(planned=True, azimuth=g["azimuth_deg"], grasp_level=g["grasp_level"])
        rec.update(execute(e, segs))
        break
    rec["fail_reason"] = "" if rec["planned"] else rec["tried"][0][2].split(" | ")[0]
    rec["sec"] = round(time.time() - t0, 1)
    return rec


def cases(stage, cellset):
    big = ALL if cellset == "all" else CROSS
    if stage in ("A", "G2A"):
        sets, nbrs, grip = ["A"], (["cyl"] if stage == "G2A" else ["cyl", "box"]), "G2" if stage == "G2A" else "std"
    else:
        sets, nbrs, grip = ["B"], ["cyl"], "std"
    for azset in sets:
        for obj in OBJECTS:
            for layout in LAYOUTS:
                for nbr in (["cyl"] if layout == "N0" else nbrs):
                    for cell in (ALL if (obj == "T" and layout == "N0") else big):
                        yield grip, azset, obj, layout, nbr, cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["A", "B", "G2A"], required=True)
    ap.add_argument("--cells", choices=["cross", "all"], default="cross")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    xml = G2_XML if a.stage == "G2A" else Path(S.XARM7_XML)
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            r = json.loads(line)
            done.add((r["gripper"], r["azset"], r["obj"], r["layout"], r["nbr"], tuple(r["cell"])))
    n = 0
    with open(a.out, "a") as f:
        for k in cases(a.stage, a.cells):
            key = (k[0], k[1], k[2], k[3], k[4] if k[3] != "N0" else "-", tuple(k[5]))
            if key in done:
                continue
            done.add(key)
            rec = run_case(*k, xml)
            f.write(json.dumps(rec) + "\n"); f.flush()
            print(f"{key} planned={rec['planned']} lifted={rec.get('lifted')} knocked={rec.get('knocked')} "
                  f"clr={rec.get('min_clear_mm')} {rec['fail_reason']} {rec['sec']}s", flush=True)
            n += 1
            if a.limit and n >= a.limit:
                break


if __name__ == "__main__":
    main()
