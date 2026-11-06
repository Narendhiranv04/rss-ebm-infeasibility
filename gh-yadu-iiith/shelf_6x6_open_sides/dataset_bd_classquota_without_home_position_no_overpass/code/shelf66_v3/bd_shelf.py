# SHELF66_V3 COPY (2026-11-05) of shelf66_full/: identical except shelf_scene_6x6/shelf_scene.py (the fingers are checked PRE-SHAPED -- GRIP_JOINTS --
# and pre-shaped before every pick) and the version strings (generator SHELF66V3, teacher BD3DS66_TEACHER_V2).  shelf66_full/ = V2, unchanged.
"""SHELF66_FULL COPY (2,500-scene run; identical to shelf66/bd_shelf.py apart from this line: it imports the bd_3dt / bd_3dt_plus
copies of shelf66_full/ and the scene copy shelf66_full/shelf_scene_6x6).  SHELF66 COPY: the 6 x 6, 7.5 cm, open-sided shelf of shelf66/shelf_scene_6x6 (no-overpass columns, arm ceiling = tallest + 10 cm,
16 level grasp directions, mid-height grip with a near-the-top fallback).  Uses the SHELF66 copies of bd_3dt / bd_3dt_plus in this
folder (grid size from the scene); dataset-bd's bd_core is shared.  Root ids: domain "3ds66".  Original text follows.

CONFINED-SHELF domain for the B/D dataset: bd_3dt + bd_3dt_plus (construction, teacher, record) run unchanged on the 8 x 8
confined shelf of shelf_scene.py instead of the open table.  dataset-bd's files and bd_3dt_plus.py are imported, never modified:
importing this module points bd_3dt's env() at shelf_scene (before anything else calls it).

    import bd_shelf            # first; then use bd_3dt / bd_3dt_plus / make_combined_3dt as usual

shelf_scene.py (SHELF_DIR, default ~/Downloads): floor 0.50 m, 0.60 m clear height, 6.5 cm pitch, rows x = 0.31 .. 0.77 m, 8 level
grasps (every 45 deg; only those from the front side reach into the shelf), the arm / open hand checked against the shelf,
itself and every object but the one handled.  Cells (row, col) as in table_scene: row 0 nearest the robot, col 0 the -y edge.
Moves chain (teacher V2): the arm starts at the ready pose and, after each relocation, goes straight from its retreat to the next
pick -- no return to the ready pose (shelf_scene.RETURN_TO_READY, recorded in the geometry / protocol)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dataset-bd"))                 # bd_core (shared)
sys.path.insert(0, str(HERE))                                       # this folder's bd_3dt / bd_3dt_plus copies come first
SHELF_DIR = os.environ.get("SHELF66_DIR", str(HERE / "shelf_scene_6x6"))   # never the old SHELF_DIR scenes
sys.path.insert(2, SHELF_DIR)

import bd_core as bc  # noqa: E402
import bd_3dt as A  # noqa: E402

DOMAIN = "3ds66"
SCENE_FILE = "shelf_scene.py"
A.TEACHER_VERSION = (f"BD3DS66_TEACHER_V2:shelf_scene[confined].plan_pick_place(first feasible level grasp, fingers pre-shaped in every check, RRT max_iters={A.RRT_ITERS}, "
                     "no wall-clock)+MuJoCo run_plan, moves chained (no return to the ready pose between them)")
_record0 = A.record


def env():
    if not A._G:
        import shelf_scene as S
        S.select_scene("confined")
        scene = S.make_scene("confined")
        S.apply_scene_grasps(scene)                      # module-level grasp set = the shelf's (also set by every env)
        S.N_AZIMUTH = len(scene["azimuths"])
        A.set_grid(scene["rows"], scene["cols"])
        base = S.RRTConnect

        class RRT(base):                                 # the scene's planner with its iteration ceiling only (no wall-clock cut-off)
            def __init__(self, *a, **kw):
                kw["max_time"], kw["max_iters"] = 1e9, A.RRT_ITERS
                super().__init__(*a, **kw)
        S.RRTConnect = RRT
        src = {f: bc.sha(open(os.path.join(SHELF_DIR, f), "rb").read().hex())[:12] for f in (SCENE_FILE, "rrt_connect.py", "rrt_star.py")}
        lv = scene["levels"][0]
        geo = {"scene": "confined", "pitch": scene["pitch"], "x0": scene["x0"], "table_z": lv["z"], "clear_height": lv["clearance"],
               "azimuths": list(scene["azimuths"]), "tilts": list(scene["tilts"]), "lift": S.CARRY_LIFT,
               "return_to_ready": getattr(S, "RETURN_TO_READY", True),       # older shelf_scene.py files always returned
               "rows": scene["rows"], "cols": scene["cols"], "grasp_at": S.GRASP_AT, "no_overpass": S.NO_OVERPASS, "arm_ceiling": S.ARM_CEILING,
               "ceiling_margin": S.CEILING_MARGIN, "side_walls": S.SHELF66["side_walls"]}
        A._G.update(TS=S, protocol="XARM7_SHELF66_LEVELGRASP_V1:" + bc.sha(geo)[:12], geometry=geo, source_files=src, scene=scene,
                    config_hash=bc.sha({"v": A.TEACHER_VERSION, "geo": geo, "rrt": A.RRT_ITERS, "src": src})[:16])
    return A._G


def build(objs, cells):
    """bd_3dt.build with ShelfEnv: shelf_scene's env, whose cell_of() gives (row, col) like table_scene's (the shelf has one level;
    shelf_scene itself says (row, col, level)), so bd's 'mover ended in its destination cell' checks compare like with like."""
    S = env()["TS"]
    if "ShelfEnv" not in A._G:
        class ShelfEnv(S.XArm7TableEnv):
            def cell_of(self, name):
                c = super().cell_of(name)
                return None if c is None else c[:2]
        A._G["ShelfEnv"] = ShelfEnv
    return A._G["ShelfEnv"]([{**{k: o[k] for k in ("shape", "size", "height", "color")}, "name": o["id"], "cell": tuple(cells[o["id"]])}
                             for o in objs], verbose=False)


def record(spec, ids, c, t, store) -> dict:
    rec = _record0(spec, ids, c, t, store)
    rec.update(domain=DOMAIN, cell_convention="shelf_scene confined shelf (row, col), 0-based, row 0 nearest the robot; centre = "
               "(x0 + (r-2.5)*pitch, (c-2.5)*pitch) on the 6 x 6 grid, on the shelf floor (table_z)")
    return rec


A.DOMAIN = DOMAIN
A.record = record
A.build = build
env()

import bd_3dt_plus as P  # noqa: E402
_reach0 = P.reach_azimuths


def reach_azimuths():
    """bd_3dt_plus.reach_azimuths on the empty shelf: its reference object is moved from cell to cell only in the grasp
    geometry, so the arm-vs-objects check (which sees the object where it stands) is off while the map is made."""
    build0 = A.build

    def build(objs, cells):
        e = build0(objs, cells)
        e.object_check = False
        return e
    A.build = build
    try:
        return _reach0()
    finally:
        A.build = build0


P.reach_azimuths = reach_azimuths
