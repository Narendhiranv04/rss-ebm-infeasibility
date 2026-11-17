"""Conflict library for REAP training on the 6x6 shelf: REAP's own idea (TAMPanda's grasp library), built from our teacher.

REAP's confined-shelf checker answers in microseconds from precomputed motions with per-cell conflict sets. Our objects
differ per scene, so the library is keyed by a CLASS:

    mover class  (shape, size bucket, height bucket)      20 classes cover every object of V4
    ceiling      the scene's arm ceiling bucket            (tallest object; NO_OVERPASS arm ceiling = tallest + 0.10)

For each class + ceiling, every motion is planned ONCE by the teacher (shelf66_v3 plan_pick_place / _plan_approach) in an
otherwise empty shelf, from the ready pose, with up to K grasps per motion:
    pick (cell)        free move ready -> hover, straight approach            (REAP's pick filter)
    move (src, dst)    the teacher's whole pick-carry-place                   (REAP's put)
For every planned motion and every other cell c the build stores two CLEARANCES, measured with a probe obstacle of the
largest footprint radius R in c:
    arm[c]   = min over the motion's waypoints of (distance robot <-> the probe's no-overpass column - margin)
               (margin = OBJECT_MARGIN on cart segments; the free move's growing FREE_MOVE_MARGIN schedule)
    carry[c] = min over the carry route of the footprint rule's slack before the obstacle radius is subtracted
An obstacle of footprint radius r in c blocks the motion iff  arm[c] < r - R  or  carry[c] < r
(a smaller obstacle at the same centre is R - r farther from the arm).
A motion is feasible in a layout iff some stored candidate has no blocking obstacle. Lookups are numpy compares.

Conservative by construction: size buckets use the larger size, the ceiling bucket the lower ceiling, and a stored
motion never detours (the teacher's RRT may route around an obstacle). validate_lib.py measures the agreement with the
teacher. Test-time evaluation never uses the library.
"""
from __future__ import annotations

import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from shelf66_oracle import Shelf66Oracle, _teacher

LIB_DIR = Path(os.environ.get("YADU_LIB_DIR", Path.home() / "reap_lib66v2"))
K_CANDIDATES = int(os.environ.get("YADU_LIB_K", 3))
MAX_GRASP_TRIES = int(os.environ.get("YADU_LIB_TRIES", 12))      # grasps tried per motion (the teacher's preference order)
# v2 (2026-11-15): the free move ready -> hover is stored apart from the rest, split at FREE_TAIL_FROM of its length.
# The teacher's joint-space RRT re-routes it around objects; the stored path cannot. FREE_MODE: strict = all of it,
# tail = only its last part (in front of the shelf, where a detour is hardest), ignore = not checked.
FREE_TAIL_FROM = 0.7
FREE_MODE = os.environ.get("YADU_LIB_FREE", "tail")             # strict | tail | ignore
PICK_MODE = os.environ.get("YADU_LIB_PICK", "lenient")            # lenient | strict (see LibOracle.pick)
R_TOL = float(os.environ.get("YADU_LIB_TOL", "0"))   # clearance tolerance (m) subtracted from every stored-clearance test
EXT_K, EXT_TRIES, EXT_FRONTAL_DEG = 2, 8, 22.6                    # v3 extension: 2 non-frontal motions, <= 8 tries
LIB_EXT = os.environ.get("YADU_LIB_EXT", "0") == "1"             # query: also use the ext_src_*.pkl files (v3)
R_PROBE = 0.0230                       # >= the largest footprint radius in V4 (0.0226)
SIZE_EDGES, SIZE_REP = (0.0145, 0.0170), (0.0145, 0.0170, 0.0200)
HEIGHT_EDGES, HEIGHT_REP = (0.09, 0.12, 0.15), (0.075, 0.105, 0.135, 0.165)
CEIL_EDGES, CEIL_REP = (0.15, 0.17), (0.120, 0.150, 0.170)          # lower edge of each bucket (lower ceiling)
ROWS = COLS = 6


def bucket(v, edges):
    return sum(v >= e for e in edges)


def mover_class(o):
    return (o["shape"], bucket(o["size"], SIZE_EDGES), bucket(o["height"], HEIGHT_EDGES))


def ceil_class(objects):
    return bucket(max(o["height"] for o in objects), CEIL_EDGES)


def combo_name(mc, cc):
    return f"{mc[0]}_s{mc[1]}_h{mc[2]}_c{cc}"


def cells():
    return [(r, c) for r in range(ROWS) for c in range(COLS)]


# ===================================================================================== build
class Builder:
    """One env: the mover (class representative) + 35 probe cylinders of radius R_PROBE and the ceiling-bucket height."""

    def __init__(self, mc, cc):
        self.A = A = _teacher()
        self.S = S = A.env()["TS"]
        shape, sb, hb = mc
        h = HEIGHT_REP[hb]
        hc = max(CEIL_REP[cc], h)                                        # the ceiling never below the mover itself
        self.mover = dict(id="m", shape=shape, size=SIZE_REP[sb] if shape == "cylinder" else min(SIZE_REP[sb], 0.017),
                          height=h, color="green")
        probes = [dict(id=f"p{k:02d}", shape="cylinder", size=R_PROBE, height=hc, color="red") for k in range(35)]
        allc = cells()
        start = {"m": allc[0], **{p["id"]: allc[k + 1] for k, p in enumerate(probes)}}
        self.e = A.build([self.mover] + probes, start)
        self.probe_ids = [p["id"] for p in probes]
        e = self.e
        self.pj = [e.obj_id(p) for p in self.probe_ids]
        self.col = [e._col_ids[j] for j in self.pj]

    # -- layout without simulation (as shelf66_table._place)
    def place(self, cells_map):
        e = self.e
        for j, o in enumerate(e.objects):
            a = e.obj_qadr[j]
            c = cells_map.get(o["name"])
            if c is None:
                e.data.qpos[a:a + 3] = [5.0 + j, 5.0, e.table_z + 0.1]
            else:
                x, y = e.cell_xy(tuple(c))[:2]
                e.data.qpos[a:a + 3] = [x, y, e.levels[0]["z"] + o["height"] / 2 + 0.0005]
                o["pos"] = np.array(e.data.qpos[a:a + 3])
            e.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        e.data.qpos[e.qadr] = e.q_ready
        self.S.mujoco.mj_forward(e.model, e.data)

    # -- clearances of one robot configuration to every probe column (distance - margin), +inf when far
    def col_clear(self, q, margin, width):
        S, e = self.S, self.e
        m, d = e.model, e.ik_data
        d.qpos[:] = e.data.qpos
        d.qpos[e.qadr] = q
        if width is not None:
            d.qpos[e.grip_qadr] = e._finger_q(width)
        for j, c in zip(self.pj, self.col):
            m.geom_pos[c, :2] = e.obj_pos(j)[:2]
        S.mujoco.mj_kinematics(m, d)
        rg = e._robot_geoms
        out = np.full(len(self.pj), np.inf)
        cut = 0.06
        for k, (j, c) in enumerate(zip(self.pj, self.col)):
            if np.linalg.norm(e.obj_pos(j)[:2]) > 4.0:
                continue
            near = np.linalg.norm(d.geom_xpos[rg, :2] - d.geom_xpos[c, :2], axis=1) < m.geom_rbound[rg] + R_PROBE + cut
            for g in rg[near]:
                dist = S.mujoco.mj_geomDistance(m, d, int(g), int(c), cut, None)
                out[k] = min(out[k], dist - margin)
        return out

    def motion_clearance(self, segs, width, route, g, src, dst):
        """arm[36], carry[36] for one planned motion (probes in every cell except src / dst)."""
        S = self.S
        allc = cells()
        free = [c for c in allc if c != tuple(src) and (dst is None or c != tuple(dst))]
        cm = {"m": tuple(src), **{pid: c for pid, c in zip(self.probe_ids, free)}}     # spare probe(s) stay parked
        self.place(cm)
        self.e._focus(self.e.obj_id("m"))
        arm = np.full(len(self.pj), np.inf)
        head = np.full(len(self.pj), np.inf); tail = np.full(len(self.pj), np.inf)
        for seg in segs:
            if seg["kind"] == "grip":
                continue
            qs = [np.asarray(q, float) for q in seg["q"]]
            if seg["phase"] == "to_hover" and seg["kind"] == "joint":
                qa, qb = qs[0], qs[-1]
                start = self.col_clear(qa, S.OBJECT_MARGIN, width)            # leaving a column it starts in is allowed
                excused = start < 0
                pts = []
                for a, b in zip(qs[:-1], qs[1:]):
                    n = max(2, int(np.ceil(np.max(np.abs(b - a)) / 0.02)))
                    pts += [a + (b - a) * k / n for k in range(1, n + 1)]
                cut = int(len(pts) * FREE_TAIL_FROM)
                for k, q in enumerate(pts):
                    s = min(np.max(np.abs(q - qa)), np.max(np.abs(q - qb)))
                    mg = S.OBJECT_MARGIN + (S.FREE_MOVE_MARGIN - S.OBJECT_MARGIN) * min(1.0, s / 0.2)
                    c = self.col_clear(q, mg, width)
                    c[excused] = np.inf
                    if k < cut:
                        head = np.minimum(head, c)
                    else:
                        tail = np.minimum(tail, c)
            else:
                for q in qs:
                    arm = np.minimum(arm, self.col_clear(q, S.OBJECT_MARGIN, width))
        carry = np.full(len(self.pj), np.inf)
        if route is not None:
            carry = self.carry_clear(g, route, cm)
        out = np.full((4, 36), np.inf)                    # rows: arm (cart), free_head, free_tail, carry
        for k, pid in enumerate(self.probe_ids):
            if pid not in cm:
                continue
            r, c = cm[pid]
            out[:, r * COLS + c] = (arm[k], head[k], tail[k], carry[k])
        return out.astype(np.float32)

    def carry_clear(self, g, route, cm):
        """plan_carry_xy's footprint rule (NO_OVERPASS: all hand bands): slack before the obstacle radius, per probe."""
        S, e = self.S, self.e
        o = e.objects[e.obj_id("m")]
        c = np.asarray(g["closing"][:2], float); c = c / np.linalg.norm(c)
        v = np.array([-c[1], c[0]])
        gap, tilt = g["open_width"], math.radians(g["tilt_deg"])
        side = np.array([math.cos(math.radians(g["azimuth_deg"])), math.sin(math.radians(g["azimuth_deg"]))])
        r_me = S.footprint_radius(o)
        if g["tilt_deg"] >= S.LEVEL_TILT_DEG:
            rects = [(side * (lo + hi) / 2, S._HAND_EY[k](gap), (hi - lo) / 2, side) for k, (lo, hi) in enumerate(S.HAND_BANDS)]
        else:
            rects = [(side * min(hi, 0.12) * math.tan(tilt), S._HAND_EY[k](gap), S._HAND_EX[k], v) for k, (lo, hi) in enumerate(S.HAND_BANDS)]
        pts = []
        P = [np.asarray(p, float) for p in route]
        for a, b in zip(P[:-1], P[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.003))
            pts += [a + (b - a) * k / n for k in range(n + 1)]
        pts = np.array(pts or P)
        out = []
        for pid in self.probe_ids:
            if pid not in cm:
                out.append(np.inf)
                continue
            q = np.asarray(e.cell_xy(cm[pid])[:2], float)
            d = q[None, :] - pts
            slack = np.linalg.norm(d, axis=1) - r_me - S.CARRY_GAP                     # the object itself
            for off, ey, ex, across in rects:
                dd = d - off
                aa, bb = np.abs(dd @ c), np.abs(dd @ across)
                slack = np.minimum(slack, np.hypot(np.maximum(aa - ey, 0.0), np.maximum(bb - ex, 0.0)) - S.CARRY_GAP)
            out.append(float(slack.min()))
        return np.array(out)

    # -- the teacher's candidate motions in the empty shelf
    def plan_moves(self, src, dst):
        S, e = self.S, self.e
        self.place({"m": tuple(src)})
        i = e.obj_id("m")
        e.objects[i]["pos"] = e.obj_pos(i)
        out, seen = [], set()
        for t, g in enumerate(S.grasp_poses(e.objects[i], e.table_z)):
            if len(out) >= K_CANDIDATES or t >= MAX_GRASP_TRIES:
                break
            key = (g["azimuth_deg"], g.get("grasp_level"))
            if key in seen:
                continue
            p = e.plan_pick_place("m", tuple(dst), g, q_start=e.q_ready, seed=0)
            if p["feasible"]:
                seen.add(key)
                keep = {k: g[k] for k in ("closing", "open_width", "tilt_deg", "azimuth_deg")}
                out.append((p["segments"], g["open_width"], p["carry_xy"], keep))
        return out

    def plan_picks(self, cell):
        S, e = self.S, self.e
        self.place({"m": tuple(cell)})
        i = e.obj_id("m")
        e._focus(i)
        e.objects[i]["pos"] = e.obj_pos(i)
        out, seen = [], set()
        for t, g in enumerate(S.grasp_poses(e.objects[i], e.table_z)):
            if len(out) >= K_CANDIDATES or t >= MAX_GRASP_TRIES:
                break
            key = (g["azimuth_deg"], g.get("grasp_level"))
            if key in seen:
                continue
            e._check_width = g["open_width"]
            try:
                for branch in range(e.n_branches):
                    try:
                        segs, _ = e._plan_approach(g, e.q_ready, seed=0, branch=branch)
                        seen.add(key)
                        out.append((segs, g["open_width"], None, None))
                        break
                    except (RuntimeError, ValueError) as err:
                        if isinstance(err, S.ReachError) or "out of reach" in str(err):
                            break
            finally:
                e._check_width = None
        return out

    # -- v3 extension (2026-11-17): extra motions with NON-frontal grasps (diagonal / side), which the v2 build rarely
    # stored (it kept the first K feasible grasps in the teacher's order = frontal ones). Up to EXT_K per entry.
    def _ext_grasps(self, obj):
        for t, g in enumerate(self.S.grasp_poses(obj, self.e.table_z)):
            if abs(((g["azimuth_deg"] - 180 + 180) % 360) - 180) > EXT_FRONTAL_DEG:
                yield g

    def plan_moves_ext(self, src, dst):
        e = self.e
        self.place({"m": tuple(src)})
        i = e.obj_id("m")
        e.objects[i]["pos"] = e.obj_pos(i)
        out, seen = [], set()
        for t, g in enumerate(self._ext_grasps(e.objects[i])):
            if len(out) >= EXT_K or t >= EXT_TRIES:
                break
            key = (g["azimuth_deg"], g.get("grasp_level"))
            if key in seen:
                continue
            p = e.plan_pick_place("m", tuple(dst), g, q_start=e.q_ready, seed=0)
            if p["feasible"]:
                seen.add(key)
                keep = {k: g[k] for k in ("closing", "open_width", "tilt_deg", "azimuth_deg")}
                out.append((p["segments"], g["open_width"], p["carry_xy"], keep))
        return out

    def plan_picks_ext(self, cell):
        S, e = self.S, self.e
        self.place({"m": tuple(cell)})
        i = e.obj_id("m")
        e._focus(i)
        e.objects[i]["pos"] = e.obj_pos(i)
        out, seen = [], set()
        for t, g in enumerate(self._ext_grasps(e.objects[i])):
            if len(out) >= EXT_K or t >= EXT_TRIES:
                break
            key = (g["azimuth_deg"], g.get("grasp_level"))
            if key in seen:
                continue
            e._check_width = g["open_width"]
            try:
                for branch in range(e.n_branches):
                    try:
                        segs, _ = e._plan_approach(g, e.q_ready, seed=0, branch=branch)
                        seen.add(key)
                        out.append((segs, g["open_width"], None, None))
                        break
                    except (RuntimeError, ValueError) as err:
                        if isinstance(err, S.ReachError) or "out of reach" in str(err):
                            break
            finally:
                e._check_width = None
        return out

    def build_src_ext(self, src, dsts=None):
        ent = {}
        src = tuple(src)
        cand = self.plan_picks_ext(src)
        ent[("pick", src)] = [self.motion_clearance(s, w, None, None, src, None) for s, w, _, _ in cand]
        if not cand:
            return ent
        for dst in (dsts or cells()):
            if dst == src:
                continue
            cm = self.plan_moves_ext(src, dst)
            ent[("move", src, dst)] = [self.motion_clearance(s, w, r, g, src, dst) for s, w, r, g in cm]
        return ent

    def build_src(self, src):
        """All entries with this source cell: ('pick', src) and ('move', src, dst) for every dst."""
        ent = {}
        src = tuple(src)
        cand = self.plan_picks(src)
        ent[("pick", src)] = [self.motion_clearance(s, w, None, None, src, None) for s, w, _, _ in cand]
        if not cand:                                       # unpickable in the empty shelf: every move from here too
            for dst in cells():
                if dst != src:
                    ent[("move", src, dst)] = []
            return ent
        for dst in cells():
            if dst == src:
                continue
            cm = self.plan_moves(src, dst)
            ent[("move", src, dst)] = [self.motion_clearance(s, w, r, g, src, dst) for s, w, r, g in cm]
        return ent


def build_job(job):
    """(mover class, ceiling class, src) -> LIB_DIR/<combo>/src_r_c.pkl (resumable)."""
    mc, cc, src = job
    out = LIB_DIR / combo_name(mc, cc) / f"src_{src[0]}_{src[1]}.pkl"
    if out.exists():
        return combo_name(mc, cc), src, "skip", 0.0
    t = time.time()
    b = Builder(mc, cc)
    ent = b.build_src(src)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(pickle.dumps(ent))
    os.replace(tmp, out)
    return combo_name(mc, cc), src, f"{sum(len(v) for v in ent.values())} motions", time.time() - t


def build_ext_job(job):
    """(mover class, ceiling class, src) -> LIB_DIR/<combo>/ext_src_r_c.pkl (resumable); needs the v2 file to exist."""
    mc, cc, src = job
    out = LIB_DIR / combo_name(mc, cc) / f"ext_src_{src[0]}_{src[1]}.pkl"
    if out.exists():
        return combo_name(mc, cc), src, "skip", 0.0
    t = time.time()
    b = Builder(mc, cc)
    ent = b.build_src_ext(src)
    tmp = out.with_suffix(f".tmp{os.getpid()}")
    tmp.write_bytes(pickle.dumps(ent))
    os.replace(tmp, out)
    return combo_name(mc, cc), src, f"{sum(len(v) for v in ent.values())} ext motions", time.time() - t


# ===================================================================================== query
class Library:
    def __init__(self):
        self.combos = {}

    def entries(self, mc, cc):
        key = combo_name(mc, cc)
        if key not in self.combos:
            d = {}
            for f in (LIB_DIR / key).glob("src_*.pkl"):
                d.update(pickle.loads(f.read_bytes()))
            if LIB_EXT:
                for f in (LIB_DIR / key).glob("ext_src_*.pkl"):
                    for k, v in pickle.loads(f.read_bytes()).items():
                        d[k] = list(d.get(k, [])) + list(v)
            self.combos[key] = d if len(list((LIB_DIR / key).glob("src_*.pkl"))) == 36 else None
        return self.combos[key]

    @staticmethod
    def feasible(cands, obstacles, mode=None):
        """obstacles: [(cell index, footprint radius)]."""
        mode = mode or FREE_MODE
        if not cands:
            return False
        if not obstacles:
            return True
        idx = np.array([k for k, _ in obstacles]); r = np.array([x for _, x in obstacles])
        need = r - R_PROBE - R_TOL
        rc = r - R_TOL
        for m in cands:                                       # m: (4, 36) arm, free_head, free_tail, carry
            if not np.all(m[0, idx] >= need) or not np.all(m[3, idx] >= rc):
                continue
            if mode in ("strict", "tail") and not np.all(m[2, idx] >= need):
                continue
            if mode == "strict" and not np.all(m[1, idx] >= need):
                continue
            return True
        return False


class LibOracle(Shelf66Oracle):
    """REAP checker backed by the library (teacher fallback when a combo is missing)."""
    _lib = None

    def __init__(self, side: dict):
        super().__init__(side)
        if LibOracle._lib is None:
            LibOracle._lib = Library()
        self.S_ = self.S
        self.radius = {o["id"]: self.S.footprint_radius(o) for o in side["objects"]}
        self.mc = {o["id"]: mover_class(o) for o in side["objects"]}
        self.cc = ceil_class(side["objects"])
        self.lib_hits = self.lib_miss = 0

    def _obst(self, layout, obj):
        return [(r * COLS + c, self.radius[k]) for k, (r, c) in layout.items() if k != obj]

    def pick(self, layout, obj, cell):
        others = frozenset((k, v) for k, v in layout.items() if k != obj)
        self.sources.setdefault((others, obj), set()).add(tuple(cell))
        ent = self._lib.entries(self.mc[obj], self.cc)
        if ent is None:
            self.lib_miss += 1
            return super().pick(layout, obj, cell)
        self.lib_hits += 1
        cands = ent.get(("pick", tuple(cell)))
        if PICK_MODE == "lenient":
            # pick = a cheap filter only: the put's stored motions contain the same approach and are checked against
            # the obstacles; the stored pick motions (2 per cell, fixed free-move path) refuse ~2/3 of the picks the
            # teacher allows (validate_lib / diag_lib, 2026-11-15), so only an object that cannot be picked even in
            # an empty shelf is refused here
            return bool(cands)
        return self._lib.feasible(cands, self._obst(layout, obj))

    def move(self, others, obj, src, dst):
        ent = self._lib.entries(self.mc[obj], self.cc)
        if ent is None:
            self.lib_miss += 1
            return super().move(others, obj, src, dst)
        self.lib_hits += 1
        return self._lib.feasible(ent.get(("move", tuple(src), tuple(dst))), self._obst(dict(others), obj))
