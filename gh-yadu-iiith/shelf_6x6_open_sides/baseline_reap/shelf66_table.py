"""Feasibility table for REAP training on the 6x6 shelf: the teacher's own plans, stored and re-checked.

REAP's training loop needs microsecond feasibility answers; the teacher (bd_3dt.query) takes 2-20 s. REAP's authors
precomputed a grasp library for their identical bottles; our objects differ per scene (boxes / cylinders, 1.2-2.0 cm,
6-18 cm, scene-dependent arm ceiling), so the table is built per scene from the teacher's own plans:

  feasible  : a stored trajectory of the same action (pick: ready -> hover -> grasp; move: the whole pick-carry-place)
              is still clear in the new layout -> feasible. The re-check runs the teacher's own object checks on the
              fixed waypoints (_object_hits per cart waypoint, the free-move clearance schedule on the densified
              joint path, the carry footprint rule on the route), with the objects where the state puts them. The
              shelf structure and arm ceiling do not change within a scene, so the static checks are not repeated.
              A valid stored trajectory is a valid plan, so this direction is sound.
  infeasible: the teacher found no plan with the other objects at O -> infeasible for every layout containing O
              (more obstacles never open a plan). Only approximately true for a randomised planner.
  otherwise : the teacher is asked (Shelf66Oracle), and the answer goes into the table.

The table persists per problem (YADU_TABLE_DIR/<problem>.pkl, merged on save), so all workers and later passes share
it. Test-time evaluation never uses it: there the plain teacher (Shelf66Oracle) decides.
"""
import math
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

from shelf66_oracle import Shelf66Oracle

TABLE_DIR = Path(os.environ.get("YADU_TABLE_DIR", Path.home() / "reap_table66"))
SAVE_EVERY = int(os.environ.get("YADU_TABLE_SAVE_EVERY", 5))


class TableOracle(Shelf66Oracle):
    def __init__(self, side: dict):
        super().__init__(side)
        self.plans = {}          # ("pick", o, c) / ("move", o, src, dst) -> [stored plan]
        self.infeas = {}         # same key -> [frozenset((obj, cell)) of the others when the teacher found nothing]
        self.stats = dict(hit_exact=0, hit_plan=0, hit_infeas=0, teacher=0, teacher_s=0.0, recheck_s=0.0)
        self._new = 0
        self.path = TABLE_DIR / f"{side['problem']}.pkl"
        self._load()

    # ---------------------------------------------------------------- persistence (merged, last writer adds)
    def _load(self):
        try:
            d = pickle.loads(self.path.read_bytes())
            self.plans, self.infeas = d["plans"], d["infeas"]
        except (OSError, EOFError, KeyError, pickle.UnpicklingError):
            pass

    def save(self):
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        try:
            old = pickle.loads(self.path.read_bytes())
        except (OSError, EOFError, KeyError, pickle.UnpicklingError):
            old = {"plans": {}, "infeas": {}}
        for k, v in old["plans"].items():
            mine = self.plans.setdefault(k, [])
            mine += [p for p in v if not any(p is q for q in mine)][: max(0, 4 - len(mine))]
        for k, v in old["infeas"].items():
            s = self.infeas.setdefault(k, [])
            s += [x for x in v if x not in s]
        tmp = self.path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_bytes(pickle.dumps({"plans": self.plans, "infeas": self.infeas}))
        os.replace(tmp, self.path)
        self._new = 0

    def _added(self):
        self._new += 1
        if self._new >= SAVE_EVERY:
            self.save()

    # ---------------------------------------------------------------- scene placement without simulation
    def _place(self, cells: dict):
        """Objects at the centres of their cells (kinematics only); objects not in `cells` parked off the grid."""
        e = self.e
        for j, o in enumerate(e.objects):
            a = e.obj_qadr[j]
            c = cells.get(o["name"])
            if c is None:
                e.data.qpos[a:a + 3] = [5.0 + j, 5.0, e.table_z + 0.1]
            else:
                x, y = e.cell_xy(tuple(c))[:2]
                e.data.qpos[a:a + 3] = [x, y, e.levels[0]["z"] + o["height"] / 2 + 0.0005]
                o["pos"] = np.array(e.data.qpos[a:a + 3])
            e.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        e.data.qpos[e.qadr] = e.q_ready
        self.S.mujoco.mj_forward(e.model, e.data)

    # ---------------------------------------------------------------- re-check a stored trajectory
    def _carry_free(self, i, g, route) -> bool:
        """plan_carry_xy's footprint rule (NO_OVERPASS: every hand band counts) along the stored route."""
        S, e = self.S, self.e
        o = e.objects[i]
        c = np.asarray(g["closing"][:2], float); c = c / np.linalg.norm(c)
        v = np.array([-c[1], c[0]])
        gap, tilt = g["open_width"], math.radians(g["tilt_deg"])
        side = np.array([math.cos(math.radians(g["azimuth_deg"])), math.sin(math.radians(g["azimuth_deg"]))])
        r_me = S.footprint_radius(o)
        if g["tilt_deg"] >= S.LEVEL_TILT_DEG:
            rects = [(side * (lo + hi) / 2, S._HAND_EY[k](gap), (hi - lo) / 2, side) for k, (lo, hi) in enumerate(S.HAND_BANDS)]
        else:
            rects = [(side * min(hi, 0.12) * math.tan(tilt), S._HAND_EY[k](gap), S._HAND_EX[k], v) for k, (lo, hi) in enumerate(S.HAND_BANDS)]
        parts = [(e.obj_pos(j)[:2], S.footprint_radius(oj)) for j, oj in enumerate(e.objects)
                 if j != i and np.linalg.norm(e.obj_pos(j)[:2]) < 4.0]

        def free(xy):
            for q, r in parts:
                if np.linalg.norm(q - xy) < r_me + r + S.CARRY_GAP:
                    return False
                for off, ey, ex, across in rects:
                    d = q - xy - off
                    a, b = abs(np.dot(d, c)), abs(np.dot(d, across))
                    if math.hypot(max(a - ey, 0.0), max(b - ex, 0.0)) < r + S.CARRY_GAP:
                        return False
            return True
        pts = [np.asarray(p, float) for p in route]
        for a, b in zip(pts[:-1], pts[1:]):
            n = max(2, int(np.linalg.norm(b - a) / 0.003))
            if not all(free(a + (b - a) * k / n) for k in range(n + 1)):
                return False
        return free(pts[0]) and free(pts[-1])

    def _free_move_clear(self, path) -> bool:
        """plan_free_move's edge check on the stored joint path: object clearance growing from OBJECT_MARGIN at the
        ends to FREE_MOVE_MARGIN 0.2 rad of travel away from them, every object counted (the hand is empty)."""
        S, e = self.S, self.e
        qa, qb = np.asarray(path[0], float), np.asarray(path[-1], float)
        excuse = set(e._object_hits(qa, every=True))          # leaving a column the arm starts in is allowed
        e._excuse = excuse
        try:
            for a, b in zip(path[:-1], path[1:]):
                a, b = np.asarray(a, float), np.asarray(b, float)
                n = max(2, int(np.ceil(np.max(np.abs(b - a)) / 0.02)))
                for k in range(1, n + 1):
                    q = a + (b - a) * k / n
                    s = min(np.max(np.abs(q - qa)), np.max(np.abs(q - qb)))
                    m = S.OBJECT_MARGIN + (S.FREE_MOVE_MARGIN - S.OBJECT_MARGIN) * min(1.0, s / 0.2)
                    if e._object_hits(q, m, every=True):
                        return False
        finally:
            e._excuse = set()
        return True

    def _recheck(self, plan, cells: dict, obj: str) -> bool:
        e = self.e
        t = time.time()
        try:
            self._place(cells)
            i = e.obj_id(obj)
            e._focus(i)
            e._check_width = plan["open_width"]
            for seg in plan["segments"]:
                if seg["kind"] == "grip":
                    continue
                if seg["phase"] == "to_hover" and seg["kind"] == "joint":
                    if not self._free_move_clear(seg["q"]):
                        return False
                elif any(e._object_hits(np.asarray(q, float)) for q in seg["q"]):
                    return False
            return plan.get("route") is None or self._carry_free(i, plan["grasp"], plan["route"])
        finally:
            e._check_width = None
            self.stats["recheck_s"] += time.time() - t

    # ---------------------------------------------------------------- table lookups wrapped around the teacher
    def _lookup(self, key, others: frozenset, cells: dict, obj: str):
        if any(I <= others for I in self.infeas.get(key, ())):
            self.stats["hit_infeas"] += 1
            return False
        for p in self.plans.get(key, ()):
            if self._recheck(p, cells, obj):
                self.stats["hit_plan"] += 1
                return True
        return None

    def _store(self, key, others, plan):
        if plan is None:
            self.infeas.setdefault(key, []).append(others)
        else:
            lst = self.plans.setdefault(key, [])
            if len(lst) < 4:
                lst.append(plan)
        self._added()

    def _teacher_plan(self, obj, dst):
        """bd_3dt.query, keeping the plan (segments, grasp, route) instead of a yes / no."""
        A, S, e = self.A, self.S, self.e
        i = e.obj_id(obj)
        e.objects[i]["pos"] = e.obj_pos(i)
        q0 = e.q_now()
        for g in S.grasp_poses(e.objects[i], e.table_z):
            p = e.plan_pick_place(obj, tuple(dst), g, q_start=q0, seed=0)
            if p["feasible"]:
                keep = {k: g[k] for k in ("closing", "open_width", "tilt_deg", "azimuth_deg")}
                return dict(segments=[{k: s[k] for k in ("phase", "kind", "q") if k in s} for s in p["segments"]],
                            open_width=g["open_width"], grasp=keep, route=p["carry_xy"])
        return None

    def _teacher_pick(self, obj):
        S, e = self.S, self.e
        i = e.obj_id(obj)
        e._focus(i)
        e.objects[i]["pos"] = e.obj_pos(i)
        for g in S.grasp_poses(e.objects[i], e.table_z):
            e._check_width = g["open_width"]
            try:
                for branch in range(e.n_branches):
                    try:
                        segs, _ = e._plan_approach(g, e.q_now(), seed=0, branch=branch)
                        return dict(segments=[{k: s[k] for k in ("phase", "kind", "q") if k in s} for s in segs],
                                    open_width=g["open_width"], grasp=None, route=None)
                    except (RuntimeError, ValueError) as err:
                        if isinstance(err, S.ReachError) or "out of reach" in str(err):
                            break
            finally:
                e._check_width = None
        return None

    def pick(self, layout: dict, obj: str, cell) -> bool:
        others = frozenset((k, v) for k, v in layout.items() if k != obj)
        self.sources.setdefault((others, obj), set()).add(tuple(cell))
        ekey = ("pick", others, obj, tuple(cell))
        if ekey in self.cache:
            self.stats["hit_exact"] += 1
            return self.cache[ekey]
        key = ("pick", obj, tuple(cell))
        cells = {**dict(others), obj: tuple(cell)}
        r = self._lookup(key, others, cells, obj)
        if r is None:
            t = time.time()
            self.queries += 1
            self._settle(cells)
            try:
                plan = self._teacher_pick(obj)
            except Exception:
                plan = None
                self.stats["timeout"] = self.stats.get("timeout", 0) + 1
            self.stats["teacher"] += 1
            self.stats["teacher_s"] += time.time() - t
            self._store(key, others, plan)
            r = plan is not None
        self.cache[ekey] = r
        return r

    def move(self, others: frozenset, obj: str, src, dst) -> bool:
        ekey = ("move", others, obj, tuple(src), tuple(dst))
        if ekey in self.cache:
            self.stats["hit_exact"] += 1
            return self.cache[ekey]
        key = ("move", obj, tuple(src), tuple(dst))
        cells = {**dict(others), obj: tuple(src)}
        r = self._lookup(key, others, cells, obj)
        if r is None:
            t = time.time()
            self.queries += 1
            self._settle(cells)
            try:
                plan = self._teacher_plan(obj, dst)
            except Exception:              # REAP's per-check timeout (or a planner error) = infeasible, like the
                plan = None                # search treats it; recorded, so the same slow query is not repeated
                self.stats["timeout"] = self.stats.get("timeout", 0) + 1
            self.stats["teacher"] += 1
            self.stats["teacher_s"] += time.time() - t
            self._store(key, others, plan)
            r = plan is not None
        self.cache[ekey] = r
        return r
