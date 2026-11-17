"""The dataset's own 2D motion layer and finalized X / Y checker, booted exactly as approach_b_final/teacher2d.py does.

boot(geom)                 one geometry implementation per process (fastgeom1 iff the record says so)
verify(rec, seq, cache)    X and Y of one ordered repair, identical to teacher2d.query (bxy_teacher X with connector + replay, Y from q_ref + replay)
Motion(rec, shared, ...)   the motion primitives every baseline plans with: the same DetPhysics (P16_FAST, per-query budget bd_2d.QB) and the
                           same exact per-root cache the X / Y teacher uses.

Budget accounting (DetPhysics' own "exact virtual accounting"): every primitive is charged the CPU seconds its computation took, also when
the result comes from the per-root cache shared by all runs of one root. A run therefore spends exactly the budget it would spend alone,
while identical queries of the 3 planners x 3 seeds of a root are computed once. Planner-side CPU (search bookkeeping) is charged as measured."""
from __future__ import annotations

import os
import sys
import time

import numpy as np

from . import paths as P

_M: dict = {}
SCREEN_NODES = 200           # RRT nodes of obstacle-free path planning (the dataset's executable planner uses P16_FAST's 1500)
CALL_CAP_S = 45.0            # per-call CPU timeout of every motion primitive (the papers' planners all fail fast; successes here take 3-33 s)
PICK_PHASES = ("approach", "pregrasp", "close_attach")
PLACE_PHASES = ("carry", "release", "retreat")


def boot(geom: str) -> None:
    if _M.get("geom") == geom:
        return
    if "geom" in _M:
        raise RuntimeError("one geometry implementation per process")
    sys.path[:0] = [str(P.CODE / "dataset-bd"), str(P.CODE), str(P.LCT)]
    os.chdir(P.LCT)                                                   # config.toml, as the original runner
    import warnings
    warnings.filterwarnings("ignore")
    if geom == "fastgeom1":
        import fastgeom
        fastgeom.install()
    import bd_2d  # noqa: F401
    sys.path.insert(0, str(P.WT))
    import bxy_teacher as T
    import signal
    signal.signal(signal.SIGVTALRM, _vt)
    _M.update(geom=geom, T=T, E=T.env(), wout=str(P.OUT / "witnesses"))


def verify(rec: dict, seq, cache: dict) -> dict:
    """-> {"x": X result, "y": Y result} exactly as teacher2d.query (empty seq: X = 1 by definition)."""
    T, E = _M["T"], _M["E"]
    rid = rec["ids"]["root_id"]
    seq = [(o, int(d)) for o, d in seq]
    if seq:
        r = T.x_query(E, rid, rec, seq, cache, out=_M["wout"])
        x = {"x": r.get("y"), "status": r.get("teacher_status"), "phase": r.get("failed_phase"), "replay": r.get("independent_replay"),
             "witnesses": (r.get("witness_relocations") or []) + ([r["witness_connector"]] if r.get("witness_connector") else []),
             "budget_spent": r.get("budget_spent"), "travel_parts": r.get("travel_parts")}
    else:
        x = {"x": 1, "status": "EMPTY", "witnesses": []}
    r = T.y_query(E, rid, rec, seq, cache, out=_M["wout"])
    y = {"y": r.get("y"), "status": r.get("teacher_status"), "replay": r.get("independent_replay"), "witness": r.get("target_witness"),
         "J_target": r.get("J_target")}
    return {"x": x, "y": y}


class Budget(Exception):
    """The planner's CPU budget for this root is spent."""


class _Cap(Exception):
    """CPU timer of one motion primitive (SIGVTALRM; the checker's own sector certificate uses SIGPROF)."""


def _vt(*_):
    raise _Cap()


CAPPED = "CAPPED"            # memo value of a primitive that hit CALL_CAP_S


class Shared:
    """Per-root state shared by every run on that root: the exact DetPhysics cache and a memo of our own deterministic primitives
    (result, CPU seconds of the computation)."""

    def __init__(self):
        self.cache: dict = {}
        self.memo: dict = {}


class Motion:
    def __init__(self, rec: dict, shared: Shared, budget_s: float | None = None):
        import bd_2d
        import bxy_teacher as T
        import p15_search as ps
        import p16_motion as pm
        from scene import scene_from_dict
        self.rec, self.rid = rec, rec["ids"]["root_id"]
        self.task = T.task_of(self.rid, rec)
        self.sh = shared
        self.qref = np.asarray(rec["q0"], float)
        self.gap = float(rec["scene"]["robot"]["gripper_gap_m"])
        self.cells0 = {o["id"]: int(o["cell"]) for o in rec["scene"]["objects"]}
        self.movable = sorted(rec["roles"])                           # every obstacle; the target is never relocated
        self.legal = sorted(int(c) for c in rec["scene"]["legal_cells"])
        self.goal_cell = int(rec["goal_cell"])
        self.s0 = scene_from_dict(rec["scene"])
        self.radius = {o.id: o.radius_m for o in self.s0.objects}
        self.support = {o.id: o.support for o in self.s0.objects}
        self.lat = self.s0.lattice
        self.QB, self._pm, self._ps, self._bd = dict(bd_2d.QB), pm, ps, bd_2d
        self.E = _M["E"]
        self.robot, self.p, self.cfg = self.E["robot"], self.E["p"], self.E["cfg"]
        self.budget_s = budget_s
        self.calls = {"edge": 0, "goal": 0, "connector": 0, "skill": 0, "screen": 0}
        self.n_object_checks = 0
        self._akeys: dict = {}
        self.n_capped = 0            # primitives that hit CALL_CAP_S (planner-side failures, never labels)
        self.trace: list = []        # [kind, used() at start, cost incl. planner-side work until the next call]
        self.vcpu = 0.0              # virtual CPU of motion primitives (recorded cost, cache hits included)
        self._prim_real = 0.0        # actual CPU spent inside primitives (excluded from the planner-side part)
        self._t0 = time.process_time()

    # ------------------------------------------------------------------ accounting
    def used(self) -> float:
        """CPU seconds charged to this run: virtual primitive CPU + measured planner-side CPU."""
        return self.vcpu + (time.process_time() - self._t0 - self._prim_real)

    def _gate(self, kind: str) -> None:
        u = self.used()
        if self.trace and self.trace[-1][2] is None:                     # close the previous call's entry
            self.trace[-1][2] = round(u - self.trace[-1][1], 2)
        if self.budget_s is not None and u >= self.budget_s:
            raise Budget(kind)
        if kind in self.calls:
            self.calls[kind] += 1
        self.trace.append([kind, round(u, 2), None])

    def _timed(self, fn):
        """Run fn under a CPU timer of min(CALL_CAP_S, remaining budget). -> (result, measured CPU, capped)."""
        import signal
        lim = CALL_CAP_S if self.budget_s is None else min(CALL_CAP_S, self.budget_s - self.used())
        if lim <= 0:
            raise Budget("spent")
        c0 = time.process_time()
        signal.setitimer(signal.ITIMER_VIRTUAL, lim)
        try:
            res, capped = fn(), False
        except _Cap:
            res, capped = None, True
        finally:
            signal.setitimer(signal.ITIMER_VIRTUAL, 0)
        return res, time.process_time() - c0, capped

    def _charge(self, cpu: float, real: float) -> bool:
        """Charge a primitive; -> True if it counts as capped (its cost exceeds CALL_CAP_S). Raises Budget past the run's budget."""
        capped = cpu > CALL_CAP_S
        self.vcpu += min(cpu, CALL_CAP_S)
        self._prim_real += real
        if capped:
            self.n_capped += 1
        if self.budget_s is not None and self.used() > self.budget_s:
            raise Budget("spent during a call")
        return capped

    def _memo(self, key, fn, fail=None):
        """Deterministic primitive with exact reuse across runs of this root; charged its recorded CPU. A computation that hits
        CALL_CAP_S is a failure (`fail`), remembered for the root."""
        c0 = time.process_time()
        if key in self.sh.memo:
            res, cpu = self.sh.memo[key]
        else:
            res, cpu, capped = self._timed(fn)
            if capped:
                if cpu < CALL_CAP_S - 1e-3:                                # the run's budget, not the call cap, stopped it
                    self._charge(cpu, time.process_time() - c0)
                    raise Budget("spent during a call")
                res = CAPPED
            self.sh.memo[key] = (res, cpu)
        capped = self._charge(cpu, time.process_time() - c0)
        return fail if (capped or res == CAPPED) else res

    def n_calls(self) -> int:
        return sum(self.calls.values())

    def phys(self):
        return self._pm.DetPhysics(self.task, None, budget=dict(self.QB), cache=self.sh.cache)

    def _with_phys(self, key, fn, fail=None):
        """fn(ph) on a fresh per-query DetPhysics (the dataset's exact cache), charged its virtual CPU (cache hits at their recorded cost).
        A query that hits CALL_CAP_S is a failure, remembered for the root under `key`."""
        c0 = time.process_time()
        if self.sh.memo.get(key, (None,))[0] == CAPPED:
            self._charge(self.sh.memo[key][1], time.process_time() - c0)
            return fail
        ph = self.phys()
        res, cpu, capped = self._timed(lambda: fn(ph))
        if capped:
            if cpu < CALL_CAP_S - 1e-3:
                self._charge(cpu, time.process_time() - c0)
                raise Budget("spent during a call")
            self.sh.memo[key] = (CAPPED, cpu)
            self._charge(cpu, time.process_time() - c0)
            return fail
        return fail if self._charge(ph.vcpu, time.process_time() - c0) else res

    def xy(self, cell: int) -> np.ndarray:
        return np.asarray(self.lat.center(int(cell)), float)

    def key(self, cells: dict, q=None):
        return self._ps.state_key(cells, self.qref if q is None else q, ())

    # ------------------------------------------------------------------ the dataset's planners
    def relocate(self, cells: dict, q, o: str, d: int):
        """One relocation from the achieved state, exactly as X plans it (cheapest verified mode). -> (q_end, J) or None."""
        import p16_common as pc
        self._gate("edge")
        q = np.asarray(q, float)
        e = self._with_phys(("edge", self.key(cells, q), o, int(d)), lambda ph: pc.best_edge(ph.edges(cells, q, o, int(d))))
        if e is None:
            return None
        _, _, q2, J, _ = e
        return np.asarray(q2, float), float(J)

    def connector_ok(self, cells: dict, q) -> bool:
        """X's terminal connector q -> q_ref in the edited scene (certified straight segment, else one bounded RRT)."""
        import geometry as ge
        import motion as mo
        q = np.asarray(q, float)
        if np.allclose(q, self.qref, atol=1e-12):
            return True
        self._gate("connector")

        def run():
            ph = self.phys()
            s = ph.scene(cells, q)
            ck = mo.checkers(self.robot, s, "t0", None, None, self.p["delta"], self.p["eps"])[("start", False)]
            if ck.status_at(q, self.gap).status != ge.CLEAR or ck.status_at(self.qref, self.gap).status != ge.CLEAR:
                return False
            conn, _ = ph._transit(s, "t0", q, {"q": [self.qref], "gap": [self.gap]}, ("bxy_connector", self._ps.state_key(cells, q, ())))
            return conn is not None
        return self._memo(("connector", self.key(cells, q)), run, fail=False)

    def target(self, cells: dict) -> bool:
        """Y semantics: the target task from q_ref in the scene `cells` (bd_2d.t_goal: same DetPhysics, same budget)."""
        self._gate("goal")
        if len(cells) > 1:
            st = self._with_phys(("goal", self.key(cells)), lambda ph: ph.goal(cells, self.qref)[0], fail="CAPPED")
            return st == "VERIFIED"
        return self._memo(("t_goal1", self.key(cells)), lambda: self._bd.t_goal(self.task, {}, cells, self.qref)[0] == "VERIFIED", fail=False)

    def done(self, cells: dict, q) -> bool:
        """The task is achieved from this state: target from q_ref (Y) and the connector from the achieved q (X)."""
        return self.target(cells) and self.connector_ok(cells, q)

    def skill(self, cells: dict, o: str, d: int, keep=(), q=None, nodes: int = SCREEN_NODES):
        """Screening plan with the dataset's skill planner at a small RRT budget (`nodes` per RRT, one plan failure): moving o to cell d in
        a scene holding only `keep` (+ o; keep='all' = every object), from q (default q_ref). keep=() gives the swept motion of an action
        when the other objects are ignored. -> arrays or None (a None is a screening failure, never a certificate)."""
        import copy
        import skill as sk
        from scene import scene_from_dict
        self._gate("skill")
        keep = tuple(sorted(cells)) if keep == "all" else tuple(sorted(set(keep) | {o}))
        q = self.qref if q is None else np.asarray(q, float)

        def run():
            sd = dict(self.rec["scene"])
            sd["objects"] = [dict(x, cell=int(cells[x["id"]]), xy_m=list(self.xy(cells[x["id"]]))) for x in self.rec["scene"]["objects"]
                             if x["id"] in keep]
            cfg = copy.deepcopy(self.cfg)
            cfg["planner2d"]["max_wall_seconds"] = 1e9
            cfg["planner2d"]["max_nodes"], cfg["skill2d"]["max_plan_failures"] = int(nodes), 1
            r = sk.plan_skill(self.robot, scene_from_dict(sd), o, int(d), cfg, f"b2d/{self.rid}/{o}/{d}/{keep}/{nodes}", q_start=q, q_goal=None)
            return r.traj.arrays() if (r.status == sk.VERIFIED and r.traj is not None) else None
        return self._memo(("skill", o, int(d), tuple((k, cells[k]) for k in keep), tuple(np.round(q, 12)), int(nodes)), run)

    def screen(self, cells: dict, o: str, d: int | None = None) -> dict:
        """The skill planner's own static screens over all 48 grasps in the scene `cells` (skill.screen_pick / screen_place: grasp pose,
        certified straight approach line, closure; place pose, release, certified retreat line) -- no RRT. The objects in the way are
        read from the failed candidates' first colliding pairs (definite collisions only).
        -> {"pick_ok", "pick_blockers", "place_ok", "place_blockers"}"""
        import motion as mo
        import skill as sk
        self._gate("screen")

        def blk(cands, ok_stage="OK"):
            out = set()
            for c in cands:
                if c.stage != ok_stage and c.certified and "|" in (c.reason or ""):
                    for part in c.reason.split("|"):
                        if part in cells and part != o:
                            out.add(part)
            return sorted(out)

        def run():
            ph = self.phys()
            s = ph.scene(cells, self.qref)
            pp = sk.params(self.cfg)
            ck0 = mo.checkers(self.robot, s, o, None, None, pp["delta"], pp["eps"])
            picks = sk.screen_pick(self.robot, ck0, self.xy(cells[o]), self.radius[o], pp)
            res = {"pick_ok": any(c.stage == "OK" for c in picks), "pick_blockers": blk(picks), "place_ok": None, "place_blockers": []}
            if d is not None:
                local = (self.robot.grasp_x, 0.0)
                ck = mo.checkers(self.robot, s, o, tuple(self.xy(d)), local, pp["delta"], pp["eps"])
                places = sk.screen_place(self.robot, ck, self.xy(d), local, 2 * self.radius[o], pp)
                res.update(place_ok=any(c.stage == "OK" for c in places), place_blockers=blk(places))
            return res
        return self._memo(("screen", tuple(sorted(cells.items())), o, d), run,
                          fail={"pick_ok": False, "pick_blockers": [], "place_ok": False, "place_blockers": []})

    # ------------------------------------------------------------------ geometry on a fixed trajectory (no planning)
    def check(self, arrays: dict, mover: str, mover_cell: int, obj: str, obj_cell: int) -> dict:
        """motion.object_check of a recorded action of `mover` (resting at mover_cell) against one object (or 'probe') at obj_cell.
        -> {"status": VALID/INVALID/UNRESOLVED, "phases": {phase: status}}"""
        import motion as mo
        self.n_object_checks += 1

        def run():
            r = mo.object_check(self.robot, arrays, (mover, tuple(self.xy(mover_cell)), self.radius[mover]),
                                (obj, tuple(self.xy(obj_cell)), self.radius.get(obj, self.radius[mover])),
                                self.support.get(obj, "table"), self.p["delta"], self.p["eps"], self.p["evals"])
            return {"status": r.status, "phases": dict(r.phase_status)}
        return self._memo(("ocheck", self.akey(arrays), mover, int(mover_cell), obj, int(obj_cell)), run,
                          fail={"status": "UNRESOLVED", "phases": {"timeout": "UNRESOLVED"}})

    def akey(self, arrays: dict) -> str:
        """Content key of a recorded action (memo keys must survive pickling of the per-root memo)."""
        import hashlib
        k = id(arrays)
        if k not in self._akeys:
            h = hashlib.sha1()
            for f in ("q", "gap", "phase", "held_local", "place_xy"):
                h.update(np.ascontiguousarray(np.asarray(arrays[f])).tobytes())
            self._akeys[k] = (arrays, h.hexdigest()[:20])                   # keep a reference so the id stays unique
        return self._akeys[k][1]

    def blockers(self, arrays: dict, mover: str, mover_cell: int, cells: dict, objs, phases=None) -> list[str]:
        """Objects of `objs` (at their cells) that the recorded action of `mover` does not certifiably clear (optionally only in `phases`)."""
        out = []
        for o in objs:
            if o == mover:
                continue
            r = self.check(arrays, mover, mover_cell, o, cells[o])
            st = [s for p, s in r["phases"].items() if phases is None or p in phases]
            if any(s != "VALID" for s in st):
                out.append(o)
        return out

    def swept_certified(self, arrays: dict, mover: str, mover_cell: int, cells_to_test, phases=None) -> set[int]:
        """Cells where a disc of the mover's radius would not be certifiably clear of the recorded action (exact, ~1 s per cell)."""
        out = set()
        for c in cells_to_test:
            r = self.check(arrays, mover, mover_cell, "probe", c)
            if any(s != "VALID" for p, s in r["phases"].items() if phases is None or p in phases):
                out.add(int(c))
        return out

    def swept(self, arrays: dict, mover: str, mover_cell: int, cells_to_test, phases=None, step=0.02) -> set[int]:
        """The action's swept footprint by sampling (the papers' own construction: M4M voxelises sampled robot poses, MO-SeGMan places
        clones along the path): probe discs of the mover's radius at every tested cell, robot bodies (+ the held object while carrying)
        evaluated at configurations sampled along every recorded segment (joint step <= `step` rad); a cell is swept when some sample comes
        closer than the certifier's clearance margin delta. The mover's resting cell and its placement cell are always included."""
        import geometry as ge
        import motion as mo
        cells_to_test = tuple(sorted(int(c) for c in cells_to_test))

        def run():
            r = self.radius[mover]
            world = ge.World(tuple(f"c{c}" for c in cells_to_test), np.array([self.xy(c) for c in cells_to_test], float).reshape(-1, 2),
                             np.full(len(cells_to_test), r), (self.lat.cols * self.lat.pitch_m, self.lat.rows * self.lat.pitch_m))
            held_local = tuple(arrays["held_local"]) if np.asarray(arrays["held_local"]).size else None
            plain = ge.Checker(self.robot, world, self.p["delta"], self.p["eps"], self_pairs=False)
            held = ge.Checker(self.robot, world, self.p["delta"], self.p["eps"], (mover, r, held_local), self_pairs=False) if held_local else plain
            q, gap = np.asarray(arrays["q"], float), np.asarray(arrays["gap"], float)
            dmin = np.full(len(cells_to_test), np.inf)
            for i, ph in enumerate(str(x) for x in arrays["phase"]):
                if phases is not None and ph not in phases:
                    continue
                ck = held if mo.CONTEXT[ph][0] == "held" else plain
                n = max(1, int(np.ceil(np.max(np.abs(q[i + 1] - q[i])) / step)))
                for s in np.linspace(0.0, 1.0, n + 1):
                    d = ck.distances(q[i] + s * (q[i + 1] - q[i]), gap[i] + s * (gap[i + 1] - gap[i]))[:ck.n_world]
                    dmin = np.minimum(dmin, d.reshape(len(ck.names), -1).min(0))
            out = {c for c, d in zip(cells_to_test, dmin) if d < self.p["delta"]}
            out.add(int(mover_cell))
            if np.asarray(arrays["place_xy"]).size and (phases is None or set(phases) & set(PLACE_PHASES)):
                pc = int(np.argmin([np.linalg.norm(self.xy(c) - np.asarray(arrays["place_xy"])) for c in self.legal]))
                out.add(self.legal[pc])
            return frozenset(out)
        return set(self._memo(("swept", self.akey(arrays), mover, int(mover_cell), cells_to_test, None if phases is None else tuple(phases), step),
                              run, fail=frozenset()))
