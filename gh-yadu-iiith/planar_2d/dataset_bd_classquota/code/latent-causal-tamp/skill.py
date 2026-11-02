"""Six-phase side-grasp skill and per-cell reachability audit (Methodology v1.4 §2.1).

Declared grasp set: tool axis phi in {2 pi k / n : k < n} and both elbow branches.
A pick candidate needs IK, a clear open-finger grasp pose, a certified straight
approach and a certified contact closure. A place candidate needs IK, a clear held
pose, a certified release and a certified straight retreat. motion.plan supplies
the connectors home -> pregrasp, grasp -> place (carry) and retreat -> home.

Split from motion.py (size): motion.py owns paths and planning, this module owns
the composition of one skill out of certified pieces.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

import geometry as ge
import motion as mo
import planner as pn
from scene import Scene, make_scene, place_scene

VERIFIED, INFEASIBLE, UNRESOLVED = "VERIFIED_FEASIBLE", "CERTIFIED_INFEASIBLE", "UNRESOLVED"
SCOPE = "PLANAR_ARM_KINEMATIC_SKILL"


@dataclass
class Candidate:
    phi: float
    branch: int
    stage: str                      # first failed check, or "OK"
    reason: str = ""
    certified: bool = True          # failure is definite (not gray band / budget)
    q_end: np.ndarray | None = None
    line: list | None = None
    gap: float | None = None
    local: tuple | None = None


@dataclass
class SkillResult:
    status: str
    reason: str
    scope: str = SCOPE
    traj: mo.Trajectory | None = None
    picks: list = field(default_factory=list)
    places: list = field(default_factory=list)
    plans: list = field(default_factory=list)       # (label, planner.Plan)
    place_error_m: float | None = None
    used: dict = field(default_factory=dict)        # connector method per phase of the returned skill
    termination: str = ""                           # explicit end event (G4D); `reason` keeps its historical string


def params(cfg: dict) -> dict:
    return {"delta": cfg["suite"]["margin_m"], "eps": cfg["geometry"]["numeric_margin_band_m"],
            "evals": cfg["geometry"]["certify_max_evals"], **cfg["skill2d"],
            "contact_tol": cfg["gripper2d"]["contact_pos_tolerance_m"],
            "release_tol": cfg["gripper2d"]["release_xy_tolerance_m"]}


def _fail(c: Candidate, stage: str, v: ge.Verdict | None = None, why: str | None = None) -> Candidate:
    """Record the first failed check. A geometry verdict is definite only if it is a
    COLLISION (not gray band / budget). An IK-tracking reason is definite only for
    motion.DEFINITE_IK_FAILURES, and then only for the specified straight path and
    branch; numerical continuation failures stay uncertified (UNRESOLVED)."""
    c.stage = stage
    if v is not None:
        c.reason, c.certified = v.pair or "", v.status == ge.COLLISION
    elif why is not None:
        c.reason, c.certified = why, why in mo.DEFINITE_IK_FAILURES
    return c


def _ik(robot, anchor, local, phi):
    u = np.array([math.cos(phi), math.sin(phi)])
    tcp = np.asarray(anchor) - local[0] * u - local[1] * np.array([-u[1], u[0]])
    return {b: q for b, q in ge.ik(robot, tcp, phi)}


def directions(p: dict) -> list[tuple[float, int]]:
    return [(2 * math.pi * k / p["approach_dirs"], b) for k in range(p["approach_dirs"]) for b in (1, -1)]


def screen_pick(robot, ck: dict, xy, radius: float, p: dict) -> list[Candidate]:
    out, rest, touch, gopen = [], ck[("start", False)], ck[("start", True)], robot.open_gap
    for phi, b in directions(p):
        c = Candidate(phi, b, "OK")
        out.append(c)
        q = c.q_end = _ik(robot, xy, (robot.grasp_x, 0.0), phi).get(b)
        if q is None:
            _fail(c, "NO_IK", why=mo.NO_IK)              # exact for this declared (phi, branch)
            continue
        v = rest.status_at(q, gopen)
        if v.status != ge.CLEAR:
            _fail(c, "GRASP_POSE", v)
            continue
        line, why = mo.ik_line(robot, xy, (robot.grasp_x, 0.0), phi, b, p["approach_dist_m"], 0.0,
                               p["line_step_m"], p["max_ik_jump_rad"])
        v = None if line is None else mo.certify_chain(rest, line, gopen, p["evals"])
        if v is None or v.status != ge.CLEAR:
            _fail(c, "APPROACH", v, why)
            continue
        local = ge.tool_local(robot, q, xy)
        gap = 2 * (radius + abs(local[1]))
        if abs(local[1]) > p["contact_tol"] or not 0 <= local[0] <= robot.finger_len \
                or gap < robot.closed_gap:
            _fail(c, "CLOSURE", why="CLOSURE_GEOMETRY")   # exact under the declared contact model
            c.certified = True
            continue
        v = ge.certify(touch, q, q, gopen, gap, p["evals"])
        if v.status != ge.CLEAR:
            _fail(c, "CLOSURE", v)
            continue
        c.line, c.gap, c.local = line, gap, tuple(local)
    return out


def screen_place(robot, ck: dict, dest, local, gap: float, p: dict) -> list[Candidate]:
    out = []
    for phi, b in directions(p):
        c = Candidate(phi, b, "OK", gap=gap, local=local)
        out.append(c)
        q = c.q_end = _ik(robot, dest, local, phi).get(b)
        if q is None:
            _fail(c, "NO_IK", why=mo.NO_IK)              # exact for this declared (phi, branch)
            continue
        v = ck[("held", False)].status_at(q, gap)
        if v.status != ge.CLEAR:
            _fail(c, "PLACE_POSE", v)
            continue
        v = ge.certify(ck[("placed", True)], q, q, gap, robot.open_gap, p["evals"])
        if v.status != ge.CLEAR:
            _fail(c, "RELEASE", v)
            continue
        line, why = mo.ik_line(robot, dest, local, phi, b, 0.0, p["retreat_dist_m"], p["line_step_m"],
                               p["max_ik_jump_rad"])
        v = None if line is None else mo.certify_chain(ck[("placed", False)], line, robot.open_gap,
                                                       p["evals"])
        if v is None or v.status != ge.CLEAR:
            _fail(c, "RETREAT", v, why)
            continue
        c.line = line
    return out


def _none_left(stage: str, cands: list, picks, places=()) -> SkillResult:
    status = INFEASIBLE if all(c.certified for c in cands) else UNRESOLVED
    return SkillResult(status, f"NO_{stage}_CANDIDATE", f"{SCOPE}/DECLARED_GRASP_SET",
                       picks=picks, places=list(places), termination=f"{stage}_SCREEN_EXHAUSTED")


def pick_certificate(robot, scene, obj_id: str, cfg: dict) -> SkillResult | None:
    """The pick screen alone. It depends only on object poses (not on the robot's start
    state or the destination), so CERTIFIED_INFEASIBLE NO_PICK_CANDIDATE here holds for
    every relocation of obj_id begun in this scene, within the declared grasp family."""
    p = params(cfg)
    o = scene.get(obj_id)
    picks = screen_pick(robot, mo.checkers(robot, scene, obj_id, None, None, p["delta"], p["eps"]), o.xy_m,
                        o.radius_m, p)
    return None if any(c.stage == "OK" for c in picks) else _none_left("PICK", picks, picks)


def plan_skill(robot, scene, target_id: str, dest_cell: int, cfg: dict, key: str, q_start=None,
               q_goal="home") -> SkillResult:
    """start -> pregrasp -> approach -> close/attach -> carry -> release -> retreat -> goal.

    q_start is the robot's actual configuration (default: home, open gripper, nothing
    held). q_goal is where the final connector ends (default: home); None ends the
    record after the retreat, so the next skill starts exactly where this one stopped."""
    p, pc = params(cfg), cfg["planner2d"]
    target, dest = scene.get(target_id), scene.lattice.center(dest_cell)
    home = np.array(robot.q_home if q_start is None else q_start, float)
    goal = np.array(robot.q_home) if isinstance(q_goal, str) else (None if q_goal is None else np.asarray(q_goal, float))
    gopen = robot.open_gap
    ck0 = mo.checkers(robot, scene, target_id, None, None, p["delta"], p["eps"])
    v = ck0[("start", False)].status_at(home, gopen)
    if v.status != ge.CLEAR:
        st = INFEASIBLE if v.status == ge.COLLISION else UNRESOLVED
        return SkillResult(st, f"HOME_NOT_CLEAR:{v.pair}", f"{SCOPE}/START_CONFIG", termination="START_STATE_NOT_CLEAR")
    picks = screen_pick(robot, ck0, target.xy_m, target.radius_m, p)
    good = [c for c in picks if c.stage == "OK"]
    if not good:
        return _none_left("PICK", picks, picks)
    local, gap = (robot.grasp_x, 0.0), 2 * target.radius_m       # nominal attachment for screening
    ck = mo.checkers(robot, scene, target_id, dest, local, p["delta"], p["eps"])
    places = screen_place(robot, ck, dest, local, gap, p)
    goodp = [c for c in places if c.stage == "OK"]
    if not goodp:
        return _none_left("PLACE", places, picks, places)
    branches = {c.branch for c in goodp}                          # prefer picks a same-branch carry can serve
    good.sort(key=lambda c: (c.branch not in branches, np.linalg.norm(c.line[0] - home)))
    res, failures, direct_pairs = SkillResult(UNRESOLVED, "PLANNER_BUDGET", picks=picks, places=places), 0, 0

    def connect(label, checker, a, b, gap, rrt, guide=None, pose=None):
        nonlocal failures
        pl = pn.plan(checker, a, b, gap, pc, p["evals"], f"{key}/{label}", rrt, guide, pose)
        if rrt:
            res.plans.append((label, pl))
            failures += pl.status == ge.UNRESOLVED
        return pl if pl.status == ge.CLEAR else None

    for rrt in (False, True):  # a few cheap certified direct connectors first, then RRT-Connect
        for i, pick in enumerate(good):
            if failures >= p["max_plan_failures"] or (not rrt and direct_pairs >= p["max_direct_pairs"]):
                break
            p1 = connect(f"pregrasp{i}", ck0[("start", False)], home, pick.line[0], gopen, rrt)
            if p1 is None:
                continue
            order = sorted(goodp, key=lambda c: (c.branch != pick.branch, np.linalg.norm(c.q_end - pick.q_end)))
            for j, place in enumerate(order):
                if failures >= p["max_plan_failures"] or (not rrt and direct_pairs >= p["max_direct_pairs"]):
                    break
                direct_pairs += not rrt
                same = pick.branch == place.branch          # pose-space search keeps one elbow branch
                guide = None if not same else mo.carry_line(
                    robot, target.xy_m, dest, pick.local, pick.phi, place.phi, pick.branch,
                    p["line_step_m"], p["max_ik_jump_rad"])[0]
                pose = None if not same else ((*target.xy_m, pick.phi), (*dest, place.phi), pick.local,
                                              pick.branch, p["line_step_m"], p["max_ik_jump_rad"])
                p2 = connect(f"carry{i}.{j}", ck[("held", False)], pick.q_end, place.q_end, pick.gap,
                             rrt, guide, pose)
                if p2 is None:
                    continue
                p3 = None if goal is None else connect(f"home{i}.{j}", ck[("placed", False)], place.line[-1],
                                                       goal, gopen, rrt)
                if goal is None or p3:
                    return _finish(res, robot, scene, target_id, dest, (p1, pick, p2, place, p3), cfg)
    if failures >= p["max_plan_failures"]:
        res.termination = "SEARCH_LIMIT_PLAN_FAILURES"            # the declared failure allowance ended the search
        return res
    rejected = [c for c in picks + res.places if c.stage != "OK"]
    if failures == 0 and all(c.certified for c in rejected):  # every pair certified blocked
        res.status, res.reason, res.scope = INFEASIBLE, "ALL_CONNECTORS_CERTIFIED_BLOCKED", \
            f"{SCOPE}/DECLARED_GRASP_SET+LINK0_SWEEP"
        res.termination = "SCHEDULE_COMPLETED_ALL_CERTIFIED_BLOCKED"
    else:   # enumeration finished below the failure allowance; historical reason string kept as PLANNER_BUDGET
        res.termination = "SCHEDULE_COMPLETED_NO_WITNESS"
    return res


def _finish(res, robot, scene, target_id, dest, parts, cfg) -> SkillResult:
    """Assemble the waypoint record and re-certify it independently (replay)."""
    p1, pick, p2, place, p3 = parts
    gopen, p = robot.open_gap, params(cfg)
    t = mo.Trajectory(target_id, held_local=pick.local,
                      place_xy=tuple(ge.tool_point(robot, place.q_end, pick.local)))
    t.add(p1.path[0], gopen, None)
    for qs, gap, phase in ((p1.path[1:], gopen, "pregrasp"), (pick.line[1:], gopen, "approach"),
                           ([pick.q_end], pick.gap, "close_attach"), (p2.path[1:], pick.gap, "carry"),
                           ([place.q_end], gopen, "release"), (place.line[1:], gopen, "retreat"),
                           ([] if p3 is None else p3.path[1:], gopen, "return_home")):
        for q in qs:
            t.add(q, gap, phase)
    res.traj, res.place_error_m = t, float(np.linalg.norm(np.subtract(t.place_xy, dest)))
    res.used = {"pregrasp": p1.reason, "carry": p2.reason, "return_home": p3 and p3.reason,
                "costs": {k: {"nodes": c.nodes, "evals": c.evals, "seconds": round(c.seconds, 2),
                              "wall_clock_cut": c.wall_clock_cut} for k, c in (("pregrasp", p1), ("carry", p2),
                                                                                ("return_home", p3)) if c},
                "pick": {"phi_rad": pick.phi, "branch": pick.branch},
                "place": {"phi_rad": place.phi, "branch": place.branch}}
    verdicts = mo.certify_trajectory(robot, scene, target_id, t.arrays(), p["delta"], p["eps"], p["evals"])
    bad = [v for v in verdicts if v.status != ge.CLEAR]
    if bad or res.place_error_m > p["release_tol"]:
        res.status, res.reason = UNRESOLVED, f"REPLAY_REJECTED:{bad[0].pair if bad else 'PLACE_ERROR'}"
        res.termination = "WITNESS_REPLAY_REJECTED"
    else:
        res.status, res.reason = VERIFIED, "ALL_SEGMENTS_CERTIFIED"
        res.termination = "WITNESS_RECHECKED"
    return res


def count_reasons(cands: list) -> dict:
    """'STAGE:reason:definite|unresolved' -> count, over rejected candidates."""
    out = {}
    for c in cands:
        if c.stage != "OK":
            k = f"{c.stage}:{c.reason}:{'definite' if c.certified else 'unresolved'}"
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def summary(res: SkillResult, scene_hash: str, robot_hash: str) -> dict:
    """JSON-able record of one skill query (statuses, budgets, reasons, hash)."""
    count = lambda cs: {s: sum(c.stage == s for c in cs) for s in sorted({c.stage for c in cs})}  # noqa: E731
    out = {"status": res.status, "reason": res.reason, "verification_scope": res.scope,
           "pick_candidates": count(res.picks), "place_candidates": count(res.places),
           "rrt_pass_connector_log": [{"label": label, "status": pl.status, "method": pl.reason, "nodes": pl.nodes,
                           "evals": pl.evals, "seconds": round(pl.seconds, 2)} for label, pl in res.plans],
           "place_error_m": res.place_error_m, "connectors_used": res.used, "action_hash": None,
           "waypoints": 0, "wall_clock_dependent": any(pl.wall_clock_cut for _, pl in res.plans),
           "rejection_reasons": count_reasons(res.picks + res.places)}
    if res.traj is not None:
        arrays = res.traj.arrays()
        out["action_hash"] = mo.trajectory_hash(arrays, res.traj.target, scene_hash, robot_hash)
        out["waypoints"] = len(res.traj.q)
        out["phase_segments"] = {p: int(np.sum(arrays["phase"] == p)) for p in mo.PHASES}
    return out


def audit_rows(done: dict) -> list[dict]:
    """Per-cell candidate breakdown of the audit tasks (task id audit_cNN)."""
    rows = []
    for tid in sorted(k for k in done if k.startswith("audit_")):
        _, res, _ = done[tid]
        st = [c.stage for c in res.picks]
        rows.append({"cell": int(tid[-2:]), "status": res.status,
                     "n_ik": sum(s != "NO_IK" for s in st),
                     "n_pose": sum(s not in ("NO_IK", "GRASP_POSE") for s in st),
                     "n_grasp": sum(s == "OK" for s in st),
                     "ok_phis": [c.phi for c in res.picks if c.stage == "OK"]})
    return rows


# ------------------------------------------------------------------ G1 job helpers (moved from run.py)
def task_scene(cfg: dict, fx: dict, task_id: str, src: int) -> Scene:
    """Seeded fixtures reuse their generated scene; designated ones put t0 at src."""
    if "seed" in fx:
        s = make_scene(cfg, fx["root_id"], "ref", fx["seed"])
        if s.get("t0").cell != src:
            raise SystemExit(f"{task_id}: fixture target rests at {s.get('t0').cell}, not {src}")
        return s
    return place_scene(cfg, fx["root_id"], task_id, [src], fx["obstacle_cells"])


def run_task(job):
    """Pool worker: plan one six-phase skill (independent per-query seeds)."""
    cfg, fx, task_id, src, dst = job
    s = task_scene(cfg, fx, task_id, src)
    t0 = time.time()
    res = plan_skill(ge.make_robot(cfg), s, "t0", dst, cfg, task_id)
    return task_id, s, res, time.time() - t0


def g1_jobs(cfg: dict, fixtures: dict) -> list:
    jobs = []
    for case, fx in fixtures.items():
        if fx["stage"] != "G1":
            continue
        if fx.get("audit"):
            n = cfg["suite"]["rows"] * cfg["suite"]["cols"]
            jobs += [(cfg, fx, f"audit_c{c:02d}", c, n - 1 - c) for c in range(n)]
        jobs += [(cfg, fx, f"{case}_t{i}", a, b) for i, (a, b) in enumerate(fx["tasks"])]
    return jobs


def home_check(cfg: dict) -> dict:
    """Old provisional vs current home against every one of the 64 cells occupied."""
    robot = ge.make_robot(cfg)
    full = place_scene(cfg, "root_full", "all_cells", [], list(range(64)))
    ck = ge.Checker(robot, ge.make_world(full), cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"])
    out = {}
    for name, q in (("provisional_q0", (0.0, 0.0, 0.0)), ("home", robot.q_home)):
        d = ck.distances(q, robot.open_gap)
        k = int(np.argmin(d - ck.clear_thr))
        out[name] = {"q": list(q), "status": ck.status_at(q, robot.open_gap).status,
                     "closest_pair": ck.labels[k], "distance_m": float(d[k])}
    return out
