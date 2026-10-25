"""PLANRETRIEVAL: an arm trajectory that grasps the OoI and extracts it from the shelf.

M4M Alg. 1 (L4, L9) and E-M4M Alg. 1 (L7, L28) call PLANRETRIEVAL with the set
of objects to treat as obstacles. The reference implementation (robot.cpp
ComputeGrasps / PlanRetrieval) samples a grasp at shelf + U(2.5, 7.5) cm, plans
an approach to a pre-grasp pose, moves in, grasps, lifts by 7.5 cm and carries
the OoI out of the shelf. This module does the same with the xArm7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mamo.config import Config
from mamo.geometry import footprint, Circle
from mamo.robot.arm import Arm, ObstacleSet, tool_rotation
from mamo.robot.motion import cartesian_path, plan
from mamo.scene import SceneObject, Shelf


@dataclass
class Segment:
    name: str
    path: list[np.ndarray]
    gripper_closed: bool
    carrying: bool


@dataclass
class RetrievalPlan:
    segments: list[Segment] = field(default_factory=list)
    grasp: tuple | None = None   # (yaw, grasp height, pitch, flip) it uses

    @property
    def full_path(self) -> list[np.ndarray]:
        out: list[np.ndarray] = []
        for s in self.segments:
            out.extend(s.path if not out else s.path[1:])
        return out


def _grasp_width(o: SceneObject, approach: np.ndarray) -> float:
    """Extent of the OoI footprint along the finger-closing axis (perpendicular to approach)."""
    s = footprint(o)
    if isinstance(s, Circle):
        return 2 * s.r
    n = np.array([-approach[1], approach[0]])
    return float(np.ptp(s.corners() @ n))


def plan_retrieval(arm: Arm, cfg: Config, ooi: SceneObject, object_states: dict[int, np.ndarray],
                   obstacles: ObstacleSet, q_start: np.ndarray, rng: np.random.Generator,
                   shelf: Shelf, stats: dict[str, int] | None = None,
                   grasp: tuple | None = None) -> RetrievalPlan | None:
    """Plan grasp-and-extract for the OoI with ``obstacles`` as the collision set.

    ``obstacles`` must not include the OoI itself (it is the target); the OoI is
    avoided during the approach and carried afterwards. ``grasp`` (yaw, height, pitch,
    flip) fixes the grasp; by default up to ``grasp_tries`` grasps are sampled.
    """
    rc = cfg.retrieval
    arm.detach()
    arm.set_object_states(object_states)
    approach_obs = _with_ooi(obstacles, ooi.id)
    # Moving in to the grasp, the open fingers straddle the OoI; nothing else may touch it.
    reach_obs = ObstacleSet(approach_obs.categories, approach_obs.ignore_objects, frozenset({ooi.id}))
    top = ooi.top_z
    floor_top = shelf.floor_top
    # [cpp] lift 7.5 cm; [xarm] capped so tall objects clear the shelf ceiling.
    lift_h = max(0.01, min(rc.lift, shelf.ceiling - top - 0.01))
    if grasp is not None:
        candidates = [grasp]
    else:
        candidates = []
        for yaw in _grasp_yaws(ooi, rc, rng):
            gz = min(floor_top + rng.uniform(*rc.grasp_height_range), top - rc.grasp_top_margin)
            candidates.append((yaw, gz, math.radians(rng.uniform(*rc.approach_pitch_range_deg)),
                               bool(rng.random() < 0.5)))
    for yaw, gz, pitch, flip in candidates:
        approach = np.array([math.cos(yaw), math.sin(yaw)])
        # Fingertips reach ~3.5 cm past the OoI's near face (or its centre if it is thin).
        depth = _extent_along(ooi, approach)
        c = np.array([ooi.x, ooi.y])
        p_grasp = np.append(c - max(0.0, depth - 0.035) * approach, gz)
        R = tool_rotation(approach, pitch, flip=flip)
        p_pre = p_grasp - rc.pregrasp_backoff * R[:, 2]

        arm.set_gripper(closed=False)
        q_pre = arm.ik_multi(p_pre, R, [q_start], rng, restarts=rc.ik_restarts, obstacles=approach_obs)
        if q_pre is None:
            _note(stats, "pregrasp")
            continue
        reach = cartesian_path(arm, q_pre, p_grasp, R, reach_obs)
        if reach is None:
            _note(stats, "reach")
            continue
        approach_path = plan(arm, q_start, [q_pre], approach_obs, rng,
                             time_limit=rc.plan_time, max_iters=rc.max_rrt_iters)
        if approach_path is None:
            _note(stats, "approach")
            continue
        q_grasp = reach[-1]

        # Grasp: close the fingers and carry the OoI rigidly.
        arm.set_gripper(closed=True)
        arm.attach(ooi.id, q_grasp)
        # The wrist can reach the ceiling before the OoI does: fall back to lower lifts.
        lift = None
        for lh in (lift_h, 0.5 * lift_h, 0.01):
            lift = cartesian_path(arm, q_grasp, p_grasp + np.array([0, 0, lh]), R, obstacles)
            if lift is not None:
                break
        if lift is None:
            arm.detach()
            _note(stats, "lift")
            continue
        # Carry the OoI out of the shelf keeping the grasp orientation, so it
        # stays upright (cpp: retract with roll/pitch <= 5 deg path constraint).
        p_out = p_grasp + np.array([0, 0, lh])
        p_out[0] = min(p_out[0], shelf.x_min - ooi.radius - 0.03)
        extract = cartesian_path(arm, lift[-1], p_out, R, obstacles)
        if extract is None:
            goals = [q for q in (arm.ik_multi(p_out, R, [lift[-1]], rng, restarts=rc.ik_restarts,
                                              obstacles=obstacles),) if q is not None]
            extract = plan(arm, lift[-1], goals, obstacles, rng, time_limit=rc.plan_time,
                           max_iters=rc.max_rrt_iters) if goals else None
        arm.detach()
        if extract is None:
            _note(stats, "extract")
            continue
        _note(stats, "ok")
        return RetrievalPlan([
            Segment("approach", approach_path, False, False),
            Segment("reach", reach, False, False),
            Segment("lift", lift, True, True),
            Segment("extract", extract, True, True),
        ], (yaw, gz, pitch, flip))
    arm.set_gripper(closed=True)
    return None


def retrieval_free(arm: Arm, plan: RetrievalPlan, ooi_id: int, object_states: dict[int, np.ndarray],
                   obstacles: ObstacleSet) -> bool:
    """Re-check an existing retrieval trajectory against ``obstacles`` in another scene state.

    Each segment is checked the way plan_retrieval planned it: the OoI is avoided
    on the approach, only the fingers may touch it on the reach, and it is carried
    (with the planning margin) on lift and extract.
    """
    arm.detach()
    arm.set_object_states(object_states)
    approach_obs = _with_ooi(obstacles, ooi_id)
    reach_obs = ObstacleSet(approach_obs.categories, approach_obs.ignore_objects, frozenset({ooi_id}))
    ok = True
    for seg in plan.segments:
        arm.set_gripper(closed=seg.gripper_closed)
        if seg.carrying and arm.attached is None:
            arm.attach(ooi_id, seg.path[0])
        obs = obstacles if seg.carrying else reach_obs if seg.name == "reach" else approach_obs
        if not all(arm.edge_free(a, b, obs) for a, b in zip(seg.path[:-1], seg.path[1:])):
            ok = False
            break
    arm.detach()
    arm.set_gripper(closed=True)
    return ok


def _extent_along(o: SceneObject, direction: np.ndarray) -> float:
    """Half extent of the footprint along a horizontal direction."""
    s = footprint(o)
    if isinstance(s, Circle):
        return s.r
    return 0.5 * float(np.ptp(s.corners() @ direction))


def _note(stats: dict[str, int] | None, key: str) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1


def _grasp_yaws(ooi: SceneObject, rc, rng: np.random.Generator) -> list[float]:
    """Approach yaws (around +x) whose finger span fits the OoI.

    Face-aligned approaches for boxes are tried first, then random yaws; yaws
    whose span exceeds the gripper stroke are discarded and do not count
    toward ``grasp_tries``.
    """
    lim = math.radians(rc.approach_yaw_range_deg)
    cands: list[float] = []
    if ooi.shape != 2:
        for k in range(4):
            y = math.atan2(math.sin(ooi.yaw + k * math.pi / 2), math.cos(ooi.yaw + k * math.pi / 2))
            if abs(y) <= lim:
                cands.append(y)
    cands += list(rng.uniform(-lim, lim, size=10 * rc.grasp_tries))
    out = []
    for y in cands:
        a = np.array([math.cos(y), math.sin(y)])
        if _grasp_width(ooi, a) <= rc.gripper_max_width - 0.006:
            out.append(y)
        if len(out) >= rc.grasp_tries:
            break
    return out


def _with_ooi(obs: ObstacleSet, ooi_id: int) -> ObstacleSet:
    """Same obstacle set, but the OoI (category OOI_GEOM = 5) must be avoided."""
    from mamo.world import OOI_GEOM
    return ObstacleSet(obs.categories | {OOI_GEOM}, obs.ignore_objects - {ooi_id})
