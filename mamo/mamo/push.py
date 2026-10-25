"""PLANPUSH: non-prehensile push trajectories from MAPF paths.

M4M Sec. IV-B / E-M4M 'Nonprehensile Push Planner':
  1. shortcut the MAPF path pi_m (respecting immovable obstacles) into straight
     segments x_1 .. x_n
  2. x_aabb = intersection of the ray from x_1 along (x_2 -> x_1) with the
     object's axis-aligned bounding box
  3. x^0_push ~ N(x_aabb, sigma I), sigma = 2.5 cm, z fixed 3 cm above the shelf
  4. plan a collision-free approach to x^0_push with all objects as obstacles
  5. sample x^i_push ~ N(x_i, sigma I) around each x_i and run IK along every
     segment (x^{i-1}_push, x^i_push)
  6. as in the reference implementation, the push is then reversed back to its
     start, so the next approach starts clear of the object just pushed

Two variants are provided:
  mode="paper": the papers' planner above (default).
  mode="cpp":   one straight push from x_aabb backed off by ``backoff`` (5 cm in
                robot.cpp) towards the path end, extended by 5 cm with Gaussian
                noise on the end point (robot.cpp PlanPush).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from mamo.config import Config
from mamo.geometry import aabb, footprint, point_dist
from mamo.mapf.agent import Agent
from mamo.robot.arm import Arm, ObstacleSet, tool_rotation
from mamo.robot.motion import cartesian_path, plan
from mamo.scene import Scene, SceneObject
from mamo.world import GROUND, IMMOVABLE_GEOM, OOI_GEOM, SHELF_GEOM, quat_to_rpy


@dataclass
class PushFailure:
    """Why PLANPUSH failed. ``kinematic`` failures (IK along the push or contact
    with static obstacles) do not depend on the other movables, so E-M4M marks
    the (object, start, goal) push invalid in every scene (mamo_node.cpp codes 3-5)."""
    reason: str
    kinematic: bool


@dataclass
class PushAction:
    obj_id: int
    cells: list[tuple[int, int]]            # MAPF path (local cells) this push realises
    waypoints: np.ndarray                   # shortcut object waypoints (x, y)
    tool_points: np.ndarray                 # tool targets (x, y, z)
    approach: list[np.ndarray]
    push: list[np.ndarray]
    retract: list[np.ndarray]
    info: dict = field(default_factory=dict)

    @property
    def goal_cell(self) -> tuple[int, int]:
        return self.cells[-1]


def shortcut_path(agent: Agent, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Greedy line-of-sight shortcutting on the agent's collision-free cell map."""
    pts = [cells[0]]
    for c in cells[1:]:
        if c != pts[-1]:
            pts.append(c)
    if len(pts) <= 2:
        return pts
    out = [pts[0]]
    i = 0
    while i < len(pts) - 1:
        j = len(pts) - 1
        while j > i + 1 and not _line_free(agent, pts[i], pts[j]):
            j -= 1
        out.append(pts[j])
        i = j
    return out


def _line_free(agent: Agent, a: tuple[int, int], b: tuple[int, int]) -> bool:
    n = int(max(abs(b[0] - a[0]), abs(b[1] - a[1]))) * 2 + 1
    for t in np.linspace(0.0, 1.0, n):
        c = (int(round(a[0] + t * (b[0] - a[0]))), int(round(a[1] + t * (b[1] - a[1]))))
        if not agent.valid[c]:
            return False
    return True


def _ray_exit(p0: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """Point where the ray p0 + t d (p0 inside the box) leaves the AABB."""
    t_exit = math.inf
    for k in range(2):
        if abs(d[k]) > 1e-12:
            t_exit = min(t_exit, ((hi[k] if d[k] > 0 else lo[k]) - p0[k]) / d[k])
    return p0 + t_exit * d


class PushPlanner:
    def __init__(self, arm: Arm, cfg: Config, floor_top: float) -> None:
        self.arm = arm
        self.cfg = cfg
        self.z = floor_top + cfg.push.height_above_shelf
        # Pushing may move other movables, but never touch immovables or the shelf.
        self.push_obs = ObstacleSet(frozenset({GROUND, SHELF_GEOM, IMMOVABLE_GEOM, OOI_GEOM}))

    def _near_obstacle(self, p: np.ndarray, states: dict[int, np.ndarray], target: int) -> bool:
        return _start_clearance(self.arm.world.scene, states, p) <= self.cfg.mapf.res

    def plan(self, obj: SceneObject, agent: Agent, cells: list[tuple[int, int]],
             object_states: dict[int, np.ndarray], q_start: np.ndarray,
             rng: np.random.Generator, stats: dict | None = None,
             plan_approach: bool = True) -> PushAction | PushFailure:
        """``plan_approach=False`` skips step 4 (the approach is then just [x^0_push]); DOGAR
        plans its approaches separately (GoTo)."""
        pc = self.cfg.push
        arm = self.arm
        arm.detach()
        arm.set_gripper(closed=True)
        arm.set_object_states(object_states)
        key = shortcut_path(agent, cells)
        wps = np.array([agent.xy(c) for c in key])
        wps[0] = (obj.x, obj.y)  # start from the true (continuous) object position
        if len(wps) < 2 or np.linalg.norm(wps[-1] - wps[0]) < 1e-6:
            return PushFailure("no motion", False)
        d1 = wps[1] - wps[0]
        d1 /= np.linalg.norm(d1)
        lo, hi = aabb(obj)
        x_aabb = _ray_exit(wps[0], -d1, lo, hi)
        x0 = x_aabb - pc.backoff * d1 + rng.normal(0.0, pc.sigma, 2)

        if pc.mode == "cpp":
            total = sum(np.linalg.norm(b - a) for a, b in zip(wps[:-1], wps[1:]))
            end = x0 + (total + pc.backoff + pc.overshoot) * d1 + rng.normal(0.0, pc.sigma, 2)
            tool_xy = [x0, end]
        else:
            # [paper] Fig. 5: x^i_push ~ N(x_i, sigma I), i = 0..n, x_0 := x_aabb (x_1 is the start)
            tool_xy = [x0] + [w + rng.normal(0.0, pc.sigma, 2) for w in wps]
        tool = np.array([[p[0], p[1], self.z] for p in tool_xy])
        # robot.cpp PlanPush: reject a push start within one grid cell (1 cm) of any obstacle.
        if pc.start_clearance_check and self._near_obstacle(tool[0, :2], object_states, obj.id):
            _note(stats, "push_start_near_obstacle")
            return PushFailure("push start inside obstacle", False)

        everything = ObstacleSet.everything()
        q0 = None
        alts = list(pc.tool_alternatives) or [(pc.tool_pitch_deg, False, 0.0)]
        for k in rng.permutation(len(alts)):
            alt = alts[k]
            q0 = arm.ik_multi(tool[0], push_rotation(d1, alt), [q_start], rng, restarts=4,
                              obstacles=everything)
            if q0 is not None:
                break
        if q0 is None:
            _note(stats, "push_start_ik")
            return PushFailure("push start unreachable", False)
        push = [q0]
        for a, b in zip(tool[:-1], tool[1:]):
            d = b[:2] - a[:2]
            if np.linalg.norm(d) < 1e-6:
                continue
            seg = cartesian_path(arm, push[-1], b, push_rotation(d, alt), self.push_obs)
            if seg is None:
                _note(stats, "push_ik")
                return PushFailure("push IK / static collision", True)
            push.extend(seg[1:])
        approach = plan(arm, q_start, [q0], everything, rng, time_limit=pc.plan_time,
                        max_iters=pc.max_rrt_iters) if plan_approach else [q0]
        if approach is None:
            _note(stats, "approach")
            return PushFailure("no approach path", False)
        retract = push[::-1] if pc.retract_by_reversing else []
        _note(stats, "planned")
        return PushAction(obj.id, list(cells), wps, tool, approach, push, retract, {"tool": alt})


def push_rotation(d: np.ndarray, alt: tuple[float, bool, float]) -> np.ndarray:
    """Tool rotation for pushing along horizontal ``d`` with a (pitch, flip, yaw offset) alternative."""
    pitch, flip, yaw = alt
    c, s = math.cos(math.radians(yaw)), math.sin(math.radians(yaw))
    e = np.array([c * d[0] - s * d[1], s * d[0] + c * d[1]])
    return tool_rotation(e, math.radians(pitch), flip)


def _start_clearance(scene: Scene, states: dict[int, np.ndarray], p: np.ndarray) -> float:
    """Distance from a point at push height to the nearest object footprint or shelf wall."""
    best = np.inf
    for o in scene.objects:
        if o.is_shelf:
            if not scene.spans_band(o):
                continue
            fp = footprint(o)
        else:
            s = states[o.id]
            fp = footprint(o, float(s[0]), float(s[1]), quat_to_rpy(s[3:7])[2])
        best = min(best, float(point_dist(fp, np.array([p[0]]), np.array([p[1]]))[0]))
    return best


def _note(stats: dict | None, key: str) -> None:
    if stats is not None:
        stats[key] = stats.get(key, 0) + 1
