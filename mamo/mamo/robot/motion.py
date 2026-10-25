"""Joint-space motion planning: RRT-Connect with shortcut smoothing.

Stands in for the reference implementation's search-based arm planners (SMPL
MHA* for retrieval, ARA* for push approaches). Plans are collision-checked
against an ObstacleSet on the Arm's planning model.
"""

from __future__ import annotations

import time

import numpy as np

from mamo.robot.arm import Arm, ObstacleSet


class _Tree:
    def __init__(self, roots: list[np.ndarray]) -> None:
        self.q = np.array(roots, dtype=float)
        self.parent = np.full(len(roots), -1, dtype=np.int64)

    def nearest(self, q: np.ndarray) -> int:
        return int(np.argmin(np.sum((self.q - q) ** 2, axis=1)))

    def add(self, q: np.ndarray, parent: int) -> int:
        self.q = np.vstack([self.q, q])
        self.parent = np.append(self.parent, parent)
        return len(self.parent) - 1

    def path_to_root(self, i: int) -> list[np.ndarray]:
        out = []
        while i >= 0:
            out.append(self.q[i])
            i = int(self.parent[i])
        return out


def _steer(a: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    d = b - a
    n = np.linalg.norm(d)
    return b.copy() if n <= eps else a + d * (eps / n)


def rrt_connect(arm: Arm, q_start: np.ndarray, goals: list[np.ndarray], obs: ObstacleSet,
                rng: np.random.Generator, time_limit: float = 2.0, max_iters: int = 3000,
                eps: float = 0.25) -> list[np.ndarray] | None:
    """Bidirectional RRT from q_start to any of the goal configurations."""
    if not arm.collision_free(q_start, obs):
        return None
    goals = [g for g in goals if arm.collision_free(g, obs)]
    if not goals:
        return None
    for g in goals:
        if arm.edge_free(q_start, g, obs):
            return [q_start.copy(), g.copy()]
    ta, tb = _Tree([q_start]), _Tree(goals)
    a_is_start = True
    t0 = time.perf_counter()
    for _ in range(max_iters):
        if time.perf_counter() - t0 > time_limit:
            return None
        q_rand = rng.uniform(arm.lo, arm.hi)
        i_near = ta.nearest(q_rand)
        q_new = _steer(ta.q[i_near], q_rand, eps)
        if not arm.edge_free(ta.q[i_near], q_new, obs):
            ta, tb, a_is_start = tb, ta, not a_is_start
            continue
        i_new = ta.add(q_new, i_near)
        # Greedily connect the other tree to q_new.
        j = tb.nearest(q_new)
        while True:
            q_next = _steer(tb.q[j], q_new, eps)
            if not arm.edge_free(tb.q[j], q_next, obs):
                break
            j = tb.add(q_next, j)
            if np.allclose(q_next, q_new):
                # ta root -> q_new -> tb root; reverse if ta is the goal tree.
                path = ta.path_to_root(i_new)[::-1] + tb.path_to_root(j)[1:]
                return path if a_is_start else path[::-1]
        ta, tb, a_is_start = tb, ta, not a_is_start
    return None


def shortcut(arm: Arm, path: list[np.ndarray], obs: ObstacleSet, rng: np.random.Generator,
             iters: int = 60) -> list[np.ndarray]:
    """Random shortcutting (the reference planner also shortcuts: robot/shortcut: true)."""
    path = [p.copy() for p in path]
    for _ in range(iters):
        if len(path) < 3:
            break
        i, j = sorted(rng.choice(len(path), size=2, replace=False))
        if j - i < 2:
            continue
        if arm.edge_free(path[i], path[j], obs):
            path = path[:i + 1] + path[j:]
    return path


def densify(path: list[np.ndarray], max_step: float) -> list[np.ndarray]:
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_step)))
        for k in range(1, n + 1):
            out.append(a + (b - a) * (k / n))
    return out


def cartesian_path(arm: Arm, q_start: np.ndarray, p_goal: np.ndarray, R_goal: np.ndarray,
                   obs: ObstacleSet | None, step: float = 0.01,
                   max_joint_jump: float = 0.25) -> list[np.ndarray] | None:
    """Straight-line TCP motion by IK at ``step`` spacing, seeded by the previous solution.

    Orientation is interpolated (slerp) from the start pose to ``R_goal``.
    Returns None if IK fails, the joints jump, or (when ``obs`` is given) a
    waypoint collides.
    """
    import mujoco

    p0, R0 = arm.fk(q_start)
    n = max(1, int(np.ceil(np.linalg.norm(p_goal - p0) / step)))
    q0_quat, q1_quat = np.zeros(4), np.zeros(4)
    mujoco.mju_mat2Quat(q0_quat, R0.flatten())
    mujoco.mju_mat2Quat(q1_quat, R_goal.flatten())
    if q0_quat @ q1_quat < 0:
        q1_quat = -q1_quat
    path = [q_start.copy()]
    q = q_start.copy()
    for k in range(1, n + 1):
        a = k / n
        quat = (1 - a) * q0_quat + a * q1_quat
        quat /= np.linalg.norm(quat)
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, quat)
        q_next = arm.ik(p0 + a * (p_goal - p0), R.reshape(3, 3), q)
        if q_next is None or np.max(np.abs(q_next - q)) > max_joint_jump:
            return None
        if obs is not None and not arm.edge_free(q, q_next, obs):
            return None
        path.append(q_next)
        q = q_next
    return path


def plan(arm: Arm, q_start: np.ndarray, goals: list[np.ndarray], obs: ObstacleSet,
         rng: np.random.Generator, time_limit: float = 2.0, max_iters: int = 3000) -> list[np.ndarray] | None:
    path = rrt_connect(arm, q_start, goals, obs, rng, time_limit, max_iters)
    if path is None:
        return None
    return shortcut(arm, path, obs, rng)
