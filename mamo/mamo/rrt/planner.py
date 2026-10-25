"""RRT: Rapidly-exploring Random Trees (LaValle & Kuffner, "Randomized Kinodynamic Planning",
IJRR 2001), as the MAMO baseline of the M4M papers.

Paper, Fig. 5:

    BUILD_RRT(x_init)                       EXTEND(T, x)
      T.init(x_init)                          x_near <- NEAREST_NEIGHBOR(x, T)
      for k = 1 to K                          if NEW_STATE(x, x_near, x_new, u_new)
        x_rand <- RANDOM_STATE()                T.add_vertex(x_new); T.add_edge(x_near, x_new, u_new)
        EXTEND(T, x_rand)                       return Reached if x_new = x else Advanced
      return T                                return Trapped

The M4M papers run "standard implementations of sampling-based algorithms KPIECE and RRT
from OMPL that search the entire MAMO state space by randomly sampling robot motions"
(reference: em4m-main/test/run_ompl.cpp, ompl::geometric::RRT), so the remaining choices are
OMPL's and the reference's:

* single tree from the start: forward simulation cannot be run backwards, so the paper's
  RRT_BIDIRECTIONAL (Fig. 7) does not apply (KPIECE paper Sec. II-C)
* RANDOM_STATE: uniform joints; with probability 0.05 an IK solution of the goal pose
  (OMPL goal bias)
* NEW_STATE: steer from x_near toward x_rand by at most 0.2 x the maximum extent of the
  joint space (OMPL range self-configuration); the motion is STEERACTION (mamo.kinodynamic):
  it must not touch an immovable obstacle and is simulated if it touches movables -- the
  new vertex keeps the simulated object poses (the reference's OMPL wrapper copied the
  sampled state and dropped them)
* NEAREST_NEIGHBOR: Euclidean joint distance
* the goal: pre-grasp pose of the OoI, and the grasp succeeds in simulation
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from mamo.context import MAMOContext
from mamo.kinodynamic import KState, MotionAction, PhysicsProblem, finish
from mamo.solution import Solution


@dataclass
class Vertex:
    state: KState
    parent: int
    simulated: bool  # the edge from the parent touched movables
    tcp: np.ndarray | None = None  # TCP along the edge from the parent


def rrt(ctx: MAMOContext, budget: float, goal_bias: float = 0.05, range_frac: float = 0.2) -> Solution:
    t0 = time.perf_counter()
    prob = PhysicsProblem(ctx)
    if not prob.setup_goal():
        return finish(prob, ctx, Solution(False, reason="no grasp for the OoI"), t0)
    if not prob.state_valid(prob.start):
        return finish(prob, ctx, Solution(False, reason="start state invalid"), t0)
    step = range_frac * prob.max_extent
    verts = [Vertex(prob.start, -1, False)]
    qs = [prob.start.q]
    Q = np.array(qs)
    iters = 0
    while time.perf_counter() - t0 < budget:
        iters += 1
        x_rand = prob.sample_goal() if prob.rng.random() < goal_bias else prob.sample_uniform()
        if len(qs) != len(Q):
            Q = np.array(qs)
        i_near = int(np.argmin(np.sum((Q - x_rand) ** 2, axis=1)))       # NEAREST_NEIGHBOR
        q_near = verts[i_near].state.q
        d = prob.distance(q_near, x_rand)
        x = x_rand if d <= step else q_near + (x_rand - q_near) * (step / d)
        res = prob.steer(verts[i_near].state, x, partial=False)          # NEW_STATE
        if not res.ok:
            continue                                                     # Trapped
        verts.append(Vertex(res.end, i_near, res.simulated, np.vstack([res.tcp[::5], res.tcp[-1:]])))
        qs.append(res.end.q)
        if prob.is_satisfied(res.end):
            return finish(prob, ctx, _solution(verts, len(verts) - 1, prob), t0, iterations=iters,
                          tree_size=len(verts))
    return finish(prob, ctx, Solution(False, reason="budget exhausted"), t0, iterations=iters, tree_size=len(verts))


def _solution(verts: list[Vertex], i: int, prob: PhysicsProblem) -> Solution:
    edges = []
    while verts[i].parent >= 0:
        v, p = verts[i], verts[verts[i].parent]
        edges.append(MotionAction([p.state.q.copy(), v.state.q.copy()], v.simulated, list(v.tcp)))
        i = v.parent
    return Solution(True, edges[::-1], prob.retrieval)
