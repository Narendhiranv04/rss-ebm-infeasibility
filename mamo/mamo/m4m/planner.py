"""M4M: Multi-Agent Pathfinding for Manipulation Among Movable Objects.

Saxena & Likhachev, ICRA 2023, Algorithm 1:

    O_M <- O_M^init ; Psi <- {}
    pi_hat_R <- PLANRETRIEVAL(O_I) ; compute V(pi_hat_R)
    replan <- true
    while time remains:
        if replan:
            pi_R <- PLANRETRIEVAL(O_I U O_M)
            if pi_R exists: Psi <- Psi U {pi_R}; done; break
            {pi_om} <- CBS(O_M, O_I, V(pi_hat_R)) ; replan <- false
        for o_m in O_M (that move in the CBS solution):
            psi <- PLANPUSH(o_m, pi_om, O_M, O_I)
            (valid, o_m') <- SIMULATEPUSH(psi)
            if valid: Psi <- Psi U {psi}; replan <- true; UPDATEPOSE; break

M4M greedily commits to the first valid push. The 'one_shot' flag gives
M4M-hat (paper Table I): CBS is called once and never re-invoked.
"""

from __future__ import annotations

import time

from mamo.context import MAMOContext
from mamo.push import PushAction, PushFailure
from mamo.simulate import simulate_push
from mamo.solution import Solution, _finish


def m4m(ctx: MAMOContext, budget: float | None = None, one_shot: bool = False) -> Solution:
    budget = ctx.cfg.search.m4m_budget if budget is None else budget
    t_wall = time.perf_counter()
    if not ctx.setup_ngr(include_ooi=False):  # M4M: "the volume occupied by the robot arm"
        return _finish(ctx, Solution(False, reason="no retrieval trajectory without movables"), t_wall)

    world = ctx.world
    pushes: list[PushAction] = []
    replan = True
    solution = None
    agents = {}
    while ctx.elapsed() < budget:
        states = world.object_states()
        if replan:
            plan = ctx.retrieve_all(states, world.arm_q())
            if plan is not None:
                return _finish(ctx, Solution(True, pushes, plan), t_wall)
            if solution is None or not one_shot:
                agents = ctx.build_agents(states)
                solution = ctx.run_mapf(agents)
                if solution is None:
                    # The reference retries until the budget expires; CBS is
                    # deterministic here, so a failed call is final.
                    return _finish(ctx, Solution(False, pushes, reason="CBS failed"), t_wall)
            replan = False
        # M4M-hat never re-plans MAPF: it keeps pushing objects towards the goals of its one CBS
        # solution until Line 9 succeeds or time runs out ([cpp] coop_plan_no_iterate.cpp).
        moving = [aid for aid, p in solution.paths.items() if p[0] != p[-1]]
        if not moving:
            return _finish(ctx, Solution(False, pushes, reason="no MAPF-suggested push left"), t_wall)
        snap = world.snapshot()
        # Line 15: for o_m in O_M -- in random order ([cpp] planner.cpp chooseObjDTS draws every
        # object from Beta(1, 1): its Thompson-sampling updates are commented out).
        for aid in ctx.rng.permutation(sorted(moving)).tolist():
            if ctx.elapsed() >= budget:
                break
            action = ctx.plan_push(states, agents[aid], solution.paths[aid], world.arm_q())
            if isinstance(action, PushFailure):
                continue
            t = time.perf_counter()
            res = simulate_push(ctx.sim, ctx.arm, action, ctx.cfg)
            ctx.timers.sim += time.perf_counter() - t
            ctx.timers.bump("push_sims")
            if res.valid:
                pushes.append(action)
                ctx.timers.bump("valid_pushes")
                replan = True
                break
            world.restore(snap)
            ctx.timers.bump("invalid_pushes")
    return _finish(ctx, Solution(False, pushes, reason="budget exhausted"), t_wall)
