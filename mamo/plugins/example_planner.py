"""Example plugin: a minimal planner that shows the API your own method uses.

Every .py file in this folder is imported by the benchmark app and its workers; the
@register decorator adds a planner to the method list. Copy this file, rename the
planner and replace the body with your method.

What a planner gets -- ``ctx`` (mamo.context.MAMOContext) for one scene:
    ctx.scene                 objects (ctx.scene.movables, .immovables, .ooi, .shelf)
    ctx.scene.task            {"type": "retrieve", "targets": [ooi id]} or, in the generated
                              environments (mamo.envs), {"type": "relocate", "targets": [{"id",
                              "goal": [x, y, z, yaw], "tol": [xy m, yaw deg or -1]}, ...]}
    ctx.scene.env, .surfaces  environment name; every support surface (shelf boards, decks, tables)
    ctx.world / ctx.sim       MuJoCo model + simulator (world.snapshot() / world.restore())
    ctx.arm                   planning copy of the xArm7: fk, ik, collision checks
    ctx.home                  home configuration
    ctx.setup_ngr()           PLANRETRIEVAL with only immovables -> ctx.initial_retrieval, ctx.ngr
    ctx.retrieve_all(states, q)   PLANRETRIEVAL with every object as an obstacle (None if blocked)
    ctx.blockers(states)      movables inside the NGR
    ctx.build_agents(states)  per-object grids (agent.valid, agent.outside = cells outside the NGR)
    ctx.run_mapf(agents)      CBS over all movables
    ctx.plan_push(states, agent, cells, q)  push trajectory along a cell path (or PushFailure)
    mamo.simulate.simulate_push(ctx.sim, ctx.arm, action, ctx.cfg)  physics check -> SimResult
    mamo.im4m.pickplace.PickPlacePlanner / simulate_pick_place     prehensile actions
    mamo.kinodynamic.PhysicsProblem(ctx)  arm + object state space of RRT / KPIECE: steer()
                              (motion checked at 1 deg, simulated if it touches movables),
                              is_satisfied() (pre-grasp pose + simulated grasp), sample_goal()
    ctx.elapsed()             planning time used so far (planner + MAPF + simulation timers)

What a planner returns -- mamo.solution.Solution(solved, actions, retrieval, reason):
    actions    PushAction / PickPlaceAction / MotionAction (mamo.kinodynamic: a joint-space
               path, gripper open) objects in execution order
    retrieval  the RetrievalPlan that extracts the OoI after the last action (retrieve tasks;
               None for relocate tasks, which succeed when every target ends at its goal)
The tool replays every solution from the initial scene (mamo.execute) and scores all
methods the same way. ``tasks=`` in @register says which scene tasks a planner solves. Use
``mamo.solution._finish(ctx, solution, t_start)`` to fill in the standard timing statistics.
"""

from __future__ import annotations

import time

import numpy as np

from mamo.bench import register
from mamo.solution import Solution, _finish
from mamo.im4m.pickplace import PickPlaceFailure, PickPlacePlanner, simulate_pick_place
from mamo.push import PushFailure
from mamo.simulate import simulate_push


@register("example_greedy", description="Example plugin: push each NGR blocker to its nearest free cell "
          "outside the NGR, one at a time (no MAPF)", family="user", budget=120, tries=5)
def example_greedy(ctx, budget: float, tries: int = 5) -> Solution:
    t0 = time.perf_counter()
    if not ctx.setup_ngr():
        return _finish(ctx, Solution(False, reason="no retrieval without movables"), t0)
    world, actions = ctx.world, []
    while ctx.elapsed() < budget:
        states = world.object_states()
        plan = ctx.retrieve_all(states, world.arm_q())   # done if the OoI can be retrieved now
        if plan is not None:
            return _finish(ctx, Solution(True, actions, plan), t0)
        blockers = ctx.blockers(states)
        if not blockers:
            break
        agents = ctx.build_agents(states)
        progress = False
        for oid in blockers:
            a = agents[oid]
            goals = np.argwhere(a.outside)
            if len(goals) == 0:
                continue
            goal = tuple(goals[np.argmin(np.linalg.norm(goals - np.array(a.start), axis=1))])
            for _ in range(tries):
                action = ctx.plan_push(states, a, [a.start, goal], world.arm_q())
                if isinstance(action, PushFailure):
                    continue
                snap = world.snapshot()
                t = time.perf_counter()
                res = simulate_push(ctx.sim, ctx.arm, action, ctx.cfg)
                ctx.timers.sim += time.perf_counter() - t
                if res.valid:
                    actions.append(action)
                    progress = True
                    break
                world.restore(snap)
            if progress or ctx.elapsed() >= budget:
                break
        if not progress:
            break
    return _finish(ctx, Solution(False, actions, reason="stuck" if ctx.elapsed() < budget else "budget exhausted"), t0)


@register("example_relocate", description="Example plugin for relocate tasks: pick-and-place each target "
          "straight to its goal, in order (clears nothing)", family="user", budget=120, tasks=("relocate",), tries=3)
def example_relocate(ctx, budget: float, tries: int = 3) -> Solution:
    t0 = time.perf_counter()
    world, actions = ctx.world, []
    # Moves between decks or tables are longer than I-M4M's on-shelf placements: give the
    # transport planner more time, and try every grasp whose reach is free (not only the first).
    ctx.cfg.pick_place.plan_time, ctx.cfg.pick_place.max_rrt_iters = 5.0, 5000
    ctx.cfg.pick_place.reach_first = True
    pnp = PickPlacePlanner(ctx.arm, ctx.cfg, ctx.scene.shelf)
    for g in ctx.scene.task["targets"]:
        moved = False
        for _ in range(tries):
            if ctx.elapsed() >= budget:
                return _finish(ctx, Solution(False, actions, reason="budget exhausted"), t0)
            states = world.object_states()
            obj = ctx.scene_at(states).by_id(g["id"])
            # slide out of the object's compartment and into the goal's (their front edges, less
            # a hand's clearance), so the motion planner only has to connect the two fronts
            here = ctx.scene.surface_of(obj.x, obj.y, obj.bottom_z + 0.01)
            there = ctx.scene.surface_of(g["goal"][0], g["goal"][1], g["goal"][2])
            ahead = [s for s in (here, there) if s is not None and s.x_min > ctx.cfg.robot.base_pos[0]]
            via = (here.x_min - 0.05, there.x_min - 0.05) if len(ahead) == 2 and here is not there else None
            t = time.perf_counter()
            action = pnp.plan(obj, np.array(g["goal"][:2]), [], states, world.arm_q(), ctx.rng,
                              goal_z=g["goal"][2], via=via)
            ctx.timers.robot_planner += time.perf_counter() - t
            if isinstance(action, PickPlaceFailure):
                continue
            snap = world.snapshot()
            t = time.perf_counter()
            res = simulate_pick_place(ctx.sim, ctx.arm, action, ctx.cfg)  # physics check of grasp and placement
            ctx.timers.sim += time.perf_counter() - t
            if res.valid:
                actions.append(action)
                moved = True
                break
            world.restore(snap)
        if not moved:
            return _finish(ctx, Solution(False, actions, reason=f"could not move target {g['id']}"), t0)
    return _finish(ctx, Solution(True, actions), t0)
