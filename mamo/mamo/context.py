"""Shared machinery for M4M and E-M4M: world, arm, NGR, MAPF, push planning, timing."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from mamo.config import Config
from mamo.mapf.agent import Agent, build_agent
from mamo.mapf.cbs import CBSResult, cbs
from mamo.ngr import NGR, compute_ngr
from mamo.push import PushAction, PushFailure, PushPlanner
from mamo.retrieval import RetrievalPlan, plan_retrieval
from mamo.robot.arm import Arm, ObstacleSet
from mamo.scene import Scene
from mamo.simulate import Simulator
from mamo.world import World


@dataclass
class Timers:
    """Planning-time breakdown (the reference reports the same categories)."""
    robot_planner: float = 0.0   # PLANRETRIEVAL calls
    push_planner: float = 0.0    # PLANPUSH calls
    mapf: float = 0.0            # CBS calls
    sim: float = 0.0             # forward simulation (wall clock)
    counts: dict = field(default_factory=dict)

    @property
    def total(self) -> float:
        return self.robot_planner + self.push_planner + self.mapf + self.sim

    def bump(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n


class MAMOContext:
    def __init__(self, scene: Scene, cfg: Config | None = None, seed: int = 0) -> None:
        self.scene = scene
        self.world = World(scene, cfg or Config())
        self.cfg = self.world.cfg  # with the scene's robot base and home pose
        self.arm = Arm(self.world)
        self.sim = Simulator(self.world, self.cfg)
        self.pusher = PushPlanner(self.arm, self.cfg, scene.shelf.floor_top)
        self.rng = np.random.default_rng(seed)
        self.timers = Timers()
        self.home = np.array(self.cfg.robot.home_q)
        self.ngr: NGR | None = None
        self.initial_retrieval: RetrievalPlan | None = None
        self.grasp: tuple | None = None   # (yaw, height, pitch, flip) of the OoI grasp, once chosen
        self.retrieval_stats: dict = {}
        self.push_stats: dict = {}

    # --------------------------------------------------------------- NGR
    def setup_ngr(self, include_ooi: bool = True) -> bool:
        """PLANRETRIEVAL(O_I) and V(gamma_OoI) (M4M L4-5, E-M4M L28-29). The grasp found here
        is kept for every later PLANRETRIEVAL ([cpp] planner.cpp Init: ComputeGrasps once)."""
        t = time.perf_counter()
        plan = self.retrieve(self.world.object_states(), ObstacleSet.static({self.scene.ooi_id}), self.home)
        if plan is not None:
            self.ngr = compute_ngr(self.arm, plan, self.scene.shelf, self.scene.ooi_id,
                                   self.world.object_states(), self.cfg.ngr.res, self.cfg.ngr.traj_step,
                                   shrink=self.cfg.ngr.shrink, include_ooi=include_ooi)
            self.grasp = plan.grasp
        self.timers.robot_planner += time.perf_counter() - t
        self.initial_retrieval = plan
        return plan is not None

    def retrieve(self, states: dict[int, np.ndarray], obstacles: ObstacleSet,
                 q_start: np.ndarray) -> RetrievalPlan | None:
        self.timers.bump("retrieval_calls")
        return plan_retrieval(self.arm, self.cfg, self.scene.ooi, states, obstacles, q_start, self.rng,
                              self.scene.shelf, self.retrieval_stats, grasp=self.grasp)

    def retrieve_all(self, states: dict[int, np.ndarray], q_start: np.ndarray) -> RetrievalPlan | None:
        """PLANRETRIEVAL(O_I U O_M): every object is an obstacle."""
        t = time.perf_counter()
        plan = self.retrieve(states, ObstacleSet.everything({self.scene.ooi_id}), q_start)
        self.timers.robot_planner += time.perf_counter() - t
        return plan

    # -------------------------------------------------------------- MAPF
    def scene_at(self, states: dict[int, np.ndarray]) -> Scene:
        return self.world.scene_with_states(states)

    def build_agents(self, states: dict[int, np.ndarray]) -> dict[int, Agent]:
        """Per-agent MAPF maps (counted as MAPF time: it is part of setting up the MAPF problem)."""
        t = time.perf_counter()
        sc = self.scene_at(states)
        agents = {o.id: build_agent(o, sc, self.ngr, self.cfg.mapf) for o in sc.movables}
        self.timers.mapf += time.perf_counter() - t
        return agents

    def run_mapf(self, agents: dict[int, Agent]) -> CBSResult | None:
        t = time.perf_counter()
        self.timers.bump("mapf_calls")
        res = cbs(agents, self.cfg.mapf.cbs_time_limit, self.cfg.mapf.max_time_steps, {})
        self.timers.mapf += time.perf_counter() - t
        return res

    def blockers(self, states: dict[int, np.ndarray]) -> list[int]:
        sc = self.scene_at(states)
        return [o.id for o in sc.movables if self.ngr.intersects(o)]

    # -------------------------------------------------------------- pushes
    def plan_push(self, states: dict[int, np.ndarray], agent: Agent, path: list,
                  q_start: np.ndarray) -> PushAction | PushFailure:
        t = time.perf_counter()
        self.timers.bump("push_plans")
        sc = self.scene_at(states)
        a = self.pusher.plan(sc.by_id(agent.id), agent, path, states, q_start, self.rng, self.push_stats)
        self.timers.push_planner += time.perf_counter() - t
        return a

    def elapsed(self) -> float:
        return self.timers.total
