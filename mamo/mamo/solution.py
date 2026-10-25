"""The result every planner returns: rearrangement actions in execution order and the final
retrieval (mamo.execute replays it), plus planning statistics."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from mamo.push import PushAction
from mamo.retrieval import RetrievalPlan

if TYPE_CHECKING:
    from mamo.context import MAMOContext


@dataclass
class Solution:
    solved: bool
    pushes: list[PushAction] = field(default_factory=list)
    retrieval: RetrievalPlan | None = None
    reason: str = ""
    stats: dict = field(default_factory=dict)


def _finish(ctx: MAMOContext, sol: Solution, t_wall: float) -> Solution:
    tm = ctx.timers
    sol.stats = {
        "planning_time": tm.total, "wall_time": time.perf_counter() - t_wall,
        "robot_planner_time": tm.robot_planner, "push_planner_time": tm.push_planner,
        "mapf_time": tm.mapf, "sim_time": tm.sim, "num_pushes": len(sol.pushes), **tm.counts,
    }
    return sol
