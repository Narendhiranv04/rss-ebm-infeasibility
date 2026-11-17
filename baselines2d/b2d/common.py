"""Shared by the three baselines: output contract, dataset rules, and how the dataset's motion planner is used.

Output (approach_b_final/evaluate.py): NOOP (empty repair), REPAIR (ordered [(object, cell)]) or ABSTAIN (no repair found).
Dataset rules (bxy_teacher.eligible, bd_core.check_sequence): only obstacles move, each at most once, to a legal free cell; at most
MAX_RELOC relocations (the dataset's K <= 4 contract), then the target task.

Motion planning (common adaptation, all three planners): every executable action is planned by the dataset's own planner (the one the
X / Y checker uses). That planner has no fail-fast bound (a failing query exhausts its RRT budget, ~300 CPU-s here) while the papers'
planners all time out quickly, so every call runs under a per-call CPU timeout (teacher.CALL_CAP_S); a timed-out call is a planner
failure, never a label. Geometric tests (which objects a path sweeps) use the action planned with the other objects ignored and the
dataset's per-object certification (motion.object_check), as the papers use swept volumes / clones / linear paths."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

MAX_RELOC = 4


@dataclass
class Result:
    status: str                                   # NOOP | REPAIR | ABSTAIN
    seq: list = field(default_factory=list)       # [(object id, cell)]
    reason: str = ""
    stats: dict = field(default_factory=dict)


def free_cells(m, cells: dict) -> list[int]:
    occ = set(cells.values())
    return [c for c in m.legal if c not in occ]


def dist(m, a: int, b: int) -> float:
    return float(np.linalg.norm(m.xy(a) - m.xy(b)))


def relaxed_target(m):
    """The target task planned with every obstacle ignored, from q_ref: M4M's gamma_OoI / MO-SeGMan's task trajectory /
    MS-MCTS's picking + placing paths of the object of interest. -> arrays or None."""
    return m.skill(m.cells0, "t0", m.goal_cell)


def target_blockers(m, gamma, cells: dict) -> list[str]:
    return m.blockers(gamma, "t0", m.cells0["t0"], cells, [o for o in cells if o != "t0"])


def task_ok(m, gamma, cells: dict, q, gate: bool = False) -> bool:
    """The task from this state: target from q_ref + connector from the achieved state (the dataset's planner). gate=True runs it only
    when the obstacle-free target path clears every object (MS-MCTS: the goal child exists only when O_b is empty)."""
    if gate and target_blockers(m, gamma, cells):
        return False
    return m.done(cells, q)


def relocation_blockers(m, cells: dict, o: str, d: int):
    """Objects in the way of o's obstacle-free path to d (None if there is no such path)."""
    tr = m.skill(cells, o, d)
    return None if tr is None else m.blockers(tr, o, cells[o], cells, [x for x in cells if x != o])


def start(m, gamma) -> Result | None:
    """Common protocol step: every planner first tries the task in the scene as it is (the dataset's planner; the arm is at q_ref)."""
    if gamma is None:
        return Result("ABSTAIN", [], "no target path even with every obstacle ignored")
    if m.done(m.cells0, m.qref):
        return Result("NOOP", [], "target feasible in the initial scene")
    return None
