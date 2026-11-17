"""MS-MCTS (Ren & Qureshi, arXiv 2305.17175) on the 2D pick-and-place dataset.

Only the target T has a goal, so the object stage topology has one stage and MS-MCTS is one SS-MCTS with object of interest o_k = T
(Section IV-C), on the paper's discrete-buffer variant (Table II, 'MS-MCTS w/ Discrete buffer region'): buffers are free legal cells.

* Motion planner MP: the dataset's planner (the paper: 'this task planning method suits all underlying motion planning methods').
  A picking (placing) path is available when some grasp passes the planner's static pick (place) screen: grasp pose + certified straight
  approach (retreat) line -- the analogue of the paper's linear paths; collision_objs = the objects those screens collide with.
  T's paths for O_b and the swept regions used by Algorithm 1 = T's / o_i's action planned with the other objects ignored, certified per
  object (motion.object_check) / footprint sampled along the path.
* Expansion = Algorithm 2: blocking objects O_b of T's picking and placing paths; an accessible blocker gets buffer regions from
  Algorithm 1; an inaccessible one makes the objects blocking its picking path the focus, with it as their dependency object; the
  halting condition moves the focus to the inaccessible objects of the round; after STUCK_DEPTH moves without fewer blockers of T,
  O_b becomes every feasible object. O_b empty (or T's picking + placing paths available) -> one child placing T at its goal, verified
  by MP; if MP refutes it, the node keeps clearing O_b.
* Algorithm 1: candidates = free legal cells, nearest to o_i first; rejected if they lie on T's picking / placing paths or on the picking
  path of a dependency object; accepted if MP verifies the relocation; at most M_REGIONS per call (the paper's expansion threshold).
* Selection: tuned UCB (UCB1-Tuned) on rewards normalised to [0, 1]; simulation: rollout choosing a random region of Algorithm 1 until T
  can be placed; reward = -(accumulated Euclidean object relocation distance); back-propagation to the root.
* Halting: when a tree node achieves the objective (T placed at its goal); the plan is recovered by backward traversal. Post-optimisation
  (merging consecutive moves of one object) is vacuous under the move-once rule.

Dataset rules (deviation from the paper's non-monotone setting): each object moves at most once, at most 4 relocations, cell destinations.
Constants the paper does not give: M_REGIONS = 2, MAX_TRIES = 4 candidate cells per Algorithm 1 call, STUCK_DEPTH = 2, failure reward
= -FAIL_R with FAIL_R = 4 m (above any successful plan's relocation distance on the 0.64 m board)."""
from __future__ import annotations

import collections
import math

import numpy as np

from .common import MAX_RELOC, Result, dist, free_cells, relaxed_target, start, target_blockers, task_ok
from .teacher import PICK_PHASES, PLACE_PHASES

M_REGIONS, MAX_TRIES, STUCK_DEPTH, FAIL_R = 2, 4, 2, 4.0


class Node:
    __slots__ = ("cells", "q", "seq", "parent", "children", "n", "w", "w2", "expanded", "dead", "goal")

    def __init__(self, cells, q, seq, parent=None, goal=False):
        self.cells, self.q, self.seq, self.parent = cells, q, seq, parent
        self.children, self.n, self.w, self.w2 = [], 0, 0.0, 0.0
        self.expanded, self.dead, self.goal = False, False, goal

    def moved(self):
        return {o for o, _ in self.seq}


class MSMCTS:
    def __init__(self, m, rng):
        self.m, self.rng = m, rng
        self.st = collections.Counter()
        self.pick_traj = {}

    # ------------------------------------------------------------------ paths and blockers (relaxed MP + per-object certification)
    def t_blockers(self, cells) -> list[str]:
        return target_blockers(self.m, self.gamma, cells)

    def picking_path(self, cells, o):
        """o's action with the other objects ignored, to the nearest free cell off T's paths (only its pick phases are used)."""
        k = (o, cells[o])
        if k not in self.pick_traj:
            fc = [c for c in free_cells(self.m, cells) if c not in self.t_sweep] or free_cells(self.m, cells)
            d = min(fc, key=lambda c: (dist(self.m, cells[o], c), c))
            self.pick_traj[k] = self.m.skill(cells, o, d)
        return self.pick_traj[k]

    def pick_blockers(self, cells, o) -> list[str] | None:
        """O_ip_b: [] if o's picking path is available (some grasp passes the planner's static pick screen: grasp pose + certified
        straight approach line, the analogue of the paper's linear picking path); else the objects in the way (None if none can be named)."""
        sc = self.m.screen(cells, o)
        if sc["pick_ok"]:
            return []
        return sc["pick_blockers"] or None

    def pick_sweep(self, cells, o) -> set[int]:
        tr = self.picking_path(cells, o)
        return set() if tr is None else self.m.swept(tr, o, cells[o], self.m.legal, phases=PICK_PHASES)

    # ------------------------------------------------------------------ Algorithm 1
    def new_region(self, node, oi, deps, n_max=M_REGIONS):
        m, cells = self.m, node.cells
        banned = set(self.t_sweep) | {m.goal_cell}
        for od in deps:
            banned |= self.pick_sweep(cells, od)
        cand = sorted((c for c in free_cells(m, cells) if c not in banned), key=lambda c: (dist(m, cells[oi], c), c))
        out = []
        for c in cand[:MAX_TRIES]:
            r = m.relocate(cells, node.q, oi, c)                            # lines 8-9: MP(p_i) with collision checking (the dataset's planner)
            self.st["mp_relocations"] += 1
            if r is not None:
                out.append((oi, c, r[0]))
                if len(out) >= n_max:
                    break
        return out

    # ------------------------------------------------------------------ Algorithm 2 (returns candidate actions; the caller adds children)
    def goal_available(self, cells) -> bool:
        """Line 1-2: O_b is empty -- T's obstacle-free paths clear every object, or T's picking and placing paths are available in the
        scene (the planner's static pick / place screens)."""
        if not self.t_blockers(cells):
            return True
        sc = self.m.screen(cells, "t0", self.m.goal_cell)
        return bool(sc["pick_ok"] and sc["place_ok"])

    def actions(self, node, n_regions=M_REGIONS, goal=True):
        """-> ('goal', None) if O_b is empty, ('move', [(o, cell, q_end)...]) or ('dead', None)."""
        moved = node.moved()
        if goal and self.goal_available(node.cells):
            return "goal", None
        if len(node.seq) >= MAX_RELOC:
            return "dead", None
        tb = self.t_blockers(node.cells)
        ob = [o for o in tb if o not in moved]
        if len(node.seq) >= STUCK_DEPTH and len(tb) >= self.n_tb0:            # no progress: every feasible object
            ob = [o for o in self.m.movable if o not in moved]
        ob.sort(key=lambda o: (float(np.linalg.norm(self.m.xy(node.cells[o]) - np.asarray(self.m.robot.base))), o))
        seen = set()
        while ob:
            nxt = []
            for oi in ob:
                seen.add(oi)
                pb = self.pick_blockers(node.cells, oi)
                if pb is None:
                    continue
                pb_mov = [o for o in pb if o != "t0" and o not in moved]
                if not pb:                                                    # only oi's picking path is available (lines 8-14)
                    regs = self.new_region(node, oi, [], n_regions)
                    if regs:
                        return "move", regs
                    continue
                for oj in pb_mov:                                             # lines 16-26
                    pj = self.pick_blockers(node.cells, oj)
                    if pj is not None and not pj:
                        regs = self.new_region(node, oj, [oi], n_regions)
                        if regs:
                            return "move", regs
                    elif oj not in seen:
                        nxt.append(oj)
            ob = list(dict.fromkeys(nxt))                                     # halting condition: next round's focus
        return "dead", None

    def goal_ok(self, node) -> bool:
        self.st["goal_checks"] += 1
        return self.m.done(node.cells, node.q)

    def child(self, node, a):
        o, c, q2 = a
        return Node({**node.cells, o: c}, q2, node.seq + [(o, c)], node)

    def reward(self, seq) -> float:
        return -sum(dist(self.m, self.m.cells0[o], c) for o, c in seq)

    # ------------------------------------------------------------------ MCTS
    def expand(self, node):
        """Expansion of a visited leaf. Returns a terminal node achieving the objective, or None."""
        node.expanded = True
        kind, acts = self.actions(node)
        self.st["expansions"] += 1
        if kind == "goal":
            g = Node(node.cells, node.q, node.seq, node, goal=True)
            if self.goal_ok(g):
                return g
            kind, acts = self.actions(node, goal=False)                       # MP refutes T's placement: keep clearing O_b
        if kind == "dead":
            node.dead = True
            return None
        node.children = [self.child(node, a) for a in acts]
        return None

    def simulate(self, node) -> float:
        cur = node
        for _ in range(MAX_RELOC + 1):
            kind, acts = self.actions(cur, n_regions=M_REGIONS)
            self.st["rollout_steps"] += 1
            if kind == "goal":
                return self.reward(cur.seq) if self.goal_ok(cur) else -FAIL_R
            if kind == "dead":
                return -FAIL_R
            cur = self.child(cur, acts[int(self.rng.integers(len(acts)))])
        return -FAIL_R

    def ucb(self, parent, c) -> float:
        if c.n == 0:
            return math.inf
        mean = c.w / c.n
        var = max(0.0, c.w2 / c.n - mean ** 2)
        lg = math.log(max(1, parent.n))
        return mean + math.sqrt(lg / c.n * min(0.25, var + math.sqrt(2 * lg / c.n)))

    def solve(self) -> Result:
        m = self.m
        self.gamma = relaxed_target(m)
        r = start(m, self.gamma)
        if r:
            return r
        self.t_sweep = m.swept(self.gamma, "t0", m.cells0["t0"], m.legal)
        root = Node(dict(m.cells0), m.qref, [])
        self.n_tb0 = len(self.t_blockers(root.cells))
        while True:
            if root.dead:
                return Result("ABSTAIN", [], "tree exhausted", dict(self.st))
            path, node = [root], root                                          # selection
            while node.expanded and not node.dead:
                live = [c for c in node.children if not c.dead]
                if not live:
                    node.dead = True
                    break
                node = max(live, key=lambda c: (self.ucb(path[-1], c), -len(c.seq)))
                path.append(node)
            if node.dead:
                for p in reversed(path[:-1]):                                  # propagate dead subtrees upward
                    if p.expanded and all(c.dead for c in p.children):
                        p.dead = True
                continue
            if node.n > 0 or node is root:                                     # expansion of a visited leaf
                g = self.expand(node)
                if g is not None:
                    return Result("REPAIR", list(g.seq), "objective achieved", dict(self.st))
                if node.dead or not node.children:
                    continue
                node = node.children[0]                                        # line 30: the first child found
                path.append(node)
            z = 1.0 + max(self.simulate(node), -FAIL_R) / FAIL_R              # normalised reward in [0, 1]
            for p in path:
                p.n += 1
                p.w += z
                p.w2 += z * z


def solve(m, rng) -> Result:
    return MSMCTS(m, rng).solve()
