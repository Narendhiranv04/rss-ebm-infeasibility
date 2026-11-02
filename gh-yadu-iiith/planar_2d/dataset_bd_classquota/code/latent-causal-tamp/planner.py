"""Certified connectors: direct segment, guide chain, link-0 sweep certificate and a
seeded RRT-Connect in joint space or in held-object pose space (Guide v1.6 G1).

Split from motion.py (size): motion.py owns trajectory records and IK-tracked
lines; this module owns search. Every returned path consists only of segments that
geometry.certify accepted; budget exhaustion is UNRESOLVED, never a proof.
"""
from __future__ import annotations

import math
import time
import zlib
from dataclasses import dataclass

import numpy as np

import geometry as ge
import motion as mo


@dataclass
class Plan:
    status: str                      # CLEAR (certified path), COLLISION (certified) or UNRESOLVED
    path: list
    nodes: int = 0
    seconds: float = 0.0
    reason: str = ""
    tree_xy: np.ndarray | None = None  # tree vertices as TCP (joint space) or object (pose space) xy
    direct: ge.Verdict | None = None
    ends: tuple | None = None        # (qa, qb) of the requested connection
    evals: int = 0                   # distance evaluations spent (verifier calls)
    wall_clock_cut: bool = False     # the wall-time safety cap ended search or shortcutting:
                                     # the outcome may differ under another machine load


def call_seed(base: int, key: str) -> int:
    """Per-query seed, independent of how many plans ran before in the process."""
    return (base * 1_000_003 + zlib.crc32(key.encode())) % 2 ** 32


def link0_sweep_blocker(checker, qa, qb) -> str | None:
    """Sound connector certificate. Link 0 and the elbow housing depend on q1 alone
    and joint 1 cannot wrap (limits inside [-pi, pi]), so every path from qa to qb
    points link 0 at every angle between qa[0] and qb[0]. With link 0 aimed at a
    resting disc (centre at distance rho, angle alpha), the disc is at most
    max(rho - L1, 0) from link 0's end; that is a definite violation whenever
    rho - L1 - max(w0/2, housing_r) - r < delta - eps (rho <= L1: the axis crosses it)."""
    robot, w = checker.robot, checker.world
    if not w.ids or min(robot.limits[0]) < -math.pi or max(robot.limits[0]) > math.pi:
        return None
    rel = w.xy - robot.base
    ang, rho = np.arctan2(rel[:, 1], rel[:, 0]), np.hypot(rel[:, 0], rel[:, 1])
    reach = robot.lengths[0] + max(robot.widths[0] / 2, robot.housing_r) + w.radius + checker.delta - checker.eps
    lo, hi = sorted((qa[0], qb[0]))
    hit = np.flatnonzero((rho < reach) & (ang >= lo) & (ang <= hi))
    if not hit.size:
        return None
    k = hit[0]
    return f"{'link0' if rho[k] <= robot.lengths[0] else 'housing1'}|{w.ids[k]}"


# ------------------------------------------------------------------ search spaces
class JointSpace:
    """Nodes are joint vectors; an edge is the certified joint-linear segment."""

    def __init__(self, checker, gap, max_evals, rng, step):
        self.ck, self.gap, self.evals, self.rng, self.step = checker, gap, max_evals, rng, step
        self.lo, self.hi = np.array(checker.robot.limits).T

    def sample(self):
        """Uniform joints; while carrying, an object pose drawn over the support and a
        random elbow branch mapped through exact IK, so samples satisfy the carry
        constraint on both branches (branch switches stay reachable)."""
        if self.ck.held is None:
            return self.rng.uniform(self.lo, self.hi)
        (w, h), r, local = self.ck.world.support, self.ck.held[1], self.ck.held[2]
        for _ in range(100):
            x, y, phi = self.rng.uniform([r, r, -math.pi], [w - r, h - r, math.pi])
            u = np.array([math.cos(phi), math.sin(phi)])
            tcp = np.array([x, y]) - local[0] * u - local[1] * np.array([-u[1], u[0]])
            sols = ge.ik(self.ck.robot, tcp, phi)
            if sols:
                return sols[self.rng.integers(len(sols))][1]
        return self.rng.uniform(self.lo, self.hi)

    def near(self, nodes, z) -> int:
        return int(np.argmin(np.sum((nodes - z) ** 2, 1)))

    def steer(self, a, b):
        n = float(np.linalg.norm(b - a))
        return b if n <= self.step else a + (b - a) * self.step / n

    def edge(self, a, b):
        ok = ge.certify(self.ck, a, b, self.gap, self.gap, self.evals).status == ge.CLEAR
        return [a, b] if ok else None

    def xy(self, z):
        return ge.fk(self.ck.robot, z)[0][3]


class PoseSpace:
    """Nodes are held-object poses z = (x, y, phi) on one elbow branch. An edge moves
    the object straight while phi turns the short way, IK-tracked on that branch and
    certified; sampling in [r, W-r] x [r, H-r] keeps the object over the support."""

    def __init__(self, checker, gap, max_evals, rng, step, phi_weight, local, branch, line_step, max_jump):
        self.ck, self.gap, self.evals, self.rng, self.step = checker, gap, max_evals, rng, step
        self.w, self.local, self.branch, self.line_step, self.jump = phi_weight, local, branch, line_step, max_jump
        (sw, sh), r = checker.world.support, checker.held[1]
        self.lo, self.hi = np.array([r, r, -math.pi]), np.array([sw - r, sh - r, math.pi])

    def sample(self):
        return self.rng.uniform(self.lo, self.hi)

    def _delta(self, a, b):
        d = np.asarray(b, float) - a
        d[..., 2] = ge.wrap(d[..., 2]) * self.w
        return d

    def near(self, nodes, z) -> int:
        return int(np.argmin(np.sum(self._delta(nodes, z) ** 2, 1)))

    def steer(self, a, b):
        d = self._delta(a, b)
        n = float(np.linalg.norm(d))
        if n <= self.step:
            return b
        out = a + d * self.step / n / np.array([1.0, 1.0, self.w])
        out[2] = float(ge.wrap(out[2]))
        return out

    def edge(self, a, b):
        d = self._delta(a, b)
        n = max(1, math.ceil(float(np.linalg.norm(d)) / self.line_step))
        s = np.linspace(0.0, 1.0, n + 1)
        qs, _ = mo.ik_track(self.ck.robot, a[:2] + s[:, None] * d[:2], a[2] + s * d[2] / self.w,
                            self.local, self.branch, self.jump)
        if qs is None or mo.certify_chain(self.ck, qs, self.gap, self.evals).status != ge.CLEAR:
            return None
        return qs

    def xy(self, z):
        return np.asarray(z[:2])


# ------------------------------------------------------------------ RRT-Connect
def _rrt_connect(space, za, zb, max_nodes: int, deadline: float, rounds: int):
    """Generic RRT-Connect; edges store their certified joint waypoints."""
    trees = [{"z": [za], "parent": [-1], "wp": [None]}, {"z": [zb], "parent": [-1], "wp": [None]}]
    count = lambda: len(trees[0]["z"]) + len(trees[1]["z"])  # noqa: E731
    spent = lambda: count() >= max_nodes or time.time() > deadline  # noqa: E731

    def extend(tree, target):
        nodes = np.array(tree["z"])
        i = space.near(nodes, target)
        new = space.steer(nodes[i], target)
        wp = space.edge(nodes[i], new)
        if wp is None:
            return None
        tree["z"].append(new), tree["parent"].append(i), tree["wp"].append(wp)
        return new

    while not spent():
        new = extend(trees[0], space.sample())
        reached = None if new is None else extend(trees[1], new)
        while reached is not None and not np.array_equal(reached, new) and not spent():
            reached = extend(trees[1], new)          # CONNECT: keep extending toward new
        if reached is not None and np.array_equal(reached, new):  # trees met: every edge certified
            nodes, edges = _join(trees, za)
            path, cut = _shortcut(space, nodes, edges, rounds, deadline)
            return path, count(), trees, "SOLVED", cut
        trees.reverse()
    stop = "NODES" if count() >= max_nodes else "WALL"   # node budget = deterministic; wall = safety cap
    return None, count(), trees, stop, stop == "WALL"


def _chain(tree) -> list[int]:
    i, out = len(tree["z"]) - 1, []
    while i >= 0:
        out.append(i)
        i = tree["parent"][i]
    return out                                        # newest node first, root last


def _join(trees, za):
    a, b = _chain(trees[0]), _chain(trees[1])
    nodes = [trees[0]["z"][i] for i in a[::-1]] + [trees[1]["z"][i] for i in b[1:]]
    edges = [trees[0]["wp"][i] for i in a[::-1][1:]] + [trees[1]["wp"][i][::-1] for i in b[:-1]]
    if not np.array_equal(nodes[0], za):
        nodes, edges = nodes[::-1], [e[::-1] for e in edges[::-1]]
    return nodes, edges


def _shortcut(space, nodes, edges, rounds: int, deadline: float):
    """Certified shortcutting on the node path; returns (joint waypoint path, whether
    the wall-time cap cut the rounds short, i.e. the result is load-dependent)."""
    cut = False
    for _ in range(rounds):
        if len(nodes) < 3:
            break
        if time.time() > deadline:
            cut = True
            break
        i, j = sorted(space.rng.choice(len(nodes), 2, replace=False))
        wp = space.edge(nodes[i], nodes[j]) if j - i > 1 else None
        if wp is not None:
            nodes, edges = nodes[:i + 1] + nodes[j:], edges[:i] + [wp] + edges[j:]
    return [edges[0][0]] + [q for e in edges for q in e[1:]], cut


def plan(checker, qa, qb, gap: float, pcfg: dict, max_evals: int, key: str, rrt: bool = True,
         guide=None, pose=None) -> Plan:
    """Certified connection qa -> qb, cheapest first: direct segment; optional guide
    chain; link-0 sweep certificate (COLLISION); then (rrt=True) RRT-Connect in joint
    space, or in object-pose space when pose = (za, zb, local, branch, line_step,
    max_jump) is given for a same-branch carry. Search edges get their own small
    evaluation budget (an edge that does not certify cheaply is discarded; CLEAR is
    sound at any budget)."""
    t0, e0 = time.time(), checker.n_evals
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    out = lambda st, path, reason, n=0, tree=None: Plan(st, path, n, time.time() - t0, reason,  # noqa: E731
                                                        tree, direct, (qa, qb), checker.n_evals - e0)
    direct = ge.certify(checker, qa, qb, gap, gap, max_evals)
    if direct.status == ge.CLEAR:
        return out(ge.CLEAR, [qa, qb], "DIRECT", 2)
    if guide is not None and np.allclose(guide[0], qa) and np.allclose(guide[-1], qb):
        path = [qa] + list(guide[1:-1]) + [qb]
        if mo.certify_chain(checker, path, gap, max_evals).status == ge.CLEAR:
            return out(ge.CLEAR, path, "GUIDE_LINE", len(path))
    blocker = link0_sweep_blocker(checker, qa, qb)
    if blocker:
        return out(ge.COLLISION, [], f"LINK0_SWEEP:{blocker}")
    if not rrt:
        return out(ge.UNRESOLVED, [], "DIRECT_ONLY")
    rng, ev = np.random.default_rng(call_seed(pcfg["seed"], key)), pcfg["edge_max_evals"]
    if pose is None:
        space, za, zb, name = JointSpace(checker, gap, ev, rng, pcfg["extend_step_rad"]), qa, qb, "RRT_JOINT"
    else:
        za, zb, local, branch, line_step, jump = pose
        space, name = PoseSpace(checker, gap, ev, rng, pcfg["pose_step_m"], pcfg["pose_phi_weight_m"],
                                local, branch, line_step, jump), "RRT_OBJECT_POSE"
        za, zb = np.asarray(za, float), np.asarray(zb, float)
    path, n, trees, stop, cut = _rrt_connect(space, za, zb, pcfg["max_nodes"], t0 + pcfg["max_wall_seconds"],
                                             pcfg["shortcut_rounds"])
    tree = np.array([space.xy(z) for t in trees for z in t["z"]])
    if path is None:
        res = out(ge.UNRESOLVED, [], f"BUDGET_EXHAUSTED:{stop}", n, tree)
    elif not (np.allclose(path[0], qa, atol=1e-9) and np.allclose(path[-1], qb, atol=1e-9)):
        res = out(ge.UNRESOLVED, [], "ENDPOINT_IK_MISMATCH", n, tree)
    else:
        res = out(ge.CLEAR, [qa] + path[1:-1] + [qb], name, n, tree)
    res.wall_clock_cut = cut
    return res
