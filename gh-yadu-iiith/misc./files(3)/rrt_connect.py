"""
RRT-Connect (Kuffner & LaValle, ICRA 2000) in joint space.

Two trees are grown, one from the start and one from the goal.  Every iteration:

    1. EXTEND  tree A one step (<= `step` rad) toward a uniformly random configuration;
    2. CONNECT tree B toward the node just added to A: keep stepping straight at it until
               it is reached (-> path found) or something is in the way;
    3. swap A and B.

The greedy CONNECT step is what makes RRT-Connect fast in mostly-free spaces: as soon as the
two trees can "see" each other they join in one go.  The result is feasible but not optimal,
so callers should shortcut / smooth it (see rrt_star.shortcut, rrt_star.smooth).

Like RRTStar, the planner is robot-agnostic: it only needs joint limits and the two
callbacks `is_free(q)` and `edge_free(qa, qb)`.
"""
import time
import numpy as np

TRAPPED, ADVANCED, REACHED = 0, 1, 2


class _Tree:
    def __init__(self, root, capacity):
        self.Q = np.empty((capacity, len(root)))
        self.Q[0] = root
        self.parent = np.full(capacity, -1, dtype=int)
        self.n = 1

    def nearest(self, q):
        return int(np.argmin(np.linalg.norm(self.Q[:self.n] - q, axis=1)))

    def add(self, q, parent):
        if self.n == len(self.Q):                       # grow storage if needed
            self.Q = np.vstack([self.Q, np.empty_like(self.Q)])
            self.parent = np.concatenate([self.parent, np.full(len(self.parent), -1)])
        k = self.n
        self.Q[k], self.parent[k] = q, parent
        self.n += 1
        return k

    def branch(self, k):
        """Configurations from node k back to the root."""
        out = []
        while k >= 0:
            out.append(self.Q[k].copy())
            k = self.parent[k]
        return out


class RRTConnect:
    """Bidirectional RRT-Connect.

    Parameters
    ----------
    is_free     : q -> bool, state validity
    edge_free   : (qa, qb) -> bool, validity of the straight segment qa -> qb
    lower/upper : joint limits used for sampling
    step        : maximum extension per EXTEND step (rad)
    max_iters / max_time : budgets
    try_direct  : first test the straight start->goal segment (off by default, so the trees
                  are always grown -- shortcutting straightens an easy path afterwards anyway)
    """

    def __init__(self, is_free, edge_free, lower, upper, step=0.3, max_iters=5000, max_time=10.0,
                 try_direct=False, seed=None):
        self.is_free, self.edge_free = is_free, edge_free
        self.lo, self.hi = np.asarray(lower, float), np.asarray(upper, float)
        self.step, self.max_iters, self.max_time = step, max_iters, max_time
        self.try_direct = try_direct
        self.rng = np.random.default_rng(seed)
        self.stats = {}

    def _extend(self, T, q_target):
        """One step of tree T toward q_target."""
        i = T.nearest(q_target)
        d = np.linalg.norm(q_target - T.Q[i])
        if d <= self.step:
            q_new, status = q_target, REACHED
        else:
            q_new, status = T.Q[i] + (q_target - T.Q[i]) * (self.step / d), ADVANCED
        if not self.is_free(q_new) or not self.edge_free(T.Q[i], q_new):
            return TRAPPED, -1
        return status, T.add(q_new, i)

    def _connect(self, T, q_target):
        """Greedily extend T toward q_target until it is reached or blocked."""
        while True:
            status, k = self._extend(T, q_target)
            if status != ADVANCED:
                return status, k

    def plan(self, q_start, q_goal):
        t0 = time.time()
        start, goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        self.stats = dict(iters=0, nodes=2, time=0.0, direct=False)
        if not self.is_free(start):
            raise RuntimeError("RRT-Connect: start configuration is in collision")
        if not self.is_free(goal):
            raise RuntimeError("RRT-Connect: goal configuration is in collision")
        if self.try_direct and self.edge_free(start, goal):
            self.stats.update(direct=True, time=time.time() - t0)
            return [start, goal]

        cap = 2 * self.max_iters + 2
        trees = [_Tree(start, cap), _Tree(goal, cap)]   # 0 = start tree, 1 = goal tree
        a = 0                                            # index of the tree to EXTEND
        it = 0
        for it in range(1, self.max_iters + 1):
            if time.time() - t0 > self.max_time:
                break
            A, B = trees[a], trees[1 - a]
            status, k = self._extend(A, self.rng.uniform(self.lo, self.hi))
            if status != TRAPPED:
                status_b, j = self._connect(B, A.Q[k])
                if status_b == REACHED:                  # A[k] and B[j] are the same configuration
                    S, s_node, G, g_node = (A, k, B, j) if a == 0 else (B, j, A, k)
                    path = S.branch(s_node)[::-1] + G.branch(g_node)[1:]
                    self.stats.update(iters=it, nodes=A.n + B.n, time=time.time() - t0)
                    return path
            a = 1 - a                                    # swap the roles of the trees
        self.stats.update(iters=it, nodes=trees[0].n + trees[1].n, time=time.time() - t0)
        return None
