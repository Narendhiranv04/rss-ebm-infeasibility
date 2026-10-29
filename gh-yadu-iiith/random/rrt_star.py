"""
Joint-space RRT* (with informed sampling) + MuJoCo collision checking + path post-processing.

The planner itself (`RRTStar`) is robot-agnostic: it only needs joint limits and an
`is_free(q) -> bool` callback.  `MujocoCollisionChecker` provides that callback for a
MuJoCo arm, including support for an object rigidly held in the gripper.

Pipeline used by the environment
--------------------------------
    path = RRTStar(checker.is_free, lo, hi).plan(q_start, q_goal)   # sparse waypoints
    path = shortcut(path, checker)                                  # remove detours
    path = smooth(path, checker)                                    # round the corners
    traj = time_parameterize(path, vmax, amax, dt)                  # one q per sim step
"""
import time
from contextlib import contextmanager
import numpy as np
import mujoco
from math import gamma as _gamma_fn, log, pi


# ============================================================================ collision checking
class MujocoCollisionChecker:
    """Collision checker for the arm, run on a *separate planning model*.

    The planning model should be compiled from the same spec as the simulation model but with
    (a) excludes for contact pairs that are always touching by design (gripper internals,
        base <-> link1), and
    (b) a positive `geom_margin` on moving geoms, so near-misses closer than `margin`
        are reported as contacts (a safety buffer around the arm and held object).

    Every geom is given a role:
        0 = static obstacle   (world geoms, link_base, objects not being held)
        1 = robot             (anything kinematically downstream of the arm joints)
        2 = held object       (set with `attach`, follows the grasp site rigidly)
    A configuration is in collision if
        robot/held  vs static           : dist < margin
        robot       vs robot (diff grp) : dist < self_margin   (self-collision)
        held        vs robot (non-hand) : dist < self_margin
    Contacts static-static (e.g. cubes resting on the shelf) and held-vs-hand are ignored, as are
    contacts with geoms temporarily listed in `ignoring(...)` (e.g. the object the fingers close on).
    `margin` is a plain attribute and can be lowered at run time (down to 0) for tight spaces;
    it cannot exceed the geom_margin compiled into the planning model.
    """

    def __init__(self, model, arm_qadr, robot_bodies, hand_bodies, site_id,
                 margin=0.015, self_margin=0.0):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.asarray(arm_qadr)
        self.site_id = site_id
        self.margin = margin
        self.self_margin = self_margin

        robot_bodies, hand_bodies = set(robot_bodies), set(hand_bodies)
        gb = model.geom_bodyid
        self.base_role = np.array([1 if gb[g] in robot_bodies else 0 for g in range(model.ngeom)])
        # self-collision groups: each arm link is its own group, the whole hand is one group
        self.group = np.array([-1 if gb[g] in hand_bodies else gb[g] for g in range(model.ngeom)])
        self.is_hand = np.array([gb[g] in hand_bodies for g in range(model.ngeom)])
        self.role = self.base_role.copy()
        self.ignored = np.zeros(model.ngeom, dtype=bool)
        self._attached = []           # list of (geom_id, p_rel, R_rel) in grasp-site frame
        self.n_checks = 0

    # -- scene state
    def sync(self, qpos):
        """Copy the full simulation qpos (object poses etc.) into the planning data."""
        self.d.qpos[:] = qpos

    def attach(self, geom_ids, sim_model, sim_data):
        """Treat `geom_ids` as rigidly attached to the grasp site (uses the current sim pose)."""
        ps = sim_data.site_xpos[self.site_id]
        Rs = sim_data.site_xmat[self.site_id].reshape(3, 3)
        self._attached = []
        self.role = self.base_role.copy()
        for g in geom_ids:
            pg = sim_data.geom_xpos[g]
            Rg = sim_data.geom_xmat[g].reshape(3, 3)
            self._attached.append((g, Rs.T @ (pg - ps), Rs.T @ Rg))
            self.role[g] = 2

    def attach_relative(self, items):
        """Attach geoms with given poses relative to the grasp site: items = [(geom, p_rel, R_rel)].
        Used to check a carry before the object is actually picked up."""
        self._attached = [(g, np.asarray(p, float), np.asarray(R, float)) for g, p, R in items]
        self.role = self.base_role.copy()
        for g, _, _ in items:
            self.role[g] = 2

    def detach(self):
        self._attached = []
        self.role = self.base_role.copy()

    @contextmanager
    def ignoring(self, geom_ids):
        """Temporarily ignore every contact involving `geom_ids`."""
        old = self.ignored.copy()
        self.ignored[list(geom_ids)] = True
        try:
            yield
        finally:
            self.ignored = old

    # -- queries
    def _colliding(self, q):
        """Boolean mask over the current contacts that count as collisions at configuration q."""
        m, d = self.m, self.d
        self.n_checks += 1
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(m, d)
        if self._attached:                     # move held geoms with the grasp site
            ps = d.site_xpos[self.site_id]
            Rs = d.site_xmat[self.site_id].reshape(3, 3)
            for g, p_rel, R_rel in self._attached:
                d.geom_xpos[g] = ps + Rs @ p_rel
                d.geom_xmat[g] = (Rs @ R_rel).ravel()
        mujoco.mj_collision(m, d)
        if d.ncon == 0:
            return np.zeros(0, dtype=bool)
        geom = d.contact.geom
        dist = d.contact.dist
        g1, g2 = geom[:, 0], geom[:, 1]
        r1, r2 = self.role[g1], self.role[g2]
        live = ~(self.ignored[g1] | self.ignored[g2])
        moving_vs_static = ((r1 == 0) != (r2 == 0)) & (dist < self.margin)
        robot_robot = (r1 == 1) & (r2 == 1) & (self.group[g1] != self.group[g2]) & (dist < self.self_margin)
        held_robot = (((r1 == 2) & (r2 == 1) & ~self.is_hand[g2]) |
                      ((r2 == 2) & (r1 == 1) & ~self.is_hand[g1])) & (dist < self.self_margin)
        return live & (moving_vs_static | robot_robot | held_robot)

    def is_free(self, q):
        return not self._colliding(q).any()

    def collisions(self, q):
        """Set of (geom1, geom2) id pairs in collision at q -- for error messages."""
        mask = self._colliding(q)
        return {tuple(sorted(map(int, g))) for g in self.d.contact.geom[:len(mask)][mask]}

    def edge_free(self, qa, qb, res=0.03):
        """Check the straight joint-space segment qa->qb (endpoints assumed already valid).
        Points are visited coarse-to-fine (bisection order) so collisions are found early."""
        n = int(np.ceil(np.max(np.abs(qb - qa)) / res))
        if n <= 1:
            return True
        for k in _bisection_order(n):
            if not self.is_free(qa + (k / n) * (qb - qa)):
                return False
        return True

    def path_free(self, path, res=0.03):
        return all(self.is_free(q) for q in path) and \
            all(self.edge_free(a, b, res) for a, b in zip(path[:-1], path[1:]))


_BISECT_CACHE = {}


def _bisection_order(n):
    """Interior indices 1..n-1 ordered midpoint-first (breadth-first bisection)."""
    if n not in _BISECT_CACHE:
        out, queue = [], [(0, n)]
        while queue:
            lo, hi = queue.pop(0)
            if hi - lo < 2:
                continue
            mid = (lo + hi) // 2
            out.append(mid)
            queue += [(lo, mid), (mid, hi)]
        _BISECT_CACHE[n] = out
    return _BISECT_CACHE[n]


# ============================================================================ RRT*
class _Tree:
    """Search tree with parent pointers, cost-to-root and child lists (for cost propagation)."""

    def __init__(self, root, capacity):
        self.Q = np.empty((capacity, len(root))); self.Q[0] = root
        self.parent = np.full(capacity, -1, dtype=int)
        self.cost = np.zeros(capacity)
        self.children = [[] for _ in range(capacity)]
        self.n = 1

    def add(self, q, parent, cost):
        k = self.n; self.n += 1
        self.Q[k], self.parent[k], self.cost[k] = q, parent, cost
        self.children[parent].append(k)
        return k

    def reparent(self, j, new_parent, new_cost):
        self.children[self.parent[j]].remove(j)
        self.parent[j] = new_parent
        self.children[new_parent].append(j)
        diff = new_cost - self.cost[j]
        stack = [j]                            # propagate the cost change down the subtree
        while stack:
            u = stack.pop()
            self.cost[u] += diff
            stack += self.children[u]

    def branch(self, k):
        """Configurations from node k back to the root."""
        out = []
        while k >= 0:
            out.append(self.Q[k].copy()); k = self.parent[k]
        return out


class RRTStar:
    """Asymptotically optimal RRT* in joint space (Karaman & Frazzoli 2011).

    * bidirectional=True  : grows one RRT* tree from the start and one from the goal and
                            connects them (as in RRT*-Connect / B-RRT*).  Much better when start
                            and goal sit in narrow pockets, e.g. right in front of a cabinet.
      bidirectional=False : textbook single-tree RRT* with goal biasing.
    * informed=True       : once a solution of cost c_best exists, samples are drawn only from
                            the ellipsoid of states that could still improve it (Gammell 2014).

    Cost = Euclidean joint-space path length.

    Parameters
    ----------
    is_free     : q -> bool, state validity
    edge_free   : (qa, qb) -> bool, segment validity (endpoints already valid)
    lower/upper : joint limits used for sampling
    step        : max extension per iteration (rad)
    r_max       : cap on the rewiring / connection radius (rad)
    goal_bias   : probability of sampling the goal (single-tree mode, before a solution exists)
    max_iters / max_time : hard budgets
    post_solution_iters  : keep optimising this many iterations after the first solution
    """

    def __init__(self, is_free, edge_free, lower, upper, step=0.5, r_max=2.0, goal_bias=0.1,
                 max_iters=5000, max_time=20.0, post_solution_iters=400, informed=True,
                 bidirectional=True, try_direct=True, seed=None):
        self.is_free, self.edge_free = is_free, edge_free
        self.lo, self.hi = np.asarray(lower, float), np.asarray(upper, float)
        self.dim = len(self.lo)
        self.step, self.r_max, self.goal_bias = step, r_max, goal_bias
        self.max_iters, self.max_time = max_iters, max_time
        self.post_solution_iters = post_solution_iters
        self.informed, self.bidirectional, self.try_direct = informed, bidirectional, try_direct
        self.rng = np.random.default_rng(seed)
        # gamma_RRT* > 2 (1 + 1/d)^(1/d) (mu(X_free) / zeta_d)^(1/d); X_free is upper-bounded by the box
        d = self.dim
        zeta_d = pi ** (d / 2) / _gamma_fn(d / 2 + 1)                # volume of the unit d-ball
        mu = np.prod(self.hi - self.lo)
        self.gamma = 2.0 * (1 + 1 / d) ** (1 / d) * (mu / zeta_d) ** (1 / d)
        self.stats = {}

    # -- helpers
    def _radius(self, n):
        return min(self.gamma * (log(n) / n) ** (1 / self.dim), self.r_max) if n > 1 else self.r_max

    def _sample(self, start, goal, c_best):
        if c_best < np.inf and self.informed:
            return self._sample_informed(start, goal, c_best)
        if not self.bidirectional and self.rng.random() < self.goal_bias:
            return goal.copy()
        return self.rng.uniform(self.lo, self.hi)

    def _sample_informed(self, start, goal, c_best):
        """Uniform sample inside the prolate hyperspheroid of states that could improve c_best."""
        c_min = np.linalg.norm(goal - start)
        centre = 0.5 * (start + goal)
        a1 = (goal - start) / max(c_min, 1e-12)
        U, _, Vt = np.linalg.svd(np.outer(a1, np.eye(self.dim)[0]))
        diag = np.ones(self.dim); diag[-1] = np.linalg.det(U) * np.linalg.det(Vt)
        C = U @ np.diag(diag) @ Vt                               # rotation e1 -> a1
        r1 = c_best / 2.0
        ri = np.sqrt(max(c_best ** 2 - c_min ** 2, 0.0)) / 2.0
        L = np.diag([r1] + [ri] * (self.dim - 1))
        for _ in range(50):                                     # rejection against joint limits
            x = self.rng.normal(size=self.dim)
            x *= self.rng.random() ** (1 / self.dim) / np.linalg.norm(x)   # uniform in unit ball
            q = C @ L @ x + centre
            if np.all(q >= self.lo) and np.all(q <= self.hi):
                return q
        return np.clip(q, self.lo, self.hi)

    def _extend(self, T, q_rand):
        """One RRT* step on tree T: steer, choose best parent, rewire.  Returns new index or None."""
        dists = np.linalg.norm(T.Q[:T.n] - q_rand, axis=1)
        i_near = int(np.argmin(dists))
        dn = dists[i_near]
        if dn < 1e-9:
            return None
        q_new = T.Q[i_near] + (q_rand - T.Q[i_near]) * min(1.0, self.step / dn)
        if not self.is_free(q_new):
            return None
        # choose the lowest-cost collision-free parent within radius r
        d_new = np.linalg.norm(T.Q[:T.n] - q_new, axis=1)
        r = self._radius(T.n)
        near = np.flatnonzero(d_new <= max(r, d_new[i_near] + 1e-9))
        p = -1
        for j in near[np.argsort(T.cost[near] + d_new[near])]:   # lazy: cheapest first
            if self.edge_free(T.Q[j], q_new):
                p = int(j); break
        if p < 0:
            return None
        k = T.add(q_new, p, T.cost[p] + d_new[p])
        # rewire neighbours through the new node if that is cheaper
        for j in near:
            if j == p:
                continue
            c_via = T.cost[k] + d_new[j]
            if c_via + 1e-9 < T.cost[j] and self.edge_free(q_new, T.Q[j]):
                T.reparent(j, k, c_via)
        return k

    def _connect(self, A, k, B, c_best):
        """Try to join node k of tree A to the cheapest reachable node of tree B.
        Returns (j, d) or None."""
        q = A.Q[k]
        d = np.linalg.norm(B.Q[:B.n] - q, axis=1)
        r = max(self._radius(A.n + B.n), self.step)
        near = np.flatnonzero(d <= r)
        if near.size == 0:
            return None
        for j in near[np.argsort(B.cost[near] + d[near])]:
            if A.cost[k] + d[j] + B.cost[j] >= c_best:
                return None                    # sorted: nothing further can improve
            if self.edge_free(q, B.Q[j]):
                return int(j), float(d[j])
        return None

    # -- main loop
    def plan(self, q_start, q_goal):
        t0 = time.time()
        start, goal = np.asarray(q_start, float), np.asarray(q_goal, float)
        self.stats = dict(iters=0, nodes=1, time=0.0, first_cost=None, cost=None, first_iter=None,
                          direct=False)
        if not self.is_free(start):
            raise RuntimeError("RRT*: start configuration is in collision")
        if not self.is_free(goal):
            raise RuntimeError("RRT*: goal configuration is in collision")
        if self.try_direct and self.edge_free(start, goal):     # straight line is already optimal
            c = float(np.linalg.norm(goal - start))
            self.stats.update(direct=True, first_cost=c, cost=c, time=time.time() - t0)
            return [start, goal]

        cap = self.max_iters + 2
        trees = [_Tree(start, cap), _Tree(goal, cap)]           # 0 = start tree, 1 = goal tree
        links = []                                               # (node in start tree, node in goal tree, d)
        c_best, best, first_iter, it = np.inf, None, None, 0

        for it in range(1, self.max_iters + 1):
            if time.time() - t0 > self.max_time:
                break
            if first_iter is not None and it - first_iter > self.post_solution_iters:
                break
            a = it % 2 if self.bidirectional else 0
            A, B = trees[a], trees[1 - a]
            k = self._extend(A, self._sample(start, goal, c_best))
            if k is None:
                continue
            con = self._connect(A, k, B, c_best)
            if con is not None:
                j, d = con
                links.append((k, j, d) if a == 0 else (j, k, d))
            if links:                          # rewiring may have lowered old links' costs
                S, G = trees
                costs = [S.cost[i] + d + G.cost[j] for i, j, d in links]
                b = int(np.argmin(costs))
                if costs[b] < c_best:
                    c_best, best = float(costs[b]), links[b]
                    if first_iter is None:
                        first_iter = it
                        self.stats.update(first_cost=c_best, first_iter=it)

        self.stats.update(iters=it, nodes=trees[0].n + trees[1].n, time=time.time() - t0,
                          cost=None if best is None else c_best)
        if best is None:
            return None
        i, j, _ = best
        return trees[0].branch(i)[::-1] + trees[1].branch(j)


# ============================================================================ post-processing
def path_length(path):
    return float(sum(np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])))


def shortcut(path, edge_free, iters=300, seed=None):
    """Random partial shortcutting: pick two random points anywhere along the path (not only at
    vertices) and replace the stretch between them with a straight segment if it is collision-free.
    Points on an already-validated path are themselves valid, so only the new segment is checked."""
    rng = np.random.default_rng(seed)
    path = [np.asarray(q, float) for q in path]
    for _ in range(iters):
        if len(path) < 3:
            break
        seg = np.array([np.linalg.norm(b - a) for a, b in zip(path[:-1], path[1:])])
        s = np.concatenate([[0.0], np.cumsum(seg)])
        t1, t2 = np.sort(rng.uniform(0.0, s[-1], 2))
        i1 = min(int(np.searchsorted(s, t1, "right")) - 1, len(seg) - 1)
        i2 = min(int(np.searchsorted(s, t2, "right")) - 1, len(seg) - 1)
        if i1 == i2:
            continue                           # same segment: already straight
        qa = path[i1] + (t1 - s[i1]) / seg[i1] * (path[i1 + 1] - path[i1])
        qb = path[i2] + (t2 - s[i2]) / seg[i2] * (path[i2 + 1] - path[i2])
        if edge_free(qa, qb):
            path = path[:i1 + 1] + [qa, qb] + path[i2 + 1:]
    # drop (near-)collinear interior vertices left behind by partial shortcuts
    out = [path[0]]
    for k in range(1, len(path) - 1):
        u, v = path[k] - out[-1], path[k + 1] - path[k]
        nu, nv = np.linalg.norm(u), np.linalg.norm(v)
        if nu < 1e-6 or (nv > 1e-9 and np.dot(u, v) / (nu * nv) > 1 - 1e-6):
            continue
        out.append(path[k])
    return simplify(out + [path[-1]], edge_free)


def simplify(path, edge_free):
    """Greedy vertex reduction: from each kept vertex, jump to the farthest one it can see."""
    out, i = [path[0]], 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not edge_free(path[i], path[j]):
            j -= 1
        out.append(path[j]); i = j
    return out


def densify(path, max_step=0.04):
    out = [path[0]]
    for a, b in zip(path[:-1], path[1:]):
        n = max(1, int(np.ceil(np.max(np.abs(b - a)) / max_step)))
        out += [a + (k / n) * (b - a) for k in range(1, n + 1)]
    return np.array(out)


def smooth(path, checker, max_step=0.04, passes=(60, 30, 15, 5)):
    """Round the corners of a piecewise-linear path by repeated neighbour averaging
    (endpoints fixed).  Tries the strongest smoothing first and keeps the first result
    that is still collision-free; falls back to the un-smoothed dense path."""
    dense = densify(path, max_step)
    for n_pass in passes:
        p = dense.copy()
        for _ in range(n_pass):
            p[1:-1] = 0.25 * p[:-2] + 0.5 * p[1:-1] + 0.25 * p[2:]
        if checker.path_free(list(p)):
            return p
    return dense


def time_parameterize(path, vmax=1.0, amax=2.0, dt=0.002, ds=0.005):
    """Time-scale a geometric path under joint velocity and acceleration limits.

    The path is resampled uniformly in inf-norm arc length s (so |dq_i/ds| <= 1).  Along the path
    q_dot = q'(s) s_dot and q_ddot = q'(s) s_ddot + q''(s) s_dot^2, so we bound
        s_dot  <= vmax                                   (velocity)
        s_dot  <= sqrt(a_c / max_i |q_i''(s)|)            (centripetal part: slow down in curves)
        |s_ddot| <= a_t                                   (tangential part)
    with a_t + a_c = amax, then run the classic forward / backward pass over the speed-limit
    curve (a simplified TOPP), starting and ending at rest.  Returns one configuration per `dt`.
    """
    P = np.asarray(path, float)
    seg = np.max(np.abs(np.diff(P, axis=0)), axis=1)
    keep = np.concatenate([[True], seg > 1e-12])
    P = P[keep]
    if len(P) < 2:
        return P[-1:].copy()
    s = np.concatenate([[0.0], np.cumsum(np.max(np.abs(np.diff(P, axis=0)), axis=1))])
    L = s[-1]
    n = max(2, int(np.ceil(L / ds)))
    su = np.linspace(0.0, L, n + 1); h = su[1] - su[0]
    Q = np.stack([np.interp(su, s, P[:, j]) for j in range(P.shape[1])], axis=1)

    a_t, a_c = 0.5 * amax, 0.5 * amax
    d1 = np.gradient(Q, h, axis=0)
    d2 = np.gradient(d1, h, axis=0)
    vlim = np.minimum(vmax / np.maximum(np.max(np.abs(d1), axis=1), 1e-9),
                      np.sqrt(a_c / np.maximum(np.max(np.abs(d2), axis=1), 1e-9)))
    v = vlim.copy(); v[0] = v[-1] = 0.0
    for k in range(n):                              # forward: accelerate as hard as allowed
        v[k + 1] = min(v[k + 1], np.sqrt(v[k] ** 2 + 2 * a_t * h))
    for k in range(n - 1, -1, -1):                  # backward: brake in time for what's ahead
        v[k] = min(v[k], np.sqrt(v[k + 1] ** 2 + 2 * a_t * h))
    t = np.concatenate([[0.0], np.cumsum(2 * h / np.maximum(v[:-1] + v[1:], 1e-9))])
    tt = np.arange(0.0, t[-1] + dt, dt)
    s_t = np.interp(tt, t, su)
    return np.stack([np.interp(s_t, su, Q[:, j]) for j in range(Q.shape[1])], axis=1)
