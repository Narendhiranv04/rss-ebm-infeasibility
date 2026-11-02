"""Planar 3R robot geometry: kinematics, collision bodies, clearance certification.

Methodology v1.4 §2.1 and §3; Guide v1.6 G1.

Kinematics: theta_j = q_1 + ... + q_j. Joints J0 = base b, J1 = elbow, J2 = wrist,
J3 = TCP = b + sum_j L_j (cos theta_j, sin theta_j). The TCP sits at the palm tip;
the tool x axis points along theta_3 (phi_TCP = q_1 + q_2 + q_3).

Bodies are convex cores (point, segment or CCW polygon) dilated by a radius: a link
capsule is a segment with radius w/2, a disc is a point with radius r. The signed
distance between two bodies is core distance minus both radii (<= 0 if cores meet).
Every tested pair must keep distance >= delta, except fingers touching the object
named by the contact schedule, which must only not penetrate (distance >= 0).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

CLEAR, COLLISION, UNRESOLVED = "CLEAR", "COLLISION", "UNRESOLVED"
ROBOT_BODIES = ("base", "link0", "housing1", "link1", "housing2", "link2",
                "palm", "rail", "finger_l", "finger_r")
FRAME = {"base": 0, "link0": 1, "housing1": 2, "link1": 2, "housing2": 3, "link2": 3,
         "palm": 3, "rail": 3, "finger_l": 3, "finger_r": 3, "held": 3}
FINGERS = ("finger_l", "finger_r")
# |cos q2| may exceed 1 by rounding for poses exactly on the reach boundary; such poses
# are solved (clipped), so "no IK" is only claimed beyond this tolerance.
IK_BOUNDARY_TOL = 1e-12
# Bodies joined by a joint or a rigid mount; never tested against each other. The
# config adds its link pairs (link0-link1, link1-link2) on top of these.
MOUNTED = {("base", "link0"), ("link0", "housing1"), ("housing1", "link1"),
           ("link1", "housing2"), ("housing2", "link2"), ("link2", "palm"),
           ("palm", "rail"), ("palm", "finger_l"), ("palm", "finger_r"),
           ("rail", "finger_l"), ("rail", "finger_r"), ("finger_l", "finger_r")}


@dataclass(frozen=True)
class Robot:
    base: tuple
    lengths: tuple
    widths: tuple
    limits: tuple
    q_home: tuple
    base_r: float
    housing_r: float
    palm_len: float
    palm_w: float
    rail_depth: float
    finger_len: float
    finger_t: float
    open_gap: float
    closed_gap: float
    grasp_x: float
    excluded: frozenset


def make_robot(cfg: dict) -> Robot:
    a, g = cfg["arm2d"], cfg["gripper2d"]
    links = {frozenset((f"link{i}", f"link{j}")) for i, j in a["self_collision_exclusions"]}
    return Robot(tuple(a["base_xy_m"]), tuple(a["link_lengths_m"]), tuple(a["link_full_widths_m"]),
                 tuple(map(tuple, a["joint_limits_rad"])), tuple(a["q_home_rad"]), a["base_radius_m"],
                 a["joint_housing_radius_m"], g["palm_length_m"], g["palm_full_width_m"],
                 g["rail_depth_m"], g["finger_length_m"], g["finger_thickness_m"], g["open_gap_m"],
                 g["closed_gap_m"], g["grasp_center_x_m"],
                 frozenset(links | {frozenset(p) for p in MOUNTED}))


# ------------------------------------------------------------------ kinematics
def wrap(angle):
    """Angle(s) wrapped to [-pi, pi)."""
    return (np.asarray(angle) + math.pi) % (2 * math.pi) - math.pi


def fk(robot: Robot, q) -> tuple[np.ndarray, np.ndarray]:
    """Joint positions J (4x2: base, elbow, wrist, TCP) and absolute angles theta (3,)."""
    theta = np.cumsum(np.asarray(q, dtype=float))
    steps = np.asarray(robot.lengths)[:, None] * np.stack([np.cos(theta), np.sin(theta)], 1)
    return np.vstack([robot.base, robot.base + np.cumsum(steps, 0)]), theta


def within_limits(robot: Robot, q) -> bool:
    return all(lo <= v <= hi for v, (lo, hi) in zip(q, robot.limits))


def ik(robot: Robot, tcp, phi: float) -> list[tuple[int, np.ndarray]]:
    """Exact IK for TCP pose (x, y, phi): subtract the third link, solve both
    two-link elbow branches (+1: q2 > 0, -1: q2 < 0), reconstruct q3, wrap and keep
    solutions inside joint limits. Returns [(branch, q)]."""
    l1, l2, l3 = robot.lengths
    w = np.asarray(tcp) - l3 * np.array([math.cos(phi), math.sin(phi)]) - robot.base
    c2 = (w @ w - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    if abs(c2) > 1.0 + IK_BOUNDARY_TOL:   # outside the annulus beyond rounding: no IK
        return []
    out = []
    for branch in (1, -1):
        q2 = branch * math.acos(min(1.0, max(-1.0, c2)))
        q1 = math.atan2(w[1], w[0]) - math.atan2(l2 * math.sin(q2), l1 + l2 * math.cos(q2))
        q = wrap(np.array([q1, q2, phi - q1 - q2]))
        if within_limits(robot, q) and not any(np.allclose(q, p) for _, p in out):
            out.append((branch, q))
    return out


def tool_point(robot: Robot, q, local) -> np.ndarray:
    """World position of a point given in the tool (TCP) frame."""
    joints, theta = fk(robot, q)
    c, s = math.cos(theta[2]), math.sin(theta[2])
    return joints[3] + np.array([c * local[0] - s * local[1], s * local[0] + c * local[1]])


def tool_local(robot: Robot, q, world) -> np.ndarray:
    """Tool-frame coordinates of a world point (inverse of tool_point)."""
    joints, theta = fk(robot, q)
    d = np.asarray(world) - joints[3]
    c, s = math.cos(theta[2]), math.sin(theta[2])
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


# ------------------------------------------------------------------ bodies
def _box(origin, angle, x0, x1, y0, y1) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    local = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])  # CCW
    return np.asarray(origin) + local @ np.array([[c, s], [-s, c]])


def bodies(robot: Robot, q, gap: float, held=None) -> dict[str, tuple[np.ndarray, float]]:
    """All robot bodies (and a held disc) as name -> (core vertices, radius).

    Gripper (tool frame): palm x in [-palm_len, 0], |y| <= palm_w/2; the finger rail
    x in [-rail_depth, 0], |y| <= open_gap/2 + finger_t carries both fingers, which
    slide on it with inner faces at y = +-gap/2 and span x in [0, finger_len]. Link 2
    ends at the palm's back face. held = (obj_id, radius, (x, y) in tool frame).
    """
    j, th = fk(robot, q)
    phi, (w0, w1, w2) = th[2], robot.widths
    u = np.array([math.cos(phi), math.sin(phi)])
    half, t, rail = gap / 2, robot.finger_t, robot.open_gap / 2 + robot.finger_t
    out = {"base": (j[:1], robot.base_r), "link0": (j[0:2], w0 / 2), "housing1": (j[1:2], robot.housing_r),
           "link1": (j[1:3], w1 / 2), "housing2": (j[2:3], robot.housing_r),
           "link2": (np.array([j[2], j[3] - robot.palm_len * u]), w2 / 2),
           "palm": (_box(j[3], phi, -robot.palm_len, 0, -robot.palm_w / 2, robot.palm_w / 2), 0.0),
           "rail": (_box(j[3], phi, -robot.rail_depth, 0, -rail, rail), 0.0),
           "finger_l": (_box(j[3], phi, 0, robot.finger_len, half, half + t), 0.0),
           "finger_r": (_box(j[3], phi, 0, robot.finger_len, -half - t, -half), 0.0)}
    if held is not None:
        out["held"] = (tool_point(robot, q, held[2])[None], held[1])
    return out


# ------------------------------------------------------------------ distances
def _pt_seg(p, a, b) -> float:
    """Distance from point p to segment ab (plain floats for speed)."""
    abx, aby, apx, apy = b[0] - a[0], b[1] - a[1], p[0] - a[0], p[1] - a[1]
    den = abx * abx + aby * aby
    t = 0.0 if den == 0.0 else min(1.0, max(0.0, (apx * abx + apy * aby) / den))
    return math.hypot(apx - t * abx, apy - t * aby)


def _cross(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _seg_seg(a, b, c, d) -> float:
    d1, d2, d3, d4 = _cross(c, d, a), _cross(c, d, b), _cross(a, b, c), _cross(a, b, d)
    if d1 * d2 < 0 and d3 * d4 < 0:
        return 0.0
    return min(_pt_seg(a, c, d), _pt_seg(b, c, d), _pt_seg(c, a, b), _pt_seg(d, a, b))


def _edges(core):
    n = len(core)
    return [(core[0], core[-1])] if n <= 2 else [(core[i], core[(i + 1) % n]) for i in range(n)]


def _inside(p, poly) -> bool:
    n = len(poly)
    return n >= 3 and all(_cross(poly[i], poly[(i + 1) % n], p) >= 0 for i in range(n))


def core_distance(a, b) -> float:
    """Euclidean distance between two convex cores (0 if they touch or overlap)."""
    a, b = np.asarray(a).tolist(), np.asarray(b).tolist()
    if _inside(a[0], b) or _inside(b[0], a):
        return 0.0
    return min(_seg_seg(p, q, r, s) for p, q in _edges(a) for r, s in _edges(b))


def points_to_core(points: np.ndarray, core: np.ndarray) -> np.ndarray:
    """Vectorised distance from many points (N,2) to one convex core."""
    best = np.full(len(points), np.inf)
    for a, b in _edges(core):
        ab, ap = b - a, points - a
        den = ab @ ab
        t = np.clip((ap @ ab) / den, 0, 1) if den > 0 else np.zeros(len(points))
        best = np.minimum(best, np.hypot(*(ap - t[:, None] * ab).T))
    if len(core) >= 3:
        nxt = np.roll(core, -1, 0)
        cross = ((nxt - core)[None, :, 0] * (points[:, None, 1] - core[None, :, 1])
                 - (nxt - core)[None, :, 1] * (points[:, None, 0] - core[None, :, 0]))
        best[np.all(cross >= 0, 1)] = 0.0
    return best


# ------------------------------------------------------------------ clearance
@dataclass(frozen=True)
class World:
    """Static collision context: resting discs and the support rectangle [0,W]x[0,H]."""
    ids: tuple
    xy: np.ndarray
    radius: np.ndarray
    support: tuple


def make_world(scene, exclude=(), extra=()) -> World:
    """Discs of every scene object not in `exclude`, plus extra (id, xy, r) discs."""
    discs = [(o.id, o.xy_m, o.radius_m) for o in scene.objects if o.id not in exclude] + list(extra)
    lat = scene.lattice
    return World(tuple(d[0] for d in discs), np.array([d[1] for d in discs], float).reshape(-1, 2),
                 np.array([d[2] for d in discs], float), (lat.cols * lat.pitch_m, lat.rows * lat.pitch_m))


@dataclass(frozen=True)
class Verdict:
    status: str
    s: float | None = None        # path parameter of the violation (or unresolved interval start)
    s_end: float | None = None
    pair: str | None = None
    distance: float | None = None
    evals: int = 0


class Checker:
    """Every tested pair for one motion context: fixed world, optional held disc and
    the object fingers may touch. A pair is clear when its distance is >= clear_thr
    and definitely violated when < viol_thr; the band between is unresolved."""

    EXACT_BELOW_M = 0.05  # self pairs farther than this (bounding circles) report a lower bound

    def __init__(self, robot: Robot, world: World, delta: float, eps: float, held=None, contact=None,
                 self_pairs: bool = True):
        """self_pairs=False keeps only pairs with a world disc (the per-object oracle
        of interventions); robot-robot pairs and support containment are then the
        frozen action's own, already certified, business."""
        self.robot, self.world, self.held, self.delta, self.eps = robot, world, held, delta, eps
        self.n_evals = 0                       # distance evaluations (verifier calls) so far
        self.names = list(ROBOT_BODIES) + (["held"] if held else [])
        idx = {b: i for i, b in enumerate(self.names)}
        rows = [(b, oid, b in FINGERS and oid == contact) for b in self.names for oid in world.ids]
        self.pairs = [(a, b) for i, a in enumerate(self.names) for b in self.names[i + 1:]
                      if self_pairs and frozenset((a, b)) not in robot.excluded]
        rows += [(a, b, "held" in (a, b) and bool({a, b} & set(FINGERS))) for a, b in self.pairs]
        self.has_support = bool(held) and self_pairs
        if self.has_support:
            rows.append(("held", "support", None))
        self.labels = [f"{a}|{b}" for a, b, _ in rows]
        margin = [0.0 if c is None else (None if c else delta) for _, _, c in rows]
        self.clear_thr = np.array([-eps if m is None else m + eps for m in margin])
        self.viol_thr = np.array([-eps if m is None else m - eps for m in margin])
        self.owner_a = np.array([idx[a] for a, _, _ in rows])
        self.owner_b = np.array([idx.get(b, -1) for _, b, _ in rows])
        self.rho = self._rho()
        # self pair (a, b) with frames fa <= fb: (first self row, fa, fb, deeper body, fingers in pair)
        self.n_world = len(self.names) * len(world.ids)
        self.self_terms = [(min(FRAME[a], FRAME[b]), max(FRAME[a], FRAME[b]),
                            idx[a] if FRAME[a] > FRAME[b] else idx[b], (a in FINGERS) + (b in FINGERS))
                           for a, b in self.pairs]

    def _rho(self) -> np.ndarray:
        """rho[b, k]: max distance from joint k to points of body b (any q), so a
        joint-linear step moves every point of b by at most sum_k |dq_k| rho[b, k]."""
        joints, _ = fk(self.robot, (0.0, 0.0, 0.0))
        rho = np.zeros((len(self.names), 3))
        for i, (name, (core, r)) in enumerate(bodies(self.robot, (0, 0, 0), self.robot.open_gap,
                                                    self.held).items()):
            f = FRAME[name]
            if f:
                e = np.max(np.hypot(*(core - joints[f - 1]).T)) + r
                rho[i, :f] = [sum(self.robot.lengths[k:f - 1]) + e for k in range(f)]
        return rho

    def distances(self, q, gap: float) -> np.ndarray:
        """Signed distance (or a conservative lower bound far from contact) per entry."""
        self.n_evals += 1
        b = bodies(self.robot, q, gap, self.held)
        w = self.world
        out = [points_to_core(w.xy, b[n][0]) - w.radius - b[n][1] for n in self.names]
        circle = {}  # bounding circle (cx, cy, R) per body, R including the radius
        for n, (core, r) in b.items():
            c = core.mean(0)
            circle[n] = (c[0], c[1], float(np.max(np.hypot(*(core - c).T))) + r)
        self_d = []
        for a, c in self.pairs:
            (ax, ay, ar), (cx, cy, cr) = circle[a], circle[c]
            bound = math.hypot(ax - cx, ay - cy) - ar - cr
            self_d.append(bound if bound > self.EXACT_BELOW_M
                          else core_distance(b[a][0], b[c][0]) - b[a][1] - b[c][1])
        out.append(self_d)
        if self.has_support:
            (x, y), r = b["held"][0][0], b["held"][1]
            out.append([min(x - r, w.support[0] - x - r, y - r, w.support[1] - y - r)])
        return np.concatenate(out)

    def lipschitz(self, dq, dgap: float) -> np.ndarray:
        """Per-entry bound on |d distance / ds| along q(s) = qa + s dq, gap(s) = ga + s dgap.

        Against a static disc or the support: the body's point-speed bound. Between two
        robot bodies: a rigid motion shared by both leaves their distance unchanged, so
        only the joints between their frames (fa..fb-1) and finger travel count."""
        adq = np.abs(dq)
        speed = self.rho @ adq + np.array([abs(dgap) / 2 if n in FINGERS else 0.0 for n in self.names])
        speed = np.append(speed, 0.0)  # owner index -1 (a world disc or the support) is static
        out = speed[self.owner_a] + speed[self.owner_b]
        for k, (fa, fb, deep, fingers) in enumerate(self.self_terms):
            out[self.n_world + k] = self.rho[deep, fa:fb] @ adq[fa:fb] + fingers * abs(dgap) / 2
        return out

    def status_at(self, q, gap: float) -> Verdict:
        d = self.distances(q, gap)
        return self._classify(d, None, 1)

    def _classify(self, d, s, evals) -> Verdict:
        bad = np.flatnonzero(d < self.viol_thr)
        if bad.size:
            k = bad[np.argmin(d[bad] - self.viol_thr[bad])]
            return Verdict(COLLISION, s, s, self.labels[k], float(d[k]), evals)
        gray = np.flatnonzero(d < self.clear_thr)
        if gray.size:
            return Verdict(UNRESOLVED, s, s, self.labels[gray[0]], float(d[gray[0]]), evals)
        return Verdict(CLEAR, evals=evals)


def certify(checker: Checker, qa, qb, ga: float, gb: float, max_evals: int, ends=None) -> Verdict:
    """Conservative continuous check of the joint-linear segment (qa, ga) -> (qb, gb).

    On [s0, s1] each distance obeys d(s) >= (d(s0) + d(s1) - L (s1 - s0)) / 2 for its
    Lipschitz bound L (this dominates the spec's min(d0, d1) - L ds / 2). Intervals are
    bisected left-first until every bound clears, a sample definitely violates
    (COLLISION, first detected pair), or the evaluation budget ends (UNRESOLVED).
    `ends` may pass already computed endpoint distances (d(qa), d(qb)).
    """
    if min(ga, gb) < 0 or max(ga, gb) > checker.robot.open_gap + 1e-12:
        raise ValueError("finger gap outside [0, open_gap]: the finger speed bound would not hold")
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    dq, dg = qb - qa, gb - ga
    lip = checker.lipschitz(dq, dg)
    at = lambda s: checker.distances(qa + s * dq, ga + s * dg)  # noqa: E731
    d0, d1, evals = (at(0.0), at(1.0), 2) if ends is None else (ends[0], ends[1], 0)
    for s, d in ((0.0, d0), (1.0, d1)):
        v = checker._classify(d, s, evals)
        if v.status == COLLISION:
            return v
    stack = [(0.0, 1.0, d0, d1)]
    while stack:
        s0, s1, da, db = stack.pop()
        if np.all((da + db - lip * (s1 - s0)) / 2 >= checker.clear_thr):
            continue
        if evals >= max_evals:
            v = checker._classify(np.minimum(da, db), s0, evals)
            return Verdict(UNRESOLVED, s0, s1, v.pair, v.distance, evals)
        sm = (s0 + s1) / 2
        dm = at(sm)
        evals += 1
        v = checker._classify(dm, sm, evals)
        if v.status == COLLISION:
            return v
        stack += [(sm, s1, dm, db), (s0, sm, da, dm)]
    return Verdict(CLEAR, evals=evals)
