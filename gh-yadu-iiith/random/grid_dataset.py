"""
3-D version of the "arm among obstacles on a grid" dataset (MuJoCo, xArm7, 8 x 8 grid in a cabinet).

Scene
-----
  * one cabinet cell whose shelf carries an 8 x 8 grid; cell id = (row-1) * 8 + col  (row 1 = front,
    nearest the robot; col 1 = -y).  Rows / columns run 1..8, ids 1..64: cell 1 is front-left, 8 front-right,
    57 back-left, 64 back-right.  Everything you type or write in OBSTACLES / JSON uses this 1-based numbering.
  * ONE movable target (default: green cylinder) and N STATIC obstacles.  Obstacles are never moved.
    You choose N and each obstacle's shape / size / colour / cell (or let them be random).
  * Task = "move the target from cell A to cell B", then return the arm home (like the 2-D dataset).

Motion
------
  every travel motion is planned with RRT-Connect, outside AND inside the cabinet:
    leg 1  home -> in front of the target (enters the cabinet)               RRT-Connect
    leg 2  carry the target to in front of the goal cell (inside the cabinet) RRT-Connect, constrained:
           the hand stays level (projection + constraint), so the object stays upright
    leg 3  goal -> home                                                       RRT-Connect
  only the last ~4 cm of grasping / releasing (slide the fingers around the object, lift 2 cm, lower)
  are straight Cartesian moves -- they have to be, the fingers must go straight onto the object.

Feasibility ("without moving obstacles")
----------------------------------------
  A task is declared INFEASIBLE only after at least 3 independent attempts (--attempts, minimum 3).
  Each attempt re-runs IK from a different seed posture (every IK run may use up to --ik-steps = 10000
  iterations), tries every grasp candidate (hand pitch x grasp height x grasp offset), checks the full
  pick / carry / place / retreat motion against the obstacles and, if a candidate passes, runs
  RRT-Connect for the three legs.  One complete collision-free solution => FEASIBLE; none in all attempts
  => INFEASIBLE (with the reason and the obstacles that were in the way).

Usage
-----
    python grid_dataset.py --start 1 --goal 64 --n-obstacles 5 --seed 1
    python grid_dataset.py --tasks "1>8,8>64,64>57,57>1" --n-obstacles 5 --seed 2 --out data/ --plot data/overview.png
    python grid_dataset.py --start 3,4 --goal 6,1 --obstacles-json my_obstacles.json   # your own obstacles
    python grid_dataset.py            # asks for start / goal / number of obstacles when run in a terminal
  options: --shapes box,cylinder,sphere,capsule  --size-range 0.008 0.0145  --height-range 0.02 0.10
           --colors red,blue,...  --target-shape --target-size --target-height --target-color
           --attempts 3  --ik-steps 10000  --rrt-iters 4000  --rrt-time 10  --plan-only  --headless

my_obstacles.json = a list, e.g.
    [{"shape": "box", "size": 0.012, "height": 0.08, "color": "red", "cell": 21},
     {"shape": "cylinder", "size": 0.010, "height": 0.05, "color": [0.2, 0.4, 0.9, 1], "cell": [3, 4], "dx": 0.002}]
  size: box = half-size or [hx, hy, hz]; cylinder / capsule = radius; sphere = radius.  height = total height.
Python: see build_env(), GridPlanner.solve(), GridPlanner.execute().
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco

import xarm7_cabinet_env as E
from rrt_connect import RRTConnect
from rrt_star import shortcut, smooth, time_parameterize, path_length

GRID8 = dict(rows=8, cols=8, pitch_dy=0.05, pitch_depth=0.03, depth0=0.015)
SHAPES = ("box", "cylinder", "sphere", "capsule")
CABINET_W = 0.60            # inner width: with 0.50 the side walls block the hand in the outer columns
PALETTE = dict(E.COLORS, gray="0.5 0.5 0.5 1", cyan="0.1 0.75 0.8 1", pink="0.95 0.5 0.7 1",
               brown="0.5 0.3 0.15 1", white="0.95 0.95 0.95 1", black="0.1 0.1 0.1 1")
LIFT_UP = 0.02               # lift after the grasp / start of the lowering before the release (m)
MARGIN = 0.004               # clearance kept to obstacles on every planned motion (m)
CARRY_MARGIN = 0.015        # m: clearance to obstacles while carrying low (the held target shifts a little in the grip)
CARRY_Z_SLACK = 0.01         # m: the carried target may rise this far above its lifted height (never over obstacles)
CARRY_TOL = 0.2              # rad: the carried hand may deviate this far from the grasp orientation


# ----------------------------------------------------------------------------- cells
def cell_rc(cell, grid=GRID8):
    """cell id, 'r,c' string, or (row, col) -> (row, col)."""
    if isinstance(cell, str):
        cell = [int(v) for v in cell.split(",")] if "," in cell else int(cell)
    if isinstance(cell, (tuple, list, np.ndarray)):
        r, c = int(cell[0]), int(cell[1])
    else:
        r, c = divmod(int(cell), grid["cols"])
    if not (0 <= r < grid["rows"] and 0 <= c < grid["cols"]):
        raise ValueError(f"cell {cell} is outside the {grid['rows']}x{grid['cols']} grid")
    return r, c


def user_rc(cell, grid=GRID8):
    """Like cell_rc, but for what YOU type: rows / columns run 1..grid size and ids 1..rows*cols."""
    if isinstance(cell, str):
        cell = [int(v) for v in cell.split(",")] if "," in cell else int(cell)
    if isinstance(cell, (tuple, list, np.ndarray)):
        r, c = int(cell[0]), int(cell[1])
    else:
        r, c = divmod(int(cell) - 1, grid["cols"])
        r, c = r + 1, c + 1
    if not (1 <= r <= grid["rows"] and 1 <= c <= grid["cols"]):
        raise ValueError(f"cell {cell} is outside the grid: rows and columns run 1..{grid['rows']} / "
                         f"1..{grid['cols']}, ids 1..{grid['rows'] * grid['cols']}")
    return r - 1, c - 1


def cell_id(rc, grid=GRID8):
    return rc[0] * grid["cols"] + rc[1]


def cell_xy(rc, grid=GRID8):
    depth, dy = E.grid_cell_center(rc, grid)
    return E.CAB_X0 + depth, dy


# ----------------------------------------------------------------------------- obstacles (mutable)
# YOUR OBSTACLES -- edit this list.  Used unless --obstacles-json is given (--n-obstacles is then ignored).
# Keys: shape (box|cylinder|sphere|capsule), size (box: half-size or [hx, hy, hz]; others: radius),
#       height (total, not for sphere), color (name or [r, g, b, a]), cell (1-64 = (row-1)*8+col, or [row, col] with 1..8),
#       optional dx, dy (offset from the cell centre).  Set OBSTACLES = [] to go back to random obstacles.
OBSTACLES = [
    dict(shape="box",      size=0.012, height=0.08, color="red",    cell=[2, 1]),
    dict(shape="cylinder", size=0.010, height=0.05, color="blue",   cell=[3, 4]),
    dict(shape="sphere",   size=0.012,              color="orange", cell=[4, 6]),
]


def rgba(color):
    if isinstance(color, str):
        return PALETTE.get(color, color)
    c = list(color) + [1.0] * (4 - len(color))
    return " ".join(str(float(v)) for v in c)


def obstacle_geometry(shape, size, height=None):
    """-> (MuJoCo size list, bounding half-extents, half height of the whole shape)."""
    s = np.atleast_1d(np.asarray(size, float))
    if shape == "box":
        s = np.repeat(s, 3) if s.size == 1 else s
        if s.size != 3:
            raise ValueError("box size = half-size or [hx, hy, hz]")
        s = s.copy()
        if height is not None:
            s[2] = height / 2
        return list(s), tuple(s), s[2]
    if shape in ("cylinder", "capsule"):
        r = s[0]
        hh = s[1] if s.size > 1 else r
        if height is not None:
            hh = height / 2 if shape == "cylinder" else max(height / 2 - r, 0.0)
        top = hh if shape == "cylinder" else hh + r
        return [r, hh], (r, r, top), top
    if shape == "sphere":
        return [s[0]], (s[0],) * 3, s[0]
    raise ValueError(f"unknown shape {shape!r}; choose from {SHAPES}")


def make_obstacle(k, shape, cell, size, color="orange", dx=0.0, dy=0.0, height=None, grid=GRID8,
                  allow_overflow=False, name=None):
    """A static obstacle standing on the shelf in grid `cell`."""
    msize, bbox, hz = obstacle_geometry(shape, size, height)
    rc = cell_rc(cell, grid)
    x, y = cell_xy(rc, grid)
    if not allow_overflow and (abs(dx) + bbox[0] > grid["pitch_depth"] / 2 + 1e-9 or
                               abs(dy) + bbox[1] > grid["pitch_dy"] / 2 + 1e-9):
        raise ValueError(f"obstacle {k} ({shape}) does not fit in a {grid['pitch_depth']*100:.0f} x "
                         f"{grid['pitch_dy']*100:.0f} cm grid cell (footprint half-extents "
                         f"{bbox[0]*100:.1f} x {bbox[1]*100:.1f} cm); make it smaller or pass allow_overflow=True")
    return dict(name=name or f"obstacle{k}", shape=shape, size=msize, pos=[x + dx, y + dy, E.SHELF_Z + hz],
                bbox=[float(b) for b in bbox], rgba=rgba(color), cell=rc, height=float(2 * hz),
                color=color if isinstance(color, str) else list(color))


def random_obstacles(n, grid=GRID8, exclude=(), seed=0, shapes=SHAPES, size_range=(0.008, 0.0145),
                     height_range=(0.02, 0.10), colors=("red", "blue", "yellow", "orange", "purple", "cyan", "pink")):
    """n obstacles in distinct random cells (never in `exclude`), random shape / size / height / colour."""
    rng = np.random.default_rng(seed)
    banned = {cell_rc(c, grid) for c in exclude}   # exclude: internal 0-based cells
    free = [(r, c) for r in range(grid["rows"]) for c in range(grid["cols"]) if (r, c) not in banned]
    if n > len(free):
        raise ValueError(f"{n} obstacles do not fit: only {len(free)} free cells")
    cap = min(grid["pitch_depth"], grid["pitch_dy"]) / 2 - 0.0005
    obs = []
    for k, i in enumerate(rng.choice(len(free), n, replace=False)):
        shape = str(rng.choice(shapes))
        r = float(min(rng.uniform(*size_range), cap))
        h = float(rng.uniform(*height_range))
        obs.append(make_obstacle(k, shape, free[i], [r, r, r] if shape == "box" else r,
                                 str(rng.choice(colors)), height=None if shape == "sphere" else h, grid=grid))
    return obs


def obstacles_from_json(path, grid=GRID8):
    with open(path) as f:
        return obstacles_from_items(json.load(f), grid)


def obstacles_from_items(items, grid=GRID8):
    return [make_obstacle(k, it["shape"], user_rc(it["cell"], grid), it["size"], it.get("color", "orange"), it.get("dx", 0.0),
                          it.get("dy", 0.0), it.get("height"), grid, it.get("allow_overflow", False), it.get("name"))
            for k, it in enumerate(items)]


# ----------------------------------------------------------------------------- scene
def build_env(start_cell, obstacles, target=None, grid=GRID8, seed=0, verbose=False):
    """Environment with the target in `start_cell` and the given static obstacles."""
    t = dict(name="target", label="T", shape="cylinder", color="green", size=0.012, height=0.06)
    t.update(target or {})
    t["grid"] = cell_rc(start_cell, grid)
    return E.XArm7CabinetEnv(n_cells=1, cell_w=CABINET_W, grid=grid, objects=[t], obstacles=obstacles,
                             planner="rrt_connect", seed=seed, verbose=verbose)


class Fail(Exception):
    def __init__(self, stage):
        super().__init__(stage)
        self.stage = stage


# ----------------------------------------------------------------------------- planner
class GridPlanner:
    def __init__(self, env, attempts=3, ik_steps=10000, rrt_iters=4000, rrt_time=10.0, seed=0,
                 time_limit=300.0, verbose=True):
        self.env, self.ck, self.grid = env, env.checker, env.grid
        self.attempts, self.ik_steps = max(3, int(attempts)), int(ik_steps)
        self.rrt_iters, self.rrt_time, self.seed = int(rrt_iters), float(rrt_time), seed
        self.time_limit, self.verbose = time_limit, verbose
        self.obst = [ob["name"] for ob in env.obstacles]
        self.obst_ids = [env.plan_model.geom(n).id for n in self.obst]
        self.ignore_obst = False          # True: analysis 'as if the cabinet were empty'
        self.q_ready = env.data.qpos[env.qadr].copy()
        self.dt = env.model.opt.timestep

    # -- collision-model state (the planning model, not the simulation)
    def _prep(self, margin, width, held=None, target_at=None, constraint=None):
        env, ck = self.env, self.ck
        ck.margin = margin
        ck.sync(env.data.qpos)
        if target_at is not None:                       # target already put down at the goal
            a = env.obj_qadr[0]
            ck.d.qpos[a:a + 3], ck.d.qpos[a + 3:a + 7] = target_at, [1, 0, 0, 0]
        ck.detach() if held is None else ck.attach_relative([held])
        env._set_fingers(width)
        ck.constraint = constraint

    def _path_ok(self, path, ignore=(), blockers=None):
        with self.ck.ignoring(list(ignore) + (self.obst_ids if self.ignore_obst else [])):
            for q in path:
                if blockers is None:
                    if not self.ck.is_free(q):
                        return False
                else:
                    hits = self.ck.collisions(q)
                    if hits:
                        self.env._tally(hits, blockers)
                        return False
        return True

    # -- IK
    def _ik(self, p, R, q0, iters):
        env = self.env
        q = env.ik(p, R, q0=np.clip(q0, env.jrange[:, 0], env.jrange[:, 1]), iters=iters, tol=1e-6)
        ep, er = env._pose_error(q, p, R)
        return q, bool(ep < 1e-3 and er < 1e-2)

    def _chain(self, q0, p0, pts, R):
        try:
            return self.env._ik_chain(q0, p0, pts, R, step=0.01)
        except RuntimeError:
            raise Fail("out of reach") from None

    def _fk(self, q):
        d = self.env.ik_data
        d.qpos[self.env.qadr] = q
        mujoco.mj_kinematics(self.env.model, d)
        return d.site_xpos[self.env.site_id].copy()

    # -- constrained planning helpers (hand stays level while carrying)
    def _constraint(self, R, tol=CARRY_TOL, z_max=None):
        """Hand stays level and (with z_max) the carried target stays low: never lifted over an obstacle."""
        site, cos_tol = self.ck.site_id, np.cos(tol)
        return lambda d: ((np.trace(d.site_xmat[site].reshape(3, 3).T @ R) - 1) / 2 >= cos_tol
                          and (z_max is None or d.site_xpos[site][2] <= z_max))

    def _project(self, q, R):
        env = self.env
        p = self._fk(q)
        q2 = env.ik(p, R, q0=q, iters=40, tol=1e-4)
        ep, er = env._pose_error(q2, p, R)
        return q2 if (ep < 2e-3 and er < 2e-2) else None

    def _sample(self, R, rng, z_max=None):
        env = self.env
        z_hi = E.SHELF_Z + 0.28 if z_max is None else max(z_max, E.SHELF_Z + 0.03)
        p = np.array([rng.uniform(E.CAB_X0 - 0.25, E.CAB_X0 + E.CAB_DEPTH), rng.uniform(-0.30, 0.30),
                      rng.uniform(E.SHELF_Z + 0.02, z_hi)])
        q = env.ik(p, R, q0=env._seed(p), iters=60, tol=1e-4)
        ep, er = env._pose_error(q, p, R)
        return q if (ep < 2e-3 and er < 2e-2) else None

    def _rrt_leg(self, name, qa, qb, width, seed, held=None, target_at=None, R=None, z_max=None):
        """RRT-Connect from qa to qb (+ shortcut, smoothing, time scaling).  With R: constrained
        (level hand, and the hand / target below height z_max if given)."""
        env, ck = self.env, self.ck
        self._prep(MARGIN if z_max is None else CARRY_MARGIN, width, held, target_at,
                   self._constraint(R, z_max=z_max) if R is not None else None)
        edge = lambda a, b: ck.edge_free(a, b, res=0.03)
        rng = np.random.default_rng(seed)
        kw = {}
        if R is not None:
            kw = dict(project=lambda q: self._project(q, R), sampler=lambda: self._sample(R, rng, z_max))
        planner = RRTConnect(ck.is_free, edge, env.jrange[:, 0], env.jrange[:, 1],
                             step=0.25 if R is not None else 0.3, max_iters=self.rrt_iters,
                             max_time=self.rrt_time, seed=seed, **kw)
        try:
            raw = planner.plan(qa, qb)
        except RuntimeError as err:
            raise Fail(f"{name}: {err}") from None
        if raw is None:
            raise Fail(f"{name}: RRT-Connect found no path in {planner.stats['iters']} iterations "
                       f"({planner.stats['time']:.0f}s)")
        short = shortcut(raw, edge, seed=seed)
        smoothed = smooth(short, ck)
        vmax, amax = (0.6, 1.0) if R is not None else (E.PLAN_VMAX, E.PLAN_AMAX)
        traj = time_parameterize(smoothed, vmax, amax, self.dt)
        return dict(name=name, traj=traj, event=None, iters=planner.stats["iters"], nodes=planner.stats["nodes"],
                    plan_time=planner.stats["time"], raw_len=path_length(raw), short_len=path_length(short))

    # -- one grasp candidate: IK + every straight piece checked against the obstacles
    def _candidates(self):
        o = self.env.objects[0]
        zs = list(np.arange(E.PAD_HALF_H, o["height"] - E.PAD_HALF_H + 1e-9, 0.004)) or [o["height"] / 2]
        zs.sort(key=lambda z: abs(z - o["height"] / 2))
        return [(pitch, z, off) for pitch in E.PITCHES_DEG for off in E.GRASP_OFFSETS for z in zs]

    def _build(self, cand, q_g0, q_p0, goal_rc, blockers):
        env = self.env
        pitch, z, off = cand
        R = E.R_pitch(pitch)
        o, g = env.objects[0], env.obj_geom[0]
        p0 = env.obj_pos(0)
        gp = np.array([p0[0] - off, p0[1], E.SHELF_Z + z])
        rel = p0 - gp                                                   # object centre relative to the site
        gx, gy = cell_xy(goal_rc, self.grid)
        goal_c = np.array([gx, gy, E.SHELF_Z + o["height"] / 2 + 0.004])
        pp = goal_c - rel
        back, up = np.array([o["size"] + E.APPROACH_DIST, 0, 0]), np.array([0, 0, LIFT_UP])
        pre, pre_p = gp - back, pp - back
        w_open, w_shut = env._finger_widths(0)

        q_g, ok = self._ik(gp, R, q_g0, 300)
        if not ok:
            raise Fail("grasp pose unreachable")
        q_p, ok = self._ik(pp, R, q_g, 300)
        if not ok:
            q_p, ok = self._ik(pp, R, q_p0, 300)
        if not ok:
            raise Fail("place pose unreachable")

        slide_in = self._chain(q_g, gp, [pre], R)[::-1]                 # pre -> gp
        lift = self._chain(q_g, gp, [gp + up], R)                       # gp -> gp + up
        retreat = self._chain(lift[-1], gp + up, [pre + up], R)         # gp + up -> pre + up
        c1 = self._chain(q_p, pp, [pp + up], R)                         # pp -> pp + up      (reversed: lowering)
        c2 = self._chain(c1[-1], pp + up, [pre_p + up], R)              # pp + up -> pre_p + up
        put_out = self._chain(q_p, pp, [pre_p], R)                      # pp -> pre_p (after the release)
        held = (g, R.T @ rel, R.T @ env.data.geom_xmat[g].reshape(3, 3))

        self._prep(MARGIN, w_open)                                      # fingers pre-shaped around the target
        if not self._path_ok(slide_in, [g], blockers):
            raise Fail("approach to the target is blocked")
        self._prep(MARGIN, w_shut)
        if not self._path_ok([q_g], [g], blockers):
            raise Fail("grasp pose is blocked")
        self._prep(MARGIN, w_shut, held=held)
        if not self._path_ok(lift, [g], blockers):
            raise Fail("lift is blocked")
        if not self._path_ok(retreat, (), blockers):
            raise Fail("backing out with the target is blocked")
        if not self._path_ok(c2, (), blockers):
            raise Fail("slide-in to the goal cell is blocked")
        if not self._path_ok(c1, [g], blockers):
            raise Fail("lowering at the goal cell is blocked")
        self._prep(MARGIN, w_open, target_at=goal_c)
        if not self._path_ok(put_out, (), blockers):
            raise Fail("backing out of the goal cell is blocked")

        mk = lambda name, path, vmax=E.CELL_VMAX, amax=E.CELL_AMAX, event=None: dict(
            name=name, traj=time_parameterize(path, vmax, amax, self.dt), event=event)
        return dict(R=R, pitch=pitch, z=z, off=off, held=held, goal_c=goal_c, q_pre=slide_in[0], q_A=retreat[-1],
                    q_B=c2[-1], q_after=put_out[-1], w_open=w_open, w_shut=w_shut,
                    segs=dict(slide_in=mk("slide_in", slide_in, event="close"), lift=mk("lift", lift),
                              retreat=mk("retreat", retreat), put_slide=mk("put_slide", c2[::-1]),
                              put_down=mk("put_down", c1[::-1], event="open"), put_out=mk("put_out", put_out)))

    # -- the analysis: FEASIBLE / INFEASIBLE
    def solve(self, start, goal):
        env, t0 = self.env, time.time()
        s_rc, g_rc = cell_rc(start, self.grid), cell_rc(goal, self.grid)
        res = dict(start=cell_id(s_rc, self.grid), goal=cell_id(g_rc, self.grid), feasible=False, reason="",
                   attempts=0, ik_runs=0, ik_limit_hits=0, candidates=0, rrt_runs=0, blocked_by=[], plan=None,
                   stages={}, time=0.0)
        taken = {ob["cell"]: ob["name"] for ob in env.obstacles if "cell" in ob}
        for tag, rc in (("start", s_rc), ("goal", g_rc)):
            if rc in taken:
                res["reason"] = f"the {tag} cell {cell_id(rc, self.grid) + 1} is occupied by {taken[rc]} (obstacles are not moved)"
                res["blocked_by"] = [taken[rc]]
                return res
        if s_rc == g_rc:
            res["reason"] = "start and goal are the same cell"
            return res

        p0, o = env.obj_pos(0), env.objects[0]
        gx, gy = cell_xy(g_rc, self.grid)
        z_mid = E.SHELF_Z + o["height"] / 2
        gp_base = np.array([p0[0], p0[1], z_mid])
        pp_base = np.array([gx, gy, z_mid])
        blockers, stage_count = {}, {}
        cands = self._candidates()
        for attempt in range(self.attempts):
            res["attempts"] = attempt + 1
            if time.time() - t0 > self.time_limit:
                res["reason"] = f"time limit ({self.time_limit:.0f}s) reached before a verdict"
                return res
            rng = np.random.default_rng(self.seed * 1000 + attempt)
            seeds = self._reach_seeds(gp_base, pp_base, attempt, rng, res)   # full IK step budget
            if len(seeds) < 2:
                stage_count["target position unreachable (IK)"] = stage_count.get("target position unreachable (IK)", 0) + 1
                continue
            rrt_fails = 0
            for cand in cands:
                res["candidates"] += 1
                try:
                    plan = self._build(cand, seeds["grasp"], seeds["place"], g_rc, blockers)
                except Fail as f:
                    stage_count[f.stage] = stage_count.get(f.stage, 0) + 1
                    continue
                try:
                    legs = self._legs(plan, attempt)
                except Fail as f:
                    res["rrt_runs"] += 1
                    stage_count[f.stage] = stage_count.get(f.stage, 0) + 1
                    rrt_fails += 1
                    if rrt_fails >= 3 or time.time() - t0 > self.time_limit:
                        break
                    continue
                res["rrt_runs"] += 3
                res.update(feasible=True, plan=self._assemble(plan, legs), time=time.time() - t0,
                           grasp=dict(pitch=cand[0], z=cand[1], offset=cand[2]),
                           reason=f"collision-free solution found in attempt {attempt + 1} of {self.attempts}")
                return res
        res["stages"] = stage_count
        self.ignore_obst = True                          # same analysis with the obstacles taken away:
        try:                                             # is it the obstacles, or the cabinet / arm?
            seeds = self._reach_seeds(gp_base, pp_base, 0, np.random.default_rng(self.seed), dict(ik_runs=0, ik_limit_hits=0))
            res["obstacle_caused"] = False
            for cand in (cands if len(seeds) == 2 else []):
                try:
                    self._build(cand, seeds["grasp"], seeds["place"], g_rc, {})
                    res["obstacle_caused"] = True
                    break
                except Fail:
                    continue
        finally:
            self.ignore_obst = False
        res["cabinet_parts"] = [k for k in blockers if k in ("shelf", "ceiling", "back", "div0", "div1")]
        names = [k for k in blockers if k in self.obst]
        res["blocked_by"] = sorted(names, key=lambda k: -blockers[k])
        top = sorted(stage_count.items(), key=lambda kv: -kv[1])[:2]
        res["reason"] = "; ".join(f"{k} ({v}x)" for k, v in top) or "no candidate"
        res["time"] = time.time() - t0
        return res

    def _reach_seeds(self, gp, pp, attempt, rng, res):
        """IK of the grasp and place position (level and fully tilted hand), each run with the full
        step budget and, from the 2nd attempt on, a perturbed start posture."""
        seeds = {}
        for tag, p in (("grasp", gp), ("place", pp)):
            for pitch in (0, E.PITCHES_DEG[-1]):
                q0 = self.env._seed(p) + (rng.normal(scale=0.3, size=7) if attempt else 0.0)
                q, ok = self._ik(p, E.R_pitch(pitch), q0, self.ik_steps)
                res["ik_runs"] += 1
                res["ik_limit_hits"] += int(not ok)
                if ok:
                    seeds.setdefault(tag, q)
        return seeds

    def _legs(self, plan, attempt):
        s = self.seed * 7919 + attempt * 31
        z_max = max(self._fk(plan["q_A"])[2], self._fk(plan["q_B"])[2]) + CARRY_Z_SLACK   # target stays low
        carry = self._rrt_leg("carry", plan["q_A"], plan["q_B"], plan["w_shut"], s + 1, held=plan["held"], R=plan["R"],
                              z_max=z_max)
        leg1 = self._rrt_leg("leg1_to_target", self.q_ready, plan["q_pre"], plan["w_open"], s + 2)
        leg3 = self._rrt_leg("leg3_return_home", plan["q_after"], self.q_ready, plan["w_open"], s + 3,
                             target_at=plan["goal_c"])
        return leg1, carry, leg3

    @staticmethod
    def _assemble(plan, legs):
        leg1, carry, leg3 = legs
        s = plan["segs"]
        return dict(order=[leg1, s["slide_in"], s["lift"], s["retreat"], carry, s["put_slide"], s["put_down"],
                           s["put_out"], leg3], legs=[leg1, carry, leg3], w_open=plan["w_open"])

    # -- run the plan in the simulator and verify it
    def execute(self, res):
        env = self.env
        d, m = env.data, env.model
        plan = res["plan"]
        obst_ids = {m.geom(n).id for n in self.obst}
        arm_b = {b for b in range(m.nbody) if m.body(b).name.startswith(("link", "xarm_gripper", "left_", "right_"))}
        arm_b.discard(m.body("link_base").id)
        tgt = env.obj_geom[0]
        touched, rec = set(), dict(q=[], ee=[], seg=[], grip=[])
        seg_idx, step0, grip_w = [0], env.step, [plan["w_open"]]

        def step(n=1):
            step0(n)
            for c in d.contact[:d.ncon]:
                a, b = c.geom1, c.geom2
                if (a in obst_ids and (m.geom_bodyid[b] in arm_b or b == tgt)) or \
                        (b in obst_ids and (m.geom_bodyid[a] in arm_b or a == tgt)):
                    touched.add(m.geom(a).name or m.geom(b).name)
            if d.time * 1e3 % 20 < 2:                                   # ~every 10 steps
                rec["q"].append(d.qpos[env.qadr].copy()); rec["ee"].append(d.site_xpos[env.site_id].copy())
                rec["seg"].append(seg_idx[0]); rec["grip"].append(grip_w[0])
        env.step = step
        try:
            env.set_gripper_width(plan["w_open"]); env.wait(150)
            for k, seg in enumerate(plan["order"]):
                seg_idx[0] = k
                env.execute(seg["traj"], settle=30)
                if seg["event"] == "close":
                    env.wait(100); env.set_gripper(False); env.wait(400); env.held = 0; grip_w[0] = 0.0
                elif seg["event"] == "open":
                    env.set_gripper_width(plan["w_open"]); env.wait(300); env.held = None; grip_w[0] = plan["w_open"]
        finally:
            env.step = step0
        goal_rc = cell_rc(res["goal"], self.grid)
        gx, gy = cell_xy(goal_rc, self.grid)
        p = env.obj_pos(0)
        err = float(np.hypot(p[0] - gx, p[1] - gy))
        z_axis = d.xmat[m.body("target").id].reshape(3, 3)[:, 2]
        tilt = float(np.degrees(np.arccos(np.clip(z_axis[2], -1, 1))))
        in_goal = env.cell_of(0) == goal_rc
        res["verified"] = bool(in_goal and err < 0.015 and tilt < 20 and not touched)
        res["exec"] = dict(error_cm=err * 100, tilt_deg=tilt, obstacle_contacts=sorted(touched), in_goal_cell=in_goal)
        res["record"] = {k: np.array(v) for k, v in rec.items()}
        return res

    def place_target(self, rc):
        """Reset the arm and put the target in grid cell rc (independent dataset items)."""
        env = self.env
        env.reset()
        x, y = cell_xy(rc, self.grid)
        a = env.obj_qadr[0]
        env.data.qpos[a:a + 3] = [x, y, E.SHELF_Z + env.objects[0]["height"] / 2 + 0.001]
        env.data.qpos[a + 3:a + 7] = [1, 0, 0, 0]
        env.data.qvel[:] = 0
        mujoco.mj_forward(env.model, env.data)
        env.step(200)
        self.q_ready = env.data.qpos[env.qadr].copy()


# ----------------------------------------------------------------------------- output
def describe(res):
    if res["feasible"]:
        g = res["grasp"]
        legs = ", ".join(f"{l['name']}: {l['iters']} it" for l in res["plan"]["legs"])
        return (f"FEASIBLE  ({res['reason']}; grasp pitch {g['pitch']} deg, {g['z']*100:.1f} cm above the shelf, "
                f"offset {g['offset']*100:.1f} cm; RRT-Connect {legs})")
    by = f"; obstacles in the way: {', '.join(res['blocked_by'])}" if res["blocked_by"] else ""
    if "obstacle_caused" in res:
        by += ("; the obstacles are what make it infeasible" if res["obstacle_caused"] else
               "; NOT the obstacles: unreachable even with an empty cabinet (arm / cabinet geometry)")
    return (f"INFEASIBLE without moving obstacles  ({res['reason']}{by})\n"
            f"      checked: {res['attempts']} attempts, {res['ik_runs']} full-budget IK runs "
            f"({res['ik_limit_hits']} used all steps without converging), {res['candidates']} grasp candidates, "
            f"{res['rrt_runs']} RRT-Connect runs")


def save_task(out_dir, k, tag, res, scene):
    os.makedirs(out_dir, exist_ok=True)
    meta = dict(tag=tag, start=res["start"], goal=res["goal"], feasible=res["feasible"], reason=res["reason"],
                verified=res.get("verified"), attempts=res["attempts"], ik_runs=res["ik_runs"], candidates=res["candidates"],
                blocked_by=res["blocked_by"], exec=res.get("exec"), scene=scene)
    with open(os.path.join(out_dir, f"{tag}.json"), "w") as f:
        json.dump(meta, f, indent=1, default=float)
    if res["feasible"]:
        arrays = dict(segments=np.array([s["name"] for s in res["plan"]["order"]]))
        for s in res["plan"]["order"]:
            arrays[f"traj_{s['name']}"] = s["traj"]
        if "record" in res:
            arrays.update({f"rec_{n}": v for n, v in res["record"].items()})
        np.savez_compressed(os.path.join(out_dir, f"{tag}.npz"), **arrays)


def plot_results(results, env, scene_name, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle
    n = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.2 * rows), squeeze=False)
    g = env.grid
    d_lo = E.CAB_X0 + g["depth0"] - g["pitch_depth"] / 2
    for ax, (tag, res) in zip(axes.ravel(), results):
        ax.set_title(f"{tag}: cell {res['start'] + 1} -> {res['goal'] + 1}", fontsize=8, loc="left")
        for r in range(g["rows"]):
            for c in range(g["cols"]):
                ax.add_patch(Rectangle((d_lo + r * g["pitch_depth"], (c - g["cols"] / 2) * g["pitch_dy"]),
                                       g["pitch_depth"], g["pitch_dy"], fill=False, ec="0.8", lw=0.4))
        for ob in env.obstacles:
            x, y, _ = ob["pos"]
            col = ob["color"] if isinstance(ob["color"], str) and ob["color"] in PALETTE else "tab:orange"
            if ob["shape"] == "box":
                ax.add_patch(Rectangle((x - ob["bbox"][0], y - ob["bbox"][1]), 2 * ob["bbox"][0], 2 * ob["bbox"][1],
                                       fc=col, ec="k", lw=0.5, alpha=0.8))
            else:
                ax.add_patch(Circle((x, y), ob["bbox"][0], fc=col, ec="k", lw=0.5, alpha=0.8))
        xs, ys = cell_xy(cell_rc(res["start"], g), g)
        xg, yg = cell_xy(cell_rc(res["goal"], g), g)
        ax.plot(xs, ys, "o", mfc="limegreen", mec="k", ms=8, zorder=5, label="start (target)")
        ax.plot(xg, yg, "*", mfc="gold", mec="k", ms=13, zorder=5, label="goal")
        if res["feasible"]:
            for s in res["plan"]["order"]:
                ee = np.array([env_fk(env, q) for q in s["traj"][::25]])
                ax.plot(ee[:, 0], ee[:, 1], color="tab:orange" if s["name"] in ("carry", "put_slide", "put_down", "retreat", "lift") else "tab:blue", lw=1)
        ax.text(0.98, 0.97, "FEASIBLE" if res["feasible"] else "INFEASIBLE", transform=ax.transAxes, ha="right", va="top",
                fontsize=7, bbox=dict(fc="w", ec="k"))
        if res.get("verified"):
            ax.text(0.02, 0.03, "VERIFIED", transform=ax.transAxes, color="w", fontsize=7, bbox=dict(fc="g", ec="g"))
        ax.legend(loc="lower right", fontsize=5, framealpha=0.9)
        ax.set_xlim(0.3, 1.0); ax.set_ylim(-0.35, 0.35); ax.set_aspect("equal"); ax.tick_params(labelsize=6)
        ax.set_xlabel("x [m]", fontsize=7); ax.set_ylabel("y [m]", fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(scene_name, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def env_fk(env, q):
    d = env.ik_data
    d.qpos[env.qadr] = q
    mujoco.mj_kinematics(env.model, d)
    return d.site_xpos[env.site_id].copy()


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("Usage")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start"), ap.add_argument("--goal"), ap.add_argument("--tasks", help='e.g. "0>7,7>63,63>56,56>0"')
    ap.add_argument("--n-obstacles", type=int), ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--obstacles-json"), ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--size-range", type=float, nargs=2, default=(0.008, 0.0145))
    ap.add_argument("--height-range", type=float, nargs=2, default=(0.02, 0.10))
    ap.add_argument("--colors", default="red,blue,yellow,orange,purple,cyan,pink")
    ap.add_argument("--target-shape", default="cylinder"), ap.add_argument("--target-size", type=float, default=0.012)
    ap.add_argument("--target-height", type=float, default=0.06), ap.add_argument("--target-color", default="green")
    ap.add_argument("--attempts", type=int, default=3), ap.add_argument("--ik-steps", type=int, default=10000)
    ap.add_argument("--rrt-iters", type=int, default=4000), ap.add_argument("--rrt-time", type=float, default=10.0)
    ap.add_argument("--time-limit", type=float, default=300.0)
    ap.add_argument("--plan-only", action="store_true"), ap.add_argument("--headless", action="store_true")
    ap.add_argument("--out"), ap.add_argument("--plot")
    a = ap.parse_args(argv)

    if a.tasks:
        tasks = [tuple(t.split(">")) for t in a.tasks.split(",")]
    elif a.start is not None and a.goal is not None:
        tasks = [(a.start, a.goal)]
    elif sys.stdin.isatty():
        print("cells are numbered (row-1)*8+col, rows / columns 1..8 (1 = front-left, 8 = front-right, 57 = back-left, 64 = back-right); "
              "'r,c' also works")
        tasks = [(input("target cell: ").strip(), input("goal cell:   ").strip())]
        if a.n_obstacles is None and not OBSTACLES:
            a.n_obstacles = int(input("number of obstacles [5]: ").strip() or 5)
    else:
        tasks = [("0", "7")]
    n_obs = 5 if a.n_obstacles is None else a.n_obstacles
    tasks = [(user_rc(s), user_rc(g)) for s, g in tasks]

    if a.obstacles_json:
        obstacles = obstacles_from_json(a.obstacles_json)
    elif OBSTACLES:
        if a.n_obstacles is not None:
            print(f"note: --n-obstacles {a.n_obstacles} ignored, using the {len(OBSTACLES)} obstacles in OBSTACLES "
                  f"(set OBSTACLES = [] for random ones)")
        obstacles = obstacles_from_items(OBSTACLES)
    else:
        obstacles = random_obstacles(n_obs, exclude=[c for t in tasks for c in t], seed=a.seed,
                                     shapes=a.shapes.split(","), size_range=a.size_range, height_range=a.height_range,
                                     colors=a.colors.split(","))
    target = dict(shape=a.target_shape, size=a.target_size, height=a.target_height, color=a.target_color)
    env = build_env(tasks[0][0], obstacles, target, seed=a.seed)
    planner = GridPlanner(env, a.attempts, a.ik_steps, a.rrt_iters, a.rrt_time, a.seed, a.time_limit)
    print(f"{len(obstacles)} static obstacles, target {a.target_shape} (size {a.target_size*100:.1f} cm, "
          f"height {a.target_height*100:.0f} cm); {len(tasks)} task(s)")
    for ob in obstacles:
        print(f"   {ob['name']:<10} {ob['shape']:<8} cell {cell_id(ob['cell']) + 1:>2} {(ob['cell'][0] + 1, ob['cell'][1] + 1)}  "
              f"{ob['height']*100:4.1f} cm tall, footprint {2*ob['bbox'][0]*100:.1f} x {2*ob['bbox'][1]*100:.1f} cm")
    scene = dict(grid=GRID8, target=target, obstacles=[{k: (list(v) if isinstance(v, tuple) else v) for k, v in ob.items()}
                                                       for ob in obstacles])
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        with open(os.path.join(a.out, "scene.json"), "w") as f:
            json.dump(scene, f, indent=1, default=float)
    if not a.headless and not a.plan_only:
        env.launch_viewer()

    results = []
    for k, (s, g) in enumerate(tasks):
        tag = f"t{k}"
        print(f"\n[{tag}] cell {cell_id(s) + 1} -> {cell_id(g) + 1}")
        planner.place_target(s)
        res = planner.solve(s, g)
        print("   " + describe(res))
        if res["feasible"] and not a.plan_only:
            planner.execute(res)
            ex = res["exec"]
            print(f"   executed in simulation: {'VERIFIED' if res['verified'] else 'NOT VERIFIED'} "
                  f"(target in cell {cell_id(env.cell_of(0), env.grid) + 1 if env.cell_of(0) else '?'}, error {ex['error_cm']:.1f} cm, "
                  f"tilt {ex['tilt_deg']:.0f} deg, obstacle contacts: {ex['obstacle_contacts'] or 'none'})")
        results.append((tag, res))
        if a.out:
            save_task(a.out, k, tag, res, scene)
    n_ok = sum(r["feasible"] for _, r in results)
    print(f"\nsummary: {n_ok} feasible, {len(results) - n_ok} infeasible of {len(results)} task(s)")
    if a.plot:
        plot_results(results, env, f"{len(obstacles)} obstacles, seed {a.seed}", a.plot)
        print("plot:", a.plot)
    if env.viewer is not None:
        while env.viewer.is_running():
            env.step(10)
    return results


if __name__ == "__main__":
    main()
