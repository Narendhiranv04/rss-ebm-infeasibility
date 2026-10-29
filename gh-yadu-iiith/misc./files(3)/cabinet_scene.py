#!/usr/bin/env python3
"""
cabinet_scene.py -- a configurable xArm7 + cabinet scene where EVERY transit is planned by
RRT-Connect, outside the cabinet and inside it.

Put this file next to xarm7_cabinet_env.py, rrt_connect.py and rrt_star.py (they are not changed).

You choose
    * the target's start cell and the goal cell     (row, col)  or a flat index row * cols + col
    * the obstacles: how many, and each one's shape / size / height / colour
      (give an explicit list, or let a seeded random generator make them -> datasets)

Two modes
    "rearrange"  the existing grid planner decides which obstacles are in the way and where to park
                 them, then the target goes to the goal                      (default)
    "direct"     obstacles stay where they are; the target is carried to the goal. The arm weaves /
                 lifts over them wherever RRT-Connect can find a path; fails cleanly if there is none

How the arm moves
    RRT-Connect (rrt_connect.py), then shortcut + smoothing + time scaling, plans every transit:
        current pose -> pre-grasp pose (inside the cabinet),
        lifted object -> above the goal cell (carrying it, object attached in the collision model),
        pull-back pose -> home.
    All of these start or end inside the cabinet, so they use tight settings: 5 mm safety margin,
    0.01 rad edge checks, slow joint limits.  Each carry is dry-run before the arm touches the object.
    Only the short contact moves stay straight, collision-checked Cartesian lines: the last
    3 cm + half the object's width in to the object, the 1.5 cm lift, the lowering, the pull-back.
    Planning in "rearrange" mode can take a few minutes (grasp-pose checks, not RRT-Connect);
    "direct" mode takes seconds.

Examples
    python cabinet_scene.py                                   # demo scene, viewer
    python cabinet_scene.py --headless --target 3,4 --goal 3,0 --n-obstacles 4 --seed 7
    python cabinet_scene.py --config scene_example.json --mode direct
    python cabinet_scene.py --print-config                    # show the full default config

From Python (dataset loops):
    from cabinet_scene import run_scene
    result = run_scene(dict(goal=(2, 1), obstacles=dict(n=5, seed=11, shapes=["box", "cylinder"])),
                       headless=True)
"""
import argparse
import copy
import itertools
import json
import sys
import time
from contextlib import contextmanager

import numpy as np
import mujoco

from xarm7_cabinet_env import (
    XArm7CabinetEnv, GraspError, GRID, CELL_W, CELL_H, CAB_X0, SHELF_Z, STANDOFF, LIFT, MIN_LIFT, MAX_LIFT,
    CELL_MARGIN, PLAN_MARGIN, PLAN_SLACK, PLAN_VMAX, PLAN_AMAX, CELL_VMAX, CELL_AMAX, APPROACH_DIST,
    PITCHES_DEG, GRASP_OFFSETS, PAD_HALF_H, FOOTPRINT_GAP, COLORS, R_pitch, grid_cell_center)
from rrt_connect import RRTConnect
from rrt_star import shortcut, smooth, time_parameterize, path_length

# ============================================================================ the scene config
DEFAULT_CONFIG = dict(
    # --- the target (a cylinder by default) and where it must go ------------------------------
    target=dict(cell=(3, 4), shape="cylinder", size=0.015, height=0.05, color="green"),
    goal=(3, 0),                       # (row, col); row 0 = front of the shelf, col 0 = -y side
    # --- the obstacles --------------------------------------------------------------------------
    # EITHER a list of explicit obstacles, e.g.
    #   [dict(cell=(2, 3), shape="box", size=0.015, height=0.06, color="red"),
    #    dict(cell=(3, 2), shape="cylinder", size=0.012, height=0.07, color="#3050ff")]
    # OR a random spec (same seed -> same scene):
    obstacles=dict(
        n=3,                           # number of obstacles (0 = empty shelf)
        seed=0,
        placement="path",              # "path": in the target's pick / place corridors (hard)
                                       # "anywhere": any free cell
        shapes=["box"],                # choose from "box", "cylinder"
        size=(0.012, 0.018),           # half-width / radius range in metres (max 0.018 on this grid)
        height=(0.05, 0.07),           # height range in metres (>= 0.035; ~6 cm reaches every row)
        colors=["red", "blue", "orange", "purple"],   # names from COLORS, or "r g b a" strings
    ),
    mode="rearrange",                  # "rearrange" | "direct"
    lift=LIFT,                         # carry height above the shelf for the contact moves (m)
    rrt_time=15.0,                     # seconds RRT-Connect may spend on one leg (3 attempts)
    seed=0,                            # RRT random seed
)


def to_cell(c, cols=GRID["cols"]):
    """(row, col) or flat index (row * cols + col) -> (row, col)."""
    if isinstance(c, (int, np.integer)):
        return divmod(int(c), cols)
    r, k = c
    return int(r), int(k)


def _color(c):
    if isinstance(c, str) and c.startswith("#") and len(c) == 7:      # "#rrggbb"
        return " ".join(f"{int(c[k:k + 2], 16) / 255:.3f}" for k in (1, 3, 5)) + " 1"
    if isinstance(c, (list, tuple)):
        return " ".join(str(float(v)) for v in (list(c) + [1.0])[:4])
    return c


def merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def build_objects(cfg):
    """Config -> list of object dicts for XArm7CabinetEnv (target first, then the obstacles)."""
    rows, cols = GRID["rows"], GRID["cols"]
    t = dict(cfg["target"])
    t_cell, goal = to_cell(t.pop("cell"), cols), to_cell(cfg["goal"], cols)
    for name, c in (("target", t_cell), ("goal", goal)):
        if not (0 <= c[0] < rows and 0 <= c[1] < cols):
            raise ValueError(f"{name} cell {c} is outside the {rows}x{cols} grid")
    if t_cell == goal:
        raise ValueError("target and goal are the same cell")
    t.update(name="target", label="T", grid=t_cell, color=_color(t.get("color", "green")))
    objs = [t]

    ob = cfg["obstacles"]
    if isinstance(ob, list):                                          # explicit obstacles
        for k, o in enumerate(ob, 1):
            o = dict(o)
            o.setdefault("name", f"obstacle{k}")
            o.setdefault("label", str(k))
            o.setdefault("shape", "box"); o.setdefault("size", 0.015); o.setdefault("height", 0.06)
            o["grid"] = to_cell(o.pop("cell"), cols)
            o["color"] = _color(o.get("color", "red"))
            objs.append(o)
        return objs, t_cell, goal

    rng = np.random.default_rng(ob.get("seed", 0))
    taken = {t_cell}
    corridor = lambda cell: [(r, c) for r in range(cell[0]) for c in (cell[1] - 1, cell[1], cell[1] + 1)
                             if 0 <= c < cols]
    everywhere = [(r, c) for r in range(rows) for c in range(cols)]
    pool = [c for c in sorted(set(corridor(t_cell)) | set(corridor(goal))) if c not in taken | {goal}]
    if ob.get("placement", "path") != "path":
        pool = []
    rest = [c for c in everywhere if c not in taken | {goal} and c not in pool]
    n = int(ob.get("n", 0))
    if n > len(pool) + len(rest):
        raise ValueError(f"cannot place {n} obstacles on a {rows}x{cols} grid")
    pool_pick = [pool[k] for k in rng.choice(len(pool), min(n, len(pool)), replace=False)] if pool else []
    extra = n - len(pool_pick)
    rest_pick = [rest[k] for k in rng.choice(len(rest), extra, replace=False)] if extra > 0 else []
    for k, cell in enumerate(pool_pick + rest_pick, 1):
        objs.append(dict(name=f"obstacle{k}", label=str(k), grid=cell,
                         shape=str(rng.choice(ob.get("shapes", ["box"]))),
                         size=round(float(rng.uniform(*ob.get("size", (0.012, 0.018)))), 4),
                         height=round(float(rng.uniform(*ob.get("height", (0.05, 0.07)))), 3),
                         color=_color(str(rng.choice(ob.get("colors", ["red"]))))))
    return objs, t_cell, goal


# ============================================================================ the environment
class RRTConnectEnv(XArm7CabinetEnv):
    """XArm7CabinetEnv in which pick_and_place() plans all transits with RRT-Connect.

    Compared with the parent class (RRT-Connect only outside the cabinet, straight Cartesian lanes
    inside), a grasp only has to satisfy LOCAL conditions: arrive pre-shaped, close, lift, and the
    same at the place spot.  How the arm gets there -- including through / over the clutter inside
    the cabinet -- is left to RRT-Connect, planned with the held object attached to the hand."""

    def __init__(self, *args, rrt_time=15.0, rrt_tries=3, max_candidates=4, **kw):
        self._tight, self._tag = False, "leg"
        self.rrt_time, self.rrt_tries, self.max_candidates = rrt_time, rrt_tries, max_candidates
        self.legs = []                                     # one record per RRT-Connect plan
        kw["planner"] = "rrt_connect"
        super().__init__(*args, **kw)

    @contextmanager
    def tight(self):
        """Plan with the in-cabinet settings: 5 mm margin, fine edge checks, slow joint limits."""
        old, self._tight = self._tight, True
        try:
            yield
        finally:
            self._tight = old

    def _ee(self, q):
        d = self.ik_data
        d.qpos[:] = self.data.qpos
        d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.model, d)
        return d.site_xpos[self.site_id].copy()

    # -- RRT-Connect, with settings that depend on where the goal is
    def plan_joint(self, q_goal, max_time=None, q_start=None, attached=None, width=None, tag=None):
        tag = tag or self._tag
        tight = self._tight
        margin = CELL_MARGIN + PLAN_SLACK if tight else PLAN_MARGIN
        res, step = (0.01, 0.15) if tight else (0.03, 0.3)
        vmax, amax = (CELL_VMAX, CELL_AMAX) if tight else (PLAN_VMAX, PLAN_AMAX)
        max_time = self.rrt_time if max_time is None else max_time
        self._sync_checker(margin)
        if attached is not None:                           # carry planned before the object is lifted
            self.checker.attach_relative(attached)
        if width is not None:
            self._set_fingers(width)
        q_start = self.data.qpos[self.qadr].copy() if q_start is None else np.asarray(q_start, float)
        ck = self.checker
        ck.n_checks = 0
        edge = lambda a, b: ck.edge_free(a, b, res=res)
        lo, hi = self.jrange[:, 0], self.jrange[:, 1]
        raw, stats = None, {}
        for attempt in range(self.rrt_tries):
            planner = RRTConnect(ck.is_free, edge, lo, hi, step=step, max_time=max_time / self.rrt_tries,
                                 seed=self.seed + attempt)
            raw = planner.plan(q_start, q_goal)
            stats = planner.stats
            if raw is not None:
                break
        if raw is None:
            raise RuntimeError(f"RRT-Connect found no path for '{tag}' ({self.rrt_tries} attempts, "
                               f"margin {margin * 1000:.0f} mm)")
        short = shortcut(raw, edge, seed=self.seed)
        smoothed = smooth(short, ck, max_step=0.02 if tight else 0.04)
        traj = time_parameterize(smoothed, vmax, amax, self.model.opt.timestep)
        pts = [self._ee(q) for q in smoothed[::max(1, len(smoothed) // 40)]]
        inside = float(np.mean([p[0] > CAB_X0 and p[2] < SHELF_Z + CELL_H for p in pts]))
        leg = dict(tag=tag, tight=tight, iters=stats.get("iters"), nodes=stats.get("nodes"),
                   plan_s=stats.get("time"), raw_len=path_length(raw), final_len=path_length(smoothed),
                   duration=len(traj) * self.model.opt.timestep, inside=inside)
        self.legs.append(leg)
        self.last_plan = dict(leg, traj=traj)
        if self.verbose:
            print(f"   RRT-Connect [{tag}]: {leg['iters']} iters, {leg['nodes']} nodes, "
                  f"{leg['plan_s']:.2f}s | path {leg['raw_len']:.2f} -> {leg['final_len']:.2f} rad | "
                  f"{inside * 100:.0f}% of the path with the hand inside the cabinet")
        return traj

    # -- grasp candidates: local conditions only (the transits are RRT-Connect's job)
    def _grasp_candidates(self, i, place, lift):
        o, p, g = self.objects[i], self.obj_pos(i), self.obj_geom[i]
        z_c = p[2]
        zs = np.arange(SHELF_Z + PAD_HALF_H, SHELF_Z + o["height"] - PAD_HALF_H + 1e-9, 0.004)
        zs = sorted(zs, key=lambda z: abs(z - z_c)) or [z_c]
        back = np.array([o["size"] + APPROACH_DIST, 0, 0])
        w_open, w_shut = self._finger_widths(i)
        up = np.array([0, 0, lift])
        blk, n_found = {}, 0
        for pitch in PITCHES_DEG:
            R = R_pitch(pitch)
            for off, z in itertools.product(GRASP_OFFSETS, zs):
                self._sync_checker(CELL_MARGIN + PLAN_SLACK)       # RRT planning in between resets it
                gp = np.array([p[0] - off, p[1], z])
                q_g = self._pose_free(gp, R, [g], blk, (w_open, w_shut))
                if q_g is None or self._pose_free(gp - back, R, (), blk, (w_open,)) is None:
                    continue
                if not (self._lines_free(q_g, gp, [gp - back], R, [g], blk, w_open) and
                        self._lines_free(q_g, gp, [gp + up], R, [g], blk, w_shut)):
                    continue
                pp = np.array([place[0] - off, place[1], SHELF_Z + o["height"] / 2 + 0.004 + (z - z_c)])
                q_p = self._pose_free(pp, R, [g], blk, (w_shut, w_open))
                if q_p is None or self._pose_free(pp - back, R, [g], blk, (w_open,)) is None:
                    continue
                if not (self._lines_free(q_p, pp, [pp + up], R, [g], blk, w_shut) and
                        self._lines_free(q_p, pp, [pp - back], R, [g], blk, w_open)):
                    continue
                n_found += 1
                yield dict(pitch=pitch, R=R, z_rel=z - z_c, offset=off, pick=gp, place=pp, q_g=q_g, q_p=q_p,
                           route="rrt")
        top = ", ".join(k for k, _ in sorted(blk.items(), key=lambda kv: -kv[1])[:3])
        raise GraspError(f"no collision-free grasp / place pose for {o['name']}"
                         + (f": the hand would hit {top}" if top else ""), 0, blk)

    def _plan_carry(self, i, cand, lift):
        """Dry run (nothing moves): RRT-Connect from the lifted pick pose to above the place spot,
        with the object rigidly attached to the hand.  Raises RuntimeError if there is no path."""
        o, g, R = self.objects[i], self.obj_geom[i], cand["R"]
        gp, pp, up = cand["pick"], cand["place"], np.array([0, 0, lift])
        q_up = self._ik_chain(cand["q_g"], gp, [gp + up], R)[-1]
        q_above = self._ik_chain(cand["q_p"], pp, [pp + up], R)[-1]
        rel_p = self.obj_pos(i) - gp                                     # object centre in the hand
        attached = [(g, R.T @ rel_p, R.T @ self.data.geom_xmat[g].reshape(3, 3))]
        with self.tight():
            return self.plan_joint(q_above, q_start=q_up, attached=attached, width=self._finger_widths(i)[1],
                                   tag="dry-run carry")

    def _choose_grasp(self, i, place, lift):
        last, tried = None, 0
        try:
            for cand in self._grasp_candidates(i, place, lift):
                tried += 1
                try:
                    verbose, self.verbose = self.verbose, False
                    try:
                        self._plan_carry(i, cand, lift)
                    finally:
                        self.verbose = verbose
                    self.legs.pop()                                      # dry run is not a motion leg
                    return cand
                except RuntimeError as err:
                    last = err
                    if tried >= self.max_candidates:
                        break
        except GraspError:
            if tried == 0:
                raise
        raise GraspError(f"{self.objects[i]['name']}: grasp poses exist but the carry fails -- {last}", 2)

    def _move_check(self, i, cell):
        """Used by the grid planner (plan_rearrangement / blockers_of): can object i be picked and
        put into grid `cell` right now?  Only the local pick / place poses are required to be free;
        the carry in between is RRT-Connect's job (it is dry-run before the arm moves)."""
        depth, dy = grid_cell_center(cell, self.grid)
        x, y = self.check_place_location(i, 0, depth, dy)
        return next(self._grasp_candidates(i, (x, y, 0), self.lift))      # GraspError if none

    # -- the skills
    def pick_and_place(self, target, cell=0, depth=0.17, dy=0.0, lift=LIFT, allow_high_transit=True):
        if not MIN_LIFT <= lift <= MAX_LIFT:
            raise ValueError(f"lift must be within {MIN_LIFT * 100:.1f}-{MAX_LIFT * 100:.0f} cm")
        i = self.obj_id(target)
        o, g = self.objects[i], self.obj_geom[i]
        x, y = self.check_place_location(i, cell, depth, dy)
        cand = self._choose_grasp(i, (x, y, cell), lift)                 # raises before anything moves
        R, gp = cand["R"], cand["pick"]
        if self.verbose:
            print(f"   grasp {o['name']}: pitch {cand['pitch']} deg, {(gp[2] - SHELF_Z) * 100:.1f} cm above the shelf")
        # ---- pick: RRT-Connect to the pre-grasp pose, straight in, close, lift
        pre = gp - np.array([o["size"] + APPROACH_DIST, 0, 0])
        self.set_gripper_width(self._finger_widths(i)[0])
        self._tag = f"{o['name']}: to pre-grasp"
        with self.tight():
            self.approach([pre, gp], R, ignores=[[g]])
        self.wait(100)
        self.set_gripper(False); self.wait(400)
        self.held, self.grasp = i, cand
        self.move_cartesian([gp + np.array([0, 0, lift])], R, ignore=[g])
        # ---- place: RRT-Connect (carrying) to above the spot, lower, release, pull back
        rel = self.obj_pos(i) - self.grasp_pos()
        put = np.array([x, y, SHELF_Z + o["height"] / 2 + 0.004]) - rel
        above = put + np.array([0, 0, lift])
        self._tag = f"{o['name']}: carry to cell"
        with self.tight():
            self.approach([above, put], R, ignores=[[g]])
        self.set_gripper_width(self._finger_widths(i)[0]); self.wait(300)
        self.held = self.grasp = None
        self.move_cartesian([put - np.array([o["size"] + APPROACH_DIST, 0, 0])], R, ignore=[g])
        return self.obj_pos(i)

    def go_home(self):
        ready = np.array([CAB_X0 - STANDOFF, 0.0, SHELF_Z + 0.20])
        q = self.ik(ready, q0=self.q_home, iters=300)
        self.set_gripper_width(0.08); self.wait(100)
        with self.tight():
            self.move_joint(q, tag="return home")


# ============================================================================ running a scene
def run_scene(user_cfg=None, headless=True, verbose=True, render_prefix=None):
    """Build the scene from the config, plan + execute, return a result dict."""
    cfg = merge(DEFAULT_CONFIG, user_cfg)
    objs, t_cell, goal = build_objects(cfg)
    env = RRTConnectEnv(n_cells=1, cell_w=CELL_W, grid=GRID, objects=objs, lift=cfg["lift"],
                        rrt_time=cfg["rrt_time"], seed=cfg["seed"], verbose=verbose)
    if verbose:
        for o in env.objects:
            print(f"{o['label']} = {o['name']:<10}: {o['color'][:14]:<14} {o['shape']:<8} "
                  f"{2 * o['size'] * 100:.1f} cm wide, {o['height'] * 100:.1f} cm tall, "
                  f"cell {env.cell_of(o['name'])}")
        print(f"\ngoal cell {goal}   mode: {cfg['mode']}\n" + env.grid_text(goal=goal) + "\n")
    if not headless:
        env.launch_viewer()
    if render_prefix:
        _render(env, f"{render_prefix}_before.png")
    start_pos = {o["name"]: env.obj_pos(o["name"]) for o in env.objects}
    t0, error = time.time(), None
    try:
        if cfg["mode"] == "rearrange":
            env.rearrange("target", goal)
        elif cfg["mode"] == "direct":
            depth, dy = grid_cell_center(goal, GRID)
            env.pick_and_place("target", 0, depth, dy, lift=cfg["lift"])
        else:
            raise ValueError("mode must be 'rearrange' or 'direct'")
        env.go_home()
    except (RuntimeError, ValueError) as err:                    # GraspError is a RuntimeError
        error = str(err)
        print(f"\nFAILED: {error}")
    if verbose:
        print("\ngrid after:\n" + env.grid_text(goal=goal))
    if render_prefix:
        _render(env, f"{render_prefix}_after.png")
    reached = env.cell_of("target") == goal
    moved = [n for n in start_pos if n != "target" and
             np.linalg.norm(env.obj_pos(n)[:2] - start_pos[n][:2]) > 0.01]
    ee_in = [l["inside"] for l in env.legs]
    res = dict(success=bool(reached and error is None), target_cell=env.cell_of("target"), goal=goal,
               moved_obstacles=moved, n_rrt_legs=len(env.legs), plan_s=sum(l["plan_s"] for l in env.legs),
               legs=env.legs, error=error, sim_s=time.time() - t0, objects=env.objects,
               hand_inside_fraction=float(np.mean(ee_in)) if ee_in else 0.0)
    if verbose:
        print(f"\n{'DONE' if res['success'] else 'NOT DONE'}: target in {res['target_cell']} (goal {goal}); "
              f"{res['n_rrt_legs']} RRT-Connect plans, {res['plan_s']:.1f}s planning, "
              f"obstacles moved: {moved or 'none'}")
    if env.viewer is not None:
        while env.viewer.is_running():
            env.step(10)
    return res


def _render(env, path):
    try:
        env.render(path, w=800, h=600, lookat=(CAB_X0 + 0.13, 0, SHELF_Z + 0.03), distance=0.85,
                   azimuth=0, elevation=-68)                 # shelf seen from the robot's side
        print(f"saved {path}")
    except Exception as err:                                    # no OpenGL on this machine
        print(f"(render skipped: {str(err)[:80]})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", help="JSON file; any key of DEFAULT_CONFIG")
    ap.add_argument("--target", help="start cell: 'row,col' or a flat index")
    ap.add_argument("--goal", help="goal cell: 'row,col' or a flat index")
    ap.add_argument("--n-obstacles", type=int)
    ap.add_argument("--seed", type=int, help="seed of the random obstacles")
    ap.add_argument("--mode", choices=["rearrange", "direct"])
    ap.add_argument("--lift", type=float)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--render", metavar="PREFIX", help="save PREFIX_before.png / PREFIX_after.png")
    ap.add_argument("--print-config", action="store_true")
    a = ap.parse_args()
    cell = lambda s: tuple(int(v) for v in s.split(",")) if "," in s else int(s)
    cfg = {}
    if a.config:
        cfg = json.load(open(a.config))
    if a.target:
        cfg = merge(cfg, dict(target=dict(cell=cell(a.target))))
    if a.goal:
        cfg["goal"] = cell(a.goal)
    if a.n_obstacles is not None and not isinstance(cfg.get("obstacles"), list):
        cfg = merge(cfg, dict(obstacles=dict(n=a.n_obstacles)))
    if a.seed is not None and not isinstance(cfg.get("obstacles"), list):
        cfg = merge(cfg, dict(obstacles=dict(seed=a.seed)))
    for k in ("mode", "lift"):
        if getattr(a, k) is not None:
            cfg[k] = getattr(a, k)
    if a.print_config:
        print(json.dumps(merge(DEFAULT_CONFIG, cfg), indent=2, default=list))
        return
    try:
        res = run_scene(cfg, headless=a.headless, render_prefix=a.render)
    except ValueError as err:                                  # bad config: size, cell, overlap ...
        print(f"\nCONFIG ERROR: {err}")
        sys.exit(2)
    sys.exit(0 if res["success"] else 1)


if __name__ == "__main__":
    main()
