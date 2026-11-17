"""MO-SeGMan (Tuncer, Toussaint & Oguz, arXiv 2511.01476) on the 2D pick-and-place dataset, ported from the reference implementation
(github.com/CankutBoraTuncer/MO-SeGMan, src/MOSeGMan.py + src/SeGManv2.py) where the paper leaves details open.

Single goal object (T): the object placement sequence is S = [T] (dependency graph and ATSP are trivial), so the planner reduces to
SeGMan's solve() for T with Selective Guided Forward Search (SGFS) relocating critical obstacles.

* Task trajectory: the target task planned with every movable ignored (the reference's run_rrt with contacts disabled).
* Critical objects (find_critical_objects): objects colliding with the task trajectory; if more than one and at most 4, the first subset
  (in order of size) whose removal makes the task trajectory feasible; more than 4: the 4 closest to the robot (here: its TCP).
  Feasibility with a subset removed = the task trajectory clears the remaining objects (screen; the reference re-runs RRT, which with
  the dataset's planner costs ~300 CPU-s per failing subset, see common.py).
* Global Occupancy Matrix on a 1 cm raster of the board: obstacles 0, free floor 80, task trajectory footprint 200 (scene_adjust);
  reachability = wavefront propagation of the gripper-sized block from the TCP over non-obstacle pixels, 1.1 reachable / 0.99 not;
  scene score = sum(GOM * reach) / (h w 10).
* Object weights (calc_object_weights): raw-score gain of removing the object minus 80 per object pixel, clipped at 0, normalised.
* Node score = scene score + 5 / sqrt(visits + 1); select_node with the reference's weight update and its secondary-obstacle rule
  (no score improvement -> the object with most collisions during relocation attempts joins the critical set, weight ~ its size).
* Relocation points (generate_reloc_pts): local window around the object, cells where the object fits the free floor (not on the task
  trajectory), clearance (EDT) at or above the 75th percentile of the window; random sample of k = min(max(ceil(3 %), amount), n);
  up to 20 placements tried until `amount` succeed; amount = int(w / sum w * 5 * ceil(sqrt(|critical|))); the 3 best new nodes are kept.
* Reachability of an object (is_reachable): some grasp passes the planner's static pick screen (grasp pose + certified approach line).
* SGFS ends successfully when the selected node admits the task (the dataset's planner: target from q_ref + connector); otherwise a
  second SGFS with an alternative critical set (remove_obstacle); otherwise no solution.

Changed for this dataset (deviations): motion planning with the dataset's planner instead of KOMO + Bi-RRT; T moves once (the dataset's
task), so adaptive subgoals / refinement for T do not apply; the SGFS iteration limit is raised from the reference's |O_g| + 1 = 2 to
MAX_RELOC + 1 = 5 so that the search can reach the dataset's 4-relocation depth; each object moves at most once; cell destinations."""
from __future__ import annotations

import collections
import itertools
import math
from collections import deque

import numpy as np
from scipy.ndimage import distance_transform_edt

from .common import MAX_RELOC, Result, dist, free_cells, relaxed_target, relocation_blockers, start, target_blockers, task_ok
from .teacher import PICK_PHASES

PX = 0.01                   # raster resolution (m)
FREE, OBST, TRAJ = 80, 0, 200
WINDOW = 3                  # LOM half-width in lattice cells
EDT_PCT, MAX_PTS, SAMPLE_COUNT, C_EXPL = 75, 20, 3, 5.0


class Node:
    _ids = itertools.count()

    def __init__(self, cells, q, seq, objs, weights, scores, raw_prev, moved_obj=None):
        self.id = next(Node._ids)
        self.cells, self.q, self.seq = cells, q, seq
        self.objs, self.obj_weights, self.obj_scores = list(objs), dict(weights), dict(scores)
        self.scene_score_raw_prev, self.moved_obj = raw_prev, moved_obj
        self.visit, self.scene_score, self.scene_score_raw, self.total_score = 0, 0.0, 0.0, 0.0

    def reloc(self):
        return {o for o, _ in self.seq}


class MOSeGMan2D:
    def __init__(self, m, rng):
        self.m, self.rng = m, rng
        self.st = collections.Counter()
        lat = m.lat
        self.W, self.H = lat.cols * lat.pitch_m, lat.rows * lat.pitch_m
        self.nx, self.ny = int(round(self.W / PX)), int(round(self.H / PX))
        ys, xs = np.mgrid[0:self.ny, 0:self.nx]
        self.px = np.stack([(xs + 0.5) * PX, (ys + 0.5) * PX], -1)          # pixel centres (x, y); row = y
        self.r = m.radius["t0"]
        self.rh = max(1, int(math.ceil(self.r / PX)))                         # gripper-sized block half-width

    # ------------------------------------------------------------------ images
    def cell_mask(self, c):
        x0, y0 = self.m.xy(c) - self.m.lat.pitch_m / 2
        return (self.px[..., 0] >= x0) & (self.px[..., 0] < x0 + self.m.lat.pitch_m) & (self.px[..., 1] >= y0) & (self.px[..., 1] < y0 + self.m.lat.pitch_m)

    def disc(self, c):
        return np.hypot(*(self.px - self.m.xy(c)).transpose(2, 0, 1)) <= self.r

    def gom(self, cells, hide=()):
        img = np.full((self.ny, self.nx), FREE, float)
        for c in self.traj_cells:
            img[self.cell_mask(c)] = TRAJ
        for o, c in cells.items():
            if o != "t0" and o not in hide:
                img[self.disc(c)] = OBST
        return img

    def reach(self, img, q):
        tcp = self.m._ge.fk(self.m.robot, q)[0][3]
        sx = int(np.clip(tcp[0] / PX, 0, self.nx - 1))
        sy = int(np.clip(tcp[1] / PX, 0, self.ny - 1))
        h, w, rh = self.ny, self.nx, self.rh
        dist_ = np.full((h, w), 0.99)
        vis = np.zeros((h, w), bool)
        ok = lambda y, x: rh <= y < h - rh and rh <= x < w - rh  # noqa: E731
        if not ok(sy, sx):                                                    # TCP outside the board: enter at the nearest valid pixel
            sy, sx = int(np.clip(sy, rh, h - rh - 1)), int(np.clip(sx, rh, w - rh - 1))
        dq = deque()
        if (img[sy - rh:sy + rh + 1, sx - rh:sx + rh + 1] > 0).all():
            dist_[sy - rh:sy + rh + 1, sx - rh:sx + rh + 1] = 1.1
            dq.append((sy, sx))
            vis[sy, sx] = True
        st = max(1, rh // 2)
        moves = [(st, 0), (-st, 0), (0, st), (0, -st), (st, st), (st, -st), (-st, st), (-st, -st)]
        while dq:
            y, x = dq.popleft()
            for dy, dx in moves:
                ny_, nx_ = y + dy, x + dx
                if ok(ny_, nx_) and not vis[ny_, nx_] and (img[ny_ - rh:ny_ + rh + 1, nx_ - rh:nx_ + rh + 1] > 0).all():
                    dist_[ny_ - rh:ny_ + rh + 1, nx_ - rh:nx_ + rh + 1] = 1.1
                    vis[ny_, nx_] = True
                    dq.append((ny_, nx_))
        return dist_

    def scene_score(self, cells, q):
        img = self.gom(cells)
        return float((img * self.reach(img, q)).sum() / (img.size * 10.0)), float(img.sum())

    # ------------------------------------------------------------------ critical objects, weights
    def task_feasible_without(self, cells, removed) -> bool:
        """Screen: the task trajectory clears every object that is not removed."""
        return not [o for o in target_blockers(self.m, self.gamma, cells) if o not in removed]

    def find_critical(self, cells, q, obj_list):
        blk = [o for o in target_blockers(self.m, self.gamma, cells) if o in obj_list]
        if not blk:
            return None
        if len(blk) == 1:
            return blk
        if len(blk) <= 4:
            for k in range(1, len(blk) + 1):
                for comb in itertools.combinations(blk, k):
                    if self.task_feasible_without(cells, comb):
                        return list(comb)
            return None
        tcp = self.m._ge.fk(self.m.robot, q)[0][3]
        return sorted(blk, key=lambda o: (float(np.linalg.norm(self.m.xy(cells[o]) - tcp)), o))[:4]

    def weights(self, cells, objs):
        init = self.gom(cells).sum()
        sc = {}
        for o in objs:
            g = self.gom(cells, hide=(o,)).sum() - self.disc(cells[o]).sum() * FREE
            sc[o] = max(0.0, float(g - init))
        tot = sum(sc.values())
        w = {o: (s / tot if tot > 0 else 1 / len(sc)) for o, s in sc.items()}
        return w, sc, float(init), tot

    def node_score(self, n):
        n.total_score = n.scene_score + C_EXPL / math.sqrt(n.visit + 1)

    def node_eval(self, n):
        n.scene_score, n.scene_score_raw = self.scene_score(n.cells, n.q)
        self.node_score(n)

    def select_node(self, N, prev, col):
        if prev is not None:
            prev.visit += 1
            self.node_score(prev)
        best = max(N, key=lambda n: n.total_score)
        if prev is not None and prev.scene_score >= best.scene_score and col:
            red = max(col, key=col.get)
            if red not in best.objs:
                col.pop(red)
                best.objs.append(red)
                best.obj_scores[red] = float(self.disc(best.cells[red]).sum()) * 200 * 0.6
                tot = sum(best.obj_scores.values())
                best.obj_weights = {o: s / tot for o, s in best.obj_scores.items()}
                self.st["secondary_added"] += 1
                return best
        if best.moved_obj is not None:
            delta = best.scene_score_raw - best.scene_score_raw_prev
            best.scene_score_raw_prev = best.scene_score_raw
            if delta > 0:
                z = sum(1 for v in best.obj_scores.values() if v == 0)
                if z > 0:
                    best.obj_scores = {k: (v + delta / z if v == 0 else v) for k, v in best.obj_scores.items()}
            if best.moved_obj in best.obj_scores:
                best.obj_scores[best.moved_obj] = max(best.obj_scores[best.moved_obj] - delta, 0.001)
            tot = sum(best.obj_scores.values())
            best.obj_weights = {o: s / tot for o, s in best.obj_scores.items()}
        return best

    # ------------------------------------------------------------------ relocation
    def reachable(self, n, o) -> bool:
        """is_reachable: a pick of o is available in the node's scene (the planner's static pick screen over all grasps: grasp pose +
        certified approach line; the reference runs find_pick_path)."""
        if o == n.moved_obj:
            return True
        return bool(self.m.screen(n.cells, o)["pick_ok"])

    def reloc_pts(self, n, o):
        m, cells = self.m, n.cells
        r0, c0 = divmod(cells[o], m.lat.cols)
        win = [c for c in m.legal if abs(c // m.lat.cols - r0) <= WINDOW and abs(c % m.lat.cols - c0) <= WINDOW]
        cand = [c for c in free_cells(m, cells) if c in win and c not in self.traj_cells and c != m.goal_cell]
        if not cand:
            return []
        img = self.gom(cells)
        wmask = np.zeros_like(img, bool)
        for c in win:
            wmask |= self.cell_mask(c)
        edt = distance_transform_edt(img > OBST) * PX
        tr = np.percentile(edt[wmask], EDT_PCT)
        pts = []
        for c in cand:
            x, y = m.xy(c)
            if edt[min(self.ny - 1, int(y / PX)), min(self.nx - 1, int(x / PX))] >= tr:
                pts.append(c)
        return pts

    def generate_new_nodes(self, n, o, N, col):
        amount = int((n.obj_weights[o] / sum(n.obj_weights.values())) * 5 * math.ceil(math.sqrt(len(n.objs))))
        if amount <= 0:
            return
        pts = self.reloc_pts(n, o)
        if not pts:
            return
        k = min(max(math.ceil(len(pts) * 0.03), amount), len(pts))
        pts = [pts[i] for i in self.rng.choice(len(pts), size=k, replace=False)]
        new, count = [], 0
        for c in pts[:MAX_PTS]:
            if count >= amount:
                break
            self.st["relocation_attempts"] += 1
            r = self.m.relocate(n.cells, n.q, o, c)                           # the placement plan (the dataset's planner, capped)
            if r is None:
                for b in relocation_blockers(self.m, n.cells, o, c) or ():    # objects in the way of the failed placement
                    if b not in n.objs and b != "t0":
                        col[b] += 1
                continue
            count += 1
            u = Node({**n.cells, o: c}, r[0], n.seq + [(o, c)], n.objs, n.obj_weights, n.obj_scores, n.scene_score_raw_prev, o)
            self.node_eval(u)
            new.append(u)
        new.sort(key=lambda u: u.total_score, reverse=True)
        N.extend(new[:SAMPLE_COUNT])

    # ------------------------------------------------------------------ SGFS
    def sgfs(self, root):
        N, prev, idx, stuck = [root], None, 0, 0
        col = collections.Counter()
        max_iter, stuck_lim = MAX_RELOC + 1, 2
        while idx < max_iter and stuck < stuck_lim:
            idx += 1
            node = self.select_node(N, prev, col)
            if prev is not None and node.id != prev.id:
                self.st["task_checks"] += 1
                if task_ok(self.m, self.gamma, node.cells, node.q):
                    return node
            stuck = stuck + 1 if (prev is not None and node.scene_score <= prev.scene_score) else 0
            prev = node
            any_reach = False
            for o in list(node.obj_weights):
                if o in node.reloc() or len(node.seq) >= MAX_RELOC:
                    continue
                if not self.reachable(node, o):
                    continue
                any_reach = True
                self.generate_new_nodes(node, o, N, col)
            if not any_reach:
                N.remove(node)
                prev = None
                if not N:
                    return None
        return None

    def remove_obstacle(self, cells, q):
        objs = [o for o in self.m.movable]
        for attempt in range(2):
            crit = self.find_critical(cells, q, objs)
            if crit is None:
                return None
            w, sc, init, tot = self.weights(cells, crit)
            root = Node(dict(cells), q, [], crit, w, sc, init)
            self.node_eval(root)
            self.st["sgfs_runs"] += 1
            res = self.sgfs(root)
            if res is not None:
                return res
            objs = [o for o in objs if o not in crit]                         # alternative critical set
        return None

    def solve(self) -> Result:
        m = self.m
        import geometry as ge
        m._ge = ge
        self.gamma = relaxed_target(m)
        r = start(m, self.gamma)
        if r:
            return r
        self.traj_cells = m.swept(self.gamma, "t0", m.cells0["t0"], m.legal)
        node = self.remove_obstacle(dict(m.cells0), m.qref)
        if node is None:
            return Result("ABSTAIN", [], "SGFS found no feasible configuration", dict(self.st))
        return Result("REPAIR", list(node.seq), "task feasible", dict(self.st))


def solve(m, rng) -> Result:
    return MOSeGMan2D(m, rng).solve()
