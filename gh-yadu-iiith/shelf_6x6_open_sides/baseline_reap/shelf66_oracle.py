"""REAP pick / put feasibility on the 6x6 open-sided shelf (SHELF66V3, no overpass, no return to ready).

The scene, planner and move check are the dataset teacher's own: bd_samples/shelf66_v3 (bd_shelf -> bd_3dt.build /
settle / query, shelf_scene_6x6 with the V3 pre-shaped-finger fix, RRT_ITERS ceiling, seed 0).

REAP's actions vs our moves
    put(o, d)   = the teacher's move check  bd_3dt.query(o -> d)  with o standing in its SOURCE cell and the others
                  where the state says: approach, grasp, lift, in-shelf carry, lower, release, retreat (first feasible
                  grasp, planned from the ready pose, as the teacher's scene queries are).
                  REAP's state forgets the source once o is held, so the source is remembered when pick(o, c) is
                  checked (GBFS checks pick before it can generate the held state). Two different picks can reach the
                  same symbolic state (same others, o held); the put is then feasible if any remembered source works.
                  The replay (replay66.py) executes the plan with the real sources, so the final verdict is exact.
    pick(o, c)  = a cheap filter: some grasp of o in c has a free move ready -> hover and a straight approach, fingers
                  pre-shaped. Necessary for any move of o, not sufficient (the carry is judged at the put).
"""
import os
import sys
from pathlib import Path

V3_DIR = Path(os.environ.get("SHELF66_V3_DIR", Path.home() / "codebase/bd_samples/shelf66_v3"))


def _teacher():
    if str(V3_DIR) not in sys.path:
        sys.path.insert(0, str(V3_DIR))
    import bd_shelf          # noqa: F401  sets up bd_3dt with the 6x6 scene (env(), build = ShelfEnv)
    import bd_3dt as A
    return A


class Shelf66Oracle:
    def __init__(self, side: dict):
        self.A = A = _teacher()
        self.S = A.env()["TS"]
        self.side = side
        self.objs = [dict(id=o["id"], shape=o["shape"], size=o["size"], height=o["height"], color=o["color"])
                     for o in side["objects"]]
        self.pddl = {o["id"]: o["pddl"] for o in side["objects"]}          # dataset id -> PDDL name
        self.ids = {v: k for k, v in self.pddl.items()}
        self.cells = {k: tuple(v) for k, v in side["cells"].items()}       # PDDL cell -> (row, col)
        self.e = A.build(self.objs, {o["id"]: tuple(o["cell"]) for o in side["objects"]})
        self.sources = {}            # (others, obj) -> set of cells obj was picked from
        self.cache = {}
        self.queries = 0
        self.errors = []

    def _settle(self, cells: dict):
        """bd_3dt.settle, but safe for swaps: the teacher teleports one object at a time and refuses a cell another
        object still stands in, which REAP's states can need (a <- b's cell, b <- elsewhere). So every object that
        moves is first parked off the grid, then teleported (each teleport settles 100 steps, as in the teacher)."""
        e = self.e
        e.reset()
        movers = [o for o, c in cells.items() if e.cell_of(o) != tuple(c)]
        for k, o in enumerate(movers):
            a = e.obj_qadr[e.obj_id(o)]
            e.data.qpos[a:a + 3] = [5.0 + k, 5.0, e.table_z + 0.1]
        self.S.mujoco.mj_forward(e.model, e.data)
        for o in movers:
            e.teleport(o, tuple(cells[o]))

    def pick(self, layout: dict, obj: str, cell) -> bool:
        """layout: {dataset id: (row, col)} of every object on the shelf (obj included, at `cell`)."""
        others = frozenset((k, v) for k, v in layout.items() if k != obj)
        self.sources.setdefault((others, obj), set()).add(tuple(cell))
        key = ("pick", others, obj, tuple(cell))
        if key not in self.cache:
            self.queries += 1
            self._settle({**dict(others), obj: tuple(cell)})
            e, S = self.e, self.S
            i = e.obj_id(obj)
            e._focus(i)
            e.objects[i]["pos"] = e.obj_pos(i)
            ok = False
            for g in S.grasp_poses(e.objects[i], e.table_z):
                e._check_width = g["open_width"]
                try:
                    for branch in range(e.n_branches):
                        try:
                            e._plan_approach(g, e.q_now(), seed=0, branch=branch)
                            ok = True
                            break
                        except (RuntimeError, ValueError) as err:
                            if isinstance(err, S.ReachError) or "out of reach" in str(err):
                                break
                finally:
                    e._check_width = None
                if ok:
                    break
            self.cache[key] = ok
        return self.cache[key]

    def move(self, others: frozenset, obj: str, src, dst) -> bool:
        """The teacher's check of obj: src -> dst with the others where they are (from the ready pose)."""
        key = ("move", others, obj, tuple(src), tuple(dst))
        if key not in self.cache:
            self.queries += 1
            self._settle({**dict(others), obj: tuple(src)})
            p, _ = self.A.query(self.e, obj, tuple(dst))
            self.cache[key] = p is not None
        return self.cache[key]

    def put(self, layout: dict, obj: str, cell) -> bool:
        """layout: the objects on the shelf (obj is held, not in it)."""
        others = frozenset((k, v) for k, v in layout.items() if k != obj)
        if tuple(cell) in dict(others).values():
            return False
        srcs = self.sources.get((others, obj), set())
        return any(self.move(others, obj, s, cell) for s in sorted(srcs) if tuple(s) != tuple(cell))

    def check(self, layout_pddl: dict, held, action) -> bool:
        """REAP side: layout {PDDL obj: PDDL cell}; action (name, PDDL obj, PDDL cell). True iff FEASIBLE."""
        name, obj, cname = action
        layout = {self.ids[o]: self.cells[c] for o, c in layout_pddl.items()}
        oid, cell = self.ids[obj], self.cells[cname]
        if name not in ("pick", "put"):
            raise ValueError(f"pick / put only, got {name!r}")
        try:
            return self.pick(layout, oid, cell) if name == "pick" else self.put(layout, oid, cell)
        except Exception as err:                       # a scene set-up / planner error must not kill the search
            self.errors.append(f"{name} {oid} {cell}: {type(err).__name__}: {err}")
            print(f"[shelf66_oracle] {self.side['problem']}: {self.errors[-1]} -> infeasible", file=sys.stderr, flush=True)
            return False
