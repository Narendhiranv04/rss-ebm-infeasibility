"""Scene records (Methodology v1.4 §2, §4; Guide v1.6 §1-2).

Owns: the physical cell lattice, stable object IDs,
the restorable 2D scene state S, per-object candidate maps (index 0 = stay),
ghost pose substitution, JSON/NPZ snapshots and the physical-state hash.

Conventions (Methodology §2): metres, radians, float64 geometry. World x points
right and world y points backward on the support. Lattice row r increases with
world +y and column c with world +x; cell (r, c) has physical ID r*cols + c and
centre ((c + 1/2) h, (r + 1/2) h) for pitch h.

This module never imports training, simulation or plotting code.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from config import robot_profile_hash, stable_hash

# Two poses closer than this are "identical pose values" (Methodology §4 dedup).
POSE_EQ_TOL_M = 1e-9


# ------------------------------------------------------------------ lattice
@dataclass(frozen=True)
class Lattice:
    rows: int
    cols: int
    pitch_m: float

    @property
    def n_cells(self) -> int:
        return self.rows * self.cols

    def cell_id(self, r: int, c: int) -> int:
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            raise ValueError(f"(r, c) = ({r}, {c}) outside lattice")
        return r * self.cols + c

    def rc(self, cell: int) -> tuple[int, int]:
        if not 0 <= cell < self.n_cells:
            raise ValueError(f"cell {cell} outside lattice")
        return divmod(cell, self.cols)

    def center(self, cell: int) -> tuple[float, float]:
        """World (x, y) [m] of a cell centre: x = (c + 1/2) h, y = (r + 1/2) h."""
        r, c = self.rc(cell)
        return ((c + 0.5) * self.pitch_m, (r + 0.5) * self.pitch_m)


# ------------------------------------------------------------------ state S
KINDS = ("obstacle", "target")


@dataclass(frozen=True)
class Obj:
    """Rigid upright disc (2D top view of a cylinder). xy_m is the authoritative
    pose; `cell` is the supported resting cell, or None when off-lattice."""
    id: str
    kind: str
    radius_m: float
    xy_m: tuple[float, float]
    cell: int | None
    support: str = "table"


@dataclass(frozen=True)
class RobotState:
    """Joint angles q [rad], gripper clear gap g [m], attachment mode m."""
    q_rad: tuple[float, float, float]
    gripper_gap_m: float
    attachment: str = "free"
    attached_id: str | None = None


@dataclass(frozen=True)
class Scene:
    """Restorable scene state. Objects are kept sorted by stable ID, so the
    order in which they are supplied never changes the record or its hash."""
    root_id: str
    variant_id: str
    lattice: Lattice
    objects: tuple[Obj, ...]
    robot: RobotState
    legal_cells: tuple[int, ...]
    domain_policy: str
    robot_profile_hash: str
    action_hash: str | None = None  # frozen gamma_0 hash; none exists before G1/G2

    def __post_init__(self):
        object.__setattr__(self, "objects", tuple(sorted(self.objects, key=lambda o: o.id)))
        ids = [o.id for o in self.objects]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate object IDs")
        for o in self.objects:
            if o.kind not in KINDS:
                raise ValueError(f"{o.id}: unknown kind {o.kind}")
            if o.cell is not None and math.dist(self.lattice.center(o.cell), o.xy_m) > POSE_EQ_TOL_M:
                raise ValueError(f"{o.id}: pose does not match cell {o.cell}")
        if list(self.legal_cells) != sorted(set(self.legal_cells)):
            raise ValueError("legal_cells must be sorted and unique")
        if any(not 0 <= c < self.lattice.n_cells for c in self.legal_cells):
            raise ValueError("legal cell outside lattice")
        if (self.robot.attachment == "free") != (self.robot.attached_id is None):
            raise ValueError("attachment mode and attached_id disagree")

    def get(self, obj_id: str) -> Obj:
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(obj_id)


def make_scene(cfg: dict, root_id: str, variant_id: str, seed: int) -> Scene:
    """Seeded scene: targets then obstacles drawn without replacement from all
    cells, so every object rests at a distinct supported cell centre."""
    s = cfg["suite"]
    order = np.random.default_rng(seed).permutation(s["rows"] * s["cols"]).tolist()
    nt, no = s["num_targets"], s["num_obstacles"]
    return place_scene(cfg, root_id, variant_id, order[:nt], order[nt:nt + no])


def place_scene(cfg: dict, root_id: str, variant_id: str, target_cells, obstacle_cells) -> Scene:
    """Scene with targets t0.. and obstacles o00.. at the given distinct cells."""
    s, a, g = cfg["suite"], cfg["arm2d"], cfg["gripper2d"]
    if len(set(target_cells) | set(obstacle_cells)) != len(target_cells) + len(obstacle_cells):
        raise ValueError("objects must occupy distinct cells")
    lat, rad = Lattice(s["rows"], s["cols"], s["pitch_m"]), s["obstacle_radius_m"]
    objs = [Obj(f"t{i}", "target", rad, lat.center(c), c) for i, c in enumerate(target_cells)]
    objs += [Obj(f"o{i:02d}", "obstacle", rad, lat.center(c), c) for i, c in enumerate(obstacle_cells)]
    robot = RobotState(tuple(a["q_home_rad"]), g["open_gap_m"])
    return Scene(root_id, variant_id, lat, tuple(objs), robot, tuple(range(lat.n_cells)),
                 s["domain_policy"], robot_profile_hash(cfg))


def occupancy(scene: Scene) -> dict[int, tuple[str, ...]]:
    """Physical cell ID -> IDs of objects resting there (several only for ghosts)."""
    occ: dict[int, list[str]] = {}
    for o in scene.objects:
        if o.cell is not None:
            occ.setdefault(o.cell, []).append(o.id)
    return {c: tuple(v) for c, v in sorted(occ.items())}


# ------------------------------------------------------------------ candidates
@dataclass(frozen=True)
class CandidateMap:
    """Categorical domain of one movable obstacle (Methodology §4).

    x_i = 0 keeps the actual current pose; x_i = k > 0 places the SAME object at
    physical cell cells[k - 1]. Indexes are per-object, not global cell IDs.
    """
    obj_id: str
    stay_xy_m: tuple[float, float]
    cells: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.cells) + 1

    def cell_of(self, k: int) -> int | None:
        if not 0 <= k < self.size:
            raise IndexError(k)
        return None if k == 0 else self.cells[k - 1]


def candidate_map(scene: Scene, obj_id: str) -> CandidateMap:
    """Legal cells, minus the object's own current pose (identical-pose dedup)
    and minus target-occupied cells (targets are never relocated). Cells held by
    other obstacles stay in the domain; pair factors decide their coexistence."""
    o = scene.get(obj_id)
    if o.kind != "obstacle":
        raise ValueError(f"{obj_id} is a {o.kind}; only obstacles have relocation domains")
    target_cells = {t.cell for t in scene.objects if t.kind == "target"}
    cells = tuple(c for c in scene.legal_cells if c not in target_cells
                  and math.dist(scene.lattice.center(c), o.xy_m) > POSE_EQ_TOL_M)
    return CandidateMap(obj_id, o.xy_m, cells)


def substitute_pose(scene: Scene, obj_id: str, cell: int) -> Scene:
    """Ghost hypothesis: the same object ID at a legal cell. Never clones; the
    source pose is vacated and every other field is unchanged."""
    if cell not in scene.legal_cells:
        raise ValueError(f"cell {cell} is not legal")
    scene.get(obj_id)
    objs = tuple(replace(o, xy_m=scene.lattice.center(cell), cell=cell) if o.id == obj_id else o
                 for o in scene.objects)
    return replace(scene, objects=objs)


# ------------------------------------------------------------------ snapshots
def scene_to_dict(scene: Scene) -> dict:
    return json.loads(json.dumps(asdict(scene)))


def scene_from_dict(d: dict) -> Scene:
    objs = tuple(Obj(**{**o, "xy_m": tuple(o["xy_m"])}) for o in d["objects"])
    robot = RobotState(**{**d["robot"], "q_rad": tuple(d["robot"]["q_rad"])})
    return Scene(**{**d, "lattice": Lattice(**d["lattice"]), "objects": objs, "robot": robot,
                    "legal_cells": tuple(d["legal_cells"])})


def scene_hash(scene: Scene) -> str:
    """Physical-state hash: every field except the root/variant labels."""
    d = scene_to_dict(scene)
    del d["root_id"], d["variant_id"]
    return stable_hash(d)


def save_scene_json(scene: Scene, path: str | Path) -> None:
    record = {"record": "Scene", "version": 1, "scene_hash": scene_hash(scene),
              "scene": scene_to_dict(scene)}
    Path(path).write_text(json.dumps(record, indent=1, sort_keys=True))


def load_scene_json(path: str | Path) -> Scene:
    record = json.loads(Path(path).read_text())
    scene = scene_from_dict(record["scene"])
    if scene_hash(scene) != record["scene_hash"]:
        raise ValueError(f"{path}: stored scene_hash does not match contents")
    return scene


def scene_arrays(scene: Scene) -> dict[str, np.ndarray]:
    """Numeric mirror of the scene for NPZ (object rows in stable-ID order).
    candidate_cells[i, k-1] is obstacle i's candidate k; -1 pads short rows."""
    obstacles = [o.id for o in scene.objects if o.kind == "obstacle"]
    maps = [candidate_map(scene, i).cells for i in obstacles]
    width = max((len(m) for m in maps), default=0)
    table = np.full((len(maps), width), -1, dtype=np.int64)
    for row, m in enumerate(maps):
        table[row, :len(m)] = m
    return {
        "object_ids": np.array([o.id for o in scene.objects], dtype="U16"),
        "object_kind": np.array([o.kind for o in scene.objects], dtype="U16"),
        "object_xy_m": np.array([o.xy_m for o in scene.objects], dtype=np.float64).reshape(-1, 2),
        "object_radius_m": np.array([o.radius_m for o in scene.objects], dtype=np.float64),
        "object_cell": np.array([-1 if o.cell is None else o.cell for o in scene.objects],
                                dtype=np.int64),
        "robot_q_rad": np.array(scene.robot.q_rad, dtype=np.float64),
        "robot_gripper_gap_m": np.array(scene.robot.gripper_gap_m, dtype=np.float64),
        "legal_cells": np.array(scene.legal_cells, dtype=np.int64),
        "candidate_obstacle_ids": np.array(obstacles, dtype="U16"),
        "candidate_cells": table,
    }
