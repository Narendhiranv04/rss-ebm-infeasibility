"""Point clouds for SceneCollisionNet on the 6x6 shelf, from the teacher's own geometry (no rendering).

  gripper_cloud()      1024 surface points of the xArm7 hand + fingers (the geoms of the hand subtree the teacher's collision checker
                       uses, fingers pre-shaped to a nominal opening), in the GRASP frame: origin = the pad centre (the teacher's
                       "grasp" site), axes = the grasp rotation R (hand orientation).  World cloud of a grasp = R @ pts + position.
  scene_cloud(...)     N surface points of the scene at a layout: floor board, back wall, the virtual arm ceiling (tallest object's top +
                       CEILING_MARGIN, a constraint of the teacher's checks) and every object (cylinder / box, standing on its cell).
  grasps_of(...)       the 32 grasp poses (position, R) of the target at its start cell: shelf_scene.grasp_poses.
Everything is expressed in the world frame of the shelf scene (robot base at the origin).
"""
import os
import sys
from pathlib import Path

import numpy as np

V3_DIR = Path(os.environ.get("SHELF66_V3_DIR", Path.home() / "codebase/bd_samples/shelf66_v3"))
HERE = Path(__file__).resolve().parent
X0, PITCH, ZB, ROWS, COLS = 0.42, 0.075, 0.55, 6, 6
CLR, T_BOARD, CEILING_MARGIN = 0.60, 0.02, 0.10
XF, XB = X0 - ROWS * PITCH / 2 - 0.04, X0 + ROWS * PITCH / 2 + 0.04
HW = COLS * PITCH / 2 + 0.14
NOMINAL_OPEN = 0.05                                    # pad gap of the gripper cloud (the grasps' open_width is object width + 1.2 cm)
_T = {}


def teacher():
    if "A" not in _T:
        sys.path.insert(0, str(V3_DIR))
        import bd_shelf  # noqa: F401
        import bd_3dt as A
        _T["A"], _T["S"] = A, A.env()["TS"]
    return _T["A"], _T["S"]


def cell_xy(cell):
    r, c = cell
    return X0 + (r - 2.5) * PITCH, (c - 2.5) * PITCH


# ------------------------------------------------------------------ surface sampling
def _tri_sample(verts, faces, n, rng):
    tri = verts[faces]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    k = rng.choice(len(faces), n, p=area / area.sum())
    u, v = rng.random(n), rng.random(n)
    f = u + v > 1
    u[f], v[f] = 1 - u[f], 1 - v[f]
    t = tri[k]
    return t[:, 0] + u[:, None] * (t[:, 1] - t[:, 0]) + v[:, None] * (t[:, 2] - t[:, 0])


def box_points(half, n, rng):
    """n points on the surface of a box with half extents `half` (area weighted), centred at the origin"""
    a = np.array([half[1] * half[2], half[0] * half[2], half[0] * half[1]]) * 4
    face = rng.choice(6, n, p=np.repeat(a, 2) / (2 * a.sum()))
    p = (rng.random((n, 3)) * 2 - 1) * np.asarray(half)
    ax, sgn = face // 2, np.where(face % 2 == 0, 1.0, -1.0)
    p[np.arange(n), ax] = sgn * np.asarray(half)[ax]
    return p


def cylinder_points(r, half_h, n, rng):
    a_side, a_cap = 2 * np.pi * r * 2 * half_h, 2 * np.pi * r * r / 2 * 2          # side, two caps (pi r^2 each)
    a_cap = 2 * np.pi * r * r
    side = rng.random(n) < a_side / (a_side + a_cap)
    th = rng.random(n) * 2 * np.pi
    rad = np.where(side, r, r * np.sqrt(rng.random(n)))
    z = np.where(side, (rng.random(n) * 2 - 1) * half_h, np.where(rng.random(n) < 0.5, half_h, -half_h))
    return np.stack([rad * np.cos(th), rad * np.sin(th), z], 1)


# ------------------------------------------------------------------ gripper
def gripper_cloud(n=1024, seed=0, cache=True):
    f = HERE / "data" / f"gripper_cloud_{n}.npz"
    if cache and f.exists():
        return np.load(f)["pts"].astype(np.float32)
    import mujoco
    A, S = teacher()
    rng = np.random.default_rng(seed)
    objs = [dict(id="t0", shape="cylinder", size=0.017, height=0.15, color="green")]
    e = A.build(objs, {"t0": (3, 3)})
    m, d = e.model, e.data
    e.reset()
    d.qpos[e.grip_qadr] = e._finger_q(NOMINAL_OPEN)
    mujoco.mj_forward(m, d)
    hand = e._subtree(m, m.body(S.HAND_ROOT).id)
    site = m.site("grasp").id
    Rs, ps = d.site_xmat[site].reshape(3, 3), d.site_xpos[site].copy()
    geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] in hand and (m.geom_contype[g] or m.geom_conaffinity[g])]
    chunks, areas = [], []
    for g in geoms:
        R, p = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g]
        t, sz = m.geom_type[g], m.geom_size[g]
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            mid = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            fc = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
            gen = lambda k, v=v, fc=fc: _tri_sample(v, fc, k, rng)                      # noqa: E731
            tri = v[fc]; area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1).sum()
        elif t == mujoco.mjtGeom.mjGEOM_BOX:
            gen = lambda k, sz=sz: box_points(sz[:3], k, rng)                              # noqa: E731
            area = 8 * (sz[0] * sz[1] + sz[0] * sz[2] + sz[1] * sz[2])
        elif t == mujoco.mjtGeom.mjGEOM_CYLINDER:
            gen = lambda k, sz=sz: cylinder_points(sz[0], sz[1], k, rng)                   # noqa: E731
            area = 2 * np.pi * sz[0] * (2 * sz[1] + sz[0])
        elif t == mujoco.mjtGeom.mjGEOM_CAPSULE:
            gen = lambda k, sz=sz: cylinder_points(sz[0], sz[1] + sz[0], k, rng)           # noqa: E731 (approximation)
            area = 2 * np.pi * sz[0] * (2 * sz[1] + sz[0])
        else:
            continue
        chunks.append((gen, R, p)); areas.append(max(area, 1e-9))
    w = np.array(areas) / np.sum(areas)
    counts = rng.multinomial(n, w)
    pts = []
    for (gen, R, p), k in zip(chunks, counts):
        if k:
            world = gen(k) @ R.T + p
            pts.append((world - ps) @ Rs)                                   # into the grasp frame (site frame)
    pts = np.concatenate(pts).astype(np.float32)
    if cache:
        f.parent.mkdir(exist_ok=True)
        np.savez(f, pts=pts, n_geoms=len(geoms), site_z=float(m.site_pos[site][2]))
    return pts


# ------------------------------------------------------------------ scene
def static_cloud(top_z, n_floor=1200, n_wall=800, n_ceil=500, seed=1):
    """floor board top, back wall (front face) and the virtual arm ceiling (a thin slab at the tallest top + margin)"""
    rng = np.random.default_rng(seed)
    z_floor = ZB
    floor = np.stack([rng.uniform(XF, XB, n_floor), rng.uniform(-HW, HW, n_floor), np.full(n_floor, z_floor)], 1)
    wall = np.stack([np.full(n_wall, XB), rng.uniform(-HW, HW, n_wall), rng.uniform(ZB, ZB + 0.45, n_wall)], 1)
    zc = top_z + CEILING_MARGIN
    ceil = np.stack([rng.uniform(XF, XB, n_ceil), rng.uniform(-HW, HW, n_ceil), np.full(n_ceil, zc)], 1)
    return np.concatenate([floor, wall, ceil]).astype(np.float32)


def scene_cloud(objects, cells, n_obj=450, seed=0, n_total=8192, with_static=True):
    """objects: [{id, shape, size, height}], cells: {id: (row, col)}.  -> (n_total, 3) float32 (static points + object points, resampled to n_total)"""
    rng = np.random.default_rng(seed)
    pts = [static_cloud(max(ZB + o["height"] for o in objects))] if with_static else []
    for o in objects:
        x, y = cell_xy(cells[o["id"]])
        h, s = o["height"], o["size"]
        p = cylinder_points(s, h / 2, n_obj, rng) if o["shape"] == "cylinder" else box_points((s, s, h / 2), n_obj, rng)
        pts.append(p + np.array([x, y, ZB + h / 2]))
    P = np.concatenate(pts).astype(np.float32)
    if len(P) >= n_total:
        return P[rng.choice(len(P), n_total, replace=False)]
    return P[np.concatenate([np.arange(len(P)), rng.integers(0, len(P), n_total - len(P))])]


def grasps_of(obj, cell):
    """the 32 grasps of the target standing in `cell`: positions [32,3] (pad centre) and rotations [32,3,3]"""
    A, S = teacher()
    x, y = cell_xy(cell)
    g = S.grasp_poses(dict(pos=np.array([x, y, ZB]), height=obj["height"], size=obj["size"], shape=obj["shape"]), ZB)
    return np.stack([q["position"] for q in g]).astype(np.float32), np.stack([q["R"] for q in g]).astype(np.float32), [(q["azimuth_deg"], q["grasp_level"]) for q in g]


if __name__ == "__main__":
    pc = gripper_cloud(cache=False)
    print("gripper cloud", pc.shape, "extent (m)", np.ptp(pc, 0).round(3), "min", pc.min(0).round(3), "max", pc.max(0).round(3))
    objs = [dict(id="t0", shape="cylinder", size=0.017, height=0.15), dict(id="o00", shape="box", size=0.02, height=0.10)]
    sc = scene_cloud(objs, {"t0": (3, 3), "o00": (2, 2)})
    print("scene cloud", sc.shape, "min", sc.min(0).round(3), "max", sc.max(0).round(3))
    pos, R, meta = grasps_of(objs[0], (3, 3))
    print("grasps", pos.shape, R.shape, meta[:3], "pad centre z", pos[0].round(3))
