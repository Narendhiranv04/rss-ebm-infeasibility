"""Validate xarm7_g2.xml against the menagerie xarm7.xml (standard gripper).

    <env3d python> -I check_g2.py          (1 process, CPU, ~1 min)

Loads each model the way shelf_scene.py does (MjSpec.from_file + attach into a scene), checks the names the sim uses,
then sweeps the gripper actuator 0 (open) -> 255 (closed) with the arm held at the 'home' keyframe and reports, in the
hand frame (xarm_gripper_base_link = flange frame, z = approach): pad gap, pad size, flange -> pad-centre distance,
and the outer size of the whole gripper (base + linkage) from its mesh vertices.
"""
from pathlib import Path

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
MODELS = {"standard (menagerie)": HERE.parents[1] / "scene/mujoco_menagerie/ufactory_xarm7/xarm7.xml",
          "G2": HERE / "xarm7_g2.xml"}
HAND = "xarm_gripper_base_link"
GRIP_BODIES = [HAND, "left_outer_knuckle", "left_finger", "left_inner_knuckle",
               "right_outer_knuckle", "right_finger", "right_inner_knuckle"]
NAMES = {  # names shelf_scene.py / xarm7_cabinet_env.py look up
    "body": ["link_base", "link5", "link6", "link7"] + GRIP_BODIES,
    "joint": [f"joint{i}" for i in range(1, 8)] + [f"{s}_{j}" for s in ("left", "right")
                                                   for j in ("driver_joint", "finger_joint", "inner_knuckle_joint")],
    "actuator": [f"act{i}" for i in range(1, 8)] + ["gripper"],
    "geom": [f"{s}_finger_pad_{k}" for s in ("left", "right") for k in (1, 2)],
    "site": ["link_tcp"],
}
PADS = [f"{s}_finger_pad_{k}" for s in ("left", "right") for k in (1, 2)]


def load(path):
    scene = mujoco.MjSpec.from_string("<mujoco><worldbody/></mujoco>")
    scene.attach(mujoco.MjSpec.from_file(str(path)), frame=scene.worldbody.add_frame(), prefix="")
    return scene.compile()


def check_names(m):
    get = {"body": m.body, "joint": m.joint, "actuator": m.actuator, "geom": m.geom, "site": m.site}
    missing = []
    for kind, names in NAMES.items():
        for n in names:
            try:
                get[kind](n)
            except KeyError:
                missing.append(f"{kind}:{n}")
    return missing


def hand_frame(m, d):
    b = m.body(HAND).id
    return d.xpos[b].copy(), d.xmat[b].reshape(3, 3).copy()


def gripper_extent(m, d, p0, R):
    """Min/max (hand frame) over all mesh vertices of the gripper bodies' geoms."""
    ids = {m.body(n).id for n in GRIP_BODIES}
    pts = []
    for g in range(m.ngeom):
        if m.geom_bodyid[g] in ids and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            k = m.geom_dataid[g]
            v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
            pts.append(d.geom_xpos[g] + v @ d.geom_xmat[g].reshape(3, 3).T)
    P = (np.vstack(pts) - p0) @ R
    return P.min(0), P.max(0)


def base_extent(m):
    g = next(g for g in range(m.ngeom) if m.geom_bodyid[g] == m.body(HAND).id)
    k = m.geom_dataid[g]
    v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, m.geom_quat[g])
    P = m.geom_pos[g] + v @ R.reshape(3, 3).T                   # in the hand frame
    return P.min(0), P.max(0)


def sweep(m):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.keyframe("home").id)
    act = m.actuator("gripper").id
    pads = [m.geom(n).id for n in PADS]
    rows = []
    for ctrl in np.linspace(0, 255, 6):
        d.ctrl[act] = ctrl
        for _ in range(int(1.5 / m.opt.timestep)):
            mujoco.mj_step(m, d)
        p0, R = hand_frame(m, d)
        c = (d.geom_xpos[pads] - p0) @ R                       # pad centres, hand frame
        thick = 2 * m.geom_size[pads[0]][1]
        gap = np.linalg.norm(d.geom_xpos[pads[0]] - d.geom_xpos[pads[2]]) - thick   # same formula as the sim
        lo, hi = gripper_extent(m, d, p0, R)
        own = {m.body(n).id for n in GRIP_BODIES}
        ncon = sum(1 for i in range(d.ncon) if m.geom_bodyid[d.contact[i].geom1] in own
                   and m.geom_bodyid[d.contact[i].geom2] in own)
        rows.append(dict(ctrl=ctrl, gap=gap, pad_z=c[:, 2].mean(), lo=lo, hi=hi, self_contacts=ncon))
    return rows


def main():
    res = {}
    for label, path in MODELS.items():
        m = load(path)
        missing = check_names(m)
        assert not missing, f"{label}: missing names {missing}"
        sz = m.geom_size[m.geom("left_finger_pad_1").id]
        z1, z2 = (m.geom_pos[m.geom(f"left_finger_pad_{k}").id][2] for k in (1, 2))
        rows = sweep(m)
        res[label] = (rows, base_extent(m))
        print(f"\n== {label}: {path}\n   names OK ({sum(map(len, NAMES.values()))} checked); nbody={m.nbody} "
              f"nmesh={m.nmesh} njnt={m.njnt} nu={m.nu}")
        print(f"   pad box (each of 2 per finger): {2e3 * sz[0]:.1f} wide x {2e3 * sz[2]:.1f} long x "
              f"{2e3 * sz[1]:.1f} thick mm; both pads span {1e3 * (z2 - z1 + 2 * sz[2]):.1f} mm along the finger")
        lo, hi = res[label][1]
        print(f"   base housing alone (hand frame, mm): x {1e3 * lo[0]:.1f}..{1e3 * hi[0]:.1f}  "
              f"y {1e3 * lo[1]:.1f}..{1e3 * hi[1]:.1f}  z {1e3 * lo[2]:.1f}..{1e3 * hi[2]:.1f}")
        print("   ctrl  gap_mm  flange->pad_mm  whole-gripper width y / depth x / length z (mm)  self-contacts")
        for r in rows:
            w = 1e3 * (r["hi"] - r["lo"])
            print(f"   {r['ctrl']:5.0f}  {1e3 * r['gap']:6.1f}  {1e3 * r['pad_z']:14.1f}  "
                  f"{w[1]:6.1f} / {w[0]:5.1f} / {1e3 * r['hi'][2]:5.1f}{'':22s}{r['self_contacts']}")
    # self-check: G2 keeps the standard linkage, so the gap sweep must match within 1 mm
    a, b = ([r["gap"] for r in res[k][0]] for k in MODELS)
    assert np.allclose(a, b, atol=1e-3), "G2 gap sweep differs from standard: linkage changed?"
    print("\nself-check passed: G2 gap sweep matches the standard linkage within 1 mm")


if __name__ == "__main__":
    main()
