"""Does the gripper cloud, placed at a grasp pose, penetrate the object the teacher says the hand would hit?  (frame / scale sanity check)"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
import build_scn_data as BG  # noqa: E402  (find_root, parse_reason)
import scn_geometry as G  # noqa: E402

find_root = BG.find_root


def inside(p, o, cell, margin=0.0):
    x, y = G.cell_xy(cell)
    q = p - np.array([x, y, G.ZB + o["height"] / 2])
    if o["shape"] == "cylinder":
        return (np.hypot(q[:, 0], q[:, 1]) < o["size"] + margin) & (np.abs(q[:, 2]) < o["height"] / 2 + margin)
    return (np.abs(q[:, 0]) < o["size"] + margin) & (np.abs(q[:, 1]) < o["size"] + margin) & (np.abs(q[:, 2]) < o["height"] / 2 + margin)


def main():
    split = json.loads(BG.SPLIT.read_text())
    rids = [r for c in ("1", "2", "3") for r in split["train"][c]][:150]
    pc = G.gripper_cloud()
    hit_named = hit_named_pen = other_pen = other_n = 0
    pen_pts = []
    for rid in rids:
        rec = json.loads(find_root(rid).read_text())
        objs = {o["id"]: o for o in rec["objects"]}
        init = {k: tuple(v) for k, v in rec["initial_cells"].items()}
        q = rec["labels"]["original"]
        gr = q.get("grasp_reasons") or {}
        if len(gr) != 32:
            continue
        pos, R, _ = G.grasps_of(objs["t0"], init["t0"])
        for g in range(32):
            kind = BG.parse_reason(gr[str(g)])
            world = pc @ R[g].T + pos[g]
            named = [o for o in kind[1]] if kind[0] == "obj" else []
            pen_any = [bool(inside(world, objs[o], init[o]).any()) for o in objs if o != "t0"]
            if named:
                hit_named += 1
                ok = any(inside(world, objs[o], init[o], 0.003).any() for o in named if o in objs)
                hit_named_pen += ok
                pen_pts.append(max(int(inside(world, objs[o], init[o], 0.003).sum()) for o in named if o in objs))
            elif kind[0] in ("ik", "struct", "other"):
                other_n += 1
                other_pen += any(pen_any)
    print(f"grasps the teacher says hit a named object: {hit_named}; our hand cloud penetrates one of them: {hit_named_pen} ({100 * hit_named_pen / max(1, hit_named):.1f}%)")
    print(f"  penetrating points per such grasp: median {np.median(pen_pts):.0f}, mean {np.mean(pen_pts):.1f}")
    print(f"grasps failing for another reason (unreachable / structure / goal): {other_n}; hand cloud penetrates some obstacle: {other_pen} ({100 * other_pen / max(1, other_n):.1f}%)")


if __name__ == "__main__":
    main()
