#!/usr/bin/env python3
"""PIGINet's image tokens for the shelf dataset: render every scene state in MuJoCo (the dataset's own scene, xArm7 at its ready
pose), crop each object with the segmentation render, embed the crops with the frozen CLIP ViT-B/32 (PIGINet's encoder).

    MUJOCO_GL=egl python render_feats.py [--workers 5] [--preview <root id>]

States per root: "orig" (the scene as it is; used by the original query and every B plan) and one per D query (the teacher's
edited scene: obstacles teleported to parking cells).  Output: data/feats/<root id>.pkl = {state: {object id: float16[512]}},
data/text.npz (CLIP text features of piginet.PHRASES).  Resumable (existing files are skipped).  One camera (PIGINet's Fridges
set-up); an object with < 20 visible pixels gets a grey crop, as in Sajid's port.
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
V3_DIR = Path(os.environ.get("SHELF66_V3_DIR", Path.home() / "codebase/bd_samples/shelf66_v3"))
FEATS = HERE / "data" / "feats"
W, H = 640, 480
CAM = dict(lookat=[0.42, 0.0, 0.64], distance=1.2, azimuth=90.0, elevation=-40.0)       # through the open side, under the top board
_G = {}


def teacher():
    if "A" not in _G:
        sys.path.insert(0, str(V3_DIR))
        import bd_shelf  # noqa: F401
        import bd_3dt as A
        import mujoco
        _G["A"], _G["mujoco"], _G["S"] = A, mujoco, A.env()["TS"]
    return _G["A"], _G["mujoco"]


def clip():
    if "clip" not in _G:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        torch.set_num_threads(int(os.environ.get("CLIP_THREADS", "1")))
        name = "openai/clip-vit-base-patch32"
        _G["clip"] = CLIPModel.from_pretrained(name).eval()
        _G["proc"] = CLIPProcessor.from_pretrained(name)
        _G["torch"] = torch
    return _G["clip"], _G["proc"], _G["torch"]


def _emb(out):
    torch = _G["torch"]
    if torch.is_tensor(out):
        return out
    for a in ("image_embeds", "text_embeds", "pooler_output"):
        if getattr(out, a, None) is not None:
            return getattr(out, a)
    raise RuntimeError("no embedding in the CLIP output")


def image_feats(crops):
    m, p, torch = clip()
    out = []
    with torch.no_grad():
        for k in range(0, len(crops), 64):
            px = p(images=list(crops[k:k + 64]), return_tensors="pt")["pixel_values"]
            out.append(_emb(m.get_image_features(pixel_values=px)).float().numpy())
    return np.concatenate(out).astype(np.float16)


def text_feats(phrases):
    m, p, torch = clip()
    with torch.no_grad():
        tok = p(text=list(phrases), return_tensors="pt", padding=True)
        return _emb(m.get_text_features(**tok)).float().numpy()


def settle(e, cells, mujoco):
    """Scene at `cells` (as shelf66_oracle._settle: movers are parked off the grid first, so swaps cannot collide)."""
    e.reset()
    movers = [o for o, c in cells.items() if e.cell_of(o) != tuple(c)]
    for k, o in enumerate(movers):
        a = e.obj_qadr[e.obj_id(o)]
        e.data.qpos[a:a + 3] = [5.0 + k, 5.0, e.table_z + 0.1]
    mujoco.mj_forward(e.model, e.data)
    for o in movers:
        e.teleport(o, tuple(cells[o]))
    mujoco.mj_forward(e.model, e.data)


def render(e, mujoco, r):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = CAM["lookat"]
    cam.distance, cam.azimuth, cam.elevation = CAM["distance"], CAM["azimuth"], CAM["elevation"]
    r.disable_segmentation_rendering()
    r.update_scene(e.data, cam)
    rgb = r.render().copy()
    r.enable_segmentation_rendering()
    r.update_scene(e.data, cam)
    seg = r.render()[..., 0].copy()
    return rgb, seg


def crops_of(e, rgb, seg, size=224):
    from PIL import Image
    out = {}
    for i, o in enumerate(e.objects):
        m = np.isin(seg, e._obj_geoms[i])
        ys, xs = np.where(m)
        if len(ys) < 20:
            out[o["name"]] = np.full((size, size, 3), 128, np.uint8)
            continue
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        cy, cx, half = (y0 + y1) // 2, (x0 + x1) // 2, max(y1 - y0, x1 - x0) // 2 + 4
        patch = rgb[max(0, cy - half):cy + half + 1, max(0, cx - half):cx + half + 1]
        out[o["name"]] = np.array(Image.fromarray(patch).resize((size, size), Image.BICUBIC))
    return out


def do_root(job):
    rid, objects, states = job
    out_f = FEATS / f"{rid}.pkl"
    if out_f.exists():
        return rid, 0, 0.0
    t0 = time.time()
    A, mujoco = teacher()
    e = A.build(objects, states["orig"])
    r = mujoco.Renderer(e.model, H, W)
    keys, crops = [], []
    for st, cells in states.items():
        settle(e, cells, mujoco)
        rgb, seg = render(e, mujoco, r)
        for o, c in crops_of(e, rgb, seg).items():
            keys.append((st, o))
            crops.append(c)
    r.close()
    f = image_feats(crops)
    res = {}
    for (st, o), v in zip(keys, f):
        res.setdefault(st, {})[o] = v
    tmp = out_f.with_suffix(".tmp")
    pickle.dump(res, open(tmp, "wb"))
    tmp.rename(out_f)
    return rid, len(crops), time.time() - t0


def jobs():
    d = pickle.load(open(HERE / "data" / "samples.pkl", "rb"))
    states = {}
    for s in d["samples"]:
        st = states.setdefault(s["rid"], {"orig": d["roots"][s["rid"]]["init"]})
        if s["kind"] == "D":
            st[f"D{s['idx']}"] = s["cells"]
    return [(rid, d["roots"][rid]["objects"], st) for rid, st in states.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--preview", help="root id: write preview_<state>.png + the crops, no features")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    J = jobs()
    if a.preview:
        from PIL import Image
        rid, objects, states = next(j for j in J if j[0] == a.preview)
        A, mujoco = teacher()
        e = A.build(objects, states["orig"])
        r = mujoco.Renderer(e.model, H, W)
        for st in list(states)[:2]:
            settle(e, states[st], mujoco)
            rgb, seg = render(e, mujoco, r)
            Image.fromarray(rgb).save(HERE / f"preview_{st}.png")
            c = crops_of(e, rgb, seg)
            Image.fromarray(np.concatenate(list(c.values()), 1)).save(HERE / f"preview_{st}_crops.png")
            print(st, {o: int(np.isin(seg, e._obj_geoms[i]).sum()) for i, o in enumerate(c)})
        return
    FEATS.mkdir(parents=True, exist_ok=True)
    if not (HERE / "data" / "text.npz").exists():
        from piginet import PHRASES
        np.savez(HERE / "data" / "text.npz", feats=text_feats(PHRASES))
    J = [j for j in J if not (FEATS / f"{j[0]}.pkl").exists()]
    if a.limit:
        J = J[:a.limit]
    print(f"{len(J)} roots to render, {sum(len(j[2]) for j in J)} states, {a.workers} workers", flush=True)
    import multiprocessing as mp
    t0, n, nc = time.time(), 0, 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=40) as pool:
        for rid, c, dt in pool.imap_unordered(do_root, J):
            n += 1
            nc += c
            if n % 50 == 0 or n == len(J):
                el = time.time() - t0
                print(f"{n}/{len(J)} roots, {nc} crops, {el / 60:.1f} min, ETA {el / n * (len(J) - n) / 60:.1f} min", flush=True)
    print("done")


if __name__ == "__main__":
    main()
