"""GIFs + images for a bd_3dt (open tabletop) B/D output directory, and an index.html to browse them.

    python viz_3dt.py <out_dir> [root_id ...]          # e.g. python viz_3dt.py 3dt_20

For every root: before.png / after.png (top + oblique camera), and sequence.gif = the verified minimal B sequence (or, without one, the
target's own move if it plans) re-executed in MuJoCo with bd_3dt's own query() + run_plan(), captioned with the current move.  A root
with no verified repair gets the initial scene only.  Written to <out_dir>/viz/<root_id>/ and <out_dir>/viz/index.html."""
import html
import json
import os
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset-bd"))
import bd_3dt as A  # noqa: E402

W, H, EVERY = 480, 360, 60                    # gif frame size; one frame every EVERY physics steps (0.12 s)
CAMS = {"top": dict(lookat=[0.40, 0.0, 0.08], distance=1.15, azimuth=180, elevation=-89),
        "oblique": dict(lookat=[0.40, 0.0, 0.10], distance=1.25, azimuth=150, elevation=-35)}


def camera(c):
    cam = mujoco.MjvCamera()
    cam.lookat[:], cam.distance, cam.azimuth, cam.elevation = c["lookat"], c["distance"], c["azimuth"], c["elevation"]
    return cam


def snap(e, rnd, cam, text):
    rnd.update_scene(e.data, cam)
    im = Image.fromarray(rnd.render())
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 18], fill=(0, 0, 0))
    d.text((5, 3), text, fill=(255, 255, 255))
    return im


def stills(e, d, tag):
    for name, c in CAMS.items():
        rnd = mujoco.Renderer(e.model, 480, 640)
        rnd.update_scene(e.data, camera(c))
        Image.fromarray(rnd.render()).save(d / f"{tag}__{name}.png")
        rnd.close()


def one(out, r):
    rid, objs, cells0, goal = r["ids"]["root_id"], r["objects"], r["initial_cells"], r["goal"]
    d = out / "viz" / rid
    d.mkdir(parents=True, exist_ok=True)
    e = A.build(objs, cells0)
    A.settle(e, cells0)
    stills(e, d, "before")
    B = [b for b in r["labels"]["B"] if b["role"] == "minimal" and b["y"]]
    if B:
        steps, what = [tuple(x) for x in B[0]["sequence"]] + [("t0", goal)], f"verified K={r['verified_K']}"
    elif r["labels"]["original"]["y"]:
        steps, what = [("t0", goal)], "target only"
    else:
        return dict(root=rid, gif=None, note="no verified repair: initial scene only")
    rnd, cam, frames, cap = mujoco.Renderer(e.model, H, W), camera(CAMS["oblique"]), [], [""]
    step0, k = e.step, [0]

    def step(n=1):
        step0(n)
        k[0] += 1
        if k[0] % EVERY == 0:
            frames.append(snap(e, rnd, cam, cap[0]))
    e.step = step
    ok = True
    for j, (o, dst) in enumerate(steps):
        cap[0] = f"{rid}  step {j + 1}/{len(steps)}: {'TARGET' if o == 't0' else o} -> {tuple(dst)}"
        p, _ = A.query(e, o, dst)
        if p is None:
            cap[0] += "  (no plan)"
            frames.append(snap(e, rnd, cam, cap[0]))
            ok = False
            break
        e.run_plan(p)
    e.step = step0
    frames += [snap(e, rnd, cam, f"{rid}  done ({what})")] * 12
    rnd.close()
    frames = [f.quantize(colors=64, method=0) for f in frames]
    frames[0].save(d / "sequence.gif", save_all=True, append_images=frames[1:], duration=100, loop=0, optimize=True)
    stills(e, d, "after")
    return dict(root=rid, gif=f"{rid}/sequence.gif", note=what if ok else "replay did not plan")


def index(out, rows, recs):
    cards = []
    for r in recs:
        rid, lab = r["ids"]["root_id"], r["labels"]
        row = rows.get(rid, {})
        seq = next((b["sequence"] for b in lab["B"] if b["role"] == "minimal" and b["y"]), None)
        media = (f'<img src="{row["gif"]}">' if row.get("gif") else f'<img src="{rid}/before__oblique.png">')
        info = [f"verified K: <b>{r['observed']}</b>", f"target {tuple(r['target_src'])} &rarr; goal {tuple(r['goal'])}",
                f"original feasible: {lab['original']['y']}", f"intended blockers: {', '.join(r['intended_blockers']) or '-'}",
                f"minimal sequence: {html.escape(str([(o, tuple(c)) for o, c in seq])) if seq else '-'}",
                f"spec: {html.escape(', '.join(f'{k}={v}' for k, v in r['spec'].items()))}", f"D queries {len(lab['D'])}, B records {len(lab['B'])}",
                f"<i>{html.escape(row.get('note', ''))}</i>",
                " ".join(f'<a href="{rid}/{t}__{c}.png">{t} {c}</a>' for t in ("before", "after") for c in CAMS if (out / "viz" / rid / f"{t}__{c}.png").exists())]
        cards.append(f'<div class="c"><h3>{rid}</h3>{media}<p>{"<br>".join(info)}</p></div>')
    (out / "viz" / "index.html").write_text(f"""<!doctype html><meta charset="utf-8"><title>{out.name} scenes</title>
<style>body{{font:13px sans-serif;margin:16px;background:#fafafa}} .g{{display:grid;grid-template-columns:repeat(auto-fill,minmax(500px,1fr));gap:14px}}
.c{{background:#fff;border:1px solid #ddd;padding:8px}} .c img{{width:100%}} h3{{margin:2px 0 6px;font-size:14px}}</style>
<h2>{out.name}: {len(recs)} scenes</h2><div class="g">{''.join(cards)}</div>""")


def main():
    out = Path(sys.argv[1]).resolve()
    recs = [json.loads(p.read_text()) for p in sorted((out / "roots").glob("*.json"))]
    want = set(sys.argv[2:])
    rows = {}
    for r in recs:
        if want and r["ids"]["root_id"] not in want:
            continue
        res = one(out, r)
        rows[res["root"]] = res
        print(res, flush=True)
    index(out, rows, [r for r in recs if not want or r["ids"]["root_id"] in want])
    print(out / "viz" / "index.html")


if __name__ == "__main__":
    main()
