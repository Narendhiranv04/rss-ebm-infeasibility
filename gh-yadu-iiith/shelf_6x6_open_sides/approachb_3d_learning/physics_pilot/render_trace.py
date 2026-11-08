"""Render a bwlphys.py trace (exact simulator states recorded on the physics host) to a GIF.  NO simulation: each frame sets qpos and calls
mj_forward on the same pinned scene model, so the frames show the verified run itself.

    MUJOCO_GL=glfw python render_trace.py REPO_SHELF_DIR TRACE.npz OUT.gif "caption" """
import json, sys
from pathlib import Path

import numpy as np

repo, trace, out, caption = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
src = repo / "dataset_bd_classquota_without_home_position_no_overpass"
sys.path.insert(0, str(src / "code/shelf66_v3"))
import bd_shelf  # noqa: F401,E402
import bd_3dt as A  # noqa: E402
import mujoco  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

z = np.load(trace, allow_pickle=False)
rec = json.loads((src / f"raw/roots/{str(z['root'])}.json").read_text())
e = A.build(rec["objects"], rec["initial_cells"])
r = mujoco.Renderer(e.model, 360, 480)
c = e.scene["camera"]
cam = mujoco.MjvCamera(); cam.lookat[:] = c["lookat"]; cam.distance, cam.azimuth, cam.elevation = c["distance"] * 0.9, c["azimuth"], c["elevation"]
frames = []
for q in z["qpos"]:
    e.data.qpos[:] = q
    mujoco.mj_forward(e.model, e.data)
    r.update_scene(e.data, cam)
    im = Image.fromarray(r.render().copy())
    ImageDraw.Draw(im).text((6, 6), caption, fill=(255, 255, 255))
    frames.append(im)
frames[0].save(out, save_all=True, append_images=frames[1:], duration=50, loop=0)
print(json.dumps({"frames": len(frames), "gif": str(out), "mujoco_render_version": mujoco.__version__, "verdict": str(z["verdict"])}))
