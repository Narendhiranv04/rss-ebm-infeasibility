"""Can each listed object be picked from its cell when it is ALONE on the empty shelf (SHELF66V3 planner)?  Read-only diagnostic.
argv: <shelf66_v3 dir> <jobs.json> <out.jsonl> <part k> <parts n>.  jobs: [{"tag", "shape", "size", "height", "cell"}]; this process
does jobs k, k+n, ...  Each object is planned pick -> carry -> place to a front cell (1,2) (or (1,3)), every grasp pose until one
plans; if none does and some pose failed for another reason than reach, a second destination (2,4) is tried.  One JSON line per job (resumable: tags already in <out> are skipped)."""
import json
import sys
import time
from pathlib import Path

HERE = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(HERE.parent / "dataset-bd"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import bd_shelf as D  # noqa: E402,F401
import bd_3dt as A  # noqa: E402

assert "shelf66_v3" in A.__file__, A.__file__
TS = A.env()["TS"]
jobs = json.loads(Path(sys.argv[2]).read_text())
out, k, n = Path(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
done = {json.loads(x)["tag"] for x in out.read_text().splitlines()} if out.exists() else set()


def try_pick(obj, cell, dst):
    e = A.build([obj], {"x": tuple(cell)})
    i = e.obj_id("x")
    e.objects[i]["pos"] = e.obj_pos(i)
    q0 = e.q_now()
    reasons, tried = {}, 0
    for g in TS.grasp_poses(e.objects[i], e.table_z):
        tried += 1
        p = e.plan_pick_place("x", tuple(dst), g, q_start=q0)
        if p["feasible"]:
            return g["azimuth_deg"], g.get("grasp_level"), tried, reasons
        reasons[f"{g['azimuth_deg']}/{g.get('grasp_level')}"] = p["reason"][:90]
    return None, None, tried, reasons


with out.open("a") as f:
    for j in jobs[k::n]:
        if j["tag"] in done:
            continue
        t0 = time.time()
        obj = {"id": "x", "shape": j["shape"], "size": float(j["size"]), "height": float(j["height"]), "color": "red"}
        cell = tuple(j["cell"])
        res = {"tag": j["tag"], "cell": list(cell), "shape": j["shape"], "size": j["size"], "height": j["height"]}
        az, lvl, tried, reasons = try_pick(obj, cell, (1, 2) if cell != (1, 2) else (1, 3))
        if az is None and cell != (2, 4) and not all("out of reach" in x for x in reasons.values()):
            az, lvl, tried2, reasons2 = try_pick(obj, cell, (2, 4))
            tried += tried2
        res.update(pickable=az is not None, azimuth=az, level=lvl, poses_tried=tried, s=round(time.time() - t0, 1))
        if az is None:
            res["reasons_sample"] = dict(list(reasons.items())[:6])
        f.write(json.dumps(res) + "\n")
        f.flush()
