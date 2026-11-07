"""Finger-width diagnostic for the STRUCTURE check (inherited SHELF66V3 behaviour, kept unchanged in V2: static_ck uses the default =
fully open fingers, while every OBJECT check uses the fingers pre-shaped to the grasp's open width).  For every nominal pose of every V2
mock query: the ik_candidates seed solutions that are structure-clear with fully open fingers vs with pre-shaped fingers, and the poses
that have NO structure-clear solution with open fingers but would have one pre-shaped (the only poses the conservatism can remove).
Diagnostic only: labels are unaffected.     python finger_diag.py SHARD NSHARD OUT.json"""
import gzip, json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement3d as P  # noqa: E402
S = P.S
shard, nshard, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
rows = [json.loads(x) for x in gzip.open(P.HERE.parent / "mock_v2/roster_v2.jsonl.gz", "rt")]
rows = [r for r in rows if not r["static"]][shard::nshard]
res = []
for r in rows:
    raw = json.loads((P.SRC / "raw/roots" / f"{r['root_id']}.json").read_text())
    sc = P.Scene(raw["objects"], r["cells"]); e = sc.e; ck = e.static_ck; pm = ck.m
    adr = np.array([pm.joint(j).qposadr[0] for j in S.GRIP_JOINTS]); q0 = ck.d.qpos[adr].copy()
    cnt = {"poses": 0, "sols": 0, "clear_open": 0, "clear_preshaped": 0, "poses_valid_open": 0, "poses_valid_preshaped": 0,
           "only_preshaped_poses": 0, "only_open_poses": 0}
    for g in P.poses(sc, r["object"]):
        p = np.round(np.asarray(g["position"], float), 4); R = g["R"]
        seeds = [e.q_home, e.q_ready]; base = math.atan2(p[1], p[0])
        for dq1 in (0.0, 0.5, -0.5):
            s = e.q_home.copy(); s[0] = base + dq1; seeds.append(s)
        rng = np.random.default_rng(7)
        for _ in range(12):
            s = e.q_home.copy(); s[0] = base; seeds.append(s + rng.normal(scale=0.9, size=7))
        vo = vp = 0
        for s in seeds:
            q = e.ik(p, R, s, iters=250, tol=1e-5)
            ep, er = e.pose_error(q, p, R)
            if not (ep < 1e-3 and er < 1e-2):
                continue
            cnt["sols"] += 1
            ck.margin = S.STATIC_MARGIN; ck.d.qpos[adr] = q0; o = not ck.collisions(q)
            ck.d.qpos[adr] = e._finger_q(g["open_width"]); pre = not ck.collisions(q); ck.d.qpos[adr] = q0
            cnt["clear_open"] += o; cnt["clear_preshaped"] += pre; vo += o; vp += pre
        cnt["poses"] += 1; cnt["poses_valid_open"] += vo > 0; cnt["poses_valid_preshaped"] += vp > 0
        cnt["only_preshaped_poses"] += vo == 0 and vp > 0; cnt["only_open_poses"] += vo > 0 and vp == 0
    res.append({"qid": r["qid"], "cohort": r["cohort"], **cnt}); print(json.dumps(res[-1]), flush=True)
Path(out).write_text(json.dumps(res, indent=1))
