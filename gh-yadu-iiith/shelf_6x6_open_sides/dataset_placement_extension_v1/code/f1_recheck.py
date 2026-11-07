"""Re-check f1 mode-level candidates with the source free-move budget pinned to 5000 (bug fix: it had inherited the last stage budget)."""
import glob, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement3d as P, policy3d as Q, fixtures3d as F  # noqa: E401
shard, nshard = int(sys.argv[1]), int(sys.argv[2])
c = []
for f in sorted(glob.glob(str(P.HERE.parent / "mock_v1/dev/f1[bc]_shard*.log"))):
    for line in Path(f).read_text().splitlines():
        if line.startswith("{") and '"label": 1' in line:
            d = json.loads(line); c.append((d["root_id"], d["object"], tuple(d["dest"])))
c = sorted(set(c))[shard::nshard]
for rid, o, d in c:
    raw = json.loads((P.SRC / "raw/roots" / f"{rid}.json").read_text())
    sc = P.Scene(raw["objects"], raw["initial_cells"])
    rec, _ = Q.evaluate(sc, o, d, Q.POLICIES["reference_v1"], early_stop=False)
    ver = [m["rank"] for m in rec["modes"] if m["state"] == "VERIFIED"]
    first = rec["verified_rank"]
    blocked = first is not None and F.pose_pick_access(sc, o, first) == []
    out = {"root_id": rid, "object": o, "dest": list(d), "label": rec["label"], "verified_rank": first, "n_verified": len(ver),
           "verified_pose_pick_blocked": blocked}
    print(json.dumps(out), flush=True)
