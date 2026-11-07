"""Diagnostic search for the pick-inaccessible-but-held-placeable fixture (flagged auxiliary use of source B outcomes; fixture only).
Candidates: step-0 failures 'no feasible side grasp' of the SOURCE teacher in TRAIN roots outside the mock roster.  Confirm:
(1) our placement label 1 (reference_v1); (2) for every pose, every held IK branch: shelf_scene._plan_approach (hover chain + joint-space
free move from the ready pose, RRT 5000, fingers pre-shaped) fails => the pick is inaccessible under the source's own pick policy."""
import gzip, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement3d as P, policy3d as Q, plan3d as L  # noqa: E401

shard, nshard, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
man = json.loads((P.HERE.parent / "mock_v1/split_manifest.json").read_text())
mock = {json.loads(x)["root_id"] for x in gzip.open(P.HERE.parent / "mock_v1/roster.jsonl.gz", "rt")}
roots = L.load_selected(P.SRC)
cand = []
MODE = sys.argv[4] if len(sys.argv) > 4 else "front_row"
for r in sorted(roots, key=lambda r: L.u("f1", r)):
    if man["split"][r] != "train" or r in mock:
        continue
    cells = roots[r]["cells"]
    occ = {tuple(c) for c in cells.values()}
    for o, c in sorted(cells.items()):
        if o == "t0" or c[0] != 0:
            continue                                            # constructed query: a FRONT-ROW obstacle to the sha-first free near/mid cell
        free = sorted(((rr, cc) for rr in range(1, 3) for cc in range(6) if (rr, cc) not in occ), key=lambda x: L.u("f1dst", r, o, x))
        if free:
            cand.append((r, o, free[0]))
seen, res = set(), []
cand = [c for c in cand if not (c in seen or seen.add(c))][shard::nshard]
print(len(cand), "candidates in shard", flush=True)
for r, o, d in cand[:30]:
    raw = json.loads((P.SRC / "raw/roots" / f"{r}.json").read_text())
    if Q.static_reason({k: tuple(v) for k, v in raw["initial_cells"].items()}, o, d):
        continue
    sc = P.Scene(raw["objects"], raw["initial_cells"])
    t = time.process_time()
    recq, w = Q.evaluate(sc, o, d, Q.POLICIES["adaptive_v1"])
    acc, mode_acc = None, None
    if recq["label"] == 1:
        e = sc.e
        g = Q.ranked(P.poses(sc, o))[recq["verified_rank"]]
        P.CUR["iters"] = 5000                                   # source pick policy (free-move RRT 5000), not the last stage budget
        sc.restore(); e._focus(e.obj_id(o)); e._check_width = g["open_width"]; mode_acc = []
        try:                                                    # the VERIFIED pose: is its pick approach feasible (any branch)?
            for b in range(e.n_branches):
                try:
                    e._plan_approach(g, e.q_ready, seed=0, branch=b); mode_acc.append(b); break
                except (RuntimeError, ValueError) as err:
                    mode_acc_reason = str(err)[:120]
        finally:
            e._check_width = None
        if not mode_acc:
            acc = []
            for g2 in Q.ranked(P.poses(sc, o)):                 # query level: any pose pick-accessible?
                sc.restore(); e._focus(e.obj_id(o)); e._check_width = g2["open_width"]
                try:
                    for b in range(e.n_branches):
                        try:
                            e._plan_approach(g2, e.q_ready, seed=0, branch=b); acc.append([g2["id"], b]); break
                        except (RuntimeError, ValueError):
                            pass
                finally:
                    e._check_width = None
                if acc:
                    break
    res.append({"root_id": r, "object": o, "dest": list(d), "label": recq["label"], "status": recq["status"], "approach_ok": acc, "verified_pose_approach_ok": mode_acc, "verified_pose_reason": (locals().get("mode_acc_reason") if mode_acc == [] else None),
                "cpu": time.process_time() - t, "witness_id": recq.get("witness_id"), "verified_rank": recq["verified_rank"]})
    print(json.dumps(res[-1]), flush=True)
    Path(out).write_text(json.dumps(res, indent=1))
    if recq["label"] == 1 and mode_acc == []:
        print("FOUND_MODE_LEVEL", "QUERY_LEVEL" if acc == [] else "", flush=True)
        if acc == []:
            break
