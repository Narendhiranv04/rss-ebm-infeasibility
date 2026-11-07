"""Deterministic unit fixtures from TRAINING roots outside the mock roster (discovery + freeze), and the pick-access diagnostic.

  find OUT.json [--roots N] [--cpu-budget S]   evaluate dev-roster queries (reference policy) until every fixture kind is found
The pick-access diagnostic (never part of a label): a pose's PICK is certified inaccessible when the pose has no static-clear IK
branch, or every branch hits another object, or the straight approach from the hover pose (PRE_DIST back along the approach axis,
shelf_scene._plan_approach's chain) fails for every held branch.  A query is 'pick-inaccessible' when that holds for all 32 poses.
"""
import gzip, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import placement3d as P  # noqa: E402
import plan3d as L  # noqa: E402
import policy3d as Q  # noqa: E402

S = P.S
KINDS = ("pick_inaccessible_held_placeable", "obstructed_carry_certified", "budget_exhausted_negative", "target_goal_destination",
         "vacated_after_edit")


def pick_access(sc, obj):
    """-> (n poses whose pick approach is feasible, details).  Deterministic, no RRT (the free move to the hover pose is not even tried,
    so 'accessible' here is optimistic and 'inaccessible' is certified)."""
    e, ok = sc.e, []
    P.CUR["iters"] = 5000
    for g in Q.ranked(P.poses(sc, obj)):
        sc.restore(); e._focus(e.obj_id(obj)); e._check_width = g["open_width"]
        try:
            for q in [q for q in e.ik_candidates(g["position"], g["R"], e.q_ready) if not e._object_hits(q)]:
                try:
                    e.chain(q, g["position"], [g["position"] - g["approach"] * S.PRE_DIST], g["R"]); ok.append(g["id"]); break
                except RuntimeError:
                    pass
        finally:
            e._check_width = None
    return len(ok), ok


def pose_pick_access(sc, obj, rank):
    """Diagnostic: IK branches of the ranked pose for which shelf_scene._plan_approach (hover chain + RRT free move from the ready
    pose, fingers pre-shaped) succeeds.  Empty => this pose's pick is inaccessible under the source pick policy."""
    e, ok = sc.e, []
    P.CUR["iters"] = 5000                       # the SOURCE pick policy: free-move RRT 5000 iterations, never a placement stage budget
    g = Q.ranked(P.poses(sc, obj))[rank]
    for b in range(e.n_branches):
        sc.restore(); e._focus(e.obj_id(obj)); e._check_width = g["open_width"]
        try:
            e._plan_approach(g, e.q_ready, seed=0, branch=b); ok.append(b); break      # one accessible branch is enough
        except (RuntimeError, ValueError):
            pass
        finally:
            e._check_width = None
    return ok


def find(out, n_roots=40, budget=1500, shard=0, nshard=1):
    man = json.loads((P.HERE.parent / "mock_v1/split_manifest.json").read_text())
    mock = {json.loads(x)["root_id"] for x in gzip.open(P.HERE.parent / "mock_v1/roster.jsonl.gz", "rt")}
    roots = L.load_selected(P.SRC)
    dev = [r for r in sorted(roots, key=lambda r: L.u("fixture", r)) if man["split"][r] == "train" and r not in mock and roots[r]["K"] in "123"][:n_roots][shard::nshard]
    found, t0 = {}, time.process_time()
    for rid in dev:
        raw, ikc = json.loads((P.SRC / "raw/roots" / f"{rid}.json").read_text()), {}
        qs = [q for q in L.roster_for(roots[rid]) if not q["static"]]
        goal = tuple(raw["goal"])
        if goal not in {tuple(c) for c in roots[rid]["cells"].values()}:
            o = sorted(x for x in roots[rid]["cells"] if x != "t0")
            if o:
                qs.append({**qs[0], "qid": f"{rid}:goal", "depth": 0, "cells": {k: list(v) for k, v in roots[rid]["cells"].items()}, "object": o[0],
                           "dest": list(goal), "dest_is_t_goal": True})
        for q in qs:
            if time.process_time() - t0 > budget or len(found) == len(KINDS):
                break
            sc = P.Scene(raw["objects"], q["cells"], ikc)
            rec, _ = Q.evaluate(sc, q["object"], tuple(q["dest"]), Q.POLICIES["reference_v1"], early_stop=False)
            kind = None
            if rec["label"] == 1 and "pick_inaccessible_held_placeable" not in found and pick_access(sc, q["object"])[0] == 0:
                kind = "pick_inaccessible_held_placeable"
            elif rec["status"] == "NEG_CERTIFIED_GEOMETRY" and "obstructed_carry_certified" not in found:
                kind = "obstructed_carry_certified"
            elif rec["status"] == "NEG_UNDER_POLICY" and rec.get("negative_kind") == "BOUNDED_SEARCH" and "budget_exhausted_negative" not in found:
                kind = "budget_exhausted_negative"
            elif q.get("dest_is_t_goal") and rec["label"] is not None and "target_goal_destination" not in found:
                kind = "target_goal_destination"
            elif q["band"] == "vacated" and q["depth"] == 1 and rec["label"] is not None and "vacated_after_edit" not in found:
                kind = "vacated_after_edit"
            print(rid, q["qid"], rec["label"], rec["status"], rec.get("negative_kind"), kind, flush=True)
            if kind:
                found[kind] = {"root_id": rid, "qid": q["qid"], "cells": q["cells"], "object": q["object"], "dest": q["dest"], "depth": q["depth"],
                               "expect": {"label": rec["label"], "status": rec["status"], "negative_kind": rec.get("negative_kind"),
                                          "verified_rank": rec["verified_rank"], "witness_id": rec.get("witness_id"), "accounting": rec["accounting"]}}
    Path(out).write_text(json.dumps({"rule": "first dev-roster query (sha-ordered TRAIN roots outside the mock roster) showing each kind",
                                     "fixtures": found, "missing": [k for k in KINDS if k not in found]}, indent=1, sort_keys=True))
    print(json.dumps({"found": sorted(found), "missing": [k for k in KINDS if k not in found]}))


def freeze(out, logs, f1logs):
    """Choose per kind the found query of the sha-earliest root (deterministic), rebuild it, re-evaluate (reference_v1) -> fixtures.json."""
    roots = L.load_selected(P.SRC)
    pick = {}
    for f in logs:
        for line in Path(f).read_text().splitlines():
            t = line.split()
            if len(t) >= 6 and t[-1] in KINDS:
                rid, qid, kind = t[0], t[1], t[-1]
                if kind not in pick or L.u("fixture", rid) < L.u("fixture", pick[kind][0]):
                    pick[kind] = (rid, qid)
    f1 = []
    for f in f1logs:                                            # f1_recheck logs (source free-move RRT pinned to 5000)
        for line in Path(f).read_text().splitlines():
            if line.startswith("{") and '"verified_pose_pick_blocked": true' in line:
                f1.append(json.loads(line))
    f1.sort(key=lambda d: (d["n_verified"] > 1, L.u("f1", d["root_id"])))     # prefer: EVERY placement-verified pose pick-blocked
    fixtures = {}
    for kind, (rid, qid) in pick.items():
        if kind == "pick_inaccessible_held_placeable":
            continue
        raw = json.loads((P.SRC / "raw/roots" / f"{rid}.json").read_text())
        qs = {q["qid"]: q for q in L.roster_for(roots[rid])}
        if qid.endswith(":goal"):
            o = sorted(x for x in roots[rid]["cells"] if x != "t0")[0]
            q = {"qid": qid, "cells": {k: list(v) for k, v in roots[rid]["cells"].items()}, "object": o, "dest": list(raw["goal"]), "depth": 0}
        else:
            q = qs[qid]
        fixtures[kind] = {"root_id": rid, "qid": qid, "cells": q["cells"], "object": q["object"], "dest": q["dest"], "depth": q["depth"], "constructed": False}
    if f1:
        d = f1[0]
        raw = json.loads((P.SRC / "raw/roots" / f"{d['root_id']}.json").read_text())
        fixtures["pick_inaccessible_held_placeable"] = {"root_id": d["root_id"], "qid": f"{d['root_id']}:f1", "object": d["object"], "dest": d["dest"],
                                                        "cells": raw["initial_cells"], "depth": 0, "constructed": True,
                                                        "level": "mode (the verified pose's source pick approach -- hover chain + 5000-iteration free move -- fails on every IK branch)",
                                                        "diagnostic": {k: d[k] for k in ("verified_rank", "n_verified")}}
    for kind, f in fixtures.items():
        raw = json.loads((P.SRC / "raw/roots" / f"{f['root_id']}.json").read_text())
        sc = P.Scene(raw["objects"], f["cells"])
        rec, _ = Q.evaluate(sc, f["object"], tuple(f["dest"]), Q.POLICIES["reference_v1"], early_stop=False)
        f["expect"] = {"label": rec["label"], "status": rec["status"], "negative_kind": rec.get("negative_kind"), "verified_rank": rec["verified_rank"],
                       "witness_id": rec.get("witness_id"), "accounting": rec["accounting"]}
        print(kind, f["root_id"], f["object"], f["dest"], f["expect"]["label"], f["expect"]["status"], flush=True)
    Path(out).write_text(json.dumps({"rule": "per kind: the qualifying dev-roster query of the sha-earliest TRAIN root outside the mock roster; "
                                     "pick_inaccessible: f1_search (source-guided / front-row constructed queries), query-level preferred",
                                     "fixtures": fixtures, "missing": [k for k in KINDS if k not in fixtures]}, indent=1, sort_keys=True))


if __name__ == "__main__":
    if sys.argv[1] == "freeze":
        import glob
        freeze(sys.argv[2], sorted(glob.glob(sys.argv[3])), sorted(glob.glob(sys.argv[4])))
    elif sys.argv[1] == "merge":
        parts = [json.loads(Path(p).read_text()) for p in sys.argv[3:]]
        fx = {}
        for k in KINDS:                                            # deterministic: the shard holding the sha-earliest root wins
            c = [p["fixtures"][k] for p in parts if k in p["fixtures"]]
            if c:
                fx[k] = min(c, key=lambda f: L.u("fixture", f["root_id"]))
        Path(sys.argv[2]).write_text(json.dumps({"rule": parts[0]["rule"], "fixtures": fx, "missing": [k for k in KINDS if k not in fx]}, indent=1, sort_keys=True))
        print(sorted(fx), [k for k in KINDS if k not in fx])
    else:
        find(sys.argv[2], int(sys.argv[3]), shard=int(sys.argv[4]), nshard=int(sys.argv[5]))
