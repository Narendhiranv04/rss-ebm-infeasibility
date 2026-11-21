"""Physics check of an ordered plan with the pinned SHELF66V3 teacher (BD3DS66_TEACHER_V2), unmodified.

execute(raw, steps) mirrors bd_3dt.teach.seq() call for call: settle the initial scene at the ready pose, then for every move and finally
the target (t0 -> goal): query (first feasible level grasp, RRT 5000 iterations, seed 0) -> run_plan in MuJoCo -> the same checks
(mover ends in its cell, upright within 1 cm, no other object changed cell).  success = every step and the target pass (= X1 and Y1).

    python ../hybrid/src/physics.py selftest --n 12      re-run labelled VAL plans and compare with their labels (run from shelf_6x6_open_sides/)
"""
import argparse, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # shelf_6x6_open_sides/
sys.path.insert(0, str(ROOT / "dataset_bd_classquota_without_home_position_no_overpass/code/shelf66_v3"))
_E = {}


def _env(raw):
    import bd_shelf  # noqa: F401  (points bd_3dt at the shelf scene exactly as production does)
    import bd_3dt as A
    rid = raw["ids"]["root_id"]
    if rid not in _E:
        _E.clear()  # one scene env per worker at a time
        _E[rid] = A.build(raw["objects"], {k: tuple(v) for k, v in raw["initial_cells"].items()})
    return A, _E[rid]


def execute(raw, steps):
    A, e = _env(raw)
    cells0 = {k: tuple(v) for k, v in raw["initial_cells"].items()}
    A.settle(e, cells0)
    t0 = time.time()
    for j, (o, d) in enumerate(list(steps) + [("t0", tuple(raw["goal"]))]):
        before = {n: e.cell_of(n) for n in e.obj_index if n != o}
        p, qi = A.query(e, o, d)
        if p is None:
            return {"success": 0, "failed_step": j, "why": "NO_PLAN", "s": round(time.time() - t0, 1)}
        try:
            e.run_plan(p)
        except A.env()["TS"].GraspFailed as err:
            return {"success": 0, "failed_step": j, "why": f"GRASP_FAILED:{str(err)[:60]}", "s": round(time.time() - t0, 1)}
        h = next(x["height"] for x in raw["objects"] if x["id"] == o)
        z = e.obj_pos(o)[2] - e.table_z - h / 2
        moved = [n for n in before if e.cell_of(n) != before[n]]
        if e.cell_of(o) != tuple(d) or abs(z) > 0.01 or moved:
            return {"success": 0, "failed_step": j, "why": "EXECUTION_NOT_VERIFIED", "s": round(time.time() - t0, 1)}
    return {"success": 1, "failed_step": None, "why": "executed", "s": round(time.time() - t0, 1)}


def _job(args):
    rid, steps = args
    import xy3d_data as XD
    return execute(XD.load(rid), [(o, tuple(c)) for o, c in steps])


def selftest(n):
    sys.path.insert(0, str(ROOT / "approachb_3d_learning/src"))
    import xy3d_data as XD
    sc = json.loads((ROOT / "approachb_3d_learning/runs/XY3D_REAL/BD3DS66_XY_SIDECAR_V1.json").read_text())
    y = {(r["root_id"], r["y_key"]): r["y"] for r in sc["y_rows"]}
    split = XD.qc_roots()
    rows = []
    for r in sc["x_rows"]:
        if split.get(r["root_id"]) == "val":
            yy = y.get((r["root_id"], r["y_key"]))
            s = 0 if r["x"] == 0 else (None if yy is None else int(yy == 1))
            if s is not None:
                rows.append((r, s))
    random.Random(0).shuffle(rows)
    pick = [x for x in rows if x[1] == 1][: n // 2] + [x for x in rows if x[1] == 0][: n - n // 2]
    agree = 0
    for r, s in pick:
        steps = [(m.split(">")[0], tuple(map(int, m.split(">")[1].split(",")))) for m in r["x_key"].split(".")]
        res = execute(XD.load(r["root_id"]), steps)
        agree += res["success"] == s
        print(json.dumps({"root": r["root_id"], "plan": r["x_key"], "label": s, "physics": res}), flush=True)
    print(json.dumps({"agree": agree, "of": len(pick)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest"]); ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "approachb_3d_learning/src"))
    selftest(a.n)
