"""Execute the ALTERNATIVE minimal sets of the counted scenes (the 2nd / 3rd minimal set the teacher found in its T-only lattice but never ran).

The teacher (bd_3dt.teach) stops at the first executed repair of the minimal size K.  Its T-only lattice (labels.blocker.minimal_sets) already lists every
minimal set of the scene (e.g. {o00,o01} and {o00,o02}), but only one of them was executed and stored in labels.B.  This pass takes every OTHER minimal set
of a scene, finds an order in which its relocations execute in MuJoCo (all orders are tried, up to MAX_ORDERS), then moves T to its goal, and stores the
result next to the original root, never inside it:

    <out>/roots/<root_id>.json      {"alt": [{"set", "sequence", "y", "independent", "orders_tried", "steps", "witnesses", ...}, ...], ...}
    <out>/witnesses/<aa>/<sha>.npz  same content-addressed trajectories as raw/witnesses

Same teacher, same parking cells (the first of each object's stored list, as in the lattice), same checks as the dataset: every relocation ends in its cell,
nothing else moves, T ends at its goal; and bd_3dt_plus.independent() (each object moves once, in any order) is recorded for the set.

    python shelf66_v3/alt_sets66.py --out ../raw_alt_sets --workers 12 [--only id1,id2] [--classes 1,2,3]
Resumable (scenes with a sidecar are skipped); touch <out>/STOP to stop cleanly.  Nothing in raw/ is modified."""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "dataset-bd"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

MAX_ORDERS = 6          # orders tried per set (3 relocations = 6 orders; 4 = 24, capped)
VERSION = "ALT_SETS66_V1"


def run_seq(A, bc, e, store, objs, cells0, goal, steps):
    """The teacher's seq(): execute the relocations in order, then T -> goal; every step verified (end cell, height, nothing else displaced)."""
    TS = A.env()["TS"]
    bc.check_sequence(steps)
    A.settle(e, cells0)
    info, wids = [], []
    for j, (o, d) in enumerate(list(steps) + [("t0", goal)]):
        before = {n: e.cell_of(n) for n in e.obj_index if n != o}
        p, qi = A.query(e, o, d)
        step = {"object": o, "dest": list(d), **qi}
        fail = None
        if p is None:
            fail = A.label(None, qi)
        else:
            wids.append(A.witness(store, p))
            try:
                e.run_plan(p)
            except TS.GraspFailed as err:
                fail = bc.binary(False, f"GRASP_FAILED:{str(err)[:80]}", "OTHER_FAILURE")
            if fail is None:
                h = next(x["height"] for x in objs if x["id"] == o)
                z = e.obj_pos(o)[2] - e.table_z - h / 2
                moved = sorted(n for n in before if e.cell_of(n) != before[n])
                step.update(pushed=p.get("pushed", []), end_cell=e.cell_of(o), z_err=float(z))
                if e.cell_of(o) != tuple(d) or abs(z) > 0.01 or moved:
                    fail = bc.binary(False, f"EXECUTION_NOT_VERIFIED:end={e.cell_of(o)} z_err={z:.3f} displaced={moved}", "OTHER_FAILURE")
        info.append({**step, "ok": fail is None})
        if fail is not None:
            return {**fail, "failed_step": j if o != "t0" else "target", "sequence": [[x, list(y)] for x, y in steps], "n_actions": len(steps) + 1,
                    "steps": info, "witnesses": wids}
    return bc.binary(True, "executed", failed_step=None, sequence=[[x, list(y)] for x, y in steps], n_actions=len(steps) + 1, steps=info, witnesses=wids,
                     final_cells={n: list(e.cell_of(n)) for n in e.obj_index})


def do_one(arg):
    root_id, src, out = arg
    import bd_shelf  # noqa: F401  (points bd_3dt at the shelf scene before anything else uses it)
    import bd_3dt as A
    import bd_3dt_plus as P
    import bd_core as bc
    t0 = time.time()
    r = json.loads((Path(src) / "roots" / f"{root_id}.json").read_text())
    store = bc.Store(out)
    objs, goal = r["objects"], tuple(r["goal"])
    cells0 = {o: tuple(c) for o, c in r["initial_cells"].items()}
    park = {o: tuple(v[0]) for o, v in r["parking"].items() if v}
    mins = [sorted(s) for s in r["labels"]["blocker"]["minimal_sets"]]
    have = sorted(o for o, _ in next((b["sequence"] for b in r["labels"]["B"] if b["role"] == "minimal" and b["y"]), []))
    e = A.build(objs, cells0)
    alt = []
    for S in sorted((s for s in mins if s != have), key=lambda s: (len(s), s)):
        if any(o not in park for o in S):
            alt.append({"set": S, "y": 0, "reason": "NO_PARKING_CELL", "orders_tried": 0})
            continue
        ts = time.time()
        indep = bool(P.independent(objs, cells0, {o: park[o] for o in S}, S, goal))
        got, tried, why = None, 0, []
        for order in itertools.islice(itertools.permutations(S), MAX_ORDERS):
            tried += 1
            res = run_seq(A, bc, e, store, objs, cells0, goal, [(o, park[o]) for o in order])
            if res["y"]:
                got = res
                break
            why.append({"order": list(order), "reason": res["reason"][:90], "failed_step": res.get("failed_step")})
        rec = {"set": S, "size": len(S), "independent": indep, "orders_tried": tried, "wall_s": round(time.time() - ts, 1)}
        rec.update({**got, "role": "minimal_alt"} if got else {"y": 0, "role": "minimal_alt", "failures": why})
        alt.append(rec)
    doc = {"ids": r["ids"], "observed": r["observed"], "verified_K": r["verified_K"], "version": VERSION, "executed_minimal_set": have, "minimal_sets": mins,
           "parking_used": {o: list(v) for o, v in park.items()}, "alt": alt, "wall_s": round(time.time() - t0, 1),
           "budget": {"max_orders": MAX_ORDERS, "rrt_max_iters": r["budget"].get("rrt_max_iters")}}
    store.save_root({**doc, "ids": r["ids"]})
    return {"root_id": root_id, "observed": r["observed"], "alt_sets": len(alt), "alt_ok": sum(a["y"] for a in alt), "wall_s": doc["wall_s"]}


def todo_ids(classes):
    sel = json.loads((DS / "selected.json").read_text())["selected"]
    ids = []
    for k in classes:
        ids += sel[k]
    out = []
    for i in ids:
        p = DS / "raw" / "roots" / f"{i}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text())
        b = r["labels"].get("blocker") or {}
        if b.get("blocker_supervision_valid") and len(b.get("minimal_sets") or []) >= 2:
            out.append((i, len(b["minimal_sets"])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DS / "raw_alt_sets"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    ap.add_argument("--classes", default="1,2,3")
    ap.add_argument("--list", action="store_true", help="only list the scenes this pass covers")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    cand = [(i, n) for i, n in todo_ids(a.classes.split(",")) if not only or i in only]
    if a.list:
        print(len(cand), "scenes with >= 2 minimal sets;", sum(n - 1 for _, n in cand), "alternative sets to execute")
        return
    done = {p.stem for p in (out / "roots").glob("*.json")} if (out / "roots").exists() else set()
    pend = [(i, str(DS / "raw"), str(out)) for i, n in sorted(cand, key=lambda x: -x[1]) if i not in done]
    print(f"{len(cand)} scenes, {len(done)} done, {len(pend)} to do, {a.workers} workers", flush=True)
    t_start, n_done, n_ok, n_alt = time.time(), 0, 0, 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        it = iter(pend)
        running = {}
        while True:
            for r in [r for r in running if r.ready()]:
                running.pop(r)
                try:
                    res = r.get()
                except Exception as ex:                                   # a scene that crashes is skipped, the run goes on
                    print("ERROR", type(ex).__name__, str(ex)[:200], flush=True)
                    continue
                n_done += 1
                n_ok += res["alt_ok"]
                n_alt += res["alt_sets"]
                with open(out / "ledger.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"[{n_done}/{len(pend)}] {res['root_id']} K{res['observed']}: alt sets executed {res['alt_ok']}/{res['alt_sets']}  {res['wall_s']}s   "
                      f"total {n_ok}/{n_alt}  elapsed {(time.time() - t_start) / 3600:.2f}h", flush=True)
            if not (out / "STOP").exists():
                while len(running) < a.workers:
                    x = next(it, None)
                    if x is None:
                        break
                    running[pool.apply_async(do_one, (x,))] = x[0]
            if not running:
                break
            time.sleep(2)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
