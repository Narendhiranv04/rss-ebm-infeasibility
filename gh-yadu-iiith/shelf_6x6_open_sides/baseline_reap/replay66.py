#!/usr/bin/env python3
"""Execute REAP plans on the 6x6 shelf exactly as the dataset teacher executes a sequence (bd_3dt.teach.seq):
start scene, then per move: plan from the arm's CURRENT pose (no return to ready) with the teacher's query, run it in
MuJoCo, and require the mover to end in its cell (|z err| <= 1 cm) with no other object displaced.

    python replay66.py <eval csv> <problem dir> [--out csv]

Columns: reap_solved, reap_moves (obstacle moves + target), executed (all moves ran, target at goal), and
excess = reap_moves - (verified K + 1) for executed plans of classes K0-K3.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from shelf66_oracle import _teacher  # noqa: E402

ACT = re.compile(r"\((pick|put) (cyl_\w+) (shelf_interior__\d+_\d+)\)")


def execute(side: dict, plan: str) -> dict:
    A = _teacher()
    S = A.env()["TS"]
    objs = [dict(id=o["id"], shape=o["shape"], size=o["size"], height=o["height"], color=o["color"]) for o in side["objects"]]
    cells0 = {o["id"]: tuple(o["cell"]) for o in side["objects"]}
    ids = {o["pddl"]: o["id"] for o in side["objects"]}
    cells = {k: tuple(v) for k, v in side["cells"].items()}
    moves = [(ids[o], cells[c]) for a, o, c in ACT.findall(plan) if a == "put"]
    e = A.build(objs, cells0)
    A.settle(e, cells0)
    done, err = [], ""
    for o, d in moves:
        before = {n: e.cell_of(n) for n in e.obj_index if n != o}
        p, _ = A.query(e, o, d)
        if p is None:
            err = f"{o}->{d}: no plan from the current pose"
            break
        try:
            e.run_plan(p)
        except S.GraspFailed as x:
            err = f"{o}->{d}: GRASP_FAILED {str(x)[:60]}"
            break
        h = next(x["height"] for x in objs if x["id"] == o)
        z = e.obj_pos(o)[2] - e.table_z - h / 2
        moved = sorted(n for n in before if e.cell_of(n) != before[n])
        if e.cell_of(o) != tuple(d) or abs(z) > 0.01 or moved:
            err = f"{o}->{d}: not verified end={e.cell_of(o)} z_err={z:.3f} displaced={moved}"
            break
        done.append(f"{o}->{d}")
    ok = not err and e.cell_of("t0") == tuple(side["goal_cell"])
    return dict(reap_moves=len(moves), executed=ok, n_executed=len(done), error=err, moves=" ".join(done))


def _one(job):
    r, problems = job
    stem = Path(str(r["problem"])).stem
    side = json.loads((Path(problems) / f"{stem}.json").read_text())
    row = dict(problem=stem, cls=side["cls"], verified_K=side["verified_K"], reap_solved=bool(r["solved"]),
               termination=r["termination_reason"], wall_time=r["wall_time"], mp_calls=r["mp_calls"], v0=r.get("v0"))
    if r["solved"]:
        try:
            row.update(execute(side, str(r["plan"])))
        except Exception as err:                              # set-up error: count as not executed, keep going
            row.update(reap_moves=None, executed=False, n_executed=0, error=f"{type(err).__name__}: {err}")
        if row["executed"] and side["cls"] != "NONE":
            row["excess"] = row["reap_moves"] - (side["verified_K"] + 1)
    print({k: v for k, v in row.items() if k != "moves"}, flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("problems")
    ap.add_argument("--out")
    ap.add_argument("--workers", type=int, default=1)
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    jobs = [(r.to_dict(), a.problems) for _, r in df.iterrows()]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers) as pool:
            rows = pool.map(_one, jobs, chunksize=1)
    else:
        rows = [_one(j) for j in jobs]
    out = pd.DataFrame(rows)
    out.to_csv(a.out or Path(a.csv).with_name(Path(a.csv).stem + "__replay.csv"), index=False)
    out["executed"] = out.get("executed", False).fillna(False).astype(bool)
    print("\nper class: REAP solved / executed in our env / mean excess moves (executed)")
    for cls, g in out.groupby("cls"):
        ex = g["excess"].dropna() if "excess" in g else []
        print(f"  {cls:>4}: {int(g.reap_solved.sum())}/{len(g)}  {int(g.executed.sum())}/{len(g)}  "
              f"{(ex.mean() if len(ex) else float('nan')):.2f}")


if __name__ == "__main__":
    main()
