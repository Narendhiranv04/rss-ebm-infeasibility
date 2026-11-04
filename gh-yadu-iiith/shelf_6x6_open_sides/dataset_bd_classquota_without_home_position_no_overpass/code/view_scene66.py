"""Replay one scene of this dataset in MuJoCo: the verified minimal repair (each blocker to its parking cell), then T to its goal.
Each move is planned with the dataset's own teacher query (bd_3dt.query: the scene's first feasible level grasp) from the exact
start state the teacher used, and executed in MuJoCo; the arm goes straight on to the next pick (no return to the ready pose).

    python view_scene66.py ../raw/roots/<root_id>.json              # MuJoCo viewer (needs a display)
    python view_scene66.py ../raw/roots/<root_id>.json --no-viewer  # headless: plan + execute, print every step
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "shelf66_full"))
import bd_shelf  # noqa: E402,F401  (paths + the 6 x 6 shelf environment)
import bd_3dt as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="a scene file, raw/roots/<root_id>.json")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="viewer playback speed (1 = real time)")
    a = ap.parse_args()
    r = json.load(open(a.root))
    B = next((b for b in r["labels"]["B"] if b["role"] == "minimal" and b["y"]), None)
    moves = ([(o, c) for o, c in B["sequence"]] + [("t0", r["goal"])]) if B else []
    print(f"{r['ids']['root_id']}: class {r['observed']}, T {r['target_src']} -> goal {r['goal']}, {len(r['objects']) - 1} obstacles; "
          f"moves: {moves or 'none (class NONE: no verified repair)'}", flush=True)
    e = A.build(r["objects"], r["initial_cells"])
    A.settle(e, r["initial_cells"])
    v = None
    if not a.no_viewer:
        from mujoco import viewer as mjv
        v = mjv.launch_passive(e.model, e.data)
        step0, dt = e.step, e.model.opt.timestep

        def step(n=1):
            step0(n)
            v.sync()
            time.sleep(n * dt / a.speed)
        e.step = step
    for j, (o, d) in enumerate(moves, 1):
        p, info = A.query(e, o, d)
        if p is None:
            sys.exit(f"step {j}: no plan for {o} -> {tuple(d)}")
        e.run_plan(p)
        print(f"step {j}/{len(moves)}: {o} -> {tuple(d)}, grasp {info['grasp_id']} (azimuth {info['azimuth']}); "
              f"{o} is now in cell {e.cell_of(o)}", flush=True)
    if v is not None:
        print("done; close the viewer window to exit", flush=True)
        while v.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
