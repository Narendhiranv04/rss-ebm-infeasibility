"""Interactive 3D view of a scene of a bd_3dt B/D output directory (default 3dt_plus_raw) in the MuJoCo viewer: the initial scene,
then (after Enter in the terminal) its verified minimal B sequence and the target's move, re-executed in MuJoCo with bd_3dt's own
query() + run_plan(), as viz_3dt.py does.  Nothing is written to disk.

    python view_3dt.py                       # counted scenes per class
    python view_3dt.py K3 [n]                # the n-th (default 0) counted scene of class NONE / K0 .. K4
    python view_3dt.py <root_id>             # one scene by id
    options: --dir DIR (default 3dt_plus_raw)   --speed X (playback speed, default 1)   --no-window (replay without a viewer)

Viewer: drag to rotate, right-drag to move, scroll to zoom, double-click a body to select it; close the window to quit."""
import argparse
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("BD_RSS3D_ROOT", "/home/ayush/codebase/rss-ebm-infeasibility/gh-yadu-iiith")
sys.path.insert(0, str(HERE / "dataset-bd"))
import bd_3dt as A  # noqa: E402


def counted(out):
    """Root ids per class as the runner counts them: accepted roots in ledger order, up to the class quota."""
    led = [json.loads(x) for x in (out / "ledger.jsonl").read_text().splitlines() if x.strip()]
    quota = json.loads((out / "PROGRESS.json").read_text())["quota"]
    by = {k: [] for k in quota}
    for e in led:
        k = e.get("observed")
        if e.get("accepted") and k in by and len(by[k]) < quota[k]:
            by[k].append(e["root_id"])
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("which", nargs="?")
    ap.add_argument("n", nargs="?", type=int, default=0)
    ap.add_argument("--dir", default=str(HERE / "3dt_plus_raw"))
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--no-window", action="store_true")
    a = ap.parse_args()
    out = Path(a.dir)
    by = counted(out)
    if not a.which:
        for k, ids in by.items():
            print(f"{'NONE' if k == 'NONE' else 'K' + k}: {len(ids)} scenes, e.g. {', '.join(ids[:3])}")
        print("usage: python view_3dt.py K3 [n]   or   python view_3dt.py <root_id>")
        return
    w = a.which.upper()
    if w == "NONE" or (w.startswith("K") and w[1:].isdigit()):
        ids = by["NONE" if w == "NONE" else w[1:]]
        rid = ids[a.n]
    else:
        rid = a.which
    r = json.loads((out / "roots" / f"{rid}.json").read_text())
    objs, cells0, goal = r["objects"], r["initial_cells"], r["goal"]
    B = [b for b in r["labels"]["B"] if b["role"] == "minimal" and b["y"]]
    if B:
        steps = [tuple(x) for x in B[0]["sequence"]] + [("t0", goal)]
    elif r["labels"]["original"]["y"]:
        steps = [("t0", goal)]
    else:
        steps = []
    print(f"{rid}: verified K = {r['observed']}, target t0 {tuple(r['target_src'])} -> goal {tuple(goal)}, "
          f"intended blockers {', '.join(r['intended_blockers']) or '-'}")
    print("minimal sequence:", [(o, tuple(c)) for o, c in steps[:-1]] if B else "-" if not steps else "none needed (target moves directly)")
    e = A.build(objs, cells0)
    A.settle(e, cells0)
    v = None
    if not a.no_window:
        import mujoco.viewer
        v = mujoco.viewer.launch_passive(e.model, e.data)
        e.viewer = v
        v.sync()
    step0, dt = e.step, e.model.opt.timestep

    def paced(n=1):                                        # about real time / speed (the scene's own step runs flat out)
        t = time.perf_counter()
        step0(n)
        lag = n * dt / a.speed - (time.perf_counter() - t)
        if lag > 0 and v is not None:
            time.sleep(lag)
    e.step = paced

    def say(text):
        print(text, flush=True)
        if v is not None:
            v.set_texts((None, None, text, ""))
    if not steps:
        say(f"{rid}: no verified repair (class {r['observed']}): initial scene only")
    else:
        if v is not None:
            say(f"{rid}  K={r['observed']}: initial scene")
            input("press Enter here to replay the moves ")
        for j, (o, dst) in enumerate(steps):
            say(f"step {j + 1}/{len(steps)}: {'TARGET' if o == 't0' else o} -> {tuple(dst)}  (planning)")
            p, _ = A.query(e, o, dst)
            if p is None:
                say(f"step {j + 1}: no plan found")
                break
            say(f"step {j + 1}/{len(steps)}: {'TARGET' if o == 't0' else o} -> {tuple(dst)}")
            e.run_plan(p)
        else:
            say(f"{rid}  done: target at goal {tuple(goal)}")
    if v is not None:
        print("close the viewer window to quit")
        while v.is_running():
            time.sleep(0.1)


if __name__ == "__main__":
    main()
