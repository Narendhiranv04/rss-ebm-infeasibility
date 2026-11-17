"""Cost of the motion primitives on one root:  python scripts/probe_costs.py ROOT_ID"""
import json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2d import dataset as D, teacher as TT  # noqa: E402

rid = sys.argv[1]
rec = D.record(rid)
TT.boot(D.geom_of(rec))
m = TT.Motion(rec, TT.Shared())
cells = dict(m.cells0)


def timed(name, fn):
    t, v0 = time.time(), m.vcpu
    r = fn()
    print(f"{name:34s} wall {time.time() - t:6.2f}s  vcpu {m.vcpu - v0:6.2f}s", flush=True)
    return r


print(rid, "K", D.masks()[rid]["K"], "objects", len(cells), "goal", m.goal_cell, "t0 at", cells["t0"])
ok = timed("target(initial)", lambda: m.target(cells))
print("  target ok:", ok)
tr = timed("skill(t0 relaxed, only t0)", lambda: m.skill(cells, "t0", m.goal_cell))
print("  relaxed plan:", None if tr is None else len(tr["q"]), "waypoints; phases", None if tr is None else sorted(set(map(str, tr["phase"]))))
if tr is not None:
    bl = timed("blockers(t0 traj, all obstacles)", lambda: m.blockers(tr, "t0", cells["t0"], cells, m.movable))
    print("  blockers:", bl, " recorded minimal:", next((b["sequence"] for b in rec["labels"]["B"] if b["role"] == "minimal"), None))
    rec_sw = set().union(*map(set, rec["empty_scene_T_swept"].values()))
    for st in (0.05, 0.02, 0.01):
        sw = timed(f"swept sampled step {st}", lambda: m.swept(tr, "t0", cells["t0"], m.legal, step=st))
        print(f"  {len(sw)} cells; missing vs certified {sorted(rec_sw - sw)}; extra {sorted(sw - rec_sw)}")
    pk = timed("swept pick phases", lambda: m.swept(tr, "t0", cells["t0"], m.legal, phases=TT.PICK_PHASES))
    print("  pick-phase cells", sorted(pk))
o = m.movable[0]
free = [c for c in m.legal if c not in cells.values()]
r = timed(f"relocate({o} -> {free[0]})", lambda: m.relocate(cells, m.qref, o, free[0]))
print("  relocate:", None if r is None else ("ok", round(r[1], 3)))
if r is not None:
    c2 = {**cells, o: free[0]}
    timed("connector_ok(after)", lambda: m.connector_ok(c2, r[0]))
print("calls", m.calls, "object_checks", m.n_object_checks, "used", round(m.used(), 1))
