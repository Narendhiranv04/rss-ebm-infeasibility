"""Cost of screening plans vs the executable planner:  python scripts/probe_screen.py ROOT_ID"""
import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2d import dataset as D, teacher as TT  # noqa: E402

rec = D.record(sys.argv[1])
TT.boot(D.geom_of(rec))
m = TT.Motion(rec, TT.Shared())
c = dict(m.cells0)


def timed(name, fn):
    v0 = m.vcpu
    r = fn()
    print(f"{name:52s} {('ok' if r is not None and r is not False else 'FAIL'):4s} vcpu {m.vcpu - v0:6.1f}s", flush=True)
    return r


for n in (100, 200, 400):
    timed(f"screen target, all objects, nodes={n}", lambda: m.skill(c, "t0", m.goal_cell, keep="all", nodes=n))
timed("screen target, obstacles ignored, nodes=200", lambda: m.skill(c, "t0", m.goal_cell, keep=()))
mini = next(b["sequence"] for b in rec["labels"]["B"] if b["role"] == "minimal")
o, d = mini[0]
c2 = {**c, o: d}
timed(f"screen target after minimal repair {mini}", lambda: m.skill(c2, "t0", m.goal_cell, keep="all"))
timed(f"screen relocation {o}->{d}, all objects", lambda: m.skill(c, o, d, keep="all"))
free = [x for x in m.legal if x not in c.values()]
far = max(free, key=lambda x: abs(x % 8 - c[o] % 8) + abs(x // 8 - c[o] // 8))
timed(f"screen relocation {o}->{far} (far), all objects", lambda: m.skill(c, o, far, keep="all"))
