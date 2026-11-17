"""Obstacle-free relocation paths at two RRT budgets vs the executable planner:  python scripts/probe_relaxed.py ROOT_ID"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2d import dataset as D, teacher as TT  # noqa: E402

rec = D.record(sys.argv[1])
TT.boot(D.geom_of(rec))
m = TT.Motion(rec, TT.Shared())
c = dict(m.cells0)
mini = next(b["sequence"] for b in rec["labels"]["B"] if b["role"] == "minimal")
o = mini[0][0]
free = [x for x in m.legal if x not in c.values()]
dests = [mini[0][1]] + sorted(free, key=lambda x: abs(x % 8 - c[o] % 8) + abs(x // 8 - c[o] // 8))[:5]
print("object", o, "at", c[o], "minimal dest", mini[0][1])
for d in dests:
    row = [f"{o}->{d:2d}"]
    for n in (200, 1500):
        v0 = m.vcpu
        r = m.skill(c, o, d, nodes=n)
        row.append(f"free@{n}: {'ok ' if r is not None else 'FAIL'} {m.vcpu - v0:5.1f}s")
    v0 = m.vcpu
    r = m.relocate(c, m.qref, o, d)
    row.append(f"executable: {'ok ' if r is not None else 'FAIL'} {m.vcpu - v0:5.1f}s")
    print("  ".join(row), flush=True)
