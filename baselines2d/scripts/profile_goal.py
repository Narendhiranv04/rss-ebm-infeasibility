"""Profile the initial target query of a root:  python scripts/profile_goal.py ROOT_ID"""
import cProfile, pstats, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2d import dataset as D, teacher as TT  # noqa: E402

rec = D.record(sys.argv[1])
TT.boot(D.geom_of(rec))
m = TT.Motion(rec, TT.Shared())
import p15_retarget as pr  # noqa: E402
ph = m.phys()
s = ph.scene(m.cells0, m.qref)
t = time.process_time()
c = pr.certified(ph.cfg, ph.robot, s, "t0", m.goal_cell)
print("certified():", c, round(time.process_time() - t, 1), "cpu-s", flush=True)
pr_ = cProfile.Profile()
pr_.enable()
t = time.process_time()
ok = m.target(m.cells0)
pr_.disable()
print("target():", ok, round(time.process_time() - t, 1), "cpu-s", flush=True)
pstats.Stats(pr_).sort_stats("cumulative").print_stats(22)
