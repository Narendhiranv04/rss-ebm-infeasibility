"""Profile one X / Y verification:  python scripts/profile_query.py ROOT_ID [o00:54,o01:3]"""
import cProfile, pstats, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2d import dataset as D, teacher as TT  # noqa: E402

rid = sys.argv[1]
seq = [(a.split(":")[0], int(a.split(":")[1])) for a in sys.argv[2].split(",")] if len(sys.argv) > 2 else []
rec = D.record(rid)
TT.boot(D.geom_of(rec))
for i in range(2):                                   # warm-up (imports), then a cold-cache repeat: per-query cost in a long-lived worker
    t0 = time.time()
    v = TT.verify(rec, seq, {})
    print(f"run {i}: wall {time.time() - t0:.1f}s x={v['x']['x']} y={v['y']['y']}", flush=True)
t0, c0 = time.time(), time.process_time()
pr = cProfile.Profile()
pr.enable()
v = TT.verify(rec, seq, {})
pr.disable()
print("wall", round(time.time() - t0, 1), "cpu", round(time.process_time() - c0, 1), v["x"]["x"], v["y"]["y"], v["x"].get("budget_spent"))
pstats.Stats(pr).sort_stats("cumulative").print_stats(28)
