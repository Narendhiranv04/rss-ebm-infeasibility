"""Time the dataset's own checker (no per-call cap) on K0 scenes the baselines missed: X/Y with an empty repair and its CPU cost.

    .\dk.ps1 python scripts/k0_uncapped_probe.py 2d-plus2d-w0008-s006 2d-plus2d-w0019-s004 2d-conv-plus2d-w0120-s010"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2d import dataset as D, teacher as TT  # noqa: E402

for rid in sys.argv[1:]:
    rec = D.record(rid)
    TT.boot(D.geom_of(rec))
    t = time.process_time()
    v = TT.verify(rec, [], {})
    print(rid, "x", v["x"]["x"], v["x"]["status"], "y", v["y"]["y"], v["y"]["status"], "cpu", round(time.process_time() - t, 1), flush=True)
