"""Diagnostic (no label is written): re-run one X item in teacher mode and dump the per-grasp planning reasons that A.kind classifies."""
import inspect, json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import bwlphys as B
A, _ = B.env()
cap, q0 = [], A.query
def q(*a, **k):
    p, info = q0(*a, **k); cap.append({"found": p is not None, "grasp_reasons": info.get("grasp_reasons"), "keys": sorted(info)}); return p, info
A.query = q
man = json.loads(Path(sys.argv[2]).read_text()); it = next(x for x in man["items"] if x["item"] == sys.argv[3])
res = B.run_x(it, "teacher", Path(tempfile.mkdtemp()))
print(json.dumps({"verdict": res["verdict"], "last_step": {k: v for k, v in res["steps"][-1].items()}, "queries": cap,
                  "kind_src": inspect.getsource(A.kind)}, default=str, indent=1))
