"""Blackwell vs gvlab parity on identical items (same pinned teacher, same witnesses).  PASS requires identical labels (teacher and replay
verdicts) and per-move agreement (end cell; |z_err| difference <= 1e-6 m); final-state differences are reported (relocation replay is
expected bitwise or near-bitwise; a freshly planned connector may differ at the 1e-5 rad level, as R2 showed for planning).

    python3 -I compare_parity.py PARITY_MANIFEST.json GVLAB_ITEMS_DIR BLACKWELL_ITEMS_DIR OUT.json"""
import json, sys
from pathlib import Path

import numpy as np

man, gv, bw, out = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])
rows, ok = [], True
for it in man["items"]:
    for mode in ("teacher", "replay"):
        a, b = gv / f"{it['item']}.{mode}.json", bw / f"{it['item']}.{mode}.json"
        if not a.exists() and not b.exists():
            continue
        if not (a.exists() and b.exists()):
            rows.append({"item": it["item"], "mode": mode, "missing": [str(p.name) for p in (a, b) if not p.exists()]}); ok = False; continue
        A, B = json.loads(a.read_text()), json.loads(b.read_text())
        same_v = A.get("verdict") == B.get("verdict")
        sa, sb = A.get("steps") or ([A["step"]] if A.get("step") else []), B.get("steps") or ([B["step"]] if B.get("step") else [])
        step_ok = len(sa) == len(sb) and all(x.get("ok") == y.get("ok") and x.get("end_cell") == y.get("end_cell") and
                                            (x.get("z_err") is None or abs(x["z_err"] - y["z_err"]) <= 1e-6) for x, y in zip(sa, sb))
        zd = max([abs(x["z_err"] - y["z_err"]) for x, y in zip(sa, sb) if x.get("z_err") is not None and y.get("z_err") is not None] or [0.0])
        fd = None
        if "final" in A and "final" in B:
            fd = max(float(np.max(np.abs(np.array(A["final"]["q"]) - np.array(B["final"]["q"])))),
                     max(float(np.max(np.abs(np.array(A["final"]["obj"][k]) - np.array(B["final"]["obj"][k])))) for k in A["final"]["obj"]))
        good = same_v and step_ok
        ok &= good
        rows.append({"item": it["item"], "mode": mode, "verdict_gvlab": A.get("verdict"), "verdict_blackwell": B.get("verdict"), "labels_agree": same_v,
                     "steps_agree": step_ok, "max_z_err_diff_m": zd, "final_state_max_diff": fd,
                     "connector_dq_gv_bw": [(A.get("connector") or {}).get("dq_ready"), (B.get("connector") or {}).get("dq_ready")]})
res = {"pass": ok, "pairs": len(rows), "labels_agree": sum(1 for r in rows if r.get("labels_agree")), "steps_agree": sum(1 for r in rows if r.get("steps_agree")),
       "max_z_err_diff_m": max([r.get("max_z_err_diff_m") or 0 for r in rows] or [0]),
       "max_final_state_diff": max([r["final_state_max_diff"] for r in rows if r.get("final_state_max_diff") is not None] or [None], key=lambda v: -1 if v is None else v),
       "rows": rows}
out.write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))
