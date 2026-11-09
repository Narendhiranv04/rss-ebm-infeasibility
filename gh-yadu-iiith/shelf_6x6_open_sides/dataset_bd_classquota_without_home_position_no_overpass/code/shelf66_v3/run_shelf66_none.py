"""NONE-only runner for the 2,500-scene 6 x 6 shelf dataset (2026-11-07).  Separate store: the NONE class of the dataset comes from
this run; the NONE lines of the main run (V3 audit, flawed: its 3 checked NONE were all K1) are not used.

Every request is a NONE: dense construction (none_dense.construct_dense_none: every feasible direction cut by >= 5 cells, filled up to
14 objects, every object pickable alone), then the blocking certificate (none_cert66_v4.certificate, with the REAL objects; a scene whose
certificate is short is rejected before teaching and the next start / goal pair is tried), then the V3 teacher + consistency checks
(run_shelf66_full.label) with the NONE gate replaced by none_cert66_v4 (CONFIRMED = certificate proven + direct fails + one move first
fails).  Same live controls as run_shelf66_full (STOP, MAX_RUNNING, QUOTA.json).

    python shelf66_v3/run_shelf66_none.py --out <none run dir> --name lapN --workers 5"""
from __future__ import annotations

import json
import signal
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_shelf66_full as F  # noqa: E402
import none_dense as ND  # noqa: E402
import none_cert66_v4 as NC  # noqa: E402

bc, R = F.bc, F.R
GEN_NONE = "DENSE_NONE_V2+CERT66_V4"
B_CELLS = 6                     # the one-move check per obstacle (the certificate is the proof)
CERT_TRIES = 4                  # dense constructions (each up to ND.ATTEMPTS start / goal pairs) per root until a certificate holds
F.QUOTA.clear()
F.QUOTA.update({"NONE": 408, "0": 0, "1": 0, "2": 0, "3": 0, "4": 0})
R.WALL_S = F.TEACH_WALL_S + F.AUDIT_BUDGET_S + 600
ND.ATTEMPTS = 3


def plan(led, n, wave, name, pending=()):
    allowed = [x for x in bc.SPECS if x["K"] == 0 and x["start_pose"] == "home" and x["distractors"] == F.NONE_DISTRACTORS]
    return [bc.plan_wave(led + [{"accepted": True, "spec": x, "verified_K": x["K"]} for x in pending], 1, wave, f"{name}-none", allowed)[0]]


def none_gate(rec):
    signal.alarm(F.AUDIT_BUDGET_S + 600)
    try:
        aud = NC.audit_root(json.loads(json.dumps(rec, default=bc._jd)), budget=F.AUDIT_BUDGET_S, b_cells=B_CELLS)
    except Exception as e:  # noqa: BLE001
        aud = {"audit_version": NC.AUDIT_VERSION, "status": f"ERROR:{type(e).__name__}:{str(e)[:120]}", "trace": traceback.format_exc()[-1500:]}
    finally:
        signal.alarm(0)
    st = aud["status"]
    return ("NONE" if st == "CONFIRMED" else "NONE_SOLVABLE" if st.startswith("SOLVABLE") else
            "NONE_NOT_PROVEN" if st == "NOT_PROVEN" else "NONE_UNAUDITED"), aud


def construct(spec, seed, rid, A, P):
    tries = []
    for i in range(CERT_TRIES):
        c = ND.construct_dense_none(spec, seed + 7919 * i, rid, A, P, bc)
        cert = NC.certificate(A, c["objs"], c["cells"], c["src"], c["goal"])
        tries.append({"min_count": cert["min_count"], "F0": cert["F0"], "n_obj": len(c["objs"]) - 1})
        if cert["proven"]:
            c["construction"]["certificate"] = cert
            c["construction"]["cert_tries"] = tries
            return c
    raise A.Reject("CERT_SHORT:" + json.dumps(tries))


def work(job):
    D, A, P = F._modules()
    F.none_gate = none_gate
    name, wave, slot, spec, out = job
    G = A.env()
    ids = bc.ids(D.DOMAIN, name, wave, slot, spec, G["config_hash"], A.TEACHER_VERSION, G["protocol"])
    store = bc.Store(out)
    t0 = time.time()
    signal.signal(signal.SIGALRM, R._alarm)
    signal.alarm(F.TEACH_WALL_S)
    entry = {"gen": P.GENERATOR_VERSION, "gen_none": GEN_NONE, "wave": wave, "slot": slot, "spec": spec, "root_id": ids["root_id"],
             "construction_seed": ids["construction_seed"]}
    try:
        A.POOL_CAP = P.POOL_CAP
        c = construct(spec, ids["construction_seed"], ids["root_id"], A, P)
        entry.update(label_none(spec, ids, c, store))
    except (A.Reject, R.WallCap) as e:
        entry.update(accepted=False, reason=f"{type(e).__name__}:{str(e)[:400]}")
    except Exception as e:  # noqa: BLE001
        entry.update(accepted=False, reason=f"ERROR:{type(e).__name__}:{str(e)[:200]}", trace=traceback.format_exc()[-1500:])
    finally:
        signal.alarm(0)
    entry["wall_s"] = time.time() - t0
    print(f"root {entry['root_id']} NONE -> {entry.get('observed', entry.get('reason'))} ({entry['wall_s']:.0f} s)", flush=True)
    return entry


def label_none(spec, ids, c, store):
    out = F.label(spec, ids, c, store, {"generator_none": GEN_NONE})
    if out.get("observed") not in ("NONE", "NONE_SOLVABLE", "NONE_NOT_PROVEN", "NONE_UNAUDITED"):
        out["observed_teacher"] = out["observed"]       # e.g. the teacher found a repair: recorded, not counted as NONE
    return out


R.plan = plan
R.work = work
R.quota = F.quota

if __name__ == "__main__":
    F.main()
