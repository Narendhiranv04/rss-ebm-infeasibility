# SHELF66_V3 COPY (2026-11-05) of shelf66_full/: shelf_scene_6x6/shelf_scene.py checks the fingers PRE-SHAPED (GRIP_JOINTS) and pre-shapes
# them before every pick; version strings generator SHELF66V3, teacher BD3DS66_TEACHER_V2.  This runner also (1) counts a NONE only when the
# NONE audit (none_audit66_v3, in the worker) is CONFIRMED, (2) has reteach() for reteach_shelf66_v3.py (the SHELF66V2 roots, same scenes,
# V3 labels).  shelf66_full/ = V2, unchanged.
"""SHELF66_V3: the 2,500-scene class-quota B/D dataset on the 6 x 6, 7.5 cm, open-sided shelf (shelf66_v3/: bd_shelf / bd_3dt /
bd_3dt_plus copies + shelf_scene_6x6/).  The arm never returns to the ready pose between moves, and the arm and hand never pass over
an obstacle.  Class shares as in the 1,225-scene open-tabletop dataset (NONE 16.3 %, K0 20.4 %, K1 / K2 / K3 14.3 % each, K4 20.4 %).

    python shelf66_v3/run_shelf66_full.py --out <run dir> --name lap3 --workers 5      (laptop)
    python shelf66_v3/run_shelf66_full.py --out <run dir> --name srv3 --workers 18     (server)
    python shelf66_v3/run_shelf66_full.py --out <run dir> --select-only                (selected.json from the ledger)

run_3dt_plus's runner (requests planned from the ledger, resumable, <out>/STOP for a clean stop, <out>/MAX_RUNNING) with:
  * QUOTA below; <out>/QUOTA.json, when present, overrides it live (e.g. to move K4's share if K4 stays out of reach);
  * NONE is requested through K0-type constructions with 3-4 distractors: run_3dt_plus builds NONE through the K = 4 route
    construction, which never finishes on this shelf, while 3-4-distractor K0 scenes gave NONE about a third of the time;
  * at most K4_MAX_RUNNING K4 roots run at once on one machine (default 2; <out>/K4_MAX_RUNNING overrides it live);
  * consistency: a root is not counted ("INCONSISTENT") when an executed non-minimal sequence B with FEWER moves than its verified
    K succeeds (for NONE: any) -- e.g. a NONE whose target moves after one irrelevant relocation (the arm then starts elsewhere).
    DEPENDENT (a relocation needs another one first) is not counted either, as before.
  * NONE audit gate (V3): the teacher's NONE is "no repair found within its limits", not a proof, and borderline scenes flip on
    sub-mm settling.  A consistent teacher NONE is audited in the worker (none_audit66_v3: strong planner, one move first, every
    removal set x move order x up to 3 parking cells, chained MuJoCo checks; AUDIT_BUDGET_S) and counts only when CONFIRMED.
    Otherwise the ledger says NONE_SOLVABLE (the audit found a solution: the label is wrong) or NONE_UNAUDITED (budget / error),
    neither counted; the root file keeps the audit ("none_audit").
The laptop and server runs share one ledger through merge_with_server_3ds66_v3.sh (both runners count every root toward the quotas)."""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dataset-bd"))                 # bd_core
sys.path.insert(0, str(HERE.parent))                                # run_3dt_plus (the runner; work / plan / quota replaced below)
sys.path.insert(0, str(HERE))                                       # bd_shelf / bd_3dt / bd_3dt_plus: the SHELF66_FULL copies first
import bd_core as bc  # noqa: E402
import run_3dt_plus as R  # noqa: E402

GEN = "BD_GEN_V1+ROUTE_V15+SHELF66V3"                                # = shelf66_v3/bd_3dt_plus.GENERATOR_VERSION
QUOTA = {"NONE": 408, "0": 511, "1": 357, "2": 357, "3": 357, "4": 510}   # open-tabletop shares x 2500 / 1225 (largest remainder)
NONE_DISTRACTORS = "3-4"                                             # K0-type requests made for NONE use this many distractors
OUT: list = [None]                                                   # the run directory (QUOTA.json, K4_MAX_RUNNING), set in main()
_NO_K4 = [False]


def quota(led=None):
    """QUOTA, updated from <out>/QUOTA.json when present (read on every call: live changes without a restart)."""
    q = dict(QUOTA)
    if OUT[0] is not None:
        try:
            q.update({k: int(v) for k, v in json.loads((OUT[0] / "QUOTA.json").read_text()).items()})
        except (OSError, ValueError):
            pass
    if _NO_K4[0]:
        q["4"] = 0
    return q


def k4_cap():
    try:
        return int((OUT[0] / "K4_MAX_RUNNING").read_text())
    except (OSError, ValueError, TypeError):
        return int(os.environ.get("K4_MAX_RUNNING", "2"))


R.quota = quota                                  # run_3dt_plus.counts / plan / main look quota() up at call time
R.QUOTA = R.QUOTA_MAX3 = QUOTA                   # (no automatic K4 drop: the user decides; QUOTA.json)
R.GEN = GEN
R.REQUEST_FOR = {**R.REQUEST_FOR, "NONE": 0}     # NONE through K0-type constructions (see plan below)
TEACH_WALL_S = 4200                              # construction + teacher (K = 3: 45 min construction), as in SHELF66V2
AUDIT_BUDGET_S = 3600                            # NONE audit on top of it (V2.1 audits: CONFIRMED <= 34 min, repairs <= 46 min)
R.WALL_S = TEACH_WALL_S + AUDIT_BUDGET_S + 600   # run_3dt_plus.main frees a slot as lost only after R.WALL_S + 900 s
_plan0 = R.plan


def plan(led, n, wave, name, pending=()):
    """run_3dt_plus.plan for one root, then (1) the K4 running cap, (2) a K0 request is made for NONE (3-4 distractors) with
    probability need_NONE * rate_NONE / (need_NONE * rate_NONE + need_K0 * rate_K0), the rates learned from the ledger."""
    s = _plan0(led, 1, wave, name, pending)[0]
    if s["K"] == 4 and sum(p["K"] == 4 for p in pending) >= k4_cap():
        got, Q = R.counts(led), quota(led)
        if any(Q[k] > got[k] for k in Q if k != "4"):
            _NO_K4[0] = True
            try:
                for j in range(10):
                    s = _plan0(led, 1, wave, f"{name}-x{j}", pending)[0]
                    if s["K"] != 4:
                        break
            finally:
                _NO_K4[0] = False
    if s["K"] == 0:
        got, Q = R.counts(led), quota(led)
        nN, n0 = max(Q["NONE"] - got["NONE"], 0), max(Q["0"] - got["0"], 0)
        if nN:
            k0 = [e for e in led if e["spec"]["K"] == 0]
            t = [e for e in k0 if e["spec"]["distractors"] == NONE_DISTRACTORS]
            rN = (sum(e.get("accepted", False) and e.get("observed") == "NONE" for e in t) + 1) / (len(t) + 3)
            r0 = (sum(e.get("accepted", False) and e.get("observed") == "0" for e in k0) + 3) / (len(k0) + 4)
            pN = nN * rN / (nN * rN + n0 * r0)
            if np.random.default_rng(int(bc.sha([name, wave, "none"])[:8], 16)).random() < pN:
                allowed = [x for x in bc.SPECS if x["K"] == 0 and x["start_pose"] == "home" and x["distractors"] == NONE_DISTRACTORS]
                s = bc.plan_wave(led + [{"accepted": True, "spec": x, "verified_K": x["K"]} for x in pending], 1, wave, f"{name}-none", allowed)[0]
    return [s]


R.plan = plan


def _modules():
    """bd_shelf first (so bd_3dt's env() is the shelf), and a check that these are the SHELF66V3 copies."""
    import bd_shelf as D
    import bd_3dt as A
    import bd_3dt_plus as P
    assert P.GENERATOR_VERSION == R.GEN and D.DOMAIN == "3ds66" and "shelf66_v3" in A.__file__ and "shelf66_v3" in P.__file__, \
        (P.GENERATOR_VERSION, A.__file__, P.__file__)
    return D, A, P


def none_gate(rec):
    """(ledger label, audit) for a consistent teacher NONE: "NONE" only when the audit is CONFIRMED.  The teacher's alarm is replaced
    by the audit's budget (+ 600 s hard stop) meanwhile."""
    import none_audit66_v3 as NA
    signal.alarm(AUDIT_BUDGET_S + 600)
    try:
        aud = NA.audit_root(json.loads(json.dumps(rec, default=bc._jd)), budget=AUDIT_BUDGET_S)
    except Exception as e:  # noqa: BLE001  (incl. the hard stop: recorded, the root just is not counted)
        aud = {"audit_version": NA.AUDIT_VERSION, "status": f"ERROR:{type(e).__name__}:{str(e)[:120]}", "trace": traceback.format_exc()[-1500:]}
    finally:
        signal.alarm(0)
    st = aud["status"]
    return ("NONE" if st == "CONFIRMED" else "NONE_SOLVABLE" if st.startswith("SOLVABLE") else "NONE_UNAUDITED"), aud


def label(spec, ids, c, store, extra=None):
    """The V3 teacher's label of scene c, the consistency checks (DEPENDENT / INCONSISTENT) and the NONE audit gate; stores the root
    record (plus `extra`).  Returns the ledger fields of an accepted root."""
    _, A, P = _modules()
    t = A.teach(c, store, ids["root_id"])
    rec = P.record(spec, ids, c, t, store)
    rec.update(extra or {})
    observed = str(rec["observed"])
    ms = next((b["sequence"] for b in rec["labels"]["B"] if b["role"] == "minimal" and b["y"]), [])
    if ms and not P.independent(c["objs"], c["cells"], dict(ms), [o for o, _ in ms], c["goal"]):
        observed = "DEPENDENT"                    # a relocation needs another one first: not counted
    K = rec["verified_K"]
    short = [b["role"] for b in rec["labels"]["B"] if b["role"] != "minimal" and b["y"] and (K is None or len(b["sequence"]) < K)]
    rec["label_consistency"] = {"consistent": not short, "shorter_sequences_that_succeeded": short}
    if short and observed != "DEPENDENT":
        observed = "INCONSISTENT"                 # a shorter executed sequence succeeds: the label is not the minimum -> not counted
    rec["independent_moves"] = observed != "DEPENDENT"
    if observed == "NONE":
        observed, rec["none_audit"] = none_gate(rec)
    rec["ledger_observed"] = observed
    store.save_root(rec)
    out = dict(accepted=True, verified_K=rec["verified_K"], observed=observed, y_original=rec["labels"]["original"]["y"],
               method=c["construction"]["method"], K_by_construction=c["construction"].get("K_by_construction"),
               distractors=c["construction"].get("distractors"))
    if "none_audit" in rec:
        out.update(none_audit=rec["none_audit"]["status"], none_audit_s=rec["none_audit"].get("wall_s"))
    return out


def work(job):
    """run_3dt_plus.work with the shelf domain, the consistency checks and the NONE audit gate (label())."""
    D, A, P = _modules()
    name, wave, slot, spec, out = job
    G = A.env()
    ids = bc.ids(D.DOMAIN, name, wave, slot, spec, G["config_hash"], A.TEACHER_VERSION, G["protocol"])
    store = bc.Store(out)
    t0 = time.time()
    signal.signal(signal.SIGALRM, R._alarm)
    signal.alarm(TEACH_WALL_S)
    entry = {"gen": P.GENERATOR_VERSION, "wave": wave, "slot": slot, "spec": spec, "root_id": ids["root_id"], "construction_seed": ids["construction_seed"]}
    try:
        c = P.construct(spec, ids["construction_seed"], ids["root_id"])
        entry.update(label(spec, ids, c, store))
    except (A.Reject, R.WallCap) as e:
        entry.update(accepted=False, reason=f"{type(e).__name__}:{e}")
    except Exception as e:  # noqa: BLE001  (recorded, never silently dropped)
        entry.update(accepted=False, reason=f"ERROR:{type(e).__name__}:{str(e)[:200]}", trace=traceback.format_exc()[-1500:])
    finally:
        signal.alarm(0)
    entry["wall_s"] = time.time() - t0
    print(f"root {entry['root_id']} requested K={spec['K']} (distractors {spec['distractors']}) -> "
          f"{entry.get('observed', entry.get('reason'))} ({entry['wall_s']:.0f} s)", flush=True)
    return entry


R.work = work


def scene_of(r):
    """The construction dict (bd_3dt_plus.construct's) of a stored root (as reteach_3dt.scene_of)."""
    import bd_3dt_plus as P
    src, goal = tuple(r["target_src"]), tuple(r["goal"])
    occ = {tuple(v) for v in r["initial_cells"].values()}
    return {"objs": r["objects"], "cells": r["initial_cells"], "src": r["target_src"], "goal": r["goal"], "roles": r["roles"], "parking": r["parking"],
            "alt_parking": r["alt_parking"], "intended": r["intended_blockers"], "wrong_dest": [list(x) for x in P.line(src, goal) if x not in occ][:2],
            "pools": r["mechanism_pools"], "empty_solve": r["empty_table_T_solve"], "greedy": r["greedy_blockers"], "construction": r["construction"]}


def reteach(job):
    """One accepted root of a SHELF66V2 run, taught again by the V3 teacher: the same scene (rebuilt from its root file; the construction
    is not re-run), the same root id, a new record / ledger line ("retaught_from" = the V2 ledger line's label)."""
    _, A, P = _modules()
    e0, srcdir, out = job
    rid, spec = e0["root_id"], e0["spec"]
    store = bc.Store(out)
    t0 = time.time()
    signal.signal(signal.SIGALRM, R._alarm)
    signal.alarm(TEACH_WALL_S)
    was = {k: e0.get(k) for k in ("gen", "observed", "verified_K", "y_original", "wall_s")}
    entry = {"gen": P.GENERATOR_VERSION, "wave": e0["wave"], "slot": e0["slot"], "spec": spec, "root_id": rid,
             "construction_seed": e0["construction_seed"], "retaught_from": was}
    try:
        old = json.loads((Path(srcdir) / "roots" / f"{rid}.json").read_text())
        G = A.env()
        A.POOL_CAP = P.POOL_CAP_K4 if spec["K"] >= 4 else P.POOL_CAP          # as bd_3dt_plus.construct sets it for this root
        ids = {**old["ids"], "config_hash": G["config_hash"], "teacher_version": A.TEACHER_VERSION, "robot_protocol": G["protocol"]}
        extra = {"reteach": {"from_generator_version": old.get("generator_version_plus"), "from_config_hash": old["ids"]["config_hash"],
                             "from_teacher_version": old["ids"]["teacher_version"], "from_ledger_observed": e0.get("observed"),
                             "from_observed": str(old["observed"]), "from_verified_K": old["verified_K"]}}
        entry.update(label(spec, ids, scene_of(old), store, extra))
    except (A.Reject, R.WallCap) as e:
        entry.update(accepted=False, reason=f"{type(e).__name__}:{e}")
    except Exception as e:  # noqa: BLE001  (recorded, never silently dropped)
        entry.update(accepted=False, reason=f"ERROR:{type(e).__name__}:{str(e)[:200]}", trace=traceback.format_exc()[-1500:])
    finally:
        signal.alarm(0)
    entry["wall_s"] = time.time() - t0
    print(f"re-taught {rid}: {e0.get('observed')} -> {entry.get('observed', entry.get('reason'))} ({entry['wall_s']:.0f} s)", flush=True)
    return entry


def select(out: Path):
    """selected.json: the accepted roots of the ledger, in ledger order, per verified class up to the quota (= what the runner
    counts; DEPENDENT and INCONSISTENT roots are never counted)."""
    led = bc.Store(out).ledger()
    Q = quota(led)
    got, pick = {k: 0 for k in Q}, {k: [] for k in Q}
    for e in led:
        if e.get("accepted") and e["observed"] in got and got[e["observed"]] < Q[e["observed"]]:
            got[e["observed"]] += 1
            pick[e["observed"]].append(e["root_id"])
    sel = {"rule": "accepted roots of raw/ledger.jsonl in ledger order, per verified class, up to the class quota (as the runner counts "
                   "them; DEPENDENT and INCONSISTENT roots are not counted)", "quota": Q, "counts": got, "selected": pick}
    (out / "selected.json").write_text(json.dumps(sel, indent=1))
    print(f"selected {sum(got.values())} roots {got} -> {out / 'selected.json'}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="lap")
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--select-only", action="store_true")
    a = ap.parse_args()
    OUT[0] = Path(a.out)
    if not a.select_only:
        sys.argv = [sys.argv[0], "--out", a.out, "--name", a.name, "--workers", str(a.workers), "--wave-size", "24"]
        R.main()
    select(Path(a.out))


if __name__ == "__main__":
    main()
