"""Stage 1 of the class-quota 3D dataset: base scenes (bd_3dt_plus construction + bd_3dt teacher) until every VERIFIED class
reaches its quota.  Resumable (ledger), clean stop with <out>/STOP, one spawned worker per root under an administrative wall cap.

    python run_3dt_plus.py --out 3dt_plus_raw --workers 6

Each wave requests K levels in proportion to (remaining quota / observed yield of that request); the other factors come from
bd_core.plan_wave restricted to that K.  A root whose verified class is already full is kept (ledger: surplus) but not selected."""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import signal
import sys
import time
import traceback
from pathlib import Path

import numpy as np

import bd_core as bc

QUOTA = {"NONE": 200, "0": 250, "1": 175, "2": 175, "3": 175, "4": 250}       # 1225 base scenes
QUOTA_MAX3 = {"NONE": 200, "0": 250, "1": 175, "2": 175, "3": 225, "4": 0}    # if 4 independent moves prove impossible (total
                                                                               # stays above 1000; verified K=4 roots are kept)
REQUEST_FOR = {"NONE": 4, "0": 0, "1": 1, "2": 2, "3": 3, "4": 4}
K4_TRIAL, K4_MIN_YIELD = 60, 0.05
MIN_YIELD = 0.25
WALL_S = 3600                                         # K = 4: 30 min construction + ~6-12 min teacher (pool of 8)
GEN = "BD_GEN_V1+ROUTE_V15"                            # = bd_3dt_plus.GENERATOR_VERSION


class WallCap(Exception):
    pass


def _alarm(sig, frame):
    raise WallCap("administrative wall cap")


def work(job):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dataset-bd"))
    import bd_3dt as A
    import bd_3dt_plus as P
    name, wave, slot, spec, out = job
    G = A.env()
    ids = bc.ids("3dt", name, wave, slot, spec, G["config_hash"], A.TEACHER_VERSION, G["protocol"])
    store = bc.Store(out)
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(WALL_S)
    entry = {"gen": P.GENERATOR_VERSION, "wave": wave, "slot": slot, "spec": spec, "root_id": ids["root_id"], "construction_seed": ids["construction_seed"]}
    try:
        c = P.construct(spec, ids["construction_seed"], ids["root_id"])
        t = A.teach(c, store, ids["root_id"])
        rec = P.record(spec, ids, c, t, store)
        observed = str(rec["observed"])
        ms = next((b["sequence"] for b in rec["labels"]["B"] if b["role"] == "minimal" and b["y"]), [])
        if ms and not P.independent(c["objs"], c["cells"], dict(ms), [o for o, _ in ms], c["goal"]):
            observed = "DEPENDENT"                    # a relocation needs another one first: not selected
        rec["independent_moves"] = observed != "DEPENDENT"
        store.save_root(rec)
        entry.update(accepted=True, verified_K=rec["verified_K"], observed=observed, y_original=rec["labels"]["original"]["y"],
                     method=c["construction"]["method"], K_by_construction=c["construction"].get("K_by_construction"))
    except (A.Reject, WallCap) as e:
        entry.update(accepted=False, reason=f"{type(e).__name__}:{e}")
    except Exception as e:  # noqa: BLE001  (recorded, never silently dropped)
        entry.update(accepted=False, reason=f"ERROR:{type(e).__name__}:{str(e)[:200]}", trace=traceback.format_exc()[-1500:])
    finally:
        signal.alarm(0)
    entry["wall_s"] = time.time() - t0
    return entry


def quota(led):
    """QUOTA, or QUOTA_MAX3 once K=4 has been requested K4_TRIAL times with a verified yield below K4_MIN_YIELD."""
    k4 = [e for e in led if e["spec"]["K"] == 4 and e.get("gen") == GEN]     # K=4 trial with the current construction
    hit = sum(e.get("accepted", False) and e["observed"] == "4" for e in k4)
    return QUOTA_MAX3 if len(k4) >= K4_TRIAL and hit < K4_MIN_YIELD * len(k4) else QUOTA


def counts(led):
    Q = quota(led)
    got = {k: 0 for k in Q}
    for e in led:
        if e.get("accepted") and e["observed"] in got and got[e["observed"]] < Q[e["observed"]]:
            got[e["observed"]] += 1
    return got


def plan(led, n, wave, name, pending=()):
    """Requested K for n jobs, in proportion to sum over classes of (remaining quota x P(class | requested K)), P learned from the ledger.
    pending: specs of roots still running, counted in bd_core's factor balance."""
    got, Q = counts(led), quota(led)
    need = {k: Q[k] - got[k] for k in Q if Q[k] > got[k]}
    weight = {}
    for r in range(5):                                # expected useful roots per request of K = r, learned from the ledger
        past = [e for e in led if e["spec"]["K"] == r and (r < 3 or e["spec"]["displacement"] != "short") and (r < 4 or e.get("gen") == GEN)]
        w = sum(need[k] * (sum(e.get("accepted", False) and e["observed"] == k for e in past) + (1.0 if REQUEST_FOR[k] == r else 0.1))
                / (len(past) + 2) for k in need)
        if w > 0:
            weight[r] = w
    total = sum(weight.values())
    weight = {r: max(w / total, 0.05 if any(REQUEST_FOR[k] == r for k in need) else 0.0)     # >= 5% (exploration) while
              for r, w in weight.items()}                                                    # the requested class needs roots
    rng = np.random.default_rng(int(bc.sha([name, wave])[:8], 16))
    Ks = sorted(weight)
    req = rng.choice(Ks, size=n, p=np.array([weight[k] for k in Ks]) / sum(weight.values()))
    specs = []
    for i, k in enumerate(req):
        allowed = [s for s in bc.SPECS if s["K"] == int(k) and s["start_pose"] == "home"
                   and (k < 3 or s["displacement"] != "short")]   # short moves keep 4-6 grasps feasible: no independent K >= 3 repair
        if k >= 3:                   # start area x move length drawn by its verified yield so far (prior 0.5 / 2: every one keeps tries)
            combos = sorted({(s["source"], s["displacement"]) for s in allowed})
            tried = collections.Counter((e["spec"]["source"], e["spec"]["displacement"]) for e in led if e["spec"]["K"] == k)
            hits = collections.Counter((e["spec"]["source"], e["spec"]["displacement"]) for e in led
                                       if e["spec"]["K"] == k and e.get("accepted") and e.get("observed") == str(int(k)))
            w = np.array([(hits[c] + 0.5) / (tried[c] + 2) for c in combos])
            c = combos[rng.choice(len(combos), p=w / w.sum())]
            allowed = [s for s in allowed if (s["source"], s["displacement"]) == c]
        specs.append(bc.plan_wave(led + [{"accepted": True, "spec": s, "verified_K": s["K"]} for s in list(pending) + specs], 1, wave, f"{name}-{i}", allowed)[0])
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="plus3dt")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wave-size", type=int, default=24)
    a = ap.parse_args()
    out = Path(a.out)
    store = bc.Store(out)
    t_start = time.time()
    led = store.ledger()
    wave, slot, running = 1 + max([e["wave"] for e in led], default=-1), 0, {}       # running: AsyncResult -> (spec, start time)
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=20) as pool:
        while True:                  # one root per free worker, planned from the ledger at that moment (no waiting for a wave's slowest root)
            done = [r for r in running if r.ready()]
            for r in done:
                store.append(r.get())
                running.pop(r)
            for r in [r for r, (_, t0) in running.items() if time.time() - t0 > WALL_S + 900]:
                print(f"lost job (worker died?) after {WALL_S + 900} s: slot freed", flush=True)
                running.pop(r)
            led = store.ledger()
            got, Q = counts(led), quota(led)
            if (out / "STOP").exists() or all(got[k] >= Q[k] for k in Q):
                if not running:
                    break
            else:
                cap = a.workers
                try:                                 # <out>/MAX_RUNNING lowers the number of roots running at once, without a restart
                    cap = min(cap, int((out / "MAX_RUNNING").read_text()))
                except (OSError, ValueError):
                    pass
                while len(running) < cap:
                    if slot == a.wave_size:          # ids stay (name, wave, slot); a wave is now just a block of wave_size slots
                        wave, slot = max(wave + 1, 1 + max([e["wave"] for e in led], default=-1)), 0
                    spec = plan(led, 1, wave, f"{a.name}-{slot}", pending=[s for s, _ in running.values()])[0]
                    running[pool.apply_async(work, ((a.name, wave, slot, spec, str(out)),))] = (spec, time.time())
                    slot += 1
            if done:
                prog = {"wave": wave, "running": len(running), "attempts": len(led), "accepted": sum(e.get("accepted", False) for e in led),
                        "selected": counts(led), "quota": quota(led), "dependent": sum(e.get("observed") == "DEPENDENT" for e in led),
                        "elapsed_h": round((time.time() - t_start) / 3600, 2), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
                (out / "PROGRESS.json").write_text(json.dumps(prog, indent=1))
                print(json.dumps(prog), flush=True)
            time.sleep(5)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent / "dataset-bd"))
    main()
