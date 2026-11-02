"""Stage 1 of the class-quota 2D dataset: base scenes (bd_2d_plus construction + bd_2d teacher) until every VERIFIED class reaches its quota
(5000 base scenes).  Same runner as run_3dt_plus.py; a verified repair whose relocations are not all independent is DEPENDENT and
not selected.  K = 2-4 with mechanism grasp / place / mixed come from the exact layout search (bd_2d_exact); a K=4 spec carries
intent "k4" (exact K=4 layout) or "none" (NONE construction).  Run from the latent-causal-tamp checkout (config.toml):

    cd ~/codebase/latent-causal-tamp && python ~/codebase/bd_samples/run_2d_plus.py --out ~/codebase/bd_samples/2d_plus_raw"""
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

QUOTA = {"NONE": 1000, "0": 1250, "1": 800, "2": 800, "3": 1150, "4": 800}   # K=4 movable-only infeasible (user, 1 Nov: K4_MODE harvest);
                                                                              # its 800 moved: K3 +500, K1 +150, K2 +150 (user)
QUOTA_MAX3 = {"NONE": 1000, "0": 1250, "1": 875, "2": 875, "3": 1000, "4": 0}
REQUEST_FOR = {"NONE": "N", "0": 0, "1": 1, "2": 2, "3": 3, "4": 4}   # "N": a K=4 spec built as NONE (intent "none")
K4_TRIAL, K4_MIN_YIELD = 60, 0.05
K4_MODE = "on"               # "on": QUOTA, K=4 always requested and never dropped (user, 1 Nov 2026); "trial": QUOTA_MAX3 if the
                             # exact K=4 yield stays below K4_MIN_YIELD; "interim": K=4 off, K1-K3 at the levels both quotas share;
                             # "harvest": QUOTA counted, but no K=4 requests (K=4 roots that arise anyway -- conversions, K=3 requests
                             # verifying 4 -- still count); <out>/K4_MODE overrides it while the runner runs
NONE_MODE = "on"             # "on": NONE requested (V5: movable blockers only); "pause": no NONE requests; <out>/NONE_MODE overrides
MIN_YIELD = 0.25
WALL_S = 2400                                         # exact layouts: ~5 min construction + teacher over up to 9 movers
GEN = "BD_GEN_V1+ROUTE2D_V6"                           # = bd_2d_plus.GENERATOR_VERSION
CURRENT = ("BD_GEN_V1+ROUTE2D_V4", "BD_GEN_V1+ROUTE2D_V5", GEN)   # K >= 2 movable-only construction (V6: shared extra blockers)
NONE_GENS = ("BD_GEN_V1+ROUTE2D_V5", GEN)              # generators of the current NONE construction
EXACT = ("grasp", "place", "mixed")                   # mechanisms of the exact layout search (K >= 3 is requested only for these)


def effective(led):
    """The ledger as counted: a root replaced by its all-movable conversion (bd_2d_convert: line with converted_from) drops out, and
    a conversion done on both machines counts once (its accepted line first)."""
    gone = {e["converted_from"] for e in led if e.get("converted_from")}
    if not gone:
        return led
    pick = {}
    for i, e in enumerate(led):
        rid = e.get("root_id")
        if e.get("converted_from") and rid and (rid not in pick or (e.get("accepted") and not led[pick[rid]].get("accepted"))):
            pick[rid] = i
    return [e for i, e in enumerate(led) if e.get("root_id") not in gone and (not e.get("converted_from") or pick.get(e.get("root_id")) == i)]


def level(spec):
    """Request level of a spec: K, or "N" for a K=4 spec built as NONE (every K=4 spec without intent "k4")."""
    return ("N" if spec.get("intent", "none") == "none" else 4) if spec["K"] == 4 else spec["K"]


class WallCap(Exception):
    pass


def _alarm(sig, frame):
    raise WallCap("administrative wall cap")


def work(job):
    import bd_2d as A
    import bd_2d_plus as P
    import fastgeom
    fastgeom.install()                                 # same distances as geometry.py to rounding, ~2.5x faster (fastgeom.py)
    name, wave, slot, spec, out = job
    G = A.env()
    ids = bc.ids("2d", name, wave, slot, spec, G["config_hash"], A.TEACHER_VERSION, G["protocol"])
    store = bc.Store(out)
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(WALL_S)
    entry = {"gen": P.GENERATOR_VERSION, "geom": fastgeom.VERSION, "wave": wave, "slot": slot, "spec": spec, "root_id": ids["root_id"], "construction_seed": ids["construction_seed"]}
    try:
        c = P.construct(spec, ids["construction_seed"], ids["root_id"])
        t = A.teach(c, store, ids["root_id"])
        rec = P.record(spec, ids, c, t, store)
        observed = str(rec["observed"])
        ms = next((b["sequence"] for b in rec["labels"]["B"] if b["role"] == "minimal" and b["y"]), [])
        if ms and not P.independent(c, ids["root_id"], ms):
            observed = "DEPENDENT"
        rec["independent_moves"] = observed != "DEPENDENT"
        rec["geometry_impl"] = fastgeom.VERSION
        store.save_root(rec)
        entry.update(accepted=True, verified_K=rec["verified_K"], observed=observed, y_original=rec["labels"]["original"]["y"],
                     method=c["construction"].get("method"))
    except (A.Reject, A.AdminCap, WallCap) as e:
        entry.update(accepted=False, reason=f"{type(e).__name__}:{e}")
    except Exception as e:  # noqa: BLE001  (recorded, never silently dropped)
        entry.update(accepted=False, reason=f"ERROR:{type(e).__name__}:{str(e)[:200]}", trace=traceback.format_exc()[-1500:])
    finally:
        signal.alarm(0)
    entry["wall_s"] = time.time() - t0
    return entry


def quota(led):
    """QUOTA, or QUOTA_MAX3 once exact K=4 layouts of the current generator were tried K4_TRIAL times with yield < K4_MIN_YIELD."""
    if K4_MODE == "interim":
        return {**QUOTA, "4": 0}
    if K4_MODE in ("on", "harvest"):
        return QUOTA
    k4 = [e for e in led if level(e["spec"]) == 4 and e.get("gen") == GEN]
    hit = sum(e.get("accepted", False) and e["observed"] == "4" for e in k4)
    return QUOTA_MAX3 if len(k4) >= K4_TRIAL and hit < K4_MIN_YIELD * len(k4) else QUOTA


def counts(led):
    Q = quota(led)
    got = {k: 0 for k in Q}
    for e in effective(led):
        if e.get("accepted") and e["observed"] in got and got[e["observed"]] < Q[e["observed"]]:
            got[e["observed"]] += 1
    return got


def plan(led, n, wave, name, allowed, pending=()):
    """Request levels in proportion to sum over classes of (remaining quota x P(class | level)), learned from the ledger (levels 2-4:
    roots of the current generator only).  pending: specs of roots still running, counted in bd_core's factor balance."""
    got, Q = counts(led), quota(led)
    need = {k: Q[k] - got[k] for k in Q if Q[k] > got[k]}
    weight = {}
    for r in (0, 1, 2, 3, 4, "N"):                    # expected useful roots per request of level r, learned from the ledger
        if (r == 4 and (not Q["4"] or K4_MODE == "harvest")) or (r == "N" and NONE_MODE == "pause"):
            continue
        past = [e for e in led if level(e["spec"]) == r and "converted_from" not in e   # generator attempts only
                and (r in (0, 1) or e.get("gen") in (CURRENT if r != "N" else NONE_GENS))]
        w = sum(need[k] * (sum(e.get("accepted", False) and e["observed"] == k for e in past) + (1.0 if REQUEST_FOR[k] == r else 0.1))
                / (len(past) + 2) for k in need)
        if w > 0:
            weight[r] = w
    total = sum(weight.values())
    weight = {r: max(w / total, 0.05 if any(REQUEST_FOR[k] == r for k in need) else 0.0)     # >= 5% (exploration) while
              for r, w in weight.items()}                                                    # the requested class needs roots
    rng = np.random.default_rng(int(bc.sha([name, wave])[:8], 16))
    levels = sorted(weight, key=str)
    req = rng.choice(len(levels), size=n, p=np.array([weight[r] for r in levels]) / sum(weight.values()))
    specs = []
    for i, j in enumerate(req):
        r = levels[j]
        k = 4 if r == "N" else r
        pool = [s for s in allowed if s["K"] == k and (k < 3 or r == "N" or s["mechanism"] in EXACT)]
        if r in (2, 3, 4, "N"):      # start area x move length x mechanism drawn by its verified yield so far (prior 0.5 / 2);
                                     # K=2: keeps the slow route-built transfer combos (~3% yield) from taking most K=2 slots
            combos = sorted({(s["source"], s["displacement"], s["mechanism"]) for s in pool})
            mine = [e for e in led if level(e["spec"]) == r and e.get("gen") in (CURRENT if r != "N" else NONE_GENS)]
            tried = collections.Counter((e["spec"]["source"], e["spec"]["displacement"], e["spec"]["mechanism"]) for e in mine)
            hits = collections.Counter((e["spec"]["source"], e["spec"]["displacement"], e["spec"]["mechanism"]) for e in mine
                                       if e.get("accepted") and e.get("observed") == ("NONE" if r == "N" else str(k)))
            w = np.array([(hits[c] + 0.5) / (tried[c] + 2) for c in combos])
            c = combos[rng.choice(len(combos), p=w / w.sum())]
            pool = [s for s in pool if (s["source"], s["displacement"], s["mechanism"]) == c]
        spec = bc.plan_wave(effective(led) + [{"accepted": True, "spec": s, "verified_K": s["K"]} for s in list(pending) + specs], 1, wave,
                            f"{name}-{i}", pool)[0]
        specs.append({**spec, "intent": "none" if r == "N" else "k4"} if k == 4 else spec)
    return specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", default="plus2d")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--wave-size", type=int, default=24)
    a = ap.parse_args()
    import bd_2d as A
    allowed = [s for s in bc.SPECS if A.supports(s)]
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
            try:
                mode = (out / "K4_MODE").read_text().strip()
                globals()["K4_MODE"] = mode if mode in ("interim", "trial", "on", "harvest") else K4_MODE
            except OSError:
                pass
            try:
                mode = (out / "NONE_MODE").read_text().strip()
                globals()["NONE_MODE"] = mode if mode in ("pause", "on") else NONE_MODE
            except OSError:
                pass
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
                    spec = plan(led, 1, wave, f"{a.name}-{slot}", allowed, pending=[s for s, _ in running.values()])[0]
                    running[pool.apply_async(work, ((a.name, wave, slot, spec, str(out)),))] = (spec, time.time())
                    slot += 1
            if done:
                prog = {"wave": wave, "running": len(running), "attempts": len(led), "accepted": sum(e.get("accepted", False) for e in led),
                        "selected": counts(led), "quota": quota(led), "dependent": sum(e.get("observed") == "DEPENDENT" for e in led),
                        "converted": sum(bool(e.get("converted_from")) for e in led),
                        "elapsed_h": round((time.time() - t_start) / 3600, 2), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
                (out / "PROGRESS.json").write_text(json.dumps(prog, indent=1))
                print(json.dumps(prog), flush=True)
            time.sleep(5)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
