"""Resumable B/D dataset runner (both domains).

    python bd_run.py --domain 2d --name smoke2d --roots 20 --workers 8 --out /data/user1/rss_ebm_dataset/data/2d/smoke2d
    touch <out>/STOP          # clean stop after the current wave (in-flight roots finish, nothing is half-written)

Waves are planned deterministically from the ledger entries of EARLIER waves only (bd_core.plan_wave), so a restart replans the same wave
and skips its finished slots. Each root is constructed + labelled in a spawned worker under a per-root administrative wall cap (SIGALRM);
a cap hit or a construction failure is a ledger rejection with a reason, never a label. Per-root JSON writes are atomic."""
from __future__ import annotations

import argparse
import importlib
import json
import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from pathlib import Path

import bd_core as bc

ADMIN_WALL_S = {"2d": 1500, "3d": 2400, "3dt": 2400}


class WallCap(Exception):
    pass


def _alarm(sig, frame):
    raise WallCap("administrative wall cap")


def work(job):
    dom, name, wave, slot, spec, out = job
    A = importlib.import_module(f"bd_{dom}")
    G = A.env()
    ids = bc.ids(dom, name, wave, slot, spec, G["config_hash"], A.TEACHER_VERSION, G["protocol"])
    store = bc.Store(out)
    t0, c0 = time.time(), time.process_time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(ADMIN_WALL_S[dom])
    entry = {"wave": wave, "slot": slot, "spec": spec, "root_id": ids["root_id"], "parent_family_id": ids["parent_family_id"],
             "construction_seed": ids["construction_seed"]}
    try:
        c = A.construct(spec, ids["construction_seed"], ids["root_id"])
        t = A.teach(c, store, ids["root_id"])
        rec = A.record(spec, ids, c, t, store)
        store.save_root(rec)
        lab = rec["labels"]
        entry.update(accepted=True, verified_K=rec["verified_K"], observed=rec["observed"], y_original=lab["original"]["y"],
                     n_D=len(lab["D"]), n_B=len(lab["B"]), mask_valid=lab["blocker"]["blocker_supervision_valid"], calls=rec["teacher_calls"])
    except A.Reject as e:
        entry.update(accepted=False, reason=f"REJECT:{e}")
    except (A.AdminCap, WallCap) as e:
        entry.update(accepted=False, reason=f"ADMIN_CAP:{e}")
    except Exception as e:  # noqa: BLE001  (recorded, never silently dropped)
        entry.update(accepted=False, reason=f"ERROR:{type(e).__name__}:{str(e)[:200]}", trace=traceback.format_exc()[-1500:])
    finally:
        signal.alarm(0)
    entry.update(wall_s=time.time() - t0, cpu_s=time.process_time() - c0, pid=os.getpid())
    return entry


def run(dom, name, target, workers, wave_size, out, max_waves=10_000):
    out = Path(out)
    store = bc.Store(out)
    t_start = time.time()
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=25) as pool:
        while True:
            led = store.ledger()
            acc = sum(e.get("accepted", False) for e in led)
            waves = sorted({e["wave"] for e in led})
            wave = 0
            for w in waves:                                   # first wave that is not complete
                if sum(e["wave"] == w for e in led) >= wave_size:
                    wave = w + 1
                else:
                    wave = w
                    break
            if acc >= target or wave >= max_waves or (out / "STOP").exists():
                break
            if len(led) >= 3 * wave_size and acc == 0 or len(led) >= 6 * target + 4 * wave_size:   # broken design / runaway guard
                (out / "ABORTED.json").write_text(json.dumps({"why": "zero acceptance after 3 waves" if acc == 0 else "attempts > 6x target",
                                                              "ledger": len(led), "accepted": acc}))
                break
            A = importlib.import_module(f"bd_{dom}")
            allowed = [s for s in bc.SPECS if A.supports(s)]
            specs = bc.plan_wave([e for e in led if e["wave"] < wave], wave_size, wave, name, allowed)
            done = {e["slot"] for e in led if e["wave"] == wave}
            jobs = [(dom, name, wave, i, s, str(out)) for i, s in enumerate(specs) if i not in done]
            for e in pool.imap_unordered(work, jobs):
                store.append(e)
            led = store.ledger()
            prog = {"domain": dom, "name": name, "target": target, "wave": wave, "ledger": len(led), "accepted": sum(e.get("accepted", False) for e in led),
                    "rejected": sum(not e.get("accepted", False) for e in led), "elapsed_s": time.time() - t_start, "workers": workers,
                    "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
            (out / "PROGRESS.json").write_text(json.dumps(prog, indent=1))
            print(json.dumps(prog), flush=True)
    bc.manifest(store, dom)
    return store.ledger()


def bench(dom, out, workers=(1, 2, 4, 8), n=8):
    """Worker scaling on IDENTICAL jobs (same specs, same seeds): wall time, accepted roots per hour, CPU and RSS per worker."""
    import psutil
    A = importlib.import_module(f"bd_{dom}")
    specs = bc.plan_wave([], n, 0, "bench", [s for s in bc.SPECS if A.supports(s)])
    res = {}
    for w in workers:
        o = Path(out) / f"w{w}"
        jobs = [(dom, "bench", 0, i, s, str(o)) for i, s in enumerate(specs)]
        t0 = time.time()
        with mp.get_context("spawn").Pool(w) as pool:
            rss = []
            asy = pool.map_async(work, jobs)
            while not asy.ready():
                rss.append(sum(c.memory_info().rss for c in psutil.Process().children(recursive=True)) / 1e9)
                asy.wait(5)
            es = asy.get()
        wall = time.time() - t0
        acc = sum(e.get("accepted", False) for e in es)
        res[w] = {"wall_s": wall, "accepted": acc, "accepted_per_hour": 3600 * acc / wall, "cpu_s_sum": sum(e["cpu_s"] for e in es),
                  "peak_rss_gb": max(rss or [0])}
        print(w, json.dumps(res[w]), flush=True)
    (Path(out) / "BENCH.json").write_text(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    if sys.argv[1:2] == ["bench"]:
        print(bench(sys.argv[2], sys.argv[3], tuple(int(x) for x in sys.argv[4].split(","))))
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=["2d", "3d", "3dt"], required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--roots", type=int, required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--wave-size", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sys.setrecursionlimit(10000)
    run(a.domain, a.name, a.roots, a.workers, a.wave_size, a.out)
