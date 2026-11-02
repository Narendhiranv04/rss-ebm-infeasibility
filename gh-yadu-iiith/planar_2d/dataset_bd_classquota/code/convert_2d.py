"""Converts every counted 2D root that contains immovable context structures into its all-movable version (bd_2d_convert) and
appends one ledger line per old root (converted_from = the old root id) to <out>/ledger.jsonl; run_2d_plus then counts the new root
instead of the old one.  Resumable: roots already converted on either machine (the ledgers are merged) are skipped.  Run on both
machines with opposite --order so they meet in the middle.  <out>/CONVERT_MAX_RUNNING lowers the parallelism while it runs;
<out>/CONVERT_STOP (or the runner's / watchdog's <out>/STOP) stops it after the running jobs.  --handback N writes N to the runner's
<out>/MAX_RUNNING when every root is converted, so the runner takes the cores back.  Run from the latent-causal-tamp checkout:

    python convert_2d.py --out RUN_DIR --workers N [--order down] [--handback N]"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

import bd_core as bc

ORDER = {"4": 0, "3": 1, "2": 2, "NONE": 3, "1": 4, "0": 5}     # classes that may change first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--order", choices=("up", "down"), default="up")
    ap.add_argument("--handback", type=int, default=None)
    a = ap.parse_args()
    import bd_2d_convert as C
    out = Path(a.out)
    store = bc.Store(out)
    import bd_2d as A
    G = A.env()
    todo, retired = [], 0
    for rid, cls, spec, rec in C.todo(store.ledger(), str(out)):
        k = C.static_bound(rec, G) if cls == "NONE" else None
        if cls == "NONE" and (k is None or k < 3):     # identical line on both machines: the merge keeps one
            store.append(C.retire_line(rid, spec, k))
            retired += 1
        else:
            todo.append((rid, cls, spec))
    todo.sort(key=lambda rc: (ORDER[rc[1]], bc.sha(["conv", rc[0]])), reverse=a.order == "down")
    print(f"{time.strftime('%FT%T')} {len(todo)} roots to convert ({a.order}), {retired} old NONE roots retired", flush=True)
    running, n_done, stopped = {}, 0, False
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=10) as pool:
        while todo or running:
            for r in [r for r in running if r.ready()]:
                e = r.get()
                store.append(e)
                running.pop(r)
                n_done += 1
                print(json.dumps({k: e.get(k) for k in ("converted_from", "old_observed", "observed", "reason")} | {"wall_s": round(e["wall_s"])}),
                      flush=True)
            for r in [r for r, (_, t0) in running.items() if time.time() - t0 > C.WALL_S + 900]:
                print(f"lost job {running[r][0]} (worker died?): slot freed", flush=True)
                running.pop(r)
            stopped = (out / "CONVERT_STOP").exists() or (out / "STOP").exists()
            if stopped:
                todo = []
            cap = a.workers
            try:
                cap = min(cap, int((out / "CONVERT_MAX_RUNNING").read_text()))
            except (OSError, ValueError):
                pass
            if todo and len(running) < cap:
                done = {e["converted_from"] for e in store.ledger() if e.get("converted_from")} | {rid for rid, _ in running.values()}
                todo = [rc for rc in todo if rc[0] not in done]          # also those the other machine finished meanwhile
                while todo and len(running) < cap:
                    rid, _, spec = todo.pop(0)
                    running[pool.apply_async(C.work, ((rid, str(out), str(out), spec),))] = (rid, time.time())
            time.sleep(5)
    print(f"{time.strftime('%FT%T')} finished: {n_done} converted here", flush=True)
    if a.handback and not stopped:
        (out / "MAX_RUNNING").write_text(f"{a.handback}\n")
        print(f"runner MAX_RUNNING -> {a.handback}", flush=True)


if __name__ == "__main__":
    sys.setrecursionlimit(10000)
    main()
