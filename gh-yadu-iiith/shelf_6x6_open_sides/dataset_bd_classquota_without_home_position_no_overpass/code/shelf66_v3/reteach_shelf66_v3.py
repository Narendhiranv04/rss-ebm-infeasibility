"""Re-teach the accepted roots of the SHELF66V2 run (dataset_2500_raw) with the SHELF66V3 teacher (fingers checked pre-shaped) into the
SHELF66V3 run directory: same scenes, same root ids, new records / labels (run_shelf66_full.reteach: consistency checks + NONE audit gate).
The V3 runner then resumes from that ledger, so the re-taught roots count toward the quotas.

    python shelf66_v3/reteach_shelf66_v3.py --src <V2 run> --out <V3 run> --name srv --cap 18 --split split.json [--v2-pid PID]

  --split   {root_id: "lap" | "srv"}: which machine re-teaches which root (balanced by expected teacher time; made from the roots
            both ledgers had).  A root not in it (finished later) is re-taught by the machine whose name its id carries.
  --cap     busy processes allowed in total on this machine; while the V2 runner (--v2-pid, stopping with <src>/STOP) still finishes
            its last roots, its PROGRESS.json "running" count is subtracted, so the total never goes above the cap.
Resumable (roots already in <out>/ledger.jsonl are skipped); <out>/STOP stops it cleanly; progress in <out>/PROGRESS_reteach_<name>.json.
Exits when every root it owns is done and the V2 runner is gone."""
from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "dataset-bd"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))


def ledger(p):
    out = []
    if p.exists():
        for x in p.read_text().splitlines():
            try:
                out.append(json.loads(x))
            except ValueError:
                pass                          # a line still being written
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--name", required=True, choices=["lap", "srv"])
    ap.add_argument("--cap", type=int, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--v2-pid", type=int, default=0)
    ap.add_argument("--only", default="", help="comma-separated root ids (a test into a scratch --out)")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    import bd_core as bc
    import run_shelf66_full as RS              # V3 runner module: reteach() (spawned workers import it the same way)
    src, out = Path(a.src), Path(a.out)
    store = bc.Store(out)
    split = json.loads(Path(a.split).read_text())
    t_start = time.time()

    def v2_running():
        if not a.v2_pid or not Path(f"/proc/{a.v2_pid}").exists():
            return 0
        try:
            return int(json.loads((src / "PROGRESS.json").read_text())["running"])
        except (OSError, ValueError, KeyError):
            return a.cap                      # unreadable for a moment: wait

    def mine():
        """accepted V2 roots this machine owns, not re-taught yet; V2 NONE first (they may need the audit), then longest teacher first."""
        done = {e["root_id"] for e in ledger(out / "ledger.jsonl")}
        todo = [e for e in ledger(src / "ledger.jsonl") if e.get("accepted") and e["root_id"] not in done
                and split.get(e["root_id"], e["root_id"].split("-")[1]) == a.name and (not only or e["root_id"] in only)]
        cost = {}
        for e in todo:
            try:
                cost[e["root_id"]] = json.loads((src / "roots" / f"{e['root_id']}.json").read_text())["teacher_wall_s"]
            except (OSError, ValueError, KeyError):
                cost[e["root_id"]] = 0.0
        return sorted(todo, key=lambda e: (e.get("observed") != "NONE", -cost[e["root_id"]]))

    def progress(running):
        led = ledger(out / "ledger.jsonl")
        own = [e for e in led if e.get("retaught_from") and split.get(e["root_id"], e["root_id"].split("-")[1]) == a.name]
        prog = {"name": a.name, "retaught_here": len(own), "assigned_in_split": sum(v == a.name for v in split.values()), "running": running,
                "v2_running": v2_running(), "observed_here": dict(collections.Counter(str(e.get("observed", e.get("reason", "")[:20])) for e in own)),
                "transitions_here": dict(collections.Counter(f"{e['retaught_from'].get('observed')}->{e.get('observed', 'REJ')}" for e in own)),
                "elapsed_h": round((time.time() - t_start) / 3600, 2), "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}
        (out / f"PROGRESS_reteach_{a.name}.json").write_text(json.dumps(prog, indent=1))
        print(json.dumps(prog), flush=True)

    with mp.get_context("spawn").Pool(a.cap, maxtasksperchild=1) as pool:
        running, pending, last_scan, inflight = {}, mine(), time.time(), set()
        print(f"{len(pending)} roots to re-teach here ({a.name}); V2 runner running {v2_running()}", flush=True)
        while True:
            done = [r for r in running if r.ready()]
            for r in done:
                inflight.discard(running.pop(r))
                store.append(r.get())
            if done:
                progress(len(running))
            stop = (out / "STOP").exists()
            if not stop:
                while pending and len(running) < a.cap - v2_running():
                    e0 = pending.pop(0)
                    running[pool.apply_async(RS.reteach, ((e0, str(src), str(out)),))] = e0["root_id"]
                    inflight.add(e0["root_id"])
            if not pending and time.time() - last_scan > 60:        # roots the V2 runner finished since (or merged in since)
                pending, last_scan = [e for e in mine() if e["root_id"] not in inflight], time.time()
            if not running and (stop or (not pending and not v2_running() and not (a.v2_pid and Path(f"/proc/{a.v2_pid}").exists())
                                         and not [e for e in mine() if e["root_id"] not in inflight])):
                missing = sorted(k for k, v in split.items() if v == a.name and k not in {e["root_id"] for e in ledger(out / "ledger.jsonl")})
                progress(0)
                print(f"done ({'STOP' if stop else 'all roots of this machine re-taught'}); assigned roots not re-taught here: {len(missing)} "
                      f"{missing[:10]}", flush=True)
                break
            time.sleep(5)


if __name__ == "__main__":
    main()
