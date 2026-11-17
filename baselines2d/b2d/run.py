"""Run the baselines on a split and verify their outputs with the dataset's finalized X / Y checker.

    python -m b2d.run --tag main --split test --methods em4m,msmcts,mosegman --seeds 0 --workers 6 [--limit N] [--roots a,b]

One task = one root: every (method, seed) of the root runs in one worker, sharing the root's exact motion cache (persisted to
out/cache/<root>.pkl.gz so that later passes, e.g. more seeds, reuse it). Each run gets BUDGET CPU-seconds (virtual accounting, see
teacher.py). Every distinct REPAIR is then verified (X with connector + replay, Y from q_ref + replay); travel = joint travel of the
witnessed relocations + connector + target (approach_b_final/cost.jrec). Rows are appended to out/runs/<tag>/results.jsonl (resumable)."""
from __future__ import annotations

import argparse
import gzip
import json
import multiprocessing as mp
import pickle
import signal
import time
import traceback
import zlib
from pathlib import Path

import numpy as np

from . import dataset as D
from . import paths as P

METHODS = {"em4m": "b2d.em4m2d", "msmcts": "b2d.msmcts", "mosegman": "b2d.mosegman"}
BUDGET = 300.0
ADMIN_WALL = 3600            # per run; a hit is CENSORED, never a label


class AdminWall(Exception):
    pass


def _alarm(sig, frm):
    raise AdminWall("administrative wall cap")


def _boot(geom):
    from . import teacher as TT
    TT.boot(geom)
    signal.signal(signal.SIGALRM, _alarm)


def jrec(q):
    q = np.asarray(q, float)
    return float((np.abs(np.diff(q, axis=0)) / (2 * np.pi)).sum()) if len(q) > 1 else 0.0


def travel(ver) -> float | None:
    ws = list(ver["x"].get("witnesses") or []) + ([ver["y"]["witness"]] if ver["y"].get("witness") else [])
    tot = 0.0
    for w in ws:
        f = P.OUT / "witnesses" / "witnesses" / w[:2] / f"{w}.npz"
        if not f.exists():
            f = P.OUT / "witnesses" / w[:2] / f"{w}.npz"
        if not f.exists():
            return None
        z = np.load(f)
        tot += jrec(z["connector_q"] if "connector_q" in z.files else z["q"])
    return tot


def cache_path(rid):
    return P.OUT / "cache" / f"{rid}.pkl.gz"


def load_shared(rid):
    from . import teacher as TT
    sh = TT.Shared()
    f = cache_path(rid)
    if f.exists():
        try:
            with gzip.open(f, "rb") as fh:
                sh.cache, sh.memo = pickle.load(fh)
        except Exception:  # noqa: BLE001  (a corrupt cache only costs recomputation)
            sh = TT.Shared()
    return sh


def save_shared(rid, sh):
    f = cache_path(rid)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    with gzip.open(tmp, "wb", compresslevel=3) as fh:
        pickle.dump((sh.cache, sh.memo), fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(f)


def job(task):
    import importlib
    from . import teacher as TT
    rid, todo, budget = task["root"], task["todo"], task["budget"]
    rec = D.record(rid)
    sh = load_shared(rid)
    rows = []
    for method, seed in todo:
        solve = importlib.import_module(METHODS[method]).solve
        rng = np.random.default_rng([seed, zlib.crc32(rid.encode())])
        m = TT.Motion(rec, sh, budget_s=budget)
        t0 = time.time()
        row = {"root": rid, "K": D.masks()[rid]["K"], "method": method, "seed": seed, "budget_s": budget}
        signal.alarm(ADMIN_WALL)
        try:
            res = solve(m, rng)
            row.update(status=res.status, seq=[[o, int(c)] for o, c in res.seq], reason=res.reason, stats=res.stats)
        except TT.Budget as e:
            row.update(status="ABSTAIN", seq=[], reason=f"budget ({e})")
        except AdminWall:
            row.update(status="CENSORED", seq=[], reason="administrative wall cap")
        except Exception as e:  # noqa: BLE001
            row.update(status="ERROR", seq=[], reason=f"{type(e).__name__}: {str(e)[:300]}", trace=traceback.format_exc()[-1500:])
        finally:
            signal.alarm(0)
        row.update(cpu_s=round(m.used(), 2), vcpu_s=round(m.vcpu, 2), wall_s=round(time.time() - t0, 1), calls=dict(m.calls),
                   n_calls=m.n_calls(), capped=m.n_capped, object_checks=m.n_object_checks, trace=m.trace)
        rows.append(row)
    verified = {}
    for row in rows:
        if row["status"] != "REPAIR":
            continue
        k = json.dumps(row["seq"])
        if k not in verified:
            t0 = time.process_time()
            try:
                v = TT.verify(rec, row["seq"], sh.cache)
                verified[k] = {"x": v["x"]["x"], "y": v["y"]["y"], "x_status": v["x"]["status"], "x_phase": v["x"].get("phase"),
                               "y_status": v["y"]["status"], "travel": travel(v) if v["x"]["x"] == 1 and v["y"]["y"] == 1 else None,
                               "verify_cpu_s": round(time.process_time() - t0, 1)}
            except Exception as e:  # noqa: BLE001
                verified[k] = {"x": None, "y": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        row["verify"] = verified[k]
    save_shared(rid, sh)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--split", default="test")
    ap.add_argument("--methods", default="em4m,msmcts,mosegman")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--budget", type=float, default=BUDGET)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--roots", default=None, help="comma-separated root ids (overrides --split)")
    ap.add_argument("--classes", default=None, help="comma-separated K classes to keep, e.g. 1,2,3")
    a = ap.parse_args()
    methods, seeds = a.methods.split(","), [int(s) for s in a.seeds.split(",")]
    out = P.OUT / "runs" / a.tag
    out.mkdir(parents=True, exist_ok=True)
    res_f = out / "results.jsonl"
    done = set()
    if res_f.exists():
        for l in open(res_f):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:                                  # a line cut by a crash / power loss: that root is redone
                continue
            done.add((r["root"], r["method"], r["seed"]))
    roots = a.roots.split(",") if a.roots else D.roots(a.split)
    if a.classes:
        keep = set(a.classes.split(","))
        roots = [r for r in roots if D.masks()[r]["K"] in keep]
    roots = roots[:a.limit] if a.limit else roots
    tasks = {}
    for rid in roots:
        todo = [(mth, s) for s in seeds for mth in methods if (rid, mth, s) not in done]
        if todo:
            tasks.setdefault(D.geom_of(D.record(rid)), []).append({"root": rid, "todo": todo, "budget": a.budget})
    n = sum(len(v) for v in tasks.values())
    print(json.dumps({"tag": a.tag, "roots": len(roots), "tasks": n, "by_geom": {g: len(v) for g, v in tasks.items()},
                      "methods": methods, "seeds": seeds, "budget_s": a.budget, "workers": a.workers}), flush=True)
    t0, k = time.time(), 0
    with open(res_f, "a") as f:
        for geom, ts in sorted(tasks.items()):
            ts.sort(key=lambda t: {"NONE": 0, "3": 1, "2": 2, "1": 3, "0": 4}[D.masks()[t["root"]]["K"]])   # long tasks first
            with mp.get_context("spawn").Pool(a.workers, initializer=_boot, initargs=(geom,), maxtasksperchild=20) as pool:
                for rows in pool.imap_unordered(job, ts):
                    for r in rows:
                        f.write(json.dumps(r) + "\n")
                    f.flush()
                    k += 1
                    s = " ".join(f"{r['method'][:4]}/{r['seed']}:{r['status'][:3]}{'+' if r.get('verify', {}).get('x') == 1 and r.get('verify', {}).get('y') == 1 else ''}"
                                 f"({r['cpu_s']:.0f}s)" for r in rows)
                    print(f"{time.strftime('%H:%M:%S')} [{k}/{n}] {rows[0]['root']} K={rows[0]['K']} {s}  elapsed {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
