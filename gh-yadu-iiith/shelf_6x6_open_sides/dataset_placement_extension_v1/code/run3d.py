"""Bounded mock coordinator: pinned single-thread workers, atomic per-root units, resumable ledger, fail-closed host monitor.

  python run3d.py RUN_DIR --roster R --policy adaptive_v1 --workers 12 [--cpus 0-11] [--subset all|reference|sweep36|sweep8|FILE]
                  [--roots a,b] [--guard-cpu-s 1200]
A unit = one root lineage (all its selected queries, slot order) on ONE worker; IK branch cache shared inside the unit only.
Never writes outside RUN_DIR.  Refuses: > 18 logical CPUs, a CPU outside the allowed affinity, a RUN_DIR made under another
policy / roster / code version.  SIGKILL of this process group at any time is safe: rerun the same command to resume.
"""
import argparse, gzip, hashlib, json, multiprocessing as mp, os, signal, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CEILING = 18
TEACHER_CODE = ("placement3d.py", "policy3d.py", "plan3d.py", "run3d.py")   # frozen teacher + collector (analysis / figure code excluded)
LIMITS = {"mem_avail_gb": 4.0, "swap_growth_gb": 1.0, "disk_free_gb": 5.0, "worker_rss_gb": 2.0, "pkg_temp_pause_c": 90.0,
          "pkg_temp_stop_c": 95.0, "temp_resume_c": 82.0}


def cpus_of(s):
    out = []
    for p in s.split(","):
        a, _, b = p.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def default_cpus(n):
    """Physical cores first (CPU k and k+12 share core k on this Xeon), then SMT siblings; >= 6 logical CPUs stay outside."""
    return (list(range(12)) + list(range(12, 24)))[:n]


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_units(roster, subset, roots, by="root"):
    rows = [json.loads(x) for x in gzip.open(roster, "rt")]
    rows = [r for r in rows if not r["static"] and r["split"] == "train"]
    if subset == "reference":
        rows = [r for r in rows if r["reference"]]
    elif subset == "sweep":                                   # matched worker-sweep subset: root-scene slots 0-2 of every roster root
        rows = [r for r in rows if r["depth"] == 0 and r["slot"] in (0, 1, 2)]
    elif subset == "serial":                                  # serial overlap: slot 1 (mid) of every roster root (subset of 'sweep')
        rows = [r for r in rows if r["depth"] == 0 and r["slot"] == 1]
    elif subset != "all":
        keep = set(Path(subset).read_text().split())
        rows = [r for r in rows if r["qid"] in keep]
    if roots:
        rows = [r for r in rows if r["root_id"] in roots]
    units = {}
    for r in rows:
        units.setdefault(r["root_id"] if by == "root" else r["qid"].replace(":", "_"), []).append(r)
    return {k: sorted(v, key=lambda r: r["slot"]) for k, v in sorted(units.items())}


def worker(cpu, tasks, results, run, stages, guard):
    os.sched_setaffinity(0, {cpu})
    os.nice(10)
    import numpy as np
    import placement3d as P
    import policy3d as Q
    early = stages != Q.POLICIES["reference_v1"]

    def on_guard(*_):
        raise Q.Interrupted("per-query CPU guard")
    signal.signal(signal.SIGPROF, on_guard)
    roots = {}
    while True:
        item = tasks.get()
        if item is None:
            return
        uid, qs = item
        t0, c0, out, arrays, ikc = time.time(), time.process_time(), [], {}, {}
        root = json.loads((P.SRC / "raw/roots" / f"{qs[0]['root_id']}.json").read_text())
        scenes = {}
        for q in qs:
            tq, cq = time.time(), time.process_time()
            assert q["root_id"] == root["ids"]["root_id"]
            try:
                sc = scenes.get(q["scene_hash"]) or scenes.setdefault(q["scene_hash"], P.Scene(root["objects"], q["cells"], ikc))
                signal.setitimer(signal.ITIMER_PROF, guard)
                try:
                    rec, w = Q.evaluate(sc, q["object"], tuple(q["dest"]), stages, early_stop=early)
                finally:
                    signal.setitimer(signal.ITIMER_PROF, 0)
            except Q.Interrupted as err:
                rec, w = {"label": None, "status": "INTERRUPTED", "reason": str(err)}, None
            if w is not None:
                arrays.update({f"{rec['witness_id']}.{k}": v for k, v in w.items()})
            out.append({**{k: q[k] for k in ("qid", "slot", "depth", "scene_hash", "object", "dest", "band", "shape", "cheb", "n_obj", "K",
                                             "reference", "dest_is_t_goal", "clutter")}, **rec, "wall_s": time.time() - tq,
                        "cpu_query_s": time.process_time() - cq})
        doc = {"unit": uid, "records": out, "protocol": P.protocol()[0], "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
        u = Path(run) / "units"
        tmp = u / f".{uid}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(doc, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        os.replace(tmp, u / f"{uid}.json")
        if arrays:
            tmpz = u / f".{uid}.{os.getpid()}.tmp.npz"
            np.savez_compressed(tmpz, **arrays); os.replace(tmpz, u / f"{uid}.npz")
        results.put((uid, cpu, doc["wall_s"], doc["cpu_s"], len(out)))


def sensors():
    for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:                                                   # some hwmon entries have no 'name' (seen on gvlab: hwmon2)
            if (h / "name").read_text().strip() == "coretemp":
                return int((h / "temp1_input").read_text()) / 1000
        except OSError:
            continue
    return None


def monitor(run, procs, n, stop, state):
    """Fail closed: any unexpected monitor error stops the run (a run is never left unmonitored)."""
    try:
        _monitor(run, procs, n, stop, state)
    except Exception as err:  # noqa: BLE001
        state["safety_stop"] = [f"monitor_error: {type(err).__name__}: {err}"[:200]]
        stop.set()


def _monitor(run, procs, n, stop, state):
    import psutil
    me, swap0, log = psutil.Process(), psutil.swap_memory().used, open(Path(run) / "monitor.jsonl", "a")
    paused = False
    while not stop.is_set():
        ps = [me] + [psutil.Process(p.pid) for p in procs if p.is_alive()]
        cpu = [p.cpu_percent(None) for p in ps]
        time.sleep(2.0)
        try:
            cpu = [p.cpu_percent(None) for p in ps]
            rss = [p.memory_info().rss / 2 ** 30 for p in ps]
            kids = sum(len(p.children(recursive=True)) for p in ps[1:])
        except psutil.NoSuchProcess:
            continue
        vm, t, du = psutil.virtual_memory(), sensors(), psutil.disk_usage(run)
        aff = set().union(*[os.sched_getaffinity(p.pid) for p in ps])
        dead = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
        rec = {"t": time.time(), "cpu_pct_sum": sum(cpu), "workers_alive": len(ps) - 1, "rss_max_gb": max(rss), "descendants": kids,
               "mem_avail_gb": vm.available / 2 ** 30, "swap_growth_gb": (psutil.swap_memory().used - swap0) / 2 ** 30,
               "disk_free_gb": du.free / 2 ** 30, "pkg_temp_c": t, "load1": os.getloadavg()[0], "host_cpu_pct": psutil.cpu_percent(None),
               "done": state["done"], "affinity_union": sorted(aff)}
        bad = [k for k, ok in (("cpu_ceiling", sum(cpu) <= 100 * (min(n, CEILING) + 1)), ("mem", rec["mem_avail_gb"] >= LIMITS["mem_avail_gb"]),
                               ("swap", rec["swap_growth_gb"] <= LIMITS["swap_growth_gb"]), ("disk", rec["disk_free_gb"] >= LIMITS["disk_free_gb"]),
                               ("rss", rec["rss_max_gb"] <= LIMITS["worker_rss_gb"]), ("nested_processes", kids == 0),
                               ("temp", t is None or t < LIMITS["pkg_temp_stop_c"]), ("affinity", len(aff) <= min(n, CEILING)),
                               ("worker_death", not dead)) if not ok]
        if t is not None and not paused and t >= LIMITS["pkg_temp_pause_c"]:
            [os.kill(p.pid, signal.SIGSTOP) for p in procs if p.is_alive()]; paused = True
        elif paused and (t is None or t <= LIMITS["temp_resume_c"]):
            [os.kill(p.pid, signal.SIGCONT) for p in procs if p.is_alive()]; paused = False
        rec.update(paused=paused, violations=bad)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if bad:
            state["safety_stop"] = bad
            stop.set()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--roster", required=True); ap.add_argument("--policy", default="adaptive_v1")
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--cpus"); ap.add_argument("--subset", default="all")
    ap.add_argument("--roots", default=""); ap.add_argument("--guard-cpu-s", type=float, default=1200.0)
    ap.add_argument("--unit", default="root", choices=("root", "query"))
    a = ap.parse_args()
    import policy3d as Q
    import placement3d as P
    stages = (Q.POLICIES | Q.FIXED)[a.policy]
    cpus = cpus_of(a.cpus) if a.cpus else default_cpus(a.workers)
    allowed = os.sched_getaffinity(0)
    if a.workers > CEILING or len(cpus) != a.workers or len(set(cpus)) != len(cpus) or not set(cpus) <= allowed:
        sys.exit(f"REFUSED: {a.workers} workers on CPUs {cpus} (ceiling {CEILING}, allowed {sorted(allowed)})")
    os.setpgrp()
    os.sched_setaffinity(0, set(cpus))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    run = Path(a.run); (run / "units").mkdir(parents=True, exist_ok=True)
    units = load_units(a.roster, a.subset, set(filter(None, a.roots.split(","))), a.unit)
    code = {n: sha_file(HERE / n) for n in TEACHER_CODE}
    man = {"policy": a.policy, "stages": stages, "roster_sha256": sha_file(a.roster), "subset": a.subset, "roots": a.roots,
           "units": {k: [q["qid"] for q in v] for k, v in units.items()}, "protocol": P.protocol(), "code": code, "guard_cpu_s": a.guard_cpu_s}
    mp_ = run / "run_manifest.json"
    if mp_.exists():
        old = json.loads(mp_.read_text())
        if {k: old[k] for k in man} != json.loads(json.dumps(man)):
            sys.exit("REFUSED: RUN_DIR holds a different frozen run (policy / roster / code / protocol differ)")
    else:
        mp_.write_text(json.dumps(man, indent=1, sort_keys=True))
    led = run / "ledger.jsonl"
    done = {}
    for x in (led.read_text().splitlines() if led.exists() else []):
        d = json.loads(x)
        if d.get("event") == "unit_done" and (run / "units" / f"{d['unit']}.json").exists() and sha_file(run / "units" / f"{d['unit']}.json") == d["sha256"]:
            done[d["unit"]] = d
    todo = [u for u in units if u not in done]
    log = open(led, "a")
    log.write(json.dumps({"event": "start", "t": time.time(), "pid": os.getpid(), "workers": a.workers, "cpus": cpus, "todo": len(todo),
                          "resumed": len(done)}) + "\n"); log.flush()
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    for u in sorted(todo, key=lambda u: -len(units[u])):
        tasks.put((u, units[u]))
    procs = [ctx.Process(target=worker, args=(c, tasks, results, str(run), stages, a.guard_cpu_s), daemon=True) for c in cpus[:max(1, min(a.workers, len(todo)))]]
    for p in procs:
        tasks.put(None); p.start()
    stop, state = threading.Event(), {"done": len(done)}
    th = threading.Thread(target=monitor, args=(str(run), procs, a.workers, stop, state), daemon=True); th.start()
    t0, n = time.time(), 0
    while n < len(todo) and not stop.is_set():
        try:
            uid, cpu, wall, cs, nq = results.get(timeout=1.0)
        except Exception:  # noqa: BLE001  (queue.Empty)
            if not any(p.is_alive() for p in procs) and results.empty():
                state["safety_stop"] = ["worker_death"]; break
            continue
        n += 1; state["done"] += 1
        log.write(json.dumps({"event": "unit_done", "unit": uid, "cpu": cpu, "wall_s": wall, "cpu_s": cs, "queries": nq, "t": time.time(),
                              "sha256": sha_file(run / "units" / f"{uid}.json")}) + "\n"); log.flush()
    stop.set()
    for p in procs:
        p.terminate() if state.get("safety_stop") else p.join(timeout=60)
    end = {"event": "end", "t": time.time(), "wall_s": time.time() - t0, "units_done_now": n, "complete": n == len(todo),
           "safety_stop": state.get("safety_stop")}
    log.write(json.dumps(end) + "\n"); log.flush()
    print(json.dumps(end))
    sys.exit(0 if end["complete"] else 3)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
