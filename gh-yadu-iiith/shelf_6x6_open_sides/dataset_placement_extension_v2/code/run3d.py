"""Bounded collector V2: pinned single-thread workers, atomic per-unit JSON + NPZ, hash-validated resume, frozen source pin, fail-closed
host monitor (temperature-sensor loss, CPU-affinity union, descendant identity, memory / swap / disk / RSS).

  python run3d.py RUN_DIR --roster R --policy adaptive_v1 --workers 12 [--cpus 0-11] [--subset all|FILE] [--unit root|query]
                  [--guard-cpu-s 1200] [--source-pin PIN.json] [--dry-run]
  production (all splits) ONLY with:  --production --authorization production_policy.json --confirm TOKEN   (guard3d.check)

A unit = one root lineage (unit=root) or one query (unit=query) on ONE worker; the IK cache is shared inside a unit only.
Write order per unit: NPZ (tmp -> os.replace), then JSON (tmp -> os.replace, it records the NPZ sha256 and every witness id), then the
coordinator re-validates both and appends the ledger line.  On resume a unit counts as done only if its ledger line, JSON sha256, NPZ
sha256 and every witness re-hash agree; anything else (absent / truncated / mismatched / orphan files) is quarantined and recomputed.
Refuses: > 18 logical CPUs in the mask, a CPU outside the allowed affinity, an unreadable temperature sensor, a RUN_DIR frozen under a
different policy / roster / code / protocol / worker policy / source pin, a source root whose bytes differ from the pin.
SIGKILL of this process group at any time is safe: rerun the same command to resume.
"""
import argparse, gzip, hashlib, io, json, multiprocessing as mp, os, shutil, signal, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CEILING = 18
TEACHER_CODE = ("placement3d.py", "policy3d.py", "plan3d.py", "run3d.py", "guard3d.py")
LIMITS = {"mem_avail_gb": 4.0, "swap_growth_gb": 1.0, "disk_free_gb": 5.0, "worker_rss_gb": 2.0, "pkg_temp_pause_c": 90.0,
          "pkg_temp_stop_c": 95.0, "temp_resume_c": 82.0}
SENSOR_OVERRIDE = os.environ.get("RUN3D_SENSOR_OVERRIDE")        # TEST ONLY (a file standing in for temp1_input); refused in production


class SensorError(Exception):
    pass


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
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sensors():
    """-> (package temperature C, sensor identity).  Raises SensorError if no readable coretemp sensor exists (fail closed)."""
    if SENSOR_OVERRIDE:
        try:
            return int(Path(SENSOR_OVERRIDE).read_text()) / 1000, f"override:{SENSOR_OVERRIDE}"
        except (OSError, ValueError) as err:
            raise SensorError(f"override:{SENSOR_OVERRIDE}: {err}") from err
    errs = []
    for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:                                                   # some hwmon entries have no 'name' (seen on gvlab: hwmon2)
            name = (h / "name").read_text().strip()
        except OSError:
            continue
        if name == "coretemp":
            try:
                return int((h / "temp1_input").read_text()) / 1000, f"{h}/temp1_input ({name})"
            except (OSError, ValueError) as err:
                errs.append(f"{h}: {err}")
    raise SensorError("no readable coretemp sensor" + (f" ({'; '.join(errs)})" if errs else ""))


def load_units(roster, subset, roots, by="root", splits=("train",)):
    rows = [json.loads(x) for x in gzip.open(roster, "rt")]
    n_all = len(rows)
    rows = [r for r in rows if not r["static"] and r["split"] in splits]
    if subset not in ("all", None):
        keep = set(Path(subset).read_text().split())
        rows = [r for r in rows if r["qid"] in keep]
    if roots:
        rows = [r for r in rows if r["root_id"] in roots]
    units = {}
    for r in rows:
        units.setdefault(r["root_id"] if by == "root" else r["qid"].replace(":", "_"), []).append(r)
    return {k: sorted(v, key=lambda r: r["slot"]) for k, v in sorted(units.items())}, n_all


def npz_witnesses(path):
    """-> {witness_id: {array: value}} from a unit NPZ (keys '<witness_id>.<array>')."""
    import numpy as np
    out = {}
    with np.load(path) as z:
        for k in z.files:
            w, a = k.split(".", 1)
            out.setdefault(w, {})[a] = z[k]
    return out


def validate_unit(run, uid, expect_qids, led=None):
    """Full check of one unit's JSON + NPZ (and its ledger line when given).  -> (ok, reason)."""
    import policy3d as Q
    u = Path(run) / "units"
    j, z = u / f"{uid}.json", u / f"{uid}.npz"
    if not j.exists():
        return False, "json_missing"
    try:
        doc = json.loads(j.read_text())
    except (OSError, ValueError) as err:
        return False, f"json_unreadable: {err}"[:120]
    if led is not None and sha_file(j) != led.get("json_sha256"):
        return False, "json_sha_mismatch"
    if doc.get("unit") != uid or [r["qid"] for r in doc.get("records", [])] != list(expect_qids):
        return False, "json_content_mismatch"
    want = set(doc.get("witness_index", {}))
    refs = {w["witness_id"] for r in doc["records"] for w in r.get("witnesses", [])}
    if refs != want:
        return False, "witness_index_mismatch"
    if not want:
        return (not z.exists() and doc.get("npz_sha256") is None), ("ok" if not z.exists() else "orphan_npz")
    if not z.exists():
        return False, "npz_missing"
    zs = sha_file(z)
    if zs != doc.get("npz_sha256") or (led is not None and zs != led.get("npz_sha256")):
        return False, "npz_sha_mismatch"
    try:
        ws = npz_witnesses(z)
    except Exception as err:  # noqa: BLE001  (truncated / corrupt archive)
        return False, f"npz_unreadable: {type(err).__name__}"
    if set(ws) != want or any(Q.wid(a) != w for w, a in ws.items()):
        return False, "npz_witness_mismatch"
    return True, "ok"


def worker(cpu, tasks, results, run, stages, guard, early):
    os.sched_setaffinity(0, {cpu})
    os.nice(10)
    import numpy as np
    import placement3d as P
    import policy3d as Q

    def on_guard(*_):
        raise Q.Interrupted("per-query CPU guard")
    signal.signal(signal.SIGPROF, on_guard)
    while True:
        item = tasks.get()
        if item is None:
            return
        uid, qs, root_sha = item
        t0, c0, out, arrays, index, ikc = time.time(), time.process_time(), [], {}, {}, {}
        rp = P.SRC / "raw/roots" / f"{qs[0]['root_id']}.json"
        if sha_file(rp) != root_sha:                           # source drift between pin and use: refuse the unit
            results.put(("SOURCE_DRIFT", uid, cpu, 0, 0, 0, None, None)); continue
        root = json.loads(rp.read_text())
        scenes = {}
        for q in qs:
            tq, cq = time.time(), time.process_time()
            assert q["root_id"] == root["ids"]["root_id"]
            try:
                sc = scenes.get(q["scene_hash"]) or scenes.setdefault(q["scene_hash"], P.Scene(root["objects"], q["cells"], ikc))
                signal.setitimer(signal.ITIMER_PROF, guard)
                try:
                    rec, ws = Q.evaluate(sc, q["object"], tuple(q["dest"]), stages, early_stop=early)
                finally:
                    signal.setitimer(signal.ITIMER_PROF, 0)
            except Q.Interrupted as err:
                rec, ws = {"label": None, "status": "INTERRUPTED", "reason": str(err)}, []
            for w, a, prov in ws:
                if w not in index:
                    arrays.update({f"{w}.{k}": v for k, v in a.items()})
                    index[w] = {"qid": q["qid"], **prov, "keys": sorted(a)}
            out.append({**{k: q.get(k) for k in ("qid", "slot", "depth", "scene_hash", "object", "dest", "band", "shape", "cheb", "n_obj", "K",
                                                 "reference", "dest_is_t_goal", "clutter", "cohort", "split")}, **rec, "wall_s": time.time() - tq,
                        "cpu_query_s": time.process_time() - cq})
        u = Path(run) / "units"
        npz_sha = None
        if arrays:                                             # 1. NPZ first (atomic)
            tmpz = u / f".{uid}.{os.getpid()}.tmp.npz"
            np.savez_compressed(tmpz, **arrays); os.replace(tmpz, u / f"{uid}.npz")
            npz_sha = sha_file(u / f"{uid}.npz")
        doc = {"unit": uid, "records": out, "protocol": P.protocol()[0], "witness_index": index, "npz_sha256": npz_sha, "root_sha256": root_sha,
               "wall_s": time.time() - t0, "cpu_s": time.process_time() - c0}
        tmp = u / f".{uid}.{os.getpid()}.tmp"                  # 2. JSON (atomic), recording the NPZ hash
        tmp.write_text(json.dumps(doc, sort_keys=True, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
        os.replace(tmp, u / f"{uid}.json")
        results.put(("DONE", uid, cpu, doc["wall_s"], doc["cpu_s"], len(out), sha_file(u / f"{uid}.json"), npz_sha))


def own_tree(me, workers):
    """Coordinator + descendants; -> (processes, unexpected descendants).  Expected: the workers and multiprocessing's resource tracker."""
    import psutil
    kids, bad = me.children(recursive=True), []
    wp = {p.pid for p in workers}
    for k in kids:
        try:
            ok = k.pid in wp or (k.ppid() == me.pid and "resource_tracker" in " ".join(k.cmdline()))
        except psutil.NoSuchProcess:
            continue
        if not ok:
            bad.append(k.pid)
    return [me] + kids, bad


def monitor(run, procs, mask, stop, state):
    """Fail closed: any monitor error, sensor loss or violation stops the run (a run is never left unmonitored)."""
    try:
        _monitor(run, procs, mask, stop, state)
    except Exception as err:  # noqa: BLE001
        state["safety_stop"] = state.get("safety_stop") or [f"monitor_error: {type(err).__name__}: {err}"[:200]]
        stop.set()


def _monitor(run, procs, mask, stop, state):
    import psutil
    me, swap0, log = psutil.Process(), psutil.swap_memory().used, open(Path(run) / "monitor.jsonl", "a")
    paused, tick, cache = False, 0, {}                        # AMENDMENT_V2_001: one persistent psutil.Process per pid, so cpu_percent()
    while not stop.is_set():                                   # measures the interval since the previous tick (a fresh object reads 0.0)
        time.sleep(2.0); tick += 1
        tree, unexpected = own_tree(me, procs)
        cpu, rss, aff = [], [], set()
        for p in tree:
            q = cache.setdefault(p.pid, p)
            try:
                cpu.append(q.cpu_percent(None)); rss.append(q.memory_info().rss / 2 ** 30); aff |= set(os.sched_getaffinity(q.pid))
            except (psutil.NoSuchProcess, ProcessLookupError):
                continue
        try:
            t, ident = sensors()
            sensor_err = None
        except SensorError as err:
            t, ident, sensor_err = None, None, str(err)
        vm, du = psutil.virtual_memory(), psutil.disk_usage(run)
        dead = [p.exitcode for p in procs if p.exitcode not in (None, 0)]
        rec = {"t": time.time(), "tick": tick, "cpu_pct_sum": sum(cpu), "procs": len(tree), "workers_alive": sum(p.is_alive() for p in procs),
               "rss_max_gb": max(rss or [0]), "unexpected_descendants": unexpected, "mem_avail_gb": vm.available / 2 ** 30,
               "swap_growth_gb": (psutil.swap_memory().used - swap0) / 2 ** 30, "disk_free_gb": du.free / 2 ** 30, "pkg_temp_c": t,
               "sensor": ident, "sensor_error": sensor_err, "load1": os.getloadavg()[0], "host_cpu_pct": psutil.cpu_percent(None),
               "done": state["done"], "affinity_union": sorted(aff)}
        bad = [k for k, ok in (("temp_sensor_lost", sensor_err is None),
                               ("cpu_ceiling", sum(cpu) <= 100 * len(mask) + 50),
                               ("mem", rec["mem_avail_gb"] >= LIMITS["mem_avail_gb"]), ("swap", rec["swap_growth_gb"] <= LIMITS["swap_growth_gb"]),
                               ("disk", rec["disk_free_gb"] >= LIMITS["disk_free_gb"]), ("rss", rec["rss_max_gb"] <= LIMITS["worker_rss_gb"]),
                               ("nested_processes", not unexpected), ("temp", t is None or t < LIMITS["pkg_temp_stop_c"]),
                               ("affinity", aff <= set(mask) and len(mask) <= CEILING), ("worker_death", not dead)) if not ok]
        if not bad and t is not None and not paused and t >= LIMITS["pkg_temp_pause_c"]:
            [os.kill(p.pid, signal.SIGSTOP) for p in procs if p.is_alive()]; paused = True
        elif not bad and paused and t is not None and t <= LIMITS["temp_resume_c"]:
            [os.kill(p.pid, signal.SIGCONT) for p in procs if p.is_alive()]; paused = False
        rec.update(paused=paused, violations=bad)
        log.write(json.dumps(rec) + "\n"); log.flush()
        if bad:                                                # never resume on an unreadable sensor: stop (workers are killed below)
            state["safety_stop"] = bad + ([f"sensor: {sensor_err}"] if sensor_err else [])
            stop.set()


def kill_workers(procs):
    for p in procs:
        if p.is_alive():
            try:
                os.kill(p.pid, signal.SIGCONT); os.kill(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    for p in procs:
        p.join(timeout=10)


def quarantine(run, names, why):
    q = Path(run) / "quarantine" / time.strftime("%Y%m%dT%H%M%S")
    q.mkdir(parents=True, exist_ok=True)
    for n in names:
        src = Path(run) / "units" / n
        if src.exists():
            shutil.move(str(src), str(q / n))
    (q / "REASON.json").write_text(json.dumps(why, indent=1, sort_keys=True))
    return str(q)


def resume_audit(run, units, apply=True):
    """Ledger lines whose unit fully re-validates are done; every other unit file (failed validation, orphans, temporaries) is
    quarantined (apply=True) and its unit recomputed."""
    led, done, why = Path(run) / "ledger.jsonl", {}, {}
    for x in (led.read_text().splitlines() if led.exists() else []):
        try:
            d = json.loads(x)
        except ValueError:
            continue                                           # a torn last line (killed mid-write) is ignored
        if d.get("event") == "unit_done" and d["unit"] in units:
            done[d["unit"]] = d
    ok = {}
    for uid, d in done.items():
        good, reason = validate_unit(run, uid, [q["qid"] for q in units[uid]], d)
        if good:
            ok[uid] = d
        else:
            why[uid] = reason
    files = {p.name for p in (Path(run) / "units").iterdir()} if (Path(run) / "units").exists() else set()
    keep = {f"{u}.json" for u in ok} | {f"{u}.npz" for u in ok if ok[u].get("npz_sha256")}
    stray = sorted(files - keep)
    failed = dict(why)
    for n in stray:
        base = n[:-5] if n.endswith(".json") else n[:-4] if n.endswith(".npz") else None
        why[n] = f"failed_validation: {failed[base]}" if base in failed else "orphan_or_temporary"
    qdir = quarantine(run, stray, why) if (stray and apply) else None
    return ok, {"validated": len(ok), "ledger_units_failed": {u: r for u, r in why.items() if u in units}, "quarantined_files": stray,
                "quarantine_dir": qdir}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run"); ap.add_argument("--roster", required=True); ap.add_argument("--policy", default="adaptive_v1")
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--cpus"); ap.add_argument("--subset", default="all")
    ap.add_argument("--roots", default=""); ap.add_argument("--guard-cpu-s", type=float, default=1200.0)
    ap.add_argument("--unit", default="root", choices=("root", "query"))
    ap.add_argument("--source-pin", default=str(HERE.parent / "mock_v2/configs/source_pin.json"))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--production", action="store_true"); ap.add_argument("--authorization"); ap.add_argument("--confirm")
    a = ap.parse_args()
    import policy3d as Q
    import placement3d as P
    stages = (Q.POLICIES | Q.FIXED)[a.policy]
    early = a.policy not in Q.REFERENCE
    splits = ("train",)
    if a.production and a.dry_run:                             # metadata-only look at the all-splits roster: no authorization needed
        splits = ("train", "val", "test")
    elif a.production:                                         # all splits: only with a recorded, hash-matching authorization
        import guard3d
        if SENSOR_OVERRIDE:
            sys.exit("REFUSED: sensor override is test-only")
        okp, why = guard3d.check(a.authorization, a.confirm, roster=a.roster, policy=a.policy, workers=a.workers,
                                 cpus=a.cpus or "0-11", source_pin=a.source_pin)
        if not okp:
            sys.exit("REFUSED (production): " + "; ".join(why))
        splits = ("train", "val", "test")
    cpus = cpus_of(a.cpus) if a.cpus else default_cpus(a.workers)
    allowed = os.sched_getaffinity(0)
    if a.workers > CEILING or len(cpus) != a.workers or len(set(cpus)) != len(cpus) or not set(cpus) <= allowed or len(cpus) > CEILING:
        sys.exit(f"REFUSED: {a.workers} workers on CPUs {cpus} (ceiling {CEILING}, allowed {sorted(allowed)})")
    try:
        t0c, ident = sensors()
    except SensorError as err:
        sys.exit(f"REFUSED: temperature sensor unreadable ({err})")
    if t0c >= LIMITS["temp_resume_c"]:
        sys.exit(f"REFUSED: package temperature {t0c} C >= {LIMITS['temp_resume_c']} C at start")
    pin = json.loads(Path(a.source_pin).read_text())
    units, n_rows = load_units(a.roster, a.subset, set(filter(None, a.roots.split(","))), a.unit, splits)
    roots = sorted({q["root_id"] for v in units.values() for q in v})
    root_sha = {r: sha_file(P.SRC / "raw/roots" / f"{r}.json") for r in roots}
    drift = [r for r in roots if pin["roots"].get(r) != root_sha[r]]
    if drift:
        sys.exit(f"REFUSED: {len(drift)} source roots differ from (or are absent in) the source pin, e.g. {drift[:3]}")
    code = {n: sha_file(HERE / n) for n in TEACHER_CODE}
    man = {"policy": a.policy, "stages": stages, "early_stop": early, "roster_sha256": sha_file(a.roster), "subset": a.subset, "roots": a.roots,
           "unit_mode": a.unit, "splits": list(splits), "units": {k: [q["qid"] for q in v] for k, v in units.items()}, "protocol": P.protocol(),
           "code": code, "guard_cpu_s": a.guard_cpu_s, "worker_policy": {"workers": a.workers, "cpus": cpus},
           "limits": LIMITS, "source_pin": {"file_sha256": sha_file(a.source_pin), "snapshot": pin.get("snapshot"),
                                            "manifest_sha256": pin.get("manifest_sha256"), "roots": root_sha},
           "sensor_override": bool(SENSOR_OVERRIDE), "production": a.production}
    run = Path(a.run)
    if a.dry_run:                                              # metadata only: no worker, no teacher call
        mp_ = run / "run_manifest.json"
        same = (not mp_.exists()) or {k: json.loads(mp_.read_text()).get(k) for k in man} == json.loads(json.dumps(man))
        ok, audit = resume_audit(run, units, apply=False) if (run / "units").exists() else ({}, {"validated": 0})
        res = {"dry_run": True, "units": len(units), "queries": sum(map(len, units.values())), "roots": len(roots), "roster_rows": n_rows,
               "splits": list(splits), "cpus": cpus, "sensor": ident, "temp_c": t0c, "manifest_compatible": same, "resume_audit": audit,
               "planner_calls": sum(P.W.values()), "work_counters": dict(P.W)}
        print(json.dumps(res, default=str))
        sys.exit(0 if same and res["planner_calls"] == 0 else 4)
    os.setpgrp()
    os.sched_setaffinity(0, set(cpus))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    (run / "units").mkdir(parents=True, exist_ok=True)
    mp_ = run / "run_manifest.json"
    if mp_.exists():
        old = json.loads(mp_.read_text())
        if {k: old.get(k) for k in man} != json.loads(json.dumps(man)):
            diff = sorted(k for k in man if old.get(k) != json.loads(json.dumps(man))[k])
            sys.exit(f"REFUSED: RUN_DIR holds a different frozen run (differs in {diff})")
    else:
        mp_.write_text(json.dumps(man, indent=1, sort_keys=True))
    done, audit = resume_audit(run, units)
    todo = [u for u in units if u not in done]
    led = run / "ledger.jsonl"
    log = open(led, "a")
    log.write(json.dumps({"event": "start", "t": time.time(), "pid": os.getpid(), "workers": a.workers, "cpus": cpus, "todo": len(todo),
                          "resumed": len(done), "resume_audit": audit, "sensor": ident}) + "\n"); log.flush()
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    for u in sorted(todo, key=lambda u: -len(units[u])):
        tasks.put((u, units[u], root_sha[units[u][0]["root_id"]]))
    procs = [ctx.Process(target=worker, args=(c, tasks, results, str(run), stages, a.guard_cpu_s, early), daemon=True)
             for c in cpus[:max(1, min(a.workers, len(todo)))]]
    for p in procs:
        tasks.put(None); p.start()
    stop, state = threading.Event(), {"done": len(done)}
    th = threading.Thread(target=monitor, args=(str(run), procs, cpus, stop, state), daemon=True); th.start()
    t0, n = time.time(), 0
    while n < len(todo) and not stop.is_set():
        try:
            kind, uid, cpu, wall, cs, nq, js, zs = results.get(timeout=1.0)
        except Exception:  # noqa: BLE001  (queue.Empty)
            if not any(p.is_alive() for p in procs) and results.empty():
                state["safety_stop"] = ["worker_death"]; break
            continue
        if kind != "DONE":
            state["safety_stop"] = [f"{kind}: {uid}"]; break
        good, reason = validate_unit(run, uid, [q["qid"] for q in units[uid]], {"json_sha256": js, "npz_sha256": zs})
        if not good:
            state["safety_stop"] = [f"unit_validation_failed: {uid}: {reason}"]; break
        n += 1; state["done"] += 1
        log.write(json.dumps({"event": "unit_done", "unit": uid, "cpu": cpu, "wall_s": wall, "cpu_s": cs, "queries": nq, "t": time.time(),
                              "json_sha256": js, "npz_sha256": zs}) + "\n"); log.flush()
    stop.set()
    if state.get("safety_stop"):
        kill_workers(procs)
    else:
        for p in procs:
            p.join(timeout=60)
    end = {"event": "end", "t": time.time(), "wall_s": time.time() - t0, "units_done_now": n, "complete": n == len(todo),
           "safety_stop": state.get("safety_stop")}
    log.write(json.dumps(end) + "\n"); log.flush()
    print(json.dumps(end))
    sys.exit(0 if end["complete"] else 3)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
