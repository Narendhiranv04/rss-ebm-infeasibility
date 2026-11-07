"""Local runner for the 2D placement sidecar (pilot now; full collection only behind the launch guard).

Workers are long-lived processes, one per selected logical CPU (sched_setaffinity), nice 10, one BLAS/OpenMP thread, killed with
the coordinator (PR_SET_PDEATHSIG). Affinity is NOT a quota: no cgroup is reachable here, so hard_cpu_quota = false. The
coordinator samples k10temp Tctl every 0.25 s: >= temp_pause_c freezes every worker (SIGSTOP) and holds dispatch until
<= temp_resume_c; >= temp_cancel_c (one sample), sensor loss, cooling stall, excessive pausing, low memory / disk, swap growth,
worker RSS, process leak, worker death or AC loss cancel the run. Work budgets are counted, so freezing never changes a result; a
query in flight at a cancel is recorded completed=false / label=null and re-runs identically on resume. Roots are packed when all
their queries are final. Control: touch <run>/PAUSE (hold), rm it (resume), touch <run>/DRAIN (finish in-flight, exit), kill -TERM
<coordinator pid> (cancel now).

    python launch_runner.py topology <launch_dir>
    python launch_runner.py preflight <launch_dir> [full]
    python launch_runner.py pilot <launch_dir> <run_name> <pairs> <FULL|FIRST_SUCCESS> <workers|serial> [all|first_per_root|one]
    python launch_runner.py token <launch_dir>                        # the confirmation token of the current frozen files
    python launch_runner.py guard <launch_dir> --execute-full --confirm <token>   # checks only
    python launch_runner.py full <launch_dir> --execute-full --confirm <token>
    python launch_runner.py status <run_dir>
"""
import os
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "1"                                  # before any numerical import
import ctypes, fcntl, gc, gzip, json, multiprocessing as mp, resource, signal, sys, time  # noqa: E402
from collections import Counter, deque  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import psutil  # noqa: E402
import launch_plan as LP  # noqa: E402
import placement_teacher as T  # noqa: E402
P = T.P
SCHEMA = "placement_launch_v1"


def cfg_of(L):
    return json.loads((Path(L) / "configs" / "launch_config.json").read_text())


def topology(workers, reserve):
    """Allowed online logical CPUs -> (worker cpus, reserved cpus, detail): one thread per physical core first, then SMT siblings
    alternating between L3 complexes (CCDs); the remainder is reserved for the host, UI and this coordinator."""
    cores = {}
    for c in sorted(os.sched_getaffinity(0)):
        b = Path(f"/sys/devices/system/cpu/cpu{c}")
        if (b / "online").exists() and (b / "online").read_text().strip() != "1":
            continue
        l3 = int((b / "cache/index3/id").read_text()) if (b / "cache/index3/id").exists() else 0
        cores.setdefault((int((b / "topology/physical_package_id").read_text()), l3, int((b / "topology/core_id").read_text())), []).append(c)
    if len(sum(cores.values(), [])) < workers + reserve:
        raise SystemExit(f"FAIL_CLOSED: {len(sum(cores.values(), []))} allowed online CPUs < {workers} workers + {reserve} reserved")
    keys = sorted(cores)
    l3s = sorted({k[1] for k in keys})
    by = {g: [k for k in keys if k[1] == g] for g in l3s}
    sel = [cores[k][0] for k in keys]                     # one thread per physical core
    for i in range(1, max(len(v) for v in cores.values())):
        for j in range(max(len(v) for v in by.values())):
            sel += [cores[by[g][j]][i] for g in l3s if j < len(by[g]) and i < len(cores[by[g][j]])]   # siblings, CCDs alternating
    sel = sel[:workers]
    return sel, sorted(set(sum(cores.values(), [])) - set(sel)), {"cores": {f"{k}": v for k, v in cores.items()}, "l3_groups": l3s}


def sensors():
    for hw in Path("/sys/class/hwmon").glob("hwmon*"):
        if (hw / "name").read_text().strip() == "k10temp":
            lab = {(f.read_text().strip()): f.with_name(f.name.replace("_label", "_input")) for f in hw.glob("temp*_label")}
            if "Tctl" in lab:
                return lab
    return None


def ac_online():
    for s in Path("/sys/class/power_supply").glob("*"):
        if (s / "type").exists() and (s / "type").read_text().strip() == "Mains":
            return (s / "online").read_text().strip() == "1"
    return None


def read_temp(lab, name="Tctl"):
    try:
        return int(lab[name].read_text()) / 1000
    except (OSError, KeyError, ValueError, TypeError):
        return None


def _worker(wid, cpu, tq, rq, out, policy, emerg, nice):
    ctypes.CDLL("libc.so.6").prctl(1, signal.SIGKILL)    # PR_SET_PDEATHSIG: die with the coordinator
    os.sched_setaffinity(0, {cpu})
    os.nice(nice)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    while (t := tq.get()) is not None:
        rq.put(("start", wid, t["trial_id"]))
        try:
            r = T.run_query(t, out, policy, emerg)
            v, m = r.get("volatile", {}), r.get("mode_counts") or {}
            rq.put(("done", wid, t["trial_id"], r.get("label"), r.get("completed"), r.get("eligible"), v.get("cpu_s", 0.0),
                    m.get("motion_attempted", 0) + m.get("reused_reverified", 0), m.get("valid_pairs", 0)))
        except Exception as e:  # noqa: BLE001
            rq.put(("error", wid, t["trial_id"], f"{type(e).__name__}: {e}"[:400]))


def header(L, p, policy, scope):
    G, h = P.A.env(), json.loads((Path(L) / "plan_hashes.json").read_text())
    return {"schema_version": SCHEMA, "resume_version": T.RESUME_VERSION, "scope": scope, "domain": "planar_2d", "root_id": p["root_id"],
            "split": p["split"], "root_group_id": p["root_group_id"], "source_selected_sha256": h["source_selected_sha256"],
            "split_manifest_sha256": h["split_manifest_sha256"], "query_plan_sha256": h["query_plan_sha256"], "root_file_sha256": p["root_file_sha256"],
            "original_scene_hash": p["original_scene_hash"], "robot_hash": G["protocol"], "teacher_hash": G["config_hash"],
            "pair_protocol_id": T.pair_protocol()[0], "budget_hash": T.budget_hash(policy), "policy": policy, "support_id": "table",
            "held_grasp_policy": T.HELD_POLICY, "privileged_teacher": {"verified_class": p["verified_class"]}}


def plan_rows(L):
    return [json.loads(x) for x in gzip.decompress((Path(L) / "query_plan.jsonl.gz").read_bytes()).decode().splitlines()]


def tasks_for(L, select):
    out = []
    for p in plan_rows(L):
        rows = [r for r in p["queries"] if select(r)]
        if rows:
            out.append((p, [{**dict(zip(LP.COLS, r)), "root": p["root_id"]} for r in rows]))
    return out


def run(L, out, roots, policy, workers, cpus, reserved, scope):
    """The coordinator. roots: [(plan_row, [task, ...])] in plan order. Returns the run summary (also run_<t0>.json)."""
    C, S = cfg_of(L), cfg_of(L)["safety"]
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock = open(out / ".lock", "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)      # one coordinator per run directory
    man = {"scope": scope, "policy": policy, "budget_hash": T.budget_hash(policy), "pair_protocol_id": T.pair_protocol()[0],
           "plan_hashes": json.loads((Path(L) / "plan_hashes.json").read_text()), "config_sha256": P.sha_file(Path(L) / "configs" / "launch_config.json"),
           "tasks_sha256": P.sha(json.dumps([t["trial_id"] for _, ts in roots for t in ts]).encode()), "roots": len(roots), "queries": sum(len(t) for _, t in roots)}
    mf = out / "run_manifest.json"
    if mf.exists() and json.loads(mf.read_text()) != man:
        raise SystemExit(f"REFUSED: {mf} records a different frozen plan / policy / task list")
    mf.write_text(json.dumps(man, indent=1, sort_keys=True))
    for f in [*out.glob("parts/*/*.tmp"), *out.glob("*/*.tmp")]:
        f.unlink()                                        # torn writes (no worker alive yet)
    todo, left = deque(), {}
    for p, ts in roots:
        rid, d = p["root_id"], out / "parts" / p["root_id"]
        if (out / "records" / f"{rid}.json").exists():
            if d.exists():                                # crashed after packing: the parts are redundant
                for f in d.iterdir():
                    f.unlink()
                d.rmdir()
            continue
        keep = {w for f in d.glob("*.json") for w in json.loads(f.read_text())["witness_refs"] if T.final(f)} if d.exists() else set()
        for f in d.glob("w_*.npz") if d.exists() else []:
            if f.stem[2:] not in keep:
                f.unlink()                                # witness of an unfinished query: rewritten identically on re-run
        q = [t for t in ts if not T.final(d / f"{t['trial_id']}.json")]
        left[rid] = (p, ts, len(q))
        todo += q
        if not q:
            T.pack_root(out, header(L, p, policy, scope), ts)
            del left[rid]
    os.sched_setaffinity(0, set(reserved))
    lab, t0, swap0 = sensors(), time.time(), psutil.swap_memory().used
    if lab is None or read_temp(lab) is None:
        raise SystemExit("FAIL_CLOSED: k10temp Tctl not readable")
    gc.freeze()                                           # the task list stays shared copy-on-write: the GC never touches it in workers
    ctx = mp.get_context("fork")
    tq, rq = ctx.Queue(), ctx.Queue()
    procs = [ctx.Process(target=_worker, args=(i, c, tq, rq, str(out), policy, S["query_emergency_cpu_s"], C["execution"]["nice"]), daemon=True)
             for i, c in enumerate(cpus[:workers])]
    for pr in procs:
        pr.start()
    pids = [pr.pid for pr in procs]
    inflight, done, stop, frozen, why = {}, Counter(), None, False, None
    events, hist, log, cnt = [], deque(), open(out / "monitor.jsonl", "a"), Counter()
    hot_since = last_log = last_stat = 0.0
    loss = 0
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    task_ctx = {t["trial_id"]: (p, t) for p, ts in roots for t in ts}
    try:
        while (todo or inflight) and stop is None:
            now, tc = time.time(), read_temp(lab)
            loss = loss + 1 if tc is None else 0
            user = (out / "PAUSE").exists()
            if not frozen and ((tc or 0) >= S["temp_pause_c"] or user):
                [os.kill(x, signal.SIGSTOP) for x in pids]
                frozen, why, hot_since = True, "USER" if user else "THERMAL", now
                events.append(["PAUSE", round(now - t0, 2), tc, why])
            elif frozen and not user and (tc or 99) <= S["temp_resume_c"]:
                [os.kill(x, signal.SIGCONT) for x in pids]
                frozen = False
                events.append(["RESUME", round(now - t0, 2), tc, why])
            hist.append((now, frozen and why == "THERMAL"))
            while hist[0][0] < now - 600:
                hist.popleft()
            if now - last_log >= C["execution"]["log_every_s"]:
                vm, kids = psutil.virtual_memory(), psutil.Process().children(recursive=True)
                rss = [k.memory_info().rss // 2 ** 20 for k in kids if k.is_running()]
                wcpu = sum(sum(k.cpu_times()[:2]) for k in kids if k.is_running())
                s = {"t": round(now - t0, 2), "temp_c": tc, **{k: read_temp(lab, k) for k in ("Tccd1", "Tccd2")}, "frozen": frozen, "inflight": len(inflight),
                     "done": sum(done.values()), "mem_available_mb": vm.available // 2 ** 20, "swap_growth_mb": (psutil.swap_memory().used - swap0) // 2 ** 20,
                     "disk_free_mb": psutil.disk_usage(str(out)).free // 2 ** 20, "load1": os.getloadavg()[0], "descendants": len(kids),
                     "workers_rss_mb": sum(rss), "max_worker_rss_mb": max(rss, default=0), "workers_cpu_s": round(wcpu, 1), "queued": len(todo), "ac": ac_online()}
                log.write(json.dumps(s) + "\n")
                log.flush()
                last_log = now
                stop = next((m for m, bad in (
                    ("MEMORY", s["mem_available_mb"] < S["min_mem_available_mb"]), ("SWAP", s["swap_growth_mb"] > S["max_swap_growth_mb"]),
                    ("DISK", s["disk_free_mb"] < S["min_disk_free_mb"]), ("WORKER_RSS", s["max_worker_rss_mb"] > S["max_worker_rss_mb"]),
                    ("PROCESS_LEAK", len(kids) > workers + S["max_descendants_extra"]), ("WORKER_DIED", not all(pr.is_alive() for pr in procs)),
                    ("AC_POWER_LOST", S["require_ac_power"] and s["ac"] is False)) if bad), None)
            stop = stop or next((m for m, bad in (
                ("TEMPERATURE", (tc or 0) >= S["temp_cancel_c"]), ("SENSOR_LOSS", loss >= S["sensor_loss_samples"]),
                ("COOLING_STALL", frozen and why == "THERMAL" and now - hot_since > S["cooling_stall_s"]),
                ("EXCESSIVE_PAUSING", now - t0 > 600 and sum(f for _, f in hist) / len(hist) > S["max_paused_fraction_10min"])) if bad), None)
            if (out / "DRAIN").exists():
                todo.clear()
            room = min(workers - len(inflight), C["execution"]["dispatch_per_sample"])   # staggered ramp-up
            while stop is None and not frozen and todo and room > 0:
                room -= 1
                t = todo.popleft()
                inflight[t["trial_id"]] = None
                tq.put(t)
            while not rq.empty():
                m = rq.get()
                if m[0] == "start":
                    inflight[m[2]] = m[1]
                elif m[0] == "error":
                    stop = f"ERROR:{m[2]}:{m[3]}"
                else:
                    inflight.pop(m[2], None)
                    p, t = task_ctx[m[2]]
                    done[(p["verified_class"], t["block"].split("_")[0], t["dest_bin"], "null" if m[3] is None else str(m[3]))] += 1
                    cnt["cpu_s"] += m[6]
                    cnt["planned"] += m[7]
                    cnt["valid"] += m[8]
                    rid = p["root_id"]
                    p, ts, n = left[rid]
                    left[rid] = (p, ts, n - 1)
                    if n - 1 == 0:
                        T.pack_root(out, header(L, p, policy, scope), ts)
                        del left[rid]
            if now - last_stat >= C["execution"]["status_every_s"]:
                status(out, man, done, cnt, left, inflight, events, tc, frozen, t0)
                last_stat = now
            time.sleep(C["execution"]["sample_every_s"])
    except KeyboardInterrupt:
        stop = "USER_CANCEL"
    [os.kill(x, signal.SIGCONT) for x in pids if psutil.pid_exists(x)]
    if stop:
        for pr in procs:
            pr.kill()
    else:
        for _ in procs:
            tq.put(None)
    for pr in procs:
        pr.join(30)
    for tid in inflight:                                  # interrupted: no observed label
        p, t = task_ctx[tid]
        part = out / "parts" / p["root_id"] / f"{tid}.json"
        if not T.final(part):
            part.parent.mkdir(parents=True, exist_ok=True)
            T.commit(part, {"schema_version": SCHEMA, "trial_id": tid, "eligible": True, "attempted": True, "completed": False, "label": None,
                            "safety_interruption": stop, "witness_refs": []}, None)
    ru = resource.getrusage(resource.RUSAGE_CHILDREN)
    paused = sum(b[1] - a[1] for a, b in zip(events[::2], events[1::2])) + ((time.time() - t0 - events[-1][1]) if len(events) % 2 else 0)
    res = {"scope": scope, "workers": workers, "worker_cpus": cpus[:workers], "reserved_cpus": reserved, "hard_cpu_quota": False,
           "enforcement": "sched_setaffinity (1 logical CPU per worker) + fixed process count + nice + 1 BLAS/OMP thread; no cgroup quota",
           "sensor": "k10temp Tctl", "safety": S, "queries": man["queries"], "finished_now": sum(done.values()), "interrupted": sorted(inflight),
           "not_started": len(todo), "stop": stop, "wall_s": round(time.time() - t0, 1), "children_cpu_s": round(ru.ru_utime + ru.ru_stime, 1),
           "paused_s": round(paused, 1), "pauses": len(events[::2]), "events": events, "roots_open": sorted(left), "leftover_descendants": len(psutil.Process().children(recursive=True))}
    (out / f"run_{int(t0)}.json").write_text(json.dumps(res, indent=1))
    status(out, man, done, cnt, left, inflight, events, None, False, t0, final_=res)
    return res


def status(out, man, done, cnt, left, inflight, events, tc, frozen, t0, final_=None):
    lab = Counter()
    for (c, b, d, y), n in done.items():
        lab[f"label_{y}"] += n
        lab[f"class_{c}"] += n
        lab[f"block_{b}"] += n
        lab[f"dest_{d}"] += n
    el = time.time() - t0
    rate = sum(done.values()) / el * 3600 if el > 0 else 0
    s = {"updated": time.strftime("%Y-%m-%dT%H:%M:%S"), "pid": os.getpid(), "elapsed_h": round(el / 3600, 3), "queries_total": man["queries"],
         "queries_done_this_session": sum(done.values()), "roots_open": len(left), "inflight": len(inflight), "temp_c": tc, "frozen": frozen,
         "pauses": len(events[::2]), "worker_cpu_h": round(cnt["cpu_s"] / 3600, 3), "queries_per_h": round(rate, 1),
         "survivor_pair_coverage": round(cnt["planned"] / cnt["valid"], 4) if cnt["valid"] else None, "counts": dict(sorted(lab.items())),
         "roots_packed": len(list(out.glob("records/*.json"))), "final": final_ is not None, "stop": (final_ or {}).get("stop")}
    (out / "status.tmp").write_text(json.dumps(s, indent=1))
    os.replace(out / "status.tmp", out / "status.json")


if __name__ == "__main__":
    cmd, L = sys.argv[1], Path(sys.argv[2]).resolve()
    if cmd == "topology":
        C = cfg_of(L)
        print(json.dumps(topology(C["execution"]["workers"], C["execution"]["reserved_logical_cpus"]), indent=1))
    elif cmd == "preflight":
        from launch_guard import preflight
        r = preflight(L, full=len(sys.argv) > 3 and sys.argv[3] == "full")
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["passed"] else 1)
    elif cmd == "pilot":
        name, B, cov, w = sys.argv[3], int(sys.argv[4]), sys.argv[5], sys.argv[6]
        sub = sys.argv[7] if len(sys.argv) > 7 else "all"
        C = cfg_of(L)
        roots = tasks_for(L, lambda r: r[11])
        if sub in ("first_per_root", "one"):
            roots = [(p, [next(t for t in ts if not t["static_ineligible"])]) for p, ts in roots][:1 if sub == "one" else None]
        assert all(p["split"] == "train" for p, _ in roots), "pilot roots must be train"
        cpus, workers = ([C["execution"]["serial_cpu"]], 1) if w == "serial" else (C["execution"]["worker_cpus"], int(w))
        if cov == "ADAPTIVE_FULL":
            rescue = int(C.get("adaptive_rescue_pairs", 32))
            pol = {"pairs_per_query": B, "rescue_pairs": rescue, "coverage": cov,
                   "witness_store_max": rescue, "reuse_archived": False}
        else:
            # V2 pilot is oracle-independent even for the archived-witness baseline.
            pol = {"pairs_per_query": B, "coverage": cov, "witness_store_max": B, "reuse_archived": False}
        print(json.dumps({k: v for k, v in run(L, L / "pilot" / name, roots, pol, workers, cpus, C["execution"]["reserved_cpus"], "PILOT").items()
                          if k not in ("safety", "events")}, indent=1))
    elif cmd in ("full", "guard", "token"):
        import launch_guard as LG
        print(LG.token(L) if cmd == "token" else json.dumps(LG.full(L, sys.argv[3:], execute=cmd == "full"), indent=1, default=str))
    elif cmd == "status":
        print((L / "status.json").read_text())
