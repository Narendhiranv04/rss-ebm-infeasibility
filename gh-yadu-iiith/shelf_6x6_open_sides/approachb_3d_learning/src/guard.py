"""Fail-closed resource guard + independent watchdog for GPU-only training on the shared server (user1@10.4.25.63).

    python src/guard.py check   [--config configs/guard.json]           one probe -> JSON (no GPU allocation, no training)
    python src/guard.py watch   --pid PID --run RUN_DIR                  watchdog: probe every period_s; on a violation that persists
                                                                         `consecutive` probes, write RUN_DIR/STOP_RESOURCE and SIGTERM
                                                                         ONLY our trainer PID (SIGKILL after grace_s); exits with it
    python src/guard.py selftest                                         bounded self-stop test against a dummy child (no GPU)

Priority rule: other people's work wins.  Any foreign GPU compute process, a pending line in sajid's GPU queue, an active yadu 3D generator,
a rise in other-CPU load, low RAM / disk, a hot GPU / CPU or a missing sensor stops OUR training only.  Nothing else is ever signalled."""
from __future__ import annotations

import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DEFAULT = {
    "gpu_index": 0, "cpu": 7,
    "gpu_temp_max_c": 80, "gpu_vram_budget_mb": 8192, "gpu_power_max_w": 250,
    "cpu_pkg_temp_max_c": 85, "mem_available_min_gb": 40, "disk_free_min_gb": 30, "disk_path": str(Path.home()),
    "other_cpus_busy_max_pct": 25.0, "load1_max": 6.0,
    "foreign_gpu_procs_allowed": 0,
    "yield_queue_files": [["/data/sajid/idd_nav/queue/bigbox_g0.q", "/data/sajid/idd_nav/queue/bigbox_g0.started"],
                          ["/data/sajid/idd_nav/queue/bigbox_cpu.q", "/data/sajid/idd_nav/queue/bigbox_cpu.started"]],
    # the generator's own WORK processes (its tmux server keeps 'run_3dt_server.sh' in its command line forever, so shells / tmux are ignored)
    "yield_process_patterns": ["yadu_bd/venv/bin/python", "run_3dt_plus.py", "collect4.py", "collect3d.py", "replay3d.py"],
    "period_s": 10, "consecutive": 2, "grace_s": 60,
}


def load_cfg(path=None):
    cfg = dict(DEFAULT)
    p = Path(path) if path else ROOT / "configs/guard.json"
    if p.exists():
        cfg.update(json.loads(p.read_text()))
    return cfg


def sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.stdout if r.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        return ""


def cpu_times():
    out = {}
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu") and line[3].isdigit():
            p = line.split(); v = list(map(int, p[1:]))
            out[int(p[0][3:])] = (sum(v), v[3] + v[4])
    return out


def cpu_busy(dt=1.0):
    a = cpu_times(); time.sleep(dt); b = cpu_times()
    return {c: 100.0 * (1 - (b[c][1] - a[c][1]) / max(1, b[c][0] - a[c][0])) for c in a}


def coretemp_pkg():
    for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (h / "name").read_text().strip() == "coretemp":
                return int((h / "temp1_input").read_text()) / 1000
        except OSError:
            pass
    return None


def queue_pending(qf, sf):
    try:
        started = set(Path(sf).read_text().splitlines())
        return [l for l in Path(qf).read_text().splitlines() if l.strip() and not l.lstrip().startswith("#") and l not in started]
    except OSError:
        return []                                                           # absent queue files = no queue (not a sensor)


def descendants(pid):
    kids, todo = set(), [pid]
    while todo:
        p = todo.pop()
        for d in Path("/proc").iterdir():
            if d.name.isdigit():
                try:
                    if int((d / "stat").read_text().rsplit(")", 1)[1].split()[1]) == p and int(d.name) not in kids:
                        kids.add(int(d.name)); todo.append(int(d.name))
                except (OSError, ValueError, IndexError):
                    pass
    return kids


def probe(cfg, own_pids=(), sample_cpu=True):
    """-> {facts, violations}.  own_pids: our trainer (and its children) -- excluded from 'foreign' GPU / CPU checks."""
    f, v, own = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, [], set(own_pids)
    g = sh(["nvidia-smi", "-i", str(cfg["gpu_index"]), "--query-gpu=name,uuid,memory.used,memory.total,temperature.gpu,utilization.gpu,power.draw",
            "--format=csv,noheader,nounits"])
    if not g:
        v.append("SENSOR_MISSING: nvidia-smi GPU query failed")
    else:
        name, uuid, mu, mt, t, u, pw = [x.strip() for x in g.strip().split(",")]
        f["gpu"] = {"name": name, "mem_used_mb": float(mu), "mem_total_mb": float(mt), "temp_c": float(t), "util_pct": float(u),
                    "power_w": float(pw) if pw not in ("[N/A]", "N/A") else None}
        if f["gpu"]["temp_c"] > cfg["gpu_temp_max_c"]:
            v.append(f"GPU_TEMP {f['gpu']['temp_c']} > {cfg['gpu_temp_max_c']} C")
        if f["gpu"]["power_w"] is not None and f["gpu"]["power_w"] > cfg["gpu_power_max_w"]:
            v.append(f"GPU_POWER {f['gpu']['power_w']} > {cfg['gpu_power_max_w']} W")
        apps = sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"])
        if apps is None:
            v.append("SENSOR_MISSING: nvidia-smi compute-apps query failed")
        else:
            rows = [[x.strip() for x in l.split(",")] for l in apps.strip().splitlines() if l.strip()]
            mine = [r for r in rows if r[0] == uuid and int(r[1]) in own]
            foreign = [r for r in rows if r[0] == uuid and int(r[1]) not in own]
            f["gpu"]["our_mem_mb"] = sum(float(r[2]) for r in mine)
            f["gpu"]["foreign_procs"] = [{"pid": int(r[1]), "mem_mb": float(r[2]), "cmd": cmdline(r[1])[:80]} for r in foreign]
            if len(foreign) > cfg["foreign_gpu_procs_allowed"]:
                v.append(f"FOREIGN_GPU_PROCESS {[r[1] for r in foreign]}")
            if f["gpu"]["our_mem_mb"] > cfg["gpu_vram_budget_mb"]:
                v.append(f"OUR_VRAM {f['gpu']['our_mem_mb']} > budget {cfg['gpu_vram_budget_mb']} MB")
    t = coretemp_pkg(); f["cpu_pkg_temp_c"] = t
    if t is None:
        v.append("SENSOR_MISSING: coretemp package sensor")
    elif t > cfg["cpu_pkg_temp_max_c"]:
        v.append(f"CPU_TEMP {t} > {cfg['cpu_pkg_temp_max_c']} C")
    mi = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines()}
    f["mem_available_gb"] = round(mi["MemAvailable"] / 1048576, 1)
    if f["mem_available_gb"] < cfg["mem_available_min_gb"]:
        v.append(f"MEM_AVAILABLE {f['mem_available_gb']} < {cfg['mem_available_min_gb']} GB")
    st = os.statvfs(cfg["disk_path"]); f["disk_free_gb"] = round(st.f_bavail * st.f_frsize / 1e9, 1)
    if f["disk_free_gb"] < cfg["disk_free_min_gb"]:
        v.append(f"DISK_FREE {f['disk_free_gb']} < {cfg['disk_free_min_gb']} GB")
    f["load1"] = os.getloadavg()[0]
    if f["load1"] > cfg["load1_max"]:
        v.append(f"LOAD1 {f['load1']:.2f} > {cfg['load1_max']}")
    if sample_cpu:
        busy = cpu_busy(1.0)
        others = [b for c, b in busy.items() if c != cfg["cpu"]]
        f["other_cpus_busy_mean_pct"] = round(sum(others) / max(1, len(others)), 1)
        f["our_cpu_busy_pct"] = round(busy.get(cfg["cpu"], 0.0), 1)
        if f["other_cpus_busy_mean_pct"] > cfg["other_cpus_busy_max_pct"]:
            v.append(f"OTHER_CPUS_BUSY {f['other_cpus_busy_mean_pct']} > {cfg['other_cpus_busy_max_pct']} %")
    pend = {Path(q).name: len(queue_pending(q, s)) for q, s in cfg["yield_queue_files"]}
    f["queue_pending"] = pend
    if any(pend.values()):
        v.append(f"YIELD_QUEUE_PENDING {pend}")
    me = {os.getpid(), os.getppid()} | own
    hits = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) not in me:
            c = cmdline(d.name)
            if any(p in c for p in cfg["yield_process_patterns"]) and "guard.py" not in c and not c.startswith(("tmux", "bash ", "sh ")):
                hits.append({"pid": int(d.name), "cmd": c[:100]})
    f["yield_processes"] = hits
    if hits:
        v.append(f"YIELD_PROCESS_ACTIVE {[h['pid'] for h in hits][:5]}")
    return {"facts": f, "violations": v, "pass": not v}


def is_ours(pid):
    """Only a process of our own uid whose command line is our trainer may ever be signalled."""
    try:
        if os.stat(f"/proc/{pid}").st_uid != os.getuid():
            return False
        argv = Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace").split("\0")
    except OSError:
        return False
    if argv and argv[0] == "approachb_3d_learning GUARD_SELFTEST_DUMMY":
        return True
    # argv[0] must be OUR venv interpreter and argv[1] OUR script (a shell whose command text merely mentions them never qualifies)
    return len(argv) > 1 and argv[0] in (str(ROOT / "venv/bin/python"), "venv/bin/python") and argv[1] in OUR_SCRIPTS


# every project GPU entry point (BWL-0 finding: study.py was missing, so its watchdog could log but not stop it)
OUR_SCRIPTS = tuple(p for s in ("train.py", "xy3d.py", "study.py", "xy3d_tests.py", "figs.py", "figs_bwl.py", "xy3d_data.py", "xy3d_train.py", "r3_manifest3d.py") for p in (str(HERE / s), f"src/{s}"))


def watch(pid, run, cfg):
    run = Path(run); run.mkdir(parents=True, exist_ok=True)
    log, bad = run / "watchdog.jsonl", 0
    while Path(f"/proc/{pid}").exists():
        p = probe(cfg, own_pids={pid} | descendants(pid))
        bad = bad + 1 if p["violations"] else 0
        with open(log, "a") as fh:
            fh.write(json.dumps({**p, "consecutive_bad": bad}) + "\n")
        if bad >= cfg["consecutive"]:
            (run / "STOP_RESOURCE").write_text(json.dumps({"utc": p["facts"]["utc"], "violations": p["violations"]}, indent=1))
            if not is_ours(pid):
                (run / "STOP_REFUSED_NOT_OURS").write_text(str(pid))         # never signal anything we cannot prove is ours; make it visible
            if is_ours(pid):
                os.kill(pid, signal.SIGTERM)
                t0 = time.time()
                while Path(f"/proc/{pid}").exists() and time.time() - t0 < cfg["grace_s"]:
                    time.sleep(1)
                if Path(f"/proc/{pid}").exists() and is_ours(pid):
                    os.kill(pid, signal.SIGKILL)
            return {"stopped": True, "violations": p["violations"]}
        time.sleep(cfg["period_s"])
    return {"stopped": False}


def selftest(cfg):
    """Bounded self-stop: a dummy child, a forced violation (gpu_temp_max_c = -1), the watchdog must stop it within ~3 periods."""
    import tempfile
    run = Path(tempfile.mkdtemp(prefix="guard_selftest_", dir=ROOT / "runs"))
    child = subprocess.Popen(["bash", "-c", f"exec -a 'approachb_3d_learning GUARD_SELFTEST_DUMMY' sleep 120"])
    c = {**cfg, "gpu_temp_max_c": -1, "period_s": 1, "consecutive": 2, "grace_s": 5}
    t0 = time.time()
    res = watch(child.pid, run, c)
    child.wait(timeout=30)
    out = {"stopped": res["stopped"], "seconds": round(time.time() - t0, 1), "child_returncode": child.returncode,
           "stop_file": (run / "STOP_RESOURCE").exists(), "violations": res.get("violations"), "run": str(run)}
    out["pass"] = out["stopped"] and out["stop_file"] and child.returncode is not None and out["seconds"] < 20
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "watch", "selftest"])
    ap.add_argument("--config"); ap.add_argument("--pid", type=int); ap.add_argument("--run")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    if a.cmd == "check":
        print(json.dumps({**probe(cfg), "thresholds": {k: cfg[k] for k in cfg if k not in ("yield_queue_files",)}}, indent=1))
    elif a.cmd == "watch":
        print(json.dumps(watch(a.pid, a.run, cfg)))
    else:
        r = selftest(cfg); print(json.dumps(r, indent=1)); sys.exit(0 if r["pass"] else 1)
