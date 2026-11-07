"""10-minute machine-side status, GitHub telemetry and verified backups for the 3D v4 (Q2_3D_PARITY_V1) collection and the 2D Q2 job.
Stdlib only.  Every write is atomic (temp file + fsync + rename); every run is serialized by flock (in the cron line).

  gvlab (user cron, every 600 s):
    python status4.py snapshot   3D v4 + v3 state -> ~/approachb3d_v3/telemetry/{latest_3d.json, queue/<utc>_3d.json} (retry queue)
    python status4.py backup     every ledger-validated unit of the v4 run -> BACKUP (second physical disk), hash-verified
  laptop host (user cron, every 600 s):
    python3 status4.py state OUT measured service / backup posture (crontabs, telemetry commits, backup paths) for the v4 decision
    python3 status4.py relay     pull the gvlab queue, add the 2D Q2 state, commit + fast-forward push the small status files to
                                 branch naren/3d-dataset-relocations-telemetry (own clone, git lock); remove a queue item only after
                                 its push succeeded; mirror the hash-verified backup off-host and re-verify every unit
A status commit is NOT a dataset backup; the dataset copies are the BACKUP and the laptop mirror (see the v4 runbook for their status).
"""
import fcntl, glob, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

HOME = Path.home()
SIDE = Path(__file__).resolve().parent.parent                         # the sidecar of THIS checkout
RUN4 = SIDE / "production_v4/run_q2_parity"
RUN3 = SIDE / "production_v3/run_first_cohort"
TELE = HOME / "approachb3d_v3/telemetry"                               # gvlab
BACKUP = Path("/ssd_scratch/nav_shelf66v3_backup")                     # gvlab, second physical disk (sda1; the run is on sdb3)
LOCK = HOME / ".cache/shelf66v3_place/collector.lock"
RELAY = HOME / ".cache/shelf66v3_place/relay"                          # laptop
TCLONE = HOME / "Desktop/rss-3d-telemetry"                             # laptop telemetry clone (branch below)
TBRANCH = "naren/3d-dataset-relocations-telemetry"
MIRROR = HOME / "Desktop/shelf66v3_offhost_mirror"                     # laptop off-host copy of the verified units
RUN2D = HOME / "Desktop/rss-placement-ready-v2/gh-yadu-iiith/planar_2d/dataset_relocations_v1/full_2d_v2"
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jl(p):
    out = []
    for x in Path(p).read_text().splitlines() if Path(p).exists() else []:
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def alive(pid, word):
    try:
        return word in Path(f"/proc/{pid}/cmdline").read_bytes().decode(errors="replace")
    except OSError:
        return False


def run_state(run, cache_name):
    """Progress of one collector run directory (no teacher call)."""
    if not (run / "run_manifest.json").exists():
        return {"state": "NOT_STARTED", "run_dir": str(run), "run_dir_exists": run.exists()}
    man = json.loads((run / "run_manifest.json").read_text())
    led = jl(run / "ledger.jsonl")
    done = {e["unit"]: e for e in led if e.get("event") == "unit_done"}
    total_q = sum(len(v) for v in man["units"].values())
    cache_p = TELE / f"label_cache_{cache_name}.json"
    cache = json.loads(cache_p.read_text()) if cache_p.exists() else {}
    for u, e in done.items():
        if cache.get(u, {}).get("json_sha256") != e["json_sha256"] and (run / "units" / f"{u}.json").exists():
            recs = json.loads((run / "units" / f"{u}.json").read_text())["records"]
            cache[u] = {"json_sha256": e["json_sha256"], "q": len(recs), "1": sum(r["label"] == 1 for r in recs),
                        "0": sum(r["label"] == 0 for r in recs), "None": sum(r["label"] is None for r in recs)}
    atomic(cache_p, json.dumps(cache))
    lab = {k: sum(cache[u][k] for u in done if u in cache) for k in ("q", "1", "0", "None")}
    now = time.time()
    recent = [e for e in done.values() if e.get("t", 0) > now - 3600]
    qph = sum(e["queries"] for e in recent)
    try:
        holder = json.loads(LOCK.read_text())
    except (OSError, ValueError):
        holder = {}
    running = bool(holder.get("pid")) and alive(holder["pid"], "collect") and str(run) in str(holder.get("run_dir", ""))
    ends = [e for e in led if e.get("event") == "end"]
    mon = jl(run / "monitor.jsonl")[-1:] or [{}]
    m = mon[0]
    state = "RUNNING" if running else "COMPLETED" if ends and ends[-1].get("complete") and len(done) >= len(man["units"]) else "STOPPED_INCOMPLETE"
    return {"state": state, "run_dir": str(run), "mode": man.get("mode"), "policy_file_sha256": man.get("policy_file_sha256"),
            "roster_sha256": man.get("roster_sha256"), "protocol": man.get("protocol", [None])[0],
            "units_done": len(done), "units_total": len(man["units"]), "queries_done": lab["q"], "queries_total": total_q,
            "labels": {"1_verified": lab["1"], "0_negative": lab["0"], "None_unlabeled": lab["None"]},
            "bytes": sum(p.stat().st_size for p in (run / "units").glob("*") if p.is_file()), "queries_last_hour": qph,
            "eta_hours": round((total_q - lab["q"]) / qph, 2) if qph else None, "collector_pid": holder.get("pid") if running else None,
            "errors": [e.get("safety_stop") for e in ends if e.get("safety_stop")][-3:],
            "monitor_last": {k: m.get(k) for k in ("t", "pkg_temp_c", "cpu_pct_sum", "rss_max_gb", "disk_free_gb", "mem_avail_gb", "violations",
                                                    "affinity_union", "workers_alive")}}


def host():
    t = None
    for h in sorted(glob.glob("/sys/class/hwmon/hwmon*")):
        try:
            if Path(h, "name").read_text().strip() == "coretemp":
                t = int(Path(h, "temp1_input").read_text()) / 1000
        except OSError:
            pass
    mem = {l.split(":")[0]: int(l.split()[1]) for l in Path("/proc/meminfo").read_text().splitlines() if ":" in l}
    return {"load1": os.getloadavg()[0], "pkg_temp_c": t, "disk_free_gb": round(shutil.disk_usage(str(SIDE)).free / 2 ** 30, 1),
            "mem_avail_gb": round(mem.get("MemAvailable", 0) / 2 ** 20, 1)}


def git_head():
    return subprocess.run(["git", "-C", str(SIDE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def cmd_snapshot():
    s = {"utc": utc(), "host": os.uname().nodename, "checkout": str(SIDE), "checkout_head": git_head(),
         "3D_V4_Q2_PARITY": run_state(RUN4, "v4"), "3D_V3_FIRST_COHORT": run_state(RUN3, "v3"), "host_state": host(),
         "backup": json.loads((BACKUP / "run_q2_parity" / "backup_index.json").read_text())["summary"]
         if (BACKUP / "run_q2_parity" / "backup_index.json").exists() else {"status": "no v4 run data to back up yet"}}
    txt = json.dumps(s, indent=1, sort_keys=True)
    atomic(TELE / "latest_3d.json", txt)
    atomic(TELE / "queue" / f"{s['utc'].replace(':', '')}_3d.json", txt)
    print(json.dumps({"utc": s["utc"], "v4": s["3D_V4_Q2_PARITY"]["state"], "v3": s["3D_V3_FIRST_COHORT"]["state"]}))


def cmd_backup():
    dst = BACKUP / "run_q2_parity"
    if not (RUN4 / "run_manifest.json").exists():
        print(json.dumps({"backup": "nothing to back up (v4 run not started)"})); return
    (dst / "units").mkdir(parents=True, exist_ok=True)
    idx_p = dst / "backup_index.json"
    idx = json.loads(idx_p.read_text())["units"] if idx_p.exists() else {}
    done = {e["unit"]: e for e in jl(RUN4 / "ledger.jsonl") if e.get("event") == "unit_done"}
    new = bad = 0
    for u, e in done.items():
        for ext, key in (("json", "json_sha256"), ("npz", "npz_sha256")):
            want = e.get(key)
            if not want or idx.get(u, {}).get(key) == want:
                continue
            src, out = RUN4 / "units" / f"{u}.{ext}", dst / "units" / f"{u}.{ext}"
            if not src.exists() or sha(src) != want:
                bad += 1; continue                                      # never copy a unit that does not match its ledger line
            tmp = out.with_name(f".{out.name}.tmp")
            shutil.copyfile(src, tmp)
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            if sha(tmp) != want:
                tmp.unlink(); bad += 1; continue
            os.replace(tmp, out)
            idx.setdefault(u, {})[key] = want; new += 1
    for f in ("run_manifest.json", "ledger.jsonl", "monitor.jsonl"):
        if (RUN4 / f).exists():
            tmp = dst / f".{f}.tmp"
            shutil.copyfile(RUN4 / f, tmp); os.replace(tmp, dst / f)
    summ = {"utc": utc(), "destination": str(dst), "units_backed_up": len(idx), "units_done_in_ledger": len(done), "copied_now": new,
            "skipped_mismatch": bad, "kind": "local second-disk copy on gvlab (not off-host)"}
    atomic(idx_p, json.dumps({"summary": summ, "units": idx}, sort_keys=True))
    print(json.dumps(summ))


def state_2d():
    procs = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True, text=True).stdout.splitlines()
    hits = [p.strip() for p in procs if any(k in p for k in ("launch_runner.py", "launch_guard.py", "launch_full_2d.sh")) and "grep" not in p]
    led = jl(RUN2D / "ledger.jsonl") if RUN2D.exists() else []
    st = "RUNNING" if hits else "NOT_STARTED" if not RUN2D.exists() else "STOPPED_OR_COMPLETED_SEE_LEDGER"
    return {"utc": utc(), "host": os.uname().nodename, "job": "2D_Q2", "state": st, "run_dir": str(RUN2D), "run_dir_exists": RUN2D.exists(),
            "ledger_lines": len(led), "processes": hits[:5],
            "note": "observed read-only; the 2D job is independent of the 3D v4 job and is never touched by this service"}


def git(*a, check=True):
    return subprocess.run(["git", "-C", str(TCLONE), *a], capture_output=True, text=True, check=check)


def cmd_relay():
    (RELAY / "queue").mkdir(parents=True, exist_ok=True)
    pulled = subprocess.run(["rsync", "-a", "gvlab:approachb3d_v3/telemetry/queue/", f"{RELAY}/queue/"], capture_output=True, text=True)
    s2 = state_2d()
    atomic(RELAY / "queue" / f"{s2['utc'].replace(':', '')}_2d.json", json.dumps(s2, indent=1, sort_keys=True))
    mirror = {}
    if subprocess.run(["ssh", "-o", "BatchMode=yes", "gvlab", f"test -f {BACKUP}/run_q2_parity/backup_index.json"]).returncode == 0:
        MIRROR.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(["rsync", "-a", f"gvlab:{BACKUP}/run_q2_parity/", f"{MIRROR}/run_q2_parity/"], capture_output=True, text=True)
        idx = json.loads((MIRROR / "run_q2_parity/backup_index.json").read_text())["units"]
        bad = [f"{u}.{e}" for u, v in idx.items() for e, k in (("json", "json_sha256"), ("npz", "npz_sha256"))
               if v.get(k) and (not (MIRROR / "run_q2_parity/units" / f"{u}.{e}").exists() or sha(MIRROR / "run_q2_parity/units" / f"{u}.{e}") != v[k])]
        mirror = {"utc": utc(), "rsync_rc": r.returncode, "units": len(idx), "files_failing_verification": bad[:10], "verified": not bad}
        atomic(MIRROR / "run_q2_parity_verification.json", json.dumps(mirror, indent=1))
    items = sorted(RELAY.glob("queue/*.json"))
    lk = open(HOME / ".cache/shelf66v3_place/telemetry.gitlock", "w")
    fcntl.flock(lk, fcntl.LOCK_EX)
    try:
        git("fetch", "-q", "origin", TBRANCH, check=False)
        git("merge", "-q", "--ff-only", f"origin/{TBRANCH}", check=False)
        hist = TCLONE / "status/history"
        hist.mkdir(parents=True, exist_ok=True)
        latest = {}
        for p in items:
            d = json.loads(p.read_text())
            kind = "2d" if p.name.endswith("_2d.json") else "3d"
            h = hist / f"{d['utc'][:10]}_{kind}.jsonl"
            if (d["utc"], d.get("host")) not in {(x.get("utc"), x.get("host")) for x in jl(h)}:   # idempotent across retried pushes
                with open(h, "a") as f:
                    f.write(json.dumps(d, sort_keys=True) + "\n")
            latest[kind] = d
        for kind, d in latest.items():
            atomic(TCLONE / f"status/{'3d_v4' if kind == '3d' else '2d_q2'}_latest.json", json.dumps(d, indent=1, sort_keys=True))
        if mirror:
            atomic(TCLONE / "status/offhost_mirror_latest.json", json.dumps(mirror, indent=1))
        three = latest.get("3d", {})
        msg = (f"telemetry {utc()}: 3D v4 {three.get('3D_V4_Q2_PARITY', {}).get('state', 'no new gvlab snapshot')}, "
               f"2D Q2 {latest.get('2d', {}).get('state')} ({len(items)} snapshots)") + TRAILER
        git("add", "status")
        if git("diff", "--cached", "--quiet", check=False).returncode != 0:
            git("commit", "-q", "-m", msg)
        push = git("push", "-q", "origin", f"HEAD:refs/heads/{TBRANCH}", check=False)
        if push.returncode == 0:
            remote = [p.name for p in items if p.name.endswith("_3d.json")]
            for p in items:
                p.unlink()
            if remote:
                subprocess.run(["ssh", "-o", "BatchMode=yes", "gvlab", "cd approachb3d_v3/telemetry/queue && rm -f " + " ".join(remote)], check=False)
        print(json.dumps({"utc": utc(), "pulled_rc": pulled.returncode, "snapshots": len(items), "push_rc": push.returncode,
                          "head": git("rev-parse", "HEAD").stdout.strip()[:12], "mirror": mirror or None}))
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN); lk.close()


def cmd_state(out):
    """Laptop: the service / backup posture, measured (crontabs, telemetry commits, backup paths), for the v4 decision."""
    sh = lambda *a: subprocess.run(list(a), capture_output=True, text=True)  # noqa: E731
    gv = sh("ssh", "-o", "BatchMode=yes", "gvlab", "crontab -l 2>/dev/null | grep -c 'status4.py snapshot'; test -w /ssd_scratch && echo W").stdout.split()
    lap = sh("crontab", "-l").stdout.count("status4.py relay")
    log = sh("git", "-C", str(TCLONE), "log", "--format=%H %s", f"origin/{TBRANCH}").stdout.splitlines()
    tele = [x for x in log if x.split(" ", 1)[1].startswith("telemetry ")]
    sched = bool(gv and int(gv[0]) > 0) and lap > 0
    doc = {"utc": utc(), "observed_2d": state_2d(),
           "service": {"implemented": "code/status4.py snapshot | backup | relay", "telemetry_branch": TBRANCH, "telemetry_commits": len(tele),
                       "last_telemetry_commit": tele[0] if tele else None, "gvlab_cron_installed": bool(gv and int(gv[0]) > 0), "laptop_cron_installed": lap > 0,
                       "state": "ACTIVE_EVERY_10_MIN" if sched else "IMPLEMENTED_AND_CYCLE_VERIFIED__SCHEDULER_NOT_INSTALLED",
                       "why_not_scheduled": None if sched else "installing the two cron entries (V4_RUNBOOK.md section 3) was refused by this "
                                                               "session's permission policy as persistence; it needs the user's explicit go-ahead"},
           "backup": {"local_second_disk": {"path": str(BACKUP), "writable": "W" in gv, "kind": "gvlab /ssd_scratch (sda1), run disk is sdb3; shared scratch, not durable"},
                      "laptop_mirror": {"path": str(MIRROR), "kind": "off-host copy pulled and hash-verified by the relay; the laptop is not a confirmed durable store"},
                      "confirmed_durable_offhost_destination": None,
                      "state": "OFFHOST_BACKUP_BLOCKED",
                      "why": "no durable off-host destination has been designated by the user; not invented. Interim: second-disk copy + laptop mirror."}}
    atomic(out, json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps({"service": doc["service"]["state"], "backup": doc["backup"]["state"], "2d": doc["observed_2d"]["state"], "telemetry_commits": len(tele)}))


if __name__ == "__main__":
    if sys.argv[1] == "state":
        cmd_state(sys.argv[2])
    else:
        {"snapshot": cmd_snapshot, "backup": cmd_backup, "relay": cmd_relay}[sys.argv[1]]()
