"""10-minute machine-side status, GitHub telemetry and verified cross-machine backups for the 3D v4 (Q2_3D_PARITY_V1, gvlab) and the
2D Q2 (laptop) placement collections.  Read-only towards both collectors; stdlib only (restore-test uses numpy if present).  Every write is
atomic (temp + fsync + rename); every scheduled run is serialized by flock in its cron line.

  gvlab (user cron, every 600 s):
    snapshot   3D v4 + v3 state -> ~/approachb3d_v3/telemetry/{latest_3d.json, queue/<utc>_3d.json}   (retry queue, kept until pushed)
    backup     every ledger-validated 3D unit -> /ssd_scratch (second physical disk); EVERY cycle re-verifies each destination file's
               existence and SHA-256 (an index entry never excuses a missing / corrupt copy: it is re-copied)
  laptop host (user cron, every 600 s):
    relay      (status only, fast) 2D Q2 summary (from the runner's own status / monitor / packed roots) -> pull the gvlab queue -> health flags
               from the latest transfer results
               (STALE / STOPPED / BACKUP_LAGGING / MIRROR_FAILED / PUSH_FAILED / QUARANTINE) -> commit + fast-forward push of small status
               files to naren/3d-dataset-relocations-telemetry from its own clone under a git lock; queue items are removed only after
               their push succeeded
    transfer   (own cron line + lock) 2D packed roots -> gvlab second disk, rehashed on gvlab every run; gvlab 3D backup -> laptop mirror,
               re-hashed (a failed or partial rsync is never 'verified'); rsync I/O timeouts so a stalled link cannot hang a cycle
  either host:
    restore-test KIND OUT   copy one backed-up unit / packed root into a fresh temporary directory, re-hash and open it
    state OUT               measured scheduler / backup posture
A status commit is NOT a dataset backup.  DURABLE_OFFSITE_NOT_CONFIGURED: the two-host copies are interim protection only.
"""
import fcntl, glob, hashlib, json, os, shutil, subprocess, sys, tempfile, time
from pathlib import Path

HOME = Path.home()
SIDE = Path(os.environ.get("SHELF66V3_SIDE") or Path(__file__).resolve().parent.parent)   # gvlab copy outside the live worktree
RUN4 = SIDE / "production_v4/run_q2_parity"
RUN3 = SIDE / "production_v3/run_first_cohort"
TELE = HOME / "approachb3d_v3/telemetry"                                        # gvlab
BACKUP3 = Path(os.environ.get("SHELF66V3_BACKUP3", "/ssd_scratch/nav_shelf66v3_backup")) / "run_q2_parity"   # gvlab second disk
LOCK = HOME / ".cache/shelf66v3_place/collector.lock"
RELAY = HOME / ".cache/shelf66v3_place/relay"                                   # laptop
TCLONE = HOME / "Desktop/rss-3d-telemetry"
TBRANCH = "naren/3d-dataset-relocations-telemetry"
MIRROR3 = Path(os.environ.get("SHELF66V3_MIRROR3", str(HOME / "Desktop/shelf66v3_offhost_mirror"))) / "run_q2_parity"
RUN2D = Path(os.environ.get("PLACEMENT2D_RUN", str(HOME / "Desktop/rss-placement-ready-v2/gh-yadu-iiith/planar_2d/dataset_relocations_v1/full_2d_v2")))
BACKUP2D = os.environ.get("PLACEMENT2D_BACKUP", "/ssd_scratch/nav_placement2d_backup/full_2d_v2")    # on gvlab
REMOTE = os.environ.get("STATUS4_REMOTE", "gvlab")
REM2D = RUN2D.parent / "q2_spillover/runs/remainder_laptop"                       # laptop 2D remainder (after a verified handoff only)
SPILL_STATE = RUN2D.parent / "q2_spillover/state/QUEUE_TRIGGER_STATUS.json"
HANDOFF2D = RUN2D.parent / "q2_spillover/handoff/HANDOFF_ROOT_OWNER_MANIFEST.json"  # frozen owner map: after it the original run is immutable
G_SPILL_RUN = HOME / "approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2_gvlab/runs/spillover_gvlab"
BACKUP2D_BASE = "/ssd_scratch/nav_placement2d_backup"
MIRROR2D_SPILL = HOME / "Desktop/placement2d_spillover_mirror/spillover_gvlab"
TRAILER = "\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
STALE_S, LAG_UNITS, LAG_S = 1200, 60, 1500


def utc(t=None):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def atomic(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def sha(p):
    """Streamed sha256; the file's pages are dropped from the page cache afterwards (POSIX_FADV_DONTNEED) so the 10-min re-hash of
    backups / mirrors does not inflate the laptop's file cache, which under vm.swappiness=180 + zram pushes anon pages to swap and
    can trip the 2D runner's frozen swap-growth guard (incident 2026-11-07, see q2_spillover/reports)."""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
        try:
            os.posix_fadvise(f.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    return h.hexdigest()


def jl(p):
    out = []
    for x in Path(p).read_text().splitlines() if Path(p).exists() else []:
        try:
            out.append(json.loads(x))
        except ValueError:
            pass
    return out


def jload(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def last_line(p):
    try:
        with open(p, "rb") as f:
            f.seek(0, 2); n = f.tell(); f.seek(max(0, n - 8192))
            return json.loads(f.read().splitlines()[-1])
    except (OSError, ValueError, IndexError):
        return {}


# ---------------------------------------------------------------- 3D (gvlab)
def run_state(run, cache_name):
    if not (run / "run_manifest.json").exists():
        return {"state": "NOT_STARTED", "run_dir": str(run), "run_dir_exists": run.exists()}
    man = json.loads((run / "run_manifest.json").read_text())
    led = jl(run / "ledger.jsonl")
    done = {e["unit"]: e for e in led if e.get("event") == "unit_done"}
    total_q = sum(len(v) for v in man["units"].values())
    cache_p = TELE / f"label_cache_{cache_name}.json"
    cache = jload(cache_p, {})
    for u, e in done.items():
        if cache.get(u, {}).get("json_sha256") != e["json_sha256"] and (run / "units" / f"{u}.json").exists():
            recs = json.loads((run / "units" / f"{u}.json").read_text())["records"]
            cache[u] = {"json_sha256": e["json_sha256"], "q": len(recs), "1": sum(r["label"] == 1 for r in recs), "0": sum(r["label"] == 0 for r in recs),
                        "None": sum(r["label"] is None for r in recs), "w": sum(len(r.get("witnesses") or []) for r in recs)}
    atomic(cache_p, json.dumps(cache))
    lab = {k: sum(cache[u][k] for u in done if u in cache) for k in ("q", "1", "0", "None", "w")}
    now = time.time()
    qph = sum(e["queries"] for e in done.values() if e.get("t", 0) > now - 3600)
    holder = jload(LOCK, {}) or {}
    running = bool(holder.get("pid")) and "collect" in cmdline(holder["pid"]) and str(run) == str(holder.get("run_dir", ""))
    ends = [e for e in led if e.get("event") == "end"]
    starts = [e for e in led if e.get("event") == "start"]
    m = last_line(run / "monitor.jsonl")
    complete = bool(ends) and ends[-1].get("complete") and len(done) >= len(man["units"])
    workers = 0
    if running:
        workers = sum(1 for d in Path("/proc").iterdir() if d.name.isdigit() and "spawn_main" in cmdline(d.name)
                      and Path(f"/proc/{d.name}/stat").exists() and Path(f"/proc/{d.name}/stat").read_text().rsplit(")", 1)[1].split()[1] == str(holder["pid"]))
    return {"state": "RUNNING" if running else "COMPLETED" if complete else "STOPPED_INCOMPLETE", "run_dir": str(run), "mode": man.get("mode"),
            "policy_file_sha256": man.get("policy_file_sha256"), "roster_sha256": man.get("roster_sha256"), "protocol": man.get("protocol", [None])[0],
            "first_start_utc": utc(starts[0]["t"]) if starts else None, "units_done": len(done), "units_total": len(man["units"]),
            "queries_done": lab["q"], "queries_total": total_q, "labels": {"1_verified": lab["1"], "0_negative": lab["0"], "None_unlabeled": lab["None"]},
            "witness_associations": lab["w"], "bytes": sum(p.stat().st_size for p in (run / "units").glob("*") if p.is_file()),
            "queries_last_hour": qph, "eta_hours": round((total_q - lab["q"]) / qph, 2) if qph else None,
            "collector_pid": holder.get("pid") if running else None, "worker_processes": workers,
            "quarantine_dirs": len(list((run / "quarantine").glob("*"))) if (run / "quarantine").exists() else 0,
            "errors": [e.get("safety_stop") for e in ends if e.get("safety_stop")][-3:],
            "monitor_last": {k: m.get(k) for k in ("t", "pkg_temp_c", "cpu_pct_sum", "rss_max_gb", "disk_free_gb", "mem_avail_gb", "violations",
                                                    "affinity_union", "workers_alive", "paused")},
            "monitor_age_s": round(now - m["t"], 1) if m.get("t") else None}


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


def cmd_snapshot():
    s = {"utc": utc(), "t": time.time(), "host": os.uname().nodename, "checkout_head":
         subprocess.run(["git", "-C", str(SIDE), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
         "3D_V4_Q2_PARITY": run_state(RUN4, "v4"), "3D_V3_FIRST_COHORT": run_state(RUN3, "v3"), "host_state": host(),
         "2D_Q2_GVLAB_SPILLOVER": state_2d(G_SPILL_RUN, "spillover_gvlab", "2D_Q2_GVLAB_SPILLOVER") if G_SPILL_RUN.exists() else {"state": "NOT_STARTED"},
         "2D_spillover_backup": jload(Path(BACKUP2D_BASE) / "spillover_gvlab/backup_summary.json") or {"state": "NOTHING_TO_BACK_UP"},
         "backup": (jload(BACKUP3 / "backup_index.json", {}) or {}).get("summary") or {"status": "no v4 backup yet"}}
    txt = json.dumps(s, indent=1, sort_keys=True)
    atomic(TELE / "latest_3d.json", txt)
    atomic(TELE / "queue" / f"{s['utc'].replace(':', '')}_3d.json", txt)
    print(json.dumps({"utc": s["utc"], "v4": s["3D_V4_Q2_PARITY"]["state"], "units": s["3D_V4_Q2_PARITY"].get("units_done")}))


def backup_units(run, dst):
    """Copy every ledger-validated unit (JSON + NPZ) to dst; re-verify EVERY destination file each call (existence + SHA-256).  -> summary"""
    (dst / "units").mkdir(parents=True, exist_ok=True)
    done = {e["unit"]: e for e in jl(run / "ledger.jsonl") if e.get("event") == "unit_done"}
    idx, copied, recopied, skipped = {}, 0, 0, []
    for u, e in done.items():
        for ext, key in (("json", "json_sha256"), ("npz", "npz_sha256")):
            want = e.get(key)
            if not want:
                continue
            out = dst / "units" / f"{u}.{ext}"
            if out.exists() and sha(out) == want:                 # verified copy present (checked every cycle, never trusted from the index)
                idx.setdefault(u, {})[key] = want; continue
            src = run / "units" / f"{u}.{ext}"
            if not src.exists() or sha(src) != want:
                skipped.append(f"{u}.{ext}"); continue            # source does not match its ledger line: never back it up
            tmp = out.with_name(f".{out.name}.tmp")
            shutil.copyfile(src, tmp)
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            if sha(tmp) != want:
                tmp.unlink(); skipped.append(f"{u}.{ext}"); continue
            recopied += out.exists(); copied += 1
            os.replace(tmp, out)
            idx.setdefault(u, {})[key] = want
    for f in ("run_manifest.json", "ledger.jsonl", "monitor.jsonl"):
        if (run / f).exists():
            tmp = dst / f".{f}.tmp"
            shutil.copyfile(run / f, tmp); os.replace(tmp, dst / f)
    complete = [u for u in done if all(idx.get(u, {}).get(k) == done[u].get(k) for k in ("json_sha256", "npz_sha256") if done[u].get(k))]
    summ = {"utc": utc(), "t": time.time(), "destination": str(dst), "units_done_in_ledger": len(done), "units_verified_in_backup": len(complete),
            "copied_now": copied, "recopied_missing_or_corrupt": recopied, "skipped_source_mismatch": skipped[:10],
            "kind": "gvlab second physical disk (/ssd_scratch, sda1; the run is on sdb3); shared scratch, interim"}
    atomic(dst / "backup_index.json", json.dumps({"summary": summ, "units": idx}, sort_keys=True))
    return summ


def backup_packed_local(run, dst):
    """gvlab: packed 2D spillover roots -> second disk; every destination file re-hashed against the source each cycle."""
    roots = packed_roots(run)
    if not roots:
        return {"state": "NOTHING_TO_BACK_UP", "utc": utc(), "t": time.time()}
    dst.mkdir(parents=True, exist_ok=True)
    man = jload(dst / "manifest.json", {}) or {}
    bad = 0
    for rid, files in roots.items():
        for p in files:
            rel = str(p.relative_to(run))
            man.setdefault(rel, sha(p))
            out = dst / rel
            if out.exists() and sha(out) == man[rel]:
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(f".{out.name}.tmp")
            shutil.copyfile(p, tmp)
            with open(tmp, "rb") as f:
                os.fsync(f.fileno())
            if sha(tmp) != man[rel]:
                tmp.unlink(); bad += 1; continue
            os.replace(tmp, out)
    atomic(dst / "manifest.json", json.dumps(man, sort_keys=True))
    ok = [r for r, fs in roots.items() if all((dst / str(p.relative_to(run))).exists() for p in fs)]
    summ = {"utc": utc(), "t": time.time(), "destination": str(dst), "roots_packed": len(roots), "roots_verified": len(ok), "files_failing": bad,
            "state": "VERIFIED" if not bad and len(ok) == len(roots) else "BACKUP_FAILED_OR_PARTIAL"}
    atomic(dst / "backup_summary.json", json.dumps(summ))
    return summ


def mirror_2d_spillover():
    """gvlab second-disk copy of the 2D spillover -> laptop mirror; verified only after a successful rsync and a full re-hash."""
    t = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", REMOTE, f"test -f {BACKUP2D_BASE}/spillover_gvlab/manifest.json"],
                       capture_output=True).returncode
    if t == 255:
        return {"state": "MIRROR_FAILED", "utc": utc(), "reason": "ssh / network failure"}
    if t != 0:
        return {"state": "NOTHING_TO_MIRROR", "utc": utc(), "t": time.time()}
    MIRROR2D_SPILL.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["rsync", "-a", "--timeout=300", f"{REMOTE}:{BACKUP2D_BASE}/spillover_gvlab/", f"{MIRROR2D_SPILL}/"], capture_output=True, text=True)
    man = jload(MIRROR2D_SPILL / "manifest.json", {}) or {}
    bad = [f for f, h in man.items() if not (MIRROR2D_SPILL / f).exists() or sha(MIRROR2D_SPILL / f) != h]
    summ = {"utc": utc(), "t": time.time(), "path": str(MIRROR2D_SPILL), "rsync_rc": r.returncode, "files": len(man), "files_failing": len(bad),
            "state": "VERIFIED" if r.returncode == 0 and not bad else "MIRROR_FAILED"}
    atomic(MIRROR2D_SPILL.parent / "spillover_gvlab_verification.json", json.dumps(summ, indent=1))
    return summ


def cmd_backup():
    if (G_SPILL_RUN / "run_manifest.json").exists():
        print(json.dumps({"2d_spillover_backup": backup_packed_local(G_SPILL_RUN, Path(BACKUP2D_BASE) / "spillover_gvlab")}))
    if not (RUN4 / "run_manifest.json").exists():
        print(json.dumps({"backup": "v4 run not started"})); return
    print(json.dumps(backup_units(RUN4, BACKUP3)))


# ---------------------------------------------------------------- 2D (laptop, read-only)
def packed_roots(run):
    """Completed 2D roots: records/<root>.json exists (pack_root writes witnesses, modes, then the record JSON last, each atomically)."""
    out = {}
    for rec in sorted((run / "records").glob("*.json")) if (run / "records").exists() else []:
        rid = rec.stem
        out[rid] = [p for p in (rec, run / "modes" / f"{rid}.npz", run / "witnesses" / f"{rid}.npz") if p.exists()]
    return out


def state_2d(run=None, tag="full_2d_v2", job="2D_Q2_LAPTOP_ORIGINAL"):
    run = Path(run) if run else RUN2D
    procs = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            c = cmdline(d.name)
            if ("launch_runner.py" in c and (" full " in c or " guard " in c) and tag == "full_2d_v2") or \
                    ("spill2d.py laptop-run" in c and tag == "remainder_laptop") or ("spill2d.py server-run --scope SPILLOVER" in c and tag == "spillover_gvlab"):
                # decide "coordinator" on the FULL command line: the scoped runs' long paths push "laptop-run" / "server-run" past any display cut
                procs.append({"pid": int(d.name), "cmd": c[:160], "coordinator": " full " in c or "spill2d.py laptop-run" in c or "spill2d.py server-run" in c})
    s = {"utc": utc(), "t": time.time(), "host": os.uname().nodename, "job": job, "run_dir": str(run), "run_dir_exists": run.exists(),
         "coordinator_processes": procs, "cpu_boost": (Path("/sys/devices/system/cpu/cpufreq/boost").read_text().strip()
                                                      if Path("/sys/devices/system/cpu/cpufreq/boost").exists() else None)}
    try:
        s["ac_online"] = Path("/sys/class/power_supply/ACAD/online").read_text().strip() == "1"
    except OSError:
        s["ac_online"] = None
    if not (run / "run_manifest.json").exists():
        s["state"] = "RUNNING_STARTUP" if procs else "NOT_STARTED"
        return s
    man = json.loads((run / "run_manifest.json").read_text())
    st = jload(run / "status.json", {}) or {}
    m = last_line(run / "monitor.jsonl")
    roots = packed_roots(run)
    cache_p = (RELAY if RELAY.parent.exists() else TELE) / ("label_cache_2d.json" if tag == "full_2d_v2" else f"label_cache_2d_{tag}.json")
    cache = jload(cache_p, {})
    for rid, files in roots.items():
        if rid not in cache:
            r = json.loads(files[0].read_text())
            qs = r.get("queries", [])
            cache[rid] = {"q": len(qs), "teacher": sum(1 for q in qs if q.get("eligible")), "1": sum(q.get("label") == 1 for q in qs),
                          "0": sum(q.get("label") == 0 for q in qs), "null": sum(q.get("eligible") and q.get("label") is None for q in qs),
                          "w": sum(len(q.get("witness_refs") or []) for q in qs)}
    atomic(cache_p, json.dumps(cache))
    agg = {k: sum(cache[r][k] for r in roots) for k in ("q", "teacher", "1", "0", "null", "w")}
    stopped = st.get("final") and st.get("stop")
    coord = [p for p in procs if p["coordinator"]]
    st_age = time.time() - Path(run / "status.json").stat().st_mtime if (run / "status.json").exists() else None
    handed = tag == "full_2d_v2" and st.get("final") and (run / "DRAIN").exists() and HANDOFF2D.exists()   # owner map frozen (any stop was ACKed)
    s.update(state="RUNNING" if coord else ("COMPLETED" if st.get("final") and not st.get("stop") and len(roots) >= man["roots"] else
                                              "HANDED_OFF" if handed else
                                              "DRAINED_FOR_HANDOFF" if st.get("final") and not st.get("stop") and (run / "DRAIN").exists() else
                                              "STOPPED_INCOMPLETE"),
             manifest={k: man.get(k) for k in ("scope", "budget_hash", "pair_protocol_id", "config_sha256", "tasks_sha256", "roots", "queries")},
             quota_queries_total=man.get("queries"), roots_total=man.get("roots"), roots_packed=len(roots),
             queries_packed=agg["q"], teacher_queries_packed=agg["teacher"],
             labels_packed={"1": agg["1"], "0": agg["0"], "null_unlabeled": agg["null"]}, witness_refs_packed=agg["w"],
             runner_status={k: st.get(k) for k in ("updated", "pid", "elapsed_h", "queries_done_this_session", "inflight", "roots_open", "temp_c",
                                                   "frozen", "pauses", "queries_per_h", "final", "stop", "worker_cpu_h")},
             status_age_s=round(st_age, 1) if st_age is not None else None,
             eta_hours=round((man["queries"] - agg["q"]) / st["queries_per_h"], 1) if st.get("queries_per_h") else None,
             monitor_last={k: m.get(k) for k in ("t", "temp_c", "Tccd1", "Tccd2", "frozen", "inflight", "done", "mem_available_mb", "swap_growth_mb",
                                                 "disk_free_mb", "descendants", "workers_rss_mb", "max_worker_rss_mb", "ac")},
             stop_reason=stopped or None, drain_or_pause={"PAUSE": (run / "PAUSE").exists(), "DRAIN": (run / "DRAIN").exists()},
             bytes=sum(p.stat().st_size for fs in roots.values() for p in fs))
    return s


def backup_2d(run=None, tag="full_2d_v2"):
    """Packed 2D roots -> gvlab second disk; source manifest, rsync, then a REMOTE sha256 of every backed-up file each cycle."""
    RUN2D_, BACKUP2D_ = (Path(run) if run else RUN2D), (f"{BACKUP2D_BASE}/{tag}" if run else BACKUP2D)
    roots = packed_roots(RUN2D_)
    subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, f"mkdir -p {BACKUP2D_}"], capture_output=True)
    if not roots:
        return {"state": "NOTHING_TO_BACK_UP", "utc": utc()}
    sfx = "" if tag == "full_2d_v2" else f"_{tag}"                   # original run keeps its pre-spillover cache/summary names
    man_p = RELAY / f"backup2d_source_manifest{sfx}.json"
    man = jload(man_p, {})
    for rid, files in roots.items():
        for p in files:
            rel = str(p.relative_to(RUN2D_))
            if rel not in man:
                man[rel] = sha(p)                                 # packed files are immutable once written
    atomic(man_p, json.dumps(man, sort_keys=True))
    lst = RELAY / f"backup2d_files{sfx}.txt"
    atomic(lst, "\n".join(sorted(man)) + "\n")
    r = subprocess.run(["rsync", "-a", "--timeout=300", "--files-from", str(lst), f"{RUN2D_}/", f"{REMOTE}:{BACKUP2D_}/"], capture_output=True, text=True)
    meta = [f for f in ("run_manifest.json", "status.json") if (RUN2D_ / f).exists()]
    subprocess.run(["rsync", "-a", *[str(RUN2D_ / f) for f in meta], f"{REMOTE}:{BACKUP2D_}/meta/"], capture_output=True, text=True)
    v = subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, f"cd {BACKUP2D_} && xargs -d '\\n' nice -n 19 ionice -c3 sha256sum 2>/dev/null"],
                       input="\n".join(sorted(man)) + "\n", capture_output=True, text=True)
    remote = {ln[66:].strip(): ln[:64] for ln in v.stdout.splitlines()}
    bad = [f for f, h in man.items() if remote.get(f) != h]
    if bad:                                                       # corrupt or missing on gvlab: delete so the next cycle re-copies
        subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, f"cd {BACKUP2D_} && rm -f " + " ".join(f"'{b}'" for b in bad[:500])], capture_output=True)
    ok_roots = [rid for rid, files in roots.items() if all(remote.get(str(p.relative_to(RUN2D_))) == man[str(p.relative_to(RUN2D_))] for p in files)]
    summ = {"utc": utc(), "t": time.time(), "destination": f"{REMOTE}:{BACKUP2D_}", "rsync_rc": r.returncode, "verify_rc": v.returncode,
            "roots_packed": len(roots), "roots_verified_on_destination": len(ok_roots), "files_failing": len(bad), "failing_examples": bad[:5],
            "state": "VERIFIED" if r.returncode == 0 and v.returncode == 0 and not bad else "BACKUP_FAILED_OR_PARTIAL"}
    atomic(RELAY / f"backup2d_last{sfx}.json", json.dumps(summ, indent=1))
    return summ


def mirror_3d():
    """gvlab backup -> laptop mirror; verified ONLY if rsync succeeded and every file re-hashes to the gvlab backup index."""
    t = subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", REMOTE, f"test -f {BACKUP3}/backup_index.json"], capture_output=True).returncode
    if t == 255:
        return {"state": "MIRROR_FAILED", "utc": utc(), "reason": "ssh / network failure"}
    if t != 0:
        return {"state": "NOTHING_TO_MIRROR", "utc": utc()}
    MIRROR3.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["rsync", "-a", "--timeout=300", f"{REMOTE}:{BACKUP3}/", f"{MIRROR3}/"], capture_output=True, text=True)
    idx = (jload(MIRROR3 / "backup_index.json", {}) or {}).get("units", {})
    bad = [f"{u}.{e}" for u, v in idx.items() for e, k in (("json", "json_sha256"), ("npz", "npz_sha256"))
           if v.get(k) and (not (MIRROR3 / "units" / f"{u}.{e}").exists() or sha(MIRROR3 / "units" / f"{u}.{e}") != v[k])]
    summ = {"utc": utc(), "t": time.time(), "path": str(MIRROR3), "rsync_rc": r.returncode, "units_in_index": len(idx), "files_failing": len(bad),
            "failing_examples": bad[:5], "state": "VERIFIED" if r.returncode == 0 and not bad else "MIRROR_FAILED"}
    atomic(MIRROR3.parent / "run_q2_parity_verification.json", json.dumps(summ, indent=1))
    return summ


def health(s3, s2, b2, m3, relay_state):
    flags, now = [], time.time()
    if not s3:
        flags.append("STALE_3D: no gvlab snapshot received this cycle")
    else:
        v = s3.get("3D_V4_Q2_PARITY", {})
        if now - s3.get("t", 0) > STALE_S:
            flags.append(f"STALE_3D: last gvlab snapshot {round((now - s3.get('t', 0)) / 60)} min old")
        if v.get("state") == "STOPPED_INCOMPLETE":
            flags.append(f"STOPPED_3D: {v.get('errors')}")
        if v.get("state") == "RUNNING" and (v.get("monitor_age_s") or 0) > 120:
            flags.append("STALE_3D_MONITOR")
        if v.get("quarantine_dirs"):
            flags.append(f"QUARANTINE_3D: {v['quarantine_dirs']} resume quarantine dir(s)")
        b = s3.get("backup") or {}
        if v.get("state") in ("RUNNING", "COMPLETED", "STOPPED_INCOMPLETE") and (
                not b.get("t") or now - b["t"] > LAG_S or (v.get("units_done", 0) - b.get("units_verified_in_backup", 0)) > LAG_UNITS):
            flags.append("BACKUP_LAGGING_3D")
    if m3.get("state") == "MIRROR_FAILED":
        flags.append("MIRROR_FAILED_3D")
    if m3.get("state") == "VERIFIED" and s3 and (s3.get("backup") or {}).get("t") and now - m3.get("t", 0) > LAG_S:
        flags.append(f"MIRROR_LAGGING_3D: last verified mirror {round((now - m3.get('t', 0)) / 60)} min old")
    if s2.get("state") == "STOPPED_INCOMPLETE":
        flags.append(f"STOPPED_2D: {s2.get('stop_reason')}")
    if s2.get("state") == "RUNNING" and (s2.get("status_age_s") or 0) > STALE_S:
        flags.append("STALE_2D: runner status.json not updated for > 20 min")
    if s2.get("state") in ("RUNNING", "COMPLETED", "STOPPED_INCOMPLETE", "DRAINED_FOR_HANDOFF", "HANDED_OFF") and (
            b2.get("state") != "VERIFIED" or s2.get("roots_packed", 0) - b2.get("roots_verified_on_destination", 0) > LAG_UNITS
            or now - b2.get("t", 0) > LAG_S):
        flags.append(f"BACKUP_LAGGING_2D: {b2.get('state')}, {round((now - b2.get('t', 0)) / 60)} min since the last 2D backup run")
    if relay_state.get("consecutive_push_failures"):
        flags.append(f"PUSH_FAILED: {relay_state['consecutive_push_failures']} previous cycle(s)")
    return {"overall": "HEALTHY" if not flags else "ATTENTION", "flags": flags,
            "durable_offsite": "DURABLE_OFFSITE_NOT_CONFIGURED (two-host interim copies only)"}


def git(*a, check=True):
    return subprocess.run(["git", "-C", str(TCLONE), *a], capture_output=True, text=True, check=check)


def cmd_relay():
    (RELAY / "queue").mkdir(parents=True, exist_ok=True)
    rs = jload(RELAY / "relay_state.json", {}) or {}
    s2 = state_2d()
    b2 = jload(RELAY / "backup2d_last.json", {"state": "NO_TRANSFER_RUN_YET"}) or {}
    m3 = jload(MIRROR3.parent / "run_q2_parity_verification.json", {"state": "NO_TRANSFER_RUN_YET"}) or {}
    pulled = subprocess.run(["rsync", "-a", "--timeout=120", f"{REMOTE}:approachb3d_v3/telemetry/queue/", f"{RELAY}/queue/"], capture_output=True, text=True)
    q3 = sorted(RELAY.glob("queue/*_3d.json"))
    s3 = jload(q3[-1]) if q3 else None
    h = health(s3, s2, b2, m3, rs)
    sr = state_2d(REM2D, "remainder_laptop", "2D_Q2_LAPTOP_REMAINDER") if REM2D.exists() else {"state": "NOT_STARTED"}
    q = jload(SPILL_STATE, {"state": "NOT_INSTALLED"}) or {}
    sp = (s3 or {}).get("2D_Q2_GVLAB_SPILLOVER", {"state": "UNKNOWN"})
    uniq = (s2.get("roots_packed") or 0) + (sr.get("roots_packed") or 0) + (sp.get("roots_packed") or 0)
    b2r = jload(RELAY / "backup2d_last_remainder_laptop.json", {"state": "NO_TRANSFER_RUN_YET"}) or {}
    b2s = (s3 or {}).get("2D_spillover_backup") or {}
    m2s = jload(MIRROR2D_SPILL.parent / "spillover_gvlab_verification.json", {"state": "NO_TRANSFER_RUN_YET"}) or {}
    now = time.time()
    for tag, st_, bk in (("REMAINDER", sr, b2r), ("SPILLOVER", sp, b2s)):
        f = []
        if st_.get("state") == "STOPPED_INCOMPLETE":
            f.append(f"STOPPED_2D_{tag}: {st_.get('stop_reason')}")
        if st_.get("state") == "RUNNING" and (st_.get("status_age_s") or 0) > STALE_S:
            f.append(f"STALE_2D_{tag}")
        if st_.get("state") in ("RUNNING", "COMPLETED", "STOPPED_INCOMPLETE") and (st_.get("roots_packed") or 0) and (
                bk.get("state") != "VERIFIED" or now - bk.get("t", 0) > LAG_S
                or (st_.get("roots_packed") or 0) - (bk.get("roots_verified_on_destination", bk.get("roots_verified")) or 0) > LAG_UNITS):
            f.append(f"BACKUP_LAGGING_2D_{tag}: {bk.get('state')}")
        if f:
            h["flags"] += f; h["overall"] = "ATTENTION"
    if m2s.get("state") == "MIRROR_FAILED":
        h["flags"].append("MIRROR_FAILED_2D_SPILLOVER"); h["overall"] = "ATTENTION"
    if str(q.get("state", "")).startswith("BLOCKED"):
        h["flags"].append(f"SPILLOVER_QUEUE_{q['state']}"); h["overall"] = "ATTENTION"
    beat = {"utc": utc(), "t": time.time(), "host": os.uname().nodename, "health": h, "2D_Q2": s2, "2D_backup_to_gvlab": b2,
            "2D_Q2_LAPTOP_REMAINDER": sr, "2D_Q2_GVLAB_SPILLOVER": sp, "2D_remainder_backup_to_gvlab": b2r, "2D_spillover_backup_on_gvlab": b2s,
            "2D_spillover_mirror_to_laptop": m2s,
            "2D_Q2_SPILLOVER_QUEUE": {k: q.get(k) for k in ("state", "last_tick_utc", "last_error", "block_reason", "progress", "server_share", "attention")},
            "2D_Q2_COMBINED": {"unique_roots_packed": uniq, "roots_total": 4620, "note": "disjoint by the frozen owner map; before a handoff only the original run counts"},
            "3D_mirror_to_laptop": m3, "3D_latest_gvlab_snapshot_utc": (s3 or {}).get("utc"), "gvlab_queue_pull_rc": pulled.returncode}
    atomic(RELAY / "queue" / f"{beat['utc'].replace(':', '')}_heartbeat.json", json.dumps(beat, indent=1, sort_keys=True))
    items = sorted(RELAY.glob("queue/*.json"))
    lk = open(HOME / ".cache/shelf66v3_place/telemetry.gitlock", "w")
    fcntl.flock(lk, fcntl.LOCK_EX)
    push_rc = -1
    try:
        git("fetch", "-q", "origin", check=False)
        git("merge", "-q", "--ff-only", f"origin/{TBRANCH}", check=False)
        hist = TCLONE / "status/history"
        hist.mkdir(parents=True, exist_ok=True)
        latest = {}
        for p in items:
            d = jload(p)
            if d is None:
                continue
            kind = p.name.rsplit("_", 1)[1].split(".")[0]
            hp = hist / f"{d['utc'][:10]}_{kind}.jsonl"
            if (d["utc"], d.get("host")) not in {(x.get("utc"), x.get("host")) for x in jl(hp)}:
                with open(hp, "a") as f:
                    f.write(json.dumps(d, sort_keys=True) + "\n")
            latest[kind] = d
        if "3d" in latest:
            atomic(TCLONE / "status/3d_v4_latest.json", json.dumps(latest["3d"], indent=1, sort_keys=True))
        atomic(TCLONE / "status/2d_q2_latest.json", json.dumps(s2, indent=1, sort_keys=True))
        atomic(TCLONE / "status/heartbeat_latest.json", json.dumps(beat, indent=1, sort_keys=True))
        if SPILL_STATE.exists():
            atomic(TCLONE / "status/2d_q2_spillover_queue.json", json.dumps(q, indent=1, sort_keys=True))
        v3 = (latest.get("3d") or s3 or {}).get("3D_V4_Q2_PARITY", {})
        msg = (f"telemetry {beat['utc']}: {h['overall']} | 3D v4 {v3.get('state')} {v3.get('queries_done')}/{v3.get('queries_total')} | "
               f"2D Q2 {s2.get('state')} {s2.get('queries_packed')}/{s2.get('quota_queries_total')} | spill queue {q.get('state')}") + TRAILER
        git("add", "status")
        if git("diff", "--cached", "--quiet", check=False).returncode != 0:
            git("commit", "-q", "-m", msg)
        push_rc = git("push", "-q", "origin", f"HEAD:refs/heads/{TBRANCH}", check=False).returncode
        if push_rc == 0:
            remote = [p.name for p in items if p.name.endswith("_3d.json")]
            for p in items:
                p.unlink()
            if remote:
                subprocess.run(["ssh", "-o", "BatchMode=yes", REMOTE, "cd approachb3d_v3/telemetry/queue && rm -f " + " ".join(remote)], capture_output=True)
            rs = {"last_push_ok_utc": beat["utc"], "consecutive_push_failures": 0}
        else:
            rs = {**rs, "consecutive_push_failures": rs.get("consecutive_push_failures", 0) + 1, "last_push_failed_utc": beat["utc"]}
        atomic(RELAY / "relay_state.json", json.dumps(rs))
    finally:
        fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
    print(json.dumps({"utc": beat["utc"], "health": h["overall"], "flags": h["flags"], "push_rc": push_rc, "queued": len(items),
                      "2d": s2.get("state"), "3d": v3.get("state"), "b2": b2.get("state"), "m3": m3.get("state")}))


def cmd_transfer():
    """Bulk copies, separate from the status push (own cron line and lock): 2D packed roots -> gvlab, gvlab 3D backup -> laptop mirror."""
    s2 = state_2d()
    b2 = backup_2d() if s2.get("run_dir_exists") else {"state": "NOTHING_TO_BACK_UP", "utc": utc(), "t": time.time()}
    b2r = backup_2d(REM2D, "remainder_laptop") if (REM2D / "run_manifest.json").exists() else {"state": "NOTHING_TO_BACK_UP"}
    m2s = mirror_2d_spillover()
    m3 = mirror_3d()
    print(json.dumps({"utc": utc(), "b2": {k: b2.get(k) for k in ("state", "roots_packed", "roots_verified_on_destination", "files_failing")},
                      "b2_remainder": b2r.get("state"), "m2_spillover": m2s.get("state"),
                      "m3": {k: m3.get(k) for k in ("state", "units_in_index", "files_failing", "rsync_rc")}}))


def cmd_restore_test(kind, out):
    """Copy one backed-up unit into a fresh temporary directory, re-hash against the source of truth and open its arrays."""
    t = Path(tempfile.mkdtemp(prefix=f"restore_{kind}_"))
    res = {"utc": utc(), "kind": kind, "restore_dir": str(t)}
    try:
        import numpy as np
    except ImportError:
        np = None
    if kind == "3d":
        idx = (jload(BACKUP3 / "backup_index.json", {}) or {}).get("units", {})
        led = {e["unit"]: e for e in jl(RUN4 / "ledger.jsonl") if e.get("event") == "unit_done"}
        u = next((u for u, v in sorted(idx.items()) if v.get("npz_sha256")), None)
        if u:
            for e in ("json", "npz"):
                shutil.copyfile(BACKUP3 / "units" / f"{u}.{e}", t / f"{u}.{e}")
            ok = sha(t / f"{u}.json") == led[u]["json_sha256"] and sha(t / f"{u}.npz") == led[u]["npz_sha256"]
            doc = json.loads((t / f"{u}.json").read_text())
            arrays = sorted(np.load(t / f"{u}.npz").files)[:6] if np else None
            res.update(unit=u, hashes_equal_ledger=ok, records=len(doc["records"]), npz_arrays_sample=arrays, opened=True)
    else:
        man = jload(RELAY / "backup2d_source_manifest.json", {})
        rid = next((Path(f).stem for f in sorted(man) if f.startswith("witnesses/")), None) or next((Path(f).stem for f in sorted(man)), None)
        if rid:
            files = sorted(f for f in man if Path(f).stem == rid)
            rcs = []
            for f in files:                                       # keep the relative layout (modes/ and witnesses/ share basenames)
                (t / f).parent.mkdir(parents=True, exist_ok=True)
                rcs.append(subprocess.run(["rsync", "-a", f"{REMOTE}:{BACKUP2D}/{f}", str(t / f)], capture_output=True, text=True).returncode)
            ok = not any(rcs) and all(sha(t / f) == man[f] for f in files)
            rec = json.loads((t / "records" / f"{rid}.json").read_text()) if (t / "records" / f"{rid}.json").exists() else {}
            wz = sorted(np.load(t / "witnesses" / f"{rid}.npz").files)[:6] if np and (t / "witnesses" / f"{rid}.npz").exists() else None
            res.update(root=rid, files=sorted(files), hashes_equal_source_manifest=ok, queries=len(rec.get("queries", [])), npz_arrays_sample=wz,
                       opened=bool(rec))
    atomic(out, json.dumps(res, indent=1, sort_keys=True))
    print(json.dumps(res))


def cmd_state(out):
    sh = lambda *a: subprocess.run(list(a), capture_output=True, text=True)  # noqa: E731
    gv = sh("ssh", "-o", "BatchMode=yes", REMOTE, "crontab -l 2>/dev/null | grep -c 'status4.py snapshot'").stdout.strip()
    lap = sh("crontab", "-l").stdout.count("status4.py relay")
    log = sh("git", "-C", str(TCLONE), "log", "--format=%H %cI %s", f"origin/{TBRANCH}").stdout.splitlines()
    tele = [x for x in log if " telemetry " in x]
    doc = {"utc": utc(), "gvlab_cron_installed": gv not in ("", "0"), "laptop_cron_installed": lap > 0, "telemetry_commits": len(tele),
           "last_telemetry_commits": tele[:4], "backup_3d_second_disk": str(BACKUP3), "mirror_3d_laptop": str(MIRROR3),
           "backup_2d_gvlab": f"{REMOTE}:{BACKUP2D}", "durable_offsite": "DURABLE_OFFSITE_NOT_CONFIGURED"}
    atomic(out, json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps(doc))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "restore-test":
        cmd_restore_test(sys.argv[2], sys.argv[3])
    elif c == "state":
        cmd_state(sys.argv[2])
    else:
        {"snapshot": cmd_snapshot, "backup": cmd_backup, "relay": cmd_relay, "transfer": cmd_transfer}[c]()
