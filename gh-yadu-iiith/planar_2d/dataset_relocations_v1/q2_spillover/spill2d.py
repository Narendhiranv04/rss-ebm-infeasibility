#!/usr/bin/env python3
"""Q2_2D_SPILLOVER_V1: share the REMAINING 2D Q2 workload between the laptop and gvlab, ONLY after 3D v4 has finished, passed its audit and
its 100% witness replay, and released gvlab.  Orchestration only: the 2D teacher, plan, policy, runner and audit (code_v2/) are imported
UNCHANGED; nothing here changes labels, seeds, task selection or the ADAPTIVE_FULL 8->32 policy.

Exactly-once by ROOT ownership (a 2D root is packed only when its whole Q2 task list is final, so a root is never split across hosts):
  ORIGINAL  every root already PACKED in the laptop's full_2d_v2 when it drained      (immutable; never written again)
  LAPTOP    every PARTIAL / touched root (resumed from its verified parts) + a measured share of UNTOUCHED roots   -> runs/remainder_laptop
  GVLAB     only UNTOUCHED roots                                                                                   -> gvlab runs/spillover_gvlab
The owner map is frozen and hashed before either scoped run starts; the global union must be exactly the 4,620 Q2 roots / 249,965 tasks
(238,160 teacher queries + 11,805 static rows) with zero overlap.  The final merged dataset is a NEW directory, audited with launch_check.

State machine (laptop host cron every 2 min, flock; one transition per tick, idempotent, fail closed):
  WAIT_3D -> VERIFY_3D (post-run audit + replay on gvlab) -> VERIFY_2D_SERVER (stage + 16-root TRAIN canary, fingerprints vs laptop)
  -> DRAIN_2D (documented DRAIN of the laptop coordinator) -> PARTITION -> LAUNCH -> MONITOR -> RECONCILE -> DONE ;  any failure -> BLOCKED_*
  (the laptop original run keeps collecting until DRAIN; a BLOCKED_* queue never touches it).

  laptop:  spill2d.py tick | status | snapshot-source OUT | classify | partition SHARE | laptop-run | merge DIR
  gvlab:   spill2d.py gate3d | post3d | canary-prep LAPTOP_BACKUP_MANIFEST | server-run --scope CANARY|SPILLOVER | canary-audit | canary-compare
           | stage-verify MANIFEST
  While the laptop's original 2D run is live, the laptop side runs ONLY stdlib work (ssh / rsync / JSON): every step that imports the 2D code or
  loads the plan runs on gvlab or after the drain (a laptop-side plan load tripped the runner's swap guard once; see reports/).
"""
import fcntl, gzip, hashlib, json, os, shutil, subprocess, sys, time
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "Q2_2D_SPILLOVER_V1"
HERE = Path(__file__).resolve().parent
EXPECT = {"pair_protocol_id": "938b09a23a581152", "budget_hash": "0cd2597b0d9141fb", "teacher_queries": 238160, "tasks": 249965, "roots": 4620,
          "policy": {"coverage": "ADAPTIVE_FULL", "pairs_per_query": 8, "rescue_pairs": 32, "reuse_archived": False, "witness_store_max": 32},
          "quota": "Q2", "production_policy_sha256": "cf6db768eb981695045349b13d411c42732b25c97235c931dc8bc0b9baf32f44"}
# ---- laptop (host) layout
L2D = Path("/home/naren/Desktop/rss-placement-ready-v2/gh-yadu-iiith/planar_2d/dataset_relocations_v1")
LAUNCH, ORIG, CODE = L2D / "launch_ready_2d_v2", L2D / "full_2d_v2", L2D / "code_v2"
SPILL = L2D / "q2_spillover"
STATE = SPILL / "state" / "QUEUE_TRIGGER_STATUS.json"
REM = SPILL / "runs" / "remainder_laptop"
MERGED = SPILL / "merged" / "q2_canonical"
LPY = "/home/naren/Desktop/rss3d_env/venv/bin/python"
# ---- gvlab layout
GROOT = "approachb3d_v3/2d_q2_spillover"                                   # relative to gvlab $HOME
GSTAGE = f"{GROOT}/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1"
GLAUNCH = f"{GSTAGE}/launch_ready_2d_v2_gvlab"                              # server launch dir: identical plan / split / policy, server config
GPY = f"{GROOT}/venv313/bin/python"
G3D = "approachb3d_v3/repo/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2"
GBACK = "/ssd_scratch/nav_placement2d_backup/full_2d_v2"                     # gvlab: hash-verified copy of the laptop's packed 2D roots
LAP_BACKUP = Path.home() / ".cache/shelf66v3_place/relay"                    # laptop: status4 transfer's source manifest + last verification
GPIN = "taskset -c 12,13 nice -n 15"                                        # gvlab light commands: the reserved coordinator CPUs
SRV = {"workers": 12, "worker_cpus": list(range(12)), "reserved_cpus": [12, 13], "project_cpu_ceiling": 18, "host_profile": "GVLAB_XEON5317_CORETEMP_V1"}
PROJECT_PROCS = ("collect3d.py", "collect4.py", "run3d.py", "replay3d.py", "launch_v4.sh", "qc3d.py", "finger3d.py", "integration3d.py",
                 "integration4.py", "launch_runner.py", "spill2d.py server-run")
CANARY_ROOTS = 16


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def H(*a):
    return hashlib.sha256(json.dumps([VERSION, *a]).encode()).hexdigest()


def utc():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic(path, text):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as f:
        f.write(text); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


def jload(p, d=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return d


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
    except OSError:
        return ""


def ancestors(pid=None):
    """This process and every ancestor up to init (the tmux/sh/subshell chain that launched it; it can never be a 3D job)."""
    out, p = set(), pid or os.getpid()
    while p > 1 and p not in out:
        out.add(p)
        try:
            p = int(Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return out


def project_processes(exclude=(), names=PROJECT_PROCS):
    # exclude the whole launch chain: the 2026-11-07 canary was refused because its own tmux shell's command line contains "spill2d.py server-run"
    me, out = {*ancestors(), os.getppid(), *exclude}, []
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) not in me:
            c = cmdline(d.name)
            if any(k in c for k in names) and "--dry-run" not in c and " status" not in c and " gate3d" not in c:
                out.append({"pid": int(d.name), "cmd": c[:140]})
    return out


def code2d(code_dir):
    """Import the UNCHANGED 2D modules from a code_v2 directory."""
    sys.path.insert(0, str(code_dir))
    import launch_plan as LP, launch_runner as R, launch_check as K, placement_teacher as T  # noqa: E401
    return LP, R, K, T


def q2_tasks(L, LP, R):
    """{root_id: (plan_row, [task, ...])} for the Q2 quota, in plan order (static rows included; they are recorded without a teacher call)."""
    return {p["root_id"]: (p, ts) for p, ts in R.tasks_for(L, lambda r: LP.in_quota(r, LP.QUOTAS["Q2"]))}


def check_frozen(L, T):
    """The frozen Q2 identity: policy file hash, policy, quota, pair protocol, budget hash."""
    pol = json.loads((Path(L) / "configs/production_policy.json").read_text())
    bad = [k for k, ok in (("production_policy_sha256", sha(Path(L) / "configs/production_policy.json") == EXPECT["production_policy_sha256"]),
                           ("policy", pol["policy"] == EXPECT["policy"]), ("quota", pol["quota"] == EXPECT["quota"]),
                           ("pair_protocol_id", T.pair_protocol()[0] == EXPECT["pair_protocol_id"]),
                           ("budget_hash", T.budget_hash(pol["policy"]) == EXPECT["budget_hash"])) if not ok]
    return pol, bad


# ------------------------------------------------------------------------------------------------------------------- gvlab: 3D gate
def gate3d(require_post3d=True, side=None, lock=None, host_checks=True, names=PROJECT_PROCS):
    """Fail-closed post-3D gate on gvlab (read-only).  -> {pass, reasons, facts}."""
    side = Path(side) if side else Path.home() / G3D
    run = side / "production_v4/run_q2_parity"
    why, f = [], {}
    man = jload(run / "run_manifest.json")
    if not man:
        return {"pass": False, "reasons": ["3D v4 run manifest absent"], "facts": f}
    led = [json.loads(x) for x in (run / "ledger.jsonl").read_text().splitlines() if x.strip().startswith("{") and x.strip().endswith("}")]
    done = {e["unit"] for e in led if e.get("event") == "unit_done"}
    ends = [e for e in led if e.get("event") == "end"]
    f.update(units_total=len(man["units"]), units_done=len(done), queries_total=sum(len(v) for v in man["units"].values()),
             last_end_complete=bool(ends and ends[-1].get("complete")))
    if f["queries_total"] != 59972 or f["units_total"] != 1466:
        why.append("3D manifest is not the frozen 59,972-query / 1,466-unit roster")
    if len(done) < 1466 or not f["last_end_complete"]:
        why.append(f"3D collection incomplete ({len(done)}/1466 units, last end complete={f['last_end_complete']})")
    if require_post3d:
        aud = jload(side / "production_v4/run_q2_parity_audit.json", {})
        rep = jload(side / "production_v4/run_q2_parity_replay.json", {})
        f.update(audit_complete=aud.get("complete"), audit_validated=aud.get("validated"), replay_replayed=rep.get("replayed"),
                 replay_verified=rep.get("verified"), replay_failures=len(rep.get("failures") or []), witness_associations=aud.get("witness_associations"))
        if not (aud.get("complete") is True and aud.get("validated") == 1466 and not aud.get("missing_or_invalid") and not aud.get("stray_files")):
            why.append("3D post-run audit missing or not complete")
        if not (rep.get("replayed") and rep.get("replayed") == rep.get("verified") == aud.get("witness_associations") and not rep.get("failures")):
            why.append("3D 100% positive-witness replay missing or not fully verified")
    procs = project_processes(names=names)
    f["project_processes"] = procs
    if procs:
        why.append(f"project compute still active on gvlab: {[p['cmd'][:60] for p in procs][:4]}")
    lock = Path(lock) if lock else Path.home() / ".cache/shelf66v3_place/collector.lock"
    if lock.exists():
        fd = os.open(lock, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            why.append("3D collector lock still held")
        finally:
            os.close(fd)
    if not host_checks:
        return {"pass": not why, "reasons": why, "facts": f, "utc": utc()}
    import psutil
    busy = psutil.cpu_percent(interval=2.0, percpu=True)
    t = None
    for h in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
        try:
            if (h / "name").read_text().strip() == "coretemp":
                t = int((h / "temp1_input").read_text()) / 1000
        except OSError:
            pass
    f.update(cpu0_13_busy_pct=round(sum(busy[:14]) / 14, 1), load1=os.getloadavg()[0], coretemp_c=t,
             mem_avail_gb=round(psutil.virtual_memory().available / 2 ** 30, 1), disk_free_gb=round(psutil.disk_usage(str(Path.home())).free / 2 ** 30, 1))
    for k, ok in (("CPUs 0-13 busy with other work (> 25%)", f["cpu0_13_busy_pct"] <= 25), ("coretemp unreadable", t is not None),
                  ("coretemp >= 70 C at start", (t or 99) < 70), ("available memory < 8 GB", f["mem_avail_gb"] >= 8),
                  ("free disk < 20 GB", f["disk_free_gb"] >= 20)):
        if not ok:
            why.append(k)
    return {"pass": not why, "reasons": why, "facts": f, "utc": utc()}


def post3d():
    """gvlab: the 3D post-run verification of V4_RUNBOOK.md section 8 (audit + 100% replay), only once the collection is complete and idle."""
    g = gate3d(require_post3d=False)
    if not g["pass"]:
        raise SystemExit("REFUSED post3d: " + "; ".join(g["reasons"]))
    side = Path.home() / G3D
    py, env = str(Path.home() / "approachb3d_mock/env/bin/python"), {**os.environ, "SHELF_SRC": str(side.parents[0] / "dataset_bd_classquota_without_home_position_no_overpass")}
    a = subprocess.run([py, "code/collect4.py", "--policy-file", "production_v4/production_policy_v4.json", "--audit"], cwd=side, env=env,
                       capture_output=True, text=True)
    if a.returncode != 0:
        raise SystemExit("3D audit failed to run: " + a.stderr[-500:])
    atomic(side / "production_v4/run_q2_parity_audit.json", a.stdout)
    r = subprocess.run(["taskset", "-c", "0-11", py, "code/replay3d.py", "ours", "production_v4/run_q2_parity",
                        "production_v4/frozen/q2_parity_roster_v4.jsonl.gz", "production_v4/run_q2_parity_replay.json", "--workers", "12", "--cpus", "0-11"],
                       cwd=side, env=env, capture_output=True, text=True)
    print(json.dumps({"audit_rc": a.returncode, "replay_rc": r.returncode, "replay_tail": r.stdout[-300:]}))


# ------------------------------------------------------------------------------------------------------------------- gvlab: server runner
def server_profile(R):
    """GVLAB_XEON5317_CORETEMP_V1: the coordinator's thermal input is the coretemp package sensor (the laptop's k10temp Tctl does not exist on
    this Intel server); thresholds and execution live in the server launch config.  Nothing about planning, seeds or tasks changes."""
    def sensors():
        for hw in sorted(Path("/sys/class/hwmon").glob("hwmon*")):
            try:
                if (hw / "name").read_text().strip() != "coretemp":
                    continue
            except OSError:
                continue
            lab = {f.read_text().strip(): f.with_name(f.name.replace("_label", "_input")) for f in hw.glob("temp*_label")}
            pk = next((v for k, v in lab.items() if k.startswith("Package id")), None)
            if pk is not None:
                return {"Tctl": pk}
        return None
    R.sensors = sensors


def check_server_profile(C):
    """-> reasons the server launch config is not the frozen GVLAB_XEON5317_CORETEMP_V1 profile (12 workers on 0-11, coordinator 12-13, <= 18)."""
    e = C.get("execution", {})
    cpus = set(e.get("worker_cpus", [])) | set(e.get("reserved_cpus", []))
    return [k for k, ok in (("host_profile", C.get("host_profile") == SRV["host_profile"]), ("workers", e.get("workers") == SRV["workers"]),
                            ("worker_cpus", e.get("worker_cpus") == SRV["worker_cpus"]), ("reserved_cpus", e.get("reserved_cpus") == SRV["reserved_cpus"]),
                            ("cpu_ceiling", len(cpus) <= SRV["project_cpu_ceiling"] and max(cpus, default=0) < 24),
                            ("no_ac_requirement_on_server", C.get("safety", {}).get("require_ac_power") is False)) if not ok]


def validate_allowlist(scope, allow, tasks, own=None, own_sha=None):
    """-> reasons an allowlist may not run on gvlab.  SPILLOVER: exactly the frozen GVLAB share, UNTOUCHED only; CANARY: <= 64 TRAIN roots."""
    why, roots = [], allow.get("roots", [])
    if len(set(roots)) != len(roots):
        why.append("a root is listed twice")
    if scope == "SPILLOVER":
        if own is None or allow.get("owner_manifest_sha256") != own_sha:
            why.append("allowlist does not reference the frozen owner manifest hash")
        else:
            if sorted(roots) != sorted(r for r, o in own["owner"].items() if o == "GVLAB"):
                why.append("allowlist is not the frozen GVLAB share")
            if any(own["class"].get(r) != "UNTOUCHED" for r in roots):
                why.append("a PACKED / PARTIAL root is assigned to gvlab")
    elif len(roots) > 64 or any(tasks[r][0]["split"] != "train" for r in roots):
        why.append("canary must be <= 64 TRAIN roots")
    for r in roots:
        if r not in tasks or [t["trial_id"] for t in tasks[r][1]] != allow.get("trial_ids", {}).get(r):
            why.append(f"trial ids of {r} differ from the frozen plan"); break
    return why


def server_run(scope, allowlist_file, manifest_file):
    """gvlab: run ONLY the allowlisted roots (CANARY: finished TRAIN laptop roots; SPILLOVER: the frozen gvlab-owned UNTOUCHED roots)."""
    L = Path.home() / GLAUNCH
    LP, R, K, T = code2d(L.parent / "code_v2")
    server_profile(R)
    pol, bad = check_frozen(L, T)
    if bad:
        raise SystemExit(f"REFUSED server-run: frozen Q2 identity differs: {bad}")
    stage = json.loads(Path(manifest_file).read_text())
    base = L.parents[1]                                                       # .../src/gh-yadu-iiith/planar_2d (manifest paths are relative to it)
    drift = [f for f, h in stage["files"].items() if not (base / f).exists() or sha(base / f) != h]
    drift += [f"server copy {f}" for f in ("query_plan.jsonl.gz", "plan_hashes.json", "split_manifest.json", "configs/production_policy.json")
              if sha(L / f) != stage["files"].get(f"dataset_relocations_v1/launch_ready_2d_v2/{f}")]
    if drift:
        raise SystemExit(f"REFUSED server-run: staged 2D source differs from the laptop manifest: {drift[:3]}")
    g = gate3d(require_post3d=True)
    if not g["pass"]:
        raise SystemExit("REFUSED server-run (3D gate): " + "; ".join(g["reasons"]))
    C = R.cfg_of(L)
    bad = check_server_profile(C)
    if bad:
        raise SystemExit(f"REFUSED server-run: server execution profile differs from GVLAB_XEON5317_CORETEMP_V1: {bad}")
    if R.sensors() is None or R.read_temp(R.sensors()) is None:
        raise SystemExit("REFUSED server-run: coretemp package sensor unreadable")
    allow = json.loads(Path(allowlist_file).read_text())
    tasks = q2_tasks(L, LP, R)
    roots = allow["roots"]
    own_f = Path(allowlist_file).parent / "HANDOFF_ROOT_OWNER_MANIFEST.json"
    own = json.loads(own_f.read_text()) if scope == "SPILLOVER" and own_f.exists() else None
    bad = validate_allowlist(scope, allow, tasks, own, sha(own_f) if own is not None else None)
    if bad:
        raise SystemExit(f"REFUSED server-run: {bad}")
    out = L / "runs" / ("canary_gvlab" if scope == "CANARY" else "spillover_gvlab")
    sel = [tasks[r] for r in sorted(roots, key=lambda r: H("order", r))]
    res = R.run(L, out, sel, pol["policy"], C["execution"]["workers"], C["execution"]["worker_cpus"], C["execution"]["reserved_cpus"],
                f"Q2_{scope}_GVLAB")
    print(json.dumps({k: v for k, v in res.items() if k not in ("events", "safety")}, default=str))


def stage_verify(manifest_file):
    stage = json.loads(Path(manifest_file).read_text())
    base = (Path.home() / GSTAGE).parent
    bad = [f for f, h in stage["files"].items() if not (base / f).exists() or sha(base / f) != h]
    print(json.dumps({"files": len(stage["files"]), "differ_or_missing": bad[:10], "ok": not bad}))
    sys.exit(0 if not bad else 2)


# ------------------------------------------------------------------------------------------------------------------- laptop side
def snapshot_source(out):
    """SHA-256 manifest of the RUNNING 2D source (relative to planar_2d): code_v2, the frozen launch files, the 2D dataset (roots, code)."""
    base = L2D.parent
    files = sorted([p for p in (L2D / "code_v2").glob("*.py")] +
                   [LAUNCH / f for f in ("query_plan.jsonl.gz", "plan_hashes.json", "split_manifest.json", "plan_summary.json", "pilot_roster.json",
                                         "configs/launch_config.json", "configs/production_policy.json", "configs/frozen_hashes.json")] +
                   [p for p in (base / "dataset_bd_classquota").rglob("*") if p.is_file() and "__pycache__" not in p.parts])
    doc = {"version": VERSION, "utc": utc(), "base": str(base), "files": {str(p.relative_to(base)): sha(p) for p in files}}
    atomic(out, json.dumps(doc, indent=0, sort_keys=True))
    print(json.dumps({"files": len(doc["files"]), "out": str(out)}))


def classify(run=ORIG, L=LAUNCH):
    """Root classes of a STOPPED original run: PACKED (record + packs verified), PARTIAL (any parts/<root>/ content; final parts verified to
    belong to the root's Q2 task list), UNTOUCHED.  Read-only."""
    LP, R, K, T = code2d(L.parent / "code_v2")
    tasks = q2_tasks(L, LP, R)
    cls, snap, bad = {}, {}, []
    for rid, (p, ts) in tasks.items():
        rec, d = run / "records" / f"{rid}.json", run / "parts" / rid
        want = [t["trial_id"] for t in ts]
        if rec.exists():
            h = json.loads(rec.read_text())
            if [q["trial_id"] for q in h["queries"]] != want:
                bad.append(f"{rid}: packed record trial ids differ from the Q2 plan")
            files = [x for x in (rec, run / "modes" / f"{rid}.npz", run / "witnesses" / f"{rid}.npz") if x.exists()]
            cls[rid], snap[rid] = "PACKED", {str(x.relative_to(run)): sha(x) for x in files}
            if d.exists() and any(d.iterdir()):
                bad.append(f"{rid}: packed but parts/ still populated")
        elif d.exists() and any(d.iterdir()):
            fin = sorted(x.stem for x in d.glob("*.json") if T.final(x))
            if not set(fin) <= set(want):
                bad.append(f"{rid}: parts hold trial ids outside its Q2 task list")
            cls[rid], snap[rid] = "PARTIAL", {str(x.relative_to(run)): sha(x) for x in sorted(d.iterdir())}
        else:
            cls[rid] = "UNTOUCHED"
    return tasks, cls, snap, bad


def root_cost(ts, L=LAUNCH):
    """Laptop CPU-s estimate of a root: per-(block, dest_bin) means of the Q2 amendment forecast (static rows ~0)."""
    st = json.loads((L / "amendment_q2/Q2_AMENDMENT_DECISION.json").read_text())["forecast"]["estimators"]["strata"]
    return sum(st.get(f"{t['block'].split('_')[0]}/{t['dest_bin']}", {}).get("cpu_s", 0.0) for t in ts if not t["static_ineligible"])


def partition(server_share, run=ORIG, L=LAUNCH, out=SPILL / "handoff"):
    """Freeze the owner map (requires the original coordinator stopped and its lock free)."""
    lk = run / ".lock"
    if lk.exists():
        fd = os.open(lk, os.O_RDONLY)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(fd, fcntl.LOCK_UN)
        except BlockingIOError:
            raise SystemExit("REFUSED partition: the original laptop coordinator still holds its lock")
        finally:
            os.close(fd)
    tasks, cls, snap, bad = classify(run, L)
    if bad:
        raise SystemExit(f"REFUSED partition: classification failures {bad[:3]}")
    if not 0.0 <= server_share < 1.0:
        raise SystemExit("REFUSED partition: server share outside [0, 1)")
    own = {r: "ORIGINAL" for r, c in cls.items() if c == "PACKED"}
    own.update({r: "LAPTOP" for r, c in cls.items() if c == "PARTIAL"})
    unt = [r for r, c in cls.items() if c == "UNTOUCHED"]
    cost = {r: root_cost(tasks[r][1], L) for r in unt}
    by = defaultdict(list)
    for r in unt:
        by[(tasks[r][0]["split"], tasks[r][0]["verified_class"])].append(r)
    for k in sorted(by):                                     # per split x class: keep each host's mix; deterministic hash order
        acc = {"GVLAB": 0.0, "LAPTOP": 0.0}
        for r in sorted(by[k], key=lambda r: H("assign", r)):
            tot = acc["GVLAB"] + acc["LAPTOP"] + cost[r]
            host = "GVLAB" if acc["GVLAB"] + cost[r] <= server_share * tot + 1e-9 else "LAPTOP"
            own[r] = host; acc[host] += cost[r]
    assert set(own) == set(tasks) and len(own) == EXPECT["roots"], (len(own), len(tasks))
    ids = {r: [t["trial_id"] for t in tasks[r][1]] for r in tasks}
    allids = [i for r in sorted(ids) for i in ids[r]]
    teach = sum(1 for r in tasks for t in tasks[r][1] if not t["static_ineligible"])
    if len(allids) != len(set(allids)) or len(allids) != EXPECT["tasks"] or teach != EXPECT["teacher_queries"]:
        raise SystemExit("REFUSED partition: trial-id union is not the exact Q2 set")
    LPm, Rm, Km, Tm = code2d(L.parent / "code_v2")
    man = {"version": VERSION, "utc": utc(), "server_share_target": server_share, "class": cls, "owner": dict(sorted(own.items())),
           "counts": {o: sum(v == o for v in own.values()) for o in ("ORIGINAL", "LAPTOP", "GVLAB")},
           "class_counts": dict(Counter(cls.values())), "teacher_queries_by_owner": {o: sum(1 for r, v in own.items() if v == o for t in tasks[r][1]
                                                                                           if not t["static_ineligible"]) for o in ("ORIGINAL", "LAPTOP", "GVLAB")},
           "estimated_laptop_cpu_s_by_owner": {o: round(sum(cost.get(r, 0) for r, v in own.items() if v == o), 1) for o in ("LAPTOP", "GVLAB")},
           "trial_ids": ids, "snapshot_sha256": snap,
           "frozen": {"plan_hashes": json.loads((L / "plan_hashes.json").read_text()), "production_policy_sha256": sha(L / "configs/production_policy.json"),
                      "pair_protocol_id": Tm.pair_protocol()[0], "budget_hash": EXPECT["budget_hash"], "split_manifest_sha256": sha(L / "split_manifest.json")},
           "invariants": {"roots": len(own), "tasks": len(allids), "teacher_queries": teach, "duplicates": 0,
                          "gvlab_only_untouched": all(cls[r] == "UNTOUCHED" for r, v in own.items() if v == "GVLAB"),
                          "partial_roots_laptop_owned": all(own[r] == "LAPTOP" for r, c in cls.items() if c == "PARTIAL")}}
    out.mkdir(parents=True, exist_ok=True)
    mf = out / "HANDOFF_ROOT_OWNER_MANIFEST.json"
    if mf.exists():
        raise SystemExit("REFUSED partition: an owner map already exists (versioned handoffs only; never overwrite)")
    atomic(mf, json.dumps(man, sort_keys=True))
    msha = sha(mf)
    for host in ("GVLAB", "LAPTOP"):
        rs = sorted(r for r, v in own.items() if v == host)
        atomic(out / f"allowlist_{host.lower()}.json", json.dumps({"host": host, "owner_manifest_sha256": msha, "roots": rs,
                                                                  "trial_ids": {r: ids[r] for r in rs}}, sort_keys=True))
    atomic(out / "handoff_sha256sums", "".join(f"{sha(out / f)}  {f}\n" for f in sorted(os.listdir(out)) if f != "handoff_sha256sums"))
    print(json.dumps({k: man[k] for k in ("counts", "class_counts", "teacher_queries_by_owner", "estimated_laptop_cpu_s_by_owner", "invariants")}))
    return man


def copy_partials(own, allow, orig, rem):
    """Copy each laptop-owned PARTIAL root's parts from the (stopped, immutable) original run into the remainder run, verifying every file
    against the handoff snapshot before and after; idempotent (an already copied or packed root is skipped); never writes to orig."""
    rem.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in allow["roots"]:
        if own["class"][r] != "PARTIAL" or (rem / "parts" / r).exists() or (rem / "records" / f"{r}.json").exists():
            continue
        tmp = rem / "parts" / f".{r}.copy"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        for rel, h in own["snapshot_sha256"][r].items():
            if sha(orig / rel) != h:
                raise SystemExit(f"REFUSED laptop-run: original partial file changed since the handoff snapshot: {rel}")
            shutil.copy2(orig / rel, tmp / Path(rel).name)
            if sha(tmp / Path(rel).name) != h:
                raise SystemExit(f"copy of {rel} does not re-hash")
        os.replace(tmp, rem / "parts" / r); n += 1
    return n


def cache_hygiene(dirs):
    """Drop OUR OWN files from the page cache (POSIX_FADV_DONTNEED, read-only opens; no root, nothing deleted) so 25 workers starting at once
    find free memory instead of forcing reclaim, which under vm.swappiness=180 + zram swaps idle pages and eats the runner's 512 MB
    swap-growth allowance (measured: ~280 MB at the 2026-11-07 resume).  Lowers real memory pressure; the guard and its threshold are untouched."""
    n = 0
    for d in dirs:
        for f in Path(d).rglob("*") if Path(d).exists() else []:
            try:
                if f.is_file() and not f.is_symlink():
                    fd = os.open(f, os.O_RDONLY)
                    try:
                        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); n += 1
                    finally:
                        os.close(fd)
            except OSError:
                pass
    return n


def laptop_run(handoff=SPILL / "handoff"):
    """Laptop: resume ONLY the laptop-owned roots into the separate remainder run dir (verified partial parts copied first)."""
    LP, R, K, T = code2d(CODE)
    pol, bad = check_frozen(LAUNCH, T)
    if bad:
        raise SystemExit(f"REFUSED laptop-run: frozen Q2 identity differs: {bad}")
    own = json.loads((handoff / "HANDOFF_ROOT_OWNER_MANIFEST.json").read_text())
    allow = json.loads((handoff / "allowlist_laptop.json").read_text())
    if allow["owner_manifest_sha256"] != sha(handoff / "HANDOFF_ROOT_OWNER_MANIFEST.json") or \
            sorted(allow["roots"]) != sorted(r for r, v in own["owner"].items() if v == "LAPTOP"):
        raise SystemExit("REFUSED laptop-run: allowlist is not the frozen LAPTOP share")
    if (ORIG / "run_manifest.json").exists() and R.sensors() is None:
        raise SystemExit("REFUSED laptop-run: k10temp Tctl unreadable")
    tasks = q2_tasks(LAUNCH, LP, R)
    copy_partials(own, allow, ORIG, REM)
    C = R.cfg_of(LAUNCH)
    sel = [tasks[r] for r in sorted(allow["roots"], key=lambda r: H("order", r))]
    del tasks
    cache_hygiene([ORIG, REM, SPILL / "handoff", Path.home() / "Desktop/shelf66v3_offhost_mirror", Path.home() / "Desktop/placement2d_spillover_mirror",
                   Path.home() / "Desktop/rss-3d-telemetry"])
    res = R.run(LAUNCH, REM, sel, pol["policy"], C["execution"]["workers"], C["execution"]["worker_cpus"], C["execution"]["reserved_cpus"],
                "Q2_REMAINDER_LAPTOP")
    print(json.dumps({k: v for k, v in res.items() if k not in ("events", "safety")}, default=str))


def merge(gvlab_run_local, handoff=SPILL / "handoff"):
    """NEW canonical Q2 directory from the three disjoint sources by the frozen owner map; then launch_check.audit (re-certifies every witness)."""
    LP, R, K, T = code2d(CODE)
    own = json.loads((handoff / "HANDOFF_ROOT_OWNER_MANIFEST.json").read_text())
    src = {"ORIGINAL": ORIG, "LAPTOP": REM, "GVLAB": Path(gvlab_run_local)}
    if MERGED.exists():
        raise SystemExit("REFUSED merge: the canonical merged directory exists (never overwrite)")
    shutil.copyfile(LAUNCH / "split_manifest.json", SPILL / "split_manifest.json")          # launch_check finds the split in an ancestor
    tmp = MERGED.with_name(".q2_canonical.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for d in ("records", "modes", "witnesses"):
        (tmp / d).mkdir(parents=True)
    prov, seen = {}, set()
    stray = {o: sorted(f.stem for f in (src[o] / "records").glob("*.json") if own["owner"].get(f.stem) != o) for o in ("LAPTOP", "GVLAB")}
    if any(stray.values()):
        raise SystemExit(f"REFUSED merge: a scoped run packed roots it does not own: { {k: v[:3] for k, v in stray.items()} }")
    for rid, o in sorted(own["owner"].items()):
        if o == "ORIGINAL" and any(sha(ORIG / rel) != h for rel, h in own["snapshot_sha256"][rid].items()):
            raise SystemExit(f"REFUSED merge: original packed root {rid} changed since the frozen handoff snapshot")
        rec = src[o] / "records" / f"{rid}.json"
        if not rec.exists():
            raise SystemExit(f"REFUSED merge: {rid} owned by {o} is not packed in its run dir")
        h = json.loads(rec.read_text())
        ids = [q["trial_id"] for q in h["queries"]]
        if ids != own["trial_ids"][rid] or seen & set(ids):
            raise SystemExit(f"REFUSED merge: trial ids of {rid} differ from the owner map or collide")
        seen |= set(ids)
        for d, ext in (("records", "json"), ("modes", "npz"), ("witnesses", "npz")):
            f = src[o] / d / f"{rid}.{ext}"
            if f.exists():
                shutil.copy2(f, tmp / d / f.name)
                assert sha(f) == sha(tmp / d / f.name)
        prov[rid] = {"owner": o, "scope": h.get("scope"), "source_dir": str(src[o])}
    if len(seen) != EXPECT["tasks"]:
        raise SystemExit(f"REFUSED merge: {len(seen)} trial ids != {EXPECT['tasks']}")
    recon = {"version": VERSION, "utc": utc(), "owner_manifest_sha256": sha(handoff / "HANDOFF_ROOT_OWNER_MANIFEST.json"),
             "roots_by_owner": dict(Counter(own["owner"].values())), "roots_packed_by_source": {o: len(list((src[o] / "records").glob("*.json")))
                                                                                                for o in src},
             "tasks_union": len(seen), "expected_tasks": EXPECT["tasks"], "overlap_trial_ids": 0, "roots_not_owned_but_packed": stray,
             "original_roots_byte_identical_to_handoff_snapshot": True, "gvlab_roots_all_untouched_at_handoff": own["invariants"]["gvlab_only_untouched"],
             "sources": {o: str(p) for o, p in src.items()}}
    atomic(SPILL / "2D_SPLIT_RECONCILIATION.json", json.dumps(recon, indent=1, sort_keys=True))
    pol = json.loads((LAUNCH / "configs/production_policy.json").read_text())
    atomic(tmp / "run_manifest.json", json.dumps({"scope": "Q2_MERGED_CANONICAL", "policy": pol["policy"], "budget_hash": T.budget_hash(pol["policy"]),
                                                  "pair_protocol_id": T.pair_protocol()[0], "queries": len(seen), "roots": len(prov),
                                                  "owner_manifest_sha256": sha(handoff / "HANDOFF_ROOT_OWNER_MANIFEST.json")}, sort_keys=True))
    atomic(tmp / "provenance.json", json.dumps(prov, sort_keys=True))
    os.replace(tmp, MERGED)
    a = K.audit(MERGED)
    teach = sum(1 for _, q, _, _ in K.load(MERGED) if q["eligible"])
    rep = {"version": VERSION, "utc": utc(), "merged_dir": str(MERGED), "roots": len(prov), "tasks": len(seen), "teacher_queries": teach,
           "expected": {k: EXPECT[k] for k in ("roots", "tasks", "teacher_queries")}, "by_owner": dict(Counter(v["owner"] for v in prov.values())),
           "audit": a["verification"], "summary": a["summary"], "run_fingerprint": a["fingerprint"]["run_fingerprint"],
           "pass": a["verification"]["failed"] == 0 and teach == EXPECT["teacher_queries"] and len(seen) == EXPECT["tasks"]}
    atomic(SPILL / "MERGED_Q2_AUDIT.json", json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps({k: rep[k] for k in ("pass", "roots", "tasks", "teacher_queries", "by_owner")}))


def canary_candidates(tasks, packed):
    """Label-blind canary order: finished TRAIN roots with >= 3 blocks (root + prefixes), round-robin over verified class, hash order inside."""
    by = defaultdict(list)
    for r in packed:
        if r in tasks and tasks[r][0]["split"] == "train" and len({t["block"] for t in tasks[r][1]}) >= 3:
            by[tasks[r][0]["verified_class"]].append(r)
    for c in by:
        by[c].sort(key=lambda r: H("canary", r))
    out = []
    while any(by.values()):
        for c in sorted(by):
            if by[c]:
                out.append(by[c].pop(0))
    return out


def canary_prep(lap_manifest_file, L=None, backup=GBACK, ref=None, allow_out=None, n=CANARY_ROOTS):
    """gvlab: choose the canary from laptop roots ALREADY in the gvlab backup, verify every file byte-for-byte against the laptop's own
    source manifest (sha256 computed on the laptop from the original run), and copy them into runs/canary_laptop_ref -- the laptop reference
    of the comparison.  A root with any unverifiable file is skipped (recorded).  No laptop compute; never overwrites."""
    L = Path(L) if L else Path.home() / GLAUNCH
    LP, R, K, T = code2d(L.parent / "code_v2")
    tasks = q2_tasks(L, LP, R)
    man = json.loads(Path(lap_manifest_file).read_text())
    files = defaultdict(dict)
    for rel, h in man.items():
        files[Path(rel).stem][rel] = h
    B = Path(backup)
    ref = Path(ref) if ref else L / "runs" / "canary_laptop_ref"
    allow_out = Path(allow_out) if allow_out else Path.home() / GSTAGE / "q2_spillover/canary/allowlist_canary.json"
    if ref.exists():
        raise SystemExit("REFUSED canary-prep: the laptop reference dir exists (never overwrite)")
    tmp = ref.with_name(f".{ref.name}.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    pick, skipped = [], []
    for r in canary_candidates(tasks, [r for r in files if f"records/{r}.json" in files[r]]):
        if len(pick) == n:
            break
        rec = B / f"records/{r}.json"
        if not rec.exists() or [q["trial_id"] for q in json.loads(rec.read_text())["queries"]] != [t["trial_id"] for t in tasks[r][1]]:
            skipped.append([r, "record missing or trial ids differ"]); continue
        if any(not (B / rel).exists() or sha(B / rel) != h for rel, h in files[r].items()):
            skipped.append([r, "backup file does not match the laptop manifest"]); continue
        for rel, h in files[r].items():
            (tmp / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(B / rel, tmp / rel)
            if sha(tmp / rel) != h:
                raise SystemExit(f"copy of {rel} does not re-hash")
        pick.append(r)
    if len(pick) < n:
        raise SystemExit(f"REFUSED canary-prep: only {len(pick)} verifiable TRAIN roots in the backup (need {n}); skipped {skipped[:3]}")
    os.replace(tmp, ref)
    doc = {"version": VERSION, "utc": utc(), "roots": sorted(pick), "trial_ids": {r: [t["trial_id"] for t in tasks[r][1]] for r in pick},
           "laptop_reference_dir": str(ref), "laptop_manifest_sha256": sha(lap_manifest_file), "skipped": skipped,
           "reference_files_verified": sum(len(files[r]) for r in pick)}
    atomic(allow_out, json.dumps(doc, sort_keys=True))
    print(json.dumps({k: doc[k] for k in ("roots", "reference_files_verified", "skipped")}))
    return doc


def compare_core(lap, lap_labels, srv, srv_labels, srv_ver):
    """Exact per-trial fingerprint equality + label / negative-kind equality + a clean server audit (witness re-certification) = PASS_EXACT.
    Anything else (a missing trial, a label drift, an unexplained byte difference, a failed witness) is a HARD_BLOCK."""
    same = sum(lap[t] == srv.get(t) for t in lap)
    label_diff = [t for t in lap if tuple(srv_labels.get(t, [None, "MISSING"])) != tuple(lap_labels[t])]
    ok = len(lap) > 0 and same == len(lap) and set(srv) == set(lap) and srv_ver.get("failed") == 0 and not label_diff
    return {"trials": len(lap), "fingerprints_identical": same, "fingerprint_differences": [t for t in lap if lap[t] != srv.get(t)][:50],
            "label_or_negative_kind_differences": label_diff[:50], "pass": ok, "verdict": "PASS_EXACT" if ok else "HARD_BLOCK"}


def acquire_tick_lock(path):
    """Non-blocking exclusive queue lock: -> open file (held) or None when another tick owns the queue."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    lk = open(path, "w")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return lk
    except BlockingIOError:
        lk.close(); return None


def host_facts():
    import platform
    f = {"host": os.uname().nodename, "python": platform.python_version(), "cpu": next((ln.split(":", 1)[1].strip() for ln in
         open("/proc/cpuinfo") if ln.startswith("model name")), None)}
    for m in ("numpy", "scipy", "mujoco", "psutil"):
        try:
            f[m] = __import__(m).__version__
        except Exception:  # noqa: BLE001
            f[m] = None
    return f


def canary_compare(canary_fp_file, out, ref=ORIG, code=CODE):
    """Per-trial content fingerprints (launch_check.digest: canonical query fields, modes, attempts, witness arrays) gvlab vs the laptop
    reference (on gvlab: runs/canary_laptop_ref, byte-verified against the laptop manifest by canary-prep)."""
    LP, R, K, T = code2d(code)
    cf = json.loads(Path(canary_fp_file).read_text())
    srv, srv_ver = cf["fingerprint"]["queries"], cf["verification"]
    rows = K.load(ref, cf["roots"])
    lap = {q["trial_id"]: K.digest(ref, h, q, m, a) for h, q, m, a in rows}
    labs = {q["trial_id"]: (q.get("label"), q.get("negative_kind")) for _, q, _, _ in rows}
    core = compare_core(lap, labs, srv, cf.get("labels", {}), srv_ver)
    runs = sorted(Path(canary_fp_file).parent.glob("run_*.json"))
    rr = json.loads(runs[-1].read_text()) if runs else {}
    if rr.get("stop") or rr.get("interrupted") or rr.get("not_started"):
        core = {**core, "pass": False, "verdict": "HARD_BLOCK", "canary_run_incomplete": {k: rr.get(k) for k in ("stop", "interrupted", "not_started")}}
    rep = {"version": VERSION, "utc": utc(), "roots": len(cf["roots"]), **core, "server_audit": srv_ver, "server_cpu_s_per_query": cf.get("cpu_s_per_query"),
           "laptop_reference": str(ref), "server_host": host_facts(), "canary_run": {k: rr.get(k) for k in ("scope", "workers", "worker_cpus", "reserved_cpus",
                                                                                                         "finished_now", "stop", "wall_s")},
           "server_sensor": "coretemp 'Package id 0' via server_profile (GVLAB_XEON5317_CORETEMP_V1); the runner's result field 'sensor' is a fixed label",
           "laptop_cpu_s_per_query": round(sum(q["volatile"]["cpu_s"] for _, q, _, _ in rows if q["eligible"] and q["completed"]) /
                                           max(1, sum(1 for _, q, _, _ in rows if q["eligible"] and q["completed"])), 3)}

    atomic(out, json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps({k: rep[k] for k in ("verdict", "trials", "fingerprints_identical", "server_cpu_s_per_query", "laptop_cpu_s_per_query")}))
    return rep


# ------------------------------------------------------------------------------------------------------------------- state machine (laptop)
def ssh(cmd, timeout=600):
    return subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", "gvlab", cmd], capture_output=True, text=True, timeout=timeout)


def coordinator_pid(run):
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            c = cmdline(d.name)
            if "launch_runner.py full" in c or (f"spill2d.py laptop-run" in c and run == REM):
                return int(d.name)
    return None


HOLDER = "placement2d-q2-queue"


def holder_listed():
    try:
        return HOLDER in subprocess.run(["systemd-inhibit", "--list", "--no-pager"], capture_output=True, text=True, timeout=20).stdout
    except (OSError, subprocess.TimeoutExpired):
        return False


def hold():
    """Run under `systemd-inhibit --who=placement2d-q2-queue ...` started from the LOGIN session: keeps the laptop awake until the queue is DONE or
    BLOCKED_* and no laptop 2D coordinator (original or remainder) is alive.  Reads state only."""
    while True:
        st = (jload(STATE, {}) or {}).get("state", "")
        if (st == "DONE" or st.startswith("BLOCKED")) and coordinator_pid(ORIG) is None and coordinator_pid(REM) is None:
            return
        time.sleep(60)


def inhibitor_ok():
    """Can this (cron) context hold a logind sleep inhibitor?  polkit may refuse processes outside a login session."""
    try:
        return subprocess.run(["systemd-inhibit", "--what=sleep:idle:handle-lid-switch", "--who=placement2d-probe", "--why=probe", "--mode=block",
                               "true"], capture_output=True, timeout=20).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def tick():
    """One idempotent state transition.  Writes QUEUE_TRIGGER_STATUS.json; never touches the laptop original run before DRAIN_2D."""
    s = jload(STATE, {"version": VERSION, "state": "WAIT_3D", "history": []})
    st, now = s["state"], utc()

    def go(new, **kw):
        s.update(kw); s["history"] = (s.get("history") or [])[-40:] + [[now, st, new]]; s["state"] = new
    s["last_tick_utc"] = now
    if (SPILL / "state/PAUSE_QUEUE").exists():                              # operator hold: no transition, no remote action
        s["paused"] = True
        atomic(STATE, json.dumps(s, indent=1, sort_keys=True))
        print(json.dumps({"state": s["state"], "paused": True}))
        return
    s.pop("paused", None)
    try:
        if not s.get("inhibitor_probe") or time.time() - s["inhibitor_probe"].get("t", 0) > 3600:
            s["inhibitor_probe"] = {"t": time.time(), "utc": now, "cron_can_inhibit": inhibitor_ok(), "queue_holder_listed": holder_listed()}
        if st == "WAIT_3D":
            r = ssh(f"cd $HOME/{GSTAGE} && {GPIN} $HOME/{GPY} $HOME/{GSTAGE}/q2_spillover/spill2d.py gate3d --collection-only")
            if not r.stdout.strip():
                raise RuntimeError(f"gate3d over ssh produced no verdict (rc {r.returncode}): {r.stderr[-200:]}")
            g = json.loads(r.stdout.strip().splitlines()[-1])
            s["gate3d_collection"] = g
            if g.get("pass"):
                go("VERIFY_3D")
        elif st == "VERIFY_3D":
            r = ssh(f"cd $HOME/{GSTAGE} && {GPIN} $HOME/{GPY} $HOME/{GSTAGE}/q2_spillover/spill2d.py gate3d")
            if not r.stdout.strip():
                raise RuntimeError(f"gate3d over ssh produced no verdict (rc {r.returncode}): {r.stderr[-200:]}")
            g = json.loads(r.stdout.strip().splitlines()[-1])
            s["gate3d"] = g
            if g.get("pass"):
                go("VERIFY_2D_SERVER")
            elif not s.get("post3d_started") and all("audit" in x or "replay" in x for x in g.get("reasons", [])):
                ssh(f"tmux new-session -d -s q2_post3d 'taskset -c 0-13 $HOME/{GPY} $HOME/{GSTAGE}/q2_spillover/spill2d.py post3d > ~/{GROOT}/post3d.log 2>&1'")
                s["post3d_started"] = now
            elif s.get("post3d_started") and "q2_post3d" not in ssh("tmux ls 2>/dev/null").stdout and not g.get("pass"):
                go("BLOCKED_3D_POST_RUN_VERIFICATION", block_reason=g.get("reasons"))
        elif st == "VERIFY_2D_SERVER":
            pass_file = SPILL / "PORTABILITY_CANARY_REPORT.json"
            rep = jload(pass_file)
            gq, gqr = f"$HOME/{GSTAGE}/q2_spillover", f"{GSTAGE}/q2_spillover"     # shell paths absolute; rsync paths relative to the remote $HOME
            if rep:
                go("DRAIN_2D" if rep.get("pass") else "BLOCKED_CANARY", block_reason=None if rep.get("pass") else rep.get("verdict"))
            elif not s.get("canary_started"):
                last = jload(LAP_BACKUP / "backup2d_last.json", {})
                if last.get("state") != "VERIFIED":
                    raise RuntimeError(f"laptop -> gvlab 2D backup not verified ({last.get('state')}); canary waits")
                r = subprocess.run(["rsync", "-a", str(LAP_BACKUP / "backup2d_source_manifest.json"), f"gvlab:{gqr}/canary/laptop_backup_manifest.json"],
                                   capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError("could not send the laptop backup manifest to gvlab")
                chain = " && ".join([
                    f"taskset -c 0-11 $HOME/{GPY} {gq}/spill2d.py canary-prep {gq}/canary/laptop_backup_manifest.json",
                    f"OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $HOME/{GPY} {gq}/spill2d.py server-run --scope CANARY --allowlist {gq}/canary/allowlist_canary.json "
                    f"--manifest {gq}/source_manifest.json",
                    f"taskset -c 0-11 $HOME/{GPY} {gq}/spill2d.py canary-audit", f"taskset -c 0-11 $HOME/{GPY} {gq}/spill2d.py canary-compare"])
                ssh(f"mkdir -p {gq}/canary && tmux new-session -d -s q2_canary 'cd $HOME/{GSTAGE} && ({chain}) > ~/{GROOT}/canary.log 2>&1'")
                s["canary_started"] = now
            elif "q2_canary" not in ssh("tmux ls 2>/dev/null").stdout:
                r = subprocess.run(["rsync", "-a", f"gvlab:{gqr}/PORTABILITY_CANARY_REPORT.json", str(pass_file)], capture_output=True, text=True)
                if r.returncode != 0 or not pass_file.exists():
                    go("BLOCKED_CANARY", block_reason="the gvlab canary chain ended without a report (see gvlab ~/" + GROOT + "/canary.log)")
        elif st == "DRAIN_2D":
            if not (ORIG / "DRAIN").exists():
                (ORIG / "DRAIN").write_text(f"{VERSION} handoff drain requested {now}\n"); s["drain_requested"] = now
            elif coordinator_pid(ORIG) is None:
                last = jload(ORIG / "status.json", {})
                if last.get("final") and last.get("stop") is None:
                    go("PARTITION", original_end="DRAINED")
                elif (SPILL / "state/ACK_ORIGINAL_STOP").exists():         # the user acknowledged an earlier safety stop of the original run
                    go("PARTITION", original_end=f"SAFETY_STOP_{last.get('stop')}_ACKNOWLEDGED")
                else:                                                       # never turn a safety stop into an automatic laptop relaunch
                    s["attention"] = [f"ORIGINAL_2D_SAFETY_STOP:{last.get('stop')} -- inspect, then touch q2_spillover/state/ACK_ORIGINAL_STOP to hand off"]
        elif st == "PARTITION":
            rep = jload(SPILL / "PORTABILITY_CANARY_REPORT.json", {})
            srv = 12 / max(rep.get("server_cpu_s_per_query") or 1e9, 1e-6)
            lap = 25 / max(rep.get("laptop_cpu_s_per_query") or 1e9, 1e-6)
            share = round(srv / (srv + lap), 3)
            subprocess.run([LPY, str(HERE / "spill2d.py"), "partition", str(share)], check=True, capture_output=True, text=True)
            go("LAUNCH", server_share=share)
        elif st == "LAUNCH":
            h = SPILL / "handoff"
            if not s.get("gvlab_launched"):
                r = subprocess.run(["rsync", "-a", f"{h}/", f"gvlab:{GSTAGE}/q2_spillover/handoff/"], capture_output=True, text=True)
                if r.returncode != 0:
                    raise RuntimeError("handoff transfer to gvlab failed")
                ssh(f"tmux new-session -d -s q2_spillover 'cd $HOME/{GSTAGE} && OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 $HOME/{GPY} $HOME/{GSTAGE}/q2_spillover/spill2d.py "
                    f"server-run --scope SPILLOVER --allowlist $HOME/{GSTAGE}/q2_spillover/handoff/allowlist_gvlab.json "
                    f"--manifest $HOME/{GSTAGE}/q2_spillover/source_manifest.json >> ~/{GROOT}/spillover.log 2>&1'")
                s["gvlab_launched"] = now
            if not s.get("laptop_launched"):
                inh = ["systemd-inhibit", "--what=sleep:idle:handle-lid-switch", "--who=placement2d-q2-remainder", "--why=2D Q2 remainder collection",
                       "--mode=block"] if inhibitor_ok() else []
                env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
                       "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1"}
                subprocess.Popen(["setsid", "nohup", *inh, LPY, str(HERE / "spill2d.py"), "laptop-run"], env=env,
                                 stdout=open(SPILL / "remainder_laptop.log", "a"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(CODE))
                s["laptop_launched"] = now
                s["laptop_sleep_inhibitor"] = "OWN" if inh else ("QUEUE_HOLDER" if holder_listed() else "NONE")
            go("MONITOR")
        elif st == "MONITOR":
            own = json.loads((SPILL / "handoff/HANDOFF_ROOT_OWNER_MANIFEST.json").read_text())
            lap_done = sum(1 for r, v in own["owner"].items() if v == "LAPTOP" and (REM / "records" / f"{r}.json").exists())
            g = ssh(f"ls {GLAUNCH}/runs/spillover_gvlab/records 2>/dev/null | wc -l").stdout.strip()
            srv_alive = "q2_spillover" in ssh("tmux ls 2>/dev/null").stdout
            s["progress"] = {"laptop_remainder_roots_packed": lap_done, "laptop_remainder_roots_owned": own["counts"]["LAPTOP"],
                             "gvlab_roots_packed": int(g or 0), "gvlab_roots_owned": own["counts"]["GVLAB"],
                             "laptop_remainder_running": coordinator_pid(REM) is not None, "gvlab_spillover_running": srv_alive}
            s["attention"] = [x for x, bad in (("LAPTOP_REMAINDER_STOPPED_INCOMPLETE", lap_done < own["counts"]["LAPTOP"] and coordinator_pid(REM) is None),
                                               ("GVLAB_SPILLOVER_STOPPED_INCOMPLETE", int(g or 0) < own["counts"]["GVLAB"] and not srv_alive)) if bad]
            # stopped scoped runs are reported, never restarted automatically (fail closed; resume = the same command, finished roots kept)
            if lap_done == own["counts"]["LAPTOP"] and int(g or 0) == own["counts"]["GVLAB"]:
                go("RECONCILE")
        elif st == "RECONCILE":
            loc = SPILL / "gvlab_spillover_copy"
            r = subprocess.run(["rsync", "-a", f"gvlab:{GLAUNCH}/runs/spillover_gvlab/", f"{loc}/"], capture_output=True, text=True)
            if r.returncode == 0:
                m = subprocess.run([LPY, str(HERE / "spill2d.py"), "merge", str(loc)], capture_output=True, text=True)
                a = jload(SPILL / "MERGED_Q2_AUDIT.json", {})
                go("DONE" if a.get("pass") else "BLOCKED_MERGE", block_reason=None if a.get("pass") else m.stderr[-300:])
    except Exception as e:  # noqa: BLE001  -- fail closed: record and stay
        s["last_error"] = f"{now} {type(e).__name__}: {e}"[:400]
    atomic(STATE, json.dumps(s, indent=1, sort_keys=True))
    print(json.dumps({"state": s["state"], "last_error": s.get("last_error")}))


def retry():
    """Operator retry of a BLOCKED_* state after the cause is fixed (tick lock held; failed attempts are archived, never deleted)."""
    s = jload(STATE, {})
    st, now, tag = s.get("state", ""), utc(), utc().replace(":", "")
    back = {"BLOCKED_3D_POST_RUN_VERIFICATION": "VERIFY_3D", "BLOCKED_CANARY": "VERIFY_2D_SERVER", "BLOCKED_MERGE": "RECONCILE"}
    if st not in back:
        raise SystemExit(f"REFUSED retry: state {st!r} is not a retryable BLOCKED_* state")
    if st == "BLOCKED_CANARY":
        rp = SPILL / "PORTABILITY_CANARY_REPORT.json"
        if rp.exists():
            (SPILL / "reports").mkdir(exist_ok=True); os.replace(rp, SPILL / f"reports/canary_attempt_{tag}.json")
        gl, gq = f"$HOME/{GLAUNCH}/runs", f"$HOME/{GSTAGE}/q2_spillover"
        r = ssh(f"for d in {gl}/canary_gvlab {gl}/canary_laptop_ref {gq}/PORTABILITY_CANARY_REPORT.json; do [ -e $d ] && mv $d $d.attempt_{tag}; done; true")
        if r.returncode != 0:
            raise SystemExit("REFUSED retry: could not archive the gvlab canary attempt")
        s.pop("canary_started", None)
    if st == "BLOCKED_3D_POST_RUN_VERIFICATION":
        s.pop("post3d_started", None)
    if st == "BLOCKED_MERGE" and MERGED.exists():
        raise SystemExit(f"REFUSED retry: move {MERGED} aside first (the merged directory is never overwritten)")
    s["history"] = (s.get("history") or [])[-40:] + [[now, st, back[st], "operator retry"]]
    s.update(state=back[st], block_reason=None)
    atomic(STATE, json.dumps(s, indent=1, sort_keys=True))
    print(json.dumps({"state": s["state"]}))


def canary_audit():
    """gvlab, after the canary run: launch_check.audit of the canary run (re-certifies its witnesses) -> fingerprint file for the laptop."""
    L = Path.home() / GLAUNCH
    LP, R, K, T = code2d(L.parent / "code_v2")
    run = L / "runs" / "canary_gvlab"
    a = K.audit(run, procs=12)
    rows = K.load(run)
    comp = [q for _, q, _, _ in rows if q["eligible"] and q["completed"]]
    doc = {"roots": sorted({h["root_id"] for h, _, _, _ in rows}), "verification": a["verification"], "fingerprint": a["fingerprint"],
           "labels": {q["trial_id"]: [q.get("label"), q.get("negative_kind")] for _, q, _, _ in rows},
           "cpu_s_per_query": round(sum(q["volatile"]["cpu_s"] for q in comp) / max(1, len(comp)), 3)}
    atomic(run / "canary_fingerprint.json", json.dumps(doc, sort_keys=True))
    print(json.dumps({"failed": a["verification"]["failed"], "trials": len(rows)}))


if __name__ == "__main__":
    a = sys.argv[1:]
    c = a[0]
    if c == "gate3d":
        print(json.dumps(gate3d(require_post3d="--collection-only" not in a)))
    elif c == "post3d":
        post3d()
    elif c == "server-run":
        server_run(a[a.index("--scope") + 1], a[a.index("--allowlist") + 1], a[a.index("--manifest") + 1])
    elif c == "canary-audit":
        canary_audit()
    elif c == "stage-verify":
        stage_verify(a[1])
    elif c == "snapshot-source":
        snapshot_source(a[1])
    elif c == "classify":
        t, cl, sn, bad = classify()
        print(json.dumps({"classes": dict(Counter(cl.values())), "failures": bad[:5]}))
    elif c == "partition":
        partition(float(a[1]))
    elif c == "laptop-run":
        laptop_run()
    elif c == "merge":
        merge(a[1])
    elif c == "canary-prep":
        canary_prep(a[1])
    elif c == "canary-compare":                                            # gvlab: server canary vs the byte-verified laptop reference
        GL = Path.home() / GLAUNCH
        canary_compare(GL / "runs/canary_gvlab/canary_fingerprint.json", Path.home() / GSTAGE / "q2_spillover/PORTABILITY_CANARY_REPORT.json",
                       ref=GL / "runs/canary_laptop_ref", code=GL.parent / "code_v2")
    elif c == "tick":
        lk = acquire_tick_lock(SPILL / "state" / "tick.lock")
        if lk is None:
            sys.exit(0)                                                  # another tick holds the queue: exactly one winner
        tick()
    elif c == "retry":
        lk = acquire_tick_lock(SPILL / "state" / "tick.lock")
        if lk is None:
            raise SystemExit("a tick is running; try again in a minute")
        retry()
    elif c == "hold":
        hold()
    elif c == "status":
        print(json.dumps(jload(STATE, {"state": "NOT_INSTALLED"}), indent=1))
