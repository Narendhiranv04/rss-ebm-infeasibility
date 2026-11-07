"""Production v3 collector for the FIRST K0-K3 cohort of the SHELF66V3 placement dataset.  It reuses the V2 collector (run3d.py: worker,
atomic JSON + NPZ writes, hash-validated resume, fail-closed monitor) unchanged and adds, around it:
  * every parameter from ONE frozen policy file, checked by prod3d.check (authorization, token, hashes, implemented schedule, QC roots,
    source pin, SHELF_SRC, sensor, disk, CPU idleness, other collectors, lock); no CLI parameter can differ from what was authorized;
  * the exclusive kernel flock (prod3d.acquire), taken AFTER every refusal check and BEFORE any directory is made, held to the end;
  * workers that die with the coordinator (PR_SET_PDEATHSIG = SIGKILL), so a killed coordinator never leaves writers behind;
  * unit validation extended with the frozen protocol, the pinned root bytes and the label semantics (1 = VERIFIED with witnesses;
    0 = NEG_CERTIFIED_GEOMETRY with its certificate or NEG_UNDER_POLICY with its scope; None = UNLABELED_* / INTERRUPTED, never 0),
    used for resume, for every finished unit and by `audit`;
  * fsynced ledger lines; integration-test runs marked in the manifest, the ledger and a marker file.

  python collect3d.py --policy-file POLICY.json --dry-run          metadata-only preflight of every roster row: 0 planner calls, creates
                                                                  nothing; exit 0 iff every non-authorization condition holds
  python collect3d.py --policy-file POLICY.json --confirm TOKEN    run / resume (identical command); SIGKILL of the group is safe
  python collect3d.py --policy-file POLICY.json --audit            full post-run audit of the run directory (no teacher call)
"""
import argparse, ctypes, gzip, json, os, signal, sys, threading, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod3d  # noqa: E402
import run3d  # noqa: E402

NEG_KINDS = {"NEG_UNDER_POLICY": {"BOUNDED_SEARCH", "EXECUTION", "DETERMINISTIC_PLAN"}, "NEG_CERTIFIED_GEOMETRY": {"CERTIFIED_GEOMETRY"}}
UNLABELED = {"UNLABELED_NO_VALID_POST_GRASP_STATE", "UNLABELED_SCENE_NOT_SETTLED", "INTERRUPTED"}
CTX = {}                                                        # protocol / root pin the extended validation checks against


def semantics(rec):
    """-> list of label-semantics problems of one record (empty = valid)."""
    lab, st, ws = rec.get("label"), rec.get("status"), rec.get("witnesses") or []
    bad = []
    if lab == 1:
        if st != "VERIFIED" or not ws or rec.get("witness_id") not in {w["witness_id"] for w in ws} or rec.get("negative_kind"):
            bad.append("label 1 without VERIFIED status / witnesses")
    elif lab == 0:
        if st not in NEG_KINDS or rec.get("negative_kind") not in NEG_KINDS[st] or ws:
            bad.append("label 0 without a negative kind of its status, or with witnesses")
        if st == "NEG_CERTIFIED_GEOMETRY" and not (rec.get("certificate") and rec.get("certified_by")):
            bad.append("certified negative without certificate")
        if st == "NEG_UNDER_POLICY" and not rec.get("scope"):
            bad.append("policy negative without scope")
    elif lab is None:
        if st not in UNLABELED or ws or rec.get("negative_kind"):
            bad.append("unlabeled record with a non-UNLABELED status, witnesses or a negative kind")
    else:
        bad.append(f"label {lab!r} not in (1, 0, None)")
    return bad


_V2_VALIDATE = run3d.validate_unit


def validate_unit_v3(run, uid, expect_qids, led=None):
    ok, why = _V2_VALIDATE(run, uid, expect_qids, led)
    if not ok:
        return ok, why
    doc = json.loads((Path(run) / "units" / f"{uid}.json").read_text())
    if CTX and doc.get("protocol") != CTX["protocol"]:
        return False, "protocol_mismatch"
    if CTX and doc.get("root_sha256") != CTX["roots"].get(doc["records"][0]["qid"].split(":")[0]):
        return False, "root_pin_mismatch"
    bad = [b for r in doc["records"] for b in semantics(r)]
    return (False, f"label_semantics: {bad[0]}") if bad else (True, "ok")


run3d.validate_unit = validate_unit_v3                          # resume_audit (run3d) resolves validate_unit at call time


def own_group():
    """Make this process the leader of its process group (kill -9 -PGID stops the whole run).  A session leader -- e.g. a collector
    exec'd by the shell tmux started -- already leads its own group and setpgid() would fail with EPERM: that case is accepted."""
    try:
        os.setpgrp()
    except PermissionError:
        if os.getsid(0) != os.getpid():
            raise
    return os.getpgrp() == os.getpid()


def ledger_open(path):
    """Append handle on the ledger.  A torn last line (a kill mid-write) is terminated first, so the next record starts on its own line
    (resume ignores the unparseable fragment; without this the new record would be glued to it and lost)."""
    path = Path(path)
    if path.exists() and path.stat().st_size:
        with open(path, "rb") as f:
            f.seek(-1, 2)
            torn = f.read(1) != b"\n"
        if torn:
            with open(path, "ab") as f:
                f.write(b"\n"); f.flush(); os.fsync(f.fileno())
    return open(path, "a")


def worker3(cpu, tasks, results, run, stages, guard, early, ppid):
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)   # PR_SET_PDEATHSIG: die with the coordinator
    if os.getppid() != ppid:
        os._exit(1)
    run3d.worker(cpu, tasks, results, run, stages, guard, early)


def load(pol):
    units, n_rows = run3d.load_units(prod3d.ROOT / pol["inputs"]["roster"], "all", set(), pol["unit"], tuple(pol["splits"]))
    return units, n_rows


def preflight(pol, units, n_rows, cur):
    """Metadata only.  -> facts about every row (counts per split / K / depth / band; static rows; regenerated roster equality)."""
    import plan3d
    rows = [json.loads(x) for x in gzip.open(prod3d.ROOT / pol["inputs"]["roster"], "rt")]
    phys = [r for r in rows if not r["static"]]
    coh = json.loads((prod3d.ROOT / pol["inputs"]["cohort_manifest"]).read_text())
    regen = None
    if pol["mode"] == "PRODUCTION_FIRST_COHORT":                # the roster must be exactly the frozen rule applied to the cohort roots
        sel = plan3d.load_selected(cur["source"]["SHELF_SRC"])
        spl = json.loads((prod3d.ROOT / pol["inputs"]["split_manifest"]).read_text())["split"]
        want = []
        for r in sorted(coh["cohort_roots"]):
            for x in plan3d.roster_for(sel[r]):
                x["split"], x["reference"] = spl[r], False
                want.append(json.dumps(x, sort_keys=True))
        regen = want == [json.dumps(x, sort_keys=True) for x in rows]
    c = lambda key: dict(sorted(Counter(str(key(r)) for r in phys).items()))  # noqa: E731
    return {"roster_rows": n_rows, "physical_queries": len(phys), "static_contrasts": len(rows) - len(phys),
            "units": len(units), "queries_in_units": sum(map(len, units.values())), "roots": len({r["root_id"] for r in phys}),
            "by_split": c(lambda r: r["split"]), "by_K": c(lambda r: r["K"]), "by_depth": c(lambda r: r["depth"]), "by_band": c(lambda r: r["band"]),
            "by_split_K": c(lambda r: f"{r['split']}:K{r['K']}"), "roster_regenerated_equal": regen,
            "static_rows_excluded_from_units": all(not q["static"] for v in units.values() for q in v)}


def audit(pol, cur):
    """Post-run audit: every roster unit done, ledger <-> JSON <-> NPZ <-> witness hashes, protocol, root pin, label semantics."""
    run = prod3d.ROOT / pol["run_dir"]
    units, _ = load(pol)
    pin = json.loads((prod3d.ROOT / pol["inputs"]["source_pin"]).read_text())
    CTX.update(protocol=cur["teacher_protocol"], roots=pin["roots"])
    ok, rep = run3d.resume_audit(run, units, apply=False)
    recs = [r for u in ok for r in json.loads((run / "units" / f"{u}.json").read_text())["records"]]
    lab = Counter(f"{r['label']}:{r['status']}:{r.get('negative_kind')}" for r in recs)
    return {"units": len(units), "validated": len(ok), "missing_or_invalid": sorted(set(units) - set(ok)), "failures": rep["ledger_units_failed"],
            "stray_files": rep["quarantined_files"], "queries": len(recs), "labels": dict(sorted(lab.items())),
            "witness_associations": sum(len(r["witnesses"]) for r in recs), "complete": len(ok) == len(units) and not rep["quarantined_files"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy-file", required=True); ap.add_argument("--confirm"); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--audit", action="store_true")
    a = ap.parse_args()
    pf = Path(a.policy_file).resolve()
    pol = json.loads(pf.read_text())
    os.environ.setdefault("SHELF_SRC", pol.get("source_layout", {}).get("SHELF_SRC", ""))
    run = prod3d.ROOT / pol.get("run_dir", "production_v3/__invalid__")
    existed = run.exists()
    okc, why, auth, cur = prod3d.check(pf, a.confirm, dry=a.dry_run or a.audit)
    if cur is None:
        sys.exit("REFUSED: " + "; ".join(why))
    if a.audit:
        print(json.dumps(audit(pol, cur), indent=1)); return
    import placement3d as P
    import policy3d as Q
    units, n_rows = load(pol)
    if a.dry_run:
        calls = Counter()                                      # every teacher entry point is counted: a dry run must call none of them
        for mod, name in ((P, "screen"), (P, "hold"), (P, "plan_branch"), (P, "execute"), (P, "verify"), (Q, "evaluate"), (Q, "run_unit")):
            setattr(mod, name, (lambda f, k: lambda *x, **kw: (calls.update([k]), f(*x, **kw))[1])(getattr(mod, name), name))
        init = P.Scene.__init__
        P.Scene.__init__ = lambda self, *x, **kw: (calls.update(["Scene"]), init(self, *x, **kw))[1]
        pre = preflight(pol, units, n_rows, cur)
        pre["teacher_entry_calls"] = sum(calls.values())
        res = {"dry_run": True, "mode": pol["mode"], "policy": pol["policy"], "stages": cur["implemented_stages"], "workers": pol["workers"],
               "cpus": pol["cpus"], "teacher_protocol": cur["teacher_protocol"], "SHELF_SRC": cur["source"]["SHELF_SRC"],
               "refusals_other_than_authorization": why, "authorization_refusals": auth, **pre,
               "planner_calls": sum(P.W.values()), "work_counters": dict(P.W), "run_dir_existed": existed,
               "run_dir_created": (not existed) and run.exists(), "lock_free": prod3d.probe(prod3d.lock_path(pol["mode"])) is None}
        print(json.dumps(res, default=str))
        good = (not why and res["planner_calls"] == 0 and res["teacher_entry_calls"] == 0 and not res["run_dir_created"]
                and res["static_rows_excluded_from_units"]
                and res["roster_regenerated_equal"] in (True, None))
        sys.exit(0 if good else 4)
    if not okc:
        sys.exit("REFUSED: " + "; ".join(why + auth))
    stages, early = [tuple(s) for s in cur["implemented_stages"]], cur["early_stop"]
    cpus = prod3d.cpus_of(pol["cpus"])
    if len(cpus) != pol["workers"] or len(cpus) > run3d.CEILING or not set(cpus) <= os.sched_getaffinity(0):
        sys.exit(f"REFUSED: {pol['workers']} workers on CPUs {cpus} (allowed {sorted(os.sched_getaffinity(0))})")
    t0c, ident = run3d.sensors()
    if t0c >= run3d.LIMITS["temp_resume_c"]:
        sys.exit(f"REFUSED: package temperature {t0c} C at start")
    try:                                                        # exclusive, atomic, released by the kernel when this process dies
        lock_fd = prod3d.acquire(prod3d.lock_path(pol["mode"]), {"run_dir": str(run), "mode": pol["mode"], "policy_file": str(pf)})
    except prod3d.LockHeld as err:
        sys.exit(f"REFUSED: the exclusive collector lock is held by {err}")
    pin = json.loads((prod3d.ROOT / pol["inputs"]["source_pin"]).read_text())
    CTX.update(protocol=cur["teacher_protocol"], roots=pin["roots"])
    roots = sorted({q["root_id"] for v in units.values() for q in v})
    root_sha = {r: pin["roots"][r] for r in roots}
    man = {"mode": pol["mode"], "policy_file_sha256": prod3d.sha_file(pf), "policy": pol["policy"], "stages": [list(s) for s in stages],
           "early_stop": early, "unit_mode": pol["unit"], "splits": pol["splits"], "roster_sha256": cur["inputs"]["roster"]["sha256"],
           "inputs": cur["inputs"], "units": {k: [q["qid"] for q in v] for k, v in units.items()}, "protocol": P.protocol(),
           "code": cur["code_sha256"], "source": cur["source"], "guard_cpu_s": pol["guard_cpu_s"],
           "worker_policy": {"workers": pol["workers"], "cpus": cpus}, "limits": run3d.LIMITS, "source_pin_roots": root_sha}
    own_group()
    os.sched_setaffinity(0, set(cpus))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[k] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    (run / "units").mkdir(parents=True, exist_ok=True)
    if pol["mode"] == "INTEGRATION_TEST":
        (run / "TEST_RUN_NOT_PRODUCTION.txt").write_text("Integration test of the production path (<= 12 TRAIN queries). Never part of the dataset.\n")
    mp_ = run / "run_manifest.json"
    norm = json.loads(json.dumps(man))
    if mp_.exists():
        old = json.loads(mp_.read_text())
        if {k: old.get(k) for k in man} != norm:
            sys.exit(f"REFUSED: RUN_DIR holds a different frozen run (differs in {sorted(k for k in man if old.get(k) != norm[k])})")
    else:
        mp_.write_text(json.dumps(man, indent=1, sort_keys=True))
    done, rep = run3d.resume_audit(run, units)
    todo = [u for u in units if u not in done]
    log = ledger_open(run / "ledger.jsonl")

    def write(rec):
        log.write(json.dumps(rec) + "\n"); log.flush(); os.fsync(log.fileno())
    write({"event": "start", "mode": pol["mode"], "t": time.time(), "pid": os.getpid(), "workers": pol["workers"], "cpus": cpus, "todo": len(todo),
           "resumed": len(done), "resume_audit": rep, "sensor": ident, "collector": "collect3d.py (production v3)"})
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    for u in sorted(todo, key=lambda u: -len(units[u])):
        tasks.put((u, units[u], root_sha[units[u][0]["root_id"]]))
    procs = [ctx.Process(target=worker3, args=(c, tasks, results, str(run), stages, pol["guard_cpu_s"], early, os.getpid()), daemon=True)
             for c in cpus[:max(1, min(pol["workers"], len(todo)))]]
    for p in procs:
        tasks.put(None); p.start()
    stop, state = threading.Event(), {"done": len(done)}
    th = threading.Thread(target=run3d.monitor, args=(str(run), procs, cpus, stop, state), daemon=True); th.start()
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
        good, reason = validate_unit_v3(run, uid, [q["qid"] for q in units[uid]], {"json_sha256": js, "npz_sha256": zs})
        if not good:
            state["safety_stop"] = [f"unit_validation_failed: {uid}: {reason}"]; break
        n += 1; state["done"] += 1
        write({"event": "unit_done", "unit": uid, "cpu": cpu, "wall_s": wall, "cpu_s": cs, "queries": nq, "t": time.time(), "json_sha256": js,
               "npz_sha256": zs})
    stop.set()
    if state.get("safety_stop"):
        run3d.kill_workers(procs)
    else:
        for p in procs:
            p.join(timeout=60)
    end = {"event": "end", "mode": pol["mode"], "t": time.time(), "wall_s": time.time() - t0, "units_done_now": n, "complete": n == len(todo),
           "safety_stop": state.get("safety_stop")}
    write(end)
    os.close(lock_fd)
    print(json.dumps(end))
    sys.exit(0 if end["complete"] else 3)


if __name__ == "__main__":
    main()
