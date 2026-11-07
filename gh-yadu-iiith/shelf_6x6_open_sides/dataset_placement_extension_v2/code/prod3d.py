"""Production v3 guard, exclusive run lock and policy template for the FIRST K0-K3 cohort of the SHELF66V3 placement collection.

Every run parameter comes from ONE frozen policy file; nothing is taken from the command line or a default path.  A run needs ALL of:
  * mode PRODUCTION_FIRST_COHORT: the user's recorded authorization (authorized_by / authorized_utc / statement) and no test fixture;
    mode INTEGRATION_TEST: the test fixture, EMPTY user-authorization fields, <= 12 TRAIN queries on <= 4 roots, run dir under integration/;
  * policy == the policy locked by the mock-V2 decision (production), stages == the IMPLEMENTED schedule of that policy, 12 workers on
    CPUs 0-11 (production), teacher protocol == the protocol the mock-V2 evidence was produced with;
  * frozen hashes == current: roster, source pin, split manifest, QC roots, cohort manifest, collector / guard / teacher code, and the
    source at SHELF_SRC (path, selected.json, EXCLUDED_FIXED_OBJECT.json, generator + scene code, robot model, every roster root);
  * --confirm TOKEN == token(current)  (the token also binds the authorization text; any change invalidates it);
  * readable temperature sensor, enough free disk, no other collector process, CPUs 0-11 idle, and the exclusive lock (flock) free.
The lock is a kernel flock on LOCK_FILE held by the collector for its whole life: atomic (no check-then-start race) and released by the
kernel when the holder dies, so a crash / SIGKILL can never leave a permanent lock (the file's text is diagnostic only).

  python prod3d.py template --mode M --policy P --workers W --cpus C --run-dir D --roster R --source-pin S --split SP --qc Q
                            --cohort CM --shelf-src DIR --forecast F --out OUT.json        unauthorized template (authorization = null)
  python prod3d.py token POLICY_FILE        the token the CURRENT hashes require (prints nothing else)
  python prod3d.py check POLICY_FILE TOKEN  exit 0 only if every condition holds
"""
import argparse, fcntl, hashlib, json, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                                              # the sidecar directory; every policy path is relative to it
VERSION = "SHELF66V3_PLACE_PRODUCTION_V3"
MODES = ("PRODUCTION_FIRST_COHORT", "INTEGRATION_TEST")
CODE = ("placement3d.py", "policy3d.py", "plan3d.py", "run3d.py", "guard3d.py", "collect3d.py", "prod3d.py")
INPUTS = ("roster", "source_pin", "split_manifest", "qc_roots", "cohort_manifest", "forecast")
LOCK_FILE = Path.home() / ".cache/shelf66v3_place/collector.lock"
TEST_FIXTURE = "INTEGRATION_TEST_ONLY_NOT_PRODUCTION"
PROD = {"workers": 12, "cpus": "0-11", "unit": "root", "splits": ["train", "val", "test"], "guard_cpu_s": 1200.0, "run_prefix": "production_v3/run_"}
TEST = {"max_queries": 12, "max_roots": 4, "max_workers": 4, "run_prefix": "production_v3/integration/"}
OVERRIDES = ("RUN3D_SENSOR_OVERRIDE", "SHELF66V3_LOCK_FILE")     # test-only environment hooks: refused in production


class LockHeld(Exception):
    pass


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def lock_path(mode):
    o = os.environ.get("SHELF66V3_LOCK_FILE")
    return Path(o) if (o and mode == "INTEGRATION_TEST") else LOCK_FILE


def acquire(path, info):
    """Exclusive non-blocking flock.  -> fd (keep it open for the life of the run).  Raises LockHeld with the holder's text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        held = os.pread(fd, 4096, 0).decode(errors="replace")
        os.close(fd)
        raise LockHeld(held or "(holder wrote nothing yet)") from None
    os.ftruncate(fd, 0); os.pwrite(fd, json.dumps({**info, "pid": os.getpid(), "t": time.time()}).encode(), 0); os.fsync(fd)
    return fd


def probe(path):
    """-> None if the lock is free, else the holder's text.  Creates nothing (a missing lock file is a free lock)."""
    path = Path(path)
    if not path.exists():
        return None
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return None
    except BlockingIOError:
        return os.pread(fd, 4096, 0).decode(errors="replace") or "(holder wrote nothing yet)"
    finally:
        os.close(fd)


def other_collectors():
    """Other collector / v2 launcher processes on the host (any user), excluding this process and its ancestors."""
    mine, p = set(), os.getpid()
    while p > 1:
        mine.add(p)
        try:
            p = int(Path(f"/proc/{p}/stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    out = []
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) not in mine:
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if any(k in cmd for k in ("run3d.py", "collect3d.py", "launch_full_3d.sh", "launch_first_cohort.sh")) and "--dry-run" not in cmd \
                    and "pytest" not in cmd and "prod3d.py" not in cmd:
                out.append(int(d.name))
    return out


def implemented_schedule(policy):
    import policy3d as Q
    allp = Q.POLICIES | Q.FIXED
    if policy not in allp:
        raise KeyError(f"unknown policy {policy}")
    return [list(s) for s in allp[policy]], policy not in Q.REFERENCE


def locked_policy():
    """The policy the frozen mock-V2 decision selected, and the teacher protocol its evidence was produced with."""
    d = json.loads((ROOT / "3D_MOCK_V2_DECISION.json").read_text())
    m = json.loads((ROOT / "mock_v2/runs/adaptive_v1_A/run_manifest.json").read_text())
    return d["policy"]["choice_by_frozen_rule"], m["protocol"][0]


def source_files(src):
    """The generator / scene code, robot model and selection files the teacher reads from SHELF_SRC (relative path -> sha256)."""
    src = Path(src).resolve()
    code = src / "code"
    files = sorted(p for p in (code / "shelf66_v3").glob("*.py")) + sorted((code / "dataset-bd").glob("*.py"))
    scene = (code / "shelf66_v3" / "shelf_scene_6x6").resolve()
    files += sorted(scene.glob("*.py")) + sorted(p for p in (scene / "mujoco_menagerie" / "ufactory_xarm7").rglob("*") if p.is_file())
    out = {"selected.json": sha_file(src / "selected.json"), "raw/EXCLUDED_FIXED_OBJECT.json": sha_file(src / "raw/EXCLUDED_FIXED_OBJECT.json")}
    for p in files:
        rel = str(p.relative_to(src)) if p.is_relative_to(src) else "scene/" + str(p.relative_to(scene))
        out[rel] = sha_file(p)
    return out


def current(pol):
    """Everything the policy file freezes, recomputed now (paths are relative to the sidecar root)."""
    sys.path.insert(0, str(HERE))
    import placement3d as P
    stages, early = implemented_schedule(pol["policy"])
    return {"version": pol.get("version"), "mode": pol.get("mode"), "policy": pol["policy"], "stages": pol.get("stages"),
            "implemented_stages": stages, "early_stop": early, "workers": pol["workers"], "cpus": pol["cpus"], "unit": pol["unit"],
            "splits": pol["splits"], "guard_cpu_s": pol["guard_cpu_s"], "run_dir": pol["run_dir"],
            "inputs": {k: {"path": pol["inputs"][k], "sha256": sha_file(ROOT / pol["inputs"][k])} for k in INPUTS},
            "code_sha256": {n: sha_file(HERE / n) for n in CODE}, "teacher_protocol": P.protocol()[0],
            "source": {"SHELF_SRC": str(Path(P.SRC).resolve()), "files_sha256": source_files(P.SRC)}}


def frozen_view(cur):
    return {k: v for k, v in cur.items() if k not in ("authorization", "test_authorization")}


def token(cur, pol):
    return hashlib.sha256(json.dumps({**cur, "authorization": pol.get("authorization"), "test_authorization": pol.get("test_authorization")},
                                     sort_keys=True).encode()).hexdigest()[:32]


def cpus_of(s):
    out = []
    for p in str(s).split(","):
        a, _, b = p.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def check(policy_file, confirm, dry=False):
    """-> (ok, reasons, auth_reasons, cur).  reasons: every failed non-authorization condition; auth_reasons: missing authorization / token
    (reported separately so a metadata-only dry run of the UNAUTHORIZED template can show that everything else passes)."""
    why, auth = [], []
    if not policy_file or not Path(policy_file).is_file():
        return False, ["no policy file given"], [], None
    pol = json.loads(Path(policy_file).read_text())
    mode = pol.get("mode")
    if pol.get("version") != VERSION or mode not in MODES:
        return False, [f"policy file is not a {VERSION} file with a known mode"], [], None
    for k in OVERRIDES:
        if os.environ.get(k) and mode != "INTEGRATION_TEST":
            why.append(f"test-only environment override {k} set")
    try:
        cur = current(pol)
    except Exception as err:  # noqa: BLE001
        return False, why + [f"cannot compute current hashes: {type(err).__name__}: {err}"], [], None
    fz = pol.get("frozen") or {}
    for k, v in frozen_view(cur).items():
        if fz.get(k) != json.loads(json.dumps(v)):
            why.append(f"frozen {k} differs from the current one")
    if pol.get("stages") != cur["implemented_stages"]:
        why.append(f"stages {pol.get('stages')} != the implemented schedule of {pol['policy']} {cur['implemented_stages']}")
    lp, lproto = locked_policy()
    if cur["teacher_protocol"] != lproto:
        why.append(f"teacher protocol {cur['teacher_protocol']} != the protocol validated by mock V2 ({lproto}): revalidate first")
    a, t = pol.get("authorization") or {}, pol.get("test_authorization")
    rost = [json.loads(x) for x in __import__("gzip").open(ROOT / pol["inputs"]["roster"], "rt")]
    phys = [r for r in rost if not r["static"]]
    if mode == "PRODUCTION_FIRST_COHORT":
        for k, v in (("policy", lp), *((k, PROD[k]) for k in ("workers", "cpus", "unit", "splits", "guard_cpu_s"))):
            if pol.get(k) != v:
                why.append(f"{k} {pol.get(k)!r} != the frozen production value {v!r}")
        if not str(pol["run_dir"]).startswith(PROD["run_prefix"]):
            why.append(f"run_dir must start with {PROD['run_prefix']}")
        if t is not None:
            why.append("a production policy must not carry the integration-test fixture")
        if not all(isinstance(a.get(k), str) and a.get(k).strip() for k in ("authorized_by", "authorized_utc", "statement")):
            auth.append("no recorded user authorization (authorization.authorized_by / authorized_utc / statement)")
    else:
        if not (isinstance(t, dict) and t.get("fixture") == TEST_FIXTURE):
            why.append("integration test without the test fixture")
        if any(a.get(k) for k in ("authorized_by", "authorized_utc", "statement")):
            why.append("an integration test must leave the user-authorization fields empty")
        if len(phys) > TEST["max_queries"] or len({r["root_id"] for r in phys}) > TEST["max_roots"] or {r["split"] for r in rost} - {"train"}:
            why.append(f"test roster exceeds {TEST['max_queries']} TRAIN queries / {TEST['max_roots']} roots")
        if pol.get("splits") != ["train"]:
            why.append("an integration test runs the train split only")
        if int(pol["workers"]) > TEST["max_workers"] or not set(cpus_of(pol["cpus"])) <= set(range(12)):
            why.append("integration test: <= 4 workers on CPUs 0-11")
        if not str(pol["run_dir"]).startswith(TEST["run_prefix"]):
            why.append(f"run_dir must start with {TEST['run_prefix']}")
    if Path(cur["source"]["SHELF_SRC"]) != Path(pol.get("source_layout", {}).get("SHELF_SRC", "")).resolve():
        why.append(f"SHELF_SRC resolves to {cur['source']['SHELF_SRC']}, not the authorized {pol.get('source_layout', {}).get('SHELF_SRC')}")
    pin = json.loads((ROOT / pol["inputs"]["source_pin"]).read_text())
    qc = json.loads((ROOT / pol["inputs"]["qc_roots"]).read_text())["roots"]
    spl = json.loads((ROOT / pol["inputs"]["split_manifest"]).read_text())["split"]
    if pin.get("source_files_sha256") != cur["source"]["files_sha256"]:
        why.append("source code / model / selection files at SHELF_SRC differ from the pin")
    coh = json.loads((ROOT / pol["inputs"]["cohort_manifest"]).read_text())
    if mode == "PRODUCTION_FIRST_COHORT":
        for k in ("roster", "source_pin", "split_manifest", "qc_roots"):
            if coh.get("sha256", {}).get(k) != cur["inputs"][k]["sha256"]:
                why.append(f"{k} is not the one the cohort manifest froze")
    else:
        full = {json.dumps(json.loads(x), sort_keys=True) for x in __import__("gzip").open(ROOT / coh["files"]["roster"], "rt")}
        alien = [r["qid"] for r in rost if json.dumps(r, sort_keys=True) not in full]
        if alien:
            why.append(f"{len(alien)} test rows are not rows of the first-cohort roster, e.g. {alien[:3]}")
    roots = sorted({r["root_id"] for r in rost})
    import placement3d as P
    drift = [r for r in roots if pin["roots"].get(r) != sha_file(Path(P.SRC) / "raw/roots" / f"{r}.json")]
    inel = [r for r in roots if not (qc.get(r, {}).get("eligible") and qc[r]["root_sha256"] == pin["roots"].get(r))]
    wrong = [r["qid"] for r in rost if spl.get(r["root_id"]) != r["split"]]
    for bad, msg in ((drift, "roster roots differ from the source pin"), (inel, "roster roots without a QC pass"), (wrong, "rows whose split != manifest")):
        if bad:
            why.append(f"{len(bad)} {msg}, e.g. {bad[:3]}")
    if not dry:
        if not confirm or confirm != token(cur, pol):
            auth.append("confirmation token missing or != token(current hashes + authorization)")
    oc = other_collectors()
    if oc:
        why.append(f"another collector is running: {oc}")
    held = probe(lock_path(mode))
    if held:
        why.append(f"the exclusive collector lock is held: {held}")
    try:
        import run3d
        run3d.sensors()
    except Exception as err:  # noqa: BLE001
        why.append(f"temperature sensor unreadable: {err}")
    try:
        import psutil
        fc = json.loads((ROOT / pol["inputs"]["forecast"]).read_text())
        need = float(fc["disk_gb_buffered"]) + 5.0
        base = ROOT / pol["run_dir"]
        while not base.exists():
            base = base.parent
        free = psutil.disk_usage(str(base)).free / 2 ** 30
        if free < need:
            why.append(f"free disk {free:.1f} GB < forecast x1.5 + 5 GB = {need:.1f} GB")
        if mode == "PRODUCTION_FIRST_COHORT":
            busy = psutil.cpu_percent(interval=1.0, percpu=True)
            mb = sum(busy[c] for c in cpus_of(pol["cpus"])) / len(cpus_of(pol["cpus"]))
            if mb > 25.0:
                why.append(f"CPUs {pol['cpus']} are {mb:.0f}% busy with other work (> 25%): do not compete")
    except Exception as err:  # noqa: BLE001
        why.append(f"resource check failed: {type(err).__name__}: {err}")
    return not why and not auth, why, auth, cur


def cmd_template(a):
    sys.path.insert(0, str(HERE))
    stages, early = implemented_schedule(a.policy)
    rel = lambda p: str(Path(p).resolve().relative_to(ROOT))  # noqa: E731
    test = a.mode == "INTEGRATION_TEST"
    pol = {"version": VERSION, "mode": a.mode,
           "status": ("INTEGRATION TEST FIXTURE -- NOT A PRODUCTION AUTHORIZATION" if test else
                      "TEMPLATE -- NOT AUTHORIZED (first-cohort collection not started)"),
           "policy": a.policy, "stages": stages, "workers": a.workers, "cpus": a.cpus, "unit": "root",
           "splits": ["train"] if test else ["train", "val", "test"], "guard_cpu_s": 1200.0, "run_dir": a.run_dir,
           "inputs": {"roster": rel(a.roster), "source_pin": rel(a.source_pin), "split_manifest": rel(a.split), "qc_roots": rel(a.qc),
                      "cohort_manifest": rel(a.cohort), "forecast": rel(a.forecast)},
           "source_layout": {"SHELF_SRC": str(Path(a.shelf_src).resolve()),
                             "what": "the native dataset directory (raw/, selected.json, code/) of the checkout the run uses; its selected.json, "
                                     "EXCLUDED_FIXED_OBJECT.json, generator / scene code, robot model and every roster root must equal the source pin"},
           "authorization": {"authorized_by": None, "authorized_utc": None, "statement": None},
           "test_authorization": ({"fixture": TEST_FIXTURE, "max_queries": TEST["max_queries"], "max_roots": TEST["max_roots"],
                                   "note": "authorizes ONLY this bounded integration test; never the first-cohort or full collection"} if test else None),
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    os.environ["SHELF_SRC"] = pol["source_layout"]["SHELF_SRC"]
    pol["frozen"] = json.loads(json.dumps(frozen_view(current(pol))))
    pol["token_procedure"] = "after filling `authorization`: python code/prod3d.py token <this file>  (any later change invalidates it)"
    Path(a.out).write_text(json.dumps(pol, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"out": a.out, "mode": a.mode, "teacher_protocol": pol["frozen"]["teacher_protocol"],
                      "inputs": {k: v["sha256"][:16] for k, v in pol["frozen"]["inputs"].items()}}))


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("template")
    for k in ("mode", "policy", "cpus", "run-dir", "roster", "source-pin", "split", "qc", "cohort", "shelf-src", "forecast", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--workers", type=int, required=True)
    p = sp.add_parser("token"); p.add_argument("policy_file")
    p = sp.add_parser("check"); p.add_argument("policy_file"); p.add_argument("token", nargs="?")
    a = ap.parse_args()
    if a.cmd == "template":
        cmd_template(a)
    elif a.cmd == "token":
        pol = json.loads(Path(a.policy_file).read_text())
        os.environ.setdefault("SHELF_SRC", pol["source_layout"]["SHELF_SRC"])
        print(token(current(pol), pol))
    else:
        ok, why, auth, _ = check(a.policy_file, a.token)
        print(json.dumps({"ok": ok, "reasons": why, "authorization_reasons": auth}, indent=1))
        sys.exit(0 if ok else 2)
