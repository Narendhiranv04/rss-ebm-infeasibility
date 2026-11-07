"""Fail-closed authorization for a FUTURE full SHELF66V3 placement collection (all splits).  Nothing here starts a worker.

A production run needs ALL of:
  * production_policy.json with a recorded user authorization (authorized_by / authorized_utc / statement, written by the user),
  * the policy file's frozen hashes equal to the CURRENT roster, source pin, split manifest, teacher/collector code and protocol,
  * --confirm TOKEN equal to token(current hashes)  (any change of policy / roster / source / code invalidates the token),
  * the requested policy / workers / CPU mask equal to the authorized ones, <= 18 logical CPUs, readable temperature sensor,
  * no other run3d.py process on the host.
  python guard3d.py token   POLICY_FILE                 print the token the CURRENT hashes would require (for the user's record)
  python guard3d.py check   POLICY_FILE TOKEN [workers cpus]   exit 0 only if every condition holds
"""
import hashlib, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = ("placement3d.py", "policy3d.py", "plan3d.py", "run3d.py", "guard3d.py")


def sha_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def current(pol):
    """Current hashes of everything the policy file freezes (paths are relative to the sidecar root)."""
    root = HERE.parent
    sys.path.insert(0, str(HERE))
    import placement3d as P
    return {"roster_sha256": sha_file(root / pol["roster"]), "source_pin_sha256": sha_file(root / pol["source_pin"]),
            "split_sha256": sha_file(root / pol["split_manifest"]), "code_sha256": {n: sha_file(HERE / n) for n in CODE},
            "teacher_protocol": P.protocol()[0], "policy": pol["policy"], "stages": pol["stages"], "workers": pol["workers"], "cpus": pol["cpus"]}


def token(h):
    return hashlib.sha256(json.dumps(h, sort_keys=True).encode()).hexdigest()[:32]


def other_collectors():
    me, out = os.getpid(), []
    for d in Path("/proc").iterdir():
        if d.name.isdigit() and int(d.name) not in (me, os.getppid()):
            try:
                cmd = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            if "run3d.py" in cmd and "--dry-run" not in cmd and "guard3d.py" not in cmd:
                out.append(int(d.name))
    return out


def check(policy_file, confirm, roster=None, policy=None, workers=None, cpus=None, source_pin=None):
    """-> (ok, reasons).  Every failed condition is listed; ok only if the list is empty."""
    why = []
    if not policy_file or not Path(policy_file).exists():
        return False, ["no production_policy.json given"]
    pol = json.loads(Path(policy_file).read_text())
    auth = pol.get("authorization") or {}
    if not all(auth.get(k) for k in ("authorized_by", "authorized_utc", "statement")):
        why.append("no recorded user authorization in production_policy.json (authorized_by / authorized_utc / statement)")
    try:
        cur = current(pol)
    except Exception as err:  # noqa: BLE001
        return False, why + [f"cannot compute current hashes: {type(err).__name__}: {err}"]
    for k in ("roster_sha256", "source_pin_sha256", "split_sha256", "code_sha256", "teacher_protocol"):
        if pol.get("frozen", {}).get(k) != cur[k]:
            why.append(f"frozen {k} differs from the current one")
    if not confirm or confirm != token(cur):
        why.append("confirmation token missing or not equal to token(current hashes)")
    root = HERE.parent
    if roster is not None and Path(roster).resolve() != (root / pol["roster"]).resolve():
        why.append("roster differs from the authorized roster")
    if source_pin is not None and Path(source_pin).resolve() != (root / pol["source_pin"]).resolve():
        why.append("source pin differs from the authorized pin")
    if policy is not None and policy != pol["policy"]:
        why.append("policy differs from the authorized policy")
    if workers is not None and int(workers) != int(pol["workers"]):
        why.append("worker count differs from the authorized one")
    if cpus is not None and str(cpus) != str(pol["cpus"]):
        why.append("CPU mask differs from the authorized one")
    if int(pol["workers"]) > 18:
        why.append("more than 18 workers")
    if other_collectors():
        why.append(f"another run3d.py is running: {other_collectors()}")
    try:
        sys.path.insert(0, str(HERE))
        import run3d
        run3d.sensors()
    except Exception as err:  # noqa: BLE001
        why.append(f"temperature sensor unreadable: {err}")
    return not why, why


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: guard3d.py token POLICY_FILE | check POLICY_FILE TOKEN [workers cpus]")
    if sys.argv[1] == "token":
        print(token(current(json.loads(Path(sys.argv[2]).read_text()))))
    elif sys.argv[1] == "check":
        ok, why = check(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None, workers=sys.argv[4] if len(sys.argv) > 4 else None,
                        cpus=sys.argv[5] if len(sys.argv) > 5 else None)
        print(json.dumps({"ok": ok, "reasons": why}, indent=1))
        sys.exit(0 if ok else 2)
