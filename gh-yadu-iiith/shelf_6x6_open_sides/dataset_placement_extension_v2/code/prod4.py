"""Production v4 guard for the Q2_3D_PARITY_V1 roster: the production-v3 guard (prod3d.py, byte-identical) re-parameterized, nothing else.
Importing this module configures prod3d for v4 in THIS process only:
  * VERSION SHELF66V3_PLACE_PRODUCTION_V4_Q2_PARITY: a V3 policy file (or its token) is refused here, and a V4 file is refused by the
    V3 entry points; the token binds the version, the v4 roster / manifest / forecast hashes and the v4 code set;
  * run directories under production_v4/run_ (production) and production_v4/integration/ (tests);
  * integration tests: <= 32 TRAIN queries on <= 4 roots, <= 4 workers (enough to touch every Q2 block, band and both shapes);
  * the frozen code set adds q2roster3d.py, prod4.py and collect4.py; the collector scan also sees the v4 entry points.
The lock file is the v3 one, so a v3 and a v4 collector can never run together.

  python prod4.py template ...   (same arguments as prod3d.py template)     python prod4.py token POLICY     python prod4.py check POLICY TOKEN
"""
import argparse, json, os, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod3d as G  # noqa: E402

VERSION = "SHELF66V3_PLACE_PRODUCTION_V4_Q2_PARITY"
G.VERSION = VERSION
G.PROD = {**G.PROD, "run_prefix": "production_v4/run_"}
G.TEST = {"max_queries": 32, "max_roots": 4, "max_workers": 4, "run_prefix": "production_v4/integration/"}
G.CODE = G.CODE + ("q2roster3d.py", "prod4.py", "collect4.py")
NAMES = ("run3d.py", "collect3d.py", "collect4.py", "launch_full_3d.sh", "launch_first_cohort.sh", "launch_v4.sh")


def other_collectors():
    """Other v2 / v3 / v4 collector or launcher processes on the host (any user), excluding this process and its ancestors."""
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
            if any(k in cmd for k in NAMES) and "--dry-run" not in cmd and "pytest" not in cmd and "prod3d.py" not in cmd and "prod4.py" not in cmd:
                out.append(int(d.name))
    return out


G.other_collectors = other_collectors

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("template")
    for k in ("mode", "policy", "cpus", "run-dir", "roster", "source-pin", "split", "qc", "cohort", "shelf-src", "forecast", "out"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--workers", type=int, required=True)
    p = sp.add_parser("token"); p.add_argument("policy_file")
    p = sp.add_parser("check"); p.add_argument("policy_file"); p.add_argument("token", nargs="?")
    a = ap.parse_args()
    if a.cmd == "template":
        G.cmd_template(a)
    elif a.cmd == "token":
        pol = json.loads(Path(a.policy_file).read_text())
        os.environ.setdefault("SHELF_SRC", pol["source_layout"]["SHELF_SRC"])
        print(G.token(G.current(pol), pol))
    else:
        ok, why, auth, _ = G.check(a.policy_file, a.token)
        print(json.dumps({"ok": ok, "reasons": why, "authorization_reasons": auth}, indent=1))
        sys.exit(0 if ok else 2)
