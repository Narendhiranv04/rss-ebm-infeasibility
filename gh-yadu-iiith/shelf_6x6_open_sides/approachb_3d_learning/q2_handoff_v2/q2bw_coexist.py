"""Blackwell coexistence watchdog for the Q2 V2 BLACKWELL run (user-approved 2026-11-09 ~10:42 IST: run alongside sajid's jobs on E-cores 10-23).
Every 10 s: PAUSE the run (runner SIGSTOPs its workers; never touches anyone else's process) if fewer than 4 of CPUs 0-9 are idle (the >= 4 free
cores rule; our workers are pinned to 10-23 so pausing never feeds back), or yadu's generator appears; RESUME when both clear.  Removes only a
PAUSE it wrote.  Exits when the run's status.json is final.        usage: python3 q2bw_coexist.py RUN_DIR"""
import json, sys, time
from pathlib import Path

run = Path(sys.argv[1]); log = run.parent / "coexist_watchdog.log"; mine = run / "PAUSE.coexist"
YP = ("yadu_bd/venv/bin/python", "run_3dt_plus.py", "collect4.py", "collect3d.py")


def stat():
    return [list(map(int, l.split()[1:])) for l in Path("/proc/stat").read_text().splitlines() if l.startswith("cpu") and l[3].isdigit()]


def yadu():
    n = 0
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                c = (d / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            except OSError:
                continue
            n += any(k in c for k in YP) and not c.startswith(("bash", "sh ", "tmux"))
    return n


def say(m):
    with open(log, "a") as f:
        f.write(f"{time.strftime('%FT%TZ', time.gmtime())} {m}\n")


say("start")
while True:
    st = json.loads((run / "status.json").read_text()) if (run / "status.json").exists() else {}
    if st.get("final"):
        mine.unlink(missing_ok=True); say("run final, exit"); break
    a = stat(); time.sleep(1.0); b = stat()
    idle = sum(1 for x, y in list(zip(a, b))[:10] if (y[3] + y[4] - x[3] - x[4]) / max(1, sum(y) - sum(x)) > 0.5)
    y = yadu()
    bad = idle < 4 or y > 0
    if bad and not (run / "PAUSE").exists():
        (run / "PAUSE").write_text(f"coexist-watchdog idle0_9={idle} yadu={y}\n"); mine.touch(); say(f"PAUSE idle0_9={idle} yadu={y}")
    elif not bad and mine.exists():
        (run / "PAUSE").unlink(missing_ok=True); mine.unlink(missing_ok=True); say(f"RESUME idle0_9={idle}")
    time.sleep(9)
