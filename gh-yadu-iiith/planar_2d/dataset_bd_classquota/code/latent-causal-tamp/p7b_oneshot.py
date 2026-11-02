"""P7b.6 driver: run the frozen one-shot rankings (runs/p7/training/rankings.json) as independent episodes."""
from __future__ import annotations

import json
from pathlib import Path

import config as cf
import p7_oneshot as po


def run(args) -> int:
    out, rk = Path(args.out), json.loads(Path(args.train, "rankings.json").read_text())
    if cf.stable_hash(rk["jobs"]) != rk["hash"]:
        raise SystemExit("ranking file changed after it was frozen")
    order = sorted(rk["jobs"], key=lambda j: ({"A": 0, "B": 1, "C": 2, "U": 3}[j["arm"]], j["root"]))
    led = po.run_episodes(out, Path(args.collect, "plan.json"), order, lambda m: open(out / "log.txt", "a").write(m + "\n"))
    print(json.dumps({k: v["outcome"] for k, v in led["episodes"].items()}, indent=1))
    return 0
