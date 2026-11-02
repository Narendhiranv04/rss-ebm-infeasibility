"""P8d: independent one-shot 2D simulator episodes of the frozen P8c rankings (A, B-full, B-structured, SF, U, G+M).

Per-episode guard 180 CPU-s / 600 s wall, aggregate 12,000 CPU-s / 24,000 s, at most 5 ranked candidates, interleaved per
root in the predeclared seeded arm order stored with the rankings; ordered.primary_v2 commits the first relocation witness
and attempts the target exactly once (no fallback). Rankings are checked against their frozen hash.
"""
from __future__ import annotations

import json
from pathlib import Path

import config as cf
import p7_oneshot as po


def run(args) -> int:
    out, rk = Path(args.out), json.loads(Path(args.rankings).read_text())
    if cf.stable_hash(rk["jobs"]) != rk["hash"]:
        raise SystemExit("rankings changed after they were frozen")
    out.mkdir(parents=True, exist_ok=True)
    po.CAPS.update({"cpu_s": 12000.0, "wall_s": 24000.0, "episode_cpu_s": 180.0, "episode_wall_s": 600.0, "max_episodes": 72, "max_rank": 5})
    led = po.run_episodes(out, Path(args.collect, "plan.json"), rk["jobs"], lambda m: open(out / "log.txt", "a").write(m + "\n"))
    print(json.dumps({k: v["outcome"] for k, v in led["episodes"].items()}, indent=1))
    return 0
