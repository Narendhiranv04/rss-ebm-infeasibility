"""P7b.6: independent one-shot 2D simulator episodes of model-ranked repairs (A / B / C, + uniform control).

Each arm's ranking is frozen (hash recorded) before any episode. An episode is ordered.primary_v2: static
check, fixed-order complete relocation planning in rank order under the declared repair allowance, COMMIT of
the first relocation witness, execution / re-observation of every move, ONE target evaluation with
SAFE_RETREAT from the achieved state, no fallback after a target failure. Its own seed namespace
('p7_oneshot/<arm>/<root>') makes it independent of the teacher's collection runs. One process-tree guard per
episode (250 CPU-s, 600 s wall); a censored episode counts as UNRESOLVED. An empty ranking is an abstention.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

import config as cf
import data
import evaluate as ev
import g4d_run as gr
import geometry as ge
import ordered as od
import sequences as sq
import teacher
from scene import scene_from_dict

CAPS = {"cpu_s": 9000.0, "wall_s": 18000.0, "episode_cpu_s": 250.0, "episode_wall_s": 600.0, "max_episodes": 48, "max_rank": 5}


def worker(job_path) -> None:
    job = json.loads(Path(job_path).read_text())
    cfg, v2 = cf.load_config("config.toml"), cf.load_config_v2("config_v2_r4.toml")
    robot = ge.make_robot(cfg)
    root = json.loads(Path(job["plan"]).read_text())["roots"][job["root"]]
    scene = scene_from_dict(root["scene"])
    query = data.make_query(cfg, f"p7os_{job['arm']}_{job['root']}", scene, "t0", root["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
    act, d = query["learner"]["action"], Path(job["dir"])
    arrays = {}

    def target_eval(st, k):
        return ev.evaluate_target(robot, cfg, st, act, f"p7_oneshot/{job['arm']}/{job['root']}/{k}",
                                  on_witness=lambda rec, arr: arrays.setdefault("target", arr))
    ranked = [{"label": f"r{i}", "sequence": [tuple(a) for a in s], "assignment": sq.assignment([tuple(a) for a in s])}
              for i, s in enumerate(job["ranked"])]
    out = od.primary_v2(robot, cfg, query, scene, ranked, target_eval, cf.robot_profile_hash(cfg))
    w = out["commit"] and out["commit"]["attempt"]
    rel = [r.traj.arrays() for r in w.skills if r.traj is not None] if w else []
    if rel:
        gr._atomic_npz(d / "relocation.npz", {f"r{i}__{k}": v for i, a in enumerate(rel) for k, v in a.items()})
    if "target" in arrays:
        gr._atomic_npz(d / "target_witness.npz", {f"r0__{k}": v for k, v in arrays["target"].items()})
    cost = od.cost_record(robot, v2, ev.start(scene, query), w, cf.robot_profile_hash(cfg)) if w else None
    gr._atomic_json(d / "result.json", {
        "arm": job["arm"], "root": job["root"], "outcome": out["outcome"], "committed_sequence": out["commit"] and out["commit"]["order"],
        "committed_rank": out["commit"] and out["commit"]["rank"], "moves": len(w.order) if w else None,
        "relocation_plans": sum((t.get("plan") or {}).get("relocation_plans", 0) for t in out["tried"]), "tried": len(out["tried"]),
        "target_calls": len(out["target_invocations"]), "J_first": (cost or {}).get("J"), "audit": out["audit"]})


def run_episodes(out: Path, plan_path, jobs: list, log) -> dict:
    """jobs: [{arm, root, ranked}] in predeclared order; abstentions (empty ranking) are recorded, not run."""
    out.mkdir(parents=True, exist_ok=True)
    led = json.loads((out / "ledger.json").read_text()) if (out / "ledger.json").exists() else {"episodes": {}, "cpu_s": 0.0}
    w0 = time.time()
    for j in jobs[:CAPS["max_episodes"]]:
        key = f"{j['arm']}__{j['root']}"
        if key in led["episodes"]:
            continue
        if not j["ranked"]:
            led["episodes"][key] = {"outcome": "ABSTAINED", "arm": j["arm"], "root": j["root"], "cpu_s": 0.0}
            continue
        if led["cpu_s"] + CAPS["episode_cpu_s"] > CAPS["cpu_s"] or time.time() - w0 > CAPS["wall_s"] - CAPS["episode_wall_s"]:
            led["episodes"][key] = {"outcome": "NOT_RUN_BUDGET", "arm": j["arm"], "root": j["root"], "cpu_s": 0.0}
            continue
        d = out / key
        d.mkdir(parents=True, exist_ok=True)
        job = {**j, "plan": str(plan_path), "dir": str(d), "ranked": [list(map(list, s)) for s in j["ranked"]][:CAPS["max_rank"]]}
        (d / "job.json").write_text(json.dumps(job))
        g = teacher.guard([sys.executable, "-c", "import sys, p7_oneshot; p7_oneshot.worker(sys.argv[1])", str(d / "job.json")],
                          CAPS["episode_cpu_s"], CAPS["episode_wall_s"], d / "heartbeat.txt", 10.0, 0.25, Path("."))
        res = json.loads((d / "result.json").read_text()) if (d / "result.json").exists() else {"outcome": "UNRESOLVED_CENSORED"}
        led["episodes"][key] = {**res, "arm": j["arm"], "root": j["root"], "guard": {k: g[k] for k in ("stop", "cpu_s", "wall_s", "overshoot_s")},
                                "cpu_s": g["cpu_s"], "ranking_hash": cf.stable_hash(job["ranked"])}
        led["cpu_s"] += g["cpu_s"]
        (out / "ledger.json").write_text(json.dumps(led, indent=1, default=float))
        log(f"episode {key}: {led['episodes'][key]['outcome']} ({g['cpu_s']:.1f} CPU-s)")
    (out / "ledger.json").write_text(json.dumps(led, indent=1, default=float))
    return led


def uniform_ranking(cands, seed) -> list:
    rng = np.random.default_rng(seed)
    seqs = [c for c in cands if c]
    return [seqs[i] for i in rng.permutation(len(seqs))]
