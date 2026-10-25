"""Benchmark M4M / E-M4M (and ablations) on the reference scenes.

Reproduces the papers' protocol as far as the xArm7 allows:
  * scenes from the reference release (dat/clutter_scenes/{5,10,15})
  * difficulty = movables overlapping the initial NGR (E-M4M paper): Easy 1,
    Medium 2, Hard > 2 (labels from scripts/classify_scenes.py)
  * budgets: M4M 120 s (M4M paper), E-M4M 300 s (E-M4M paper)
  * metrics: solved, total planning time, simulation time, MAPF / push /
    robot planner time, number of pushes

    python scripts/run_benchmark.py --labels results/scene_labels.csv \
        --algos m4m em4m --per-difficulty 10 --workers 6 --out results/benchmark.csv

Algorithms: m4m, m4m_hat (single CBS call), em4m, em4m_negdb (no positive
cache), em4m_posdb (no negative cache), em4m_nodb, em4m_tiebreak.
"""

from __future__ import annotations

import argparse
import csv
import random
import multiprocessing as mp
import sys
import time
import traceback

import numpy as np
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SCENES = ROOT.parent / "archive/ros_src/em4m-main/dat/clutter_scenes"

ALGOS = ["m4m", "m4m_hat", "em4m", "em4m_negdb", "em4m_posdb", "em4m_nodb", "em4m_tiebreak",
         "im4m", "im4m_pnp", "lazy_im4m", "lazy_im4m_pnp", "robust_im4m", "tilde_im4m",
         "dogar", "dogar_uniform"]
FIELDS = ["scene", "level", "difficulty", "algo", "seed", "solved", "reason", "planning_time", "sim_time",
          "mapf_time", "push_planner_time", "robot_planner_time", "num_pushes", "num_pick_places", "expansions", "push_sims",
          "valid_pushes", "db_hits", "executed", "exec_reason", "wall_time"]


def run_one(job: dict) -> dict:
    from mamo.config import Config
    from mamo.context import MAMOContext
    from mamo.em4m.planner import em4m
    from mamo.execute import execute
    from mamo.m4m.planner import m4m
    from mamo.scene import parse_scene_file

    row = {k: "" for k in FIELDS}
    row.update(scene=job["scene"], level=job["level"], difficulty=job["difficulty"], algo=job["algo"],
               seed=job["seed"])
    t0 = time.perf_counter()
    try:
        scene = parse_scene_file(SCENES / job["level"] / job["scene"])
        cfg = Config()
        algo = job["algo"]
        # Paper Table 2: NEG-DB caches only failures, POS-DB only successes, NO-DB neither.
        caches = {"em4m_negdb": (True, False), "em4m_posdb": (False, True), "em4m_nodb": (False, False)}
        if algo in caches:
            cfg.search.use_negative_cache, cfg.search.use_positive_cache = caches[algo]
        if algo == "em4m_tiebreak":
            cfg.search.priority = "tiebreak"
        robust = None
        plan_scene = scene
        if algo in ("robust_im4m", "tilde_im4m"):
            # I-M4M paper Sec. V-A, second experiment: masses and friction coefficients are
            # uncertain. Both plan with one sampled copy of each object; Robust I-M4M validates
            # pushes in 6 x 2 sampled simulations. Execution (--execute) uses the true values.
            from mamo.im4m.robust import RobustSim, sample_scene
            prng = np.random.default_rng(1000 + job["seed"])
            plan_scene = sample_scene(scene, prng)
            if algo == "robust_im4m":
                robust = RobustSim(scene, cfg, prng)
        ctx = MAMOContext(plan_scene, cfg, seed=job["seed"])
        if algo.startswith("em4m"):
            sol = em4m(ctx, job.get("budget_em4m"))
        elif algo.startswith("dogar"):
            from mamo.dogar.planner import dogar
            sol = dogar(ctx, job.get("budget_dogar"), goal_sampling="uniform" if algo == "dogar_uniform" else "closest")
        elif "im4m" in algo:
            from mamo.im4m.planner import im4m
            sol = im4m(ctx, job.get("budget_im4m"), pick_place=algo.endswith("_pnp"), lazy=algo.startswith("lazy"),
                       robust=robust)
            if robust is not None:
                robust.close()
        else:
            sol = m4m(ctx, job.get("budget_m4m"), one_shot=(algo == "m4m_hat"))
        st = sol.stats
        row.update(solved=int(sol.solved), reason=sol.reason, num_pushes=len(sol.pushes),
                   planning_time=round(st.get("planning_time", 0), 2), sim_time=round(st.get("sim_time", 0), 2),
                   mapf_time=round(st.get("mapf_time", 0), 2),
                   push_planner_time=round(st.get("push_planner_time", 0), 2),
                   robot_planner_time=round(st.get("robot_planner_time", 0), 2),
                   expansions=st.get("em4m_expansions", ""),
                   push_sims=st.get("em4m_push_sims", st.get("push_sims", 0)),
                   valid_pushes=st.get("em4m_valid_pushes", st.get("valid_pushes", 0)),
                   num_pick_places=st.get("num_pick_places", 0),
                   db_hits=st.get("em4m_db_hits", ""))
        if job.get("execute") and sol.solved:
            ex = execute(scene, sol, cfg)
            row.update(executed=int(ex.success), exec_reason=ex.reason)
    except Exception as exc:  # keep the batch going; record the failure
        row.update(solved=0, reason=f"ERROR {type(exc).__name__}: {exc}")
        traceback.print_exc()
    row["wall_time"] = round(time.perf_counter() - t0, 1)
    return row


def load_jobs(args) -> list[dict]:
    jobs = []
    with open(args.labels, newline="", encoding="utf-8") as fh:
        rows = [r for r in csv.DictReader(fh) if r["difficulty"] in args.difficulties]
    excluded: set[str] = set()
    for path in args.exclude or []:
        with open(path, newline="", encoding="utf-8") as fh:
            excluded |= {r["scene"] for r in csv.DictReader(fh)}
    rows = [r for r in rows if r["scene"] not in excluded]
    if args.scenes_from:
        # Exactly the scenes of earlier result CSVs (e.g. to add algorithms to a finished comparison).
        wanted: set[str] = set()
        for path in args.scenes_from:
            with open(path, newline="", encoding="utf-8") as fh:
                wanted |= {r["scene"] for r in csv.DictReader(fh)}
        rows = [r for r in rows if r["scene"] in wanted]
    if args.sample_seed is not None:
        # A random (reproducible) choice of scenes instead of the first ones in the label file.
        random.Random(args.sample_seed).shuffle(rows)
    taken: dict[str, int] = defaultdict(int)
    for r in rows:
        if taken[r["difficulty"]] >= args.per_difficulty:
            continue
        taken[r["difficulty"]] += 1
        for algo in args.algos:
            for seed in range(args.seeds):
                jobs.append({"scene": r["scene"], "level": r["level"], "difficulty": r["difficulty"],
                             "algo": algo, "seed": seed, "execute": args.execute,
                             "budget_m4m": args.budget_m4m, "budget_em4m": args.budget_em4m,
                             "budget_im4m": args.budget_im4m, "budget_dogar": args.budget_dogar})
    return jobs


def summarise(rows: list[dict]) -> str:
    diffs = ["easy", "medium", "hard"]
    algos = sorted({r["algo"] for r in rows}, key=ALGOS.index)
    count = defaultdict(lambda: [0, 0])
    times = defaultdict(list)
    for r in rows:
        k = (r["difficulty"], r["algo"])
        count[k][1] += 1
        if str(r["solved"]) == "1":
            count[k][0] += 1
            times[k].append(float(r["planning_time"]))
    lines = ["| Difficulty | " + " | ".join(algos) + " |", "|---|" + "---|" * len(algos)]
    for dif in diffs:
        cells = []
        for a in algos:
            s, n = count[(dif, a)]
            if n == 0:
                cells.append("-")
                continue
            ts = sorted(times[(dif, a)])
            med = f", median {ts[len(ts) // 2]:.0f} s" if ts else ""
            cells.append(f"{s}/{n}{med}")
        lines.append(f"| {dif} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(ROOT / "results/scene_labels.csv"))
    ap.add_argument("--algos", nargs="+", default=["m4m", "em4m"], choices=ALGOS)
    ap.add_argument("--difficulties", nargs="+", default=["easy", "medium", "hard"])
    ap.add_argument("--per-difficulty", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) // 2))
    ap.add_argument("--budget-m4m", type=float, default=None)
    ap.add_argument("--budget-em4m", type=float, default=None)
    ap.add_argument("--budget-im4m", type=float, default=None)
    ap.add_argument("--budget-dogar", type=float, default=None)
    ap.add_argument("--execute", action="store_true", help="replay solved plans end to end in simulation")
    ap.add_argument("--exclude", nargs="+", help="result CSVs whose scenes are skipped")
    ap.add_argument("--scenes-from", nargs="+", help="result CSVs whose scenes are used (and no others)")
    ap.add_argument("--sample-seed", type=int, help="pick scenes at random with this seed")
    ap.add_argument("--out", default=str(ROOT / "results/benchmark.csv"))
    args = ap.parse_args()

    jobs = load_jobs(args)
    print(f"{len(jobs)} runs on {args.workers} workers", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        with mp.Pool(args.workers) as pool:
            for row in pool.imap_unordered(run_one, jobs):
                wr.writerow(row)
                fh.flush()
                rows.append(row)
                print(f"[{len(rows)}/{len(jobs)}] {row['algo']:14s} {row['difficulty']:6s} {row['scene']}: "
                      f"solved={row['solved']} t={row['planning_time']}s pushes={row['num_pushes']} "
                      f"{row['reason']} {row['exec_reason']}", flush=True)
    table = summarise(rows)
    Path(args.out).with_suffix(".md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
