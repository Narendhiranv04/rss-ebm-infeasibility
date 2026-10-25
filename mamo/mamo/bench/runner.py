"""Experiments: methods x datasets x scenes x seeds, run in separate processes.

An experiment lives in ``<workspace>/experiments/<id>/``:

* ``config.json``   what to run (see ``new_experiment``)
* ``results.jsonl`` one JSON row per finished job (appended; a rerun skips finished jobs)
* ``status.json``   progress, written by the runner process
* ``solutions/``    pickled solutions (for plan figures and videos), if enabled
* ``log.txt``       runner log

Every job runs in its own process with a hard wall-clock limit (a hung or crashing
planner is recorded, not fatal). Planning time is the planner's own accounting
(``Solution.stats["planning_time"]``, the M4M-style sum of planner, MAPF and simulation
time); wall time is measured too. Solved plans are replayed from the initial scene with
mamo.execute, identically for every method.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import pickle
import random
import time
import traceback
from datetime import datetime
from pathlib import Path

from mamo.bench.datasets import list_datasets, load_labels

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKSPACE = Path(os.environ.get("MAMO_BENCH_HOME", ROOT / "workspace"))
DEFAULT_PLUGINS = [ROOT / "plugins"]


def plugin_dirs(ws: Path) -> list[Path]:
    extra = ws / "plugin_dirs.json"
    dirs = list(DEFAULT_PLUGINS)
    if extra.exists():
        dirs += [Path(p) for p in json.loads(extra.read_text(encoding="utf-8"))]
    return dirs


# ------------------------------------------------------------ experiments
def new_experiment(ws: Path, name: str, methods: list[dict], datasets: list[str], difficulties: list[str],
                   per_difficulty: int | None, seeds: list[int], workers: int, execute: bool = True,
                   save_solutions: bool = True, sample_seed: int | None = 0, timeout_factor: float = 2.0,
                   envs: list[str] | None = None, tasks: list[str] | None = None) -> Path:
    """Create an experiment folder. methods: [{"name", "budget", "params"}]; ``envs`` / ``tasks``
    restrict generated datasets to some environments / tasks (None: all)."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    slug = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)[:40] or "exp"
    d = ws / "experiments" / f"{stamp}_{slug}"
    d.mkdir(parents=True)
    cfg = {"name": name, "created": datetime.now().isoformat(timespec="seconds"), "methods": methods,
           "datasets": datasets, "difficulties": difficulties, "per_difficulty": per_difficulty,
           "seeds": seeds, "workers": workers, "execute": execute, "save_solutions": save_solutions,
           "sample_seed": sample_seed, "timeout_factor": timeout_factor, "envs": envs, "tasks": tasks}
    (d / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return d


def import_benchmark_csv(ws: Path, path: Path, name: str) -> Path:
    """Turn a scripts/run_benchmark.py CSV into an experiment (results only)."""
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    methods = sorted({r["algo"] for r in rows})
    exp = new_experiment(ws, name, [{"name": m, "budget": 0, "params": {}} for m in methods],
                                sorted({f"reference-{r['level']}" for r in rows}), ["imported"], None,
                                sorted({int(r["seed"]) for r in rows}), 0, execute=True, save_solutions=False)
    num = lambda v: float(v) if v not in ("", None) else None  # noqa: E731
    with open(exp / "results.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            row = {"id": f"{r['algo']}|reference-{r['level']}|{r['scene']}|{r['seed']}", "method": r["algo"],
                   "dataset": f"reference-{r['level']}", "scene": r["scene"], "difficulty": r["difficulty"],
                   "seed": int(r["seed"]), "solved": r["solved"] == "1", "reason": r.get("reason", ""),
                   "planning_time": num(r.get("planning_time")), "wall_time": num(r.get("wall_time")),
                   "sim_time": num(r.get("sim_time")), "mapf_time": num(r.get("mapf_time")),
                   "num_actions": num(r.get("num_pushes")), "num_pick_places": num(r.get("num_pick_places")) or 0,
                   "executed": (r["executed"] == "1") if r.get("executed") not in ("", None) else None,
                   "exec_reason": r.get("exec_reason", "")}
            fh.write(json.dumps(row) + "\n")
    (exp / "status.json").write_text(json.dumps({"state": "done (imported)", "total": len(rows), "done": len(rows)}))
    return exp


def list_experiments(ws: Path) -> list[Path]:
    root = ws / "experiments"
    return sorted((p for p in root.iterdir() if (p / "config.json").exists()), reverse=True) if root.exists() else []


def read_config(exp: Path) -> dict:
    return json.loads((exp / "config.json").read_text(encoding="utf-8"))


def read_status(exp: Path) -> dict:
    f = exp / "status.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {"state": "not started"}
    except json.JSONDecodeError:
        return {"state": "unknown"}


def read_results(exp: Path) -> list[dict]:
    f = exp / "results.jsonl"
    if not f.exists():
        return []
    rows = []
    for line in f.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def request_cancel(exp: Path) -> None:
    (exp / "cancel").write_text("1", encoding="utf-8")


def _scene_info(ds, rel: str, labels: dict) -> tuple[str, str]:
    """(environment, task) of a scene, from its label row or its file."""
    lab = labels.get(rel, {})
    if lab.get("task"):
        return lab.get("env") or "reference_shelf", lab["task"]
    from mamo.scene import parse_scene_file
    sc = parse_scene_file(ds.path(rel))
    return sc.env, sc.task["type"]


def build_jobs(ws: Path, cfg: dict, skipped: list | None = None) -> list[dict]:
    """Expand an experiment config into jobs (scenes chosen per dataset and difficulty).

    A method only runs on scenes whose task it solves (PlannerSpec.tasks); the (method, scene)
    pairs left out are appended to ``skipped``.
    """
    from mamo.bench.planners import REGISTRY
    dss = list_datasets(ws)
    jobs = []
    for dname in cfg["datasets"]:
        ds = dss[dname]
        labels = {r["scene"]: r for r in load_labels(ws, ds)}
        scenes = [ds.rel(p) for p in ds.scenes()]
        want = cfg["difficulties"]
        picked: list[tuple[str, str]] = []
        if "all" in want or not labels:
            picked = [(s, labels.get(s, {}).get("difficulty", "unlabeled")) for s in scenes]
        else:
            # generated datasets: "per difficulty" counts per (environment, task, difficulty)
            by: dict[tuple, list[str]] = {}
            for s in scenes:
                lab = labels.get(s, {})
                d = lab.get("difficulty")
                if d in want and (not cfg.get("envs") or lab.get("env") in cfg["envs"])                         and (not cfg.get("tasks") or (lab.get("task") or "retrieve") in cfg["tasks"]):
                    by.setdefault((lab.get("env") or "", lab.get("task") or "", d), []).append(s)
            for key in sorted(by, key=lambda k: (k[0], k[1], want.index(k[2]))):
                lst = by[key]
                if cfg.get("sample_seed") is not None:
                    lst = lst[:]
                    random.Random(cfg["sample_seed"]).shuffle(lst)
                if cfg.get("per_difficulty"):
                    lst = lst[: cfg["per_difficulty"]]
                picked += [(s, key[2]) for s in lst]
        for scene, diff in picked:
            env, task = _scene_info(ds, scene, labels)
            for m in cfg["methods"]:
                spec = REGISTRY.get(m["name"])
                if spec is not None and task not in spec.tasks:
                    if skipped is not None:
                        skipped.append((m["name"], f"{dname}/{scene}"))
                    continue
                for seed in cfg["seeds"]:
                    jobs.append({"id": f"{m['name']}|{dname}|{scene}|{seed}", "method": m["name"],
                                 "budget": float(m["budget"]), "params": m.get("params", {}), "dataset": dname,
                                 "scene": scene, "scene_path": str(ds.path(scene)), "difficulty": diff,
                                 "env": env, "task": task, "seed": int(seed), "execute": cfg["execute"],
                                 "save_solutions": cfg["save_solutions"]})
    if cfg.get("interleave"):
        # k-th scene of every (dataset, environment, task, difficulty) group before the (k+1)-th, so
        # an experiment stopped early still covers every group evenly
        rank: dict = {}
        seen: dict = {}
        for j in jobs:
            g = (j["dataset"], j["env"], j["task"], j["difficulty"])
            if (g, j["scene"]) not in rank:
                rank[(g, j["scene"])] = seen.get(g, 0)
                seen[g] = seen.get(g, 0) + 1
        jobs.sort(key=lambda j: rank[((j["dataset"], j["env"], j["task"], j["difficulty"]), j["scene"])])
    return jobs


# ------------------------------------------------------------------ jobs
ROW_KEYS = ("id", "method", "dataset", "scene", "difficulty", "env", "task", "seed", "budget")


def _run_job(job: dict, plugins: list[str], out: str, sol_dir: str) -> None:
    """Child process: plan one scene with one method, replay it, write the result row."""
    row = {k: job.get(k) for k in ROW_KEYS}
    t0 = time.perf_counter()
    try:
        from mamo.bench.planners import REGISTRY, load_plugins
        from mamo.config import Config
        from mamo.context import MAMOContext
        from mamo.execute import execute
        from mamo.scene import parse_scene_file
        load_plugins(plugins)
        spec = REGISTRY[job["method"]]
        row["family"] = spec.family
        scene = parse_scene_file(job["scene_path"])
        cfg = Config()
        if spec.configure:
            spec.configure(cfg)
        ctx = spec.make_context(scene, cfg, job["seed"]) if spec.make_context else MAMOContext(scene, cfg, job["seed"])
        sol = spec.fn(ctx, job["budget"], **{**spec.params, **job["params"]})
        wall = time.perf_counter() - t0
        robust = getattr(ctx, "robust", None)
        if robust is not None:
            robust.close()
        st = dict(sol.stats or {})
        acts = list(sol.pushes or [])
        n_pp = sum(getattr(a, "kind", "push") == "pick_place" for a in acts)
        row.update(solved=bool(sol.solved), reason=sol.reason or "",
                   planning_time=float(st.get("planning_time", wall)), wall_time=wall,
                   sim_time=float(st.get("sim_time", 0.0)), mapf_time=float(st.get("mapf_time", 0.0)),
                   push_planner_time=float(st.get("push_planner_time", 0.0)),
                   robot_planner_time=float(st.get("robot_planner_time", 0.0)),
                   num_actions=len(acts), num_pushes=len(acts) - n_pp, num_pick_places=n_pp)
        row["stats"] = {k: (v if isinstance(v, (int, float, str, bool)) else str(v)) for k, v in st.items()}
        if job["execute"] and sol.solved:
            ex = execute(scene, sol, cfg)
            row.update(executed=bool(ex.success), exec_reason=ex.reason)
        if job["save_solutions"]:
            f = Path(sol_dir) / (job["id"].replace("|", "__").replace("/", "_") + ".pkl")
            with open(f, "wb") as fh:
                pickle.dump({"job": job, "solution": sol, "ngr": ctx.ngr}, fh)
            row["solution_file"] = f.name
    except Exception as exc:
        row.update(solved=False, reason=f"ERROR {type(exc).__name__}: {exc}", error=traceback.format_exc(),
                   wall_time=time.perf_counter() - t0)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(row, fh)


def _remove(f: Path) -> None:
    """Delete a finished job's temporary file; on Windows a scanner may hold it for a moment."""
    for _ in range(20):
        try:
            f.unlink(missing_ok=True)
            return
        except PermissionError:
            time.sleep(0.25)
    # left behind: harmless, a resumed run never reads it for a finished job


def run_experiment(ws: Path, exp: Path) -> None:
    """Run (or resume) an experiment; blocks until done or cancelled."""
    cfg = read_config(exp)
    jobs = build_jobs(ws, cfg)
    done = {r["id"] for r in read_results(exp)}
    todo = [j for j in jobs if j["id"] not in done]
    (exp / "jobs.json").write_text(json.dumps(jobs, indent=1), encoding="utf-8")
    sol_dir = exp / "solutions"
    tmp = exp / "tmp"
    sol_dir.mkdir(exist_ok=True)
    tmp.mkdir(exist_ok=True)
    (exp / "cancel").unlink(missing_ok=True)
    plugins = [str(p) for p in plugin_dirs(ws)]
    log = open(exp / "log.txt", "a", encoding="utf-8")
    started = time.time()
    status = {"state": "running", "total": len(jobs), "done": len(done), "pid": os.getpid(),
              "started": datetime.now().isoformat(timespec="seconds"), "running": []}

    def write_status() -> None:
        status["updated"] = datetime.now().isoformat(timespec="seconds")
        status["elapsed"] = round(time.time() - started)
        (exp / "status.json").write_text(json.dumps(status, indent=1), encoding="utf-8")

    def finish(job: dict, proc, t_start: float, reason: str | None = None) -> None:
        f = tmp / f"{abs(hash(job['id']))}.json"
        row = None
        if reason is None and f.exists():
            for _ in range(20):  # the file may be held for a moment by a scanner (Windows)
                try:
                    row = json.loads(f.read_text(encoding="utf-8"))
                    break
                except json.JSONDecodeError:
                    break
                except OSError:
                    time.sleep(0.25)
        if row is None:
            row = {k: job.get(k) for k in ROW_KEYS}
            row.update(solved=False, reason=reason or f"CRASH (exit code {proc.exitcode})",
                       wall_time=time.time() - t_start)
        _remove(f)
        with open(exp / "results.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")
        status["done"] += 1
        log.write(f"[{status['done']}/{status['total']}] {job['id']}: solved={row.get('solved')} "
                  f"t={row.get('planning_time', row.get('wall_time', 0)):.1f}s {row.get('reason', '')} "
                  f"exec={row.get('executed', '')} {row.get('exec_reason', '')}\n")
        log.flush()

    ctx = mp.get_context("spawn")
    running: dict = {}
    write_status()
    try:
        while todo or running:
            if (exp / "cancel").exists():
                for proc, (job, t_start) in list(running.items()):
                    proc.terminate()
                status["state"] = "cancelled"
                break
            while todo and len(running) < max(1, int(cfg["workers"])):
                job = todo.pop(0)
                out = tmp / f"{abs(hash(job['id']))}.json"
                proc = ctx.Process(target=_run_job, args=(job, plugins, str(out), str(sol_dir)), daemon=True)
                proc.start()
                running[proc] = (job, time.time())
            for proc, (job, t_start) in list(running.items()):
                limit = job["budget"] * float(cfg.get("timeout_factor", 2.0)) + 180
                if not proc.is_alive():
                    proc.join()
                    finish(job, proc, t_start)
                    del running[proc]
                elif time.time() - t_start > limit:
                    proc.terminate()
                    proc.join()
                    finish(job, proc, t_start, f"TIMEOUT (killed after {limit:.0f} s)")
                    del running[proc]
            status["running"] = [j["id"] for j, _ in running.values()]
            write_status()
            time.sleep(1.0)
        else:
            status["state"] = "done"
    except Exception:
        status["state"] = "error"
        log.write(traceback.format_exc())
    status["running"] = []
    write_status()
    log.close()


def launch(ws: Path, exp: Path) -> int:
    """Start ``run_experiment`` in a detached background process; returns its pid."""
    import subprocess
    import sys
    flags = 0
    if os.name == "nt":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    p = subprocess.Popen([sys.executable, "-m", "mamo.bench", "run", str(exp), "--workspace", str(ws)],
                         cwd=str(ROOT), creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         close_fds=True)
    return p.pid


# ---------------------------------------------------------------- labelling
def _classify_one(args: tuple[str, str]) -> dict:
    rel, path = args
    from mamo.bench.datasets import classify_scene
    try:
        return {"scene": rel, **classify_scene(Path(path))}
    except Exception as exc:
        return {"scene": rel, "movables": "", "immovables": "", "blockers": "", "difficulty": f"error: {exc}"}


def label_dataset(ws: Path, name: str, workers: int = 6) -> None:
    from mamo.bench.datasets import labels_path, write_labels
    ds = list_datasets(ws)[name]
    scenes = ds.scenes()
    status_f = labels_path(ws, ds).with_suffix(".status.json")
    status_f.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with mp.get_context("spawn").Pool(workers) as pool:
        for row in pool.imap_unordered(_classify_one, [(ds.rel(p), str(p)) for p in scenes]):
            rows.append(row)
            status_f.write_text(json.dumps({"state": "running", "done": len(rows), "total": len(scenes)}),
                                encoding="utf-8")
    rows.sort(key=lambda r: r["scene"])
    write_labels(ws, ds, rows)
    status_f.write_text(json.dumps({"state": "done", "done": len(rows), "total": len(scenes)}), encoding="utf-8")


def launch_labelling(ws: Path, name: str, workers: int = 6) -> int:
    import subprocess
    import sys
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
    p = subprocess.Popen([sys.executable, "-m", "mamo.bench", "label", name, "--workers", str(workers),
                          "--workspace", str(ws)], cwd=str(ROOT), creationflags=flags,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, close_fds=True)
    return p.pid


def launch_generation(ws: Path, name: str, envs: list[str], tasks: list[str], difficulties: list[str], per: int,
                      start: int = 0, check: bool = True, workers: int = 4) -> int:
    """Run datasets.generate_dataset in a detached background process; returns its pid."""
    import subprocess
    import sys
    flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS if os.name == "nt" else 0
    cmd = [sys.executable, "-m", "mamo.bench", "--workspace", str(ws), "generate", name, "--envs", *envs,
           "--tasks", *tasks, "--difficulties", *difficulties, "--per", str(per), "--start", str(start),
           "--workers", str(workers)] + ([] if check else ["--no-check"])
    p = subprocess.Popen(cmd, cwd=str(ROOT), creationflags=flags, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, close_fds=True)
    return p.pid
