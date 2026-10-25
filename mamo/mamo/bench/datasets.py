"""Datasets: folders of scene files in the reference format, with difficulty labels.

Built-in: the E-M4M release's clutter scenes with 5, 10 and 15 movable objects. Any other
folder of ``*.txt`` scene files (the format of mamo.scene.parse_scene_file /
write_scene_file) can be added in the app or in ``<workspace>/datasets.json``.

Labels follow the E-M4M / I-M4M papers: Easy, Medium or Hard if one, two or more than two
movable objects overlap the initial NGR V(gamma_OoI) of the xArm7's PLANRETRIEVAL with only
immovable obstacles; ``trivial`` if none do, ``infeasible`` if no such retrieval exists.

Generated datasets (generate_dataset, the environments of mamo.envs) are labelled by
construction instead: the difficulty preset they were sampled from (numbers of movable and
immovable objects, targets and blockers), stored in each scene's ``meta`` section.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT.parent / "archive/ros_src/em4m-main/dat/clutter_scenes"
DIFFICULTIES = ["trivial", "easy", "medium", "hard", "infeasible"]
LABEL_FIELDS = ["scene", "env", "task", "movables", "immovables", "targets", "blockers", "difficulty"]


@dataclass
class Dataset:
    name: str
    root: Path
    builtin: bool = False

    def scenes(self) -> list[Path]:
        return sorted(p for p in self.root.rglob("*.txt") if _looks_like_scene(p))

    def rel(self, p: Path) -> str:
        return p.relative_to(self.root).as_posix()

    def path(self, rel: str) -> Path:
        return self.root / rel


def _looks_like_scene(p: Path) -> bool:
    try:
        with open(p, encoding="utf-8") as fh:
            return fh.readline().strip() == "O"
    except (OSError, UnicodeDecodeError):
        return False


def workspace_datasets_file(ws: Path) -> Path:
    return ws / "datasets.json"


def list_datasets(ws: Path) -> dict[str, Dataset]:
    out: dict[str, Dataset] = {}
    for lvl in ("5", "10", "15"):
        if (REFERENCE / lvl).is_dir():
            out[f"reference-{lvl}"] = Dataset(f"reference-{lvl}", REFERENCE / lvl, True)
    f = workspace_datasets_file(ws)
    if f.exists():
        for name, path in json.loads(f.read_text(encoding="utf-8")).items():
            out[name] = Dataset(name, Path(path))
    return out


def add_dataset(ws: Path, name: str, folder: str | Path) -> Dataset:
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    f = workspace_datasets_file(ws)
    data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    data[name] = str(folder)
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return Dataset(name, folder)


def remove_dataset(ws: Path, name: str) -> None:
    f = workspace_datasets_file(ws)
    if f.exists():
        data = json.loads(f.read_text(encoding="utf-8"))
        data.pop(name, None)
        f.write_text(json.dumps(data, indent=2), encoding="utf-8")


def labels_path(ws: Path, ds: Dataset) -> Path:
    return ws / "labels" / f"{ds.name}.csv"


def load_labels(ws: Path, ds: Dataset) -> list[dict]:
    p = labels_path(ws, ds)
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def classify_scene(path: Path, seed: int = 0) -> dict:
    """Difficulty label of one scene (see module docstring)."""
    from mamo.ngr import compute_ngr
    from mamo.retrieval import plan_retrieval
    from mamo.robot.arm import Arm, ObstacleSet
    from mamo.scene import parse_scene_file
    from mamo.world import World

    sc = parse_scene_file(path)
    if sc.meta.get("difficulty"):  # generated (mamo.envs): labelled by construction
        c = sc.meta.get("counts", {})
        return {"env": sc.env, "task": sc.task["type"], "movables": c.get("movables", ""),
                "immovables": c.get("immovables", ""), "targets": c.get("targets", ""),
                "blockers": " ".join(map(str, sc.meta.get("blockers", []))), "difficulty": sc.meta["difficulty"]}
    w = World(sc)
    arm = Arm(w)
    plan = plan_retrieval(arm, w.cfg, sc.ooi, w.object_states(), ObstacleSet.static({sc.ooi_id}),
                          np.array(w.cfg.robot.home_q), np.random.default_rng(seed), sc.shelf)
    blockers = None
    if plan is not None:
        ngr = compute_ngr(arm, plan, sc.shelf, sc.ooi_id, w.object_states())
        blockers = [o.id for o in sc.movables if ngr.intersects(o)]
    n = None if blockers is None else len(blockers)
    difficulty = "infeasible" if n is None else {0: "trivial", 1: "easy", 2: "medium"}.get(n, "hard")
    return {"env": sc.env, "task": "retrieve", "movables": len(sc.movables), "immovables": len(sc.immovables) - 1,
            "targets": 1, "blockers": "" if blockers is None else " ".join(map(str, blockers)),
            "difficulty": difficulty}


def write_labels(ws: Path, ds: Dataset, rows: list[dict]) -> None:
    p = labels_path(ws, ds)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=LABEL_FIELDS, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(rows)


def import_reference_labels(ws: Path, csv_path: Path) -> list[str]:
    """Reuse results/scene_labels.csv (classify_scenes.py) for the built-in datasets."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    done = []
    for lvl in ("5", "10", "15"):
        name = f"reference-{lvl}"
        sel = [{"scene": r["scene"], "movables": r["movables"], "immovables": "", "blockers": r["blockers"],
                "difficulty": r["difficulty"]} for r in rows if r["level"] == lvl]
        if sel and not labels_path(ws, Dataset(name, REFERENCE / lvl)).exists():
            write_labels(ws, Dataset(name, REFERENCE / lvl), sel)
            done.append(name)
    return done


# ---------------------------------------------------------------- generation
def _generate_one(args: tuple) -> dict:
    env, task, diff, index, check, folder = args
    from mamo.envs import make_problem
    from mamo.scene import write_scene_file
    rel = f"{env}/{task}/{env}_{task}_{diff}_{index:04d}.txt"
    try:
        sc = make_problem(env, task, diff, index, check=check)
    except RuntimeError as exc:
        return {"scene": rel, "env": env, "task": task, "difficulty": f"error: {exc}"}
    f = Path(folder) / rel
    f.parent.mkdir(parents=True, exist_ok=True)
    write_scene_file(sc, f)
    c = sc.meta["counts"]
    return {"scene": rel, "env": env, "task": task, "movables": c["movables"], "immovables": c["immovables"],
            "targets": c["targets"], "blockers": " ".join(map(str, sc.meta["blockers"])), "difficulty": diff}


def generate_dataset(ws: Path, name: str, envs: list[str], tasks: list[str], difficulties: list[str], per: int,
                     start: int = 0, check: bool = True, workers: int = 4) -> Dataset:
    """Generate ``per`` problems for every (environment, task, difficulty) into
    ``<workspace>/datasets/<name>/<env>/<task>/``, label them and add the dataset.

    ``check``: keep only problems whose targets can be retrieved with the movables removed
    (mamo.envs.target_feasible). Problem k of a combination is always the same scene, so a
    dataset can be grown by raising ``per`` (or ``start``)."""
    from concurrent.futures import ProcessPoolExecutor
    folder = ws / "datasets" / name
    folder.mkdir(parents=True, exist_ok=True)
    ds = add_dataset(ws, name, folder)
    status_f = labels_path(ws, ds).with_suffix(".status.json")
    status_f.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(e, t, d, start + i, check, str(folder)) for e in envs for t in tasks for d in difficulties
            for i in range(per)]
    old = {r["scene"]: r for r in load_labels(ws, ds)}
    rows = []
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for k, row in enumerate(ex.map(_generate_one, jobs, chunksize=2)):
            rows.append(row)
            status_f.write_text(json.dumps({"state": "running", "done": k + 1, "total": len(jobs)}))
    old.update({r["scene"]: r for r in rows})
    write_labels(ws, ds, sorted(old.values(), key=lambda r: r["scene"]))
    status_f.write_text(json.dumps({"state": "done", "done": len(jobs), "total": len(jobs)}))
    return ds
