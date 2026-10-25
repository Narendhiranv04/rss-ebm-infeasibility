"""GIFs of an experiment's scenes (with their task) and of every solved run's replay.

    python scripts/make_gifs.py scenes    <experiment folder> --out results/overnight
    python scripts/make_gifs.py solutions <experiment folder> --out results/overnight [--watch]

scenes:    <out>/scenes/<env>/<scene>.gif  camera circling the scene, then the OoI's retrieval
           (planned with only immovables) with its NGR and the movables in its way; plus
           <scene>_2d.png / _3d.png top and 3D views (mamo.bench.media.scene_views)
solutions: <out>/solutions/<env>/<scene>__<method>.gif  replay of the solved plan (iso + top)
           --watch keeps rendering new solutions until the experiment is done
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mamo.bench import media, runner  # noqa: E402


def _stem(rel: str) -> str:
    return Path(rel).stem


def scenes(exp: Path, out: Path, dataset: str | None = None) -> None:
    jobs = json.loads((exp / "jobs.json").read_text(encoding="utf-8"))
    seen = {}
    for j in jobs:
        seen.setdefault((j["env"], j["scene"]), j)
    if dataset:  # every scene of the dataset, not only those the experiment ran
        from mamo.bench.datasets import list_datasets, load_labels
        ds = list_datasets(runner.DEFAULT_WORKSPACE)[dataset]
        for lab in load_labels(runner.DEFAULT_WORKSPACE, ds):
            if lab["difficulty"] in ("easy", "medium", "hard") and ds.path(lab["scene"]).exists():
                seen.setdefault((lab["env"], lab["scene"]), {"env": lab["env"], "scene": lab["scene"],
                                                             "difficulty": lab["difficulty"],
                                                             "scene_path": str(ds.path(lab["scene"]))})
    index_f = out / "scenes" / "index.json"
    index = [i for i in (json.loads(index_f.read_text(encoding="utf-8")) if index_f.exists() else [])
             if "error" not in i]
    have = {(i["env"], i["scene"]) for i in index}
    seen = {k: v for k, v in seen.items() if k not in have}
    for (env, rel), j in sorted(seen.items()):
        d = out / "scenes" / env
        gif = d / f"{_stem(rel)}.gif"
        info = {"env": env, "scene": rel, "difficulty": j["difficulty"], "gif": gif.relative_to(out).as_posix()}
        try:
            if not gif.exists():
                r = media.scene_gif(Path(j["scene_path"]), gif, title=f"{env} - {j['difficulty']} - {_stem(rel)}")
                info.update(r)
            p2, p3 = media.scene_views(Path(j["scene_path"]), d / _stem(rel))
            info.update(png2d=p2.relative_to(out).as_posix(), png3d=p3.relative_to(out).as_posix())
        except Exception:
            info["error"] = traceback.format_exc()
        index.append(info)
        print(f"scene {env} {rel}: {'ok' if 'error' not in info else 'ERROR'}", flush=True)
        (out / "scenes" / "index.json").write_text(json.dumps(index, indent=1), encoding="utf-8")


def solutions(exp: Path, out: Path, watch: bool) -> None:
    done: set[str] = set()
    index_f = out / "solutions" / "index.json"
    index = json.loads(index_f.read_text(encoding="utf-8")) if index_f.exists() else []
    done |= {i["id"] for i in index}
    while True:
        rows = runner.read_results(exp)
        todo = [r for r in rows if r.get("solved") and r.get("solution_file") and r["id"] not in done]
        for r in todo:
            gif = out / "solutions" / (r.get("env") or "env") / f"{_stem(r['scene'])}__{r['method']}.gif"
            info = {"id": r["id"], "env": r.get("env"), "scene": r["scene"], "difficulty": r["difficulty"],
                    "method": r["method"], "planning_time": r.get("planning_time"), "executed": r.get("executed"),
                    "gif": gif.relative_to(out).as_posix()}
            try:
                media.video(exp / "solutions" / r["solution_file"], gif, size=(240, 320), every=2,
                            info=f"{_stem(r['scene'])} ({r['difficulty']}, {r.get('planning_time', 0):.0f} s)")
            except Exception:
                info["error"] = traceback.format_exc()
            done.add(r["id"])
            index.append(info)
            index_f.parent.mkdir(parents=True, exist_ok=True)
            index_f.write_text(json.dumps(index, indent=1), encoding="utf-8")
            print(f"solution {r['id']}: {'ok' if 'error' not in info else 'ERROR'}", flush=True)
        state = runner.read_status(exp).get("state")
        if not watch or (state not in ("running", None) and not todo):
            break
        time.sleep(60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["scenes", "solutions"])
    ap.add_argument("experiment")
    ap.add_argument("--out", default="results/overnight")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--dataset", default=None, help="scenes: also every scene of this dataset")
    a = ap.parse_args()
    exp, out = Path(a.experiment), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.what == "scenes":
        scenes(exp, out, a.dataset)
    else:
        solutions(exp, out, a.watch)
