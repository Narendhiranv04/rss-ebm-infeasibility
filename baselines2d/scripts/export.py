r"""Export the finished runs to plain files in results/ (for the paper and for merging with Approach B's per-scene outputs).

    .\dk.ps1 python -m b2d.table --tag main --cuts 300,60      # first: out/runs/main/table.{md,json}
    .\dk.ps1 python scripts/export.py --tag main

results/table.md, table.json        Table I(a)-style metrics, mean ± std over seeds, budgets 300 s and 60 s (copied from out/runs/<tag>)
results/per_scene.csv               one row per (scene, method, seed): status, plan, X / Y verdict, travel, CPU, calls
results/per_seed_summary.csv        the table's metrics for every seed separately
results/flagged_scenes.csv          scenes where a baseline output disagrees with the dataset label (NONE solved, K0 not a no-op)"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2d import dataset as D, paths as P, table as T  # noqa: E402

RES = Path(__file__).resolve().parents[1] / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    run = P.OUT / "runs" / a.tag
    for f in ("table.md", "table.json"):
        shutil.copy(run / f, RES / f)
    rows = sorted(T.load([a.tag]), key=lambda r: (r["root"], r["method"], r["seed"]))
    M = D.masks()

    with open(RES / "per_scene.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["root", "K", "eval_task", "eval_repair", "eval_noop", "eval_abstain", "method", "seed", "status", "n_moves", "plan",
                    "x", "y", "task_success", "travel", "cpu_s", "n_calls", "capped_calls", "reason"])
        for r in rows:
            mk, v = M[r["root"]], r.get("verify") or {}
            _, ok = T.outcome(r, r["budget_s"])
            w.writerow([r["root"], r["K"], int(mk["eval_task"]), int(mk["eval_repair"]), int(mk["eval_noop"]), int(mk["eval_abstain"]),
                        T.NAMES[r["method"]], r["seed"], r["status"], len(r["seq"]), json.dumps(r["seq"]),
                        v.get("x", ""), v.get("y", ""), int(ok), "" if v.get("travel") is None else round(v["travel"], 4),
                        r["cpu_s"], r["n_calls"], r["capped"], r["reason"]])

    tab = json.loads((run / "table.json").read_text())
    with open(RES / "per_seed_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["budget_s", "method", "seed"] + list(T.SPREAD))
        for cut, res in tab.items():
            for meth, s in res.items():
                for sd, vals in s["per_seed"].items():
                    w.writerow([cut, T.NAMES[meth], sd] + [vals[k] for k in T.SPREAD])

    with open(RES / "flagged_scenes.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["root", "K", "issue", "method", "seeds", "plan", "x", "y"])
        flag = {}
        for r in rows:
            v = r.get("verify") or {}
            if r["K"] == "NONE" and r["status"] == "REPAIR" and v.get("x") == 1 and v.get("y") == 1:
                issue = "labelled NONE but a checker-verified repair exists"
            elif r["K"] == "0" and r["status"] != "NOOP":
                issue = "K0 scene not returned as no-op (target plan exceeds the 45 s per-call cap)"
            else:
                continue
            k = (r["root"], r["K"], issue, r["method"], json.dumps(r["seq"]), v.get("x", ""), v.get("y", ""))
            flag.setdefault(k, []).append(r["seed"])
        for (root, K, issue, meth, seq, x, y), seeds in sorted(flag.items()):
            w.writerow([root, K, issue, T.NAMES[meth], " ".join(map(str, sorted(seeds))), seq, x, y])
    print("wrote", ", ".join(sorted(p.name for p in RES.iterdir())))


if __name__ == "__main__":
    main()
