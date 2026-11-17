#!/usr/bin/env python3
"""REAP's row of table (a) "3D baseline comparison" from an eval + replay run on the 6x6 shelf test split.

    python metrics66.py <replay csv> [--exclude-classes 4] [--json out.json]

Columns (per test scene; all from the same run):
    AUC     ROC-AUC of REAP's start-state value V(s0) = max_a Q(s0, a) for the ground truth "solvable" (K0-K3) vs NONE
            (higher V = REAP expects fewer remaining actions; a dead end tends to -50)
    Repair  % of K0-K3 scenes executed in our env whose obstacle moves = verified K (minimal); also mean excess moves
    Task    % of scenes with the verified outcome: K0-K3 -> the plan executes in our env (target at goal, nothing
            displaced); NONE -> no plan returned
    Time    REAP-GBFS wall time per scene, s (search + feasibility checks), mean and median
    Calls   distinct feasibility queries per scene (MP queries: a put = the teacher's full move plan, a pick = an
            approach check), mean and median
"""
import argparse
import json

import numpy as np
import pandas as pd


def roc_auc(y, s):
    y, s = np.asarray(y, bool), np.asarray(s, float)
    pos, neg = s[y], s[~y]
    if not len(pos) or not len(neg):
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    vals = np.concatenate([pos, neg])[order]
    i = 0
    while i < len(vals):                                   # average ranks for ties
        j = i
        while j + 1 < len(vals) and vals[j + 1] == vals[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def row(d: pd.DataFrame) -> dict:
    d = d.copy()
    d["executed"] = d.get("executed", False).fillna(False).astype(bool) if "executed" in d else False
    solvable = d.cls != "NONE"
    task_ok = np.where(solvable, d.executed, ~d.reap_solved.astype(bool))
    ex = d[solvable & d.executed]
    minimal = (ex.reap_moves == ex.verified_K + 1)
    return {
        "n": int(len(d)),
        "AUC": roc_auc(solvable, d.v0) if "v0" in d and d.v0.notna().all() else float("nan"),
        "Repair_minimal_pct": 100 * float(minimal.mean()) if len(ex) else float("nan"),
        "Repair_excess_mean": float((ex.reap_moves - ex.verified_K - 1).mean()) if len(ex) else float("nan"),
        "Task_pct": 100 * float(task_ok.mean()),
        "Time_mean_s": float(d.wall_time.mean()), "Time_median_s": float(d.wall_time.median()),
        "Calls_mean": float(d.mp_calls.mean()), "Calls_median": float(d.mp_calls.median()),
        "reap_solved_pct": 100 * float(d.reap_solved.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--exclude-classes", default="", help="e.g. 4 (the ordered K4 roots)")
    ap.add_argument("--json")
    a = ap.parse_args()
    d = pd.read_csv(a.csv, dtype={"cls": str})
    if a.exclude_classes:
        d = d[~d.cls.isin(a.exclude_classes.split(","))]
    out = {"all": row(d), "per_class": {c: row(g) for c, g in d.groupby("cls")}}
    a_ = out["all"]
    print(f"REAP  AUC {a_['AUC']:.3f} | Repair {a_['Repair_minimal_pct']:.1f}% minimal (excess {a_['Repair_excess_mean']:.2f}) | "
          f"Task {a_['Task_pct']:.1f}% | Time {a_['Time_mean_s']:.1f} s (median {a_['Time_median_s']:.1f}) | "
          f"Calls {a_['Calls_mean']:.1f} (median {a_['Calls_median']:.0f})   n={a_['n']}")
    for c, r in out["per_class"].items():
        print(f"  {c:>4}: n={r['n']:3d} solved {r['reap_solved_pct']:5.1f}%  task {r['Task_pct']:5.1f}%  "
              f"minimal {r['Repair_minimal_pct']:5.1f}%  time {r['Time_mean_s']:6.1f}s  calls {r['Calls_mean']:6.1f}")
    if a.json:
        open(a.json, "w").write(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
