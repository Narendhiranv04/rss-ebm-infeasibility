"""Table I(a)-style metrics for the baselines, defined exactly as approach_b_final/evaluate.py summarize():

  Task %    REPAIR with verified X = 1 and Y = 1, over eval_task roots (K1-K3)
  Repair %  Task success with exactly K relocations, over eval_repair roots   (+ count of successes shorter than K)
  No-op %   status NOOP on eval_noop roots (K0)            Abstain %  status ABSTAIN on eval_abstain roots (NONE)
  Travel    mean witnessed joint travel of successful repairs (cost.jrec)
plus what the analytical planners spend: motion-planner calls and CPU per run. A budget cut T reads the same runs at a shorter budget: a
plan found after T CPU-s counts as ABSTAIN. CENSORED / ERROR runs count as ABSTAIN and are reported.

    python -m b2d.table --tag main [--cuts 300,60]"""
from __future__ import annotations

import argparse
import collections
import json

import numpy as np

from . import dataset as D
from . import paths as P

NAMES = {"em4m": "E-M4M", "msmcts": "MS-MCTS", "mosegman": "MO-SeGMan"}


def load(tags):
    rows = {}
    for t in tags:
        for l in open(P.OUT / "runs" / t / "results.jsonl"):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            rows[(r["root"], r["method"], r["seed"])] = r
    return list(rows.values())


def outcome(r, cut):
    st = r["status"] if r["status"] in ("NOOP", "REPAIR", "ABSTAIN") else "ABSTAIN"
    if st != "ABSTAIN" and r["cpu_s"] > cut:
        st = "ABSTAIN"
    v = r.get("verify") or {}
    ok = st == "REPAIR" and v.get("x") == 1 and v.get("y") == 1
    return st, ok


def summarize(rows, cut):
    M = D.masks()
    out = {}
    for meth in sorted({r["method"] for r in rows}):
        rs = [r for r in rows if r["method"] == meth]
        g = collections.defaultdict(list)
        for r in rs:
            st, ok = outcome(r, cut)
            mk, K = M[r["root"]], M[r["root"]]["K"]
            m = len(r["seq"]) if st == "REPAIR" else None
            if mk["eval_task"]:
                g["task"].append(ok)
                g[f"task_K{K}"].append(ok)
                if ok and r["verify"].get("travel") is not None:
                    g["travel"].append(r["verify"]["travel"])
                if ok:
                    g["cpu_success"].append(r["cpu_s"])
            if mk["eval_repair"]:
                g["repair"].append(ok and m == int(K))
                g["shorter"].append(ok and m < int(K))
            if mk["eval_noop"]:
                g["noop"].append(st == "NOOP")
            if mk["eval_abstain"]:
                g["abstain"].append(st == "ABSTAIN")
            g["calls"].append(sum(1 for e in r["trace"] if e[1] < cut))       # calls started within the budget cut
            g["cpu"].append(min(r["cpu_s"], cut))
            g["unverified"].append(st == "REPAIR" and (r.get("verify") or {}).get("x") is None)
            g["failed_verify"].append(st == "REPAIR" and not ok)
            g["censored_or_error"].append(r["status"] in ("CENSORED", "ERROR"))
        mean = lambda k: float(np.mean(g[k])) if g[k] else None  # noqa: E731
        out[meth] = {"task_success": mean("task"), "n_task": len(g["task"]), "repair_success": mean("repair"), "n_repair": len(g["repair"]),
                     "shorter_than_K": int(sum(g["shorter"])), "noop_correct": mean("noop"), "n_k0": len(g["noop"]),
                     "abstain_correct": mean("abstain"), "n_none": len(g["abstain"]),
                     "travel_success_mean": mean("travel"), "calls_mean": mean("calls"), "cpu_mean_s": mean("cpu"),
                     "cpu_success_median_s": float(np.median(g["cpu_success"])) if g["cpu_success"] else None,
                     "repair_failed_verification": int(sum(g["failed_verify"])), "unverified": int(sum(g["unverified"])),
                     "censored_or_error": int(sum(g["censored_or_error"])),
                     "task_by_K": {k[-1]: mean(k) for k in ("task_K1", "task_K2", "task_K3")},
                     "seeds": sorted({r["seed"] for r in rs}), "roots": len({r["root"] for r in rs})}
    return out


SPREAD = ("task_success", "repair_success", "noop_correct", "abstain_correct", "calls_mean", "travel_success_mean", "cpu_mean_s")


def per_seed(rows, cut, res):
    """Adds <metric>_std over seeds (population std of the per-seed values, as for Approach B's 3 seeds) and the per-seed tables."""
    seeds = sorted({r["seed"] for r in rows})
    by = {s: summarize([r for r in rows if r["seed"] == s], cut) for s in seeds}
    for meth, s in res.items():
        for k in SPREAD:
            v = [by[sd][meth][k] for sd in seeds if meth in by[sd] and by[sd][meth][k] is not None]
            s[k + "_std"] = float(np.std(v)) if len(v) > 1 else None
        s["per_seed"] = {str(sd): {k: by[sd][meth][k] for k in SPREAD} for sd in seeds if meth in by[sd]}
    return res


def pct(x, sd=None):
    return "—" if x is None else f"{100 * x:.1f}" + ("" if sd is None else f" ± {100 * sd:.1f}")


def num(x, sd=None, f=".1f"):
    return "—" if x is None else f"{x:{f}}" + ("" if sd is None else f" ± {sd:{f}}")


def markdown(res, cut):
    lines = [f"### Budget {cut:.0f} CPU-s (mean ± std over seeds)", "",
             "| Method | Task % ↑ | Repair % ↑ | No-op % ↑ | Abstain % ↑ | Calls ↓ | Travel ↓ | Task K1 / K2 / K3 % | CPU/run (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for meth, s in res.items():
        tk, sd = s["task_by_K"], lambda k: s.get(k + "_std")  # noqa: E731
        lines.append(f"| {NAMES.get(meth, meth)} | {pct(s['task_success'], sd('task_success'))} | "
                     f"{pct(s['repair_success'], sd('repair_success'))} | {pct(s['noop_correct'], sd('noop_correct'))} | "
                     f"{pct(s['abstain_correct'], sd('abstain_correct'))} | {num(s['calls_mean'], sd('calls_mean'))} | "
                     f"{num(s['travel_success_mean'], sd('travel_success_mean'), '.2f')} | "
                     f"{pct(tk['1'])} / {pct(tk['2'])} / {pct(tk['3'])} | {num(s['cpu_mean_s'], sd('cpu_mean_s'), '.0f')} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    ap.add_argument("--cuts", default="300,60")
    a = ap.parse_args()
    tags = a.tag.split(",")
    rows = load(tags)
    res = {}
    md = []
    for c in (float(x) for x in a.cuts.split(",")):
        res[str(int(c))] = per_seed(rows, c, summarize(rows, c))
        md.append(markdown(res[str(int(c))], c))
    out = P.OUT / "runs" / tags[0]
    (out / "table.json").write_text(json.dumps(res, indent=1))
    (out / "table.md").write_text("\n\n".join(md) + "\n")
    print("\n\n".join(md))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
