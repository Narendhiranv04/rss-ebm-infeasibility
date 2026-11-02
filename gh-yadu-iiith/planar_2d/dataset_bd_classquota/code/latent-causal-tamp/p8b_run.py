"""P8b worker: selection funnel, thresholds, sampler mixing and G+M baseline (0 teacher calls, one guard, 3,000 CPU-s).

Predeclared rules:
  models     seed-0 FULL_DECIDED checkpoint of every P8a arm that ran (chosen config), validation-only temperature
  threshold  per arm from {0.5, 0.7, 0.85, 0.95} by p8_select.choose_threshold on P7b VALIDATION roots
  sampler    retained for P8d only if, for every learned arm on all three development pools, the mean 4,000-step TV to the exact
             Boltzmann distribution is <= 0.10; otherwise the deterministic bounded-pool fallback is used in P8d
  G+M        offline ranking for every root; plus a protocol smoke run through ordered.primary_v2 on two validation roots with
             REAL relocation planning and a stub target evaluator (no target is executed in P8b)
"""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import abc7_train as t7
import abc8
import config as cf
import data
import geometry as ge
import ordered as od
import p7_eval as ev
import p7_oneshot as po
import p8_select as ps
import sequences as sq
import teacher
from scene import scene_from_dict

CAPS = {"cpu_s": 3000.0, "wall_s": 6000.0}
C7 = Path("runs/p7/collection")
ARMS = ["A", "B-full", "B-structured", "B-none", "SF", "C-full"]


def run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", "import sys, p8b_run; p8b_run.worker(sys.argv[1])", str(out)], CAPS["cpu_s"], CAPS["wall_s"],
                      out / "heartbeat.txt", 30.0, 0.5, Path("."))
    cf.write_json(out / "guard.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"caps": CAPS})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s", "overshoot_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def _model(arm, res, adir):
    tag = next((t for t, v in res["runs"].items() if v.get("status") == "RUN" and v["arm"] == arm and v["regime"] == "FULL_DECIDED" and v["seed"] == 0
                and v["config"] == res["chosen"].get(arm, res["chosen"]["B-full"])), None)
    if tag is None:
        return None, None
    m = abc8.build(arm)
    m.load_state_dict(torch.load(adir / f"ckpt_{tag.replace('|', '_')}.pt", weights_only=True))
    return m.eval(), tag


def _stub_target(st, key):
    return {"status": "NOT_RUN", "reason": "P8B_STUB_NO_TARGET_EXECUTION", "result": None, "start": None}


def gm_smoke(cfg, robot, plan, keys) -> list:
    out = []
    for k in keys:
        r = plan["roots"][k]
        s = scene_from_dict(r["scene"])
        q = data.make_query(cfg, f"p8b_gm_{k}", s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
        ranked = [{"label": f"r{i}", "sequence": [tuple(a) for a in x], "assignment": sq.assignment([tuple(a) for a in x])}
                  for i, x in enumerate(ps.gm_rank(cfg, robot, plan, k))]
        c0 = time.process_time()
        ep = od.primary_v2(robot, cfg, q, s, ranked, _stub_target, cf.robot_profile_hash(cfg))
        out.append({"root": k, "ranked": [x["sequence"] for x in ranked], "outcome": ep["outcome"], "committed": ep["commit"] and ep["commit"]["order"],
                    "relocation_plans": sum((t.get("plan") or {}).get("relocation_plans", 0) for t in ep["tried"]), "cpu_s": round(time.process_time() - c0, 1),
                    "target_calls": len(ep["target_invocations"]), "audit": ep["audit"]["decision"]})
    return out


def worker(out) -> None:
    torch.set_num_threads(1)
    out, t0 = Path(out), time.process_time()
    log = lambda msg: open(out / "log.txt", "a").write(f"{time.process_time() - t0:8.1f} {msg}\n")  # noqa: E731
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    res = json.loads(Path("runs/p8/a/results.json").read_text())
    plan = json.loads((C7 / "plan.json").read_text())
    records = json.loads((C7 / "decisions.json").read_text()) | json.loads((C7 / "sealed_test.json").read_text())
    ds = torch.load("runs/p7/training/dataset.pt", weights_only=False)
    dsl = torch.load("runs/p7/training/dataset_sealed.pt", weights_only=False)
    full, legacy = t7.prepare(ds), t7.prepare(dsl)
    val = [r for r in full if r["split"] == "validation"]
    recs = {x["audit"]["root"]: x for x in ds["roots"] + dsl["roots"]}
    out_d, rows_all = {"arms": {}}, []
    for arm in ARMS:
        m, tag = _model(arm, res, Path("runs/p8/a"))
        if m is None:
            out_d["arms"][arm] = {"status": "NOT_RUN (no P8a checkpoint)"}
            continue
        ve, le = [ev.evaluate(m, r) for r in val], [ev.evaluate(m, r) for r in legacy]
        tau = ev.fit_temperature(ve)
        th = ps.choose_threshold(ve, records, C7, tau)
        d = {"tag": tag, "tau": tau, "threshold": th}
        for name, e in (("validation", ve), ("legacy_p7_test", le)):
            for thr in sorted({0.5, th["chosen"]}):
                f = ps.funnel(e, records, C7, tau, thr)
                d[f"{name}@{thr}"] = {k: sum(x[k] for x in f) for k in ("verified_candidate_exists", "abstain", "success", "false_feasible_empty",
                                                                          "shorter_infeasible_preempted")} | {
                    "roots": len(f), "relocation_witness_but_target_failed": sum(bool(x["chosen_relocation_witness"]) and not x["success"] for x in f),
                    "minimal_successes": sum(x["success"] and x["chosen_moves"] == x["min_verified_moves"] for x in f),
                    "first_verified_rank_mean": float(np.mean([x["first_verified_rank"] for x in f if x["first_verified_rank"] is not None]))}
                rows_all += [{"arm": arm, "split": name, "threshold": thr, **x} for x in f]
        out_d["arms"][arm] = d
        log(f"{arm}: tau {tau:.2f}, threshold {th['chosen']}")
        if arm in ("A", "B-full", "SF"):
            pools = sorted([r for r in val if not r["key"].endswith("_ref") and len(r["cands"]) > 4], key=lambda r: len(r["cands"]))
            pick = [pools[0], pools[len(pools) // 2], max(pools, key=lambda r: max(len(c["seq"]) for c in r["cands"]))]
            d["mixing"] = {r["key"]: ps.mixing(m, r, recs[r["key"]]) for r in pick}
            log(f"{arm}: mixing {[round(v['chain_4000']['tv_mean'], 3) for v in d['mixing'].values()]}")
    for name, roots in (("validation", [r["key"] for r in val]), ("legacy_p7_test", [r["key"] for r in legacy])):
        for base in ("U", "G+M"):
            f = []
            for i, k in enumerate(x for x in roots if not x.endswith("_ref")):
                pool = [p["sequence"] for p in plan["roots"][k]["queries"]["pool"]]
                rk = po.uniform_ranking([s for s in pool], 20261022 + i) if base == "U" else ps.gm_rank(cfg, robot, plan, k)
                q = f"seq:{k}:{ps.z4.sig([tuple(a) for a in rk[0]])}" if rk else None
                rec = records.get(q)
                f.append({"root": k, "chosen": q, "success": bool(rec and rec["teacher_status"] == "VERIFIED_FEASIBLE"), "chosen_moves": rk and len(rk[0])})
            out_d.setdefault("baselines", {}).setdefault(base, {})[name] = {"roots": len(f), "success": sum(x["success"] for x in f)}
            rows_all += [{"arm": base, "split": name, "threshold": None, **x} for x in f]
    mix = [v["chain_4000"]["tv_mean"] for a in ("A", "B-full", "SF") for v in out_d["arms"].get(a, {}).get("mixing", {}).values()]
    out_d["sampler_decision"] = {"max_tv_4000": max(mix) if mix else None, "rule": "retain sampler iff every mean TV(4k) <= 0.10",
                                 "use": "SAMPLER" if mix and max(mix) <= 0.10 else "DETERMINISTIC_BOUNDED_POOL"}
    out_d["gm_smoke"] = gm_smoke(cfg, robot, plan, [r["key"] for r in val if not r["key"].endswith("_ref")][:2])
    with open(out / "decision_funnel.csv", "w", newline="") as fh:
        keys = sorted({k for x in rows_all for k in x})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows_all)
    out_d["cpu_s"] = round(time.process_time() - t0, 1)
    cf.write_json(out / "results.json", json.loads(json.dumps(out_d, default=float)))
    log("done")

