"""P7b: two-wave collection of the frozen P7 plan under the unchanged 2D-ABC-OPNEG-v1 policy.

Runs only after the P7a gate passed. All 48 root definitions (seeds, families, splits, strata, batches) are
predeclared in p7_scenes.ROOTS; construction, catalogues and static admission are frozen before the first
teacher call. Per root: reference first (a non-VERIFIED reference stops the lineage), main, catalogue
order, edits; statically certified queries never call the teacher; per-query cap = min(870, root 1,000,
batch 5,000, aggregate 20,000 remainders) so no cap can be exceeded. Wave 2 (batches 3-4) runs only if the
predeclared wave-1 gate (development roots only; test roots sealed) passes. Progress is persisted after
every query; an interrupted query is never repeated.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import config as cf
import d3lite as d3
import d3lite_run as d3r
import g4c
import g4d_gate as gg
import g4d_run as gr
import geometry as ge
import operational_teacher as ot
import p6_run as p6r
import p7_scenes as p7
import sector_cert
import teacher_pilot as tp
from scene import scene_from_dict

CAPS = {"teacher_cpu_s": 20000.0, "static_cpu_s": 1200.0, "root_cpu_s": 1000.0, "batch_cpu_s": 5000.0, "wall_s": 48000.0,
        "query_cpu_s": 870.0, "query_wall_s": 1800.0, "max_queries": 1536, "max_seed_attempts": 4608, "max_queries_per_root": 32,
        "unknown_fraction_halt": 0.20, "workers": 1}
FROZEN = p6r.FROZEN + ("runs/p6/collect",)
GATE = ("wave-1 gate on development (train + validation) roots of batches 1-2 only: (1) no FATAL / integrity violation and "
        "UNKNOWN < 20% of attempted development queries; (2) >= 2 decided kappa = +1, >= 2 kappa = -1 and >= 2 kappa = 0 roots; "
        "(3) >= 4 decided same-endpoint order contrasts over >= 2 roots and >= 2 VERIFIED repairs of length >= 3 over >= 2 roots; "
        "(4) >= 3 roots with two differently-sized VERIFIED repairs (else MINIMALITY_EVIDENCE_MISSING, STOP); (5) >= 2 roots whose "
        "nontrivial failure is decided by a LINK0_SECTOR_PICK (motion-level) certificate and that hold >= 1 VERIFIED repair; "
        "(6) >= 8 joint-pool-ready roots over >= 3 families; (7) budgets held.")


def gate(plan, records, used) -> dict:
    dev = {q: r for q, r in records.items() if r["query"]["split"] in ("train", "validation") and r["query"]["batch"] <= 2}
    att = [r for r in dev.values() if r["teacher_status"] != "NOT_RUN"]
    unk = sum(r["teacher_status"] == ot.UNKNOWN for r in att)
    droots = {k: v for k, v in plan["roots"].items() if v["queries"] and v["split"] in ("train", "validation") and v["batch"] <= 2}
    kap = p6r.kappas({"roots": droots}, dev)
    eff = ot.effects(dev, droots)
    swaps = [s for s in eff["swap"] if s["decided_contrast"]]
    longv = [r for r in dev.values() if r["teacher_status"] == ot.VERIFIED and len(r["query"]["sequence"]) >= 3]
    sizes = {}
    for r in dev.values():
        if r["teacher_status"] == ot.VERIFIED and r["query"]["kind"] == "ORDERED_TEACHER" and r["query"]["sequence"]:
            sizes.setdefault(r["query"]["lineage"], set()).add(len(r["query"]["sequence"]))
    motion = {r["query"]["lineage"] for r in dev.values() if (r.get("static_certificate") or {}).get("kind") == "LINK0_SECTOR_PICK"}
    motion = [k for k in motion if any(x["teacher_status"] == ot.VERIFIED and x["query"]["sequence"] and x["query"]["lineage"] == k for x in dev.values())]
    pools = d3r.pools({"roots": droots}, dev)
    ready = [k for k, v in pools.items() if v["joint_pool_ready"]]
    c = {"fatal": any(r["teacher_status"] == "FATAL_CONTRADICTION" for r in records.values()), "attempted_dev": len(att), "unknown_dev": unk,
         "unknown_fraction": round(unk / max(1, len(att)), 3),
         "kappa_counts": {s: sum(x["kappa"] == s for x in kap) for s in (1, 0, -1)}, "kappa_operational": sum(x["operational"] for x in kap),
         "order_contrasts": len(swaps), "order_roots": len({s["orders"][0].split(":")[1] for s in swaps}),
         "long_verified": len(longv), "long_roots": len({r["query"]["lineage"] for r in longv}),
         "multi_size_roots": sorted(k for k, v in sizes.items() if len(v) >= 2), "motion_roots": sorted(motion),
         "joint_pool_ready_roots": len(ready), "joint_pool_families": len({plan["roots"][k]["family"] for k in ready}),
         "budget_held": used <= CAPS["teacher_cpu_s"]}
    checks = {"1_integrity": not c["fatal"] and c["unknown_fraction"] < CAPS["unknown_fraction_halt"],
              "2_kappa_signs": all(c["kappa_counts"][s] >= 2 for s in (1, 0, -1)),
              "3_order_and_long": c["order_contrasts"] >= 4 and c["order_roots"] >= 2 and c["long_verified"] >= 2 and c["long_roots"] >= 2,
              "4_minimality_evidence": len(c["multi_size_roots"]) >= 3, "5_motion": len(c["motion_roots"]) >= 2,
              "6_pools": c["joint_pool_ready_roots"] >= 8 and c["joint_pool_families"] >= 3, "7_budget": c["budget_held"]}
    return {"rule": GATE, **c, "checks": checks, "pass": all(checks.values())}


def run(args) -> int:
    out, t0, w0 = Path(args.out), time.process_time(), time.time()
    if (out / "summary.json").exists():
        raise SystemExit("P7b collection already finished; refusing to overwrite")
    if not p6r._rd(args.p7a_gate).get("pass") or not p6r._rd(Path(args.regression, "regression_result.json")).get("passed"):
        raise SystemExit("P7a gate or regression not passed: no P7b collection")
    cfg, v2, pol = cf.load_config(args.config), cf.load_config_v2(args.config_v2), ot.load_policy(args.policy)
    if v2["proposal"]["move_cap"] != 4 or ot.config_mismatch(pol, cfg):
        raise SystemExit("P7b requires the R4 overlay and the unchanged OPNEG-v1 policy/config")
    robot = ge.make_robot(cfg)
    out.mkdir(parents=True, exist_ok=True)
    frozen = {r: gr._sha(Path(r) / "manifest.json") for r in FROZEN}
    calls, ops = sector_cert.guard_and_count()
    if (out / "plan.json").exists():
        plan, static, cons = p6r._rd(out / "plan.json"), p6r._rd(out / "static_admission.json")["certificates"], p6r._rd(out / "construction.json")
    else:
        known = d3._known(Path("."))
        for f in ("runs/p5/d3lite/plan.json", "runs/p6/collect/plan.json"):
            for r in p6r._rd(f)["roots"].values():
                known.add(g4c.stable_geometry(scene_from_dict(r["scene"]), r["goal_cell"]))
        roots, ledger, cons = p7.construct(cfg, robot, known)
        plan = p7.plan(cfg, robot, pol, roots)
        per_root = max(sum(q["lineage"] == r["root"] for q in plan["queries"]) for r in roots)
        if len(plan["queries"]) > CAPS["max_queries"] or len(plan["units"]) > CAPS["max_seed_attempts"] or per_root > CAPS["max_queries_per_root"]:
            raise SystemExit("frozen plan exceeds the authorized query / seed-attempt caps")
        with open(out / "construction_attempts.jsonl", "w") as fh:
            fh.writelines(json.dumps(x) + "\n" for x in ledger)
        cf.write_json(out / "construction.json", cons)
        cf.write_json(out / "plan.json", {**plan, "caps": CAPS, "gate": GATE})
        c1 = time.process_time()
        static = tp.static_admission(cfg, robot, plan, use_sector=True)
        cf.write_json(out / "static_admission.json", {"certificates": static, "cpu_s": round(time.process_time() - c1, 2), "ops": dict(ops)})
    static_cpu = cons["cpu_s"] + p6r._rd(out / "static_admission.json")["cpu_s"]
    if static_cpu > CAPS["static_cpu_s"]:
        raise SystemExit("static construction + admission exceeded 1,200 CPU-s")
    paths = {"plan": out / "plan.json", "config": Path(args.config), "config_v2": Path(args.config_v2)}
    prog = p6r._rd(out / "progress.json") if (out / "progress.json").exists() else {"done": {}, "status": {}, "wall_before_s": 0.0}
    wall0 = prog["wall_before_s"]
    used = lambda: sum(d["unit_cpu_s"] + d["coord_cpu_s"] for d in prog["done"].values())  # noqa: E731
    wall = lambda: wall0 + time.time() - w0  # noqa: E731
    built = {r["lineage"] for r in plan["roots"].values()}
    for b in (1, 2, 3, 4):
        if b == 3:
            if "wave1" not in prog["status"]:
                recs = {q["query"]: p6r.judge(static, pol, q, prog["done"].get(q["query"])) for q in plan["queries"]}
                prog["status"]["wave1"] = gate(plan, recs, used()) | {"cpu_used": round(used(), 2), "wall": round(wall(), 1)}
                cf.write_json(out / "progress.json", prog)
            if not prog["status"]["wave1"]["pass"]:
                break
        b0 = sum(d["unit_cpu_s"] + d["coord_cpu_s"] for d in prog["done"].values() if d["batch"] < b)
        for spec in [s for s in p7.ROOTS if s["batch"] == b and f"p7_{s['seed']}" in built]:
            lin = f"p7_{spec['seed']}"
            for q in p6r.order(plan, lin):
                qid = q["query"]
                if qid in prog["done"]:
                    continue
                mine = [d for d in prog["done"].values() if d["lineage"] == lin]
                stop = None
                if any(d["role"] == "reference" and d["status"] != ot.VERIFIED for d in mine):
                    stop = "lineage stopped: reference scene has no verified base witness"
                elif sum(d["censored_twice"] for d in mine) >= 2:
                    stop = "root censoring stop: two queries with two consecutive CPU-censored seeds"
                c1 = time.process_time()
                cap = min(CAPS["query_cpu_s"], CAPS["root_cpu_s"] - sum(d["unit_cpu_s"] + d["coord_cpu_s"] for d in mine),
                          CAPS["batch_cpu_s"] - (used() - b0), CAPS["teacher_cpu_s"] - used())
                seeds, uc, qdir = {qid: []}, 0.0, out / "q" / qid.replace(":", "__").replace(">", "-")
                if stop is None and static.get(qid) is None and qdir.exists():
                    stop = "interrupted before this query was recorded; never repeated"
                if stop is None and static.get(qid) is None and cap < pol["policy"]["per_seed_cpu_s"] + pol["policy"]["per_seed_guard_margin_s"] + \
                        pol["pilot"]["orchestration_reserve_s"]:
                    stop = f"budget remainder {cap:.0f} CPU-s below one guarded seed attempt"
                if stop is None and static.get(qid) is None:
                    seeds, uc = tp.run_rounds(plan, pol, qdir, paths, cfg, v2, robot, static, [qid], cap,
                                              min(CAPS["query_wall_s"], CAPS["wall_s"] - wall()), seeds)
                prog["done"][qid] = {"lineage": lin, "batch": b, "role": q["role"], "stop": stop, "attempts": seeds[qid], "unit_cpu_s": uc,
                                     "coord_cpu_s": time.process_time() - c1, "censored_twice": p6r.censored_twice(seeds[qid]), "status": None}
                prog["done"][qid]["status"] = p6r.judge(static, pol, q, prog["done"][qid])["teacher_status"]
                prog["wall_before_s"] = wall0 + time.time() - w0
                cf.write_json(out / "progress.json", prog)
    records = {q["query"]: p6r.judge(static, pol, q, prog["done"].get(q["query"])) for q in plan["queries"]}
    if "wave1" not in prog["status"]:
        prog["status"]["wave1"] = gate(plan, records, used())
    return report(out, cfg, plan, pol, records, prog, static_cpu, calls, ops, t0, used(), wall(), frozen)


def report(out, cfg, plan, pol, records, prog, static_cpu, calls, ops, t0, used, wall, frozen) -> int:
    sealed = {q: r for q, r in records.items() if r["query"]["split"] == "test"}
    dev = {q: r for q, r in records.items() if q not in sealed}
    cf.write_json(out / "decisions.json", dev)
    cf.write_json(out / "sealed_test.json", sealed)
    att = [{**a, "query": q} for q, r in records.items() for a in r["attempts"]]
    with open(out / "attempts.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in att)
    droots = {k: v for k, v in plan["roots"].items() if v["queries"] and v["split"] != "test"}
    cf.write_json(out / "effects.json", {**ot.effects(dev, droots), "kappa_by_kind": p6r.kappas({"roots": droots}, dev),
                                         "dependencies": p6r.dependencies(dev)})
    cf.write_json(out / "pools.json", d3r.pools({"roots": droots}, dev))
    cf.write_json(out / "loss_masks.json", ot.loss_masks({q: r for q, r in records.items() if r["teacher_status"] != "NOT_RUN"}))
    st = (ot.VERIFIED, ot.CERTIFIED, ot.BUDGET, ot.UNKNOWN, "NOT_RUN", "FATAL_CONTRADICTION")
    raw = lambda rs: {s: sum(r["teacher_status"] == s for r in rs.values()) for s in st}  # noqa: E731
    by_root = {}
    for d in prog["done"].values():
        x = by_root.setdefault(d["lineage"], {"cpu_s": 0.0, "teacher_queries": 0, "censored_twice": 0})
        x["cpu_s"] = round(x["cpu_s"] + d["unit_cpu_s"] + d["coord_cpu_s"], 2)
        x["teacher_queries"] += bool(d["attempts"])
        x["censored_twice"] += d["censored_twice"]
    comp = {"static_cpu_s": round(static_cpu, 2), "teacher_aggregate_cpu_s": round(used, 2), "wall_s": round(wall, 1), "caps": CAPS,
            "seed_attempts": len(att), "seed_stops": {s: sum(a["stop"] == s for a in att) for s in {a["stop"] for a in att}}, "by_root": by_root,
            "coordinator_guarded_calls": calls, "coordinator_static_ops": ops}
    summ = {"raw_all": raw(records), "raw_dev": raw(dev), "raw_sealed_test_counts_only": raw(sealed), "gates": prog["status"], "compute": comp,
            "frozen_manifests_unchanged": {r: gr._sha(Path(r) / "manifest.json") == h for r, h in frozen.items()}}
    cf.write_json(out / "summary.json", summ)
    gg.write_manifest(out, {"root_seeds": [s["seed"] for s in p7.ROOTS], "policy_seeds": pol["policy"]["seeds"]}, cf.config_hash(cfg),
                      {"plan_hash": plan["plan_hash"], "policy_hash": pol["policy_hash"]})
    print(json.dumps({k: summ[k] for k in ("raw_all", "raw_dev")} | {"wave1": prog["status"]["wave1"].get("pass"), "cpu": round(used, 1)}))
    return 1 if any(r["teacher_status"] == "FATAL_CONTRADICTION" for r in records.values()) else 0
