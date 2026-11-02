"""G4A bounded development audit (G4A instruction §5-9): run the frozen plan.

Per query: the primary one-shot episode with the automatic candidates (commit once,
one target evaluation, no feedback), then the first-pass OFFLINE_CASE_AUDIT of every
planned candidate. Diagnostics follow, separately accounted: the five-seed target
panel on bit-identical achieved states, the relocation-order panel (two seeds per
end state) and an old-scope (HOME_REQUIRED) comparison. Unknowns stay unknown and
stay in operational denominators; diagnostics may upgrade a label to positive but
never overwrite first-pass evidence.
"""
from __future__ import annotations

import csv
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import cases
import config as cf
import controls
import coverage
import data
import evaluate as ev
import geometry as ge
import proposals as pr
import roots
import verify as vf
from config import config_hash, robot_profile_hash, stable_hash
from scene import scene_from_dict

FIRST, ENRICHED = "first_pass", "diagnostic_enriched"


def _js(x):
    return json.loads(json.dumps(x))


def query_of(cfg: dict, q: dict):
    """Rebuild the query from its plan entry; identity must match the frozen hashes."""
    scene = scene_from_dict(q["scene"])
    query = data.make_query(cfg, q["query_id"], scene, "t0", q["goal_cell"],
                            {"root_id": q["root_id"], "stratum": q["stratum"], "construction_seed": q["seed"]}, "SAFE_RETREAT")
    if query["hashes"] != q["hashes"]:
        raise ValueError(f"{q['query_id']}: query identity differs from the plan")
    return scene, query


def check_plan(cfg: dict, plan: dict) -> None:
    """Integrity before any collection: plan hash, config hash, and a deterministic
    regeneration of every candidate list and primary ranking from learner fields."""
    if stable_hash({k: v for k, v in plan.items() if k != "plan_hash"}) != plan["plan_hash"]:
        raise ValueError("g4a_plan.json was modified after it was frozen")
    if plan["config_hash"] != config_hash(cfg):
        raise ValueError("config changed since the plan was frozen")
    for q in plan["queries"]:
        _, query = query_of(cfg, q)
        s0, act = data.initial_scene(query), query["learner"]["action"]
        gen = pr.generate_candidates(s0, act, cfg)
        if _js(gen) != q["generator"] or _js(pr.rank_proposals(s0, act, gen["candidates"], cfg)) != q["primary_ranking"]:
            raise ValueError(f"{q['query_id']}: candidates are not reproducible from the initial inputs")


def _cells(c: dict) -> dict:
    return {o: d if isinstance(d, int) else tuple(d) for o, d in c.items()}


def run_query(job) -> dict:
    """Pool worker: primary episode, then the first-pass teacher audit of every candidate."""
    cfg, q = job
    c0, w0 = time.process_time(), time.perf_counter()
    robot, rhash = ge.make_robot(cfg), robot_profile_hash(cfg)
    scene, query = query_of(cfg, q)
    act = query["learner"]["action"]

    def target(state, key):
        return ev.evaluate_target(robot, cfg, state, act, key)
    ranked = [{**c, "cells": _cells(c["cells"])} for c in q["primary_ranking"]]
    primary = ev.primary_episode(robot, cfg, query, scene, ranked, target, rhash)
    audit = [ev.audit_candidate(robot, cfg, query, scene, {**c, "cells": _cells(c["cells"])}, target,
                                f"{q['query_id']}/audit/{i}") for i, c in enumerate(q["audit_pool"])]
    return {"query_id": q["query_id"], "q": q, "scene": scene, "query": query, "primary": primary, "audit": audit,
            "cpu_s": round(time.process_time() - c0, 2), "wall_s": round(time.perf_counter() - w0, 2)}


def _costs(t: dict) -> dict:
    res = t.get("result")
    plans = res.plans if res is not None else []
    return {"status": t["status"], "reason": t["reason"], "seconds": t.get("seconds"), "nodes": sum(p.nodes for _, p in plans),
            "evals": sum(p.evals for _, p in plans), "queries": len(plans), "home": t.get("home_return_status"),
            "wall_clock_dependent": t.get("wall_clock_dependent"),
            "witness_hash": res is not None and res.traj is not None and stable_hash(_js({k: v.tolist() if hasattr(v, "tolist")
                                                                                            else v for k, v in res.traj.arrays().items()}))}


def seed_case(job) -> dict:
    """Five predetermined target seeds (keys) from one bit-identical achieved state, plus
    one old-scope HOME_REQUIRED evaluation of the same state (end-policy comparison)."""
    cfg, case, q, state = job
    c0 = time.process_time()
    robot = ge.make_robot(cfg)
    scene, query = query_of(cfg, q)
    act = query["learner"]["action"]
    runs = [_costs(ev.evaluate_target(robot, cfg, state, act, k)) for k in case["keys"]]
    old = data.action_context(cfg, scene, "t0", q["goal_cell"])          # G3.5 HOME-constrained scope
    home_required = _costs(ev.evaluate_target(robot, cfg, state, old, f"{case['keys'][0]}/home_required"))
    return {**case, "start_q": list(state.q), "start_gap": state.gap, "start_scene_hash": stable_hash(_js(
        [o.xy_m for o in state.scene.objects])), "runs": runs, "home_required": home_required,
            "cpu_s": round(time.process_time() - c0, 2)}


def order_case(job) -> dict:
    """Up to n executable orders of one assignment; two seeds per achieved end state."""
    cfg, case, q = job
    c0 = time.process_time()
    robot = ge.make_robot(cfg)
    scene, query = query_of(cfg, q)
    act, s0 = query["learner"]["action"], ev.start(scene, query)
    res = vf.executable_orders(robot, cfg, s0, _cells(case["cells"]), f"{case['keys'][0]}/orders", case["orders"])
    rows = []
    for att in res["attempts"]:
        row = {"order": [list(m) for m in att.order], "relocation_status": att.status, "reason": att.reason, "targets": []}
        if att.status == vf.RELOCATED:
            st = vf.final_state(s0, att)
            row.update(end_q=list(st.q), end_gap=st.gap, end_poses={o.id: list(o.xy_m) for o in st.scene.objects})
            row["targets"] = [_costs(ev.evaluate_target(robot, cfg, st, act, f"{k}/{'.'.join(f'{o}>{c}' for o, c in att.order)}"))
                              for k in case["keys"]]
        rows.append(row)
    return {**case, "orders_found": rows, "pruned_orders": res["pruned_orders"], "relocation_plans": res["relocation_plans"],
            "cpu_s": round(time.process_time() - c0, 2)}


def collect(cfg: dict, plan: dict, limit: int | None = None) -> dict:
    """Main audit within the CPU cap, then diagnostics within theirs (and the target-call cap)."""
    a = cfg["audit4a"]
    qs = plan["queries"][:limit] if limit else plan["queries"]
    main, spent, stopped = {}, 0.0, None
    with Pool(min(a["workers"], len(qs))) as pool:
        for r in pool.imap_unordered(run_query, [(cfg, q) for q in qs]):
            main[r["query_id"]] = r
            spent += r["cpu_s"]
            if spent > 3600 * a["main_cpu_hours"]:
                stopped = "MAIN_CPU_CAP"
                pool.terminate()
                break
    ids = set(main)
    byq = {q["query_id"]: q for q in qs}
    seed_jobs = []
    for case in plan["seed_panel"]:
        if case["query_id"] not in ids:
            continue
        r = main[case["query_id"]]
        row = next(x for x in r["audit"] if pr.signature(x["cells"]) == pr.signature(_cells(case["cells"])))
        att = next((o["attempt"] for o in row["orders"] if o["attempt"].status == vf.RELOCATED), None)
        if att is not None or not case["cells"]:
            state = vf.final_state(ev.start(r["scene"], r["query"]), att) if att else ev.start(r["scene"], r["query"])
            seed_jobs.append((cfg, case, byq[case["query_id"]], state))
    order_jobs = [(cfg, c, byq[c["query_id"]]) for c in plan["order_panel"] if c["query_id"] in ids]
    budget = a["max_diagnostic_target_calls"]
    calls = len(seed_jobs) * (a["seed_panel_seeds"] + 1)
    order_jobs = order_jobs[: max(0, (budget - calls) // (a["order_panel_orders"] * a["order_panel_seeds"]))]
    with Pool(a["workers"]) as pool:
        seeds = pool.map(seed_case, seed_jobs) if seed_jobs else []
        orders = pool.map(order_case, order_jobs) if order_jobs else []
    diag_cpu = sum(x["cpu_s"] for x in seeds + orders)
    return {"main": main, "seed_panel": seeds, "order_panel": orders, "main_cpu_s": round(spent, 1), "stopped": stopped,
            "diagnostic_cpu_s": round(diag_cpu, 1), "diagnostic_cap_exceeded": diag_cpu > 3600 * a["diagnostic_cpu_hours"],
            "missing_seed_cases": [c for c in plan["seed_panel"] if c["query_id"] in ids and not any(
                s["keys"] == c["keys"] for s in seeds)], "skipped_order_cases": len(plan["order_panel"]) - len(order_jobs)}


# ------------------------------------------------------------------ records and commands
def primary_record(r: dict) -> dict:
    p = r["primary"]
    t = p["target"]
    return {"query_id": r["query_id"], "stratum": r["q"]["stratum"], "hashes": r["query"]["hashes"], "outcome": p["outcome"],
            "commit": p["commit"] and {k: v for k, v in p["commit"].items() if k != "attempt"},
            "tried": [{"rank": x["rank"], "label": x["label"], "static": x["static"]["status"],
                       "stop": x["plan"] and x["plan"]["stop"]} for x in p["tried"]],
            "events": p["trace"].events, "boundary_ok": cases._boundary_ok(p["trace"].events),
            "target": t and {k: t.get(k) for k in ("status", "reason", "goal_achieved", "safe_retreat_status",
                                                     "home_return_status", "complete_target_status", "seconds",
                                                     "wall_clock_dependent", "start")},
            "cpu_s": r["cpu_s"], "wall_s": r["wall_s"]}


def cmd_g4a_plan(args) -> int:
    """Freeze roots, candidates, ranking, panels, budgets and targets before any outcome."""
    cfg = cf.load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    plan = roots.make_plan(cfg, ge.make_robot(cfg))
    cf.write_json(out / "g4a_plan.json", plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "roots": len(plan["queries"]), "attempts": len(plan["ledger"])}))
    return 0


def cmd_g4a_audit(args) -> int:
    import audit_figures as afg                                   # figures only after collection
    cfg = cf.load_config(args.config)
    out = Path(args.out)
    plan = json.loads((out / "g4a_plan.json").read_text())
    check_plan(cfg, plan)
    robot, rhash, t0 = ge.make_robot(cfg), robot_profile_hash(cfg), time.perf_counter()
    col = collect(cfg, plan, args.limit)
    suite, faults = controls.certificate_suite(robot, cfg), controls.fault_injection(robot, cfg)
    rows = coverage.label_table(col)
    leak = {q: cases.poison_check(r["query"], 64)["tensors_unchanged_after_poison_and_delete"] for q, r in col["main"].items()}
    prim = {q: primary_record(r) for q, r in sorted(col["main"].items())}
    protocol = {"plan_integrity_and_regeneration": True, "learner_tensors_unchanged_after_poisoning": all(leak.values()),
                "primary_target_boundary": all(p["boundary_ok"] for p in prim.values()),
                "certificate_suite_as_expected": all(c["as_expected"] for c in suite),
                "fault_injection_aborts": all((f["outcome"] == "ABORTED") != ("within" in f["case"]) and
                                              (f["outcome"] != "ABORTED" or (f["target_calls"] == 0 and f["relocations_executed"] == 1))
                                              for f in faults)}
    cov = coverage.coverage(plan, col, rows, protocol)
    tasks, scenes, arrays = cases.archive(col["main"], rhash)
    for name, obj in (("root_ledger.json", plan["ledger"]), ("candidate_labels.json", rows), ("primary_episodes.json", prim),
                      ("certificate_checks.json", {"suite": suite, "produced": [r for r in rows if r["certificates"]]}),
                      ("seed_panel.json", col["seed_panel"]), ("order_panel.json", col["order_panel"]),
                      ("fault_injection.json", faults), ("config.json", {"config_hash": cf.config_hash(cfg), "config": cfg}),
                      ("scenes.json", scenes)):
        cf.write_json(out / name, obj)
    np.savez_compressed(out / "arrays.npz", **arrays)
    afg.g4a_figures(out, robot, cfg, plan, col, rows, cov, suite, faults)
    cf.write_json(out / "result.json", {
        "stage": "G4A", "protocol": "INITIAL_SCENE_ONE_SHOT_V1", "plan_hash": plan["plan_hash"], "limit": args.limit,
        "coverage": cov, "costs": {"main_cpu_s": col["main_cpu_s"], "diagnostic_cpu_s": col["diagnostic_cpu_s"],
                                   "stopped": col["stopped"], "diagnostic_cap_exceeded": col["diagnostic_cap_exceeded"],
                                   "skipped_order_cases": col["skipped_order_cases"],
                                   "missing_seed_cases": [c["query_id"] for c in col["missing_seed_cases"]],
                                   "wall_s": round(time.perf_counter() - t0, 1)},
        "tasks": tasks, "config_hash": cf.config_hash(cfg), "robot_profile_hash": rhash,
        "verification_scope": "KINEMATIC_3R_RELOCATIONS_AND_NEWLY_PLANNED_TARGET_SKILL (no dynamics, no simulation)",
        "simulation_status": "NOT_RUN", "hardware_status": "NOT_RUN", "learned_components": "NONE"})
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["query_id", "stratum", "outcome", "committed", "moved", "assignments", "first_pass_pos",
                                "first_pass_neg", "first_pass_unk", "cpu_s"])
        w.writeheader()
        for q, p in prim.items():
            mine = [r for r in rows if r["query_id"] == q]
            w.writerow({"query_id": q, "stratum": p["stratum"], "outcome": p["outcome"],
                        "committed": p["commit"] and p["commit"]["label"], "moved": p["commit"] and len(p["commit"]["cells"]),
                        "assignments": len(mine), **{f"first_pass_{k}": sum(r["first_pass"] == v for r in mine) for k, v in
                                                     (("pos", data.POSITIVE), ("neg", data.NEGATIVE), ("unk", data.UNKNOWN))},
                        "cpu_s": p["cpu_s"]})
    cf.write_manifest(out, {"root_seed": cfg["audit4a"]["root_seed"], "generator_seed": cfg["generator"]["seed"],
                            "planner_base": cfg["planner2d"]["seed"], "per_query": "crc32(query/component key)"},
                      cf.config_hash(cfg), {"plan_hash": plan["plan_hash"]})
    print(json.dumps({"out": str(out), "decision": cov["decision"], "readiness": cov["readiness"], "costs": {
        "main_cpu_s": col["main_cpu_s"], "diagnostic_cpu_s": col["diagnostic_cpu_s"]}}))
    return 0
