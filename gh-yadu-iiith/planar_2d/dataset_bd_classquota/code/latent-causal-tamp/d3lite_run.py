"""P5.2/P5.3 D3-lite collection under the unchanged 2D-ABC-OPNEG-v1 policy.

Predeclared flow per batch: (a) every reference (clean-base) query; (b) the 11 obstructed
queries only for lineages whose reference is VERIFIED (otherwise NOT_RUN: lineage stopped).
Batch 2 runs only if the batch-1 checkpoint gate passes. Holdout-root records go to a sealed
file and are excluded from every development summary.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import config as cf
import d3lite as d3
import g4d_gate as gg
import g4d_run as gr
import geometry as ge
import operational_teacher as ot
import sector_cert
import teacher_pilot as tp

CAPS = {"collection_cpu_s": 4800.0, "batch1_cpu_s": 2400.0, "wall_s": 7200.0, "static_cpu_s": 300.0, "max_queries": 96,
        "max_seed_attempts": 288, "unknown_fraction_halt": 0.25}
GATE = ("batch-1 gate: STOP on FATAL / integrity; HALT if (UNKNOWN + NOT_RUN) / planned > 0.25; HALT if no root has >= 2 "
        "fully decided nonempty repairs (V2 layout) or no same-endpoint swap pair is decided with differing labels")


def pools(plan, records) -> dict:
    out = {}
    for k, r in plan["roots"].items():
        if not r["queries"]:
            continue
        rows = []
        for p in r["queries"]["pool"]:
            lay = p["z_layout"]
            ys = {s: (records.get(q) or {}).get("binary_training_label") for s, q in lay["query"].items()}
            rows.append({"sequence": p["sequence"], "role": p["role"], "active": lay["active_count"], "decided": sum(y is not None for y in ys.values()),
                         "complete": all(y is not None for y in ys.values()), "y": ys.get("zR", ys.get("z0")), "slots": ys,
                         "operational": any((records.get(q) or {}).get("label_source") == "PLANNER_BUDGET" for q in lay["query"].values())})
        full = [x for x in rows if x["complete"]]
        out[k] = {"candidates": rows, "complete_candidates": len(full), "complete_nonempty": sum(bool(x["sequence"]) for x in full),
                  "joint_pool_ready": sum(bool(x["sequence"]) for x in full) >= 2 and {x["y"] for x in full} == {0, 1}}
    return out


def gate(plan, records, batch) -> dict:
    qs = [q["query"] for q in plan["queries"] if q["batch"] == batch]
    und = sum(records[q]["teacher_status"] in (ot.UNKNOWN, "NOT_RUN") for q in qs)
    eff = ot.effects({q: records[q] for q in qs}, {k: v for k, v in plan["roots"].items() if v["batch"] == batch and v["queries"]})
    pl = pools({"roots": {k: v for k, v in plan["roots"].items() if v["batch"] == batch}}, records)
    multi = sum(v["complete_nonempty"] >= 2 for v in pl.values())
    swaps = sum(s["decided_contrast"] for s in eff["swap"])
    fatal = any(records[q]["teacher_status"] == "FATAL_CONTRADICTION" for q in qs)
    ok = not fatal and und / len(qs) <= CAPS["unknown_fraction_halt"] and multi >= 1 and swaps >= 1
    return {"rule": GATE, "planned": len(qs), "undecided": und, "undecided_fraction": round(und / len(qs), 3), "roots_with_multi_repair_pool": multi,
            "decided_order_contrasts": swaps, "fatal": fatal, "pass": ok}


def run(args) -> int:
    out, t0, w0 = Path(args.out), time.process_time(), time.time()
    if (out / "decisions.json").exists():
        raise SystemExit("D3-lite already collected; refusing to overwrite")
    reg = json.loads(Path(args.regression, "regression_result.json").read_text())
    if not reg.get("passed"):
        raise SystemExit("P5.0 regression gate not passed: no collection")
    cfg, v2, pol = cf.load_config(args.config), cf.load_config_v2(args.config_v2), ot.load_policy(args.policy)
    robot = ge.make_robot(cfg)
    out.mkdir(parents=True, exist_ok=True)
    roots, ledger = d3.construct(cfg, robot)
    plan = d3.plan(cfg, robot, pol, roots)
    if len(plan["queries"]) > CAPS["max_queries"] or len(plan["units"]) > CAPS["max_seed_attempts"]:
        raise SystemExit("frozen plan exceeds the authorized query / seed-attempt caps")
    with open(out / "construction_attempts.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in ledger)
    cf.write_json(out / "plan.json", {**plan, "caps": CAPS, "gate": GATE})
    calls, ops = sector_cert.guard_and_count()            # the coordinator itself never plans
    static = tp.static_admission(cfg, robot, plan, use_sector=True)
    static_cpu = time.process_time() - t0
    cf.write_json(out / "static_admission.json", {"certificates": static, "cpu_s": round(static_cpu, 2), "ops": dict(ops)})
    if static_cpu > CAPS["static_cpu_s"]:
        raise SystemExit("static admission exceeded 300 CPU-s")
    paths = {"plan": out / "plan.json", "config": Path(args.config), "config_v2": Path(args.config_v2)}
    seeds, unit_cpu, records, status = {q["query"]: [] for q in plan["queries"]}, 0.0, {}, {}
    used = lambda: time.process_time() - t0 + unit_cpu  # noqa: E731
    for b in (1, 2):
        cap = (min(CAPS["batch1_cpu_s"], CAPS["collection_cpu_s"]) if b == 1 else CAPS["collection_cpu_s"]) - used()
        bq = [q for q in plan["queries"] if q["batch"] == b]
        refs = [q["query"] for q in bq if q["role"] == "reference"]
        c0 = used()
        sd, uc = tp.run_rounds(plan, pol, out / f"b{b}a", paths, cfg, v2, robot, static, refs, cap, CAPS["wall_s"] - (time.time() - w0), seeds)
        unit_cpu += uc
        ok_lin = {q["lineage"] for q in bq if q["role"] == "reference" and ot.adjudicate_query(static[q["query"]], sd[q["query"]], pol)
                  ["teacher_status"] == ot.VERIFIED}
        rest = [q["query"] for q in bq if q["role"] != "reference" and q["lineage"] in ok_lin]
        sd, uc = tp.run_rounds(plan, pol, out / f"b{b}b", paths, cfg, v2, robot, static, rest, cap - (used() - c0),
                               CAPS["wall_s"] - (time.time() - w0), sd)
        unit_cpu += uc
        for q in bq:
            qid = q["query"]
            try:
                d = ot.adjudicate_query(static[qid], sd[qid], pol)
            except ot.Fatal as e:
                d = {**ot.adjudicate_query(None, [], pol), "teacher_status": "FATAL_CONTRADICTION", "reason": str(e)}
            if q["role"] != "reference" and q["lineage"] not in ok_lin:
                d = {**d, "teacher_status": "NOT_RUN", "reason": "lineage stopped: reference scene has no verified base witness"}
            records[qid] = {**d, "query": q, "static_certificate": static[qid], "attempts": sd[qid],
                            "J_first": next((x["J_first"] for x in sd[qid] if x["J_first"] is not None), None)}
        status[b] = gate(plan, records, b) | {"cpu_used_after_batch": round(used(), 2), "wall_after_batch": round(time.time() - w0, 1)}
        if status[b]["fatal"] or b == 2 or not status[b]["pass"]:
            break
    for q in plan["queries"]:
        if q["query"] not in records:
            records[q["query"]] = {**ot.adjudicate_query(None, [], pol), "teacher_status": "NOT_RUN", "query": q, "static_certificate": None,
                                   "attempts": [], "J_first": None, "reason": "batch 2 not run: batch-1 checkpoint gate failed"}
    return report(out, cfg, robot, plan, pol, records, status, static_cpu, calls, ops, t0, w0, unit_cpu)


def report(out, cfg, robot, plan, pol, records, status, static_cpu, calls, ops, t0, w0, unit_cpu) -> int:
    import d3lite_figures as fig
    sealed = {q: r for q, r in records.items() if r["query"]["split"] == "holdout"}
    dev = {q: r for q, r in records.items() if q not in sealed}
    for x in records.values():
        x.setdefault("policy_hash", pol["policy_hash"])
    cf.write_json(out / "decisions.json", dev)
    cf.write_json(out / "sealed_holdout.json", sealed)
    att = [{**a, "query": q} for q, r in records.items() for a in r["attempts"]]
    with open(out / "attempts.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in att)
    droots = {k: v for k, v in plan["roots"].items() if v["queries"] and v["split"] != "holdout"}
    cf.write_json(out / "effects.json", ot.effects(dev, droots))
    cf.write_json(out / "pools.json", pools({"roots": droots}, dev))
    cf.write_json(out / "loss_masks.json", ot.loss_masks({q: r for q, r in records.items() if r["teacher_status"] != "NOT_RUN"}))
    raw = lambda rs: {s: sum(r["teacher_status"] == s for r in rs.values()) for s in (ot.VERIFIED, ot.CERTIFIED, ot.BUDGET, ot.UNKNOWN,  # noqa: E731
                                                                                      "NOT_RUN", "FATAL_CONTRADICTION")}
    comp = {"static_cpu_s": round(static_cpu, 2), "unit_cpu_s": round(unit_cpu, 2), "coordinator_cpu_s": round(time.process_time() - t0, 2),
            "aggregate_cpu_s": round(time.process_time() - t0 + unit_cpu, 2), "wall_s": round(time.time() - w0, 1), "caps": CAPS,
            "seed_attempts": len(att), "seed_stops": {s: sum(a["stop"] == s for a in att) for s in {a["stop"] for a in att}},
            "inner_calls": {k: sum((a.get("calls") or {}).get(k, 0) or 0 for a in att) for k in ("relocation_planner", "target_evaluations")},
            "censored_in_flight_unrecorded": sum(a["in_flight_calls_unrecorded"] for a in att), "coordinator_guarded_calls": calls,
            "coordinator_static_ops": ops}
    summ = {"raw_all": raw(records), "raw_dev": raw(dev), "raw_sealed_holdout_counts_only": raw(sealed), "gates": status, "compute": comp,
            "frozen_manifests_unchanged": all(gr._sha(Path(r) / "manifest.json") == h for r, h in FROZEN.items())}
    cf.write_json(out / "summary.json", summ)
    fig.figures(out, robot, cfg, plan, records)
    gg.write_manifest(out, {"root_seeds": d3.SEEDS, "policy_seeds": pol["policy"]["seeds"]}, cf.config_hash(cfg),
                      {"plan_hash": plan["plan_hash"], "policy_hash": pol["policy_hash"]})
    print(json.dumps({k: summ[k] for k in ("raw_all", "raw_dev", "gates")} | {"cpu": comp["aggregate_cpu_s"], "wall": comp["wall_s"]}))
    return 1 if any(s.get("fatal") for s in status.values()) else 0


FROZEN = {r: gr._sha(Path(r) / "manifest.json") for r in ("runs/g4c_d2_poc_s", "runs/g4c_pilot", "runs/g4d_evidence", "runs/g4c_lf_p4/pilot")}
