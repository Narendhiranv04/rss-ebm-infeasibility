"""P6.2/P6.3: four-batch collection of the frozen P6 plan under the unchanged 2D-ABC-OPNEG-v1 policy.

Per root (predeclared order): reference first (a non-VERIFIED reference stops the lineage), then the
main repair, then the catalogue order, then the fixed-robot edits. Queries with an accepted static
certificate never call the teacher. Every seed attempt is a process-tree guarded unit of the
unchanged runner (teacher_pilot.run_rounds -> g4d_run.run_collection); the per-query CPU cap is
min(870, root, batch, aggregate remainders), so no cap can be exceeded (a unit is admitted only if
250 s + reserve still fits). A root whose queries show two consecutive CPU-censored seeds on two
queries starts no new teacher query. Batch b+1 runs only if the predeclared gate of batch b passes;
gates read development roots only (holdout records are sealed and appear as counts only).
Progress is persisted after every query; a restart never repeats an interrupted query.
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
import p6_scenes as p6
import sector_cert
import teacher_pilot as tp
from scene import scene_from_dict

CAPS = {"teacher_cpu_s": 20000.0, "static_cpu_s": 700.0, "root_cpu_s": 1500.0, "batch_cpu_s": 5000.0, "wall_s": 36000.0,
        "query_cpu_s": 870.0, "query_wall_s": 1800.0, "max_queries": 384, "max_seed_attempts": 1152, "max_queries_per_root": 24,
        "unknown_fraction_halt": 0.25, "min_complete_per_kcpu": 1.0, "workers": 1}
FROZEN = ("runs/g4c_d2_poc_s", "runs/g4c_pilot", "runs/g4d_evidence", "runs/g4c_lf_p4/pilot", "runs/p5/d3lite", "runs/p5/g4e")
GATE = ("batch 1: no FATAL/integrity violation; (a VERIFIED 4-move repair OR a decided 3-move dependency = VERIFIED |r|>=3 with "
        ">=2 decided negative omissions); (>=1 decided same-endpoint order contrast OR >=1 decided nonzero kappa); UNKNOWN <= 25% "
        "of attempted queries; budgets held; >=1 development root with >=2 fully decided nonempty V3 repairs. Batches 2-3 additionally: "
        "cumulative F1 order contrast and F3 3/4-step evidence, GOAL_OVERLAP < 1/3 of decided negatives, >= 1 complete repair per "
        "1000 teacher CPU-s, and >= 1500 CPU-s aggregate remaining.")


def _rd(p):
    return json.loads(Path(p).read_text())


def order(plan, lin) -> list:
    qs = [q for q in plan["queries"] if q["lineage"] == lin]
    rank = {"reference": 0, "main": 1}
    return sorted(qs, key=lambda q: (rank.get(q["role"], 2 if q["kind"] == "ORDERED_TEACHER" else 3), qs.index(q)))


def censored_twice(attempts) -> bool:
    st = [a["stop"] for a in attempts]
    return any(a == b == "CPU_CENSORED" for a, b in zip(st, st[1:]))


def kappas(plan, records) -> list:
    out = []
    for k, r in plan["roots"].items():
        if not r["queries"]:
            continue
        y = lambda q: (records.get(q) or {}).get("binary_training_label")  # noqa: E731
        sq = {q["kind"]: f"scene:{k}:{p6.z4.sig(sorted(q['edit'].items()))}" for q in r["queries"]["scene_queries"]}
        v = [y(f"seq:{k}:empty")] + [y(sq.get(x)) for x in ("A", "B", "A+B")]
        if None not in v:
            out.append({"root": k, "family": r["family"], "y0": v[0], "yA": v[1], "yB": v[2], "yAB": v[3], "kappa": v[3] - v[1] - v[2] + v[0],
                        "operational": any((records.get(q) or {}).get("label_source") == "PLANNER_BUDGET" for q in [f"seq:{k}:empty", *sq.values()])})
    return out


def dependencies(records) -> list:
    """VERIFIED repairs with >= 3 moves and their decided omission labels (multi-step dependency evidence)."""
    out = []
    for q, r in records.items():
        s = r["query"]["sequence"]
        if r["teacher_status"] == ot.VERIFIED and len(s) >= 3:
            om = [(records.get(f"seq:{r['query']['root']}:{p6.z4.sig([a for i, a in enumerate(s) if i != t])}") or {}).get("binary_training_label")
                  for t in range(len(s))]
            out.append({"query": q, "moves": len(s), "omissions": om, "necessary_steps": sum(x == 0 for x in om)})
    return out


def gate(plan, records, roots, b, used, cum) -> dict:
    qs = [q for q in plan["queries"] if q["batch"] == b]
    att = [q["query"] for q in qs if records[q["query"]]["teacher_status"] != "NOT_RUN"]
    unk = sum(records[q]["teacher_status"] == ot.UNKNOWN for q in att)
    fatal = any(records[q["query"]]["teacher_status"] == "FATAL_CONTRADICTION" for q in qs)
    dev = {q: r for q, r in records.items() if r["query"]["split"] != "holdout" and r["query"]["batch"] <= b}
    devb = {q: r for q, r in dev.items() if r["query"]["batch"] == b}
    droots = {k: v for k, v in plan["roots"].items() if v["queries"] and v["split"] != "holdout" and v["batch"] <= b}
    eff, kap, dep = ot.effects(dev, droots), kappas({"roots": droots}, dev), dependencies(dev)
    pl = d3r.pools({"roots": {k: v for k, v in droots.items() if v["batch"] == b}}, devb)
    four = [q for q, r in dev.items() if r["teacher_status"] == ot.VERIFIED and len(r["query"]["sequence"]) == 4]
    swaps = [s for s in eff["swap"] if s["decided_contrast"]]
    nz = [x for x in kap if x["kappa"] != 0]
    neg = [r for r in dev.values() if r["binary_training_label"] == 0]
    trivial = sum("GOAL_OVERLAP" in str((r.get("static_certificate") or {}).get("kind", "")) + str(r.get("reason")) for r in neg)
    complete = sum(v["complete_candidates"] for v in d3r.pools({"roots": droots}, dev).values())
    c = {"fatal": fatal, "attempted": len(att), "unknown": unk, "unknown_fraction": round(unk / max(1, len(att)), 3),
         "four_move_witnesses": four, "three_move_dependencies": [d for d in dep if d["necessary_steps"] >= 2],
         "decided_order_contrasts": len(swaps), "nonzero_kappa": len(nz), "multi_repair_roots": sum(v["complete_nonempty"] >= 2 for v in pl.values()),
         "decided_negatives": len(neg), "goal_overlap_negatives": trivial, "complete_candidates_dev": complete,
         "complete_per_kcpu": round(1000 * complete / max(1.0, used), 3), "budget_held": cum["held"]}
    ok = (not fatal and cum["held"] and c["unknown_fraction"] <= CAPS["unknown_fraction_halt"] and c["multi_repair_roots"] >= 1
          and bool(four or c["three_move_dependencies"]) and bool(swaps or nz))
    if b >= 2:
        fam = {r["query"]["family"] for r in dev.values()}
        f1 = any(s for s in swaps if plan["roots"][s["orders"][0].split(":")[1]]["family"] == "F1")
        f3 = any(plan["roots"][q.split(":")[1]]["family"] == "F3" for q in four + [d["query"] for d in c["three_move_dependencies"]])
        c.update(f1_order_evidence=f1, f3_multistep_evidence=f3, families_seen=sorted(fam))
        ok = ok and f1 and f3 and trivial < len(neg) / 3 and c["complete_per_kcpu"] >= CAPS["min_complete_per_kcpu"] and cum["remaining"] >= 1500
    return {"rule": GATE, "batch": b, **c, "pass": bool(ok)}


def run(args) -> int:
    out, t0, w0 = Path(args.out), time.process_time(), time.time()
    if (out / "summary.json").exists():
        raise SystemExit("P6 collection already finished; refusing to overwrite")
    for f, ok in ((Path(args.regression, "regression_result.json"), lambda r: r.get("passed") is True),
                  (Path(args.g4b, "result.json"), lambda r: r.get("decision") == "PASS")):
        if not ok(_rd(f)):
            raise SystemExit(f"pre-collection gate not passed: {f}")
    cfg, v2, pol = cf.load_config(args.config), cf.load_config_v2(args.config_v2), ot.load_policy(args.policy)
    if v2["proposal"]["move_cap"] != 4 or ot.config_mismatch(pol, cfg):
        raise SystemExit("P6 requires the R4 overlay (move_cap = 4) and an unchanged OPNEG-v1 policy/config")
    robot = ge.make_robot(cfg)
    out.mkdir(parents=True, exist_ok=True)
    frozen = {r: gr._sha(Path(r) / "manifest.json") for r in FROZEN}
    calls, ops = sector_cert.guard_and_count()                   # the coordinator itself never plans
    known = d3._known(Path("."))
    for r in _rd("runs/p5/d3lite/plan.json")["roots"].values():
        known.add(g4c.stable_geometry(scene_from_dict(r["scene"]), r["goal_cell"]))
    if (out / "plan.json").exists():
        plan, static, cons = _rd(out / "plan.json"), _rd(out / "static_admission.json")["certificates"], _rd(out / "construction.json")
    else:
        roots, ledger, cons = p6.construct(cfg, robot, known)
        plan = p6.plan(cfg, robot, pol, roots)
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
    static_cpu = cons["cpu_s"] + _rd(out / "static_admission.json")["cpu_s"]
    if static_cpu > CAPS["static_cpu_s"]:
        raise SystemExit("static construction + admission exceeded 700 CPU-s")
    paths = {"plan": out / "plan.json", "config": Path(args.config), "config_v2": Path(args.config_v2)}
    prog = _rd(out / "progress.json") if (out / "progress.json").exists() else {"done": {}, "status": {}, "wall_before_s": 0.0}
    wall0 = prog["wall_before_s"]
    unit = lambda: sum(d["unit_cpu_s"] for d in prog["done"].values())  # noqa: E731
    used = lambda: unit() + sum(d["coord_cpu_s"] for d in prog["done"].values())  # noqa: E731
    wall = lambda: wall0 + time.time() - w0  # noqa: E731
    for b in (1, 2, 3, 4):
        if str(b) in prog["status"]:
            if not prog["status"][str(b)]["pass"]:
                break
            continue
        b_used0 = sum(d["unit_cpu_s"] + d["coord_cpu_s"] for q, d in prog["done"].items() if d["batch"] < b)
        for spec in [s for s in p6.ROOTS if s["batch"] == b]:
            lin = f"p6_{spec['seed']}"
            for i, q in enumerate(order(plan, lin)):
                qid = q["query"]
                if qid in prog["done"]:
                    continue
                mine = [d for d in prog["done"].values() if d["lineage"] == lin]
                stop = None
                if any(d["role"] == "reference" and d["status"] != ot.VERIFIED for d in mine):
                    stop = "lineage stopped: reference scene has no verified base witness"
                elif sum(d["censored_twice"] for d in mine) >= 2:
                    stop = "root censoring stop: two queries with two consecutive CPU-censored seeds"
                c1, root_used = time.process_time(), sum(d["unit_cpu_s"] + d["coord_cpu_s"] for d in mine)
                cap = min(CAPS["query_cpu_s"], CAPS["root_cpu_s"] - root_used, CAPS["batch_cpu_s"] - (used() - b_used0),
                          CAPS["teacher_cpu_s"] - used())
                seeds, uc, qdir = {qid: []}, 0.0, out / "q" / qid.replace(":", "__").replace(">", "-")
                if stop is None and static.get(qid) is None and qdir.exists():
                    stop = "interrupted before this query was recorded; never repeated"
                if stop is None and static.get(qid) is None and cap < pol["policy"]["per_seed_cpu_s"] + pol["policy"]["per_seed_guard_margin_s"] + \
                        pol["pilot"]["orchestration_reserve_s"]:
                    stop = f"budget remainder {cap:.0f} CPU-s below one guarded seed attempt (root / batch / aggregate cap)"
                if stop is None and static.get(qid) is None:
                    seeds, uc = tp.run_rounds(plan, pol, qdir, paths, cfg, v2, robot, static, [qid], cap,
                                              min(CAPS["query_wall_s"], CAPS["wall_s"] - wall()), seeds)
                prog["done"][qid] = {"lineage": lin, "batch": b, "role": q["role"], "stop": stop, "attempts": seeds[qid], "unit_cpu_s": uc,
                                     "coord_cpu_s": time.process_time() - c1, "censored_twice": censored_twice(seeds[qid]), "status": None}
                prog["done"][qid]["status"] = judge(static, pol, q, prog["done"][qid])["teacher_status"]
                prog["wall_before_s"] = wall0 + time.time() - w0
                cf.write_json(out / "progress.json", prog)
        records = {q["query"]: judge(static, pol, q, prog["done"].get(q["query"])) for q in plan["queries"]}
        cum = {"held": used() <= CAPS["teacher_cpu_s"] and wall() <= CAPS["wall_s"], "remaining": CAPS["teacher_cpu_s"] - used()}
        prog["status"][str(b)] = gate(plan, records, None, b, unit(), cum) | {"cpu_used_after_batch": round(used(), 2),
                                                                                "wall_after_batch": round(wall(), 1)}
        cf.write_json(out / "progress.json", prog)
        if not prog["status"][str(b)]["pass"]:
            break
    records = {q["query"]: judge(static, pol, q, prog["done"].get(q["query"])) for q in plan["queries"]}
    return report(out, cfg, plan, pol, records, prog, static_cpu, calls, ops, t0, used(), unit(), wall(), frozen)


def judge(static, pol, q, d) -> dict:
    qid = q["query"]
    base = {"query": q, "static_certificate": static.get(qid)}
    if d is None or (d["stop"] and static.get(qid) is None):
        why = d["stop"] if d else "batch not run: an earlier checkpoint gate failed"
        return {**ot.adjudicate_query(None, [], pol), **base, "teacher_status": "NOT_RUN", "attempts": [], "J_first": None, "reason": why}
    try:
        r = ot.adjudicate_query(static.get(qid), d["attempts"], pol)
    except ot.Fatal as e:
        r = {**ot.adjudicate_query(None, [], pol), "teacher_status": "FATAL_CONTRADICTION", "reason": str(e)}
    return {**r, **base, "attempts": d["attempts"], "J_first": next((x["J_first"] for x in d["attempts"] if x["J_first"] is not None), None)}


def report(out, cfg, plan, pol, records, prog, static_cpu, calls, ops, t0, used, unit, wall, frozen) -> int:
    sealed = {q: r for q, r in records.items() if r["query"]["split"] == "holdout"}
    dev = {q: r for q, r in records.items() if q not in sealed}
    cf.write_json(out / "decisions.json", dev)
    cf.write_json(out / "sealed_holdout.json", sealed)
    att = [{**a, "query": q} for q, r in records.items() for a in r["attempts"]]
    with open(out / "attempts.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in att)
    droots = {k: v for k, v in plan["roots"].items() if v["queries"] and v["split"] != "holdout"}
    cf.write_json(out / "effects.json", {**ot.effects(dev, droots), "kappa_by_kind": kappas({"roots": droots}, dev), "dependencies": dependencies(dev)})
    cf.write_json(out / "pools.json", d3r.pools({"roots": droots}, dev))
    cf.write_json(out / "loss_masks.json", ot.loss_masks({q: r for q, r in records.items() if r["teacher_status"] != "NOT_RUN"}))
    st = (ot.VERIFIED, ot.CERTIFIED, ot.BUDGET, ot.UNKNOWN, "NOT_RUN", "FATAL_CONTRADICTION")
    raw = lambda rs: {s: sum(r["teacher_status"] == s for r in rs.values()) for s in st}  # noqa: E731
    by_root = {}
    for q, d in prog["done"].items():
        x = by_root.setdefault(d["lineage"], {"cpu_s": 0.0, "teacher_queries": 0, "censored_twice": 0})
        x["cpu_s"] = round(x["cpu_s"] + d["unit_cpu_s"] + d["coord_cpu_s"], 2)
        x["teacher_queries"] += bool(d["attempts"])
        x["censored_twice"] += d["censored_twice"]
    comp = {"static_cpu_s": round(static_cpu, 2), "unit_cpu_s": round(unit, 2), "teacher_aggregate_cpu_s": round(used, 2),
            "coordinator_cpu_s_this_process": round(time.process_time() - t0, 2), "wall_s": round(wall, 1), "caps": CAPS,
            "seed_attempts": len(att), "seed_stops": {s: sum(a["stop"] == s for a in att) for s in {a["stop"] for a in att}},
            "by_root": by_root, "censored_in_flight_unrecorded": sum(a["in_flight_calls_unrecorded"] for a in att),
            "coordinator_guarded_calls": calls, "coordinator_static_ops": ops}
    summ = {"raw_all": raw(records), "raw_dev": raw(dev), "raw_sealed_holdout_counts_only": raw(sealed), "gates": prog["status"],
            "compute": comp, "frozen_manifests_unchanged": {r: gr._sha(Path(r) / "manifest.json") == h for r, h in frozen.items()}}
    cf.write_json(out / "summary.json", summ)
    gg.write_manifest(out, {"root_seeds": [s["seed"] for s in p6.ROOTS], "policy_seeds": pol["policy"]["seeds"]}, cf.config_hash(cfg),
                      {"plan_hash": plan["plan_hash"], "policy_hash": pol["policy_hash"]})
    print(json.dumps({k: summ[k] for k in ("raw_all", "raw_dev")} | {"gates": {b: g["pass"] for b, g in prog["status"].items()},
                                                                       "cpu": comp["teacher_aggregate_cpu_s"], "wall": comp["wall_s"]}))
    return 1 if any(r["teacher_status"] == "FATAL_CONTRADICTION" for r in records.values()) else 0
