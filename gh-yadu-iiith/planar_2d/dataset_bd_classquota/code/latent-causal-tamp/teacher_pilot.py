"""G4C-LF-P4: guarded regression gate and the bounded 2D-ABC-OPNEG-v1 calibration pilot.

Reuses the unchanged runner (g4d_run.run_collection: process-tree guard, ledger, atomic
checkpoints, ambiguous-resume rule), the unchanged worker/teacher (ordered.teach_sequence),
unit_evidence revalidation and the existing certificate producers/checkers. A seed is a key
namespace (RRT seed = planner.call_seed(41, key/label)); seeds run seed-major and a query
short-circuits only on a revalidated witness or an accepted certificate.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import config as cf
import g4c_pilot as gp
import g4d_gate as gg
import g4d_run as gr
import geometry as ge
import operational_teacher as ot
import ordered as od
import teacher

SRC = {"POC_S": Path("runs/g4c_d2_poc_s"), "PILOT": Path("runs/g4c_pilot")}
PLAN_FILE = {"POC_S": "final_plan.json", "PILOT": "pilot_plan.json"}
QUERIES = [("POC_S", u) for u in ("seq:d2_g4c_r00_AB:empty", "scene:d2_g4c_r00_AB:o04>49", "scene:d2_g4c_r00_AB:o05>41",
                                  "scene:d2_g4c_r00_AB:o04>49.o05>41", "seq:d2_g4c_r00_AB:o04>49.o05>41",
                                  "seq:d2_g4c_r00_AB:o04>49", "seq:d2_g4c_r00_AB:o05>41")] + \
          [("PILOT", u) for u in ("seq:g4c_r19:o01>28.o00>27", "seq:g4c_r19:o00>27.o01>28", "seq:g4c_r19:o00>27",
                                  "seq:g4c_r19:empty", "scene:g4c_r19:o02>26")]
REPLAYS = ("g1_robot", "g1_hardening", "g2_interventions", "g3_repairs", "g3_5_initial_scene", "g4a_evidence_audit")
_rd = lambda p: json.loads(Path(p).read_text())  # noqa: E731


# ------------------------------------------------------------------ P4.2 regression gate (process-tree guarded)
def regression(args) -> int:
    out = Path(args.out)
    if (out / "regression_result.json").exists():
        raise SystemExit("regression already run; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    lines = ["set -u"] + [f"python run.py replay --run runs/{r} --report {out}/replay_{r}.json > {out}/replay_{r}.log 2>&1; "
                          f"echo \"{r} $?\" >> {out}/replay_exit.txt" for r in REPLAYS]
    lines.append(f"COUNT_OUT={out}/calls.json PROGRESS={out}/progress.jsonl PYTHONPATH={out} "
                 f"python -m pytest -q -p countcalls tests > {out}/pytest.log 2>&1; echo \"pytest $?\" >> {out}/pytest_exit.txt")
    (out / "regression.sh").write_text("\n".join(lines) + "\n")
    g = teacher.guard(["bash", str(out / "regression.sh")], args.cpu, args.wall, out / "heartbeat.txt", 15.0, 0.25, Path("."))
    ex = dict(x.split() for x in (out / "replay_exit.txt").read_text().splitlines()) if (out / "replay_exit.txt").exists() else {}
    reps = {r: (_rd(out / f"replay_{r}.json").get("replay_passed") if (out / f"replay_{r}.json").exists() else None) for r in REPLAYS}
    prog = [json.loads(x) for x in (out / "progress.jsonl").read_text().splitlines()] if (out / "progress.jsonl").exists() else []
    outcome = {o: sum(p["outcome"] == o for p in prog) for o in ("passed", "failed", "skipped")}
    py = (out / "pytest_exit.txt").read_text().split()[-1] if (out / "pytest_exit.txt").exists() else None
    res = {"guard": {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "trigger_cpu_s", "cpu_limit_s")},
           "caps": {"cpu_s": args.cpu, "wall_s": args.wall}, "replays": reps, "replay_exit": ex, "tests": outcome,
           "pytest_exit": py, "calls": _rd(out / "calls.json") if (out / "calls.json").exists() else None}
    res["passed"] = g["stop"] == "COMPLETED" and py == "0" and all(reps.values()) and len(reps) == 6 and outcome["failed"] == 0
    res["verdict"] = "P3A_REGRESSION_CLOSED" if res["passed"] else "REGRESSION_BLOCKED"
    cf.write_json(out / "regression_result.json", res)
    print(json.dumps(res))
    return 0 if res["passed"] else 1


# ------------------------------------------------------------------ frozen pilot plan (before any teacher call)
def build_plan(cfg, pol) -> dict:
    roots, units, queries = {}, [], []
    for run, uid in QUERIES:
        src = _rd(SRC[run] / PLAN_FILE[run])
        u = next(x for x in src["units"] if x["unit"] == uid)
        roots[u["root"]] = src["roots"][u["root"]]
        sig = uid.split(":", 2)[2]
        queries.append({"query": uid, "source_run": run, "root": u["root"], "kind": u["kind"], "sequence": u.get("sequence", []),
                        "edit": u.get("edit"), "lineage": roots[u["root"]].get("lineage") or roots[u["root"]].get("family"),
                        "source_unit_sha": cf.stable_hash(u)})
        for s in pol["policy"]["seeds"]:
            key = pol["policy"]["seed_key"].format(root=u["root"], sig=("scene/" if u["kind"] == "SCENE_TARGET_ONLY" else "") + sig,
                                                   seed=s)
            units.append({**{k: v for k, v in u.items() if k not in ("target_key", "relocation_key", "unit")},
                          "unit": f"{uid}#s{s}", "query": uid, "seed": s, "target_key": key + "/target",
                          "relocation_key": key + "/reloc"})
    lin = {q["root"] for q in queries}
    if len(queries) > pol["pilot"]["max_queries"] or len(units) > pol["pilot"]["max_seed_attempts"] or \
            len(lin) > pol["pilot"]["max_lineages"]:
        raise SystemExit("pilot plan exceeds the authorized query / attempt / lineage caps")
    plan = {"stage": "G4C_LF_P4_PILOT", "policy_id": pol["policy"]["policy_id"], "policy_hash": pol["policy_hash"],
            "seeds": pol["policy"]["seeds"], "queries": queries, "roots": roots, "units": units, "config_hash": cf.config_hash(cfg),
            "robot_profile_hash": cf.robot_profile_hash(cfg), "limits": {}}
    plan["plan_hash"] = cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    return plan


def static_admission(cfg, robot, plan, use_sector: bool) -> dict:
    """Zero-planner certificates: existing producers/checkers (ordered.certificates_for) and,
    once the P3a regression gate is closed, the LINK0_SECTOR_PICK checker for fixed starts."""
    import scene as sc_
    import sector_cert
    import sector_check
    out = {}
    for q in plan["queries"]:
        unit = next(u for u in plan["units"] if u["query"] == q["query"])
        scene, seq, query, _ = gr.unit_scene(cfg, robot, plan, unit)
        acc = [{"kind": c["kind"], "scope": c["scope"]} for c in od.certificates_for(robot, cfg, scene, query["learner"], seq)
               if c["checker_accepted"]]
        root = plan["roots"][q["root"]]
        if use_sector and not acc:
            scope = "SCENE_TARGET_ONLY" if q["kind"] == "SCENE_TARGET_ONLY" else "FIRST_ACTION_ORDER" if seq else "ORIGINAL_TARGET"
            moved, dest = (seq[0][0], seq[0][1]) if seq else ("t0", root["goal_cell"])
            rec = sector_cert.produce(robot, cfg, scene, moved, scope, seq, dest)
            src = {"scene": sector_check.rebuild(sc_.scene_from_dict(root["scene"]), edit=unit.get("edit")), "scope": scope,
                   "moved": moved, "sequence": seq, "kind": q["kind"], "expected_hash": unit.get("intervened_scene_hash") or root["scene_hash"]}
            if sector_check.check(robot, cfg, rec, src)["accepted"]:
                acc.append({"kind": "LINK0_SECTOR_PICK", "scope": "ORDER" if scope == "FIRST_ACTION_ORDER" else scope})
        out[q["query"]] = acc[0] if acc else None
    return out


# ------------------------------------------------------------------ the bounded pilot
def pilot(args) -> int:
    out, t0, w0 = Path(args.out), time.process_time(), time.time()
    if (out / "decisions.json").exists():
        raise SystemExit("pilot already adjudicated; refusing to overwrite")
    reg = _rd(Path(args.regression) / "regression_result.json")
    if not reg.get("passed"):
        raise SystemExit("REGRESSION_BLOCKED: P4.3 physical calibration is not allowed")
    cfg, v2, pol = cf.load_config(args.config), cf.load_config_v2(args.config_v2), ot.load_policy(args.policy)
    robot = ge.make_robot(cfg)
    if ot.config_mismatch(pol, cfg):
        raise SystemExit(f"policy/config mismatch: {ot.config_mismatch(pol, cfg)}")
    frozen = {r: gr._sha(SRC[r] / "manifest.json") for r in SRC}
    out.mkdir(parents=True, exist_ok=True)
    plan = build_plan(cfg, pol)
    cf.write_json(out / "pilot_plan.json", plan)
    cf.write_json(out / "policy.json", pol)
    import sector_cert
    calls, ops = sector_cert.guard_and_count()             # the coordinator itself may never plan
    static = static_admission(cfg, robot, plan, use_sector=True)
    static_cpu = time.process_time() - t0
    cf.write_json(out / "static_admission.json", {"certificates": static, "cpu_s": round(static_cpu, 2), "ops": dict(ops)})
    if static_cpu > pol["pilot"]["static_admission_cpu_s"]:
        raise SystemExit("static admission exceeded its CPU cap")
    paths = {"plan": out / "pilot_plan.json", "config": Path(args.config), "config_v2": Path(args.config_v2)}
    seeds, unit_cpu = run_rounds(plan, pol, out, paths, cfg, v2, robot, static, [q["query"] for q in plan["queries"]],
                                 pol["pilot"]["aggregate_cpu_s"] - (time.process_time() - t0), pol["pilot"]["aggregate_wall_s"])
    records, fatal = {}, []
    for q in plan["queries"]:
        try:
            d = ot.adjudicate_query(static[q["query"]], seeds[q["query"]], pol)
        except ot.Fatal as e:
            fatal.append(str(e))
            d = {**ot.adjudicate_query(None, [], pol), "teacher_status": "FATAL_CONTRADICTION", "reason": str(e)}
        records[q["query"]] = {**d, "query": q, "static_certificate": static[q["query"]], "attempts": seeds[q["query"]],
                               "J_first": next((x["J_first"] for x in seeds[q["query"]] if x["J_first"] is not None), None)}
    report(out, cfg, robot, plan, pol, records, seeds, static, fatal, frozen, calls, ops, t0, w0, unit_cpu)
    return 1 if fatal else 0


def run_rounds(plan, pol, out, paths, cfg, v2, robot, static, queries, cpu_cap, wall_cap, seeds=None) -> tuple:
    """Seed-major rounds over `queries` through the unchanged guarded runner. A query stops at a
    revalidated witness or an accepted (static or worker) certificate. Each round's admission uses
    the remaining CPU / wall of THIS call (process time of the caller + guarded unit CPU)."""
    P, seeds, unit_cpu = pol["policy"], seeds or {q: [] for q in queries}, 0.0
    c0, w0 = time.process_time(), time.time()
    for s in P["seeds"]:
        todo = [q for q in queries if static.get(q) is None and
                not any(x["terminal"] in ("WITNESS", "CERTIFICATE") for x in seeds[q])]
        if not todo:
            break
        rp = {"units": [u for u in plan["units"] if u["seed"] == s and u["query"] in todo], "limits": {
            "cpu_per_unit_s": P["per_seed_cpu_s"], "guard_margin_s": P["per_seed_guard_margin_s"], "wall_per_unit_s": P["per_seed_wall_s"],
            "cpu_aggregate_s": cpu_cap - (time.process_time() - c0 + unit_cpu), "wall_total_s": wall_cap - (time.time() - w0),
            "poll_s": 0.25, "orchestration_reserve_s": pol["pilot"]["orchestration_reserve_s"], "workers": 1}}
        rp["plan_hash"] = cf.stable_hash(rp)
        led = gr.run_collection(rp, out / f"seed{s}", paths, Path("."))
        unit_cpu += led["unit_cpu_s"]
        for u in rp["units"]:
            st = led["units"].get(u["unit"], {"stop": "NOT_RUN"})
            if "dir" not in st:
                continue
            e = gr.unit_evidence(cfg, v2, robot, plan, u, out / f"seed{s}" / st["dir"])
            seeds[u["query"]].append({"seed": s, "unit": u["unit"], **ot.seed_terminal(e, st, set(P["allowed_terminals"])),
                                      "J_first": (e.get("cost") or {}).get("recomputed_J"), "witness_hash": (e.get("cost") or {}).get("witness_hash"),
                                      "in_flight_calls_unrecorded": st["stop"] != "COMPLETED"})
        if led.get("enforcement_failure"):
            break
    return seeds, unit_cpu


def report(out, cfg, robot, plan, pol, records, seeds, static, fatal, frozen, calls, ops, t0, w0, unit_cpu):
    import teacher_pilot_figures as tf
    hist = {}
    for run, uid in QUERIES:
        hist[uid] = gp.adjudicate(_rd(SRC[run] / "unit_evidence.json")[uid])["label"]
    stab = {q: {"terminals": [x["terminal"] for x in seeds[q]], "historical": hist[q], "p4": r["teacher_status"],
                "success_after_failed_seed": bool(seeds[q]) and seeds[q][0]["terminal"] != "WITNESS" and
                any(x["terminal"] == "WITNESS" for x in seeds[q][1:]),
                "conflicts_with_historical": (hist[q] == "VERIFIED_FEASIBLE" and r["binary_training_label"] == 0) or
                (hist[q] == "CERTIFIED_INFEASIBLE" and r["binary_training_label"] == 1)} for q, r in records.items()}
    proj = ot.projections({q: r for q, r in records.items() if r["teacher_status"] != "FATAL_CONTRADICTION"}, pol["policy_hash"])
    eff = ot.effects(records, plan["roots"])
    pools = {f"{k}:{gp._sig(s)}": ot.pool(k, plan["roots"][k], s, records) for k, s in
             (("d2_g4c_r00_AB", [["o04", 49], ["o05", 41]]), ("g4c_r19", [["o01", 28], ["o00", 27]]))}
    att = [{**x, "query": q} for q in seeds for x in seeds[q]]
    with open(out / "attempts.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in att)
    calls_in = {k: sum((x.get("calls") or {}).get(k, 0) or 0 for x in att) for k in ("relocation_planner", "target_evaluations")}
    comp = {"static_cpu_s": _rd(out / "static_admission.json")["cpu_s"], "unit_cpu_s": round(unit_cpu, 2),
            "coordinator_cpu_s": round(time.process_time() - t0, 2), "aggregate_cpu_s": round(time.process_time() - t0 + unit_cpu, 2),
            "wall_s": round(time.time() - w0, 2), "caps": pol["pilot"], "seed_attempts": len(att),
            "seed_stops": {s: sum(x["stop"] == s for x in att) for s in {x["stop"] for x in att}},
            "recorded_inner_calls": calls_in, "censored_in_flight_unrecorded": sum(x["in_flight_calls_unrecorded"] for x in att),
            "coordinator_guarded_calls": calls, "coordinator_static_ops": ops}
    integ = {"frozen_manifests_unchanged": all(gr._sha(SRC[r] / "manifest.json") == h for r, h in frozen.items()),
             "fatal": fatal, "mixed_policy": False}
    cf.write_json(out / "decisions.json", records)
    for name, obj in (("projections", proj), ("effects", eff), ("pools", pools), ("stability", stab), ("compute", comp),
                      ("integrity", integ)):
        cf.write_json(out / f"{name}.json", obj)
    tf.figures(out, robot, cfg, plan, records, att)
    gg.write_manifest(out, {"seeds": pol["policy"]["seeds"]}, cf.config_hash(cfg), {"policy_hash": pol["policy_hash"],
                                                                                     "plan_hash": plan["plan_hash"]})
    print(json.dumps({"raw": proj["raw"], "pools": {k: [v.get("complete"), v.get("decided"), v.get("active")] for k, v in pools.items()},
                      "compute": {k: comp[k] for k in ("aggregate_cpu_s", "wall_s", "seed_attempts", "seed_stops")}, "fatal": fatal}))
