"""G4D evidence stage commands (report 0011): legacy recovery, final plan, guarded
collection, evidence assembly and the gate judgement. Writes runs/g4d_evidence/.

Order (each step refuses to overwrite earlier frozen outputs):
  g4d-legacy    revalidate archived evidence (CPU measured separately; a contradiction blocks)
  g4d-finalize  link the final execution plan to the pending plan (no outcome)
  g4d-run       verify sources, then execute the units under the authorized envelope (resumable)
  g4d-judge     rebuild unit evidence from re-checked checkpoints; gate decision; figures
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import config as cf
import data
import g4d_run as gr
import geometry as ge
import legacy
import sequences as sq
from scene import scene_from_dict

G4A = Path("runs/g4a_evidence_audit")


def _rd(path: Path):
    return json.loads(path.read_text())


# ------------------------------------------------------------------ commands: legacy, finalize, run
def cmd_g4d_legacy(args) -> int:
    cfg, v2, out = cf.load_config(args.config), cf.load_config_v2(args.config_v2), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "legacy_revalidation.json").exists():
        raise SystemExit("legacy revalidation already recorded; refusing to overwrite")
    robot, c0, t0 = ge.make_robot(cfg), time.process_time(), time.perf_counter()
    files = {f: gr._sha(G4A / f) for f in ("g4a_plan.json", "candidate_labels.json", "result.json", "scenes.json", "arrays.npz")}
    rv = legacy.revalidate(cfg, robot, G4A)
    om = legacy.omission_pairs(cfg, v2, rv, files)
    tm = legacy.two_move_accounting(rv)
    neg = [r for r in rv["rows"] if r["v1_first_pass"] == data.NEGATIVE]
    verdicts = {"negatives": len(neg), "transferred": sum(r["v2_status"] == data.NEGATIVE for r in neg),
                "rejected": sum(r["v2_status"] != data.NEGATIVE for r in neg),
                "certificate_records_rechecked": sum(len(r["certificates"]) for r in neg),
                "certificate_records_accepted": sum(v["new_checker_accepted"] for r in neg for v in r["certificates"]),
                "by_kind": {k: [sum(v["kind"] == k and v["new_checker_accepted"] for r in neg for v in r["certificates"]),
                                sum(v["kind"] == k for r in neg for v in r["certificates"])]
                            for k in ("GOAL_OVERLAP", "TARGET_PICK", "FIRST_MOVER", "STATIC_ENDPOINT_INVALID")},
                "archived_acceptance_disagreements": sum(
                    any(a["kind"] == v["kind"] and a["checker_accepted"] != v["new_checker_accepted"]
                        for a in r["archived_certificates"]) for r in neg for v in r["certificates"])}
    pos = [r for r in rv["rows"] if r["v1_first_pass"] == data.POSITIVE]
    rec = {"source_files": files, "contradictions": rv["contradictions"], "negative_transfer": verdicts,
           "positives": {"validated": sum(r["v2_status"] == data.POSITIVE for r in pos), "of": len(pos)},
           "rows": [{k: v for k, v in r.items() if k != "archived_certificates"} for r in rv["rows"]],
           "cpu_s": round(time.process_time() - c0, 1), "wall_s": round(time.perf_counter() - t0, 1),
           "accounting": "legacy revalidation; outside the collection envelope"}
    cf.write_json(out / "legacy_revalidation.json", rec)
    cf.write_json(out / "legacy_omission_pairs.json", om)
    cf.write_json(out / "legacy_two_move.json", tm)
    print(json.dumps({"contradictions": rv["contradictions"], **verdicts, "omission": om["summary"], "two_move": len(tm),
                      "cpu_s": rec["cpu_s"]}))
    return 1 if rv["contradictions"] else 0


def cmd_g4d_finalize(args) -> int:
    cfg, v2, out = cf.load_config(args.config), cf.load_config_v2(args.config_v2), Path(args.out)
    if (out / "final_plan.json").exists():
        raise SystemExit("final_plan.json already frozen; refusing to overwrite")
    leg = _rd(out / "legacy_revalidation.json")
    if leg["contradictions"]:
        raise SystemExit(f"legacy contradictions block collection: {leg['contradictions']}")
    plan = gr.make_final_plan(cfg, v2, ge.make_robot(cfg), Path("."))
    plan["legacy_revalidation_sha256"] = gr._sha(out / "legacy_revalidation.json")
    plan["plan_hash"] = cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    cf.write_json(out / "final_plan.json", plan)
    print(json.dumps({"plan_hash": plan["plan_hash"], "units": len(plan["units"]), "pending": plan["pending_plan"]["plan_hash"]}))
    return 0


def cmd_g4d_run(args) -> int:
    out = Path(args.out)
    plan = _rd(out / "final_plan.json")
    bad = gr.verify_sources(plan, Path("."))
    if bad:
        raise SystemExit(f"source or plan hash mismatch, collection refused: {bad}")
    if (out / "result.json").exists():
        raise SystemExit("collection already judged; refusing to run again")
    led = gr.run_collection(plan, out, {"plan": out / "final_plan.json", "config": Path(args.config),
                                        "config_v2": Path(args.config_v2)}, Path("."))
    print(json.dumps({"units": {k: v["stop"] for k, v in led["units"].items()}, "unit_cpu_s": round(led["unit_cpu_s"], 1),
                      "coordinator_cpu_s": round(led["coordinator_cpu_s"], 2), "failure": led.get("enforcement_failure")}))
    return 1 if led.get("enforcement_failure") else 0


def write_manifest(out: Path, seeds: dict, chash: str, extra: dict) -> None:
    """config.write_manifest, extended to the per-unit checkpoint subdirectories."""
    import platform
    import sys
    from importlib.metadata import version
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    cf.write_json(out / "manifest.json", {
        "command": sys.argv, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "seeds": seeds,
        "config_hash": chash, **extra, "versions": {"python": platform.python_version(), "numpy": version("numpy"),
                                                    "matplotlib": version("matplotlib"), "psutil": version("psutil")},
        "files": {n: gr._sha(out / n) for n in files}})


# ------------------------------------------------------------------ evidence assembly and gate judgement
P, N, U = data.POSITIVE, data.NEGATIVE, data.UNKNOWN


def assemble(cfg, v2, robot, plan: dict, led: dict, out: Path) -> dict:
    """Unit evidence rebuilt only from re-checked checkpoints; never-run units carry their stop."""
    ev_ = {}
    for unit in plan["units"]:
        st = led["units"].get(unit["unit"], {"stop": "NOT_RUN"})
        if "dir" in st:
            e = gr.unit_evidence(cfg, v2, robot, plan, unit, out / st["dir"])
        else:
            e = {"unit": unit["unit"], "kind": unit["kind"], "root": unit["root"], "sequence": unit.get("sequence", []),
                 "label": "NOT_RUN", "checks": {}, "certificates": [], "cost": None, "relocation": None, "target": None}
        ev_[unit["unit"]] = {**e, "stop": st["stop"], "cpu_s": st.get("cpu_s"), "wall_s": st.get("wall_s")}
    return ev_


def _sig(seq) -> str:
    return ".".join(f"{o}>{c}" for o, c in seq) or "empty"


def _pair(ref: str, alt: str) -> dict:
    lab = data.effect_label(ref, alt) if ref in (P, N, U) and alt in (P, N, U) else {"delta": None, "mask": False}
    return {"reference": ref, "alternative": alt, **lab}


def judge(cfg, v2, plan: dict, ev_: dict, leg: dict) -> dict:
    lab = lambda uid: ev_.get(uid, {}).get("label", "NOT_RUN")  # noqa: E731
    seq_u = lambda k, s: f"seq:{k}:{_sig(s)}"  # noqa: E731
    witnesses = [e for e in ev_.values() if any(x["kind"] == "complete_witness" for x in e.get("evidence", []))]
    # B1 two-move witness
    new2 = [e["unit"] for e in witnesses if len(e["sequence"]) == 2]
    leg2 = [f"{t['query_id']}.audit{t['audit_index']}" for t in leg["two_move"] if all(
        t[k] for k in ("joint_continuity", "achieved_state_consistency", "target_start_equality", "safe_retreat", "scope"))]
    # B2 order contrasts
    contrasts = []
    for k, good, bad in next(c for c in plan["cases"] if c["id"] == "D-V2-2")["contrast"]:
        for a, b in ((good, bad), (bad, good)):
            ua, ub = seq_u(k, a), seq_u(k, b)
            order_neg = any(c["accepted"] and c["scope"] == "ORDER" for c in ev_.get(ub, {}).get("certificates", []))
            if lab(ua) == P and lab(ub) == N and order_neg:
                contrasts.append({"root": k, "positive": a, "negative": b, "negative_certificate": [
                    c["kind"] for c in ev_[ub]["certificates"] if c["accepted"]]})
    contrast_rows = [{"root": k, "orders": [good, bad], "labels": [lab(seq_u(k, good)), lab(seq_u(k, bad))],
                      "stops": [ev_[seq_u(k, x)]["stop"] for x in (good, bad)]}
                     for k, good, bad in next(c for c in plan["cases"] if c["id"] == "D-V2-2")["contrast"]]
    # B3 scene pairs
    scene = []
    for u in plan["units"]:
        if u["kind"] == "SCENE_TARGET_ONLY":
            scene.append({"unit": u["unit"], "edit": u["edit"], "eligible": u["eligible"],
                          **_pair(lab(u["reference_unit"]), lab(u["unit"]))})
    # B4 omissions (new) and derived legacy
    om_new = []
    for key, plans in plan["omission_plans"].items():
        k, sig = key.split(":", 1)
        full = next(u["sequence"] for u in plan["units"] if u["unit"] == f"seq:{k}:{sig}")
        for o in plans:
            om_new.append({"root": k, "sequence": full, "omitted": o["omitted_objects"], "remaining": o["remaining"],
                           **_pair(lab(seq_u(k, o["remaining"])), lab(f"seq:{k}:{sig}"))})
    om_leg = leg["omission"]["summary"]
    # B5 continuity / achieved state / target start on every claimed witness
    b5 = {e["unit"]: {k: e["checks"].get(k) for k in ("joint_continuity", "achieved_state_consistency",
                                                        "target_start_equality")} for e in witnesses}
    blocking = {
        "B1_two_move_witness": {"pass": bool(new2 or leg2), "new_v2": new2, "derived_legacy": leg2},
        "B2_order_contrast": {"pass": bool(contrasts), "decided": contrasts, "attempted": contrast_rows},
        "B3_scene_effect_pair": {"pass": any(x["mask"] and x["eligible"] for x in scene), "pairs": scene},
        "B4_omission_pair": {"pass": any(x["mask"] for x in om_new) or om_leg["decided"] > 0,
                             "new_v2_decided": sum(x["mask"] for x in om_new), "derived_legacy_decided": om_leg["decided"],
                             "new_v2": om_new},
        "B5_witness_consistency": {"pass": bool(b5) and all(all(v.values()) for v in b5.values()), "per_witness": b5}}
    # D-V2 cases
    r2 = lambda s: lab(seq_u("R2", s))  # noqa: E731
    goal_cert = any(c["accepted"] and c["kind"] == "GOAL_OVERLAP" for c in ev_.get("seq:R2:empty", {}).get("certificates", []))
    costs = [{"unit": e["unit"], "root": e["root"], "sequence": e["sequence"], "J": e["cost"]["recomputed_J"],
              "endpoint_distance": round(sq.endpoint_distance(scene_from_dict(plan["roots"][e["root"]]["scene"]),
                                                              [tuple(a) for a in e["sequence"]]), 6)}
             for e in ev_.values() if e.get("cost") and e["cost"].get("label_available") and e["sequence"]]
    by_root = {}
    for c in costs:
        by_root.setdefault(c["root"], []).append(c)
    r4 = [x for x in om_new if x["root"] == "R4"]
    sing = [x for x in r4 if len(x["omitted"]) == 1]
    joint = [x for x in r4 if len(x["omitted"]) == 2]
    inter = None
    if r4 and all(x["mask"] for x in r4):
        inter = joint[0]["delta"] != sum(x["delta"] for x in sing)
    r5 = [x for x in scene if x["unit"].startswith("scene:R5")]
    r5_inter = None
    if r5 and all(x["mask"] for x in r5):
        s5 = [x for x in r5 if len(x["edit"]) == 1]
        r5_inter = next(x for x in r5 if len(x["edit"]) == 2)["delta"] != sum(x["delta"] for x in s5)
    cases = {
        "D-V2-0": {"met": lab("seq:R0:empty") == P or bool(leg["zero_move_roots"]),
                   "new_v2": {"seq:R0:empty": lab("seq:R0:empty"), "J": (ev_["seq:R0:empty"].get("cost") or {}).get(
                       "recomputed_J")}, "derived_legacy_zero_move_roots": leg["zero_move_roots"]},
        "D-V2-1": {"met": goal_cert and P in (r2([["o03", 60]]), r2([["o03", 38]]))
                   and any(x["mask"] for x in scene if x["unit"].startswith("scene:R2")),
                   "empty_goal_overlap_certificate": goal_cert, "repairs": {"o03>60": r2([["o03", 60]]),
                                                                           "o03>38": r2([["o03", 38]])},
                   "scene_pairs": [x for x in scene if x["unit"].startswith("scene:R2")]},
        "D-V2-2": {"met": bool(contrasts), "decided": contrasts, "attempted": contrast_rows},
        "D-V2-3": {"met": r2([["o03", 13]]) == N and r2([["o03", 60]]) == P,
                   "o03>60 (o01 stays)": r2([["o03", 60]]), "o03>13 alone (o01 still on 13)": r2([["o03", 13]]),
                   "chain o01>21,o03>13": r2([["o01", 21], ["o03", 13]])},
        "D-V2-4": {"met": bool(inter or r5_inter), "omission_singletons": sing, "omission_joint": joint,
                   "omission_non_additive": inter, "scene_R5": r5, "scene_non_additive": r5_inter},
        "D-V2-5": {"met": any(len(v) >= 2 for v in by_root.values()), "cost_labels": costs},
        "D-V2-6": {"met": lab("seq:R1:empty") in (P, N) and lab("start:R1:0") in (P, N),
                   "home": lab("seq:R1:empty"), "alternative_start": lab("start:R1:0"),
                   "pair": _pair(lab("seq:R1:empty"), lab("start:R1:0"))}}
    gate = "PASS" if all(b["pass"] for b in blocking.values()) else "BLOCKED"
    return {"gate": gate, "blocking": blocking, "cases": cases,
            "unmet_case_requirements": sorted(k for k, v in cases.items() if not v["met"]),
            "evidence_suite_complete": gate == "PASS" and all(v["met"] for v in cases.values())}


def cmd_g4d_judge(args) -> int:
    import g4d_evidence_figures as fig
    cfg, v2, out = cf.load_config(args.config), cf.load_config_v2(args.config_v2), Path(args.out)
    if (out / "result.json").exists():
        raise SystemExit("already judged; refusing to overwrite")
    c0, t0 = time.process_time(), time.perf_counter()
    plan, led = _rd(out / "final_plan.json"), _rd(out / "ledger.json")
    robot = ge.make_robot(cfg)
    lr = _rd(out / "legacy_revalidation.json")
    leg = {"two_move": _rd(out / "legacy_two_move.json"), "omission": _rd(out / "legacy_omission_pairs.json"),
           "zero_move_roots": sorted({r["query_id"] for r in lr["rows"] if r["v2_status"] == P and r["moved"] == 0})}
    ev_ = assemble(cfg, v2, robot, plan, led, out)
    j = judge(cfg, v2, plan, ev_, leg)
    calls = {"relocation_planner": sum((e.get("calls") or {}).get("relocation_planner", 0) for e in ev_.values()),
             "target_evaluations": sum((e.get("calls") or {}).get("target_evaluations", 0) for e in ev_.values()),
             "home_diagnostics": sum((e.get("target") or {}).get("home_return_status") not in (None, "NOT_RUN")
                                     for e in ev_.values()), "caps": plan["call_caps"]}
    cf.write_json(out / "unit_evidence.json", ev_)
    figs = fig.draw_all(out, robot, cfg, plan, ev_, j, leg)
    res = {"stage": "G4D", "protocol": data.PROTOCOL_V2, "final_plan_hash": plan["plan_hash"],
           "pending_plan_hash": plan["pending_plan"]["plan_hash"], "gate": j["gate"], "blocking": {
               k: v["pass"] for k, v in j["blocking"].items()}, "unmet_case_requirements": j["unmet_case_requirements"],
           "evidence_suite_complete": j["evidence_suite_complete"], "judgement": j, "calls": calls,
           "collection_cpu_s": round(led["unit_cpu_s"] + led["coordinator_cpu_s"], 1), "unit_cpu_s": round(led["unit_cpu_s"], 1),
           "coordinator_cpu_s": round(led["coordinator_cpu_s"], 2), "enforcement_failure": led.get("enforcement_failure"),
           "units": {k: {x: v.get(x) for x in ("stop", "label", "cpu_s", "wall_s")} for k, v in ev_.items()},
           "judge_cpu_s": round(time.process_time() - c0, 1), "judge_wall_s": round(time.perf_counter() - t0, 1),
           "figures": figs, "learned_components": "NONE", "simulation_status": "NOT_RUN", "hardware_status": "NOT_AUTHORIZED"}
    cf.write_json(out / "result.json", res)
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "kind", "root", "sequence", "stop", "label", "cpu_s", "wall_s", "relocation_calls", "target_calls", "J"])
        for k, e in ev_.items():
            c = e.get("calls") or {}
            w.writerow([k, e["kind"], e["root"], _sig(e["sequence"]), e["stop"], e["label"], e.get("cpu_s"), e.get("wall_s"),
                        c.get("relocation_planner"), c.get("target_evaluations"), (e.get("cost") or {}).get("recomputed_J")])
    write_manifest(out, {"planner_base": cfg["planner2d"]["seed"], "keys": "final_plan.json units"}, cf.config_hash(cfg),
                   {"final_plan_hash": plan["plan_hash"], "config_v2_hash": cf.stable_hash(v2)})
    print(json.dumps({"gate": j["gate"], "blocking": res["blocking"], "unmet": j["unmet_case_requirements"], "calls": calls}))
    return 0
