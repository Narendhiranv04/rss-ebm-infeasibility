"""G4C-P1 judgement outputs: rebuild every unit's evidence from re-checked checkpoints,
apply the G4C judge, check integrity, and write pilot_units.csv, pilot_roots.csv,
pilot_coverage.json, bc_pool_coverage.json, budget_ledger.json, unit_evidence.json,
pilot_result.json, figures and a recursive manifest to runs/g4c_pilot/.

Figures: A initial scenes with frozen ghost actions (PILOT, SCHEMATIC for proposals);
B coverage (decided / unknown / not-run per category; full-gate targets as FUTURE);
C per-unit CPU/wall against the guard and cumulative CPU against the cap; D genuine
recorded evidence panels (or UNDECIDED / NOT_AVAILABLE).
"""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

import config as cf
import data
import g4c_figures as g4cf
import g4c_pilot as gp
import g4d_evidence_figures as gef
import g4d_gate as gg
import g4d_run as gr
import geometry as ge
from viz import COLORS, plt

P, N, U = data.POSITIVE, data.NEGATIVE, data.UNKNOWN
COL = {P: COLORS["valid"], N: COLORS["invalid"], U: COLORS["unresolved"], "NOT_RUN": "#bdbdbd"}


def entitlement(j: dict, plan: dict) -> dict:
    """What A, B and C may each consume: identical source units; B/C add typed z views."""
    lab = j["labels"]
    dec = [u for u, v in lab.items() if v["label"] in (P, N)]
    pairs = [x for r in j["roots"].values() for x in r["scene"].values() if x["mask"]]
    om = [x for r in j["roots"].values() for x in r["decided_omissions"]]
    cost = [c for r in j["roots"].values() for c in r["costs"]]
    common = {"source_units": len(plan["units"]), "decided_units": len(dec), "decided_ordered": sum(u.startswith("seq:") for u in dec),
              "decided_scene_pairs": len(pairs), "decided_omission_pairs": len(om), "cost_labels": len(cost)}
    pools = [k for k, r in j["roots"].items() if r["bc_pool"]["qualifies"]]
    return {"A_LOCAL_SEQUENCE": {**common, "joint_pool_loss": "A finite-pool loss over decided ordered candidates"},
            "B_JOINT_CAUSAL": {**common, "z_fields_decided": sum(r["bc_pool"].get("decided_slots", 0) for r in j["roots"].values()),
                               "z_fields_active": sum(r["bc_pool"].get("active_slots", 0) for r in j["roots"].values()),
                               "exact_joint_pool_roots": pools},
            "C_HYBRID_JOINT": "identical to B (same units, same z views); the GRU is a future component",
            "equal_source_information": True, "learner_inputs": "data.learner_tensors_v2 allowlist; no labels/witnesses/root ids"}


def run(args) -> int:
    t0, c0, out, root = time.perf_counter(), time.process_time(), Path(args.out), Path(".")
    if (out / "pilot_result.json").exists():
        raise SystemExit("pilot already judged; refusing to overwrite")
    cfg, v2 = cf.load_config(args.config), cf.load_config_v2(args.config_v2)
    plan, led, parent = gp._rd(out / "pilot_plan.json"), gp._rd(out / "ledger.json"), gp._rd(gp.PARENT)
    robot = ge.make_robot(cfg)
    ev = gg.assemble(cfg, v2, robot, plan, led, out)
    j = gp.judge(plan, parent, ev)
    integ = {"sources_unchanged": gr.verify_sources(plan, root) == [], "parent_plan_file_unchanged":
             gp._sha(gp.PARENT) == plan["parent_plan_file_sha256"], "parent_plan_hash": parent["plan_hash"] == gp.PARENT_HASH,
             "authorization_unchanged": gp._sha(out / "authorization.json") == plan["authorization_file_sha256"],
             "units_run_subset_of_pilot": set(led["units"]) <= {u["unit"] for u in plan["units"]},
             "excluded_roots_never_run": not any(u.split(":")[1] not in gp.PILOT for u in led["units"]),
             "no_contradiction": data.CONTRADICTION not in j["label_counts"]}
    calls = {"relocation_planner": sum((e.get("calls") or {}).get("relocation_planner", 0) for e in ev.values()),
             "target_evaluations": sum((e.get("calls") or {}).get("target_evaluations", 0) for e in ev.values()),
             "home_diagnostics": sum((e.get("target") or {}).get("home_return_status") not in (None, "NOT_RUN") for e in ev.values()),
             "caps": {"relocation_skills": 60, "target_evaluations": 59}}
    stops = {k: v["stop"] for k, v in led["units"].items()}
    cpu = led["unit_cpu_s"] + led["coordinator_cpu_s"]
    budget = {"units": {k: {x: v.get(x) for x in ("stop", "cpu_s", "wall_s", "overshoot_s", "trigger_cpu_s", "cpu_limit_s",
                                                   "returncode", "start_utc", "end_utc")} for k, v in led["units"].items()},
              "unit_cpu_s": round(led["unit_cpu_s"], 2), "coordinator_cpu_s": round(led["coordinator_cpu_s"], 3),
              "aggregate_cpu_s": round(cpu, 2), "aggregate_cap_s": 10800, "max_unit_cpu_s": max((v.get("cpu_s") or 0) for v in
                                                                                                led["units"].values()),
              "max_overshoot_s": max((v.get("overshoot_s") or 0) for v in led["units"].values()),
              "cancellations": sum(s in ("CPU_CENSORED", "WALL_CENSORED") for s in stops.values()), "resumes": led["resumes"],
              "started_utc": led["started_utc"], "finished_utc": led.get("finished_utc"), "enforcement_failure":
              led.get("enforcement_failure"), "calls": calls}
    complete = len(stops) == 59 and all(s == "COMPLETED" for s in stops.values()) and all(integ.values()) and \
        calls["relocation_planner"] <= 60 and calls["target_evaluations"] <= 59 and not led.get("enforcement_failure")
    blocked = not all(integ.values()) or bool(led.get("enforcement_failure"))
    decision = "G4C_PILOT_BLOCKED" if blocked else "G4C_PILOT_COMPLETE" if complete else "G4C_PILOT_PARTIAL"
    cf.write_json(out / "unit_evidence.json", ev)
    cf.write_json(out / "pilot_coverage.json", {"tags": j["tags"], "full_gate": j["full_gate"], "roots": {
        k: {kk: vv for kk, vv in r.items() if kk != "bc_pool"} for k, r in j["roots"].items()}, "labels": j["labels"]})
    cf.write_json(out / "bc_pool_coverage.json", {k: r["bc_pool"] for k, r in j["roots"].items()})
    cf.write_json(out / "budget_ledger.json", budget)
    with open(out / "pilot_units.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["unit", "root", "kind", "sequence_or_edit", "stop", "label", "reason", "cpu_s", "wall_s", "reloc_calls",
                    "target_calls", "J_first"])
        for u in plan["units"]:
            e, la = ev[u["unit"]], j["labels"][u["unit"]]
            w.writerow([u["unit"], u["root"], u["kind"], json.dumps(u.get("sequence", u.get("edit"))), e["stop"], la["label"],
                        la.get("reason"), e.get("cpu_s"), e.get("wall_s"), (e.get("calls") or {}).get("relocation_planner"),
                        (e.get("calls") or {}).get("target_evaluations"), (e.get("cost") or {}).get("recomputed_J")])
    with open(out / "pilot_roots.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["root", "family", "units", "positive", "negative", "unknown", "not_run", "empty", "nonempty_positive",
                    "order_contrasts", "decided_scene_pairs", "decided_omissions", "cost_labels", "bc_pool_qualifies", "tags"])
        for k, r in j["roots"].items():
            labs = [j["labels"][u["unit"]]["label"] for u in plan["units"] if u["root"] == k]
            w.writerow([k, r["family"], len(labs), labs.count(P), labs.count(N), labs.count(U), labs.count("NOT_RUN"),
                        r["empty_label"], len(r["nonempty_positive"]), len(r["order_contrasts"]),
                        sum(v["mask"] for v in r["scene"].values()), len(r["decided_omissions"]), len(r["costs"]),
                        r["bc_pool"]["qualifies"], " ".join(j["tags"][k])])
    figures(out, robot, cfg, plan, parent, ev, j, led, budget)
    res = {"stage": "G4C_P1_PILOT", "decision": decision, "full_g4c_readiness_claim": "NO (pilot only; 15 roots not run)",
           "label_counts": j["label_counts"], "tags": j["tags"], "integrity": integ, "budget": {k: v for k, v in budget.items()
                                                                                                if k != "units"},
           "entitlement": entitlement(j, plan), "pilot_plan_hash": plan["plan_hash"], "parent_plan_hash": gp.PARENT_HASH,
           "authorization_hash": plan["authorization_hash"], "judge_cpu_s": round(time.process_time() - c0, 2),
           "judge_wall_s": round(time.perf_counter() - t0, 2), "learned_components": "NONE"}
    cf.write_json(out / "pilot_result.json", res)
    gg.write_manifest(out, {"planner_base": cfg["planner2d"]["seed"], "keys": "pilot_plan.json units"}, cf.config_hash(cfg),
                      {"pilot_plan_hash": plan["plan_hash"], "parent_plan_hash": gp.PARENT_HASH})
    print(json.dumps({"decision": decision, "labels": j["label_counts"], "integrity_ok": all(integ.values()), "calls": calls,
                      "aggregate_cpu_s": budget["aggregate_cpu_s"]}))
    return 0 if decision != "G4C_PILOT_BLOCKED" else 1


def figures(out, robot, cfg, plan, parent, ev, j, led, budget):
    f, ax = plt.subplots(1, 5, figsize=(33, 7.5))
    for a, k in zip(ax, plan["roots"]):
        g4cf._panel(a, robot, cfg, {"roots": plan["roots"]}, k, f"PILOT {k} ({plan['roots'][k]['family']}): initial S0, "
                    f"longest frozen sequence\nempty repair: {j['roots'][k]['empty_label']}")
    f.tight_layout()
    f.savefig(out / "A_initial_scenes.png", dpi=65)
    plt.close(f)
    cats = [(r["threshold"], r) for r in j["full_gate"]]
    f, ax = plt.subplots(figsize=(13, 5.5))
    for i, (name, r) in enumerate(cats):
        ax.barh(i + 0.2, r["pilot_measured"], 0.4, color=COLORS["valid"])
        ax.barh(i - 0.2, r["required"], 0.4, color="white", edgecolor=COLORS["muted"], hatch="//")
        ax.text(max(r["pilot_measured"], r["required"]) + 0.15, i, f"pilot {r['pilot_measured']} | remaining planned "
                f"{r['remaining_planned_capacity']} | {'reachable' if r['full_set_reachable_if_all_remaining_decide'] else 'UNREACHABLE'}",
                va="center", fontsize=7.5)
    ax.set_yticks(range(len(cats)), [c for c, _ in cats], fontsize=8)
    lc = j["label_counts"]
    ax.set_title(f"PILOT coverage (green, 5 roots) vs FUTURE 20-root targets (hatched; NOT_EVALUATED_FULL_SET). Unit labels: "
                 f"{lc.get(P, 0)} POSITIVE, {lc.get(N, 0)} NEGATIVE, {lc.get(U, 0)} UNKNOWN, {lc.get('NOT_RUN', 0)} NOT RUN",
                 fontsize=9, loc="left")
    ax.set(xlabel="independent roots")
    f.tight_layout()
    f.savefig(out / "B_coverage.png", dpi=80)
    plt.close(f)
    units = [u["unit"] for u in plan["units"]]
    c = [budget["units"].get(u, {}).get("cpu_s") or 0 for u in units]
    w = [budget["units"].get(u, {}).get("wall_s") or 0 for u in units]
    f, ax = plt.subplots(2, 1, figsize=(16, 8))
    ax[0].bar(range(59), c, color=[COL[j["labels"][u]["label"]] for u in units], label="unit CPU (colour = label)")
    ax[0].plot(range(59), w, "k.", ms=4, label="unit wall")
    for y, t in ((870, "cancel 870"), (900, "cap 900")):
        ax[0].axhline(y, color=COLORS["invalid"], ls="--", lw=0.8)
        ax[0].text(0, y, t, fontsize=7, va="bottom")
    ax[0].set(yscale="symlog", ylabel="seconds", xticks=range(59))
    ax[0].set_xticklabels(units, rotation=90, fontsize=5)
    ax[0].legend(fontsize=7)
    ax[0].set_title("PILOT compute per unit under the process-tree guard (one worker)", fontsize=9, loc="left")
    ax[1].plot(np.cumsum(c), color=COLORS["hypothetical"], label="cumulative unit CPU")
    ax[1].axhline(10800, color=COLORS["invalid"], ls="--", label="aggregate cap 10,800 CPU-s")
    ax[1].set(ylabel="CPU-s", xlabel="unit (frozen order)", yscale="symlog")
    ax[1].legend(fontsize=7)
    f.tight_layout()
    f.savefig(out / "C_compute.png", dpi=75)
    plt.close(f)
    show = []
    for k, r in j["roots"].items():
        for x in r["order_contrasts"][:1]:
            show += [(f"seq:{k}:{x['positive']}", "ORDER CONTRAST: positive order"), (f"seq:{k}:{x['negative']}",
                                                                                     "ORDER CONTRAST: same endpoints, other order")]
        for kind, x in r["scene"].items():
            if x["mask"] and x["delta"] != 0 and len(show) < 10:
                show += [(f"seq:{k}:empty", "baseline (empty repair, robot at q0)"), (x["unit"], f"scene edit {kind}: Delta={x['delta']}")]
                break
    show = list(dict.fromkeys(show))[:12] or [(units[0], "no decided contrast available")]
    f, ax = plt.subplots((len(show) + 3) // 4, 4, figsize=(26, 7.2 * ((len(show) + 3) // 4)), squeeze=False)
    for a in ax.ravel()[len(show):]:
        a.axis("off")
    for a, (uid, t) in zip(ax.ravel(), show):
        gef.unit_panel(a, robot, cfg, plan, {**ev[uid], "label": j["labels"][uid]["label"] if j["labels"][uid]["label"] != "NOT_RUN"
                                             else "NOT_RUN"}, out, led, f"PILOT {uid}\n{t}")
    f.tight_layout()
    f.subplots_adjust(bottom=0.1)
    f.savefig(out / "D_recorded_evidence.png", dpi=65)
    plt.close(f)
