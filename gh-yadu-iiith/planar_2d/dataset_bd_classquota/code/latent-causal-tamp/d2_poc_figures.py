"""G4C-D2-P1 POC_S judge outputs: revalidated evidence, integrity, Q1-Q6, JSON artifacts and
figures A-E. Reads checkpoints only (g4d_gate.assemble -> g4d_run.unit_evidence); no teacher call.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle

import config as cf
import d2
import d2_poc as dp
import data
import g4d_evidence_figures as gef
import g4d_gate as gg
import g4d_run as gr
import geometry as ge
import motion as mo
import viz
from scene import scene_from_dict
from viz import COLORS, plt

P, N, U = data.POSITIVE, data.NEGATIVE, data.UNKNOWN
COL = {P: COLORS["valid"], N: COLORS["invalid"], U: COLORS["unresolved"], "NOT_RUN": "#bdbdbd", data.CONTRADICTION: "#000000"}
FROZEN_INPUTS = ("runs/g4c_preflight/plan.json", "runs/g4c_pilot/pilot_plan.json", "runs/g4c_pilot/authorization.json",
                 "runs/g4c_pilot/pilot_result.json", "runs/g4d_evidence/final_plan.json", "runs/g4d_evidence/result.json")


def integrity(plan, auth, led, ev, j, out: Path, root: Path) -> dict:
    p0 = json.loads((root / dp.P0 / "manifest.json").read_text())["files"]
    sa = json.loads((root / dp.P0 / "source_audit.json").read_text())["frozen_inputs"]
    calls = accounting(plan, ev)
    return {"sources_unchanged": gr.verify_sources(plan, root) == [],
            "authorization_unchanged": dp._sha(out / "authorization.json") == plan["authorization_file_sha256"]
            and dp.check_authorization(auth) == [],
            "p0_manifest_unchanged": dp._sha(root / dp.P0 / "manifest.json") == auth["p0_manifest_sha256"],
            "p0_artifacts_byte_identical": all(gr._sha(root / dp.P0 / f) == h for f, h in p0.items()),
            "historic_frozen_inputs_unchanged": all(dp._sha(root / f) == sa[f] for f in FROZEN_INPUTS),
            "archived_witnesses_unchanged": all(dp._sha(root / s[3]) == plan["roots"][dp.rkey(s[0], "base")]["archived_witness_sha256"]
                                                for s in d2.BASES if s[0] in dp.FAMILIES),
            "units_run_subset_of_plan": set(led["units"]) <= {u["unit"] for u in plan["units"]},
            "every_unit_has_evidence": set(ev) == {u["unit"] for u in plan["units"]},
            "no_contradiction": data.CONTRADICTION not in j["label_counts"],
            "relocation_cap_respected": calls["relocation_skills"] <= 46, "target_cap_respected": calls["target_evaluations"] <= 70,
            "no_enforcement_failure": not led.get("enforcement_failure")}


def accounting(plan, ev) -> dict:
    by = lambda f, key: sum(((ev[u["unit"]].get("calls") or {}).get(key, 0) or 0) for u in plan["units"] if f(u))  # noqa: E731
    return {"relocation_skills": by(lambda u: True, "relocation_planner"), "target_evaluations": by(lambda u: True, "target_evaluations"),
            "fresh_base_target_evaluations": by(lambda u: u.get("fresh_base"), "target_evaluations"),
            "ordered_relocation_skills": by(lambda u: u["kind"] == "ORDERED_TEACHER" and not u.get("fresh_base"), "relocation_planner"),
            "ordered_target_evaluations": by(lambda u: u["kind"] == "ORDERED_TEACHER" and not u.get("fresh_base"), "target_evaluations"),
            "scene_target_evaluations": by(lambda u: u["kind"] == "SCENE_TARGET_ONLY", "target_evaluations"),
            "robot_start_target_evaluations": by(lambda u: u["kind"] == "ROBOT_START_TARGET_ONLY", "target_evaluations"),
            "home_diagnostics": sum((e.get("target") or {}).get("home_return_status") not in (None, "NOT_RUN") for e in ev.values()),
            "caps": {"relocation_skills": 46, "target_evaluations": 70}}


def run(args) -> int:
    t0, c0, out, root = time.perf_counter(), time.process_time(), Path(args.out), Path(".")
    if (out / "poc_result.json").exists():
        raise SystemExit("POC_S already judged; refusing to overwrite")
    cfg, v2 = cf.load_config(args.config), cf.load_config_v2(args.config_v2)
    rd = lambda n: json.loads((out / n).read_text())  # noqa: E731
    plan, led, auth = rd("final_plan.json"), rd("ledger.json"), rd("authorization.json")
    robot = ge.make_robot(cfg)
    ev = gg.assemble(cfg, v2, robot, plan, led, out)
    j = dp.judge(plan, ev)
    integ = integrity(plan, auth, led, ev, j, out, root)
    calls = accounting(plan, ev)
    frozen = {u: led["units"].get(u, {"stop": "NOT_RUN"})["stop"] for u in plan["frozen_units"]}
    j["questions"]["Q6"] = "MET" if all(integ.values()) else "NOT_MET"
    j["questions"]["Q6_evidence"] = integ
    decision = ("G4C_D2_P1_FAILED" if not all(integ.values()) else
                "G4C_D2_P1_COLLECT_COMPLETE" if all(s == "COMPLETED" for s in frozen.values()) else "G4C_D2_P1_COLLECT_PARTIAL")
    cpu = led["unit_cpu_s"] + led["coordinator_cpu_s"]
    cs = {"units": {k: {x: v.get(x) for x in ("stop", "cpu_s", "wall_s", "overshoot_s", "trigger_cpu_s", "cpu_limit_s", "returncode",
                                               "start_utc", "end_utc")} for k, v in led["units"].items()},
          "unit_cpu_s": round(led["unit_cpu_s"], 2), "coordinator_cpu_s": round(led["coordinator_cpu_s"], 3), "aggregate_cpu_s": round(cpu, 2),
          "aggregate_cap_s": 18000, "max_unit_cpu_s": max([v.get("cpu_s") or 0 for v in led["units"].values()] or [0]),
          "unit_cpu_quantiles_s": np.quantile([v.get("cpu_s") or 0 for v in led["units"].values()] or [0], [0.5, 0.9, 1.0]).round(2).tolist(),
          "max_overshoot_s": max([v.get("overshoot_s") or 0 for v in led["units"].values()] or [0]),
          "cancellations": sum(v["stop"] in ("CPU_CENSORED", "WALL_CENSORED") for v in led["units"].values()),
          "frozen_stops": {s: list(frozen.values()).count(s) for s in set(frozen.values())}, "resumes": led["resumes"],
          "started_utc": led["started_utc"], "finished_utc": led.get("finished_utc"), "calls": calls,
          "enforcement_failure": led.get("enforcement_failure")}
    by_kind = {}
    for u in plan["units"]:
        kk = "FRESH_BASE" if u.get("fresh_base") else u["kind"]
        la = j["labels"][u["unit"]]["label"]
        by_kind.setdefault(kk, {}).setdefault(la, 0)
        by_kind[kk][la] += 1
    cf.write_json(out / "unit_evidence.json", ev)
    cf.write_json(out / "per_root_coverage.json", {k: {kk: vv for kk, vv in r.items() if kk != "bc_pool"} for k, r in j["roots"].items()})
    cf.write_json(out / "bc_pool_coverage.json", {k: {**r["bc_pool"], "substantive": r["substantive_complete_pool"],
                                                      "trivial_complete": r["trivial_complete_pool"], "pool_release": r["pool_release"]}
                                                  for k, r in j["roots"].items()})
    cf.write_json(out / "certificate_breakdown.json", {"by_reason_and_scope": j["negative_kinds"], "trivial": j["trivial_negatives"],
                                                       "nontrivial": j["nontrivial_negatives"], "trivial_kinds": sorted(dp.TRIVIAL),
                                                       "new_certificate_kinds": "NONE (existing checker scopes only)"})
    cf.write_json(out / "compute_summary.json", cs)
    cf.write_json(out / "comparison_effects.json", {k: {"scene": r["scene"], "robot_start": r["robot_start"], "omissions": r["omissions"],
                                                        "nonadditivity": r["nonadditivity"], "order_contrasts": r["order_contrasts"],
                                                        "costs": r["costs"]} for k, r in j["roots"].items()})
    figures(out, robot, cfg, plan, ev, j, led, cs)
    res = {"stage": "G4C_D2_P1_POC_S", "decision": decision, "questions": j["questions"], "label_counts": j["label_counts"],
           "labels_by_kind": by_kind, "no_repair": {k: r["no_repair"] for k, r in j["roots"].items()}, "integrity": integ,
           "compute": {k: v for k, v in cs.items() if k != "units"}, "plan_hash": plan["plan_hash"],
           "authorization_hash": plan["authorization_hash"], "dataset_ready_claim": "NO (two-family development POC)",
           "judge_cpu_s": round(time.process_time() - c0, 2), "judge_wall_s": round(time.perf_counter() - t0, 2),
           "learned_components": "NONE"}
    cf.write_json(out / "poc_result.json", res)
    gg.write_manifest(out, {"planner_base": cfg["planner2d"]["seed"], "fresh_seeds": list(dp.FRESH_SEEDS)}, cf.config_hash(cfg),
                      {"plan_hash": plan["plan_hash"], "p0_manifest_sha256": auth["p0_manifest_sha256"]})
    print(json.dumps({"decision": decision, "labels": j["label_counts"], "by_kind": by_kind, "calls": calls,
                      "questions": {q: v for q, v in j["questions"].items() if len(q) == 2}, "aggregate_cpu_s": cs["aggregate_cpu_s"]}))
    return 0 if decision != "G4C_D2_P1_FAILED" else 1


# ------------------------------------------------------------------ figures (recorded data only)
def _save(f, path, dpi=65):
    f.tight_layout()
    f.savefig(path, dpi=dpi)
    plt.close(f)


def _labels_of(plan, j, k):
    return [j["labels"][u["unit"]]["label"] for u in plan["units"] if u["root"] == k]


def figures(out, robot, cfg, plan, ev, j, led, cs):
    vk = [(f, v, dp.rkey(f, v)) for f in dp.FAMILIES for v in dp.VARIANTS]
    bases = {s[0]: d2.load_base(s, Path(".")) for s in d2.BASES if s[0] in dp.FAMILIES}
    # A: archived verified base motion, inserted blockers, nominal-sweep diagnostic (never a label)
    f, ax = plt.subplots(2, 6, figsize=(42, 15))
    for a, (fam, v, k) in zip(ax.ravel(), vk):
        r, b = plan["roots"][k], bases[fam]
        s = scene_from_dict(r["scene"])
        viz.scene_panel(a, s, cfg, f"{fam} {v}  (root {k})")
        a.set(xlim=viz.LIMITS["grid"][0], ylim=viz.LIMITS["grid"][1])
        viz.draw_robot(a, robot, s.robot.q_rad, s.robot.gripper_gap_m, alpha=0.8, zorder=6)
        a.add_patch(Circle(s.lattice.center(r["goal_cell"]), 0.024, fc="none", ec=COLORS["target_edge"], ls="--", lw=1.5, zorder=9))
        arr = b["arrays"]
        tp = np.array([mo.target_xy(robot, arr, ph, q, b["scene"].get("t0").xy_m) for q, _, ph in mo.samples(arr, 200)])
        a.plot(tp[:, 0], tp[:, 1], color=COLORS["target"], lw=1.5, ls="--", zorder=7, label="ARCHIVED nominal target path (pre-insertion)")
        for q, gap, _ in mo.samples(arr, 8):
            viz.draw_robot(a, robot, q, gap, alpha=0.07, zorder=4)
        for role, c in r["construction_privileged"]["added"].items():
            xy = s.lattice.center(c["cell"])
            a.add_patch(Circle(xy, 0.03, fc="none", ec="#c0392b", lw=2.4, zorder=10))
            a.text(xy[0], xy[1] + 0.036, f"{role}: {c['id']}", fontsize=8, ha="center", color="#c0392b", weight="bold", zorder=11)
        nom = r["construction_privileged"]["nominal"]
        fi = nom.get("first_interruption")
        a.text(0.01, 0.99, f"nominal path blocked: {nom['blocks_nominal_witness']}" + (f" (first: seg {fi['segment']} {fi['pair']})" if fi else "")
               + f"\nNEW no-repair outcome: {j['roots'][k]['no_repair']}\n(nominal collision is a diagnostic, never a label)",
               transform=a.transAxes, va="top", fontsize=8, family="monospace", bbox=dict(fc="white", ec=COLORS["muted"]))
        a.legend(loc="lower right", fontsize=7)
    _save(f, out / "A_base_and_blockers.png", 55)
    # B: outcomes by variant (all planned units in the denominator) + fresh bases
    f, ax = plt.subplots(figsize=(16, 6.5))
    names = [k for _, _, k in vk] + list({u["root"] for u in plan["units"] if u.get("fresh_base")})
    bottom = np.zeros(len(names))
    for la in (P, N, U, "NOT_RUN", data.CONTRADICTION):
        h = np.array([_labels_of(plan, j, k).count(la) for k in names])
        if h.any():
            ax.bar(range(len(names)), h, bottom=bottom, color=COL[la], edgecolor="white", lw=2, label=la)
            for i, (y0, hh) in enumerate(zip(bottom, h)):
                if hh:
                    ax.text(i, y0 + hh / 2, str(hh), ha="center", va="center", fontsize=8, color="white", weight="bold")
            bottom += h
    for i, k in enumerate(names):
        ax.text(i, bottom[i] + 0.15, j["roots"][k]["no_repair"].replace("_", "\n"), ha="center", fontsize=6.5)
    ax.set_xticks(range(len(names)), [n.replace("d2_", "") for n in names], rotation=30, ha="right", fontsize=8)
    ax.set(ylabel="units (planned denominator)", ylim=(0, bottom.max() + 2))
    lc = j["label_counts"]
    ax.set_title(f"POC_S unit outcomes per variant root (text: no-repair outcome). Totals: {lc.get(P, 0)} POSITIVE, {lc.get(N, 0)} "
                 f"NEGATIVE, {lc.get(U, 0)} UNKNOWN, {lc.get('NOT_RUN', 0)} NOT RUN", fontsize=10, loc="left")
    ax.legend(fontsize=8)
    _save(f, out / "B_outcomes_by_variant.png", 80)
    # C: real recorded order/repair evidence only
    show = []
    for k, r in j["roots"].items():
        for x in r["order_contrasts"][:1]:
            show += [(f"seq:{k}:{x['positive']}", "ORDER CONTRAST: witnessed order"), (f"seq:{k}:{x['negative']}", "same endpoints, "
                     f"other order: {x['certificate']}")]
    for k, r in j["roots"].items():
        if r["no_repair"] in ("NEGATIVE", "REROUTABLE_POSITIVE"):
            show.append((f"seq:{k}:empty", f"no repair: {r['no_repair']}"))
            show += [(f"seq:{k}:{s}", "recovery witness on this root") for s in r["nonempty_positive"][:1]]
    show = list(dict.fromkeys(show))[:12] or [(plan["frozen_units"][0], "no order/repair evidence decided")]
    _units_figure(out / "C_order_and_repair_evidence.png", robot, cfg, plan, ev, j, led, show)
    # D: interventions and B/C pool completeness
    f = plt.figure(figsize=(26, 15))
    gs = f.add_gridspec(2, 4, height_ratios=(1, 1.3))
    a = f.add_subplot(gs[0, :])
    ks = [k for _, _, k in vk]
    dec = [j["roots"][k]["bc_pool"].get("decided_slots", 0) for k in ks]
    act = [j["roots"][k]["bc_pool"].get("active_slots", 0) for k in ks]
    a.bar(range(len(ks)), dec, color=COLORS["valid"], edgecolor="white", lw=2, label="active z slots decided (POS/NEG)")
    a.bar(range(len(ks)), np.subtract(act, dec), bottom=dec, color=COLORS["unresolved"], edgecolor="white", lw=2,
          label="active z slots UNKNOWN / NOT RUN")
    for i, k in enumerate(ks):
        r = j["roots"][k]
        a.text(i, act[i] + 0.4, "SUBSTANTIVE COMPLETE" if r["substantive_complete_pool"] else "trivial complete" if r["trivial_complete_pool"]
               else "incomplete", ha="center", fontsize=7, weight="bold" if r["substantive_complete_pool"] else None)
    a.set_xticks(range(len(ks)), [k.replace("d2_", "") for k in ks], rotation=20, ha="right", fontsize=8)
    a.set(ylabel="active B/C slots across the frozen pool")
    a.set_title("B/C pools: every active slot of every declared sequence must be decided; substantive = >=2 decided nonempty repairs "
                "incl. a witnessed POSITIVE", fontsize=10, loc="left")
    a.legend(fontsize=8)
    pairs = [(x["unit"], kk, x, k) for k, r in j["roots"].items() for kk, x in r["scene"].items() if x["mask"]]
    pairs = sorted(pairs, key=lambda t: t[2]["delta"] == 0)[:4]
    for i, (uid, kk, x, k) in enumerate(pairs or [(plan["frozen_units"][0], "none", {"delta": None}, "")]):
        b = f.add_subplot(gs[1, i])
        gef.unit_panel(b, robot, cfg, plan, {**ev[uid], "label": j["labels"][uid]["label"]}, out, led,
                       f"{uid}\nscene edit {kk}: Delta={x['delta']} vs seq:{k}:empty ({j['labels'].get(f'seq:{k}:empty', {}).get('label')})")
    _save(f, out / "D_interventions_and_bc_pools.png", 55)
    # E: compute and cost
    f, ax = plt.subplots(1, 3, figsize=(27, 7))
    us = [u["unit"] for u in plan["units"]]
    c = [cs["units"].get(u, {}).get("cpu_s") or 0 for u in us]
    ax[0].bar(range(len(us)), c, color=[COL[j["labels"][u]["label"]] for u in us])
    for y, t in ((870, "cancel 870"), (900, "cap 900")):
        ax[0].axhline(y, color=COLORS["invalid"], ls="--", lw=0.8)
        ax[0].text(0, y, t, fontsize=7, va="bottom")
    ax[0].set(yscale="symlog", ylabel="unit CPU-s (colour = label)", xlabel="unit (frozen order; first two = fresh bases)")
    ax[1].plot(np.cumsum(c), color=COLORS["hypothetical"], label=f"cumulative unit CPU (total {cs['aggregate_cpu_s']} incl. coordinator)")
    ax[1].axhline(18000, color=COLORS["invalid"], ls="--", label="aggregate cap 18,000 CPU-s")
    k_ = cs["calls"]
    ax[1].set(ylabel="CPU-s", yscale="symlog", xlabel="unit")
    ax[1].set_title(f"calls: relocation skills {k_['relocation_skills']}/46, target evaluations {k_['target_evaluations']}/70, "
                    f"HOME diagnostics {k_['home_diagnostics']}", fontsize=9, loc="left")
    ax[1].legend(fontsize=8)
    cost = [x for r in j["roots"].values() for x in r["costs"] if x["moved"] > 0]
    for x in cost:
        ax[2].scatter(x["endpoint_distance"], x["J_first"], color=COL.get(x["ordered_label"], "#999"), s=50, edgecolor="white", lw=2)
    ax[2].set(xlabel="endpoint displacement C_d (m)", ylabel="first-witness joint travel J (normalized)")
    ax[2].set_title(f"{len(cost)} first relocation witnesses (colour = ordered label; recorded only)", fontsize=9, loc="left")
    _save(f, out / "E_compute_and_cost.png", 70)


def _units_figure(path, robot, cfg, plan, ev, j, led, show):
    n = len(show)
    f, ax = plt.subplots((n + 3) // 4, 4, figsize=(26, 7.2 * ((n + 3) // 4)), squeeze=False)
    for a in ax.ravel()[n:]:
        a.axis("off")
    for a, (uid, t) in zip(ax.ravel(), show):
        gef.unit_panel(a, robot, cfg, plan, {**ev[uid], "label": j["labels"][uid]["label"]}, path.parent, led, f"{uid}\n{t}")
    f.tight_layout()
    f.subplots_adjust(bottom=0.1)
    f.savefig(path, dpi=60)
    plt.close(f)
