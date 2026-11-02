"""G3.5 figures A-F (Guide v1.7 §5, §11) from recorded states and trajectories only.

Proposed endpoints are dashed arrows (proposals, not motion). Relocation and target
motions are replayed from stored records; a missing witness is drawn as text
(UNRESOLVED / NOT_RUN), never as an invented path. Frozen-path panels are labelled
privileged diagnostics. Clean learner images are written separately (no overlays).
"""
from __future__ import annotations

import csv
import json
from multiprocessing import Pool
from pathlib import Path
import config as cf
import geometry as ge

import numpy as np
from matplotlib.patches import Circle
from PIL import Image

import cases
import data
import evaluate as ev
import figures as fig
import intervention_figures as ifig
import motion as mo
import repair_figures as rf
import skill as sk
import proposals as pr
import verify as vf
import viz
from viz import COLORS, plt

SHORT = {sk.VERIFIED: "VERIFIED", sk.INFEASIBLE: "CERT_INFEASIBLE", sk.UNRESOLVED: "UNRESOLVED", mo.VALID: "VALID",
         mo.INVALID: "INVALID", None: "-"}
TAGS = {"SUCCESS": "VERIFIED_FEASIBLE", "TARGET_UNRESOLVED": "UNRESOLVED", "TARGET_CERTIFIED_INFEASIBLE": "CERTIFIED_INFEASIBLE",
        "NO_REPAIR_WITNESS_WITHIN_BUDGET": "UNRESOLVED", "ABORTED": "UNRESOLVED"}


def _goal(ax, r):
    act = r["query"]["learner"]["action"]
    ax.add_patch(Circle(act["goal_xy_m"], 0.024, fc="none", ec=COLORS["target_edge"], ls="--", lw=1.6, zorder=9))
    ax.text(act["goal_xy_m"][0], act["goal_xy_m"][1] + 0.032, "GOAL", fontsize=6, ha="center", color=COLORS["target_edge"],
            weight="bold", zorder=10)


def _proposal(ax, scene, cells, color, lw=1.6, alpha=1.0):
    for o, d in cells.items():
        rf._arrow(ax, scene.get(o).xy_m, pr.endpoint_xy(scene, d), color, ls="--", lw=lw, alpha=alpha)


def case_panel(ax, robot, cfg, r):
    """Initial scene, goal, committed proposal and every audited candidate's statuses."""
    sc, rec = r["scene"], cases.record(r)
    viz.scene_panel(ax, sc, cfg, f"{r['name']}  [{', '.join(r['fx']['cases'])}]")
    viz.draw_robot(ax, robot, sc.robot.q_rad, sc.robot.gripper_gap_m, alpha=0.85, zorder=8)
    rf._view(ax)
    rf._ids(ax, sc)
    _goal(ax, r)
    c = r["primary"]["commit"]
    if c:
        _proposal(ax, sc, c["cells"], COLORS["hypothetical"])
    b = rec["bounds"]
    lines = [f"primary: {rec['execution']['outcome']}  committed: {c['label'] if c else 'none'}",
             f"bounds: LB {b['lower_bound']}  UB {b['upper_bound']}  {b['minimality']}",
             "audit: candidate | static | reloc | target | assignment"]
    for a in rec["offline_case_audit"]:
        o = a["orders"][-1] if a["orders"] else None
        lines.append(f" {a['label'][:24]:24s}|{SHORT[a['static_status']]:10s}|{SHORT[o and o['relocation_status']]:10s}"
                     f"|{SHORT[o and o['target'] and o['target']['status']]:15s}|{SHORT[a['assignment_feasibility_status']]}")
    rf._box(ax, lines, loc=(0.0, -0.09))
    viz.status_tag(ax, TAGS[rec["execution"]["outcome"]], "primary one-shot episode")


def protocol_cases(robot, cfg, res):
    names = sorted(res)
    f, axes = plt.subplots(2, 3, figsize=(21, 18), squeeze=False)
    for ax, n in zip(axes.flat, names):
        case_panel(ax, robot, cfg, res[n])
    ax = axes.flat[len(names)]
    ax.axis("off")
    lines = ["R6 controls and leakage audit (recorded):"]
    for n in names:
        rec = cases.record(res[n])
        lines.append(f"{n}: learner tensors unchanged after poison/delete = "
                     f"{rec['leakage_audit']['tensors_unchanged_after_poison_and_delete']}")
        for a in rec["offline_case_audit"]:
            if a["certificate"] == "STATIC_ENDPOINT_INVALID" or a["unknown_reason"] == "STATIC_GRAY_BAND":
                fail = a["static_failed"][0]
                lines.append(f"   {a['label']}: static {a['static_status']} ({fail['factor'] if isinstance(fail, dict) else fail})")
        if rec.get("controls"):
            t = rec["controls"]["planner_exhaustion"]["target"]
            lines.append(f"   forced planner exhaustion {rec['controls']['planner_exhaustion']['overrides']}:")
            lines.append(f"      target {t['status']} {t['reason']} -> assignment UNRESOLVED (no negative, no bound)")
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=8.5, family="monospace", transform=ax.transAxes)
    f.suptitle("G3.5 cases (INITIAL_SCENE_ONE_SHOT_V1): blue dashed = committed proposal (endpoints, not motion); "
               "orange dashed = unique goal", fontsize=11, x=0.01, ha="left")
    f.tight_layout(rect=(0, 0.02, 1, 0.98))
    return f


def target_panel(ax, robot, cfg, scene, tgt, title, act):
    """Keyframes of a recorded target skill from its achieved scene, or the missing-witness text."""
    res = tgt and tgt["result"]
    if res is not None and res.traj is not None:
        fig.keyframe_panel(ax, robot, scene, cfg, res.traj.arrays(), title, res.status, np.asarray(act["goal_xy_m"]))
    else:
        viz.scene_panel(ax, scene, cfg, title)
        ax.text(0.5, 0.5, f"no target trajectory\n{tgt['status'] if tgt else 'NOT_RUN'} {tgt['reason'] if tgt else ''}",
                transform=ax.transAxes, ha="center", fontsize=10, bbox=dict(fc="white", ec=COLORS["muted"]))
        viz.status_tag(ax, tgt["status"] if tgt else "UNRESOLVED", "target evaluation")
    rf._ids(ax, scene, skip=("t0",))
    rf._view(ax)


def r1_route(robot, cfg, r):
    fr, snap = r["frozen"]
    f, axes = plt.subplots(1, 2, figsize=(14, 7.6))
    ifig.snapshot_panel(axes[0], robot, cfg, fr, snap, f"{r['name']}: PRIVILEGED DIAGNOSTIC (FROZEN_CONTROL)\n"
                        f"stored gamma_0 {fr.action_hash[:23]} replayed unchanged: collides", [
                            i for i, c in snap.checks.items() if c.status == mo.INVALID])
    rf._view(axes[0])
    p = r["primary"]
    target_panel(axes[1], robot, cfg, r["scene"], p["target"], f"{r['name']}: primary episode, all-stay committed\n"
                 f"newly planned target action from the same initial state ({p['outcome']})", r["query"]["learner"]["action"])
    _goal(axes[1], r)
    f.tight_layout()
    return f


def r2_gif(robot, cfg, r, path, n=45):
    p, act = r["primary"], r["query"]["learner"]["action"]
    w, s0 = p["commit"]["attempt"], ev_start(r)
    occ = pr.goal_blockers(r["scene"], {}, act, cfg)
    head = f"goal occupied by {', '.join(occ)}" if occ else "goal free"
    seq = [lambda ax: (viz.scene_panel(ax, r["scene"], cfg, f"{r['name']}: initial scene, {head}"),
                       viz.draw_robot(ax, robot, s0.q, s0.gap, zorder=8),
                       rf._ids(ax, r["scene"]), _goal(ax, r), _proposal(ax, r["scene"], p["commit"]["cells"],
                                                                        COLORS["hypothetical"]), rf._view(ax))] * 8
    state = s0
    for (oid, cell), res, nxt in zip(w.order, w.skills, w.states):
        arr, fr = res.traj.arrays(), mo.samples(res.traj.arrays(), n)
        seq += [lambda ax, i=i, sc=state.scene, o=oid, c=cell, a=arr, fr=fr, m=nxt.moved: (
            rf.motion_frame(ax, robot, cfg, sc, o, c, a, fr, i, f"committed relocation {o} -> cell {c} (from S0)", m),
            _goal(ax, r), viz.status_tag(ax, "VERIFIED_FEASIBLE", "certified relocation")) for i in range(n)]
        state = nxt
    tr = p["target"]["result"].traj.arrays()
    fr = mo.samples(tr, n)
    seq += [lambda ax, i=i: (fig.task_frame(ax, robot, state.scene, cfg, tr, fr, i, "newly planned target action from "
                             f"the achieved state  H = [{', '.join(state.moved)}]", p["target"]["status"],
                             np.asarray(act["goal_xy_m"])), rf._ids(ax, state.scene, skip=("t0",)), rf._view(ax))
            for i in range(n)]
    f, ax = plt.subplots(figsize=(7.0, 7.4))

    def draw(k):
        ax.clear()
        seq[min(k, len(seq) - 1)](ax)
        f.tight_layout()
    fig.write_gif(f, draw, len(seq) + 8, path, ms=80)


def ev_start(r):
    return ev.start(r["scene"], r["query"])


def witness_rows(robot, cfg, r, rows, title):
    """One row per audited order: relocation panels (or the rejection) then the target."""
    cols = max(len(o["attempt"].order) for _, o in rows) + 1
    f, axes = plt.subplots(len(rows), cols, figsize=(5.0 * cols, 5.5 * len(rows)), squeeze=False)
    act, s0 = r["query"]["learner"]["action"], ev_start(r)
    for i, (label, o) in enumerate(rows):
        att, state = o["attempt"], s0
        for k, ((oid, cell), res) in enumerate(zip(att.order, att.skills)):
            head = f"{label}\nstep {k + 1}/{len(att.order)}"
            if k < len(att.states):
                rf.relocation_panel(axes[i, k], robot, cfg, state.scene, oid, cell, res, head, att.states[k].moved)
                state = att.states[k]
            else:
                rf.failure_panel(axes[i, k], robot, cfg, state, oid, cell, res, head)
            _goal(axes[i, k], r)
        for ax in axes[i, len(att.skills):len(att.order)]:          # steps never reached: no motion exists
            ax.axis("off")
        last = axes[i, len(att.order)]
        if att.status == vf.RELOCATED:
            st = vf.final_state(s0, att)
            target_panel(last, robot, cfg, st.scene, o["target"], f"target from this order's achieved state\n"
                         f"start q = [{', '.join(f'{v:.2f}' for v in st.q)}] rad", act)
        else:
            last.axis("off")
            last.text(0.05, 0.5, f"order rejected at relocation {att.stop + 1}:\n{att.reason}\n(this order only; "
                      "not an assignment-level negative)\ntarget evaluation: NOT_RUN", fontsize=10, transform=last.transAxes)
        for ax in axes[i, len(att.order) + 1:]:
            ax.axis("off")
    f.suptitle(title, fontsize=12, x=0.01, ha="left")
    f.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (5.5 * len(rows))))
    return f


def call_trace(res):
    names = sorted(res)
    f, ax = plt.subplots(figsize=(16, 1.1 * len(names) + 1.8))
    colour = {"proposal": COLORS["muted"], "static_screen": "#7c5cbf", "relocation_planner": COLORS["hypothetical"],
              "commitment": COLORS["text"], "executor": COLORS["valid"], "target_evaluator": COLORS["target"],
              "episode": COLORS["obstacle_edge"]}
    for y, n in enumerate(names):
        ev_ = res[n]["primary"]["trace"].events
        t_end = max(e["t_s"] for e in ev_) or 1.0
        for k, e in enumerate(ev_):
            x = k
            ax.plot(x, y, "o", ms=11, color=colour[e["component"]], zorder=3)
            ax.text(x, y + 0.22, e["kind"].replace("_", "\n"), fontsize=6.5, ha="center", va="bottom")
            ax.text(x, y - 0.25, f"{e['t_s']:.1f}s", fontsize=6, ha="center", va="top", color=COLORS["muted"])
        c = next(k for k, e in enumerate(ev_) if e["kind"] in ("COMMIT", "FINISH"))
        ax.axvline(c + 0.5, ymin=(y + 0.1) / len(names), ymax=(y + 0.9) / len(names), color=COLORS["text"], ls="--", lw=1)
        kinds = [e["kind"] for e in ev_]
        before = kinds[:kinds.index("COMMIT")].count("TARGET_EVALUATE") if "COMMIT" in kinds else 0
        ax.text(len(ev_) + 0.3, y, f"target calls: before commit {before}, between relocations 0, total "
                f"{kinds.count('TARGET_EVALUATE')} | boundary ok = {cases._boundary_ok(ev_)} | {t_end:.1f} s",
                va="center", fontsize=8)
    ax.set_yticks(range(len(names)), names, fontsize=8)
    ax.set_xticks([])
    ax.set_xlim(-0.6, 22)
    ax.set_ylim(-0.7, len(names) - 0.2)
    ax.set_title("Primary episode event traces (ordered; dashed line = commitment boundary). Colours: grey proposal, "
                 "purple static screen, blue relocation planner, black commit, green execution, orange target "
                 "evaluator", fontsize=9, loc="left")
    f.tight_layout()
    return f


def g35_figures(out, robot, cfg, res) -> None:
    save = lambda f, name, dpi=80: (f.savefig(out / name, dpi=dpi), plt.close(f))  # noqa: E731
    save(protocol_cases(robot, cfg, res), "A_protocol_cases.png", 70)
    r1 = next((r for r in res.values() if "R1" in r["fx"]["cases"] and r["frozen"]), None)
    if r1:
        save(r1_route(robot, cfg, r1), "B_r1_route_comparison.png")
    r2 = next((r for r in res.values() if r["fx"]["cases"][0] == "R2" and r["primary"]["outcome"] == "SUCCESS"), None)
    if r2:
        r2_gif(robot, cfg, r2, out / "C_r2_repair.gif")
        alts = [(f"{a['label']} (assignment {a['status']})", a["orders"][-1]) for a in r2["audit"]
                if a["status"] == sk.VERIFIED]
        save(witness_rows(robot, cfg, r2, alts, f"{r2['name']}: independently witnessed alternative repairs "
                          "(OFFLINE_CASE_AUDIT; neither assumed minimal by the audit)"), "D_r3_alternatives.png")
    r4 = next((r for r in res.values() if "R4" in r["fx"]["cases"]), None)
    if r4:
        a = max(r4["audit"], key=lambda a: len(a["orders"]))
        save(witness_rows(robot, cfg, r4, [(f"order {i + 1}", o) for i, o in enumerate(a["orders"])],
                          f"{r4['name']}: same assignment {a['label']}, two orders (OFFLINE_CASE_AUDIT)"), "E_r4_orders.png")
    save(call_trace(res), "F_call_trace.png", 90)
    for n, r in res.items():                                       # clean learner observations (no overlays)
        Image.fromarray(data.learner_tensors(r["query"], cfg["render"]["image_px"])["rgb"]).save(out / f"obs_{n}.png")


# ------------------------------------------------------------------ G3.5 command (moved from run.py at G4A)
def cmd_case_analysis(args) -> int:
    """R0-R6 under INITIAL_SCENE_ONE_SHOT_V1; the work lives in cases/evaluate (thin I/O here)."""
    cfg = cf.load_config(args.config)
    fixtures = cf.validate_fixtures(json.loads(Path(args.fixtures).read_text())["fixtures"])
    queries = {k: fx for k, fx in fixtures.items() if fx["stage"] == "G3_5"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    robot, chash, rhash = ge.make_robot(cfg), cf.config_hash(cfg), cf.robot_profile_hash(cfg)
    with Pool(len(queries)) as pool:
        res = {r["name"]: r for r in pool.imap_unordered(cases.run_query, [(cfg, k, fx) for k, fx in queries.items()])}
    records = {n: cases.record(r) for n, r in sorted(res.items())}
    tasks, scenes, arrays = cases.archive(res, rhash)
    g35_figures(out, robot, cfg, res)
    gate = cases.readiness(res)
    cf.write_json(out / "config.json", {"config_hash": chash, "config": cfg})
    cf.write_json(out / "cases.json", records)
    cf.write_json(out / "scenes.json", scenes)
    np.savez_compressed(out / "arrays.npz", **arrays)
    rows = cases.summary_rows(records)
    cf.write_json(out / "result.json", {
        "stage": "G3.5", "protocol": cfg["protocol"]["name"], "readiness": gate, "summary": rows, "tasks": tasks,
        "config_hash": chash, "robot_profile_hash": rhash, "proposal_source": cfg["proposal"]["source"],
        "verification_scope": "KINEMATIC_3R_RELOCATIONS_AND_NEWLY_PLANNED_TARGET_SKILL (no dynamics, no simulation)",
        "simulation_status": "NOT_RUN", "hardware_status": "NOT_RUN", "learned_components": "NONE"})
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    cf.write_manifest(out, {"planner_base": cfg["planner2d"]["seed"], "per_query": "crc32(query/component key)"}, chash,
                    {"queries": {n: r["hashes"] for n, r in records.items()}})
    print(json.dumps({"out": str(out), "readiness": gate, "summary": rows}))
    return 0
