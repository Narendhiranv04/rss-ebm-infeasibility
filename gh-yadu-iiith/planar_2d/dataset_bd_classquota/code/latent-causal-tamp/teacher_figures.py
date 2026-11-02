"""G4A follow-up figures (from saved diagnostic records): reporting reconciliation,
per-root budget panel, initial scenes with checked witnesses, witness GIFs. All
scenes are 2D kinematic evidence; no path is drawn for an unresolved case."""
from __future__ import annotations

from pathlib import Path

import numpy as np

import data
import figures as fig
import geometry as ge
import motion as mo
import repair_figures as rf
import skill as sk
import viz
from scene import scene_from_dict
from teacher import ROOTS, SCHEDULE, TIERS
from viz import COLORS, plt


def figures(out: Path, cfg: dict, plan: dict, trace: list, recon: dict) -> None:
    fill = {data.POSITIVE: COLORS["valid"], data.UNKNOWN: COLORS["unresolved"], "NOT_RUN": "#bdbdbd"}
    f, ax = plt.subplots(1, 2, figsize=(15, 4.8))
    c, a = recon["certificates"], recon["primary_target_calls"]
    combos = sorted(c["certificate_combinations"].items(), key=lambda x: -x[1])
    ax[0].barh(range(len(combos)), [v for _, v in combos], color=COLORS["invalid"])
    for i, (k, v) in enumerate(combos):
        ax[0].text(v + 0.5, i, f"{k}: {v}", va="center", fontsize=8)
    ax[0].set_yticks([])
    ax[0].set_title(f"{c['accepted_records']} accepted certificate records -> {c['unique_negative_assignments']} unique negative "
                    f"assignments\n(duplicates {c['duplicate_records']}; {c['assignments_with_two_or_more_distinct_certificates']} "
                    "assignments carry two distinct certificates)", fontsize=9, loc="left")
    cats = [("completed repair, 1 target call", a["completed_with_exactly_one_call"]), ("no repair witness, 0 calls",
            a["no_repair_with_zero_calls"]), ("aborted", a["aborted_episodes"]), ("calls before commit", a["calls_before_commit"]),
            ("calls between relocations", a["calls_between_relocations"])]
    ax[1].barh(range(len(cats)), [v for _, v in cats], color=COLORS["hypothetical"])
    for i, (k, v) in enumerate(cats):
        ax[1].text(v + 0.2, i, f"{k}: {v}", va="center", fontsize=8)
    ax[1].set_yticks([])
    ax[1].set_title(f"G4A primary target calls from traces ({a['episodes']} episodes; {a['all_stay_commits']} all-stay commits)",
                    fontsize=9, loc="left")
    f.tight_layout()
    f.savefig(out / "A_reporting_reconciliation.png", dpi=90)
    plt.close(f)
    f, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    for i, q in enumerate(ROOTS):
        cells = [("B0", plan["cases"][i]["baseline"]["first_pass"]["status"], "archived first pass: "
                  + plan["cases"][i]["baseline"]["first_pass"]["reason"].split(":")[1])]
        for t, _ in TIERS:
            e = next(x for x in trace if x["query_id"] == q and x["tier"] == t)
            txt = e["stop"] if e["status"] == "NOT_RUN" else f"{e['status'][:10]} | {e['stop']} | {e['cpu_s']} CPU-s"
            cells.append((t, e["status"], txt))
        for j, (t, st, txt) in enumerate(cells):
            ax.add_patch(plt.Rectangle((j * 4.2, -i), 4.0, 0.85, fc=fill.get(st, "white"), ec="white"))
            ax.text(j * 4.2 + 2.0, -i + 0.42, f"{t}: {txt}", ha="center", va="center", fontsize=7.5,
                    color="white" if st != "NOT_RUN" else COLORS["text"])
        ax.text(-0.2, -i + 0.42, q, ha="right", va="center", fontsize=9)
    ax.set(xlim=(-1.5, 12.6), ylim=(-len(ROOTS) + 0.8, 1.1))
    ax.set_title(f"{SCHEDULE}: all-stay target evaluation per tier (B1 = 2x, B2 = 4x node and failure allowances; "
                 "UNRESOLVED = UNKNOWN; censoring is not a budget result)", fontsize=9, loc="left")
    f.tight_layout()
    f.savefig(out / "B_budget_panel.png", dpi=90)
    plt.close(f)
    robot = ge.make_robot(cfg)
    f, axes = plt.subplots(1, len(ROOTS), figsize=(6.2 * len(ROOTS), 6.6), squeeze=False)
    for ax, case in zip(axes[0], plan["cases"]):
        sc = scene_from_dict(case["scene"])
        npz = sorted(out.glob(f"eval_{case['query_id']}_B*.npz"))
        title = f"{case['query_id']}: initial scene, all-stay (2D KINEMATIC EVIDENCE)"
        if npz:
            arr = dict(np.load(npz[0]))
            fig.keyframe_panel(ax, robot, sc, cfg, arr, f"{title}\nchecked witness {npz[0].stem} through retreat",
                               sk.VERIFIED, np.asarray(case["action"]["goal_xy_m"]))
        else:
            viz.scene_panel(ax, sc, cfg, f"{title}\nno witness recorded: UNKNOWN (not an impossibility proof)")
            viz.draw_robot(ax, robot, sc.robot.q_rad, sc.robot.gripper_gap_m, alpha=0.85, zorder=8)
            last = [e for e in trace if e["query_id"] == case["query_id"] and e.get("evaluator")]
            cens = [e for e in trace if e["query_id"] == case["query_id"] and e["stop"].endswith("CENSORED")]
            if cens and not last:
                rf._box(ax, [f"{e['tier']}: {e['stop']} at {e['cpu_s']} CPU-s (trigger {e['trigger_cpu_s']}), active "
                             f"phase {e['active_phase']}" for e in cens] + ["no evaluator record kept (censored)"])
            if last:
                r = last[-1]["evaluator"]
                rf._box(ax, [f"last run {last[-1]['tier']}: {r['status']} {r['reason']}", f"picks OK {r['picks_ok']}/48, places OK "
                             f"{r['places_ok']}/48", f"connector queries {len(r['queries'])}"])
        ax.add_patch(plt.Circle(case["action"]["goal_xy_m"], 0.024, fc="none", ec=COLORS["target_edge"], ls="--", lw=1.5))
        rf._ids(ax, sc, skip=("t0",))
        rf._view(ax)
    f.tight_layout()
    f.savefig(out / "C_scene_evidence.png", dpi=80)
    plt.close(f)
    for npz in sorted(out.glob("eval_*_B*.npz")):
        q = "_".join(npz.stem.split("_")[1:3])
        case = next(c for c in plan["cases"] if c["query_id"] == q)
        sc, arr = scene_from_dict(case["scene"]), dict(np.load(npz))
        frames = mo.samples(arr, 70)
        g, ax = plt.subplots(figsize=(7.0, 7.4))

        def draw(k, sc=sc, arr=arr, frames=frames, q=q, case=case, ax=ax):
            ax.clear()
            fig.task_frame(ax, robot, sc, cfg, arr, frames, min(k, len(frames) - 1), f"{q} {npz.stem.split('_')[-1]}: all-stay target "
                           "witness (2D kinematic)", sk.VERIFIED, np.asarray(case["action"]["goal_xy_m"]))
            rf._ids(ax, sc, skip=("t0",))
            rf._view(ax)
        fig.write_gif(g, draw, len(frames) + 8, out / f"D_witness_{q}.gif", ms=80)
