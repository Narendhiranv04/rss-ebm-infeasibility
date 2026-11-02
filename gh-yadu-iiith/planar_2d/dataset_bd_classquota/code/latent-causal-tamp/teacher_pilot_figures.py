"""G4C-LF-P4 figures from recorded data only: labels by source, compute per seed attempt, and
the two development scenes (full 3R arm at q0, physical IDs, recorded witnesses only)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import g4d_evidence_figures as gef
import operational_teacher as ot
import viz
from scene import scene_from_dict
from viz import COLORS, plt

COL = {ot.VERIFIED: COLORS["valid"], ot.CERTIFIED: COLORS["invalid"], ot.BUDGET: "#8e44ad", ot.UNKNOWN: COLORS["unresolved"],
       "FATAL_CONTRADICTION": "#000000"}


def figures(out: Path, robot, cfg, plan, records, att):
    qs = [q["query"] for q in plan["queries"]]
    f, ax = plt.subplots(figsize=(15, 6.5))
    for i, q in enumerate(qs):
        r = records[q]
        ax.barh(i, 1, color=COL[r["teacher_status"]], edgecolor="white", lw=2)
        ax.text(1.02, i, f"{r['teacher_status']} [{r['label_source']}]  y={r['binary_training_label']}  seeds run "
                f"{r['seeds_run']}/{r['seeds_declared']}  {r['reason'][:70]}", va="center", fontsize=7.5)
    ax.set_yticks(range(len(qs)), qs, fontsize=8)
    ax.set(xlim=(0, 4.2), xticks=[])
    ax.invert_yaxis()
    for k, c in COL.items():
        ax.barh([], [], color=c, label=k)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("P4 teacher status per query under 2D-ABC-OPNEG-v1 (BUDGET_INFEASIBLE = operational, policy-relative; "
                 "never a proof)", fontsize=9, loc="left")
    f.tight_layout()
    f.savefig(out / "labels_by_source.png", dpi=80)
    plt.close(f)
    f, ax = plt.subplots(1, 2, figsize=(18, 5.5))
    tc = {}
    for x in att:
        tc.setdefault(x["terminal"], COLORS["hypothetical"] if x["terminal"] == "WITNESS" else "#8e44ad" if x["clean"] else
                      COLORS["unresolved"])
    ax[0].bar(range(len(att)), [x["cpu_s"] or 0 for x in att], color=[tc[x["terminal"]] for x in att])
    ax[0].axhline(250, color=COLORS["invalid"], ls="--", lw=0.8)
    ax[0].set_xticks(range(len(att)), [f"{x['query'].split(':', 1)[1][:22]} s{x['seed'] % 100}" for x in att], rotation=90, fontsize=6)
    ax[0].set(ylabel="CPU-s per seed attempt (process tree)", yscale="symlog")
    ax[0].set_title("seed attempts (blue witness, purple clean terminal, orange not clean); dashed: 250 CPU-s seed guard",
                    fontsize=8, loc="left")
    ax[1].plot(np.cumsum([x["cpu_s"] or 0 for x in att]), color=COLORS["hypothetical"])
    ax[1].axhline(3600, color=COLORS["invalid"], ls="--", label="aggregate cap 3,600 CPU-s")
    ax[1].set(ylabel="cumulative CPU-s", xlabel="seed attempt (execution order)")
    ax[1].legend(fontsize=8)
    f.tight_layout()
    f.savefig(out / "compute.png", dpi=80)
    plt.close(f)
    show = []
    for q in qs:
        w = next((x for x in records[q]["attempts"] if x["terminal"] == "WITNESS"), None)
        if w and len(records[q]["query"]["sequence"]) == 2:
            show.append((q, w))
    f, axs = plt.subplots(1, 2 + len(show), figsize=(7 * (2 + len(show)), 7.5), squeeze=False)
    for a, k in zip(axs[0], plan["roots"]):
        s = scene_from_dict(plan["roots"][k]["scene"])
        viz.scene_panel(a, s, cfg, f"{k}: S0, robot at q0 (full 3R arm)")
        viz.draw_robot(a, robot, s.robot.q_rad, s.robot.gripper_gap_m, alpha=0.7, zorder=6)
        lines = [f"{q.split(':', 2)[0]}:{q.split(':', 2)[2]} -> {records[q]['teacher_status']}" for q in qs if records[q]["query"]["root"] == k]
        a.text(0.01, 0.99, "\n".join(lines), transform=a.transAxes, va="top", fontsize=6.5, family="monospace",
               bbox=dict(fc="white", ec=COLORS["muted"]))
    for a, (q, w) in zip(axs[0][2:], show):
        sd = Path(out) / f"seed{w['seed']}"
        led = json.loads((sd / "ledger.json").read_text())
        e = {"unit": w["unit"], "root": records[q]["query"]["root"], "sequence": records[q]["query"]["sequence"],
             "label": "VERIFIED_FEASIBLE", "stop": w["stop"], "cpu_s": w["cpu_s"], "relocation": None, "target": None,
             "certificates": [], "evidence": [{"kind": "complete_witness"}], "checks": {}}
        gef.unit_panel(a, robot, cfg, plan, e, sd, led, f"{q} (seed {w['seed']}): RECORDED witness")
    f.tight_layout()
    f.savefig(out / "scenes_and_witnesses.png", dpi=60)
    plt.close(f)
