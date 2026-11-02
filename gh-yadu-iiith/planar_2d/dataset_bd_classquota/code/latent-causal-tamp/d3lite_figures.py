"""P5 D3-lite figures from the frozen plan and recorded evidence only (holdout outcomes sealed)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle

import g4d_evidence_figures as gef
import operational_teacher as ot
import viz
from scene import scene_from_dict
from viz import COLORS, plt

COL = {ot.VERIFIED: COLORS["valid"], ot.CERTIFIED: COLORS["invalid"], ot.BUDGET: "#8e44ad", ot.UNKNOWN: COLORS["unresolved"],
       "NOT_RUN": "#d0d0d0", "FATAL_CONTRADICTION": "#000000", "SEALED": "#ffffff"}
ROLES = ["reference", "original", "single", "single ", "order_AB", "order_BA", "alternative", "alternative ", "singleton:A",
         "singleton:B", "joint:A+B", "nuisance"]


def _role_order(plan, k):
    qs = [q for q in plan["queries"] if q["lineage"] == k]
    seen, out = {}, []
    for q in qs:
        r = q["role"] + (" " if q["role"] in seen else "")
        seen[q["role"]] = 1
        out.append((r, q["query"]))
    return dict(out)


def figures(out: Path, robot, cfg, plan, records):
    obs = [k for k, v in plan["roots"].items() if v["queries"]]
    f, axs = plt.subplots(2, 4, figsize=(28, 14))
    for ax, k in zip(axs.ravel(), obs):
        r = plan["roots"][k]
        s = scene_from_dict(r["scene"])
        viz.scene_panel(ax, s, cfg, f"{k} [{r['family']}, {r['split']}]: obstructed S0, robot at q0")
        viz.draw_robot(ax, robot, s.robot.q_rad, s.robot.gripper_gap_m, alpha=0.7, zorder=6)
        lab = {r["roles"]["A"]: "A", r["roles"]["B"]: "B", r["roles"]["N"]: "N"}
        for o in s.objects:
            if o.id in lab:
                ax.text(o.xy_m[0], o.xy_m[1] + 0.033, lab[o.id], ha="center", fontsize=9, weight="bold", color="#c0392b")
        g = s.lattice.center(r["goal_cell"])
        ax.add_patch(Circle(g, 0.024, fc="none", ec=COLORS["target_edge"], ls="--", lw=1.5))
        for p in r["queries"]["pool"] + [{"sequence": [list(x) for x in q["edit"].items()]} for q in r["queries"]["scene_queries"]]:
            for oid, c in p["sequence"]:
                ax.add_patch(Circle(s.lattice.center(c), 0.024, fc="none", ec=COLORS["hypothetical"], ls=":", lw=1.0, zorder=3))
        ax.text(0.01, 0.01, "dotted rings: predeclared destinations (ghosts, not executed)", transform=ax.transAxes, fontsize=7)
    f.tight_layout()
    f.savefig(out / "A_roots_and_catalogues.png", dpi=55)
    plt.close(f)
    f, ax = plt.subplots(figsize=(16, 6))
    for i, k in enumerate(obs):
        sealed = plan["roots"][k]["split"] == "holdout"
        for j, (role, q) in enumerate(_role_order(plan, k).items()):
            st = "SEALED" if sealed else records[q]["teacher_status"]
            ax.add_patch(plt.Rectangle((j, i), 0.95, 0.9, fc=COL[st], ec="#555", hatch="//" if st in ("UNKNOWN", "NOT_RUN") else None))
            if not sealed and records[q]["teacher_status"] == ot.BUDGET:
                ax.text(j + 0.47, i + 0.45, "op", ha="center", va="center", fontsize=7, color="white")
    ax.set(xlim=(0, 12), ylim=(len(obs), 0))
    ax.set_xticks(np.arange(12) + 0.47, [r.strip() for r in ROLES], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(obs)) + 0.45, [f"{k} ({plan['roots'][k]['split']})" for k in obs], fontsize=8)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc=c, ec="#555", label=st) for st, c in COL.items()], fontsize=7, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("D3-lite outcomes (green witness, red certificate, purple 'op' = BUDGET_INFEASIBLE operational negative, "
                 "hatched = UNKNOWN/NOT_RUN mask; holdout rows SEALED)", fontsize=9, loc="left")
    f.tight_layout()
    f.savefig(out / "B_outcome_grid.png", dpi=80)
    plt.close(f)
    show = []
    for q, r in records.items():
        w = next((x for x in r["attempts"] if x["terminal"] == "WITNESS"), None)
        if w and len(r["query"]["sequence"]) == 2 and r["query"]["split"] != "holdout":
            show.append((q, w, r))
    show = show[:8]
    if show:
        f, axs = plt.subplots(1, len(show), figsize=(6.5 * len(show), 7), squeeze=False)
        for a, (q, w, r) in zip(axs[0], show):
            sd = next(Path(out).glob(f"b*b/seed{w['seed']}/ledger.json"), None)
            for p in Path(out).glob(f"b*b/seed{w['seed']}/ledger.json"):
                if w["unit"] in json.loads(p.read_text())["units"]:
                    sd = p
            led = json.loads(sd.read_text())
            e = {"unit": w["unit"], "root": r["query"]["root"], "sequence": r["query"]["sequence"], "label": ot.VERIFIED,
                 "stop": w["stop"], "cpu_s": w["cpu_s"], "relocation": None, "target": None, "certificates": [],
                 "evidence": [{"kind": "complete_witness"}], "checks": {}}
            gef.unit_panel(a, robot, cfg, plan, e, sd.parent, led, f"{q}\nRECORDED witness (seed {w['seed']})")
        f.tight_layout()
        f.savefig(out / "C_recorded_two_move_witnesses.png", dpi=55)
        plt.close(f)
    att = [a for r in records.values() for a in r["attempts"]]
    f, ax = plt.subplots(figsize=(16, 4.5))
    ax.bar(range(len(att)), [a["cpu_s"] or 0 for a in att], color=[COLORS["hypothetical"] if a["terminal"] == "WITNESS" else "#8e44ad"
                                                                  if a["clean"] else COLORS["unresolved"] for a in att])
    ax.axhline(250, color=COLORS["invalid"], ls="--", lw=0.8)
    ax.set(ylabel="CPU-s per seed attempt", xlabel="seed attempt (execution order)", yscale="symlog")
    ax.set_title("blue witness, purple clean terminal, orange not clean; dashed 250 CPU-s seed guard", fontsize=9, loc="left")
    f.tight_layout()
    f.savefig(out / "D_compute.png", dpi=80)
    plt.close(f)
