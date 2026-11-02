"""G4A figures A-H and repair GIFs (G4A instruction §10), from recorded states only.

Algorithmic candidates are drawn before (and separately from) the teacher statuses.
Status cells always carry text as well as colour. Missing diagnostic entries stay
visible as NOT_RUN. Fault-injection panels are labelled as such.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
from matplotlib.patches import Circle
from PIL import Image

import assignment_figures as af
import case_figures as cfig
import data
import repair_figures as rf
import skill as sk
import viz
from scene import place_scene, scene_from_dict
from viz import COLORS, plt

FILL = {data.POSITIVE: COLORS["valid"], data.NEGATIVE: COLORS["invalid"], data.UNKNOWN: COLORS["unresolved"],
        sk.VERIFIED: COLORS["valid"], sk.INFEASIBLE: COLORS["invalid"], sk.UNRESOLVED: COLORS["unresolved"],
        "NOT_RUN": "#bdbdbd", data.CONTRADICTION: "#000000"}
SHORT = {data.POSITIVE: "POS", data.NEGATIVE: "NEG", data.UNKNOWN: "UNK", data.CONTRADICTION: "CONTRA"}


def _cell(ax, x, y, status, text, w=1.0, h=1.0):
    ax.add_patch(plt.Rectangle((x, y), w, h, fc=FILL.get(status, "white"), ec="white", lw=1.5))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=7,
            color="white" if status in FILL and status != "NOT_RUN" else COLORS["text"])


def certificate_figure(robot, cfg, suite):
    f = plt.figure(figsize=(19, 0.32 * len(suite) + 2.5))
    gs = f.add_gridspec(2, 2, width_ratios=(2.3, 1))
    rows = [{"case": c["case"], "kind": c["kind"], "expected": "accept" if c["expected_accept"] else "reject",
             "checker": "ACCEPTED" if c["accepted"] else "REJECTED", "status": sk.VERIFIED if c["as_expected"] else sk.INFEASIBLE,
             "first reason": (c["reasons"][0] if c["reasons"] else c["note"])[:78]} for c in suite]
    for r in rows:
        r["as expected"] = "yes" if r["status"] == sk.VERIFIED else "NO"
    tab_ax = f.add_subplot(gs[:, 0])
    af._table(tab_ax, rows, ["case", "kind", "expected", "checker", "status", "first reason"],
              "Independent certificate checker: green = outcome as expected (sound proof accepted / broken premise rejected)")
    tab = tab_ax.tables[0]
    for i, r in enumerate(rows, start=1):                       # the status column states the comparison in words
        ok = r["status"] == sk.VERIFIED
        tab[i, 4].get_text().set_text("as expected" if ok else "UNEXPECTED")
        tab[i, 4].set_facecolor(COLORS["valid"] if ok else COLORS["invalid"])
        tab[i, 4].get_text().set_color("white")
    s2 = place_scene(cfg, "c_r2", "initial", [18], [3, 13, 50, 45])
    ax = f.add_subplot(gs[0, 1])
    viz.scene_panel(ax, s2, cfg, "goal-overlap premise (R2): goal disc + tolerance vs obstacle")
    g, tol = s2.lattice.center(45), cfg["gripper2d"]["release_xy_tolerance_m"]
    for rad, ls in ((0.024, "--"), (0.024 + tol, ":")):
        ax.add_patch(Circle(g, rad, fc="none", ec=COLORS["target_edge"], ls=ls, lw=1.2, zorder=9))
    rf._ids(ax, s2)
    ax.set(xlim=(g[0] - 0.12, g[0] + 0.12), ylim=(g[1] - 0.12, g[1] + 0.12))
    for txt in ax.texts:
        txt.set_clip_on(True)
    s4 = place_scene(cfg, "c_r4", "initial", [20], [28, 27, 31, 58, 63])
    ax = f.add_subplot(gs[1, 1])
    viz.scene_panel(ax, s4, cfg, "R4 scope: o00 on goal 28, pick blocked in S0 by o01/t0 (48/48 definite)")
    viz.draw_robot(ax, robot, s4.robot.q_rad, s4.robot.gripper_gap_m, alpha=0.8)
    rf._ids(ax, s4)
    rf._view(ax)
    rf._box(ax, ["LB 2 within: declared 48-grasp family,", "move-once, no pushing, target not a repair object"])
    f.tight_layout()
    return f


def candidates_figure(robot, cfg, plan, rows):
    qs = [next(q for q in plan["queries"] if q["stratum"] == st) for st in plan["strata"]]
    f, axes = plt.subplots(2, len(qs), figsize=(5.4 * len(qs), 12), gridspec_kw={"height_ratios": (1.15, 1)}, squeeze=False)
    cmap = plt.get_cmap("tab20")
    for k, q in enumerate(qs):
        sc, ax = scene_from_dict(q["scene"]), axes[0, k]
        viz.scene_panel(ax, sc, cfg, f"{q['query_id']} [{q['stratum']}]\nALGORITHMIC CANDIDATES (before any evaluation)")
        af._zoom(ax, sc)
        rf._ids(ax, sc)
        cfig._goal(ax, {"query": {"learner": {"action": q["action"]}}})
        for i, c in enumerate(q["audit_pool"][1:], start=1):
            for o, d in c["cells"].items():
                a, b = sc.get(o).xy_m, sc.lattice.center(d)
                rf._arrow(ax, a, b, cmap(i % 20), ls="--", lw=1.0)
                ax.text(*b, f"#{i}", fontsize=6, color=cmap(i % 20), weight="bold", zorder=11)
        ax2 = axes[1, k]
        ax2.axis("off")
        mine = [r for r in rows if r["query_id"] == q["query_id"]]
        ax2.set_title("OFFLINE TEACHER STATUS (first pass | enriched), after evaluation", fontsize=8, loc="left")
        for j, r in enumerate(mine):
            y = 1 - (j + 1) / 13
            ax2.text(0.0, y + 0.03, f"#{j} {r['label'][:30]}", fontsize=7, transform=ax2.transAxes, va="center")
            for x, t in ((0.72, "first_pass"), (0.86, "diagnostic_enriched")):
                ax2.add_patch(plt.Rectangle((x, y), 0.13, 0.06, transform=ax2.transAxes, fc=FILL[r[t]], ec="white"))
                ax2.text(x + 0.065, y + 0.03, SHORT[r[t]], transform=ax2.transAxes, fontsize=7, ha="center",
                         va="center", color="white")
    f.tight_layout()
    return f


def coverage_figure(cov, rows, plan):
    f, axes = plt.subplots(1, 3, figsize=(20, 6.2))
    strata = plan["strata"]
    for ax, tag in zip(axes[:2], ("first_pass", "diagnostic_enriched")):
        bottom = np.zeros(len(strata))
        for lab in (data.POSITIVE, data.NEGATIVE, data.UNKNOWN):
            vals = np.array([sum(r[tag] == lab and r["stratum"] == st for r in rows) for st in strata])
            ax.bar(range(len(strata)), vals, bottom=bottom, color=FILL[lab], edgecolor="white", width=0.7, label=lab)
            for i, (v, b) in enumerate(zip(vals, bottom)):
                if v:
                    ax.text(i, b + v / 2, str(v), ha="center", va="center", fontsize=8, color="white")
            bottom += vals
        ax.set_xticks(range(len(strata)), [s.replace("_", "\n") for s in strata], fontsize=7)
        ax.set_ylabel("audited assignments")
        u = cov[tag]["unknown_fraction_static_valid"]
        ax.set_title(f"{tag.replace('_', ' ')} labels by stratum; unknown fraction (static-valid) = {u:.2f}", fontsize=9, loc="left")
        ax.legend(fontsize=7)
    ax = axes[2]
    T, fp = plan["targets"], cov["first_pass"]
    items = [("zero-move roots", fp["zero_move_roots"], T["zero_move_roots"]),
             ("nonempty-witness roots", fp["nonempty_witness_roots"], T["nonempty_witness_roots"]),
             ("alternative roots", fp["alternative_roots"], T["alternative_roots"]),
             ("goal-negative roots", cov["goal_negative_roots"], T["goal_negative_roots"]),
             ("access-negative roots", cov["access_negative_roots"], T["access_negative_roots"])]
    for i, (name, got, need) in enumerate(items):
        ok = got >= need
        ax.barh(i, got, color=COLORS["valid"] if ok else COLORS["invalid"], height=0.6)
        ax.plot([need, need], [i - 0.4, i + 0.4], color=COLORS["text"], lw=2)
        ax.text(got + 0.2, i, f"{got} (target {need}) {'PASS' if ok else 'BLOCKED'}", va="center", fontsize=8)
    ax.set_yticks(range(len(items)), [x[0] for x in items], fontsize=8)
    ax.set_xlabel("distinct roots (first pass)")
    ax.set_title(f"targets vs first-pass coverage; negative sources {cov['negative_sources']}", fontsize=9, loc="left")
    f.tight_layout()
    return f


def seed_figure(seeds, plan):
    cases = plan["seed_panel"]
    f, ax = plt.subplots(figsize=(14, 0.5 * len(cases) + 1.6))
    got = {tuple(s["keys"]): s for s in seeds}
    n = len(cases[0]["keys"]) if cases else 5
    for i, c in enumerate(cases):
        s = got.get(tuple(c["keys"]))
        for j in range(n):
            if s:
                r = s["runs"][j]
                _cell(ax, j, -i, r["status"], f"{r['status'][:4]} {r['seconds']}s")
            else:
                _cell(ax, j, -i, "NOT_RUN", "NOT_RUN (no achieved state)")
        if s:
            h = s["home_required"]
            _cell(ax, n + 0.2, -i, h["status"], f"{h['status'][:4]} {h['seconds']}s", w=1.6)
        ax.text(-0.1, -i + 0.5, f"{c['query_id']} {c['cells'] or 'all-stay'}", ha="right", va="center", fontsize=7)
    ax.set(xlim=(-0.2, n + 2), ylim=(-len(cases) + 0.8, 1.4))
    ax.axis("off")
    for j in range(n):
        ax.text(j + 0.5, 1.15, f"seed {j}", ha="center", fontsize=8)
    ax.text(n + 1.0, 1.15, "HOME_REQUIRED\n(G3.5 scope)", ha="center", fontsize=8)
    ax.set_title("Target-planner seed panel: same bit-identical achieved state per row, same budgets; "
                 "VERI = complete witness, UNRE = unresolved within budget (finite-planner outcomes, not probabilities)",
                 fontsize=9, loc="left")
    f.tight_layout()
    return f


def order_figure(robot, cfg, orders, plan):
    rows = [(o, x) for o in orders for x in o["orders_found"]][:12]
    cols = 3
    f, axes = plt.subplots(max(1, (len(rows) + cols - 1) // cols), cols, figsize=(5.6 * cols, 5.4 * max(1, (len(rows) + 2) // 3)),
                           squeeze=False)
    byq = {q["query_id"]: q for q in plan["queries"]}
    for ax, (o, x) in zip(axes.flat, rows):
        sc = scene_from_dict(byq[o["query_id"]]["scene"])
        if x["relocation_status"] == "VERIFIED_FEASIBLE":
            sc = replace(sc, objects=tuple(replace(ob, xy_m=tuple(x["end_poses"][ob.id]), cell=None) for ob in sc.objects))
            viz.scene_panel(ax, sc, cfg, f"{o['query_id']} order {x['order']}\nachieved end state (robot at retreat end)")
            viz.draw_robot(ax, robot, x["end_q"], x["end_gap"], zorder=8)
            tg = " | ".join(f"seed{i}: {t['status'][:4]}" for i, t in enumerate(x["targets"]))
            viz.status_tag(ax, "VERIFIED_FEASIBLE", f"relocated; target {tg}")
        else:
            viz.scene_panel(ax, sc, cfg, f"{o['query_id']} order {x['order']}\nrelocation not completed (robot at S0)")
            viz.draw_robot(ax, robot, sc.robot.q_rad, sc.robot.gripper_gap_m, alpha=0.8, zorder=8)
            rf._box(ax, [x["reason"][:60]])
            viz.status_tag(ax, "UNRESOLVED", "order only (not an assignment negative)")
        rf._ids(ax, sc)
        rf._view(ax)
    for ax in list(axes.flat)[len(rows):]:
        ax.axis("off")
    f.tight_layout()
    return f


def abort_figure(cfg, faults):
    f, axes = plt.subplots(1, len(faults), figsize=(4.4 * len(faults), 5.6), squeeze=False)
    tol = cfg["evidence"]["reobserve_pose_tolerance_m"]
    for ax, fl in zip(axes[0], faults):
        c = fl["commanded_o01_xy"]
        ax.add_patch(Circle(c, 0.024, fc="none", ec=COLORS["muted"], ls="--", lw=1.4))
        ax.add_patch(Circle(c, tol, fc="none", ec=COLORS["hypothetical"], lw=1.0))
        if fl["observed_o01_xy"]:
            ax.add_patch(Circle(fl["observed_o01_xy"], 0.024, fc=COLORS["obstacle"], ec=COLORS["obstacle_edge"], alpha=0.6))
        ax.set(xlim=(c[0] - 0.04, c[0] + 0.04), ylim=(c[1] - 0.04, c[1] + 0.06), aspect="equal")
        ok = (fl["outcome"] == "ABORTED") != ("within" in fl["case"])
        lines = [f"outcome: {fl['outcome']}", f"relocations executed: {fl['relocations_executed']} of 2",
                 f"target calls: {fl['target_calls']}", f"proposals {fl['proposals']}, commits {fl['commits']}",
                 f"gap commanded {fl['commanded_gap']:.3f} / observed {fl['observed_gap']:.3f}"]
        lines += [f"abort: {r[:38]}" for r in fl["abort_reasons"][:2]]
        ax.text(0.02, 0.98, "\n".join(lines), transform=ax.transAxes, va="top", fontsize=7, family="monospace",
                bbox=dict(fc="white", ec=COLORS["muted"]))
        ax.set_title(f"FAULT-INJECTION TEST\n{fl['case']}\n(dashed: commanded o01; ring: {tol * 1000:.0f} mm tolerance)",
                     fontsize=8, loc="left")
        viz.status_tag(ax, "VERIFIED_FEASIBLE" if ok else "CERTIFIED_INFEASIBLE", "expected behaviour" if ok else "UNEXPECTED")
    f.tight_layout()
    return f


def completion_figure(col, seeds):
    evs = [o["target"] for r in col["main"].values() for a in r["audit"] for o in a["orders"] if o["target"]]
    evs += [r["primary"]["target"] for r in col["main"].values() if r["primary"]["target"]]
    keys = ("goal_achieved", "safe_retreat_status", "home_return_status", "complete_target_status")
    f, axes = plt.subplots(1, 2, figsize=(17, 5.5))
    ax = axes[0]
    for i, k in enumerate(keys):
        c = Counter(str(e.get(k)) for e in evs)
        left = 0
        for lab, n in sorted(c.items()):
            col_ = {"True": COLORS["valid"], "False": COLORS["invalid"]}.get(lab, FILL.get(lab, "#bdbdbd"))
            ax.barh(i, n, left=left, color=col_, edgecolor="white")
            ax.text(left + n / 2, i, f"{lab[:10]}\n{n}", ha="center", va="center", fontsize=7, color="white")
            left += n
    ax.set_yticks(range(len(keys)), keys, fontsize=8)
    ax.set_title(f"SAFE_RETREAT scope (addendum A1): {len(evs)} target evaluations (first pass + primary)", fontsize=9, loc="left")
    ax = axes[1]
    ax.axis("off")
    lines = ["case | SAFE_RETREAT seed0 (new scope) | home diag | HOME_REQUIRED (G3.5 scope)"]
    for s in seeds:
        r0, h = s["runs"][0], s["home_required"]
        lines.append(f"{s['query_id']} {str(s['cells'])[:22]:22s} | {r0['status'][:10]:10s} | {str(r0['home'])[:10]:10s} | {h['status'][:10]}")
    ax.text(0, 1, "\n".join(lines), va="top", fontsize=7.5, family="monospace", transform=ax.transAxes)
    ax.set_title("Old vs new end policy on bit-identical cloned states (separately identified evaluations)", fontsize=9, loc="left")
    f.tight_layout()
    return f


def g4a_figures(out, robot, cfg, plan, col, rows, cov, suite, faults) -> None:
    save = lambda f, name, dpi=80: (f.savefig(out / name, dpi=dpi), plt.close(f))  # noqa: E731
    save(certificate_figure(robot, cfg, suite), "A_certificate_audit.png")
    save(candidates_figure(robot, cfg, plan, rows), "B_automatic_candidates.png")
    save(coverage_figure(cov, rows, plan), "C_label_coverage.png")
    save(seed_figure(col["seed_panel"], plan), "D_seed_variance.png")
    if col["order_panel"]:
        save(order_figure(robot, cfg, col["order_panel"], plan), "E_order_end_states.png")
    save(abort_figure(cfg, faults), "F_abort_injection.png")
    save(completion_figure(col, col["seed_panel"]), "G_target_completion.png")
    res = {q: {"primary": r["primary"]} for q, r in col["main"].items()}
    save(cfig.call_trace(res), "H_primary_call_trace.png", 70)
    done = [q for q, r in sorted(col["main"].items()) if r["primary"]["outcome"] == "SUCCESS" and r["primary"]["commit"]["cells"]]
    for q in done[:2]:
        r = col["main"][q]
        cfig.r2_gif(robot, cfg, {"name": q, "primary": r["primary"], "query": r["query"], "scene": r["scene"]},
                    out / f"repair_{q}.gif")
    for q, r in col["main"].items():
        Image.fromarray(data.learner_tensors(r["query"], cfg["render"]["image_px"])["rgb"]).save(out / f"obs_{q}.png")
