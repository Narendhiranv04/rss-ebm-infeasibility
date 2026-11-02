"""G3 analytic figures and the G3 figure set: A0 assignment boards, restricted-domain
control with its all-supported counterpart, verification summary (Guide v1.6 G3).

Split from repair_figures.py (size): these panels show terminal assignments on the
support only; robot motion (relocation panels, repair GIFs) lives in repair_figures.py.
"""
from __future__ import annotations

import csv
import json
from multiprocessing import Pool
from pathlib import Path
import scipy
import config as cf
import geometry as ge
import interventions as iv

from collections import Counter

import numpy as np
from matplotlib.patches import Circle, Rectangle

import motion as mo
import repair_figures as rf
import verify as vf
import viz
from repair_figures import TINT, _arrow, _box, _gamma0_trace, _ids
from viz import COLORS, plt


def _tint(ax, scene, status):
    lat = scene.lattice
    for c, v in status.items():
        r, k = lat.rc(c)
        ax.add_patch(Rectangle((k * lat.pitch_m, r * lat.pitch_m), lat.pitch_m, lat.pitch_m, fc=TINT[v], ec="none", zorder=0.5))


def _zoom(ax, scene):
    """Support-only view with physical cell IDs (analytic boards: no robot motion shown)."""
    lat = scene.lattice
    ax.set(xlim=(-0.03, lat.cols * lat.pitch_m + 0.03), ylim=(-0.03, lat.rows * lat.pitch_m + 0.03))
    for c in scene.legal_cells:
        r, k = lat.rc(c)
        ax.text(k * lat.pitch_m + 0.003, (r + 1) * lat.pitch_m - 0.003, str(c), fontsize=5.5, color=COLORS["muted"],
                ha="left", va="top", zorder=1)


def matching_board(robot, cfg, items):
    """A0 per fixture: cell status for an obstacle-sized disc against the frozen
    action, direct blockers, the k-best certified assignments, the lower bound."""
    f, axes = plt.subplots(1, len(items), figsize=(6.0 * len(items), 8.0), squeeze=False)
    for ax, (name, out) in zip(axes[0], items):
        sc, t = out["scene"], out["terminal"]["ALL_SUPPORTED"]
        viz.scene_panel(ax, sc, cfg, f"{name}: A0 terminal assignment (ALL_SUPPORTED)")
        _tint(ax, sc, out["status"])
        _gamma0_trace(ax, robot, out["frozen"], sc)
        _ids(ax, sc)
        _zoom(ax, sc)
        for b in t["direct_blockers"]:
            ax.add_patch(Circle(sc.get(b).xy_m, 0.032, fc="none", ec=COLORS["invalid"], lw=2, zorder=9))
        for rank, a in reversed(list(enumerate(t["assignments"][:3]))):
            for oid, cell in a["cells"].items():
                if sc.get(oid).cell != cell:
                    _arrow(ax, sc.get(oid).xy_m, sc.lattice.center(cell), COLORS["hypothetical"],
                           ls="-" if rank == 0 else "--", lw=2.0 if rank == 0 else 1.0, alpha=1.0 if rank == 0 else 0.6)
                    ax.text(*sc.lattice.center(cell), f"#{rank}", fontsize=6, color=COLORS["hypothetical"],
                            ha="center", va="bottom", weight="bold", zorder=10)
        n = Counter(out["status"].values())
        _box(ax, [f"objects in matching: {len(t['objects'])} (all present obstacles)",
                  f"direct blockers D = {t['direct_blockers']}",
                  f"lower bound LB = {t['lower_bound']}  (= |D| here)",
                  f"k-best costs: {[a['cost'] for a in t['assignments']]}",
                  f"cells: VALID {n[mo.VALID]}  INVALID {n[mo.INVALID]}  UNRESOLVED {n[mo.UNRESOLVED]}",
                  "#0 solid = cheapest; #1, #2 dashed"], loc=(0.0, -0.09))
    f.suptitle("cell tint: green = an obstacle there is clear of gamma_0 and supported (VALID), red = it would "
           "block gamma_0 (INVALID), amber = UNRESOLVED (lower bound only). Orange line = target path of gamma_0.",
           fontsize=9, x=0.01, ha="left")
    f.tight_layout(rect=(0, 0.14, 1, 0.98))
    return f


def restricted_figure(robot, cfg, items):
    """Restricted domain vs all-supported counterpart: Hall certificate vs restoration witness."""
    f, axes = plt.subplots(len(items), 2, figsize=(12.5, 7.2 * len(items)), squeeze=False)
    for row, (name, label, out) in zip(axes, items):
        sc, rest = out["scene"], out["restoration"]
        for ax, policy in zip(row, ("ALL_SUPPORTED", "RESTRICTED")):
            t = out["terminal"][policy]
            viz.scene_panel(ax, sc, cfg, f"{name} ({label})\n{policy}")
            _tint(ax, sc, out["status"])
            _gamma0_trace(ax, robot, out["frozen"], sc)
            _ids(ax, sc)
            _zoom(ax, sc)
            for b in t["direct_blockers"]:
                ax.add_patch(Circle(sc.get(b).xy_m, 0.032, fc="none", ec=COLORS["invalid"], lw=2, zorder=9))
            if policy == "ALL_SUPPORTED":
                for oid, cell in rest["assignment"].items():
                    _arrow(ax, sc.get(oid).xy_m, sc.lattice.center(cell), COLORS["valid"])
                _box(ax, [f"LB = {t['lower_bound']} = |D| = {len(rest['direct_blockers'])}",
                          f"restoration witness (green): {rest['assignment']}",
                          f"terminal-valid: {rest['terminal_valid']}"], loc=(0.0, -0.09))
                viz.status_tag(ax, "VERIFIED_FEASIBLE", "terminal assignment exists (terminal only)")
            else:
                for c in sorted(set().union(*out["restricted_domain"].values())):
                    r, k = sc.lattice.rc(c)
                    ax.add_patch(Rectangle((k * sc.lattice.pitch_m, r * sc.lattice.pitch_m), sc.lattice.pitch_m,
                                           sc.lattice.pitch_m, fc="none", ec=COLORS["hypothetical"], hatch="///", lw=1.2, zorder=2))
                h = t["hall"]
                _box(ax, [f"blockers may use only the hatched cells {sorted(set().union(*out['restricted_domain'].values()))}",
                          f"Hall set U = {h['objects']}", f"N(U) = {h['neighbourhood_cells']}",
                          f"|N(U)| = {len(h['neighbourhood_cells'])} < |U| = {len(h['objects'])}: no assignment"], loc=(0.0, -0.09))
                viz.status_tag(ax, "CERTIFIED_INFEASIBLE", "terminal, under the restricted domain")
    f.tight_layout()
    return f


STATUS_FILL = {vf.EXECUTABLE: COLORS["valid"], vf.UNRESOLVED: COLORS["unresolved"],
               vf.TERMINAL_INFEASIBLE: COLORS["invalid"], "NOT_RUN": COLORS["muted"]}


def _table(ax, rows, head, title):
    ax.axis("off")
    tab = ax.table(cellText=[[str(r.get(h, "")) for h in head] for r in rows], colLabels=head, loc="upper left",
                   cellLoc="left")
    tab.auto_set_font_size(False)
    tab.set_fontsize(8.5)
    tab.scale(1, 1.5)
    for i, r in enumerate(rows, start=1):
        if r.get("status") in STATUS_FILL:
            tab[i, head.index("status")].set_facecolor(STATUS_FILL[r["status"]])
            tab[i, head.index("status")].get_text().set_color("white")
    tab.auto_set_column_width(list(range(len(head))))
    ax.set_title(title, fontsize=10, loc="left")


def summary_figure(robot, cfg, rows, witnesses):
    """Per-fixture bounds and outcomes (table) above keyframes of every executed
    witness: grasp configurations of each relocation (ghosts), the robot at
    gamma_0's start (solid), the unchanged gamma_0 path and its re-verified hash."""
    head = ["fixture", "domain", "|D|", "LB", "best verified", "minimality", "status", "stop", "plans", "attempts", "replay"]
    n = max(len(witnesses), 1)
    f = plt.figure(figsize=(max(19, 5.6 * n), 0.42 * len(rows) + 7.4))
    gs = f.add_gridspec(2, n, height_ratios=(0.42 * len(rows) + 1.0, 6.2))
    _table(f.add_subplot(gs[0, :]), rows, head, "G3 verification summary: LB = A0 terminal lower bound; best verified = "
           "moves of an executed, replayed witness; MINIMUM_CERTIFIED only when they meet")
    for k, (name, frozen, att) in enumerate(witnesses):
        ax = f.add_subplot(gs[1, k])
        frozen.verify()
        rf.final_panel(ax, robot, cfg, frozen, att, f"{name}: executed witness {' then '.join(f'{o}->{c}' for o, c in att.order)}"
                       f"\ngamma_0 {frozen.action_hash[:30]}... (re-verified, unchanged)")
        for res in att.skills:
            arr = res.traj.arrays()
            g = int(np.flatnonzero(arr["phase"] == "close_attach")[-1]) + 1
            viz.draw_robot(ax, robot, arr["q"][g], arr["gap"][g], alpha=0.3, zorder=7)
    f.tight_layout()
    return f


def attempt_rows(res: dict) -> list:
    """Every executed or rejected candidate: where it came from, terminal cost, real outcome."""
    rows = []
    for name in sorted(n for n, r in res.items() if r.get("search")):
        r = res[name]
        src = [(f"A0 rank {a.rank}", a) for a in r["search"]["attempts"]] + [(c["label"][:34], c["attempt"]) for c in r["compare"]]
        src += [(f"forced into {r['forced']['owner']}'s cell", a) for a in (r["forced"] or {"attempts": []})["attempts"]]
        rows += [{"fixture": name, "candidate": lab, "terminal cost": a.cost,
                  "order": " , ".join(f"{o}->{c}" for o, c in a.order), "status": a.status,
                  "first failure / result": a.reason[:70]} for lab, a in src]
    return rows


def attempts_figure(rows):
    head = ["fixture", "candidate", "terminal cost", "order", "status", "first failure / result"]
    f, ax = plt.subplots(figsize=(19, 0.36 * len(rows) + 1.2))
    _table(ax, rows, head, "Per-candidate terminal cost vs actual robot outcome (executed in the listed order; "
           "a rejected order is never an impossibility proof)")
    f.tight_layout()
    return f


def summary_rows(res: dict) -> list:
    rows = []
    for name in sorted(res):
        r = res[name]
        if "unavailable" in r:
            rows.append({"fixture": name, "status": "CONSTRUCTION_UNAVAILABLE", "stop": r["unavailable"][:60]})
            continue
        for policy, t in r["terminal"].items():
            row = {"fixture": name, "domain": policy, "|D|": len(t["direct_blockers"]), "LB": t["lower_bound"]}
            s = r["search"] if policy == "ALL_SUPPORTED" else None
            if t["hall"]:
                row.update(LB="none (Hall)", status=vf.TERMINAL_INFEASIBLE,
                           stop=f"|U|={len(t['hall']['objects'])} > |N(U)|={len(t['hall']['neighbourhood_cells'])}")
            elif s:
                n = Counter(a.status for a in s["attempts"])
                row.update({"best verified": s["upper_bound"], "minimality": s["minimality"], "status": s["status"],
                            "stop": s["stop_reason"], "plans": s["relocation_plans"],
                            "attempts": f"{n[vf.EXECUTABLE]} exec / {n[vf.UNRESOLVED]} rejected / {s['pruned_orders']} pruned",
                            "replay": r["replay"] and r["replay"]["replay_passed"]})
            else:
                row.update(status="NOT_RUN", stop="terminal-only control (no robot execution)")
            rows.append(row)
    return rows


def g3_figures(out, robot, cfg, res: dict) -> None:
    ok = {n: r for n, r in res.items() if "unavailable" not in r}
    save = lambda f, name, dpi=80: (f.savefig(out / name, dpi=dpi), plt.close(f))  # noqa: E731
    save(matching_board(robot, cfg, [(n, ok[n]) for n in ("G3_D1", "G3_D2", "G3_D3", "G3_T07_DENSE") if n in ok]),
         "A_matching_board.png", 80)
    for n, tag in (("G3_D1", "B_d1"), ("G3_D2", "C_d2"), ("G3_D3", "D_d3")):
        r = ok.get(n)
        if not r or not r["search"]:
            continue
        s0, w = vf.start_state(r["scene"], r["frozen"], n), r["search"]["witness"]
        same = [a for a in r["search"]["attempts"] if w and set(a.order) == set(w.order) and a is not w]
        rows = [(f"rejected order (A0 rank {a.rank})", a) for a in same] + ([(f"A0 witness (rank {w.rank})", w)] if w else [])
        rows += [(c["label"], c["attempt"]) for c in r["compare"]]
        save(rf.stepwise_figure(robot, cfg, r["frozen"], s0, rows, f"{n}: executed and rejected repair orders"),
             f"{tag}_stepwise.png")
        if r["forced"]:
            fo = r["forced"]
            save(rf.stepwise_figure(robot, cfg, r["frozen"], s0, [(f"forced {fo['blocker']} -> cell {fo['cell']} "
                                                              f"(held by {fo['owner']})", a) for a in fo["attempts"]],
                                 f"{n}: occupied preferred destination (cost {fo['cost']}; LB unchanged "
                                 f"{r['terminal']['ALL_SUPPORTED']['lower_bound']})"), f"{tag}_forced.png")
        if w:
            rf.repair_gif(robot, cfg, r["frozen"], s0, [(f"rejected order {a.order}", a) for a in same[:1]]
                       + [("A0 witness", w)], out / f"{tag}_repair.gif", n)
    r = ok.get("G3_T07_DENSE")
    if r and r["search"] and r["search"]["witness"]:
        s0, w = vf.start_state(r["scene"], r["frozen"], "G3_T07_DENSE"), r["search"]["witness"]
        save(rf.stepwise_figure(robot, cfg, r["frozen"], s0, [("A0 witness (dense, ALL_SUPPORTED)", w)],
                                f"G3_T07_DENSE: executed {len(w.order)}-relocation witness"), "E_t07_dense_witness.png")
        rf.repair_gif(robot, cfg, r["frozen"], s0, [("A0 witness", w)], out / "E_t07_dense_repair.gif", "G3_T07_DENSE")
    items = [(n, lab, ok[n]) for n, lab in (("G3_T07_PROXY", "DEVELOPMENT PROXY, 6 obstacles"),
                                            ("G3_T07_DENSE", "dense, 35 obstacles")) if n in ok]
    if items:
        save(restricted_figure(robot, cfg, items), "E_restricted_vs_all_supported.png")
    wit = [(n, ok[n]["frozen"], ok[n]["search"]["witness"]) for n in sorted(ok)
           if ok[n]["search"] and ok[n]["search"]["witness"]]
    save(summary_figure(robot, cfg, summary_rows(res), wit), "F_verification_summary.png")
    save(attempts_figure(attempt_rows(res)), "F_attempts.png")


# ------------------------------------------------------------------ G3 command (moved from run.py at G4A)
def run_repair(job):
    """Pool worker: A0 terminal problem, executable repair search and replay for one G3 fixture."""
    cfg, name, fx = job
    try:
        return name, vf.run_fixture(ge.make_robot(cfg), cfg, name, fx)
    except iv.FixtureUnavailable as e:
        return name, {"name": name, "unavailable": f"CONSTRUCTION_UNAVAILABLE: {e}"}


def cmd_solve(args) -> int:
    cfg = cf.load_config(args.config)
    fixtures = cf.validate_fixtures(json.loads(Path(args.fixtures).read_text())["fixtures"])
    g3 = {k: fx for k, fx in fixtures.items() if fx["stage"] == "G3"}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    robot, chash, rhash = ge.make_robot(cfg), cf.config_hash(cfg), cf.robot_profile_hash(cfg)
    with Pool(len(g3)) as pool:
        res = dict(pool.imap_unordered(run_repair, [(cfg, k, fx) for k, fx in g3.items()]))
    records, tasks, scenes, arrays = vf.archive(res, rhash)
    g3_figures(out, robot, cfg, res)
    rows = summary_rows(res)
    ok = all(r["replay"]["replay_passed"] for r in res.values() if r.get("replay"))
    cf.write_json(out / "config.json", {"config_hash": chash, "config": cfg})
    cf.write_json(out / "scenes.json", scenes)
    np.savez_compressed(out / "arrays.npz", **arrays)
    cf.write_json(out / "result.json", {
        "stage": "G3", "gate_checks_passed": ok, "fixtures": records, "tasks": tasks, "summary": rows,
        "config_hash": chash, "robot_profile_hash": rhash,
        "solver": {"matching": "scipy.optimize.linear_sum_assignment", "k_best": "Murty partitioning (solve.kbest)",
                   "scipy": scipy.__version__},
        "verification_scope": "KINEMATIC_3R_RELOCATIONS_AND_FIXED_GAMMA0_REPLAY (no dynamics, no simulation)",
        "simulation_replay_status": "NOT_RUN", "physical_execution_status": "NOT_RUN"})
    with open(out / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, ["fixture", "domain", "|D|", "LB", "best verified", "minimality", "status", "stop",
                                "plans", "attempts", "replay"])
        w.writeheader()
        w.writerows(rows)
    cf.write_manifest(out, {"planner_base": cfg["planner2d"]["seed"], "per_query": "crc32(episode key)"}, chash,
                    {"frozen_actions": {n: r["frozen"].action_hash for n, r in res.items() if "frozen" in r}})
    print(json.dumps({"out": str(out), "replay_passed": ok, "summary": rows}))
    return 0 if ok else 1
