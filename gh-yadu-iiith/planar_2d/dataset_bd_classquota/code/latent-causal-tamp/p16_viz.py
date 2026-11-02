"""P16 figures: scene-only family gallery (initial scene, pool, parking slots, lattice outcome), certified plan GIF (deterministic re-run,
plan must equal the recorded one; only replay-CLEAR arrays are drawn), B decision plots, D search comparison plots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DST = Path("visualizations/p16")
COL = {"LB": "#2a6fdb", "LB-F2": "#7fa7ec", "LC": "#8e5bd0", "LB+G": "#2e9e5b", "A": "#e07b00", "P13:K-B": "#52514e", "P13:A": "#c08a3e",
       "P13:SF": "#a7b4c2", "P13:K-C": "#b38bd6", "P13:B-F2": "#9fb7e8", "geometry": "#c0392b", "empty_always": "#7a7975", "random": "#d6d5d0",
       "Dp16": "#2a6fdb", "Dp16-norec": "#7fa7ec", "Dp": "#8e5bd0", "D": "#52514e", "geometry-norec": "#e8a090", "uniform": "#7a7975", "oracle": "#2e9e5b"}


def rd(p):
    return json.loads(Path(p).read_text())


def gallery(name="dev", out="shared/P16_dev_families.png") -> None:
    import p16_eval as pe
    import p6_viz as pv
    from scene import scene_from_dict
    from viz import plt
    import config as cf
    import geometry as ge
    plan, labs, tags = rd(pe.DATA / name / "plan.json"), pe.merged(name, "labels"), pe.merged(name, "tags")
    keys = sorted(labs)
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    n = len(keys)
    f, ax = plt.subplots(2, (n + 1) // 2, figsize=(4.2 * ((n + 1) // 2), 8.8), squeeze=False)
    for a, k in zip(ax.flat, keys):
        r, lab = plan["roots"][k], labs[k]
        s = scene_from_dict(r["scene"])
        t = tags.get(k, {})
        pv._panel(a, cfg, robot, s, r["goal_cell"], f"{k[-5:]} {r['family']}: y0={lab['original']['status'][:9]}")
        for o, d in lab["catalogue"]["parking"].items():
            x0, y0 = s.get(o).xy_m
            x1, y1 = s.lattice.center(d)
            a.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="->", color="#2a6fdb", lw=1.0, ls="--"))
        fixed = set(r.get("fixed", []))
        for o in s.objects:
            if o.cell in fixed:
                a.add_patch(plt.Circle(o.xy_m, o.radius_m * 1.25, fill=False, ec="#3b4a5a", lw=1.6, ls=":"))
        import p16_d as pdd
        d = pdd.lattice(lab, lab["catalogue"]["pool"])
        txt = f"pool {','.join(lab['catalogue']['pool'])}\nlattice: {d['type']}\nunion {d['mask']} nec {d['necessary']}"
        if t:
            txt += f"\nsearch: {'plan ' + str(t['moves']) + ' moves' if t.get('found') else 'no plan'}"
        a.text(0.02, 0.98, txt, transform=a.transAxes, va="top", fontsize=7, bbox=dict(fc="white", ec="#d6d5d0", alpha=0.9))
    for a in ax.flat[n:]:
        a.axis("off")
    (DST / out).parent.mkdir(parents=True, exist_ok=True)
    pv.png(f, DST / out, f"P16 {name} roots: scene-only pool (dashed = parking slot), dotted ring = fixed structure, lattice semantics, search result")


def plan_gif(name, key, out, n_max=4, per_obj=2, node_cap=500, guide="uniform", expect=None) -> dict:
    """Deterministic re-run of the uniform scene-only search on one root; GIF of the replay-CLEAR certified arrays + a static summary."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_eval as pe
    import p16_motion as pm
    import p6_viz as pv
    import p14_eval as p14
    from viz import COLORS, plt
    plan, idx = rd(pe.DATA / name / "plan.json"), pl.index_v2()[0]
    task = pc.task_from_root(plan["roots"][key], key)
    phys = pm.DetPhysics(task, idx)
    cells0 = {o.id: o.cell for o in phys.s0.objects}
    movers, _ = pc.propose_movers(task, cells0)
    import p16_d as pdd
    order = pe.uniform_order if guide == "uniform" else pdd.guide(guide, task, phys, movers).order
    r = pc.search(phys, order, movers, pc.slot_fn(phys, idx), cells0, phys.s0.robot.q_rad, n_max, per_obj, node_cap)
    if not r["found"]:
        return {"found": False}
    rep = pm.replay(phys, r["plan"])
    if not rep["ok"]:
        return {"found": True, "replay": rep}
    if expect is not None and rep["traj"] != expect:
        return {"found": True, "replay": rep, "matches_record": False}
    steps, recs = r["plan"]["steps"], r["plan"]["records"]
    rel, i = [], 0
    for st in steps:
        rel.append(p14._merge(recs[i], recs[i + 1]) if st["via"] == "lookup" else recs[i])
        i += 2 if st["via"] == "lookup" else 1
    frames = []
    for qq, gp, pos, mv in pv.trace_frames(phys.robot, phys.s0, [(s["object"], s["dest"]) for s in steps], rel, recs[-1:], n_move=10, n_target=12):
        fg, a = plt.subplots(figsize=(5, 5))
        pv._panel(a, phys.cfg, phys.robot, phys.s0, phys.goal_cell, f"{key[-5:]} {plan['roots'][key]['family']} [{guide}]: {len(steps)} moves, travel "
                  f"{r['plan']['travel']:.2f}\n{r['claim'][:44]}", q=qq, gap=gp, pos=pos, held=mv)
        pv._tag(a, f"replayed certified motion {rep['traj']}", COLORS["valid"])
        frames.append(pv._rgb(fg))
    (DST / out).parent.mkdir(parents=True, exist_ok=True)
    pv.gif(DST / out, frames, f"P16 {name} {key}: certified scene-only plan replay")
    return {"found": True, "replay": rep, "steps": [(s["object"], s["dest"], s["via"], s.get("transit")) for s in steps], "claim": r["claim"]}


def b_figures(rows, an, out="b/P16_B_test") -> None:
    """(1) decision breakdown per arm on decided feasible / infeasible originals; (2) ensemble P(empty) distributions per learned arm."""
    import p6_viz as pv
    from viz import plt
    arms = [a for a in an["per_arm"]]
    f, ax = plt.subplots(1, 3, figsize=(22, 5.6))
    x = np.arange(len(arms))
    P = an["per_arm"]
    nf, ni = max(1, an["feasible"]), max(1, an["infeasible"])
    parts = [("correct no-op", [P[a]["noop_correct"] / nf for a in arms], "#2e9e5b"),
             ("unnecessary relocation", [P[a]["unnecessary_relocation"] / nf for a in arms], "#e34948"),
             ("abstain", [P[a]["abstain_feasible"] / nf for a in arms], "#a7b4c2")]
    bot = np.zeros(len(arms))
    for lab, v, c in parts:
        ax[0].bar(x, v, 0.65, bottom=bot, color=c, label=lab, edgecolor="white")
        bot += np.array(v)
    ax[0].set_xticks(x, arms, rotation=40, ha="right")
    ax[0].set(ylabel="fraction of decided feasible originals", title=f"feasible originals (n = {an['feasible']})", ylim=(0, 1.05))
    ax[0].legend(fontsize=7, loc="lower right", framealpha=0.95)
    parts = [("repair verified", [P[a]["repair_success"] / ni for a in arms], "#2e9e5b"),
             ("attempt not verified", [(P[a]["attempted_infeasible"] - P[a]["repair_success"]) / ni for a in arms], "#eda100"),
             ("wrong no-op", [P[a]["wrong_noop"] / ni for a in arms], "#e34948"), ("abstain", [P[a]["abstain_infeasible"] / ni for a in arms], "#a7b4c2")]
    bot = np.zeros(len(arms))
    for lab, v, c in parts:
        ax[1].bar(x, v, 0.65, bottom=bot, color=c, label=lab, edgecolor="white")
        bot += np.array(v)
    ax[1].set_xticks(x, arms, rotation=40, ha="right")
    ax[1].set(ylabel="fraction of decided infeasible originals", title=f"infeasible originals (n = {an['infeasible']})", ylim=(0, 1.05))
    ax[1].legend(fontsize=7, loc="lower right", framealpha=0.95)
    learned = [a for a in arms if a in COL and a not in ("geometry", "empty_always", "random")]
    feas = [k for k, v in rows.items() if v["y0_status"] == "VERIFIED"]
    inf = [k for k, v in rows.items() if v["y0_status"] in ("CERTIFIED_INFEASIBLE", "STATIC_INVALID")]
    for j, a in enumerate(learned):
        pf = [1 / (1 + np.exp(-rows[k]["scores"][a][0][1])) for k in feas if a in rows[k]["scores"]]
        pi = [1 / (1 + np.exp(-rows[k]["scores"][a][0][1])) for k in inf if a in rows[k]["scores"]]
        ax[2].scatter(np.full(len(pf), j) - 0.12 + np.random.default_rng(j).uniform(-0.06, 0.06, len(pf)), pf, s=10, color="#2e9e5b", alpha=0.6)
        ax[2].scatter(np.full(len(pi), j) + 0.12 + np.random.default_rng(j + 9).uniform(-0.06, 0.06, len(pi)), pi, s=10, color="#e34948", alpha=0.6)
    ax[2].set_xticks(range(len(learned)), learned, rotation=40, ha="right")
    ax[2].set(ylabel="ensemble P(empty repair feasible)", title="no-op probability: green = feasible original, red = infeasible", ylim=(0, 1))
    (DST / out).parent.mkdir(parents=True, exist_ok=True)
    pv.png(f, DST / f"{out}.png", "P16-B test: decision breakdown and no-op probabilities (denominators = decided originals)")


def d_figures(rows, an, out="d/P16_D_test") -> None:
    """Anytime success vs search nodes, mean calls, virtual CPU (incl. D inference), co-solved travel vs geometry."""
    import p6_viz as pv
    from viz import plt
    g = list(an["per_guide"])
    f, ax = plt.subplots(1, 3, figsize=(20, 5.4))
    ts = [25, 50, 100, 200, 500]
    for k in g:
        ax[0].plot(ts, [an["per_guide"][k]["anytime_nodes"][str(t)] for t in ts], marker="o", ms=4, lw=2 if k in ("Dp16", "geometry") else 1.2,
                   color=COL.get(k, "#555"), label=f"{k} ({an['per_guide'][k]['success']}/{an['common_roots']})")
    ax[0].set(xscale="log", xlabel="search nodes (deterministic work)", ylabel="roots solved (replay CLEAR)", title="anytime success vs nodes")
    ax[0].legend(fontsize=7, loc="lower right", framealpha=0.95)
    x = np.arange(len(g))
    ax[1].bar(x, [an["per_guide"][k]["calls_mean"]["motion_checks"] for k in g], 0.6, color=[COL.get(k, "#555") for k in g])
    for i, k in enumerate(g):
        ax[1].text(i, an["per_guide"][k]["calls_mean"]["motion_checks"] + 2, f"{an['per_guide'][k]['cpu_virtual_mean']:.0f}s", ha="center", fontsize=7)
    ax[1].set_xticks(x, g, rotation=35, ha="right")
    ax[1].set(ylabel="mean motion checks per root (label: mean virtual CPU incl. D)", title="search effort")
    b = "geometry"
    for k in g:
        if k == b:
            continue
        both = [r for r in rows if rows[r]["guides"][k]["found"] and rows[r]["guides"][k]["replay"] and rows[r]["guides"][b]["found"] and rows[r]["guides"][b]["replay"]]
        ax[2].scatter([rows[r]["guides"][b]["travel"] for r in both], [rows[r]["guides"][k]["travel"] for r in both], s=18, color=COL.get(k, "#555"), label=k, alpha=0.7)
    lim = ax[2].get_xlim()[1]
    ax[2].plot([0, lim], [0, lim], color="#7a7975", ls="--", lw=1)
    ax[2].set(xlabel="geometry: plan travel", ylabel="guide: plan travel", title="co-solved roots")
    ax[2].legend(fontsize=7, framealpha=0.95)
    (DST / out).parent.mkdir(parents=True, exist_ok=True)
    pv.png(f, DST / f"{out}.png", "P16-D test: anytime success, effort and co-solved travel (identical action domain, budgets and certifier)")


def figures() -> None:
    import p16_eval as pe
    rows = pe.merged("test", "sealed_exec")
    b = {k: {"family": v["family"], **v["b"]} for k, v in rows.items() if "b" in v}
    d = {k: {"family": v["family"], "guides": v["d"]} for k, v in rows.items() if "d" in v}
    b_figures(b, rd(pe.OUT / "eval" / "B_TEST.json"))
    d_figures(d, rd(pe.OUT / "eval" / "D_TEST.json"))
