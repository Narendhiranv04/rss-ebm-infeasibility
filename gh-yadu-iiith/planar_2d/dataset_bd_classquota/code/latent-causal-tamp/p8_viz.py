"""P8 diagnostics (PNG / GIF) from saved records only; anything without data is listed NOT_AVAILABLE, never synthesized.

  factor_graph_modes.png          B-full / B-structured / B-none factor graphs (a DIAGRAM of the code, not a learned graph)
  coupling_conditionals.gif       unconditional vs ORACLE-REVEALED conditional completions on real decided roots
  kappa_signs.gif                 teacher (z0, zA, zB, zAB) tuples with provenance vs every arm's predicted marginals
  loss_gradient_balance.gif       per-term losses, pre-clip norms, clip level, updates and CPU per arm
  empty_repair_false_positive.gif roots where an arm chose the empty repair against a certified / operational negative
  sampler_mixing.gif              exact Boltzmann vs 4k / 16k chains (TV, top-5 discovery) and the deterministic fallback cost
  repair_cost_tradeoff.gif        two different-length VERIFIED repairs of one root from recorded teacher witnesses (J, displacement)
  ABC_S_GM_oneshot.gif            P8d episodes, all arms side by side per root, from each episode's own recorded arrays
  P8_root_decision_funnel.png, P8_paired_success.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle

import config as cf
import g4d_run as gr
import geometry as ge
import p6_viz as pv
import teacher
from scene import scene_from_dict
from viz import COLORS, plt

ARMS = ["A", "B-full", "B-structured", "B-none", "SF", "C-full"]
SCENE = ["z0", "zS[A]", "zS[B]", "zS[A+B]"]
NA = []


def _rd(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def factor_graph(dst):
    f, axs = plt.subplots(1, 3, figsize=(16, 5.6))
    pos = {"z0": (0.1, 0.8), "zS[A]": (0.1, 0.55), "zS[B]": (0.1, 0.3), "zS[A+B]": (0.1, 0.05), "zR": (0.9, 0.8), "zO[0]": (0.9, 0.5), "zO[1]": (0.9, 0.2)}
    for ax, mode in zip(axs, ("B-full", "B-structured", "B-none")):
        ax.set(xlim=(-0.15, 1.15), ylim=(-0.2, 1.05), title=mode)
        ax.axis("off")
        if mode != "B-none":
            for s in ("zS[A]", "zS[B]", "zS[A+B]"):
                ax.plot(*zip(pos["z0"], pos[s]), color="#2a78d6", lw=2.2)
            for o in ("zO[0]", "zO[1]"):
                ax.plot(*zip(pos["zR"], pos[o]), color="#16a085", lw=2.2)
        if mode == "B-full":
            for a, b in [("zS[A]", "zS[B]"), ("zS[A]", "zS[A+B]"), ("zS[B]", "zS[A+B]"), ("zS[A+B]", "zR"), ("z0", "zR"), ("zS[A]", "zO[0]"),
                         ("zO[0]", "zO[1]")]:
                ax.plot(*zip(pos[a], pos[b]), color="#c0392b", lw=1.0, ls="--")
            ax.text(0.5, 0.97, "phi_HO(r, z): higher-order residual on all z", ha="center", fontsize=8, color="#c0392b")
        for n, (x, y) in pos.items():
            ax.add_patch(Circle((x, y), 0.06, fc="#f4f4f4", ec="#333", zorder=5))
            ax.text(x, y, n, ha="center", va="center", fontsize=7, zorder=6)
        ax.text(0.5, -0.12, "travel J(r): separate readout, NOT in E(r, z)", ha="center", fontsize=7,
                bbox=dict(boxstyle="round", fc="#eee", ec="#888"))
        ax.text(0.5, 0.88, "P_S(z_s | c0) (left)  x  P_Q(z_q | z_s, r) (right)", ha="center", fontsize=7)
    f.text(0.01, 0.01, "blue: phi_scene z0-zS pairs; green: phi_omit zR-zO pairs; red dashed: generic coupling (illustrative subset) - code diagram, "
           "not a learned graph", fontsize=8)
    f.savefig(dst / "factor_graph_modes.png", dpi=80)
    plt.close(f)
    pv._register(dst / "factor_graph_modes.png", "B-full / B-structured / B-none factor graphs of ABC7 (diagram of the code)")


def _seed0(res, arm):
    return next((v for v in res["runs"].values() if v.get("status") == "RUN" and v["arm"] == arm and v["regime"] == "FULL_DECIDED"
                 and v["seed"] == 0 and v["config"] == res["chosen"].get(arm, res["chosen"]["B-full"])), None)


def conditionals(dst, res):
    runs = {a: _seed0(res, a) for a in ("B-full", "B-structured", "B-none", "SF")}
    runs = {a: r for a, r in runs.items() if r}
    roots = sorted({k for r in runs.values() for k, rows in r["cond_rows"]["validation"].items() if any(x["task"] in "ab" for x in rows)})
    frames = []
    for k in roots:
        f, axs = plt.subplots(1, len(runs), figsize=(4.2 * len(runs), 4.6), squeeze=False)
        for ax, (a, r) in zip(axs[0], runs.items()):
            e = next(x for x in r["evals"]["validation"] if x["root"] == k)
            rows = r["cond_rows"]["validation"][k]
            hid = ["zS[B]", "zS[A+B]"]
            un = [e["kappa"]["scene_pred"][h] for h in hid] if e["kappa"] else [np.nan, np.nan]
            co = [next((x["p"] for x in rows if x["task"] == "a" and x["hidden"] == h), np.nan) for h in hid]
            ys = [e["kappa"]["scene_obs"][h] for h in hid] if e["kappa"] else [np.nan, np.nan]
            x = np.arange(2)
            ax.bar(x - 0.25, un, 0.25, color="#999", label="unconditional P")
            ax.bar(x, co, 0.25, color="#8e44ad", label="P | revealed z0, zS[A]")
            ax.bar(x + 0.25, ys, 0.25, color="#333", label="teacher")
            ax.set_xticks(x, hid)
            ax.set(ylim=(0, 1.05), title=f"{a}")
            ax.legend(fontsize=6)
        f.suptitle(f"{k}: ORACLE-REVEALED DIAGNOSTIC, NOT AVAILABLE TO ROBOT (offline conditional completion, task a)", fontsize=9, color="#c0392b")
        frames += [pv._rgb(f)] * pv.STATIC_FRAMES
    if frames:
        pv.gif(dst / "coupling_conditionals.gif", frames, "Unconditional vs oracle-revealed conditional predictions (B-full / B-structured / B-none / SF)")
    else:
        NA.append("coupling_conditionals.gif")


def kappa_signs(dst, res, records):
    runs = {a: _seed0(res, a) for a in ARMS}
    runs = {a: r for a, r in runs.items() if r}
    frames = []
    for split in ("validation", "legacy_p7_test"):
        ref = next(iter(runs.values()))
        for e0 in ref["evals"][split]:
            if not e0["kappa"]:
                continue
            k = e0["root"]
            f, ax = plt.subplots(figsize=(11, 4.6))
            x = np.arange(4)
            w = 0.8 / (len(runs) + 1)
            ax.bar(x, [e0["kappa"]["scene_obs"][n] for n in SCENE], w, color="#333", label=f"teacher (kappa {e0['kappa']['obs']:+d})")
            for j, (a, r) in enumerate(runs.items(), 1):
                e = next(z for z in r["evals"][split] if z["root"] == k)
                ax.bar(x + j * w, [e["kappa"]["scene_pred"][n] for n in SCENE], w, label=f"{a} kappa_hat {e['kappa']['pred']:+.2f}")
            prov = [records.get(q, {}).get("label_source", "?") for q in (f"seq:{k}:empty",)]
            ax.set_xticks(x + 0.4, SCENE)
            ax.set(ylim=(0, 1.05), title=f"{k} ({split}): fixed-robot edit outcomes; empty-repair label source {prov[0]}")
            ax.legend(fontsize=6, ncol=2)
            frames += [pv._rgb(f)] * pv.STATIC_FRAMES
    if frames:
        pv.gif(dst / "kappa_signs.gif", frames, "Teacher kappa tuples vs predicted root-level marginals of every arm (seed 0, FULL_DECIDED)")
    else:
        NA.append("kappa_signs.gif")


def loss_balance(dst, res):
    frames = []
    for a in ARMS:
        r = _seed0(res, a)
        if not r:
            continue
        h = [x for x in r["history"] if "total" in x]
        for cut in list(range(20, len(h), 30)) + [len(h)]:
            f, ax = plt.subplots(1, 2, figsize=(12, 4.4))
            for t in ("static", "task", "scene", "omit", "effect", "cost", "pool"):
                v = [x.get(t) for x in h[:cut]]
                if any(y is not None for y in v):
                    ax[0].plot(range(len(v)), v, label=t)
            ax[0].set(title=f"{a}: per-term losses (update {cut}/{r['updates']}, {r['cpu_s']} CPU-s total)", xlabel="update")
            ax[0].legend(fontsize=6)
            ax[1].plot([x["grad_pre"] for x in h[:cut]], color="#333")
            ax[1].axhline(r["config"]["clip"], color=COLORS["invalid"], ls="--", label=f"clip {r['config']['clip']}")
            ax[1].set(yscale="log", title=f"pre-clip global norm; clipped fraction {np.mean([x['clipped'] for x in h[:cut]]):.2f}")
            ax[1].legend(fontsize=7)
            frames.append(pv._rgb(f))
    if frames:
        pv.gif(dst / "loss_gradient_balance.gif", frames, "Per-term losses and pre-clip gradient norms for every arm (matched updates)")


def false_empty(dst, cfg, robot, plan, funnel_rows):
    rows = [x for x in funnel_rows if x.get("false_feasible_empty") in ("True", True)]
    frames = []
    for x in rows[:12]:
        r = plan["roots"].get(x["root"])
        if not r:
            continue
        s = scene_from_dict(r["scene"])
        f, ax = plt.subplots(figsize=(5.4, 5.4))
        pv._panel(ax, cfg, robot, s, r["goal_cell"], f"{x['arm']} on {x['root']} ({x['split']}, thr {x['threshold']})\nchose the EMPTY repair")
        pv._tag(ax, f"original task {x.get('empty_negative_source') or 'negative'} - NO MOTION DRAWN", COLORS["invalid"], size=6)
        frames += [pv._rgb(f)] * pv.STATIC_FRAMES
    if frames:
        pv.gif(dst / "empty_repair_false_positive.gif", frames, "Roots where an arm selected the empty repair against a decided negative original task")
    else:
        NA.append("empty_repair_false_positive.gif")


def mixing(dst, resb):
    frames = []
    for a, d in resb["arms"].items():
        for k, mx in (d.get("mixing") or {}).items():
            f, ax = plt.subplots(1, 2, figsize=(11, 4.2))
            ax[0].bar(["4k", "16k"], [mx["chain_4000"]["tv_mean"], mx["chain_16000"]["tv_mean"]], color=["#e67e22", "#2a78d6"])
            ax[0].axhline(0.10, color="#333", ls=":", label="retain-sampler threshold 0.10")
            ax[0].set(ylim=(0, 1), title=f"{a} on {k}: TV(chain, exact pi), {mx['states']} states; detailed balance "
                      f"{'PASS' if mx['detailed_balance_pass'] else 'FAIL'}")
            ax[0].title.set_fontsize(8)
            ax[0].legend(fontsize=7)
            ax[1].bar(["4k chain", "16k chain", "deterministic"], [mx["chain_4000"]["cpu_s_mean"], mx["chain_16000"]["cpu_s_mean"],
                                                                    mx["deterministic_fallback"]["cpu_s"]], color="#888")
            ax[1].set(title=f"CPU-s; top-5 found 4k {mx['chain_4000']['top5_found_mean']:.1f} / 16k {mx['chain_16000']['top5_found_mean']:.1f} / det 5; "
                      f"det model calls {mx['deterministic_fallback']['model_calls']}")
            ax[1].title.set_fontsize(8)
            frames += [pv._rgb(f)] * pv.STATIC_FRAMES
    if frames:
        pv.gif(dst / "sampler_mixing.gif", frames, "Exact Boltzmann vs 4k / 16k chains and the deterministic bounded-pool fallback (P8b)")
    else:
        NA.append("sampler_mixing.gif")


def repair_tradeoff(dst, cfg, robot, collect: Path):
    plan, rec = _rd(collect / "plan.json"), {**(_rd(collect / "decisions.json") or {})}
    if not plan:
        NA.append("repair_cost_tradeoff.gif")
        return
    frames = []
    for k, r in plan["roots"].items():
        ver = [x for x in rec.values() if x["query"]["lineage"] == k and x["teacher_status"] == "VERIFIED_FEASIBLE" and x["query"]["sequence"]
               and x["query"]["root"] == k]
        lens = sorted({len(x["query"]["sequence"]) for x in ver})
        if len(lens) < 2 or len(frames) > 400:
            continue
        s = scene_from_dict(r["scene"])
        for n in (lens[0], lens[-1]):
            x = sorted([v for v in ver if len(v["query"]["sequence"]) == n], key=lambda v: v["query"]["query"])[0]
            w = pv.witness(collect, x)
            seq = [tuple(a) for a in x["query"]["sequence"]]
            for q, g, pos, mv in pv.trace_frames(robot, s, seq, w[1], w[2], n_move=8, n_target=10):
                fg, ax = plt.subplots(figsize=(5.4, 5.4))
                pv._panel(ax, cfg, robot, s, r["goal_cell"], f"{k}: {n}-move VERIFIED repair\nJ_first {x.get('J_first') or float('nan'):.2f}",
                          q=q, gap=g, pos=pos, held=mv)
                pv._tag(ax, "VERIFIED - recorded teacher witness", COLORS["valid"], size=6)
                frames.append(pv._rgb(fg))
        if len(frames) > 300:
            break
    if frames:
        pv.gif(dst / "repair_cost_tradeoff.gif", frames, "Shortest vs longest VERIFIED repair of the same roots (recorded teacher witnesses, measured J)")
    else:
        NA.append("repair_cost_tradeoff.gif")


def oneshot(dst, cfg, robot, plan, os_dir: Path):
    led = _rd(os_dir / "ledger.json")
    if not led:
        NA.append("ABC_S_GM_oneshot.gif")
        return
    by = {}
    for key, e in led["episodes"].items():
        by.setdefault(e["root"], {})[e["arm"]] = (key, e)
    arms, frames = ["A", "B-full", "B-structured", "SF", "U", "G+M"], []
    for k, eps in sorted(by.items()):
        r = plan["roots"][k]
        s = scene_from_dict(r["scene"])
        tr = {}
        for a in arms:
            if a in eps:
                key, e = eps[a]
                rel, tw = gr._arrays(os_dir / key / "relocation.npz"), gr._arrays(os_dir / key / "target_witness.npz")
                seq = [tuple(x) for x in (e.get("committed_sequence") or [])]
                tr[a] = (e, pv.trace_frames(robot, s, seq, rel, tw, n_move=6, n_target=8) if rel or tw else None)
        n = max([len(t) for _, t in tr.values() if t] + [pv.STATIC_FRAMES])
        for i in range(n):
            f, axs = plt.subplots(2, 3, figsize=(13.5, 9))
            for ax, a in zip(axs.ravel(), arms):
                if a not in tr:
                    ax.axis("off")
                    continue
                e, t = tr[a]
                if t:
                    q, g, pos, mv = t[min(i, len(t) - 1)]
                    pv._panel(ax, cfg, robot, s, r["goal_cell"], f"{a}: {e['outcome']}", q=q, gap=g, pos=pos, held=mv)
                else:
                    pv._panel(ax, cfg, robot, s, r["goal_cell"], f"{a}: {e['outcome']}")
                pv._tag(ax, "NEW P8d ONE-SHOT (recorded)" if t else f"{e['outcome']} - NO TRACE", COLORS["valid"] if e["outcome"] == "SUCCESS"
                        else COLORS["invalid"] if t else "#888", size=6)
            f.suptitle(f"{k} ({r['intent']}, sealed test)", fontsize=10)
            frames.append(pv._rgb(f))
    pv.gif(dst / "ABC_S_GM_oneshot.gif", frames, "P8d: A | B-full | B-structured | SF | U | G+M one-shot episodes per root from their own recorded arrays")
    arms_ = [a for a in arms if any(a in v for v in by.values())]
    roots = sorted(by)
    m = np.array([[1.0 if by[k].get(a, (0, {}))[1].get("outcome") == "SUCCESS" else 0.0 for a in arms_] for k in roots])
    f, ax = plt.subplots(figsize=(8, 0.4 * len(roots) + 2))
    ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(arms_)), arms_)
    ax.set_yticks(range(len(roots)), [f"{k} ({plan['roots'][k]['intent']})" for k in roots], fontsize=7)
    ax.set_title("P8d paired success per root (green = GOAL_AND_SAFE_RETREAT)\ntotals " +
                 ", ".join(f"{a} {int(v)}" for a, v in zip(arms_, m.sum(0))), fontsize=8)
    f.tight_layout()
    f.savefig(dst / "P8_paired_success.png", dpi=80)
    plt.close(f)
    pv._register(dst / "P8_paired_success.png", "P8d paired success matrix (roots x arms)")


def funnel_png(dst, rows):
    arms = sorted({x["arm"] for x in rows})
    stages = ["verified_candidate_exists", "admitted", "success"]
    f, ax = plt.subplots(figsize=(10, 4.2))
    for j, a in enumerate(arms):
        rr = [x for x in rows if x["arm"] == a and x["split"] == "validation" and str(x.get("threshold")) in ("0.5", "None")]
        vals = [sum(1 for x in rr if str(x.get(s)) in ("True",) or (s == "admitted" and str(x.get(s, "0")) not in ("0", "", "None")))
                for s in stages]
        ax.plot(range(3), vals, "o-", label=f"{a} (n={len(rr)})")
    ax.set_xticks(range(3), ["root has a VERIFIED candidate", "arm admits >= 1 candidate", "chosen candidate VERIFIED"])
    ax.set(title="P8b offline decision funnel on P7 validation roots (threshold 0.5; baselines: U, G+M)", ylabel="roots")
    ax.legend(fontsize=7)
    f.savefig(dst / "P8_root_decision_funnel.png", dpi=80)
    plt.close(f)
    pv._register(dst / "P8_root_decision_funnel.png", "P8b offline root decision funnel per arm")


def run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", "import sys, p8_viz; p8_viz.worker(*sys.argv[1:])", str(out), args.stage],
                      float(args.cpu), 2 * float(args.cpu), out / "viz_heartbeat.txt", 20.0, 0.5, Path("."))
    cf.write_json(out / f"viz_guard_{args.stage}.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def worker(out, stage) -> None:
    import csv
    out = Path(out)
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    pv.INDEX.clear()
    res = _rd("runs/p8/a/results.json")
    p7plan = _rd("runs/p7/collection/plan.json")
    records = {**_rd("runs/p7/collection/decisions.json"), **_rd("runs/p7/collection/sealed_test.json")}
    if stage in ("a", "all"):
        factor_graph(out)
        conditionals(out, res)
        kappa_signs(out, res, records)
        loss_balance(out, res)
    if stage in ("b", "all") and _rd("runs/p8/b/results.json"):
        rows = list(csv.DictReader(open("runs/p8/b/decision_funnel.csv")))
        false_empty(out, cfg, robot, p7plan, rows)
        mixing(out, _rd("runs/p8/b/results.json"))
        funnel_png(out, rows)
    if stage in ("cd", "all"):
        repair_tradeoff(out, cfg, robot, Path("runs/p8/collection"))
        plan8 = _rd("runs/p8/collection/plan.json")
        if plan8:
            oneshot(out, cfg, robot, plan8, Path("runs/p8/execution"))
    idx = _rd(out / "index.json") or {"files": []}
    keep = {x["file"] for x in pv.INDEX}
    cf.write_json(out / "index.json", {"files": [x for x in idx["files"] if x["file"] not in keep] + pv.INDEX,
                                       "not_available": sorted(set((idx.get("not_available") or []) + NA))})
