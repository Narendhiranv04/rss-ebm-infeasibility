"""P7b GIFs from recorded data only (guarded 1,800 CPU-s / 3,600 s): per-root scene / catalogue / recorded witnesses /
order pairs / hypothetical edits / proposals (p6_viz.root_visuals), A | B | C synchronized comparisons on the same test
roots (p_R, per-edit marginals, kappa_hat vs teacher kappa, selection), training snapshots, Boltzmann samples on the
learned energy (exact vs chain), and the independent one-shot episodes animated from THEIR OWN recorded arrays."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

import abc7_models as a7
import abc7_train as t7
import config as cf
import g4b
import g4d_run as gr
import geometry as ge
import p6_viz as pv
import p7_eval as ev
import sampler as sp
import teacher
import zlayout_r4 as z4
from scene import scene_from_dict
from viz import COLORS, plt

KINDS = ["z0", "zS[A]", "zS[B]", "zS[A+B]"]


def compare(dst, cfg, robot, plan, res):
    ev_ = {a: {e["root"]: e for e in res["runs"][f"{a}#s0"]["eval"]["test"]} for a in "ABC" if f"{a}#s0" in res["runs"]}
    roots = [k for k in next(iter(ev_.values())) if not k.endswith("_ref")]
    frames = []
    for k in roots:
        f, ax = plt.subplots(2, 3, figsize=(15, 8.5))
        for j, a in enumerate(ev_):
            e, tau = ev_[a][k], res["runs"][f"{a}#s0"]["tau"]
            rows = sorted(e["cands"], key=lambda c: c["query"])
            p = [1 / (1 + np.exp(-c["u"] / tau)) for c in rows]
            ax[0, j].bar(range(len(rows)), p, color=[pv.COL[{1: "VERIFIED_FEASIBLE", 0: "CERTIFIED_INFEASIBLE"}.get(c["y"], "UNKNOWN")] for c in rows])
            ax[0, j].set_xticks(range(len(rows)), [c["query"].split(":", 2)[2][:18] for c in rows], rotation=80, fontsize=5)
            sel = ev.select(e, tau)
            ax[0, j].set(ylim=(0, 1.05), title=f"{a} on {k} ({plan['roots'][k]['family']}, test): calibrated p_R\nselected: "
                         f"{sel['query'].split(':', 2)[2] if sel else 'ABSTAIN'} (teacher y={sel and sel['y']})")
            ax[0, j].title.set_fontsize(7)
            ax[0, j].axhline(0.5, color="#888", ls=":", lw=0.7)
            kap = e["kappa"]
            if kap:
                x = np.arange(4)
                ax[1, j].bar(x - 0.2, [kap["scene_pred"][n] for n in KINDS], 0.4, color=COLORS["hypothetical"], label="predicted P(z=1)")
                ax[1, j].bar(x + 0.2, [kap["scene_obs"][n] for n in KINDS], 0.4, color="#555", label="teacher outcome")
                ax[1, j].set_xticks(x, ["z0", "A", "B", "A+B"])
                ax[1, j].set(ylim=(0, 1.05), title=f"kappa_hat {kap['pred']:+.2f}  vs teacher kappa {kap['obs']:+d}")
                ax[1, j].legend(fontsize=6)
            else:
                ax[1, j].axis("off")
        frames += [pv._rgb(f)] * pv.STATIC_FRAMES
    if frames:
        pv.gif(dst / "ABC_side_by_side.gif", frames, "A | B | C (seed 0) on the same sealed P7b test roots: calibrated p_R, selection, per-edit marginals, kappa_hat vs teacher")


def training(dst, res):
    for a in ("A", "B", "C"):
        run = res["runs"].get(f"{a}#s0", {})
        sn = run.get("snapshots") or []
        h = [x for x in run.get("history", []) if "total" in x]
        frames = []
        for s in sn:
            f, ax = plt.subplots(1, 2, figsize=(11, 4.4))
            hh = [x for x in h if x["epoch"] <= s["epoch"]]
            ax[0].plot([x["epoch"] for x in hh], [x["total"] for x in hh], color="#333", label="train total")
            vv = [x for x in hh if "val" in x]
            ax[0].plot([x["epoch"] for x in vv], [x["val"] for x in vv], "o-", ms=3, color=COLORS["invalid"], label="validation outcome NLL")
            ax[0].set(title=f"{a} seed 0, epoch {s['epoch']}", xlabel="epoch")
            ax[0].legend(fontsize=7)
            p = [s["scene"].get(n, np.nan) for n in KINDS]
            obs = (s["kappa"] or {}).get("scene_obs", {})
            ax[1].bar(range(4), p, color=COLORS["hypothetical"])
            ax[1].scatter(range(4), [obs.get(n, np.nan) for n in KINDS], marker="_", s=200, color="black", zorder=5)
            ax[1].set_xticks(range(4), ["z0", "A", "B", "A+B"])
            ax[1].set(ylim=(0, 1.05), title=f"validation root: scene marginals (bars) vs teacher (black); kappa_hat "
                      f"{(s['kappa'] or {}).get('pred', float('nan')):+.2f}")
            ax[1].title.set_fontsize(8)
            frames.append(pv._rgb(f))
        if frames:
            pv.gif(dst / f"{a}_training.gif", frames, f"{a} seed 0: loss curves and recorded validation snapshots over training")


def boltzmann(dst, res, ds_test, train_dir, root):
    """G4B kernels (cap 4) on each model's learned sampling energy for one test root: exact pi vs 4,000-step chain."""
    recs = {r["audit"]["root"]: r for r in ds_test["roots"]}
    r = next(x for x in t7.prepare({"roots": [recs[root]]}))
    ids = recs[root]["audit"]["ids"]
    cell = lambda xy: int(round((xy[1] - 0.04) / 0.08)) * 8 + int(round((xy[0] - 0.04) / 0.08))  # noqa: E731
    main = max(r["cands"], key=lambda c: len(c["seq"]))
    keep = {ids[i] for i, _ in main["seq"]}
    dests = {}
    for c in r["cands"]:
        for i, d in c["seq"]:
            if ids[i] in keep:
                dests.setdefault(ids[i], set()).add(cell(d))
    dests = {o: sorted(v) for o, v in dests.items()}
    cat = [{"kind": e["kind"], "edit": {ids[i]: cell(p) for i, p in e["poses"].items()}, "eligible": e["eligible"]} for e in recs[root]["edits"]]
    frames, out = [], {}
    for a in ("A", "B", "C"):
        m = a7.build(a)
        m.load_state_dict(torch.load(Path(train_dir) / f"ckpt_{a}#s0.pt", weights_only=True))
        m.eval()
        cache = {}

        @torch.no_grad()
        def energy(state, m=m, cache=cache):
            if state not in cache:
                seq = [(ids.index(o), ((d % 8 + 0.5) * 0.08, (d // 8 + 0.5) * 0.08)) for o, d in state]
                m._memo = {}
                o = m.seq(r["R"], seq) if a == "A" else m.joint(r["R"], seq, list(z4.build(r["key"], list(state), cat)["query"]), r["edits"])
                m._memo = None
                cache[state] = float(o["E"]) + ev._cd(r, seq) + float(o["J"])
            return cache[state]
        dom, states = sp.Domain.make(dests, 4), g4b.reference_states(dests, 4)
        cert, pi, _ = g4b.certify(dom, energy, 1.0, states)
        chain, _ = sp.run_chain(dom, (), energy, 1.0, 4000, 0)
        freq = {s: 0 for s in states}
        for s in chain:
            freq[s] += 1
        top = np.argsort(-np.array(pi))[:12]
        out[a] = {"states": len(states), "detailed_balance_pass": cert["pass"], "max_error": cert["max_error"],
                  "tv": 0.5 * sum(abs(freq[s] / len(chain) - p) for s, p in zip(states, pi))}
        f, ax = plt.subplots(figsize=(10, 4.2))
        ax.bar(np.arange(len(top)) - 0.2, [pi[i] for i in top], 0.4, color=COLORS["hypothetical"], label="exact Boltzmann P(r)")
        ax.bar(np.arange(len(top)) + 0.2, [freq[states[i]] / len(chain) for i in top], 0.4, color="#888", label="4,000-step chain frequency")
        ax.set_xticks(range(len(top)), ["/".join(f"{o}>{d}" for o, d in states[i]) or "empty" for i in top], rotation=60, fontsize=6)
        ax.set_title(f"{a} on {root}: Boltzmann samples over {len(states)} ordered repairs (cap 4); exact detailed balance "
                     f"{'PASS' if cert['pass'] else 'FAIL'}, TV {out[a]['tv']:.3f}", fontsize=8)
        ax.legend(fontsize=7)
        frames += [pv._rgb(f)] * pv.STATIC_FRAMES
    pv.gif(dst / "ABC_boltzmann_samples.gif", frames, f"A | B | C learned sampling energy on {root}: exact Boltzmann vs chain samples (G4B kernels)")
    return out


def oneshot(dst, cfg, robot, plan, os_dir):
    led = json.loads(Path(os_dir, "ledger.json").read_text())
    frames = []
    for key, e in sorted(led["episodes"].items()):
        r = plan["roots"][e["root"]]
        s = scene_from_dict(r["scene"])
        d = Path(os_dir) / key
        rel, tw = gr._arrays(d / "relocation.npz"), gr._arrays(d / "target_witness.npz")
        seq = [tuple(a) for a in (e.get("committed_sequence") or [])]
        steps = pv.trace_frames(robot, s, seq, rel, tw) if rel or tw else [(None, None, None, None)] * pv.STATIC_FRAMES
        for q, g, pos, mv in steps:
            f, ax = plt.subplots(figsize=(5.4, 5.4))
            pv._panel(ax, cfg, robot, s, r["goal_cell"], f"one-shot {e['arm']} on {e['root']} ({r['family']})\ncommitted "
                      f"{'.'.join(f'{o}>{c}' for o, c in seq) or '-'}", q=q, gap=g, pos=pos, held=mv)
            ok = e["outcome"] == "SUCCESS"
            pv._tag(ax, f"NEW ONE-SHOT EXECUTION (recorded) - {e['outcome']}" if rel or tw else f"{e['outcome']} - NO TRACE",
                    COLORS["valid"] if ok else COLORS["invalid"] if rel or tw else "#888", size=6)
            frames.append(pv._rgb(f))
    if frames:
        pv.gif(dst / "oneshot_episodes.gif", frames, "Independent model-chosen one-shot 2D episodes (A, B, C, uniform), recorded arrays of each episode only")


def run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", "import sys, p7b_viz; p7b_viz.worker(*sys.argv[1:])", str(out), args.collect, args.train, args.oneshot],
                      1800.0, 3600.0, out / "viz_heartbeat.txt", 20.0, 0.5, Path("."))
    cf.write_json(out / "viz_guard.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def worker(out, collect, train, oneshot_dir) -> None:
    torch.set_num_threads(1)
    out, collect = Path(out), Path(collect)
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    plan = json.loads((collect / "plan.json").read_text())
    records = json.loads((collect / "decisions.json").read_text()) | json.loads((collect / "sealed_test.json").read_text())
    pv.INDEX.clear()
    full = pv.trace_frames                                # coarser sampling of the SAME recorded arrays (48 roots in 1,800 CPU-s)
    pv.trace_frames = lambda robot, s, seq, rel, tw: full(robot, s, seq, rel, tw, n_move=8, n_target=10)
    for k, r in plan["roots"].items():
        if r["variant"] == "obstructed" and any(x["query"]["lineage"] == k and x["teacher_status"] != "NOT_RUN" for x in records.values()):
            dst = out / "roots" / f"{k}_{r['family']}_{r['split']}"
            dst.mkdir(parents=True, exist_ok=True)
            pv.root_visuals(dst, cfg, robot, plan, records, collect, k)
    res = json.loads(Path(train, "results.json").read_text())
    md = out / "models"
    md.mkdir(exist_ok=True)
    compare(md, cfg, robot, plan, res)
    training(md, res)
    ds_test = torch.load(Path(train) / "dataset_sealed.pt", weights_only=False)
    roots = [r["audit"]["root"] for r in ds_test["roots"] if r["audit"].get("family") == "ORD"] or [r["audit"]["root"] for r in ds_test["roots"]]
    samp = boltzmann(md, res, ds_test, train, next(x for x in roots if not x.endswith("_ref")))
    if Path(oneshot_dir, "ledger.json").exists():
        oneshot(md, cfg, robot, plan, oneshot_dir)
    cf.write_json(out / "index.json", {"files": pv.INDEX, "sampler": samp})
    pv.readme(out)
