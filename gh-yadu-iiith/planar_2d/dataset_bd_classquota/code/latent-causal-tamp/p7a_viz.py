"""P7a diagnostic GIFs (DIAGNOSTIC RENDERS - no robot execution): scene edits A / B / A+B with teacher outcomes and
predicted root-level marginals (P6 B vs ABC7 arms), loss / gradient / clipping behaviour over training, and the
B / C coupling-mode comparison over recorded checkpoint snapshots. Guarded (700 CPU-s, 1,400 s wall)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.patches import Circle, FancyArrowPatch

import abc7_models as a7
import abc7_train as t7
import config as cf
import geometry as ge
import p6_scenes as p6s
import p6_viz as pv
import p7_audit as pa
import teacher
from scene import scene_from_dict
from viz import COLORS, plt

ROOTS = ["p6_2026102301", "p6_2026102303"]
KINDS = ["z0", "zS[A]", "zS[B]", "zS[A+B]"]


def _chosen(res, arm):
    c = res["chosen"][arm.split("-")[0]]
    return f"{arm}#s0|clip{c['clip']}|pool{c['lam']['pool']}"


def _load(out_a, res, arm):
    tag = _chosen(res, arm)
    m = a7.build(arm)
    m.load_state_dict(torch.load(Path(out_a) / f"ckpt_{tag.replace('|', '_')}.pt", weights_only=True))
    return m.eval(), tag


def scene_edits(dst, cfg, robot, plan, roots, out_a, res):
    preds = {}
    for arm in ("A", "B", "C", "B-no-coupling"):
        m, _ = _load(out_a, res, arm)
        with torch.no_grad():
            preds[arm] = {r["key"]: {k: float(v) for k, v in t7.scene_probs(m, r, t7.forward(m, r)).items()} for r in roots}
    p6b = pa.load("B#s0")
    with torch.no_grad():                                # P6 B: zS marginals of the EMPTY repair's joint (P6 convention)
        for r in roots:
            c = next(c for c in r["cands"] if not c["seq"])
            o = p6b.joint(r["R"], c["seq"], c["slots"], r["edits"])
            preds.setdefault("P6 B", {})[r["key"]] = {n: float(o["marg"][c["slots"].index(n)]) for n in KINDS}
    frames = []
    for r in roots:
        pr = plan["roots"][r["key"]]
        s = scene_from_dict(pr["scene"])
        ys = {"z0": next(c["y"] for c in r["cands"] if not c["seq"])} | r["scene_y"]
        for kind in KINDS:
            ed = {} if kind == "z0" else next(q["edit"] for q in pr["queries"]["scene_queries"] if f"zS[{q['kind']}]" == kind)
            f, ax = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1, 1.1]})
            se = p6s.moved(s, ed) if ed else s
            pv._panel(ax[0], cfg, robot, se, pr["goal_cell"], f"{r['key']} [{pr['family']}]  {kind}: edit {ed or 'none'}")
            for oid, cell in ed.items():
                a, b = s.get(oid).xy_m, s.lattice.center(cell)
                ax[0].add_patch(Circle(a, 0.024, fc="none", ec=COLORS["hypothetical"], ls=":", lw=1.2))
                ax[0].add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, color=COLORS["hypothetical"]))
            pv._tag(ax[0], "DIAGNOSTIC RENDER - HYPOTHETICAL SCENE EDIT - NOT ROBOT RELOCATION (robot fixed at q0)", "#34495e", (0.01, 0.94), 6)
            arms = list(preds)
            ax[1].bar(range(len(arms)), [preds[a][r["key"]][kind] for a in arms], color=["#999" if a == "P6 B" else COLORS["hypothetical"] for a in arms])
            ax[1].axhline(ys[kind], color=COLORS["valid"] if ys[kind] else COLORS["invalid"], lw=2, label=f"teacher outcome = {ys[kind]}")
            ax[1].set_xticks(range(len(arms)), arms)
            ax[1].set(ylim=(0, 1.05), title=f"predicted root-level P({kind} = 1); teacher pattern (z0,A,B,A+B) = "
                      f"{tuple(ys[k] for k in KINDS)}")
            ax[1].title.set_fontsize(8)
            kap = {a: preds[a][r["key"]]["zS[A+B]"] - preds[a][r["key"]]["zS[A]"] - preds[a][r["key"]]["zS[B]"] + preds[a][r["key"]]["z0"] for a in arms}
            ax[1].text(0.0, -0.16, "kappa_hat: " + "  ".join(f"{a} {v:+.2f}" for a, v in kap.items()), transform=ax[1].transAxes, fontsize=7)
            ax[1].legend(fontsize=7)
            frames += [pv._rgb(f)] * pv.STATIC_FRAMES
    pv.gif(dst / "scene_edits_A_B_AB.gif", frames, "P7a: fixed-robot edits none / A / B / A+B, teacher outcome vs predicted root-level marginals (P6 B vs ABC7)")


def loss_gradients(dst, res):
    frames = []
    for arm in ("A", "B", "C"):
        run = res["runs"][_chosen(res, arm)]
        h = [x for x in run["history"] if "total" in x]
        sn = [x for x in run["snapshots"] if "term_grads" in x]
        for cut in list(range(5, len(h), max(5, len(h) // 12))) + [len(h)]:
            f, ax = plt.subplots(1, 3, figsize=(15, 4.6))
            hh = h[:cut]
            for k in t7.TERMS:
                v = [x.get(k) for x in hh]
                if any(x is not None for x in v):
                    ax[0].plot([x["epoch"] for x in hh], v, label=k)
            ax[0].set(title=f"{arm} ({run['config']}): per-term loss (unweighted, root-normalized)", xlabel="epoch")
            ax[0].legend(fontsize=6)
            ax[1].plot([x["epoch"] for x in hh], [x["grad_pre"] for x in hh], color="#333", label="global grad norm (pre-clip)")
            ax[1].axhline(run["config"]["clip"], color=COLORS["invalid"], ls="--", label=f"clip {run['config']['clip']}")
            ax[1].set(yscale="log", title=f"clipped fraction so far {np.mean([x['clipped'] for x in hh]):.2f}", xlabel="epoch")
            ax[1].legend(fontsize=7)
            ss = [x for x in sn if x["epoch"] <= hh[-1]["epoch"]]
            for k in t7.TERMS:
                v = [x["term_grads"].get(k) for x in ss]
                if any(x is not None for x in v):
                    ax[2].plot([x["epoch"] for x in ss], v, "o-", ms=3, label=k)
            ax[2].set(yscale="log", title="unclipped per-term gradient norm (recorded snapshots)", xlabel="epoch")
            ax[2].legend(fontsize=6)
            frames.append(pv._rgb(f))
    pv.gif(dst / "P7a_loss_gradients.gif", frames, "P7a: per-term losses, pre-clip global gradient norm with clip level, per-term unclipped gradient norms (A, B, C)")


def coupling(dst, res, root):
    arms = [a for a in ("B", "B-quadratic", "B-no-coupling", "C", "C-no-coupling") if any(t.startswith(a + "#s0") for t in res["runs"])]
    snaps = {}
    for a in arms:
        tag = next(t for t in res["runs"] if t.startswith(a + "#s0") and res["runs"][t].get("snapshots"))
        snaps[a] = res["runs"][tag]["snapshots"]
    n = max(len(v) for v in snaps.values())
    frames = []
    for i in range(n):
        f, ax = plt.subplots(1, len(arms), figsize=(3.4 * len(arms), 4.4), squeeze=False)
        for a_, arm in zip(ax[0], arms):
            sn = snaps[arm][min(i, len(snaps[arm]) - 1)]
            p = [sn["scene"].get(k, np.nan) for k in KINDS]
            obs = sn["kappa"]["scene_obs"] if sn["kappa"] else {}
            a_.bar(range(4), p, color=COLORS["hypothetical"])
            a_.scatter(range(4), [obs.get(k, np.nan) for k in KINDS], marker="_", s=200, color="black", zorder=5)
            a_.set_xticks(range(4), ["z0", "A", "B", "A+B"])
            kh = sn["kappa"]["pred"] if sn["kappa"] else float("nan")
            a_.set(ylim=(0, 1.05), title=f"{arm} ep {sn['epoch']}\nkappa_hat {kh:+.2f} (teacher {sn['kappa']['obs'] if sn['kappa'] else '-'})")
            a_.title.set_fontsize(8)
        f.suptitle(f"{root}: root-level scene marginals per coupling mode (bars) vs teacher (black) - recorded checkpoint snapshots", fontsize=9)
        frames.append(pv._rgb(f))
    pv.gif(dst / "B_C_coupling_ablation.gif", frames, f"P7a: scene marginals and kappa_hat over training for full / quadratic / no-coupling B and C on {root}")


def run(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", "import sys, p7a_viz; p7a_viz.worker(*sys.argv[1:])", str(out), args.train],
                      700.0, 1400.0, out / "viz_heartbeat.txt", 20.0, 0.5, Path("."))
    cf.write_json(out / "viz_guard.json", {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")})
    print(json.dumps({k: g[k] for k in ("stop", "cpu_s", "wall_s")}))
    return 0 if g["stop"] == "COMPLETED" else 1


def worker(out, train) -> None:
    torch.set_num_threads(1)
    out = Path(out)
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    res = json.loads(Path(train, "results.json").read_text())
    plan = json.loads(Path("runs/p6/collect/plan.json").read_text())
    roots = [r for r in t7.prepare(torch.load("runs/p6/train/dataset.pt", weights_only=False)) if r["key"] in ROOTS]
    pv.INDEX.clear()
    scene_edits(out, cfg, robot, plan, roots, train, res)
    loss_gradients(out, res)
    coupling(out, res, "p6_2026102303")
    cf.write_json(out / "index.json", {"files": pv.INDEX})
