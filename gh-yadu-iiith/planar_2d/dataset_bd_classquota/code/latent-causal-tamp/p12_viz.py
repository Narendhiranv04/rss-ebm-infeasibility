"""P12 figures (recorded data only): holdout kappa panel (hypothetical edits labelled; certificates with kind), paired factor-ablation
plot, separate-protocol execution plot, gentle vs severe appearance panel, D blocker panel, recorded executed-success GIF."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import config as cf
import effects as ef
import geometry as ge
import p6_viz as pv
import p12_closure as pc12
from scene import scene_from_dict
from viz import COLORS, plt

DST = Path("visualizations/p12")


def _rd(p):
    return json.loads(Path(p).read_text())


def kappa_panel(cfg, robot, P, rec, ft):
    ks = sorted(ft["per_root_kappa"], key=lambda k: (-ft["per_root_kappa"][k], k))[:2] + \
        [k for k, v in sorted(ft["per_root_kappa"].items()) if v == 0][:1] + [k for k, v in sorted(ft["per_root_kappa"].items()) if v == -1][:1]
    f, ax = plt.subplots(len(ks), 4, figsize=(20, 5 * len(ks)), squeeze=False)
    for i, k in enumerate(ks):
        r = P[k]
        s, t = scene_from_dict(r["scene"]), {it["token"]: it for it in r["items"]}
        for j, tok in enumerate(("", "edit:A", "edit:B", "edit:A+B")):
            it, a = t[tok], ax[i, j]
            x = rec[it["query"]]
            se = ef.pose_substitution(robot, cfg, s, {"target_id": "t0"}, it["edit"])["scene"] if it.get("edit") else s
            pv._panel(a, cfg, robot, se, r["goal_cell"], f"{k[-3:]} {r['family'][:1]} kappa={ft['per_root_kappa'][k]}: {tok or 'original'}")
            kind = (x.get("static_certificate") or {}).get("kind")
            lab = {"VERIFIED_FEASIBLE": "VERIFIED (recorded witness)", "CERTIFIED_INFEASIBLE": f"CERTIFIED ({kind or 'teacher certificate'})"}.get(x["teacher_status"], x["teacher_status"])
            pv._tag(a, lab, pv.COL.get(x["teacher_status"], "#999"), size=6)
            if tok:
                pv._tag(a, "HYPOTHETICAL FIXED-q0 SCENE EDIT - NOT A ROBOT RELOCATION", "#34495e", loc=(0.02, 0.93), size=6)
    pv.png(f, DST / "P12_kappa_ground_truth.png", "P12 holdout: proof-grounded four-tuples (original, A, B, A+B) for kappa = +1, 0, -1 roots")


def factor_plot(ft):
    rows = [("F3 (primary)", ft["primary"]["F3"]), ("F5s (primary)", ft["primary"]["F5s"])] + [(k, v) for k, v in ft["secondary"].items()]
    f, ax = plt.subplots(figsize=(9, 5.5))
    for i, (n, v) in enumerate(rows):
        lo, hi = v["ci95"]
        ax.errorbar(v["mean_delta"], i, xerr=[[v["mean_delta"] - lo], [hi - v["mean_delta"]]], fmt="o", capsize=3, color="#2a6fdb" if "primary" in n else "#777")
        ax.text(hi + 0.01, i, f"p={v['p_raw']:.3g}" + (f" Holm={v['p_holm']:.3g}" if "p_holm" in v else ""), va="center", fontsize=7)
    ax.axvline(0, color="#bbb")
    ax.set(yticks=range(len(rows)), yticklabels=[n for n, _ in rows], xlabel="mean paired delta |kappa_hat_abl - kappa| - |kappa_hat_B - kappa| (>0: full B better)",
           title=f"P12 holdout ({ft['cohort']['proof_grounded_four_tuples']} proof-grounded roots): factor ablations vs full keyed B")
    pv.png(f, DST / "P12_factor_ablation.png", "P12 prospective F3 / F5 primaries and secondary ablations (root bootstrap CI)")


def exec_plot(ex):
    arms = ["K-B", "K-C", "A", "SF", "geometry", "G+M"]
    f, ax = plt.subplots(figsize=(9, 4.5))
    v = [ex["success_all_covered"][a] for a in arms]
    ax.bar(range(6), v, color=["#2a6fdb", "#e07b00", "#2e9e5b", "#8e5bd0", "#7f7f7f", "#c0392b"], hatch=None)
    ax.patches[-1].set_hatch("//")
    for i, x in enumerate(v):
        ax.text(i, x + 0.01, f"{x:.3f}", ha="center", fontsize=8)
    ax.set(xticks=range(6), xticklabels=[a + ("\n(planner-assisted)" if a == "G+M" else "") for a in arms], ylim=(0, 1),
           ylabel="executed success (seed mean per root)", title=f"P12 holdout strict TOP1 ({ex['coverage']['roots_fully_executed']} fully executed roots of 80)")
    pv.png(f, DST / "P12_executed_TOP1.png", "P12 executed TOP1 per arm; G+M hatched (separate protocol)")


def photo_plot(ph):
    arms = ["K-B", "K-C", "A", "SF", "B-V0"]
    f, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for j, (m, sm) in enumerate((("dp", "mean_abs_dp"), ("top1", "top1_changed"))):
        for i, v in enumerate(("G1_brightness_contrast", "G2_whitebalance_desat", "G3_mild_combined")):
            ax[j].bar([x + i * 0.2 for x in range(5)], [ph[f"{a}|{v}|{m}"]["mean"] for a in arms], 0.18, label=v)
        ax[j].bar([x + 0.6 for x in range(5)], [ph["_severe_P11_H9"].get(a, {}).get(sm, 0) for a in arms], 0.18, color="#c0392b", label="severe remap (P11)")
        ax[j].set(xticks=[x + 0.3 for x in range(5)], xticklabels=arms, title=["mean |delta p| per root", "TOP1 changed (fraction of roots)"][j])
    ax[0].legend(fontsize=7)
    pv.png(f, DST / "P12_appearance.png", "P12 appearance: realistic photometric (masks fixed) vs severe remap; POST-HOC on 100 P10 roots")


def d_plot(dh, dp):
    f, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for j, (r, name) in enumerate(((dp, "P10b blind (infeasible-only)"), (dh, "P12 holdout (infeasible-only)"))):
        st = [s for s in ("proof_grounded", "operational", "infeasible_all") if s in r["blockers_infeasible_only"]]
        for i, a in enumerate(("D", "D-pairless", "geometry_heuristic")):
            ax[j].bar([x + i * 0.27 for x in range(len(st))], [r["blockers_infeasible_only"][s][a]["f1"] for s in st], 0.25, label=a)
        ax[j].set(xticks=[x + 0.27 for x in range(len(st))], xticklabels=[f"{s}\nN={r['blockers_infeasible_only'][s]['D']['roots']}" for s in st], ylim=(0, 1), title=name)
    ax[0].legend(fontsize=8)
    pv.png(f, DST / "P12_D_blockers.png", "P12 D blocker F1 on infeasible originals: P10b vs new holdout, by provenance")


def exec_gif(cfg, robot, P):
    import g4d_run as gr
    man = _rd(pc12.OUT / "p12_5" / "EXEC_JOBS.json")
    for f in sorted((pc12.OUT / "p12_5" / "episodes").glob("*/ledger.json")):
        led = _rd(f)["episodes"]
        for j in man["jobs"]:
            e = led.get(f"{j['arm']}__{j['root']}")
            d = f.parent / f"{j['arm']}__{j['root']}"
            if e and e.get("outcome") == "SUCCESS" and j["choice"] != "GM" and len(j["ranked"][0]) >= 2 and (d / "relocation.npz").exists():
                s, seq, frames = scene_from_dict(P[j["root"]]["scene"]), [tuple(a) for a in j["ranked"][0]], []
                for qq, g, pos, mv in pv.trace_frames(robot, s, seq, gr._arrays(d / "relocation.npz"), gr._arrays(d / "target_witness.npz"), n_move=10, n_target=12):
                    fg, a = plt.subplots(figsize=(5, 5))
                    pv._panel(a, cfg, robot, s, P[j["root"]]["goal_cell"], f"{j['root'][-3:]} {P[j['root']]['family']}: TOP1 '{j['choice'].split(':', 2)[2]}'\nRECORDED episode",
                              q=qq, gap=g, pos=pos, held=mv)
                    pv._tag(a, "TOP1_COMMIT SUCCESS - recorded trajectory", COLORS["valid"])
                    frames.append(pv._rgb(fg))
                pv.gif(DST / "P12_executed_success.gif", frames, "P12 holdout: one recorded strict-TOP1 success (multi-move)")
                return


def run() -> None:
    DST.mkdir(parents=True, exist_ok=True)
    cfg = cf.load_config("config.toml")
    robot, P = ge.make_robot(cfg), pc12.plans()
    rec = {}
    for b in sorted(pc12.HOLD.glob("batch*/plan.json")):
        rec |= pc12.sealed_records(b.parent.name)
    ft = _rd(pc12.OUT / "p12_4" / "FACTOR_TESTS.json")
    kappa_panel(cfg, robot, P, rec, ft)
    factor_plot(ft)
    exec_plot(_rd(pc12.OUT / "p12_5" / "EXEC_RESULTS.json"))
    photo_plot(_rd(pc12.OUT / "p12_6" / "PHOTOMETRIC.json"))
    d_plot(_rd(pc12.OUT / "p12_6" / "D_SEMANTICS_holdout.json"), _rd(pc12.OUT / "p12_6" / "D_SEMANTICS_p10b.json"))
    exec_gif(cfg, robot, P)
    cf.write_json(DST / "INDEX.json", pv.INDEX)
