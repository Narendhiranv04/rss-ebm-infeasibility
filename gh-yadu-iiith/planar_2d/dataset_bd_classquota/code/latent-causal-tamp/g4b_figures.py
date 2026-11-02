"""G4B stage runner, guards and figures. Writes config.json, result.json, arrays.npz,
overview.png, sampler_trace.png and manifest.json to runs/g4b_sampler_exact/.

Figures are labelled SYNTHETIC ENERGY / MATHEMATICAL TEST. The single scene panel is the
archived G4D R4 root drawn from saved records, marked CONTEXT ONLY - NO PLANNER/NEW
EVIDENCE. Histograms from seeded chains are supplementary; the proof is the exact matrix.
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

import config as cf
import g4b
import geometry as ge
import sampler as sp
import sequences as sq
import viz
from scene import scene_from_dict
from viz import COLORS, plt

GRAD_TOL = 1e-7          # central difference, eps = 1e-6: truncation O(eps^2) + roundoff ~1e-16/eps
WATCH = (("planner", "plan"), ("skill", "plan_skill"), ("evaluate", "evaluate_target"), ("verify", "relocate"),
         ("verify", "plan_fixed_order"), ("ordered", "teach_sequence"))


def _instrument() -> dict:
    """Wrap every loaded planner/target/teacher entry point with a call counter."""
    calls = {}
    for mod, fn in WATCH:
        m = sys.modules.get(mod)
        if m is None or not hasattr(m, fn):
            continue
        key, real = f"{mod}.{fn}", getattr(m, fn)
        calls[key] = 0

        def wrapped(*a, _k=key, _f=real, **kw):
            calls[_k] += 1
            return _f(*a, **kw)
        setattr(m, fn, wrapped)
    return calls


def guards() -> dict:
    """Mutation, invariance and bookkeeping checks recorded in result.json (also in tests)."""
    dom, E, d, cap, _ = g4b.space("A1")
    st = g4b.reference_states(d, cap)
    pi = g4b.reference_pi(st, E, g4b.T0)
    db = lambda P: float(np.abs(pi[:, None] * P - (pi[:, None] * P).T).max())  # noqa: E731
    out, real_h, real_a = {}, sp.log_hastings, sp.log_accept
    try:
        sp.log_hastings = lambda *a: real_h(*a) + (0.5 if a[3] in ("BIRTH", "DEATH") else 0.0)
        out["mutation_wrong_hastings_caught"] = db(sp.transition_matrix(dom, st, E, g4b.T0)) > 1e-3
        sp.log_hastings = real_h
        sp.log_accept = lambda e0, e1, T, lr: min(0.0, (e1 - e0) / T + lr)
        out["mutation_flipped_acceptance_caught"] = db(sp.transition_matrix(dom, st, E, g4b.T0)) > 1e-3
    finally:
        sp.log_hastings, sp.log_accept = real_h, real_a
    stale = lambda r: E(tuple(sorted(r)))  # noqa: E731  (order term not recomputed)
    out["mutation_missing_order_recompute_caught"] = db(sp.transition_matrix(dom, st, stale, g4b.T0)) > 1e-3
    base = sp.MarginalEnergy(g4b.joint_energy("B"), g4b.layout_fn, g4b.T0, {"v": 1})
    pb = g4b.reference_pi(st, base, g4b.T0)
    free = lambda r: sp.Layout(g4b.B_SLOTS, tuple(s != "zS2" for s in g4b.B_SLOTS), (0,) * 6)  # noqa: E731
    bad = sp.MarginalEnergy(g4b.joint_energy("B"), free, g4b.T0, {"v": 1})
    out["mutation_inactive_z_multiplicity_caught"] = float(np.abs(g4b.reference_pi(st, bad, g4b.T0) - pb).max()) > 1e-3
    pad = lambda r: g4b.layout_fn(r, ("zS1",), ("zS9", "pad"))  # noqa: E731
    out["inactive_slot_insertion_max_dp"] = float(np.abs(g4b.reference_pi(
        st, sp.MarginalEnergy(g4b.joint_energy("B"), pad, g4b.T0, {"v": 1}), g4b.T0) - pb).max())
    perm = lambda r: (lambda L: sp.Layout(L.slots[::-1], L.active[::-1], tuple(1 for _ in L.slots)))(g4b.layout_fn(r))  # noqa: E731
    out["inactive_metadata_permutation_max_dp"] = float(np.abs(g4b.reference_pi(
        st, sp.MarginalEnergy(g4b.joint_energy("B"), perm, g4b.T0, {"v": 1}), g4b.T0) - pb).max())
    cat2 = sp.MarginalEnergy(g4b.joint_energy("B"), lambda r: g4b.layout_fn(r, ("zS1", "zS2")), g4b.T0,
                             {"version": "toy-1", "scene_queries": ["zS1", "zS2"]})
    cat1 = sp.MarginalEnergy(g4b.joint_energy("B"), g4b.layout_fn, g4b.T0, {"version": "toy-1", "scene_queries": ["zS1"]})
    out["catalogue_change_recorded"] = cat1.catalogue_hash != cat2.catalogue_hash
    out["catalogue_change_max_dp"] = float(np.abs(g4b.reference_pi(st, cat2, g4b.T0) - g4b.reference_pi(st, cat1, g4b.T0)).max())
    try:
        sp.z_states(sp.Layout(tuple(f"q{i}" for i in range(9)), (True,) * 9, (0,) * 9))
        out["more_than_8_active_status"] = "ENUMERATED (wrong)"
    except sp.Unsupported:
        out["more_than_8_active_status"] = "NOT_RUN (Unsupported raised)"
    lam = 0.3
    C = lambda r: 0.1 * sum(abs(dd - 2) for _, dd in r)  # noqa: E731  toy known displacement (z-independent)
    outer = [base(r) + lam * C(r) for r in st]
    inner = sp.MarginalEnergy(lambda r, z: g4b.joint_energy("B")(r, z) + lam * C(r), g4b.layout_fn, g4b.T0, {"v": 1})
    out["cost_inside_vs_outside_max_diff"] = float(np.abs(np.array(outer) - np.array([inner(r) for r in st])).max())
    out["cost_added_twice_detected"] = float(np.abs(np.array([inner(r) + lam * C(r) for r in st]) - np.array(outer)).max()) > 1e-3
    try:
        sp.transition_row(dom, (), sp.MarginalEnergy(g4b.joint_energy("B"), g4b.layout_fn, 0.5, {}), g4b.T0)
        out["temperature_mismatch_refused"] = False
    except ValueError:
        out["temperature_mismatch_refused"] = True
    wrongT = sp.MarginalEnergy(g4b.joint_energy("B"), g4b.layout_fn, 0.5, {"v": 1})
    wt = np.array([np.exp(-wrongT(r) / g4b.T0) for r in st])
    out["inner_T_0.5_vs_gold_max_dp"] = float(np.abs(wt / wt.sum() - pb).max())
    ren = {"a": "z", "b": "y"}
    dom2 = sp.Domain.make({ren[o]: ds for o, ds in d.items()}, cap)
    E2 = lambda r: E(tuple((next(k for k, v in ren.items() if v == o), x) for o, x in r))  # noqa: E731
    st2 = [tuple((ren[o], x) for o, x in s) for s in st]
    out["relabel_matrix_max_diff"] = float(np.abs(sp.transition_matrix(dom2, st2, E2, g4b.T0) - sp.transition_matrix(
        dom, st, E, g4b.T0)).max())
    out["order_changes_energy"] = abs(E((("a", 1), ("b", 3))) - E((("b", 3), ("a", 1)))) > 1e-6
    out["order_swap_same_assignment"] = sq.assignment((("a", 1), ("b", 3))) == sq.assignment((("b", 3), ("a", 1)))
    s1, _ = sp.run_chain(dom, (), E, g4b.T0, 500, 1)
    s2, _ = sp.run_chain(dom, (), E, g4b.T0, 500, 1)
    s3, _ = sp.run_chain(dom, (), E, g4b.T0, 500, 2)
    out["seed_reproducible"], out["different_seed_differs"] = s1 == s2, s1 != s3
    return out


def _empirical(dom, energy, states, steps, seed):
    chain, _ = sp.run_chain(dom, (), energy, g4b.T0, steps, seed)
    idx = {s: i for i, s in enumerate(states)}
    h = np.bincount([idx[s] for s in chain], minlength=len(states)).astype(float)
    return h / h.sum()


def r4_context(ax, root: Path):
    """Archived G4D R4 root and its decided order contrast, from saved records only."""
    plan = json.loads((root / "runs/g4d_evidence/final_plan.json").read_text())
    ev = json.loads((root / "runs/g4d_evidence/unit_evidence.json").read_text())
    s = scene_from_dict(plan["roots"]["R4"]["scene"])
    cfg = cf.load_config(root / "config.toml")
    viz.scene_panel(ax, s, cfg, "G4D R4 root (archived) and its decided order contrast")
    viz.draw_robot(ax, ge.make_robot(cfg), s.robot.q_rad, s.robot.gripper_gap_m, alpha=0.8, zorder=6)
    ax.set(xlim=viz.LIMITS["grid"][0], ylim=viz.LIMITS["grid"][1])
    for o in s.objects:
        ax.text(*o.xy_m, o.id, fontsize=6, ha="center", va="center", zorder=10)
    g = s.lattice.center(28)
    ax.add_patch(Circle(g, 0.024, fc="none", ec=COLORS["target_edge"], ls="--", lw=1.4, zorder=9))
    for col, seq, dy in ((COLORS["valid"], (("o01", 18), ("o00", 29)), 0.012), (COLORS["invalid"], (("o00", 29), ("o01", 18)), -0.012)):
        for t, (o, c) in enumerate(seq):
            a, b = np.array(s.get(o).xy_m), np.array(s.lattice.center(c))
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=10, color=col, ls="--", lw=1.3, zorder=8))
            ax.text(*((a + b) / 2 + [0, dy]), str(t + 1), fontsize=7, color="white", ha="center", va="center", zorder=11,
                    bbox=dict(boxstyle="circle,pad=0.15", fc=col, ec="none"))
    lab = lambda u: ev[u]["label"]  # noqa: E731
    ax.text(0.01, 0.99, "\n".join([f"green  o01>18, o00>29: {lab('seq:R4:o01>18.o00>29')} (G4D witness)",
                                   f"red    o00>29, o01>18: {lab('seq:R4:o00>29.o01>18')} (ORDER_FIRST_ACTION)"]),
            transform=ax.transAxes, va="top", fontsize=7, family="monospace", bbox=dict(fc="white", ec=COLORS["muted"]))
    ax.text(0.01, 0.01, "CONTEXT ONLY - NO PLANNER/NEW EVIDENCE", transform=ax.transAxes, fontsize=8, color="white",
            weight="bold", bbox=dict(fc="#7a7975", ec="none", pad=3))


def _bars(ax, pi, emp, title, labels):
    x = np.arange(len(pi))
    ax.bar(x - 0.2, pi, 0.4, color=COLORS["hypothetical"], label="exact pi (reference enumeration)")
    ax.bar(x + 0.2, emp, 0.4, color="#9db7d6", label=f"empirical, seeded chain ({g4b.CHAIN['steps']} steps; supplementary)")
    ax.set_xticks(x, labels, rotation=90, fontsize=5)
    ax.set_title(title, fontsize=9, loc="left")
    ax.legend(fontsize=6)


def _lab(s):
    return ".".join(f"{o}{d}" for o, d in s) or "[]"


def run_stage(args, t0, c0, cfg_hash) -> int:
    calls = _instrument()
    out, root = Path(args.out), Path(".")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        raise SystemExit("result.json exists; refusing to overwrite a recorded G4B run")
    res, arrays, emp = {"spaces": {}, "bc": {}}, {}, {}
    for n in g4b.SPACES:
        dom, E, d, cap, nexp = g4b.space(n)
        st = g4b.reference_states(d, cap)
        c, pi, P = g4b.certify(dom, E, g4b.T0, st)
        c2, _, _ = g4b.certify(dom, E, g4b.T0, st, probs=(0.4, 0.3, 0.2, 0.1))
        res["spaces"][n] = {**c, "expected_states": nexp, "count_ok": len(st) == nexp, "unequal_kernel_probs": c2,
                            "pass": c["pass"] and c2["pass"] and len(st) == nexp}
        arrays.update({f"{n}__energy": np.array([E(s) for s in st]), f"{n}__pi": pi, f"{n}__P": P})
        if n in ("A1", "A2"):
            emp[n] = _empirical(dom, E, st, g4b.CHAIN["steps"], g4b.CHAIN["seed"])
            res["spaces"][n]["empirical_tv_supplementary"] = float(0.5 * np.abs(emp[n] - pi).sum())
    for k in "BC":
        c, pi, P, marg, st = g4b.certify_bc(k)
        c["gradient_max_error"] = g4b.gradient_check(k, (("a", 2), ("b", 3)))
        c["gradient_max_error_empty"] = g4b.gradient_check(k, ())
        c["pass"] = c["pass"] and max(c["gradient_max_error"], c["gradient_max_error_empty"]) <= GRAD_TOL
        emp[k] = _empirical(sp.Domain.make(g4b.SPACES["A1"][0], 2), marg, st, g4b.CHAIN["steps"], g4b.CHAIN["seed"] + 1)
        c["empirical_tv_supplementary"] = float(0.5 * np.abs(emp[k] - pi).sum())
        res["bc"][k] = c
        arrays.update({f"{k}__marginal_energy": np.array([marg(s) for s in st]), f"{k}__pi": pi, f"{k}__P": P,
                       f"{k}__p_zR1": np.array([sum(q for z, q in marg.conditional(s) if z.get("zR")) for s in st])})
    res["guards"] = g = guards()
    gpass = (all(g[k] for k in g if k.endswith(("_caught", "_recorded", "_refused", "_detected"))) and g["order_changes_energy"]
             and g["order_swap_same_assignment"] and g["seed_reproducible"] and g["different_seed_differs"]
             and max(g["inactive_slot_insertion_max_dp"], g["inactive_metadata_permutation_max_dp"],
                     g["cost_inside_vs_outside_max_diff"], g["relabel_matrix_max_diff"]) <= g4b.TOL
             and g["catalogue_change_max_dp"] > 1e-3 and g["inner_T_0.5_vs_gold_max_dp"] > 1e-3
             and g["more_than_8_active_status"].startswith("NOT_RUN"))
    res["guards_pass"] = bool(gpass)
    res["max_proof_error"] = float(max([v["max_error"] for v in res["spaces"].values()] + [v["unequal_kernel_probs"]["max_error"]
                                        for v in res["spaces"].values()] + [v["max_error"] for v in res["bc"].values()]))
    ok = all(v["pass"] for v in res["spaces"].values()) and all(v["pass"] for v in res["bc"].values()) and gpass
    res["planner_target_calls_by_g4b"] = calls
    res["decision"] = "PASS" if ok and not any(calls.values()) else "BLOCKED"
    dom1 = sp.Domain.make(g4b.SPACES["A2"][0], 2)
    E2 = g4b.space("A2")[1]
    _, recs = sp.run_chain(dom1, (), E2, g4b.T0, g4b.CHAIN["trace_steps"], g4b.CHAIN["seed"])
    f, ax = plt.subplots(2, 3, figsize=(21, 11))
    for a, k, title in ((ax[0, 0], "A1", "A1: N=2, K=[2,2], b=2 (13 states)"), (ax[0, 1], "A2", "A2: K=[1,2,3]+context, b=2 (29 states)"),
                        (ax[1, 0], "B", "B1 exact z-marginal pi(r) (JOINT toy energy)"), (ax[1, 1], "C", "C1 exact z-marginal pi(r)")):
        sts = g4b.reference_states(*g4b.SPACES["A1" if k in "BC" else k][:2])
        _bars(a, arrays[f"{k}__pi"], emp[k], f"SYNTHETIC ENERGY / MATHEMATICAL TEST - {title}", [_lab(s) for s in sts])
    a = ax[0, 2]
    stB = g4b.reference_states(*g4b.SPACES["A1"][:2])
    for k, col in (("B", COLORS["hypothetical"]), ("C", COLORS["target"])):
        a.plot(range(len(stB)), arrays[f"{k}__p_zR1"], "o-", color=col, label=f"{k}: P(zR=1 | r) (exact)")
    a.set_xticks(range(len(stB)), [_lab(s) for s in stB], rotation=90, fontsize=6)
    a.set_title("SYNTHETIC ENERGY / MATHEMATICAL TEST - conditional outcome P(zR=1|r);\nempty repair has no zR slot (0)",
                fontsize=9, loc="left")
    a.legend(fontsize=7)
    r4_context(ax[1, 2], root)
    f.suptitle(f"G4B {res['decision']}: exact transition-matrix proof, max error {res['max_proof_error']:.1e} "
               f"(tolerance {g4b.TOL:g}); histograms are supplementary", fontsize=11)
    f.tight_layout()
    f.savefig(out / "overview.png", dpi=80)
    plt.close(f)
    f, ax = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    col = {"DESTINATION": COLORS["hypothetical"], "SWAP": "#8e44ad", "BIRTH": COLORS["valid"], "DEATH": COLORS["invalid"]}
    ax[0].plot([r["E_old"] for r in recs], color=COLORS["muted"], lw=0.8, label="E(current state)")
    for i, r in enumerate(recs):
        k = r["kernel"].split("(")[0]
        if r["proposal"] is not None:
            ax[0].plot(i, r["E_new"], "o", ms=5, mfc=col[k] if r["accepted"] else "white", mec=col[k])
    for k, c in col.items():
        ax[0].plot([], [], "o", color=c, label=f"{k} proposal (filled accepted, hollow rejected)")
    ax[0].set(ylabel="synthetic energy")
    ax[0].legend(fontsize=7, ncol=3)
    sts = g4b.reference_states(*g4b.SPACES["A2"][:2])
    idx = {s: i for i, s in enumerate(sts)}
    ax[1].step(range(len(recs)), [idx[r["proposal"] if r["accepted"] else r["state"]] for r in recs],
               where="post", color=COLORS["text"], lw=0.8)
    ax[1].set(xlabel="update", ylabel="state index (A2 enumeration)")
    ax[0].set_title("SYNTHETIC ENERGY / MATHEMATICAL TEST - seeded A2 chain: true discrete proposals; unavailable moves are "
                    "self-transitions (no marker)", fontsize=10, loc="left")
    f.tight_layout()
    f.savefig(out / "sampler_trace.png", dpi=80)
    plt.close(f)
    np.savez_compressed(out / "arrays.npz", **arrays)
    src = {p: "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in
           ("sampler.py", "g4b.py", "g4b_figures.py", "sequences.py", "config.py", "tests/test_sampler.py")}
    cf.write_json(out / "config.json", {"stage_config": g4b.stage_config(), "stage_config_hash": cfg_hash,
                                        "deployment_config_hash": cf.config_hash(cf.load_config("config.toml")),
                                        "config_v2_hash": cf.stable_hash(cf.load_config_v2("config_v2.toml"))})
    res.update(stage="G4B", tolerance=g4b.TOL, gradient_tolerance=GRAD_TOL, sources=src, seeds=g4b.CHAIN,
               wall_s=round(time.perf_counter() - t0, 2), cpu_s=round(time.process_time() - c0, 2),
               evidence_scope="SYNTHETIC MATHEMATICAL FIXTURES ONLY; no learned model, no robot evidence")
    cf.write_json(out / "result.json", res)
    cf.write_manifest(out, {"chain_seed": g4b.CHAIN["seed"]}, cfg_hash, {"sources": src})
    print(json.dumps({"decision": res["decision"], "max_proof_error": res["max_proof_error"], "calls": calls,
                      "cpu_s": res["cpu_s"]}))
    return 0 if res["decision"] == "PASS" else 1
