"""P9-L3d: input-controlled visual ablation, optional keyed-vs-unique matched training, freeze manifest (saved data only).

Visual-zero control: the SAME vision=True B-full architecture, parameter count and seed initialization as the reference U-B-full
400-update runs; the precomputed DINOv2 object features are replaced by a fixed zero tensor for every root in training,
validation and evaluation (consistently, not only at test time). Same RootSampler batches, clip, schedule and selection.
Jobs run through p9_l3c.job (shared ledger runs/p9/l3c/guards); no teacher, IK or planner call.
"""
from __future__ import annotations

import collections
import contextlib
import math

import numpy as np
import torch

import config as cf
import joint_likelihood as jl
import p9_l3 as l3
import p9_l3c as l3c
import unique_factor as uf
import unique_factor_keyed as ufk

OUT = l3c.Path("runs/p9/l3d")
L3B = l3c.Path("runs/p9/l3/l3b")
REF = {s: f"U-B-full#s{s}#u400" for s in (0, 1, 2)}


def mask_visual(roots):
    for r in roots:
        r["R"].vis = torch.zeros_like(r["R"].vis)
    return roots


def data(visual=True):
    train, val, test = l3.data()
    return (train, val, test) if visual else tuple(mask_visual(x) for x in (train, val, test))


def plan_l3d() -> None:
    p = {"version": "P9_L3D_PLAN_V1", "l3c0_gate": l3._rd("runs/p9/l3c/L3C_0_GATE.json")["verdict"],
         "l3c1_gate": l3._rd("runs/p9/l3c/L3C_1_GATE.json")["verdict"],
         "visual_control": {"jobs": [{"name": f"V0-B-full#s{s}#u400", "kind": "viszero", "seed": s, "updates": 400} for s in (0, 1, 2)],
                            "reference": {s: {"run": REF[s], "checkpoint_sha256": l3._rd(L3B / f"runs/{REF[s]}.json")["checkpoint_sha256"]} for s in (0, 1, 2)},
                            "schedule": "400 updates (reference 400-update full-vision checkpoints verified for all 3 seeds)",
                            "matching": "torch.manual_seed(seed); abc8.build('B-full') (vision=True, identical parameters); RootSampler(train, 12, seed); "
                                        "AdamW 1e-3 / 1e-4; clip 10; best validation own root NLL every 20 updates"},
         "keyed_optional": {"jobs": [{"name": f"K-B-full#s{s}", "kind": "keyed", "seed": s, "updates": 200} for s in (0, 1, 2)],
                            "reference": {s: f"U-B-full#s{s}" for s in (0, 1, 2)},
                            "rule": "only if L3c-1 PASS and >= 1,200 CPU-s remain after the visual control and regression reserve (700 CPU-s)"},
         "decision_inputs": "validation only (12 roots): own exact root NLL, per-query NLL, calibrated repair / scene metrics, top-3 concentration, "
                            "leave-one-root-out; disclosed test descriptive only",
         "sources": {f: l3.sha(f) for f in ("unique_factor.py", "unique_factor_keyed.py", "joint_likelihood.py", "abc7_models.py", "p9_l3.py", "p9_l3c.py", "p9_l3d.py")},
         "caps": l3c.CAPS, "teacher_calls": 0}
    p["hash"] = cf.stable_hash(p)
    OUT.mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "L3D_plan.json", p)


def train_job(name, kind, seed, updates) -> None:
    plan = l3._rd(OUT / "L3D_plan.json")
    train, val, test = data(visual=kind != "viszero")
    ctx = ufk.installed() if kind == "keyed" else contextlib.nullcontext()
    with ctx:
        c0 = l3c.time.process_time()
        m, rec = l3.fit("B-full", seed, updates, train, val, "unique")
        rec["train_cpu_s"] = round(l3c.time.process_time() - c0, 1)
        jl.PEAK[0] = 0
        rec["eval"] = l3.evaluate9(m, "unique", val, test)
        rec["peak_scope"] = jl.PEAK[0]
    ck = OUT / "ckpt" / f"ckpt_{name}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ck)
    rec |= {"name": name, "kind": kind, "seed": seed, "plan_hash": plan["hash"], "checkpoint_sha256": l3.sha(ck),
            "model_version": ufk.VERSION if kind == "keyed" else uf.VERSION, "vision_input": "zero tensor" if kind == "viszero" else "DINOv2",
            "vproj_weight_norm": float(m.vproj.weight.norm()), "vproj_bias_norm": float(m.vproj.bias.norm())}
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "runs" / f"{name}.json", rec)


def run_l3d() -> dict:
    plan, status = l3._rd(OUT / "L3D_plan.json"), {}
    jobs = plan["visual_control"]["jobs"]
    for j in jobs:
        if not (OUT / "runs" / f"{j['name']}.json").exists():
            status[j["name"]] = l3c.job(j["name"].replace("#", "_"), f"train_job({j['name']!r}, {j['kind']!r}, {j['seed']}, {j['updates']})", 1500, "p9_l3d")["stop"]
    if plan["l3c1_gate"] == "PASS" and l3c.CAPS["aggregate_cpu_s"] - l3c.spent()[0] - 700 >= 1200:
        for j in plan["keyed_optional"]["jobs"]:
            if not (OUT / "runs" / f"{j['name']}.json").exists():
                status[j["name"]] = l3c.job(j["name"].replace("#", "_"), f"train_job({j['name']!r}, {j['kind']!r}, {j['seed']}, {j['updates']})", 1500, "p9_l3d")["stop"]
    else:
        status["keyed_optional"] = "NOT_RUN"
    out = {"status": status, "aggregate_cpu_s": round(l3c.spent()[0], 1), "plan_hash": plan["hash"]}
    cf.write_json(OUT / "run_status.json", out)
    return out


# ------------------------------------------------------------------ validation-only comparison
def _run(name):
    p = OUT / "runs" / f"{name}.json"
    return l3._rd(p if p.exists() else L3B / f"runs/{name}.json")


def _contrast(a_names, b_names, split="validation") -> dict:
    """Seed-averaged per-root NLL difference a - b, root-cluster bootstrap, top-3 share and leave-one-root-out range."""
    pa, pb = (collections.defaultdict(list) for _ in range(2))
    for names, d in ((a_names, pa), (b_names, pb)):
        for n in names:
            for k, (nll, _) in _run(n)["eval"][split]["per_root"].items():
                d[k].append(nll)
    ks = sorted(set(pa) & set(pb))
    diff = np.array([np.mean(pa[k]) - np.mean(pb[k]) for k in ks])
    rng = np.random.default_rng(0)
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    loo = [np.delete(diff, i).mean() for i in range(len(diff))]
    tot = {nm: sorted((np.mean(d[k]) for k in ks), reverse=True) for nm, d in (("a", pa), ("b", pb))}
    return {"mean": float(diff.mean()), "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))], "roots": len(ks),
            "leave_one_root_out": [float(min(loo)), float(max(loo))], "sign_stable_under_loo": bool(min(loo) * max(loo) > 0),
            "top3_share": {nm: float(sum(v[:3]) / sum(v)) for nm, v in tot.items()},
            "largest_root_differences": dict(sorted(zip(ks, diff.round(4).tolist()), key=lambda kv: -abs(kv[1]))[:3])}


@torch.no_grad()
def provenance(names) -> dict:
    """Validation repair NLL (root prior, calibrated) split by label source, recomputed from the stored checkpoints."""
    out = {}
    for n in names:
        rec = _run(n)
        vis = rec.get("kind") != "viszero"
        _, val, _ = data(visual=vis)
        m = l3.load("B-full", (OUT / "ckpt" if (OUT / "ckpt" / f"ckpt_{n}.pt").exists() else L3B / "ckpt") / f"ckpt_{n}.pt")
        tau, src = rec["eval"]["root"]["tau"], collections.defaultdict(list)
        ctx = ufk.installed() if rec.get("kind") == "keyed" else contextlib.nullcontext()
        with ctx:
            for r in val:
                for c in l3.root_eval9(m, r, "unique")["root"]["cands"]:
                    if c["y"] is not None:
                        p = min(max(1 / (1 + math.exp(-c["u"] / tau)), 1e-9), 1 - 1e-9)
                        src["operational (BUDGET_INFEASIBLE)" if c["source"] == "PLANNER_BUDGET" else "witness / certificate"].append(-math.log(p if c["y"] else 1 - p))
        out[n] = {k: [float(np.mean(v)), len(v)] for k, v in src.items()}
    return out


def summarize() -> None:
    vis, zero = [REF[s] for s in (0, 1, 2)], [f"V0-B-full#s{s}#u400" for s in (0, 1, 2)]
    keyed = [f"K-B-full#s{s}" for s in (0, 1, 2) if (OUT / "runs" / f"K-B-full#s{s}.json").exists()]
    rows = {}
    for n in vis + zero + keyed + [f"U-B-full#s{s}" for s in (0, 1, 2)]:
        r, e = _run(n), _run(n)["eval"]
        rows[n] = {"val_root_nll": e["validation"]["root_nll"], "val_nll_per_query": e["validation"]["nll_per_observed_query"],
                   "val_repair_nll": e["root"]["validation"]["repair"]["nll"], "val_repair_acc": e["root"]["validation"]["repair"]["accuracy"],
                   "val_ece": e["root"]["validation"]["repair"]["ece"], "val_scene_nll": e["root"]["validation"]["intervention"]["scene_nll"],
                   "val_kappa": e["root"]["validation"]["intervention"]["kappa_by_sign"], "val_cond": e["validation"]["offline_conditionals"],
                   "val_funnel": e["root"]["validation"]["funnel"], "test_root_nll_descriptive": e["test_disclosed"]["root_nll"],
                   "best_update": r["best_update"], "clip_fraction": r["clip_fraction"], "curve": [(h["update"] + 1, h["val"]) for h in r["history"] if "val" in h],
                   "sensitivity": e.get("sensitivity"), "vproj_grad_mean": float(np.mean([h["grad_by_module"].get("vproj", 0) for h in r["history"] if "grad_by_module" in h]))}
    con = {"visual: DINOv2 - zero (400 upd)": _contrast(vis, zero)}
    if keyed:
        con["keyed - unique (200 upd)"] = _contrast(keyed, [f"U-B-full#s{s}" for s in (0, 1, 2)][:len(keyed)])
    cf.write_json(OUT / "summary.json", {"rows": rows, "contrasts_validation": con, "provenance_validation": provenance(vis + zero + keyed),
                                         "aggregate_cpu_s": l3c.spent()[0]})


# ------------------------------------------------------------------ regression, gate, freeze (appended after every L3d training job finished)
GROUPS = {"full-vision U-B-full (400)": [REF[s] for s in (0, 1, 2)], "visual-zero U-B-full (400)": [f"V0-B-full#s{s}#u400" for s in (0, 1, 2)],
          "keyed K-B-full (200)": [f"K-B-full#s{s}" for s in (0, 1, 2)], "unique U-B-full (200)": [f"U-B-full#s{s}" for s in (0, 1, 2)]}


@torch.no_grad()
def calibration() -> dict:
    """Validation reliability bins (calibrated root-prior p vs observed), pooled over seeds, per model group."""
    out = {}
    for g, names in GROUPS.items():
        pts = []
        for n in names:
            if not ((OUT / "runs" / f"{n}.json").exists() or (L3B / f"runs/{n}.json").exists()):
                continue
            rec = _run(n)
            _, val, _ = data(visual=rec.get("kind") != "viszero")
            m = l3.load("B-full", (OUT / "ckpt" if (OUT / "ckpt" / f"ckpt_{n}.pt").exists() else L3B / "ckpt") / f"ckpt_{n}.pt")
            ctx = ufk.installed() if rec.get("kind") == "keyed" else contextlib.nullcontext()
            with ctx:
                for r in val:
                    pts += [(1 / (1 + math.exp(-c["u"] / rec["eval"]["root"]["tau"])), c["y"]) for c in l3.root_eval9(m, r, "unique")["root"]["cands"] if c["y"] is not None]
        if pts:
            p, y = np.array(pts).T
            b = np.minimum((p * 5).astype(int), 4)
            out[g] = [(float(p[b == i].mean()), float(y[b == i].mean()), int((b == i).sum())) for i in range(5) if (b == i).any()]
    return out


def finish() -> None:
    summarize()
    s = l3._rd(OUT / "summary.json")
    s["calibration_validation"] = calibration()
    cf.write_json(OUT / "summary.json", s)


def regression() -> None:
    """Historical suite + six archived replays (planner FIXTURE calls counted separately; isolated from learned inference)."""
    d = l3c.OUT / "regression"
    d.mkdir(parents=True, exist_ok=True)
    (d / "countcalls.py").write_text(l3c.Path("runs/p8/audit/regression/countcalls.py").read_text())
    l3c.subprocess.run([l3c.sys.executable, "run.py", "g4c-lf-p4-regression", "--out", str(d), "--cpu", "1400", "--wall", "2900"], capture_output=True, text=True)


def gate_l3d() -> dict:
    plan, s = l3._rd(OUT / "L3D_plan.json"), l3._rd(OUT / "summary.json")
    runs = {p.stem: l3._rd(p) for p in (OUT / "runs").glob("*.json")}
    vis = [j["name"] for j in plan["visual_control"]["jobs"]]
    gs = [l3._rd(p) for p in (l3c.OUT / "guards").glob("*.json")]
    reg = l3._rd(l3c.OUT / "regression/regression_result.json") if (l3c.OUT / "regression/regression_result.json").exists() else None
    _, val, _ = data(visual=False)
    m = l3.load("B-full", OUT / "ckpt/ckpt_V0-B-full#s0#u400.pt")
    with torch.no_grad():
        replay = float(uf.total(m, val)[1]["nll"])
    inv = {n: max(v for k, v in r["eval"]["sensitivity"].items() if k.endswith("EXPECTED_INVARIANT")) for n, r in runs.items()}
    checks = {"l3c0_pass": plan["l3c0_gate"] == "PASS",
              "plan_frozen_before_training": (OUT / "L3D_plan.json").stat().st_mtime < min(p.stat().st_mtime for p in (OUT / "runs").glob("*.json")),
              "freeze_rule_written_before_keyed_results": all((OUT / "freeze_rule.json").stat().st_mtime < (OUT / "runs" / f"{n}.json").stat().st_mtime
                                                             for n in runs if n.startswith("K-")),
              "visual_control_complete_3_seeds": all(n in runs for n in vis),
              "plan_hash_in_every_run": all(r["plan_hash"] == plan["hash"] for r in runs.values()),
              "same_architecture_and_parameters": all(r["params"] == 698154 for r in runs.values()),
              "reference_checkpoints_verified": all(l3.sha(L3B / f"ckpt/ckpt_{v['run']}.pt") == v["checkpoint_sha256"] for v in plan["visual_control"]["reference"].values()),
              "no_nonfinite": not any(r["nonfinite"] for r in runs.values()),
              "trained_models_invariant": all(v < 1e-9 for v in inv.values()),
              "checkpoint_replay_reproduces_val_nll": abs(replay - runs["V0-B-full#s0#u400"]["val_root_nll"]) < 1e-9,
              "aggregate_budget": l3c.spent()[0] <= l3c.CAPS["aggregate_cpu_s"] and l3c.spent()[1] <= l3c.CAPS["aggregate_wall_s"],
              "per_job_caps": all(g["cpu_s"] <= l3c.CAPS["job_cpu_s"] + 30 and g["wall_s"] <= l3c.CAPS["job_wall_s"] for g in gs),
              "historical_suite_and_replays": reg is not None and reg["pytest_exit"] == "0" and reg["tests"]["failed"] == 0 and all(reg["replays"].values()),
              "zero_physical_calls_learned_paths": sum(l3._rd("runs/p9/l3c/l3c1/artifacts.json")["physical_module_calls"].values()) == 0}
    c = s["contrasts_validation"]
    sci = {"visual": ("DINOv2 better on validation" if c["visual: DINOv2 - zero (400 upd)"]["ci95"][1] < 0 else
                      "visual-zero better on validation" if c["visual: DINOv2 - zero (400 upd)"]["ci95"][0] > 0 else "INCONCLUSIVE"),
           "keyed_vs_unique": (("keyed worse" if c["keyed - unique (200 upd)"]["ci95"][0] > 0 else "keyed better" if c["keyed - unique (200 upd)"]["ci95"][1] < 0
                                else "INCONCLUSIVE (no significant difference)") if "keyed - unique (200 upd)" in c else "NOT_RUN")}
    g = {"stage": "P9-L3d", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "scientific_verdict": sci,
         "trained_max_invariance_change": inv, "regression": reg and {"tests": reg["tests"], "replays": reg["replays"], "cpu_s": reg["guard"]["cpu_s"],
                                                                      "planner_fixture_calls": reg["calls"]},
         "aggregate_cpu_s": round(l3c.spent()[0], 1), "aggregate_wall_s": round(l3c.spent()[1], 1)}
    cf.write_json(OUT / "L3D_GATE.json", g)
    return g


def freeze() -> dict:
    """Apply the pre-registered freeze rule (runs/p9/l3d/freeze_rule.json) and write runs/p9/2d_freeze_manifest.json."""
    rule, s, g = l3._rd(OUT / "freeze_rule.json"), l3._rd(OUT / "summary.json"), l3._rd(OUT / "L3D_GATE.json")
    rows, con = s["rows"], s["contrasts_validation"]
    gates = {"L3C_0": l3._rd("runs/p9/l3c/L3C_0_GATE.json")["verdict"], "L3C_1": l3._rd("runs/p9/l3c/L3C_1_GATE.json")["verdict"], "L3D": g["verdict"]}
    mean = lambda names, k: float(np.mean([rows[n][k] for n in names]))  # noqa: E731
    K, U2 = GROUPS["keyed K-B-full (200)"], GROUPS["unique U-B-full (200)"]
    elig = {}
    if all(n in rows for n in K):
        kc = con["keyed - unique (200 upd)"]
        elig = {"root_nll_not_significantly_worse": kc["ci95"][0] <= 0,
                "repair_accuracy_drop_le_0.03": mean(U2, "val_repair_acc") - mean(K, "val_repair_acc") <= 0.03,
                "scene_nll_increase_le_0.05": mean(K, "val_scene_nll") - mean(U2, "val_scene_nll") <= 0.05}
    if "FAIL" in gates.values():
        dec = "FREEZE_BLOCKED"
    elif gates["L3C_1"] == "PASS" and elig and all(elig.values()):
        dec = "FREEZE_KEYED"
    else:
        dec = "FREEZE_UNIQUE"
    fam = K if dec == "FREEZE_KEYED" else GROUPS["full-vision U-B-full (400)"]
    best = min(fam, key=lambda n: rows[n]["val_root_nll"]) if dec != "FREEZE_BLOCKED" else None
    vc = con["visual: DINOv2 - zero (400 upd)"]
    rec = _run(best) if best else {}
    ck = (OUT / "ckpt" if (OUT / "ckpt" / f"ckpt_{best}.pt").exists() else L3B / "ckpt") / f"ckpt_{best}.pt" if best else None
    man = {"decision": dec, "rule_hash": rule["hash"], "gates": gates, "keyed_eligibility": elig,
           "model": {"family": "ABC7_V1 B-full factor modules (abc8.build('B-full'), vision=True)",
                     "root_model": "ABC9K_KEYED_V1 (unique_factor_keyed.py)" if dec == "FREEZE_KEYED" else "ABC9U_UNIQUE_V1 (unique_factor.py)",
                     "factor_ontology": "see responses-phases/0034 (unique) / 0035 (keyed); five families F1-F5, F1 and J outside P(z|r); one variable per physical query",
                     "allowed_sensitivity": ("none for re-listing modelled queries; changes only when new physical outcomes or relations are modelled" if dec == "FREEZE_KEYED" else
                                             "re-listing a modelled omission as a candidate adds role-specific scene-query pairs and an HO block (B-full <= 0.04, C-full <= 0.30 at frozen weights)"),
                     "parameters": 698154},
           "objective": {"terms": "mean_roots(-log P(D_c)) + 1.0 static BCE (endpoint-deduplicated) + 1.0 log-Huber witnessed J", "T": 1.0,
                         "optimizer": "AdamW lr 1e-3 wd 1e-4, clip 10, RootSampler(train, 12, seed)",
                         "updates": 200 if dec == "FREEZE_KEYED" else 400, "selection": "best validation own root NLL every 20 updates"},
           "visual_features": {"mode": "DINOv2-small ed25f3a3, mask-pooled, frozen; ENABLED", "visual_zero_contrast_validation": vc,
                               "rule": "kept enabled unless visual-zero is better with CI excluding 0 and LOO-stable"},
           "checkpoint": {"name": best, "path": str(ck) if ck else None, "sha256": l3.sha(ck) if ck else None, "seed": rec.get("seed"),
                          "val_root_nll": rec.get("val_root_nll"), "tau": rec.get("eval", {}).get("root", {}).get("tau"),
                          "threshold": rec.get("eval", {}).get("root", {}).get("threshold"),
                          "other_seeds": {n: rows[n]["val_root_nll"] for n in fam}},
           "selector": "static-certificate precedence; root-prior admission sigmoid(logit p / tau) >= threshold; then (moves, J_pred, C_d, "
                       "E_static + T softplus(-logit p), id); empty admitted set = ABSTAIN",
           "inference_limits": {"exact_elimination_MAX_SCOPE": jl.MAX_SCOPE, "active_slots_per_layout": 9, "fail_closed": ["CombinatoricsError", "IdentityError"]},
           "data": {"dataset_sha256": l3.sha(l3.DEV), "test_sha256": l3.sha(l3.TEST), "splits": "178 train / 12 validation (P8c); 24 disclosed test (diagnostic only)"},
           "teacher_status": {"VERIFIED": 1, "CERTIFIED": 0, "BUDGET_INFEASIBLE": "0 under OPNEG-v1 (operational)", "UNKNOWN": "missing", "NOT_RUN": "missing"},
           "sources": {f: l3.sha(f) for f in ("unique_factor.py", "unique_factor_keyed.py", "joint_likelihood.py", "abc7_models.py", "abc_models.py",
                                              "abc7_train.py", "abc8.py", "p7_eval.py", "p9_l3.py", "p9_l3c.py", "p9_l3d.py")},
           "not_established": ["no proof-grounded kappa=+1", "no decided same-endpoint order contrast in fresh data", "no omission-only labels",
                               "structured z0-zS pair unidentifiable", "12 validation roots, top-3 dominated", "no physical execution of any P9 model",
                               "G+M remains the best executed method (P8d 9/12)"]}
    man["hash"] = cf.stable_hash(man)
    cf.write_json(l3c.Path("runs/p9/2d_freeze_manifest.json"), man)
    return man


def figures() -> None:
    import p9_viz
    p9_viz.l3c_factor_graph()
    p9_viz.l3c_selection()
    p9_viz.l3d_visual()
