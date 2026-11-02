"""P10-0: validation readiness for A / B / C (saved data only; no teacher, IK or planner call in any learned path).

Ledger: every compute job runs under the process-tree guard and is charged to ONE category with its own cap
(learner 4,000 CPU-s; regression fixtures declared separately; construction, teacher and prediction lines for P10a).
TOP1 selector (deployable; frozen here): canonical identity check (fail closed) -> analytic static veto (endpoint support /
overlap and goal overlap: pure disc geometry, no robot) BEFORE learned admission -> keyed root-prior admission
sigmoid(logit p / tau) >= threshold -> (moves, predicted J, analytic C_d, energy, id); empty = ABSTAIN; nonfinite -> error.
"""
from __future__ import annotations

import collections
import contextlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import config as cf
import joint_likelihood as jl
import p9_l3 as l3
import p9_l3c as l3c
import unique_factor as uf
import unique_factor_keyed as ufk

OUT = Path("runs/p10")
P0 = OUT / "p10_0"
CAPS = {"learner": 4000.0, "regression": 1500.0, "construction": 1500.0, "teacher": 5000.0, "prediction": 600.0, "figures": 1500.0}
JOB = {"cpu_s": 1800.0, "wall_s": 3600.0, "aggregate_wall_s": 10000.0}
FROZEN = Path("runs/p9/2d_freeze_manifest.json")
ARMS = {"K-B-full": ("B-full", "keyed"), "K-C-full": ("C-full", "keyed"), "A": ("A", "factorized"), "SF": ("SF", "factorized")}


def _rd(p):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------ categorized ledger
def spent(cat=None) -> tuple:
    gs = [_rd(p) for p in (OUT / "guards").glob("*.json")]
    gs = [g for g in gs if cat is None or g["category"] == cat]
    return sum(g["cpu_s"] for g in gs), sum(g["wall_s"] for g in gs)


def job(name, call, cpu, module="p10_protocol", category="learner", wall=None) -> dict:
    import teacher
    (OUT / "guards").mkdir(parents=True, exist_ok=True)
    left = CAPS[category] - spent(category)[0]
    cpu = min(cpu, JOB["cpu_s"], left)
    if cpu < 5:
        raise SystemExit(f"{category} budget exhausted ({CAPS[category] - left:.0f} / {CAPS[category]:.0f} CPU-s)")
    g = teacher.guard([sys.executable, "-c", f"import torch; torch.set_num_threads(1); import {module}; {module}.{call}"], cpu,
                      wall or JOB["wall_s"], OUT / "heartbeat.txt", 15.0, 0.25, Path("."))
    n = 1
    while (OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json").exists():
        n += 1
    rec = {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | \
          {"cpu_cap_s": cpu, "call": f"{module}.{call}", "category": category, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cf.write_json(OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json", rec)
    (OUT / "heartbeat.txt").unlink(missing_ok=True)
    print(json.dumps({k: rec[k] for k in ("stop", "returncode", "cpu_s", "wall_s")} | {category: round(spent(category)[0], 1)}))
    return rec


# ------------------------------------------------------------------ physical-call instrumentation (IK / collision / screen / planner / teacher)
PHYS_MODULES = ("planner", "skill", "ordered", "verify", "evaluate", "sector_check", "sector_cert", "operational_teacher", "teacher")
PHYS_FUNCS = {"geometry": ("ik", "certify", "make_world", "bodies", "core_distance", "points_to_core"),
              "motion": ("checkers", "ik_track", "ik_line", "carry_line", "certify_chain", "certify_trajectory", "object_check", "check_record")}


def instrument() -> collections.Counter:
    """Import and wrap every IK / collision / screen / planner / teacher entry point with a counter. Forward kinematics and
    pure disc geometry (proposals.static_endpoints, motion._band) are deliberately NOT counted: they are not feasibility checks."""
    import importlib
    import types
    calls = collections.Counter()
    targets = [(m, None) for m in PHYS_MODULES] + [(m, fs) for m, fs in PHYS_FUNCS.items()]
    for name, only in targets:
        mod = importlib.import_module(name)
        for k, f in list(vars(mod).items()):
            if isinstance(f, types.FunctionType) and f.__module__ == name and (only is None or k in only):
                def w(*a, _f=f, _n=f"{name}.{k}", **kw):
                    calls[_n] += 1
                    return _f(*a, **kw)
                setattr(mod, k, w)
    chk = sys.modules["geometry"].Checker
    init = chk.__init__

    def ci(self, *a, **kw):
        calls["geometry.Checker"] += 1
        return init(self, *a, **kw)
    chk.__init__ = ci
    return calls


# ------------------------------------------------------------------ deployable TOP1 selector
def static_veto(scene, cand, goal_cell, cfg) -> bool:
    """True if the candidate's endpoints are analytically invalid (support / overlap) or leave every goal pose occupied.
    Pure disc geometry from learner-visible poses: no robot, IK, collision sweep or planner."""
    import data
    import proposals as pr
    cells = {o: d for o, d in cand}
    act = data.make_query(cfg, "veto", scene, "t0", goal_cell, {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
    return pr.static_endpoints(scene, cells, "t0", cfg)["status"] == "INVALID" or bool(pr.goal_blockers(scene, cells, act, cfg))


@torch.no_grad()
def top1(m, r, tau, thr, vetoed=frozenset(), keyed=True) -> dict:
    """One choice (or ABSTAIN) for prepared root r; `vetoed` = candidate ids failing the analytic static veto."""
    l3c.check_identity(r)
    ctx = ufk.installed() if keyed else contextlib.nullcontext()
    with ctx:
        _, rows = l3c.deployed(m, r, uf.instances(m, r), tau, thr) if m.kind in ("B", "C") else (None, l3.root_eval9(m, r, "factorized")["root"]["cands"])
    for x in rows:
        if not all(math.isfinite(x[k]) for k in ("u", "J_pred", "C_d", "energy")):
            raise FloatingPointError(f"{r['key']}: nonfinite score for {x['query']}")
    keep = [x for x in rows if x["query"] not in vetoed and 1 / (1 + math.exp(-x["u"] / tau)) >= thr]
    best = min(keep, key=lambda x: (x["m"], x["J_pred"], x["C_d"], x["energy"], x["query"])) if keep else None
    return {"choice": best["query"] if best else "ABSTAIN", "vetoed": sorted(vetoed), "admitted": len(keep), "scored": len(rows)}


# ------------------------------------------------------------------ audit (nonmutating)
def audit() -> None:
    P0.mkdir(parents=True, exist_ok=True)
    fz = _rd(FROZEN)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.rstrip()  # noqa: E731
    st = git("status", "--porcelain").splitlines()
    tracked = set(git("ls-files").splitlines())
    ck = Path(fz["checkpoint"]["path"])
    man = {**{f"l3b/{k}": (Path("runs/p9/l3/l3b/ckpt") / k, v["sha256"]) for k, v in _rd("runs/p9/l3/l3b/ckpt/checkpoint_manifest.json").items()},
           **{f"l3d/{k}": (Path("runs/p9/l3d/ckpt") / k, v["sha256"]) for k, v in _rd("runs/p9/l3d/ckpt/checkpoint_manifest.json").items() if isinstance(v, dict)}}
    train, val, test = l3.data()
    ident = collections.Counter()
    for r in train + val + test:
        try:
            l3c.check_identity(r)
            ident["ok"] += 1
        except jl.IdentityError:
            ident["failed"] += 1
    m = l3.load("B-full", ck)
    with ufk.installed():
        replay = float(uf.total(m, val)[1]["nll"])
    orig = uf.instances
    try:
        with ufk.installed():
            raise RuntimeError("probe")
    except RuntimeError:
        pass
    calls = instrument()
    plan = _rd("runs/p8/collection/plan.json")
    import p9_l3d
    cfg = cf.load_config("config.toml")
    from scene import scene_from_dict
    vetoes = {}
    for r in val:
        pr_ = plan["roots"].get(r["key"])
        if pr_:
            s = scene_from_dict(pr_["scene"])
            ids = r["ids"]
            vetoes[r["key"]] = sorted(c["query"] for c in r["cands"] if c["seq"] and static_veto(
                s, [(ids[i], l3.cell(d)) for i, d in c["seq"]], pr_["goal_cell"], cfg))
    out = {"head": git("rev-parse", "HEAD"), "origin_main": git("rev-parse", "origin/main"),
           "tracked_changes": [x for x in st if not x.startswith("??")], "untracked_count": sum(x.startswith("??") for x in st),
           "freeze_manifest_hash_ok": cf.stable_hash({k: v for k, v in fz.items() if k != "hash"}) == fz["hash"],
           "freeze_decision": fz["decision"], "freeze_sources_match": {f: l3.sha(f) == h for f, h in fz["sources"].items()},
           "frozen_checkpoint": {"path": str(ck), "committed": str(ck) in tracked, "sha_ok": l3.sha(ck) == fz["checkpoint"]["sha256"]},
           "local_checkpoints_sha_ok": {k: p.exists() and l3.sha(p) == h for k, (p, h) in man.items()},
           "datasets_sha_ok": l3.sha(l3.DEV) == fz["data"]["dataset_sha256"] and l3.sha(l3.TEST) == fz["data"]["test_sha256"],
           "canonical_identity": dict(ident), "frozen_val_root_nll": {"stored": fz["checkpoint"]["val_root_nll"], "replay": replay},
           "keyed_patch_restored_after_exception": uf.instances is orig and uf.instances is not ufk.instances,
           "static_veto_validation": vetoes, "static_veto_physical_calls": dict(calls),
           "historical_freeze_rule_hash": _rd("runs/p9/l3d/freeze_rule.json")["hash"], "p9_l3d_module": p9_l3d.__name__}
    cf.write_json(P0 / "audit.json", out)


# ------------------------------------------------------------------ pre-registration and training plan (written before any new label)
def preregister() -> None:
    from p10_prereg import PREREG
    p = dict(PREREG)
    p["hash"] = cf.stable_hash(p)
    cf.write_json(P0 / "P10_0_PREREG.json", p)
    t = {"version": "P10_0_TRAIN_PLAN_V1", "prereg_hash": p["hash"], "data": "P8c learner view: 178 train / 12 validation roots; disclosed test descriptive only",
         "schedule": {"updates": 400, "batch": 12, "sampler": "abc8.RootSampler(train, 12, seed)", "optimizer": "AdamW 1e-3 / 1e-4", "clip": 10.0,
                      "selection": "best validation own-likelihood root NLL every 20 updates"},
         "objectives": {"keyed B / C": "mean_roots(-log P_K(D_c)) + 1 static BCE + 1 log-Huber witnessed J (unique_factor_keyed.installed)",
                        "A / SF": "documented factorized objective (0029 1.8): sum of Bernoulli log-losses over the same distinct observed queries "
                                  "+ 1 static BCE + 1 log-Huber J (joint_likelihood.losses factorized branch); NOT a joint NLL"},
         "jobs": [{"name": f"{a}#s{s}#u400", "arm": a, "seed": s, "priority": pr} for pr, arms in ((1, ("K-B-full", "K-C-full")), (2, ("A",)), (3, ("SF",)))
                  for a in arms for s in (0, 1, 2)],
         "rule": "priorities 1, 2, 3 in order; a priority-2/3 job starts only if >= 450 learner CPU-s remain; otherwise NOT_RUN",
         "sources": {f: l3.sha(f) for f in ("unique_factor.py", "unique_factor_keyed.py", "joint_likelihood.py", "abc7_models.py", "p9_l3.py",
                                           "p9_l3c.py", "p10_protocol.py", "p10_prereg.py")}, "caps": CAPS}
    t["hash"] = cf.stable_hash(t)
    cf.write_json(P0 / "P10_0_TRAIN_PLAN.json", t)


def train_job(name, arm, seed, updates) -> None:
    base, kind = ARMS[arm]
    train, val, test = l3.data()
    ctx = ufk.installed() if kind == "keyed" else contextlib.nullcontext()
    with ctx:
        c0 = time.process_time()
        m, rec = l3.fit(base, seed, updates, train, val, "unique" if kind == "keyed" else "poe")
        rec["train_cpu_s"] = round(time.process_time() - c0, 1)
        rec["eval"] = l3.evaluate9(m, "unique" if kind == "keyed" else "factorized", val, test)
    ck = P0 / "ckpt" / f"ckpt_{name}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ck)
    rec |= {"name": name, "arm": arm, "kind": kind, "seed": seed, "checkpoint_sha256": l3.sha(ck), "plan_hash": _rd(P0 / "P10_0_TRAIN_PLAN.json")["hash"],
            "model_version": ufk.VERSION if kind == "keyed" else "factorized (no joint graph)"}
    (P0 / "runs").mkdir(parents=True, exist_ok=True)
    cf.write_json(P0 / "runs" / f"{name}.json", rec)


def run_training() -> dict:
    plan, status = _rd(P0 / "P10_0_TRAIN_PLAN.json"), {}
    for j in plan["jobs"]:
        if (P0 / "runs" / f"{j['name']}.json").exists():
            status[j["name"]] = "RUN"
        elif j["priority"] > 1 and CAPS["learner"] - spent("learner")[0] < 450:
            status[j["name"]] = "NOT_RUN (learner budget)"
        else:
            g = job(j["name"].replace("#", "_"), f"train_job({j['name']!r}, {j['arm']!r}, {j['seed']}, 400)", 1800)
            status[j["name"]] = "RUN" if g["returncode"] == 0 and (P0 / "runs" / f"{j['name']}.json").exists() else f"FAILED:{g['stop']}"
    cf.write_json(P0 / "run_status.json", {"status": status, "learner_cpu_s": round(spent("learner")[0], 1)})
    return status


# ------------------------------------------------------------------ P10-0 summary and gate (validation decides; disclosed test descriptive)
def _per_root(rec, split="validation") -> dict:
    """Own-likelihood NLL of the observed bits per independent root (reference + obstructed variant of one lineage summed)."""
    out = collections.defaultdict(lambda: [0.0, 0])
    for k, (nll, n) in rec["eval"][split]["per_root"].items():
        x = out[k.removesuffix("_ref")]
        x[0], x[1] = x[0] + nll, x[1] + n
    return dict(out)


def contrast(runs, a, b) -> dict:
    """Seed-averaged per-root NLL difference a - b, root-cluster bootstrap CI and leave-one-root-out range (same bits required)."""
    import p10_eval as pe
    ra, rb = ([_per_root(r) for r in runs if r["arm"] == x] for x in (a, b))
    if not ra or not rb:
        return {"status": "NOT_RUN"}
    keys = sorted(ra[0])
    if any(r[k][1] != ra[0][k][1] for r in ra + rb for k in keys):
        return {"status": "INCOMPARABLE (different observed bits)"}
    d = [float(np.mean([r[k][0] for r in ra]) - np.mean([r[k][0] for r in rb])) for k in keys]
    loro = [float(np.mean(d[:i] + d[i + 1:])) for i in range(len(d))]
    return {"status": "OK", "mean": float(np.mean(d)), "ci95": pe.boot(d), "loro": [min(loro), max(loro)],
            "sign_stable": all(x < 0 for x in loro) or all(x > 0 for x in loro), "roots": len(d),
            "largest": sorted(zip(keys, d), key=lambda x: -abs(x[1]))[:3]}


def summarize() -> dict:
    runs = [_rd(f) for f in sorted((P0 / "runs").glob("*.json"))]
    rows = []
    for r in runs:
        rv, rt = r["eval"]["root"]["validation"], r["eval"]["root"]["test_disclosed"]
        rows.append({"name": r["name"], "arm": r["arm"], "seed": r["seed"], "objective": "own exact keyed joint" if r["kind"] == "keyed" else "factorized",
                     "val_root_nll": r["eval"]["validation"]["root_nll"], "val_nll_per_query": r["eval"]["validation"]["nll_per_observed_query"],
                     "repair_nll": rv["repair"]["nll"], "repair_acc": rv["repair"]["accuracy"], "repair_ece": rv["repair"]["ece"],
                     "scene_nll": rv["intervention"]["scene_nll"], "kappa_macro_mae": rv["intervention"]["kappa_macro_mae"],
                     "funnel": rv["funnel"], "tau": r["eval"]["root"]["tau"], "threshold": r["eval"]["root"]["threshold"],
                     "test_root_nll_descriptive": r["eval"]["test_disclosed"]["root_nll"], "test_funnel_descriptive": rt["funnel"],
                     "best_update": r["best_update"], "clip": r["clip_fraction"], "train_cpu_s": r["train_cpu_s"], "nonfinite": r["nonfinite"]})
    con = {f"{a} - {b}": contrast(runs, a, b) for a, b in (("K-B-full", "A"), ("K-C-full", "K-B-full"), ("K-B-full", "SF"), ("K-C-full", "A"))}
    st = _rd(P0 / "run_status.json")["status"] if (P0 / "run_status.json").exists() else {}
    g = {n: _rd(f) for n, f in ((f.stem, f) for f in (OUT / "guards").glob("*.json"))}
    lg = [x for x in g.values() if x.get("category", "learner") == "learner"]
    checks = {"audit_pass": all(_rd(P0 / "audit.json")[k] for k in ("freeze_manifest_hash_ok", "datasets_sha_ok", "keyed_patch_restored_after_exception"))
              and _rd(P0 / "audit.json")["frozen_val_root_nll"]["replay"] == _rd(P0 / "audit.json")["frozen_val_root_nll"]["stored"],
              "prereg_before_training": all(r["plan_hash"] == _rd(P0 / "P10_0_TRAIN_PLAN.json")["hash"] for r in runs),
              "priority1_complete": all(st.get(f"{a}#s{s}#u400") == "RUN" for a in ("K-B-full", "K-C-full") for s in (0, 1, 2)),
              "A_documented_objective_run": any(r["arm"] == "A" for r in runs), "no_nonfinite": not any(r["nonfinite"] for r in runs),
              "learner_cap": spent("learner")[0] <= CAPS["learner"], "per_job_caps": all(x["cpu_s"] <= JOB["cpu_s"] and x["wall_s"] <= JOB["wall_s"] for x in lg),
              "aggregate_wall": sum(x["wall_s"] for x in lg) <= JOB["aggregate_wall_s"]}
    out = {"stage": "P10-0", "runs": rows, "contrasts_val_root_nll": con, "status": st, "learner_cpu_s": round(spent("learner")[0], 1),
           "learner_wall_s": round(sum(x["wall_s"] for x in lg), 1), "checks": checks, "engineering": "PASS" if all(checks.values()) else "FAIL",
           "note": "validation (12 P8c roots) decides; disclosed test descriptive only; own-likelihood NLL of the same observed bits"}
    cf.write_json(OUT / "P10_0_GATE.json", out)
    return out


def regression() -> None:
    """Declared historical suite + six archived replays (category 'regression'; planner FIXTURE calls counted by countcalls, never learned scoring)."""
    d = P0 / "regression"
    d.mkdir(parents=True, exist_ok=True)
    (d / "countcalls.py").write_text(Path("runs/p8/audit/regression/countcalls.py").read_text())
    subprocess.run([sys.executable, "run.py", "g4c-lf-p4-regression", "--out", str(d), "--cpu", "1400", "--wall", "3400"], capture_output=True, text=True)
