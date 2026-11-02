"""P16 ledger (runs/p16/guards, contract caps, one guarded OS process per job), P16-0 non-mutating source / evidence audit, stage gates,
statistics and manifests. P10-P15 artifacts are read-only; every P16 output is a new file under runs/p16."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import config as cf

OUT = Path("runs/p16")
CAPS = {"audit": 3000.0, "motion": 9000.0, "shared": 10000.0, "b_dev": 10000.0, "d_dev": 10000.0, "data": 55000.0, "training": 20000.0,
        "execution": 38000.0, "final": 6000.0}
JOB = {"cpu_s": 8000.0, "wall_s": 80000.0}
HIST_PATHS = ("runs/p10", "runs/p10b", "runs/p10c", "runs/p11", "runs/p12", "runs/p13", "runs/p14", "runs/p15", "visualizations")
CORE = ("motion.py", "geometry.py", "skill.py", "planner.py", "data.py", "certificates.py", "sector_cert.py", "sector_check.py", "ordered.py",
        "unique_factor.py", "unique_factor_keyed.py", "abc_models.py", "abc7_models.py", "abc7_train.py", "abc8.py", "d_blockers.py",
        "learn_data.py", "p10c_train_eval.py", "p13_factors.py", "p13_pipeline.py", "p14_sweep_index.py", "p14_d_search.py", "p15_retarget.py",
        "p15_lookup.py", "p15_search.py", "p15_bench.py", "config.toml")


def rd(p):
    return json.loads(Path(p).read_text())


def sha(p) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spent(cat=None) -> float:
    return sum(g["cpu_s"] for g in (rd(p) for p in (OUT / "guards").glob("*.json")) if cat is None or g["category"] == cat)


def job(name, call, cpu, module="p16_eval", category="audit", wall=None, env=None) -> dict:
    """One guarded child per OS process, charged to one P16 category; refuses past the cap (fail closed)."""
    import os
    import teacher
    (OUT / "guards").mkdir(parents=True, exist_ok=True)
    res_dir = OUT / "reservations"
    res_dir.mkdir(parents=True, exist_ok=True)
    reserved = sum(rd(f)["cpu"] for f in res_dir.glob(f"{category}__*.json"))
    cpu = min(cpu, JOB["cpu_s"], CAPS[category] - spent(category) - reserved)    # caps of RUNNING jobs are reserved (no oversubscription)
    if cpu < 5:
        raise SystemExit(f"{category} budget exhausted")
    resf = res_dir / f"{category}__{name}.json"
    cf.write_json(resf, {"cpu": cpu})
    old = dict(os.environ)
    os.environ.update(env or {})
    try:
        g = teacher.guard([sys.executable, "-c", f"import torch; torch.set_num_threads(1); import {module}; {module}.{call}"], cpu,
                          wall or JOB["wall_s"], OUT / f"heartbeat_{name}.txt", 15.0, 0.25, Path("."))
    finally:
        os.environ.clear()
        os.environ.update(old)
    n = 1
    while (OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json").exists():
        n += 1
    rec = {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | \
          {"cpu_cap_s": cpu, "call": f"{module}.{call}", "category": category, "env": env or {}, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cf.write_json(OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json", rec)
    (OUT / f"heartbeat_{name}.txt").unlink(missing_ok=True)
    resf.unlink(missing_ok=True)
    print(json.dumps({k: rec[k] for k in ("stop", "returncode", "cpu_s", "wall_s")} | {category: round(spent(category), 1)}))
    return rec


def gate(stage, **kw) -> None:
    cf.write_json(OUT / f"P16_{stage}_GATE.json", {"stage": f"P16-{stage}", **kw, "spend": {c: round(spent(c), 1) for c in CAPS}})


# ------------------------------------------------------------------ P16-0 audit
def _git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True).stdout


def historical_manifest() -> dict:
    """Immutable manifest: git blob ids of every tracked file under the historical paths + P10-P15 reports, plus working-tree state."""
    rows = {}
    for line in _git("ls-files", "-s", "--", *HIST_PATHS, "responses-phases").splitlines():
        meta, path = line.split("\t", 1)
        if path.startswith("responses-phases/") and not any(t in path for t in [f"{n:04d}_" for n in range(22, 82)] + ["DECISION"]):
            continue
        rows[path] = meta.split()[1]
    dirty = [ln for ln in _git("status", "--porcelain", "--", *HIST_PATHS, "responses-phases").splitlines() if not ln.startswith("??")]
    out = {"head": _git("rev-parse", "HEAD").strip(), "files": len(rows), "blobs": rows, "dirty_tracked": dirty,
           "core_sources": {p: sha(p) for p in CORE if Path(p).exists()},
           "checkpoints": {str(p): sha(p) for p in sorted(Path("runs/p13/models/ckpt").glob("*.pt"))}}
    out["hash"] = cf.stable_hash({k: v for k, v in out.items() if k != "hash"})
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "audit" / "HISTORICAL_MANIFEST.json", out)
    return out


def verify_historical() -> list:
    """Paths whose current blob differs from the P16-0 manifest (must stay empty for the whole phase)."""
    m = rd(OUT / "audit" / "HISTORICAL_MANIFEST.json")["blobs"]
    now = {}
    for line in _git("ls-files", "-s", "--", *HIST_PATHS, "responses-phases").splitlines():
        meta, path = line.split("\t", 1)
        now[path] = meta.split()[1]
    dirty = {ln[3:] for ln in _git("status", "--porcelain", "--", *HIST_PATHS, "responses-phases").splitlines() if not ln.startswith("??")}
    return sorted(p for p in m if now.get(p) != m[p] or p in dirty)


def label_audit() -> dict:
    """D label semantics in the frozen P13 datasets: which record types entered the blocker NLL (p10c_train_eval.d_items)."""
    import collections
    import os
    import torch
    os.environ["P10C_OUT"] = "runs/p13/models"
    out = {}
    for name in ("trainval", "test"):
        ds = torch.load(f"runs/p13/models/dataset_{name}.pt", weights_only=False)
        c = collections.Counter()
        for r in ds["roots"]:
            d = r.get("d") or {}
            used = d.get("mask") is not None and d.get("provenance") == "proof_grounded" and d.get("y0") == 0
            c[f"{r['audit'].get('split')}|{d.get('type')}|{d.get('provenance')}|mask_nll={used}"] += 1
        out[name] = dict(sorted(c.items()))
    return out


def privileged_audit() -> dict:
    """Static scan of runtime inputs: where candidate movers / catalogues come from in P15 and the frozen learners."""
    src = {p: Path(p).read_text() for p in ("p15_bench.py", "p15_search.py", "p14_d_search.py", "abc7_train.py", "p10c_train_eval.py")}
    hits = {p: [i + 1 for i, ln in enumerate(s.splitlines()) if "ORDERED_TEACHER" in ln or "root[\"items\"]" in ln or "dl[\"mask\"]" in ln]
            for p, s in src.items()}
    return {"lines": hits,
            "findings": {
                "p15_movers": "p15_bench.run_arm: movers = objects in ORDERED_TEACHER catalogue sequences (constructor-provided): CATALOGUE_RESTRICTED, "
                              "not scene-only. Kept as historical replay protocol; P16 uses p16_common.propose_movers (scene-only).",
                "p15_oracle": "Guide('oracle') reads the teacher blocker mask (declared non-deployable).",
                "b_catalogue": "Frozen B scores the constructor catalogue (root['items'] ORDERED_TEACHER / SCENE_TARGET_ONLY): candidate GENERATION is "
                               "privileged (mechanism-aware constructor) although labels are not read at inference. P16 B scores scene-only candidates.",
                "d_pool": "Frozen D's pool = root['parking'] (constructor lattice objects). P16 D pool = all movable non-target objects (cap) in the scene."}}


def determinism_audit() -> dict:
    import config as c
    cfg = c.load_config("config.toml")
    return {"planner_budget": "planner.plan: RRT-Connect stops at max_nodes (deterministic, seed = call_seed(planner2d.seed, key)) or "
                              f"max_wall_seconds={cfg['planner2d']['max_wall_seconds']} (safety; flagged wall_clock_cut)",
            "p15_cpu_cap": "p15_search.Physics.truncated: per-root process_time cap (150 s soft) -> search frontier depends on CPU speed / contention",
            "hash_seed": "Python str hashing is randomized per process (PYTHONHASHSEED); any order derived from set iteration is process-dependent "
                         "(tested empirically in P16-1)",
            "p15_prospective_replay": "PROSP 053: D re-run plan differed from the benchmark record (P15 0078 section 6); geometry steps equal, travel "
                                      "differed (different witness / online path)"}


def errata() -> dict:
    return {"P15_0080_lean_inference": "0080 section 5 called B-F3pm / B-F5spm / B-F5qpm 'post-hoc inference masks'. They are separately trained, "
                                       "parameter-matched checkpoints (p13_pipeline._arms: removed factor + widened scene head, PM widths 320/641/448). "
                                       "B-F5spm removes generic AND higher-order coupling (abc8 'B-structured'); it is not generic-only.",
            "P15_prospective_CI": "P15 prospective bootstrap CI [0, 0] reflects zero observed discordances; it is not an equivalence interval.",
            "D_no_clearing_set": "d_blockers.minimal_sets returns mask [0]*n for NO_CLEARING_SET_WITHIN_POOL; p10c_train_eval.d_items trained "
                                 "proof-grounded ones as an all-zero blocker mask (P13 train 2, validation 1). P16 D masks these records.",
            "P12_kappa": "P12 kappa endpoint = abs(mean_seed(kappa_hat) - kappa_true) (corrected in P13); always report the kappa==0 baseline."}


def audit() -> dict:
    m = historical_manifest()
    out = {"head": m["head"], "historical_files": m["files"], "historical_dirty_tracked": m["dirty_tracked"], "manifest_hash": m["hash"],
           "labels": label_audit(), "privileged": privileged_audit(), "determinism": determinism_audit(), "errata": errata(),
           "p15_spend": rd("runs/p15/P15_FINAL_MANIFEST.json")["spend_cpu_s"]}
    cf.write_json(OUT / "audit" / "P16_0_AUDIT.json", out)
    return out


def regression(name="regression") -> None:
    """Historical suite + P12-P16 tests + six replays (same harness as P13-P15)."""
    d = OUT / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "countcalls.py").write_text(Path("runs/p8/audit/regression/countcalls.py").read_text())
    subprocess.run([sys.executable, "run.py", "g4c-lf-p4-regression", "--out", str(d), "--cpu", "2400", "--wall", "6000"], capture_output=True, text=True)


# ------------------------------------------------------------------ P16-1 determinism / transit experiments (development roots only)
DET_ROOTS = ["p13_2026180003", "p13_2026180009", "p13_2026180012", "p13_2026180024",      # P15 retrospective (historical, development use)
             "p13_2026200000", "p13_2026200002", "p13_2026200040", "p13_2026200053"]      # P15 prospective (now historical, development use)


def hist_roots() -> dict:
    """Plan roots of the P14 test pool and the P15 prospective cohort (scene + goal + fixed metadata are all P16 reads)."""
    import p13_pipeline as pp
    P = {}
    for d in sorted(pp.HOLD.glob("*_b*")):
        P |= rd(d / "plan.json")["roots"]
    return P | rd("runs/p15/prospective/plan.json")["roots"]


def uniform_order(cells, q, moved, cands):
    return cands


def det_run(tag, keys=None, transit="planned", lookup=True, n_max=3, per_obj=1, node_cap=400) -> None:
    import numpy as np
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    P, idx = hist_roots(), pl.index_v2()[0]
    f = OUT / "shared" / f"DET_{tag}.json"
    res = rd(f) if f.exists() else {}
    for k in keys or DET_ROOTS:
        if k in res:
            continue
        c0, w0 = time.process_time(), time.time()
        task = pc.task_from_root(P[k], k)
        phys = pm.DetPhysics(task, idx, transit=transit, use_lookup=lookup)
        cells0 = {o.id: o.cell for o in phys.s0.objects}
        movers, omitted = pc.propose_movers(task, cells0)
        out = pc.search(phys, uniform_order, movers, pc.slot_fn(phys, idx), cells0, phys.s0.robot.q_rad, n_max, per_obj, node_cap)
        rep = pm.replay(phys, out["plan"]) if out["found"] else None
        res[k] = {"found": out["found"], "claim": out["claim"], "replay": rep, "movers": movers, "omitted_movers": omitted,
                  "plan": None if not out["found"] else {"moves": out["plan"]["moves"], "travel": round(out["plan"]["travel"], 9),
                                                         "steps": [(s["object"], s["dest"], s["via"], s.get("transit")) for s in out["plan"]["steps"]],
                                                         "traj": pm.traj_hash(out["plan"]["records"])},
                  "edges": out["log"]["edges"], "goals": out["log"]["goals"], "nodes": out["log"]["nodes"], "node_cap_hit": out["log"]["node_cap_hit"],
                  "used": phys.used, "stats": phys.stats, "flags": out["flags"], "cpu_s": time.process_time() - c0, "wall_s": time.time() - w0,
                  "hashseed": __import__("os").environ.get("PYTHONHASHSEED")}
        (OUT / "shared").mkdir(parents=True, exist_ok=True)
        cf.write_json(f, res)


# ------------------------------------------------------------------ P16-2 data: splits, scene-only labels, physics-only tags
DATA = OUT / "data"
SEED0 = {"dev": 2026210000, "train": 2026220000, "calib": 2026230000, "test": 2026240000, "val": 2026250000}   # <= 10 families per split
LABEL_BUDGET = {"motion_checks": 600, "online_moves": 12, "goal_plans": 24, "transit_plans": 12}
DEV_FAMS = ("NAT", "PN_SIMP", "PN_GIMP", "PC_SG", "PR_SIDES", "ORD_CHAIN", "OMIT_OPT", "TRI_OR", "TRI_CTRL", "DENSE")


def build_split(name, counts) -> None:
    """counts {family: n}; seeds SEED0[split] + 1000 * family index + j (disjoint per split). Plans frozen by p10b_dataset.build_plan."""
    import p16_common as pc
    pd = pc.register()
    split = name.split("_")[0]
    fams = sorted(counts)
    specs = [{"family": f, "seed": SEED0[split] + 1000 * i + j, "index": len(fams) * j + i, "wave": 0, "batch": 1, "split": "blind" if split == "test" else "dev"}
             for i, f in enumerate(fams) for j in range(counts[f])]
    if not (DATA / name / "plan.json").exists():
        pd.build_plan(DATA / name, specs, max_attempts=600, root_cpu=300.0)


def label_split(name, keys=None, budget=None, part="", sequences=True) -> None:
    """Scene-only catalogue + deterministic physical labels per root (sealed file for the test split)."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    plan = rd(DATA / name / "plan.json")
    idx = pl.index_v2()[0]
    f = DATA / name / (("sealed_labels" if name.startswith("test") else "labels") + (f"_{part}" if part else "") + ".json")
    res = rd(f) if f.exists() else {}
    for k in keys or sorted(plan["roots"]):
        if k in res or k in merged(name, f.name.split("_")[0]):
            continue
        c0 = time.process_time()
        task = pc.task_from_root(plan["roots"][k], k)
        phys = pm.DetPhysics(task, idx, budget=budget or LABEL_BUDGET)
        cat = pc.catalogue(task, phys, idx)
        lab = pc.label(task, phys, cat, sequences)
        res[k] = {"family": plan["roots"][k]["family"], "catalogue": {x: cat[x] for x in ("pool", "parking", "movers")}, **lab,
                  "cpu_s": time.process_time() - c0}
        cf.write_json(f, res)


def tag_split(name, keys=None, n_max=4, per_obj=2, node_cap=500, part="") -> None:
    """Physics-only mechanism tags from the scene-only expanded-domain search (uniform order): plan length, repeat move, freed-source
    reuse, and slot conflict (a VERIFIED first move whose subtree contains no plan while another first move leads to one is not observable
    cheaply; recorded instead: number of distinct VERIFIED first moves). Never conditions on any model."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    plan = rd(DATA / name / "plan.json")
    idx = pl.index_v2()[0]
    f = DATA / name / (f"tags_{part}.json" if part else "tags.json")
    res = rd(f) if f.exists() else {}
    for k in keys or sorted(plan["roots"]):
        if k in res:
            continue
        c0 = time.process_time()
        task = pc.task_from_root(plan["roots"][k], k)
        phys = pm.DetPhysics(task, idx)
        cells0 = {o.id: o.cell for o in phys.s0.objects}
        movers, omitted = pc.propose_movers(task, cells0)
        out = pc.search(phys, uniform_order, movers, pc.slot_fn(phys, idx), cells0, phys.s0.robot.q_rad, n_max, per_obj, node_cap)
        steps = out["plan"]["steps"] if out["found"] else []
        objs = [s["object"] for s in steps]
        res[k] = {"family": plan["roots"][k]["family"], "found": out["found"], "claim": out["claim"], "moves": len(steps),
                  "steps": [(s["object"], s["dest"], s["via"]) for s in steps], "repeat_move": len(objs) != len(set(objs)),
                  "freed_source_reuse": any(s["dest"] == cells0.get(p["object"]) for i, s in enumerate(steps) for p in steps[:i]),
                  "first_moves_verified": sum(1 for (o, d) in [(o, d) for o in movers for d in pc.slot_fn(phys, idx)(cells0, o)]
                                             if any(x[0] == pm.V for x in phys.memo.get(("edge", __import__("p15_search").state_key(cells0, phys.s0.robot.q_rad, ()), o, d), []))),
                  "movers": movers, "omitted": omitted, "used": phys.used, "nodes": out["log"]["nodes"], "edges": out["log"]["edges"],
                  "goals": out["log"]["goals"], "replay": pm.replay(phys, out["plan"]) if out["found"] else None, "cpu_s": time.process_time() - c0}
        cf.write_json(f, res)


def merged(name, kind) -> dict:
    """Union of per-worker files (labels / sealed_labels / tags)."""
    out = {}
    for f in sorted((DATA / name).glob(f"{kind}*.json")):
        out |= rd(f)
    return out


def launch(tag, call, cpu, category, env=None, module="p16_eval") -> None:
    """Fire-and-forget guarded job in its own session (no shell pattern matching involved)."""
    code = f"import p16_eval as p; p.job({tag!r}, {call!r}, {cpu}, module={module!r}, category={category!r}, env={env!r})"
    subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


# ------------------------------------------------------------------ frozen evaluation runners (all arms on one root share an exact cache)
def b_select(arm, rows, pol, R=None, pool_idx=None, rng=None):
    """Learned arms: p16_b.choose with the frozen calibrated policy; non-learned selectors: EMPTY-always, random, geometry."""
    import p16_b as pb
    if arm == "empty_always":
        return next(r["query"] for r in rows if r["m"] == 0)
    if arm == "random":
        return rows[int(rng.integers(len(rows)))]["query"]
    if arm == "geometry":
        import d_blockers as db
        H = {i for i, h in zip(pool_idx, db.heuristic(R, pool_idx)) if h}
        if not H:
            return next(r["query"] for r in rows if r["m"] == 0)
        m = [r for r in rows if r["m"] == len(H) and {i for i, _ in r["seq_idx"]} == H]
        m = sorted(m, key=lambda r: r["query"])
        return m[0]["query"] if m else "ABSTAIN"
    return pb.choose(rows, pol["t0"], pol["t"], pol.get("temp", 1.0), pol.get("bias", 0.0))


def b_exec(name, keys, arms, policies, seeds=(0, 1, 2), part="") -> None:
    """Frozen B evaluation on split `name`: y0 label (deterministic certifier) + physical execution of every arm's choice
    (verify_sequence; EMPTY = original target action; ABSTAIN = no action). Ensemble = mean prior marginal over seeds."""
    import numpy as np
    import p15_lookup as pl
    import p16_b as pb
    import p16_common as pc
    import p16_motion as pm
    plan, idx = rd(DATA / name / "plan.json"), pl.index_v2()[0]
    ms = {a: [pb.load_any(a, s) for s in seeds] for a in arms if a not in ("empty_always", "random", "geometry")}
    f = DATA / name / (("sealed_b_exec" if name.startswith("test") else "b_exec") + (f"_{part}" if part else "") + ".json")
    res = rd(f) if f.exists() else {}
    import learn_data as ld
    dino = ld.dino()
    for k in keys:
        if k in res:
            continue
        c0 = time.process_time()
        task = pc.task_from_root(plan["roots"][k], k)
        cache = {}
        phys = pm.DetPhysics(task, idx, budget=LABEL_BUDGET, cache=cache)
        cat = pc.catalogue(task, phys, idx)
        st0 = phys.goal({o.id: o.cell for o in phys.s0.objects}, np.asarray(phys.s0.robot.q_rad, float))[0]
        rec = pb.records(plan, {k: {"catalogue": cat, "original": {"y": None}, "edits": {}, "sequences": {}}}, "test", dino, with_labels=False)
        r = pb.prepare(rec)[0]
        out = {"family": plan["roots"][k]["family"], "y0_status": st0, "catalogue": {x: cat[x] for x in ("pool", "parking", "movers")}, "arms": {}}
        rows_by = {}
        for a, mm in ms.items():
            sc = [pb.score(m, a, [r])[k] for m in mm]
            rows_by[a] = [dict(x, u=float(np.log(np.mean([s[i]["p"] for s in sc])) - np.log1p(-np.mean([s[i]["p"] for s in sc])))) for i, x in enumerate(sc[0])]
        rng = np.random.default_rng(int(k.split("_")[-1]))
        base = next(iter(rows_by.values())) if rows_by else [{"query": c["query"], "seq": [(r["ids"][i], d) for i, d in c["seq"]], "m": len(c["seq"]),
                                                               "u": 0.0, "p": 0.5} for c in r["cands"]]
        for a in arms:
            rows = rows_by.get(a, [dict(x, seq_idx=[(r["ids"].index(o), dd) for o, dd in x["seq"]]) for x in base])
            q = b_select(a, rows, policies.get(a, {}), r["R"], [r["ids"].index(o) for o in cat["pool"]], rng)
            if q == "ABSTAIN":
                out["arms"][a] = {"choice": q, "status": "ABSTAIN", "moves": None, "travel": None}
                continue
            sig = q.rsplit(":", 1)[1]                                   # keyed query identity -> exact (object, cell) sequence
            seq = [] if sig == "empty" else [(a.split(">")[0], int(a.split(">")[1])) for a in sig.split(".")]
            v = pc.verify_sequence(phys, seq)
            rep = pm.replay(phys, v["plan"]) if v["plan"] else None
            out["arms"][a] = {"choice": q, "m": len(seq), "status": v["status"], "failed_step": v["failed_step"], "replay": rep and rep["ok"],
                              "moves": v["plan"]["moves"] if v["plan"] else None, "travel": v["plan"]["travel"] if v["plan"] else None,
                              "p_choice": next((x["p"] for x in rows if x["query"] == q), None)}
        out["scores"] = {a: [(x["query"], round(x["u"], 4)) for x in rw] for a, rw in rows_by.items()}
        out |= {"used": phys.used, "stats": phys.stats, "cpu_s": time.process_time() - c0, "pcpu": phys.pcpu}
        res[k] = out
        cf.write_json(f, res)


def d_exec(name, keys, guides, n_max=4, per_obj=2, node_cap=500, part="", oracles=None) -> None:
    """Frozen D / baseline search comparison on split `name`: identical scene-only movers, slots, index, budgets and certifier per guide;
    guides run sequentially per root over one exact shared cache (virtual CPU accounting); each found plan independently replayed."""
    import numpy as np
    import p15_lookup as pl
    import p16_common as pc
    import p16_d as pdd
    import p16_motion as pm
    plan, idx = rd(DATA / name / "plan.json"), pl.index_v2()[0]
    f = DATA / name / (("sealed_d_exec" if name.startswith("test") else "d_exec") + (f"_{part}" if part else "") + ".json")
    res = rd(f) if f.exists() else {}
    import learn_data as ld
    dino = ld.dino()
    for k in keys:
        if k in res:
            continue
        task = pc.task_from_root(plan["roots"][k], k)
        cache, out = {}, {"family": plan["roots"][k]["family"], "guides": {}}
        for g in guides:
            c0 = time.process_time()
            phys = pm.DetPhysics(task, idx, cache=cache)
            cells0 = {o.id: o.cell for o in phys.s0.objects}
            movers, omitted = pc.propose_movers(task, cells0)
            order = pdd.guide(g, task, phys, movers, plan["roots"][k], dino, (oracles or {}).get(k))
            r = pc.search(phys, order.order, movers, pc.slot_fn(phys, idx), cells0, phys.s0.robot.q_rad, n_max, per_obj, node_cap)
            rep = pm.replay(phys, r["plan"]) if r["found"] else None
            steps = r["plan"]["steps"] if r["found"] else []
            out["guides"][g] = {"found": r["found"], "replay": rep and rep["ok"], "claim": r["claim"], "min_moves_proven": r["min_moves_proven"],
                                "min_travel_proven": r["min_travel_proven"], "moves": len(steps), "travel": r["plan"]["travel"] if r["found"] else None,
                                "steps": [(s["object"], s["dest"], s["via"], s.get("mode"), s.get("transit")) for s in steps],
                                "repeat_move": len({s["object"] for s in steps}) != len(steps), "flags": r["flags"], "log": r["log"],
                                "used": phys.used, "stats": phys.stats, "d_calls": getattr(order, "d_calls", 0), "d_cpu": getattr(order, "d_cpu", 0.0),
                                "cpu_actual": time.process_time() - c0, "cpu_virtual": phys.vcpu + (time.process_time() - c0 - phys.pcpu),
                                "movers": movers, "omitted": omitted, "order_trace": getattr(order, "trace", [])[:20]}
        res[k] = out
        cf.write_json(f, res)


# ------------------------------------------------------------------ P16-B0/B1 development pilot (DEV roots only)
def b_dev_pilot() -> dict:
    """Score every frozen P13 arm (3 seeds, ensemble mean) on the labelled DEV roots: P(empty) on feasible / infeasible originals,
    within-root AUROC on decided candidates, and the decision of the historical rule. Development evidence only."""
    import numpy as np
    import torch
    import p16_b as pb
    plan, labs = rd(DATA / "dev" / "plan.json"), merged("dev", "labels")
    recs = pb.records(plan, labs, "dev")
    torch.save({"roots": recs}, OUT / "b" / "DEV_records.pt") if (OUT / "b").exists() else None
    rs = pb.prepare(recs)
    out = {}
    for a in ("P13:K-B", "P13:A", "P13:SF", "P13:K-C", "P13:B-F2"):
        ms = [pb.load_any(a, s) for s in range(3)]
        sc = [pb.score(m, a, rs) for m in ms]
        per = {}
        for r in rs:
            rows = sc[0][r["key"]]
            p = [float(np.mean([x[r["key"]][i]["p"] for x in sc])) for i in range(len(rows))]
            y = [r["y"].get(row["query"]) for row in rows]
            pe_, ye = p[0], y[0]
            pos = [pp for pp, yy in zip(p, y) if yy == 1]
            neg = [pp for pp, yy in zip(p, y) if yy == 0]
            per[r["key"]] = {"p_empty": pe_, "y0": ye, "auc": float(np.mean([(a_ > b_) + 0.5 * (a_ == b_) for a_ in pos for b_ in neg])) if pos and neg else None,
                             "top1_y": y[int(np.argmax(p))]}
        f = [v["p_empty"] for v in per.values() if v["y0"] == 1]
        i = [v["p_empty"] for v in per.values() if v["y0"] == 0]
        aucs = [v["auc"] for v in per.values() if v["auc"] is not None]
        out[a] = {"n": len(per), "feasible": len(f), "p_empty_feasible_mean": float(np.mean(f)) if f else None,
                  "p_empty_infeasible_mean": float(np.mean(i)) if i else None, "auroc_mean": float(np.mean(aucs)) if aucs else None,
                  "top1_decided_feasible": float(np.mean([v["top1_y"] == 1 for v in per.values()])), "per_root": per}
    (OUT / "b").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "b" / "B_DEV_PILOT.json", out)
    return out


def temp_lattices(name, keys, part="") -> None:
    """Temporary-target validation lattices (p16_d.temp_lattice) on the scene-only pool of each root; deterministic certifier."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_d as pdd
    import p16_motion as pm
    plan, idx = rd(DATA / name / "plan.json"), pl.index_v2()[0]
    f = DATA / name / (f"temp_lattice_{part}.json" if part else "temp_lattice.json")
    res = rd(f) if f.exists() else {}
    for k in keys:
        if k in res:
            continue
        c0 = time.process_time()
        task = pc.task_from_root(plan["roots"][k], k)
        phys = pm.DetPhysics(task, idx, budget=LABEL_BUDGET)
        cat = pc.catalogue(task, phys, idx)
        if len(cat["pool"]) < 2:
            res[k] = {"skipped": "pool < 2"}
        else:
            res[k] = {"catalogue": {x: cat[x] for x in ("pool", "parking")}, "lattices": pdd.temp_lattice(task, phys, cat), "used": phys.used,
                      "cpu_s": time.process_time() - c0}
        cf.write_json(f, res)


def d_dev_pilot(name="dev") -> dict:
    """Frozen D / D-pairless on DEV: original-target diagnosis vs the scene-only lattice; temporary-target feasibility and blocker
    marginals vs the temporary-target lattices (only decided entries)."""
    import numpy as np
    import p15_retarget as pr
    import p16_b as pb
    import p16_d as pdd
    from scene import scene_from_dict
    plan, labs, tl = rd(DATA / name / "plan.json"), merged(name, "labels"), merged(name, "temp_lattice")
    rs = pb.prepare(pb.records(plan, labs, name))
    out = {}
    cfg = cf.load_config("config.toml")
    for kind in ("D", "Dp") + (("Dp16",) if (pdd.OUT / "ckpt" / "P16-Dp-l2.0-s0.pt").exists() else ()):
        ms = pdd.models(kind)
        dg = pdd.diagnose(ms, rs, labs)
        rows = []
        for k, v in tl.items():
            if "lattices" not in v:
                continue
            s0 = scene_from_dict(plan["roots"][k]["scene"])
            for L in v["lattices"]:
                st_empty = L["statuses"].get("-")
                rec = pr.build_record(cfg, pr.retarget_scene(s0, {}, L["target"], s0.robot.q_rad), L["target"], L["dest"])
                pdp = pr.d_predict(ms, rec, L["target"], L["others"])
                rows.append({"root": k, "target": L["target"], "type": L["type"], "y_feas": 1 if st_empty == "VERIFIED_FEASIBLE" else 0 if st_empty == "CERTIFIED_INFEASIBLE" else None,
                             "p_raw": pdp["p_feasible"], "mask": L["mask"], "marg": pdp["marg"]})
        dec = [r_ for r_ in rows if r_["y_feas"] is not None]
        ms_ = [r_ for r_ in rows if r_["y_feas"] == 0 and r_["type"] == "MINIMAL_SETS"]
        tp = sum(int(m_ >= 0.5) and b for r_ in ms_ for m_, b in zip(r_["marg"], r_["mask"]))
        fp = sum(int(m_ >= 0.5) and not b for r_ in ms_ for m_, b in zip(r_["marg"], r_["mask"]))
        fn = sum((not int(m_ >= 0.5)) and b for r_ in ms_ for m_, b in zip(r_["marg"], r_["mask"]))
        out[kind] = {"original": {k_: v_ for k_, v_ in dg.items() if k_ != "rows"}, "original_rows": dg["rows"],
                     "temp": {"probes": len(rows), "decided_feasibility": len(dec), "feasible": sum(r_["y_feas"] == 1 for r_ in dec),
                              "raw_feasible_recall": float(np.mean([r_["p_raw"] >= 0.5 for r_ in dec if r_["y_feas"] == 1])) if any(r_["y_feas"] == 1 for r_ in dec) else None,
                              "raw_infeasible_recall": float(np.mean([r_["p_raw"] < 0.5 for r_ in dec if r_["y_feas"] == 0])) if any(r_["y_feas"] == 0 for r_ in dec) else None,
                              "marginal_labels": len(ms_), "marginal_f1": 2 * tp / max(1, 2 * tp + fp + fn), "rows": rows}}
    cf.write_json(OUT / "d" / f"D_{name.upper()}_PILOT.json", out)
    return out


def leakage() -> dict:
    """Root-id and geometry-fingerprint disjointness across P16 splits, and against every motion-index source root (trajectory
    provenance: the index predates P16, so no P16 trajectory can appear in it; checked anyway)."""
    import p14_sweep_index as sx
    import p15_lookup as pl
    fp = {}
    for d in sorted(DATA.glob("*/plan.json")):
        for k, r in rd(d)["roots"].items():
            fp.setdefault(pl.json_fp(r), []).append((d.parent.name, k))
    idx = pl.index_v2()[0]
    P = sx._plans()
    idx_fp = {pl.json_fp(P[e["root"]]) for es in idx.by_key.values() for e in es}
    idx_roots = {e["root"] for es in idx.by_key.values() for e in es}
    cross = {f: v for f, v in fp.items() if len({s for s, _ in v}) > 1 or len(v) > 1}
    out = {"roots": sum(len(v) for v in fp.values()), "duplicate_fingerprints": cross,
           "index_overlap": sorted({k for v in fp.values() for _, k in v} & idx_roots) + [f for f in fp if f in idx_fp]}
    cf.write_json(DATA / "LEAKAGE.json", out)
    return out


# ------------------------------------------------------------------ training data (frozen P13 trainval + P16 scene-only labelled splits)
def build_training_data() -> None:
    """runs/p16/b/TRAIN.pt: B records {train: P13 train + P16 train + P16 dev, val: P13 validation + P16 val} and the P16 label dicts
    (for the D items). Excluded duplicate fingerprints are dropped. No calib / test root enters."""
    import os
    import torch
    import p16_b as pb
    os.environ["P10C_OUT"] = "runs/p13/models"
    ex = rd(DATA / "EXCLUDED.json")
    p13 = torch.load("runs/p13/models/dataset_trainval.pt", weights_only=False)["roots"]
    import learn_data as ld
    dino = ld.dino()
    new, labs = {}, {}
    for n in ("train", "dev", "val"):
        plan, lab = rd(DATA / n / "plan.json"), merged(n, "labels")
        lab = {k: v for k, v in lab.items() if k not in ex.get(n, [])}
        new[n], labs[n] = pb.records(plan, lab, "train" if n != "val" else "validation", dino), lab
    d = {"train": [r for r in p13 if r["audit"]["split"] == "train"] + new["train"] + new["dev"],
         "val": [r for r in p13 if r["audit"]["split"] == "validation"] + new["val"], "p16_labels": labs,
         "counts": {"p13_train": sum(r["audit"]["split"] == "train" for r in p13), "p13_val": sum(r["audit"]["split"] == "validation" for r in p13),
                    **{f"p16_{n}": len(v) for n, v in new.items()}}}
    (OUT / "b").mkdir(parents=True, exist_ok=True)
    torch.save(d, OUT / "b" / "TRAIN.pt")
    cf.write_json(OUT / "b" / "TRAIN_COUNTS.json", d["counts"] | {"sha": sha(OUT / "b" / "TRAIN.pt")})


# ------------------------------------------------------------------ B calibration (calibration split only) and analyses
B_LEARNED = ("LB", "LB-F2", "LC", "LB+G", "A", "P13:K-B", "P13:A", "P13:SF", "P13:K-C", "P13:B-F2")
B_ARMS = B_LEARNED + ("geometry", "empty_always", "random")


def b_calibrate(arms=B_LEARNED, seeds=(0, 1, 2)) -> dict:
    """Per arm: ensemble-mean prior marginals on the calibration split; (t0, t) of the no-op-first rule maximizing teacher-scored correct
    decisions there (p16_b.calibrate). Nothing from val / test enters."""
    import numpy as np
    import p16_b as pb
    plan, labs = rd(DATA / "calib" / "plan.json"), merged("calib", "labels")
    rs = pb.prepare(pb.records(plan, labs, "calib"))
    y = {r["key"]: dict(r["y"]) for r in rs}
    out = {}
    for a in arms:
        ms = [pb.load_any(a, s) for s in seeds]
        sc = [pb.score(m, a, rs) for m in ms]
        ens = {k: [dict(x, p=float(np.mean([s[k][i]["p"] for s in sc]))) for i, x in enumerate(sc[0][k])] for k in sc[0]}
        for k in ens:
            for x in ens[k]:
                x["u"] = float(np.log(max(x["p"], 1e-9)) - np.log1p(-min(x["p"], 1 - 1e-9)))
        out[a] = pb.calibrate({k: v for k, v in ens.items() if y[k].get(next(x["query"] for x in v if x["m"] == 0)) is not None}, y)
    cf.write_json(OUT / "b" / "POLICIES.json", out)
    return out


def paired(a, b, seed=0) -> dict:
    """Exact two-sided sign test on paired binary root outcomes + root-bootstrap CI of the mean difference (never an equivalence test)."""
    import math
    import numpy as np
    w, l = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    n = w + l
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(w, l) + 1)) / 2 ** n)
    A, B = np.asarray(a, float), np.asarray(b, float)
    ci = None
    if len(A):
        ix = np.random.default_rng(seed).integers(0, len(A), (10000, len(A)))
        ci = [float(x) for x in np.percentile((A[ix] - B[ix]).mean(1), [2.5, 97.5])]
    return {"a_only": w, "b_only": l, "discordant": n, "p": p, "diff": float((A - B).mean()) if len(A) else None, "ci95": ci,
            "status": "UNDERPOWERED" if n < 6 else "TESTED"}


def holm(ps: dict) -> dict:
    ks = sorted(ps, key=ps.get)
    out, run = {}, 0.0
    for i, k in enumerate(ks):
        run = max(run, min(1.0, ps[k] * (len(ks) - i)))
        out[k] = run
    return out


def b_analysis(exec_rows, primary=("LB", "P13:K-B")) -> dict:
    """Root-level outcomes per arm on DECIDED originals (y0 from the deterministic certifier): feasible originals -> correct no-op (EMPTY
    chosen and VERIFIED), unnecessary relocation, abstain; infeasible -> strict repair success (repair VERIFIED + replay CLEAR), wrong
    no-op, abstain. Strict TOP1 success = correct no-op OR strict repair success."""
    import numpy as np
    rows = {k: v for k, v in exec_rows.items() if v["y0_status"] in ("VERIFIED", "CERTIFIED_INFEASIBLE", "STATIC_INVALID")}
    feas = [k for k, v in rows.items() if v["y0_status"] == "VERIFIED"]
    inf = [k for k in rows if k not in feas]
    arms = sorted({a for v in rows.values() for a in v["arms"]})
    ok_noop = lambda v, a: v["arms"][a]["choice"].endswith(":empty") and v["arms"][a]["status"] == "VERIFIED"  # noqa: E731
    ok_rep = lambda v, a: v["arms"][a]["choice"] != "ABSTAIN" and not v["arms"][a]["choice"].endswith(":empty") and v["arms"][a]["status"] == "VERIFIED" and v["arms"][a].get("replay")  # noqa: E731
    per = {}
    for a in arms:
        f = [rows[k]["arms"][a]["choice"] for k in feas]
        i = [rows[k]["arms"][a]["choice"] for k in inf]
        per[a] = {"feasible_n": len(feas), "noop_correct": sum(ok_noop(rows[k], a) for k in feas),
                  "unnecessary_relocation": sum(c != "ABSTAIN" and not c.endswith(":empty") for c in f), "abstain_feasible": f.count("ABSTAIN"),
                  "infeasible_n": len(inf), "repair_success": sum(ok_rep(rows[k], a) for k in inf), "wrong_noop": sum(c.endswith(":empty") for c in i),
                  "abstain_infeasible": i.count("ABSTAIN"), "attempted_infeasible": sum(c != "ABSTAIN" and not c.endswith(":empty") for c in i),
                  "strict_top1": float(np.mean([ok_noop(rows[k], a) for k in feas] + [ok_rep(rows[k], a) for k in inf])) if rows else None,
                  "repair_moves_mean": float(np.mean([rows[k]["arms"][a]["moves"] for k in inf if ok_rep(rows[k], a)])) if any(ok_rep(rows[k], a) for k in inf) else None,
                  "repair_travel_mean": float(np.mean([rows[k]["arms"][a]["travel"] for k in inf if ok_rep(rows[k], a)])) if any(ok_rep(rows[k], a) for k in inf) else None}
    a, b = primary
    prim = paired([ok_noop(rows[k], a) for k in feas], [ok_noop(rows[k], b) for k in feas]) if a in arms and b in arms else None
    sec = {}
    for o in arms:
        if o != a:
            sec[f"{a}_vs_{o}|strict_top1"] = paired([ok_noop(rows[k], a) if k in feas else ok_rep(rows[k], a) for k in rows],
                                                     [ok_noop(rows[k], o) if k in feas else ok_rep(rows[k], o) for k in rows])
    sec[f"{a}_vs_{b}|repair_on_infeasible"] = paired([ok_rep(rows[k], a) for k in inf], [ok_rep(rows[k], b) for k in inf])
    h = holm({k: v["p"] for k, v in sec.items()})
    for k in sec:
        sec[k]["p_holm"] = h[k]
    fam = {}
    for k, v in rows.items():
        fam.setdefault(v["family"], []).append(k)
    by_family = {f: {x: float(np.mean([ok_noop(rows[k], x) if k in feas else ok_rep(rows[k], x) for k in ks])) for x in arms} | {"n": len(ks)} for f, ks in fam.items()}
    return {"decided_roots": len(rows), "undecided_originals": len(exec_rows) - len(rows), "feasible": len(feas), "infeasible": len(inf),
            "per_arm": per, "primary_noop": prim, "secondary": sec, "by_family": by_family}


def d_analysis(rows, primary=("Dp16", "geometry"), unions=None) -> dict:
    """Per guide on the common roots: strict success (found + independent replay CLEAR), claims / proof coverage, moves / travel on co-solved
    roots, budget counts, virtual CPU (shared-cache exact accounting) and D inference; paired sign tests vs geometry (Holm across the
    secondary family); extra moves only against proven optima; guide-disagreement analysis (first expanded move differs from geometry's?)."""
    import collections
    import numpy as np
    guides = sorted({g for v in rows.values() for g in v["guides"]})
    ks = [k for k, v in rows.items() if all(g in v["guides"] for g in guides)]
    ok = lambda k, g: bool(rows[k]["guides"][g]["found"] and rows[k]["guides"][g]["replay"])  # noqa: E731
    per = {}
    for g in guides:
        R = [rows[k]["guides"][g] for k in ks]
        per[g] = {"n": len(ks), "success": sum(ok(k, g) for k in ks), "claims": dict(collections.Counter(r["claim"] for r in R)),
                  "min_moves_proven": sum(r["min_moves_proven"] for r in R), "min_travel_proven": sum(r["min_travel_proven"] for r in R),
                  "repeat_move_plans": sum(r["repeat_move"] for r in R if r["found"]),
                  "calls_mean": {c: float(np.mean([r["used"][c] for r in R])) for c in R[0]["used"]} if R else {},
                  "nodes_mean": float(np.mean([r["log"]["nodes"] for r in R])) if R else None,
                  "node_cap_hits": sum(r["log"]["node_cap_hit"] for r in R),
                  "cpu_virtual_mean": float(np.mean([r["cpu_virtual"] for r in R])) if R else None, "d_cpu_mean": float(np.mean([r["d_cpu"] for r in R])) if R else None,
                  "d_calls_mean": float(np.mean([r["d_calls"] for r in R])) if R else None,
                  "unknown": int(sum(c for r in R for kind in ("edges", "goals") for s, c in r["log"][kind].items() if s.startswith("UNKNOWN"))),
                  "anytime_nodes": {str(t): sum(ok(k, g) and (rows[k]["guides"][g]["log"].get("anytime") or [{}])[0].get("nodes", 1e9) <= t for k in ks) for t in (25, 50, 100, 200, 500)}}
        if unions:
            out_u = collections.Counter()
            for k in ks:
                if ok(k, g) and unions.get(k) is not None:
                    out_u[str(sum(s[0] not in unions[k] for s in rows[k]["guides"][g]["steps"]))] += 1
            per[g]["moves_outside_original_blocker_union"] = dict(out_u)
    a, b = primary
    prim = paired([ok(k, a) for k in ks], [ok(k, b) for k in ks]) if a in guides and b in guides else None
    sec = {f"{g}_vs_{b}": paired([ok(k, g) for k in ks], [ok(k, b) for k in ks]) for g in guides if g not in (a, b) and g != "oracle"}
    h = holm({k: v["p"] for k, v in sec.items()}) if sec else {}
    for k in sec:
        sec[k]["p_holm"] = h[k]
    co = {}
    for g in guides:
        both = [k for k in ks if ok(k, g) and ok(k, b)]
        co[g] = {"n": len(both), "moves": [float(np.mean([rows[k]["guides"][g]["moves"] for k in both])) if both else None,
                                          float(np.mean([rows[k]["guides"][b]["moves"] for k in both])) if both else None],
                 "travel": [float(np.mean([rows[k]["guides"][g]["travel"] for k in both])) if both else None,
                            float(np.mean([rows[k]["guides"][b]["travel"] for k in both])) if both else None]}
    extra = {}
    for g in guides:
        e = []
        for k in ks:
            pr = [rows[k]["guides"][x]["moves"] for x in guides if rows[k]["guides"][x]["min_moves_proven"] and ok(k, x)]
            if pr and ok(k, g):
                e.append(rows[k]["guides"][g]["moves"] - min(pr))
        extra[g] = {"n_with_proven_optimum": len(e), "extra_moves": dict(collections.Counter(e))}
    dis = []
    for k in ks:
        fa = (rows[k]["guides"][a]["order_trace"] or [{}])[0].get("first") if a in guides else None
        fb = (rows[k]["guides"][b]["order_trace"] or [{}])[0].get("first")
        dis.append({"root": k, "family": rows[k]["family"], "first_differs": fa != fb, f"{a}_first": fa, f"{b}_first": fb,
                    "outcome": [ok(k, a), ok(k, b)] if a in guides else None,
                    "motion_checks": [rows[k]["guides"][a]["used"]["motion_checks"], rows[k]["guides"][b]["used"]["motion_checks"]] if a in guides else None})
    return {"common_roots": len(ks), "per_guide": per, "primary": prim, "secondary": sec, "co_solved_vs_" + b: co, "extra_moves_vs_proven_optimum": extra,
            "disagreement": {"first_move_differs": sum(x["first_differs"] for x in dis), "rows": dis}}


# ------------------------------------------------------------------ pre-registration (written and pushed BEFORE any test label)
D_GUIDES = ("Dp16", "Dp16-norec", "Dp", "D", "geometry", "geometry-norec", "uniform")
D_TEST_FAMS = ("DENSE", "PN_SIMP", "PN_GIMP", "PC_SG", "PR_SIDES", "ORD_CHAIN", "OMIT_OPT", "TRI_OR", "TRI_CTRL")


def write_prereg() -> dict:
    import glob
    test = rd(DATA / "test" / "plan.json")
    d_keys = sorted(k for k, r in test["roots"].items() if r["family"] in D_TEST_FAMS)
    b_keys = sorted(k for k, r in test["roots"].items() if r["family"] in B_TEST_FAMS)
    pre = {"version": "P16_PREREG_V1", "base": "20f775483077a568efb225f22aa93ca619160c53",
           "test_roster": {"plan": sha(DATA / "test" / "plan.json"), "roots": len(test["roots"]), "d_roots": d_keys, "b_roots": b_keys,
                           "order": "p16_eval.test_order (family-interleaved), 12 workers round-robin; one process per root list"},
           "execution": "p16_eval.test_exec: per root, D guides (D roots) then every B arm (B roots) on ONE exact shared cache, physics budget "
                        "{400, 6, 16, 8} (P16_FAST); sealed output; lattice-only sealed labels on the D roots (original + parked subsets)",
           "scope_reason": "measured cost pilots (D: ~255 virtual / root on mechanism roots; B: ~200 CPU-s / root) vs the 38k execution cap: B on "
                           "NAT 120 + 24 mechanism roots (DENSE excluded: B's catalogue is <= 2 moves), D on DENSE 24 + 24 mechanism roots",
           "splits": rd(DATA / "SPLIT_COUNTS.json"), "excluded": rd(DATA / "EXCLUDED.json"), "leakage": sha(DATA / "LEAKAGE.json"),
           "code": {p: sha(p) for p in ("p16_common.py", "p16_motion.py", "p16_b.py", "p16_d.py", "p16_eval.py")},
           "planner_profile": "P16_FAST (max_nodes 1500, max_plan_failures 2, transit 500 nodes); no CPU / wall adjudication",
           "checkpoints": {p: sha(p) for p in sorted(glob.glob("runs/p16/b/ckpt/*.pt") + glob.glob("runs/p16/d/ckpt/*.pt"))},
           "training_data": sha(OUT / "b" / "TRAIN.pt"), "policies": rd(OUT / "b" / "POLICIES.json"),
           "B": {"arms": list(B_ARMS), "ensemble": "mean prior marginal over seeds 0-2",
                 "rule": "no-op-first: EMPTY if p(empty) >= t0, else admitted (p >= t) repair with fewest moves then highest p, else ABSTAIN; "
                         "(t0, t) per arm from the calibration split only",
                 "execution": "chosen candidate certified step by step (verify_sequence, cheapest certified mode) + independent replay",
                 "primary": "H-B1: on B test roots whose ORIGINAL is decided feasible by the certifier, correct no-op rate LB vs P13:K-B (exact two-sided "
                            "sign test, alpha 0.05, root bootstrap CI); minimal meaningful effect +15 pp; UNDERPOWERED if < 6 discordant or < 60 "
                            "decided feasible originals",
                 "guard": "non-inferiority check (reported, not tested): LB strict repair success on decided infeasible originals not lower than "
                          "P13:K-B by more than 10 pp (CI)",
                 "secondary": ["LB vs every other arm on strict TOP1 (Holm)", "H-B2 generic-only coupling: LB+G vs LB strict TOP1 (in the Holm family)",
                               "unnecessary relocation, wrong no-op, abstain, attempted coverage, success | attempt, by family",
                               "candidate ranking on the labelled test subset (within-root AUROC, UNKNOWN censored)"]},
           "D": {"guides": list(D_GUIDES) + ["oracle (after unsealing; original scene-only lattice union)"],
                 "search": {"n_max": 4, "per_obj": 2, "node_cap": 500, "budget": {"motion_checks": 400, "online_moves": 6, "goal_plans": 16,
                                                                                  "transit_plans": 8}, "movers": "scene-only, cap 6", "slots": "scene-only, <= 6"},
                 "primary": "H-D1: strict success (plan found + independent replay CLEAR) Dp16 vs geometry on the 48 D test roots, exact two-sided "
                            "sign test, alpha 0.05; minimal meaningful effect +10 pp or >= 20 % fewer motion checks at non-inferior success; "
                            "UNDERPOWERED if < 6 discordant",
                 "secondary": ["each other guide vs geometry (Holm)", "virtual-CPU / node anytime success", "proof coverage, unknown counts",
                               "co-solved moves / travel", "extra moves vs proven optima only", "first-move disagreement analysis",
                               "diagnosis on the test lattice: feasibility AUROC / Brier / recalls, union-blocker F1 / exact, union vs necessary"]},
           "order": "freeze + push -> test labels + B / D execution (sealed) -> label / outcome manifest hash commit -> unseal -> oracle guide -> analysis",
           "no_adaptive_stopping": True, "budget_stop": "a resource stop is reported as incomplete coverage on the fixed root order"}
    pre["hash"] = cf.stable_hash({k: v for k, v in pre.items() if k != "hash"})
    cf.write_json(OUT / "P16_PREREG.json", pre)
    return pre


def test_order(keys) -> list:
    """Fixed, family-interleaved processing order (seed % 1000, seed // 1000) so that a budget stop leaves balanced coverage."""
    return sorted(keys, key=lambda k: (int(k.split("_")[-1]) % 1000, int(k.split("_")[-1]) // 1000))


def label_many(pairs, part, full=("calib",)) -> None:
    """One worker over mixed (split, key) pairs (skips keys already labelled in any part file). Splits not in `full` get lattice-only
    labels (original + parked subsets; no sequence labels) after the measured cost (~390 CPU-s / full root) threatened the data cap."""
    for n, k in pairs:
        label_split(n, [k], part=part, sequences=n in full)


# ------------------------------------------------------------------ sealed prospective run: one process per root list, D guides then B arms on ONE exact cache
B_TEST_FAMS = ("NAT", "PN_SIMP", "PN_GIMP", "PC_SG", "PR_SIDES", "ORD_CHAIN", "OMIT_OPT", "TRI_OR", "TRI_CTRL")


def _d_root(k, task, plan, idx, guides, dino, cache, oracle=None, n_max=4, per_obj=2, node_cap=500):
    import p16_common as pc
    import p16_d as pdd
    import p16_motion as pm
    out = {}
    for g in guides:
        c0 = time.process_time()
        phys = pm.DetPhysics(task, idx, cache=cache)
        cells0 = {o.id: o.cell for o in phys.s0.objects}
        movers, omitted = pc.propose_movers(task, cells0)
        order = pdd.guide(g, task, phys, movers, plan["roots"][k], dino, oracle)
        r = pc.search(phys, order.order, movers, pc.slot_fn(phys, idx), cells0, phys.s0.robot.q_rad, n_max, per_obj, node_cap)
        rep = pm.replay(phys, r["plan"]) if r["found"] else None
        steps = r["plan"]["steps"] if r["found"] else []
        out[g] = {"found": r["found"], "replay": rep and rep["ok"], "traj": rep and rep.get("traj"), "claim": r["claim"],
                  "min_moves_proven": r["min_moves_proven"], "min_travel_proven": r["min_travel_proven"], "moves": len(steps),
                  "travel": r["plan"]["travel"] if r["found"] else None, "steps": [(s["object"], s["dest"], s["via"], s.get("mode"), s.get("transit")) for s in steps],
                  "repeat_move": len({s["object"] for s in steps}) != len(steps), "flags": r["flags"], "log": r["log"], "used": phys.used, "stats": phys.stats,
                  "d_calls": order.d_calls, "d_cpu": order.d_cpu, "cpu_actual": time.process_time() - c0,
                  "cpu_virtual": phys.vcpu + (time.process_time() - c0 - phys.pcpu), "movers": movers, "omitted": omitted, "order_trace": order.trace[:20]}
    return out


def _b_root(k, task, plan, idx, ms, arms, policies, dino, cache):
    import numpy as np
    import p16_b as pb
    import p16_common as pc
    import p16_motion as pm
    c0 = time.process_time()
    phys = pm.DetPhysics(task, idx, cache=cache)
    cat = pc.catalogue(task, phys, idx)
    st0 = phys.goal({o.id: o.cell for o in phys.s0.objects}, np.asarray(phys.s0.robot.q_rad, float))[0]
    r = pb.prepare(pb.records(plan, {k: {"catalogue": cat, "original": {"y": None}, "edits": {}, "sequences": {}}}, "test", dino, with_labels=False))[0]
    rows_by = {}
    for a, mm in ms.items():
        sc = [pb.score(m, a, [r])[k] for m in mm]
        rows_by[a] = []
        for i, x in enumerate(sc[0]):
            p_ = float(np.mean([s[i]["p"] for s in sc]))
            rows_by[a].append(dict(x, p=p_, u=float(np.log(max(p_, 1e-9)) - np.log1p(-min(p_, 1 - 1e-9))), seq_idx=[(r["ids"].index(o), d) for o, d in x["seq"]]))
    base = next(iter(rows_by.values()))
    rng = np.random.default_rng(int(k.split("_")[-1]))
    out = {"y0_status": st0, "catalogue": {x: cat[x] for x in ("pool", "parking", "movers")}, "arms": {}}
    for a in arms:
        rows = rows_by.get(a, base)
        q = b_select(a, rows, policies.get(a, {}), r["R"], [r["ids"].index(o) for o in cat["pool"]], rng)
        if q == "ABSTAIN":
            out["arms"][a] = {"choice": q, "status": "ABSTAIN", "moves": None, "travel": None}
            continue
        sig = q.rsplit(":", 1)[1]
        seq = [] if sig == "empty" else [(x.split(">")[0], int(x.split(">")[1])) for x in sig.split(".")]
        v = pc.verify_sequence(phys, seq)
        rep = pm.replay(phys, v["plan"]) if v["plan"] else None
        out["arms"][a] = {"choice": q, "m": len(seq), "status": v["status"], "failed_step": v["failed_step"], "replay": rep and rep["ok"],
                          "moves": v["plan"]["moves"] if v["plan"] else None, "travel": v["plan"]["travel"] if v["plan"] else None,
                          "p_choice": next((x["p"] for x in rows if x["query"] == q), None)}
    out["scores"] = {a: [(x["query"], round(x["u"], 4)) for x in rw] for a, rw in rows_by.items()}
    out |= {"used": phys.used, "stats": phys.stats, "cpu_s": time.process_time() - c0}
    return out


def test_exec(name, keys, part, seeds=(0, 1, 2), oracles=None, guides=None, do_b=True) -> None:
    """Sealed frozen evaluation: per root, D guides (D roots only) then every B arm (B roots only) on one exact shared cache."""
    import p15_lookup as pl
    import p16_b as pb
    import p16_common as pc
    import learn_data as ld
    plan, idx = rd(DATA / name / "plan.json"), pl.index_v2()[0]
    pol = rd(OUT / "b" / "POLICIES.json")
    ms = {a: [pb.load_any(a, s) for s in seeds] for a in B_LEARNED}
    dino = ld.dino()
    f = DATA / name / (("sealed_exec" if name.startswith("test") and do_b else "exec") + f"_{part}.json")
    res = rd(f) if f.exists() else {}
    for k in keys:
        if k in res:
            continue
        fam, c0 = plan["roots"][k]["family"], time.process_time()
        task, cache = pc.task_from_root(plan["roots"][k], k), {}
        out = {"family": fam}
        if fam in D_TEST_FAMS:
            out["d"] = _d_root(k, task, plan, idx, guides or D_GUIDES, dino, cache, (oracles or {}).get(k))
        if fam in B_TEST_FAMS and do_b:
            out["b"] = _b_root(k, task, plan, idx, ms, B_ARMS, pol, dino, cache)
        out["cpu_root"] = time.process_time() - c0
        res[k] = out
        cf.write_json(f, res)


# ------------------------------------------------------------------ custody: manifest -> push -> unseal -> oracle -> analysis
def test_manifest() -> None:
    files = sorted(str(f) for f in (DATA / "test").glob("sealed_*.json"))
    cf.write_json(DATA / "test" / "SEALED_MANIFEST.json", {"files": {f: sha(f) for f in files}, "n": len(files)})


def _on_origin(path) -> bool:
    subprocess.run(["git", "fetch", "-q", "origin"], check=True)
    r = subprocess.run(["git", "show", f"origin/main:{path}"], capture_output=True)
    return r.returncode == 0 and r.stdout == Path(path).read_bytes()


def test_unseal() -> None:
    m = DATA / "test" / "SEALED_MANIFEST.json"
    if not _on_origin(str(m)):
        raise SystemExit("sealed manifest not on origin/main")
    if any(sha(f) != h for f, h in rd(m)["files"].items()):
        raise SystemExit("sealed file changed after the manifest")
    (DATA / "test" / "UNSEALED").write_text(sha(m) + "\n")


def test_oracles() -> dict:
    """Original scene-only lattice union per D test root (unsealed labels); None where the lattice is not MINIMAL_SETS / feasible."""
    import p16_d as pdd
    if not (DATA / "test" / "UNSEALED").exists():
        raise SystemExit("sealed")
    out = {}
    for k, v in merged("test", "sealed_labels").items():
        d = pdd.lattice(v, v["catalogue"]["pool"])
        out[k] = ({o: m for o, m in zip(v["catalogue"]["pool"], d["mask"])} if d["mask"] is not None else None, d["type"])
    return out


def run_oracle(keys, part) -> None:
    orc = {k: v[0] for k, v in test_oracles().items() if v[0] is not None}
    test_exec("test", [k for k in keys if k in orc], part, oracles=orc, guides=("oracle",), do_b=False)


def analyze() -> dict:
    """B and D prospective analyses from the unsealed outputs (oracle merged into D rows)."""
    if not (DATA / "test" / "UNSEALED").exists():
        raise SystemExit("sealed")
    rows = merged("test", "sealed_exec")
    orc_rows = {}
    for f in sorted((DATA / "test").glob("exec_oracle_*.json")):
        orc_rows |= rd(f)
    b = {k: {"family": v["family"], **v["b"]} for k, v in rows.items() if "b" in v}
    d = {k: {"family": v["family"], "guides": {**v["d"], **(orc_rows.get(k, {}).get("d", {}))}} for k, v in rows.items() if "d" in v}
    orc = test_oracles()
    unions = {k: {o for o, m in x[0].items() if m} for k, x in orc.items() if x[0] is not None}
    ab, ad = b_analysis(b), d_analysis({k: v for k, v in d.items()}, unions=unions)
    labs = merged("test", "sealed_labels")
    diag = {}
    import p16_b as pb
    import p16_d as pdd
    plan = rd(DATA / "test" / "plan.json")
    rs = pb.prepare(pb.records(plan, labs, "test"))
    for kind in ("D", "Dp", "Dp16"):
        x = pdd.diagnose(pdd.models(kind), rs, labs)
        diag[kind] = {k_: v_ for k_, v_ in x.items() if k_ != "rows"}
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "eval" / "B_TEST.json", ab)
    cf.write_json(OUT / "eval" / "D_TEST.json", ad | {"diagnosis": diag, "oracle_types": {k: v[1] for k, v in orc.items()}})
    return {"b": ab, "d": ad, "diag": diag}


def test_queue(part, seeds=(0, 1, 2)) -> None:
    """Amendment 1: workers claim roots atomically from runs/p16/data/test/queue.json (NAT B-only first, then D roots in frozen order);
    models loaded once per worker; identical per-root computation to test_exec."""
    import os
    import p15_lookup as pl
    import p16_b as pb
    import p16_common as pc
    import learn_data as ld
    plan, idx, q = rd(DATA / "test" / "plan.json"), pl.index_v2()[0], rd(DATA / "test" / "queue.json")
    pol, dino = rd(OUT / "b" / "POLICIES.json"), ld.dino()
    ms = {a: [pb.load_any(a, s) for s in seeds] for a in B_LEARNED}
    claims = DATA / "test" / "claims"
    claims.mkdir(exist_ok=True)
    f = DATA / "test" / f"sealed_exec_{part}.json"
    res = rd(f) if f.exists() else {}
    for k in q["nat"] + q["d"]:
        try:
            os.close(os.open(claims / f"{k}.claim", os.O_CREAT | os.O_EXCL))
        except FileExistsError:
            continue
        fam, c0 = plan["roots"][k]["family"], time.process_time()
        task, cache = pc.task_from_root(plan["roots"][k], k), {}
        out = {"family": fam}
        if fam in D_TEST_FAMS:
            out["d"] = _d_root(k, task, plan, idx, D_GUIDES, dino, cache)
        if fam in B_TEST_FAMS:
            out["b"] = _b_root(k, task, plan, idx, ms, B_ARMS, pol, dino, cache)
        out["cpu_root"] = time.process_time() - c0
        res[k] = out
        cf.write_json(f, res)


SOURCES = ("p16_common.py", "p16_motion.py", "p16_b.py", "p16_d.py", "p16_eval.py", "p16_viz.py", "tests/test_p16.py")


def manifests(status="PARTIAL") -> None:
    import glob
    cf.write_json(OUT / "P16_CODE_MANIFEST.json", {"sources": {p: sha(p) for p in SOURCES},
                                                   "checkpoints": {p: sha(p) for p in sorted(glob.glob("runs/p16/b/ckpt/*.pt") + glob.glob("runs/p16/d/ckpt/*.pt"))},
                                                   "planner_profile": "P16_FAST", "search": "p16_common.search (expanded finite, repeat <= 2)"})
    reg = rd(OUT / "regression_final" / "regression_result.json")
    out = {"stage": "P16", "base": "20f775483077a568efb225f22aa93ca619160c53", "status": status,
           "gates": {g.stem: {k: v for k, v in rd(g).items() if k != "spend"} for g in sorted(OUT.glob("P16_*_GATE.json"))},
           "prereg": sha(OUT / "P16_PREREG.json"), "amendment_1": sha(OUT / "P16_PREREG_AMENDMENT_1.json"),
           "sealed_manifest": sha(DATA / "test" / "SEALED_MANIFEST.json"), "training_data": sha(OUT / "b" / "TRAIN.pt"),
           "regression": {"tests": reg.get("tests"), "replays": reg.get("replays")}, "spend_cpu_s": {c: round(spent(c), 1) for c in CAPS}, "caps": CAPS,
           "results": {"B": sha(OUT / "eval" / "B_TEST.json"), "D": sha(OUT / "eval" / "D_TEST.json")},
           "figures": {p: sha(p) for p in sorted(glob.glob("visualizations/p16/*/*")) if not p.endswith(".sha256")},
           "historical_mismatches": verify_historical(), "head_at_manifest": _git("rev-parse", "HEAD").strip()}
    out["hash"] = cf.stable_hash({k: v for k, v in out.items() if k != "hash"})
    cf.write_json(OUT / "P16_FINAL_MANIFEST.json", out)
