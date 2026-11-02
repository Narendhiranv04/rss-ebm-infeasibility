"""P9-L3a / L3b driver (0 teacher, IK or planner calls; every compute job guarded and ledgered in runs/p9/l3/guards).

L3a-0 (`audit`): source / checkpoint integrity, canonical query identity, catalogue multiplicity of every sequence query by
factor type, per-candidate vs root-prior vs root-posterior predictions, and catalogue-sensitivity cases of the P9 PoE root model.
"""
from __future__ import annotations

import collections
import copy
import itertools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import abc7_train as t7
import abc8
import config as cf
import joint_likelihood as jl
import teacher

OUT = Path("runs/p9/l3")
L2 = Path("runs/p9/l2")
DEV, TEST = Path("runs/p8/c_training/dataset.pt"), Path("runs/p8/c_training/dataset_sealed.pt")
SOURCES = ["abc7_models.py", "abc7_train.py", "abc8.py", "abc_models.py", "joint_likelihood.py", "p9_run.py", "p9_viz.py", "p8_eval.py",
           "p8_select.py", "p7_eval.py", "tests/test_p9_joint.py", "teacher_opneg.toml", "config.toml"]
CAPS = {"aggregate_cpu_s": 8000.0, "aggregate_wall_s": 18000.0, "job_cpu_s": 1800.0, "job_wall_s": 3600.0}


def sha(p) -> str:
    import hashlib
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rd(p):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------ guarded jobs and the CPU ledger
def spent() -> tuple:
    gs = [_rd(p) for p in (OUT / "guards").glob("*.json")]
    return sum(g["cpu_s"] for g in gs), sum(g["wall_s"] for g in gs)


def job(name, call, cpu) -> dict:
    """Run `p9_l3.<call>` in a process-tree guarded worker; refuse if the aggregate caps would be exceeded."""
    (OUT / "guards").mkdir(parents=True, exist_ok=True)
    c, w = spent()
    cpu = min(cpu, CAPS["job_cpu_s"], CAPS["aggregate_cpu_s"] - c)
    if cpu < 10 or w > CAPS["aggregate_wall_s"] - 60:
        raise SystemExit(f"budget exhausted: {c:.0f} CPU-s / {w:.0f} wall-s spent")
    g = teacher.guard([sys.executable, "-c", f"import torch; torch.set_num_threads(1); import p9_l3; p9_l3.{call}"], cpu,
                      CAPS["job_wall_s"], OUT / "heartbeat.txt", 15.0, 0.25, Path("."))
    n = 1
    while (OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json").exists():
        n += 1
    rec = {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"cpu_cap_s": cpu, "call": call}
    cf.write_json(OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json", rec)
    (OUT / "heartbeat.txt").unlink(missing_ok=True)
    print(json.dumps({k: rec[k] for k in ("stop", "cpu_s", "wall_s")} | {"aggregate_cpu_s": round(spent()[0], 1)}))
    return rec


# ------------------------------------------------------------------ data and canonical identity
def data(certified_only=False):
    raw = {k: torch.load(p, weights_only=False) for k, p in (("dev", DEV), ("test", TEST))}
    dev, test = t7.prepare(raw["dev"], certified_only), t7.prepare(raw["test"])
    for rs, recs in ((dev, raw["dev"]["roots"]), (test, raw["test"]["roots"])):
        for r, rec in zip(rs, recs):
            assert r["key"] == rec["audit"]["root"]
            r["ids"] = rec["audit"]["ids"]
    return [r for r in dev if r["split"] == "train"], [r for r in dev if r["split"] == "validation"], test


def cell(d) -> int:
    return int(round((float(d[1]) - 0.04) / 0.08)) * 8 + int(round((float(d[0]) - 0.04) / 0.08))


def canon(r, seq) -> str:
    """Canonical physical identity of a sequence query: root (scene + robot state), exact ordered (object id, destination cell)."""
    return f"seq:{r['key']}:" + (".".join(f"{r['ids'][i]}>{cell(d)}" for i, d in seq) or "empty")


def identity_audit(roots) -> dict:
    c, bad = collections.Counter(), []
    for r in roots:
        seen = {}
        for cand in r["cands"]:
            for s, q in zip(cand["slots"], cand["z_q"]):
                if s[:2] in ("z0", "zR", "zO"):
                    seq = [] if s == "z0" else cand["seq"] if s == "zR" else [a for k, a in enumerate(cand["seq"]) if k != int(s[3:-1])]
                    ok = canon(r, seq) == q
                    c["slots_checked"] += 1
                    c["slots_mismatch"] += not ok
                    if not ok and len(bad) < 5:
                        bad.append({"root": r["key"], "slot": s, "stored": q, "canonical": canon(r, seq)})
                    if q in seen and seen[q] != canon(r, seq):
                        c["query_two_meanings"] += 1
                    seen[q] = canon(r, seq)
    return {"counts": dict(c), "examples": bad}


def multiplicity(roots) -> dict:
    """Per distinct sequence query: PoE unary multiplicity (blocks containing it), roles, and pair-factor instances touching it."""
    blk, role, sr, om, oo = (collections.Counter() for _ in range(5))
    for r in roots:
        ks = sum(s[:2] in ("z0", "zS") for s in r["cands"][0]["slots"])
        for cand in r["cands"]:
            if not cand["seq"]:
                continue
            qs = [(s, q) for s, q in zip(cand["slots"], cand["z_q"]) if s[:2] in ("zR", "zO")]
            for s, q in qs:
                blk[(r["key"], q)] += 1
                role[(r["key"], q, s[:2])] += 1
                sr[(r["key"], q)] += ks
                om[(r["key"], q)] += (len(qs) - 1) if s == "zR" else 1
                oo[(r["key"], q)] += max(len(qs) - 2, 0) if s != "zR" else 0
    roles = collections.Counter()
    for (k, q) in blk:
        roles["+".join(sorted(x for x in ("zR", "zO") if role[(k, q, x)]))] += 1
    hist = lambda cn: dict(sorted(collections.Counter(cn.values()).items()))  # noqa: E731
    return {"distinct_sequence_queries": len(blk), "role_mix": dict(roles), "poe_unary_multiplicity": hist(blk),
            "queries_with_unary_counted_more_than_once": sum(v > 1 for v in blk.values()),
            "scene_repair_pair_instances_per_query": hist(sr), "zR_zO_pair_instances_per_query": hist(om),
            "zO_zO_pair_instances_per_query": hist(oo)}


# ------------------------------------------------------------------ prediction modes and PoE catalogue sensitivity
def load(arm, path):
    m = abc8.build(arm)
    m.load_state_dict(torch.load(path, weights_only=True))
    m.eval()
    return m


@torch.no_grad()
def poe_posterior(S, blocks, lab) -> dict:
    """Offline diagnostic only: P(q = 1 | all observed labels) under the PoE root model."""
    base = jl.log_evidence(S, blocks, lab)[0]
    qs = list(S["queries"]) + sorted({q for b in blocks for q in b["queries"]})
    return {q: float(lab[q]) if lab.get(q) is not None else float(torch.exp(jl.log_evidence(S, blocks, lab | {q: 1})[0] - base)) for q in qs}


@torch.no_grad()
def modes(m, r) -> dict:
    S, blocks, lab, outs = jl.tables(m, r)
    prior, post = jl.marginals(S, blocks), poe_posterior(S, blocks, lab)
    rows = []
    for cand, o in zip(r["cands"], outs):
        for s, q, p in zip(o["slots"], cand["z_q"], o["marg"].tolist()):
            if s[:2] in ("zR", "zO") or (s == "z0" and not cand["seq"]):
                rows.append({"candidate": cand["query"], "slot": s, "query": q, "label": lab.get(q), "per_candidate": p,
                             "root_prior": prior[q], "root_posterior_offline": post[q]})
    return {"root": r["key"], "rows": rows}


def synth(r, seq, base) -> dict:
    """An UNLABELLED hypothetical candidate (model scoring only; never a teacher query) with canonical slot identities."""
    sl = [s for s in base["slots"] if s[:2] in ("z0", "zS")]
    zq = [q for s, q in zip(base["slots"], base["z_q"]) if s[:2] in ("z0", "zS")]
    slots, qs = sl + (["zR"] if seq else []), zq + ([canon(r, seq)] if seq else [])
    if len(seq) >= 2:
        for t in range(len(seq)):
            slots.append(f"zO[{t}]")
            qs.append(canon(r, [a for k, a in enumerate(seq) if k != t]))
    n = r["R"].n
    return {"seq": list(seq), "slots": slots, "query": canon(r, seq), "y": None, "w": 1.0, "src": None, "cost": None,
            "z": [None] * len(slots), "z_q": qs, "su": [None] * n, "sp": [None] * (n * (n - 1) // 2)}


def cases(r, generic):
    """Saved root -> {case: (expected, modified root)}; modified roots only add / reorder / duplicate / rename candidates.
    `generic`: the arm has scene-repair coupling, so a new zR-role block brings a genuinely new relational factor."""
    cand, empty = [c for c in r["cands"] if c["seq"]], [c for c in r["cands"] if not c["seq"]]
    out = {"permutation": ("EXPECTED_INVARIANT", r | {"cands": cand[::-1] + empty}),
           "exact_duplicate": ("EXPECTED_INVARIANT", r | {"cands": r["cands"] + [copy.deepcopy(max(cand, key=lambda c: len(c["seq"])))]})}
    used = {(i, cell(d)) for c in cand for i, d in c["seq"]}
    objs = sorted({i for c in cand for i, _ in c["seq"]})
    dests = sorted({tuple(map(float, d)) for c in cand for _, d in c["seq"]})
    free = [(i, d) for i in objs for d in dests if (i, cell(d)) not in used]
    if free:
        out["disconnected_candidate"] = ("EXPECTED_INVARIANT", r | {"cands": r["cands"] + [synth(r, [free[0]], cand[0])]})
    have = {c["query"] for c in cand}
    om_only = [(c, s, q) for c in cand for s, q in zip(c["slots"], c["z_q"]) if s.startswith("zO") and q not in have]
    if om_only:                                       # shortest existing omission query, added as a complete candidate
        c, s, q = min(om_only, key=lambda x: (len(x[0]["seq"]), x[2]))
        seq = [a for k, a in enumerate(c["seq"]) if k != int(s[3:-1])]
        new_factor = generic or len(seq) >= 2                # scene-repair coupling, or its own new zR-zO omission pairs
        out["alias_as_new_candidate"] = ("ALLOWED_CHANGE" if new_factor else "EXPECTED_INVARIANT", r | {"cands": r["cands"] + [synth(r, seq, cand[0])]})
    ones = [c for c in cand if len(c["seq"]) == 1]
    pair = next(((a, b) for a in ones for b in ones if a["seq"][0][0] != b["seq"][0][0] and canon(r, a["seq"] + b["seq"]) not in have), None)
    if pair:
        out["related_candidate"] = ("ALLOWED_CHANGE", r | {"cands": r["cands"] + [synth(r, pair[0]["seq"] + pair[1]["seq"], cand[0])]})
    ren = copy.deepcopy(r)
    mp = {q: f"seq:{r['key']}:renamed{k}" for k, q in enumerate(sorted({q for c in cand for s, q in zip(c["slots"], c["z_q"]) if s[:2] in ("zR", "zO")}))}
    for c in ren["cands"]:
        c["z_q"] = [mp.get(q, q) for q in c["z_q"]]
        c["query"] = mp.get(c["query"], c["query"])
    ren["y"] = {mp.get(q, q): y for q, y in ren["y"].items()}
    out["renamed_ids"] = ("EXPECTED_INVARIANT", ren)
    return out, mp


@torch.no_grad()
def sensitivity(m, roots, marg_fn) -> list:
    """Max |change| of prior marginals of the ORIGINAL queries (renamed back) and of the scene prior, per case."""
    rows = []
    for r in roots:
        base = marg_fn(m, r)
        cs, mp = cases(r, m.couple in ("full", "quadratic"))
        inv = {v: k for k, v in mp.items()}
        for name, (exp, rr) in cs.items():
            new = {inv.get(q, q): p for q, p in marg_fn(m, rr).items()}
            scene = sorted((q for q in base if q.startswith("scene") or q.endswith(":empty")), key=lambda q: not q.endswith(":empty"))
            d_rep = max((abs(new[q] - base[q]) for q in base if q not in scene), default=0.0)
            d_sc = max((abs(new[q] - base[q]) for q in scene), default=0.0)
            pick = lambda mg: min(((len(c["seq"]), -mg[c["query"] if c["seq"] else scene[0]], c["query"]) for c in r["cands"]  # noqa: E731
                                   if mg[c["query"] if c["seq"] else scene[0]] >= 0.5), default=None)
            rows.append({"root": r["key"], "case": name, "expected": exp, "max_abs_change_repair_marginals": d_rep, "max_abs_change_scene_prior": d_sc,
                         "choice_changed": (pick(base) or (0, 0, None))[2] != (pick(new) or (0, 0, None))[2]})
    return rows


def poe_marg(m, r):
    S, blocks, _, _ = jl.tables(m, r)
    return jl.marginals(S, blocks)


def dedup_poe_marg(m, r):
    """DIAGNOSTIC ONLY: PoE after dropping exactly repeated candidate blocks (not the unique-factor model)."""
    S, blocks, _, _ = jl.tables(m, r)
    seen, keep = set(), []
    for b in blocks:
        if b["cand"] not in seen:
            seen.add(b["cand"])
            keep.append(b)
    return jl.marginals(S, keep)


def pytest_file(name, path) -> None:
    """Run one test module inside a guarded job (ledgered); result in runs/p9/l3/<name>.txt."""
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", path], capture_output=True, text=True)
    (OUT / f"{name}.txt").write_text(r.stdout[-6000:] + r.stderr[-2000:] + f"\nEXIT {r.returncode}\n")


def audit() -> None:
    a = OUT / "l3a0"
    a.mkdir(parents=True, exist_ok=True)
    train, val, test = data()
    st = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.rstrip().splitlines()
    head = subprocess.run(["git", "rev-parse", "HEAD", "origin/main"], capture_output=True, text=True).stdout.split()
    man = _rd(L2 / "ckpt/checkpoint_manifest.json")
    ck = {k: (L2 / "ckpt" / k).exists() and sha(L2 / "ckpt" / k) == v["sha256"] for k, v in man.items()}
    gates = {g: _rd(f"runs/p9/{g}.json")["verdict"] for g in ("P9_L0_GATE", "P9_L1_GATE", "P9_L2_GATE")}
    cf.write_json(a / "source_state.json", {"head": head[0], "origin_main": head[1], "tracked_changes": [x for x in st if not x.startswith("??")],
                                            "sources": {f: sha(f) for f in SOURCES}, "datasets": {"dev": sha(DEV), "test": sha(TEST)},
                                            "l2_checkpoints_match_manifest": ck, "p9_gates": gates,
                                            "docs": {f: sha(f"instruction-phase-md/{f}") for f in (
                                                "latent_causal_tamp_methodology_v1_8.md", "latent_causal_tamp_implementation_guide_v2_0.md",
                                                "0028_P9_Opus_5_5_High_Joint_Boltzmann_Migration_Prompt.md",
                                                "0031_P9_L3a_Factor_Graph_Semantics_and_Saved_Data_Controls.md")}})
    allr = train + val + test
    cf.write_json(a / "identity.json", identity_audit(allr))
    cf.write_json(a / "multiplicity.json", {"dev_and_disclosed_test": multiplicity(allr), "validation": multiplicity(val)})
    rich = [r for r in val if len(r["cands"]) > 1]
    rich += [r for r in train if any(s_.startswith("zO") and q not in {c["query"] for c in r["cands"]}
                                     for c in r["cands"] for s_, q in zip(c["slots"], c["z_q"]))][:6]
    pick = max(rich[:6], key=lambda r: sum(len(c["z_q"]) for c in r["cands"]))
    bf, bs = load("B-full", L2 / "ckpt/ckpt_B-full#s0.pt"), load("B-structured", L2 / "ckpt/ckpt_B-structured#s0.pt")
    cf.write_json(a / "prediction_modes.json", {"model": "P9-L2 B-full seed 0 (PoE)", **modes(bf, pick)})
    scr = copy.deepcopy(pick)
    scr["y"] = {q: None for q in scr["y"]}
    for c in scr["cands"]:
        c["z"], c["y"] = [None] * len(c["z"]), None
    iso = max(abs(poe_marg(bf, pick)[q] - p) for q, p in poe_marg(bf, scr).items())
    sens = {}
    for arm, m in (("B-full", bf), ("B-structured", bs)):
        for name, fn in (("P9-PoE", poe_marg), ("dedup-PoE diagnostic", dedup_poe_marg)):
            sens[f"{arm}|{name}"] = sensitivity(m, rich, fn)
    cf.write_json(a / "sensitivity.json", {"roots": [r["key"] for r in rich], "label_isolation_max_abs_prior_change": iso, "rows": sens})
    cf.write_json(a / "audit_done.json", {"teacher_calls": 0, "planner_calls": 0, "cpu_s": round(time.process_time(), 1)})


def unique_marg(m, r):
    import unique_factor as uf
    return uf.prior_marginals(uf.instances(m, r))


def l3a1_artifacts() -> None:
    """Semantics comparison with FROZEN P9-L2 (PoE-trained) weights: sensitivity of PoE / dedup-PoE / unique, brute force,
    normalization, PoE-vs-unique prior marginals by unary multiplicity, cost per root, factor-instance counts."""
    import unique_factor as uf
    a = OUT / "l3a1"
    a.mkdir(parents=True, exist_ok=True)
    train, val, test = data()
    rich = [r for r in val if len(r["cands"]) > 1]
    rich += [r for r in train if any(s_.startswith("zO") and q not in {c["query"] for c in r["cands"]}
                                     for c in r["cands"] for s_, q in zip(c["slots"], c["z_q"]))][:6]
    ms = {arm: load(arm, L2 / f"ckpt/ckpt_{arm}#s0.pt") for arm in ("B-full", "B-structured", "C-full")}
    sens = {f"{arm}|{name}": sensitivity(m, rich, fn) for arm, m in ms.items()
            for name, fn in (("P9-PoE", poe_marg), ("dedup-PoE diagnostic", dedup_poe_marg), ("unique-factor", unique_marg))}
    rows, bf, sc, cost, cnt = [], [], 0.0, [], []
    jl.PEAK[0] = 0
    m = ms["B-full"]
    for r in val + test + [r for r in train if len(r["cands"]) > 1][:40]:
        with torch.no_grad():
            c0 = time.process_time()
            inst = uf.instances(m, r)
            lab = jl.labels(r)
            ll = float(uf.log_evidence(inst, lab)[0])
            pu = uf.prior_marginals(inst)
            c1 = time.process_time()
            S, blocks, _, _ = jl.tables(m, r)
            pp = jl.marginals(S, blocks)
            c2 = time.process_time()
        cost.append((c1 - c0, c2 - c1))
        sc = max(sc, max(abs(pu[q] - float(torch.exp(S["logp"]) @ S["Z"][:, i])) for i, q in enumerate(S["queries"])))
        if len(uf.variables(inst)) <= 10:
            bf.append(abs(ll - uf.brute_force(inst, lab)))
        mult = collections.Counter(q for b in blocks for q in b["queries"])
        cnt.append({"root": r["key"], "distinct_queries": len(uf.variables(inst)), "poe_unary_contributions": sum(mult.values()),
                    "unique_unary_instances": len(inst["unary"]), "candidate_blocks": len(blocks), "unique_blocks": len(inst["blocks"])})
        for q in pp:
            if q in mult:
                rows.append({"root": r["key"], "split": r["split"], "query": q, "multiplicity": mult[q], "label": lab.get(q),
                             "poe_prior": pp[q], "unique_prior": pu[q]})
    d = np.array([abs(x["poe_prior"] - x["unique_prior"]) for x in rows])
    k = np.array([x["multiplicity"] for x in rows])
    cf.write_json(a / "artifacts.json", {
        "weights": "P9-L2 seed-0 checkpoints trained under the PoE objective (semantics comparison, not a fitted comparison)",
        "sensitivity": sens, "brute_force_roots": len(bf), "brute_force_max_abs_diff": max(bf), "scene_prior_max_abs_diff": sc,
        "peak_elimination_scope": jl.PEAK[0], "cpu_s_per_root": {"unique": float(np.mean([x for x, _ in cost])), "poe": float(np.mean([y for _, y in cost]))},
        "poe_vs_unique_prior_abs_diff_by_multiplicity": {int(j): {"queries": int((k == j).sum()), "mean": float(d[k == j].mean()), "max": float(d[k == j].max())}
                                                         for j in sorted(set(k.tolist()))},
        "factor_instance_counts": {"roots": len(cnt), "poe_unary_contributions": int(sum(x["poe_unary_contributions"] for x in cnt)),
                                   "unique_unary_instances": int(sum(x["unique_unary_instances"] for x in cnt)),
                                   "candidate_blocks": int(sum(x["candidate_blocks"] for x in cnt)), "unique_blocks": int(sum(x["unique_blocks"] for x in cnt))},
        "teacher_calls": 0, "planner_calls": 0})
    cf.write_json(a / "marginal_rows.json", rows)


def regression() -> None:
    """Full historical suite + six archived replays (planner FIXTURE calls counted separately by countcalls; no learned inference)."""
    d = OUT / "regression"
    d.mkdir(parents=True, exist_ok=True)
    (d / "countcalls.py").write_text(Path("runs/p8/audit/regression/countcalls.py").read_text())
    subprocess.run([sys.executable, "run.py", "g4c-lf-p4-regression", "--out", str(d), "--cpu", "1700", "--wall", "3500"], capture_output=True, text=True)


ISOLATION = r'''
import json, sys, torch
import abc7_train as t7, abc8, joint_likelihood as jl, unique_factor as uf
torch.set_num_threads(1)
roots = [r for f in ("runs/p8/c_training/dataset.pt", "runs/p8/c_training/dataset_sealed.pt")
         for r in t7.prepare(torch.load(f, weights_only=False)) if r["split"] in ("validation", "test")]
n = 0
for arm in ("B-full", "B-structured"):
    m = abc8.build(arm)
    m.load_state_dict(torch.load(f"runs/p9/l2/ckpt/ckpt_{arm}#s0.pt", weights_only=True))
    for r in roots:
        uf.prior_marginals(uf.instances(m, r)); uf.losses(m, r)
        S, b, lab, _ = jl.tables(m, r); jl.marginals(S, b)
        n += 1
bad = sorted(k for k in sys.modules if k in ("planner", "motion", "skill", "ordered", "evaluate", "teacher", "verify", "geometry", "operational_teacher"))
print(json.dumps({"root_inferences": n, "physical_modules_loaded": bad}))
'''


def isolation_check() -> None:
    """Learned inference (PoE and unique) in a fresh interpreter that never loads planner / motion / skill / teacher code."""
    r = subprocess.run([sys.executable, "-c", ISOLATION], capture_output=True, text=True)
    cf.write_json(OUT / "l3a1" / "isolation.json", json.loads(r.stdout.strip().splitlines()[-1]) | {"returncode": r.returncode})


def figures_l3a() -> None:
    import p9_viz
    p9_viz.l3a_factor_graph()
    p9_viz.l3a_invariance()


def _ho_dependence() -> bool:
    """The unique-factor evidence of B-full must depend on its higher-order (ho_q, ho_s) and generic-pair (cpl) modules."""
    import unique_factor as uf
    _, val, _ = data()
    r = next(r for r in val if len(r["cands"]) > 3)
    m = load("B-full", L2 / "ckpt/ckpt_B-full#s0.pt")
    with torch.no_grad():
        base = float(uf.log_evidence(uf.instances(m, r), jl.labels(r))[0])
        ok = True
        for mod in ("ho_q", "ho_s", "cpl"):
            last = list(getattr(m, mod).parameters())[-2]          # final-layer WEIGHT (an output bias is z-constant and cancels)
            last += 0.05
            ok &= abs(float(uf.log_evidence(uf.instances(m, r), jl.labels(r))[0]) - base) > 1e-6
            last -= 0.05
    return bool(ok)


def gate_job() -> None:
    gate_l3a1()


def gate_l3a1() -> dict:
    a = _rd(OUT / "l3a1" / "artifacts.json")
    reg = _rd(OUT / "regression" / "regression_result.json")
    tests = (OUT / "l3a1_tests.txt").read_text()
    iso = _rd(OUT / "l3a1" / "isolation.json")
    inv = {}
    for key, rows in a["sensitivity"].items():
        if key.endswith("unique-factor"):
            for x in rows:
                if x["expected"] == "EXPECTED_INVARIANT":
                    inv[f"{key}|{x['case']}"] = max(inv.get(f"{key}|{x['case']}", 0.0), x["max_abs_change_repair_marginals"])
    figs = ["P9_L3a_real_root_factor_graph.png", "P9_L3a_invariance.png"]
    checks = {"l3a0_pass": _rd(OUT / "L3A_0_GATE.json")["verdict"] == "PASS",
              "unique_factor_tests_pass": "26 passed" in tests and "EXIT 0" in tests,
              "historical_suite_and_six_replays_pass": reg["pytest_exit"] == "0" and reg["tests"]["failed"] == 0 and all(reg["replays"].values()) and len(reg["replays"]) == 6,
              "brute_force_on_saved_roots": a["brute_force_roots"] >= 50 and a["brute_force_max_abs_diff"] < 1e-9,
              "scene_prior_root_consistent": a["scene_prior_max_abs_diff"] < 1e-12,
              "unique_invariant_in_every_expected_case": all(v < 1e-10 for v in inv.values()) and len(inv) >= 12,
              "bfull_higher_order_and_generic_pairs_retained": _ho_dependence(),
              "learned_inference_loads_no_physical_module": iso["physical_modules_loaded"] == [] and iso["returncode"] == 0,
              "complexity_within_cap": a["peak_elimination_scope"] <= jl.MAX_SCOPE,
              "figures_hashed_with_manifests": all((Path("visualizations/p9") / f"{f}.manifest.json").exists() for f in figs)}
    g = {"stage": "P9-L3a-1", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "unique_invariance_max_change": inv,
         "arms": {"B-full": "PASS (unary, z0-zS, zR-zO, generic pairs and BOTH higher-order terms retained)", "B-structured": "PASS",
                  "C-full": "PASS (one-repair equivalence to ABC7 joint and invariances)", "B-quadratic": "NOT_RUN (not required)"},
         "suite": reg["tests"], "planner_fixture_calls_in_suite": reg["calls"], "regression_cpu_s": reg["guard"]["cpu_s"]}
    cf.write_json(OUT / "L3A_1_GATE.json", g)
    return g


# ------------------------------------------------------------------ L3b: matched saved-data comparison (gated)
L3B = OUT / "l3b"
ARMS = {"U-B-full": ("B-full", "unique"), "U-B-structured": ("B-structured", "unique"), "U-B-full-geo": ("B-full-geo", "unique"),
        "U-C-full": ("C-full", "unique")}
BASE = {"PoE-B-full": ("B-full", "poe"), "PoE-B-structured": ("B-structured", "poe"), "PoE-C-full": ("C-full", "poe"),
        "SF": ("SF", "factorized"), "A": ("A", "factorized")}


def build(arm):
    import abc7_models as a7
    return a7.ModelBC7("B", "full", vision=False).double() if arm == "B-full-geo" else abc8.build(arm)


def plan_l3b() -> dict:
    train, val, test = data()
    p = {"version": "P9_L3B_PLAN_V1", "model_version": "ABC9U_UNIQUE_V1 (unique_factor.py) on ABC7_V1 factor modules; trained from scratch",
         "objective": "mean_roots(-log P_U(D_c)) + 1.0 static BCE (endpoint-deduplicated) + 1.0 log-Huber witnessed J (identical auxiliaries to P9-L2)",
         "factor_schema": "one unary phi_seq per distinct sequence query; per distinct repair block its zR-zO pairs, scene-repair and zO-zO generic "
                          "pairs (context g_r, role types) and higher-order term (full); P_S once; one conditional normalizer per scene state",
         "data": {"dataset_sha256": sha(DEV), "test_sha256": sha(TEST), "train_roots": [r["key"] for r in train],
                  "validation_roots": [r["key"] for r in val], "disclosed_test_roots": [r["key"] for r in test],
                  "labels": "every decided status (VERIFIED 1, CERTIFIED 0, OPNEG-v1 BUDGET_INFEASIBLE 0); UNKNOWN / NOT_RUN missing"},
         "matching": "torch.manual_seed(seed) before build, abc8.RootSampler(train, 12, seed), AdamW lr 1e-3 wd 1e-4, clip 10, 200 updates: "
                     "identical initialization (except geometry-only, whose layer list differs) and identical minibatches as the P9-L2 PoE runs",
         "selection": "best validation root NLL of the model's own likelihood every 20 updates (restored); tau / threshold on validation only",
         "jobs": [{"name": f"{a}#s{s}", "arm": a, "seed": s, "updates": 200} for a in ARMS for s in (0, 1, 2)],
         "baselines_no_retraining": {k: f"runs/p9/l2/ckpt/ckpt_{v[0]}#s<seed>.pt evaluated in mode {v[1]}" for k, v in BASE.items()},
         "optional": {"jobs": [{"name": f"{a}#s{s}#u400", "arm": a, "seed": s, "updates": 400} for s in (0, 1, 2) for a in ("U-B-full", "PoE-B-full")],
                      "rule": "only after every mandatory job and baseline evaluation, if remaining CPU >= 2,400 s; both arms restart from scratch"},
         "decision_rule": "invariances mandatory; prefer the simpler model unless predeclared VALIDATION root-cluster contrasts favour the other; "
                          "disclosed test is a legacy diagnostic and never selects", "caps": CAPS, "teacher_calls": 0,
         "sources": {f: sha(f) for f in ("unique_factor.py", "joint_likelihood.py", "abc7_models.py", "abc7_train.py", "abc8.py", "p9_l3.py")}}
    p["hash"] = cf.stable_hash(p)
    return p


def fit(arm, seed, updates, train, val, objective):
    import unique_factor as uf
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = build(arm)
    tot_fn = uf.total if objective == "unique" else jl.total
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    smp, hist, best = abc8.RootSampler(train, 12, seed), [], (float("inf"), None, -1)
    for u in range(updates):
        batch = [train[i] for i in smp.next()]
        opt.zero_grad()
        tot, rec = tot_fn(m, batch)
        if not torch.isfinite(tot):
            hist.append({"update": u, "nonfinite": True})
            break
        tot.backward()
        g = collections.defaultdict(float)
        for n, p in m.named_parameters():
            if p.grad is not None:
                g[n.split(".")[0]] += float(p.grad.norm()) ** 2
        pre = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0))
        opt.step()
        rec |= {"update": u, "total": float(tot.detach()), "grad_pre": pre, "clipped": pre > 10.0, "grad_by_module": {k: round(v ** 0.5, 5) for k, v in g.items()}}
        if (u + 1) % 20 == 0:
            with torch.no_grad():
                rec["val"] = tot_fn(m, val)[1]["nll"]
            if rec["val"] < best[0]:
                best = (rec["val"], copy.deepcopy(m.state_dict()), u + 1)
        hist.append(rec)
    if best[1] is not None:
        m.load_state_dict(best[1])
    return m, {"history": hist, "best_update": best[2], "val_root_nll": best[0], "updates": updates, "params": sum(p.numel() for p in m.parameters()),
               "clip_fraction": float(np.mean([h.get("clipped", False) for h in hist])), "nonfinite": any(h.get("nonfinite") for h in hist)}


def _cond_poe(S, blocks, q, L):
    return float(torch.exp(jl.log_evidence(S, blocks, L | {q: 1})[0] - jl.log_evidence(S, blocks, L)[0]))


@torch.no_grad()
def root_eval9(m, r, mode) -> dict:
    """Per-candidate rows (P8 convention), ROOT-PRIOR rows of the model's own joint, own exact evidence, and OFFLINE conditionals."""
    import p7_eval as ev
    import unique_factor as uf
    e, lab = ev.evaluate(m, r), jl.labels(r)
    sq = [q for c in r["cands"] if not c["seq"] for s, q in zip(c["slots"], c["z_q"]) if s[:2] in ("z0", "zS")]
    if mode == "unique":
        inst = uf.instances(m, r)
        pm, (ll, n) = uf.prior_marginals(inst), uf.log_evidence(inst, lab)
        cond = lambda q, L: uf.conditional_marginal(inst, q, L)  # noqa: E731
    elif mode == "poe":
        S, blocks, _, _ = jl.tables(m, r)
        pm, (ll, n, _) = jl.marginals(S, blocks), jl.log_evidence(S, blocks, lab)
        cond = lambda q, L: float(L[q]) if L.get(q) is not None else _cond_poe(S, blocks, q, L)  # noqa: E731
    else:
        lg = jl.factorized_logits(m, r, t7.forward(m, r))
        pm = {q: float(torch.sigmoid(x)) for q, x in lg.items()}
        obs = [(q, y) for q, y in lab.items() if y is not None and q in lg]
        ll, n = sum(np.log(pm[q] if y else 1 - pm[q]) for q, y in obs), len(obs)
        cond = lambda q, L: float(L[q]) if L.get(q) is not None else pm[q]  # noqa: E731
    root = copy.deepcopy(e)
    for row, c in zip(root["cands"], r["cands"]):
        p = min(max(pm[c["query"] if c["seq"] else sq[0]], 1e-9), 1 - 1e-9)
        u = float(np.log(p) - np.log1p(-p))
        row |= {"u": u, "p": p, "energy": row["energy"] - float(np.logaddexp(0, -row["u"])) + float(np.logaddexp(0, -u))}
    root["effects"] = [x for x in e["effects"] if x["kind"] == "scene"]
    conds = []
    sl = {q: lab.get(q) for q in sq}
    for c in r["cands"]:
        if not c["seq"]:
            continue
        zo = [(s, q) for s, q in zip(c["slots"], c["z_q"]) if s.startswith("zO")]
        for s, q in zo:
            if lab.get(q) is not None and c["y"] is not None:
                root["effects"].append({"kind": "omission", "query": c["query"], "pred": pm[c["query"]] - pm[q], "observed": c["y"] - lab[q]})
        if c["y"] is not None and all(v is not None for v in sl.values()):
            conds.append({"task": "zR|all scene bits", "p": cond(c["query"], sl), "y": c["y"]})
        for a in sq[1:]:
            conds.append({"task": "dep zR on zS[I]", "p": abs(cond(c["query"], {a: 1}) - cond(c["query"], {a: 0})), "y": None})
        for s, q in zo:
            if lab.get(q) is not None and c["y"] is not None:
                conds.append({"task": "zO|zR", "p": cond(q, {c["query"]: c["y"]}), "y": lab[q]})
            conds.append({"task": "dep zO on zR", "p": abs(cond(q, {c["query"]: 1}) - cond(q, {c["query"]: 0})), "y": None})
    bits = collections.defaultdict(list)
    for q, y in lab.items():
        if y is not None and q in pm:
            k = "scene" if q in sq else ("repair" if any(c["query"] == q for c in r["cands"]) else "omission_only")
            p = min(max(pm[q], 1e-9), 1 - 1e-9)
            bits[k].append(-np.log(p if y else 1 - p))
    return {"key": r["key"], "expert": e, "root": root, "nll": -float(ll) if n else None, "n_obs": n, "catalogue": len(r["cands"]) > 1,
            "bits": {k: list(map(float, v)) for k, v in bits.items()}, "cond": conds}


def evaluate9(m, mode, val, test) -> dict:
    import p7_eval as ev
    import p8_eval as pe
    import p8_select as ps
    rec = _rd("runs/p8/collection/decisions.json") | _rd("runs/p8/collection/sealed_test.json")
    C8 = Path("runs/p8/collection")
    res = {s: [root_eval9(m, r, mode) for r in rs] for s, rs in (("validation", val), ("test_disclosed", test))}
    out = {"mode": mode}
    for kind in ("root", "expert"):
        vev = [x[kind] for x in res["validation"]]
        tau = ev.fit_temperature(vev)
        th = ps.choose_threshold(vev, rec, C8, tau)["chosen"]
        out[kind] = {"tau": tau, "threshold": th}
        for split, xs in res.items():
            evs = [x[kind] for x in xs]
            met, fun = ev.metrics(evs, tau), ps.funnel(evs, rec, C8, tau, th)
            out[kind][split] = {"repair": {k: met.get(k) for k in ("decided", "accuracy", "auroc", "nll", "ece", "ci95_accuracy", "effect_mae",
                                                                   "selection_success", "selection_minimal")},
                                "intervention": pe.unconditional(evs),
                                "funnel": {k: sum(f[k] for f in fun) for k in ("success", "abstain", "false_feasible_empty",
                                                                               "shorter_infeasible_preempted", "verified_candidate_exists")} | {"roots": len(fun)}}
    for split, xs in res.items():
        nl = [x for x in xs if x["nll"] is not None]
        bits = collections.defaultdict(list)
        cond = collections.defaultdict(list)
        for x in xs:
            for k, v in x["bits"].items():
                bits[k] += v
            for c in x["cond"]:
                p = min(max(c["p"], 1e-9), 1 - 1e-9)
                cond[c["task"]].append(c["p"] if c["y"] is None else -np.log(p if c["y"] else 1 - p))
        out[split] = {"root_nll": float(np.mean([x["nll"] for x in nl])), "root_nll_catalogue": float(np.mean([x["nll"] for x in nl if x["catalogue"]])),
                      "root_nll_single_bit": float(np.mean([x["nll"] for x in nl if not x["catalogue"]])),
                      "nll_per_observed_query": float(sum(x["nll"] for x in nl) / sum(x["n_obs"] for x in nl)), "roots_with_labels": len(nl),
                      "per_root": {x["key"]: [x["nll"], x["n_obs"]] for x in nl},
                      "bit_nll": {k: [float(np.mean(v)), len(v)] for k, v in bits.items()},
                      "offline_conditionals": {k: [float(np.mean(v)), len(v)] for k, v in cond.items()}}
        if mode != "factorized":
            out[split]["legacy_per_candidate_conditional"] = pe.summarize_conditional({r["key"]: pe.conditional(m, r) for r in (val if split == "validation" else test)})
    sens_fn = {"unique": unique_marg, "poe": poe_marg}.get(mode)
    if sens_fn:
        rows = sensitivity(m, [r for r in val if len(r["cands"]) > 1], sens_fn)
        out["sensitivity"] = {f"{x['case']}|{x['expected']}": max(y["max_abs_change_repair_marginals"] for y in rows
                                                                  if (y["case"], y["expected"]) == (x["case"], x["expected"])) for x in rows}
    return out


def train_job(name, arm, seed, updates) -> None:
    plan = _rd(L3B / "L3B_plan.json")
    train, val, test = data()
    base, objective = (ARMS | {"PoE-B-full": ("B-full", "poe")})[arm]
    c0 = time.process_time()
    m, rec = fit(base, seed, updates, train, val, objective)
    rec["train_cpu_s"] = round(time.process_time() - c0, 1)
    ck = L3B / "ckpt" / f"ckpt_{name}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ck)
    jl.PEAK[0] = 0
    rec |= {"name": name, "arm": arm, "seed": seed, "objective": objective, "plan_hash": plan["hash"], "checkpoint_sha256": sha(ck),
            "eval": evaluate9(m, objective, val, test), "peak_scope": jl.PEAK[0]}
    (L3B / "runs").mkdir(parents=True, exist_ok=True)
    cf.write_json(L3B / "runs" / f"{name}.json", rec)


def eval_baselines() -> None:
    _, val, test = data()
    out = {}
    for name, (arm, mode) in BASE.items():
        for s in (0, 1, 2):
            m = load(arm, L2 / f"ckpt/ckpt_{arm}#s{s}.pt") if s == 0 else _load_unmanaged(arm, s)
            if m is None:
                out[f"{name}#s{s}"] = {"status": "NO_CHECKPOINT"}
                continue
            out[f"{name}#s{s}"] = {"arm": name, "seed": s, "mode": mode, "eval": evaluate9(m, mode, val, test),
                                   "l2_record": {k: _rd(L2 / f"runs/{arm}#s{s}.json")[k] for k in ("best_update", "clip_fraction", "cpu_s", "clip")}}
    cf.write_json(L3B / "baselines.json", out)


def _load_unmanaged(arm, s):
    p = L2 / f"ckpt/ckpt_{arm}#s{s}.pt"
    man = _rd(L2 / "ckpt/checkpoint_manifest.json")
    if not p.exists() or sha(p) != man[p.name]["sha256"]:
        return None
    return load(arm, p)


def smoke() -> None:
    """Code smoke before the plan is frozen: 2 updates of each objective path and every evaluation mode on 2 + 2 roots."""
    train, val, test = data()
    vv, tt = [r for r in val if len(r["cands"]) > 1][:2], [r for r in test if len(r["cands"]) > 1][:1] + [r for r in test if len(r["cands"]) == 1][:1]
    out = {}
    for arm, mode in (("B-full-geo", "unique"), ("B-structured", "poe")):
        m, rec = fit(arm, 0, 2, train, vv, mode)
        e = evaluate9(m, mode, vv, tt)
        out[arm] = {"val_nll": rec["val_root_nll"], "test_root_nll": e["test_disclosed"]["root_nll"], "keys": sorted(e)}
    e = evaluate9(load("SF", L2 / "ckpt/ckpt_SF#s0.pt"), "factorized", vv, tt)
    out["SF"] = {"test_root_nll": e["test_disclosed"]["root_nll"]}
    cf.write_json(OUT / "smoke.json", out)


def run_l3b() -> dict:
    for g in ("L3A_0_GATE", "L3A_1_GATE"):
        if _rd(OUT / f"{g}.json")["verdict"] != "PASS":
            raise SystemExit(f"{g} not PASS: L3b not authorized")
    L3B.mkdir(parents=True, exist_ok=True)
    if not (L3B / "L3B_plan.json").exists():
        cf.write_json(L3B / "L3B_plan.json", plan_l3b())
    plan, status = _rd(L3B / "L3B_plan.json"), {}
    for j in plan["jobs"]:
        if not (L3B / "runs" / f"{j['name']}.json").exists():
            g = job(j["name"].replace("#", "_"), f"train_job({j['name']!r}, {j['arm']!r}, {j['seed']}, {j['updates']})", 1800)
            status[j["name"]] = g["stop"]
    if not (L3B / "baselines.json").exists():
        status["baselines"] = job("baselines", "eval_baselines()", 1800)["stop"]
    if CAPS["aggregate_cpu_s"] - spent()[0] >= 2400:
        for j in plan["optional"]["jobs"]:
            if not (L3B / "runs" / f"{j['name']}.json").exists() and CAPS["aggregate_cpu_s"] - spent()[0] >= 400:
                status[j["name"]] = job(j["name"].replace("#", "_"), f"train_job({j['name']!r}, {j['arm']!r}, {j['seed']}, {j['updates']})", 1800)["stop"]
    else:
        status["optional"] = "NOT_RUN (remaining CPU < 2,400 s)"
    out = {"status": status, "aggregate_cpu_s": round(spent()[0], 1), "aggregate_wall_s": round(spent()[1], 1), "plan_hash": plan["hash"]}
    cf.write_json(L3B / "run_status.json", out)
    return out


def gate_l3a0() -> dict:
    a = OUT / "l3a0"
    s, idn, sens = _rd(a / "source_state.json"), _rd(a / "identity.json"), _rd(a / "sensitivity.json")
    tracked_ok = sorted(s["tracked_changes"]) == sorted([" D latent_causal_tamp_implementation_guide_v1_8.pdf", " D latent_causal_tamp_specification_v1_6.pdf"])
    summ = {}
    for key, rows in sens["rows"].items():
        for case, exp in sorted({(x["case"], x["expected"]) for x in rows}):
            v = [x for x in rows if (x["case"], x["expected"]) == (case, exp)]
            d = max(x["max_abs_change_repair_marginals"] for x in v)
            summ[f"{key}|{case}|{exp}"] = {"expected": exp, "roots": len(v), "max_abs_change": d,
                                     "scene_prior_max_change": max(x["max_abs_change_scene_prior"] for x in v),
                                     "status": ("PASS" if d < 1e-9 else "FAIL") if exp.startswith("EXPECTED_INVARIANT") else "MEASURED"}
    checks = {"head_equals_origin_main_4be6d80": s["head"] == s["origin_main"] == "4be6d803bbfcff693b8055eb4e2e25707a8da4ee",
              "only_preexisting_user_changes_in_tracked_files": tracked_ok,
              "p9_gates_pass": all(v == "PASS" for v in s["p9_gates"].values()),
              "l2_checkpoints_match_manifest": all(s["l2_checkpoints_match_manifest"].values()),
              "query_ids_match_canonical_meaning": idn["counts"].get("slots_mismatch", 0) == 0 and idn["counts"].get("query_two_meanings", 0) == 0,
              "prior_marginals_label_isolated": sens["label_isolation_max_abs_prior_change"] == 0.0,
              "scene_prior_root_consistent_in_every_case": all(v["scene_prior_max_change"] < 1e-12 for v in summ.values()),
              "teacher_planner_calls_zero": _rd(a / "audit_done.json")["teacher_calls"] == 0}
    g = {"stage": "P9-L3a-0", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "poe_sensitivity": summ,
         "note": "PoE FAIL on an EXPECTED_INVARIANT case is a finding about the baseline, not a gate failure"}
    cf.write_json(OUT / "L3A_0_GATE.json", g)
    return g


# ------------------------------------------------------------------ L3b summary and gate (appended after every L3b job finished)
FORMS = {"U-B-full": "unique", "U-B-structured": "unique", "U-B-full-geo": "unique", "U-C-full": "unique",
         "PoE-B-full": "poe", "PoE-B-structured": "poe", "PoE-C-full": "poe", "SF": "factorized", "A": "factorized"}


def _flat(e) -> dict:
    out = {}
    for sp, tag in (("validation", "val"), ("test_disclosed", "test")):
        x = e[sp]
        out |= {f"{tag}_root_nll": x["root_nll"], f"{tag}_root_nll_catalogue": x["root_nll_catalogue"],
                f"{tag}_root_nll_single_bit": x["root_nll_single_bit"], f"{tag}_nll_per_query": x["nll_per_observed_query"]}
        out |= {f"{tag}_bit_nll_{k}": v[0] for k, v in x["bit_nll"].items()}
        out |= {f"{tag}_cond {k}": v[0] for k, v in x["offline_conditionals"].items()}
        if "legacy_per_candidate_conditional" in x:
            out |= {f"{tag}_legacy_cond_{k}": v["nll"] for k, v in x["legacy_per_candidate_conditional"].items()}
        for kind in ("root", "expert"):
            rp, fn, iv = e[kind][sp]["repair"], e[kind][sp]["funnel"], e[kind][sp]["intervention"]
            out |= {f"{kind}.{tag}.{k}": rp.get(k) for k in ("accuracy", "auroc", "nll", "ece", "selection_success")}
            out |= {f"{kind}.{tag}.funnel_{k}": fn[k] for k in ("success", "abstain", "false_feasible_empty", "shorter_infeasible_preempted")}
            out |= {f"{kind}.{tag}.scene_nll": iv["scene_nll"], f"{kind}.{tag}.kappa_mae": iv["kappa_macro_mae"], f"{kind}.{tag}.kappa_by_sign": iv["kappa_by_sign"]}
    out |= {"root.tau": e["root"]["tau"], "root.threshold": e["root"]["threshold"], "sensitivity": e.get("sensitivity")}
    return out


def _famgrad(hist) -> dict:
    h = [x for x in hist if "grad_by_module" in x]
    fam = {"F2": ["order_s", "pairo", "mix", "gru", "h0"], "F3 unary": ["scene"], "F3 z0-zS": ["pair_scene"], "F4 zR-zO": ["pair_omit"],
           "F5 pairs": ["cpl"], "F5 HO": ["ho_u", "ho_s", "ho_q"], "F1": ["unary", "pairh"], "J": ["cost"]}
    out = {}
    for f, mods in fam.items():
        if h and any(m in h[0]["grad_by_module"] for m in mods):
            v = [sum(x["grad_by_module"].get(m, 0) ** 2 for m in mods) ** 0.5 for x in h]
            out[f] = {"mean": float(np.mean(v)), "zero_updates": int(sum(x == 0 for x in v))}
    return out


def summarize_l3b() -> dict:
    rows = {}
    for p in sorted((L3B / "runs").glob("*.json")):
        r = _rd(p)
        rows[r["name"]] = {"arm": r["arm"], "form": FORMS.get(r["arm"], r["objective"]), "seed": r["seed"], "updates": r["updates"],
                           "best_update": r["best_update"], "clip_fraction": r["clip_fraction"], "train_cpu_s": r["train_cpu_s"], "params": r["params"],
                           "peak_scope": r["peak_scope"], "grad": _famgrad(r["history"]), **_flat(r["eval"])}
    for name, b in _rd(L3B / "baselines.json").items():
        if b.get("status") == "NO_CHECKPOINT":
            continue
        rows[name] = {"arm": b["arm"], "form": b["mode"], "seed": b["seed"], "updates": 200, "best_update": b["l2_record"]["best_update"],
                      "clip_fraction": b["l2_record"]["clip_fraction"], "train_cpu_s": b["l2_record"]["cpu_s"], **_flat(b["eval"])}
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    agg = {}
    for key in sorted({(r["arm"], r["updates"]) for r in rows.values()}):
        vs = [r for r in rows.values() if (r["arm"], r["updates"]) == key]
        agg[f"{key[0]}|u{key[1]}"] = {"seeds": len(vs)} | {k: [float(np.mean([v[k] for v in vs if v.get(k) is not None])),
                                                               float(np.std([v[k] for v in vs if v.get(k) is not None]))]
                                                           for k in vs[0] if num(vs[0][k]) and k != "seed"}

    def per_root(arm, split, updates=200):
        d = collections.defaultdict(list)
        for name, r in rows.items():
            if r["arm"] == arm and r["updates"] == updates:
                src = _rd(L3B / "runs" / f"{name}.json")["eval"] if (L3B / "runs" / f"{name}.json").exists() else _rd(L3B / "baselines.json")[name]["eval"]
                for k, (nll, _) in src[split]["per_root"].items():
                    d[k].append(nll)
        return {k: float(np.mean(v)) for k, v in d.items()}

    def boot(dd, n=2000):
        dd, rng = np.asarray(dd, float), np.random.default_rng(0)
        b = [rng.choice(dd, len(dd)).mean() for _ in range(n)]
        return {"mean": float(dd.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "roots": len(dd)}
    pairs = [("U-B-full", "PoE-B-full", "formulation B-full: unique - PoE"), ("U-B-structured", "PoE-B-structured", "formulation B-structured: unique - PoE"),
             ("U-C-full", "PoE-C-full", "formulation C-full: unique - PoE"), ("U-B-full", "U-B-structured", "generic coupling (unique): full - structured"),
             ("U-B-full", "SF", "joint vs factorized: U-B-full - SF"), ("U-B-structured", "SF", "U-B-structured - SF"),
             ("U-B-full", "U-B-full-geo", "appearance: DINOv2+geometry - geometry-only"), ("U-C-full", "U-B-full", "order: GRU (C) - prefix (B)")]
    con = {}
    for split in ("validation", "test_disclosed"):
        for a, b, n in pairs:
            ra, rb = per_root(a, split), per_root(b, split)
            ks = sorted(set(ra) & set(rb))
            if ks:
                con[f"{split}|{n}"] = boot([ra[k] - rb[k] for k in ks])
        ra, rb = per_root("U-B-full", split, 400), per_root("PoE-B-full", split, 400)
        if set(ra) & set(rb):
            con[f"{split}|400 updates: U-B-full - PoE-B-full"] = boot([ra[k] - rb[k] for k in sorted(set(ra) & set(rb))])
    curves = {}
    for p in sorted((L3B / "runs").glob("*#u400.json")):
        r = _rd(p)
        curves[r["name"]] = [(h["update"] + 1, h["val"]) for h in r["history"] if "val" in h]
    top = collections.Counter()
    for split in ("validation", "test_disclosed"):
        nl = per_root("U-B-full", split)
        if nl:
            tot = sum(nl.values())
            top[split] = sum(sorted(nl.values(), reverse=True)[:3]) / tot if tot else None
    out = {"rows": rows, "aggregate": agg, "contrasts": con, "curves": curves, "nll_share_of_top3_roots_U-B-full": dict(top),
           "aggregate_cpu_s": spent()[0], "aggregate_wall_s": spent()[1]}
    cf.write_json(L3B / "summary.json", out)
    return out


def _append_only(f) -> bool:
    """A historic P9 module may only be EXTENDED (new figure functions): zero deleted lines against the committed version."""
    out = subprocess.run(["git", "diff", "--numstat", "4be6d80", "--", f], capture_output=True, text=True).stdout.split()
    return not out or out[1] == "0"


def gate_l3b() -> dict:
    plan, st, s = _rd(L3B / "L3B_plan.json"), _rd(L3B / "run_status.json"), summarize_l3b()
    runs = {p.stem: _rd(p) for p in (L3B / "runs").glob("*.json")}
    mand = [j["name"] for j in plan["jobs"]]
    gs = [_rd(p) for p in (OUT / "guards").glob("*.json")]
    inv = {n: max(v for k, v in r["eval"]["sensitivity"].items() if k.endswith("EXPECTED_INVARIANT")) for n, r in runs.items() if r["objective"] == "unique"}
    import unique_factor as uf
    _, val, _ = data()
    m = build("B-full")
    m.load_state_dict(torch.load(L3B / "ckpt/ckpt_U-B-full#s0.pt", weights_only=True))
    with torch.no_grad():
        replay = float(uf.total(m, val)[1]["nll"])
    src0 = _rd(OUT / "l3a0/source_state.json")
    man = _rd(L2 / "ckpt/checkpoint_manifest.json")
    checks = {"l3a_gates_pass": all(_rd(OUT / f"{g}.json")["verdict"] == "PASS" for g in ("L3A_0_GATE", "L3A_1_GATE")),
              "plan_frozen_before_training_and_hashed_in_every_run": all(r["plan_hash"] == plan["hash"] for r in runs.values()),
              "all_mandatory_jobs_complete": all(n in runs for n in mand),
              "baselines_evaluated": (L3B / "baselines.json").exists(),
              "no_nonfinite": not any(r["nonfinite"] for r in runs.values()),
              "trained_unique_models_invariant": all(v < 1e-9 for v in inv.values()),
              "checkpoint_replay_reproduces_val_nll": abs(replay - runs["U-B-full#s0"]["val_root_nll"]) < 1e-9,
              "aggregate_cpu_within_8000": spent()[0] <= CAPS["aggregate_cpu_s"],
              "aggregate_wall_within_18000": spent()[1] <= CAPS["aggregate_wall_s"],
              "per_job_caps": all(g["cpu_s"] <= CAPS["job_cpu_s"] + 30 and g["wall_s"] <= CAPS["job_wall_s"] for g in gs),
              "historic_sources_and_checkpoints_unchanged": all(sha(f) == h for f, h in src0["sources"].items() if f != "p9_viz.py")
                                                           and _append_only("p9_viz.py")
                                                           and all(sha(L2 / "ckpt" / k) == v["sha256"] for k, v in man.items() if (L2 / "ckpt" / k).exists()),
              "learned_paths_load_no_physical_module": _rd(OUT / "l3a1/isolation.json")["physical_modules_loaded"] == []}
    g = {"stage": "P9-L3b", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "jobs": st["status"],
         "trained_unique_max_invariance_change": inv, "aggregate_cpu_s": round(spent()[0], 1), "aggregate_wall_s": round(spent()[1], 1),
         "note": "engineering gate; scientific decision in P9_L3_DECISION.md"}
    cf.write_json(OUT / "L3B_GATE.json", g)
    return g


def figures_l3b() -> None:
    import p9_viz
    p9_viz.l3b_comparison()
    p9_viz.l3b_health()


def finish_l3b() -> None:
    gate_l3b()
    import p9_viz
    p9_viz.l3b_comparison()
    p9_viz.l3b_health()


if __name__ == "__main__":
    if sys.argv[1] == "job":
        job(sys.argv[2], sys.argv[3], float(sys.argv[4]))
    else:
        print(json.dumps(globals()[sys.argv[1]](), default=str)[:4000])
