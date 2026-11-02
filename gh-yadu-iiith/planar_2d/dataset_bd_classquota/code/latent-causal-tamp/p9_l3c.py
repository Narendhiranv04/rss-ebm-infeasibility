"""P9-L3c: factor ontology, catalogue transformations under the DEPLOYED selector, fail-closed query identity (saved data only).

No teacher, IK, collision or planner call anywhere: learned paths import only abc*/joint_likelihood/unique_factor/p7_eval.
Every compute job runs through `job` (process-tree guard) and is ledgered in runs/p9/l3c/guards (shared with p9_l3d).
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

import config as cf
import joint_likelihood as jl
import p7_eval as ev
import p9_l3 as l3
import unique_factor as uf
from abc_models import T, TYPES

OUT = Path("runs/p9/l3c")
CK = Path("runs/p9/l3/l3b/ckpt")
CAPS = {"aggregate_cpu_s": 5000.0, "aggregate_wall_s": 12000.0, "job_cpu_s": 1500.0, "job_wall_s": 3000.0}
ARMS = {"B-full": "U-B-full#s0", "B-structured": "U-B-structured#s0", "C-full": "U-C-full#s0"}
SOURCES = ["unique_factor.py", "joint_likelihood.py", "abc7_models.py", "abc7_train.py", "abc8.py", "abc_models.py", "p9_l3.py", "p9_run.py",
           "p8_select.py", "p8_eval.py", "p7_eval.py", "p9_viz.py", "tests/test_p9_joint.py", "tests/test_p9_l3_unique.py", "teacher_opneg.toml"]


# ------------------------------------------------------------------ ledgered guarded jobs (one worker at a time)
def spent() -> tuple:
    gs = [l3._rd(p) for p in (OUT / "guards").glob("*.json")]
    return sum(g["cpu_s"] for g in gs), sum(g["wall_s"] for g in gs)


def job(name, call, cpu, module="p9_l3c") -> dict:
    import teacher
    (OUT / "guards").mkdir(parents=True, exist_ok=True)
    c, w = spent()
    cpu = min(cpu, CAPS["job_cpu_s"], CAPS["aggregate_cpu_s"] - c)
    if cpu < 10 or w > CAPS["aggregate_wall_s"] - 60:
        raise SystemExit(f"budget exhausted: {c:.0f} CPU-s / {w:.0f} wall-s")
    g = teacher.guard([sys.executable, "-c", f"import torch; torch.set_num_threads(1); import {module}; {module}.{call}"], cpu,
                      CAPS["job_wall_s"], OUT / "heartbeat.txt", 15.0, 0.25, Path("."))
    n = 1
    while (OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json").exists():
        n += 1
    rec = {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"cpu_cap_s": cpu, "call": f"{module}.{call}"}
    cf.write_json(OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json", rec)
    (OUT / "heartbeat.txt").unlink(missing_ok=True)
    print(json.dumps({k: rec[k] for k in ("stop", "returncode", "cpu_s", "wall_s")} | {"aggregate_cpu_s": round(spent()[0], 1)}))
    return rec


# ------------------------------------------------------------------ identity (fail closed BEFORE any deduplication)
def check_identity(r) -> None:
    """Every z0 / zR / zO slot ID must equal the canonical physical identity of its ordered moves, and one candidate ID
    must name exactly one sequence and one slot structure. uf.instances alone dedupes by ID string (fail-open)."""
    seen = {}
    for c in r["cands"]:
        for s, q in zip(c["slots"], c["z_q"]):
            if s[:2] in ("z0", "zR", "zO"):
                seq = [] if s == "z0" else c["seq"] if s == "zR" else [a for k, a in enumerate(c["seq"]) if k != int(s[3:-1])]
                if q != l3.canon(r, seq):
                    raise jl.IdentityError(f"{r['key']}: slot {s} id {q} != canonical {l3.canon(r, seq)}")
        sig = (l3.canon(r, c["seq"]), tuple(c["slots"]), tuple(c["z_q"]))
        if seen.setdefault(c["query"], sig) != sig:
            raise jl.IdentityError(f"{r['key']}: candidate id {c['query']} names two different physical candidates")


def checked(builder=uf.instances):
    def f(m, r):
        check_identity(r)
        return builder(m, r)
    return f


# ------------------------------------------------------------------ exact per-family decomposition of one repair block
@torch.no_grad()
def components(m, r, c) -> dict:
    """Raw energy tables [2^ks, 2^kq] of candidate c's block split into zR-zO pairs, scene-query pairs, zO-zO pairs and HO."""
    e = next(x for x in r["cands"] if not x["seq"])
    sn = ["z0"] + sorted(n for n in e["slots"] if n.startswith("zS"))
    m._memo = {}
    try:
        S = m.scene_block(r["R"], sn, r["edits"])
        ks, qn = len(sn), ["zR"] + sorted(n for n in c["slots"] if n.startswith("zO"))
        vs, g = [], None
        for n in qn:
            sq = c["seq"] if n == "zR" else [a for k, a in enumerate(c["seq"]) if k != int(n[3:-1])]
            _, gq, _ = m.order(r["R"], sq)
            vs.append(gq + m.typ(torch.tensor(TYPES[n[:2]])))
            g = gq if n == "zR" else g
        kq = len(qn)
        V, Z = torch.cat([S["v"], torch.stack(vs)]), uf._states(ks + kq)
        specs = {"zR-zO pairs (F4)": [(ks, ks + j, "omit") for j in range(1, kq)],
                 "scene-query pairs (F5)": [(a, b, "couple") for a in range(ks) for b in range(ks, ks + kq)],
                 "zO-zO pairs (F5)": [(ks + a, ks + b, "couple") for a, b in itertools.combinations(range(1, kq), 2)]}
        out = {k: (-0.5 * ((Z @ m._pairs(V, g, sp)) * Z).sum(1)).reshape(2 ** ks, 2 ** kq) for k, sp in specs.items()}
        ho = -m.ho_q(torch.cat([Z @ m.ho_u(V), g.expand(len(Z), -1)], 1))[:, 0] if m.couple == "full" else torch.zeros(len(Z), dtype=torch.float64)
        out["higher-order (F5)"] = ho.reshape(2 ** ks, 2 ** kq)
        return out
    finally:
        m._memo = None


# ------------------------------------------------------------------ deployed selector (frozen calibration; P8 lexicographic)
@torch.no_grad()
def deployed(m, r, inst, tau, thr) -> tuple:
    """Root-prior admission p_cal = sigmoid(logit(p)/tau) >= thr, then (moves, predicted J, analytic C_d, energy, id).
    Energy = E_static + T softplus(-logit p_root). Returns (chosen id or None = ABSTAIN, rows)."""
    rows, prior = ev.evaluate(m, r)["cands"], uf.prior_marginals(inst)
    z0 = next(q for c in r["cands"] if not c["seq"] for s, q in zip(c["slots"], c["z_q"]) if s == "z0")
    for row, c in zip(rows, r["cands"]):
        p = min(max(prior[c["query"] if c["seq"] else z0], 1e-12), 1 - 1e-12)
        u = float(np.log(p) - np.log1p(-p))
        row |= {"u": u, "p": p, "energy": row["energy"] - T * float(np.logaddexp(0, -row["u"])) + T * float(np.logaddexp(0, -u))}
    keep = [x for x in rows if 1 / (1 + np.exp(-x["u"] / tau)) >= thr]
    best = min(keep, key=lambda x: (x["m"], x["J_pred"], x["C_d"], x["energy"], x["query"])) if keep else None
    return (best["query"] if best else None), rows


# ------------------------------------------------------------------ ontology: literal variables and factor instances
@torch.no_grad()
def ontology(m, r) -> dict:
    inst = uf.instances(m, r)
    S = inst["S"]
    F = []
    for i, n in enumerate(S["names"]):
        F.append({"family": "F3 scene unary", "net": "scene", "scope": [n], "key": f"scene:{n}", "context": "c0 + edited poses", "in": "P_S (once)"})
        if i:
            F.append({"family": "F3 z0-zS pair", "net": "pair_scene", "scope": ["z0", n], "key": f"pair:z0|{n}", "context": "c0", "in": "P_S"})
    for a, b in itertools.combinations(S["names"][1:], 2):
        if m.couple in ("full", "quadratic"):
            F.append({"family": "F5 scene-scene pair", "net": "cpl", "scope": [a, b], "key": f"pair:{a}|{b}", "context": "c0", "in": "P_S"})
    if m.couple == "full":
        F.append({"family": "F5 scene HO", "net": "ho_s", "scope": S["names"], "key": "ho_s", "context": "c0", "in": "P_S"})
    for q in inst["unary"]:
        F.append({"family": "F2/F4 sequence unary", "net": "order_s", "scope": [q], "key": q, "context": "q's own ordered prefixes", "in": "E_Q (once per query)"})
    for b, blk in inst["blocks"].items():
        c = next(x for x in r["cands"] if x["query"] == b)
        roles = dict(zip(c["z_q"], c["slots"]))
        qs = blk["queries"]
        for q in qs[1:]:
            F.append({"family": "F4 zR-zO pair", "net": "pair_omit", "scope": [qs[0], q], "key": f"{b}|omit {roles[q]}", "context": f"g({b})", "in": "E_Q"})
        if m.couple in ("full", "quadratic"):
            for q in qs:
                for a in S["names"]:
                    F.append({"family": "F5 scene-query pair", "net": "cpl", "scope": [a, q], "key": f"{b}|{a}|{roles[q]}", "context": f"g({b}), role {roles[q][:2]}", "in": "E_Q"})
            for q1, q2 in itertools.combinations(qs[1:], 2):
                F.append({"family": "F5 zO-zO pair", "net": "cpl", "scope": [q1, q2], "key": f"{b}|{roles[q1]}|{roles[q2]}", "context": f"g({b})", "in": "E_Q"})
        if m.couple == "full":
            F.append({"family": "F5 repair HO", "net": "ho_q/ho_u", "scope": S["names"] + qs, "key": f"ho|{b}", "context": f"g({b})", "in": "E_Q"})
    endpoints = {jl._endpoint(c["seq"]) for c in r["cands"]}
    F += [{"family": "F1 static (outside P(z|r))", "net": "unary/pairh", "scope": [], "key": str(k)[:60], "context": "endpoint xy", "in": "static BCE"} for k in endpoints]
    F += [{"family": "travel J (outside P(z|r))", "net": "cost", "scope": [], "key": c["query"], "context": "g(r)", "in": "log-Huber J"} for c in r["cands"] if c["seq"]]
    per_q = collections.defaultdict(collections.Counter)
    for f in F:
        for v in f["scope"]:
            if v.startswith("seq:"):
                per_q[v][f["family"]] += 1
    return {"root": r["key"], "variables": {"scene": S["names"], "sequence_queries": sorted(inst["unary"])}, "factors": F,
            "family_counts": dict(collections.Counter(f["family"] for f in F)), "per_query_factor_counts": {q: dict(v) for q, v in per_q.items()}}


# ------------------------------------------------------------------ catalogue transformations (identical frozen weights, no labels in priors)
def _replace_block(inst, bid, keep, comps):
    out = copy.copy(inst)
    out["blocks"] = dict(inst["blocks"])
    out["blocks"][bid] = dict(inst["blocks"][bid]) | {"energy": sum(comps[k] for k in keep) if keep else torch.zeros_like(inst["blocks"][bid]["energy"])}
    return out


@torch.no_grad()
def transform_rows(m, r, tau, thr, decompose=True) -> list:
    base = uf.instances(m, r)
    pb, lab = uf.prior_marginals(base), jl.labels(r)
    lb, choice0 = float(uf.log_evidence(base, lab)[0]), deployed(m, r, base, tau, thr)[0]
    post0 = {q: uf.conditional_marginal(base, q, lab) for q in pb if lab.get(q) is None}
    cs, mp = l3.cases(r, m.couple in ("full", "quadratic"))
    variants = []
    for name, (exp, rr) in cs.items():
        variants.append((name, exp, rr, uf.instances(m, rr)))
        if decompose and name == "alias_as_new_candidate" and m.couple in ("full", "quadratic"):
            new = rr["cands"][-1]
            comps = components(m, rr, new)
            inst = variants[-1][3]
            for keep, label in (([], "alias: unary only (no new factor)"), (["scene-query pairs (F5)"], "alias: + scene-query pairs only"),
                                (["higher-order (F5)"], "alias: + higher-order only"), (["zR-zO pairs (F4)", "zO-zO pairs (F5)"], "alias: + omission links only")):
                variants.append((label, "DECOMPOSITION", rr, _replace_block(inst, new["query"], keep, comps)))
    out = []
    for name, exp, rr, inst in variants:
        fwd = mp if name == "renamed_ids" else {}                 # only the renamed variant uses different ids
        bwd = {v: k for k, v in fwd.items()}
        p = {bwd.get(q, q): x for q, x in uf.prior_marginals(inst).items()}
        labr = {fwd.get(q, q): y for q, y in lab.items()}         # ONLY the original labels; synthetic candidates are unlabelled
        post = {q: uf.conditional_marginal(inst, fwd.get(q, q), labr) for q in post0}
        ch = deployed(m, rr, inst, tau, thr)[0]
        out.append({"root": r["key"], "case": name, "expected": exp, "new_variables": len(set(uf.variables(inst)) - set(uf.variables(base)) - set(fwd.values())),
                    "max_abs_prior_change": max((abs(p[q] - pb[q]) for q in pb), default=0.0),
                    "scene_prior_change": max(abs(p[q] - pb[q]) for q in base["S"]["queries"]),
                    "existing_label_log_evidence_change": float(uf.log_evidence(inst, labr)[0]) - lb,
                    "max_abs_posterior_change_offline": max((abs(post[q] - post0[q]) for q in post0), default=0.0),
                    "deployed_choice_changed": bwd.get(ch, ch) != choice0,
                    "chose_new_unlabelled_candidate": ch is not None and bwd.get(ch, ch) not in {c["query"] for c in r["cands"]}})
    return out


def conflicting_id(r) -> dict:
    """Same candidate ID, different physical sequence: the wrapper must raise; record what uf.instances does unguarded."""
    rr = copy.deepcopy(r)
    c = copy.deepcopy(next(x for x in rr["cands"] if x["seq"]))
    i, d = c["seq"][0]
    c["seq"][0] = (i, tuple(float(v) + 0.08 for v in d))
    rr["cands"].append(c)
    try:
        check_identity(rr)
        guarded = "NOT_RAISED"
    except jl.IdentityError as e:
        guarded = f"FAIL_CLOSED: {e}"
    return {"root": r["key"], "wrapper": guarded, "unguarded_uf_instances": "silently keeps the first block for a repeated ID (dedup by string)"}


# ------------------------------------------------------------------ instrumentation, structural multiplicity, plan, audit, gate
PHYS = ("planner", "skill", "motion", "ordered", "verify", "evaluate", "geometry", "teacher", "operational_teacher", "sector_cert")


def instrument() -> collections.Counter:
    """Wrap every function of every loaded physical-checking module with a call counter (expected to stay at 0)."""
    import types
    calls = collections.Counter()
    for name in PHYS:
        mod = sys.modules.get(name)
        for k, f in list(vars(mod).items()) if mod else []:
            if isinstance(f, types.FunctionType) and f.__module__ == name:
                def w(*a, _f=f, _n=f"{name}.{k}", **kw):
                    calls[_n] += 1
                    return _f(*a, **kw)
                setattr(mod, k, w)
    return calls


def multiplicity(roots) -> dict:
    """Per distinct sequence query in the unique model: unary instances (always 1) and relational instances by role / scope."""
    h = collections.defaultdict(collections.Counter)
    for r in roots:
        ks = sum(s[:2] in ("z0", "zS") for s in r["cands"][0]["slots"])
        per = collections.defaultdict(collections.Counter)
        for c in {c["query"]: c for c in r["cands"] if c["seq"]}.values():
            qs = [(s, q) for s, q in zip(c["slots"], c["z_q"]) if s[:2] in ("zR", "zO")]
            for s, q in qs:
                per[q][f"scene-query pairs as {s[:2]}"] += ks
                per[q][f"blocks as {s[:2]}"] += 1
                per[q]["higher-order scopes (B-full/C-full)"] += 1
                per[q]["zR-zO pairs"] += (len(qs) - 1) if s == "zR" else 1
                per[q]["zO-zO pairs"] += max(len(qs) - 2, 0) if s != "zR" else 0
        for q, cnt in per.items():
            for k, v in cnt.items():
                h[k][v] += 1
    return {k: dict(sorted(v.items())) for k, v in h.items()}


def _arm(arm):
    rec = l3._rd(f"runs/p9/l3/l3b/runs/{ARMS[arm]}.json")
    return l3.load(arm, CK / f"ckpt_{ARMS[arm]}.pt"), rec["eval"]["root"]["tau"], rec["eval"]["root"]["threshold"]


def plan_l3c0() -> None:
    train, val, test = l3.data()
    multi = [r["key"] for r in val if len(r["cands"]) > 1]
    alias = [r["key"] for r in train if any(s.startswith("zO") and q not in {c["query"] for c in r["cands"]}
                                            for c in r["cands"] for s, q in zip(c["slots"], c["z_q"]))][:6]
    st = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.rstrip().splitlines()
    head = subprocess.run(["git", "rev-parse", "HEAD", "origin/main"], capture_output=True, text=True).stdout.split()
    p = {"version": "P9_L3C0_PLAN_V1", "head": head[0], "origin_main": head[1], "tracked_changes": [x for x in st if not x.startswith("??")],
         "sources": {f: l3.sha(f) for f in SOURCES}, "datasets": {"dev": l3.sha(l3.DEV), "test": l3.sha(l3.TEST)},
         "checkpoints": {a: {"file": f"ckpt_{v}.pt", "sha256": l3.sha(CK / f"ckpt_{v}.pt")} for a, v in ARMS.items()},
         "l3b_plan_sha256": l3.sha("runs/p9/l3/l3b/L3B_plan.json"), "p9_l3_gates": {g: l3._rd(f"runs/p9/l3/{g}.json")["verdict"] for g in ("L3A_0_GATE", "L3A_1_GATE", "L3B_GATE")},
         "ontology_roots": multi + [next(r["key"] for r in val if len(r["cands"]) == 1)], "transformation_roots": multi + alias,
         "selector": "root-prior admission p_cal >= validation threshold, then (moves, J_pred, C_d, energy, id); tau / threshold from each checkpoint's L3b validation record",
         "transformations": ["permutation", "renamed_ids", "exact_duplicate", "disconnected_candidate", "alias_as_new_candidate (+ per-family decomposition)",
                             "related_candidate", "conflicting_id (must fail closed)"], "no_new_training": True, "teacher_calls": 0}
    p["hash"] = cf.stable_hash(p)
    OUT.mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "L3C_0_PLAN.json", p)


def audit() -> None:
    calls = instrument()
    plan = l3._rd(OUT / "L3C_0_PLAN.json")
    train, val, test = l3.data()
    by = {r["key"]: r for r in train + val + test}
    ident = collections.Counter()
    for r in by.values():
        try:
            check_identity(r)
            ident["roots_ok"] += 1
        except jl.IdentityError:
            ident["roots_failed"] += 1
    a = OUT / "l3c0"
    a.mkdir(parents=True, exist_ok=True)
    m, tau, thr = _arm("B-full")
    cf.write_json(a / "ontology.json", [ontology(m, by[k]) for k in plan["ontology_roots"]])
    cf.write_json(a / "multiplicity_unique.json", multiplicity(list(by.values())))
    rows, xchk = {}, {}
    import p8_select as ps                                      # cross-check only (not a learned path)
    rec = l3._rd("runs/p8/collection/decisions.json") | l3._rd("runs/p8/collection/sealed_test.json")
    for arm in ARMS:
        m, tau, thr = _arm(arm)
        rows[arm] = [x for k in plan["transformation_roots"] for x in transform_rows(m, by[k], tau, thr)]
        agree = []
        for r in val:
            ch, rr = deployed(m, r, uf.instances(m, r), tau, thr)
            f = ps.funnel([{"root": r["key"], "cands": rr}], rec, Path("runs/p8/collection"), tau, thr)
            agree.append(not f or f[0]["chosen"] == ch)
        xchk[arm] = {"roots": len(agree), "deployed_equals_p8_funnel_choice": all(agree), "tau": tau, "threshold": thr}
    m, tau, thr = _arm("B-full")
    scr = copy.deepcopy(by[plan["ontology_roots"][0]])
    scr["y"] = {q: None for q in scr["y"]}
    for c in scr["cands"]:
        c |= {"y": None, "z": [None] * len(c["z"]), "src": None, "cost": None}
    iso = max(abs(uf.prior_marginals(uf.instances(m, scr))[q] - p) for q, p in uf.prior_marginals(uf.instances(m, by[scr["key"]])).items())
    cf.write_json(a / "transformations.json", rows)
    cf.write_json(a / "summary.json", {"identity": dict(ident), "conflicting_id": [conflicting_id(by[k]) for k in plan["ontology_roots"][:3]],
                                       "selector_crosscheck": xchk, "label_isolation_max_prior_change": iso,
                                       "physical_module_calls": dict(calls), "physical_modules_loaded": sorted(k for k in PHYS if k in sys.modules)})


def gate_l3c0() -> dict:
    a = OUT / "l3c0"
    plan, s, rows = l3._rd(OUT / "L3C_0_PLAN.json"), l3._rd(a / "summary.json"), l3._rd(a / "transformations.json")
    inv = {}
    for arm, rs in rows.items():
        for x in rs:
            if x["expected"] == "EXPECTED_INVARIANT":
                k = f"{arm}|{x['case']}"
                among_existing = x["deployed_choice_changed"] and not x["chose_new_unlabelled_candidate"]   # a NEW option may be chosen
                inv[k] = max(inv.get(k, 0.0), x["max_abs_prior_change"], abs(x["existing_label_log_evidence_change"]), float(among_existing))
    checks = {"plan_frozen_before_audit": (OUT / "L3C_0_PLAN.json").stat().st_mtime < (a / "summary.json").stat().st_mtime,
              "sources_unchanged_since_plan": all(l3.sha(f) == h for f, h in plan["sources"].items()),
              "datasets_and_checkpoints_verified": all(l3.sha(CK / v["file"]) == v["sha256"] for v in plan["checkpoints"].values()),
              "p9_l3_gates_pass": all(v == "PASS" for v in plan["p9_l3_gates"].values()),
              "canonical_query_identity_all_roots": s["identity"].get("roots_failed", 0) == 0,
              "conflicting_id_fails_closed": all(x["wrapper"].startswith("FAIL_CLOSED") for x in s["conflicting_id"]),
              "unique_invariances_exact_incl_selection": all(v < 1e-9 for v in inv.values()) and len(inv) >= 12,
              "scene_prior_never_moves": all(x["scene_prior_change"] < 1e-12 for rs in rows.values() for x in rs),
              "deployed_selector_matches_p8_funnel": all(v["deployed_equals_p8_funnel_choice"] for v in s["selector_crosscheck"].values()),
              "prior_label_isolation": s["label_isolation_max_prior_change"] == 0.0,
              "zero_physical_module_calls": sum(s["physical_module_calls"].values()) == 0,
              "ontology_written": (a / "ontology.json").exists()}
    new_opt = {f"{arm}|{x['case']}|{x['root']}": "chose the newly added unlabelled candidate" for arm, rs in rows.items() for x in rs
               if x["expected"] == "EXPECTED_INVARIANT" and x["deployed_choice_changed"]}
    g = {"stage": "P9-L3c-0", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "invariance_max": inv,
         "selection_changes_to_new_option_in_invariant_cases": new_opt,
         "criterion_note": "first gate run counted ANY changed choice (FAIL); the invariant is: no switch among PRE-EXISTING candidates",
         "plan_hash": plan["hash"], "cpu_s_aggregate": round(spent()[0], 1)}
    cf.write_json(OUT / "L3C_0_GATE.json", g)
    return g



# ------------------------------------------------------------------ L3c-1: keyed competitor at FROZEN checkpoints (no retraining)
def pytest_file(name, path) -> None:
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", path], capture_output=True, text=True)
    (OUT / f"{name}.txt").write_text(r.stdout[-6000:] + r.stderr[-2000:] + f"\nEXIT {r.returncode}\n")


@torch.no_grad()
def keyed_artifacts() -> None:
    import unique_factor_keyed as ufk
    calls = instrument()
    plan = l3._rd(OUT / "L3C_0_PLAN.json")
    train, val, test = l3.data()
    by = {r["key"]: r for r in train + val + test}
    out, rows = {}, {}
    for arm in ARMS:
        m, tau, thr = _arm(arm)
        d, ll, bf, t_u, t_k = [], [], [], 0.0, 0.0
        jl.PEAK[0] = 0
        for r in val + test:
            c0 = time.process_time()
            iu = uf.instances(m, r)
            pu = uf.prior_marginals(iu)
            c1 = time.process_time()
            ik = ufk.instances(m, r)
            pk = uf.prior_marginals(ik)
            t_u, t_k = t_u + c1 - c0, t_k + time.process_time() - c1
            d += [abs(pk[q] - pu[q]) for q in pu]
            ll.append(float(uf.log_evidence(ik, jl.labels(r))[0] - uf.log_evidence(iu, jl.labels(r))[0]))
            if len(uf.variables(ik)) <= 9:
                bf.append(abs(float(uf.log_evidence(ik, jl.labels(r))[0]) - uf.brute_force(ik, jl.labels(r))))
        with ufk.installed():
            rr = [x for k in plan["transformation_roots"] for x in transform_rows(m, by[k], tau, thr, decompose=False)]
        for x in rr:                                           # keyed expectation: invariant iff no new physical outcome is modelled
            if x["case"] == "alias_as_new_candidate":
                x["expected"] = "EXPECTED_INVARIANT" if x["new_variables"] == 0 else "ALLOWED_CHANGE"
        rows[arm] = rr
        out[arm] = {"frozen_checkpoint": ARMS[arm], "keyed_minus_unique_prior": {"max": max(d), "mean": float(np.mean(d)), "n": len(d)},
                    "keyed_minus_unique_log_evidence": {"min": min(ll), "max": max(ll), "mean": float(np.mean(ll))},
                    "brute_force_roots": len(bf), "brute_force_max_abs_diff": max(bf) if bf else None, "peak_scope": jl.PEAK[0],
                    "cpu_s_per_root": {"unique": t_u / len(val + test), "keyed": t_k / len(val + test)},
                    "parameters": sum(p.numel() for p in m.parameters())}
    a = OUT / "l3c1"
    a.mkdir(parents=True, exist_ok=True)
    cf.write_json(a / "transformations_keyed.json", rows)
    cf.write_json(a / "artifacts.json", out | {"physical_module_calls": dict(calls)})


def gate_l3c1() -> dict:
    a = OUT / "l3c1"
    art, rows, tests = l3._rd(a / "artifacts.json"), l3._rd(a / "transformations_keyed.json"), (OUT / "l3c_tests.txt").read_text()
    inv = {}
    for arm, rs in rows.items():
        for x in rs:
            if x["expected"] == "EXPECTED_INVARIANT":
                k = f"{arm}|{x['case']}"
                among = x["deployed_choice_changed"] and not x["chose_new_unlabelled_candidate"]
                inv[k] = max(inv.get(k, 0.0), x["max_abs_prior_change"], abs(x["existing_label_log_evidence_change"]), float(among))
    arms = [k for k in art if k in ARMS]
    checks = {"l3c0_pass": l3._rd(OUT / "L3C_0_GATE.json")["verdict"] == "PASS",
              "tests_pass": "EXIT 0" in tests and " failed" not in tests,
              "brute_force_exact": all(art[k]["brute_force_roots"] >= 3 and art[k]["brute_force_max_abs_diff"] < 1e-9 for k in arms),
              "scene_prior_never_moves": all(x["scene_prior_change"] < 1e-12 for rs in rows.values() for x in rs),
              "keyed_invariant_in_every_expected_case_incl_alias": all(v < 1e-9 for v in inv.values())
                                                                 and all(any(k.startswith(f"{arm}|alias") for k in inv) for arm in arms),
              "no_new_parameters": len({art[k]["parameters"] for k in ("B-full", "C-full")}) <= 2,
              "complexity_within_cap": all(art[k]["peak_scope"] <= jl.MAX_SCOPE for k in arms),
              "zero_physical_module_calls": sum(art["physical_module_calls"].values()) == 0}
    g = {"stage": "P9-L3c-1", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "model_version": "ABC9K_KEYED_V1",
         "invariance_max": inv, "frozen_checkpoint_differences": {k: art[k] for k in arms}, "cpu_s_aggregate": round(spent()[0], 1)}
    cf.write_json(OUT / "L3C_1_GATE.json", g)
    return g

if __name__ == "__main__":
    if sys.argv[1] == "job":
        job(sys.argv[2], sys.argv[3], float(sys.argv[4]), *(sys.argv[5:6]))
    else:
        print(json.dumps(globals()[sys.argv[1]](), default=str)[:3000])
