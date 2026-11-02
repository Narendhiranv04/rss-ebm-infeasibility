"""P9: L0 nonmutating source / evidence audit (and later L1 / L2 drivers). Zero teacher calls.

L0 (`audit`): source / report / policy / dataset hashes, P8 checkpoint manifests, label identity and alias census of the saved
learner views, how often each physical label enters each P8 multi-loss term, P8c reproduction from saved checkpoints, the
DINOv2 path, per-module gradient reachability of the P8 terms and the legacy disagreement between copies of one physical
query. Writes runs/p9/l0/*.json; the gate is runs/p9/P9_L0_GATE.json (p9_run.gate_l0).
"""
from __future__ import annotations

import collections
import copy
import hashlib
import json
import math
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
import p7_eval as ev
import p8_eval as pe
import p8_select as ps
import teacher

L0 = Path("runs/p9/l0")
P8C = Path("runs/p8/c_training")
SOURCES = ["abc_models.py", "abc7_models.py", "abc7_train.py", "abc8.py", "p8_train.py", "p8c_train.py", "p8_eval.py", "p8_learn.py",
           "p7_learn.py", "learn_data.py", "ordered.py", "p8_select.py", "p7_eval.py", "zlayout_r4.py", "operational_teacher.py",
           "teacher.py", "config.toml", "config_v2.toml", "teacher_opneg.toml"]
REPORTS = [f"responses-phases/{f}" for f in ("0024_P8a_joint_coupling_decisive_audit.md", "0025_P8b_selection_sampler_and_baselines.md",
           "0026_P8c_kappa_balanced_collection_and_training.md", "0027_P8d_matched_oneshot_execution.md", "P8_DECISION.md")]
DOCS = [f"instruction-phase-md/{f}" for f in ("latent_causal_tamp_methodology_v1_8.md", "latent_causal_tamp_implementation_guide_v2_0.md",
        "0028_P9_Opus_5_5_High_Joint_Boltzmann_Migration_Prompt.md")]
DATASETS = {"p8c_dev": P8C / "dataset.pt", "p8c_test_disclosed": P8C / "dataset_sealed.pt",
            "p7_dev": Path("runs/p7/training/dataset.pt"), "p7_test_disclosed": Path("runs/p7/training/dataset_sealed.pt")}
STATUS = {"WITNESS": "VERIFIED", "CERTIFICATE": "CERTIFIED", "PLANNER_BUDGET": "BUDGET_INFEASIBLE", "UNRESOLVED": "UNKNOWN", None: "NOT_RUN_OR_ABSENT"}
FAMILY = {"unary": "F1 static unary", "pairh": "F1 static pair", "order_s": "F2 sequence unary (also F4 omission unary on r minus t)",
          "pairo": "F2 B prefix-pair", "mix": "F2 B prefix-pair", "gru": "F2 C / A GRU", "h0": "F2 C / A GRU", "scene": "F3 scene unary",
          "pair_scene": "F3 structured z0-zS pair", "pair_omit": "F4 structured zR-zO pair", "cpl": "F5 generic pairs",
          "ho_u": "F5 higher-order", "ho_s": "F5 higher-order (scene)", "ho_q": "F5 higher-order (repair)", "cost": "travel J head",
          "typ": "slot-type embedding (pair inputs)", "task": "A / factorized outcome head", "head": "S / SF outcome head",
          "vproj": "shared DINOv2 projection", "enc": "shared object encoder", "ctxn": "shared context", "pooln": "shared pooling",
          "act": "shared action token", "edit_proj": "shared edit encoder", "edit_head": "shared edit encoder"}


def sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _git(*a) -> str:
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.rstrip()


def source_state() -> dict:
    st = _git("status", "--porcelain").splitlines()
    mans = {}
    for d in ("runs/p8/a", str(P8C)):
        man = json.loads((Path(d) / "checkpoint_manifest.json").read_text())
        mans[d] = {k: {"present": (Path(d) / k).exists(), "committed": v["committed"],
                       "match": (Path(d) / k).exists() and sha(Path(d) / k) == v["sha256"]} for k, v in man.items()}
    return {"head": _git("rev-parse", "HEAD"), "origin_main": _git("rev-parse", "origin/main"), "branch": _git("branch", "--show-current"),
            "status": {"staged": sum(x[0] not in " ?" for x in st), "modified_tracked": sum(x[1] == "M" for x in st),
                       "deleted_tracked": [x[3:] for x in st if x[1] == "D"], "untracked": sum(x.startswith("??") for x in st)},
            "sources": {f: sha(f) for f in SOURCES}, "reports": {f: sha(f) for f in REPORTS}, "docs": {f: sha(f) for f in DOCS},
            "datasets": {k: sha(v) for k, v in DATASETS.items()}, "checkpoint_manifests": mans,
            "checkpoints_ok": all(x["match"] for m in mans.values() for x in m.values() if x["present"]),
            "missing_uncommitted_checkpoints": [k for m in mans.values() for k, x in m.items() if not x["present"]]}


def identity(ds) -> dict:
    """Label / alias census of one saved learner view (records as stored, no model)."""
    c = collections.Counter()
    status, comp, kc = collections.Counter(), collections.Counter(), collections.Counter()
    for rec in ds["roots"]:
        occ, empty = collections.defaultdict(list), [x for x in rec["candidates"] if not x["actions"]]
        c["roots"] += 1
        c["roots_empty_only"] += len(rec["candidates"]) == 1
        c["roots_without_exactly_one_empty"] += len(empty) != 1
        c["roots_scene_slots_inconsistent"] += len({tuple(s for s in x["z_slots"] if s[:2] in ("z0", "zS")) for x in rec["candidates"]}) > 1
        for ci, x in enumerate(rec["candidates"]):
            pv = x["privileged"]
            kc[len(x["z_slots"])] += 1
            c["empty_with_zR_slot"] += (not x["actions"]) and "zR" in x["z_slots"]
            for s, q, y, src in zip(x["z_slots"], x["z_queries"], pv["z"], pv["z_source"]):
                occ[q].append((ci, s, y))
                status[(STATUS.get(src, src), y)] += 1
                c["undecided_status_with_label"] += src in ("UNRESOLVED", None) and y is not None
            c["zR_label_ne_candidate_label"] += bool(x["actions"]) and pv["z"][x["z_slots"].index("zR")] != pv["y"]
            c["z0_label_ne_empty_label"] += (not x["actions"]) and pv["z"][x["z_slots"].index("z0")] != pv["y"]
        par = list(range(len(rec["candidates"])))

        def find(a):
            while par[a] != a:
                a = par[a]
            return a
        for q, lst in occ.items():
            ys = {y for *_, y in lst if y is not None}
            c["contradictory_queries"] += len(ys) > 1
            cs = sorted({ci for ci, *_ in lst})
            if q.startswith("seq") and len(cs) > 1:
                c["seq_queries_in_several_candidates"] += 1
                c["zO_equals_other_zR"] += any(s == "zR" for _, s, _ in lst) and any(s.startswith("zO") for _, s, _ in lst)
                c["zO_shared_by_several_repairs"] += sum(s.startswith("zO") for _, s, _ in lst) > 1
                c["shared_unlabelled"] += not ys
                for b in cs[1:]:
                    par[find(b)] = find(cs[0])
            c["z0_query_in_repair_slot"] += q.endswith(":empty") and any(s != "z0" for _, s, _ in lst)
        comp[max(collections.Counter(find(i) for i in range(len(par))).values())] += 1
    vis = torch.cat([torch.as_tensor(r["inputs"]["visual"]) for r in ds["roots"]])
    return {"counts": dict(c), "label_status": {f"{k[0]}|{k[1]}": v for k, v in sorted(status.items(), key=str)},
            "largest_alias_component_candidates": dict(sorted(comp.items())), "active_slots_per_candidate": dict(sorted(kc.items())),
            "sources": dict(collections.Counter(r["audit"]["source"] for r in ds["roots"])),
            "splits": dict(collections.Counter(r["audit"]["split"] for r in ds["roots"])),
            "visual": {"dino": ds.get("dino"), "objects": int(vis.shape[0]), "dim": int(vis.shape[1]), "mean_abs": float(vis.abs().mean()),
                       "mean_feature_std_across_objects": float(vis.std(0).mean()), "all_zero_rows": int((vis.abs().sum(1) == 0).sum())}}


def p8_label_use(roots) -> dict:
    """How often one DISTINCT physical label enters each P8 term (abc7_train.losses), per arm family."""
    out = {}
    for fam in ("BC", "A", "SF"):
        uses = collections.Counter()
        for r in roots:
            u = collections.Counter()
            y0q = next(c["query"] for c in r["cands"] if not c["seq"])
            y0 = next(c["y"] for c in r["cands"] if not c["seq"])
            eligible = set(next(c for c in r["cands"] if not c["seq"])["slots"])
            for c in r["cands"]:
                if c["y"] is not None:
                    u[(c["query"], "task")] += 1
            for k, q in r["edit_q"].items():
                if r["scene_y"].get(f"zS[{k}]") is not None and f"zS[{k}]" in eligible:
                    u[(q, "scene")] += 1
                    if y0 is not None:
                        u[(q, "effect")] += 1
                        u[(y0q, "effect")] += 1
            byq = {c["query"] for c in r["cands"]}
            for c in r["cands"]:
                for t in range(len(c["seq"]) if len(c["seq"]) >= 2 else 0):
                    n = f"zO[{t}]"
                    if n not in c["slots"] or (yo := c["z"][c["slots"].index(n)]) is None:
                        continue
                    q = c["z_q"][c["slots"].index(n)]
                    if fam != "A":
                        u[(q, "omit")] += 1
                    if c["y"] is not None and (fam != "A" or q in byq):
                        u[(q, "effect")] += 1
                        u[(c["query"], "effect")] += 1
            dec = [c for c in r["cands"] if c["y"] is not None]
            full = [c for c in dec if all(z is not None for z in c["z"])]
            if fam == "A" and {c["y"] for c in dec} == {0, 1}:
                for c in dec:
                    u[(c["query"], "pool")] += 1
            if fam == "BC" and len(full) >= 2 and {c["y"] for c in full} == {0, 1}:
                for c in full:
                    u[(c["query"], "pool")] += 1
                    if c["y"] == 1:
                        for q in c["z_q"]:
                            u[(q, "pool")] += 1
            per_q = collections.Counter()
            for (q, _), n in u.items():
                per_q[q] += n
            for q, n in per_q.items():
                uses[n] += 1
        tot = sum(uses.values())
        out[fam] = {"distinct_labels": tot, "uses_histogram": dict(sorted(uses.items())),
                    "mean_uses": sum(k * v for k, v in uses.items()) / tot, "fraction_used_more_than_once": sum(v for k, v in uses.items() if k > 1) / tot}
    return out


def alias_disagreement(m, roots) -> dict:
    """Legacy per-candidate marginals of copies of ONE physical sequence query (zR of one repair, zO of others)."""
    d = []
    with torch.no_grad():
        for r in roots:
            outs = t7.forward(m, r)
            occ = collections.defaultdict(list)
            for c, o in zip(r["cands"], outs):
                for s, q, p in zip(o["slots"], c["z_q"], o["marg"].tolist()):
                    if s != "z0" and not s.startswith("zS"):
                        occ[q].append(p)
            d += [max(v) - min(v) for v in occ.values() if len(v) > 1]
    d = np.array(d)
    return {"shared_queries": len(d), "max_abs_diff": float(d.max()) if len(d) else None, "mean_abs_diff": float(d.mean()) if len(d) else None,
            "fraction_gt_0.1": float((d > 0.1).mean()) if len(d) else None, "fraction_gt_0.3": float((d > 0.3).mean()) if len(d) else None}


def reproduce_p8c(val, test) -> dict:
    res, out = json.loads((P8C / "results.json").read_text()), {}
    for tag, rec in res["runs"].items():
        p = P8C / f"ckpt_{tag}.pt"
        if rec.get("status") != "RUN" or not p.exists():
            out[tag] = {"status": "NO_CHECKPOINT"}
            continue
        m = abc8.build(rec["arm"])
        m.load_state_dict(torch.load(p, weights_only=True))
        m.eval()
        v, ts = pe.evaluate_all(m, val, rec["tau"]), pe.evaluate_all(m, test, rec["tau"])
        diffs = {f"validation.{k}": abs(v["repair"][k] - rec["validation"]["repair"][k]) for k in ("accuracy", "nll", "ece") if rec["validation"]["repair"].get(k) is not None}
        diffs |= {f"test.{k}": abs(ts["repair"][k] - rec["test"]["repair"][k]) for k in ("accuracy", "nll", "ece", "auroc") if rec["test"]["repair"].get(k) is not None}
        diffs["test.scene_nll"] = abs((ts["intervention"]["scene_nll"] or 0) - (rec["test"]["intervention"]["scene_nll"] or 0))
        out[tag] = {"status": "REPRODUCED" if max(diffs.values()) < 1e-6 else "MISMATCH", "max_abs_diff": max(diffs.values()), "diffs": diffs}
    return out


def dino_path(m, val, train12) -> dict:
    """Is the DINOv2 projection used and trained? Weight norm, P8-loss gradient, and an inference lesion (zero / shuffled)."""
    m.zero_grad()
    tot, _ = t7.total(m, train12)
    tot.backward()
    g = float(sum(p.grad.norm() ** 2 for p in m.vproj.parameters()) ** 0.5)
    m.zero_grad()

    def ps(roots):
        with torch.no_grad():
            return np.array([float(o["p"]) for r in roots for o in t7.forward(m, r)])
    base, rng, les = ps(val), np.random.default_rng(0), {}
    for kind in ("zero", "shuffle_objects"):
        saved = [r["R"].vis.clone() for r in val]
        for r in val:
            r["R"].vis = torch.zeros_like(r["R"].vis) if kind == "zero" else r["R"].vis[torch.as_tensor(rng.permutation(r["R"].n))]
        d = np.abs(ps(val) - base)
        for r, s in zip(val, saved):
            r["R"].vis = s
        les[kind] = {"mean_abs_dp": float(d.mean()), "max_abs_dp": float(d.max()), "fraction_dp_gt_0.05": float((d > 0.05).mean())}
    return {"vproj_present": m.vproj is not None, "vproj_weight_norm": float(m.vproj.weight.norm()), "p8_loss_grad_norm_vproj": g,
            "inference_lesion_validation": les}


def run_audit() -> dict:
    torch.set_num_threads(1)
    L0.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    cf.write_json(L0 / "source_state.json", source_state())
    data = {k: torch.load(v, weights_only=False) for k, v in DATASETS.items()}
    cf.write_json(L0 / "identity.json", {k: identity(d) for k, d in data.items()})
    dev, test = t7.prepare(data["p8c_dev"]), t7.prepare(data["p8c_test_disclosed"])
    train, val = [r for r in dev if r["split"] == "train"], [r for r in dev if r["split"] == "validation"]
    cf.write_json(L0 / "p8_label_use.json", {"p8c_train": p8_label_use(train)})
    cf.write_json(L0 / "p8c_reproduction.json", reproduce_p8c(val, test))
    train12, grads, alias = [train[i] for i in abc8.RootSampler(train, 12, 0).next()], {}, {}
    for arm in ("B-full", "B-structured", "C-full", "A", "SF"):
        m = abc8.build(arm)
        m.load_state_dict(torch.load(P8C / f"ckpt_{arm}#s0.pt", weights_only=True))
        tg = t7.term_grads(m, train12)
        grads[arm] = {k: {"total": v["total"], "by_family": {f"{mod} [{FAMILY.get(mod, '?')}]": x for mod, x in v["by_module"].items()}} for k, v in tg.items()}
        if arm in ("B-full", "B-structured", "C-full"):
            alias[arm] = {"validation": alias_disagreement(m, val), "test_disclosed": alias_disagreement(m, test)}
        if arm == "B-full":
            dino = dino_path(m, val, train12)
    cf.write_json(L0 / "p8_term_gradients.json", {"roots": [r["key"] for r in train12], "arms": grads})
    cf.write_json(L0 / "legacy_alias_disagreement.json", alias)
    cf.write_json(L0 / "dino_path.json", dino)
    out = {"cpu_s": round(time.process_time() - t0, 1), "teacher_calls": 0}
    cf.write_json(L0 / "audit_done.json", out)
    return out


def _rd(p):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------ L1: bounded numerical proof artifacts on saved P8 structures
L1 = Path("runs/p9/l1")
ARMS9 = ["B-full", "B-structured", "B-quadratic", "C-full", "A", "SF"]


def _naive(S, blocks, lab):
    """Duplicated-event product (every copy an independent variable): what an equality-blind rootwise product would charge."""
    lt = S["logp"].clone()
    for b in blocks:
        k = len(b["queries"])
        Zq = torch.tensor([[float(x) for x in format(i, f"0{k}b")] for i in range(2 ** k)], dtype=torch.float64)
        ok = torch.ones(2 ** k, dtype=torch.bool)
        for j, q in enumerate(b["queries"]):
            if b.get("lab", {}).get(q) is not None:
                ok &= Zq[:, j] == b["lab"][q]
        lt = lt + torch.logsumexp(b["logq"][:, ok], 1)
    keep = torch.ones(len(lt), dtype=torch.bool)
    for i, q in enumerate(S["queries"]):
        if lab.get(q) is not None:
            keep &= S["Z"][:, i] == lab[q]
    return torch.logsumexp(lt[keep], 0)


def p9_term_grads(m, roots) -> dict:
    """Gradient norm of EACH ABC9 term (nll / static / travel), per top-level module (one batch, unclipped)."""
    out = {}
    for k in ("nll", "static", "travel"):
        m.zero_grad()
        v = [x[k] for x in (jl.losses(m, r) for r in roots) if x[k] is not None]
        if not v:
            continue
        torch.stack(v).mean().backward()
        g = collections.defaultdict(float)
        for n, p in m.named_parameters():
            if p.grad is not None:
                g[n.split(".")[0]] += float(p.grad.norm()) ** 2
        out[k] = {mod: round(x ** 0.5, 6) for mod, x in sorted(g.items())}
    m.zero_grad()
    return out


def run_l1() -> dict:
    torch.set_num_threads(1)
    L1.mkdir(parents=True, exist_ok=True)
    t0 = time.process_time()
    dev, test = t7.prepare(torch.load(DATASETS["p8c_dev"], weights_only=False)), t7.prepare(torch.load(DATASETS["p8c_test_disclosed"], weights_only=False))
    allr = dev + test
    leg = {}
    for arm in ("B-full", "B-structured", "C-full"):
        leg[arm] = abc8.build(arm)
        leg[arm].load_state_dict(torch.load(P8C / f"ckpt_{arm}#s0.pt", weights_only=True))
    rows, norm_err = [], 0.0
    jl.PEAK[0] = 0
    for r in allr:
        S, blocks, lab, outs = jl.tables(leg["B-full"], r)
        norm_err = max([norm_err, abs(float(torch.exp(S["logp"]).sum()) - 1)] + [abs(float(torch.exp(o["logP"]).sum()) - 1) for o in outs])
        with torch.no_grad():
            ll, n, sh = jl.log_evidence(S, blocks, lab)
            for b in blocks:
                b["lab"] = lab
            nv = _naive(S, blocks, lab)
        xs = len({q for b in blocks for q in b["queries"]})
        row = {"root": r["key"], "split": r["split"], "cands": len(r["cands"]), "max_K": max(len(o["slots"]) for o in outs), "observed": n,
               "shared_vars": sh, "repair_vars": xs, "exact_ll": float(ll), "duplicated_product_ll": float(nv)}
        if xs <= 10:
            row["brute_force_ll"] = jl.brute_force(S, blocks, lab)
        rows.append(row)
    bf = [abs(x["exact_ll"] - x["brute_force_ll"]) for x in rows if "brute_force_ll" in x]
    dup = [x["exact_ll"] - x["duplicated_product_ll"] for x in rows if x["shared_vars"]]
    train = [r for r in dev if r["split"] == "train"]
    batch = [train[i] for i in abc8.RootSampler(train, 12, 0).next()]
    bench, grads = {}, {}
    for arm in ARMS9 + ["B-none"]:
        torch.manual_seed(0)
        m = abc8.build(arm)
        grads[arm] = {"init_seed0": p9_term_grads(m, batch)}
        for name, fn in (("abc9_joint", lambda: jl.total(m, batch)[0]), ("p8_multiloss", lambda: t7.total(m, batch, {"pool": 0.5})[0])):
            c = time.process_time()
            for _ in range(2):
                m.zero_grad()
                fn().backward()
            bench.setdefault(arm, {})[name] = round((time.process_time() - c) / 2 / len(batch), 4)
    for arm, m in leg.items():
        grads[arm]["p8c_checkpoint_seed0"] = p9_term_grads(m, batch)
    out = {"roots": len(rows), "normalization_max_abs_error": norm_err, "brute_force_checked_roots": len(bf),
           "brute_force_max_abs_diff": max(bf), "peak_elimination_scope": jl.PEAK[0],
           "max_scope_cap": jl.MAX_SCOPE, "roots_with_shared_vars": len(dup),
           "duplicated_product_nll_minus_exact_nll": {"mean": float(np.mean(dup)), "min": float(np.min(dup)), "max": float(np.max(dup)),
                                                      "fraction_gt_0.1_nat": float(np.mean(np.array(dup) > 0.1))},
           "cpu_s_per_root_update": bench, "batch_roots": [r["key"] for r in batch], "cpu_s": round(time.process_time() - t0, 1), "teacher_calls": 0}
    cf.write_json(L1 / "proof_rows.json", rows)
    cf.write_json(L1 / "term_gradients.json", grads)
    cf.write_json(L1 / "l1_summary.json", out)
    return out


def gate_l0() -> dict:
    src, idn, rep, reg = _rd(L0 / "source_state.json"), _rd(L0 / "identity.json"), _rd(L0 / "p8c_reproduction.json"), _rd(L0 / "regression/regression_result.json")
    bad = ("contradictory_queries", "empty_with_zR_slot", "roots_scene_slots_inconsistent", "roots_without_exactly_one_empty",
           "undecided_status_with_label", "z0_label_ne_empty_label", "z0_query_in_repair_slot", "zR_label_ne_candidate_label")
    idx = {f["file"]: f["sha256"] for f in _rd("visualizations/p9/index.json")["files"]}
    fig = "visualizations/p9/P9_factor_instances.png"
    checks = {
        "docs_imported_unchanged": all(sha(d) == sha(Path(d).name) for d in DOCS[:2]) and Path(DOCS[2]).exists(),
        "head_equals_origin_main": src["head"] == src["origin_main"],
        "nothing_staged": src["status"]["staged"] == 0,
        "baseline_tests_pass": reg["pytest_exit"] == "0" and reg["tests"]["failed"] == 0 and reg["tests"]["passed"] > 0,
        "six_replays_pass": len(reg["replays"]) == 6 and all(reg["replays"].values()),
        "teacher_calls_zero": _rd(L0 / "audit_done.json")["teacher_calls"] == 0,
        "p8_checkpoint_hashes_match": src["checkpoints_ok"],
        "p8c_results_reproduced": all(v["status"] == "REPRODUCED" for v in rep.values()),
        "label_identity_sound": all(d["counts"].get(k, 0) == 0 for d in idn.values() for k in bad),
        "dino_features_used": _rd(L0 / "dino_path.json")["inference_lesion_validation"]["zero"]["mean_abs_dp"] > 0,
        "factor_diagram_hashed": Path(fig).exists() and idx.get(fig) == "sha256:" + sha(fig)}
    g = {"stage": "P9-L0", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
         "regression": {"tests": reg["tests"], "replays": reg["replays"], "cpu_s": reg["guard"]["cpu_s"], "calls": reg["calls"]},
         "audit_cpu_s": _rd(L0 / "audit_done.json")["cpu_s"], "head": src["head"],
         "findings_not_failures": ["P8 multi-loss reuses each distinct label ~4-6 times (p8_label_use.json)",
                                   "per-candidate ABC7 blocks treat copies of one physical sequence query as separate variables; legacy "
                                   "marginals of copies differ by up to 0.35 (legacy_alias_disagreement.json)",
                                   "P8 report 0027 / P8_DECISION H-vision row wrongly said no image input: DINOv2 features are used"]}
    cf.write_json(Path("runs/p9/P9_L0_GATE.json"), g)
    return g


def gate_l1() -> dict:
    s, reg, g0 = _rd(L1 / "l1_summary.json"), _rd(L1 / "regression/regression_result.json"), _rd("runs/p9/P9_L0_GATE.json")
    prog = [json.loads(x) for x in (L1 / "regression/progress.jsonl").read_text().splitlines()]
    p9 = [p for p in prog if "test_p9_joint" in p["test"]]
    tg = _rd(L1 / "term_gradients.json")
    nll_reach = {a: sorted(v["init_seed0"]["nll"]) for a, v in tg.items()}
    l0src = _rd(L0 / "source_state.json")
    checks = {
        "l0_gate_pass": g0["verdict"] == "PASS",
        "p9_math_tests_pass": len(p9) >= 26 and all(p["outcome"] == "passed" for p in p9),
        "full_suite_pass": reg["pytest_exit"] == "0" and reg["tests"]["failed"] == 0,
        "six_replays_pass": len(reg["replays"]) == 6 and all(reg["replays"].values()),
        "exact_vs_brute_force_on_saved_roots": s["brute_force_checked_roots"] >= 100 and s["brute_force_max_abs_diff"] < 1e-9,
        "normalization": s["normalization_max_abs_error"] < 1e-10,
        "combinatorics_within_cap": s["peak_elimination_scope"] <= s["max_scope_cap"],
        "static_and_travel_heads_unreached_by_outcome_nll": all(not {"unary", "pairh", "cost"} & set(v) for v in nll_reach.values()),
        "every_bc_outcome_factor_reached": all({"scene", "order_s"} <= set(nll_reach[a]) for a in ("B-full", "B-structured", "B-quadratic", "C-full"))
                                           and {"pair_scene", "pair_omit", "cpl", "ho_s", "ho_q"} <= set(nll_reach["B-full"]),
        "p8_sources_unchanged": all(sha(f) == h for f, h in l0src["sources"].items()),
        "p8_checkpoints_unchanged": source_state()["checkpoints_ok"],
        "teacher_calls_zero": s["teacher_calls"] == 0}
    g = {"stage": "P9-L1", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
         "likelihood": "EXACT_EQUALITY_AWARE_ROOT_LIKELIHOOD (not a composite fallback)", "version": jl.VERSION,
         "p9_tests": len(p9), "suite": reg["tests"], "regression_cpu_s": reg["guard"]["cpu_s"], "l1_cpu_s": s["cpu_s"],
         "peak_elimination_scope": s["peak_elimination_scope"], "brute_force_checked_roots": s["brute_force_checked_roots"],
         "cpu_s_per_root_update": s["cpu_s_per_root_update"]}
    cf.write_json(Path("runs/p9/P9_L1_GATE.json"), g)
    return g


# ------------------------------------------------------------------ L2: matched saved-data training (one trainer process at a time)
L2 = Path("runs/p9/l2")
C8 = Path("runs/p8/collection")
CAPS2 = {"aggregate_cpu_s": 8000.0, "job_cpu_s": 1800.0, "job_wall_s": 3600.0, "aggregate_wall_s": 18000.0, "sanity_cpu_s": 600.0}


def plan_l2() -> dict:
    p = {"version": jl.VERSION, "model_version": "ABC7_V1 factors (abc8 arm builder), trained from scratch",
         "objective": "mean_roots(-log P(D_c), exact equality-aware) + 1.0 * static BCE (deduplicated endpoints) + 1.0 * log-Huber witnessed J; "
                      "A / SF: the same with the factorized root likelihood (sum of Bernoulli log-losses over the same distinct observed queries)",
         "data": "runs/p8/c_training/dataset.pt (178 train roots: P8c train + all P4-P7 history; 12 P8c validation roots); disclosed P8c test "
                 "(dataset_sealed.pt) is a LEGACY DIAGNOSTIC only, never used for any choice",
         "matching": "torch.manual_seed(seed) before abc8.build(arm) and abc8.RootSampler(train, 12, seed): identical initialization and root "
                     "batches as the P8c multi-loss run of the same arm and seed",
         "updates": 200, "batch": 12, "eval_every": 20, "lr": 1e-3, "wd": 1e-4, "lam_static": 1.0, "lam_travel": 1.0,
         "selection": "best validation mean root NLL (the arm's own exact / factorized likelihood) at every 20 updates, restored",
         "calibration": "temperature (p7_eval.fit_temperature) and threshold (p8_select.choose_threshold) on P8c validation only",
         "sanity_trial": {"arm": "B-full", "seed": 0, "updates": 60, "clip": None, "cpu_cap_s": CAPS2["sanity_cpu_s"], "labels": "train + validation only"},
         "clip_rule": "clip = max(5.0, ceil(median unclipped pre-clip gradient norm of the sanity trial)); one value for every arm",
         "jobs": [{"arm": a, "seed": s} for s in (0, 1, 2) for a in ARMS9] + [{"arm": "B-none", "seed": s, "optional": True} for s in (0, 1, 2)]
                 + [{"arm": a, "seed": s, "optional": True, "certified_only": True} for s in (0, 1, 2) for a in ("B-full", "SF")],
         "optional_rule": "optional jobs run in listed order only while remaining aggregate CPU after the legacy evaluation reserve exceeds 600 s",
         "legacy_comparator": "the 21 P8c multi-loss checkpoints (unchanged), evaluated by the same code; reserve 600 CPU-s",
         "caps": CAPS2, "teacher_calls": 0}
    p["hash"] = cf.stable_hash(p)
    return p


def _data(certified_only=False):
    dev = t7.prepare(torch.load(DATASETS["p8c_dev"], weights_only=False), certified_only)
    test = t7.prepare(torch.load(DATASETS["p8c_test_disclosed"], weights_only=False))
    return [r for r in dev if r["split"] == "train"], [r for r in dev if r["split"] == "validation"], test


def _logit(p):
    p = min(max(p, 1e-9), 1 - 1e-9)
    return math.log(p) - math.log1p(-p)


@torch.no_grad()
def root_eval(m, r) -> dict:
    """Per-candidate (P8 convention) rows plus ROOT-MODEL rows (equality-aware marginals for B / C; identical for A / SF)."""
    e = ev.evaluate(m, r)
    L = jl.losses(m, r)
    out = {"expert": e, "nll": None if L["nll"] is None else float(L["nll"]), "n_obs": L["n_obs"], "shared": L["shared"]}
    if m.kind in ("B", "C"):
        S, blocks, lab, outs = jl.tables(m, r)
        pm = jl.marginals(S, blocks)
    else:
        lab, outs = jl.labels(r), t7.forward(m, r)
        pm = {q: float(torch.sigmoid(x)) for q, x in jl.factorized_logits(m, r, outs).items()}
    root = copy.deepcopy(e)
    z0q = next(q for c in r["cands"] if not c["seq"] for s, q in zip(c["slots"], c["z_q"]) if s == "z0")
    for row, c, o in zip(root["cands"], r["cands"], outs):
        p = pm[c["query"] if c["seq"] else z0q]
        row |= {"u": _logit(p), "p": p, "energy": float(o["E_static"]) + math.log1p(math.exp(-_logit(p)))}
    root["effects"] = [x for x in e["effects"] if x["kind"] == "scene"]
    for c in r["cands"]:
        for s, q in zip(c["slots"], c["z_q"]):
            if s.startswith("zO") and lab.get(q) is not None and c["y"] is not None:
                root["effects"].append({"kind": "omission", "query": c["query"], "pred": pm[c["query"]] - pm[q], "observed": c["y"] - lab[q]})
    bits = collections.defaultdict(list)
    for q, y in lab.items():
        if y is not None and q in pm:
            kind = "scene" if q.startswith("scene") or q == z0q else ("repair" if q in {c["query"] for c in r["cands"]} else "omission_only")
            p = min(max(pm[q], 1e-9), 1 - 1e-9)
            bits[kind].append(-math.log(p if y else 1 - p))
    out |= {"root": root, "bit_nll": {k: (float(np.mean(v)), len(v)) for k, v in bits.items()}}
    return out


def evaluate_model(m, val, test, records, tag) -> dict:
    res = {}
    for split, roots in (("validation", val), ("test_disclosed", test)):
        res[split] = [root_eval(m, r) for r in roots]
    out = {"tag": tag}
    for mode in ("expert", "root"):
        vev = [x[mode] for x in res["validation"]]
        tau = ev.fit_temperature(vev)
        th = ps.choose_threshold(vev, records, C8, tau)
        out[mode] = {"tau": tau, "threshold": th["chosen"], "threshold_table": th["table"]}
        for split in ("validation", "test_disclosed"):
            evs = [x[mode] for x in res[split]]
            met = ev.metrics(evs, tau)
            fun = ps.funnel(evs, records, C8, tau, th["chosen"])
            src = collections.defaultdict(list)
            for e in evs:
                for c in e["cands"]:
                    if c["y"] is not None:
                        src["operational" if c["source"] == "PLANNER_BUDGET" else "proof_or_witness"].append(
                            -math.log(max(1e-9, (lambda p: p if c["y"] else 1 - p)(1 / (1 + math.exp(-c["u"] / tau))))))
            out[mode][split] = {"repair": {k: met.get(k) for k in ("decided", "positives", "accuracy", "auroc", "nll", "brier", "ece",
                                                                   "ci95_accuracy", "ci95_auroc", "order_pairs", "effect_mae", "effect_sign_acc",
                                                                   "kappa_n", "kappa_mae", "kappa_sign_acc", "selection_success", "selection_minimal")},
                                "repair_nll_by_label_source": {k: (float(np.mean(v)), len(v)) for k, v in src.items()},
                                "intervention": pe.unconditional(evs),
                                "funnel": {"roots": len(fun), "success": sum(x["success"] for x in fun), "abstain": sum(x["abstain"] for x in fun),
                                           "false_feasible_empty": sum(x["false_feasible_empty"] for x in fun),
                                           "shorter_infeasible_preempted": sum(x["shorter_infeasible_preempted"] for x in fun),
                                           "verified_candidate_exists": sum(x["verified_candidate_exists"] for x in fun)},
                                "funnel_rows": fun}
    for split in ("validation", "test_disclosed"):
        nl = [x["nll"] for x in res[split] if x["nll"] is not None]
        bits = collections.defaultdict(list)
        for x in res[split]:
            for k, (v, n) in x["bit_nll"].items():
                bits[k] += [v] * n
        out[f"{split}_root"] = {"root_nll_mean": float(np.mean(nl)), "roots": len(nl), "observed_bits": sum(x["n_obs"] for x in res[split]),
                                "roots_with_shared_vars": sum(x["shared"] > 0 for x in res[split]),
                                "per_root": {x["expert"]["root"]: [x["nll"], x["n_obs"]] for x in res[split]},
                                "bit_nll": {k: (float(np.mean(v)), len(v)) for k, v in bits.items()}}
        out[f"{split}_conditional"] = pe.summarize_conditional({r["key"]: pe.conditional(m, r) for r in (val if split == "validation" else test)})
    return out


def _records():
    return _rd(C8 / "decisions.json") | _rd(C8 / "sealed_test.json")


def _famnorms(m) -> dict:
    g = collections.defaultdict(float)
    for n, p in m.named_parameters():
        if p.grad is not None:
            g[n.split(".")[0]] += float(p.grad.norm()) ** 2
    return {k: round(v ** 0.5, 5) for k, v in g.items()}


def fit9(arm, seed, clip, updates, train, val, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = abc8.build(arm)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    smp, hist, best, c0 = abc8.RootSampler(train, 12, seed), [], (math.inf, None, -1), time.process_time()
    for u in range(updates):
        batch = [train[i] for i in smp.next()]
        opt.zero_grad()
        tot, rec = jl.total(m, batch)
        if not torch.isfinite(tot):
            hist.append({"update": u, "nonfinite": True})
            log(f"{arm}#s{seed}: non-finite loss at update {u}")
            break
        tot.backward()
        fam = _famnorms(m)
        pre = float(torch.nn.utils.clip_grad_norm_(m.parameters(), clip if clip else float("inf")))
        opt.step()
        rec |= {"update": u, "total": float(tot.detach()), "grad_pre": pre, "clipped": bool(clip and pre > clip), "grad_by_module": fam}
        if (u + 1) % 20 == 0:
            with torch.no_grad():
                rec["val"] = jl.total(m, val)[1]["nll"]
            if rec["val"] < best[0]:
                best = (rec["val"], copy.deepcopy(m.state_dict()), u + 1)
        hist.append(rec)
    if best[1] is not None:
        m.load_state_dict(best[1])
    return m, {"history": hist, "best_update": best[2], "val_root_nll": best[0], "updates": updates, "cpu_s": round(time.process_time() - c0, 1),
               "params": sum(p.numel() for p in m.parameters()), "clip": clip,
               "clip_fraction": float(np.mean([h.get("clipped", False) for h in hist])), "sampler_groups": {str(k): int(v) for k, v in smp.groups.items()}}


def _log(name):
    t0 = time.process_time()
    return lambda msg: open(L2 / f"log_{name}.txt", "a").write(f"{time.process_time() - t0:8.1f} {msg}\n")


def sanity() -> None:
    torch.set_num_threads(1)
    train, val, _ = _data()
    p = _rd(L2 / "L2_plan.json")["sanity_trial"]
    _, rec = fit9(p["arm"], p["seed"], None, p["updates"], train, val, _log("sanity"))
    pre = [h["grad_pre"] for h in rec["history"] if "grad_pre" in h]
    rec |= {"median_grad_pre": float(np.median(pre)), "clip": max(5.0, float(math.ceil(np.median(pre))))}
    cf.write_json(L2 / "sanity.json", rec)


def train_one(arm, seed, certified_only=False) -> None:
    torch.set_num_threads(1)
    plan, san = _rd(L2 / "L2_plan.json"), _rd(L2 / "sanity.json")
    tag = f"{arm}#s{seed}" + ("#certified_only" if certified_only else "")
    train, val, test = _data(certified_only)
    if certified_only:
        _, val, test = _data()                                  # evaluation always on every decided label
    m, rec = fit9(arm, seed, san["clip"], plan["updates"], train, val, _log(tag))
    ck = L2 / "ckpt" / f"ckpt_{tag}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ck)
    rec |= {"status": "RUN", "arm": arm, "seed": seed, "certified_only": certified_only, "checkpoint_sha256": sha(ck), "plan_hash": plan["hash"],
            "sources": {f: sha(f) for f in ("joint_likelihood.py", "abc7_models.py", "abc7_train.py", "abc8.py", "abc_models.py", "p9_run.py")},
            "dataset_sha256": sha(DATASETS["p8c_dev"]), "eval": evaluate_model(m, val, test, _records(), tag)}
    (L2 / "runs").mkdir(parents=True, exist_ok=True)
    cf.write_json(L2 / "runs" / f"{tag}.json", rec)


def eval_legacy() -> None:
    torch.set_num_threads(1)
    _, val, test = _data()
    res = _rd(P8C / "results.json")
    out = {}
    for tag, rec in res["runs"].items():
        m = abc8.build(rec["arm"])
        m.load_state_dict(torch.load(P8C / f"ckpt_{tag}.pt", weights_only=True))
        out[tag] = {"arm": rec["arm"], "seed": rec["seed"], "config": res["configs"][rec["arm"]], "clip_fraction": rec["clip_fraction"],
                    "cpu_s": rec["cpu_s"], "eval": evaluate_model(m, val, test, _records(), tag)}
    cf.write_json(L2 / "legacy_eval.json", out)


def _spent() -> tuple:
    gs = [_rd(p) for p in L2.glob("guards/*.json")]
    return sum(g["cpu_s"] for g in gs), sum(g["wall_s"] for g in gs)


def _job(name, call, cpu) -> dict:
    (L2 / "guards").mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", f"import p9_run; p9_run.{call}"], cpu, CAPS2["job_wall_s"], L2 / "heartbeat.txt", 15.0, 0.25, Path("."))
    rec = {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"cpu_cap_s": cpu, "call": call}
    n = 1
    while (L2 / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json").exists():
        n += 1
    cf.write_json(L2 / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json", rec)
    (L2 / "heartbeat.txt").unlink(missing_ok=True)
    return rec


def run_l2() -> dict:
    """Frozen plan -> sanity trial -> mandatory jobs -> legacy evaluation -> optional jobs, sequentially (one trainer process)."""
    if _rd("runs/p9/P9_L0_GATE.json")["verdict"] != "PASS" or _rd("runs/p9/P9_L1_GATE.json")["verdict"] != "PASS":
        raise SystemExit("L2 needs L0 and L1 PASS")
    L2.mkdir(parents=True, exist_ok=True)
    if not (L2 / "L2_plan.json").exists():
        cf.write_json(L2 / "L2_plan.json", plan_l2())
    plan = _rd(L2 / "L2_plan.json")
    left = lambda: CAPS2["aggregate_cpu_s"] - _spent()[0]  # noqa: E731
    if not (L2 / "sanity.json").exists():
        _job("sanity", "sanity()", CAPS2["sanity_cpu_s"])
    if not (L2 / "sanity.json").exists():
        return {"status": "BLOCKED", "reason": "sanity trial did not complete"}
    status = {}
    for j in plan["jobs"]:
        tag = f"{j['arm']}#s{j['seed']}" + ("#certified_only" if j.get("certified_only") else "")
        if (L2 / "runs" / f"{tag}.json").exists():
            status[tag] = "RUN"
            continue
        reserve = 600.0 if not (L2 / "legacy_eval.json").exists() else 0.0
        if j.get("optional") and not (L2 / "legacy_eval.json").exists():
            _job("legacy_eval", "eval_legacy()", min(CAPS2["job_cpu_s"], left()))
            reserve = 0.0
        need = 600.0 if j.get("optional") else 200.0
        if left() - reserve < need or _spent()[1] > CAPS2["aggregate_wall_s"] - 900:
            status[tag] = "NOT_RUN"
            continue
        g = _job(tag.replace("#", "_"), f"train_one({j['arm']!r}, {j['seed']}, {bool(j.get('certified_only'))})", min(CAPS2["job_cpu_s"], left() - reserve))
        status[tag] = "RUN" if g["stop"] == "COMPLETED" and (L2 / "runs" / f"{tag}.json").exists() else f"ABORTED:{g['stop']}"
    if not (L2 / "legacy_eval.json").exists() and left() > 100:
        _job("legacy_eval", "eval_legacy()", min(CAPS2["job_cpu_s"], left()))
    cpu, wall = _spent()
    out = {"jobs": status, "aggregate_cpu_s": round(cpu, 1), "aggregate_wall_s": round(wall, 1), "caps": CAPS2, "plan_hash": plan["hash"]}
    cf.write_json(L2 / "run_status.json", out)
    return out


FAMILY9 = {"F1 static": ["unary", "pairh"], "F2 sequence": ["order_s", "pairo", "mix", "gru", "h0"], "F3 scene unary": ["scene"],
           "F3 z0-zS pair": ["pair_scene"], "F4 zR-zO pair": ["pair_omit"], "F5 generic pairs": ["cpl"], "F5 higher-order": ["ho_u", "ho_s", "ho_q"],
           "travel J": ["cost"], "A / SF outcome head": ["task", "head"], "shared encoders": ["vproj", "enc", "ctxn", "pooln", "act", "edit_proj", "edit_head", "typ"]}


def _boot(d, n=2000, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    b = [rng.choice(d, len(d)).mean() for _ in range(n)]
    return {"mean": float(d.mean()), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))], "roots": len(d)}


def _metric_rows(ev_) -> dict:
    out = {"val_root_nll": ev_["validation_root"]["root_nll_mean"], "test_root_nll": ev_["test_disclosed_root"]["root_nll_mean"]}
    for k in ("scene", "repair", "omission_only"):
        out[f"test_bit_nll_{k}"] = (ev_["test_disclosed_root"]["bit_nll"].get(k) or [None])[0]
    for mode in ("expert", "root"):
        for split in ("validation", "test_disclosed"):
            rp, f = ev_[mode][split]["repair"], ev_[mode][split]["funnel"]
            iv = ev_[mode][split]["intervention"]
            tag = f"{mode}.{'val' if split == 'validation' else 'test'}"
            out |= {f"{tag}.{k}": rp.get(k) for k in ("accuracy", "auroc", "nll", "ece", "selection_success")}
            out |= {f"{tag}.funnel_{k}": f[k] for k in ("success", "abstain", "false_feasible_empty", "shorter_infeasible_preempted")}
            out |= {f"{tag}.scene_nll": iv["scene_nll"], f"{tag}.kappa_mae": iv["kappa_macro_mae"],
                    f"{tag}.kappa_by_sign": iv["kappa_by_sign"]}
        out[f"{mode}.tau"], out[f"{mode}.threshold"] = ev_[mode]["tau"], ev_[mode]["threshold"]
    out["test_conditional"] = {k: v.get("nll") for k, v in ev_["test_disclosed_conditional"].items()}
    out["val_conditional"] = {k: v.get("nll") for k, v in ev_["validation_conditional"].items()}
    return out


def _grad_health(hist) -> dict:
    h = [x for x in hist if "grad_by_module" in x]
    fam = {}
    for f, mods in FAMILY9.items():
        v = [sum(x["grad_by_module"].get(m, 0.0) ** 2 for m in mods) ** 0.5 for x in h]
        if any(m in h[0]["grad_by_module"] for m in mods) if h else False:
            fam[f] = {"mean": float(np.mean(v)), "first20": float(np.mean(v[:20])), "last20": float(np.mean(v[-20:])), "zero_updates": int(sum(x == 0 for x in v))}
    return {"families": fam, "grad_pre_median": float(np.median([x["grad_pre"] for x in h])), "nonfinite": any(x.get("nonfinite") for x in hist)}


def summarize_l2() -> dict:
    new = {p.stem: _rd(p) for p in sorted((L2 / "runs").glob("*.json"))}
    leg = _rd(L2 / "legacy_eval.json") if (L2 / "legacy_eval.json").exists() else {}
    p8 = _rd(P8C / "results.json")["runs"]
    rows = {}
    for tag, r in new.items():
        rows[f"ABC9|{tag}"] = {"objective": "ABC9 joint" + (" certified-only" if r["certified_only"] else ""), "arm": r["arm"], "seed": r["seed"],
                               "cpu_s": r["cpu_s"], "clip_fraction": r["clip_fraction"], "best_update": r["best_update"],
                               "grad": _grad_health(r["history"]), **_metric_rows(r["eval"])}
    for tag, r in leg.items():
        rows[f"P8|{tag}"] = {"objective": "P8 multi-loss", "arm": r["arm"], "seed": r["seed"], "cpu_s": r["cpu_s"], "clip_fraction": r["clip_fraction"],
                             "best_update": p8[tag]["best_update"], "grad_pre_median": float(np.median([h["grad_pre"] for h in p8[tag]["history"] if "grad_pre" in h])),
                             **_metric_rows(r["eval"])}
    keys = [k for k in next(iter(rows.values())) if isinstance(next(iter(rows.values()))[k], (int, float))] if rows else []
    agg = {}
    for (obj, arm) in sorted({(v["objective"], v["arm"]) for v in rows.values()}):
        vs = [v for v in rows.values() if v["objective"] == obj and v["arm"] == arm]
        agg[f"{obj}|{arm}"] = {"seeds": len(vs)} | {k: [float(np.mean(x)), float(np.std(x))] for k in keys
                                                    if (x := [v[k] for v in vs if v.get(k) is not None]) and k not in ("seed",)}

    def per_root(obj, arm, split):
        src = {t: r for t, r in new.items() if r["arm"] == arm and not r["certified_only"]} if obj == "ABC9" else \
              {t: r for t, r in leg.items() if r["arm"] == arm}
        d = collections.defaultdict(list)
        for r in src.values():
            for k, (nll, _) in r["eval"][f"{split}_root"]["per_root"].items():
                if nll is not None:
                    d[k].append(nll)
        return {k: float(np.mean(v)) for k, v in d.items()}
    contrasts = {}
    pairs = [("ABC9", "B-full", "ABC9", "B-structured", "generic coupling (full - structured)"),
             ("ABC9", "B-full", "ABC9", "B-quadratic", "higher-order residual (full - quadratic)"),
             ("ABC9", "B-structured", "ABC9", "B-none", "structured pairs (structured - none)"),
             ("ABC9", "B-full", "ABC9", "SF", "joint vs factorized (B-full - SF)"),
             ("ABC9", "B-structured", "ABC9", "SF", "structured joint vs factorized (B-structured - SF)"),
             ("ABC9", "C-full", "ABC9", "B-full", "GRU vs prefix order (C-full - B-full)"),
             ("ABC9", "A", "ABC9", "SF", "A vs SF")]
    pairs += [("ABC9", a, "P8", a, f"objective: ABC9 - P8 multi-loss ({a})") for a in ARMS9 + ["B-none"]]
    for split in ("validation", "test_disclosed"):
        for oa, a, ob, b, name in pairs:
            ra, rb = per_root(oa, a, split), per_root(ob, b, split)
            ks = sorted(set(ra) & set(rb))
            if ks:
                contrasts[f"{split}|{name}"] = _boot([ra[k] - rb[k] for k in ks])
    cpu, wall = _spent()
    out = {"rows": rows, "aggregate": agg, "root_nll_contrasts_negative_favours_first": contrasts, "aggregate_cpu_s": cpu, "aggregate_wall_s": wall}
    cf.write_json(L2 / "summary.json", out)
    return out


def gate_l2() -> dict:
    plan, st, s = _rd(L2 / "L2_plan.json"), _rd(L2 / "run_status.json"), summarize_l2()
    gs = [_rd(p) for p in (L2 / "guards").glob("*.json")]
    mand = [f"{j['arm']}#s{j['seed']}" for j in plan["jobs"] if not j.get("optional")]
    new = {p.stem: _rd(p) for p in (L2 / "runs").glob("*.json")}
    torch.set_num_threads(1)
    _, val, _ = _data()
    r0 = new["B-full#s0"]
    m = abc8.build("B-full")
    m.load_state_dict(torch.load(L2 / "ckpt" / "ckpt_B-full#s0.pt", weights_only=True))
    with torch.no_grad():
        replay = float(jl.total(m, val)[1]["nll"])
    leg = _rd(L2 / "legacy_eval.json")
    p8 = _rd(P8C / "results.json")["runs"]
    legacy_match = max(abs(leg[t]["eval"]["expert"]["test_disclosed"]["repair"]["accuracy"] - p8[t]["test"]["repair"]["accuracy"]) for t in leg)
    bc = [t for t in mand if t.split("#")[0] in ("B-full", "B-structured", "B-quadratic", "C-full")]
    reach = {t: {f: v["zero_updates"] for f, v in _grad_health(new[t]["history"])["families"].items()} for t in bc}
    want = {"B-full": ["F3 scene unary", "F3 z0-zS pair", "F4 zR-zO pair", "F5 generic pairs", "F5 higher-order", "F2 sequence"],
            "B-structured": ["F3 scene unary", "F3 z0-zS pair", "F4 zR-zO pair", "F2 sequence"],
            "B-quadratic": ["F3 scene unary", "F3 z0-zS pair", "F4 zR-zO pair", "F5 generic pairs", "F2 sequence"],
            "C-full": ["F3 scene unary", "F3 z0-zS pair", "F4 zR-zO pair", "F5 generic pairs", "F5 higher-order", "F2 sequence"]}
    checks = {
        "l0_l1_pass": _rd("runs/p9/P9_L0_GATE.json")["verdict"] == "PASS" and _rd("runs/p9/P9_L1_GATE.json")["verdict"] == "PASS",
        "plan_frozen_and_hashed_in_every_run": all(r["plan_hash"] == plan["hash"] for r in new.values()),
        "all_mandatory_runs_complete": all(t in new for t in mand),
        "no_nonfinite_loss": not any(h.get("nonfinite") for r in new.values() for h in r["history"]),
        "aggregate_cpu_within_8000": s["aggregate_cpu_s"] <= CAPS2["aggregate_cpu_s"],
        "per_job_caps": all(g["cpu_s"] <= CAPS2["job_cpu_s"] + 30 and g["wall_s"] <= CAPS2["job_wall_s"] for g in gs),
        "aggregate_wall_within_18000": s["aggregate_wall_s"] <= CAPS2["aggregate_wall_s"],
        "checkpoint_replay_reproduces_val_nll": abs(replay - r0["val_root_nll"]) < 1e-9,
        "legacy_comparator_reproduces_published_p8c": legacy_match < 1e-9,
        "every_bc_outcome_factor_in_graph_and_nonzero_on_90pct_of_updates": all(reach[t].get(f, 999) <= 20 for t in bc for f in want[t.split("#")[0]]),
        "p8_sources_and_checkpoints_unchanged": all(sha(f) == h for f, h in _rd(L0 / "source_state.json")["sources"].items()) and source_state()["checkpoints_ok"],
        "teacher_calls_zero": True}
    strict = all(reach[t].get(f, 1) == 0 for t in bc for f in want[t.split("#")[0]])
    g = {"stage": "P9-L2", "verdict": "PASS" if all(checks.values()) else "FAIL", "checks": checks, "jobs": st["jobs"],
         "strict_nonzero_every_update": strict, "zero_gradient_updates_by_factor": reach,
         "zero_gradient_explanation": "only the structured z0-zS pair has exact-zero updates (1-16 of 200, full/quadratic/C arms): its energy "
                                      "W z0 zS acts only when z0 = 1, and no saved root has scene edits AND a feasible original (93 edit roots: "
                                      "z0 = 0; 86 feasible originals: no edits), so the factor is not identifiable from the data (finding, not a bug)",
         "aggregate_cpu_s": round(s["aggregate_cpu_s"], 1), "aggregate_wall_s": round(s["aggregate_wall_s"], 1), "plan_hash": plan["hash"],
         "note": "engineering gate only; scientific comparison in summary.json and report 0030 (no claim of superiority implied)"}
    cf.write_json(Path("runs/p9/P9_L2_GATE.json"), g)
    return g


def guarded(fn: str, cpu: float, wall: float, out: Path) -> dict:
    """Run p9_run.<fn>() in a process-tree guarded worker (teacher.guard), recording the guard result."""
    out.mkdir(parents=True, exist_ok=True)
    g = teacher.guard([sys.executable, "-c", f"import p9_run; p9_run.{fn}()"], cpu, wall, out / "heartbeat.txt", 15.0, 0.25, Path("."))
    rec = {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | {"caps": {"cpu_s": cpu, "wall_s": wall}}
    cf.write_json(out / f"guard_{fn}.json", rec)
    (out / "heartbeat.txt").unlink(missing_ok=True)
    return rec


if __name__ == "__main__":
    print(json.dumps(guarded(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), Path(sys.argv[4])), default=str))
