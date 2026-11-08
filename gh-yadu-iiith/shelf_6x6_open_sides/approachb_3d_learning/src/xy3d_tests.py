"""BWL-2 checks of the 3D X/Y adapter (contract section 5 tests 1-10).  Unit tests: forward / backward only (no optimizer step), float64.
Test 8 (planted-coefficient recovery) is the only training here: GPU, SYNTHETIC labels, clearly labelled.

    N="taskset -c 7 nice -n 15 venv/bin/python"
    $N src/xy3d_tests.py unit --out reports/10_xy3d_unit_tests.json        tests 1-7, 9, 10 (+ no-op / connector-trap regressions)
    P="env CUDA_VISIBLE_DEVICES=0 taskset -c 7 nice -n 10 venv/bin/python"
    $P src/xy3d_tests.py synthetic --out reports/10_xy3d_synthetic.json    test 8 (SYNTHETIC planted D / E / DIRECT-GEO / G recovery)
Parity (test 10) imports the 2D source modules from parity_ref/ (git show e113c6680:{abc_models,abc7_models,bxy_core}.py; hashes checked)."""
from __future__ import annotations

import argparse, copy, hashlib, json, math, os, sys, time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xy3d as XY  # noqa: E402
import xy3d_data as XD  # noqa: E402
from data3d import cell_xy  # noqa: E402

torch.set_num_threads(1)
D64 = torch.float64
PARITY = {"abc_models.py": "7d8ca5de954d8fce", "abc7_models.py": "27b04b480610a7cb", "bxy_core.py": "f575bc03ffdff7be"}
FORBID = ("roles", "intended_blockers", "parking", "alt_parking", "construction", "greedy_blockers", "mechanism_pools", "empty_table_T_solve",
          "grasp_azimuths", "verified_K", "observed", "ledger_observed", "spec", "label_consistency", "independent_moves", "teacher_calls")


# ---------------------------------------------------------------- synthetic 3D-format roots (SYNTHETIC: never physical labels)
def synth_root(rng, key, n_extra=None):
    n = int(rng.integers(3, 7)) if n_extra is None else n_extra
    cells = [tuple(divmod(int(c), 6)) for c in rng.choice(36, n + 1, replace=False)]
    goal = cells.pop()
    tid = int(rng.integers(n))
    attrs = [[float(rng.uniform(0.012, 0.02)), float(rng.uniform(0.06, 0.18)), float(rng.integers(2)), float(i == tid), float(i != tid)] for i in range(n)]
    ctx = XD.REF + list(cell_xy(*goal)) + [attrs[tid][0], attrs[tid][1], 1.0, 1.0, 0.0]
    return {"key": key, "cells": cells, "goal": goal, "tid": tid, "attrs": attrs, "ctx": ctx}


def free_cells(cells):
    return [(r, c) for r in range(6) for c in range(6) if (r, c) not in set(cells)]


def synth_design(rng, s):
    """Y edits = single moves; X = singles, AB / BA pairs (same endpoint), one triple in two orders + its omissions."""
    mov = [i for i in range(len(s["cells"])) if i != s["tid"]]
    fr = free_cells(s["cells"])
    dest = {i: fr[int(k)] for i, k in zip(mov, rng.choice(len(fr), len(mov), replace=False))}
    xyd = {i: cell_xy(*d) for i, d in dest.items()}
    edits = [{i: xyd[i]} for i in mov[:2]]
    seqs = [[(i, xyd[i])] for i in mov[:3]]
    for a_, b_ in list(zip(mov, mov[1:]))[:2]:
        seqs += [[(a_, xyd[a_]), (b_, xyd[b_])], [(b_, xyd[b_]), (a_, xyd[a_])]]
    if len(mov) >= 3:
        A, B, C = mov[:3]
        seqs += [[(A, xyd[A]), (B, xyd[B]), (C, xyd[C])], [(C, xyd[C]), (B, xyd[B]), (A, xyd[A])], [(A, xyd[A]), (C, xyd[C])]]
    return edits, seqs


def synth_view(s, edits, seqs):
    return XY.make_view(s["key"], [cell_xy(*c) for c in s["cells"]], s["attrs"], s["tid"], cell_xy(*s["goal"]), (0.0, 0.0), s["ctx"], edits, seqs, XD.GEO3D)


def planted(v, coupled):
    """PLANTED SYNTHETIC truth on observable geometry: F3 from target / goal access, F2 from ideal-predecessor pick access + a pure ORDER term
    (tall-before-short), F5 w = 1.5 (coupled truth) or 0 (uncoupled truth)."""
    b = []
    for e in v["y_xy"]:
        f = XD.GEO3D.phi_y(v, e)
        b.append(3.0 * f[8] + 1.5 * f[9] - 1.5 * f[12] - 2.0)
    a = []
    for q in v["x_seqs"]:
        f = XD.GEO3D.phi_x(v, q, True)
        h = [v["attrs"][i][1] for i, _ in q]
        tall_first = float(all(x >= y for x, y in zip(h, h[1:]))) if len(h) > 1 else 0.5
        a.append(0.8 - 0.5 * len(q) + 2.5 * (f[0] - 0.5) - 1.0 * f[4] + 1.6 * (tall_first - 0.5))
    return np.array(b), np.array(a), (1.5 if coupled else 0.0)


def synth_set(n, seed, coupled):
    rng = np.random.default_rng(seed)
    views, labs, truth = [], [], []
    for r in range(n):
        s = synth_root(rng, f"syn{seed}-{r}")
        v, ym, xm = synth_view(s, *synth_design(rng, s))
        b, a, w = planted(v, coupled)
        y = (rng.random(len(b)) < 1 / (1 + np.exp(-b))).astype(int)
        x = (rng.random(len(a)) < 1 / (1 + np.exp(-(a + w * y[v["parent"]])))).astype(int)
        oy = {k: int(y[k]) for k in range(len(b)) if rng.random() < 0.8}
        ox = {k: int(x[k]) for k in range(len(a)) if rng.random() < 0.85}
        lab = XY.labels_for(v, list(range(len(b))), list(range(len(a))), obs_y=oy, obs_x=ox)
        views.append(v); labs.append(lab); truth.append((b, a, w))
    return views, labs, truth


# ---------------------------------------------------------------- unit tests (CPU, float64, no optimizer)
def nets(attr=5, ctx=14, vis=0, seed=0):
    out = {}
    for arm in XY.ARMS:
        torch.manual_seed(seed)
        out[arm] = XY.XYNet(arm, attr, ctx, vis).double()
    return out


def t1_normalization_and_gradients():
    r = XY.selftest()
    g = torch.Generator().manual_seed(3)
    worst = 0.0
    for _ in range(50):
        b, a, w = (torch.randn(1, generator=g, dtype=D64) * 3 for _ in range(3))
        tot = sum(math.exp(float(XY.log_py(b, y) + XY.log_px(a, w, x, y))) for x in (0, 1) for y in (0, 1))
        worst = max(worst, abs(tot - 1))
    b, a, w = (torch.randn(n, dtype=D64, requires_grad=True) for n in (2, 3, 3))
    gc = torch.autograd.gradcheck(lambda b, a, w: XY.log_evidence(b, a, w, [0, 0, 1], {0: 1}, {0: 1, 2: 0}), (b, a, w))
    rng = np.random.default_rng(1)
    views, labs, _ = synth_set(6, 1, True)
    for l in labs:
        l["travel"] = [(0, 1.3)] if l["obs_x"] else []
    B, L = XY.collate(views, labs)
    active = {}
    for arm, m in nets().items():
        loss, _ = m.loss(B, L)
        loss.backward()
        active[arm] = sorted(n for n, p in m.named_parameters() if p.grad is None or float(p.grad.abs().sum()) == 0)
    expected = {"DIRECT-GEO": {n for n, _ in M_named("DIRECT-GEO") if n.split(".")[0] in ("jt", "cost")}}      # source DIRECT: lambda_J = 0 (travel=False)
    ok_g = all(set(v) == expected.get(arm, set()) for arm, v in active.items())
    return {"pass": r["pass"] and worst < 1e-12 and gc and ok_g, "four_cell_sum_max_err": worst, "gradcheck_log_evidence": gc,
            "exact_math": r, "params_without_gradient": active, "expected_without_gradient": {k: sorted(v) for k, v in expected.items()}}


def M_named(arm):
    return list(XY.XYNet(arm, 5, 14).named_parameters())


def t2_shared_y_and_duplicates():
    rng = np.random.default_rng(2)
    s = synth_root(rng, "t2", 4)
    mov = [i for i in range(4) if i != s["tid"]]
    fr = free_cells(s["cells"])
    A, Bo = mov[:2]
    pa, pb = cell_xy(*fr[0]), cell_xy(*fr[1])
    ab, ba = [(A, pa), (Bo, pb)], [(Bo, pb), (A, pa)]
    v, ym, xm = synth_view(s, [{A: pa}], [ab, ba])
    vd, ymd, xmd = synth_view(s, [{A: pa}, {A: pa}], [ab, ba, ab])
    Bt, _ = XY.collate([v])
    out = {"same_parent_AB_BA": v["parent"][0] == v["parent"][1], "dedup_nodes": (len(vd["y_xy"]), len(vd["x_seqs"])) == (len(v["y_xy"]), len(v["x_seqs"])),
           "dup_maps": ymd == [ym[0], ym[0]] and xmd == [0, 1, 0]}
    M = nets()
    with torch.no_grad():
        out["a_AB_minus_BA"] = {arm: float((m.coeffs(Bt)["a"][0] - m.coeffs(Bt)["a"][1]).abs()) for arm, m in M.items()}
    l1 = XY.labels_for(v, ym, xm, obs_y={0: 1}, obs_x={0: 1, 1: 0})
    l2 = XY.labels_for(vd, ymd, xmd, obs_y={0: 1, 1: 1}, obs_x={0: 1, 1: 0, 2: 1})
    with torch.no_grad():
        out["dup_loss_equal"] = {arm: float((m.loss(*XY.collate([v], [l1]))[0] - m.loss(*XY.collate([vd], [l2]))[0]).abs()) for arm, m in M.items()}
    try:
        XY.labels_for(vd, ymd, xmd, obs_x={0: 1, 2: 0})
        out["conflict_raises"] = False
    except ValueError:
        out["conflict_raises"] = True
    ok = out["same_parent_AB_BA"] and out["dedup_nodes"] and out["dup_maps"] and out["conflict_raises"] and all(x < 1e-12 for x in out["dup_loss_equal"].values())
    ok = ok and out["a_AB_minus_BA"]["D"] > 1e-6 and out["a_AB_minus_BA"]["E"] > 1e-6 and out["a_AB_minus_BA"]["G"] < 1e-10
    return {"pass": ok, **out}


def t3_invariances():
    views, labs, _ = synth_set(5, 3, True)
    M = nets()
    res = {}
    with torch.no_grad():
        for arm, m in M.items():
            Ball, Lall = XY.collate(views, labs)
            Ca = m.coeffs(Ball)
            per = [m.coeffs(XY.collate([v])[0]) for v in views]
            cat = {k: torch.cat([p[k] for p in per]) for k in ("b", "a", "w", "J")}
            batch_err = max(float((Ca[k] - cat[k]).abs().max()) for k in cat)
            v0 = next(v for v in views if any(len(q) >= 3 for q in v["x_seqs"]))
            rv = copy.deepcopy(v0)
            order = list(range(len(v0["x_seqs"])))[::-1]
            for k in ("x_seqs", "phi_x", "phi_x0", "parent"):
                rv[k] = [v0[k][i] for i in order]
            c0, c1 = m.coeffs(XY.collate([v0])[0]), m.coeffs(XY.collate([rv])[0])
            perm_err = float((c0["a"][order] - c1["a"]).abs().max())
            cache_err = float((m.coeffs(Ball)["a"] - Ca["a"]).abs().max())
            ext = copy.deepcopy(v0)                                           # one more X query (a new order) on an existing endpoint
            qi = next(i for i, q in enumerate(v0["x_seqs"]) if len(q) >= 3)
            q = v0["x_seqs"][qi]
            extra = [q[1], q[0]] + q[2:]
            assert all(extra != x for x in v0["x_seqs"])
            ext["x_seqs"] = v0["x_seqs"] + [extra]; ext["parent"] = v0["parent"] + [v0["parent"][qi]]
            ext["phi_x"] = v0["phi_x"] + [XD.GEO3D.phi_x(v0, extra, True)]; ext["phi_x0"] = v0["phi_x0"] + [XD.GEO3D.phi_x(v0, extra, False)]
            pe, p0 = m.predict(XY.collate([ext])[0]), m.predict(XY.collate([v0])[0])
            key = "py" if "py" in p0 else "p_noop"
            y_err = float((pe[key] - p0[key]).abs().max())
            res[arm] = {"batch_vs_single_max_err": batch_err, "query_permutation_err": perm_err, "cache_err": cache_err, "y_marginal_change_after_extra_X": y_err}
    return {"pass": all(max(r.values()) < 1e-10 for r in res.values()), **res}


def t4_coupling():
    views, labs, _ = synth_set(4, 4, True)
    m = nets()["D"]
    with torch.no_grad():
        m.xy[2].weight.zero_(); m.xy[2].bias.zero_()
    B, L = XY.collate(views, labs)
    C = m.coeffs(B)
    p = m.predict(B)
    red = float((p["px"] - torch.sigmoid(C["a"])).abs().max())
    le_w0 = XY.batched_log_evidence(C["b"], C["a"], torch.zeros_like(C["a"]), B["parent"], L["y_obs"], L["x_obs"], B["s_root"], B["n_roots"])
    le = XY.batched_log_evidence(C["b"], C["a"], C["w"], B["parent"], L["y_obs"], L["x_obs"], B["s_root"], B["n_roots"])
    m2 = nets()["D"]
    v = views[0]
    joint = XY.labels_for(v, list(range(len(v["y_xy"]))), list(range(len(v["x_seqs"]))), obs_y={v["parent"][0]: 1}, obs_x={0: 1})
    yonly = XY.labels_for(v, list(range(len(v["y_xy"]))), list(range(len(v["x_seqs"]))), obs_y={s: 1 for s in range(len(v["y_xy"]))})
    g = {}
    for name, lab in (("joint_decided", joint), ("y_only", yonly)):
        m2.zero_grad()
        m2.loss(*XY.collate([v], [lab]))[0].backward()
        g[name] = {h: float(sum(p.grad.abs().sum() for p in getattr(m2, h).parameters() if p.grad is not None)) for h in ("xy", "order_s", "pairo", "mix")}
    ok = red < 1e-12 and float((le - le_w0).abs().max()) < 1e-12 and g["joint_decided"]["xy"] > 0 and all(x == 0 for x in g["y_only"].values())
    return {"pass": ok, "w0_px_equals_sigmoid_a_err": red, "w0_evidence_equals_uncoupled_err": float((le - le_w0).abs().max()), "grad_abs_sum": g}


def t5_marginal_loss_cases():
    rng = np.random.default_rng(5)
    s = synth_root(rng, "t5", 4)
    edits, seqs = synth_design(rng, s)
    v, ym, xm = synth_view(s, edits, seqs)
    m = nets()["D"]
    nX, nY = len(v["x_seqs"]), len(v["y_xy"])
    shared = max(set(v["parent"]), key=v["parent"].count)
    kids = [q for q in range(nX) if v["parent"][q] == shared]
    cases = {"neither": ({}, {}), "y_only": ({0: 1, v["parent"][0]: 0}, {}), "x_only": ({}, {0: 1, 1: 0}), "both": ({v["parent"][0]: 1}, {0: 0}),
             "multi_x_children_share_y": ({}, {q: int(i % 2) for i, q in enumerate(kids)})}
    out = {}
    with torch.no_grad():
        B, _ = XY.collate([v])
        C = m.coeffs(B)
        for name, (oy, ox) in cases.items():
            lab = XY.labels_for(v, list(range(nY)), list(range(nX)), obs_y=oy, obs_x=ox)
            Bb, L = XY.collate([v], [lab])
            mine = float(XY.batched_log_evidence(C["b"], C["a"], C["w"], Bb["parent"], L["y_obs"], L["x_obs"], Bb["s_root"], 1)[0])
            touched = sorted(set(oy) | {v["parent"][q] for q in ox})
            sub = {s_: i for i, s_ in enumerate(touched)}
            qs = [q for q in range(nX) if v["parent"][q] in sub and q in ox]
            ref = 0.0 if not touched else XY.brute_force(C["b"][touched], C["a"][qs], C["w"][qs], [sub[v["parent"][q]] for q in qs],
                                                         {sub[k]: val for k, val in oy.items()}, {i: ox[q] for i, q in enumerate(qs)})
            out[name] = {"batched": mine, "brute_force": ref, "err": abs(mine - ref), "children_sharing_y": len(kids) if name.startswith("multi") else None}
    return {"pass": all(o["err"] < 1e-10 for o in out.values()) and out["neither"]["batched"] == 0.0 and len(kids) >= 2, **out}


def t6_order_goal_permutation():
    rng = np.random.default_rng(6)
    s = synth_root(rng, "t6", 5)
    edits, seqs = synth_design(rng, s)
    v, _, _ = synth_view(s, edits, seqs)
    M = nets()
    out = {}
    s2 = copy.deepcopy(s)
    s2["goal"] = next(c for c in free_cells(s["cells"]) if c != s["goal"])
    s2["ctx"] = XD.REF + list(cell_xy(*s2["goal"])) + s["ctx"][9:]
    v2, _, _ = synth_view(s2, edits, seqs)
    perm = list(np.random.default_rng(0).permutation(len(s["cells"])))
    inv = {o: n for n, o in enumerate(perm)}
    s3 = {**s, "cells": [s["cells"][o] for o in perm], "attrs": [s["attrs"][o] for o in perm], "tid": inv[s["tid"]]}
    v3, _, _ = XY.make_view(s3["key"], [cell_xy(*c) for c in s3["cells"]], s3["attrs"], s3["tid"], cell_xy(*s3["goal"]), (0.0, 0.0), s3["ctx"],
                            [{inv[i]: p for i, p in e.items()} for e in edits], [[(inv[i], p) for i, p in q] for q in seqs], XD.GEO3D,
                            okey=[int(o) for o in perm])                     # stable ids travel with the objects
    pairs = [(i, j) for i in range(len(v["x_seqs"])) for j in range(i + 1, len(v["x_seqs"])) if v["parent"][i] == v["parent"][j]]
    with torch.no_grad():
        for arm, m in M.items():
            C, C2, C3 = (m.coeffs(XY.collate([x])[0]) for x in (v, v2, v3))
            out[arm] = {"same_endpoint_order_pairs": len(pairs), "max_a_order_diff": max(float((C["a"][i] - C["a"][j]).abs()) for i, j in pairs),
                        "b_shared_identical": all(v["parent"][i] == v["parent"][j] for i, j in pairs),
                        "goal_change_max_b_diff": float((C["b"] - C2["b"]).abs().max()),
                        "object_permutation_err": max(float((C[k] - C3[k]).abs().max()) for k in ("b", "a", "w", "J"))}
    ok = all(o["goal_change_max_b_diff"] > 1e-6 and o["object_permutation_err"] < 1e-10 and o["b_shared_identical"] for o in out.values())
    ok = ok and out["D"]["max_a_order_diff"] > 1e-6 and out["G"]["max_a_order_diff"] < 1e-10
    return {"pass": ok, **out}


def real_roots(n=24):
    roots = XD.qc_roots()
    out = []
    for K in (0, 1, 2, 3):
        ks = [r for r in sorted(roots, key=lambda r: hashlib.sha256(f"BWL2|{r}".encode()).hexdigest()) if XD.load(r)["verified_K"] == K]
        out += ks[: n // 4]
    return out


def t7_real_format_roundtrip():
    rows, M, views, labs = [], nets(), [], []
    for rid in real_roots():
        raw = XD.load(rid)
        edits, seqs = XD.design(raw)
        om = [[x for k, x in enumerate(q) if k != t] for q in seqs if len(q) >= 2 for t in range(len(q))]      # F4 omissions recomputed from S0
        v, ym, xm, ids = XD.view3d(raw, extra_seqs=om)
        lab = XD.labels3d(raw, v, ym, xm)
        cells0 = {o: tuple(c) for o, c in raw["initial_cells"].items()}
        okD = all(XY.sig_xy(v["y_xy"][ym[k]]) == XY.sig_xy([cell_xy(*({**cells0, **{o: tuple(c) for o, c in e.items()}})[o]) for o in ids])
                  and lab["obs_y"][ym[k]] == raw["labels"]["D"][k]["y"] for k, e in enumerate(edits))
        okY0 = lab["obs_y"][0] == raw["labels"]["original"]["y"] and XY.sig_xy(v["y_xy"][0]) == XY.sig_xy(v["xy0"])
        n0 = len(seqs)
        okO = all(XY.sig_xy(v["y_xy"][v["parent"][xm[n0 + j]]]) == XY.sig_xy([({**{ids.index(o): cell_xy(*c) for o, c in q}}).get(i, p) for i, p in enumerate(v["xy0"])])
                  for j, q in enumerate(om))
        rows.append({"root": rid, "K": raw["verified_K"], "y_nodes": len(v["y_xy"]), "x_nodes": len(v["x_seqs"]), "obs_y": len(lab["obs_y"]),
                     "obs_x": len(lab["obs_x"]), "no_op_only": len(v["x_seqs"]) == 0, "D_labels_on_their_endpoints": okD, "Y0_label": okY0,
                     "omission_endpoints_recomputed_from_S0": okO, "negatives": sum(1 for r in raw["labels"]["D"] if r["y"] == 0) + (raw["labels"]["original"]["y"] == 0)})
        syn_x = {q: int(hashlib.sha256(f"SYN|{rid}|{q}".encode()).digest()[0] % 2) for q in range(len(v["x_seqs"]))}             # SYNTHETIC X labels
        lab_syn = XY.labels_for(v, list(range(len(v["y_xy"]))), list(range(len(v["x_seqs"]))), obs_y=lab["obs_y"], obs_x=syn_x)
        views.append(v); labs.append(lab_syn)
    B, L = XY.collate(views, labs)
    fwd = {}
    for arm, m in M.items():
        m.zero_grad()
        loss, parts = m.loss(B, L)
        loss.backward()
        fwd[arm] = {"loss": float(loss), "finite": bool(torch.isfinite(loss)), "grad_finite": all(bool(torch.isfinite(p.grad).all()) for p in m.parameters() if p.grad is not None),
                    "parts": parts}
    real_x = sum(r["obs_x"] for r in rows)
    ok = all(r["D_labels_on_their_endpoints"] and r["Y0_label"] and r["omission_endpoints_recomputed_from_S0"] for r in rows) and real_x == 0
    ok = ok and all(f["finite"] and f["grad_finite"] for f in fwd.values()) and any(r["no_op_only"] for r in rows)
    return {"pass": ok, "roots": rows, "real_X_labels_attached": real_x, "forward_backward_with_SYNTHETIC_X": fwd}


def t9_direct_g_and_leakage():
    M = nets()
    caps = {arm: {"params": sum(p.numel() for p in m.parameters()), "heads": sorted({n.split(".")[0] for n, _ in m.named_parameters()})} for arm, m in M.items()}
    res = {}
    with torch.no_grad():
        for rid in real_roots(8):
            raw = XD.load(rid)
            v = XD.view3d(raw)[0]
            mut = copy.deepcopy(raw)
            for k in FORBID:
                if k in mut:
                    mut[k] = {"MUTATED": hashlib.sha256(k.encode()).hexdigest()} if isinstance(mut[k], dict) else "MUTATED"
            for r in [mut["labels"]["original"]] + mut["labels"]["D"] + mut["labels"]["B"]:
                r.update(y=1 - r["y"], negative_kind="MUTATED", witness="MUTATED", reason="MUTATED", failed_step="MUTATED", role="MUTATED")
            mut["labels"]["blocker"] = "MUTATED"
            vm = XD.view3d(mut)[0]
            for arm, m in M.items():
                c0, c1 = m.coeffs(XY.collate([v])[0]), m.coeffs(XY.collate([vm])[0])
                res.setdefault(arm, 0.0)
                res[arm] = max(res[arm], max(float((c0[k] - c1[k]).abs().max()) if len(c0[k]) else 0.0 for k in ("b", "a", "w", "J", "J0")))
    ok = all(x == 0.0 for x in res.values()) and "xy" not in caps["DIRECT-GEO"]["heads"] and "xs" in caps["DIRECT-GEO"]["heads"]
    ok = ok and "xy" in caps["G"]["heads"] and "xs" in caps["E"]["heads"] and "xy" not in caps["E"]["heads"]
    return {"pass": ok, "forbidden_field_mutation_max_coeff_change": res, "capacity": caps,
            "direct_success_logit": "a_r = order_s + xs + geoX + geoY_parent, trained by success-event BCE (independent of any b + a sum)"}


def t_noop_and_connector_trap():
    rng = np.random.default_rng(7)
    s = synth_root(rng, "noop", 4)
    v, ym, xm = synth_view(s, [], [])
    M = nets()
    out = {}
    with torch.no_grad():
        for arm, m in M.items():
            B, _ = XY.collate([v])
            p = m.predict(B)
            C = m.coeffs(B)
            out[arm] = {"x_queries": int(len(B["q_root"])), "p_noop_equals_sigmoid_b0": abs(float(p["p_noop"][0] - torch.sigmoid(C["b"][0]))) < 1e-12}
    rows = [{"admissible": True, "m": 0, "cd": 0.0, "sig": "empty", "p": 0.6, "J_hat": 0.0}, {"admissible": True, "m": 1, "cd": 1.0, "sig": "a", "p": 0.9, "J_hat": 0.0}]
    gate = XY.choose(rows, {"noop_gate": 0.5, "eta": 0.0, "alpha": 0.3, "beta": 0.0, "gamma": 0.0}) == 0
    nogate = XY.choose(rows, {"noop_gate": None, "eta": 0.0, "alpha": 0.3, "beta": 0.0, "gamma": 0.0}) == 1
    rows[1]["p"] = 0.55
    phantom = XY.choose(rows, {"noop_gate": None, "eta": 0.0, "alpha": 0.3, "beta": 0.0, "gamma": 0.0}) == 0     # no-op pays no action cost
    s2 = synth_root(rng, "trap", 4)
    mov = [i for i in range(4) if i != s2["tid"]]
    fr = free_cells(s2["cells"])
    A_, B_ = [(mov[0], cell_xy(*fr[0]))], [(mov[1], cell_xy(*fr[1]))]
    v2, ym2, xm2 = synth_view(s2, [], [A_, A_ + B_])
    lab = XY.labels_for(v2, ym2, xm2, obs_x={0: 0, 1: 1})                      # X_A = 0 (connector failed) but X_AB = 1: both kept, distinct
    loss = float(M["D"].loss(*XY.collate([v2], [lab]))[0])
    ok = all(o["x_queries"] == 0 and o["p_noop_equals_sigmoid_b0"] for o in out.values()) and gate and nogate and phantom and math.isfinite(loss)
    return {"pass": ok, "noop_only_root": out, "noop_gate": gate, "no_gate_prefers_better": nogate, "noop_has_no_action_cost": phantom,
            "connector_trap_distinct_X_nodes": len(v2["x_seqs"]) == 2 and v2["parent"][0] != v2["parent"][1], "connector_trap_loss": loss}


def t10_parity():
    ref = ROOT / "parity_ref"
    got = {f: hashlib.sha256((ref / f).read_bytes()).hexdigest()[:16] if (ref / f).exists() else None for f in PARITY}
    if got != PARITY:
        return {"pass": None, "status": "SOURCE_PARITY_UNVERIFIED", "hashes": got}
    sys.path.insert(0, str(ref))
    import abc_models as am, abc7_models as a7, bxy_core as bc  # noqa: E402
    g = torch.Generator().manual_seed(10)
    kern = 0.0
    for t in range(40):
        nS, nQ = 3, 5
        b, a, w = (torch.randn(n, generator=g, dtype=D64) for n in (nS, nQ, nQ))
        par = [int(x) for x in torch.randint(0, nS, (nQ,), generator=g)]
        oy, ox = {0: 1, 2: 0}, {1: 1, 3: 0, 4: 1}
        kern = max(kern, abs(float(bc.log_evidence(b, a, w, par, oy, ox) - XY.log_evidence(b, a, w, par, oy, ox))),
                   float(max((bc.marginals(b, a, w, torch.tensor(par))[k] - XY.marginals(b, a, w, torch.tensor(par))[k]).abs().max() for k in ("py", "px", "psucc"))))
    res = {"kernel_max_err": kern, "arms": {}}
    rng = np.random.default_rng(11)
    for arm, cfg in XY.ARMS.items():
        torch.manual_seed(5)
        m2 = a7.ModelBC7("B", "none", False, "pred").double()
        heads = {"xy": am.mlp(3 * am.W, 1), "xs": am.mlp(3 * am.W, 1), "jt": am.mlp(2 * am.W, 1), "geoY": am.mlp(13, 1), "geoX": am.mlp(8, 1)}
        heads = {k: h.double() for k, h in heads.items()}
        mine = XY.XYNet(arm, 3, 16, 32).double()
        sd = {**{k: v for k, v in m2.state_dict().items()}, **{f"{h}.{k}": v for h, mod in heads.items() for k, v in mod.state_dict().items()}}
        missing = [k for k in mine.state_dict() if k not in sd]
        mine.load_state_dict({k: sd[k] for k in mine.state_dict()})
        worst = 0.0
        for r in range(4):
            n = int(rng.integers(3, 6))
            inp = {"object_xy": rng.uniform(0, 0.64, (n, 2)), "visual": np.zeros((n, 384)), "object_radius": rng.uniform(0.02, 0.04, n),
                   "is_active_target": np.eye(n)[0], "is_obstacle": 1 - np.eye(n)[0], "robot_fk_joints": rng.uniform(-0.2, 0.4, (4, 2)),
                   "goal_xy": rng.uniform(0, 0.64, 2), "robot_q": rng.uniform(-2, 2, 3)}
            R = am.Root({"inputs": inp})
            fr = [tuple(rng.uniform(0, 0.64, 2)) for _ in range(4)]
            y_poses = [{}, {1: fr[0]}, {1: fr[0], 2: fr[1]}]
            x_seqs = [[(1, fr[0])], [(1, fr[0]), (2, fr[1])], [(2, fr[1]), (1, fr[0])]] + ([[(2, fr[1]), (n - 1, fr[2]), (1, fr[0])]] if n > 3 else [])
            stub = Stub()
            v, ym, xm = XY.make_view(f"p{r}", inp["object_xy"].tolist(), np.stack([inp["object_radius"], inp["is_active_target"], inp["is_obstacle"]], 1).tolist(),
                                     0, inp["goal_xy"].tolist(), inp["robot_fk_joints"][0].tolist(), R.ctx.tolist(), y_poses[1:], x_seqs, stub)
            with torch.no_grad():
                C = mine.coeffs(XY.collate([v])[0])
                c = m2.ctxn(R.ctx)
                vv = [m2.scene_vec(R, m2.edited(R, diff(v["xy0"], e))) for e in v["y_xy"]]
                fy = heads["geoY"](torch.tensor(v["phi_y"], dtype=D64))[:, 0]
                b = torch.stack([m2.scene(torch.cat([x, c]))[0] for x in vv]) + fy
                a, w, J = [], [], []
                for q, (seq, s_) in enumerate(zip(v["x_seqs"], v["parent"])):
                    sq, gq, _ = (m2._order(R, seq) if cfg["order"] == "pred" else p17_invariant(m2, R, seq))
                    h = torch.cat([gq, vv[s_], c])
                    aa = sq + (heads["xs"](h)[0] if cfg["sa"] else 0.0) + heads["geoX"](torch.tensor(v["phi_x" if cfg["order"] == "pred" else "phi_x0"][q], dtype=D64))[0]
                    a.append(aa + (fy[s_] if cfg["family"] == "direct" else 0.0))
                    w.append(heads["xy"](h)[0] if cfg["coupled"] else torch.zeros((), dtype=D64))
                    J.append(torch.nn.functional.softplus(heads["jt"](torch.cat([vv[s_], c]))[0]) + torch.nn.functional.softplus(m2.cost(gq)[0]))
                J0 = torch.nn.functional.softplus(heads["jt"](torch.cat([vv[0], c]))[0])
                worst = max(worst, float((C["b"] - b).abs().max()), float((C["a"] - torch.stack(a)).abs().max()), float((C["w"] - torch.stack(w)).abs().max()),
                            float((C["J"] - torch.stack(J)).abs().max()), float((C["J0"][0] - J0).abs()))
        res["arms"][arm] = {"max_abs_err_vs_2D_source_composition": worst, "unmapped_params": missing}
    ok = kern < 1e-12 and all(r["max_abs_err_vs_2D_source_composition"] < 1e-10 and not r["unmapped_params"] for r in res["arms"].values())
    return {"pass": ok, "status": "SOURCE_PARITY_VERIFIED" if ok else "SOURCE_PARITY_FAILED", "hashes": got, **res}


def diff(xy0, e):
    return {i: tuple(p) for i, (p, q) in enumerate(zip(e, xy0)) if tuple(p) != tuple(q)}


class Stub:
    """Deterministic pseudo-random GEO inputs for the parity test (identical values are fed to both implementations)."""
    @staticmethod
    def _h(tag, n):
        d = hashlib.sha256(tag.encode()).digest()
        return [d[i] / 255.0 for i in range(n)]

    @classmethod
    def phi_y(cls, v, xy):
        return cls._h(f"Y|{XY.sig_xy(xy)}", 13)

    @classmethod
    def phi_x(cls, v, seq, ordered):
        return cls._h(f"X|{seq}|{ordered}", 8)


def p17_invariant(self, R, seq):
    """bxy_p17.Model._order, invariant branch (e113c6680 bxy_p17.py l.108-113), verbatim."""
    W = 128
    toks, xy, _ = self.actions(R, seq, False)
    n = len(toks)
    rows = [self.mix(torch.cat([a, torch.stack([self.pairo(torch.cat([toks[u], a])) for u in range(n) if u != t]).mean(0) if n > 1
                                else torch.zeros(W, dtype=torch.float64)])) for t, a in enumerate(toks)]
    g = torch.stack(rows).mean(0) if rows else torch.zeros(W, dtype=torch.float64)
    return self.order_s(torch.cat([g, self.scene_vec(R, xy)]))[0], g, xy


def cmd_unit(a):
    out, t0 = {}, time.time()
    for name, f in (("t1_normalization_bruteforce_gradients", t1_normalization_and_gradients), ("t2_shared_Y_AB_BA_duplicates", t2_shared_y_and_duplicates),
                    ("t3_batching_permutation_caching_Y_marginal", t3_invariances), ("t4_w0_reduction_F5_gradients", t4_coupling),
                    ("t5_observed_data_marginal_cases", t5_marginal_loss_cases), ("t6_order_goal_object_permutation", t6_order_goal_permutation),
                    ("t7_real_format_roundtrip", t7_real_format_roundtrip), ("t9_DIRECT_G_interfaces_and_leakage", t9_direct_g_and_leakage),
                    ("t_noop_and_connector_trap", t_noop_and_connector_trap), ("t10_source_parity", t10_parity)):
        try:
            out[name] = f()
        except Exception as e:                                               # a crashed test is a FAILED test, never skipped
            import traceback
            out[name] = {"pass": False, "error": repr(e), "trace": traceback.format_exc()[-1500:]}
    out["summary"] = {k: v.get("pass") for k, v in out.items() if isinstance(v, dict)}
    out["all_pass"] = all(v is True for v in out["summary"].values())
    out["wall_s"] = round(time.time() - t0, 1)
    out["affinity"] = sorted(os.sched_getaffinity(0))
    out["note"] = "forward / backward only (no optimizer step); SYNTHETIC X labels where real 3D X labels are absent"
    s = json.dumps(out, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(s)
    print(json.dumps(out["summary"], indent=1), out["all_pass"], out["wall_s"])


# ---------------------------------------------------------------- test 8: planted recovery on the GPU (SYNTHETIC)
def cmd_synthetic(a):
    import guard, train as TR
    _, dev = TR.preflight({}, guard.load_cfg(a.guard_config), False)
    run = ROOT / "runs/XY3D_BWL/synthetic"
    wd = TR.start_watchdog(run, a.guard_config)
    dt = torch.float32
    res = {"label": "SYNTHETIC planted coefficients on random 6x6 3D-format scenes; NOT a 3D X/Y training result", "runs": [],
           "protocol": {"train_roots": 800, "early_stop_roots": 200, "heldout_roots": 400, "batch_roots": 32, "max_updates": 1500, "eval_every": 25,
                        "patience_evals": 12, "optimizer": "AdamW lr 1e-3 wd 1e-4 clip 10 (bxy_p11.TRAIN)", "dtype": "float32", "seeds": [0, 1]}}
    try:
        t_all = time.time()
        for truth in ("coupled", "uncoupled"):
            cp = truth == "coupled"
            sets = {k: synth_set(n, sd + (0 if cp else 100), cp) for k, n, sd in (("tr", 800, 21), ("es", 200, 22), ("ho", 400, 23))}
            chunks = {k: [XY.collate(sets[k][0][i:i + 32], sets[k][1][i:i + 32], dev, dt) for i in range(0, len(sets[k][0]), 32)] for k in sets}
            ho_views, ho_labs, ho_truth = sets["ho"]
            tb = torch.tensor(np.concatenate([t[0] for t in ho_truth]), dtype=dt, device=dev)
            ta = torch.tensor(np.concatenate([t[1] for t in ho_truth]), dtype=dt, device=dev)
            Bh, Lh = XY.collate(ho_views, ho_labs, dev, dt)
            truth_nll = float(-XY.batched_log_evidence(tb, ta, torch.full_like(ta, 1.5 if cp else 0.0), Bh["parent"], Lh["y_obs"], Lh["x_obs"],
                                                        Bh["s_root"], Bh["n_roots"]).mean())
            pairs = [(i, j) for i in range(len(ta)) for j in range(i + 1, min(i + 8, len(ta)))
                     if int(Bh["parent"][i]) == int(Bh["parent"][j]) and int(Bh["q_len"][i]) == int(Bh["q_len"][j]) >= 2 and abs(float(ta[i] - ta[j])) > 0.5]
            for arm in XY.ARMS:
                for seed in (0, 1):
                    torch.manual_seed(seed)
                    m = XY.XYNet(arm, 5, 14).to(dev, dt)
                    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
                    rng = np.random.default_rng(seed)
                    best, state, stale, hist, t0 = float("inf"), None, 0, [], time.time()
                    for u in range(1500):
                        B, L = chunks["tr"][int(rng.integers(len(chunks["tr"])))]
                        loss, _ = m.loss(B, L, lam=0.0)
                        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0); opt.step()
                        if (u + 1) % 25 == 0:
                            with torch.no_grad():
                                es = float(np.mean([float(m.loss(B, L, lam=0.0, kappa=0.0)[0]) for B, L in chunks["es"]]))
                            hist.append({"update": u + 1, "es_nll": es})
                            if es < best * (1 - 1e-3):
                                best, state, stale = es, copy.deepcopy(m.state_dict()), 0
                            else:
                                stale += 1
                            if stale >= 12 or TR.stopped(run):
                                break
                    m.load_state_dict(state)
                    with torch.no_grad():
                        C, P = m.coeffs(Bh), m.predict(Bh)
                        ev = [(int(Bh["parent"][q]), q) for q in range(len(ta))]
                        sq = Lh["succ_q"]
                        dec = sq >= 0
                        ps = P["psucc"].clamp(1e-6, 1 - 1e-6)
                        succ_ll = float(-(sq[dec] * ps[dec].log() + (1 - sq[dec]) * (1 - ps[dec]).log()).mean())
                        corr = lambda x, y: float(np.corrcoef(x.cpu().numpy(), y.cpu().numpy())[0, 1])  # noqa: E731
                        conc = [1.0 if (float(C["a"][i] - C["a"][j]) * float(ta[i] - ta[j])) > 0 else 0.5 if float(C["a"][i] - C["a"][j]) == 0 else 0.0 for i, j in pairs]
                        row = {"truth": truth, "arm": arm, "seed": seed, "updates": hist[-1]["update"], "best_es": best, "wall_s": round(time.time() - t0, 1),
                               "heldout_success_logloss": succ_ll, "heldout_success_events": int(dec.sum()), "order_pairs": len(pairs),
                               "order_concordance_a": float(np.mean(conc)) if conc else None, "corr_a_vs_planted": corr(C["a"], ta)}
                        if XY.ARMS[arm]["family"] == "xy":
                            row.update(heldout_joint_nll=float(-XY.batched_log_evidence(C["b"], C["a"], C["w"], Bh["parent"], Lh["y_obs"], Lh["x_obs"], Bh["s_root"],
                                                                                        Bh["n_roots"]).mean()),
                                       corr_b_vs_planted=corr(C["b"], tb), mean_w=float(C["w"].mean()), sd_w=float(C["w"].std()))
                        res["runs"].append(row)
                        print(json.dumps(row), flush=True)
            res[f"planted_truth_heldout_joint_nll_{truth}"] = truth_nll
        res["wall_s"] = round(time.time() - t_all, 1)
        res["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
        res["device"], res["affinity"], res["watchdog_stop"] = torch.cuda.get_device_name(0), sorted(os.sched_getaffinity(0)), (run / "STOP_RESOURCE").exists()
        if a.out:
            Path(a.out).write_text(json.dumps(res, indent=1))
    finally:
        wd.terminate()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["unit", "synthetic"]); ap.add_argument("--out"); ap.add_argument("--guard-config")
    a = ap.parse_args()
    {"unit": cmd_unit, "synthetic": cmd_synthetic}[a.cmd](a)
