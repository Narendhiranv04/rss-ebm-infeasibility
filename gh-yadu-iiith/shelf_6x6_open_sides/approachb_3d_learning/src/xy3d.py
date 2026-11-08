"""3D adapter for the EXISTING Approach B joint X/Y Boltzmann model (scene-first, root-consistent; F2 / F3 / F5).

Probability model ported unchanged from latent-causal-tamp branch approach-b-xy-v1 (e113c6680) bxy_core.py (log_py / log_px / log_evidence /
marginals / energy_root / brute_force), methodology final-fixed-methodology-1.pdf §10.2-10.3, §11.2-11.3:
    P(Y_s = 1) = sigma(b_s / tau)                       b_s = h_Y(v_s, c)            F3, scene coefficient of the EDITED ENDPOINT scene s
    P(X_r = 1 | Y_s = y) = sigma((a_r + w_r y) / tau)   a_r = h_X(g_r, v_s, c)       F2, ordered-prefix executability (order-aware)
                                                         w_r = h_XY(g_r, v_s, c)      F5, the single quadratic X-Y interaction
    P(X_r = 1, Y_s = 1) = sigma(b_s / tau) sigma((a_r + w_r) / tau)
    L_joint = -(1 / N_roots) sum_roots log P(D_root)    exact observed-data marginal (unobserved X / Y summed out), one Y node per unique
                                                         endpoint scene (distinct orders reaching it share Y, keep distinct X)
    L_B = L_joint + lambda_J L_J                         optional Huber travel auxiliary on complete witnesses only (never invented for failures)
Arms for the eventual 3D study: COUPLED (w learned), UNCOUPLED (w = 0), DIRECT (a single success logit; labelled ablation only),
ORDER_INVARIANT (mean-pooled actions), plus matched AB/BA same-endpoint contrasts.

3D status (see reports/05): no genuine 3D X record exists (no ordered relocation sequence with carried robot state and a terminal connector to
q_ref); this module is therefore validated ONLY on clearly labelled SYNTHETIC fixtures.  `inventory` counts what real 3D supervision exists.

    python src/xy3d.py selftest                      exact-math + normalization + gradient + order / Y-sharing tests (synthetic, CPU or GPU)
    python src/xy3d.py synthetic-gpu                 tiny GPU proof that the adapter recovers PLANTED synthetic coefficients (no real labels)
    python src/xy3d.py inventory --snapshot DIR      read-only inventory of 3D sequence / target-attempt labels in the source roots"""
from __future__ import annotations

import argparse, itertools, json, math, os, sys, time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import SceneEncoder, mlp  # noqa: E402

TAU = 1.0


# ---------------------------------------------------------------- exact probability model (verbatim semantics of bxy_core.py)
def log_py(b, y, tau=TAU):
    return y * b / tau - F.softplus(b / tau)


def log_px(a, w, x, y, tau=TAU):
    u = a + w * y
    return x * u / tau - F.softplus(u / tau)


def log_evidence(b, a, w, parent, obs_y, obs_x, tau=TAU):
    """log P(D_c) of one root: per unique scene s, logsumexp over allowed y of [log P^Y(y) + sum over observed child X log P^X(x | y)]."""
    kids = {}
    for q, x in obs_x.items():
        kids.setdefault(parent[q], []).append((q, x))
    tot = b.new_zeros(())
    for s in sorted(set(obs_y) | set(kids)):
        ys = [obs_y[s]] if s in obs_y else [0, 1]
        terms = [log_py(b[s], y, tau) + sum((log_px(a[q], w[q], x, y, tau) for q, x in kids.get(s, [])), b.new_zeros(())) for y in ys]
        tot = tot + torch.logsumexp(torch.stack(terms), 0)
    return tot


def marginals(b, a, w, parent, tau=TAU):
    py = torch.sigmoid(b / tau)
    pb = py[parent] if len(parent) else py.new_zeros(0)
    p1, p0 = torch.sigmoid((a + w) / tau), torch.sigmoid(a / tau)
    return {"py": py, "px": pb * p1 + (1 - pb) * p0, "psucc": pb * p1}


def energy_root(b, a, w, parent, y, x, tau=TAU):
    e = sum(-b[s] * y[s] + tau * F.softplus(b[s] / tau) for s in range(len(b)))
    for q in range(len(a)):
        u = a[q] + w[q] * y[parent[q]]
        e = e + (-u * x[q] + tau * F.softplus(u / tau))
    return e


def brute_force(b, a, w, parent, obs_y, obs_x, tau=TAU):
    out = []
    for y in itertools.product((0, 1), repeat=len(b)):
        if any(y[s] != v for s, v in obs_y.items()):
            continue
        for x in itertools.product((0, 1), repeat=len(a)):
            if any(x[q] != v for q, v in obs_x.items()):
                continue
            out.append(-energy_root(b, a, w, parent, y, x, tau) / tau)
    return float(torch.logsumexp(torch.stack(out), 0))


def batched_log_evidence(b, a, w, parent, y_obs, x_obs, root_of_scene, n_roots, tau=TAU):
    """Vectorized exact evidence for many roots.  b [S], a,w [Q], parent [Q] (scene index), y_obs [S] in {-1,0,1}, x_obs [Q] in {-1,0,1},
    root_of_scene [S].  Scenes with no observation and no observed child contribute exactly 0.  -> per-root log P(D) [n_roots]."""
    lp = []
    for y in (0, 1):
        ly = log_py(b, float(y), tau)
        lx = torch.where(x_obs >= 0, log_px(a, w, x_obs.clamp(min=0).float(), float(y), tau), torch.zeros_like(a))
        tot = ly + torch.zeros_like(b).index_add(0, parent, lx)
        lp.append(torch.where((y_obs < 0) | (y_obs == y), tot, torch.full_like(tot, -math.inf)))
    per_scene = torch.logsumexp(torch.stack(lp), 0)
    observed_child = torch.zeros_like(b).index_add(0, parent, (x_obs >= 0).to(b.dtype)) > 0
    per_scene = torch.where((y_obs >= 0) | observed_child, per_scene, torch.zeros_like(per_scene))
    return torch.zeros(n_roots, dtype=b.dtype, device=b.device).index_add(0, root_of_scene, per_scene)


# ---------------------------------------------------------------- 3D neural coefficients
class XY3D(nn.Module):
    """b_s from the endpoint scene; a_r, w_r from the ORDERED action sequence g_r (GRU over action tokens) and the endpoint scene v_s.
    Action token = [moved object's 3D features (14), destination cell centre (x, y m), destination (r/5, c/5), step index / 4]."""

    def __init__(self, arm="COUPLED", h=64, nfeat=14):
        super().__init__()
        assert arm in ("COUPLED", "UNCOUPLED", "DIRECT", "ORDER_INVARIANT")
        self.arm = arm
        self.scene = SceneEncoder(nfeat, h)
        self.tok = mlp(nfeat + 5, h, h, 1)
        self.gru = nn.GRU(h, h, batch_first=True)
        self.hY, self.hX, self.hXY = mlp(h, h, 1), mlp(2 * h, h, 1), mlp(2 * h, h, 1)
        self.jt = mlp(2 * h, h, 1)

    def seq(self, tokens, lens):
        if self.arm == "ORDER_INVARIANT":                                  # the control sees the action SET: the step index is dropped
            tokens = tokens.clone(); tokens[..., -1] = 0
        e = self.tok(tokens)
        if self.arm == "ORDER_INVARIANT":
            m = (torch.arange(e.shape[1], device=e.device)[None] < lens[:, None]).float()[..., None]
            return (e * m).sum(1) / lens[:, None].clamp(min=1)
        out, _ = self.gru(nn.utils.rnn.pack_padded_sequence(e, lens.cpu(), batch_first=True, enforce_sorted=False))
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=e.shape[1])
        return out[torch.arange(len(lens), device=e.device), (lens - 1).clamp(min=0)]

    def forward(self, s_obj, s_mask, s_occ, tokens, lens, parent):
        v = self.scene(s_obj, s_mask, s_occ)                               # [S, h] endpoint scenes
        b = self.hY(v).squeeze(-1)
        g = self.seq(tokens, lens)                                         # [Q, h] ordered prefixes
        gv = torch.cat([g, v[parent]], -1)
        a = self.hX(gv).squeeze(-1)
        w = self.hXY(gv).squeeze(-1) if self.arm == "COUPLED" else torch.zeros_like(a)
        return b, a, w, F.softplus(self.jt(gv).squeeze(-1))

    def loss(self, batch, lambda_j=0.0):
        b, a, w, jhat = self(batch["s_obj"], batch["s_mask"], batch["s_occ"], batch["tokens"], batch["lens"], batch["parent"])
        if self.arm == "DIRECT":                                           # labelled ablation: one success logit on complete (x, y) pairs only
            ok = (batch["x_obs"] >= 0) & (batch["y_obs"][batch["parent"]] >= 0)
            tgt = (batch["x_obs"] == 1) & (batch["y_obs"][batch["parent"]] == 1)
            return F.binary_cross_entropy_with_logits((a + b[batch["parent"]])[ok], tgt[ok].float()), {}
        lp = batched_log_evidence(b, a, w, batch["parent"], batch["y_obs"], batch["x_obs"], batch["root_of_scene"], batch["n_roots"])
        l = -lp.mean()
        parts = {"nll": l.item()}
        if lambda_j and "travel" in batch:
            m = batch["travel"] >= 0                                       # complete witnesses only
            if m.any():
                lj = F.huber_loss(torch.log1p(jhat[m]), torch.log1p(batch["travel"][m]), delta=1.0)
                l = l + lambda_j * lj; parts["travel"] = lj.item()
        return l, parts


# ---------------------------------------------------------------- SYNTHETIC fixtures (clearly not data)
def synthetic_batch(n_roots=64, seed=0, device="cpu", planted=None):
    """SYNTHETIC roots: each root = 1-2 endpoint scenes; each scene reached by 1-3 orders (AB / BA style permutations of the same moves).
    Labels are drawn from PLANTED coefficients so recovery can be checked.  Nothing here is a physical label."""
    g = torch.Generator().manual_seed(seed)
    S_obj, S_mask, S_occ, toks, lens, parent, ros, ys, xs, b_true, a_true, w_true = [], [], [], [], [], [], [], [], [], [], [], []
    si = 0
    for r in range(n_roots):
        for _ in range(int(torch.randint(1, 3, (1,), generator=g))):
            n = int(torch.randint(2, 6, (1,), generator=g))
            obj = torch.zeros(8, 14); obj[:n] = torch.randn(n, 14, generator=g) * 0.5
            mask = torch.zeros(8, dtype=torch.bool); mask[:n] = True
            occ = torch.zeros(36); occ[torch.randperm(36, generator=g)[:n]] = 1
            S_obj.append(obj); S_mask.append(mask); S_occ.append(occ); ros.append(r)
            bt = float(obj[:n, 3].mean() * 3 - occ[:18].sum() * 0.4)       # planted F3: depends on scene content
            moves = torch.randn(int(torch.randint(1, 4, (1,), generator=g)), 19, generator=g) * 0.5
            yv = int(torch.rand(1, generator=g) < torch.sigmoid(torch.tensor(bt)))
            ys.append(yv if torch.rand(1, generator=g) < 0.7 else -1); b_true.append(bt)
            perms = list(itertools.permutations(range(len(moves))))[:3]
            for p in perms:
                t = moves[list(p)].clone(); t[:, -1] = torch.arange(len(p)) / 4
                at = float(t[0, 0] * 2 - t[-1, 1] * 1.5)                    # planted F2: ORDER-dependent (first vs last move)
                wt = 1.0 if (planted or {}).get("coupled", True) else 0.0
                xv = int(torch.rand(1, generator=g) < torch.sigmoid(torch.tensor(at + wt * yv)))
                pad = torch.zeros(3, 19); pad[:len(p)] = t
                toks.append(pad); lens.append(len(p)); parent.append(si)
                xs.append(xv if torch.rand(1, generator=g) < 0.8 else -1); a_true.append(at); w_true.append(wt)
            si += 1
    T = lambda v, dt=torch.float32: torch.tensor(v, dtype=dt, device=device)
    return {"s_obj": torch.stack(S_obj).to(device), "s_mask": torch.stack(S_mask).to(device), "s_occ": torch.stack(S_occ).to(device),
            "tokens": torch.stack(toks).to(device), "lens": T(lens, torch.long), "parent": T(parent, torch.long), "root_of_scene": T(ros, torch.long),
            "n_roots": n_roots, "y_obs": T(ys, torch.long), "x_obs": T(xs, torch.long),
            "truth": {"b": T(b_true), "a": T(a_true), "w": T(w_true)}}


def selftest(device="cpu"):
    torch.manual_seed(0)
    res = {}
    # 1. exact evidence == brute force over E_root, with missing X / Y (float64)
    worst = 0.0
    for t in range(60):
        g = torch.Generator().manual_seed(t)
        nS, nQ = int(torch.randint(1, 4, (1,), generator=g)), int(torch.randint(1, 5, (1,), generator=g))
        b, a, w = (torch.randn(n, generator=g, dtype=torch.float64) * 2 for n in (nS, nQ, nQ))
        parent = [int(torch.randint(0, nS, (1,), generator=g)) for _ in range(nQ)]
        oy = {s: int(torch.randint(0, 2, (1,), generator=g)) for s in range(nS) if torch.rand(1, generator=g) < 0.5}
        ox = {q: int(torch.randint(0, 2, (1,), generator=g)) for q in range(nQ) if torch.rand(1, generator=g) < 0.6}
        le = float(log_evidence(b, a, w, parent, oy, ox))
        touched = set(oy) | {parent[q] for q in ox}
        bf = brute_force(b, a, w, parent, {s: v for s, v in oy.items()}, ox)
        # brute force sums every scene; untouched scenes contribute log 1 = 0 there as well
        worst = max(worst, abs(le - bf))
        yo = torch.tensor([oy.get(s, -1) for s in range(nS)]); xo = torch.tensor([ox.get(q, -1) for q in range(nQ)])
        bl = float(batched_log_evidence(b, a, w, torch.tensor(parent), yo, xo, torch.zeros(nS, dtype=torch.long), 1)[0])
        worst = max(worst, abs(bl - bf))
    res["evidence_vs_bruteforce_max_abs_err"] = worst
    # 2. normalization: the joint over every (x, y) assignment sums to 1
    b, a, w = torch.randn(2, dtype=torch.float64), torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64)
    par = [0, 0, 1]
    Z = torch.logsumexp(torch.stack([-energy_root(b, a, w, par, y, x) for y in itertools.product((0, 1), repeat=2)
                                     for x in itertools.product((0, 1), repeat=3)]), 0)
    res["joint_log_partition"] = float(Z)
    m = marginals(b, a, w, torch.tensor(par))
    res["psucc_equals_sigma_b_sigma_a_plus_w"] = float((m["psucc"] - torch.sigmoid(b[par]) * torch.sigmoid(a + w)).abs().max())
    # 3. network: shared Y per endpoint, order sensitivity, gradients
    batch = synthetic_batch(16, seed=1, device=device)
    out = {}
    for arm in ("COUPLED", "UNCOUPLED", "ORDER_INVARIANT", "DIRECT"):
        net = XY3D(arm).to(device)
        l, _ = net.loss(batch)
        l.backward()
        grads = {n: (p.grad is not None and bool(torch.isfinite(p.grad).all()) and float(p.grad.abs().sum()) > 0) for n, p in net.named_parameters()}
        b_, a_, w_, _ = net(batch["s_obj"], batch["s_mask"], batch["s_occ"], batch["tokens"], batch["lens"], batch["parent"])
        same = [(i, j) for i in range(len(batch["parent"])) for j in range(i + 1, len(batch["parent"]))
                if batch["parent"][i] == batch["parent"][j] and batch["lens"][i] == batch["lens"][j] > 1]
        ab_ba = max((float((a_[i] - a_[j]).detach().abs()) for i, j in same), default=0.0)
        out[arm] = {"loss_finite": bool(torch.isfinite(l)), "heads_with_gradient": sorted({n.split(".")[0] for n, ok in grads.items() if ok}),
                    "w_identically_zero": bool((w_ == 0).all()),
                    "one_Y_node_per_endpoint_scene": len(b_) == len(batch["root_of_scene"]) and len(same) > 0,
                    "max_abs_a_diff_same_endpoint_reordered": ab_ba}
    res["arms"] = out
    res["pass"] = (worst < 1e-9 and abs(res["joint_log_partition"]) < 1e-9 and res["psucc_equals_sigma_b_sigma_a_plus_w"] < 1e-12
                   and out["UNCOUPLED"]["w_identically_zero"] and not out["COUPLED"]["w_identically_zero"]
                   and out["COUPLED"]["max_abs_a_diff_same_endpoint_reordered"] > 1e-6
                   and out["ORDER_INVARIANT"]["max_abs_a_diff_same_endpoint_reordered"] < 1e-5
                   and all(v["loss_finite"] for v in out.values()))
    return res


def synthetic_gpu(device, steps=4000, patience=400):
    """SYNTHETIC proof of learning on the GPU: fit the arms to labels drawn from planted coefficients (coupled truth), early-stopped on a
    held-out synthetic split (best checkpoint restored); compare with the planted-truth NLL.  No physical label is involved."""
    tr, va = synthetic_batch(3000, seed=11, device=device), synthetic_batch(1000, seed=12, device=device)
    out = {}
    for arm in ("COUPLED", "UNCOUPLED", "ORDER_INVARIANT"):
        torch.manual_seed(0)
        net = XY3D(arm).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-3)
        curve, best, best_state, since = [], float("inf"), None, 0
        for s in range(steps):
            l, _ = net.loss(tr)
            opt.zero_grad(); l.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step()
            if s % 20 == 0:
                with torch.no_grad():
                    v = net.loss(va)[0].item()
                if s % 100 == 0:
                    curve.append({"step": s, "train_nll": l.item(), "val_nll": v})
                if v < best - 1e-4:
                    best, since = v, 0
                    best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
                else:
                    since += 20
                if since >= patience:
                    break
        net.load_state_dict(best_state)
        curve.append({"step": s, "train_nll": l.item(), "val_nll": best, "restored_best": True})
        with torch.no_grad():
            b, a, w, _ = net(va["s_obj"], va["s_mask"], va["s_occ"], va["tokens"], va["lens"], va["parent"])
            tb, ta = va["truth"]["b"], va["truth"]["a"]
            corr = lambda u, v: float(torch.corrcoef(torch.stack([u, v]))[0, 1])
            out[arm] = {"curve": curve, "final_val_nll": curve[-1]["val_nll"], "corr_b_vs_planted": corr(b, tb), "corr_a_vs_planted": corr(a, ta),
                        "mean_w": float(w.mean())}
        tv = va["truth"]
        out["planted_truth_val_nll"] = float(-batched_log_evidence(tv["b"], tv["a"], tv["w"], va["parent"], va["y_obs"], va["x_obs"],
                                                                    va["root_of_scene"], va["n_roots"]).mean())
    out["label"] = "SYNTHETIC FIXTURE ONLY -- planted coefficients; not a 3D X/Y training result"
    return out


# ---------------------------------------------------------------- real 3D label inventory (read-only)
def inventory(snapshot):
    snap = Path(snapshot)
    src = snap / "upstream/gh-yadu-iiith/shelf_6x6_open_sides/dataset_bd_classquota_without_home_position_no_overpass/raw/roots"
    v3 = snap / "upstream/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/production_v3"
    split = json.loads((v3 / "frozen/split_manifest_first_cohort.json").read_text())["split"]
    qc = set(json.loads((v3 / "QC_ELIGIBLE_ROOTS.json").read_text())["roots"])
    c = Counter(); ex = {}
    for p in sorted(src.glob("*.json")):
        d = json.loads(p.read_text()); rid = d["ids"]["root_id"]
        if rid not in qc:
            continue
        sp = split.get(rid, "UNSPLIT")
        L = d.get("labels", {})
        for k, v in L.items():
            items = v if isinstance(v, list) else [v]
            for it in items:
                if not isinstance(it, dict):
                    continue
                c[(k, sp, it.get("reason") or it.get("negative_kind") or "?", str(it.get("y", it.get("label"))))] += 1
                ex.setdefault(k, {kk: (vv if not isinstance(vv, (list, dict)) else type(vv).__name__) for kk, vv in it.items()})
    rows = [{"label_family": k[0], "split": k[1], "reason": k[2], "y": k[3], "n": n} for k, n in sorted(c.items())]
    return {"qc_roots": len(qc), "label_families": sorted({r["label_family"] for r in rows}), "rows": rows, "examples": ex}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "synthetic-gpu", "inventory"]); ap.add_argument("--snapshot"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.cmd == "selftest":
        r = selftest("cuda:0" if torch.cuda.is_available() and os.environ.get("XY_GPU") == "1" else "cpu")
    elif a.cmd == "synthetic-gpu":
        import guard, train as TR                                          # same preflight (affinity, guard) + independent watchdog
        gcfg = guard.load_cfg()
        _, dev = TR.preflight({}, gcfg, False)
        run = Path(__file__).resolve().parents[1] / "runs/XY3D_SYNTHETIC"
        wd = TR.start_watchdog(run, None)
        try:
            r = synthetic_gpu(dev)
            r["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1); r["device"] = torch.cuda.get_device_name(0)
            r["watchdog_stop"] = (run / "STOP_RESOURCE").exists()
        finally:
            wd.terminate()
    else:
        r = inventory(a.snapshot)
    s = json.dumps(r, indent=1)
    if a.out:
        Path(a.out).write_text(s)
    print(s[:4000])
