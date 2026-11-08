"""3D Approach B joint X/Y Boltzmann model: the November 5 / source-verified 2D F2 / F3 / F5 family with 3D geometry inputs (BWL-2).

Probability model (UNCHANGED from the kickoff; semantics of latent-causal-tamp approach-b-xy-v1 e113c6680 bxy_core.py l.28-78,
final-fixed-methodology-1.pdf section 10.2-10.3 / 11.2-11.3), tau = 1:
    P(Y_s = 1) = sigma(b_s)      P(X_r = 1 | Y_s = y) = sigma(a_r + w_r y)      p_succ(r) = sigma(b_s) sigma(a_r + w_r),   p_succ(()) = P(Y_0 = 1)
    L_joint = root mean of -log P(observed X / Y)   (unobserved X / Y summed out exactly; one Y node per unique endpoint, one X node per ordered key)
Neural coefficients = a batched, padded re-expression of the 2D source modules (abc_models.Shared l.54-89, abc7_models.Shared7 l.37-62,
ModelBC7(kind B, couple none, order pred) l.115-152, bxy_core.ModelXY l.82-108, bxy_v2.ModelXY2.j_hat, bxy_p11.Model.coeffs l.236-266,
bxy_p17.Model l.95-124 with ARMS l.49-54 and sym l.133-151):
    F3  b_s = scene([scene_vec(S_s), c]) (+ geoY(phi_y(S_s)))      scene_vec: edit-aware tokens [moved, dx, dy, |d|, x0, y0], pooled + moved + TARGET token
    F2  a_r = order_s([g_r, scene_vec(S_r)]) (+ xs([g_r, v_s, c]) for E / DIRECT) (+ geoX(phi_x))    g_r: IDEAL-PREFIX action tokens + predecessor pair
        mixing (pairo / mix over u < t); G: prefix-free S0 tokens of the canonically sorted actions, pair mixing over u != t, phi_x evaluated in S0
    F5  w_r = xy([g_r, v_s, c]) in D and G only; E sets w = 0 (capacity-matched by xs); DIRECT: independent success logit a_r (+ geoY_s), no-op logit b_0
    J   = softplus(jt([v_s, c])) + softplus(cost(g_r));  J_0 = softplus(jt([v_0, c]))        travel readout (Huber on complete witnesses only)
    c   = ctxn(root context) -- 3D: declared ready reference pose q_ref (= q0 for SHELF66 roots), goal, target size / height, robot capability flags
F1 is the deterministic legality screen of the sampler / search (not learned); F4 omissions are new ordered queries of the same F2 / F3.
3D differs from 2D only in the per-object numeric inputs (+ height, shape), the context vector, no DINO visual features (the 2D vision=False
form) and the GEO feature definitions (observable occupancy proxies, see xy3d_data.GEO3D).  No new energy, causal graph or q-state head.
No genuine 3D X label exists (reports/11): this module is exercised on SYNTHETIC labels and on real-format 3D records only.

    python src/xy3d.py selftest          exact-math checks (CPU, no training)"""
from __future__ import annotations

import itertools, json, math, sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

TAU = 1.0
W = 128


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
        lx = torch.where(x_obs >= 0, log_px(a, w, x_obs.clamp(min=0).to(a.dtype), float(y), tau), torch.zeros_like(a))
        tot = ly + torch.zeros_like(b).index_add(0, parent, lx)
        lp.append(torch.where((y_obs < 0) | (y_obs == y), tot, torch.full_like(tot, -math.inf)))
    per_scene = torch.logsumexp(torch.stack(lp), 0)
    observed_child = torch.zeros_like(b).index_add(0, parent, (x_obs >= 0).to(b.dtype)) > 0
    per_scene = torch.where((y_obs >= 0) | observed_child, per_scene, torch.zeros_like(per_scene))
    return torch.zeros(n_roots, dtype=b.dtype, device=b.device).index_add(0, root_of_scene, per_scene)


# ---------------------------------------------------------------- views: one Y node per endpoint, one X node per ordered key (domain-agnostic)
def sig_xy(xy):
    return tuple((round(float(p[0]), 9), round(float(p[1]), 9)) for p in xy)


def make_view(key, xy0, attrs, tid, goal, base, ctx, y_edits, x_seqs, geo, okey=None):
    """Learner-visible view of one root.  y_edits: [{slot: (x, y)}] (an empty edit = the no-op endpoint Y0, always added); x_seqs: nonempty
    ordered [(slot, (x, y))] with no slot repeated.  okey: stable per-object key (default: slot index = the 2D source's sorted id order) that fixes
    the canonical action order of the order-invariant arm, so relabelling slots with stable ids cannot change it.  Endpoints are deduplicated by full object-pose signature (AB / BA share one Y node),
    ordered sequences by their ordered signature (a repeated query is one X node).  geo: callbacks phi_y(view, xy) / phi_x(view, seq, ordered).
    Returns (view, y_map, x_map): input index -> node index (labels are attached through these maps, never by list position)."""
    base_xy = [tuple(map(float, p)) for p in xy0]

    def endpoint(edit):
        e = list(base_xy)
        for i, p in edit.items():
            e[int(i)] = tuple(map(float, p))
        return e
    ys, ysig, y_map = [], {}, []
    for edit in [{}] + list(y_edits):
        e = endpoint(edit)
        y_map.append(ysig.setdefault(sig_xy(e), len(ys)))
        if len(ys) < len(ysig):
            ys.append(e)
    xs, xsig, x_map, parent = [], {}, [], []
    for seq in x_seqs:
        seq = [(int(i), tuple(map(float, p))) for i, p in seq]
        if not seq or len({i for i, _ in seq}) != len(seq):
            raise ValueError(f"{key}: X query must be nonempty with no object moved twice: {seq}")
        k = tuple((i, round(p[0], 9), round(p[1], 9)) for i, p in seq)
        if k not in xsig:
            xsig[k] = len(xs)
            xs.append(seq)
            e = endpoint(dict(seq))
            s = ysig.setdefault(sig_xy(e), len(ys))
            if s == len(ys):
                ys.append(e)
            parent.append(s)
        x_map.append(xsig[k])
    v = {"key": key, "xy0": base_xy, "attrs": [list(map(float, a)) for a in attrs], "tid": int(tid), "goal": tuple(map(float, goal)),
         "base": tuple(map(float, base)), "ctx": list(map(float, ctx)), "y_xy": ys, "y0": 0, "x_seqs": xs, "parent": parent,
         "okey": list(range(len(base_xy))) if okey is None else [int(k) for k in okey]}
    v["phi_y"] = [geo.phi_y(v, e) for e in ys]
    v["phi_x"] = [geo.phi_x(v, q, True) for q in xs]
    v["phi_x0"] = [geo.phi_x(v, q, False) for q in xs]
    return v, y_map[1:], x_map


def labels_for(v, y_map, x_map, obs_y=None, obs_x=None, travel=(), noop_y=None):
    """Attach observed outcomes through the node maps: obs_y {edit index: 0/1}, obs_x {seq index: 0/1}, travel [(seq index | None, J)].
    A repeated query must carry the same value (raises otherwise); unattempted / censored queries are simply absent."""
    oy, ox = {}, {}
    if noop_y is not None:
        oy[v["y0"]] = int(noop_y)
    for d, src, m in ((oy, obs_y or {}, y_map), (ox, obs_x or {}, x_map)):
        for i, val in src.items():
            n = m[i]
            if d.setdefault(n, int(val)) != int(val):
                raise ValueError(f"{v['key']}: conflicting duplicate observation for node {n}")
    tr = [(None if i is None else x_map[i], float(j)) for i, j in travel]
    succ = {}
    for q, s in enumerate(v["parent"]):                                       # DIRECT targets (bxy_p11.success_events l.156-167)
        x, y = ox.get(q), oy.get(s)
        if x == 0 or y == 0:
            succ[q] = 0
        elif x == 1 and y == 1:
            succ[q] = 1
    return {"obs_y": oy, "obs_x": ox, "travel": tr, "succ": succ, "succ0": oy.get(v["y0"])}


def collate(views, labs=None, device="cpu", dtype=torch.float64):
    """Stack views (and labels) into padded tensors; scene / query indices are global within the batch."""
    T = lambda x, dt=dtype: torch.as_tensor(x, dtype=dt, device=device)  # noqa: E731
    N = max(len(v["xy0"]) for v in views)
    Mx = max([len(q) for v in views for q in v["x_seqs"]] + [1])
    A, C = len(views[0]["attrs"][0]), len(views[0]["ctx"])
    pad = lambda rows, n, w: [list(r) + [[0.0] * w] * (n - len(r)) for r in rows]  # noqa: E731
    b = {"xy0": T(pad([v["xy0"] for v in views], N, 2)), "attrs": T(pad([v["attrs"] for v in views], N, A)),
         "omask": T([[True] * len(v["xy0"]) + [False] * (N - len(v["xy0"])) for v in views], torch.bool),
         "tid": T([v["tid"] for v in views], torch.long), "goal": T([v["goal"] for v in views]), "base": T([v["base"] for v in views]),
         "ctx": T([v["ctx"] for v in views]), "okey": T([v["okey"] + [1 << 30] * (N - len(v["okey"])) for v in views], torch.long)}
    s_root, s_xy, y0, phi_y, q_root, parent, q_obj, q_dest, q_len, phi_x, phi_x0, so = [], [], [], [], [], [], [], [], [], [], [], 0
    for r, v in enumerate(views):
        s_root += [r] * len(v["y_xy"]); s_xy += pad(v["y_xy"], N, 2); y0.append(so + v["y0"]); phi_y += v["phi_y"]
        for q, seq in enumerate(v["x_seqs"]):
            q_root.append(r); parent.append(so + v["parent"][q]); q_len.append(len(seq))
            q_obj.append([i for i, _ in seq] + [-1] * (Mx - len(seq))); q_dest.append([list(p) for _, p in seq] + [[0.0, 0.0]] * (Mx - len(seq)))
        phi_x += v["phi_x"]; phi_x0 += v["phi_x0"]
        so += len(v["y_xy"])
    b.update(s_root=T(s_root, torch.long), s_xy=T(s_xy), y0=T(y0, torch.long), phi_y=T(phi_y).reshape(-1, 13), q_root=T(q_root, torch.long),
             parent=T(parent, torch.long), q_obj=T(q_obj, torch.long).reshape(-1, Mx), q_dest=T(q_dest).reshape(-1, Mx, 2), q_len=T(q_len, torch.long),
             phi_x=T(phi_x).reshape(-1, 8), phi_x0=T(phi_x0).reshape(-1, 8), n_roots=len(views))
    if labs is None:
        return b, None
    yo, xo, sq, s0, jq, jr, jv, qo, so = [], [], [], [], [], [], [], 0, 0
    for r, (v, l) in enumerate(zip(views, labs)):
        yo += [l["obs_y"].get(s, -1) for s in range(len(v["y_xy"]))]
        xo += [l["obs_x"].get(q, -1) for q in range(len(v["x_seqs"]))]
        sq += [l["succ"].get(q, -1) for q in range(len(v["x_seqs"]))]
        s0.append(-1 if l["succ0"] is None else l["succ0"])
        for i, j in l["travel"]:
            jq.append(-1 if i is None else qo + i); jr.append(r); jv.append(j)
        qo += len(v["x_seqs"]); so += len(v["y_xy"])
    lab = {"y_obs": T(yo, torch.long), "x_obs": T(xo, torch.long), "succ_q": T(sq, torch.long), "succ0": T(s0, torch.long),
           "j_q": T(jq, torch.long), "j_root": T(jr, torch.long), "j_val": T(jv)}
    return b, lab


# ---------------------------------------------------------------- neural coefficients (2D source family, batched)
def mlp2(i, o, h=W):
    """Identical to abc_models.mlp (e113c6680 abc_models.py l.28-29)."""
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


ARMS = {"D": {"family": "xy", "coupled": True, "sa": False, "geo": True, "loss": "SYM", "order": "pred", "lam": 1.0},            # D:XYQ-GEO-SYM
        "E": {"family": "xy", "coupled": False, "sa": True, "geo": True, "loss": "SYM", "order": "pred", "lam": 1.0},            # E:XY-UNC-SA-GEO-SYM
        "DIRECT-GEO": {"family": "direct", "coupled": False, "sa": True, "geo": True, "loss": None, "order": "pred", "lam": 0.0},  # A:DIRECT-GEO
        "G": {"family": "xy", "coupled": True, "sa": False, "geo": True, "loss": "SYM", "order": "invariant", "lam": 1.0}}   # G:XYQ-GEO-SYM-ORDINV
KAPPA, HUBER = 1.0, 1.0


class XYNet(nn.Module):
    def __init__(self, arm, attr_dim, ctx_dim, vis_dim=0):
        super().__init__()
        self.arm, self.cfg, self.vis_dim = arm, ARMS[arm], vis_dim
        self.enc, self.ctxn, self.pooln = mlp2(11 + attr_dim + vis_dim, W), mlp2(ctx_dim, W), nn.Linear(2 * W, W)
        self.act, self.edit_proj, self.edit_head = mlp2(3 * W + 1, W), nn.Linear(6, W), mlp2(3 * W + 1, W)
        self.scene, self.order_s, self.cost, self.pairo, self.mix, self.jt = mlp2(2 * W, 1), mlp2(2 * W, 1), mlp2(W, 1), mlp2(2 * W, W), mlp2(2 * W, W), mlp2(2 * W, 1)
        if self.cfg["coupled"]:
            self.xy = mlp2(3 * W, 1)
        if self.cfg["sa"]:
            self.xs = mlp2(3 * W, 1)
        if self.cfg["geo"]:
            self.geoY, self.geoX = mlp2(13, 1), mlp2(8, 1)

    # -- abc_models.Root.numeric / Shared.encode / pooled, abc7_models.Shared7.scene_vec (padded objects masked out of every pool)
    def numeric(self, b, xy, ri):
        rel = lambda p: torch.cat([xy - p, (xy - p).norm(dim=-1, keepdim=True)], -1)  # noqa: E731
        d = xy - b["base"][ri][:, None]
        ang = torch.atan2(d[..., 1], d[..., 0])
        tgt = xy[torch.arange(len(ri), device=xy.device), b["tid"][ri]][:, None]
        return torch.cat([xy, b["attrs"][ri], d.norm(dim=-1, keepdim=True), ang.sin()[..., None], ang.cos()[..., None],
                          rel(b["goal"][ri][:, None]), rel(tgt)], -1)

    def encode(self, b, xy, ri):
        x = self.numeric(b, xy, ri)
        if self.vis_dim:
            x = torch.cat([x, x.new_zeros(*x.shape[:2], self.vis_dim)], -1)        # the 2D vision=False form (projection of zeros)
        return self.enc(x)

    def pooled(self, tok, mask):
        m = mask[..., None]
        mean = (tok * m).sum(1) / m.sum(1).clamp(min=1)
        return self.pooln(torch.cat([mean, tok.masked_fill(~m, -torch.inf).max(1).values], -1))

    def scene_vec(self, b, xy, ri):
        x0, mask = b["xy0"][ri], b["omask"][ri]
        d = xy - x0
        moved = ((d.norm(dim=-1) > 1e-9) & mask).to(xy.dtype)
        tok = self.encode(b, xy, ri) + self.edit_proj(torch.cat([moved[..., None], d, d.norm(dim=-1, keepdim=True), x0], -1))
        mv = (tok * moved[..., None]).sum(1) / moved.sum(1, keepdim=True).clamp(min=1.0)
        tt = tok[torch.arange(len(ri), device=xy.device), b["tid"][ri]]
        return self.edit_head(torch.cat([self.pooled(tok, mask), mv, tt, moved.sum(1, keepdim=True) / 3.0], -1))

    # -- abc_models.Shared.actions (ideal-prefix tokens); ordered=False = sorted actions with S0 tokens (the 2D order-invariant form)
    def actions(self, b, ordered):
        ri, obj, dest, L = b["q_root"], b["q_obj"], b["q_dest"], b["q_len"]
        Q, M = obj.shape
        if not ordered:
            key = b["okey"][ri].gather(1, obj.clamp(min=0))
            o = torch.sort(torch.where(obj < 0, torch.full_like(obj, 1 << 31), key), dim=1, stable=True).indices
            obj, dest = obj.gather(1, o), dest.gather(1, o[..., None].expand(-1, -1, 2))
        ar = torch.arange(Q, device=obj.device)
        xy, x0, mask, toks = b["xy0"][ri].clone(), b["xy0"][ri], b["omask"][ri], []
        for t in range(M):
            on = (t < L)[:, None]
            i = obj[:, t].clamp(min=0)
            tok = self.encode(b, xy if ordered else x0, ri)
            xy2 = xy.clone()
            xy2[ar, i] = torch.where(on, dest[:, t], xy[ar, i])
            end = self.encode(b, xy2, ri)[ar, i]
            toks.append(self.act(torch.cat([tok[ar, i], end, self.pooled(tok, mask), xy.new_full((Q, 1), float(t))], -1)))
            xy = xy2
        return torch.stack(toks, 1), xy

    # -- ModelBC7._order (kind B, order pred: predecessor pairs u < t) / bxy_p17.Model._order (invariant: all other actions u != t)
    def order(self, b):
        pred = self.cfg["order"] == "pred"
        toks, xy = self.actions(b, pred)
        Q, M, _ = toks.shape
        L = b["q_len"]
        valid = torch.arange(M, device=toks.device)[None] < L[:, None]
        P = self.pairo(torch.cat([toks[:, None].expand(Q, M, M, W), toks[:, :, None].expand(Q, M, M, W)], -1))   # [q, t, u] = pairo(tok_u, a_t)
        t_, u_ = torch.arange(M, device=toks.device)[:, None], torch.arange(M, device=toks.device)[None]
        allow = ((u_ < t_) if pred else (u_ != t_))[None] & valid[:, None, :]
        pm = (P * allow[..., None]).sum(2) / allow.sum(-1, keepdim=True).clamp(min=1)
        rows = self.mix(torch.cat([toks, pm], -1))
        g = (rows * valid[..., None]).sum(1) / L[:, None].clamp(min=1)
        return self.order_s(torch.cat([g, self.scene_vec(b, xy, b["q_root"])], -1))[:, 0], g

    def coeffs(self, b):
        cfg = self.cfg
        c = self.ctxn(b["ctx"])
        v = self.scene_vec(b, b["s_xy"], b["s_root"])
        bb = self.scene(torch.cat([v, c[b["s_root"]]], -1))[:, 0]
        fy = self.geoY(b["phi_y"])[:, 0] if cfg["geo"] else torch.zeros_like(bb)
        bb = bb + fy
        J0 = F.softplus(self.jt(torch.cat([v[b["y0"]], c], -1))[:, 0])
        if len(b["q_root"]) == 0:
            z = bb.new_zeros(0)
            return {"b": bb, "a": z, "w": z, "J": z, "J0": J0}
        s, g = self.order(b)
        h = torch.cat([g, v[b["parent"]], c[b["q_root"]]], -1)
        a = s + (self.xs(h)[:, 0] if cfg["sa"] else 0.0)
        if cfg["geo"]:
            a = a + self.geoX(b["phi_x"] if cfg["order"] == "pred" else b["phi_x0"])[:, 0] + (fy[b["parent"]] if cfg["family"] == "direct" else 0.0)
        w = self.xy(h)[:, 0] if cfg["coupled"] else torch.zeros_like(a)
        J = F.softplus(self.jt(torch.cat([v[b["parent"]], c[b["q_root"]]], -1))[:, 0]) + F.softplus(self.cost(g)[:, 0])
        return {"b": bb, "a": a, "w": w, "J": J, "J0": J0}

    def loss(self, b, lab, lam=None, kappa=KAPPA):
        """bxy_p17.batch_loss: root-mean exact observed-data NLL (DIRECT: per-root success-event BCE sum) + lambda_J L_J + kappa SYM."""
        cfg, C, R = self.cfg, self.coeffs(b), b["n_roots"]
        lam = cfg["lam"] if lam is None else lam
        if cfg["family"] == "xy":
            nll_r = -batched_log_evidence(C["b"], C["a"], C["w"], b["parent"], lab["y_obs"], lab["x_obs"], b["s_root"], R)
        else:
            lg = torch.cat([C["a"], C["b"][b["y0"]]])
            y = torch.cat([lab["succ_q"], lab["succ0"]])
            rt = torch.cat([b["q_root"], torch.arange(R, device=lg.device)])
            m = y >= 0
            bce = F.binary_cross_entropy_with_logits(lg[m], y[m].to(lg.dtype), reduction="none")
            nll_r = torch.zeros(R, dtype=lg.dtype, device=lg.device).index_add(0, rt[m], bce)
        nll = nll_r.mean()
        tot, parts = nll, {"nll": float(nll.detach())}
        if len(lab["j_val"]):
            jq = lab["j_q"]
            jp = torch.where(jq >= 0, C["J"][jq.clamp(min=0)] if len(C["J"]) else C["J0"][lab["j_root"]], C["J0"][lab["j_root"]])
            tr = F.huber_loss(torch.log1p(jp), torch.log1p(lab["j_val"]), reduction="sum", delta=HUBER) / len(jp)
            parts["travel"] = float(tr.detach())
            if lam:
                tot = tot + lam * tr
        if cfg["loss"] == "SYM":
            aux = sym_batch(C["b"], lab["y_obs"], b["s_root"], b["y0"], R)
            parts["aux"] = float(aux.detach())
            tot = tot + kappa * aux
        return tot, parts

    @torch.no_grad()
    def predict(self, b):
        C = self.coeffs(b)
        if self.cfg["family"] == "direct":
            return {"psucc": torch.sigmoid(C["a"]), "p_noop": torch.sigmoid(C["b"][b["y0"]]), "J": C["J"], "J0": C["J0"]}
        mg = marginals(C["b"], C["a"], C["w"], b["parent"])
        return {**mg, "w": C["w"], "p_noop": mg["py"][b["y0"]], "J": C["J"], "J0": C["J0"]}


def sym_batch(b, y_obs, s_root, y0, R):
    """bxy_p17.sym (l.133-151) for every root of the batch, batch mean over roots (roots without terms contribute 0)."""
    tot = b.new_zeros(R)
    is0 = torch.zeros_like(y_obs, dtype=torch.bool)
    is0[y0] = True
    y0v = y_obs[y0][s_root]
    ed = (y_obs >= 0) & ~is0
    d = b - b[y0][s_root]
    groups = [ed & (y0v == 0) & (y_obs == 1), ed & (y0v == 1) & (y_obs == 0), ed & (y0v >= 0) & (y_obs == y0v)]
    vals = [F.softplus(-d), F.softplus(d), F.softplus(d.abs() - 1.0)]
    num, cnt = b.new_zeros(R), b.new_zeros(R)
    for gm, val in zip(groups, vals):
        n = torch.zeros(R, dtype=b.dtype, device=b.device).index_add(0, s_root, gm.to(b.dtype))
        sm = torch.zeros(R, dtype=b.dtype, device=b.device).index_add(0, s_root, torch.where(gm, val, torch.zeros_like(val)))
        num, cnt = num + torch.where(n > 0, sm / n.clamp(min=1), torch.zeros_like(sm)), cnt + (n > 0).to(b.dtype)
    tot = torch.where(cnt > 0, num / cnt.clamp(min=1), tot)
    e = torch.nonzero(ed).squeeze(1)                                          # P14 contrast over discordant EDITED pairs, mean over pairs
    if len(e):
        pos, neg = e[y_obs[e] == 1], e[y_obs[e] == 0]
        if len(pos) and len(neg):
            same = s_root[pos][:, None] == s_root[neg][None]
            sp = F.softplus(-(b[pos][:, None] - b[neg][None]))
            pr = torch.zeros(R, dtype=b.dtype, device=b.device).index_add(0, s_root[pos], (sp * same).sum(1))
            pc = torch.zeros(R, dtype=b.dtype, device=b.device).index_add(0, s_root[pos], same.sum(1).to(b.dtype))
            tot = tot + torch.where(pc > 0, pr / pc.clamp(min=1), torch.zeros_like(pr))
    return tot.mean()


def choose(rows, rule):
    """bxy_p11.choose (l.387-398) verbatim semantics: no-op gate, admission threshold eta, argmin -log p + alpha m + beta C_d + gamma J_hat."""
    adm = [i for i, r in enumerate(rows) if r["admissible"]]
    e = next(i for i in adm if rows[i]["m"] == 0)
    if rule.get("noop_gate") is not None and rows[e]["p"] >= rule["noop_gate"]:
        return e
    pool = [i for i in adm if rows[i]["p"] >= rule["eta"]]
    if not pool:
        return None
    cost = lambda r: -math.log(max(r["p"], 1e-12)) + rule["alpha"] * r["m"] + rule["beta"] * r["cd"] + rule["gamma"] * (r.get("J_hat") or 0.0)  # noqa: E731
    return min(pool, key=lambda i: (round(cost(rows[i]), 12), rows[i]["m"], rows[i]["cd"], rows[i]["sig"]))


def selftest():
    """Exact math only (no network): evidence == brute force over E_root with missing X / Y, joint sums to 1, p_succ identity."""
    worst = 0.0
    for t in range(60):
        g = torch.Generator().manual_seed(t)
        nS, nQ = int(torch.randint(1, 4, (1,), generator=g)), int(torch.randint(1, 5, (1,), generator=g))
        b, a, w = (torch.randn(n, generator=g, dtype=torch.float64) * 2 for n in (nS, nQ, nQ))
        parent = [int(torch.randint(0, nS, (1,), generator=g)) for _ in range(nQ)]
        oy = {s: int(torch.randint(0, 2, (1,), generator=g)) for s in range(nS) if torch.rand(1, generator=g) < 0.5}
        ox = {q: int(torch.randint(0, 2, (1,), generator=g)) for q in range(nQ) if torch.rand(1, generator=g) < 0.6}
        bf = brute_force(b, a, w, parent, oy, ox)
        yo, xo = torch.tensor([oy.get(s, -1) for s in range(nS)]), torch.tensor([ox.get(q, -1) for q in range(nQ)])
        bl = float(batched_log_evidence(b, a, w, torch.tensor(parent), yo, xo, torch.zeros(nS, dtype=torch.long), 1)[0])
        worst = max(worst, abs(float(log_evidence(b, a, w, parent, oy, ox)) - bf), abs(bl - bf))
    b, a, w, par = torch.randn(2, dtype=torch.float64), torch.randn(3, dtype=torch.float64), torch.randn(3, dtype=torch.float64), [0, 0, 1]
    Z = float(torch.logsumexp(torch.stack([-energy_root(b, a, w, par, y, x) for y in itertools.product((0, 1), repeat=2)
                                           for x in itertools.product((0, 1), repeat=3)]), 0))
    m = marginals(b, a, w, torch.tensor(par))
    ps = float((m["psucc"] - torch.sigmoid(b[par]) * torch.sigmoid(a + w)).abs().max())
    return {"evidence_vs_bruteforce_max_abs_err": worst, "joint_log_partition": Z, "psucc_identity_err": ps,
            "pass": worst < 1e-9 and abs(Z) < 1e-9 and ps < 1e-12}


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        print(json.dumps(selftest(), indent=1))
    else:
        raise SystemExit(__doc__)
