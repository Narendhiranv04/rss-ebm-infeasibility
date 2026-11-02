"""G4E-ABC: the three energy architectures of Methodology v1.7 on one shared representation.

Shared (identical code for A, B, C): frozen DINOv2 object features -> trainable projection, numeric
object/robot/goal descriptors, width-128 object tokens, nonnegative unary U and symmetric pair V
endpoint energies (e = softplus(-logit)), ideal-prefix action tokens.
  A  LOCAL-SEQUENCE: GRU over action tokens -> task logit u, travel J;
     E_A(r) = sum U + sum V + softplus(-u).
  B  JOINT-CAUSAL-EBM: explicit binary z over the Z_LAYOUT_V2 active slots of r;
     E_B(r,z) = E_static(r) - sum_k s_k z_k - sum_{k<l} W_kl(r) z_k z_l, with scene potentials s_z0,
     s_zS from (edited) scenes, order potentials s_zR, s_zO from PREFIX-PAIR interactions (no GRU),
     and a learned coupling W(r) between typed slots conditioned on the repair. Exact enumeration.
  C  HYBRID-JOINT: B's z, scene and coupling factors; B's order factor REPLACED by A's GRU.
`coupling=False` gives the B/C-no-coupling ablations; `ordered=False` gives A-no-order.
"""
from __future__ import annotations

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

W, VIS, T = 128, 32, 1.0
TYPES = {"z0": 0, "zS": 1, "zR": 2, "zO": 3}


def mlp(i, o, h=W):
    return nn.Sequential(nn.Linear(i, h), nn.SiLU(), nn.Linear(h, o))


class Root:
    """Learner inputs of one root as tensors (never any privileged field)."""

    def __init__(self, rec):
        x = {k: torch.as_tensor(v, dtype=torch.float64) for k, v in rec["inputs"].items()}
        self.xy0, self.vis, self.n = x["object_xy"], x["visual"], len(x["object_xy"])
        self.flags = torch.stack([x["object_radius"], x["is_active_target"], x["is_obstacle"]], 1)
        self.tid = int(torch.argmax(x["is_active_target"]))
        self.base, self.goal = x["robot_fk_joints"][0], x["goal_xy"]
        self.ctx = torch.cat([torch.sin(x["robot_q"]), torch.cos(x["robot_q"]), self.goal, x["robot_fk_joints"].reshape(-1)])
        self.pairs = list(itertools.combinations(range(self.n), 2))

    def numeric(self, xy):
        def rel(p):
            d = xy - p
            return torch.cat([d, d.norm(dim=1, keepdim=True)], 1)
        b = xy - self.base
        ang = torch.atan2(b[:, 1], b[:, 0])
        return torch.cat([xy, self.flags, b.norm(dim=1, keepdim=True), torch.sin(ang)[:, None], torch.cos(ang)[:, None],
                          rel(self.goal), rel(xy[self.tid])], 1)


class Shared(nn.Module):
    def __init__(self, vision=True):
        super().__init__()
        self.vproj, self.enc = (nn.Linear(384, VIS) if vision else None), mlp(14 + VIS, W)   # vision=False: geometry-only arm
        self.ctxn, self.pooln = mlp(16, W), nn.Linear(2 * W, W)
        self.act, self.unary, self.pairh = mlp(3 * W + 1, W), mlp(2 * W, 1), mlp(2 * W + 1, 1)

    def encode(self, R, xy):
        vis = self.vproj(R.vis) if self.vproj is not None else R.vis.new_zeros(R.n, VIS)
        return self.enc(torch.cat([R.numeric(xy), vis], 1))

    def pooled(self, tok):
        return self.pooln(torch.cat([tok.mean(0), tok.max(0).values]))

    def static(self, R, xy):
        """(E_static, unary logits [n], pair logits [n(n-1)/2]); pair logits symmetric in (i, j)."""
        tok, c = self.encode(R, xy), self.ctxn(R.ctx)
        lu = self.unary(torch.cat([tok, c.expand(R.n, -1)], 1))[:, 0]
        if not R.pairs:
            return F.softplus(-lu).sum(), lu, lu.new_zeros(0)
        i, j = (torch.tensor(v) for v in zip(*R.pairs))
        d = (xy[i] - xy[j]).norm(dim=1, keepdim=True)
        lp = 0.5 * (self.pairh(torch.cat([tok[i], tok[j], d], 1)) + self.pairh(torch.cat([tok[j], tok[i], d], 1)))[:, 0]
        return F.softplus(-lu).sum() + F.softplus(-lp).sum(), lu, lp

    def actions(self, R, seq, ordered=True):
        """Ideal-prefix action tokens: (tokens [m, W], final xy, final pooled scene)."""
        xy, out = R.xy0.clone(), []
        for t, (i, dest) in enumerate(sorted(seq) if not ordered else seq):
            tok = self.encode(R, xy if ordered else R.xy0)
            xy2 = xy.clone()
            xy2[i] = torch.as_tensor(dest, dtype=torch.float64)
            end = self.encode(R, xy2)[i]
            out.append(self.act(torch.cat([tok[i], end, self.pooled(tok), torch.tensor([float(t)], dtype=torch.float64)])))
            xy = xy2
        return (torch.stack(out) if out else torch.zeros(0, W, dtype=torch.float64)), xy, self.pooled(self.encode(R, xy))


class ModelA(Shared):
    kind = "A"

    def __init__(self, ordered=True, vision=True, extra=0):
        super().__init__(vision)
        self.ordered = ordered
        self.h0, self.gru, self.task, self.cost = nn.Linear(W, W), nn.GRUCell(W, W), mlp(2 * W, 1), mlp(W, 1)
        if extra:                                        # capacity-matched A: residual MLP on the GRU state (same factor types)
            self.extra = mlp(W, W, h=extra)

    def _hx(self, h):
        return h + self.extra(h) if hasattr(self, "extra") else h

    def seq(self, R, seq):
        toks, xy, pool = self.actions(R, seq, self.ordered)
        h = torch.tanh(self.h0(self.ctxn(R.ctx)))
        for a in toks:
            h = self.gru(a[None], h[None])[0]
        h = self._hx(h)
        u = self.task(torch.cat([h, pool]))[0]
        e_st, lu, lp = self.static(R, xy)
        j = F.softplus(self.cost(h)[0]) * (len(seq) > 0)
        return {"u": u, "p": torch.sigmoid(u), "E": e_st + F.softplus(-u), "E_static": e_st, "J": j, "lu": lu, "lp": lp}

    def scene_p(self, R, xy):
        """A's explanation by re-scoring: task probability of the empty repair under edited poses."""
        h = self._hx(torch.tanh(self.h0(self.ctxn(R.ctx))))
        return torch.sigmoid(self.task(torch.cat([h, self.pooled(self.encode(R, xy))]))[0])


class ModelBC(Shared):
    def __init__(self, kind="B", coupling=True, vision=True, measure="raw", higher_order=False):
        super().__init__(vision)
        self.kind, self.coupling, self.measure = kind, coupling, measure
        if coupling and higher_order:                    # P6: smallest higher-order coupling residual phi_HO(r, z)
            self.ho_u, self.ho = nn.Linear(W, W), mlp(2 * W, 1)
        self.scene, self.typ, self.cpl = mlp(2 * W, 1), nn.Embedding(4, W), mlp(3 * W, 1)
        self.order_s, self.cost = mlp(2 * W, 1), mlp(W, 1)
        if kind == "B":
            self.pairo, self.mix = mlp(2 * W, W), mlp(2 * W, W)
        else:
            self.h0, self.gru = nn.Linear(W, W), nn.GRUCell(W, W)

    def order(self, R, seq):
        """(s, repair summary g): B = prefix-pair interactions, C = GRU (replaces, never adds).
        Memoized within one forward pass (abc_train.forward sets / clears `_memo`): exact reuse."""
        memo, key = self.__dict__.get("_memo"), ("order", tuple((i, tuple(map(float, d))) for i, d in seq))
        if memo is not None and key in memo:
            return memo[key]
        out = self._order(R, seq)
        if memo is not None:
            memo[key] = out
        return out

    def _order(self, R, seq):
        toks, xy, pool = self.actions(R, seq)
        if self.kind == "B":
            rows = [self.mix(torch.cat([a, torch.stack([self.pairo(torch.cat([toks[u], a])) for u in range(t)]).mean(0) if t
                                        else torch.zeros(W, dtype=torch.float64)])) for t, a in enumerate(toks)]
            g = torch.stack(rows).mean(0) if rows else torch.zeros(W, dtype=torch.float64)
        else:
            g = torch.tanh(self.h0(self.ctxn(R.ctx)))
            for a in toks:
                g = self.gru(a[None], g[None])[0]
        return self.order_s(torch.cat([g, pool]))[0], g, xy

    def slots(self, R, seq, z_slots, edits):
        """Per active slot: (logit s_k, slot vector v_k); plus repair summary, final xy, travel J."""
        c = self.ctxn(R.ctx)
        s_r, g, xy = self.order(R, seq)
        s, v = [], []
        for name in z_slots:
            typ = name.split("[")[0]
            if typ in ("z0", "zS"):
                memo = self.__dict__.get("_memo")
                if memo is None or ("scene", name) not in memo:
                    pxy = R.xy0.clone()
                    for i, p in (edits[name[3:-1]] if typ == "zS" else {}).items():
                        pxy[i] = torch.as_tensor(p, dtype=torch.float64)
                    pv = self.pooled(self.encode(R, pxy))
                    sv = (self.scene(torch.cat([pv, c]))[0], pv)
                    if memo is not None:
                        memo[("scene", name)] = sv
                sv = sv if memo is None else memo[("scene", name)]
                s.append(sv[0])
                pv = sv[1]
            elif typ == "zR":
                s.append(s_r)
                pv = g
            else:
                t = int(name[3:-1])
                s_o, pv, _ = self.order(R, [a for k, a in enumerate(seq) if k != t])
                s.append(s_o)
            v.append(pv + self.typ(torch.tensor(TYPES[typ])))
        return torch.stack(s), torch.stack(v), g, xy, F.softplus(self.cost(g)[0]) * (len(seq) > 0)

    def joint(self, R, seq, z_slots, edits):
        """Exact joint over the active slots: energies of all 2^K states and derived quantities."""
        s, v, g, xy, j = self.slots(R, seq, z_slots, edits)
        e_st, lu, lp = self.static(R, xy)
        K = len(z_slots)
        Wm = torch.zeros(K, K, dtype=torch.float64)
        if self.coupling and K > 1:                      # all pairs in one batched call (same symmetric W(r))
            ia, ib = torch.tensor(list(itertools.combinations(range(K), 2))).T
            gg = g.expand(len(ia), -1)
            w = 0.5 * (self.cpl(torch.cat([v[ia], v[ib], gg], 1)) + self.cpl(torch.cat([v[ib], v[ia], gg], 1)))[:, 0]
            Wm = Wm.index_put((torch.cat([ia, ib]), torch.cat([ib, ia])), torch.cat([w, w]))
        Z = torch.tensor(list(itertools.product((0.0, 1.0), repeat=K)), dtype=torch.float64)
        E = e_st - Z @ s - 0.5 * ((Z @ Wm) * Z).sum(1)
        if hasattr(self, "ho"):                          # nonlinear in z: can represent interactions of order > 2
            E = E - self.ho(torch.cat([Z @ self.ho_u(v), g.expand(len(Z), -1)], 1))[:, 0]
        if self.measure == "uniform":                    # Z_LAYOUT_V3_R4 reference measure: E + T K ln 2
            E = E + T * K * math.log(2.0)
        lse = torch.logsumexp(-E / T, 0)
        p = torch.exp(-E / T - lse)
        marg = p @ Z
        return {"E_states": E, "Z": Z, "lse": lse, "E_bar": -T * lse, "marg": marg, "s": s, "W": Wm, "E_static": e_st,
                "lu": lu, "lp": lp, "J": j, "slots": list(z_slots)}


def build(name: str, vision=True, measure="raw", higher_order=False, extra=0):
    """Defaults reproduce the P5 architectures exactly (P5 checkpoints load unchanged)."""
    bc = dict(vision=vision, measure=measure, higher_order=higher_order)
    return {"A": lambda: ModelA(vision=vision, extra=extra), "A-no-order": lambda: ModelA(ordered=False, vision=vision),
            "B": lambda: ModelBC("B", **bc), "B-no-coupling": lambda: ModelBC("B", False, **bc), "C": lambda: ModelBC("C", **bc),
            "C-no-coupling": lambda: ModelBC("C", False, **bc)}[name]().double()
