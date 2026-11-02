"""P7 (versioned "ABC7"): A / B / C with edit-specific scene evidence and a root-consistent exact joint.

Changes against the P6 parameterization (abc_models.py, unchanged and still loadable):
  * EDIT-SPECIFIC SCENE EVIDENCE. Object tokens of a hypothetical fixed-robot pose edit carry, per object,
    [moved, dx, dy, |d|, x0, y0] (computable from c0 and the declared edit only); the edit vector e_I pools
    all tokens, the MOVED tokens and the target token. P6 pooled mean/max of the edited scene only, which
    moved by 0.04-0.2 % for A / B / A+B (runs/p7/audit).
  * ROOT-CONSISTENT JOINT. z = (z_s, z_q), z_s = (z0, zS[.]) root-scoped, z_q = (zR, zO[.]) repair-scoped:
        E(r, z) = Ebar_R(r) + [E_S(z_s) + T log Z_S] + [E_Q(z_q; z_s, r) + T log Z_Q(z_s, r)]
    so P(z | r) = P_S(z_s | c0) P_Q(z_q | z_s, r, c0) exactly, scene beliefs are identical for every r of a
    root, cross-outcome coupling lives in E_S (scene pairs, context c0) and E_Q (repair and scene-repair
    pairs, context g(r)), and -T log sum_z exp(-E/T) = Ebar_R(r) = E_static(r) + T softplus(-l_T(r)) with
    l_T the DERIVED logit of P(zR = 1 | r) (P(z0 = 1) for the empty repair). Ebar is per-r normalized, so no
    entropic preference from the number of active slots (the P6 T K ln 2 correction is not needed here).
  * NAMED FACTORS. phi_static (U, V); phi_seq (s_R: B prefix-pair, C GRU, never both); phi_scene (s_I and the
    z0-zS[I] pair); phi_omit (s_O[t] from the RECOMPUTED omitted prefix and the zR-zO[t] pair); phi_couple
    (all other pairs + higher-order residual). couple = "full" | "quadratic" (no phi_HO) | "structured"
    (phi_scene / phi_omit pairs only) | "none" (no cross-outcome term at all: independent outcome fields).
  * S: a supervised NON-ENERGY sequence classifier (DeepSets over ideal-prefix tokens) as a control.
Travel cost J is a separate readout of the repair summary, never a function of z.
"""
from __future__ import annotations

import itertools

import torch
from torch import nn
from torch.nn import functional as F

import abc_models as am
from abc_models import TYPES, T, W, mlp

VERSION = "ABC7_V1"
COUPLE = ("full", "quadratic", "structured", "none")


class Shared7(am.Shared):
    def __init__(self, vision=True):
        super().__init__(vision)
        self.edit_proj, self.edit_head = nn.Linear(6, W), mlp(3 * W + 1, W)

    def scene_vec(self, R, xy):
        """Edit-aware vector of a fixed-robot scene state xy relative to the initial poses (root-scoped)."""
        d = xy - R.xy0
        moved = (d.norm(dim=1) > 1e-9).double()
        tok = self.encode(R, xy) + self.edit_proj(torch.cat([moved[:, None], d, d.norm(dim=1, keepdim=True), R.xy0], 1))
        mv = (tok * moved[:, None]).sum(0) / moved.sum().clamp(min=1.0)
        return self.edit_head(torch.cat([self.pooled(tok), mv, tok[R.tid], moved.sum()[None] / 3.0]))

    def edited(self, R, poses):
        xy = R.xy0.clone()
        for i, p in poses.items():
            xy[i] = torch.as_tensor(p, dtype=torch.float64)
        return xy

    def memo(self, key, fn):
        m = self.__dict__.get("_memo")
        if m is None:
            return fn()
        if key not in m:
            m[key] = fn()
        return m[key]


class ModelA7(Shared7):
    """A: GRU task logit over ideal-prefix tokens; scene explanations by re-scoring edited scenes."""
    kind = "A"

    def __init__(self, ordered=True, vision=True, extra=0):
        super().__init__(vision)
        self.ordered = ordered
        self.h0, self.gru, self.task, self.cost = nn.Linear(W, W), nn.GRUCell(W, W), mlp(2 * W, 1), mlp(W, 1)
        if extra:
            self.extra = mlp(W, W, h=extra)

    def _h(self, R, toks):
        h = torch.tanh(self.h0(self.ctxn(R.ctx)))
        for a in toks:
            h = self.gru(a[None], h[None])[0]
        return h + self.extra(h) if hasattr(self, "extra") else h

    def seq(self, R, seq):
        toks, xy, _ = self.actions(R, seq, self.ordered)
        h = self._h(R, toks)
        u = self.task(torch.cat([h, self.scene_vec(R, xy)]))[0]
        e_st, lu, lp = self.static(R, xy)
        return {"u": u, "p": torch.sigmoid(u), "E": e_st + T * F.softplus(-u), "E_static": e_st, "lu": lu, "lp": lp,
                "J": F.softplus(self.cost(h)[0]) * (len(seq) > 0)}

    def scene_logit(self, R, poses):
        return self.task(torch.cat([self._h(R, []), self.scene_vec(R, self.edited(R, poses))]))[0]


class ModelS7(Shared7):
    """S: supervised non-energy control. DeepSets over ideal-prefix tokens (order enters only via the
    prefix scenes), same features; trained by BCE only (no energy, no pool term)."""
    kind = "S"

    def __init__(self, vision=True):
        super().__init__(vision)
        self.head, self.cost = mlp(3 * W, 1), mlp(W, 1)

    def seq(self, R, seq):
        toks, xy, _ = self.actions(R, seq)
        s = toks.mean(0) if len(seq) else torch.zeros(W, dtype=torch.float64)
        u = self.head(torch.cat([s, self.scene_vec(R, xy), self.ctxn(R.ctx)]))[0]
        e_st, lu, lp = self.static(R, xy)
        return {"u": u, "p": torch.sigmoid(u), "E": e_st + T * F.softplus(-u), "E_static": e_st, "lu": lu, "lp": lp,
                "J": F.softplus(self.cost(s)[0]) * (len(seq) > 0)}

    def scene_logit(self, R, poses):
        return self.head(torch.cat([torch.zeros(W, dtype=torch.float64), self.scene_vec(R, self.edited(R, poses)), self.ctxn(R.ctx)]))[0]


class ModelBC7(Shared7):
    def __init__(self, kind="B", couple="full", vision=True, order="pred", drop=()):
        super().__init__(vision)
        assert couple in COUPLE and order in ("pred", "none") and set(drop) <= {"pair_scene", "pair_omit"}
        self.kind, self.couple, self.order_mode = kind, couple, order
        self.scene, self.typ, self.order_s, self.cost = mlp(2 * W, 1), nn.Embedding(4, W), mlp(2 * W, 1), mlp(W, 1)
        if couple != "none":
            self.pair_scene, self.pair_omit = mlp(3 * W, 1), mlp(3 * W, 1)
        if couple in ("full", "quadratic"):
            self.cpl = mlp(3 * W, 1)
        if couple == "full":
            self.ho_u, self.ho_s, self.ho_q = nn.Linear(W, W), mlp(2 * W, 1), mlp(2 * W, 1)
        for name in drop:                                 # P10c retrained factor ablation (F3 scene pair / F4 omission pair)
            delattr(self, name)
        if kind == "B" and order == "none":              # P10c F2 ablation: order-invariant repair summary (no predecessor term)
            pass
        elif kind == "B":
            self.pairo, self.mix = mlp(2 * W, W), mlp(2 * W, W)
        else:
            self.h0, self.gru = nn.Linear(W, W), nn.GRUCell(W, W)

    def order(self, R, seq):
        """phi_seq logit s and repair summary g for an ordered move list (memoized per forward pass)."""
        return self.memo(("order", tuple((i, tuple(map(float, d))) for i, d in seq)), lambda: self._order(R, seq))

    def _order(self, R, seq):
        toks, xy, _ = self.actions(R, seq, getattr(self, "order_mode", "pred") == "pred")
        if getattr(self, "order_mode", "pred") == "none":  # sorted, prefix-free tokens and their mean: invariant to action order
            g = toks.mean(0) if len(toks) else torch.zeros(W, dtype=torch.float64)
        elif self.kind == "B":                           # prefix-conditioned predecessor interactions, no GRU
            rows = [self.mix(torch.cat([a, torch.stack([self.pairo(torch.cat([toks[u], a])) for u in range(t)]).mean(0) if t
                                        else torch.zeros(W, dtype=torch.float64)])) for t, a in enumerate(toks)]
            g = torch.stack(rows).mean(0) if rows else torch.zeros(W, dtype=torch.float64)
        else:                                            # GRU replaces B's prefix factor (never added to it)
            g = torch.tanh(self.h0(self.ctxn(R.ctx)))
            for a in toks:
                g = self.gru(a[None], g[None])[0]
        return self.order_s(torch.cat([g, self.scene_vec(R, xy)]))[0], g, xy

    def scene_block(self, R, names, edits):
        """Root-scoped scene block: names (z0 first), logits, slot vectors, E_S over all 2^Ks states."""
        def build():
            c = self.ctxn(R.ctx)
            vs = [self.scene_vec(R, self.edited(R, edits[n[3:-1]] if n != "z0" else {})) for n in names]
            s = torch.stack([self.scene(torch.cat([v, c]))[0] for v in vs])
            v = torch.stack([x + self.typ(torch.tensor(TYPES[n[:2]])) for x, n in zip(vs, names)])
            Z = torch.tensor(list(itertools.product((0.0, 1.0), repeat=len(names))), dtype=torch.float64)
            Wm = self._pairs(v, c, [(0, k, "scene") for k in range(1, len(names))] +
                             [(a, b, "couple") for a, b in itertools.combinations(range(1, len(names)), 2)])
            E = -Z @ s - 0.5 * ((Z @ Wm) * Z).sum(1)
            if self.couple == "full":
                E = E - self.ho_s(torch.cat([Z @ self.ho_u(v), c.expand(len(Z), -1)], 1))[:, 0]
            return {"names": list(names), "s": s, "v": v, "Z": Z, "E": E, "logP": -E / T - torch.logsumexp(-E / T, 0)}
        return self.memo(("scene", tuple(names)), build)

    def _pairs(self, v, ctx, spec):
        K = len(v)
        Wm = torch.zeros(K, K, dtype=torch.float64)
        net = {"scene": getattr(self, "pair_scene", None), "omit": getattr(self, "pair_omit", None), "couple": getattr(self, "cpl", None)}
        for kind, f in net.items():                      # one batched call per named pair factor (symmetric in a, b)
            ab = [(a, b) for a, b, k in spec if k == kind]
            if f is None or not ab:
                continue
            ia, ib = torch.tensor(ab).T
            cc = ctx.expand(len(ia), -1)
            w = 0.5 * (f(torch.cat([v[ia], v[ib], cc], 1)) + f(torch.cat([v[ib], v[ia], cc], 1)))[:, 0]
            Wm = Wm.index_put((torch.cat([ia, ib]), torch.cat([ib, ia])), torch.cat([w, w]), accumulate=True)
        return Wm

    def joint(self, R, seq, z_slots, edits):
        """Exact P(z | r) = P_S(z_s) P_Q(z_q | z_s, r) over the active slots, marginals, Ebar_R, J."""
        sn = ["z0"] + sorted(n for n in z_slots if n.startswith("zS"))
        qn = [n for n in z_slots if n == "zR"] + sorted(n for n in z_slots if n.startswith("zO"))
        if sorted(sn + qn) != sorted(z_slots):
            raise ValueError(f"unexpected slots {z_slots}")
        S = self.scene_block(R, sn, edits)
        s_r, g, xy = self.order(R, seq)
        s, v = [], []
        for n in qn:
            if n == "zR":
                s.append(s_r)
                v.append(g + self.typ(torch.tensor(TYPES["zR"])))
            else:
                so, go, _ = self.order(R, [a for k, a in enumerate(seq) if k != int(n[3:-1])])
                s.append(so)
                v.append(go + self.typ(torch.tensor(TYPES["zO"])))
        ks, kq = len(sn), len(qn)
        if kq:
            s, vq = torch.stack(s), torch.stack(v)
            V = torch.cat([S["v"], vq])
            spec = [(ks, ks + j, "omit") for j in range(1, kq)] + [(a, b, "couple") for a in range(ks) for b in range(ks, ks + kq)]
            spec += [(ks + a, ks + b, "couple") for a, b in itertools.combinations(range(1, kq), 2)]
            Wm = self._pairs(V, g, spec)
            Z = torch.tensor(list(itertools.product((0.0, 1.0), repeat=ks + kq)), dtype=torch.float64)
            Eq = -Z[:, ks:] @ s - 0.5 * ((Z @ Wm) * Z).sum(1)
            if self.couple == "full":
                Eq = Eq - self.ho_q(torch.cat([Z @ self.ho_u(V), g.expand(len(Z), -1)], 1))[:, 0]
            Eq = Eq.reshape(2 ** ks, 2 ** kq)
            logPq = -Eq / T - torch.logsumexp(-Eq / T, 1, keepdim=True)
            logP = (S["logP"][:, None] + logPq).reshape(-1)
        else:
            Z, logPq, logP = S["Z"], None, S["logP"]
        p = torch.exp(logP)
        marg_all = p @ Z
        order = sn + qn
        marg = torch.stack([marg_all[order.index(n)] for n in z_slots])
        pt = marg_all[ks] if kq else marg_all[0]
        lt = torch.log(pt.clamp(1e-12)) - torch.log1p(-pt.clamp(max=1 - 1e-12))
        e_st, lu, lp = self.static(R, xy)
        return {"slots": list(z_slots), "order": order, "ks": ks, "Z": Z, "logP": logP, "logP_S": S["logP"], "logP_Q": logPq,
                "marg": marg, "p": pt, "u": lt, "E": e_st + T * F.softplus(-lt), "E_static": e_st, "lu": lu, "lp": lp,
                "J": F.softplus(self.cost(g)[0]) * (len(seq) > 0), "scene_marg": dict(zip(sn, S["Z"].T @ torch.exp(S["logP"])))}

    def scene_logit(self, R, poses):                     # not used for B/C effects (marginals are), kept for API parity
        raise NotImplementedError


def build(name: str, vision=True, extra=0):
    """A, A-no-order, S, B / C with couple in {full, quadratic, structured, none} as 'B', 'B-quadratic', ..."""
    base, _, mode = name.partition("-")
    if base in ("B", "C"):
        mode = {"": "full", "no-coupling": "none"}.get(mode, mode)
        return ModelBC7(base, mode, vision).double()
    if base == "S":
        return ModelS7(vision).double()
    return ModelA7(ordered=mode != "no-order", vision=vision, extra=extra).double()
