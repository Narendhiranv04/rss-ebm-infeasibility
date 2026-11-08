"""Shared learned 3D scene encoder, cVAE and conditional flow-matching placement samplers, nonlearned controls and a separate ranker.

Destination representation (shared by every sampler): the 6x6 shelf's support-constrained UPRIGHT placement at a cell centre
(x = 0.42 + (r-2.5)*0.075, y = (c-2.5)*0.075 m, on the shelf floor, object resting on its base; azimuth/grasp are the teacher's, not ours).
No arbitrary SE(3) pose is generated.  legal(S, o) = in-bounds cells free in the hypothetical scene S (the queried object's own cell is occupied).

cVAE (A):  posterior q_psi(z | S, o, d) = N(mu, diag sigma^2) from [h(S,o), onehot(d)];  decoder p_phi(d | z, S, o) = masked categorical over
           legal cells;  L = E_q[-log p_phi(d | z, S, o)] + beta_KL * KL(q || N(0, I)),  beta linearly warmed up.  The decoder density is a
           proposal distribution, NOT a calibrated feasibility probability.
FM (B):    support-local continuous coordinate u(d) = 2 * ((r + 0.5 + j_r, c + 0.5 + j_c) / 6) - 1 in [-1, 1]^2 with uniform within-cell jitter
           j ~ U(-0.5, 0.5)^2 (dequantization);  x_0 ~ N(0, I_2),  x_t = (1 - t) x_0 + t x_1,  u_t = x_1 - x_0,
           L = E || v_phi(x_t, t | S, o) - u_t ||^2;  sampling = Euler ODE (n_steps) from x_0, then deterministic projection to the NEAREST
           legal cell centre (raw out-of-domain / illegal-cell rates are recorded).  This is distributional interpolation, not robot motion.
Encoder:   per-object MLP + query-keyed attention pooling + mean pooling + occupancy-grid MLP (permutation-invariant over non-query objects).
           Each sampler owns its encoder (same architecture and capacity); the cVAE's posterior encoder q_psi is a separate head."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

NCELL = 36
CENTRES = torch.tensor([[2 * (r + 0.5) / 6 - 1, 2 * (c + 0.5) / 6 - 1] for r in range(6) for c in range(6)])


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n):
        layers += [nn.Linear(d, h), nn.SiLU()]; d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class SceneEncoder(nn.Module):
    def __init__(self, nfeat=14, h=128):
        super().__init__()
        self.phi = mlp(nfeat, h, h)
        self.q, self.k, self.v = nn.Linear(h, h), nn.Linear(h, h), nn.Linear(h, h)
        self.grid = mlp(NCELL, h, h, 1)
        self.out = mlp(4 * h, h, h, 1)
        self.h = h

    def forward(self, obj, mask, occ):
        e = self.phi(obj)                                                   # [B, N, h]
        e = e * mask[..., None]
        qtok = e[:, 0]                                                      # the queried object is always slot 0
        att = (self.q(qtok)[:, None] * self.k(e)).sum(-1) / math.sqrt(self.h)
        att = att.masked_fill(~mask, -1e9).softmax(-1)
        ctx = (att[..., None] * self.v(e)).sum(1)
        mean = e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.out(torch.cat([qtok, ctx, mean, self.grid(occ)], -1))


def masked_logits(logits, legal):
    return logits.masked_fill(~legal, -1e9)


class CVAE(nn.Module):
    kind = "cvae"

    def __init__(self, h=128, z=8, nfeat=14):
        super().__init__()
        self.enc, self.zdim = SceneEncoder(nfeat, h), z
        self.post = mlp(h + NCELL, h, 2 * z)
        self.dec = mlp(h + z, h, NCELL)

    def loss(self, obj, mask, occ, legal, dest, beta):
        c = self.enc(obj, mask, occ)
        mu, lv = self.post(torch.cat([c, F.one_hot(dest, NCELL).float()], -1)).chunk(2, -1)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp()
        rec = F.cross_entropy(masked_logits(self.dec(torch.cat([c, z], -1)), legal), dest)
        kl = 0.5 * (mu.pow(2) + lv.exp() - 1 - lv).sum(-1).mean()
        return rec + beta * kl, {"rec": rec.item(), "kl": kl.item()}

    @torch.no_grad()
    def sample(self, obj, mask, occ, legal, n, gen=None):
        c = self.enc(obj, mask, occ)
        B = c.shape[0]
        z = torch.randn(B, n, self.zdim, device=c.device, generator=gen)
        lg = masked_logits(self.dec(torch.cat([c[:, None].expand(B, n, -1), z], -1)), legal[:, None])
        u = torch.rand(lg.shape, device=c.device, generator=gen).clamp(1e-9, 1 - 1e-9)
        return (lg - (-u.log()).log()).argmax(-1), {"raw_illegal": torch.zeros(B, n, dtype=torch.bool, device=c.device),
                                                     "raw_out_of_domain": torch.zeros(B, n, dtype=torch.bool, device=c.device)}

    @torch.no_grad()
    def log_prob(self, obj, mask, occ, legal, dest, k=64):
        """IWAE estimate of log p(d | S, o) (proposal density of held-out positives; not a feasibility probability)."""
        c = self.enc(obj, mask, occ)
        B = c.shape[0]
        mu, lv = self.post(torch.cat([c, F.one_hot(dest, NCELL).float()], -1)).chunk(2, -1)
        z = mu[:, None] + torch.randn(B, k, self.zdim, device=c.device) * (0.5 * lv).exp()[:, None]
        lp = masked_logits(self.dec(torch.cat([c[:, None].expand(B, k, -1), z], -1)), legal[:, None]).log_softmax(-1)
        lpd = lp.gather(-1, dest[:, None, None].expand(B, k, 1)).squeeze(-1)
        lpz = -0.5 * (z.pow(2) + math.log(2 * math.pi)).sum(-1)
        lq = -0.5 * (((z - mu[:, None]) / (0.5 * lv).exp()[:, None]).pow(2) + lv[:, None] + math.log(2 * math.pi)).sum(-1)
        return torch.logsumexp(lpd + lpz - lq, 1) - math.log(k)


class TimeEmb(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.register_buffer("f", torch.exp(torch.linspace(0, math.log(100.0), d // 2)))

    def forward(self, t):
        a = t[:, None] * self.f[None] * 2 * math.pi
        return torch.cat([a.sin(), a.cos()], -1)


class FlowMatch(nn.Module):
    kind = "fm"

    def __init__(self, h=128, nfeat=14, steps=32):
        super().__init__()
        self.enc, self.temb, self.steps = SceneEncoder(nfeat, h), TimeEmb(32), steps
        self.v = mlp(2 + 32 + h, h, 2, 3)

    @staticmethod
    def coord(dest, jitter=True, gen=None):
        rc = torch.stack([dest // 6, dest % 6], -1).float() + 0.5
        if jitter:
            rc = rc + torch.rand(rc.shape, device=dest.device, generator=gen) - 0.5
        return 2 * rc / 6 - 1

    def loss(self, obj, mask, occ, legal, dest, beta=None):
        c = self.enc(obj, mask, occ)
        x1 = self.coord(dest)
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        l = (self.v(torch.cat([xt, self.temb(t), c], -1)) - (x1 - x0)).pow(2).sum(-1).mean()
        return l, {"fm": l.item()}

    @torch.no_grad()
    def sample(self, obj, mask, occ, legal, n, gen=None):
        c = self.enc(obj, mask, occ)
        B = c.shape[0]
        cc = c[:, None].expand(B, n, -1).reshape(B * n, -1)
        x = torch.randn(B * n, 2, device=c.device, generator=gen)
        for i in range(self.steps):
            t = torch.full((B * n,), i / self.steps, device=c.device)
            x = x + self.v(torch.cat([x, self.temb(t), cc], -1)) / self.steps
        x = x.view(B, n, 2)
        ood = (x.abs() > 1).any(-1)
        rc = ((x + 1) * 3).floor().clamp(0, 5).long()
        raw_cell = rc[..., 0] * 6 + rc[..., 1]
        raw_illegal = ood | ~legal.gather(1, raw_cell)
        d2 = (x[:, :, None] - CENTRES.to(x.device)[None, None]).pow(2).sum(-1)
        d2 = d2.masked_fill(~legal[:, None], float("inf"))
        return d2.argmin(-1), {"raw_illegal": raw_illegal, "raw_out_of_domain": ood}


class Uniform(nn.Module):
    """Exact legality-matched uniform control (sampled WITH replacement under the same raw budget as the learned samplers)."""
    kind = "uniform"

    def sample(self, obj, mask, occ, legal, n, gen=None):
        p = legal.float() / legal.float().sum(1, keepdim=True)
        d = torch.multinomial(p, n, replacement=True, generator=gen)
        z = torch.zeros_like(d, dtype=torch.bool)
        return d, {"raw_illegal": z, "raw_out_of_domain": z}


class BandStratified(nn.Module):
    """Legality-matched stratified control: round robin over Chebyshev bands from the source (near 1 / mid 2-3 / far 4-5, the roster's 3D bands),
    uniform inside a band; an empty band is skipped."""
    kind = "stratified"

    def sample(self, obj, mask, occ, legal, n, gen=None, src=None):
        B = legal.shape[0]
        r, c = torch.arange(36, device=legal.device) // 6, torch.arange(36, device=legal.device) % 6
        ch = torch.maximum((r[None] - (src // 6)[:, None]).abs(), (c[None] - (src % 6)[:, None]).abs())
        bands = [(ch == 1), (ch >= 2) & (ch <= 3), (ch >= 4)]
        out = torch.empty(B, n, dtype=torch.long, device=legal.device)
        for j in range(n):
            sel = legal                                                      # fallback: any legal cell
            for k in (2, 1, 0):                                              # the first non-empty band in round-robin order wins
                m = bands[(j + k) % 3] & legal
                sel = torch.where(m.any(1, keepdim=True), m, sel)
            p = sel.float()
            out[:, j] = torch.multinomial(p / p.sum(1, keepdim=True), 1, generator=gen).squeeze(1)
        z = torch.zeros_like(out, dtype=torch.bool)
        return out, {"raw_illegal": z, "raw_out_of_domain": z}


class Ranker(nn.Module):
    """Separate supervised feasibility ranker on declared labelled trials (VERIFIED vs NEG_*; UNLABELED excluded).  Its scores are a
    discriminative estimate, distinct from the generators' proposal densities; calibration is reported on VAL only."""
    kind = "ranker"

    def __init__(self, h=128, nfeat=14):
        super().__init__()
        self.enc = SceneEncoder(nfeat, h)
        self.head = mlp(h + 4, h, 1)

    def logits_all(self, obj, mask, occ):
        c = self.enc(obj, mask, occ)
        B = c.shape[0]
        ce = CENTRES.to(c.device)
        qce = 2 * (obj[:, 0, 4:6] * 5 + 0.5) / 6 - 1                         # query cell centre (columns r/5, c/5 are never standardized)
        dfeat = torch.cat([ce[None].expand(B, -1, -1), ce[None] - qce[:, None]], -1)
        return self.head(torch.cat([c[:, None].expand(B, NCELL, -1), dfeat], -1)).squeeze(-1)

    def loss(self, obj, mask, occ, legal, dest, y):
        lg = self.logits_all(obj, mask, occ).gather(1, dest[:, None]).squeeze(1)
        l = F.binary_cross_entropy_with_logits(lg, y.float())
        return l, {"bce": l.item()}


def build(kind, cfg):
    h, nf = cfg.get("hidden", 128), cfg.get("nfeat", 14)
    return {"cvae": lambda: CVAE(h, cfg.get("zdim", 8), nf), "fm": lambda: FlowMatch(h, nf, cfg.get("ode_steps", 32)),
            "ranker": lambda: Ranker(h, nf), "uniform": Uniform, "stratified": BandStratified}[kind]()
