"""D (D_BLOCKERS_V1): a minimal conditional Boltzmann model of INITIAL target feasibility and causal blockers. Nothing else.

    P_D(y0=1, b=0 | S) = sigmoid(u)
    P_D(y0=0, b | S)   = (1 - sigmoid(u)) exp(sum_i a_i b_i + sum_{i<j} w_ij b_i b_j) / Z_B(S),   Z_B = sum_{b in {0,1}^n} exp(...)

u, a_i, w_ij come from compact MLPs over one object encoder (frozen mask-pooled DINOv2-small + geometry, the project's
abc_models.Root tensors), the target token and the robot / goal context. n = |movable pool| <= CAP: exact logsumexp enumeration,
fail closed above CAP (never subsampled). pair=False is the independent-Bernoulli control (w = 0, retrained). Fixed non-movable
bodies stay geometry context and are never blocker candidates. No destination, placement, order, recursion or execution.

Labels (blocker semantics, contract §3.1): `minimal_sets` derives, from the teacher statuses of ALL 2^n parking subsets, the
inclusion-minimal sufficient clearing sets, the union-participation mask b and the intersection `necessary`; anything incomplete,
contradictory, ineligible or non-monotone -> mask UNKNOWN (feasibility term only).
"""
from __future__ import annotations

import itertools
import math

import torch
from torch import nn
from torch.nn import functional as F

import abc_models as am
from abc_models import VIS, W, mlp

VERSION, CAP = "D_BLOCKERS_V1", 6
V, C, B = "VERIFIED_FEASIBLE", "CERTIFIED_INFEASIBLE", "BUDGET_INFEASIBLE"


class CapError(RuntimeError):
    """More movable candidates than the exact-enumeration cap."""


class ModelD(nn.Module):
    def __init__(self, pair=True, vision=True):
        super().__init__()
        self.pair = pair
        self.vproj, self.enc = (nn.Linear(384, VIS) if vision else None), mlp(14 + VIS, W)
        self.ctxn, self.pooln = mlp(16, W), nn.Linear(2 * W, W)
        self.feas, self.unary = mlp(3 * W, 1), mlp(3 * W, 1)
        if pair:
            self.pairf = mlp(3 * W, 1)

    def tokens(self, R):
        vis = self.vproj(R.vis) if self.vproj is not None else R.vis.new_zeros(R.n, VIS)
        return self.enc(torch.cat([R.numeric(R.xy0), vis], 1))

    def forward(self, R, pool):
        """(u, a [n], w [n, n] symmetric with zero diagonal) for the movable pool (object indices, any order)."""
        if len(pool) > CAP:
            raise CapError(f"{len(pool)} movable candidates exceed the exact-enumeration cap {CAP}")
        tok, c = self.tokens(R), self.ctxn(R.ctx)
        t = tok[R.tid]
        u = self.feas(torch.cat([self.pooln(torch.cat([tok.mean(0), tok.max(0).values])), t, c]))[0]
        x = tok[list(pool)] if pool else tok.new_zeros(0, W)
        a = self.unary(torch.cat([x, t.expand(len(pool), -1), c.expand(len(pool), -1)], 1))[:, 0] if pool else tok.new_zeros(0)
        w = tok.new_zeros(len(pool), len(pool))
        if self.pair and len(pool) > 1:
            i, j = torch.tensor(list(itertools.combinations(range(len(pool)), 2))).T
            cc = c.expand(len(i), -1)
            s = 0.5 * (self.pairf(torch.cat([x[i] + x[j], (x[i] - x[j]).abs(), cc], 1)) +
                       self.pairf(torch.cat([x[j] + x[i], (x[j] - x[i]).abs(), cc], 1)))[:, 0]
            w = w.index_put((torch.cat([i, j]), torch.cat([j, i])), torch.cat([s, s]))
        return u, a, w


def states(n):
    return torch.tensor(list(itertools.product((0.0, 1.0), repeat=n)), dtype=torch.float64).reshape(2 ** n, n)


def blocker_logp(a, w):
    """log P(b | y0 = 0) for all 2^n states (rows of states(n)) by exact logsumexp."""
    Z = states(len(a))
    s = Z @ a + 0.5 * ((Z @ w) * Z).sum(1)
    return s - torch.logsumexp(s, 0)


def nll(u, a, w, y0, b=None, lam=1.0):
    """BCE(sigmoid u, y0) + lam * 1[y0 = 0 and mask complete] * (-log P(b | y0 = 0)); b None = incomplete mask."""
    out = F.binary_cross_entropy_with_logits(u, torch.tensor(float(y0), dtype=torch.float64))
    if y0 == 0 and b is not None and len(a):
        idx = int(sum(int(v) << (len(b) - 1 - k) for k, v in enumerate(b)))
        out = out - lam * blocker_logp(a, w)[idx]
    return out


@torch.no_grad()
def predict(model, R, pool) -> dict:
    """p_feasible, conditional marginals P(b_i = 1 | y0 = 0), unconditional (1 - p) * marginal, conditional MAP set."""
    u, a, w = model(R, pool)
    p = float(torch.sigmoid(u))
    if not len(a):
        return {"p_feasible": p, "marg": [], "uncond": [], "map": []}
    lp, Z = blocker_logp(a, w), states(len(a))
    m = (lp.exp()[:, None] * Z).sum(0)
    return {"p_feasible": p, "marg": [float(x) for x in m], "uncond": [float((1 - p) * x) for x in m],
            "map": [int(x) for x in Z[int(torch.argmax(lp))]]}


# ------------------------------------------------------------------ ground truth from teacher statuses (independent of the collector)
def minimal_sets(n, status) -> dict:
    """status: {frozenset of pool positions: teacher status} for subsets of range(n) (frozenset() = original task).
    Returns {type, y0, mask, necessary, sets, provenance}; mask None = UNKNOWN (train feasibility only)."""
    subs = [frozenset(c) for k in range(n + 1) for c in itertools.combinations(range(n), k)]
    y = {s: 1 if status.get(s) == V else 0 if status.get(s) in (C, B) else None for s in subs}
    y0 = y[frozenset()]
    prov = "operational" if any(status.get(s) == B for s in subs if y[s] is not None) else "proof_grounded"
    if y0 == 1:
        return {"type": "FEASIBLE_ORIGINAL", "y0": 1, "mask": [0] * n, "necessary": [0] * n, "sets": [], "provenance": prov}
    if y0 is None:
        return {"type": "UNKNOWN_ORIGINAL", "y0": None, "mask": None, "necessary": None, "sets": [], "provenance": None}
    if any(v is None for v in y.values()):
        return {"type": "INCOMPLETE", "y0": 0, "mask": None, "necessary": None, "sets": [], "provenance": prov}
    if any(y[s] == 1 and y[t] == 0 for s in subs for t in subs if s < t):
        return {"type": "NON_MONOTONE", "y0": 0, "mask": None, "necessary": None, "sets": [], "provenance": prov}
    sets = [s for s in subs if y[s] == 1 and all(y[t] == 0 for t in subs if t < s)]
    if not sets:
        return {"type": "NO_CLEARING_SET_WITHIN_POOL", "y0": 0, "mask": [0] * n, "necessary": [0] * n, "sets": [], "provenance": prov}
    used = [s for s in sets]
    sound = all(status[t] == C for s in used for t in subs if t < s)
    return {"type": "MINIMAL_SETS", "y0": 0, "mask": [int(any(i in s for s in sets)) for i in range(n)],
            "necessary": [int(all(i in s for s in sets)) for i in range(n)], "sets": [sorted(s) for s in sets],
            "provenance": "proof_grounded" if sound and prov == "proof_grounded" else "operational"}


# ------------------------------------------------------------------ fixed non-learned geometry heuristic (control ii)
def heuristic(R, pool, adj=0.085, margin=0.12) -> list:
    """1 if the object is grasp-adjacent to the target (centre distance <= adj) or sits inside the link-0 annulus at an angle
    strictly between the start link-0 angle and the target angle (sweep occlusion); else 0. No learning, no robot model."""
    base, xy = R.base, R.xy0
    ang = lambda p: math.atan2(float(p[1] - base[1]), float(p[0] - base[0]))  # noqa: E731
    a0, at = math.atan2(float(R.ctx[0]), float(R.ctx[3])), ang(xy[R.tid])
    lo, hi = sorted((a0, at))
    rt = float((xy[R.tid] - base).norm())
    out = []
    for i in pool:
        d, r = float((xy[i] - xy[R.tid]).norm()), float((xy[i] - base).norm())
        out.append(int(d <= adj or (r < rt - 0.04 and lo + margin < ang(xy[i]) < hi - margin)))
    return out


def root_tensors(rec):
    return am.Root(rec)
