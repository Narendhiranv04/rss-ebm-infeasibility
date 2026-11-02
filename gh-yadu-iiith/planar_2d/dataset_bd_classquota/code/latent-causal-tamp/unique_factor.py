"""ABC9U_UNIQUE_V1: canonical unique-factor conditional root model for ABC7 B / C (P9-L3a; alternative to the P9 PoE).

    P_U(z_S, x | c0, C) = P_S(z_S | c0) · exp(-E_Q(x, z_S) / T) / Z_U(z_S),   Z_U(z_S) = sum_x exp(-E_Q(x, z_S) / T)

    E_Q = sum over DISTINCT sequence queries q of  -s_q x_q
              (F2 phi_seq on q's own ordered sequence; for an omission role this IS the F4 omission unary, shared scorer)
        + sum over DISTINCT repair blocks b of  -1/2 z_b' W_b z_b - HO_b(z_b),   z_b = (z_S, x of b's slots)
              (F4 zR-zO pairs, F5 scene-repair and zO-zO pairs with b's context g_b and role types; F5 higher order: full only)

Every potential is the RAW ABC7 potential (abc7_models.ModelBC7.joint without its per-block normalizer), re-evaluated through the
model's own modules; each query's unary enters once, each block's relational terms once (exact repeated blocks collapse), one
conditional normalizer per scene state, so sum_x P_U = P_S for every catalogue. For a one-repair catalogue P_U equals the ABC7 joint.
F1 static and travel J stay outside P(z | r) with their own supervision. Exact elimination via joint_likelihood.contract.
"""
from __future__ import annotations

import itertools

import torch
from torch.nn import functional as F

import abc7_train as t7
import joint_likelihood as jl
from abc_models import T, TYPES

VERSION = "ABC9U_UNIQUE_V1"
D = torch.float64


def _states(k):
    return torch.tensor(list(itertools.product((0.0, 1.0), repeat=k)), dtype=D)


def instances(m, r) -> dict:
    """Canonical factor instances of one root for an ABC7 B / C model (one memoized pass)."""
    e = next(c for c in r["cands"] if not c["seq"])
    sn = ["z0"] + sorted(n for n in e["slots"] if n.startswith("zS"))
    qmap = dict(zip(e["slots"], e["z_q"]))
    m._memo = {}
    try:
        S = m.scene_block(r["R"], sn, r["edits"])
        ks = len(sn)
        inst = {"S": {"names": sn, "queries": [qmap[n] for n in sn], "logp": S["logP"], "Z": S["Z"]}, "unary": {}, "seq": {}, "blocks": {}}
        for c in r["cands"]:
            if not c["seq"] or c["query"] in inst["blocks"]:
                continue                                           # an exactly repeated block is one physical instance
            qn = ["zR"] + sorted(n for n in c["slots"] if n.startswith("zO"))
            qc = dict(zip(c["slots"], c["z_q"]))
            seqs = [c["seq"] if n == "zR" else [a for k, a in enumerate(c["seq"]) if k != int(n[3:-1])] for n in qn]
            g = None
            vs = []
            for n, sq in zip(qn, seqs):
                s, gq, _ = m.order(r["R"], sq)
                q = qc[n]
                key = tuple((i, tuple(map(float, d))) for i, d in sq)
                if inst["seq"].setdefault(q, key) != key:
                    raise jl.IdentityError(f"{r['key']}: query {q} names two different sequences")
                inst["unary"][q] = s
                vs.append(gq + m.typ(torch.tensor(TYPES[n[:2]])))
                g = gq if n == "zR" else g
            kq = len(qn)
            V = torch.cat([S["v"], torch.stack(vs)])
            spec = [(ks, ks + j, "omit") for j in range(1, kq)] + [(a, b, "couple") for a in range(ks) for b in range(ks, ks + kq)]
            spec += [(ks + a, ks + b, "couple") for a, b in itertools.combinations(range(1, kq), 2)]
            Z = _states(ks + kq)
            E = -0.5 * ((Z @ m._pairs(V, g, spec)) * Z).sum(1)
            if m.couple == "full":
                E = E - m.ho_q(torch.cat([Z @ m.ho_u(V), g.expand(len(Z), -1)], 1))[:, 0]
            inst["blocks"][c["query"]] = {"queries": [qc[n] for n in qn], "energy": E.reshape(2 ** ks, 2 ** kq)}
    finally:
        m._memo = None
    return inst


def _factors(inst, clamp=None, temp=T):
    """Log-potential factors over repair variables, batched over scene states; scene-tied and clamped variables resolved."""
    S, clamp = inst["S"], clamp or {}
    sidx, B = {q: i for i, q in enumerate(S["queries"])}, len(S["logp"])
    out = []
    raw = [((q,), torch.stack([s * 0, s]).expand(B, 2) / temp) for q, s in inst["unary"].items()]
    raw += [(tuple(b["queries"]), (-b["energy"] / temp).reshape([B] + [2] * len(b["queries"]))) for b in inst["blocks"].values()]
    for vs, t in raw:
        kept = list(vs)
        for j in reversed(range(len(vs))):
            q = vs[j]
            if q in sidx:
                z = S["Z"][:, sidx[q]].reshape([-1] + [1] * (t.dim() - 2)) == 1.0
                t = torch.where(z, t.select(1 + j, 1), t.select(1 + j, 0))
                kept.pop(j)
            elif q in clamp:
                t = t.select(1 + j, clamp[q])
                kept.pop(j)
        out.append((tuple(kept), t))
    return out


def log_partition(inst, clamp=None, temp=T):
    return jl.contract(_factors(inst, clamp, temp), len(inst["S"]["logp"]))


def variables(inst):
    return sorted(set(inst["unary"]) | {q for b in inst["blocks"].values() for q in b["queries"]})


def log_evidence(inst, lab, temp=T) -> tuple:
    """(log P_U(observed labels), distinct observed variables). UNKNOWN / NOT_RUN (None) are summed, never zeros."""
    S, xs = inst["S"], set(variables(inst))
    obs = {q: y for q, y in lab.items() if y is not None and (q in xs or q in S["queries"])}
    keep = torch.ones(len(S["logp"]), dtype=torch.bool)
    for i, q in enumerate(S["queries"]):
        if q in obs:
            keep &= S["Z"][:, i] == float(obs[q])
    num = log_partition(inst, {q: y for q, y in obs.items() if q in xs}, temp)
    lt = S["logp"] + num - log_partition(inst, None, temp)
    return torch.logsumexp(lt[keep], 0), len(obs)


@torch.no_grad()
def prior_marginals(inst, temp=T) -> dict:
    """P_U(variable = 1) with NO evidence (the deployable prediction; scene marginals are exactly those of P_S)."""
    S = inst["S"]
    p = torch.exp(S["logp"])
    out = {q: float(p @ S["Z"][:, i]) for i, q in enumerate(S["queries"])}
    den = log_partition(inst, None, temp)
    for q in variables(inst):
        if q not in out:
            out[q] = float(torch.exp(torch.logsumexp(S["logp"] + log_partition(inst, {q: 1}, temp) - den, 0)))
    return out


@torch.no_grad()
def conditional_marginal(inst, q, lab, temp=T) -> float:
    """OFFLINE oracle-revealed diagnostic only: P_U(q = 1 | observed labels). Never a ranking input."""
    if lab.get(q) is not None:
        return float(lab[q])
    return float(torch.exp(log_evidence(inst, lab | {q: 1}, temp)[0] - log_evidence(inst, lab, temp)[0]))


def brute_force(inst, lab, temp=T) -> float:
    """Independent reference: enumerate every distinct variable jointly (tiny roots only)."""
    S, xs = inst["S"], [q for q in variables(inst) if q not in inst["S"]["queries"]]
    tot = []
    for s in range(len(S["logp"])):
        zs = dict(zip(S["queries"], S["Z"][s].tolist()))
        rows = []
        for bits in itertools.product((0, 1), repeat=len(xs)):
            val = zs | dict(zip(xs, bits))
            e = sum(-float(u) * val[q] for q, u in inst["unary"].items())
            e += sum(float(b["energy"][s, int("".join(str(int(val[q])) for q in b["queries"]), 2)]) for b in inst["blocks"].values())
            rows.append((all(val[q] == y for q, y in lab.items() if y is not None and q in val), -e / temp))
        lz = torch.logsumexp(torch.tensor([u for _, u in rows], dtype=D), 0)
        ok = [u for m_, u in rows if m_]
        if ok:
            tot.append(float(S["logp"][s]) + float(torch.logsumexp(torch.tensor(ok, dtype=D), 0) - lz))
    return float(torch.logsumexp(torch.tensor(tot, dtype=D), 0))


def losses(m, r) -> dict:
    """{nll: -log P_U(D_c) or None, n_obs, static: endpoint-deduplicated BCE, travel: witnessed log-Huber J}."""
    inst = instances(m, r)
    ll, n = log_evidence(inst, jl.labels(r))
    seen, st, tr = set(), [], []
    m._memo = {}
    try:
        for c in r["cands"] + r.get("static_checks", []):
            if "query" in c:
                _, g, xy = m.order(r["R"], c["seq"])
            else:                                              # static-check endpoint: positions only, no scorer call
                xy = r["R"].xy0.clone()
                for i, d in c["seq"]:
                    xy[i] = torch.as_tensor(d, dtype=D)
            if (k := jl._endpoint(c["seq"])) not in seen:
                _, lu, lp = m.static(r["R"], xy)
                if (v := t7._static({"lu": lu, "lp": lp}, c)) is not None:
                    seen.add(k)
                    st.append(v)
            if c.get("cost") is not None and c["seq"]:
                tr.append(F.huber_loss(torch.log1p(F.softplus(m.cost(g)[0])), torch.log1p(torch.tensor(c["cost"], dtype=D))))
    finally:
        m._memo = None
    return {"nll": -ll if n else None, "n_obs": n, "shared": 0, "static": torch.stack(st).mean() if st else None,
            "travel": torch.stack(tr).mean() if tr else None}


def total(m, roots, lam_static=1.0, lam_travel=1.0) -> tuple:
    per = [losses(m, r) for r in roots]
    tot, rec = 0.0, {}
    for k, lam in (("nll", 1.0), ("static", lam_static), ("travel", lam_travel)):
        v = [p[k] for p in per if p[k] is not None]
        if v:
            mean = torch.stack(v).mean()
            tot, rec[k], rec[f"n_{k}"] = tot + lam * mean, float(mean.detach()), len(v)
    rec["n_obs"] = sum(p["n_obs"] for p in per)
    return tot, rec
