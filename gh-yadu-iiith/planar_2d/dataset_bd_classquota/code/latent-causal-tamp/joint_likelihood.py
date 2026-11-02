"""ABC9_JOINTNLL_V1: exact, equality-aware, masked observed-data ROOT likelihood for the ABC7 joint (P9-L1).

Root outcome model (Methodology v1.8 6.2, with the explicit equality constraint it requires for reused queries):
    P(z_S, x | c0, C) = P_S(z_S | c0) * [prod_r P_Q,r(z_Q,r | z_S, r, c0)] * [prod_q 1(every copy of query q agrees)] / Z_eq(z_S)
  * ONE variable per distinct physical query: z0 (= zR[empty], the empty repair has no zR slot in ABC7), each eligible scene
    edit, and each sequence query x_q, whether it occurs as the complete repair zR[r'] or as the omission zO[r, t] of one or
    several other repairs (zO[r, t] of r and zR of r minus t are the same teacher query);
  * Z_eq(z_S) = sum_x prod_r P_Q,r renormalizes the per-repair conditional blocks (experts) on the event that all copies of
    a query agree, separately for every scene state, so the scene marginal stays EXACTLY P_S (root consistency);
  * without reused queries Z_eq = 1 and this is the guide's rootwise evidence (distinct repairs conditionally independent
    given z_S, a modelling assumption, not a physical claim).
log P(D_c) = logsumexp over observed-consistent z_S of [log P_S(z_S) + log N(z_S) - log Z_eq(z_S)], where N sums the same
product over every unobserved variable consistent with the observed labels. Both sums are exact variable elimination over the
shared query variables, batched over the 2^K_S scene states. UNKNOWN / NOT_RUN labels are None: marginalized, never zeros.
Static energy (r-only) and travel J are outside P(z | r); they are trained by their own two auxiliary terms.
"""
from __future__ import annotations

import collections
import itertools

import torch
from torch.nn import functional as F

import abc7_train as t7

VERSION = "ABC9_JOINTNLL_V1"
MAX_SCOPE = 16                                          # largest intermediate elimination scope (2^16 states per scene state)
PEAK = [0]                                              # largest scope actually used (diagnostic)


class IdentityError(ValueError):
    """Two copies of one physical query carry different decided labels (hard data-integrity failure)."""


class CombinatoricsError(RuntimeError):
    """Exact elimination would exceed MAX_SCOPE (fail closed instead of approximating)."""


def labels(r) -> dict:
    """query -> decided label (0 / 1) or None, merged over every place the query occurs in the root."""
    lab = {}
    pairs = [(q, y) for c in r["cands"] for q, y in zip(c["z_q"], c["z"])] + list(r["y"].items())
    for q, y in pairs:
        if q is None:
            continue
        if y is not None and lab.get(q) is not None and lab[q] != y:
            raise IdentityError(f"{r['key']}: query {q} labelled {lab[q]} and {y}")
        lab[q] = y if y is not None else lab.get(q)
    return lab


def _align(t, vs, scope):
    t = t.permute([0] + [1 + vs.index(u) for u in scope if u in vs])
    return t.reshape([t.shape[0]] + [2 if u in vs else 1 for u in scope])


def contract(factors, B) -> torch.Tensor:
    """sum over every variable of prod(factors), in log space, by greedy exact elimination. factors: [(vars, [B, 2, ..])]."""
    fs, out = [(tuple(v), t) for v, t in factors], torch.zeros(B, dtype=torch.float64)
    while vs := sorted({v for f, _ in fs for v in f}):
        scope = lambda v: sorted({u for f, _ in fs if v in f for u in f})  # noqa: E731
        v = min(vs, key=lambda v: (len(scope(v)), v))
        sc = scope(v)
        PEAK[0] = max(PEAK[0], len(sc))
        if len(sc) > MAX_SCOPE:
            raise CombinatoricsError(f"elimination scope {len(sc)} > {MAX_SCOPE}")
        acc = sum(_align(t, f, sc) for f, t in fs if v in f)
        fs = [(f, t) for f, t in fs if v not in f] + [(tuple(u for u in sc if u != v), torch.logsumexp(acc, 1 + sc.index(v)))]
    for _, t in fs:
        out = out + t
    return out


def _factor(b, S, keep, clamp):
    """One repair block as a log factor over the kept shared variables: scene-tied slots follow the batch scene state,
    clamped slots are fixed, kept slots stay variables, every other slot is summed out."""
    L, kept = b["logq"].reshape([b["logq"].shape[0]] + [2] * len(b["queries"])), []
    for j in reversed(range(len(b["queries"]))):
        q = b["queries"][j]
        if q in S["index"]:
            z = S["Z"][:, S["index"][q]].reshape([-1] + [1] * (L.dim() - 2)) == 1.0
            L = torch.where(z, L.select(1 + j, 1), L.select(1 + j, 0))
        elif q in clamp:
            L = L.select(1 + j, clamp[q])
        elif q in keep:
            kept.insert(0, q)
        else:
            L = torch.logsumexp(L, 1 + j)
    return tuple(kept), L


def _setup(S, blocks):
    S["index"] = {q: i for i, q in enumerate(S["queries"])}
    count = collections.Counter(q for b in blocks for q in b["queries"])
    shared = {q for q, n in count.items() if n > 1 and q not in S["index"]}
    tied = any(q in S["index"] for b in blocks for q in b["queries"])
    return shared, tied


def _den(S, blocks, shared, tied, clamp=None):
    if not (shared or tied or clamp):
        return torch.zeros(len(S["logp"]), dtype=torch.float64)
    return contract([_factor(b, S, shared | set(clamp or {}), clamp or {}) for b in blocks], len(S["logp"]))


def log_evidence(S, blocks, lab) -> tuple:
    """(log P(D_c), number of distinct observed variables, shared-variable count) of one root."""
    shared, tied = _setup(S, blocks)
    model_q = set(S["queries"]) | {q for b in blocks for q in b["queries"]}
    obs = {q: y for q, y in lab.items() if y is not None and q in model_q}
    num = contract([_factor(b, S, shared, {q: obs[q] for q in b["queries"] if q in obs and q not in S["index"]}) for b in blocks],
                   len(S["logp"]))
    keep = torch.ones(len(S["logp"]), dtype=torch.bool)
    for q, i in S["index"].items():
        if q in obs:
            keep &= S["Z"][:, i] == float(obs[q])
    lt = S["logp"] + num - _den(S, blocks, shared, tied)
    return torch.logsumexp(lt[keep], 0), len(obs), len(shared)


@torch.no_grad()
def marginals(S, blocks) -> dict:
    """Exact prior marginal P(variable = 1) of every distinct query under the root model (no evidence)."""
    shared, tied = _setup(S, blocks)
    p = torch.exp(S["logp"])
    out = {q: float(p @ S["Z"][:, i]) for q, i in S["index"].items()}
    den = _den(S, blocks, shared, tied)
    for q in sorted({q for b in blocks for q in b["queries"]} - set(S["index"])):
        out[q] = float(torch.exp(torch.logsumexp(S["logp"] + _den(S, blocks, shared, tied, {q: 1}) - den, 0)))
    return out


def brute_force(S, blocks, lab) -> float:
    """Reference: enumerate every distinct variable jointly (tiny roots only); same model, no elimination."""
    xs = sorted({q for b in blocks for q in b["queries"]} - set(S["queries"]))
    tot = []
    for s in range(len(S["logp"])):
        zs = dict(zip(S["queries"], S["Z"][s].tolist()))
        rows = []
        for bits in itertools.product((0, 1), repeat=len(xs)):
            val = zs | dict(zip(xs, bits))
            u = sum(float(b["logq"][s, int("".join(str(int(val[q])) for q in b["queries"]), 2)]) for b in blocks)
            rows.append((all(val[q] == y for q, y in lab.items() if y is not None and q in val), u))
        lz = torch.logsumexp(torch.tensor([u for _, u in rows] or [0.0], dtype=torch.float64), 0)
        ok = [u for m, u in rows if m] if rows else [0.0]
        if ok and all(zs[q] == y for q, y in lab.items() if y is not None and q in zs):
            tot.append(float(S["logp"][s]) + float(torch.logsumexp(torch.tensor(ok, dtype=torch.float64), 0) - lz))
    return float(torch.logsumexp(torch.tensor(tot, dtype=torch.float64), 0))


# ------------------------------------------------------------------ ABC7 models: tables, factorized arms, the three terms
def tables(model, r) -> tuple:
    """(scene table S, repair blocks, labels, per-candidate outputs) from ONE memoized ABC7 forward pass of a root."""
    outs = t7.forward(model, r)
    e = next(i for i, c in enumerate(r["cands"]) if not c["seq"])
    o, ks = outs[e], outs[e]["ks"]
    names, qm = o["order"][:ks], dict(zip(r["cands"][e]["slots"], r["cands"][e]["z_q"]))
    S = {"names": names, "queries": [qm[n] for n in names], "logp": o["logP_S"], "Z": o["Z"][:, :ks]}
    blocks = []
    for c, oc in zip(r["cands"], outs):
        if oc["order"][:ks] != names:
            raise IdentityError(f"{r['key']}: scene slots differ across candidates")
        if c["seq"]:
            qc = dict(zip(c["slots"], c["z_q"]))
            blocks.append({"logq": oc["logP_Q"], "queries": [qc[n] for n in oc["order"][ks:]], "cand": c["query"]})
    return S, blocks, labels(r), outs


def factorized_logits(model, r, outs) -> dict:
    """A / S / SF: one logit per distinct query of the B / C variable set (same label access, aliases identical by design)."""
    e = next(i for i, c in enumerate(r["cands"]) if not c["seq"])
    lg = {q: outs[e]["u"] for s, q in zip(r["cands"][e]["slots"], r["cands"][e]["z_q"]) if s == "z0"}
    lg |= {q: model.scene_logit(r["R"], r["edits"][s[3:-1]]) for s, q in zip(r["cands"][e]["slots"], r["cands"][e]["z_q"]) if s.startswith("zS")}
    lg |= {c["query"]: o["u"] for c, o in zip(r["cands"], outs) if c["seq"]}
    for c in r["cands"]:
        for s, q in zip(c["slots"], c["z_q"]):
            if s.startswith("zO") and q not in lg:
                lg[q] = model.seq(r["R"], [a for k, a in enumerate(c["seq"]) if k != int(s[3:-1])])["u"]
    return lg


def _endpoint(seq):
    return tuple(sorted((i, tuple(map(float, d))) for i, d in seq))


def losses(model, r) -> dict:
    """{nll: -log P(D_c) (None if nothing observed), n_obs, shared, static: deduplicated endpoint BCE, travel: witnessed J}."""
    if model.kind in ("B", "C"):
        S, blocks, lab, outs = tables(model, r)
        ll, n, sh = log_evidence(S, blocks, lab)
    else:
        outs = t7.forward(model, r)
        lab, lg = labels(r), None
        lg = factorized_logits(model, r, outs)
        obs = [(q, y) for q, y in lab.items() if y is not None and q in lg]
        n, sh = len(obs), 0
        ll = -sum(F.binary_cross_entropy_with_logits(lg[q], torch.tensor(float(y), dtype=torch.float64), reduction="sum") for q, y in obs) if obs else None
    seen, st = set(), []
    for c, o in zip(r["cands"], outs):
        if _endpoint(c["seq"]) not in seen and (v := t7._static(o, c)) is not None:
            seen.add(_endpoint(c["seq"]))
            st.append(v)
    for c in r.get("static_checks", []):
        if _endpoint(c["seq"]) in seen:
            continue
        xy = r["R"].xy0.clone()
        for i, d in c["seq"]:
            xy[i] = torch.as_tensor(d, dtype=torch.float64)
        _, lu, lp = model.static(r["R"], xy)
        if (v := t7._static({"lu": lu, "lp": lp}, c)) is not None:
            seen.add(_endpoint(c["seq"]))
            st.append(v)
    tr = [F.huber_loss(torch.log1p(o["J"]), torch.log1p(torch.tensor(c["cost"], dtype=torch.float64)))
          for c, o in zip(r["cands"], outs) if c["cost"] is not None and c["seq"]]
    return {"nll": -ll if n else None, "n_obs": n, "shared": sh, "static": torch.stack(st).mean() if st else None,
            "travel": torch.stack(tr).mean() if tr else None}


def total(model, roots, lam_static=1.0, lam_travel=1.0) -> tuple:
    """L = mean_roots(-log P(D_c)) + lam_s mean_roots(static) + lam_J mean_roots(travel), each over roots that have the term."""
    per = [losses(model, r) for r in roots]
    tot, rec = 0.0, {}
    for k, lam in (("nll", 1.0), ("static", lam_static), ("travel", lam_travel)):
        v = [p[k] for p in per if p[k] is not None]
        if v:
            m = torch.stack(v).mean()
            tot, rec[k], rec[f"n_{k}"] = tot + lam * m, float(m.detach()), len(v)
    rec["n_obs"] = sum(p["n_obs"] for p in per)
    return tot, rec


def rank_energy(e_static, p_r, t_s=1.0, log_rho=0.0):
    """Success-conditioned candidate energy E_static - T_s log p_R - T_s log rho (uniform rho: log_rho = 0)."""
    return e_static - t_s * torch.log(torch.as_tensor(p_r, dtype=torch.float64).clamp(1e-12)) - t_s * log_rho
