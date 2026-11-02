"""ABC9K_KEYED_V1: physically keyed relational instantiation for the ABC9U unique-factor root model (P9-L3c-1 competitor).

ABC9U_UNIQUE_V1 instantiates relational factors per LISTED parent candidate: a scene-query pair carries the parent's context
g_r and the slot's role embedding (zR / zO), and a higher-order term exists only for listed candidates. Listing an already
modelled omission query as a candidate therefore adds new factor instances although no new physical outcome is modelled.

Keyed rule (no new parameter, no new family, no factor dropped): the variable set V is the distinct sequence queries of the
catalogue (repairs and their omissions, as in ABC9U). EVERY q in V owns one self-block, keyed by its physical sequence:
  * F2 unary s_q (once, unchanged; it is also the F4 omission unary);
  * F5 scene-query pairs (a, q) for every scene bit a: cpl(v_a, g_q + type(zR), context g_q) - one instance per (a, q),
    computed from q's own representation, whatever role q was listed in;
  * F4 zR-zO pairs (q, q minus t) and F5 zO-zO pairs among q's omissions, for omissions that are themselves in V (context g_q);
  * F5 higher-order ho_q over (z_S, x_q, x_(q minus t) in V) with context g_q (full only).
Scene factors stay in P_S once; one conditional normalizer per scene state; inference reuses unique_factor unchanged.
The factor set is a function of V only, so listing a modelled query again (any role) adds nothing. For a one-move single
candidate it equals ABC7 exactly; for multi-move candidates it is NOT ABC7 (scene-omission pairs use the omission's own
context and role, and each modelled omission gets its own self-block).
"""
from __future__ import annotations

import contextlib
import itertools

import torch

import joint_likelihood as jl
import unique_factor as uf
from abc_models import TYPES

VERSION = "ABC9K_KEYED_V1"


def _key(seq):
    return tuple((int(i), tuple(map(float, d))) for i, d in seq)


def instances(m, r) -> dict:
    """Keyed factor instances of one root (same dict format as unique_factor.instances)."""
    e = next(c for c in r["cands"] if not c["seq"])
    sn = ["z0"] + sorted(n for n in e["slots"] if n.startswith("zS"))
    qmap = dict(zip(e["slots"], e["z_q"]))
    seqs, byseq = {}, {}
    for c in r["cands"]:
        for s, q in zip(c["slots"], c["z_q"]):
            if s[:2] in ("zR", "zO"):
                sq = c["seq"] if s == "zR" else [a for k, a in enumerate(c["seq"]) if k != int(s[3:-1])]
                if seqs.setdefault(q, _key(sq)) != _key(sq) or byseq.setdefault(_key(sq), q) != q:
                    raise jl.IdentityError(f"{r['key']}: query {q} and its physical sequence disagree")
    m._memo = {}
    try:
        S = m.scene_block(r["R"], sn, r["edits"])
        ks = len(sn)
        inst = {"S": {"names": sn, "queries": [qmap[n] for n in sn], "logp": S["logP"], "Z": S["Z"]}, "unary": {}, "seq": seqs, "blocks": {}}
        for q in sorted(seqs):
            sq = list(seqs[q])
            s, g, _ = m.order(r["R"], sq)
            inst["unary"][q] = s
            om = [byseq[_key([a for k, a in enumerate(sq) if k != t])] for t in range(len(sq)) if len(sq) >= 2
                  and _key([a for k, a in enumerate(sq) if k != t]) in byseq]
            vs = [g + m.typ(torch.tensor(TYPES["zR"]))] + [m.order(r["R"], list(seqs[o]))[1] + m.typ(torch.tensor(TYPES["zO"])) for o in om]
            kq = 1 + len(om)
            V = torch.cat([S["v"], torch.stack(vs)])
            spec = [(ks, ks + j, "omit") for j in range(1, kq)] + [(a, ks, "couple") for a in range(ks)]
            spec += [(ks + a, ks + b, "couple") for a, b in itertools.combinations(range(1, kq), 2)]
            Z = uf._states(ks + kq)
            E = -0.5 * ((Z @ m._pairs(V, g, spec)) * Z).sum(1)
            if m.couple == "full":
                E = E - m.ho_q(torch.cat([Z @ m.ho_u(V), g.expand(len(Z), -1)], 1))[:, 0]
            inst["blocks"][q] = {"queries": [q] + om, "energy": E.reshape(2 ** ks, 2 ** kq)}
    finally:
        m._memo = None
    return inst


@contextlib.contextmanager
def installed():
    """Route unique_factor's losses / evaluation through the keyed instantiation (inference code itself is unchanged)."""
    old = uf.instances
    uf.instances = instances
    try:
        yield
    finally:
        uf.instances = old
