"""Z_LAYOUT_V3_R4: versioned B/C outcome layout for repairs of up to four moves (P6).

Same slot semantics as Z_LAYOUT_V2 (which stays frozen at three moves / eight slots): z0, the
predeclared eligible fixed-robot edits zS[k], zR (m >= 1) and the real leave-one-out omissions
zO[t] for t < m when m >= 2 (deleting action t, keeping relative order, restarting from S0).
A four-move repair with edits {A, B, A+B} therefore has exactly K = 9 active binary fields
(512 states). Every active slot names a distinct physical query (aliases are refused).

Uniform reference measure (P6 §2.2): repairs with different K would otherwise be preferred a
priori in proportion to 2^K when raw energies are equal. The corrected joint energy is
E_ref(r, z) = E_raw(r, z) + T K(r) ln 2, i.e. E_bar(r) = -T log mean_z exp(-E_raw / T). P(z | r)
is unchanged. `measure="raw"` reproduces the historical P5 convention for reproducibility only.
"""
from __future__ import annotations

import itertools
import math

from config import stable_hash

VERSION = "Z_LAYOUT_V3_R4"
MOVE_CAP, MAX_ACTIVE = 4, 9


def sig(seq) -> str:
    return ".".join(f"{o}>{c}" for o, c in seq) or "empty"


def build(root: str, seq, catalogue: list) -> dict:
    """catalogue: [{"kind", "edit": {obj: cell}, "eligible"}] in frozen order (nuisance excluded by caller)."""
    seq = [(o, int(c)) for o, c in seq]
    m = len(seq)
    if m > MOVE_CAP or len({o for o, _ in seq}) != m:
        raise ValueError(f"not a canonical move-once sequence (m <= {MOVE_CAP}): {seq}")
    slots, active, query = ["z0"], [True], {"z0": f"seq:{root}:empty"}
    for q in catalogue:
        s = f"zS[{q['kind']}]"
        slots.append(s)
        active.append(bool(q["eligible"]))
        query[s] = f"scene:{root}:{sig(sorted((o, int(c)) for o, c in q['edit'].items()))}"
    slots.append("zR")
    active.append(m >= 1)
    query["zR"] = f"seq:{root}:{sig(seq)}"
    for t in range(MOVE_CAP):
        slots.append(f"zO[{t}]")
        active.append(m >= 2 and t < m)
        query[f"zO[{t}]"] = f"seq:{root}:{sig([a for i, a in enumerate(seq) if i != t])}" if t < m else None
    act_q = [query[s] for s, a in zip(slots, active) if a]
    if len(set(act_q)) != len(act_q):
        raise ValueError(f"active slots alias one physical query: {act_q}")
    if sum(active) > MAX_ACTIVE:
        raise ValueError(f"{sum(active)} active slots exceed the exact-enumeration bound {MAX_ACTIVE}")
    lay = {"version": VERSION, "root": root, "sequence": [list(a) for a in seq], "slots": slots, "active": active,
           "query": {s: query[s] for s, a in zip(slots, active) if a}, "active_count": sum(active)}
    lay["hash"] = stable_hash(lay)
    return lay


def verify(lay: dict) -> None:
    if lay.get("version") != VERSION:
        raise ValueError(f"layout version {lay.get('version')!r} is not {VERSION}")
    if stable_hash({k: v for k, v in lay.items() if k != "hash"}) != lay.get("hash"):
        raise ValueError("layout hash mismatch")
    m = len(lay["sequence"])
    want = {f"zO[{t}]": m >= 2 and t < m for t in range(MOVE_CAP)} | {"zR": m >= 1, "z0": True}
    if any(a != want[s] for s, a in zip(lay["slots"], lay["active"]) if s in want):
        raise ValueError("active mask violates the V3 rule")


def states(lay: dict) -> list:
    verify(lay)
    names = [s for s, a in zip(lay["slots"], lay["active"]) if a]
    return [dict(zip(names, b)) for b in itertools.product((0, 1), repeat=len(names))]


def _lse(xs) -> float:
    top = max(xs)
    return top + math.log(math.fsum(math.exp(x - top) for x in xs))


def exact_marginal(joint, lay: dict, temperature: float, measure: str = "uniform") -> tuple:
    """(E_bar, [(z, P(z | r))]) by exact enumeration of the active slots; normalization checked."""
    zs = states(lay)
    logits = [-joint(z) / temperature for z in zs]
    lse = _lse(logits)
    probs = [math.exp(x - lse) for x in logits]
    if abs(math.fsum(probs) - 1.0) > 1e-9:
        raise ArithmeticError("exact z-marginal not normalized")
    e_bar = -temperature * lse + (temperature * lay["active_count"] * math.log(2) if measure == "uniform" else 0.0)
    return e_bar, list(zip(zs, probs))
