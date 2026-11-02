"""Z_LAYOUT_V2: corrected, versioned B/C outcome layout (P5.0).

The frozen G4C/D2 layout (g4c.layout) activates zO[t] for every t < 3 whenever m >= 2, so a
two-move repair carries a third "omit nothing" slot that repeats zR (report 0019, §4.6). This
learner-side view keeps every stored record untouched and defines, deterministically and
before outcomes:  z0 (original task), zS[k] for each predeclared eligible fixed-robot edit,
zR for m >= 1, and zO[t] for t < m only when m >= 2 (omitting the single action of a
one-move repair IS the empty repair, i.e. z0). Every active slot names one physical teacher
query; two active slots may never alias the same query (refused, never silently merged).
"""
from __future__ import annotations

import math

import sampler as sp
from config import stable_hash

VERSION = "Z_LAYOUT_V2"
MOVE_CAP = 3


def sig(seq) -> str:
    return ".".join(f"{o}>{c}" for o, c in seq) or "empty"


def build(root: str, seq, catalogue: list) -> dict:
    """catalogue: [{"kind", "edit": {obj: cell}, "eligible": bool}] in frozen order."""
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
        s = f"zO[{t}]"
        slots.append(s)
        active.append(m >= 2 and t < m)
        query[s] = f"seq:{root}:{sig([a for i, a in enumerate(seq) if i != t])}" if t < m else None
    act_q = [query[s] for s, a in zip(slots, active) if a]
    if len(set(act_q)) != len(act_q):
        raise ValueError(f"active slots alias one physical query: {act_q}")
    if sum(active) > sp.MAX_ACTIVE_Z:
        raise ValueError(f"{sum(active)} active slots exceed the exact-enumeration bound {sp.MAX_ACTIVE_Z}")
    lay = {"version": VERSION, "root": root, "sequence": [list(a) for a in seq], "slots": slots, "active": active,
           "fixed": [0] * len(slots), "query": {s: query[s] for s, a in zip(slots, active) if a},
           "active_count": sum(active)}
    lay["hash"] = stable_hash(lay)
    return lay


def verify(lay: dict) -> None:
    """Refuse any record whose version, content hash or active-slot rule is not V2's."""
    if lay.get("version") != VERSION:
        raise ValueError(f"layout version {lay.get('version')!r} is not {VERSION}")
    if stable_hash({k: v for k, v in lay.items() if k != "hash"}) != lay.get("hash"):
        raise ValueError("layout hash mismatch (edited or foreign record)")
    m = len(lay["sequence"])
    want = {f"zO[{t}]": m >= 2 and t < m for t in range(MOVE_CAP)} | {"zR": m >= 1, "z0": True}
    if any(a != want[s] for s, a in zip(lay["slots"], lay["active"]) if s in want):
        raise ValueError("active mask violates the V2 rule")


def from_v1(root: str, seq, stored_layout: dict, catalogue: list) -> dict:
    """Derive the V2 view of one stored G4C/D2 bc_layout without touching it: identical slot
    meaning except that no-op omission slots (t >= m) become inactive."""
    v2 = build(root, seq, catalogue)
    kept = [s for s, a in zip(stored_layout["slots"], stored_layout["active"]) if a]
    dropped = [s for s in kept if s not in v2["query"]]
    if any(not (s.startswith("zO[") and int(s[3:-1]) >= len(seq)) for s in dropped):
        raise ValueError(f"V1 -> V2 would drop a non-duplicate slot: {dropped}")
    return {**v2, "v1_active": len(kept), "v1_dropped_duplicates": dropped}


def sampler_layout(lay: dict) -> sp.Layout:
    verify(lay)
    return sp.Layout(tuple(lay["slots"]), tuple(lay["active"]), tuple(lay["fixed"]))


def exact_marginal(joint, lay: dict, temperature: float) -> tuple:
    """(E_bar, [(z, p)]) by exact enumeration of the ACTIVE slots only (inactive slots never
    enter the state count); checks normalization."""
    zs = sp.z_states(sampler_layout(lay))
    logits = [-joint(z) / temperature for z in zs]
    lse = sp.logsumexp(logits)
    probs = [math.exp(x - lse) for x in logits]
    if abs(math.fsum(probs) - 1.0) > 1e-9:
        raise ArithmeticError("exact z-marginal not normalized")
    return -temperature * lse, list(zip(zs, probs))
