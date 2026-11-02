"""P17 shared scene-only 2D action domain for B and D.

Task / state come only from the physical scene, target, goal, robot q and declared fixed structures (p16_common.task_from_root, deletion-
tested). Movers = every movable non-fixed non-target object (MOVER_CAP only as a safety bound; omissions recorded). Slots are recomputed on
the CURRENT hypothetical occupancy: free annulus-safe cells at Chebyshev >= 2 from the (current) target cell and the goal, nearest the
object first, then freed source cells of earlier-moved objects, then library-witnessed destinations; widening to Chebyshev >= 1 when fewer
than `width` exist (deterministic, model-free). Ordered repairs (B's candidates) are enumerated deterministically from the SAME slot
function: lengths 0..4, an object may appear up to `per_obj` times but never twice in a row (D's rule), lexicographic priority
(length, mover rank, slot rank) with a per-length cap; every discarded candidate is counted."""
from __future__ import annotations

import itertools

import numpy as np


MOVER_CAP = 8
DOMAIN_VERSION = "P17_DOMAIN_V1"
CAPS = {0: 1, 1: 12, 2: 24, 3: 12, 4: 8}                                    # per-length candidate caps (declared)


def movers(task, cells):
    """Every movable non-fixed non-target object, nearest the current target first; (kept, omitted beyond MOVER_CAP)."""
    from scene import scene_from_dict
    s = scene_from_dict(task["scene"])
    T = np.asarray(s.lattice.center(cells["t0"]))
    mov = [o.id for o in s.objects if o.kind != "target" and o.id not in set(task["fixed_ids"])]
    mov.sort(key=lambda o: (float(np.linalg.norm(np.asarray(s.lattice.center(cells[o])) - T)), o))
    return mov[:MOVER_CAP], mov[MOVER_CAP:]


def slot_fn(phys, idx=None, width=4, origin=None):
    """Deterministic multi-slot proposal on the current state (see module doc); returns at most width + freed + witnessed cells."""
    import p6_scenes as p6
    origin = origin or {o.id: o.cell for o in phys.s0.objects}

    def f(cells, o):
        T, occ = cells["t0"], set(cells.values())
        s = phys.scene(cells, phys.s0.robot.q_rad)
        out = p6._dests(phys.scr, s, o, [T, phys.goal_cell], [], width)
        if len(out) < width:
            near = [c for c in sorted(phys.scr.valid) if c not in occ and c not in out and p6._cheb(c, T) >= 1 and p6._cheb(c, phys.goal_cell) >= 1]
            out += sorted(near, key=lambda c: (float(np.linalg.norm(np.asarray(s.lattice.center(c)) - np.asarray(s.get(o).xy_m))), c))[:width - len(out)]
        freed = sorted(c for x, c in origin.items() if x not in ("t0", o) and cells.get(x) != c and c not in occ and c not in out and c in phys.scr.valid)
        lib = sorted(d for (src, d) in (idx.by_key if idx is not None else {}) if src == cells[o] and d not in occ and d not in out and d not in freed
                     and d in phys.scr.valid and p6._cheb(d, T) >= 2 and p6._cheb(d, phys.goal_cell) >= 2)
        return out + freed + lib[:2]
    return f


def repairs(task, phys, idx=None, max_moves=4, per_obj=2, width=2, caps=None, top_movers=None) -> dict:
    """Deterministic ordered-repair candidates (no physics oracle, no labels): returns {'sequences', 'generated', 'kept', 'omitted'}.
    Each step's destinations come from slot_fn on the hypothetically updated occupancy (first `width` slots + freed sources)."""
    caps = caps or CAPS
    cells0 = {o.id: o.cell for o in phys.s0.objects}
    mv, _ = movers(task, cells0)
    mv = mv[:top_movers] if top_movers else mv
    sl = slot_fn(phys, idx, width=width, origin=cells0)
    by_len, gen = {0: [[]]}, {0: 1}

    def ext(seq, cells, last, counts):
        for o in mv:
            if counts.get(o, 0) >= per_obj or o == last:
                continue
            for d in sl(cells, o)[:width + 2]:
                yield o, d

    frontier = [([], cells0, None, {})]
    for L in range(1, max_moves + 1):
        nxt, cand = [], []
        for seq, cells, last, counts in frontier:
            for o, d in ext(seq, cells, last, counts):
                s2 = seq + [(o, d)]
                cand.append(s2)
                nxt.append((s2, {**cells, o: d}, o, {**counts, o: counts.get(o, 0) + 1}))
        gen[L] = len(cand)
        by_len[L] = _stride(cand, caps.get(L, 0))
        frontier = _stride(nxt, FRONTIER)
    kept = [s for L in sorted(by_len) for s in by_len[L]]
    return {"version": DOMAIN_VERSION, "sequences": kept, "generated": gen, "kept": {L: len(v) for L, v in by_len.items()},
            "omitted": {L: gen[L] - len(by_len[L]) for L in gen}, "movers": mv}


FRONTIER = 400


def _stride(xs, n):
    """Deterministic evenly spaced subsample (keeps order; model-free, label-free)."""
    if n <= 0:
        return []
    if len(xs) <= n:
        return list(xs)
    step = len(xs) / n
    return [xs[int(i * step)] for i in range(n)]


def exhaustive_sequences(task, phys, idx=None, max_moves=3, per_obj=2, width=2):
    """All legal sequences of the same domain without caps (tiny scenes only; recall reference)."""
    cells0 = {o.id: o.cell for o in phys.s0.objects}
    mv, _ = movers(task, cells0)
    sl = slot_fn(phys, idx, width=width, origin=cells0)
    out = [[]]

    def go(seq, cells, last, counts):
        if len(seq) == max_moves:
            return
        for o in mv:
            if counts.get(o, 0) >= per_obj or o == last:
                continue
            for d in sl(cells, o)[:width + 2]:
                s2 = seq + [(o, d)]
                out.append(s2)
                go(s2, {**cells, o: d}, o, {**counts, o: counts.get(o, 0) + 1})
    go([], cells0, None, {})
    return out


def query_id(task_key, seq) -> str:
    """Physical-query identity: scene / task key + full ordered (object, destination) list; omissions recompute intermediate poses."""
    return f"seq:{task_key}:" + (".".join(f"{o}>{d}" for o, d in seq) or "empty")


__all__ = ["movers", "slot_fn", "repairs", "exhaustive_sequences", "query_id", "itertools"]
