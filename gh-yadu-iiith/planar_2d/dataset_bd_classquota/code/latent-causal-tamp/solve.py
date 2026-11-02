"""A0: analytic terminal assignment (Methodology v1.4 §7, §7.1, §7.2; Guide v1.6 G3).

Terminal problem. Every present obstacle i takes one physical cell from its declared
domain; its current cell means "stay" (cost 0), any other cell is a relocation
(cost 1), so the cost is the number of distinct moved IDs. A state is terminal-valid
iff each pose is frozen-action compatible (y^a), supported (y^w) and pairwise
compatible (y^p). For identical discs on the uniform grid, two distinct cells are
always pair-compatible (pitch > 2r + delta, checked below), so y^p is exactly "one
object per cell" and the problem is an exact min-cost bipartite matching (objects x
cells). y^a, y^w of a disc at a cell do not depend on which identical disc it is,
so they are computed once per physical cell.

Edges: VALID cells give witnesses; VALID or UNRESOLVED cells give the optimistic
problem whose optimum is a sound lower bound. Cell IDs are physical (r*cols + c).
"""
from __future__ import annotations

import heapq
import itertools
import math

import numpy as np
from scipy.optimize import linear_sum_assignment

import interventions as iv
import motion as mo


class ReductionInvalid(ValueError):
    """Matching is not exact for this scene (pair constraints are not one-per-cell)."""


def cell_status(robot, cfg: dict, frozen, scene) -> dict:
    """y^a AND y^w of an obstacle-sized disc at each legal cell except the target's resting cell."""
    delta, eps, evals = iv._params(cfg)
    lat, t = scene.lattice, scene.get(frozen.target_id)
    r, support = cfg["suite"]["obstacle_radius_m"], (lat.cols * lat.pitch_m, lat.rows * lat.pitch_m)
    out = {}
    for c in scene.legal_cells:
        if c == t.cell:
            continue
        x, y = lat.center(c)
        inside = r <= x <= support[0] - r and r <= y <= support[1] - r
        chk = mo.object_check(robot, frozen.arrays, (t.id, t.xy_m, t.radius_m), ("cell", (x, y), r), support,
                              delta, eps, evals)
        out[c] = mo._conjoin([chk.status, mo.VALID if inside else mo.INVALID])
    return out


def check_reduction(scene, delta: float) -> None:
    radii = {o.radius_m for o in scene.objects}
    if len(radii) != 1 or scene.lattice.pitch_m <= 2 * max(radii) + delta:
        raise ReductionInvalid("matching needs identical discs with pitch > 2r + delta")
    if any(o.cell is None for o in scene.objects):
        raise ReductionInvalid("an object rests off the lattice")


def domains(scene, policy: str, restricted: dict | None = None) -> dict:
    """Per-obstacle physical destination cells. ALL_SUPPORTED: every legal cell except
    the target's; RESTRICTED: the published list (staying is always allowed)."""
    t_cells = {o.cell for o in scene.objects if o.kind == "target"}
    out = {}
    for o in (o for o in scene.objects if o.kind == "obstacle"):
        cells = [c for c in scene.legal_cells if c not in t_cells] if policy == "ALL_SUPPORTED" \
            else sorted(set(restricted.get(o.id, [])) | {o.cell})
        out[o.id] = tuple(cells)
    return out


def cost_matrix(scene, dom: dict, status: dict, optimistic: bool):
    """(object ids, physical cells, C): C[i, j] = 0 to stay, inf if not allowed, and
    1 + w * (carry distance / longest possible) to move, with w = 1/(n + 1). The
    integer part is the moved-object count (the A0 objective); the distance term
    only orders ties, since n moves add less than n/(n + 1) < 1 in total."""
    ids = sorted(dom)
    cells = sorted(set().union(*dom.values()))
    ok = {mo.VALID, mo.UNRESOLVED} if optimistic else {mo.VALID}
    lat, w = scene.lattice, 1.0 / (len(ids) + 1)
    dmax = math.hypot(lat.cols * lat.pitch_m, lat.rows * lat.pitch_m)
    C = np.full((len(ids), len(cells)), np.inf)
    for i, oid in enumerate(ids):
        cur = scene.get(oid).cell
        for j, c in enumerate(cells):
            if c in dom[oid] and status.get(c) in ok:
                C[i, j] = 0.0 if c == cur else 1.0 + w * math.dist(lat.center(cur), lat.center(c)) / dmax
    return ids, cells, C


def moved_count(cost: float) -> int:
    """The A0 objective (number of moved IDs) from a lexicographic cost."""
    return int(math.floor(cost + 1e-9))


def _solve(C):
    if C.shape[0] == 0:
        return 0.0, ()
    try:
        r, c = linear_sum_assignment(C)
    except ValueError:                      # no finite complete assignment
        return None
    cost = float(C[r, c].sum())
    return (cost, tuple(int(x) for x in c)) if math.isfinite(cost) else None


def kbest(C, k: int) -> list:
    """Murty's partitioning: the k cheapest distinct complete assignments, in
    nondecreasing cost (ties in deterministic discovery order)."""
    first = _solve(C)
    if first is None:
        return []
    heap, n, out, count = [(first[0], 0, first[1], C)], C.shape[0], [], 1
    while heap and len(out) < k:
        cost, _, assign, sub = heapq.heappop(heap)
        out.append((cost, assign))
        fixed = sub.copy()
        for t in range(n):                  # no-good: exclude (t, assign[t]) after fixing rows < t
            excl = fixed.copy()
            excl[t, assign[t]] = np.inf
            s = _solve(excl)
            if s is not None:
                heapq.heappush(heap, (s[0], count, s[1], excl))
                count += 1
            keep = fixed[t, assign[t]]
            fixed[t, :], fixed[:, assign[t]] = np.inf, np.inf
            fixed[t, assign[t]] = keep
    return out


def exhaustive(C) -> list:
    """Every finite complete assignment, sorted (small cases only: the test oracle)."""
    n, m = C.shape
    if math.perm(m, n) > 200000:
        raise ValueError("domain too large for complete enumeration")
    out = [(float(sum(C[i, j] for i, j in enumerate(p))), p) for p in itertools.permutations(range(m), n)]
    return sorted((c, p) for c, p in out if math.isfinite(c))


def hall_certificate(C):
    """None if every object can be matched; else (rows U, cols N) with |N(U)| < |U|:
    a finite, checkable proof that no complete assignment exists."""
    n, m = C.shape
    adj = [[j for j in range(m) if math.isfinite(C[i, j])] for i in range(n)]
    owner = [-1] * m

    def augment(i, seen):
        for j in adj[i]:
            if j not in seen:
                seen.add(j)
                if owner[j] < 0 or augment(owner[j], seen):
                    owner[j] = i
                    return True
        return False

    free = [i for i in range(n) if not augment(i, set())]
    if not free:
        return None
    rows, cols, todo = {free[0]}, set(), [free[0]]
    while todo:                              # alternating reachability from an unmatched object
        for j in adj[todo.pop()]:
            if j not in cols:
                cols.add(j)
                if owner[j] >= 0 and owner[j] not in rows:
                    rows.add(owner[j])
                    todo.append(owner[j])
    return sorted(rows), sorted(cols)


def a0(scene, dom: dict, status: dict, k: int, delta: float) -> dict:
    """Terminal lower bound (optimistic edges), Hall certificate if infeasible, and the
    k-best certified-valid assignments as physical {object: cell} maps."""
    check_reduction(scene, delta)
    ids, cells, Copt = cost_matrix(scene, dom, status, optimistic=True)
    _, _, Cval = cost_matrix(scene, dom, status, optimistic=False)
    lb = _solve(Copt)
    out = {"objects": ids, "cells": cells, "lower_bound": None if lb is None else moved_count(lb[0]),
           "direct_blockers": [i for i in ids if status.get(scene.get(i).cell) == mo.INVALID],
           "hall": None, "assignments": []}
    if lb is None:
        rows, cols = hall_certificate(Copt)
        out["hall"] = {"objects": [ids[r] for r in rows], "neighbourhood_cells": [cells[c] for c in cols]}
        return out
    for cost, assign in kbest(Cval, k):
        out["assignments"].append({"cost": moved_count(cost), "tie_break": round(cost - moved_count(cost), 6),
                                   "cells": {ids[i]: cells[j] for i, j in enumerate(assign)}})
    return out


def moves_of(scene, assignment: dict) -> list:
    """The relocations an assignment requires: [(object id, destination cell)]."""
    return [(o, c) for o, c in sorted(assignment.items()) if scene.get(o).cell != c]


def restoration(scene, reference, status: dict) -> dict:
    """Methodology §7 lemma witness: every current direct blocker back to its reference
    cell (others stay). Valid iff those cells are action-clear, supported and vacant."""
    blockers = [o.id for o in scene.objects if o.kind == "obstacle" and status.get(o.cell) == mo.INVALID]
    cells = {o: reference.get(o).cell for o in blockers}
    occupied = {o.cell for o in scene.objects if o.id not in blockers}
    valid = all(status.get(c) == mo.VALID and c not in occupied for c in cells.values()) \
        and len(set(cells.values())) == len(cells)
    return {"direct_blockers": blockers, "assignment": cells, "terminal_valid": bool(valid), "moves": len(blockers)}


def forced(scene, dom: dict, status: dict, obj: str, cell: int) -> dict | None:
    """Cheapest certified-valid assignment with obj forced to a given physical cell."""
    ids, cells, C = cost_matrix(scene, dom, status, optimistic=False)
    if obj not in ids or cell not in cells:
        return None
    i, j = ids.index(obj), cells.index(cell)
    F = C.copy()
    F[i, :], F[:, j] = np.inf, np.inf
    F[i, j] = C[i, j]
    s = _solve(F)
    return None if s is None else {"cost": moved_count(s[0]), "cells": {ids[r]: cells[c] for r, c in enumerate(s[1])}}


def check_hall(C, rows, cols) -> bool:
    """Independent check: N(U) is exactly the allowed cells of U, and |N(U)| < |U|."""
    neighbourhood = {j for i in rows for j in range(C.shape[1]) if math.isfinite(C[i, j])}
    return neighbourhood == set(cols) and len(cols) < len(rows)


def restricted_domain(scene, status: dict, blockers: list, count: int, exclude=()) -> dict:
    """Published restricted domain: the direct blockers may use only the `count` vacant
    action-clear cells nearest their centroid, never the excluded cells (their
    reference restoration cells); every other obstacle may only stay."""
    centre = np.mean([scene.get(b).xy_m for b in blockers], axis=0)
    occ = {o.cell for o in scene.objects} | set(exclude)
    free = sorted((c for c, v in status.items() if v == mo.VALID and c not in occ),
                  key=lambda c: (math.dist(scene.lattice.center(c), centre), c))
    return {b: free[:count] for b in blockers}
