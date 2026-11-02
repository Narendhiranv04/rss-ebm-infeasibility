"""P17 fixed-sequence EXISTENTIAL physical realization (B's supervision / execution verifier; replaces p16_common.verify_sequence,
kept as the frozen greedy baseline `verify_sequence_greedy_P16`).

The ordered (object, destination) list is held fixed. Each step branches over EVERY certified motion alternative returned by the provider
(distinct terminal q, library modes with certified / planned transits, online fallback), with its own successor state and measured
travel; the target action ends each branch. Status:
  VERIFIED              some branch completes; the returned witness is the minimum-travel branch among those adjudicated;
                        travel_minimal_over_alternatives = no branch remained undecided and no edge reported unexplored modes.
  CERTIFIED_INFEASIBLE  only when a MOTION-INDEPENDENT sound certificate applies at a step: occupancy at step i is fixed by the
                        sequence, so a pose-only certificate (STATIC_INVALID / ORDER_FIRST_ACTION / STATIC_ENDPOINT_INVALID /
                        TARGET_PICK / GOAL_OVERLAP / FIRST_MOVER) at step i refutes every realization. A q-dependent certificate
                        (LINK0_SECTOR_PICK) refutes only the branch whose q it was computed for; for the EMPTY sequence that branch
                        (q0) is the only realization, so it certifies.
  UNKNOWN_*             otherwise (budget, planner, transit, branch cap), with per-branch provenance. Never a negative label."""
from __future__ import annotations

import numpy as np

import p15_search as ps

V, CI, SI = ps.V, ps.CI, ps.SI
Q_DEPENDENT = {"LINK0_SECTOR_PICK"}
MAX_BRANCHES = 81


def _q_independent(meta) -> bool:
    c = (meta or {}).get("certificates") or []
    return bool(c) and not (set(c) & Q_DEPENDENT)


def verify_existential(phys, seq, max_branches=MAX_BRANCHES, first=False) -> dict:
    """first=True: stop at the first VERIFIED branch (branches explored cheapest-first); the witness is then flagged not travel-minimal."""
    cells0, q0 = {o.id: o.cell for o in phys.s0.objects}, np.asarray(phys.s0.robot.q_rad, float)
    found, fails, n_branch = [], [], [0]
    proof = {"step": None, "kind": None}
    unexplored = [False]

    def rec(i, cells, q, steps, recs, J):
        if n_branch[0] >= max_branches or (first and found):
            fails.append(("UNKNOWN_BRANCH_CAP", i))
            return
        if i == len(seq):
            n_branch[0] += 1
            st, r, tj = phys.goal(cells, q)
            if st == V:
                found.append({"moves": len(seq), "travel": J + tj, "steps": steps, "records": recs + [r]})
            elif st == CI and (_q_independent(r if isinstance(r, dict) else None) or not seq):   # empty sequence: q0 is the only realization
                proof.update(step="target", kind=sorted(r["certificates"]))
                fails.append(("CERTIFIED_TARGET_Q_INDEPENDENT", i))
            else:
                fails.append((st if st != CI else "CERTIFIED_TARGET_Q_DEPENDENT", i))
            return
        o, d = seq[i]
        if d in {c for x, c in cells.items() if x != o}:
            proof.update(step=i, kind=["DESTINATION_OCCUPIED"])
            fails.append((SI, i))
            return
        outs = phys.edges(cells, q, o, d)
        ver = sorted([x for x in outs if x[0] == V], key=lambda x: x[3])
        if any(x[0] not in ps.DECIDED for x in outs):
            unexplored[0] = True
        if not ver:
            st, meta = outs[0][0], outs[0][4] if outs else None
            if st == SI or (st == CI and _q_independent(meta)):
                proof.update(step=i, kind=sorted((meta or {}).get("certificates") or ["STATIC_INVALID"]))
                fails.append(("CERTIFIED_STEP", i))
            else:
                fails.append((st, i))
            return
        for st, rs, q2, tj, meta in ver:
            rec(i + 1, {**cells, o: d}, q2, steps + [{"object": o, "dest": d, **meta}], recs + rs, J + tj)

    rec(0, cells0, q0, [], [], 0.0)
    if found:
        best = min(found, key=lambda p: p["travel"])
        undecided = [f for f in fails if not str(f[0]).startswith("CERTIFIED")]
        return {"status": V, "plan": best, "branches_verified": len(found), "branches_failed": len(fails),
                "travel_minimal_over_alternatives": not first and not undecided and not unexplored[0], "first_mode": first, "fails": fails[:20]}
    if proof["kind"] is not None:
        return {"status": CI, "plan": None, "certificate": proof, "branches_failed": len(fails), "fails": fails[:20]}
    first = next((str(f[0]) for f in fails), "UNKNOWN_PLANNER")
    st = first if first.startswith("UNKNOWN") else "UNKNOWN_" + first.replace("UNKNOWN_", "")
    return {"status": st, "plan": None, "branches_failed": len(fails), "fails": fails[:20]}


def verify_sequence_greedy_P16(phys, seq) -> dict:
    """Frozen retrospective baseline (P16 semantics: cheapest certified mode per step, no backtracking)."""
    import p16_common as pc
    return pc.verify_sequence(phys, seq)
