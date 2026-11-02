"""INITIAL_SCENE_ORDERED_ONE_SHOT_V2 canonical ordered repairs (Methodology v1.6 §3, §8,
§12; Guide v1.8 §3, §5). G4D migration: proposal side only.

A repair is ONE canonical sequence r = ((i1, k1), ..., (im, km)): distinct movable
obstacles, each with a supported physical destination cell that differs from its
initial pose. The empty sequence is the unique no-repair proposal. The assignment
x(r) is derived from r (selected objects at their destinations, all others stay);
there is no second permutation that could disagree with it, no padding entry and no
stay entry. Omitting an action restores that object's stay; physical deletion is a
different, diagnostic-only operation and never a repair.

Everything here reads only the initial scene, robot state and action/goal. It
imports no planner, relocation verifier or target evaluator, and reads no label.
Split from proposals.py (which keeps the V1 endpoint generator) to keep that module
under the size guideline; proposals.py's primitives are reused unchanged.
"""
from __future__ import annotations

import itertools
import math

import motion as mo
import proposals as pr
from config import stable_hash
from scene import POSE_EQ_TOL_M, scene_hash, substitute_pose

V2 = "INITIAL_SCENE_ORDERED_ONE_SHOT_V2"
EMPTY = ()


def canonical(seq) -> tuple:
    """Stored form ((object id, cell), ...) in the proposed order (never re-sorted). It
    never converts: a non-str id or a non-int (or bool) cell raises instead of being
    coerced (int(True) == 1, int(29.9) == 29 would silently change the request)."""
    out = tuple(tuple(a) for a in seq)
    if any(len(a) != 2 or not isinstance(a[0], str) or isinstance(a[1], bool) or not isinstance(a[1], int) for a in out):
        raise TypeError(f"not a canonical sequence: {seq!r}")
    return out


def parse_sequence(scene, action: dict, raw, move_cap: int) -> tuple:
    """The one public validation contract (ranking, teaching, explanations): returns
    (canonical sequence or None, reasons). Raw shape and types are checked BEFORE any
    conversion; then the physical rules of validate_sequence."""
    if not isinstance(raw, (list, tuple)):
        return None, [f"sequence must be a list of (object, cell) pairs, got {type(raw).__name__}"]
    why = []
    for t, a in enumerate(raw):
        if not isinstance(a, (list, tuple)) or len(a) != 2:
            why.append(f"action {t}: malformed pair {a!r}")
        elif not isinstance(a[0], str):
            why.append(f"action {t}: object id {a[0]!r} is not a string")
        elif isinstance(a[1], bool) or not isinstance(a[1], int):
            why.append(f"action {t}: destination {a[1]!r} is not an integer cell (bool/float/other rejected)")
    if why:
        return None, why
    seq = canonical(raw)
    why = validate_sequence(scene, action, seq, move_cap)
    return (None, why) if why else (seq, [])


def validate_sequence(scene, action: dict, seq, move_cap: int) -> list:
    """Reasons a sequence is not a legal canonical V2 repair ([] = legal)."""
    ids, why, seen = {o.id: o for o in scene.objects}, [], set()
    target_cells = {o.cell for o in scene.objects if o.kind == "target"}
    if len(seq) > move_cap:
        why.append(f"length {len(seq)} exceeds the move cap {move_cap}")
    for t, item in enumerate(seq):
        if not (isinstance(item, (list, tuple)) and len(item) == 2) or not isinstance(item[0], str):
            why.append(f"action {t}: not an (object id, destination) pair")
            continue
        o, d = item
        if o in seen:
            why.append(f"action {t}: {o} repeated (move-once; no duplicate or padding entries)")
        seen.add(o)
        if o == action["target_id"]:
            why.append(f"action {t}: {o} is the active target (never a repair object)")
        elif o not in ids or ids[o].kind != "obstacle":
            why.append(f"action {t}: {o} is not a movable obstacle of this scene")
        if isinstance(d, bool) or not isinstance(d, int) or d not in scene.legal_cells:
            why.append(f"action {t}: destination {d!r} is not a supported physical cell")
        elif d in target_cells:
            why.append(f"action {t}: destination {d} is a target's resting cell")
        elif o in ids and math.dist(scene.lattice.center(d), ids[o].xy_m) <= POSE_EQ_TOL_M:
            why.append(f"action {t}: {o} -> {d} is its initial pose (stay is not an action)")
    return why


def assignment(seq) -> dict:
    """Derived endpoint assignment x(r): only selected objects appear (the rest stay)."""
    return {o: d for o, d in seq}


def ordered_signature(scene, seq) -> tuple:
    """Stable identities and physical destination poses in action order; two orders of
    the same endpoints are different signatures."""
    return tuple((o, tuple(round(v, 9) for v in scene.lattice.center(d))) for o, d in seq)


def prefix_scenes(scene, seq) -> list:
    """Ideal prefix geometry: entry t is the scene before action t (earlier selected
    objects at their proposed destinations, all others at initial poses); the last entry
    is the final endpoint scene. Hypothetical substitutions only: no robot state, no
    achieved pose, not an execution witness. Overlaps (a destination still occupied by a
    later mover) are kept visible rather than repaired."""
    out = [scene]
    for o, d in seq:
        out.append(substitute_pose(out[-1], o, d))
    return out


def omit(seq, positions) -> tuple:
    """r \\ S: delete the actions at `positions`, keep the others' relative order. The
    omitted objects stay at their initial poses in every recomputed prefix."""
    drop = set(positions)
    return tuple(a for t, a in enumerate(seq) if t not in drop)


def omission_subsets(seq, max_size: int, max_tested: int) -> list:
    """Finite explanation budget: singletons, then pairs, ... up to max_size; capped and
    logged by the caller. The full set (= empty repair) is included when it fits."""
    subsets = [c for k in range(1, max_size + 1) for c in itertools.combinations(range(len(seq)), k)]
    return subsets[:max_tested]


def endpoint_distance(scene, seq) -> float:
    """C_d(r) = sum ||p_k - p_0|| / L_W with L_W the support diagonal (dimensionless,
    geometry-computed; not carried-path length and not robot effort)."""
    lat = scene.lattice
    lw = math.hypot(lat.cols * lat.pitch_m, lat.rows * lat.pitch_m)
    return sum(math.dist(lat.center(d), scene.get(o).xy_m) for o, d in seq) / lw


def task_score(scene, action: dict, seq, cfg: dict) -> float:
    """Deterministic geometric stand-in for the learned task score (no model exists at
    G4D): 0 if the derived endpoints leave every goal pose provably occupied, else 1."""
    return 0.0 if pr.goal_blockers(scene, assignment(seq), action, cfg) else 1.0


def rank_sequences(scene, action: dict, proposals: list, cfg: dict, v2: dict) -> dict:
    """Frozen V2 pool. The empty sequence is always considered first; each proposal is
    validated, deduplicated by ordered physical signature, statically screened on its
    derived assignment and thresholded, then ranked lexicographically by (move count,
    predicted joint travel, endpoint distance, feasibility energy, signature). Learned
    terms do not exist yet: they are recorded as None and contribute equally (0)."""
    # context identity: every physical scene field incl. robot joints, gap, attachment, lattice, legal cells, profile
    q, ctx = v2["proposal"], stable_hash({"scene": scene_hash(scene), "action": action})
    pool = [{"label": "empty", "sequence": EMPTY, "source": "EMPTY"}] + list(proposals)
    seen, rows, rejected = set(), [], []
    for k, p in enumerate(pool):
        seq, why = parse_sequence(scene, action, p["sequence"], q["move_cap"])
        sig = None if why else ordered_signature(scene, seq)
        if not why and sig in seen:
            why = ["duplicate ordered physical signature"]
        if why:
            rejected.append({"label": p["label"], "sequence": repr(p["sequence"]), "reasons": why})
            continue
        seen.add(sig)
        st = pr.static_endpoints(scene, assignment(seq), action["target_id"], cfg)["status"]
        rows.append({"label": p["label"], "sequence": seq, "assignment": assignment(seq), "moved": len(seq),
                     "source": p.get("source", "MANUAL"), "proposal_seed": p.get("seed"), "initial_context_hash": ctx,
                     "static_status": st, "task_score": task_score(scene, action, seq, cfg), "feasibility_energy": None,
                     "sample_energy": None, "predicted_joint_travel": None,
                     "endpoint_distance": round(endpoint_distance(scene, seq), 9), "declared_rank": k,
                     "signature": [[o, list(xy)] for o, xy in sig]})
    passed = [r for r in rows if r["static_status"] == mo.VALID and r["task_score"] >= q["task_score_threshold"]]
    passed.sort(key=lambda r: (r["moved"], r["predicted_joint_travel"] or 0.0, r["endpoint_distance"],
                               r["feasibility_energy"] or 0.0, [(xy, o) for o, xy in r["signature"]]))
    ranked = passed[: q["max_candidates_to_verify"]]
    for i, r in enumerate(ranked):
        r["frozen_rank"] = i
    return {"ranked": ranked, "screened_out": [r for r in rows if r not in passed], "rejected": rejected,
            "pool_hash": stable_hash([[list(a) for a in r["sequence"]] for r in ranked]),
            "unavailable_rank_terms": ["predicted_joint_travel", "feasibility_energy"]}
