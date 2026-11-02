"""G4C-D2-P0 witness-first constructive dataset: STATIC PRECHECK ONLY (D2 contract §6).

A new, separately versioned construction lineage (`G4C-D2`), not an edit of G4C-P0/P1.
From an ARCHIVED, replay-validated target-only witness (full planar 3R robot, gripper,
carried target; goal + safe retreat) it proposes legal obstacle insertions that intersect
the RECORDED full-body sweep, nested variants, and a frozen candidate repair/intervention
catalogue per variant. Everything here is deterministic static geometry:
`motion.certify_trajectory` re-certifies the recorded segments in the edited scene; the
initial robot is checked with `geometry.Checker`. No planner, IK search, skill, verifier,
target evaluator or teacher is imported or called.

Semantics (the guard this precheck exists for): `blocks_nominal_witness` describes ONE
recorded path. It is never a feasibility label: another grasp, IK branch or path may
succeed, so a constructed scene's target status is NOT_RUN until an independent teacher
decides it, and only a sound full-scope certificate can make it NEGATIVE.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np

import config as cf
import data
import effects as ef
import g4c
import geometry as ge
import motion as mo
import proposals as pr
import sequences as sq
from scene import Obj, scene_from_dict, scene_hash

VERSION = "G4C-D2-v0"
STATUSES = ("PRECHECK_STATIC", "BASE_WITNESS_REPLAYED", "NOT_RUN", "ARCHIVED_EVIDENCE")
BASES = (   # (base label, plan file, root key, witness npz, archived unit, scene lineage)
    ("g4c_r00", "runs/g4c_pilot/pilot_plan.json", "g4c_r00", "runs/g4c_pilot/units/seq__g4c_r00__empty/target_witness.npz",
     "G4C-P1 seq:g4c_r00:empty", "G4C:g4c_r00 (seed 2026102001)"),
    ("g4d_R0", "runs/g4d_evidence/final_plan.json", "R0", "runs/g4d_evidence/units/seq__R0__empty/target_witness.npz",
     "G4D seq:R0:empty", "G2_D1 (R0 = reference scene)"),
    ("g4d_R1", "runs/g4d_evidence/final_plan.json", "R1", "runs/g4d_evidence/units/seq__R1__empty/target_witness.npz",
     "G4D seq:R1:empty", "G2_D1 (R1 = blocked variant of the same reference)"))


def _sha(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _tol(cfg):
    return cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"], cfg["geometry"]["certify_max_evals"]


# ------------------------------------------------------------------ archived base witness
def load_base(spec, root: Path = Path(".")) -> dict:
    fam, plan_f, key, npz_f, unit, lineage = spec
    r = json.loads((root / plan_f).read_text())["roots"][key]
    z = np.load(root / npz_f)
    arr = {k.split("__")[1]: z[k] for k in z.files if k.startswith("r0__")}
    witness_hash = cf.stable_hash({k: np.asarray(v).tolist() for k, v in sorted(arr.items())})
    return {"family": fam, "scene": scene_from_dict(r["scene"]), "goal_cell": r["goal_cell"], "arrays": arr,
            "provenance": {"plan_file": plan_f, "plan_file_sha256": _sha(root / plan_f), "root_key": key, "witness_file": npz_f,
                           "witness_file_sha256": _sha(root / npz_f), "witness_hash": witness_hash, "archived_unit": unit,
                           "status": "ARCHIVED_EVIDENCE"},
            "lineage": lineage, "family_id": cf.stable_hash(lineage.split(" ")[0])}   # split key: shared scene lineage


def replay_base(robot, cfg, base) -> dict:
    """Independent replay: every recorded segment certified CLEAR in its own scene, record
    starts at the scene's q0/gap, ends open-handed in retreat, places within the goal tol."""
    s, a = base["scene"], base["arrays"]
    v = mo.certify_trajectory(robot, s, "t0", a, *_tol(cfg))
    goal = s.lattice.center(base["goal_cell"])
    chk = {"all_segments_clear": all(x.status == ge.CLEAR for x in v),
           "starts_at_q0": bool(np.array_equal(a["q"][0], np.asarray(s.robot.q_rad)) and a["gap"][0] == s.robot.gripper_gap_m),
           "ends_in_safe_retreat": str(a["phase"][-1]) == "retreat" and float(a["gap"][-1]) == robot.open_gap,
           "goal_within_tolerance": math.dist(a["place_xy"], goal) <= cfg["gripper2d"]["release_xy_tolerance_m"],
           "attached_only_in_carry": all(bool(m) == (str(p) == "carry") for p, m in zip(a["phase"], a["attached"]))}
    return {"checks": chk, "segments": len(v), "status": "BASE_WITNESS_REPLAYED" if all(chk.values()) else "REPLAY_FAILED"}


# ------------------------------------------------------------------ nominal-sweep diagnostic (path-specific, never a label)
def with_obstacles(scene, cells) -> object:
    """Clone with NEW obstacle objects at `cells` (fresh ids after the existing ones)."""
    n = 1 + max(int(o.id[1:]) for o in scene.objects if o.kind == "obstacle")
    add = tuple(Obj(f"o{n + i:02d}", "obstacle", scene.objects[0].radius_m, scene.lattice.center(c), c)
                for i, c in enumerate(cells))
    return replace(scene, objects=scene.objects + add)


def nominal_diagnostic(robot, cfg, scene, arrays) -> dict:
    v = mo.certify_trajectory(robot, scene, "t0", arrays, *_tol(cfg))
    cols = [{"segment": i, "phase": str(arrays["phase"][i]), "s": x.s, "pair": x.pair} for i, x in enumerate(v)
            if x.status == ge.COLLISION]
    unres = [i for i, x in enumerate(v) if x.status == ge.UNRESOLVED]
    blocks = "YES" if cols else ("UNVERIFIED" if unres else "NO")
    return {"blocks_nominal_witness": blocks, "first_interruption": cols[0] if cols else None, "collision_segments": cols,
            "unresolved_segments": unres, "segments": len(v), "target_task_status": "NOT_RUN",
            "scope": "the single recorded witness path only; not a task-feasibility label"}


def role(cell: int, goal_cell: int, diag: dict) -> str:
    if cell == goal_cell:
        return "target_goal_occupant"
    if diag["blocks_nominal_witness"] != "YES":
        return "harmless_nearby" if diag["blocks_nominal_witness"] == "NO" else "unverified"
    f = diag["first_interruption"]
    body, ph = f["pair"].split("|")[0], f["phase"]
    if body == "held":
        return "transport_path"
    if ph in ("release", "retreat"):
        return "retreat_blocker"
    if body in ("palm", "rail", "finger_l", "finger_r") and ph in ("pregrasp", "approach", "close_attach"):
        return "grasp_access_blocker"
    return "arm_link_sweep_blocker"


def insertion_attempts(robot, cfg, base, scene=None) -> list:
    """Every cell, in id order: accepted only if free, supported, distinct and the robot at
    q0 stays clear with the new obstacle; the nominal diagnostic is attached (no label)."""
    s, out = scene or base["scene"], []
    occ = {o.cell for o in s.objects}
    d, e, _ = _tol(cfg)
    for c in range(len(s.legal_cells)):
        rec = {"family": base["family"], "cell": c, "accepted": False}
        if c in occ:
            out.append({**rec, "reason": "occupied by an existing object"})
            continue
        s2 = with_obstacles(s, [c])
        st = ge.Checker(robot, ge.make_world(s2), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m)
        if st.status != ge.CLEAR:
            out.append({**rec, "reason": f"initial robot at q0 not clear ({st.status} {st.pair})"})
            continue
        diag = nominal_diagnostic(robot, cfg, s2, base["arrays"])
        out.append({**rec, "accepted": True, "reason": "PRECHECK_STATIC", "role": role(c, base["goal_cell"], diag),
                    "blocks_nominal_witness": diag["blocks_nominal_witness"], "first_interruption": diag["first_interruption"],
                    "n_collision_segments": len(diag["collision_segments"]), "target_task_status": "NOT_RUN"})
    return out


# ------------------------------------------------------------------ nested variants (construction plan level)
def choose_blockers(attempts: list) -> dict:
    """Frozen rule: A = earliest-interrupting non-goal blocker; B, C = earliest blockers of
    other roles (else next earliest), pairwise non-adjacent; N = certified-clear nuisance
    farthest from every blocker; G = goal occupant if eligible. Ties by cell id."""
    ok = [a for a in attempts if a["accepted"]]
    blk = sorted((a for a in ok if a["blocks_nominal_witness"] == "YES" and a["role"] != "target_goal_occupant"),
                 key=lambda a: (a["first_interruption"]["segment"], a["cell"]))
    adj = lambda x, y: abs(x % 8 - y % 8) <= 1 and abs(x // 8 - y // 8) <= 1  # noqa: E731
    pick = []
    for a in blk:
        if len(pick) == 3:
            break
        if any(adj(a["cell"], p["cell"]) for p in pick):
            continue
        if a["role"] in {p["role"] for p in pick} and any(b["role"] not in {p["role"] for p in pick} | {a["role"]}
                                                          for b in blk if not any(adj(b["cell"], p["cell"]) for p in pick)):
            continue
        pick.append(a)
    clear = [a for a in ok if a["blocks_nominal_witness"] == "NO" and a["role"] == "harmless_nearby"]
    nuis = max(clear, key=lambda a: (min((abs(a["cell"] % 8 - p["cell"] % 8) + abs(a["cell"] // 8 - p["cell"] // 8))
                                         for p in pick) if pick else 0, -a["cell"]), default=None)
    goal = next((a for a in ok if a["role"] == "target_goal_occupant"), None)
    return {"A": pick[0] if pick else None, "B": pick[1] if len(pick) > 1 else None, "C": pick[2] if len(pick) > 2 else None,
            "N": nuis, "G": goal}


SPECS = (("base", ()), ("+A", ("A",)), ("+B", ("B",)), ("+A+B", ("A", "B")), ("+A+B+C", ("A", "B", "C")), ("+N", ("N",)),
         ("+A+N", ("A", "N")), ("+G", ("G",)), ("+A+S", ("A", "S")))
ALT_STARTS = [[math.pi / 2, -math.pi / 2, -math.pi / 2], [math.pi / 4, math.pi / 2, math.pi / 2]]


def staging(robot, cfg, base, chosen):
    """S = an obstacle on A's first G4C destination in the +A scene (occupied staging slot,
    D2-F4), if that cell is free and the robot at q0 stays clear; else None."""
    if chosen["A"] is None:
        return None
    sA = with_obstacles(base["scene"], [chosen["A"]["cell"]])
    oA = next(o.id for o in sA.objects if o.cell == chosen["A"]["cell"])
    d = g4c._dests(sA, oA, sA.get("t0").cell, base["goal_cell"], 1)
    if not d:
        return None
    at = insertion_attempts(robot, cfg, base, sA)[d[0]]
    return {**at, "role": "staging_slot_occupant"} if at["accepted"] else None


def nested_variants(robot, cfg, base, chosen) -> list:
    out, chosen = [], {**chosen, "S": staging(robot, cfg, base, chosen)}
    for name, keys in SPECS:
        if any(chosen[k] is None for k in keys):
            out.append({"variant": name, "status": "CONSTRUCTION_UNAVAILABLE", "missing": [k for k in keys if chosen[k] is None]})
            continue
        s = with_obstacles(base["scene"], [chosen[k]["cell"] for k in keys])
        d, e, _ = _tol(cfg)
        q0 = ge.Checker(robot, ge.make_world(s), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status
        diag = nominal_diagnostic(robot, cfg, s, base["arrays"])
        ids = {k: next(o.id for o in s.objects if o.cell == chosen[k]["cell"]) for k in keys}   # Scene sorts by id
        out.append({"variant": name, "status": "PRECHECK_STATIC" if q0 == ge.CLEAR else "REJECTED_Q0_NOT_CLEAR",
                    "added": {k: {"id": ids[k], "cell": chosen[k]["cell"], "role": chosen[k]["role"]} for k in keys},
                    "scene": s, "scene_hash": scene_hash(s), "q0_clear": q0 == ge.CLEAR, "nominal": diag,
                    "goal_cell": base["goal_cell"], "family_id": base["family_id"], "target_task_status": "NOT_RUN"})
    return out


# ------------------------------------------------------------------ predeclared catalogue per variant
def variant_queries(robot, cfg, v) -> dict:
    """Frozen geometric rule before outcomes: participants = the added obstacles (blockers
    first, then nuisance/goal occupant); destinations via the G4C rule (initial scene only,
    no witness information); sequences = [], singles to two destinations, both orders of
    each pair, three orders of the triple, omission remainders; fixed-robot scene edits =
    singletons, first joint pair, nuisance; B/C catalogue = first <= 3 eligible edits."""
    s, g = v["scene"], v["goal_cell"]
    t = s.get("t0").cell
    act = data.make_query(cfg, "d2", s, "t0", g, {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
    parts = [v["added"][k]["id"] for k in ("A", "B", "C", "G", "S", "N") if k in v["added"]]
    used, dest = set(), {}
    for o in parts:
        dest[o] = g4c._dests(s, o, t, g, 2, used)
        used |= set(dest[o])
    raw = [[]] + [[[o, c]] for o in parts for c in dest[o][:2]]
    if "S" in v["added"]:                                            # occupied staging slot P = S's cell
        a, st, P = v["added"]["A"]["id"], v["added"]["S"]["id"], v["added"]["S"]["cell"]
        raw += [[[a, P]], [[st, dest[st][0]], [a, P]], [[a, P], [st, dest[st][0]]]]
    blk = [o for k, o in ((k, v["added"][k]["id"]) for k in ("A", "B", "C") if k in v["added"])]
    for x, y in zip(blk, blk[1:]):
        raw += [[[x, dest[x][0]], [y, dest[y][0]]], [[y, dest[y][0]], [x, dest[x][0]]]]
    if len(blk) == 3:
        tri = [[o, dest[o][0]] for o in blk]
        raw += [tri, [tri[1], tri[0], tri[2]], [tri[2], tri[1], tri[0]]]
    pool, seen = [], set()
    for x in raw:
        seq, why = sq.parse_sequence(s, act, x, 3)
        if why or seq in seen:
            continue
        seen.add(seq)
        pool.append({"sequence": [list(a) for a in seq], "origin": "declared"})
    for p in [p for p in pool if len(p["sequence"]) >= 2]:
        full = tuple(tuple(a) for a in p["sequence"])
        for drop in [(i,) for i in range(len(full))] + ([(i, j) for i in range(len(full)) for j in range(i + 1, len(full))]
                                                       if len(full) == 3 else []):
            rem = sq.omit(full, drop)
            if rem not in seen:
                seen.add(rem)
                pool.append({"sequence": [list(a) for a in rem], "origin": f"omission_of:{g4c._sig(full)}"})
    for p in pool:
        p["static_status"] = pr.static_endpoints(s, sq.assignment(p["sequence"]), "t0", cfg)["status"]
        p["endpoint_distance"] = round(sq.endpoint_distance(s, [tuple(a) for a in p["sequence"]]), 9)
        p["status"] = "NOT_RUN"
    movers = blk + [v["added"][k]["id"] for k in ("G", "S") if k in v["added"]]
    edits = [({o: dest[o][0]}, f"singleton:{o}") for o in movers if dest[o]]
    if len(blk) >= 2:
        edits.append(({blk[0]: dest[blk[0]][0], blk[1]: dest[blk[1]][0]}, f"joint:{blk[0]}+{blk[1]}"))
    if "N" in v["added"] and dest[v["added"]["N"]["id"]]:
        edits.append(({v["added"]["N"]["id"]: dest[v["added"]["N"]["id"]][0]}, "nuisance"))
    sq_ = []
    for e, kind in edits:
        sub = ef.pose_substitution(robot, cfg, s, act, e)
        sq_.append({"edit": e, "kind": kind, "eligible": sub["eligible"], "reasons": sub["reasons"],
                    "robot_unchanged": sub["changed_fields"]["robot_unchanged"], "status": "NOT_RUN"})
    cat = [q["kind"] for q in sq_ if q["eligible"]][:3]
    for p in pool:
        p["bc_layout"] = g4c.layout(p["sequence"], cat)
    starts = []
    if v["variant"] == "base":                                       # D2-F9: same objects, different initial joints
        for j, q in enumerate(ALT_STARTS):
            rs = ef.robot_start_substitution(robot, cfg, s, act, q)
            starts.append({"q_start": rs["q_to"], "eligible": rs["eligible"], "rule_index": j, "status": "NOT_RUN"})
    return {"pool": pool, "scene_queries": sq_, "catalogue": cat, "participants": parts, "destinations": dest,
            "robot_starts": starts}


# ------------------------------------------------------------------ release semantics (the guard) and splits
def root_release_label(nominal_blocked: bool, target_witness: bool | None, sound_certificate: bool | None) -> str:
    """No-repair label of a CONSTRUCTED scene. Nominal obstruction never decides anything."""
    if target_witness and sound_certificate:
        return "CONTRADICTION"
    if target_witness:
        return "REROUTABLE_POSITIVE" if nominal_blocked else "POSITIVE"
    if sound_certificate:
        return "NEGATIVE"
    return "NOT_RUN" if target_witness is None and sound_certificate is None else "UNKNOWN"


def pool_release(labels: dict, declared: list) -> str:
    """A declared closed pool is DECIDED_COMPLETE only if every declared item is decided;
    otherwise the WHOLE pool is PARTIAL_OBSERVATION (no item is dropped after labels)."""
    if set(labels) != set(declared):
        return "INVALID_POOL (declared items missing or added)"
    return "DECIDED_COMPLETE" if all(labels[k] in ("POSITIVE", "NEGATIVE") for k in declared) else "PARTIAL_OBSERVATION"


def substantive_pool(labels: dict) -> bool:
    """>= 2 distinct nonempty repairs with a POS/NEG distinction or two competing POS."""
    ne = {k: v for k, v in labels.items() if k != "empty" and v in ("POSITIVE", "NEGATIVE")}
    return len(ne) >= 2 and "POSITIVE" in ne.values()


def family_split(family_id: str, val=0.15, test=0.15) -> str:
    """Split keyed by the base-witness lineage only: every descendant inherits it."""
    u = int(hashlib.sha256(family_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if u < test else "validation" if u < test + val else "train"
