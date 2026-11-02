"""Matched controlled interventions on a frozen target action (Methodology v1.4 §3,
§4, §6; Guide v1.6 G2).

A reference action gamma_0 is planned in its final scene, independently replayed and
frozen; its identity hash binds the trajectory to the *reference* scene. Descendant
snapshots differ from their parent by exactly one privileged diagnostic edit
(relocate an existing obstacle ID to a vacant legal cell, or remove it); edits are
never robot-executed repairs. Every snapshot is judged against the unchanged
gamma_0 by the per-object oracle (motion.object_check):

  y^a_i in {VALID, INVALID, UNRESOLVED}     fixed-action compatibility of object i
  F_fixed(S; gamma_0) = AND_i y^a_i          three-valued conjunction
  c_i = 1[y^a_i = INVALID] (None if unresolved), with deficit max(0, delta - d_min_i)
  d_I = c(S^I) - c(S),  Delta_I = F(S^I) - F(S)   (removed IDs contribute 0)

c_i is an operational direct-blocking indicator; it makes no claim of causal
identifiability. A whole-scene replay (motion.certify_trajectory with every object
present, plus the static placed-target pairs) must agree with the conjunction.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

import geometry as ge
import motion as mo
from config import robot_profile_hash
import skill as sk
from scene import Scene, occupancy, place_scene, scene_hash

DIAGNOSTIC = "PRIVILEGED_DIAGNOSTIC"
NUM = {mo.VALID: 1, mo.INVALID: 0, mo.UNRESOLVED: None}


@dataclass(frozen=True, eq=False)
class FrozenAction:
    target_id: str
    arrays: dict
    reference_scene_hash: str
    robot_profile_hash: str
    action_hash: str

    def verify(self) -> None:
        """The stored trajectory, target and reference scene still produce the hash."""
        h = mo.trajectory_hash(self.arrays, self.target_id, self.reference_scene_hash, self.robot_profile_hash)
        if h != self.action_hash:
            raise ValueError("frozen action identity mismatch: trajectory, target or reference scene changed")


def freeze(robot, cfg: dict, scene: Scene, target_id: str, arrays: dict) -> FrozenAction:
    """Freeze only an action that independently replays clear in its reference scene."""
    p = _params(cfg)
    if any(v.status != ge.CLEAR for v in mo.certify_trajectory(robot, scene, target_id, arrays, *p)):
        raise ValueError("reference action does not replay clear in its reference scene")
    rh, sh = robot_profile_hash(cfg), scene_hash(scene)
    return FrozenAction(target_id, arrays, sh, rh, mo.trajectory_hash(arrays, target_id, sh, rh))


def _params(cfg):
    return cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"], cfg["geometry"]["certify_max_evals"]


# ------------------------------------------------------------------ edits
@dataclass(frozen=True)
class Edit:
    kind: str                      # RELOCATE or REMOVE
    obj_id: str
    src_cell: int
    dst_cell: int | None = None
    role: str = DIAGNOSTIC         # privileged state edit, never a robot-executed relocation
    robot_executed: bool = False


def apply_edit(scene: Scene, edit: Edit, variant_id: str, delta: float) -> Scene:
    o = scene.get(edit.obj_id)
    if o.kind != "obstacle" or o.cell != edit.src_cell:
        raise ValueError(f"{edit.obj_id} is not an obstacle resting at cell {edit.src_cell}")
    if edit.kind == "RELOCATE":
        if edit.dst_cell not in scene.legal_cells or edit.dst_cell in occupancy(scene):
            raise ValueError(f"cell {edit.dst_cell} is not a vacant legal cell")
        objs = tuple(replace(x, xy_m=scene.lattice.center(edit.dst_cell), cell=edit.dst_cell)
                     if x.id == o.id else x for x in scene.objects)
    elif edit.kind == "REMOVE":
        objs = tuple(x for x in scene.objects if x.id != o.id)
    else:
        raise ValueError(edit.kind)
    child = replace(scene, variant_id=variant_id, objects=objs)
    audit_edit(scene, child, edit, delta)
    return child


def audit_edit(parent: Scene, child: Scene, edit: Edit, delta: float) -> None:
    """One-edit invariants: identical non-object state, identical IDs (or exactly the
    removed one missing), every other object bit-identical, no present-state overlap."""
    same = ("root_id", "lattice", "robot", "legal_cells", "domain_policy", "robot_profile_hash", "action_hash")
    if any(getattr(parent, f) != getattr(child, f) for f in same) or edit.role != DIAGNOSTIC or edit.robot_executed:
        raise ValueError("edit changed state other than one object pose, or claims robot execution")
    ids = [o.id for o in parent.objects if not (edit.kind == "REMOVE" and o.id == edit.obj_id)]
    if [o.id for o in child.objects] != ids:
        raise ValueError("edit inserted, deleted or renamed objects")
    if any(a != b for a, b in zip((o for o in parent.objects if o.id in ids), child.objects) if a.id != edit.obj_id):
        raise ValueError("edit changed an object other than its own")
    if edit.kind == "RELOCATE" and child.get(edit.obj_id).cell != edit.dst_cell:
        raise ValueError("edited object is not at its destination cell")
    if pair_status(child, delta, 0.0)[0] != mo.VALID:
        raise ValueError("edit produced overlapping objects")


def pair_status(scene: Scene, delta: float, eps: float) -> tuple[str, float]:
    """y^p over all present resting pairs (symmetric), and the closest pair gap [m]."""
    objs = scene.objects
    gaps = [math.dist(a.xy_m, b.xy_m) - a.radius_m - b.radius_m for i, a in enumerate(objs) for b in objs[i + 1:]]
    d = min(gaps, default=math.inf)
    return mo._band(d, delta, eps), d


def support_status(scene: Scene, o) -> str:
    """y^w: resting at a legal supported cell with the disc inside the support."""
    w, h = scene.lattice.cols * scene.lattice.pitch_m, scene.lattice.rows * scene.lattice.pitch_m
    x, y, r = o.xy_m[0], o.xy_m[1], o.radius_m
    inside = r <= x <= w - r and r <= y <= h - r
    return mo.VALID if o.cell in scene.legal_cells and inside else mo.INVALID


# ------------------------------------------------------------------ evaluation
@dataclass
class Snapshot:
    variant: str
    parent: str | None
    edit: Edit | None
    scene: Scene
    checks: dict = field(default_factory=dict)     # obstacle id -> motion.ObjectCheck
    F: str = mo.UNRESOLVED
    present_state: str = mo.UNRESOLVED             # AND of y^w and y^p
    whole_replay: str = mo.UNRESOLVED              # independent whole-scene replay


def evaluate(robot, cfg: dict, frozen: FrozenAction, snap: Snapshot) -> Snapshot:
    frozen.verify()
    delta, eps, evals = _params(cfg)
    t = snap.scene.get(frozen.target_id)
    q0 = frozen.arrays["q"][int(np.flatnonzero(frozen.arrays["phase"] == "carry")[0])]
    if np.linalg.norm(ge.tool_point(robot, q0, frozen.arrays["held_local"]) - t.xy_m) > 1e-9:
        raise ValueError("the target is not where the frozen action grasps it")
    lat = snap.scene.lattice
    support = (lat.cols * lat.pitch_m, lat.rows * lat.pitch_m)
    for o in snap.scene.objects:
        if o.kind == "obstacle":
            snap.checks[o.id] = mo.object_check(robot, frozen.arrays, (t.id, t.xy_m, t.radius_m),
                                                (o.id, o.xy_m, o.radius_m), support, delta, eps, evals)
    snap.F = mo._conjoin(c.status for c in snap.checks.values())
    snap.present_state = mo._conjoin([pair_status(snap.scene, delta, eps)[0]] +
                                     [support_status(snap.scene, o) for o in snap.scene.objects])
    snap.whole_replay = whole_scene_replay(robot, frozen, snap.scene, delta, eps, evals)
    return snap


def whole_scene_replay(robot, frozen: FrozenAction, scene: Scene, delta, eps, evals) -> str:
    """Independent check of F_fixed: G1 replay with every object present at once
    (full checker), plus the placed target against every obstacle (static pairs)."""
    seg = [v.status for v in mo.certify_trajectory(robot, scene, frozen.target_id, frozen.arrays, delta, eps, evals)]
    placed = [mo._band(math.dist(o.xy_m, frozen.arrays["place_xy"]) - o.radius_m - scene.get(frozen.target_id).radius_m,
                       delta, eps) for o in scene.objects if o.kind == "obstacle"]
    return mo._conjoin([{ge.CLEAR: mo.VALID, ge.COLLISION: mo.INVALID}.get(s, mo.UNRESOLVED) for s in seg] + placed)


def contributions(snap: Snapshot, delta: float) -> dict:
    """Per-ID operational contribution: blocking indicator, deficit and first offence."""
    out = {}
    for i, c in snap.checks.items():
        dmin = min(c.min_by_phase.values())
        out[i] = {"c": {mo.INVALID: 1, mo.VALID: 0}.get(c.status), "status": c.status,
                  "deficit_m": max(0.0, delta - dmin), "min_clearance_m": dmin, "first": c.first}
    return out


def edge_effect(parent: Snapshot, child: Snapshot, delta: float) -> dict:
    """d_I per ID (an absent object contributes 0) and Delta_I of the whole action."""
    cp, cc = contributions(parent, delta), contributions(child, delta)
    d = {}
    for i in sorted(set(cp) | set(cc)):
        a, b = cp.get(i, {"c": 0})["c"], cc.get(i, {"c": 0})["c"]
        d[i] = None if a is None or b is None else b - a
    fa, fb = NUM[parent.F], NUM[child.F]
    return {"d_I": d, "Delta_I": None if fa is None or fb is None else fb - fa}


# ------------------------------------------------------------------ families
class FixtureUnavailable(RuntimeError):
    """The requested construction does not exist under the selected geometry."""


def scan_cells(robot, cfg: dict, frozen: FrozenAction, scene: Scene) -> dict:
    """y^a of an obstacle-sized disc at every vacant legal cell, except the target's
    resting and placement cells (endpoint-occupancy conflicts are a later family)."""
    delta, eps, evals = _params(cfg)
    lat, t, occ = scene.lattice, scene.get(frozen.target_id), occupancy(scene)
    place = min(scene.legal_cells, key=lambda c: math.dist(lat.center(c), frozen.arrays["place_xy"]))
    r = cfg["suite"]["obstacle_radius_m"]
    return {c: mo.object_check(robot, frozen.arrays, (t.id, t.xy_m, t.radius_m), ("probe", lat.center(c), r),
                               (lat.cols * lat.pitch_m, lat.rows * lat.pitch_m), delta, eps, evals)
            for c in scene.legal_cells if c not in occ and c != place}


def build_family(robot, cfg: dict, frozen: FrozenAction, ref_scene: Scene, n_blockers: int) -> list:
    """Reference -> one blocker per edit -> correct blockers one by one, plus removal
    twins and an unrelated edit. Deterministic rules (report 0005):
      intrusion cell: vacant legal cell whose disc genuinely overlaps the swept full
        body (negative waypoint clearance, not just a delta-margin graze), preferring a
        first-offending phase not used by earlier blockers, then the deepest overlap,
        then the lowest cell ID;
      moved ID: a currently non-blocking obstacle nearest to that cell (then lowest ID);
      correction: the nearest vacant cell that is VALID for it, other than its
        original reference cell (an alternative admissible endpoint, not an undo)."""
    ref = evaluate(robot, cfg, frozen, Snapshot("ref", None, None, ref_scene))
    if ref.F != mo.VALID:
        raise FixtureUnavailable("the frozen action is not clear in its own reference scene")
    out = add_blockers(robot, cfg, frozen, ref, n_blockers)
    return _corrections(robot, cfg, frozen, ref, out, [s.edit.obj_id for s in out[1:]])


def add_blockers(robot, cfg: dict, frozen: FrozenAction, ref: Snapshot, n_blockers: int) -> list:
    """ref plus one blocking relocation per edit (the intrusion rule of build_family)."""
    out, cur, blockers, phases = [ref], ref, [], set()
    for k in range(n_blockers):
        cells = {c: v for c, v in scan_cells(robot, cfg, frozen, cur.scene).items()
                 if v.status == mo.INVALID and min(v.min_by_phase.values()) < 0}
        if not cells:
            raise FixtureUnavailable(f"no vacant cell overlaps the frozen sweep for blocker {k + 1}")
        dst = min(cells, key=lambda c: (cells[c].first["phase"] in phases, min(cells[c].min_by_phase.values()), c))
        movable = [o for o in cur.scene.objects if o.kind == "obstacle" and cur.checks[o.id].status == mo.VALID
                   and o.id not in blockers]
        o = min(movable, key=lambda o: (math.dist(o.xy_m, cur.scene.lattice.center(dst)), o.id))
        name = "block_" + "".join("abcdefgh"[: k + 1])
        cur = _child(robot, cfg, frozen, cur, name, Edit("RELOCATE", o.id, o.cell, dst))
        out.append(cur)
        blockers.append(o.id)
        phases.add(cells[dst].first["phase"])
    return out


def _child(robot, cfg, frozen, parent: Snapshot, name: str, edit: Edit) -> Snapshot:
    """Evaluate the one-edit child of a snapshot."""
    child = apply_edit(parent.scene, edit, name, cfg["suite"]["margin_m"])
    return evaluate(robot, cfg, frozen, Snapshot(name, parent.variant, edit, child))


def _corrections(robot, cfg, frozen, ref, out, blockers) -> list:
    """Diagnostic corrections one per edit, removal twins and unrelated edits."""
    full = cur = out[-1]
    for k, oid in enumerate(blockers):                       # diagnostic corrections, one per edit
        scan = scan_cells(robot, cfg, frozen, cur.scene)
        here, home = cur.scene.get(oid), ref.scene.get(oid).cell
        dst = min((c for c, v in scan.items() if v.status == mo.VALID and c != home),
                  key=lambda c: (math.dist(cur.scene.lattice.center(c), here.xy_m), c))
        cur = _child(robot, cfg, frozen, cur, "clear_" + "".join("abcdefgh"[: k + 1]),
                     Edit("RELOCATE", oid, here.cell, dst))
        out.append(cur)
    for k, oid in enumerate(blockers):                       # removal twins of the fully blocked snapshot
        out.append(_child(robot, cfg, frozen, full, f"remove_{'abcdefgh'[k]}",
                          Edit("REMOVE", oid, full.scene.get(oid).cell)))
    for parent in (ref, full):                               # unrelated edits leave the outcome unchanged
        scan = scan_cells(robot, cfg, frozen, parent.scene)
        o = min((o for o in parent.scene.objects if o.kind == "obstacle" and o.id not in blockers), key=lambda o: o.id)
        dst = min((c for c, v in scan.items() if v.status == mo.VALID),
                  key=lambda c: (math.dist(parent.scene.lattice.center(c), o.xy_m), c))
        out.append(_child(robot, cfg, frozen, parent, f"unrelated_{parent.variant}", Edit("RELOCATE", o.id, o.cell, dst)))
    return out


def reference_family(robot, cfg: dict, name: str, fx: dict) -> dict:
    """Plan gamma_0 in the fixture's final scene (all background present), freeze it
    only after independent replay, then build its intervention family."""
    src, dst = fx["tasks"][0]
    scene = place_scene(cfg, fx["root_id"], "ref", [src], fx["obstacle_cells"])
    res = sk.plan_skill(robot, scene, "t0", dst, cfg, name)
    out = {"name": name, "scene": scene, "plan": res, "frozen": None, "snaps": [], "status": "OK"}
    if res.status != sk.VERIFIED:
        out["status"] = f"REFERENCE_{res.status}:{res.reason}"
        return out
    out["frozen"] = freeze(robot, cfg, scene, "t0", res.traj.arrays())
    try:
        out["snaps"] = build_family(robot, cfg, out["frozen"], scene, fx["blockers"])
    except FixtureUnavailable as e:
        out["status"] = f"CONSTRUCTION_UNAVAILABLE:{e}"
    return out


def family_summary(fam: dict, cfg: dict) -> dict:
    delta = cfg["suite"]["margin_m"]
    fr, snaps = fam["frozen"], fam["snaps"]
    by = {s.variant: s for s in snaps}
    nodes = [{"variant": s.variant, "parent": s.parent, "scene_hash": scene_hash(s.scene),
              "edit": None if s.edit is None else s.edit.__dict__, "F_fixed": s.F, "whole_scene_replay": s.whole_replay,
              "oracle_agrees": s.F == s.whole_replay, "present_state": s.present_state,
              "action_hash_used": fr.action_hash if fr else None,
              "contributions": contributions(s, delta),
              "phase_status": {i: c.phase_status for i, c in s.checks.items()},
              "oracle_evals": sum(c.evals for c in s.checks.values()),
              **({} if s.parent is None else edge_effect(by[s.parent], s, delta))} for s in snaps]
    return {"status": fam["status"], "target": "t0", "task": [int(fam["scene"].get("t0").cell)],
            "frozen": None if fr is None else {"action_hash": fr.action_hash, "reference_scene_hash":
                                               fr.reference_scene_hash, "robot_profile_hash": fr.robot_profile_hash},
            "reference_plan": {"status": fam["plan"].status, "connectors_used": fam["plan"].used},
            "executed_repairs": 0, "moved_history": [], "nodes": nodes}


def family_gates(summary: dict, blockers: int) -> dict:
    """G2 pass conditions, evaluated from the recorded snapshots."""
    if summary["status"] != "OK":
        return {"family_available": False}
    n = {x["variant"]: x for x in summary["nodes"]}
    ids = [n[f"block_{'abcdefgh'[: k + 1]}"]["edit"]["obj_id"] for k in range(blockers)]
    full, cleared = n[f"block_{'abcdefgh'[: blockers]}"], n[f"clear_{'abcdefgh'[: blockers]}"]
    gates = {"reference_clear": n["ref"]["F_fixed"] == mo.VALID,
             "each_blocker_definitely_blocks": all(full["contributions"][i]["c"] == 1 for i in ids),
             "full_block_invalid": full["F_fixed"] == mo.INVALID,
             "all_corrections_restore_unchanged_action": cleared["F_fixed"] == mo.VALID,
             "oracle_agrees_with_whole_scene_replay": all(x["oracle_agrees"] for x in summary["nodes"]),
             "present_state_valid_everywhere": all(x["present_state"] == mo.VALID for x in summary["nodes"]),
             "same_frozen_hash_everywhere": len({x["action_hash_used"] for x in summary["nodes"]}) == 1,
             "unrelated_edits_change_nothing": all(x["Delta_I"] == 0 and not any(x["d_I"].values())
                                                   for v, x in n.items() if v.startswith("unrelated"))}
    if blockers >= 2:                                   # one correction: own conflict gone, action still blocked
        part = n["clear_a"]
        gates["partial_correction_keeps_failure"] = (part["F_fixed"] == mo.INVALID and part["d_I"][ids[0]] == -1
                                                     and part["contributions"][ids[1]]["c"] == 1)
    return gates


def load_archived(run_dir, family: str):
    """(FrozenAction, {variant: Scene}) from a G2 run, read-only; identity re-verified."""
    import json
    from pathlib import Path
    from scene import scene_from_dict
    run = Path(run_dir)
    res, scenes = json.loads((run / "result.json").read_text()), json.loads((run / "scenes.json").read_text())
    data = np.load(run / "arrays.npz", allow_pickle=False)
    f = res["families"][family]["frozen"]
    frozen = FrozenAction("t0", {k.split("__")[1]: data[k] for k in data.files if k.startswith(family + "__")},
                          f["reference_scene_hash"], f["robot_profile_hash"], f["action_hash"])
    frozen.verify()
    return frozen, {k.split("/")[1]: scene_from_dict(v) for k, v in scenes.items() if k.startswith(family + "/")}
