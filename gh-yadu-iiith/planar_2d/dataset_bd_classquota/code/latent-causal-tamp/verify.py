"""Executable move-once repair verification (Methodology v1.4 §8, §10; Guide v1.6 G3).

A repair candidate is a set of real object IDs with physical destination cells and
an order. Each relocation is the full six-phase 3R skill (skill.plan_skill) planned
from the actual achieved scene and robot configuration; its record ends after the
retreat, except the last one, whose connector ends exactly at the frozen action's
start configuration. A moved ID becomes stay-only at its achieved pose (moved
history). The repair is EXECUTABLE_VERIFIED only if every relocation certifies and
the unchanged frozen gamma_0 replays clear in the final achieved scene.

Bounds: LB is A0's optimistic terminal optimum (every direct blocker must move;
nothing cheaper is terminal-valid), UB the move count of a verified witness;
MINIMUM_CERTIFIED only when they meet. A failed or exhausted search is never proof
that a repair is impossible.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field, replace

import numpy as np

import geometry as ge
import interventions as iv
import motion as mo
import skill as sk
import solve
from scene import Scene, scene_hash, scene_to_dict

EXECUTABLE, TERMINAL_INFEASIBLE, UNRESOLVED = "EXECUTABLE_VERIFIED", "CERTIFIED_TERMINAL_INFEASIBLE", "UNRESOLVED"


@dataclass(frozen=True)
class RepairState:
    scene: Scene
    q: tuple
    gap: float
    moved: tuple = ()                  # moved-object history H, in execution order
    episode: str = ""
    action_hash: str = ""


@dataclass
class Attempt:
    rank: int
    cost: int
    order: tuple                       # ((object id, destination cell), ...)
    status: str = UNRESOLVED
    stop: int | None = None            # index of the relocation that failed
    reason: str = ""
    states: list = field(default_factory=list)   # RepairState after each verified relocation
    skills: list = field(default_factory=list)   # SkillResult per attempted relocation
    final: object = None               # interventions.Snapshot of the final achieved scene


def start_state(scene: Scene, frozen, episode: str) -> RepairState:
    return RepairState(scene, tuple(scene.robot.q_rad), scene.robot.gripper_gap_m, (), episode, frozen.action_hash)


def relocate(robot, cfg, state: RepairState, obj_id: str, cell: int, key: str, q_goal=None):
    """One real relocation from the actual state. Returns (SkillResult, next state or None)."""
    if obj_id in state.moved:
        return sk.SkillResult(sk.INFEASIBLE, "MOVE_ONCE_VIOLATION", "MOVE_ONCE_RULE"), None
    res = sk.plan_skill(robot, state.scene, obj_id, cell, cfg, key, q_start=state.q, q_goal=q_goal)
    if res.status != sk.VERIFIED:
        return res, None
    xy = tuple(res.traj.place_xy)                  # achieved pose (exact within release tolerance)
    objs = tuple(replace(o, xy_m=xy, cell=cell) if o.id == obj_id else o for o in state.scene.objects)
    scene = replace(state.scene, objects=objs)
    return res, replace(state, scene=scene, q=tuple(res.traj.q[-1]), gap=float(res.traj.gap[-1]),
                        moved=state.moved + (obj_id,))


def execute(robot, cfg, frozen, s0: RepairState, rank: int, cost: int, order, memo: dict, key: str) -> Attempt:
    """Run one ordered move list; memo maps an exact prefix to its (deterministic) outcome."""
    att, state, q0 = Attempt(rank, cost, tuple(order)), s0, tuple(frozen.arrays["q"][0])
    for k, (oid, cell) in enumerate(order):
        last = k == len(order) - 1
        mkey = (tuple(order[: k + 1]), last)
        if mkey not in memo:
            memo[mkey] = relocate(robot, cfg, state, oid, cell, f"{key}/{'.'.join(f'{o}>{c}' for o, c in order[:k + 1])}"
                                  + ("/final" if last else ""), q0 if last else None)
        res, nxt = memo[mkey]
        att.skills.append(res)
        if nxt is None:
            att.stop, att.reason = k, f"{oid}->{cell}: {res.status} {res.reason}"
            att.status = UNRESOLVED                   # a failed order is never an impossibility proof
            return att
        state = nxt
        att.states.append(state)
    att.final = iv.evaluate(robot, cfg, frozen, iv.Snapshot("final", None, None, state.scene))
    at_start = np.array_equal(np.asarray(state.q), np.asarray(q0)) and state.gap == float(frozen.arrays["gap"][0])
    if att.final.F == mo.VALID and att.final.whole_replay == mo.VALID and at_start:
        att.status, att.reason = EXECUTABLE, "ALL_RELOCATIONS_CONNECTOR_AND_GAMMA0_REPLAY_CERTIFIED"
    else:
        att.reason = f"TERMINAL_{att.final.F}/WHOLE_{att.final.whole_replay}/AT_START_{at_start}"
    return att


def fresh_orders(moves, memo: dict, limit: int, pruned: list):
    """Up to `limit` move orders that are not dominated: an order whose prefix already
    failed would replay that identical (memoized, deterministic) failure, so it is
    counted in `pruned` instead of spending the order budget."""
    n = 0
    for order in itertools.permutations(moves):
        if n >= limit:
            return
        if any(memo.get((order[:k + 1], k == len(order) - 1), (None, 1))[1] is None for k in range(len(order))):
            pruned.append(order)
            continue
        n += 1
        yield order


def repair_cfg(cfg: dict) -> dict:
    """The skill budget used inside repair search (smaller per-skill failure allowance)."""
    return {**cfg, "skill2d": {**cfg["skill2d"], "max_plan_failures": cfg["repair2d"]["skill_plan_failures"]}}


def search(robot, cfg, frozen, scene: Scene, terminal: dict, key: str) -> dict:
    """A0 k-best assignments in nondecreasing cost x legal move orders, within an
    explicit relocation-planning budget; stops at the first executable witness."""
    rc, rcfg = cfg["repair2d"], repair_cfg(cfg)
    s0, memo, attempts, stop = start_state(scene, frozen, key), {}, [], "SEARCH_SPACE_EXHAUSTED"
    witness, pruned = None, []
    for rank, a in enumerate(terminal["assignments"]):
        for order in fresh_orders(solve.moves_of(scene, a["cells"]), memo, rc["max_orders"], pruned):
            if len(memo) >= rc["max_relocations"]:
                stop = "RELOCATION_BUDGET"
                break
            att = execute(robot, rcfg, frozen, s0, rank, a["cost"], order, memo, key)
            attempts.append(att)
            if att.status == EXECUTABLE:
                witness, stop = att, "WITNESS_FOUND"
                break
        if witness or stop == "RELOCATION_BUDGET":
            break
    lb, ub = terminal["lower_bound"], (witness.cost if witness else None)
    status = EXECUTABLE if witness else (TERMINAL_INFEASIBLE if terminal["hall"] else UNRESOLVED)
    minimum = "MINIMUM_CERTIFIED" if ub is not None and ub == lb else (
        "NOT_APPLICABLE" if terminal["hall"] else "MINIMUM_UNRESOLVED")
    return {"status": status, "lower_bound": lb, "upper_bound": ub, "minimality": minimum, "stop_reason": stop,
            "relocation_plans": len(memo), "pruned_orders": len(pruned), "witness": witness, "attempts": attempts}


def replay_step(robot, cfg, state: RepairState, oid: str, res, nxt: RepairState) -> bool:
    """One recorded relocation re-checked in its own starting scene: record starts at the
    achieved q and g, certifies clear, ends at the next state's q and g, and the moved
    object's achieved pose and the move-once history match."""
    delta, eps, evals = iv._params(cfg)
    arr = res.traj.arrays()
    ok = oid not in state.moved and np.array_equal(arr["q"][0], np.asarray(state.q)) and arr["gap"][0] == state.gap
    ok &= all(v.status == ge.CLEAR for v in mo.certify_trajectory(robot, state.scene, oid, arr, delta, eps, evals))
    ok &= nxt.scene.get(oid).xy_m == tuple(arr["place_xy"]) and nxt.moved == state.moved + (oid,)
    ok &= np.array_equal(np.asarray(nxt.q), arr["q"][-1]) and nxt.gap == float(arr["gap"][-1])
    return bool(ok)


def motion_valid(robot, cfg, state: RepairState, oid: str, res) -> bool:
    """A committed relocation record may run from `state`: it starts at the state's exact q
    and g, the object is not yet moved, and the record certifies in the state's scene."""
    delta, eps, evals = iv._params(cfg)
    arr = res.traj.arrays()
    return bool(oid not in state.moved and np.array_equal(arr["q"][0], np.asarray(state.q)) and arr["gap"][0] == state.gap
                and all(v.status == ge.CLEAR for v in mo.certify_trajectory(robot, state.scene, oid, arr, delta, eps, evals)))


def replay_repair(robot, cfg, frozen, s0: RepairState, att: Attempt) -> dict:
    """FROZEN_CONTROL re-check of a recorded episode: every relocation (replay_step), final
    pose at gamma_0's start, and the unchanged gamma_0 clear in the final scene."""
    frozen.verify()
    state, ok = s0, True
    for (oid, _), res, nxt in zip(att.order, att.skills, att.states):
        ok &= replay_step(robot, cfg, state, oid, res, nxt)
        state = nxt
    final = iv.evaluate(robot, cfg, frozen, iv.Snapshot("replay", None, None, state.scene))
    ok &= np.array_equal(np.asarray(state.q), frozen.arrays["q"][0]) and state.gap == float(frozen.arrays["gap"][0])
    ok &= final.F == final.whole_replay == mo.VALID
    return {"replay_passed": bool(ok), "moves": len(att.states), "final_F_fixed": final.F,
            "moved_history": list(state.moved)}


# ------------------------------------------------------------------ INITIAL_SCENE_ONE_SHOT_V1 (G3.5)
RELOCATED = "VERIFIED_FEASIBLE"        # complete relocation witness for one order (no target claim)


def realize(robot, cfg, s0: RepairState, order, memo: dict, key: str) -> Attempt:
    """New-mode relocation sequence: each skill planned from the achieved state and ending
    after its retreat. There is no connector to any reference start and no target query.
    The memo maps an executed prefix from s0 to its deterministic outcome."""
    att, state = Attempt(-1, len(order), tuple(order)), s0
    for k, (oid, cell) in enumerate(order):
        mkey = (tuple(order[: k + 1]), False)
        if mkey not in memo:
            memo[mkey] = relocate(robot, cfg, state, oid, cell, f"{key}/{'.'.join(f'{o}>{c}' for o, c in order[:k + 1])}")
        res, nxt = memo[mkey]
        att.skills.append(res)
        if nxt is None:
            att.stop, att.reason = k, f"{oid}->{cell}: {res.status} {res.reason}"
            return att                                  # UNRESOLVED for the assignment; this order only
        state = nxt
        att.states.append(state)
    att.status, att.reason = RELOCATED, "ALL_RELOCATIONS_CERTIFIED"
    return att


def plan_repair(robot, cfg, s0: RepairState, assignment: dict, key: str, used: dict) -> dict:
    """Relocation planner/verifier for one assignment. Works on immutable clones of S0
    (frozen dataclasses: a failed search cannot change the live state). Tries
    non-dominated orders within the order cap and the episode's shared relocation-plan
    budget (`used`). It has no access to any target evaluator."""
    rc, rcfg = cfg["repair2d"], repair_cfg(cfg)
    memo, pruned, attempts, stop = {}, [], [], "ORDERS_EXHAUSTED"
    for order in fresh_orders(sorted(assignment.items()), memo, rc["max_orders"], pruned):
        if used["plans"] + len(memo) >= rc["max_relocations"] and order:
            stop = "RELOCATION_BUDGET"
            break
        attempts.append(realize(robot, rcfg, s0, order, memo, key))
        if attempts[-1].status == RELOCATED:
            stop = "WITNESS_FOUND"
            break
    used["plans"] += len(memo)
    return {"witness": attempts[-1] if stop == "WITNESS_FOUND" else None, "attempts": attempts,
            "pruned_orders": len(pruned), "relocation_plans": len(memo), "stop": stop}


def executable_orders(robot, cfg, s0: RepairState, assignment: dict, key: str, n: int) -> dict:
    """Order diagnostic (G4A §8.2): keep searching non-dominated orders (same caps as
    plan_repair) until n complete relocation sequences; every attempt is returned."""
    rc, rcfg = cfg["repair2d"], repair_cfg(cfg)
    memo, pruned, attempts = {}, [], []
    for order in fresh_orders(sorted(assignment.items()), memo, rc["max_orders"], pruned):
        if len(memo) >= rc["max_relocations"]:
            break
        attempts.append(realize(robot, rcfg, s0, order, memo, key))
        if sum(a.status == RELOCATED for a in attempts) >= n:
            break
    return {"attempts": attempts, "pruned_orders": len(pruned), "relocation_plans": len(memo)}


def final_state(s0: RepairState, att: Attempt) -> RepairState:
    return att.states[-1] if att.states else s0


# ------------------------------------------------------------------ INITIAL_SCENE_ORDERED_ONE_SHOT_V2 (G4D)
FIXED_ORDER_POLICY = "FIXED_ORDER_GREEDY_SKILL2D_V1"


def plan_fixed_order(robot, cfg, s0: RepairState, sequence, key: str, used: dict) -> dict:
    """V2 relocation verifier: exactly the proposed order, never another (V1 order search
    above stays for V1 and named baselines only). Each action is the complete six-phase
    skill planned from the previous achieved state with the repair skill budget; grasps
    and paths vary only inside that skill's declared family and schedule, with no
    backtracking into earlier actions. The episode's relocation-planning allowance
    (`used["plans"]` against repair2d.max_relocations) is checked before EVERY planner
    invocation and every invocation is charged, failed or not. A sequence cut by the
    allowance or by a failed action is UNRESOLVED for this order only: no witness, no
    cost label; its achieved prefix is kept as a diagnostic. No target evaluator is
    reachable from here."""
    seq = tuple((o, d) for o, d in sequence)
    if any(not isinstance(o, str) or isinstance(d, bool) or not isinstance(d, int) for o, d in seq) \
            or len({o for o, _ in seq}) != len(seq):
        raise ValueError("fixed-order verification needs a canonical sequence (distinct str ids, int cells)")
    rcfg, cap = repair_cfg(cfg), cfg["repair2d"]["max_relocations"]
    att, state, calls, stop = Attempt(-1, len(seq), seq), s0, 0, "WITNESS_FOUND"
    for k, (oid, cell) in enumerate(seq):
        if used["plans"] >= cap:
            att.stop, att.reason, stop = k, f"relocation allowance {cap} exhausted before action {k}", "RELOCATION_BUDGET"
            break
        used["plans"] += 1
        calls += 1
        res, nxt = relocate(robot, rcfg, state, oid, cell, f"{key}/{'.'.join(f'{o}>{c}' for o, c in seq[:k + 1])}")
        att.skills.append(res)
        if nxt is None:
            att.stop, att.reason, stop = k, f"{oid}->{cell}: {res.status} {res.reason}", f"ACTION_{k}_UNRESOLVED"
            break
        state = nxt
        att.states.append(state)
    if stop == "WITNESS_FOUND":
        att.status, att.reason = RELOCATED, "ALL_RELOCATIONS_CERTIFIED"
    return {"witness": att if stop == "WITNESS_FOUND" else None, "attempts": [att], "pruned_orders": 0,
            "relocation_plans": calls, "stop": stop, "policy": FIXED_ORDER_POLICY, "order_scope": "THIS_ORDER_ONLY"}


# ------------------------------------------------------------------ fixtures
def blocked_state(robot, cfg, name: str, fx: dict):
    """(frozen gamma_0, reference scene, blocked scene, lineage). Archived G2 states are
    loaded read-only; new roots plan gamma_0 in their final scene, then block it with
    privileged one-edit relocations of existing IDs (explicit edits or the G2 rule)."""
    if "source_run" in fx:
        frozen, snaps = iv.load_archived(fx["source_run"], fx["family"])
        return frozen, snaps["ref"], snaps[fx["variant"]], [f"archived {fx['family']}/{fx['variant']}"]
    from scene import place_scene
    ref = place_scene(cfg, fx["root_id"], "ref", [fx["tasks"][0][0]], fx["obstacle_cells"])
    plan = sk.plan_skill(robot, ref, "t0", fx["tasks"][0][1], cfg, name)
    if plan.status != sk.VERIFIED:
        raise iv.FixtureUnavailable(f"reference action {plan.status}: {plan.reason}")
    frozen = iv.freeze(robot, cfg, ref, "t0", plan.traj.arrays())
    root = iv.evaluate(robot, cfg, frozen, iv.Snapshot("ref", None, None, ref))
    if "blocking_edits" in fx:
        snaps = [root]
        for k, (oid, src, dst) in enumerate(fx["blocking_edits"]):
            snaps.append(iv._child(robot, cfg, frozen, snaps[-1], f"block_{k}", iv.Edit("RELOCATE", oid, src, dst)))
    else:
        snaps = iv.add_blockers(robot, cfg, frozen, root, fx["blockers"])
    return frozen, ref, snaps[-1].scene, [f"{s.edit.obj_id} {s.edit.src_cell}->{s.edit.dst_cell}" for s in snaps[1:]]


def run_fixture(robot, cfg: dict, name: str, fx: dict) -> dict:
    """Terminal A0 under the declared domain(s), restoration lemma, executable search,
    explicit comparison assignments, forced occupied-destination chain, replay."""
    delta, k = cfg["suite"]["margin_m"], cfg["repair2d"]["k_best"]
    frozen, ref, scene, lineage = blocked_state(robot, cfg, name, fx)
    status = solve.cell_status(robot, cfg, frozen, scene)
    rest = solve.restoration(scene, ref, status)
    out = {"name": name, "frozen": frozen, "reference": ref, "scene": scene, "lineage": lineage, "status": status,
           "restoration": rest, "terminal": {}, "search": None, "compare": [], "forced": None, "replay": None}
    out["terminal"]["ALL_SUPPORTED"] = solve.a0(scene, solve.domains(scene, "ALL_SUPPORTED"), status, k, delta)
    if "restricted" in fx:
        excl = list(rest["assignment"].values())
        dom = solve.restricted_domain(scene, status, rest["direct_blockers"], fx["restricted"]["count"], excl)
        out["restricted_domain"] = dom
        out["terminal"]["RESTRICTED"] = solve.a0(scene, solve.domains(scene, "RESTRICTED", dom), status, k, delta)
    if not fx.get("execute", True):
        return out
    out["search"] = search(robot, cfg, frozen, scene, out["terminal"]["ALL_SUPPORTED"], name)
    s0, memo, rcfg = start_state(scene, frozen, name), {}, repair_cfg(cfg)
    extra = list(fx.get("compare", []))
    if rest["terminal_valid"]:
        extra.append({"label": "reference restoration (lemma witness)", "cells": rest["assignment"]})
    for c in extra:
        moves = [(o, cell) for o, cell in sorted(c["cells"].items()) if scene.get(o).cell != cell]
        for order in fresh_orders(moves, memo, cfg["repair2d"]["max_orders"], []):
            att = execute(robot, rcfg, frozen, s0, -1, len(moves), order, memo, f"{name}/compare")
            out["compare"].append({"label": c["label"], "attempt": att})
    if fx.get("forced_destination"):
        out["forced"] = forced_chain(robot, rcfg, frozen, scene, status, s0, memo, name, cfg["repair2d"]["max_orders"])
    if out["search"]["witness"]:
        out["replay"] = replay_repair(robot, cfg, frozen, s0, out["search"]["witness"])
    return out


def forced_chain(robot, cfg, frozen, scene, status, s0, memo, name, max_orders) -> dict | None:
    """Occupied preferred destination: force the first direct blocker into the nearest
    action-clear cell held by a non-blocking obstacle; A0 then has to move that owner
    as well. Every order is executed for real; the terminal lower bound is unchanged."""
    blockers = [o.id for o in scene.objects if o.kind == "obstacle" and status.get(o.cell) == mo.INVALID]
    owners = [o for o in scene.objects if o.kind == "obstacle" and o.id not in blockers and status.get(o.cell) == mo.VALID]
    if not blockers or not owners:
        return None
    a = scene.get(blockers[0])
    owner = min(owners, key=lambda o: (np.linalg.norm(np.subtract(o.xy_m, a.xy_m)), o.id))
    plan = solve.forced(scene, solve.domains(scene, "ALL_SUPPORTED"), status, a.id, owner.cell)
    if plan is None:
        return None
    moves = solve.moves_of(scene, plan["cells"])
    atts = [execute(robot, cfg, frozen, s0, -2, plan["cost"], order, memo, f"{name}/forced")
            for order in fresh_orders(moves, memo, max_orders, [])]
    return {"blocker": a.id, "owner": owner.id, "cell": owner.cell, "cost": plan["cost"], "moves": moves, "attempts": atts}


# ------------------------------------------------------------------ run record
def attempt_record(att: Attempt) -> dict:
    return {"rank": att.rank, "cost": att.cost, "order": [list(m) for m in att.order], "status": att.status,
            "failed_relocation": att.stop, "reason": att.reason, "skill_status": [s.status for s in att.skills],
            "moved_history": list(att.states[-1].moved) if att.states else [],
            "final_F_fixed": att.final and att.final.F, "final_whole_replay": att.final and att.final.whole_replay,
            "wall_clock_dependent": any(p.wall_clock_cut for s in att.skills for _, p in s.plans)}


def record(out: dict) -> dict:
    """JSON-ready fixture record (episodes themselves go to arrays.npz / scenes.json)."""
    st, s = out["status"], out["search"]
    rec = {k: out[k] for k in ("lineage", "restoration", "terminal", "replay") if k in out}
    rec.update(frozen_action_hash=out["frozen"].action_hash, restricted_domain=out.get("restricted_domain"),
               cell_status={v: sorted(c for c, x in st.items() if x == v) for v in (mo.INVALID, mo.UNRESOLVED)}
               | {"VALID_count": sum(x == mo.VALID for x in st.values())},
               compare=[{"label": c["label"], **attempt_record(c["attempt"])} for c in out["compare"]])
    if s:
        rec["search"] = {**{k: v for k, v in s.items() if k not in ("witness", "attempts")},
                         "attempts": [attempt_record(a) for a in s["attempts"]],
                         "witness": s["witness"] and attempt_record(s["witness"])}
    if out["forced"]:
        rec["forced"] = {**{k: out["forced"][k] for k in ("blocker", "owner", "cell", "cost")},
                         "attempts": [attempt_record(a) for a in out["forced"]["attempts"]]}
    return rec


def episodes(out: dict) -> list:
    """(tag, attempt) for every executable episode worth archiving for replay."""
    eps = [("witness", out["search"]["witness"])] if out["search"] and out["search"]["witness"] else []
    eps += [(f"compare{i}", c["attempt"]) for i, c in enumerate(out["compare"]) if c["attempt"].status == EXECUTABLE]
    if out["forced"]:
        eps += [(f"forced{i}", a) for i, a in enumerate(out["forced"]["attempts"]) if a.status == EXECUTABLE]
    return eps


def archive(res: dict, robot_hash: str):
    """(records, replay tasks, scenes, arrays) of a G3 run: gamma_0 in its reference
    scene, and for every executable episode each relocation in its own pre-move scene
    plus the unchanged gamma_0 in the final achieved scene (hash bound to its reference)."""
    records, tasks, scenes, arrays = {}, {}, {}, {}
    for name in sorted(res):
        r = res[name]
        if "unavailable" in r:
            records[name] = {"status": r["unavailable"]}
            continue
        records[name] = record(r)
        for key, s in (("reference", r["reference"]), ("blocked", r["scene"])):
            scenes[f"{name}.{key}"] = scene_to_dict(s)
        tasks[f"{name}.gamma0"] = {"action_hash": r["frozen"].action_hash, "target": "t0"}
        scenes[f"{name}.gamma0"] = scenes[f"{name}.reference"]
        arrays.update({f"{name}.gamma0__{k}": v for k, v in r["frozen"].arrays.items()})
        for tag, att in episodes(r):     # every relocation in its own pre-move scene; gamma_0 in the final one
            before = [r["scene"]] + [st.scene for st in att.states[:-1]]
            for k, ((oid, _), res_k, s) in enumerate(zip(att.order, att.skills, before)):
                tid, arr = f"{name}.{tag}.m{k}", res_k.traj.arrays()
                scenes[tid] = scene_to_dict(s)
                tasks[tid] = {"target": oid, "action_hash": mo.trajectory_hash(arr, oid, scene_hash(s), robot_hash)}
                arrays.update({f"{tid}__{a}": v for a, v in arr.items()})
            tid = f"{name}.{tag}.gamma0_final"
            scenes[tid] = scene_to_dict(att.states[-1].scene)
            tasks[tid] = {"target": "t0", "action_hash": r["frozen"].action_hash, "hash_scene": f"{name}.reference"}
            arrays.update({f"{tid}__{a}": v for a, v in r["frozen"].arrays.items()})
    return records, tasks, scenes, arrays
