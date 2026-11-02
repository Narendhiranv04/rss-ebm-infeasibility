"""P15-3: tri-state, multimodal, finite-graph lexicographic rearrangement search (non-learned; guidance only orders expansion).

Statuses (never a bare None): VERIFIED; CERTIFIED_INFEASIBLE (sound, scoped: certificate kind recorded); STATIC_INVALID (that destination
only); UNKNOWN_PLANNER / UNKNOWN_BUDGET / UNKNOWN_NO_LIBRARY_WITNESS / UNKNOWN_TRANSIT (no impossibility established).
State = (object -> cell for EVERY object, robot q, moved set). An edge = (object, slot, one certified motion with its transit); the edge
provider returns ALL certified library modes with distinct terminal q (up to K_MODES) plus an online witness if no mode certifies.
Objective lexmin (relocations, J), J = ordered.joint_travel over transits, relocations and the final target record. Iterative deepening over
the move count; inside a depth, DFS in guidance order with travel branch-and-bound (J >= 0) and exact-state dominance (same occupancy, same
q, same moved set). Flags: search_enumeration_complete (per depth, no budget truncation) and all_shorter_depths_physically_decided (every
edge / goal at shallower depths VERIFIED or CERTIFIED / STATIC_INVALID); MIN_MOVES / MIN_TRAVEL claims only when they hold."""
from __future__ import annotations

import time

import numpy as np

import config as cf

V, CI, SI = "VERIFIED", "CERTIFIED_INFEASIBLE", "STATIC_INVALID"
U_PLAN, U_BUD, U_LIB, U_TR = "UNKNOWN_PLANNER", "UNKNOWN_BUDGET", "UNKNOWN_NO_LIBRARY_WITNESS", "UNKNOWN_TRANSIT"
DECIDED = (V, CI, SI)
K_MODES, K_DEST, N_MAX = 3, 4, 3
SIM_THR = 1.0499131944444444                                  # TRANSFER_SUMMARY.json threshold.value (DEV-fitted)


def state_key(cells, q, moved):
    return (tuple(sorted(cells.items())), tuple(np.round(np.asarray(q, float), 12)), tuple(sorted(moved)))


def search(provider, order, movers, slots, start_cells, start_q, n_max=N_MAX) -> dict:
    """provider.goal(cells, q) -> (status, record, travel); provider.edges(cells, q, o, d) -> [(status, records, q2, travel, meta)];
    order(cells, q, moved, cands) -> cands reordered (guidance: reorder only). Returns plan, depth, flags, per-status counts."""
    log = {"edges": {}, "goals": {}, "pruned_bnb": 0, "pruned_dominated": 0, "nodes": 0, "terminal_q_alternatives": 0}
    t0 = time.process_time()
    best = {"plan": None}
    undecided_at = set()

    def rec_status(kind, depth, st):
        log[kind][st] = log[kind].get(st, 0) + 1
        if st not in DECIDED:
            undecided_at.add(depth)

    def dfs(cells, q, moved, steps, recs, J, depth_left, depth_total, seen):
        log["nodes"] += 1
        if best["plan"] is not None and best["plan"]["moves"] == depth_total and J >= best["plan"]["travel"]:
            log["pruned_bnb"] += 1
            return
        k = state_key(cells, q, moved)
        if k in seen and seen[k] <= J:
            log["pruned_dominated"] += 1
            return
        seen[k] = J
        if depth_left == 0:
            st, rec, tj = provider.goal(cells, q)
            rec_status("goals", depth_total, st)
            if st == V and (best["plan"] is None or J + tj < best["plan"]["travel"]):
                best["plan"] = {"moves": depth_total, "travel": J + tj, "steps": steps, "records": recs + [rec]}
                log.setdefault("anytime", []).append({"cpu_s": time.process_time() - t0, "moves": depth_total, "travel": J + tj})
            return
        cands = [(o, d) for o in movers if o not in moved for d in slots(cells, o)]
        for o, d in order(cells, q, moved, cands):
            if provider.truncated():
                return
            outs = provider.edges(cells, q, o, d)
            for st, *_ in outs:
                rec_status("edges", depth_total, st)
            ver = [x for x in outs if x[0] == V]
            log["terminal_q_alternatives"] += max(0, len(ver) - 1)
            for st, rs, q2, tj, meta in sorted(ver, key=lambda x: x[3]):
                dfs({**cells, o: d}, q2, moved | {o}, steps + [{"object": o, "dest": d, **meta}], recs + rs, J + tj, depth_left - 1, depth_total, seen)

    flags = {}
    for n in range(n_max + 1):
        trunc0 = provider.truncated()
        dfs(dict(start_cells), np.asarray(start_q, float), frozenset(), [], [], 0.0, n, n, {})
        flags[n] = {"search_enumeration_complete": not provider.truncated() and not trunc0, "all_decided": n not in undecided_at}
        if best["plan"] is not None or provider.truncated():
            break
    found = best["plan"]
    nf = found["moves"] if found else None
    shorter_ok = found is not None and all(flags[m]["search_enumeration_complete"] and flags[m]["all_decided"] for m in range(nf))
    depth_ok = found is not None and flags[nf]["search_enumeration_complete"] and flags[nf]["all_decided"]
    claim = ("MIN_MOVES_AND_MIN_TRAVEL_WITHIN_FULLY_ADJUDICATED_FINITE_GRAPH" if shorter_ok and depth_ok else
             "MIN_MOVES_WITHIN_FULLY_ADJUDICATED_FINITE_GRAPH" if shorter_ok else "BEST_VERIFIED_WITHIN_BUDGET" if found else "NO_PLAN_FOUND")
    return {"found": found is not None, "plan": found, "flags": {str(k): v for k, v in flags.items()}, "claim": claim,
            "min_moves_proven": shorter_ok, "min_travel_proven": shorter_ok and depth_ok, "log": log}


# ------------------------------------------------------------------ real physics provider (exact certifier + planner + sound certificates)
class Physics:
    BUDGET = {"motion_checks": 120, "online_moves": 6, "goal_plans": 16}

    def __init__(self, root, idx=None, lookup="combined", modal_only=False, budget=None, cpu_cap=None):
        import geometry as ge
        import p6_scenes as p6
        import skill as sk
        from scene import scene_from_dict
        self.cfg = cf.load_config("config.toml")
        self.robot, self.p = ge.make_robot(self.cfg), sk.params(self.cfg)
        self.scr, self.idx, self.lookup, self.modal_only = p6.Screens(self.cfg, self.robot), idx, lookup, modal_only
        self.s0, self.goal_cell = scene_from_dict(root["scene"]), root["goal_cell"]
        self.budget, self.cpu_cap, self.c0 = dict(budget or self.BUDGET), cpu_cap, time.process_time()
        self.used = {k: 0 for k in self.budget}
        self.memo, self.stats = {}, {"lookup_hits": 0, "lookup_candidates_tried": 0, "certificate_checks": 0, "rare_mode_used": 0}
        self._trunc, self.exhausted = False, set()

    def truncated(self):
        """Search-wide stop: CPU cap, or goal-plan budget exhausted (no new plan can be verified). An exhausted motion-check or
        online budget only makes the affected edges UNKNOWN_BUDGET (P15 PREREG amendment 1)."""
        if self.cpu_cap is not None and time.process_time() - self.c0 > self.cpu_cap or "goal_plans" in self.exhausted:
            self._trunc = True
        return self._trunc

    def _take(self, k):
        if self.used[k] >= self.budget[k] or self.truncated():
            self.exhausted.add(k)
            return False
        self.used[k] += 1
        return True

    def scene(self, cells, q):
        from dataclasses import replace
        objs = tuple(replace(o, cell=cells[o.id], xy_m=self.s0.lattice.center(cells[o.id])) for o in self.s0.objects)
        return replace(self.s0, objects=objs, robot=replace(self.s0.robot, q_rad=tuple(map(float, q))))

    def _key(self, *parts):
        return "p15/" + cf.stable_hash([str(x) for x in parts] + [str(self.budget), self.robot.lengths])[7:23]

    def goal(self, cells, q):
        k = ("goal", state_key(cells, q, ()))
        if k not in self.memo or self.memo[k][0] == U_BUD:
            self.memo[k] = self._goal(cells, q, k)
        return self.memo[k]

    def _goal(self, cells, q, k):
        import p15_retarget as pr
        import skill as sk
        import ordered as od
        s = self.scene(cells, q)
        self.stats["certificate_checks"] += 1
        c = pr.certified(self.cfg, self.robot, s, "t0", self.goal_cell)
        if c:
            return (CI, {"certificates": c}, 0.0)
        if not self._take("goal_plans"):
            return (U_BUD, None, 0.0)
        r = sk.plan_skill(self.robot, s, "t0", self.goal_cell, self.cfg, self._key(*k), q_start=np.asarray(q, float), q_goal=None)
        if r.status == sk.VERIFIED and r.traj is not None:
            a = r.traj.arrays()
            return (V, a, float(od.joint_travel(self.robot, [a])["J"] or 0.0))
        return (U_PLAN, {"planner": r.status, "reason": r.reason}, 0.0)

    def edges(self, cells, q, o, d):
        k = ("edge", state_key(cells, q, ()), o, d)
        if k not in self.memo or any(x[0] == U_BUD for x in self.memo[k]):
            self.memo[k] = self._edges(cells, q, o, d, k)
        return self.memo[k]

    def _edges(self, cells, q, o, d, k):
        import data
        import ordered as od
        import p14_d_search as ds
        import p14_sweep_index as sx
        import skill as sk
        import p15_lookup as pl
        s = self.scene(cells, q)
        if d in cells.values():
            return [(SI, None, None, 0.0, {"why": "destination occupied"})]
        ln = data.make_query(self.cfg, "c", s, "t0", self.goal_cell, {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]
        self.stats["certificate_checks"] += 1
        cert = [c["kind"] for c in od.certificates_for(self.robot, self.cfg, s, ln, [[o, d]]) if c["checker_accepted"]
                and c["kind"] in ("ORDER_FIRST_ACTION", "STATIC_ENDPOINT_INVALID")]
        if cert:
            return [(SI if cert == ["STATIC_ENDPOINT_INVALID"] else CI, None, None, 0.0, {"certificates": cert})]
        outs, why = [], U_LIB
        if self.idx is not None:
            mode, _, var = self.lookup.partition("+")
            res = sx.lookup_candidates(self.idx, o, cells[o], d, s, self.goal_cell, s.get(o).radius_m, mode, self.modal_only)
            if var == "thr":                       # P15-6 ablation: DEV-fitted minimum combined similarity (rejected for deployment, 0075)
                res["candidates"] = [c for c in res["candidates"] if c["similarity"] >= SIM_THR]
            elif var == "rep":                     # P15-6 ablation: one representative per motion mode (clustering loss)
                res["candidates"] = [c for c in res["candidates"] if c["rank_in_mode"] == 0]
            seen_q = []
            for c in res["candidates"]:
                if len(outs) >= K_MODES:
                    break
                if not self._take("motion_checks"):
                    why = U_BUD
                    break
                self.stats["lookup_candidates_tried"] += 1
                a = self.idx.arrays(c["entry"])
                st, _ = pl._certify(self, s, o, q, a)
                why = U_TR if st == "UNKNOWN_TRANSIT" and why == U_LIB else (why if st != "COLLISION" else "LIBRARY_COLLISION")
                if st != "VERIFIED" or any(np.max(np.abs(np.asarray(a["q"][-1]) - x)) < 1e-9 for x in seen_q):
                    continue
                tr = ds._transit(self, s, o, q, a["q"][0], float(a["gap"][0]))
                seen_q.append(np.asarray(a["q"][-1], float))
                self.stats["lookup_hits"] += 1
                self.stats["rare_mode_used"] += c["mode"] > 0
                outs.append((V, [tr, a], np.asarray(a["q"][-1], float), float(od.joint_travel(self.robot, [tr])["J"] or 0.0) + c["travel_J"],
                             {"via": "lookup", "mode": c["mode"], "support": c["support"], "entry": f"{c['entry']['file']}#{c['entry']['index']}"}))
        if outs:
            return outs + ([(U_BUD, None, None, 0.0, {"lookup": "motion_checks exhausted before all modes were tried"})] if why == U_BUD else [])
        if not self._take("online_moves"):
            return [(U_BUD, None, None, 0.0, {"lookup": why})]
        r = sk.plan_skill(self.robot, s, o, d, self.cfg, self._key(*k), q_start=np.asarray(q, float), q_goal=None)
        if r.status == sk.VERIFIED and r.traj is not None:
            a = r.traj.arrays()
            return [(V, [a], np.asarray(a["q"][-1], float), float(od.joint_travel(self.robot, [a])["J"] or 0.0), {"via": "online", "lookup": why})]
        return [(U_PLAN, None, None, 0.0, {"planner": r.status, "lookup": why})]


def replay(phys, plan) -> bool:
    """Independent rollout: every transit / relocation / target record re-certified in the actual sequential scene."""
    import geometry as ge
    import motion as mo
    cells = {o.id: o.cell for o in phys.s0.objects}
    q, recs, i = np.asarray(phys.s0.robot.q_rad, float), plan["records"], 0
    for st in plan["steps"]:
        s = phys.scene(cells, q)
        parts = recs[i:i + (2 if st["via"] == "lookup" else 1)]
        i += len(parts)
        for a in parts:
            if len(a["q"]) == 2 and a["phase"][0] == "pregrasp" and not a["held_local"].size:
                ck = mo.checkers(phys.robot, s, st["object"], None, None, phys.p["delta"], phys.p["eps"])[("start", False)]
                if not np.allclose(a["q"][0], a["q"][1]) and ge.certify(ck, a["q"][0], a["q"][1], a["gap"][0], a["gap"][1], phys.p["evals"]).status != ge.CLEAR:
                    return False
            elif not all(x.status == ge.CLEAR for x in mo.certify_trajectory(phys.robot, s, st["object"], a, phys.p["delta"], phys.p["eps"], phys.p["evals"])):
                return False
        cells[st["object"]] = st["dest"]
        q = np.asarray(parts[-1]["q"][-1], float)
    tw = recs[-1]
    return all(x.status == ge.CLEAR for x in mo.certify_trajectory(phys.robot, phys.scene(cells, q), "t0", tw, phys.p["delta"], phys.p["eps"], phys.p["evals"]))
