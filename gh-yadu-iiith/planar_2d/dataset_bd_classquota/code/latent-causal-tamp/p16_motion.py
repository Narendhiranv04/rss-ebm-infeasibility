"""P16 deterministic physics provider. Reuses the P15 tri-state provider (p15_search.Physics) with: no CPU-time cap (deterministic work
budgets only), planner wall cap disabled (node budgets decide; a process watchdog outside returns UNKNOWN_BUDGET for the whole root),
certified ALTERNATE transit from the current q to a library witness start (bounded RRT in joint space, nothing held), per-edge
instrumentation with trajectory hashes, and a stricter independent replay (record continuity + every transit waypoint certified)."""
from __future__ import annotations

import copy
import hashlib

import numpy as np

import p15_search as ps

V, CI, SI, U_PLAN, U_BUD, U_LIB, U_TR = ps.V, ps.CI, ps.SI, ps.U_PLAN, ps.U_BUD, ps.U_LIB, ps.U_TR
BUDGET = {"motion_checks": 400, "online_moves": 6, "goal_plans": 16, "transit_plans": 8}
FAST = {"max_nodes": 1500, "max_plan_failures": 2, "transit_nodes": 500}         # declared P16 planner profile (re-validated in P16-1)


def traj_hash(recs) -> str:
    h = hashlib.sha256()
    for a in recs:
        h.update(np.round(np.asarray(a["q"], float), 9).tobytes())
        h.update(np.round(np.asarray(a["gap"], float), 9).tobytes())
    return h.hexdigest()[:16]


def transit_record(path, gap):
    path = np.asarray(path, float)
    return {"q": path, "gap": np.full(len(path), gap), "phase": np.array(["pregrasp"] * (len(path) - 1)), "attached": np.array([False] * (len(path) - 1)),
            "held_local": np.array([]), "place_xy": np.array([])}


class DetPhysics(ps.Physics):
    """transit: 'direct' (P15 semantics: straight joint segment only) | 'planned' (direct, else bounded certified RRT transit)."""

    def __init__(self, root, idx=None, lookup="combined", modal_only=False, budget=None, transit="planned", transit_nodes=800, use_lookup=True,
                 cache=None, profile="fast"):
        super().__init__(root, idx if use_lookup else None, lookup, modal_only, dict(budget or BUDGET), None)
        if profile == "fast":                                            # P16_FAST: deterministic node budgets, cheaper failures
            transit_nodes = min(transit_nodes, FAST["transit_nodes"])
        self.cache, self.vcpu, self.pcpu = cache, 0.0, 0.0
        self.tag = (lookup, modal_only, transit, transit_nodes, use_lookup, profile)
        self.cfg = copy.deepcopy(self.cfg)
        self.cfg["planner2d"]["max_wall_seconds"] = 1e9                 # node budgets decide; no wall-clock adjudication
        if profile == "fast":
            self.cfg["planner2d"]["max_nodes"], self.cfg["skill2d"]["max_plan_failures"] = FAST["max_nodes"], FAST["max_plan_failures"]
        self.profile = profile
        self.transit, self.tcfg = transit, dict(self.cfg["planner2d"], max_nodes=transit_nodes)
        self.stats |= {"transit_direct": 0, "transit_planned": 0, "transit_unknown": 0, "library_collision": 0, "wall_clock_cut": 0, "cache_hits": 0}
        self.edge_log = []

    def _cached(self, k, fn):
        """Shared deterministic cache across arms on one root (exact virtual accounting): a cached result is reused only if this arm's
        remaining budget covers the recorded consumption (then the computation would have run identically); the arm is charged the
        recorded counts, statistics and CPU. Budget-censored results are never cached."""
        import time
        if k in self.memo and not (self.memo[k][0] == U_BUD if k[0] == "goal" else any(x[0] == U_BUD for x in self.memo[k])):
            return self.memo[k]
        sk = self.tag + k
        if self.cache is not None and sk in self.cache and not self.truncated():
            res, du, ds, cpu = self.cache[sk]
            if all(self.used[x] + n <= self.budget[x] for x, n in du.items()):
                for x, n in du.items():
                    self.used[x] += n
                for x, n in ds.items():
                    self.stats[x] += n
                self.stats["cache_hits"] += 1
                self.vcpu += cpu
                self.memo[k] = res
                return res
        u0, s0, c0 = dict(self.used), dict(self.stats), time.process_time()
        res = fn()
        cpu = time.process_time() - c0
        first = res[0] if k[0] == "goal" else (res[0][0] if res else None)
        self.edge_log.append({"kind": k[0], "obj": k[2] if k[0] == "edge" else None, "dest": k[3] if k[0] == "edge" else None,
                              "status": first if isinstance(first, str) else None, "cpu": round(cpu, 3),
                              "used": {x: self.used[x] - u0[x] for x in self.used if self.used[x] != u0[x]}})
        self.vcpu, self.pcpu = self.vcpu + cpu, self.pcpu + cpu
        self.memo[k] = res
        cens = res[0] == U_BUD if k[0] == "goal" else any(x[0] == U_BUD for x in res)
        if self.cache is not None and not cens:
            self.cache[sk] = (res, {x: self.used[x] - u0[x] for x in self.used}, {x: self.stats[x] - s0.get(x, 0) for x in self.stats if x != "cache_hits"}, cpu)
        return res

    def edges(self, cells, q, o, d):
        k = ("edge", ps.state_key(cells, q, ()), o, d)
        return self._cached(k, lambda: self._edges(cells, q, o, d, k))

    def goal(self, cells, q):
        k = ("goal", ps.state_key(cells, q, ()))
        return self._cached(k, lambda: self._goal(cells, q, k))

    def _take(self, k):
        if k not in self.budget:
            return True
        return super()._take(k)

    def _transit(self, s, o, q, a, k):
        """(record | None, how). Straight segment first; then (planned mode) a bounded RRT in the current scene with nothing held."""
        import geometry as ge
        import motion as mo
        import planner as pl
        q, qb, gap = np.asarray(q, float), np.asarray(a["q"][0], float), float(a["gap"][0])
        if np.allclose(q, qb, atol=1e-12):
            return transit_record([q, qb], gap), "same"
        ck = mo.checkers(self.robot, s, o, None, None, self.p["delta"], self.p["eps"])[("start", False)]
        if ge.certify(ck, q, qb, gap, gap, self.p["evals"]).status == ge.CLEAR:
            self.stats["transit_direct"] += 1
            return transit_record([q, qb], gap), "direct"
        if self.transit != "planned" or not self._take("transit_plans"):
            self.stats["transit_unknown"] += 1
            return None, "unknown"
        r = pl.plan(ck, q, qb, gap, self.tcfg, self.p["evals"], self._key("transit", *k, tuple(np.round(qb, 9))), rrt=True)
        self.stats["wall_clock_cut"] += bool(r.wall_clock_cut)
        if r.status == ge.CLEAR:
            self.stats["transit_planned"] += 1
            return transit_record(r.path, gap), "planned"
        self.stats["transit_unknown"] += 1
        return None, "unknown"

    def _edges(self, cells, q, o, d, k):
        import geometry as ge
        import motion as mo
        import ordered as od
        import p14_sweep_index as sx
        import skill as sk
        s = self.scene(cells, q)
        if d in cells.values():
            return [(SI, None, None, 0.0, {"why": "destination occupied"})]
        import data
        ln = data.make_query(self.cfg, "c", s, "t0", self.goal_cell, {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]
        self.stats["certificate_checks"] += 1
        cert = [c["kind"] for c in od.certificates_for(self.robot, self.cfg, s, ln, [[o, d]]) if c["checker_accepted"]
                and c["kind"] in ("ORDER_FIRST_ACTION", "STATIC_ENDPOINT_INVALID")]
        if cert:
            return [(SI if cert == ["STATIC_ENDPOINT_INVALID"] else CI, None, None, 0.0, {"certificates": cert})]
        outs, why, seen_q = [], U_LIB, []
        if self.idx is not None:
            mode, _, var = self.lookup.partition("+")
            res = sx.lookup_candidates(self.idx, o, cells[o], d, s, self.goal_cell, s.get(o).radius_m, mode, self.modal_only)
            cands = res["candidates"] if var != "rep" else [c for c in res["candidates"] if c["rank_in_mode"] == 0]
            for c in cands:
                if len(outs) >= ps.K_MODES:
                    break
                a = self.idx.arrays(c["entry"])
                if any(np.max(np.abs(np.asarray(a["q"][-1]) - x)) < 1e-9 for x in seen_q):
                    continue
                if not self._take("motion_checks"):
                    why = U_BUD
                    break
                self.stats["lookup_candidates_tried"] += 1
                try:
                    v = mo.certify_trajectory(self.robot, s, o, a, self.p["delta"], self.p["eps"], self.p["evals"])
                except ValueError:
                    continue
                if not all(x.status == ge.CLEAR for x in v):
                    self.stats["library_collision"] += any(x.status == ge.COLLISION for x in v)
                    why = "LIBRARY_COLLISION" if why == U_LIB else why
                    continue
                tr, how = self._transit(s, o, q, a, k)
                if tr is None:
                    why = U_TR if why in (U_LIB, "LIBRARY_COLLISION") else why
                    continue
                seen_q.append(np.asarray(a["q"][-1], float))
                self.stats["lookup_hits"] += 1
                self.stats["rare_mode_used"] += c["mode"] > 0
                J = float(od.joint_travel(self.robot, [tr])["J"] or 0.0) + c["travel_J"]
                outs.append((V, [tr, a], np.asarray(a["q"][-1], float), J,
                             {"via": "lookup", "mode": c["mode"], "support": c["support"], "transit": how, "traj": traj_hash([tr, a]),
                              "entry": f"{c['entry']['file']}#{c['entry']['index']}"}))
        if outs:
            return outs + ([(U_BUD, None, None, 0.0, {"lookup": "motion_checks exhausted before all modes were tried"})] if why == U_BUD else [])
        if not self._take("online_moves"):
            return [(U_BUD, None, None, 0.0, {"lookup": why})]
        r = sk.plan_skill(self.robot, s, o, d, self.cfg, self._key(*k), q_start=np.asarray(q, float), q_goal=None)
        self.stats["wall_clock_cut"] += any(p.wall_clock_cut for _, p in r.plans)
        if r.status == sk.VERIFIED and r.traj is not None:
            a = r.traj.arrays()
            return [(V, [a], np.asarray(a["q"][-1], float), float(od.joint_travel(self.robot, [a])["J"] or 0.0),
                     {"via": "online", "lookup": why, "traj": traj_hash([a])})]
        return [(U_PLAN, None, None, 0.0, {"planner": r.status, "lookup": why})]


def replay(phys, plan) -> dict:
    """Independent sequential re-certification: each record must start where the previous ended (continuity), every transit waypoint
    segment is certified with the free-robot checker, every relocation / target record by motion.certify_trajectory in the actual scene."""
    import geometry as ge
    import motion as mo
    cells = {o.id: o.cell for o in phys.s0.objects}
    q, recs, i = np.asarray(phys.s0.robot.q_rad, float), plan["records"], 0
    for j, st in enumerate(plan["steps"]):
        s = phys.scene(cells, q)
        parts = recs[i:i + (2 if st["via"] == "lookup" else 1)]
        i += len(parts)
        for a in parts:
            if np.max(np.abs(np.asarray(a["q"][0], float) - q)) > 1e-6:
                return {"ok": False, "step": j, "why": "discontinuous record"}
            if not a["held_local"].size and all(p == "pregrasp" for p in a["phase"]) and not any(a["attached"]):
                ck = mo.checkers(phys.robot, s, st["object"], None, None, phys.p["delta"], phys.p["eps"])[("start", False)]
                for qa, qb in zip(a["q"][:-1], a["q"][1:]):
                    if not np.allclose(qa, qb) and ge.certify(ck, qa, qb, a["gap"][0], a["gap"][0], phys.p["evals"]).status != ge.CLEAR:
                        return {"ok": False, "step": j, "why": "transit segment"}
            elif not all(x.status == ge.CLEAR for x in mo.certify_trajectory(phys.robot, s, st["object"], a, phys.p["delta"], phys.p["eps"], phys.p["evals"])):
                return {"ok": False, "step": j, "why": "relocation record"}
            q = np.asarray(a["q"][-1], float)
        cells[st["object"]] = st["dest"]
    tw = recs[-1]
    if np.max(np.abs(np.asarray(tw["q"][0], float) - q)) > 1e-6:
        return {"ok": False, "step": "target", "why": "discontinuous record"}
    ok = all(x.status == ge.CLEAR for x in mo.certify_trajectory(phys.robot, phys.scene(cells, q), "t0", tw, phys.p["delta"], phys.p["eps"], phys.p["evals"]))
    return {"ok": ok, "step": None if ok else "target", "why": None if ok else "target record", "traj": traj_hash(recs)}
