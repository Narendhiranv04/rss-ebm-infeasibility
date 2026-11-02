"""P14-D2: finite joint destination / order search for corrective rearrangement (non-learned search; D only GUIDES expansion order).

State = (object -> cell occupancy, robot configuration q, executed prefix). An edge relocates one movable object (each at most once) to a
destination slot with a VERIFIED motion: a library witness (p14_sweep_index) certified by the exact continuous certifier in the current
hypothetical scene, preceded by a certified joint-linear transit from q to the witness start; or, if no library witness certifies, the
online planner (skill.plan_skill, budgeted). A plan is FOUND only when the final target pick-and-place is witnessed by the online planner in
the final scene. Objective: lexicographic (number of relocations, measured joint travel = ordered.joint_travel units), by iterative
deepening over the move count with travel branch-and-bound inside a depth. Guidance (D / D-pairless / geometry / uninformed / oracle)
changes only the expansion order and recursive promotion, never prunes. Lookup misses are UNKNOWN, never infeasibility. Optimality is
claimed only when a depth was enumerated without budget truncation (finite candidate graph), never globally."""
from __future__ import annotations

import copy
import time

import numpy as np

import config as cf
import p14_sweep_index as sx

BUDGET = {"motion_checks": 60, "online_moves": 6, "target_checks": 10}
K_DEST, N_MAX = 4, 3


class Ctx:
    def __init__(self, lookup="combined", modal_only=False, online=True, budget=None):
        import geometry as ge
        import p6_scenes as p6
        import skill as sk
        self.cfg = cf.load_config("config.toml")
        self.robot, self.p = ge.make_robot(self.cfg), sk.params(self.cfg)
        self.scr = p6.Screens(self.cfg, self.robot)
        self.idx = sx.Index() if lookup else None
        self.lookup, self.modal_only, self.online = lookup, modal_only, online
        self.budget = dict(budget or BUDGET)
        self.memo = {}                                   # (occupancy, q, object, dest) / (occupancy, q) -> verified result (charged once)
        self.stats = {"motion_checks": 0, "online_moves": 0, "target_checks": 0, "lookup_hits": 0, "lookup_misses": 0, "lookup_unknown": 0,
                      "nodes": 0, "modal_rejections_rescued": 0, "D_calls": 0, "truncated": False}

    def left(self, k):
        if self.stats[k] >= self.budget[k]:
            self.stats["truncated"] = True
            return False
        return True


def travel(robot, recs):
    import ordered as od
    return float(sum(od.joint_travel(robot, [r])["J"] or 0.0 for r in recs))


def _transit(cx, scene, oid, qa, qb, gap):
    """Certified joint-linear transit (open gripper, nothing held) from qa to qb in `scene`; None if not certified CLEAR."""
    import geometry as ge
    import motion as mo
    qa, qb = np.asarray(qa, float), np.asarray(qb, float)
    if np.allclose(qa, qb, atol=1e-12):
        return {"q": np.array([qa, qb]), "gap": np.array([gap, gap]), "phase": np.array(["pregrasp"]), "attached": np.array([False]),
                "held_local": np.array([]), "place_xy": np.array([])}
    ck = mo.checkers(cx.robot, scene, oid, None, None, cx.p["delta"], cx.p["eps"])[("start", False)]
    v = ge.certify(ck, qa, qb, gap, gap, cx.p["evals"])
    if v.status != ge.CLEAR:
        return None
    return {"q": np.array([qa, qb]), "gap": np.array([gap, gap]), "phase": np.array(["pregrasp"]), "attached": np.array([False]),
            "held_local": np.array([]), "place_xy": np.array([])}


def _occkey(scene, q):
    return (tuple(sorted((o.id, o.cell) for o in scene.objects)), tuple(np.round(np.asarray(q, float), 9)))


def _key(*parts):
    """Planner seed key as a pure function of the physical query (identical witnesses for every guidance method)."""
    return "p14/" + cf.stable_hash([str(x) for x in parts])[7:23]


def move(cx, scene, goal, oid, dst, q, key=None):
    k = ("move", _occkey(scene, q), oid, dst)
    if k not in cx.memo:
        cx.memo[k] = _move(cx, scene, goal, oid, dst, q, _key(*k))
    return cx.memo[k]


def _move(cx, scene, goal, oid, dst, q, key):
    """Verified relocation of oid to dst from configuration q: library witnesses in mode order (each certified in this scene with a certified
    transit), then the online planner. Returns (records, new q, provenance) or (None, None, reason)."""
    import geometry as ge
    import motion as mo
    import skill as sk
    o = scene.get(oid)
    tried_modes = set()
    if cx.idx is not None:
        res = sx.lookup_candidates(cx.idx, oid, o.cell, dst, scene, goal, o.radius_m, cx.lookup, cx.modal_only)
        if res["status"] != "CANDIDATES":
            cx.stats["lookup_unknown"] += 1
        for c in res["candidates"]:
            if not cx.left("motion_checks"):
                return None, None, "BUDGET"
            cx.stats["motion_checks"] += 1
            a = cx.idx.arrays(c["entry"])
            tr = _transit(cx, scene, oid, q, a["q"][0], float(a["gap"][0]))
            if tr is None:
                continue
            try:
                v = mo.certify_trajectory(cx.robot, scene, oid, a, cx.p["delta"], cx.p["eps"], cx.p["evals"])
            except ValueError:
                continue
            if all(x.status == ge.CLEAR for x in v):
                cx.stats["lookup_hits"] += 1
                if c["mode"] > 0 and 0 in tried_modes:
                    cx.stats["modal_rejections_rescued"] += 1
                return [tr, a], np.asarray(a["q"][-1], float), {"via": "lookup", "mode": c["mode"], "support": c["support"],
                                                               "entry": f"{c['entry']['file']}#{c['entry']['index']}"}
            tried_modes.add(c["mode"])
        cx.stats["lookup_misses"] += 1
    if not cx.online or not cx.left("online_moves"):
        return None, None, "LOOKUP_MISS_NO_FALLBACK" if not cx.online else "BUDGET"
    cx.stats["online_moves"] += 1
    r = sk.plan_skill(cx.robot, scene, oid, dst, cx.cfg, key, q_start=q, q_goal=None)
    if r.status != sk.VERIFIED or r.traj is None:
        return None, None, f"ONLINE_{r.status}"
    a = r.traj.arrays()
    return [a], np.asarray(a["q"][-1], float), {"via": "online"}


def target(cx, scene, goal, q, key=None):
    k = ("target", _occkey(scene, q))
    if k not in cx.memo or cx.memo[k] == "BUDGET":
        cx.memo[k] = _target(cx, scene, goal, q, _key(*k))
    return None if cx.memo[k] == "BUDGET" else cx.memo[k]


def _target(cx, scene, goal, q, key):
    import skill as sk
    if not cx.left("target_checks"):
        return "BUDGET"
        return None
    cx.stats["target_checks"] += 1
    r = sk.plan_skill(cx.robot, scene, "t0", goal, cx.cfg, key, q_start=q, q_goal=None)
    return r.traj.arrays() if r.status == sk.VERIFIED and r.traj is not None else None


# ------------------------------------------------------------------ guidance (expansion order only)
def root_variant(R, robot, xy, tid, q):
    import torch
    import geometry as ge
    R2 = copy.copy(R)
    R2.xy0, R2.tid = xy, tid
    fl = R.flags.clone()
    fl[:, 1] = 0.0
    fl[tid, 1] = 1.0
    R2.flags = fl
    j = torch.as_tensor(np.asarray(ge.fk(robot, q)[0]), dtype=torch.float64)
    qt = torch.as_tensor(np.asarray(q), dtype=torch.float64)
    R2.ctx = torch.cat([torch.sin(qt), torch.cos(qt), R.goal, j.reshape(-1)])
    R2.base = j[0]
    return R2


class Guide:
    """score(target object index, xy, q, pool) -> blocker priority per pool object. 'oracle' uses teacher union masks (original target only)."""

    def __init__(self, kind, R, robot, d_models=None, oracle=None):
        self.kind, self.R, self.robot, self.models, self.oracle = kind, R, robot, d_models or [], oracle

    def score(self, cx, tid, xy, q, pool):
        import d_blockers as db
        if self.kind == "uninformed" or not pool:
            return [0.0] * len(pool)
        R2 = root_variant(self.R, self.robot, xy, tid, q)
        if self.kind == "oracle" and tid == self.R.tid and self.oracle is not None:
            return [float(self.oracle.get(i, 0)) for i in pool]
        if self.kind in ("geometry", "oracle"):
            return [float(x) for x in db.heuristic(R2, pool)]
        cx.stats["D_calls"] += 1
        ps = [db.predict(m, R2, pool) for m in self.models]
        return [float(np.mean([(1 - p["p_feasible"]) * p["marg"][i] for p in ps])) for i in range(len(pool))]


# ------------------------------------------------------------------ search
def search(cx, root, rr, guide, key="p14") -> dict:
    """Iterative deepening over the number of relocations (0..N_MAX), budgeted; returns the best verified plan found, its cost and stats."""
    import torch
    import p6_scenes as p6
    from scene import scene_from_dict
    c0, w0 = time.process_time(), time.time()
    s0, goal = scene_from_dict(root["scene"]), root["goal_cell"]
    R, ids = rr["R"], rr["ids"]
    movers = sorted({o for it in root["items"] if it["kind"] == "ORDERED_TEACHER" for o, _ in it["sequence"]})
    pool = [ids.index(o) for o in movers]
    T = s0.get("t0").cell
    best, found_depth, exhaustive = None, None, False

    def xy_of(scene):
        return torch.as_tensor(np.array([scene.get(o).xy_m for o in ids]), dtype=torch.float64)

    def order(scene, q, moved):
        sc = dict(zip(pool, guide.score(cx, R.tid, xy_of(scene), q, pool)))
        cand = []
        for i in pool:
            o = ids[i]
            if o in moved:
                continue
            ds = p6._dests(cx.scr, scene, o, [T, goal], [], K_DEST)
            if cx.idx is not None:
                ds += [d for (s, d) in cx.idx.by_key if s == scene.get(o).cell and d not in ds and d in cx.scr.valid and
                       d not in {x.cell for x in scene.objects} and p6._cheb(d, T) >= 2 and p6._cheb(d, goal) >= 2]
            for d in ds[:K_DEST + 2]:
                cand.append((-sc[i], o, d))
        return sorted(cand)

    def dfs(scene, q, recs, steps, depth, moved, J):
        nonlocal best
        cx.stats["nodes"] += 1
        if best is not None and J >= best["travel"]:
            return
        if depth == 0:
            tw = target(cx, scene, goal, q)
            if tw is not None:
                tot = J + travel(cx.robot, [tw])
                if best is None or tot < best["travel"]:
                    best = {"moves": len(steps), "travel": tot, "steps": steps, "records": recs + [tw]}
            return
        cands = order(scene, q, moved)
        promoted = []
        while cands:
            _, o, d = cands.pop(0)
            if cx.stats["truncated"]:
                return
            out, q2, how = move(cx, scene, goal, o, d, q)
            if out is None:
                if o not in promoted and guide.kind != "uninformed":       # recursive query: what blocks o? promote those objects
                    promoted.append(o)
                    sc = dict(zip(pool, guide.score(cx, ids.index(o), xy_of(scene), q, pool)))
                    cands.sort(key=lambda c: (-sc.get(ids.index(c[1]), 0.0), c[0], c[1], c[2]))
                continue
            s2 = p6.moved(scene, {o: d})
            dfs(s2, q2, recs + out, steps + [{"object": o, "dest": d, **how}], depth - 1, moved | {o}, J + travel(cx.robot, out))

    for depth in range(N_MAX + 1):
        before = cx.stats["truncated"]
        dfs(s0, np.asarray(s0.robot.q_rad, float), [], [], depth, frozenset(), 0.0)
        if best is not None:
            found_depth, exhaustive = depth, not cx.stats["truncated"] and not before
            break
        if cx.stats["truncated"]:
            break
    return {"found": best is not None, "plan": best, "found_depth": found_depth, "depth_exhaustive": exhaustive,
            "stats": dict(cx.stats), "cpu_s": time.process_time() - c0, "wall_s": time.time() - w0}


def replay(cx, root, plan) -> dict:
    """Independent execution rollout in the simulator model: every recorded relocation (with its transit) and the final target record is
    re-certified segment by segment in the actual sequential scene; success only if all are CLEAR and the target is placed."""
    import geometry as ge
    import motion as mo
    import p6_scenes as p6
    from scene import scene_from_dict
    s = scene_from_dict(root["scene"])
    recs, i = plan["records"], 0
    for st in plan["steps"]:
        if st["via"] == "lookup":
            tr, a = recs[i], recs[i + 1]
            i += 2
            ck = mo.checkers(cx.robot, s, st["object"], None, None, cx.p["delta"], cx.p["eps"])[("start", False)]
            if not np.allclose(tr["q"][0], tr["q"][1]) and ge.certify(ck, tr["q"][0], tr["q"][1], tr["gap"][0], tr["gap"][1], cx.p["evals"]).status != ge.CLEAR:
                return {"success": False, "failed_at": st}
        else:
            a = recs[i]
            i += 1
        if not all(x.status == ge.CLEAR for x in mo.certify_trajectory(cx.robot, s, st["object"], a, cx.p["delta"], cx.p["eps"], cx.p["evals"])):
            return {"success": False, "failed_at": st}
        s = p6.moved(s, {st["object"]: st["dest"]})
    tw = recs[-1]
    ok = all(x.status == ge.CLEAR for x in mo.certify_trajectory(cx.robot, s, "t0", tw, cx.p["delta"], cx.p["eps"], cx.p["evals"]))
    return {"success": bool(ok)}


def d_models(pairless=False):
    """Frozen P13 D models at the validation-selected lambda_B (3 seeds); never retrained here."""
    import torch
    import d_blockers as db
    snap = sx.rd("runs/p13/holdout/PREDICTIONS_FROZEN.json")
    arm = "D-pairless" if pairless else "D"
    lam = snap["baselines"][f"{arm}_lambda_B"]
    out = []
    for s in (0, 1, 2):
        m = db.ModelD(pair=not pairless).double()
        m.load_state_dict(torch.load(f"runs/p13/models/ckpt/P13-{arm}-l{lam}-s{s}.pt", weights_only=True))
        m.eval()
        out.append(m)
    return out


def interface_probe(n_roots=20) -> dict:
    """Integration hypothesis check (DEV roots only): D queried with an OBSTACLE as temporary retrieval target vs a physical probe (online
    planner: pick that object, place it at its first free annulus-safe slot). Agreement of P(feasible) >= 0.5 with VERIFIED / CERTIFIED."""
    import os
    import torch
    import p6_scenes as p6
    import skill as sk
    from scene import scene_from_dict
    os.environ["P10C_OUT"] = "runs/p13/models"
    import p13_pipeline as pp
    import d_blockers as db
    cx = Ctx(lookup=None)
    ms = d_models()
    rs = [r for r in pp._prepared("trainval") if str(r["audit"].get("source", "")).startswith("P13")][:n_roots]
    P = {k: r for d in sorted((pp.OUT / "data").glob("*_b*")) for k, r in pp.rd(d / "plan.json")["roots"].items()}
    rows = []
    for rr in rs:
        root = P[rr["key"]]
        s, ids = scene_from_dict(root["scene"]), rr["ids"]
        movers = sorted({o for it in root["items"] if it["kind"] == "ORDERED_TEACHER" for o, _ in it["sequence"]})
        xy = torch.as_tensor(np.array([s.get(o).xy_m for o in ids]), dtype=torch.float64)
        for o in movers:
            others = [ids.index(x) for x in movers if x != o]
            R2 = root_variant(rr["R"], cx.robot, xy, ids.index(o), np.asarray(s.robot.q_rad, float))
            pf = float(np.mean([db.predict(m, R2, others)["p_feasible"] for m in ms]))
            dd = p6._dests(cx.scr, s, o, [s.get("t0").cell, root["goal_cell"]], [], 1)
            if not dd:
                continue
            r = sk.plan_skill(cx.robot, s, o, dd[0], cx.cfg, f"probe/{rr['key']}/{o}", q_start=np.asarray(s.robot.q_rad, float), q_goal=None)
            rows.append({"root": rr["key"], "object": o, "p_feasible_D": pf, "physical": r.status})
    dec = [x for x in rows if x["physical"] in (sk.VERIFIED, sk.INFEASIBLE)]
    agree = [int((x["p_feasible_D"] >= 0.5) == (x["physical"] == sk.VERIFIED)) for x in dec]
    out = {"probes": len(rows), "decided": len(dec), "agreement": float(np.mean(agree)) if agree else None,
           "physical_counts": {k: sum(x["physical"] == k for x in rows) for k in {x["physical"] for x in rows}},
           "D_feasible_given_verified": float(np.mean([x["p_feasible_D"] >= 0.5 for x in dec if x["physical"] == sk.VERIFIED] or [np.nan])),
           "D_infeasible_given_certified": float(np.mean([x["p_feasible_D"] < 0.5 for x in dec if x["physical"] == sk.INFEASIBLE] or [np.nan])),
           "rows": rows, "note": "DEV only; D was trained with the true target only - this tests transfer, it does not assume it"}
    (sx.OUT / "search").mkdir(parents=True, exist_ok=True)
    cf.write_json(sx.OUT / "search" / "D_INTERFACE_PROBE.json", out)
    return out
