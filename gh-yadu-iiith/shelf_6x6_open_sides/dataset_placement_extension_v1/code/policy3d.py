"""Query-level placement policy PI_PLACE_3D: predeclared mode ranking, lazy screening, staged schedules, labels, mode accounting,
and exact virtual replay of smaller / adaptive budgets from a full reference record.

Scheduling unit = one grasp POSE (azimuth, grip height) that is VALID (static-clear IK branch, no other object within 5 mm with the
fingers pre-shaped, carry footprint clear at start and goal) AND physically held (placement3d.hold).  Its <= 4 IK branches are tried
inside the unit (plan_pick_place's branch loop) and never counted as extra modes.  A stage (M, B) tests the first M units in rank
order with B RRT outer iterations for the carry route; units unresolved at a smaller B are re-planned at the larger B.
"""
import hashlib, time

import numpy as np

import placement3d as P

POLICIES = {"adaptive_v1": [(4, 500), (8, 1500), (16, 5000)], "reference_v1": [(32, 5000)]}
FIXED = {f"fixed_{m}x{b}": [(m, b)] for m in (4, 8, 16, 32) for b in (500, 1500, 5000)}
FINAL = ("PLAN_FAILED", "EXEC_FAILED", "VERIFIED")


class Interrupted(Exception):
    """Emergency CPU guard / administrative stop: the query is UNLABELED (never a 0)."""


def ranked(ps):
    """shelf_scene.grasp_poses order is (grip height, azimuth nearest the robot).  Predeclared diverse rank: k-th nearest azimuth with
    the grip height alternating (mid on even k), then every azimuth at the other height."""
    n = len(ps) // 2
    return [ps[k] if k % 2 == 0 else ps[n + k] for k in range(n)] + [ps[n + k] if k % 2 == 0 else ps[k] for k in range(n)]


def static_reason(cells, obj, dest, rows=6, cols=6):
    if obj not in cells:
        return "UNKNOWN_OBJECT"
    if not (0 <= dest[0] < rows and 0 <= dest[1] < cols):
        return "OUT_OF_BOUNDS"
    if tuple(cells[obj]) == tuple(dest):
        return "SAME_CELL"
    if any(tuple(v) == tuple(dest) for k, v in cells.items() if k != obj):
        return "OCCUPIED"
    return None


def _d(a, b):
    return {k: a[k] - b[k] for k in a if k in b and k != "cpu"}


def _t():
    return time.process_time()


def evaluate(sc, obj, dest, stages, early_stop=True):
    """-> (record, witness arrays or None).  Deterministic given the scene, query and stages (CPU fields excepted)."""
    e, t_all = sc.e, _t()
    P.reset_work()
    sc.restore()
    if any(e.cell_of(o["name"]) is None for o in e.objects):
        return {"label": None, "status": "UNLABELED_SCENE_NOT_SETTLED"}, None
    ps, modes, valid, nxt, wit, first = ranked(P.poses(sc, obj)), [], [], [0], None, None

    def screen_next():
        g, t, w0 = ps[nxt[0]], _t(), dict(P.W)
        st, br, why = P.screen(sc, obj, dest, g)
        m = {"rank": nxt[0], "grasp_id": g["id"], "az": g["azimuth_deg"], "level": g["grasp_level"], "screen": st, "reason": why,
             "n_branches": len(br), "tests": [], "state": st, "cpu_screen": _t() - t, "work_screen": _d(P.W, w0), "_g": g, "_br": br}
        if st == "VALID":
            t = _t(); ok, m["hold"], _ = P.hold(sc, obj, g, br[0]); m["cpu_hold"] = _t() - t
            if ok:
                m["state"] = "UNTESTED"; valid.append(len(modes))
            else:
                m["state"] = m["screen"] = "HELD_INVALID_PHYSICAL"
        modes.append(m); nxt[0] += 1

    for M, B in stages:
        while len(valid) < M and nxt[0] < len(ps):
            screen_next()
        for vi in valid[:M]:
            m = modes[vi]
            if m["state"] in FINAL or (m["tests"] and m["tests"][-1]["iters"] >= B):
                continue
            test, w = run_mode(sc, obj, dest, m, B)
            m["tests"].append(test)
            if w is not None and wit is None:
                wit, first = w, m["rank"]
            if early_stop and wit is not None:
                break
        if early_stop and wit is not None:
            break
    rec = {"label": None, "status": None, "verified_rank": first}
    if wit is not None:
        rec.update(label=1, status="VERIFIED")
    elif valid:
        st = [modes[v]["state"] for v in valid]
        kind = "BOUNDED_SEARCH" if "UNRESOLVED" in st else "EXECUTION" if "EXEC_FAILED" in st else "DETERMINISTIC_PLAN"
        rec.update(label=0, status="NEG_UNDER_POLICY", negative_kind=kind)
    else:
        while nxt[0] < len(ps):
            screen_next()
        for m in modes:                                        # a geometry-certified 0 needs one physically held pose
            if m["screen"] == "GEOMETRY_INVALID":
                t = _t(); ok, m["hold"], _ = P.hold(sc, obj, m["_g"], m["_br"][0]); m["cpu_hold"] = _t() - t
                if ok:
                    rec.update(label=0, status="NEG_CERTIFIED_GEOMETRY", negative_kind="CERTIFIED_GEOMETRY", certified_by=m["rank"]); break
                m["geometry_hold_failed"] = True
        else:
            rec.update(status="UNLABELED_NO_VALID_POST_GRASP_STATE")
    acc = {"nominal": len(ps), "screened": len(modes), "unscreened": len(ps) - len(modes), "valid": len(valid)}
    for s in ("IK_INVALID", "HELD_INVALID", "HELD_INVALID_PHYSICAL", "GEOMETRY_INVALID"):
        acc[s.lower()] = sum(m["screen"] == s for m in modes)
    for s in ("UNTESTED", "UNRESOLVED") + FINAL:
        acc[s.lower()] = sum(m["state"] == s for m in modes)
    acc["motion_tested"] = sum(bool(m["tests"]) for m in modes)
    acc["attempted_configurations"] = sum(len(m["tests"]) for m in modes)
    arrays = None
    if wit is not None:
        t = _t(); arrays = P.pack(*wit); rec["cpu_pack"] = _t() - t
        rec["witness_id"] = wid(arrays)
    cpu = {k: sum(m.get(f"cpu_{k}", 0.0) for m in modes) for k in ("screen", "hold")}
    cpu.update({k: sum(t.get(f"cpu_{k}", 0.0) for m in modes for t in m["tests"]) for k in ("plan", "exec", "verify")})
    rec.update(accounting=acc, modes=[{k: v for k, v in m.items() if not k.startswith("_")} for m in modes], work=dict(P.W),
               cpu_parts=cpu, cpu_total=_t() - t_all)
    return rec, arrays


def run_mode(sc, obj, dest, m, B):
    g, br = m["_g"], m["_br"]
    w0, t = dict(P.W), _t()
    pl = P.plan(sc, obj, dest, g, br, B)
    r = pl["route"]
    test = {"iters": B, "plan_ok": pl["ok"], "phase": pl.get("phase"), "reason": pl.get("reason"), "budget_limited": pl.get("budget_limited", False),
            "branch": pl.get("branch"), "route_iters": r.get("iters"), "route_found": r.get("found"), "route_direct": r.get("direct"),
            "work_plan": _d(P.W, w0), "cpu_plan": _t() - t}
    if r.get("start"):
        test["work_pre_route"] = _d(r["start"], w0)            # AMENDMENT_002: lift chains before the route (needed for exact
        test["cpu_pre_route"] = r["start"]["cpu"] - t          # virtual accounting of budget-truncated units)
        test["route_work"] = _d(r["end"], r["start"])
        test["route_milestones"] = {str(it): _d(ms, r["start"]) for it, ms in r["milestones"].items()}
        test["cpu_route"] = {"total": r["end"]["cpu"] - r["start"]["cpu"], **{str(it): ms["cpu"] - r["start"]["cpu"] for it, ms in r["milestones"].items()}}
    if not pl["ok"]:
        m["state"] = "UNRESOLVED" if test["budget_limited"] else "PLAN_FAILED"
        return test, None
    w1, t = dict(P.W), _t()
    ok, hev, before = P.hold(sc, obj, g, pl["q_g"])
    if not ok:
        test.update(exec={"first_failed": "held_state_lost", "hold": hev}, cpu_exec=_t() - t); m["state"] = "EXEC_FAILED"
        return test, None
    rec, ev = P.execute(sc, obj, pl, before); test["cpu_exec"] = _t() - t
    t = _t(); v, vi = P.verify(sc, obj, dest, rec, ev); test["cpu_verify"] = _t() - t
    test.update(exec={**ev, **vi, "end_cell": list(vi["end_cell"]) if vi["end_cell"] else None, "hold": hev}, work_exec=_d(P.W, w1))
    m["state"] = "VERIFIED" if v else "EXEC_FAILED"
    return test, ((pl, g, hev, rec) if v else None)


def wid(arrays):
    """Content address over the arrays (bd_core.Store.witness scheme: sorted keys, dtype, shape, bytes; 24 hex)."""
    h = hashlib.sha256()
    for k in sorted(arrays):
        v = np.ascontiguousarray(np.asarray(arrays[k]))
        h.update(f"{k}|{v.dtype.str}|{v.shape}".encode()); h.update(v.tobytes())
    return h.hexdigest()[:24]


def canonical(x, drop=("work_screen",)):
    """Scientific fingerprint: no CPU / wall fields; screening work depends on the shared IK cache's history, so it is excluded."""
    if isinstance(x, dict):
        return {k: canonical(v, drop) for k, v in x.items() if not k.startswith(("cpu", "wall", "t_")) and k not in drop and k != "worker"}
    if isinstance(x, list):
        return [canonical(v, drop) for v in x]
    return round(x, 9) if isinstance(x, float) else x


def virtual(rec, stages):
    """Outcome, unit states and counted plan/exec work of a smaller or adaptive policy, replayed from a reference record in which every
    valid unit was tested once at B_ref = 5000 (no early stop).  Same seed => a route found after r iterations is found under B iff
    r <= B; otherwise the unit is UNRESOLVED at B with the work of the first B iterations (milestone snapshot)."""
    valid = [m for m in rec["modes"] if m["screen"] == "VALID"]
    state, work, cpu, out, need = {}, {}, {"screen": 0.0, "hold": 0.0, "plan": 0.0, "exec": 0.0, "verify": 0.0}, None, 0
    add = lambda d: [work.__setitem__(k, work.get(k, 0) + v) for k, v in d.items()]  # noqa: E731
    for M, B in stages:
        need = max(need, M)
        for m in valid[:M]:
            r, s = m["rank"], state.get(m["rank"])
            if s in FINAL or (s and int(s.split("@")[1]) >= B):
                continue
            t = m["tests"][-1]
            cut = "route_work" in t and (not t["route_found"] or t["route_iters"] > B)
            if cut:                                            # truncated at B: pre-route lift chains + the first B route iterations only
                ms = t["route_milestones"].get(str(B), t["route_work"])
                add({k: t["work_pre_route"].get(k, 0) + ms.get(k, 0) for k in t["work_plan"]})
                cpu["plan"] += t["cpu_pre_route"] + t["cpu_route"].get(str(B), t["cpu_route"]["total"])
                state[r] = f"UNRESOLVED@{B}"
                continue
            add(t["work_plan"]); add(t.get("work_exec", {}))
            for k in ("plan", "exec", "verify"):
                cpu[k] += t.get(f"cpu_{k}", 0.0)
            state[r] = m["state"] if m["state"] != "UNRESOLVED" else f"UNRESOLVED@{B}"
            if state[r] == "VERIFIED":
                out = r
                break
        if out is not None:
            break
    k_valid = min(need, len(valid))
    last = (out if out is not None else (valid[k_valid - 1]["rank"] if len(valid) >= need else rec["modes"][-1]["rank"])) if valid else rec["modes"][-1]["rank"]
    for m in rec["modes"]:
        if m["rank"] <= last:
            cpu["screen"] += m.get("cpu_screen", 0.0); cpu["hold"] += m.get("cpu_hold", 0.0)
    label = 1 if out is not None else 0 if valid else rec["label"]
    return {"label": label, "verified_rank": out, "states": {str(k): v for k, v in sorted(state.items())}, "work": work, "cpu": cpu,
            "screened_through_rank": last}
