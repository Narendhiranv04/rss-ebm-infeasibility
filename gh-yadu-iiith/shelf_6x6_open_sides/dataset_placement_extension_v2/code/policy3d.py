"""Query-level placement policy PI_PLACE_3D, V2 (branch-aware): predeclared mode ranking, lazy screening, staged schedules, labels, mode
AND branch accounting, witness retention, and virtual replay of smaller / adaptive budgets from a full reference record.

Unit = one ranked grasp POSE (azimuth, grip height) that is VALID (>= 1 eligible IK branch: structure / ceiling clear and no other object
within 5 mm with the fingers pre-shaped; carry footprint clear at start and destination) and has >= 1 PHYSICALLY HELD eligible branch.
Physical hold certificates are tested lazily in candidate order (screening stops at the first held branch; the others are certified when
a unit test reaches them) and cached (a hold is a deterministic function of the restored scene, pose and branch).  A unit test at budget B
tries every eligible branch in candidate order: hold certificate -> plan_branch -> re-instantiated hold -> MuJoCo execution -> verify,
until one verifies (early-stop policies) or every eligible branch is exhausted (reference).  Branches are never extra units.
V1 difference: V1 certified the hold of branch 0 only and executed only the branch that planned first.
"""
import hashlib, json, time

import numpy as np

import placement3d as P

POLICIES = {"adaptive_v1": [(4, 500), (8, 1500), (16, 5000)], "reference_v2": [(32, 5000)]}
FIXED = {f"fixed_{m}x{b}": [(m, b)] for m in (4, 8, 16, 32) for b in (500, 1500, 5000)}
REFERENCE = {"reference_v2"}
FINAL = ("PLAN_FAILED", "EXEC_FAILED", "VERIFIED")
NOT_UNIT = ("IK_INVALID", "HELD_INVALID", "HELD_INVALID_PHYSICAL", "GEOMETRY_INVALID")


class Interrupted(Exception):
    """Emergency CPU guard / administrative / safety stop: the query is UNLABELED (never a 0)."""


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


def _same_hold(a, b):
    return json.dumps(a, sort_keys=True, default=float) == json.dumps(b, sort_keys=True, default=float)


def unit_state(states):
    return ("VERIFIED" if "VERIFIED" in states else "UNRESOLVED" if "UNRESOLVED" in states else "EXEC_FAILED" if "EXEC_FAILED" in states
            else "PLAN_FAILED" if "PLAN_FAILED" in states else "HOLD_FAILED_ALL")


def evaluate(sc, obj, dest, stages, early_stop=True):
    """-> (record, [(witness_id, arrays, provenance)]).  Deterministic given the scene, query and stages (CPU fields excepted)."""
    e, t_all = sc.e, _t()
    P.reset_work()
    sc.restore()
    if any(e.cell_of(o["name"]) is None for o in e.objects):
        return {"label": None, "status": "UNLABELED_SCENE_NOT_SETTLED"}, []
    ps, modes, valid, nxt, wits = ranked(P.poses(sc, obj)), [], [], [0], []

    def hold_branch(m, b, at):
        br = m["branches"][b]
        if br["hold"] is None:
            t, w0 = _t(), dict(P.W)
            ok, ev, _ = P.hold(sc, obj, m["_g"], m["_q"][b])
            br["hold"] = {"ok": bool(ok), "ev": ev, "at": at, "work": _d(P.W, w0), "cpu": _t() - t}
        return br["hold"]["ok"]

    def screen_next():
        g, t, w0 = ps[nxt[0]], _t(), dict(P.W)
        st, cands, elig, why, hits = P.screen(sc, obj, dest, g)
        m = {"rank": nxt[0], "grasp_id": g["id"], "az": g["azimuth_deg"], "level": g["grasp_level"], "screen": st, "reason": why,
             "n_candidates": len(cands), "n_eligible": len(elig), "tests": [], "state": st, "cpu_screen": _t() - t, "work_screen": _d(P.W, w0),
             "branches": [{"b": k, "eligible": k in elig, "object_hits": hits[k], "hold": None, "results": []} for k in range(len(cands))],
             "_g": g, "_q": cands}
        if st == "VALID":
            for b in elig:
                if hold_branch(m, b, "screen"):
                    break
            if any(br["hold"] and br["hold"]["ok"] for br in m["branches"]):
                m["state"] = "UNTESTED"; valid.append(len(modes))
            else:
                m["state"] = m["screen"] = "HELD_INVALID_PHYSICAL"        # EVERY eligible branch failed its physical hold
        modes.append(m); nxt[0] += 1

    for M, B in stages:
        while len(valid) < M and nxt[0] < len(ps):
            screen_next()
        for vi in valid[:M]:
            m = modes[vi]
            if m["state"] in FINAL or (m["tests"] and m["tests"][-1]["iters"] >= B):
                continue
            run_unit(sc, obj, dest, m, B, early_stop, hold_branch, wits)
            if early_stop and wits:
                break
        if early_stop and wits:
            break
    rec = {"label": None, "status": None, "verified_rank": wits[0][2]["rank"] if wits else None,
           "verified_branch": wits[0][2]["branch"] if wits else None}
    if wits:
        rec.update(label=1, status="VERIFIED")
    elif valid:
        st = [modes[v]["state"] for v in valid if modes[v]["tests"]]
        kind = "BOUNDED_SEARCH" if "UNRESOLVED" in st else "EXECUTION" if "EXEC_FAILED" in st else "DETERMINISTIC_PLAN"
        rec.update(label=0, status="NEG_UNDER_POLICY", negative_kind=kind,
                   scope={"stages": stages, "units_tested": len(st), "units_valid_screened": len(valid), "poses_unscreened": len(ps) - len(modes),
                          "final_iters": stages[-1][1]})
    else:
        while nxt[0] < len(ps):
            screen_next()
        cert = None
        for m in modes:                                        # the conditional event needs a physically held state somewhere
            if m["screen"] != "GEOMETRY_INVALID":
                continue
            for br in m["branches"]:
                if br["eligible"] and hold_branch(m, br["b"], "certificate"):
                    cert = (m["rank"], br["b"]); break
            if cert:
                break
        cnt = {s: sum(m["screen"] == s for m in modes) for s in NOT_UNIT + ("VALID",)}
        if cert:
            rec.update(label=0, status="NEG_CERTIFIED_GEOMETRY", negative_kind="CERTIFIED_GEOMETRY", certified_by=list(cert),
                       certificate={"scope": "32 SHELF66V3 level grasp poses x the source's <= 4 fixed-seed IK candidates x the source 2-D hand-"
                                             "footprint carry model (not a global impossibility proof)",
                                    "poses_screened": len(modes), "screen_counts": cnt,
                                    "held_invalid_physical_all_branches_tested": all(all(br["hold"] is not None for br in m["branches"] if br["eligible"])
                                                                                     for m in modes if m["screen"] == "HELD_INVALID_PHYSICAL"),
                                    "geometry_reasons_footprint_only": all("would overlap a neighbour" in (m["reason"] or "")
                                                                           for m in modes if m["screen"] == "GEOMETRY_INVALID")})
        else:
            rec.update(status="UNLABELED_NO_VALID_POST_GRASP_STATE", screen_counts=cnt)
    rec["accounting"] = accounting(modes, len(ps), len(valid), wits)
    arrays_out = []
    for w, arrays, prov in wits:
        arrays_out.append((w, arrays, prov))
    rec["witness_id"] = wits[0][0] if wits else None
    rec["witnesses"] = [{"witness_id": w, **prov} for w, _, prov in wits]
    cpu = {"screen": sum(m.get("cpu_screen", 0.0) for m in modes),
           "hold": sum(br["hold"]["cpu"] for m in modes for br in m["branches"] if br["hold"]),
           "plan": sum(r.get("cpu_plan", 0.0) for m in modes for br in m["branches"] for r in br["results"]),
           "exec": sum(r.get("cpu_exec", 0.0) for m in modes for br in m["branches"] for r in br["results"]),
           "verify": sum(r.get("cpu_verify", 0.0) for m in modes for br in m["branches"] for r in br["results"])}
    rec.update(modes=[{k: v for k, v in m.items() if not k.startswith("_")} for m in modes], work=dict(P.W), cpu_parts=cpu, cpu_total=_t() - t_all)
    return rec, arrays_out


def run_unit(sc, obj, dest, m, B, early_stop, hold_branch, wits):
    """One unit test at budget B over the unit's eligible branches (candidate order)."""
    rt, test, w_u, t_u, states, seq = P.route_state(), {"iters": B}, dict(P.W), _t(), [], []
    for br in m["branches"]:
        if not br["eligible"]:
            continue
        b, res = br["b"], {"iters": B}
        tested_now = br["hold"] is None
        held = hold_branch(m, b, "test")
        res["hold_tested_here"] = tested_now
        if not held:
            res["state"] = "HOLD_FAILED"
        else:
            w0, t = dict(P.W), _t()
            pl = P.plan_branch(sc, obj, dest, m["_g"], m["_q"][b], B, rt)
            res.update(plan_ok=pl["ok"], phase=pl.get("phase"), reason=pl.get("reason"), route_here=pl["route_here"], work_plan=_d(P.W, w0),
                       cpu_plan=_t() - t, work_pre_route=pl.get("work_pre_route"), cpu_pre_route=pl.get("cpu_pre_route"))
            if pl["route_here"]:
                test["route"], test["route_branch"] = rt["info"], b
            if not pl["ok"]:
                res["state"] = "UNRESOLVED" if pl.get("route_error") and rt["info"]["budget_limited"] else "PLAN_FAILED"
                res["route_error"] = bool(pl.get("route_error"))
            else:
                w1, t = dict(P.W), _t()
                ok, hev, before = P.hold(sc, obj, m["_g"], m["_q"][b])          # instantiate the certified held state
                same = bool(ok) and _same_hold(hev, br["hold"]["ev"])
                if not same:
                    res.update(state="EXEC_FAILED", exec={"first_failed": "held_state_not_reproduced", "hold": hev}, work_exec=_d(P.W, w1),
                               cpu_exec=_t() - t)
                else:
                    rec, ev = P.execute(sc, obj, pl, before); res["cpu_exec"] = _t() - t
                    t = _t(); v, vi = P.verify(sc, obj, dest, rec, ev); res["cpu_verify"] = _t() - t
                    res.update(exec={**ev, **vi, "end_cell": list(vi["end_cell"]) if vi["end_cell"] else None}, work_exec=_d(P.W, w1),
                               state="VERIFIED" if v else "EXEC_FAILED")
                    if v:
                        arrays = P.pack(pl, m["_g"], hev, rec, b, len(m["_q"]))
                        res["witness_id"] = wid(arrays)
                        if res["witness_id"] not in {x[0] for x in wits}:
                            wits.append((res["witness_id"], arrays, {"rank": m["rank"], "grasp_id": m["grasp_id"], "branch": b,
                                                                       "az": m["az"], "level": m["level"]}))
        br["results"].append(res); states.append(res["state"]); seq.append(b)
        if res.get("route_error") or (early_stop and res["state"] == "VERIFIED"):
            break                                              # route-level failure is branch-independent; or label found
    test.update(state=unit_state(states), branch_seq=seq, branch_states=states, work=_d(P.W, w_u), cpu=_t() - t_u)
    m["tests"].append(test)
    m["state"] = test["state"]
    return test


def accounting(modes, nominal, n_valid, wits):
    acc = {"nominal": nominal, "screened": len(modes), "unscreened": nominal - len(modes), "valid": n_valid}
    for s in NOT_UNIT:
        acc[s.lower()] = sum(m["screen"] == s for m in modes)
    acc["valid_screen"] = sum(m["screen"] == "VALID" for m in modes)
    for s in ("UNTESTED", "UNRESOLVED") + FINAL:
        acc[s.lower()] = sum(m["state"] == s for m in modes)
    acc["motion_tested"] = sum(bool(m["tests"]) for m in modes)
    acc["attempted_unit_tests"] = sum(len(m["tests"]) for m in modes)
    brs = [br for m in modes for br in m["branches"]]
    res = [r for br in brs for r in br["results"]]
    acc.update(branch_candidates=len(brs), branch_eligible=sum(br["eligible"] for br in brs),
               branch_hold_tested=sum(br["hold"] is not None for br in brs), branch_held=sum(bool(br["hold"] and br["hold"]["ok"]) for br in brs),
               branch_hold_failed=sum(bool(br["hold"] and not br["hold"]["ok"]) for br in brs),
               branch_attempts=len(res), branch_plan_attempts=sum("plan_ok" in r for r in res),
               branch_plan_failed=sum(r["state"] == "PLAN_FAILED" for r in res), branch_unresolved=sum(r["state"] == "UNRESOLVED" for r in res),
               branch_exec_attempts=sum("exec" in r for r in res), branch_exec_failed=sum(r["state"] == "EXEC_FAILED" for r in res),
               branch_verified=sum(r["state"] == "VERIFIED" for r in res), verified_witnesses=len(wits))
    first_el = {m["rank"]: next((br["b"] for br in m["branches"] if br["eligible"]), None) for m in modes}
    rescue = {"HOLD_FAILED": 0, "EXEC_FAILED": 0, "PLAN_FAILED": 0}
    for m in modes:                                            # unit verified on a later branch, by the FIRST earlier branch failure type
        for T in m["tests"]:
            st = T["branch_states"]
            if "VERIFIED" in st and st.index("VERIFIED") > 0:
                prior = st[:st.index("VERIFIED")]
                k = "HOLD_FAILED" if "HOLD_FAILED" in prior else "EXEC_FAILED" if "EXEC_FAILED" in prior else "PLAN_FAILED"
                rescue[k] += 1
                break
    acc["units_verified_on_later_branch"] = sum(rescue.values())
    acc["units_rescued_after_hold_failure"] = rescue["HOLD_FAILED"]       # V1: pose rejected (branch-0 hold) or EXEC_FAILED (held_state_lost)
    acc["units_rescued_after_exec_failure"] = rescue["EXEC_FAILED"]       # V1: unit EXEC_FAILED (no other branch executed)
    acc["units_rescued_after_plan_failure"] = rescue["PLAN_FAILED"]       # V1 equivalent (its plan loop tried the next branch)
    acc["units_first_eligible_hold_failed_later_held"] = sum(
        bool(first_el[m["rank"]] is not None and m["branches"][first_el[m["rank"]]]["hold"] is not None
             and not m["branches"][first_el[m["rank"]]]["hold"]["ok"] and any(br["hold"] and br["hold"]["ok"] for br in m["branches"])) for m in modes)
    # every candidate branch has exactly one terminal class (zero unaccounted branches)
    cls = {}
    for m in modes:
        for br in m["branches"]:
            k = ("NOT_ELIGIBLE" if not br["eligible"] else
                 ("HOLD_UNTESTED_POSE_" + m["screen"] if m["screen"] != "VALID" and m["screen"] != "HELD_INVALID_PHYSICAL" else "HOLD_UNTESTED_UNSCHEDULED")
                 if br["hold"] is None else "HOLD_FAILED" if not br["hold"]["ok"]
                 else ("HELD_" + br["results"][-1]["state"]) if br["results"] else "HELD_NOT_MOTION_TESTED")
            cls[k] = cls.get(k, 0) + 1
    acc["branch_terminal"] = cls
    acc["branch_unaccounted"] = acc["branch_candidates"] - sum(cls.values())
    return acc


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


def motion_work(rec):
    """Counted work of the motion phase (test-time hold certificates + branch plans + executions): comparable across policies."""
    tot = {}
    for m in rec.get("modes", []):
        for br in m["branches"]:
            if br["hold"] and br["hold"]["at"] == "test":
                for k, v in br["hold"]["work"].items():
                    tot[k] = tot.get(k, 0) + v
            for r in br["results"]:
                for part in ("work_plan", "work_exec"):
                    for k, v in (r.get(part) or {}).items():
                        tot[k] = tot.get(k, 0) + v
    return tot


def virtual(rec, stages):
    """Label, verified (rank, branch), unit states, witness id and counted motion work of an early-stop policy, replayed from a reference_v2
    record (every valid unit tested once at B_ref = 5000 over all eligible branches, no early stop).  Prefix property: a route found after
    r iterations is found under B iff r <= B, and the same seed gives the same route, so every branch result downstream is identical;
    otherwise the unit is UNRESOLVED at B with the pre-route work of the branch that planned the route plus the first B iterations
    (milestone snapshot).  Holds are deterministic and cached, so each test-time hold certificate is charged once."""
    valid = [m for m in rec["modes"] if m["screen"] == "VALID"]
    state, work, charged, out, need = {}, {}, set(), None, 0
    add = lambda d: [work.__setitem__(k, work.get(k, 0) + v) for k, v in (d or {}).items()]  # noqa: E731
    for M, B in stages:
        need = max(need, M)
        for m in valid[:M]:
            r, s = m["rank"], state.get(m["rank"])
            if s in FINAL or (s and int(s.split("@")[1]) >= B):
                continue
            T, states = m["tests"][-1], []
            for b in T["branch_seq"]:
                br = m["branches"][b]
                res = br["results"][-1]
                if br["hold"]["at"] == "test" and (r, b) not in charged:
                    add(br["hold"]["work"]); charged.add((r, b))
                if res["state"] == "HOLD_FAILED":
                    states.append("HOLD_FAILED"); continue
                if res.get("route_here"):
                    ri = T["route"]
                    if not (ri["found"] and (ri["direct"] or ri["iters"] <= B)):
                        add(res.get("work_pre_route")); add(ri["work"] if B >= T["iters"] else ri["milestones"].get(str(B), ri["work"]))
                        states.append("UNRESOLVED"); break
                add(res.get("work_plan")); add(res.get("work_exec"))
                states.append(res["state"])
                if res.get("route_error"):
                    break
                if res["state"] == "VERIFIED":
                    out = (r, b, res["witness_id"]); break
            st = unit_state(states)
            state[r] = f"UNRESOLVED@{B}" if st == "UNRESOLVED" else st
            if out is not None:
                break
        if out is not None:
            break
    label = 1 if out is not None else 0 if valid else rec["label"]
    return {"label": label, "verified": list(out) if out else None, "states": {str(k): v for k, v in sorted(state.items())}, "work": work}
