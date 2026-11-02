"""P17-3 physically realized 2D mechanisms (geometry-first; no model output consulted).

Finite-domain NEGATIVE proofs are pose-only: every sequence of the declared P17 domain (p17_domain.slot_fn on the hypothetical occupancy,
<= n moves, per-object limit, no consecutive repeat) ends in an occupancy whose target pick is refuted by a MOTION-INDEPENDENT sound
certificate (TARGET_PICK / GOAL_OVERLAP / STATIC_ENDPOINT_INVALID / FIRST_MOVER via ordered.certificates_for, checker-accepted).
That refutes every plan of that kind whatever the motions. POSITIVE evidence is always a physically certified, independently replayed
plan from the corrected search (p17_search) on the same domain. Generator: random scenes (target, goal, 2-4 movers near the target,
0-12 fixed structures, random start stratum); every attempt and rejection reason is counted."""
from __future__ import annotations

import time

import numpy as np

Q_INDEP = {"TARGET_PICK", "GOAL_OVERLAP", "STATIC_ENDPOINT_INVALID", "FIRST_MOVER"}


class PoseOracle:
    """Memoized MOTION-INDEPENDENT refutations (checker-accepted, pose-only): target pick at an occupancy, and a relocation (o -> d) at an
    occupancy (ORDER_FIRST_ACTION / STATIC_ENDPOINT_INVALID). Unrefuted steps are treated as possible (sound over-approximation)."""

    def __init__(self, phys):
        self.phys, self.memo, self.mmemo, self.calls = phys, {}, {}, 0

    def _ln(self, s):
        import data
        return data.make_query(self.phys.cfg, "c", s, "t0", self.phys.goal_cell, {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]

    def blocked(self, cells) -> list:
        k = tuple(sorted(cells.items()))
        if k not in self.memo:
            import ordered as od
            self.calls += 1
            s = self.phys.scene(cells, self.phys.s0.robot.q_rad)
            self.memo[k] = sorted({c["kind"] for c in od.certificates_for(self.phys.robot, self.phys.cfg, s, self._ln(s), []) if c["checker_accepted"]} & Q_INDEP)
        return self.memo[k]

    def move_refuted(self, cells, o, d) -> list:
        k = (tuple(sorted(cells.items())), o, d)
        if k not in self.mmemo:
            import ordered as od
            self.calls += 1
            s = self.phys.scene(cells, self.phys.s0.robot.q_rad)
            self.mmemo[k] = sorted(c["kind"] for c in od.certificates_for(self.phys.robot, self.phys.cfg, s, self._ln(s), [[o, d]])
                                   if c["checker_accepted"] and c["kind"] in ("ORDER_FIRST_ACTION", "STATIC_ENDPOINT_INVALID"))
        return self.mmemo[k]


def pose_reach(task, phys, idx=None, n_max=4, width=4, early=True) -> dict:
    """Breadth-first over the declared domain with pose-only refutations of steps and of the final target pick. For per_obj = 1 (no repeat)
    and, when no-repeat is fully refuted (or early=False), per_obj = 2: the shortest length with an UNREFUTED plan shape, examples, and
    whether that per_obj class is fully refuted up to n_max (a sound finite-domain negative)."""
    import p17_domain as dm
    cells0 = {o.id: o.cell for o in phys.s0.objects}
    mv, _ = dm.movers(task, cells0)
    sl = dm.slot_fn(phys, idx, width=width, origin=cells0)
    po = PoseOracle(phys)
    out = {}
    for per_obj in (1, 2):
        frontier, first, ex, sizes = [([], cells0, None, {})], None, [], {}
        for L in range(n_max + 1):
            sizes[L] = len(frontier)
            hits = [f for f in frontier if not po.blocked(f[1])]
            if hits:
                first, ex = L, [h[0] for h in hits[:6]]
                if early:
                    break
            if L == n_max:
                break
            nxt = []
            for seq, cells, last, counts in frontier:
                for o in mv:
                    if counts.get(o, 0) >= per_obj or o == last:
                        continue
                    for d in sl(cells, o):
                        if po.move_refuted(cells, o, d):
                            continue
                        nxt.append((seq + [(o, d)], {**cells, o: d}, o, {**counts, o: counts.get(o, 0) + 1}))
            uniq = {}
            for f in nxt:
                uniq.setdefault((tuple(sorted(f[1].items())), f[2], tuple(sorted(f[3].items()))), f)
            frontier = list(uniq.values())
        out[per_obj] = {"min_unrefuted_len": first, "examples": ex, "frontier_sizes": sizes, "fully_refuted_to_n": first is None}
        if per_obj == 1 and first is not None and early:
            break
    out["oracle_calls"] = po.calls
    return out


def random_spec(rng):
    """Physics-first random layout: target, goal, movers within Chebyshev 2 of the target, fixed structures near the target."""
    import config as cf
    import geometry as ge
    import p6_scenes as p6
    cfg = cf.load_config("config.toml")
    scr = p6.Screens(cfg, ge.make_robot(cfg))
    valid = sorted(scr.valid)
    T = int(rng.choice([c for c in valid if c % 8 >= 4]))
    goal = int(rng.choice([c for c in valid if p6._cheb(c, T) >= 3]))
    adj = [c for c in range(64) if c != T and p6._cheb(c, T) == 1 and p6._cheb(c, goal) >= 2]
    ring2 = [c for c in range(64) if p6._cheb(c, T) == 2 and p6._cheb(c, goal) >= 2]
    a = [int(x) for x in rng.permutation(adj)]
    k_m = int(rng.integers(2, 5))
    n_open = int(rng.integers(1, 3))                       # target neighbours left to movers (rest fixed): enclosed target
    movers = a[:n_open]
    fixed = a[n_open:]
    r2 = [int(x) for x in rng.permutation(ring2) if x not in movers + fixed]
    movers += r2[:k_m - len(movers)] if k_m > len(movers) else []
    fixed += r2[len(movers):len(movers) + int(rng.integers(0, 6))]
    return {"T": T, "goal": goal, "movers": movers, "fixed": fixed, "stratum": str(rng.choice(["S0", "S1", "S2", "S3"]))}


def build_root(spec, key):
    """A plan-root-like dict (scene, goal_cell, fixed) for p16_common.task_from_root; None if the start configuration collides."""
    from dataclasses import replace
    import config as cf
    import geometry as ge
    import p6_scenes as p6
    from scene import place_scene, scene_to_dict
    cfg = cf.load_config("config.toml")
    s = place_scene(cfg, key, "p17mech", [spec["T"]], spec["movers"] + spec["fixed"])
    s = replace(s, robot=replace(s.robot, q_rad=tuple(p6.STRATA[spec["stratum"]])))
    d, e = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"]
    if ge.Checker(ge.make_robot(cfg), ge.make_world(s), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status != ge.CLEAR:
        return None
    return {"scene": scene_to_dict(s), "goal_cell": spec["goal"], "fixed": spec["fixed"], "scene_hash": key, "spec": spec,
            "fingerprint": [spec["T"], spec["goal"], sorted(spec["movers"] + spec["fixed"])]}


def screen(seed0, n, want, n_max=4) -> dict:
    """Generate n random roots; keep those whose POSE-ONLY analysis matches `want` ('repeat': no-repeat plans all refuted but a repeat
    end state unrefuted; 'long3' / 'long4': every shorter end state refuted). Physics only; acceptance statistics returned."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    idx, stats, keep = pl.index_v2()[0], {"attempts": 0, "start_collision": 0, "trivial": 0, "accepted": 0}, []
    for i in range(n):
        rng = np.random.default_rng([seed0, i])
        spec = random_spec(rng)
        stats["attempts"] += 1
        root = build_root(spec, f"p17mech_{seed0}_{i}")
        if root is None:
            stats["start_collision"] += 1
            continue
        task = pc.task_from_root(root, root["scene_hash"])
        phys = pm.DetPhysics(task, idx)
        pr = pose_reach(task, phys, idx, n_max, early=want == "repeat")
        a = pr[1]["min_unrefuted_len"]
        b = pr.get(2, {}).get("min_unrefuted_len")
        ok = {"repeat": a is None and b is not None, "long3": a is not None and a >= 3, "long4": a is not None and a >= 4,
              "any_nontrivial": a is not None and a >= 1}[want]
        if not ok:
            stats["trivial"] += 1
            continue
        stats["accepted"] += 1
        keep.append({"seed": [seed0, i], "root": root, "pose": {str(k): v for k, v in pr.items()}})
    return {"stats": stats, "kept": keep}


def physical(root, n_max=4, per_obj=2, node_cap=3000, budget=None) -> dict:
    """Positive evidence: corrected search on the same domain, independent replay; plus the same search with per_obj = 1."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    import p17_domain as dm
    import p17_search as s17
    idx = pl.index_v2()[0]
    task = pc.task_from_root(root, root["scene_hash"])
    out = {}
    for po in (per_obj, 1):
        c0 = time.process_time()
        phys = pm.DetPhysics(task, idx, budget=budget or {"motion_checks": 1500, "online_moves": 24, "goal_plans": 48, "transit_plans": 24})
        cells0 = {o.id: o.cell for o in phys.s0.objects}
        mv, _ = dm.movers(task, cells0)
        r = s17.search(phys, lambda c, q, m, cand: cand, mv, dm.slot_fn(phys, idx, width=4, origin=cells0), cells0, phys.s0.robot.q_rad, n_max, po, node_cap)
        rep = pm.replay(phys, r["plan"]) if r["found"] else None
        out[po] = {"found": r["found"], "claim": r["claim"], "replay": rep, "cpu_s": time.process_time() - c0, "used": phys.used,
                   "steps": [(x["object"], x["dest"], x["via"], x.get("mode"), x.get("transit")) for x in r["plan"]["steps"]] if r["found"] else None,
                   "travel": r["plan"]["travel"] if r["found"] else None, "edges": r["log"]["edges"], "goals": r["log"]["goals"], "nodes": r["log"]["nodes"]}
        if po == 1 or not r["found"]:
            break
    return out


def reach_spec(rng):
    """Reach-line generator: movers on the cells crossed by the segment robot base -> target (where a parked object re-blocks the final
    pick configuration), fixed structures flanking that corridor and around the target; random stratum. Geometry only."""
    import config as cf
    import geometry as ge
    import p6_scenes as p6
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    scr = p6.Screens(cfg, robot)
    valid = sorted(scr.valid)
    T = int(rng.choice([c for c in valid if c % 8 >= 5]))
    goal = int(rng.choice([c for c in valid if p6._cheb(c, T) >= 3]))
    b, t = np.asarray(robot.base, float), np.asarray(p6._xy(T), float)
    seg = lambda c: float(np.abs(np.cross(t - b, np.asarray(p6._xy(c)) - b)) / np.linalg.norm(t - b))  # noqa: E731
    along = lambda c: float(np.dot(np.asarray(p6._xy(c)) - b, t - b) / np.linalg.norm(t - b) ** 2)  # noqa: E731
    line = sorted([c for c in range(64) if c != T and seg(c) < 0.05 and 0.2 < along(c) < 1.0 and p6._cheb(c, goal) >= 2], key=lambda c: -along(c))
    if len(line) < 2:
        return None
    k = int(rng.integers(1, 3)) + 1
    movers = line[:k]
    flank = sorted({x for c in movers + [T] for x in range(64) if p6._cheb(x, c) == 1} - set(movers) - {T, goal} - set(line))
    flank = [x for x in flank if p6._cheb(x, goal) >= 2]
    fixed = [int(x) for x in rng.permutation(flank)[:int(rng.integers(max(1, len(flank) // 2), len(flank) + 1))]]
    extra = [c for c in valid if c not in movers + fixed + [T, goal] and p6._cheb(c, T) <= 3 and p6._cheb(c, goal) >= 2]
    movers += [int(x) for x in rng.permutation(extra)[:int(rng.integers(0, 2))]]
    return {"T": T, "goal": goal, "movers": movers, "fixed": fixed, "stratum": str(rng.choice(["S0", "S1", "S2", "S3"])), "gen": "reach"}


def screen2(seed0, n, want, n_max=4, gen="reach") -> dict:
    """screen() with a selectable generator; writes nothing (caller persists)."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    idx, stats, keep = pl.index_v2()[0], {"attempts": 0, "no_layout": 0, "start_collision": 0, "rejected": 0, "accepted": 0, "cpu_s": 0.0}, []
    for i in range(n):
        c0 = time.process_time()
        rng = np.random.default_rng([seed0, i])
        spec = {"reach": reach_spec, "corridor": corridor_spec, "corridor2": corridor2_spec}.get(gen, random_spec)(rng)
        stats["attempts"] += 1
        if spec is None:
            stats["no_layout"] += 1
            continue
        root = build_root(spec, f"p17mech_{gen}_{seed0}_{i}")
        if root is None:
            stats["start_collision"] += 1
            continue
        task = pc.task_from_root(root, root["scene_hash"])
        phys = pm.DetPhysics(task, idx)
        pr = pose_reach(task, phys, idx, n_max, early=want == "repeat")
        a, b = pr[1]["min_unrefuted_len"], pr.get(2, {}).get("min_unrefuted_len")
        ok = {"repeat": a is None and b is not None, "long3": a is not None and a >= 3, "long4": a is not None and a >= 4,
              "any_nontrivial": a is not None and a >= 1}[want]
        stats["cpu_s"] += time.process_time() - c0
        stats.setdefault("min_len_no_repeat", {})
        stats["min_len_no_repeat"][str(a)] = stats["min_len_no_repeat"].get(str(a), 0) + 1
        stats.setdefault("scenes", []).append([seed0, i, a, b])
        if not ok and a is not None and a >= 3:                      # long-repair (M8) candidate kept whatever the screen's target
            keep.append({"seed": [seed0, i], "root": root, "pose": {str(k): v for k, v in pr.items()}, "tag": f"long{a}"})
        if not ok:
            stats["rejected"] += 1
            continue
        stats["accepted"] += 1
        keep.append({"seed": [seed0, i], "root": root, "pose": {str(k): v for k, v in pr.items()}})
    return {"stats": stats, "kept": keep}


def mine_m5(root, key, n_max=4, node_cap=3000) -> dict:
    """M5 intermediate swept obstruction from a physically certified plan: for each relocation step j and each ALTERNATIVE domain slot d'
    of that object, re-certify every LATER recorded trajectory (later relocations, final target action) in the occupancy where the object
    sits at d' instead. An instance = both endpoints of a later trajectory are collision-free there but the continuous certification
    reports COLLISION mid-sweep (certified, with the colliding pair), while the original plan replays CLEAR. The alternate-slot
    sequence is also run through the existential verifier (status recorded; UNKNOWN is never called a proof)."""
    import geometry as ge
    import motion as mo
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    import p17_domain as dm
    import p17_search as s17
    import p17_verify as pv
    idx = pl.index_v2()[0]
    task = pc.task_from_root(root, key)
    phys = pm.DetPhysics(task, idx, budget={"motion_checks": 1500, "online_moves": 24, "goal_plans": 48, "transit_plans": 24})
    cells0 = {o.id: o.cell for o in phys.s0.objects}
    mv, _ = dm.movers(task, cells0)
    sl = dm.slot_fn(phys, idx, width=4, origin=cells0)
    r = s17.search(phys, lambda c, q, m, cand: cand, mv, sl, cells0, phys.s0.robot.q_rad, n_max, 2, node_cap)
    if not r["found"] or not r["plan"]["steps"]:
        return {"found": r["found"], "instances": []}
    rep = pm.replay(phys, r["plan"])
    steps, recs = r["plan"]["steps"], r["plan"]["records"]
    seg, i = [], 0                                           # records per step (transit + relocation for lookup edges)
    for st in steps:
        n = 2 if st["via"] == "lookup" else 1
        seg.append(recs[i:i + n])
        i += n
    target_rec = recs[-1]
    p = phys.p
    out = []
    cells = dict(cells0)
    states = [dict(cells)]
    for st in steps:
        cells = {**cells, st["object"]: st["dest"]}
        states.append(dict(cells))
    for j, st in enumerate(steps):
        o = st["object"]
        for d2 in sl(states[j], o):
            if d2 == st["dest"]:
                continue
            for k in range(j + 1, len(steps) + 1):                # later actions: relocation k or the target (k == len(steps))
                occ = {**states[k], o: d2} if all(s["object"] != o for s in steps[j + 1:k]) else None
                if occ is None or d2 in {c for x, c in occ.items() if x != o and x != (steps[k]["object"] if k < len(steps) else "t0")}:
                    continue
                s_alt = phys.scene(occ, phys.s0.robot.q_rad)
                moved = steps[k]["object"] if k < len(steps) else "t0"
                for a in (seg[k] if k < len(steps) else [target_rec]):
                    if not a["held_local"].size and all(ph == "pregrasp" for ph in a["phase"]):
                        continue                                  # transit segments are not the swept action of interest
                    try:
                        v = mo.certify_trajectory(phys.robot, s_alt, moved, a, p["delta"], p["eps"], p["evals"])
                    except ValueError:
                        continue
                    ck = mo.checkers(phys.robot, s_alt, moved, None, None, p["delta"], p["eps"])[("start", False)]
                    ends_ok = all(ck.status_at(np.asarray(qq), a["gap"][0]).status == ge.CLEAR for qq in (a["q"][0],)) if not a["attached"][0] else True
                    col = [x for x in v if x.status == ge.COLLISION]
                    if col and ends_ok:
                        alt_seq = [(s["object"], d2 if s is st else s["dest"]) for s in steps]
                        e = pv.verify_existential(phys, alt_seq) if k < len(steps) + 1 else None
                        out.append({"step": j, "object": o, "slot_used": st["dest"], "slot_alt": d2, "later_action": k, "later_object": moved,
                                    "colliding_pairs": sorted({x.pair for x in col}), "segments_colliding": len(col),
                                    "alt_sequence_status": e["status"] if e else None})
                        break
    return {"found": True, "plan": [(s["object"], s["dest"], s["via"]) for s in steps], "replay": rep, "claim": r["claim"], "instances": out}


def corridor_spec(rng):
    """Dead-end shelf corridor along an edge row (row 0 or 7, columns 3..7) with a fixed roof (adjacent row, columns 3..7); target at
    column 7, 1-4 movers chained in front of it, 0-1 movers outside, a few free buffer cells, the remaining valid cells filled with fixed
    structures (fill fraction random). Geometry only; every rejection is counted by the caller."""
    import config as cf
    import geometry as ge
    import p6_scenes as p6
    cfg = cf.load_config("config.toml")
    scr = p6.Screens(cfg, ge.make_robot(cfg))
    row = int(rng.choice([0, 7]))
    roof = 1 if row == 0 else 6
    cell = lambda r, c: 8 * r + c  # noqa: E731
    T = cell(row, 7)
    k = int(rng.integers(1, 5))
    chain = [cell(row, 7 - i) for i in range(1, k + 1)]
    roofc = [cell(roof, c) for c in range(3, 8)]
    outside = [c for c in sorted(scr.valid) if c not in chain + roofc + [T] and c // 8 != row]
    rng.shuffle(outside)
    goal = int(outside[0])
    rest = [c for c in outside[1:] if p6._cheb(c, goal) >= 2]
    n_free = int(rng.integers(1, 6))
    free = rest[:n_free]
    extra = rest[n_free:n_free + int(rng.integers(0, 2))]
    fill = rest[n_free + len(extra):]
    fill = fill[:int(len(fill) * float(rng.uniform(0.3, 1.0)))]
    return {"T": T, "goal": goal, "movers": chain + extra, "fixed": roofc + fill, "stratum": str(rng.choice(["S0", "S1", "S2", "S3"])),
            "gen": "corridor", "free_buffers": free}


EVIDENCE_BUDGET = {"motion_checks": 1500, "online_moves": 24, "goal_plans": 48, "transit_plans": 24}


def evidence(root, key, positive, counterfactuals, tag, out_dir="runs/p17/mechanisms/evidence", render=True) -> dict:
    """Physical evidence package for one mechanism: existential verification + independent replay of the positive ordered sequence
    (witness arrays saved as .npz with the replay hash), existential verification of every counterfactual sequence (status + scope),
    and (render) initial / intermediate / final panels and a GIF drawn only from the replayed arrays."""
    from pathlib import Path
    import config as cf
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    import p17_verify as pv
    idx = pl.index_v2()[0]
    task, cache = pc.task_from_root(root, key), {}
    fac = lambda: pm.DetPhysics(task, idx, budget=EVIDENCE_BUDGET, cache=cache)  # noqa: E731
    ph = fac()
    st0 = ph.goal({o.id: o.cell for o in ph.s0.objects}, np.asarray(ph.s0.robot.q_rad, float))
    pos = pv.verify_existential(ph, [tuple(a) for a in positive])
    rep = pm.replay(ph, pos["plan"]) if pos["plan"] else None
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rec = {"tag": tag, "root": key, "fingerprint": root.get("fingerprint"), "original_status": st0[0],
           "original_certificates": (st0[1] or {}).get("certificates") if isinstance(st0[1], dict) else None,
           "positive": {"sequence": positive, "status": pos["status"], "travel": pos["plan"]["travel"] if pos["plan"] else None,
                        "steps": [(s["object"], s["dest"], s.get("via"), s.get("mode"), s.get("transit")) for s in pos["plan"]["steps"]] if pos["plan"] else None,
                        "replay": rep, "branches_verified": pos.get("branches_verified")},
           "counterfactuals": []}
    if pos["plan"] and rep and rep["ok"]:
        arrs = {f"r{i}_{k}": v for i, a in enumerate(pos["plan"]["records"]) for k, v in a.items()}
        np.savez_compressed(out / f"{tag}_witness.npz", **arrs)
        rec["positive"]["witness_file"] = str(out / f"{tag}_witness.npz")
    for seq in counterfactuals:
        c = pv.verify_existential(fac(), [tuple(a) for a in seq])
        rec["counterfactuals"].append({"sequence": seq, "status": c["status"], "certificate": c.get("certificate"), "fails": c.get("fails", [])[:6]})
    if render and pos["plan"] and rep and rep["ok"]:
        import p17_viz as vz
        rec["figures"] = vz.mechanism(ph, root, pos["plan"], tag, rec)
    cf.write_json(out / f"{tag}.json", rec)
    return rec


def corridor2_spec(rng):
    """Two-wall shelf corridor (as P13 ORD_CHAIN, generalised): interior row r (1..6), target at column 7, a chain of 2-4 movers at
    columns 6..3, fixed walls on rows r - 1 and r + 1 over the chain and the target; 0-1 extra movers; scarce free buffers (the rest of
    the valid region partly filled with fixed structures). Geometry only."""
    import config as cf
    import geometry as ge
    import p6_scenes as p6
    cfg = cf.load_config("config.toml")
    scr = p6.Screens(cfg, ge.make_robot(cfg))
    r = int(rng.integers(1, 7))
    k = int(rng.integers(2, 5))
    cell = lambda rr, c: 8 * rr + c  # noqa: E731
    T = cell(r, 7)
    chain = [cell(r, 7 - i) for i in range(1, k + 1)]
    walls = [cell(rr, c) for rr in (r - 1, r + 1) for c in range(7 - k, 8)]
    used = set(chain + walls + [T])
    outside = [c for c in sorted(scr.valid) if c not in used]
    rng.shuffle(outside)
    goal = next((c for c in outside if all(p6._cheb(c, x) >= 2 for x in chain + [T])), None)
    if goal is None:
        return None
    rest = [c for c in outside if c != goal and p6._cheb(c, goal) >= 2]
    n_free = int(rng.integers(1, 6))
    free, extra = rest[:n_free], rest[n_free:n_free + int(rng.integers(0, 2))]
    fill = rest[n_free + len(extra):]
    fill = fill[:int(len(fill) * float(rng.uniform(0.0, 1.0)))]
    return {"T": T, "goal": goal, "movers": chain + extra, "fixed": walls + fill, "stratum": str(rng.choice(["S0", "S1", "S2", "S3"])),
            "gen": "corridor2", "free_buffers": free}


# ------------------------------------------------------------------ M7: exact finite LIBRARY motion domain (declared scope, not global)
LIB_DOMAIN = "P17_LIBRARY_DOMAIN_V1"


def lib_domain(task, idx, cache=None, target="planner"):
    """Finite, exactly decidable motion domain for relocations: the frozen lookup library's candidate trajectories (first K_MODES verified
    terminal configurations) with DIRECT joint-space transits only; no online relocation planning. A relocation edge with no realization in
    this finite set is REFUTED_IN_LIBRARY_DOMAIN (decided within the domain, never a global negative). Target action: the P16_FAST
    deterministic planner (VERIFIED is sound) and the scoped certificates; a planner failure stays UNKNOWN and blocks any proof.
    target='library': the target action is decided in the same finite set (library target trajectories for (target cell -> goal) with a
    direct transit; none realizable -> REFUTED_IN_LIBRARY_DOMAIN); the planner is used only when the library has no target candidate."""
    import p15_search as ps
    import p16_motion as pm

    class LibDomain(pm.DetPhysics):
        def _goal(self, cells, q, k):
            if target != "library":
                return super()._goal(cells, q, k)
            import geometry as ge
            import motion as mo
            import ordered as od
            import p14_eval as p14
            import p14_sweep_index as sx
            import p15_retarget as pr
            s = self.scene(cells, q)
            c = pr.certified(self.cfg, self.robot, s, "t0", self.goal_cell)
            if c:
                return (ps.CI, {"certificates": c}, 0.0)
            res = sx.lookup_candidates(self.idx, "t0", cells["t0"], self.goal_cell, s, self.goal_cell, s.get("t0").radius_m, "combined", self.modal_only)
            if not res["candidates"]:
                return super()._goal(cells, q, k)
            for cand in res["candidates"]:
                a = self.idx.arrays(cand["entry"])
                tr, how = self._transit(s, "t0", q, a, k)
                if tr is None:
                    continue
                try:
                    v = mo.certify_trajectory(self.robot, s, "t0", a, self.p["delta"], self.p["eps"], self.p["evals"])
                except ValueError:
                    continue
                if all(x.status == ge.CLEAR for x in v):
                    rec = p14._merge(tr, a)
                    return (ps.V, rec, float(od.joint_travel(self.robot, [tr])["J"] or 0.0) + cand["travel_J"])
            return (ps.CI, {"scope": LIB_DOMAIN, "why": "no library target realization", "candidates": len(res["candidates"])}, 0.0)

        def _edges(self, cells, q, o, d, k):
            res = super()._edges(cells, q, o, d, k)
            if res and not any(x[0] == ps.V for x in res) and res[0][0] == ps.U_BUD and \
                    (res[0][4] or {}).get("lookup") in ("LIBRARY_COLLISION", ps.U_TR, ps.U_LIB):
                return [(ps.CI, None, None, 0.0, {"scope": LIB_DOMAIN, "why": res[0][4]["lookup"]})]
            return res

    return LibDomain(task, idx, budget={"motion_checks": 10 ** 9, "online_moves": 0, "goal_plans": 10 ** 6, "transit_plans": 0},
                     transit="direct", cache=cache)


def m7_library(root, key, positive=None, n_max=4, node_cap=10 ** 6, target="library") -> dict:
    """M7 within LIB_DOMAIN: (i) the repeat witness is VERIFIED in the domain and replayed; (ii) the corrected search with per_obj = 1 (no
    repeat) over the same declared slots returns NO_PLAN_IN_FULLY_ADJUDICATED_FINITE_GRAPH (every edge and goal decided); (iii) per_obj = 2
    finds a plan. M7 holds in the domain iff (i) and (ii)."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    import p17_domain as dm
    import p17_search as s17
    import p17_verify as pv
    idx = pl.index_v2()[0]
    task, cache = pc.task_from_root(root, key), {}
    out = {"domain": LIB_DOMAIN, "target_action": target, "root": key}
    ph = lib_domain(task, idx, cache, target)
    cells0, q0 = {o.id: o.cell for o in ph.s0.objects}, np.asarray(ph.s0.robot.q_rad, float)
    mv, _ = dm.movers(task, cells0)
    if positive:
        c0 = time.process_time()
        v = pv.verify_existential(ph, [tuple(a) for a in positive])
        rep = pm.replay(ph, v["plan"]) if v["plan"] else None
        out["positive"] = {"sequence": positive, "status": v["status"], "replay": rep, "travel": v["plan"]["travel"] if v["plan"] else None,
                           "steps": [(s["object"], s["dest"], s.get("via"), s.get("mode"), s.get("transit")) for s in v["plan"]["steps"]] if v["plan"] else None,
                           "cpu_s": time.process_time() - c0}
    for per_obj in (1, 2):
        c0 = time.process_time()
        ph = lib_domain(task, idx, cache, target)
        r = s17.search(ph, lambda c, q, m, cands: cands, mv, dm.slot_fn(ph, idx, width=4, origin=cells0), cells0, q0, n_max, per_obj, node_cap)
        out[f"per_obj_{per_obj}"] = {"found": r["found"], "claim": r["claim"], "flags": r["flags"], "moves": r["plan"]["moves"] if r["found"] else None,
                                     "steps": [(s["object"], s["dest"], s.get("via")) for s in r["plan"]["steps"]] if r["found"] else None,
                                     "edges": r["log"]["edges"], "goals": r["log"]["goals"], "nodes": r["log"]["nodes"], "cpu_s": time.process_time() - c0}
        if per_obj == 1 and r["claim"] != "NO_PLAN_IN_FULLY_ADJUDICATED_FINITE_GRAPH" and not positive:
            break
    out["M7_in_domain"] = bool(out.get("positive", {}).get("status") == pv.V and (out["positive"]["replay"] or {}).get("ok")
                               and out["per_obj_1"]["claim"] == "NO_PLAN_IN_FULLY_ADJUDICATED_FINITE_GRAPH")
    return out
