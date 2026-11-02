"""Exact blocker layouts for the 2D class-quota dataset (requested K >= 2), the 2D counterpart of bd_3dt_plus's exact cover.

The planar skill screens a declared grasp set (24 tool angles x 2 elbow branches) at the target's start (pick) and at its goal
(place); a grasp is killed when any object collides with it in that screen, so kills add up object by object and are exact (no
planner involved).  The target's move needs one live pick grasp and one live place grasp, hence the fewest relocations that can
free it is
    K_static = min over live (pick c, place d) of |movable objects killing c or d|,
a lower bound on the verified K (an object moved to parking can only kill more).  The construction fixes the pair (a_p, a_q) left
cheapest, puts exactly k_p / k_q movers on it (Q; k_p + k_q = K: grasp K+0, place 0+K, mixed split), raises every other grasp of
that end to >= k+1 (else k) movable cutters (R) or kills it with immovable structures next to the start / goal (X), never touching
a_p / a_q, and requires every Q object to keep a screened pick grasp with everything else in place.  The planner then confirms
that the target's move verifies with Q parked and that every Q relocation is certified from the original scene and with the other
Q parked (no causal dependence between the objects); the teacher labels K as usual."""
from __future__ import annotations

import itertools
import json
import math
import time
from dataclasses import replace

import numpy as np

import bd_core as bc
import bd_2d as A

SEARCH_S = 240.0                    # cover search per (start, goal), incl. the repair-motion witnesses
N_Q = 10                            # repair sets tried per grasp
MAX_FIX = 0                         # no immovable structures (user, 1 Nov 2026): movable objects only
FIX_LEVELS = (0,)                   # immovable-structure levels the search may use, in order
X_WEIGHT = 0.3                      # greedy weight of an immovable structure against an extra movable blocker (prefer movable)
MAX_VERIFY = 4                      # layouts sent to the planner per (start, goal) and immovable-structure level
N_PARK_TRY = 6                      # parking cells tried per must-move object
MIN_FREE = {2: 1, 3: 1, 4: 1}       # screened pick grasps each must-move object keeps with everything in place (by K; V6: K=4 1)
N_COLLECT = 8                       # static layouts collected per (start, goal) before ranking
FIX_RADIUS = 2                      # immovable structures within this Chebyshev distance of the start / goal
SPLITS = {"grasp": lambda K: [(K, 0)], "place": lambda K: [(0, K)],
          "mixed": lambda K: [(k, K - k) for k in range(1, K)]}


_BARRIERS, _TABLE = set(), {}


def capacity(cell, side):
    """Most cuts every grasp of this end can get from movable objects alone (min over grasps of its movable cutter cells); 3 when
    the table is missing."""
    side_ok(cell, side, 1)                            # loads the table
    t = _TABLE["t"]
    if t is None:
        return 3
    return min((len(v["movable"]) for v in t["cells"][str(cell)]["pick" if side == "p" else "place"]["per_grasp"].values()), default=0)


def side_ok(cell, side, k):
    """Necessary condition from bd_2d_cover_table.json (per-cell grasp kill maps, geometry only): some grasp of this end has >= k
    movable cutter cells and every grasp has >= k of them or a cell for an immovable structure within FIX_RADIUS.  True when the
    table is missing or was made for another configuration."""
    if "t" not in _TABLE:
        import os
        f = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bd_2d_cover_table.json")
        t = json.load(open(f)) if os.path.exists(f) else None
        _TABLE["t"] = t if t and t.get("config_hash") == A.env()["config_hash"] else None
    t = _TABLE["t"]
    if t is None:
        return True
    ways = list(t["cells"][str(cell)]["pick" if side == "p" else "place"]["per_grasp"].values())
    fixable = lambda v: any(A.cheb(x, cell) <= FIX_RADIUS and x not in barriers() for x in v["movable"] + v["fixed_only"])  # noqa: E731
    return any(len(v["movable"]) >= k for v in ways) and all(len(v["movable"]) >= k or (MAX_FIX and fixable(v)) for v in ways)


def barriers():
    """Cells whose disc would be a link-0 barrier (sector_cert._sector): joint 0 cannot rotate past it, which kills every grasp
    beyond its angle -- not visible to the pose screens, so no object of a layout may sit there."""
    if not _BARRIERS:
        import sector_check as sc
        G = A.env()
        cfg, robot = G["cfg"], G["robot"]
        k, r = sc.constants(robot, cfg), cfg["suite"]["obstacle_radius_m"]
        reach = k["L1"] + max(k["w0_half"], k["housing_r"]) + r + k["delta"] - k["eps"] - sc.TAU
        _BARRIERS.update(c for c in range(64) if math.dist(((c % 8 + 0.5) * 0.08, (c // 8 + 0.5) * 0.08), robot.base) < reach)
    return _BARRIERS


def _mods():
    import motion as mo
    import skill as sk
    from scene import place_scene
    return mo, sk, place_scene


def _screen(src, goal, obst, mover=None, side="p"):
    """Screened grasps: side 'p' = pick of the target (or of `mover`, then the target is an obstacle), 'q' = place of the target."""
    mo, sk, place_scene = _mods()
    G = A.env()
    cfg, robot = G["cfg"], G["robot"]
    P = sk.params(cfg)
    step = 2 * math.pi / P["approach_dirs"]
    s = place_scene(cfg, "cut", "s", [src], ([mover] if mover is not None else []) + list(obst))
    oid = "o00" if mover is not None else "t0"
    o = s.get(oid)
    if side == "p":
        ck = mo.checkers(robot, s, oid, None, None, P["delta"], P["eps"])
        cands = sk.screen_pick(robot, ck, o.xy_m, o.radius_m, P)
    else:
        dest, local = s.lattice.center(goal), (robot.grasp_x, 0.0)
        ck = mo.checkers(robot, s, oid, dest, local, P["delta"], P["eps"])
        cands = sk.screen_place(robot, ck, dest, local, 2 * o.radius_m, P)
    return {(int(round(c.phi / step)) % P["approach_dirs"], c.branch) for c in cands if c.stage == "OK"}


class _Restricted:
    """Context: the skill's pick / place screens keep only grasp ap / aq (None: all) -- a witness for one repair grasp pair."""

    def __init__(self, ap, aq):
        self.ap, self.aq = ap, aq

    def __enter__(self):
        _, sk, _ = _mods()
        P = sk.params(A.env()["cfg"])
        step = 2 * math.pi / P["approach_dirs"]
        self.orig = sk.screen_pick, sk.screen_place

        def only(fn, a):
            def screen(*args, **kw):
                out = fn(*args, **kw)
                for c in out:
                    if a is not None and c.stage == "OK" and (int(round(c.phi / step)) % P["approach_dirs"], c.branch) != a:
                        c.stage = "RESTRICTED"
                return out
            return screen
        sk.screen_pick, sk.screen_place = only(self.orig[0], self.ap), only(self.orig[1], self.aq)

    def __exit__(self, *exc):
        _, sk, _ = _mods()
        sk.screen_pick, sk.screen_place = self.orig


def repair_sweep(task, src, qa, ap, aq):
    """Cells swept by a planner witness of the target's move (empty scene) using pick grasp ap and place grasp aq; None if none."""
    import p14_sweep_index as sx
    G = A.env()
    with _Restricted(ap, aq):
        st, a, _ = A.t_goal(task, {}, {"t0": src}, qa)
    return frozenset(set().union(*map(set, sx.swept_cells(G["robot"], G["cfg"], a).values()))) if st == "VERIFIED" else None


def cut_map(src, goal, cells):
    Fp, Fq = _screen(src, goal, []), _screen(src, goal, [], side="q")
    return {"src": src, "goal": goal, "Fp": Fp, "Fq": Fq,
            "kp": {x: Fp - _screen(src, goal, [x]) for x in cells}, "kq": {x: Fq - _screen(src, goal, [x], side="q") for x in cells}}


def k_static(cm, mov, fixed):
    """(K_static, set of minimal mover sets) over the grasp pairs alive under the immovable objects."""
    live_p = [c for c in cm["Fp"] if not any(c in cm["kp"][x] for x in fixed)]
    live_q = [d for d in cm["Fq"] if not any(d in cm["kq"][x] for x in fixed)]
    if not live_p or not live_q:
        return None, set()
    cp = {frozenset(x for x in mov if c in cm["kp"][x]) for c in live_p}
    cq = {frozenset(x for x in mov if d in cm["kq"][x]) for d in live_q}
    best, sets = None, set()
    for u in {a | b for a in cp for b in cq}:
        if best is None or len(u) < best:
            best, sets = len(u), {u}
        elif len(u) == best:
            sets.add(u)
    return best, sets


def near(c, cells, r=1):
    return any(A.cheb(c, y) <= r for y in cells)


def _greedy(need, kill, Rc, Xc, r_cap, x_cap, Q, rng, banned, soft=frozenset()):
    """Fewest extra movers (R, +1 cut each) / immovable structures (X, kill outright) bringing every need to <= 0; None if impossible."""
    need, R, X = dict(need), [], []
    while any(v > 0 for v in need.values()):
        opts = []
        for kind, cand, cap, w in (("R", Rc, r_cap, 1.0), ("X", Xc, x_cap, X_WEIGHT)):
            if (len(R) if kind == "R" else len(X)) >= cap:
                continue
            for x in cand:
                if x in R or x in X or x in banned:
                    continue
                g = sum((1 if kind == "R" else need[t]) for t in kill[x] if need.get(t, 0) > 0)
                if g:
                    opts.append((g * w * (0.5 if near(x, Q) else 1.0) * (0.25 if x in soft else 1.0), rng.random(), kind, x))
        if not opts:
            return None
        _, _, kind, x = max(opts)
        (R if kind == "R" else X).append(x)
        for t in kill[x]:
            if t in need:
                need[t] = need[t] - 1 if kind == "R" else 0
    return R, X


def _side(cm, side, k, V, Xcells, r_cap, rng, deadline, sweep_of, max_fix=MAX_FIX):
    """Per end: (a, Q, R, X, extra, sweep) with exactly k cutters of a in Q, every other grasp of that end at >= k + extra, and
    no R / X cell on the sweep of a planner witness that uses grasp a (sweep_of(a); None: a is not executable)."""
    F, kill = (cm["Fp"], cm["kp"]) if side == "p" else (cm["Fq"], cm["kq"])
    out, ways = [], sorted(F)
    rng.shuffle(ways)
    for a in ways:
        if time.time() > deadline:
            break
        cut = [x for x in V if a in kill[x]]
        sw = None
        combos = list(itertools.combinations(cut, k))
        combos.sort(key=lambda Q: (sum(A.cheb(x, y) < 2 for x, y in itertools.combinations(Q, 2)), rng.random()))
        for Q in combos[:N_Q]:
            if time.time() > deadline:
                break
            Rc = [x for x in V if x not in Q and a not in kill[x]]
            Xc = [x for x in Xcells if x not in Q and a not in kill[x]]
            for extra in (1, 0):
                need = {b: k + extra - sum(b in kill[x] for x in Q) for b in F if b != a}
                if not _greedy(need, kill, Rc, Xc, r_cap, max_fix, Q, rng, set()):
                    continue                          # cheap static test first; the witness only for coverable ways
                if sw is None:
                    sw = sweep_of(a) or frozenset(["none"])
                if "none" in sw:
                    break
                g = _greedy(need, kill, [x for x in Rc if x not in sw], [x for x in Xc if x not in sw], r_cap, max_fix, Q, rng, set())
                hard = g is not None
                if not hard:                          # off the witness if possible, else on it at a penalty (the planner decides)
                    g = _greedy(need, kill, Rc, Xc, r_cap, max_fix, Q, rng, set(), soft=sw)
                if g:                                 # keep the unique-repair and the lean variant (fewer extra movers)
                    out.append({"a": a, "Q": list(Q), "R": g[0], "X": g[1], "extra": extra, "sweep": sw, "hard": hard})
            if sw is not None and "none" in sw:
                break
    return out


def layouts(cm, kp_, kq_, budget_mov, rng, V, Xcells, search_s, pickable, sweep, max_fix=MAX_FIX, stats=None):
    """Statically exact layouts, best first (unique repair, then fewest objects); `pickable(cells, x)` is the mover screen,
    `sweep(ap, aq)` the witness sweep of the repair motion (None: not executable)."""
    K, t0 = kp_ + kq_, time.time()
    share = 0.4 if kp_ and kq_ else 0.6
    none = [{"a": None, "Q": [], "R": [], "X": [], "extra": 1, "sweep": frozenset(), "hard": True}]
    stats = {} if stats is None else stats
    bump = lambda k: stats.__setitem__(k, stats.get(k, 0) + 1)  # noqa: E731
    r_cap = budget_mov - K                                     # extra movers, shared by both ends
    sp = _side(cm, "p", kp_, V, Xcells, r_cap, rng, t0 + share * search_s, lambda a: sweep(a, None), max_fix) if kp_ else none
    sq = _side(cm, "q", kq_, V, Xcells, r_cap, rng, time.time() + share * search_s, lambda a: sweep(None, a), max_fix) if kq_ else none
    if not sp or not sq:
        bump("no_side_cover")
    deadline = t0 + search_s
    combos = sorted(itertools.product(sp, sq), key=lambda ab: (-(ab[0]["hard"] + ab[1]["hard"]), -(ab[0]["extra"] + ab[1]["extra"]),
                                                             sum(len(s["Q"]) + len(s["R"]) + len(s["X"]) for s in ab), rng.random()))
    found = []
    for p, q in combos:
        if time.time() > deadline:
            break
        Q = p["Q"] + q["Q"]
        if len(set(Q)) != len(Q):                              # one object cutting both repair grasps: fewer than K movers
            bump("combo_qq")
            continue
        Rp, Rq = [x for x in p["R"] if x not in Q], [x for x in q["R"] if x not in Q]   # V6: an extra blocker both ends chose
        if (p["a"] is not None and any(p["a"] in cm["kp"][x] for x in Rq + q["X"])) or \
           (q["a"] is not None and any(q["a"] in cm["kq"][x] for x in Rp + p["X"])):
            bump("combo_cross")                                # (or a mover of the other end) is one object serving both
            continue
        R, X = list(dict.fromkeys(Rp + Rq)), list(dict.fromkeys(p["X"] + q["X"]))
        if len(Q) + len(R) > budget_mov or len(X) > max_fix:
            bump("combo_budget")
            continue
        k, sets = k_static(cm, Q + R, X)
        if k != K or frozenset(Q) not in sets:
            bump("K_static")
            continue
        sw = p["sweep"] | q["sweep"]
        hard = p["hard"] and q["hard"]
        if p["a"] is not None and q["a"] is not None:         # mixed: one witness must use both repair grasps
            sw = sweep(p["a"], q["a"])
            if sw is None:
                bump("no_joint_witness")
                continue
            hard = hard and not any(x in sw for x in R + X)
        free = min(pickable(Q + R + X, x) for x in Q)
        if free < MIN_FREE[K]:
            bump("mover_grasps")
            continue
        found.append({"Q": Q, "R": R, "X": X, "K_static": k, "unique": len(sets) == 1, "ap": p["a"], "aq": q["a"], "sweep": sw,
                      "hard": hard, "free": free, "size": len(Q + R + X)})
        if len(found) >= N_COLLECT or time.time() > t0 + 0.6 * search_s:
            break
    # most robust first: repair motion kept clear, then the must-move objects' spare grasps, unique repair, fewest objects
    yield from sorted(found, key=lambda L: (not L["hard"], -min(L["free"], 6), not L["unique"], L["size"]))


def construct_exact(spec, seed, root_id, n_obj, attempts=3, construct_s=1200.0) -> dict:
    """One root with requested K >= 2 and mechanism grasp / place / mixed.  Raises A.Reject."""
    import p14_sweep_index as sx
    import p16_common as pc
    import p6_scenes as p6
    from scene import place_scene, scene_to_dict
    G = A.env()
    cfg, robot = G["cfg"], G["robot"]
    rng = np.random.default_rng(seed)
    K, t_start = spec["K"], time.time()
    lo, hi = A.DISP[spec["displacement"]]
    stratum = "S0" if spec["start_pose"] == "home" else str(rng.choice(["S1", "S2", "S3"]))
    q0 = tuple(p6.STRATA[stratum])
    qa = np.asarray(q0, float)
    lo_d, hi_d = bc.N_DISTRACT[spec["distractors"]]
    budget_mov = n_obj - lo_d
    stats = {"maps": 0, "layouts": 0, "T_blocked_after_repair": 0, "dependent": 0, "distractors": 0}

    def scene_of(cells, tag):
        s = place_scene(cfg, root_id, tag, [src], cells)
        return replace(s, robot=replace(s.robot, q_rad=q0))

    def task_of(s, tag):
        return {"scene": scene_to_dict(s), "goal_cell": goal, "fixed_ids": [], "target": "t0", "key": f"{root_id}:{tag}"}

    def t_ok(cells, tag):
        s = scene_of(cells, tag)
        st, a, _ = A.t_goal(task_of(s, tag), {}, {o.id: o.cell for o in s.objects}, qa)
        return (a if st == "VERIFIED" else None)

    def edge_ok(cells, mover_cell, dest, tag):
        """Certified relocation of the object at mover_cell to dest, everything in `cells` in place (robot at q0)."""
        s = scene_of(cells, tag)
        oid = next(o.id for o in s.objects if o.cell == mover_cell)
        phys = A._phys(task_of(s, tag), {})
        return pc.best_edge(phys.edges({o.id: o.cell for o in s.objects}, qa, oid, int(dest))) is not None

    for attempt in range(attempts):
        if time.time() - t_start > construct_s:
            break
        cands = [(x, g, kp_, kq_) for x in G["region"][spec["source"]] for g in G["valid"] if lo <= A.cheb(g, x) <= hi
                 for kp_, kq_ in SPLITS[spec["mechanism"]](K) if (not kp_ or side_ok(x, "p", kp_)) and (not kq_ or side_ok(g, "q", kq_))]
        if not cands:
            raise A.Reject("NO_BLOCKABLE_START_GOAL")         # this start area / move length cannot hold K independent blockers
        w = np.array([4.0 if kp_ and kq_ and abs(kp_ - kq_) <= 1 else 1.0 for _, _, kp_, kq_ in cands])   # mixed: balanced split
        src, goal, kp_, kq_ = (int(v) for v in cands[int(rng.choice(len(cands), p=w / w.sum()))])
        a_empty = t_ok([], f"empty{attempt}")
        if a_empty is None:
            continue
        sw_empty = sx.swept_cells(robot, cfg, a_empty)
        V = [c for c in G["valid"] if c not in (src, goal)]
        anchors = ([src] if kp_ else []) + ([goal] if kq_ else [])
        Xcells = [c for c in range(64) if c not in (src, goal) and c not in barriers() and min(A.cheb(c, x) for x in anchors) <= FIX_RADIUS]
        cm = cut_map(src, goal, sorted(set(V) | set(Xcells)))
        stats["maps"] += 1
        stats.setdefault("tried", []).append(f"{src}>{goal}:{kp_}+{kq_}")
        memo = {}

        def pickable(cells, x):
            k = (frozenset(cells), x)
            if k not in memo:
                memo[k] = len(_screen(src, goal, [c for c in cells if c != x], mover=x))
            return memo[k]
        task0 = task_of(scene_of([], f"wit{attempt}"), f"wit{attempt}")
        wmemo = {}

        def sweep(ap, aq):
            if (ap, aq) not in wmemo:
                wmemo[(ap, aq)] = repair_sweep(task0, src, qa, ap, aq)
                stats["witnesses"] = stats.get("witnesses", 0) + 1
            return wmemo[(ap, aq)]
        def staged():                                 # movable objects only first; one immovable structure more per level
            for lvl in FIX_LEVELS:
                stats["fix_level"] = lvl
                for i, lay in enumerate(layouts(cm, kp_, kq_, budget_mov, rng, V, Xcells, SEARCH_S / 2, pickable, sweep, lvl, stats)):
                    if i >= MAX_VERIFY:
                        break
                    yield lay
        for lay in staged():
            if time.time() - t_start > construct_s:
                break
            stats["layouts"] += 1
            stats["soft"] = stats.get("soft", 0) + (not lay["hard"])
            Q, R, X = lay["Q"], lay["R"], lay["X"]
            occ = Q + R + X
            seen = set().union(*map(set, sw_empty.values()))
            spoil = lambda c: (lay["ap"] is not None and lay["ap"] in cm["kp"].get(c, ())) or \
                              (lay["aq"] is not None and lay["aq"] in cm["kq"].get(c, ()))  # noqa: E731
            forbid = lambda c: spoil(c) or c in lay["sweep"]  # noqa: E731
            park, nearby = {}, {c for x in occ for c in G["valid"] if A.cheb(c, x) == 1}
            for x in Q:                               # per mover: ranked parking cells, place screen, certified relocation edge
                ranked = sorted((c for c in G["valid"] if c not in occ and c not in park.values() and c not in (src, goal)
                                 and A.cheb(c, src) >= 2 and A.cheb(c, goal) >= 2 and not forbid(c)),
                                key=lambda c: (c in seen, c in nearby, A.cheb(c, x), c))
                for c in ranked[:N_PARK_TRY]:
                    if not _screen(src, c, [y for y in occ if y != x], mover=x, side="q"):
                        stats["no_place"] = stats.get("no_place", 0) + 1
                    elif edge_ok(occ, x, c, f"ind{attempt}"):
                        park[x] = c
                        break
                    else:
                        stats["dep_orig"] = stats.get("dep_orig", 0) + 1
                if x not in park:
                    break
            if len(park) < len(Q):
                stats["dependent"] += 1
                continue
            if not all(edge_ok([park[y] if y in park and y != x else y for y in occ], x, park[x], f"indp{attempt}") for x in Q):
                stats["dependent"] += 1
                stats["dep_parked"] = stats.get("dep_parked", 0) + 1
                continue
            a_rep = t_ok([park[x] if x in park else x for x in occ], f"rep{attempt}.{stats['layouts']}")
            if a_rep is None:
                stats["T_blocked_after_repair"] += 1
                continue
            seen |= set().union(*map(set, sx.swept_cells(robot, cfg, a_rep).values()))
            # distractors: requested role, never killing the repair pair, clear of the must-move objects and their parking
            near_sw = {c for c in G["valid"] for x in seen if A.cheb(c, x) == 1}
            roles = {"near_T": [c for c in G["valid"] if 1 <= A.cheb(c, src) <= 3 and c not in seen],
                     "near_goal": [c for c in G["valid"] if 1 <= A.cheb(c, goal) <= 3 and c not in seen],
                     "near_path": [c for c in near_sw if c not in seen],
                     "far": [c for c in G["valid"] if A.cheb(c, src) >= 3 and A.cheb(c, goal) >= 3 and c not in seen]}
            dpool = sorted({c for c in roles[spec["distractor_role"]] if c in V and c not in occ and c not in park.values()
                            and not spoil(c) and not near(c, Q) and c not in lay["sweep"]})
            want = min(int(rng.integers(lo_d, hi_d + 1)), n_obj - len(Q) - len(R))
            nd = min(want, len(dpool))
            if nd < lo_d:
                stats["distractors"] += 1
                continue
            for _ in range(3):
                dis = [int(x) for x in rng.choice(dpool, nd, replace=False)] if nd else []
                full = Q + R + dis + X
                if not nd or (t_ok([park[x] if x in park else x for x in full], f"repd{attempt}") is not None and
                              all(pickable(full, x) > 0 and edge_ok(full, x, park[x], f"indd{attempt}") for x in Q)):
                    break
            else:
                stats["distractors"] += 1
                continue
            return _record(spec, G, src, goal, q0, stratum, Q, R, dis, X, park, seen, sw_empty, a_empty, lay | {"split": [kp_, kq_]},
                           stats, attempt, scene_of, lambda c: spoil(c) or c in lay["sweep"])
    raise A.Reject("EXACT_LAYOUT_NOT_FOUND:" + ",".join(f"{k}={v}" for k, v in stats.items()))


NONE_K = 5                          # NONE: freeing the target needs >= 5 relocations, more than the dataset's K <= 4


def pair_fill(cm, V, budget, target, rng):
    """Movable blockers, added greedily, until every (pick grasp, place grasp) pair is cut by >= target distinct objects
    (K_static >= target); None if the budget runs out or some pair cannot get there."""
    mov = []
    while True:
        cp = {c: frozenset(x for x in mov if c in cm["kp"][x]) for c in cm["Fp"]}
        cq = {d: frozenset(x for x in mov if d in cm["kq"][x]) for d in cm["Fq"]}
        low = [(c, d) for c in cp for d in cq if len(cp[c] | cq[d]) < target]
        if not low:
            return mov
        if len(mov) >= budget:
            return None
        best = max((sum(1 for c, d in low if c in cm["kp"][x] or d in cm["kq"][x]), rng.random(), x) for x in V if x not in mov)
        if best[0] == 0:
            return None
        mov.append(best[2])


def construct_none(spec, seed, root_id, n_obj, attempts=4) -> dict:
    """NONE with movable objects only (user, 1 Nov 2026): movable blockers until every (pick grasp, place grasp) pair is cut by >=
    NONE_K distinct objects, so K_static >= NONE_K (exact): no repair of <= 4 relocations exists among ANY objects, and the teacher
    (pool = 4 of the blockers) finds none.  The target verifies in the empty scene; no immovable structures.  The 4 blockers cutting
    most grasps are the intended ones, the rest are extra blockers."""
    import p14_sweep_index as sx
    import p6_scenes as p6
    from scene import place_scene, scene_to_dict
    G = A.env()
    cfg, robot = G["cfg"], G["robot"]
    rng = np.random.default_rng(seed)
    lo, hi = A.DISP[spec["displacement"]]
    stratum = "S0" if spec["start_pose"] == "home" else str(rng.choice(["S1", "S2", "S3"]))
    q0 = tuple(p6.STRATA[stratum])
    qa = np.asarray(q0, float)
    lo_d, hi_d = bc.N_DISTRACT[spec["distractors"]]
    budget = n_obj - lo_d
    cands = [(x, g) for x in G["region"][spec["source"]] for g in G["valid"] if lo <= A.cheb(g, x) <= hi
             and capacity(x, "p") + capacity(g, "q") >= NONE_K]        # necessary: the least-cut grasp pair must reach NONE_K
    if not cands:
        raise A.Reject("NO_BLOCKABLE_START_GOAL")
    stats = {"maps": 0, "cover_failed": 0, "k_low": 0, "empty_blocked": 0}
    for attempt in range(attempts):
        src, goal = (int(v) for v in cands[int(rng.integers(len(cands)))])
        s0 = replace(place_scene(cfg, root_id, f"nempty{attempt}", [src], []), robot=replace(place_scene(cfg, root_id, "r", [src], []).robot, q_rad=q0))
        task0 = {"scene": scene_to_dict(s0), "goal_cell": goal, "fixed_ids": [], "target": "t0", "key": f"{root_id}:nempty{attempt}"}
        st, a_empty, _ = A.t_goal(task0, {}, {"t0": src}, qa)
        if st != "VERIFIED":
            stats["empty_blocked"] += 1
            continue
        V = [c for c in G["valid"] if c not in (src, goal)]
        cm = cut_map(src, goal, V)
        stats["maps"] += 1
        kill = {x: {("p", c) for c in cm["kp"][x]} | {("q", d) for d in cm["kq"][x]} for x in V}
        R = pair_fill(cm, V, budget, NONE_K, rng)
        if R is None:
            stats["cover_failed"] += 1
            continue
        k, _ = k_static(cm, R, [])
        if k is None or k < NONE_K:
            stats["k_low"] += 1
            continue
        R.sort(key=lambda x: (-len(kill[x]), x))
        sw_empty = sx.swept_cells(robot, cfg, a_empty)
        seen = set().union(*map(set, sw_empty.values()))
        near_sw = {c for c in G["valid"] for x in seen if A.cheb(c, x) == 1}
        roles = {"near_T": [c for c in G["valid"] if 1 <= A.cheb(c, src) <= 3 and c not in seen],
                 "near_goal": [c for c in G["valid"] if 1 <= A.cheb(c, goal) <= 3 and c not in seen],
                 "near_path": [c for c in near_sw if c not in seen],
                 "far": [c for c in G["valid"] if A.cheb(c, src) >= 3 and A.cheb(c, goal) >= 3 and c not in seen]}
        dpool = sorted(c for c in roles[spec["distractor_role"]] if c in V and c not in R)
        nd = min(int(rng.integers(lo_d, hi_d + 1)), n_obj - len(R), len(dpool))
        if nd < lo_d:
            stats["distractors"] = stats.get("distractors", 0) + 1
            continue
        dis = [int(x) for x in rng.choice(dpool, nd, replace=False)] if nd else []
        s = replace(place_scene(cfg, root_id, "orig", [src], R + dis), robot=replace(s0.robot, q_rad=q0))
        by_cell = {o.cell: o.id for o in s.objects}
        npark = bc.PARKING_N[spec["parking"]]
        occ = R + dis
        alt, taken = {}, []
        for c in occ:                                 # per object (intended first): an object without a free cell stays unparked
            got = _parking(G, src, goal, seen, occ + taken, [c], lambda x: False, n=max(npark, 5))
            if got:
                alt[c] = got[c]
                taken.append(got[c][0])
        parking = {by_cell[c]: [{"cell": int(x), "in_target_sweep": x in seen} for x in alt[c][:npark]] for c in occ if c in alt}
        wrong = sorted((c for c in seen if c in G["valid"] and c not in set(occ + [src]) and c != goal), key=lambda c: (A.cheb(c, src), c))
        intended = [by_cell[c] for c in R[:4]]
        return {"scene": s, "goal": goal, "q0": q0, "stratum": stratum, "fixed_ids": [], "fixed_cells": [],
                "roles": {by_cell[c]: ("intended_blocker" if i < 4 else "extra_blocker") for i, c in enumerate(R)}
                         | {by_cell[c]: f"distractor:{spec['distractor_role']}" for c in dis},
                "parking": parking, "alt_parking": {by_cell[c]: [int(x) for x in alt.get(c, [])] for c in occ}, "wrong_dest": wrong[:2],
                "empty_witness": a_empty, "swept": {kk: list(v) for kk, v in sw_empty.items()}, "intended": intended,
                "pools": {"route": sorted(seen)}, "chase_queries": 0, "blockers_only_scene_blocks_T": True,
                "construction": {"method": "none_cover2d", "attempts": attempt + 1, "K_static": k,
                                 "blockers": len(R), "immovable": 0, "stats": stats}}
    raise A.Reject("NONE_COVER_NOT_FOUND:" + ",".join(f"{kk}={v}" for kk, v in stats.items()))


def _parking(G, src, goal, seen, occ, ids, spoil, n=1):
    """Parking cells: free valid cells >= 2 from start and goal, off the target's sweep, not next to another object, never killing
    the repair pair (else the K movers would not free the target); nearest first.  n = 1: {cell: cell}; else {cell: [cells]}."""
    taken, out = set(), {}
    occupied = set(occ) | {src, goal}
    nearby = {c for x in occ for c in G["valid"] if A.cheb(c, x) == 1}
    for x in ids:
        cands = sorted((c for c in G["valid"] if c not in occupied and c not in taken and A.cheb(c, src) >= 2 and A.cheb(c, goal) >= 2
                        and not spoil(c)), key=lambda c: (c in seen, c in nearby, A.cheb(c, x), c))
        if not cands:
            return None
        pick = [cands[0]] + [next((c for c in cands if A.cheb(c, x) >= w and c != cands[0]), None) for w in (3, 5, 2, 4)]
        pick = list(dict.fromkeys(c for c in pick if c is not None))[:n]
        taken.add(pick[0])
        out[x] = pick[0] if n == 1 else pick
    return out


def _record(spec, G, src, goal, q0, stratum, Q, R, dis, X, park, seen, sw_empty, a_empty, lay, stats, attempt, scene_of, spoil):
    s = scene_of(Q + R + dis + X, "orig")
    by_cell = {o.cell: o.id for o in s.objects}
    occ, npark = Q + R + dis + X, bc.PARKING_N[spec["parking"]]
    first = {**park, **(_parking(G, src, goal, seen, occ + list(park.values()), R + dis, spoil) or {})}
    alt = _parking(G, src, goal, seen, occ, Q + R + dis, spoil, n=5) or {}
    parking, alt_parking = {}, {}
    for c in Q + R + dis:
        if c not in first:                        # no parking cell: not movable in the teacher's repair domain
            continue
        cells = list(dict.fromkeys([first[c]] + [x for x in alt.get(c, []) if x not in first.values()]))
        parking[by_cell[c]] = [{"cell": int(x), "in_target_sweep": x in seen} for x in cells[:npark]]
        alt_parking[by_cell[c]] = [int(x) for x in cells]
    wrong = sorted((c for c in seen if c in G["valid"] and c not in set(occ + [src]) and c != goal), key=lambda c: (A.cheb(c, src), c))
    return {"scene": s, "goal": goal, "q0": q0, "stratum": stratum, "fixed_ids": [by_cell[c] for c in X], "fixed_cells": X,
            "roles": {by_cell[c]: "intended_blocker" for c in Q + R} | {by_cell[c]: f"distractor:{spec['distractor_role']}" for c in dis},
            "parking": parking, "alt_parking": alt_parking, "wrong_dest": wrong[:2], "empty_witness": a_empty,
            "swept": {k: list(v) for k, v in sw_empty.items()}, "intended": [by_cell[c] for c in Q + R], "pools": {"route": sorted(seen)},
            "chase_queries": 0, "blockers_only_scene_blocks_T": True,
            "construction": {"method": "exact_cover2d", "attempts": attempt + 1, "must_move": [by_cell[c] for c in Q],
                             "extra_blockers": [by_cell[c] for c in R], "K_static": lay["K_static"], "unique_repair": lay["unique"],
                             "split_pick_place": lay["split"], "repair_grasps": {"pick": lay["ap"], "place": lay["aq"]},
                             "immovable": len(X), "stats": stats}}
