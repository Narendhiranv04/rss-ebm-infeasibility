"""G4C-P0 preflight: the frozen 20-root 2D A/B/C development data-collection PLAN, its
validator and no-planner coverage tooling (G4C plan-only contract; Methodology v1.7 §3,
§9; Guide v1.9 "Detailed data plan"). Nothing here collects or relabels evidence.

Roots are built from seeded geometry under explicit construction INTENTS (J0..J9 are
construction questions, never labels). Acceptance uses only static geometry: distinct
supported cells, target-goal separation, the intent's cell predicate, unique geometry
(also against archived G4A/G4D roots) and a clear robot at its initial configuration.
No planner, skill search, certificate producer, pose screen or target evaluator is called.

Per root the plan predeclares, before any outcome: the ordered-sequence pool (empty,
singles, same-endpoint reversals, alternative destinations, deliberate static negatives),
single/pair omission remainders, fixed-robot scene edits with geometric eligibility,
robot-start variants, the bounded B/C outcome catalogue and its deterministic masks
(<= 8 active), and teacher units in the G4D runner's unit schema. Budgets stay null.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

import config as cf
import data
import effects as ef
import geometry as ge
import proposals as pr
import sequences as sq
from scene import place_scene, scene_from_dict, scene_hash, scene_to_dict

VERSION, SEED0, MAX_ATTEMPTS, CORRIDOR_CELLS = "2D-ABC-v1", 2026102001, 200, 1.5
FAMILIES = [("J0", 6, 0), ("J1", 2, 1), ("J2", 2, 1), ("J3", 4, 1), ("J4", 2, 1), ("J7", 2, 1), ("J5J6", 1, 0), ("J9", 1, 1)]
SUPPORTS = {   # coverage tags each intent is DESIGNED to exercise (plans, not outcomes); extras are informative
    "J0": ["zero_move", "nuisance_zero_effect"],
    "J1": ["nonempty_witness", "scene_effect", "nonadditive", "omission", "cost", "multi_move_cost", "bc_full_mini_pool"],
    "J2": ["nonempty_witness", "order_contrast", "omission", "scene_effect", "cost", "multi_move_cost", "bc_full_mini_pool"],
    "J3": ["nonempty_witness", "alternative_repairs", "order_contrast", "omission", "scene_effect", "cost", "multi_move_cost",
           "destination_dependency"],
    "J4": ["nonempty_witness", "order_contrast", "omission", "nonadditive", "scene_effect", "cost", "multi_move_cost",
           "three_move"],
    "J7": ["nonempty_witness", "alternative_repairs", "scene_effect", "cost", "cost_inversion_if_measured"],
    "J5J6": ["zero_move", "nuisance_zero_effect", "harmful_edit_if_measured", "robot_start_contrast"],
    "J9": ["nonempty_witness", "bc_full_mini_pool", "order_contrast", "omission", "scene_effect", "alternative_repairs",
           "cost", "multi_move_cost", "nonadditive"]}
TAGS = {"zero_move_roots": "zero_move", "nonempty_witness_roots": "nonempty_witness", "alternative_repair_roots":
        "alternative_repairs", "order_contrast_roots": "order_contrast", "nonzero_scene_effect_roots": "scene_effect",
        "omission_effect_roots": "omission", "zero_effect_control_roots": "nuisance_zero_effect", "cost_label_roots": "cost",
        "multi_move_cost_roots": "multi_move_cost", "bc_full_mini_pool_roots": "bc_full_mini_pool",
        "nonadditive_examples": "nonadditive"}
THRESHOLDS = [("zero_move_roots", 6), ("nonempty_witness_roots", 8), ("alternative_repair_roots", 4),
              ("order_contrast_roots", 4), ("nonzero_scene_effect_roots", 4), ("omission_effect_roots", 4),
              ("zero_effect_control_roots", 4), ("cost_label_roots", 8), ("multi_move_cost_roots", 4),
              ("bc_full_mini_pool_roots", 4), ("nonadditive_examples", 2)]
BUDGET_FIELDS = ("per_teacher_unit_cpu_s", "aggregate_cpu_s", "workers", "unit_wall_guard_s", "aggregate_wall_guard_s",
                 "cancellation_margin_s", "max_motion_nodes", "relocation_skill_attempt_cap", "target_eval_attempt_cap",
                 "max_root_construction_attempts")
FORBIDDEN = ("label", "outcome", "witness", "certificate", "trajectory", "verdict", "teacher", "oracle")
ALT_STARTS = [[math.pi / 2, -math.pi / 2, -math.pi / 2], [math.pi / 4, math.pi / 2, math.pi / 2]]


# ------------------------------------------------------------------ geometric construction (no planning)
def _xy(c, cols=8, pitch=0.08):
    return ((c % cols + 0.5) * pitch, (c // cols + 0.5) * pitch)


def _nbrs(c, cols=8, rows=8):
    r, k = divmod(c, cols)
    return [rr * cols + kk for rr, kk in ((r - 1, k), (r + 1, k), (r, k - 1), (r, k + 1)) if 0 <= rr < rows and 0 <= kk < cols]


def _corr(c, t, g):
    return pr._seg_dist(_xy(c), _xy(t), _xy(g)) / 0.08            # distance to the transport segment, in cells


def _layout(rng, fam: str):
    """(target, goal, participant cells by role, all obstacle cells) or None for one intent."""
    t, g = (int(x) for x in rng.choice(64, 2, replace=False))
    if math.dist(_xy(t), _xy(g)) < 3 * 0.08:
        return None
    nb = [c for c in _nbrs(g) if c != t]
    role = {}
    if fam in ("J1", "J2", "J3", "J4", "J7", "J9"):
        role["A"] = g                                              # goal occupant
    if fam in ("J2", "J4", "J9") and nb:
        k = 1 if fam in ("J2", "J9") else 2
        if len(nb) < k:
            return None
        for name, c in zip("BC", rng.choice(nb, k, replace=False)):
            role[name] = int(c)                                     # access blockers adjacent to the occupant
    if fam == "J1":
        mid = [c for c in range(64) if c not in (t, g) and _corr(c, t, g) < 0.5 and 0.3 < math.dist(_xy(c), _xy(t)) /
               max(1e-9, math.dist(_xy(g), _xy(t))) < 0.7]
        if not mid:
            return None
        role["C"] = int(sorted(mid, key=lambda c: (math.dist(_xy(c), [(a + b) / 2 for a, b in zip(_xy(t), _xy(g))]), c))[0])
    if fam == "J3":
        stage = [c for c in nb if _corr(c, t, g) >= 1.0]
        if not stage:
            return None
        role["C"] = int(stage[int(rng.integers(len(stage)))])        # occupied staging cell P next to the goal
    taken = set(role.values()) | {t, g}
    far = [c for c in range(64) if c not in taken and _corr(c, t, g) >= CORRIDOR_CELLS]
    n_extra = int(rng.integers(1 if fam == "J9" else 3, 2 if fam == "J9" else 6))
    if len(far) < n_extra:
        return None
    extra = [int(c) for c in rng.choice(far, n_extra, replace=False)]
    return t, g, role, list(role.values()) + extra


def _dests(scene, oid, t, g, k=2, exclude=()):
    """Geometric, ID-invariant destination rule: free cells outside the corridor, not the
    target/goal cell, nearest to the object (ties by x, y)."""
    occ = {o.cell for o in scene.objects}
    o = scene.get(oid)
    free = [c for c in range(64) if c not in occ and c not in (t, g) and c not in exclude and _corr(c, t, g) >= CORRIDOR_CELLS]
    return sorted(free, key=lambda c: (math.dist(_xy(c), o.xy_m), _xy(c)))[:k]


def build_roots(cfg, robot, historical: set) -> tuple:
    roots, ledger, seen, seed = [], [], set(historical), SEED0
    for fam, want, _ in FAMILIES:
        got = 0
        for attempt in range(MAX_ATTEMPTS):
            if got == want:
                break
            rng = np.random.default_rng(seed)
            lay, why = _layout(rng, fam), None
            if lay is None:
                why = "intent predicate not met"
            else:
                t, g, role, obs = lay
                s = place_scene(cfg, f"g4c_{seed}", "initial", [t], obs)
                sig = stable_geometry(s, g)
                ok = ge.Checker(robot, ge.make_world(s), cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"]
                                ).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status == ge.CLEAR
                why = "duplicate geometry (incl. archived roots)" if sig in seen else None if ok else "initial robot not clear"
                seen.add(sig)
            ledger.append({"family": fam, "attempt": attempt, "seed": seed, "accepted": why is None, "rejection": why})
            seed += 1
            if why is None:
                ids = {o.cell: o.id for o in s.objects}
                roots.append({"root_id": f"g4c_r{len(roots):02d}", "family": fam, "seed": seed - 1, "scene": s, "goal_cell": g,
                              "target_cell": t, "roles": {k: ids[c] for k, c in role.items()}, "supports": SUPPORTS[fam]})
                got += 1
        if got < want:
            roots += [{"root_id": f"g4c_unavailable_{fam}_{i}", "family": fam, "status": "CONSTRUCTION_UNAVAILABLE"}
                      for i in range(want - got)]
    return roots, ledger


def stable_geometry(scene, goal) -> str:
    return cf.stable_hash([sorted((o.kind, o.xy_m) for o in scene.objects), goal])


# ------------------------------------------------------------------ predeclared queries per root
def _sig(seq):
    return ".".join(f"{o}>{c}" for o, c in seq) or "empty"


def root_queries(cfg, robot, r) -> dict:
    s, t, g, R, fam = r["scene"], r["target_cell"], r["goal_cell"], r["roles"], r["family"]
    act = data.make_query(cfg, r["root_id"], s, "t0", g, {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
    nuis = max((o for o in s.objects if o.kind == "obstacle" and o.id not in R.values()),
               key=lambda o: (_corr(o.cell, t, g), o.xy_m), default=None)
    d, used = {}, set()
    for k in ("A", "B", "C"):                                       # joint rule: later roles avoid earlier roles' cells
        if k in R:
            d[k] = _dests(s, R[k], t, g, 4 if fam == "J7" else 2, () if fam == "J9" else used)
            used |= set(d[k])
    seqs = [[]]
    if "A" in R:
        seqs += [[[R["A"], c]] for c in d["A"][:2]]
    if fam == "J7":
        far = sorted(d["A"], key=lambda c: -math.dist(_xy(c), _xy(g)))[0]
        seqs.append([[R["A"], far]])
    if fam in ("J1", "J2"):
        other = "C" if fam == "J1" else "B"
        pair = [[R[other], d[other][0]], [R["A"], d["A"][0]]]
        seqs += [pair, pair[::-1]]                                  # same endpoints, both orders
    if fam == "J3":
        P = s.get(R["C"]).cell
        seqs += [[[R["A"], P]], [[R["C"], d["C"][0]], [R["A"], P]], [[R["A"], P], [R["C"], d["C"][0]]]]
    if fam == "J4":
        tri = [[R["B"], d["B"][0]], [R["C"], d["C"][0]], [R["A"], d["A"][0]]]
        seqs += [tri, [tri[1], tri[0], tri[2]], [tri[2], tri[0], tri[1]], [tri[0], tri[2]], [tri[1], tri[2]]]
    if fam == "J9":
        seqs = [[list(a) for a in x] for x in _j9_space(R, d)]
    pool, seen = [], set()
    for x in seqs:
        seq, why = sq.parse_sequence(s, act, x, 3)
        if why or seq in seen:
            continue
        seen.add(seq)
        pool.append({"sequence": [list(a) for a in seq], "origin": "declared", "moved": len(seq),
                     "static_status": pr.static_endpoints(s, sq.assignment(seq), "t0", cfg)["status"],
                     "endpoint_distance": round(sq.endpoint_distance(s, seq), 9)})
    for p in [p for p in pool if p["moved"] >= 2]:                  # omission remainders (single and pair omissions)
        seq = tuple(tuple(a) for a in p["sequence"])
        for k in (1, 2):
            for drop in itertools.combinations(range(len(seq)), k):
                rem = sq.omit(seq, drop)
                if rem not in seen:
                    seen.add(rem)
                    pool.append({"sequence": [list(a) for a in rem], "origin": f"omission_of:{_sig(seq)}", "moved": len(rem),
                                 "static_status": pr.static_endpoints(s, sq.assignment(rem), "t0", cfg)["status"],
                                 "endpoint_distance": round(sq.endpoint_distance(s, rem), 9)})
    edits = [({R[k]: d[k][0]}, f"singleton:{k}") for k in ("A", "B", "C") if k in R and d[k]]
    if len(R) >= 2:
        k1, k2 = [k for k in ("A", "B", "C") if k in R][:2]
        edits.append(({R[k1]: d[k1][0], R[k2]: d[k2][0]}, f"joint:{k1}+{k2}"))
    if nuis is not None:
        nd = _dests(s, nuis.id, t, g, 1)
        edits += [({nuis.id: nd[0]}, "nuisance")] if nd else []
        if fam == "J5J6":
            mid = sorted((c for c in range(64) if c not in {o.cell for o in s.objects} | {t, g} and _corr(c, t, g) < 0.5),
                         key=lambda c: (abs(math.dist(_xy(c), _xy(t)) - math.dist(_xy(c), _xy(g))), c))
            edits += [({nuis.id: mid[0]}, "harmful_candidate:on_corridor")] if mid else []
    scene_q = []
    for e, kind in edits:
        sub = ef.pose_substitution(robot, cfg, s, act, e)
        scene_q.append({"edit": e, "kind": kind, "eligible": sub["eligible"], "reasons": sub["reasons"],
                        "intervened_scene_hash": sub["intervened_scene_hash"], "changed_fields": sub["changed_fields"]})
    starts = []
    if fam == "J5J6":
        for j, q in enumerate(ALT_STARTS):
            rs = ef.robot_start_substitution(robot, cfg, s, act, q)
            starts.append({"q_start": rs["q_to"], "eligible": rs["eligible"], "intervened_scene_hash": rs["intervened_scene_hash"],
                           "rule_index": j})
    cat = [q for q in scene_q if q["eligible"]][:3]                 # bounded B/C catalogue: first 3 eligible, rule order
    return {"pool": pool, "scene_queries": scene_q, "robot_starts": starts, "catalogue": {
        "version": VERSION, "rule": "first <=3 geometrically eligible scene edits in declared rule order",
        "queries": [q["kind"] for q in cat], "hash": cf.stable_hash([[q["kind"], q["edit"]] for q in cat])}, "action": act}


def _j9_space(R, d) -> list:
    """Tiny exhaustible domain: A and B with 2 destinations each, cap 2 (13 sequences)."""
    dom = {R["A"]: d["A"][:2], R["B"]: d["B"][:2]}
    out = [()]
    for m in (1, 2):
        for objs in itertools.permutations(sorted(dom), m):
            for ds in itertools.product(*[dom[o] for o in objs]):
                out.append(tuple(zip(objs, ds)))
    return out


def layout(seq, catalogue: list) -> dict:
    """Deterministic B/C mask: z0, catalogue scene slots, zR for m >= 1, zO[t] for m >= 2."""
    m = len(seq)
    active = {"z0": True, **{f"zS[{q}]": True for q in catalogue}, "zR": m >= 1, **{f"zO[{t}]": m >= 2 for t in range(3)}}
    return {"slots": list(active), "active": [bool(v) for v in active.values()], "inactive_value": 0,
            "active_count": sum(active.values())}


# ------------------------------------------------------------------ plan, units and budget request
def make_plan(cfg, v2, robot, root_dir: Path) -> dict:
    hist = {stable_geometry(scene_from_dict(q["scene"]), q["goal_cell"])
            for q in json.loads((root_dir / "runs/g4a_evidence_audit/g4a_plan.json").read_text())["queries"]}
    hist |= {stable_geometry(scene_from_dict(r["scene"]), r["goal_cell"])
             for r in json.loads((root_dir / "runs/g4d_evidence/final_plan.json").read_text())["roots"].values()}
    roots, ledger = build_roots(cfg, robot, hist)
    proots, units = {}, []
    for r in roots:
        if "scene" not in r:
            proots[r["root_id"]] = {k: v for k, v in r.items()}
            continue
        Q = root_queries(cfg, robot, r)
        q = data.make_query(cfg, r["root_id"], r["scene"], "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
        k = r["root_id"]
        for p in Q["pool"]:
            p["bc_layout"] = layout(p["sequence"], Q["catalogue"]["queries"])
            units.append({"unit": f"seq:{k}:{_sig(p['sequence'])}", "kind": "ORDERED_TEACHER", "root": k, "sequence": p["sequence"],
                          "relocation_key": f"g4c/{k}/{_sig(p['sequence'])}/reloc", "target_key": f"g4c/{k}/{_sig(p['sequence'])}/target",
                          "relocation_skills_max": len(p["sequence"]), "target_evaluations_max": 1})
        for e in Q["scene_queries"]:
            if e["eligible"]:
                sig = _sig(sorted(e["edit"].items()))
                units.append({"unit": f"scene:{k}:{sig}", "kind": "SCENE_TARGET_ONLY", "root": k, "edit": e["edit"],
                              "eligible": True, "intervened_scene_hash": e["intervened_scene_hash"], "reference_unit": f"seq:{k}:empty",
                              "target_key": f"g4c/{k}/scene/{sig}/target", "relocation_skills_max": 0, "target_evaluations_max": 1})
        for st in [x for x in Q["robot_starts"] if x["eligible"]][:1]:
            units.append({"unit": f"start:{k}:{st['rule_index']}", "kind": "ROBOT_START_TARGET_ONLY", "root": k,
                          "q_start": st["q_start"], "intervened_scene_hash": st["intervened_scene_hash"],
                          "reference_unit": f"seq:{k}:empty", "target_key": f"g4c/{k}/start{st['rule_index']}/target",
                          "relocation_skills_max": 0, "target_evaluations_max": 1})
        proots[k] = {"family": r["family"], "seed": r["seed"], "lineage": {"family_id": k, "split": "DEVELOPMENT"},
                     "scene": scene_to_dict(r["scene"]), "scene_hash": scene_hash(r["scene"]), "hashes": q["hashes"],
                     "goal_cell": r["goal_cell"], "roles": r["roles"], "supports": r["supports"], "queries": {
                         kk: v for kk, v in Q.items() if kk != "action"},
                     "ceilings": {"sequences": 14, "scene_edits": 5, "robot_starts": 1, "omission_remainders": 8},
                     "bc_mini_pool": {"sequences": [p["sequence"] for p in Q["pool"]], "catalogue_hash": Q["catalogue"]["hash"],
                                      "required_units": sorted(u["unit"] for u in units if u["root"] == k and u["kind"] != "ROBOT_START_TARGET_ONLY"),
                                      "full_decision_rule": "every active slot of every pool sequence decided (z0, catalogue zS, zR, zO)"}}
    plan = {"stage": "G4C_PREFLIGHT", "dataset_version": VERSION, "protocol": data.PROTOCOL_V2,
            "status": "FROZEN_PENDING_BUDGET_AUTHORIZATION", "families": FAMILIES, "roots": proots, "ledger": ledger,
            "units": units, "thresholds": dict(THRESHOLDS), "budget": {f: None for f in BUDGET_FIELDS},
            "unit_upper_bounds": {"units": len(units), "relocation_skills": sum(u["relocation_skills_max"] for u in units),
                                  "target_evaluations": sum(u["target_evaluations_max"] for u in units)},
            "config_hash": cf.config_hash(cfg), "config_v2_hash": cf.stable_hash(v2), "robot_profile_hash": cf.robot_profile_hash(cfg),
            "policies": {"execution": "INITIAL_SCENE_ORDERED_ONE_SHOT_V2 via g4d_run unit schema", "relocation_backend":
                         "FIXED_ORDER_GREEDY_SKILL2D_V1", "cost": v2["cost"]["policy"], "target_end": "GOAL_AND_SAFE_RETREAT",
                         "target_when_certified": "NOT_RUN", "unknowns": "never negative", "construction_cap_per_family": MAX_ATTEMPTS,
                         "max_active_z": 8, "no_root_replacement": True}}
    plan["pilot"] = pilot(plan)
    plan["budget_request"] = budget_request(plan, root_dir)
    plan["plan_hash"] = cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    return plan


def pilot(plan: dict) -> dict:
    """Fixed rule: the first constructed root of each family exercising order, destination,
    higher-order and exact-pool questions, plus the first zero-move control."""
    first = {}
    for k, v in plan["roots"].items():
        first.setdefault(v["family"], k)
    roots = [first[f] for f in ("J0", "J2", "J3", "J4", "J9") if f in first]
    return {"roots": roots, "units": sum(u["root"] in roots for u in plan["units"]),
            "rule": "first root of J0, J2, J3, J4, J9; pilot roots stay inside the 20-root development set; "
                    "pilot outcomes never retune or replace other roots"}


def budget_request(plan: dict, root_dir: Path) -> dict:
    """Proposed numbers for authorization with their measured basis; NOT authorized values."""
    led = json.loads((root_dir / "runs/g4d_evidence/ledger.json").read_text())["units"]
    g4d = sorted(u["cpu_s"] for u in led.values())
    prim = json.loads((root_dir / "runs/g4a_evidence_audit/primary_episodes.json").read_text())
    rows = json.loads((root_dir / "runs/g4a_evidence_audit/candidate_labels.json").read_text())
    tb = sum((r["unknown_reason"] or "").startswith("TARGET_UNRESOLVED:PLANNER_BUDGET") for r in rows) / len(rows)
    up = lambda x: int(math.ceil(x / 1800.0) * 1800)  # noqa: E731
    out = {"status": "REQUESTED_NOT_AUTHORIZED", "basis": {
        "g4d_unit_cpu_s": {"n": len(g4d), "median": g4d[len(g4d) // 2], "mean": round(sum(g4d) / len(g4d), 2), "max": g4d[-1]},
        "g4a_query_cpu_s_median": sorted(p["cpu_s"] for p in prim.values())[len(prim) // 2],
        "g4a_target_budget_reason_share": round(tb, 3), "followup_censored_eval_cpu_s": 870.0,
        "note": "G4A rows carry raw PLANNER_BUDGET strings (addendum A2): an upper-side proxy for slow target searches"}}
    for name, units in (("pilot", [u for u in plan["units"] if u["root"] in plan["pilot"]["roots"]]), ("full", plan["units"])):
        n, tev = len(units), sum(u["target_evaluations_max"] for u in units)
        low, central = n * g4d[len(g4d) // 2], n * sum(g4d) / len(g4d)
        high = tb * tev * 870.0 + (n - tb * tev) * g4d[-1]
        out[name] = {"units": n, "relocation_skills_max": sum(u["relocation_skills_max"] for u in units),
                     "target_evaluations_max": tev, "estimate_cpu_s": {"low": round(low), "central": round(central),
                                                                       "high": round(high)},
                     "requested": {"per_teacher_unit_cpu_s": 900, "cancellation_margin_s": 30, "aggregate_cpu_s": up(high * 1.1),
                                   "workers": 1, "unit_wall_guard_s": 1800, "aggregate_wall_guard_s": up(high * 1.1 * 1.25),
                                   "max_motion_nodes": 4000, "relocation_skill_attempt_cap": sum(u["relocation_skills_max"] for u in units),
                                   "target_eval_attempt_cap": tev, "max_root_construction_attempts": 0}}
    out["notes"] = ["max_motion_nodes 4000 is the unchanged planner profile (semantic consistency), not new effort",
                    "roots are already constructed geometrically in this plan; no construction attempt at collection time",
                    "one worker: the tested guard/runner (g4d_run.run_collection) is single-worker",
                    "aggregate = high estimate x 1.1 rounded up to 30 min of CPU; wall = aggregate x 1.25"]
    return out


# ------------------------------------------------------------------ validator (fail closed) and coverage (no planner)
def validate(plan: dict, cfg=None) -> list:
    why = []
    if cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"}) != plan.get("plan_hash"):
        why.append("plan hash mismatch")
    roots = {k: v for k, v in plan["roots"].items() if "scene" in v}
    fams = {}
    for v in plan["roots"].values():
        fams[v["family"]] = fams.get(v["family"], 0) + 1
    if fams != {f: n for f, n, _ in FAMILIES} or len(plan["roots"]) != 20:
        why.append(f"construction-intent counts {fams} differ from the declared mixture")
    blob = json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("scene", "hashes", "roles")} for k, v in roots.items()})
    why += [f"forbidden learner field '{w}'" for w in FORBIDDEN if f'"{w}' in blob]
    if len({v["seed"] for v in roots.values()}) != len(roots):
        why.append("duplicated root seed")
    try:
        geo = [stable_geometry(scene_from_dict(v["scene"]), v["goal_cell"]) for v in roots.values()]
    except (TypeError, KeyError, ValueError) as e:                  # fail closed on malformed scene records
        return why + [f"malformed scene record: {e}"]
    if len(set(geo)) != len(geo):
        why.append("duplicated root geometry")
    if len({v["lineage"]["family_id"] for v in roots.values()}) != len(roots) or any(
            v["lineage"]["family_id"] != k for k, v in roots.items()):
        why.append("mixed lineage across roots")
    for u in plan["units"]:
        if u["root"] not in roots:
            why.append(f"{u['unit']}: unit references an unknown root")
    if len({u["unit"] for u in plan["units"]}) != len(plan["units"]):
        why.append("duplicated unit")
    for k, v in roots.items():
        Q = v["queries"]
        cat = [q for q in Q["scene_queries"] if q["eligible"]][:3]
        if [q["kind"] for q in cat] != Q["catalogue"]["queries"] or cf.stable_hash([[q["kind"], q["edit"]] for q in cat]) \
                != Q["catalogue"]["hash"]:
            why.append(f"{k}: catalogue changed after freezing")
        for p in Q["pool"]:
            L, m = p["bc_layout"], len(p["sequence"])
            exp = layout(p["sequence"], Q["catalogue"]["queries"])
            if L != exp or L["active_count"] > 8 or (m == 0 and L["active"][L["slots"].index("zR")]):
                why.append(f"{k}:{_sig(p['sequence'])}: malformed outcome mask")
            if len({o for o, _ in p["sequence"]}) != m:
                why.append(f"{k}: non-canonical sequence")
        for q in Q["scene_queries"]:
            cf_ = q["changed_fields"]
            if not cf_["robot_unchanged"] or cf_["other_fields_changed"] or not set(cf_["changed_objects"]) <= set(q["edit"]):
                why.append(f"{k}: scene edit {q['kind']} changes more than its listed objects")
    if any(plan["budget"].get(f) is not None for f in BUDGET_FIELDS) and plan["status"] == "FROZEN_PENDING_BUDGET_AUTHORIZATION":
        why.append("budget values present without an authorization record")
    return why


def admit(plan: dict) -> tuple:
    """Fail-closed collection admission: every budget field must be numeric and authorized."""
    missing = [f for f in BUDGET_FIELDS if not isinstance(plan["budget"].get(f), (int, float))]
    if missing or plan.get("authorization") is None:
        return False, f"NOT_ADMITTED: budget fields unset {missing}" if missing else "NOT_ADMITTED: no authorization record"
    return True, "ADMITTED"


def coverage(plan: dict, evidence: dict | None = None) -> list:
    """Threshold rows: planned supporting roots always; measured status only from evidence
    (root_id -> set of decided coverage tags). Without evidence every measured value is NOT_RUN."""
    rows = []
    for name, need in THRESHOLDS:
        planned = sorted(k for k, v in plan["roots"].items() if TAGS[name] in v.get("supports", []))
        meas = None if evidence is None else sorted(k for k, tags in evidence.items() if TAGS[name] in tags)
        status = "NOT_RUN" if meas is None else ("MET" if len(meas) >= need else "NOT_MET")
        rows.append({"threshold": name, "required": need, "planned_roots": planned, "planned_count": len(planned),
                     "measured_roots": meas, "status": status,
                     "plan_capacity": "SUFFICIENT_IF_ALL_DECIDE" if len(planned) >= need else "INSUFFICIENT_EVEN_IF_ALL_DECIDE"})
    return rows


def _sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def cmd_g4c_plan(args) -> int:
    import g4c_figures
    return g4c_figures.run(args)
