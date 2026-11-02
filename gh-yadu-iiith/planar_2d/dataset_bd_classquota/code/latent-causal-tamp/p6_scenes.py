"""P6.1: frozen construction of 16 fresh roots (4 families x 4 batches) and their <= 24-query catalogues.

Static and outcome-blind: no teacher, planner, skill search or target evaluation. Families are built
from the local GRASP-ACCESS mechanism of the trusted pose-only screens (skill.screen_pick and
skill.screen_place) at the root's fixed start: a disc whose N, S and W cells are occupied has no
screen-passing grasp, and a goal whose N, S and W cells are occupied has no screen-passing place.
Screen counts are construction INTENT predicates recorded with the plan, never labels; the teacher
(2D-ABC-OPNEG-v1, unchanged) decides every query.
  F1 access order:     B -> A -> target (A's grasp needs B gone; the target needs A and B gone).
  F2 dual blockers:    A closes the target's grasp side, B the goal's place side (kappa intent +1).
  F3 3/4-step chain:   C -> B -> A on the grasp side plus D on the place side (main repair: 4 moves).
  F4 alternatives:     A (grasp side) + goal occupant G; A's alternative destination closes the goal's
                       place side (destination dependency); nuisance object and a non-home start.
Every object, goal and destination lies outside the link-0 annulus (rho >= R0 + 0.01 m), so no
relocation can create a joint-0 sector barrier. Start strata are predeclared per root and must be
clear; attempts are seeded and every rejection is logged; no root is replaced after any outcome.
"""
from __future__ import annotations

import math
import time
from dataclasses import replace

import numpy as np

import config as cf
import data
import effects as ef
import geometry as ge
import motion as mo
import sector_check as sc
import sequences as sq
import skill as sk
import zlayout_r4 as z4
from scene import place_scene, scene_hash, scene_to_dict

VERSION = "P6_FAMILIES_V1"
FAMS = ["F1", "F2", "F3", "F4"]
SPLITS = {"F1": ["train", "validation", "train", "holdout"], "F2": ["train", "holdout", "train", "validation"],
          "F3": ["validation", "train", "holdout", "train"], "F4": ["holdout", "train", "validation", "train"]}
STRATA = {"S0": [math.pi / 2, math.pi / 2, math.pi / 2], "S1": [math.pi / 2, -math.pi / 2, -math.pi / 2],
          "S2": [math.pi / 4, math.pi / 2, math.pi / 2], "S3": [3 * math.pi / 4, -math.pi / 2, -math.pi / 2]}
SEED0, MAX_ATTEMPTS, CONSTRUCTION_CPU_S = 2026102301, 400, 300.0
ALT = {"F1": "A", "F2": "B", "F3": "D", "F4": "G"}
ROOTS = [{"index": 4 * b + f, "batch": b + 1, "family": FAMS[f], "split": SPLITS[FAMS[f]][b], "seed": SEED0 + 4 * b + f,
          "stratum": f"S{(f + b) % 4}"} for b in range(4) for f in range(4)]
CATALOGUE = {   # role tokens; "X*" = X to its alternative destination; omissions of the main repair are added and deduplicated
    "F1": {"main": ["B", "A"], "seqs": [("single", ["B"]), ("single", ["A"]), ("swap", ["A", "B"]), ("triple", ["B", "A", "N"]),
                                        ("triple", ["N", "B", "A"]), ("single", ["N"]), ("alternative", ["B", "A*"])],
           "edits": {"A": ["A"], "B": ["B"], "A+B": ["A", "B"]}},
    "F2": {"main": ["A", "B"], "seqs": [("single", ["A"]), ("single", ["B"]), ("swap", ["B", "A"]), ("triple", ["A", "B", "N"]),
                                        ("triple", ["A", "N", "B"]), ("single", ["N"]), ("alternative", ["A", "B*"])],
           "edits": {"A": ["A"], "B": ["B"], "A+B": ["A", "B"]}},
    "F3": {"main": ["C", "B", "A", "D"], "seqs": [("single", ["C"]), ("single", ["D"]), ("single", ["B"]), ("single", ["A"]),
                                                  ("pair", ["C", "B"]), ("pair", ["B", "C"]), ("pair", ["C", "D"]),
                                                  ("quad", ["B", "C", "A", "D"]), ("quad", ["D", "C", "B", "A"]),
                                                  ("alternative", ["C", "B", "A", "D*"])],
           "edits": {"A": ["C", "B", "A"], "B": ["D"], "A+B": ["C", "B", "A", "D"]}},
    "F4": {"main": ["A", "G"], "seqs": [("single", ["A"]), ("single", ["G"]), ("swap", ["G", "A"]), ("pair", ["A*", "G"]),
                                        ("pair", ["G", "A*"]), ("triple", ["G", "A", "N"]), ("alternative", ["A", "G*"])],
           "edits": {"A": ["A"], "B": ["G"], "A+B": ["A", "G"]}}}


def _cell(r, c):
    return 8 * r + c if 0 <= r < 8 and 0 <= c < 8 else None


def _xy(c):
    return ((c % 8 + 0.5) * 0.08, (c // 8 + 0.5) * 0.08)


def _cheb(a, b):
    return max(abs(a // 8 - b // 8), abs(a % 8 - b % 8))


class Screens:
    """Pose-only pick / place screen counts at the fixed start (static; CPU and calls counted)."""

    def __init__(self, cfg, robot):
        self.cfg, self.robot, self.p, self.calls = cfg, robot, sk.params(cfg), 0
        k = sc.constants(robot, cfg)
        self.r0 = k["L1"] + max(k["w0_half"], k["housing_r"]) + cfg["suite"]["obstacle_radius_m"] + k["delta"] - k["eps"]
        self.valid = {c for c in range(64) if math.dist(_xy(c), robot.base) >= self.r0 + 0.01}

    def pick(self, s, oid):
        self.calls += 1
        o = s.get(oid)
        ck = mo.checkers(self.robot, s, oid, None, None, self.p["delta"], self.p["eps"])
        return [c for c in sk.screen_pick(self.robot, ck, o.xy_m, o.radius_m, self.p) if c.stage == "OK"]

    def place(self, s, oid, cell) -> int:
        xy, n = s.lattice.center(cell), 0
        for c in self.pick(s, oid)[:4]:
            ck = mo.checkers(self.robot, s, oid, xy, c.local, self.p["delta"], self.p["eps"])
            n += sum(x.stage == "OK" for x in sk.screen_place(self.robot, ck, xy, c.local, c.gap, self.p))
        return n


def moved(s, moves: dict):
    """Static pose substitution {object id: cell} on a clone (robot unchanged)."""
    return replace(s, objects=tuple(replace(o, xy_m=s.lattice.center(moves[o.id]), cell=moves[o.id]) if o.id in moves else o
                                    for o in s.objects))


def _blocks(rng, fam):
    """(role cells, fixed cells, goal) for one family intent, or None. N/S/W = row +1 / row -1 / column -1.
    Grasp-side and place-side cells must be >= 2 cells apart (Chebyshev) so the two mechanisms stay separate."""
    r, c = int(rng.integers(1, 7)), int(rng.integers(6 if fam == "F1" else 5, 8)) if fam != "F3" else 7
    rg, cg = int(rng.integers(1, 7)), int(rng.integers(5, 8))
    chain = {"F1": "AB", "F3": "ABC"}.get(fam, "A")
    role = {"T": _cell(r, c), **{k: _cell(r, c - 1 - i) for i, k in enumerate(chain)}}
    pick_fixed = [_cell(r + d, c - i) for i in range(len(chain)) for d in (1, -1)]
    place = {"F2": {"B": _cell(rg, cg - 1)}, "F3": {"D": _cell(rg, cg - 1)}, "F4": {"G": _cell(rg, cg)}}.get(fam, {})
    place_fixed = [_cell(rg + 1, cg), _cell(rg - 1, cg)] if place else []
    goal = _cell(rg, cg)
    role.update(place)
    cells = list(role.values()) + pick_fixed + place_fixed
    if None in cells or len(set(cells)) != len(cells) or (goal in cells and fam != "F4") or math.dist(_xy(goal), _xy(role["T"])) < 3 * 0.08:
        return None
    pick_side = [role[k] for k in "T" + chain] + pick_fixed
    if any(_cheb(a, b) < 2 for a in pick_side for b in [goal] + list(place.values()) + place_fixed):
        return None
    return role, pick_fixed + place_fixed, goal


def _dests(scr, s, oid, far, taken, k):
    """ID-invariant rule: free annulus-safe cells, not taken, >= 2 cells from `far` (target, goal), nearest to the object."""
    occ = {o.cell for o in s.objects} | set(taken)
    free = [c for c in sorted(scr.valid) if c not in occ and all(_cheb(c, a) >= 2 for a in far)]
    return sorted(free, key=lambda c: (math.dist(_xy(c), s.get(oid).xy_m), _xy(c)))[:k]


def _try(cfg, robot, scr, spec, attempt, known):
    rng = np.random.default_rng([spec["seed"], attempt])
    blk = _blocks(rng, spec["family"])
    if blk is None:
        return None, "family block geometry not met"
    role, fixed, goal = blk
    if any(c not in scr.valid for c in list(role.values()) + fixed + [goal]):
        return None, "object or goal inside the link-0 annulus"
    movers = [k for k in role if k != "T"]
    key = f"p6_{spec['seed']}"
    s = place_scene(cfg, key, "obstructed", [role["T"]], [role[k] for k in movers] + fixed)
    s = replace(s, robot=replace(s.robot, q_rad=tuple(STRATA[spec["stratum"]])))
    ids = {k: f"o{i:02d}" for i, k in enumerate(movers)}
    d, e = cfg["suite"]["margin_m"], cfg["geometry"]["numeric_margin_band_m"]
    if ge.Checker(robot, ge.make_world(s), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status != ge.CLEAR:
        return None, f"start stratum {spec['stratum']} not clear"
    import g4c
    if g4c.stable_geometry(s, goal) in known or ("tg", role["T"], goal) in known:
        return None, "duplicate of a historical geometry or of an earlier P6 (target, goal) pair"
    dest = {}
    for k in movers:                                                 # one destination each; the family's ALT role keeps two
        dd = _dests(scr, s, ids[k], [role["T"], goal], [x for v in dest.values() for x in v], 2 if k == ALT[spec["family"]] else 1)
        if len(dd) < (2 if k == ALT[spec["family"]] else 1):
            return None, f"destinations for {k} unavailable"
        dest[k] = dd
    used = {o.cell for o in s.objects} | {goal} | {x for v in dest.values() for x in v}
    nu = [c for c in sorted(scr.valid) if all(_cheb(c, u) >= 2 for u in used)]
    if nu:                                                           # optional nuisance object, >= 2 cells from everything
        ncell = nu[int(rng.integers(len(nu)))]
        nd = [c for c in sorted(scr.valid) if c not in used | {ncell} and _cheb(c, role["T"]) >= 2 and _cheb(c, goal) >= 2]
        if nd:
            s = place_scene(cfg, key, "obstructed", [role["T"]], [role[k] for k in movers] + fixed + [ncell])
            s = replace(s, robot=replace(s.robot, q_rad=tuple(STRATA[spec["stratum"]])))
            ids["N"] = f"o{len(movers) + len(fixed):02d}"
            dest["N"] = sorted(nd, key=lambda c: (math.dist(_xy(c), _xy(ncell)), _xy(c)))[:1]
    if spec["family"] == "F4":                                       # A's alternative destination: the goal's W cell
        dest["A"] = [dest["A"][0], goal - 1]
    ok, pred = _predicates(scr, s, spec["family"], ids, dest, goal)
    if not ok:
        return None, f"static intent predicate failed: {pred[-1]}"
    ref = place_scene(cfg, f"{key}_ref", "reference", [role["T"]], fixed + ([s.get(ids['N']).cell] if "N" in ids else []))
    ref = replace(ref, robot=s.robot)
    return {**spec, "attempt": attempt, "root": key, "scene": s, "ref_scene": ref, "target_cell": role["T"], "goal_cell": goal,
            "ids": ids, "dest": dest, "role_cells": role, "fixed": fixed, "predicates": pred}, None


def _predicates(scr, s, fam, ids, dest, goal):
    """Construction intent at the fixed start, in order; stops at the first failure."""
    main = CATALOGUE[fam]["main"]
    pred, mv = [], {}
    chk = lambda name, v, want: pred.append({"check": name, "value": v, "want": want, "ok": (v > 0) == want})  # noqa: E731
    chk("target grasp at S0", len(scr.pick(s, "t0")), False)
    for k in main:                                                  # each mover is graspable when its turn comes
        if not pred[-1]["ok"]:
            return False, pred
        chk(f"{k} grasp after {list(mv)}", len(scr.pick(moved(s, mv), ids[k])), True)
        mv[ids[k]] = dest[k][0]
    if not pred[-1]["ok"]:
        return False, pred
    end = moved(s, mv)
    chk("target grasp after main", len(scr.pick(end, "t0")), True)
    if pred[-1]["ok"]:
        chk("target place at goal after main", scr.place(end, "t0", goal), True)
    if fam in ("F1", "F3") and pred[-1]["ok"]:                      # order intent: the second mover is enclosed at S0
        chk(f"{main[1]} grasp at S0", len(scr.pick(s, ids[main[1]])), False)
    if fam in ("F2", "F3") and pred[-1]["ok"]:                      # place-side intent: pick side cleared, place side closed
        side = {ids[k]: dest[k][0] for k in main if k not in ("B" if fam == "F2" else "D")}
        chk("target place with only the grasp side cleared", scr.place(moved(s, side), "t0", goal), False)
    if fam == "F4" and pred[-1]["ok"]:
        chk("target place after A* then G", scr.place(moved(s, {ids["A"]: dest["A"][1], ids["G"]: dest["G"][0]}), "t0", goal), False)
    return all(p["ok"] for p in pred), pred


def construct(cfg, robot, known: set) -> tuple:
    scr, roots, ledger, c0 = Screens(cfg, robot), [], [], time.process_time()
    for spec in ROOTS:
        for attempt in range(MAX_ATTEMPTS):
            if time.process_time() - c0 > CONSTRUCTION_CPU_S:
                raise SystemExit(f"construction exceeded {CONSTRUCTION_CPU_S} CPU-s (STOP before any teacher call)")
            rec, why = _try(cfg, robot, scr, spec, attempt, known)
            ledger.append({"index": spec["index"], "seed": spec["seed"], "attempt": attempt, "accepted": rec is not None, "reason": why})
            if rec:
                import g4c
                known |= {g4c.stable_geometry(rec["scene"], rec["goal_cell"]), ("tg", rec["target_cell"], rec["goal_cell"])}
                roots.append(rec)
                break
        else:
            raise SystemExit(f"root {spec['index']}: no admissible layout in {MAX_ATTEMPTS} attempts (STOP before any teacher call)")
    return roots, ledger, {"cpu_s": round(time.process_time() - c0, 2), "screen_calls": scr.calls, "annulus_r0_m": scr.r0}


def _seq(r, toks):
    return [(r["ids"][t.rstrip("*")], r["dest"][t.rstrip("*")][1 if t.endswith("*") else 0]) for t in toks]


def catalogue(cfg, robot, r) -> dict:
    """The frozen <= 24-query catalogue: reference root + empty + sequences + omissions + A/B/A+B edits + nuisance."""
    spec, s, k = CATALOGUE[r["family"]], r["scene"], r["root"]
    act = data.make_query(cfg, k, s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
    edits = [(kind, {r["ids"][t]: r["dest"][t][0] for t in toks}) for kind, toks in spec["edits"].items()]
    if "N" in r["ids"]:
        edits.append(("nuisance", {r["ids"]["N"]: r["dest"]["N"][0]}))
    scene_q = []
    for kind, ed in edits:
        sub = ef.pose_substitution(robot, cfg, s, act, ed)
        scene_q.append({"kind": kind, "edit": ed, "eligible": bool(sub["eligible"]), "intervened_scene_hash": sub["intervened_scene_hash"]})
    main = _seq(r, spec["main"])
    rows = [("original", []), ("main", main)] + [(rl, _seq(r, t)) for rl, t in spec["seqs"] if all(x.rstrip("*") in r["ids"] for x in t)]
    rows += [("omission", [a for i, a in enumerate(main) if i != t]) for t in range(len(main))]
    room = 24 - 1 - len(scene_q)                                     # reference root + fixed-robot edits share the cap
    for _, seq in list(rows):                                        # omission closure, catalogue order, until the cap
        rows += [("closure", [a for i, a in enumerate(seq) if i != t]) for t in range(len(seq))] if len(seq) >= 2 else []
    pool, seen = [], set()
    for rl, seq in rows:
        if z4.sig(seq) in seen or len(pool) >= room:
            continue
        seen.add(z4.sig(seq))
        canon, why = sq.parse_sequence(s, act, [list(x) for x in seq], z4.MOVE_CAP)
        lay = z4.build(k, seq, [q for q in scene_q if q["kind"] != "nuisance"])
        pool.append({"sequence": [list(x) for x in seq], "role": rl, "canonical": not why, "parse": why, "z_layout": lay})
    return {"scene_queries": scene_q, "pool": pool}


def plan(cfg, robot, pol, roots, cat_fn=None, meta=None) -> dict:
    """Frozen plan. `cat_fn` / `meta` (stage, version, root_schedule, strata) let P7 reuse this builder; the defaults
    reproduce the P6 plan (and its hash) exactly."""
    out = {"stage": "P6", "version": VERSION, "policy_id": pol["policy"]["policy_id"], "policy_hash": pol["policy_hash"],
           "seeds_policy": pol["policy"]["seeds"], "root_schedule": ROOTS, "strata": STRATA, "roots": {}, "queries": [], "units": [],
           "ineligible": [], "z_layout_version": z4.VERSION, "config_hash": cf.config_hash(cfg), "robot_profile_hash": cf.robot_profile_hash(cfg)}
    out.update(meta or {})
    for r in roots:
        cat = (cat_fn or catalogue)(cfg, robot, r)
        meta = {k: r[k] for k in ("family", "split", "batch", "seed", "attempt", "stratum", "target_cell", "goal_cell", "ids", "dest",
                                  "role_cells", "fixed", "predicates")}
        for key, s, variant in ((r["root"] + "_ref", r["ref_scene"], "reference"), (r["root"], r["scene"], "obstructed")):
            q = data.make_query(cfg, key, s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
            out["roots"][key] = {"scene": scene_to_dict(s), "scene_hash": scene_hash(s), "hashes": q["hashes"], "lineage": r["root"],
                                 "variant": variant, **meta, "queries": cat if variant == "obstructed" else None}
        todo = [(r["root"] + "_ref", [], None, "reference")] + [(r["root"], p["sequence"], None, p["role"]) for p in cat["pool"]]
        todo += [(r["root"], [], q, q["kind"]) for q in cat["scene_queries"]]
        for key, seq, sqr, role in todo:
            if sqr is not None and not sqr["eligible"]:
                out["ineligible"].append({"root": key, "kind": role, "edit": sqr["edit"]})
                continue
            sig = z4.sig(sorted(sqr["edit"].items())) if sqr else z4.sig(seq)
            qid = f"{'scene' if sqr else 'seq'}:{key}:{sig}"
            kind = "SCENE_TARGET_ONLY" if sqr else "ORDERED_TEACHER"
            out["queries"].append({"query": qid, "root": key, "lineage": r["root"], "kind": kind, "sequence": seq, "edit": sqr and sqr["edit"],
                                   "role": role, "split": r["split"], "batch": r["batch"], "family": r["family"]})
            for sd in pol["policy"]["seeds"]:
                k_ = pol["policy"]["seed_key"].format(root=key, sig=("scene/" if sqr else "") + sig, seed=sd)
                u = {"unit": f"{qid}#s{sd}", "query": qid, "seed": sd, "kind": kind, "root": key, "target_key": k_ + "/target",
                     "relocation_key": k_ + "/reloc"}
                u.update({"edit": sqr["edit"], "eligible": True, "intervened_scene_hash": sqr["intervened_scene_hash"]} if sqr
                         else {"sequence": seq})
                out["units"].append(u)
    out["plan_hash"] = cf.stable_hash({k: v for k, v in out.items() if k != "plan_hash"})
    return out
