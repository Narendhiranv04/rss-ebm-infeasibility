"""All-movable conversion of the 2D roots that contain immovable context structures (dataset-bd's original 2D generator and
ROUTE2D_V1-V3), uniform with the current generator (user, 1 Nov 2026: every object movable, as in the rest of the dataset).

One old root -> one new root with the same spec, start, goal and start pose:
  * an object the arm cannot pick anywhere (a cell outside the pickable cells) is removed -- the current generator never puts an
    object there;
  * every other formerly fixed object becomes an ordinary movable object: role extra_blocker when it kills a grasp of the target
    (static screens, bd_2d_cover_table.json) or lies on the target's empty-scene sweep, else a distractor role by location;
    parking cells by the current generator's rule (bd_2d_exact._parking), per object (an object without a free cell stays
    unparked, as in generated roots); the old objects keep their roles and parking cells;
  * the teacher relabels the scene from scratch (pool: the objects of the old minimal repair first, then the old intended blockers,
    then the other blockers by static rank) and the independence check runs as for generated roots.
The class is the verified K, kept only where it is certainly minimal (the generated roots' standard): K <= 1 always; K = 2 when no
single object (D edits: every parked object) frees the target; K >= 3 when also the static grasp bound over ALL objects reaches K.
Otherwise UNCERTIFIED (not counted).  A conversion that verifies no repair of <= 4 moves is UNCERTIFIED_NONE and not counted: a NONE
root of the dataset carries the static >= 5-move certificate of bd_2d_exact.construct_none."""
from __future__ import annotations

import json
import signal
import time
import traceback
from dataclasses import replace

import numpy as np

import bd_core as bc
import bd_2d as A

GEN = "BD_GEN_V1+ROUTE2D_V5+CONV1"
POOL_CAP = 4                        # teacher pool and intended blockers (as V5 NONE): repairs of <= 4 relocations, and every B sequence
                                    # (redundant+2 = 2 + K relocations, intended union) stays within MAX_ACTIONS
ADMIN_CPU_S = 1500.0
WALL_S = 2400
QUOTA_CLASSES = ("NONE", "0", "1", "2", "3", "4")


def kills(cell, side, at):
    """Per declared grasp of the target at `cell` ('pick' or 'place'): the ids of the objects killing it (static screens)."""
    import bd_2d_exact as E
    E.side_ok(cell, "p", 1)                                      # loads the table
    if E._TABLE["t"] is None:
        raise RuntimeError("bd_2d_cover_table.json missing or made for another configuration")
    ways = E._TABLE["t"]["cells"][str(cell)][side]["per_grasp"].values()
    return [frozenset().union(*[at.get(x, set()) for x in v["movable"] + v["fixed_only"]]) for v in ways]


def static_rank(src, goal, at):
    """(K_static over all objects, union of its cheapest kill sets, per-object number of killed grasps)."""
    kp, kq = kills(src, "pick", at), kills(goal, "place", at)
    n = {}
    for u in kp + kq:
        for o in u:
            n[o] = n.get(o, 0) + 1
    best, cheap = None, set()
    for a in set(kp):
        for b in set(kq):
            u = a | b
            if best is None or len(u) < best:
                best, cheap = len(u), set(u)
            elif len(u) == best:
                cheap |= u
    return best, cheap, n


def convert(old, root_id):
    """Construction dict (bd_2d.teach / bd_2d_plus.record format) of the all-movable version of old root record `old`."""
    import bd_2d_exact as E
    import p14_sweep_index as sx
    from scene import place_scene, scene_to_dict
    G = A.env()
    cfg, robot = G["cfg"], G["robot"]
    spec, valid = old["spec"], set(G["valid"])
    cell = {o["id"]: int(o["cell"]) for o in old["scene"]["objects"]}
    src, goal, q0 = cell["t0"], int(old["goal_cell"]), tuple(old["q0"])
    fixed = set(old["fixed_ids"])
    gone = sorted(o for o in fixed if cell[o] not in valid)               # cannot be picked anywhere: removed
    keep = sorted(o for o in cell if o != "t0" and o not in gone)
    new = {o: f"o{i:02d}" for i, o in enumerate(keep)}                     # place_scene numbers obstacles in order
    occ = [cell[o] for o in keep]
    s = place_scene(cfg, root_id, "orig", [src], occ)
    s = replace(s, robot=replace(s.robot, q_rad=q0))
    s0 = replace(place_scene(cfg, root_id, "empty", [src], []), robot=s.robot)
    task0 = {"scene": scene_to_dict(s0), "goal_cell": goal, "fixed_ids": [], "target": "t0", "key": f"{root_id}:empty"}
    st, a_empty, _ = A.t_goal(task0, {}, {"t0": src}, np.asarray(q0, float))
    if st != "VERIFIED":
        raise A.Reject("CONV_EMPTY_SCENE_BLOCKED")
    sw_empty = sx.swept_cells(robot, cfg, a_empty)
    seen = set().union(*map(set, sw_empty.values()))
    at = {}
    for o in keep:
        at.setdefault(cell[o], set()).add(new[o])
    k_stat, cheap, nkill = static_rank(src, goal, at)
    near_sw = {c for c in valid for x in seen if A.cheb(c, x) == 1}
    region = {"near_T": lambda c: 1 <= A.cheb(c, src) <= 3 and c not in seen,          # bd_2d_exact.construct_none's regions
              "near_goal": lambda c: 1 <= A.cheb(c, goal) <= 3 and c not in seen,
              "near_path": lambda c: c in near_sw and c not in seen,
              "far": lambda c: A.cheb(c, src) >= 3 and A.cheb(c, goal) >= 3 and c not in seen}
    order = [spec["distractor_role"]] + [r for r in ("near_path", "near_T", "near_goal", "far") if r != spec["distractor_role"]]
    roles = {}
    for o in keep:
        if o not in fixed and o in old["roles"]:
            roles[new[o]] = old["roles"][o]
        elif (nkill.get(new[o]) or cell[o] in seen) and str(old["observed"]) != "0":    # K=0 roots have no blockers
            roles[new[o]] = "extra_blocker"
        else:
            roles[new[o]] = "distractor:" + next((r for r in order if region[r](cell[o])), "far")
    first = {new[o]: int(old["parking"][o][0]["cell"]) for o in keep if old["parking"].get(o)}   # old parking: the recorded repair's
    for o in keep:                                                                              # cells
        if new[o] not in first:
            got = E._parking(G, src, goal, seen, occ + list(first.values()), [cell[o]], lambda x: False, n=5)
            if got:
                first[new[o]] = int(got[cell[o]][0])
    npark, parking, alt_parking = bc.PARKING_N[spec["parking"]], {}, {}
    for o in keep:
        n = new[o]
        if n not in first:                    # no free cell: not movable in the teacher's repair domain (as in generated roots)
            continue
        alts = E._parking(G, src, goal, seen, occ + [v for k, v in first.items() if k != n], [cell[o]], lambda x: False, n=5) or {}
        cells = list(dict.fromkeys([first[n]] + [int(p["cell"]) for p in old["parking"].get(o, [])] + [int(x) for x in alts.get(cell[o], [])]))
        parking[n] = [{"cell": x, "in_target_sweep": x in seen} for x in cells[:npark]]
        alt_parking[n] = cells
    old_class = str(old["observed"])
    minimal = next((b["sequence"] for b in old["labels"]["B"] if b["role"] == "minimal" and b["y"]), [])
    head = list(dict.fromkeys([new[o] for o, _ in minimal if o in new] + [new[o] for o in old.get("intended_blockers", []) if o in new]))
    if old_class in ("NONE", "0"):
        head = []                                                     # no known repair: static rank only
    rest = sorted((n for n, r in roles.items() if n not in head and r in ("intended_blocker", "extra_blocker")),
                  key=lambda n: (n not in cheap, s.get(n).cell not in seen, -nkill.get(n, 0), n))
    wrong = sorted((c for c in seen if c in valid and c not in set(occ + [src]) and c != goal), key=lambda c: (A.cheb(c, src), c))
    return {"scene": s, "goal": goal, "q0": q0, "stratum": old["start_stratum"], "fixed_ids": [], "fixed_cells": [],
            "roles": roles, "parking": parking, "alt_parking": alt_parking, "wrong_dest": wrong[:2], "empty_witness": a_empty,
            "swept": {k: list(v) for k, v in sw_empty.items()}, "intended": [] if old_class == "0" else (head + rest)[:POOL_CAP], "pools": {"route": sorted(seen)},
            "chase_queries": 0, "blockers_only_scene_blocks_T": True,
            "construction": {"method": "convert2d", "converted_from": old["ids"]["root_id"], "old_observed": old_class,
                             "old_method": old.get("construction", {}).get("method", "bd_2d"), "old_generator": old.get("generator_version_plus"),
                             "removed_unreachable_cells": [cell[o] for o in gone], "formerly_fixed": [new[o] for o in keep if o in fixed],
                             "K_static": k_stat, "immovable": 0, "pool_cap": POOL_CAP}}


class WallCap(Exception):
    pass


def _alarm(sig, frame):
    raise WallCap("administrative wall cap")


def new_line(old_id, spec):
    """Ledger line skeleton of the conversion of old_id (every ledger line carries spec and wave: run_2d_plus reads them)."""
    name, wave, slot = old_id[3:old_id.rindex("-w")], int(old_id[old_id.rindex("-w") + 2:old_id.rindex("-s")]), int(old_id[old_id.rindex("-s") + 2:])
    return {"gen": GEN, "converted_from": old_id, "wave": wave, "slot": slot, "spec": spec, "root_id": f"2d-conv-{name}-w{wave:04d}-s{slot:03d}"}


def retire_line(old_id, spec, k_stat):
    """Old NONE root built with immovable walls, retired without a conversion run: with every object movable it cannot carry the
    NONE certificate (static grasp bound >= 5) and cannot certify K >= 3 (bound < 3); K1 / K2 outcomes are full / cheap to make.
    Deterministic content, so the same line on both machines merges once."""
    return new_line(old_id, spec) | {"accepted": False, "old_observed": "NONE", "wall_s": 0.0,
                                     "reason": f"RETIRED:NONE_WITH_IMMOVABLE_WALLS(K_static_all={k_stat})"}


def static_bound(old, G):
    """Static grasp bound of the all-movable version of old root record `old` (objects the arm cannot pick removed)."""
    cell = {o["id"]: int(o["cell"]) for o in old["scene"]["objects"]}
    fixed, valid = set(old["fixed_ids"]), set(G["valid"])
    at = {}
    for o, c in cell.items():
        if o != "t0" and not (o in fixed and c not in valid):
            at.setdefault(c, set()).add(o)
    return static_rank(cell["t0"], int(old["goal_cell"]), at)[0]


def work(job):
    """One conversion (worker process): (old root id, old roots dir, output run dir, old spec) -> ledger entry of the new root, which
    carries converted_from = the old root id (run_2d_plus then counts the new root instead of the old one)."""
    import bd_2d_plus as P
    old_id, src_dir, out, spec = job
    G = A.env()
    store = bc.Store(out)
    t0 = time.time()
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(WALL_S)
    entry = new_line(old_id, spec)
    try:
        old = json.load(open(f"{src_dir}/roots/{old_id}.json"))
        name = old_id[3:old_id.rindex("-w")]
        ids = bc.ids("2d", f"conv-{name}", entry["wave"], entry["slot"], spec, G["config_hash"], A.TEACHER_VERSION, G["protocol"])
        assert ids["root_id"] == entry["root_id"], ids["root_id"]
        entry.update(construction_seed=ids["construction_seed"])
        c = convert(old, ids["root_id"])
        A.POOL_CAP, A.ADMIN_CPU_S = c["construction"]["pool_cap"], ADMIN_CPU_S
        t = A.teach(c, store, ids["root_id"])
        rec = P.record(spec, ids, c, t, store)
        rec["generator_version_plus"] = GEN
        observed = str(rec["observed"])
        ms = next((b["sequence"] for b in rec["labels"]["B"] if b["role"] == "minimal" and b["y"]), [])
        if ms and not P.independent(c, ids["root_id"], ms):
            observed = "DEPENDENT"
        if observed == "NONE" or (observed.isdigit() and int(observed) > 4):
            observed = "UNCERTIFIED_NONE"
        elif observed.isdigit() and int(observed) >= 2:               # minimality not certain -> not counted
            single = any(d["y"] for d in rec["labels"]["D"] if len(d.get("edit") or {}) == 1)
            if single or (int(observed) >= 3 and (c["construction"]["K_static"] or 0) < int(observed)):
                observed = "UNCERTIFIED"
        rec["independent_moves"] = observed != "DEPENDENT"
        store.save_root(rec)
        entry.update(accepted=True, verified_K=rec["verified_K"], observed=observed, y_original=rec["labels"]["original"]["y"],
                     method="convert2d", old_observed=c["construction"]["old_observed"])
    except (A.Reject, A.AdminCap, WallCap) as e:
        entry.update(accepted=False, reason=f"{type(e).__name__}:{e}")
    except Exception as e:  # noqa: BLE001  (recorded, never silently dropped)
        entry.update(accepted=False, reason=f"ERROR:{type(e).__name__}:{str(e)[:200]}", trace=traceback.format_exc()[-1500:])
    finally:
        signal.alarm(0)
    entry["wall_s"] = time.time() - t0
    return entry


def todo(led, roots_dir):
    """Old roots to convert: (root id, class, spec, record) of counted-class roots with immovable objects and no conversion line yet."""
    done = {e["converted_from"] for e in led if e.get("converted_from")}
    out = []
    for e in led:
        if e.get("accepted") and e.get("observed") in QUOTA_CLASSES and "converted_from" not in e and e["root_id"] not in done:
            rec = json.load(open(f"{roots_dir}/roots/{e['root_id']}.json"))
            if rec.get("fixed_ids"):
                out.append((e["root_id"], e["observed"], e["spec"], rec))
    return out
