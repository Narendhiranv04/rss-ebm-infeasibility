"""P14-D0/D1: provenance of REAL relocation witnesses and a deterministic, non-learned similarity index over them.

Every library entry is one stored relocation record (teacher unit or strict-TOP1 episode): the exact joint-space waypoint path q, gripper gap,
phase labels, attachment flags, held-object offset and placement, with root / query / seed / split provenance. Swept occupancy is derived
from that recorded path (robot links + gripper + carried disc, per phase, inflated by a per-step point-displacement bound); it is used for
indexing, clustering and screening only. Reuse in another scene is decided by the exact continuous certifier (motion.certify_trajectory)
in that scene. The robot base is fixed: a witness is reusable only for the SAME absolute source and destination cells (and a certified
joint-linear transit from the current configuration to the witness start); identical relative displacement elsewhere is not reusable."""
from __future__ import annotations

import collections
import json
import math
from pathlib import Path

import numpy as np

import config as cf
from p14_audit import OUT, rd

ROOTS_GLOB = ("runs/p10b", "runs/p11", "runs/p12", "runs/p13")


def _cell(xy):
    return int(round(xy[1] / 0.08 - 0.5)) * 8 + int(round(xy[0] / 0.08 - 0.5))


def _plans():
    out = {}
    for base in ROOTS_GLOB:
        for f in sorted(Path(base).glob("**/plan.json")):
            for k, r in rd(f)["roots"].items():
                out.setdefault(k, {**r, "_plan": str(f)})
    return out


def records():
    """Yield (provenance, relocation arrays) for every stored relocation record whose move sequence is known."""
    import g4d_run as gr
    P = _plans()
    for base in ROOTS_GLOB:
        for f in sorted(Path(base).glob("**/relocation.npz")):
            d = f.parent
            if not (d / "job.json").exists():
                continue
            job = rd(d / "job.json")
            if "unit" in job:                                   # teacher unit
                u = job["unit"]
                root, seq, src = u["root"], u.get("sequence") or [], "teacher"
                qid, seed = u["query"], u["seed"]
            else:                                               # strict-TOP1 episode
                res = rd(d / "result.json") if (d / "result.json").exists() else {}
                root, seq, src = job["root"], res.get("committed_sequence") or [], "episode"
                qid, seed = job.get("choice"), job.get("arm")
            if root not in P or not seq:
                continue
            arrs = gr._arrays(f)
            if len(arrs) != len(seq):
                continue
            for i, (a, (oid, dest)) in enumerate(zip(arrs, seq)):
                yield {"file": str(f), "index": i, "root": root, "query": qid, "seed": seed, "source": src, "split": P[root].get("split"),
                       "family": P[root].get("family"), "moved": oid, "dest_cell": int(dest), "prefix": [list(x) for x in seq[:i]]}, a, P[root]


def scene_at(root_rec, prefix):
    from scene import scene_from_dict
    import p6_scenes as p6
    s = scene_from_dict(root_rec["scene"])
    return p6.moved(s, {o: int(c) for o, c in prefix}) if prefix else s


def audit(sample=150) -> dict:
    """Counts by source / split / phase schedule, exact-field presence, and trace fidelity: replay of a stratified sample of records in
    their own reconstructed scene with the exact certifier (every segment must be CLEAR) and placement consistency."""
    import geometry as ge
    import motion as mo
    import skill as sk
    cfg = cf.load_config("config.toml")
    robot, p = ge.make_robot(cfg), sk.params(cfg)
    cnt, keys, fields, sched = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    roots_per_key = collections.defaultdict(set)
    picked = []
    rng = np.random.default_rng(0)
    for prov, a, r in records():
        cnt[(prov["source"], prov["split"])] += 1
        fields.update(k for k in ("q", "gap", "phase", "attached", "held_local", "place_xy") if k in a)
        sched[tuple(dict.fromkeys(map(str, a["phase"])))] += 1
        s = scene_at(r, prov["prefix"])
        src = s.get(prov["moved"]).cell
        key = (src, prov["dest_cell"])
        keys[key] += 1
        roots_per_key[key].add(prov["root"])
        if rng.random() < sample / 6000:
            picked.append((prov, a, r))
    fid = collections.Counter()
    for prov, a, r in picked[:sample]:
        s = scene_at(r, prov["prefix"])
        try:
            v = mo.certify_trajectory(robot, s, prov["moved"], a, p["delta"], p["eps"], 4000)
            st = "ALL_CLEAR" if all(x.status == ge.CLEAR for x in v) else "NOT_ALL_CLEAR:" + ",".join(sorted({x.status for x in v if x.status != ge.CLEAR}))
        except ValueError as ex:
            st = "RECORD_INCONSISTENT:" + str(ex)[:60]
        fid[st] += 1
    sup = [len(v) for v in roots_per_key.values()]
    out = {"records_by_source_split": {f"{a}|{b}": n for (a, b), n in cnt.items()}, "records": sum(cnt.values()),
           "fields_present": dict(fields), "phase_schedules": {"->".join(k): n for k, n in sched.items()},
           "distinct_absolute_src_dst_keys": len(keys), "records_per_key": {"median": float(np.median(list(keys.values()))), "max": max(keys.values())},
           "independent_roots_per_key": {"median": float(np.median(sup)), "keys_with_ge2_roots": sum(x >= 2 for x in sup), "keys_with_ge5_roots": sum(x >= 5 for x in sup)},
           "trace_fidelity_sample": {"n": min(sample, len(picked)), "replay_in_own_scene": dict(fid)},
           "reconstructable_sweep": "yes: q (joint waypoints), gap, phase, attached and held_local give every robot body, the gripper and the carried "
                                    "disc at any interpolated configuration (geometry.bodies); joint-linear segments are what the certifier checks",
           "missing": "no Cartesian timestamps or velocities (not needed: geometric sweep only); no per-waypoint scene snapshot (reconstructed from root "
                      "scene + executed prefix); P10a / historical G-stage witnesses without job provenance are excluded"}
    (OUT / "lookup").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "lookup" / "PROVENANCE_AUDIT.json", out)
    return out


# ------------------------------------------------------------------ D1: index entries (descriptors + conservative per-phase swept cells)
PHASES = ("pregrasp", "approach", "close_attach", "carry", "release", "retreat")
STEP_M = 0.004                                         # max point displacement between swept samples (inflation added to the test)


def _occ(scene, goal, moving=None):
    """64-cell categorical occupancy: 0 free, 1 object, 2 target t0, 3 goal cell (free), 4 the moving object."""
    g = np.zeros(64, int)
    for o in scene.objects:
        g[o.cell] = 4 if o.id == moving else 2 if o.id == "t0" else 1
    if g[goal] == 0:
        g[goal] = 3
    return g


def windows(occ, c):
    """3x3 window centred on cell c and a 4x4 window anchored with c at local (1, 1); off-board = -1."""
    r0, c0 = divmod(c, 8)
    get = lambda r, cc: int(occ[r * 8 + cc]) if 0 <= r < 8 and 0 <= cc < 8 else -1  # noqa: E731
    return [get(r0 + dr, c0 + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)], [get(r0 + dr, c0 + dc) for dr in (-1, 0, 1, 2) for dc in (-1, 0, 1, 2)]


def swept_cells(robot, cfg, a) -> dict:
    """Per phase: set of cells whose obstacle disc (radius r) would come within margin delta of a robot body or the carried disc anywhere
    on the recorded joint-linear path; samples are dense enough that no body point moves more than STEP_M between samples, and STEP_M is
    added to the threshold (conservative screen; the exact certifier remains decisive)."""
    import geometry as ge
    import skill as sk
    p = sk.params(cfg)
    centers = np.array([((c % 8 + 0.5) * 0.08, (c // 8 + 0.5) * 0.08) for c in range(64)])
    thr = cfg["suite"]["obstacle_radius_m"] + p["delta"] + STEP_M
    rho = np.cumsum(robot.lengths[::-1])[::-1] + 0.08
    q, gap = np.asarray(a["q"], float), np.asarray(a["gap"], float)
    held = (None, cfg["suite"]["obstacle_radius_m"], tuple(a["held_local"])) if a["held_local"].size else None
    out = {ph: set() for ph in PHASES}
    for i, ph in enumerate(map(str, a["phase"])):
        dq = q[i + 1] - q[i]
        n = max(2, int(math.ceil(float(rho @ np.abs(dq)) / STEP_M)) + 1)
        hit = np.zeros(64, bool)
        for s in np.linspace(0, 1, n):
            for core, rad in ge.bodies(robot, q[i] + s * dq, gap[i] + s * (gap[i + 1] - gap[i]), held if a["attached"][i] else None).values():
                core = np.atleast_2d(core)
                dist = np.hypot(*(centers - core[0]).T) if len(core) == 1 else ge.points_to_core(centers, core)
                hit |= dist - rad < thr
        out[ph] |= set(np.flatnonzero(hit).tolist())
    return {ph: sorted(v) for ph, v in out.items()}


def build_index(exclude="runs/p14/eval_roster.json") -> None:
    """INDEX.json: one entry per stored relocation record not from an excluded root (all P13 blind roots: D3 evaluation pool)."""
    import geometry as ge
    import ordered as od
    cfg = cf.load_config("config.toml")
    robot = ge.make_robot(cfg)
    excl = set(rd(exclude)["excluded_roots"])
    ents, skipped = [], collections.Counter()
    for prov, a, r in records():
        if prov["root"] in excl:
            skipped["excluded_eval_pool_root"] += 1
            continue
        s = scene_at(r, prov["prefix"])
        o = s.get(prov["moved"])
        src, dst = o.cell, prov["dest_cell"]
        occ = _occ(s, r["goal_cell"], prov["moved"])
        w3s, w4s = windows(occ, src)
        w3d, w4d = windows(occ, dst)
        ents.append({**{k: prov[k] for k in ("file", "index", "root", "query", "seed", "source", "split", "family")},
                     "src": src, "dst": dst, "delta_rc": [dst // 8 - src // 8, dst % 8 - src % 8], "radius": o.radius_m, "kind": o.kind,
                     "q_start": np.asarray(a["q"][0]).round(9).tolist(), "q_end": np.asarray(a["q"][-1]).round(9).tolist(),
                     "held_local": np.asarray(a["held_local"]).round(9).tolist(), "travel_J": od.joint_travel(robot, [a])["J"],
                     "w3_src": w3s, "w4_src": w4s, "w3_dst": w3d, "w4_dst": w4d, "global": occ.tolist(), "swept": swept_cells(robot, cfg, a),
                     "support_id": prov["root"]})
    idx = {"version": "P14_SWEEP_INDEX_V1", "entries": ents, "skipped": dict(skipped), "step_m": STEP_M, "excluded_roots": len(excl)}
    idx["hash"] = cf.stable_hash({"n": len(ents), "keys": sorted((e["src"], e["dst"], e["file"], e["index"]) for e in ents)})
    cf.write_json(OUT / "lookup" / "INDEX.json", idx)


# ------------------------------------------------------------------ D1: retrieval (hard filter -> similarity -> swept-mode clusters)
W_GLOBAL = 0.25          # combined score = local + W_GLOBAL * global (fixed before any evaluation; DEV only)
MODE_IOU = 0.6           # two witnesses belong to one motion mode if their whole-path swept-cell IoU >= MODE_IOU


class Index:
    def __init__(self, path=OUT / "lookup" / "INDEX.json"):
        d = rd(path)
        self.version, self.hash = d["version"], d["hash"]
        self.by_key = collections.defaultdict(list)
        for e in d["entries"]:
            e["_all"] = set().union(*map(set, e["swept"].values()))
            self.by_key[(e["src"], e["dst"])].append(e)
        self._arr = {}

    def arrays(self, e):
        import g4d_run as gr
        k = (e["file"], e["index"])
        if k not in self._arr:
            self._arr[k] = gr._arrays(Path(e["file"]))[e["index"]]
        return self._arr[k]


def _sim(a, b):
    m = [(x, y) for x, y in zip(a, b)]
    return sum(x == y for x, y in m) / len(m)


def lookup_candidates(idx, oid, src, dst, scene, goal, radius, mode="combined", modal_only=False) -> dict:
    """Ordered candidate witnesses for 'object at src -> dst' in `scene`. Hard filters first (absolute src / dst for the fixed-base arm, object
    footprint); then similarity (local = 3x3 + 4x4 windows at source and destination; global = 64-cell occupancy); then motion modes by
    swept-cell IoU, ranked by independent supporting roots, best similarity, travel. Nothing here is a feasibility decision: candidates are
    certified in the exact current scene by the caller; an empty result is UNKNOWN under lookup."""
    pool = idx.by_key.get((src, dst), [])
    rej = collections.Counter()
    ok = []
    for e in pool:
        if abs(e["radius"] - radius) > 1e-9:
            rej["footprint"] += 1
            continue
        ok.append(e)
    if not ok:
        return {"status": "UNKNOWN_NO_COMPATIBLE_WITNESS", "candidates": [], "modes": 0, "rejected": dict(rej), "pool": len(pool)}
    occ = _occ(scene, goal, oid)
    w3s, w4s = windows(occ, src)
    w3d, w4d = windows(occ, dst)
    score = {}
    for e in ok:
        loc = (_sim(w3s, e["w3_src"]) + _sim(w4s, e["w4_src"]) + _sim(w3d, e["w3_dst"]) + _sim(w4d, e["w4_dst"])) / 4
        glo = _sim(occ, e["global"])
        score[id(e)] = {"local": loc, "global": glo, "combined": loc + W_GLOBAL * glo}[mode]
    ranked = sorted(ok, key=lambda e: (-score[id(e)], e["travel_J"], e["file"], e["index"]))
    modes = []
    for e in ranked:
        for m in modes:
            u = m[0]["_all"]
            if len(u & e["_all"]) / max(1, len(u | e["_all"])) >= MODE_IOU:
                m.append(e)
                break
        else:
            modes.append([e])
    modes.sort(key=lambda m: (-len({x["support_id"] for x in m}), -max(score[id(x)] for x in m), min(x["travel_J"] for x in m)))
    occupied = {o.cell for o in scene.objects if o.id != oid}
    cands = []
    for mi, m in enumerate(modes):
        for j, e in enumerate(m if not modal_only else m[:1]):
            cands.append({"entry": e, "mode": mi, "rank_in_mode": j, "support": len({x["support_id"] for x in m}), "similarity": score[id(e)],
                          "travel_J": e["travel_J"], "screen_hits": sorted(e["_all"] & occupied)})
        if modal_only:
            break
    return {"status": "CANDIDATES", "candidates": cands, "modes": len(modes), "rejected": dict(rej), "pool": len(pool),
            "index": idx.version, "index_hash": idx.hash}


def mode_stats() -> dict:
    """Library structure: motion modes per absolute (src, dst) key, independent support per mode, duplicate inflation."""
    idx = Index()
    per = []
    for key, es in idx.by_key.items():
        modes = []
        for e in sorted(es, key=lambda e: (e["travel_J"], e["file"], e["index"])):
            for m in modes:
                if len(m[0]["_all"] & e["_all"]) / max(1, len(m[0]["_all"] | e["_all"])) >= MODE_IOU:
                    m.append(e)
                    break
            else:
                modes.append([e])
        per.append({"key": key, "records": len(es), "roots": len({e["support_id"] for e in es}), "modes": len(modes),
                    "modes_ge2_roots": sum(len({x["support_id"] for x in m}) >= 2 for m in modes)})
    out = {"keys": len(per), "entries": sum(p["records"] for p in per), "modes_total": sum(p["modes"] for p in per),
           "keys_with_ge2_modes": sum(p["modes"] >= 2 for p in per), "independent_supported_modes": sum(p["modes_ge2_roots"] for p in per),
           "records_per_root_ratio": float(sum(p["records"] for p in per) / max(1, sum(p["roots"] for p in per))),
           "median_modes_per_key": float(np.median([p["modes"] for p in per]))}
    cf.write_json(OUT / "lookup" / "MODE_STATS.json", out)
    return out
