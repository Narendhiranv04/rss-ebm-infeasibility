"""P15-2: multimodal motion evidence over the P14 index (real witnesses only): deterministic canonical mode clustering with terminal-pose
signature, leakage exclusion by root AND geometry fingerprint, and a leave-one-root-out transfer benchmark on real held-out scenes.

Query = a held-out witness's own reconstructed scene and start q (its existence proves the move was available there); candidates = witnesses
of the SAME absolute move from OTHER roots (other fingerprints). Every candidate is decided by the exact certifier: certified joint-linear
transit from the query q to the candidate start, then motion.certify_trajectory in the query scene. Similarity only orders candidates; a
threshold only skips candidates (cost control) and is fitted on a DEV half of the queries and scored on the other half."""
from __future__ import annotations

import collections
import time

import numpy as np

import config as cf
import p14_sweep_index as sx
from p15_bench import OUT, rd

MODE_IOU, MODE_QEND = 0.6, 0.3      # merge only if whole-path swept IoU >= 0.6 AND terminal configurations within 0.3 rad (L-inf)


def modes(entries):
    """Deterministic canonical clustering: entries in (travel, file, index) order; join the first mode whose seed has IoU >= MODE_IOU and
    terminal-q distance <= MODE_QEND; order-invariant because the canonical sort fixes the processing order."""
    out = []
    for e in sorted(entries, key=lambda e: (round(e["travel_J"], 12), e["file"], e["index"])):
        for m in out:
            s = m[0]
            iou = len(s["_all"] & e["_all"]) / max(1, len(s["_all"] | e["_all"]))
            if iou >= MODE_IOU and np.max(np.abs(np.asarray(s["q_end"]) - np.asarray(e["q_end"]))) <= MODE_QEND:
                m.append(e)
                break
        else:
            out.append([e])
    return out


def index_v2():
    """P14 entries minus any whose root fingerprint matches an excluded (evaluation-pool) root fingerprint; mode statistics."""
    idx = sx.Index()
    P = sx._plans()
    excl = set(rd("runs/p14/eval_roster.json")["excluded_roots"])
    bad_fp = {json_fp(P[k]) for k in excl if k in P}
    keep = collections.defaultdict(list)
    dropped = 0
    for key, es in idx.by_key.items():
        for e in es:
            if json_fp(P[e["root"]]) in bad_fp:
                dropped += 1
                continue
            keep[key].append(e)
    idx.by_key = keep
    return idx, dropped


def json_fp(r):
    import json
    return json.dumps(r["fingerprint"])


def mode_stats() -> dict:
    idx, dropped = index_v2()
    per = []
    kinds, grasp = collections.Counter(), collections.Counter()
    for key, es in idx.by_key.items():
        ms = modes(es)
        perm = modes(list(reversed(es)))
        per.append({"records": len(es), "roots": len({e["support_id"] for e in es}), "modes": len(ms),
                    "modes_ge2_roots": sum(len({x["support_id"] for x in m}) >= 2 for m in ms),
                    "permutation_invariant": [sorted(id(x) for x in m) for m in ms] == [sorted(id(x) for x in m) for m in perm]})
        kinds.update(e["kind"] for e in es)
        grasp.update(tuple(np.round(e["held_local"], 4)) for e in es)
    out = {"keys": len(per), "entries": sum(p["records"] for p in per), "fingerprint_dropped": dropped, "modes": sum(p["modes"] for p in per),
           "independent_supported_modes": sum(p["modes_ge2_roots"] for p in per), "keys_ge2_modes": sum(p["modes"] >= 2 for p in per),
           "permutation_invariant_keys": sum(p["permutation_invariant"] for p in per), "kinds": dict(kinds),
           "grasp_offsets_distinct": len(grasp), "note": "kind / grasp offset are recorded but not hard filters: the certifier checks the actual "
           "held disc against the current scene and check_record verifies the grasp holds the object where it rests"}
    (OUT / "lookup").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "lookup" / "MODE_STATS_V2.json", out)
    return out


def _certify(cx, scene, oid, q, a):
    """(status, reason): VERIFIED or the failing stage (transit / trajectory / record). Never an infeasibility claim."""
    import geometry as ge
    import motion as mo
    import p14_d_search as ds
    if ds._transit(cx, scene, oid, q, a["q"][0], float(a["gap"][0])) is None:
        return "UNKNOWN_TRANSIT", "straight transit to the witness start not certified"
    try:
        v = mo.certify_trajectory(cx.robot, scene, oid, a, cx.p["delta"], cx.p["eps"], cx.p["evals"])
    except ValueError as ex:
        return "INCOMPATIBLE_RECORD", str(ex)[:60]
    if all(x.status == ge.CLEAR for x in v):
        return "VERIFIED", None
    return ("COLLISION" if any(x.status == ge.COLLISION for x in v) else "UNRESOLVED"), next(x.pair for x in v if x.status != ge.CLEAR)


def transfer(max_per_key=2, seed=0) -> dict:
    """Leave-one-root-out transfer benchmark (see module docstring); per query, every compatible other-root candidate is certified."""
    import p14_d_search as ds
    idx, _ = index_v2()
    cx = ds.Ctx(lookup=None)
    rng = np.random.default_rng(seed)
    provs = {(p["file"], p["index"]): (p, r) for p, a, r in sx.records()}
    rows = []
    for key in sorted(idx.by_key):
        es = idx.by_key[key]
        roots = sorted({e["support_id"] for e in es})
        if len(roots) < 2:
            continue
        for qr in [roots[i] for i in rng.permutation(len(roots))[:max_per_key]]:
            qe = next(e for e in sorted(es, key=lambda e: (e["file"], e["index"])) if e["support_id"] == qr)
            prov, root = provs[(qe["file"], qe["index"])]
            scene = sx.scene_at(root, prov["prefix"])
            q = np.asarray(qe["q_start"], float)
            cands = [e for e in es if e["support_id"] != qr]
            t0 = time.process_time()
            res = {}
            for mode_name in ("combined", "local", "global"):
                fake = type("I", (), {})()
                fake.by_key, fake.version, fake.hash = {key: cands}, idx.version, idx.hash
                res[mode_name] = [(c["entry"]["file"], c["entry"]["index"], c["mode"], c["similarity"])
                                  for c in sx.lookup_candidates(fake, prov["moved"], key[0], key[1], scene, root["goal_cell"], qe["radius"], mode_name)["candidates"]]
            ms = modes(cands)
            mode_of = {(e["file"], e["index"]): mi for mi, m in enumerate(sorted(ms, key=lambda m: (-len({x["support_id"] for x in m}), min(x["travel_J"] for x in m)))) for e in m}
            st = {}
            for e in cands:
                st[(e["file"], e["index"])] = (*_certify(cx, scene, prov["moved"], q, sx.Index.arrays(idx, e)), e["travel_J"])
            rows.append({"key": list(key), "query_root": qr, "family": qe["family"], "candidates": len(cands), "modes": len(ms),
                         "status": {f"{f}#{i}": v[0] for (f, i), v in st.items()}, "travel": {f"{f}#{i}": v[2] for (f, i), v in st.items()},
                         "mode_of": {f"{f}#{i}": m for (f, i), m in mode_of.items()}, "ranking": {k2: [f"{f}#{i}" for f, i, _, _ in v] for k2, v in res.items()},
                         "similarity": {k2: [s for *_, s in v] for k2, v in res.items()}, "cpu_s": time.process_time() - t0})
    cf.write_json(OUT / "lookup" / "TRANSFER.json", {"rows": rows, "version": "P15_TRANSFER_V1"})
    return summarize()


def summarize(dev_frac=0.5) -> dict:
    """Top-1 / any-certified, calls to first certified, first- vs best-certified travel, rare-mode rescue, transit / collision failures;
    similarity threshold fitted on DEV queries (minimum calls with >= 95 % of DEV availability kept), scored on TEST queries."""
    rows = rd(OUT / "lookup" / "TRANSFER.json")["rows"]
    keys = sorted({tuple(r["key"]) for r in rows})
    dev = {k for i, k in enumerate(keys) if i % 2 == 0}
    def stats(rs, mode="combined", thr=None):
        out = collections.Counter()
        calls, gap = [], []
        for r in rs:
            rank = r["ranking"][mode]
            sims = r["similarity"][mode]
            avail = any(v == "VERIFIED" for v in r["status"].values())
            out["queries"] += 1
            out["available_any"] += avail
            tried = [c for c, s in zip(rank, sims) if thr is None or s >= thr]
            first = next((i for i, c in enumerate(tried) if r["status"][c] == "VERIFIED"), None)
            out["top1_certifies"] += bool(tried) and r["status"][tried[0]] == "VERIFIED"
            out["found_within_tried"] += first is not None
            calls.append((first + 1) if first is not None else len(tried))
            if first is not None:
                best = min(r["travel"][c] for c in r["status"] if r["status"][c] == "VERIFIED")
                gap.append(r["travel"][tried[first]] - best)
                top_mode = r["mode_of"][rank[0]]
                if r["status"][rank[0]] != "VERIFIED" and r["mode_of"][tried[first]] != top_mode:
                    out["rare_mode_rescue"] += 1
            for c, s in r["status"].items():
                out[f"candidate_{s}"] += 1
        return dict(out) | {"mean_certifier_calls": float(np.mean(calls)) if calls else None,
                            "mean_first_minus_best_travel": float(np.mean(gap)) if gap else None}
    D = [r for r in rows if tuple(r["key"]) in dev]
    T = [r for r in rows if tuple(r["key"]) not in dev]
    grid = sorted({s for r in D for s in r["similarity"]["combined"]})
    base = stats(D)["found_within_tried"]
    thr = max([t for t in grid if stats(D, thr=t)["found_within_tried"] >= 0.95 * base] or [None], key=lambda t: -1 if t is None else t)
    out = {"queries": len(rows), "dev_queries": len(D), "test_queries": len(T),
           "all": {m: stats(rows, m) for m in ("combined", "local", "global")},
           "threshold": {"value": thr, "dev": stats(D, thr=thr), "test_no_threshold": stats(T), "test_threshold": stats(T, thr=thr)},
           "note": "availability = some other-root witness certifies in the query scene; the query's own witness proves the move exists there"}
    cf.write_json(OUT / "lookup" / "TRANSFER_SUMMARY.json", out)
    return out
