"""P8c: 36 predeclared independent roots (12 per kappa construction intent) and <= 20-query priority catalogues.

Static and outcome-blind. Construction reuses the P7 constructors (p7_scenes._try) with their pose-screen / sector intent
predicates; the INTENT (K_PLUS / K_MINUS / K_ZERO) is never a label and never a model input. Variation per root: which
grasp-side neighbours are the edited blockers (random N / S / W choice), target and goal cells, start stratum S0-S3, 0-2
clutter objects, nuisance object and destinations. Splits (predeclared before construction): per intent 6 train /
2 validation / 4 test; wave 1 and wave 2 each hold 3 / 1 / 2 per intent. Catalogue priority (outcome-blind, essential and
cheapest first): reference, original empty repair, edits A, B, A+B, nuisance, single moves of the core movers, main repair,
longer alternative (main + extra mover, <= 4 moves), same-endpoint swap, omissions of the longer alternative, alternative
destination, remaining ordered pairs, omission closure - truncated at 20 unique teacher queries per root.
"""
from __future__ import annotations

import itertools

import p6_scenes as p6
import p7_scenes as p7
import sequences as sq
import zlayout_r4 as z4

VERSION = "P8_KAPPA_FAMILIES_V1"
INTENT = {"K_PLUS": "KP", "K_MINUS": "KM", "K_ZERO": "K0"}
SEED0, QUERY_CAP = 2026102501, 20
ROOTS = [{"index": i, "wave": w, "batch": w, "intent": it, "family": INTENT[it], "split": sp, "seed": SEED0 + i, "stratum": f"S{(j + fi + 2 * w) % 4}"}
         for i, (w, fi, it, j, sp) in enumerate((w, fi, it, j, sp) for w in (1, 2) for fi, it in enumerate(INTENT)
                                                  for j, sp in enumerate(("train", "train", "train", "validation", "test", "test")))]


def construct(cfg, robot, known: set) -> tuple:
    saved = (p7.ROOTS, p7.PREFIX)
    try:
        p7.ROOTS, p7.PREFIX = ROOTS, "p8_"
        return p7.construct(cfg, robot, known)
    finally:
        p7.ROOTS, p7.PREFIX = saved


def catalogue(cfg, robot, r) -> dict:
    """<= 20 unique teacher queries in the declared priority order (scene edits first among non-reference queries)."""
    base = p7.catalogue(cfg, robot, r)
    s, k = r["scene"], r["root"]
    core = [x for x in r["ids"] if x != "N"]
    main = p7._seq(r, r["main"])
    extra = [x for x in core if x not in r["main"]] or (["N"] if "N" in r["ids"] else [])
    long_ = (main + p7._seq(r, extra))[:4]
    seqs = [("original", [])] + [("single", p7._seq(r, [x])) for x in core] + [("main", main), ("long", long_)]
    if len(main) >= 2:
        seqs.append(("swap", list(reversed(main))))
    seqs += [("omission", [a for i, a in enumerate(long_) if i != t]) for t in range(len(long_))] if len(long_) >= 2 else []
    seqs += [("alternative", p7._seq(r, [r["main"][0] + "*"] + r["main"][1:]))]
    seqs += [("pair", p7._seq(r, list(p))) for p in itertools.permutations(core, 2)]
    for _, q in list(seqs):
        seqs += [("closure", [a for i, a in enumerate(q) if i != t]) for t in range(len(q))] if len(q) >= 2 else []
    scene_q = base["scene_queries"]
    room = QUERY_CAP - 1 - sum(q["eligible"] for q in scene_q)
    act = p7.data.make_query(cfg, k, s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", p7.data.PROTOCOL_V2)["learner"]["action"]
    pool, seen = [], set()
    for prio, (rl, q) in enumerate(seqs):
        if z4.sig(q) in seen or len(pool) >= room:
            continue
        seen.add(z4.sig(q))
        _, why = sq.parse_sequence(s, act, [list(x) for x in q], z4.MOVE_CAP)
        pool.append({"sequence": [list(x) for x in q], "role": rl, "priority": len(pool), "canonical": not why, "parse": why,
                     "z_layout": z4.build(k, q, [x for x in scene_q if x["kind"] != "nuisance"])})
    return {"scene_queries": scene_q, "pool": pool}


def plan(cfg, robot, pol, roots) -> dict:
    meta = {"stage": "P8C", "version": VERSION, "root_schedule": ROOTS, "strata": p6.STRATA}
    out = p6.plan(cfg, robot, pol, roots, catalogue, meta)
    for kk, r in out["roots"].items():
        src = next(x for x in roots if x["root"] == r["lineage"])
        r.update({"wave": src["wave"], "intent": src["intent"], "clutter": src["clutter"], "main": src["main"], "signature": src["signature"]})
    prio = {}
    for kk, r in out["roots"].items():
        if r["queries"]:
            for p in r["queries"]["pool"]:
                prio[f"seq:{kk}:{z4.sig([tuple(a) for a in p['sequence']])}"] = p["priority"]
    for q in out["queries"]:
        q["priority"] = (0 if q["role"] == "reference" else 1 if q["kind"] == "SCENE_TARGET_ONLY" else 2, prio.get(q["query"], 0))
        q["intent"] = out["roots"][q["root"]]["intent"]
    out["plan_hash"] = p6.cf.stable_hash({k: v for k, v in out.items() if k != "plan_hash"})
    return out
