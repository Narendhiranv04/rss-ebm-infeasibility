"""P17 evaluation runners (per-root checkpoint / resume; one exact shared cache per root so every arm or guide is charged virtual work
exactly as if computed alone; identical per-query budgets for all arms).

B root: scene-only catalogue -> each arm's frozen calibrated no-op-first choice -> existential verification (own per-sequence budget) +
independent replay. Baselines: ALWAYS_EMPTY, random, geometry-static selector, and VERIFY_FIRST (physically verify the original target;
if not VERIFIED run the common geometry-guided corrected search; every physical call and its CPU counted).
D root: every guide runs the corrected finite search (p17_search) on the same scene-only domain with the same budgets; plans replayed."""
from __future__ import annotations

import time

import numpy as np

import config as cf

D_BUDGET = {"motion_checks": 400, "online_moves": 6, "goal_plans": 16, "transit_plans": 8}     # P16 / P17 search budget (per guide)


def _seq(q):
    s = q.rsplit(":", 1)[1]
    return [] if s == "empty" else [(a.split(">")[0], int(a.split(">")[1])) for a in s.split(".")]


def d_root(task, idx, guides, cache, oracle=None, dino=None, n_max=4, per_obj=2, node_cap=500) -> dict:
    import p16_motion as pm
    import p17_d as pd
    import p17_domain as dm
    import p17_search as s17
    out = {}
    for g in guides:
        c0 = time.process_time()
        phys = pm.DetPhysics(task, idx, budget=D_BUDGET, cache=cache)
        cells0 = {o.id: o.cell for o in phys.s0.objects}
        mv, om = dm.movers(task, cells0)
        gd = pd.Guide(g, task, phys, mv, oracle, dino)
        r = s17.search(phys, gd.order, mv, dm.slot_fn(phys, idx, width=4, origin=cells0), cells0, phys.s0.robot.q_rad, n_max, per_obj, node_cap)
        rep = pm.replay(phys, r["plan"]) if r["found"] else None
        steps = r["plan"]["steps"] if r["found"] else []
        out[g] = {"found": r["found"], "replay": bool(rep and rep["ok"]), "traj": rep and rep.get("traj"), "claim": r["claim"],
                  "min_moves_proven": r["min_moves_proven"], "min_travel_proven": r["min_travel_proven"], "moves": len(steps),
                  "travel": r["plan"]["travel"] if r["found"] else None,
                  "steps": [(x["object"], x["dest"], x["via"], x.get("mode"), x.get("transit")) for x in steps],
                  "repeat_move": len({x["object"] for x in steps}) != len(steps), "flags": r["flags"], "log": r["log"], "used": phys.used,
                  "stats": phys.stats, "d_calls": gd.d_calls, "d_cpu": gd.d_cpu, "cpu_actual": time.process_time() - c0,
                  "cpu_virtual": phys.vcpu + (time.process_time() - c0 - phys.pcpu), "movers": mv, "omitted": om, "trace": gd.trace[:20]}
    return out


def b_root(task, idx, plan_root, key, arms, policies, ms, dino, cache) -> dict:
    """One B root: catalogue, scores (ensemble mean over seeds), choices, existential execution, baselines. Arm names: 'X' (own scores and
    policy), 'X|nocpl' (inference-only coupling toggle, own policy), 'X@Y' (X's scores, Y's thresholds), 'X#topK' (verify-then-next over X's
    admitted list, at most K physical verifications, all charged)."""
    import p16_motion as pm
    import p17_b as pb
    import p17_verify as pv
    import p16_eval as p16e
    c0 = time.process_time()
    fac = lambda: pm.DetPhysics(task, idx, budget=pb.SEQ_BUDGET, cache=cache)  # noqa: E731
    base = fac()
    cat = pb.catalogue(task, base, idx)
    st0 = base.goal({o.id: o.cell for o in base.s0.objects}, np.asarray(base.s0.robot.q_rad, float))[0]
    r = pb.prepare(pb.records({key: plan_root}, {key: {"catalogue": cat, "original": {"y": None}, "edits": {}, "sequences": {}}}, "test", dino,
                              with_labels=False))[0]
    rows_by = {}
    for a in {x.split("#")[0].split("@")[0] for x in arms if x.split("#")[0].split("@")[0].split("|")[0] in ms}:
        mm = ms[a.split("|")[0]]
        if a.endswith("|nocpl"):
            sc = []
            for m in mm:
                with pb.coupling_off(m):
                    sc.append(pb.score(m, a, [r])[key])
        else:
            sc = [pb.score(m, a, [r])[key] for m in mm]
        rows_by[a] = []
        for i, x in enumerate(sc[0]):
            p_ = float(np.mean([s[i]["p"] for s in sc]))
            rows_by[a].append(dict(x, p=p_, u=float(np.log(max(p_, 1e-9)) - np.log1p(-min(p_, 1 - 1e-9))),
                                   seq_idx=[(r["ids"].index(o), d) for o, d in x["seq"]]))
    any_rows = next(iter(rows_by.values()))
    rng = np.random.default_rng(int(key.split("_")[-1]) % (2 ** 31))
    out = {"y0_status": st0, "catalogue": {x: cat[x] for x in ("pool", "parking", "movers", "generated", "kept", "omitted")}, "arms": {}}
    verified = {}

    def ver(q):
        if q not in verified:
            ph = fac()
            v = pv.verify_existential(ph, _seq(q), first=True)
            rep = pm.replay(ph, v["plan"]) if v["plan"] else None
            verified[q] = {"status": v["status"] if not (v["status"] == pv.V and not (rep or {}).get("ok")) else "UNKNOWN_REPLAY_FAILED",
                           "moves": v["plan"]["moves"] if v["plan"] else None, "travel": v["plan"]["travel"] if v["plan"] else None,
                           "replay": bool(rep and rep["ok"]), "used": ph.used}
        return verified[q]

    for a in arms:
        a0, _, k = a.partition("#top")                       # 'X#top3': verify-then-next over X's admitted list (a separate condition)
        sk, _, thr = a0.partition("@")                       # 'X@Y': X's scores under Y's frozen thresholds (swapped-threshold analysis)
        rows = rows_by.get(sk, any_rows)
        pol = policies.get(thr or a0, {})
        if k:
            pp = lambda u: 1 / (1 + np.exp(-(u - pol.get("bias", 0.0)) / pol.get("temp", 1.0)))  # noqa: E731
            e = next(x for x in rows if x["m"] == 0)
            lst = ([e["query"]] if pp(e["u"]) >= pol["t0"] else []) + [x["query"] for x in sorted((x for x in rows if x["m"] > 0 and pp(x["u"]) >= pol["t"]),
                                                                                                   key=lambda x: (x["m"], -x["u"], x["query"]))]
            tried, used = [], {}
            for q in lst[:int(k)]:
                v = ver(q)
                tried.append((q, v["status"]))
                used = {c: used.get(c, 0) + n for c, n in v["used"].items()}
                if v["status"] == pv.V:
                    break
            q = tried[-1][0] if tried else "ABSTAIN"
            out["arms"][a] = {"choice": q, "status": "ABSTAIN"} if not tried else {"choice": q, "m": len(_seq(q)), **verified[q], "used": used, "tried": tried}
            continue
        q = p16e.b_select(sk, rows, pol, r["R"], [r["ids"].index(o) for o in cat["pool"]], rng)
        if q == "ABSTAIN":
            out["arms"][a] = {"choice": q, "status": "ABSTAIN"}
            continue
        out["arms"][a] = {"choice": q, "m": len(_seq(q)), **ver(q), "p_choice": next((x["p"] for x in rows if x["query"] == q), None)}
    out["scores"] = {a: [(x["query"], round(x["u"], 4)) for x in rw] for a, rw in rows_by.items()}
    out["cpu_s"] = time.process_time() - c0
    return out


def verify_first(task, idx, cache, dino=None) -> dict:
    """Strong non-learned baseline: certify the original target first (charged); if not VERIFIED run the geometry-guided corrected search."""
    import p16_motion as pm
    c0 = time.process_time()
    ph = pm.DetPhysics(task, idx, budget=D_BUDGET, cache=cache)
    st0 = ph.goal({o.id: o.cell for o in ph.s0.objects}, np.asarray(ph.s0.robot.q_rad, float))[0]
    if st0 == pm.V:
        return {"choice": "EMPTY", "status": pm.V, "moves": 0, "used": ph.used, "cpu_virtual": ph.vcpu, "cpu_actual": time.process_time() - c0}
    g = d_root(task, idx, ["geometry"], cache, dino=dino)["geometry"]
    return {"choice": "SEARCH", "status": pm.V if g["found"] and g["replay"] else "NO_PLAN", "moves": g["moves"], "travel": g["travel"],
            "used": {k: ph.used.get(k, 0) + g["used"].get(k, 0) for k in set(ph.used) | set(g["used"])},
            "cpu_virtual": ph.vcpu + g["cpu_virtual"], "cpu_actual": time.process_time() - c0}


def run_roots(split, keys, part, mode, guides=None, arms=None, oracles=None, sealed=False, tag="exec", claims=None) -> None:
    """mode 'd', 'b' or 'bd'. Per-root checkpointed output runs/p17/data/<split>/{sealed_}<tag>_<part>.json. claims: a directory for atomic
    per-root claims shared by several workers (a root is computed by exactly one worker)."""
    import os
    import json
    from pathlib import Path
    import p15_lookup as pl
    import p16_common as pc
    import p17_b as pb
    import p17_eval as pe
    import learn_data as ld
    idx, plan = pl.index_v2()[0], pe._plan(split)
    f = pe.DATA / split / (("sealed_" if sealed else "") + f"{tag}_{part}.json")
    res = pe.rd(f) if f.exists() else {}
    dino = ld.dino()
    ms, pol = {}, {}
    if "b" in mode:
        pol = json.loads(Path("runs/p17/b/POLICIES.json").read_text())
        ms = {a: [pb.load(a, s) for s in range(pol.get("_seeds", 3))] for a in {x.split("#")[0].split("@")[0].split("|")[0] for x in arms}
              if a not in ("empty_always", "random", "geometry", "verify_first")}
    for k in keys:
        if k in res:
            continue
        if claims is not None:
            Path(claims).mkdir(parents=True, exist_ok=True)
            try:
                os.close(os.open(Path(claims) / f"{k}.claim", os.O_CREAT | os.O_EXCL))
            except FileExistsError:
                continue
        c0, w0 = time.process_time(), time.time()
        task, cache = pc.task_from_root(plan["roots"][k], k), {}
        out = {"family": plan["roots"][k].get("family")}
        if "d" in mode:
            out["d"] = d_root(task, idx, guides, cache, (oracles or {}).get(k), dino)
        if "b" in mode:
            out["b"] = b_root(task, idx, plan["roots"][k], k, [a for a in arms if a != "verify_first"], pol, ms, dino, cache)
            if "verify_first" in arms:
                out["b"]["verify_first"] = verify_first(task, idx, cache, dino)
        out |= {"cpu_root": time.process_time() - c0, "wall_root": time.time() - w0}
        res[k] = out
        f.parent.mkdir(parents=True, exist_ok=True)
        cf.write_json(f, res)
