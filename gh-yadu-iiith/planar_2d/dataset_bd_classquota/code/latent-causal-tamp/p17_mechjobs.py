"""P17-3 mechanism jobs: pose-only screens, M5 mining, post-hoc physics-only mechanism mining from labels, exemplar evidence
packages, M7 finite library-domain checks."""
from __future__ import annotations

import time

import config as cf
from p17_eval import OUT, rd

from p17_data import _plan, merged


def mech_screen(seed0, n, want, gen, part) -> None:
    """Checkpointed per scene (resumable): each scene's outcome is appended before the next one starts."""
    import p17_mech as m
    f = OUT / "mechanisms" / "screen" / f"{want}_{gen}_{part}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    acc = rd(f) if f.exists() else {"stats": {}, "kept": [], "done": 0}
    for i in range(acc["done"], n):
        r = m.screen2(seed0 * 1000 + i, 1, want, 4, gen)
        for k, v in r["stats"].items():
            if isinstance(v, list):
                acc["stats"].setdefault(k, [])
                acc["stats"][k] += v
            elif isinstance(v, dict):
                acc["stats"].setdefault(k, {})
                for kk, vv in v.items():
                    acc["stats"][k][kk] = acc["stats"][k].get(kk, 0) + vv
            else:
                acc["stats"][k] = acc["stats"].get(k, 0) + v
        acc["kept"] += r["kept"]
        acc["done"] = i + 1
        cf.write_json(f, acc)


def m5_mine(items, part) -> None:
    """items: [(split, key)] from P16 dev / train / val / calib plans (development scenes)."""
    import p17_mech as m
    f = OUT / "mechanisms" / f"M5_{part}.json"
    res = rd(f) if f.exists() else {}
    for n, k in items:
        if f"{n}|{k}" in res:
            continue
        c0 = time.process_time()
        root = rd(f"runs/p16/data/{n}/plan.json")["roots"][k]
        out = m.mine_m5(root, k)
        res[f"{n}|{k}"] = out | {"cpu_s": time.process_time() - c0}
        (OUT / "mechanisms").mkdir(parents=True, exist_ok=True)
        cf.write_json(f, res)


def rescreen(seed0, idxs, gen, part) -> None:
    """Deterministic re-run of specific screening scenes (seed = seed0 * 1000 + i) with full per-scene recording."""
    import p17_mech as m
    f = OUT / "mechanisms" / "screen" / f"rescreen_{gen}_{part}.json"
    acc = rd(f) if f.exists() else {"kept": [], "scenes": []}
    for i in idxs:
        r = m.screen2(seed0 * 1000 + i, 1, "repeat", 4, gen)
        acc["kept"] += r["kept"]
        acc["scenes"] += r["stats"].get("scenes", [])
        cf.write_json(f, acc)


def mech_physical(kept, part) -> None:
    """kept: [(screen file, index in kept)]; corrected search per_obj 2 then per_obj 1 on the same domain; plans independently replayed."""
    import p17_mech as m
    f = OUT / "mechanisms" / f"PHYSICAL_{part}.json"
    res = rd(f) if f.exists() else {}
    for fn, i in kept:
        key = f"{fn}#{i}"
        if key in res:
            continue
        x = rd(OUT / "mechanisms" / "screen" / fn)["kept"][i]
        c0 = time.process_time()
        res[key] = {"seed": x["seed"], "tag": x.get("tag"), "spec": x["root"]["spec"], "pose": {k: v.get("min_unrefuted_len") for k, v in x["pose"].items() if k in ("1", "2")},
                    "physical": m.physical(x["root"]), "cpu_s": time.process_time() - c0}
        (OUT / "mechanisms").mkdir(parents=True, exist_ok=True)
        cf.write_json(f, res)


# ------------------------------------------------------------------ P17-3 post-hoc physical mechanism mining (labels -> candidates)
MECH_SPLITS = ("p17train", "p17calib", "train", "dev", "val", "calib")


def _lat(v):
    """Parking lattice (original + parked-subset edits) as d_blockers.minimal_sets over the catalogue pool."""
    import d_blockers as db
    pool = v["catalogue"]["pool"]
    to = lambda st: db.V if st == "VERIFIED" else db.C if st in ("CERTIFIED_INFEASIBLE", "STATIC_INVALID") else None  # noqa: E731
    st = {frozenset(): to(v["original"]["status"])}
    for kind, e in v["edits"].items():
        st[frozenset(pool.index(o) for o in kind.split("+"))] = to(e["status"])
    return db.minimal_sets(len(pool), {k: x for k, x in st.items() if x is not None}), pool


def mine_mechanisms(splits=MECH_SPLITS) -> dict:
    """Mechanism tags derived post hoc from physics labels only (no model output): M0-M4, M6 (pose-level slot preference), M8, M9.
    Every candidate lists its witness sequence and counterfactuals; evidence() re-verifies and replays them."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    import p17_domain as dm
    idx = pl.index_v2()[0]
    out = {m: [] for m in ("M0", "M1", "M2", "M3", "M4", "M6", "M8_3", "M8_4", "M9")}
    for s in splits:
        plan = _plan(s)["roots"]
        for k, v in sorted(merged(s).items()):
            seqs = {a: b for a, b in v["sequences"].items()}
            ver = {a: b for a, b in seqs.items() if b["y"] == 1}
            parse = lambda a: [(x.split(">")[0], int(x.split(">")[1])) for x in a.split(".")]  # noqa: E731
            lat, pool = _lat(v)
            y0 = v["original"]["y"]
            base = {"split": s, "root": k, "family": v["family"], "original": v["original"]["status"], "lattice": lat["type"]}
            if y0 == 1 and pool and any(e["y"] == 1 for e in v["edits"].values()):
                out["M0"].append(base | {"positive": [], "decoys": pool, "counterfactuals": [parse(a) for a in sorted(ver)[:2]]})
            one = {}
            for a in ver:
                if ver[a]["len"] == 1:
                    o, d = parse(a)[0]
                    one.setdefault(o, []).append(d)
            if y0 == 0 and any(len(d) >= 2 for d in one.values()):
                o = min((o for o in one if len(one[o]) >= 2), key=lambda o: (min(ver[f"{o}>{d}"]["travel"] for d in one[o]), o))
                out["M1"].append(base | {"positive": [(o, sorted(one[o], key=lambda d: ver[f"{o}>{d}"]["travel"])[0])], "alternates": sorted(one[o]),
                                         "counterfactuals": [[]]})
            if lat["type"] == "MINIMAL_SETS":
                if any(len(S) >= 2 and all([i] not in lat["sets"] for i in S) for S in lat["sets"]):
                    S = next(S for S in lat["sets"] if len(S) >= 2)
                    out["M2"].append(base | {"set": [pool[i] for i in S], "sets": [[pool[i] for i in S_] for S_ in lat["sets"]],
                                             "provenance": lat["provenance"]})
                if len(lat["sets"]) >= 2:
                    out["M3"].append(base | {"sets": [[pool[i] for i in S] for S in lat["sets"]], "union": [p_ for p_, m in zip(pool, lat["mask"]) if m],
                                             "necessary": [p_ for p_, m in zip(pool, lat["necessary"]) if m], "provenance": lat["provenance"]})
            for a in ver:
                q = parse(a)
                if len(q) == 2 and q[0][0] != q[1][0]:
                    rv = ".".join(f"{o}>{d}" for o, d in q[::-1])
                    if rv in seqs and seqs[rv]["y"] == 0:
                        out["M4"].append(base | {"positive": q, "counterfactuals": [q[::-1]], "reverse_status": seqs[rv]["status"],
                                                 "reverse_certificate": seqs[rv].get("certificate")})
            two = [parse(a) for a in ver if ver[a]["len"] == 2 and parse(a)[0][0] != parse(a)[1][0]]
            if two:
                task = pc.task_from_root(plan[k], k)
                ph = pm.DetPhysics(task, idx)
                c0 = {o.id: o.cell for o in ph.s0.objects}
                sl = dm.slot_fn(ph, idx, width=4, origin=c0)
                for (A, u), (B, w) in two:
                    pref = sl(c0, B)[:1]
                    if pref == [u] and w != u:
                        out["M6"].append(base | {"positive": [(A, u), (B, w)], "counterfactuals": [[(A, u), (B, u)], [(B, u), (A, u)]],
                                                 "b_first_slot_initial": u})
                        break
            for n in (3, 4):
                L = sorted((a for a in ver if ver[a]["len"] == n), key=lambda a: ver[a]["travel"])
                if L and y0 == 0:
                    shorter = sorted({ver[a]["len"] for a in ver if ver[a]["len"] < n})
                    out[f"M8_{n}"].append(base | {"positive": parse(L[0]), "verified_shorter_lengths": shorter, "counterfactuals": [[]]})
            ci = [a for a in seqs if seqs[a]["y"] == 0 and seqs[a].get("certificate")]
            un = [a for a in seqs if str(seqs[a]["status"]).startswith("UNKNOWN")]
            if ci and un:
                out["M9"].append(base | {"certified": [(a, seqs[a]["status"], seqs[a]["certificate"]) for a in ci[:3]],
                                         "unknown": [(a, seqs[a]["status"]) for a in un[:3]],
                                         "q_dependent_unknown": [a for a in un if seqs[a]["status"] == "UNKNOWN_CERTIFIED_TARGET_Q_DEPENDENT"][:3]})
    summ = {m: len(v) for m, v in out.items()}
    cf.write_json(OUT / "mechanisms" / "MINED.json", {"counts": summ, "candidates": out, "splits": list(splits)})
    return summ


def _verified(split, key) -> list:
    v = merged(split)[key]
    return sorted(((a, b["travel"]) for a, b in v["sequences"].items() if b["y"] == 1), key=lambda x: (x[0].count(".") + 1 if x[0] else 0, x[1]))


def mech_exemplars() -> dict:
    """Deterministic exemplar choice per family from MINED.json (physics-only tags) and the M5 mining files; preference rules are fixed
    before evidence is computed: M8 prefers roots whose shortest verified repair has the stated length; M2 / M3 prefer proof-grounded
    lattices with a verified relocation of the clearing set; M5 requires a free alternative endpoint (not a later destination), a colliding
    recorded later sweep and a non-VERIFIED alternative sequence."""
    P = lambda a: [(x.split(">")[0], int(x.split(">")[1])) for x in a.split(".")] if a else []  # noqa: E731
    mined = rd(OUT / "mechanisms" / "MINED.json")["candidates"]
    ex = {}
    for fam in ("M0", "M1", "M4", "M6"):
        c = sorted(mined[fam], key=lambda x: (x["split"] not in ("p17train", "p17calib"), x["split"], x["root"]))
        if c:
            ex[fam] = c[0] | {"positive": c[0].get("positive", []), "counterfactuals": c[0].get("counterfactuals", [])}
    if "M1" in ex:
        o = ex["M1"]["positive"][0][0]
        ex["M1"]["counterfactuals"] = [[]] + [[(o, d)] for d in ex["M1"]["alternates"] if d != ex["M1"]["positive"][0][1]][:2]
    if "M0" in ex:
        ex["M0"]["counterfactuals"] = [[(o, d)] for o, d in (ex["M0"]["counterfactuals"][0][:1] if ex["M0"]["counterfactuals"] else [])]
    for fam in ("M2", "M3"):
        for c in sorted(mined[fam], key=lambda x: (x["provenance"] != "proof_grounded", x["split"], x["root"])):
            ver = _verified(c["split"], c["root"])
            S = c["set"] if fam == "M2" else c["sets"][0]
            pos = next((P(a) for a, _ in ver if a and {o for o, _ in P(a)} == set(S)), None)
            if pos is None:
                continue
            par = merged(c["split"])[c["root"]]["catalogue"]["parking"]
            union = sorted({o for S_ in c["sets"][:2] for o in S_})
            cfs = [[(o, par[o])] for o in S] if fam == "M2" else [[(o, par[o]) for o in c["sets"][1]], [(o, par[o]) for o in union]]
            ex[fam] = c | {"positive": pos, "counterfactuals": cfs}
            break
    for n in (3, 4):
        c = sorted(mined[f"M8_{n}"], key=lambda x: (bool(x["verified_shorter_lengths"]), -min(x["verified_shorter_lengths"] or [0]), x["split"], x["root"]))
        if c:
            ex[f"M8_{n}"] = c[0]
    if mined["M9"]:
        c = sorted(mined["M9"], key=lambda x: (not x["q_dependent_unknown"], x["split"], x["root"]))[0]
        ver = _verified(c["split"], c["root"])
        ex["M9"] = c | {"positive": P(ver[0][0]) if ver else [], "counterfactuals": [P(a) for a, *_ in c["certified"][:2]] + [P(a) for a, *_ in c["unknown"][:2]]}
    m5 = []
    for f in sorted((OUT / "mechanisms").glob("M5_*.json")):
        for k, x in rd(f).items():
            dests = {d for _, d, *_ in x.get("plan", [])}
            for i in x.get("instances", []):
                if i["alt_sequence_status"] != "VERIFIED" and i["slot_alt"] not in dests and i["segments_colliding"] > 0:
                    m5.append((k, x, i))
    if m5:
        k, x, i = sorted(m5, key=lambda t: (not t[2]["alt_sequence_status"].startswith("CERT"), -t[2]["segments_colliding"], t[0]))[0]
        pos = [(o, d) for o, d, *_ in x["plan"]]
        alt = [(o, i["slot_alt"] if j == i["step"] else d) for j, (o, d) in enumerate(pos)]
        ex["M5"] = {"split": k.split("|")[0], "root": k.split("|")[1], "positive": pos, "counterfactuals": [alt], "instance": i,
                    "strong_candidates": len(m5)}
    return ex


def mech_evidence(fams=None) -> None:
    """Evidence package per chosen exemplar (checkpointed by output file)."""
    import p17_mech as m
    ex = mech_exemplars()
    cf.write_json(OUT / "mechanisms" / "EXEMPLARS.json", ex)
    for fam, c in ex.items():
        if fams and fam not in fams or (OUT / "mechanisms" / "evidence" / f"{fam}.json").exists():
            continue
        root = _plan(c["split"])["roots"][c["root"]]
        m.evidence(root, c["root"], [list(a) for a in c["positive"]], [[list(a) for a in s] for s in c["counterfactuals"]], fam,
                   out_dir=str(OUT / "mechanisms" / "evidence"))


def m7_jobs(items, part) -> None:
    """items: [(split, root key, positive sequence | None)]; checkpointed per root."""
    import p17_mech as m
    f = OUT / "mechanisms" / f"M7_LIB_{part}.json"
    res = rd(f) if f.exists() else {}
    for s, k, pos in items:
        if f"{s}|{k}" in res:
            continue
        res[f"{s}|{k}"] = m.m7_library(_plan(s)["roots"][k], k, pos)
        cf.write_json(f, res)


# ------------------------------------------------------------------ shared-domain proposal recall on small exhaustive scenes (contract 5.2 / 6.3)
def recall_candidates(n=12, max_movers=3) -> list:
    """Labelled development roots with a decided infeasible original and <= max_movers movers (plan order; no outcome-based choice)."""
    out = []
    for s in ("dev", "train", "calib", "val", "p17train"):
        for k, v in sorted(merged(s).items()):
            if v["original"]["y"] == 0 and len(v["catalogue"]["movers"]) <= max_movers:
                out.append((s, k))
    return out[:n]


def proposal_recall(items, part, max_moves=3, max_seqs=400) -> None:
    """Exhaustive enumeration of the B generator's domain (p17_domain.exhaustive_sequences; <= max_moves for <= 2 movers, else max_moves - 1)
    with existential verification of
    every sequence (P17_SEQ_BUDGET_V1, shared exact cache); recall of the capped B catalogue over certified solutions (all / per length /
    minimal length); corrected D search (uniform order, width-4 domain) minimum moves vs the exhaustive minimum."""
    import p15_lookup as pl
    import p16_common as pc
    import p16_motion as pm
    import p17_b as pb
    import p17_domain as dm
    import p17_exec as ex
    import p17_verify as pv
    idx = pl.index_v2()[0]
    f = OUT / "shared" / f"PROPOSAL_RECALL_{part}.json"
    res = rd(f) if f.exists() else {}
    for s, k in items:
        if f"{s}|{k}" in res:
            continue
        c0 = time.process_time()
        task, cache = pc.task_from_root(_plan(s)["roots"][k], k), {}
        fac = lambda: pm.DetPhysics(task, idx, budget=pb.SEQ_BUDGET, cache=cache)  # noqa: E731
        base = fac()
        mm = max_moves if len(dm.movers(task, {o.id: o.cell for o in base.s0.objects})[0]) <= 2 else max_moves - 1
        seqs = [q for q in dm.exhaustive_sequences(task, base, idx, max_moves=mm) if q]
        if len(seqs) > max_seqs:
            res[f"{s}|{k}"] = {"skipped": f"{len(seqs)} sequences > {max_seqs}"}
            cf.write_json(f, res)
            continue
        st = {}
        for q in seqs:
            v = pv.verify_existential(fac(), [tuple(a) for a in q], first=True)
            st[pb.sig(q)] = (v["status"], len(q))
        cat = {pb.sig(q) for q in pb.catalogue(task, base, idx)["sequences"] if q}
        cert = {a for a, (x, _) in st.items() if x == pv.V}
        L = min((n for a, (x, n) in st.items() if x == pv.V), default=None)
        d = ex.d_root(task, idx, ["uniform"], cache)["uniform"]
        res[f"{s}|{k}"] = {"sequences": len(seqs), "certified": len(cert), "unknown": sum(str(x).startswith("UNKNOWN") for x, _ in st.values()),
                           "catalogue": len(cat), "recall_all": len(cert & cat) / len(cert) if cert else None,
                           "max_moves": mm, "recall_by_len": {n: [len({a for a in cert & cat if st[a][1] == n}), len({a for a in cert if st[a][1] == n})] for n in range(1, mm + 1)},
                           "min_len_exhaustive": L, "min_len_in_catalogue": bool(L and any(st[a][1] == L for a in cert & cat)),
                           "d_search": {"found": d["found"], "moves": d["moves"], "claim": d["claim"]}, "cpu_s": time.process_time() - c0}
        cf.write_json(f, res)
