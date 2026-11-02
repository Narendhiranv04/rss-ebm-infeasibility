"""P10a: frozen plan, static admission, prediction freeze, capped teacher collection, adjudication and contrasts.

Budgets (hard, guarded, ledgered in runs/p10/guards): construction (static) <= 600 CPU-s; teacher <= 5,000 CPU-s including
static admission, coordinator, guard overhead and every seed attempt; <= 14 roots, <= 16 queries per root, <= 224 queries.
Per query: <= 120 CPU-s (one guarded seed <= 100 CPU-s plus a 10 s orchestration reserve; a later seed starts only while
spent + 110 <= 120) and <= 360 s wall. This is TIGHTER than OPNEG-v1 (250 CPU-s per seed): a cut seed is CPU_CENSORED and the
query becomes UNKNOWN, never BUDGET_INFEASIBLE. Stop rule (predeclared): if a root's ANCHOR query is not VERIFIED, its remaining
planner-requiring queries are NOT_RUN (statically certified queries are still recorded; they cost no planner call).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import config as cf
import geometry as ge
import operational_teacher as ot
import p10_scenes as ps
from scene import scene_from_dict, scene_hash, scene_to_dict

OUT = Path("runs/p10/p10a")
CAPS = {"roots": 14, "queries_per_root": 16, "queries": 224, "teacher_cpu_s": 5000.0, "query_cpu_s": 120.0, "seed_cpu_s": 100.0,
        "reserve_s": 10.0, "query_wall_s": 360.0, "aggregate_wall_s": 14400.0}
HIST_PLANS = ("runs/p5/d3lite/plan.json", "runs/p6/collect/plan.json", "runs/p7/collection/plan.json", "runs/p8/collection/plan.json")


def _rd(p):
    return json.loads(Path(p).read_text())


def _known():
    import d3lite as d3
    import g4c
    known = d3._known(Path("."))
    for f in HIST_PLANS:
        for r in _rd(f)["roots"].values():
            known.add(g4c.stable_geometry(scene_from_dict(r["scene"]), r["goal_cell"]))
    return known


def construct_dev() -> None:
    """Static dry run (no teacher, no plan written): constructibility of every mechanism and rejection reasons."""
    import collections
    cfg = cf.load_config("config.toml")
    roots, ledger, cons = ps.construct(cfg, ge.make_robot(cfg), _known())
    why = collections.defaultdict(collections.Counter)
    for x in ledger:
        if not x["accepted"]:
            why[next(s["mechanism"] for s in ps.ROOTS if s["seed"] == x["seed"])][x["reason"]] += 1
    OUT.mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "construction_dev.json", {"accepted": [{k: r[k] for k in ("index", "mechanism", "wave", "attempt", "stratum", "goal_cell", "role_cells")}
                                                               for r in roots], "rejections": {k: dict(v.most_common(6)) for k, v in why.items()}, **cons})


def probe(n=120, only=None) -> None:
    """Cheap diagnostic: n attempts per mechanism (seed of its wave-1 root), rejection reasons and CPU per attempt."""
    import collections
    cfg = cf.load_config("config.toml")
    f = OUT / f"construction_probe_n{n}.json"
    cx, out = ps.Ctx(cfg, ge.make_robot(cfg)), (_rd(f) if f.exists() else {})   # merge: earlier mechanisms' probes are kept
    for spec in [s for s in ps.ROOTS if (s["wave"] == 1 or s["variant"] == "kappa0") and (only is None or s["mechanism"] in only)]:
        c0, why = time.process_time(), collections.Counter()
        for a in range(n):
            rec, w = ps._try(cx, spec, a, set())
            why["ACCEPTED" if rec else w] += 1
        out[f"{spec['mechanism']}{spec['variant']}"] = {"cpu_per_attempt": round((time.process_time() - c0) / n, 4), "reasons": dict(why.most_common(8))}
        OUT.mkdir(parents=True, exist_ok=True)
        cf.write_json(f, out)                                               # incremental: survives a guard stop


def build_plan() -> None:
    """Construct the 14 roots and freeze plan.json (catalogues, query ids, priorities, seed units) before any teacher call."""
    import data
    import zlayout_r4 as z4
    cfg, pol = cf.load_config("config.toml"), ot.load_policy("teacher_opneg.toml")
    robot = ge.make_robot(cfg)
    if (OUT / "plan.json").exists():
        raise SystemExit("P10a plan already frozen; refusing to overwrite")
    roots, ledger, cons = ps.construct(cfg, robot, _known())
    plan = {"stage": "P10A", "version": ps.VERSION, "policy_id": pol["policy"]["policy_id"], "policy_hash": pol["policy_hash"],
            "seeds_policy": pol["policy"]["seeds"], "root_schedule": ps.ROOTS, "roots": {}, "queries": [], "units": [], "ineligible": [],
            "config_hash": cf.config_hash(cfg), "robot_profile_hash": cf.robot_profile_hash(cfg), "caps": CAPS, "z_layout_version": z4.VERSION,
            "stop_rule": "anchor not VERIFIED -> remaining planner-requiring queries of the root NOT_RUN (certified ones still recorded)"}
    for r in roots:
        cat, k, s = ps.catalogue(cfg, robot, r), r["root"], r["scene"]
        meta = {x: r[x] for x in ("mechanism", "variant", "wave", "seed", "attempt", "stratum", "goal_cell", "ids", "dest", "role_cells", "fixed",
                                  "predicates", "construction")}
        q0 = data.make_query(cfg, k, s, "t0", r["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
        plan["roots"][k] = {"scene": scene_to_dict(s), "scene_hash": scene_hash(s), "hashes": q0["hashes"], "lineage": k, "variant_kind": "obstructed",
                            "split": "P10A_PILOT", "batch": r["wave"], "family": r["mechanism"], **meta,
                            "queries": {"scene_queries": cat["scene_queries"], "pool": [i for i in cat["items"] if i["kind"] == "ORDERED_TEACHER"]}}
        for it in cat["items"]:
            sq = it["kind"] == "SCENE_TARGET_ONLY"
            if sq and not it["eligible"]:
                plan["ineligible"].append({"root": k, "edit_kind": it["edit_kind"], "edit": it["edit"]})
                continue
            sig = z4.sig(sorted(it["edit"].items())) if sq else z4.sig([tuple(x) for x in it["sequence"]])
            qid = f"{'scene' if sq else 'seq'}:{k}:{sig}"
            plan["queries"].append({"query": qid, "root": k, "lineage": k, "kind": it["kind"], "sequence": [] if sq else it["sequence"],
                                    "edit": it.get("edit") if sq else None, "edit_kind": it.get("edit_kind"), "priority": it["priority"],
                                    "anchor": it["anchor"], "mechanism": r["mechanism"], "wave": r["wave"], "split": "P10A_PILOT", "batch": r["wave"],
                                    "family": r["mechanism"], "role": "anchor" if it["anchor"] else "catalogue"})
            for sd in pol["policy"]["seeds"]:
                key = pol["policy"]["seed_key"].format(root=k, sig=("scene/" if sq else "") + sig, seed=sd)
                u = {"unit": f"{qid}#s{sd}", "query": qid, "seed": sd, "kind": it["kind"], "root": k, "target_key": key + "/target",
                     "relocation_key": key + "/reloc"}
                u.update({"edit": it["edit"], "eligible": True, "intervened_scene_hash": it["intervened_scene_hash"]} if sq else {"sequence": it["sequence"]})
                plan["units"].append(u)
    per = max(sum(q["root"] == k for q in plan["queries"]) for k in plan["roots"]) if plan["roots"] else 0
    if len(plan["roots"]) > CAPS["roots"] or per > CAPS["queries_per_root"] or len(plan["queries"]) > CAPS["queries"]:
        raise SystemExit("frozen plan exceeds the authorized P10a caps")
    plan["plan_hash"] = cf.stable_hash({x: v for x, v in plan.items() if x != "plan_hash"})
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "construction_attempts.jsonl", "w") as fh:
        fh.writelines(json.dumps(x) + "\n" for x in ledger)
    cf.write_json(OUT / "construction.json", cons)
    cf.write_json(OUT / "plan.json", plan)


# ------------------------------------------------------------------ static admission and capped collection (after the prediction freeze)
GATE = ("wave-1 yield gate per mechanism (frozen before any teacher call): its wave-1 root has no FATAL; the anchor is decided; >= 1 decided "
        "primary contrast (H1/H2 complete kappa four-tuple; H3 both orders; H4 the two-move repair and >= 1 omission; H5 the empty repair and "
        ">= 1 edit; H7 edit A and the executable M relocation; H8 the anchor); for H1 and H3 that contrast must contain a SOUND negative "
        "(CERTIFIED_INFEASIBLE; operational negatives never substitute); UNKNOWN + NOT_RUN <= 50 % of the root's queries. "
        "Wave 2 runs a mechanism's root only if its wave-1 gate passes and budget remains.")
PRIMARY = {"H1": [["edit:A+B"], ["edit:A"], ["edit:B"], [""]], "H2": [["edit:A+B"], ["edit:A"], ["edit:B"], [""]], "H3": [["B A"], ["A B"]],
           "H4": [["C A"], ["A", "C"]], "H5": [[""], ["edit:A", "edit:B", "edit:A+B"]], "H7": [["edit:A"], ["M"]], "H8": [["M"]]}


def _xpol(pol):
    """Execution limits for P10a: TIGHTER than OPNEG-v1 (seed guard 100 CPU-s, kill at 95; reserve 10; a later seed only while
    spent + 110 <= 120). Adjudication always uses the unchanged policy, so a cut seed is censored -> UNKNOWN, never BUDGET."""
    import copy
    x = copy.deepcopy(pol)
    x["policy"].update(per_seed_cpu_s=CAPS["seed_cpu_s"], per_seed_guard_margin_s=5.0, per_seed_wall_s=CAPS["query_wall_s"])
    x["pilot"]["orchestration_reserve_s"] = CAPS["reserve_s"]
    return x


def admit() -> None:
    import teacher_pilot as tp
    if not (OUT / "predictions.json").exists():
        raise SystemExit("predictions not frozen: static certificates are teacher outcomes")
    if (OUT / "static_admission.json").exists():
        raise SystemExit("static admission already done")
    cfg, plan, c0 = cf.load_config("config.toml"), _rd(OUT / "plan.json"), time.process_time()
    static = tp.static_admission(cfg, ge.make_robot(cfg), plan, use_sector=True)
    cf.write_json(OUT / "static_admission.json", {"certificates": static, "cpu_s": round(time.process_time() - c0, 2)})


def _tok(plan, q):
    """Catalogue token of a planned query (same grammar as p10_scenes.CATALOGUE)."""
    r = plan["roots"][q["root"]]
    if q["kind"] == "SCENE_TARGET_ONLY":
        return "edit:" + q["edit_kind"]
    inv = {v: k for k, v in r["ids"].items()}
    return " ".join(inv[o] + ("*" if d != r["dest"][inv[o]][0] else "") for o, d in q["sequence"])


def yield_gate(plan, recs, mech) -> dict:
    k = next(k for k, r in plan["roots"].items() if r["mechanism"] == mech and r["wave"] == 1)
    mine = {_tok(plan, r["query"]): r for r in recs.values() if r["query"]["root"] == k}
    dec = lambda t: t in mine and mine[t]["binary_training_label"] is not None  # noqa: E731
    anchor = next(r for r in mine.values() if r["query"]["anchor"])
    miss = sum(r["teacher_status"] in (ot.UNKNOWN, "NOT_RUN") for r in mine.values())
    c = {"no_fatal": not any(r["teacher_status"] == "FATAL_CONTRADICTION" for r in mine.values()),
         "anchor_decided": anchor["binary_training_label"] is not None,
         "primary_contrast": all(any(dec(t) for t in grp) for grp in PRIMARY[mech]), "missing_le_50pct": miss <= 0.5 * len(mine)}
    if mech in ("H1", "H3"):
        c["sound_negative"] = any(t in mine and mine[t]["teacher_status"] == ot.CERTIFIED for grp in PRIMARY[mech] for t in grp)
    return {"root": k, "checks": c, "pass": all(c.values()), "missing": miss, "queries": len(mine)}


def collect(chunk_cpu=1700.0) -> None:
    """Resumable: one guarded job processes queries until its chunk budget is spent; rerun to continue."""
    import g4d_run as gr
    import p6_run as p6r
    import teacher_pilot as tp
    cfg, v2, pol = cf.load_config("config.toml"), cf.load_config_v2("config_v2_r4.toml"), ot.load_policy("teacher_opneg.toml")
    plan, robot, xp = _rd(OUT / "plan.json"), ge.make_robot(cfg), _xpol(pol)
    static = _rd(OUT / "static_admission.json")["certificates"]
    if v2["proposal"]["move_cap"] != 4 or ot.config_mismatch(pol, cfg) or pol["policy_hash"] != plan["policy_hash"]:
        raise SystemExit("P10a requires the R4 overlay and the unchanged OPNEG-v1 policy")
    prog = _rd(OUT / "progress.json") if (OUT / "progress.json").exists() else {"done": {}, "gates": {}, "wall_s": 0.0, "xpol": xp["policy"],
                                                                                "gate_rule": GATE}
    import p10_protocol as pp
    before, c0, w0, wall0 = pp.spent("teacher")[0], time.process_time(), time.time(), prog["wall_s"]
    paths = {"plan": OUT / "plan.json", "config": Path("config.toml"), "config_v2": Path("config_v2_r4.toml")}
    mine_cpu = lambda: sum(d["unit_cpu_s"] for d in prog["done"].values() if d.get("job_utc") == w0) + time.process_time() - c0  # noqa: E731
    recs = lambda: {q["query"]: p6r.judge(static, pol, q, prog["done"].get(q["query"])) for q in plan["queries"]}  # noqa: E731
    for wv in (1, 2):
        if wv == 2:
            for mech in ps.MECH:
                prog["gates"].setdefault(mech, yield_gate(plan, recs(), mech))
            cf.write_json(OUT / "progress.json", prog)
        for k, r in plan["roots"].items():
            if r["wave"] != wv or (wv == 2 and not prog["gates"][r["mechanism"]]["pass"]):
                continue
            for q in sorted([q for q in plan["queries"] if q["root"] == k], key=lambda q: q["priority"]):
                qid = q["query"]
                if qid in prog["done"]:
                    continue
                left = CAPS["teacher_cpu_s"] - before - mine_cpu()
                if static.get(qid) is None and (mine_cpu() > chunk_cpu or time.time() - w0 > 3600 - 2 * CAPS["query_wall_s"] or wall0 + time.time() - w0 > CAPS["aggregate_wall_s"] - CAPS["query_wall_s"]):
                    prog["wall_s"] = wall0 + time.time() - w0
                    cf.write_json(OUT / "progress.json", prog)
                    return
                anc = [d for d in prog["done"].values() if d["root"] == k and d["anchor"]]
                stop = "root stopped: anchor not VERIFIED" if anc and anc[0]["status"] != ot.VERIFIED else None
                qdir = OUT / "q" / qid.replace(":", "__").replace(">", "-")
                if stop is None and static.get(qid) is None and qdir.exists():
                    stop = "interrupted before this query was recorded; never repeated"
                cap = min(CAPS["query_cpu_s"], left)
                if stop is None and static.get(qid) is None and cap < CAPS["seed_cpu_s"] + CAPS["reserve_s"]:
                    stop = f"teacher budget remainder {cap:.0f} CPU-s below one guarded seed"
                c1, seeds, uc = time.process_time(), {qid: []}, 0.0
                if stop is None and static.get(qid) is None:
                    seeds, uc = tp.run_rounds(plan, xp, qdir, paths, cfg, v2, robot, static, [qid], cap, CAPS["query_wall_s"], seeds)
                d = {"root": k, "batch": wv, "anchor": q["anchor"], "role": q["role"], "stop": stop, "attempts": seeds[qid], "unit_cpu_s": uc,
                     "coord_cpu_s": time.process_time() - c1, "job_utc": w0, "status": None}
                prog["done"][qid] = d
                j = p6r.judge(static, pol, q, d)
                d["status"] = j["teacher_status"]
                u0 = next(u for u in plan["units"] if u["query"] == qid)
                with open(OUT / "teacher_calls.jsonl", "a") as fh:            # append-only LABEL_ACQUISITION ledger
                    fh.write(json.dumps({"caller_stage": "LABEL_ACQUISITION", "root_id": k, "query_id": qid, "kind": q["kind"],
                                         "input_hash": cf.stable_hash({"scene_hash": r["scene_hash"], "unit": {x: v for x, v in u0.items() if x != "seed"}}),
                                         "status": j["teacher_status"], "provenance": j["label_source"], "static_certificate": static.get(qid),
                                         "stop": stop, "seeds": [{x: a.get(x) for x in ("seed", "stop", "terminal", "clean", "cpu_s", "wall_s", "witness_hash")}
                                                                 for a in d["attempts"]], "unit_cpu_s": round(uc, 2), "coord_cpu_s": round(d["coord_cpu_s"], 2),
                                         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "plan_hash": plan["plan_hash"]}) + "\n")
                prog["wall_s"] = wall0 + time.time() - w0
                cf.write_json(OUT / "progress.json", prog)
    prog["finished"], prog["wall_s"] = True, wall0 + time.time() - w0
    cf.write_json(OUT / "progress.json", prog)
