"""G4C-D2-P1 POC_S: finalization, budget admission, guarded collection and the D2 judge.

The plan is REBUILT deterministically from the P0 construction code and must equal the
saved P0 artifacts (variant scene hashes and the full query catalogue) for the two POC_S
families x six variants; any difference refuses finalization. Units use the G4D runner
schema and run through the unchanged g4d_run.run_collection (process-tree CPU guard, one
worker, atomic checkpoints, revalidation). Two predeclared fresh-base diagnostics (seeds
fixed now, G4C J0 rule, one target evaluation each) share the same ledger and budget and
never add descendants. The judge reuses g4c_pilot.adjudicate / judge_root / bc_pool on
evidence rebuilt by g4d_run.unit_evidence; no teacher call is made by the judge.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np

import config as cf
import d2
import data
import effects as ef
import g4c
import g4c_pilot as gp
import g4d_run as gr
import geometry as ge
import sequences as sq
from scene import place_scene, scene_hash, scene_to_dict

P, N, U = data.POSITIVE, data.NEGATIVE, data.UNKNOWN
P0 = Path("runs/g4c_d2_precheck")
FAMILIES = ("g4d_R1", "g4c_r00")
VARIANTS = ("base", "+A", "+A+B", "+A+S", "+G", "+N")
FRESH_SEEDS = (2026102101, 2026102102)
AUTH = {"scope": "G4C_D2_P1_POC_S_ONLY", "units": 68, "ordered": 50, "scene": 16, "robot_start": 2, "fresh_base_attempts": 2,
        "per_teacher_unit_cpu_s": 900, "cancellation_margin_s": 30, "aggregate_cpu_s": 18000, "workers": 1,
        "unit_wall_guard_s": 1800, "aggregate_wall_guard_s": 23400, "max_motion_nodes": 4000,
        "relocation_skill_attempt_cap": 46, "target_eval_attempt_cap": 70, "max_root_construction_attempts": 2}
RESERVE_S = 60.0
TRIVIAL = {"GOAL_OVERLAP", "STATIC_ENDPOINT_INVALID"}
WORD = {P: "POSITIVE", N: "NEGATIVE", U: "UNKNOWN"}       # d2 release rules use the abstract words
SOURCES = gr.SOURCES[:-1] + ("g4c.py", "g4c_pilot.py", "d2.py", "d2_poc.py", "runs/g4c_d2_precheck/precheck_queries.json",
                             "runs/g4c_d2_precheck/precheck_variants.json", "runs/g4c_d2_precheck/manifest.json")


def _sha(p) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rkey(fam: str, v: str) -> str:
    return f"d2_{fam}_{v.replace('+', '') or 'base'}"


# ------------------------------------------------------------------ authorization and the frozen plan
def authorization_record(instruction: Path) -> dict:
    rec = {**AUTH, "families": list(FAMILIES), "variants": list(VARIANTS), "fresh_seeds": list(FRESH_SEEDS),
           "orchestration_reserve_s": RESERVE_S, "instruction_sha256": _sha(instruction), "p0_manifest_sha256": _sha(P0 / "manifest.json"),
           "granted_by": "user message of 20 October 2026: 'I explicitly authorize G4C-D2-P1 POC_S ONLY'",
           "excluded": "POC_F (194 units), the remaining 15 G4C roots, new certificate kinds, training, 3D, hardware"}
    rec["authorization_hash"] = cf.stable_hash(rec)
    return rec


def check_authorization(a: dict) -> list:
    why = [f"{k}={a.get(k)!r} != {v!r}" for k, v in AUTH.items() if a.get(k) != v]
    if a.get("families") != list(FAMILIES) or a.get("variants") != list(VARIANTS) or a.get("fresh_seeds") != list(FRESH_SEEDS):
        why.append("scope differs from POC_S")
    if cf.stable_hash({k: v for k, v in a.items() if k != "authorization_hash"}) != a.get("authorization_hash"):
        why.append("authorization record modified")
    return why


def fresh_base(cfg, robot, seed: int, known: set) -> dict:
    """G4C J0 construction rule (static geometry only) for ONE predeclared seed; a failing seed
    consumes its attempt (no reseeding)."""
    lay = g4c._layout(np.random.default_rng(seed), "J0")
    if lay is None:
        return {"seed": seed, "accepted": False, "reason": "J0 intent predicate not met"}
    t, g, _, obs = lay
    s = place_scene(cfg, f"d2_fresh_{seed}", "initial", [t], obs)
    d, e, _ = d2._tol(cfg)
    if ge.Checker(robot, ge.make_world(s), d, e).status_at(np.asarray(s.robot.q_rad), s.robot.gripper_gap_m).status != ge.CLEAR:
        return {"seed": seed, "accepted": False, "reason": "initial robot not clear"}
    if g4c.stable_geometry(s, g) in known:
        return {"seed": seed, "accepted": False, "reason": "duplicate geometry"}
    return {"seed": seed, "accepted": True, "scene": s, "goal_cell": g, "target_cell": t}


def _root(cfg, k, s, g, extra):
    q = data.make_query(cfg, k, s, "t0", g, {}, "SAFE_RETREAT", data.PROTOCOL_V2)
    return {"scene": scene_to_dict(s), "scene_hash": scene_hash(s), "hashes": q["hashes"], "goal_cell": g, **extra}


def build_plan(cfg, robot, root: Path = Path(".")) -> dict:
    """Rebuild the POC_S roots/units from P0 code and require equality with the P0 artifacts."""
    pv, pq = json.loads((root / P0 / "precheck_variants.json").read_text()), json.loads((root / P0 / "precheck_queries.json").read_text())
    roots, units, ledger, why = {}, [], [], []
    for spec in [b for b in d2.BASES if b[0] in FAMILIES]:
        base = d2.load_base(spec, root)
        rep = d2.replay_base(robot, cfg, base)
        if rep["status"] != "BASE_WITNESS_REPLAYED":
            why.append(f"{base['family']}: archived base witness no longer replays")
        ch = d2.choose_blockers(d2.insertion_attempts(robot, cfg, base))
        vs = {v["variant"]: v for v in d2.nested_variants(robot, cfg, base, ch)}
        saved = {v["variant"]: v for v in pv["variants"][base["family"]]}
        for name in VARIANTS:
            v, k = vs[name], rkey(base["family"], name)
            if v["status"] != "PRECHECK_STATIC" or v["scene_hash"] != saved[name]["scene_hash"]:
                why.append(f"{k}: variant differs from P0")
                continue
            Q = d2.variant_queries(robot, cfg, v)
            if json.loads(json.dumps(Q)) != pq[base["family"]][name]:
                why.append(f"{k}: query catalogue differs from P0")
            act = data.make_query(cfg, k, v["scene"], "t0", v["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)["learner"]["action"]
            sqs = []
            for q in Q["scene_queries"]:
                sub = ef.pose_substitution(robot, cfg, v["scene"], act, q["edit"])
                sqs.append({**q, "intervened_scene_hash": sub["intervened_scene_hash"]})
            cat = Q["catalogue"]
            roots[k] = _root(cfg, k, v["scene"], v["goal_cell"], {
                "family": f"D2:{name}", "base_family": base["family"], "variant": name, "lineage": base["lineage"],
                "family_id": base["family_id"], "split": d2.family_split(base["family_id"]),
                "construction_privileged": {"added": v["added"], "nominal": v["nominal"]},
                **({"archived_base_replay": rep, "archived_witness_sha256": _sha(root / spec[3])} if name == "base" else {}),
                "queries": {"pool": Q["pool"], "scene_queries": sqs,
                            "catalogue": {"queries": cat, "hash": cf.stable_hash(cat)}}})
            for p in Q["pool"]:
                sig = g4c._sig(p["sequence"])
                units.append({"unit": f"seq:{k}:{sig}", "kind": "ORDERED_TEACHER", "root": k, "sequence": p["sequence"],
                              "relocation_key": f"g4c_d2/{k}/{sig}/reloc", "target_key": f"g4c_d2/{k}/{sig}/target",
                              "relocation_skills_max": len(p["sequence"]), "target_evaluations_max": 1})
            for q in sqs:
                if q["eligible"]:
                    sig = g4c._sig(sorted(q["edit"].items()))
                    units.append({"unit": f"scene:{k}:{sig}", "kind": "SCENE_TARGET_ONLY", "root": k, "edit": q["edit"],
                                  "eligible": True, "intervened_scene_hash": q["intervened_scene_hash"],
                                  "reference_unit": f"seq:{k}:empty", "target_key": f"g4c_d2/{k}/scene/{sig}/target",
                                  "relocation_skills_max": 0, "target_evaluations_max": 1})
            for st in [x for x in Q["robot_starts"] if x["eligible"]][:1]:
                rs = ef.robot_start_substitution(robot, cfg, v["scene"], act, st["q_start"])
                units.append({"unit": f"start:{k}:{st['rule_index']}", "kind": "ROBOT_START_TARGET_ONLY", "root": k,
                              "q_start": st["q_start"], "intervened_scene_hash": rs["intervened_scene_hash"],
                              "reference_unit": f"seq:{k}:empty", "target_key": f"g4c_d2/{k}/start{st['rule_index']}/target",
                              "relocation_skills_max": 0, "target_evaluations_max": 1})
    known = {g4c.stable_geometry(scene_from(r), r["goal_cell"]) for r in roots.values()}
    for doc in ("runs/g4c_preflight/plan.json", "runs/g4d_evidence/final_plan.json"):
        known |= {g4c.stable_geometry(scene_from(r), r["goal_cell"]) for r in json.loads((root / doc).read_text())["roots"].values()
                  if "scene" in r}
    fresh = []
    for seed in FRESH_SEEDS:
        fb = fresh_base(cfg, robot, seed, known)
        ledger.append({k: v for k, v in fb.items() if k not in ("scene",)})
        if fb["accepted"]:
            k = f"d2_fresh_{seed}"
            roots[k] = _root(cfg, k, fb["scene"], fb["goal_cell"], {"family": "D2:fresh_base", "base_family": k, "variant": "fresh",
                                                                  "lineage": f"fresh:{seed}", "family_id": cf.stable_hash(f"fresh:{seed}"),
                                                                  "queries": {"pool": [{"sequence": [], "origin": "declared",
                                                                                        "static_status": "VALID", "endpoint_distance": 0.0,
                                                                                        "bc_layout": g4c.layout([], [])}],
                                                                              "scene_queries": [], "catalogue": {"queries": [], "hash": cf.stable_hash([])}}})
            fresh.append({"unit": f"seq:{k}:empty", "kind": "ORDERED_TEACHER", "root": k, "sequence": [], "fresh_base": True,
                          "relocation_key": f"g4c_d2/{k}/empty/reloc", "target_key": f"g4c_d2/{k}/empty/target",
                          "relocation_skills_max": 0, "target_evaluations_max": 1})
    return {"roots": roots, "units": fresh + units, "frozen_units": [u["unit"] for u in units],
            "fresh_units": [u["unit"] for u in fresh], "fresh_ledger": ledger, "errors": why}


def scene_from(r):
    from scene import scene_from_dict
    return scene_from_dict(r["scene"])


UNIT_KEYS = {"unit", "kind", "root", "sequence", "edit", "eligible", "q_start", "intervened_scene_hash", "reference_unit", "target_key",
             "relocation_key", "relocation_skills_max", "target_evaluations_max", "fresh_base"}


def validate_plan(cfg, robot, plan: dict, auth: dict) -> list:
    """Every structural requirement of the frozen POC_S plan ([] = admissible). Used at
    finalization AND immediately before collection; never consults an outcome."""
    why = check_authorization(auth)
    roots, units = plan["roots"], plan["units"]
    frozen = [u for u in units if not u.get("fresh_base")]
    fresh = [u for u in units if u.get("fresh_base")]
    kinds = [sum(u["kind"] == k for u in frozen) for k in ("ORDERED_TEACHER", "SCENE_TARGET_ONLY", "ROBOT_START_TARGET_ONLY")]
    if len(frozen) != auth["units"] or kinds != [auth["ordered"], auth["scene"], auth["robot_start"]]:
        why.append(f"unit schedule {kinds} ({len(frozen)}) differs from 50/16/2 = 68")
    if [u["unit"] for u in frozen] != plan["frozen_units"] or [u["unit"] for u in fresh] != plan["fresh_units"]:
        why.append("frozen unit order differs from the declared order")
    if len(fresh) > auth["fresh_base_attempts"] or len(plan["fresh_ledger"]) > auth["max_root_construction_attempts"] or \
            any(u["sequence"] or u["relocation_skills_max"] or u["target_evaluations_max"] != 1 for u in fresh):
        why.append("fresh-base attempts exceed the authorization")
    want = {rkey(f, v) for f in FAMILIES for v in VARIANTS} | {u["root"] for u in fresh}
    if set(roots) != want or {(r["base_family"], r["variant"]) for r in roots.values() if r["variant"] != "fresh"} != \
            {(f, v) for f in FAMILIES for v in VARIANTS}:
        why.append(f"roots differ from POC_S 2 x 6 variants: {sorted(set(roots) ^ want)}")
    for k, r in roots.items():
        if r["variant"] != "fresh" and (r["family_id"] != cf.stable_hash(r["lineage"].split(" ")[0]) or
                                        r["split"] != d2.family_split(r["family_id"])):
            why.append(f"{k}: lineage/split binding broken")
        if r["queries"]["catalogue"]["hash"] != cf.stable_hash(r["queries"]["catalogue"]["queries"]):
            why.append(f"{k}: catalogue changed")
    if len({r["family_id"] for r in roots.values() if r["variant"] != "fresh"}) != len(FAMILIES):
        why.append("lineage families are not exactly the two POC_S families")
    for f in FAMILIES:
        fam = [scene_from(r) for r in roots.values() if r["base_family"] == f]
        if len({(tuple(s.robot.q_rad), s.robot.gripper_gap_m) for s in fam}) != 1:
            why.append(f"{f}: robot state differs between variants")
    rel, tgt = sum(u["relocation_skills_max"] for u in units), sum(u["target_evaluations_max"] for u in units)
    if rel > auth["relocation_skill_attempt_cap"] or tgt > auth["target_eval_attempt_cap"]:
        why.append(f"worst-case calls {rel} relocation / {tgt} target exceed {auth['relocation_skill_attempt_cap']}/{auth['target_eval_attempt_cap']}")
    if cfg["planner2d"]["max_nodes"] != auth["max_motion_nodes"]:
        why.append("planner node budget differs from 4000")
    if len({u["unit"] for u in units}) != len(units):
        why.append("duplicate unit identity")
    for u in units:
        if set(u) - UNIT_KEYS:
            why.append(f"{u['unit']}: non-schema (possibly teacher) fields {sorted(set(u) - UNIT_KEYS)}")
            continue
        try:
            scene, seq, query, _ = gr.unit_scene(cfg, robot, {"roots": roots}, u)
        except (SystemExit, KeyError) as e:
            why.append(f"{u['unit']}: {e}")
            continue
        if u["kind"] == "ORDERED_TEACHER":
            canon, bad = sq.parse_sequence(scene, query["learner"]["action"], seq, 3)
            pool = [p["sequence"] for p in roots[u["root"]]["queries"]["pool"]]
            if bad or [list(a) for a in canon] != seq or seq not in pool or u["relocation_skills_max"] != len(seq) or \
                    u["unit"] != f"seq:{u['root']}:{g4c._sig(seq)}":
                why.append(f"{u['unit']}: noncanonical or undeclared sequence {bad}")
    return why


def finalize(cfg, auth: dict, root: Path = Path(".")) -> dict:
    """Immutable POC_S plan; SystemExit (before any teacher call) on any failure."""
    robot = ge.make_robot(cfg)
    b = build_plan(cfg, robot, root)
    if b["errors"]:
        raise SystemExit(f"G4C-D2-P1 finalization refused (zero teacher calls): {b['errors']}")
    rel, tgt = sum(u["relocation_skills_max"] for u in b["units"]), sum(u["target_evaluations_max"] for u in b["units"])
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()  # noqa: E731
    plan = {"stage": "G4C_D2_P1_POC_S", "protocol": data.PROTOCOL_V2, "construction_mode": d2.VERSION, "status": "FINAL_AUTHORIZED",
            "roots": b["roots"], "units": b["units"], "frozen_units": b["frozen_units"], "fresh_units": b["fresh_units"],
            "fresh_ledger": b["fresh_ledger"], "authorization_hash": auth["authorization_hash"],
            "p0_manifest_sha256": _sha(root / P0 / "manifest.json"),
            "limits": {"cpu_per_unit_s": 900.0, "guard_margin_s": 30.0, "cpu_aggregate_s": 18000.0, "workers": 1,
                       "wall_per_unit_s": 1800.0, "wall_total_s": 23400.0, "poll_s": 0.25, "orchestration_reserve_s": RESERVE_S},
            "call_caps": {"relocation_skills": 46, "target_evaluations": 70, "worst_case": {"relocation_skills": rel, "target_evaluations": tgt}},
            "threads": gr.teacher.THREADS, "config_hash": cf.config_hash(cfg), "robot_profile_hash": cf.robot_profile_hash(cfg),
            "source": {"head": git("rev-parse", "HEAD"), "files": {f: _sha(root / f) for f in SOURCES}}}
    why = validate_plan(cfg, robot, plan, auth)
    if why:
        raise SystemExit(f"G4C-D2-P1 finalization refused (zero teacher calls): {why}")
    plan["plan_hash"] = cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    return plan


# ------------------------------------------------------------------ the D2 judge (no teacher calls)
def _trivial(reason: str) -> bool:
    return set(reason.split("+")) <= TRIVIAL


def judge(plan: dict, ev: dict) -> dict:
    """Labels come only from gp.adjudicate over revalidated evidence; nominal obstruction is
    carried as a construction annotation and never decides a label."""
    lab = {u: gp.adjudicate(e) for u, e in ev.items()}
    roots = {}
    for k, r in plan["roots"].items():
        jr = gp.judge_root(k, r, [u for u in plan["units"] if u["root"] == k], lab, ev)
        e = lab.get(f"seq:{k}:empty", {"label": "NOT_RUN"})
        blocked = (r.get("construction_privileged") or {}).get("nominal", {}).get("blocks_nominal_witness")
        dec = {P: (True, False), N: (False, True), "NOT_RUN": (None, None)}.get(e["label"], (False, False))
        start = [{"unit": u["unit"], **gp._pair(e["label"], lab.get(u["unit"], {}).get("label", "MISSING"))}
                 for u in plan["units"] if u["root"] == k and u["kind"] == "ROBOT_START_TARGET_ONLY"]
        words = {s: WORD.get(v, v) for s, v in jr["labels"].items()}
        sub = jr["bc_pool"]["qualifies"] and d2.substantive_pool(words)
        roots[k] = {**jr, "variant": r["variant"], "base_family": r["base_family"], "nominal_blocked": blocked,
                    "no_repair": d2.root_release_label(blocked == "YES", *dec), "no_repair_reason": e.get("reason"),
                    "robot_start": start, "pool_release": d2.pool_release(words, list(words)),
                    "substantive_complete_pool": bool(sub), "trivial_complete_pool": bool(jr["bc_pool"]["qualifies"] and not sub)}
    kinds = {}
    for u, v in lab.items():
        if v["label"] == N:
            kk = f"{v['reason']} [{'/'.join(v['scope'])}]"
            kinds[kk] = kinds.get(kk, 0) + 1
    inserted = {k: r for k, r in roots.items() if r["variant"] not in ("base", "fresh")}
    cert = {k: r for k, r in inserted.items() if r["no_repair"] == "NEGATIVE"}
    nontriv = [k for k, r in cert.items() if not _trivial(r["no_repair_reason"])]
    rec = {k: r["nonempty_positive"] for k, r in cert.items()}
    arch = {plan["roots"][k]["base_family"]: plan["roots"][k]["archived_base_replay"]["status"] for k in plan["roots"]
            if plan["roots"][k]["variant"] == "base"}
    q = {"Q1": "MET" if len(arch) == len(FAMILIES) and set(arch.values()) == {"BASE_WITNESS_REPLAYED"} else "NOT_MET",
         "Q1_evidence": {"archived_replay": arch, "independent_lineages": len({plan["roots"][k]["family_id"] for k in plan["roots"]
                                                                              if plan["roots"][k]["variant"] == "base"}),
                         "base_variant_new_target_runs": {k: roots[k]["no_repair"] for k in roots if roots[k]["variant"] == "base"},
                         "fresh_bases": {u: lab.get(u, {}).get("label", "NOT_RUN") for u in plan["fresh_units"]},
                         "fresh_construction_ledger": plan["fresh_ledger"]},
         "Q2": "MET" if nontriv else "TRIVIAL_ONLY" if cert else "NOT_MET",
         "Q2_evidence": {f"seq:{k}:empty": r["no_repair_reason"] for k, r in cert.items()},
         "Q3": "MET" if any(rec.values()) else "NOT_MET",
         "Q3_evidence": {"recovery_witnesses": rec, "on_nontrivial_blocked_root": any(rec[k] for k in nontriv)},
         "Q4": "MET" if any(r["order_contrasts"] for r in roots.values()) else "NOT_MET",
         "Q4_evidence": {k: {"variant": r["variant"], "contrasts": r["order_contrasts"]} for k, r in roots.items() if r["order_contrasts"]},
         "Q5": "MET" if any(r["substantive_complete_pool"] for r in roots.values()) else "NOT_MET",
         "Q5_evidence": {"substantive": sorted(k for k, r in roots.items() if r["substantive_complete_pool"]),
                         "trivial_complete": sorted(k for k, r in roots.items() if r["trivial_complete_pool"])}}
    counts = {}
    for v in lab.values():
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    return {"labels": lab, "roots": roots, "questions": q, "label_counts": counts, "negative_kinds": kinds,
            "trivial_negatives": sum(v["label"] == N and _trivial(v["reason"]) for v in lab.values()),
            "nontrivial_negatives": sum(v["label"] == N and not _trivial(v["reason"]) for v in lab.values())}


# ------------------------------------------------------------------ commands
def cmd_finalize(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final_plan.json").exists() or (out / "authorization.json").exists():
        raise SystemExit("POC_S plan already frozen; refusing to overwrite")
    auth = authorization_record(Path(args.instruction))
    plan = finalize(cf.load_config(args.config), auth)
    cf.write_json(out / "authorization.json", auth)
    plan["authorization_file_sha256"] = _sha(out / "authorization.json")
    plan["plan_hash"] = cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    cf.write_json(out / "final_plan.json", plan)
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    p0 = json.loads((P0 / "manifest.json").read_text())["files"]
    cf.write_json(out / "source_audit.json", {
        "git": {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"), "origin_main": git("rev-parse", "origin/main"),
                "working_tree": git("status", "--short").splitlines()},
        "p0_manifest_sha256": plan["p0_manifest_sha256"], "p0_files_verified": {f: gr._sha(P0 / f) == h for f, h in p0.items()},
        "sources": plan["source"]["files"], "config_hash": plan["config_hash"], "robot_profile_hash": plan["robot_profile_hash"],
        "archived_bases": {r["base_family"]: {"replay": r["archived_base_replay"], "witness_sha256": r["archived_witness_sha256"],
                                              "lineage": r["lineage"], "family_id": r["family_id"], "split": r["split"]}
                           for r in plan["roots"].values() if r["variant"] == "base"}})
    with open(out / "construction_attempts.jsonl", "w") as fh:
        fh.write(json.dumps({"kind": "P0_reference", "file": str(P0 / "construction_attempts.jsonl"),
                             "sha256": _sha(P0 / "construction_attempts.jsonl"), "note": "variant obstacles reused byte-identically"}) + "\n")
        for x in plan["fresh_ledger"]:
            fh.write(json.dumps({"kind": "fresh_base_attempt", "rule": "g4c J0 layout, static checks only", **x}) + "\n")
    print(json.dumps({"plan_hash": plan["plan_hash"], "units": len(plan["units"]), "frozen": len(plan["frozen_units"]),
                      "fresh": plan["fresh_units"], "fresh_ledger": plan["fresh_ledger"], "worst_case": plan["call_caps"]["worst_case"]}))
    return 0


def cmd_run(args) -> int:
    out, root = Path(args.out), Path(".")
    plan, auth = json.loads((out / "final_plan.json").read_text()), json.loads((out / "authorization.json").read_text())
    cfg = cf.load_config(args.config)
    bad = gr.verify_sources(plan, root) + validate_plan(cfg, ge.make_robot(cfg), plan, auth)
    if plan["p0_manifest_sha256"] != _sha(root / P0 / "manifest.json"):
        bad.append("P0 manifest changed")
    if plan["authorization_file_sha256"] != _sha(out / "authorization.json") or (out / "poc_result.json").exists():
        bad.append("authorization changed or POC already judged")
    if bad:
        raise SystemExit(f"POC_S collection refused (zero teacher calls): {bad}")
    t0 = time.time()
    led = gr.run_collection(plan, out, {"plan": out / "final_plan.json", "config": Path(args.config),
                                        "config_v2": Path(args.config_v2)}, root)
    print(json.dumps({"stops": {s: sum(v["stop"] == s for v in led["units"].values()) for s in {v["stop"] for v in led["units"].values()}},
                      "unit_cpu_s": round(led["unit_cpu_s"], 1), "coordinator_cpu_s": round(led["coordinator_cpu_s"], 2),
                      "wall_s": round(time.time() - t0, 1), "failure": led.get("enforcement_failure")}))
    return 1 if led.get("enforcement_failure") else 0


def cmd_judge(args) -> int:
    import d2_poc_figures
    return d2_poc_figures.run(args)



