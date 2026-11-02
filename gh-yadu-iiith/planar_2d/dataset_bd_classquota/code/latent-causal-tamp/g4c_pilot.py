"""G4C-P1 authorized five-root pilot (G4C-P1 contract): authorization record, finalization
adapter, guarded collection through the unchanged G4D runner, and the G4C evidence judge.

The frozen preflight plan (runs/g4c_preflight/plan.json) is never modified: its budget
fields stay null. `finalize` writes a separate, immutable derived execution plan holding
exactly the 59 pilot units (verbatim, frozen order), the five roots, the authorized
limits, parent plan hashes and a source manifest. Collection reuses g4d_run.run_collection
(process-tree CPU guard, one worker, atomic checkpoints, resume without rerun). The judge
rebuilds every unit's evidence with g4d_run.unit_evidence (independent re-checks) and then
applies G4C rules fail-closed: a POSITIVE needs every witness check true, a NEGATIVE needs
an accepted certificate of ORDER/ASSIGNMENT scope, everything else stays UNKNOWN.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import time
from pathlib import Path

import config as cf
import data
import g4c
import g4d_run as gr
import geometry as ge

P, N, U = data.POSITIVE, data.NEGATIVE, data.UNKNOWN
PARENT = Path("runs/g4c_preflight/plan.json")
PARENT_HASH = "sha256:c761c68497309f6b631f6740237e75f08c2376ade179b7f644731dc775f193a3"
PILOT = {"g4c_r00": 2, "g4c_r08": 10, "g4c_r10": 11, "g4c_r14": 19, "g4c_r19": 17}
AUTH = {"scope": "G4C_P1_PILOT_ONLY", "per_teacher_unit_cpu_s": 900, "cancellation_margin_s": 30, "aggregate_cpu_s": 10800,
        "workers": 1, "unit_wall_guard_s": 1800, "aggregate_wall_guard_s": 14400, "max_motion_nodes": 4000,
        "relocation_skill_attempt_cap": 60, "target_eval_attempt_cap": 59, "max_root_construction_attempts": 0}
RESERVE_S = 60.0
SOURCES = gr.SOURCES[:-1] + ("g4c.py", "g4c_pilot.py", "runs/g4c_preflight/plan.json")   # g4d_run's list minus the G4D plan
SOUND = {"ORDER", "ASSIGNMENT"}


def _sha(p: Path) -> str:
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _rd(p: Path):
    return json.loads(Path(p).read_text())


# ------------------------------------------------------------------ authorization and finalization (no teacher calls)
def authorization_record(instruction: Path) -> dict:
    rec = {**AUTH, "roots": PILOT, "units": sum(PILOT.values()), "parent_plan_hash": PARENT_HASH,
           "parent_plan_file_sha256": _sha(PARENT), "instruction_sha256": _sha(instruction),
           "granted_by": "user message of 20 October 2026: 'I explicitly authorize the G4C five-root pilot only'",
           "excluded": "the remaining 15 roots / 108 units, any new root, query, budget, training, 3D or hardware",
           "orchestration_reserve_s": RESERVE_S}
    rec["authorization_hash"] = cf.stable_hash(rec)
    return rec


def check_authorization(auth: dict) -> list:
    why = [f"{k}={auth.get(k)!r} != authorized {v!r}" for k, v in AUTH.items() if auth.get(k) != v]
    if auth.get("roots") != PILOT or auth.get("units") != 59:
        why.append("pilot scope differs from the authorized five roots / 59 units")
    if auth.get("parent_plan_hash") != PARENT_HASH:
        why.append("authorization bound to another parent plan")
    if cf.stable_hash({k: v for k, v in auth.items() if k != "authorization_hash"}) != auth.get("authorization_hash"):
        why.append("authorization record modified")
    return why


def finalize(parent: dict, auth: dict, cfg: dict, root: Path) -> dict:
    """Derived immutable pilot plan; raises SystemExit (before any teacher call) on any failure."""
    why = check_authorization(auth)
    if parent.get("plan_hash") != PARENT_HASH or cf.stable_hash({k: v for k, v in parent.items() if k != "plan_hash"}) != PARENT_HASH:
        why.append("parent plan hash mismatch")
    why += [f"parent validator: {w}" for w in g4c.validate(parent)]
    if any(v is not None for v in parent["budget"].values()):
        why.append("parent plan budget fields are not null")
    if cfg["planner2d"]["max_nodes"] != auth["max_motion_nodes"] or cfg["skill2d"]["approach_dirs"] * 2 != 48:
        why.append("planner/grasp profile differs from the frozen one")
    units = [u for u in parent["units"] if u["root"] in PILOT]
    counts = {r: sum(u["root"] == r for u in units) for r in PILOT}
    if counts != PILOT or len(units) != 59 or len({u["unit"] for u in units}) != 59:
        why.append(f"pilot units {counts} differ from {PILOT}")
    if sum(u["relocation_skills_max"] for u in units) > auth["relocation_skill_attempt_cap"] or \
            sum(u["target_evaluations_max"] for u in units) > auth["target_eval_attempt_cap"]:
        why.append("structural call maxima exceed the authorized caps")
    if why:
        raise SystemExit(f"G4C pilot finalization refused (zero teacher calls): {why}")
    robot = ge.make_robot(cfg)
    roots = {k: parent["roots"][k] for k in PILOT}
    stub = {"roots": roots}
    for u in units:                                                   # identity: scene/query/eligibility/seed keys
        gr.unit_scene(cfg, robot, stub, u)
    plan = {"stage": "G4C_P1_PILOT", "protocol": data.PROTOCOL_V2, "dataset_version": parent["dataset_version"],
            "status": "FINAL_AUTHORIZED_PILOT", "parent_plan_hash": PARENT_HASH, "parent_plan_file_sha256": _sha(PARENT),
            "authorization_hash": auth["authorization_hash"], "roots": roots, "units": units,
            "unit_order": "parent plan order, filtered to the pilot roots (frozen before outcomes)",
            "limits": {"cpu_per_unit_s": float(auth["per_teacher_unit_cpu_s"]), "guard_margin_s": float(auth["cancellation_margin_s"]),
                       "cpu_aggregate_s": float(auth["aggregate_cpu_s"]), "workers": auth["workers"],
                       "wall_per_unit_s": float(auth["unit_wall_guard_s"]), "wall_total_s": float(auth["aggregate_wall_guard_s"]),
                       "poll_s": 0.25, "orchestration_reserve_s": RESERVE_S},
            "call_caps": {"relocation_skills": auth["relocation_skill_attempt_cap"], "target_evaluations": auth["target_eval_attempt_cap"],
                          "structural_maxima": {"relocation_skills": sum(u["relocation_skills_max"] for u in units),
                                                "target_evaluations": sum(u["target_evaluations_max"] for u in units)},
                          "enforcement": "each unit plans at most len(sequence) relocations and one target evaluation; the "
                                         "summed maxima equal the caps, so no admitted schedule can exceed them"},
            "threads": gr.teacher.THREADS, "config_hash": cf.config_hash(cfg), "robot_profile_hash": cf.robot_profile_hash(cfg),
            "source": {"head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip(),
                       "files": {f: _sha(root / f) for f in SOURCES}}}
    plan["plan_hash"] = cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    return plan


# ------------------------------------------------------------------ G4C judge (fail closed, over revalidated evidence)
def adjudicate(e: dict) -> dict:
    """G4C label from rebuilt unit evidence. POSITIVE requires a recorded complete witness AND
    every consistency check true; NEGATIVE requires an accepted certificate of sound scope."""
    c = e.get("checks", {})
    wit = any(x["kind"] == "complete_witness" for x in e.get("evidence", []))
    wc = (e.get("target") or {}).get("witness_check") or {}
    pos = wit and wc.get("accepted") is True and wc.get("ends_in_safe_retreat") is True and \
        all(c.get(k) is True for k in ("target_start_equality", "achieved_state_consistency", "joint_continuity"))
    certs = [x for x in e.get("certificates", []) if x.get("accepted") is True and x.get("scope") in SOUND]
    if e.get("label") == "NOT_RUN":
        return {"label": "NOT_RUN", "reason": e.get("stop")}
    if pos and certs:
        return {"label": data.CONTRADICTION, "reason": "witness and certificate"}
    if pos:
        return {"label": P, "reason": "complete witness, all checks true"}
    if certs:
        return {"label": N, "reason": "+".join(sorted({x["kind"] for x in certs})), "scope": sorted({x["scope"] for x in certs})}
    return {"label": U, "reason": _unknown_reason(e)}


def _unknown_reason(e) -> str:
    r, t = e.get("relocation") or {}, e.get("target") or {}
    if e.get("stop") not in ("COMPLETED", None):
        return f"stop {e.get('stop')}"
    if r and not r.get("complete_witness"):
        return f"relocation {r.get('stop')}: {','.join(r.get('normalized_reasons') or [])}"
    if t.get("witness_check") is not None:
        return "target witness failed a re-check"
    return f"target {t.get('status')}: {t.get('normalized_reason') or t.get('reason')}" if t else "no decisive evidence"


def _sig(seq):
    return ".".join(f"{o}>{c}" for o, c in seq) or "empty"


def _pair(ref, alt):
    lab = data.effect_label(ref, alt) if ref in (P, N, U) and alt in (P, N, U) else {"delta": None, "mask": False}
    return {"reference": ref, "alternative": alt, **lab}


def judge_root(k: str, root: dict, units: list, lab: dict, ev: dict) -> dict:
    """Per-root coverage over the frozen queries; every planned unit stays in the denominator."""
    Q, seqs = root["queries"], {_sig(p["sequence"]): p for p in root["queries"]["pool"]}
    L = lambda s: lab.get(f"seq:{k}:{s}", {}).get("label", "MISSING")  # noqa: E731
    pos = [s for s in seqs if L(s) == P]
    contrasts = []
    for a, b in itertools.combinations(seqs, 2):
        A, B = seqs[a]["sequence"], seqs[b]["sequence"]
        if len(A) >= 2 and sorted(map(tuple, A)) == sorted(map(tuple, B)):
            for x, y in ((a, b), (b, a)):
                if L(x) == P and L(y) == N and "ORDER" in lab[f"seq:{k}:{y}"].get("scope", []):
                    contrasts.append({"positive": x, "negative": y, "certificate": lab[f"seq:{k}:{y}"]["reason"]})
    scene = {}
    for q in Q["scene_queries"]:
        uid = f"scene:{k}:{_sig(sorted(q['edit'].items()))}"
        scene[q["kind"]] = {"unit": uid, "eligible": q["eligible"], **_pair(L("empty"), lab.get(uid, {}).get("label", "MISSING"))}
    omissions = []
    for s, p in seqs.items():
        seq = [tuple(a) for a in p["sequence"]]
        for n in (1, 2) if len(seq) >= 2 else ():
            for drop in itertools.combinations(range(len(seq)), n):
                rem = _sig([a for i, a in enumerate(seq) if i not in drop])
                omissions.append({"sequence": s, "omitted_positions": list(drop), "remaining": rem,
                                  **_pair(L(rem), L(s))})
    nonadd = []
    single = [v for kk, v in scene.items() if kk.startswith("singleton:")]
    for kk, v in scene.items():
        if kk.startswith("joint:"):
            parts = [scene.get(f"singleton:{x}") for x in kk.split(":")[1].split("+")]
            if v["mask"] and all(x and x["mask"] for x in parts):
                nonadd.append({"kind": "scene", "joint": kk, "delta_joint": v["delta"], "deltas_single": [x["delta"] for x in parts],
                               "kappa": v["delta"] - sum(x["delta"] for x in parts)})
    for s in {o["sequence"] for o in omissions}:
        om = [o for o in omissions if o["sequence"] == s]
        if len(seqs[s]["sequence"]) == 2 and all(o["mask"] for o in om):
            si = [o["delta"] for o in om if len(o["omitted_positions"]) == 1]
            jo = next(o["delta"] for o in om if len(o["omitted_positions"]) == 2)
            nonadd.append({"kind": "omission", "sequence": s, "delta_joint": jo, "deltas_single": si, "kappa": jo - sum(si)})
    costs = [{"sequence": s, "moved": len(p["sequence"]), "J_first": ev[f"seq:{k}:{s}"]["cost"]["recomputed_J"],
              "endpoint_distance": p["endpoint_distance"], "ordered_label": L(s)}
             for s, p in seqs.items() if (ev.get(f"seq:{k}:{s}", {}).get("cost") or {}).get("label_available")]
    pool = bc_pool(k, root, lab)
    return {"family": root["family"], "zero_move_positive": L("empty") == P, "empty_label": L("empty"),
            "nonempty_positive": [s for s in pos if s != "empty"],
            "alternative_assignments": len({tuple(sorted(map(tuple, seqs[s]["sequence"]))) for s in pos if s != "empty"}),
            "order_contrasts": contrasts, "scene": scene, "omissions": omissions, "nonadditivity": nonadd,
            "decided_omissions": [o for o in omissions if o["mask"]], "costs": costs, "bc_pool": pool,
            "labels": {s: L(s) for s in seqs}, "single_scene_effects": single}


def bc_pool(k: str, root: dict, lab: dict) -> dict:
    """Every active slot of every pool sequence must be decided by its own scoped unit."""
    Q, missing, slots = root["queries"], [], 0
    cat = {q["kind"]: q for q in Q["scene_queries"]}
    for p in Q["pool"]:
        seq = [tuple(a) for a in p["sequence"]]
        L = p["bc_layout"]
        if L != g4c.layout(p["sequence"], Q["catalogue"]["queries"]):
            return {"qualifies": False, "error": f"layout of {_sig(seq)} differs from the frozen mask rule"}
        for slot, act in zip(L["slots"], L["active"]):
            if not act:
                continue
            slots += 1
            if slot == "z0":
                uid = f"seq:{k}:empty"
            elif slot == "zR":
                uid = f"seq:{k}:{_sig(seq)}"
            elif slot.startswith("zS["):
                q = cat.get(slot[3:-1])
                uid = q and f"scene:{k}:{_sig(sorted(q['edit'].items()))}"
            else:
                t = int(slot[3:-1])
                uid = f"seq:{k}:{_sig([a for i, a in enumerate(seq) if i != t])}"
            got = lab.get(uid, {}).get("label") if uid else None
            if got not in (P, N):
                missing.append({"sequence": _sig(seq), "slot": slot, "unit": uid, "status": got or "UNMAPPED"})
    return {"qualifies": not missing, "active_slots": slots, "decided_slots": slots - len(missing), "missing": missing,
            "catalogue_hash": Q["catalogue"]["hash"]}


def judge(plan: dict, parent: dict, ev: dict) -> dict:
    lab = {uid: adjudicate(e) for uid, e in ev.items()}
    roots = {k: judge_root(k, plan["roots"][k], [u for u in plan["units"] if u["root"] == k], lab, ev) for k in plan["roots"]}
    tags = {k: set() for k in roots}
    for k, r in roots.items():
        if r["zero_move_positive"]:
            tags[k].add("zero_move")
        if r["nonempty_positive"]:
            tags[k].add("nonempty_witness")
        if r["alternative_assignments"] >= 2:
            tags[k].add("alternative_repairs")
        if r["order_contrasts"]:
            tags[k].add("order_contrast")
        if any(v["mask"] and v["delta"] != 0 for v in r["scene"].values()):
            tags[k].add("scene_effect")
        if r["decided_omissions"]:
            tags[k].add("omission")
        if any(v["mask"] and v["delta"] == 0 and kk == "nuisance" for kk, v in r["scene"].items()):
            tags[k].add("nuisance_zero_effect")
        if r["costs"]:
            tags[k].add("cost")
        if any(c["moved"] >= 2 for c in r["costs"]):
            tags[k].add("multi_move_cost")
        if r["bc_pool"]["qualifies"]:
            tags[k].add("bc_full_mini_pool")
        if any(x["kappa"] != 0 for x in r["nonadditivity"]):
            tags[k].add("nonadditive")
    pilot_cov = g4c.coverage({"roots": {k: parent["roots"][k] for k in plan["roots"]}}, {k: sorted(v) for k, v in tags.items()})
    rest = [k for k in parent["roots"] if k not in plan["roots"]]
    full = []
    for row in pilot_cov:
        remaining = sum(g4c.TAGS[row["threshold"]] in parent["roots"][k].get("supports", []) for k in rest)
        best = len(row["measured_roots"]) + remaining
        full.append({"threshold": row["threshold"], "required": row["required"], "pilot_measured": len(row["measured_roots"]),
                     "pilot_measured_roots": row["measured_roots"], "remaining_planned_capacity": remaining,
                     "status": "NOT_EVALUATED_FULL_SET / PILOT_ONLY",
                     "full_set_reachable_if_all_remaining_decide": best >= row["required"]})
    counts = {}
    for v in lab.values():
        counts[v["label"]] = counts.get(v["label"], 0) + 1
    return {"labels": lab, "roots": roots, "tags": {k: sorted(v) for k, v in tags.items()}, "full_gate": full,
            "label_counts": counts}


# ------------------------------------------------------------------ commands
def cmd_g4c_pilot_finalize(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "pilot_plan.json").exists() or (out / "authorization.json").exists():
        raise SystemExit("pilot authorization/plan already frozen; refusing to overwrite")
    auth = authorization_record(Path(args.instruction))
    plan = finalize(_rd(PARENT), auth, cf.load_config(args.config), Path("."))
    cf.write_json(out / "authorization.json", auth)
    plan["authorization_file_sha256"] = _sha(out / "authorization.json")
    plan["plan_hash"] = cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"})
    cf.write_json(out / "pilot_plan.json", plan)
    print(json.dumps({"pilot_plan_hash": plan["plan_hash"], "authorization_hash": auth["authorization_hash"],
                      "units": len(plan["units"]), "parent": PARENT_HASH}))
    return 0


def cmd_g4c_pilot_run(args) -> int:
    out, root = Path(args.out), Path(".")
    plan, auth = _rd(out / "pilot_plan.json"), _rd(out / "authorization.json")
    bad = gr.verify_sources(plan, root) + check_authorization(auth)
    if plan["authorization_file_sha256"] != _sha(out / "authorization.json") or plan["parent_plan_file_sha256"] != _sha(PARENT):
        bad.append("authorization or parent plan file changed")
    if (out / "pilot_result.json").exists():
        bad.append("pilot already judged")
    if bad:
        raise SystemExit(f"pilot collection refused (zero teacher calls): {bad}")
    t0 = time.time()
    led = gr.run_collection(plan, out, {"plan": out / "pilot_plan.json", "config": Path(args.config),
                                        "config_v2": Path(args.config_v2)}, root)
    print(json.dumps({"units": {k: v["stop"] for k, v in led["units"].items()}, "unit_cpu_s": round(led["unit_cpu_s"], 1),
                      "coordinator_cpu_s": round(led["coordinator_cpu_s"], 2), "wall_s": round(time.time() - t0, 1),
                      "failure": led.get("enforcement_failure")}))
    return 1 if led.get("enforcement_failure") else 0


def cmd_g4c_pilot_judge(args) -> int:
    import g4c_pilot_figures
    return g4c_pilot_figures.run(args)
