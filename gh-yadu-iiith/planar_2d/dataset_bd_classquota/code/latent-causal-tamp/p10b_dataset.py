"""P10b: categorized ledger, read-only audit, constructibility probes, roster / preregistration, frozen plans, capped teacher
collection (one worker) and D blocker-mask derivation. Learned inference never calls the teacher, IK or a planner.

Ledger categories and aggregate caps (contract §6): constructor_dev 3,000; teacher 24,000 (incl. static admission and D
counterfactual labels); training 24,000; execution 15,000; evaluation 5,000 (regression, figures, prediction, scoring).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import config as cf
import geometry as ge
import p10_scenes as ps
import p10b_scenes as pb

OUT = Path("runs/p10b")
CAPS = {"constructor_dev": 3000.0, "teacher": 24000.0, "training": 200000.0, "execution": 15000.0, "evaluation": 5000.0}
JOB = {"cpu_s": 3000.0, "wall_s": 7200.0}


def _rd(p):
    return json.loads(Path(p).read_text())


def spent(cat=None) -> tuple:
    gs = [_rd(p) for p in (OUT / "guards").glob("*.json")]
    gs = [g for g in gs if cat is None or g["category"] == cat]
    return sum(g["cpu_s"] for g in gs), sum(g["wall_s"] for g in gs)


def job(name, call, cpu, module="p10b_dataset", category="evaluation", wall=None) -> dict:
    """Process-tree guarded job charged to ONE category; refuses to start when the category cap is exhausted."""
    import teacher
    (OUT / "guards").mkdir(parents=True, exist_ok=True)
    left = CAPS[category] - spent(category)[0]
    cpu = min(cpu, JOB["cpu_s"], left)
    if cpu < 5:
        raise SystemExit(f"{category} budget exhausted ({CAPS[category] - left:.0f} / {CAPS[category]:.0f} CPU-s)")
    g = teacher.guard([sys.executable, "-c", f"import torch; torch.set_num_threads(1); import {module}; {module}.{call}"], cpu,
                      wall or JOB["wall_s"], OUT / f"heartbeat_{name}.txt", 15.0, 0.25, Path("."))
    n = 1
    while (OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json").exists():
        n += 1
    rec = {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | \
          {"cpu_cap_s": cpu, "call": f"{module}.{call}", "category": category, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cf.write_json(OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json", rec)
    (OUT / f"heartbeat_{name}.txt").unlink(missing_ok=True)
    print(json.dumps({k: rec[k] for k in ("stop", "returncode", "cpu_s", "wall_s")} | {category: round(spent(category)[0], 1)}))
    return rec


# ------------------------------------------------------------------ P10b-0 read-only audit
def audit() -> None:
    import subprocess
    import torch
    import operational_teacher as ot
    import p10_eval as pe
    import p10_protocol as pp
    import p6_run as p6r
    import p9_l3 as l3
    sh = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
    out = {"head": sh("rev-parse", "HEAD"), "origin": sh("rev-parse", "origin/main"),
           "ancestry_ok": all(subprocess.run(["git", "merge-base", "--is-ancestor", a, "HEAD"]).returncode == 0
                              for a in ("3afa81c2f8804fc1a8d11e08e2b3f010286fc784", "dd12f0ce354c16e2525f88805cf2a8b5e811aadf")),
           "tracked_changes": sh("status", "--short", "--untracked-files=no").splitlines()}
    man = _rd("runs/p10/p10a/MANIFEST.json")["files"]
    out["p10a_manifest"] = {f: l3.sha(f) == h.removeprefix("sha256:") or "sha256:" + l3.sha(f) == h for f, h in man.items()}
    out["p10a_manifest_drift"] = sorted(f for f, ok in out["p10a_manifest"].items() if not ok)
    ck = {}
    for f in sorted(Path("runs/p10/p10_0/runs").glob("*.json")):
        r = _rd(f)
        ck[r["name"]] = l3.sha(pp.P0 / "ckpt" / f"ckpt_{r['name']}.pt") == r["checkpoint_sha256"]
    fz = _rd(pp.FROZEN)["checkpoint"]
    ck["FROZEN:" + fz["name"]] = l3.sha(fz["path"]) == fz["sha256"]
    out["checkpoints_ok"] = ck
    pred = _rd("runs/p10/p10a/predictions.json")
    out["predictions_hash_ok"] = cf.stable_hash({k: v for k, v in pred.items() if k != "hash"}) == pred["hash"]
    plan, prog, adm = _rd("runs/p10/p10a/plan.json"), _rd("runs/p10/p10a/progress.json"), _rd("runs/p10/p10a/static_admission.json")
    out["plan_hash_ok"] = cf.stable_hash({k: v for k, v in plan.items() if k != "plan_hash"}) == plan["plan_hash"]
    pol = ot.load_policy("teacher_opneg.toml")
    rec = {q["query"]: p6r.judge(adm["certificates"], pol, q, prog["done"].get(q["query"])) for q in plan["queries"]}
    dec = _rd("runs/p10/p10a/decisions.json")
    out["decisions_replay_ok"] = all(rec[q]["teacher_status"] == dec[q]["teacher_status"] for q in dec)
    cfg = cf.load_config("config.toml")
    cx = ps.Ctx(cfg, ge.make_robot(cfg))
    r1 = plan["roots"]["p10a_2026102801"]
    spec = next(s for s in ps.ROOTS if s["seed"] == r1["seed"])
    re_, _ = ps._try(cx, spec, r1["attempt"], set())
    from scene import scene_hash
    out["p10a_constructor_replay_ok"] = re_ is not None and scene_hash(re_["scene"]) == r1["scene_hash"]
    out["p10a_gates"] = {g: _rd(f"runs/p10/{g}.json").get("engineering") for g in ("P10_0_GATE", "P10A_GATE")}
    gate = {}                                                  # corrected yield gate (contract P10b-0): mechanism REALIZED, not merely decided
    for k, r in plan["roots"].items():
        st = {pe_tok: rec[q["query"]]["teacher_status"] for q in plan["queries"] if q["root"] == k for pe_tok in [_p10a_tok(plan, q)]}
        if r["mechanism"] == "H7":
            gate[k] = {"edit_feasible": st["edit:A"] == ot.VERIFIED, "relocation_infeasible": st["M"] in (ot.CERTIFIED, ot.BUDGET)}
        if r["mechanism"] == "H8":
            gate[k] = {"original_infeasible": st[""] in (ot.CERTIFIED, ot.BUDGET)}
    out["p10a_corrected_gate"] = {k: {**v, "realized": all(v.values())} for k, v in gate.items()}
    r01 = {(_p10a_tok(plan, q)): (rec[q["query"]]["teacher_status"], (rec[q["query"]]["static_certificate"] or {}).get("kind"))
           for q in plan["queries"] if q["root"] == "p10a_2026102801"}
    out["erratum_h1_root01"] = {"sequence A (physical)": r01["A"], "edit A (scene)": r01["edit:A"]}
    with pe.torch.no_grad():
        ds = torch.load(pe.REC, weights_only=False)
    out["p10a_records_bound"] = ds["plan_hash"] == plan["plan_hash"]
    (OUT / "p10b0").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "p10b0" / "audit.json", out)


def _p10a_tok(plan, q):
    import p10_collect as pc
    return pc._tok(plan, q)


# ------------------------------------------------------------------ constructibility probes (development seeds only; never roster seeds)
DEV_SEED0 = 2026119000


def probe(n=40, fams=None, tag="") -> None:
    import collections
    cfg = cf.load_config("config.toml")
    cx = ps.Ctx(cfg, ge.make_robot(cfg))
    f = OUT / "p10b0" / f"probe_n{n}{tag}.json"
    f.parent.mkdir(parents=True, exist_ok=True)
    out = _rd(f) if f.exists() else {}
    for i, fam in enumerate(fams or pb.FAMILIES):
        c0, why = time.process_time(), collections.Counter()
        for a in range(n):
            rec, w = pb.try_root(cx, {"family": fam, "seed": DEV_SEED0 + 97 * i, "index": -1, "wave": 0}, a, set())
            why["ACCEPTED" if rec else w] += 1
        out[fam] = {"cpu_per_attempt": round((time.process_time() - c0) / n, 4), "accept_rate": why["ACCEPTED"] / n, "reasons": dict(why.most_common(8))}
        cf.write_json(f, out)


# ------------------------------------------------------------------ frozen plans (before any teacher call) and capped collection
POLICY = {"version": "P10B_S100_V1", "adjudication": "unchanged 2D-ABC-OPNEG-v1 (teacher_opneg.toml); BUDGET only if all 3 declared seeds "
          "ran to a clean allowed terminal", "seed_cpu_s": 100.0, "seed_kill_margin_s": 5.0, "reserve_s": 10.0, "query_cpu_s": 330.0,
          "query_wall_s": 900.0, "root_cpu_s": 900.0, "stop_after_censored_seed": True,
          "options_predeclared": {"A_P10A": "120 CPU-s per query (later seed only while spent + 110 <= 120)",
                                  "B_FULL_SEEDS": "330 CPU-s per query: all three 100 CPU-s seeds fit; a censored seed ends the query (UNKNOWN)"},
          "chosen": "B_FULL_SEEDS", "workers": 1}


def build_plan(out, specs, max_attempts=200, root_cpu=60.0) -> dict:
    """Construct the given roster entries and freeze plan.json (refuses to overwrite). Unavailable roots are recorded, never replaced."""
    import data
    import operational_teacher as ot
    from scene import scene_hash, scene_to_dict
    out = Path(out)
    if (out / "plan.json").exists():
        raise SystemExit(f"{out}/plan.json already frozen")
    cfg, pol = cf.load_config("config.toml"), ot.load_policy("teacher_opneg.toml")
    robot = ge.make_robot(cfg)
    cx, known = ps.Ctx(cfg, robot), _known()
    plan = {"stage": "P10B", "version": pb.VERSION, "policy_id": pol["policy"]["policy_id"], "policy_hash": pol["policy_hash"],
            "execution_policy": POLICY, "config_hash": cf.config_hash(cfg), "robot_profile_hash": cf.robot_profile_hash(cfg),
            "roots": {}, "queries": [], "units": [], "ineligible": [], "unavailable": [], "attempts": []}
    for spec in specs:
        c0, rec = time.process_time(), None
        for a in range(max_attempts):
            if time.process_time() - c0 > root_cpu:
                break
            rec, why = pb.try_root(cx, spec, a, known)
            plan["attempts"].append({"seed": spec["seed"], "attempt": a, "accepted": rec is not None, "reason": why})
            if rec and tuple(map(str, rec["fingerprint"])) not in known:
                break
            rec = None
        if rec is None:
            plan["unavailable"].append({**spec, "status": "CONSTRUCTION_UNAVAILABLE"})
            continue
        import g4c
        known |= {g4c.stable_geometry(rec["scene"], rec["goal_cell"]), tuple(map(str, rec["fingerprint"]))}
        k, s = rec["root"], rec["scene"]
        q0 = data.make_query(cfg, k, s, "t0", rec["goal_cell"], {}, "SAFE_RETREAT", data.PROTOCOL_V2)
        items = pb.catalogue(cfg, robot, rec)
        meta = {x: rec[x] for x in ("family", "mechanism", "variant", "seed", "attempt", "stratum", "goal_cell", "ids", "dest", "role_cells",
                                    "fixed", "predicates", "construction", "parking", "fingerprint")}
        plan["roots"][k] = {"scene": scene_to_dict(s), "scene_hash": scene_hash(s), "hashes": q0["hashes"], "lineage": k, "split": spec["split"],
                            "batch": spec["batch"], **meta, "n_objects": len(s.objects), "items": items}
        for it in items:
            if it["kind"] == "SCENE_TARGET_ONLY" and not it["eligible"]:
                plan["ineligible"].append({"root": k, "query": it["query"]})
                continue
            plan["queries"].append({**{x: it.get(x) for x in ("query", "kind", "priority", "anchor", "token", "edit_kind", "d_subset")},
                                    "root": k, "lineage": k, "sequence": it.get("sequence", []), "edit": it.get("edit"), "split": spec["split"],
                                    "family": rec["family"], "mechanism": rec["family"], "batch": spec["batch"], "role": "anchor" if it["anchor"] else "catalogue"})
            sq = it["kind"] == "SCENE_TARGET_ONLY"
            sig = it["query"].split(":", 2)[2]
            for sd in pol["policy"]["seeds"]:
                key = pol["policy"]["seed_key"].format(root=k, sig=("scene/" if sq else "") + sig, seed=sd)
                u = {"unit": f"{it['query']}#s{sd}", "query": it["query"], "seed": sd, "kind": it["kind"], "root": k,
                     "target_key": key + "/target", "relocation_key": key + "/reloc"}
                u.update({"edit": it["edit"], "eligible": True, "intervened_scene_hash": it["intervened_scene_hash"]} if sq else {"sequence": it["sequence"]})
                plan["units"].append(u)
    plan["plan_hash"] = cf.stable_hash({x: v for x, v in plan.items() if x != "plan_hash"})
    out.mkdir(parents=True, exist_ok=True)
    cf.write_json(out / "plan.json", plan)
    return plan


def _known() -> set:
    import p10_collect as pc
    known = pc._known()
    for f in ["runs/p10/p10a/plan.json"] + sorted(str(p) for p in OUT.glob("*/plan.json")):
        import g4c
        from scene import scene_from_dict
        for k, r in _rd(f)["roots"].items():
            known.add(g4c.stable_geometry(scene_from_dict(r["scene"]), r["goal_cell"]))
            if "fingerprint" in r:
                known.add(tuple(map(str, r["fingerprint"])))
    return known


def admit(out) -> None:
    import teacher_pilot as tp
    out = Path(out)
    if (out / "static_admission.json").exists():
        raise SystemExit("static admission already done")
    cfg, plan, c0 = cf.load_config("config.toml"), _rd(out / "plan.json"), time.process_time()
    static = tp.static_admission(cfg, ge.make_robot(cfg), plan, use_sector=True)
    cf.write_json(out / "static_admission.json", {"certificates": static, "cpu_s": round(time.process_time() - c0, 2), "plan_hash": plan["plan_hash"]})


def _xpol(pol):
    import copy
    x = copy.deepcopy(pol)
    x["policy"].update(per_seed_cpu_s=POLICY["seed_cpu_s"], per_seed_guard_margin_s=POLICY["seed_kill_margin_s"], per_seed_wall_s=POLICY["query_wall_s"])
    x["pilot"]["orchestration_reserve_s"] = POLICY["reserve_s"]
    return x


def collect(out, chunk_cpu=2500.0) -> None:
    """Resumable, one worker. Queries in frozen priority order per root; seeds one at a time (a censored seed ends the query).
    Blind-split outcomes are written ONLY to <out>/sealed/ (never to progress.json or the shared ledger)."""
    import p6_run as p6r
    import teacher_pilot as tp
    import operational_teacher as ot
    out = Path(out)
    cfg, v2, pol = cf.load_config("config.toml"), cf.load_config_v2("config_v2_r4.toml"), ot.load_policy("teacher_opneg.toml")
    plan, robot, static = _rd(out / "plan.json"), ge.make_robot(cfg), _rd(out / "static_admission.json")["certificates"]
    if v2["proposal"]["move_cap"] != 4 or ot.config_mismatch(pol, cfg) or pol["policy_hash"] != plan["policy_hash"]:
        raise SystemExit("P10b requires the R4 overlay and the unchanged OPNEG-v1 policy")
    paths = {"plan": out / "plan.json", "config": Path("config.toml"), "config_v2": Path("config_v2_r4.toml")}
    c0, unit = time.process_time(), [0.0]
    for k, r in plan["roots"].items():
        sealed = r["split"] == "blind"
        pf = (out / "sealed" / f"{k}.json") if sealed else (out / "progress" / f"{k}.json")
        pf.parent.mkdir(parents=True, exist_ok=True)
        prog = _rd(pf) if pf.exists() else {"root": k, "done": {}}
        for q in sorted([q for q in plan["queries"] if q["root"] == k], key=lambda q: q["priority"]):
            qid = q["query"]
            if qid in prog["done"]:
                continue
            used = sum(d["unit_cpu_s"] + d["coord_cpu_s"] for d in prog["done"].values())
            if static.get(qid) is None and time.process_time() - c0 + unit[0] > chunk_cpu:
                return
            qdir, stop, seeds, uc, c1 = out / "q" / qid.replace(":", "__").replace(">", "-"), None, {qid: []}, 0.0, time.process_time()
            if static.get(qid) is None and qdir.exists():
                stop = "interrupted before this query was recorded; never repeated"
            if stop is None and static.get(qid) is None and POLICY["root_cpu_s"] - used < POLICY["seed_cpu_s"] + POLICY["reserve_s"]:
                stop = "root CPU cap reached"
            if stop is None and static.get(qid) is None:
                for sd in pol["policy"]["seeds"]:
                    xp = _xpol(pol)
                    xp["policy"]["seeds"] = [sd]
                    seeds, u = tp.run_rounds(plan, xp, qdir, paths, cfg, v2, robot, static, [qid],
                                             min(POLICY["query_cpu_s"] - uc, POLICY["root_cpu_s"] - used - uc), POLICY["query_wall_s"], seeds)
                    uc += u
                    last = seeds[qid][-1] if seeds[qid] else None
                    if last is None or last["terminal"] in ("WITNESS", "CERTIFICATE") or (POLICY["stop_after_censored_seed"] and str(last["stop"]).endswith("CENSORED")):
                        break
            unit[0] += uc
            d = {"stop": stop, "attempts": seeds[qid], "unit_cpu_s": uc, "coord_cpu_s": time.process_time() - c1}
            j = p6r.judge(static, pol, q, d)
            d["status"] = j["teacher_status"]
            prog["done"][qid] = d
            cf.write_json(pf, prog)
            if not sealed:
                with open(out / "teacher_calls.jsonl", "a") as fh:
                    fh.write(json.dumps({"caller_stage": "LABEL_ACQUISITION", "root_id": k, "query_id": qid, "status": j["teacher_status"],
                                         "provenance": j["label_source"], "static_certificate": static.get(qid), "stop": stop,
                                         "seeds": [{x: a.get(x) for x in ("seed", "stop", "terminal", "clean", "cpu_s", "wall_s", "witness_hash")}
                                                   for a in d["attempts"]], "unit_cpu_s": round(uc, 2), "plan_hash": plan["plan_hash"],
                                         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}) + "\n")
    (out / "COLLECTED").write_text(plan["plan_hash"] + "\n")


def records(out, split_ok=("train", "validation", "dev")) -> dict:
    """Adjudicated records of NON-blind roots only. Blind outcomes require runs/p10b/UNSEALED (written after the freeze commit)."""
    import p6_run as p6r
    import operational_teacher as ot
    out = Path(out)
    plan, static, pol = _rd(out / "plan.json"), _rd(out / "static_admission.json")["certificates"], ot.load_policy("teacher_opneg.toml")
    if "blind" in split_ok and not (OUT / "UNSEALED").exists():
        raise SystemExit("blind labels are sealed")
    res = {}
    for k, r in plan["roots"].items():
        if r["split"] not in split_ok:
            continue
        pf = out / ("sealed" if r["split"] == "blind" else "progress") / f"{k}.json"
        done = _rd(pf)["done"] if pf.exists() else {}
        for q in plan["queries"]:
            if q["root"] == k:
                res[q["query"]] = p6r.judge(static, pol, q, done.get(q["query"]))
    return res


def d_labels(plan, rec) -> dict:
    """Per root: pool (role order), statuses of all parking subsets, d_blockers.minimal_sets result, and an INDEPENDENT check
    (minimal = VERIFIED subsets with no VERIFIED proper subset; union must equal the mask) for every complete monotone mask."""
    import itertools
    import d_blockers as db
    out = {}
    for k, r in plan["roots"].items():
        pool = list(r["parking"])
        inv = {v: kk for kk, v in r["ids"].items()}
        roles = [inv[o] for o in pool]
        st, byq = {}, {it["query"]: it for it in r["items"]}
        for it in r["items"]:
            if it.get("d_subset") is not None:
                key = frozenset(roles.index(x) for x in it["d_subset"])
                st[key] = rec[it["query"]]["teacher_status"] if it["query"] in rec else ("INELIGIBLE" if it["kind"] == "SCENE_TARGET_ONLY" and not it["eligible"] else "MISSING")
        g = db.minimal_sets(len(pool), st)
        if g["mask"] is not None and g["type"] == "MINIMAL_SETS":
            vs = [s for s, x in st.items() if x == db.V]
            mins = [s for s in vs if not any(t < s for t in vs)]
            g["independent_check"] = sorted(map(sorted, mins)) == sorted(g["sets"]) and [int(any(i in s for s in mins)) for i in range(len(pool))] == g["mask"]
        out[k] = {"pool": pool, "roles": roles, "status": {"+".join(roles[i] for i in sorted(s)) or "(original)": v for s, v in st.items()},
                  **g, "n": len(pool), "_q": len(byq)}
    return out


def _sha(f) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(Path(f).read_bytes()).hexdigest()


def pytest_d(path="runs/p10b/p10b0/test_d_blockers.log") -> None:
    import subprocess
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_d_blockers.py"], capture_output=True, text=True)
    Path(path).write_text(r.stdout[-4000:] + f"\nEXIT {r.returncode}\n")


def regression() -> None:
    """Historical suite + six replays after the P10b-P10e code changes (fixture planner calls itemized by countcalls)."""
    import subprocess
    d = OUT / "regression"
    d.mkdir(parents=True, exist_ok=True)
    (d / "countcalls.py").write_text(Path("runs/p8/audit/regression/countcalls.py").read_text())
    subprocess.run([sys.executable, "run.py", "g4c-lf-p4-regression", "--out", str(d), "--cpu", "1400", "--wall", "3400"], capture_output=True, text=True)
