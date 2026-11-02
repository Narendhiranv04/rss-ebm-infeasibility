"""G4C-D2-P3a: LINK0_SECTOR_PICK_V1 producer, frozen static adjudication and science delta.

Producer side of the sector certificate (theorem and checker: sector_check.py). For one
fixed start it records, for each of the 48 declared pick candidates, the IK, the pose-only
screen verdict and whether q[0] lies outside the link-0 free sector. The checker, not the
producer, decides. The adjudication runs under hard guards: any motion-planner, skill,
target-evaluation or relocation call raises; static IK / screen / distance work is counted.
Labels are a NEW versioned adjudication; every historical label and file stays unchanged.
"""
from __future__ import annotations

import json
import math
import signal
import time
from pathlib import Path

import numpy as np

import config as cf
import d2_poc as dp
import data
import g4c_pilot as gp
import g4d_gate as gg
import g4d_run as gr
import geometry as ge
import motion as mo
import sector_check as sc
import skill as sk
from scene import scene_from_dict, scene_hash

P, N, U = data.POSITIVE, data.NEGATIVE, data.UNKNOWN
RUNS = {"POC_S": Path("runs/g4c_d2_poc_s"), "PILOT": Path("runs/g4c_pilot"), "G4D": Path("runs/g4d_evidence")}
PLANS = {"POC_S": "final_plan.json", "PILOT": "pilot_plan.json", "G4D": "final_plan.json"}
FROZEN = [("POC_S", u) for u in ("scene:d2_g4c_r00_AB:o05>41", "scene:d2_g4c_r00_AS:o04>56", "scene:d2_g4c_r00_AS:o05>48",
                                 "seq:d2_g4c_r00_A:empty", "seq:d2_g4c_r00_AB:empty", "seq:d2_g4c_r00_AS:empty")] + \
         [("PILOT", u) for u in ("scene:g4c_r19:o00>9", "seq:g4c_r19:o00>27", "seq:g4c_r19:o00>27.o01>28", "seq:g4c_r08:o00>15",
                                 "seq:g4c_r08:o00>47", "seq:g4c_r08:o00>47.o01>46", "seq:g4c_r08:o01>46.o00>47",
                                 "seq:g4c_r14:o02>39.o00>38", "seq:g4c_r14:o02>39.o01>7.o00>38")]
LIMITS = {"cpu_total_s": 900.0, "cpu_query_s": 60.0, "wall_total_s": 1800.0, "workers": 1}
OBLIGATIONS = {
    "enabled": {"pick_side_sector": True, "pick_side_definite_screen": True, "place_screen": False, "place_sector": "DIAGNOSTIC_ONLY"},
    "assumptions": {"A1": "ACCEPTED (user, 21 Oct 2026): declared six-phase skill, 48 grasps, fixed start; never physical impossibility",
                    "A2": "exact IK, independently re-derived (sector_check._ik) and cross-checked per candidate",
                    "A3": "joint 1 cannot wrap: limits inside (-pi, pi), checked per query", "A4": "geometry.bodies: link0 w0/2, housing1 r_h",
                    "A5": "geometry.Checker COLLISION is sound (as for TARGET_PICK / FIRST_MOVER)", "A6": "state_pose_uncertainty_m = 0",
                    "A7": "OPEN: 2 mm goal tolerance vs exact-centre placement; place side disabled", "A8": "fixed start only"}}
GUARDED = (("planner", "plan"), ("skill", "plan_skill"), ("evaluate", "evaluate_target"), ("evaluate", "evaluate_safe_retreat"),
           ("evaluate", "home_diagnostic"), ("verify", "relocate"), ("verify", "plan_fixed_order"), ("ordered", "teach_sequence"))


# ------------------------------------------------------------------ producer
def _sector(robot, cfg, scene, q0):
    k = sc.constants(robot, cfg)
    xy = np.array([o.xy_m for o in scene.objects]) - np.asarray(robot.base)
    rho, ang = np.hypot(xy[:, 0], xy[:, 1]), np.arctan2(xy[:, 1], xy[:, 0])
    reach = k["L1"] + max(k["w0_half"], k["housing_r"]) + np.array([o.radius_m for o in scene.objects]) + k["delta"] - k["eps"]
    hit = rho < reach - sc.TAU
    bars = [[o.id, float(r), float(a)] for o, r, a, h in zip(scene.objects, rho, ang, hit) if h]
    lo, hi = k["joint0_limits"]
    return bars, [max([a for _, _, a in bars if a < q0] + [lo]), min([a for _, _, a in bars if a > q0] + [hi])]


def produce(robot, cfg, scene, moved: str, scope: str, sequence, dest_cell) -> dict:
    p, o, q = sk.params(cfg), scene.get(moved), [float(v) for v in scene.robot.q_rad]
    bars, (lo, hi) = _sector(robot, cfg, scene, q[0])
    screens = sk.screen_pick(robot, mo.checkers(robot, scene, moved, None, None, p["delta"], p["eps"]), o.xy_m, o.radius_m, p)
    cands = []
    for c in screens:
        qe = None if c.stage == "NO_IK" else [float(v) for v in c.q_end]
        why = ["NO_IK"] if c.stage == "NO_IK" and c.certified else \
            [f"DEFINITE_SCREEN:{c.stage}"] if c.stage != "OK" and c.certified else []
        if qe is not None and (qe[0] < lo - sc.TAU or qe[0] > hi + sc.TAU):
            why.append("SECTOR")
        cands.append({"phi": c.phi, "branch": c.branch, "q": qe, "screen": {"stage": c.stage, "reason": c.reason,
                      "certified": bool(c.certified)}, "excluded_by": why})
    held = [x for x in scene.objects if x.id != moved]   # place side: diagnostic only, never released (A7)
    hb = _sector(robot, cfg, type(scene)(scene.root_id, scene.variant_id, scene.lattice, tuple(held), scene.robot,
                                         scene.legal_cells, scene.domain_policy, scene.robot_profile_hash), q[0])[1]
    pl = [sc._ik(robot, scene.lattice.center(dest_cell), phi, b) for phi, b in sk.directions(p)]
    diag = {"released": False, "dest_cell": dest_cell, "held_sector": hb, "no_ik": sum(x is None for x in pl),
            "outside": sum(x is not None and not hb[0] < x[0] < hb[1] for x in pl)}
    return {"version": sc.VERSION, "query": {"scope": scope, "moved": moved, "sequence": [list(a) for a in sequence]},
            "start": {"q": q, "gap": scene.robot.gripper_gap_m, "attachment": scene.robot.attachment},
            "world": sc.world_rows(scene), "scene_hash": scene_hash(scene),
            "profile": {"robot_profile_hash": cf.robot_profile_hash(cfg), "config_hash": cf.config_hash(cfg)},
            "constants": sc.constants(robot, cfg), "barriers": bars, "sector": [lo, hi], "candidates": cands,
            "place_sector_diag": diag, "claim": "CERTIFIED_INFEASIBLE" if all(c["excluded_by"] for c in cands) else "UNKNOWN"}


# ------------------------------------------------------------------ frozen queries and positive controls
def _rd(p):
    return json.loads(Path(p).read_text())


def load(run):
    d = RUNS[run]
    return _rd(d / PLANS[run]), _rd(d / "ledger.json"), _rd(d / "unit_evidence.json")


def frozen_queries(cfg, robot):
    out = []
    for run, uid in FROZEN:
        plan, led, ev = load(run)
        unit = next(u for u in plan["units"] if u["unit"] == uid)
        root = plan["roots"][unit["root"]]
        scene, seq, _, _ = gr.unit_scene(cfg, robot, plan, unit)          # producer's path (plan-checked)
        kind, d = unit["kind"], RUNS[run] / led["units"][uid]["dir"]
        scope = "SCENE_TARGET_ONLY" if kind == "SCENE_TARGET_ONLY" else "FIRST_ACTION_ORDER" if seq else "ORIGINAL_TARGET"
        if scope == "FIRST_ACTION_ORDER":
            saved = _rd(d / "relocation.json")["relocation"]["achieved_states"][0]
        else:
            saved = _rd(d / "final.json")["target_invocations"][0]
        moved, dest = (seq[0][0], seq[0][1]) if seq else ("t0", root["goal_cell"])
        src = {"scene": sc.rebuild(scene_from_dict(root["scene"]), edit=unit.get("edit")), "scope": scope, "moved": moved,
               "sequence": seq, "kind": kind, "expected_hash": unit.get("intervened_scene_hash") or root["scene_hash"]}
        same = saved["q"] == list(scene.robot.q_rad) and {k: v[2] for k, v in saved["objects"].items()} == \
            {o.id: list(o.xy_m) for o in scene.objects}
        out.append({"run": run, "unit": uid, "root": unit["root"], "scope": scope, "scene": scene, "src": src, "moved": moved,
                    "dest": dest, "seq": seq, "old_label": gp.adjudicate(ev[uid]), "saved_state_identical": same})
    return out


def controls(cfg, robot):
    """Every recorded witnessed pick (target and relocation actions) at its own start."""
    seen, out = {}, []
    for run in RUNS:
        plan, led, ev = load(run)
        for unit in plan["units"]:
            uid, st = unit["unit"], led["units"].get(unit["unit"], {})
            if "dir" not in st:
                continue
            d, root = RUNS[run] / st["dir"], scene_from_dict(plan["roots"][unit["root"]]["scene"])
            items = []
            if gp.adjudicate(ev[uid])["label"] == P and (d / "target_witness.npz").exists():
                inv = _rd(d / "final.json")["target_invocations"][0]
                items.append(("target", inv, "t0", plan["roots"][unit["root"]]["goal_cell"], np.load(d / "target_witness.npz"), 0))
            r = ev[uid].get("relocation") or {}
            if r.get("records_recertified") and (d / "relocation.npz").exists():
                states, z = _rd(d / "relocation.json")["relocation"]["achieved_states"], np.load(d / "relocation.npz")
                items += [("relocation", states[k], unit["sequence"][k][0], unit["sequence"][k][1], z, k)
                          for k in range(len(unit["sequence"])) if f"r{k}__q" in z]
            for kind, s0, moved, dest, z, k in items:
                ph = [str(x) for x in z[f"r{k}__phase"]]
                pick = z[f"r{k}__q"][ph.index("close_attach")]
                key = (cf.stable_hash(s0["objects"]), tuple(s0["q"]), moved)
                if key in seen:
                    seen[key]["witnesses"].append(f"{run}:{uid}:{kind}{k}")
                    continue
                seen[key] = {"witnesses": [f"{run}:{uid}:{kind}{k}"], "scene": sc.rebuild(root, s0["objects"], q=s0["q"]),
                             "moved": moved, "dest": dest, "pick_q": [float(v) for v in pick], "kind": kind}
                out.append(seen[key])
    return out


# ------------------------------------------------------------------ guards, counters, budget
class _Cap(Exception):
    pass


def guard_and_count(setter=setattr):
    import importlib
    calls, ops = {}, {"ik": 0, "screen_pick": 0, "distance_evals": 0, "certify": 0}
    for mod, fn in GUARDED:
        m = importlib.import_module(mod)
        calls[f"{mod}.{fn}"] = 0

        def g(*a, _k=f"{mod}.{fn}", **kw):
            calls[_k] += 1
            raise RuntimeError(f"P3a zero-call guard: {_k}")
        setter(m, fn, g)
    for name, owner, attr in (("ik", ge, "ik"), ("screen_pick", sk, "screen_pick"), ("certify", ge, "certify"),
                              ("distance_evals", ge.Checker, "distances")):
        real = getattr(owner, attr)

        def w(*a, _n=name, _f=real, **kw):
            ops[_n] += 1
            return _f(*a, **kw)
        setter(owner, attr, w)
    return calls, ops


def budgeted(fn, t0, w0):
    if time.process_time() - t0 + LIMITS["cpu_query_s"] > LIMITS["cpu_total_s"] or \
            time.time() - w0 + LIMITS["cpu_query_s"] > LIMITS["wall_total_s"]:
        return {"status": "NOT_RUN_BUDGET"}

    def stop(*_):
        raise _Cap()
    old, c0 = signal.signal(signal.SIGPROF, stop), time.process_time()
    signal.setitimer(signal.ITIMER_PROF, LIMITS["cpu_query_s"])
    try:
        return {"status": "DONE", **fn(), "cpu_s": round(time.process_time() - c0, 3)}
    except _Cap:
        return {"status": "QUERY_CPU_CAP", "cpu_s": round(time.process_time() - c0, 3)}
    finally:
        signal.setitimer(signal.ITIMER_PROF, 0)
        signal.signal(signal.SIGPROF, old)


def evaluate(cfg, robot, q):
    rec = produce(robot, cfg, q["scene"], q["moved"], q["scope"], q["seq"], q["dest"])
    return {"record": rec, "check": sc.check(robot, cfg, rec, q["src"])}


def evaluate_control(cfg, robot, c):
    rec = produce(robot, cfg, c["scene"], c["moved"], "CONTROL_FIXED_START", [], c["dest"])
    chk = sc.check(robot, cfg, rec, {"scene": c["scene"], "scope": "CONTROL_FIXED_START", "moved": c["moved"],
                                     "sequence": [], "kind": "CONTROL"})
    idx = [i for i, x in enumerate(rec["candidates"]) if x["q"] is not None and np.allclose(x["q"], c["pick_q"], rtol=0, atol=1e-7)]
    return {"witness_candidate": idx, "checker_reasons": chk["reasons"], "certified": chk["verdict"] == "CERTIFIED_INFEASIBLE",
            "witness_excluded": any(chk["excluded"][i] for i in idx), "excluded_count": sum(chk["excluded"]),
            "contradiction": chk["verdict"] == "CERTIFIED_INFEASIBLE" or any(chk["excluded"][i] for i in idx) or not idx}


# ------------------------------------------------------------------ science delta (new versioned adjudication)
def _judged(new: dict):
    """Old and new judgements of POC_S (d2_poc.judge) and the pilot (g4c_pilot.judge). New =
    historical adjudication + checker-accepted LINK0_SECTOR_PICK negatives, nothing else."""
    real, out = gp.adjudicate, {}
    for run in ("POC_S", "PILOT"):
        plan, _, ev = load(run)
        for tag, over in (("old", {}), ("new", {u: v for (r, u), v in new.items() if r == run})):
            gp.adjudicate = lambda e, _o=over: _o.get(e["unit"]) or real(e)  # noqa: E731
            try:
                out[(run, tag)] = dp.judge(plan, ev) if run == "POC_S" else gp.judge(plan, _rd(gp.PARENT), ev)
            finally:
                gp.adjudicate = real
    return out


def summarize(j) -> dict:
    roots = {}
    for k, r in j["roots"].items():
        sc_ = {kk: v["delta"] for kk, v in r["scene"].items() if v["mask"]}
        roots[k] = {"empty": r["empty_label"], "no_repair": r.get("no_repair"), "nonempty_positive": r["nonempty_positive"],
                    "order_contrasts": r["order_contrasts"], "decided_scene_effects": sc_,
                    "decided_omissions": [[o["sequence"], o["omitted_positions"], o["delta"]] for o in r["decided_omissions"]],
                    "nonadditivity": r["nonadditivity"], "bc_pool_complete": r["bc_pool"]["qualifies"],
                    "bc_decided_slots": [r["bc_pool"].get("decided_slots"), r["bc_pool"].get("active_slots")],
                    "substantive_pool": r.get("substantive_complete_pool")}
    return {"label_counts": j["label_counts"], "questions": {q: v for q, v in j.get("questions", {}).items() if len(q) == 2},
            "q_evidence": {q: v for q, v in j.get("questions", {}).items() if q in ("Q2_evidence", "Q3_evidence", "Q4_evidence",
                                                                                    "Q5_evidence")}, "roots": roots}


# ------------------------------------------------------------------ the one P3a run
def run(args) -> int:
    out, t0, w0 = Path(args.out), time.process_time(), time.time()
    if (out / "summary.json").exists() or (out / "decisions.json").exists():
        raise SystemExit("P3a output already exists; refusing to overwrite")
    out.mkdir(parents=True, exist_ok=True)
    cfg = cf.load_config(args.config)
    robot = ge.make_robot(cfg)
    before = {r: _rd(d / "manifest.json")["files"] for r, d in RUNS.items()}
    srcs = ["sector_cert.py", "sector_check.py", "skill.py", "geometry.py", "motion.py", "scene.py", "config.toml", "g4d_run.py",
            "g4c_pilot.py", "d2_poc.py", "effects.py", "data.py", "config.py"]
    cf.write_json(out / "input_manifest.json", {"sources": {f: gr._sha(Path(f)) for f in srcs},
                                                 "frozen_inputs": {f"{r}/{PLANS[r]}": gr._sha(d / PLANS[r]) for r, d in RUNS.items()},
                                                 "frozen_manifests": {r: gr._sha(d / "manifest.json") for r, d in RUNS.items()},
                                                 "frozen_targets": [list(x) for x in FROZEN], "limits": LIMITS})
    cf.write_json(out / "proof_obligations.json", {"theorem": sc.__doc__, "version": sc.VERSION, "constants": sc.constants(robot, cfg),
                                                   "tau": sc.TAU, "joint1_limits": list(robot.limits[0]), **OBLIGATIONS})
    calls, ops = guard_and_count()
    queries, ctrls = frozen_queries(cfg, robot), controls(cfg, robot)
    rows, decisions, new = [], [], {}
    for q in queries:
        r = budgeted(lambda: evaluate(cfg, robot, q), t0, w0)
        chk = r.get("check", {})
        acc = r["status"] == "DONE" and chk.get("accepted") and q["saved_state_identical"] and q["old_label"]["label"] == U
        clash = bool(chk.get("accepted")) and q["old_label"]["label"] == P          # certificate vs witness: stage failure
        rec = r.get("record") or {"candidates": [], "place_sector_diag": None}
        for i, c in enumerate(rec["candidates"]):
            rows.append({"unit": q["unit"], "i": i, **c, "sector": rec["sector"], "checker_excluded": chk["excluded"][i]})
        by = {k: sum(any(x.startswith(k) for x in c["excluded_by"]) for c in rec["candidates"]) for k in ("NO_IK", "DEFINITE", "SECTOR")}
        un = [rec["candidates"][i] for i in chk.get("unexcluded", [])]
        decisions.append({"run": q["run"], "unit": q["unit"], "scope": q["scope"], "old_label": q["old_label"],
                          "status": r["status"], "cpu_s": r.get("cpu_s"), "saved_state_identical": q["saved_state_identical"],
                          "new_verdict": "CERTIFIED_INFEASIBLE" if acc else "UNKNOWN", "checker_accepted": bool(chk.get("accepted")),
                          "checker_reasons": chk.get("reasons"), "barriers": rec.get("barriers"), "sector_deg": [
                              round(math.degrees(v), 3) for v in rec.get("sector", [])], "exclusion_counts": by,
                          "unexcluded": [{"phi_deg": round(math.degrees(c["phi"]), 1), "branch": c["branch"],
                                          "q0_deg": c["q"] and round(math.degrees(c["q"][0]), 2), "screen": c["screen"]} for c in un],
                          "place_sector_diag": rec["place_sector_diag"], "contradicts_old_positive": clash})
        if acc:
            new[(q["run"], q["unit"])] = {"label": N, "reason": "LINK0_SECTOR_PICK", "scope": [
                "ORDER" if q["scope"] == "FIRST_ACTION_ORDER" else q["scope"]]}
    crow = []
    for c in ctrls:
        r = budgeted(lambda: evaluate_control(cfg, robot, c), t0, w0)
        crow.append({"witnesses": c["witnesses"], "kind": c["kind"], "moved": c["moved"], "start_q": list(c["scene"].robot.q_rad), **r})
    static_cpu, static_wall = time.process_time() - t0, time.time() - w0
    ops_static = dict(ops)
    judged = _judged(new)
    sci = {f"{run}_{tag}": summarize(j) for (run, tag), j in judged.items()}
    contra = [c for c in crow if c.get("status") != "DONE" or c.get("contradiction")]
    after = {r: _rd(d / "manifest.json")["files"] for r, d in RUNS.items()}
    intact = before == after and all(gr._sha(d / f) == h for r, d in RUNS.items() for f, h in after[r].items())
    with open(out / "candidates.jsonl", "w") as fh:
        for x in rows:
            fh.write(json.dumps(x) + "\n")
    cf.write_json(out / "decisions.json", decisions)
    cf.write_json(out / "controls.json", {"n_starts": len(crow), "n_witnesses": sum(len(c["witnesses"]) for c in crow),
                                          "contradictions": len(contra), "rows": crow})
    cf.write_json(out / "science.json", sci)
    cf.write_json(out / "calls.json", {"guarded_calls": calls, "static_ops": ops_static, "static_cpu_s": round(static_cpu, 2),
                                       "static_wall_s": round(static_wall, 2), "limits": LIMITS})
    ok = not any(calls.values()) and not contra and intact and all(d["status"] == "DONE" for d in decisions) and \
        not any(d["contradicts_old_positive"] for d in decisions)
    summ = {"stage": "G4C_D2_P3A_STATIC", "certificate": sc.VERSION, "validated_run": ok, "frozen_runs_intact": intact,
            "newly_certified": [list(k) for k in new], "decisions": {d["unit"]: d["new_verdict"] for d in decisions},
            "controls": {"starts": len(crow), "contradictions": len(contra)}, "guarded_calls": calls, "static_ops": ops_static,
            "static_cpu_s": round(static_cpu, 2), "static_wall_s": round(static_wall, 2),
            "total_cpu_s": round(time.process_time() - t0, 2), "total_wall_s": round(time.time() - w0, 2)}
    cf.write_json(out / "summary.json", summ)
    figure(out, robot, cfg, queries, decisions, rows)
    gg.write_manifest(out, {"none": "deterministic static computation"}, cf.config_hash(cfg), {"certificate": sc.VERSION})
    print(json.dumps({k: summ[k] for k in ("validated_run", "newly_certified", "decisions", "controls", "guarded_calls",
                                             "static_ops", "static_cpu_s", "static_wall_s")}))
    return 0 if ok else 1


def figure(out, robot, cfg, queries, decisions, rows):
    import viz
    from matplotlib.patches import Wedge
    n, k0 = len(queries), sc.constants(robot, cfg)
    f, axs = viz.plt.subplots((n + 4) // 5, 5, figsize=(31, 6.4 * ((n + 4) // 5)), squeeze=False)
    col = {"no IK": "#9e9e9e", "definite pick-screen rejection": viz.COLORS["invalid"], "outside link-0 sector":
           viz.COLORS["hypothetical"], "NOT excluded": "#ff9800"}
    for ax, q, d in zip(axs.ravel(), queries, decisions):
        s = q["scene"]
        viz.scene_panel(ax, s, cfg, f"{q['unit']}\n{q['scope']}: {d['new_verdict']}")
        ax.set(xlim=viz.LIMITS["grid"][0], ylim=viz.LIMITS["grid"][1])
        viz.draw_robot(ax, robot, s.robot.q_rad, s.robot.gripper_gap_m, alpha=0.7, zorder=6)
        r0 = k0["L1"] + max(k0["w0_half"], k0["housing_r"]) + 0.024 + k0["delta"] - k0["eps"]
        ax.add_patch(Wedge(robot.base, r0, d["sector_deg"][1], d["sector_deg"][0] + 360, fc="#cfd8dc", alpha=0.45, zorder=2))
        for x in (r for r in rows if r["unit"] == q["unit"] and r["q"] is not None):
            e = x["excluded_by"]
            c = list(col)[0 if "NO_IK" in e else 1 if any(v.startswith("DEF") for v in e) else 2 if "SECTOR" in e else 3]
            ax.plot([robot.base[0], robot.base[0] + 0.42 * math.cos(x["q"][0])],
                    [robot.base[1], robot.base[1] + 0.42 * math.sin(x["q"][0])], color=col[c], lw=1.0, alpha=0.8, zorder=7)
        ax.text(0.01, 0.99, "STATIC PROOF SCHEMATIC (not a trajectory)\nlines: link-0 angle of each declared pick\ngrey "
                f"wedge: link-0 angles barred by the annulus\nproved excluded {48 - len(d['unexcluded'])}/48", transform=ax.transAxes,
                va="top", fontsize=7, family="monospace", bbox=dict(fc="white", ec="#999"))
    for a in axs.ravel()[n:]:
        a.axis("off")
    for k, c in col.items():
        axs[0, 0].plot([], [], color=c, label=k)
    axs[0, 0].legend(fontsize=7, loc="lower right")
    f.tight_layout()
    f.savefig(out / "A_sector_proofs.png", dpi=55)
    viz.plt.close(f)
