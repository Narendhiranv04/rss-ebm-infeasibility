"""G4C-D2-P0 runner: writes runs/g4c_d2_precheck/ (source_audit.json, construction_attempts.jsonl,
precheck_variants.json, precheck_queries.json, precheck_coverage.json, budget_request.json,
precheck_overview.png, protocol_proposal.md is authored separately, manifest.json).

Static geometry only; any loaded planner/skill/target/teacher entry point is wrapped with a
call counter (expected 0). New statuses are restricted to PRECHECK_STATIC,
BASE_WITNESS_REPLAYED, NOT_RUN and ARCHIVED_EVIDENCE.
"""
from __future__ import annotations

import collections
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch

import config as cf
import d2
import g4b_figures as g4bf
import g4d_gate as gg
import geometry as ge
import motion as mo
import viz
from scene import scene_to_dict
from viz import COLORS, plt

FAMILIES = {"D2-F0": ("base", "+N"), "D2-F1": ("+A", "+B"), "D2-F2": ("+G",), "D2-F3": ("+A+B",), "D2-F4": ("+A+S",),
            "D2-F5": ("+A+B",), "D2-F6": ("+N", "+A+N"), "D2-F7": ("+A", "+G"), "D2-F8": ("+A+B+C",), "D2-F9": ("base",)}
CLAIM = {"D2-F0": "both original/edit target successes, delta 0", "D2-F1": "whether ANY alternative target trajectory succeeds",
         "D2-F2": "certified no-repair NEG + alternative successful destinations", "D2-F3": "same endpoints, opposite orders decided",
         "D2-F4": "PLACE-scope order NEG vs witnessed reversed order", "D2-F5": "four decided S/A/B/A+B target-only outcomes",
         "D2-F6": "POS<->NEG when sound, POS<->POS nuisance", "D2-F7": "multiple complete successful repairs; J vs C_d",
         "D2-F8": "q-continuity across 3 moves, omissions, safe retreat", "D2-F9": "decided outcomes at each valid start"}
SOURCES = ("d2.py", "d2_precheck.py", "tests/test_d2.py", "motion.py", "geometry.py", "scene.py", "sequences.py", "effects.py",
           "proposals.py", "g4c.py", "data.py", "config.py", "config.toml", "config_v2.toml")


def pilot_unknowns(root: Path) -> dict:
    """The 20 G4C-P1 UNKNOWNs by existing reason (read-only; nothing relabelled or rerun)."""
    lab = json.loads((root / "runs/g4c_pilot/pilot_coverage.json").read_text())["labels"]
    rows = {u: v["reason"] for u, v in lab.items() if v["label"] == "UNRESOLVED"}
    cls = {u: ("NEEDS_NEW_SOUND_CERTIFICATE_KIND (q0 all-pairs-blocked connector verdict has no checker)"
               if "ALL_PAIRS_CERTIFIED_BLOCKED" in r else
               "SEARCH_INCOMPLETENESS (cannot be avoided without hindsight; D2 release policy excludes the whole pool)")
           for u, r in rows.items()}
    return {"units": rows, "class": cls, "counts": dict(collections.Counter(cls.values())), "relabelled": 0, "rerun": 0}


def audit(root: Path, bases: list) -> dict:
    git = lambda *a: subprocess.run(["git", *a], cwd=root, capture_output=True, text=True).stdout.strip()  # noqa: E731
    return {"git": {"branch": git("branch", "--show-current"), "head": git("rev-parse", "HEAD"),
                    "untracked": git("status", "--short").splitlines()},
            "sources": {f: d2._sha(root / f) for f in SOURCES if (root / f).exists()},
            "frozen_inputs": {f: d2._sha(root / f) for f in ("runs/g4c_preflight/plan.json", "runs/g4c_pilot/pilot_plan.json",
                                                            "runs/g4c_pilot/authorization.json", "runs/g4c_pilot/pilot_result.json",
                                                            "runs/g4d_evidence/final_plan.json", "runs/g4d_evidence/result.json")},
            "bases": [b["provenance"] | {"family": b["family"], "family_id": b["family_id"]} for b in bases],
            "interface_map": {
                "reused_unchanged": {"motion.certify_trajectory": "conservative continuous replay of a recorded record (full body)",
                                     "geometry.Checker.status_at": "initial-robot clearance", "scene.Obj/replace": "clone + new obstacle",
                                     "sequences.parse_sequence/omit/endpoint_distance": "canonical grammar, omission, C_d",
                                     "proposals.static_endpoints": "static endpoint screen", "effects.pose_substitution": "fixed-robot "
                                     "eligibility + changed-field audit", "effects.robot_start_substitution": "robot-start variants",
                                     "g4c._dests / g4c.layout": "geometric destination rule; bounded B/C mask",
                                     "data.learner_tensors_v2": "learner allowlist (leakage tests)"},
                "new_here": {"d2.py": "static D2 construction + release semantics", "d2_precheck.py": "artifacts"},
                "later_D2_P1_only": {"base witness for a fresh scene": "evaluate.evaluate_target (SAFE_RETREAT) under teacher.guard",
                                     "labels": "g4d_run.run_collection/unit_evidence + g4c_pilot.adjudicate (reused, unchanged)",
                                     "finalize/judge": "a D2 adapter analogous to g4c_pilot (not written)"}},
            "missing_or_limited": [
                "certify_trajectory reports the FIRST violating pair per segment, not every body pair (diagnostic lists first pairs)",
                "no checker-accepted certificate kind for full-domain target/connector impossibility (e.g. q0 all-pairs-blocked)",
                "no D2 finalize/judge adapter yet (would mirror g4c_pilot)"]}


def overview(out, robot, cfg, recs):
    f, axs = plt.subplots(1, len(recs), figsize=(9.5 * len(recs), 9))
    for ax, (b, ch, vs) in zip(np.atleast_1d(axs), recs):
        s, a = b["scene"], b["arrays"]
        viz.scene_panel(ax, s, cfg, f"{b['family']} ({b['provenance']['archived_unit']})")
        ax.set(xlim=viz.LIMITS["grid"][0], ylim=viz.LIMITS["grid"][1])
        for q, gap, ph in mo.samples(a, 14):
            viz.draw_robot(ax, robot, q, gap, alpha=0.12, zorder=4)
        viz.draw_robot(ax, robot, s.robot.q_rad, s.robot.gripper_gap_m, alpha=0.8, zorder=6)
        tp = np.array([mo.target_xy(robot, a, ph, q, s.get("t0").xy_m) for q, _, ph in mo.samples(a, 200)])
        ax.plot(tp[:, 0], tp[:, 1], color=COLORS["target"], lw=1.8, zorder=7, label="recorded carried-target path")
        g = s.lattice.center(b["goal_cell"])
        ax.add_patch(Circle(g, 0.024, fc="none", ec=COLORS["target_edge"], ls="--", lw=1.5, zorder=9))
        cols = {"A": "#c0392b", "B": "#8e44ad", "C": "#d35400", "G": "#7f8c8d", "S": "#16a085", "N": COLORS["valid"]}
        for k, c in ch.items():
            if c is None:
                continue
            xy = s.lattice.center(c["cell"])
            ax.add_patch(Circle(xy, 0.024, fc="none", ec=cols[k], lw=2.2, ls="-", zorder=10))
            ax.text(xy[0], xy[1] + 0.033, f"{k}: {c['role']}", fontsize=6.5, ha="center", color=cols[k], weight="bold", zorder=11)
            fi = c.get("first_interruption")
            if fi:
                q = a["q"][fi["segment"]]
                viz.draw_robot(ax, robot, q, float(a["gap"][fi["segment"]]), alpha=0.35, zorder=5, highlight=(fi["pair"].split("|")[0],))
        v3 = next((v for v in vs if v["variant"] == "+A+B+C" and v["status"] == "PRECHECK_STATIC"), None)
        if v3:
            Q = d2.variant_queries(robot, cfg, v3)
            for o, ds in Q["destinations"].items():
                if ds:
                    p0 = v3["scene"].get(o).xy_m
                    ax.add_patch(FancyArrowPatch(p0, v3["scene"].lattice.center(ds[0]), arrowstyle="-|>", mutation_scale=9,
                                                 color=COLORS["hypothetical"], ls="--", zorder=9))
                    ax.add_patch(Circle(v3["scene"].lattice.center(ds[0]), 0.024, fc="none", ec=COLORS["hypothetical"], ls="--"))
        ax.text(0.01, 0.99, "ARCHIVED VERIFIED NOMINAL TARGET TRAJECTORY (faint arms, orange path)\n"
                "CONSTRUCTED OBSTACLE PREVIEW (coloured rings; red outline = first colliding body)\n"
                "dashed blue = predeclared ghost relocations of +A+B+C\nNEW TARGET FEASIBILITY: NOT RUN",
                transform=ax.transAxes, va="top", fontsize=8, family="monospace", bbox=dict(fc="white", ec=COLORS["muted"]))
    f.tight_layout()
    f.savefig(out / "precheck_overview.png", dpi=70)
    plt.close(f)


TIERS = {"POC_S": ({"g4d_R1", "g4c_r00"}, {"base", "+A", "+A+B", "+A+S", "+G", "+N"}, 2),
         "POC_F": ({"g4c_r00", "g4d_R0", "g4d_R1"}, {v for v, _ in d2.SPECS}, 4)}


def budget(variants_q: dict, root: Path) -> dict:
    """Proposed D2-P1 POC numbers per tier (NOT authorization), from exact planned unit counts."""
    led = json.loads((root / "runs/g4c_pilot/budget_ledger.json").read_text())["units"]
    cpu = sorted(v["cpu_s"] for v in led.values())
    up = lambda x: int(math.ceil(x / 1800.0) * 1800)  # noqa: E731
    out = {"status": "REQUESTED_NOT_AUTHORIZED", "basis": {"g4c_p1_unit_cpu_s": {"n": len(cpu), "median": cpu[len(cpu) // 2],
           "max": cpu[-1], "mean": round(sum(cpu) / len(cpu), 2)}, "slow_target_share_proxy": 0.121},
           "rule": "high = 12.1% of units at the 870 s cancel + every other unit at the pilot maximum; aggregate = 1.1 x high + "
                   "fresh-base attempts at 900 s, rounded up to 30 CPU-min; wall = 1.25 x aggregate; STOP after the POC",
           "fresh_bases": "optional new S_base scenes (G4C geometric construction rule) whose target witness is planned fresh: "
                          "one target evaluation each; a failed base is recorded, never replaced beyond the attempt cap"}
    for tier, (fams, vnames, fresh) in TIERS.items():
        u = {"ordered_teacher": 0, "scene_target_only": 0, "robot_start": 0, "relocation_skills_max": 0}
        for fam, vs in variants_q.items():
            for name, v in vs.items():
                if fam in fams and name in vnames:
                    u["ordered_teacher"] += len(v["pool"])
                    u["relocation_skills_max"] += sum(len(p["sequence"]) for p in v["pool"])
                    u["scene_target_only"] += sum(q["eligible"] for q in v["scene_queries"])
                    u["robot_start"] += sum(x["eligible"] for x in v.get("robot_starts", [])[:1])
        n = u["ordered_teacher"] + u["scene_target_only"] + u["robot_start"]
        high = 0.121 * n * 870 + n * cpu[-1]
        out[tier] = {"families": sorted(fams), "variants": sorted(vnames), "planned_units": {**u, "total_units": n},
                     "fresh_base_attempts": fresh,
                     "estimate_cpu_s": {"low": round(n * cpu[len(cpu) // 2]), "central": round(n * sum(cpu) / len(cpu)),
                                        "high": round(high)},
                     "requested": {"per_teacher_unit_cpu_s": 900, "cancellation_margin_s": 30,
                                   "aggregate_cpu_s": up(1.1 * high + fresh * 900), "workers": 1, "unit_wall_guard_s": 1800,
                                   "aggregate_wall_guard_s": up(1.25 * (1.1 * high + fresh * 900)), "max_motion_nodes": 4000,
                                   "relocation_skill_attempt_cap": u["relocation_skills_max"], "target_eval_attempt_cap": n + fresh,
                                   "max_root_construction_attempts": fresh}}
    return out


def run(args) -> int:
    calls = g4bf._instrument()
    t0, c0, out, root = time.perf_counter(), time.process_time(), Path(args.out), Path(".")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "precheck_variants.json").exists():
        raise SystemExit("precheck outputs exist; refusing to overwrite")
    cfg = cf.load_config(args.config)
    robot = ge.make_robot(cfg)
    bases = [d2.load_base(s) for s in d2.BASES]
    attempts, variants, queries, recs, replays = [], {}, {}, [], {}
    for b in bases:
        replays[b["family"]] = d2.replay_base(robot, cfg, b)
        at = d2.insertion_attempts(robot, cfg, b)
        ch = d2.choose_blockers(at)
        vs = d2.nested_variants(robot, cfg, b, ch)
        attempts += at
        st = next((v["added"]["S"] for v in vs if v["variant"] == "+A+S" and "added" in v), None)
        attempts.append({"family": b["family"], "cell": st and st["cell"], "accepted": st is not None,
                         "reason": "staging_slot_occupant on A's first destination" if st else "staging occupant unavailable"})
        variants[b["family"]] = [{k: (scene_to_dict(x) if k == "scene" else x) for k, x in v.items()} for v in vs]
        queries[b["family"]] = {v["variant"]: d2.variant_queries(robot, cfg, v) for v in vs if v["status"] == "PRECHECK_STATIC"}
        recs.append((b, ch, vs))
    cov = {}
    for fam, vnames in FAMILIES.items():
        cov[fam] = {"claim_requires": CLAIM[fam], "label_outcomes": "NOT_RUN", "statically_eligible": {
            b: [v for v in vnames if v in queries[b]] for b in queries}}
    with open(out / "construction_attempts.jsonl", "w") as fh:
        for a in attempts:
            fh.write(json.dumps(a, sort_keys=True) + "\n")
    cf.write_json(out / "precheck_variants.json", {"version": d2.VERSION, "replays": replays, "variants": variants,
                                                   "splits": {b["family"]: d2.family_split(b["family_id"]) for b in bases}})
    cf.write_json(out / "precheck_queries.json", queries)
    cf.write_json(out / "precheck_coverage.json", {"families": cov, "pilot_unknowns": pilot_unknowns(root)})
    cf.write_json(out / "budget_request.json", budget(queries, root))
    cf.write_json(out / "source_audit.json", audit(root, bases))
    overview(out, robot, cfg, recs)
    acc = collections.Counter(a["reason"] if not a["accepted"] else "accepted" for a in attempts)
    statuses = sorted({v["status"] for vs in variants.values() for v in vs} | {r["status"] for r in replays.values()})
    res = {"replays": {k: v["status"] for k, v in replays.items()}, "attempts": dict(acc), "variant_statuses": statuses,
           "calls": calls, "cpu_s": round(time.process_time() - c0, 1), "wall_s": round(time.perf_counter() - t0, 1),
           "new_labels": 0}
    cf.write_json(out / "precheck_result.json", res)
    gg.write_manifest(out, {"none": "static precheck (no planner seeds used)"}, cf.config_hash(cfg), {"version": d2.VERSION})
    print(json.dumps(res))
    return 0
