"""P15 ledger (runs/p15/guards, contract caps, one guarded OS process per job), P15-0 source / evidence audit, benchmark runners, summaries,
B closure and figures. P10-P14 artifacts are read-only; every P15 output is a new versioned file under runs/p15."""
from __future__ import annotations

import collections
import json
import sys
import time
from pathlib import Path

import numpy as np

import config as cf

OUT = Path("runs/p15")
P14 = Path("runs/p14")
CAPS = {"audit": 3000.0, "retarget_probe": 8000.0, "retarget_train": 10000.0, "retarget_teacher": 15000.0, "index": 6000.0,
        "search_dev": 8000.0, "retro24": 27000.0, "prospective": 38000.0, "ablation": 12000.0, "b_diag": 3000.0, "regression": 3000.0}
JOB = {"cpu_s": 6500.0, "wall_s": 60000.0}


def rd(p):
    return json.loads(Path(p).read_text())


def sha(p) -> str:
    import hashlib
    return "sha256:" + hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spent(cat=None) -> float:
    return sum(g["cpu_s"] for g in (rd(p) for p in (OUT / "guards").glob("*.json")) if cat is None or g["category"] == cat)


def job(name, call, cpu, module="p15_bench", category="audit", wall=None) -> dict:
    """One guarded child per OS process (never threads in one parent), charged to one P15 category; refuses past the cap."""
    import teacher
    (OUT / "guards").mkdir(parents=True, exist_ok=True)
    cpu = min(cpu, JOB["cpu_s"], CAPS[category] - spent(category))
    if cpu < 5:
        raise SystemExit(f"{category} budget exhausted")
    g = teacher.guard([sys.executable, "-c", f"import torch; torch.set_num_threads(1); import {module}; {module}.{call}"], cpu,
                      wall or JOB["wall_s"], OUT / f"heartbeat_{name}.txt", 15.0, 0.25, Path("."))
    n = 1
    while (OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json").exists():
        n += 1
    rec = {k: g[k] for k in ("stop", "returncode", "cpu_s", "wall_s", "overshoot_s", "stderr_tail")} | \
          {"cpu_cap_s": cpu, "call": f"{module}.{call}", "category": category, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    cf.write_json(OUT / "guards" / f"{name}{'' if n == 1 else f'_attempt{n}'}.json", rec)
    (OUT / f"heartbeat_{name}.txt").unlink(missing_ok=True)
    print(json.dumps({k: rec[k] for k in ("stop", "returncode", "cpu_s", "wall_s")} | {category: round(spent(category), 1)}))
    return rec


def gate(stage, **kw) -> None:
    cf.write_json(OUT / f"P15_{stage}_GATE.json", {"stage": f"P15-{stage}", **kw, "spend": {c: round(spent(c), 1) for c in CAPS}})


# ------------------------------------------------------------------ P15-0 audit
def audit() -> dict:
    """Recompute P14 D3 (identical-cohort) and matched-admission tables from stored JSON; verify P14 manifest hashes; claim matrix."""
    import p14_audit as pa
    m = rd(P14 / "P14_FINAL_MANIFEST.json")
    mism = [p for sec in ("P14_A", "P14_D", "sources", "figures") for p, h in m[sec].items() if sha(p) != h]
    runs = {k: rd(P14 / "search" / f"EVAL_{k}.json") for k in ("D", "D-pairless", "geometry", "uninformed", "oracle")}
    common = sorted(set.intersection(*[set(v) for v in runs.values()]))
    ok = lambda v: bool(v.get("found") and (v.get("replay") or {}).get("success"))  # noqa: E731
    d3 = {k: {"success": sum(ok(v[r]) for r in common), "n": len(common), "cpu_mean": float(np.mean([v[r]["cpu_s"] for r in common])),
              "roots_completed": len(v)} for k, v in runs.items()}
    rep = rd(P14 / "eval" / "D3_SUMMARY.json")["methods"]
    d3_agree = all(abs(d3[k]["success"] / d3[k]["n"] - rep[k]["success"]) < 1e-12 for k in d3)
    ma = rd(P14 / "calibration" / "MATCHED_ADMISSION.json")["summary"]
    adm = {k: {"val_coverage_mean": float(np.mean(v["val_coverage"])), "test_admission": v["test_admission"], "executed_success": v["executed_success_lower_bound"]}
           for k, v in ma.items() if k.split("|")[1] == "coverage_matched"}
    claims = {
        "first_certified_witness_only": "p14_d_search._move returns at the first CLEAR candidate: alternative modes / terminal q are dropped; minimum travel over modes NOT established",
        "target_unknown_as_none": "p14_d_search._target returns None for any non-VERIFIED planner outcome: UNKNOWN collapsed with failure; depth_exhaustive is not a physical proof",
        "restricted_graph": "movers from ORDERED_TEACHER tokens, move-once, K slots, depth <= 3: a restricted finite graph",
        "root_variant_goal": "p14 root_variant kept R.goal and the original visual tokens; the physical probe used another destination: unmatched comparison",
        "unnecessary_moves": "counted moves outside the ORIGINAL blocker union; not redundant moves",
        "no_similarity_threshold": "lookup ranks only; no validated minimum-similarity cut",
        "validation_vs_test_admission": "coverage matched on VALIDATION; test admissions differ (see table)"}
    out = {"p14_manifest_mismatches": mism, "replay_fidelity_sample": rd(P14 / "lookup" / "PROVENANCE_AUDIT.json")["trace_fidelity_sample"],
           "d3_common_roots": common, "d3_recomputed": d3, "d3_matches_report": d3_agree, "coverage_matched_admission": adm,
           "index": {"entries": len(rd(P14 / "lookup" / "INDEX.json")["entries"]), "excluded_roots": len(rd(P14 / "eval_roster.json")["excluded_roots"])},
           "p14_roster_primary": rd(P14 / "P14_PREREG.json")["D_track"]["roster_primary"], "claims": claims,
           "historical_hashes": {p: sha(p) for p in ("runs/p14/P14_FINAL_MANIFEST.json", "runs/p14/P14_PREREG.json", "runs/p14/lookup/INDEX.json",
                                                     "runs/p14/eval/D3_SUMMARY.json", "runs/p14/calibration/MATCHED_ADMISSION.json")},
           "p14_spend": {c: round(pa.spent(c), 1) for c in pa.CAPS}}
    (OUT / "audit").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "audit" / "P15_0_AUDIT.json", out)
    return out


# ------------------------------------------------------------------ P15-4 / 5: guidance policies and the benchmark runner
GUIDES = ("D", "D-pairless", "geometry", "uniform", "oracle")


class Guide:
    """Reorder-only guidance. D / D-pairless: original-target blocker score from the rebuilt current state (p15_retarget.build_record);
    recursion: an object whose RECALIBRATED temporary-target feasibility (to its first candidate slot) is < 0.5 is deferred behind the
    other candidates. geometry: analytic heuristic on updated positions; geometry recursion defers an object with heuristic blockers among
    the other movers. uniform: catalogue order. oracle: original teacher blocker union (non-deployable), no recursion."""

    def __init__(self, kind, root, rr, recursion=True):
        import p14_d_search as ds
        self.kind, self.root, self.rr, self.recursion = kind, root, rr, recursion and kind != "oracle" and kind != "uniform"
        self.models = ds.d_models(kind == "D-pairless") if kind in ("D", "D-pairless") else []
        self.recal = rd(OUT / "retarget" / "RECALIBRATION.json")["fit_all_dev"]
        self.cache, self.d_calls, self.d_cpu = {}, 0, 0.0
        self.oracle = None

    def _d(self, phys, cells, q, target, goal, pool):
        import p15_retarget as pr
        k = (tuple(sorted(cells.items())), tuple(np.round(q, 12)), target, goal, tuple(pool))
        if k not in self.cache:
            t0 = time.process_time()
            s = pr.retarget_scene(phys.s0, cells, target, q)
            self.cache[k] = pr.d_predict(self.models, pr.build_record(phys.cfg, s, target, goal), target, list(pool) + ["t0"])
            self.d_calls += 1
            self.d_cpu += time.process_time() - t0
        return self.cache[k]

    def _geo(self, phys, cells, q, target, pool):
        import torch
        import d_blockers as db
        import p14_d_search as ds
        ids = self.rr["ids"]
        xy = torch.as_tensor(np.array([phys.s0.lattice.center(cells[o]) for o in ids]), dtype=torch.float64)
        R2 = ds.root_variant(self.rr["R"], phys.robot, xy, ids.index(target), q)
        return dict(zip(pool, db.heuristic(R2, [ids.index(o) for o in pool])))

    def order(self, phys, movers):
        def f(cells, q, moved, cands):
            if self.kind == "uniform" or not cands:
                return cands
            pool = [o for o in movers if o not in moved]
            if self.kind == "oracle":
                sc = {o: float(self.oracle.get(o, 0)) for o in pool}
            elif self.kind == "geometry":
                sc = {o: float(v) for o, v in self._geo(phys, cells, q, "t0", pool).items()}
            else:
                pr = self._d(phys, cells, q, "t0", phys.goal_cell, pool)
                sc = {o: (1 - pr["p_feasible"]) * m for o, m in zip(pr["pool"], pr["marg"])}
            defer = set()
            if self.recursion:
                first = {}
                for o, d in cands:
                    first.setdefault(o, d)
                for o, d in first.items():
                    others = [x for x in pool if x != o]
                    if self.kind == "geometry":
                        if any(self._geo(phys, cells, q, o, others).values()):
                            defer.add(o)
                    else:
                        p = self._d(phys, cells, q, o, d, others)["p_feasible"]
                        lg = np.log(max(p, 1e-9) / max(1 - p, 1e-9))
                        if 1 / (1 + np.exp(-(lg - self.recal["bias"]) / self.recal["temperature"])) < 0.5:
                            defer.add(o)
            return sorted(cands, key=lambda c: (c[0] in defer, -sc.get(c[0], 0.0), cands.index(c)))
        return f


def _slots(phys, idx):
    import p6_scenes as p6
    T = phys.s0.get("t0").cell

    def f(cells, o):
        s = phys.scene(cells, phys.s0.robot.q_rad)
        ds_ = p6._dests(phys.scr, s, o, [T, phys.goal_cell], [], 4)
        occ = set(cells.values())
        ds_ += sorted(d for (src, d) in idx.by_key if src == cells[o] and d not in ds_ and d in phys.scr.valid and d not in occ
                      and p6._cheb(d, T) >= 2 and p6._cheb(d, phys.goal_cell) >= 2)
        return ds_[:6]
    return f


def run_arm(guide, roster, out_name, split="test", recursion=True, budget=None, cpu_cap=400.0, lookup="combined", modal_only=False, use_index=True):
    """One process per arm; writes after every root. Catalogue-restricted action domain (declared), identical for every guide."""
    import p14_eval as pe
    import p15_lookup as pl
    import p15_search as ps
    rs, P, D = pe._roots(split) if split in ("test", "dev") else _prosp_roots()
    idx, _ = pl.index_v2()
    res = rd(OUT / "search" / f"{out_name}.json") if (OUT / "search" / f"{out_name}.json").exists() else {}
    for k in roster:
        if k in res:
            continue
        root, rr, dl = P[k], rs[k], D[k]
        c0, w0 = time.process_time(), time.time()
        phys = ps.Physics(root, idx if use_index else None, lookup, modal_only, budget, cpu_cap)
        g = Guide(guide, root, rr, recursion)
        g.oracle = {o: m for o, m in zip(dl["pool"], dl["mask"])} if dl.get("mask") else ({o: 0 for o in dl["pool"]} if dl.get("y0") == 1 else {})
        movers = sorted({o for it in root["items"] if it["kind"] == "ORDERED_TEACHER" for o, _ in it["sequence"]})
        out = ps.search(phys, g.order(phys, movers), movers, _slots(phys, idx), {o.id: o.cell for o in phys.s0.objects}, phys.s0.robot.q_rad)
        rep = ps.replay(phys, out["plan"]) if out["found"] else None
        union = {o for o, m in zip(dl["pool"], dl["mask"]) if m} if dl.get("mask") else (set() if dl.get("y0") == 1 else None)
        moved = [s["object"] for s in out["plan"]["steps"]] if out["found"] else []
        res[k] = {"family": root["family"], "y0": dl.get("y0"), "found": out["found"], "replay_success": rep, "claim": out["claim"],
                  "min_moves_proven": out["min_moves_proven"], "min_travel_proven": out["min_travel_proven"], "flags": out["flags"],
                  "plan": None if not out["found"] else {"moves": out["plan"]["moves"], "travel": out["plan"]["travel"], "steps": out["plan"]["steps"]},
                  "moves_outside_original_blocker_union": None if union is None or not out["found"] else sum(o not in union for o in moved),
                  "log": out["log"], "used": phys.used, "stats": phys.stats, "truncated": phys._trunc, "d_calls": g.d_calls, "d_cpu_s": g.d_cpu,
                  "cpu_s": time.process_time() - c0, "wall_s": time.time() - w0, "guide": guide, "recursion": g.recursion}
        (OUT / "search").mkdir(parents=True, exist_ok=True)
        cf.write_json(OUT / "search" / f"{out_name}.json", res)


# ------------------------------------------------------------------ prospective cohort (new seeds; frozen before any outcome)
PROSP = OUT / "prospective"
PROSP_FAMS = ("PC_SG", "PR_SIDES", "PN_SIMP", "PN_GIMP", "ORD_CHAIN", "OMIT_OPT")


def build_prospective(seed0=2026200000) -> None:
    """32 new roots: 4 per executable mechanism family (mechanism-conditioned) + 8 NAT (unconditioned, feasible controls). Labels later: only
    the original task and the parking-lattice edits are teacher-labelled (oracle arm / blocker-union metric)."""
    import p13_pipeline as pp
    pd = pp._register()
    specs = [{"family": f, "seed": seed0 + 10 * i + j, "index": 4 * i + j, "wave": 0, "batch": 1, "split": "blind"} for i, f in enumerate(PROSP_FAMS) for j in range(4)]
    specs += [{"family": "NAT", "seed": seed0 + 100 + j, "index": 24 + j, "wave": 0, "batch": 1, "split": "blind"} for j in range(8)]
    if not (PROSP / "plan.json").exists():
        pd.build_plan(PROSP, specs, max_attempts=600, root_cpu=300.0)
    plan = rd(PROSP / "plan.json")
    keep = lambda q: q["token"] == "" or str(q["token"]).startswith("edit:")  # noqa: E731
    plan["queries"] = [q for q in plan["queries"] if keep(q)]
    qs = {q["query"] for q in plan["queries"]}
    plan["units"] = [u for u in plan["units"] if u["query"] in qs]
    plan["label_scope"] = "original task + parking-lattice edits only (oracle / blocker union); search outcomes are the primary physical endpoint"
    plan["plan_hash"] = cf.stable_hash({x: v for x, v in plan.items() if x != "plan_hash"})
    cf.write_json(PROSP / "plan.json", plan)


def prospective_records() -> None:
    """Unlabelled learner records (same pipeline as P13) written under runs/p15/prospective; leakage check against the index."""
    import os
    import torch
    os.environ["P10C_OUT"] = "runs/p13/models"
    import learn_data as ld
    import p10c_train_eval as pc
    import p15_lookup as pl
    cfg, dino, recs = cf.load_config("config.toml"), ld.dino(), []
    plan = rd(PROSP / "plan.json")
    for k, r in plan["roots"].items():
        ed = [it for it in r["items"] if it["kind"] == "SCENE_TARGET_ONLY" and it.get("d_subset")]
        sq = [it for it in r["items"] if it["kind"] == "ORDERED_TEACHER"]
        rec, _ = pc._record(cfg, dino, k, r, {}, "blind", "P15_PROSPECTIVE", ed, sq)
        rec["audit"] |= {"family": r["family"], "variant": r["variant"]}
        ids = rec["audit"]["ids"]
        rec["d"] = {"pool": [ids.index(o) for o in r["parking"]], "roles": [{v: kk for kk, v in r["ids"].items()}[o] for o in r["parking"]]}
        recs.append(rec)
    torch.save({"version": "P15_PROSPECTIVE_RECORDS_V1", "roots": recs}, PROSP / "records.pt")
    idx, _ = pl.index_v2()
    import p14_sweep_index as sx
    P = sx._plans()
    fps = {pl.json_fp(P[e["root"]]) for es in idx.by_key.values() for e in es}
    leak = [k for k, r in plan["roots"].items() if pl.json_fp(r) in fps or k in {e["root"] for es in idx.by_key.values() for e in es}]
    cf.write_json(PROSP / "LEAKAGE_CHECK.json", {"roots": len(plan["roots"]), "index_overlap_by_root_or_fingerprint": leak})


def _prosp_sealed():
    """Prospective teacher outcomes; refused before runs/p15/prospective/UNSEALED (written only after the label hash commit)."""
    import p6_run as p6r
    import operational_teacher as ot
    if not (PROSP / "UNSEALED").exists():
        raise SystemExit("P15 prospective labels are sealed")
    plan, static, pol = rd(PROSP / "plan.json"), rd(PROSP / "static_admission.json")["certificates"], ot.load_policy("teacher_opneg.toml")
    res = {}
    for k in plan["roots"]:
        f = PROSP / "sealed" / f"{k}.json"
        if f.exists():
            done = rd(f)["done"]
            for q in plan["queries"]:
                if q["root"] == k and q["query"] in done:
                    res[q["query"]] = {**p6r.judge(static, pol, q, done[q["query"]]), "query": q, "root": k}
    return res


def _prosp_roots():
    import os
    import torch
    os.environ["P10C_OUT"] = "runs/p13/models"
    import abc7_train as t7
    import p10b_dataset as pd
    ds = torch.load(PROSP / "records.pt", weights_only=False)
    rs = {}
    for r, rec in zip(t7.prepare(ds), ds["roots"]):
        r["ids"], r["audit"], r["d"] = rec["audit"]["ids"], rec["audit"], rec.get("d")
        rs[r["key"]] = r
    plan = rd(PROSP / "plan.json")
    D = pd.d_labels(plan, _prosp_sealed()) if (PROSP / "UNSEALED").exists() else {k: {} for k in plan["roots"]}
    return rs, plan["roots"], D


def prospective_label(chunk_cpu=6000.0) -> None:
    """Static admission + P13_S100_V2 teacher collection of the prospective plan; split 'blind' -> outcomes only in PROSP/sealed."""
    import p13_pipeline as pp
    pd = pp._register()
    if not (PROSP / "static_admission.json").exists():
        pd.admit(PROSP)
    pd.collect(PROSP, chunk_cpu=chunk_cpu)


def prospective_label_manifest() -> None:
    files = sorted(str(f) for f in (PROSP / "sealed").glob("*.json"))
    cf.write_json(PROSP / "LABELS_MANIFEST.json", {"files": {f: sha(f) for f in files}, "n": len(files)})


def prospective_unseal() -> None:
    import p13_pipeline as pp
    if not pp._on_origin(str(PROSP / "LABELS_MANIFEST.json")):
        raise SystemExit("label manifest not on origin/main")
    man = rd(PROSP / "LABELS_MANIFEST.json")
    if any(sha(f) != h for f, h in man["files"].items()):
        raise SystemExit("sealed outcome changed after the manifest")
    (PROSP / "UNSEALED").write_text(sha(PROSP / "LABELS_MANIFEST.json") + "\n")


# ------------------------------------------------------------------ P15-5 analysis (paired, root-level; never pooled over queries)
ARMS5 = ("D", "D-norec", "D-pairless", "geometry", "uniform", "oracle")
CPU_GRID = (10.0, 25.0, 50.0, 100.0, 200.0, 400.0)


def sign_test(a, b) -> dict:
    """Exact two-sided sign test on paired binary outcomes (discordant pairs only)."""
    from math import comb
    w, l = sum(x and not y for x, y in zip(a, b)), sum(y and not x for x, y in zip(a, b))
    n = w + l
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(0, min(w, l) + 1)) / 2 ** n)
    return {"a_only": w, "b_only": l, "discordant": n, "p_two_sided": p, "status": "UNDERPOWERED" if n < 6 else "TESTED"}


def boot_diff(a, b, n=10000, seed=0) -> list:
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a, float), np.asarray(b, float)
    ix = rng.integers(0, len(a), (n, len(a)))
    return [float(x) for x in np.percentile((a[ix] - b[ix]).mean(1), [2.5, 97.5])]


def ok(v) -> bool:
    return bool(v["found"] and v["replay_success"])


def summarize(cohort, arms=ARMS5, primary=("D", "geometry")) -> dict:
    """cohort RETRO or PROSP. Strict success = plan found AND independent replay CLEAR; paired on roots completed by both arms."""
    import p14_eval as pe
    runs = {a: rd(OUT / "search" / f"{cohort}_{a}.json") for a in arms if (OUT / "search" / f"{cohort}_{a}.json").exists()}
    common = sorted(set.intersection(*[set(v) for v in runs.values()])) if runs else []
    D = (pe._roots("test") if cohort == "RETRO" else _prosp_roots())[2]          # labels only score the union metric (unsealed)
    union = {k: ({o for o, m in zip(D[k]["pool"], D[k]["mask"]) if m} if D[k].get("mask") else (set() if D[k].get("y0") == 1 else None)) for k in common}
    outside = lambda r, k: None if union[k] is None or not r["found"] else sum(s_["object"] not in union[k] for s_ in r["plan"]["steps"])  # noqa: E731
    per = {}
    for a, v in runs.items():
        rows = [v[k] for k in common]
        solved = [r for r in rows if ok(r)]
        st = collections.Counter()
        for r in rows:
            for kind in ("edges", "goals"):
                for s, c in r["log"][kind].items():
                    st[f"{kind}:{s}"] += c
        per[a] = {"roots_completed": len(v), "n": len(rows), "success": len(solved), "found": sum(r["found"] for r in rows),
                  "replay_fail": sum(r["found"] and not r["replay_success"] for r in rows),
                  "claims": dict(collections.Counter(r["claim"] for r in rows)),
                  "min_moves_proven": sum(r["min_moves_proven"] for r in rows), "min_travel_proven": sum(r["min_travel_proven"] for r in rows),
                  "truncated": sum(bool(r["truncated"]) for r in rows),
                  "calls_mean": {k: float(np.mean([r["used"][k] for r in rows])) for k in rows[0]["used"]} if rows else {},
                  "nodes_mean": float(np.mean([r["log"]["nodes"] for r in rows])) if rows else None,
                  "cpu_mean": float(np.mean([r["cpu_s"] for r in rows])) if rows else None,
                  "cpu_median": float(np.median([r["cpu_s"] for r in rows])) if rows else None,
                  "d_calls_mean": float(np.mean([r["d_calls"] for r in rows])) if rows else None,
                  "d_cpu_mean": float(np.mean([r["d_cpu_s"] for r in rows])) if rows else None,
                  "status_counts": dict(st),
                  "anytime_success": {str(t): sum(ok(r) and r["log"].get("anytime") and r["log"]["anytime"][0]["cpu_s"] <= t for r in rows)
                                      for t in CPU_GRID},
                  "moves_outside_original_blocker_union": dict(collections.Counter(str(outside(v[k], k)) for k in common if v[k]["found"])),
                  "lookup_stats": {k: int(sum(r["stats"][k] for r in rows)) for k in (rows[0]["stats"] if rows else {})}}
    pairs = {}
    ref = primary[1]
    for a in runs:
        if a == ref or ref not in runs:
            continue
        x, y = [ok(runs[a][k]) for k in common], [ok(runs[ref][k]) for k in common]
        co = [k for k in common if ok(runs[a][k]) and ok(runs[ref][k])]
        pairs[f"{a}_vs_{ref}"] = {**sign_test(x, y), "diff": (sum(x) - sum(y)) / max(1, len(common)), "ci95": boot_diff(x, y) if common else None,
                                  "co_solved": len(co),
                                  "co_solved_moves": [float(np.mean([runs[a][k]["plan"]["moves"] for k in co])) if co else None,
                                                      float(np.mean([runs[ref][k]["plan"]["moves"] for k in co])) if co else None],
                                  "co_solved_travel": [float(np.mean([runs[a][k]["plan"]["travel"] for k in co])) if co else None,
                                                       float(np.mean([runs[ref][k]["plan"]["travel"] for k in co])) if co else None],
                                  "calls_motion_mean": [float(np.mean([runs[a][k]["used"]["motion_checks"] for k in common])),
                                                        float(np.mean([runs[ref][k]["used"]["motion_checks"] for k in common]))]}
    ex = {}
    for a, v in runs.items():                      # extra moves only against PROVEN optima (any arm's min-moves proof on that root)
        e = []
        for k in common:
            pr = [runs[b][k]["plan"]["moves"] for b in runs if runs[b][k]["min_moves_proven"]]
            if pr and ok(v[k]):
                e.append(v[k]["plan"]["moves"] - min(pr))
        ex[a] = {"n_with_proven_optimum": len(e), "extra_moves": dict(collections.Counter(e))}
    fam = {a: dict(collections.Counter(f"{v[k]['family']}:{int(ok(v[k]))}" for k in common)) for a, v in runs.items()}
    out = {"cohort": cohort, "common_roots": common, "n_common": len(common), "arms": per, "paired_vs_" + ref: pairs,
           "extra_moves_vs_proven_optimum": ex, "by_family": fam,
           "per_root": {k: {a: [int(ok(runs[a][k])), runs[a][k]["plan"]["moves"] if runs[a][k]["found"] else None,
                                round(runs[a][k]["cpu_s"], 1), runs[a][k]["claim"][:20]] for a in runs} for k in common}}
    (OUT / "eval").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "eval" / f"{cohort}_SUMMARY.json", out)
    return out


# ------------------------------------------------------------------ P15-7 B closure (saved P13 frozen predictions + P13/P14 labels only)
def _fit_tb(z, y, iters=200):
    """Validation-only temperature + bias: p' = sigmoid(a z + b) by Newton on the logistic NLL (a = 1 / T)."""
    z, y, w = np.asarray(z, float), np.asarray(y, float), np.array([1.0, 0.0])
    X = np.stack([z, np.ones_like(z)], 1)
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        g, H = X.T @ (p - y), (X * (p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(2)
        w = w - np.linalg.solve(H, g)
    return float(w[0]), float(w[1])


def b_closure() -> dict:
    """(1) validation vs test admission; (2) common-candidate ranking by u with UNKNOWN candidates censored (never negative), paired vs
    K-B; (3) coverage-risk with censoring bounds; (4) original-feasible zero-move behaviour; (5) validation-only temperature + bias
    calibration and its effect on decisions under frozen rankings. No training, no new labels."""
    import math
    import p13_factors as f13
    import p14_eval as pe
    snap, P, y, st, cost = f13._p13_blind()
    A, vr = f13._by_arm(snap), rd(P14 / "calibration" / "VALIDATION_ROWS.json")
    ma, nd = rd(P14 / "calibration" / "MATCHED_ADMISSION.json")["summary"], rd(P14 / "calibration" / "NATURAL_DIAG.json")
    arms = pe.ARMS
    adm = {a: {n: {"val_coverage_mean": float(np.mean(ma[f"{a}|{n}"]["val_coverage"])), "test_admission": ma[f"{a}|{n}"]["test_admission"],
                   "executed_success_lower_bound": ma[f"{a}|{n}"]["executed_success_lower_bound"], "label_correct": ma[f"{a}|{n}"]["label_correct_root_rate"]}
               for n in ("historical", "coverage_matched", "false_admission_matched")} for a in arms}
    qsets = {k: {a: {tuple(sorted(r["query"] for r in A[a][s]["roots"][k]["rows"])) for s in A[a]} for a in arms} for k in P}
    common = sorted(k for k in P if len({x for a in arms for x in qsets[k][a]}) == 1)
    rank = {}
    for a in arms + ("fewest_moves", "random"):
        per = {}
        for k in common:
            vals = []
            for s in (A[a] if a in arms else [None]):
                rows = A[a][s]["roots"][k]["rows"] if a in arms else A["K-B"][0]["roots"][k]["rows"]
                if a == "fewest_moves":
                    order = sorted(rows, key=lambda r: (r["m"], r["query"]))
                elif a == "random":
                    order = rows
                else:
                    order = sorted(rows, key=lambda r: (-r["u"], r["query"]))
                lab = [y.get(r["query"]) for r in order]
                dec = [v for v in lab if v is not None]
                f, n = sum(dec), len(dec)
                if f == 0:
                    continue
                pos = [i for i, v in enumerate(dec) if v == 1]
                ffr = (n + 1) / (f + 1) if a == "random" else pos[0] + 1
                unk_ahead = 0 if a == "random" else lab[:lab.index(1)].count(None)
                auc = None
                if 0 < f < n and a not in ("random",):
                    sc = [(-i) for i in range(len(order))]                    # rank position as score (ties impossible)
                    pp_ = [sc[i] for i, v in enumerate(lab) if v == 1]
                    nn_ = [sc[i] for i, v in enumerate(lab) if v == 0]
                    auc = float(np.mean([p_ > q_ for p_ in pp_ for q_ in nn_]))
                vals.append({"ffr": ffr, "top1": lab[0] if a != "random" else f / n, "unk_ahead": unk_ahead, "auc": auc, "n_dec": n, "n_unk": len(lab) - n})
            if vals:
                per[k] = {x: (float(np.mean([v[x] for v in vals if v[x] is not None])) if any(v[x] is not None for v in vals) else None)
                          for x in ("ffr", "auc", "unk_ahead")} | {"top1_feasible": float(np.mean([v["top1"] == 1 if a != "random" else v["top1"] for v in vals])),
                                                                   "top1_unknown": float(np.mean([v["top1"] is None for v in vals]))}
        aucs = [v["auc"] for v in per.values() if v["auc"] is not None]
        rank[a] = {"roots_with_decided_feasible": len(per), "first_feasible_rank_mean": float(np.mean([v["ffr"] for v in per.values()])),
                   "top1_decided_feasible": float(np.mean([v["top1_feasible"] for v in per.values()])),
                   "top1_unknown_censored": float(np.mean([v["top1_unknown"] for v in per.values()])),
                   "unknown_ranked_ahead_mean": float(np.mean([v["unk_ahead"] for v in per.values()])),
                   "within_root_auroc_mean": float(np.mean(aucs)) if aucs else None, "auroc_roots": len(aucs), "_per": per}
    paired_rank = {}
    for a in arms[1:] + ("fewest_moves",):
        ks = [k for k in rank["K-B"]["_per"] if k in rank[a]["_per"] and rank["K-B"]["_per"][k]["auc"] is not None and rank[a]["_per"][k]["auc"] is not None]
        x, z = [rank["K-B"]["_per"][k]["auc"] for k in ks], [rank[a]["_per"][k]["auc"] for k in ks]
        w, l = sum(p_ > q_ + 1e-12 for p_, q_ in zip(x, z)), sum(q_ > p_ + 1e-12 for p_, q_ in zip(x, z))
        paired_rank[f"K-B_vs_{a}"] = {"roots": len(ks), "K-B_better": w, "other_better": l, "auc_diff_mean": float(np.mean(np.subtract(x, z))) if ks else None,
                                      "sign_p": sign_test([True] * w + [False] * l, [False] * w + [True] * l)["p_two_sided"],
                                      "ci95": boot_diff(x, z) if ks else None}
    cr = {}
    for a in arms:
        rows = []
        for t in pe.GRID[::5]:
            cov, lo, hi, n = 0, 0, 0, 0
            for s in A[a]:
                for k in common:
                    q = pe._choose(A[a][s]["roots"][k]["rows"], A[a][s]["tau"], t)
                    n += 1
                    if q != "ABSTAIN":
                        cov += 1
                        lo += y.get(q) == 0
                        hi += y.get(q) in (0, None)
            rows.append({"t": float(t), "coverage": cov / n, "risk_lower": lo / max(cov, 1), "risk_upper_unknown_as_failure": hi / max(cov, 1)})
        cr[a] = rows
    cal = {}
    empty = {k: f"seq:{k}:empty" for k in P}
    for a in arms:
        res = []
        for s in sorted(A[a]):
            tau, thr = A[a][s]["tau"], A[a][s]["threshold"]
            v = [r for rs in vr[f"P13-{a}-s{s}"].values() for r in rs if r["y"] in (0, 1)]
            sl, b = _fit_tb([r["u"] / tau for r in v], [r["y"] for r in v])
            te = [(r["u"] / tau, y[r["query"]], r["query"]) for k in common for r in A[a][s]["roots"][k]["rows"] if r["query"] in y]
            p0 = np.array([1 / (1 + math.exp(-z)) for z, _, _ in te])
            p1 = np.array([1 / (1 + math.exp(-(sl * z + b))) for z, _, _ in te])
            yy = np.array([t_ for _, t_, _ in te])
            em = np.array([q == empty[q.split(":")[1]] for _, _, q in te])
            nll = lambda p, m: float(np.mean(-np.log(np.clip(np.where(yy[m] == 1, p[m], 1 - p[m]), 1e-12, 1))))  # noqa: E731
            ch0 = {k: pe._choose(A[a][s]["roots"][k]["rows"], tau, thr) for k in common}
            rec_rows = {k: [dict(r, u=tau * (sl * r["u"] / tau + b)) for r in A[a][s]["roots"][k]["rows"]] for k in common}
            ch1 = {k: pe._choose(rec_rows[k], tau, thr) for k in common}
            vcov0 = np.mean([pe._choose(rs, tau, thr) != "ABSTAIN" for rs in vr[f"P13-{a}-s{s}"].values()])
            vr1 = {k: [dict(r, u=tau * (sl * r["u"] / tau + b)) for r in rs] for k, rs in vr[f"P13-{a}-s{s}"].items()}
            tm = min(pe.GRID, key=lambda t: (abs(np.mean([pe._choose(rs, tau, t) != "ABSTAIN" for rs in vr1.values()]) - vcov0), t))
            ch2 = {k: pe._choose(rec_rows[k], tau, tm) for k in common}
            lc = lambda ch: float(np.mean([y.get(q) == 1 for q in ch.values()]))  # noqa: E731
            res.append({"seed": s, "slope": sl, "bias": b, "temperature": 1 / sl if sl else None, "val_rows": len(v),
                        "test_candidates": len(te), "brier": [float(np.mean((p0 - yy) ** 2)), float(np.mean((p1 - yy) ** 2))],
                        "nll": [nll(p0, np.ones_like(em)), nll(p1, np.ones_like(em))],
                        "empty_feasible_recall": [float((p0[em & (yy == 1)] >= 0.5).mean()) if (em & (yy == 1)).any() else None,
                                                  float((p1[em & (yy == 1)] >= 0.5).mean()) if (em & (yy == 1)).any() else None],
                        "empty_nll": [nll(p0, em), nll(p1, em)] if em.any() else None,
                        "fixed_threshold_choice_changes": sum(ch0[k] != ch1[k] for k in common),
                        "label_correct_fixed": [lc(ch0), lc(ch1)], "admission_fixed": [float(np.mean([q != "ABSTAIN" for q in ch0.values()])),
                                                                                         float(np.mean([q != "ABSTAIN" for q in ch1.values()]))],
                        "val_coverage_matched_threshold": float(tm), "coverage_matched_choice_changes": sum(ch0[k] != ch2[k] for k in common)})
        cal[a] = res
    zero = {st_: {a: {x: nd[st_][a][x] for x in ("feasible_recall", "infeasible_recall", "brier", "nll", "auroc", "TOP1_empty_on_feasible_originals", "reliability")}
                  for a in arms if a in nd[st_]} | {"N": nd[st_]["N"], "feasible": nd[st_]["feasible"]} for st_ in ("natural", "blind")}
    for a in rank.values():
        a.pop("_per")
    out = {"admission_validation_vs_test": adm, "common_roots": len(common), "roots_total": len(P), "ranking_censored": rank,
           "ranking_paired_vs_K-B": paired_rank, "coverage_risk": cr, "zero_move_original": zero, "calibration_validation_only": cal,
           "lean_inference": "B-F3pm / B-F5spm / B-F5qpm are post-hoc inference masks of trained checkpoints, NOT matched retrains; "
                             "generic-only F5 remains NOT experimentally established",
           "note": "saved P13 frozen predictions and P13 teacher labels (VERIFIED / CERTIFIED); UNKNOWN candidates censored"}
    (OUT / "B").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "B" / "B_CLOSURE.json", out)
    return out


# ------------------------------------------------------------------ P15-6 lookup ablations (geometry guide, first 12 retro roots, same budget / cap)
ABL = {"local": dict(lookup="local"), "global": dict(lookup="global"), "threshold": dict(lookup="combined+thr"),
       "modal_top1": dict(modal_only=True), "representative": dict(lookup="combined+rep"), "online_only": dict(use_index=False)}


def ablation_summary() -> dict:
    """Each variant vs combined (RETRO_geometry on the same roots): success, calls, lookup hits / rare modes, online fallbacks, CPU,
    co-solved moves / travel, paired sign test; index build CPU amortized against the online-only per-root CPU difference."""
    roots = rd(OUT / "P15_PREREG.json")["retrospective"]["roster"][:12]
    base = rd(OUT / "search" / "RETRO_geometry.json")
    runs = {"combined": base} | {a: rd(OUT / "search" / f"ABL_{a}.json") for a in ABL if (OUT / "search" / f"ABL_{a}.json").exists()}
    ks = [k for k in roots if all(k in v for v in runs.values())]
    per = {}
    for a, v in runs.items():
        rows = [v[k] for k in ks]
        co = [k for k in ks if ok(v[k]) and ok(base[k])]
        x, y = [ok(v[k]) for k in ks], [ok(base[k]) for k in ks]
        per[a] = {"n": len(ks), "success": sum(x), "calls_mean": {c: float(np.mean([r["used"][c] for r in rows])) for c in rows[0]["used"]} if rows else {},
                  "lookup": {c: int(sum(r["stats"][c] for r in rows)) for c in rows[0]["stats"]} if rows else {},
                  "online_fallback_edges": int(sum(r["used"]["online_moves"] for r in rows)), "cpu_mean": float(np.mean([r["cpu_s"] for r in rows])) if rows else None,
                  "capped": sum(bool(r["truncated"]) for r in rows), "min_moves_proven": sum(r["min_moves_proven"] for r in rows),
                  "co_solved_with_combined": len(co),
                  "co_moves": [float(np.mean([v[k]["plan"]["moves"] for k in co])), float(np.mean([base[k]["plan"]["moves"] for k in co]))] if co else None,
                  "co_travel": [float(np.mean([v[k]["plan"]["travel"] for k in co])), float(np.mean([base[k]["plan"]["travel"] for k in co]))] if co else None,
                  "vs_combined": sign_test(x, y)}
    build = rd(P14 / "guards" / "p14_d1_build_index.json")["cpu_s"] if (P14 / "guards" / "p14_d1_build_index.json").exists() else None
    if "online_only" in per and build:
        d = per["online_only"]["cpu_mean"] - per["combined"]["cpu_mean"]
        per["amortization"] = {"index_build_cpu_s": build, "per_root_cpu_saving": d, "break_even_roots": build / d if d > 0 else None}
    out = {"roots": ks, "variants": per, "note": "resource / coverage ablations: threshold, modal_top1 and representative change the number of "
                                                 "available edges; downstream differences are not attributed to similarity scoring alone"}
    cf.write_json(OUT / "eval" / "ABLATION_SUMMARY.json", out)
    return out


# ------------------------------------------------------------------ regression and manifests
def regression() -> None:
    """Historical suite + P12-P15 tests + six replays (same harness as P13 / P14), outputs under runs/p15/regression."""
    import subprocess
    d = OUT / "regression"
    d.mkdir(parents=True, exist_ok=True)
    (d / "countcalls.py").write_text(Path("runs/p8/audit/regression/countcalls.py").read_text())
    subprocess.run([sys.executable, "run.py", "g4c-lf-p4-regression", "--out", str(d), "--cpu", "2400", "--wall", "6000"], capture_output=True, text=True)


SOURCES = ("p15_bench.py", "p15_retarget.py", "p15_lookup.py", "p15_search.py", "p15_viz.py", "tests/test_p15.py")


def code_manifest() -> None:
    cf.write_json(OUT / "P15_CODE_MANIFEST.json", {"sources": {p: sha(p) for p in SOURCES}, "search_version": "P15_TRISTATE_SEARCH_V1+AMENDMENT_1",
                                                   "d_checkpoints": {str(p): sha(p) for p in sorted(Path("runs/p13/models/ckpt").glob("P13-D*-s*.pt"))},
                                                   "index_p14": sha(P14 / "lookup" / "INDEX.json")})


def manifest(status) -> None:
    import hashlib
    import subprocess
    reg = rd(OUT / "regression" / "regression_result.json")
    arts = sorted(str(p) for sub in ("audit", "retarget", "lookup", "search", "eval", "B", "prospective") for p in (OUT / sub).glob("*.json"))
    out = {"stage": "P15", "base": "cbd5d9b1509fcfdad733545b7d0e9cba4181ee4f", "status": status,
           "gates": {g.stem: {k: v for k, v in rd(g).items() if k != "spend"} for g in sorted(OUT.glob("P15_*_GATE.json"))},
           "regression": {"tests": reg.get("tests"), "replays": reg.get("replays")}, "spend_cpu_s": {c: round(spent(c), 1) for c in CAPS}, "caps": CAPS,
           "prereg": sha(OUT / "P15_PREREG.json"), "amendment_1": sha(OUT / "P15_PREREG_AMENDMENT_1.json"),
           "sources": {p: sha(p) for p in SOURCES}, "artifacts": {p: sha(p) for p in arts},
           "figures": {str(p): sha(p) for p in sorted(Path("visualizations/p15").glob("*")) if p.suffix not in (".sha256", ".json")},
           "historical_unchanged": {p: (sha(p), h) for p, h in rd(OUT / "audit" / "P15_0_AUDIT.json")["historical_hashes"].items()},
           "head_at_manifest": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
    out["historical_mismatches"] = [p for p, (a, b) in out["historical_unchanged"].items() if a != b]
    out["hash"] = "sha256:" + hashlib.sha256(json.dumps(out, sort_keys=True).encode()).hexdigest()
    cf.write_json(OUT / "P15_FINAL_MANIFEST.json", out)
