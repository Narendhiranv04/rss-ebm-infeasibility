"""P8b: repair-selection diagnostics (offline, saved teacher records only for ASSESSMENT, never for ranking).

  funnel      per root: VERIFIED candidate exists -> scored -> admitted (calibrated p >= thr) -> chosen (lexicographic:
              fewest moves, predicted J, analytic displacement, energy, id) -> chosen's recorded relocation witness and target
              outcome (unit final.json of the teacher) -> GOAL_AND_SAFE_RETREAT; plus first VERIFIED rank, false-feasible empty
              choices (certified vs operational), restricted-catalogue minimality
  thresholds  frozen family {0.5, 0.7, 0.85, 0.95}; validation-only lexicographic choice: (1) fewer false-feasible empty
              repairs on known-negative original tasks, (2) more successful offline choices (abstain = failure), (3) fewer moves
  mixing      G4B kernels on a learned sampling energy vs the exact finite Boltzmann distribution: 4 seeded chains of 4,000 and
              16,000 steps; TV, energy-trace ESS, top-5 discovery, CPU and model calls; deterministic bounded-pool fallback =
              score every canonical pool member once and sort by the same lexicographic rule (calls charged)
  G+M         geometry + full-motion baseline ranking: deterministic pose screens (sequential mover grasps, target grasp and
              place) and the LINK0 sector claim at the fixed start, then fewest moves and analytic displacement; its relocations
              are planned by ordered.primary_v2 under the same budget as every arm (no teacher verdict is ever read)
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import g4b
import p6_scenes as p6
import p7_eval as ev
import sampler as sp
import sector_cert
import zlayout_r4 as z4
from scene import scene_from_dict

THRESHOLDS = (0.5, 0.7, 0.85, 0.95)


def unit_outcome(collect: Path, rec) -> dict:
    """Recorded relocation witness / target result of the teacher's attempts for one query (assessment only)."""
    out = {"relocation_witness": None, "target": None}
    for a in rec.get("attempts", []):
        qd = collect / "q" / rec["query"]["query"].replace(":", "__").replace(">", "-") / f"seed{a['seed']}"
        led = qd / "ledger.json"
        if not led.exists():
            continue
        st = json.loads(led.read_text())["units"].get(a["unit"], {})
        f = qd / st.get("dir", "_") / "final.json"
        if f.exists():
            fin = json.loads(f.read_text())
            out["relocation_witness"] = out["relocation_witness"] or fin.get("relocation_attempt_status") == "RELOCATED"
            out["target"] = (fin.get("target") or {}).get("status") or out["target"]
    return out


def funnel(evals, records, collect, tau, thr) -> list:
    rows = []
    for e in evals:
        if e["root"].endswith("_ref"):
            continue
        p = {c["query"]: 1 / (1 + math.exp(-c["u"] / tau)) for c in e["cands"]}
        order = sorted(e["cands"], key=lambda c: (c["m"], c["J_pred"], c["C_d"], c["energy"], c["query"]))
        adm = [c for c in order if p[c["query"]] >= thr]
        ver = [c for c in e["cands"] if c["y"] == 1]
        rank = sorted(e["cands"], key=lambda c: -p[c["query"]])
        ch = adm[0] if adm else None
        y0 = next((c["y"] for c in e["cands"] if c["m"] == 0), None)
        rec = records.get(ch["query"]) if ch else None
        uo = unit_outcome(collect, rec) if rec else {}
        rows.append({"root": e["root"], "verified_candidate_exists": bool(ver), "scored": len(e["cands"]), "admitted": len(adm),
                     "chosen": ch and ch["query"], "chosen_moves": ch and ch["m"], "chosen_label": rec and rec["teacher_status"],
                     "chosen_relocation_witness": uo.get("relocation_witness"), "chosen_target": uo.get("target"),
                     "success": bool(rec and rec["teacher_status"] == "VERIFIED_FEASIBLE"), "abstain": ch is None,
                     "false_feasible_empty": bool(ch and ch["m"] == 0 and y0 == 0),
                     "empty_negative_source": (records.get(ch["query"]) or {}).get("label_source") if ch and ch["m"] == 0 and y0 == 0 else None,
                     "min_verified_moves": min((c["m"] for c in ver), default=None),
                     "first_verified_rank": next((i for i, c in enumerate(rank) if c["y"] == 1), None),
                     "shorter_infeasible_preempted": bool(ch and ch["y"] == 0 and any(c["y"] == 1 and c["m"] > ch["m"] for c in adm))})
    return rows


def choose_threshold(val_evals, records, collect, tau) -> dict:
    table = {}
    for thr in THRESHOLDS:
        f = funnel(val_evals, records, collect, tau, thr)
        table[str(thr)] = {"false_feasible_empty": sum(x["false_feasible_empty"] for x in f), "success": sum(x["success"] for x in f),
                           "abstain": sum(x["abstain"] for x in f), "roots": len(f),
                           "moves_on_success": float(np.mean([x["chosen_moves"] for x in f if x["success"]])) if any(x["success"] for x in f) else None}
    best = min(THRESHOLDS, key=lambda t: (table[str(t)]["false_feasible_empty"], -table[str(t)]["success"],
                                          table[str(t)]["moves_on_success"] if table[str(t)]["moves_on_success"] is not None else 9, t))
    return {"table": table, "chosen": best}


def pool_energy(m, r, rec):
    """(dests, energy(state)) of a root's canonical pool; energy = learned Ebar + displacement + predicted J (calls counted)."""
    ids = rec["audit"]["ids"]
    cell = lambda xy: int(round((xy[1] - 0.04) / 0.08)) * 8 + int(round((xy[0] - 0.04) / 0.08))  # noqa: E731
    dests = {}
    for c in r["cands"]:
        for i, d in c["seq"]:
            dests.setdefault(ids[i], set()).add(cell(d))
    dests = {o: sorted(v) for o, v in dests.items()}
    cat = [{"kind": e["kind"], "edit": {ids[i]: cell(p) for i, p in e["poses"].items()}, "eligible": e["eligible"]} for e in rec["edits"]]
    cache, calls = {}, [0]

    @torch.no_grad()
    def energy(state):
        if state not in cache:
            calls[0] += 1
            seq = [(ids.index(o), ((d % 8 + 0.5) * 0.08, (d // 8 + 0.5) * 0.08)) for o, d in state]
            m._memo = {}
            o = m.seq(r["R"], seq) if m.kind in ("A", "S", "SF") else m.joint(r["R"], seq, list(z4.build(r["key"], list(state), cat)["query"]), r["edits"])
            m._memo = None
            cache[state] = float(o["E"]) + ev._cd(r, seq) + float(o["J"])
        return cache[state]
    return dests, energy, calls


def mixing(m, r, rec, steps=(4000, 16000), chains=4) -> dict:
    dests, energy, calls = pool_energy(m, r, rec)
    dom, states = sp.Domain.make(dests, 4), g4b.reference_states(dests, 4)
    c0 = time.process_time()
    cert, pi, _ = g4b.certify(dom, energy, 1.0, states)
    exact_cpu, exact_calls = time.process_time() - c0, calls[0]
    pi = np.array(pi)
    top5 = set(np.argsort(-pi)[:5].tolist())
    idx = {s: i for i, s in enumerate(states)}
    out = {"states": len(states), "detailed_balance_pass": cert["pass"], "max_error": cert["max_error"], "exact_cpu_s": round(exact_cpu, 2),
           "exact_model_calls": exact_calls, "deterministic_fallback": {"model_calls": len(states), "cpu_s": round(exact_cpu, 2),
                                                                        "top1_state": [list(a) for a in states[int(np.argmax(pi))]]}}
    for n in steps:
        res = []
        for ch in range(chains):
            c1 = time.process_time()
            chain, _ = sp.run_chain(dom, (), energy, 1.0, n, 1000 + ch)
            freq = np.bincount([idx[s] for s in chain], minlength=len(states)) / len(chain)
            e = np.array([energy(s) for s in chain])
            ac = [np.corrcoef(e[:-k], e[k:])[0, 1] if e.std() > 0 else 1.0 for k in range(1, min(200, n // 4))]
            tau_int = 1 + 2 * sum(a for a in ac if a > 0.05)
            res.append({"tv": float(0.5 * np.abs(freq - pi).sum()), "ess": float(n / tau_int), "top5_found": len(top5 & set(np.nonzero(freq)[0].tolist())),
                        "cpu_s": round(time.process_time() - c1, 2)})
        out[f"chain_{n}"] = {"tv_mean": float(np.mean([x["tv"] for x in res])), "ess_mean": float(np.mean([x["ess"] for x in res])),
                             "top5_found_mean": float(np.mean([x["top5_found"] for x in res])), "cpu_s_mean": float(np.mean([x["cpu_s"] for x in res])),
                             "chains": res}
    return out


def gm_rank(cfg, robot, plan, root_key, k=5) -> list:
    """Geometry + full-motion baseline ranking of a root's canonical catalogue (no learned model, no teacher verdict)."""
    scr, r = p6.Screens(cfg, robot), plan["roots"][root_key]
    s0 = scene_from_dict(r["scene"])
    rows = []
    for p in r["queries"]["pool"]:
        seq, s, ok = [tuple(a) for a in p["sequence"]], s0, True
        for t, (o, d) in enumerate(seq):
            if not scr.pick(s, o) or (t == 0 and sector_cert.produce(robot, cfg, s, o, "FIRST_ACTION_ORDER", seq, d)["claim"] == "CERTIFIED_INFEASIBLE"):
                ok = False
                break
            s = p6.moved(s, {o: d})
        if ok:
            ok = bool(scr.pick(s, "t0")) and scr.place(s, "t0", r["goal_cell"]) > 0
            if ok and not seq:
                ok = sector_cert.produce(robot, cfg, s0, "t0", "ORIGINAL_TARGET", [], r["goal_cell"])["claim"] != "CERTIFIED_INFEASIBLE"
        disp = sum(np.hypot(*(np.array(s0.lattice.center(d)) - np.array(s0.get(o).xy_m))) for o, d in seq)
        rows.append((not ok, len(seq), disp, z4.sig(seq), p["sequence"]))
    return [x[4] for x in sorted(rows)[:k]]
