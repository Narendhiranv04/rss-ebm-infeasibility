"""2D-ABC-OPNEG-v1: operational-negative teacher policy (G4C-LF-P4). Pure logic, no planner.

Four teacher statuses for NEW records (historical labels are never touched):
  VERIFIED_FEASIBLE    some seed produced an independently re-checked complete witness   (WITNESS)
  CERTIFIED_INFEASIBLE an independent checker accepted a sound certificate of the scope  (CERTIFICATE)
  BUDGET_INFEASIBLE    every declared seed ran cleanly to an allowed algorithmic terminal (PLANNER_BUDGET)
                       -- operational and policy-relative: Y_B = 0 never implies Y* = 0
  UNKNOWN              anything else: censored / crashed / missing seed / numeric / invalid (UNRESOLVED)
A witness beats budget failures; a witness against a sound certificate is FATAL. Effects
and joint z pools are computed only from records of ONE policy hash; any side that is a
budget negative makes the effect operational, never a proof of necessity.
"""
from __future__ import annotations

import itertools
import tomllib
from pathlib import Path

import config as cf
import data
import g4c_pilot as gp

VERIFIED, CERTIFIED, BUDGET, UNKNOWN = "VERIFIED_FEASIBLE", "CERTIFIED_INFEASIBLE", "BUDGET_INFEASIBLE", "UNKNOWN"
SOURCE = {VERIFIED: "WITNESS", CERTIFIED: "CERTIFICATE", BUDGET: "PLANNER_BUDGET", UNKNOWN: "UNRESOLVED"}
BINARY = {VERIFIED: 1, CERTIFIED: 0, BUDGET: 0, UNKNOWN: None}
KEYS = {"policy": {"policy_id": str, "seeds": list, "seed_key": str, "planner_max_nodes": int, "max_unresolved_connectors": int,
                   "per_seed_cpu_s": float, "per_seed_guard_margin_s": float, "per_seed_wall_s": float,
                   "per_query_cpu_cancel_s": float, "per_query_cpu_hard_s": float, "per_query_wall_s": float,
                   "target_success": str, "short_circuit": list, "budget_negative_weight": float, "allowed_terminals": list},
        "pilot": {"max_queries": int, "max_seed_attempts": int, "max_lineages": int, "aggregate_cpu_s": float,
                  "aggregate_wall_s": float, "static_admission_cpu_s": float, "orchestration_reserve_s": float, "workers": int}}


class Fatal(Exception):
    """A verified witness contradicts a sound certificate: stop the stage, keep both records."""


def load_policy(path=Path("teacher_opneg.toml")) -> dict:
    doc = tomllib.loads(Path(path).read_text())
    if set(doc) != set(KEYS):
        raise ValueError(f"policy sections {sorted(doc)} != {sorted(KEYS)}")
    for sec, spec in KEYS.items():
        if set(doc[sec]) != set(spec):
            raise ValueError(f"[{sec}] keys differ: unknown {sorted(set(doc[sec]) - set(spec))}, missing {sorted(set(spec) - set(doc[sec]))}")
        for k, t in spec.items():
            if not isinstance(doc[sec][k], t) or isinstance(doc[sec][k], bool):
                raise ValueError(f"[{sec}].{k} must be {t.__name__}")
    p = doc["policy"]
    if len(p["seeds"]) != 3 or len(set(p["seeds"])) != 3 or p["per_seed_cpu_s"] * 3 > p["per_query_cpu_cancel_s"] or \
            p["per_query_cpu_cancel_s"] > p["per_query_cpu_hard_s"] or set(p["short_circuit"]) != {"WITNESS", "CERTIFICATE"}:
        raise ValueError("seed schedule or per-query guards inconsistent")
    return {**doc, "policy_hash": cf.stable_hash(doc)}


def config_mismatch(pol: dict, cfg: dict) -> list:
    p = pol["policy"]
    why = [] if cfg["planner2d"]["max_nodes"] == p["planner_max_nodes"] else ["planner node budget differs from the policy"]
    return why + ([] if cfg["skill2d"]["max_plan_failures"] == p["max_unresolved_connectors"] else ["connector failure stop differs"])


# ------------------------------------------------------------------ one seed attempt -> terminal
def seed_terminal(e: dict, st: dict, allowed) -> dict:
    """Classify one seeded full-query attempt from revalidated evidence e and its ledger entry st."""
    lab = gp.adjudicate(e)
    out = {"stop": st.get("stop"), "cpu_s": st.get("cpu_s"), "wall_s": st.get("wall_s"), "label": lab["label"],
           "reason": lab.get("reason"), "calls": e.get("calls"), "clean": False, "terminal": None}
    if st.get("stop") != "COMPLETED" or st.get("returncode") not in (0, None) or "final" not in (e.get("checkpoints") or []):
        out["terminal"] = f"NOT_CLEAN:{st.get('stop')}"
        return out
    if lab["label"] in (data.POSITIVE, data.NEGATIVE):
        out.update(clean=True, terminal="WITNESS" if lab["label"] == data.POSITIVE else "CERTIFICATE")
        return out
    r, t = e.get("relocation") or {}, e.get("target") or {}
    if e.get("sequence") and not r:
        term = "NO_RELOCATION_RECORD"
    elif r and not r.get("complete_witness"):
        term = (r.get("normalized_reasons") or ["NONE"])[-1] if r.get("records_recertified") else "PARTIAL_RECORD_NOT_RECERTIFIED"
    else:
        term = t.get("normalized_reason") or "NO_TARGET_TERMINAL"
    if term.startswith("WITNESS"):          # a path the independent recheck did not confirm: never a positive
        term = "UNCONFIRMED_" + term
    out.update(terminal=term, clean=term in allowed)
    return out


# ------------------------------------------------------------------ one query -> teacher status
def adjudicate_query(static_cert: dict | None, seeds: list, policy: dict) -> dict:
    """static_cert: an accepted static certificate {kind, scope} or None; seeds: seed_terminal()
    records in schedule order (only those actually run)."""
    n = len(policy["policy"]["seeds"])
    pos = [s for s in seeds if s["terminal"] == "WITNESS"]
    certs = ([static_cert] if static_cert else []) + [s for s in seeds if s["terminal"] == "CERTIFICATE"]
    if pos and certs:
        raise Fatal(f"witness on seed(s) {[s.get('seed') for s in pos]} contradicts certificate {certs[0]}")
    if pos:
        st, why, scope = VERIFIED, f"complete witness on seed {pos[0].get('seed')}", "EXISTENTIAL_WITNESS"
    elif certs:
        c = certs[0]
        st, why, scope = CERTIFIED, f"certificate {c.get('kind') or c.get('reason')}", c.get("scope") or "ORDER/ASSIGNMENT"
    elif len(seeds) == n and all(s["clean"] for s in seeds):
        st, why = BUDGET, "all declared seeds exhausted: " + ", ".join(sorted({s["terminal"] for s in seeds}))
        scope = f"OPERATIONAL_ONLY:{policy['policy_hash']}"
    else:
        bad = [s for s in seeds if not s["clean"]]
        st, scope = UNKNOWN, None
        why = f"{n - len(seeds)} declared seed(s) not run" if len(seeds) < n else f"unclean seed terminal {bad[0]['terminal']}"
    w = policy["policy"]["budget_negative_weight"] if st == BUDGET else 1.0
    return {"teacher_status": st, "label_source": SOURCE[st], "binary_training_label": BINARY[st], "proof_scope": scope,
            "is_budget_relative": st == BUDGET, "reason": why, "decided_mask": BINARY[st] is not None,
            "training_weight": w if BINARY[st] is not None else 0.0, "policy_id": policy["policy"]["policy_id"],
            "policy_hash": policy["policy_hash"], "seeds_run": len(seeds), "seeds_declared": n}


# ------------------------------------------------------------------ projections, effects, pools (same policy only)
def projections(records: dict, policy_hash: str) -> dict:
    mixed = sorted(q for q, r in records.items() if r["policy_hash"] != policy_hash)
    if mixed:
        raise ValueError(f"mixed teacher policies: {mixed}")
    admitted = {q: r for q, r in records.items() if r["binary_training_label"] is not None}
    return {"raw": {q: r["teacher_status"] for q, r in records.items()},
            "admitted": {q: {"y": r["binary_training_label"], "source": r["label_source"], "weight": r["training_weight"]}
                         for q, r in admitted.items()},
            "excluded": {q: r["reason"] for q, r in records.items() if r["binary_training_label"] is None}}


def _y(records, q):
    r = records.get(q)
    return (r["binary_training_label"], r["label_source"]) if r else (None, None)


def _effect(records, a, b, kind, **info):
    (ya, sa), (yb, sb) = _y(records, a), _y(records, b)
    if ya is None or yb is None:
        return {"kind": kind, "reference": b, "alternative": a, "decided": False, **info}
    return {"kind": kind, "reference": b, "alternative": a, "decided": True, "delta": ya - yb,
            "operational": "PLANNER_BUDGET" in (sa, sb), **info}


def effects(records: dict, roots: dict) -> dict:
    """Scene effects vs the root's empty repair, omissions (Delta^O = Y(r) - Y(r minus A)),
    nonadditivity only with all four sides decided, and same-endpoint order swaps."""
    out = {"scene": [], "omission": [], "kappa": [], "swap": []}
    for k in roots:
        empty = f"seq:{k}:empty"
        scenes = {q: r for q, r in records.items() if q.startswith(f"scene:{k}:")}
        for q, r in scenes.items():
            out["scene"].append(_effect(records, q, empty, "scene", edit=r["query"]["edit"]))
        for q, r in scenes.items():
            ed = r["query"]["edit"]
            if len(ed) == 2:
                singles = [f"scene:{k}:{gp._sig([(o, c)])}" for o, c in sorted(ed.items())]
                d = [_effect(records, s, empty, "scene") for s in [q] + singles]
                if all(x["decided"] for x in d):
                    out["kappa"].append({"kind": "scene", "joint": q, "kappa": d[0]["delta"] - d[1]["delta"] - d[2]["delta"],
                                         "deltas": [x["delta"] for x in d], "operational": any(x["operational"] for x in d)})
        seqs = {q: r["query"]["sequence"] for q, r in records.items() if q.startswith(f"seq:{k}:")}
        for q, s in seqs.items():
            for n in (1, 2) if len(s) >= 2 else ((1,) if s else ()):
                for drop in itertools.combinations(range(len(s)), n):
                    rem = f"seq:{k}:{gp._sig([a for i, a in enumerate(s) if i not in drop])}"
                    if rem in records:
                        out["omission"].append(_effect(records, q, rem, "omission", omitted=list(drop)))
            if len(s) == 2:
                d = [_effect(records, q, f"seq:{k}:{gp._sig([a for i, a in enumerate(s) if i not in drop])}", "omission")
                     for drop in ((0, 1), (0,), (1,))]
                if all(x["decided"] for x in d):
                    out["kappa"].append({"kind": "omission", "sequence": q, "kappa": d[0]["delta"] - d[1]["delta"] - d[2]["delta"],
                                         "deltas": [x["delta"] for x in d], "operational": any(x["operational"] for x in d)})
        for a, b in itertools.combinations(sorted(seqs), 2):
            if len(seqs[a]) >= 2 and sorted(map(tuple, seqs[a])) == sorted(map(tuple, seqs[b])):
                (ya, sa), (yb, sb) = _y(records, a), _y(records, b)
                out["swap"].append({"orders": [a, b], "labels": [ya, yb], "sources": [sa, sb],
                                    "decided_contrast": ya is not None and yb is not None and ya != yb,
                                    "operational": "PLANNER_BUDGET" in (sa, sb)})
    return out


def pool(root_key: str, root: dict, sequence, records: dict) -> dict:
    """Exact B/C z-pool for one declared repair: every ACTIVE slot of its frozen bc_layout must be
    a decided record of this policy (the slot -> query map mirrors g4c_pilot.bc_pool)."""
    seq = [tuple(a) for a in sequence]
    entry = next((p for p in root["queries"]["pool"] if [tuple(a) for a in p["sequence"]] == seq), None)
    if entry is None:
        return {"complete": False, "reason": "sequence not in the frozen root pool"}
    cat, slots = {q["kind"]: q for q in root["queries"]["scene_queries"]}, []
    for slot, act in zip(entry["bc_layout"]["slots"], entry["bc_layout"]["active"]):
        if not act:
            continue
        if slot == "z0":
            q = f"seq:{root_key}:empty"
        elif slot == "zR":
            q = f"seq:{root_key}:{gp._sig(seq)}"
        elif slot.startswith("zS["):
            c = cat.get(slot[3:-1])
            q = c and f"scene:{root_key}:{gp._sig(sorted(c['edit'].items()))}"
        else:
            t = int(slot[3:-1])
            q = f"seq:{root_key}:{gp._sig([a for i, a in enumerate(seq) if i != t])}"
        y, s = _y(records, q) if q else (None, None)
        slots.append({"slot": slot, "query": q, "y": y, "source": s})
    miss = [x for x in slots if x["y"] is None]
    return {"complete": not miss, "active": len(slots), "decided": len(slots) - len(miss), "slots": slots,
            "operational": any(x["source"] == "PLANNER_BUDGET" for x in slots), "missing": [x["slot"] for x in miss]}


def loss_masks(records: dict) -> dict:
    """Loss-facing view: binary target / mask / weight per query, and a cost label only where
    a repair cost exists (J(empty) = 0; ordered repair: the first-seed first relocation witness;
    fixed-robot scene edits are diagnostics and never carry a cost)."""
    out = {}
    for q, r in records.items():
        fam = "fixed_robot_scene_edit" if q.startswith("scene:") else "original_task" if q.endswith(":empty") else "ordered_repair"
        j = 0.0 if fam == "original_task" else r.get("J_first") if fam == "ordered_repair" else None
        out[q] = {"family": fam, "y": r["binary_training_label"], "decided_mask": r["decided_mask"], "weight": r["training_weight"],
                  "label_source": r["label_source"], "budget_relative": r["is_budget_relative"], "cost_J": j, "cost_mask": j is not None}
    return out
