"""G4A label tables, coverage and readiness (G4A instruction §7, §9).

First-pass labels come only from the frozen first-pass teacher audit. Diagnostic-
enriched labels add complete witnesses found by the seed/order panels and by the
primary episode (positives only; a diagnostic never adds a negative and never
replaces an earlier attempt). Every count is over distinct roots or distinct
physical assignments; repeated seeds or orders never inflate them.
"""
from __future__ import annotations

from collections import Counter

import data
import proposals as pr
import skill as sk

FIRST = "first_pass"


def label_table(col: dict) -> list:
    """One row per (query, assignment) with first-pass and enriched labels."""
    extra = {}
    for s in col["seed_panel"]:
        hits = [k for k, r in zip(s["keys"], s["runs"]) if r["status"] == sk.VERIFIED]
        extra.setdefault((s["query_id"], pr.signature(_c(s["cells"]))), []).extend(f"seed_panel:{k}" for k in hits)
    for o in col["order_panel"]:
        for row in o["orders_found"]:
            hits = [f"order_panel:{row['order']}" for t in row["targets"] if t["status"] == sk.VERIFIED]
            extra.setdefault((o["query_id"], pr.signature(_c(o["cells"]))), []).extend(hits)
    rows = []
    for qid, r in sorted(col["main"].items()):
        p = r["primary"]
        if p["outcome"] == "SUCCESS":
            extra.setdefault((qid, pr.signature(p["commit"]["cells"])), []).append("primary_episode")
        for a in r["audit"]:
            sig = pr.signature(a["cells"])
            first = a["aggregate"]["label"]
            more = extra.get((qid, sig), [])
            ev = a["evidence"] + [{"kind": "complete_witness", "source": s} for s in more]
            rows.append({"query_id": qid, "root_id": r["q"]["root_id"], "stratum": r["q"]["stratum"], "label": a["label"],
                         "category": next((c.get("category", "") for c in r["q"]["audit_pool"]
                                           if pr.signature(_c(c["cells"])) == sig), ""),
                         "signature": [list(x) for x in sig], "moved": a["moved"], "static_status": a["static"]["status"],
                         FIRST: first, "first_pass_sources": a["aggregate"],
                         "diagnostic_enriched": data.aggregate(ev)["label"], "extra_witnesses": more,
                         "certificates": [{k: c[k] for k in ("kind", "checker_accepted", "checker_reasons")}
                                          for c in a["certificates"]], "unknown_reason": a["unknown_reason"]})
    return rows


def _c(cells: dict) -> dict:
    return {o: d if isinstance(d, int) else tuple(d) for o, d in cells.items()}


def _roots(rows, pred) -> set:
    return {r["root_id"] for r in rows if pred(r)}


def coverage(plan: dict, col: dict, rows: list, protocol_ok: dict) -> dict:
    T = plan["targets"]
    sv = [r for r in rows if r["static_status"] == "VALID"]
    out = {"roots_planned": len(plan["queries"]), "roots_completed": len(col["main"]),
           "construction_attempts": len(plan["ledger"]), "assignments_audited": len(rows), "static_valid_assignments": len(sv)}
    for tag in (FIRST, "diagnostic_enriched"):
        pos = [r for r in rows if r[tag] == data.POSITIVE]
        sigs = {}
        for r in pos:
            sigs.setdefault(r["root_id"], set()).add(str(r["signature"]))
        out[tag] = {
            "zero_move_roots": len(_roots(pos, lambda r: r["moved"] == 0)),
            "nonempty_witness_roots": len(_roots(pos, lambda r: r["moved"] > 0)),
            "alternative_roots": sum(len(v) >= 2 for v in sigs.values()),
            "unknown_fraction_static_valid": round(sum(r[tag] == data.UNKNOWN for r in sv) / max(1, len(sv)), 3),
            "labels": dict(Counter(r[tag] for r in rows)), "labels_static_valid": dict(Counter(r[tag] for r in sv))}
    acc = lambda r, k: any(c["kind"] == k and c["checker_accepted"] for c in r["certificates"])  # noqa: E731
    out["goal_negative_roots"] = len(_roots(sv, lambda r: r[FIRST] == data.NEGATIVE and acc(r, "GOAL_OVERLAP")))
    out["access_negative_roots"] = len(_roots(sv, lambda r: r[FIRST] == data.NEGATIVE and not acc(r, "GOAL_OVERLAP")
                                              and (acc(r, "FIRST_MOVER") or acc(r, "TARGET_PICK"))))
    out["access_negative_roots_by_kind"] = {k: len(_roots(sv, lambda r, k=k: r[FIRST] == data.NEGATIVE and acc(r, k)
                                                          and not acc(r, "GOAL_OVERLAP"))) for k in ("FIRST_MOVER", "TARGET_PICK")}
    out["negative_sources"] = dict(Counter(c["kind"] for r in rows if r[FIRST] == data.NEGATIVE for c in r["certificates"]
                                           if c["checker_accepted"]))
    out["certificates_rejected_by_checker"] = sum(not c["checker_accepted"] for r in rows for c in r["certificates"])
    out["contradictions"] = sum(r[t] == data.CONTRADICTION for r in rows for t in (FIRST, "diagnostic_enriched"))
    out["unknown_reasons"] = dict(Counter((r["unknown_reason"] or "").split(":")[0] for r in rows if r[FIRST] == data.UNKNOWN))
    stay = {r["root_id"]: r for r in rows if r["moved"] == 0}
    out["cohorts"] = {k: sorted(x for x, r in stay.items() if r[FIRST] == v) for k, v in
                      (("initially_feasible", data.POSITIVE), ("certified_initially_infeasible", data.NEGATIVE),
                       ("initially_unresolved", data.UNKNOWN))}
    ne = _roots([r for r in rows if r[FIRST] == data.POSITIVE and r["moved"] > 0], lambda r: True)
    out["nonempty_witness_roots_by_cohort"] = {k: len(ne & set(v)) for k, v in out["cohorts"].items()}
    out["by_stratum"] = {st: {"queries": sum(r["q"]["stratum"] == st for r in col["main"].values()),
                              "assignments": sum(r["stratum"] == st for r in rows),
                              **{t: dict(Counter(r[t] for r in rows if r["stratum"] == st)) for t in (FIRST, "diagnostic_enriched")}}
                         for st in plan["strata"]}
    prim = Counter(r["primary"]["outcome"] for r in col["main"].values())
    out["primary"] = {"outcomes": dict(prim), "queries": len(col["main"]),
                      "complete_success_rate": round(prim.get("SUCCESS", 0) / max(1, len(col["main"])), 3)}
    sem = out["contradictions"] + sum(not v for v in protocol_ok.values())
    f = out[FIRST]
    out["readiness"] = {
        "roots_complete": "PASS" if out["roots_completed"] >= T["roots_complete"] else "BLOCKED",
        "zero_move_roots": "PASS" if f["zero_move_roots"] >= T["zero_move_roots"] else "BLOCKED",
        "nonempty_witness_roots": "PASS" if f["nonempty_witness_roots"] >= T["nonempty_witness_roots"] else "BLOCKED",
        "alternative_roots": "PASS" if f["alternative_roots"] >= T["alternative_roots"] else "BLOCKED",
        "goal_negative_roots": "PASS" if out["goal_negative_roots"] >= T["goal_negative_roots"] else "BLOCKED",
        "access_negative_roots": "PASS" if out["access_negative_roots"] >= T["access_negative_roots"] else "BLOCKED",
        "distinct_signatures_per_multi_alternative_query": "PASS" if f["alternative_roots"] else "BLOCKED",
        "first_pass_unknown_fraction": "PASS" if f["unknown_fraction_static_valid"] <= T["max_first_pass_unknown_fraction"]
        else "BLOCKED",
        "semantic_protocol_failures": "PASS" if sem == 0 else "BLOCKED"}
    out["protocol_checks"] = protocol_ok
    out["decision"] = "READY_TO_SCALE" if all(v == "PASS" for v in out["readiness"].values()) else "NOT_READY_TO_SCALE"
    return out


# ------------------------------------------------------------------ G4A follow-up reconciliation (from saved records)
def certificate_reconciliation(rows: list) -> dict:
    """Records vs unique negative assignments, keyed by (root, query, physical signature)."""
    import json as _json
    key = lambda r: f"{r['root_id']}|{r['query_id']}|{_json.dumps(r['signature'])}"  # noqa: E731
    acc = [(key(r), c["kind"]) for r in rows for c in r["certificates"] if c["checker_accepted"]]
    neg = {key(r): r for r in rows if r[FIRST] == data.NEGATIVE}
    kinds = sorted({k for _, k in acc})
    combos = Counter("+".join(sorted({k for kk, k in acc if kk == n})) for n in neg)
    dup = sum(v - 1 for v in Counter(acc).values() if v > 1)
    return {"accepted_records": len(acc), "rejected_records": sum(not c["checker_accepted"] for r in rows for c in r["certificates"]),
            "records_by_kind": dict(Counter(k for _, k in acc)), "unique_negative_assignments": len(neg),
            "unique_negative_assignments_by_kind": {k: sum(any(kk == n and kind == k for kk, kind in acc) for n in neg) for k in kinds},
            "certificate_combinations": dict(combos), "duplicate_records": dup,
            "assignments_with_two_or_more_distinct_certificates": sum(c.count("+") >= 1 for c in combos.elements()),
            "negatives_without_accepted_certificate": sum(not any(kk == n for kk, _ in acc) for n in neg),
            "accepted_certificate_on_non_negative": sum(1 for r in rows if r[FIRST] != data.NEGATIVE and any(
                c["checker_accepted"] for c in r["certificates"])),
            "contradictions": sum(r[t] == data.CONTRADICTION for r in rows for t in (FIRST, "diagnostic_enriched"))}


def call_accounting(primary: dict) -> dict:
    """Primary target evaluations per episode from the recorded event traces."""
    per = {}
    for q, p in sorted(primary.items()):
        k = [e["kind"] for e in p["events"]]
        c = k.index("COMMIT") if "COMMIT" in k else None
        t = [i for i, x in enumerate(k) if x == "TARGET_EVALUATE"]
        r = [i for i, x in enumerate(k) if x == "RELOCATE"]
        per[q] = {"outcome": p["outcome"], "committed": c is not None, "all_stay_commit": bool(p["commit"]) and not p["commit"]["cells"],
                  "aborted": "ABORT" in k, "primary_target_calls": len(t), "calls_before_commit": sum(c is None or i < c for i in t),
                  "calls_between_relocations": sum(bool(r) and i < max(r) for i in t),
                  "home_diagnostic": (p.get("target") or {}).get("home_return_status")}
    v = list(per.values())
    completed = [x for x in v if x["committed"] and not x["aborted"]]
    return {"episodes": len(v), "completed_repair_episodes": len(completed),
            "completed_with_exactly_one_call": sum(x["primary_target_calls"] == 1 for x in completed),
            "no_repair_episodes": sum(not x["committed"] for x in v),
            "no_repair_with_zero_calls": sum(not x["committed"] and x["primary_target_calls"] == 0 for x in v),
            "aborted_episodes": sum(x["aborted"] for x in v), "all_stay_commits": sum(x["all_stay_commit"] for x in v),
            "calls_before_commit": sum(x["calls_before_commit"] for x in v),
            "calls_between_relocations": sum(x["calls_between_relocations"] for x in v),
            "home_diagnostics_in_primary": dict(Counter(str(x["home_diagnostic"]) for x in v)), "per_episode": per,
            "invariant_holds": all(x["primary_target_calls"] <= 1 and x["calls_before_commit"] == 0
                                   and x["calls_between_relocations"] == 0 and (x["committed"] and not x["aborted"]
                                                                                 or x["primary_target_calls"] == 0) for x in v)}
