"""P17 custody after the sealed runs (sealed-output manifest -> push -> unseal), the prospective oracle guide, and the final analyses:
the frozen p17_analysis functions applied to the unsealed prospective outputs and to the retrospective P16-root re-analysis. Reads and
hashes outputs only; every decision rule it applies is in the PREREG-hashed code."""
from __future__ import annotations

import collections
from pathlib import Path

import config as cf
from p17_eval import OUT, rd, sha

from p17_data import DATA, _plan, merged

T = DATA / "p17test"
EVAL = OUT / "eval"
ADMIT_ARMS = ("LB", "LB+G", "LB+G|nocpl", "LB-W", "FULLB", "LB-F2", "LC", "A", "SF")


# ------------------------------------------------------------------ custody: manifest -> push -> unseal
def p_manifest() -> dict:
    """runs/p17/data/p17test/SEALED_MANIFEST.json: sha256 of every sealed output; refuses while a sealed worker runs or a roster is incomplete
    (counts root keys only, never outcomes)."""
    import p17_prosp as pp
    run = [f.name for f in (OUT / "running").glob("p17_4_seal_*.json")]
    if run:
        raise SystemExit(f"sealed workers still running: {run}")
    ro = pp.p_rosters()
    n = {kind: len(merged("p17test", f"sealed_{kind}")) for kind in ("labels", "exec_b", "exec_d")}
    want = {"labels": len(ro["b"]), "exec_b": len(ro["b"]), "exec_d": len(ro["d"])}
    if n != want:
        raise SystemExit(f"incomplete sealed outputs: {n} vs {want}")
    files = sorted(str(f) for f in T.glob("sealed_*.json"))
    m = {"files": {f: sha(f) for f in files}, "n": len(files), "roots": n, "prereg": sha(OUT / "PREREG.json")}
    cf.write_json(T / "SEALED_MANIFEST.json", m)
    return m


def p_unseal() -> None:
    import p16_eval as p16e
    m = T / "SEALED_MANIFEST.json"
    if not p16e._on_origin(str(m)):
        raise SystemExit("sealed manifest not on origin/main")
    if any(sha(f) != h for f, h in rd(m)["files"].items()):
        raise SystemExit("sealed file changed after the manifest")
    (T / "UNSEALED").write_text(sha(m) + "\n")


def _unsealed():
    if not (T / "UNSEALED").exists():
        raise SystemExit("sealed")


# ------------------------------------------------------------------ prospective oracle guide (non-deployable reference)
def p_oracles() -> dict:
    """{root: ({pool object: in union}, lattice type)} on the D roster: P17 parking-lattice union from the sealed labels (PREREG); None where
    the lattice gives no defined mask."""
    import p17_mechjobs as mj
    import p17_prosp as pp
    _unsealed()
    labs, out = merged("p17test", "sealed_labels"), {}
    for k in pp.p_rosters()["d"]:
        d, pool = mj._lat(labs[k])
        out[k] = ({o: m for o, m in zip(pool, d["mask"])} if d["mask"] is not None else None, d["type"])
    return out


def run_p_oracle(part) -> None:
    import p17_exec as ex
    import p17_prosp as pp
    orc = {k: v[0] for k, v in p_oracles().items() if v[0] is not None}
    ex.run_roots("p17test", [k for k in pp.p_rosters()["d"] if k in orc], part, "d", guides=["oracle"], oracles=orc, tag="exec_oracle",
                 claims=T / "oracle_claims")


# ------------------------------------------------------------------ analyses
def _unions(orc) -> dict:
    return {k: {o for o, m in x[0].items() if m} for k, x in orc.items() if x[0] is not None}


def _d_sets(rows, orc_rows, unions) -> dict:
    """All roots without the oracle (primary contrast), the oracle-defined subset with it, and NAT / constructed strata. The P17 guide
    record names its order trace 'trace' (same entries, incl. 'first'); the frozen p16 analysis reads it as 'order_trace' (key alias only)."""
    import p17_analysis as pa
    al = lambda x: {**x, "order_trace": x["trace"]}  # noqa: E731
    base = {k: {"family": v["family"], "guides": {g: al(x) for g, x in v["guides"].items() if g != "oracle"}} for k, v in rows.items()}
    wo = {k: {"family": base[k]["family"], "guides": {**base[k]["guides"], "oracle": al(orc_rows[k])}} for k in base if k in orc_rows and k in unions}
    st = {s: {k: v for k, v in base.items() if (v["family"] == "NAT") == (s == "NATURAL")} for s in ("NATURAL", "CONSTRUCTED")}
    return {"all_roots": pa.d_analysis(base, unions), "oracle_subset": pa.d_analysis(wo, unions) if wo else None,
            "strata": {s: pa.d_analysis(r, unions) if r else None for s, r in st.items()}}


def label_summary(labs) -> dict:
    fam = collections.defaultdict(collections.Counter)
    seq = collections.defaultdict(collections.Counter)
    for v in labs.values():
        fam[v["family"]][v["original"]["status"]] += 1
        for x in v["sequences"].values():
            seq[str(x["len"])][{1: "VERIFIED", 0: "CERTIFIED"}.get(x["y"], "UNKNOWN")] += 1
    return {"roots": len(labs), "original_by_family": {f: dict(c) for f, c in fam.items()}, "sequences_by_length": {n: dict(c) for n, c in sorted(seq.items())},
            "cpu_s": sum(v["cpu_s"] for v in labs.values())}


def matched_rows(labs, b, d, rs, kinds=("Dp16", "Dp17")) -> dict:
    """Per D-roster root (B and D both executed): original label status, every B arm's choice / execution status / replay, every D guide's
    found / replay / moves, and each D family's raw original-target feasibility (seed-ensemble mean; the same quantity d_offline scores)."""
    import numpy as np
    import d_blockers as db
    import p17_d as pd
    ms = {kind: pd.models(kind) for kind in kinds}
    by = {r["key"]: r for r in rs}
    out = {}
    for k in d:
        if k not in b or k not in labs or k not in by:
            continue
        r, pool = by[k], labs[k]["catalogue"]["pool"]
        ix = [r["ids"].index(o) for o in pool]
        out[k] = {"family": d[k]["family"], "original": labs[k]["original"]["status"],
                  "b": {a: {"choice": x["choice"], "status": x["status"], "replay": bool(x.get("replay"))} for a, x in b[k]["b"]["arms"].items()},
                  "d": {g: {"found": x["found"], "replay": x["replay"], "moves": x["moves"]} for g, x in d[k]["d"].items()},
                  "d_pfeas": {kind: float(np.mean([db.predict(m, r["R"], ix)["p_feasible"] for m in mm])) for kind, mm in ms.items()}}
    return out


def _w(name, obj) -> None:
    EVAL.mkdir(parents=True, exist_ok=True)
    cf.write_json(EVAL / name, obj)


def analyze_prospective() -> dict:
    import p17_analysis as pa
    import p17_b as pb
    _unsealed()
    labs = merged("p17test", "sealed_labels")
    b = merged("p17test", "sealed_exec_b")
    brows = {k: {"family": v["family"], "b": v["b"]} for k, v in b.items()}
    nat = {s: {k: v for k, v in brows.items() if (v["family"] == "NAT") == (s == "NATURAL")} for s in ("NATURAL", "CONSTRUCTED")}
    B = {"analysis": pa.b_analysis(brows), "strata": {s: pa.b_analysis(r) for s, r in nat.items() if r}, "offline": pa.b_offline(brows, labs),
         "noop_kappa": pa.noop_kappa(brows), "admission": pa.admission_curves(brows, labs, ADMIT_ARMS),
         "cpu_root_mean": sum(v["cpu_root"] for v in b.values()) / len(b)}
    orc = p_oracles()
    d = merged("p17test", "sealed_exec_d")
    orc_rows = {k: v["d"]["oracle"] for k, v in merged("p17test", "exec_oracle").items()}
    D = _d_sets({k: {"family": v["family"], "guides": v["d"]} for k, v in d.items()}, orc_rows, _unions(orc))
    D["oracle_types"] = dict(collections.Counter(x[1] for x in orc.values()))
    rs = pb.prepare(pb.records(_plan("p17test")["roots"], labs, "p17test"))
    D["offline"] = {kind: pa.d_offline(kind, rs, labs) for kind in ("D", "Dp", "Dp16", "Dp17")}
    out = {"cohort": "p17test", "unsealed": (T / "UNSEALED").read_text().strip(), "labels": label_summary(labs), "b": B, "d": D,
           "matched": matched_rows(labs, b, d, rs)}
    _w("P17_PROSPECTIVE.json", out)
    return out


def analyze_retrospective() -> dict:
    """P17_REANALYSIS_OF_P16_ROOTS: every guide incl. the oracle (P16 unsealed parking-lattice union) on all 48 P16 D test roots."""
    import p16_eval as p16e
    import p17_prosp as pp
    r = merged("test", "retro")
    miss = [k for k in pp.retro_keys() if k not in r]
    orc = p16e.test_oracles()
    rows = {k: {"family": v["family"], "guides": v["d"]} for k, v in r.items()}
    orc_rows = {k: v["d"]["oracle"] for k, v in r.items() if "oracle" in v["d"]}
    out = {"label": "P17_REANALYSIS_OF_P16_ROOTS", "roots": len(r), "missing": miss, **_d_sets(rows, orc_rows, _unions(orc)),
           "oracle_types": dict(collections.Counter(x[1] for x in orc.values())), "cpu_root_total": sum(v["cpu_root"] for v in r.values())}
    _w("P17_RETROSPECTIVE.json", out)
    return out


# ------------------------------------------------------------------ gates and final manifest
def final_gates() -> None:
    import p17_eval as pe
    P, R = rd(EVAL / "P17_PROSPECTIVE.json"), rd(EVAL / "P17_RETROSPECTIVE.json")
    m7 = {k: v["M7_in_domain"] for w in (0, 1, 2) for k, v in rd(OUT / "mechanisms" / f"M7_LIB_w{w}.json").items()}
    pe.gate("3", status="PARTIAL", realized=["M0", "M1", "M2", "M3", "M4", "M5", "M6", "M8_3", "M8_4", "M9"], m7={"witness": "p17train|p13_2026271039",
            "finite_domain_proof": "BLOCKED", "roots": m7}, report="0093")
    pe.gate("4", status="PASS", sealed_manifest=sha(T / "SEALED_MANIFEST.json"), unsealed=(T / "UNSEALED").read_text().strip(),
            rosters=rd(T / "SEALED_MANIFEST.json")["roots"], oracle_roots=len(merged("p17test", "exec_oracle")), report="0096")
    pe.gate("5", status={"METHOD_B_2D_COMPLETE": True, "METHOD_D_2D_COMPLETE": "PARTIAL_BLOCKED_M7", "P17_2D_COMPLETE": False},
            b_primary=P["b"]["analysis"]["primary"], d_primary=P["d"]["all_roots"]["primary"], d_retro_primary=R["all_roots"]["primary"],
            reports=["0097", "0098", "0099"])


def final_manifest() -> dict:
    import glob
    import p17_eval as pe
    import p17_prosp as pp
    P, R = rd(EVAL / "P17_PROSPECTIVE.json"), rd(EVAL / "P17_RETROSPECTIVE.json")
    g = [rd(f) for f in sorted(glob.glob(str(OUT / "guards" / "*.json")))]
    cpu, wall = collections.Counter(), collections.Counter()
    for x in g:
        cpu[x["category"]] += x["cpu_s"]
        wall[x["category"]] += x["wall_s"]
    fam = lambda split: dict(collections.Counter(r.get("family") for r in _plan(split)["roots"].values()))  # noqa: E731
    reg = rd(OUT / "regression_final" / "regression_result.json") if (OUT / "regression_final" / "regression_result.json").exists() else None
    out = {"stage": "P17-5", "base": "bd9a2e8", "head_at_manifest": pe._git("rev-parse", "HEAD").strip(), "no_3d": True,
           "verdicts": {"METHOD_B_2D_COMPLETE": True, "METHOD_D_2D_COMPLETE": "PARTIAL_BLOCKED_M7", "P17_2D_COMPLETE": False},
           "gates": {s: rd(OUT / f"P17_{s}_GATE.json").get("status", "PASS") for s in ("0", "1", "3", "4", "5") if (OUT / f"P17_{s}_GATE.json").exists()},
           "historical_mismatches": pe.verify_historical(), "prereg": sha(OUT / "PREREG.json"), "sealed_manifest": sha(T / "SEALED_MANIFEST.json"),
           "unsealed": (T / "UNSEALED").read_text().strip(), "code": {c: sha(c) for c in pp.CODE + ("p17_final.py", "p17_viz.py", "p17_data.py", "p17_prosp.py",
                                                                                                "p17_mechjobs.py", "p17_mech.py")},
           "checkpoints": {c: sha(c) for c in sorted(glob.glob("runs/p17/b/ckpt/*.pt") + glob.glob("runs/p17/d/ckpt/*.pt"))},
           "training_data": {f: sha(f) for f in ("runs/p17/b/TRAIN.pt", "runs/p17/d/D_ITEMS.pt")},
           "splits": {s: fam(s) for s in ("p17train", "p17calib", "p17test")}, "rosters": pp.p_rosters() | {"retro": pp.retro_keys()},
           "labels": {"p17test": P["labels"], "unknown_originals_p17test": sum(c.get("UNKNOWN_PLANNER", 0) for c in P["labels"]["original_by_family"].values())},
           "planner_profile": "P16_FAST", "budgets": rd(OUT / "PREREG.json")["budgets"],
           "cpu_h_by_category": {k: round(v / 3600, 2) for k, v in cpu.items()}, "wall_h_by_category": {k: round(v / 3600, 2) for k, v in wall.items()},
           "cpu_h_total": round(sum(cpu.values()) / 3600, 1),
           "cancellations": [{"job": x["call"][:80], "stop": x["stop"], "returncode": x["returncode"], "cpu_s": round(x["cpu_s"])} for x in g if x["stop"] != "COMPLETED"],
           "cancellation_note": "all non-COMPLETED jobs are SIGTERM operator stops of mechanism screens / M5 mining / the label pilot; partial outputs are "
                                "kept and only used where stated (0093); a temporary MAX_JOBS edit was reverted before any sealed run (PREREG hashes re-verified)",
           "contrasts": {"b_primary": P["b"]["analysis"]["primary"], "b_secondary": P["b"]["analysis"]["secondary"], "d_primary": P["d"]["all_roots"]["primary"],
                         "d_secondary": P["d"]["all_roots"]["secondary"], "d_retro_primary": R["all_roots"]["primary"], "d_retro_secondary": R["all_roots"]["secondary"]},
           "proofs": {"m7_library_domain": {k: v["M7_in_domain"] for w in (0, 1, 2) for k, v in rd(OUT / "mechanisms" / f"M7_LIB_w{w}.json").items()},
                      "d_claims_prospective": {gd: x["claims"] for gd, x in P["d"]["all_roots"]["per_guide"].items()}},
           "figures": sorted(str(p) for p in Path("visualizations/p17").rglob("*") if p.suffix in (".png", ".gif")),
           "reports": sorted(str(p) for p in Path("responses-phases").glob("009*_P17_*.md")) + ["responses-phases/P17_COMPLETE_2D_B_AND_D_SCIENTIFIC_DECISION.md"],
           "regression": reg and {k: reg[k] for k in ("tests", "replays", "passed", "verdict") if k in reg}}
    out["hash"] = cf.stable_hash({k: v for k, v in out.items() if k != "hash"})
    cf.write_json(OUT / "P17_FINAL_MANIFEST.json", out)
    return out
