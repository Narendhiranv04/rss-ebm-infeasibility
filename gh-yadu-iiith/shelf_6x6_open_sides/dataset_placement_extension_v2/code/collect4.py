"""Production v4 collector entry for the Q2_3D_PARITY_V1 roster.  It runs the production-v3 collector (collect3d.py, byte-identical:
guard, exclusive lock, pinned PDEATHSIG workers, atomic JSON + NPZ, fail-closed monitor, validated resume, label semantics) with
  * the v4 guard bound in (prod4 configures prod3d: v4 version, run prefixes, test bounds, code set), and
  * a v4 metadata preflight that independently REGENERATES the whole roster with q2roster3d.build from SHELF_SRC, the inherited split
    and the snapshot id, and compares it line by line with the frozen roster (plus block / band / shape counts, unique semantic keys).
  python collect4.py --policy-file POLICY.json --dry-run | --confirm TOKEN | --audit       (as collect3d.py)
"""
import gzip, json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import prod4  # noqa: E402,F401  (must precede collect3d: configures prod3d for v4)
import collect3d as C  # noqa: E402
import q2roster3d as Q4  # noqa: E402

MARK = "Integration test of the v4 production path (<= 32 TRAIN queries of the Q2_3D_PARITY_V1 roster). Never part of the dataset.\n"


def preflight(pol, units, n_rows, cur):
    rows = [json.loads(x) for x in gzip.open(C.prod3d.ROOT / pol["inputs"]["roster"], "rt")]
    phys = [r for r in rows if not r["static"]]
    regen = None
    if pol["mode"] == "PRODUCTION_FIRST_COHORT":
        spl = json.loads((C.prod3d.ROOT / pol["inputs"]["split_manifest"]).read_text())["split"]
        snap = json.loads((C.prod3d.ROOT / pol["inputs"]["source_pin"]).read_text())["manifest_sha256"]
        _, want, _, _, _ = Q4.build(cur["source"]["SHELF_SRC"], spl, snap)
        regen = [json.dumps(x, sort_keys=True, default=list) for x in want] == [json.dumps(x, sort_keys=True) for x in rows]
    c = lambda f: dict(sorted(Counter(str(f(r)) for r in phys).items()))  # noqa: E731
    return {"roster_rows": n_rows, "physical_queries": len(phys), "static_contrasts": len(rows) - len(phys), "units": len(units),
            "queries_in_units": sum(map(len, units.values())), "roots": len({r["root_id"] for r in phys}),
            "by_split": c(lambda r: r["split"]), "by_K": c(lambda r: r["K"]), "by_block": c(lambda r: r["block"]), "by_depth": c(lambda r: r["depth"]),
            "by_band": c(lambda r: r["band"]), "by_shape": c(lambda r: r["shape"]), "by_split_block": c(lambda r: f"{r['split']}:{r['block']}"),
            "semantic_keys_unique": len({r["semantic_key"] for r in phys}) == len(phys), "roster_regenerated_equal": regen,
            "static_rows_excluded_from_units": all(not q["static"] for v in units.values() for q in v)}


C.preflight = preflight

if __name__ == "__main__":
    try:
        C.main()
    except SystemExit:
        pf = sys.argv[sys.argv.index("--policy-file") + 1] if "--policy-file" in sys.argv else None
        if pf and Path(pf).is_file():
            pol = json.loads(Path(pf).read_text())
            m = C.prod3d.ROOT / pol.get("run_dir", "__none__") / "TEST_RUN_NOT_PRODUCTION.txt"
            if pol.get("mode") == "INTEGRATION_TEST" and m.exists():
                m.write_text(MARK)                               # collect3d writes the v3 wording; the v4 bound is 32 queries
        raise
