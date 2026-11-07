"""First-cohort freeze (production v3): QC-eligible selected K0-K3 roots -> inherited lineage split, regenerated roster, source pin,
cohort-specific forecast, and the label-blind integration-test roster.  No teacher call, no label, no protected metadata.

  python cohort3d.py build SRC QC_ROOTS PARENT_SPLIT LIVE_ROSTER SOURCE_MANIFEST OUT_DIR     (OUT_DIR = production_v3)
  python cohort3d.py smoke OUT_DIR                                                           (<= 12 TRAIN queries on 4 roots)
"""
import gzip, hashlib, json, sys
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import plan3d  # noqa: E402
import prod3d  # noqa: E402

RULE = "SHELF66V3_PLACE_FIRST_COHORT_V3"
SMOKE_SLOTS = {0: 2, 1: 4, 2: 9}                                # depth -> preferred slot (far / vacated / far); first present slot otherwise


def u(*key):
    return int(hashlib.sha256("|".join(map(str, (RULE,) + key)).encode()).hexdigest()[:15], 16) / 16 ** 15


def write_roster(rows, out):
    with gzip.GzipFile(out, "wb", mtime=0) as f:               # mtime 0: byte-reproducible
        for x in rows:
            f.write((json.dumps(x, sort_keys=True) + "\n").encode())


def build(src, qc_path, parent_split, live_roster, source_manifest, out):
    out, fz = Path(out), Path(out) / "frozen"
    fz.mkdir(parents=True, exist_ok=True)
    qc = json.loads(Path(qc_path).read_text())
    sel = plan3d.load_selected(src)
    par = json.loads(Path(parent_split).read_text())
    counted = {r for r, v in sel.items() if v["K"] in ("0", "1", "2", "3")}
    assert counted == set(qc["roots"]), "QC file must cover exactly the selected K0-K3 roots"
    cohort = sorted(r for r in counted if qc["roots"][r]["eligible"])
    excluded = {r: qc["roots"][r]["status"] for r in sorted(counted) if not qc["roots"][r]["eligible"]}
    # split: inherit, never reassign; closure over the parent groups + family + identical layout of ALL selected roots
    split = {r: par["split"][r] for r in cohort}
    links = list(par["groups"].values())
    by = defaultdict(list)
    for r, v in sel.items():
        by[("fam", v["family"])].append(r); by[("layout", plan3d._lay(v))].append(r)
    links += list(by.values())
    comps = plan3d._uf(sorted(set(sel) | set(par["split"])), links)
    mixed = [g for g in comps if len({par["split"][r] for r in g if r in par["split"]}) > 1]
    groups = {min(g): sorted(set(g) & set(cohort)) for g in comps if set(g) & set(cohort)}
    sman = {"rule": "restriction of the extended live split (production_v2/split_manifest_live.json) to the first-cohort roots: every "
                    "assignment inherited unchanged; groups = union-find over the parent groups + parent_family_id + identical layout of all "
                    "selected roots", "parent_sha256": par["sha256"], "n_roots": len(cohort), "counts": Counter(split.values()),
            "by_class": {s: Counter(sel[r]["K"] for r in cohort if split[r] == s) for s in ("train", "val", "test")},
            "groups": groups, "split": dict(sorted(split.items()))}
    sman["sha256"] = hashlib.sha256(json.dumps(sman["split"], sort_keys=True).encode()).hexdigest()
    (fz / "split_manifest_first_cohort.json").write_text(json.dumps(sman, indent=1, sort_keys=True, default=dict))
    # roster: the frozen V1 production rule (plan3d.roster_for) over the cohort roots, sorted by root id
    rows = []
    for r in cohort:
        for x in plan3d.roster_for(sel[r]):
            x["split"], x["reference"] = split[r], False
            rows.append(x)
    write_roster(rows, fz / "first_cohort_roster.jsonl.gz")
    live = [json.loads(x) for x in gzip.open(live_roster, "rt")]
    cs = set(cohort)
    live_c = [json.dumps(x, sort_keys=True) for x in live if x["root_id"] in cs]
    equal = live_c == [json.dumps(x, sort_keys=True) for x in rows]
    seen = defaultdict(set)
    for x in rows:
        seen[x["scene_hash"]].add(split[x["root_id"]])
    leak = {h: sorted(s) for h, s in seen.items() if len(s) > 1}
    # source pin: cohort root bytes == the published snapshot manifest; plus the source code / model / selection files
    man = json.loads(gzip.open(source_manifest, "rt").read())
    ent = man.get("files") or man.get("entries") or man
    roots_sha = {r: prod3d.sha_file(Path(src) / "raw/roots" / f"{r}.json") for r in cohort}
    drift = [r for r in cohort if (ent.get(f"raw/roots/{r}.json") or {}).get("sha256") != roots_sha[r] or qc["roots"][r]["root_sha256"] != roots_sha[r]]
    assert not drift, f"source drift: {drift[:3]}"
    pin = {"version": "SHELF66V3_PLACE_SOURCE_PIN_V3", "snapshot": "naren/3d-dataset-relocations@32c3194f: live SHELF66V3 store cut 2026-11-07T05:28:58Z",
           "manifest_sha256": prod3d.sha_file(source_manifest), "qc_roots_sha256": prod3d.sha_file(qc_path),
           "source_files_sha256": prod3d.source_files(src), "n_roots": len(cohort), "roots": roots_sha}
    (fz / "source_pin_first_cohort.json").write_text(json.dumps(pin, indent=1, sort_keys=True))
    # forecast: the V2 method (report3d.forecast) on this roster, measured adaptive_v1 costs (run A), V2 efficiency / storage constants
    import report3d
    an = json.loads((HERE.parent / "mock_v2/validation/analysis_v2.json").read_text())
    st = an["storage"]["adaptive_v1_A"]
    eff = an["forecast_live"]["adaptive_v1"]["parallel_efficiency"]
    pos = an["forecast_live"]["note"]["positive_rate_adaptive_A"]
    A = report3d.load_run(HERE.parent / "mock_v2/runs/adaptive_v1_A")
    fc = report3d.forecast(fz / "first_cohort_roster.jsonl.gz", A, 12, eff, bytes_per_witness=st["bytes_per_witness"],
                           json_per_query=st["json_bytes_per_query"], pos_rate=pos)
    mock_strata = Counter((r["K"], r["depth"], r["band"]) for r in A["recs"].values())
    phys = [x for x in rows if not x["static"]]
    strata = Counter((x["K"], x["depth"], x["band"]) for x in phys)
    fc.update(wall_hours_expected=round(fc["wall_days_expected"] * 24, 2), wall_hours_p95_x1_5=round(fc["wall_days_p95_x1_5"] * 24, 2),
              disk_gb_buffered=fc["disk_gb_x1_5"], parallel_efficiency_source="mock_v2 sweep_12 vs serial (analysis_v2.json)",
              strata_backed_off={f"K{k}:d{d}:{b}": n for (k, d, b), n in sorted(strata.items()) if mock_strata[(k, d, b)] < 3},
              not_forecast=["K4 (0 selected roots)", "NONE (out of the first cohort)", f"{len(excluded)} QC-excluded roots"])
    (fz / "forecast_first_cohort.json").write_text(json.dumps(fc, indent=1, sort_keys=True, default=dict))
    files = {"roster": "production_v3/frozen/first_cohort_roster.jsonl.gz", "source_pin": "production_v3/frozen/source_pin_first_cohort.json",
             "split_manifest": "production_v3/frozen/split_manifest_first_cohort.json", "qc_roots": str(Path(qc_path).resolve().relative_to(prod3d.ROOT)),
             "forecast": "production_v3/frozen/forecast_first_cohort.json"}
    coh = {"rule": RULE + ": every root of the published snapshot's selected.json in classes K0-K3 whose fixed-object QC status is eligible "
                          "(AUTHOR_QC_PASS, VACUOUS_PASS_NO_NON_TARGET, SIDECAR_REPRO_QC_PASS); no replacement for excluded roots; K4 / NONE excluded",
           "cohort_roots": cohort, "n_cohort_roots": len(cohort), "roots_with_queries": len({x["root_id"] for x in phys}),
           "by_class": Counter(sel[r]["K"] for r in cohort), "by_qc_status": Counter(qc["roots"][r]["status"] for r in cohort),
           "by_class_qc": {f"K{k}": Counter(qc["roots"][r]["status"] for r in cohort if sel[r]["K"] == k) for k in ("0", "1", "2", "3")},
           "excluded": excluded, "files": files, "sha256": {k: prod3d.sha_file(prod3d.ROOT / v) for k, v in files.items()}}
    (fz / "cohort_manifest.json").write_text(json.dumps(coh, indent=1, sort_keys=True, default=dict))
    need = {"K": {"0", "1", "2", "3"}, "depth": {0, 1, 2}, "band": {"near", "mid", "far", "vacated"}}
    audit = {"split": {"old_assignments_changed": sum(split[r] != par["split"][r] for r in cohort), "parent_sha256": par["sha256"],
                       "components_over_all_selected_roots": len(comps), "components_spanning_two_splits": len(mixed),
                       "counts": sman["counts"], "by_class": sman["by_class"], "sha256": sman["sha256"]},
             "leakage": {"derived_scene_layouts": len(seen), "cross_split_layouts": len(leak), "examples": list(leak.items())[:5],
                         "cross_split_roots": 0 if not mixed else sum(len(g) for g in mixed)},
             "roster": {"rule": plan3d.RULE + " (plan3d.roster_for; production: every slot of every root, reference = False)",
                        "rows": len(rows), "physical_queries": len(phys), "static_contrasts": len(rows) - len(phys),
                        "roots_with_queries": len({x["root_id"] for x in phys}), "equal_to_v2_live_roster_restricted_to_cohort": equal,
                        "by_split": Counter(x["split"] for x in phys), "by_K": Counter(x["K"] for x in phys), "by_depth": Counter(x["depth"] for x in phys),
                        "by_band": Counter(x["band"] for x in phys), "by_split_K": Counter(f"{x['split']}:K{x['K']}" for x in phys),
                        "by_depth_band": Counter(f"d{x['depth']}:{x['band']}" for x in phys), "static_by_kind": Counter(x["static"] for x in rows if x["static"]),
                        "coverage_missing": {k: sorted(map(str, v - {x[k] for x in phys})) for k, v in need.items()},
                        "every_row_qc_eligible_and_pinned": all(qc["roots"][x["root_id"]]["eligible"] and x["root_id"] in roots_sha for x in rows),
                        "sha256": prod3d.sha_file(fz / "first_cohort_roster.jsonl.gz")},
             "semantics": {"physical query": "V_place(S_hyp, o, d) conditional on a certified post-grasp hold: already held -> lift -> carry -> lower "
                                             "-> release -> retreat -> settle (13 MuJoCo gates); not pick accessibility, not target-task success",
                           "label 1": "VERIFIED: a stored witness passes every gate (re-verified by fresh-process replay)",
                           "label 0": "NEG_CERTIFIED_GEOMETRY (scope: 32 poses x <= 4 source IK candidates x the 2-D footprint carry model) or "
                                      "NEG_UNDER_POLICY (BOUNDED_SEARCH / EXECUTION / DETERMINISTIC_PLAN under adaptive_v1, with its scope record)",
                           "label None": "UNLABELED_NO_VALID_POST_GRASP_STATE / UNLABELED_SCENE_NOT_SETTLED / INTERRUPTED: never a negative",
                           "static contrast": "OCCUPIED / OUT_OF_BOUNDS destinations: static ineligible rows, never sent to the physical teacher, "
                                              "kept apart from learned negatives"},
             "protected_metadata": "the roster rule reads object ids / shapes / sizes / heights, initial cells and the grid only (plan3d docstring);"
                                   " no roles, blockers, B/D labels, K-construction spec or outcomes; K is a reporting field",
             "qc_excluded_roots": len(excluded), "cohort_roots": len(cohort)}
    (out / "SPLIT_AND_ROSTER_AUDIT.json").write_text(json.dumps(audit, indent=1, sort_keys=True, default=dict))
    print(json.dumps({"cohort": len(cohort), "excluded": len(excluded), "physical": len(phys), "static": len(rows) - len(phys),
                      "roster_equal_v2_restricted": equal, "cross_split_layouts": len(leak), "mixed_components": len(mixed),
                      "cpu_hours": fc["cpu_hours"], "wall_h_p95x1.5": fc["wall_hours_p95_x1_5"], "disk_gb_x1.5": fc["disk_gb_x1_5"]}, default=dict))


def smoke(out):
    """Label-blind integration-test roster: 4 TRAIN cohort roots (one per K0..K3, hash-ranked, at least one author-approved and one
    sidecar-certified root), one query per depth (preferred slots far / vacated / far): 12 rows identical to production rows."""
    fz = Path(out) / "frozen"
    rows = [json.loads(x) for x in gzip.open(fz / "first_cohort_roster.jsonl.gz", "rt")]
    qc = json.loads((prod3d.ROOT / json.loads((fz / "cohort_manifest.json").read_text())["files"]["qc_roots"]).read_text())["roots"]
    by = defaultdict(list)
    for x in rows:
        if not x["static"] and x["split"] == "train":
            by[x["root_id"]].append(x)
    cand = {k: sorted((r for r, v in by.items() if v[0]["K"] == k and {q["depth"] for q in v} == {0, 1, 2}), key=lambda r: u("smoke", r))
            for k in ("0", "1", "2", "3")}
    for i in range(50):                                        # deterministic: first hash-ranked combination meeting the provenance mix
        pick = [cand[k][(i >> (2 * j)) % min(4, len(cand[k]))] for j, k in enumerate(("0", "1", "2", "3"))]
        st = {qc[r]["status"] for r in pick}
        if "AUTHOR_QC_PASS" in st and "SIDECAR_REPRO_QC_PASS" in st:
            break
    sel = []
    for r in pick:
        for d, slot in SMOKE_SLOTS.items():
            qs = [q for q in by[r] if q["depth"] == d]
            sel.append(next((q for q in qs if q["slot"] == slot), qs[0]))
    write_roster(sel, Path(out) / "integration" / "smoke_roster.jsonl.gz")
    print(json.dumps({"roots": pick, "qc": [qc[r]["status"] for r in pick], "queries": [q["qid"] for q in sel],
                      "bands": [q["band"] for q in sel]}))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(*sys.argv[2:8])
    elif sys.argv[1] == "smoke":
        (Path(sys.argv[2]) / "integration").mkdir(parents=True, exist_ok=True)
        smoke(sys.argv[2])
