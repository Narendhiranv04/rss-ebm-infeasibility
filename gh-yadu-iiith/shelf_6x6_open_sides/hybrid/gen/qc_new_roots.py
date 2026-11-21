"""Post-filter for NEW shelf66_v3 roots (run_shelf66_full.py output): which accepted K1-K3 roots are usable for the confirmatory test.

A root is USABLE when
  * its ledger class (ledger_observed) is 1, 2 or 3 (DEPENDENT / INCONSISTENT / NONE_* / K0 / K4 are not);
  * the root file loads and has the fields the existing raw roots have (labels.B/D, verified_K, goal, initial_cells, objects, ...);
  * it has <= MAX_MOVABLE (7) non-target objects (the models' MAXOBJ = 8 cap incl. the target);
  * fixed-object QC passes: every non-target object can be picked from where it stands with every other object gone, under the
    scene's arm ceiling (Approach-B's QC_ELIGIBLE_ROOTS.json rule).  The check is the team's own pinned, UNMODIFIED tool
    dataset_placement_extension_v2/production_v3/qc/author_evidence/obj_pickable3.py, given the same V:<root>:<obj> jobs that
    dataset_placement_extension_v2/code/qc3d.py builds (no physics is reimplemented here);
  * its layout (objects + cells + target goal) is not an existing raw root's layout, nor another new root's (duplicate guard).

    python qc_new_roots.py RUN_DIR [--workers 4]       -> RUN_DIR/QC_NEW_ROOTS.json (+ RUN_DIR/qc/ jobs and tool output; resumable)
    python qc_new_roots.py --selftest                  2 existing roots with known QC status (one pass, one fixed-object fail)
"""
import argparse, collections, hashlib, json, os, subprocess, sys, tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]                       # shelf_6x6_open_sides/
DS = REPO / "dataset_bd_classquota_without_home_position_no_overpass"
# defaults = this repo's layout; QC_SHELF66 / QC_EXISTING / QC_TOOL override them (home server: README_homeserver.md)
SHELF66 = Path(os.environ.get("QC_SHELF66", DS / "code/shelf66_v3"))
EXISTING = Path(os.environ.get("QC_EXISTING", DS / "raw/roots"))
TOOL = Path(os.environ.get("QC_TOOL", REPO / "dataset_placement_extension_v2/production_v3/qc/author_evidence/obj_pickable3.py"))
TOOL_SHA = "f301812e7399cb5a452e1852dfb7ec864375350635f7122b0c891ca960fa8cf4"   # = qc3d.TOOL_SHA (QC_ELIGIBLE_ROOTS.json "tool")
MAX_MOVABLE = 7
CLASSES = ("1", "2", "3")
FIELDS = ("ids", "objects", "initial_cells", "target_src", "goal", "verified_K", "observed", "ledger_observed", "labels", "n_movable",
          "roles", "parking", "independent_moves", "label_consistency", "spec")
PROVENANCE = {"config_hash": "8d521b1f6d23db23", "label_protocol": "BD_BINARY_BUDGET_V1"}       # = every existing raw root
GEN_PLUS, TEACHER = "BD_GEN_V1+ROUTE_V15+SHELF66V3", "BD3DS66_TEACHER_V2:"


def non_targets(root):
    """= qc3d.non_targets: the non-target objects as obj_pickable3 jobs (ceil_h = the scene's tallest object)."""
    tall = max(o["height"] for o in root["objects"])
    return [dict(id=o["id"], shape=o["shape"], size=o["size"], height=o["height"], cell=list(root["initial_cells"][o["id"]]), ceil_h=tall)
            for o in root["objects"] if not o["id"].startswith("t")]


def layout_key(root):
    objs = sorted((o["shape"], round(o["size"], 5), round(o["height"], 5), tuple(root["initial_cells"][o["id"]]), o["id"].startswith("t"))
                  for o in root["objects"])
    return hashlib.sha256(json.dumps([objs, list(root["goal"])]).encode()).hexdigest()[:16]


def field_problems(root):
    bad = [f for f in FIELDS if f not in root]
    lab = root.get("labels", {})
    bad += [f"labels.{k}" for k in ("B", "D", "original") if k not in lab]
    if not bad and not any(b.get("role") == "minimal" and b.get("y") for b in lab["B"]):
        bad.append("labels.B:no_successful_minimal_sequence")
    if not bad and set(root["initial_cells"]) != {o["id"] for o in root["objects"]}:
        bad.append("initial_cells!=objects")
    if not bad:
        bad += [f"ids.{k}" for k, v in PROVENANCE.items() if root["ids"].get(k) != v]
        bad += [] if root.get("generator_version_plus") == GEN_PLUS else ["generator_version_plus"]
        bad += [] if root["ids"].get("teacher_version", "").startswith(TEACHER) else ["ids.teacher_version"]
    return bad


def run_tool(roots, work, workers):
    """{tag: obj_pickable3 result} for every non-target object of `roots` ({root_id: record}); resumable (the tool skips done tags)."""
    if hashlib.sha256(TOOL.read_bytes()).hexdigest() != TOOL_SHA:
        sys.exit(f"REFUSED: {TOOL} is not the pinned team tool")
    work.mkdir(parents=True, exist_ok=True)
    def results():
        return {x["tag"]: x for f in sorted(work.glob("part*.jsonl")) for x in map(json.loads, f.read_text().splitlines())}
    done = results()                                              # tags done by an earlier run (any part file, any machine)
    jobs = [{"tag": f"V:{r}:{o['id']}", **{k: o[k] for k in ("shape", "size", "height", "cell", "ceil_h")}}
            for r, rec in sorted(roots.items()) for o in non_targets(rec) if f"V:{r}:{o['id']}" not in done]
    (work / "jobs.json").write_text(json.dumps(jobs))
    n = max(1, min(workers, len(jobs)))
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "PYTHONWARNINGS": "ignore",
           "MUJOCO_GL": "disable", "PYTHONDONTWRITEBYTECODE": "1"}
    if jobs:
        procs = [subprocess.Popen([sys.executable, str(TOOL), str(SHELF66), str(work / "jobs.json"), str(work / f"part{k:02d}.jsonl"), str(k), str(n)],
                                  cwd=str(SHELF66.parent), env=env) for k in range(n)]
        codes = [p.wait() for p in procs]
        if any(codes):
            sys.exit(f"obj_pickable3 failed: exit codes {codes}")
    return results()


def qc_status(rid, rec, res):
    nts = non_targets(rec)
    v = [res.get(f"V:{rid}:{o['id']}") for o in nts]
    if not nts:
        return "VACUOUS_PASS_NO_NON_TARGET", []
    if any(x is None for x in v):
        return "NOT_CHECKED", []
    fails = [o["id"] for o, x in zip(nts, v) if not x["pickable"]]
    return ("QC_FAIL_FIXED_OBJECT" if fails else "QC_PASS"), fails


def check_run(run, workers):
    led = [json.loads(x) for x in (run / "ledger.jsonl").read_text().splitlines() if x.strip()]
    acc = [e for e in led if e.get("accepted")]
    cand, out = {}, {}
    for e in acc:
        rid, cls = e["root_id"], str(e["observed"])
        p = run / "roots" / f"{rid}.json"
        try:
            rec = json.loads(p.read_text())
        except (OSError, ValueError) as err:
            out[rid] = {"class": cls, "usable": False, "why": [f"unreadable:{type(err).__name__}"]}
            continue
        why = [] if cls in CLASSES else [f"class_{cls}"]
        why += [f"missing:{f}" for f in field_problems(rec)]
        nm = len(non_targets(rec))
        if nm > MAX_MOVABLE:
            why.append(f"movable_{nm}>{MAX_MOVABLE}")
        out[rid] = {"class": cls, "n_movable": nm, "why": why, "layout": layout_key(rec)}
        if cls in CLASSES:
            cand[rid] = rec
    res = run_tool(cand, run / "qc", workers)
    seen = {layout_key(json.loads(f.read_text())): f.stem for f in EXISTING.glob("*.json") if not f.name.startswith("._")} if EXISTING.is_dir() else {}
    n_existing = len(seen)
    for rid, rec in cand.items():
        st, fails = qc_status(rid, rec, res)
        o = out[rid]
        o.update(qc=st, fixed_objects=fails)
        if st not in ("QC_PASS", "VACUOUS_PASS_NO_NON_TARGET"):
            o["why"].append(st)
        if o["layout"] in seen:
            o["why"].append(f"duplicate_layout_of:{seen[o['layout']]}")
        seen.setdefault(o["layout"], rid)
    for o in out.values():
        o["usable"] = not o["why"]
    rej = collections.Counter(e["reason"].split(":")[1] if e["reason"].startswith(("Reject", "WallCap")) else e["reason"][:40]
                              for e in led if not e.get("accepted"))
    summary = {"ledger_lines": len(led), "accepted": len(acc), "by_ledger_class": dict(collections.Counter(str(e["observed"]) for e in acc)),
               "rejected": dict(rej), "k1_k3_accepted": len(cand),
               "k1_k3_movable_le7": sum(o["n_movable"] <= MAX_MOVABLE for r, o in out.items() if r in cand),
               "k1_k3_qc_pass": sum(o.get("qc") in ("QC_PASS", "VACUOUS_PASS_NO_NON_TARGET") for o in out.values()),
               "usable_by_class": {c: sum(o["usable"] and o["class"] == c for o in out.values()) for c in CLASSES},
               "usable_total": sum(o["usable"] for o in out.values()),
               "mean_movable_k1_k3": round(sum(len(non_targets(r)) for r in cand.values()) / max(len(cand), 1), 2),
               "existing_layouts_compared": n_existing}
    doc = {"rule": "usable = ledger class 1-3, fields present, <= 7 non-target objects, fixed-object QC pass (team tool obj_pickable3.py, "
                   "pinned sha256), layout not a duplicate", "tool_sha256": TOOL_SHA, "summary": summary,
           "usable": sorted(r for r, o in out.items() if o["usable"]), "roots": out}
    (run / "QC_NEW_ROOTS.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(json.dumps(summary, indent=1))
    return doc


def selftest():
    """QC_ELIGIBLE_ROOTS.json: 3ds66-lap-w0000-s004 (K1) AUTHOR_QC_PASS; raw/EXCLUDED_FIXED_OBJECT.json: 3ds66-srv-w0004-s021 (K1) has the
    fixed object o00 (short cylinder in the back-right cell (5,5))."""
    roots = {r: json.loads((EXISTING / f"{r}.json").read_text()) for r in ("3ds66-lap-w0000-s004", "3ds66-srv-w0004-s021")}
    for rec in roots.values():
        assert not field_problems(rec), field_problems(rec)
    with tempfile.TemporaryDirectory() as d:
        res = run_tool(roots, Path(d), 2)
    assert qc_status("3ds66-lap-w0000-s004", roots["3ds66-lap-w0000-s004"], res) == ("QC_PASS", []), res
    assert qc_status("3ds66-srv-w0004-s021", roots["3ds66-srv-w0004-s021"], res) == ("QC_FAIL_FIXED_OBJECT", ["o00"]), res
    print("selftest OK: lap-w0000-s004 QC_PASS, srv-w0004-s021 QC_FAIL_FIXED_OBJECT [o00]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    else:
        check_run(Path(a.run), a.workers)
