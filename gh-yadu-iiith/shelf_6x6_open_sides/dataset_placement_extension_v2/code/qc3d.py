"""Fixed-object QC of the first K0-K3 production cohort (SHELF66V3 placement sidecar, production v3).

Rule (the generator author's, raw/EXCLUDED_FIXED_OBJECT.json): a root is not counted when one of its NON-TARGET objects cannot be picked
from where it stands with every other object gone, under the scene's own arm ceiling (tallest object of the scene + CEILING_MARGIN).
Check = the author's obj_pickable3.py, run UNMODIFIED (sha256 pinned below): the object alone on the empty shelf, every grasp pose planned
pick-and-place to a front cell (1,2) [(1,3)], then to (2,4) when some pose failed for another reason than reach.

  python qc3d.py jobs     SRC AUTHOR_DIR OUT_JOBS.json           V:<root>:<obj> for every non-target object of every selected K0-K3 root,
                                                                 plus the author's own F:/N: jobs verbatim (equivalence test)
  python qc3d.py run      SHELF66_DIR JOBS OUT_DIR [--workers 12 --cpus 0-11]   the pinned tool, one process per CPU, monitored (fail closed)
  python qc3d.py classify SRC AUTHOR_DIR OUT_DIR                 per-object / per-root verdicts, equivalence, QC_ELIGIBLE_ROOTS.json
"""
import argparse, collections, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE.parent / "production_v3/qc/author_evidence/obj_pickable3.py"
TOOL_SHA = "f301812e7399cb5a452e1852dfb7ec864375350635f7122b0c891ca960fa8cf4"   # == ~/yadu_bd/diag66/obj_pickable3.py on the source host
COUNTED = ("0", "1", "2", "3")
STOP_C, MEM_GB, DISK_GB = 90.0, 4.0, 5.0


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def selected_roots(src):
    sel = json.loads((Path(src) / "selected.json").read_text())["selected"]
    return [(k, r) for k in COUNTED for r in sel.get(k, [])]


def non_targets(root):
    tall = max(o["height"] for o in root["objects"])
    return [dict(id=o["id"], shape=o["shape"], size=o["size"], height=o["height"], cell=list(root["initial_cells"][o["id"]]), ceil_h=tall)
            for o in root["objects"] if not o["id"].startswith("t")], tall


def author(adir):
    """-> {(root, obj): {scan: result}} from the author's result files (S: own-height ceiling; F:/N: scene ceiling)."""
    ev = collections.defaultdict(dict)
    for f in ("scan_pickable.jsonl", "scan2_pickable.jsonl", "obj_pickable.jsonl"):
        for x in map(json.loads, open(Path(adir) / f)):
            t = x["tag"].split(":")
            if len(t) == 3 and t[1].startswith("3ds66"):
                ev[(t[1], t[2])][t[0]] = x
    return ev


def cmd_jobs(a):
    jobs = []
    for _, r in selected_roots(a.src):
        root = json.loads((Path(a.src) / "raw/roots" / f"{r}.json").read_text())
        for o in non_targets(root)[0]:
            jobs.append({"tag": f"V:{r}:{o['id']}", **{k: o[k] for k in ("shape", "size", "height", "cell", "ceil_h")}})
    auth = [j for j in json.loads((Path(a.author) / "scan2_jobs.json").read_text()) if j["tag"][0] in "FN"]
    Path(a.out).write_text(json.dumps(auth + jobs))
    print(json.dumps({"V_jobs": len(jobs), "author_F_N_jobs": len(auth), "roots": len(selected_roots(a.src))}))


def tree(pids):
    import psutil
    out = []
    for p in pids:
        try:
            q = psutil.Process(p); out += [q] + q.children(recursive=True)
        except psutil.NoSuchProcess:
            pass
    return out


def cmd_run(a):
    sys.path.insert(0, str(HERE))
    import psutil
    import run3d
    if sha(TOOL) != TOOL_SHA:
        sys.exit(f"REFUSED: {TOOL} is not the pinned author tool")
    cpus = run3d.cpus_of(a.cpus)
    if len(cpus) != a.workers or len(cpus) > run3d.CEILING or not set(cpus) <= os.sched_getaffinity(0):
        sys.exit(f"REFUSED: {a.workers} workers on {cpus}")
    run3d.sensors()                                             # fail closed before anything starts
    out = Path(a.out).resolve(); out.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
           "PYTHONWARNINGS": "ignore", "MUJOCO_GL": "disable", "PYTHONDONTWRITEBYTECODE": "1"}
    os.setpgrp()
    os.sched_setaffinity(0, set(cpus))
    procs = [subprocess.Popen([sys.executable, str(TOOL), str(Path(a.shelf66).resolve()), str(Path(a.jobs).resolve()), str(out / f"part{k:02d}.jsonl"),
                               str(k), str(a.workers)], cwd=str(Path(a.shelf66).resolve().parent), env=env,
                              preexec_fn=lambda c=c: (os.sched_setaffinity(0, {c}), os.nice(10)))
             for k, c in enumerate(cpus)]
    log, t0, bad = open(out / "monitor.jsonl", "a"), time.time(), []
    while any(p.poll() is None for p in procs):
        time.sleep(2.0)
        try:
            temp, ident = run3d.sensors()
        except run3d.SensorError as err:
            temp, ident, bad = None, None, [f"temp_sensor_lost: {err}"]
        tr = tree([p.pid for p in procs])
        aff = set()
        for q in tr:
            try:
                aff |= set(os.sched_getaffinity(q.pid))
            except OSError:
                pass
        vm, du = psutil.virtual_memory(), psutil.disk_usage(str(out))
        rec = {"t": time.time(), "alive": sum(p.poll() is None for p in procs), "temp_c": temp, "sensor": ident, "affinity_union": sorted(aff),
               "mem_avail_gb": vm.available / 2 ** 30, "disk_free_gb": du.free / 2 ** 30, "load1": os.getloadavg()[0],
               "done": sum(1 for f in out.glob("part*.jsonl") for _ in open(f))}
        bad += [k for k, ok in (("temp", temp is None or temp < STOP_C), ("affinity", aff <= set(cpus)), ("mem", rec["mem_avail_gb"] >= MEM_GB),
                                ("disk", rec["disk_free_gb"] >= DISK_GB), ("worker_failed", all(p.poll() in (None, 0) for p in procs))) if not ok]
        rec["violations"] = bad
        log.write(json.dumps(rec) + "\n"); log.flush()
        if bad:
            for p in procs:
                if p.poll() is None:
                    os.kill(p.pid, signal.SIGKILL)
            break
    end = {"wall_s": time.time() - t0, "violations": bad, "exit_codes": [p.wait() for p in procs]}
    (out / "run_end.json").write_text(json.dumps(end)); print(json.dumps(end))
    sys.exit(3 if bad else 0)


def cmd_classify(a):
    src, out = Path(a.src), Path(a.out)
    res = {}
    for f in sorted(out.glob("part*.jsonl")):
        for x in map(json.loads, open(f)):
            res[x["tag"]] = x
    ev = author(a.author)
    # (E2) the author's own scene-ceiling jobs, re-run here: verdict and first feasible grasp must agree
    a2 = {x["tag"]: x for x in map(json.loads, open(Path(a.author) / "scan2_pickable.jsonl")) if x["tag"][0] in "FN"}
    eq = {"author_jobs": len(a2), "rerun": sum(t in res for t in a2), "verdict_equal": 0, "first_grasp_equal": 0, "poses_tried_equal": 0,
          "differences": []}
    for t, x in a2.items():
        y = res.get(t)
        if y is None:
            continue
        eq["verdict_equal"] += x["pickable"] == y["pickable"]
        eq["first_grasp_equal"] += (x["azimuth"], x["level"]) == (y["azimuth"], y["level"])
        eq["poses_tried_equal"] += x["poses_tried"] == y["poses_tried"]
        if (x["pickable"], x["azimuth"], x["level"], x["poses_tried"]) != (y["pickable"], y["azimuth"], y["level"], y["poses_tried"]):
            eq["differences"].append({"tag": t, "author": [x["pickable"], x["azimuth"], x["level"], x["poses_tried"]],
                                      "here": [y["pickable"], y["azimuth"], y["level"], y["poses_tried"]]})
    excl = json.loads((src / "raw/EXCLUDED_FIXED_OBJECT.json").read_text())["excluded"]
    fails_here = sorted({t.split(":")[1] for t, x in a2.items() if t in res and not res[t]["pickable"]})
    eq["author_excluded_roots"], eq["excluded_reproduced_from_rerun"] = sorted(excl), fails_here
    eq["exclusions_equal"] = fails_here == sorted(excl)
    roots, objs, cnt = {}, [], collections.Counter()
    s_pass_conf = collections.Counter()
    for k, r in selected_roots(src):
        rp = src / "raw/roots" / f"{r}.json"
        root = json.loads(rp.read_text())
        nts, tall = non_targets(root)
        per = []
        for o in nts:
            e, v = ev.get((r, o["id"]), {}), res.get(f"V:{r}:{o['id']}")
            auth_v = ("pass" if (e.get("F") or e.get("N"))["pickable"] else "fail") if ("F" in e or "N" in e) else \
                     ("pass" if "S" in e and e["S"]["pickable"] else "s_fail_no_rerun" if "S" in e else "none")
            here_v = None if v is None else ("pass" if v["pickable"] else "fail")
            if "S" in e and e["S"]["pickable"]:
                s_pass_conf[here_v] += 1
            objs.append({"root": r, "obj": o["id"], "cell": o["cell"], "shape": o["shape"], "height": o["height"], "ceil_h": tall,
                         "author": auth_v, "author_scans": sorted(e), "here": here_v,
                         "here_first_grasp": None if v is None else [v["azimuth"], v["level"]], "here_poses_tried": None if v is None else v["poses_tried"]})
            per.append((auth_v, here_v))
        if not nts:
            st = "VACUOUS_PASS_NO_NON_TARGET"
        elif any(h is None for _, h in per):
            st = "NOT_CHECKED"
        elif any(av == "pass" and h == "fail" for av, h in per) or any(av == "fail" and h == "pass" for av, h in per):
            st = "QC_DISAGREE_EXCLUDED"
        elif any(h == "fail" for _, h in per):
            st = "QC_FAIL_FIXED_OBJECT"
        elif all(av == "pass" for av, _ in per):
            st = "AUTHOR_QC_PASS"
        else:
            st = "SIDECAR_REPRO_QC_PASS"
        roots[r] = {"class": k, "status": st, "eligible": st in ("AUTHOR_QC_PASS", "VACUOUS_PASS_NO_NON_TARGET", "SIDECAR_REPRO_QC_PASS"),
                    "root_sha256": sha(rp), "non_target_objects": len(nts),
                    "objects_author_evidence": sum(av != "none" for av, _ in per), "objects_rechecked_here": sum(h is not None for _, h in per)}
        cnt[(k, st)] += 1
    summary = {"by_class_status": {f"K{k}": {s: cnt[(k, s)] for kk, s in cnt if kk == k} for k in COUNTED},
               "eligible_by_class": {f"K{k}": sum(1 for v in roots.values() if v["class"] == k and v["eligible"]) for k in COUNTED},
               "eligible_total": sum(v["eligible"] for v in roots.values()), "roots": len(roots),
               "objects_checked_here": sum(o["here"] is not None for o in objs), "objects_failing_here": sum(o["here"] == "fail" for o in objs),
               "author_s_pass_objects_rechecked_under_scene_ceiling": dict(s_pass_conf)}
    doc = {"version": "SHELF66V3_FIXED_OBJECT_QC_V3", "rule": json.loads((src / "raw/EXCLUDED_FIXED_OBJECT.json").read_text())["rule"],
           "tool": {"path": str(TOOL.relative_to(HERE.parent)), "sha256": sha(TOOL), "pinned": TOOL_SHA},
           "statuses": {"AUTHOR_QC_PASS": "author's per-object evidence covers every non-target object and all pass; re-check here agrees",
                        "VACUOUS_PASS_NO_NON_TARGET": "the root has no non-target object (provable from the root file)",
                        "SIDECAR_REPRO_QC_PASS": "at least one non-target object had no author result; the author's unmodified tool, re-run here, "
                                                 "passes every non-target object (kept separate from author-approved roots)",
                        "QC_FAIL_FIXED_OBJECT": "a non-target object cannot be picked (excluded)",
                        "QC_DISAGREE_EXCLUDED": "author and re-check disagree on some object (excluded, conservative)",
                        "NOT_CHECKED": "some object has no result (excluded)"},
           "equivalence": eq, "summary": summary, "roots": roots}
    (out / "QC_ELIGIBLE_ROOTS.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    with open(out / "qc_objects.jsonl", "w") as f:
        for o in objs:
            f.write(json.dumps(o, sort_keys=True) + "\n")
    print(json.dumps({"equivalence": {k: v for k, v in eq.items() if k != "differences"}, "n_diff": len(eq["differences"]), **summary}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("jobs"); p.add_argument("src"); p.add_argument("author"); p.add_argument("out")
    p = sp.add_parser("run"); p.add_argument("shelf66"); p.add_argument("jobs"); p.add_argument("out")
    p.add_argument("--workers", type=int, default=12); p.add_argument("--cpus", default="0-11")
    p = sp.add_parser("classify"); p.add_argument("src"); p.add_argument("author"); p.add_argument("out")
    a = ap.parse_args()
    {"jobs": cmd_jobs, "run": cmd_run, "classify": cmd_classify}[a.cmd](a)
