"""Targeted finger-width sensitivity of the placement labels (production v3, work package C).  Two protocol identities, never mixed:
  SOURCE_EXACT   the frozen V2 teacher: the STRUCTURE check (static_ck) keeps the source's default (fully open) finger qpos, while every
                 OBJECT check pre-shapes the fingers to the grasp's open width (SHELF66V3)
  ALT_PRESHAPED  identical, except the structure check ALSO pre-shapes the fingers whenever the teacher has set a check width
                 (screening via ik_candidates and every planning chain).  Patched at class level in THIS process only; its own protocol hash.
The production collector never imports this module.

  python finger3d.py static ROSTER QIDS_FILE OUT.json [--workers 12 --cpus 0-11]     per pose: IK candidates / eligible branches under both
  python finger3d.py run    ROSTER QIDS_FILE RUN_DIR --variant alt|source [--workers --cpus]    adaptive_v1, run3d unit format (replayable)
  python finger3d.py compare ROSTER SOURCE_RUN ALT_RUN OUT.json                          label transitions per source label class
"""
import argparse, gzip, json, os, sys, time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ALT = {"on": False}
ALT_NAME = "SHELF66V3_PLACE_V2_ALT_PRESHAPED_STRUCTURE"


def install(P, variant):
    """Install the switchable alternate structure check (class level) and, for variant alt, the alternate protocol identity."""
    import numpy as np
    S = P.S
    base = S.XArm7TableEnv._static_hits

    def hits(self, q):
        ck = self.static_ck
        if not ALT["on"] or self._check_width is None:
            return base(self, q)
        if not hasattr(ck, "_fadr"):
            ck._fadr = np.array([ck.m.joint(j).qposadr[0] for j in S.GRIP_JOINTS])
        saved = ck.d.qpos[ck._fadr].copy()
        ck.d.qpos[ck._fadr] = self._finger_q(self._check_width)
        try:
            return base(self, q)
        finally:
            ck.d.qpos[ck._fadr] = saved
    S.XArm7TableEnv._static_hits = hits
    if variant == "alt":
        ALT["on"] = True
        P.TEACHER = ALT_NAME
        orig = P.protocol

        def protocol():
            _, sem = orig()
            sem = {**sem, "name": ALT_NAME, "check_width": "grasp open_width in every object AND structure check (ALTERNATE identity, not the source)"}
            return P.sha(sem)[:16], sem
        P.protocol = protocol


def rows_of(roster, qids_file):
    keep = set(Path(qids_file).read_text().split()) if qids_file != "all" else None
    return [r for r in (json.loads(x) for x in gzip.open(roster, "rt")) if not r["static"] and (keep is None or r["qid"] in keep)]


def _static_job(args):
    cpu, rows = args
    os.sched_setaffinity(0, {cpu})
    import numpy as np
    import placement3d as P
    install(P, "source")
    out = []
    for r in rows:
        root = json.loads((P.SRC / "raw/roots" / f"{r['root_id']}.json").read_text())
        sc = P.Scene(root["objects"], r["cells"])
        e = sc.e
        per = []
        for g in P.poses(sc, r["object"]):
            res = {}
            for v in ("source", "alt"):
                ALT["on"] = v == "alt"; e._ikc = {}
                sc.restore(); e._focus(e.obj_id(r["object"])); e._check_width = g["open_width"]
                c = e.ik_candidates(g["position"], g["R"], e.q_ready)
                el = [k for k, q in enumerate(c) if not e._object_hits(q)]
                e._check_width = None
                res[v] = (np.round(np.array(c), 4).tolist(), el)
            cls = lambda c, el: "IK_INVALID" if not c else "HELD_INVALID" if not el else "TO_CARRY_TEST"  # noqa: E731
            per.append({"pose": g["id"], "source": cls(*res["source"]), "alt": cls(*res["alt"]),
                        "n_cand": [len(res["source"][0]), len(res["alt"][0])], "n_elig": [len(res["source"][1]), len(res["alt"][1])],
                        "candidates_equal": res["source"][0] == res["alt"][0], "eligible_equal": res["source"] == res["alt"]})
        ALT["on"] = False
        out.append({"qid": r["qid"], "poses": len(per), "pose_status_changes": [p for p in per if p["source"] != p["alt"]],
                    "poses_candidates_differ": sum(not p["candidates_equal"] for p in per), "poses_eligible_differ": sum(not p["eligible_equal"] for p in per)})
    return out


def cmd_static(a):
    import multiprocessing as mp
    rows = rows_of(a.roster, a.qids)
    import run3d
    cpus = run3d.cpus_of(a.cpus)[:a.workers]
    os.sched_setaffinity(0, set(cpus))
    with mp.get_context("spawn").Pool(len(cpus)) as pool:
        parts = pool.map(_static_job, [(c, rows[i::len(cpus)]) for i, c in enumerate(cpus)])
    res = sorted((x for p in parts for x in p), key=lambda x: x["qid"])
    summ = {"queries": len(res), "queries_any_pose_status_change": sum(bool(x["pose_status_changes"]) for x in res),
            "queries_any_candidate_difference": sum(x["poses_candidates_differ"] > 0 for x in res),
            "pose_status_transitions": dict(Counter(f"{p['source']}->{p['alt']}" for x in res for p in x["pose_status_changes"])),
            "poses": sum(x["poses"] for x in res), "poses_candidates_differ": sum(x["poses_candidates_differ"] for x in res)}
    Path(a.out).write_text(json.dumps({"summary": summ, "queries": res}, indent=1))
    print(json.dumps(summ))


def worker_v(cpu, tasks, results, run, stages, guard, early, variant):
    import placement3d as P
    install(P, variant)
    import run3d
    run3d.worker(cpu, tasks, results, run, stages, guard, early)


def cmd_run(a):
    import multiprocessing as mp
    import placement3d as P
    import policy3d as Q
    import run3d
    rows = rows_of(a.roster, a.qids)
    run = Path(a.run); (run / "units").mkdir(parents=True, exist_ok=True)
    install(P, a.variant)
    stages = Q.POLICIES["adaptive_v1"]
    (run / "run_manifest.json").write_text(json.dumps({"variant": a.variant, "protocol": P.protocol(), "policy": "adaptive_v1", "stages": stages,
                                                       "qids": [r["qid"] for r in rows], "roster_sha256": run3d.sha_file(a.roster),
                                                       "note": "finger-width sensitivity run: NOT production data"}, indent=1, default=list))
    cpus = run3d.cpus_of(a.cpus)[:a.workers]
    os.sched_setaffinity(0, set(cpus))
    ctx = mp.get_context("spawn")
    tasks, results = ctx.Queue(), ctx.Queue()
    qid_of = {r["qid"].replace(":", "_"): r["qid"] for r in rows}
    for r in rows:
        tasks.put((r["qid"].replace(":", "_"), [r], run3d.sha_file(P.SRC / "raw/roots" / f"{r['root_id']}.json")))
    procs = [ctx.Process(target=worker_v, args=(c, tasks, results, str(run), stages, 1200.0, True, a.variant), daemon=True) for c in cpus]
    for p in procs:
        tasks.put(None); p.start()
    led, t0 = open(run / "ledger.jsonl", "a"), time.time()
    for _ in rows:
        kind, uid, cpu, wall, cs, nq, js, zs = results.get()
        ok, why = run3d.validate_unit(run, uid, [qid_of[uid]], {"json_sha256": js, "npz_sha256": zs})
        led.write(json.dumps({"event": "unit_done", "unit": uid, "kind": kind, "valid": ok, "why": why, "cpu_s": cs, "json_sha256": js, "npz_sha256": zs}) + "\n")
        led.flush()
    for p in procs:
        p.join(timeout=60)
    print(json.dumps({"variant": a.variant, "queries": len(rows), "wall_s": round(time.time() - t0, 1)}))


def cmd_compare(a):
    def recs(d):
        out = {}
        for f in Path(d).glob("units/*.json"):
            for r in json.loads(f.read_text())["records"]:
                out[r["qid"]] = r
        return out
    s, t = recs(a.source), recs(a.alt)
    lab = lambda r: f"{r['label']}:{r.get('negative_kind') or r['status']}"  # noqa: E731
    tr, ex = Counter(), {}
    for q in sorted(set(s) & set(t)):
        k = f"{lab(s[q])} -> {lab(t[q])}"
        tr[k] += 1
        if s[q]["label"] != t[q]["label"]:
            ex.setdefault(k, []).append(q)
    by_src = Counter(lab(s[q]) for q in set(s) & set(t))
    chg = Counter(lab(s[q]) for q in set(s) & set(t) if s[q]["label"] != t[q]["label"])
    res = {"compared": sum(tr.values()), "label_changed": sum(chg.values()), "transitions": dict(sorted(tr.items())), "changed_qids": ex,
           "by_source_class": {k: {"n": n, "label_changed": chg[k]} for k, n in sorted(by_src.items())},
           "protocols": {"source": json.loads((Path(a.source) / "run_manifest.json").read_text())["protocol"][0],
                         "alt": json.loads((Path(a.alt) / "run_manifest.json").read_text())["protocol"][0]}}
    Path(a.out).write_text(json.dumps(res, indent=1)); print(json.dumps({k: v for k, v in res.items() if k != "changed_qids"}))


def cmd_summary(a):
    """Fold the finger-width evidence of DIR into finger_width_summary.json (Wilson 95% intervals on label-change rates)."""
    import report3d
    d = Path(a.dir)
    j = lambda n: json.loads((d / n).read_text())  # noqa: E731
    sm, sc = j("static_mock_v2.json")["summary"], j("static_cohort_sample.json")["summary"]
    tm, tc, h = j("compare_alt_vs_source.json"), j("compare_cohort_sample.json"), j("source_rerun_vs_mock_A.json")
    rep = {k: {x: j(f"replay_{k}.json").get(x) for x in ("replayed", "verified", "failures", "hold_identical")} for k in ("mock_alt", "cohort_alt")}
    aff_c = {x["qid"] for x in j("static_cohort_sample.json")["queries"] if x["pose_status_changes"]}
    out = {"complete": True, "protocols": {"source_exact": tm["protocols"]["source"], "alt_preshaped": tm["protocols"]["alt"]},
           "static_mock_v2": sm, "static_cohort_sample": sc, "harness_source_rerun_vs_mock_A": h,
           "targeted_selection": {k: (len(v) if isinstance(v, list) else v) for k, v in j("targeted_selection.json").items()},
           "targeted_comparison": {k: tm[k] for k in ("compared", "label_changed", "transitions", "by_source_class", "changed_qids")},
           "cohort_sample_comparison": {k: tc[k] for k in ("compared", "label_changed", "transitions", "by_source_class", "changed_qids")},
           "cohort_sample_screen_affected": len(aff_c),
           "label_change_rate_wilson95": {"targeted_mock_screen_affected": report3d.wilson(
                                              sum(1 for q in tm["changed_qids"].values() for x in q if x in set(j("targeted_selection.json")["screen_affected"])),
                                              len(j("targeted_selection.json")["screen_affected"])),
                                          "cohort_sample": report3d.wilson(tc["label_changed"], tc["compared"])},
           "alt_positive_replays": rep,
           "decision": ("NO_PROTOCOL_CHANGE: the alternate pre-shaped structure check changed no label in the targeted mock set or the cohort sample; "
                        "the frozen source-exact protocol stays the first-cohort protocol") if tm["label_changed"] == 0 and tc["label_changed"] == 0 else
                       "FOLLOW_ON_PROTOCOL_RECOMMENDED: labels differ between the identities; keep the frozen protocol for this cohort and register a "
                       "separate protocol for any change"}
    Path(a.out).write_text(json.dumps(out, indent=1, sort_keys=True)); print(json.dumps({k: out[k] for k in ("label_change_rate_wilson95", "decision")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("summary"); p.add_argument("dir"); p.add_argument("out")
    p = sp.add_parser("static"); p.add_argument("roster"); p.add_argument("qids"); p.add_argument("out")
    p.add_argument("--workers", type=int, default=12); p.add_argument("--cpus", default="0-11")
    p = sp.add_parser("run"); p.add_argument("roster"); p.add_argument("qids"); p.add_argument("run"); p.add_argument("--variant", required=True, choices=("alt", "source"))
    p.add_argument("--workers", type=int, default=12); p.add_argument("--cpus", default="0-11")
    p = sp.add_parser("compare"); p.add_argument("roster"); p.add_argument("source"); p.add_argument("alt"); p.add_argument("out")
    a = ap.parse_args()
    {"static": cmd_static, "run": cmd_run, "compare": cmd_compare, "summary": cmd_summary}[a.cmd](a)
