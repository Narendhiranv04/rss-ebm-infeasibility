#!/usr/bin/env python3
"""Q2_2D_SPILLOVER_V2 (user-approved 2026-11-08 plan, step 5): after the DRAIN of the laptop remainder run, hand every laptop-owned root that is
NOT yet packed to a second, separate gvlab scoped run.  A versioned layer ON TOP of spill2d.py (imported unchanged, never edited); the 2D teacher,
plan, policy, runner and audit are imported unchanged; nothing changes labels, seeds, task selection or the ADAPTIVE_FULL 8->32 policy.

Owner map V2 (frozen + hashed before the gvlab run starts; never overwritten):
  ORIGINAL  unchanged from V1 (laptop original run, packed)
  GVLAB     unchanged from V1 (gvlab spillover run, CPUs 0-11)
  LAPTOP    V1 laptop-owned roots PACKED in runs/remainder_laptop at the drain   (immutable from then on)
  GVLAB2    V1 laptop-owned roots NOT packed at the drain: PARTIAL (verified final parts copied, hash-checked) or UNTOUCHED
            -> gvlab runs/spillover2_gvlab, 4 workers on CPUs 14-17 (Q2 uses 0-13: 18 logical CPUs = the project ceiling; 18-23 stay free)
Union = exactly the 4,620 Q2 roots / 249,965 tasks / 238,160 teacher queries, zero overlap.  The final merge (4 disjoint sources) is a NEW
directory re-audited with launch_check; it runs when the user is back (merge-v2).

  laptop (only AFTER the remainder coordinator has exited):   spill2d_v2.py partition-v2 | stage-v2 | merge-v2 GVLAB_COPY GVLAB2_COPY
  gvlab:                                                       spill2d_v2.py server-run-v2 [--dry-run]
"""
import json, os, subprocess, sys
from collections import Counter
from pathlib import Path

HOST = "gvlab" if Path.home().name == "nav" else "laptop"
Q2S = (Path.home() / "approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/q2_spillover" if HOST == "gvlab" else
       Path("/home/naren/Desktop/rss-placement-ready-v2/gh-yadu-iiith/planar_2d/dataset_relocations_v1/q2_spillover"))
sys.path.insert(0, str(Q2S))
import spill2d as S  # noqa: E402  (unchanged V1 orchestrator: helpers, constants, classify, gates)

if os.environ.get("SPILL2D_V2_TEST"):                                     # TEST ONLY: point at a scratch copy (never set in production)
    _t = Path(os.environ["SPILL2D_V2_TEST"])
    S.REM, S.SPILL = _t / "remainder_laptop", _t
    S.LAUNCH = Path.home() / S.GLAUNCH
VERSION2 = "Q2_2D_SPILLOVER_V2_LAPTOP_DRAIN_TO_GVLAB2"
H1 = S.SPILL / "handoff"
H2 = S.SPILL / "handoff_v2"
MF2 = "HANDOFF_ROOT_OWNER_MANIFEST_V2.json"
V2CPUS, V2RES = [14, 15, 16, 17], [12, 13]
GH2 = f"{S.GSTAGE}/q2_spillover/handoff_v2"
GRUN2 = f"{S.GLAUNCH}/runs/spillover2_gvlab"
if os.environ.get("SPILL2D_V2_TEST"):                                     # TEST ONLY
    GH2, GRUN2 = str(Path(os.environ["SPILL2D_V2_TEST"]).relative_to(Path.home()) / "handoff_v2"), str(Path(os.environ["SPILL2D_V2_TEST"]).relative_to(Path.home()) / "run2")


def lock_free(run):
    import fcntl
    lk = run / ".lock"
    if not lk.exists():
        return True
    fd = os.open(lk, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(fd, fcntl.LOCK_UN); return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def partition_v2():
    """Laptop, after the drain: freeze owner map V2 from the V1 map + the stopped remainder run (read-only on every run dir)."""
    if not lock_free(S.REM):
        raise SystemExit("REFUSED partition-v2: the laptop remainder coordinator still holds its lock (DRAIN not finished)")
    st = S.jload(S.REM / "status.json", {})
    if not (st.get("final") and st.get("stop") is None) and not (S.SPILL / "state/ACK_REMAINDER_STOP").exists():
        raise SystemExit(f"REFUSED partition-v2: the remainder run did not end by a clean DRAIN (stop={st.get('stop')}); inspect, then touch state/ACK_REMAINDER_STOP")
    sums = {l.split()[1]: l.split()[0] for l in (H1 / "handoff_sha256sums").read_text().splitlines() if l.strip()}
    if S.sha(H1 / "HANDOFF_ROOT_OWNER_MANIFEST.json") != sums.get("HANDOFF_ROOT_OWNER_MANIFEST.json"):
        raise SystemExit("REFUSED partition-v2: the V1 owner map does not match its frozen checksum")
    own1 = json.loads((H1 / "HANDOFF_ROOT_OWNER_MANIFEST.json").read_text())
    tasks, cls, snap, bad = S.classify(run=S.REM, L=S.LAUNCH)              # classes relative to the REMAINDER run dir
    if bad:
        raise SystemExit(f"REFUSED partition-v2: remainder classification failures {bad[:3]}")
    stray = [r for r, c in cls.items() if c != "UNTOUCHED" and own1["owner"][r] != "LAPTOP"]
    if stray:
        raise SystemExit(f"REFUSED partition-v2: the remainder run touched roots it does not own: {stray[:3]}")
    own, cls2, snap2 = {}, {}, {}
    for r, o in own1["owner"].items():
        if o != "LAPTOP":
            own[r] = o
        elif cls[r] == "PACKED":
            own[r], cls2[r], snap2[r] = "LAPTOP", "PACKED", snap[r]
        else:
            own[r], cls2[r] = "GVLAB2", cls[r]
            if cls[r] == "PARTIAL":
                snap2[r] = snap[r]
    ids = {r: [t["trial_id"] for t in tasks[r][1]] for r in tasks}
    if ids != own1["trial_ids"]:
        raise SystemExit("REFUSED partition-v2: trial ids differ from the V1 owner map")
    allids = [i for r in sorted(ids) for i in ids[r]]
    teach = sum(1 for r in tasks for t in tasks[r][1] if not t["static_ineligible"])
    if set(own) != set(tasks) or len(own) != S.EXPECT["roots"] or len(allids) != len(set(allids)) or len(allids) != S.EXPECT["tasks"] or \
            teach != S.EXPECT["teacher_queries"]:
        raise SystemExit("REFUSED partition-v2: owner union is not the exact Q2 set")
    names = ("ORIGINAL", "GVLAB", "LAPTOP", "GVLAB2")
    man = {"version": VERSION2, "utc": S.utc(), "v1_owner_manifest_sha256": S.sha(H1 / "HANDOFF_ROOT_OWNER_MANIFEST.json"), "owner": dict(sorted(own.items())),
           "class_v2": cls2, "snapshot_sha256_v2": snap2, "trial_ids": ids, "remainder_final_status": {k: st.get(k) for k in ("updated", "final", "stop", "roots_packed")},
           "counts": {o: sum(v == o for v in own.values()) for o in names},
           "teacher_queries_by_owner": {o: sum(1 for r, v in own.items() if v == o for t in tasks[r][1] if not t["static_ineligible"]) for o in names},
           "gvlab2_execution": {"worker_cpus": V2CPUS, "reserved_cpus": V2RES, "workers": len(V2CPUS), "run_dir": "runs/spillover2_gvlab"},
           "invariants": {"roots": len(own), "tasks": len(allids), "teacher_queries": teach, "duplicates": 0,
                          "v1_original_and_gvlab_unchanged": all(own[r] == o for r, o in own1["owner"].items() if o in ("ORIGINAL", "GVLAB")),
                          "gvlab2_never_packed_at_drain": all(cls2[r] in ("PARTIAL", "UNTOUCHED") for r, v in own.items() if v == "GVLAB2")}}
    H2.mkdir(parents=True, exist_ok=True)
    if (H2 / MF2).exists():
        raise SystemExit("REFUSED partition-v2: an owner map V2 already exists (versioned handoffs only; never overwrite)")
    S.atomic(H2 / MF2, json.dumps(man, sort_keys=True))
    msha = S.sha(H2 / MF2)
    rs = sorted(r for r, v in own.items() if v == "GVLAB2")
    S.atomic(H2 / "allowlist_gvlab2.json", json.dumps({"host": "GVLAB2", "owner_manifest_v2_sha256": msha, "roots": rs, "partial": sorted(r for r in rs if cls2[r] == "PARTIAL"),
                                                       "trial_ids": {r: ids[r] for r in rs}}, sort_keys=True))
    S.atomic(H2 / "handoff_sha256sums", "".join(f"{S.sha(H2 / f)}  {f}\n" for f in sorted(os.listdir(H2)) if f != "handoff_sha256sums"))
    print(json.dumps({k: man[k] for k in ("counts", "teacher_queries_by_owner", "invariants")}))


def stage_v2():
    """Laptop: copy handoff_v2 and the PARTIAL parts of GVLAB2 roots to gvlab (rsync -a; verified on gvlab by server-run-v2 before any work)."""
    allow = json.loads((H2 / "allowlist_gvlab2.json").read_text())
    if allow["owner_manifest_v2_sha256"] != S.sha(H2 / MF2):
        raise SystemExit("REFUSED stage-v2: allowlist does not reference owner map V2")
    for cmd in (["ssh", "gvlab", f"mkdir -p {GH2} {GRUN2}/parts"], ["rsync", "-a", f"{H2}/", f"gvlab:{GH2}/"]):
        subprocess.run(cmd, check=True)
    for r in allow["partial"]:
        subprocess.run(["rsync", "-a", f"{S.REM}/parts/{r}/", f"gvlab:{GRUN2}/parts/{r}/"], check=True)
    print(json.dumps({"staged": True, "gvlab2_roots": len(allow["roots"]), "partial_roots_copied": len(allow["partial"])}))


def other_project_processes():
    """Project compute on gvlab other than (a) the V1 SPILLOVER server-run (allowed alongside) and (b) this process's own launch chain."""
    out = []
    for p in S.project_processes():
        c = S.cmdline(p["pid"])                                           # full command line (project_processes truncates to 140 chars)
        if "spill2d.py server-run" in c and "--scope SPILLOVER" in c and "spill2d_v2" not in c:
            continue
        out.append(p)
    return out


def server_run_v2(dry):
    L = Path.home() / S.GLAUNCH
    LP, R, K, T = S.code2d(L.parent / "code_v2")
    S.server_profile(R)
    pol, bad = S.check_frozen(L, T)
    if bad:
        raise SystemExit(f"REFUSED server-run-v2: frozen Q2 identity differs: {bad}")
    stage = json.loads((Path.home() / S.GSTAGE / "q2_spillover/source_manifest.json").read_text())
    base = L.parents[1]
    drift = [f for f, h in stage["files"].items() if not (base / f).exists() or S.sha(base / f) != h]
    if drift:
        raise SystemExit(f"REFUSED server-run-v2: staged 2D source differs from the laptop manifest: {drift[:3]}")
    C = R.cfg_of(L)
    bad = S.check_server_profile(C)
    if bad:
        raise SystemExit(f"REFUSED server-run-v2: server execution profile differs: {bad}")
    g = S.gate3d(require_post3d=True, host_checks=False, names=tuple(n for n in S.PROJECT_PROCS if n != "spill2d.py server-run"))
    procs = other_project_processes()
    if not g["pass"] or procs:
        raise SystemExit(f"REFUSED server-run-v2 (gate): {g['reasons']} {[p['cmd'][:60] for p in procs][:3]}")
    import psutil
    busy = psutil.cpu_percent(interval=2.0, percpu=True)
    facts = {"cpu14_17_busy_pct": [busy[c] for c in V2CPUS], "cpu18_23_busy_pct": [busy[c] for c in range(18, 24)],
             "mem_avail_gb": round(psutil.virtual_memory().available / 2 ** 30, 1), "disk_free_gb": round(psutil.disk_usage(str(Path.home())).free / 2 ** 30, 1),
             "coretemp_c": R.read_temp(R.sensors()) if R.sensors() else None}
    why = [k for k, ok in (("CPUs 14-17 busy with other work (> 25%)", max(facts["cpu14_17_busy_pct"]) <= 25), ("coretemp unreadable", facts["coretemp_c"] is not None),
                           ("coretemp >= 80 C at start", (facts["coretemp_c"] or 99) < 80), ("available memory < 8 GB", facts["mem_avail_gb"] >= 8),
                           ("free disk < 20 GB", facts["disk_free_gb"] >= 20)) if not ok]
    h2 = Path.home() / GH2
    if dry and not (h2 / MF2).exists():                                   # pre-drain check: host + identity gates only
        print(json.dumps({"dry_run": True, "handoff_v2": "ABSENT (pre-drain)", "facts": facts, "refusals": why}), flush=True)
        return
    sums = {l.split()[1]: l.split()[0] for l in (h2 / "handoff_sha256sums").read_text().splitlines() if l.strip()}
    man = json.loads((h2 / MF2).read_text())
    allow = json.loads((h2 / "allowlist_gvlab2.json").read_text())
    if any(S.sha(h2 / f) != h for f, h in sums.items()):
        why.append("handoff_v2 files do not match their checksums")
    if allow["owner_manifest_v2_sha256"] != S.sha(h2 / MF2) or sorted(allow["roots"]) != sorted(r for r, v in man["owner"].items() if v == "GVLAB2"):
        why.append("allowlist is not the frozen GVLAB2 share")
    tasks = S.q2_tasks(L, LP, R)
    out = Path.home() / GRUN2
    for r in allow["roots"]:
        if [t["trial_id"] for t in tasks[r][1]] != allow["trial_ids"][r]:
            why.append(f"trial ids of {r} differ from the frozen plan"); break
        if (out / "records" / f"{r}.json").exists():
            continue                                                      # packed by an earlier v2 session (resume)
        d = out / "parts" / r
        if man["class_v2"][r] == "PARTIAL" and not (out / "run_manifest.json").exists():
            for rel, h in man["snapshot_sha256_v2"][r].items():
                if not (d / Path(rel).name).exists() or S.sha(d / Path(rel).name) != h:
                    why.append(f"copied partial part of {r} does not match the drain snapshot: {rel}"); break
        if man["class_v2"][r] == "UNTOUCHED" and d.exists() and any(d.iterdir()) and not (out / "run_manifest.json").exists():
            why.append(f"{r} is UNTOUCHED in the owner map but has parts before the first v2 session")
    print(json.dumps({"dry_run": dry, "roots": len(allow["roots"]), "partial": len(allow["partial"]), "facts": facts, "refusals": why[:6]}), flush=True)
    if why:
        raise SystemExit("REFUSED server-run-v2: " + "; ".join(why[:6]))
    if dry:
        return
    sel = [tasks[r] for r in sorted(allow["roots"], key=lambda r: S.H("order", r))]
    res = R.run(L, out, sel, pol["policy"], len(V2CPUS), V2CPUS, V2RES, "Q2_SPILLOVER2_GVLAB")
    print(json.dumps({k: v for k, v in res.items() if k not in ("events", "safety")}, default=str))


def merge_v2(gvlab_copy, gvlab2_copy):
    """Laptop, when the user is back and BOTH gvlab runs are complete: NEW canonical Q2 directory from the 4 disjoint sources of owner map V2,
    then launch_check.audit (re-certifies every witness).  Mirrors spill2d.merge (V1) with the GVLAB2 source added; never overwrites."""
    import shutil
    LP, R, K, T = S.code2d(S.CODE)
    own = json.loads((H2 / MF2).read_text())
    if own["v1_owner_manifest_sha256"] != S.sha(H1 / "HANDOFF_ROOT_OWNER_MANIFEST.json"):
        raise SystemExit("REFUSED merge-v2: V1 owner map changed since V2 was frozen")
    snap1 = json.loads((H1 / "HANDOFF_ROOT_OWNER_MANIFEST.json").read_text())["snapshot_sha256"]
    src = {"ORIGINAL": S.ORIG, "LAPTOP": S.REM, "GVLAB": Path(gvlab_copy), "GVLAB2": Path(gvlab2_copy)}
    merged = S.SPILL / "merged" / "q2_canonical_v2"
    if merged.exists():
        raise SystemExit("REFUSED merge-v2: the canonical merged directory exists (never overwrite)")
    stray = {o: sorted(f.stem for f in (src[o] / "records").glob("*.json") if own["owner"].get(f.stem) != o) for o in ("LAPTOP", "GVLAB", "GVLAB2")}
    if any(stray.values()):
        raise SystemExit(f"REFUSED merge-v2: a scoped run packed roots it does not own: { {k: v[:3] for k, v in stray.items()} }")
    tmp = merged.with_name(".q2_canonical_v2.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for d in ("records", "modes", "witnesses"):
        (tmp / d).mkdir(parents=True)
    shutil.copyfile(S.LAUNCH / "split_manifest.json", S.SPILL / "split_manifest.json")
    prov, seen = {}, set()
    for rid, o in sorted(own["owner"].items()):
        frozen = snap1.get(rid) if o == "ORIGINAL" else own["snapshot_sha256_v2"].get(rid) if o == "LAPTOP" else None
        if frozen and any(S.sha(src[o] / rel) != h for rel, h in frozen.items()):
            raise SystemExit(f"REFUSED merge-v2: {o} root {rid} changed since its frozen snapshot")
        rec = src[o] / "records" / f"{rid}.json"
        if not rec.exists():
            raise SystemExit(f"REFUSED merge-v2: {rid} owned by {o} is not packed in its run dir")
        ids = [q["trial_id"] for q in json.loads(rec.read_text())["queries"]]
        if ids != own["trial_ids"][rid] or seen & set(ids):
            raise SystemExit(f"REFUSED merge-v2: trial ids of {rid} differ from the owner map or collide")
        seen |= set(ids)
        for d, ext in (("records", "json"), ("modes", "npz"), ("witnesses", "npz")):
            f = src[o] / d / f"{rid}.{ext}"
            if f.exists():
                shutil.copy2(f, tmp / d / f.name)
                assert S.sha(f) == S.sha(tmp / d / f.name)
        prov[rid] = {"owner": o, "source_dir": str(src[o])}
    if len(seen) != S.EXPECT["tasks"]:
        raise SystemExit(f"REFUSED merge-v2: {len(seen)} trial ids != {S.EXPECT['tasks']}")
    pol = json.loads((S.LAUNCH / "configs/production_policy.json").read_text())
    S.atomic(tmp / "run_manifest.json", json.dumps({"scope": "Q2_MERGED_CANONICAL_V2", "policy": pol["policy"], "budget_hash": T.budget_hash(pol["policy"]),
                                                    "pair_protocol_id": T.pair_protocol()[0], "queries": len(seen), "roots": len(prov),
                                                    "owner_manifest_v2_sha256": S.sha(H2 / MF2)}, sort_keys=True))
    S.atomic(tmp / "provenance.json", json.dumps(prov, sort_keys=True))
    os.replace(tmp, merged)
    a = K.audit(merged)
    teach = sum(1 for _, q, _, _ in K.load(merged) if q["eligible"])
    rep = {"version": VERSION2, "utc": S.utc(), "merged_dir": str(merged), "roots": len(prov), "tasks": len(seen), "teacher_queries": teach,
           "by_owner": dict(Counter(v["owner"] for v in prov.values())), "audit": a["verification"], "summary": a["summary"],
           "run_fingerprint": a["fingerprint"]["run_fingerprint"],
           "pass": a["verification"]["failed"] == 0 and teach == S.EXPECT["teacher_queries"] and len(seen) == S.EXPECT["tasks"]}
    S.atomic(S.SPILL / "MERGED_Q2_V2_AUDIT.json", json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps({k: rep[k] for k in ("pass", "roots", "tasks", "teacher_queries", "by_owner")}))


if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "partition-v2":
        partition_v2()
    elif a[0] == "stage-v2":
        stage_v2()
    elif a[0] == "merge-v2":
        merge_v2(a[1], a[2])
    elif a[0] == "server-run-v2":
        server_run_v2("--dry-run" in a)
    else:
        raise SystemExit(__doc__)
