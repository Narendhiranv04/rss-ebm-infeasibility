#!/usr/bin/env python3
"""Q2_2D_SPILLOVER_V2 (user 2026-11-09 09:30 IST: "I want it all by 5 pm tonight, push every aspect in the server"): a versioned layer ON TOP of
spill2d.py (imported unchanged, never edited).  The 2D teacher, plan, policy, runner and audit are imported unchanged; nothing changes labels,
seeds, task selection or the ADAPTIVE_FULL 8->32 policy.  Execution profiles (CPUs, sensors) cannot change labels.

One cut: DRAIN the laptop remainder run AND the gvlab V1 spillover run (in-flight queries finish; packed roots are kept), classify each run on its
own host (classify-export), then freeze owner map V2 (never overwritten):
  ORIGINAL  unchanged (V1)                     GVLAB   V1 gvlab-owned roots PACKED at the drain     LAPTOP  V1 laptop-owned roots PACKED at the drain
  GVLAB2    gvlab PARTIAL roots (+ verified parts) + a share of UNTOUCHED  -> gvlab  runs/spillover2_gvlab   16 workers CPUs 0-11,14-17 (coord 12-13; <= 18)
  LAPTOP2   laptop PARTIAL roots (+ parts) + a share of UNTOUCHED          -> laptop q2_spillover/runs/remainder2_laptop (laptop config)
  BLACKWELL UNTOUCHED only                                                 -> Blackwell runs/spillover_blackwell  16 workers E-cores 8-23 (coord 7;
            P-cores 0-6 idle >= 4 free); a yield watchdog PAUSEs it while a collaborator (yadu generator / sajid queue) is active
Union = exactly 4,620 roots / 249,965 tasks / 238,160 teacher queries, zero overlap.  merge-v2 builds a NEW canonical directory and re-audits it.

  any host : spill2d_v2.py classify-export RUN_DIR OWNER OUT.json         (OWNER = LAPTOP for the remainder run, GVLAB for the V1 gvlab run)
  laptop   : spill2d_v2.py partition-v2 CLS_LAPTOP CLS_GVLAB SHARE_GVLAB2 SHARE_BLACKWELL SHARE_LAPTOP2
  per host : spill2d_v2.py copy-partials SHARE      (LAPTOP2 on the laptop, GVLAB2 on gvlab; hash-verified, never writes the drained runs)
  per host : spill2d_v2.py run SHARE [--dry-run]    (GVLAB2 | BLACKWELL | LAPTOP2)
  Blackwell: spill2d_v2.py canary                    (16 laptop TRAIN roots re-run; PASS_EXACT required before any BLACKWELL bulk run)
  Blackwell: spill2d_v2.py merge-v2 DIR_ORIGINAL DIR_LAPTOP DIR_GVLAB DIR_GVLAB2 DIR_BLACKWELL DIR_LAPTOP2
"""
import json, os, shutil, sys
from collections import Counter, defaultdict
from pathlib import Path

HOST = {"nav": "gvlab", "user1": "blackwell"}.get(Path.home().name, "laptop")
Q2S = {"gvlab": Path.home() / "approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/q2_spillover",
       "blackwell": Path.home() / "approachb_3d_learning/q2bw/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/q2_spillover",
       "laptop": Path("/home/naren/Desktop/rss-placement-ready-v2/gh-yadu-iiith/planar_2d/dataset_relocations_v1/q2_spillover")}[HOST]
sys.path.insert(0, str(Q2S))
import spill2d as S  # noqa: E402

VERSION2 = "Q2_2D_SPILLOVER_V2_THREE_HOST_CUT"
SPILL = Q2S                                                  # this host's q2_spillover dir
H1, H2, MF2 = SPILL / "handoff", SPILL / "handoff_v2", "HANDOFF_ROOT_OWNER_MANIFEST_V2.json"
LDIR = {"gvlab": Path.home() / S.GLAUNCH, "blackwell": Q2S.parent / "launch_ready_2d_v2_gvlab", "laptop": S.LAUNCH}[HOST]
SHARES = {"GVLAB2": {"host": "gvlab", "run": "runs/spillover2_gvlab", "cpus": list(range(12)) + [14, 15, 16, 17], "reserved": [12, 13]},
          "BLACKWELL": {"host": "blackwell", "run": "runs/spillover_blackwell", "cpus": [0, 1, 2] + list(range(8, 24)), "reserved": [7]},
          "LAPTOP2": {"host": "laptop", "run": None, "cpus": None, "reserved": None}}       # laptop: its own frozen launch config
NEW = ("GVLAB2", "BLACKWELL", "LAPTOP2")
COEXIST = "--coexist" in sys.argv        # Blackwell only: E-cores 10-23, start despite a pending collaborator queue; q2bw_coexist.py guards


def run_dir(share):
    return S.SPILL / "runs" / "remainder2_laptop" if share == "LAPTOP2" else LDIR / SHARES[share]["run"]


def lock_free(run):
    import fcntl
    lk = Path(run) / ".lock"
    if not lk.exists():
        return True
    fd = os.open(lk, os.O_RDONLY)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(fd, fcntl.LOCK_UN); return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def classify_export(run, owner, out):
    """Read-only: classes + snapshots of the roots OWNER owns in V1, relative to a DRAINED run dir (refuses while its coordinator holds the lock)."""
    run, out = Path(run).resolve(), Path(out).resolve()                  # the 2D code chdirs: resolve before importing it
    if not lock_free(run):
        raise SystemExit(f"REFUSED classify-export: {run} coordinator still holds its lock (DRAIN not finished)")
    st = S.jload(run / "status.json", {})
    if not (st.get("final") and st.get("stop") is None):
        raise SystemExit(f"REFUSED classify-export: {run} did not end by a clean DRAIN (stop={st.get('stop')})")
    own1 = json.loads((H1 / "HANDOFF_ROOT_OWNER_MANIFEST.json").read_text())
    tasks, cls, snap, bad = S.classify(run=run, L=LDIR)
    if bad:
        raise SystemExit(f"REFUSED classify-export: classification failures {bad[:3]}")
    stray = [r for r, c in cls.items() if c != "UNTOUCHED" and own1["owner"][r] != owner]
    if stray:
        raise SystemExit(f"REFUSED classify-export: the run touched roots it does not own: {stray[:3]}")
    mine = [r for r, o in own1["owner"].items() if o == owner]
    doc = {"version": VERSION2, "utc": S.utc(), "host": HOST, "run": str(run), "owner_v1": owner, "v1_owner_manifest_sha256": S.sha(H1 / "HANDOFF_ROOT_OWNER_MANIFEST.json"),
           "final_status": {k: st.get(k) for k in ("updated", "final", "stop", "roots_packed")}, "class": {r: cls[r] for r in mine},
           "snapshot_sha256": {r: snap[r] for r in mine if r in snap}, "class_counts": dict(Counter(cls[r] for r in mine))}
    S.atomic(Path(out), json.dumps(doc, sort_keys=True))
    print(json.dumps({"owner": owner, "class_counts": doc["class_counts"]}))


def partition_v2(cls_lap_f, cls_gv_f, sg, sb, sl):
    """Laptop: freeze owner map V2 from the V1 map and the two drained-run classifications.  Shares are cost fractions of the UNTOUCHED roots."""
    own1 = json.loads((H1 / "HANDOFF_ROOT_OWNER_MANIFEST.json").read_text())
    v1sha = S.sha(H1 / "HANDOFF_ROOT_OWNER_MANIFEST.json")
    cls_lap_f, cls_gv_f = Path(cls_lap_f).resolve(), Path(cls_gv_f).resolve()
    cl, cg = json.loads(cls_lap_f.read_text()), json.loads(cls_gv_f.read_text())
    if cl["owner_v1"] != "LAPTOP" or cg["owner_v1"] != "GVLAB" or cl["v1_owner_manifest_sha256"] != v1sha or cg["v1_owner_manifest_sha256"] != v1sha:
        raise SystemExit("REFUSED partition-v2: classification files do not match the V1 owner map / owners")
    LP, R, K, T = S.code2d(LDIR.parent / "code_v2")
    tasks = S.q2_tasks(LDIR, LP, R)
    own, cls2, snap2, unt = {}, {}, {}, []
    for r, o in own1["owner"].items():
        if o == "ORIGINAL":
            own[r] = o; continue
        src = cl if o == "LAPTOP" else cg
        c = src["class"][r]
        cls2[r] = c
        if c == "PACKED":
            own[r], snap2[r] = o, src["snapshot_sha256"][r]
        elif c == "PARTIAL":
            own[r], snap2[r] = ("LAPTOP2" if o == "LAPTOP" else "GVLAB2"), src["snapshot_sha256"][r]
        else:
            unt.append(r)
    tot = sg + sb + sl
    frac = {"GVLAB2": sg / tot, "BLACKWELL": sb / tot, "LAPTOP2": sl / tot}
    cost = {r: S.root_cost(tasks[r][1], LDIR) for r in unt}
    by = defaultdict(list)
    for r in unt:
        by[(tasks[r][0]["split"], tasks[r][0]["verified_class"])].append(r)
    for k in sorted(by):                                     # per split x class: keep each host's mix; deterministic hash order
        acc = {h: 0.0 for h in NEW}
        for r in sorted(by[k], key=lambda r: S.H("assign_v2", r)):
            h = min((h for h in NEW if frac[h] > 0), key=lambda h: (acc[h] + cost[r]) / frac[h])
            own[r] = h; acc[h] += cost[r]
    ids = {r: [t["trial_id"] for t in tasks[r][1]] for r in tasks}
    if ids != own1["trial_ids"]:
        raise SystemExit("REFUSED partition-v2: trial ids differ from the V1 owner map")
    allids = [i for r in sorted(ids) for i in ids[r]]
    teach = sum(1 for r in tasks for t in tasks[r][1] if not t["static_ineligible"])
    if set(own) != set(tasks) or len(own) != S.EXPECT["roots"] or len(allids) != len(set(allids)) or len(allids) != S.EXPECT["tasks"] or \
            teach != S.EXPECT["teacher_queries"]:
        raise SystemExit("REFUSED partition-v2: owner union is not the exact Q2 set")
    names = ("ORIGINAL", "LAPTOP", "GVLAB", *NEW)
    man = {"version": VERSION2, "utc": S.utc(), "v1_owner_manifest_sha256": v1sha, "owner": dict(sorted(own.items())), "class_v2": cls2,
           "snapshot_sha256_v2": snap2, "trial_ids": ids, "share_fractions_untouched_cost": frac,
           "drain_status": {"LAPTOP": cl["final_status"], "GVLAB": cg["final_status"]},
           "classification_sha256": {"LAPTOP": S.sha(cls_lap_f), "GVLAB": S.sha(cls_gv_f)},
           "counts": {o: sum(v == o for v in own.values()) for o in names},
           "teacher_queries_by_owner": {o: sum(1 for r, v in own.items() if v == o for t in tasks[r][1] if not t["static_ineligible"]) for o in names},
           "execution": SHARES,
           "blackwell_standard": {"decision": "LABEL_EQUIVALENT_NOT_BITWISE (user-approved 2026-11-09 ~10:34 IST, deadline 17:00)",
                                  "canary": "16 laptop TRAIN roots / 848 trials: labels + negative kinds 848/848 identical, per-trial fingerprints 206/848 "
                                            "identical, Blackwell audit 21,547 checks 0 failed (every witness re-certifies); cause: Core Ultra 9 285K has no "
                                            "AVX-512 (laptop / gvlab do) -> different floating-point paths", "report": "PORTABILITY_CANARY_REPORT_BLACKWELL.json"},
           "invariants": {"roots": len(own), "tasks": len(allids), "teacher_queries": teach, "duplicates": 0,
                          "original_unchanged": all(own[r] == "ORIGINAL" for r, o in own1["owner"].items() if o == "ORIGINAL"),
                          "blackwell_only_untouched": all(r in unt for r, v in own.items() if v == "BLACKWELL"),
                          "partials_stay_on_their_host": all(own[r] in ("LAPTOP2", "GVLAB2") for r, c in cls2.items() if c == "PARTIAL")}}
    H2.mkdir(parents=True, exist_ok=True)
    if (H2 / MF2).exists():
        raise SystemExit("REFUSED partition-v2: an owner map V2 already exists (versioned handoffs only; never overwrite)")
    S.atomic(H2 / MF2, json.dumps(man, sort_keys=True))
    msha = S.sha(H2 / MF2)
    for h in NEW:
        rs = sorted(r for r, v in own.items() if v == h)
        S.atomic(H2 / f"allowlist_{h.lower()}.json", json.dumps({"share": h, "owner_manifest_v2_sha256": msha, "roots": rs,
                                                                "partial": sorted(r for r in rs if cls2.get(r) == "PARTIAL"), "trial_ids": {r: ids[r] for r in rs}}, sort_keys=True))
    S.atomic(H2 / "handoff_sha256sums", "".join(f"{S.sha(H2 / f)}  {f}\n" for f in sorted(os.listdir(H2)) if f != "handoff_sha256sums"))
    print(json.dumps({k: man[k] for k in ("counts", "teacher_queries_by_owner", "invariants")}))


def load_v2(share):
    sums = {l.split()[1]: l.split()[0] for l in (H2 / "handoff_sha256sums").read_text().splitlines() if l.strip()}
    bad = [f for f, h in sums.items() if S.sha(H2 / f) != h]
    man = json.loads((H2 / MF2).read_text())
    allow = json.loads((H2 / f"allowlist_{share.lower()}.json").read_text())
    if bad or allow["owner_manifest_v2_sha256"] != S.sha(H2 / MF2) or sorted(allow["roots"]) != sorted(r for r, v in man["owner"].items() if v == share):
        raise SystemExit(f"REFUSED: handoff_v2 checksums / {share} allowlist do not match the frozen owner map V2 {bad[:2]}")
    return man, allow


def copy_partials(share):
    """Copy the share's PARTIAL roots' parts from the drained V1 run on THIS host into the share's run dir, hash-verified against the V2 snapshot."""
    man, allow = load_v2(share)
    src = {"LAPTOP2": S.REM, "GVLAB2": LDIR / "runs" / "spillover_gvlab"}[share]
    dst = run_dir(share)
    n = 0
    for r in allow["partial"]:
        if (dst / "parts" / r).exists() or (dst / "records" / f"{r}.json").exists():
            continue
        tmp = dst / "parts" / f".{r}.copy"
        shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
        for rel, h in man["snapshot_sha256_v2"][r].items():
            if S.sha(src / rel) != h:
                raise SystemExit(f"REFUSED copy-partials: drained partial file changed since the V2 snapshot: {rel}")
            shutil.copy2(src / rel, tmp / Path(rel).name)
            if S.sha(tmp / Path(rel).name) != h:
                raise SystemExit(f"copy of {rel} does not re-hash")
        os.replace(tmp, dst / "parts" / r); n += 1
    print(json.dumps({"share": share, "partial_roots_copied": n, "dst": str(dst)}))


def host_gate(R, cpus):
    import psutil
    busy = psutil.cpu_percent(interval=2.0, percpu=True)
    f = {"cpus_busy_pct_max": max(busy[c] for c in cpus), "idle_cpus": sum(1 for b in busy if b < 25),
         "mem_avail_gb": round(psutil.virtual_memory().available / 2 ** 30, 1), "disk_free_gb": round(psutil.disk_usage(str(Path.home())).free / 2 ** 30, 1),
         "temp_c": R.read_temp(R.sensors()) if R.sensors() else None}
    why = [k for k, ok in (("worker CPUs busy with other work (> 25%)", f["cpus_busy_pct_max"] <= 25), ("temperature sensor unreadable", f["temp_c"] is not None),
                           ("temperature >= 80 C at start", (f["temp_c"] or 99) < 80), ("available memory < 8 GB", f["mem_avail_gb"] >= 8),
                           ("free disk < 20 GB", f["disk_free_gb"] >= 20)) if not ok]
    return f, why


def identity(R, T):
    pol, bad = S.check_frozen(LDIR, T)
    if HOST != "laptop":
        stage = json.loads((SPILL / "source_manifest.json").read_text())
        base = LDIR.parents[1]
        bad += [f"source {f}" for f, h in stage["files"].items() if not (base / f).exists() or S.sha(base / f) != h][:3]
        bad += [f"server copy {f}" for f in ("query_plan.jsonl.gz", "plan_hashes.json", "split_manifest.json", "configs/production_policy.json")
                if S.sha(LDIR / f) != stage["files"].get(f"dataset_relocations_v1/launch_ready_2d_v2/{f}")]
    return pol, bad


def blackwell_blockers():
    sys.path.insert(0, str(Path.home() / "approachb_3d_learning/physics"))
    os.environ.setdefault("BWL_REPO", str(Path.home() / "approachb_3d_learning/teacher3d/shelf_6x6_open_sides"))
    import bwlphys as B
    f, _ = B.guard({"temp": 999, "mem_gb": 0, "swap_growth_mb": 1e9, "swap0": 0, "host": "blackwell"})
    q = f["queue_pending"]
    return f["yield_processes"] + sum(v for v in q.values() if isinstance(v, int)) + (1 if "ERROR" in q else 0)


def run_share(share, dry):
    if SHARES[share]["host"] != HOST:
        raise SystemExit(f"REFUSED: {share} runs on {SHARES[share]['host']}, this is {HOST}")
    LP, R, K, T = S.code2d(LDIR.parent / "code_v2")
    if HOST != "laptop":
        S.server_profile(R)                                  # coretemp 'Package id 0' (gvlab Xeon 5317 / Blackwell Core Ultra 9 285K)
    pol, why = identity(R, T)
    C = R.cfg_of(LDIR)
    cpus, res = (SHARES[share]["cpus"], SHARES[share]["reserved"]) if HOST != "laptop" else (C["execution"]["worker_cpus"], C["execution"]["reserved_cpus"])
    if share == "BLACKWELL" and COEXIST:                     # user-approved 2026-11-09 ~10:42 IST: run alongside sajid's jobs on E-cores only
        cpus = list(range(10, 24))
    if "--cpus" in sys.argv:                                  # execution-only override (thermal scaling); labels cannot depend on it
        a, b = sys.argv[sys.argv.index("--cpus") + 1].split("-")
        cpus = list(range(int(a), int(b) + 1))
    workers = len(cpus) if HOST != "laptop" else C["execution"]["workers"]
    if HOST == "gvlab":
        why += S.gate3d(require_post3d=True, host_checks=False)["reasons"]
    if HOST == "blackwell" and not COEXIST and blackwell_blockers():
        why.append("collaborator active on Blackwell (yadu generator / sajid queue)")
    f, w = host_gate(R, cpus) if HOST != "laptop" else ({}, [])
    why += w
    man, allow = load_v2(share)
    tasks = S.q2_tasks(LDIR, LP, R)
    out = run_dir(share)
    for r in allow["roots"]:
        if [t["trial_id"] for t in tasks[r][1]] != allow["trial_ids"][r]:
            why.append(f"trial ids of {r} differ from the frozen plan"); break
        d = out / "parts" / r
        if (out / "run_manifest.json").exists() or (out / "records" / f"{r}.json").exists():
            continue
        if r in allow["partial"]:
            if any(not (d / Path(rel).name).exists() or S.sha(d / Path(rel).name) != h for rel, h in man["snapshot_sha256_v2"][r].items()):
                why.append(f"partial parts of {r} not copied / do not match the V2 snapshot"); break
        elif d.exists() and any(d.iterdir()):
            why.append(f"{r} is UNTOUCHED in V2 but has parts before the first session"); break
    print(json.dumps({"share": share, "host": HOST, "dry_run": dry, "roots": len(allow["roots"]), "partial": len(allow["partial"]), "workers": workers,
                      "cpus": cpus, "reserved": res, "facts": f, "refusals": why[:6]}), flush=True)
    if why:
        raise SystemExit("REFUSED run: " + "; ".join(why[:6]))
    if dry:
        return
    sel = [tasks[r] for r in sorted(allow["roots"], key=lambda r: S.H("order", r))]
    del tasks
    r = R.run(LDIR, out, sel, pol["policy"], workers, cpus, res, f"Q2_V2_{share}")
    print(json.dumps({k: v for k, v in r.items() if k not in ("events", "safety")}, default=str))


def canary():
    """Blackwell: re-run the 16 byte-verified laptop TRAIN roots of the gvlab portability canary; audit; compare per-trial fingerprints with the
    laptop reference (runs/canary_laptop_ref).  PASS_EXACT or nothing."""
    LP, R, K, T = S.code2d(LDIR.parent / "code_v2")
    S.server_profile(R)
    pol, why = identity(R, T)
    if blackwell_blockers():
        why.append("collaborator active on Blackwell")
    f, w = host_gate(R, SHARES["BLACKWELL"]["cpus"])
    why += w
    allow = json.loads((SPILL / "canary/allowlist_canary.json").read_text())
    tasks = S.q2_tasks(LDIR, LP, R)
    why += [f"trial ids differ: {r}" for r in allow["roots"] if [t["trial_id"] for t in tasks[r][1]] != allow["trial_ids"][r]][:1]
    print(json.dumps({"canary_roots": len(allow["roots"]), "facts": f, "refusals": why}), flush=True)
    if why:
        raise SystemExit("REFUSED canary: " + "; ".join(why))
    out = LDIR / "runs" / "canary_blackwell"
    if out.exists():
        raise SystemExit("REFUSED canary: runs/canary_blackwell exists (never overwrite)")
    sel = [tasks[r] for r in sorted(allow["roots"], key=lambda r: S.H("order", r))]
    res = R.run(LDIR, out, sel, pol["policy"], 16, SHARES["BLACKWELL"]["cpus"], [7], "Q2_CANARY_BLACKWELL")
    print(json.dumps({k: res.get(k) for k in ("stop", "wall_s", "finished_now", "interrupted", "not_started")}, default=str))


def canary_check():
    """Blackwell, separate fresh process (never fork an audit pool from the runner's coordinator): audit the canary run + compare with the laptop."""
    LP, R, K, T = S.code2d(LDIR.parent / "code_v2")
    os.sched_setaffinity(0, set(SHARES["BLACKWELL"]["cpus"]))
    out = LDIR / "runs" / "canary_blackwell"
    res = json.loads(sorted(out.glob("run_*.json"))[-1].read_text())
    a = K.audit(out, procs=16)
    rows = K.load(out)
    comp = [q for _, q, _, _ in rows if q["eligible"] and q["completed"]]
    doc = {"roots": sorted({h["root_id"] for h, _, _, _ in rows}), "verification": a["verification"], "fingerprint": a["fingerprint"],
           "labels": {q["trial_id"]: [q.get("label"), q.get("negative_kind")] for _, q, _, _ in rows},
           "cpu_s_per_query": round(sum(q["volatile"]["cpu_s"] for q in comp) / max(1, len(comp)), 3), "run": {k: res.get(k) for k in ("stop", "wall_s", "finished_now")}}
    S.atomic(out / "canary_fingerprint.json", json.dumps(doc, sort_keys=True))
    S.canary_compare(out / "canary_fingerprint.json", SPILL / "PORTABILITY_CANARY_REPORT_BLACKWELL.json", ref=LDIR / "runs" / "canary_laptop_ref",
                     code=LDIR.parent / "code_v2")


def merge_v2(*dirs):
    """NEW canonical Q2 directory from the 6 disjoint sources of owner map V2, then launch_check.audit (re-certifies every witness)."""
    LP, R, K, T = S.code2d(LDIR.parent / "code_v2")
    own = json.loads((H2 / MF2).read_text())
    snap1 = json.loads((H1 / "HANDOFF_ROOT_OWNER_MANIFEST.json").read_text())["snapshot_sha256"]
    names = ("ORIGINAL", "LAPTOP", "GVLAB", *NEW)
    src = {o: Path(d) for o, d in zip(names, dirs)}
    merged = SPILL / "merged" / "q2_canonical_v2"
    if merged.exists():
        raise SystemExit("REFUSED merge-v2: the canonical merged directory exists (never overwrite)")
    stray = {o: sorted(f.stem for f in (src[o] / "records").glob("*.json") if own["owner"].get(f.stem) != o) for o in names if o != "ORIGINAL"}
    if any(stray.values()):
        raise SystemExit(f"REFUSED merge-v2: a run packed roots it does not own: { {k: v[:3] for k, v in stray.items()} }")
    tmp = merged.with_name(".q2_canonical_v2.tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    for d in ("records", "modes", "witnesses"):
        (tmp / d).mkdir(parents=True)
    shutil.copyfile(LDIR / "split_manifest.json", SPILL / "split_manifest.json")
    prov, seen = {}, set()
    for rid, o in sorted(own["owner"].items()):
        frozen = snap1.get(rid) if o == "ORIGINAL" else own["snapshot_sha256_v2"].get(rid) if o in ("LAPTOP", "GVLAB") else None
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
    pol = json.loads((LDIR / "configs/production_policy.json").read_text())
    S.atomic(tmp / "run_manifest.json", json.dumps({"scope": "Q2_MERGED_CANONICAL_V2", "policy": pol["policy"], "budget_hash": T.budget_hash(pol["policy"]),
                                                    "pair_protocol_id": T.pair_protocol()[0], "queries": len(seen), "roots": len(prov),
                                                    "owner_manifest_v2_sha256": S.sha(H2 / MF2)}, sort_keys=True))
    S.atomic(tmp / "provenance.json", json.dumps(prov, sort_keys=True))
    os.replace(tmp, merged)
    a = K.audit(merged, procs=16)
    teach = sum(1 for _, q, _, _ in K.load(merged) if q["eligible"])
    rep = {"version": VERSION2, "utc": S.utc(), "merged_dir": str(merged), "roots": len(prov), "tasks": len(seen), "teacher_queries": teach,
           "by_owner": dict(Counter(v["owner"] for v in prov.values())), "audit": a["verification"], "summary": a["summary"],
           "run_fingerprint": a["fingerprint"]["run_fingerprint"],
           "pass": a["verification"]["failed"] == 0 and teach == S.EXPECT["teacher_queries"] and len(seen) == S.EXPECT["tasks"]}
    S.atomic(SPILL / "MERGED_Q2_V2_AUDIT.json", json.dumps(rep, indent=1, sort_keys=True))
    print(json.dumps({k: rep[k] for k in ("pass", "roots", "tasks", "teacher_queries", "by_owner")}))


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        raise SystemExit(__doc__)
    c = a[0]
    if c == "classify-export":
        classify_export(a[1], a[2], a[3])
    elif c == "partition-v2":
        partition_v2(a[1], a[2], float(a[3]), float(a[4]), float(a[5]))
    elif c == "copy-partials":
        copy_partials(a[1])
    elif c == "run":
        run_share(a[1], "--dry-run" in a)
    elif c == "canary":
        canary()
    elif c == "canary-check":
        canary_check()
    elif c == "merge-v2":
        merge_v2(*a[1:7])
    else:
        raise SystemExit(__doc__)
