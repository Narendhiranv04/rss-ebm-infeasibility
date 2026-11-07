"""Live SHELF66V3 source snapshot: consistency cut, audit against the pinned GitHub snapshot, generator-identical selection, manifest.

  select-check PINNED_DATASET                      the selection rule below must reproduce the pinned selected.json exactly
  audit STAGING PINNED_DATASET CUT_ISO OUT_DIR     STAGING = <snapshot>/raw (+ meta/); writes STAGING/../selected.json, the manifest and
                                                   the audit; files written after the cut are moved to <snapshot>/meta/after_cut/

Consistency: the generator writes witnesses (atomic, content addressed) and the root JSON (atomic) BEFORE its coordinator appends +
fsyncs the root's ledger line (run_3dt_plus / bd_core.Store).  The snapshot is the complete-line ledger prefix read at the cut plus
exactly the roots it lists as accepted and the witnesses they reference; root / witness files that appeared after the cut are excluded.
Selection = run_shelf66_full.select (accepted roots in ledger order, per verified class, up to the quota) with the roots of
EXCLUDED_FIXED_OBJECT.json skipped (the pinned selected.json's rule).
"""
import gzip, hashlib, json, os, re, shutil, sys
from collections import Counter
from pathlib import Path

import numpy as np

QUOTA = {"NONE": 408, "0": 511, "1": 357, "2": 357, "3": 357, "4": 510}      # run_shelf66_full.QUOTA (no QUOTA.json on the live host)
WID = re.compile(r"^[0-9a-f]{24}$")
V3 = {"generator_version": "BD_GEN_V1", "robot_protocol": "XARM7_SHELF66_LEVELGRASP_V1:cd7f78f44022", "config_hash": "8d521b1f6d23db23",
      "label_protocol": "BD_BINARY_BUDGET_V1"}
FIXED_CHECKED_ACCEPTED = 1147                                  # EXCLUDED_FIXED_OBJECT.json: "every accepted root up to 2026-11-05 09:20"


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def wid_of(arrays):
    h = hashlib.sha256()
    for k in sorted(arrays):
        v = np.ascontiguousarray(np.asarray(arrays[k]))
        h.update(f"{k}|{v.dtype.str}|{v.shape}".encode()); h.update(v.tobytes())
    return h.hexdigest()[:24]


def wids(x, out):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("witness", "witnesses"):
                for w in (v if isinstance(v, list) else [v]):
                    if isinstance(w, str) and WID.match(w):
                        out.add(w)
            else:
                wids(v, out)
    elif isinstance(x, list):
        for v in x:
            wids(v, out)
    return out


def ledger(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def select(led, excluded):
    got, pick = {k: 0 for k in QUOTA}, {k: [] for k in QUOTA}
    for e in led:
        if e.get("accepted") and e["observed"] in got and got[e["observed"]] < QUOTA[e["observed"]] and e["root_id"] not in excluded:
            got[e["observed"]] += 1
            pick[e["observed"]].append(e["root_id"])
    return got, pick


def select_check(pinned):
    pinned = Path(pinned)
    sel = json.loads((pinned / "selected.json").read_text())
    exc = json.loads((pinned / "raw/EXCLUDED_FIXED_OBJECT.json").read_text())["excluded"]
    got, pick = select(ledger(pinned / "raw/ledger.jsonl"), set(exc))
    ok = got == sel["counts"] and pick == sel["selected"] and sel["quota"] == QUOTA
    print(json.dumps({"reproduces_pinned_selected_json": ok, "counts": got, "pinned_counts": sel["counts"], "keys": sorted(sel)}))
    return ok, sel


def audit(staging, pinned, cut, out):
    raw, pinned, out = Path(staging), Path(pinned), Path(out)
    snap = raw.parent
    meta = snap / "meta"
    out.mkdir(parents=True, exist_ok=True)
    ok_rule, sel_pinned = select_check(pinned)
    assert ok_rule, "selection rule does not reproduce the pinned selected.json"
    led, led_old = ledger(raw / "ledger.jsonl"), ledger(pinned / "raw/ledger.jsonl")
    new_lines, old_lines = (raw / "ledger.jsonl").read_bytes().splitlines(), (pinned / "raw/ledger.jsonl").read_bytes().splitlines()
    prefix = new_lines[:len(old_lines)] == old_lines
    acc = [e["root_id"] for e in led if e.get("accepted")]
    rfiles = {p.stem: p for p in (raw / "roots").glob("*.json")}
    missing_roots = sorted(set(acc) - set(rfiles))
    after_roots = sorted(set(rfiles) - set(acc))
    (meta / "after_cut/roots").mkdir(parents=True, exist_ok=True)
    for r in after_roots:
        shutil.move(str(rfiles.pop(r)), str(meta / "after_cut/roots" / f"{r}.json"))
    refs, per_root, prot, bad_json = set(), {}, Counter(), []
    for r in acc:
        if r not in rfiles:
            continue
        try:
            rec = json.loads(rfiles[r].read_text())
        except ValueError:
            bad_json.append(r); continue
        ws = wids(rec, set()); per_root[r] = sorted(ws); refs |= ws
        ids = rec["ids"]
        prot[(ids.get("generator_version"), rec.get("generator_version_plus"), ids.get("robot_protocol"), ids.get("config_hash"),
              ids.get("label_protocol"), ids.get("teacher_version", "")[:18])] += 1
    wfiles = {p.stem: p for p in (raw / "witnesses").glob("*/*.npz")}
    missing_w = sorted(refs - set(wfiles))
    corrupt = []
    for w in sorted(refs & set(wfiles)):
        try:
            with np.load(wfiles[w]) as z:
                if wid_of({k: z[k] for k in z.files}) != w:
                    corrupt.append(w)
        except Exception:  # noqa: BLE001
            corrupt.append(w)
    # unreferenced witness files: the pinned snapshot's convention decides (it kept the whole store); after-cut ones are excluded
    from datetime import datetime, timezone
    tcut = datetime.fromisoformat(cut.replace("Z", "+00:00")).timestamp()
    unref = sorted(set(wfiles) - refs)
    unref_after = [w for w in unref if wfiles[w].stat().st_mtime > tcut]
    (meta / "after_cut/witnesses").mkdir(parents=True, exist_ok=True)
    for w in unref_after:
        shutil.move(str(wfiles.pop(w)), str(meta / "after_cut/witnesses" / f"{w}.npz"))
    pin_w = {p.stem for p in (pinned / "raw/witnesses").glob("*/*.npz")}
    pin_refs = set()
    for p in (pinned / "raw/roots").glob("*.json"):
        pin_refs |= wids(json.loads(p.read_text()), set())
    exc = json.loads((raw / "EXCLUDED_FIXED_OBJECT.json").read_text())["excluded"]
    got, pick = select(led, set(exc))
    order = {r: i for i, r in enumerate(acc)}
    pending = sorted(r for v in pick.values() for r in v if order[r] >= FIXED_CHECKED_ACCEPTED)
    sel = {"rule": sel_pinned["rule"], "status": f"IN PROGRESS: consistent snapshot of the live store cut at {cut} (ledger prefix of {len(led)} complete"
                                                 f" lines); generation (K4 / NONE) continues on the lab server; {len(pending)} selected roots accepted after"
                                                 " 2026-11-05 09:20 have not been through the fixed-object check yet",
           "quota": QUOTA, "counts": got, "excluded_fixed_object": {r: v["ledger_class"] for r, v in exc.items()},
           "fixed_object_check_pending": pending, "selected": pick}
    (snap / "selected.json").write_text(json.dumps(sel, indent=1))
    old_sel = {r for v in sel_pinned["selected"].values() for r in v}
    new_sel = {r for v in pick.values() for r in v}
    # file-level comparison with the pinned snapshot
    files = {}
    for p in sorted(raw.rglob("*")):
        if p.is_file():
            files[str(p.relative_to(snap))] = {"sha256": fsha(p), "bytes": p.stat().st_size}
    files["selected.json"] = {"sha256": fsha(snap / "selected.json"), "bytes": (snap / "selected.json").stat().st_size}
    pinfiles = {str(p.relative_to(pinned)): p for p in pinned.rglob("*") if p.is_file() and (p.parts[len(pinned.parts)] in ("raw",) or p.name == "selected.json")}
    same, changed, removed = [], [], []
    for k, p in pinfiles.items():
        if k not in files:
            removed.append(k)
        elif fsha(p) == files[k]["sha256"]:
            same.append(k)
        else:
            changed.append(k)
    new = sorted(set(files) - set(pinfiles))
    changed_kind = Counter(k.split("/")[1] if k.startswith("raw/") and "/" in k[4:] else k for k in changed)
    summ = {"cut_utc": cut, "ledger_lines": len(led), "ledger_prefix_of_pinned_identical": prefix, "pinned_ledger_lines": len(old_lines),
            "accepted": len(acc), "observed": Counter(e.get("observed") for e in led if e.get("accepted")),
            "rejected_in_construction": sum(not e.get("accepted") for e in led), "root_files": len(rfiles), "missing_root_files": missing_roots,
            "after_cut_root_files_excluded": after_roots, "unparseable_roots": bad_json,
            "witness_files": len(wfiles), "referenced_witnesses": len(refs), "missing_witnesses": missing_w, "corrupt_witnesses": corrupt,
            "unreferenced_witness_files_kept_pre_cut": len(set(wfiles) - refs), "unreferenced_after_cut_excluded": len(unref_after),
            "pinned_unreferenced_witness_files": len(pin_w - pin_refs), "protocol_tuples": {" | ".join(map(str, k)): v for k, v in prot.items()},
            "all_roots_v3_protocol": all(k[0] == V3["generator_version"] and k[2] == V3["robot_protocol"] and k[3] == V3["config_hash"]
                                         and k[4] == V3["label_protocol"] for k in prot),
            "selected_counts": got, "selected_total": sum(got.values()), "pinned_selected_counts": sel_pinned["counts"],
            "pinned_selected_still_selected": len(old_sel & new_sel), "pinned_selected_dropped": sorted(old_sel - new_sel),
            "newly_selected": len(new_sel - old_sel), "fixed_object_check_pending_selected": len(pending),
            "files_vs_pinned": {"identical": len(same), "changed": len(changed), "changed_by_kind": changed_kind, "changed_examples": changed[:10],
                                "removed": removed[:20], "n_removed": len(removed), "new": len(new)},
            "k4_none_status": {"K4_selected": got["4"], "K4_quota": QUOTA["4"], "NONE_selected": got["NONE"], "NONE_quota": QUOTA["NONE"]}}
    man = {"snapshot": {"source_host": "user1@10.4.25.63 (lab generator host; read-only access from the user's laptop)",
                        "source_path": "~/yadu_bd/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3",
                        "cut_utc": cut, "method": "complete-line ledger prefix read first; then rsync -a --partial of roots/ + witnesses/; files not"
                        " covered by the prefix excluded; second-pass sha256 on the source host must equal the staged bytes",
                        "generator_code": "byte-identical (git blob sha1) to yadu/dataset-3d@8032ad0b shelf66_v3/, scene/, dataset-bd/bd_core.py, run_3dt_plus.py",
                        "parent_commit": "8032ad0bf842a490897a8247ee46312036c12546"},
           "summary": summ, "files": files, "per_root_witnesses": per_root}
    with gzip.open(out / "SOURCE_SNAPSHOT_MANIFEST.json.gz", "wt") as f:
        json.dump(man, f, sort_keys=True, default=dict)
    (out / "source_snapshot_summary.json").write_text(json.dumps(summ, indent=1, sort_keys=True, default=dict))
    print(json.dumps(summ, default=dict)[:4000])


def pin_live(snap, out):
    """Collector source pin for the live snapshot: sha256 of every selected root (from the snapshot manifest, re-hashed now)."""
    snap = Path(snap)
    man = json.load(gzip.open(snap / "audit/SOURCE_SNAPSHOT_MANIFEST.json.gz", "rt"))
    sel = json.loads((snap / "selected.json").read_text())
    roots = sorted(r for v in sel["selected"].values() for r in v)
    cur = {r: fsha(snap / "raw/roots" / f"{r}.json") for r in roots}
    bad = [r for r in roots if man["files"].get(f"raw/roots/{r}.json", {}).get("sha256") != cur[r]]
    if bad:
        sys.exit(f"SOURCE DRIFT vs snapshot manifest: {bad[:3]}")
    doc = {"snapshot": f"live SHELF66V3 store, cut {man['snapshot']['cut_utc']} (user1@10.4.25.63)", "manifest_sha256": fsha(snap / "audit/SOURCE_SNAPSHOT_MANIFEST.json.gz"),
           "selected_sha256": fsha(snap / "selected.json"), "n_roots": len(cur), "roots": cur}
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True)); print({k: doc[k] for k in ("snapshot", "manifest_sha256", "n_roots")})


if __name__ == "__main__":
    if sys.argv[1] == "pin-live":
        pin_live(sys.argv[2], sys.argv[3]); sys.exit(0)
    if sys.argv[1] == "select-check":
        sys.exit(0 if select_check(sys.argv[2])[0] else 1)
    audit(*sys.argv[2:6])
