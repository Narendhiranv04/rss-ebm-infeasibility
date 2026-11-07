"""Verify a FRESH clone of naren/3d-dataset-relocations (run on gvlab, MuJoCo 3.16.0).  python verify_clone.py CLONE WORKING_V2 OUT.json"""
import gzip, hashlib, json, re, subprocess, sys
from pathlib import Path
import numpy as np
clone, work, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
S6 = clone / "gh-yadu-iiith/shelf_6x6_open_sides"; DS = S6 / "dataset_bd_classquota_without_home_position_no_overpass"; V2 = S6 / "dataset_placement_extension_v2"
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
res = {"clone_head": subprocess.run(["git", "-C", str(clone), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()}
# 1. snapshot files == manifest; selected roots + referenced witnesses present and re-hash
man = json.load(gzip.open(V2 / "source_sync/SOURCE_SNAPSHOT_MANIFEST.json.gz", "rt"))
bad = [k for k, v in man["files"].items() if not (DS / k).exists() or sha(DS / k) != v["sha256"]]
raw_files = {str(p.relative_to(DS)) for p in (DS / "raw").rglob("*") if p.is_file()}
sel = json.loads((DS / "selected.json").read_text())
ids = [r for v in sel["selected"].values() for r in v]
WID = re.compile(r"^[0-9a-f]{24}$")
def wids(x, o):
    if isinstance(x, dict):
        for k, v in x.items():
            if k in ("witness", "witnesses"):
                for w in (v if isinstance(v, list) else [v]):
                    if isinstance(w, str) and WID.match(w): o.add(w)
            else: wids(v, o)
    elif isinstance(x, list):
        for v in x: wids(v, o)
    return o
refs = set(); miss_root = []
for r in ids:
    p = DS / "raw/roots" / f"{r}.json"
    if not p.exists(): miss_root.append(r); continue
    refs |= wids(json.loads(p.read_text()), set())
def wid_of(a):
    h = hashlib.sha256()
    for k in sorted(a):
        v = np.ascontiguousarray(np.asarray(a[k])); h.update(f"{k}|{v.dtype.str}|{v.shape}".encode()); h.update(v.tobytes())
    return h.hexdigest()[:24]
miss_w, corrupt = [], []
for w in sorted(refs):
    p = DS / "raw/witnesses" / w[:2] / f"{w}.npz"
    if not p.exists(): miss_w.append(w); continue
    with np.load(p) as z:
        if wid_of({k: z[k] for k in z.files}) != w: corrupt.append(w)
led = [json.loads(x) for x in (DS / "raw/ledger.jsonl").read_text().splitlines() if x.strip()]
exc = set(json.loads((DS / "raw/EXCLUDED_FIXED_OBJECT.json").read_text())["excluded"])
Qt = {"NONE": 408, "0": 511, "1": 357, "2": 357, "3": 357, "4": 510}; got = {k: [] for k in Qt}
for e in led:
    if e.get("accepted") and e["observed"] in got and len(got[e["observed"]]) < Qt[e["observed"]] and e["root_id"] not in exc:
        got[e["observed"]].append(e["root_id"])
res["snapshot"] = {"manifest_entries": len(man["files"]), "manifest_mismatch_or_missing": len(bad), "raw_files_not_in_manifest": len(raw_files - set(man["files"])),
                   "selected_roots": len(ids), "selected_counts": {k: len(v) for k, v in sel["selected"].items()}, "missing_selected_root_files": miss_root,
                   "referenced_witnesses_by_selected": len(refs), "missing_witnesses": len(miss_w), "corrupt_witnesses": len(corrupt),
                   "selection_reproduced_from_cloned_ledger": got == sel["selected"]}
# 2. v2 sidecar bytes == working copy on gvlab
cl = {str(p.relative_to(V2)): sha(p) for p in V2.rglob("*") if p.is_file()}
wk = {str(p.relative_to(work)): sha(p) for p in work.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
res["v2_sidecar"] = {"files_in_clone": len(cl), "files_in_working_copy": len(wk), "differ": sorted(k for k in cl if k in wk and cl[k] != wk[k])[:10],
                     "only_in_clone": sorted(set(cl) - set(wk))[:10], "only_in_working": sorted(set(wk) - set(cl))[:10]}
# 3. native SOURCE witness replay (unchanged shelf_scene.run_plan) on newly selected K3 roots of the live snapshot
sys.path.insert(0, str(V2 / "code"))
import placement3d as P  # noqa: E402  (SRC = the clone's dataset path)
assert str(P.SRC.resolve()) == str(DS.resolve()), P.SRC
old = set(json.load(gzip.open(S6 / "dataset_placement_extension_v1/audit/source_manifest.json.gz", "rt"))["files"])
new_k3 = [r for r in sel["selected"]["3"] if f"raw/roots/{r}.json" not in old][:6]
rep = []
for rid in new_k3:
    rec = json.loads((DS / "raw/roots" / f"{rid}.json").read_text())
    b = next((x for x in rec["labels"]["B"] if x.get("y") == 1 and x.get("witnesses")), None)
    if b is None: rep.append({"root": rid, "skipped": "no B witness"}); continue
    st, w = b["steps"][0], b["witnesses"][0]
    z = np.load(DS / "raw/witnesses" / w[:2] / f"{w}.npz"); wz = {k: z[k] for k in z.files}
    e = P.bd_shelf.build(rec["objects"], rec["initial_cells"]); i = e.obj_id(st["object"]); e._focus(i); e.objects[i]["pos"] = e.obj_pos(i)
    g = P.S.grasp_poses(e.objects[i], e.table_z)[st["grasp_id"]]
    ph = [str(x) for x in wz["phases"]]
    segs = [dict(phase=p, kind="grip", grip=(None if p == "close" else g["open_width"])) if p in ("open", "close", "release")
            else dict(phase=p, kind="joint" if p in ("to_hover", "to_ready") else "cart", q=[r_ for r_ in wz["q"][wz["segment"] == k]]) for k, p in enumerate(ph)]
    others = {o: e.cell_of(o) for o in e.obj_index if o != st["object"]}
    e.run_plan({"name": st["object"], "segments": segs})
    zerr = float(e.obj_pos(st["object"])[2] - e.table_z - e.objects[i]["height"] / 2)
    ok = e.cell_of(st["object"]) == tuple(st["dest"]) and abs(zerr) <= 0.01 and all(e.cell_of(o) == c for o, c in others.items())
    rep.append({"root": rid, "witness": w, "replay_ok": bool(ok), "recorded_ok": bool(st["ok"]), "z_err": zerr, "recorded_z_err": st["z_err"]})
res["new_root_source_replay"] = {"roots": len(rep), "agree_with_recorded": sum(x.get("replay_ok") == x.get("recorded_ok") for x in rep), "results": rep}
Path(out).write_text(json.dumps(res, indent=1, default=str)); print(json.dumps({k: v for k, v in res.items() if k != "new_root_source_replay"}), json.dumps({k: v for k, v in res["new_root_source_replay"].items() if k != "results"}))
