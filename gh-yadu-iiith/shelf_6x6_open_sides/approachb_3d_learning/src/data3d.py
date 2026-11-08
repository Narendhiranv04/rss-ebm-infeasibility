"""3D v4 (Q2_3D_PARITY_V1) immutable snapshot, verification, label-blind features and GPU loader.

    python src/data3d.py build      --incoming snapshots/incoming --out snapshots      verify + freeze -> snapshots/v4full_<id>/ (read-only)
    python src/data3d.py featurize  --snapshot snapshots/v4full_<id>                    -> <snap>/derived/features.npz + scalers.json
    python src/data3d.py audit      --snapshot snapshots/v4full_<id>                    -> reports/02_3D_DATASET_AUDIT.json

Joins are by root_id / qid / witness_id only (never list position).  Generator inputs are the label-blind root fields of q2roster3d.allowed()
(object shape / size / height, cells, shelf geometry); K, goal, roles, blockers, repairs, construction spec and teacher outcomes are never
features (K is stored for report stratification only).  Static contrasts are a separate stream and never physical negatives."""
from __future__ import annotations

import argparse, gzip, hashlib, json, math, os, shutil, stat, tarfile, time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROWS = COLS = 6
X0, PITCH = 0.42, 0.075                         # cell centre = (x0 + (r - 2.5) pitch, (c - 2.5) pitch); xArm base at the world origin
MAXOBJ = 8
FEAT = ["is_box", "is_cyl", "size", "height", "r", "c", "x", "y", "dist", "angle", "is_query", "dr_q", "dc_q", "cheb_q"]
NORM = [2, 3, 6, 7, 8]                          # continuous columns standardized with TRAIN statistics
E = "gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2"
SRC = "gh-yadu-iiith/shelf_6x6_open_sides/dataset_bd_classquota_without_home_position_no_overpass"
UPSTREAM_COMMIT = "6c72eb4be17b"                # rss-ebm-infeasibility naren/3d-dataset-relocations (contains the v4 freeze 83c1135f)
STATUS = ["VERIFIED", "NEG_CERTIFIED_GEOMETRY", "NEG_UNDER_POLICY", "UNLABELED_NO_VALID_POST_GRASP_STATE"]
SPLITS = ["train", "val", "test"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cell_xy(r, c):
    return X0 + (r - 2.5) * PITCH, (c - 2.5) * PITCH


def cheb(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def allowed(raw):
    """The label-blind root fields (identical to q2roster3d.allowed minus goal/K): everything else in the raw root is privileged."""
    return {"root_id": raw["ids"]["root_id"], "objects": {o["id"]: {x: o[x] for x in ("shape", "size", "height")} for o in raw["objects"]},
            "geom": {x: raw["geometry"][x] for x in ("x0", "pitch", "rows", "cols")}}


# ----------------------------------------------------------------------------------------------------------------- snapshot
def build(incoming, out):
    inc, out = Path(incoming), Path(out)
    mir, gv = inc / "run_q2_parity_mirror", inc / "gvlab_v4"
    up = inc / "upstream"
    if not up.exists():
        with tarfile.open(inc / "upstream_6c72eb4b.tar") as t:
            t.extractall(up, filter="data")
    man = json.loads((gv / "run_manifest.json").read_text())
    idx = json.loads((mir / "backup_index.json").read_text())["units"]
    aud = json.loads((gv / "run_q2_parity_audit.json").read_text())
    rep = json.loads((gv / "run_q2_parity_replay.json").read_text())
    pol = json.loads((gv / "production_policy_v4.sanitized.json").read_text())
    bad, units = [], sorted(man["units"])
    if sha(mir / "run_manifest.json") != sha(gv / "run_manifest.json"):
        bad.append("mirror run_manifest differs from the gvlab run_manifest")
    led = [json.loads(x) for x in (gv / "ledger.jsonl").read_text().splitlines() if x.strip()]
    done = {e["unit"] for e in led if e.get("event") == "unit_done"}
    ends = [e for e in led if e.get("event") == "end"]
    if done != set(units) or not (ends and ends[-1].get("complete")):
        bad.append(f"ledger: {len(done)} unit_done vs {len(units)} units, last end complete={bool(ends and ends[-1].get('complete'))}")
    if not (aud.get("complete") and aud.get("validated") == 1466 and aud.get("queries") == 59972 and not aud.get("missing_or_invalid")):
        bad.append("gvlab collect4 --audit not complete")
    if not (rep["replayed"] == rep["verified"] == aud["witness_associations"] and not rep["failures"] and not rep["missing_witnesses"]):
        bad.append("gvlab 100% witness replay not verified")
    rost = {}
    with gzip.open(up / E / "production_v4/frozen/q2_parity_roster_v4.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line); rost[r["qid"]] = r
    for k, v in man["inputs"].items():
        if sha(up / E / v["path"]) != v["sha256"]:
            bad.append(f"upstream input {k} sha differs from the run manifest")
    if pol["_sanitized"]["original_sha256"] != man["policy_file_sha256"]:
        bad.append("production policy sha differs from the run manifest")
    unit_sha, npz_sha, root_sha, labels, nrec = {}, {}, {}, Counter(), 0
    for u in units:
        j, z = mir / "units" / f"{u}.json", mir / "units" / f"{u}.npz"
        unit_sha[u] = sha(j)
        if unit_sha[u] != idx[u]["json_sha256"]:
            bad.append(f"{u}: json sha differs from the gvlab backup index"); continue
        d = json.loads(j.read_text())
        if idx[u].get("npz_sha256"):                    # witness NPZ: hash BOUND here from the gvlab index; files verified by attach-witnesses
            npz_sha[u] = idx[u]["npz_sha256"]
            if d.get("npz_sha256") != npz_sha[u]:
                bad.append(f"{u}: unit json npz_sha256 differs from the gvlab backup index")
        elif any(r.get("witnesses") for r in d["records"]):
            bad.append(f"{u}: witnesses recorded but no npz")
        if {r["qid"] for r in d["records"]} != set(man["units"][u]):
            bad.append(f"{u}: record qids differ from the manifest")
        for r in d["records"]:
            ro = rost.get(r["qid"])
            if ro is None or ro["split"] != r["split"] or ro["object"] != r["object"] or list(ro["dest"]) != list(r["dest"]):
                bad.append(f"{r['qid']}: record does not join the roster by qid"); break
            labels[f"{r['label']}:{r['status']}"] += 1; nrec += 1
            for w in r.get("witnesses") or []:
                if w["witness_id"] not in d["witness_index"]:
                    bad.append(f"{r['qid']}: ghost witness {w['witness_id']}")
        rp = up / SRC / "raw/roots" / f"{u}.json"
        root_sha[u] = sha(rp)
        if root_sha[u] != man["source_pin_roots"][u]:
            bad.append(f"{u}: source root sha differs from the source pin")
    alab = Counter({":".join(k.split(":")[:2]): 0 for k in aud["labels"]})
    for k, n in aud["labels"].items():
        alab[":".join(k.split(":")[:2])] += n
    if dict(labels) != dict(alab):
        bad.append(f"label counts differ from the gvlab audit: {dict(labels)} vs {dict(alab)}")
    if bad:
        raise SystemExit("REFUSED snapshot: " + json.dumps(bad[:20], indent=1))
    core = {"upstream_commit": UPSTREAM_COMMIT, "roster_sha256": man["roster_sha256"], "inputs": man["inputs"], "policy": man["policy"],
            "stages": man["stages"], "protocol_id": man["protocol"][0], "policy_file_sha256": man["policy_file_sha256"],
            "units_json_sha256": unit_sha, "units_npz_sha256": npz_sha, "source_root_sha256": root_sha}
    sid = hashlib.sha256(json.dumps(core, sort_keys=True).encode()).hexdigest()[:16]
    dst = out / f"v4full_{sid}"
    if dst.exists():
        raise SystemExit(f"REFUSED snapshot: {dst} exists (immutable; never overwritten)")
    dst.mkdir(parents=True)
    ex = inc / "npz_excluded_from_label_layer"; ex.mkdir(exist_ok=True)
    for z in (mir / "units").glob("*.npz"):                 # the label layer holds JSON only; witnesses form their own verified layer
        os.replace(z, ex / z.name)
    os.replace(mir, dst / "run_q2_parity")
    os.replace(up, dst / "upstream")
    shutil.copytree(gv, dst / "gvlab_reports")
    summary = {"snapshot_id": sid, "kind": "FULL_COMPLETE_V4", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "units": len(units), "query_bearing_roots": len(units), "queries": nrec, "labels": dict(labels),
               "split_queries": dict(Counter(rost[q]["split"] for u in units for q in man["units"][u])),
               "static_contrasts_in_roster": sum(1 for r in rost.values() if r.get("static")),
               "verification": {"ledger_units_done": len(done), "audit": {k: aud[k] for k in ("complete", "validated", "queries", "witness_associations")},
                                "replay": {k: rep[k] for k in ("replayed", "verified", "unique_witnesses", "max_q_exec_diff", "max_obj_end_diff_m")},
                                "unit_json_checked_against_gvlab_backup_index": len(unit_sha), "source_roots_checked_against_pin": len(root_sha),
                                "witness_npz_bound_from_gvlab_index": len(npz_sha),
                                "witness_layer": "separate read-only layer <snapshot>_witnesses, verified file-by-file by `attach-witnesses`"},
               "binding": {k: v for k, v in core.items() if not k.startswith("units_") and k != "source_root_sha256"},
               "source": "laptop off-host mirror of gvlab production_v4/run_q2_parity (verified 2026-11-08T00:26Z), relayed laptop -> server"}
    (dst / "SNAPSHOT_MANIFEST.json").write_text(json.dumps({**summary, "core": core}, indent=1, sort_keys=True))
    sums = [f"{sha(p)}  {p.relative_to(dst)}" for p in sorted(dst.rglob("*")) if p.is_file()]
    (dst / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    readonly(dst)
    print(json.dumps(summary, indent=1))
    return dst


def attach_witnesses(snapshot, staging):
    """Verify every bound witness NPZ (sha256 from the gvlab index, bound into the snapshot id) and freeze them as <snapshot>_witnesses."""
    snap = Path(snapshot)
    core = json.loads((snap / "SNAPSHOT_MANIFEST.json").read_text())["core"]
    lay = Path(str(snap) + "_witnesses")
    if lay.exists():
        raise SystemExit(f"REFUSED: {lay} exists (immutable)")
    bad = [u for u, h in core["units_npz_sha256"].items() if not (Path(staging) / "units" / f"{u}.npz").exists()
           or sha(Path(staging) / "units" / f"{u}.npz") != h]
    if bad:
        raise SystemExit(f"REFUSED: {len(bad)} witness files missing or mismatched, e.g. {bad[:3]}")
    (lay / "units").mkdir(parents=True)
    for u in core["units_npz_sha256"]:
        os.link(Path(staging) / "units" / f"{u}.npz", lay / "units" / f"{u}.npz")
    (lay / "WITNESS_LAYER_MANIFEST.json").write_text(json.dumps({"snapshot_id": snap.name, "files": len(core["units_npz_sha256"]),
                                                                 "verified": "every file sha256 == bound gvlab index hash", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, indent=1))
    readonly(lay)
    print(json.dumps({"witness_layer": str(lay), "files": len(core["units_npz_sha256"])}))


def readonly(d):
    for p in sorted(Path(d).rglob("*"), reverse=True) + [Path(d)]:
        m = p.stat().st_mode
        p.chmod(m & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


# ----------------------------------------------------------------------------------------------------------------- features
def scene_features(objs, cells, query):
    """objs: {id: {shape,size,height}}, cells: {id: (r,c)} of the HYPOTHETICAL scene, query: object id -> (F [MAXOBJ,14], mask, occ[36], legal[36]).
    Object order = sorted ids with the query first (the encoder is permutation-invariant; ids themselves are never features)."""
    ids = [query] + sorted(o for o in cells if o != query)
    F, M, occ = np.zeros((MAXOBJ, len(FEAT)), np.float32), np.zeros(MAXOBJ, bool), np.zeros(36, np.float32)
    qr, qc = cells[query]
    for i, o in enumerate(ids):
        r, c = cells[o]; x, y = cell_xy(r, c); g = objs[o]
        F[i] = [g["shape"] == "box", g["shape"] == "cylinder", g["size"], g["height"], r / 5, c / 5, x, y, math.hypot(x, y),
                math.atan2(y, x) / math.pi, o == query, (r - qr) / 5, (c - qc) / 5, cheb((r, c), (qr, qc)) / 5]
        M[i] = True; occ[r * COLS + c] = 1
    return F, M, occ, (occ == 0)


def featurize(snapshot):
    snap = Path(snapshot)
    man = json.loads((snap / "gvlab_reports/run_manifest.json").read_text())
    rost = {}
    with gzip.open(snap / "upstream" / E / "production_v4/frozen/q2_parity_roster_v4.jsonl.gz", "rt") as f:
        for line in f:
            r = json.loads(line); rost[r["qid"]] = r
    roots = {u: allowed(json.loads((snap / "upstream" / SRC / "raw/roots" / f"{u}.json").read_text())) for u in man["units"]}
    rows, F, M, OCC, LEG = [], [], [], [], []
    for u in sorted(man["units"]):
        d = json.loads((snap / "run_q2_parity/units" / f"{u}.json").read_text())
        for r in d["records"]:
            ro = rost[r["qid"]]
            cells = {o: tuple(c) for o, c in ro["cells"].items()}
            f, m, occ, leg = scene_features(roots[u]["objects"], cells, ro["object"])
            dest = ro["dest"][0] * COLS + ro["dest"][1]
            if not leg[dest]:
                raise SystemExit(f"{r['qid']}: recorded destination is not free in its hypothetical scene")
            F.append(f); M.append(m); OCC.append(occ); LEG.append(leg)
            rows.append({"qid": r["qid"], "root": u, "ctx": f"{u}|{ro['scene_hash']}|{ro['object']}", "dest": dest,
                         "label": -1 if r["label"] is None else int(r["label"]), "status": r["status"], "split": ro["split"],
                         "shape": ro["shape"], "height_bin": ro["height_bin"], "block": ro["block"], "band": ro["band"], "depth": ro["depth"],
                         "n_obj": ro["n_obj"], "cheb": ro["cheb"], "source_bin": ro["source_bin"], "K": ro["K"],
                         "src": ro["src"][0] * COLS + ro["src"][1], "witness_id": r.get("witness_id")})
    F, M, OCC, LEG = map(np.stack, (F, M, OCC, LEG))
    split = np.array([SPLITS.index(r["split"]) for r in rows], np.int8)
    tr = (split == 0)
    mu, sd = np.zeros(len(FEAT), np.float32), np.ones(len(FEAT), np.float32)
    for j in NORM:
        v = F[tr][:, :, j][M[tr]]
        mu[j], sd[j] = v.mean(), v.std() + 1e-6
    ctx_ids = {c: i for i, c in enumerate(dict.fromkeys(r["ctx"] for r in rows))}
    cat = lambda k: np.array([r[k] for r in rows])
    out = snap / "derived" if os.access(snap, os.W_OK) else Path(str(snap) + "_derived")
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "features.npz", obj=F, mask=M, occ=OCC, legal=LEG, dest=cat("dest").astype(np.int16), label=cat("label").astype(np.int8),
                        status=np.array([STATUS.index(r["status"]) for r in rows], np.int8), split=split,
                        ctx=np.array([ctx_ids[r["ctx"]] for r in rows], np.int32), src=cat("src").astype(np.int16),
                        qid=cat("qid"), root=cat("root"), shape=cat("shape"), height_bin=cat("height_bin"), block=cat("block"), band=cat("band"),
                        depth=cat("depth").astype(np.int8), n_obj=cat("n_obj").astype(np.int8), cheb=cat("cheb").astype(np.int8),
                        source_bin=cat("source_bin"), K=cat("K"))
    sc = {"features": FEAT, "normalized_columns": [FEAT[j] for j in NORM], "mean": mu.tolist(), "std": sd.tolist(), "fit_on": "TRAIN split only",
          "maxobj": MAXOBJ, "grid": [ROWS, COLS], "cell_centre": "x = 0.42 + (r-2.5)*0.075, y = (c-2.5)*0.075 m (xArm base at origin)",
          "features_sha256": sha(out / "features.npz"), "snapshot": snap.name, "n": len(rows), "contexts": len(ctx_ids)}
    (out / "scalers.json").write_text(json.dumps(sc, indent=1))
    print(json.dumps({k: sc[k] for k in ("n", "contexts", "features_sha256")}))
    return out


# ----------------------------------------------------------------------------------------------------------------- audit
def audit(snapshot, report):
    snap = Path(snapshot)
    d = snap / "derived" if (snap / "derived").exists() else Path(str(snap) + "_derived")
    z = np.load(d / "features.npz", allow_pickle=False)
    man = json.loads((snap / "SNAPSHOT_MANIFEST.json").read_text())
    sp, lab, st = z["split"], z["label"], z["status"]
    out = {"snapshot_id": man["snapshot_id"], "kind": man["kind"], "queries": int(len(sp)), "contexts": int(z["ctx"].max() + 1),
           "roots_by_split": {}, "labels_by_split": {}, "verification": man["verification"], "binding": man["binding"]}
    roots = z["root"]
    for i, s in enumerate(SPLITS):
        m = sp == i
        out["roots_by_split"][s] = int(len(set(roots[m])))
        out["labels_by_split"][s] = {STATUS[k]: int(((st == k) & m).sum()) for k in range(len(STATUS))}
    # split hygiene: every root / context / scene layout in exactly one split
    rs = defaultdict(set)
    for r, s in zip(roots, sp):
        rs[r].add(int(s))
    out["roots_in_more_than_one_split"] = int(sum(len(v) > 1 for v in rs.values()))
    lay = defaultdict(set)                          # full physical layout: every object's (cell, shape, size, height), order-free
    for o, mk, s in zip(z["obj"], z["mask"], sp):
        lay[tuple(sorted(tuple(np.round(row[:6], 5)) for row in o[mk]))].add(int(s))
    out["scene_layouts_in_more_than_one_split"] = int(sum(len(v) > 1 for v in lay.values()))
    strata = {}
    for k in ("shape", "height_bin", "block", "band", "depth", "n_obj", "source_bin", "K"):
        strata[k] = {str(v): {"n": int((z[k] == v).sum()), "pos_rate_labeled": round(float(((z[k] == v) & (lab == 1)).sum() / max(1, ((z[k] == v) & (lab >= 0)).sum())), 3)}
                     for v in sorted(set(z[k].tolist()))}
    out["strata_all_splits"] = strata
    out["static_contrasts"] = {"count": man["static_contrasts_in_roster"], "use": "separate auxiliary stream; never physical negatives; not used here"}
    ctx = z["ctx"]
    pos_ctx = Counter(ctx[(lab == 1) & (sp == 0)])
    out["train_generator_support"] = {"positive_examples": int(((lab == 1) & (sp == 0)).sum()), "contexts_with_positive": len(pos_ctx),
                                      "positives_per_context_mean": round(float(np.mean(list(pos_ctx.values()))), 2)}
    out["notes"] = ["K is a reporting field only (never an input)", "UNLABELED rows are excluded from binary negatives",
                    "NEG_UNDER_POLICY is operational (bounded adaptive_v1 search), not impossibility",
                    "Q2 destinations are a stratified subsample (<= 8 per object), so untested cells are UNKNOWN, never negative"]
    Path(report).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("queries", "contexts", "roots_by_split", "roots_in_more_than_one_split", "scene_layouts_in_more_than_one_split")}))
    return out


# ----------------------------------------------------------------------------------------------------------------- GPU loader
class Data:
    """Whole-snapshot tensors on one device (no DataLoader workers): per-query rows + per-context evaluation tables."""

    def __init__(self, derived, device="cpu", geometry_only=False):
        import torch
        z = np.load(Path(derived) / "features.npz", allow_pickle=False)
        sc = json.loads((Path(derived) / "scalers.json").read_text())
        obj = z["obj"].copy()
        mu, sd = np.array(sc["mean"], np.float32), np.array(sc["std"], np.float32)
        obj = (obj - mu) / sd * z["mask"][..., None]
        if geometry_only:                                         # ablation: drop object shape / size / height (keep positions + occupancy)
            obj[..., [0, 1, 2, 3]] = 0.0
        T = lambda a, dt=None: torch.as_tensor(a, dtype=dt, device=device)
        self.obj, self.mask, self.occ, self.legal = T(obj, torch.float32), T(z["mask"]), T(z["occ"], torch.float32), T(z["legal"])
        self.dest, self.label, self.split, self.ctx = T(z["dest"], torch.long), T(z["label"], torch.long), T(z["split"], torch.long), T(z["ctx"], torch.long)
        self.meta = {k: z[k] for k in ("qid", "root", "shape", "height_bin", "block", "band", "depth", "n_obj", "cheb", "status", "src")}
        self.n, self.device = len(z["dest"]), device
        # per-context tables (first row of each context carries the scene features; destinations aggregated)
        ctx = z["ctx"]; C = int(ctx.max()) + 1
        first = np.full(C, -1); [first.__setitem__(c, i) for i, c in enumerate(ctx) if first[c] < 0]
        pos, neg = np.zeros((C, 36), bool), np.zeros((C, 36), bool)
        for i, c in enumerate(ctx):
            if z["label"][i] == 1:
                pos[c, z["dest"][i]] = True
            elif z["label"][i] == 0:
                neg[c, z["dest"][i]] = True
        self.c_row, self.c_pos, self.c_neg = T(first, torch.long), T(pos), T(neg)
        self.c_split = self.split[self.c_row]

    def rows(self, split, positive_only=False):
        import torch
        m = self.split == SPLITS.index(split)
        if positive_only:
            m = m & (self.label == 1)
        return torch.nonzero(m).squeeze(1)

    def contexts(self, split, need_positive=False):
        import torch
        m = self.c_split == SPLITS.index(split)
        if need_positive:
            m = m & self.c_pos.any(1)
        return torch.nonzero(m).squeeze(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "featurize", "audit", "attach-witnesses"])
    ap.add_argument("--incoming"); ap.add_argument("--out"); ap.add_argument("--snapshot"); ap.add_argument("--report")
    a = ap.parse_args()
    if a.cmd == "build":
        build(a.incoming, a.out)
    elif a.cmd == "attach-witnesses":
        attach_witnesses(a.snapshot, a.incoming)
    elif a.cmd == "featurize":
        featurize(a.snapshot)
    else:
        audit(a.snapshot, a.report)
