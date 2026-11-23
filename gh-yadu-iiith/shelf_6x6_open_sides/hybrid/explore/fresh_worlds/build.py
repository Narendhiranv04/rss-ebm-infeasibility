#!/usr/bin/env python3
"""Benchmark rows for the fresh roots, in the format of K3ABL/runs/public/{public,private}_test_3d.jsonl.  No method is run, nothing is scored.

Per candidate root (accepted, counted class, generator ledger order) one QC record, cached in <out>/qc.jsonl (resumable), made with the
evaluation teacher's own code (fin3d, booted as evaluate._boot does):
  * provenance = the published roots' (config hash, generator, teacher, label protocol); <= 16 objects (V3.4 MAXOBJ for 3D)
  * no fixed object: fin3d.fixed_one = none_dense.pickable_alone (the published no-fixed-object rule)
  * K0: Y(empty) = fin3d.run_y on the unchanged scene (= evaluate._verify_job's Y for the empty sequence; scripts/v34_gt.py)
Disjointness (asserted, never filtered): root id, construction seed and layout differ from every TRAIN / VAL / TEST root.
Selection: per class the first n QC-passing roots in ledger order (as run_shelf66_full.select), n = generate.targets(--n).  K0 counts only
with Y(empty) = 1 (a K0 with Y(empty) != 1 is in no denominator of the team's index either).
Masks: eval_task K1-K3, eval_noop K0, eval_abstain NONE; eval_repair False (no finalized X/Y certificate here), train_prediction False.
evaluation_group_id = root_id.  Outputs (never overwritten): <out>/public_fresh_3d.jsonl, private_fresh_3d.jsonl, FRESH_MANIFEST.json.
    python build.py [--workers 4] [--n 250] [--quota JSON] [--out DIR] [--allow-short]"""
import argparse, collections, hashlib, json, os, sys, time
from pathlib import Path

import generate as G

PROV = {"config_hash": "8d521b1f6d23db23", "label_protocol": "BD_BINARY_BUDGET_V1"}
GEN_PLUS, TEACHER = "BD_GEN_V1+ROUTE_V15+SHELF66V3", "BD3DS66_TEACHER_V2:"
MAXOBJ = 16
F = {}


def boot(store):
    os.environ.update(FIN3D_SRC=str(G.CODE.parent), FIN3D_CODE=str(G.ABF / "fin3d/code"), FIN3D_STORE=store, MUJOCO_GL="disable")
    sys.path.insert(0, os.environ["FIN3D_CODE"])
    import fin3d
    fin3d.boot()
    F["F"] = fin3d


def qc_one(c):
    fin = F["F"]
    assert "boot_error" not in fin.Q._M, fin.Q._M.get("boot_error")
    rec = json.loads(Path(c["path"]).read_text())
    ids = rec["ids"]
    prov = [k for k, v in PROV.items() if ids.get(k) != v] + ([] if rec.get("generator_version_plus") == GEN_PLUS else ["generator"]) + \
           ([] if ids.get("teacher_version", "").startswith(TEACHER) else ["teacher"])
    out = {**c, "seed": int(ids["construction_seed"]), "layout": G.layout_key(rec), "n_obj": len(rec["objects"]), "provenance_bad": prov,
           "fixed": fin.capped(fin.fixed_one, rec)}
    if c["K"] == "0":
        out["y_empty"] = fin.capped(fin.run_y, rec, {k: list(v) for k, v in rec["initial_cells"].items()})
    return out


def resolved(q):
    return q["fixed"].get("resolution") == "EVALUATED" and (q["K"] != "0" or (q.get("y_empty") or {}).get("resolution") == "EVALUATED")


def passes(q):
    ok = not q["provenance_bad"] and q["n_obj"] <= MAXOBJ and q["fixed"].get("resolution") == "EVALUATED" and not q["fixed"].get("fixed_objects")
    if q["K"] == "0":
        y = q.get("y_empty") or {}
        ok = ok and y.get("resolution") == "EVALUATED" and y.get("y") == 1
    return ok


def candidates(gen):
    """accepted roots of a counted class, generator ledger order: K0-K3 run first, then the trimmed NONE run."""
    out = []
    for sub, keep in (("k", ("0", "1", "2", "3")), ("none_trim", ("NONE",))):
        led = gen / sub / "ledger.jsonl"
        for e in (map(json.loads, open(led)) if led.exists() else ()):
            if e.get("accepted") and str(e.get("observed")) in keep:
                out.append({"root_id": e["root_id"], "K": str(e["observed"]), "path": str(gen / sub / "roots" / f"{e['root_id']}.json")})
    return out


def write_new(p, text):
    if p.exists():
        if p.read_text() != text:
            raise SystemExit(f"REFUSED: {p} exists with other content (never overwritten; move it away yourself)")
        return
    p.write_text(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--quota", default=None)
    ap.add_argument("--out", default=str(G.OUT))
    ap.add_argument("--allow-short", action="store_true", help="smoke only: write even when a class is short")
    a = ap.parse_args()
    out = Path(a.out)
    gen, qf = out / "gen", out / "qc.jsonl"
    want = G.targets(a.n, json.loads(a.quota) if a.quota else None)
    cand = candidates(gen)
    have = {json.loads(l)["root_id"] for l in open(qf)} if qf.exists() else set()
    todo = [c for c in cand if c["root_id"] not in have]
    print(json.dumps({"candidates": len(cand), "qc_todo": len(todo), "targets": want}), flush=True)
    if todo:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(a.workers, initializer=boot, initargs=(str(out / "qc_store"),), maxtasksperchild=20) as pool, open(qf, "a") as f:
            for i, q in enumerate(pool.imap_unordered(qc_one, todo), 1):
                if resolved(q):                                     # unresolved QC (timeout / error) is retried on the next run, never cached
                    f.write(json.dumps(q) + "\n"); f.flush()
                print(f"{time.strftime('%H:%M:%S')} qc {i}/{len(todo)} {q['root_id']} K{q['K']} pass={passes(q)}", flush=True)
    Q = {q["root_id"]: q for q in map(json.loads, open(qf))} if qf.exists() else {}
    ids, seeds, lays = G.split_index()
    clash = [q["root_id"] for q in Q.values() if q["root_id"] in ids or q["seed"] in seeds or q["layout"] in lays]
    assert not clash, f"fresh roots that are NOT disjoint from TRAIN/VAL/TEST: {clash[:5]}"
    sel, drop = collections.defaultdict(list), collections.Counter()
    # selection order = a fixed hash of the root id, NOT the ledger (= finishing) order, which would favour fast (easy) roots
    for c in sorted(cand, key=lambda c: hashlib.sha256(("fresh-v1" + c["root_id"]).encode()).hexdigest()):
        q = Q.get(c["root_id"])
        if q is None:
            drop[f"{c['K']}:qc_unresolved"] += 1
        elif not passes(q):
            drop[c["K"]] += 1
        elif len(sel[c["K"]]) < want.get(c["K"], 0):
            sel[c["K"]].append(q)
    got = {k: len(sel[k]) for k in want}
    short = {k: want[k] - got[k] for k in want if got[k] < want[k]}
    if short and not a.allow_short:
        raise SystemExit(f"not enough QC-passing roots yet (missing {short}); keep generating (or --allow-short for a smoke)")
    sys.path.insert(0, str(G.ABF / "code_v34"))
    import contracts as CT                                          # view3d = the learner inputs of every existing public row
    pub, priv = [], []
    for k, qs in sel.items():
        for q in qs:
            rec = json.loads(Path(q["path"]).read_text())
            v = CT.view3d(rec)
            pub.append({"root_id": q["root_id"], "domain": "3d", "inputs": v})
            priv.append({"root_id": q["root_id"], "domain": "3d", "K": k, "path": q["path"], "geom": "original", "ids": v["ids"],
                         "evaluation_group_id": q["root_id"], "source": "FRESH_WORLDS_V1",
                         "masks": {"eval_task": k in ("1", "2", "3"), "eval_noop": k == "0", "eval_abstain": k == "NONE", "eval_repair": False,
                                   "train_prediction": False}})
    jl = lambda rows: "".join(json.dumps(r, sort_keys=True) + "\n" for r in sorted(rows, key=lambda r: r["root_id"]))  # noqa: E731
    tp, tv = jl(pub), jl(priv)
    write_new(out / "public_fresh_3d.jsonl", tp)
    write_new(out / "private_fresh_3d.jsonl", tv)
    man = {"label": "FRESH 3D roots (new generator names, disjoint from TRAIN/VAL/TEST; asserted)", "targets": want, "selected": got, "short": short,
           "qc_dropped_by_class": dict(drop), "candidates": len(cand), "worlds": len({r["root_id"].rsplit("-", 1)[0] for r in priv}),
           "sha256": {"public_fresh_3d.jsonl": hashlib.sha256(tp.encode()).hexdigest(), "private_fresh_3d.jsonl": hashlib.sha256(tv.encode()).hexdigest()},
           "disjoint_checked_against": {"root_ids": len(ids), "seeds": len(seeds), "layouts": len(lays)}}
    write_new(out / "FRESH_MANIFEST.json", json.dumps(man, indent=1))
    print(json.dumps(man), flush=True)


if __name__ == "__main__":
    main()
