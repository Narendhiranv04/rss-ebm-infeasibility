"""EXPLORATORY helper for em4m_test3d.sh: the team's E-M4M (branch naren/analytical-baselines-k3-finalize, config paper, seed 0,
300 CPU-s) on a 3D split of this Mac.  Their planner code runs unchanged; this file only does the two Mac-specific pieces.

    python em4m_test3d.py paths SPLIT OUT.json        {root_id: local raw-root path} for run.py --raw-paths (gvlab path -> runs/ABF3 copy)
    python em4m_test3d.py verify SPLIT RUNDIR WORKERS  their eval.sh steps (seal -> validate -> teach -> evaluate -> summarize), on the Mac

`verify` = benchmark.py seal / validate / evaluate / summarize unchanged; only `queries + teach` is replaced by hz_eval._run (same
evaluate._verify_job behind construct._boot, minus Linux sched_setaffinity) with LOCAL root paths.  Teacher cache = RUNDIR/../cache
(fresh, like their $AB_OUT/cache).  Writes RUNDIR/predictions_3d_results.jsonl (+ SUMMARY_3d.json), the format vs_classical.py reads."""
import json, os, sys
from pathlib import Path

OUTROOT = Path(os.environ["EM4M_OUT"])                      # set by em4m_test3d.sh; teacher scratch stays out of runs/hz
os.environ.setdefault("V3OUT", str(OUTROOT / "teach_out"))
os.environ.setdefault("FIN3D_STORE", str(OUTROOT / "teach_out/fin3d_store"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402

BM, DOM = H.BM, "3d"


def pub_of(split):
    return H.K3 / "runs/public" / f"public_{split}_{DOM}.jsonl"


def paths(split, out):
    m = {rid: H.local(e["path"]) for rid, e in H.priv_of(DOM, split).items()}
    miss = [p for p in m.values() if not Path(p).exists()]
    assert not miss, f"{len(miss)} raw roots missing locally, e.g. {miss[:2]}"
    Path(out).write_text(json.dumps(m))
    print(json.dumps({"split": split, "roots": len(m)}))


def verify(split, rundir, workers):
    pred, pub, cache = Path(rundir) / f"predictions_{DOM}.jsonl", pub_of(split), Path(rundir).parent / "cache"
    cache.mkdir(exist_ok=True)
    if not Path(str(pred) + ".SEAL.json").exists():
        BM.seal(pred, pub)                                  # sealed before any teacher call (chmod 444)
    BM.validate(pub, pred)
    priv, (L, _) = H.priv_of(DOM, split), BM.cache_labels(cache, DOM)
    jobs = []
    for r in BM.jl(pred):
        if r["decision"] != "REPAIR" or None not in L.get((r["root_id"], json.dumps(BM.seq_of(r))), (None, None)):
            continue
        e = priv[r["root_id"]]
        jobs.append({"root_id": r["root_id"], "dom": DOM.upper(), "path": H.local(e["path"]), "geom": e.get("geom", "original"),
                     "ids": e["ids"], "seqs": [[[m["object_id"], int(m["destination_cell"])] for m in r["moves"]]]})
    print(json.dumps({"teacher_queries": len(jobs)}), flush=True)
    if jobs:
        H._run(DOM, jobs, workers, cache / f"verify_{DOM}.jsonl")
    res = Path(rundir) / f"predictions_{DOM}_results.jsonl"
    idx = H.K3 / "runs/public" / f"private_{split}_{DOM}.jsonl"
    BM.evaluate(pub, pred, idx, cache, res)
    BM.summarize(idx, Path(rundir) / f"SUMMARY_{DOM}.json", [res])


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "paths":
        paths(sys.argv[2], sys.argv[3])
    elif c == "verify":
        verify(sys.argv[2], sys.argv[3], int(sys.argv[4]))
    else:
        raise SystemExit(__doc__)
