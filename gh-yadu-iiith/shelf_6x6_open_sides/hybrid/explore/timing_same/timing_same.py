"""EXPLORATORY timing, 3D TEST public inputs only (the 370 roots E-M4M was timed on).  No labels or outcomes are read: the only TEST files
opened are the public inputs and the already-recorded PICKS (ours pool_3d_test_s0*, FULL_s0 test_3d predictions) for the self-check.

Both arms run their FULL decision in one process on 1 CPU thread (OMP / MKL / torch = 1), models loaded once (load time reported apart):
  cvae   search.Root + exhaustive.proposals (frozen bundle task-cVAE, proposal seed 0) -> top-4 per blocker        (shared by both arms)
  ours   exhaustive.score_root(hazard scorer model_3d_s0.pt, frozen rule noop_gate 0.5 / eta 0.3): enumerate K1-K3 + score + pick
  xynet  exhaustive.score_root(FULL_s0 X/Y scorer = bundle scorer_s0.pt, same rule)  (= ablations_k3/run.py predict, on CPU)
  ours_total = cvae + ours, xynet_total = cvae + xynet.  Wall (perf_counter) and CPU (process_time) seconds per stage.
The first WARM roots are run once untimed before timing starts; every root (those included) is then timed.

    python timing_same.py run OUTDIR [--limit N]     -> OUTDIR/timing_test_3d.jsonl (append, resumes) + OUTDIR/load.json
    python timing_same.py summary OUTDIR             -> OUTDIR/summary.json
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"                                         # before numpy / torch load
import json, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SH = HERE.parents[2]                                             # shelf_6x6_open_sides/
sys.path.insert(0, str(SH / "hybrid/src"))
import torch  # noqa: E402
torch.set_num_threads(1); torch.set_num_interop_threads(1)
import twod_hazard as HZ  # noqa: E402  (sets K3 / ABF env + V3.4 code path, XY_DEV=cpu)
sys.path.insert(0, str(HZ.CODE / "ablations_k3"))
import benchmark as BM, exhaustive as EX, search as S  # noqa: E402
import run as RUN  # noqa: E402  (ablations_k3/run.py: FULL arm loader, rule, pool_hash)

WARM = 3
LABEL = "EXPLORATORY timing; split: 3D TEST public inputs (370 roots), no labels / outcomes read"


def clock(fn):
    w, c = time.perf_counter(), time.process_time()
    out = fn()
    return out, time.perf_counter() - w, time.process_time() - c


def recorded():
    """already-recorded TEST picks (decisions only): ours frozen rule = decisions[0]; XYNet = FULL_s0 decision + moves."""
    ours = {}
    for p in sorted(HZ.OUT.glob("pool_3d_test_s0*.jsonl")):
        ours.update({r["root_id"]: r["decisions"][0] for r in BM.jl(p)})
    xy = {}
    for r in BM.jl(HZ.K3 / "runs/predictions/FULL_s0/test_3d.jsonl"):
        xy[r["root_id"]] = (None if r["decision"] == "ABSTAIN" else [[m["object_index"], m["destination_cell"]] for m in r["moves"]], r["pool_hash"],
                            {int(o): sorted(v) for o, v in r["proposals"].items()})
    return ours, xy


def cvae(r, samp, man):
    root = S.Root(r, "3d")
    return root, {o: v[:4] for o, v in EX.proposals(root, samp, man["config"]["proposal_seed"]).items()}


def decide(r, samp, man, hz, xy):
    t = {}
    (root, D), t["cvae_wall"], t["cvae_cpu"] = clock(lambda: cvae(r, samp, man))
    ro, t["ours_wall"], t["ours_cpu"] = clock(lambda: EX.score_root(root, D, hz, [HZ.FROZEN], 4, set(), 8192))
    rx, t["xynet_wall"], t["xynet_cpu"] = clock(lambda: EX.score_root(root, D, xy, [RUN.RULE], 4, set(), 8192))
    for a in ("ours", "xynet"):
        for k in ("wall", "cpu"):
            t[f"{a}_total_{k}"] = t[f"cvae_{k}"] + t[f"{a}_{k}"]
    pick = lambda rec: (lambda d: None if d is None else d["seq"])(rec["per_M"]["4"]["decisions"][0])  # noqa: E731
    return {"root_id": r["root_id"], "n_movable": len(root.movable), "n_scored": ro["per_M"]["4"]["n_scored"], "pool_hash": RUN.pool_hash(D, 4),
            "pool_sets": {o: sorted(v) for o, v in D.items()},
            "ours_pick": pick(ro), "xynet_pick": pick(rx), **{k: round(v, 5) for k, v in t.items()}}


def run(outdir, limit=None):
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    fo = out / "timing_test_3d.jsonl"
    rows = BM.jl(HZ.K3 / "runs/public/public_test_3d.jsonl")[:limit]
    done = {r["root_id"] for r in BM.jl(fo)} if fo.exists() else set()
    (hz, lw_h, lc_h) = clock(lambda: HZ.HazardScorer(HZ.OUT / "model_3d_s0.pt"))
    (ld, lw_x, lc_x) = clock(lambda: RUN.load("FULL_s0", "3d", "cpu"))
    _, samp, xy, _, man, files = ld
    (out / "load.json").write_text(json.dumps({"label": LABEL, "hazard_load_wall_s": round(lw_h, 4), "hazard_load_cpu_s": round(lc_h, 4),
                                               "cvae_plus_xynet_load_wall_s": round(lw_x, 4), "cvae_plus_xynet_load_cpu_s": round(lc_x, 4),
                                               "torch_threads": torch.get_num_threads(), "torch": torch.__version__, "host_cpu": platform.processor(),
                                               "machine": platform.machine(), "python": sys.version.split()[0], **files}, indent=1))
    for r in rows[:WARM]:
        decide(r, samp, man, hz, xy)                             # warm-up, untimed, not written
    ours_ref, xy_ref = recorded()
    with open(fo, "a") as f:
        for i, r in enumerate(x for x in rows if x["root_id"] not in done):
            d = decide(r, samp, man, hz, xy)
            d["ours_matches_recorded"] = d["ours_pick"] == ours_ref.get(r["root_id"], "missing")
            d["xynet_matches_recorded"] = d["xynet_pick"] == xy_ref.get(r["root_id"], ("missing",))[0]
            d["pool_hash_matches_recorded"] = d["pool_hash"] == xy_ref.get(r["root_id"], (None, None))[1]   # rank order too (info only)
            d["pool_set_matches_recorded"] = {int(o): v for o, v in d.pop("pool_sets").items()} == xy_ref.get(r["root_id"], (0, 0, None))[2]
            f.write(json.dumps(d) + "\n"); f.flush()
            if i % 25 == 0:
                print(f"{time.strftime('%H:%M:%S')} {i + 1} roots  ours {d['ours_total_cpu']:.3f}s  xynet {d['xynet_total_cpu']:.3f}s cpu", flush=True)
    summary(outdir)


def summary(outdir):
    import numpy as np
    out = Path(outdir)
    R = BM.jl(out / "timing_test_3d.jsonl")
    stages = ["cvae", "ours", "xynet", "ours_total", "xynet_total"]
    st = {f"{s}_{k}": {q: round(float(f(np.array([r[f'{s}_{k}'] for r in R]))), 4) for q, f in
                       (("median", np.median), ("p90", lambda v: np.percentile(v, 90)), ("max", np.max), ("mean", np.mean))}
          for s in stages for k in ("cpu", "wall")}
    bad = {k: [r["root_id"] for r in R if not r[k]] for k in ("ours_matches_recorded", "xynet_matches_recorded", "pool_set_matches_recorded")}
    try:                                                        # self-check: timed picks = recorded TEST picks (report, do not crash)
        assert R and not any(bad.values()), {k: len(v) for k, v in bad.items()}
        check = "PASS"
    except AssertionError as e:
        check = f"MISMATCH {e}"
    s = {"label": LABEL, "n_roots": len(R), "warmup_roots_untimed": WARM, "unit": "seconds per scene; cpu = time.process_time, wall = perf_counter",
         "load": json.loads((out / "load.json").read_text()), "stages": st, "self_check": check, "mismatch_roots": {k: v[:20] for k, v in bad.items()},
         "pool_rank_order_differs_from_gpu_run": sum(not r["pool_hash_matches_recorded"] for r in R),
         "notes": ["cvae is computed once per scene and counted in both totals (both arms use the same frozen cVAE proposals).",
                   "ours uses only the frozen rule here; the recorded pool scored 37 rules in one pass (the pick under rule 0 is the same).",
                   "CPU cVAE draws are deterministic; on a few roots they list the same 4 spots in another order than the GPU run "
                   "(pool_hash differs, same candidate set, same M4 pool). The self-check compares sets and picks.",
                   "E-M4M median 109 s was measured on another machine (lab server); compare machines only with that caveat."]}
    (out / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps({"n": len(R), "self_check": check, **{k: v["median"] for k, v in st.items()}}, indent=1))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "run":
        run(a[0], int(a[a.index("--limit") + 1]) if "--limit" in a else None)
    elif c == "summary":
        summary(a[0])
    else:
        raise SystemExit(__doc__)
