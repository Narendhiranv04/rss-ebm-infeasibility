"""Phase 2: do the pickers' top cells actually work in MuJoCo?  VAL only; rules fixed before any physics result was seen.

Contexts: 30 VAL contexts (root, labelled prefix, object moved last) that have >= 1 known-good cell, sampled with seed 0.
Pickers, top M = 4 cells each:
    random   4 legal cells uniformly at random (seed = context index)
    cvae     highest q_cVAE(d | S, o)
    critic   highest p11 of frozen critic D
    v1b      critic order after a cVAE veto: cells with q < 1 / (4 n) are dropped (n = legal cells); vetoed cells only fill a short list
Each (context, cell) runs the full teacher via hybrid/physics.execute(prefix + (o, cell)); shared cells run once; results append to a JSONL.

    python ../hybrid/src/phase2.py run --workers 6        (run from shelf_6x6_open_sides/; resumable)
    python ../hybrid/src/phase2.py summary
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # shelf_6x6_open_sides/
OUT = ROOT / "approachb_3d_learning/runs/PHASE2"
N_CTX, M, ARM, PICKERS = 30, 4, "D", ("random", "cvae", "critic", "v1b")


def plan_jobs():
    import math
    import torch
    sys.path.insert(0, str(HERE))
    import v1_check as V
    sc = json.loads((V.A / "runs/XY3D_REAL/BD3DS66_XY_SIDECAR_V1.json").read_text())
    ctxs = [(k, lab) for k, lab in V.val_contexts(sc).items() if any(lab.values())]
    ctxs.sort(key=lambda x: (x[0][0], str(x[0][1]), x[0][2]))
    random.Random(0).shuffle(ctxs)
    score = V.scorer(ARM)
    picks = []
    for i, ((rid, prefix, o), lab) in enumerate(ctxs[:N_CTX]):
        cand, lq, p11 = score(rid, prefix, o)
        n = len(cand)
        veto = lq < math.log(1 / (4 * n))
        top = lambda s: [cand[j] for j in s.topk(min(M, n)).indices.tolist()]  # noqa: E731
        picks.append({"ctx": i, "root": rid, "prefix": [[a, list(c)] for a, c in prefix], "object": o, "n_legal": n,
                      "known_good": [list(c) for c, s in lab.items() if s == 1], "vetoed": int(veto.sum()),
                      "random": [list(c) for c in random.Random(i).sample(cand, min(M, n))],
                      "cvae": [list(c) for c in top(lq)], "critic": [list(c) for c in top(p11.log())],
                      "v1b": [list(c) for c in top(p11.log() - 1e3 * veto.float())]})
    return picks


def _run(job):
    os.environ.setdefault("MUJOCO_GL", "disable")
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "approachb_3d_learning/src"))
    import physics, xy3d_data as XD
    rid, steps = job
    return job, physics.execute(XD.load(rid), [(o, tuple(c)) for o, c in steps])


def cmd_run(a):
    OUT.mkdir(parents=True, exist_ok=True)
    pf = OUT / "picks.json"
    if not pf.exists():
        pf.write_text(json.dumps(plan_jobs(), indent=1))
    picks = json.loads(pf.read_text())
    rf = OUT / "results.jsonl"
    done = {json.dumps(json.loads(l)["job"]) for l in rf.read_text().splitlines()} if rf.exists() else set()
    jobs = {}
    for p in picks:
        for k in PICKERS:
            for c in p[k]:
                job = [p["root"], p["prefix"] + [[p["object"], c]]]
                jobs.setdefault(json.dumps(job), job)
    todo = [j for key, j in jobs.items() if key not in done]
    print(json.dumps({"contexts": len(picks), "unique_physics_jobs": len(jobs), "already_done": len(jobs) - len(todo)}), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, rf.open("a") as f:
        futs = [ex.submit(_run, j) for j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            job, res = fu.result()
            f.write(json.dumps({"job": job, **res}) + "\n"); f.flush()
            if i % 10 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done, {time.time() - t0:.0f}s elapsed", flush=True)
    cmd_summary(a)


def cmd_summary(a):
    picks = json.loads((OUT / "picks.json").read_text())
    res = {json.dumps(r["job"]): r for r in map(json.loads, (OUT / "results.jsonl").read_text().splitlines())}
    ok = lambda p, c: res.get(json.dumps([p["root"], p["prefix"] + [[p["object"], c]]]), {}).get("success")  # noqa: E731
    out, per = {"contexts": len(picks), "M": M, "critic_arm": ARM}, {}
    for k in PICKERS:
        top1 = [ok(p, p[k][0]) for p in picks]
        any4 = [max(ok(p, c) for c in p[k]) for p in picks]
        frac = [sum(ok(p, c) for c in p[k]) / len(p[k]) for p in picks]
        per[k] = top1
        out[k] = {"top1_works": f"{sum(top1)}/{len(picks)}", "any_of_top4_works": f"{sum(any4)}/{len(picks)}",
                  "share_of_top4_that_work": round(sum(frac) / len(picks), 3)}
    for x, y in (("critic", "cvae"), ("v1b", "critic"), ("v1b", "cvae"), ("critic", "random")):
        w = sum(a_ > b_ for a_, b_ in zip(per[x], per[y])); l = sum(a_ < b_ for a_, b_ in zip(per[x], per[y]))
        out[f"top1 {x} vs {y}"] = f"{x} wins {w}, loses {l}, ties {len(picks) - w - l}"
    times = [r["s"] for r in res.values()]
    out["physics_jobs"], out["mean_s_per_job"] = len(times), round(sum(times) / max(1, len(times)), 1)
    print(json.dumps(out, indent=1))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"]); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary}[a.cmd](a)
