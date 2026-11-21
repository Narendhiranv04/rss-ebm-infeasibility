"""E7 physics (pre-registered in hybrid/docs/EXPERIMENT_LOG.md): #1 cell of each goal-aware cVAE variant on Phase 3's 90 placement contexts,
full teacher in MuJoCo; paired sign tests vs `base`, Holm-corrected over the 6 comparisons.  VAL, exploratory.

    python ../hybrid/src/e7_physics.py run --workers 2 | summary          (run from approachb_3d_learning/)
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase3 as P3, phase3b as P3B, phase4 as P4  # noqa: E402

OUT = P3.OUT.parent / "GCVAE_ABL"
VARIANTS = ("base", "beta1", "pessimistic", "hazard", "agree", "anchored", "nolatent")


def results():
    res = P4.results()
    f = OUT / "results.jsonl"
    if f.exists():
        res.update({json.dumps(r["job"]): r for r in map(json.loads, f.read_text().splitlines())})
    return res


def jobs():
    ctx, V = json.loads((P3.OUT / "picks_ctx.json").read_text()), json.loads((OUT / "picks_ctx.json").read_text())["variants"]
    return ctx, V, {v: [P3.ctx_job(p, V[v][i][0]) for i, p in enumerate(ctx)] for v in VARIANTS}


def holm(ps):
    order, out, run = sorted(range(len(ps)), key=lambda i: ps[i]), [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i])); out[i] = round(run, 4)
    return out


def cmd_run(a):
    _, _, J = jobs()
    res = results()
    todo = {json.dumps(j): j for js in J.values() for j in js if json.dumps(j) not in res}
    print(json.dumps({"e7_jobs": len(todo)}), flush=True)
    P3B.OUT = OUT
    P3B.run_jobs(list(todo.values()), a.workers)
    cmd_summary(a)


def cmd_summary(a):
    _, _, J = jobs()
    res = results()
    per = {v: [res.get(json.dumps(j), {}).get("success") for j in js] for v, js in J.items()}
    if any(x is None for v in per.values() for x in v):
        print("physics incomplete"); return
    out = {v: {"top1_works": f"{sum(x)}/90", "95ci": P3.wilson(sum(x), 90)} for v, x in per.items()}
    cmp = P3.compare(per, [(v, "base") for v in VARIANTS if v != "base"])
    for (k, c), p in zip(cmp.items(), holm([c["sign_test_p"] for c in cmp.values()])):
        c["holm_p"] = p
    out["vs_base"] = cmp
    print(json.dumps(out, indent=1))
    (OUT / "physics_summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    assert holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"]); ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary}[a.cmd](a)
