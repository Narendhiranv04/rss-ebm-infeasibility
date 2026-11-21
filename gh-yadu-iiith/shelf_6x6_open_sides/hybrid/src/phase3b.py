"""Phase 3b: the team's sequential pipeline and a retry budget, on Phase 3's 90 contexts / 160 roots.  VAL only; rules fixed before any
Phase 3b physics result was seen (Phase 3's partial plan results had been seen: 47 roots).

Naming used in reports: "boltzmann" = the X/Y Boltzmann model D (3-seed mean p11; Phase 3's "critic"), "no_structure" = DIRECT-GEO.

pipeline        the team's design (study.propose 'cvae+rank_P2', r3 manifest): cvae_s0 draws 64 cells, the first 8 unique form the pool,
                ranker_s0 keeps its top 4, and the Boltzmann model orders those 4.
                A: the 4 cells in Boltzmann order.  B: the Phase 3 beam search, but each object may only go to its 4 pipeline cells.
boltzmann_retry B only: the Phase 3 Boltzmann search's plans in p11 order; the next plan is executed only if every earlier one failed,
                at most 5 plans per root (a physics-call budget, for comparison with REAP, which checks moves with the teacher while it searches).

    python ../hybrid/src/phase3b.py picks | run --workers 6 | summary         (after phase3.py run has finished)
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase3 as P3  # noqa: E402

OUT = P3.OUT.parent / "PHASE3B"
POOL, M, RETRIES = 8, 4, 5
NAMES = {"critic": "boltzmann", "critic_s0": "boltzmann_s0", "critic_s1": "boltzmann_s1", "critic_s2": "boltzmann_s2",
         "v1b_ranker": "boltzmann+ranker_veto", "direct": "no_structure", "critic_plan": "boltzmann_plan", "direct_plan": "no_structure_plan",
         "cvae_cells": "boltzmann_objects+cvae_cells"}
# REAP (yadu/REAP RESULTS.md @671f419ed9, 24 h label, 1 seed): V4 TEST scenes (not these), teacher-checked search, ~51 checks and ~150 s per scene
REAP_TASK_BY_K = {1: 90.6, 2: 79.1, 3: 57.8}


def propose(Mm, raw, cells, o, gen, exclude=()):
    """the team's 'cvae+rank_P2' proposal for one object in one (hypothetical) scene -> up to 4 cells, Ranker order"""
    import train as TR
    torch, np, D = Mm.torch, Mm.np, Mm.V.D
    f, msk, occ, leg = D.scene_features(D.allowed(raw)["objects"], cells, o)
    f = (f - Mm.mu) / Mm.sd * msk[:, None]
    for r, c in exclude:
        leg[r * D.COLS + c] = False
    X = (torch.as_tensor(f)[None], torch.as_tensor(msk)[None], torch.as_tensor(occ, dtype=torch.float32)[None])
    with torch.no_grad():
        d, _ = Mm.cvaes[0].sample(*X, torch.as_tensor(leg)[None], 8 * POOL, gen=gen)
        pool = TR.first_unique(d, POOL)[0]
        lg = Mm.rankers[0].logits_all(*X).masked_fill(~pool, -1e9)[0]
    return [(i // D.COLS, i % D.COLS) for i in lg.topk(min(M, int(pool.sum()))).indices.tolist()]


def search(Mm, raw, cells0, moves, score=None):
    """Phase 3's beam search with a pluggable move generator and scorer (default: seed-mean Boltzmann p11) -> {plan: score}"""
    beam, scored = [()], {}
    for _ in range(P3.DEPTH):
        cands = [seq + (m,) for seq in beam for m in moves({**cells0, **dict(seq)}, {o for o, _ in seq})]
        if not cands:
            break
        p = score(raw, cands) if score else Mm.p11(raw, cands).mean(0)
        scored.update(zip(cands, p.tolist()))
        beam = [cands[k] for k in p.topk(min(P3.BEAM, len(cands))).indices.tolist()]
    return scored


def cmd_picks(a):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "picks.json"
    if f.exists():
        return json.loads(f.read_text())
    torch = __import__("torch")
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(0)
    Mm = P3.Models()
    out = {"ctx": [], "plan": []}
    for p in json.loads((P3.OUT / "picks_ctx.json").read_text()):
        raw = Mm.load(p["root"])
        prefix = [(o, tuple(c)) for o, c in p["prefix"]]
        cells = {**{k: tuple(v) for k, v in raw["initial_cells"].items()}, **dict(prefix)}
        cand = propose(Mm, raw, cells, p["object"], gen)
        pp = Mm.p11(raw, [prefix + [(p["object"], c)] for c in cand]).mean(0)
        out["ctx"].append({"ctx": p["ctx"], "pipeline": [list(cand[j]) for j in pp.argsort(descending=True).tolist()]})
    for p in json.loads((P3.OUT / "picks_plan.json").read_text()):
        raw, t0 = Mm.load(p["root"]), time.time()
        goal, cells0 = tuple(raw["goal"]), {k: tuple(v) for k, v in raw["initial_cells"].items()}
        objs = sorted(o for o in cells0 if o != "t0")
        free = lambda cells: [(r, c) for r in range(6) for c in range(6) if (r, c) not in set(cells.values()) and (r, c) != goal]  # noqa: E731
        allm = search(Mm, raw, cells0, lambda cells, moved: [(o, d) for o in objs if o not in moved for d in free(cells)])
        ranked = sorted(allm, key=allm.get, reverse=True)[:RETRIES]
        J = lambda s: [[o, list(c)] for o, c in s]  # noqa: E731
        assert J(ranked[0]) == p["critic_plan"] or abs(allm[ranked[0]] - p["critic_p11"]) < 1e-4, p["root"]  # same search as Phase 3
        pipe = search(Mm, raw, cells0, lambda cells, moved: [(o, d) for o in objs if o not in moved for d in propose(Mm, raw, cells, o, gen, {goal})])
        best = max(pipe, key=pipe.get)
        J = lambda s: [[o, list(c)] for o, c in s]  # noqa: E731
        out["plan"].append({"root": p["root"], "retry": [J(s) for s in ranked], "retry_p11": [round(allm[s], 4) for s in ranked],
                            "pipeline_plan": J(best), "pipeline_p11": round(pipe[best], 4), "s": round(time.time() - t0, 1)})
        print(json.dumps({"root": p["root"], "K": p["K"], "pipeline_plan": J(best), "s": out["plan"][-1]["s"]}), flush=True)
    f.write_text(json.dumps(out, indent=1))
    return out


def results():
    res = P3.results()
    f = P3.OUT.parent / "PHASE3B/results.jsonl"  # fixed path: later phases repoint OUT for run_jobs
    if f.exists():
        res.update({json.dumps(r["job"]): r for r in map(json.loads, f.read_text().splitlines())})
    return res


def run_jobs(todo, workers):
    with ProcessPoolExecutor(workers) as ex, (OUT / "results.jsonl").open("a") as f:
        for i, (job, res) in enumerate(ex.map(P3._run, todo), 1):
            f.write(json.dumps({"job": job, **res}) + "\n"); f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done", flush=True)


def cmd_run(a):
    pk = cmd_picks(a)
    ctx, plans = json.loads((P3.OUT / "picks_ctx.json").read_text()), json.loads((P3.OUT / "picks_plan.json").read_text())
    jobs = {json.dumps([p["root"], q["pipeline_plan"]]): [p["root"], q["pipeline_plan"]] for p, q in zip(plans, pk["plan"])}
    for p, q in zip(ctx, pk["ctx"]):
        for c in q["pipeline"]:
            jobs.setdefault(json.dumps(P3.ctx_job(p, c)), P3.ctx_job(p, c))
    done = results()
    todo = [j for k, j in jobs.items() if k not in done]
    print(json.dumps({"pipeline_jobs": len(jobs), "todo": len(todo)}), flush=True)
    run_jobs(todo, a.workers)
    for r in range(1, 1 if os.environ.get("NO_RETRY") else RETRIES):  # retry round r; NO_RETRY=1: superseded by phase4's lazy retries
        res = results()
        todo = [[q["root"], q["retry"][r]] for q in pk["plan"] if len(q["retry"]) > r
                and not any(res[json.dumps([q["root"], s])]["success"] for s in q["retry"][:r])
                and json.dumps([q["root"], q["retry"][r]]) not in res]
        print(json.dumps({"retry_round": r + 1, "todo": len(todo)}), flush=True)
        run_jobs(todo, a.workers)
    cmd_summary(a)


def cmd_summary(a):
    res, s3 = results(), {}
    ok = lambda job: res.get(json.dumps(job), {}).get("success")  # noqa: E731
    pk = json.loads((OUT / "picks.json").read_text())
    ctx, plans = json.loads((P3.OUT / "picks_ctx.json").read_text()), json.loads((P3.OUT / "picks_plan.json").read_text())
    for p, q in zip(ctx, pk["ctx"]):
        p["pipeline"] = q["pipeline"]
    for name, sub in (("A_all_90", ctx), ("A_fresh_60", [p for p in ctx if not p["phase2"]])):
        if any(ok(P3.ctx_job(p, c)) is None for p in sub for k in P3.CTX_PICKERS + ("pipeline",) for c in p[k]):
            s3[name] = "physics incomplete"; continue
        n, per, s = len(sub), {}, {"contexts": len(sub)}
        for k in P3.CTX_PICKERS + ("pipeline",):
            top1, r4 = [ok(P3.ctx_job(p, p[k][0])) for p in sub], [[ok(P3.ctx_job(p, c)) for c in p[k]] for p in sub]
            per[NAMES.get(k, k)] = top1
            s[NAMES.get(k, k)] = {"top1_works": f"{sum(top1)}/{n}", "top1_95ci": P3.wilson(sum(top1), n), "any_of_top4": f"{sum(max(r) for r in r4)}/{n}",
                                  "share_of_top4": round(sum(sum(r) / len(r) for r in r4) / n, 3)}
        s["top1_head_to_head"] = P3.compare(per, [("boltzmann", "pipeline"), ("pipeline", "cvae"), ("boltzmann", "cvae"), ("boltzmann", "random"),
                                                  ("boltzmann", "ranker"), ("boltzmann", "no_structure"), ("no_structure", "cvae"),
                                                  ("boltzmann+ranker_veto", "boltzmann")])
        s3[name] = s
    for p, q in zip(plans, pk["plan"]):
        p["pipeline_plan"] = q["pipeline_plan"]
        tries = [ok([p["root"], x]) for x in q["retry"]]
        first = next((i for i, v in enumerate(tries) if v), None)
        p["retry_ok"], p["tries"] = int(first is not None), (first + 1 if first is not None else len([v for v in tries if v is not None]))
    keys = P3.PLAN_PICKERS + ("pipeline_plan",)
    per = {NAMES.get(k, k): [ok([p["root"], p[k]]) for p in plans] for k in keys}
    per["boltzmann_up_to_5_plans"] = [p["retry_ok"] for p in plans]
    s = {"roots": len(plans)}
    for k, v in per.items():
        s[k] = {"works": f"{sum(v)}/{len(v)}", "95ci": P3.wilson(sum(v), len(v)),
                "by_K_%": {K: round(100 * sum(x for x, p in zip(v, plans) if p["K"] == K) / sum(p["K"] == K for p in plans), 1) for K in (1, 2, 3)}}
    s["boltzmann_up_to_5_plans"]["mean_plans_tried"] = round(sum(p["tries"] for p in plans) / len(plans), 2)
    s["REAP_published_task_%_by_K (V4 TEST scenes, ~42 teacher checks/scene; context only)"] = REAP_TASK_BY_K
    s["head_to_head"] = P3.compare(per, [("boltzmann_plan", "pipeline_plan"), ("pipeline_plan", "boltzmann_objects+cvae_cells"),
                                         ("boltzmann_plan", "no_structure_plan"), ("boltzmann_plan", "random_plan"), ("boltzmann_plan", "boltzmann_objects+cvae_cells")])
    import collections
    s["where_plans_fail"] = {NAMES.get(k, k): dict(collections.Counter(
        ("target" if r["failed_step"] == len(p[k]) else f"move{r['failed_step'] + 1}") + ":" + r["why"].split(":")[0]
        for p in plans for r in [res.get(json.dumps([p["root"], p[k]]))] if r and not r["success"])) for k in keys}
    s3["B_plan_search"] = s
    print(json.dumps(s3, indent=1))
    (OUT / "summary.json").write_text(json.dumps(s3, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["picks", "run", "summary"]); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"picks": cmd_picks, "run": cmd_run, "summary": cmd_summary}[a.cmd](a)
