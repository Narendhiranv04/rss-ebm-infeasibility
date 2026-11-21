"""Phase 4: literature-driven fixes, scored as success vs physics checks, on Phase 3's 160 VAL roots (verified_K >= 1).
Rules fixed before any Phase 4 physics result was seen.

Planners (each = the Phase 3 beam search, width 16, depth <= 3, cells != goal, with a different scorer / move generator):
    boltz_mean      Boltzmann model D, mean p11 of 3 seeds (= Phase 3 "critic_plan" search)
    boltz_min       cautious: the MINIMUM p11 over the 3 seeds (ensemble lower bound; Coste et al. 2024, Smith & Winkler 2006)
    hazard          per-step hazard model (hybrid/hazard.py), mean of 3 seeds; only if trained
    factor          E8 factor graph (hybrid/factor.py): hazard move factors x noisy-AND target factor over per-object blockers
    gcvae_pipeline  moves limited to the goal-aware cVAE's 4 proposals per object (hybrid/gcvae.py), scored by boltz_mean; only if trained
Execution = lazy retries (LazySP-style use of failure information): run the top-ranked plan; if it fails at move j, every plan sharing its
first j+1 moves is dropped (chained execution is deterministic, so they would fail the same way); if the target fails, only that exact plan
is dropped; then run the best remaining plan.  At most 5 plans per root.  Reported: success at try 1, success within 5 tries, plans tried,
motion queries (= steps attempted, comparable to REAP's per-move feasibility calls).
Placement contexts (Phase 3's 90): gcvae = the goal-aware cVAE's top 4 cells by log q (3-seed mixture), same scoring as Phase 3's pickers.

    python ../hybrid/src/phase4.py picks [--planners boltz_mean,boltz_min,...] | run --workers 6 | summary
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase3 as P3, phase3b as P3B  # noqa: E402

OUT = P3.OUT.parent / "PHASE4"
TRIES, KEEP = 5, 300
ALPHAS = (0.05, 0.1, 0.2, 0.5, 1.0)  # E10 length-penalty grid, fixed before any E10 physics
ALL = ("boltz_mean", "boltz_min", "hazard", "gcvae_pipeline", "factor", "team_pipeline", "step_team_props", "boltz_ranker_filter", "boltz_B", "outcome_B") + tuple(f"team_len_a{a}" for a in ALPHAS)


def plans_for(name, Mm, raw, cells0, objs, goal, gen, extra):
    free = lambda cells: [(r, c) for r in range(6) for c in range(6) if (r, c) not in set(cells.values()) and (r, c) != goal]  # noqa: E731
    allm = lambda cells, moved: [(o, d) for o in objs if o not in moved for d in free(cells)]  # noqa: E731
    if name == "boltz_mean":
        return P3B.search(Mm, raw, cells0, allm)
    if name == "boltz_min":
        return P3B.search(Mm, raw, cells0, allm, score=lambda r, s: Mm.p11(r, s).min(0).values)
    if name in ("hazard", "factor"):
        return P3B.search(Mm, raw, cells0, allm, score=lambda r, s: extra(r, s).mean(0))
    if name == "step_team_props":  # E10: per-step scorer on the team's cVAE+Ranker proposals
        return P3B.search(Mm, raw, cells0, lambda cells, moved: [(o, d) for o in objs if o not in moved
                                                                 for d in P3B.propose(Mm, raw, cells, o, gen, {goal})], score=lambda r, s: extra(r, s).mean(0))
    if name == "boltz_ranker_filter":  # E10: Boltzmann, moves limited to Ranker p >= 0.5 (else the single best cell)
        def ranked(cells, o):
            cand, _, pr = Mm.place(raw, cells, o, exclude={goal})
            keep = [c for c, q in zip(cand, pr.tolist()) if q >= 0.5]
            return keep or [cand[int(pr.argmax())]]
        return P3B.search(Mm, raw, cells0, lambda cells, moved: [(o, d) for o in objs if o not in moved for d in ranked(cells, o)])
    if name in ("boltz_B", "outcome_B"):  # E10: label-source controls (boltz_B is blocked by the team gate; outcome_B replaces it)
        return P3B.search(Mm, raw, cells0, allm, score=lambda r, s: extra(r, s).mean(0))
    if name == "gcvae_pipeline":
        return P3B.search(Mm, raw, cells0, lambda cells, moved: [(o, d) for o in objs if o not in moved
                                                                 for d in extra.propose(raw, cells, o, gen, exclude={goal})])
    raise ValueError(name)


def team_len_plans():
    """E10: Algorithm 1's length penalty on the team pipeline's own scored plans: rank by log p11 - alpha*len over ALL plans its
    (replayed, identical) search scored.  Needs plans_team_pipeline_all.json (every scored plan), written by team_pipeline_plans(full=True)."""
    import math
    full = json.loads((OUT / "plans_team_pipeline_all.json").read_text())
    for a in ALPHAS:
        f = OUT / f"plans_team_len_a{a}.json"
        if f.exists():
            continue
        out = []
        for e in full:
            key = [math.log(max(s, 1e-12)) - a * len(p) for p, s in zip(e["plans"], e["scores"])]
            order = sorted(range(len(key)), key=lambda i: -key[i])[:KEEP]
            out.append({"root": e["root"], "K": e["K"], "plans": [e["plans"][i] for i in order], "scores": [e["scores"][i] for i in order]})
        f.write_text(json.dumps(out))


def team_pipeline_plans(full=False):
    """Replays phase3b.cmd_picks call for call (same generator, same order: 90 contexts, then 160 roots), keeping the pipeline search's
    full ranking so the team pipeline gets the same lazy retries; its first plan must equal Phase 3b's measured pipeline_plan."""
    f = OUT / ("plans_team_pipeline_all.json" if full else "plans_team_pipeline.json")
    if f.exists():
        return
    torch = __import__("torch")
    torch.manual_seed(0)
    gen, Mm = torch.Generator().manual_seed(0), P3.Models()
    for p in json.loads((P3.OUT / "picks_ctx.json").read_text()):  # consumes the generator exactly as 3b did
        raw = Mm.load(p["root"])
        prefix = [(o, tuple(c)) for o, c in p["prefix"]]
        P3B.propose(Mm, raw, {**{k: tuple(v) for k, v in raw["initial_cells"].items()}, **dict(prefix)}, p["object"], gen)
    old, out = json.loads((P3B.OUT.parent / "PHASE3B/picks.json").read_text())["plan"], []
    for p, q in zip(json.loads((P3.OUT / "picks_plan.json").read_text()), old):
        raw = Mm.load(p["root"])
        goal, cells0 = tuple(raw["goal"]), {k: tuple(v) for k, v in raw["initial_cells"].items()}
        objs = sorted(o for o in cells0 if o != "t0")
        sc = P3B.search(Mm, raw, cells0, lambda cells, moved: [(o, d) for o in objs if o not in moved
                                                                for d in P3B.propose(Mm, raw, cells, o, gen, {goal})])
        ranked = sorted(sc, key=sc.get, reverse=True)[:None if full else KEEP]
        plans = [[[o, list(c)] for o, c in s] for s in ranked]
        if full:  # keep the search's own order; the first plan is still the measured one
            assert plans[0] == q["pipeline_plan"], p["root"]
            out.append({"root": p["root"], "K": p["K"], "plans": plans, "scores": [round(sc[s], 6) for s in ranked]}); continue
        if plans[0] != q["pipeline_plan"]:  # keep the measured first plan first; report the mismatch
            plans = [q["pipeline_plan"]] + [x for x in plans if x != q["pipeline_plan"]]
            print(json.dumps({"replay_mismatch": p["root"]}), flush=True)
        out.append({"root": p["root"], "K": p["K"], "plans": plans, "scores": [q["pipeline_p11"]] + [round(sc[s], 5) for s in ranked][1:]})
    f.write_text(json.dumps(out))
    print(json.dumps({"planner": "team_pipeline", "roots": len(out)}), flush=True)


def cmd_picks(a):
    OUT.mkdir(parents=True, exist_ok=True)
    torch = __import__("torch")
    names = [n for n in a.planners.split(",") if not (OUT / f"plans_{n}.json").exists()]
    if not names:
        return
    Mm = P3.Models()
    for name in names:
        torch.manual_seed(0)
        gen, extra = torch.Generator().manual_seed(0), None
        if name == "hazard":
            import hazard; extra = hazard.Hazard()
        if name == "gcvae_pipeline":
            import gcvae; extra = gcvae.GCVAE()
        if name == "step_team_props":
            import hazard; extra = hazard.Hazard()
        if name == "boltz_B":
            import sidecar_from_b; extra = sidecar_from_b.BoltzB()
        if name == "outcome_B":
            import outcome_b; extra = outcome_b.OutcomeB()
        if name == "factor":  # E8 factor graph (hybrid/factor.py); added after its offline gate, before any of its physics
            import factor; extra = factor.FactorModel()
        out, t0 = [], time.time()
        for p in json.loads((P3.OUT / "picks_plan.json").read_text()):
            raw = Mm.load(p["root"])
            goal, cells0 = tuple(raw["goal"]), {k: tuple(v) for k, v in raw["initial_cells"].items()}
            sc = plans_for(name, Mm, raw, cells0, sorted(o for o in cells0 if o != "t0"), goal, gen, extra)
            ranked = sorted(sc, key=sc.get, reverse=True)[:KEEP]
            out.append({"root": p["root"], "K": p["K"], "plans": [[[o, list(c)] for o, c in s] for s in ranked],
                        "scores": [round(sc[s], 5) for s in ranked]})
        (OUT / f"plans_{name}.json").write_text(json.dumps(out))
        print(json.dumps({"planner": name, "roots": len(out), "s": round(time.time() - t0)}), flush=True)


def cmd_ctx_picks():
    f = OUT / "picks_ctx_gcvae.json"
    if f.exists() or not (P3.OUT.parent / "GCVAE/s0.pt").exists():
        return
    import gcvae
    G, Mm, out = gcvae.GCVAE(), P3.Models(), []
    for p in json.loads((P3.OUT / "picks_ctx.json").read_text()):
        raw = Mm.load(p["root"])
        cells = {**{k: tuple(v) for k, v in raw["initial_cells"].items()}, **{o: tuple(c) for o, c in p["prefix"]}}
        cand, lq = G.logq(raw, cells, p["object"])
        out.append({"ctx": p["ctx"], "gcvae": [list(cand[j]) for j in lq.topk(min(P3.M, len(cand))).indices.tolist()]})
    f.write_text(json.dumps(out, indent=1))


def results():
    res = P3B.results()
    f = OUT / "results.jsonl"
    if f.exists():
        res.update({json.dumps(r["job"]): r for r in map(json.loads, f.read_text().splitlines())})
    return res


def lazy(entry, res):
    """-> (tried [(plan, result or None)], next plan to run or None).  Replays the lazy policy over known results."""
    tried, dead = [], []
    for plan in entry["plans"]:
        if len(tried) == TRIES or (tried and tried[-1][1]["success"]):
            break
        if plan in [t for t, _ in tried] or any(plan[:len(d)] == d if kind == "prefix" else plan == d for kind, d in dead):
            continue
        r = res.get(json.dumps([entry["root"], plan]))
        if r is None:
            return tried, plan
        tried.append((plan, r))
        if not r["success"]:
            dead.append(("exact", plan) if r["failed_step"] >= len(plan) else ("prefix", plan[:r["failed_step"] + 1]))
    return tried, None


ONLY = None  # --only: restrict the run loop to these planners (lets a second process run without touching the others' rounds)


def planners():
    return [n for n in ALL if (OUT / f"plans_{n}.json").exists() and (ONLY is None or n in ONLY)]


def cmd_run(a):
    cmd_picks(a); cmd_ctx_picks()
    P3B.OUT = OUT  # run_jobs appends to this phase's results file
    if (OUT / "picks_ctx_gcvae.json").exists() and ONLY is None:
        res, ctx = results(), json.loads((P3.OUT / "picks_ctx.json").read_text())
        todo = {json.dumps(P3.ctx_job(p, c)): P3.ctx_job(p, c) for p, q in zip(ctx, json.loads((OUT / "picks_ctx_gcvae.json").read_text())) for c in q["gcvae"]}
        todo = [j for k, j in todo.items() if k not in res]
        print(json.dumps({"gcvae_ctx_jobs": len(todo)}), flush=True)
        P3B.run_jobs(todo, a.workers)
    for rnd in range(1, TRIES + 1):
        res, todo = results(), {}
        for n in planners():
            for e in json.loads((OUT / f"plans_{n}.json").read_text()):
                _, nxt = lazy(e, res)
                if nxt is not None:
                    todo.setdefault(json.dumps([e["root"], nxt]), [e["root"], nxt])
        print(json.dumps({"round": rnd, "planners": planners(), "new_physics_jobs": len(todo)}), flush=True)
        if not todo:
            break
        P3B.run_jobs(list(todo.values()), a.workers)
    cmd_summary(a)


def cmd_summary(a):
    res, out = results(), {}
    per1, perN = {}, {}
    for n in planners():
        E = json.loads((OUT / f"plans_{n}.json").read_text())
        T = [lazy(e, res)[0] for e in E]
        if any(not t for t in T):
            out[n] = "physics incomplete"; continue
        first = [t[0][1]["success"] for t in T]
        within = [int(any(r["success"] for _, r in t)) for t in T]
        queries = [sum((len(p) + 1) if r["success"] else (r["failed_step"] + 1) for p, r in t) for t in T]
        per1[n], perN[n] = first, within
        out[n] = {"roots": len(E), "success_try1": f"{sum(first)}/{len(E)}", "try1_95ci": P3.wilson(sum(first), len(E)),
                  "success_within_5": f"{sum(within)}/{len(E)}", "within5_95ci": P3.wilson(sum(within), len(E)),
                  "mean_plans_tried": round(sum(len(t) for t in T) / len(E), 2), "mean_motion_queries": round(sum(queries) / len(E), 1),
                  "within5_by_K_%": {K: round(100 * sum(w for w, e in zip(within, E) if e["K"] == K) / sum(e["K"] == K for e in E), 1) for K in (1, 2, 3)},
                  "predicted_vs_real_try1": [round(sum(e["scores"][0] for e in E) / len(E), 3), round(sum(first) / len(E), 3)]}
    ns = list(per1)
    out["try1_head_to_head"] = P3.compare(per1, [(x, y) for i, x in enumerate(ns) for y in ns[i + 1:]])
    out["within5_head_to_head"] = P3.compare(perN, [(x, y) for i, x in enumerate(ns) for y in ns[i + 1:]])
    if (OUT / "picks_ctx_gcvae.json").exists():
        ok = lambda job: res.get(json.dumps(job), {}).get("success")  # noqa: E731
        ctx, g = json.loads((P3.OUT / "picks_ctx.json").read_text()), json.loads((OUT / "picks_ctx_gcvae.json").read_text())
        pb = json.loads((P3B.OUT.parent / "PHASE3B/picks.json").read_text())["ctx"]
        per = {"gcvae": [ok(P3.ctx_job(p, q["gcvae"][0])) for p, q in zip(ctx, g)], "cvae": [ok(P3.ctx_job(p, p["cvae"][0])) for p in ctx],
               "boltzmann": [ok(P3.ctx_job(p, p["critic"][0])) for p in ctx], "pipeline": [ok(P3.ctx_job(p, q["pipeline"][0])) for p, q in zip(ctx, pb)]}
        if all(v is not None for x in per.values() for v in x):
            r4 = [[ok(P3.ctx_job(p, c)) for c in q["gcvae"]] for p, q in zip(ctx, g)]
            out["A_gcvae_90"] = {"top1_works": f"{sum(per['gcvae'])}/90", "top1_95ci": P3.wilson(sum(per["gcvae"]), 90),
                                 "share_of_top4": round(sum(sum(r) / len(r) for r in r4) / 90, 3),
                                 "head_to_head": P3.compare(per, [("gcvae", "cvae"), ("gcvae", "pipeline"), ("boltzmann", "gcvae")])}
        else:
            out["A_gcvae_90"] = "physics incomplete (needs Phase 3 / 3b context results too)"
    out["REAP_published_task_%_by_K (V4 TEST scenes, ~42 checks/scene; context only)"] = P3B.REAP_TASK_BY_K
    print(json.dumps(out, indent=1))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))


def selftest():
    e = {"root": "r", "plans": [[["a", [0, 0]], ["b", [1, 1]]], [["a", [0, 0]], ["c", [2, 2]]], [["c", [3, 3]]]]}
    res = {json.dumps(["r", e["plans"][0]]): {"success": 0, "failed_step": 0}}
    tried, nxt = lazy(e, res)
    assert len(tried) == 1 and nxt == [["c", [3, 3]]], (tried, nxt)  # plan 2 shares the failed first move -> skipped
    res[json.dumps(["r", e["plans"][2]])] = {"success": 1, "failed_step": None}
    assert lazy(e, res) == ([(e["plans"][0], res[json.dumps(["r", e["plans"][0]])]), (e["plans"][2], res[json.dumps(["r", e["plans"][2]])])], None)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["picks", "run", "summary", "selftest"]); ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--planners", default="boltz_mean,boltz_min")
    ap.add_argument("--only", help="comma list: run rounds for these planners only; skips the gcvae context jobs")
    a = ap.parse_args()
    ONLY = set(a.only.split(",")) if a.only else None
    selftest() if a.cmd == "selftest" else {"picks": cmd_picks, "run": cmd_run, "summary": cmd_summary}[a.cmd](a)
