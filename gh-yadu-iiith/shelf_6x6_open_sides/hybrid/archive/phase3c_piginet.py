"""ARCHIVED (PIGINet dropped from the comparison on 2026-11-20; model only partly trained). Phase 3c: PIGINet (published plan-feasibility predictor, Ayush's yadu/PIGINet port, no-image variant, seed 0, 15 epochs instead of 60 [laptop time; cosine schedule over 15]) in place of the Boltzmann
model, on Phase 3's 90 contexts / 160 roots, same search and the same physics.  VAL only; rules fixed before any Phase 3c physics result.

PIGINet was trained here with Ayush's code and recipe, unchanged except: 5 fixed random vectors stand in for the CLIP text features of its
5 token phrases (CLIP not installed; the phrases only label token types), and 344 V4 roots not on any reachable branch are absent
(1,524 instead of 1,766 train roots).  Its epoch is chosen on V4 val AUROC, which contains these roots, as the Boltzmann model's
checkpoint is chosen on Approach-B val (also these roots).

    A  piginet          top 4 cells by PIGINet's probability for (prefix moves, o -> cell, t0 -> goal)
    B  piginet_plan     Phase 3 beam search (width 16, depth <= 3, cells != goal) scored by PIGINet; best plan of any length executed

    python hybrid/phase3c.py run --workers 6 | summary
"""
import argparse, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))  # archived: code lives in ../src
import phase3 as P3, phase3b as P3B  # noqa: E402

OUT = P3.OUT.parent / "PHASE3C"
PIG = P3.OUT.parent / "PIGINET/src"


class Pig:
    def __init__(self):
        import numpy as np, torch
        sys.path.insert(0, str(PIG))
        import piginet as PG
        self.PG, self.torch = PG, torch
        self.m = PG.PIGINet(use_image=False)
        self.m.load_state_dict(torch.load(PIG / "runs/noimg_s0_e15/best.pt", map_location="cpu"))
        self.m.set_text(np.load(PIG / "data/text.npz")["feats"]); self.m.eval()

    def __call__(self, raw, seqs):
        """-> PIGINet feasibility probability of each plan (moves from the initial scene, then t0 -> goal)"""
        objs = [{k: o[k] for k in ("id", "shape", "size", "height", "color")} for o in raw["objects"]]
        init, goal = {k: tuple(v) for k, v in raw["initial_cells"].items()}, tuple(raw["goal"])
        out = []
        for k in range(0, len(seqs), 1024):
            items = [(self.PG.tokens(dict(objects=objs, cells=init, goal=goal,
                                          plan=[(o, init[o], tuple(c)) for o, c in s] + [("t0", init["t0"], goal)])), 0) for s in seqs[k:k + 1024]]
            b = self.PG.collate(items)
            with self.torch.no_grad():
                out.append(self.torch.sigmoid(self.m(b["img"], b["v1"], b["v2"], b["kind"], b["step"], b["mask"])))
        return self.torch.cat(out)


def cmd_picks(a):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "picks.json"
    if f.exists():
        return json.loads(f.read_text())
    pig, Mm = Pig(), P3.Models()
    out = {"ctx": [], "plan": []}
    for p in json.loads((P3.OUT / "picks_ctx.json").read_text()):
        raw = Mm.load(p["root"])
        prefix = [(o, tuple(c)) for o, c in p["prefix"]]
        cells = {**{k: tuple(v) for k, v in raw["initial_cells"].items()}, **dict(prefix)}
        cand, _, _ = Mm.place(raw, cells, p["object"])  # the same legal cells as Phase 3
        s = pig(raw, [prefix + [(p["object"], c)] for c in cand])
        out["ctx"].append({"ctx": p["ctx"], "piginet": [list(cand[j]) for j in s.topk(min(P3.M, len(cand))).indices.tolist()]})
    for p in json.loads((P3.OUT / "picks_plan.json").read_text()):
        raw = Mm.load(p["root"])
        goal, cells0 = tuple(raw["goal"]), {k: tuple(v) for k, v in raw["initial_cells"].items()}
        objs = sorted(o for o in cells0 if o != "t0")
        free = lambda cells: [(r, c) for r in range(6) for c in range(6) if (r, c) not in set(cells.values()) and (r, c) != goal]  # noqa: E731
        sc = P3B.search(Mm, raw, cells0, lambda cells, moved: [(o, d) for o in objs if o not in moved for d in free(cells)], score=pig)
        best = max(sc, key=sc.get)
        out["plan"].append({"root": p["root"], "piginet_plan": [[o, list(c)] for o, c in best], "piginet_p": round(sc[best], 4)})
        print(json.dumps({"root": p["root"], "K": p["K"], **out["plan"][-1]}), flush=True)
    f.write_text(json.dumps(out, indent=1))
    return out


def results():
    res = P3B.results()
    f = OUT / "results.jsonl"
    if f.exists():
        res.update({json.dumps(r["job"]): r for r in map(json.loads, f.read_text().splitlines())})
    return res


def cmd_run(a):
    pk = cmd_picks(a)
    ctx = json.loads((P3.OUT / "picks_ctx.json").read_text())
    jobs = {json.dumps([q["root"], q["piginet_plan"]]): [q["root"], q["piginet_plan"]] for q in pk["plan"]}
    for p, q in zip(ctx, pk["ctx"]):
        for c in q["piginet"]:
            jobs.setdefault(json.dumps(P3.ctx_job(p, c)), P3.ctx_job(p, c))
    done = results()
    todo = [j for k, j in jobs.items() if k not in done]
    print(json.dumps({"piginet_jobs": len(jobs), "todo": len(todo)}), flush=True)
    P3B.OUT = OUT  # run_jobs appends to this phase's results file
    P3B.run_jobs(todo, a.workers)
    cmd_summary(a)


def cmd_summary(a):
    res = results()
    ok = lambda job: res.get(json.dumps(job), {}).get("success")  # noqa: E731
    pk, b = json.loads((OUT / "picks.json").read_text()), json.loads((P3B.OUT / "picks.json").read_text())
    ctx, plans = json.loads((P3.OUT / "picks_ctx.json").read_text()), json.loads((P3.OUT / "picks_plan.json").read_text())
    n = len(ctx)
    pig1 = [ok(P3.ctx_job(p, q["piginet"][0])) for p, q in zip(ctx, pk["ctx"])]
    pig4 = [[ok(P3.ctx_job(p, c)) for c in q["piginet"]] for p, q in zip(ctx, pk["ctx"])]
    per = {"piginet": pig1, "boltzmann": [ok(P3.ctx_job(p, p["critic"][0])) for p in ctx],
           "pipeline": [ok(P3.ctx_job(p, q["pipeline"][0])) for p, q in zip(ctx, b["ctx"])], "cvae": [ok(P3.ctx_job(p, p["cvae"][0])) for p in ctx]}
    out = {"A_all_90": {"piginet": {"top1_works": f"{sum(pig1)}/{n}", "top1_95ci": P3.wilson(sum(pig1), n), "any_of_top4": f"{sum(max(r) for r in pig4)}/{n}",
                                    "share_of_top4": round(sum(sum(r) / len(r) for r in pig4) / n, 3)},
                        "top1_head_to_head": P3.compare(per, [("boltzmann", "piginet"), ("piginet", "pipeline"), ("piginet", "cvae")])}}
    per = {"piginet_plan": [ok([q["root"], q["piginet_plan"]]) for q in pk["plan"]], "boltzmann_plan": [ok([p["root"], p["critic_plan"]]) for p in plans],
           "pipeline_plan": [ok([q["root"], q["pipeline_plan"]]) for q in b["plan"]], "random_plan": [ok([p["root"], p["random_plan"]]) for p in plans]}
    v = per["piginet_plan"]
    out["B_plan_search"] = {"piginet_plan": {"works": f"{sum(v)}/{len(v)}", "95ci": P3.wilson(sum(v), len(v)),
                                             "by_K_%": {K: round(100 * sum(x for x, p in zip(v, plans) if p["K"] == K) / sum(p["K"] == K for p in plans), 1) for K in (1, 2, 3)},
                                             "length_vs_K": {f"K{K}": {f"len{L}": sum(p["K"] == K and len(q["piginet_plan"]) == L for p, q in zip(plans, pk["plan"])) for L in (1, 2, 3)} for K in (1, 2, 3)}},
                            "head_to_head": P3.compare(per, [("boltzmann_plan", "piginet_plan"), ("piginet_plan", "pipeline_plan"), ("piginet_plan", "random_plan")])}
    print(json.dumps(out, indent=1))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["picks", "run", "summary"]); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"picks": cmd_picks, "run": cmd_run, "summary": cmd_summary}[a.cmd](a)
