"""Phase 3: (A) placement on more contexts + critic seeds + a Ranker veto, (B) full plan search by the critic.  VAL only; every rule below
was fixed before any Phase 3 physics result was seen (Phase 2 results were seen; part A's first 30 contexts are Phase 2's, reported apart).

A. Placement (where to put the object), first N_CTX = 90 contexts of Phase 2's seed-0 order; top M = 4 cells each:
    random, cvae                      as Phase 2
    ranker                            team Ranker (3 seeds, mean probability): goal-blind "can the box sit here", trained on labels
    critic_s0 / _s1 / _s2             frozen X/Y critic D, training seeds 0 / 1 / 2 (critic_s0 = Phase 2's critic)
    critic                            mean p11 of the three seeds
    v1b_ranker                        critic order after a Ranker veto: cells with Ranker p < 0.5 drop to the end
    direct                            DIRECT-GEO critic (no X/Y structure), mean p11 of its three seeds   [added after seeing the
                                      3-seed log-losses, before any Phase 3 physics: tests whether the X/Y structure matters]
B. Plan search (which objects, which order, where), every VAL root with verified_K >= 1 (the critic cannot score an empty plan):
    critic_plan    beam search (width 16, depth <= 3) over plans of distinct non-target objects to free cells != goal, scored by the seed-mean
                   p11 of the whole plan; the highest-p11 plan of any length 1..3 is executed
    direct_plan    the same search scored by the DIRECT-GEO seed mean
    cvae_cells     the critic_plan's objects in its order, each put on the cVAE's top cell (!= goal) in the scene left by the earlier moves
    random_plan    same length as critic_plan, random distinct objects to random free cells != goal
Physics: hybrid/physics.execute (full teacher, chained); one run per unique (root, plan); Phase 2 results are reused for identical jobs.

    python ../hybrid/src/phase3.py picks                  (models only, no physics)
    python ../hybrid/src/phase3.py run --workers 6        (picks if missing, then physics; resumable)
    python ../hybrid/src/phase3.py summary
"""
import argparse, json, math, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # shelf_6x6_open_sides/
OUT = ROOT / "approachb_3d_learning/runs/PHASE3"
PREV = ROOT / "approachb_3d_learning/runs/PHASE2/results.jsonl"
N_CTX, N_OLD, M = 90, 30, 4
CRITICS = ("smoke_cpu", "smoke_cpu_s1", "smoke_cpu_s2")
CTX_PICKERS = ("random", "cvae", "ranker", "critic_s0", "critic_s1", "critic_s2", "critic", "v1b_ranker", "direct")
PLAN_PICKERS = ("critic_plan", "direct_plan", "cvae_cells", "random_plan")
BEAM, DEPTH, VETO_P = 16, 3, 0.5


class Models:
    def __init__(self):
        import numpy as np, torch
        sys.path.insert(0, str(HERE))
        import v1_check as V
        self.V, self.np, self.torch = V, np, torch
        D, Mo, XY = V.D, V.Mo, V.XY
        cfg = json.loads((V.A / "configs/sampler_v1.json").read_text())
        scal = json.loads((V.A / cfg["derived"] / "scalers.json").read_text())
        self.mu, self.sd = np.array(scal["mean"], np.float32), np.array(scal["std"], np.float32)

        def load(kind, s):
            m = Mo.build(kind, cfg); m.load_state_dict(torch.load(V.A / f"runs/SAMPLER_V1/{kind}_s{s}/best.pt", weights_only=False)["state"])
            return m.eval()
        self.cvaes, self.rankers = [load("cvae", s) for s in (0, 1, 2)], [load("ranker", s) for s in (0, 1, 2)]
        self.critics = {}
        for arm in ("D", "DIRECT-GEO"):
            for d in CRITICS:
                ck = torch.load(V.A / f"runs/XY3D_REAL/{d}/{arm}_best.pt", weights_only=False)
                c = XY.XYNet(ck["arm"], 5, 14).to(torch.float32); c.load_state_dict(ck["state"]); self.critics.setdefault(arm, []).append(c.eval())
        self.raw = {}

    def load(self, rid):
        return self.raw.setdefault(rid, self.V.XD.load(rid))

    def place(self, raw, cells, o, exclude=()):
        """-> legal cells (free, not in exclude), log q_cVAE (3-seed mixture), Ranker probability (3-seed mean)."""
        np, torch, D = self.np, self.torch, self.V.D
        f, msk, occ, leg = D.scene_features(D.allowed(raw)["objects"], cells, o)
        f = (f - self.mu) / self.sd * msk[:, None]
        L = [int(i) for i in np.nonzero(leg)[0] if (int(i) // D.COLS, int(i) % D.COLS) not in exclude]
        cand = [(i // D.COLS, i % D.COLS) for i in L]
        T = lambda x, dt=None: torch.as_tensor(x, dtype=dt)[None].expand(len(L), *np.shape(x))  # noqa: E731
        args = (T(f, torch.float32), T(msk), T(occ, torch.float32), T(leg))
        with torch.no_grad():
            lq = torch.stack([m.log_prob(*args, torch.tensor(L), k=64) for m in self.cvaes]).logsumexp(0) - math.log(len(self.cvaes))
            one = tuple(a[:1] for a in args[:3])
            pr = torch.stack([r.logits_all(*one)[0, L].sigmoid() for r in self.rankers]).mean(0)
        return cand, lq, pr

    def p11(self, raw, seqs, arm="D"):
        """-> [n_critics, len(seqs)] success probability of each whole plan (moves, then t0 -> goal)."""
        torch, XY = self.torch, self.V.XY
        v, _, xm, _ = self.V.XD.view3d(raw, extra_seqs=[list(s) for s in seqs], use_history=False)
        b, _ = XY.collate([v], None, "cpu", torch.float32)
        with torch.no_grad():
            return torch.stack([c.predict(b)["psucc"][torch.tensor(xm)] for c in self.critics[arm]]).clamp_min(1e-9)


def ctx_picks(Mm):
    sc = json.loads((Mm.V.A / "runs/XY3D_REAL/BD3DS66_XY_SIDECAR_V1.json").read_text())
    ctxs = [(k, lab) for k, lab in Mm.V.val_contexts(sc).items() if any(lab.values())]
    ctxs.sort(key=lambda x: (x[0][0], str(x[0][1]), x[0][2]))
    random.Random(0).shuffle(ctxs)  # identical to Phase 2, so its 30 contexts are the first 30 here
    out = []
    for i, ((rid, prefix, o), lab) in enumerate(ctxs[:N_CTX]):
        raw = Mm.load(rid)
        cells = {k: tuple(v) for k, v in raw["initial_cells"].items()}; cells.update(dict(prefix))
        cand, lq, pr = Mm.place(raw, cells, o)
        p = Mm.p11(raw, [list(prefix) + [(o, c)] for c in cand])
        n = len(cand)
        top = lambda s: [list(cand[j]) for j in s.topk(min(M, n)).indices.tolist()]  # noqa: E731
        veto = pr < VETO_P
        pk = {"ctx": i, "phase2": i < N_OLD, "root": rid, "prefix": [[a, list(c)] for a, c in prefix], "object": o, "n_legal": n,
              "ranker_vetoed": int(veto.sum()), "random": [list(c) for c in random.Random(i).sample(cand, min(M, n))],
              "cvae": top(lq), "ranker": top(pr), "critic": top(p.mean(0)), "v1b_ranker": top(p.mean(0) - 1e3 * veto.float()),
              "direct": top(Mm.p11(raw, [list(prefix) + [(o, c)] for c in cand], "DIRECT-GEO").mean(0))}
        pk.update({f"critic_s{s}": top(p[s]) for s in range(len(CRITICS))})
        out.append(pk)
    return out


def search(Mm, raw, cells0, objs, goal, arm):
    """beam search over plans of distinct non-target objects to free cells != goal -> (best plan of any length 1..DEPTH, its seed-mean p11)"""
    beam, best = [()], ((), -1.0)
    for _ in range(DEPTH):
        cands = []
        for seq in beam:
            cells = {**cells0, **dict(seq)}
            occ, moved = set(cells.values()), {o for o, _ in seq}
            cands += [seq + ((o, (r, c)),) for o in objs if o not in moved for r in range(6) for c in range(6)
                      if (r, c) not in occ and (r, c) != goal]
        if not cands:
            break
        p = Mm.p11(raw, cands, arm).mean(0)
        j = int(p.argmax())
        if p[j] > best[1]:
            best = (cands[j], float(p[j]))
        beam = [cands[k] for k in p.topk(min(BEAM, len(cands))).indices.tolist()]
    return best


def plan_picks(Mm):
    split = Mm.V.XD.qc_roots()
    roots = [r for r, s in split.items() if s == "val" and Mm.load(r)["verified_K"] >= 1]
    out = []
    for i, rid in enumerate(roots):
        raw = Mm.load(rid)
        goal, cells0 = tuple(raw["goal"]), {k: tuple(v) for k, v in raw["initial_cells"].items()}
        objs = sorted(o for o in cells0 if o != "t0")
        t0 = time.time()
        best, dbest = search(Mm, raw, cells0, objs, goal, "D"), search(Mm, raw, cells0, objs, goal, "DIRECT-GEO")
        plan = list(best[0])
        cells, cv = dict(cells0), []
        for o, _ in plan:  # same objects and order, cVAE's top cell each time
            cand, lq, _ = Mm.place(raw, cells, o, exclude={goal})
            d = cand[int(lq.argmax())]; cv.append((o, d)); cells[o] = d
        rng, cells, rp = random.Random(1000 + i), dict(cells0), []
        for o in rng.sample(objs, len(plan)):
            occ = set(cells.values())
            d = rng.choice([(r, c) for r in range(6) for c in range(6) if (r, c) not in occ and (r, c) != goal]); rp.append((o, d)); cells[o] = d
        J = lambda s: [[o, list(c)] for o, c in s]  # noqa: E731
        out.append({"i": i, "root": rid, "K": raw["verified_K"], "minimal_sets": raw["labels"]["blocker"]["minimal_sets"],
                    "critic_p11": round(best[1], 4), "search_s": round(time.time() - t0, 1),
                    "critic_plan": J(plan), "direct_plan": J(dbest[0]), "direct_p11": round(dbest[1], 4), "cvae_cells": J(cv), "random_plan": J(rp)})
        print(json.dumps({k: out[-1][k] for k in ("i", "root", "K", "critic_p11", "critic_plan", "search_s")}), flush=True)
    return out


def ctx_job(p, c):
    return [p["root"], p["prefix"] + [[p["object"], c]]]


def _run(job):
    os.environ.setdefault("MUJOCO_GL", "disable")
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "approachb_3d_learning/src"))
    import physics, xy3d_data as XD
    rid, steps = job
    return job, physics.execute(XD.load(rid), [(o, tuple(c)) for o, c in steps])


def results():
    res = {}
    for f in (PREV, OUT / "results.jsonl"):
        if f.exists():
            res.update({json.dumps(r["job"]): r for r in map(json.loads, f.read_text().splitlines())})
    return res


def cmd_picks(a):
    OUT.mkdir(parents=True, exist_ok=True)
    pc, pp = OUT / "picks_ctx.json", OUT / "picks_plan.json"
    if not (pc.exists() and pp.exists()):
        import torch
        torch.manual_seed(0)
        Mm = Models()
        if not pc.exists():
            pc.write_text(json.dumps(ctx_picks(Mm), indent=1))
        if not pp.exists():
            pp.write_text(json.dumps(plan_picks(Mm), indent=1))
    return pc, pp


def cmd_run(a):
    pc, pp = cmd_picks(a)
    jobs = {}
    for p in json.loads(pp.read_text()):  # plan search first: it is the new question
        for k in PLAN_PICKERS:
            jobs.setdefault(json.dumps([p["root"], p[k]]), [p["root"], p[k]])
    for p in json.loads(pc.read_text()):
        for k in CTX_PICKERS:
            for c in p[k][:1] if os.environ.get("TOP1_ONLY") or (OUT / "TOP1_ONLY").exists() else p[k]:  # TOP1_ONLY=1: laptop time; backup picks only where already run
                jobs.setdefault(json.dumps(ctx_job(p, c)), ctx_job(p, c))
    done = results()
    todo = [j for key, j in jobs.items() if key not in done]
    print(json.dumps({"unique_physics_jobs": len(jobs), "already_done": len(jobs) - len(todo)}), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, (OUT / "results.jsonl").open("a") as f:
        futs = [ex.submit(_run, j) for j in todo]
        for i, fu in enumerate(as_completed(futs), 1):
            job, res = fu.result()
            f.write(json.dumps({"job": job, **res}) + "\n"); f.flush()
            if i % 20 == 0 or i == len(todo):
                print(f"{i}/{len(todo)} done, {time.time() - t0:.0f}s elapsed", flush=True)
    cmd_summary(a)


def sign_p(w, l):
    """exact two-sided sign test on the non-tied pairs"""
    n = w + l
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(w, l) + 1)) / 2 ** n)


def wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 1.0]
    p, d = k / n, 1 + z * z / n
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round((p + z * z / (2 * n)) / d - h, 3), round((p + z * z / (2 * n)) / d + h, 3)]


def compare(per, pairs):
    out = {}
    for x, y in pairs:
        w = sum(a > b for a, b in zip(per[x], per[y])); l = sum(a < b for a, b in zip(per[x], per[y]))
        out[f"{x} vs {y}"] = {"wins": w, "losses": l, "ties": len(per[x]) - w - l, "sign_test_p": round(sign_p(w, l), 4)}
    return out


def cmd_summary(a):
    res = results()
    ok = lambda job: res.get(json.dumps(job), {}).get("success")  # noqa: E731
    out = {}
    picks = json.loads((OUT / "picks_ctx.json").read_text())
    for name, sub in (("A_all_90", picks), ("A_fresh_60", [p for p in picks if not p["phase2"]])):
        if any(ok(ctx_job(p, p[k][0])) is None for p in sub for k in CTX_PICKERS):
            out[name] = "physics incomplete"; continue
        n, per, s = len(sub), {}, {"contexts": len(sub)}
        for k in CTX_PICKERS:
            top1 = [ok(ctx_job(p, p[k][0])) for p in sub]
            r4 = [[ok(ctx_job(p, c)) for c in p[k]] for p in sub]
            r4 = [r for r in r4 if None not in r]  # contexts whose 4 backup picks were all run
            per[k] = top1
            s[k] = {"top1_works": f"{sum(top1)}/{n}", "top1_95ci": wilson(sum(top1), n),
                    "any_of_top4": f"{sum(max(r) for r in r4)}/{len(r4)}", "share_of_top4": round(sum(sum(r) / len(r) for r in r4) / max(1, len(r4)), 3),
                    "top4_contexts": len(r4)}
        s["top1_head_to_head"] = compare(per, [("critic", "cvae"), ("critic", "random"), ("critic", "ranker"), ("ranker", "cvae"),
                                               ("v1b_ranker", "critic"), ("critic", "direct"), ("direct", "cvae"), ("critic_s0", "cvae"), ("critic_s1", "cvae"), ("critic_s2", "cvae")])
        s["ranker_vetoed_mean"] = round(sum(p["ranker_vetoed"] for p in sub) / n, 2)
        s["v1b_ranker_top1_differs_from_critic"] = sum(p["v1b_ranker"][0] != p["critic"][0] for p in sub)
        out[name] = s
    plans = json.loads((OUT / "picks_plan.json").read_text())
    if any(ok([p["root"], p[k]]) is None for p in plans for k in PLAN_PICKERS):
        out["B_plan_search"] = "physics incomplete"
    else:
        per = {k: [ok([p["root"], p[k]]) for p in plans] for k in PLAN_PICKERS}
        s = {"roots": len(plans)}
        for k in PLAN_PICKERS:
            s[k] = {"works": f"{sum(per[k])}/{len(plans)}", "95ci": wilson(sum(per[k]), len(plans)),
                    "by_K": {K: f"{sum(v for v, p in zip(per[k], plans) if p['K'] == K)}/{sum(p['K'] == K for p in plans)}" for K in (1, 2, 3)}}
        s["head_to_head"] = compare(per, [("critic_plan", "cvae_cells"), ("critic_plan", "direct_plan"), ("direct_plan", "cvae_cells"), ("critic_plan", "random_plan"), ("cvae_cells", "random_plan")])
        s["critic_plan_length_vs_K"] = {f"K{K}": {f"len{L}": sum(p["K"] == K and len(p["critic_plan"]) == L for p in plans) for L in (1, 2, 3)} for K in (1, 2, 3)}
        s["critic_objects_contain_a_minimal_blocker_set"] = sum(any(set(m) <= {o for o, _ in p["critic_plan"]} for m in p["minimal_sets"]) for p in plans)
        s["critic_p11_when_works_vs_fails"] = [round(sum(p["critic_p11"] for p, v in zip(plans, per["critic_plan"]) if v == x) / max(1, sum(v == x for v in per["critic_plan"])), 3) for x in (1, 0)]
        out["B_plan_search"] = s
    print(json.dumps(out, indent=1))
    (OUT / "summary.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary", "picks"]); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    {"run": cmd_run, "summary": cmd_summary, "picks": cmd_picks}[a.cmd](a)
