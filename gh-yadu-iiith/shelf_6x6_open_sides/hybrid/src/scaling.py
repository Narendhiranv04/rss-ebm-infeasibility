"""Timing / scaling of the planners (no physics): wall time and number of plans scored by the Phase-3 beam search (width 16), per scorer,
vs number of movable objects and vs search depth 1..5 (depths 4-5 exceed the trained plan lengths: speed only, not quality).
Single thread, laptop CPU, run with the physics workers idle.  Also: physics time per executed attempt by plan length, from saved results.

    python ../hybrid/src/scaling.py            (run from approachb_3d_learning/)
"""
import collections, json, random, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import phase3 as P3, phase3b as P3B, phase4 as P4  # noqa: E402

PER_GROUP, DEPTHS = 6, (1, 2, 3, 4, 5)


def main():
    torch = __import__("torch")
    torch.set_num_threads(1)
    import hazard, factor
    Mm, H, F = P3.Models(), hazard.Hazard(), factor.FactorModel()
    gen = torch.Generator().manual_seed(0)
    scorers = {"boltzmann": None, "per_step": lambda r, s: H(r, s).mean(0), "factor": lambda r, s: F(r, s).mean(0)}
    plans = json.loads((P3.OUT / "picks_plan.json").read_text())
    by_n = collections.defaultdict(list)
    for p in plans:
        raw = Mm.load(p["root"]); by_n[len(raw["initial_cells"]) - 1].append(p["root"])
    rng, sample = random.Random(0), {}
    for n, roots in sorted(by_n.items()):
        sample[n] = rng.sample(roots, min(PER_GROUP, len(roots)))
    out = {"note": "seconds per scene (mean), plans scored per scene (mean); beam 16; 1 CPU thread", "by_objects_depth3": {}, "by_depth": {}}
    for depth in DEPTHS:
        P3.DEPTH = depth
        row = {}
        for name in list(scorers) + (["team_pipeline"] if depth <= 3 else []):
            ts, ns = [], []
            for n, roots in sample.items():
                for rid in roots:
                    raw = Mm.load(rid)
                    goal, cells0 = tuple(raw["goal"]), {k: tuple(v) for k, v in raw["initial_cells"].items()}
                    objs = sorted(o for o in cells0 if o != "t0")
                    free = lambda cells: [(r, c) for r in range(6) for c in range(6) if (r, c) not in set(cells.values()) and (r, c) != goal]  # noqa: E731
                    if name == "team_pipeline":
                        moves = lambda cells, moved: [(o, d) for o in objs if o not in moved for d in P3B.propose(Mm, raw, cells, o, gen, {goal})]  # noqa: E731
                        sc_fn = None
                    else:
                        moves = lambda cells, moved: [(o, d) for o in objs if o not in moved for d in free(cells)]  # noqa: E731
                        sc_fn = scorers[name]
                    t0 = time.perf_counter()
                    sc = P3B.search(Mm, raw, cells0, moves, score=sc_fn)
                    dt = time.perf_counter() - t0
                    ts.append(dt); ns.append(len(sc))
                    if depth == 3:
                        out["by_objects_depth3"].setdefault(name, {}).setdefault(n, []).append(round(dt, 3))
            row[name] = {"s": round(sum(ts) / len(ts), 3), "plans_scored": round(sum(ns) / len(ns))}
        out["by_depth"][depth] = row
        print(json.dumps({"depth": depth, **row}), flush=True)
    for name, d in out["by_objects_depth3"].items():
        out["by_objects_depth3"][name] = {n: round(sum(v) / len(v), 3) for n, v in sorted(d.items())}
    res = P4.results()
    phys = collections.defaultdict(list)
    for r in res.values():
        if "s" in r and r.get("success") is not None:
            phys[len(r["job"][1]) + 1].append(r["s"])  # steps incl. the target move
    out["physics_s_per_attempt_by_steps"] = {k: {"mean_s": round(sum(v) / len(v), 1), "n": len(v)} for k, v in sorted(phys.items())}
    out["objects_in_sample"] = {n: len(r) for n, r in sample.items()}
    print(json.dumps(out, indent=1))
    (P4.OUT / "scaling.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
