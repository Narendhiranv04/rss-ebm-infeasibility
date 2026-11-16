"""Second pass over the alternative minimal sets of the counted scenes: the sets that alt_sets66.py could NOT execute with the first stored parking
cell of each object are retried with the other stored parking cells (alt_parking, up to 5 per object) and every order.

Why: a set can clear T in the teacher's T-only lattice (objects teleported to their FIRST parking cell) while one of its relocations has no plan to that
cell (out of reach, the arm hits a neighbour, the hand overlaps a neighbour at the cell).  A different parking cell often works.

For one failed set S of a scene: candidate cells per object = alt_parking[o] (stored order first); assignments (distinct free cells, not the goal) are
ranked by position in those lists; for each assignment every order is tried.  Relocation plans are memoised per (scene state, object, cell), so shared
prefixes are planned once; a chain whose relocations plan (and then T plans) is EXECUTED in MuJoCo with the dataset's checks (alt_sets66.run_seq) and
the first one that verifies is stored.  Budgets per set: MAX_ASSIGN assignments, MAX_EXEC executions, MAX_PLAN fresh plan queries, MAX_MIN minutes.

    python shelf66_v3/alt_dest66.py --alt ../raw_alt_sets --out ../raw_alt_dest --workers 18
Writes <out>/roots/<root_id>.json {"alt_dest": [...]} + witnesses; resumable; touch <out>/STOP to stop.  Nothing else is modified."""
from __future__ import annotations

import argparse
import itertools
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "dataset-bd"))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

MAX_ASSIGN, MAX_EXEC, MAX_PLAN, MAX_MIN, MAX_ORDERS = 40, 8, 160, 30, 6
SEEDS = tuple(int(x) for x in os.environ.get("ALT_DEST_SEEDS", "0").split(","))
VERSION = "ALT_DEST66_V1"


def try_set(A, P, bc, AS, e, store, r, S):
    objs, goal = r["objects"], tuple(r["goal"])
    cells0 = {o: tuple(c) for o, c in r["initial_cells"].items()}
    cand = {o: [tuple(c) for c in (r.get("alt_parking", {}).get(o) or r["parking"].get(o) or [])] for o in S}
    if any(not cand[o] for o in S):
        return {"set": S, "y": 0, "reason": "NO_PARKING_CELL"}
    blocked_cells = {c for o, c in cells0.items() if o not in S} | {goal}
    t0, memo, st = time.time(), {}, {"plan": 0, "exec": 0, "chains": 0}

    def plans(state, o, d):
        key = (tuple(sorted(state.items())), o, d)
        if key not in memo:
            if st["plan"] >= MAX_PLAN:
                return None
            st["plan"] += 1
            A.settle(e, state)
            ok = False
            for sd in SEEDS:
                if A.query(e, o, d, seed=sd)[0] is not None:
                    ok = True
                    break
            memo[key] = ok
        return memo[key]

    def chain_ok(order, dest):
        state = dict(cells0)
        for o in order:
            ok = plans(state, o, dest[o])
            if not ok:
                return ok
            state[o] = dest[o]
        return plans(state, "t0", goal)

    assigns = []
    for combo in itertools.product(*[list(enumerate(cand[o])) for o in S]):
        ds = [c for _, c in combo]
        if len(set(ds)) < len(ds) or any(d in blocked_cells for d in ds):
            continue
        assigns.append((sum(i for i, _ in combo), {o: c for o, (_, c) in zip(S, combo)}))
    assigns.sort(key=lambda x: x[0])
    tried, fails = 0, []
    for _, dest in assigns[:MAX_ASSIGN]:
        tried += 1
        for order in itertools.islice(itertools.permutations(S), MAX_ORDERS):
            if (time.time() - t0) / 60 > MAX_MIN or st["exec"] >= MAX_EXEC or st["plan"] >= MAX_PLAN:
                break
            ok = chain_ok(order, dest)
            if not ok:
                continue
            st["chains"] += 1
            st["exec"] += 1
            res = AS.run_seq(A, bc, e, store, objs, cells0, goal, [(o, dest[o]) for o in order])
            if res["y"]:
                indep = bool(P.independent(objs, cells0, dest, S, goal))
                return {"set": S, "size": len(S), "role": "minimal_alt_dest", "independent": indep, "dest": {o: list(c) for o, c in dest.items()},
                        "assignments_tried": tried, **st, "wall_s": round(time.time() - t0, 1), **res}
            fails.append({"order": list(order), "dest": {o: list(c) for o, c in dest.items()}, "reason": res["reason"][:90], "failed_step": res.get("failed_step")})
    return {"set": S, "size": len(S), "y": 0, "role": "minimal_alt_dest", "assignments_tried": tried, "assignments_total": len(assigns), **st,
            "wall_s": round(time.time() - t0, 1), "executed_failures": fails[:6],
            "stopped_by": "time" if (time.time() - t0) / 60 > MAX_MIN else "plan_budget" if st["plan"] >= MAX_PLAN else "exec_budget" if st["exec"] >= MAX_EXEC else "exhausted"}


def do_one(arg):
    root_id, src, alt_dir, out = arg
    import bd_shelf  # noqa: F401
    import bd_3dt as A
    import bd_3dt_plus as P
    import bd_core as bc
    import alt_sets66 as AS
    t0 = time.time()
    r = json.loads((Path(src) / "roots" / f"{root_id}.json").read_text())
    prev = json.loads((Path(alt_dir) / "roots" / f"{root_id}.json").read_text())
    store = bc.Store(out)
    e = A.build(r["objects"], {o: tuple(c) for o, c in r["initial_cells"].items()})
    res = [try_set(A, P, bc, AS, e, store, r, a["set"]) for a in prev["alt"] if not a["y"]]
    store.save_root({"ids": r["ids"], "observed": r["observed"], "verified_K": r["verified_K"], "version": VERSION, "alt_dest": res, "wall_s": round(time.time() - t0, 1),
                     "budget": {"max_assign": MAX_ASSIGN, "max_exec": MAX_EXEC, "max_plan": MAX_PLAN, "max_min": MAX_MIN, "max_orders": MAX_ORDERS, "seeds": list(SEEDS)}})
    return {"root_id": root_id, "observed": r["observed"], "sets": len(res), "ok": sum(x["y"] for x in res), "wall_s": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alt", default=str(DS / "raw_alt_sets"))
    ap.add_argument("--out", default=str(DS / "raw_alt_dest"))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    only = set(filter(None, a.only.split(",")))
    cand = []
    for p in sorted(Path(a.alt, "roots").glob("*.json")):
        d = json.loads(p.read_text())
        n = sum(1 for x in d["alt"] if not x["y"])
        if n and (not only or p.stem in only):
            cand.append((p.stem, n))
    done = {p.stem for p in (out / "roots").glob("*.json")}
    pend = [(i, str(DS / "raw"), a.alt, str(out)) for i, n in sorted(cand, key=lambda x: -x[1]) if i not in done]
    print(f"{len(cand)} scenes with failed alternative sets ({sum(n for _, n in cand)} sets), {len(done)} done, {len(pend)} to do, {a.workers} workers", flush=True)
    t_start, nd, nok, nset = time.time(), 0, 0, 0
    with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=1) as pool:
        it, running = iter(pend), {}
        while True:
            for r in [r for r in running if r.ready()]:
                running.pop(r)
                try:
                    res = r.get()
                except Exception as ex:
                    print("ERROR", type(ex).__name__, str(ex)[:200], flush=True)
                    continue
                nd += 1
                nok += res["ok"]
                nset += res["sets"]
                with open(out / "ledger.jsonl", "a") as f:
                    f.write(json.dumps(res) + "\n")
                print(f"[{nd}/{len(pend)}] {res['root_id']} K{res['observed']}: executed {res['ok']}/{res['sets']} {res['wall_s']}s  total {nok}/{nset}  "
                      f"elapsed {(time.time() - t_start) / 3600:.2f}h", flush=True)
            if not (out / "STOP").exists():
                while len(running) < a.workers:
                    x = next(it, None)
                    if x is None:
                        break
                    running[pool.apply_async(do_one, (x,))] = x[0]
            if not running:
                break
            time.sleep(2)
    print("finished", flush=True)


if __name__ == "__main__":
    main()
