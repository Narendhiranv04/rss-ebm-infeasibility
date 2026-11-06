"""Stage 2 of the class-quota 3D dataset: per base scene, up to five verified SOLUTIONS and three verified REDUNDANT variants, every
one EXECUTED in MuJoCo with the V2 teacher (bd_3dt.execute: ready pose once, after the last relocation; each object moves at most once).

Solutions (role alt_solution): K relocations then T, K = the scene's verified K (V2). Candidates, in this order: the clearing sets of
size K of the blocker lattice (the minimal repair's set first) x their orders (the minimal repair's order first) x parking cells
(each object's teacher cell, then its alt_parking cells; ranked by the sum of the cells' ranks). A destination is never an initially
occupied cell, the goal, or another mover's destination. The stored minimal repair counts as the first solution.
Redundant variants (role redundant_variant): a verified solution plus 1 or 2 relocations of objects outside its set (excess_actions),
inserted before the repair, after it, or in its middle, round-robin over those kinds. K = 0 scenes: distractor moves, then T.
Every executed candidate is kept as a label, verified (y = 1) or not (y = 0 with its failed step); a candidate equal to an existing B
sequence reuses that label. Budgets: SOL_EXEC / RED_EXEC executions per scene. NONE and DEPENDENT scenes are skipped.

    python stage2_3dt.py run <raw> <out> [workers]     # resumable: <out>/STAGE2.jsonl, <out>/roots/, <out>/witnesses/
    python stage2_3dt.py merge <raw> <out>              # roots + referenced witnesses into raw/, raw/STAGE2.jsonl
"""
import itertools
import json
import os
import shutil
import sys
import time
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "dataset-bd"))
sys.path.insert(0, str(HERE))
VERSION = "STAGE2_V1"
N_SOL, N_RED, SOL_EXEC, RED_EXEC = 5, 3, 12, 8


def choices(r, o):
    return [list(c) for c in dict.fromkeys(tuple(c) for c in r["parking"][o] + r["alt_parking"].get(o, []))]


def solution_candidates(r, K, minimal):
    """K-relocation repairs, most teacher-like first (see module docstring)."""
    if K == 0:
        return [[]]
    occ = {tuple(c) for c in r["initial_cells"].values()} | {tuple(r["goal"])}
    first = [o for o, _ in minimal]
    sets = [first] + [s for s in r["labels"]["blocker"]["minimal_sets"] if len(s) == K and sorted(s) != sorted(first)]
    out = []
    for si, S in enumerate(sets):
        orders = [tuple(S)] + [p for p in itertools.permutations(S) if p != tuple(S)]
        opts = {o: [c for c in choices(r, o) if tuple(c) not in occ] for o in S}
        for oi, order in enumerate(orders):
            for idx in itertools.product(*(range(len(opts[o])) for o in order)):
                dst = [opts[o][i] for o, i in zip(order, idx)]
                if len({tuple(d) for d in dst}) == K:
                    out.append(((sum(idx), oi, si), [[o, d] for o, d in zip(order, dst)]))
    return [seq for _, seq in sorted(out, key=lambda x: x[0])]


def redundant_candidates(r, sols):
    """Round-robin over kinds: +1 before the repair, +1 after it, +2 (one before, one after), +1 in its middle."""
    occ = {tuple(c) for c in r["initial_cells"].values()} | {tuple(r["goal"])}
    kinds = {k: [] for k in ("+1_before", "+1_after", "+2", "+1_middle")}
    for base in sols:
        used = {o for o, _ in base}
        taken = occ | {tuple(d) for _, d in base}
        extra = [(o, next(c for c in choices(r, o) if tuple(c) not in taken)) for o in r["roles"]
                 if o not in used and any(tuple(c) not in taken for c in choices(r, o))]
        for o, d in extra:
            kinds["+1_before"].append([[o, d]] + base)
            kinds["+1_after"].append(base + [[o, d]])
            if len(base) >= 2:
                kinds["+1_middle"].append(base[:len(base) // 2] + [[o, d]] + base[len(base) // 2:])
        for (o1, d1), (o2, d2) in itertools.combinations(extra, 2):
            if d1 != d2:
                kinds["+2"].append([[o1, d1]] + base + [[o2, d2]])
    rr = [x for tup in itertools.zip_longest(*kinds.values()) for x in tup if x is not None]
    return [x for i, x in enumerate(rr) if x not in rr[:i]]


def work(job):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import bd_core as bc
    import bd_3dt as A
    src, out = Path(job[0]), Path(job[1])
    r = json.loads(src.read_text())
    rid, t0, c0 = r["ids"]["root_id"], time.time(), time.process_time()
    K, line = r["verified_K"], {"root_id": rid, "K": r["verified_K"]}
    if K is None or not r["independent_moves"]:
        line.update(action="skipped", why="NONE" if K is None else "DEPENDENT")
        return line
    store, calls = bc.Store(out), {"plan": 0, "execute": 0}
    known = {json.dumps(b["sequence"]): b for b in r["labels"]["B"] if "y" in b}
    minimal = next(b["sequence"] for b in r["labels"]["B"] if b["role"] == "minimal" and b["y"])
    e = A.build(r["objects"], r["initial_cells"])

    def run(seq, role, budget, **meta):
        k = json.dumps(seq)
        if k in known:
            return {**known[k], "reused": True}
        if budget[0] <= 0:
            return None
        budget[0] -= 1
        bc.check_sequence([tuple(x) for x in seq], K)
        res = A.execute(e, r["objects"], r["initial_cells"], r["goal"], [(o, tuple(d)) for o, d in seq], store, calls)
        res = {**res, "role": role, **meta}
        known[k] = res
        return res
    new, sols, budget = [], [], [SOL_EXEC]
    for seq in solution_candidates(r, K, minimal):
        if len(sols) >= N_SOL:
            break
        res = run(seq, "alt_solution", budget, excess_actions=0)
        if res is None:
            break
        if not res.get("reused"):
            new.append(res)
        if res["y"]:
            sols.append(seq)
    red, budget = 0, [RED_EXEC]
    for seq in redundant_candidates(r, sols):
        if red >= N_RED or len(seq) + 1 > bc.MAX_ACTIONS:
            continue
        res = run(seq, "redundant_variant", budget, excess_actions=len(seq) - K)
        if res is None:
            break
        if not res.get("reused"):
            new.append(res)
        red += res["y"]
    r["labels"]["B"] += new
    r["labels"]["stage2"] = {"version": VERSION, "teacher_version": A.TEACHER_VERSION, "solutions_verified": sols,
                             "n_solutions": len(sols), "n_redundant_verified": red, "executions": calls["execute"],
                             "sequences_executed": len(new), "budget": {"N_SOL": N_SOL, "N_RED": N_RED, "SOL_EXEC": SOL_EXEC, "RED_EXEC": RED_EXEC}}
    (out / "roots").mkdir(parents=True, exist_ok=True)
    tmp = out / "roots" / f".{rid}.tmp"
    tmp.write_text(json.dumps(r, indent=1, sort_keys=True))
    tmp.replace(out / "roots" / f"{rid}.json")
    line.update(action="done", n_solutions=len(sols), n_redundant=red, executed=len(new),
                new=[[b["role"], len(b["sequence"]), b["y"]] for b in new], cpu_s=round(time.process_time() - c0, 1), wall_s=round(time.time() - t0, 1))
    return line


def safe(job):
    try:
        return work(job)
    except Exception as e:  # noqa: BLE001  (recorded, never silently dropped)
        return {"root_id": Path(job[0]).stem, "action": "ERROR", "error": f"{type(e).__name__}:{str(e)[:300]}"}


def run_all(raw, out, workers=4):
    raw, out = Path(raw), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    led = [json.loads(x) for x in (raw / "ledger.jsonl").read_text().splitlines() if x.strip()]
    ids = [e["root_id"] for e in led if e.get("accepted")]
    log = out / "STAGE2.jsonl"
    prev = [json.loads(x) for x in log.read_text().splitlines()] if log.exists() else []
    done = {x["root_id"] for x in prev if x["action"] == "skipped" or (out / "roots" / f"{x['root_id']}.json").exists()}
    jobs = sorted(((str(raw / "roots" / f"{i}.json"), str(out)) for i in ids if i not in done), key=lambda j: -Path(j[0]).stat().st_size)
    print(f"{len(ids)} accepted roots, {len(done)} done, {len(jobs)} to go, {workers} workers", flush=True)
    with Pool(workers, maxtasksperchild=10) as pool, log.open("a") as f:
        for n, line in enumerate(pool.imap_unordered(safe, jobs), 1):
            f.write(json.dumps(line, sort_keys=True) + "\n")
            f.flush()
            print(n, line["root_id"], line["action"], line.get("K"), line.get("n_solutions"), line.get("n_redundant"), line.get("cpu_s"), flush=True)


def merge(raw, out):
    raw, out = Path(raw), Path(out)
    log = {}
    for x in (out / "STAGE2.jsonl").read_text().splitlines():
        line = json.loads(x)
        log[line["root_id"]] = line
    bad = [i for i, x in log.items() if x["action"] == "ERROR"]
    if bad:
        raise SystemExit(f"{len(bad)} roots failed, e.g. {bad[:3]}")
    for i, x in log.items():
        if x["action"] != "done":
            continue
        r = json.loads((out / "roots" / f"{i}.json").read_text())
        for w in {w for b in r["labels"]["B"] for w in b.get("witnesses", [])}:
            dst = raw / "witnesses" / w[:2] / f"{w}.npz"
            if not dst.exists():
                dst.parent.mkdir(exist_ok=True)
                shutil.copy2(out / "witnesses" / w[:2] / f"{w}.npz", dst)
        shutil.copy2(out / "roots" / f"{i}.json", raw / "roots" / f"{i}.json")
    (raw / "STAGE2.jsonl").write_text("".join(json.dumps(log[i], sort_keys=True) + "\n" for i in sorted(log)))


if __name__ == "__main__":
    if sys.argv[1] == "one":
        print(json.dumps(work((sys.argv[2], sys.argv[3])), indent=1))
    elif sys.argv[1] == "run":
        run_all(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 4)
    else:
        merge(sys.argv[2], sys.argv[3])
