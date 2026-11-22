"""EXPLORATORY, 3D VAL only: "grasp check before acting" on top of the hazard scorer (seed 0) and the frozen V3.4 rule.

For every VAL root where the frozen rule (noop_gate 0.5, eta 0.3) REPAIRs, walk the scorer's ranked list (same pool, same key as the rule;
only admissible plans p11 >= eta; top K = 5, fixed) and accept the first plan whose grasps ALL plan under the shifted layout.
The check = the teacher's own grasp search (bd_3dt.query logic: grasp poses in the scene's order, plan_pick_place = IK + RRT dry run),
no MuJoCo execution of any plan:
  * X: build the root with the ROBUST shift hook (robust_check._init, unchanged), settle at the initial cells, then for each move query
    from the end joint pose of the previous move's plan; after a move the object is TELEPORTED to its destination cell centre (unshifted:
    the shift hook acts only when the scene is built, so in the real run a moved object also ends near the cell centre).
  * Y: settle the endpoint cells (= fin3d.run_y: reset to the shifted home positions, moved objects teleported to cell centres), query
    t0 -> goal from the ready pose.
  * connector (back to the ready pose) is NOT checked.
If none of the K plans passes, the original top-1 is kept (no abstain).  Then the full teacher (robust_check._job, board 0.55, no bar,
original gripper) runs on the top-1 and, where different, on the selected plan.

    python grasp_check.py rank                    -> OUT/ranked.jsonl   (scorer only, no teacher)
    python grasp_check.py run SHIFT_M [WORKERS]   -> OUT/run_{SHIFT}.jsonl (resumable); SHIFT 0 = self-check (top-1 only)
    python grasp_check.py report                  -> OUT/report.md
Workers: 1 until hardware/robust/hw_tests.log contains HW_TESTS_DONE, then WORKERS (default 4)."""
import collections, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SH = HERE.parents[2]                                             # shelf_6x6_open_sides/
OUT = SH / "approachb_3d_learning/runs/explore/robust_grasp_check"
HZDIR = SH / "approachb_3d_learning/runs/ABF3/runs/hz"
DONE_LOG = SH / "hardware/robust/hw_tests.log"
K, SEED = 5, 0


def _rc():
    """robust_check (unchanged) with its stores redirected into OUT (it sets them at import; the teacher reads them at boot)."""
    sys.path.insert(0, str(SH / "hardware/robust"))
    import robust_check as RC
    os.environ.update(V3OUT=str(OUT / "teach_out"), FIN3D_STORE=str(OUT / "store/fin3d_store"), ABF_P5DIR=str(OUT / "p5"))
    return RC


# ------------------------------------------------------------------------------------------------------- ranking (scorer only)
def rank():
    sys.path.insert(0, str(SH / "hybrid/src"))
    import hz_eval as HE
    HZ = HE.HZ
    import exhaustive as EX, search as S
    rule = HZ.FROZEN
    pool = {r["root_id"]: r for r in map(json.loads, open(HZDIR / f"pool_3d_val_s{SEED}.jsonl"))}
    rows = {r["root_id"]: r for r in map(json.loads, open(HZ.K3 / "runs/public/public_val_3d.jsonl"))}
    ref = {r["root_id"]: r for r in map(json.loads, open(HZ.K3 / "runs/predictions/FULL_s0/val_3d.jsonl"))}
    priv, L = HE.priv_of("3d", "val"), HE.labels("3d")
    scor = HZ.HazardScorer(HZDIR / f"model_3d_s{SEED}.pt")
    OUT.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT / "ranked.jsonl", "w") as f:
        for rid, pr in pool.items():
            if not pr["decisions"][0]:                           # rule 0 = FROZEN; only REPAIR roots
                continue
            root = S.Root(rows[rid], "3d")
            D = {int(o): [int(d) for d in v[:4]] for o, v in ref[rid]["proposals"].items()}
            assert HZ.pool_hash(D) == ref[rid]["pool_hash"], rid
            rec = EX.score_root(root, D, scor, [rule], 4, set(), 8192)
            dec = rec["per_M"]["4"]["decisions"][0]["seq"]
            assert dec == pr["decisions"][0], (rid, dec, pr["decisions"][0])
            cand = [(p, json.loads(sig)) for p, sig, *_ in rec["per_M"]["4"]["top"] if p >= rule["eta"]]
            cand.sort(key=lambda t: EX.key(-math.log(max(t[0], 1e-12)), len(t[1]), root.displacement(t[1]), t[1]))
            assert cand[0][1] == dec, rid                        # the ranked list starts with the frozen decision
            e = priv[rid]
            f.write(json.dumps({"root_id": rid, "K": e["K"], "path": HE.local(e["path"]), "ids": e["ids"], "geom": e["geom"],
                                "p_noop": rec["p_noop"], "ranked": [{"seq": s, "p11": p} for p, s in cand[:K]],
                                "cached_top1": L.get((rid, json.dumps(dec)))}) + "\n")
            n += 1
    print(n, "REPAIR roots ranked")


# ------------------------------------------------------------------------------------------------------- worker side
def _init(robust):
    os.environ["ROBUST"] = robust
    RC = _rc()
    RC._init("original")
    RC._config(0.55, "none")
    globals()["RC"] = RC


def _last_q(plan):
    return [s for s in plan["segments"] if s["kind"] != "grip"][-1]["q"][-1]


def _query(A, e, name, dst, q0):
    """= bd_3dt.query, but from q0 (the end pose of the previous move's plan) instead of the arm's current pose."""
    TS = A.env()["TS"]
    i = e.obj_id(name)
    e.objects[i]["pos"] = e.obj_pos(i)
    why = {}
    for g in TS.grasp_poses(e.objects[i], e.table_z):
        p = e.plan_pick_place(name, tuple(dst), g, q_start=q0)
        if p["feasible"]:
            return p, None
        why[g["id"]] = p["reason"][:90]
    return None, A.label(None, {"grasp_reasons": why, "tried": len(why)}).get("reason")


def grasp_check(rec, steps):
    """steps [(name, (r, c))] -> dict(ok, fail ('relocation:k' | 'target' | None), reason, s).  Plans only; nothing is executed."""
    A = sys.modules["bd_3dt"]
    t0 = time.time()
    objs, cells0 = rec["objects"], rec["initial_cells"]
    e = A.build(objs, cells0)
    A.settle(e, cells0)
    q0 = e.q_now()
    for j, (o, d) in enumerate(steps):
        p, why = _query(A, e, o, d, q0)
        if p is None:
            return dict(ok=False, fail=f"relocation:{j}", reason=why, s=round(time.time() - t0, 2))
        q0 = _last_q(p)
        e.teleport(o, tuple(d))
    A.settle(e, {**{k: tuple(v) for k, v in cells0.items()}, **dict(steps)})
    p, qi = A.query(e, "t0", tuple(rec["goal"]))
    ok = p is not None
    return dict(ok=ok, fail=None if ok else "target", reason=None if ok else A.label(None, qi).get("reason"), s=round(time.time() - t0, 2))


def _job(job):
    rec = json.loads(Path(job["path"]).read_text())
    steps_of = lambda seq: [(job["ids"][o], (c // 6, c % 6)) for o, c in seq]  # noqa: E731
    cands = job["ranked"] if job["shift"] else []               # self-check run (shift 0): teacher on top-1 only, no grasp check
    checks, pick = [], None
    for i, c in enumerate(cands):
        checks.append({"i": i, **grasp_check(rec, steps_of(c["seq"]))})
        if checks[-1]["ok"]:
            pick = i
            break
    use = pick if pick is not None else 0                        # none passed -> keep the original decision
    base = dict(plan=0, zb=0.55, bar="none", path=job["path"], ids=job["ids"], cached=job["cached_top1"])
    teach = {"top1": RC._job({**base, "tag": "top1", "seq": job["ranked"][0]["seq"]})}
    if use:
        teach["gc"] = RC._job({**base, "tag": "gc", "seq": job["ranked"][use]["seq"]})
    return dict(root_id=job["root_id"], K=job["K"], shift=job["shift"], robust=os.environ["ROBUST"], n_ranked=len(job["ranked"]),
                checks=checks, pick=pick, use=use, teach=teach)


def run(shift, workers):
    from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
    import multiprocessing as mp
    jobs = [json.loads(l) for l in open(OUT / "ranked.jsonl")]
    fo = OUT / f"run_{float(shift)}.jsonl"
    done = {json.loads(l)["root_id"] for l in open(fo)} if fo.exists() else set()
    todo = [{**j, "shift": float(shift)} for j in jobs if j["root_id"] not in done]
    robust = f"{shift},1,1"
    print(f"{time.strftime('%H:%M:%S')} shift {shift}: {len(todo)} jobs", flush=True)
    cap = lambda: workers if "HW_TESTS_DONE" in DONE_LOG.read_text() else 1  # noqa: E731
    # spawn context: workers start on demand, so only `cap()` processes ever exist while the cap is 1
    with open(fo, "a") as f, ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init, initargs=(robust,)) as ex:
        live, n = set(), 0
        while todo or live:
            while todo and len(live) < cap():
                live.add(ex.submit(_job, todo.pop(0)))
            fin, live = wait(live, timeout=30, return_when=FIRST_COMPLETED)
            for fu in fin:
                f.write(json.dumps(fu.result()) + "\n"); f.flush()
                n += 1
                if n % 10 == 0:
                    print(f"{time.strftime('%H:%M:%S')} {n} done, {len(todo)} left, cap {cap()}", flush=True)


# ------------------------------------------------------------------------------------------------------- report
def report():
    ok = lambda t: t is not None and tuple(t["got"]) == (1, 1)  # noqa: E731
    lines = ["# grasp check before acting -- EXPLORATORY, 3D VAL, hazard s0, frozen rule, K = 5", ""]
    R0 = {r["root_id"]: r for r in map(json.loads, open(OUT / "run_0.0.jsonl"))} if (OUT / "run_0.0.jsonl").exists() else {}
    if R0:
        dec = [r for r in R0.values() if r["teach"]["top1"]["cached"] and None not in r["teach"]["top1"]["cached"]]
        ag = sum(tuple(r["teach"]["top1"]["got"]) == tuple(r["teach"]["top1"]["cached"]) for r in dec)
        nf = sum(r["teach"]["top1"]["x_status"] == "no feasible side grasp" or r["teach"]["top1"]["y_status"] == "no feasible side grasp" for r in R0.values())
        lines += [f"Self-check (shift 0): top-1 teacher (x,y) == cached label on {ag}/{len(dec)} roots with a decided cached label "
                  f"({100 * ag / max(1, len(dec)):.1f}%); {len(R0) - len(dec)} roots have no decided cached label.  Teacher 'no feasible side grasp' "
                  f"on top-1 at shift 0: {nf}.", ""]
    lines += ["| shift mm | roots | baseline top-1 success | grasp-check success | paired: base ok/gc fail | base fail/gc ok | plan changed | none of 5 passed |",
              "|---|---|---|---|---|---|---|---|"]
    tail = []
    for sh in ("0.005", "0.01"):
        p = OUT / f"run_{float(sh)}.jsonl"
        if not p.exists():
            continue
        R = [json.loads(l) for l in open(p)]
        n = len(R)
        b = [ok(r["teach"]["top1"]) for r in R]
        g = [ok(r["teach"]["gc"]) if r["use"] else ok(r["teach"]["top1"]) for r in R]
        lines.append(f"| {float(sh) * 1000:.0f} | {n} | {sum(b)} ({100 * sum(b) / n:.1f}%) | {sum(g)} ({100 * sum(g) / n:.1f}%) | "
                     f"{sum(x and not y for x, y in zip(b, g))} | {sum(y and not x for x, y in zip(b, g))} | {sum(bool(r['use']) for r in R)} | "
                     f"{sum(r['pick'] is None for r in R)} |")
        fin = [r["teach"]["gc"] if r["use"] else r["teach"]["top1"] for r in R]
        reasons = collections.Counter(f"{'X' if t['got'][0] != 1 else 'Y'}: {t['x_status'] if t['got'][0] != 1 else t['y_status']}"[:70]
                                      for t in fin if not ok(t))
        base_reasons = collections.Counter(f"{'X' if t['got'][0] != 1 else 'Y'}: {t['x_status'] if t['got'][0] != 1 else t['y_status']}"[:70]
                                           for t in (r["teach"]["top1"] for r in R) if not ok(t))
        secs = [c["s"] for r in R for c in r["checks"]]
        first_fail = sum(not r["checks"][0]["ok"] for r in R)
        tail += [f"## shift {float(sh) * 1000:.0f} mm", f"- grasp check time per plan: median {sorted(secs)[len(secs) // 2]:.1f} s, mean "
                 f"{sum(secs) / len(secs):.1f} s, max {max(secs):.1f} s ({len(secs)} checks)",
                 f"- top-1 failed the grasp check: {first_fail}; roots with < {K} admissible plans: {sum(r['n_ranked'] < K for r in R)}",
                 "- remaining failures with grasp check: " + ", ".join(f"{k} x{v}" for k, v in reasons.most_common()),
                 "- baseline failures: " + ", ".join(f"{k} x{v}" for k, v in base_reasons.most_common()), ""]
    (OUT / "report.md").write_text("\n".join(lines + [""] + tail) + "\n")
    print("\n".join(lines + [""] + tail))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "rank":
        rank()
    elif c == "run":
        run(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4)
    elif c == "report":
        report()
    else:
        raise SystemExit(__doc__)
