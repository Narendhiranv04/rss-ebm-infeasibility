"""EXPLORATORY (VAL-exploratory throughout) noise-aware labels for the per-step hazard scorer (3D).  Labels from TRAIN, evaluation on
VAL; TEST is never read.  The frozen rule (noop_gate 0.5, eta 0.3) was tuned with the CURRENT model, so it is not tuned for A / B / C.

  relabel noisy W [N]  re-run the UNCHANGED teacher (evaluate._verify_job, as robust_check.py) on every TRAIN plan with one random shift:
                       length ~ U[0, 10) mm per plan, direction per object as robust_check (layout + object name) but SALTED, so it
                       differs from the robust_check evaluation draws.  Plans = every X seq of a root + each Y edit whose end cells no
                       X seq reaches.  -> relabel_train_3d.jsonl (append, resumes; N = first N plans, smoke only)
  relabel c W [N]      the same job at ZERO shift, only on nominal TRAIN X failures without a failed-step label -> relabel_c_train_3d.jsonl
  train V SEED         twod_hazard.train unchanged (same Net, hyper-parameters, nominal = clean VAL early stop) on V's TRAIN rows:
                       A = noisy rows only, B = nominal + noisy rows, C = nominal rows with the zero-shift failed step filled in (control:
                       same step information as A / B, no noise).  -> V/model_3d_sSEED.pt, V/train_3d_sSEED.json
  pool V SEED i/22     twod_hazard.pool_runs on the V3.4 VAL pools: every rule incl. the FROZEN one = decisions[0]
  robust LVLS W [N]    robust_check._job (unchanged harness, unsalted evaluation draws) on the frozen-rule VAL task picks of current / A /
                       B / C, seeds 0-2, shift JIT_M in LVLS (0,0.005,0.01).  -> robust_val_3d.jsonl (append, resumes)
  report               -> report_val_3d.json (see report())
  check | selftest
Env NL_SMOKE=1: outputs under .../noise_labels/smoke/, 1 training epoch, no completeness / root-count asserts (smoke test only).
Nothing outside runs/explore/noise_labels/ is written; frozen code is imported read-only and monkeypatched in this process only.
"""
import collections, hashlib, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SH = HERE.parents[2]                                             # shelf_6x6_open_sides/
sys.path[:0] = [str(SH / "hardware/robust"), str(SH / "hybrid/src")]
import robust_check as RC  # noqa: E402  (sets the teacher env + V3.4 code path)

SMOKE = os.environ.get("NL_SMOKE", "")                          # "1" -> smoke/, other value v -> smoke_v/
OUT = (SH / "approachb_3d_learning/runs/explore/noise_labels" / ({"": ".", "1": "smoke"}.get(SMOKE) or f"smoke_{SMOKE}")).resolve()
SMOKE = bool(SMOKE)
ENV_OUT = dict(V3OUT=str(OUT / "teach_out"), FIN3D_STORE=str(OUT / "teach_out/fin3d_store"), ABF_P5DIR=str(OUT / "p5"))
os.environ.update(ENV_OUT)
SALT = "noise_labels_v1|"
MAXJ = 0.010                                                     # metres
SEEDS, VARIANTS = (0, 1, 2), ("A", "B", "C")
NPART = 60 if SMOKE else 22                                      # pool parts (= launcher W)
SNAP = RC.ABF / "snapshot/abf_v32_fb0eef4f5ff433ee/data3d.jsonl"
PUB = RC.K3 / "runs/public"
FROZEN_HZ = RC.ABF / "runs/hz"                                   # current model_3d_s{s}.pt + its VAL pools (read only)
RELABEL, RELABEL_C, ROBUST_F = OUT / "relabel_train_3d.jsonl", OUT / "relabel_c_train_3d.jsonl", OUT / "robust_val_3d.jsonl"
BAD = collections.Counter()                                      # malformed jsonl lines skipped, per file


def jl(p):
    if not Path(p).exists():
        return []
    out = []
    for l in open(p):
        try:
            out.append(json.loads(l))
        except json.JSONDecodeError:                             # e.g. a line cut by a kill; the item is simply redone
            BAD[Path(p).name] += 1
    return out


def train_rows():
    return [r for r in map(json.loads, open(SNAP)) if r["split"] == "train"]


def plans():
    """-> [(row, seq)]: every distinct X seq, then each Y edit whose end cells (sorted seq, = steps_of target key) no plan covers yet."""
    out = []
    for r in train_rows():
        ks = list(dict.fromkeys(json.dumps(q["seq"]) for q in r["labels"]["X"]))
        ends = {json.dumps(sorted(json.loads(k))) for k in ks}
        for y in r["labels"]["Y"]:
            if (e := json.dumps(sorted(y["edit"]))) not in ends:
                ends.add(e); ks.append(json.dumps(y["edit"]))
        out += [(r, json.loads(k)) for k in ks]
    return out


def c_plans():
    """nominal TRAIN X failures without a failed-step label (distinct seq per root)."""
    return [(r, json.loads(k)) for r in train_rows()
            for k in dict.fromkeys(json.dumps(q["seq"]) for q in r["labels"]["X"] if q["x"] == 0 and not q.get("x_phase"))]


def jit_of(rid, seq):
    """one shift length per plan, uniform in [0, MAXJ), deterministic (resumable)."""
    return int(hashlib.sha256(f"{SALT}{rid}|{json.dumps(seq)}".encode()).hexdigest(), 16) % 10**6 / 10**6 * MAXJ


def angle(key, name, salt):
    """salt '' = robust_check's evaluation direction (selftest checks the formula against its source)."""
    return int(hashlib.sha256((salt + key + name).encode()).hexdigest(), 16) % 3600 / 3600 * 2 * math.pi


# ------------------------------------------------------------------------------------------- teacher workers
_J = {"jit": 0.0, "npert": 0}


def _init_relabel(geom):
    os.environ["ROBUST"] = "0,1,1"                                # robust_check's own (unsalted) shift off; the salted one below
    RC._init(geom)
    S = RC._W["S"]
    norm0 = S.XArm7TableEnv._normalize

    def normalize(self, objects):
        out = norm0(self, objects)
        if _J["jit"]:
            key = repr(sorted(tuple(o["cell"]) for o in out))
            for o in out:
                a = angle(key, o["name"], SALT)
                o["pos"][:2] += (_J["jit"] * math.cos(a), _J["jit"] * math.sin(a))
                o["home_pos"] = o["pos"].copy()
                _J["npert"] += 1
        return out
    S.XArm7TableEnv._normalize = normalize


def _relabel_job(job):
    import evaluate as EV
    _J["jit"], _J["npert"] = job["jit"], 0
    r = EV._verify_job({"dom": "3D", "path": job["path"], "seqs": [[[job["ids"][o], c] for o, c in job["seq"]]]})
    x = r["results"][0]
    xr, yr = x["x"] or {}, x["y"] or {}
    return dict(root_id=job["root_id"], seq=job["seq"], jit_m=round(job["jit"], 6), salt=SALT, got=list(RC.xy(x)), x_phase=xr.get("phase"),
                x_status=xr.get("status") or xr.get("error", "")[:120], y_status=yr.get("status") or yr.get("error", "")[:120],
                shifted_objects=_J["npert"], cpu_s=r["cpu_s"])


def _pool(jobs, init, fn, out, workers):
    import multiprocessing as mp
    (OUT / "teach_out").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"{time.strftime('%H:%M:%S')} {fn.__name__}: {len(jobs)} jobs, {workers} workers -> {out}", flush=True)
    with open(out, "a") as f, mp.get_context("spawn").Pool(workers, initializer=init, initargs=("original",), maxtasksperchild=50) as pool:
        for i, r in enumerate(pool.imap_unordered(fn, jobs), 1):
            f.write(json.dumps(r) + "\n"); f.flush()
            if i % 25 == 0 or i == len(jobs):
                print(f"{time.strftime('%H:%M:%S')} {i}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)


def _src(kind):
    return {"noisy": (plans, RELABEL), "c": (c_plans, RELABEL_C)}[kind]


def missing(kind):
    fn, fo = _src(kind)
    done = {(r["root_id"], json.dumps(r["seq"])) for r in jl(fo)}
    return [(r, s) for r, s in fn() if (r["root_id"], json.dumps(s)) not in done]


def relabel(kind, workers, limit=None):
    priv = {e["root_id"]: e for e in jl(PUB / "private_train_3d.jsonl")}
    fn, fo = _src(kind)
    ps = fn()[:limit] if limit else fn()
    done = {(r["root_id"], json.dumps(r["seq"])) for r in jl(fo)}
    jobs = []
    for r, s in ps:
        e = priv[r["root_id"]]
        assert r["split"] == "train" and e["ids"] == r["inputs"]["ids"] and e["geom"] == "original", r["root_id"]
        if (r["root_id"], json.dumps(s)) not in done:
            jobs.append(dict(root_id=r["root_id"], seq=s, jit=jit_of(r["root_id"], s) if kind == "noisy" else 0.0,
                             path=RC.local(e["path"]), ids=e["ids"]))
    OUT.mkdir(parents=True, exist_ok=True)
    _pool(jobs, _init_relabel, _relabel_job, fo, workers)
    R = jl(fo)
    assert all(r["shifted_objects"] > 0 for r in R if r["jit_m"] > 0), "shift hook did not run"
    assert kind == "noisy" or all(r["jit_m"] == 0 for r in R)
    s = {"kind": kind, "plans_done": len(R), "of": len(fn()), "undecided": sum(None in r["got"] for r in R), "malformed_lines": dict(BAD),
         "x0_by_phase": collections.Counter(str(r["x_phase"]) for r in R if r["got"][0] == 0)}
    if kind == "c":   # determinism self-check: zero shift must reproduce the stored nominal x = 0
        s["zero_shift_x_equals_stored_0"] = sum(r["got"][0] == 0 for r in R)
        s["zero_shift_x_differs"] = sum(r["got"][0] == 1 for r in R)
    print(json.dumps(s), flush=True)


def noisy_rows():
    """TRAIN rows with labels from the shifted teacher: X seq -> its own run (x, failed step); Y edit -> y of a run with the same end cells."""
    by = collections.defaultdict(dict)
    for r in jl(RELABEL):
        by[r["root_id"]][json.dumps(r["seq"])] = r
    out = []
    for r in train_rows():
        lab = by.get(r["root_id"])
        if not lab:
            continue
        ends = {}
        for k in sorted(lab):
            ends.setdefault(json.dumps(sorted(lab[k]["seq"])), lab[k])
        X = [{"seq": q["seq"], "x": v["got"][0], "x_phase": v["x_phase"] if v["got"][0] == 0 else None}
             for q in r["labels"]["X"] if (v := lab.get(json.dumps(q["seq"]))) and v["got"][0] is not None]
        Y = [{"edit": y["edit"], "y": v["got"][1]}
             for y in r["labels"]["Y"] if (v := ends.get(json.dumps(sorted(y["edit"])))) and v["got"][1] is not None]
        if X or Y:
            out.append({**r, "labels": {"X": X, "Y": Y}})
    return out


def c_rows():
    """nominal TRAIN rows; an unlabelled-step failure gets the zero-shift failed step only when that run also says x = 0."""
    lab = {(r["root_id"], json.dumps(r["seq"])): r for r in jl(RELABEL_C)}
    out, n = [], collections.Counter()
    for r in train_rows():
        X = []
        for q in r["labels"]["X"]:
            v = lab.get((r["root_id"], json.dumps(q["seq"])))
            if q["x"] == 0 and not q.get("x_phase") and v and v["got"][0] == 0 and v["x_phase"]:
                q = {**q, "x_phase": v["x_phase"]}; n["filled"] += 1
            elif q["x"] == 0 and not q.get("x_phase"):
                n["left_unlabelled"] += 1
            X.append(q)
        out.append({**r, "labels": {**r["labels"], "X": X}})
    print(json.dumps({"C": dict(n)}), flush=True)
    return out


# ------------------------------------------------------------------------------------------- scorer
def bind(variant):
    import twod_hazard as HZ
    assert variant in VARIANTS, variant
    os.environ.update(ENV_OUT)                                    # twod_hazard just pointed ABF_P5DIR at the frozen runs/hz
    if "evaluate" in sys.modules:
        sys.modules["evaluate"].P5 = ENV_OUT["ABF_P5DIR"]
    HZ.OUT = OUT / variant
    HZ.OUT.mkdir(parents=True, exist_ok=True)
    assert_not_frozen(HZ)
    return HZ


def assert_not_frozen(HZ):
    fz = str(FROZEN_HZ.resolve())
    for p in [os.environ[k] for k in ENV_OUT] + [str(HZ.OUT), str(getattr(sys.modules.get("evaluate"), "P5", "") or "")]:
        assert not str(Path(p).resolve()).startswith(fz), f"would write into the frozen runs/hz: {p}"


def train(variant, seed):
    HZ = bind(variant)
    if not SMOKE:
        for k in {"A": ["noisy"], "B": ["noisy"], "C": ["c"]}[variant]:
            assert not missing(k), f"relabel {k} incomplete: {len(missing(k))} plans missing"
    load0 = HZ.load_rows
    rows = {"A": noisy_rows, "B": lambda: load0("3d", "train") + noisy_rows(), "C": c_rows}[variant]()
    assert rows, "no TRAIN rows"
    HZ.load_rows = lambda dom, split: load0(dom, split) if split != "train" else rows
    HZ.train("3d", seed, **({"epochs": 1} if SMOKE else {}))
    assert_not_frozen(HZ)
    f = HZ.OUT / f"train_3d_s{seed}.json"
    meta = json.loads(f.read_text())
    meta.update(label="EXPLORATORY (VAL early stop on clean nominal labels)", variant=variant, n_train_rows=len(rows), salt=SALT, smoke=SMOKE)
    f.write_text(json.dumps(meta, indent=1))


def pool(variant, seed, part):
    assert part.endswith(f"/{NPART}"), f"use i/{NPART} parts"
    HZ = bind(variant)
    HZ.pool_runs("3d", "val", seed, part)
    assert_not_frozen(HZ)


def picks():
    """{(model, seed): {root_id: frozen-rule decision}}  (None = abstain, [] = no-op).  Models not pooled yet are left out."""
    P = {}
    for s in SEEDS:
        srcs = [("current", sorted(FROZEN_HZ.glob(f"pool_3d_val_s{s}*.jsonl")))]
        srcs += [(v, sorted((OUT / v).glob(f"pool_3d_val_s{s}.part*of{NPART}.jsonl"))) for v in VARIANTS]
        for m, files in srcs:
            d = {}
            for p in files:
                for r in jl(p):
                    assert d.setdefault(r["root_id"], r["decisions"][0]) == r["decisions"][0], f"{p.name}: pool files disagree"
            if d:
                P[(m, s)] = d
        cur = set(P.get(("current", s), {}))
        for v in VARIANTS:
            if (v, s) in P:
                got = set(P[(v, s)])
                assert got <= cur if SMOKE else got == cur, f"{v} s{s}: {len(got)} roots scored, current has {len(cur)}"
    return P


def robust(levels, workers, limit=None):
    priv = {e["root_id"]: e for e in jl(PUB / "private_val_3d.jsonl")}
    P = picks()
    common = set.intersection(*map(set, P.values()))
    want = sorted({(rid, json.dumps(d[rid])) for d in P.values() for rid in common if d[rid]})
    want = want[:limit] if limit else want
    done = {(r["robust"], r["plan"]) for r in jl(ROBUST_F)}
    for lv in levels.split(","):
        os.environ["ROBUST"] = f"{lv},1,1"                        # read by robust_check._init in each new worker
        jobs = [dict(tag="noise_labels", plan=f"{rid}|{s}", zb=0.55, bar="none", path=RC.local(priv[rid]["path"]), ids=priv[rid]["ids"],
                     seq=json.loads(s), cached=[None, None]) for rid, s in want if (os.environ["ROBUST"], f"{rid}|{s}") not in done]
        assert all(priv[j["plan"].split("|")[0]]["geom"] == "original" for j in jobs)
        _pool(jobs, RC._init, RC._job, ROBUST_F, workers)


# ------------------------------------------------------------------------------------------- report
def cluster_boot(items, B=2000, seed=0):
    """items [(world, diff)] -> [mean, lo, hi] (95 %, resampling worlds)."""
    import numpy as np
    by = collections.defaultdict(list)
    for c, d in items:
        by[c].append(d)
    sums, ns = np.array([sum(v) for v in by.values()], float), np.array([len(v) for v in by.values()], float)
    idx = np.random.default_rng(seed).integers(0, len(by), (B, len(by)))
    est = sums[idx].sum(1) / ns[idx].sum(1)
    return [round(100 * sums.sum() / ns.sum(), 2), round(100 * float(np.percentile(est, 2.5)), 2), round(100 * float(np.percentile(est, 97.5)), 2)]


def report():
    import hz_eval as H
    priv, P, R = H.priv_of("3d", "val"), picks(), jl(ROBUST_F)
    common = sorted(set.intersection(*map(set, P.values())))
    mm = lambda r: round(float(r["robust"].split(",")[0]) * 1000)  # noqa: E731
    run = {(mm(r), *r["plan"].split("|", 1)): tuple(r["got"]) for r in R}   # (mm, root, seq) -> (x, y)
    ok = lambda lv, rid, d: None if None in run.get((lv, rid, json.dumps(d)), (None,)) else float(run[(lv, rid, json.dumps(d))] == (1, 1))  # noqa: E731
    stored = H.labels("3d")                                       # nominal teacher labels (their cache + ours), read only
    levels = sorted({k[0] for k in run})
    res = {"label": "EXPLORATORY, VAL-exploratory throughout (3D VAL; TEST never read)",
           "notes": ["The frozen rule (noop_gate 0.5, eta 0.3) was tuned on VAL with the CURRENT model; A / B / C use it untuned.",
                     "A, B and C are early-stopped on clean (nominal) VAL labels; A never sees a clean TRAIN label.",
                     "Shift rows are TASK success only (x = y = 1 in the shifted re-run of the picked plan). No-op and abstain were NOT "
                     "tested under shift; the nominal task / noop / abstain line uses stored nominal labels (gaps: 0 mm re-run).",
                     "C = control: nominal labels + the zero-shift failed step, i.e. the step information A / B get, without noise."],
           "n_roots": len(common), "salt": SALT, "smoke": SMOKE, "malformed_lines": dict(BAD), "models": {}, "paired": {}}
    for (m, s), d in sorted(P.items()):
        task = [rid for rid in common if d[rid]]
        L = dict(stored)
        for (lv, rid, q), v in run.items():
            if lv == 0 and None in L.get((rid, q), (None, None)):
                L[(rid, q)] = v
        e = {"decisions": dict(collections.Counter("abstain" if d[r] is None else "noop" if d[r] == [] else "task" for r in common)),
             "nominal_frozen_rule": H.rates([{"masks": priv[rid]["masks"], "outcome": H.V.outcome(d[rid], L, rid)} for rid in common])}
        for lv in levels:
            v = [x for rid in task if (x := ok(lv, rid, d[rid])) is not None]
            e[f"task_success_{lv}mm"] = {"rate": round(100 * sum(v) / len(v), 1) if v else None, "n": len(v), "undecided": len(task) - len(v)}
        for lv in [x for x in levels if x and 0 in levels]:       # own drop, on picks decided at both levels
            pr = [(a, b) for rid in task if (a := ok(0, rid, d[rid])) is not None and (b := ok(lv, rid, d[rid])) is not None]
            e[f"drop_0_to_{lv}mm"] = round(100 * sum(a - b for a, b in pr) / len(pr), 1) if pr else None
        sv = [(stored[k][0] == 1 and stored[k][1] == 1, run[(0, rid, k[1])] == (1, 1)) for rid in task
              if None not in stored.get(k := (rid, json.dumps(d[rid])), (None,)) and None not in run.get((0, rid, k[1]), (None,))]
        e["sanity_0mm_vs_stored"] = {"n": len(sv), "agree": sum(a == b for a, b in sv)}
        tf = (FROZEN_HZ if m == "current" else OUT / m) / f"train_3d_s{s}.json"
        if tf.exists():
            e["best_val_nll"] = json.loads(tf.read_text())["best_val_nll"]
        res["models"][f"{m}_s{s}"] = e
    for m1, m2 in [("A", "C"), ("B", "C"), ("A", "current"), ("B", "current"), ("C", "current")]:
        for lv in [x for x in levels if x]:
            items, disc = [], collections.Counter()
            for s in SEEDS:
                if (m1, s) not in P or (m2, s) not in P:
                    continue
                for rid in common:
                    d1, d2 = P[(m1, s)][rid], P[(m2, s)][rid]
                    if d1 and d2 and (a := ok(lv, rid, d1)) is not None and (b := ok(lv, rid, d2)) is not None:
                        items.append((H._cluster(rid), a - b)); disc[(int(a), int(b))] += 1
            if items:
                res["paired"][f"{m1}_vs_{m2}_{lv}mm"] = {
                    "unit": "(root, seed) where both models pick a task; seeds paired (A_s vs C_s)", "n": len(items),
                    "worlds": len({c for c, _ in items}), "diff_pts_mean_ci95_world_bootstrap_B2000": cluster_boot(items),
                    "only_m1_ok": disc[(1, 0)], "only_m2_ok": disc[(0, 1)], "both_ok": disc[(1, 1)], "both_fail": disc[(0, 0)]}
    (OUT / "report_val_3d.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


def check():
    need = [SNAP, PUB / "private_train_3d.jsonl", PUB / "private_val_3d.jsonl", PUB / "public_val_3d.jsonl",
            RC.K3 / "runs/predictions/FULL_s0/val_3d.jsonl", Path(os.environ["FIN3D_CODE"])]
    need += [FROZEN_HZ / f"model_3d_s{s}.pt" for s in SEEDS]
    miss = [str(p) for p in need if not p.exists()]
    nval = len(jl(PUB / "public_val_3d.jsonl"))
    for s in SEEDS:
        n = len({r["root_id"] for p in FROZEN_HZ.glob(f"pool_3d_val_s{s}*.jsonl") for r in jl(p)})
        if n != nval:
            miss.append(f"{FROZEN_HZ}/pool_3d_val_s{s}*.jsonl covers {n} of {nval} VAL roots")
    if not miss:
        priv = {e["root_id"]: e for e in jl(PUB / "private_train_3d.jsonl")}
        miss = sorted({p for r in train_rows() if not Path(p := RC.local(priv[r["root_id"]]["path"])).exists()})
    print(json.dumps({"label": "EXPLORATORY", "missing": miss[:20], "n_missing": len(miss)}))
    raise SystemExit(1 if miss else 0)


def selftest():
    import inspect
    ps = plans()
    assert len({(r["root_id"], json.dumps(s)) for r, s in ps}) == len(ps)
    j = [jit_of(r["root_id"], s) for r, s in ps[:2000]]
    assert 0 <= min(j) and max(j) < MAXJ and max(j) > 0.9 * MAXJ and abs(sum(j) / len(j) - MAXJ / 2) < 0.0005
    assert jit_of("a", [[1, 2]]) == jit_of("a", [[1, 2]]) != jit_of("a", [[1, 3]])
    # angle(.., '') must be robust_check's evaluation formula; then the noise OFFSETS must differ from the 5 / 10 mm evaluation offsets
    assert 'hashlib.sha256((key + o["name"]).encode()).hexdigest(), 16) % 3600 / 3600 * 2 * math.pi' in inspect.getsource(RC._init)
    close = tot = 0
    for r, s in ps[:300]:
        key = repr(sorted((c // 6, c % 6) for c in r["inputs"]["cells0"]))
        for name in r["inputs"]["ids"]:
            jn, an, ae = jit_of(r["root_id"], s), angle(key, name, SALT), angle(key, name, "")
            for ev in (0.005, 0.01):
                close += math.hypot(jn * math.cos(an) - ev * math.cos(ae), jn * math.sin(an) - ev * math.sin(ae)) < 0.001; tot += 1
    assert close / tot < 0.05, (close, tot)
    cov = collections.defaultdict(set)                            # every X seq and every Y end-cell set is covered by some plan
    for r, s in ps:
        cov[r["root_id"]] |= {json.dumps(s), "E" + json.dumps(sorted(s))}
    for r in train_rows():
        assert all(json.dumps(q["seq"]) in cov[r["root_id"]] for q in r["labels"]["X"])
        assert all("E" + json.dumps(sorted(y["edit"])) in cov[r["root_id"]] for y in r["labels"]["Y"])
    assert cluster_boot([("w1", 1.0), ("w1", 0.0), ("w2", 1.0)], B=200)[0] == 66.67
    print(json.dumps({"ok": True, "plans": len(ps), "c_plans": len(c_plans()), "offsets_within_1mm": f"{close}/{tot}"}))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "relabel":
        relabel(a[0], int(a[1]), int(a[2]) if len(a) > 2 else None)
    elif c == "train":
        train(a[0], int(a[1]))
    elif c == "pool":
        pool(a[0], int(a[1]), a[2])
    elif c == "robust":
        robust(a[0], int(a[1]), int(a[2]) if len(a) > 2 else None)
    elif c in ("report", "check", "selftest"):
        globals()[c]()
    else:
        raise SystemExit(__doc__)
