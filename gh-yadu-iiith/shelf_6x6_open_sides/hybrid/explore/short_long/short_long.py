"""EXPLORATORY, 3D VAL only: train on short-plan scenes (K0, K1, K2, NONE), test on long-plan scenes (K3).  TEST is never read or scored.

Both scorers are retrained on the expanded TRAIN snapshot (abf_v32_fb0eef4f5ff433ee) with every K3 root removed from TRAIN and from the VAL
used for early stopping.  Everything else is unchanged: frozen cVAE pools (stored FULL_s0 VAL proposals, pool hash checked), frozen rule
(noop gate 0.5, eta 0.3), unchanged teacher, BM.correct masks.
  hz  = per-step hazard scorer: twod_hazard.train / pool_runs, only SNAP and OUT re-bound in this process (as encoders.py does)
  xy  = XYNet FULL (= xy.fit arm D, the bundle scorer recipe: 60 epochs, batch 64, lr 1e-3, patience 10), on CPU
  xy_ref = the frozen bundle scorer_s0.pt through the same xy scoring path (parity check against FULL_s0's stored VAL decisions)
Caveat: non-K3 TRAIN scenes still carry 3-move labels (X/Y plans of length 3); counts are written to snapshot/manifest.json.

    python short_long.py prep                          filtered snapshot + self-checks          (venv python)
    python short_long.py train_hz SEED                                                          (venv python)
    python short_long.py train_xy SEED                                                          (venv python)
    python short_long.py pool ARM SEED i/k [MODEL]     ARM hz | xy | xy_ref                     (venv python)
    python short_long.py eval ARM SEED WORKERS         WORKERS 0 = no teacher, missing labels stay UNKNOWN   (env3d python)
Env: SL_VARIANT noK3scenes (default) | no3moveplans (also drops every 3-move X / Y label of any K) -> runs/explore/short_long/{variant}/.
     SL_SMOKE=1 -> runs/explore/short_long/{variant}/smoke, 2 epochs.  SL_THREADS (default 2) torch threads for training.
"""
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import collections, hashlib, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import twod_hazard as HZ  # noqa: E402  (sets K3 / ABF env + V3.4 code path, XY_DEV=cpu)

SMOKE = bool(os.environ.get("SL_SMOKE"))
VARIANT = os.environ.get("SL_VARIANT", "noK3scenes")      # noK3scenes | no3moveplans (also drops every 3-move X / Y label, any K)
assert VARIANT in ("noK3scenes", "no3moveplans", "noK3scenes_nostep", "no3moveplans_nostep", "fulldata_nostep"), VARIANT
FULLDATA = VARIANT.startswith("fulldata")   # keep K3 scenes: same TRAIN / VAL as the frozen scorer, only the step input changes
STRICT = VARIANT.startswith("no3moveplans")
NOSTEP = VARIANT.endswith("_nostep")     # our scorer without the step-number input (k / 3 feature set to 0 in training and scoring)
if NOSTEP:
    _feat = HZ.featurize

    def _feat_nostep(root, steps):
        out = _feat(root, steps)
        out[-1][:, -1] = 0.0                                   # last column of the per-step extras = k / 3
        assert float(out[-1][:, -1].abs().sum()) == 0.0
        return out
    HZ.featurize = _feat_nostep                                # train() and HazardScorer.score() look it up at call time
OUT = HZ.RUNS / "explore/short_long" / VARIANT / ("smoke" if SMOKE else "")
SNAP = OUT / "snapshot"
EPOCHS = 2 if SMOKE else None
THREADS = int(os.environ.get("SL_THREADS", "2"))
DOM = "3d"
assert "frozen" not in str(OUT) and OUT != HZ.ABF / "runs/hz"


def priv(split):
    return {json.loads(l)["root_id"]: json.loads(l) for l in open(HZ.K3 / "runs/public" / f"private_{split}_{DOM}.jsonl")}


def prep():
    """filtered snapshot: TRAIN + VAL rows, K3 removed; TEST rows are skipped (never written)."""
    SNAP.mkdir(parents=True, exist_ok=True)
    keep, n3 = [], collections.Counter()
    for l in open(HZ.ABF / "snapshot/abf_v32_fb0eef4f5ff433ee" / f"data{DOM}.jsonl"):
        r = json.loads(l)
        if r["split"] not in ("train", "val") or (r["K_legacy"] == "3" and not FULLDATA):
            continue
        if STRICT:                                          # a 3-move plan's label also carries its 1- and 2-move prefixes: drop it whole
            n3["X_len3_dropped"] += sum(len(q["seq"]) >= 3 for q in r["labels"]["X"])
            n3["Y_len3_dropped"] += sum(len(y["edit"]) >= 3 for y in r["labels"]["Y"])
            r["labels"]["X"] = [q for q in r["labels"]["X"] if len(q["seq"]) < 3]
            r["labels"]["Y"] = [y for y in r["labels"]["Y"] if len(y["edit"]) < 3]
            l = json.dumps(r) + "\n"
        keep.append(l)
        n3["X_len3"] += sum(len(q["seq"]) >= 3 for q in r["labels"]["X"])
        n3["Y_len3"] += sum(len(y["edit"]) >= 3 for y in r["labels"]["Y"])
    rows = [json.loads(l) for l in keep]
    # self-check: zero K3 roots, zero TEST rows, exactly the private TRAIN / VAL roots minus their K3 roots
    assert not [r for r in rows if (r["K_legacy"] == "3" and not FULLDATA) or r["split"] not in ("train", "val")]
    if STRICT:   # zero 3-move plans left in TRAIN or in the early-stopping VAL; every root keeps its no-op Y label
        assert not [1 for r in rows for q in r["labels"]["X"] if len(q["seq"]) >= 3] and not [1 for r in rows for y in r["labels"]["Y"] if len(y["edit"]) >= 3]
        assert n3["X_len3"] == n3["Y_len3"] == 0 and all(any(y["edit"] == [] for y in r["labels"]["Y"]) for r in rows)
    for sp, n in (("train", 1326), ("val", 294)):
        want = {k for k, e in priv(sp).items() if FULLDATA or str(e["K"]) != "3"}
        got = {r["root_id"] for r in rows if r["split"] == sp}
        assert got == want and (FULLDATA or len(got) == n), (sp, len(got), len(want))
    (SNAP / f"data{DOM}.jsonl").write_text("".join(keep))
    man = {"label": f"EXPLORATORY; TRAIN/VAL without K3; variant {VARIANT}", "by_split_K": {f"{s}_{k}": c for (s, k), c in
           sorted(collections.Counter((r["split"], r["K_legacy"]) for r in rows).items())}, "len3_labels_kept": dict(n3),
           "source": "abf_v32_fb0eef4f5ff433ee", "sha256": hashlib.sha256("".join(keep).encode()).hexdigest()}
    (SNAP / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man))


def train_hz(seed):
    HZ.SNAP, HZ.OUT = SNAP, OUT / "hz"
    HZ.OUT.mkdir(parents=True, exist_ok=True)
    os.environ["HZ_THREADS"] = str(THREADS)
    HZ.train(DOM, seed, **({"epochs": EPOCHS} if EPOCHS else {}))


def train_xy(seed):
    import torch
    import xy as XY
    torch.set_num_threads(THREADS)
    XY.fit(SNAP, DOM.upper(), "D", seed, out=OUT / "xy" / f"D_s{seed}", **({"epochs": EPOCHS} if EPOCHS else {}))


def pool(arm, seed, part, model=None):
    if arm == "hz":
        HZ.OUT = OUT / "hz"
        return HZ.pool_runs(DOM, "val", seed, part)
    import torch
    import exhaustive as EX, search as S
    sys.path.insert(0, str(HZ.CODE / "ablations_k3"))
    import run as RUN
    torch.set_num_threads(1)
    model = Path(model or (HZ.K3 / "stage/bundle" / DOM / f"scorer_s{seed}.pt" if arm == "xy_ref" else OUT / "xy" / f"D_s{seed}" / "model.pt"))
    rows = [json.loads(l) for l in open(HZ.K3 / "runs/public" / f"public_val_{DOM}.jsonl")]
    ref = {r["root_id"]: r for r in map(json.loads, open(HZ.K3 / "runs/predictions/FULL_s0" / f"val_{DOM}.jsonl"))}
    i_, k_ = map(int, part.split("/"))
    rows = [r for r in rows if int(hashlib.sha256(r["root_id"].encode()).hexdigest(), 16) % k_ == i_]   # = pool_runs partition
    d = OUT / arm
    d.mkdir(parents=True, exist_ok=True)
    fo = d / f"pool_{DOM}_val_s{seed}.part{i_}of{k_}.jsonl"
    done = {json.loads(l)["root_id"] for l in open(fo)} if fo.exists() else set()
    scor = RUN.SafeScorer(model, None, device="cpu")
    with open(fo, "a") as f:
        for r in (x for x in rows if x["root_id"] not in done):
            t = time.time()
            root = S.Root(r, DOM)
            D = {int(o): [int(c) for c in v[:4]] for o, v in ref[r["root_id"]]["proposals"].items()}
            assert HZ.pool_hash(D) == ref[r["root_id"]]["pool_hash"], r["root_id"]
            rec = EX.score_root(root, D, scor, [RUN.RULE], 4, set(), 8192)
            dec = rec["per_M"]["4"]["decisions"][0]
            f.write(json.dumps({"root_id": r["root_id"], "p_noop": rec["p_noop"], "decisions": [None if dec is None else dec["seq"]],
                                "model": str(model), "s": round(time.time() - t, 2)}) + "\n")
            f.flush()


def _full_dec(x):
    return None if x["decision"] == "ABSTAIN" else [] if x["decision"] == "NOOP" else x["seq"]


def by_k(H, P, dec, L):
    """{all, K0..K3, NONE: rates} over the roots in dec (decision per root: None | [] | seq)."""
    rows = collections.defaultdict(list)
    for rid, d in dec.items():
        x = {"masks": P[rid]["masks"], "outcome": H.V.outcome(d, L, rid)}
        rows["all"].append(x); rows["K" + str(P[rid]["K"])].append(x)
    return {g: H.rates(v) for g, v in sorted(rows.items())}


def evaluate(arm, seed, workers):
    os.environ["V3OUT"] = str(OUT / "teach_out")                  # teacher scratch stays in our tree
    os.environ["FIN3D_STORE"] = str(OUT / "teach_out/fin3d_store")
    import hz_eval as H
    P, dec = H.priv_of(DOM, "val"), {}
    for p in sorted((OUT / arm).glob(f"pool_{DOM}_val_s{seed}*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            dec.setdefault(r["root_id"], r["decisions"][0])        # [0] = frozen rule
    if not SMOKE:
        assert set(dec) == set(P) and len(P) == 364, f"{len(set(P) - set(dec))} VAL roots not scored"
    vf = OUT / f"verify_{DOM}.jsonl"
    L = H.labels(DOM)
    # Mac-teacher caches of the same unchanged teacher (read-only, other than our own vf)
    extra = [vf] + sorted((HZ.RUNS / "explore/enc").glob(f"*/verify_{DOM}.jsonl")) + sorted((HZ.RUNS / "explore/short_long").glob(f"*/verify_{DOM}.jsonl"))
    merge = lambda: [L.__setitem__(k, v) for f in extra if f.exists() for k, v in H.V.labels(DOM, str(f)).items() if None in L.get(k, (None, None))]  # noqa: E731
    merge()
    jobs = [{"root_id": rid, "dom": DOM.upper(), "path": H.local(P[rid]["path"]), "geom": P[rid].get("geom", "original"), "ids": P[rid]["ids"],
             "seqs": [[[P[rid]["ids"][o], c] for o, c in d]]} for rid, d in dec.items() if d and None in L.get((rid, json.dumps(d)), (None, None))]
    print(json.dumps({"arm": arm, "seed": seed, "roots": len(dec), "teacher_queries": len(jobs), "teacher_run": bool(workers and jobs)}), flush=True)
    if jobs and workers:
        H._run(DOM, jobs, workers, vf)
        merge()
    res = by_k(H, P, dec, L)
    # same-seed full-data references on the same roots (frozen files, read-only): ours runs/hz pools, XYNet FULL_s{seed}
    full_hz = {}
    for p in sorted(HZ.OUT.glob(f"pool_{DOM}_val_s{seed}*.jsonl")):
        full_hz.update({r["root_id"]: r["decisions"][0] for r in map(json.loads, open(p)) if r["root_id"] in dec})
    full_xy = {x["root_id"]: _full_dec(x) for x in H._full_rows(DOM, "val", seed) if x["root_id"] in dec}
    out = {"label": "EXPLORATORY; split 3D VAL; scorer trained without K3 scenes", "variant": VARIANT, "smoke": SMOKE, "arm": arm, "seed": seed,
           "no_k3": res, "fulldata_same_seed": {"hz": by_k(H, P, full_hz, L), "xy": by_k(H, P, full_xy, L)}}
    if arm == "xy_ref":                                             # parity: our xy scoring path == FULL's stored decisions
        mis = [r for r in dec if dec[r] != full_xy[r]]
        out["parity_mismatch"] = mis
        assert not mis, f"xy_ref differs from FULL_s{seed} on {len(mis)} roots"
    if not SMOKE:
        assert res["all"]["n_task"] + res["all"]["unknown_task"] == 220, res["all"]
    (OUT / f"metrics_{arm}_s{seed}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({g: {k: v[k] for k in ("task", "noop", "abstain", "n_task", "unknown_task")} for g, v in res.items()}))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "prep":
        prep()
    elif c == "train_hz":
        train_hz(int(a[0]))
    elif c == "train_xy":
        train_xy(int(a[0]))
    elif c == "pool":
        pool(a[0], int(a[1]), a[2], a[3] if len(a) > 3 else None)
    elif c == "eval":
        evaluate(a[0], int(a[1]), int(a[2]))
    else:
        raise SystemExit(__doc__)
