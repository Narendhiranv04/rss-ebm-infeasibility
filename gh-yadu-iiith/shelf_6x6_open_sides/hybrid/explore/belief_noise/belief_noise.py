"""EXPLORATORY perception-error test, 3D VAL seed 0 only (TEST is never read).  The planner sees WRONG object positions; the unchanged
teacher judges the picked plan on the TRUE (unshifted) scene.

Noise: every object (target included) gets an xy offset sigma * N(0, I2), the N(0, I2) draw seeded by sha256(SALT|root|object id), so the
direction is fixed per object and only its size grows with sigma.  Each axis is clamped to +-37 mm (half the 75 mm cell pitch is 37.5 mm):
both models also see the object's cell index (row / col features, occupancy), so a position outside the cell would contradict that input.
Where it enters (no file edited; patched in this process only):
  cVAE + hazard scorer  placement.scene_features -> wrapped: an object still in its start cell gets x, y (+ dist, angle, recomputed) shifted;
                        a hypothetically moved object sits at its destination cell centre, exactly as before.
  XYNet                 search.Scorer reads row["inputs"]["xy0"] (3D: start cell centres) -> the row passed in carries the shifted xy0.
Both arms use the same (noisy) cVAE pool per root and level.  Levels: '0rec' = recorded GPU pool (the stored VAL pool, reproduction check),
'0' = CPU cVAE without noise (reference for the noise rows), then each sigma (mm) > 0 with the CPU cVAE.

    python belief_noise.py check [N]            self-checks on N VAL roots (default 3): hook identity, offsets, 0rec = stored picks. No teacher.
    python belief_noise.py run SIGMAS WORKERS [--limit N]    e.g. run 0,5,10,20 28  -> decisions_val_3d.jsonl + teacher verify_3d.jsonl
    python belief_noise.py report               -> report_val_3d.json
Env BN_SMOKE=1: outputs under .../belief_noise/smoke/ and no full-VAL asserts.  Outputs: approachb_3d_learning/runs/explore/belief_noise/.
"""
import collections, hashlib, json, math, os, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SH = HERE.parents[2]                                             # shelf_6x6_open_sides/
SMOKE = bool(os.environ.get("BN_SMOKE"))
OUT = (SH / "approachb_3d_learning/runs/explore/belief_noise" / ("smoke" if SMOKE else ".")).resolve()
ENV_OUT = dict(V3OUT=str(OUT / "teach_out"), FIN3D_STORE=str(OUT / "teach_out/fin3d_store"), ABF_P5DIR=str(OUT / "p5"))
os.environ.update(ENV_OUT)
sys.path.insert(0, str(SH / "hybrid/src"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import hz_eval as H  # noqa: E402  (twod_hazard env + V3.4 code path; teacher env defaults keep our V3OUT / FIN3D_STORE)
os.environ.update(ENV_OUT)                                       # twod_hazard pointed ABF_P5DIR at the frozen runs/hz
HZ = H.HZ
sys.path.insert(0, str(HZ.CODE / "ablations_k3"))
import exhaustive as EX, placement as PL, search as S  # noqa: E402
import run as RUN  # noqa: E402  (FULL_s0 loader, rule, pool_hash)

SALT = "belief_noise_v1|"
CLAMP = 0.037                                                    # metres per axis
DEC, VER = OUT / "decisions_val_3d.jsonl", OUT / "verify_3d.jsonl"
LABEL = "EXPLORATORY; split: 3D VAL (seed 0 models); TEST never read"

# ------------------------------------------------------------------------------------------- noise hook
_N = {}                                                          # object id -> (start cell, dx, dy) for the root being decided
_sf0 = PL.scene_features


def scene_features(dom, objs, cells, query, legal_cells):
    F, M, occ, leg = _sf0(dom, objs, cells, query, legal_cells)
    if _N:
        assert dom == "3d"                                       # 3D base is (0, 0): dist = |(x, y)|, angle = atan2(y, x)
        for i, o in enumerate([query] + sorted(o for o in cells if o != query)):
            c0, dx, dy = _N[o]
            if int(cells[o]) == c0:
                x, y = float(F[i, 6]) + dx, float(F[i, 7]) + dy
                F[i, 6:10] = [x, y, math.hypot(x, y), math.atan2(y, x) / math.pi]
    return F, M, occ, leg


PL.scene_features = scene_features


def offsets(rid, ids, sigma_mm):
    """-> {id: (dx, dy)} metres, clamped per axis; and the number of clamped axes."""
    out, nc = {}, 0
    for o in ids:
        g = np.random.default_rng(int(hashlib.sha256(f"{SALT}{rid}|{o}".encode()).hexdigest()[:16], 16)).standard_normal(2) * sigma_mm / 1000
        nc += int((np.abs(g) > CLAMP).sum())
        out[o] = tuple(float(v) for v in np.clip(g, -CLAMP, CLAMP))
    return out, nc


def noisy_row(r, off):
    inp = dict(r["inputs"])
    inp["xy0"] = [[x + off[o][0], y + off[o][1]] for (x, y), o in zip(inp["xy0"], inp["ids"])]
    return {**r, "inputs": inp}


# ------------------------------------------------------------------------------------------- decisions
def models():
    hz = HZ.HazardScorer(HZ.OUT / "model_3d_s0.pt")
    _, samp, xy, _, man, _ = RUN.load("FULL_s0", "3d", "cpu")
    return hz, samp, xy, man


def decide(r, level, M, ref):
    """one (root, level) -> picks of both arms.  level '0rec' | sigma in mm as str."""
    hz, samp, xy, man = M
    sigma = 0.0 if level == "0rec" else float(level)
    off, nc = offsets(r["root_id"], r["inputs"]["ids"], sigma)
    _N.clear()
    if sigma:
        _N.update({o: (int(c), *off[o]) for o, c in zip(r["inputs"]["ids"], r["inputs"]["cells0"])})
    try:
        root = S.Root(noisy_row(r, off) if sigma else r, "3d")
        if level == "0rec":
            D = {int(o): [int(d) for d in v] for o, v in ref[r["root_id"]]["proposals"].items()}
            assert RUN.pool_hash(D, 4) == ref[r["root_id"]]["pool_hash"], r["root_id"]
        else:
            D = {o: v[:4] for o, v in EX.proposals(root, samp, man["config"]["proposal_seed"]).items()}
        pick = lambda sc, rule: (lambda d: None if d is None else d["seq"])(EX.score_root(root, D, sc, [rule], 4, set(), 8192)["per_M"]["4"]["decisions"][0])  # noqa: E731
        return {"root_id": r["root_id"], "level": level, "sigma_mm": sigma, "clamped_axes": nc, "pool_hash": RUN.pool_hash(D, 4),
                "ours": pick(hz, HZ.FROZEN), "xynet": pick(xy, RUN.RULE)}
    finally:
        _N.clear()


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if Path(p).exists() else []


def stored():
    """existing VAL picks: ours frozen rule (pool_3d_val_s0 decisions[0]), XYNet FULL_s0 (decision + moves); FULL_s0 rows (pools)."""
    ours = {r["root_id"]: r["decisions"][0] for p in sorted(HZ.OUT.glob("pool_3d_val_s0*.jsonl")) for r in jl(p)}
    ref = {r["root_id"]: r for r in jl(HZ.K3 / "runs/predictions/FULL_s0/val_3d.jsonl")}
    xy = {k: None if r["decision"] == "ABSTAIN" else [[m["object_index"], m["destination_cell"]] for m in r["moves"]] for k, r in ref.items()}
    return ours, xy, ref


def decisions(levels, limit=None):
    rows = jl(HZ.K3 / "runs/public/public_val_3d.jsonl")[:limit]
    ours0, _, ref = stored()
    done = {(r["root_id"], r["level"]) for r in jl(DEC)}
    todo = [(r, lv) for lv in levels for r in rows if (r["root_id"], lv) not in done]
    M = models() if todo else None
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(DEC, "a") as f:
        for i, (r, lv) in enumerate(todo, 1):
            d = decide(r, lv, M, ref)
            if lv == "0rec":   # reproduction: the recorded pool + frozen rule must give the stored VAL pick
                assert d["ours"] == ours0[r["root_id"]], f"{r['root_id']}: ours 0rec {d['ours']} != stored {ours0[r['root_id']]}"
            f.write(json.dumps(d) + "\n"); f.flush()
            if i % 50 == 0 or i == len(todo):
                print(f"{time.strftime('%H:%M:%S')} decisions {i}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)


# ------------------------------------------------------------------------------------------- teacher
def labels():
    """their cache + runs/hz (hz_eval.labels, read only) + ours; a decided label is never replaced by an undecided one."""
    L = H.labels("3d")
    for k, v in H.V.labels("3d", str(VER)).items():
        if L.get(k) is None or (None in L[k] and None not in v):
            L[k] = v
    return L


def teach(workers):
    priv, L = H.priv_of("3d", "val"), labels()
    tried = {(v["root"], json.dumps(x["seq"])) for v in jl(VER) for x in v["results"]}
    want = collections.defaultdict(set)
    for d in jl(DEC):
        for a in ("ours", "xynet"):
            if d[a]:
                want[d["root_id"]].add(json.dumps(d[a]))
    jobs = []
    for rid, qs in sorted(want.items()):
        e = priv[rid]
        todo = [json.loads(q) for q in sorted(qs) if None in L.get((rid, q), (None, None)) and (rid, q) not in tried]
        if todo:
            jobs.append({"root_id": rid, "dom": "3D", "path": H.local(e["path"]), "geom": e.get("geom", "original"), "ids": e["ids"],
                         "seqs": [[[e["ids"][o], c] for o, c in q] for q in todo]})
    print(json.dumps({"picks": sum(map(len, want.values())), "teacher_queries": sum(len(j["seqs"]) for j in jobs), "roots": len(jobs)}), flush=True)
    if jobs:
        H._run("3d", jobs, workers, VER)


def run(sigmas, workers, limit=None):
    lv = [str(int(s)) if float(s).is_integer() else s for s in sigmas.split(",")]
    decisions((["0rec"] if "0" in lv else []) + lv, limit)
    teach(workers)
    report()


# ------------------------------------------------------------------------------------------- report
def report():
    priv, L = H.priv_of("3d", "val"), labels()
    D = collections.defaultdict(dict)
    for d in jl(DEC):
        D[d["level"]][d["root_id"]] = d
    levels = sorted(D, key=lambda k: (-1, 0) if k == "0rec" else (float(k), 1))
    common = sorted(set.intersection(*(set(v) for v in D.values())))
    ours0, xy0, _ = stored()
    out = {"label": LABEL, "n_roots": len(common), "smoke": SMOKE, "salt": SALT, "clamp_m_per_axis": CLAMP,
           "notes": ["Rates = hz_eval.rates (benchmark.correct masks) with the teacher run on the TRUE scene; UNKNOWN = no decided label.",
                     "0rec = recorded GPU cVAE pool; 0 = CPU cVAE, no noise (reference for noise rows). CPU and GPU cVAE draws differ only "
                     "on near-tied spots on a few roots.", "Noise = sigma * N(0, I2) per object, clamped to +-37 mm per axis (stays in its cell)."],
           "levels": {}}
    for lv in levels:
        e = {"clamped_axes": sum(D[lv][r]["clamped_axes"] for r in common)}
        for a in ("ours", "xynet"):
            picks = {r: D[lv][r][a] for r in common}
            rows = [{"masks": priv[r]["masks"], "outcome": H.V.outcome(picks[r], L, r)} for r in common]
            e[a] = {**H.rates(rows), "decisions": dict(collections.Counter("abstain" if p is None else "noop" if p == [] else "task" for p in picks.values())),
                    "unknown_repair": sum(x["outcome"] == "UNKNOWN" for x in rows),
                    "pick_changed_vs_0": sum(picks[r] != D["0"][r][a] for r in common) if "0" in D else None}
        e["pool_differs_from_0"] = sum(D[lv][r]["pool_hash"] != D["0"][r]["pool_hash"] for r in common) if "0" in D else None
        out["levels"][lv] = e
    if "0rec" in D:   # reproduction self-check vs the stored VAL picks / numbers
        out["repro"] = {"ours_0rec_equals_stored_picks": sum(D["0rec"][r]["ours"] == ours0[r] for r in common),
                        "xynet_0rec_equals_FULL_s0_picks": sum(D["0rec"][r]["xynet"] == xy0[r] for r in common), "of": len(common)}
        assert out["repro"]["ours_0rec_equals_stored_picks"] == len(common), out["repro"]
        full = {x["root_id"]: x for x in H._full_rows("3d", "val", 0)}
        out["repro"]["FULL_s0_stored_rates_same_roots"] = H.rates([full[r] for r in common])
        if not SMOKE and len(common) == len(priv):
            m = json.loads((HZ.OUT / "metrics_3d_val_s0.json").read_text())["hazard_by_rule"]
            want = next(v for k, v in m.items() if k.endswith("FROZEN"))
            got = out["levels"]["0rec"]["ours"]
            assert all(got[k] == want[k] for k in ("task", "noop", "abstain", "n_task")), (got, want)
            out["repro"]["ours_0rec_equals_metrics_3d_val_s0"] = True
    (OUT / "report_val_3d.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({lv: {a: {k: v[a][k] for k in ("task", "noop", "abstain", "unknown_repair", "pick_changed_vs_0")} for a in ("ours", "xynet")}
                      for lv, v in out["levels"].items()} | {"repro": out.get("repro")}, indent=1))


def check(n=3):
    """self-checks (no teacher): hook = identity at 0 noise; it shifts exactly the unmoved objects; 0rec reproduces stored picks."""
    rows = [r for r in jl(HZ.K3 / "runs/public/public_val_3d.jsonl") if len(r["inputs"]["ids"]) >= 3][:n]
    ours0, xy0, ref = stored()
    for p in [str(OUT)] + list(ENV_OUT.values()):
        assert not str(Path(p).resolve()).startswith(str(HZ.OUT.resolve())), f"would write into the frozen runs/hz: {p}"
    r = rows[0]
    root = S.Root(r, "3d")
    cd = {root.ids[i]: c for i, c in enumerate(root.cells0)}
    q = root.ids[root.movable[0]]
    F0 = _sf0("3d", root.pattr, cd, q, root.legal)[0]
    assert np.array_equal(scene_features("3d", root.pattr, cd, q, root.legal)[0], F0)                    # no noise -> identical
    ids = [q] + sorted(o for o in cd if o != q)
    assert all(np.allclose(F0[i, 6:8], PL.cell_xy("3d", cd[o])) for i, o in enumerate(ids))              # cols 6, 7 = x, y
    assert all(np.allclose(r["inputs"]["xy0"][j], PL.cell_xy("3d", c)) for j, c in enumerate(root.cells0))  # xy0 = start cell centres
    off, _ = offsets(r["root_id"], root.ids, 10)
    _N.update({o: (int(c), *off[o]) for o, c in cd.items()})
    moved = dict(cd); moved[q] = root.legal_dests(root.cells0)[0]
    F1 = scene_features("3d", root.pattr, moved, q, root.legal)[0]
    _N.clear()
    F1r = _sf0("3d", root.pattr, moved, q, root.legal)[0]
    assert np.allclose(F1[0], F1r[0])                                                                    # moved query: unchanged
    for i, o in list(enumerate(ids))[1:]:
        assert np.allclose(F1[i, 6:8] - F1r[i, 6:8], off[o], atol=1e-6) and np.array_equal(F1[i, :6], F1r[i, :6])
    assert offsets("a", ["x"], 5)[0] == offsets("a", ["x"], 5)[0] != offsets("a", ["y"], 5)[0]
    assert max(abs(v) for d in offsets("a", [str(i) for i in range(400)], 1000)[0].values() for v in d) == CLAMP
    M = models()
    res = []
    for r in rows:
        a, b = decide(r, "0rec", M, ref), decide(r, "10", M, ref)
        assert a["ours"] == ours0[r["root_id"]], (r["root_id"], a["ours"], ours0[r["root_id"]])
        res.append({"root": r["root_id"], "ours_0rec": a["ours"], "xynet_0rec_eq_FULL": a["xynet"] == xy0[r["root_id"]], "ours_10mm": b["ours"],
                    "xynet_10mm": b["xynet"]})
    assert all(x["xynet_0rec_eq_FULL"] for x in res), "XYNet 0rec does not reproduce the stored FULL_s0 picks"
    print(json.dumps({"label": LABEL, "ok": True, "roots": res}, indent=1))


if __name__ == "__main__":
    torch.set_num_threads(1)
    c, a = sys.argv[1], sys.argv[2:]
    if c == "check":
        check(int(a[0]) if a else 3)
    elif c == "run":
        run(a[0], int(a[1]), int(a[a.index("--limit") + 1]) if "--limit" in a else None)
    elif c == "report":
        report()
    else:
        raise SystemExit(__doc__)
