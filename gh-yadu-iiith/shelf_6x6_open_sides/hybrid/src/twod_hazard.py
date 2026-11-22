"""Per-step hazard scorer plugged into Approach B (2D and 3D).

p(plan works) = prod_k P(relocation k ok | scene after moves < k) * P(connector to q_ref ok | endpoint) * P(target reaches goal | endpoint).
No-op: P(target ok | unedited scene).  Only the scorer differs from Approach B; the data are B's own snapshot (TRAIN rows; VAL for early stop).

Two modes (env HZ_MODE):
  v2   "ours alone": B V2 pipeline (gvlab copy runs/ABF2D, snapshot abf_v2_*, V2 cVAE sampler, capped best-first search.search), 2D only.
  v34  "ours + B's improvements" (default): scores V3.4's IDENTICAL candidate pools (proposals stored in the sealed FULL_s0 predictions,
       pool hash checked) with exhaustive.score_root (every legal K1-K3 ordered sequence, no cap); trained on the expanded TRAIN
       snapshot abf_v32_fb0eef4f5ff433ee (the snapshot every V3.4 ablation scorer arm used).  Rules: the frozen V3.4 rule first, then a
       VAL-tuning grid over (noop_gate, eta), all decided in one scoring pass.

Labels: Y rows -> target step.  X rows: x = 1 -> every relocation + connector step is 1; x = 0 with a known failed phase ('relocation:k',
0-based, or 'connector') -> earlier steps 1, failed step 0, later dropped; x = 0 with no phase -> set likelihood -log(1 - prod_k p_k).
A step key seen with both 0 and 1 is dropped.  TEST labels are never read here.

    HZ_MODE=v34 python twod_hazard.py train DOM SEED
    HZ_MODE=v34 python twod_hazard.py pool DOM SPLIT SEED [i/k]     -> runs/ABF3/runs/hz/pool_{dom}_{split}_s{seed}[.part].jsonl
    HZ_MODE=v2  python twod_hazard.py train 2d SEED ; HZ_MODE=v2 python twod_hazard.py search SEED SPLIT
    python twod_hazard.py selftest DOM
"""
import collections, hashlib, json, math, os, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RUNS = HERE.parents[1] / "approachb_3d_learning/runs"
MODE = os.environ.get("HZ_MODE", "v34")
if MODE == "v2":
    ABF = RUNS / "ABF2D"
    _snaps = sorted((ABF / "snapshot").glob("abf_v2_*"))
    assert len(_snaps) == 1, _snaps
    SNAP, CODE = _snaps[0], ABF / "code"
else:
    ABF, K3 = RUNS / "ABF3", RUNS / "K3ABL"
    SNAP, CODE = ABF / "snapshot/abf_v32_fb0eef4f5ff433ee", ABF / "code_v34"
    os.environ.update(K3=str(K3), K3_BUNDLE=str(K3 / "stage/bundle"))
OUT = ABF / "runs/hz"
os.environ.update(ABF_ROOT=str(ABF), ABF_SNAPSHOT=str(SNAP), ABF_P5DIR=str(OUT), XY_DEV="cpu")
sys.path.insert(0, str(CODE))
import placement as PL, search as S  # noqa: E402

H = 64
KINDS = ("reloc", "conn", "target")
FROZEN = {"noop_gate": 0.5, "eta": 0.3, "alpha": 0.0, "beta": 0.0, "gamma": 0.0}
RULES = [FROZEN] + [{**FROZEN, "noop_gate": g, "eta": e} for g in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8) for e in (0.05, 0.1, 0.2, 0.3, 0.4, 0.5)]


def steps_of(root, seq):
    """seq [(slot, cell)] -> [(key, cells_before, cells_after, query slot, dest cell, kind, k)]."""
    out, cells = [], list(root.cells0)
    for k, (i, d) in enumerate(seq):
        after = list(cells); after[i] = int(d)
        out.append(((tuple(map(tuple, seq[:k + 1])), "reloc"), cells, after, i, int(d), "reloc", k))
        cells = after
    if seq:
        out.append(((tuple(map(tuple, seq)), "conn"), cells, cells, root.tid, cells[root.tid], "conn", len(seq)))
    tgt = list(cells); tgt[root.tid] = root.goal
    out.append(((tuple(sorted(map(tuple, seq))), "target"), cells, tgt, root.tid, root.goal, "target", len(seq)))
    return out


def featurize(root, steps):
    """-> F [n,2,MAXOBJ,14], M [n,2,MAXOBJ], O [n,2,R*C], X [n,8]; scene features memoised per (cells, query) within the call."""
    R, C = PL.GRID[root.dom]
    memo, Fs, Ms, Os, Xs = {}, [], [], [], []
    for _, before, after, q, d, kind, k in steps:
        for c in (before, after):
            key = (tuple(c), q)
            if key not in memo:
                memo[key] = PL.scene_features(root.dom, root.pattr, {root.ids[j]: v for j, v in enumerate(c)}, root.ids[q], root.legal)[:3]
            f, m, o = memo[key]
            Fs.append(f); Ms.append(m); Os.append(o)
        Xs.append([d // C / (R - 1), d % C / (C - 1), root.goal // C / (R - 1), root.goal % C / (C - 1), *(float(kind == x) for x in KINDS), k / 3])
    n = len(steps)
    return (torch.from_numpy(np.stack(Fs)).view(n, 2, PL.MAXOBJ[root.dom], -1), torch.from_numpy(np.stack(Ms)).view(n, 2, -1),
            torch.from_numpy(np.stack(Os)).view(n, 2, -1), torch.tensor(Xs, dtype=torch.float32))


class Net(nn.Module):
    def __init__(self, mu, sd, ncell):
        super().__init__()
        self.enc, self.head = PL.SceneEncoder(ncell, 14, H), PL.mlp(2 * H + 8, H, 1)
        self.register_buffer("mu", torch.as_tensor(mu)); self.register_buffer("sd", torch.as_tensor(sd))

    def forward(self, Fe, M, O, X):
        n = Fe.shape[0]
        Fe = (Fe - self.mu) / self.sd * M[..., None]
        z = self.enc(Fe.flatten(0, 1), M.flatten(0, 1), O.flatten(0, 1)).view(n, -1)
        return self.head(torch.cat([z, X], 1)).squeeze(-1)


# ------------------------------------------------------------------------------------------------------------------ labels
def phases():
    """V2 only: on-policy X results carry the teacher's failed phase; the V2 snapshot keeps only x -> join back by (root, seq)."""
    out = {}
    for f in (ABF / "runs/v2/2d" / f"labels_{k}.jsonl" for k in ("coverage", "collect")):
        if MODE != "v2" or not f.exists():
            continue
        for line in open(f):
            r = json.loads(line)
            for x in r["results"]:
                if x.get("x") and x["x"].get("x") == 0:
                    out[(r["root"], json.dumps(x["seq"]))] = x["x"].get("phase")
    return out


def fail_step(phase, m):
    """'relocation:k' (0-based) / 'connector...' -> index of the failed step in steps_of order (m relocations, then connector) | None."""
    if not phase:
        return None
    p = str(phase).lower()
    if "connector" in p:
        return m
    digits = "".join(ch if ch.isdigit() else " " for ch in p).split()
    return int(digits[-1]) if "reloc" in p and digits and int(digits[-1]) < m else None


def examples(rows, ph, dom):
    """-> (root, step, y) examples, set-negatives (root, steps), counters.  A step key seen with both 0 and 1 is dropped (counted)."""
    ex, neg, unk = [], [], collections.Counter()
    for r in rows:
        root = S.Root(r, dom)
        votes, step = collections.defaultdict(set), {}

        def vote(s, y):
            step[s[0]] = s; votes[s[0]].add(int(y))
        for y in r["labels"]["Y"]:
            vote(steps_of(root, [tuple(e) for e in y["edit"]])[-1], y["y"])
        for q in r["labels"]["X"]:
            seq = [tuple(a) for a in q["seq"]]
            st = [s for s in steps_of(root, seq) if s[5] != "target"]
            if q["x"] == 1:
                for s in st:
                    vote(s, 1)
                continue
            phase, src = (q["x_phase"], "snapshot") if q.get("x_phase") else (ph.get((r["root_id"], json.dumps(q["seq"]))), "onpolicy")
            j = fail_step(phase, len(seq))
            if j is None:
                unk[f"x0_no_phase:{phase}"[:60]] += 1
                neg.append((root, st))
                continue
            unk[f"x0_phase_from_{src}"] += 1
            for s in st[:j]:
                vote(s, 1)
            vote(st[j], 0)
        for k, v in votes.items():
            if len(v) == 1:
                ex.append((root, step[k], v.pop()))
            else:
                unk["conflicting_step_dropped"] += 1
    return ex, neg, unk


def tensors(items):
    by = collections.defaultdict(list)
    for root, s, y in items:
        by[id(root)].append((root, s, y))
    parts = [(featurize(g[0][0], [s for _, s, _ in g]), [y for *_, y in g]) for g in by.values()]
    return (*(torch.cat([p[0][i] for p in parts]) for i in range(4)), torch.tensor(sum((p[1] for p in parts), []), dtype=torch.float32))


def neg_feats(negs):
    return [featurize(root, st) for root, st in negs]


def set_nll(net, nf):
    """-mean log(1 - prod p) over plans that failed at an unknown step; nf from neg_feats."""
    if not nf:
        return torch.zeros(())
    lp = F.logsigmoid(net(*map(torch.cat, zip(*nf))))
    idx = torch.repeat_interleave(torch.arange(len(nf)), torch.tensor([len(t[0]) for t in nf]))
    s = torch.zeros(len(nf)).index_add(0, idx, lp).clamp(max=-1e-6)
    return -torch.log(-torch.expm1(s)).mean()


def load_rows(dom, split):
    assert split in ("train", "val")                      # TEST labels are never read by this script
    return [r for r in map(json.loads, open(SNAP / f"data{dom}.jsonl")) if r["split"] == split]


def train(dom, seed, epochs=60, patience=8, bs=512):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    torch.set_num_threads(int(os.environ.get("HZ_THREADS", "2")))  # keep the laptop responsive
    ph = phases()
    tr, trn, unk = examples(load_rows(dom, "train"), ph, dom)
    va, van, _ = examples(load_rows(dom, "val"), ph, dom)
    Ft, Mt, Ot, Xt, yt = tensors(tr); Fv, Mv, Ov, Xv, yv = tensors(va)
    trn, van = neg_feats(trn), neg_feats(van)
    flat = Ft[Mt].numpy()
    net = Net(flat.mean(0), flat.std(0) + 1e-6, Ot.shape[-1])
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    OUT.mkdir(parents=True, exist_ok=True)
    best, bad, curve, t0 = math.inf, 0, [], time.time()
    nb = math.ceil(len(yt) / bs)
    for ep in range(epochs):
        net.train()
        perm, nperm = torch.from_numpy(rng.permutation(len(yt))), rng.permutation(len(trn))
        for b in range(nb):
            i = perm[b * bs:(b + 1) * bs]
            l = F.binary_cross_entropy_with_logits(net(Ft[i], Mt[i], Ot[i], Xt[i]), yt[i])
            negs = [trn[j] for j in nperm[b::nb]]
            l = l + set_nll(net, negs) * len(negs) / max(1, len(i))
            opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(F.binary_cross_entropy_with_logits(net(Fv, Mv, Ov, Xv), yv, reduction="sum") + set_nll(net, van) * len(van)) / (len(yv) + len(van))
        curve.append({"epoch": ep, "val_nll": round(v, 5), "s": round(time.time() - t0)})
        print(json.dumps(curve[-1]), flush=True)
        if v < best - 1e-5:
            best, bad = v, 0
            torch.save({"state": net.state_dict(), "mu": net.mu, "sd": net.sd, "ncell": Ot.shape[-1]}, OUT / f"model_{dom}_s{seed}.pt")
        elif (bad := bad + 1) >= patience:
            break
    meta = {"mode": MODE, "dom": dom, "seed": seed, "best_val_nll": best, "curve": curve, "n_train_steps": len(yt), "train_pos_rate": float(yt.mean()),
            "n_train_set_neg": len(trn), "n_val_steps": len(yv), "counters": dict(unk), "snapshot": SNAP.name}
    (OUT / f"train_{dom}_s{seed}.json").write_text(json.dumps(meta, indent=1))


# ------------------------------------------------------------------------------------------------------------------ scorer
class HazardScorer:
    """Drop-in for search.Scorer: score(root, seqs) -> [{p11, J, py, px}].  Step probabilities are cached per root (plans share prefixes)."""

    def __init__(self, path, chunk=4096):
        ck = torch.load(path, weights_only=False)
        self.net = Net(ck["mu"], ck["sd"], ck["ncell"]); self.net.load_state_dict(ck["state"]); self.net.eval()
        self.evals, self.root, self.cache, self.chunk = 0, None, {}, chunk

    @torch.no_grad()
    def score(self, root, seqs):
        if root is not self.root:                         # cache is valid for one Root object only
            self.root, self.cache = root, {}
        st = [steps_of(root, [tuple(a) for a in s]) for s in seqs]
        new = list({x[0]: x for s in st for x in s if x[0] not in self.cache}.values())
        for i in range(0, len(new), self.chunk):
            part = new[i:i + self.chunk]
            p = torch.sigmoid(self.net(*featurize(root, part))).numpy()
            self.cache.update(zip((x[0] for x in part), p.tolist()))
        self.evals += sum(1 for s in seqs if s)
        return [{"p11": float(np.prod([self.cache[x[0]] for x in s])), "J": 0.0, "py": float("nan"), "px": float("nan")} for s in st]


def pool_hash(D, M=4):
    """= ablations_k3/run.py pool_hash."""
    return hashlib.sha256(json.dumps({str(o): v[:M] for o, v in sorted(D.items())}, sort_keys=True).encode()).hexdigest()[:20]


def pool_runs(dom, split, seed, part=None):
    """v34: score V3.4's identical per-root pools (from the sealed FULL_s0 predictions) and decide under every rule in RULES."""
    import exhaustive as EX
    torch.set_num_threads(1)
    rows = [json.loads(l) for l in open(K3 / "runs/public" / f"public_{split}_{dom}.jsonl")]
    ref = {r["root_id"]: r for r in map(json.loads, open(K3 / "runs/predictions/FULL_s0" / f"{split}_{dom}.jsonl"))}
    assert {r["root_id"] for r in rows} <= set(ref), "FULL_s0 predictions do not cover every root"
    if part:
        i_, k_ = map(int, part.split("/"))
        rows = [r for r in rows if int(hashlib.sha256(r["root_id"].encode()).hexdigest(), 16) % k_ == i_]
    OUT.mkdir(parents=True, exist_ok=True)
    fo = OUT / (f"pool_{dom}_{split}_s{seed}" + (f".part{part.replace('/', 'of')}" if part else "") + ".jsonl")
    done = {json.loads(l)["root_id"] for l in open(fo)} if fo.exists() else set()
    scor = HazardScorer(OUT / f"model_{dom}_s{seed}.pt")
    t0 = time.time()
    with open(fo, "a") as f:
        for n, r in enumerate(x for x in rows if x["root_id"] not in done):
            t1 = time.time()
            root = S.Root(r, dom)
            D = {int(o): [int(d) for d in v[:4]] for o, v in ref[r["root_id"]]["proposals"].items()}
            assert pool_hash(D) == ref[r["root_id"]]["pool_hash"], r["root_id"]
            rec = EX.score_root(root, D, scor, RULES, 4, set(), 8192)
            dec = rec["per_M"]["4"]["decisions"]
            f.write(json.dumps({"root_id": r["root_id"], "p_noop": rec["p_noop"], "n_scored": rec["per_M"]["4"]["n_scored"],
                                "decisions": [None if d is None else d["seq"] for d in dec],
                                "p11": [None if d is None else d["p11"] for d in dec], "s": round(time.time() - t1, 2)}) + "\n")
            f.flush()
            if n % 25 == 0:
                print(f"{dom} {split} s{seed} {part or ''}: {n + 1} roots, {time.time() - t0:.0f}s", flush=True)


def search_runs(seed, split):
    """v2 ('ours alone'): B V2's best-first search with its V2 cVAE sampler, hazard scorer in place of the X/Y model."""
    import evaluate as E
    assert MODE == "v2" and split in ("valsub", "val", "test")
    out = E.p5dir("2D") / "search" / f"hazard_s{seed}_{split}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = {r["root"] for r in E.jl(out)}
    samp = S.Sampler("2d", "cvae", seed, ABF / "data/q2_2d", device="cpu")
    scor = HazardScorer(OUT / f"model_2d_s{seed}.pt")
    with open(out, "a") as f:
        for r in (x for x in E.rows_of("2D", split) if x["root_id"] not in done):
            res = S.search(S.Root(r, "2d"), samp, scor, {**E.SEARCH, "seed": seed}, mode="iterative")
            res.update(arm="hazard", seed=seed, sampler="cvae", K=r["K_legacy"])
            f.write(json.dumps(res) + "\n"); f.flush()


def selftest(dom):
    allr = load_rows(dom, "train")
    r = next(x for x in allr if x["K_legacy"] == "2" and x["labels"]["X"])
    rows = allr[:60] + [r]
    root = S.Root(r, dom)
    seq = [tuple(a) for a in r["labels"]["X"][0]["seq"]]
    st = steps_of(root, seq)
    assert [s[5] for s in st] == ["reloc"] * len(seq) + ["conn", "target"], st
    assert st[1][1] == st[0][2] and st[-1][2][root.tid] == root.goal
    assert fail_step("relocation:1", 3) == 1 and fail_step("connector", 2) == 2 and fail_step(None, 2) is None and fail_step("relocation:5", 2) is None
    Fe, M, O, X = featurize(root, st)
    R, C = PL.GRID[dom]
    assert Fe.shape[:2] == (len(st), 2) and O.shape == (len(st), 2, R * C) and X.shape == (len(st), 8)
    ex, neg, unk = examples(rows, {}, dom)
    net = Net(np.zeros(14, np.float32), np.ones(14, np.float32), R * C)
    assert net(Fe, M, O, X).shape == (len(st),)
    assert torch.isfinite(set_nll(net, neg_feats(neg[:3]))) if neg else True
    sc = HazardScorer.__new__(HazardScorer); sc.net, sc.evals, sc.root, sc.cache, sc.chunk = net.eval(), 0, None, {}, 4096
    a = sc.score(root, [[], seq]); b = sc.score(root, [seq])
    assert 0 < a[0]["p11"] < 1 and abs(a[1]["p11"] - b[0]["p11"]) < 1e-9 and len(sc.cache) == len(st) + 1  # + the no-op target step
    print(json.dumps({"ok": True, "mode": MODE, "dom": dom, "examples": len(ex), "set_neg": len(neg), "counters": dict(unk)}))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "train":
        train(a[0], int(a[1]))
    elif c == "pool":
        pool_runs(a[0], a[1], int(a[2]), a[3] if len(a) > 3 else None)
    elif c == "search":
        search_runs(int(a[0]), a[1])
    elif c == "selftest":
        selftest(a[0])
    else:
        raise SystemExit(__doc__)
