"""EXPLORATORY (VAL only) "PIGINet-style" JOINT whole-plan scorer (after Yang et al., RSS 2023): ONE success probability per plan,
not the hazard model's product of per-step probabilities.  Everything else is the encoders.py flow: same per-step featurisation
(twod_hazard.steps_of / featurize, base PL.SceneEncoder via HZ.Net), same V3.4 VAL pools, same frozen rule, same teacher.

Model: each step of the plan (relocations, connector, target pick; the no-op plan is the target step alone, exactly the steps HazardScorer
multiplies) -> HZ.Net's encoder on (scene before, scene after) + the 8 step features -> linear to H -> + learned step-position embedding ->
2-layer, 4-head transformer (padding masked) -> masked mean -> MLP -> 1 logit = P(plan succeeds).

Labels (same TRAIN / VAL rows as twod_hazard.train): per X record, plan = its ordered seq; label 1 if x == 1 and Y(endpoint) == 1, 0 if
x == 0, or x == 1 and Y(endpoint) == 0; skipped if x == 1 and Y unknown.  Endpoint mapping: a Y record's `edit` is the UNORDERED
assignment; it matches an X record iff contracts.ekey(edit) == contracts.ekey(seq) (sorted [slot, cell] pairs; the same mapping
exhaustive.known_success uses).  The no-op plan is labelled by the Y record with edit == [] (x trivially 1).
An ordered plan seen with both labels is dropped (counted).  TEST labels are never read.

    python piginet.py selftest [DOM]
    python piginet.py train DOM SEED          (venv python)
    python piginet.py pool DOM SEED i/k       (venv python)
    python piginet.py eval DOM SEED W         (env3d python for 3d) -> runs/explore/enc/piginet/metrics_{dom}_s{seed}.json
env PIG_SMOKE=E: smoke mode, writes to runs/explore/enc/piginet/smoke/ and trains at most E epochs (eval is not supported there).
"""
import collections, json, math, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import encoders as E  # noqa: E402  (imports twod_hazard read-only)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402
from torch.nn import functional as F  # noqa: E402

HZ, PL, Hd = E.HZ, E.PL, E.Hd
S = HZ.S
import contracts as CT  # noqa: E402  (frozen code_v34, on sys.path via twod_hazard)

KIND, MAXLEN = "piginet", 5                    # <= 3 relocations + connector + target
SMOKE = int(os.environ.get("PIG_SMOKE", "0"))


class PlanNet(HZ.Net):
    def __init__(self, mu, sd, ncell):
        super().__init__(mu, sd, ncell)        # keeps the base scene encoder + mu/sd buffers
        del self.head
        self.proj, self.pos = nn.Linear(2 * Hd + 8, Hd), nn.Embedding(MAXLEN, Hd)
        layer = nn.TransformerEncoderLayer(Hd, 4, 2 * Hd, dropout=0.0, batch_first=True)
        self.tf = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)
        self.out = PL.mlp(Hd, Hd, 1)

    def steps(self, Fe, M, O, X):
        """featurize() tensors of n steps -> step embeddings [n, H] (same normalisation and encoder call as HZ.Net.forward)."""
        n = Fe.shape[0]
        Fe = (Fe - self.mu) / self.sd * M[..., None]
        z = self.enc(Fe.flatten(0, 1), M.flatten(0, 1), O.flatten(0, 1)).view(n, -1)
        return self.proj(torch.cat([z, X], 1))

    def plans(self, h, idx):
        """h [n, H] step embeddings, idx [B, L] step indices in plan order, -1 = padding -> logits [B]."""
        m = idx >= 0
        s = (h[idx.clamp(min=0)] + self.pos.weight[:idx.shape[1]]) * m[..., None]
        e = self.tf(s, src_key_padding_mask=~m) * m[..., None]
        return self.out(e.sum(1) / m.sum(1, keepdim=True)).squeeze(-1)

    def forward(self, Fe, M, O, X, idx):
        return self.plans(self.steps(Fe, M, O, X), idx)


# ------------------------------------------------------------------------------------------------------------------ labels / data
def plan_labels(rows, dom):
    """-> [(root, seq, y)], counters.  See the module docstring for the rule."""
    out, cnt = [], collections.Counter()
    for r in rows:
        root = S.Root(r, dom)
        Y, lab = collections.defaultdict(set), collections.defaultdict(set)
        for y in r["labels"]["Y"]:
            Y[CT.ekey(y["edit"])].add(int(y["y"]))
        if len(Y.get(CT.ekey([]), ())) == 1:
            lab[()].add(next(iter(Y[CT.ekey([])])))
        for q in r["labels"]["X"]:
            seq, yv = tuple(tuple(map(int, a)) for a in q["seq"]), Y.get(CT.ekey(q["seq"]), set())
            if q["x"] == 0:
                lab[seq].add(0); cnt["neg_x0"] += 1
            elif len(yv) == 1:
                lab[seq].add(yv.copy().pop()); cnt["pos_x1_y1" if 1 in yv else "neg_x1_y0"] += 1
            else:
                cnt["skip_x1_y_unknown"] += 1
        for seq, v in lab.items():
            if len(v) == 1:
                out.append((root, list(seq), next(iter(v))))
                if not seq:
                    cnt[f"noop_y{next(iter(v))}"] += 1
            else:
                cnt["conflicting_plan_dropped"] += 1
    cnt["plans"], cnt["plans_pos"] = len(out), sum(y for *_, y in out)
    return out, cnt


def tensors(items):
    """-> step tensors (F, M, O, X) of the unique steps of every root, plan index [N, MAXLEN] (-1 pad), labels [N]."""
    by = collections.defaultdict(list)
    for root, seq, y in items:
        by[id(root)].append((root, seq, y))
    parts, idx, ys, base = [], [], [], 0
    for g in by.values():
        root, keys = g[0][0], {}
        for _, seq, y in g:
            st = HZ.steps_of(root, [tuple(a) for a in seq])
            ii = [keys.setdefault(s[0], (len(keys), s))[0] + base for s in st]
            idx.append(ii + [-1] * (MAXLEN - len(ii))); ys.append(y)
        parts.append(HZ.featurize(root, [s for _, s in sorted(keys.values(), key=lambda t: t[0])]))
        base += len(keys)
    return (*(torch.cat([p[i] for p in parts]) for i in range(4)), torch.tensor(idx), torch.tensor(ys, dtype=torch.float32))


def batch_logits(net, T, sel):
    """logits of the plans with global step indices sel [B, L]; encodes only the steps those plans use."""
    m = sel >= 0
    u, inv = torch.unique(sel[m], return_inverse=True)
    loc = torch.full_like(sel, -1); loc[m] = inv
    return net(T[0][u], T[1][u], T[2][u], T[3][u], loc)


def train(dom, seed, epochs=60, patience=8, bs=512):
    """Mirrors HZ.train: same rows, seed handling, AdamW(1e-3, wd 1e-4), epochs / patience, batch size (in plans here)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    torch.set_num_threads(int(os.environ.get("HZ_THREADS", "2")))
    epochs = min(epochs, SMOKE) if SMOKE else epochs
    t0 = time.time()
    tr, ctr = plan_labels(HZ.load_rows(dom, "train"), dom)
    va, cva = plan_labels(HZ.load_rows(dom, "val"), dom)
    Tt, Tv = tensors(tr), tensors(va)
    print(json.dumps({"train_counts": dict(ctr), "val_counts": dict(cva), "steps_train": len(Tt[0]), "build_s": round(time.time() - t0)}), flush=True)
    flat = Tt[0][Tt[1]].numpy()
    net = PlanNet(flat.mean(0), flat.std(0) + 1e-6, Tt[2].shape[-1])
    opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    yt, yv = Tt[5], Tv[5]
    best, bad, curve, t0 = math.inf, 0, [], time.time()
    nb = math.ceil(len(yt) / bs)
    for ep in range(epochs):
        net.train()
        perm = torch.from_numpy(rng.permutation(len(yt)))
        for b in range(nb):
            i = perm[b * bs:(b + 1) * bs]
            l = F.binary_cross_entropy_with_logits(batch_logits(net, Tt, Tt[4][i]), yt[i])
            opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(F.binary_cross_entropy_with_logits(net(*Tv[:5]), yv))
        curve.append({"epoch": ep, "val_nll": round(v, 5), "s": round(time.time() - t0)})
        print(json.dumps(curve[-1]), flush=True)
        if v < best - 1e-5:
            best, bad = v, 0
            torch.save({"state": net.state_dict(), "mu": net.mu, "sd": net.sd, "ncell": Tt[2].shape[-1]}, HZ.OUT / f"model_{dom}_s{seed}.pt")
        elif (bad := bad + 1) >= patience:
            break
    meta = {"kind": KIND, "mode": HZ.MODE, "dom": dom, "seed": seed, "best_val_nll": best, "val_nll_unit": "per plan (not per step as in hazard)",
            "curve": curve, "n_train_plans": len(yt), "train_pos_rate": float(yt.mean()), "n_val_plans": len(yv), "n_train_unique_steps": len(Tt[0]),
            "counts_train": dict(ctr), "counts_val": dict(cva), "snapshot": HZ.SNAP.name, "smoke_epochs": SMOKE or None}
    (HZ.OUT / f"train_{dom}_s{seed}.json").write_text(json.dumps(meta, indent=1))


# ------------------------------------------------------------------------------------------------------------------ scorer
class PlanScorer:
    """Drop-in for HZ.HazardScorer: score(root, seqs) -> [{p11, J, py, px}], p11 = P(whole plan).  Step embeddings cached per root;
    the no-op plan [] is the target step alone (score_root uses it for the no-op gate and as the no-op candidate, as with the hazard)."""

    def __init__(self, path, chunk=4096):
        ck = torch.load(path, weights_only=False)
        self.net = PlanNet(ck["mu"], ck["sd"], ck["ncell"]); self.net.load_state_dict(ck["state"]); self.net.eval()
        self.evals, self.root, self.chunk = 0, None, chunk

    @torch.no_grad()
    def score(self, root, seqs):
        if root is not self.root:                     # cache is valid for one Root object only
            self.root, self.key, self.h = root, {}, torch.zeros(0, Hd)
        st = [HZ.steps_of(root, [tuple(a) for a in s]) for s in seqs]
        new = list({x[0]: x for s in st for x in s if x[0] not in self.key}.values())
        for i in range(0, len(new), self.chunk):
            part = new[i:i + self.chunk]
            self.key.update({x[0]: len(self.h) + j for j, x in enumerate(part)})
            self.h = torch.cat([self.h, self.net.steps(*HZ.featurize(root, part))])
        idx = torch.tensor([[self.key[x[0]] for x in s] + [-1] * (MAXLEN - len(s)) for s in st])
        p = torch.cat([torch.sigmoid(self.net.plans(self.h, idx[i:i + self.chunk])) for i in range(0, len(idx), self.chunk)]).tolist()
        self.evals += sum(1 for s in seqs if s)
        return [{"p11": float(x), "J": 0.0, "py": float("nan"), "px": float("nan")} for x in p]


def bind(dom):
    """encoders.bind (OUT -> runs/explore/enc/piginet) + swap the scorer class pool_runs instantiates, in THIS process only."""
    E.bind(dom, KIND)
    HZ.Net, HZ.HazardScorer = PlanNet, PlanScorer
    if SMOKE:
        HZ.OUT = HZ.OUT / "smoke"
        HZ.OUT.mkdir(parents=True, exist_ok=True)
    assert HZ.OUT.is_relative_to(E.EXP / KIND) and not HZ.OUT.is_relative_to(HZ.ABF / "runs/hz"), HZ.OUT


def selftest(dom="3d"):
    torch.manual_seed(0)
    R, C = PL.GRID[dom]
    net = PlanNet(np.zeros(14, np.float32), np.ones(14, np.float32), R * C).eval()
    # shapes + padding: the same plan with extra padding columns, or batched next to other plans, scores the same
    h = torch.randn(6, Hd)
    a = net.plans(h, torch.tensor([[1, 2, -1]]))
    b = net.plans(h, torch.tensor([[1, 2, -1, -1, -1], [0, 3, 4, 5, -1]]))
    assert a.shape == (1,) and b.shape == (2,) and torch.allclose(a, b[:1], atol=1e-5), "padding leak"
    h2 = h.clone(); h2[0] = 99.0                       # step 0 is what padding clamps to; must not reach a plan that does not use it
    assert torch.allclose(net.plans(h2, torch.tensor([[1, 2, -1]])), a, atol=1e-5), "padding leak via clamp"
    # order sensitivity on identical step embeddings (position embedding alone)
    assert (net.plans(h, torch.tensor([[1, 2, 3]])) - net.plans(h, torch.tensor([[2, 1, 3]]))).abs() > 1e-6, "order-blind"
    # label construction on real TRAIN rows: one x=1&y=1, one x=1&y=0, one x=0, checked by a direct scan (no ekey)
    rows = HZ.load_rows(dom, "train")
    want = {"11": None, "10": None, "0": None}
    for r in rows:
        for q in r["labels"]["X"]:
            ys = [y["y"] for y in r["labels"]["Y"] if sorted(map(list, y["edit"])) == sorted(map(list, q["seq"]))]
            k = "0" if q["x"] == 0 else f"1{ys[0]}" if len(ys) == 1 else None
            if k in want and want[k] is None:
                want[k] = (r, q, int(k == "11"))
        if all(want.values()):
            break
    for k, (r, q, y) in want.items():
        got = {tuple(map(tuple, s)): v for _, s, v in plan_labels([r], dom)[0]}
        assert got[tuple(map(tuple, q["seq"]))] == y, (k, r["root_id"], q["seq"])
        noop = [y_["y"] for y_ in r["labels"]["Y"] if not y_["edit"]]
        assert len(noop) != 1 or got[()] == noop[0], "noop label"
    # featurisation + full forward on a real plan, scorer 0<p<1, noop = target step alone, order sensitivity on real steps
    r, q, _ = want["11"]
    root, seq = S.Root(r, dom), [tuple(a) for a in q["seq"]]
    T = tensors([(root, seq, 1), (root, [], 0)])
    assert T[0].shape[:2] == (len(seq) + 3, 2) and T[4].shape == (2, MAXLEN) and int((T[4][1] >= 0).sum()) == 1, T[4]
    assert net(*T[:5]).shape == (2,)
    sc = PlanScorer.__new__(PlanScorer); sc.net, sc.evals, sc.root, sc.chunk = net, 0, None, 4096
    out = sc.score(root, [[], seq, seq[::-1]])
    assert all(0 < o["p11"] < 1 for o in out), out
    assert abs(sc.score(root, [seq])[0]["p11"] - out[1]["p11"]) < 1e-6 and sc.evals == 3
    assert len(seq) < 2 or abs(out[1]["p11"] - out[2]["p11"]) > 1e-7, "order-blind on a real plan"
    print(json.dumps({"ok": True, "dom": dom, "params": sum(p.numel() for p in net.parameters()), "label_rows": {k: v[0]["root_id"] for k, v in want.items()},
                      "p_noop_seq_rev": [round(o["p11"], 4) for o in out]}))


if __name__ == "__main__":
    c, a = sys.argv[1], sys.argv[2:]
    if c == "selftest":
        selftest(*a[:1])
    elif c == "train":
        bind(a[0]); train(a[0], int(a[1]))
    elif c == "pool":
        bind(a[0]); HZ.pool_runs(a[0], "val", int(a[1]), a[2])
    elif c == "eval":
        assert not SMOKE, "eval needs every VAL root pooled"
        E.evaluate(a[0], KIND, int(a[1]), int(a[2]))
    else:
        raise SystemExit(__doc__)
