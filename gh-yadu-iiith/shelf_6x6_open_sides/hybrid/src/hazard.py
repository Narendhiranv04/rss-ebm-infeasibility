"""Per-step hazard success model: p(plan works) = prod_k P(move k ok | scene after moves < k) * P(t0 -> goal ok | scene after all moves).

Labels are the executed B sequences of each root (no new physics): move k is 1 if the chain got past it, 0 if it failed there, moves after
the failure are dropped; the target step (only reached when every move worked) is y.  Deduplicated by (root, steps up to and incl. this one).
Fit on TRAIN roots, early stopping / metrics on VAL roots; TEST roots are never read.

Step input: the shared SceneEncoder (models.py) applied to the scene BEFORE the step and the scene AFTER it (query object at its destination),
both with the query object in slot 0, plus [dest r/5, c/5, goal r/5, c/5, is_target, step/3] -> MLP -> logit.  Plain BCE, no pos_weight:
the product over steps needs calibrated per-step probabilities, and re-weighting would bias them.

    python ../hybrid/src/hazard.py train       (3 seeds -> runs/HAZARD/s{seed}.pt, metrics.json)
    python ../hybrid/src/hazard.py eval        (plan-level VAL AUROC vs phase3 p11, __call__ timing -> metrics.json)
    python ../hybrid/src/hazard.py selftest
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # shelf_6x6_open_sides/
A = ROOT / "approachb_3d_learning"
sys.path[:0] = [str(A / "src"), str(HERE)]
import data3d as D, models as Mo, xy3d_data as XD  # noqa: E402

OUT = A / "runs/HAZARD"
H, SEEDS = 64, (0, 1, 2)
_cfg = json.loads((A / "configs/sampler_v1.json").read_text())
_sc = json.loads((A / _cfg["derived"] / "scalers.json").read_text())
MU, SD = np.array(_sc["mean"], np.float32), np.array(_sc["std"], np.float32)


def plan_steps(seq, goal):
    """moves then the target step -> [(prefix key incl. this step, moved-before dict, obj, dest, k, is_target)]"""
    seq = [(o, tuple(c)) for o, c in seq] + [("t0", tuple(goal))]
    return [(tuple(seq[:k + 1]), dict(seq[:k]), o, d, k, k == len(seq) - 1) for k, (o, d) in enumerate(seq)]


def featurize(raw, steps):
    """steps from plan_steps -> tensors F [n,2,MAXOBJ,14], M [n,2,MAXOBJ] bool, O [n,2,36], X [n,6]"""
    objs, cells0, goal = D.allowed(raw)["objects"], {k: tuple(v) for k, v in raw["initial_cells"].items()}, raw["goal"]
    F, M, O, X = [], [], [], []
    for _, moved, o, d, k, is_t in steps:
        before = {**cells0, **moved}
        for c in (before, {**before, o: d}):
            f, m, occ, _ = D.scene_features(objs, c, o)
            F.append((f - MU) / SD * m[:, None]); M.append(m); O.append(occ)
        X.append([d[0] / 5, d[1] / 5, goal[0] / 5, goal[1] / 5, float(is_t), k / 3])
    n = len(steps)
    return (torch.from_numpy(np.stack(F)).view(n, 2, D.MAXOBJ, -1), torch.from_numpy(np.stack(M)).view(n, 2, -1),
            torch.from_numpy(np.stack(O)).view(n, 2, -1), torch.tensor(X, dtype=torch.float32))


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc, self.head = Mo.SceneEncoder(14, H), Mo.mlp(2 * H + 6, H, 1)

    def forward(self, F, M, O, X):
        e = self.enc(F.flatten(0, 1), M.flatten(0, 1), O.flatten(0, 1)).view(len(F), -1)
        return self.head(torch.cat([e, X], -1))[:, 0]


class Hazard:
    def __init__(self):
        self.nets = []
        for s in SEEDS:
            n = Net(); n.load_state_dict(torch.load(OUT / f"s{s}.pt", weights_only=False)["state"]); self.nets.append(n.eval())

    def _probs(self, raw, seqs):
        """-> unique-step probabilities P [3, U+1] (last column = 1, the padding slot) and per-plan step indices idx [n, maxlen]"""
        uniq, idx = {}, []
        for s in seqs:
            assert len(s) >= 1, "a plan needs at least one move"
            idx.append([uniq.setdefault(st[0], (len(uniq), st))[0] for st in plan_steps(s, raw["goal"])])
        batch = featurize(raw, [st for _, st in uniq.values()])
        with torch.no_grad():
            P = torch.stack([n(*batch).sigmoid() for n in self.nets])
        P = torch.cat([P, torch.ones(len(P), 1)], 1)
        L = max(map(len, idx))
        return P, torch.tensor([i + [len(uniq)] * (L - len(i)) for i in idx])

    def step_probs(self, raw, seqs):
        """-> list (one per plan) of [3, len(plan)+1] per-step success probabilities, last column = target step"""
        P, idx = self._probs(raw, seqs)
        return [P[:, i[:len(s) + 1]] for s, i in zip(seqs, idx)]

    def __call__(self, raw, seqs):
        """-> [3, len(seqs)] per-seed plan success probability (product over steps)"""
        P, idx = self._probs(raw, seqs)
        return P[:, idx].prod(-1)


# ----------------------------------------------------------------------------------------------------------------- data / training
def stype(k, is_t):
    return "target" if is_t else ("move1", "move2")[k] if k < 2 else "move3+"


def examples(split):
    assert split in ("train", "val")  # never the test split
    steps, y, kind, conflicts = {}, {}, {}, 0
    roots = [r for r, s in XD.qc_roots().items() if s == split]
    for rid in roots:
        raw = XD.load(rid)
        for b in raw["labels"]["B"]:
            f, seq = b["failed_step"], b["sequence"]
            assert (f is None) == (b["y"] == 1) and len({o for o, _ in seq}) == len(seq) and "t0" not in {o for o, _ in seq}
            n_ok = len(seq) if f in (None, "target") else f  # number of moves that worked
            for st in plan_steps(seq, raw["goal"])[:n_ok + 1]:
                lab = int(b["y"]) if st[5] else int(st[4] < n_ok)
                key = (rid, st[0])
                if key in y:
                    conflicts += y[key] != lab; continue
                steps.setdefault(rid, []).append(st); y[key] = lab; kind[key] = stype(st[4], st[5])
    feats = [featurize(XD.load(rid), sts) for rid, sts in steps.items()]
    keys = [(rid, st[0]) for rid, sts in steps.items() for st in sts]
    batch = tuple(torch.cat(t) for t in zip(*feats))
    return batch, torch.tensor([y[k] for k in keys], dtype=torch.float32), np.array([kind[k] for k in keys]), conflicts, len(roots)


def metrics(p, y, kind):
    from sklearn.metrics import log_loss, roc_auc_score
    out = {}
    for t in ("move1", "move2", "move3+", "target", "all"):
        m = np.ones(len(y), bool) if t == "all" else kind == t
        yy, pp = y[m], np.clip(p[m], 1e-6, 1 - 1e-6)
        out[t] = {"n": int(m.sum()), "pos": int(yy.sum()), "logloss": round(float(log_loss(yy, pp, labels=[0, 1])), 4),
                  "auroc": round(float(roc_auc_score(yy, pp)), 4) if 0 < yy.sum() < len(yy) else None,
                  "mean_p": round(float(pp.mean()), 4)}
    return out


def train_seed(seed, tr, va, max_epochs=80, patience=10, budget_s=90):
    torch.manual_seed(seed)
    net, bce = Net(), nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    (Xtr, ytr), (Xva, yva) = tr, va
    best, bad, t0, log = (1e9, None, -1), 0, time.time(), []
    for ep in range(max_epochs):
        net.train()
        for i in torch.randperm(len(ytr)).split(256):
            opt.zero_grad(); bce(net(*(x[i] for x in Xtr)), ytr[i]).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            v = float(bce(net(*Xva), yva))
        log.append(round(v, 4))
        if v < best[0]:
            best, bad = (v, {k: t.clone() for k, t in net.state_dict().items()}, ep), 0
        else:
            bad += 1
        if bad >= patience or time.time() - t0 > budget_s:
            break
    net.load_state_dict(best[1])
    torch.save({"state": best[1], "seed": seed, "epoch": best[2], "val_logloss": best[0], "H": H}, OUT / f"s{seed}.pt")
    return net.eval(), {"best_epoch": best[2], "epochs_run": len(log), "val_bce_curve": log, "train_s": round(time.time() - t0, 1)}


def cmd_train():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    Xtr, ytr, ktr, ctr, ntr = examples("train")
    Xva, yva, kva, cva, nva = examples("val")
    counts = {s: {t: {"pos": int(y[k == t].sum()), "neg": int((k == t).sum() - y[k == t].sum())} for t in ("move1", "move2", "move3+", "target")}
              for s, y, k in (("train", ytr, ktr), ("val", yva, kva))}
    out = {"counts": counts, "roots": {"train": ntr, "val": nva}, "duplicate_label_conflicts": {"train": ctr, "val": cva},
           "featurize_s": round(time.time() - t0, 1), "loss": "BCE, no pos_weight (calibration for the product)", "H": H, "seeds": {}}
    print(json.dumps(out), flush=True)
    ps = []
    for s in SEEDS:
        net, info = train_seed(s, (Xtr, ytr), (Xva, yva))
        with torch.no_grad():
            p = net(*Xva).sigmoid().numpy()
        ps.append(p); info["val"] = metrics(p, yva.numpy(), kva); out["seeds"][s] = info
        print(json.dumps({"seed": s, **{k: v for k, v in info.items() if k != "val_bce_curve"}}), flush=True)
    out["val_seed_mean"] = metrics(np.mean(ps, 0), yva.numpy(), kva)
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["val_seed_mean"], indent=1))


def cmd_eval():
    from sklearn.metrics import roc_auc_score
    import phase3 as P3
    Hz, Mm = Hazard(), P3.Models()
    ys, ph, pc, skipped = [], [], [], 0
    for rid, s in XD.qc_roots().items():
        if s != "val":
            continue
        raw, seen = XD.load(rid), {}
        for b in raw["labels"]["B"]:
            seq = tuple((o, tuple(c)) for o, c in b["sequence"])
            if not seq or len({o for o, _ in seq}) < len(seq):
                skipped += 1; continue  # empty plan (no p11) or repeated object
            seen.setdefault(seq, b["y"])
        if seen:
            seqs = list(seen)
            ys += list(seen.values()); ph += Hz(raw, seqs).mean(0).tolist(); pc += Mm.p11(raw, seqs).mean(0).tolist()
    ys = np.array(ys)
    plan = {"n": len(ys), "pos": int(ys.sum()), "skipped_empty_or_invalid": skipped,
            "auroc_hazard": round(float(roc_auc_score(ys, ph)), 4), "auroc_p11": round(float(roc_auc_score(ys, pc)), 4),
            "mean_p_hazard": round(float(np.mean(ph)), 4), "mean_p11": round(float(np.mean(pc)), 4)}
    # timing: 5,000 random 1-3 move plans of distinct non-target objects to free cells != goal on one VAL root, beam-search-like
    raw = XD.load(next(r for r, s in XD.qc_roots().items() if s == "val" and len(XD.load(r)["initial_cells"]) >= 4))
    plans = random_plans(raw, 5000)
    Hz(raw, plans[:10])
    t = time.time(); Hz(raw, plans); dt = time.time() - t
    out = json.loads((OUT / "metrics.json").read_text())
    out["plan_level_val"] = plan
    out["timing"] = {"plans": len(plans), "call_s": round(dt, 3), "threads": torch.get_num_threads(), "root": raw["ids"]["root_id"]}
    (OUT / "metrics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({"plan_level_val": plan, "timing": out["timing"]}, indent=1))


def random_plans(raw, n, seed=0):
    rng, goal = np.random.default_rng(seed), tuple(raw["goal"])
    cells0 = {k: tuple(v) for k, v in raw["initial_cells"].items()}
    objs, plans = sorted(o for o in cells0 if o != "t0"), []
    for _ in range(n):
        cells, plan = dict(cells0), []
        for o in rng.permutation(objs)[:rng.integers(1, min(3, len(objs)) + 1)]:
            free = [(r, c) for r in range(6) for c in range(6) if (r, c) not in set(cells.values()) and (r, c) != goal]
            d = free[rng.integers(len(free))]; plan.append((str(o), d)); cells[str(o)] = d
        plans.append(plan)
    return plans


def cmd_selftest():
    Hz = Hazard()
    raw = XD.load(next(r for r, s in XD.qc_roots().items() if s == "val" and len(XD.load(r)["initial_cells"]) >= 4))
    plans = random_plans(raw, 200, seed=1)
    p, sp = Hz(raw, plans), Hz.step_probs(raw, plans)
    assert p.shape == (3, 200) and bool(((p > 0) & (p <= 1)).all())
    assert all(s.shape == (3, len(q) + 1) for s, q in zip(sp, plans))
    assert torch.allclose(torch.stack([s.prod(1) for s in sp], 1), p)
    assert torch.allclose(Hz(raw, plans[7:8])[:, 0], p[:, 7], atol=1e-5)  # batching / prefix dedupe does not change scores
    # a plan's first step equals the one-move plan's first step (shared prefix)
    assert torch.allclose(Hz.step_probs(raw, [plans[0][:1]])[0][:, 0], sp[0][:, 0], atol=1e-5)
    print("selftest ok")


if __name__ == "__main__":
    torch.set_num_threads(1)
    {"train": cmd_train, "eval": cmd_eval, "selftest": cmd_selftest}[sys.argv[1]]()
