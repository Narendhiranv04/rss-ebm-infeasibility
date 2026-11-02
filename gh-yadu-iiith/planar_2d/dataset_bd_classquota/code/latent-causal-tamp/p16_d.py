"""P16 D: pairless Boltzmann feasibility + blocker-membership model and its adapter to the common non-learned search.

Learned scope only: P(y0 = 1 | S, T, g, q) = sigmoid(u) and, given y0 = 0, P(b | ...) = exp(sum_i a_i b_i) / prod_i (1 + exp(a_i)) (pairless:
independent Bernoullis, no inter-blocker coupling claimed); the paired variant (frozen P13 D) adds 1/2 sum w_ij b_i b_j with the exact
normalizer. Loss = BCE(y0) + lambda_B * 1[y0 = 0, mask defined] * (-log P(b)); the mask is the UNION of minimal VERIFIED clearing
subsets within the declared pool lattice; NO_CLEARING_SET_WITHIN_POOL, INCOMPLETE, NON_MONOTONE and UNKNOWN originals get NO blocker
term (fixed vs P13, which fitted proof-grounded NO_CLEARING_SET records as all-zero masks). Recursion uses only the DEV-recalibrated
temporary-target FEASIBILITY; temporary-target blocker marginals are validated separately (temp_lattice) and never used unvalidated."""
from __future__ import annotations

import copy
import itertools
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("P10C_OUT", "runs/p13/models")
import config as cf  # noqa: E402
import d_blockers as db  # noqa: E402

OUT = Path("runs/p16/d")
STATUS = {1: db.V, 0: db.C}


def lattice(lab, pool):
    """d_blockers.minimal_sets on the scene-only lattice (edit y: 1 VERIFIED, 0 CERTIFIED, None missing)."""
    st = {frozenset(): STATUS.get(lab["original"]["y"])}
    for kind, v in lab["edits"].items():
        st[frozenset(pool.index(o) for o in kind.split("+"))] = STATUS.get(v["y"])
    return db.minimal_sets(len(pool), {k: v for k, v in st.items() if v is not None})


def item(R, d):
    """(R, pool, y0, mask | None) with the P16 semantics: blocker NLL only for MINIMAL_SETS with a defined mask (any provenance is
    recorded; training uses proof-grounded and operational alike only if the caller allows); feasible originals keep y0 = 1, empty pool."""
    y0 = d.get("y0")
    if y0 is None:
        return None
    ok = y0 == 0 and d.get("type") == "MINIMAL_SETS" and d.get("mask") is not None and d.get("provenance") == "proof_grounded"
    return (R, d["pool"] if ok else [], y0, d["mask"] if ok else None)


def p13_items():
    """Frozen P13 trainval D items re-labelled with the P16 rule (NO_CLEARING_SET masked)."""
    import p13_pipeline as pp
    rs = pp._prepared("trainval")
    out = {"train": [], "val": []}
    for r in rs:
        d = dict(r.get("d") or {})
        if d.get("y0") is None:
            y0 = next((c["y"] for c in r["cands"] if not c["seq"]), None)
            d = {"y0": y0, "pool": []} if y0 is not None else d
        it = item(r["R"], d)
        if it:
            out["train" if r["split"] == "train" else "val"].append(it)
    return out


def p16_items(roots, labs):
    out = []
    for r in roots:
        lab, pool_ids = labs[r["key"]], labs[r["key"]]["catalogue"]["pool"]
        d = lattice(lab, pool_ids)
        d["pool"] = [r["ids"].index(o) for o in pool_ids]
        it = item(r["R"], d)
        if it:
            out.append(it)
    return out


def items_from(data="runs/p16/b/TRAIN.pt"):
    """D items from the shared training file: P13 records (stored d labels, NO_CLEARING_SET masked) + P16 records (scene-only lattice)."""
    import p16_b as pb
    d = torch.load(data, weights_only=False)
    labs = {k: v for L in d["p16_labels"].values() for k, v in L.items()}
    out = {}
    for part in ("train", "val"):
        rs = pb.prepare(d[part])
        its = []
        for r, rec in zip(rs, d[part]):
            if r["key"] in labs:
                its += p16_items([r], labs)
            else:
                dd = dict(rec.get("d") or {})
                if dd.get("y0") is None:
                    y0 = next((c["y"] for c in r["cands"] if not c["seq"]), None)
                    dd = {"y0": y0, "pool": []} if y0 is not None else dd
                it = item(r["R"], dd)
                if it:
                    its.append(it)
        out[part] = its
    return out


def train(seed, lam=2.0, pair=False, data="runs/p16/b/TRAIN.pt", updates=600) -> None:
    """Same optimizer / schedule as P13 (p10c_train_eval.train_d) on P13 + P16 items with the corrected label rule."""
    d = items_from(data)
    tr, va = d["train"], d["val"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = db.ModelD(pair=pair).double()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, hist, c0 = (math.inf, None, -1), [], time.process_time()
    vl = lambda: float(np.mean([float(db.nll(*m(R, pool), y0, b, 1.0)) for R, pool, y0, b in va]))  # noqa: E731
    for u in range(updates):
        batch = [tr[i] for i in rng.choice(len(tr), 12, replace=False)]
        opt.zero_grad()
        loss = torch.stack([db.nll(*m(R, pool), y0, b, lam) for R, pool, y0, b in batch]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        if (u + 1) % 20 == 0:
            with torch.no_grad():
                v = vl()
            hist.append({"update": u + 1, "train": float(loss), "val": v})
            if v < best[0]:
                best = (v, copy.deepcopy(m.state_dict()), u + 1)
    m.load_state_dict(best[1])
    name = f"P16-{'D' if pair else 'Dp'}-l{lam}-s{seed}"
    ck = OUT / "ckpt" / f"{name}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ck)
    import p16_eval as pe
    cf.write_json(OUT / "runs" / f"{name}.json", {"name": name, "pair": pair, "seed": seed, "lambda_B": lam, "val_nll": best[0], "best_update": best[2],
                                                  "history": hist, "train_items": len(tr), "train_masks": sum(b is not None for *_, b in tr),
                                                  "val_items": len(va), "params": sum(p.numel() for p in m.parameters()),
                                                  "train_cpu_s": round(time.process_time() - c0, 1), "checkpoint": pe.sha(ck), "data": pe.sha(data)})


def models(kind):
    """'D' / 'Dp' frozen P13 (lambda 1.0 / 2.0, 3 seeds) or 'Dp16' retrained P16 pairless (3 seeds)."""
    if kind in ("D", "Dp"):
        import p14_d_search as ds
        return ds.d_models(kind == "Dp")
    out = []
    for s in range(3):
        m = db.ModelD(pair=False).double()
        m.load_state_dict(torch.load(OUT / "ckpt" / f"P16-Dp-l2.0-s{s}.pt", weights_only=True))
        out.append(m.eval())
    return out


@torch.no_grad()
def diagnose(ms, roots, labs) -> dict:
    """Original-target diagnosis on decided roots: feasibility AUROC / Brier / recalls; blocker F1 / exact (feasibility-gated: predicted
    mask = (1 - p) * marg >= 0.5 on infeasible originals) against the union mask, and agreement with the necessary intersection."""
    rows = []
    for r in roots:
        lab, pool_ids = labs[r["key"]], labs[r["key"]]["catalogue"]["pool"]
        d = lattice(lab, pool_ids)
        if d["y0"] is None:
            continue
        pool = [r["ids"].index(o) for o in pool_ids]
        pr = [db.predict(m, r["R"], pool) for m in ms]
        p = float(np.mean([x["p_feasible"] for x in pr]))
        marg = np.mean([x["marg"] for x in pr], 0) if pool else np.zeros(0)
        rows.append({"key": r["key"], "family": lab["family"], "y0": d["y0"], "p": p, "type": d["type"], "mask": d["mask"], "necessary": d["necessary"],
                     "pred": [int((1 - p) * m_ >= 0.5) for m_ in marg], "marg": [float(x) for x in marg]})
    ys, ps = np.array([x["y0"] for x in rows]), np.array([x["p"] for x in rows])
    pos, neg = ps[ys == 1], ps[ys == 0]
    auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in pos for b in neg])) if len(pos) and len(neg) else None
    dec = [x for x in rows if x["y0"] == 0 and x["type"] == "MINIMAL_SETS"]
    tp = sum(a and b for x in dec for a, b in zip(x["pred"], x["mask"]))
    fp = sum(a and not b for x in dec for a, b in zip(x["pred"], x["mask"]))
    fn = sum(b and not a for x in dec for a, b in zip(x["pred"], x["mask"]))
    return {"n": len(rows), "feasible": int(ys.sum()), "auroc": auc, "brier": float(np.mean((ps - ys) ** 2)) if len(rows) else None,
            "feasible_recall": float((pos >= 0.5).mean()) if len(pos) else None, "infeasible_recall": float((neg < 0.5).mean()) if len(neg) else None,
            "masks": len(dec), "blocker_f1": 2 * tp / max(1, 2 * tp + fp + fn), "exact": float(np.mean([x["pred"] == x["mask"] for x in dec])) if dec else None,
            "union_ne_necessary": sum(x["mask"] != x["necessary"] for x in dec), "rows": rows}


def temp_lattice(task, phys, cat) -> list:
    """Temporary-target validation lattice: for each pool object o (temporary target, destination parking[o]) and every subset U of the
    other pool objects parked, the status of the move o -> parking[o] from q0 in the edited scene (VERIFIED / CERTIFIED / UNKNOWN). The
    union of minimal clearing subsets is the temporary-target blocker label (UNKNOWN -> incomplete, no label)."""
    cells0, q0 = {o.id: o.cell for o in phys.s0.objects}, np.asarray(phys.s0.robot.q_rad, float)
    out = []
    for o in cat["pool"]:
        others = [x for x in cat["pool"] if x != o]
        st = {}
        for n in range(len(others) + 1):
            for U in itertools.combinations(range(len(others)), n):
                cells = {**cells0, **{others[i]: cat["parking"][others[i]] for i in U}}
                res = phys.edges(cells, q0, o, cat["parking"][o])
                s = next((x[0] for x in res if x[0] == "VERIFIED"), res[0][0] if res else None)   # physics vocabulary
                st[frozenset(U)] = db.V if s == "VERIFIED" else db.C if s in ("CERTIFIED_INFEASIBLE", "STATIC_INVALID") else None
        lab = db.minimal_sets(len(others), {k: v for k, v in st.items() if v is not None})
        out.append({"target": o, "dest": cat["parking"][o], "others": others, "statuses": {"+".join(others[i] for i in sorted(k)) or "-": v for k, v in st.items()}, **lab})
    return out


# ------------------------------------------------------------------ guidance for the common search (reorder only; never prunes)
class Guide:
    """kinds: D | Dp | Dp16 (frozen P13 paired / frozen P13 pairless / P16-retrained pairless), each optionally '-norec'; geometry,
    geometry-norec (analytic heuristic on updated positions, geometric recursion), uniform (catalogue order), oracle (union of the
    ORIGINAL scene-only lattice from the unsealed label; scope-limited, non-deployable). Recursion = defer an object whose
    DEV-recalibrated temporary-target FEASIBILITY to its first candidate slot is < 0.5 (D) or that has heuristic blockers among the other
    movers (geometry); temporary-target blocker MARGINALS are never used. Sort key: (deferred, -score, original index)."""

    def __init__(self, kind, task, phys, movers, oracle=None, dino=None):
        import abc_models as am
        import p15_retarget as pr
        base, _, flag = kind.partition("-")
        self.base, self.recursion = base, flag != "norec" and base not in ("uniform", "oracle")
        self.phys, self.movers, self.oracle = phys, movers, oracle or {}
        self.models = models(base) if base in ("D", "Dp", "Dp16") else []
        self.recal = json_load("runs/p15/retarget/RECALIBRATION.json")["fit_all_dev"]
        if dino is not None:
            pr._DINO = dino
        self.rec0 = pr.build_record(phys.cfg, phys.s0, "t0", phys.goal_cell)
        self.R0, self.ids = am.Root(self.rec0), self.rec0["audit"]["ids"]
        self.cache, self.d_calls, self.d_cpu, self.trace = {}, 0, 0.0, []

    def _d(self, cells, q, target, goal, pool):
        import p15_retarget as pr
        k = (tuple(sorted(cells.items())), tuple(np.round(q, 12)), target, goal, tuple(pool))
        if k not in self.cache:
            t0 = time.process_time()
            s = pr.retarget_scene(self.phys.s0, cells, target, q)
            self.cache[k] = pr.d_predict(self.models, pr.build_record(self.phys.cfg, s, target, goal), target, list(pool) + ["t0"])
            self.d_calls += 1
            self.d_cpu += time.process_time() - t0
        return self.cache[k]

    def _geo(self, cells, q, target, pool):
        import torch
        import p14_d_search as ds
        xy = torch.as_tensor(np.array([self.phys.s0.lattice.center(cells[o]) for o in self.ids]), dtype=torch.float64)
        R2 = ds.root_variant(self.R0, self.phys.robot, xy, self.ids.index(target), q)
        return dict(zip(pool, db.heuristic(R2, [self.ids.index(o) for o in pool])))

    def order(self, cells, q, moved, cands):
        if self.base == "uniform" or not cands:
            return cands
        pool = [o for o in self.movers if o in {c[0] for c in cands}]
        if self.base == "oracle":
            sc = {o: float(self.oracle.get(o, 0)) for o in pool}
        elif self.base == "geometry":
            sc = {o: float(v) for o, v in self._geo(cells, q, "t0", pool).items()}
        else:
            p = self._d(cells, q, "t0", self.phys.goal_cell, pool)
            sc = {o: (1 - p["p_feasible"]) * m for o, m in zip(p["pool"], p["marg"])}
        defer = set()
        if self.recursion:
            first = {}
            for o, d in cands:
                first.setdefault(o, d)
            for o, d in first.items():
                others = [x for x in pool if x != o]
                if self.base == "geometry":
                    if any(self._geo(cells, q, o, others).values()):
                        defer.add(o)
                else:
                    p = self._d(cells, q, o, d, others)["p_feasible"]
                    lg = np.log(max(p, 1e-9) / max(1 - p, 1e-9))
                    if 1 / (1 + np.exp(-(lg - self.recal["bias"]) / self.recal["temperature"])) < 0.5:
                        defer.add(o)
        out = sorted(cands, key=lambda c: (c[0] in defer, -sc.get(c[0], 0.0), cands.index(c)))
        if len(self.trace) < 40:
            self.trace.append({"cells": sorted(cells.items()), "scores": {o: round(v, 4) for o, v in sc.items()}, "deferred": sorted(defer),
                               "first": out[0] if out else None})
        return out


def json_load(p):
    import json
    return json.loads(Path(p).read_text())


def guide(kind, task, phys, movers, root=None, dino=None, oracle=None):
    return Guide(kind, task, phys, movers, oracle, dino)
