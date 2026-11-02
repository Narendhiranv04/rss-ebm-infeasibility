"""P17 D: pairless feasibility + scoped blocker-union model; per-model temporary-target calibration; label protocols; search guides.

Label protocols (never silently mixed):
  PARKING_LATTICE_V1     P16: every pool subset parked at its single scene-only parking slot, target queried at q0 (p16_d.lattice).
  MULTI_DEST_LATTICE_V1  P17: subset S is SUFFICIENT if SOME admissible assignment of S's objects to their first k declared domain slots
                         (distinct cells, original occupancy) makes the ORIGINAL target VERIFIED at q0; INSUFFICIENT only if EVERY
                         assignment is certified negative at q0; otherwise UNKNOWN. Says nothing about executability of the moves.
Blocker union = union of minimal sufficient subsets (d_blockers.minimal_sets); NO_CLEARING_SET / INCOMPLETE / NON_MONOTONE / UNKNOWN get
no blocker term. Recursion: each model family has its OWN temporary-target feasibility calibration (logistic T, b fitted on root-disjoint
development probes); P16 reused one P15 calibration for every family (defect). Guides only reorder candidates."""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

import d_blockers as db

OUT = Path("runs/p17/d")
STATUS = {1: db.V, 0: db.C}


def multi_dest_lattice(phys, pool, slots, k=2) -> dict:
    """{subset (sorted tuple of pool ids): VERIFIED / CERTIFIED / None} with per-assignment provenance."""
    import p15_search as ps
    cells0, q0 = {o.id: o.cell for o in phys.s0.objects}, np.asarray(phys.s0.robot.q_rad, float)
    opts = {o: slots(cells0, o)[:k] for o in pool}
    out, prov = {}, {}
    for n in range(len(pool) + 1):
        for S in itertools.combinations(pool, n):
            sts = []
            for assign in itertools.product(*[opts[o] for o in S]) if S else [()]:
                if len(set(assign)) != len(assign):
                    continue
                st = phys.goal({**cells0, **dict(zip(S, assign))}, q0)[0]
                sts.append((dict(zip(S, assign)), st))
                if st == ps.V:
                    break
            v = db.V if any(s == ps.V for _, s in sts) else db.C if sts and all(s == ps.CI for _, s in sts) else None
            out[S], prov["+".join(S) or "-"] = v, [(a, s) for a, s in sts]
    return {"statuses": out, "provenance": prov, "k": k}


def lattice_from(statuses, pool) -> dict:
    st = {frozenset(pool.index(o) for o in S): v for S, v in statuses.items() if v is not None}
    return db.minimal_sets(len(pool), st)


def item(R, d):
    y0 = d.get("y0")
    if y0 is None:
        return None
    ok = y0 == 0 and d.get("type") == "MINIMAL_SETS" and d.get("mask") is not None and d.get("provenance") == "proof_grounded"
    return (R, d["pool"] if ok else [], y0, d["mask"] if ok else None)


def temp_lattice(phys, cells, q, pool, dest) -> list:
    """Temporary-target lattices at an arbitrary state (cells, q): for each pool object o with destination dest[o], status of the move
    o -> dest[o] with every subset of the other pool objects parked at their dest (VERIFIED / CERTIFIED (pose-only) / UNKNOWN)."""
    import p15_search as ps
    out = []
    for o in pool:
        others = [x for x in pool if x != o]
        st = {}
        for n in range(len(others) + 1):
            for U in itertools.combinations(range(len(others)), n):
                c2 = {**cells, **{others[i]: dest[others[i]] for i in U}}
                if dest[o] in {c for x, c in c2.items() if x != o}:
                    continue
                res = phys.edges(c2, q, o, dest[o])
                s = next((x[0] for x in res if x[0] == ps.V), res[0][0] if res else None)
                st[frozenset(U)] = db.V if s == ps.V else db.C if s in (ps.CI, ps.SI) else None
        lab = db.minimal_sets(len(others), {k: v for k, v in st.items() if v is not None})
        out.append({"target": o, "dest": dest[o], "others": others, "statuses": {"+".join(others[i] for i in sorted(k)) or "-": v for k, v in st.items()}, **lab})
    return out


# ------------------------------------------------------------------ models and per-model calibration
def models(kind):
    """D / Dp: frozen P13; Dp16: P16 retrained pairless; Dp17: P17 retrained pairless (if trained)."""
    import torch
    if kind in ("D", "Dp"):
        import p14_d_search as ds
        return ds.d_models(kind == "Dp")
    path = {"Dp16": "runs/p16/d/ckpt/P16-Dp-l2.0-s{}.pt", "Dp17": "runs/p17/d/ckpt/P17-Dp-l2.0-s{}.pt"}[kind]
    out = []
    for s in range(3):
        m = db.ModelD(pair=False).double()
        m.load_state_dict(torch.load(path.format(s), weights_only=True))
        out.append(m.eval())
    return out


def raw_temp_p(ms, s0, cfg, cells, q, target, dest, others):
    import p15_retarget as pr
    rec = pr.build_record(cfg, pr.retarget_scene(s0, cells, target, q), target, dest)
    return pr.d_predict(ms, rec, target, list(others) + ["t0"])


def _fit_tb(z, y, form="full", iters=200, ridge=1.0):
    """MAP Platt scaling p' = sigmoid(a z + b), Gaussian prior centred on the identity map (precision `ridge`). form: 'identity' (a, b) =
    (1, 0); 'bias' (a = 1, b fitted); 'full' (a, b fitted)."""
    z, y, w0 = np.asarray(z, float), np.asarray(y, float), np.array([1.0, 0.0])
    if form == "identity" or not len(z):
        return 1.0, 0.0
    X, w, free = np.stack([z, np.ones_like(z)], 1), w0.copy(), [0, 1] if form == "full" else [1]
    for _ in range(iters):
        p = 1 / (1 + np.exp(-np.clip(X @ w, -500, 500)))
        Xf = X[:, free]
        H = (Xf * (p * (1 - p))[:, None]).T @ Xf + ridge * np.eye(len(free))
        w[free] = w[free] - np.linalg.solve(H, Xf.T @ (p - y) + ridge * (w[free] - w0[free]))
    return float(w[0]), float(w[1])


def fit_calibration(kind, probes) -> dict:
    """probes: [{'root', 'logit_raw', 'y'}] for THIS model family. The calibration form (identity / bias / full) is selected by root-split
    2-fold cross-validated log loss on the same probes, then fitted on all; p' = sigmoid(a * logit + b)."""
    roots = sorted({p["root"] for p in probes})
    half = set(roots[::2])
    sg = lambda a, b, p: 1 / (1 + np.exp(-np.clip(a * p["logit_raw"] + b, -500, 500)))  # noqa: E731
    ll = lambda rows: float(np.mean([-np.log(max(pp if y else 1 - pp, 1e-12)) for pp, y in rows])) if rows else None  # noqa: E731
    rec = lambda rows, c: float(np.mean([pp >= 0.5 for pp, y in rows if y == c])) if any(y == c for _, y in rows) else None  # noqa: E731
    forms = {}
    for form in ("identity", "bias", "full"):
        cv = []
        for tr_, te_ in ((half, set(roots) - half), (set(roots) - half, half)):
            A, B = [p for p in probes if p["root"] in tr_], [p for p in probes if p["root"] in te_]
            if A and B:
                a, b = _fit_tb([p["logit_raw"] for p in A], [p["y"] for p in A], form)
                cv += [(sg(a, b, p), p["y"]) for p in B]
        forms[form] = {"cv_logloss": ll(cv), "n": len(cv), "feasible_recall": rec(cv, 1), "infeasible_recall": rec(cv, 0)}
    best = min(forms, key=lambda f: (forms[f]["cv_logloss"] if forms[f]["cv_logloss"] is not None else np.inf, ["identity", "bias", "full"].index(f)))
    a, b = _fit_tb([p["logit_raw"] for p in probes], [p["y"] for p in probes], best)
    raw = [(sg(1.0, 0.0, p), p["y"]) for p in probes]
    return {"kind": kind, "n": len(probes), "roots": len(roots), "positives": sum(p["y"] for p in probes), "form": best, "slope": a, "bias": b,
            "raw": {"feasible_recall": rec(raw, 1), "infeasible_recall": rec(raw, 0), "logloss": ll(raw)}, "cv": forms[best], "cv_forms": forms}


class Guide:
    """Reorder-only guide. kinds: D | Dp | Dp16 | Dp17 [-norec | -blk]; geometry[-norec]; uniform; oracle.
    Recursion (default): defer an object whose OWN-family calibrated temporary-target feasibility (to its first candidate slot) < 0.5.
    '-blk' (experimental, not production): additionally promote the objects predicted (marginal >= 0.5) to block a deferred object."""

    def __init__(self, kind, task, phys, movers, oracle=None, dino=None):
        import abc_models as am
        import p15_retarget as pr
        base, _, flag = kind.partition("-")
        self.base, self.flag = base, flag
        self.recursion = flag != "norec" and base not in ("uniform", "oracle")
        self.phys, self.movers, self.oracle = phys, movers, oracle or {}
        self.models = models(base) if base in ("D", "Dp", "Dp16", "Dp17") else []
        cal = OUT / "calibration" / f"{base}.json"
        self.cal = json.loads(cal.read_text()) if cal.exists() else None
        if self.models and self.recursion and self.cal is None:
            raise SystemExit(f"no own calibration for {base}")
        if dino is not None:
            pr._DINO = dino
        self.rec0 = pr.build_record(phys.cfg, phys.s0, "t0", phys.goal_cell)
        self.R0, self.ids = am.Root(self.rec0), self.rec0["audit"]["ids"]
        self.cache, self.d_calls, self.d_cpu, self.trace = {}, 0, 0.0, []

    def _d(self, cells, q, target, goal, pool):
        import p15_retarget as pr
        k = (tuple(sorted(cells.items())), np.asarray(q, float).tobytes(), target, goal, tuple(pool))
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
        defer, promote = set(), set()
        if self.recursion:
            first = {}
            for o, d in cands:
                first.setdefault(o, d)
            for o, d in first.items():
                others = [x for x in pool if x != o]
                if self.base == "geometry":
                    if any(self._geo(cells, q, o, others).values()):
                        defer.add(o)
                    continue
                p = self._d(cells, q, o, d, others)
                lg = np.log(max(p["p_feasible"], 1e-9) / max(1 - p["p_feasible"], 1e-9))
                if 1 / (1 + np.exp(-(self.cal["slope"] * lg + self.cal["bias"]))) < 0.5:
                    defer.add(o)
                    if self.flag == "blk":
                        promote |= {x for x, m in zip(p["pool"], p["marg"]) if m >= 0.5 and x != "t0"}
        out = sorted(cands, key=lambda c: (c[0] in defer and c[0] not in promote, -(c[0] in promote), -sc.get(c[0], 0.0), cands.index(c)))
        if len(self.trace) < 40:
            self.trace.append({"scores": {o: round(v, 4) for o, v in sc.items()}, "deferred": sorted(defer), "promoted": sorted(promote),
                               "first": out[0] if out else None, "order": [list(c) for c in out[:12]]})
        return out


# ------------------------------------------------------------------ probes, per-family calibration, Dp17 training
def _lattice_probes(L, split, root, base, dest, extra):
    """One probe per decided state of a temporary lattice: target L['target'] -> L['dest'] with the subset U of the other pool objects
    parked at their destinations (U = '-' is the unedited state)."""
    out = []
    for U, st in L["statuses"].items():
        y = 1 if st == db.V else 0 if st == db.C else None
        if y is None:
            continue
        parked = {} if U == "-" else {x: int(dest[x]) for x in U.split("+")}
        out.append({"split": split, "root": root, "target": L["target"], "dest": L["dest"], "others": L["others"], "y": y, "type": L["type"],
                    "mask": L["mask"] if U == "-" else None, "U": U, "cells": {**base, **parked} if base or parked else None, **extra})
    return out


def probes_from_inter(inter, split) -> list:
    out = []
    for k, v in inter.items():
        for L in v.get("lattices", []):
            out += _lattice_probes(L, split, k, v["state"]["cells"], v.get("dest", {}), {"q": v["state"]["q"], "state": "intermediate"})
    return out


def probes_from_labels(labs, plan_roots, split) -> list:
    """Temporary-target probes at the original state: (root, target o -> parking[o], other pool objects parked in subset U) -> y (1 VERIFIED /
    0 CERTIFIED); UNKNOWN states are not probes."""
    from scene import scene_from_dict
    out = []
    for k, v in labs.items():
        if not v.get("temp_lattices"):
            continue
        base = {o.id: o.cell for o in scene_from_dict(plan_roots[k]["scene"]).objects}
        for L in v["temp_lattices"]:
            out += _lattice_probes(L, split, k, base, v["catalogue"]["parking"], {"state": "original"})
    return out


def score_probes(kind, probes, plan_by_split) -> list:
    import config as cf
    from scene import scene_from_dict
    ms, cfg = models(kind), cf.load_config("config.toml")
    out = []
    for p in probes:
        s0 = scene_from_dict(plan_by_split[p["split"]]["roots"][p["root"]]["scene"])
        cells = p.get("cells") or {o.id: o.cell for o in s0.objects}
        r = raw_temp_p(ms, s0, cfg, cells, p.get("q") or s0.robot.q_rad, p["target"], p["dest"], p["others"])
        pf = min(max(r["p_feasible"], 1e-9), 1 - 1e-9)
        out.append({**p, "kind": kind, "logit_raw": float(np.log(pf / (1 - pf))), "marg": r["marg"], "pool": r["pool"]})
    return out


def train_dp17(seed, lam=2.0, updates=3000, data="runs/p17/d/D_ITEMS.pt", patience_evals=15, tol=1e-3) -> None:
    """Pairless D on P13 items (PARKING_LATTICE_V1, NO_CLEARING masked) + P16 / P17 roots with MULTI_DEST_LATTICE_V1 labels (tagged). Validation
    every 20 updates; early stop after patience_evals evaluations without relative improvement > tol; best checkpoint restored."""
    import copy
    import math
    import torch
    d = torch.load(data, weights_only=False)
    tr, va = d["train"], d["val"]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    m = db.ModelD(pair=False).double()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    best, hist, c0, stale = (math.inf, None, -1), [], time.process_time(), 0
    for u in range(updates):
        batch = [tr[i] for i in rng.choice(len(tr), 12, replace=False)]
        opt.zero_grad()
        loss = torch.stack([db.nll(*m(R, pool), y0, b, lam) for R, pool, y0, b in batch]).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0)
        opt.step()
        if (u + 1) % 20 == 0:
            with torch.no_grad():
                v = float(np.mean([float(db.nll(*m(R, pool), y0, b, 1.0)) for R, pool, y0, b in va]))
            hist.append({"update": u + 1, "train": float(loss), "val": v})
            if v < best[0] * (1 - tol) if math.isfinite(best[0]) else True:
                best, stale = (v, copy.deepcopy(m.state_dict()), u + 1), 0
            else:
                stale += 1
            if stale >= patience_evals:
                break
    m.load_state_dict(best[1])
    (OUT / "ckpt").mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), OUT / "ckpt" / f"P17-Dp-l{lam}-s{seed}.pt")
    import config as cf
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    cf.write_json(OUT / "runs" / f"P17-Dp-l{lam}-s{seed}.json", {"seed": seed, "lambda_B": lam, "val_nll": best[0], "best_update": best[2], "history": hist,
                                                                "updates_cap": updates, "stopped_early": stale >= patience_evals, "patience_evals": patience_evals,
                                                                "train_items": len(tr), "train_masks": sum(b is not None for *_, b in tr), "val_items": len(va),
                                                                "train_cpu_s": round(time.process_time() - c0, 1), "protocols": d.get("protocols")})
