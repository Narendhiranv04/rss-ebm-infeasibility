"""P16 lean B: order-sensitive keyed Boltzmann evaluation of COMPLETE ordered repair queries (including the empty repair).

Default (arm LB): DINOv2 + numeric context, keyed physical-query identities, scene unaries (z0 = no-op, zS = parked-subset scenes) and
order-sensitive repair unaries (B prefix-pair F2), no learned F1 / F3 / F4 / F5 / HO / J (lambda_static = lambda_travel = 0; the static
veto is analytic). With couple = 'none' the keyed joint is EXACTLY a product of Bernoullis, one per unique physical query:
    P(y | S, Q) = prod_q sigma((2 y_q - 1) s_q / T),   loss = -sum_{observed q} log sigma((2 y_q - 1) s_q / T)   (UNKNOWN summed out)
so no interaction reasoning is claimed. Controls: LB-F2 (order-invariant summary), LC (GRU order), A / SF (P13 architectures retrained on
the same data), frozen P13 K-B / A / SF / K-C / B-F2, and non-learned selectors (geometry, empty-always, random).
Records come from p16_common.catalogue (scene-only) and deterministic physical labels (p16_common.label)."""
from __future__ import annotations

import contextlib
import functools
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("P10C_OUT", "runs/p13/models")
import config as cf  # noqa: E402

OUT = Path("runs/p16/b")
ARMS = {"LB": ("B", "pred", "unique", True), "LB-F2": ("B", "none", "unique", True), "LC": ("C", "pred", "unique", True),
        "LB+G": ("B", "pred", "unique", True), "A": ("A", None, "poe", False), "SF": ("SF", None, "poe", False)}
FROZEN = ("K-B", "A", "SF", "K-C", "B-F2")


def model(arm):
    import abc7_models as a7
    import abc8
    kind, order, _, _ = ARMS[arm]
    if arm == "LB+G":                         # generic-only coupling: cpl kept, F3 / F4 pairs dropped, HO absent (couple 'quadratic')
        return a7.ModelBC7("B", "quadratic", order=order, drop=("pair_scene", "pair_omit")).double()
    if kind in ("B", "C"):
        return a7.ModelBC7(kind, "none", order=order).double()
    return abc8.build(kind)


# ------------------------------------------------------------------ records (scene-only catalogue + deterministic labels)
def items(k, cat):
    """(edit_items, seq_items): every lattice edit; per sequence a z-layout whose scene slots are the edits over the sequence's own
    objects (all edits for the empty repair, so the scene block is the full lattice), within the exact-enumeration bound."""
    import itertools
    import zlayout_r4 as z4
    edits = [{"kind": "+".join(u), "edit": {o: cat["parking"][o] for o in u}, "eligible": True}
             for n in range(1, len(cat["pool"]) + 1) for u in itertools.combinations(cat["pool"], n)]
    seqs = [[]] + [[o] for o in cat["pool"]] + [list(p) for p in itertools.permutations(cat["pool"], 2)]
    out = []
    for s in seqs:
        objs = set(s)
        sub = edits if not s else [e for e in edits if set(e["edit"]) <= objs and (len(objs) < 3 or len(e["edit"]) >= 2)]
        seq = [(o, cat["parking"][o]) for o in s]
        out.append({"kind": "SCENE_ONLY_CANDIDATE", "sequence": [list(a) for a in seq], "z_layout": z4.build(k, seq, sub)})
    return [{**e, "edit_kind": e["kind"]} for e in edits], out


def labels_of(k, lab, cat) -> dict:
    import zlayout_r4 as z4
    out = {f"seq:{k}:empty": lab["original"]["y"]}
    for kind, v in lab["edits"].items():
        out[f"scene:{k}:{z4.sig(sorted((o, cat['parking'][o]) for o in kind.split('+')))}"] = v["y"]
    for sig, v in lab["sequences"].items():
        out[f"seq:{k}:{sig}"] = v["y"]
    return {q: {"y": y, "source": "P16_DET_CERT" if y is not None else "UNKNOWN", "weight": 1.0, "cost": None} for q, y in out.items()}


def records(plan, labs, split, dino=None, with_labels=True) -> list:
    import p10c_train_eval as pc
    import learn_data as ld
    cfg, dino = cf.load_config("config.toml"), dino or ld.dino()
    recs = []
    for k in sorted(labs):
        r, cat = plan["roots"][k], labs[k]["catalogue"]
        ed, sq = items(k, cat)
        rec, _ = pc._record(cfg, dino, k, r, labels_of(k, labs[k], cat) if with_labels else {}, split, "P16_SCENE_ONLY", ed, sq)
        rec["audit"] |= {"family": r["family"], "catalogue": cat}
        recs.append(rec)
    return recs


def prepare(recs):
    import abc7_train as t7
    rs = t7.prepare({"roots": recs})
    for r, rec in zip(rs, recs):
        r["ids"], r["audit"] = rec["audit"]["ids"], rec["audit"]
    return rs


# ------------------------------------------------------------------ training (matched arms, same data, validation-only checkpoint)
def fit(arm, seed, train, val, updates=400):
    import abc8
    import joint_likelihood as jl
    import p9_l3 as l3
    import unique_factor as uf
    import unique_factor_keyed as ufk
    _, _, objective, keyed = ARMS[arm]
    l3.build = lambda a: model(arm)
    old_u, old_j = uf.total, jl.total
    uf.total, jl.total = functools.partial(old_u, lam_static=0.0, lam_travel=0.0), functools.partial(old_j, lam_static=0.0, lam_travel=0.0)
    try:
        with ufk.installed() if keyed else contextlib.nullcontext():
            return l3.fit(arm, seed, updates, train, val, objective)
    finally:
        uf.total, jl.total = old_u, old_j
        del abc8


def train(arm, seed, data="runs/p16/b/TRAIN.pt", updates=400) -> None:
    d = torch.load(data, weights_only=False)
    tr, va = prepare(d["train"]), prepare(d["val"])
    c0 = time.process_time()
    m, rec = fit(arm, seed, tr, va, updates)
    ck = OUT / "ckpt" / f"P16-{arm}-s{seed}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ck)
    import p16_eval as pe
    rec["history"] = [{k: h.get(k) for k in ("update", "total", "nll", "val", "grad_pre", "clipped", "grad_by_module")} for h in rec["history"]]
    cf.write_json(OUT / "runs" / f"P16-{arm}-s{seed}.json", rec | {"arm": arm, "seed": seed, "train_roots": len(tr), "val_roots": len(va),
                                                                  "params": sum(p.numel() for p in m.parameters()),
                                                                  "train_cpu_s": round(time.process_time() - c0, 1), "checkpoint": pe.sha(ck),
                                                                  "data": pe.sha(data)})


def load(arm, seed):
    m = model(arm)
    m.load_state_dict(torch.load(OUT / "ckpt" / f"P16-{arm}-s{seed}.pt", weights_only=True))
    return m.eval()


def load_frozen(arm, seed):
    import p13_pipeline as pp
    pc = pp._arms()
    m, x = pc.load(f"P13-{arm}-s{seed}")
    return m


# ------------------------------------------------------------------ scoring and the calibrated selection policy
def score(m, arm, roots) -> dict:
    """{root: [{query, seq, m, u, p}]}: per-candidate deployable prior marginals (no labels used). arm: P16 arm or 'P13:<arm>'."""
    import p9_l3 as l3
    import unique_factor_keyed as ufk
    base = arm[4:] if arm.startswith("P13:") else arm
    mode = "factorized" if base in ("A", "SF") else "unique"
    keyed = base in ("K-B", "K-C", "B-F2", "LB", "LB-F2", "LC", "LB+G")
    out = {}
    with ufk.installed() if keyed else contextlib.nullcontext(), torch.no_grad():
        for r in roots:
            ev = l3.root_eval9(m, r, mode)["root"]
            out[r["key"]] = [{"query": c["query"], "seq": [(r["ids"][i], d) for i, d in c["seq"]], "m": len(c["seq"]), "u": row["u"], "p": row["p"]}
                             for row, c in zip(ev["cands"], r["cands"])]
    return out


def load_any(arm, seed):
    return load_frozen(arm[4:], seed) if arm.startswith("P13:") else load(arm, seed)


def choose(rows, t0, t, temp=1.0, bias=0.0):
    """No-op first: EMPTY if p'(empty) >= t0; else the admitted (p' >= t) repair with fewest moves, then highest p'; else ABSTAIN.
    p' = sigmoid((u - bias) / temp) (validation / calibration-only temperature + bias). EMPTY, ABSTAIN and repairs are distinct."""
    pp = lambda u: 1 / (1 + math.exp(-(u - bias) / temp))  # noqa: E731
    e = next(r for r in rows if r["m"] == 0)
    if pp(e["u"]) >= t0:
        return e["query"]
    adm = [r for r in rows if r["m"] > 0 and pp(r["u"]) >= t]
    return min(adm, key=lambda r: (r["m"], -r["u"], r["query"]))["query"] if adm else "ABSTAIN"


def calibrate(scores, labs, grid=np.round(np.linspace(0.05, 0.95, 91), 3)) -> dict:
    """Calibration split only: (t0, t) maximizing teacher-scored correct decisions (EMPTY on y0 = 1, a y = 1 repair on y0 = 0); ties ->
    larger t0 then larger t. UNKNOWN-labelled choices count as not correct (conservative)."""
    def acc(t0, t):
        ok = []
        for k, rows in scores.items():
            q = choose(rows, t0, t)
            ok.append(labs[k].get(q) == 1)
        return float(np.mean(ok))
    best = max(((acc(a, b), a, b) for a in grid for b in grid[::3]), key=lambda x: (x[0], x[1], x[2]))
    return {"t0": float(best[1]), "t": float(best[2]), "calib_correct": best[0], "n": len(scores)}
