"""P17 B: complete ordered-repair outcome learning on the shared scene-only domain (0-4 moves, multiple destinations, repeats).

Arms (matched data / optimiser / checkpoint protocol / calibrated rule / existential execution verifier):
  LB     lean keyed B: scene + no-op unaries, F2 prefix-order repair unaries, couple 'none' (exact product of Bernoullis); lambda_static =
         lambda_travel = 0 (analytic static check only)                                       -> production default
  LB+G   LB + generic-only coupling (cpl; F3 / F4 pairs dropped; HO absent)                    -> experimental replication arm
  LB-W   LB with the scene head widened to h = 320 (516,901 vs LB+G 516,774 parameters)        -> parameter-matched noncoupled control
  LB-F2  order-invariant summary; LC  GRU order; A, SF  P13 architectures                      -> controls
  FULLB  historical full keyed B (F1, F3, F4, F5 + HO, J) retrained on the SAME data with its historical objective -> comparator only
Queries use layout P17_LAYOUT_V1 (as Z_LAYOUT_V3_R4 but repeated objects allowed, never consecutively; one variable per physical query;
an omission recomputes all intermediate poses because the encoder replays actions from the initial poses). Labels are produced by the
existential fixed-sequence verifier (p17_verify) with per-sequence budgets over a shared exact cache; positives are replay-certified."""
from __future__ import annotations

import contextlib
import functools
import itertools
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("P10C_OUT", "runs/p13/models")
import config as cf  # noqa: E402

OUT = Path("runs/p17/b")
LAYOUT = "P17_LAYOUT_V1"
MOVE_CAP, MAX_ACTIVE = 4, 9
SEQ_BUDGET = {"motion_checks": 150, "online_moves": 4, "goal_plans": 6, "transit_plans": 4}      # P17_SEQ_BUDGET_V1 (per sequence)
B_CAPS = {0: 1, 1: 6, 2: 8, 3: 4, 4: 3}                                                         # declared B catalogue caps per length
ARMS = {"LB": ("B", "none", "pred"), "LB+G": ("B", "G", "pred"), "LB-W": ("B", "none", "pred"), "LB-F2": ("B", "none", "none"),
        "LC": ("C", "none", "pred"), "A": ("A", None, None), "SF": ("SF", None, None), "FULLB": ("B", "full", "pred")}
KEYED = {"LB", "LB+G", "LB-W", "LB-F2", "LC", "FULLB", "K-B", "K-C", "B-F2"}


def sig(seq) -> str:
    return ".".join(f"{o}>{c}" for o, c in seq) or "empty"


def layout(root, seq, catalogue) -> dict:
    """P17_LAYOUT_V1: z0, scene edits, zR, zO[t] (t < m, m >= 2); repeats allowed; distinct active queries; <= MAX_ACTIVE."""
    seq = [(o, int(c)) for o, c in seq]
    m = len(seq)
    if m > MOVE_CAP or any(seq[i][0] == seq[i + 1][0] for i in range(m - 1)):
        raise ValueError(f"not a P17 sequence: {seq}")
    slots, active, query = ["z0"], [True], {"z0": f"seq:{root}:empty"}
    for q in catalogue:
        s = f"zS[{q['kind']}]"
        slots.append(s)
        active.append(bool(q["eligible"]))
        query[s] = f"scene:{root}:{sig(sorted((o, int(c)) for o, c in q['edit'].items()))}"
    slots.append("zR")
    active.append(m >= 1)
    query["zR"] = f"seq:{root}:{sig(seq)}"
    for t in range(MOVE_CAP):
        slots.append(f"zO[{t}]")
        om = [a for i, a in enumerate(seq) if i != t]
        ok = m >= 2 and t < m and all(om[i][0] != om[i + 1][0] for i in range(len(om) - 1))
        active.append(ok)
        query[f"zO[{t}]"] = f"seq:{root}:{sig(om)}" if t < m else None
    act_q = [query[s] for s, a in zip(slots, active) if a]
    if len(set(act_q)) != len(act_q):                       # an omission aliasing another slot is the same physical query: drop it
        seen, keep = set(), []
        for s, a in zip(slots, active):
            keep.append(a and query[s] not in seen)
            if a:
                seen.add(query[s])
        active = keep
    if sum(active) > MAX_ACTIVE:
        raise ValueError(f"{sum(active)} active slots exceed {MAX_ACTIVE}")
    lay = {"version": LAYOUT, "root": root, "sequence": [list(a) for a in seq], "slots": slots, "active": active,
           "query": {s: query[s] for s, a in zip(slots, active) if a}, "active_count": sum(active)}
    lay["hash"] = cf.stable_hash(lay)
    return lay


def edits_for(seq, lattice):
    """Scene variables per sequence: parked-subset edits over the sequence's distinct objects (all edits for the empty repair), within
    the exact bound."""
    objs = set(o for o, _ in seq)
    if not seq:
        return lattice
    sub = [e for e in lattice if set(e["edit"]) <= objs]
    room = MAX_ACTIVE - 2 - (len(seq) if len(seq) >= 2 else 0)
    return sorted(sub, key=lambda e: (-len(e["edit"]), e["kind"]))[:max(0, room)]


def catalogue(task, phys, idx, caps=None) -> dict:
    """B's scene-only catalogue: P16 parking lattice (scene variables, PARKING_LATTICE_V1) + P17 domain ordered repairs (0-4 moves)."""
    import p16_common as pc
    import p17_domain as dm
    lat = pc.catalogue(task, phys, idx)
    rep = dm.repairs(task, phys, idx, caps=caps or B_CAPS)
    edits = [{"kind": "+".join(sorted(u)), "edit": {o: lat["parking"][o] for o in u}, "eligible": True}
             for n in range(1, len(lat["pool"]) + 1) for u in itertools.combinations(lat["pool"], n)]
    return {"pool": lat["pool"], "parking": lat["parking"], "edits": edits, "sequences": rep["sequences"], "generated": rep["generated"],
            "kept": rep["kept"], "omitted": rep["omitted"], "movers": rep["movers"]}


def label(task, phys_factory, cat, idx) -> dict:
    """Existential labels: one provider per sequence (own budget) over a shared exact cache. y = 1 VERIFIED (+ replay CLEAR), 0 only
    CERTIFIED_INFEASIBLE (motion-independent scope), None otherwise. Scene edits / original: target query at q0 in the edited scene."""
    import p16_motion as pm
    import p17_verify as pv
    cache = {}
    base = phys_factory(cache)
    cells0, q0 = {o.id: o.cell for o in base.s0.objects}, np.asarray(base.s0.robot.q_rad, float)
    yof = lambda st: 1 if st == pv.V else 0 if st == pv.CI else None  # noqa: E731
    st0 = base.goal(cells0, q0)[0]
    edits = {e["kind"]: base.goal({**cells0, **e["edit"]}, q0)[0] for e in cat["edits"]}
    seqs = {}
    for s in cat["sequences"]:
        if not s:
            continue
        ph = phys_factory(cache)
        c0 = time.process_time()
        r = pv.verify_existential(ph, [tuple(a) for a in s], first=True)
        rep = pm.replay(ph, r["plan"]) if r["plan"] else None
        st = r["status"] if not (r["status"] == pv.V and not (rep or {}).get("ok")) else "UNKNOWN_REPLAY_FAILED"
        seqs[sig(s)] = {"status": st, "y": yof(st), "travel": r["plan"]["travel"] if r["plan"] else None, "len": len(s),
                        "branches_verified": r.get("branches_verified"), "travel_minimal": r.get("travel_minimal_over_alternatives"),
                        "certificate": r.get("certificate"), "replay_traj": (rep or {}).get("traj"), "used": ph.used, "cpu_s": time.process_time() - c0}
    return {"original": {"status": st0, "y": yof(st0)}, "edits": {k: {"status": v, "y": yof(v)} for k, v in edits.items()}, "sequences": seqs}


def labels_of(k, lab, cat) -> dict:
    out = {f"seq:{k}:empty": lab["original"]["y"]}
    for kind, v in lab["edits"].items():
        out[f"scene:{k}:{sig(sorted((o, cat['parking'][o]) for o in kind.split('+')))}"] = v["y"]
    for s, v in lab["sequences"].items():
        out[f"seq:{k}:{s}"] = v["y"]
    costs = {f"seq:{k}:{s}": v["travel"] for s, v in lab["sequences"].items() if v["y"] == 1}
    return {q: {"y": y, "source": "P17_EXISTENTIAL" if y is not None else "UNKNOWN", "weight": 1.0, "cost": costs.get(q)} for q, y in out.items()}


def records(plan_roots, labs, split, dino=None, with_labels=True) -> list:
    import learn_data as ld
    import p10c_train_eval as pc
    cfg, dino = cf.load_config("config.toml"), dino or ld.dino()
    recs = []
    for k in sorted(labs):
        r, cat = plan_roots[k], labs[k]["catalogue"]
        lattice = [{**e, "edit_kind": e["kind"]} for e in cat["edits"]]
        items = [{"kind": "P17_CANDIDATE", "sequence": [list(a) for a in s], "z_layout": layout(k, s, edits_for(s, cat["edits"]))}
                 for s in cat["sequences"]]
        rec, _ = pc._record(cfg, dino, k, r, labels_of(k, labs[k], cat) if with_labels else {}, split, "P17_SCENE_ONLY", lattice, items)
        rec["audit"] |= {"family": r.get("family"), "catalogue": {x: cat[x] for x in ("pool", "parking", "movers")}, "layout": LAYOUT}
        recs.append(rec)
    return recs


def prepare(recs):
    import abc7_train as t7
    rs = t7.prepare({"roots": recs})
    for r, rec in zip(rs, recs):
        r["ids"], r["audit"] = rec["audit"]["ids"], rec["audit"]
    return rs


# ------------------------------------------------------------------ models, training, scoring
def model(arm):
    import abc7_models as a7
    import abc8
    import abc_models as am
    kind, couple, order = ARMS[arm]
    if arm == "LB+G":
        return a7.ModelBC7("B", "quadratic", order="pred", drop=("pair_scene", "pair_omit")).double()
    if arm == "FULLB":
        return abc8.build("B-full")
    if arm in ("A", "SF"):
        return abc8.build(arm)
    m = a7.ModelBC7(kind, "none", order=order).double()
    if arm == "LB-W":
        m.scene = am.mlp(2 * am.W, 1, h=320).double()
    return m


def fit(arm, seed, train, val, updates, patience_evals=15):
    """P13 / P16 optimiser (AdamW 1e-3, batch 12 RootSampler, clip 10) with validation early stopping on root NLL (_fit_es); FULLB keeps its historical
    lambda_static = lambda_travel = 1, every other arm uses 0."""
    import joint_likelihood as jl
    import unique_factor as uf
    import unique_factor_keyed as ufk
    objective = "poe" if arm in ("A", "SF") else "unique"
    lam = 1.0 if arm == "FULLB" else 0.0
    old_u, old_j = uf.total, jl.total
    uf.total, jl.total = functools.partial(old_u, lam_static=lam, lam_travel=lam), functools.partial(old_j, lam_static=lam, lam_travel=lam)
    try:
        with ufk.installed() if arm in KEYED else contextlib.nullcontext():
            return _fit_es(arm, seed, updates, train, val, objective, patience_evals)
    finally:
        uf.total, jl.total = old_u, old_j


def _fit_es(arm, seed, updates, train, val, objective, patience_evals, every=20, tol=1e-3):
    """p9_l3.fit (same optimiser, sampler, clip, validation cadence, best-checkpoint restore) plus early stopping: stop once validation root
    NLL has not improved by more than tol (relative) for patience_evals consecutive evaluations; updates is the hard cap."""
    import collections
    import copy
    import abc8
    import joint_likelihood as jl
    import unique_factor as uf
    torch.manual_seed(seed)
    np.random.seed(seed)
    m = model(arm)
    tot_fn = uf.total if objective == "unique" else jl.total
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    smp, hist, best, stale = abc8.RootSampler(train, 12, seed), [], (float("inf"), None, -1), 0
    for u in range(updates):
        opt.zero_grad()
        tot, rec = tot_fn(m, [train[i] for i in smp.next()])
        if not torch.isfinite(tot):
            hist.append({"update": u, "nonfinite": True})
            break
        tot.backward()
        g = collections.defaultdict(float)
        for n, p in m.named_parameters():
            if p.grad is not None:
                g[n.split(".")[0]] += float(p.grad.norm()) ** 2
        pre = float(torch.nn.utils.clip_grad_norm_(m.parameters(), 10.0))
        opt.step()
        rec |= {"update": u, "total": float(tot.detach()), "grad_pre": pre, "clipped": pre > 10.0, "grad_by_module": {k: round(v ** 0.5, 5) for k, v in g.items()}}
        if (u + 1) % every == 0:
            with torch.no_grad():
                rec["val"] = tot_fn(m, val)[1]["nll"]
            if rec["val"] < best[0] * (1 - tol) if np.isfinite(best[0]) else True:
                best, stale = (rec["val"], copy.deepcopy(m.state_dict()), u + 1), 0
            else:
                stale += 1
        hist.append(rec)
        if stale >= patience_evals:
            break
    if best[1] is not None:
        m.load_state_dict(best[1])
    return m, {"history": hist, "best_update": best[2], "val_root_nll": best[0], "updates": updates, "updates_run": len(hist),
               "stopped_early": stale >= patience_evals, "patience_evals": patience_evals, "eval_every": every, "tol": tol,
               "params": sum(p.numel() for p in m.parameters()),
               "clip_fraction": float(np.mean([h.get("clipped", False) for h in hist])), "nonfinite": any(h.get("nonfinite") for h in hist)}


def train(arm, seed, data="runs/p17/b/TRAIN.pt", updates=3000) -> None:
    import p17_eval as pe
    d = torch.load(data, weights_only=False)
    tr, va = prepare(d["train"]), prepare(d["val"])
    c0 = time.process_time()
    m, rec = fit(arm, seed, tr, va, updates)
    ck = OUT / "ckpt" / f"P17-{arm}-s{seed}.pt"
    ck.parent.mkdir(parents=True, exist_ok=True)
    torch.save(m.state_dict(), ck)
    (OUT / "runs").mkdir(parents=True, exist_ok=True)
    hist = [{k: h.get(k) for k in ("update", "total", "nll", "val", "grad_pre", "clipped", "grad_by_module")} for h in rec["history"]]
    cf.write_json(OUT / "runs" / f"P17-{arm}-s{seed}.json", {k: v for k, v in rec.items() if k != "history"} | {
        "history": hist, "arm": arm, "seed": seed, "train_roots": len(tr), "val_roots": len(va), "updates": updates,
        "params": sum(p.numel() for p in m.parameters()), "train_cpu_s": round(time.process_time() - c0, 1), "checkpoint": pe.sha(ck), "data": pe.sha(data)})


def load(arm, seed):
    if arm.startswith("P13:"):
        import p16_b
        return p16_b.load_frozen(arm[4:], seed)
    m = model(arm)
    m.load_state_dict(torch.load(OUT / "ckpt" / f"P17-{arm}-s{seed}.pt", weights_only=True))
    return m.eval()


@contextlib.contextmanager
def coupling_off(m):
    """Inference-only toggle: the generic coupling network outputs 0 (weights frozen, untouched)."""
    old = getattr(m, "cpl", None)
    if old is None:
        yield m
        return

    class Z(torch.nn.Module):
        def forward(self, x):
            return torch.zeros(x.shape[0], 1, dtype=x.dtype)
    m.cpl = Z()
    try:
        yield m
    finally:
        m.cpl = old


def score(m, arm, roots) -> dict:
    """{root: [{query, seq, m, u, p}]} deployable prior marginals (no labels used): exactly p9_l3.root_eval9's prior-marginal rows, without
    its offline conditionals / evidence (which deployment never uses)."""
    import abc7_train as t7
    import joint_likelihood as jl
    import unique_factor as uf
    import unique_factor_keyed as ufk
    base = arm[4:] if arm.startswith("P13:") else arm.split("|")[0]
    mode = "factorized" if base in ("A", "SF") else "unique"
    out = {}
    with ufk.installed() if base in KEYED else contextlib.nullcontext(), torch.no_grad():
        for r in roots:
            sq = [q for c in r["cands"] if not c["seq"] for s, q in zip(c["slots"], c["z_q"]) if s[:2] in ("z0", "zS")]
            if mode == "unique":
                pm = uf.prior_marginals(uf.instances(m, r))
            else:
                pm = {q: float(torch.sigmoid(x)) for q, x in jl.factorized_logits(m, r, t7.forward(m, r)).items()}
            rows = []
            for c in r["cands"]:
                p = min(max(pm[c["query"] if c["seq"] else sq[0]], 1e-9), 1 - 1e-9)
                rows.append({"query": c["query"], "seq": [(r["ids"][i], d) for i, d in c["seq"]], "m": len(c["seq"]), "u": float(np.log(p) - np.log1p(-p)), "p": p})
            out[r["key"]] = rows
    return out


def score_slow(m, arm, roots) -> dict:
    """Reference: the same rows through p9_l3.root_eval9 (used by the equality test)."""
    import p9_l3 as l3
    import unique_factor_keyed as ufk
    base = arm[4:] if arm.startswith("P13:") else arm.split("|")[0]
    mode = "factorized" if base in ("A", "SF") else "unique"
    out = {}
    with ufk.installed() if base in KEYED else contextlib.nullcontext(), torch.no_grad():
        for r in roots:
            ev = l3.root_eval9(m, r, mode)["root"]
            out[r["key"]] = [{"query": c["query"], "seq": [(r["ids"][i], d) for i, d in c["seq"]], "m": len(c["seq"]), "u": row["u"], "p": row["p"]}
                             for row, c in zip(ev["cands"], r["cands"])]
    return out
