"""G4B exact sampler certification (G4B contract §6): synthetic mathematical fixtures, an
independently written reference enumerator, the exact one-step transition-matrix proof
and the B/C bounded-z gold standard. Writes runs/g4b_sampler_exact/.

Every energy here is a deterministic SYNTHETIC TEST FIXTURE (seeded tables, documented
formulas): not a model prediction, not a learned result, not physical evidence. No
planner, teacher or target code is imported; `cmd_g4b` additionally wraps any loaded
planner/skill/target entry point with a call counter and records the (zero) count.
"""
from __future__ import annotations

import itertools
import math
import time

import numpy as np

import sampler as sp
from config import stable_hash

T0, TOL = 1.0, 1e-10
CHAIN = {"steps": 200_000, "seed": 20261020, "trace_steps": 160}
B_SLOTS = ("z0", "zS1", "zS2", "zR", "zO0", "zO1")


# ------------------------------------------------------------------ reference enumeration (independent of sampler.py)
def reference_states(dests: dict, cap: int) -> list:
    """Every canonical ordered state, generated directly: empty, then each ordered tuple of
    distinct eligible objects (itertools.permutations) times each destination product."""
    objs = sorted(o for o, ds in dests.items() if ds)
    out = [()]
    for m in range(1, cap + 1):
        for chosen in itertools.permutations(objs, m):
            for ds in itertools.product(*[dests[o] for o in chosen]):
                out.append(tuple(zip(chosen, ds)))
    return out


def reference_pi(states: list, energy, temperature: float) -> np.ndarray:
    """Direct exp/normalize in float64 with a max shift (independent of sampler.logsumexp)."""
    x = np.array([-energy(s) / temperature for s in states], dtype=np.float64)
    w = np.exp(x - x.max())
    return w / w.sum()


# ------------------------------------------------------------------ synthetic A energies
def toy_energy(dests: dict, seed: int, order_scale: float = 1.0, invalid=(), penalty: float = 4.0):
    """Seeded unary, symmetric pair, predecessor-order and position terms plus a move-count
    and endpoint-displacement term; `invalid` endpoint pairs get a finite static penalty
    (searchable, never masked)."""
    rng = np.random.default_rng(seed)
    acts = [(o, d) for o, ds in sorted(dests.items()) for d in ds]
    U = {a: float(rng.normal()) for a in acts}
    V = {}
    for a, b in itertools.combinations(acts, 2):
        V[(a, b)] = V[(b, a)] = float(rng.normal()) * 0.7
    O = {(o1, o2): order_scale * float(rng.normal()) for o1 in dests for o2 in dests if o1 != o2}
    P = {(a, t): 0.3 * float(rng.normal()) for a in acts for t in range(3)}
    home = {o: i * 10 for i, o in enumerate(sorted(dests))}
    bad = {frozenset(p) for p in invalid}

    def E(r) -> float:
        e = 0.15 * len(r) + sum(U[a] + P[(a, t)] + 0.02 * abs(a[1] - home[a[0]]) for t, a in enumerate(r))
        e += sum(V[(a, b)] + (penalty if frozenset((a, b)) in bad else 0.0) for a, b in itertools.combinations(r, 2))
        return e + sum(O[(r[t - 1][0], r[t][0])] for t in range(1, len(r)))
    return E


SPACES = {   # name -> (destinations, cap, energy kwargs, expected state count)
    "A1": ({"a": (1, 2), "b": (3, 4)}, 2, {"seed": 11, "order_scale": 1.5}, 13),
    "A2": ({"a": (7,), "b": (3, 4), "c": (5, 6, 8), "d": ()}, 2, {"seed": 12}, 29),
    "A3_cap0": ({"a": (1, 2), "b": (3,)}, 0, {"seed": 13}, 1),
    "A3_no_birth": ({"a": (), "b": ()}, 2, {"seed": 14}, 1),
    "A3_single_K1": ({"a": (1,)}, 1, {"seed": 15}, 2),
    "A3_static_invalid": ({"a": (9, 1), "b": (9,)}, 2, {"seed": 16, "invalid": [(("a", 9), ("b", 9))]}, 8),
}


def space(name):
    dests, cap, kw, n = SPACES[name]
    return sp.Domain.make(dests, cap), toy_energy(dests, **kw), dests, cap, n


# ------------------------------------------------------------------ synthetic B / C joint energies
def layout_fn(r, scene_queries=("zS1",), extra_inactive=()):
    """Deterministic mask rule known before outcomes: z0 and the catalogue's scene queries are
    always active; zR only for a nonempty repair (the empty repair's outcome is z0);
    single-action omission slots zO[t] only when m >= 2 (omitting the only action is the
    empty repair again). Other slots are inactive with value 0."""
    m = len(r)
    act = {"z0": True, "zS1": "zS1" in scene_queries, "zS2": "zS2" in scene_queries, "zR": m >= 1,
           "zO0": m >= 2, "zO1": m >= 2, **{x: False for x in extra_inactive}}
    slots = B_SLOTS + tuple(extra_inactive)
    return sp.Layout(slots, tuple(act[s] for s in slots), tuple(0 for _ in slots))


THETA = {"B": np.array([0.6, -1.1, 0.9, 1.4, -0.8, 0.5]), "C": np.array([0.4, -0.7, 1.2, 1.1, -1.3, 0.8])}


def joint_features(kind: str, r, z: dict) -> np.ndarray:
    """phi(r, z): E = theta . phi + static(r). Prior on z0; scene-query factor coupled to z0;
    an order-sensitive repair-success factor (B: predecessor pair order; C: a recurrent
    toy GRU-like summary); omission factors tied to zR; and a repair-dependent coupling
    between the original outcome, a scene outcome and the repair outcome."""
    g = z.get
    z0, zr = g("z0", 0), g("zR", 0)
    if kind == "B":
        seq = -1.2 if r[:2] == (("a", 2), ("b", 3)) else (0.9 if len(r) == 2 and r[0][0] == "b" else 0.2 * len(r))
    else:
        h = 0.0
        for t, (o, d) in enumerate(r):
            h = math.tanh(0.8 * h + (0.5 if o == "a" else -0.4) * d / 4 + 0.3 * t)
        seq = 1.5 * h
    scene = sum(z[s] * (0.5 * z0 - (1.0 if s == "zS1" else 0.4)) for s in ("zS1", "zS2") if s in z)
    omit = sum(zr * (1 - z[s]) * (0.7 if s == "zO0" else 0.3) for s in ("zO0", "zO1") if s in z)
    first_a = 1.0 if r and r[0][0] == "a" else -0.5
    couple = zr * (1 - z0) * first_a + g("zS1", 0) * zr * (1.0 if any(o == "a" for o, _ in r) else -1.0)
    return np.array([z0, zr * seq, scene, omit, couple, zr * len(r) - (1 - zr) * 0.3])


def joint_energy(kind: str, theta=None, static=None):
    th = THETA[kind] if theta is None else theta
    st = static or space("A1")[1]
    return lambda r, z: float(st(r) + th @ joint_features(kind, r, z))


def gold_joint(states, joint, lay, temperature):
    """Independent (r, z) enumeration: own product over each repair's active slot names."""
    rows = []
    for r in states:
        L = lay(r)
        names = [n for n, a in zip(L.slots, L.active) if a]
        for bits in itertools.product((0, 1), repeat=len(names)):
            z = dict(zip(names, bits))
            rows.append((r, z, -joint(r, z) / temperature))
    x = np.array([v for _, _, v in rows])
    w = np.exp(x - x.max())
    return [(r, z, p) for (r, z, _), p in zip(rows, w / w.sum())]


# ------------------------------------------------------------------ exact certification
def certify(domain, energy, temperature, states, probs=sp.DEFAULT_PROBS, ref=None) -> dict:
    """Counts, uniqueness, per-kernel normalization, row sums, finiteness, detailed balance
    per reversible kernel group, stationarity, reference-vs-logsumexp pi, nontrivial moves."""
    pi = reference_pi(states, energy, temperature) if ref is None else ref
    lse = np.array([math.exp(-energy(s) / temperature - sp.logsumexp([-energy(v) / temperature for v in states]))
                    for s in states])
    p = sp.check_probs(probs)
    out = {"states": len(states), "unique": len(set(states)) == len(states),
           "canonical": all(sp.canonical_state(domain, s) == s for s in states),
           "pi_reference_vs_logsumexp": float(np.abs(pi - lse).max())}
    mats = {k: sp.transition_matrix(domain, states, energy, temperature, p, (k,)) for k in sp.KERNELS}
    P = sum(mats.values())
    out["row_sum_error"] = float(np.abs(P.sum(1) - 1).max())
    out["kernel_row_sum_error"] = float(max(np.abs(mats[k].sum(1) - p[i]).max() for i, k in enumerate(sp.KERNELS)))
    out["finite_nonnegative"] = bool(np.isfinite(P).all() and (P >= -1e-15).all())
    for name, ks in (("DESTINATION", ("DESTINATION",)), ("SWAP", ("SWAP",)), ("BIRTH_DEATH", ("BIRTH", "DEATH")),
                     ("ALL", sp.KERNELS)):
        M = sum(mats[k] for k in ks)
        F = pi[:, None] * M
        out[f"detailed_balance_{name}"] = float(np.abs(F - F.T).max())
    out["stationarity_error"] = float(np.abs(pi @ P - pi).max())
    out["offdiagonal_transitions"] = {k: int(((mats[k] > 0) & ~np.eye(len(states), dtype=bool)).sum()) for k in sp.KERNELS}
    errs = [v for k, v in out.items() if k.endswith(("error", "_reference_vs_logsumexp")) or k.startswith("detailed")]
    out["max_error"] = float(max(errs))
    out["pass"] = bool(out["unique"] and out["canonical"] and out["finite_nonnegative"] and out["max_error"] <= TOL)
    return out, pi, P


def certify_bc(kind: str, temperature=T0, scene_queries=("zS1",)) -> dict:
    """Gold-standard joint vs the sampler's marginal route for one B/C fixture."""
    domain, static, dests, cap, _ = space("A1")
    states = reference_states(dests, cap)
    lay = lambda r: layout_fn(r, scene_queries)  # noqa: E731
    joint = joint_energy(kind, static=static)
    gold = gold_joint(states, joint, lay, temperature)
    marg = sp.MarginalEnergy(joint, lay, temperature, {"version": "toy-1", "scene_queries": list(scene_queries)})
    pr = np.array([sum(p for r, _, p in gold if r == s) for s in states])
    cert, pi, P = certify(domain, marg, temperature, states, ref=pr)
    cond_err = max(abs(dict((tuple(sorted(z.items())), q) for z, q in marg.conditional(s))[tuple(sorted(z.items()))]
                       - p / pr[states.index(s)]) for s, z, p in gold)
    norm_err = max(abs(math.fsum(q for _, q in marg.conditional(s)) - 1) for s in states)
    base = np.array([marg(s) - static(s) for s in states])
    pz0 = np.array([sum(q for z, q in marg.conditional(s) if z["z0"] == 1) for s in states])
    cert.update(joint_states=len(gold), max_active_z=max(sum(lay(s).active) for s in states),
                conditional_error=float(cond_err), conditional_normalization_error=float(norm_err),
                coupling_not_constant_offset=float(base.std()), p_z0_range=float(pz0.max() - pz0.min()),
                catalogue_hash=marg.catalogue_hash)
    cert["max_error"] = float(max(cert["max_error"], cond_err, norm_err))
    cert["pass"] = bool(cert["pass"] and cert["max_error"] <= TOL and cert["coupling_not_constant_offset"] > 0.05)
    return cert, pi, P, marg, states


def gradient_check(kind: str, r, eps: float = 1e-6) -> float:
    """d Ebar / d theta (central finite difference) vs E_{P(z|r)}[phi(r, z)] (E linear in theta)."""
    lay = lambda s: layout_fn(s)  # noqa: E731
    marg = sp.MarginalEnergy(joint_energy(kind), lay, T0, {"v": 1})
    expect = sum(q * joint_features(kind, r, z) for z, q in marg.conditional(r))
    fd = []
    for i in range(len(THETA[kind])):
        d = np.zeros(len(THETA[kind]))
        d[i] = eps
        hi = sp.MarginalEnergy(joint_energy(kind, THETA[kind] + d), lay, T0, {"v": 1})(r)
        lo = sp.MarginalEnergy(joint_energy(kind, THETA[kind] - d), lay, T0, {"v": 1})(r)
        fd.append((hi - lo) / (2 * eps))
    return float(np.abs(np.array(fd) - expect).max())


def stage_config() -> dict:
    return {"stage": "G4B", "temperature": T0, "kernel_probabilities": list(sp.DEFAULT_PROBS), "kernels": list(sp.KERNELS),
            "max_active_z": sp.MAX_ACTIVE_Z, "tolerance": TOL, "chain": CHAIN,
            "spaces": {k: {"dests": {o: list(d) for o, d in v[0].items()}, "cap": v[1], "energy": v[2], "expected_states": v[3]}
                       for k, v in SPACES.items()},
            "theta": {k: v.tolist() for k, v in THETA.items()}, "unequal_probs_check": [0.4, 0.3, 0.2, 0.1],
            "energy_status": "SYNTHETIC TEST FIXTURES (not model predictions, not physical evidence)"}


def cmd_g4b(args) -> int:
    import g4b_figures
    return g4b_figures.run_stage(args, time.perf_counter(), time.process_time(), stable_hash(stage_config()))
