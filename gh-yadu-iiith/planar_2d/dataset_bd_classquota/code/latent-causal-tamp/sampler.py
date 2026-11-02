"""G4B Boltzmann sampler over canonical ordered move-once repairs (Methodology v1.6 §9 and
v1.7 §5.4/§8; Guide v1.9 "Finite sampler experiments"; G4B contract §3-5).

State: a canonical sequence r = ((i1, k1), ..., (im, km)), 0 <= m <= cap, with distinct
object ids and each k a non-stay physical destination from that object's own domain;
() is the unique no-repair state. Objects with no destination stay as context. Static
collisions are NOT masked inside the chain: every canonical state is searchable and an
external screen judges proposals later.

pi(r) is proportional to exp(-E(r)/T) for a supplied energy callback E. Architecture A
passes its sampling energy directly; B/C pass `MarginalEnergy`, the exact bounded-z
marginal -T log sum_z exp(-E(r,z)/T). Four kernels with fixed selection probabilities
(default 1/4 each), an unavailable move being a self-transition:
  DESTINATION  Gibbs over one selected action's K_i destinations (current one included)
  SWAP         symmetric exchange of two action positions, Metropolis acceptance
  BIRTH        unused eligible object x destination x insertion index, MH ratio (N-m)K_i
  DEATH        remove one action (object back to stay), MH ratio 1/((N-m+1)K_i)
(birth/death ratios also carry p_death/p_birth). Acceptance is computed in log space;
randomness comes only from an injected numpy Generator. This module imports no planner,
verifier, teacher or target code and reads no labels.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np

import sequences as sq
from config import stable_hash

KERNELS = ("DESTINATION", "SWAP", "BIRTH", "DEATH")
DEFAULT_PROBS = (0.25, 0.25, 0.25, 0.25)
MAX_ACTIVE_Z = 8


class Unsupported(ValueError):
    """Exact enumeration requested beyond the declared bound (no silent fallback)."""


# ------------------------------------------------------------------ domain and states
@dataclass(frozen=True)
class Domain:
    """Per-object non-stay destination maps (K_i = len) and the move cap b."""
    dests: tuple            # ((object id, (d1, d2, ...)), ...) sorted by id; () = context-only object
    cap: int

    @staticmethod
    def make(dests: dict, cap: int) -> "Domain":
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
            raise ValueError(f"move cap must be an int >= 0, got {cap!r}")
        for o, ds in dests.items():
            if not isinstance(o, str):
                raise ValueError(f"object id {o!r} is not a string")
            sq.canonical([(o, d) for d in ds])                          # int (non-bool) cells
            if len(set(ds)) != len(ds):
                raise ValueError(f"{o}: destinations must be distinct")
        return Domain(tuple(sorted((o, tuple(ds)) for o, ds in dests.items())), cap)

    def options(self, obj: str) -> tuple:
        return dict(self.dests)[obj]

    @property
    def eligible(self) -> tuple:
        """Objects that can be born (K_i >= 1); N = len(eligible), not the object count."""
        return tuple(o for o, ds in self.dests if ds)


def canonical_state(domain: Domain, seq) -> tuple:
    r = sq.canonical(seq)
    ids = [o for o, _ in r]
    if len(set(ids)) != len(ids) or len(r) > domain.cap or any(o not in dict(domain.dests) or d not in domain.options(o)
                                                                 for o, d in r):
        raise ValueError(f"not a canonical state of this domain: {seq!r}")
    return r


def check_probs(probs) -> tuple:
    p = tuple(float(x) for x in probs)
    if len(p) != 4 or not all(math.isfinite(x) and x >= 0 for x in p) or abs(sum(p) - 1.0) > 1e-12:
        raise ValueError(f"kernel probabilities must be 4 finite nonnegative values summing to 1: {probs!r}")
    if (p[2] > 0) != (p[3] > 0):
        raise ValueError("birth and death must both be enabled or both disabled (reversibility)")
    return p


# ------------------------------------------------------------------ proposal rules (shared by sampling and the matrix)
def _set(r, t, d):
    return r[:t] + ((r[t][0], d),) + r[t + 1:]


def destination_groups(domain: Domain, r) -> list:
    """One group per selected action position (chosen with prob 1/m): every K_i candidate."""
    return [[_set(r, t, d) for d in domain.options(r[t][0])] for t in range(len(r))]


def swap_moves(r) -> list:
    pairs = list(itertools.combinations(range(len(r)), 2))
    out = []
    for t, u in pairs:
        s = list(r)
        s[t], s[u] = s[u], s[t]
        out.append((1.0 / len(pairs), tuple(s)))
    return out


def birth_moves(domain: Domain, r) -> list:
    used = {o for o, _ in r}
    unused = [o for o in domain.eligible if o not in used]
    if len(r) >= domain.cap or not unused:
        return []
    m = len(r)
    return [(1.0 / (len(unused) * len(domain.options(o)) * (m + 1)), r[:j] + ((o, d),) + r[j:])
            for o in unused for d in domain.options(o) for j in range(m + 1)]


def death_moves(r) -> list:
    return [(1.0 / len(r), r[:t] + r[t + 1:]) for t in range(len(r))]


def log_hastings(domain: Domain, r, r2, kernel: str, probs) -> float:
    """log Q(r|r2)/Q(r2|r) for the asymmetric kernels (0 for the symmetric swap)."""
    pb, pd = probs[2], probs[3]
    n = len(domain.eligible)
    if kernel == "BIRTH":
        m, new = len(r), ({o for o, _ in r2} - {o for o, _ in r}).pop()
        return math.log((n - m) * len(domain.options(new))) + math.log(pd / pb)
    if kernel == "DEATH":
        m, gone = len(r), ({o for o, _ in r} - {o for o, _ in r2}).pop()
        return -math.log((n - m + 1) * len(domain.options(gone))) + math.log(pb / pd)
    return 0.0


def log_accept(e_old: float, e_new: float, temperature: float, log_ratio: float) -> float:
    return min(0.0, -(e_new - e_old) / temperature + log_ratio)


def _softmax(energies, temperature):
    x = [-e / temperature for e in energies]
    top = max(x)
    w = [math.exp(v - top) for v in x]
    s = math.fsum(w)
    return [v / s for v in w]


# ------------------------------------------------------------------ exact one-step transition probabilities
def same_temperature(energy, temperature: float) -> None:
    """A marginal energy must have been built with the chain's own temperature."""
    if hasattr(energy, "T") and energy.T != float(temperature):
        raise ValueError(f"marginal built at T={energy.T} used by a chain at T={temperature}")


def transition_row(domain: Domain, r, energy, temperature: float, probs=DEFAULT_PROBS) -> dict:
    """{kernel: {r2: probability}} for one step from r (selection prob, proposal, acceptance
    and the self-transition mass of rejections and unavailable moves all included)."""
    p = check_probs(probs)
    same_temperature(energy, temperature)
    e0, out = energy(r), {k: {} for k in KERNELS}

    def add(k, s, v):
        out[k][s] = out[k].get(s, 0.0) + v
    groups = destination_groups(domain, r)
    if not groups:
        add("DESTINATION", r, p[0])
    for g in groups:
        for s, w in zip(g, _softmax([energy(s) for s in g], temperature)):
            add("DESTINATION", s, p[0] * w / len(groups))
    for k, pk, moves in (("SWAP", p[1], swap_moves(r)), ("BIRTH", p[2], birth_moves(domain, r)),
                         ("DEATH", p[3], death_moves(r))):
        if not moves or pk == 0.0:
            add(k, r, pk)
            continue
        for q, s in moves:
            a = math.exp(log_accept(e0, energy(s), temperature, log_hastings(domain, r, s, k, p)))
            add(k, s, pk * q * a)
            add(k, r, pk * q * (1.0 - a))
    return out


def transition_matrix(domain: Domain, states: list, energy, temperature: float, probs=DEFAULT_PROBS, kernels=KERNELS):
    """Dense P over `states` (supplied by an independent enumeration), summing `kernels`."""
    idx = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)))
    for s in states:
        for k, row in transition_row(domain, s, energy, temperature, probs).items():
            if k in kernels:
                for s2, v in row.items():
                    P[idx[s], idx[s2]] += v                              # KeyError = proposal left the space
    return P


# ------------------------------------------------------------------ seeded chain
def step(domain: Domain, r, energy, temperature: float, rng: np.random.Generator, probs=DEFAULT_PROBS):
    """One update; returns (next state, record of the true discrete proposal)."""
    p = check_probs(probs)
    same_temperature(energy, temperature)
    k = KERNELS[int(rng.choice(4, p=p))]
    e0 = energy(r)
    rec = {"kernel": k, "state": r, "proposal": None, "E_old": e0, "E_new": None, "log_alpha": None, "accepted": False}
    if k == "DESTINATION":
        if not r:
            return r, {**rec, "kernel": "DESTINATION(unavailable)"}
        g = destination_groups(domain, r)[int(rng.integers(len(r)))]
        es = [energy(s) for s in g]
        j = int(rng.choice(len(g), p=_softmax(es, temperature)))
        return g[j], {**rec, "proposal": g[j], "E_new": es[j], "accepted": True}
    m, used = len(r), {o for o, _ in r}
    if k == "SWAP":
        if m < 2:
            return r, {**rec, "kernel": "SWAP(unavailable)"}
        t, u = list(itertools.combinations(range(m), 2))[int(rng.integers(m * (m - 1) // 2))]
        s = list(r)
        s[t], s[u] = s[u], s[t]
        s = tuple(s)
    elif k == "BIRTH":
        unused = [o for o in domain.eligible if o not in used]
        if m >= domain.cap or not unused:
            return r, {**rec, "kernel": "BIRTH(unavailable)"}
        o = unused[int(rng.integers(len(unused)))]
        ds = domain.options(o)
        j = int(rng.integers(m + 1))
        s = r[:j] + ((o, ds[int(rng.integers(len(ds)))]),) + r[j:]
    else:
        if m == 0:
            return r, {**rec, "kernel": "DEATH(unavailable)"}
        t = int(rng.integers(m))
        s = r[:t] + r[t + 1:]
    e1 = energy(s)
    la = log_accept(e0, e1, temperature, log_hastings(domain, r, s, k, p))
    ok = bool(rng.random() < math.exp(la))                         # exactly one uniform per MH proposal
    return (s if ok else r), {**rec, "proposal": s, "E_new": e1, "log_alpha": la, "accepted": ok}


def run_chain(domain: Domain, r0, energy, temperature: float, steps: int, seed: int, probs=DEFAULT_PROBS):
    rng, r, states, records = np.random.default_rng(seed), canonical_state(domain, r0), [], []
    for _ in range(steps):
        r, rec = step(domain, r, energy, temperature, rng, probs)
        states.append(r)
        records.append(rec)
    return states, records


# ------------------------------------------------------------------ B/C exact bounded-z marginal
@dataclass(frozen=True)
class Layout:
    """Typed outcome slots for one repair: `active` slots are summed over; inactive slots keep
    one deterministic value and never reach the energy or the state count."""
    slots: tuple            # slot names, fixed order
    active: tuple           # bool per slot
    fixed: tuple            # deterministic value per slot (used only when inactive)


def z_states(layout: Layout) -> list:
    names = [n for n, a in zip(layout.slots, layout.active) if a]
    if len(names) > MAX_ACTIVE_Z:
        raise Unsupported(f"{len(names)} active outcome slots > {MAX_ACTIVE_Z}: exact enumeration NOT_RUN")
    return [dict(zip(names, bits)) for bits in itertools.product((0, 1), repeat=len(names))]


def logsumexp(xs) -> float:
    top = max(xs)
    return top + math.log(math.fsum(math.exp(x - top) for x in xs))


class MarginalEnergy:
    """r -> -T log sum_{active z} exp(-E(r,z)/T) with the SAME T as the outer chain."""

    def __init__(self, joint, layout_fn, temperature: float, catalogue: dict):
        self.joint, self.layout_fn, self.T, self.catalogue = joint, layout_fn, float(temperature), catalogue
        self.catalogue_hash = stable_hash(catalogue)

    def __call__(self, r) -> float:
        return -self.T * logsumexp([-self.joint(r, z) / self.T for z in z_states(self.layout_fn(r))])

    def conditional(self, r) -> list:
        """[(z, P(z | r))] by exact enumeration (for diagnostics after sampling r)."""
        zs = z_states(self.layout_fn(r))
        e = self(r)
        return [(z, math.exp(-(self.joint(r, z) - e) / self.T)) for z in zs]
