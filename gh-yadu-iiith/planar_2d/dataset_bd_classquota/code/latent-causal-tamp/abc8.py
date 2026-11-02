"""ABC8_DIAG_V1: P8a comparator arms, kappa-balanced root sampler and SPARSE_AUX masks on top of ABC7_V1.

No change to any ABC7 factor or normalization. Arms (identical inputs, identical label masks):
  B-full        all factors + generic coupling (pairs + higher-order)          = ABC7 'B'
  B-structured  phi_scene / phi_omit pairs kept, generic coupling removed       = ABC7 'B-structured'
  B-none        no cross-outcome term at all (independent outcome fields)       = ABC7 'B-no-coupling'
  B-quadratic   generic pairs kept, higher-order residual removed               = ABC7 'B-quadratic'
  C-full / C-structured   same switches with the GRU order factor
  A             local unary / pair + GRU task head (edit re-scoring)
  SF            supervised NON-ENERGY factorized control: every outcome bit (z0, zS[I], zR, zO[t]) is an independent
                sigmoid of the same edit- / prefix-aware head; trained by BCE on exactly the labels B / C see.
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

import abc7_models as a7

VERSION = "ABC8_DIAG_V1"
ARMS = {"B-full": "B", "B-structured": "B-structured", "B-none": "B-no-coupling", "B-quadratic": "B-quadratic",
        "C-full": "C", "C-structured": "C-structured", "A": "A"}


class ModelSF(a7.ModelS7):
    kind = "SF"

    def omit_p(self, R, seq):
        return torch.sigmoid(self.seq(R, seq)["u"])


def build(arm: str):
    return ModelSF().double() if arm == "SF" else a7.build(ARMS[arm])


def kappa_group(r) -> str:
    y0 = next((c["y"] for c in r["cands"] if not c["seq"]), None)
    ys = [r["scene_y"].get(k) for k in ("zS[A]", "zS[B]", "zS[A+B]")]
    if y0 is None or None in ys:
        return "none"
    return str(ys[2] - ys[0] - ys[1] + y0)


class RootSampler:
    """Seeded minibatches of training roots, weights inverse to the frequency of each root's observed kappa group
    (clipped to [0.5, 4] of uniform); identical batch sequence for every arm with the same seed."""

    def __init__(self, roots, batch, seed):
        groups = [kappa_group(r) for r in roots]
        freq = {g: groups.count(g) for g in set(groups)}
        w = np.array([1.0 / freq[g] for g in groups])
        w = np.clip(w / w.mean(), 0.5, 4.0)
        self.p, self.batch, self.rng, self.n = w / w.sum(), batch, np.random.default_rng(seed), len(roots)
        self.groups = dict(zip(*np.unique(groups, return_counts=True)))

    def next(self):
        return sorted(self.rng.choice(self.n, size=min(self.batch, self.n), replace=False, p=self.p).tolist())


def _hidden(root_key, query, frac=0.5) -> bool:
    return int(hashlib.sha256(f"P8_SPARSE_AUX|{root_key}|{query}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < frac


def sparse_aux(roots, frac=0.5) -> dict:
    """Hide a hash-chosen fraction of AUXILIARY labels (scene-edit outcomes and omission outcomes that are not themselves
    catalogue candidates) on the given (training) roots, in place, consistently in every place the same query appears."""
    hid = 0
    for r in roots:
        own = {c["query"] for c in r["cands"]}
        aux = set(r["edit_q"].values()) | {q for c in r["cands"] for s, q in zip(c["slots"], c["z_q"]) if s.startswith("zO") and q not in own}
        drop = {q for q in aux if _hidden(r["key"], q, frac) and r["y"].get(q) is not None}
        for q in drop:
            r["y"][q] = None
        for k, q in r["edit_q"].items():
            if q in drop:
                r["scene_y"][f"zS[{k}]"] = None
        for c in r["cands"]:
            c["z"] = [None if q in drop else z for q, z in zip(c["z_q"], c["z"])]
        hid += len(drop)
    return {"hidden_labels": hid, "fraction": frac}
