"""Vertex priority for the E-M4M best-first search ('Learned Priority Function').

f(v) = P(E) * P(phi1 | E) * P(phi2 | E) * prod_{O_m in NGR} P(phi3 | E)

phi1 = fraction of the initial NGR's voxels occupied by movable objects
phi2 = number of movable objects intersecting the NGR
phi3 = (fraction of the object's voxels inside the NGR) * mass * mu

P(phi1|E) ~ Beta, P(phi2|E), P(phi3|E) ~ Exponential. The paper fits these by
MLE on self-supervised data; that pipeline is not part of the reference
release, so the fitted values hard-coded in mamo_search.cpp are used. They were
fit on PR2 NGRs, so for the xArm7 they are a prior, not a calibrated model.

Priorities are compared in log space; a vertex with phi1 = 0 gets +inf.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import stats

from mamo.ngr import NGR
from mamo.scene import Scene

# mamo_search.hpp / mamo_search.cpp createDists()
PRIOR = 0.3084358524
BETA_A, BETA_B = 0.3403101839, 45.0750087432
LAMBDA_NUM_OBJS = 1.5634743875
LAMBDA_ODATA = 32.1092604858


@dataclass
class Features:
    percent_ngr: float              # phi1
    num_objs: int                   # phi2
    odata: list[float]              # phi3 per object in the NGR

    def tiebreak_key(self) -> tuple[float, int, float]:
        return (self.percent_ngr, self.num_objs, sum(self.odata))


def features(scene: Scene, ngr: NGR) -> Features:
    total = max(1, ngr.volume)
    p, n, od = 0.0, 0, []
    for o in scene.movables:
        inside, vox = ngr.object_voxels(o)
        if inside > 0:
            n += 1
            p += inside / total
            od.append((inside / max(vox, 1)) * o.mass * o.mu)
    return Features(p, n, od)


def log_priority(f: Features) -> float:
    if f.percent_ngr == 0:
        return math.inf
    x = min(max(f.percent_ngr, 1e-12), 1 - 1e-12)
    lp = math.log(PRIOR)
    lp += stats.beta.logpdf(x, BETA_A, BETA_B)
    lp += stats.expon.logpdf(f.num_objs, scale=1 / LAMBDA_NUM_OBJS)
    for d in f.odata:
        lp += stats.expon.logpdf(d, scale=1 / LAMBDA_ODATA)
    return float(lp)
