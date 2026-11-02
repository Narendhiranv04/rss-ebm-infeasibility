"""Faster geometry.Checker.distances and geometry.points_to_core for the 2D dataset processes (same values, fewer numpy calls).

geometry.Checker.distances spends most of its time in numpy call overhead: ~150 calls on 2-12 element arrays per evaluation (one
points_to_core per robot body, one loop pass per core edge, a mean / hypot / max per bounding circle).  The versions here do the
robot-vs-disc part for all bodies and edges in one batch and the bounding circles in one pass.  Every value is computed with the
same elementwise operations except the two 2-vector dot products (|ab|^2 and ap.ab): geometry.py gets them from OpenBLAS, whose
kernels use fused multiply-adds that differ per CPU (laptop Zen 4 and server Intel already disagree in the last bit), here they
are plain multiply-adds.  So results agree with geometry.py to rounding (~1e-16 m), the same agreement the two machines have
with each other.  The robot bodies (geometry.bodies), the self pairs (math.hypot bound, geometry.core_distance) and the support
term are geometry.py's own code, unchanged.

    import fastgeom; fastgeom.install()      # in each worker process, before the first Checker is used
"""
from __future__ import annotations

import math

import numpy as np

import geometry as ge

VERSION = "fastgeom1"
_ORIG = {}


def _layout(names, b):
    """Per-checker index arrays: segment endpoints (a point core is a zero-length segment, a 2-point core its segment, a polygon
    its closed edge loop -- geometry._edges), segment groups per body, polygon edges for the inside test, vertex groups."""
    sizes = [len(b[n][0]) for n in names]
    ia, ib, seg_start, poly, pb = [], [], [], [], []
    v0 = 0
    for k, n in enumerate(sizes):
        seg_start.append(len(ia))
        if n <= 2:
            ia.append(v0)
            ib.append(v0 + n - 1)
        else:
            for i in range(n):
                ia.append(v0 + i)
                ib.append(v0 + (i + 1) % n)
            poly.append((k, list(range(v0, v0 + n))))
        v0 += n
    vstart = np.cumsum([0] + sizes[:-1])
    return {"sizes": sizes, "ia": np.array(ia), "ib": np.array(ib), "seg_start": np.array(seg_start), "poly": poly,
            "vstart": vstart, "vbody": np.repeat(np.arange(len(sizes)), sizes), "count": np.array(sizes, float)}


def points_to_core(points: np.ndarray, core: np.ndarray) -> np.ndarray:
    """geometry.points_to_core, all edges in one batch."""
    points, core = np.asarray(points, float), np.asarray(core, float)
    n = len(core)
    if n <= 2:
        a, b = core[:1], core[-1:]
    else:
        a, b = core, np.concatenate((core[1:], core[:1]))
    ab = b - a                                                       # (E, 2)
    ap = points[:, None, :] - a[None, :, :]                          # (N, E, 2)
    den = ab[:, 0] * ab[:, 0] + ab[:, 1] * ab[:, 1]
    num = ap[:, :, 0] * ab[:, 0] + ap[:, :, 1] * ab[:, 1]
    pos = den > 0
    t = np.where(pos, np.minimum(np.maximum(num / np.where(pos, den, 1.0), 0.0), 1.0), 0.0)
    best = np.hypot(ap[:, :, 0] - t * ab[:, 0], ap[:, :, 1] - t * ab[:, 1]).min(1) if len(points) else np.zeros(0)
    if n >= 3 and len(points):
        cross = ab[None, :, 0] * (points[:, None, 1] - core[None, :, 1]) - ab[None, :, 1] * (points[:, None, 0] - core[None, :, 0])
        best[np.all(cross >= 0, 1)] = 0.0
    return best


def distances(self, q, gap: float) -> np.ndarray:
    """geometry.Checker.distances (signed distance, or a conservative lower bound far from contact, per entry)."""
    self.n_evals += 1
    b = ge.bodies(self.robot, q, gap, self.held)
    w = self.world
    lay = self.__dict__.get("_fg")
    if lay is None:
        lay = self._fg = _layout(self.names, b)
        lay["r"] = np.array([b[n][1] for n in self.names], float)
    V = np.concatenate([b[n][0] for n in self.names]).astype(float, copy=False)
    out = []
    N = len(w.ids)
    if N:
        A, B = V[lay["ia"]], V[lay["ib"]]
        AB = B - A                                                   # (E, 2)
        P = w.xy
        AP = P[:, None, :] - A[None, :, :]                           # (N, E, 2)
        den = AB[:, 0] * AB[:, 0] + AB[:, 1] * AB[:, 1]
        num = AP[:, :, 0] * AB[:, 0] + AP[:, :, 1] * AB[:, 1]
        pos = den > 0
        t = np.where(pos, np.minimum(np.maximum(num / np.where(pos, den, 1.0), 0.0), 1.0), 0.0)
        D = np.hypot(AP[:, :, 0] - t * AB[:, 0], AP[:, :, 1] - t * AB[:, 1])
        best = np.minimum.reduceat(D, lay["seg_start"], axis=1).T.copy()    # (bodies, N)
        for k, vi in lay["poly"]:                                    # inside a polygon core: distance 0
            core = V[vi]
            e = np.concatenate((core[1:], core[:1])) - core
            cross = e[None, :, 0] * (P[:, None, 1] - core[None, :, 1]) - e[None, :, 1] * (P[:, None, 0] - core[None, :, 0])
            best[k][np.all(cross >= 0, 1)] = 0.0
        out.append(((best - w.radius) - lay["r"][:, None]).ravel())
    else:
        out.append(np.zeros(0))
    # self pairs: bounding-circle bound, or (bound <= EXACT_BELOW_M) the exact core distance -- geometry.core_distance for every
    # pair at once (inside tests, then the closest of the edge pairs: 0 for a proper crossing, else the four point-segment distances)
    sp = lay.get("self")
    if sp is None:
        sp = lay["self"] = _self_layout(self, b)
    if not sp["n"]:
        self_d = []
    else:
        Vall = V if sp["same"] else np.concatenate([b[n][0] for n in b]).astype(float, copy=False)
        cen = np.add.reduceat(Vall, sp["vstart"], axis=0) / sp["count"][:, None]
        dv = Vall - cen[sp["vbody"]]
        R = np.maximum.reduceat(np.hypot(dv[:, 0], dv[:, 1]), sp["vstart"]) + np.array([b[n][1] for n in b], float)
        dc = cen[sp["pi"]] - cen[sp["pj"]]
        bound = (np.hypot(dc[:, 0], dc[:, 1]) - R[sp["pi"]]) - R[sp["pj"]]
        near = bound <= self.EXACT_BELOW_M
        if near.any():
            a, bb, c, d = Vall[sp["ca"]], Vall[sp["cb"]], Vall[sp["cc"]], Vall[sp["cd"]]
            cr = lambda o, p, q: (p[:, 0] - o[:, 0]) * (q[:, 1] - o[:, 1]) - (p[:, 1] - o[:, 1]) * (q[:, 0] - o[:, 0])  # noqa: E731
            d1, d2, d3, d4 = cr(c, d, a), cr(c, d, bb), cr(a, bb, c), cr(a, bb, d)
            m = np.minimum(np.minimum(_pt_seg(a, c, d), _pt_seg(bb, c, d)), np.minimum(_pt_seg(c, a, bb), _pt_seg(d, a, bb)))
            m[(d1 * d2 < 0) & (d3 * d4 < 0)] = 0.0
            ex = np.minimum.reduceat(m, sp["cstart"])
            if sp["in_o"].size:
                o, nx, p = Vall[sp["in_o"]], Vall[sp["in_n"]], Vall[sp["in_p"]]
                ok = (cr(o, nx, p) >= 0).astype(np.int8)
                inside = np.minimum.reduceat(ok, sp["in_start"]).astype(bool)
                hit = np.zeros(sp["n"], bool)
                np.logical_or.at(hit, sp["in_pair"], inside)
                ex[hit] = 0.0
            ex = (ex - sp["ri"](b)) - sp["rj"](b)
            bound = np.where(near, ex, bound)
        self_d = bound
    out.append(self_d)
    if self.has_support:
        (x, y), r = b["held"][0][0], b["held"][1]
        out.append([min(x - r, w.support[0] - x - r, y - r, w.support[1] - y - r)])
    return np.concatenate(out)


def _pt_seg(p, a, b):
    """geometry._pt_seg for arrays of points / segments (rows)."""
    abx, aby, apx, apy = b[:, 0] - a[:, 0], b[:, 1] - a[:, 1], p[:, 0] - a[:, 0], p[:, 1] - a[:, 1]
    den = abx * abx + aby * aby
    pos = den != 0.0
    t = np.where(pos, np.minimum(1.0, np.maximum(0.0, (apx * abx + apy * aby) / np.where(pos, den, 1.0))), 0.0)
    return np.hypot(apx - t * abx, apy - t * aby)


def _edges_idx(v0, n):
    return [(v0, v0 + n - 1)] if n <= 2 else [(v0 + i, v0 + (i + 1) % n) for i in range(n)]


def _self_layout(ck, b):
    """Index arrays over the bodies' stacked vertices (bodies() order) for every self pair of the checker: edge-pair combos of
    geometry.core_distance, its inside tests (first vertex of one core against the other polygon's edges), circle groups."""
    names = list(b)
    sizes = [len(b[n][0]) for n in names]
    vst = np.cumsum([0] + sizes[:-1]).tolist()
    idx = {n: i for i, n in enumerate(names)}
    pi, pj, ca, cb, cc, cd, cstart, in_o, in_n, in_p, in_start, in_pair = [], [], [], [], [], [], [], [], [], [], [], []
    for k, (x, y) in enumerate(ck.pairs):
        i, j = idx[x], idx[y]
        pi.append(i)
        pj.append(j)
        cstart.append(len(ca))
        for (p, q) in _edges_idx(vst[i], sizes[i]):
            for (r, s) in _edges_idx(vst[j], sizes[j]):
                ca.append(p); cb.append(q); cc.append(r); cd.append(s)     # noqa: E702
        for pt, poly in ((vst[i], j), (vst[j], i)):                         # _inside(a[0], b) or _inside(b[0], a)
            if sizes[poly] >= 3:
                in_start.append(len(in_o))
                in_pair.append(k)
                for (o, nx) in _edges_idx(vst[poly], sizes[poly]):
                    in_o.append(o); in_n.append(nx); in_p.append(pt)        # noqa: E702
    ri_names, rj_names = [names[i] for i in pi], [names[j] for j in pj]
    A = lambda v: np.array(v, dtype=np.intp)  # noqa: E731
    return {"n": len(pi), "same": names == list(ck.names), "vstart": A(vst), "vbody": A(np.repeat(np.arange(len(sizes)), sizes)),
            "count": np.array(sizes, float), "pi": A(pi), "pj": A(pj), "ca": A(ca), "cb": A(cb), "cc": A(cc), "cd": A(cd),
            "cstart": A(cstart), "in_o": A(in_o), "in_n": A(in_n), "in_p": A(in_p), "in_start": A(in_start), "in_pair": A(in_pair),
            "ri": lambda bd: np.array([bd[n][1] for n in ri_names], float), "rj": lambda bd: np.array([bd[n][1] for n in rj_names], float)}


def install():
    if not _ORIG:
        _ORIG.update(distances=ge.Checker.distances, points_to_core=ge.points_to_core)
        ge.Checker.distances = distances
        ge.points_to_core = points_to_core


def uninstall():
    if _ORIG:
        ge.Checker.distances, ge.points_to_core = _ORIG.pop("distances"), _ORIG.pop("points_to_core")
