"""R3 (PREPARATION ONLY): predeclared, NON-construction-guided candidate-diverse 3D ordered X/Y pilot manifest.  Nothing physical runs here.

    P="env CUDA_VISIBLE_DEVICES=0 taskset -c 7 nice -n 10 venv/bin/python"
    $P src/r3_manifest3d.py --out runs/R3/CANDIDATE_DIVERSE_3D_MANIFEST.json

Destinations come ONLY from placement samplers on the CURRENT (branch-local, hypothetically edited) scene: cvae_s0 + ranker_s0 top-2 of a
P = 2M pool ('cvae+rank'), band-stratified + ranker_s0 ('strat+rank'), and the latent-free categorical control cat_s0 ('cat'); never parking
cells, intended repairs, blocker sets or teacher outcomes.  Roots: 40 cohort roots of TRAIN / VAL only, 8 train + 2 val per K0..K3, by
sha256('R3_CD3D|'+root), disjoint from the S1 / R2 pilot roots where possible.  Per root (all predeclared, seeds fixed):
  NO_OP (Y0, X = 1 by convention) | ONE_MOVE: every movable object (<= 4) x {cvae+rank, strat+rank} top destination (+ cat) | ALT_DEST: a second
  destination per object | AB / BA: pairs among the first 3 objects with A's destination from the root scene and B's re-sampled in A's edited
  scene, kept only if both orders are legal (same endpoint, one shared Y) | OMIT: A alone and B alone with the pair's destinations ('B without A')
  | TRIPLE: A, B, C with branch re-sampling, 2 orders | NUISANCE: a far object's sampled move before AB.  Every prefix is its own stopping query
  (low-Y prefixes are kept; X has no monotone requirement).  Costs use the unit costs measured in S1 / R2 (fresh execution needs planning)."""
from __future__ import annotations

import argparse, gzip, hashlib, json, os, sys
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[k] = "1"
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--guard-config")
    ap.add_argument("--exclude-roots", default="")
    a = ap.parse_args()
    import torch
    import guard, train as TR, xy3d_data as XD
    from data3d import scene_features
    from models import build
    _, dev = TR.preflight({}, guard.load_cfg(a.guard_config), False)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    wd = TR.start_watchdog(out.parent / "r3_watch", a.guard_config)
    try:
        sc = json.loads((XD.SNAP.parent / (XD.SNAP.name + "_derived") / "scalers.json").read_text())
        mu, sd = np.array(sc["mean"], np.float32), np.array(sc["std"], np.float32)
        M = {k: TR.load_model(ROOT / f"runs/{e}/{k}_s0/best.pt", dev)[0] for k, e in (("cvae", "SAMPLER_V1"), ("ranker", "SAMPLER_V1"), ("cat", "SAMPLER_BWL1"))}
        strat = build("stratified", {})
        excl = set(filter(None, a.exclude_roots.split(",")))
        roots = {r: s for r, s in XD.qc_roots().items() if s in ("train", "val") and r not in excl}

        def propose(objs, cells, o, source, tag, k=2):
            f, m, occ, leg = scene_features({x: objs[x] for x in cells}, cells, o)
            T = lambda x, dt=torch.float32: torch.as_tensor(x[None], dtype=dt, device=dev)  # noqa: E731
            obj, mask, oc, lg = T((f - mu) / sd * m[..., None]), T(m, torch.bool), T(occ), T(leg, torch.bool)
            g = torch.Generator(device=dev).manual_seed(int(hashlib.sha256(tag.encode()).hexdigest()[:8], 16))
            with torch.no_grad():
                if source == "cat":
                    d = M["cat"].sample(obj, mask, oc, lg, 8 * k, gen=g)[0][0].tolist()
                    ranked = list(dict.fromkeys(d))
                else:
                    src = torch.as_tensor([cells[o][0] * 6 + cells[o][1]], device=dev)
                    raw = (M["cvae"].sample(obj, mask, oc, lg, 16 * k, gen=g) if source == "cvae+rank" else strat.sample(obj, mask, oc, lg, 16 * k, gen=g, src=src))[0][0].tolist()
                    pool = list(dict.fromkeys(raw))[:2 * k]
                    s = M["ranker"].logits_all(obj, mask, oc)[0]
                    ranked = sorted(pool, key=lambda c: -float(s[c]))
            return [(c // 6, c % 6) for c in ranked[:k]]

        byk = {}
        for rid, sp in roots.items():
            byk.setdefault((XD.load(rid)["verified_K"], sp), []).append(rid)
        chosen = [r for K in (0, 1, 2, 3) for sp, n in (("train", 8), ("val", 2))
                  for r in sorted(byk.get((K, sp), []), key=lambda r: hashlib.sha256(f"R3_CD3D|{r}".encode()).hexdigest())[:n]]
        rows, ny = [], {}
        for rid in chosen:
            raw = XD.load(rid)
            objs = {o["id"]: {k2: o[k2] for k2 in ("shape", "size", "height")} for o in raw["objects"]}
            c0 = {k2: tuple(v) for k2, v in raw["initial_cells"].items()}
            movers = sorted((o for o in c0 if o != "t0"), key=lambda o: hashlib.sha256(f"R3M|{rid}|{o}".encode()).hexdigest())[:4]

            def add(seq, kind, sources):
                end = dict(c0); ok = True; cur = dict(c0)
                for o, d in seq:                                              # legality of every step in its own branch scene
                    if d in set(cur.values()):
                        ok = False; break
                    cur[o] = d
                if not ok:
                    return
                key = ".".join(f"{o}>{d[0]},{d[1]}" for o, d in seq)
                yk = "|".join(f"{o}@{c[0]},{c[1]}" for o, c in sorted(cur.items()))
                for r in rows:
                    if r["root_id"] == rid and r["x_key"] == key:
                        r["tags"] = sorted(set(r["tags"]) | {kind}); return
                ny.setdefault((rid, yk), len(ny))
                rows.append({"root_id": rid, "split": roots[rid], "K_reporting_only": raw["verified_K"], "x_key": key, "m": len(seq), "tags": [kind],
                             "destination_sources": sources, "y_endpoint_key": yk, "y_endpoint_id": ny[(rid, yk)], "construction_informed": False})
            rows.append({"root_id": rid, "split": roots[rid], "K_reporting_only": raw["verified_K"], "x_key": "", "m": 0, "tags": ["NO_OP"],
                         "destination_sources": [], "y_endpoint_key": "|".join(f"{o}@{c[0]},{c[1]}" for o, c in sorted(c0.items())), "construction_informed": False})
            first = {}
            for o in movers:
                for src in ("cvae+rank", "strat+rank", "cat"):
                    ds = propose(objs, c0, o, src, f"{rid}|{o}|{src}|root")
                    for j, d in enumerate(ds):
                        add([(o, d)], "ONE_MOVE" if j == 0 else "ALT_DEST", [src])
                    first.setdefault(o, {})[src] = ds[0] if ds else None
            for i in range(min(3, len(movers))):
                for j in range(i + 1, min(3, len(movers))):
                    A, B = movers[i], movers[j]
                    for src in ("cvae+rank", "strat+rank"):
                        dA = first[A].get(src)
                        if dA is None:
                            continue
                        cA = {**c0, A: dA}
                        dB = next((d for d in propose(objs, cA, B, src, f"{rid}|{B}|{src}|after|{A}", k=4) if d != c0[A]), None)   # branch re-sample
                        if dB is None:
                            continue
                        add([(A, dA), (B, dB)], "PAIR_AB", [src, f"{src}@branch"])
                        add([(B, dB), (A, dA)], "PAIR_BA_SAME_ENDPOINT", [src, f"{src}@branch"])
                        add([(A, dA)], "OMIT_B", [src]); add([(B, dB)], "OMIT_A_B_WITHOUT_A", [f"{src}@branch"])
            if len(movers) >= 3:
                A, B, C = movers[:3]
                dA = first[A].get("cvae+rank")
                if dA:
                    cA = {**c0, A: dA}
                    dB = next((d for d in propose(objs, cA, B, "cvae+rank", f"{rid}|T|B", k=4) if d != c0[A]), None)
                    if dB:
                        cAB = {**cA, B: dB}
                        dC = next((d for d in propose(objs, cAB, C, "cvae+rank", f"{rid}|T|C", k=4) if d not in (c0[A], c0[B])), None)
                        if dC:
                            add([(A, dA), (B, dB), (C, dC)], "TRIPLE", ["cvae+rank", "cvae+rank@branch", "cvae+rank@branch"])
                            add([(C, dC), (B, dB), (A, dA)], "TRIPLE_REVERSED", ["cvae+rank@branch"] * 3)
            if len(movers) == 4 and len(movers) >= 2:
                N, A, B = movers[3], movers[0], movers[1]
                dN = first[N].get("strat+rank")
                dA, dB = first[A].get("cvae+rank"), None
                if dN and dA:
                    cN = {**c0, N: dN}
                    dB = next((d for d in propose(objs, {**cN, A: dA}, B, "cvae+rank", f"{rid}|N|B", k=4) if d not in (c0[A], c0[N])), None)
                if dN and dA and dB:
                    add([(N, dN), (A, dA), (B, dB)], "NUISANCE_PLUS_PAIR", ["strat+rank", "cvae+rank", "cvae+rank@branch"])
        x = [r for r in rows if r["m"] > 0]
        import collections
        unit = {"x_fresh_cpu_s_per_move": 10.0, "x_connector_and_replay_cpu_s": 8.0, "y_plan_cpu_s_mean": None}
        man = {"manifest": "R3_CANDIDATE_DIVERSE_3D_V1", "status": "PREPARED_NOT_EXECUTED (no physics authorized)", "roots": len(chosen),
               "roots_by_split": dict(collections.Counter(roots[r] for r in chosen)), "x_queries": len(x), "unique_y_endpoints": len({(r["root_id"], r["y_endpoint_key"]) for r in rows}),
               "by_tag": dict(collections.Counter(t for r in x for t in r["tags"])), "by_m": dict(collections.Counter(r["m"] for r in x)),
               "by_source": dict(collections.Counter(s.split("@")[0] for r in x for s in r["destination_sources"])),
               "same_endpoint_order_groups": sum(1 for v in collections.Counter((r["root_id"], r["y_endpoint_key"]) for r in x if r["m"] >= 2).values() if v >= 2),
               "cost_estimate_cpu_h_x": round(sum(r["m"] * unit["x_fresh_cpu_s_per_move"] + unit["x_connector_and_replay_cpu_s"] for r in x) / 3600, 1),
               "unit_costs_basis": "S1/R1 measured: X replay+connector+replay 10-20 s per query; fresh execution adds bd_3dt.query planning (~4.9 s/call measured in "
                                   "the generator) per move; Y planning cost from R2 (see R2 report)",
               "samplers": {"cvae": "SAMPLER_V1/cvae_s0", "ranker": "SAMPLER_V1/ranker_s0", "cat": "SAMPLER_BWL1/cat_s0", "strat": "band-stratified control"},
               "sampler_verdict_note": "offline sampler verdict INCONCLUSIVE (BWL-1); stratified+ranker and cVAE+ranker are co-equal proposal sources",
               "rows": rows}
        man["sha256_of_rows"] = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
        out.write_text(json.dumps(man, indent=1))
        print(json.dumps({k: v for k, v in man.items() if k != "rows"}, indent=1))
    finally:
        wd.terminate()


if __name__ == "__main__":
    main()
