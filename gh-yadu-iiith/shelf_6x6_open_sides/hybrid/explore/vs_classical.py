"""EXPLORATORY 3D VAL paired comparison: ours (hazard, frozen rule, 3 seeds averaged per root) vs the team's classical planners
(E-M4M / MO-SeGMan / MS-MCTS, seed 0, 300 CPU-s, their own stored per-root outcomes, branch naren/analytical-baselines-k3-finalize).
Same endpoints (BM.correct masks) and hz_eval's world-cluster bootstrap (B 10,000, rng 20261122).
    python vs_classical.py DIR [val|test]   (DIR holds {method}/predictions_3d_results.jsonl) -> runs/explore/vs_classical_3d_val.json"""
import collections, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import hz_eval as H  # noqa: E402
import numpy as np  # noqa: E402

D, dom, MET = Path(sys.argv[1]), "3d", ("task", "noop", "abstain")
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "val"   # test: post-hoc, the TEST set was already scored once (prereg 8e18429)
priv, L = H.priv_of(dom, SPLIT), H.labels(dom)
ours = [{r["root_id"]: r for r in H.ours(dom, SPLIT, s)} for s in (0, 1, 2)]
assert any((D / m).is_dir() for m in ("em4m_paper_s0_b300", "mosegman_paper_s0_b300", "msmcts_paper_s0_b300")), f"no planner folders in {D}"
out = {"label": f"EXPLORATORY 3D {SPLIT.upper()}; classical = seed 0, 300 CPU-s; UNKNOWN dropped per run"}
for m in [m for m in ("em4m_paper_s0_b300", "mosegman_paper_s0_b300", "msmcts_paper_s0_b300") if (D / m).is_dir()]:
    cls = {r["root_id"]: r for r in map(json.loads, open(D / m / "predictions_3d_results.jsonl"))}
    assert set(priv) <= set(cls), f"{m}: {len(set(priv) - set(cls))} {SPLIT} roots missing"
    per = {k: {} for k in MET}
    for rid, e in priv.items():
        a = [H.BM.correct({"masks": e["masks"], "outcome": H.V.outcome(o[rid]["decisions"][0], L, rid)}) for o in ours]
        b = H.BM.correct({"masks": cls[rid]["masks"], "outcome": cls[rid]["outcome"]})
        for i, k in enumerate(MET):
            av = [x[i] for x in a if x[i] is not None]
            if av and b[i] is not None:
                per[k][rid] = (float(np.mean(av)), float(b[i]))
    out[m] = {k: H._effect(per[k], 10000, np.random.default_rng(20261122), H._cluster) for k in MET}
    print(m, {k: {x: out[m][k][x] for x in ("n", "ours", "full", "diff", "lb97.5_cluster")} for k in MET})
(H.HZ.ABF.parent / f"explore/vs_classical_3d_{SPLIT}.json").write_text(json.dumps(out, indent=1).replace('"full"', '"classical"'))
