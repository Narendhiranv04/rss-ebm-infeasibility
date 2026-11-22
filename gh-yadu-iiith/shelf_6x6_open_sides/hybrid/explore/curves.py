"""EXPLORATORY threshold-sensitivity curves (no new compute): task / noop / abstain for BOTH arms at every eta (noop gate fixed 0.5)
and every noop gate (eta fixed 0.3), 3 seeds, VAL and TEST.  Shows whether "ours beats XYNet" depends on the chosen thresholds.
    python curves.py   -> runs/explore/curves.json  (+ printed table)"""
import json
import eta_sweep as E
import gate_sweep as G

out = {}
for dom in ("3d", "2d"):
    for split in ("val", "test"):
        priv, L, ours, full = E.load(dom, split)
        pns = {"ours": [None] * 3, "full": G.pnoop_full(dom, split)}
        rows = {"ours": ours, "full": full}
        key = f"{dom}_{split}"
        out[key] = {"eta": {}, "gate": {}}
        for eta in E.GRID:
            out[key]["eta"][str(eta)] = {a: E.rates(G.indicators(priv, L, rows[a], a, 0.5, eta, pns[a])) for a in rows}
        for g in G.GATES:
            out[key]["gate"][str(g)] = {a: E.rates(G.indicators(priv, L, rows[a], a, g, 0.3, pns[a])) for a in rows}
        wins = sum(all(v["ours"][m] >= v["full"][m] for m in ("task",)) for v in out[key]["eta"].values())
        print(key, f"ours >= XYNet on task at {wins}/{len(E.GRID)} eta values")
        for eta in ("0.3", "0.5", "0.7", "0.9"):
            v = out[key]["eta"][eta]; print("  eta", eta, "ours", v["ours"], "| xynet", v["full"])
(E.H.HZ.ABF.parent / "explore/curves.json").write_text(json.dumps(out, indent=1))
