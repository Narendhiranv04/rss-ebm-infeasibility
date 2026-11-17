"""Sanity check of the container setup: re-run recorded finalized X / Y labels of a few test roots and compare.

    python scripts/parity.py N      N roots per geometry implementation (selected assignment orders + the empty endpoint)"""
import collections, json, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from b2d import dataset as D, paths as P, teacher as TT  # noqa: E402


def main(n, geom):
    TT.boot(geom)
    by = collections.defaultdict(list)
    for l in json.loads(P.XY2.read_text())["rows"]:
        by[l["root_id"]].append(l)
    roots = [r for r in D.roots("test") if D.geom_of(D.record(r)) == geom and D.masks()[r]["K"] in ("0", "1", "2", "3")]
    roots = roots[:: max(1, len(roots) // n)][:n]
    agree = tot = 0
    for rid in roots:
        rec, cache = D.record(rid), {}
        for l in by[rid]:
            if l["row_kind"] not in ("SELECTED_ASSIGNMENT_ORDER", "EMPTY_SEQUENCE_K0"):
                continue
            t0 = time.time()
            v = TT.verify(rec, l["relocations"], cache)
            ok = v["x"]["x"] == l["x"] and v["y"]["y"] == l["y"]
            agree += ok; tot += 1
            print(f"{rid:45s} K={D.masks()[rid]['K']} {str(l['relocations']):40s} rec x/y={l['x']}/{l['y']} got {v['x']['x']}/{v['y']['y']} "
                  f"{'OK' if ok else 'DIFF'} {time.time() - t0:5.1f}s", flush=True)
    print(json.dumps({"geom": geom, "queries": tot, "agree": agree}))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2] if len(sys.argv) > 2 else "fastgeom1")
