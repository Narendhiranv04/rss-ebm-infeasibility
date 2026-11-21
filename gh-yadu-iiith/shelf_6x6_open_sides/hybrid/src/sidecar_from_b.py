"""Label-source ablation: the team's X/Y sidecar schema rebuilt from the executed chained B records (raw['labels']['B']), the labels the
hazard model trains on, so Boltzmann D (xy3d.XYNet, xy3d_train.py, both unchanged) can be fit on the SAME labels as hazard.

Per B record of a TRAIN / VAL QC root with a non-empty sequence (TEST roots are never read):
    X = 1 iff every move executed (failed_step None or 'target'); X = 0 iff a move failed (int failed_step k -> failed_phase 'relocation_k').
    For X = 1 only: one Y row, y = B's y (target reached the goal in the CHAINED execution, from the achieved robot state).
    x_key 'o00>4,1.o01>2,2' (sequence order), y_key 'o00@4,1|o01@2,2|t0@5,3' (endpoint cells of all objects, sorted ids, t0 at its start cell).
Status strings (new, for this sidecar only):
    XB1_CHAINED_MOVES_EXECUTED      all moves executed in the chain; NOT the team's X1 (no return-to-ready connector, no replay)
    XB0_CHAINED_MOVE_FAILED         move failed_step failed in the chain
    YB1_CHAINED_TARGET_EXECUTED     target reached the goal after the chained moves; NOT the team's Y (teleported endpoint, planned from ready)
    YB0_CHAINED_TARGET_FAILED       target step failed after all moves executed
Dedupe: identical (root, x_key) / (root, y_key) rows are kept once; a key observed with both values is a CONFLICT and is dropped entirely
(xy3d.labels_for raises on conflicting duplicates).  Different orders reaching one endpoint share one y_key, so their Y can conflict.
No prefix X rows: the loader (xy3d_train.views) treats every x_row as an independent query and needs none; B's executed prefixes would
also lack the connector.  Empty-sequence B records (K0 roots) are skipped (an X query must be non-empty).

RESULT (2026-11-21): the team's xy3d_train GATE REFUSES this sidecar (runs/XY3D_B/smoke_s0/GATE.json): train X0Y0 = 0 (min 5), X0Y1 = 25.
X0Y0 = 0 by construction: a chained B record that fails a move never attempts the target, so X0 rows get a Y only via another order reaching
the same endpoint, which then executed.  Not bypassed; no BoltzB checkpoints exist (BoltzB / eval are unusable).  Control used instead:
outcome_b.py (hazard network, plan-outcome-only supervision).

    MPLCONFIGDIR=$TMPDIR/mpl OMP_NUM_THREADS=1 python -W ignore ../hybrid/src/sidecar_from_b.py build|selftest|eval     (from approachb_3d_learning/)
"""
import json, sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
A = HERE.parents[1] / "approachb_3d_learning"
sys.path[:0] = [str(A / "src"), str(HERE)]
import xy3d as XY, xy3d_data as XD  # noqa: E402

OUT = A / "runs/XY3D_B"
SIDECAR = OUT / "BD3DS66_XY_SIDECAR_FROM_B.json"
ORIG = A / "runs/XY3D_REAL/BD3DS66_XY_SIDECAR_V1.json"
NEW = tuple(OUT / f"smoke_s{s}" for s in (0, 1, 2))
OLD = tuple(A / f"runs/XY3D_REAL/{d}" for d in ("smoke_cpu", "smoke_cpu_s1", "smoke_cpu_s2"))


def xkey(seq):
    return ".".join(f"{o}>{c[0]},{c[1]}" for o, c in seq)


def ykey(cells0, seq):
    end = {**{k: tuple(v) for k, v in cells0.items()}, **{o: tuple(c) for o, c in seq}}
    return "|".join(f"{o}@{c[0]},{c[1]}" for o, c in sorted(end.items()))


def rows_for(rid, split, raw):
    """-> raw (undeduplicated) x_rows, y_rows of one root"""
    xs, ys = [], []
    for i, b in enumerate(raw["labels"]["B"]):
        seq, f = [(o, tuple(c)) for o, c in b.get("sequence") or []], b["failed_step"]
        if not seq:
            continue
        assert (f is None) == (b["y"] == 1) and len({o for o, _ in seq}) == len(seq) and "t0" not in dict(seq), (rid, i)
        x = int(f in (None, "target"))
        yk, item = ykey(raw["initial_cells"], seq), f"B-{rid}-{i:03d}"
        xs.append({"root_id": rid, "x_key": xkey(seq), "x": x, "status": "XB1_CHAINED_MOVES_EXECUTED" if x else "XB0_CHAINED_MOVE_FAILED",
                   "split": split, "m": len(seq), "y_key": yk, "item": item, "failed_phase": None if x else f"relocation_{int(f)}"})
        if x:
            ys.append({"root_id": rid, "y_key": yk, "y": int(b["y"]), "status": "YB1_CHAINED_TARGET_EXECUTED" if b["y"] else "YB0_CHAINED_TARGET_FAILED",
                       "split": split, "item": item})
    return xs, ys


def dedupe(rows, key, val):
    """-> (kept rows, duplicates dropped, conflicting keys dropped)"""
    seen, vals = {}, {}
    for r in rows:
        k = (r["root_id"], r[key])
        vals.setdefault(k, set()).add(r[val])
        seen.setdefault(k, r)
    bad = {k for k, v in vals.items() if len(v) > 1}
    return [r for k, r in seen.items() if k not in bad], len(rows) - len(seen), sorted(bad)


def build():
    xs, ys = [], []
    for rid, sp in XD.qc_roots().items():
        if sp in ("train", "val"):  # never the test split
            x, y = rows_for(rid, sp, XD.load(rid)); xs += x; ys += y
    xr, xdup, xbad = dedupe(xs, "x_key", "x")
    yr, ydup, ybad = dedupe(ys, "y_key", "y")
    sc = {"sidecar": "BD3DS66_XY_SIDECAR_FROM_B", "sources": "raw labels.B of TRAIN/VAL QC roots (sidecar_from_b.py; see its docstring)",
          "x_labels": len(xr), "y_labels": len(yr),
          "conflicts": {"x_keys": [list(k) for k in xbad], "y_keys": [list(k) for k in ybad]},
          "duplicates_dropped": {"x": xdup, "y": ydup},
          "x_by_split": dict(Counter(f"{r['split']}|X{r['x']}" for r in xr)), "y_by_split": dict(Counter(f"{r['split']}|Y{r['y']}" for r in yr)),
          "x_rows": xr, "y_rows": yr}
    OUT.mkdir(parents=True, exist_ok=True)
    SIDECAR.write_text(json.dumps(sc))
    print(json.dumps({k: v for k, v in sc.items() if k not in ("x_rows", "y_rows")}, indent=1))


def selftest():
    sc = json.loads(SIDECAR.read_text())
    qc, orig = XD.qc_roots(), json.loads(ORIG.read_text())
    need_x, need_y = set(orig["x_rows"][0]), set(orig["y_rows"][0])
    assert all(set(r) == need_x for r in sc["x_rows"]) and all(set(r) == need_y for r in sc["y_rows"])
    assert all(qc[r["root_id"]] == r["split"] in ("train", "val") for r in sc["x_rows"] + sc["y_rows"])
    assert len({(r["root_id"], r["x_key"]) for r in sc["x_rows"]}) == len(sc["x_rows"])
    assert len({(r["root_id"], r["y_key"]) for r in sc["y_rows"]}) == len(sc["y_rows"])
    # y_key convention: identical to the team sidecar's y_key for every x_key both sidecars contain
    ok = {(r["root_id"], r["x_key"]): r["y_key"] for r in orig["x_rows"]}
    both = [r for r in sc["x_rows"] if (r["root_id"], r["x_key"]) in ok]
    assert both and all(ok[(r["root_id"], r["x_key"])] == r["y_key"] for r in both)
    # every Y row belongs to an X=1 row of the same endpoint; X=0 rows carry a relocation failed_phase
    ends1 = {(r["root_id"], r["y_key"]) for r in sc["x_rows"] if r["x"] == 1}
    assert all((r["root_id"], r["y_key"]) in ends1 for r in sc["y_rows"])
    assert all((r["x"] == 0) == str(r["failed_phase"]).startswith("relocation_") for r in sc["x_rows"])
    # synthetic: failed move -> X0 no Y; target failure -> X1 Y0; conflict detection
    raw = {"initial_cells": {"o00": [1, 1], "t0": [5, 3]}, "labels": {"B": [
        {"sequence": [["o00", [2, 2]]], "failed_step": 0, "y": 0}, {"sequence": [["o00", [3, 3]]], "failed_step": "target", "y": 0},
        {"sequence": [], "failed_step": None, "y": 1}]}}
    x, y = rows_for("r", "train", raw)
    assert [r["x"] for r in x] == [0, 1] and [(r["y_key"], r["y"]) for r in y] == [("o00@3,3|t0@5,3", 0)] and x[0]["x_key"] == "o00>2,2"
    assert dedupe(y + [{**y[0], "y": 1}], "y_key", "y")[2] == [("r", "o00@3,3|t0@5,3")]
    print(f"selftest ok ({len(both)} x_keys shared with the team sidecar, y_keys identical)")


class BoltzB:
    """Planner scorer: Boltzmann D trained on labels.B, 3 seeds.  __call__(raw, seqs) -> [3, len(seqs)] p11 (exactly phase3.Models.p11)."""

    def __init__(self, dirs=NEW, arm="D"):
        import torch
        self.torch, self.critics = torch, []
        for d in dirs:
            ck = torch.load(Path(d) / f"{arm}_best.pt", weights_only=False)
            c = XY.XYNet(ck["arm"], 5, 14).to(torch.float32); c.load_state_dict(ck["state"]); self.critics.append(c.eval())

    def __call__(self, raw, seqs):
        torch = self.torch
        v, _, xm, _ = XD.view3d(raw, extra_seqs=[list(s) for s in seqs], use_history=False)
        b, _ = XY.collate([v], None, "cpu", torch.float32)
        with torch.no_grad():
            return torch.stack([c.predict(b)["psucc"][torch.tensor(xm)] for c in self.critics]).clamp_min(1e-9)


def ece(p, y, bins=10):
    import numpy as np
    k = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[k == i].mean() - y[k == i].mean()) * (k == i).sum() for i in range(bins) if (k == i).any()) / len(p))


def evaluate():
    """Plan-level VAL metrics on the hazard eval set (hazard.cmd_eval: unique non-empty plans of VAL roots, first B label per plan)."""
    import numpy as np
    from sklearn.metrics import roc_auc_score
    import hazard
    scorers = {"a_boltzD_sidecar": BoltzB(OLD), "b_boltzD_labelsB": BoltzB(NEW), "c_hazard": hazard.Hazard()}
    ys, ps, skipped = [], {k: [] for k in scorers}, 0
    for rid, s in XD.qc_roots().items():
        if s != "val":
            continue
        raw, seen = XD.load(rid), {}
        for b in raw["labels"]["B"]:
            seq = tuple((o, tuple(c)) for o, c in b["sequence"])
            if not seq or len({o for o, _ in seq}) < len(seq):
                skipped += 1; continue
            seen.setdefault(seq, b["y"])
        if seen:
            seqs = list(seen); ys += list(seen.values())
            for k, f in scorers.items():
                ps[k] += f(raw, seqs).mean(0).tolist()
    y = np.array(ys, float)
    out = {"n_plans": len(y), "pos": int(y.sum()), "skipped_empty_or_invalid": skipped}
    for k, p in ps.items():
        p = np.clip(np.array(p), 0, 1)
        out[k] = {"auroc": round(float(roc_auc_score(y, p)), 4), "brier": round(float(((p - y) ** 2).mean()), 4), "ece10": round(ece(p, y), 4),
                  "mean_p": round(float(p.mean()), 4)}
    (OUT / "eval_val_plans.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    import torch
    torch.set_num_threads(1)
    {"build": build, "selftest": selftest, "eval": evaluate}[sys.argv[1]]()
