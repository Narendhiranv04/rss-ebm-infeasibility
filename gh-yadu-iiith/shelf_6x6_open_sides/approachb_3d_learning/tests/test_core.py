"""Targeted invariants: encoder permutation invariance, scene-edit updates, legality of every sampler, unique-proposal accounting,
split isolation, guard logic, dry run cannot see CUDA, exact X/Y math.   taskset -c 7 venv/bin/python -m pytest -q tests"""
import json, os, subprocess, sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))
import data3d as D, guard as G, models as Mo, xy3d as XY  # noqa: E402
import train as TR  # noqa: E402

OBJS = {"t0": {"shape": "cylinder", "size": 0.017, "height": 0.15}, "o00": {"shape": "box", "size": 0.014, "height": 0.17},
        "o01": {"shape": "cylinder", "size": 0.02, "height": 0.12}, "o02": {"shape": "box", "size": 0.018, "height": 0.14}}
CELLS = {"t0": (5, 3), "o00": (4, 3), "o01": (4, 4), "o02": (2, 1)}


def enc_in(cells, q="o00"):
    f, m, occ, leg = D.scene_features(OBJS, cells, q)
    return torch.tensor(f)[None], torch.tensor(m)[None], torch.tensor(occ)[None], torch.tensor(leg)[None]


def test_encoder_permutation_invariant_over_non_query_objects():
    torch.manual_seed(0)
    e = Mo.SceneEncoder().eval()
    obj, m, occ, _ = enc_in(CELLS)
    perm = torch.tensor([0, 3, 1, 2, 4, 5, 6, 7])
    assert torch.allclose(e(obj, m, occ), e(obj[:, perm], m[:, perm], occ), atol=1e-5)


def test_scene_edit_updates_features_occupancy_and_legality():
    a, b = enc_in(CELLS), enc_in({**CELLS, "o02": (0, 0)})
    assert not torch.equal(a[0], b[0]) and a[2][0, 2 * 6 + 1] == 1 and b[2][0, 2 * 6 + 1] == 0 and b[3][0, 2 * 6 + 1] and not b[3][0, 0]
    assert not a[3][0, 4 * 6 + 3]                                            # the queried object's own cell is never legal


def test_every_sampler_returns_legal_cells():
    torch.manual_seed(0)
    obj, m, occ, leg = enc_in(CELLS)
    obj, m, occ, leg = (x.repeat(5, *[1] * (x.dim() - 1)) for x in (obj, m, occ, leg))
    for k in ("cvae", "fm", "uniform", "stratified"):
        s = Mo.build(k, {})
        kw = {"src": torch.full((5,), 4 * 6 + 3)} if k == "stratified" else {}
        d, _ = s.sample(obj, m, occ, leg, 32, **kw)
        assert leg.gather(1, d).all(), k


def test_first_unique_keeps_first_M_unique_and_counts_raw():
    d = torch.tensor([[3, 3, 5, 3, 7, 9], [1, 1, 1, 1, 1, 1]])
    props, used, _ = TR.first_unique(d, 3)
    assert props[0].nonzero().flatten().tolist() == [3, 5, 7] and used[0] == 5
    assert props[1].nonzero().flatten().tolist() == [1] and used[1] == 6


def test_fm_coordinate_round_trip():
    d = torch.arange(36)
    x = Mo.FlowMatch.coord(d, jitter=False)
    rc = ((x + 1) * 3).floor().long()
    assert torch.equal(rc[:, 0] * 6 + rc[:, 1], d) and torch.allclose(x, Mo.CENTRES)


def test_split_isolation_of_training_rows():
    class Fake:
        split = torch.tensor([0, 0, 1, 2, 0]); label = torch.tensor([1, 0, 1, 1, -1])
        rows = D.Data.rows
    f = Fake()
    assert TR.train_rows(f, "cvae", "train").tolist() == [0]                 # generators: VERIFIED train rows only
    assert TR.train_rows(f, "ranker", "train").tolist() == [0, 1]           # ranker: labelled train rows; UNLABELED excluded


def test_guard_queue_pending_and_never_signals_foreign(tmp_path):
    q, s = tmp_path / "x.q", tmp_path / "x.started"
    q.write_text("# c\njob a\njob b\n"); s.write_text("job a\n")
    assert G.queue_pending(q, s) == ["job b"] and G.queue_pending(tmp_path / "none.q", s) == []
    assert not G.is_ours(1) and not G.is_ours(os.getppid())


def test_dry_run_hides_cuda_before_torch_import():
    code = "import sys; sys.argv=['train.py','--dry-run']; sys.path.insert(0, %r); import train, torch; print(torch.cuda.is_available())" % str(SRC)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip().splitlines()[-1] == "False", out.stderr


def test_xy_exact_math_and_arms():
    r = XY.selftest("cpu")
    assert r["pass"], json.dumps(r)[:2000]
