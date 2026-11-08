# Approach B, 3D learning kickoff: read me first

**What this is:** the first 3D learning run of Approach B, on the separate training server `user1@10.4.25.63`, carried out under a fail-closed resource guard.

**What it is not:**
- not a claim that full 3D Approach B is proven;
- not a claim that 3D X/Y is trained.

## Outcome

| Part | State | Evidence |
|---|---|---|
| Server | `GPU_TRAINING_ALLOWED_UNDER_GUARD`. 1× RTX PRO 5000 Blackwell (48.9 GB, idle at the audit). **One CPU thread: CPU 7.** sajid's GPU queue and yadu's 3D generator get priority automatically | `01_SERVER_AUDIT.md/json` |
| Data | `v4full_82926152fa9cc345`: the **complete, verified** 3D v4 Q2-parity dataset. 59,972 queries; train / val / test = 42,612 / 8,752 / 8,608; labels match the gvlab audit and 100 % witness replay; read-only | `02_3D_DATASET_AUDIT.md/json` |
| Sampler | `SAMPLER_GPU_TRAINED_ON_FULL_SNAPSHOT`. cVAE, flow matching and a ranker, 3 seeds each, plus ablations. VAL first, then the selection frozen, then TEST once | `03_SAMPLER_IMPLEMENTATION.md`, `04_SAMPLER_RESULTS.md/json`, `figures/` |
| X/Y | `XY_BLOCKED_MISSING_SEQUENCE_LABELS`. The existing finalized F2/F3/F5 model was found and ported to 3D. Exact math is verified, and a SYNTHETIC GPU proof works. **No genuine 3D X labels exist**, so no real X/Y training was run | `05_XY_MODEL_AND_LABEL_AUDIT.md` |
| Decision | Sampler: **GO** for the next stage. X/Y: **NO-GO** until 3D X labels are collected on a physics host | `06_NEXT_STAGE_DECISION.json` |
| Integrity | Tests, hashes, branch / push, backups | `07_INTEGRITY_AND_PUSH_AUDIT.md` |

## Headline numbers

TEST, M = 4 unique proposals, mean of 3 training seeds × 3 evaluation seeds.

| Sampler | hit@4 | Known-label precision @4 |
|---|---|---|
| cVAE | **0.655** | **0.801** |
| Flow matching | 0.557 | 0.756 |
| Uniform (legality-matched) | 0.451 | 0.713 |
| Band-stratified control | 0.688 | 0.758 |
| Ranker (discriminative) | 0.534 | **0.906**, AUROC 0.922 |

- 3D object context (shape, size, height) raises ranker AUROC from 0.907 to 0.931, but does not change the generators.
- About 25 % of flow matching's raw draws need repair by the projection onto legal cells.
- Metrics count recorded labels only. Untested cells are unknown, never failures.

## Reproduce on the server

```bash
cd ~/approachb_3d_learning
P="env CUDA_VISIBLE_DEVICES=0 taskset -c 7 nice -n 10 venv/bin/python"
$P src/guard.py check                                                   # must say pass
$P -m pytest -q tests                                                   # 9 tests
$P src/train.py fit --config configs/sampler_v1.json --kind cvae --seed 0 --dry-run   # cannot touch the GPU
$P src/train.py fit --config configs/sampler_v1.json --kind cvae --seed 3              # a new run (existing runs are never overwritten)
$P src/train.py evaluate --config configs/sampler_v1.json --split val
```

## Layout

| Path | What it holds |
|---|---|
| `src/` | 6 modules: `guard`, `data3d`, `models`, `train`, `figs`, `xy3d` |
| `configs/` | `guard.json`, `sampler_v1.json`, `sampler_v1_ranker_ablation.json` |
| `tests/` | `test_core.py` |
| `reports/` | these files, plus `results/` and `figures/` |

**Kept out of git (on the server):**
- the snapshot and its witness layer;
- the checkpoints, backed up on the server's second disk and on the laptop.
