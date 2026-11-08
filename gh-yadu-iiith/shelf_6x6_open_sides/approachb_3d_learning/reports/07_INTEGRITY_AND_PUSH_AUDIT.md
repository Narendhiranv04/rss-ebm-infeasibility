# 07: Integrity, tests, push and backups

## Tests

| Test | Where | Result |
|---|---|---|
| `tests/test_core.py`: 9 targeted invariants (see `03`) | server, CPU 7 | **9 / 9 pass** |
| `guard.py selftest`: bounded self-stop on a forced violation | server | **PASS** twice (8.1 s and 8.2 s; only the dummy was stopped) |
| `xy3d.py selftest`: exact evidence vs brute force, normalization, arms | server | **PASS** (error 1.8e-15; log Z = 0) |
| GPU data-path smokes for cvae / fm / ranker: finite gradients, every parameter updated, TRAIN-only rows | server, GPU 0 | **PASS** ×3 (`results/runs/smoke_*/smoke.json`) |
| `--dry-run` never sees CUDA | server | **PASS** (`cuda_visible: ""`) |
| Reproducibility: retrained ranker seed 0 | server | identical per-epoch losses and test AUROCs |

## Source hashes

sha256 prefixes. The server copy is byte-identical to this branch.

| File | sha256 prefix |
|---|---|
| `src/data3d.py` | `a0370e9a317a7526` |
| `src/figs.py` | `97ce79ea20a6e675` |
| `src/guard.py` | `c4b66988a601f1ba` |
| `src/models.py` | `1b897bb6d7e0aa95` |
| `src/train.py` | `4a0f38352ebfec7c` |
| `src/xy3d.py` | `27b333f239cff935` |
| `tests/test_core.py` | `aafc233fd32f5fe1` |
| `configs/guard.json` | `3457e17117060e01` |
| `configs/sampler_v1.json` | `500fa9b064327a87` |
| `configs/sampler_v1_ranker_ablation.json` | `fab8c1930da3a97e` |

## Immutable dataset

All of these are on the training server at `~/approachb_3d_learning/snapshots/`, and all are read-only (write tests were denied).

| Layer | Contents | Verification |
|---|---|---|
| `v4full_82926152fa9cc345/` | label layer: 1,466 unit JSON, upstream bundle, sanitized gvlab reports, `SNAPSHOT_MANIFEST.json`, `SHA256SUMS` | every unit JSON = gvlab backup-index sha; every source root = source pin; inputs = run manifest; labels = gvlab audit |
| `v4full_82926152fa9cc345_witnesses/` | 1,465 witness NPZ files | every file sha256 = the hash bound into the snapshot id from the gvlab index (`attach-witnesses`) |
| `v4full_82926152fa9cc345_derived/` | `features.npz` (sha256 `4a37b2ed…`) and `scalers.json` | built from the label layer |

**Provenance.**
- Upstream: `Narendhiranv04/rss-ebm-infeasibility` `naren/3d-dataset-relocations` at `6c72eb4be17b` (contains the v4 freeze `83c1135f`).
- The original data on gvlab and its backups were **not modified**. The snapshot is a verified copy.

**Kept out of git:** the snapshot, the witnesses and the checkpoints. Only code, configs, small JSON results and figures are committed.

## Checkpoints and backups

**21 checkpoints** (15 in `SAMPLER_V1` and 6 in `SAMPLER_V1_RANKER_ABLATION`), about 1 MB each, held in three places:
1. the server: `~/approachb_3d_learning/runs/<experiment>/<run>/best.pt`, with sha256 sums in `CHECKPOINT_SHA256SUMS` (copied to `results/runs/` and `results/ranker_ablation/`);
2. the server's **second physical disk** (`/dev/nvme1n1`, `/data/user1/approachb_3d_learning_ckpt_mirror/`, owned by `user1`): **21 / 21 verified**;
3. the laptop (a separate host): `~/Desktop/approachb_3d_learning_ckpt_mirror/`: **21 / 21 verified**.

**Gap:** there is no durable off-site, institutional or cloud copy. This is reported as a backup gap, not papered over.

## Git

- Branch `naren/approachb-3d-learning`, created from `6c72eb4be17b` in a separate sparse worktree, `/home/naren/Desktop/rss-3d-learning`. The worktree used by the live status cron was untouched.
- Only `gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/` was added.
- No force push. No collector branch, production file or frozen artefact was modified. Nothing was merged into main.
- No secrets: the policy's authorization and token fields were stripped before the policy left the laptop, and no token appears in any committed file.
- Kickoff commit `cfcdb335134a7e074653f3a5cdc62baa8128f3c5`, pushed as a new branch; `git ls-remote` confirmed the remote head equals it.

## What did not change

- **gvlab:** still running only the 2D Q2 spillover; no training was moved there.
- **The laptop 2D remainder run:** not touched. The laptop did only rsync relays and light JSON reads; swap growth stayed negative throughout.
- **The source generator and its directories:** read-only.
- **Other users' processes** on the server: never signalled.
