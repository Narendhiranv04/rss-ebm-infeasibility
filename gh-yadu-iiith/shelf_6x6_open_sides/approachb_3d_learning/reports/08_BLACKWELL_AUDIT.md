# 08: BWL-0 audit of the repository, data, runs, hardware and safety (2026-11-08)

**Stage verdict: BWL-0 PASS**, with one safety defect found and fixed. Everything below was checked directly, not taken from earlier reports.

## Authority documents

| Document | Where | Identity |
|---|---|---|
| `final-fixed-methodology-1.pdf` (5 Nov 2026) | laptop, `boltzmann_lfd/` (untracked, user copy) | sha256 `b7a39e3c07db0464185d00bf81b002af086bb680b76b400b1b2a21d92af7ad1c`, 32 pages; read in full |
| `BLACKWELL_APPROACH_B_EXECUTION_CONTRACT_v1.md` and the kickoff prompt | laptop, `boltzmann_lfd/instruction-phase-md/` | read in full |

## Repository

| Item | Verified value |
|---|---|
| Worktree | `/home/naren/Desktop/rss-3d-learning` (sparse), repo `Narendhiranv04/rss-ebm-infeasibility` |
| Branch / head at start | `naren/approachb-3d-learning` at `3a82081978101d567afd197bfffce9c19a1221bb`; equal to `origin` (checked with `git ls-remote`); worktree clean |
| Server copy | `~/approachb_3d_learning` on `user1@10.4.25.63`: `src/` was byte-identical to the branch at the start (sha256 prefixes as in `07`) and is byte-identical to this commit's `src/`, `tests/` and `configs/` at the end |
| 2D reference source | `latent-causal-tamp`, commits `e113c66808c4ab4682196f1fa62ae51885b08ca3` (`approach-b-xy-v1`, P25) and `45b0fb05` (`approach-b-universal-search`) are present locally. The files used are identical in both: `bxy_core.py` `f575bc03`, `bxy_p11.py` `e5ecee3b`, `bxy_p17.py` `0dae6354`, `abc7_models.py` `27b04b48`, `abc_models.py` `7d8ca5de` |
| 3D generator source (for BWL-3) | `rss-3d-relocations` worktree, `…/dataset_bd_classquota_without_home_position_no_overpass/code/shelf66_v3/bd_3dt.py`, sha256 `62dc853c…`, last changed in `6f042be1`. Only the local copy was read; no physics machine was accessed |

## Data

| Check | Result |
|---|---|
| Snapshot | `v4full_82926152fa9cc345`, read-only (`dr-xr-xr-x`) |
| Label layer | `sha256sum -c SHA256SUMS`: **3,574 / 3,574 files OK** |
| Witness layer | 1,465 NPZ; `WITNESS_LAYER_MANIFEST.json` says every file equals the bound gvlab hash |
| Derived features | `features.npz` sha256 `4a37b2ed…` (matches `02`); `scalers.json` `92d03f54…` |
| Counts | 59,972 queries; train / val / test 42,612 / 8,752 / 8,608; 1,466 query-bearing roots of 1,582 QC roots; label counts as in `02`. Re-derived from `features.npz` in `09` |
| Split hygiene | inherited lineage split; 0 roots or layouts in more than one split (`02`); not re-split |
| Scope | the complete **K0–K3 v4 first cohort** on the 6×6 supported upright-placement shelf (36 cells). It is **not** the planned 5,000-root 3D programme and **not** unconstrained pose sampling |

## Archived runs and checkpoints

| Item | Result |
|---|---|
| 21 checkpoints (15 `SAMPLER_V1` + 6 `SAMPLER_V1_RANKER_ABLATION`) | `CHECKPOINT_SHA256SUMS` OK on the server; **21 / 21** identical on the second-disk mirror `/data/user1/approachb_3d_learning_ckpt_mirror`; **21 / 21** OK on the laptop mirror |
| `eval_val.json` / `eval_test.json` / `FROZEN_FOR_TEST.json` | server sha256 `e00b3ef2…` / `9ddabe0d…` / `57af2f47…`, identical to the copies committed in `reports/results/` |
| Reproduction | `train.py evaluate` re-run with the frozen checkpoints into **new** files `runs/SAMPLER_BWL1/repro/`. **Every proposal metric (hit, precision, recall, duplicates, diversity, strata) is identical on VAL and TEST.** The only differences are the cVAE IWAE held-out NLL (≤ 4.9e-4 nats), because `CVAE.log_prob` draws **unseeded** Monte Carlo noise. That is a reproducibility defect in the archived code; it is documented here and left unchanged (`results_bwl/REPRO_COMPARISON.json`) |
| No-overwrite | `evaluate` now refuses an existing output path and takes `--out`; archived files were never written |

## Hardware and safety

| Item | Result |
|---|---|
| Host | `server`, 24 logical CPUs (Ultra 9 285K), 125 GB RAM; `nvidia-smi`: 1× **RTX PRO 5000 Blackwell**, 48,935 MiB, 2 MiB used, 0 % at the audit |
| Other users | sajid's `idn_q_bigbox_g0` / `idn_q_bigbox_cpu` / `wma_offload_g2` queues and yadu's `yadu3d` session exist; no foreign GPU process, no pending queue line and no yadu generator process was seen at any probe |
| CPU 7 | allowed by the process affinity (`Cpus_allowed_list 0-23`); `taskset -c 7` works; only idle kernel threads on CPU 7 |
| Guard | `guard.py check`: **pass**, no violations (121.7 GB available, other CPUs 0.2 % busy) |
| Self-stop test | `guard.py selftest`: **pass** (the dummy child got SIGTERM, return code −15; only the dummy was stopped). Re-run after the fix below: pass |

### Safety defect found and fixed

**Defect.** `guard.is_ours`, the watchdog's only permission to signal a process, accepted just `train.py` and `xy3d.py`. A watchdog started for any other GPU entry point could log a violation but could not stop that process. This was found while `study.py compare` (BWL-1) was running.

**Action.**
- I stopped my own runner and the `study.py` process by verified PID. The compare step had already finished: its 29-second run was covered by 3 watchdog probes, all passing, with 0 violations.
- `OUR_SCRIPTS` now lists every GPU entry point: `train`, `xy3d`, `study`, `xy3d_tests`, `xy3d_data`, `figs`, `figs_bwl`.
- `train.preflight` now **refuses** any entry point that is not on the list.
- If the watchdog ever cannot prove ownership, it writes `STOP_REFUSED_NOT_OURS`.
- New test: `test_every_gpu_entry_point_is_stoppable_by_its_watchdog`.
- Every later GPU step ran with the fixed guard.

**Verdict: `RESOURCES_AND_WATCHDOG: PASS` (after the fix).**

## Commands (server, `~/approachb_3d_learning`)

```bash
P="env CUDA_VISIBLE_DEVICES=0 taskset -c 7 nice -n 10 venv/bin/python"
$P src/guard.py check ; $P src/guard.py selftest ; $P -m pytest -q tests           # 11 passed
(cd snapshots/v4full_82926152fa9cc345 && taskset -c 7 nice -n 15 sha256sum -c --quiet SHA256SUMS)
(cd runs/SAMPLER_V1 && sha256sum -c --quiet CHECKPOINT_SHA256SUMS)              # likewise the mirrors
$P src/train.py evaluate --config configs/sampler_v1.json --split val  --out runs/SAMPLER_BWL1/repro/eval_val_repro.json
$P src/train.py evaluate --config configs/sampler_v1.json --split test --out runs/SAMPLER_BWL1/repro/eval_test_repro.json
```

## Blockers found at BWL-0

- **No genuine 3D X labels**: see `11`.
- **No off-site checkpoint copy**: not authorized, so none was made.
- **gvlab → server SSH**: still not configured, and it was not needed.
