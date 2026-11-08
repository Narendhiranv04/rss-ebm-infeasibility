# 01: Training-server audit and resource gate (`user1@10.4.25.63`)

**Decision: `GPU_TRAINING_ALLOWED_UNDER_GUARD`.**
- One GPU and one CPU thread (CPU 7).
- Other people's work always wins.

Machine-readable version: `01_SERVER_AUDIT.json`. Raw preflight probe: `guard_check_preflight.json`.

## Hardware (measured 2026-11-08T00:31Z)

| | |
|---|---|
| CPU | Intel Core Ultra 9 285K, 24 logical CPUs, no SMT. P-cores 0–7 (6.9–7.2 GHz), E-cores 8–23 (6.5 GHz) |
| RAM / swap | 125 GB / 8 GB. 121 GB available |
| GPU | **1× NVIDIA RTX PRO 5000 Blackwell**, 48,935 MiB, driver 595.99.02, 300 W limit |
| Disks | `/` 460 GB with 117 GB free (our workspace). `/data` 7.3 TB with 3.0 TB free (not used) |
| Sensors | coretemp `Package id 0` (max 85 °C, crit 105 °C). nvidia-smi temperature, power, memory and compute apps |

## State at the audit

- **GPU:** 2 MiB used, 0 % utilization, 24 °C, 2.9 W, **no compute process**.
- **CPU:** load 0.04, every CPU ≤ 3 % busy, package 28 °C.

## Other workloads found (none touched)

| Owner | What | How we treat it |
|---|---|---|
| saket | VS Code server (node), idle | untouched |
| sajid (shared `user1` uid) | `qrun.sh bigbox_g0`, a sequential **GPU 0 job queue** with `QGUARD=1`. 199/199 jobs started; the last ended 2026-11-06 20:06; it polls every 60 s | read-only. **Our watchdog stops our training as soon as a new line appears** in `bigbox_g0.q` or `bigbox_cpu.q`, or any foreign GPU process appears |
| sajid | `qrun.sh bigbox_cpu` (CPU queue) and `offload_epona_g2.sh` (15-minute rsync) | untouched; same yield rule for the queue |
| yadu, 3D native generator | `run_3dt_server.sh` runner has **exited**. `watchdog.sh` remains: it pkills `yadu_bd/venv` python if available RAM < 16 GB or temperature > 100 °C. An inbound rsync into the generator's dataset dir was seen and finished | read-only. We never use `yadu_bd/venv`. Our RAM floor is 40 GB. Our watchdog yields if any `yadu_bd/venv` python or `run_3dt_plus.py` process starts |

## CPU allowance

**Exactly one logical CPU: CPU 7**, an idle P-core.
- Every training, evaluation and helper process, and the watchdog itself, is launched with `taskset -c 7` at nice 10–15.
- `OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=1`, `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)`.
- No DataLoader workers: the whole snapshot sits as tensors on the GPU.
- Preflight **refuses** if the process affinity is not `[7]`.
- Measured during training: CPU 7 at 100 % (our single thread); the mean of the other 23 CPUs was 0.0–0.3 %.

## Thresholds (`configs/guard.json`, measured on this host)

| Signal | Stop our training when | Why |
|---|---|---|
| GPU temperature | > 80 °C | idle is 24 °C; well below throttling |
| Our VRAM | > 8,192 MiB (and a per-process fraction set in torch) | about 17 % of 48.9 GB; our runs peak at 0.1–0.4 GB |
| GPU power | > 250 W | under the 300 W limit |
| Foreign GPU compute process | any | another user, or sajid's queue, gets the GPU |
| Pending line in sajid's queues | any | their queue would otherwise wait on us (QGUARD) |
| yadu generator processes | any | the generator has priority |
| CPU package temperature | > 85 °C | coretemp `temp1_max` |
| Available RAM | < 40 GB | yadu's watchdog triggers at 16 GB |
| Free disk on `/` | < 30 GB | |
| Mean busy of the other 23 CPUs | > 25 % | somebody started CPU work |
| load1 | > 6 | |
| Any sensor unreadable | always | fail closed |

**Watchdog mechanics:**
- It is an **independent process** (`guard.py watch`), started by every GPU command. It probes every 10 s.
- After 2 consecutive bad probes it writes `STOP_RESOURCE`, sends SIGTERM to **our trainer PID only** (SIGKILL after 60 s), and the trainer saves its state and exits `STOPPED_FOR_RESOURCE_SAFETY`.
- `is_ours()` requires `argv[0]` = our venv python and `argv[1]` = our script. A shell that merely mentions those names never qualifies (a test caught this).
- **Self-stop test: PASS, twice.** The forced violation stopped only the dummy child in 8.1–8.2 s and wrote `STOP_RESOURCE`.
- `--dry-run` sets `CUDA_VISIBLE_DEVICES=""` before torch is imported, so it cannot allocate the GPU (tested).

## Environment

- `~/approachb_3d_learning/venv`. `python3-venv` was absent, so it was created with `python3 -m venv --without-pip` plus get-pip. No sudo, no shared environment touched.
- torch **2.11.0+cu128**, numpy, matplotlib, pytest. Installed with `pip --no-cache-dir`, pinned to CPU 7 at nice 15.

## Issues found and fixed during the audit

1. The first guard probe flagged PID 229303. That is the shared **tmux server**, whose command line still names `run_3dt_server.sh`; it is not the generator. Shells and tmux are now excluded, and the actual generator work processes are matched instead.
2. Network: laptop → server ran at about 0.2 MB/s uncompressed. The unit JSON was re-sent with compression (1,466 files in 33 s). The witness NPZ files follow as a separate, hash-bound layer (see `02`). gvlab → server SSH is not configured, and it was **not** changed.
