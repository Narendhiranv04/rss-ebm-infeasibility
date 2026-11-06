# RUNBOOK: full 2D placement-sidecar collection (launch_ready_2d_v1)

All commands are local (this laptop). Paths are absolute; copy them as they are.

```bash
L=/home/naren/Desktop/rss-placement-smoke/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v1
CODE=$L/../code
RUN=$L/../full_2d_v1                       # created only by the real launch
PY=/home/naren/Desktop/rss3d_env/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
cd $CODE
```

## 0. Status on 2026-11-06: NOT_READY_TO_LAUNCH, and the launch is locked

No `configs/production_policy.json` exists, because the pre-registered decision did not pass every gate (`audit/pilot_decision.json`).
The guard therefore refuses every launch, even with the current token. Before the first launch:

1. **Resolve the thermal blocker** (see `readiness_report.md` §8). Recommended: disable CPU boost for the run, from a host terminal
   (outside the VS Code flatpak):
   ```bash
   echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost        # turn on again later with: echo 1 | sudo tee ...
   ```
2. **Free disk on /home.** Q1 at 8 pairs with FULL coverage needs about 990 MB plus the 1,536 MB reserve, so about 2.6 GB free at
   decision time. With less, the rule picks a smaller quota.
3. **Re-validate at 25 workers and re-decide.** This is about 30 minutes of pilot-root work and writes the production policy only if
   every gate passes:
   ```bash
   $PY launch_runner.py pilot $L revalidate_r3 64 FULL 25 all
   $PY launch_runner.py pilot $L revalidate_r4 64 FULL 25 all
   $PY launch_check.py audit $L/pilot/revalidate_r3 && $PY launch_check.py audit $L/pilot/revalidate_r4
   $PY launch_decide.py gates $L $L/pilot/revalidate_r3 $L/pilot/revalidate_r4
   $PY launch_decide.py decide $L $L/audit/pilot/budget_b64_r1.json $L/pilot/pilot_b64_w25_r1 $L/pilot/revalidate_r3 $L/audit/pilot/gates.json
   ```
4. Run `./launch_full_2d.sh` once with no arguments and check that it refuses. I could not execute the shell wrapper here; its
   Python guard is covered by `test_launch_ready.py`.

## 1. Checks that never call the teacher

```bash
$PY launch_runner.py token $L                 # confirmation token of the CURRENT frozen plan, split, config and production policy
$PY launch_runner.py preflight $L full        # dataset, hashes, plan recompute, CPU sets, sensor, AC, memory, disk; planner_calls must be 0
$PY launch_runner.py topology $L              # the 25 worker CPUs and the 7 reserved CPUs
$PY launch_plan.py verify $L                  # the frozen query plan recomputes from the dataset
$PY launch_check.py integrity $L              # original dataset + both smoke stores + smoke code against their baseline hashes
```

Any change to `query_plan.jsonl.gz`, `split_manifest.json`, `configs/launch_config.json` or `configs/production_policy.json`
changes the token, and the launcher then refuses the old one.

## 2. Small physical checks (pilot roots only, never the full roster)

```bash
$PY launch_runner.py pilot $L check_one 8 FIRST_SUCCESS serial one       # 1 query, 1 worker (writes $L/pilot/check_one)
$PY launch_runner.py pilot $L check_25 8 FULL 25 first_per_root          # 25 queries, 25 workers
$PY launch_check.py audit $L/pilot/check_25
```

Re-run of the recorded pilot (as executed on 2026-11-06; each line writes a new folder, so pick a new name to repeat it):

```bash
$PY launch_runner.py pilot $L pilot_b64_w25_r1 64 FULL 25 all
bash $L/pilot/run_pilot_sequence.sh           # r2, serial reference, independent B=8, SIGKILL + resume
```

## 3. Full launch: DO NOT EXECUTE until the user authorizes production

```bash
cd $L
./launch_full_2d.sh --execute-full --confirm "$($PY $CODE/launch_runner.py token $L)"
```

The script first runs `launch_runner.py guard` in the foreground. That checks the token and runs the metadata-only preflight; on a
refusal it exits non-zero before any worker, run folder or teacher call exists. It then starts the coordinator with `nohup` and
prints its PID. The token is a hash of the query plan, the split manifest, the launch config and the production policy, so it exists
only once §0 has frozen that policy. Pasting it in full, rather than the `$(...)` form above, is the explicit confirmation step.

## 4. Watch

```bash
$PY $CODE/launch_runner.py status $RUN        # progress, labels by class/block/destination bin, CPU-hours, pauses, survivor-pair coverage
tail -n 3 $RUN/monitor.jsonl                  # Tctl, Tccd1/2, frozen, in flight, memory, swap growth, disk, worker RSS, worker CPU
tail -f $L/../full_2d_v1.launch.log
```

## 5. Pause, resume, drain, cancel

```bash
touch $RUN/PAUSE                              # freeze every worker (SIGSTOP) and hold dispatch; results are unaffected
rm $RUN/PAUSE                                 # resume (only once Tctl <= 80 C)
touch $RUN/DRAIN                              # finish the queries in flight, then exit cleanly
kill -TERM $(python3 -c "import json;print(json.load(open('$RUN/status.json'))['pid'])")   # cancel now
```

On a cancel or safety stop, queries in flight are recorded `completed=false, label=null` with the reason; they are never 0.

## 6. Resume after a cancel, drain, safety stop, crash or reboot

```bash
rm -f $RUN/DRAIN $RUN/PAUSE
cd $L && ./launch_full_2d.sh --execute-full --confirm "$($PY $CODE/launch_runner.py token $L)"
```

The run folder's `run_manifest.json` must match the frozen plan, policy and task list, or the runner refuses. Finished queries and
packed roots are kept byte-identical. Torn temp files and witnesses of unfinished queries are removed. Unfinished queries re-run
with the same seeds and counted work.

## 7. After the run

```bash
$PY launch_check.py audit $RUN                # verification.json (re-certifies every stored witness), summary.json, fingerprint.json
$PY launch_check.py integrity $L
$PY launch_viz.py all $L $RUN <budget.json> $RUN $RUN/figures      # figures; budget.json from launch_check.py budget on a 64-pair run
```

A cancelled or partial run is not a completed dataset. `status.json` must show `final: true` and `stop: null`, and the audit must
show 0 failures and no open roots.

## Safety envelope (configs/launch_config.json)

- **Pause and resume.** Workers are frozen at Tctl >= 88 °C and resume at <= 80 °C.
- **Immediate cancel** on any of:
  - Tctl >= 95 °C in a single sample;
  - sensor loss (4 samples);
  - cooling stall (frozen for more than 180 s);
  - frozen in more than 90% of the last 10 minutes;
  - MemAvailable < 2,048 MB;
  - swap growth > 512 MB;
  - disk free < 1,024 MB;
  - worker RSS > 1,500 MB;
  - a process leak, a dead worker, or AC power lost.
- **Per-query CPU-time guard:** 1,200 s. Frozen workers do not advance it.
- **CPU control.** Each worker is pinned to one logical CPU, runs at nice 10 with one BLAS/OpenMP thread, and dies with the
  coordinator. There is no hard CPU quota: `hard_cpu_quota=false`, because no cgroup is reachable from the flatpak sandbox.
