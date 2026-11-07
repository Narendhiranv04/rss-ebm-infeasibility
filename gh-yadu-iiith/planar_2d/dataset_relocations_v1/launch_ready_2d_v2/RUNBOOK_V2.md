# RUNBOOK v2: 2D placement sidecar (launch_ready_2d_v2)

**Status on 7 Nov 2026: `NOT_READY_TO_LAUNCH`.** One gate fails: `RUNTIME_FORECAST`, Q1 = 7.62 days against the 7-day limit, including
the 1.5× buffer. No production policy is frozen, so every launch is refused. See `FINAL_MOCK_V2_REPORT.md`.

```bash
R=/home/naren/Desktop/rss-placement-ready-v2                        # detached worktree at 16d1b3b4 (sparse: planar_2d)
D=$R/gh-yadu-iiith/planar_2d/dataset_relocations_v1
L=$D/launch_ready_2d_v2; CODE=$D/code_v2; RUN=$D/full_2d_v2         # RUN is created only by a real launch
GATE=$L/audit/package/reviewed/mock_v2_gate.py                      # reviewed gate (fixes R1-R5, see audit/package/*.diff)
PY=/home/naren/Desktop/rss3d_env/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
```

## 0. Host conditions (checked by the gate and by the full-mode guard)

```bash
cat /sys/devices/system/cpu/cpufreq/boost       # must print 0. From a host terminal: echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost
df -h /home                                      # at least 10 GiB free, and at least the forecast + 1.5 GiB
cat /sys/class/power_supply/ACAD/online          # 1 (AC power)
```

The v2 pilot was validated with boost **off**. A production run with boost on would be a different power configuration and needs
a new pilot.

## 1. Unlocking the launch (your decision; nothing below runs a physical query)

The pilot evidence is complete. Only the Q1 workload is too large for the pre-registered 7-day limit (with the 1.5× buffer) at the
measured boost-off throughput. Choose one:

- **(a) Production quota Q2.** 238,160 teacher queries; 3.25 days expected, 4.87 days with the buffer. Q2 keeps all four objects,
  8 root destinations, all three context blocks, 2 objects and 4 destinations per context.
- **(b) Keep Q1 and raise the limit.** Q1 is 7.62 days with the buffer; 5.08 days expected; root-resampled p95 6.76 days.

Either choice is an explicit amendment of the pre-registered rule. It would be recorded in `configs/pilot_policy_selection.json`
(new hash), and then `$PY $GATE $R decide` is re-run on the same frozen pilot evidence. No new physical runs are needed; a new
pilot is needed only if the power configuration or teacher changes.

## 2. Checks that never call the planner

```bash
cd $CODE
$PY launch_runner.py preflight $L full           # metadata only; planner_calls must be 0; passes only once a policy is frozen
$PY launch_runner.py token $L                    # token over the plan, split, launch config and production policy hashes
$PY launch_plan.py verify $L                     # the v2 plan recomputes from the 5,000 roots
$PY launch_check.py integrity $L                 # original dataset + both smoke stores unchanged
$PY $GATE $R static                              # target-goal / protected-metadata independence, two-edit contexts, hashes
"$L/launch_full_2d.sh"                           # no arguments: must REFUSE (exit 2)
```

## 3. Bounded pilot re-runs (pilot train roots only)

```bash
$PY $GATE $R mock --execute-mock                 # ref64 + 2 adaptive 25-worker runs + serial; stops on any safety stop
$PY $GATE $R interrupt --execute-mock            # SIGKILL coordinator after >= 5 packed roots, resume, compare
$PY $GATE $R decide                              # freezes configs/production_policy.json ONLY if every gate passes
$PY $CODE/v2_report.py $L $L/audit/figures       # evidence tables, figures, GIFs
```

The mock refuses to overwrite completed runs. To repeat it, move `$L/pilot/v2_*` aside first and keep it as evidence.

## 4. Full launch: DO NOT EXECUTE until the user authorizes production (refused until §1 freezes a policy)

```bash
cd $L && ./launch_full_2d.sh --execute-full --confirm <token printed by: $PY $CODE/launch_runner.py token $L>
```

The wrapper first runs `launch_runner.py guard` in the foreground. The guard checks the token and the full metadata preflight,
including:
- boost = 0;
- `reuse_archived = false`;
- every verified witness is stored;
- the policy is frozen;
- disk headroom;
- no other full run is in progress.

Only if the guard passes does it start the coordinator with `nohup`.

## 5. Operate a running collection

```bash
$PY $CODE/launch_runner.py status $RUN                    # progress, labels by class / block / destination bin, CPU-hours
tail -n 3 $RUN/monitor.jsonl                               # Tctl, frozen, in flight, memory, swap, disk, RSS, worker CPU
touch $RUN/PAUSE   ;   rm $RUN/PAUSE                       # freeze / unfreeze all workers (results unaffected)
touch $RUN/DRAIN                                           # finish queries in flight, then exit
kill -TERM $(python3 -c "import json;print(json.load(open('$RUN/status.json'))['pid'])")   # cancel now
```

**To resume after a cancel, drain, safety stop or crash,** run `rm -f $RUN/DRAIN $RUN/PAUSE`, then re-run the §4 command. The frozen
run manifest must match. Finished queries and packed roots are kept byte-identical; unfinished queries re-run with the same seeds
and counted work.

**Safety thresholds, unchanged from v1:**
- Workers are frozen at Tctl ≥ 88 °C and resume at ≤ 80 °C.
- The run is cancelled at once on any of:
  - Tctl ≥ 95 °C;
  - sensor loss;
  - a cooling stall;
  - more than 90% of the last 10 minutes paused;
  - MemAvailable < 2 GB;
  - swap growth > 512 MB;
  - disk free < 1 GB;
  - worker RSS > 1.5 GB;
  - a process leak, a dead worker, or AC lost.

There is no hard CPU quota: affinity is not a cgroup.

## 6. After the run

```bash
$PY $CODE/launch_check.py audit $RUN             # re-certifies every stored witness; 0 failures required
$PY $CODE/launch_check.py integrity $L
```
