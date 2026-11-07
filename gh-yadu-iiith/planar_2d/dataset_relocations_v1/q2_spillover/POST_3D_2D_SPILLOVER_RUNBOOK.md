# Post-3D 2D spillover: operator runbook

`Q2_2D_SPILLOVER_V1` moves part of the remaining 2D Q2 workload from the laptop to gvlab. It acts only after the 3D v4 run on gvlab has:
- finished;
- passed `collect4 --audit`;
- passed a 100% positive-witness replay;
- released gvlab's CPUs.

The queue runs from a laptop cron job. It moves through the states below, one at a time; when a check fails, it stops and does nothing further.

| State | What happens | Where it runs |
|---|---|---|
| `WAIT_3D` | `gate3d --collection-only`: all 1,466 units done, the last ledger `end` complete, no project process alive, collector lock free, CPUs 0–13 idle, coretemp < 70 °C, ≥ 8 GB RAM, ≥ 20 GB disk | gvlab, CPUs 12–13, nice 15 |
| `VERIFY_3D` | tmux `q2_post3d`: `collect4 --audit` writes `run_q2_parity_audit.json`; `replay3d ours` on CPUs 0–11 writes `run_q2_parity_replay.json`. Then the full gate also requires audit complete/validated 1466 and replayed = verified = witness associations, with 0 failures | gvlab |
| `VERIFY_2D_SERVER` | tmux `q2_canary`: `canary-prep` (16 TRAIN laptop roots from the hash-verified gvlab backup, each file checked against the laptop's sha256 manifest) → `server-run --scope CANARY` (12 workers on CPUs 0–11) → `canary-audit` (witness re-certification) → `canary-compare` (per-trial content fingerprints). The verdict is `PASS_EXACT` or `HARD_BLOCK` | gvlab |
| `DRAIN_2D` | `touch full_2d_v2/DRAIN`, the runner's documented drain: in-flight queries finish and are packed, then the runner exits. If the original run had instead stopped on a safety guard, the queue waits for `state/ACK_ORIGINAL_STOP` | laptop |
| `PARTITION` | Freezes `handoff/HANDOFF_ROOT_OWNER_MANIFEST.json`, with roots assigned as follows: <ul><li>PACKED roots → ORIGINAL;</li><li>PARTIAL roots → LAPTOP;</li><li>UNTOUCHED roots → split between GVLAB and LAPTOP in proportion to the canary throughput, within each split × class.</li></ul>The exact union is 4,620 roots, 249,965 tasks and 238,160 teacher queries. The map is never overwritten | laptop (after the drain) |
| `LAUNCH` | The handoff is copied to gvlab, then tmux `q2_spillover` runs `server-run --scope SPILLOVER`, which refuses anything outside the frozen GVLAB share. The laptop runs `laptop-run`: verified partial parts are copied, then the remainder runs with the unchanged 25-worker config | both |
| `MONITOR` | Progress and attention flags. **A stopped run is reported, never restarted automatically** | laptop |
| `RECONCILE` | rsyncs the gvlab run back, then runs `merge`, which writes a NEW directory `merged/q2_canonical` plus `2D_SPLIT_RECONCILIATION.json`, then `launch_check.audit`, which writes `MERGED_Q2_AUDIT.json` | laptop |
| `DONE` / `BLOCKED_*` | terminal until an operator acts | |

Paths below:
- `Q = ~/Desktop/rss-placement-ready-v2/gh-yadu-iiith/planar_2d/dataset_relocations_v1/q2_spillover` on the laptop host;
- `GQ = ~/approachb3d_v3/2d_q2_spillover/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/q2_spillover` on gvlab.

## Inspect

```bash
python3 $Q/spill2d.py status                       # the live QUEUE_TRIGGER_STATUS.json (state, gate verdict, progress, attention, last_error)
tail -n 20 $Q/state/tick.log                       # one line per automatic tick (every 2 min)
crontab -l | grep spill2d                          # the scheduler line (laptop host terminal, not the VS Code sandbox)
systemd-inhibit --list | grep placement2d          # sleep inhibitors: original run + queue holder
ssh gvlab 'tmux ls; tail -n 5 ~/approachb3d_v3/2d_q2_spillover/{post3d,canary,spillover}.log 2>/dev/null'
# telemetry (every 10 min): branch naren/3d-dataset-relocations-telemetry, status/heartbeat_latest.json and status/2d_q2_spillover_queue.json
```

## Pause, resume and stop

```bash
touch $Q/state/PAUSE_QUEUE         # the queue holds: no transition, no remote action (the runs themselves keep going)
rm    $Q/state/PAUSE_QUEUE         # resume the queue
# remove the scheduler entirely: crontab -e, then delete the line tagged "Q2_2D_SPILLOVER_V1 queue tick"
# laptop runs (original or remainder): touch <run>/PAUSE (hold) | rm <run>/PAUSE | touch <run>/DRAIN (finish in-flight, exit)
# gvlab spillover run: touch $GQ/../launch_ready_2d_v2_gvlab/runs/spillover_gvlab/DRAIN
```

## After a stop or a block

- **Original run safety-stopped before the hand-off:** resume it with the same command as the launch:
  `setsid bash launch_ready_2d_v2/launch_full_2d.sh --execute-full --confirm <same Q2 token>`.
  Packed roots are kept, and interrupted queries re-run identically.
- **Original run safety-stopped during `DRAIN_2D`:** inspect `full_2d_v2/run_*.json`, then `touch $Q/state/ACK_ORIGINAL_STOP` to hand off.
- **A scoped run stopped (`MONITOR` attention):** run the same command again. Finished roots are kept.
  - Laptop: `setsid nohup <venv python> $Q/spill2d.py laptop-run >> $Q/remainder_laptop.log 2>&1 &`
  - gvlab: the tmux `q2_spillover` command recorded in this runbook's LAUNCH row.
- **`BLOCKED_3D_POST_RUN_VERIFICATION` / `BLOCKED_CANARY` / `BLOCKED_MERGE`:** fix the cause first, then run `<venv python> $Q/spill2d.py retry`.
  - Failed canary attempts are archived with `.attempt_<utc>` suffixes, never deleted.
  - For a merge retry, move `merged/q2_canonical` aside first.
- **A canary `HARD_BLOCK` means:** the server does not reproduce the laptop exactly, or a witness failed re-certification. Do not hand off. The laptop original run keeps collecting untouched.

## Roll back

- **Before `DRAIN_2D`:** remove the cron line. Nothing on the laptop run has been touched.
- **After the drain, before `LAUNCH`:**
  1. `rm full_2d_v2/DRAIN`.
  2. Resume the original with the launch command. It continues the full Q2 task list alone.
  3. Keep `handoff/` as a record.
- **After `LAUNCH`:**
  1. DRAIN both scoped runs.
  2. Either keep them (their roots are disjoint and verified) and merge later, or resume everything on one host with a new versioned owner map. A map is never edited in place.
  3. The original packed roots are immutable: the merge re-checks their sha256 against the frozen snapshot.
- **Never** remove `full_2d_v2/DRAIN` after `LAUNCH`. It guarantees that a stray relaunch of the original packs nothing new.

## Rules this queue enforces

- **No double planning.** No query is planned on two hosts.
  - gvlab gets only UNTOUCHED roots.
  - Partial and live roots stay on the laptop.
  - Each scoped run refuses roots outside its frozen allowlist.
  - `merge` refuses a scoped run that packed roots it does not own.
- **Identical scientific inputs.** Trial IDs, splits, seeds, plan, policy (ADAPTIVE_FULL 8→32), budget `0cd2597b0d9141fb`, pair protocol `938b09a23a581152` and the teacher are identical on both hosts.
  - `check_frozen` on every start.
  - The staged gvlab source matches the laptop's sha256 manifest of 47,384 files.
- **gvlab CPUs.** At most 18 logical CPUs project-wide: 12 workers on CPUs 0–11, the coordinator and monitoring on CPUs 12–13. 3D and 2D never run together, because the gate requires the 3D collector, audit and replay to be gone.
- **gvlab thermal checks.** gvlab uses its own coretemp `Package id 0` sensor (profile `GVLAB_XEON5317_CORETEMP_V1`).
  - There is no AC requirement on the server.
  - The laptop keeps its own checks unchanged: k10temp Tctl, AC power, boost off.
- **Laptop load before the drain.** While the original laptop run is live, the laptop side of the queue only uses ssh, rsync and JSON.
  - Do **not** run the plan-loading tests on the laptop then.
  - Run them on gvlab:
    `SPILL2D_TEST_LAUNCH=.../launch_ready_2d_v2 SPILL2D_TEST_CODE=.../code_v2 SPILL2D_TEST_PACKED=/ssd_scratch/nav_placement2d_backup/full_2d_v2 taskset -c 12,13 nice -n 15 <gvlab venv python> -m pytest q2_spillover/test_spill2d.py`
  - Reason: a laptop-side plan load tripped the runner's swap guard once (`reports/2D_SAFETY_STOP_AND_RESUME_20261107.json`).
