# q2_spillover: post-3D 2D Q2 spillover queue (`Q2_2D_SPILLOVER_V1`)

When the 3D v4 run on gvlab is finished, audited, fully replayed and has released the machine, this queue lets gvlab take part of the remaining 2D Q2 placement workload from the laptop. Until then it only waits.

The 2D teacher, plan, policy, runner and audit in `code_v2/` are imported **unchanged**. This folder only orchestrates. It adds no new labels, seeds or task selection.

- `spill2d.py`: the queue.
  - Commands: `tick`, `status`, `retry`, `hold`, `gate3d`, `post3d`, `canary-prep`, `server-run`, `canary-audit`, `canary-compare`, `classify`, `partition`, `laptop-run`, `merge`, `stage-verify`, `snapshot-source`.
  - Usage and the state machine are in the module docstring.
- `test_spill2d.py`: 17 tests (see "Tests" below).
- `source_manifest.json`: sha256 of the 47,384 2D source files staged on gvlab. `server-run` refuses to start if any staged file differs.
- `server_launch_config.json`: gvlab execution profile `GVLAB_XEON5317_CORETEMP_V1`. Only the execution and safety sections differ from the laptop config:
  - 12 workers on CPUs 0–11, coordinator on 12–13;
  - coretemp sensor;
  - no AC requirement.
- `POST_3D_2D_SPILLOVER_RUNBOOK.md`: how to inspect, pause, resume, retry and roll back.
- Status files:
  - `QUEUE_TRIGGER_STATUS.json`: a snapshot at commit time. The live file is `state/QUEUE_TRIGGER_STATUS.json`, published every 10 min to the telemetry branch as `status/2d_q2_spillover_queue.json`.
  - `CROSS_HOST_SAFETY_AUDIT.json`, `10MIN_TELEMETRY_AND_BACKUP_STATUS.json`, `UPDATED_ETAS.json`.
- `reports/2D_SAFETY_STOP_AND_RESUME_20261107.json`: the laptop run's swap-guard stop, its cause, the resume and the mitigations.
- **Written later by the queue** (absent until their stage passes):
  - `PORTABILITY_CANARY_REPORT.json`;
  - `handoff/HANDOFF_ROOT_OWNER_MANIFEST.json` (with `allowlist_*.json`);
  - `2D_SPLIT_RECONCILIATION.json`;
  - `MERGED_Q2_AUDIT.json`;
  - the merged dataset `merged/q2_canonical/`, a new directory; the original `full_2d_v2` is never rewritten.

Only these are committed: code, configs, manifests and reports. Run data, runtime state and logs are excluded by `.gitignore`.

## Scheduling (installed and verified)

Laptop host crontab, every 2 minutes:

```
*/2 * * * * flock -n $Q/state/tick.cronlock env -i HOME=… PATH=/usr/bin:/bin nice -n 10 /usr/bin/python3 $Q/spill2d.py tick >> $Q/state/tick.log
```

- Sleep: cron cannot take a logind sleep inhibitor. The probe showed polkit refuses processes outside a session. So the user session holds one instead: `systemd-inhibit --who=placement2d-q2-queue … spill2d.py hold`. It ends itself when the queue reaches `DONE` or `BLOCKED_*` and no laptop 2D coordinator is alive.
- Telemetry: `status4.py` (3D branch) now reports four jobs:
  - laptop original;
  - laptop remainder;
  - gvlab spillover;
  - 3D v4.
- Backups, each hash-verified:
  - the remainder's packed roots → gvlab `/ssd_scratch/nav_placement2d_backup/remainder_laptop`;
  - the spillover's packed roots → gvlab `/ssd_scratch/nav_placement2d_backup/spillover_gvlab` (second disk) → laptop `~/Desktop/placement2d_spillover_mirror`.
- On gvlab, `status4.py` now runs from `~/approachb3d_v3/telemetry/bin/` (`SHELF66V3_SIDE` points at the checkout), so the live 3D worktree is never modified.

## Tests

- **Laptop:** `nice -n 19 python3 -m pytest q2_spillover/test_spill2d.py -k "not partition and not canary_prep and not merge"` runs the 14 light tests (stdlib only).
- **gvlab:** the 3 plan-loading tests, with the `SPILL2D_TEST_*` variables set as in the runbook.
- **Last run:** 14/14 light tests and 3/3 gvlab tests passed.
