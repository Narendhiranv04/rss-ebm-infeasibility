# Live collections: 3D v4 (gvlab) and 2D Q2 (laptop), with monitoring and backups

Both collections were launched on 2026-11-07 under the user's explicit authorisations. Each passed its full guard first. The records are:
- `V4_LAUNCH_RECORD.json` (this folder);
- `launch_ready_2d_v2/amendment_q2/2D_Q2_LAUNCH_DECISION.json` on branch `naren/2d-object-relocations`.

| | 3D v4 Q2-parity | 2D Q2 |
|---|---|---|
| Host / run dir | `nav@gvlab`: `…/dataset_placement_extension_v2/production_v4/run_q2_parity` | laptop host: `~/Desktop/rss-placement-ready-v2/gh-yadu-iiith/planar_2d/dataset_relocations_v1/full_2d_v2` |
| Workload | 59,972 physical queries (1,466 units), plus 4,616 static contrasts outside the teacher | 238,160 teacher queries (249,965 tasks including 11,805 static rows), 4,620 roots |
| Policy | `adaptive_v1` 4×500 → 8×1500 → 16×5000, protocol `89cbabed5d7e4847` | `ADAPTIVE_FULL` 8 → 32, `witness_store_max` 32, budget `0cd2597b`, pair protocol `938b09a2` |
| Process | tmux `shelf66v3_v4`; coordinator `collect4.py` PID 3264085, 12 workers on CPUs 0–11, nice 10 | `nohup` + `setsid`; coordinator `launch_runner.py full` PID 490644, 25 workers, boost off, AC; `systemd-inhibit` blocks sleep / idle / lid-switch |
| Started (UTC) | 15:48:32 | 15:53:40 |

## Schedulers (user level; both hosts)

```
# gvlab (crontab -l)
*/10 * * * * cd …/dataset_placement_extension_v2 && /usr/bin/flock -n ~/.cache/shelf66v3_place/status.cronlock /usr/bin/taskset -c 12-13 /usr/bin/nice -n 15 sh -c "<env python> code/status4.py backup; <env python> code/status4.py snapshot" >> ~/approachb3d_v3/telemetry/cron.log 2>&1
# laptop host (crontab -l)
2-59/10 * * * * /usr/bin/flock -n ~/.cache/shelf66v3_place/relay.cronlock    /usr/bin/env -i HOME=… PATH=/usr/bin:/bin /usr/bin/python3 …/code/status4.py relay    >> ~/.cache/shelf66v3_place/relay.log 2>&1
6-59/10 * * * * /usr/bin/flock -n ~/.cache/shelf66v3_place/transfer.cronlock /usr/bin/env -i HOME=… PATH=/usr/bin:/bin /usr/bin/python3 …/code/status4.py transfer >> ~/.cache/shelf66v3_place/transfer.log 2>&1
```

**gvlab, every 10 minutes:**
- `backup` copies every ledger-validated 3D unit to `/ssd_scratch/nav_shelf66v3_backup/run_q2_parity`. That is a second physical disk; the run is on another. Each cycle **re-hashes every destination file** and re-copies any that are missing or corrupt.
- `snapshot` then writes the status into a local retry queue.
- Both are pinned to CPUs 12–13, so they stay off the collector's CPUs.

**Laptop, `relay` (status only, fast):**
- reads the 2D runner's own `status.json`, `monitor.jsonl` and packed roots;
- pulls the gvlab queue;
- computes the health flags;
- commits small status files to `naren/3d-dataset-relocations-telemetry` from its own clone, under a git lock, with fast-forward pushes only;
- deletes queue items only after their push succeeds.

**Laptop, `transfer` (own lock):**
- sends the 2D packed roots (`records/`, `modes/`, `witnesses/`; never `parts/`) to gvlab `/ssd_scratch/nav_placement2d_backup/full_2d_v2`, then **re-hashes them on gvlab**. Mismatched files are deleted so they are re-sent.
- pulls the gvlab 3D backup into `~/Desktop/shelf66v3_offhost_mirror/run_q2_parity` and re-hashes it. A failed rsync is never reported as verified.
- rsync uses I/O timeouts.

**Health flags (`status/heartbeat_latest.json`):**
- `STALE_3D` / `STALE_2D`: no fresh snapshot, or a runner status older than 20 minutes;
- `STOPPED_3D` / `STOPPED_2D`;
- `QUARANTINE_3D`;
- `BACKUP_LAGGING_3D` / `BACKUP_LAGGING_2D`: more than 60 units behind, or more than 25 minutes since the last verified run;
- `MIRROR_FAILED_3D` / `MIRROR_LAGGING_3D`;
- `PUSH_FAILED`;
- otherwise `HEALTHY`.

Every heartbeat also states **`DURABLE_OFFSITE_NOT_CONFIGURED`**. The gvlab disk, gvlab second disk and laptop are interim copies. They do not protect against a correlated loss, and a status commit is not a backup.

**Restore proofs** (copy into a fresh temporary directory, re-hash, open):
- `restore_test_3d.json` (unit `3ds66-lap-w0000-s007`);
- `amendment_q2/restore_test_2d.json` on the 2D branch (root `2d-plus2d-srv-w0058-s005`).

## Commands

```bash
# 3D (on gvlab)
cat ~/approachb3d_v3/telemetry/latest_3d.json | python3 -m json.tool | head -60
tail -n 1 ~/approachb3d_v3/repo/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/production_v4/run_q2_parity/monitor.jsonl
kill -9 -- -$(python3 -c "import json;print(json.load(open('$HOME/.cache/shelf66v3_place/collector.lock'))['pid'])")   # controlled stop (resumable)
bash ~/approachb3d_v3/start_v4_session.sh            # resume: the same frozen policy, token and launcher; finished units are kept
# 2D (laptop host terminal)
/home/naren/Desktop/rss3d_env/venv/bin/python …/code_v2/launch_runner.py status …/full_2d_v2
touch …/full_2d_v2/PAUSE   # hold (rm to resume);   touch …/full_2d_v2/DRAIN   # finish in-flight, then exit
# resume after a stop: re-run launch_full_2d.sh --execute-full --confirm <the same Q2 token>; packed roots are never re-run
```

When each run ends, follow the post-run steps in `V4_RUNBOOK.md` §8 (3D) and in the 2D runbook (`launch_check.py audit`). The schedulers keep backing up and reporting after the session ends. Remove them with `crontab -e` once the final audited data has been copied.
