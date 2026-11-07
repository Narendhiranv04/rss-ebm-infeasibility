# Runbook: the 3D Q2-parity placement collection (production v4)

**What it collects:**
- **59,972 physical placement queries** of the `Q2_3D_PARITY_V1` roster: train 42,612, val 8,752, test 8,608, over 1,466 roots of the 1,582-root QC-eligible first cohort;
- the unchanged teacher, protocol `89cbabed5d7e4847`;
- `adaptive_v1` (4 × 500 → 8 × 1,500 → 16 × 5,000 iterations);
- 12 single-threaded workers pinned to CPUs 0–11 on `nav@gvlab`, under the 18-logical-CPU ceiling.

**Forecast:**
- **≈ 15.6–17 h expected** (base, and production-mode held-out calibration);
- 22.3 h on the conservative held-out calibration;
- **33.4 h buffered**;
- 2.6 GB of output (3.9 GB buffered).

**It supersedes v3.** The v3 13,651-query roster (`production_v3/`) is obsolete; do not launch it. **A v3 authorization or token cannot start v4**: the version string, roster, manifest, forecast and code hashes all differ, and both guards refuse the other's files.

Only the user authorizes, in step 4. Never use `pkill -f`, signal another user's process or `sudo`.

## 1. Checkout and environment (gvlab)

```bash
ssh gvlab
cd ~/approachb3d_v3/repo && git fetch origin && git checkout naren/3d-dataset-relocations && git pull --ff-only
git log -1 --format='%H %s'; git status --short          # the reported commit; clean
export SIDE=$HOME/approachb3d_v3/repo/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2
export SHELF_SRC=$HOME/approachb3d_v3/repo/gh-yadu-iiith/shelf_6x6_open_sides/dataset_bd_classquota_without_home_position_no_overpass
export PY=$HOME/approachb3d_mock/env/bin/python
cd $SIDE
```

## 2. Validate (no teacher call)

```bash
sha256sum -c --quiet production_v4/SHA256SUMS && echo V4_FILES_OK
sha256sum -c --quiet ARTIFACT_MANIFEST.sha256 && echo V2_UNCHANGED
(cd production_v3 && sha256sum -c --quiet SHA256SUMS && echo V3_UNCHANGED)
$PY code/decide_v4.py /tmp/v4_decision_recheck.json           # every gate recomputed from the saved evidence
$PY code/collect4.py --policy-file production_v4/production_policy_v4.TEMPLATE.json --dry-run | $PY -m json.tool | head -40
#   regenerates all 59,972 queries from SHELF_SRC: roster_regenerated_equal true, planner_calls 0, teacher_entry_calls 0,
#   run_dir_created false, refusals_other_than_authorization []
```

## 3. Status service and backups: install once, with the user's go-ahead

The service is `code/status4.py`. It was verified with two manual cycles; see telemetry commits `5ea10d9` and `197508b` on `naren/3d-dataset-relocations-telemetry`. Installing the schedulers below was **refused by the assistant's permission policy**, as persistence on your machines, so you run these two lines yourself. Neither machine had a crontab before.

```bash
# on gvlab: snapshot (status into a local retry queue) + verified second-disk backup, every 10 minutes
(crontab -l 2>/dev/null; echo "*/10 * * * * cd $SIDE && /usr/bin/flock -n $HOME/.cache/shelf66v3_place/status.cronlock sh -c '$PY code/status4.py snapshot; $PY code/status4.py backup' >> $HOME/approachb3d_v3/telemetry/cron.log 2>&1") | crontab -
# on the laptop (host terminal, not the VS Code sandbox): relay to GitHub + off-host mirror, every 10 minutes (offset by 2 minutes)
(crontab -l 2>/dev/null; echo "2-59/10 * * * * /usr/bin/flock -n $HOME/.cache/shelf66v3_place/relay.cronlock /usr/bin/python3 $HOME/Desktop/rss-3d-relocations/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/code/status4.py relay >> $HOME/.cache/shelf66v3_place/relay.log 2>&1") | crontab -
```

**What the service does:**
- **gvlab `snapshot`:** every 600 s, writes atomically to `~/approachb3d_v3/telemetry/{latest_3d.json, queue/}`. It records state, units and queries done, labels so far (1 / 0 / None), bytes, last-hour throughput, ETA, the last monitor record (temperature, CPU, RSS, disk, memory, violations, affinity), the collector PID, errors, the run manifest and roster hashes, the checkout HEAD, and the v3 state.
- **gvlab `backup`:** copies only units whose bytes equal their ledger hashes to `/ssd_scratch/nav_shelf66v3_backup/run_q2_parity/`. It writes to a temp file, fsyncs, re-hashes and renames. That location is a second physical disk; the run is on another.
- **Laptop `relay`:** pulls the queue and adds the read-only 2D Q2 observation. It commits **small status files only** to `naren/3d-dataset-relocations-telemetry` from its own clone (`~/Desktop/rss-3d-telemetry`), under a git lock, with fast-forward pushes only. A queue item is deleted only after its push succeeds, which gives the retry queue for offline periods.
- **Laptop mirror:** the relay also mirrors the backed-up units to `~/Desktop/shelf66v3_offhost_mirror/` and re-verifies every hash.

**Backup status: `OFFHOST_BACKUP_BLOCKED`.** No durable off-host destination has been designated, and none was invented:
- the second-disk copy is on shared scratch;
- the laptop mirror only runs while the laptop is on.

To fix this, name a durable store (for example a lab NAS path, or a private bucket you control), and add an `rsync` / `rclone` line for it next to the relay. A status commit is **not** a dataset backup.

## 4. Authorization (the user only)

```bash
cp production_v4/production_policy_v4.TEMPLATE.json production_v4/production_policy_v4.json
$PY - <<'EOF'
import json; p = 'production_v4/production_policy_v4.json'; d = json.load(open(p))
d['authorization'] = {'authorized_by': 'YOUR NAME', 'authorized_utc': 'YYYY-MM-DDTHH:MM:SSZ',
                      'statement': 'I authorize the SHELF66V3 v4 Q2-parity placement collection: 59,972 queries, adaptive_v1, 12 workers on CPUs 0-11.'}
d['status'] = 'AUTHORIZED BY THE USER (v4)'
json.dump(d, open(p, 'w'), indent=1, sort_keys=True)
EOF
TOKEN=$($PY code/prod4.py token production_v4/production_policy_v4.json); echo "$TOKEN"
$PY code/prod4.py check production_v4/production_policy_v4.json "$TOKEN"          # must print "ok": true
```

## 5. Launch (tmux)

```bash
tmux new-session -d -s shelf66v3_v4 "cd $SIDE && SHELF_SRC=$SHELF_SRC PY=$PY bash production_v4/launch_v4.sh --execute \
  --policy-file production_v4/production_policy_v4.json --confirm $TOKEN 2>&1 | tee -a production_v4/launch_v4.log"
```

The launcher runs, in order:
1. the preflight (full roster regeneration, about 30 s);
2. `prod4 check`;
3. the v4 and v2 test suites (about 15 min);
4. `exec`s the collector (`nice 10`, `taskset 0-11`).

The collector takes the exclusive kernel lock shared with v3, so a v3 and a v4 collector can never run together.

## 6. Monitor

```bash
cat ~/approachb3d_v3/telemetry/latest_3d.json | $PY -m json.tool | head -60            # the 10-minute snapshot
tail -n 1 $SIDE/production_v4/run_q2_parity/monitor.jsonl | cut -c1-400               # 2-second monitor
grep -c '"unit_done"' $SIDE/production_v4/run_q2_parity/ledger.jsonl                  # finished root units of 1,466
```

GitHub: `naren/3d-dataset-relocations-telemetry/status/3d_v4_latest.json`, once the relay is installed.

The run stops itself, fail-closed, on any of these:
- the sensor is lost;
- the temperature reaches 95 °C (it pauses at 90 °C);
- less than 4 GB of memory, more than 1 GB of swap growth, less than 5 GB of disk, or a worker above 2 GB RSS;
- any process outside CPUs 0–11;
- an unexpected process, or a worker death.

## 7. Controlled stop, resume, rollback

```bash
PGID=$($PY -c "import json;print(json.load(open('$HOME/.cache/shelf66v3_place/collector.lock'))['pid'])"); ps -o pid,pgid,args -g $PGID | head
kill -9 -- -$PGID                     # safe at any time: atomic NPZ -> JSON -> validated fsynced ledger line; workers die with the coordinator
```

- **Resume:** re-run the step-5 command unchanged. Every finished unit is re-validated (ledger ↔ JSON ↔ NPZ ↔ witness hashes, protocol, root pin, label semantics). Anything else is quarantined with a `REASON.json` and recomputed.
- **Rollback:** the frozen inputs never change. To abandon a run, stop it and move `production_v4/run_q2_parity` aside. Never edit it in place.

## 8. After the run

```bash
$PY code/collect4.py --policy-file production_v4/production_policy_v4.json --audit > production_v4/run_q2_parity_audit.json   # "complete": true
taskset -c 0-11 $PY code/replay3d.py ours production_v4/run_q2_parity production_v4/frozen/q2_parity_roster_v4.jsonl.gz \
  production_v4/run_q2_parity_replay.json --workers 12 --cpus 0-11                                                           # replayed == verified
$PY code/q2roster3d.py utility production_v4/run_q2_parity production_v4/run_q2_parity_training_utility.json                 # labels by stratum
$PY code/plan3d.py leak production_v4/frozen/q2_parity_roster_v4.jsonl.gz production_v3/frozen/split_manifest_first_cohort.json
```

## 9. Data sync to GitHub (after the audit and replay pass)

The data is about 2.6 GB in units of up to a few MB. Copy it into the dataset branch in batches, never every 10 minutes:

```bash
rsync -a gvlab:approachb3d_v3/repo/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/production_v4/run_q2_parity \
  ~/Desktop/rss-3d-relocations/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/production_v4/
# then commit units/ in a few batches (e.g. by root-id prefix) and push normally to naren/3d-dataset-relocations; never force, never main
```
