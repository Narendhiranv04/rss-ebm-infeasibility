# Runbook: the first K0–K3 SHELF66V3 placement collection (production v3)

**What it collects.** The 13,651 physical placement queries of the first cohort:
- 1,582 QC-eligible roots; train 9,664, val 2,066 and test 1,921 queries;
- the frozen `adaptive_v1` policy: 4 modes × 500 iterations, then 8 × 1,500, then 16 × 5,000;
- 12 single-threaded workers pinned to CPUs 0–11 on `nav@gvlab`.

**Forecast:** about 35 CPU-hours (32.8–37.8), about 3.3 h wall time expected and **5.35 h at p95 × 1.5**, with under 1 GB of output.

**Who authorizes it.** Only the user, in step 4. Nothing in the repository is authorized: the template's authorization fields are empty, and the collector refuses to start without them and the token they produce.

Run every command from your laptop terminal unless it is marked otherwise. Never use `pkill -f`, never signal another user's process, and never `sudo`.

## 1. Access

```bash
ssh gvlab                                  # nav@gvlab: every step below runs here
# the source host is reachable ONLY from the laptop, by key, and only for read-only re-audits:
#   ssh user1@10.4.25.63      (never write there, never signal the author's generator)
```

## 2. Checkout and environment (on gvlab)

```bash
cd ~/approachb3d_v3/repo && git fetch origin && git checkout naren/3d-dataset-relocations && git pull --ff-only
git log -1 --format='%H %s'                # the commit reported in the final status message
git status --short                         # must be empty (no local edits)
export SIDE=$HOME/approachb3d_v3/repo/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2
export SHELF_SRC=$HOME/approachb3d_v3/repo/gh-yadu-iiith/shelf_6x6_open_sides/dataset_bd_classquota_without_home_position_no_overpass
export PY=$HOME/approachb3d_mock/env/bin/python       # MuJoCo 3.16.0, NumPy 2.2.6 (the validated environment)
cd $SIDE
```

The committed template freezes `SHELF_SRC` as the path above. To run from a different checkout path, regenerate the template there, and check that nothing but the path changed:

```bash
$PY code/prod3d.py template --mode PRODUCTION_FIRST_COHORT --policy adaptive_v1 --workers 12 --cpus 0-11 \
  --run-dir production_v3/run_first_cohort --roster production_v3/frozen/first_cohort_roster.jsonl.gz \
  --source-pin production_v3/frozen/source_pin_first_cohort.json --split production_v3/frozen/split_manifest_first_cohort.json \
  --qc production_v3/QC_ELIGIBLE_ROOTS.json --cohort production_v3/frozen/cohort_manifest.json --shelf-src $SHELF_SRC \
  --forecast production_v3/frozen/forecast_first_cohort.json --out /tmp/regen.json
$PY -c "import json,sys;a,b=(json.load(open(f))['frozen'] for f in ('production_v3/production_policy.TEMPLATE.json','/tmp/regen.json'));[x['source'].pop('SHELF_SRC') for x in (a,b)];sys.exit(0 if a==b else 'FROZEN BLOCK DIFFERS')" && echo SAME_FROZEN_INPUTS
# then use /tmp/regen.json (copied into production_v3/) in place of the template below
```

## 3. Validate the source snapshot, the QC cohort and every gate (no teacher call)

```bash
sha256sum -c --quiet production_v3/SHA256SUMS && echo V3_FILES_OK
sha256sum -c --quiet ARTIFACT_MANIFEST.sha256 && echo V2_SIDECAR_UNCHANGED
$PY code/decide_v3.py /tmp/decision_recheck.json      # recomputes every gate from the saved evidence
# it must print "FIRST_COHORT": "FIRST_COHORT_READY_PENDING_USER_AUTHORIZATION"
$PY code/collect3d.py --policy-file production_v3/production_policy.TEMPLATE.json --dry-run | $PY -m json.tool | head -60
# a metadata-only preflight of all 13,651 queries:
#   planner_calls 0, teacher_entry_calls 0, run_dir_created false, roster_regenerated_equal true,
#   refusals_other_than_authorization [], and only the missing authorization under authorization_refusals
```

Optionally, re-audit the live source from the laptop (read-only; about 1 minute):

```bash
cd ~/Desktop/rss-3d-relocations/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2
python3 code/source_final3d.py ../dataset_bd_classquota_without_home_position_no_overpass \
  ~/Desktop/shelf66v3_live_snapshots/final_state_$(date -u +%Y%m%dT%H%M%SZ) /tmp/final_state_recheck.json
# "pass": true means: the published bytes are still identical on the host and the final K0-K3 selection is unchanged
```

## 4. Authorization (the user only; Claude must not do this)

```bash
cp production_v3/production_policy.TEMPLATE.json production_v3/production_policy.json
$PY - <<'EOF'
import json; p = 'production_v3/production_policy.json'; d = json.load(open(p))
d['authorization'] = {'authorized_by': 'YOUR NAME',
                      'authorized_utc': 'YYYY-MM-DDTHH:MM:SSZ',
                      'statement': 'I authorize the SHELF66V3 first-cohort placement collection (13,651 queries, adaptive_v1, 12 workers on CPUs 0-11).'}
d['status'] = 'AUTHORIZED BY THE USER'
json.dump(d, open(p, 'w'), indent=1, sort_keys=True)
EOF
TOKEN=$($PY code/prod3d.py token production_v3/production_policy.json); echo "$TOKEN"   # keep it with your record
$PY code/prod3d.py check production_v3/production_policy.json "$TOKEN"               # must print "ok": true
```

The token is the SHA-256 of every frozen hash, the implemented schedule, `SHELF_SRC`, the teacher protocol **and your authorization text**. Any later change to any of them invalidates it.

The check refuses if any of these hold:
- CPUs 0–11 are more than 25% busy;
- the temperature sensor is unreadable;
- free disk is under 6 GB;
- another collector is running, or the lock is held.

## 5. Launch in tmux

```bash
tmux new-session -d -s shelf66v3_first_cohort \
  "cd $SIDE && SHELF_SRC=$SHELF_SRC PY=$PY bash production_v3/launch_first_cohort.sh --execute \
   --policy-file production_v3/production_policy.json --confirm $TOKEN 2>&1 | tee -a production_v3/launch_first_cohort.log"
```

The launcher runs these steps in order, and stops at the first refusal:
1. the metadata preflight;
2. `prod3d check`;
3. the V3 and V2 test suites (about 8 minutes);
4. `exec`s the collector (`nice 10`, `taskset 0-11`).

The collector repeats every check. It takes the exclusive kernel lock `~/.cache/shelf66v3_place/collector.lock` and then starts its 12 pinned workers.

## 6. Monitor

```bash
tail -n 1 $SIDE/production_v3/run_first_cohort/monitor.jsonl | cut -c1-400   # temperature, affinity, memory, disk, violations (every 2 s)
grep -c '"unit_done"' $SIDE/production_v3/run_first_cohort/ledger.jsonl       # finished root units, of 1,466
tail -n 2 $SIDE/production_v3/run_first_cohort/ledger.jsonl | cut -c1-300
tmux attach -t shelf66v3_first_cohort                                         # detach with Ctrl-b d
```

The monitor stops the run itself, fail-closed, when any of these happen:
- the sensor is lost;
- the package temperature reaches 95 °C (it pauses at 90 °C and resumes at 82 °C);
- available memory falls under 4 GB, or swap grows by more than 1 GB;
- free disk falls under 5 GB;
- a worker exceeds 2 GB RSS;
- any process strays outside CPUs 0–11;
- an unexpected process appears, or a worker dies.

## 7. Controlled stop

```bash
PGID=$($PY -c "import json;print(json.load(open('$HOME/.cache/shelf66v3_place/collector.lock'))['pid'])"); echo $PGID
ps -o pid,pgid,args -g $PGID | head          # check it is the collector's own group
kill -9 -- -$PGID                            # the collector leads its process group; workers also die with it (PDEATHSIG)
```

A SIGKILL at any moment is safe. Each unit is written atomically: NPZ, then JSON, then a validated, fsynced ledger line. The kernel releases the lock when the process dies.

## 8. Resume

Re-run **exactly** the step-5 command, with the same policy file and token. On resume:
- every ledger unit is re-validated (ledger ↔ JSON ↔ NPZ ↔ witness re-hash, protocol, root pin, label semantics);
- anything else is moved to `run_first_cohort/quarantine/<time>/` with a `REASON.json` and recomputed.

A different policy, roster, code or source is refused ("RUN_DIR holds a different frozen run").

## 9. After the run: audit, 100% witness replay, lineage

```bash
cd $SIDE
$PY code/collect3d.py --policy-file production_v3/production_policy.json --audit > production_v3/run_first_cohort_audit.json
#   "complete": true, missing_or_invalid [], stray_files [], labels = {1 VERIFIED, 0 NEG_*, None UNLABELED_*} only
taskset -c 0-11 $PY code/replay3d.py ours production_v3/run_first_cohort production_v3/frozen/first_cohort_roster.jsonl.gz \
  production_v3/run_first_cohort_replay.json --workers 12 --cpus 0-11
#   replayed == verified, failures [], hold_identical == replayed
$PY code/plan3d.py leak production_v3/frozen/first_cohort_roster.jsonl.gz production_v3/frozen/split_manifest_first_cohort.json
#   cross_split_layouts 0
```

## 10. Sync and push the collected data (after the audit and replay pass)

```bash
du -sh production_v3/run_first_cohort; find production_v3/run_first_cohort -size +50M     # expect about 1 GB in total and no file over 50 MB
# from the laptop:
rsync -a gvlab:approachb3d_v3/repo/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/production_v3/run_first_cohort \
  ~/Desktop/rss-3d-relocations/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/production_v3/
cd ~/Desktop/rss-3d-relocations && git checkout naren/3d-dataset-relocations && git pull --ff-only
git add gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v2/production_v3/run_first_cohort*
git commit -m "SHELF66V3 first-cohort placement collection: units, ledger, monitor, audit, replay"
git push origin naren/3d-dataset-relocations            # a normal push; never force, never merge into main
```

## What not to do

- **Don't edit inputs:** never edit the template's frozen block, the roster, the pin, the split or QC files, `selected.json`, or any file under `code/`. The guard refuses, and that refusal is correct.
- **Don't override guards:** never set `RUN3D_SENSOR_OVERRIDE` or `SHELF66V3_LOCK_FILE` for production (refused), and never raise the worker count above 12 or use CPUs outside 0–11.
- **Don't use the v2 launcher:** `mock_v2/launch_full_3d.sh` and `production_v2/` are superseded V2 artifacts: an in-progress roster with unchecked and NONE roots. Do not use them.
