# RUNBOOK: SHELF66V3 placement sidecar (mock v1). Future-only; no enabled production launch

**Status:** see `LAUNCH_READINESS_DECISION.json`. The full 3D placement collection is **not authorized** and cannot be started from here:
- `launch_full_3d.DISABLED.sh` exits 2;
- `run3d.py` accepts train rows only;
- no `configs/production_policy.json` exists.

```bash
ssh gvlab; tmux attach -t approachb3d            # or: tmux new -s approachb3d
X=$HOME/approachb3d_mock/src/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v1
SRC=$X/../dataset_bd_classquota_without_home_position_no_overpass
PY=$HOME/approachb3d_mock/env/bin/python           # Python 3.10.12, NumPy 2.2.6, MuJoCo 3.16.0 (private conda prefix)
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
cd $X
```

## 1. Metadata preflight (no physics, single CPU)

```bash
$PY code/audit3d.py integrity audit/source_manifest.json.gz      # source roots / witnesses / selected / ledger unchanged (exit 0)
sha256sum -c mock_v1/configs/teacher_code.sha256                  # run3d.py differs by AMENDMENT_001 (monitor fix), by design
cat mock_v1/configs/AMENDMENT_001.json
ps -eo user,pid,pcpu,args | grep -E 'run_shelf66|bd_3dt|run_3ds66' | grep -v grep   # must be empty on gvlab
nproc; cat /proc/loadavg; df -h $X; free -g                        # >= 18 idle logical CPUs, >= 10 GB disk, >= 8 GB RAM
mock_v1/launch_full_3d.DISABLED.sh; echo "exit $?"                 # must print REFUSED, exit 2
```

## 2. Replay / re-verify the mock (bounded; train roots only)

```bash
taskset -c 0 $PY -m pytest -q -p no:cacheprovider code/test_mock3d.py                      # 15 tests
taskset -c 0 $PY code/replay3d.py ours mock_v1/runs/pilot_A mock_v1/roster.jsonl.gz /tmp/replay_A.json   # fresh-process witness replay
taskset -c 0 $PY code/replay3d.py source 24 /tmp/baseline.json                              # source witness baseline
$PY code/report3d.py analyze mock_v1 /tmp/analysis.json                                     # determinism / budgets / forecast
```

**To re-run the bounded mock:**
- Move `mock_v1/runs` aside first; runs refuse to mix with a different frozen manifest.
- Then run `mock_v1/run_mock.sh` inside tmux. It uses ≤ 18 logical CPUs: physical cores 0–11 first, then SMT siblings 12–17; CPUs 18–23 stay free.

## 3. Monitoring, pause, resume (any run3d run)

```bash
tail -n 2 RUN/monitor.jsonl     # busy CPUs (test-owned), affinity union, RSS, MemAvailable, swap, disk, coretemp package, violations
grep -c unit_done RUN/ledger.jsonl
PG=$($PY -c "import json;print([json.loads(x) for x in open('RUN/ledger.jsonl') if '\"start\"' in x][-1]['pid'])")
kill -STOP -- -$PG ; kill -CONT -- -$PG    # pause / continue THIS run's process group only (never pkill -f patterns)
kill -TERM -- -$PG                         # stop; then re-run the identical command to resume
```

**Resume semantics:**
- The commit unit is one root. A unit counts as done only with a ledger line whose SHA-256 matches its JSON.
- Unfinished units re-run from scratch with the same seeds and counted budgets, so results are identical (verified by the SIGKILL test, MOCK_REPORT §6).
- An interrupted or CPU-guarded query is **unlabeled**, never 0.

**The monitor fails closed and stops the run on:**
- test-owned CPU above the worker count or an affinity union above the worker count;
- MemAvailable < 4 GB, swap growth > 1 GB, disk < 5 GB, worker RSS > 2 GB;
- nested processes, worker death, package temperature ≥ 95 °C, or any monitor error.

Workers freeze at ≥ 90 °C and resume at ≤ 82 °C.

## 4. Design of a future guarded full launcher (NOT implemented / enabled)

A full collection requires all of the following, in order. Each step needs the user's explicit authorization:

1. **Live store.** Copy the live SHELF66V3 store from `user1@10.4.25.63:$HOME/yadu_bd/runs/.../dataset_2500_raw_shelf66v3` read-only to this host, once generation has stopped or is coordinated. Re-run `audit3d.py inventory` on the copy. Never run the collection while the source generator shares the 18-CPU allocation.
2. **Lineage split and production roster.** Re-run `plan3d.py split` and `plan3d.py roster --production` on the live selection. The rule `SHELF66V3_PLACE_ROSTER_V1` is unchanged: 10 slots per root plus static contrasts. Freeze the hashes.
3. **New collector version.** Create `run3d.py` v2 that accepts all splits, takes a `--production` roster, and refuses without `configs/production_policy.json`. It must also re-run the integrity check and `pytest` before starting. New hash, reviewed diff.
4. **Production policy.** Write `configs/production_policy.json` from `LAUNCH_READINESS_DECISION.json` (policy, workers, CPU mask, guard). Record the user's authorization in it.
5. **Launch** in tmux with `nohup`, using the same monitor limits. Resume with the identical command; follow §3 for pause and stop.
6. **After the run:** `replay3d.py ours` on every unit (0 failures required), `audit3d.py integrity` (0 changes), then `report3d.py`.
