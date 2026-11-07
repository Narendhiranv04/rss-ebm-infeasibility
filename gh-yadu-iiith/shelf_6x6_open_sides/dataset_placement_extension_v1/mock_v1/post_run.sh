#!/bin/bash
# Post-run validation (bounded; <= 13 concurrent single-thread processes, pinned to CPUs 0-12; never touches the source).
set -u
D=$HOME/approachb3d_mock/src/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v1
PY=$HOME/approachb3d_mock/env/bin/python; M=$D/mock_v1; V=$M/validation; mkdir -p $V
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MUJOCO_GL=egl
cd $D
echo "== $(date -u +%FT%TZ) post-run validation"
taskset -c 0 nice -n 10 $PY code/replay3d.py ours $M/runs/pilot_A $M/roster.jsonl.gz $V/replay_pilot_A.json > $V/replay_pilot_A.log 2>&1 &
taskset -c 1 nice -n 10 $PY code/replay3d.py ours $M/runs/reference $M/roster.jsonl.gz $V/replay_reference.json > $V/replay_reference.log 2>&1 &
taskset -c 2 nice -n 10 $PY code/replay3d.py source 24 $V/source_baseline.json > $V/source_baseline.log 2>&1 &
for k in 0 1 2 3 4 5 6 7; do taskset -c $((k+3)) nice -n 10 $PY code/finger_diag.py $k 8 $V/finger_diag_$k.json > $V/finger_diag_$k.log 2>&1 & done
taskset -c 11 nice -n 10 $PY code/audit3d.py integrity audit/source_manifest.json.gz > $V/integrity_after.json 2> $V/integrity_after.err; echo "integrity exit $?" >> $V/integrity_after.err
wait
taskset -c 0 nice -n 10 $PY -m pytest -q -p no:cacheprovider code/test_mock3d.py > $V/pytest_post_run.txt 2>&1; echo "pytest exit $?" >> $V/pytest_post_run.txt
echo "== $(date -u +%FT%TZ) POST_DONE"
