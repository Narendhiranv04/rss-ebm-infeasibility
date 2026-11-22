#!/bin/sh
# Reviewer-requested runs (2026-11-22), after the first exploratory queue: encoder-matched arm (setnet, 3D) + ranking / top-k check (3D VAL+TEST).
E=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/explore
Q=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/after_test.log
L=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/logs
P3=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env3d/bin/python
PT=/Users/nav/Documents/RRC_Work/RSS_2027/.venv/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HZ_MODE=v34 ABF_DEVICE=cpu HZ_THREADS=6
cd "$E" || exit 1
mkdir -p "$L"
# runs alongside queue 1 (user 2026-11-22: parallelise), 4 workers
echo "=== setnet 3d $(date +%H:%M)"
if [ ! -f "/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/enc/setnet/train_3d_s0.json" ]; then
  $PT encoders.py train 3d setnet 0 > "$L/enc_setnet_3d_train.log" 2>&1 || echo "setnet train FAILED"
fi
i=0; while [ $i -lt 4 ]; do $PT encoders.py pool 3d setnet 0 $i/4 > "$L/enc_setnet_3d_pool_p$i.log" 2>&1 & i=$((i+1)); done
wait
$P3 encoders.py eval 3d setnet 0 4 > "$L/enc_setnet_3d_eval.log" 2>&1 || echo "setnet eval FAILED"
for s in val test; do
  echo "=== rank $s $(date +%H:%M)"
  i=0; while [ $i -lt 4 ]; do $P3 rank_check.py score $s $i/4 > "$L/rank_${s}_p$i.log" 2>&1 & i=$((i+1)); done
  wait
  $P3 rank_check.py teach $s 4 > "$L/rank_teach_$s.log" 2>&1 || echo "rank teach $s FAILED"
done
$P3 rank_check.py report > "$L/rank_report.log" 2>&1 || echo "rank report FAILED"
echo "=== msweep 3d (after reviewer runs, user 2026-11-22) $(date +%H:%M)"
export MSWEEP_RUN=1
i=0; while [ $i -lt 4 ]; do $P3 msweep.py score $i/4 > "$L/msweep_score_p$i.log" 2>&1 & i=$((i+1)); done
wait
$P3 msweep.py teach 4 > "$L/msweep_teach.log" 2>&1 || echo "msweep teach FAILED"
$P3 msweep.py report > "$L/msweep_report.log" 2>&1 || echo "msweep report FAILED"
echo "=== QUEUE2_DONE $(date +%H:%M)"
