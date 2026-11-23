#!/bin/sh
# EXPLORATORY, 3D VAL: our scorer WITHOUT the step-number input, FULL data (same TRAIN as the frozen scorer). Ours only (XYNet has no
# step input). Waits until run_all.sh's current batch has finished (a 2nd SHORT_LONG_DONE line), then seeds 0-2. Restart-safe like run_all.sh.
#   nohup sh run_nostep.sh < /dev/null > .../runs/explore/short_long/logs/run_nostep.log 2>&1 &
E=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/explore/short_long
R=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/short_long
P3=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env3d/bin/python
PT=/Users/nav/Documents/RRC_Work/RSS_2027/.venv/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HZ_MODE=v34 ABF_DEVICE=cpu SL_THREADS=2 SL_VARIANT=fulldata_nostep
unset SL_SMOKE
cd "$E" || exit 1
# (no wait: nothing else running)
pmset -g batt | grep -q "AC Power" || { echo "not on AC power"; exit 1; }
O=$R/fulldata_nostep; L=$R/logs/fulldata_nostep; mkdir -p "$L"
[ -f "$O/snapshot/data3d.jsonl" ] || $PT short_long.py prep > "$L/prep.log" 2>&1 || { echo "prep FAILED"; exit 1; }
for s in 0 1 2; do
  echo "=== nostep hz s$s $(date +%H:%M)"
  [ -f "$O/hz/model_3d_s$s.pt" ] || $PT short_long.py train_hz $s > "$L/hz_s${s}_train.log" 2>&1 || { echo "nostep s$s train FAILED"; continue; }
  i=0; while [ $i -lt 4 ]; do $PT short_long.py pool hz $s $i/4 > "$L/hz_s${s}_pool_p$i.log" 2>&1 & i=$((i+1)); done
  wait
  $P3 short_long.py eval hz $s 4 > "$L/hz_s${s}_eval.log" 2>&1 || echo "nostep s$s eval FAILED"
done
echo "=== FULLNOSTEP_DONE $(date +%H:%M)"
