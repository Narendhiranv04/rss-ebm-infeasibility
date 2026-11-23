#!/bin/sh
# EXPLORATORY short->long run, 3D VAL only.
#   nohup sh run_all.sh "noK3scenes no3moveplans" "0 1 2" > .../runs/explore/short_long/logs/run_all.log 2>&1 &
# <= 5 busy cores (4 pool parts or 4 teacher workers + 1 parent; training 2 threads).  Restart-safe: finished models are kept, pools resume.
E=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/explore/short_long
R=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/short_long
P3=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env3d/bin/python
PT=/Users/nav/Documents/RRC_Work/RSS_2027/.venv/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HZ_MODE=v34 ABF_DEVICE=cpu SL_THREADS=2
unset SL_SMOKE
cd "$E" || exit 1
pmset -g batt | grep -q "AC Power" || { echo "not on AC power"; exit 1; }
for s in ${2:-0}; do for v in ${1:-noK3scenes}; do
  export SL_VARIANT=$v
  O=$R/$v; L=$R/logs/$v; mkdir -p "$L"
  [ -f "$O/snapshot/data3d.jsonl" ] || $PT short_long.py prep > "$L/prep.log" 2>&1 || { echo "$v prep FAILED"; continue; }
  for a in hz xy; do
    echo "=== $v $a s$s $(date +%H:%M)"
    if [ "$a" = hz ]; then m="$O/hz/model_3d_s$s.pt"; else m="$O/xy/D_s$s/model.pt"; fi
    [ -f "$m" ] || $PT short_long.py train_$a $s > "$L/${a}_s${s}_train.log" 2>&1 || { echo "$v $a s$s train FAILED"; continue; }
    i=0; while [ $i -lt 4 ]; do $PT short_long.py pool $a $s $i/4 > "$L/${a}_s${s}_pool_p$i.log" 2>&1 & i=$((i+1)); done
    wait
    $P3 short_long.py eval $a $s 4 > "$L/${a}_s${s}_eval.log" 2>&1 || echo "$v $a s$s eval FAILED"
  done
done; done
echo "=== SHORT_LONG_DONE $(date +%H:%M)"
