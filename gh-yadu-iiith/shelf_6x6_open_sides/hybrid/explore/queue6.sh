#!/bin/sh
# Queue 6 (user 2026-11-22 21:00): PIGINet-style whole-plan arm, 3D VAL seeds 1 and 2 (seed 0 done in queue 5).
# Waits for the grasp-check run to finish (its 0 mm file reaches 219 roots), then <= 6 busy workers (Mac cap 7).
E=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/explore
X=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore
L=$X/logs
P3=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env3d/bin/python
PT=/Users/nav/Documents/RRC_Work/RSS_2027/.venv/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HZ_MODE=v34 ABF_DEVICE=cpu HZ_THREADS=1
GC=$X/robust_grasp_check/run_0.0.jsonl
until [ -f "$GC" ] && [ "$(wc -l < "$GC")" -ge 219 ]; do sleep 60; done
sleep 120   # let the grasp-check report step finish
cd "$E" || exit 1
for s in 1 2; do
  echo "=== piginet 3d s$s $(date +%H:%M)"
  [ -f "$X/enc/piginet/train_3d_s$s.json" ] || $PT piginet.py train 3d $s > "$L/pig_3d_s${s}_train.log" 2>&1 || echo "piginet train s$s FAILED"
  i=0; while [ $i -lt 6 ]; do $PT piginet.py pool 3d $s $i/6 > "$L/pig_3d_s${s}_pool_p$i.log" 2>&1 & i=$((i+1)); done
  wait
  $P3 piginet.py eval 3d $s 6 > "$L/pig_3d_s${s}_eval.log" 2>&1 || echo "piginet eval s$s FAILED"
done
echo "=== QUEUE6_DONE $(date +%H:%M)"
