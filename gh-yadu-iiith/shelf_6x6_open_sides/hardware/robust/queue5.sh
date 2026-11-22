#!/bin/sh
# Queue 5 (user 2026-11-22: baselines today, then rigour tests). Waits for queues 2 and 3, then (<= 6 busy workers):
#  1. PIGINet-style joint whole-plan transformer arm, 3D VAL seed 0 (hybrid/explore/piginet.py)
E=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/explore
H=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hardware/robust
X=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore
L=$X/logs
P3=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env3d/bin/python
PT=/Users/nav/Documents/RRC_Work/RSS_2027/.venv/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HZ_MODE=v34 ABF_DEVICE=cpu HZ_THREADS=2
until grep -q QUEUE3_DONE "$X/queue3.log"; do sleep 60; done   # queue 2 (rank + msweep) moved to the second machine
cd "$E" || exit 1
echo "=== piginet 3d s0 $(date +%H:%M)"
[ -f "$X/enc/piginet/train_3d_s0.json" ] || $PT piginet.py train 3d 0 > "$L/pig_3d_train.log" 2>&1 || echo "piginet train FAILED"
i=0; while [ $i -lt 6 ]; do $PT piginet.py pool 3d 0 $i/6 > "$L/pig_3d_pool_p$i.log" 2>&1 & i=$((i+1)); done
wait
$P3 piginet.py eval 3d 0 6 > "$L/pig_3d_eval.log" 2>&1 || echo "piginet eval FAILED"
# robustness tests moved to the second machine (2026-11-22)
echo "=== QUEUE5_DONE $(date +%H:%M)"
