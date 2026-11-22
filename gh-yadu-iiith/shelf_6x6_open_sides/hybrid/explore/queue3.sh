#!/bin/sh
# Queue 3 (user 2026-11-22 "queue more"): seed variance for the two reviewer-confound arms (nophase, setnet), seeds 1 and 2, 3D VAL.
# Starts when queue 1 prints EXPLORE_DONE (keeps the Mac at <= 7 busy workers). Restart-safe like queue 1.
E=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/explore
Q=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/after_test.log
R=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/enc
L=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/logs
P3=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env3d/bin/python
PT=/Users/nav/Documents/RRC_Work/RSS_2027/.venv/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HZ_MODE=v34 ABF_DEVICE=cpu HZ_THREADS=4
cd "$E" || exit 1
until grep -q EXPLORE_DONE "$Q"; do sleep 60; done
for s in 1 2; do for k in nophase setnet; do
  echo "=== $k 3d s$s $(date +%H:%M)"
  if [ ! -f "$R/$k/train_3d_s$s.json" ]; then
    $PT encoders.py train 3d $k $s > "$L/enc_${k}_3d_s${s}_train.log" 2>&1 || { echo "enc $k s$s train FAILED"; continue; }
  fi
  i=0; while [ $i -lt 4 ]; do $PT encoders.py pool 3d $k $s $i/4 > "$L/enc_${k}_3d_s${s}_pool_p$i.log" 2>&1 & i=$((i+1)); done
  wait
  $P3 encoders.py eval 3d $k $s 4 > "$L/enc_${k}_3d_s${s}_eval.log" 2>&1 || echo "enc $k s$s eval FAILED"
done; done
echo "=== QUEUE3_DONE $(date +%H:%M)"
