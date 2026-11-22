#!/bin/sh
# EXPLORATORY queue, starts only after the confirmatory TEST chain prints TEST_DONE (stops if it FAILED). User order 2026-11-22:
# 1 eta on TEST, 2 scaling pilot, 3 no-failed-step test (nophase) then encoder comparison 3D (tf, gnn, cnn; seed 0). <= 7 worker processes (user OK'd on AC power).
# Frozen outputs (runs/hz) are only READ here.
E=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/explore
T=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/runs/hz/test_chain_fast.log
L=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/logs
P2=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env2d/bin/python3.14
P3=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env3d/bin/python
PT=/Users/nav/Documents/RRC_Work/RSS_2027/.venv/bin/python
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HZ_MODE=v34 ABF_DEVICE=cpu HZ_THREADS=6
cd "$E" || exit 1
mkdir -p "$L"
until grep -qE "TEST_DONE|FAILED" "$T"; do sleep 60; done
grep -q FAILED "$T" && { echo "TEST chain failed; exploratory queue not started"; exit 1; }
# hardware setup checks have priority (user 2026-11-22): wait for the board-height sweep (max 4 h)
H=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/HW_HEIGHT_DONE
n=0; until [ -f "$H" ] || [ $n -ge 240 ]; do sleep 60; n=$((n+1)); done
echo "=== hardware check done or timed out ($n min) $(date +%H:%M)"
echo "=== eta on TEST $(date +%H:%M)"
$P2 eta_sweep.py test 2d > "$L/eta_test_2d.log" 2>&1 || echo "eta 2d FAILED"
$P3 eta_sweep.py test 3d > "$L/eta_test_3d.log" 2>&1 || echo "eta 3d FAILED"
echo "=== scaling search $(date +%H:%M)"
for s in 0 1 2; do $PT scaling_search.py search $s > "$L/scaling_search_s$s.log" 2>&1 & done
wait
echo "=== scaling teach $(date +%H:%M)"
$P2 scaling_search.py teach 7 > "$L/scaling_teach.log" 2>&1 || echo "scaling teach FAILED"
$P2 scaling_search.py report > "$L/scaling_report.log" 2>&1 || echo "scaling report FAILED"
# back on the Mac after the home-server UPS failed (user 2026-11-22): why-it-works test + encoders (3D), then the 3D M-sweep.
# All restart-safe: finished training is kept (train_3d_s0.json), pools / teacher / msweep resume per root or plan.
for k in nophase tf gnn cnn; do
  echo "=== variant $k 3d $(date +%H:%M)"
  if [ ! -f "/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/enc/$k/train_3d_s0.json" ]; then
    $PT encoders.py train 3d $k 0 > "$L/enc_${k}_3d_train.log" 2>&1 || { echo "enc $k train FAILED"; continue; }
  fi
  i=0; while [ $i -lt 7 ]; do $PT encoders.py pool 3d $k 0 $i/7 > "$L/enc_${k}_3d_pool_p$i.log" 2>&1 & i=$((i+1)); done
  wait
  $P3 encoders.py eval 3d $k 0 7 > "$L/enc_${k}_3d_eval.log" 2>&1 || echo "enc $k eval FAILED"
done
echo "=== msweep 3d $(date +%H:%M)"
i=0; while [ $i -lt 7 ]; do $P3 msweep.py score $i/7 > "$L/msweep_score_p$i.log" 2>&1 & i=$((i+1)); done
wait
$P3 msweep.py teach 7 > "$L/msweep_teach.log" 2>&1 || echo "msweep teach FAILED"
$P3 msweep.py report > "$L/msweep_report.log" 2>&1 || echo "msweep report FAILED"
echo "=== EXPLORE_DONE $(date +%H:%M)"
