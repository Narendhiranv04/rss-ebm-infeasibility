#!/bin/sh
# Queue 4 (user 2026-11-22: rigour tests): robustness of both arms' frozen 3D TEST picks (462 plans), board 0.55 m, EXPLORATORY.
# Waits for queues 2 and 3 to finish, then 6 workers.  j05/j10 = objects shifted 5/10 mm in their cells; f07 = friction x0.7; m15 = mass x1.5.
H=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hardware/robust
X=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore
P3=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env3d/bin/python
cd "$H" || exit 1
until grep -q QUEUE2_DONE "$X/queue2.log" && grep -q QUEUE3_DONE "$X/queue3.log"; do sleep 60; done
for c in "j05 0.005,1,1" "j10 0.01,1,1" "f07 0,0.7,1" "m15 0,1,1.5"; do
  set -- $c
  echo "=== $1 $(date +%H:%M)"
  ROBUST=$2 $P3 robust_check.py run $1 0.55:none 6 > "$H/run_$1.log" 2>&1 || echo "$1 FAILED"
done
echo "=== QUEUE4_DONE $(date +%H:%M)"
