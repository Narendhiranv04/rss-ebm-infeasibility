#!/bin/sh
# EXPLORATORY scaling fix run (scaling_search2.py): 3 seeds in parallel, then teacher on the top-5 plans, then report.
E=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/explore
L=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/explore/logs
P2=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env2d/bin/python3.14
PT=/Users/nav/Documents/RRC_Work/RSS_2027/.venv/bin/python
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg HZ_MODE=v34 ABF_DEVICE=cpu
cd "$E" || exit 1
echo "=== search $(date +%H:%M)"
for s in 0 1 2; do $PT scaling_search2.py search $s > "$L/scal2_s$s.log" 2>&1 & done
wait
echo "=== teach $(date +%H:%M)"
$P2 scaling_search2.py teach 3 > "$L/scal2_teach.log" 2>&1 || echo "teach FAILED"
$P2 scaling_search2.py report > "$L/scal2_report.log" 2>&1 || echo "report FAILED"
echo "=== SCAL2_DONE $(date +%H:%M)"
