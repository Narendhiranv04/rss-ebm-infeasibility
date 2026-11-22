#!/bin/sh
# EXPLORATORY hardware-faithfulness replay of the frozen 3D TEST picks (both arms), trained board height 0.55 (= 43 cm riser):
#   g2all = G2 gripper, all 16 azimuths;  az180 = default gripper, straight-in only;  g2az = G2 + straight-in only.  3 workers (Mac).
H=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/hardware/robust
P3=/Users/nav/Documents/RRC_Work/RSS_2027/gh-yadu-iiith/shelf_6x6_open_sides/approachb_3d_learning/runs/ABF3/env3d/bin/python
export OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
cd "$H" || exit 1
echo "=== g2all $(date +%H:%M)"; RB_G2=1 $P3 robust_check.py run g2all 0.55:none 3 > "$H/run_g2all.log" 2>&1 || echo "g2all FAILED"
echo "=== az180 $(date +%H:%M)"; RB_AZ=180 $P3 robust_check.py run az180 0.55:none 3 > "$H/run_az180.log" 2>&1 || echo "az180 FAILED"
echo "=== g2az $(date +%H:%M)"; RB_G2=1 RB_AZ=180 $P3 robust_check.py run g2az 0.55:none 3 > "$H/run_g2az.log" 2>&1 || echo "g2az FAILED"
echo "=== HW_TESTS_DONE $(date +%H:%M)"
