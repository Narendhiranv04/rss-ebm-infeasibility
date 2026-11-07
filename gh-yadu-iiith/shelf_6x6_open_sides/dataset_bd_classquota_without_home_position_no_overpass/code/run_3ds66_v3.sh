#!/bin/bash
# 2,500-scene class-quota dataset on the 6x6 open-sided shelf (without home position, no overpass), SHELF66V3 (fingers checked pre-shaped,
# NONE counted only when the NONE audit confirms it): LAPTOP part, 5 workers, name lap3.  Resumes from the ledger of the re-taught V2 roots.
# Same code on the server (run_3ds66_v3_server.sh); both share one ledger via merge_with_server_3ds66_v3.sh.
# Clean stop: touch $OUT/STOP.  Live knobs (no restart): $OUT/MAX_RUNNING, $OUT/K4_MAX_RUNNING, $OUT/QUOTA.json
B=/home/ayush/codebase/bd_samples; OUT=$B/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
mkdir -p $OUT; [ -f $OUT/K4_MAX_RUNNING ] || echo 0 > $OUT/K4_MAX_RUNNING
cd $B
exec nice -n 10 python3 shelf66_v3/run_shelf66_full.py --out $OUT --name lap3 --workers 5 >> $OUT.log 2>&1
