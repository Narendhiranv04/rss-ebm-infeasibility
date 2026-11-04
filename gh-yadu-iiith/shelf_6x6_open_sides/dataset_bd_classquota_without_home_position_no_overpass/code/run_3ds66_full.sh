#!/bin/bash
# 2,500-scene class-quota dataset on the 6x6 open-sided shelf (without home position, no overpass): LAPTOP part, 5 workers.
# Same code on the server (run_3ds66_full_server.sh); both share one ledger via merge_with_server_3ds66.sh.
# Clean stop: touch $OUT/STOP.  Live knobs (no restart): $OUT/MAX_RUNNING, $OUT/K4_MAX_RUNNING, $OUT/QUOTA.json
B=/home/ayush/codebase/bd_samples; OUT=$B/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
mkdir -p $OUT; [ -f $OUT/K4_MAX_RUNNING ] || echo 0 > $OUT/K4_MAX_RUNNING
cd $B
exec nice -n 10 python3 shelf66_full/run_shelf66_full.py --out $OUT --name lap --workers 5 >> $OUT.log 2>&1
