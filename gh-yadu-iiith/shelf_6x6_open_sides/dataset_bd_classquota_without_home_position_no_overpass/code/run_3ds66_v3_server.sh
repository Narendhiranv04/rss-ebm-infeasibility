#!/bin/bash
# 2,500-scene class-quota dataset on the 6x6 open-sided shelf (without home position, no overpass), SHELF66V3 (fingers checked pre-shaped,
# NONE counted only when the NONE audit confirms it): SERVER part, 18 workers (lab rule), name srv3.  Resumes from the re-taught ledger.
# Clean stop: touch $OUT/STOP.  Live knobs (no restart): $OUT/MAX_RUNNING, $OUT/K4_MAX_RUNNING, $OUT/QUOTA.json
Y=$HOME/yadu_bd; B=$Y/code_nooverpass/bd_samples; OUT=$Y/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
export XARM7_XML=$Y/code/mujoco_menagerie/ufactory_xarm7/xarm7.xml
mkdir -p $OUT; [ -f $OUT/K4_MAX_RUNNING ] || echo 2 > $OUT/K4_MAX_RUNNING
cd $B
exec nice -n 15 $Y/venv/bin/python shelf66_v3/run_shelf66_full.py --out $OUT --name srv3 --workers 18 >> $OUT.log 2>&1
