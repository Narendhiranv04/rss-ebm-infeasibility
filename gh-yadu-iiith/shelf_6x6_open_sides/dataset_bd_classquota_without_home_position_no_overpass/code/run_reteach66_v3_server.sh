#!/bin/bash
# Re-teach (SHELF66V3 teacher, fingers checked pre-shaped) of the accepted roots of the SHELF66V2 run: SERVER part, <= 18 busy processes
# (lab rule; the stopping V2 runner's running roots are counted).  Its share: split_reteach66_v3.json (+ srv-* roots finished after the
# split).  Clean stop: touch $OUT/STOP.  Then run_3ds66_v3_server.sh resumes generation from $OUT's ledger.
Y=$HOME/yadu_bd; B=$Y/code_nooverpass/bd_samples; D=$Y/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position
SRC=$D/dataset_2500_raw; OUT=$D/dataset_2500_raw_shelf66v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
export XARM7_XML=$Y/code/mujoco_menagerie/ufactory_xarm7/xarm7.xml
mkdir -p $OUT; touch $OUT/ledger.jsonl
cd $B
exec nice -n 15 $Y/venv/bin/python shelf66_v3/reteach_shelf66_v3.py --src $SRC --out $OUT --name srv --cap 18 --split $B/split_reteach66_v3.json \
  --v2-pid "${V2_PID:-0}" >> $OUT.reteach.log 2>&1
