# NONE-only run of the 2,500-scene 6x6 shelf dataset (dense construction + blocking certificate, run_shelf66_none.py), laptop.
B=/home/ayush/codebase/bd_samples; OUT=$B/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3_none_dense
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
mkdir -p $OUT; [ -f $OUT/MAX_RUNNING ] || echo 2 > $OUT/MAX_RUNNING
cd $B
exec nice -n 10 python3 shelf66_v3/run_shelf66_none.py --out $OUT --name lapN --workers 5 >> $OUT.log 2>&1
