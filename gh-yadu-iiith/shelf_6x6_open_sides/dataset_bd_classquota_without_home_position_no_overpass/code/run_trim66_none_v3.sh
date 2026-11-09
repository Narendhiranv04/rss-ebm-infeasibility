# NONE trim (trim_none66_v3.py) of the 2,500-scene 6x6 shelf dataset, laptop: residues 0-4 of 23 (server 5-22), 10 workers.
B=/home/ayush/codebase/bd_samples; W=$B/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
cd $B
exec nice -n 10 python3 shelf66_v3/trim_none66_v3.py --src $W/dataset_2500_raw_shelf66v3_none_dense --out $W/dataset_2500_raw_shelf66v3_none_trim --cap 10 --part 0,1,2,3,4 --parts 23 >> $W/dataset_2500_raw_shelf66v3_none_trim.log 2>&1
