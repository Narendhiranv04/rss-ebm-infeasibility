# NONE trim (trim_none66_v3.py) of the 2,500-scene 6x6 shelf dataset, lab server: residues 5-22 of 23 (laptop 0-4), 18 workers, nice 15.
Y=$HOME/yadu_bd; B=$Y/code_nooverpass/bd_samples; W=$Y/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
export XARM7_XML=$Y/code/mujoco_menagerie/ufactory_xarm7/xarm7.xml
cd $B
exec nice -n 15 $Y/venv/bin/python shelf66_v3/trim_none66_v3.py --src $W/dataset_2500_raw_shelf66v3_none_dense --out $W/dataset_2500_raw_shelf66v3_none_trim --cap 18 --part 5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22 --parts 23 >> $W/dataset_2500_raw_shelf66v3_none_trim.log 2>&1
