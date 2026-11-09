# NONE-only run of the 2,500-scene 6x6 shelf dataset (dense construction + blocking certificate, run_shelf66_none.py), lab server.
Y=$HOME/yadu_bd; B=$Y/code_nooverpass/bd_samples; OUT=$Y/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3_none_dense
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
export XARM7_XML=$Y/code/mujoco_menagerie/ufactory_xarm7/xarm7.xml
mkdir -p $OUT
cd $B
exec nice -n 15 $Y/venv/bin/python shelf66_v3/run_shelf66_none.py --out $OUT --name srvN --workers 18 >> $OUT.log 2>&1
