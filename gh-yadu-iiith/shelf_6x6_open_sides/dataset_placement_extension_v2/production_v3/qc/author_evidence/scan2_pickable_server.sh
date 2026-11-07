#!/bin/bash
# Server: frees 4 of the V3 runner's 18 slots (MAX_RUNNING 14, removed on exit), then runs obj_pickable.py as 4 processes (scan 2: scene ceilings) at nice 15.
Y=$HOME/yadu_bd; B=$Y/code_nooverpass/bd_samples; D=$Y/diag66
OUT=$Y/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
export XARM7_XML=$Y/code/mujoco_menagerie/ufactory_xarm7/xarm7.xml
echo 14 > $OUT/MAX_RUNNING
trap 'rm -f $OUT/MAX_RUNNING' EXIT HUP INT TERM
t=$(date +%s)
until [ "$($Y/venv/bin/python -c "import json;print(json.load(open('$OUT/PROGRESS.json'))['running'])")" -le 14 ] && \
      [ "$(stat -c %Y $OUT/PROGRESS.json)" -gt "$t" ]; do sleep 10; done
echo "4 slots free after $(( $(date +%s) - t )) s"
cd $B
for k in 0 1 2 3; do nice -n 15 $Y/venv/bin/python $D/obj_pickable3.py $B/shelf66_v3 $D/scan2_jobs.json $D/scan2_pickable.jsonl $k 4 & done
wait
echo "done: $(wc -l < $D/scan2_pickable.jsonl) results"
