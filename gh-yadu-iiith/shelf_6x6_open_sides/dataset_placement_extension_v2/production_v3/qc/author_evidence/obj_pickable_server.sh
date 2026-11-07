#!/bin/bash
# Server: frees 3 of the V3 runner's 18 slots (MAX_RUNNING 15, removed on exit), then runs obj_pickable.py as 3 processes at nice 15.
Y=$HOME/yadu_bd; B=$Y/code_nooverpass/bd_samples; D=$Y/diag66
OUT=$Y/runs/3d_dataset_v2_no_overpass/shelf_6x6_open_sides/without_home_position/dataset_2500_raw_shelf66v3
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONWARNINGS=ignore MUJOCO_GL=disable
export XARM7_XML=$Y/code/mujoco_menagerie/ufactory_xarm7/xarm7.xml
echo 15 > $OUT/MAX_RUNNING
trap 'rm -f $OUT/MAX_RUNNING' EXIT HUP INT TERM
t=$(date +%s)
until [ "$($Y/venv/bin/python -c "import json;print(json.load(open('$OUT/PROGRESS.json'))['running'])")" -le 15 ] && \
      [ "$(stat -c %Y $OUT/PROGRESS.json)" -gt "$t" ]; do sleep 10; done
echo "3 slots free after $(( $(date +%s) - t )) s"
cd $B
for k in 0 1 2; do nice -n 15 $Y/venv/bin/python $D/obj_pickable.py $B/shelf66_v3 $D/obj_jobs.json $D/obj_pickable.jsonl $k 3 & done
wait
echo "done: $(wc -l < $D/obj_pickable.jsonl) results"
