#!/usr/bin/env bash
# Laptop share (part 0/2) of the library v3 extension; ext files copied to gvlab2 every 10 min and at the end.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); V4=${V4:-/tmp/claude-1000/-home-ayush/a2ac7923-2d40-4b75-8596-2054292b08d1/scratchpad/v4/tr600}
export YADU_LIB_DIR=$HERE/lib66v2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
G=long-horizon@gvlab2.iiit.ac.in:learned_reap/lib66v2/
sync() { rsync -a --include '*/' --include 'ext_src_*.pkl' --exclude '*' $YADU_LIB_DIR/ $G 2>/dev/null; }
( while true; do sleep 600; sync; done ) & LOOP=$!
cd $HERE && nice -n 15 ../reap/.venv/bin/python build_ext66.py $V4/train $V4/validation --top 20 --part 0/2 --workers ${W:-8} > $HERE/outputs/laptop_build_ext.log 2>&1
sync; kill $LOOP; echo "$(date +%FT%T) laptop ext share done + synced" >> $HERE/outputs/laptop_build_ext.log
