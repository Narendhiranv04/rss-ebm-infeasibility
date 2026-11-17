#!/usr/bin/env bash
# Phase 2 on the laptop: after its own share (laptop_build.sh) is done, help gvlab2 with ITS share, working from the
# other end of gvlab2's queue (least frequent classes, last source cells first). Files are synced both ways every
# 5 min; a build job skips itself when its file already exists, so neither machine redoes the other's finished work.
set -uo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd); V4=${V4:-/tmp/claude-1000/-home-ayush/a2ac7923-2d40-4b75-8596-2054292b08d1/scratchpad/v4/tr600}
export YADU_LIB_DIR=$HERE/lib66v2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
G=long-horizon@gvlab2.iiit.ac.in:learned_reap/lib66v2/
L=$HERE/outputs/laptop_help_lib66v2.log
until grep -q "laptop share done" $HERE/outputs/laptop_build_lib66v2.log 2>/dev/null; do sleep 120; done
sync2() { rsync -a --ignore-existing --exclude '*.tmp*' $YADU_LIB_DIR/ $G 2>/dev/null; rsync -a --ignore-existing --exclude '*.tmp*' $G $YADU_LIB_DIR/ 2>/dev/null; }
sync2
( while true; do sleep 300; sync2; done ) & LOOP=$!
echo "$(date +%FT%T) helping gvlab2 from the other end of its queue" >> $L
cd $HERE && nice -n 15 ../reap/.venv/bin/python build_lib66.py $V4/train $V4/validation --exclude-shard 8/9 --reverse --workers ${W:-6} >> $L 2>&1
sync2; kill $LOOP; echo "$(date +%FT%T) laptop help done + synced" >> $L
