#!/usr/bin/env bash
# The whole trained-REAP pipeline on gvlab2, unattended and resumable (each finished step leaves pipeline/<step>.done):
#   1 lib       build the conflict library (lib66) for the classes of the 600 train + 50 val V4 scenes   (~7 h, 29 workers)
#   2 validate  library vs the teacher + dataset labels; STOPS the pipeline if the gate fails (validate_lib.py)
#   3 scratch   REAP from random weights, 24 h (REAP's own confined-shelf budget)   -> models/reap_shelf66_scratch_s1
#   4 t_scratch test on V4's 370 test scenes with the FULL teacher + replay          -> outputs/scratch_v4_s1
#   5 finetune  REAP from the released confined weights (WARMSTART_CKPT), 24 h       -> models/reap_shelf66_finetune_s1
#   6 t_finetune same test                                                             -> outputs/finetune_v4_s1
# Before every step it waits until the server is free enough: other users' load < OTHERS_MAX (default 16 of 48 cores)
# and >= MIN_FREE_GB available memory. Training uses the GPU if >= 24 GB of it is free (16 workers), else CPU (29).
#   B=~/learned_reap setsid nohup bash reap_shelf/server/pipeline66.sh > pipeline.log 2>&1 &
set -uo pipefail
B=${B:-$(cd "$(dirname "$0")/../.." && pwd)}
HERE=$B/reap_shelf; PY=$B/reap/.venv/bin/python; P=$B/pipeline; mkdir -p $P $B/logs
OTHERS_MAX=${OTHERS_MAX:-10}; MIN_FREE_GB=${MIN_FREE_GB:-40}; BUILD_W=${BUILD_W:-36}; TRAIN_W=${TRAIN_W:-35}; WALL_SCRATCH=${WALL_SCRATCH:-24h}; WALL_FT=${WALL_FT:-24h}   # cap 38 cores (user, 2026-11-15)
export DATASET66=$B/dataset66 SHELF66_V3_DIR=$B/bd_samples/shelf66_v3 REAP_SHELF_DIR=$HERE YADU_LIB_DIR=$B/lib66v2 \
       OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
log() { echo "$(date +%FT%T) $*" | tee -a $P/pipeline.log; }

wait_free() {                       # other users' load (load average minus our running REAP processes) and free memory
  while true; do
    [ -f $B/PIPELINE_STOP ] && { log "PIPELINE_STOP file: exit"; exit 0; }
    l=$(awk '{print int($1)}' /proc/loadavg); ours=$(pgrep -fc "$B/reap/.venv/bin/python" || true)
    others=$(( l - ours )); free=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
    if [ "$others" -lt "$OTHERS_MAX" ] && [ "$free" -ge "$MIN_FREE_GB" ]; then return; fi
    log "waiting for a free server: others' load $others (max $OTHERS_MAX), free $free GB (min $MIN_FREE_GB)"; sleep 300
  done
}
guard() { pgrep -f "$HERE/server/reap_guard.sh" >/dev/null || (B=$B setsid nohup bash $HERE/server/reap_guard.sh > /dev/null 2>&1 < /dev/null &); }
step_done() { [ -f $P/$1.done ]; }
mark() { touch $P/$1.done; log "step $1 done"; }
stop_check() { [ -f $B/STOP ] && { log "guard STOP (temperature / memory): rm $B/STOP and rerun to resume"; exit 1; }; }

train_device() {                    # GPU when >= 24 GB of it is free, else CPU
  g=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [ -n "$g" ] && [ "$g" -ge 24000 ]; then echo gpu; else echo cpu; fi
}
train() {                           # train <model name> <walltime> [warmstart dir]
  local name=$1 wall=$2 ws=${3:-}
  local dev=$(train_device); local W=$TRAIN_W; local GPU_W=${GPU_W:-12}; local XENV=""
  if [ "$dev" = gpu ]; then W=$GPU_W; export CUDA_VISIBLE_DEVICES=0; else export CUDA_VISIBLE_DEVICES=; XENV="--env CUDA_VISIBLE_DEVICES="; fi
  log "train $name on $dev with $W workers, walltime $wall, warmstart ${ws:-none}"
  (cd $B && B=$B CHECKER_MODE=lib YADU_LIB_DIR=$B/lib66v2 MODEL_NAME=$name WARMSTART_CKPT=$ws bash $HERE/server/run_train66.sh 1 $W $wall \
     --search_batching 8 --worker_time_limit_s 300 $XENV --env YADU_LIB_FREE=$(cat $P/lib_free_mode) --env YADU_LIB_TOL=$(cat $P/lib_tol 2>/dev/null || echo 0) > $B/logs/$name.out 2>&1)
  unset CUDA_VISIBLE_DEVICES
  [ -f $B/STOP ] && return 1                       # killed by the guard: not finished
  [ -d $B/models/$name ] && [ -f $B/models/$name/MODEL_INFO.json ]
}
test_model() {                      # test_model <model name> <label>
  log "test $1 -> outputs/$2"
  (cd $B && rm -f $B/STOP && B=$B bash $HERE/server/run_test.sh ../models/$1 $2 29 300 > $B/logs/test_$2.log 2>&1)
  grep -q RUN_DONE $B/logs/test_$2.log
}

log "pipeline start (B=$B)"; guard
if ! step_done lib; then
  wait_free; stop_check
  D="$HERE/trainv4_n600/train $HERE/trainv4_n600/validation"
  log "build the library v2 (gvlab2 share; the laptop builds shard 8/9 and copies it here)"
  (cd $HERE && $PY build_lib66.py $D --exclude-shard 8/9 --workers $BUILD_W > $B/logs/build_lib66v2.log 2>&1)
  log "build whatever is still missing (the laptop's share if it is not complete yet)"
  (cd $HERE && $PY build_lib66.py $D --workers $BUILD_W > $B/logs/build_lib66v2_rest.log 2>&1)
  grep -q "^done" $B/logs/build_lib66v2_rest.log && mark lib || { log "library build incomplete (see logs/build_lib66v2*.log)"; exit 1; }
fi
if ! step_done validate; then
  wait_free; stop_check
  best=""; best_agree=0
  for mode in tail ignore; do                       # how much of the stored free move (ready -> hover) is checked
    log "validate the library, free-move mode $mode"
    (cd $HERE && $PY validate_lib.py $HERE/trainv4_n600/train --scenes 40 --workers $BUILD_W --free-mode $mode \
        --out $P/validate_lib_v2_$mode.json > $B/logs/validate_lib_v2_$mode.log 2>&1)
    if [ $? -eq 0 ]; then
      ag=$($PY -c "import json; print(int(1000 * json.load(open('$P/validate_lib_v2_$mode.json'))['agree']))")
      [ "$ag" -gt "$best_agree" ] && { best=$mode; best_agree=$ag; }
    fi
  done
  if [ -n "$best" ]; then echo $best > $P/lib_free_mode; log "library gate passed: free-move mode $best (agreement $best_agree/1000)"; mark validate
  else log "LIBRARY GATE FAILED for both free-move modes: see $P/validate_lib_v2_*.json - pipeline stopped"; touch $P/GATE_FAILED; exit 2; fi
fi
if ! step_done scratch; then wait_free; stop_check; train reap_shelf66_scratch_s1 $WALL_SCRATCH ${SCRATCH_WS:-} && mark scratch || { log "scratch training failed"; exit 1; }; fi
if ! step_done t_scratch; then wait_free; stop_check; test_model reap_shelf66_scratch_s1 scratch_v4_s1 && mark t_scratch || { log "scratch test failed"; exit 1; }; fi
if ! step_done finetune; then
  wait_free; stop_check
  train reap_shelf66_finetune_s1 $WALL_FT $B/models/reap_confined_released_s1 && mark finetune || { log "fine-tuning failed"; exit 1; }
fi
if ! step_done t_finetune; then wait_free; stop_check; test_model reap_shelf66_finetune_s1 finetune_v4_s1 && mark t_finetune || { log "fine-tune test failed"; exit 1; }; fi
log "PIPELINE DONE"
