#!/usr/bin/env bash
# Supervised v4 training with labelled models. Cumulative training time of the v4 line (starts at 1.7 h: the three earlier segments).
# Runs 12-worker segments; if the memory guard kills one (other users' jobs fill memory), it waits until >= 40 GB are free and resumes
# from the last weights. At 6 h / 12 h / 24 h cumulative it saves a labelled model and tests it once on the 370 test scenes.
B=$HOME/learned_reap; cd $B; P=$B/pipeline; NAME=reap_shelf66_scratch_v4_s1
log() { echo "$(date +%FT%T) [v4 driver] $*" | tee -a $P/pipeline.log; }
export B DATASET66=$B/dataset66 SHELF66_V3_DIR=$B/bd_samples/shelf66_v3 REAP_SHELF_DIR=$B/reap_shelf YADU_LIB_DIR=$B/lib66v2 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
cum=${CUM0:-6120}; warm=${WARM0:-$B/models/reap_shelf66_scratch_v4_part3}; n=${N0:-1}; short=0
wait_free() { until [ "$(awk '/MemAvailable/ {print int($2/1048576)}' /proc/meminfo)" -ge 40 ]; do log "waiting for >= 40 GB free memory"; sleep 300; done; }
for h in 6 12 24; do
  target=$((h * 3600))
  while [ $cum -lt $target ]; do
    wait_free; rm -f $B/STOP; seg=$((target - cum)); M=${NAME}_seg$n
    log "segment $n: cumulative $((cum / 60)) min, running ${seg}s (12 workers) from $(basename $warm)"
    t0=$(date +%s)
    (CUDA_VISIBLE_DEVICES=0 DATA_NAME=trainv4_all_solvable CHECKER_MODE=lib MODEL_NAME=$M WARMSTART_CKPT=$warm bash reap_shelf/server/run_train66.sh 1 12 ${seg}s \
       --search_batching 64 --worker_time_limit_s 60 --env YADU_LIB_FREE=ignore --env YADU_LIB_TOL=0.02 > logs/$M.out 2>&1)
    el=$(( $(date +%s) - t0 )); cum=$((cum + el)); [ -f models/$M/q_gnn.pt ] && warm=$B/models/$M; n=$((n + 1))
    if [ -f $B/STOP ]; then log "segment ended after $((el / 60)) min by the guard (memory/temperature); cumulative $((cum / 60)) min; cooling down 10 min"; sleep 600; fi
    if [ $el -lt 300 ]; then short=$((short + 1)); [ $short -ge 5 ] && { log "5 short segments in a row: giving up"; exit 1; }; else short=0; fi
  done
  log "${h} h label reached (cumulative $((cum / 60)) min): saving models/${NAME}_${h}h and testing it"
  rm -rf models/${NAME}_${h}h; cp -a $warm models/${NAME}_${h}h; rm -f $B/STOP
  if [ $h -lt 24 ]; then
    ( bash reap_shelf/server/run_test.sh ../models/${NAME}_${h}h scratch_v4_${h}h_v4_s1 10 300 > logs/test_scratch_v4_${h}h_v4_s1.log 2>&1; log "test of the ${h} h model finished" ) &
  else
    bash reap_shelf/server/run_test.sh ../models/${NAME}_24h scratch_v4_24h_v4_s1 20 300 > logs/test_scratch_v4_24h_v4_s1.log 2>&1; log "test of the 24 h model finished"
  fi
done
wait; log "all labelled tests finished"
