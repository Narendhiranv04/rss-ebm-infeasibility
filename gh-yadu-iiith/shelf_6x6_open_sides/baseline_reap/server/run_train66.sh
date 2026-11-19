#!/usr/bin/env bash
# Retrain REAP on the 6x6 shelf (train part of split_v4.json, Narendhiran's V4 split) with our feasibility checker (CONFINED_CHECKER=yadu).
#   bash run_train66.sh [seed=1] [workers=29] [walltime=48h] [extra train.py args...]
# B = the bundle folder (default: this script's grand-parent). 29 workers + validation + GPU trainer = 31 processes
# (gvlab2 cap: 32 cores). Log: $B/logs/train66_s<seed>.log (select_checkpoint.py reads it). reap_guard.sh runs alongside.
set -uo pipefail
B=${B:-$(cd "$(dirname "$0")/../.." && pwd)}
SEED=${1:-1}; W=${2:-29}; WALL=${3:-48h}; shift $(( $# < 3 ? $# : 3 ))
HERE=$B/reap_shelf; REAP=$B/reap; PY=$REAP/.venv/bin/python
export DATASET66=$B/dataset66 SHELF66_V3_DIR=$B/bd_samples/shelf66_v3 REAP_SHELF_DIR=$HERE
export CONFINED_CHECKER=yadu MUJOCO_GL=egl REAP_DATA=$HERE REAP_OUT=$B/outputs WANDB_MODE=offline
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FILTER_TIMEOUT_S=180
mkdir -p $B/logs $B/outputs
NTRAIN=${NTRAIN:-600}; SPLIT=${SPLIT:-split_v4.json}; DATA=${DATA_NAME:-trainv4_n$NTRAIN}
# CHECKER_MODE=lib (conflict library lib66, default) | table (per-scene teacher-plan table); MODEL_NAME: where the final
# weights go (models/<MODEL_NAME>); WARMSTART_CKPT (env): start from these weights = fine-tuning, unset = from scratch
CHECKER_MODE=${CHECKER_MODE:-lib}; MODEL_NAME=${MODEL_NAME:-reap_shelf66_scratch_s$SEED}
if [ "$CHECKER_MODE" = lib ]; then CHK_ENV="--env YADU_LIB=1 --env YADU_LIB_DIR=${YADU_LIB_DIR:-$B/lib66}"
else CHK_ENV="--env YADU_TABLE=1 --env YADU_TABLE_DIR=${YADU_TABLE_DIR:-$B/table66}"; fi
WS_ENV=""; [ -n "${WARMSTART_CKPT:-}" ] && WS_ENV="--env WARMSTART_CKPT=$WARMSTART_CKPT"
[ -f $HERE/$DATA/DONE ] || { $PY $HERE/make_train66.py $HERE/$DATA --split $HERE/$SPLIT --n-train $NTRAIN && touch $HERE/$DATA/DONE; }
pgrep -f "$HERE/server/reap_guard.sh" >/dev/null || (B=$B nohup bash $HERE/server/reap_guard.sh > /dev/null 2>&1 < /dev/null &)
cd $REAP
echo "$(date +%FT%T) train seed=$SEED workers=$W walltime=$WALL checker=$CHECKER_MODE model=$MODEL_NAME warmstart=${WARMSTART_CKPT:-none} $*" | tee -a $B/logs/$MODEL_NAME.log
nice -n 15 $PY scripts/train.py --config configs/confined.yaml --seed $SEED --data_path $DATA \
    --num_workers $W --walltime $WALL --wandb_project reap_shelf66 \
    --env CONFINED_CHECKER=yadu $CHK_ENV $WS_ENV --env FILTER_TIMEOUT_S=${TRAIN_FILTER_TIMEOUT_S:-20} --env REAP_SHELF_DIR=$HERE --env DATASET66=$B/dataset66 --env SHELF66_V3_DIR=$B/bd_samples/shelf66_v3 \
    "$@" 2>&1 | tee -a $B/logs/$MODEL_NAME.log
echo "$(date +%FT%T) train exit ${PIPESTATUS[0]}" | tee -a $B/logs/$MODEL_NAME.log
# the config sets CONFINED_CHECKER=honest itself; --env above overrides it. Refuse to keep a model trained on the wrong checker.
grep -q "checker=yadu-shelf-no-overpass" $B/logs/$MODEL_NAME.log || { echo "WRONG CHECKER in the log: model not kept"; exit 1; }

# keep the trained model separate from REAP's released (zero-shot) one: models/reap_shelf66_trained_s<seed>/
RUN=$(grep -o 'SAVE_PATH=[^ ]*' $B/logs/$MODEL_NAME.log | tail -1 | cut -d= -f2)
if [ -n "$RUN" ] && [ -d $B/outputs/model/$RUN ]; then
  M=$B/models/$MODEL_NAME; mkdir -p $M
  cp -a $B/outputs/model/$RUN/. $M/
  $PY - <<PYEOF
import json, hashlib, subprocess, datetime
split = json.load(open("$HERE/$SPLIT"))
info = {"model": "REAP retrained on the 6x6 shelf (shelf66 V3 teacher as feasibility checker, CONFINED_CHECKER=yadu)",
        "seed": $SEED, "workers": $W, "walltime": "$WALL", "run": "$RUN", "copied": datetime.datetime.now().isoformat(),
        "train_data": "$HERE/$DATA ($NTRAIN class-proportional roots of the train part of $SPLIT + 50 validation roots from its val part)",
        "feasibility_in_training": "per-scene table of the teacher's own plans (shelf66_table.py, YADU_TABLE=1), teacher on a miss",
        "split_sha256": split["sha256"], "split_file": "$SPLIT", "test_roots_never_used": sum(len(v) for v in split["test"].values()),
        "reap_commit": "d2ba23d", "reap_patch": "reap_shelf/reap_yadu.patch",
        "init": "${WARMSTART_CKPT:-from scratch (configs/confined.yaml)}", "checker_mode_in_training": "$CHECKER_MODE", "extra_train_args": "$*",
        "train_filter_timeout_s": "${TRAIN_FILTER_TIMEOUT_S:-20} (a check slower than this counts as infeasible in training; test uses 180)",
        "note": "search_batching / worker_time_limit_s adapted to the slow teacher checker (REAP's defaults assume microsecond checks)", "log": "$B/logs/$MODEL_NAME.log"}
json.dump(info, open("$M/MODEL_INFO.json", "w"), indent=1)
print("trained model ->", "$M")
PYEOF
fi
