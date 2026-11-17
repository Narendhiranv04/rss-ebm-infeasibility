#!/bin/bash
# REAP-GBFS on a test split of the 6x6 shelf (default: Narendhiran's V4 split, split_v4.json, 370 test scenes), on a server.
#   bash run_test.sh [model_path=checkpoints/confined/seed_1] [label=zeroshot_s1] [workers=18] [tl=300]
#   zero-shot (released model):  bash run_test.sh ../models/reap_confined_released_s1 zeroshot_v4_s1
#   trained on 6x6:              bash run_test.sh ../models/reap_shelf66_trained_s1 trained_v4_s1
#   (model_path is relative to the REAP repo; each label writes its own outputs/<label>/)
# Resumable: the test set is cut into chunks of CHUNK problems; each chunk's eval CSV and replay CSV are written when it
# finishes and skipped on a rerun. A STOP file (reap_guard.sh) ends the run after the chunk in flight is killed.
set -uo pipefail
B=${B:-$(cd "$(dirname "$0")/../.." && pwd)}; HERE=$B/reap_shelf; REAP=$B/reap
MODEL=${1:-checkpoints/confined/seed_1}; LABEL=${2:-zeroshot_s1}; W=${3:-18}; TL=${4:-300}; CHUNK=${CHUNK:-36}
SPLIT=${SPLIT:-split_v4.json}; SET=${SET:-test_v4}                 # V4 split (naren/dataset-done): 370 test scenes
OUT=$B/outputs/$LABEL; mkdir -p "$OUT"
export CONFINED_CHECKER=yadu MUJOCO_GL=egl REAP_SHELF_DIR=$HERE EVAL_DATA=$HERE/data DATASET66=$B/dataset66
export SHELF66_V3_DIR=$B/bd_samples/shelf66_v3
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FILTER_TIMEOUT_S=180
PY=$REAP/.venv/bin/python
rm -f $B/STOP

# the test problems, once, in chunks
T=$HERE/data/$SET
if [ ! -f $T/DONE ]; then
  rm -rf $T && mkdir -p $T/all
  $PY $HERE/roots66_to_pddl.py $T/all --split $HERE/$SPLIT --part test
  ls $T/all/*.pddl | shuf --random-source=<(yes) | split -l $CHUNK - $T/list_      # mixed classes per chunk
  for f in $T/list_*; do
    c=$T/chunk_${f##*list_}; mkdir -p $c
    while read p; do mv "$p" "${p%.pddl}.json" $c/; done < $f
  done
  rm -rf $T/list_* $T/all; touch $T/DONE
fi

cd $REAP
for c in $(ls -d $T/chunk_* | sort); do
  [ -f $B/STOP ] && { echo "STOP file: ending"; exit 1; }
  n=$(basename $c)
  if [ ! -f $OUT/$n.csv ]; then
    echo "$(date +%T) eval $n"
    nice -n 15 $PY -m reap.eval.eval_protocol --domain confined-shelf --set ID --subdir $SET/$n \
      --model_path $MODEL --snapshot live --label "$LABEL" --output_dir $OUT/tmp_$n \
      --time_limit_s $TL --max_budget 16384 --workers $W > $OUT/$n.eval.log 2>&1 \
      && mv $OUT/tmp_$n/${LABEL}__confined-shelf__ID.csv $OUT/$n.csv && rm -rf $OUT/tmp_$n \
      || { echo "eval $n failed (see $OUT/$n.eval.log)"; continue; }
  fi
  if [ ! -f $OUT/$n.replay.csv ]; then
    echo "$(date +%T) replay $n"
    nice -n 15 $PY $HERE/replay66.py $OUT/$n.csv $c --out $OUT/$n.replay.csv --workers $W > $OUT/$n.replay.log 2>&1 \
      || echo "replay $n failed (see $OUT/$n.replay.log)"
  fi
done
n_eval=$(ls $OUT/chunk_*.replay.csv 2>/dev/null | wc -l); n_all=$(ls -d $T/chunk_* | wc -l)
echo "$(date +%T) chunks done: $n_eval / $n_all"
if [ "$n_eval" = "$n_all" ]; then
  $PY - <<EOF
import pandas as pd, glob
pd.concat([pd.read_csv(f, dtype={"cls": str}) for f in sorted(glob.glob("$OUT/chunk_*.replay.csv"))]).to_csv("$OUT/${SET}__replay.csv", index=False)
EOF
  $PY $HERE/metrics66.py $OUT/${SET}__replay.csv --json $OUT/metrics_all.json | tee $OUT/metrics.txt
  echo RUN_DONE
fi
