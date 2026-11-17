#!/usr/bin/env bash
# Zero-shot REAP-GBFS (pretrained confined-shelf checkpoint, no retraining) on our confined-shelf problems, with
# feasibility from our MuJoCo scene (CONFINED_CHECKER=yadu), then a replay of every returned plan with PickAndPlace.
#
#   bash run_zeroshot.sh [set=confined10] [seed=1] [workers=4] [time_limit_s=300]
#
# Laptop: keep workers <= 4 (cap 12 busy workers in total).
set -euo pipefail
SET=${1:-shelf66_s50}; SEED=${2:-1}; WORKERS=${3:-4}; TL=${4:-300}
HERE=$(cd "$(dirname "$0")" && pwd)
REAP=$HERE/../reap
OUT=$HERE/outputs/zeroshot
mkdir -p "$OUT"
cd "$REAP"
export CONFINED_CHECKER=yadu MUJOCO_GL=egl REAP_SHELF_DIR=$HERE EVAL_DATA=$HERE/data
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 FILTER_TIMEOUT_S=180
.venv/bin/python -m reap.eval.eval_protocol --domain confined-shelf --set ID --subdir "$SET" \
    --model_path checkpoints/confined/seed_$SEED --snapshot live --label "zeroshot_s$SEED" \
    --output_dir "$OUT" --time_limit_s "$TL" --max_budget 16384 --workers "$WORKERS" \
    2>&1 | tee "$OUT/zeroshot_s${SEED}__$SET.log"
CSV="$OUT/zeroshot_s${SEED}__confined-shelf__ID.csv"
mv "$CSV" "$OUT/zeroshot_s${SEED}__$SET.csv"
"$REAP/.venv/bin/python" "$HERE/replay66.py" "$OUT/zeroshot_s${SEED}__$SET.csv" "$HERE/data/$SET" \
    2>&1 | tee -a "$OUT/zeroshot_s${SEED}__$SET.log"
