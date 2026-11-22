#!/bin/bash
# EXPLORATORY: the team's E-M4M (latent-causal-tamp @ naren/analytical-baselines-k3-finalize, config paper, seed 0, 300 CPU-s) on a 3D
# split, on this Mac.  Their code runs unchanged from runs/ext (see em4m_test3d.py for the two Mac adaptations).
#   em4m_test3d.sh run     N shard processes (nice 10, OMP 1) over every root of the split; restart-safe (done roots are skipped)
#   em4m_test3d.sh finish  merge shards -> seal -> validate -> Mac teacher -> predictions_3d_results.jsonl + SUMMARY_3d.json
#   em4m_test3d.sh all     run, then finish
# Env: N=6  SPLIT=test  BUDGET=300 (CPU-s; see the speed note in the report before changing it)  OUT=approachb_3d_learning/runs/explore/em4m_${SPLIT}3d  ROOTS=a,b (smoke only)  TW=2 (teacher workers)
#      ALLOW_MISSING=1 (smoke only: finish with roots missing)
set -euo pipefail
B=$(cd "$(dirname "$0")/../.." && pwd)                       # shelf_6x6_open_sides
RUNS=$B/approachb_3d_learning/runs; REPO=$RUNS/ext/latent-causal-tamp; A=$RUNS/ABF3
SPLIT=${SPLIT:-test}; N=${N:-6}; TW=${TW:-2}; OUT=${OUT:-$RUNS/explore/em4m_${SPLIT}3d}
PUB=$RUNS/K3ABL/runs/public/public_${SPLIT}_3d.jsonl; PY=${PY:-$A/env3d/bin/python}; BUDGET=${BUDGET:-300}; RUN=em4m_paper_s0_b$BUDGET
export ABF_ROOT=$A FIN3D_CODE=$A/fin3d/code FIN3D_SRC=$A/fin3d/src/dataset_bd_classquota_without_home_position_no_overpass \
       DISCO_ROOT=$OUT/disco_root EM4M_OUT=$OUT MUJOCO_GL=disable CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
       OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
mkdir -p "$OUT/shards"
# shards are a hash partition by N: restarting with another N would re-run done roots, so N is pinned on first launch
[ -f "$OUT/N" ] || echo "$N" > "$OUT/N"
[ "$(cat "$OUT/N")" = "$N" ] || { echo "REFUSED: $OUT was started with N=$(cat "$OUT/N")"; exit 2; }

run() {
  [ -s "$OUT/raw_paths.json" ] || "$PY" "$B/hybrid/explore/em4m_test3d.py" paths "$SPLIT" "$OUT/raw_paths.json"
  cd "$REPO"
  for ((i = 0; i < N; i++)); do
    S=$OUT/shards/3d_${i}_of_$N; mkdir -p "$S"
    nice -n 10 "$PY" -m baselines.analytical.run run --domain 3d --public "$PUB" --out "$S" --methods em4m --config paper \
      --seeds 0 --budget "$BUDGET" --part "$i/$N" --raw-paths "$OUT/raw_paths.json" ${ROOTS:+--roots "$ROOTS"} >> "$S/run.log" 2>&1 &
  done
  wait
  echo "done: $(cat "$OUT"/shards/*/$RUN/predictions_3d.jsonl 2>/dev/null | wc -l) / $(wc -l < "$PUB") roots predicted"
}

finish() {
  if [ ! -f "$OUT/$RUN/predictions_3d.jsonl.SEAL.json" ]; then           # a sealed file is never re-merged
    (cd "$REPO" && "$PY" -m baselines.analytical.run merge --domain 3d --public "$PUB" --out "$OUT" \
       ${ALLOW_MISSING:+--allow-missing} "$OUT"/shards/3d_*_of_$N)
  fi
  nice -n 10 "$PY" "$B/hybrid/explore/em4m_test3d.py" verify "$SPLIT" "$OUT/$RUN" "$TW"
}

case "${1:-}" in
  run) run ;;
  finish) finish ;;
  all) run; finish ;;
  *) sed -n 2,9p "$0"; exit 2 ;;
esac
