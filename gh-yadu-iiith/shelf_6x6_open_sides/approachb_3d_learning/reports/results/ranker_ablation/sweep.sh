#!/bin/bash
cd $HOME/approachb_3d_learning
C=configs/sampler_v1_ranker_ablation.json; P="env CUDA_VISIBLE_DEVICES=0 taskset -c 7 nice -n 10 venv/bin/python src/train.py"; L=runs/SAMPLER_V1_RANKER_ABLATION/sweep.log
for job in "0" "1" "2" "0 --geometry-only" "1 --geometry-only" "2 --geometry-only"; do set -- $job
  echo "$(date -u +%FT%TZ) START ranker $job" >> $L
  $P fit --config $C --kind ranker --seed $1 $2 >> $L 2>&1 || { echo "$(date -u +%FT%TZ) SWEEP_FAILED rc=$? at $job" >> $L; exit 1; }
done
$P evaluate --config $C --split val >> $L 2>&1 && $P freeze --config $C >> $L 2>&1 && $P evaluate --config $C --split test >> $L 2>&1 && echo "$(date -u +%FT%TZ) ABLATION_DONE" >> $L || echo "$(date -u +%FT%TZ) SWEEP_FAILED eval" >> $L
