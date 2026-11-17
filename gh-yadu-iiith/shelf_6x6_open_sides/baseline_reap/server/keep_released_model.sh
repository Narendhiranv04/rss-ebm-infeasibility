#!/usr/bin/env bash
# REAP's released confined-shelf checkpoint (the zero-shot model), copied unchanged to models/reap_confined_released_s<seed>/
set -euo pipefail
B=${B:-$(cd "$(dirname "$0")/../.." && pwd)}; SEED=${1:-1}
M=$B/models/reap_confined_released_s$SEED; mkdir -p $M
cp -a $B/reap/checkpoints/confined/seed_$SEED/. $M/
cat > $M/MODEL_INFO.json <<J
{"model": "REAP released confined-shelf checkpoint (Franka, REAP's own confined shelf), NOT trained on our data: the zero-shot baseline",
 "source": "https://github.com/elicited-preconditions/reap/releases/tag/v1.0 reap_ckpt_confined.tar.gz, seed_$SEED",
 "reap_commit": "d2ba23d", "results_on_test482": "outputs/zeroshot_s$SEED/"}
J
echo "released model -> $M"
