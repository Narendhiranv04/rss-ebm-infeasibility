#!/bin/bash
# Stage 1 of the class-quota 2D dataset (5000 base scenes), restarted automatically (resumable); clean stop: touch 2d_plus_raw/STOP
B=/home/ayush/codebase/bd_samples
cd /home/ayush/codebase/latent-causal-tamp
export PYTHONPATH=$B/py310_shim:$B/dataset-bd:$B:/home/ayush/codebase/latent-causal-tamp PYTHONWARNINGS=ignore
for n in $(seq 20); do
  python3 $B/run_2d_plus.py --out $B/2d_plus_raw --workers 14 >> $B/2d_plus_raw.log 2>&1 && break
  [ -f $B/2d_plus_raw/STOP ] && break
  echo "$(date +%FT%T) runner exited; restart #$n" >> $B/2d_plus_raw.log; sleep 30
done
