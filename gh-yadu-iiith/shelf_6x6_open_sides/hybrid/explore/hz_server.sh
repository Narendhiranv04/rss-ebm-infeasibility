#!/bin/sh
# Home-server launcher (EXPLORATORY, VAL only): the why-it-works test (nophase) and the 3D encoder comparison (tf, gnn, cnn), seed 0.
# Runs inside ~/Documents/RRC/rss_gen/hz only (copy of the Mac tree; nothing outside it is written).
# REFUSES to start unless the CPU power cap is set (UPS limit, nav 2026-11-22: 100-110 W); a reboot resets the cap.
#   usage (on the server):  sh ~/Documents/RRC/rss_gen/hz/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/explore/hz_server.sh
set -u
CAP=110000000
[ "$(cat /sys/class/powercap/intel-rapl:0/enabled 2>/dev/null)" = 1 ] || { echo "REFUSED: intel-rapl:0 not enabled (cap not enforced)"; exit 2; }
for c in 0 1; do
  v=$(cat /sys/class/powercap/intel-rapl:0/constraint_${c}_power_limit_uw 2>/dev/null)
  case $v in ''|*[!0-9]*) echo "REFUSED: constraint_$c unreadable or not a number: '$v'"; exit 2;; esac
  if [ "$v" -le 0 ] || [ "$v" -gt "$CAP" ]; then
    echo "REFUSED: intel-rapl:0 constraint_$c = $v uW (must be 1..$CAP). Set the cap with sudo first."; exit 2
  fi
done
B="$HOME/Documents/RRC/rss_gen/hz"
R="$B/gh-yadu-iiith/shelf_6x6_open_sides"
PY="$B/.venv/bin/python"
L="$R/approachb_3d_learning/runs/explore/logs_server"
W=10                                     # worker processes (16 threads; ~1-2 GB each of 30 GB RAM; cap limits power)
[ -x "$PY" ] || { echo "missing $PY"; exit 2; }
mkdir -p "$L"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HZ_MODE=v34 HZ_THREADS=8 ABF_DEVICE=cpu MUJOCO_GL=disable
cd "$R/hybrid/explore" || exit 2
for k in nophase tf gnn cnn; do
  # restart-safe: train_3d_s0.json is written only when training finished, so a re-run after a crash keeps that model
  # (pool parts and teacher labels resume per root / per plan); an unfinished training restarts from scratch.
  if [ -f "$R/approachb_3d_learning/runs/explore/enc/$k/train_3d_s0.json" ]; then
    echo "=== $k 3d train already done, kept $(date +%H:%M)"
  else
    echo "=== $k 3d train $(date +%H:%M)"
    nice -n 10 "$PY" encoders.py train 3d $k 0 > "$L/enc_${k}_3d_train.log" 2>&1 || { echo "$k train FAILED"; continue; }
  fi
  echo "=== $k 3d pool $(date +%H:%M)"
  i=0; while [ $i -lt $W ]; do nice -n 10 "$PY" encoders.py pool 3d $k 0 $i/$W > "$L/enc_${k}_3d_pool_p$i.log" 2>&1 & i=$((i+1)); done
  wait
  echo "=== $k 3d eval $(date +%H:%M)"
  nice -n 10 "$PY" encoders.py eval 3d $k 0 $W > "$L/enc_${k}_3d_eval.log" 2>&1 || echo "$k eval FAILED"
done
echo "=== SERVER_DONE $(date +%H:%M)"
