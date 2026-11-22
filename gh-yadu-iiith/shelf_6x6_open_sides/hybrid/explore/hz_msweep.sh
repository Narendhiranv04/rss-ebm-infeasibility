#!/bin/sh
# Home-server launcher (EXPLORATORY, 3D VAL): M-sweep (M = 4/6/8, XYNet vs hazard) via msweep.py. Runs next to hz_server.sh,
# so it uses W=6 workers (hz_server.sh uses up to 10; 16 threads total; the power cap bounds the whole package).
# REFUSES to start unless the CPU power cap is set and enabled (UPS limit, nav 2026-11-22: 100-110 W). Restart-safe (resumes per root / plan).
#   usage (on the server):  nohup sh ~/Documents/RRC/rss_gen/hz/gh-yadu-iiith/shelf_6x6_open_sides/hybrid/explore/hz_msweep.sh > ~/Documents/RRC/rss_gen/hz/msweep_run.log 2>&1 &
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
W=6
[ -x "$PY" ] || { echo "missing $PY"; exit 2; }
mkdir -p "$L"
export MPLBACKEND=Agg OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 HZ_MODE=v34 ABF_DEVICE=cpu MUJOCO_GL=disable
cd "$R/hybrid/explore" || exit 2
echo "=== msweep score $(date +%H:%M)"
i=0; while [ $i -lt $W ]; do nice -n 10 "$PY" msweep.py score $i/$W > "$L/msweep_score_p$i.log" 2>&1 & i=$((i+1)); done
wait
echo "=== msweep teach $(date +%H:%M)"
nice -n 10 "$PY" msweep.py teach $W > "$L/msweep_teach.log" 2>&1 || echo "msweep teach FAILED"
nice -n 10 "$PY" msweep.py report > "$L/msweep_report.log" 2>&1 || echo "msweep report FAILED"
echo "=== MSWEEP_DONE $(date +%H:%M)"
