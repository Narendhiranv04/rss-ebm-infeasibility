#!/bin/bash
# Blackwell server-side auto-launcher for the Q2 V2/V3 BLACKWELL share (survives the laptop shutting down).  Every 2 min: start (or resume) the
# share on E-cores 8-23 with the coexistence watchdog ONLY when sajid's CPU queue has nothing pending, his jobs use < 1.5 cores, yadu's generator is
# absent and the package is < 65 C.  A runner safety stop (e.g. TEMPERATURE) is resumed the same way after cooling, at most MAX_STARTS times;
# finished roots are always kept (idempotent resume).  Stops when the run's status is final with stop=null and every root packed, or on STOP_AUTOLAUNCH.
SHARE=${1:-BLACKWELL}; MAX_STARTS=${2:-6}; Q=$HOME/approachb_3d_learning/q2bw; R=$Q/src/gh-yadu-iiith/planar_2d/dataset_relocations_v1/launch_ready_2d_v2_gvlab/runs/spillover_blackwell
LOG=$Q/autolaunch.log; n=0; echo "$(date -u +%FT%TZ) start share=$SHARE max_starts=$MAX_STARTS" >> $LOG
while true; do
  [ -f $Q/STOP_AUTOLAUNCH ] && { echo "$(date -u +%FT%TZ) STOP_AUTOLAUNCH" >> $LOG; exit 0; }
  if pgrep -f "[s]pill2d_v2.py run $SHARE" > /dev/null; then sleep 120; continue; fi
  if [ -f $R/status.json ] && python3 -c "import json,sys;d=json.load(open('$R/status.json'));sys.exit(0 if d.get('final') and d.get('stop') is None and d.get('roots_open', 1) == 0 else 1)" 2>/dev/null; then
    echo "$(date -u +%FT%TZ) share complete" >> $LOG; exit 0; fi
  q=$(( $(wc -l < /data/sajid/idd_nav/queue/bigbox_cpu.q) - $(wc -l < /data/sajid/idd_nav/queue/bigbox_cpu.started) ))
  c=$(ps -eo pcpu,args | awk '/[\/]data\/sajid\/idd_nav\/envs/ {s+=$1} END {print int(s)}')
  t=99; for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name 2>/dev/null)" = coretemp ] && t=$(( $(cat $h/temp1_input)/1000 )); done
  y=$(cd $HOME/approachb_3d_learning/physics && $HOME/approachb_3d_learning/venv_phys/bin/python yield_probe.py 2>/dev/null)
  if [ "$q" -le 0 ] && [ "$c" -lt 150 ] && [ "$t" -lt 65 ] && [ "${y:-9}" = "0" ]; then
    [ $n -ge $MAX_STARTS ] && { echo "$(date -u +%FT%TZ) MAX_STARTS reached; owner attention needed" >> $LOG; exit 1; }
    n=$((n+1)); echo "$(date -u +%FT%TZ) launch #$n (queue=$q sajid_cpu=$c temp=$t)" >> $LOG
    cd $Q && (setsid nohup env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 $HOME/approachb_3d_learning/venv_q2/bin/python spill2d_v2.py run $SHARE --coexist --cpus 8-23 >> $Q/spillover_blackwell.log 2>&1 < /dev/null &)
    sleep 20; pgrep -f "[q]2bw_coexist.py" > /dev/null || (setsid nohup python3 $Q/q2bw_coexist.py $R > /dev/null 2>&1 < /dev/null &)
  fi
  sleep 120
done
