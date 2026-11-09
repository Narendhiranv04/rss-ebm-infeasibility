#!/bin/bash
# Blackwell yield watchdog for the Q2 V2 BLACKWELL run: every 20 s, if a collaborator is active (yadu generator / sajid queue, same rule as the
# physics guard via yield_probe.py) write <run>/PAUSE (the runner SIGSTOPs its workers; user pauses never count as thermal pausing); remove it when
# clear.  Only removes a PAUSE it wrote itself.  Exits when the run's status.json is final.   usage: q2bw_yield.sh RUN_DIR
RUN=$1; P=$HOME/approachb_3d_learning/physics; LOG=$RUN/../yield_watchdog.log
while true; do
  if [ -f "$RUN/status.json" ] && python3 -c "import json,sys;sys.exit(0 if json.load(open('$RUN/status.json')).get('final') else 1)" 2>/dev/null; then echo "$(date -u +%FT%TZ) run final, exit" >> $LOG; rm -f "$RUN/PAUSE.yield"; exit 0; fi
  n=$(cd $P && BWL_REPO=$HOME/approachb_3d_learning/teacher3d/shelf_6x6_open_sides $HOME/approachb_3d_learning/venv_phys/bin/python yield_probe.py 2>/dev/null)
  if [ "${n:-1}" != "0" ]; then
    [ -f "$RUN/PAUSE" ] || { echo "yield-watchdog $(date -u +%FT%TZ) blockers=$n" > "$RUN/PAUSE"; touch "$RUN/PAUSE.yield"; echo "$(date -u +%FT%TZ) PAUSE blockers=${n:-probe_failed}" >> $LOG; }
  elif [ -f "$RUN/PAUSE.yield" ]; then
    rm -f "$RUN/PAUSE" "$RUN/PAUSE.yield"; echo "$(date -u +%FT%TZ) RESUME" >> $LOG
  fi
  sleep 20
done
