#!/bin/bash
# REAP's own guard on the lab server (watchdog.sh is left untouched as the user asked; its pkill pattern
# "$Y/venv/bin/python" does not match REAP's venv). Same thresholds as watchdog.sh: if available memory falls below
# MIN_GB or the hottest thermal zone exceeds MAX_C, every REAP python process is killed and a STOP marker written.
# The run scripts are resumable (finished chunks are kept), so a trip loses at most the chunk in flight.
B=${B:-$HOME/yadu_bd/reap}; MIN_GB=${MIN_GB:-16}; MAX_C=${MAX_C:-100}
echo "$(date +%FT%T) reap_guard up (min ${MIN_GB} GB available, max ${MAX_C} C)" >> $B/guard.log
while true; do
  avail=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  # hottest CPU reading: thermal zones (Intel user1 server) or hwmon coretemp / k10temp (AMD gvlab2); 0 if none
  t=$(cat /sys/class/thermal/thermal_zone*/temp 2>/dev/null | sort -n | tail -1)
  [ -z "$t" ] && t=$(for h in /sys/class/hwmon/hwmon*; do case "$(cat $h/name 2>/dev/null)" in coretemp|k10temp|zenpower) cat $h/temp*_input;; esac; done 2>/dev/null | sort -n | tail -1)
  temp=$(( ${t:-0} / 1000 ))
  echo "$(date +%FT%T) ${avail} GB ${temp} C" > $B/guard.last
  if [ "$avail" -lt "$MIN_GB" ] || [ "$temp" -gt "$MAX_C" ]; then
    echo "$(date +%FT%T) STOP: ${avail} GB available, hottest zone ${temp} C" >> $B/guard.log
    touch $B/STOP
    pkill -f "$B/reap/.venv/bin/python"
  fi
  sleep 15
done
