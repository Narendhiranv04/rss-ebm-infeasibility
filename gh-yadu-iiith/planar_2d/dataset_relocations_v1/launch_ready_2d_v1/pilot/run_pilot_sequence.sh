#!/usr/bin/env bash
# The pre-registered pilot runs after pilot_b64_w25_r1 (configs/pilot_policy_selection.json "runs"), executed one after another.
# Pilot roots only (25 train roots); never the full launcher.
set -euo pipefail
L="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE="$L/../code"
PY=/home/naren/Desktop/rss3d_env/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
cd "$CODE"
sha256sum ./*.py > "$L/pilot/code_sha_at_sequence_start.txt"
# A run stopped by a safety guard is resumed (same frozen task list) after Tctl cools to <= 75 C, at most 3 times; every stop is logged.
tctl() { for h in /sys/class/hwmon/hwmon*; do [ "$(cat $h/name)" = k10temp ] && echo $(( $(cat $h/temp1_input) / 1000 )); done; }
run() {
  for k in 1 2 3 4; do
    echo "$(date -Is) START $* attempt $k" >> "$L/pilot/sequence.log"; rc=0
    "$PY" launch_runner.py pilot "$L" "$@" > "$L/pilot/$1.$(date +%Y%m%dT%H%M%S).log" 2>&1 || rc=$?
    stop=$(python3 -c "import json;print(json.load(open('$L/pilot/$1/status.json'))['stop'])")
    echo "$(date -Is) END $1 attempt $k rc=$rc stop=$stop" >> "$L/pilot/sequence.log"
    [ "$stop" = "None" ] && return 0
    until [ "$(tctl)" -le 75 ]; do sleep 2; done
  done
}
# pilot_b64_w25_r1 ran separately: 2 TEMPERATURE stops, completed on its second resume (see sequence.log and its run_*.json)
run pilot_b64_w25_r2 64 FULL 25 all
run pilot_b64_serial_first 64 FULL serial first_per_root
run pilot_b8_w25 8 FULL 25 all
# interrupt + resume: SIGKILL the coordinator (workers die with it: PR_SET_PDEATHSIG) once >= 8 roots are packed
I="$L/pilot/pilot_b8_w25_interrupt"
echo "$(date -Is) START interrupt" >> "$L/pilot/sequence.log"
"$PY" launch_runner.py pilot "$L" pilot_b8_w25_interrupt 8 FULL 25 all > "$I.killed.log" 2>&1 &
PID=$!
until [ "$(ls "$I/records" 2>/dev/null | wc -l)" -ge 8 ] || ! kill -0 "$PID" 2>/dev/null; do sleep 0.5; done
kill -9 "$PID"
sleep 3
echo "coordinator $PID killed; surviving workers: $(pgrep -f 'launch_runner.py pilot' | wc -l)" >> "$L/pilot/sequence.log"
(cd "$I" && find records modes witnesses parts -type f ! -name '*.tmp' -print0 | sort -z | xargs -0 sha256sum) > "$L/pilot/interrupt_committed_before_resume.sha256"
ls "$I"/parts/*/*.json 2>/dev/null | wc -l > "$L/pilot/interrupt_open_parts_at_kill.txt"
cp -r "$I/parts" "$L/pilot/interrupt_parts_at_kill"     # content of the open roots' committed parts, compared after resume
run pilot_b8_w25_interrupt 8 FULL 25 all
echo "$(date -Is) SEQUENCE DONE" >> "$L/pilot/sequence.log"
