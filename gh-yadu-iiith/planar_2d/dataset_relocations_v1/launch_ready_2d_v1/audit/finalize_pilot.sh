#!/usr/bin/env bash
# Post-pilot audit chain (reads saved files only; no planner call): audits, cross-run comparisons, nested check, interrupt check,
# gates, the pre-registered decision, figures, integrity, code diff and checksums.
set -euo pipefail
L="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PL="$L/pilot"; A="$L/audit/pilot"
PY=/home/naren/Desktop/rss3d_env/venv/bin/python
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$A"
cd "$L/../code"
for r in pilot_b64_w25_r2 pilot_b64_serial_first pilot_b8_w25 pilot_b8_w25_interrupt; do
  "$PY" launch_check.py audit "$PL/$r" > "$A/audit_$r.json"
done
"$PY" launch_check.py compare "$A/compare_b64_r1_r2_serial.json" "$PL/pilot_b64_w25_r1" "$PL/pilot_b64_w25_r2" "$PL/pilot_b64_serial_first" > /dev/null
"$PY" launch_check.py compare "$A/compare_b64_r1_r2.json" "$PL/pilot_b64_w25_r1" "$PL/pilot_b64_w25_r2" > /dev/null
"$PY" launch_check.py compare "$A/compare_b8_vs_interrupt.json" "$PL/pilot_b8_w25" "$PL/pilot_b8_w25_interrupt" > /dev/null
"$PY" launch_check.py nested "$A/nested_b64r1_vs_b8.json" "$PL/pilot_b64_w25_r1" "$PL/pilot_b8_w25" > /dev/null
"$PY" launch_check.py interrupt "$A/interrupt_check.json" "$PL/pilot_b8_w25_interrupt" "$PL/interrupt_committed_before_resume.sha256" "$PL/interrupt_parts_at_kill" > /dev/null
"$PY" launch_check.py budget "$L" "$PL/pilot_b64_w25_r1" "$A/budget_b64_r1.json" > /dev/null
"$PY" launch_decide.py gates "$L" "$PL/pilot_b64_w25_r1" "$PL/pilot_b64_w25_r2"
"$PY" launch_decide.py decide "$L" "$A/budget_b64_r1.json" "$PL/pilot_b64_w25_r1" "$PL/pilot_b64_w25_r2" "$A/gates.json" > "$A/decision_stdout.json"
"$PY" launch_viz.py all "$L" "$PL/pilot_b64_w25_r1" "$A/budget_b64_r1.json" "$PL/pilot_b64_w25_r2" "$L/audit/figures" > /dev/null
"$PY" launch_check.py integrity "$L" > /dev/null
diff -ruN "$L/audit/baseline/code_before" "$L/../code" > "$L/audit/code_changes.diff" || true
(cd "$L" && find . -type f ! -path './pilot/*/parts/*' ! -name 'checksums.sha256' -print0 | sort -z | xargs -0 sha256sum) > "$L/audit/checksums.sha256"
echo FINALIZE_DONE
