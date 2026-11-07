#!/bin/bash
# V2 interrupt / storage-damage test (run_plan_v2 'interrupt'): adaptive_v1, 12 workers on CPUs 0-11.
#  1. start the run; after >= 10 unit_done ledger lines SIGKILL the run's OWN process group (PGID = coordinator pid, run3d calls setpgrp);
#  2. damage storage: truncate one completed unit's NPZ, plant an orphan JSON and a stray temporary file;
#  3. resume with the identical command: the resume audit must quarantine the damaged unit + orphans and recompute them.
set -u
V2=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-$HOME/approachb3d_mock/env/bin/python}; cd "$V2" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
D=mock_v2/runs/adaptive_v1_interrupt; mkdir -p "$D"
CMD=(nice -n 10 "$PY" code/run3d.py "$D" --roster mock_v2/roster_v2.jsonl.gz --policy adaptive_v1 --workers 12 --cpus 0-11)
echo "$(date -u +%FT%TZ) phase1 start"; "${CMD[@]}" > "$D/console_phase1.log" 2>&1 &
BG=$!
until [ -f "$D/ledger.jsonl" ] && [ "$(grep -c '"unit_done"' "$D/ledger.jsonl")" -ge 10 ]; do sleep 1; done
PG=$(ps -o pgid= -p "$BG" | tr -d ' ')
ps -o pid,pgid,args -g "$PG" > "$D/interrupt_killed_processes.txt"
echo "{\"utc\": \"$(date -u +%FT%TZ)\", \"coordinator\": $BG, \"pgid\": $PG, \"unit_done_lines\": $(grep -c '"unit_done"' "$D/ledger.jsonl"), \"signal\": \"SIGKILL\"}" > "$D/interrupt_kill.json"
[ "$PG" = "$BG" ] && kill -KILL -- "-$PG"; wait "$BG" 2>/dev/null; sleep 2
echo "$(date -u +%FT%TZ) killed pgid $PG; remaining: $(ps -o pid= -g "$PG" | wc -l)"
U=$("$PY" -c "import json,sys; L=[json.loads(x) for x in open('$D/ledger.jsonl') if x.strip()]; print(next(d['unit'] for d in L if d.get('event')=='unit_done' and d.get('npz_sha256')))")
truncate -s 1000 "$D/units/$U.npz"; cp "$D/units/$U.json" "$D/units/orphan_copy_of_$U.json"; echo junk > "$D/units/.stray.999.tmp"
echo "{\"truncated_npz\": \"$U.npz\", \"orphan\": \"orphan_copy_of_$U.json\", \"stray\": \".stray.999.tmp\"}" > "$D/interrupt_damage.json"
echo "$(date -u +%FT%TZ) phase2 resume"; "${CMD[@]}" > "$D/console_phase2.log" 2>&1; echo "$(date -u +%FT%TZ) phase2 rc=$?"
