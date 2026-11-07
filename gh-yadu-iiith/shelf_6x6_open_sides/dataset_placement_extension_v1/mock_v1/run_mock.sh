#!/bin/bash
# Bounded mock sequence (frozen run_plan.json). Never touches the source dataset. Kills only its OWN process group (interrupt test).
set -u
D=$HOME/approachb3d_mock/src/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v1
PY=$HOME/approachb3d_mock/env/bin/python
M=$D/mock_v1; R=$M/runs; mkdir -p $R
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl
cd $D
run() { echo "== $(date -u +%FT%TZ) run3d $*"; $PY code/run3d.py "$@" --roster $M/roster.jsonl.gz; echo "== exit $?"; }
run $R/reference --policy reference_v1 --subset reference --workers 12 --cpus 0-11
run $R/serial --policy adaptive_v1 --subset serial --unit query --workers 1 --cpus 0
for n in 8 12 16 18; do run $R/sweep_$n --policy adaptive_v1 --subset all --unit query --workers $n --cpus 0-$((n-1)); done
N=$($PY - <<PYEOF
import json
best = {}
for n in (8, 12, 16, 18):
    led = [json.loads(x) for x in open("$R/sweep_%d/ledger.jsonl" % n)]
    end = [x for x in led if x["event"] == "end"][-1]
    q = sum(x["queries"] for x in led if x["event"] == "unit_done")
    best[n] = q / end["wall_s"] if end["complete"] and not end["safety_stop"] else 0
top = max(best.values())
print(min(n for n, v in best.items() if v >= 0.95 * top))
PYEOF
)
echo "== worker choice (rule: fastest, fewer workers within 5%): $N"
echo "$N" > $R/WORKER_CHOICE
run $R/pilot_A --policy adaptive_v1 --subset all --workers $N
run $R/pilot_B --policy adaptive_v1 --subset all --workers $N
echo "== $(date -u +%FT%TZ) interrupt test"
$PY code/run3d.py $R/interrupt --policy adaptive_v1 --subset all --workers $N --roster $M/roster.jsonl.gz > $R/interrupt_first.log 2>&1 &
while [ ! -f $R/interrupt/ledger.jsonl ] || [ $(grep -c unit_done $R/interrupt/ledger.jsonl) -lt 8 ]; do sleep 0.5; done
PG=$($PY -c "import json;print([json.loads(x) for x in open('$R/interrupt/ledger.jsonl') if '\"start\"' in x][0]['pid'])")
ps -eo pid,pgid,user,args | awk -v g=$PG '$2==g' > $R/interrupt_killed_processes.txt
echo "{\"pgid\": $PG, \"units_done_before_kill\": $(grep -c unit_done $R/interrupt/ledger.jsonl), \"t\": $(date +%s.%N)}" > $R/interrupt_kill.json
kill -9 -- -$PG
sleep 2
echo "== killed pgid $PG: $(wc -l < $R/interrupt_killed_processes.txt) processes; resuming"
run $R/interrupt --policy adaptive_v1 --subset all --workers $N
echo "== $(date -u +%FT%TZ) ALL_DONE"
