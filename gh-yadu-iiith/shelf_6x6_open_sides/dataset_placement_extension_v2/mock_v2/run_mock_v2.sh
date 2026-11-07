#!/bin/bash
# V2 mock runs per the frozen mock_v2/configs/run_plan_v2.json.  12 workers on CPUs 0-11 (physical cores) unless the stage says otherwise.
#   bash mock_v2/run_mock_v2.sh STAGE [STAGE...]    stages: reference adaptive_A adaptive_B fixed serial sweep
set -u
V2=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-$HOME/approachb3d_mock/env/bin/python}; cd "$V2" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
R=mock_v2/roster_v2.jsonl.gz
run() { local name=$1 pol=$2; shift 2; echo "$(date -u +%FT%TZ) start $name $pol $*"
        nice -n 10 "$PY" code/run3d.py "mock_v2/runs/$name" --roster $R --policy "$pol" "$@"; echo "$(date -u +%FT%TZ) end $name rc=$?"; }
for s in "$@"; do case $s in
  reference)  run reference_v2 reference_v2 --workers 12 --cpus 0-11;;
  adaptive_A) run adaptive_v1_A adaptive_v1 --workers 12 --cpus 0-11;;
  adaptive_B) run adaptive_v1_B adaptive_v1 --workers 12 --cpus 0-11;;
  fixed)      for p in fixed_4x1500 fixed_4x5000 fixed_8x5000; do run "$p" "$p" --workers 12 --cpus 0-11; done;;
  serial)     run serial_adaptive adaptive_v1 --workers 1 --cpus 0 --unit query --subset mock_v2/configs/serial_qids.txt;;
  sweep)      for n in 8 12 16 18; do run "sweep_$n" adaptive_v1 --workers "$n" --cpus "0-$((n-1))" --unit query --subset mock_v2/configs/sweep_qids.txt; done;;
  *) echo "unknown stage $s"; exit 2;;
esac; done
