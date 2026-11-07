#!/bin/bash
set -u
D=$HOME/approachb3d_mock/src/gh-yadu-iiith/shelf_6x6_open_sides/dataset_placement_extension_v1; PY=$HOME/approachb3d_mock/env/bin/python
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl; cd $D
for x in C1 C2; do echo "== $(date -u +%FT%TZ) pilot_$x fixed_4x1500"; $PY code/run3d.py mock_v1/runs/pilot_$x --policy fixed_4x1500 --subset all --workers 12 --roster mock_v1/roster.jsonl.gz; echo "== exit $?"; done
echo "== CONFIRM_DONE"
