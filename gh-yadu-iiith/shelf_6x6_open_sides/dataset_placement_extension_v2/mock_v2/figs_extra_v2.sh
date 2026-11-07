#!/bin/bash
# Scene stills and witness GIFs for the V2 report (MUJOCO_GL=egl; one CPU each, after the main runs).
set -u
V2=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-$HOME/approachb3d_mock/env/bin/python}; cd "$V2" || exit 1
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MUJOCO_GL=egl
F=mock_v2/figures; R=mock_v2/roster_v2.jsonl.gz
taskset -c 0 "$PY" code/figs3d.py scenes mock_v2 3ds66-srv-w0033-s016 $F/scenes_srv-w0033-s016_natural_fixture.png adaptive_v1_A &
taskset -c 1 "$PY" code/figs3d.py scenes mock_v2 3ds66-srv3-w0069-s007 $F/scenes_srv3-w0069-s007_holdout.png adaptive_v1_A &
# GIF 1: the natural fixture, pose rank 1, the verified branch V1 never tried (V1 rejected the pose: branch 0 failed its hold)
W1=$("$PY" -c "
import json,glob
for f in glob.glob('mock_v2/runs/reference_v2/units/*.json'):
    for r in json.load(open(f))['records']:
        if r['qid']=='3ds66-srv-w0033-s016:q0':
            m=next(m for m in r['modes'] if m['rank']==1); print(next(x['witness_id'] for b in m['branches'] for x in b['results'] if x['state']=='VERIFIED'))")
taskset -c 2 "$PY" code/replay3d.py gif mock_v2/runs/reference_v2 $R 3ds66-srv-w0033-s016:q0 $F/witness_natural_fixture_rank1_laterbranch.gif "$W1" > $F/gif1.log 2>&1 &
# GIF 2: a far, high-clutter holdout positive of the chosen-policy run (first in qid order)
Q2=$("$PY" -c "
import json,glob,gzip
ro={json.loads(x)['qid']:json.loads(x) for x in gzip.open('$R','rt')}
c=sorted(r['qid'] for f in glob.glob('mock_v2/runs/adaptive_v1_A/units/*.json') for r in json.load(open(f))['records']
         if r.get('label')==1 and ro[r['qid']]['cohort']=='holdout' and ro[r['qid']]['band']=='far' and ro[r['qid']]['clutter']['between']>=3)
print(c[0])")
taskset -c 3 "$PY" code/replay3d.py gif mock_v2/runs/adaptive_v1_A $R "$Q2" $F/witness_holdout_far_clutter_${Q2//:/_}.gif > $F/gif2.log 2>&1 &
wait; ls -la $F; tail -1 $F/gif1.log | cut -c1-300; tail -1 $F/gif2.log | cut -c1-300
